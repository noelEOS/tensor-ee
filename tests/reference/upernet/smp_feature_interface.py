"""Phases 1-3, 6: the installed SMP Swin-T + UPerNet and its exact encoder -> decoder contract.

The authoritative model is segmentation_models_pytorch's own UPerNet, built here exactly
as the project would build it:

    smp.UPerNet("tu-swin_tiny_patch4_window7_224", encoder_weights=None, encoder_depth=5,
                decoder_channels=256, decoder_use_norm="batchnorm", in_channels=2,
                classes=2, upsampling=4, img_size=(H, W))

in_channels=2 is Sentinel-1 VV/VH (Milestone 6); classes=2 matches the project's SegWater
model (tensor_ee/segwater_pytorch.py, two logits); the segmentation head is out of scope
for Milestone 7 and the choice does not affect C1-C4.  img_size is passed through SMP to
timm because Swin's attention masks depend on the raster (timm's strict_img_size).

The encoder is SMP's TimmUniversalEncoder around timm's FeatureListNet (modules
patch_embed, layers_0 .. layers_3; timm's final `norm` is not part of it).  It receives
Milestones 5-6's parameters (encoder: seeded M5 set, patch embedding: seeded M6 set),
remapped from SwinTransformer names (layers.i.) to FeatureListNet names (layers_i.).
The decoder gets deterministic seeded parameters (no trained checkpoint exists): conv
weights N(0, 1/fan_in), BatchNorm gamma 1 + N(0, 0.2^2), beta N(0, 0.2^2), running mean
N(0, 0.2^2) and running variance 1 + N(0, 0.1^2) (non-trivial, so inference BatchNorm is
actually exercised; SMP's own initialization leaves running stats at 0 / 1), LayerNorm2d
gamma 1 + N(0, 0.2^2), beta N(0, 0.2^2).

decoder_interface() runs the real forward pass and captures, with hooks, the exact list
UPerNet's decoder receives (C1..C4 raw) and the output of each of its feature norms
(C1..C4 prepared); it also re-runs the preparation by hand and requires bit equality.

Moved from ``upernet/reference/smp_feature_interface.py`` at L06b (test support). Its command-line writer of the
research data files was archived with the research code.
"""

from __future__ import annotations

import inspect
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from reference.swin_block.pytorch_swin_block import from_window_layout
from reference.swin_encoder import pytorch_swin_encoder as m5
from reference.swin_hierarchy.pytorch_hierarchy import generate_parameters
from reference.swin_image_encoder import pytorch_image_encoder as m6

ENCODER_NAME = "tu-swin_tiny_patch4_window7_224"
ENCODER_DEPTH = 5
IN_CHANNELS = m6.IN_CHANS
CLASSES = 2
DECODER_CHANNELS = 256
UPSAMPLING = 4
FEATURES = ("C1", "C2", "C3", "C4")
FEATURE_CHANNELS = m5.STAGE_DIMS                      # (96, 192, 384, 768): verified below
FEATURE_STRIDES = (4, 8, 16, 32)                      # input pixels per feature pixel
DECODER_SEED = 70                                     # decoder tensor i: seed 70 * 1000 + 100 + i


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def make_upernet_module(height: int, width: int):
    """The bare SMP model for an input of height x width pixels (random parameters)."""
    import segmentation_models_pytorch as smp
    return smp.UPerNet(ENCODER_NAME, encoder_depth=ENCODER_DEPTH, encoder_weights=None,
                       decoder_channels=DECODER_CHANNELS, decoder_use_norm="batchnorm",
                       in_channels=IN_CHANNELS, classes=CLASSES, upsampling=UPSAMPLING,
                       img_size=(height, width))


def encoder_state(parameters: dict, pe: dict) -> dict[str, torch.Tensor]:
    """Milestone 5-6 parameters under FeatureListNet names (layers.i. -> layers_i.).
    timm's final norm has no FeatureListNet counterpart and is left out."""
    state = {}
    for key, value in m5.timm_state(parameters).items():
        if key.startswith("norm."):
            continue
        stage, rest = key[len("layers."):].split(".", 1)
        state[f"layers_{stage}.{rest}"] = torch.from_numpy(np.asarray(value))
    state.update({f"patch_embed.{k}": v for k, v in m6.timm_pe_state(pe).items()})
    return state


_DECODER: dict = {}


def decoder_template() -> dict[str, torch.Tensor]:
    """Decoder + head parameters and BatchNorm statistics of the SMP model (shapes only)."""
    model = make_upernet_module(224, 224)
    return {k: v for k, v in model.state_dict().items()
            if not k.startswith("encoder.") and not k.endswith("num_batches_tracked")}


def decoder_spec(template: dict) -> dict[str, tuple[float, float]]:
    """(std, offset) per decoder tensor: see the module docstring."""
    spec = {}
    for name, value in template.items():
        if name.endswith(".running_var"):
            spec[name] = (0.1, 1.0)
        elif name.endswith(".running_mean"):
            spec[name] = (0.2, 0.0)
        elif value.ndim == 4:                                   # conv weight
            spec[name] = (float(1 / np.sqrt(np.prod(value.shape[1:]))), 0.0)
        elif name.endswith(".weight"):                          # norm gamma
            spec[name] = (0.2, 1.0)
        elif name.startswith("segmentation_head") and name.endswith(".bias"):
            spec[name] = (0.1, 0.0)
        else:                                                   # norm beta
            spec[name] = (0.2, 0.0)
    return spec


def decoder_parameters() -> dict[str, np.ndarray]:
    """{SMP state_dict name: float32} for the decoder and head (memoized, deterministic)."""
    if not _DECODER:
        template = decoder_template()
        values = generate_parameters(template, decoder_spec(template), DECODER_SEED)
        bad = [k for k, v in values.items() if k.endswith("running_var") and v.min() <= 0.1]
        if bad:
            raise AssertionError(f"non-positive running variance in {bad}")
        _DECODER.update(values)
    return _DECODER


def configure_encoder(model, height: int, width: int):
    """Pin timm's explicit attention path and check every block's shift and window
    against Milestone 5's expectations for this raster (raises otherwise)."""
    features = model.encoder.model
    shifts = m5.block_shifts(height // 28, width // 28)
    for s in range(4):
        stage = getattr(features, f"layers_{s}")
        for block, shift in zip(stage.blocks, shifts[s]):
            m5._configure(block, shift)


def make_upernet(height: int, width: int, parameters: dict | None = None, pe: dict | None = None,
                 decoder: dict | None = None, dtype=torch.float32):
    """The SMP model with Milestone 5-6 encoder parameters and the seeded decoder,
    in eval mode and `dtype` (attention masks rebuilt in that dtype)."""
    parameters = m5.encoder_parameters() if parameters is None else parameters
    pe = m6.pe_parameters() if pe is None else pe
    decoder = decoder_parameters() if decoder is None else decoder
    model = make_upernet_module(height, width)
    configure_encoder(model, height, width)
    features = model.encoder.model
    state = encoder_state(parameters, pe)
    missing, unexpected = features.load_state_dict(state, strict=False)
    if unexpected or any(not k.endswith(("relative_position_index", "attn_mask")) for k in missing):
        raise ValueError(f"encoder state mismatch: missing {missing[:4]}, unexpected {unexpected[:4]}")
    full = model.state_dict()
    for name, value in decoder.items():
        if full[name].shape != tuple(value.shape):
            raise ValueError(f"{name}: shape {tuple(value.shape)} != {tuple(full[name].shape)}")
    missing, unexpected = model.load_state_dict(
        {k: torch.from_numpy(np.asarray(v)) for k, v in decoder.items()}, strict=False)
    if unexpected or any(not k.startswith("encoder.") and not k.endswith("num_batches_tracked")
                         for k in missing):
        raise ValueError(f"decoder state mismatch: missing {missing[:4]}, unexpected {unexpected[:4]}")
    model = model.to(dtype).eval()
    for s in range(4):
        m5._refresh_masks(getattr(features, f"layers_{s}").blocks, dtype)
    return model


# ---------------------------------------------------------------------------
# The decoder boundary
# ---------------------------------------------------------------------------
@contextmanager
def recorded_interpolations():
    """Every torch.nn.functional.interpolate call made inside the block (decoder modules
    and nn.Upsample* call it through the module attribute)."""
    import torch.nn.functional as F
    calls = []
    original = F.interpolate

    def wrapper(input, size=None, scale_factor=None, mode="nearest", align_corners=None,
                recompute_scale_factor=None, antialias=False):
        out = original(input, size, scale_factor, mode, align_corners, recompute_scale_factor,
                       antialias)
        caller = inspect.stack()[1]
        calls.append({"caller": f"{Path(caller.filename).name}:{caller.lineno} ({caller.function})",
                      "source": list(input.shape[2:]), "target": list(out.shape[2:]),
                      "size": None if size is None else [int(v) for v in np.atleast_1d(size)],
                      "scale_factor": scale_factor, "mode": mode, "align_corners": align_corners,
                      "recompute_scale_factor": recompute_scale_factor, "antialias": antialias})
        return out

    F.interpolate = wrapper
    try:
        yield calls
    finally:
        F.interpolate = original


def decoder_interface(model, x: torch.Tensor, full_forward: bool = True) -> dict:
    """The tensors UPerNet's decoder receives and prepares, captured in the real forward.

    Returns {"raw": [C1..C4 NCHW], "prepared": [C1..C4 NCHW], "encoder_list": the
    encoder's full output list (input, dummy 1/2-scale, C1..C4), "output": logits,
    "interpolations": the recorded F.interpolate calls}.  Raises unless the prepared
    tensors equal a manual application of decoder.feature_norms to the raw ones, bit
    for bit, and the raw ones equal a direct call of the encoder.
    """
    captured: dict = {}

    def pre_hook(module, args):
        captured["list"] = [f.detach().clone() for f in args[0]]

    norms = model.decoder.feature_norms
    prepared = [None] * len(norms)
    hooks = [model.decoder.register_forward_pre_hook(pre_hook)]
    for i, norm in enumerate(norms):
        hooks.append(norm.register_forward_hook(
            lambda module, args, out, i=i: prepared.__setitem__(i, out.detach().clone())))
    try:
        with torch.no_grad(), recorded_interpolations() as calls:
            output = model(x) if full_forward else model.decoder(model.encoder(x))
    finally:
        for hook in hooks:
            hook.remove()
    raw = captured["list"][2:]
    with torch.no_grad():
        direct = model.encoder(x)
        manual = [norm(f) for norm, f in zip(norms, direct[2:])]
    for k, (a, b) in enumerate(zip(raw, direct[2:])):
        if not torch.equal(a, b):
            raise AssertionError(f"C{k + 1} at the decoder differs from the encoder's output")
    for k, (a, b) in enumerate(zip(prepared, manual)):
        if not torch.equal(a, b):
            raise AssertionError(f"C{k + 1} prepared differs from a manual feature_norms call")
    return {"raw": raw, "prepared": prepared, "encoder_list": captured["list"],
            "output": output, "interpolations": calls}


def nhwc(t: torch.Tensor) -> np.ndarray:
    """[1, C, H, W] -> [H, W, C] (values moved)."""
    return t[0].permute(1, 2, 0).numpy()


# ---------------------------------------------------------------------------
# Phase 1: the contract, determined programmatically
# ---------------------------------------------------------------------------
def module_path(obj) -> str:
    return str(Path(inspect.getsourcefile(type(obj) if not inspect.isclass(obj) else obj)))


def contract(height: int = 224, width: int = 224, seed: int = 0) -> dict:
    """Everything Phase 1 asks, established by running the installed code.

    On a random input of height x width: which module produces each C_k, whether it
    equals the Swin stage output (Milestone 5's explicit encoder, s{k}_out) and not the
    final norm, layouts, dtypes, the decoder's normalization and interpolation calls."""
    import segmentation_models_pytorch as smp
    import timm
    from segmentation_models_pytorch.decoders.upernet import decoder as upernet_decoder
    from segmentation_models_pytorch.encoders import timm_universal

    model = make_upernet(height, width)
    features = model.encoder.model
    x = torch.from_numpy(np.random.default_rng(seed).standard_normal(
        (1, IN_CHANNELS, height, width)).astype(np.float32))
    result = decoder_interface(model, x)

    # The stage outputs of Milestone 5-6's explicit encoder on the same input.
    z = x[0].permute(1, 2, 0).numpy()
    ref = m6.image_encoder_reference(m5.encoder_parameters(), m6.pe_parameters(), z,
                                     names=["s1_out", "s2_out", "s3_out", "s4_out", "norm"])
    sources = {}
    for k in range(4):
        c = result["raw"][k]
        h, w = c.shape[2:]
        stage_out = from_window_layout(ref[f"s{k + 1}_out"], h, w)
        sources[FEATURES[k]] = {
            "feature_info": features.feature_info.get_dicts()[k],
            "shape_nchw": list(c.shape), "dtype": str(c.dtype),
            "is_contiguous_nchw": bool(c.is_contiguous()),
            "equals_stage_output_bitwise": bool(np.array_equal(nhwc(c), stage_out)),
        }
    final_norm = from_window_layout(ref["norm"], height // 32, width // 32)
    c4 = nhwc(result["raw"][3])
    sources["C4"]["equals_final_encoder_norm"] = bool(np.array_equal(c4, final_norm))
    sources["C4"]["max_abs_difference_to_final_norm"] = float(np.abs(c4 - final_norm).max())

    decoder = model.decoder
    norm_types = sorted({type(m).__name__ for m in decoder.modules()
                         if isinstance(m, (torch.nn.LayerNorm, torch.nn.BatchNorm2d,
                                           torch.nn.GroupNorm, torch.nn.InstanceNorm2d))})
    conv_blocks = {name: {"kernel": list(m[0].kernel_size), "padding": list(m[0].padding),
                          "stride": list(m[0].stride), "dilation": list(m[0].dilation),
                          "groups": m[0].groups, "in": m[0].in_channels, "out": m[0].out_channels,
                          "bias": m[0].bias is not None, "norm": type(m[1]).__name__,
                          "norm_eps": getattr(m[1], "eps", None),
                          "activation": type(m[2]).__name__,
                          "inplace": getattr(m[2], "inplace", None)}
                   for name, m in decoder.named_modules()
                   if type(m).__name__ == "Conv2dReLU"}
    head = model.segmentation_head
    return {
        "versions": {"segmentation_models_pytorch": smp.__version__, "timm": timm.__version__,
                     "torch": torch.__version__},
        "sources": {"upernet_decoder": module_path(upernet_decoder.UPerNetDecoder),
                    "upernet_model": module_path(type(model)),
                    "encoder_wrapper": module_path(timm_universal.TimmUniversalEncoder),
                    "timm_feature_net": module_path(type(features)),
                    "conv2drelu": module_path(type(decoder.fusion_block))},
        "model": {"name": model.name, "encoder_name": ENCODER_NAME, "in_channels": IN_CHANNELS,
                  "classes": CLASSES, "encoder_depth": ENCODER_DEPTH,
                  "encoder_out_channels": list(model.encoder.out_channels),
                  "encoder_is_transformer_style": model.encoder._is_transformer_style,
                  "encoder_output_fmt_nhwc": model.encoder._is_channel_last,
                  "encoder_output_stride": model.encoder.output_stride,
                  "feature_net_modules": [n for n, _ in features.named_children()],
                  "feature_net_has_final_norm": hasattr(features, "norm"),
                  "decoder_channels": DECODER_CHANNELS,
                  "decoder_normalization_classes": norm_types,
                  "feature_norms": [{"class": type(n).__name__, "normalized_shape": list(n.normalized_shape),
                                     "eps": n.eps, "elementwise_affine": n.elementwise_affine}
                                    for n in decoder.feature_norms],
                  "psp_sizes": [b[0].output_size for b in decoder.psp.blocks],
                  "conv_blocks": conv_blocks,
                  "segmentation_head": {"conv": {"in": head[0].in_channels, "out": head[0].out_channels,
                                                 "kernel": list(head[0].kernel_size),
                                                 "bias": head[0].bias is not None},
                                        "upsampling": repr(head[1]),
                                        "upsampling_align_corners": getattr(head[1], "align_corners", None),
                                        "activation": repr(head[2])},
                  "classification_head": model.classification_head is not None,
                  "requires_divisible_input_shape": model.requires_divisible_input_shape},
        "features": sources,
        "encoder_list": [list(t.shape) for t in result["encoder_list"]],
        "interpolations": result["interpolations"],
        "output_shape": list(result["output"].shape),
        "decoder_receives": "features[2:] of the encoder list (C1..C4); the input (index 0) and the "
                            "empty 1/2-scale placeholder (index 1) are skipped",
    }


# ---------------------------------------------------------------------------
# Phase 6: parameters by subsystem
# ---------------------------------------------------------------------------
SUBSYSTEMS = (
    ("feature_norms", "decoder.feature_norms."),
    ("psp_pooled_1x1", "decoder.psp.blocks."),
    ("psp_out_conv_3x3", "decoder.psp.out_conv."),
    ("fpn_lateral_1x1", "decoder.fpn_lateral_blocks."),
    ("fpn_conv_3x3", "decoder.fpn_conv_blocks."),
    ("fusion_3x3", "decoder.fusion_block."),
    ("segmentation_head", "segmentation_head."),
)


def subsystem_of(name: str) -> str:
    for label, prefix in SUBSYSTEMS:
        if name.startswith(prefix):
            return label
    raise KeyError(name)


def kind_of(name: str, value) -> str:
    if np.ndim(value) == 4:
        return "conv_weight"
    if name.startswith("decoder.feature_norms."):
        return "layernorm_" + name.rsplit(".", 1)[1]
    if name.startswith("segmentation_head"):
        return "conv_bias"
    return "batchnorm_" + name.rsplit(".", 1)[1]


def parameter_counts(decoder: dict | None = None) -> dict:
    """{subsystem: {kind: count}} plus totals (trainable parameters and BN buffers)."""
    decoder = decoder_parameters() if decoder is None else decoder
    out: dict = {}
    for name, value in decoder.items():
        sub = out.setdefault(subsystem_of(name), {})
        kind = kind_of(name, value)
        sub[kind] = sub.get(kind, 0) + int(np.size(value))
    for sub in out.values():
        sub["total"] = sum(v for k, v in sub.items())
    out["all"] = {"total": sum(v["total"] for v in out.values()),
                  "conv_weights": sum(v.get("conv_weight", 0) for v in out.values()),
                  "batchnorm_running_stats": sum(v.get("batchnorm_running_mean", 0)
                                                 + v.get("batchnorm_running_var", 0)
                                                 for v in out.values())}
    return out
