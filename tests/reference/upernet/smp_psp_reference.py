"""Phases 1, 9, 13 (Tests 1, 8): the installed SMP UPerNet's Pyramid Pooling Module, read off
the installed code and run operation by operation.

The authoritative module is segmentation_models_pytorch 0.5.0's
decoders/upernet/decoder.py `PSPModule` inside `UPerNetDecoder` (Milestone 7's model,
smp_feature_interface.make_upernet).  Its forward, as installed:

    pyramid = [feature]                                   # LayerNorm2d(768)(C4), channels-last
    for block in blocks:                                  # sizes (1, 2, 3, 6), in this order
        pooled  = block(feature)                          # AdaptiveAvgPool2d(s) -> Conv2dReLU 1x1
        resized = F.interpolate(pooled, size=(H, W), mode="bilinear", align_corners=False)
        pyramid.append(resized)
    out = out_conv(torch.cat(pyramid, dim=1))             # Conv2dReLU 3x3, 1792 -> 256, padding 1

with Conv2dReLU = Conv2d(bias=False) -> BatchNorm2d(eps 1e-5) -> ReLU(inplace=True).
`contract` establishes every item programmatically (source path, classes, sizes, layer
configurations, interpolation calls, the memory format the pool receives, the
concatenation order with unique channel IDs, and that a real UPerNet forward feeds the
PSP exactly feature_norms[3](C4)).  `expanded` runs the module's own submodules one at a
time, exposing every intermediate, and raises unless the result equals the module's
forward bit for bit.

Tensors are NHWC NumPy [H, W, C] outside this module (pooled maps [s, s, C]).
"""

from __future__ import annotations

import copy
import inspect

import numpy as np
import torch
import torch.nn.functional as F

from reference.upernet import adaptive_pool_reference as AP
from reference.upernet import smp_feature_interface as S

SIZES = AP.SIZES
C4_CHANNELS = 768
DECODER = 256
BRANCH_STEPS = ("conv", "bn", "relu", "up")
TENSORS = (["C4_raw", "C4_norm"] + [f"pool_{s}" for s in SIZES]
           + [f"branch_{s}_{step}" for s in SIZES for step in BRANCH_STEPS]
           + ["psp_concat", "psp_conv", "psp_bn", "psp_relu"])
OUTPUT = "psp_relu"                                     # PSP_output


def prefix(i: int) -> str:
    """SMP state_dict prefix of PSP block i (conv at .0, BatchNorm at .1)."""
    return f"decoder.psp.blocks.{i}.1"


OUT_PREFIX = "decoder.psp.out_conv"
NORM_PREFIX = "decoder.feature_norms.3"

_MODELS: dict = {}


def model(dtype=torch.float32):
    """The installed SMP UPerNet with Milestone 5-6 encoder and the seeded decoder
    parameters (the decoder does not depend on the input size)."""
    if dtype not in _MODELS:
        _MODELS[dtype] = S.make_upernet(224, 224, dtype=dtype)
    return _MODELS[dtype]


def nchw(x: np.ndarray, dtype=torch.float32) -> torch.Tensor:
    """[H, W, C] -> NCHW-contiguous [1, C, H, W] (the encoder's output format)."""
    return torch.from_numpy(np.ascontiguousarray(x)).to(dtype).permute(2, 0, 1)[None].contiguous()


def nhwc(t: torch.Tensor) -> np.ndarray:
    return t.detach()[0].permute(1, 2, 0).contiguous().numpy()


# ---------------------------------------------------------------------------
# The expanded forward (Phase 1)
# ---------------------------------------------------------------------------
def expanded(decoder, c4: torch.Tensor) -> dict[str, torch.Tensor]:
    """Every PSP intermediate from the installed submodules, NCHW.  c4 is the raw C4 as the
    encoder delivers it (NCHW-contiguous); the PSP input is feature_norms[3](c4), whose
    channels-last strides select PyTorch's channels-last pooling kernel, exactly as in the
    real forward.  Raises unless every branch and the output equal the modules' own
    forward bit for bit."""
    psp = decoder.psp
    out = {}
    with torch.no_grad():
        n4 = decoder.feature_norms[3](c4)
        out["C4_raw"], out["C4_norm"] = c4, n4
        pyramid = [n4]
        for s, block in zip(SIZES, psp.blocks):
            pool = block[0](n4)
            conv = block[1][0](pool)
            bn = block[1][1](conv)
            relu = torch.relu(bn)                         # SMP's ReLU is in place: not reused
            if not torch.equal(relu, block(n4)):
                raise AssertionError(f"expanded PSP block {s} differs from SMP's module")
            up = F.interpolate(relu, size=n4.shape[2:], mode="bilinear", align_corners=False)
            out.update({f"pool_{s}": pool, f"branch_{s}_conv": conv, f"branch_{s}_bn": bn,
                        f"branch_{s}_relu": relu, f"branch_{s}_up": up})
            pyramid.append(up)
        concat = torch.cat(pyramid, dim=1)
        conv = psp.out_conv[0](concat)
        bn = psp.out_conv[1](conv)
        relu = torch.relu(bn)
        if not torch.equal(relu, psp(n4)):
            raise AssertionError("expanded PSP output differs from SMP's PSPModule.forward")
        out.update(psp_concat=concat, psp_conv=conv, psp_bn=bn, psp_relu=relu)
    return out


def psp_tensors(c4_raw: np.ndarray, dtype=torch.float32) -> dict[str, np.ndarray]:
    """{name: NHWC array} of the installed PSP on a raw C4 [H, W, 768]."""
    return {k: nhwc(v) for k, v in expanded(model(dtype).decoder, nchw(c4_raw, dtype)).items()}


# ---------------------------------------------------------------------------
# The contract (Phase 1, Test 1; Phase 9, Test 8)
# ---------------------------------------------------------------------------
def conv_block_config(block) -> dict:
    conv, norm, act = block[0], block[1], block[2]
    return {"conv": {"class": type(conv).__name__, "in": conv.in_channels, "out": conv.out_channels,
                     "kernel": list(conv.kernel_size), "padding": list(conv.padding),
                     "padding_mode": conv.padding_mode, "stride": list(conv.stride),
                     "dilation": list(conv.dilation), "groups": conv.groups,
                     "bias": conv.bias is not None},
            "norm": {"class": type(norm).__name__, "eps": norm.eps, "momentum": norm.momentum,
                     "affine": norm.affine, "track_running_stats": norm.track_running_stats},
            "activation": {"class": type(act).__name__, "inplace": getattr(act, "inplace", None)}}


def concat_order(decoder) -> dict:
    """Unique channel IDs through a copy of the PSP: the feature carries c + 1 in channel c
    (constant planes), branch k's Conv2dReLU outputs 1000 (k + 1) + j in channel j (zero
    conv weights, BatchNorm mean 0 and bias = ID: exact), and the input of out_conv is
    captured.  Channel sources are read back by nearest integer (the bilinear resize of a
    constant map is exact up to float32 rounding of the weights)."""
    psp = copy.deepcopy(decoder.psp).float().eval()
    h = w = 28
    with torch.no_grad():
        for k, block in enumerate(psp.blocks):
            block[1][0].weight.zero_()
            block[1][1].running_mean.zero_()
            block[1][1].bias.copy_(1000.0 * (k + 1) + torch.arange(DECODER, dtype=torch.float32))
        feature = (torch.arange(C4_CHANNELS, dtype=torch.float32) + 1)[None, :, None, None]
        feature = feature.expand(1, C4_CHANNELS, h, w).contiguous(memory_format=torch.channels_last)
        captured = {}
        hook = psp.out_conv.register_forward_pre_hook(
            lambda m, args: captured.__setitem__("cat", args[0].detach().clone()))
        psp(feature)
        hook.remove()
    cat = captured["cat"]                                             # [1, 1792, h, w]
    ids = torch.round(cat).to(torch.int64)
    if not torch.equal(ids, ids[:, :, :1, :1].expand_as(ids)):
        raise AssertionError("channel IDs are not constant over the map")
    channel_ids = ids[0, :, 0, 0].tolist()
    expected = [c + 1 for c in range(C4_CHANNELS)] + [1000 * (k + 1) + j for k in range(len(SIZES))
                                                      for j in range(DECODER)]
    segments = []
    start = 0
    for name, width in [("C4_norm (feature)", C4_CHANNELS)] + [
            (f"branch {s} (AdaptiveAvgPool2d({s}))", DECODER) for s in SIZES]:
        segments.append({"source": name, "channels": [start, start + width]})
        start += width
    return {"total_channels": len(channel_ids), "segments": segments,
            "order_matches_ids": channel_ids == expected,
            "feature_part_exact": bool(torch.equal(cat[:, :C4_CHANNELS], feature)),
            "max_branch_deviation_from_integer_id": float((cat[:, C4_CHANNELS:]
                                                           - ids[:, C4_CHANNELS:]).abs().max())}


def real_forward_check(seed: int = 0) -> dict:
    """In a real UPerNet forward (random 224 x 224 input), the PSP receives
    feature_norms[3](C4) with channels-last strides and returns the expanded output."""
    m = model(torch.float32)
    captured = {}
    hooks = [m.decoder.psp.register_forward_pre_hook(
                 lambda mod, args: captured.__setitem__("input", args[0].detach().clone())),
             m.decoder.psp.register_forward_hook(
                 lambda mod, args, out: captured.__setitem__("output", out.detach().clone()))]
    pre = m.decoder.register_forward_pre_hook(
        lambda mod, args: captured.__setitem__("features", [f.detach().clone() for f in args[0]]))
    x = torch.from_numpy(np.random.default_rng(seed).standard_normal(
        (1, S.IN_CHANNELS, 224, 224)).astype(np.float32))
    try:
        with torch.no_grad():
            m(x)
    finally:
        for hook in hooks + [pre]:
            hook.remove()
    c4 = captured["features"][-1]
    with torch.no_grad():
        n4 = m.decoder.feature_norms[3](c4)
    tensors = expanded(m.decoder, c4)
    stride_input = captured["input"].stride()
    return {"psp_input_equals_feature_norm3_of_C4": bool(torch.equal(captured["input"], n4)),
            "psp_input_strides": list(n4.stride()),
            "psp_input_is_channels_last": bool(n4.is_contiguous(memory_format=torch.channels_last)
                                               and not n4.is_contiguous()),
            "captured_input_strides": list(stride_input),
            "psp_output_equals_expanded": bool(torch.equal(captured["output"], tensors[OUTPUT])),
            "psp_output_shape": list(captured["output"].shape)}


def contract() -> dict:
    import segmentation_models_pytorch as smp
    from segmentation_models_pytorch.decoders.upernet import decoder as D
    m = model(torch.float32)
    psp = m.decoder.psp
    source, start = inspect.getsourcelines(D.PSPModule)
    forward, fstart = inspect.getsourcelines(D.PSPModule.forward)
    n4 = torch.zeros(1, C4_CHANNELS, 28, 28).contiguous(memory_format=torch.channels_last)
    with S.recorded_interpolations() as calls, torch.no_grad():
        psp(n4)
    return {
        "versions": {"segmentation_models_pytorch": smp.__version__, "torch": torch.__version__},
        "source": {"file": inspect.getsourcefile(D.PSPModule), "class": "PSPModule",
                   "class_lines": [start, start + len(source) - 1],
                   "forward_lines": [fstart, fstart + len(forward) - 1],
                   "forward": "".join(forward)},
        "used_by": "UPerNetDecoder.__init__: PSPModule(in_channels=encoder_channels[-1] (768), "
                   "out_channels=decoder_channels (256), sizes=(1, 2, 3, 6), use_norm); "
                   "forward: psp_out = self.psp(features[-1]) after feature_norms",
        "classes": sorted({type(mod).__name__ for mod in psp.modules()}),
        "pool_sizes": [b[0].output_size for b in psp.blocks],
        "pool_class": [type(b[0]).__name__ for b in psp.blocks],
        "branch_blocks": [conv_block_config(b[1]) for b in psp.blocks],
        "out_conv": conv_block_config(psp.out_conv),
        "interpolations": calls,
        "concatenation": concat_order(m.decoder),
        "real_forward": real_forward_check(),
        "parameters": {name: list(v.shape) for name, v in S.decoder_parameters().items()
                       if name.startswith(("decoder.psp.", NORM_PREFIX))},
    }


# ---------------------------------------------------------------------------
# Inputs (Phases 1, 11, 13, 17, 21)
# ---------------------------------------------------------------------------


def synthetic_c4(h: int, w: int, seed: int = 81, scale: float = 3.0) -> np.ndarray:
    """Deterministic C4 fixture: the project's hash values (multiples of 1/2048 in [-1, 1))
    times `scale`, generated identically in EE (sa.window_layout_raster)."""
    from reference import synthetic
    return (synthetic.generated_raster_numpy(h, w, C4_CHANNELS, seed) * np.float32(scale)).astype(np.float32)
