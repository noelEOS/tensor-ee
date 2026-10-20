"""SMP reference contracts for the user-approved full-model validation plan.

These functions execute only local CPU PyTorch when the maintainer invokes them. They
never initialize EE, evaluate native graphs, download weights or create tasks.
``full_reference`` captures the actual full SMP forward, then requires independent
installed-submodule traversal to match every boundary and logit float32 bit.
Isolated references consume actual incoming HWC arrays, without replacing state.
The historical sizing helper is deliberately unchanged.
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
import sys

import numpy as np

# Match the existing fixture harness's explicit checkout-local reference path.
_REFERENCES = str(Path(__file__).resolve().parents[2] / "tests")
if _REFERENCES not in sys.path:
    sys.path.insert(0, _REFERENCES)

from reference.segformer_b0.full_model import (
    DIMS, HEADS, RATIOS, SHAPES, validate_cut_inputs, validate_hwc, validate_state,
)


def _hwc(tensor):
    return tensor.detach().cpu()[0].permute(1, 2, 0).contiguous().numpy().copy()


def _nchw(array):
    import torch

    return torch.from_numpy(np.array(array, copy=True, order="C")).permute(2, 0, 1)[None].contiguous()


def _assert_reference_contract(model):
    import torch

    enc = model.encoder
    for i, dim in enumerate(DIMS):
        stage = i + 1
        embedding = getattr(enc, f"patch_embed{stage}")
        blocks = getattr(enc, f"block{stage}")
        norm = getattr(enc, f"norm{stage}")
        kernel, stride = ((7, 4) if i == 0 else (3, 2))
        if (embedding.proj.kernel_size != (kernel, kernel) or embedding.proj.stride != (stride, stride)
                or embedding.proj.padding != (kernel // 2, kernel // 2) or embedding.norm.eps != 1e-5
                or len(blocks) != 2 or norm.eps != 1e-6):
            raise ValueError(f"installed stage{stage} differs from the inspected reference")
        for b in blocks:
            if (b.norm1.eps != 1e-6 or b.norm2.eps != 1e-6 or b.attn.num_heads != HEADS[i]
                    or b.attn.sr_ratio != RATIOS[i] or b.attn.scale != (dim // HEADS[i]) ** -0.5
                    or not isinstance(b.mlp.act, torch.nn.GELU) or b.mlp.act.approximate != "none"
                    or (RATIOS[i] > 1 and b.attn.norm.eps != 1e-5)):
                raise ValueError(f"installed block in stage{stage} differs from the inspected reference")
    if (model.decoder.fuse_stage[0].bias is not None or model.decoder.fuse_stage[1].eps != 1e-5
            or model.segmentation_head[1].align_corners is not True
            or model.segmentation_head[1].scale_factor != 4.0
            or model.classification_head is not None):
        raise ValueError("installed decoder/head differs from the inspected reference")
    if any(module.training for module in model.modules()):
        raise ValueError("reference must be entirely in evaluation mode")


def make_reference(state):
    """Exact SMP0.5.0 mit_b0 eval model loaded strictly with unmodified state."""
    validate_state(state)
    if importlib.metadata.version("segmentation-models-pytorch") != "0.5.0":
        raise ValueError("validation requires the inspected SMP0.5.0 reference")
    import torch
    import segmentation_models_pytorch as smp

    torch.set_num_threads(2)
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            # Initialization is discarded through complete strict state loading;
            # the caller's state and RNG stream are never altered.
            torch.manual_seed(20261014)
            model = smp.Segformer("mit_b0", encoder_weights=None, in_channels=2, classes=2)
        tensors = {name: torch.from_numpy(np.array(value, copy=True, order="C")) for name, value in state.items()}
        model.load_state_dict(tensors, strict=True)
        model.eval()
    finally:
        torch.set_default_dtype(previous_dtype)
    _assert_reference_contract(model)
    return model


def _expanded_decoder(model, features):
    """Installed projection/fusion/head modules with explicit resize/cat order."""
    import torch
    import torch.nn.functional as F

    parts = []
    for i, name in enumerate(("C4", "C3", "C2", "C1")):
        projected = model.decoder.mlp_stage[i](features[name])
        parts.append(F.interpolate(projected, size=(56, 56), mode="bilinear", align_corners=False))
    x = torch.cat(parts, dim=1)
    for module in model.decoder.fuse_stage:
        x = module(x)
    for module in model.segmentation_head:
        x = module(x)
    return x


def expanded_reference(model, input):
    """Independent traversal used to audit actual SMP full-forward boundaries."""
    import torch

    x = _nchw(validate_hwc(input, "input"))
    outputs = {}
    with torch.no_grad():
        for stage in range(1, 5):
            x = getattr(model.encoder, f"patch_embed{stage}")(x)
            for b, module in enumerate(getattr(model.encoder, f"block{stage}")):
                x = module(x)
                if stage == 1 and b == 0:
                    outputs["S1B0"] = x
            x = getattr(model.encoder, f"norm{stage}")(x).contiguous()
            outputs[f"C{stage}"] = x
        outputs["logits"] = _expanded_decoder(model, outputs)
    return {name: _hwc(value) for name, value in outputs.items()}


def _assert_same_bits(actual, expanded):
    if set(actual) != set(expanded):
        raise AssertionError("expanded traversal omitted a full-forward boundary")
    for name, value in actual.items():
        other = expanded[name]
        if (value.shape != other.shape or value.dtype != np.float32 or other.dtype != np.float32
                or not np.array_equal(value.view(np.uint32), other.view(np.uint32))):
            raise AssertionError(f"expanded traversal is not bitwise equal at {name}")


def full_reference(state, input):
    """Actual full SMP forward hooks, audited by bitwise expanded traversal."""
    import torch

    validate_hwc(input, "input")
    model = make_reference(state)
    captures, handles = {}, []

    def capture(name):
        def hook(module, args, output):
            captures[name] = _hwc(output)
        return hook

    try:
        handles.append(model.encoder.block1[0].register_forward_hook(capture("S1B0")))
        for stage in range(1, 5):
            handles.append(getattr(model.encoder, f"norm{stage}").register_forward_hook(capture(f"C{stage}")))
        with torch.no_grad():
            captures["logits"] = _hwc(model(_nchw(input)))
    finally:
        for handle in handles:
            handle.remove()
    if set(captures) != set(SHAPES) - {"input"}:
        raise AssertionError("actual SMP forward omitted a required boundary")
    for name, value in captures.items():
        validate_hwc(value, name)
    _assert_same_bits(captures, expanded_reference(model, input))
    return captures


def isolated_reference(cut, state, actual_inputs):
    """Installed SMP modules evaluated on one cut's actual incoming arrays."""
    import torch

    inputs = validate_cut_inputs(cut, actual_inputs)
    model = make_reference(state)
    enc = model.encoder
    with torch.no_grad():
        if cut == "stage1a":
            x = enc.block1[0](enc.patch_embed1(_nchw(inputs["input"])))
        elif cut == "stage1b":
            x = enc.norm1(enc.block1[1](_nchw(inputs["S1B0"]))).contiguous()
        elif cut == "decoder":
            # SMP uses the input only for its target spatial shape; dummy is the
            # encoder's documented zero-channel placeholder. Neither's values
            # enter the decoder arithmetic, which consumes the four actual C's.
            source = torch.empty((1, 2, 224, 224), dtype=torch.float32)
            dummy = torch.empty((1, 0, 112, 112), dtype=torch.float32)
            features = [source, dummy] + [_nchw(inputs[f"C{s}"]) for s in range(1, 5)]
            x = model.segmentation_head(model.decoder(features))
        else:
            stage = int(cut[-1])
            x = getattr(enc, f"patch_embed{stage}")(_nchw(inputs[f"C{stage - 1}"]))
            x = getattr(enc, f"block{stage}")(x)
            x = getattr(enc, f"norm{stage}")(x).contiguous()
    return _hwc(x)
