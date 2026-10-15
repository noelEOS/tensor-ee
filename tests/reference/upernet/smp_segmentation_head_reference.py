"""Milestone 11, Phases 1-4 (Tests 1-2, 6-8): the installed SMP segmentation head, read off the
installed code and run.

segmentation_models_pytorch 0.5.0:

    base/heads.py         SegmentationHead(nn.Sequential): Conv2d(in, out, kernel_size,
                          padding=kernel_size // 2) [bias=True: nn.Conv2d's default],
                          nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1,
                          Activation(activation)
    decoders/upernet/model.py   UPerNet.__init__: SegmentationHead(in_channels=decoder_channels,
                          out_channels=classes, activation=activation (None),
                          kernel_size=1, upsampling=upsampling (4))
    base/model.py         SegmentationModel.forward: masks = self.segmentation_head(decoder_output)
    torch.nn.UpsamplingBilinear2d(scale_factor) = Upsample(mode="bilinear",
                          align_corners=True); forward: F.interpolate(input, None, 4.0,
                          "bilinear", True, recompute_scale_factor=None)

so for the project's model (decoder_channels 256, classes 2):

    head_conv  = Conv2d(256, 2, 1)(fused)                 [B, 2, H/4, W/4], bias added
    logits     = interpolate(head_conv, x4, align_corners=True)     [B, 2, H, W]
    output     = Identity(logits) = model.forward's return value (raw logits)

`contract()` records all of it from the installed objects and a real forward; the
coefficient study shows that PyTorch evaluates align_corners=True in the tensor's own
precision (scale = (N - 1) / (M - 1) and src = scale * o rounded to float32 in the float32
model) and that its float32 kernel equals the written float32 formula at every output
index (reference/pytorch_decoder_primitives.align_corners_formula).
"""

from __future__ import annotations

import inspect

import numpy as np
import torch
import torch.nn.functional as F

from reference.upernet import smp_feature_interface as S


def _lines(obj) -> dict:
    src, start = inspect.getsourcelines(obj)
    return {"file": inspect.getsourcefile(obj), "lines": [start, start + len(src) - 1], "source": "".join(src)}


def _grep(obj, needle: str, after: int = 0) -> dict:
    src, start = inspect.getsourcelines(obj)
    i = next(i for i, line in enumerate(src) if needle in line)
    return {"file": inspect.getsourcefile(obj), "lines": [start + i, start + i + after],
            "source": "".join(src[i:i + after + 1])}


def source() -> dict:
    from segmentation_models_pytorch.base import heads, model as base_model
    from segmentation_models_pytorch.base.modules import Activation
    from segmentation_models_pytorch.decoders.upernet import model as upernet_model
    return {"SegmentationHead": _lines(heads.SegmentationHead),
            "UPerNet_head_construction": _grep(upernet_model.UPerNet.__init__, "self.segmentation_head =", 6),
            "SegmentationModel_forward": _grep(base_model.SegmentationModel.forward,
                                               "masks = self.segmentation_head"),
            "Activation_init": _lines(Activation.__init__),
            "UpsamplingBilinear2d_init": _lines(torch.nn.UpsamplingBilinear2d.__init__),
            "Upsample_forward": _lines(torch.nn.Upsample.forward)}


def head_config(head) -> dict:
    conv, up, act = head[0], head[1], head[2]
    return {"modules": [f"{name}: {type(m).__name__}" for name, m in head.named_children()],
            "repr": repr(head),
            "conv": {"class": type(conv).__name__, "in": conv.in_channels, "out": conv.out_channels,
                     "kernel": list(conv.kernel_size), "padding": list(conv.padding),
                     "padding_mode": conv.padding_mode, "stride": list(conv.stride),
                     "dilation": list(conv.dilation), "groups": conv.groups, "bias": conv.bias is not None},
            "upsample": {"class": type(up).__name__, "mode": up.mode, "align_corners": up.align_corners,
                         "scale_factor": up.scale_factor, "size": up.size,
                         "recompute_scale_factor": up.recompute_scale_factor},
            "activation": {"class": type(act).__name__, "inner": type(act.activation).__name__}}


def interpolation_calls(size: int = 224) -> list[dict]:
    """The F.interpolate calls of a whole forward at an input of size x size (the head's is
    the last)."""
    m = S.make_upernet(size, size)
    x = torch.from_numpy(np.random.default_rng(1).standard_normal((1, S.IN_CHANNELS, size, size))
                         .astype(np.float32))
    with S.recorded_interpolations() as calls, torch.no_grad():
        m(x)
    return calls


def real_forward(size: int = 224, seed: int = 0) -> dict:
    """A real forward (random input): the head's input is the decoder output, its output is
    model.forward's, the layouts the tensors have, and the upsample's equivalents."""
    from reference.upernet import full_model_reference as FM
    m = S.make_upernet(size, size)
    x = torch.from_numpy(np.random.default_rng(seed).standard_normal((1, S.IN_CHANNELS, size, size))
                         .astype(np.float32))
    cap = FM.capture(m, x)
    FM.expanded(m, x, cap)                                     # raises on any mismatch
    conv = cap["head_conv"]
    with torch.no_grad():
        variants = {"F.interpolate(size=4h, align_corners=True)":
                    F.interpolate(conv, size=(4 * conv.shape[2], 4 * conv.shape[3]), mode="bilinear",
                                  align_corners=True),
                    "F.interpolate(scale_factor=4, align_corners=True)":
                    F.interpolate(conv, scale_factor=4, mode="bilinear", align_corners=True),
                    "channels_last input": torch.nn.UpsamplingBilinear2d(scale_factor=4)(
                        conv.contiguous(memory_format=torch.channels_last)),
                    "align_corners=False (negative)":
                    F.interpolate(conv, scale_factor=4, mode="bilinear", align_corners=False)}
    return {"input_size": size, "expanded_equals_model_forward": True,
            "head_input_is_decoder_output": True, "logits_is_upsample_output": True,
            "layouts": {n: cap["layouts"][n] for n in ("fusion_relu", "head_input", "head_conv",
                                                       "head_upsampled", "logits")},
            "upsample_equivalents_bit_identical": {k: bool(torch.equal(v, cap["head_upsampled"]))
                                                   for k, v in variants.items()},
            "output_shape": list(cap["logits"].shape), "output_dtype": str(cap["logits"].dtype)}
