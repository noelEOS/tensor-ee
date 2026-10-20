"""Fixed-input adapters for segmentation-models-pytorch CNNs, without downloads.

``adapt_smp_model`` copies an eval CPU float32 model, omits the base forward's
Python input-shape check and disables copied inplace flags. Shared compiler
semantics specialize only proven bookkeeping. Arithmetic stays subject to
``check_compatibility``; adaptation alone does not promise compilation.
"""
from __future__ import annotations

from copy import deepcopy
from numbers import Integral
from typing import Any

import torch
from torch import nn
from torch.fx import GraphModule, symbolic_trace

from .semantics import specialize_plumbing


SMP_VERSION = "0.5.0"


class _BaseForward(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.segmentation_head = model.segmentation_head

    def forward(self, x):
        return self.segmentation_head(self.decoder(self.encoder(x)))


def _validate_input(module: nn.Module, inputs: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    if kwargs:
        if inputs or set(kwargs) != {"x"}:
            raise ValueError("SMP adapter expects one input tensor named x")
        inputs = (kwargs["x"],)
    if len(inputs) != 1 or not isinstance(inputs[0], torch.Tensor):
        raise ValueError("SMP adapter expects one torch.Tensor input")
    tensor = inputs[0]
    expected = (1, *module.tensor_ee_input_shape)
    if tuple(tensor.shape) != expected:
        raise ValueError(f"SMP adapter requires fixed NCHW input {expected}; received {tuple(tensor.shape)}")
    if tensor.dtype != torch.float32 or tensor.device.type != "cpu":
        raise ValueError("SMP adapter input must be CPU float32")


def _validate_model(model, smp):
    from segmentation_models_pytorch.base import SegmentationModel

    if isinstance(model, smp.Segformer):
        raise ValueError("Segformer uses the attention path; use tensor_ee.converter.convert")
    classes = (smp.Unet, smp.UnetPlusPlus, smp.MAnet, smp.Linknet, smp.FPN, smp.PSPNet,
               smp.PAN, smp.DeepLabV3, smp.DeepLabV3Plus, smp.UPerNet)
    if not isinstance(model, classes):
        raise ValueError("Expected an SMP CNN with the inherited SegmentationModel.forward; forward overrides such as DPT are unsupported")
    if getattr(model.forward, "__func__", None) is not SegmentationModel.forward:
        raise ValueError("SMP forward overrides are unsupported; the inherited SegmentationModel.forward is required")
    if model.classification_head is not None:
        raise ValueError("SMP classification_head must be None; aux_params classification outputs cannot be discarded")
    for module in model.modules():
        if module.training:
            raise ValueError("Set every SMP module to eval() before adaptation")
        if any(getattr(module, name, None) for name in (
                "_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks")):
            raise ValueError("SMP adaptation requires modules without hooks")
    # Global hooks also change calls made by both the reference and the adapter.
    from torch.nn.modules import module as module_hooks
    if any(getattr(module_hooks, name, None) for name in (
            "_global_forward_hooks", "_global_forward_pre_hooks",
            "_global_backward_hooks", "_global_backward_pre_hooks")):
        raise ValueError("SMP adaptation requires no global module hooks")
    for name, tensor in (*model.named_parameters(), *model.named_buffers()):
        if tensor.device.type != "cpu":
            raise ValueError(f"SMP tensor {name!r} must be on CPU")
        if tensor.is_complex() or tensor.is_floating_point() and tensor.dtype != torch.float32:
            raise ValueError(f"SMP tensor {name!r} must use float32")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"SMP tensor {name!r} must be finite")


def adapt_smp_model(model: nn.Module, *, input_shape: tuple[int, int, int, int]) -> GraphModule:
    """Return an independent, frozen FX module checked at exactly ``input_shape``.

    Requires SMP 0.5.0, eval CPU float32 state, batch one, no classification
    head/auxiliary output, no hooks, and the inherited base forward. Segformer
    belongs to ``tensor_ee.converter.convert``; forward overrides (including
    DPT) are refused. Encoders and segmentation-head arithmetic are preserved,
    with operation-based support reported by ``check_compatibility``.

    The final module must equal a copy of the unmodified raw model bit for bit
    on one deterministic nonconstant probe at the requested NCHW shape. This
    check is local PyTorch inference, not an all-input or live EE claim. The
    source state, flags and CPU RNG state are preserved, including on failure.
    Version/probe facts are JSON-safe in ``tensor_ee_smp_metadata``. The eager
    guard and compiler enforce ``tensor_ee_input_shape`` (CHW, batch one).
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if (not isinstance(input_shape, tuple) or len(input_shape) != 4 or input_shape[0] != 1
            or any(isinstance(x, bool) or not isinstance(x, Integral) or x <= 0 for x in input_shape)):
        raise ValueError("input_shape must be a batch-one NCHW tuple of four positive integers")
    input_shape = tuple(int(x) for x in input_shape)
    with torch.random.fork_rng(devices=[]):
        try:
            import segmentation_models_pytorch as smp
        except ImportError as error:
            raise ImportError("SMP adaptation requires segmentation-models-pytorch==0.5.0") from error
        if smp.__version__ != SMP_VERSION:
            raise ValueError(f"SMP adaptation requires SMP {SMP_VERSION}; received {smp.__version__}")
        _validate_model(model, smp)
        # Evaluate an untouched copy so even custom encoder state cannot mutate
        # the caller's model during the mandatory reference probe.
        reference = deepcopy(model)
        copied = deepcopy(reference)
        for module in copied.modules():
            if hasattr(module, "inplace"):
                module.inplace = False
        wrapper = _BaseForward(copied).requires_grad_(False).eval()
        try:
            traced = symbolic_trace(wrapper)
        except Exception as error:
            raise ValueError(f"SMP FX trace failed: {type(error).__name__}: {error}") from error
        adapted = specialize_plumbing(traced, input_shape).graph
        adapted.requires_grad_(False).eval()
        adapted.tensor_ee_input_shape = input_shape[1:]
        adapted.register_forward_pre_hook(_validate_input, with_kwargs=True)
        # Dyadic, spatially varying, deterministic values; no RNG is consumed.
        elements = input_shape[1] * input_shape[2] * input_shape[3]
        probe = ((torch.arange(elements, device="cpu", dtype=torch.int64) % 251 - 125)
                 .to(torch.float32).reshape(input_shape) / 128)
        with torch.inference_mode():
            expected, actual = reference(probe), adapted(probe)
        if not isinstance(expected, torch.Tensor) or not isinstance(actual, torch.Tensor):
            raise ValueError("SMP adapter requires one tensor output")
        if expected.shape != actual.shape:
            raise ValueError(f"SMP probe output shape mismatch: {tuple(expected.shape)} versus {tuple(actual.shape)}")
        if not bool(torch.isfinite(expected).all() and torch.isfinite(actual).all()):
            raise ValueError("SMP probe outputs must be finite")
        equal = torch.equal(expected, actual)
        max_abs = float((expected - actual).abs().max().item())
        if not equal:
            raise ValueError(f"SMP adapter probe is not bit-equal at {input_shape}; max_abs={max_abs:.9g}")
        adapted.tensor_ee_smp_metadata = {
            "architecture": f"{type(model).__module__}.{type(model).__qualname__}",
            "segmentation_models_pytorch_version": smp.__version__,
            "torch_version": str(torch.__version__), "input_shape": list(input_shape),
            "output_shape": list(actual.shape), "dtype": "float32",
            "probe": "nonconstant dyadic arange modulo 251, divided by 128",
            "torch_equal": equal, "max_abs": max_abs,
            "parameters_frozen": True, "spatial_generalization_supported": False,
        }
        return adapted
