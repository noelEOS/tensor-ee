"""Pure FX call validation and finite-grid tensor metadata shared with compilation.

This module neither constructs EE objects nor executes model activations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import math
from numbers import Integral, Real
import operator
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.fx import GraphModule, Node

from .compiler import UnsupportedOperation, _bind_options, _input_arguments, _pair, _pointwise_call
# Block option and coefficient checks moved to the library with the blocks that
# use them, and their errors to core (L03); these names are re-exports of the
# same objects.
from .core.errors import SemanticUnknown, reject as _reject
from .blocks.raster.conv import _coefficient_field, conv_options, finite_coefficients
from .blocks.raster.pooling import aligned_size_message, pool_config, pool_options
from .blocks.raster.resample import nearest_factor


@dataclass(frozen=True)
class SemanticCall:
    kind: str
    inputs: tuple[Node, ...]
    options: dict[str, Any] = field(default_factory=dict)
    module: nn.Module | None = None


@dataclass(frozen=True)
class TensorMetadata:
    shape: tuple[int, int, int, int]
    scale: Fraction = Fraction(1)


def _relu_arguments(input, inplace=False):
    pass


def _out_arguments(input, *, out=None):
    pass


def _cat_arguments(tensors, dim=0, *, out=None):
    pass


def _add_arguments(input, other, *, alpha=1, out=None):
    pass


def _operator_add_arguments(input, other, /):
    pass


def _interpolate_arguments(input, size=None, scale_factor=None, mode="nearest", align_corners=None,
                           recompute_scale_factor=None, antialias=False):
    pass


def interpolate_options(options: dict[str, Any]) -> int:
    if options["size"] is not None or options["mode"] != "nearest":
        _reject("Only nearest interpolation with scale_factor (not size) is supported")
    if options["align_corners"] is not None or options["antialias"] is not False:
        _reject("nearest interpolation does not accept align_corners or antialias")
    recompute = options["recompute_scale_factor"]
    if recompute is not None and not isinstance(recompute, bool):
        _reject("recompute_scale_factor must be None or a static bool")
    return nearest_factor(options["scale_factor"])


def _inputs(values: Any) -> tuple[Node, ...]:
    if not isinstance(values, (tuple, list)) or not values or any(not isinstance(x, Node) for x in values):
        _reject("Operation requires tensor inputs, without scalars or broadcasting", "input_shape")
    return tuple(values)


def batchnorm_predecessor(node: Node, graph: GraphModule) -> tuple[Node, Node, nn.Conv2d]:
    module = graph.get_submodule(str(node.target))
    if module.training or not module.track_running_stats:
        _reject("BatchNorm must use fixed inference running statistics", "training_mode" if module.training else "unsupported_option")
    predecessor = _bind_options(_input_arguments, node.args, node.kwargs)["input"]
    if (not isinstance(predecessor, Node) or predecessor.op != "call_module" or len(predecessor.users) != 1):
        _reject("BatchNorm must directly follow an unshared Conv2d")
    conv = graph.get_submodule(str(predecessor.target))
    if not isinstance(conv, nn.Conv2d) or not module.track_running_stats:
        _reject("BatchNorm requires a preceding Conv2d and running statistics")
    if conv.training or module.training:
        _reject("Set model.eval() before compilation", "training_mode")
    if module.num_features != conv.out_channels or module.running_mean is None or module.running_var is None:
        _reject("BatchNorm running statistics must match convolution output channels")
    if isinstance(module.eps, bool) or not isinstance(module.eps, Real) or not math.isfinite(module.eps):
        _reject("BatchNorm eps must be a finite static real")
    conv_options(conv)
    for name in ("running_mean", "running_var"):
        _coefficient_field(getattr(module, name), f"BatchNorm2d.{name}", (module.num_features,))
    for name in ("weight", "bias"):
        value = getattr(module, name)
        if module.affine or value is not None:
            _coefficient_field(value, f"BatchNorm2d.{name}", (module.num_features,))
    finite_coefficients(module)
    try:
        if not bool(((module.running_var.detach() + module.eps) > 0).all().item()):
            _reject("BatchNorm running variance plus eps must be positive", "nonfinite_coefficients")
    except UnsupportedOperation:
        raise
    except Exception as error:
        raise SemanticUnknown(f"Cannot inspect BatchNorm statistics: {error}") from error
    # Preserve the compiler's existing folding operation and coefficient identity.
    fused = torch.nn.utils.fuse_conv_bn_eval(conv, module)
    finite_coefficients(fused)
    source = _bind_options(_input_arguments, predecessor.args, predecessor.kwargs)["input"]
    _inputs((source,))
    return predecessor, source, fused


def diagnostic_convolution(node: Node, graph: GraphModule) -> bool:
    if len(node.users) != 1:
        return False
    user = next(iter(node.users))
    return user.op == "call_module" and isinstance(graph.get_submodule(str(user.target)), nn.BatchNorm2d)


def semantic_call(node: Node, graph: GraphModule) -> SemanticCall:
    """Normalize the actual supported compiler calls, rejecting ignored options."""
    try:
        pointwise = _pointwise_call(node, graph)
        if pointwise is not None:
            kind, source, options = pointwise
            return SemanticCall(kind, _inputs((source,)), options)
        if node.op == "call_module":
            module = graph.get_submodule(str(node.target))
            source = _bind_options(_input_arguments, node.args, node.kwargs)["input"]
            inputs = _inputs((source,))
            if isinstance(module, nn.Conv2d):
                conv_options(module)
                return SemanticCall("conv", inputs, module=module)
            if isinstance(module, (nn.AvgPool2d, nn.MaxPool2d)):
                return SemanticCall("pool", inputs, {"config": pool_config(module)}, module)
            if isinstance(module, nn.Upsample):
                options = dict(size=module.size, scale_factor=module.scale_factor, mode=module.mode,
                               align_corners=module.align_corners, recompute_scale_factor=module.recompute_scale_factor,
                               antialias=False)
                return SemanticCall("nearest", inputs, {"factor": interpolate_options(options)})
            if isinstance(module, nn.ReLU):
                if module.inplace is not False:
                    _reject("In-place ReLU is unsupported")
                return SemanticCall("relu", inputs)
            if isinstance(module, nn.Sigmoid):
                return SemanticCall("sigmoid", inputs)
            if isinstance(module, nn.Identity):
                return SemanticCall("identity", inputs)
            if isinstance(module, nn.BatchNorm2d):
                predecessor, source, fused = batchnorm_predecessor(node, graph)
                return SemanticCall("batchnorm", inputs, {"source": source, "predecessor": predecessor}, fused)
            _reject(f"Unsupported module {type(module).__name__}", "unsupported_operator")
        if node.op == "call_function":
            target = node.target
            if target in (torch.cat, torch.concat, torch.concatenate):
                options = _bind_options(_cat_arguments, node.args, node.kwargs)
                if "out" in node.kwargs:
                    _reject("out is unsupported for concatenation")
                dim = options["dim"]
                if isinstance(dim, bool) or not isinstance(dim, Integral) or dim not in (1, -3):
                    _reject("Concatenation requires image channels, NCHW channel dim=1 or -3")
                return SemanticCall("cat", _inputs(options["tensors"]), {"dim": dim})
            if target in (operator.add, torch.add):
                options = _bind_options(_operator_add_arguments if target is operator.add else _add_arguments,
                                        node.args, node.kwargs)
                if "out" in node.kwargs or isinstance(options.get("alpha", 1), bool) or options.get("alpha", 1) != 1:
                    _reject("Residual addition requires two tensors and alpha=1 without out")
                return SemanticCall("add", _inputs((options["input"], options["other"])))
            if target in (torch.relu, F.relu, torch.sigmoid, F.sigmoid):
                signature = (_relu_arguments if target is F.relu else _out_arguments
                             if target in (torch.relu, torch.sigmoid) else _input_arguments)
                options = _bind_options(signature, node.args, node.kwargs)
                if "out" in node.kwargs or options.get("inplace", False) is not False:
                    _reject("out and in-place activation calls are unsupported")
                return SemanticCall("relu" if target in (torch.relu, F.relu) else "sigmoid", _inputs((options["input"],)))
            if target is F.interpolate:
                options = _bind_options(_interpolate_arguments, node.args, node.kwargs)
                return SemanticCall("nearest", _inputs((options["input"],)), {"factor": interpolate_options(options)})
            _reject(f"Unsupported function {target}", "unsupported_operator")
        if node.op == "call_method" and node.target in ("relu", "sigmoid", "add"):
            options = _bind_options(_add_arguments if node.target == "add" else _input_arguments, node.args, node.kwargs)
            if "out" in node.kwargs or isinstance(options.get("alpha", 1), bool) or options.get("alpha", 1) != 1:
                _reject("Residual addition requires two tensors and alpha=1 without out")
            values = (options["input"], options["other"]) if node.target == "add" else (options["input"],)
            return SemanticCall(str(node.target), _inputs(values))
        _reject(f"Unsupported FX operation {node.op}: {node.target}", "unsupported_operator")
    except TypeError as error:
        _reject(f"Invalid call arguments: {error}")


def infer_metadata(call: SemanticCall, inputs: tuple[TensorMetadata, ...]) -> TensorMetadata:
    first = inputs[0]
    n, c, h, w = first.shape
    if call.kind == "conv":
        if c != call.module.in_channels:
            _reject("Conv2d input channel count does not match raster", "input_shape")
        stride = _pair(call.module.stride)[0]
        return TensorMetadata((n, call.module.out_channels, (h + stride - 1) // stride,
                               (w + stride - 1) // stride), first.scale * stride)
    if call.kind == "pool":
        config = call.options["config"]
        if config.aligned and (h % config.kernel or w % config.kernel):
            _reject(aligned_size_message(config.kernel), "input_shape")
        return TensorMetadata((n, c, config.output_size(h), config.output_size(w)),
                              first.scale * config.stride)
    if call.kind == "nearest":
        factor = call.options["factor"]
        return TensorMetadata((n, c, h * factor, w * factor), first.scale / factor)
    if call.kind in ("cat", "add"):
        for other in inputs[1:]:
            if other.shape[0] != n or other.shape[2:] != (h, w) or other.scale != first.scale:
                _reject("Merge tensors must have identical spatial grids", "input_shape")
            if call.kind == "add" and other.shape[1] != c:
                _reject("Residual addition requires identical spatial grids and channel counts", "input_shape")
        if call.kind == "cat":
            return TensorMetadata((n, sum(x.shape[1] for x in inputs), h, w), first.scale)
    return first
