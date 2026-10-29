"""Pure FX call validation and finite-grid tensor metadata shared with compilation.

This module neither constructs EE objects nor executes model activations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import builtins
from copy import deepcopy
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
# use them, and their errors to core; these names are re-exports of the
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


def interpolate_options(options: dict[str, Any], input_metadata: TensorMetadata | None = None) -> int:
    if options["mode"] != "nearest":
        _reject("Only nearest interpolation with scale_factor (not size) is supported")
    if options["align_corners"] is not None or options["antialias"] is not False:
        _reject("nearest interpolation does not accept align_corners or antialias")
    recompute = options["recompute_scale_factor"]
    if recompute is not None and not isinstance(recompute, bool):
        _reject("recompute_scale_factor must be None or a static bool")
    size = options["size"]
    if size is not None:
        if options["scale_factor"] is not None:
            _reject("nearest interpolation requires exactly one of size and scale_factor")
        if recompute is True:
            _reject("size-based nearest interpolation does not accept recompute_scale_factor=True")
        if (not isinstance(size, (tuple, list)) or len(size) != 2
                or any(isinstance(x, bool) or not isinstance(x, Integral) or x <= 0 for x in size)):
            _reject("nearest interpolation size must be a static pair of positive integers")
        if input_metadata is None:
            raise SemanticUnknown("nearest interpolation size requires a proven input spatial shape")
        h, w = input_metadata.shape[2:]
        # Fraction avoids rounding a nearly integral ratio into an acceptance.
        return nearest_factor((Fraction(int(size[0]), h), Fraction(int(size[1]), w)))
    return nearest_factor(options["scale_factor"])


@dataclass(frozen=True)
class _Shape:
    dimensions: tuple[int, ...]


@dataclass(frozen=True)
class PlumbingResult:
    graph: GraphModule
    errors: dict[Node, Exception]


def _static_value(value, values):
    if isinstance(value, Node):
        return values.get(value, value)
    if isinstance(value, (tuple, list)):
        return type(value)(_static_value(x, values) for x in value)
    return value


def _literal_size(value, values):
    value = _static_value(value, values)
    if isinstance(value, _Shape):
        return value.dimensions
    if isinstance(value, (tuple, list)):
        return type(value)(_literal_size(x, values) for x in value)
    return value


def _plumbing_value(node, metadata, values):
    if node.target is builtins.getattr:
        if len(node.args) != 2 or node.kwargs or node.args[1] != "shape":
            _reject("Only getattr(tensor, 'shape') with proven tensor metadata is supported")
        source = node.args[0]
        if not isinstance(source, Node) or metadata.get(source) is None:
            raise SemanticUnknown("Tensor shape is unavailable from structural metadata")
        return _Shape(metadata[source].shape)
    if len(node.args) != 2 or node.kwargs:
        _reject("Static getitem requires a container and a literal integer or slice")
    source, index = node.args
    integer = lambda x: isinstance(x, Integral) and not isinstance(x, bool)
    if not (integer(index) or isinstance(index, slice) and all(
            x is None or integer(x) for x in (index.start, index.stop, index.step))):
        _reject("Static getitem requires a literal integer or slice; dynamic indexing is unsupported")
    container = _static_value(source, values)
    shape = isinstance(container, _Shape)
    if shape:
        container = container.dimensions
    elif not (isinstance(container, (tuple, list)) and container
              and all(isinstance(x, Node) for x in container)):
        _reject("getitem requires a proven shape tuple or literal tensor list/tuple; tensor-value indexing is unsupported")
    elif any(metadata.get(x) is None for x in container):
        raise SemanticUnknown("Literal feature selection requires proven tensor references")
    try:
        selected = container[index]
    except (IndexError, ValueError, TypeError) as error:
        _reject(f"Invalid static getitem: {error}")
    return _Shape(tuple(selected)) if shape and isinstance(index, slice) else selected


def specialize_plumbing(graph: GraphModule, input_shape: tuple[int, int, int, int] | None) -> PlumbingResult:
    """Resolve only proven shape/container bookkeeping and exact nearest sizes.

    Work on an independent FX graph, without executing activations or changing
    modules. Unsupported arithmetic and its node names remain for diagnostics.
    No dead arithmetic is removed, and no raster operation or cast is added.
    Compiler, compatibility analysis and model adapters use this same pass.
    """
    def needs_specialization(node):
        if node.op == "call_function":
            return (node.target in (builtins.getattr, operator.getitem)
                    or node.target is F.interpolate and (node.kwargs.get("size") is not None
                        or len(node.args) > 1 and node.args[1] is not None))
        return (node.op == "call_module" and isinstance(graph.get_submodule(str(node.target)), nn.Upsample)
                and graph.get_submodule(str(node.target)).size is not None)

    if not any(needs_specialization(node) for node in graph.graph.nodes):
        return PlumbingResult(graph, {})
    graph_copy = deepcopy(graph.graph)
    metadata: dict[Node, TensorMetadata] = {}
    values: dict[Node, Any] = {}
    errors: dict[Node, Exception] = {}
    changed = False
    placeholders = 0
    for node in graph_copy.nodes:
        if node.op == "placeholder":
            placeholders += 1
            if placeholders == 1 and input_shape is not None:
                metadata[node] = TensorMetadata(input_shape)
            continue
        if node.op == "output":
            continue
        plumbing = node.op == "call_function" and node.target in (builtins.getattr, operator.getitem)
        size_call = False
        try:
            if plumbing:
                value = _plumbing_value(node, metadata, values)
                values[node] = value
                if isinstance(value, Node):
                    # A tensor alias must disappear before Conv/BN adjacency is checked.
                    node.replace_all_uses_with(value)
                    if value in metadata:
                        metadata[node] = metadata[value]
                changed = True
                continue
            options = None
            if node.op == "call_function" and node.target is F.interpolate:
                options = _bind_options(_interpolate_arguments, node.args, node.kwargs)
            elif node.op == "call_module" and isinstance(graph.get_submodule(str(node.target)), nn.Upsample):
                module = graph.get_submodule(str(node.target))
                source = _bind_options(_input_arguments, node.args, node.kwargs)["input"]
                options = dict(input=source, size=module.size, scale_factor=module.scale_factor,
                               mode=module.mode, align_corners=module.align_corners,
                               recompute_scale_factor=module.recompute_scale_factor, antialias=False)
            if options is not None and options["size"] is not None:
                size_call = True
                options["size"] = _literal_size(options["size"], values)
                factor = interpolate_options(options, metadata.get(options["input"]))
                options.update(size=None, scale_factor=factor)
                node.op, node.target = "call_function", F.interpolate
                node.args, node.kwargs = (options.pop("input"),), options
                changed = True
            call = semantic_call(node, graph)
            if call.kind == "batchnorm_constant":
                continue
            if all(x in metadata for x in call.inputs):
                metadata[node] = infer_metadata(call, tuple(metadata[x] for x in call.inputs))
        except (UnsupportedOperation, SemanticUnknown, TypeError) as error:
            if plumbing or size_call:
                errors[node] = error if not isinstance(error, TypeError) else UnsupportedOperation(
                    f"Invalid call arguments: {error}")
    # Replace scalar/tuple uses only after their provenance has been checked.
    from torch.fx.node import map_arg
    for node in graph_copy.nodes:
        node.args = map_arg(node.args, lambda x: _literal_size(values[x], values) if x in values else x)
        node.kwargs = map_arg(node.kwargs, lambda x: _literal_size(values[x], values) if x in values else x)
    for node in reversed(list(graph_copy.nodes)):
        if node in values and not node.users:
            graph_copy.erase_node(node)
    if not changed:
        # Errors refer to the copied nodes; keep that graph even without a lowering.
        if not errors:
            return PlumbingResult(graph, {})
    specialized = GraphModule(graph, graph_copy)
    specialized.training = graph.training
    return PlumbingResult(specialized, errors)


def _inputs(values: Any) -> tuple[Node, ...]:
    if not isinstance(values, (tuple, list)) or not values or any(not isinstance(x, Node) for x in values):
        _reject("Operation requires tensor inputs, without scalars or broadcasting", "input_shape")
    return tuple(values)


def _batchnorm_options(mean, var, gamma, beta, eps, channels) -> dict[str, Any]:
    """Static checks shared by folded, standalone and functional inference BN."""
    if isinstance(eps, bool) or not isinstance(eps, Real) or not math.isfinite(eps):
        _reject("BatchNorm eps must be a finite static real")
    for name, value in (("running_mean", mean), ("running_var", var), ("weight", gamma), ("bias", beta)):
        if name in ("weight", "bias") and value is None:
            continue
        _coefficient_field(value, f"BatchNorm2d.{name}", (channels,))
        try:
            if not bool(torch.isfinite(value.detach()).all().item()):
                _reject(f"Local coefficient BatchNorm2d.{name} must contain only finite tensors", "nonfinite_coefficients")
        except UnsupportedOperation:
            raise
        except Exception as error:
            raise SemanticUnknown(f"Cannot inspect BatchNorm statistics: {error}") from error
    try:
        if not bool(((var.detach() + eps) > 0).all().item()):
            _reject("BatchNorm running variance plus eps must be positive", "nonfinite_coefficients")
    except UnsupportedOperation:
        raise
    except Exception as error:
        raise SemanticUnknown(f"Cannot inspect BatchNorm statistics: {error}") from error
    return dict(mean=mean, var=var, gamma=gamma, beta=beta, eps=eps, channels=channels)


def _module_batchnorm_options(module: nn.BatchNorm2d) -> dict[str, Any]:
    if module.training or not module.track_running_stats:
        _reject("BatchNorm must use fixed inference running statistics", "training_mode" if module.training else "unsupported_option")
    if module.affine:
        for name in ("weight", "bias"):
            _coefficient_field(getattr(module, name), f"BatchNorm2d.{name}", (module.num_features,))
    options = _batchnorm_options(module.running_mean, module.running_var, module.weight,
                                 module.bias, module.eps, module.num_features)
    finite_coefficients(module)
    return options


def batchnorm_predecessor(node: Node, graph: GraphModule) -> tuple[Node, Node, nn.Conv2d] | None:
    """The unchanged convolution fold, or None for a standalone inference BatchNorm."""
    module = graph.get_submodule(str(node.target))
    _module_batchnorm_options(module)
    predecessor = _bind_options(_input_arguments, node.args, node.kwargs)["input"]
    if (not isinstance(predecessor, Node) or predecessor.op != "call_module" or len(predecessor.users) != 1):
        return None
    conv = graph.get_submodule(str(predecessor.target))
    if not isinstance(conv, nn.Conv2d):
        return None
    if conv.training or module.training:
        _reject("Set model.eval() before compilation", "training_mode")
    if module.num_features != conv.out_channels:
        _reject("BatchNorm running statistics must match convolution output channels")
    conv_options(conv)
    # Preserve the compiler's existing folding operation and coefficient identity.
    fused = torch.nn.utils.fuse_conv_bn_eval(conv, module)
    finite_coefficients(fused)
    source = _bind_options(_input_arguments, predecessor.args, predecessor.kwargs)["input"]
    _inputs((source,))
    return predecessor, source, fused


_BATCHNORM_CONSTANTS = ("running_mean", "running_var", "weight", "bias")


def _batchnorm_attribute(node: Node, graph: GraphModule):
    if not isinstance(node, Node) or node.op != "get_attr":
        _reject("Functional BatchNorm coefficients must be get_attr buffers or parameters")
    try:
        return graph.get_parameter(str(node.target))
    except AttributeError:
        try:
            return graph.get_buffer(str(node.target))
        except AttributeError:
            _reject("Functional BatchNorm coefficients must be get_attr buffers or parameters")


def batchnorm_constant(node: Node, graph: GraphModule) -> bool:
    """Allow get_attr only when every use is one of F.batch_norm's four coefficients."""
    if node.op != "get_attr" or not node.users:
        return False
    for user in node.users:
        if user.op != "call_function" or user.target is not F.batch_norm:
            return False
        try:
            options = _bind_options(F.batch_norm, user.args, user.kwargs)
        except TypeError:
            return False
        if any(node in _nodes_in(value) for key, value in options.items() if key not in _BATCHNORM_CONSTANTS):
            return False
        if not any(options[key] is node for key in _BATCHNORM_CONSTANTS):
            return False
    try:
        _batchnorm_attribute(node, graph)
    except UnsupportedOperation:
        return False
    return True


def _nodes_in(value):
    from torch.fx.node import map_arg
    nodes = []
    map_arg(value, lambda node: nodes.append(node) or node)
    return nodes


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
                folded = batchnorm_predecessor(node, graph)
                if folded is not None:
                    predecessor, source, fused = folded
                    return SemanticCall("batchnorm", inputs, {"source": source, "predecessor": predecessor}, fused)
                return SemanticCall("batch_norm", inputs, _module_batchnorm_options(module))
            _reject(f"Unsupported module {type(module).__name__}", "unsupported_operator")
        if node.op == "call_function":
            target = node.target
            if target is F.batch_norm:
                options = _bind_options(F.batch_norm, node.args, node.kwargs)
                if options["training"] is not False:
                    _reject("Functional BatchNorm training must be the literal False", "training_mode")
                values = {key: (None if options[key] is None and key in ("weight", "bias")
                                else _batchnorm_attribute(options[key], graph))
                          for key in _BATCHNORM_CONSTANTS}
                mean = values["running_mean"]
                channels = mean.numel() if isinstance(mean, torch.Tensor) else 0
                return SemanticCall("batch_norm", _inputs((options["input"],)),
                                    _batchnorm_options(mean, values["running_var"], values["weight"],
                                                       values["bias"], options["eps"], channels))
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
        if batchnorm_constant(node, graph):
            return SemanticCall("batchnorm_constant", ())
        _reject(f"Unsupported FX operation {node.op}: {node.target}", "unsupported_operator")
    except TypeError as error:
        _reject(f"Invalid call arguments: {error}")


def infer_metadata(call: SemanticCall, inputs: tuple[TensorMetadata, ...]) -> TensorMetadata:
    first = inputs[0]
    n, c, h, w = first.shape
    if call.kind == "batch_norm" and c != call.options["channels"]:
        _reject("BatchNorm input channel count does not match raster", "input_shape")
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
