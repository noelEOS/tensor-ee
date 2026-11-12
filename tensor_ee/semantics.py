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
from .blocks.raster.conv import _coefficient_field, conv_options, finite_coefficients, grouped_conv_options
from .blocks.raster.pooling import aligned_size_message, pool_config, pool_options
from .blocks.raster.resample import nearest_factor
from .layouts.raster import canonical_phase, phase_pair


EMITS_NOTHING = ("batchnorm_constant", "conv_constant", "absorbed")
"""Call kinds that emit no tensor: coefficient reads that are operands of the one call
using them, and nodes absorbed into the convolution that reads them (zero padding, and an
``Identity`` directly ahead of a functional convolution)."""


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


def _adaptive_pool_arguments(input, output_size):
    pass


def _mean_arguments(input, dim=None, keepdim=False, *, dtype=None, out=None):
    pass


def _method_mean_arguments(input, dim=None, keepdim=False, *, dtype=None):
    pass


def _operator_mul_arguments(input, other, /):
    pass


def _mul_arguments(input, other, *, out=None):
    pass


def _method_mul_arguments(input, other):
    pass


def _conv2d_arguments(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """Signature only: ``F.conv2d`` (the same object as ``torch.conv2d``)."""


def _pad_arguments(input, pad, mode="constant", value=None):
    """Signature only: ``F.pad``."""


_GLOBAL_DIMS = ((2, 3), (3, 2), (-2, -1), (-1, -2))
"""The spatial axes of an NCHW mean that pool the whole map, as ``dim`` may name them."""


def _static_int(value) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def global_pool_size(output_size: Any) -> None:
    """Accept only the adaptive average pool to one pixel: ``1``, ``(1, 1)`` or ``[1, 1]``."""
    if _static_int(output_size) and output_size == 1:
        return
    if (isinstance(output_size, (tuple, list)) and len(output_size) == 2
            and all(_static_int(x) and x == 1 for x in output_size)):
        return
    _reject("Adaptive average pooling is supported only to one pixel (output_size 1, (1, 1) or [1, 1]): "
            f"global pooling; received output_size={output_size!r}")


def global_mean_options(options: dict[str, Any]) -> None:
    """Accept only the mean over both spatial axes with ``keepdim=True`` and no ``dtype``."""
    dim = options["dim"]
    if (not isinstance(dim, (tuple, list)) or len(dim) != 2 or not all(_static_int(x) for x in dim)
            or tuple(int(x) for x in dim) not in _GLOBAL_DIMS):
        _reject("mean is supported only over both spatial axes, dim (2, 3), (3, 2), (-2, -1) or (-1, -2), "
                f"with keepdim=True; received dim={dim!r}")
    if options["keepdim"] is not True:
        _reject(f"mean requires keepdim=True (a [N, C, 1, 1] tensor); received keepdim={options['keepdim']!r}")
    if options["dtype"] is not None:
        _reject(f"mean requires dtype=None; received dtype={options['dtype']!r}")


def interpolate_options(options: dict[str, Any], input_metadata: TensorMetadata | None = None) -> int:
    if options["mode"] == "bilinear":
        return bilinear_options(options, input_metadata)
    if options["mode"] != "nearest":
        _reject(f"Unsupported interpolation mode {options['mode']!r}; only nearest and bilinear are supported")
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


def bilinear_options(options: dict[str, Any], input_metadata: TensorMetadata | None = None) -> int:
    """Validate a bilinear call without deriving any interpolation coefficients."""
    if not isinstance(options["align_corners"], bool):
        _reject("Bilinear align_corners must be a static bool")
    if options["antialias"] is not False:
        _reject("Bilinear antialias must be False")
    recompute = options["recompute_scale_factor"]
    if recompute is not None and not isinstance(recompute, bool):
        _reject("recompute_scale_factor must be None or a static bool")
    size = options["size"]
    if size is not None:
        if options["scale_factor"] is not None:
            _reject("Bilinear interpolation requires exactly one of size and scale_factor")
        if recompute is True:
            _reject("Size-based bilinear interpolation does not accept recompute_scale_factor=True")
        if (not isinstance(size, (tuple, list)) or len(size) != 2
                or any(isinstance(x, bool) or not isinstance(x, Integral) or x <= 0 for x in size)):
            _reject("Bilinear interpolation size must be a static pair of positive integers")
        if input_metadata is None:
            raise SemanticUnknown("Bilinear interpolation size requires a proven input spatial shape")
        h, w = input_metadata.shape[2:]
        factors = (Fraction(int(size[0]), h), Fraction(int(size[1]), w))
    else:
        factors = _pair(options["scale_factor"])
    if any(isinstance(x, bool) or not isinstance(x, Real) or not math.isfinite(x) for x in factors):
        _reject("Bilinear upsampling requires static finite real scale factors")
    if any(x < 1 for x in factors):
        _reject("Bilinear downsampling is unsupported; scale factors must be at least one")
    if factors[0] != factors[1]:
        _reject("Bilinear upsampling requires equal scale factors on both axes")
    if any(int(x) != x for x in factors):
        _reject("Bilinear upsampling requires integer scale factors")
    return int(factors[0])


def _interpolation_call(inputs, options):
    factor = interpolate_options(options)
    if options["mode"] == "bilinear":
        return SemanticCall("bilinear", inputs, {"factor": factor, "align_corners": options["align_corners"]})
    return SemanticCall("nearest", inputs, {"factor": factor})


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
    """Resolve only proven shape/container bookkeeping and exact integer resize sizes.

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
            if call.kind in EMITS_NOTHING:
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


def _batchnorm_fold(node: Node, graph: GraphModule) -> tuple[Node, Convolution, nn.Conv2d] | None:
    """The convolution fold with the recognised convolution it folds, or None."""
    module = graph.get_submodule(str(node.target))
    _module_batchnorm_options(module)
    predecessor = _bind_options(_input_arguments, node.args, node.kwargs)["input"]
    if not isinstance(predecessor, Node) or len(predecessor.users) != 1:
        return None
    if predecessor.op == "call_module":
        conv = graph.get_submodule(str(predecessor.target))
        if not isinstance(conv, nn.Conv2d):
            return None
        if conv.training or module.training:
            _reject("Set model.eval() before compilation", "training_mode")
        if module.num_features != conv.out_channels:
            _reject("BatchNorm running statistics must match convolution output channels")
        recognised = convolution(predecessor, graph)
    elif predecessor.op == "call_function" and predecessor.target is torch.conv2d:
        recognised = convolution(predecessor, graph)
        if module.num_features != recognised.module.out_channels:
            _reject("BatchNorm running statistics must match convolution output channels")
    else:
        return None
    # Preserve the compiler's existing folding operation and coefficient identity.
    fused = torch.nn.utils.fuse_conv_bn_eval(recognised.module, module)
    finite_coefficients(fused)
    _inputs((recognised.source,))
    return predecessor, recognised, fused


def batchnorm_predecessor(node: Node, graph: GraphModule) -> tuple[Node, Node, nn.Conv2d] | None:
    """The unchanged convolution fold, or None for a standalone inference BatchNorm.

    The convolution is an ``nn.Conv2d`` module or a functional ``F.conv2d``, with any zero
    padding directly ahead of it (``convolution``); the source returned is what that
    convolution reads, the padding's input when a padding node is absorbed."""
    folded = _batchnorm_fold(node, graph)
    if folded is None:
        return None
    predecessor, recognised, fused = folded
    return predecessor, recognised.source, fused


# ---------------------------------------------------------------------------
# Convolutions: nn.Conv2d or F.conv2d, with the zero padding directly ahead of them
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Convolution:
    """A recognised convolution call.

    ``source``: the tensor it reads (the input of an absorbed padding node or Identity);
    ``module``: the ``nn.Conv2d`` the blocks run, the model's own module where today's form
    applies, else a stand-in holding the call's own weight and bias Parameters with padding
    ``kernel // 2``; ``phase``: ``canonical_phase`` of ``(phase_y, phase_x)``;
    ``padding``: the total zero padding ``(left, right, top, bottom)``, the padding node's
    plus the convolution's own; ``path``: the module path of its weight (state-dict names)."""

    source: Node
    module: nn.Conv2d
    phase: Any
    padding: tuple[int, int, int, int]
    path: str


_OTHER_PADDING = tuple(getattr(nn, name) for name in (
    "ReflectionPad2d", "ReplicationPad2d", "CircularPad2d") if hasattr(nn, name))
_PADDING_MESSAGE = ("Zero padding is supported only directly ahead of the one convolution that reads it "
                    "(nn.Conv2d or F.conv2d), as that convolution's padding")
_CONV_ATTRIBUTE_MESSAGE = ("Functional conv2d weight and bias must be get_attr parameters or buffers used only "
                           "as this call's operands")


def _is_convolution(node: Node, graph: GraphModule) -> bool:
    if node.op == "call_module":
        return isinstance(graph.get_submodule(str(node.target)), nn.Conv2d)
    return node.op == "call_function" and node.target is torch.conv2d


def _convolution_operands(node: Node) -> dict[str, Any]:
    if node.op == "call_module":
        return _bind_options(_input_arguments, node.args, node.kwargs)
    return _bind_options(_conv2d_arguments, node.args, node.kwargs)


def _read_by_one_convolution(node: Node, graph: GraphModule, *, functional: bool = False) -> bool:
    """Whether ``node``'s only use is as the input operand of one convolution call (a
    functional one only, with ``functional``). The padding node and the convolution both
    ask this, so their verdicts cannot disagree."""
    if len(node.users) != 1:
        return False
    user = next(iter(node.users))
    if not _is_convolution(user, graph) or (functional and user.op != "call_function"):
        return False
    try:
        operands = _convolution_operands(user)
    except TypeError:
        return False
    return operands["input"] is node and not any(
        node in _nodes_in(value) for key, value in operands.items() if key != "input")


def _static_padding(values, spelling: str) -> tuple[int, int, int, int]:
    if not isinstance(values, (tuple, list)) or any(isinstance(x, Node) for x in values):
        _reject(f"{spelling} padding must be static integers; padding computed from a tensor's shape "
                f"is unsupported (received {values!r})")
    if len(values) != 4:
        _reject(f"{spelling} must pad the two spatial axes with four integers (left, right, top, bottom); "
                f"received {values!r}")
    if any(isinstance(x, bool) or not isinstance(x, Integral) for x in values):
        _reject(f"{spelling} padding must be static integers; received {values!r}")
    if any(x < 0 for x in values):
        _reject(f"{spelling} with negative padding (cropping) is unsupported; received {tuple(values)!r}")
    return tuple(int(x) for x in values)


def zero_padding(node: Node, graph: GraphModule) -> tuple[Node, tuple[int, int, int, int]] | None:
    """A zero-padding node's input and ``(left, right, top, bottom)``; None for any other node.

    Spellings: ``nn.ZeroPad2d``, ``nn.ConstantPad2d`` with value 0, and ``F.pad`` with a
    static four-integer pad, mode ``"constant"`` and value 0 or None. Refused: other values
    and modes, reflection, replication and circular padding, negative pads, pads computed
    from a tensor's shape, ``F.pad`` tuples of another length, and a padding node with any
    other use than as the input of the one convolution directly after it."""
    if node.op == "call_module":
        module = graph.get_submodule(str(node.target))
        if isinstance(module, _OTHER_PADDING):
            _reject(f"{type(module).__name__} is unsupported: only zero padding directly ahead of a convolution",
                    "unsupported_operator")
        if not isinstance(module, nn.ConstantPad2d):
            return None
        spelling, value = type(module).__name__, module.value
        source = _bind_options(_input_arguments, node.args, node.kwargs)["input"]
        padding = _static_padding(module.padding, spelling)
    elif node.op == "call_function" and node.target is F.pad:
        spelling = "F.pad"
        options = _bind_options(_pad_arguments, node.args, node.kwargs)
        if options["mode"] != "constant":
            _reject(f"F.pad mode {options['mode']!r} is unsupported: only zero ('constant') padding "
                    "directly ahead of a convolution")
        value = 0 if options["value"] is None else options["value"]
        source = options["input"]
        padding = _static_padding(options["pad"], spelling)
    else:
        return None
    if (isinstance(value, bool) or not isinstance(value, Real) or value != 0
            or math.copysign(1.0, value) < 0):
        _reject(f"{spelling} value must be 0 (zero padding); received {value!r}")
    if not _read_by_one_convolution(node, graph):
        _reject(f"{_PADDING_MESSAGE}; this {spelling} has another use")
    _inputs((source,))
    return source, padding


def _absorbed_identity(node: Node, graph: GraphModule) -> bool:
    """An ``nn.Identity`` whose only use is as the input of one functional convolution.
    Elsewhere, and ahead of an ``nn.Conv2d``, an Identity keeps its own node."""
    return (node.op == "call_module" and isinstance(graph.get_submodule(str(node.target)), nn.Identity)
            and _read_by_one_convolution(node, graph, functional=True))


def _conv_attribute(node: Any, user: Node, graph: GraphModule):
    """A functional convolution's weight or bias: a get_attr parameter or buffer used only
    as that call's weight or bias operand."""
    message = _CONV_ATTRIBUTE_MESSAGE
    if not isinstance(node, Node) or node.op != "get_attr" or set(node.users) != {user}:
        _reject(message)
    operands = _bind_options(_conv2d_arguments, user.args, user.kwargs)
    if any(node in _nodes_in(value) for key, value in operands.items() if key not in ("weight", "bias")):
        _reject(message)
    try:
        return graph.get_parameter(str(node.target))
    except AttributeError:
        try:
            return graph.get_buffer(str(node.target))
        except AttributeError:
            _reject(message)


def conv_constant(node: Node, graph: GraphModule) -> bool:
    """Allow get_attr only as the weight or bias of the one functional convolution using it."""
    if node.op != "get_attr" or len(node.users) != 1:
        return False
    user = next(iter(node.users))
    if user.op != "call_function" or user.target is not torch.conv2d:
        return False
    try:
        operands = _bind_options(_conv2d_arguments, user.args, user.kwargs)
        if operands["weight"] is not node and operands["bias"] is not node:
            return False
        _conv_attribute(node, user, graph)
    except (TypeError, UnsupportedOperation):
        return False
    return True


def _static_pair(value, name: str) -> tuple[int, int]:
    pair = tuple(value) if isinstance(value, (tuple, list)) else (value, value)
    if len(pair) != 2 or any(isinstance(x, bool) or not isinstance(x, Integral) for x in pair):
        _reject(f"Functional conv2d {name} must be a static integer or pair of integers; received {value!r}")
    return int(pair[0]), int(pair[1])


def _stand_in(weight, bias, *, stride, dilation, groups, padding_mode="zeros") -> nn.Conv2d:
    """An ``nn.Conv2d`` with padding ``kernel // 2`` holding these weight and bias tensors
    themselves (no copy, no initialization and no RNG draw), for the blocks' unchanged checks."""
    if not isinstance(weight, torch.Tensor) or weight.dim() != 4:
        _reject("Conv2d weight must be a four-dimensional [out, in / groups, kh, kw] tensor", "coefficient_shape")
    out, per_group, kh, kw = (int(x) for x in weight.shape)
    try:
        conv = torch.nn.utils.skip_init(
            nn.Conv2d, per_group * groups, out, (kh, kw), stride=stride, padding=(kh // 2, kw // 2),
            dilation=dilation, groups=groups, bias=bias is not None, padding_mode=padding_mode)
    except (TypeError, ValueError, RuntimeError) as error:
        _reject(f"Unsupported Conv2d settings: {error}")
    conv.weight = weight if isinstance(weight, nn.Parameter) else nn.Parameter(weight, requires_grad=False)
    if bias is not None:
        conv.bias = bias if isinstance(bias, nn.Parameter) else nn.Parameter(bias, requires_grad=False)
    return conv.eval()


def convolution(node: Node, graph: GraphModule) -> Convolution | None:
    """The convolution a node calls, with the zero padding directly ahead of it; None for a
    node that calls no convolution.

    An ``nn.Conv2d`` with padding ``kernel // 2`` (or ``"same"``) and no padding node ahead
    is today's form: the module itself, its own checks and messages, phase 0. Otherwise, on
    each axis, the total zero padding before the input (the padding node's plus the
    convolution's own) must be ``kernel // 2 - phase`` with ``0 <= phase < stride``; the
    output count is checked against PyTorch's at the proven input size
    (``infer_metadata``). The blocks then run the centred convolution sampled at that phase.
    """
    if not _is_convolution(node, graph):
        return None
    operands = _convolution_operands(node)
    source = operands["input"]
    if node.op == "call_module":
        module = graph.get_submodule(str(node.target))
        path, own = str(node.target), module.padding
        kh, kw = _pair(module.kernel_size)
        padded = _quiet_zero_padding(source, graph)
        if padded is None and (own == "same" or own == (kh // 2, kw // 2)):
            (conv_options if module.groups == 1 else grouped_conv_options)(module)
            return Convolution(source, module, 0, (kw // 2, kw // 2, kh // 2, kh // 2), path)
        stand_in = _stand_in(module.weight, module.bias, stride=module.stride, dilation=module.dilation,
                             groups=module.groups, padding_mode=module.padding_mode)
    else:
        weight = _conv_attribute(operands["weight"], node, graph)
        bias = None if operands["bias"] is None else _conv_attribute(operands["bias"], node, graph)
        path = str(operands["weight"].target).rpartition(".")[0] or str(operands["weight"].target)
        own = operands["padding"]
        if own not in ("same", "valid"):
            own = _static_pair(own, "padding")
        groups = operands["groups"]
        if isinstance(groups, bool) or not isinstance(groups, Integral) or groups < 1:
            _reject(f"Functional conv2d groups must be a static positive integer; received {groups!r}")
        stand_in = _stand_in(weight, bias, stride=_static_pair(operands["stride"], "stride"),
                             dilation=_static_pair(operands["dilation"], "dilation"), groups=int(groups))
        kh, kw = _pair(stand_in.kernel_size)
        padded = _quiet_zero_padding(source, graph)
        if padded is None and isinstance(source, Node) and _absorbed_identity(source, graph):
            source = _bind_options(_input_arguments, source.args, source.kwargs)["input"]
    # The blocks' unchanged checks and messages: odd kernel, stride 1 or 2, dilation 1,
    # groups, zero padding mode, float32 and finite coefficients.
    _, stride = (conv_options if stand_in.groups == 1 else grouped_conv_options)(stand_in)
    own = (kh // 2, kw // 2) if own == "same" else (0, 0) if own == "valid" else _pair(own)
    if any(isinstance(x, bool) or not isinstance(x, Integral) or x < 0 for x in own):
        _reject(f"Conv2d padding must be non-negative integers; received {own!r}")
    if padded is not None:
        source, (left, right, top, bottom) = padded
    else:
        left = right = top = bottom = 0
    left, right, top, bottom = left + own[1], right + own[1], top + own[0], bottom + own[0]
    phase = (kh // 2 - top, kw // 2 - left)
    if not all(0 <= p < stride for p in phase):
        _reject(f"Conv2d must use same padding (kernel_size // 2), or zero padding that starts "
                f"kernel_size // 2 - phase pixels before the input on each axis with 0 <= phase < stride: "
                f"padding (left {left}, right {right}, top {top}, bottom {bottom}), the convolution's own plus "
                f"any zero padding directly ahead, gives phase {phase} at kernel {(kh, kw)} and stride {stride}")
    _inputs((source,))
    return Convolution(source, stand_in, canonical_phase(phase, stride), (left, right, top, bottom), path)


def _quiet_zero_padding(source: Any, graph: GraphModule):
    """``zero_padding`` for a convolution's input; a refused padding node is not absorbed
    (the padding node itself reports why)."""
    if not isinstance(source, Node):
        return None
    try:
        return zero_padding(source, graph)
    except (UnsupportedOperation, TypeError):
        return None


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
                return _convolution_call(node, graph)
            if zero_padding(node, graph) is not None:
                return SemanticCall("absorbed", ())
            if isinstance(module, (nn.AvgPool2d, nn.MaxPool2d)):
                return SemanticCall("pool", inputs, {"config": pool_config(module)}, module)
            if isinstance(module, nn.Upsample):
                options = dict(size=module.size, scale_factor=module.scale_factor, mode=module.mode,
                               align_corners=module.align_corners, recompute_scale_factor=module.recompute_scale_factor,
                               antialias=False)
                return _interpolation_call(inputs, options)
            if isinstance(module, nn.ReLU):
                if module.inplace is not False:
                    _reject("In-place ReLU is unsupported")
                return SemanticCall("relu", inputs)
            if isinstance(module, nn.Sigmoid):
                return SemanticCall("sigmoid", inputs)
            if isinstance(module, nn.Identity):
                if _absorbed_identity(node, graph):
                    return SemanticCall("absorbed", ())
                return SemanticCall("identity", inputs)
            if isinstance(module, nn.BatchNorm2d):
                folded = _batchnorm_fold(node, graph)
                if folded is not None:
                    predecessor, recognised, fused = folded
                    return SemanticCall("batchnorm", inputs, {
                        "source": recognised.source, "predecessor": predecessor, "phase": recognised.phase,
                        "padding": recognised.padding, "path": recognised.path}, fused)
                return SemanticCall("batch_norm", inputs, _module_batchnorm_options(module))
            if isinstance(module, nn.AdaptiveAvgPool2d):
                global_pool_size(module.output_size)
                return SemanticCall("global_mean", inputs)
            _reject(f"Unsupported module {type(module).__name__}", "unsupported_operator")
        if node.op == "call_function":
            target = node.target
            if target is torch.conv2d:
                return _convolution_call(node, graph)
            if target is F.pad:
                zero_padding(node, graph)
                return SemanticCall("absorbed", ())
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
                return _interpolation_call(_inputs((options["input"],)), options)
            if target is F.adaptive_avg_pool2d:
                options = _bind_options(_adaptive_pool_arguments, node.args, node.kwargs)
                global_pool_size(options["output_size"])
                return SemanticCall("global_mean", _inputs((options["input"],)))
            if target is torch.mean:
                options = _bind_options(_mean_arguments, node.args, node.kwargs)
                if "out" in node.kwargs:
                    _reject("out is unsupported for mean")
                global_mean_options(options)
                return SemanticCall("global_mean", _inputs((options["input"],)))
            if target in (operator.mul, torch.mul):
                options = _bind_options(_operator_mul_arguments if target is operator.mul else _mul_arguments,
                                        node.args, node.kwargs)
                if "out" in node.kwargs:
                    _reject("out is unsupported for multiplication")
                return SemanticCall("multiply", _inputs((options["input"], options["other"])))
            _reject(f"Unsupported function {target}", "unsupported_operator")
        if node.op == "call_method" and node.target == "mean":
            options = _bind_options(_method_mean_arguments, node.args, node.kwargs)
            global_mean_options(options)
            return SemanticCall("global_mean", _inputs((options["input"],)))
        if node.op == "call_method" and node.target == "mul":
            options = _bind_options(_method_mul_arguments, node.args, node.kwargs)
            return SemanticCall("multiply", _inputs((options["input"], options["other"])))
        if node.op == "call_method" and node.target in ("relu", "sigmoid", "add"):
            options = _bind_options(_add_arguments if node.target == "add" else _input_arguments, node.args, node.kwargs)
            if "out" in node.kwargs or isinstance(options.get("alpha", 1), bool) or options.get("alpha", 1) != 1:
                _reject("Residual addition requires two tensors and alpha=1 without out")
            values = (options["input"], options["other"]) if node.target == "add" else (options["input"],)
            return SemanticCall(str(node.target), _inputs(values))
        if batchnorm_constant(node, graph):
            return SemanticCall("batchnorm_constant", ())
        if conv_constant(node, graph):
            return SemanticCall("conv_constant", ())
        if node.op == "get_attr" and any(user.op == "call_function" and user.target is torch.conv2d
                                         for user in node.users):
            _reject(_CONV_ATTRIBUTE_MESSAGE)
        _reject(f"Unsupported FX operation {node.op}: {node.target}", "unsupported_operator")
    except TypeError as error:
        _reject(f"Invalid call arguments: {error}")


def _convolution_call(node: Node, graph: GraphModule) -> SemanticCall:
    recognised = convolution(node, graph)
    kind = "conv" if recognised.module.groups == 1 else "grouped_conv"
    return SemanticCall(kind, _inputs((recognised.source,)),
                        {"phase": recognised.phase, "padding": recognised.padding, "path": recognised.path},
                        recognised.module)


def infer_metadata(call: SemanticCall, inputs: tuple[TensorMetadata, ...]) -> TensorMetadata:
    first = inputs[0]
    n, c, h, w = first.shape
    if call.kind == "batch_norm" and c != call.options["channels"]:
        _reject("BatchNorm input channel count does not match raster", "input_shape")
    if call.kind in ("conv", "grouped_conv"):
        if c != call.module.in_channels:
            _reject("Conv2d input channel count does not match raster", "input_shape")
        stride = _pair(call.module.stride)[0]
        phase_y, phase_x = phase_pair(call.options.get("phase", 0), stride)
        height, width = (h - phase_y + stride - 1) // stride, (w - phase_x + stride - 1) // stride
        if "padding" in call.options:
            # PyTorch's output count must be ceil((n - phase) / stride) on each axis: a
            # padding that drops or adds an output is refused.
            left, right, top, bottom = call.options["padding"]
            kh, kw = _pair(call.module.kernel_size)
            torch_height = (h + top + bottom - kh) // stride + 1
            torch_width = (w + left + right - kw) // stride + 1
            if (torch_height, torch_width) != (height, width) or height < 1 or width < 1:
                _reject(f"Zero padding (left {left}, right {right}, top {top}, bottom {bottom}) on a {h} x {w} "
                        f"input gives PyTorch {torch_height} x {torch_width} outputs; the convolution sampled at "
                        f"phase ({phase_y}, {phase_x}) and stride {stride} gives ceil((n - phase) / stride), "
                        f"{height} x {width}", "input_shape")
        return TensorMetadata((n, call.module.out_channels, height, width), first.scale * stride)
    if call.kind == "pool":
        config = call.options["config"]
        if config.aligned and (h % config.kernel or w % config.kernel):
            _reject(aligned_size_message(config.kernel), "input_shape")
        return TensorMetadata((n, c, config.output_size(h), config.output_size(w)),
                              first.scale * config.stride)
    if call.kind in ("nearest", "bilinear"):
        factor = call.options["factor"]
        return TensorMetadata((n, c, h * factor, w * factor), first.scale / factor)
    if call.kind == "global_mean":
        # The [N, C, 1, 1] tensor on the map's one-pixel grid keeps the map's scale: the
        # shape relation stands for the extent, and compile_model checks the grids exactly.
        return TensorMetadata((n, c, 1, 1), first.scale)
    if call.kind == "multiply":
        second = inputs[1]
        if second.shape == first.shape and second.scale == first.scale:
            return first
        for map_, gate in ((first, second), (second, first)):
            if (gate.shape == (map_.shape[0], map_.shape[1], 1, 1) and gate.scale == map_.scale):
                return map_
        _reject("Multiplication requires identical shapes on one grid, or a [N, C, 1, 1] gate pooled from "
                "the [N, C, H, W] map it multiplies; scalars, channel and other broadcasts are unsupported",
                "input_shape")
    if call.kind in ("cat", "add"):
        for other in inputs[1:]:
            if other.shape[0] != n or other.shape[2:] != (h, w) or other.scale != first.scale:
                _reject("Merge tensors must have identical spatial grids", "input_shape")
            if call.kind == "add" and other.shape[1] != c:
                _reject("Residual addition requires identical spatial grids and channel counts", "input_shape")
        if call.kind == "cat":
            return TensorMetadata((n, sum(x.shape[1] for x in inputs), h, w), first.scale)
    return first
