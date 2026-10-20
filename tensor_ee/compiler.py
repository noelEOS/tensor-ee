"""Compile a deliberately small PyTorch inference graph to native EE images.

No initialization or computation requests happen here. Call ``ee.Initialize``
yourself, then ``compile_model(model.eval(), image, bands, grid)``. The result
contains the output image, ordered band names, output grid, and ``nodes`` for
inspecting individual FX tensors. Every node describes one NCHW tensor with
batch size one; channels are EE bands and spatial dimensions are the Grid.

Semantics are intentionally strict: finite rectangles, missing input values
filled with zero, odd same-padded stride-one/two convolutions, aligned k x k
pooling (k = 2, 3, 4), padded 3x3 max pooling (stride one or two) and stride-one
average pooling, integer nearest upsampling, residual addition, and channel
concatenation. Outside every tensor's rectangle is zero for the
next convolution, including after a biased layer. Intermediate grids retain
the original upper-left corner; strided operations sample PyTorch's correct
pixel centers and then relabel the resulting tensor's coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from numbers import Integral
from typing import Any, Callable, Sequence

import ee
import torch
from torch import nn
from torch.nn import functional as F
from torch.fx import GraphModule, Node, symbolic_trace

from .core.errors import UnsupportedOperation
from .core.grid import Grid
from .core.tensor import Raster
# R converters (layout and grid consolidation) and float32 helpers moved to the library; the old
# names still in use are kept here as re-exports. compile_model looks _finish up here.
from .layouts.raster import finish as _finish
from .core.precision import exact_scalar as _ee_scalar
from .weights.constants import inline_float32_array as _inline_float32_array
# R blocks moved to the library; these names are re-exports of the block
# objects. The blocks look their helpers up in their own modules, so tests patch
# tensor_ee.blocks.raster.conv._inline_float32_array.
from .blocks.raster.conv import _names, _pair, conv2d
from .blocks.raster.pooling import pool2d
from .blocks.raster.resample import nearest_upsample
from .blocks.raster.merge import concatenate, residual_add
# The activations were private here; the library names them without the
# underscore. Tests patch tensor_ee.blocks.raster.activations._finish and
# ._ee_scalar, where these functions look the names up.
from .blocks.raster.activations import (_FLOAT32_MAX, _dropout_probability, _exact_gelu,
                                        _real_option)
from .blocks.raster.activations import (common_score_mask as _common_score_mask,
                                        eval_dropout as _eval_dropout, gelu as _gelu,
                                        leaky_relu as _leaky_relu, relu as _relu,
                                        sigmoid as _sigmoid, silu as _silu,
                                        softmax as _softmax, tanh as _tanh)


WeightProvider = Callable[[str, nn.Conv2d], tuple[ee.Array, ee.Array | None]]


@dataclass(frozen=True)
class CompiledModel(Raster):
    nodes: dict[str, Raster] = field(default_factory=dict)
    graph_module: GraphModule | None = None
    folded_batchnorms: tuple[str, ...] = ()


_POINTWISE_MODULES = (nn.LeakyReLU, nn.SiLU, nn.GELU, nn.Tanh, nn.Dropout, nn.Dropout2d, nn.Softmax)
_UNSUPPORTED_DROPOUT_MODULES = (nn.Dropout1d, nn.Dropout3d, nn.AlphaDropout, nn.FeatureAlphaDropout)
_ROOT_MODULES = (nn.Conv2d, nn.ReLU, nn.AvgPool2d, nn.MaxPool2d,
                 nn.Upsample, nn.Sigmoid, nn.Identity, *_POINTWISE_MODULES,
                 *_UNSUPPORTED_DROPOUT_MODULES)


def _not_inplace(value: Any) -> None:
    if value is not False:
        raise UnsupportedOperation(f"inplace must be the literal False; received {value!r}")


def _leaky_arguments(input, negative_slope=0.01, inplace=False):
    """Signature only: bind the supported PyTorch functional call locally."""


def _silu_arguments(input, inplace=False):
    """Signature only."""


def _gelu_arguments(input, *, approximate="none"):
    """Signature only: approximate is keyword-only in F.gelu."""


def _dropout_arguments(input, p=0.5, training=True, inplace=False):
    """Signature only: functional dropout defaults to training=True."""


def _torch_tanh_arguments(input, *, out=None):
    """Signature only: out is valid in PyTorch but unsupported here."""


def _functional_softmax_arguments(input, dim=None, _stacklevel=3, dtype=None):
    """Signature only: F.softmax's third positional argument is _stacklevel."""


def _torch_softmax_arguments(input, dim, dtype=None, *, out=None):
    """Signature only: torch.softmax accepts positional dtype."""


def _method_softmax_arguments(input, dim, dtype=None):
    """Signature only: Tensor.softmax accepts positional dtype."""


def _input_arguments(input):
    """Signature only: bind a one-input module/method call locally."""


def _bind_options(signature, args, kwargs):
    bound = inspect.signature(signature).bind(*args, **kwargs)
    bound.apply_defaults()
    return dict(bound.arguments)


def _pointwise_call(node: Node, graph: GraphModule) -> tuple[str, Node, dict[str, Any]] | None:
    """Validate new pointwise semantics without constructing or evaluating EE.

    Compiler, support analysis and weight extraction share this rule. Options
    are bound from the raw FX call, so tensor/dynamic options cannot accidentally
    become accepted after resolving the compiler's tensor environment.
    """
    if node.op == "call_module":
        module = graph.get_submodule(str(node.target))
        if isinstance(module, _UNSUPPORTED_DROPOUT_MODULES):
            raise UnsupportedOperation(f"Unsupported dropout family {type(module).__name__}")
        if not isinstance(module, _POINTWISE_MODULES):
            return None
        options = _bind_options(_input_arguments, node.args, node.kwargs)
        if isinstance(module, nn.LeakyReLU):
            kind = "leaky_relu"
            options.update(negative_slope=module.negative_slope, inplace=module.inplace)
        elif isinstance(module, nn.SiLU):
            kind = "silu"
            options.update(inplace=module.inplace)
        elif isinstance(module, nn.GELU):
            kind = "gelu"
            options.update(approximate=module.approximate)
        elif isinstance(module, nn.Tanh):
            kind = "tanh"
        elif isinstance(module, nn.Softmax):
            kind = "softmax"
            options.update(dim=module.dim, dtype=None)
        else:
            kind = "eval_dropout"
            options.update(p=module.p, training=module.training, inplace=module.inplace)
    elif node.op == "call_function":
        if node.target in (F.dropout1d, F.dropout3d, F.alpha_dropout, F.feature_alpha_dropout):
            raise UnsupportedOperation(f"Unsupported dropout family {node.target}")
        calls = {F.leaky_relu: ("leaky_relu", _leaky_arguments),
                 F.silu: ("silu", _silu_arguments), F.gelu: ("gelu", _gelu_arguments),
                 torch.tanh: ("tanh", _torch_tanh_arguments), F.tanh: ("tanh", _input_arguments),
                 F.dropout: ("eval_dropout", _dropout_arguments),
                 F.dropout2d: ("eval_dropout", _dropout_arguments),
                 F.softmax: ("softmax", _functional_softmax_arguments),
                 torch.softmax: ("softmax", _torch_softmax_arguments)}
        if node.target not in calls:
            return None
        kind, signature = calls[node.target]
        options = _bind_options(signature, node.args, node.kwargs)
        if node.target in (torch.tanh, torch.softmax):
            if "out" in node.kwargs:
                raise UnsupportedOperation(f"out is unsupported; received {node.kwargs['out']!r}")
            options.pop("out")
    elif node.op == "call_method" and node.target in ("tanh", "softmax"):
        kind = str(node.target)
        signature = _input_arguments if kind == "tanh" else _method_softmax_arguments
        options = _bind_options(signature, node.args, node.kwargs)
    else:
        return None
    if "inplace" in options:
        _not_inplace(options.pop("inplace"))
    if kind == "leaky_relu":
        options["negative_slope"] = _real_option(options["negative_slope"], "negative_slope", float32=True)
    elif kind == "gelu":
        _exact_gelu(options["approximate"])
    elif kind == "softmax":
        dimension = options.pop("dim")
        if isinstance(dimension, bool) or not isinstance(dimension, Integral) or dimension not in (1, -3):
            raise UnsupportedOperation(f"softmax dim must be the static NCHW channel axis 1 or -3; received {dimension!r}")
        dtype = options.pop("dtype")
        if dtype is not None:
            raise UnsupportedOperation(f"softmax dtype must be None; received {dtype!r}")
        if "_stacklevel" in options:
            level = options.pop("_stacklevel")
            if isinstance(level, bool) or not isinstance(level, Integral) or level != 3:
                raise UnsupportedOperation(f"softmax _stacklevel must be its static default 3; received {level!r}")
    elif kind == "eval_dropout":
        if options.pop("training") is not False:
            raise UnsupportedOperation("training must be the literal False for evaluation dropout")
        options["p"] = _dropout_probability(options["p"])
    return kind, options.pop("input"), options


def _apply_pointwise(kind: str, raster: Raster, options: dict[str, Any],
                     *, clip_footprints: bool = False) -> Raster:
    operations = {"leaky_relu": _leaky_relu, "silu": _silu, "gelu": _gelu,
                  "tanh": _tanh, "eval_dropout": _eval_dropout, "softmax": _softmax}
    return operations[kind](raster, **options, clip_footprints=clip_footprints)


def _resolve(value: Any, environment: dict[Node, Any]) -> Any:
    if isinstance(value, Node):
        return environment[value]
    if isinstance(value, (tuple, list)):
        return type(value)(_resolve(item, environment) for item in value)
    if isinstance(value, dict):
        return {key: _resolve(item, environment) for key, item in value.items()}
    return value


def compile_model(model: nn.Module, image: ee.Image, input_bands: Sequence[str], grid: Grid,
                  *, weight_provider: WeightProvider | None = None,
                  include_unfused_nodes: bool = True,
                  execution_dtype: str = "float32",
                  clip_footprints: bool = False) -> CompiledModel:
    """Convert an eval-mode model into an EE graph; do not evaluate or export it.

    Supports Conv2d, ReLU, AvgPool2d, MaxPool2d, nearest Upsample,
    Sigmoid, Identity, LeakyReLU, SiLU, exact GELU, Tanh, eval Dropout/Dropout2d, channel Softmax,
    eval BatchNorm2d directly following Conv2d, torch.cat,
    residual tensor addition, torch.relu, F.relu, torch.sigmoid, and F.interpolate with a static integer
    scale factor, or a proven size with that same integer nearest mapping.
    Unsupported operators fail explicitly with their FX node.
    ``nodes`` includes the input placeholder and each tensor operation, keyed
    by the names in ``graph_module.graph`` for layer-by-layer parity tests.

    ``weight_provider(node_key, effective_conv)`` can return server-side EE
    arrays instead of embedding parameters in requests. A convolution followed
    only by BatchNorm requests folded weights under the BatchNorm FX node key.
    Set ``include_unfused_nodes=False`` to avoid constructing its diagnostic
    unfused convolution; this keeps large unused parameters out of client RAM.
    Float32 is the default and maximum supported model precision. Matrix
    operands and every FX activation are cast to float32 inside EE. Float64
    execution and non-float32 model parameters are rejected explicitly.
    ``clip_footprints=True`` additionally clips finite tensor extents before
    layer reprojections. It is an optional EE execution/memory experiment;
    convolution padding continues to zero-extend the preceding finite tensor.
    """
    if any(module.training for module in model.modules()):
        # Explain dropout's effective flag at its FX node even when p=0. Other
        # modules retain the existing model-wide inference requirement.
        if any(module.training and isinstance(module, (nn.Dropout, nn.Dropout2d))
               for module in model.modules()):
            wrapper = nn.Sequential(model) if isinstance(model, _ROOT_MODULES) else model
            try:
                training_graph = symbolic_trace(wrapper)
            except Exception as error:
                # Diagnostic tracing must not replace the established eval
                # rejection for an untraceable training model.
                raise UnsupportedOperation("Set model.eval() before compilation") from error
            for training_node in training_graph.graph.nodes:
                try:
                    _pointwise_call(training_node, training_graph)
                except UnsupportedOperation as error:
                    raise UnsupportedOperation(
                        f"At FX node {training_node.name} ({training_node.target}): {error}") from error
        raise UnsupportedOperation("Set model.eval() before compilation")
    if execution_dtype != "float32":
        raise ValueError("execution_dtype must be float32 (maximum supported precision)")
    if any(value.is_floating_point() and value.dtype != torch.float32
           for value in (*model.parameters(), *model.buffers())):
        raise UnsupportedOperation("Model parameters and buffers must use float32; convert with model.float()")
    if not input_bands or len(set(input_bands)) != len(input_bands):
        raise ValueError("input_bands must be a nonempty sequence of unique names")
    fixed_input = getattr(model, "tensor_ee_input_shape", None)
    if fixed_input is not None and tuple(fixed_input) != (len(input_bands), grid.height, grid.width):
        raise UnsupportedOperation(
            f"Imported model requires fixed CHW input {tuple(fixed_input)}, received "
            f"{(len(input_bands), grid.height, grid.width)}; spatial generalization needs separate validation")
    if isinstance(model, _ROOT_MODULES):
        model = nn.Sequential(model)
    traced = model if isinstance(model, GraphModule) else symbolic_trace(model)
    from .semantics import specialize_plumbing
    plumbing = specialize_plumbing(traced, (1, len(input_bands), grid.height, grid.width))
    traced = plumbing.graph
    environment: dict[Node, Any] = {}
    nodes: dict[str, Raster] = {}
    folded: list[str] = []
    placeholders = 0
    output: Raster | None = None

    def convolve(key: str, raster: Raster, module: nn.Conv2d) -> Raster:
        parameters = None if weight_provider is None else weight_provider(key, module)
        return conv2d(raster, module, parameters, execution_dtype=execution_dtype,
                      clip_footprints=clip_footprints)

    from .semantics import (TensorMetadata, diagnostic_convolution, infer_metadata, semantic_call)
    metadata: dict[Node, TensorMetadata] = {}
    for node in traced.graph.nodes:
        try:
            if node in plumbing.errors:
                raise UnsupportedOperation(str(plumbing.errors[node])) from plumbing.errors[node]
            if node.op == "placeholder":
                placeholders += 1
                if placeholders > 1:
                    raise UnsupportedOperation("Only one tensor input is supported")
                metadata[node] = TensorMetadata((1, len(input_bands), grid.height, grid.width))
                raster = _finish(image.select(list(input_bands)).unmask(0, sameFootprint=False),
                                 _names(len(input_bands)), grid, clip_footprints)
            elif node.op == "output":
                output = _resolve(node.args[0], environment)
                if not isinstance(output, Raster):
                    raise UnsupportedOperation("Model must return one tensor")
                continue
            else:
                call = semantic_call(node, traced)
                metadata[node] = infer_metadata(call, tuple(metadata[x] for x in call.inputs))
                kind = call.kind
                if kind == "conv":
                    if not include_unfused_nodes and diagnostic_convolution(node, traced):
                        environment[node] = None
                        continue
                    raster = convolve(node.name, environment[call.inputs[0]], call.module)
                elif kind == "batchnorm":
                    raster = convolve(node.name, environment[call.options["source"]], call.module)
                    folded.append(node.name)
                else:
                    inputs = tuple(environment[x] for x in call.inputs)
                    if kind == "pool":
                        raster = pool2d(inputs[0], call.module, clip_footprints=clip_footprints)
                    elif kind == "nearest":
                        raster = nearest_upsample(inputs[0], call.options["factor"], clip_footprints=clip_footprints)
                    elif kind == "cat":
                        raster = concatenate(inputs, call.options["dim"], clip_footprints=clip_footprints)
                    elif kind == "add":
                        raster = residual_add(*inputs, clip_footprints=clip_footprints)
                    elif kind == "relu":
                        raster = _relu(inputs[0], clip_footprints=clip_footprints)
                    elif kind == "sigmoid":
                        raster = _sigmoid(inputs[0], clip_footprints=clip_footprints)
                    elif kind == "identity":
                        raster = inputs[0]
                    else:
                        raster = _apply_pointwise(kind, inputs[0], call.options,
                                                  clip_footprints=clip_footprints)
            if not isinstance(raster, Raster):
                raise UnsupportedOperation("Operation did not produce a raster tensor")
            if execution_dtype == "float32":
                # Reduction and transcendental EE operations can promote
                # results. Explicitly bound every next layer's input type.
                raster = Raster(raster.image.toFloat(), raster.bands, raster.grid)
            environment[node] = raster
            nodes[node.name] = raster
        except UnsupportedOperation as error:
            raise UnsupportedOperation(f"At FX node {node.name} ({node.target}): {error}") from error
    if output is None:
        raise UnsupportedOperation("Model has no tensor output")
    return CompiledModel(output.image, output.bands, output.grid, nodes, traced, tuple(folded))
