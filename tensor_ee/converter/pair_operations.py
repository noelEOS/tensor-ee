"""Registry adapters for atomic (spatial, prefix) edges; no new block arithmetic.

One parameter binding serves both components. Pointwise adapters call the same
library function on each component with constants built for its own row count.
The position adapter follows Prithvi's constant preparation, including the class
addition before the spatial forward. These entries add no numerical evidence.
"""
from __future__ import annotations

import math
import operator


from ..blocks.window import activations, linear, merge, normalization
from ..core.errors import GridMismatch, reject
from ..core.tensor import ServerArray, WindowTensor
from ..layouts.boundaries import pin_boundary
from ..weights.constants import WindowLayout, row_constant


def _pair(inputs):
    x = inputs[0]
    if x.layout != "P":
        raise GridMismatch("this operation reads an atomic spatial/prefix pair")
    return x


def _description(x, channels=None):
    return dict(layout="P", grid=x.grid, channels=x.channels if channels is None else channels,
                window=x.window, bands=None)


def _positive(s, key):
    value = s[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        reject(f"{key} must be a positive integer")
    return value


def _shape(value, shape, name):
    if tuple(value.shape) != shape:
        reject(f"{name} must have shape {shape}; received {value.shape}")
    return value


def _reshape(value, shape):
    if isinstance(value, ServerArray):
        return ServerArray(value.array.reshape(list(shape)), shape)
    return value.reshape(shape)


def _position_infer(s, inputs):
    (x,) = inputs
    if x.layout != "W" or x.window != 14 or (x.grid.height, x.grid.width) != (14, 14):
        raise GridMismatch("class_token_position reads exactly one W14 spatial window")
    return _description(x)


def _position_constants(ops, parameters, prefix, s, inputs):
    x = inputs[0]
    dim = x.channels
    position = _shape(parameters[f"{prefix}.pos_embed"], (1, 197, dim), "pos_embed")
    token = _shape(parameters[f"{prefix}.cls_token"], (1, 1, dim), "cls_token")
    if isinstance(position, ServerArray):
        spatial = ServerArray(position.array.slice(1, 1, 197).reshape([196, dim]), (196, dim))
        first = ServerArray(position.array.slice(1, 0, 1).reshape([1, dim]), (1, dim))
    else:
        spatial, first = position[0, 1:], position[0, :1]
    spatial = ops.constant(spatial)
    cls = WindowTensor(ops.constant(_reshape(token, (1, dim))), x.grid.coarse(14), dim, 1)
    # The library owns the arithmetic and its one rounding, as in Prithvi.
    positioned_class = merge.add(cls, cls.like(ops.constant(first))).image
    return spatial, positioned_class


def _position_call(ops, tensors, constants, s):
    (spatial,) = tensors
    position, cls = constants
    spatial = merge.add(spatial, spatial.like(position))
    grid = spatial.grid.coarse(14)
    cls = pin_boundary(ops.ee, cls, grid)
    return spatial, WindowTensor(cls, grid, spatial.channels, 1)


def _select_infer(s, inputs):
    pair = _pair(inputs)
    if s["component"] not in ("spatial", "prefix"):
        reject("pair_select component is spatial or prefix")
    edge = getattr(pair, s["component"])
    return dict(layout=edge.layout, grid=edge.grid, channels=edge.channels, window=edge.window, bands=edge.bands)


def _linear_infer(s, inputs):
    x = _pair(inputs)
    if _positive(s, "in_features") != x.channels:
        raise GridMismatch("pair_linear input width differs from its edge")
    if not isinstance(s["bias"], bool):
        reject("bias must be a bool")
    return _description(x, _positive(s, "out_features"))


def _linear_constants(ops, p, prefix, s, inputs):
    weight = _shape(p[f"{prefix}.weight"], (s["out_features"], s["in_features"]), "weight")
    bias = _shape(p[f"{prefix}.bias"], (s["out_features"],), "bias") if s["bias"] else None
    return tuple(linear.LinearConstants(ops, weight, bias, WindowLayout(w)) for w in (14, 1))


def _linear_call(ops, tensors, constants, s):
    return tuple(x.like(linear.linear(ops, x.image, c, layout=x), s["out_features"])
                 for x, c in zip(tensors[0], constants))


def _linear_cost(s, inputs):
    metadata = linear.METADATA["linear"]
    # The weight is shared; the bias is repeated for 196 rows and for one row.
    values = sum(metadata.embedded_constants(s["in_features"], s["out_features"],
                 bias=s["bias"], layout=WindowLayout(w)) for w in (14, 1))
    return values - s["in_features"] * s["out_features"], metadata.asset_arrays(bias=s["bias"])


def _norm_infer(s, inputs):
    x = _pair(inputs)
    eps = s["eps"]
    if isinstance(eps, bool) or not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
        reject("eps must be positive and finite")
    return _description(x)


def _norm_constants(ops, p, prefix, s, inputs):
    dim = inputs[0].channels
    weight = _shape(p[f"{prefix}.weight"], (dim,), "weight")
    bias = _shape(p[f"{prefix}.bias"], (dim,), "bias")
    return tuple(tuple(row_constant(ops, value, WindowLayout(w)) for value in (weight, bias)) for w in (14, 1))


def _norm_call(ops, tensors, constants, s):
    return tuple(x.like(normalization.layer_norm(ops, x.image, *c, dim=x.channels, eps=s["eps"], layout=x))
                 for x, c in zip(tensors[0], constants))


def _add_infer(s, inputs):
    x = _pair(inputs)
    if any(other != x for other in inputs):
        raise GridMismatch("pair_add requires identical pair descriptions")
    return _description(x)


def pair_operations(Operation):
    """Explicit registry entries, loaded by operations.py (avoids a graph import cycle)."""
    no_roles = lambda s: ()
    no_cost = lambda s, inputs: (0, 0)
    return {
        "class_token_position": Operation(
            function=merge.add, metadata=merge.METADATA["add"], inputs=1, settings=(),
            roles=lambda s: ("cls_token", "pos_embed"), infer=_position_infer,
            cost=lambda s, inputs: (198 * inputs[0].channels, 2),
            constants=_position_constants, call=_position_call),
        "pair_select": Operation(
            function=operator.getitem, metadata=None, inputs=1, settings=("component",), roles=no_roles,
            infer=_select_infer, cost=no_cost,
            call=lambda ops, tensors, c, s: tensors[0][0 if s["component"] == "spatial" else 1]),
        "pair_linear": Operation(
            function=linear.linear, metadata=linear.METADATA["linear"], inputs=1,
            settings=("in_features", "out_features", "bias"),
            roles=lambda s: ("weight", "bias") if s["bias"] else ("weight",), infer=_linear_infer,
            cost=_linear_cost, constants=_linear_constants, call=_linear_call),
        "pair_layer_norm": Operation(
            function=normalization.layer_norm, metadata=normalization.METADATA["layer_norm"], inputs=1,
            settings=("eps",), roles=lambda s: ("weight", "bias"), infer=_norm_infer,
            cost=lambda s, inputs: (sum(normalization.METADATA["layer_norm"].embedded_constants(
                inputs[0].channels, layout=WindowLayout(w)) for w in (14, 1)),
                normalization.METADATA["layer_norm"].asset_arrays()),
            constants=_norm_constants, call=_norm_call),
        "pair_gelu": Operation(
            function=activations.gelu, metadata=activations.METADATA["gelu"], inputs=1,
            settings=(), roles=no_roles, infer=lambda s, inputs: _description(_pair(inputs)), cost=no_cost,
            call=lambda ops, tensors, c, s: tuple(x.like(activations.gelu(ops, x.image)) for x in tensors[0])),
        "pair_add": Operation(
            function=merge.add, metadata=merge.METADATA["add"], inputs=2, settings=(), roles=no_roles,
            infer=_add_infer, cost=no_cost,
            call=lambda ops, tensors, c, s: tuple(merge.add(a, b) for a, b in zip(*tensors))),
    }
