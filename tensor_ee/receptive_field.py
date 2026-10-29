"""Exact spatial dependency envelopes for the supported inference graphs.

This is a structural calculation, independent of parameter values and observed
numerical sensitivity. Bounds are inclusive input pixel indices. They describe
the union of all possible spatial paths, including residual and skip branches.
Nearest upsampling is treated as ``floor(index / scale)`` rather than as an
approximate receptive-field width, so bounds retain their sampling phase.
Bilinear upsampling adds the parent's one-pixel ring. With align_corners=True
the report marks extent dependence and refuses a halo guarantee.

For a full-resolution output, ``safe_halo`` is rounded to ``alignment`` and is
guaranteed for cores whose start and exclusive end align to that lattice. Tile
contexts must use the same aligned origin and dimensions as the full reference.
Clamping a context to a real scene boundary is valid when the global dimensions
also align: the two evaluations then have the same finite-tensor padding there.
An arbitrary phase core can instead use ``halo_for_any_phase`` or exact bounds.
Unsupported operations fail explicitly; in particular training BatchNorm and
normalization over spatial dimensions cannot be treated as pointwise operators.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
import operator
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.fx import GraphModule, Node, symbolic_trace

from .compiler import UnsupportedOperation, _ROOT_MODULES


class UnsupportedReceptiveField(ValueError):
    """An operation has no supported finite spatial dependency rule."""


@dataclass(frozen=True)
class AxisSupport:
    """A periodic, integer-valued dependency envelope on one spatial axis.

    Shifting an output index by ``period`` shifts its two bounds by
    ``input_shift``. Thus ``jump`` is the output lattice spacing in input pixels.
    The table stores bounds for output indices ``0 .. period - 1`` and also
    works for negative indices via Python's floor division.
    """

    period: int
    input_shift: int
    phase_bounds: tuple[tuple[int, int], ...]

    @property
    def jump(self) -> Fraction:
        return Fraction(self.input_shift, self.period)

    def bounds(self, index: int) -> tuple[int, int]:
        if not isinstance(index, int):
            raise TypeError("Output indices must be integers")
        quotient, phase = divmod(index, self.period)
        lower, upper = self.phase_bounds[phase]
        shift = quotient * self.input_shift
        return lower + shift, upper + shift

    @property
    def maximum_support_width(self) -> int:
        return max(upper - lower + 1 for lower, upper in self.phase_bounds)

    @property
    def left_radius(self) -> int:
        self._require_full_resolution()
        return max(index - lower for index, (lower, _) in enumerate(self.phase_bounds))

    @property
    def right_radius(self) -> int:
        self._require_full_resolution()
        return max(upper - index for index, (_, upper) in enumerate(self.phase_bounds))

    def _require_full_resolution(self) -> None:
        if self.jump != 1:
            raise UnsupportedReceptiveField("Halo requires a full-resolution output lattice")

    def as_dict(self) -> dict[str, Any]:
        return {
            "period": self.period,
            "input_shift": self.input_shift,
            "jump": [self.jump.numerator, self.jump.denominator],
            "phase_bounds": [list(bounds) for bounds in self.phase_bounds],
            "maximum_support_width": self.maximum_support_width,
        }


@dataclass(frozen=True)
class SpatialSupport:
    y: AxisSupport
    x: AxisSupport

    def input_bounds(self, x: int, y: int) -> tuple[int, int, int, int]:
        """Return inclusive ``(xmin, ymin, xmax, ymax)`` input bounds."""
        xmin, xmax = self.x.bounds(x)
        ymin, ymax = self.y.bounds(y)
        return xmin, ymin, xmax, ymax

    def as_dict(self) -> dict[str, Any]:
        return {"x": self.x.as_dict(), "y": self.y.as_dict()}


@dataclass(frozen=True)
class ReceptiveFieldReport(SpatialSupport):
    """Model output support and conservative tile requirements in input pixels."""

    alignment: int
    nodes: dict[str, SpatialSupport]
    chip_edge_dependent: bool = False

    def _require_tileable(self):
        if self.chip_edge_dependent:
            raise UnsupportedReceptiveField(
                "Bilinear align_corners=True depends on the full source extent; halos cannot reproduce it")

    @property
    def period(self) -> int:
        return math.lcm(self.x.period, self.y.period)

    @property
    def safe_halo(self) -> int:
        """Symmetric halo for lattice-aligned core start and exclusive end."""
        self._require_tileable()
        self.x._require_full_resolution()
        self.y._require_full_resolution()
        # Monotone bounds attain their extremes at the first/last core pixels.
        # A core ending at a period multiple has final phase period - 1.
        extent = max(-self.x.bounds(0)[0], -self.y.bounds(0)[0],
                     self.x.bounds(self.x.period - 1)[1] - (self.x.period - 1),
                     self.y.bounds(self.y.period - 1)[1] - (self.y.period - 1), 0)
        return _round_up(extent, self.alignment)

    @property
    def halo_for_any_phase(self) -> int:
        """Uniform conservative halo when core start/end phase is unrestricted."""
        self._require_tileable()
        extent = max(self.x.left_radius, self.x.right_radius,
                     self.y.left_radius, self.y.right_radius, 0)
        return _round_up(extent, self.alignment)

    def as_dict(self, *, include_nodes: bool = False) -> dict[str, Any]:
        result = super().as_dict()
        result.update({
            "alignment": self.alignment,
            "period": self.period,
            "safe_halo": None if self.chip_edge_dependent else self.safe_halo,
            "halo_for_any_phase": None if self.chip_edge_dependent else self.halo_for_any_phase,
            "aligned_core_requirement": "start and exclusive end divisible by alignment",
            "context_requirement": "origin and dimensions divisible by alignment",
            "bounds_convention": "inclusive input pixel indices; structural dependency envelope",
        })
        if self.chip_edge_dependent:
            result["chip_edge_dependent"] = True
            result["context_requirement"] = "full source extent required"
        if include_nodes:
            result["nodes"] = {name: support.as_dict() for name, support in self.nodes.items()}
        return result


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _pair(value: Any, *, name: str) -> tuple[int, int]:
    pair = value if isinstance(value, (tuple, list)) else (value, value)
    if (len(pair) != 2 or any(not isinstance(v, (int, float)) or
                              not math.isfinite(v) or int(v) != v for v in pair)):
        raise UnsupportedReceptiveField(f"{name} must be a scalar or pair of integers")
    return int(pair[0]), int(pair[1])


def _window_axis(source: AxisSupport, kernel: int, stride: int,
                 padding: int) -> AxisSupport:
    period = source.period // math.gcd(source.period, stride)
    shift = source.input_shift * (stride * period // source.period)
    bounds = tuple((source.bounds(stride * index - padding)[0],
                    source.bounds(stride * index - padding + kernel - 1)[1])
                   for index in range(period))
    return AxisSupport(period, shift, bounds)


def _upsample_axis(source: AxisSupport, scale: int) -> AxisSupport:
    period = source.period * scale
    return AxisSupport(period, source.input_shift,
                       tuple(source.bounds(index // scale) for index in range(period)))


def _union_axis(sources: list[AxisSupport]) -> AxisSupport:
    if any(source.jump != sources[0].jump for source in sources):
        raise UnsupportedReceptiveField("Merged branches must use the same spatial lattice")
    period = math.lcm(*(source.period for source in sources))
    shift = sources[0].input_shift * (period // sources[0].period)
    bounds = tuple((min(source.bounds(index)[0] for source in sources),
                    max(source.bounds(index)[1] for source in sources))
                   for index in range(period))
    return AxisSupport(period, shift, bounds)


def analyze_receptive_field(model: nn.Module | GraphModule) -> ReceptiveFieldReport:
    """Trace supported inference operations without executing any tensor forward.

    Supports the compiler's odd same-padded convolutions, its pools (aligned
    k x k average/max for k in 2, 3, 4; padded 3x3 max at stride 1 or 2; padded
    3x3 stride-1 average), fixed integer nearest/bilinear upsampling, inference BatchNorm,
    ReLU/Sigmoid/Identity, residual addition, and channel concatenation. Pools
    and convolutions share one window rule (kernel, stride, padding), so an
    aligned k x k pool sets the lattice alignment k and a stride-1 padded pool
    adds a one-pixel radius.
    Channel widths and weight values do not change the spatial guarantee.
    """
    primitives = _ROOT_MODULES
    if isinstance(model, primitives):
        model = nn.Sequential(model)
    graph = model if isinstance(model, GraphModule) else symbolic_trace(model)
    supports: dict[Node, SpatialSupport] = {}
    nodes: dict[str, SpatialSupport] = {}
    alignment = 1
    placeholders = 0
    output = None
    chip_edge_dependent = False

    def source(node: Any) -> SpatialSupport:
        if not isinstance(node, Node) or node not in supports:
            raise UnsupportedReceptiveField("Expected a previously supported tensor node")
        return supports[node]

    from .semantics import semantic_call
    for node in graph.graph.nodes:
        try:
            if node.op == "placeholder":
                placeholders += 1
                if placeholders != 1:
                    raise UnsupportedReceptiveField("Exactly one image tensor input is supported")
                axis = AxisSupport(1, 1, ((0, 0),))
                support = SpatialSupport(axis, axis)
            elif node.op == "output":
                output = source(node.args[0])
                continue
            else:
                call = semantic_call(node, graph)
                if call.kind == "batchnorm_constant":
                    continue
                support = source(call.inputs[0])
                if call.kind in ("conv", "grouped_conv", "pool"):
                    module = call.module
                    kernel = _pair(module.kernel_size, name="Window kernel")
                    stride = _pair(module.stride or module.kernel_size, name="Window stride")
                    padding = ((kernel[0] // 2, kernel[1] // 2) if module.padding == "same"
                               else _pair(module.padding, name="Window padding"))
                    support = SpatialSupport(_window_axis(support.y, kernel[0], stride[0], padding[0]),
                                             _window_axis(support.x, kernel[1], stride[1], padding[1]))
                elif call.kind == "nearest":
                    scale = call.options["factor"]
                    support = SpatialSupport(_upsample_axis(support.y, scale), _upsample_axis(support.x, scale))
                elif call.kind == "bilinear":
                    scale = call.options["factor"]
                    support = SpatialSupport(_upsample_axis(_window_axis(support.y, 3, 1, 1), scale),
                                             _upsample_axis(_window_axis(support.x, 3, 1, 1), scale))
                    chip_edge_dependent |= call.options["align_corners"]
                elif call.kind in ("cat", "add"):
                    branches = [source(item) for item in call.inputs]
                    support = SpatialSupport(_union_axis([branch.y for branch in branches]),
                                             _union_axis([branch.x for branch in branches]))
        except UnsupportedOperation as error:
            raise UnsupportedReceptiveField(f"At FX node {node.name} ({node.target}): {error}") from error
        supports[node] = support
        nodes[node.name] = support
        alignment = math.lcm(alignment, support.y.input_shift, support.x.input_shift)
    if output is None:
        raise UnsupportedReceptiveField("Graph has no single supported image tensor output")
    return ReceptiveFieldReport(output.y, output.x, alignment, nodes, chip_edge_dependent)
