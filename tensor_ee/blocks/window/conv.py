"""The window-layout 3 x 3 convolution and its input-channel groups.

Moved from ``upernet/earth_engine/conv2d.py`` (``Conv3x3Constants``, ``conv3x3_window``
and its helper ``_finish``) and ``upernet/earth_engine/fusion.py`` (``group_weight``,
``group_product``), research code archived at the tag ``research-archive-2026-10-12``
(docs/history/README.md). The dense formulations (``dense_im2col``, ``conv3x3_dense``,
``conv_convolve``) were research only; the fusion composites are with the model.

Inference Conv2d (stride 1, dilation 1, groups 1, zero padding k // 2), PyTorch
semantics (weight [Cout, Cin, kh, kw], padding p = k // 2):

    y[o, i, j] = b[o] + sum_{c, ky, kx} W[o, c, ky, kx] * x_pad[c, i + ky, j + kx]

i.e. an im2col matrix product: the zero-padded k x k neighbourhood of each token as a
row in (ky, kx, c) order times W^T [k*k*Cin, Cout] in the same row order (the weight
permutation (o, c, ky, kx) -> (ky, kx, c, o) is applied once, to the constant). In
window layout the neighbourhood is the [w + 2, w + 2, C] halo of each window (its
one-token ring from the 8 neighbouring windows, zeros outside the raster):
``tensor_ee.layouts.window.halo`` and ``im2col``.

**Policy** ``double_round_once``: EE's matrixMultiply accumulates every dot product
sequentially in double; the bias is added in double and the sum rounded to float32
once. ``group_product`` leaves its partial in double (no rounding): the fusion's
``grouped_double`` variant sums the four group partials in double before its one
rounding, so ``group_product`` accepts both ``double_round_once`` and
``grouped_double``.

The research ``window_taps`` method (nine [N, Cin] x [Cin, Cout] products summed in
double, with per-tap constants) was removed at L06b; the im2col product is the only form.
Kernel 3, window 7 and 4 groups are the defaults. The 3 x 3 kernel is the window
convolution's identity, not a setting: other kernels are refused. ``conv3x3_window`` is
validated at windows 5, 7, 8 and 12 (P01). ``group_product`` is validated at 1, 2, 4 and 8
contiguous input-channel groups, at windows 7 and 12 (P04), and at 4 groups at window 8 (the
capstone's fusion, C0b), its partials summed in double in group order and rounded once by
the caller, against the unsplit convolution. Other group counts and windows are not
validated.

``depthwise_conv3x3_window`` computes y[d] = bias[d] + sum_t x[t,d] weight[d,t].
Its halo/im2col taps are [rows,9,D], multiplied once in double and reduced over
the nine taps in double, with one final float32 rounding. Weights and optional
bias are pre-repeated to the layout rows. Live coverage is window7, D4 with/without
bias, D1024 with bias, and D16 with bias within the C4/D16 Mix-FFN fixture.
Other widths/windows and ServerArray asset transport have no live coverage here.
"""

from __future__ import annotations

import operator

import numpy as np

from ...core.errors import reject
from ...core.metadata import BlockMetadata, Locality, Validated, validate_metadata
from ...core.ops import WindowOps
from ...core.precision import DOUBLE_ROUND_ONCE, GROUPED_DOUBLE, validate_policy
from ...core.tensor import WINDOW
from ...layouts.window import halo, im2col
from ...weights.constants import ServerArray, WindowLayout, check_rows, constant_value
from .linear import WINDOW_LAYOUT, builder_rows

KERNEL = 3
"""The window convolution's kernel: its identity (other kernels are refused)."""

GROUPS = 4
"""Default input-channel groups of the fusion convolution (the model's four levels). 1, 2,
4 and 8 groups are validated (P04)."""


def _depthwise_parameter(value, name):
    """Validate coefficient transport before building any new depthwise expression."""
    if isinstance(value, ServerArray):
        shape = value.shape
        if not isinstance(shape, (tuple, list)) or any(
                isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1 for n in shape):
            reject(f"depthwise {name} shape must contain positive integer dimensions; received {shape!r}")
        return value
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        reject(f"depthwise {name} must contain float32 coefficients")
    if not np.isfinite(array).all():
        reject(f"depthwise {name} coefficients must be finite float32 values")
    return array


class DepthwiseConv3x3Constants:
    """Depthwise [D,1,3,3] NumPy or [D,9] ServerArray weights, pre-repeated
    as [rows,9,D] in double; optional [D] bias repeated as [rows,D].

    Only kernel3, stride1, padding1, dilation1 and multiplier1 are supported.
    ``rows`` is the window layout or its positive row count; consumption checks it.
    """

    def __init__(self, ops, weight, bias=None, rows=WINDOW_LAYOUT, *, k=3,
                 stride=1, padding=1, dilation=1, multiplier=1,
                 policy: str = DOUBLE_ROUND_ONCE):
        self.policy = validate_policy(policy, allowed=DOUBLE_ROUND_ONCE)
        for name, value, expected in (("k", k, 3), ("stride", stride, 1),
                                      ("padding", padding, 1), ("dilation", dilation, 1),
                                      ("multiplier", multiplier, 1)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value != expected:
                reject(f"depthwise_conv3x3_window requires {name}={expected}; received {value!r}")
        weight = _depthwise_parameter(weight, "weight")
        if bias is not None:
            bias = _depthwise_parameter(bias, "bias")
        shape = weight.shape
        expected = (9,) if isinstance(weight, ServerArray) else (1, 3, 3)
        if len(shape) != len(expected) + 1 or tuple(shape[1:]) != expected or shape[0] < 1:
            reject(f"depthwise weight must be [D,9] ServerArray or [D,1,3,3] NumPy; received {shape}")
        self.channels = int(shape[0])
        if bias is not None and tuple(bias.shape if isinstance(bias, ServerArray) else np.shape(bias)) != (self.channels,):
            reject(f"depthwise bias must be [{self.channels}]")
        self.rows = builder_rows(rows)
        source = constant_value(ops.ee, weight)
        repeated = source.reshape([self.channels, 9]).transpose().reshape([1, 9, self.channels]).repeat(0, self.rows)
        self.weight = ops.to_double(ops.wrap(repeated))
        self.bias = None if bias is None else ops.constant(bias, repeat_rows=self.rows)

    @property
    def embedded_values(self) -> int:
        return depthwise_conv3x3_values(self.channels, bias=self.bias is not None, layout=self.rows)


def depthwise_conv3x3_window(ops, x, c: DepthwiseConv3x3Constants, coarse, *,
                           window: int = WINDOW, policy: str = DOUBLE_ROUND_ONCE):
    """W [window²,D] → W [window²,D], finite-chip zero padding, radius1.

    One double product and one nine-tap double reduction, optional double bias
    addition, then one float32 output cast. Live window7 coverage: D4 optional
    bias, D1024 biased, D16 biased within C4/D16 Mix-FFN. Constants for mismatched
    rows/policy are refused; other widths/windows are not live validated here.
    """
    validate_policy(policy, allowed=DOUBLE_ROUND_ONCE)
    if c.policy != policy:
        reject("depthwise constants and block precision policies differ")
    check_rows(c, WindowLayout(window))
    neighbourhood = halo(ops.ee, x, coarse, c.channels, window=window, ring=1)
    columns = im2col(ops.ee, neighbourhood, c.channels, 3, window=window)
    taps = ops.reshape(ops.to_double(columns), [c.rows, 9, c.channels])
    product = taps.multiply(c.weight)
    # row_total reduces axis1: put all nine taps in that axis, one row per token/channel.
    products = ops.reshape(ops.transpose(product, 1, 2), [c.rows * c.channels, 9])
    total = ops.reshape(ops.row_total(products), [c.rows, c.channels])
    return ops.to_float(total if c.bias is None else total.add(c.bias))


class Conv3x3Constants:
    """W^T of a k x k convolution in (ky, kx, c) row order, [k*k*Cin, Cout]; the bias with
    rows repeated (None: bias-free conv).

    weight: NumPy [Cout, Cin, k, k] or a ServerArray [Cout, Cin * k * k] in PyTorch's
    flatten order (c, ky, kx) (the asset format); the permutation runs on the server.
    ``rows``: the layout the bias serves (default ``WindowLayout(7)``: 49 rows) or its
    row count; recorded as ``rows``."""

    def __init__(self, ops, weight, bias=None, rows=WINDOW_LAYOUT, k: int = KERNEL):
        ee = ops.ee
        if isinstance(weight, ServerArray):
            cout, flat = weight.shape
            if flat % (k * k):
                raise ValueError(f"a [Cout, Cin * {k * k}] weight cannot have {flat} columns (kernel {k})")
            cin = flat // (k * k)
            w = (weight.array.reshape([cout, cin, k * k]).transpose(1, 2)    # [Cout, kk, Cin]
                 .reshape([cout, k * k * cin]).transpose())                  # [kk Cin, Cout]
        else:
            weight = np.asarray(weight, dtype=np.float32)
            cout, cin = weight.shape[:2]
            w = constant_value(ee, weight.transpose(2, 3, 1, 0).reshape(k * k * cin, cout))
        self.k, self.cin, self.cout = k, cin, cout
        self.rows = builder_rows(rows)
        self.weight_t = ops.wrap(w)
        self.bias = None if bias is None else ops.constant(bias, repeat_rows=self.rows)

    @property
    def embedded_values(self) -> int:
        bias_rows = 0 if self.bias is None else self.cout
        return self.k * self.k * self.cin * self.cout + bias_rows


def _finish(ops, product, c: Conv3x3Constants):
    return ops.to_float(product if c.bias is None else product.add(c.bias))


def _check_kernel(c: Conv3x3Constants) -> None:
    """The window convolution reads a one-token ring (``halo``) and taps at offsets -1, 0
    and 1 (``taps``): its kernel is 3 x 3, the block's identity. Constants built for another
    kernel are refused (client-side; the graph is unchanged)."""
    if c.k != KERNEL:
        raise ValueError(f"the window convolution is {KERNEL} x {KERNEL}; constants were built for k = {c.k}")


def conv3x3_window(ops, x, c: Conv3x3Constants, coarse, *, window: int = WINDOW,
                   policy: str = DOUBLE_ROUND_ONCE):
    """Window layout [w^2, Cin] on the window grid `coarse` -> [w^2, Cout] (zero padding at
    the raster edge, exactly PyTorch's). Constants built for another row count than
    ``window ** 2`` are refused, and so are constants for another kernel than 3 x 3, the
    block's identity. Validated at windows 5, 7, 8 and 12 (P01)."""
    validate_policy(policy, allowed=DOUBLE_ROUND_ONCE)
    _check_kernel(c)
    check_rows(c, WindowLayout(window))
    ee = ops.ee
    halo3 = halo(ee, x, coarse, c.cin, window=window)
    return _finish(ops, ops.matmul_double(im2col(ee, halo3, c.cin, c.k, window=window), c.weight_t), c)


def group_weight(weight, g: int, groups: int = GROUPS):
    """W_g, input-channel group g of a [Cout, Cin, 3, 3] weight: channels
    [g Cin / groups, (g + 1) Cin / groups), the g-th of ``groups`` contiguous groups in
    channel order.  As an asset ServerArray [Cout, Cin * 9] in PyTorch's flatten order
    (c, ky, kx), group g is the contiguous column slice [g Cin * 9 / groups,
    (g + 1) Cin * 9 / groups).

    ``groups`` must divide Cin and 0 <= g < groups; otherwise the slice would drop or share
    channels, so it is refused (client-side; the graph is unchanged)."""
    groups, g = operator.index(groups), operator.index(g)
    if groups < 1 or not 0 <= g < groups:
        raise ValueError(f"group {g} of {groups} groups does not exist")
    if isinstance(weight, ServerArray):
        cout, flat = weight.shape
        if flat % (KERNEL * KERNEL) or (flat // (KERNEL * KERNEL)) % groups:
            raise ValueError(f"{groups} groups do not split the {flat} columns of a "
                             f"[Cout, Cin * {KERNEL * KERNEL}] weight into whole channels")
        width = flat // groups
        return ServerArray(weight.array.slice(1, g * width, (g + 1) * width), (cout, width))
    w = np.asarray(weight, np.float32)
    if w.ndim != 4 or w.shape[2:] != (KERNEL, KERNEL):
        raise ValueError(f"expected a [Cout, Cin, {KERNEL}, {KERNEL}] weight; received {w.shape}")
    if w.shape[1] % groups:
        raise ValueError(f"{groups} groups do not split {w.shape[1]} input channels evenly")
    width = w.shape[1] // groups
    return w[:, g * width:(g + 1) * width]


def group_product(ee, part, w: Conv3x3Constants, *, policy: str = DOUBLE_ROUND_ONCE):
    """X_g's [w^2, 9 C_g] im2col times W_g^T, left in double (no rounding).

    ``part`` is a W tensor (the decoder's ``Feature``); its window gives the halo and
    the rows the constants must have. Policy ``double_round_once`` (the default) or its
    ``grouped_double`` variant: the caller rounds once, after summing the groups. Validated
    at 1, 2, 4 and 8 groups (``group_weight``) at windows 7 and 12 (P04), and at 4 groups at
    window 8 (C0b), with the partials summed in double in group order and rounded once."""
    validate_policy(policy, allowed=(DOUBLE_ROUND_ONCE, GROUPED_DOUBLE))
    _check_kernel(w)
    check_rows(w, part)
    img = WindowOps(ee)
    halo3 = halo(ee, part.image, part.grid.coarse(part.window), w.cin, window=part.window)
    return img.matmul_double(im2col(ee, halo3, w.cin, w.k, window=part.window), w.weight_t)


# ---------------------------------------------------------------------------
# Cost functions (contract section 9)
# ---------------------------------------------------------------------------
def conv3x3_values(cin: int, cout: int, *, k: int = KERNEL, bias: bool = False, layout=WINDOW_LAYOUT) -> int:
    """Values of the constants: W^T [k*k*Cin, Cout], plus the bias rows [rows, Cout] (the
    research budgets' convention; ``Conv3x3Constants.embedded_values`` counts a bias as
    Cout values, before the server-side repeat)."""
    return k * k * cin * cout + (builder_rows(layout) * cout if bias else 0)


def conv3x3_arrays(*, bias: bool = False) -> int:
    """Server arrays read for ServerArray weights: the weight, and the bias if any."""
    return 2 if bias else 1


def group_product_values(cin: int, cout: int, *, k: int = KERNEL) -> int:
    """Values of one group's W_g^T [k*k*C_g, Cout] (C_g = Cin / groups): the groups
    together embed the weight once, as column slices."""
    return k * k * cin * cout


def group_product_arrays() -> int:
    """Server arrays read: the group's column slice of the one weight array."""
    return 1


def depthwise_conv3x3_values(channels: int, *, bias: bool = True, layout=WINDOW_LAYOUT) -> int:
    """Evaluated repeated weights [rows,9,D] and optional bias [rows,D]."""
    return builder_rows(layout) * channels * (9 + int(bias))


def depthwise_conv3x3_arrays(*, bias: bool = True) -> int:
    """Source ServerArrays: one weight and optional one bias; repeats are expressions."""
    return 2 if bias else 1


METADATA = validate_metadata({
    "depthwise_conv3x3_window": BlockMetadata(
        name="depthwise_conv3x3_window", layout_in="W", layout_out="W",
        locality=Locality.WINDOW_RING, radius=1,
        policy=DOUBLE_ROUND_ONCE, embedded_constants=depthwise_conv3x3_values,
        asset_arrays=depthwise_conv3x3_arrays,
        status="Ready — 3 × 3, stride1, padding1, dilation1, multiplier1; window7; D4 optional bias, D1024 biased; D16 biased in C4/D16 Mix-FFN",
        evidence=("docs/evidence/segformer_b0-offline.json", "docs/evidence/segformer_b0-live.json"),
        # Validated: the status's form at window 7, with the channel counts (D) and bias it names.
        validated=Validated(tuple(
            {"k": (3,), "stride": (1,), "padding": (1,), "dilation": (1,), "multiplier": (1,), "window": (7,),
             "channels": channels, "bias": bias}
            for channels, bias in (((4,), (True, False)), ((16, 1024), (True,)))),
            sources=(DepthwiseConv3x3Constants,))),
    "conv3x3_window": BlockMetadata(
        name="conv3x3_window", layout_in="W", layout_out="W", locality=Locality.WINDOW_RING, radius=1,
        policy=DOUBLE_ROUND_ONCE, embedded_constants=conv3x3_values, asset_arrays=conv3x3_arrays,
        status="Ready — 3 × 3 kernel, stride 1; windows 5, 7, 8, 12",
        evidence=("docs/history/building-blocks-2026-10-11.md#33-convolution", "docs/history/upernet/notes/milestone7_findings.md",
                  "docs/evidence/p01-offline.json",
                  "docs/evidence/live-fixtures/20261013T074301Z-p01a.json", "docs/evidence/live-fixtures/20261013T074330Z-p01b.json", "docs/evidence/live-fixtures/20261013T075542Z-window-relative_position_bias-w12.json",
                  "docs/evidence/live-fixtures/20261013T115558Z-p04.json"),   # P04 live record (its brief is in the archive)
        # The 3 x 3 kernel is the block's identity (k, refused otherwise); stride 1 is its only stride.
        validated=Validated(({"k": (KERNEL,), "window": (5, 7, 8, 12)},), sources=(Conv3x3Constants,))),
    "group_product": BlockMetadata(
        name="group_product", layout_in="W", layout_out="W", locality=Locality.WINDOW_RING, radius=1,
        policy=DOUBLE_ROUND_ONCE, embedded_constants=group_product_values, asset_arrays=group_product_arrays,
        status="Ready — 1, 2, 4, 8 groups at windows 7, 12; 4 groups at window 8",
        evidence=("docs/history/building-blocks-2026-10-11.md#33-convolution", "variant: grouped_double",
                  "docs/history/upernet/notes/milestone10_findings.md", "docs/evidence/p04-offline.json",
                  "docs/evidence/live-fixtures/20261013T115558Z-p04.json", "docs/evidence/c0b-offline.json",
                  "docs/evidence/live-fixtures/20261013T154410Z-c0b.json"),
        # groups is group_weight's; the 3 x 3 kernel (k) never varied.
        validated=Validated(({"groups": (1, 2, 4, 8), "k": (KERNEL,), "window": (7, 12)},
                             {"groups": (4,), "k": (KERNEL,), "window": (8,)}),
                            sources=(group_weight, Conv3x3Constants))),
})
