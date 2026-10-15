"""Patch-embedding fixtures (task P03): ``patch_embed`` and ``stride_columns`` at other patch
sizes, input bands and embedding widths.

Swin-T embeds 4 x 4 patches of two bands (VV, VH) into 96 channels, on windows of 7
(P00's calibration fixture ``window/patch_embed/p4-vvvh-d96``). Other models embed other
inputs. The p03 fixtures cover each value of each setting at least once:

| Setting | Values |
|---|---|
| Patch size | 2, 4, 8 |
| Input bands | 1, 2, 3, 4, with arbitrary names |
| Embedding width | 48, 96, 128 |
| Window of the output layout | 5, 7, 8, 12 (validated in P01) |

The calibration fixture is the regression at today's settings. ``SELECTIONS["p03"]`` holds
it and the six new fixtures (``CASES``).

**Graph.** The input is an inline [H, W, C] float32 raster on a 2.5 m ``TokenGrid`` (the
model's input grid; tokens are ``2.5 patch`` metres), built with ``core.inline_raster`` and
named by ``raster_bands``. ``patch_embed`` selects the bands by name in the order of
``bands`` (one case stores them in another order, so a block that read the raster's order
would fail), gathers the columns with ``stride_columns``, partitions the tokens into windows,
projects them and normalizes them. Three outputs are read on the token grid: ``pe_columns``
(the ``stride_columns`` row of each token, C patch^2 bands), ``pe_proj`` and ``x`` (window
layout, read with ``to_dense``); the input is echoed on its own grid.

**Request budget.** The input echo is a window-lane request on the input grid, whose side
(windows x window x patch pixels) may not exceed 8 windows: so windows x patch <= 8 per
side. Patch 8 runs on one window, patch 4 on up to 2 x 2 and patch 2 on up to 4 x 4.

**References** are timm 1.0.30 ``PatchEmbed(img_size, patch_size, in_chans, embed_dim,
norm_layer=nn.LayerNorm, output_fmt="NHWC")`` with the fixture's random weights
(``reference.swin_image_encoder.pytorch_image_encoder.make_patch_embed``), checked against
the configuration and run step by step by ``explicit_patch_embed``, which raises unless the
steps equal the module's own forward bit for bit. The columns' reference is timm's own
Conv2d with one-hot weights (``one_hot_extraction``), which must equal ``F.unfold``'s
patches. None reads the library.

**Mirrors** are ``reference.swin_image_encoder.ee_mirror_image_encoder``: the columns are
``F.unfold``'s patches (values moved); ``pe_proj`` is the [N, C p^2] x W^T product with each
dot product summed left to right in double (``sequential_sums``), the bias added in double
and one rounding; ``x`` is LayerNorm in double with left-to-right sums and one rounding.
Each mirror is exact, for these reasons:

- ``pe_columns`` moves values: no arithmetic.
- ``pe_proj`` has the graph's precision policy (``double_round_once``), and its double sums
  are exact in every summation order (``order_analysis``, certified per value). Each term,
  a product of two float32 values or the bias, is a multiple of 2^q, where q is the lowest
  set bit among the value's terms, and sum |terms| < 2^(q + 53). So every partial sum, in
  any order, is a multiple of 2^q below 2^(q + 53) in magnitude: a double, formed without
  rounding. The double result is the exact dot product, rounded once to float32, whatever
  order Earth Engine's matrixMultiply uses. The mirror sums left to right (EE's order,
  Milestone 5) and gets the same value.
- ``x``: P01's argument for ``layer_norm2d``. Per token, the sums run left to right in
  double (EE's ``arrayReduce`` order, Milestone 5; the mirror's ``sequential_sum``), with one
  rounding, as the graph. The mean sums of the float32 ``pe_proj`` are certified exact in
  every order, as above, so the mean, the centred values and their squares are the same in
  every order. Only the variance sum depends on order. ``order_analysis`` bounds its effect
  on each value to first order, with rounding slack, and divides the value's distance to the
  nearest float32 rounding boundary by that bound. Above 1, every order rounds alike.

``MARGINS`` records each fixture's widest certificate (in bits, below 53) and smallest ``x``
ratio. A test recomputes them. One ``x`` value, in ``p4-b4-d128-w12``, has a ratio just
below 1 (``ORDER_SENSITIVE``).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import math

import numpy as np

from .core import Fixture, FixtureData, MirrorSpec, Readout, as_float32, dyadic, from_window_layout, inline_raster
from .window import DRO, LANE, _layout, _metadata, _read, _seeded

INPUT_SCALE = 2.5
"""Metres per input pixel: the Swin-T model grid's 2.5 m (tokens are 2.5 patch metres)."""

REGRESSION = "window/patch_embed/p4-vvvh-d96"
"""Today's settings (patch 4, VV/VH, 96 channels, window 7): P00's calibration fixture."""


@dataclass(frozen=True)
class PatchCase:
    """One p03 fixture's settings. ``bands`` is the block's argument (the Conv2d input
    channel order); ``raster_bands`` the inline raster's band order (``bands`` when None).
    ``windows`` is the window grid (rows, columns) of the token grid."""

    patch: int
    bands: tuple[str, ...]
    dim: int
    window: int
    windows: tuple[int, int]
    seed: int
    raster_bands: tuple[str, ...] | None = None

    def __post_init__(self):
        if sorted(self.raster) != sorted(self.bands):
            raise ValueError("raster_bands must hold the same names as bands")

    @property
    def raster(self) -> tuple[str, ...]:
        return self.bands if self.raster_bands is None else self.raster_bands

    @property
    def order(self) -> list[int]:
        """Raster channel of each band of ``bands``: x[..., order] is the Conv2d input."""
        return [self.raster.index(b) for b in self.bands]

    @property
    def channels(self) -> int:
        return len(self.bands)

    @property
    def columns(self) -> int:
        return self.channels * self.patch * self.patch

    @property
    def tokens(self) -> tuple[int, int]:
        return self.windows[0] * self.window, self.windows[1] * self.window

    @property
    def side(self) -> tuple[int, int]:
        """Input pixels (rows, columns)."""
        return self.tokens[0] * self.patch, self.tokens[1] * self.patch

    @property
    def id(self) -> str:
        return f"window/patch_embed/p{self.patch}-b{self.channels}-d{self.dim}-w{self.window}"

    def input_grid(self):
        from tensor_ee.core.grid import TokenGrid

        return TokenGrid(*self.side, INPUT_SCALE)


CASES = (
    PatchCase(patch=2, bands=("B4", "B3", "B2"), raster_bands=("B2", "B3", "B4"), dim=48, window=5,
              windows=(2, 3), seed=3001),
    PatchCase(patch=2, bands=("b0",), dim=128, window=8, windows=(3, 2), seed=3002),
    PatchCase(patch=2, bands=("HH", "HV"), dim=96, window=12, windows=(2, 2), seed=3003),
    PatchCase(patch=4, bands=("b0", "b1", "b2", "b3"), dim=128, window=12, windows=(2, 2), seed=3004),
    PatchCase(patch=8, bands=("b0",), dim=96, window=7, windows=(1, 1), seed=3005),
    PatchCase(patch=8, bands=("b0", "b1", "b2", "b3"), dim=48, window=8, windows=(1, 1), seed=3006),
)
"""The p03 fixtures (seeds 3001-3006, disjoint from the calibration's 1-18, P02's 21-31 and
P01's 1050-1125):

- patch 2, three Sentinel-2 bands in RGB order (B4, B3, B2) from a raster stored B2, B3, B4,
  width 48, window 5, on 2 x 3 windows;
- patch 2, one band, width 128, window 8, on 3 x 2 windows;
- patch 2, two bands HH/HV, width 96, window 12, on 2 x 2 windows;
- patch 4, four bands, width 128, window 12, on 2 x 2 windows (a 96 x 96 input);
- patch 8, one band, width 96, window 7, on one window (a 56 x 56 input);
- patch 8, four bands (256 columns), width 48, window 8, on one window (64 x 64)."""

REGRESSION_CASE = PatchCase(patch=4, bands=("VV", "VH"), dim=96, window=7, windows=(2, 2), seed=18)
"""The settings of ``REGRESSION`` (``window._pe_data``/``_pe_build``: a 56 x 56 input). Its id
property is not the calibration's; ``case_of(REGRESSION)`` returns it."""


# ---------------------------------------------------------------------------
# Data, graph, reference and mirror
# ---------------------------------------------------------------------------
def _data(seed: int, *, case: PatchCase) -> FixtureData:
    """The calibration fixture's value ranges (``window._pe_data``) at this case's shapes:
    input in [-1, 1), proj.weight in [-0.25, 0.25), bias in [-0.125, 0.125), LayerNorm
    gamma in [0.75, 1.25) and beta in [-0.25, 0.25), all dyadic and exact in float32."""
    height, width = case.side
    return FixtureData(
        inputs={"x": dyadic(seed, "x", (height, width, case.channels))},
        weights={"proj.weight": dyadic(seed, "proj.weight", (case.dim, case.columns), scale=0.25),
                 "proj.bias": dyadic(seed, "proj.bias", (case.dim,), scale=0.125),
                 "norm.weight": dyadic(seed, "norm.weight", (case.dim,), scale=0.25, offset=1.0, bits=20),
                 "norm.bias": dyadic(seed, "norm.bias", (case.dim,), scale=0.25)})


def _build(ee, data: FixtureData, *, case: PatchCase) -> list:
    from tensor_ee.blocks.window.patch import patch_embed, patch_embed_constants
    from tensor_ee.core.ops import WindowOps
    from tensor_ee.layouts.window import flatten_row

    ops = WindowOps(ee)
    grid = case.input_grid()
    tokens = grid.coarse(case.patch)
    dense = inline_raster(ee, grid.projection(ee), data.inputs["x"], case.raster)
    c = patch_embed_constants(ops, dict(data.weights), _layout(case.window))
    t = patch_embed(ops, dense, grid, c, patch=case.patch, bands=case.bands, window=case.window, policy=DRO)
    columns = flatten_row(t["pe_columns"], case.columns).toFloat()       # the [1, C p^2] rows, moved
    return [Readout("pe_columns", columns, case.columns, tokens.pixel_grid()),
            _read("pe_proj", ee, tokens, t["pe_proj"], case.dim, case.window),
            _read("x", ee, tokens, t["x"], case.dim, case.window),
            Readout("input", dense, case.channels, grid.pixel_grid(), "input", "x")]


def conv_input(data: FixtureData, case: PatchCase) -> np.ndarray:
    """The Conv2d input [H, W, C]: the raster's channels in the order of ``bands``."""
    return np.ascontiguousarray(data.inputs["x"][..., case.order])


def _reference(data: FixtureData, *, case: PatchCase) -> dict:
    """timm ``PatchEmbed`` with the fixture's weights, step by step (``explicit_patch_embed``
    raises unless the steps equal the module's forward bit for bit); the columns from timm's
    own Conv2d with one-hot weights, which must equal ``F.unfold``'s patches."""
    from reference.swin_image_encoder.pytorch_image_encoder import (check_patch_embed, explicit_patch_embed,
                                                                    make_patch_embed, one_hot_extraction,
                                                                    to_nchw)

    x = conv_input(data, case)
    height, width, channels = x.shape
    with _seeded():
        module = make_patch_embed(height, width, dict(data.weights), embed_dim=case.dim, patch=case.patch,
                                  in_chans=channels)
        extracted = one_hot_extraction(x, patch=case.patch)
    check_patch_embed(module, patch=case.patch, in_chans=channels, embed_dim=case.dim)
    if tuple(module.img_size) != (height, width) or not module.strict_img_size:
        raise AssertionError("timm's PatchEmbed is not built for the fixture's input")
    result = explicit_patch_embed(module, to_nchw(x))
    columns = extracted.astype(np.float32)
    if not np.array_equal(columns.astype(np.float64), extracted) or \
            not np.array_equal(columns, result["pe_patches"][0].numpy()):
        raise AssertionError("timm's one-hot extraction differs from F.unfold's patches")
    return {"pe_columns": columns, "pe_proj": result["pe_proj"][0].numpy(), "x": result["x"][0].numpy()}


def _mirror(data: FixtureData, *, case: PatchCase) -> dict:
    """``patches`` (F.unfold) for the columns; ``pe_mirror`` with sequential sums for the
    projection and the LayerNorm, in window layout, read back on the token grid."""
    from reference.swin_image_encoder.ee_mirror_image_encoder import patch_windows, pe_mirror
    from reference.swin_image_encoder.pytorch_image_encoder import patches, to_nchw

    x = conv_input(data, case)
    rows, cols = case.tokens
    t = pe_mirror(dict(data.weights), patch_windows(x, patch=case.patch, window=case.window), sequential_sums=True)
    result = {"pe_columns": as_float32(patches(to_nchw(x), case.patch)[0].numpy(), "pe_columns")}
    result.update({name: as_float32(from_window_layout(t[name], rows, cols, case.window), name)
                   for name in ("pe_proj", "x")})
    return result


# ---------------------------------------------------------------------------
# Summation order: the exactness certificates and the margins of x (the exact flags' argument)
# ---------------------------------------------------------------------------
UNIT_ROUNDOFF = 2.0 ** -53
LN_EPS = 1e-5
"""timm PatchEmbed's LayerNorm eps (nn.LayerNorm's default; the library's ``LN_EPS``)."""


def gamma(n: int) -> float:
    """gamma_n = n u / (1 - n u): the relative error bound of n double additions."""
    return n * UNIT_ROUNDOFF / (1 - n * UNIT_ROUNDOFF)


def lowest_bit_exponent(values) -> np.ndarray:
    """The exponent e of each double's lowest set bit (the value is an odd integer times
    2^e), +inf for 0."""
    a = np.asarray(values, np.float64)
    mantissa, exponent = np.frexp(a)                    # a = mantissa 2^exponent, 0.5 <= |mantissa| < 1
    integer = np.abs(mantissa * 2.0 ** 53).astype(np.int64)                 # exact: below 2^53
    lowest = integer & -integer                          # the lowest set bit, a power of two
    out = np.full(a.shape, np.inf)
    nonzero = integer != 0
    out[nonzero] = exponent[nonzero] - 53 + (np.frexp(lowest[nonzero].astype(np.float64))[1] - 1)
    return out


def exact_sum_bits(terms) -> np.ndarray:
    """For each sum over the last axis, log2(sum |terms| / 2^q), with q the lowest set bit
    among its terms (``lowest_bit_exponent``); sum |terms| is bounded above (the computed sum
    times 1 + 2 gamma_n). Below 53, every partial sum of the terms, in any order, is a
    multiple of 2^q below 2^(q + 53) in magnitude, so a double: every summation order in
    double gives the exact sum."""
    terms = np.asarray(terms, np.float64)
    q = lowest_bit_exponent(terms).min(axis=-1)
    total = np.abs(terms).sum(axis=-1) * (1 + 2 * gamma(terms.shape[-1]))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.isfinite(q), np.log2(total) - q, -np.inf)


def float32_boundary_distance(values) -> np.ndarray:
    """The distance of each double to the nearest float32 rounding boundary (the midpoints
    around its float32 rounding, exact in double)."""
    values = np.asarray(values, np.float64)
    single = values.astype(np.float32)
    below = np.nextafter(single, np.float32(-np.inf)).astype(np.float64)
    above = np.nextafter(single, np.float32(np.inf)).astype(np.float64)
    center = single.astype(np.float64)
    return np.minimum(values - (center + below) / 2, (center + above) / 2 - values)


def _left_to_right(terms: np.ndarray) -> np.ndarray:
    """Sums over the last axis, strictly left to right in double (EE's order)."""
    out = terms[..., 0].copy()
    for k in range(1, terms.shape[-1]):
        out = out + terms[..., k]
    return out


def order_analysis(x: np.ndarray, weights: dict, patch: int) -> dict:
    """How much the p03 mirror's float32 outputs depend on summation order, for the Conv2d
    input x [H, W, C] and the fixture's weights. Token rows and columns, not windows.

    - ``pe_proj``: every value's terms (its C patch^2 products, exact in double, and the
      bias) get ``exact_sum_bits``; below 53, the value is the same in every order.
    - The LayerNorm's mean sums, over the float32 ``pe_proj`` of each token: likewise.
    - ``x``: with the means certified, only the variance sum S = sum c^2 depends on order;
      its terms c^2 are the same doubles in every order. Any order is within
      ``gamma_(dim - 1) S`` of the exact sum (``math.fsum``), so within that plus the
      left-to-right error of the mirror's sum. To first order, x = c (S/dim + eps)^(-1/2) g + b
      moves by ``|c| |g| r^3 dvar / 2`` (r = (var + eps)^(-1/2), dvar the bound over dim with
      the division's rounding), plus 6 u (|c| r |g| + |b|) for the rounding of the later
      operations. The ratio is the distance of the mirror's double x to the nearest float32
      boundary over that bound.

    Returns ``{"values", "proj_bits", "mean_bits", "x_ratio", "x_below_one", "x_where",
    "x_value", "x_distance", "x_bound", "x"}``: the widest certificates, the smallest ratio,
    the number of ratios at most 1, the smallest one's (row, column, channel), its double
    value, distance and bound, and the float32 x [h, w, dim] (the mirror's)."""
    from reference.swin_image_encoder.pytorch_image_encoder import patches, to_nchw

    columns = patches(to_nchw(x), patch)[0].numpy().astype(np.float64)       # [h, w, K]
    w = np.asarray(weights["proj.weight"], np.float64)
    bias = np.asarray(weights["proj.bias"], np.float64)
    gain = np.asarray(weights["norm.weight"], np.float64)
    shift = np.asarray(weights["norm.bias"], np.float64)
    h, wd, _ = columns.shape
    dim = w.shape[0]
    products = columns[:, :, None, :] * w[None, None, :, :]                   # [h, w, dim, K], exact
    terms = np.concatenate([products, np.broadcast_to(bias[:, None], (h, wd, dim, 1))], axis=-1)
    proj_bits = exact_sum_bits(terms)
    if not (proj_bits < 53).all():
        raise ValueError("a pe_proj value is not certified exact in every order")
    v = _left_to_right(terms).astype(np.float32).astype(np.float64)          # pe_proj, exact sums rounded once
    mean_bits = exact_sum_bits(v)
    if not (mean_bits < 53).all():
        raise ValueError("a LayerNorm mean sum is not certified exact in every order")
    mean = _left_to_right(v)[..., None] / dim
    centred = v - mean
    squares = centred * centred
    total = _left_to_right(squares)
    variance = total[..., None] / dim
    y = centred / np.sqrt(variance + LN_EPS) * gain + shift
    exact = np.array([math.fsum(row) for row in squares.reshape(-1, dim).tolist()]).reshape(h, wd)
    spread = gamma(dim - 1) * total + np.abs(total - exact)                    # any order vs the mirror's
    dvar = spread[..., None] / dim * (1 + 2 * UNIT_ROUNDOFF) + 2 * UNIT_ROUNDOFF * variance
    r = 1 / np.sqrt(variance + LN_EPS)
    bound = (0.5 * np.abs(centred) * np.abs(gain) * r ** 3 * dvar
             + 6 * UNIT_ROUNDOFF * (np.abs(centred) * r * np.abs(gain) + np.abs(shift)))
    distance = float32_boundary_distance(y)
    ratio = distance / bound
    worst = np.unravel_index(int(np.argmin(ratio)), ratio.shape)
    return {"values": int(ratio.size), "proj_bits": float(proj_bits.max()), "mean_bits": float(mean_bits.max()),
            "x_ratio": float(ratio[worst]), "x_below_one": int((ratio <= 1).sum()),
            "x_where": [int(i) for i in worst], "x_value": float(y[worst]),
            "x_distance": float(distance[worst]), "x_bound": float(bound[worst]), "x": y.astype(np.float32)}


def case_of(fixture_id: str) -> PatchCase:
    """The settings of a p03 fixture: one of ``CASES``, or ``REGRESSION_CASE``."""
    for case in (REGRESSION_CASE, *CASES):
        if case.id == fixture_id or (case is REGRESSION_CASE and fixture_id == REGRESSION):
            return case
    raise KeyError(fixture_id)


def margins(fixture_id: str) -> dict:
    """``order_analysis`` of a p03 fixture on its own registered data."""
    from . import registry

    case, fixture = case_of(fixture_id), registry()[fixture_id]
    data = fixture.data(fixture.seed)
    return order_analysis(conv_input(data, case), data.weights, case.patch)


def round_up(value: float, decimals: int = 1) -> float:
    """``value`` rounded up to ``decimals`` decimals (how ``MARGINS`` records a certificate's
    bits)."""
    scale = 10.0 ** decimals
    return math.ceil(value * scale) / scale


def round_down(value: float, digits: int = 3) -> float:
    """``value`` rounded towards zero to ``digits`` significant digits (how ``MARGINS`` records
    a ratio)."""
    if value == 0 or not math.isfinite(value):
        return value
    scale = 10.0 ** (digits - 1 - math.floor(math.log10(abs(value))))
    return math.trunc(value * scale) / scale


MARGINS = {
    REGRESSION: (49.9, 44.5, 200.0, 0),
    "window/patch_embed/p2-b3-d48-w5": (48.6, 42.8, 2130.0, 0),
    "window/patch_embed/p2-b1-d128-w8": (47.7, 46.3, 88.6, 0),
    "window/patch_embed/p2-b2-d96-w12": (48.4, 45.4, 44.3, 0),
    "window/patch_embed/p4-b4-d128-w12": (50.6, 45.1, 0.995, 1),
    "window/patch_embed/p8-b1-d96-w7": (50.5, 42.7, 471.0, 0),
    "window/patch_embed/p8-b4-d48-w8": (52.3, 40.8, 959.0, 0),
}
"""Per fixture (``margins``): the widest certificate of the ``pe_proj`` sums and of the
LayerNorm mean sums, in bits (rounded up to one decimal; below 53, exact in every order);
the smallest order ratio of ``x`` (rounded down to three significant digits); and the
number of ``x`` values whose ratio is at most 1. ``tests/test_p03_patch_embedding.py``
recomputes them."""

ORDER_SENSITIVE = {
    "window/patch_embed/p4-b4-d128-w12": (
        " One x value, about -0.7336123 at token row 12, column 16, channel 30 (window 3, token 4), lies "
        "within the first-order reordering bound of the LayerNorm's variance sum from a float32 rounding "
        "boundary (ratio 0.995: distance 4.22e-15, bound 4.24e-15). The reversed, exact, NumPy, pairwise, "
        "strided (2 to 16 lanes) and chunked (4 to 32) sums, and 2,000 random orders, all round it alike. "
        "A single differing x value there would point to Earth Engine's summation order, not to a library "
        "error."),
}
"""Values a summation order other than left to right could round otherwise (ratio <= 1)."""


# ---------------------------------------------------------------------------
# The fixtures
# ---------------------------------------------------------------------------
P03_NEW = ("New setting (P03): no milestone ran this patch, band list or width. The regression is "
           f"{REGRESSION} (docs/history/swin_image_encoder/notes/milestone6_findings.md, answers 1-5: "
           "patch membership, flatten order and window layout exact; projection and LayerNorm "
           "bit-identical to the mirror).")


def _fixture(case: PatchCase, requests: int) -> Fixture:
    from tensor_ee.blocks.window import patch as WP

    rows, cols = case.tokens
    height, width = case.side
    proj_bits, mean_bits, ratio, below_one = MARGINS[case.id]
    margin_text = (f" Summation order (order_analysis): the pe_proj sums (K = {case.columns} products and the "
                   f"bias) are exact in every order, certified at most {proj_bits:g} bits (below 53), and so are "
                   f"the LayerNorm mean sums ({mean_bits:g} bits). The x order ratio is at least {ratio:g} over "
                   f"{rows * cols * case.dim} values, {below_one} at most 1 (above 1, every order of the variance "
                   "sum rounds alike)." + ORDER_SENSITIVE.get(case.id, ""))
    order = ("" if case.raster_bands is None else
             f" The raster stores the bands as {', '.join(case.raster)}; the block selects them by name "
             f"in the order {', '.join(case.bands)}, the Conv2d's input channels.")
    return Fixture(
        id=case.id, lane=LANE, block="patch_embed", parts=("stride_columns",),
        settings={"patch": case.patch, "bands": case.bands, "dim": case.dim, "window": case.window},
        policy=DRO,
        shapes={"input": [height, width, case.channels], "raster_bands": list(case.raster),
                "tokens": [rows, cols], "windows": list(case.windows), "columns": case.columns,
                "dim": case.dim},
        seed=case.seed, data=partial(_data, case=case), build=partial(_build, case=case),
        reference=partial(_reference, case=case),
        reference_source=(f"timm PatchEmbed(img_size=({height}, {width}), patch_size={case.patch}, "
                          f"in_chans={case.channels}, embed_dim={case.dim}, norm_layer=nn.LayerNorm, "
                          "output_fmt='NHWC'), random weights, step by step; columns from its Conv2d with "
                          "one-hot weights"),
        mirror=MirrorSpec(partial(_mirror, case=case), True,
                          "reference.swin_image_encoder.ee_mirror_image_encoder: patches (F.unfold); "
                          f"pe_mirror(sequential_sums=True) on patch_windows(patch={case.patch}, "
                          f"window={case.window})"),
        requests=requests,
        constants=lambda data, case=case: {"patch_embed": _metadata("patch_embed").embedded_constants(
            case.dim, patch=case.patch, channels=case.channels, layout=_layout(case.window))},
        history=P03_NEW,
        accepts=(WP.patch_embed, WP.stride_columns, WP.patch_embed_values), window=case.window,
        notes=("Exact: pe_columns moves values; pe_proj sums each dot product left to right in double, as "
               "EE's matrixMultiply (Milestone 5), adds the bias in double and rounds once; x is LayerNorm "
               "with left-to-right sums in double and one rounding, on the float32 pe_proj."
               + margin_text + order))


REQUESTS = {case.id: 2 for case in CASES}
"""Requests per fixture, as the planner builds them (the token grid's readouts, then the
input echo; split further where a response's NPY header would exceed the limit)."""


def fixtures() -> list[Fixture]:
    return [_fixture(case, REQUESTS[case.id]) for case in CASES]


def _p03() -> tuple[str, ...]:
    """The regression (P00's calibration fixture) and the six new fixtures: one live run."""
    return (REGRESSION,) + tuple(case.id for case in CASES)


SELECTIONS = {"p03": _p03}
"""Named selections of this lane module (``tools.fixtures.select``): name -> a callable
returning the fixture ids. A run sends its fixtures in registry order."""
