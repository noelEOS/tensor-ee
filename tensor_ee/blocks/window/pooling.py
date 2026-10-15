"""PyTorch's AdaptiveAvgPool2d of a window-layout map (the PSP's pooled maps).

Moved from ``upernet/earth_engine/_coefficients.py`` (``pool_rows`` and its helpers
``SIZES``, ``bins``, ``onehot_bins``, ``check_rule``, ``offsets``, ``axis_membership``)
and ``upernet/earth_engine/adaptive_pool.py`` (``gather``, ``window_pool`` and their
helpers ``PoolPlan``, ``plan_for``, ``_areas``, ``window_membership``,
``window_partials``, ``Pooled``), research code archived at the tag
``research-archive-2026-10-12`` (docs/history/README.md). The ``global`` and ``dense``
formulations were research only.

Adaptive average pooling of an H x W feature map to s x s is a fixed linear map: output
(p, q) is the mean over rows [floor(pH/s), ceil((p+1)H/s)) x columns [floor(qW/s),
ceil((q+1)W/s)) (ATen's integer rule, checked against PyTorch's own kernel for every
shape: ``check_rule``). The PSP needs s = 1, 2, 3, 6 of the same map; their 50 bins are
stacked into one map with R = 50 rows (SMP's block order, bins row-major).

``window_pool`` (production), on the window layout ([w^2, C] per window):

1. per window, the [R, w^2] 0/1 membership of its tokens in every bin, built on the
   server from per-axis tables Ry [R, H], Rx [R, W] (R (H + W) embedded values), times
   the window's tokens: R partial bin sums, double, sequential over the window's tokens;
2. the window partials gathered into one value with ``sampleRectangle`` (``gather``: the
   window grid's pixels, row-major; a deterministic order) and summed over the window
   axis (ee.Array.reduce, sequential, double);
3. divided by the bin area (an exact integer) in double, rounded to float32 once
   (policy ``double_round_once``).

The result is an ee.Array [R, C] *value* (V). The pool depends on the whole chip or tile
(EXTENT), and ``sampleRectangle`` refuses a region containing a masked window instead of
dropping it. No mask is read: the bins divide by their constant areas. Bins (1, 2, 3, 6)
and window 7 are the defaults. Validated (P04) on token extents of 5 to 14: one window of
5, 7, 8 or 12 tokens and 2 x 2 windows of 7, with the bins (1, 2, 3, 6); and one window of
8 or 12 tokens with the bins (1, 2, 4, 8). Uneven and overlapping bins (more bins than
pixels, as 5 -> 6) follow PyTorch's start and end rule, checked against PyTorch's kernel on
index maps. Other extents and bins are not validated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from ...core.metadata import ANY_SETTINGS, NO_ARITHMETIC, BlockMetadata, Locality, Validated, validate_metadata
from ...core.precision import DOUBLE_ROUND_ONCE, validate_policy
from ...core.tensor import WINDOW
from ...layouts.window import reshape
from .linear import no_constants

SIZES = (1, 2, 3, 6)
"""Default PSP bins (SMP's pyramid sizes). (1, 2, 4, 8) is validated too (P04)."""


# ---------------------------------------------------------------------------
# Bins (client side, exact; checked against PyTorch's kernel)
# ---------------------------------------------------------------------------
def bins(n: int, m: int) -> list[tuple[int, int]]:
    """[(start, end)] of the m output cells of an n-pixel axis (ATen's integer rule)."""
    return [((i * n) // m, -((-(i + 1) * n) // m)) for i in range(m)]


def onehot_bins(n: int, m: int) -> list[tuple[int, int]]:
    """The same, read off PyTorch's kernel: pool n one-hot rows and record which input
    pixels each output cell sees (non-zero weight), and that the weight is 1 / count."""
    eye = torch.eye(n, dtype=torch.float64)[:, None, None, :]          # n planes [1, n]
    out = F.adaptive_avg_pool2d(eye, (1, m))[:, 0, 0, :].T             # [m, n]
    result = []
    for row in out:
        members = torch.nonzero(row).flatten().tolist()
        if members != list(range(members[0], members[-1] + 1)):
            raise AssertionError("a bin is not a contiguous range")
        if not torch.equal(row[members], torch.full((len(members),), 1.0 / len(members),
                                                     dtype=torch.float64)):
            raise AssertionError("bin weights are not 1 / count")
        result.append((members[0], members[-1] + 1))
    return result


def check_rule(n: int, m: int) -> list[tuple[int, int]]:
    rule, kernel = bins(n, m), onehot_bins(n, m)
    if rule != kernel:
        raise AssertionError(f"{n} -> {m}: rule {rule} != PyTorch {kernel}")
    return rule


def pool_rows(h: int, w: int, sizes=SIZES) -> list[dict]:
    """One record per row of the stacked pooled matrix (sizes in SMP's block order)."""
    rows = []
    for s in sizes:
        rb, cb = check_rule(h, s), check_rule(w, s)
        for p, (r0, r1) in enumerate(rb):
            for q, (c0, c1) in enumerate(cb):
                rows.append({"size": s, "p": p, "q": q, "r0": r0, "r1": r1, "c0": c0, "c1": c1,
                             "area": (r1 - r0) * (c1 - c0)})
    return rows


def offsets(sizes=SIZES) -> dict[int, tuple[int, int]]:
    """{size: (first row, end row)} of each pool in the stacked matrix."""
    out, start = {}, 0
    for s in sizes:
        out[s] = (start, start + s * s)
        start += s * s
    return out


def axis_membership(h: int, w: int, sizes=SIZES) -> tuple[np.ndarray, np.ndarray]:
    """(Ry [R, h], Rx [R, w]) 0/1: row r of the stacked matrix covers rows Ry[r] x columns Rx[r]."""
    rows = pool_rows(h, w, sizes)
    ry, rx = np.zeros((len(rows), h)), np.zeros((len(rows), w))
    for k, r in enumerate(rows):
        ry[k, r["r0"]:r["r1"]] = 1
        rx[k, r["c0"]:r["c1"]] = 1
    return ry, rx


# ---------------------------------------------------------------------------
# The plan (client side) and the window formulation
# ---------------------------------------------------------------------------
def _check_sizes(sizes) -> None:
    """Pool sizes are distinct positive integers: ``offsets`` and ``Pooled.pool`` find a
    pool by its size, so a repeated size would hide the first pool's rows. More bins than
    pixels is allowed (PyTorch's bins then overlap, as 5 -> 6 does)."""
    values = list(sizes)
    if not values or any(isinstance(s, bool) or not isinstance(s, (int, np.integer)) or s < 1 for s in values):
        raise ValueError(f"pool sizes must be positive integers; received {tuple(values)}")
    if len(set(values)) != len(values):
        raise ValueError(f"pool sizes must be distinct; received {tuple(values)}")


@dataclass(frozen=True)
class PoolPlan:
    """The stacked pooling map of an H x W feature map (client side, exact)."""

    height: int
    width: int
    sizes: tuple = SIZES
    rows: list = field(init=False, compare=False)

    def __post_init__(self):
        _check_sizes(self.sizes)
        object.__setattr__(self, "rows", pool_rows(self.height, self.width, self.sizes))

    @property
    def count(self) -> int:
        return len(self.rows)

    @property
    def areas(self) -> np.ndarray:
        return np.array([r["area"] for r in self.rows], dtype=np.float64)

    def axis_tables(self) -> tuple[np.ndarray, np.ndarray]:
        return axis_membership(self.height, self.width, self.sizes)

    def offsets(self) -> dict[int, tuple[int, int]]:
        return offsets(self.sizes)

    @property
    def embedded_values(self) -> int:
        """Window formulation: Ry, Rx and the area column (R x C after repeat, built on the
        server from R values)."""
        return self.count * (self.height + self.width) + self.count


def plan_for(tokens, *, sizes=SIZES) -> PoolPlan:
    return PoolPlan(tokens.height, tokens.width, sizes)


def _areas(ee, plan: PoolPlan, channels: int):
    """[R, C] double bin areas (exact integers), repeated on the server."""
    return ee.Array([[float(a)] for a in plan.areas]).repeat(1, channels)


def window_membership(ee, plan: PoolPlan, coarse, *, window: int = WINDOW):
    """[R, w^2] 0/1 per window (I, J): entry (r, w a + b) = Ry[r, w I + a] * Rx[r, w J + b].

    The plan must be the plan of this extent: ``coarse`` windows of ``window`` tokens. A plan
    of another extent would slice its tables at the wrong tokens, so it is refused
    (client-side; the graph is unchanged)."""
    if (plan.height, plan.width) != (coarse.height * window, coarse.width * window):
        raise ValueError(f"a pool plan for {plan.height} x {plan.width} tokens cannot pool "
                         f"{coarse.height * window} x {coarse.width * window} tokens")
    ry, rx = plan.axis_tables()
    r = plan.count
    coords = ee.Image.pixelCoordinates(ee.Projection(coarse.crs, coarse.transform)).floor()
    i0 = coords.select("y").clamp(0, coarse.height - 1).multiply(window).toInt()
    j0 = coords.select("x").clamp(0, coarse.width - 1).multiply(window).toInt()
    rows = ee.Image(ee.Array(ry.tolist())).arraySlice(1, i0, i0.add(window))   # [R, w] (a)
    cols = ee.Image(ee.Array(rx.tolist())).arraySlice(1, j0, j0.add(window))   # [R, w] (b)
    rows = reshape(ee, rows, [r, window, 1]).arrayRepeat(2, window)
    cols = reshape(ee, cols, [r, 1, window]).arrayRepeat(1, window)
    return reshape(ee, rows.multiply(cols), [r, window * window])


def gather(ee, image, coarse, band: str = "v"):
    """Every pixel of a window-grid array image as one ee.Array [n_h, n_w, ...], row-major
    windows (sampleRectangle; fails on a masked window instead of dropping it)."""
    projection = ee.Projection(coarse.crs, coarse.transform)
    feature = image.rename(band).reproject(projection).sampleRectangle(coarse.region(ee))
    return ee.Array(feature.get(band))


def window_partials(ee, x, tokens, channels: int, plan: PoolPlan, *, window: int = WINDOW):
    """[R, C] double partial bin sums per window (an image on the window grid)."""
    return window_membership(ee, plan, tokens.coarse(window), window=window).matrixMultiply(x)


@dataclass
class Pooled:
    sums: object          # ee.Array [R, C] double
    pooled: object        # ee.Array [R, C] float32
    plan: PoolPlan
    channels: int

    def pool(self, s: int):
        """ee.Array [s^2, C] float32: the s x s map, row-major."""
        a, b = self.plan.offsets()[s]
        return self.pooled.slice(0, a, b)


def window_pool(ee, x, tokens, channels: int, plan: PoolPlan | None = None, *, sizes=SIZES,
                window: int = WINDOW, policy: str = DOUBLE_ROUND_ONCE) -> Pooled:
    """AdaptiveAvgPool2d of window layout x [w^2, C] on tokens.coarse(w) to every s x s of
    ``sizes`` (or of ``plan``), as one ``Pooled`` value. Validated on the square extents 5, 7,
    8, 12 and 14 tokens with the bins (1, 2, 3, 6), and 8 and 12 with (1, 2, 4, 8), at windows
    5, 7, 8 and 12 (P04; see the module docstring). A ``plan`` must be the plan of ``sizes``:
    one for other sizes is refused (pass the plan's sizes with it), so that ``sizes`` is never
    ignored silently."""
    validate_policy(policy, allowed=DOUBLE_ROUND_ONCE)
    if plan is not None and tuple(plan.sizes) != tuple(sizes):
        raise ValueError(f"a pool plan for the sizes {tuple(plan.sizes)} cannot pool to {tuple(sizes)}; "
                         "pass the plan's sizes")
    plan = plan or plan_for(tokens, sizes=sizes)
    coarse = tokens.coarse(window)
    n = coarse.height * coarse.width
    partial = gather(ee, window_partials(ee, x, tokens, channels, plan, window=window), coarse)
    sums = (partial.reshape([n, plan.count * channels]).reduce(ee.Reducer.sum(), [0])
            .reshape([plan.count, channels]))
    return Pooled(sums, sums.divide(_areas(ee, plan, channels)).float(), plan, channels)


# ---------------------------------------------------------------------------
# Cost functions (contract section 9)
# ---------------------------------------------------------------------------
def window_pool_values(height: int, width: int, *, sizes=SIZES) -> int:
    """Values embedded for an height x width token map: Ry [R, H], Rx [R, W] and the R bin
    areas (``PoolPlan.embedded_values``), R = sum of s^2 over ``sizes``."""
    count = sum(s * s for s in sizes)
    return count * (height + width) + count


METADATA = validate_metadata({
    "window_pool": BlockMetadata(
        name="window_pool", layout_in="W", layout_out="V", locality=Locality.EXTENT,
        policy=DOUBLE_ROUND_ONCE, embedded_constants=window_pool_values, asset_arrays=no_constants,
        # The status names the square extents P04 ran (lead decision, X02b fix round).
        status=("Ready — extents 5, 7, 8, 12 and 14 tokens with bins (1, 2, 3, 6); "
                "extents 8 and 12 with bins (1, 2, 4, 8)"),
        evidence=("docs/history/building-blocks-2026-10-11.md#38-pooling", "docs/history/upernet/notes/milestone8_findings.md",
                  "docs/evidence/p04-offline.json",
                  "docs/evidence/live-fixtures/20261013T115558Z-p04.json"),
        # The token extent is window_pool_values' height x width, square; the windows varied (5, 7, 8, 12).
        validated=Validated(tuple({"height": (extent,), "width": (extent,), "sizes": (sizes,)}
                                  for sizes, extents in (((1, 2, 3, 6), (5, 7, 8, 12, 14)), ((1, 2, 4, 8), (8, 12)))
                                  for extent in extents))),
    # Validated: any accepted call (the band name is a label).
    "gather": BlockMetadata(
        name="gather", layout_in="W", layout_out="V", locality=Locality.EXTENT,
        policy=NO_ARITHMETIC, embedded_constants=no_constants, asset_arrays=no_constants,
        status="Ready",
        evidence=("docs/history/building-blocks-2026-10-11.md#38-pooling", "docs/history/upernet/notes/milestone8_findings.md"),
        validated=ANY_SETTINGS),
})
