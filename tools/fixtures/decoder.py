"""Decoder fixtures: P04's pooling extents, bins and fusion group counts, and C0b's window 8.

**C0b** (docs/tasks/C0-capstone.md): the capstone runs ``group_product``, ``window_resize`` and
``window_matrix_resize`` at window 8, which had no window-8 evidence. The selection ``c0b``
holds ``window/group_product/g4-w8``, every resize of the capstone at its own token extents
(``window/window_resize/[ac-]f<factor>-e<extent>-w8``: the FPN's and fusion's x2, x4 and x8 with
align_corners=False, and the head's x4 from 64 to 256 tokens with align_corners=True, read as
its four diagonal 64 x 64 blocks) and the PSP's branches onto C4's 8 x 8 tokens
(``window/window_matrix_resize/s<bins>-e8-w8``), with the calibration's ``group_product`` as the
window-7 regression. The references are ``torch.nn.functional``; the resize mirrors take
PyTorch's weights read off ``F.interpolate`` on one-hot inputs, not the library's (each
fixture's notes give its exactness argument; ``tests/test_c0b_window8.py`` checks them).

**P04.** Window-lane fixtures (lane ``window``) for two of the UPerNet decoder's blocks, at
settings the reference model does not use. The selection ``p04`` holds them, with the
calibration's ``window_pool`` and ``group_product`` fixtures as the window-7 regression:

- **window_pool** (``window/window_pool/e<extent>-bins<sizes>-w<window>``): PyTorch's adaptive
  average pooling of a token map of 5 x 5, 8 x 8 and 12 x 12 tokens (one window of 5, 8 and
  12) and 14 x 14 tokens (2 x 2 windows of 7) to the bins (1, 2, 3, 6); and of 8 x 8 and
  12 x 12 tokens to the bins (1, 2, 4, 8). 5 -> 6 has more bins than pixels, so its bins
  overlap; 12 -> 8 overlaps too; 8 -> (1, 2, 4, 8) divides evenly.
- **group_product** (``window/group_product/g<groups>-w<window>``): the fusion's grouped
  double product, the 3 x 3 convolution of 8 input channels split into 1, 2, 4 or 8
  contiguous channel groups, each group's im2col product left in double, the partials summed
  in double in group order 0 .. G - 1 and rounded to float32 once, at windows 7 and 12,
  against ``F.conv2d`` of the same unsplit weight.
- **order probes** (``window/group_product/order-g<groups>-w7``, 4 and 8 groups): one output
  channel whose weights are designed so that, at one token, the float32 result of the double
  group sum depends on the order of the sum (``PROBE_PARTIALS``). The production order gives
  one value; every other order tested (all of them at 4 groups, except the swap of the first
  two, which commutes; at 8 groups the reversed order and every rotation) gives another,
  and so does rounding each group's partial to float32 before summing
  (``tests/test_p04_decoder_blocks.py``).

**The sum is the fixture's.** ``group_product`` returns one group's partial in double; its
caller sums the partials and rounds once. The fixtures compose that sum themselves, over
``range(groups)``. They do not use the decoder's ``FUSION_ORDER``, which lists the model's
four groups only (``FUSION_ORDER == (0, 1, 2, 3)``, the same order at 4 groups: tested).

**References** (R1). Random weights, no library code: SMP 0.5.0 ``PSPModule(sizes=...)``'s
``AdaptiveAvgPool2d`` blocks on a channels-last input (the memory format SMP's PSP receives,
Milestone 8 §3), and ``torch.nn.functional.conv2d(padding=1)`` of the unsplit weight.

**Mirrors.** The pooling mirror is ``reference.upernet.ee_mirror_psp.pool`` in Earth Engine's
window order, with its bin tables read off PyTorch's ``adaptive_avg_pool2d`` on one-hot index
maps (``source="index_maps"``): no library code, unlike the calibration's mirror, whose tables
are the library's. The group mirror is ``ee_mirror_fpn.conv3x3`` per group in double and
``ee_mirror_fusion.sum_partials`` in group order, in double, with one rounding.

**Why the mirrors are exact.**

- *Pooling.* Inputs are multiples of 2^-23 in [-1, 1), and a bin holds at most 196 of them,
  so every partial and bin sum is a multiple of 2^-23 below 2^8 in magnitude: exact in
  double, whatever the order. The division by the bin area (an integer) is one correctly
  rounded double operation, then one float32 rounding, in Earth Engine and in the mirror.
- *Group products.* Inputs are multiples of 2^-23 in [-1, 1) and weights multiples of 2^-25
  in [-1/4, 1/4), so every product is an exact multiple of 2^-48 below 1/4, and every partial
  sum of the 72 products of an output is below 18 < 2^5: exact in double. So each group's
  partial and the group sum are exact whatever the order, and one float32 rounding follows.
- *Order probes.* The probe channel's group partials are exact (one product each), but their
  double sum is not, at the probe token by design and at about a third of the other tokens
  (64 of 196 at 4 groups, 81 of 196 at 8 groups), whose weights span 2^-63 to 1. On this
  data the float32 result depends on the order only at the probe token (reversed, rotated
  and 200 random orders tested). So the whole probe fixture, not only the probe token, rests
  on one premise: Earth Engine adds the double partial images in the order the graph adds
  them, as Milestone 10 found (formulation B: every EE tensor bit-identical to this mirror,
  819.7 M elements of inexact real data). The mirror is exact because it sums in that order,
  in double. A probe that fails live while the 8 group fixtures pass points to that premise,
  not to the block.

Channel counts are small (4 pooled channels; 8 input and 4 output channels for the groups):
channel width is not one of the settings P04 validates.
"""

from __future__ import annotations

from functools import partial

import numpy as np

from .core import Fixture, FixtureData, MirrorSpec, Readout, as_float32, dyadic, hwc, nchw
from .window import DRO, LANE, WINDOW, _dense, _echo, _layout, _metadata, _read, _seeded, _tokens, _windows

P04_SEED = 4000
"""Seeds 4001 onwards: disjoint from the calibration's 1-18, P02's 21-31 and P01's 1050-1125."""

P04_NEW = "New setting (P04): no milestone ran it in the library. The regression check is "


def _seed(index: int) -> int:
    return P04_SEED + index


# ---------------------------------------------------------------------------
# window_pool: extents and bins
# ---------------------------------------------------------------------------
POOL_CHANNELS = 4
POOL_COLUMNS = 5
"""The stacked [R, C] pooled value is read as an [R / 5, 5] grid of its rows (R = 50 or 85),
which keeps every readout within the window lane's 8-window side (40 tokens at window 5)."""

POOL_CASES = (            # (extent, window, sizes)
    (5, 5, (1, 2, 3, 6)),
    (8, 8, (1, 2, 3, 6)),
    (12, 12, (1, 2, 3, 6)),
    (14, 7, (1, 2, 3, 6)),
    (8, 8, (1, 2, 4, 8)),
    (12, 12, (1, 2, 4, 8)),
)
"""P04's extents and bins: one window of 5, 8 and 12 tokens and 2 x 2 windows of 7 with the
PSP's bins; the uneven bins (1, 2, 4, 8) on 8 x 8 and 12 x 12. The calibration's 7 x 7 with
(1, 2, 3, 6) is the regression."""


def _rows(sizes) -> int:
    return sum(s * s for s in sizes)


def _value_block(ee, value, rows: int, width: int, channels: int):
    """A V value [rows, channels] (float32 ee.Array) as ``channels`` float32 bands on the
    ``TokenGrid(rows / width, width)`` grid: pixel (a, b) holds row a width + b
    (``arraySlice`` + ``flatten_row``, as ``depth_to_space`` reads a window's row; values
    moved). Returns (image, grid)."""
    from tensor_ee.core.grid import TokenGrid
    from tensor_ee.layouts.window import flatten_row

    if rows % width:
        raise ValueError(f"{rows} rows do not fill a grid {width} wide")
    height = rows // width
    grid = TokenGrid(height, width)
    coords = ee.Image.pixelCoordinates(grid.projection(ee)).floor()
    a = coords.select("y").clamp(0, height - 1)
    b = coords.select("x").clamp(0, width - 1)
    row = a.multiply(width).add(b).toInt()
    image = flatten_row(ee.Image(value).arraySlice(0, row, row.add(1)), channels)
    return image, grid.pixel_grid()


def _block_layout(stacked: np.ndarray) -> np.ndarray:
    """The [R, C] pooled rows as the readout's [R / 5, 5, C] grid (row a 5 + b at (a, b))."""
    rows, channels = stacked.shape
    return stacked.reshape(rows // POOL_COLUMNS, POOL_COLUMNS, channels)


def _pool_data(seed: int, *, extent: int) -> FixtureData:
    return FixtureData(inputs={"x": dyadic(seed, "x", (extent, extent, POOL_CHANNELS))}, weights={})


def _pool_build(ee, data: FixtureData, *, sizes, window: int) -> list:
    from tensor_ee.blocks.window.pooling import window_pool

    extent = data.inputs["x"].shape[0]
    tokens = _tokens(extent // window, extent // window, window)
    dense = _dense(ee, tokens, data.inputs["x"])
    x = _windows(ee, tokens, dense, POOL_CHANNELS, window)
    pooled = window_pool(ee, x, tokens, POOL_CHANNELS, sizes=sizes, window=window, policy=DRO)
    image, grid = _value_block(ee, pooled.pooled, _rows(sizes), POOL_COLUMNS, POOL_CHANNELS)
    return [Readout("pooled", image, POOL_CHANNELS, grid), _echo(dense, tokens, POOL_CHANNELS)]


def _pool_reference(data: FixtureData, *, sizes) -> dict:
    import torch
    from segmentation_models_pytorch.decoders.upernet.decoder import PSPModule

    with _seeded():
        psp = PSPModule(POOL_CHANNELS, POOL_CHANNELS, sizes=tuple(sizes)).eval()
    x = nchw(data.inputs["x"]).contiguous(memory_format=torch.channels_last)
    rows = []
    with torch.no_grad():
        for block, s in zip(psp.blocks, sizes, strict=True):
            pool = block[0]                                          # nn.AdaptiveAvgPool2d(s)
            if not isinstance(pool, torch.nn.AdaptiveAvgPool2d) or pool.output_size != s:
                raise AssertionError("unexpected SMP PSP block")
            rows.append(pool(x)[0].permute(1, 2, 0).reshape(s * s, POOL_CHANNELS))
    return {"pooled": _block_layout(torch.cat(rows).numpy())}


def _pool_mirror(data: FixtureData, *, sizes, window: int) -> dict:
    from reference.upernet import ee_mirror_psp as MP

    pooled = MP.pool(data.inputs["x"], "window", sizes=sizes, window=window, source="index_maps")
    return {"pooled": as_float32(_block_layout(pooled), "pooled")}


def _pool_id(extent: int, window: int, sizes) -> str:
    return f"window/window_pool/e{extent}-bins{''.join(str(s) for s in sizes)}-w{window}"


def _pool_fixture(index: int, extent: int, window: int, sizes) -> Fixture:
    from tensor_ee.blocks.window import pooling as WO

    sizes = tuple(sizes)
    rows = _rows(sizes)
    windows = extent // window
    overlap = [f"{extent} -> {s}" for s in sizes if extent % s]
    return Fixture(
        id=_pool_id(extent, window, sizes), lane=LANE, block="window_pool",
        settings={"sizes": sizes, "window": window}, policy=DRO,
        shapes={"tokens": [extent, extent], "windows": [windows, windows], "channels": POOL_CHANNELS,
                "output": [rows, POOL_CHANNELS], "readout": [rows // POOL_COLUMNS, POOL_COLUMNS, POOL_CHANNELS]},
        seed=_seed(index), data=partial(_pool_data, extent=extent),
        build=partial(_pool_build, sizes=sizes, window=window), reference=partial(_pool_reference, sizes=sizes),
        reference_source=f"SMP 0.5.0 PSPModule(sizes={sizes}) AdaptiveAvgPool2d blocks, channels-last input",
        mirror=MirrorSpec(partial(_pool_mirror, sizes=sizes, window=window), True,
                          f"reference.upernet.ee_mirror_psp.pool (window order, window={window}, "
                          "bin tables from PyTorch on index maps)"),
        requests=2,
        constants=lambda data, extent=extent, sizes=sizes: {"window_pool": _metadata("window_pool")
                                                              .embedded_constants(extent, extent, sizes=sizes)},
        history=P04_NEW + "window/window_pool/bins1236-w7 (one 7 x 7 window). Milestone 8 pooled 7, 14, 28 "
                "and 56 tokens to (1, 2, 3, 6) at window 7 with the research code "
                "(docs/history/upernet/notes/milestone8_findings.md, answers 4-6).",
        accepts=(WO.window_pool,), window=window,
        notes=(f"{extent} x {extent} tokens ({windows} x {windows} window{'s' if windows > 1 else ''} of {window}) "
               f"pooled to {sizes}: {rows} stacked bins, read as a {rows // POOL_COLUMNS} x {POOL_COLUMNS} grid of rows. "
               + (f"Uneven bins: {', '.join(overlap)} (PyTorch's start/end rule; they overlap). " if overlap else
                  "Every bin divides the extent evenly. ")
               + "The mirror's bin tables are PyTorch's, read off adaptive_avg_pool2d on one-hot index maps, so a "
               "wrong bin rule in the library fails the mirror as well as SMP. Exact: every bin sum is a multiple "
               "of 2^-23 below 2^8, so exact in double in any order; one division and one float32 rounding.")
        + (" The only P04 pool on more than one window: the window partials are gathered with sampleRectangle "
           "and summed over the 4 windows." if windows > 1 else ""),
    )


# ---------------------------------------------------------------------------
# group_product: group counts and windows
# ---------------------------------------------------------------------------
GROUP_IN, GROUP_OUT = 8, 4
GROUP_COUNTS = (1, 2, 4, 8)
GROUP_WINDOWS = (7, 12)


def _group_data(seed: int, *, window: int, cout: int = GROUP_OUT) -> FixtureData:
    return FixtureData(
        inputs={"x": dyadic(seed, "x", (2 * window, 2 * window, GROUP_IN))},
        weights={"weight": dyadic(seed, "weight", (cout, GROUP_IN, 3, 3), scale=0.25)})


def _group_build(ee, data: FixtureData, *, groups: int, window: int) -> list:
    """The grouped double product: per group g (channels [g C / G, (g + 1) C / G)), its
    window-layout input, ``group_weight(..., g, groups)`` and ``group_product`` (double, no
    rounding); the partials added in double in group order 0 .. G - 1; one rounding."""
    from tensor_ee.blocks.window.conv import Conv3x3Constants, group_product, group_weight
    from tensor_ee.core.ops import WindowOps
    from tensor_ee.core.tensor import WindowTensor, band_names

    ops = WindowOps(ee)
    tokens = _tokens(2, 2, window)
    x, weight = data.inputs["x"], data.weights["weight"]
    width, cout = x.shape[2] // groups, weight.shape[0]
    dense = _dense(ee, tokens, x)
    terms = []
    for g in range(groups):
        names = [f"c{width * g + i}" for i in range(width)]
        part = WindowTensor(_windows(ee, tokens, dense.select(names, band_names(width)), width, window),
                            tokens, width, window=window)
        w = Conv3x3Constants(ops, group_weight(weight, g, groups=groups), rows=_layout(window), k=3)
        terms.append(group_product(ee, part, w, policy="grouped_double"))
    total = terms[0]
    for term in terms[1:]:
        total = total.add(term)                      # double + double: no rounding
    return [_read("out", ee, tokens, ops.to_float(total), cout, window), _echo(dense, tokens, x.shape[2])]


def _group_reference(data: FixtureData) -> dict:
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        return {"out": hwc(F.conv2d(nchw(data.inputs["x"]), torch.from_numpy(data.weights["weight"]),
                                    None, padding=1))}


def _group_mirror(data: FixtureData, *, groups: int) -> dict:
    from reference.upernet import ee_mirror_fpn as MF
    from reference.upernet import ee_mirror_fusion as MU
    from reference.upernet import smp_fusion_reference as R

    x, weight = data.inputs["x"], data.weights["weight"]
    width = x.shape[2] // groups
    terms = [MF.conv3x3(x[..., width * g:width * (g + 1)], R.group_weight(weight, g, groups), exact=True)
             for g in range(groups)]
    return {"out": as_float32(MU.sum_partials(terms, tuple(range(groups)), "sequential", "double"), "out")}


def _group_constants(data: FixtureData, *, groups: int) -> dict:
    cout = data.weights["weight"].shape[0]
    return {"group_product": groups * _metadata("group_product").embedded_constants(
        GROUP_IN // groups, cout, k=3)}


def _group_fixture(index: int, groups: int, window: int, *, seed: int | None = None,
                   history: str | None = None) -> Fixture:
    from tensor_ee.blocks.window import conv as WC
    from tensor_ee.weights.constants import WindowLayout

    width = GROUP_IN // groups
    return Fixture(
        id=f"window/group_product/g{groups}-w{window}", lane=LANE, block="group_product",
        settings={"groups": groups, "k": 3, "window": window}, policy="grouped_double",
        shapes={"tokens": [2 * window, 2 * window], "in": GROUP_IN, "group_in": width, "out": GROUP_OUT,
                "bias": False},
        seed=_seed(index) if seed is None else seed, data=partial(_group_data, window=window),
        build=partial(_group_build, groups=groups, window=window), reference=_group_reference,
        reference_source="torch.nn.functional.conv2d(padding=1) of the same unsplit weight",
        mirror=MirrorSpec(partial(_group_mirror, groups=groups), True,
                          f"reference.upernet.ee_mirror_fusion.sum_partials (order 0..{groups - 1}, double) "
                          f"of ee_mirror_fpn.conv3x3 per group (double); {groups} group(s)"),
        requests=1, constants=partial(_group_constants, groups=groups),
        history=history or (P04_NEW + "window/group_product/g4-w7. Milestone 10 ran the 4 groups of the fusion "
                            "(docs/history/upernet/notes/milestone10_findings.md, section 4)."),
        accepts=(WC.group_weight, WC.Conv3x3Constants, WindowLayout), window=window,
        notes=(f"{groups} contiguous group(s) of {width} input channel(s), K = {9 * width} per group; the "
               f"partials summed in double in group order 0..{groups - 1}, one float32 rounding. "
               "The 2 x 2-window raster makes every halo cross a window edge or the zero border. Exact: every "
               "product is a multiple of 2^-48 and every partial sum is below 18 < 2^5, so each group's dot "
               "and the group sum are exact in double in any order; one rounding."
               + (" One group: the product is the unsplit convolution's im2col product." if groups == 1 else "")),
    )


# ---------------------------------------------------------------------------
# Order probes: the group sum's order at 4 and 8 groups
# ---------------------------------------------------------------------------
PROBE_TOKEN = (WINDOW, WINDOW)
"""The probe token: (7, 7), the first token of window (1, 1)."""

PROBE_PARTIALS = {
    4: (-1.0, -1.5 * 2.0 ** -52, 1.0, 1.5 * 2.0 ** -55),
    8: (2.0 ** -60, 2.0 ** -55, -0.5, 0.5, 2.0 ** -49, 2.0 ** -63, -(2.0 ** -56), -1.5 * 2.0 ** -44),
}
"""Each group's partial at the probe token, in group order. Summed in double in this order
and rounded once, they give a float32 value that another order (or rounding each partial
first) does not (``tests/test_p04_decoder_blocks.py`` enumerates the orders). Each is
m 2^e with m in {1, 1.5}, made as one product: input m at the probe token in the group's
first channel, weight +-2^e at the centre tap."""


def _probe_data(seed: int, *, groups: int) -> FixtureData:
    """Random inputs, but at the probe token the first channel of each group holds m; one
    output channel whose weight is +-2^e at the centre tap of each group's first channel and
    0 elsewhere."""
    data = _group_data(seed, window=WINDOW, cout=1)
    x = data.inputs["x"].copy()
    weight = np.zeros((1, GROUP_IN, 3, 3), np.float32)
    width = GROUP_IN // groups
    for g, value in enumerate(PROBE_PARTIALS[groups]):
        mantissa, exponent = np.frexp(abs(value))           # value = m 2^e, m in [1, 2)
        mantissa, exponent = 2 * mantissa, exponent - 1
        x[PROBE_TOKEN[0], PROBE_TOKEN[1], width * g] = mantissa
        weight[0, width * g, 1, 1] = np.copysign(np.ldexp(1.0, exponent), value)
    if not all(np.float64(x[PROBE_TOKEN[0], PROBE_TOKEN[1], width * g]) * np.float64(weight[0, width * g, 1, 1])
               == value for g, value in enumerate(PROBE_PARTIALS[groups])):
        raise AssertionError("the probe's products are not its partials")
    return FixtureData(inputs={"x": x}, weights={"weight": weight})


def _probe_fixture(index: int, groups: int) -> Fixture:
    from tensor_ee.blocks.window import conv as WC
    from tensor_ee.weights.constants import WindowLayout

    width = GROUP_IN // groups
    return Fixture(
        id=f"window/group_product/order-g{groups}-w{WINDOW}", lane=LANE, block="group_product",
        settings={"groups": groups, "k": 3, "window": WINDOW}, policy="grouped_double",
        shapes={"tokens": [2 * WINDOW, 2 * WINDOW], "in": GROUP_IN, "group_in": width, "out": 1, "bias": False,
                "probe": list(PROBE_TOKEN)},
        seed=_seed(index), data=partial(_probe_data, groups=groups),
        build=partial(_group_build, groups=groups, window=WINDOW), reference=_group_reference,
        reference_source="torch.nn.functional.conv2d(padding=1) of the same unsplit (designed) weight",
        mirror=MirrorSpec(partial(_group_mirror, groups=groups), True,
                          f"reference.upernet.ee_mirror_fusion.sum_partials (order 0..{groups - 1}, double) "
                          f"of ee_mirror_fpn.conv3x3 per group (double); {groups} groups"),
        requests=1, constants=partial(_group_constants, groups=groups),
        history="New fixture (P04): Milestone 10 found formulation B insensitive to the group order on its data "
                "(docs/history/upernet/notes/milestone10_findings.md, section 4, order study); this probe makes "
                "the order visible.",
        accepts=(WC.group_weight, WC.Conv3x3Constants, WindowLayout), window=WINDOW,
        notes=(f"One output channel, weights designed: at token {PROBE_TOKEN} the {groups} group partials are "
               f"{', '.join(float(v).hex() for v in PROBE_PARTIALS[groups])} (one exact product each), and their "
               "double sum in group order, rounded once, differs from other orders and from rounding each partial. "
               "Exact: each group's partial is exact (one product), but the double group sum is inexact at the "
               "probe token and at about a third of the others; on this data the float32 result depends on the "
               "order only at the probe token. The mirror is exact because it adds the partials in the graph's "
               "order, in double, so the whole fixture rests on the premise that Earth Engine adds the double "
               "partial images in that order (Milestone 10, formulation B). If a probe fails while the 8 group "
               "fixtures pass, the premise failed, not the block."),
    )


# ---------------------------------------------------------------------------
# C0b: window 8, the capstone's window, for group_product and the two resizes
# ---------------------------------------------------------------------------
C0B_WINDOW = 8
C0B_SEED = 5000
"""Seeds 5001 onwards: disjoint from the calibration's 1-18, P02's 21-31, P01's 1050-1125,
P03's 3001-3006 and P04's 4001-4022."""

C0B_NEW = ("New setting (C0b): window 8, the capstone's (docs/tasks/C0-capstone.md, the status audit). "
           "Window 7 is the reference model's: ")

RESIZE_CHANNELS = 2
"""Channels of the resize fixtures: channel width is not a resize setting."""

HEAD_BLOCK = 64
"""The side of a request at window 8 (8 windows): the head's 256 x 256 output is read as its
four diagonal 64 x 64 blocks."""


def _c0b_seed(index: int) -> int:
    return C0B_SEED + index


class ResizeCase:
    """A ``window_resize`` fixture at window 8: an ``extent`` x ``extent`` token map, resized by
    ``factor`` (bilinear, ``align_corners``), on the capstone's path ``use``."""

    def __init__(self, extent: int, factor: int, align_corners: bool, use: str, seed: int):
        self.extent, self.factor, self.align_corners, self.use, self.seed = extent, factor, align_corners, use, seed

    @property
    def out(self) -> int:
        return self.factor * self.extent

    @property
    def id(self) -> str:
        corners = "ac-" if self.align_corners else ""
        return f"window/window_resize/{corners}f{self.factor}-e{self.extent}-w{C0B_WINDOW}"

    @property
    def blocks(self) -> int:
        """Diagonal 64 x 64 output blocks read (1: the whole output)."""
        return self.out // HEAD_BLOCK if self.out > HEAD_BLOCK else 1


RESIZE_CASES = (
    ResizeCase(8, 2, False, "the FPN: the PSP output (C4, 8 x 8 tokens) onto C3's 16 x 16", _c0b_seed(2)),
    ResizeCase(16, 2, False, "the FPN: F3 (16 x 16) onto C2's 32 x 32", _c0b_seed(3)),
    ResizeCase(32, 2, False, "the FPN: F2 (32 x 32) onto C1's 64 x 64; the fusion: P2 onto C1", _c0b_seed(4)),
    ResizeCase(16, 4, False, "the fusion: P3 (16 x 16) onto C1's 64 x 64", _c0b_seed(5)),
    ResizeCase(8, 8, False, "the fusion: the PSP output (8 x 8) onto C1's 64 x 64", _c0b_seed(6)),
    ResizeCase(64, 4, True, "the head: C1's 64 x 64 logits onto the 256 x 256 output (UpsamplingBilinear2d)",
               _c0b_seed(7)),
)
"""Every resize of the capstone at window 8, at its own token extents."""

MATRIX_EXTENT = 8
"""The PSP's extent: C4's 8 x 8 tokens, one window of 8."""

MATRIX_SIZES = (1, 2, 3, 6)
"""The PSP's bins: the pooled s x s branches resized back onto C4."""


def _block_grid(tokens, row: int, col: int, side: int) -> dict:
    """computePixels grid of the side x side block of ``tokens`` at token (row, col)."""
    s = tokens.scale
    return {"dimensions": {"width": side, "height": side},
            "affineTransform": {"scaleX": s, "shearX": 0, "translateX": 0.0 + s * col,
                                "shearY": 0, "scaleY": -s, "translateY": 0.0 - s * row},
            "crsCode": tokens.crs}


def _resize_readouts(full: np.ndarray, case: ResizeCase) -> dict:
    """The full [out, out, C] output as the fixture's readouts: "out", or the diagonal blocks
    "out_b0".. of the head."""
    if case.blocks == 1:
        return {"out": full}
    b = HEAD_BLOCK
    return {f"out_b{i}": full[b * i:b * (i + 1), b * i:b * (i + 1)] for i in range(case.blocks)}


def _resize_data(seed: int, *, case: ResizeCase) -> FixtureData:
    return FixtureData(inputs={"x": dyadic(seed, "x", (case.extent, case.extent, RESIZE_CHANNELS))}, weights={})


def _resize_build(ee, data: FixtureData, *, case: ResizeCase) -> list:
    from tensor_ee.blocks.window.resample import window_resize
    from tensor_ee.layouts.window import to_dense

    w, c = C0B_WINDOW, RESIZE_CHANNELS
    tokens = _tokens(case.extent // w, case.extent // w, w)
    dense = _dense(ee, tokens, data.inputs["x"])
    x = _windows(ee, tokens, dense, c, w)
    image, fine = window_resize(ee, x, tokens, c, case.factor, case.align_corners, window=w, policy=DRO)
    if case.blocks == 1:
        out = [_read("out", ee, fine, image, c, w)]
    else:
        values = to_dense(ee, fine, image, c, window=w)
        out = [Readout(f"out_b{i}", values, c, _block_grid(fine, HEAD_BLOCK * i, HEAD_BLOCK * i, HEAD_BLOCK))
               for i in range(case.blocks)]
    return out + [_echo(dense, tokens, c)]


def _resize_reference(data: FixtureData, *, case: ResizeCase) -> dict:
    """``F.interpolate`` as the capstone calls it: SMP's resizes with ``size=`` and
    align_corners=False; the head's ``UpsamplingBilinear2d`` with ``scale_factor=`` and
    align_corners=True."""
    import torch
    import torch.nn.functional as F

    x = nchw(data.inputs["x"])
    with torch.no_grad():
        if case.align_corners:
            y = F.interpolate(x, scale_factor=case.factor, mode="bilinear", align_corners=True)
        else:
            y = F.interpolate(x, size=(case.out, case.out), mode="bilinear", align_corners=False)
    return _resize_readouts(hwc(y), case)


def resize_weights(n_in: int, n_out: int, align_corners: bool) -> np.ndarray:
    """PyTorch's own 1-D weights [n_out, n_in], read off ``F.interpolate`` (float32) on one-hot
    inputs: ``reference.upernet.helpers.torch_resize_matrix``, no library code."""
    from reference.upernet.helpers import torch_resize_matrix

    return torch_resize_matrix(n_in, n_out, align_corners)


def _resize_mirror(data: FixtureData, *, case: ResizeCase) -> dict:
    from reference.upernet import ee_mirror_head as MH

    m = resize_weights(case.extent, case.out, case.align_corners)
    full = MH.upsample(data.inputs["x"], factor=case.factor, align_corners=case.align_corners, weights=(m, m))
    return {k: as_float32(v, k) for k, v in _resize_readouts(full, case).items()}


def _resize_constants(data: FixtureData, *, case: ResizeCase) -> dict:
    from tensor_ee.blocks.window.resample import METADATA

    return {"window_resize": METADATA["window_resize"].embedded_constants(
        case.extent, case.extent, case.factor, align_corners=case.align_corners, window=C0B_WINDOW)}


def _resize_fixture(case: ResizeCase) -> Fixture:
    from tensor_ee.blocks.window import resample as RS

    w = C0B_WINDOW
    windows = case.extent // w
    readout = ([case.out, case.out, RESIZE_CHANNELS] if case.blocks == 1
               else [case.blocks, HEAD_BLOCK, HEAD_BLOCK, RESIZE_CHANNELS])
    if case.align_corners:
        exact = ("Exact: the float32 weights are PyTorch's; every y-stage product is exact in double and each "
                 "output adds two of them, one rounding in any order. The x stage's rounded products and "
                 "rounded sum (the mirror; Milestone 11 found Earth Engine does not fuse) and both fused "
                 "multiply-adds round to the same float32 at every output on this data "
                 "(tests/test_c0b_window8.py), so the flag holds in any order, fused or not.")
        reads = (f"The {case.out} x {case.out} output is read as its {case.blocks} diagonal {HEAD_BLOCK} x "
                 f"{HEAD_BLOCK} blocks (the window lane's side limit at window 8 is 64): together they hold every "
                 f"child-window row and column, so every local matrix of both axes. ")
    else:
        exact = (f"Exact: the inputs are multiples of 2^-23 in [-1, 1) and the weights multiples of "
                 f"1/{2 * case.factor}, so every product and sum of both stages is exact in double (tested); one "
                 "float32 rounding, in any order, fused or not.")
        reads = ""
    return Fixture(
        id=case.id, lane=LANE, block="window_resize",
        settings={"factor": case.factor, "align_corners": case.align_corners, "window": w}, policy=DRO,
        shapes={"tokens": [case.extent, case.extent], "windows": [windows, windows],
                "out": [case.out, case.out], "channels": RESIZE_CHANNELS, "readout": readout},
        seed=case.seed, data=partial(_resize_data, case=case), build=partial(_resize_build, case=case),
        reference=partial(_resize_reference, case=case),
        reference_source=(f"torch.nn.functional.interpolate(scale_factor={case.factor}, bilinear, align_corners=True)"
                          if case.align_corners else
                          f"torch.nn.functional.interpolate(size=({case.out}, {case.out}), bilinear, "
                          "align_corners=False)"),
        mirror=MirrorSpec(partial(_resize_mirror, case=case), True,
                          "reference.upernet.ee_mirror_head.upsample (y, then x with rounded products and no "
                          "fused multiply-add) with PyTorch's weights read off F.interpolate on one-hot inputs "
                          "(reference.upernet.helpers.torch_resize_matrix)"),
        requests=case.blocks + 1, constants=partial(_resize_constants, case=case),
        history=C0B_NEW + ("the head's x4 (56 -> 224) ran in Milestones 11 and 12 "
                           "(docs/history/upernet/notes/milestone11_findings.md)." if case.align_corners else
                           "the FPN's and fusion's resizes ran in Milestones 7, 10 and 12 "
                           "(docs/history/upernet/notes/milestone7_findings.md)."),
        accepts=(RS.window_resize,), window=w,
        notes=(f"{case.extent} x {case.extent} tokens ({windows} x {windows} window{'s' if windows > 1 else ''} of 8) "
               f"x{case.factor}, align_corners={case.align_corners}: {case.use}. " + reads + exact),
    )


def _matrix_data(seed: int, *, size: int) -> FixtureData:
    return FixtureData(inputs={"x": dyadic(seed, "x", (size, size, RESIZE_CHANNELS))}, weights={})


def _matrix_build(ee, data: FixtureData, *, size: int) -> list:
    """The pooled s x s branch as a V value [s^2, C] (the input's float32 decimals, as the
    library embeds a constant), resized onto one 8 x 8 window. The V value itself is echoed as
    an s x s grid of its rows."""
    from tensor_ee.blocks.window.resample import window_matrix_resize
    from tensor_ee.weights.constants import constant_value

    c = RESIZE_CHANNELS
    value = constant_value(ee, data.inputs["x"].reshape(size * size, c))
    tokens = _tokens(MATRIX_EXTENT // C0B_WINDOW, MATRIX_EXTENT // C0B_WINDOW, C0B_WINDOW)
    out = window_matrix_resize(ee, value, (size, size), tokens, window=C0B_WINDOW, policy=DRO)
    echo, grid = _value_block(ee, value, size * size, size, c)
    return [_read("out", ee, tokens, out, c, C0B_WINDOW), Readout("input", echo, c, grid, "input", "x")]


def _matrix_reference(data: FixtureData) -> dict:
    """``F.interpolate(size=(8, 8), bilinear, align_corners=False)``, as SMP's PSPModule resizes
    each pooled branch."""
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        y = F.interpolate(nchw(data.inputs["x"]), size=(MATRIX_EXTENT, MATRIX_EXTENT), mode="bilinear",
                          align_corners=False)
    return {"out": hwc(y)}


def matrix_mirror(x: np.ndarray, out: int = MATRIX_EXTENT) -> np.ndarray:
    """Earth Engine's matrix resize of x [s, s, C] onto out x out: per output token one
    sequential double dot over the row-major source pixels with the products of PyTorch's
    float32 1-D weights (read off F.interpolate on one-hot inputs; exact in double), one
    float32 rounding (float32 values in float64)."""
    s, c = x.shape[0], x.shape[2]
    m = resize_weights(s, out, False)
    weights = np.einsum("yi,xj->yxij", m, m).reshape(out * out, s * s)
    source = np.asarray(x, np.float64).reshape(s * s, c)
    total = np.zeros((out * out, c))
    for k in range(s * s):
        total = total + weights[:, k:k + 1] * source[k][None, :]
    return total.reshape(out, out, c).astype(np.float32).astype(np.float64)


def _matrix_mirror(data: FixtureData) -> dict:
    return {"out": as_float32(matrix_mirror(data.inputs["x"]), "out")}


def _matrix_constants(data: FixtureData, *, size: int) -> dict:
    from tensor_ee.blocks.window.resample import METADATA

    return {"window_matrix_resize": METADATA["window_matrix_resize"].embedded_constants(
        (size, size), MATRIX_EXTENT, MATRIX_EXTENT)}


def _matrix_fixture(index: int, size: int) -> Fixture:
    from tensor_ee.blocks.window import resample as RS

    return Fixture(
        id=f"window/window_matrix_resize/s{size}-e{MATRIX_EXTENT}-w{C0B_WINDOW}", lane=LANE,
        block="window_matrix_resize",
        settings={"in_size": (size, size), "align_corners": False, "window": C0B_WINDOW}, policy=DRO,
        shapes={"source": [size * size, RESIZE_CHANNELS], "tokens": [MATRIX_EXTENT, MATRIX_EXTENT],
                "windows": [1, 1], "channels": RESIZE_CHANNELS},
        seed=_c0b_seed(index), data=partial(_matrix_data, size=size), build=partial(_matrix_build, size=size),
        reference=_matrix_reference,
        reference_source=f"torch.nn.functional.interpolate(size=({MATRIX_EXTENT}, {MATRIX_EXTENT}), bilinear, "
                         "align_corners=False)",
        mirror=MirrorSpec(_matrix_mirror, True,
                          "tools.fixtures.decoder.matrix_mirror: a sequential double dot per output with the "
                          "products of PyTorch's 1-D weights read off F.interpolate on one-hot inputs "
                          "(reference.upernet.helpers.torch_resize_matrix), one rounding"),
        requests=2, constants=partial(_matrix_constants, size=size),
        history=C0B_NEW + "the PSP's branches (1, 2, 3, 6 -> 7 x 7) ran in Milestones 8, 10 and 12 "
                "(docs/history/upernet/notes/milestone8_findings.md).",
        accepts=(RS.window_matrix_resize,), window=C0B_WINDOW,
        notes=(f"The PSP's {size} x {size} branch onto C4's 8 x 8 tokens (one window of 8), "
               "align_corners=False. Exact: the inputs are multiples of 2^-23 in [-1, 1), the 2-D weights "
               "products of 1-D weights that are multiples of 1/16, so every product is a multiple of 2^-31 "
               "and every partial sum at most 1 in magnitude: exact in double in any order (tested); one "
               "float32 rounding."),
    )


C0B_REGRESSION = ("window/group_product/g4-w7",)
"""Window 7 in the c0b selection: the calibration's group_product (4 groups). No calibration
fixture resizes."""


def c0b_fixtures() -> list[Fixture]:
    group = _group_fixture(0, 4, C0B_WINDOW, seed=_c0b_seed(1),
                           history=C0B_NEW + "window/group_product/g4-w7 (calibration) and P04's windows 7 and 12 "
                                   "(docs/tasks/P04-decoder-blocks.md); the capstone's fusion runs 4 groups at "
                                   "window 8.")
    resizes = [_resize_fixture(case) for case in RESIZE_CASES]
    matrices = [_matrix_fixture(10 + i, s) for i, s in enumerate(MATRIX_SIZES)]
    return [group] + resizes + matrices


# ---------------------------------------------------------------------------
# The registry and the selections
# ---------------------------------------------------------------------------
P04_REGRESSION = ("window/window_pool/bins1236-w7", "window/group_product/g4-w7")
"""Window 7 in the p04 selection: the calibration's window_pool (one 7 x 7 window, bins
(1, 2, 3, 6)) and group_product (4 groups) fixtures."""


def p04_fixtures() -> list[Fixture]:
    pools = [_pool_fixture(i + 1, *case) for i, case in enumerate(POOL_CASES)]
    cases = [(g, w) for w in GROUP_WINDOWS for g in GROUP_COUNTS if (g, w) != (4, WINDOW)]
    groups = [_group_fixture(11 + i, g, w) for i, (g, w) in enumerate(cases)]
    probes = [_probe_fixture(21 + i, g) for i, g in enumerate(sorted(PROBE_PARTIALS))]
    return pools + groups + probes


def fixtures() -> list[Fixture]:
    return p04_fixtures() + c0b_fixtures()


def _p04() -> tuple[str, ...]:
    return P04_REGRESSION + tuple(f.id for f in p04_fixtures())


def _c0b() -> tuple[str, ...]:
    return C0B_REGRESSION + tuple(f.id for f in c0b_fixtures())


SELECTIONS = {"p04": _p04, "c0b": _c0b}
"""Named selections of this module (``tools.fixtures.select``): each one run, within the budget."""
