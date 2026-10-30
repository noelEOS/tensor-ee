"""R <-> W converters, halos and im2col.

Moved from ``ee_attention/earth_engine/spatial_attention.py`` (``as_row``,
``flatten_row``, ``stride_samples``, ``stack_tokens``),
``swin_hierarchy/earth_engine/patch_merge.py`` (``depth_to_space``, the
production copy with the non-negative in-window index),
``swin_image_encoder/earth_engine/patch_embed.py`` and
``upernet/earth_engine/layout.py`` (``to_windows``, consolidated through an
optional domain mask; ``zero_outside``, ``neighbours``, ``halo``, ``taps``,
``im2col``, ``to_dense`` and their helpers ``zeros`` and ``reshape``), research
code archived at the tag ``research-archive-2026-10-12`` (docs/history/README.md).
``layout.to_windows`` is ``from_dense`` here (dense bands in, the masked
``to_windows`` out).

W layout (window ``w``): one ``[w**2, C]`` float32 array per pixel of the
w-times-coarser grid, token ``t = w * a + b`` of window ``(I, J)`` being pixel
``(w * I + a, w * J + b)`` of the token grid, channel ``c`` in column ``c``.
Every function only moves values.

Window, ring and kernel arguments default to today's values (7, 1, 3). Through the
window blocks, windows 5, 7, 8 and 12 are validated with ring 1 and kernel 3;
other values are not. Through the patch embedding, ``stride_samples`` is validated at
patches 2, 4 and 8.

``neighbours`` and ``halo`` take a keyword-only ``crop`` (default ``None``: the whole raster is
one zero-padding domain, today's graph). ``crop=k`` makes every k x k block of windows an
independent crop: a neighbour window in another crop reads as zeros, exactly as outside the
raster, so a grid of crops gives each crop's halo as if it ran alone. Masks only; it has not
been validated live.
"""

from __future__ import annotations

import numpy as np

from ..core.errors import reject
from ..core.grid import TokenGrid
from ..core.ops import image_reshape
from ..core.tensor import WINDOW, band_names
from .shift import domain_mask, materialize, projection, translate

RING = 1
"""Default halo ring width in tokens (3 x 3 kernels); the only one validated."""


# ---------------------------------------------------------------------------
# Rows and space-to-depth (window-attention validation)
# ---------------------------------------------------------------------------
def as_row(ee, image, channels: int):
    """d scalar bands -> one [1, d] array band (the core W reshape, ``image_reshape``)."""
    return image_reshape(ee, image.toArray(), [1, channels])


def flatten_row(output_row, channels: int):
    """[1, d] array band -> d scalar bands c0..c{d-1}."""
    return output_row.arrayProject([1]).arrayFlatten([band_names(channels)])


def stride_samples(ee, grid: TokenGrid, image, window: int, materialize: bool = True):
    """List of N images on the coarse grid; element a*w+b holds fine pixel
    (w*I + a, w*J + b) of the window (I, J), following the tiling code's
    _stride_sample: sample exactly at fine centers, then relabel with changeProj.

    The factor ``window`` is the window of a partition (5, 7, 8 and 12 validated),
    the patch of a patch embedding (2, 4 and 8 validated; also checked offline at 3)
    or 2 for patch merging. The sample offset ``s (b + 0.5 - w/2)`` puts every sample on a
    fine-pixel centre for odd and even factors alike.
    """
    coarse = grid.coarse(window)
    coarse_projection = ee.Projection(grid.crs, coarse.transform)
    if materialize:
        # As for any token raster produced by an earlier layer on the fine grid.
        image = image.reproject(grid.projection(ee))
    s = grid.scale
    samples = []
    for a in range(window):
        for b in range(window):
            sample = ee.Projection(grid.crs, [s * window, 0, s * (b + 0.5 - window / 2), 0,
                                              -s * window, -s * (a + 0.5 - window / 2)])
            samples.append(image.reproject(sample).changeProj(sample, coarse_projection))
    return samples


def stack_tokens(ee, samples: list, start: int, end: int):
    """Tokens start..end-1 (each a [1, d] array band) -> [end-start, d]."""
    parts = [samples[t].rename(f"t{t}") for t in range(start, end)]
    return ee.Image.cat(parts).toArray(0)


# ---------------------------------------------------------------------------
# R <-> W
# ---------------------------------------------------------------------------
def to_windows(ee, rows_image, tokens: TokenGrid, materialize: bool = True, *,
               window: int = WINDOW, mask: bool = False):
    """[1, C] rows on the token grid -> [w**2, C] per window on the coarse grid
    (the original attention implementation's w**2 exact stride samples, stacked in token order w a + b).

    mask=True keeps only the raster's windows (``shift.domain_mask`` of
    ``tokens.coarse(window)``), as ``upernet.earth_engine.layout.to_windows``
    does; the default leaves the result unmasked, as patch embedding does.
    Windows 5, 7, 8 and 12 are validated.
    """
    samples = stride_samples(ee, tokens, rows_image, window, materialize=materialize)
    x = stack_tokens(ee, samples, 0, window * window)
    if mask:
        x = x.updateMask(domain_mask(ee, tokens.coarse(window)))
    return x


def from_dense(ee, tokens: TokenGrid, dense, channels: int, *, window: int = WINDOW):
    """C scalar bands on the token grid -> window layout (w**2 exact stride samples),
    masked outside the raster (the image encoder's stage3_from_image without the relabel).

    ``upernet.earth_engine.layout.to_windows``: select c0.., ``as_row``, then the
    masked ``to_windows``. Windows 5, 7, 8 and 12 are validated.
    """
    rows = as_row(ee, dense.select(band_names(channels)), channels)
    return to_windows(ee, rows, tokens, materialize=True, window=window, mask=True)


def depth_to_space(ee, grid: TokenGrid, window_output, window: int, channels: int):
    """The attention layout's depth_to_space (spatial_attention.depth_to_space) with a
    non-negative in-window index.  EE's mod takes the dividend's sign, so the original attention implementation's
    index is negative above and left of the raster, where the hierarchy's ring
    windows are (live: those windows came back masked); inside the raster the
    two are identical."""
    coarse = grid.coarse(window)
    fine = grid.projection(ee)
    upsampled = window_output.reproject(ee.Projection(grid.crs, coarse.transform)).reproject(fine)
    coords = ee.Image.pixelCoordinates(fine).floor()
    a, b = (coords.select(axis).mod(window).add(window).mod(window) for axis in ("y", "x"))
    row = a.multiply(window).add(b).toInt()
    return flatten_row(upsampled.arraySlice(0, row, row.add(1)), channels)


def to_dense(ee, tokens: TokenGrid, windows, channels: int, *, window: int = WINDOW):
    """Window layout -> C float32 scalar bands c0.. on the token grid (the Swin hierarchy's
    depth_to_space; values moved). Windows 5, 7, 8 and 12 are validated."""
    return depth_to_space(ee, tokens, windows, window, channels).toFloat()


# ---------------------------------------------------------------------------
# Halos: the window and its ring from the 8 neighbouring windows
# ---------------------------------------------------------------------------
def zeros(ee, rows: int, cols: int, depth: int | None = None):
    """A float32 zero array constant built on the server (no values in the request)."""
    shape = [rows, cols] if depth is None else [rows, cols, depth]
    count = int(np.prod(shape))
    return ee.Image(ee.Array(ee.List.repeat(0, count)).reshape(shape).float())


def reshape(ee, image, shape: list[int]):
    # The one W reshape expression lives in core; WindowOps.reshape uses it too.
    return image_reshape(ee, image, shape)


def zero_outside(ee, x, coarse: TokenGrid, channels: int, *, window: int = WINDOW):
    """Window layout with exact [w**2, C] zero arrays outside the raster, pinned to the grid."""
    inside = x.updateMask(domain_mask(ee, coarse))
    return materialize(ee, inside.unmask(zeros(ee, window * window, channels), False), coarse)


def check_crop(crop, coarse: TokenGrid):
    """``crop`` is None or a positive integer number of windows dividing both extents of the
    window grid ``coarse``; anything else is refused (client-side; the graph is unchanged)."""
    if crop is None:
        return None
    if isinstance(crop, bool) or not isinstance(crop, (int, np.integer)) or crop < 1:
        reject(f"crop must be None or a positive number of windows; received {crop!r}")
    if coarse.height % crop or coarse.width % crop:
        reject(f"crop {crop} does not divide the {coarse.height} x {coarse.width} window grid")
    return int(crop)


def same_crop(ee, coarse: TokenGrid, crop: int, dx: int, dy: int):
    """1 where window (I + dy, J + dx) is in the crop of window (I, J): crops of crop x crop
    windows from the grid origin. A mask only."""
    coords = ee.Image.pixelCoordinates(projection(ee, coarse)).floor()
    inside = None
    for axis, step in (("x", dx), ("y", dy)):
        if step:
            edge = crop - 1 if step > 0 else 0
            term = coords.select(axis).mod(crop).neq(edge)
            inside = term if inside is None else inside.And(term)
    return inside


def neighbours(ee, x, coarse: TokenGrid, channels: int, *, window: int = WINDOW,
               crop: int | None = None) -> dict:
    """{(dy, dx): [w, w, C] window at (I + dy, J + dx)}, zeros outside the raster, and with
    ``crop`` also zeros where the neighbour is in another crop of crop x crop windows."""
    crop = check_crop(crop, coarse)
    base = reshape(ee, zero_outside(ee, x, coarse, channels, window=window),
                   [window, window, channels])
    n = {(dy, dx): base if dy == dx == 0 else translate(ee, base, coarse, dx, dy)
         for dy in (-1, 0, 1) for dx in (-1, 0, 1)}
    if crop is None:
        return n
    empty = zeros(ee, window, window, channels)
    return {(dy, dx): image if dy == dx == 0 else
            image.updateMask(same_crop(ee, coarse, crop, dx, dy)).unmask(empty, False)
            for (dy, dx), image in n.items()}


def halo(ee, x, coarse: TokenGrid, channels: int, *, window: int = WINDOW, ring: int = RING,
         crop: int | None = None):
    """[w + 2r, w + 2r, C] per window: the window and its r-token ring (zeros outside
    the raster, and outside the window's crop when ``crop`` is given). Ring 1 is validated
    at windows 5, 7, 8 and 12; ``crop`` is not validated live."""
    n = neighbours(ee, x, coarse, channels, window=window, crop=crop)
    last, first = (window - ring, window), (0, ring)

    def part(key, rows, cols):
        return n[key].arraySlice(0, *rows).arraySlice(1, *cols)

    full = (0, window)
    top = part((-1, -1), last, last).arrayCat(part((-1, 0), last, full), 1).arrayCat(
        part((-1, 1), last, first), 1)
    middle = part((0, -1), full, last).arrayCat(n[(0, 0)], 1).arrayCat(part((0, 1), full, first), 1)
    bottom = part((1, -1), first, last).arrayCat(part((1, 0), first, full), 1).arrayCat(
        part((1, 1), first, first), 1)
    return top.arrayCat(middle, 0).arrayCat(bottom, 0)


def taps(ee, halo3, channels: int, k: int = 3, *, window: int = WINDOW) -> list:
    """[w**2, C] per kernel tap (ky, kx), row-major: tap (ky, kx) of output token (a, b) is
    feature pixel (w I + a + ky - 1, w J + b + kx - 1). k 3 is validated at windows 5, 7, 8
    and 12."""
    out = []
    for ky in range(k):
        for kx in range(k):
            block = halo3.arraySlice(0, ky, ky + window).arraySlice(1, kx, kx + window)
            out.append(reshape(ee, block, [window * window, channels]))
    return out


def im2col(ee, halo3, channels: int, k: int = 3, *, window: int = WINDOW):
    """[w**2, k*k C] per window in (ky, kx, c) column order (values moved).
    k 3 is validated at windows 5, 7, 8 and 12."""
    parts = taps(ee, halo3, channels, k, window=window)
    result = parts[0]
    for part in parts[1:]:
        result = result.arrayCat(part, 1)
    return result
