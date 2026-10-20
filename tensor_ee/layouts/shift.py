"""W-layout translation, Swin's cyclic shift, periodic extension and domain masks.

Moved from ``swin_pair/earth_engine/shift.py`` (``_projection`` as
``projection``, ``translate``, ``materialize``, ``cyclic_neighbor``,
``_roll_axis`` as ``roll_axis``, ``roll_windows``, ``shift_windows``,
``unshift_windows``) and ``swin_hierarchy/earth_engine/periodic_extension.py``
(``periodic_extension``, ``domain_mask``), with their private helpers. The
research modules, archived at the tag ``research-archive-2026-10-12``
(docs/history/README.md), keep the derivations in their docstrings.

Window layout: a raster on the coarse grid (``window`` times coarser than the
token grid) whose pixel (I, J) holds window (I, J) as a ``[window**2, C]``
array, row ``t = window * a + b`` for token ``(window * I + a, window * J + b)``.
Coarse neighbours are read through ``changeProj``, an exact integer relabeling
of the coarse grid, so every function here moves float32 values unchanged.

timm's SW-MSA roll by ``(-shift, -shift)`` is ``shift_windows`` and its inverse
``unshift_windows``. The window across the raster edge is the translated
neighbour (the periodic wrap), which is exact when the image is periodic with
the grid's window counts; ``periodic_extension`` supplies that periodicity by
mosaicking translated copies. The measured research alternatives (the
``"mosaic"`` and ``"where"`` wraps, the ``"where"`` extension and the other
materialization choices) were removed when the research code was archived; they are in the research code
archived at the tag ``research-archive-2026-10-12``.

Defaults equal today's values: window 7, ``periodic_extension`` halo ``HIERARCHY_HALO``,
and a shift of ``window // 2`` (timm's SW-MSA shift; 3 at window 7). ``timm_shift``
gives timm's rule for a grid of windows. The cyclic shift is validated at windows 5, 7, 8
and 12 with shift window // 2; other windows and shifts are not.
"""

from __future__ import annotations

import math

import numpy as np

from ..core.grid import TokenGrid
from ..core.tensor import WINDOW

SHIFT = 3
"""Swin-T's shift: ``WINDOW // 2``, the default at window 7. The cyclic shift is validated at window // 2 for
windows 5, 7, 8 and 12."""


def timm_shift(rows: int, cols: int, window: int = WINDOW) -> int:
    """timm's SW-MSA shift for a grid of ``rows`` x ``cols`` windows of ``window`` tokens.

    timm's ``SwinTransformerBlock`` (``_calc_window_shift``, with the stage's shift
    ``window // 2``) keeps the shift when the input is larger than one window and sets it
    to 0 on an axis that holds exactly one window. So: ``window // 2`` when the grid has
    at least two windows along both axes, 0 when it is one window. One window along only
    one axis is refused: timm would shift the other axis only, which the window layout
    does not build (``shifted_window_mask`` and the cyclic shift shift both axes).
    Client-side only."""
    for name, value in (("rows", rows), ("cols", cols), ("window", window)):
        if not _is_integer(value) or value < 1:
            raise ValueError(f"{name} must be a positive integer; received {value!r}")
    if rows == 1 and cols == 1:
        return 0
    if rows == 1 or cols == 1:
        raise ValueError(f"a grid of {rows} x {cols} windows has one window along a single axis: "
                         "timm would shift only the other axis there, which the window layout does not build")
    return window // 2


def _is_integer(value) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _resolve_shift(shift, window: int) -> int:
    """``shift``, or ``window // 2`` (timm's SW-MSA shift) when it is None; refused unless an
    integer with 0 < shift < window (a shift of 0 or of a whole window moves nothing and
    labels every token alike). Client-side only."""
    shift = window // 2 if shift is None else shift
    if not _is_integer(shift) or not 0 < shift < window:
        raise ValueError(f"the shift must be an integer within one window, 0 < shift < {window}; "
                         f"received {shift!r}")
    return int(shift)


def projection(ee, coarse: TokenGrid, dx: int = 0, dy: int = 0):
    s = coarse.scale
    return ee.Projection(coarse.crs, [s, 0, -s * dx, 0, -s, s * dy])


def translate(ee, image, coarse: TokenGrid, dx: int = 0, dy: int = 0):
    """new(I, J) = old(I + dy, J + dx): an exact relabeling of the coarse grid."""
    return image.changeProj(projection(ee, coarse), projection(ee, coarse, dx, dy))


def materialize(ee, image, coarse: TokenGrid):
    """Pin an image to the coarse grid (``reproject``), so translated views read
    the computed tiles instead of re-requesting the subgraph on shifted regions."""
    return image.reproject(projection(ee, coarse))


def cyclic_neighbor(ee, image, coarse: TokenGrid, axis: str, step: int):
    """new(I, J) = old(I + step mod kh, J) for axis "y", old(I, J + step mod kw) for "x".

    The translation by `step` (the periodic wrap). Exact when `image` is periodic with
    the grid's window counts (window (I, J) = window (I mod kh, J mod kw)), which
    window-local operations preserve: the neighbor of the last window row then *is*
    row 0, recomputed as a one-window halo.
    """
    if step not in (-1, 1):
        raise ValueError("step must be -1 or +1")
    n = coarse.height if axis == "y" else coarse.width
    if n < 2:
        raise ValueError("a cyclic shift needs at least two windows along each axis")
    offsets = {"y": lambda d: {"dy": d}, "x": lambda d: {"dx": d}}[axis]
    return translate(ee, image, coarse, **offsets(step))


def roll_axis(ee, x, coarse: TokenGrid, channels: int, offset: int, axis: str,
              window: int = WINDOW, pin: bool = True):
    """Window layout of new(Y) = old(Y + offset) along one token axis, 0 < |offset| < w.

    pin=True materializes x on the coarse grid first (see ``materialize``).
    Windows 5, 7, 8 and 12 with offset window // 2 are validated.
    """
    if not 0 < abs(offset) < window:
        raise ValueError("offset must be within one window")
    if pin:
        x = materialize(ee, x, coarse)
    neighbor = cyclic_neighbor(ee, x, coarse, axis, 1 if offset > 0 else -1)
    # Slabs (in token rows/columns of a window): own [o, w) + next [0, o), or
    # previous [w+o, w) + own [0, w+o).
    if offset > 0:
        parts = ((x, offset, window), (neighbor, 0, offset))
    else:
        parts = ((neighbor, window + offset, window), (x, 0, window + offset))
    if axis == "y":
        a, b = ((image.arraySlice(0, p * window, q * window)) for image, p, q in parts)
        return a.arrayCat(b, 0)
    wide = ee.Image(ee.Array([window, window * channels]))
    tall = ee.Image(ee.Array([window * window, channels]))
    a, b = ((image.arrayReshape(wide, 2).arraySlice(1, p * channels, q * channels))
            for image, p, q in parts)
    return a.arrayCat(b, 1).arrayReshape(tall, 2)


def roll_windows(ee, x, coarse: TokenGrid, channels: int, offset: int, window: int = WINDOW):
    """new(Y, X) = old(Y + offset, X + offset) cyclically, in window layout.

    Rows first, then columns; x is materialized once, before the rows."""
    rows = roll_axis(ee, x, coarse, channels, offset, "y", window, True)
    return roll_axis(ee, rows, coarse, channels, offset, "x", window, False)


def shift_windows(ee, x, coarse: TokenGrid, channels: int, shift: int | None = None, window: int = WINDOW):
    """timm: window_partition(roll(x, (-shift, -shift))) from x in window layout.

    ``shift`` defaults to ``window // 2`` (3 at window 7) and must satisfy
    0 < shift < window. Validated at windows 5, 7, 8 and 12 with shift window // 2."""
    return roll_windows(ee, x, coarse, channels, _resolve_shift(shift, window), window)


def unshift_windows(ee, y, coarse: TokenGrid, channels: int, shift: int | None = None, window: int = WINDOW):
    """timm: roll(window_reverse(y), (+shift, +shift)) in window layout.

    ``shift`` defaults to ``window // 2`` (3 at window 7) and must satisfy
    0 < shift < window. Validated at windows 5, 7, 8 and 12 with shift window // 2."""
    return roll_windows(ee, y, coarse, channels, -_resolve_shift(shift, window), window)


# ---------------------------------------------------------------------------
# Periodic extension of a finite stage input
# ---------------------------------------------------------------------------
STAGE_HALO = 1                     # one SW-MSA block: one window on each side
HIERARCHY_HALO = STAGE_HALO + 2 * STAGE_HALO   # stage-1 SW-MSA + stage-2 SW-MSA (x2 per merge)


def _band(ee, coarse: TokenGrid, axis: str, start: int, end: int, margin: int):
    """Coarse rows (axis "y") or columns ("x") [start, end), spanning the other
    axis with `margin` windows beyond the raster on both sides."""
    s = coarse.scale
    if axis == "y":
        c0, c1, r0, r1 = -margin, coarse.width + margin, start, end
    else:
        c0, c1, r0, r1 = start, end, -margin, coarse.height + margin
    return ee.Geometry.Rectangle([s * c0, -s * r1, s * c1, -s * r0], coarse.crs, False)


def _copies(n: int, halo: int):
    """(period multiple m, rows [start, end) filled by the copy translated by m n)."""
    copies = [(0, 0, n)]
    for m in range(1, math.ceil(halo / n) + 1):
        copies.append((m, m * n, min((m + 1) * n, n + halo)))        # below / right
        copies.append((-m, max(-m * n, -halo), -(m - 1) * n))         # above / left
    return copies


def _extend_axis(ee, x, coarse: TokenGrid, axis: str, halo: int, margin: int):
    n = coarse.height if axis == "y" else coarse.width
    offset = (lambda d: {"dy": d}) if axis == "y" else (lambda d: {"dx": d})
    # new(I) = x(I - m n) on the rows that copy fills: translate by -m n.
    parts = [(x if m == 0 else translate(ee, x, coarse, **offset(-m * n)))
             .clip(_band(ee, coarse, axis, start, end, margin))
             for m, start, end in _copies(n, halo)]
    return ee.ImageCollection(parts).mosaic()


def periodic_extension(ee, x, coarse: TokenGrid, halo: int = HIERARCHY_HALO):
    """x(I mod kh, J mod kw) on the ring of `halo` windows around the raster.

    x is a window-layout image on the full raster's coarse grid; its values
    outside the raster (masked, or anything) are never read. The input and the
    row-extended image are materialized on the coarse grid, so the translated
    copies read computed tiles; each copy is clipped to the strip it fills and
    the strips are mosaicked. Any halo is defined by construction; it has only
    been run at window 7.
    """
    if halo < 1:
        return x
    x = materialize(ee, x, coarse)
    rows = _extend_axis(ee, x, coarse, "y", halo, margin=halo)
    rows = materialize(ee, rows, coarse)
    return _extend_axis(ee, rows, coarse, "x", halo, margin=halo)


def domain_mask(ee, coarse: TokenGrid):
    """1 inside the raster's coarse pixels, 0 outside."""
    coords = ee.Image.pixelCoordinates(projection(ee, coarse)).floor()
    i, j = coords.select("y"), coords.select("x")
    return i.gte(0).And(i.lt(coarse.height)).And(j.gte(0)).And(j.lt(coarse.width))
