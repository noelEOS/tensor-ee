"""Bilinear resizing in window layout, and PyTorch's own interpolation coefficients.

Moved from ``upernet/earth_engine/resize.py`` (``local_matrices``, ``window_resize``,
``window_matrix_resize`` and their helpers ``float32_exact``, ``_per_pixel_rows``,
``output_weights``, ``window_matrix_weights``, and ``bilinear_weights`` there, here
``model_bilinear_weights``) and ``upernet/earth_engine/_coefficients.py``
(``bilinear_weights``, ``window_token_order``), research code archived at the tag
``research-archive-2026-10-12`` (docs/history/README.md). The dense, matrix and native
formulations were research only.

Every resize in the installed UPerNet decoder is F.interpolate(x, size=target,
mode="bilinear", align_corners=False) (PSP: pooled 1/2/3/6 -> H/32; FPN: 2x; fusion:
8x, 4x, 2x to H/4). The segmentation head's UpsamplingBilinear2d(scale_factor=4) is
align_corners=True. PyTorch's rule along one axis (n_in -> n_out), align_corners=False:

    src = max((o + 1/2) * n_in / n_out - 1/2, 0); i0 = floor(src); i1 = min(i0 + 1, n_in - 1)
    out[o] = (1 - (src - i0)) * in[i0] + (src - i0) * in[i1]

and 2-D bilinear is the separable product. With align_corners=True,
src = o * (n_in - 1) / (n_out - 1), evaluated in the tensor's precision, so the float32
model's weights are float32 numbers (not dyadic). The weights are never re-derived:
they are read off PyTorch's own kernel with one-hot inputs (``bilinear_weights``;
``model_bilinear_weights`` picks the model's dtype, float32 for align_corners=True).
For an integer factor f every output reads its parent token and the parent's direct
neighbours: the one-token ring of the window halo (asserted per output row by
``local_matrices``).

- ``window_resize`` (integer factor f; FPN, fusion and the head): window layout ->
  window layout on the f-times finer window grid. The parent window's [w + 2, w + 2, C]
  halo is read at every child window (nearest, exact) and two small products apply the
  child's local weight matrices My, Mx [w, w + 2] (slices of PyTorch's global matrices;
  out-of-raster ring tokens get weight 0, which is the edge clamp); y first, then x.
- ``window_matrix_resize`` (any sizes, small V source; PSP branches): each output token
  is one sequential double dot over the source pixels with the products of PyTorch's
  float32 1-D weights (exact in double).

**Policy** ``double_round_once``: products and sums in double, each output rounded to
float32 once. (The research ``rounded=False`` probe of the double result was removed at
L06b.)

Window 7 is the default. Validated at windows 7 and 8: ``window_resize`` at factors 2, 4
and 8 with align_corners=False and factor 4 with align_corners=True (window 7: the Swin-T224
decoder's FPN, fusion and head; window 8: the capstone's, on its token extents, C0b),
``window_matrix_resize`` from 1 x 1, 2 x 2, 3 x 3 and 6 x 6 sources (the PSP's bins) onto one
window with align_corners=False. Other windows, factors and sources are not validated.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ...core.grid import TokenGrid
from ...core.metadata import BlockMetadata, Locality, Validated, validate_metadata
from ...core.precision import DOUBLE_ROUND_ONCE, validate_policy
from ...core.tensor import WINDOW
from ...layouts.shift import domain_mask
from ...layouts.window import halo, reshape
from ...weights.constants import constant_value
from .linear import no_constants


# ---------------------------------------------------------------------------
# Weights (client side, read off PyTorch)
# ---------------------------------------------------------------------------
def bilinear_weights(n_in: int, n_out: int, dtype=torch.float64, align_corners: bool = False) -> np.ndarray:
    """[n_out, n_in] matrix M with out = M @ in along one axis, read off PyTorch's own
    kernel by resizing one-hot rows (separable: bilinear = M_y X M_x^T).

    align_corners=True (the segmentation head's UpsamplingBilinear2d): the weights depend
    on dtype, because PyTorch computes the scale (n_in - 1) / (n_out - 1) and every source
    coordinate in the input's own precision; the float32 model's are dtype=torch.float32."""
    eye = torch.eye(n_in, dtype=dtype)                       # [n_in, n_in]: row i = e_i
    with torch.no_grad():
        out = F.interpolate(eye[:, None, None, :], size=(1, n_out), mode="bilinear",
                            align_corners=align_corners)
    return out[:, 0, 0, :].T.double().numpy()


def model_bilinear_weights(n_in: int, n_out: int, dtype=None, align_corners: bool = False) -> np.ndarray:
    """PyTorch's [n_out, n_in] weights (``bilinear_weights``).  With align_corners=True the
    default dtype is float32: the float32 model's own weights (formerly
    ``upernet.earth_engine.resize.bilinear_weights``)."""
    if dtype is None:
        dtype = torch.float32 if align_corners else torch.float64
    return bilinear_weights(n_in, n_out, dtype, align_corners)


def float32_exact(matrix: np.ndarray) -> np.ndarray:
    m32 = matrix.astype(np.float32)
    if not np.array_equal(m32.astype(np.float64), matrix):
        raise ValueError("interpolation weights are not exactly representable in float32")
    return m32


def local_matrices(n_in: int, factor: int, align_corners: bool = False, *,
                   window: int = WINDOW) -> tuple[np.ndarray, list[int]]:
    """Window-local weights of an n_in -> factor * n_in axis.

    Returns (stack [K * w, w + 2] float32, index): child window row I' (0 .. factor * n_in
    / w - 1) uses rows w k .. w k + w - 1 of the stack, k = index[I'], to map its parent
    window's w + 2 halo tokens (parent tokens w I - 1 .. w I + w) to its w output tokens.
    The rows are slices of the GLOBAL weight matrix (absolute output positions): with
    align_corners=True nearly every child window has its own matrix (K = factor * n_in /
    w). Windows 7 and 8 are validated (see the module docstring)."""
    if n_in % window:
        raise ValueError(f"window layout needs a multiple of {window} tokens")
    m = model_bilinear_weights(n_in, factor * n_in, align_corners=align_corners)
    mats, keys, index = [], {}, []
    for child in range(factor * n_in // window):
        parent = child // factor
        rows = m[window * child: window * (child + 1)]
        local = np.zeros((window, window + 2))
        for h in range(window + 2):
            i = window * parent - 1 + h
            if 0 <= i < n_in:
                local[:, h] = rows[:, i]
        if not np.array_equal(local.sum(1), rows.sum(1)):
            raise AssertionError("an output reads beyond the parent's one-token halo")
        key = local.tobytes()
        if key not in keys:
            keys[key] = len(mats)
            mats.append(local)
        index.append(keys[key])
    return float32_exact(np.concatenate(mats)), index


def window_token_order(h: int, w: int, *, window: int = WINDOW) -> np.ndarray:
    """Row-major pixel index of window-layout position (I nw + J) w^2 + w a + b."""
    nh, nw = h // window, w // window
    I, J, a, b = np.meshgrid(np.arange(nh), np.arange(nw), np.arange(window), np.arange(window),
                             indexing="ij")
    return ((window * I + a) * w + window * J + b).reshape(-1)


def output_weights(in_size: tuple[int, int], out_size: tuple[int, int],
                   align_corners: bool = False) -> np.ndarray:
    """[H_out * W_out, H_in * W_in]: row Y * W_out + X = My[Y] (x) Mx[X], the products of
    PyTorch's float32 1-D weights computed in float64 (exact: 24-bit x 24-bit)."""
    my = model_bilinear_weights(in_size[0], out_size[0], torch.float32, align_corners)
    mx = model_bilinear_weights(in_size[1], out_size[1], torch.float32, align_corners)
    return np.einsum("yi,xj->yxij", my.astype(np.float64), mx.astype(np.float64)).reshape(
        out_size[0] * out_size[1], in_size[0] * in_size[1])


def window_matrix_weights(in_size, out_size, height: int | None = None, width: int | None = None,
                          row0: int = 0, col0: int = 0, align_corners: bool = False, *,
                          window: int = WINDOW) -> np.ndarray:
    """[n w^2, H_in W_in] rows of output_weights(in_size, out_size) for the height x width
    output pixels at (row0, col0) of the out_size grid (default: all of it), in window
    layout order: window I n_w + J, token w a + b = output pixel (row0 + w I + a,
    col0 + w J + b).  Rows are indexed by the *absolute* output position, so a tile of a
    larger output grid gets the large grid's interpolation (Milestone 8, Option C)."""
    height = out_size[0] if height is None else height
    width = out_size[1] if width is None else width
    full = output_weights(in_size, out_size, align_corners).reshape(out_size[0], out_size[1], -1)
    tile = full[row0:row0 + height, col0:col0 + width].reshape(height * width, -1)
    return tile[window_token_order(height, width, window=window)]


# ---------------------------------------------------------------------------
# window: integer factor, window layout -> window layout
# ---------------------------------------------------------------------------
def _per_pixel_rows(ee, stack: np.ndarray, index: list[int], position, *, window: int = WINDOW):
    """[w, w + 2] rows w k .. w k + w - 1 of `stack` at each pixel, k = index[position]."""
    k = ee.Image(ee.Array([float(v) for v in index])).arrayGet(position.toInt()).toInt()
    start = k.multiply(window)
    return ee.Image(constant_value(ee, stack)).arraySlice(0, start, start.add(window))


def window_resize(ee, x, tokens: TokenGrid, channels: int, factor: int, align_corners: bool = False, *,
                  window: int = WINDOW, policy: str = DOUBLE_ROUND_ONCE):
    """Window layout [w^2, C] on tokens.coarse(w) -> window layout on the factor-times finer
    token grid's window grid (masked outside the raster).  Bilinear, align_corners as given
    (False: the decoder's resizes; True: the segmentation head's), the y axis first (one
    [w, w + 2] x [w + 2, (w + 2) C] product), then x ([w, w + 2] x [w + 2, w C]), one
    float32 rounding.
    Returns (image, fine token grid). Validated at windows 7 and 8, factors 2, 4 and 8 with
    align_corners=False and factor 4 with align_corners=True (C0b at window 8)."""
    validate_policy(policy, allowed=DOUBLE_ROUND_ONCE)
    coarse = tokens.coarse(window)
    fine_tokens = TokenGrid(tokens.height * factor, tokens.width * factor,
                            tokens.scale / factor, tokens.crs)
    fine = fine_tokens.coarse(window)
    stack_y, index_y = local_matrices(tokens.height, factor, align_corners, window=window)
    stack_x, index_x = local_matrices(tokens.width, factor, align_corners, window=window)
    projection = ee.Projection(fine.crs, fine.transform)
    coords = ee.Image.pixelCoordinates(projection).floor()
    rows = coords.select("y").clamp(0, fine.height - 1)
    cols = coords.select("x").clamp(0, fine.width - 1)
    my = _per_pixel_rows(ee, stack_y, index_y, rows, window=window)            # [w, w + 2]
    mx = _per_pixel_rows(ee, stack_x, index_x, cols, window=window)            # [w, w + 2]
    halo3 = halo(ee, x, coarse, channels, window=window).reproject(ee.Projection(coarse.crs, coarse.transform))
    halo3 = halo3.reproject(projection)                               # parent's halo, nearest
    a = my.matrixMultiply(reshape(ee, halo3, [window + 2, (window + 2) * channels]))
    a = reshape(ee, a, [window, window + 2, channels]).arrayTranspose(0, 1)    # [w + 2, w, C]
    b = mx.matrixMultiply(reshape(ee, a, [window + 2, window * channels]))     # [w (s), w C]
    b = reshape(ee, b, [window, window, channels]).arrayTranspose(0, 1)       # [r, s, C]
    out = reshape(ee, b, [window * window, channels]).toFloat()
    return out.updateMask(domain_mask(ee, fine)), fine_tokens


# ---------------------------------------------------------------------------
# matrix: a small V source onto window layout
# ---------------------------------------------------------------------------
def window_matrix_resize(ee, source_array, in_size, tokens: TokenGrid, out_size=None,
                         row0: int = 0, col0: int = 0, align_corners: bool = False, *,
                         window: int = WINDOW, policy: str = DOUBLE_ROUND_ONCE):
    """Milestone 7's matrix resize in window layout (PSP branches): source_array an
    ee.Array [H_in W_in, C] (row-major pixels) -> [w^2, C] per window of `tokens` (the
    output token grid, or a tile of out_size at (row0, col0)).  Each output token is one
    sequential double dot over the source pixels with the products of PyTorch's float32
    1-D weights (exact in double), rounded to float32 once. Validated at windows 7 and 8 from
    1 x 1, 2 x 2, 3 x 3 and 6 x 6 sources onto one window, align_corners=False (C0b at window
    8)."""
    validate_policy(policy, allowed=DOUBLE_ROUND_ONCE)
    tokens_per_window = window * window
    out_size = (tokens.height, tokens.width) if out_size is None else tuple(out_size)
    weights = window_matrix_weights(in_size, out_size, tokens.height, tokens.width, row0, col0,
                                    align_corners, window=window)
    coarse = tokens.coarse(window)
    coords = ee.Image.pixelCoordinates(ee.Projection(coarse.crs, coarse.transform)).floor()
    k = (coords.select("y").clamp(0, coarse.height - 1).multiply(coarse.width)
         .add(coords.select("x").clamp(0, coarse.width - 1))).toInt()
    start = k.multiply(tokens_per_window)
    table = ee.Image(ee.Array(weights.tolist()).double())
    rows = table.arraySlice(0, start, start.add(tokens_per_window))      # [w^2, n_in]
    out = rows.matrixMultiply(ee.Image(source_array)).toFloat()
    return out.updateMask(domain_mask(ee, coarse))


# ---------------------------------------------------------------------------
# Cost functions (contract section 9)
# ---------------------------------------------------------------------------
def window_resize_values(height: int, width: int, factor: int, *, align_corners: bool = False,
                         window: int = WINDOW) -> int:
    """Values embedded for a resize of an height x width token map: per axis the
    [K w, w + 2] coefficient stack and the child-window index (``_per_pixel_rows``)."""
    total = 0
    for n in (height, width):
        stack, index = local_matrices(n, factor, align_corners, window=window)
        total += stack.size + len(index)
    return total


def window_matrix_resize_values(in_size, height: int, width: int) -> int:
    """Values embedded: the [height width, H_in W_in] weight table of the output tokens."""
    return height * width * in_size[0] * in_size[1]


def window_resize_edge(*, align_corners: bool = False) -> dict:
    """Chip-edge dependence of ``window_resize`` at a setting (``BlockMetadata.per_setting``):
    with align_corners=True the weights depend on the map's size (223 distinct row patterns at
    56 -> 224; X01a's review), so tiling with halos cannot reproduce them. With False they
    repeat with the factor (5 patterns at either size), and the ring of one token covers a
    tile's edge."""
    return {"chip_edge_dependent": bool(align_corners)}


METADATA = validate_metadata({
    # Chip-edge dependent over its validated settings (align_corners=True is one), and exactly
    # by setting through window_resize_edge.
    "window_resize": BlockMetadata(
        name="window_resize", layout_in="W", layout_out="W", locality=Locality.WINDOW_RING, radius=1,
        policy=DOUBLE_ROUND_ONCE, embedded_constants=window_resize_values, asset_arrays=no_constants,
        chip_edge_dependent=True,
        status="Ready — windows 7, 8; factors 2, 4, 8 (align_corners=False), 4 (align_corners=True)",
        evidence=("docs/history/building-blocks-2026-10-11.md#37-resampling", "docs/history/upernet/notes/milestone7_findings.md",
                  "docs/history/upernet/notes/milestone11_findings.md", "docs/evidence/c0b-offline.json",
                  "docs/evidence/live-fixtures/20261013T154410Z-c0b.json"),
        validated=Validated(({"window": (7, 8), "factor": (2, 4, 8), "align_corners": (False,)},
                             {"window": (7, 8), "factor": (4,), "align_corners": (True,)})),
        per_setting=window_resize_edge),
    # One window of output tokens (height = width = window), not a tile of a larger grid
    # (out_size None, row0 = col0 = 0: never varied).
    "window_matrix_resize": BlockMetadata(
        name="window_matrix_resize", layout_in="V", layout_out="W", locality=Locality.EXTENT,
        policy=DOUBLE_ROUND_ONCE, embedded_constants=window_matrix_resize_values, asset_arrays=no_constants,
        status="Ready — windows 7, 8; sources 1, 2, 3, 6 onto one window (align_corners=False)",
        evidence=("docs/history/building-blocks-2026-10-11.md#37-resampling", "docs/history/upernet/notes/milestone8_findings.md",
                  "docs/evidence/c0b-offline.json",
                  "docs/evidence/live-fixtures/20261013T154410Z-c0b.json"),
        validated=Validated(tuple({"window": (w,), "height": (w,), "width": (w,),
                                   "in_size": ((1, 1), (2, 2), (3, 3), (6, 6)), "align_corners": (False,),
                                   "out_size": (None,), "row0": (0,), "col0": (0,)} for w in (7, 8)))),
})
