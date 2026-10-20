"""Swin patch embedding (R -> W) and patch merging (W -> W, one stage down).

Moved from ``swin_image_encoder/earth_engine/patch_embed.py`` (``patch_embed_constants``
with ``PatchEmbedConstants``, ``stride_columns``, ``patch_embed``, ``valid_fraction``)
and ``swin_hierarchy/earth_engine/patch_merge.py`` (``patch_merge_constants`` with
``PatchMergeConstants``, ``stage2_coarse``, ``gather_windows``, ``merge_windows``,
``patch_merge``, ``GROUP_OFFSETS``), research code archived at the tag
``research-archive-2026-10-12`` (docs/history/README.md).

**Patch embedding** (timm's PatchEmbed: Conv2d with kernel = stride = p and LayerNorm),
straight into window layout; Swin-T's p = 4, bands VV and VH, width 96 and window 7 are
the defaults::

    input       C bands (``bands``, selected by name) float32 on the input grid [H, W]
    columns     the p x p x C patch of every stage-1 token as one [1, C p^2] row on the
                token grid, flattened (c, ky, kx): Conv2d's weight.reshape(dim, -1)
                order (p^2 exact stride samples of the input, the original attention implementation's
                reproject + changeProj space-to-depth at factor p)
    pe_patches  window partition: [w^2, C p^2] per stage-1 window on the coarse grid
    pe_proj     ``linear``: [w^2, C p^2] x W^T [C p^2, dim] + bias, rounded once
    x           ``layer_norm(dim)``, rounded once, masked outside the crop

The Conv2d is an im2col matrix product: each token reads exactly its own p x p pixels
(equal bit for bit on an integer fixture, and against timm's own one-hot extraction on
pixel IDs at patches 2, 3, 4 and 8). The tokens are partitioned into windows before the
projection (the research ``token`` placement, which projected each token's row first,
was removed when the research code was archived). The embedding width ``dim`` comes from the projection's weight
shape.

**Patch merging** (timm's PatchMerging, from 2 x 2 stage-1 windows)::

    merged[i, j] = cat(x[2i, 2j], x[2i+1, 2j], x[2i, 2j+1], x[2i+1, 2j+1])   [4C]
    out          = Linear(4C, 2C, bias=False)(LayerNorm(4C)(merged))

Stage-2 window (P, Q) draws exactly on the 14 x 14 stage-1 tokens of the 2 x 2 stage-1
windows (2P + a, 2Q + b). Every step before the LayerNorm moves float32 values without
arithmetic: the four windows are gathered onto the stage-2 coarse grid by stride
sampling (``gather_windows``), assembled into a [2w, 2w C] block and re-sliced into the
four groups x0..x3 in timm's order (``merge_windows``). No token raster is formed.

The embedding gathers its columns by stride sampling and the merge gathers whole windows
(the research ``neighborhood`` columns and ``fine``-grid merge, with their registries
``COLUMN_GATHERS`` and ``MERGE_GATHERS``, were removed when the research code was archived).

**Policy** ``double_round_once`` (``linear`` and ``layer_norm``). Patch merging (with
``gather_windows`` and ``merge_windows``) is validated at windows 5, 7, 8 and 12. The
patch embedding (with ``stride_columns``) is validated at patches 2, 4 and 8, with 1 to 4
input bands (distinct names without regular-expression syntax, in any raster order),
embedding widths 48, 96 and 128, and windows 5, 7, 8 and 12 (against timm's
PatchEmbed); other patches and widths are not. The default bands
("VV", "VH") are recipe-facing: the Swin-T224 adapter's sensor names (contract section 11).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...core.metadata import NO_ARITHMETIC, BlockMetadata, Locality, Validated, validate_metadata
from ...core.precision import DOUBLE_ROUND_ONCE, validate_policy
from ...core.tensor import WINDOW
from ...layouts.shift import domain_mask, materialize
from ...layouts.window import as_row, stride_samples, to_windows
from ...weights.constants import ServerArray, WindowLayout, check_rows
from .linear import WINDOW_LAYOUT, LinearConstants, builder_rows, linear, linear_values, no_constants
from .normalization import layer_norm, layer_norm_values

PATCH = 4
"""Default patch size (Swin-T: kernel = stride = 4)."""

BANDS = ("VV", "VH")
"""Default input channel names (recipe-facing: the adapter's canonical sensor bands)."""

GROUP_OFFSETS = ((0, 0), (1, 0), (0, 1), (1, 1))
"""(dy, dx) of timm's merge groups x0..x3: merged = cat(x0, x1, x2, x3)."""


# ---------------------------------------------------------------------------
# Patch embedding
# ---------------------------------------------------------------------------
@dataclass
class PatchEmbedConstants:
    proj: LinearConstants          # W^T [C patch^2, dim], bias rows repeated
    norm_weight: object
    norm_bias: object
    rows: int                      # window ** 2 (49 at window 7)


def patch_embed_constants(ops, pe: dict, rows=WINDOW_LAYOUT) -> PatchEmbedConstants:
    """pe: {"proj.weight" [dim, C patch^2] (Conv2d's weight flattened in (c, ky, kx) order;
    [96, 32] today), "proj.bias", "norm.weight", "norm.bias"} as NumPy arrays (inline) or
    ServerArrays (asset-backed). ``rows``: the layout the constants serve
    (``WindowLayout(7)`` by default) or its row count."""
    rows = builder_rows(rows)
    return PatchEmbedConstants(
        proj=LinearConstants(ops, pe["proj.weight"], pe["proj.bias"], rows),
        norm_weight=ops.constant(pe["norm.weight"], repeat_rows=rows),
        norm_bias=ops.constant(pe["norm.bias"], repeat_rows=rows), rows=rows)


_REGEX_SPECIAL = frozenset("\\^$.|?*+()[]{}")
"""Characters a band selector would read as regular-expression syntax (``Image.select``)."""


def _check_patch_settings(patch, bands) -> tuple[int, tuple[str, ...]]:
    """``(patch, bands)`` as ``stride_columns`` uses them, or ``ValueError``: the patch is a
    positive integer, and the bands are a non-empty sequence of distinct names (not one
    string), each without regular-expression syntax, because ``Image.select`` reads a name
    as a pattern and could select another band. Client-side only: nothing reaches the
    graph."""
    if isinstance(patch, bool) or not isinstance(patch, (int, np.integer)) or patch < 1:
        raise ValueError(f"patch must be a positive integer; received {patch!r}")
    if isinstance(bands, str):
        raise ValueError(f"bands must be a sequence of band names, not one string; received {bands!r}")
    bands = tuple(bands)
    if not bands or not all(isinstance(b, str) and b for b in bands):
        raise ValueError(f"bands must be a non-empty sequence of non-empty names; received {bands!r}")
    if len(set(bands)) != len(bands):
        raise ValueError(f"band names must be distinct; received {bands!r}")
    special = [b for b in bands if _REGEX_SPECIAL & set(b)]
    if special:
        raise ValueError(f"band names are selected as regular expressions; rename {special} without "
                         f"any of {''.join(sorted(_REGEX_SPECIAL))}")
    return int(patch), bands


def stride_columns(ee, x, grid, *, patch: int = PATCH, bands=BANDS):
    """[1, P] per token on the token grid (P = len(bands) patch^2) from patch^2 exact stride
    samples of x (the bands on the input grid `grid`).  Sample ky * patch + kx holds input
    pixel (patch i + ky, patch j + kx) of token (i, j); columns in (c, ky, kx) order, with
    c following ``bands`` (selected by name, whatever the order of x's bands).
    ``_check_patch_settings`` refuses a patch or band list it cannot serve. Validated at
    patches 2, 4 and 8 with 1 to 4 bands."""
    patch, bands = _check_patch_settings(patch, bands)
    samples = stride_samples(ee, grid, x, patch, materialize=True)
    columns = [samples[ky * patch + kx].select(c).rename(f"{c}_{ky}{kx}")
               for c in bands for ky in range(patch) for kx in range(patch)]
    return as_row(ee, ee.Image.cat(columns), len(bands) * patch * patch)


def patch_embed(ops, x, grid, c: PatchEmbedConstants, *, patch: int = PATCH, bands=BANDS,
                window: int = WINDOW, policy: str = DOUBLE_ROUND_ONCE) -> dict:
    """{"pe_columns", "pe_patches", "pe_proj", "x"} from the input x on the input grid.

    "x" is the stage-1 input of the complete encoder: window layout on the stage-1
    coarse grid, masked outside the crop (its cyclic wrap is the encoder's
    periodic extension, as for the complete encoder's masked input).  pe_patches / pe_proj
    are window layout too.  Constants built for another row count than ``window ** 2``
    are refused, and so is a projection whose input width is not
    ``len(bands) * patch ** 2`` (proj.weight [dim, C patch^2], flattened (c, ky, kx)).
    Validated at patches 2, 4 and 8, 1 to 4 bands, widths 48, 96 and 128 and windows 5,
    7, 8 and 12.
    """
    validate_policy(policy, allowed=DOUBLE_ROUND_ONCE)
    patch, bands = _check_patch_settings(patch, bands)
    columns = len(bands) * patch * patch
    if c.proj.in_features != columns:
        raise ValueError(f"the projection reads {c.proj.in_features} columns, but {len(bands)} bands at "
                         f"patch {patch} give {columns}: proj.weight must be [dim, C patch^2], "
                         "Conv2d's weight flattened in (c, ky, kx) order")
    ee = ops.ee
    tokens = grid.coarse(patch)
    coarse = tokens.coarse(window)
    cols = stride_columns(ee, x, grid, patch=patch, bands=bands)
    dim = c.proj.out_features                       # embedding width, from the weight shape
    t = {"pe_columns": cols, "pe_patches": to_windows(ee, cols, tokens, window=window)}
    check_rows(c, WindowLayout(window))
    t["pe_proj"] = linear(ops, t["pe_patches"], c.proj)
    x1 = layer_norm(ops, t["pe_proj"], c.norm_weight, c.norm_bias, dim=dim)
    t["x"] = x1.updateMask(domain_mask(ee, coarse))
    return t


def valid_fraction(ee, validity, grid, *, patch: int = PATCH, bands=BANDS):
    """Fraction of a token's patch values that were valid in the source (scalar
    band "valid_fraction" on the token grid): 1 fully valid, 0 fully masked."""
    cols = stride_columns(ee, validity, grid, patch=patch, bands=bands)
    return cols.arrayReduce(ee.Reducer.mean(), [1]).arrayGet([0, 0]).rename("valid_fraction")


# ---------------------------------------------------------------------------
# Patch merging
# ---------------------------------------------------------------------------
@dataclass
class PatchMergeConstants:
    norm_weight: object          # [rows, 4C] (rows repeated)
    norm_bias: object
    reduction: LinearConstants   # W^T [4C, 2C], no bias
    in_dim: int                  # C
    rows: int = WINDOW_LAYOUT.rows


def patch_merge_constants(ops, parameters: dict, layout=WINDOW_LAYOUT) -> PatchMergeConstants:
    """parameters: timm PatchMerging state_dict names -> float32 NumPy arrays or ServerArrays.
    ``layout``: the layout the constants serve (default window 7: 49 rows) or its row count."""
    p = parameters
    rows = builder_rows(layout)
    width = (p["norm.weight"].shape if isinstance(p["norm.weight"], ServerArray)
             else np.shape(p["norm.weight"]))[0]
    if width % 4:
        raise ValueError("LayerNorm width must be 4C")
    return PatchMergeConstants(
        norm_weight=ops.constant(p["norm.weight"], repeat_rows=rows),
        norm_bias=ops.constant(p["norm.bias"], repeat_rows=rows),
        reduction=LinearConstants(ops, p["reduction.weight"], p.get("reduction.bias"), rows),
        in_dim=width // 4, rows=rows)


def stage2_coarse(coarse1):
    """Stage-2 coarse grid: one stage-2 window per 2 x 2 stage-1 windows."""
    if coarse1.height % 2 or coarse1.width % 2:
        raise ValueError("Patch Merging in window layout needs an even number of stage-1 "
                         "windows per side")
    return coarse1.coarse(2)


def gather_windows(ee, x, coarse1) -> dict:
    """{(a, b): image on the stage-2 coarse grid holding stage-1 window (2P + a, 2Q + b)}.

    x is materialized on the stage-1 coarse grid first, so the four samples read the
    same computed tiles (translated views must read the same tiles).
    """
    stage2_coarse(coarse1)
    x = materialize(ee, x, coarse1)
    samples = stride_samples(ee, coarse1, x, 2, materialize=False)  # element 2a + b
    return {(a, b): samples[2 * a + b] for a in (0, 1) for b in (0, 1)}


def merge_windows(ee, windows: dict, channels: int, *, window: int = WINDOW) -> dict:
    """The four groups x0..x3 [w^2, C] and their concatenation "cat" [w^2, 4C]."""
    w, c = window, channels

    def shape(*dims):
        return ee.Image(ee.Array(list(dims)))

    def side_by_side(left, right):         # [w, w C] | [w, w C] -> [w, 2w C]
        return left.arrayReshape(shape(w, w * c), 2).arrayCat(
            right.arrayReshape(shape(w, w * c), 2), 1)

    block = side_by_side(windows[0, 0], windows[0, 1]).arrayCat(
        side_by_side(windows[1, 0], windows[1, 1]), 0)                 # [2w, 2w C]
    pairs = block.arrayReshape(shape(w, 4 * w * c), 2)                 # [w, 4w C]
    by_dy = [pairs.arraySlice(1, dy * 2 * w * c, (dy + 1) * 2 * w * c)
             .arrayReshape(shape(w * w, 2 * c), 2) for dy in (0, 1)]   # [w^2, 2C] each
    t = {f"x{g}": by_dy[dy].arraySlice(1, dx * c, (dx + 1) * c)
         for g, (dy, dx) in enumerate(GROUP_OFFSETS)}
    t["cat"] = t["x0"].arrayCat(t["x1"], 1).arrayCat(t["x2"], 1).arrayCat(t["x3"], 1)
    return t


def patch_merge(ops, x, c: PatchMergeConstants, coarse1, *, window: int = WINDOW,
                policy: str = DOUBLE_ROUND_ONCE) -> dict:
    """Every intermediate of Patch Merging for stage-1 windows x [w^2, C]:
    x0..x3 [w^2, C], cat and norm [w^2, 4C], out [w^2, 2C], on the stage-2 coarse grid.
    The windows are gathered whole (``gather_windows``, ``merge_windows``). Constants
    built for another row count than ``window ** 2`` are refused.
    """
    validate_policy(policy, allowed=DOUBLE_ROUND_ONCE)
    check_rows(c, WindowLayout(window))
    t = merge_windows(ops.ee, gather_windows(ops.ee, x, coarse1), c.in_dim, window=window)
    t["norm"] = layer_norm(ops, t["cat"], c.norm_weight, c.norm_bias, dim=4 * c.in_dim)
    t["out"] = linear(ops, t["norm"], c.reduction)
    return t


# ---------------------------------------------------------------------------
# Cost functions (contract section 9)
# ---------------------------------------------------------------------------
def patch_embed_values(dim: int, *, patch: int = PATCH, channels: int = len(BANDS),
                       layout=WINDOW_LAYOUT) -> int:
    """Values embedded for NumPy weights: the projection W^T [C patch^2, dim] and its bias
    rows, and the LayerNorm's gamma and beta rows."""
    return (linear_values(channels * patch * patch, dim, bias=True, layout=layout)
            + layer_norm_values(dim, layout=layout))


def patch_embed_arrays() -> int:
    """Server arrays read: proj.weight, proj.bias, norm.weight, norm.bias."""
    return 4


def patch_merge_values(channels: int, *, bias: bool = False, layout=WINDOW_LAYOUT) -> int:
    """Values embedded for NumPy weights: LayerNorm(4C) gamma and beta rows and the
    reduction W^T [4C, 2C] (timm: no bias)."""
    return layer_norm_values(4 * channels, layout=layout) + linear_values(
        4 * channels, 2 * channels, bias=bias, layout=layout)


def patch_merge_arrays(*, bias: bool = False) -> int:
    """Server arrays read: norm.weight, norm.bias, reduction.weight (and its bias)."""
    return 4 if bias else 3


_LARGEST_PATCH = 8
"""The largest validated patch. The metadata radius of ``patch_embed`` and
``stride_columns`` is half of it, the reach of a token's patch from its centre: the largest
validated reach, as for ``pool2d``. It was ``PATCH // 2`` = 2 while only patch 4 was
validated. ``patch_locality`` gives the reach at a call's patch."""


def patch_locality(*, patch: int = PATCH) -> dict:
    """The locality of a patch embedding at a patch (``BlockMetadata.per_setting``): k x k,
    radius ``patch // 2``, the library's convention for a patch; patch 1 reads its own
    pixel."""
    if patch // 2 < 1:
        return {"locality": Locality.POINTWISE, "radius": None}
    return {"locality": Locality.KERNEL, "radius": patch // 2}


_WINDOWS = (5, 7, 8, 12)
"""The window sizes validated live for patch merging and patch embedding."""

_PATCHES = {"patch": (2, 4, 8), "channels": (1, 2, 3, 4)}
"""Validated patches and band counts (``channels``, ``patch_embed_values``'s name for the number of
bands; the names themselves are labels)."""

METADATA = validate_metadata({
    "patch_embed": BlockMetadata(
        name="patch_embed", layout_in="R", layout_out="W", locality=Locality.KERNEL, radius=_LARGEST_PATCH // 2,
        policy=DOUBLE_ROUND_ONCE, embedded_constants=patch_embed_values, asset_arrays=patch_embed_arrays,
        status="Ready — patches 2, 4, 8; 1–4 bands; widths 48, 96, 128; windows 5, 7, 8, 12",
        evidence=("docs/history/building-blocks-2026-10-11.md#33-convolution", "docs/history/swin_image_encoder/notes/milestone6_findings.md",
                  "docs/evidence/p03-offline.json",
                  "docs/evidence/live-fixtures/20261013T120034Z-p03.json"),
        validated=Validated(({**_PATCHES, "dim": (48, 96, 128), "window": _WINDOWS},)),
        per_setting=patch_locality),
    "stride_columns": BlockMetadata(
        name="stride_columns", layout_in="R", layout_out="W", locality=Locality.KERNEL, radius=_LARGEST_PATCH // 2,
        policy=NO_ARITHMETIC, embedded_constants=no_constants, asset_arrays=no_constants,
        status="Ready — patches 2, 4, 8; 1–4 bands",
        evidence=("docs/history/building-blocks-2026-10-11.md#33-convolution", "docs/history/swin_image_encoder/notes/milestone6_findings.md",
                  "docs/evidence/p03-offline.json",
                  "docs/evidence/live-fixtures/20261013T120034Z-p03.json"),
        validated=Validated((_PATCHES,), sources=(patch_embed_values,)),
        per_setting=patch_locality),
    # Validated: the window sizes above; timm's PatchMerging has no reduction bias (never varied).
    "patch_merge": BlockMetadata(
        name="patch_merge", layout_in="W", layout_out="W", locality=Locality.KERNEL, radius=1,
        policy=DOUBLE_ROUND_ONCE, embedded_constants=patch_merge_values, asset_arrays=patch_merge_arrays,
        status="Ready — windows 5, 7, 8, 12",
        evidence=("docs/history/building-blocks-2026-10-11.md#37-resampling", "docs/history/swin_hierarchy/notes/milestone4_findings.md",
                  "docs/evidence/p01-offline.json",
                  "docs/evidence/live-fixtures/20261013T074301Z-p01a.json", "docs/evidence/live-fixtures/20261013T074330Z-p01b.json", "docs/evidence/live-fixtures/20261013T075542Z-window-relative_position_bias-w12.json"),
        validated=Validated(({"window": _WINDOWS, "bias": (False,)},))),
    "gather_windows": BlockMetadata(
        name="gather_windows", layout_in="W", layout_out="W", locality=Locality.KERNEL, radius=1,
        policy=NO_ARITHMETIC, embedded_constants=no_constants, asset_arrays=no_constants,
        status="Ready — windows 5, 7, 8, 12",
        evidence=("docs/history/building-blocks-2026-10-11.md#37-resampling", "docs/history/swin_hierarchy/notes/milestone4_findings.md",
                  "docs/evidence/p01-offline.json",
                  "docs/evidence/live-fixtures/20261013T074301Z-p01a.json", "docs/evidence/live-fixtures/20261013T074330Z-p01b.json", "docs/evidence/live-fixtures/20261013T075542Z-window-relative_position_bias-w12.json"),
        validated=Validated(({"window": _WINDOWS},))),
    "merge_windows": BlockMetadata(
        name="merge_windows", layout_in="W", layout_out="W", locality=Locality.WINDOW,
        policy=NO_ARITHMETIC, embedded_constants=no_constants, asset_arrays=no_constants,
        status="Ready — windows 5, 7, 8, 12",
        evidence=("docs/history/building-blocks-2026-10-11.md#37-resampling", "docs/history/swin_hierarchy/notes/milestone4_findings.md",
                  "docs/evidence/p01-offline.json",
                  "docs/evidence/live-fixtures/20261013T074301Z-p01a.json", "docs/evidence/live-fixtures/20261013T074330Z-p01b.json", "docs/evidence/live-fixtures/20261013T075542Z-window-relative_position_bias-w12.json"),
        validated=Validated(({"window": _WINDOWS},))),
})
