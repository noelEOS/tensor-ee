"""Window attention: the naive attention core, the relative-position bias, the head split,
W-MSA/SW-MSA up to the output projection, and Swin's shifted-window mask.

Moved from ``ee_attention/earth_engine/naive_attention.py`` (``naive_attention``),
``swin_block/earth_engine/relative_position.py`` (``relative_position_index``,
``relative_position_bias``), ``swin_block/earth_engine/swin_block.py`` (``split_heads``,
``window_attention``), ``swin_pair/earth_engine/coordinates.py`` (``coarse_projection``,
``window_coordinates``) and ``swin_pair/earth_engine/shifted_mask.py``
(``region_boundaries``, ``shifted_window_mask``, ``MASK_VALUE``), research code archived
at the tag ``research-archive-2026-10-12`` (docs/history/README.md). The Swin block
composites (``BlockConstants``,
``mlp_branch``, ``swin_block``) stay with the model (L05).

**Attention core** (Milestone 1, policy ``round_each_op``)::

    scores    = Q K^T                     matrixMultiply, transpose
    scaled    = scores / sqrt(d)          divide
    row_max   = max_j scaled              reduce(max)
    e         = exp(scaled - row_max)     repeat, subtract, exp
    attention = e / sum_j e               reduce(sum), divide
    output    = attention V               matrixMultiply

The full [Nq, N] score and probability matrices are materialized. There is no
softmax primitive in EE; the numerically stable form above is explicit.

**Relative-position bias.** The learned table has (2w-1)^2 rows and one column per
head. Swin's index for query token t1 = (a1, b1) and key token t2 = (a2, b2) of a
w x w window (row-major, t = a w + b) is

    index[t1, t2] = (a1 - a2 + w - 1) * (2w - 1) + (b1 - b2 + w - 1)

(timm get_relative_position_index), and bias[h, t1, t2] = table[index, h]. Both
steps run on the server from the table alone: the index from integer sequences, the
gather as a one-hot [N*N, (2w-1)^2] matrix times the table. Each one-hot row has a
single 1, so every gathered value is exactly a table entry.

**Shifted-window mask.** timm (SwinTransformerBlock.get_attn_mask) labels the
*shifted* token raster with nine regions, rows [0, H-w), [H-w, H-s), [H-s, H) times
the same columns, partitions the labels into windows and sets

    mask[window, i, j] = 0 if label(i) == label(j) else -100.

Only the last window row and column straddle region boundaries, so for a window with
flags last_row = (I == kh-1) and last_col = (J == kw-1),

    different[i, j] = (last_row and R[i, j]) or (last_col and C[i, j]),
    R[i, j] = [a_i >= w-s] != [a_j >= w-s],   C[i, j] = [b_i >= w-s] != [b_j >= w-s],

with t = a w + b. R and C are two [N, N] ee.Array values built from integer
sequences (no request payload, computed once); per window the mask is two products, a
maximum and a scale. The mask depends on the chip edge (last window row and column).
It is float32, so adding it to float32 logits is one IEEE float32 addition, as in
PyTorch.

Window 7 (49 tokens), shift 3 and mask value -100.0 (timm's; production) are the
defaults. Task P01 validated windows 5, 7, 8 and 12 with shift w // 2 (W-MSA, SW-MSA and
the single window, where timm disables the shift and SW-MSA is refused here), against
timm; other windows and shifts, and other mask values, are not validated (the
Milestone 1 attention core also ran 25 and 64 tokens).
"""

from __future__ import annotations

import numpy as np

from ...core.errors import reject
from ...core.metadata import (ANY_SETTINGS, COMPOSITE, NO_ARITHMETIC, BlockMetadata, Locality, Validated,
                              validate_metadata)
from ...core.precision import DOUBLE_ROUND_ONCE, ROUND_EACH_OP, validate_policy
from ...core.tensor import WINDOW
from ...layouts.shift import _resolve_shift
from ...weights.constants import ServerArray, WindowLayout, check_rows, constant_value
from .linear import linear, linear_arrays, linear_values, no_constants

MASK_VALUE = -100.0
"""timm's shifted-window mask value (production). Milestone 1's research attention
used -1e30 (``ee_attention.earth_engine.common``)."""

DIM, HEADS = 96, 3
"""``split_heads`` defaults: Swin-T stage 1 (the model passes its own)."""


# ---------------------------------------------------------------------------
# Attention core (Milestone 1)
# ---------------------------------------------------------------------------
def naive_attention(ops, q, k, v, *, tokens: int, dim: int, key_bias=None,
                    query_scale: float | None = None, key_mask=None,
                    policy: str = ROUND_EACH_OP, score_scale: float | None = None) -> dict:
    """softmax(q k^T / sqrt(dim)) v for q [Nq, dim] and k, v [tokens, dim].

    key_bias, when given, is a float32 [Nq, tokens] array added to the scaled
    scores: 0 (valid key) or common.MASK_VALUE (excluded key), or a Swin
    relative-position bias.

    query_scale, when given, follows Swin's order instead: q is multiplied by
    float32(query_scale) before q k^T and the scores are not divided.  "scaled"
    includes key_bias; with a bias, "unbiased" holds the logits before it, and
    with query_scale, "query" holds the scaled q.  (Without either option the
    result has exactly the Phase 2 keys.)

    key_mask, when given, is a second float32 [Nq, tokens] array added after
    key_bias as its own float32 addition, as timm adds the shifted-window mask
    (0 / -100) after the relative-position bias.  "scaled" then includes both;
    "biased" holds the logits before the mask.

    score_scale, when given, multiplies the float32 q k^T scores with the
    pre-rounded float32 factor instead of division by sqrt(dim). It is refused
    together with query_scale. Live coverage is dim32/dim7 with 14 queries and
    seven keys, plus dim32 with 49 global keys across four W7 query windows
    (docs/evidence/segformer_b0-live.json). The existing block status is unchanged.

    Policy ``round_each_op``: each EE operation that returns double is rounded at
    once (the ops methods).
    """
    validate_policy(policy, allowed=ROUND_EACH_OP)
    if query_scale is not None and score_scale is not None:
        reject("query_scale and score_scale cannot be used together")
    if query_scale is None:
        scores = ops.matmul(q, ops.transpose(k))
        unbiased = ops.divide_by_sqrt(scores, dim) if score_scale is None else ops.scale(scores, score_scale)
    else:
        q = ops.scale(q, query_scale)
        scores = ops.matmul(q, ops.transpose(k))
        unbiased = scores
    scaled = unbiased if key_bias is None else unbiased.add(key_bias)
    biased = scaled
    if key_mask is not None:
        scaled = scaled.add(key_mask)
    row_max = ops.row_max(scaled)
    shifted_exp = ops.exp(scaled.subtract(ops.repeat_cols(row_max, tokens)))
    row_sum = ops.row_sum(shifted_exp)
    attention = shifted_exp.divide(ops.repeat_cols(row_sum, tokens))
    output = ops.matmul(attention, v)
    result = {"scores": scores, "scaled": scaled, "row_max": row_max,
              "row_sum": row_sum, "attention": attention, "output": output}
    if key_bias is not None:
        result["unbiased"] = unbiased
    if query_scale is not None:
        result["query"] = q
    if key_mask is not None:
        result["biased"] = biased
    return result


# ---------------------------------------------------------------------------
# Relative-position bias
# ---------------------------------------------------------------------------
def relative_position_index(ee, window: int = WINDOW):
    """[N, N] ee.Array of table rows (as doubles holding integers)."""
    n = window * window
    t = ee.Array(ee.List.sequence(0, n - 1)).double()
    a, b = t.divide(window).floor(), t.mod(window)

    def query(values):  # varies down the rows
        return values.reshape([n, 1]).repeat(1, n)

    def key(values):  # varies along the columns
        return values.reshape([1, n]).repeat(0, n)

    dy = query(a).subtract(key(a)).add(window - 1)
    dx = query(b).subtract(key(b)).add(window - 1)
    return dy.multiply(2 * window - 1).add(dx)


def relative_position_bias(ee, table, window: int = WINDOW, *, policy: str = DOUBLE_ROUND_ONCE) -> list:
    """Per-head float32 [N, N] ee.Array biases from a [(2w-1)^2, heads] table
    (a NumPy array or a ServerArray). The gather is exact; one float32 rounding.
    Windows 5, 7, 8 and 12 are validated (P01)."""
    validate_policy(policy, allowed=DOUBLE_ROUND_ONCE)
    entries, heads = table.shape if isinstance(table, ServerArray) else np.asarray(table).shape
    if entries != (2 * window - 1) ** 2:
        raise ValueError("table rows must be (2w-1)^2")
    n = window * window
    index = relative_position_index(ee, window).reshape([n * n, 1]).repeat(1, entries)
    rows = ee.Array(ee.List.sequence(0, entries - 1)).double().reshape([1, entries]).repeat(0, n * n)
    one_hot = index.eq(rows)                                      # [N*N, entries] of 0/1
    gathered = one_hot.matrixMultiply(constant_value(ee, table))  # [N*N, heads], exact
    return [gathered.slice(1, h, h + 1).reshape([n, n]).float() for h in range(heads)]


# ---------------------------------------------------------------------------
# Heads and window attention
# ---------------------------------------------------------------------------
def split_heads(ops, qkv, dim: int = DIM, heads: int = HEADS) -> dict:
    """q_h, k_h, v_h [N, d] from qkv [N, 3 dim] at timm's column positions,
    d = dim / heads: columns s*dim + d*h ... s*dim + d*h + d-1, s = 0/1/2 (q/k/v)."""
    head_dim = dim // heads
    tensors = {}
    for s, name in enumerate(("q", "k", "v")):
        for h in range(heads):
            start = s * dim + h * head_dim
            tensors[f"{name}_h{h}"] = ops.slice_cols(qkv, start, start + head_dim)
    return tensors


def window_attention(ops, ln1, c, mask=None, *, window: int = WINDOW) -> dict:
    """W-MSA up to (not including) the output projection.

    ``c`` carries the block's constants as the Swin block builds them (``qkv``, a
    ``LinearConstants`` for the window's rows; ``relative_bias``, one [N, N] bias per
    head; ``dim``, ``heads``, ``head_dim``). ``window`` gives the tokens per window
    (``window ** 2``; 49 by default), and constants built for another row count are
    refused.

    mask, when given (SW-MSA), is the float32 [N, N] shifted-window mask, added to
    every head's biased logits as a separate float32 addition, as in timm;
    "logits_biased_h{h}" then holds the logits before it.

    A composite: the QKV ``linear`` runs in ``double_round_once``, the attention core
    in ``round_each_op``. Windows 5, 7, 8 and 12 are validated (P01), with and without
    the shifted-window mask.
    """
    tokens = window * window
    check_rows(c.qkv, WindowLayout(window))
    t = {"qkv": linear(ops, ln1, c.qkv)}
    t.update(split_heads(ops, t["qkv"], c.dim, c.heads))
    query_scale = c.head_dim ** -0.5          # QUERY_SCALE at Swin-T's head width 32
    outputs = []
    for h in range(c.heads):
        result = naive_attention(ops, t[f"q_h{h}"], t[f"k_h{h}"], t[f"v_h{h}"], tokens=tokens,
                                 dim=c.head_dim, key_bias=c.relative_bias[h],
                                 query_scale=query_scale, key_mask=mask)
        t.update({f"q_scaled_h{h}": result["query"], f"scores_h{h}": result["unbiased"],
                  f"bias_h{h}": c.relative_bias[h], f"logits_h{h}": result["scaled"],
                  f"probs_h{h}": result["attention"], f"attn_v_h{h}": result["output"]})
        if mask is not None:
            t[f"logits_biased_h{h}"] = result["biased"]
        outputs.append(result["output"])
    t["concat"] = ops.cat_cols(outputs)
    return t


# ---------------------------------------------------------------------------
# The shifted-window mask (Milestone 3)
# ---------------------------------------------------------------------------
def coarse_projection(ee, coarse):
    return ee.Projection(coarse.crs, coarse.transform)


def window_coordinates(ee, coarse):
    """(I mod kh, J mod kw) of each coarse pixel as double images: the window
    index, periodic outside the raster (EE's mod takes the dividend's sign)."""
    coords = ee.Image.pixelCoordinates(coarse_projection(ee, coarse)).floor()
    period = ee.Image.constant([coarse.width, coarse.height])
    coords = coords.mod(period).add(period).mod(period)
    return coords.select("y"), coords.select("x")


def region_boundaries(ee, window: int, shift: int):
    """R and C: [N, N] 0/1 ee.Array values (row-slab and column-slab differ)."""
    n = window * window
    t = ee.Array(ee.List.sequence(0, n - 1)).double()
    a, b = t.divide(window).floor(), t.mod(window)

    def differs(group):  # group: [N] of 0/1
        query = group.reshape([n, 1]).repeat(1, n)
        key = group.reshape([1, n]).repeat(0, n)
        return query.neq(key)

    return differs(a.gte(window - shift)), differs(b.gte(window - shift))


def shifted_window_mask(ee, coarse, window: int = WINDOW, shift: int | None = None,
                        value: float = MASK_VALUE, *, policy: str = DOUBLE_ROUND_ONCE):
    """float32 [N, N] array image on the coarse grid of the full raster, for at least two
    windows along each axis. ``shift`` defaults to ``window // 2`` (3 at window 7) and must
    satisfy 0 < shift < window. Windows 5, 7, 8 and 12 with shift window // 2 and value
    -100.0 are validated (P01)."""
    validate_policy(policy, allowed=DOUBLE_ROUND_ONCE)
    if coarse.height < 2 or coarse.width < 2:
        raise ValueError("SW-MSA needs at least two windows along each axis")
    # A shift of 0 or of a whole window would label every token alike (an all-zero mask);
    # the cyclic shift refuses it too. Client-side: nothing reaches the graph.
    shift = _resolve_shift(shift, window)
    rows_differ, cols_differ = region_boundaries(ee, window, shift)
    i, j = window_coordinates(ee, coarse)          # periodic window index
    last_row = i.eq(coarse.height - 1)
    last_col = j.eq(coarse.width - 1)
    different = (ee.Image(rows_differ).multiply(last_row)
                 .max(ee.Image(cols_differ).multiply(last_col)))
    # 0 - |value| * different: +0.0 where allowed (never -0.0, whatever the
    # operand types; IEEE 0.0 * -100 is -0.0), value where not.
    return ee.Image.constant(0).subtract(different.multiply(-value)).toFloat()


# ---------------------------------------------------------------------------
# Cost functions (contract section 9)
# ---------------------------------------------------------------------------
def relative_bias_values(heads: int, *, window: int = WINDOW) -> int:
    """Values embedded for a NumPy table: [(2w-1)^2, heads] (the index and the one-hot
    gather are built on the server from integer sequences)."""
    return (2 * window - 1) ** 2 * heads


def relative_bias_arrays() -> int:
    """Server arrays read for a ServerArray table."""
    return 1


def window_attention_values(dim: int, heads: int, *, window: int = WINDOW) -> int:
    """Values embedded for NumPy weights: the QKV linear (W^T [dim, 3 dim], bias rows
    [w^2, 3 dim]) and the relative-position table. The output projection is not part of
    ``window_attention``."""
    return linear_values(dim, 3 * dim, bias=True, layout=WindowLayout(window)) + relative_bias_values(
        heads, window=window)


def window_attention_arrays() -> int:
    """Server arrays read: the QKV weight and bias, and the relative-position table."""
    return linear_arrays(bias=True) + relative_bias_arrays()


_WINDOWS = (5, 7, 8, 12)
"""The windows P01 validated live."""

METADATA = validate_metadata({
    # Validated: any accepted call (token counts, widths, biases, masks and both scales varied).
    "naive_attention": BlockMetadata(
        name="naive_attention", layout_in="W", layout_out="W", locality=Locality.WINDOW,
        policy=ROUND_EACH_OP, embedded_constants=no_constants, asset_arrays=no_constants,
        status="Ready",
        evidence=("docs/history/building-blocks-2026-10-11.md#36-linear-layers-and-attention", "docs/history/ee_attention/notes/findings.md",
                  "docs/evidence/segformer_b0-live.json"),
        validated=ANY_SETTINGS),
    "split_heads": BlockMetadata(
        name="split_heads", layout_in="W", layout_out="W", locality=Locality.POINTWISE,
        policy=NO_ARITHMETIC, embedded_constants=no_constants, asset_arrays=no_constants,
        status="Ready",
        evidence=("docs/history/building-blocks-2026-10-11.md#36-linear-layers-and-attention", "docs/history/swin_block/notes/findings.md"),
        validated=ANY_SETTINGS),
    "relative_position_bias": BlockMetadata(
        name="relative_position_bias", layout_in="V", layout_out="V", locality=Locality.POINTWISE,
        policy=DOUBLE_ROUND_ONCE, embedded_constants=relative_bias_values, asset_arrays=relative_bias_arrays,
        status="Ready — windows 5, 7, 8, 12",
        evidence=("docs/history/building-blocks-2026-10-11.md#36-linear-layers-and-attention", "docs/history/swin_block/notes/findings.md",
                  "docs/evidence/p01-offline.json",
                  "docs/evidence/live-fixtures/20261013T074301Z-p01a.json", "docs/evidence/live-fixtures/20261013T074330Z-p01b.json", "docs/evidence/live-fixtures/20261013T075542Z-window-relative_position_bias-w12.json"),
        validated=Validated(({"window": _WINDOWS},))),
    "window_attention": BlockMetadata(
        name="window_attention", layout_in="W", layout_out="W", locality=Locality.WINDOW,
        policy=COMPOSITE, embedded_constants=window_attention_values, asset_arrays=window_attention_arrays,
        status="Ready — windows 5, 7, 8, 12",
        evidence=("docs/history/building-blocks-2026-10-11.md#36-linear-layers-and-attention",
                  "parts: linear (double_round_once), naive_attention (round_each_op)",
                  "docs/history/swin_block/notes/findings.md", "docs/history/swin_pair/notes/findings.md",
                  "docs/evidence/p01-offline.json",
                  "docs/evidence/live-fixtures/20261013T074301Z-p01a.json", "docs/evidence/live-fixtures/20261013T074330Z-p01b.json", "docs/evidence/live-fixtures/20261013T075542Z-window-relative_position_bias-w12.json"),
        validated=Validated(({"window": _WINDOWS},))),
    # Validated: each window with its shift w // 2 (or None, which resolves to it), and
    # timm's mask value -100.0, never varied.
    "shifted_window_mask": BlockMetadata(
        name="shifted_window_mask", layout_in="W", layout_out="W", locality=Locality.WINDOW,
        policy=DOUBLE_ROUND_ONCE, embedded_constants=no_constants, asset_arrays=no_constants,
        chip_edge_dependent=True, status="Ready — windows 5, 7, 8, 12; shift ⌊w/2⌋",
        evidence=("docs/history/building-blocks-2026-10-11.md#36-linear-layers-and-attention", "docs/history/swin_pair/notes/findings.md",
                  "docs/evidence/p01-offline.json",
                  "docs/evidence/live-fixtures/20261013T074301Z-p01a.json", "docs/evidence/live-fixtures/20261013T074330Z-p01b.json", "docs/evidence/live-fixtures/20261013T075542Z-window-relative_position_bias-w12.json"),
        validated=Validated(tuple({"window": (w,), "shift": (w // 2, None), "value": (MASK_VALUE,)}
                                  for w in _WINDOWS))),
})
