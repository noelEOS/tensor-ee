"""Phases 2-3, 5: PyTorch's AdaptiveAvgPool2d, derived from the installed runtime.

Bin rule (ATen's start_index / end_index, integer arithmetic): along an axis of n input
pixels pooled to m outputs, output i averages input pixels

    start(i) = floor(i n / m)        end(i) = ceil((i + 1) n / m)        [start, end)

so bins overlap whenever m does not divide n (28 -> 3: [0,10) [9,19) [18,28); 28 -> 6:
[0,5) [4,10) [9,14) [14,19) [18,24) [23,28)), and a 2-D bin is the product of its row and
column bins, averaged over its own area (kh * kw, not n^2 / m^2).  The rule is written
out here (bins) and independently read off F.adaptive_avg_pool2d with one-hot inputs
(onehot_bins); check_rule requires both to agree for every shape the PSP meets.

Float32 semantics of the installed kernels (torch 2.14 CPU), characterized against
emulations (characterize):
  * output size 1: adaptive_avg_pool2d dispatches to mean(-1, -2): torch's own reduction
    order of the float32 sum, then one division by H * W;
  * output size > 1, channels-last input (what SMP's PSP receives: LayerNorm2d permutes
    NHWC -> NCHW without a copy): per bin a float32 sum, sequential over (row, col) in
    row-major order (vectorized across channels), then one float32 division by kh * kw;
  * output size > 1, NCHW-contiguous input: the same sum, divided by kh, then by kw.
    The kernel is chosen from the strides (suggest_memory_format), so a reference must
    reproduce SMP's memory format, not only its values (as_nchw).

The EE implementation instead accumulates in double and rounds once (adaptive_pool.py),
so EE - torch32 is PyTorch's float32 accumulation and division.

The stacked pooled matrix of the PSP is the four pools concatenated in SMP's block order
(sizes 1, 2, 3, 6), each s x s bin row-major: 1 + 4 + 9 + 36 = 50 rows.
"""

from __future__ import annotations

from tensor_ee.blocks.window.pooling import bins, check_rule, pool_rows, offsets, axis_membership
from tensor_ee.blocks.window.resample import window_token_order

import math

import numpy as np
import torch
import torch.nn.functional as F

SIZES = (1, 2, 3, 6)
WINDOW = 7


# ---------------------------------------------------------------------------
# Bins
# ---------------------------------------------------------------------------


def overlaps(n: int, m: int) -> list[int]:
    """Input pixels that belong to two consecutive bins."""
    b = bins(n, m)
    return [p for (s0, e0), (s1, _) in zip(b, b[1:]) for p in range(s1, e0)]


# ---------------------------------------------------------------------------
# The stacked PSP pooling map
# ---------------------------------------------------------------------------
def membership(h: int, w: int, sizes=SIZES) -> tuple[np.ndarray, np.ndarray]:
    """(A [R, h w] 0/1 over row-major pixels, areas [R]).

    Moved from upernet/earth_engine/_coefficients.py (the research global pool's table) at
    L06b: the stacked map built from the library's per-axis tables and bin areas."""
    ry, rx = axis_membership(h, w, sizes)
    a = np.einsum("ki,kj->kij", ry, rx).reshape(len(ry), h * w)
    areas = np.array([r["area"] for r in pool_rows(h, w, sizes)], dtype=np.float64)
    if not np.array_equal(a.sum(1), areas):
        raise AssertionError("membership rows do not sum to the bin areas")
    return a, areas


def index_map_membership(h: int, w: int, sizes=SIZES) -> tuple[np.ndarray, np.ndarray]:
    """(A [R, h w] 0/1 over row-major pixels, areas [R]) read off PyTorch alone (task P04).

    No library code. Pixel t = i w + j gets its own one-hot index map (channel t is 1 at
    pixel t and 0 elsewhere); ``F.adaptive_avg_pool2d`` pools the [1, h w, h, w] stack to
    s x s in float64, and an output cell's members are the channels it reads with a non-zero
    weight. Every member of a cell must carry the same weight, 1 / count to 1e-12 (PyTorch
    divides by the cell's own area, at once or by its rows and then its columns, so the
    last bit may differ from 1 / count); the area is the count. Sizes in ``sizes`` order,
    cells row-major, as ``membership``. With more bins than pixels (5 -> 6), cells overlap."""
    n = h * w
    maps = torch.eye(n, dtype=torch.float64).reshape(1, n, h, w)
    rows, areas = [], []
    for s in sizes:
        with torch.no_grad():
            pooled = F.adaptive_avg_pool2d(maps, s)[0]                    # [n, s, s]
        for p in range(s):
            for q in range(s):
                weights = pooled[:, p, q]
                members = weights != 0
                count = int(members.sum())
                if count == 0 or not bool((weights[members] == weights[members][0]).all()):
                    raise AssertionError(f"{h} x {w} -> {s}: cell ({p}, {q}) does not weigh its members equally")
                if abs(float(weights[members][0]) * count - 1.0) > 1e-12:
                    raise AssertionError(f"{h} x {w} -> {s}: cell ({p}, {q}) is not a mean over its members")
                rows.append(members.numpy().astype(np.float64))
                areas.append(float(count))
    return np.stack(rows), np.array(areas, dtype=np.float64)


def window_order(h: int, w: int, window: int = WINDOW) -> np.ndarray:
    """Row-major pixel index of each window-layout token (windows row-major, then tokens
    row-major within the window): Earth Engine's partial and gather order. No library code;
    it equals ``window_token_order`` (tests/test_p04_decoder_blocks.py)."""
    if h % window or w % window:
        raise ValueError("the extent must be a multiple of the window")
    index = np.arange(h * w).reshape(h // window, window, w // window, window)
    return index.transpose(0, 2, 1, 3).reshape(-1)


def split(pooled: np.ndarray, sizes=SIZES) -> dict[int, np.ndarray]:
    """[R, C] stacked -> {s: [s, s, C]}."""
    return {s: pooled[a:b].reshape(s, s, -1) for s, (a, b) in offsets(sizes).items()}


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------
def torch_pool(x: np.ndarray, s: int, dtype=torch.float32, channels_last: bool = True) -> np.ndarray:
    """F.adaptive_avg_pool2d on [H, W, C] (as NCHW; channels_last: the memory format SMP's
    PSP receives).  Returns [s, s, C]."""
    return pool_nchw(as_nchw(x, dtype, channels_last), s)


def as_nchw(x: np.ndarray, dtype=torch.float32, channels_last: bool = True) -> torch.Tensor:
    """[H, W, C] -> [1, C, H, W] in an explicit memory format.  A bare
    from_numpy(x).permute(2, 0, 1)[None] is NOT dispatched as channels-last (its batch
    stride is C, and PyTorch's suggest_memory_format then picks the NCHW kernel, which
    divides by kh and kw separately, although is_contiguous(channels_last) is True, so
    .contiguous(memory_format=channels_last) alone does not copy); SMP's LayerNorm2d
    output has the full channels-last strides (H W C, 1, W C, C)."""
    t = torch.from_numpy(np.ascontiguousarray(x)).to(dtype).permute(2, 0, 1)[None].contiguous()
    return t.contiguous(memory_format=torch.channels_last) if channels_last else t


def pool_nchw(t: torch.Tensor, s: int) -> np.ndarray:
    with torch.no_grad():
        return F.adaptive_avg_pool2d(t, s)[0].permute(1, 2, 0).numpy()


def exact_pool(x: np.ndarray, sizes=SIZES) -> np.ndarray:
    """[R, C]: the exact mean of every bin (math.fsum), correctly rounded to float32."""
    h, w, c = np.shape(x)
    flat = np.asarray(x, np.float64).reshape(h * w, c)
    out = np.empty((sum(s * s for s in sizes), c))
    for k, r in enumerate(pool_rows(h, w, sizes)):
        cells = flat.reshape(h, w, c)[r["r0"]:r["r1"], r["c0"]:r["c1"]].reshape(-1, c)
        out[k] = [math.fsum(col) / r["area"] for col in cells.T]
    return out.astype(np.float32).astype(np.float64)


def emulate_float32(x: np.ndarray, s: int, division: str = "area") -> np.ndarray:
    """Sequential row-major float32 sum per bin, then division ("area": / (kh kw) once,
    "rows_cols": / kh then / kw), in float32.  Returns [s, s, C]."""
    h, w, c = np.shape(x)
    x32 = np.asarray(x, np.float32)
    out = np.empty((s, s, c), np.float32)
    for p, (r0, r1) in enumerate(bins(h, s)):
        for q, (c0, c1) in enumerate(bins(w, s)):
            total = np.zeros(c, np.float32)
            for i in range(r0, r1):
                for j in range(c0, c1):
                    total = total + x32[i, j]
            kh, kw = np.float32(r1 - r0), np.float32(c1 - c0)
            out[p, q] = total / (kh * kw) if division == "area" else (total / kh) / kw
    return out


def characterize(x: np.ndarray, sizes=SIZES) -> dict:
    """Bit-equality fractions of torch's float32 pooling (both memory formats) against the
    candidate float32 formulations, and of the exactly rounded mean."""
    out = {}
    for s in sizes:
        row = {}
        for fmt, cl in (("channels_last", True), ("contiguous", False)):
            got = torch_pool(x, s, channels_last=cl)
            cands = {"seq_rowmajor_div_area": emulate_float32(x, s, "area"),
                     "seq_rowmajor_div_rows_then_cols": emulate_float32(x, s, "rows_cols"),
                     "exact_mean_rounded": exact_pool(x, (s,)).reshape(s, s, -1)}
            if s == 1:
                t = as_nchw(x, channels_last=cl)
                with torch.no_grad():
                    cands["torch_sum_div_N"] = (t.sum((-1, -2)) / (x.shape[0] * x.shape[1]))[0].numpy()[None, None]
                    cands["torch_mean"] = t.mean((-1, -2))[0].numpy()[None, None]
            row[fmt] = {k: float((v.astype(np.float32) == got).mean()) for k, v in cands.items()}
        out[s] = row
    return out


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def onehot_fixture(h: int, w: int) -> np.ndarray:
    """[h, w, h w]: token t = i w + j carries (t + 1) in channel t, 0 elsewhere.  Pooled
    sums are (t + 1) exactly where t belongs to the bin: the exact source set, and which
    token it is (unique ID), with no possible cancellation."""
    x = np.zeros((h, w, h * w), np.float32)
    t = np.arange(h * w)
    x.reshape(h * w, h * w)[t, t] = t + 1
    return x


def cancellation_fixture(h: int, w: int, c: int, s: int, seed: int = 0,
                         big: float = 2.0 ** 40) -> np.ndarray:
    """N(0, 1) float32 tokens plus, inside every bin of the s x s pool, a +big token at the
    first and a -big token at the last position of the bin's exclusive interior (rows and
    columns shared with no neighbouring bin of that pool).  The bigs cancel exactly in
    every bin of that pool (and in the global bin), but while one is in a running double
    sum it absorbs the low bits of the small values added meanwhile, so the result depends
    on the summation order: it separates EE's order from the alternatives.  Per-channel
    scales 2^-4 .. 2^4 and signs."""
    g = np.random.default_rng(seed)
    x = g.standard_normal((h, w, c)).astype(np.float32)
    scale = (big * 2.0 ** g.integers(-4, 5, size=c)).astype(np.float32)
    sign = np.where(g.random(c) < 0.5, -1.0, 1.0).astype(np.float32)

    def interior(n):
        b = bins(n, s)
        out = []
        for i, (start, end) in enumerate(b):
            lo = b[i - 1][1] if i else start                 # past the previous bin's end
            hi = b[i + 1][0] if i + 1 < len(b) else end      # before the next bin's start
            out.append((lo, hi - 1))
        return out

    for (r_first, r_last) in interior(h):
        for (c_first, c_last) in interior(w):
            x[r_first, c_first] = sign * scale
            x[r_last, c_last] = -sign * scale
    return x
