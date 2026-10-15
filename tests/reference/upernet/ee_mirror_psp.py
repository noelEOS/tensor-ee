"""Phase 12: the float64 mirror of the EE PSP (upernet/earth_engine/psp.py), rounding to
float32 exactly where EE rounds.  Milestone 7's decoder mirror is reused for every
primitive it already covers; the new operation is adaptive average pooling.

EE's pooling (adaptive_pool.window_pool), in order:
  1. per window (I, J), a [R, 49] x [49, C] matrixMultiply of the 0/1 bin membership with
     the window's float32 tokens: R = 50 bin sums over the window's tokens, accumulated
     sequentially in double in token order (7a + b);
  2. the n_h x n_w window partials gathered (sampleRectangle, row-major windows) and summed
     with ee.Array.reduce(sum) over the window axis: sequentially in double, window order
     I n_w + J;
  3. divided by the bin area (kh kw, an exact integer) in double, rounded to float32 once.
`pool_orders` evaluates the alternatives (one global sequential dot in window-token order,
row-major pixel order, the exact sum) so that order-revealing fixtures can tell them apart.

Task P04 parameterized the pooling: ``pool_sums`` and ``pool`` take ``sizes`` (default
(1, 2, 3, 6)), ``window`` (default 7; [R, w^2] per window) and ``source``, the bin tables:
the library's (default) or PyTorch's own, read off ``adaptive_avg_pool2d`` on one-hot index
maps with no library code (``adaptive_pool_reference.index_map_membership``). The window-token
order is computed here (``adaptive_pool_reference.window_order``), not taken from the library.

The branches run in value space on [s^2, 768] arrays (Milestone 2's linear, BatchNorm,
ReLU); the resize is Milestone 7's matrix formulation (per output pixel one sequential
double dot over the s^2 source pixels, row-major, with products of PyTorch's float32 1-D
weights), the output conv Milestone 7's window im2col.
"""

from __future__ import annotations

import numpy as np
import torch

from reference.upernet import adaptive_pool_reference as AP
from reference.upernet import ee_mirror_decoder as M

SIZES = AP.SIZES


def t64(x) -> torch.Tensor:
    return M.t64(x)


TABLES = ("library", "index_maps")
"""Where the bin tables come from: the library's per-axis tables (``AP.membership``, the
default) or PyTorch's ``adaptive_avg_pool2d`` on one-hot index maps
(``AP.index_map_membership``, no library code; task P04)."""


def tables(h: int, w: int, sizes=SIZES, source: str = "library") -> tuple[np.ndarray, np.ndarray]:
    """(A [R, h w], areas [R]) of the stacked pools, from ``source`` (``TABLES``)."""
    if source == "library":
        return AP.membership(h, w, sizes)
    if source == "index_maps":
        return AP.index_map_membership(h, w, sizes)
    raise ValueError(f"tables come from one of {TABLES}; received {source!r}")


def pool_sums(x, order: str = "window", *, sizes=SIZES, window: int = AP.WINDOW,
              source: str = "library") -> torch.Tensor:
    """[R, C] double bin sums of x [H, W, C] in one of EE's candidate orders. ``sizes`` and
    ``window`` default to the PSP's (1, 2, 3, 6) and 7; ``source`` names the bin tables."""
    h, w, c = np.shape(x)
    a, _ = tables(h, w, sizes, source)
    flat = t64(x).reshape(h * w, c)
    if order == "rowmajor":
        return torch.from_numpy(a) @ flat
    perm = AP.window_order(h, w, window)
    aw, xw = torch.from_numpy(a[:, perm]), flat[perm]
    if order == "global":
        return aw @ xw
    if order == "window":
        n = (h // window) * (w // window)
        k = window * window
        partial = torch.stack([aw[:, i * k:(i + 1) * k] @ xw[i * k:(i + 1) * k] for i in range(n)])
        return torch.cumsum(partial, 0)[-1]
    if order == "exact":
        import math
        a_np, f = a, flat.numpy()
        return torch.tensor([[math.fsum(f[a_np[r] > 0, ch]) for ch in range(c)]
                             for r in range(a.shape[0])], dtype=torch.float64)
    raise ValueError(order)


def pool(x, order: str = "window", exact: bool = False, *, sizes=SIZES, window: int = AP.WINDOW,
         source: str = "library") -> np.ndarray:
    """[R, C] pooled means: sums / area in double, one float32 rounding."""
    h, w, _ = np.shape(x)
    _, areas = tables(h, w, sizes, source)
    sums = pool_sums(x, order, sizes=sizes, window=window, source=source)
    return M.out(sums / torch.from_numpy(areas)[:, None], exact)


def pool_orders(x) -> dict[str, np.ndarray]:
    return {order: pool(x, order) for order in ("window", "global", "rowmajor", "exact")}


def branch(pooled_s, decoder: dict, i: int) -> dict[str, np.ndarray]:
    """Conv 1x1 768 -> 256 (no bias), BatchNorm, ReLU on a pooled [s, s, 768] map."""
    from reference.upernet.smp_psp_reference import prefix
    w = decoder[f"{prefix(i)}.0.weight"]
    conv = M.linear(pooled_s, w.reshape(w.shape[0], -1))
    bn = M.batch_norm(conv, *(decoder[f"{prefix(i)}.1.{k}"] for k in
                              ("weight", "bias", "running_mean", "running_var")))
    return {"conv": conv, "bn": bn, "relu": M.relu(bn)}


def psp(c4_raw, decoder: dict, c4_is_normalized: bool = False) -> dict[str, np.ndarray]:
    """Every PSP intermediate (NHWC; pooled maps [s, s, C]) with EE's roundings."""
    from reference.upernet.smp_psp_reference import NORM_PREFIX, OUT_PREFIX
    out = {}
    if c4_is_normalized:
        n4 = np.asarray(c4_raw, np.float64)
    else:
        out["C4_raw"] = np.asarray(c4_raw, np.float64)
        n4 = M.layer_norm2d(c4_raw, decoder[f"{NORM_PREFIX}.weight"], decoder[f"{NORM_PREFIX}.bias"])
    out["C4_norm"] = n4
    h, w, _ = n4.shape
    pooled = AP.split(pool(n4))
    parts = [n4]
    for i, s in enumerate(SIZES):
        out[f"pool_{s}"] = pooled[s]
        b = branch(pooled[s], decoder, i)
        for step, value in b.items():
            out[f"branch_{s}_{step}"] = value
        up = M.matrix_resize(b["relu"], (h, w))
        out[f"branch_{s}_up"] = up
        parts.append(up)
    out["psp_concat"] = M.concat(parts)
    out["psp_conv"] = M.conv3x3(out["psp_concat"], decoder[f"{OUT_PREFIX}.0.weight"])
    out["psp_bn"] = M.batch_norm(out["psp_conv"], *(decoder[f"{OUT_PREFIX}.1.{k}"] for k in
                                                    ("weight", "bias", "running_mean", "running_var")))
    out["psp_relu"] = M.relu(out["psp_bn"])
    return out


# ---------------------------------------------------------------------------
# Option C (Phases 19-20): a tile of the whole-area PSP from shared pooled branches
# ---------------------------------------------------------------------------
