"""Milestone 9, Phase 11: the float64 mirror of the EE FPN (upernet/earth_engine/fpn.py),
rounding to float32 exactly where EE rounds.  Milestone 7's decoder mirror supplies every
primitive and Milestone 8's the PSP; this module fixes only the composition order of the
installed recursion (reference/smp_fpn_reference.py):

    for k = 3, 2, 1:
        C{k}_norm = LayerNorm2d(C_k)                     sequential double sums, eps 1e-6
        lat{k}    = ReLU(BN(Conv1x1(C{k}_norm)))         one rounding per op
        up{k}     = bilinear x2 of the state             double products of PyTorch's
                                                         dyadic weights (exact), one rounding
        F{k}      = up{k} + lat{k}                       one rounding; F{k} is the next state
    P{k} = ReLU(BN(Conv3x3(F{k})))                       window im2col: one sequential double
                                                         dot over (ky, kx, c) per output

The 3 x 3 convolution is evaluated in row blocks of at most 3,136 output pixels (the row
count Milestone 8 validated bit for bit), not as one matrix product of up to 200,704 rows:
`sequential_rows_agree` checks that torch's float64 matmul still accumulates each dot
product strictly sequentially at the block shapes used.
"""

from __future__ import annotations

import numpy as np
import torch

from reference.upernet import ee_mirror_decoder as M
from reference.upernet import ee_mirror_psp as MP

LEVELS = (3, 2, 1)
BLOCK_ROWS = 3136                                   # output pixels per conv block


def conv3x3(x, weight, exact: bool = False, block_rows: int = BLOCK_ROWS) -> np.ndarray:
    """M.conv3x3 (window im2col order, zero padding) evaluated in row blocks."""
    x = np.asarray(x, np.float32)
    h, w, c = x.shape
    rows = max(1, min(h, block_rows // w))
    padded = np.pad(x, ((1, 1), (1, 1), (0, 0)))
    wt = M.conv_rows(weight)                         # [9 C, Cout] (ky, kx, c) rows
    out = np.empty((h, w, wt.shape[1]), np.float64)
    for r0 in range(0, h, rows):
        r1 = min(h, r0 + rows)
        cols = np.concatenate([padded[r0 + dy:r1 + dy, dx:dx + w] for dy in range(3)
                               for dx in range(3)], axis=-1)
        y = M.t64(cols).reshape(-1, 9 * c) @ wt
        out[r0:r1] = (y if exact else M.r32(y)).reshape(r1 - r0, w, -1).numpy()
    return out


def bn_args(decoder: dict, prefix: str):
    return tuple(decoder[f"{prefix}.{k}"] for k in ("weight", "bias", "running_mean", "running_var"))


def lateral(n, decoder: dict, level: int) -> dict[str, np.ndarray]:
    from reference.upernet.smp_fpn_reference import lateral_prefix
    p = lateral_prefix(level)
    w = decoder[f"{p}.0.weight"]
    conv = M.linear(n, w.reshape(w.shape[0], -1))
    bn = M.batch_norm(conv, *bn_args(decoder, f"{p}.1"))
    return {f"lat{level}_conv": conv, f"lat{level}_bn": bn, f"lat{level}": M.relu(bn)}


def conv_block(f, decoder: dict, level: int) -> dict[str, np.ndarray]:
    from reference.upernet.smp_fpn_reference import conv_prefix
    p = conv_prefix(level)
    conv = conv3x3(f, decoder[f"{p}.0.weight"])
    bn = M.batch_norm(conv, *bn_args(decoder, f"{p}.1"))
    return {f"P{level}_conv": conv, f"P{level}_bn": bn, f"P{level}": M.relu(bn)}


def top_down(state, laterals: dict[int, np.ndarray]) -> dict[str, np.ndarray]:
    """The recursion alone: state (PSP output) and lat{k} for k = 3, 2, 1 -> up{k}, F{k}."""
    out = {}
    for k in LEVELS:
        lat = laterals[k]
        out[f"up{k}"] = M.resize(state, np.shape(lat)[:2])
        out[f"F{k}"] = M.add(out[f"up{k}"], lat)
        state = out[f"F{k}"]
    return out


def fpn(feats, decoder: dict, psp: dict | None = None, convs: bool = True) -> dict[str, np.ndarray]:
    """Every PSP and FPN intermediate (NHWC) with EE's roundings from raw C1..C4."""
    from reference.upernet.smp_fpn_reference import norm_prefix
    c1, c2, c3, c4 = feats
    out = dict(MP.psp(c4, decoder) if psp is None else psp)
    raw = {1: c1, 2: c2, 3: c3}
    for k in (1, 2, 3):
        out[f"C{k}_raw"] = np.asarray(raw[k], np.float64)
        out[f"C{k}_norm"] = M.layer_norm2d(raw[k], decoder[f"{norm_prefix(k)}.weight"],
                                           decoder[f"{norm_prefix(k)}.bias"])
    laterals = {}
    for k in LEVELS:
        out.update(lateral(out[f"C{k}_norm"], decoder, k))
        laterals[k] = out[f"lat{k}"]
    out.update(top_down(out["psp_relu"], laterals))
    if convs:
        for k in LEVELS:
            out.update(conv_block(out[f"F{k}"], decoder, k))
    return out


def padded_variant(f, weight, mode: str) -> np.ndarray:
    """The 3 x 3 conv with a different border rule (negative control for Phase 15):
    'replicate' / 'reflect' padding instead of PyTorch's zeros."""
    x = torch.from_numpy(np.asarray(f, np.float32)).double().permute(2, 0, 1)[None]
    padded = torch.nn.functional.pad(x, (1, 1, 1, 1), mode=mode)[0].permute(1, 2, 0).numpy()
    h, w, c = np.shape(f)
    cols = np.concatenate([padded[dy:dy + h, dx:dx + w] for dy in range(3) for dx in range(3)], -1)
    return M.r32(M.t64(cols).reshape(-1, 9 * c) @ M.conv_rows(weight)).reshape(h, w, -1).numpy()


def sequential_rows_agree(rows: int = BLOCK_ROWS, k: int = 2304, cols: int = 8, seed: int = 0) -> bool:
    """torch float64 [rows, k] @ [k, cols] == strictly sequential accumulation, at the
    mirror's conv block shape (Milestone 5's finding re-checked at the FPN's shapes)."""
    g = np.random.default_rng(seed)
    a = torch.from_numpy(g.standard_normal((rows, k)).astype(np.float32)).double()
    b = torch.from_numpy(g.standard_normal((k, cols)).astype(np.float32)).double()
    seq = torch.zeros(rows, cols, dtype=torch.float64)
    for i in range(k):
        seq = seq + a[:, i:i + 1] * b[i:i + 1]
    return bool(torch.equal(seq, a @ b))
