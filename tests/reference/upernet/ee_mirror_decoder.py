"""Phase 18: a float64 mirror of the EE decoder primitives, rounding to float32 exactly
where the EE implementation rounds (Milestones 2-6's policy: double inside one PyTorch
operation, one float32 rounding at its end).

    EE - torch32 = (EE - mirror) + (mirror - torch32)

The first term is implementation fidelity (0 when EE is exact to its own specification),
the second PyTorch's float32 kernels.  Tensors are NHWC [H, W, C] (or [..., C]); values
are float32 held in float64.

Accumulation orders (established in Milestone 5): matrixMultiply sums each dot product
sequentially in double, which torch's float64 matmul reproduces bit for bit;
arrayReduce(sum) sums left to right (reference.swin_block.ee_mirror.sequential_sum).
"""

from __future__ import annotations

import numpy as np
import torch

from reference.swin_block.ee_mirror import layer_norm_formula, sequential_sum
from reference.upernet.pytorch_decoder_primitives import bilinear_weights, im2col


def t64(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.double()
    return torch.from_numpy(np.asarray(x, dtype=np.float32)).double()


def r32(t: torch.Tensor) -> torch.Tensor:
    return t.float().double()


def out(t: torch.Tensor, exact: bool) -> np.ndarray:
    return (t if exact else r32(t)).numpy()


def layer_norm2d(x, gamma, beta, eps: float = 1e-6, exact: bool = False) -> np.ndarray:
    """Milestone 2's layer_norm over the channel axis, EE's sequential sums, eps 1e-6."""
    r = (lambda v: v) if exact else r32
    return layer_norm_formula(t64(x), t64(gamma), t64(beta), r, eps, sequential_sum).numpy()


def linear(x, weight, bias=None, exact: bool = False) -> np.ndarray:
    """x [..., Cin] @ W^T + b (W [Cout, Cin]): sequential double dot products, one rounding."""
    y = t64(x) @ t64(weight).T
    if bias is not None:
        y = y + t64(bias)
    return out(y, exact)


def conv_rows(weight) -> torch.Tensor:
    """[Cout, Cin, k, k] -> [k k Cin, Cout] in (ky, kx, c) row order."""
    w = t64(weight)
    return w.permute(2, 3, 1, 0).reshape(-1, w.shape[0])


def conv3x3(x, weight, bias=None, method: str = "window_im2col", exact: bool = False):
    """Every EE formulation of conv2d.py: im2col orders (one sequential dot over (ky, kx, c)),
    taps (nine sequential dots over c, summed tap by tap), convolve (per input channel a
    row-major kernel sum, then channels in order)."""
    x64 = t64(x)
    k = np.shape(weight)[2]
    if method in ("window_im2col", "dense_nbr", "dense_bands"):
        y = t64(im2col(np.asarray(x, np.float32), k)) @ conv_rows(weight)
    elif method in ("window_taps", "dense_taps"):
        cols = t64(im2col(np.asarray(x, np.float32), k))
        cin = x64.shape[-1]
        rows = conv_rows(weight)
        y = None
        for t in range(k * k):
            term = cols[..., t * cin:(t + 1) * cin] @ rows[t * cin:(t + 1) * cin]
            y = term if y is None else y + term
    elif method == "convolve":
        cols = t64(im2col(np.asarray(x, np.float32), k))           # (ky, kx, c)
        h, w, _ = x64.shape
        cin = x64.shape[-1]
        per = cols.reshape(h, w, k * k, cin)
        wt = t64(weight)                                           # [Cout, Cin, k, k]
        y = None
        for c in range(cin):
            term = per[..., c] @ wt[:, c].reshape(wt.shape[0], -1).T  # row-major kernel order
            y = term if y is None else y + term
    else:
        raise ValueError(method)
    if bias is not None:
        y = y + t64(bias)
    return out(y, exact)


def batch_norm(x, gamma, beta, mean, var, eps: float = 1e-5, exact: bool = False):
    """((x - mean) / sqrt(var + eps)) * gamma + beta in double (EE's order), rounded once."""
    std = torch.sqrt(t64(var) + eps)
    return out((t64(x) - t64(mean)) / std * t64(gamma) + t64(beta), exact)


def relu(x) -> np.ndarray:
    return torch.clamp_min(t64(x), 0).numpy()


def resize(x, size, exact: bool = False) -> np.ndarray:
    """Separable bilinear with PyTorch's own weights: My X Mx^T, exact in float64 for the
    dyadic weights of integer factors (any order), rounded once."""
    h, w, _ = np.shape(x)
    my = torch.from_numpy(bilinear_weights(h, size[0]))
    mx = torch.from_numpy(bilinear_weights(w, size[1]))
    y = torch.einsum("yi,ijc->yjc", my, t64(x))
    y = torch.einsum("xj,yjc->yxc", mx, y)
    return out(y, exact)


def matrix_resize(x, size, exact: bool = False) -> np.ndarray:
    """resize.matrix_resize: per output pixel a sequential double dot of the kron weights
    (products of PyTorch's float32 1-D weights) with the row-major source pixels."""
    from tensor_ee.blocks.window.resample import output_weights
    h, w, c = np.shape(x)
    weights = torch.from_numpy(output_weights((h, w), tuple(size)))
    y = weights @ t64(x).reshape(h * w, c)
    return out(y.reshape(size[0], size[1], c), exact)


def add(a, b) -> np.ndarray:
    return r32(t64(a) + t64(b)).numpy()


def concat(parts) -> np.ndarray:
    return np.concatenate([np.asarray(p, np.float64) for p in parts], axis=-1)


def sequential_matmul_agrees(rows: int = 64, k: int = 2304, cols: int = 16, seed: int = 0) -> bool:
    """torch float64 matmul == strictly sequential accumulation (Milestone 5's finding,
    re-checked at the decoder's 3 x 3 im2col length)."""
    g = np.random.default_rng(seed)
    a = torch.from_numpy(g.standard_normal((rows, k)).astype(np.float32)).double()
    b = torch.from_numpy(g.standard_normal((k, cols)).astype(np.float32)).double()
    seq = torch.cumsum(a[:, :, None] * b[None], dim=1)[:, -1]
    return bool(torch.equal(seq, a @ b))
