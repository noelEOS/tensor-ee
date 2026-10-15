"""Phases 7-16, 22: PyTorch references of the UPerNet decoder primitives, run with the
installed SMP modules wherever the decoder has one.

Tensors are NHWC NumPy arrays [H, W, C] (the dense raster layout: rows, columns,
channel bands); torch sees them as NCHW [1, C, H, W].

    layer_norm2d      SMP's LayerNorm2d (decoder.feature_norms[k])
    conv2d            torch.nn.Conv2d (or an SMP Conv2dReLU's conv) with its padding
    batch_norm        torch.nn.BatchNorm2d in eval mode
    conv_block        SMP Conv2dReLU: conv -> BatchNorm2d -> ReLU, as the decoder runs it
    interpolate       F.interpolate(mode="bilinear", align_corners=False, size=...), the
                      call UPerNet makes for every resize
    bilinear_weights  PyTorch's own 1-D interpolation weights for (in, out), extracted by
                      interpolating one-hot rows (no re-derivation)
    add, concat       float32 addition; torch.cat along channels

Fixtures with unique values (IDs, coordinate ramps) prove ordering exactly before any
floating-point comparison.
"""

from __future__ import annotations

from tensor_ee.blocks.window.resample import bilinear_weights

import numpy as np
import torch
import torch.nn.functional as F


def nchw(x: np.ndarray, dtype=torch.float32) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)[None].to(dtype)


def nhwc(t: torch.Tensor) -> np.ndarray:
    return t.detach()[0].permute(1, 2, 0).contiguous().numpy()


def _cast(module, dtype):
    return module.to(dtype).eval()


# ---------------------------------------------------------------------------
# Normalization and activation
# ---------------------------------------------------------------------------
def layer_norm2d(x: np.ndarray, gamma, beta, eps: float = 1e-6, dtype=torch.float32) -> np.ndarray:
    from segmentation_models_pytorch.decoders.upernet.decoder import LayerNorm2d
    norm = LayerNorm2d(x.shape[-1], eps=eps)
    with torch.no_grad():
        norm.weight.copy_(torch.from_numpy(np.asarray(gamma)))
        norm.bias.copy_(torch.from_numpy(np.asarray(beta)))
        return nhwc(_cast(norm, dtype)(nchw(x, dtype)))


def batch_norm_module(gamma, beta, mean, var, eps: float = 1e-5) -> torch.nn.BatchNorm2d:
    bn = torch.nn.BatchNorm2d(len(gamma), eps=eps)
    with torch.no_grad():
        for name, value in (("weight", gamma), ("bias", beta), ("running_mean", mean),
                            ("running_var", var)):
            getattr(bn, name).copy_(torch.from_numpy(np.asarray(value, np.float32)))
    return bn.eval()


def relu(x: np.ndarray, dtype=torch.float32) -> np.ndarray:
    """SMP's nn.ReLU(inplace=True), applied to a copy (from_numpy shares the caller's memory)."""
    with torch.no_grad():
        return nhwc(torch.nn.ReLU(inplace=True)(nchw(np.array(x, copy=True), dtype)))


# ---------------------------------------------------------------------------
# Convolution
# ---------------------------------------------------------------------------
def conv_module(weight: np.ndarray, bias=None) -> torch.nn.Conv2d:
    cout, cin, kh, kw = weight.shape
    conv = torch.nn.Conv2d(cin, cout, (kh, kw), padding=(kh // 2, kw // 2), bias=bias is not None)
    with torch.no_grad():
        conv.weight.copy_(torch.from_numpy(np.asarray(weight, np.float32)))
        if bias is not None:
            conv.bias.copy_(torch.from_numpy(np.asarray(bias, np.float32)))
    return conv.eval()


def conv2d(x: np.ndarray, weight: np.ndarray, bias=None, dtype=torch.float32) -> np.ndarray:
    """Conv2d(stride 1, padding k // 2 (zeros), dilation 1, groups 1): SMP's decoder convs."""
    with torch.no_grad():
        return nhwc(_cast(conv_module(weight, bias), dtype)(nchw(x, dtype)))


def conv_block(x: np.ndarray, weight, gamma, beta, mean, var, dtype=torch.float32,
               names=("conv", "bn", "relu")) -> dict[str, np.ndarray]:
    """SMP's Conv2dReLU (conv without bias -> BatchNorm2d -> ReLU) step by step; raises
    unless the steps equal the SMP module's own forward bit for bit."""
    from segmentation_models_pytorch.base.modules import Conv2dReLU
    cout, cin, k, _ = np.shape(weight)
    block = Conv2dReLU(cin, cout, k, padding=k // 2, use_norm="batchnorm")
    with torch.no_grad():
        block[0].weight.copy_(torch.from_numpy(np.asarray(weight, np.float32)))
        block[1].load_state_dict(batch_norm_module(gamma, beta, mean, var).state_dict())
    block = _cast(block, dtype)
    t = nchw(x, dtype)
    with torch.no_grad():
        conv = block[0](t)
        bn = block[1](conv)
        out = torch.relu(bn)
        if not torch.equal(out, block(t)):
            raise AssertionError("explicit Conv2dReLU differs from SMP's module")
    return {names[0]: nhwc(conv), names[1]: nhwc(bn), names[2]: nhwc(out)}


def im2col(x: np.ndarray, k: int = 3) -> np.ndarray:
    """[H, W, C] -> [H, W, k*k*C]: zero-padded neighbourhoods in (ky, kx, c) order
    (values moved, not computed)."""
    r = k // 2
    padded = np.pad(x, ((r, r), (r, r), (0, 0)))
    h, w, _ = x.shape
    return np.concatenate([padded[dy:dy + h, dx:dx + w] for dy in range(k) for dx in range(k)],
                          axis=-1)


def conv_weight_rows(weight: np.ndarray) -> np.ndarray:
    """Conv2d weight [Cout, Cin, kh, kw] -> W^T [kh*kw*Cin, Cout] in the (ky, kx, c) row
    order of im2col (and of EE's neighborhoodToArray gather)."""
    return np.asarray(weight).transpose(2, 3, 1, 0).reshape(-1, np.shape(weight)[0])


# ---------------------------------------------------------------------------
# Bilinear resize
# ---------------------------------------------------------------------------
def interpolate(x: np.ndarray, size, dtype=torch.float32) -> np.ndarray:
    """UPerNet's resize: F.interpolate(x, size=size, mode="bilinear", align_corners=False)."""
    with torch.no_grad():
        return nhwc(F.interpolate(nchw(x, dtype), size=tuple(size), mode="bilinear",
                                  align_corners=False))


def bilinear_formula(n_in: int, n_out: int) -> list[tuple[float, int, int, float, float]]:
    """PyTorch's align_corners=False rule per output index, written out:
    src = max((o + 1/2) * n_in / n_out - 1/2, 0); i0 = floor(src);
    i1 = min(i0 + 1, n_in - 1); l1 = src - i0; l0 = 1 - l1.  Returns (src, i0, i1, l0, l1)."""
    rows = []
    for o in range(n_out):
        src = max((o + 0.5) * n_in / n_out - 0.5, 0.0)
        i0 = int(np.floor(src))
        i1 = min(i0 + 1, n_in - 1)
        rows.append((src, i0, i1, 1.0 - (src - i0), src - i0))
    return rows


def align_corners_formula(n_in: int, n_out: int, precision: str = "float32"
                          ) -> list[tuple[float, int, int, float, float]]:
    """PyTorch's align_corners=True rule per output index, written out (ATen's
    area_pixel_compute_scale / area_pixel_compute_source_index / guard_index_and_lambda,
    UpSample.h), in the arithmetic of `precision`:

        scale = (n_in - 1) / (n_out - 1)       (0 when n_out == 1)
        src   = scale * o
        i0    = min(floor(src), n_in - 1);  i1 = i0 + (1 if i0 < n_in - 1 else 0)
        l1    = clamp(src - i0, 0, 1);      l0 = 1 - l1
        (n_out == n_in: a copy, i0 = i1 = o, l0 = 1, l1 = 0)

    precision "float32": every operation rounded to float32, as the float32 model runs it;
    "exact": rational arithmetic (the textbook o (n_in - 1) / (n_out - 1)), returned as the
    nearest doubles.  Returns (src, i0, i1, l0, l1) per output index."""
    from fractions import Fraction
    rows = []
    f32 = np.float32
    for o in range(n_out):
        if n_out == n_in:
            rows.append((float(o), o, o, 1.0, 0.0))
            continue
        if precision == "float32":
            scale = f32(n_in - 1) / f32(n_out - 1) if n_out > 1 else f32(0)
            src = f32(scale * f32(o))
            i0 = min(int(np.floor(src)), n_in - 1)
            l1 = f32(min(max(f32(src - f32(i0)), f32(0)), f32(1)))
            l0 = f32(f32(1) - l1)
        elif precision == "exact":
            src = Fraction(o * (n_in - 1), n_out - 1) if n_out > 1 else Fraction(0)
            i0 = min(int(np.floor(src)), n_in - 1)
            l1 = min(max(src - i0, Fraction(0)), Fraction(1))
            l0 = 1 - l1
        else:
            raise ValueError(precision)
        i1 = i0 + (1 if i0 < n_in - 1 else 0)
        rows.append((float(src), i0, i1, float(l0), float(l1)))
    return rows


def formula_matrix(rows, n_in: int, float32: bool = True) -> np.ndarray:
    """[n_out, n_in] matrix of (src, i0, i1, l0, l1) rows: M[o, i0] = l0, M[o, i1] = l1; two
    coincident terms (i0 == i1, the last input) are added in float32 when float32 (what
    PyTorch's float32 kernel x * l0 + x * l1 gives a one-hot input)."""
    m = np.zeros((len(rows), n_in))
    for o, (_, i0, i1, l0, l1) in enumerate(rows):
        if i0 == i1:
            m[o, i0] = float(np.float32(l0) + np.float32(l1)) if float32 else l0 + l1
        else:
            m[o, i0], m[o, i1] = l0, l1
    return m


def upsample_align_corners(x: np.ndarray, factor: int = 4, dtype=torch.float32) -> np.ndarray:
    """The segmentation head's resize: nn.UpsamplingBilinear2d(scale_factor=factor), i.e.
    F.interpolate(x, scale_factor=factor, mode="bilinear", align_corners=True)."""
    with torch.no_grad():
        return nhwc(torch.nn.UpsamplingBilinear2d(scale_factor=factor)(nchw(x, dtype)))


# ---------------------------------------------------------------------------
# Feature combination
# ---------------------------------------------------------------------------
def add(a: np.ndarray, b: np.ndarray, dtype=torch.float32) -> np.ndarray:
    with torch.no_grad():
        return nhwc(nchw(a, dtype) + nchw(b, dtype))


def concat(parts, dtype=torch.float32) -> np.ndarray:
    with torch.no_grad():
        return nhwc(torch.cat([nchw(p, dtype) for p in parts], dim=1))


# ---------------------------------------------------------------------------
# Phase 22: a representative primitive chain from the installed SMP modules
# ---------------------------------------------------------------------------
