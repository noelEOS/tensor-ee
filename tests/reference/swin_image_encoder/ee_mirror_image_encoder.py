"""Phase 10: the float64 mirror of the EE image encoder, rounding where EE rounds.

    input       SegWater v3 preprocessing: float32 after every operation (EE casts
                after each one), so the mirror is the NumPy reference itself
    patches     exact gather of the 4 x 4 x 2 patch, (c, ky, kx) order
    pe_proj     [49, 32] x W^T [32, 96]: EE's matrixMultiply accumulates the 32
                products sequentially in double; + bias in double; rounded once
    x           LayerNorm(96) in double with EE's left-to-right sums, rounded once
    encoder     Milestone 5's mirror (swin_encoder/reference/ee_mirror_encoder.py)

    EE - torch32 = (EE - mirror) + (mirror - torch32)

With exact=True nothing is rounded after the input: the float64 reference of the
same formulas.

The patch-embedding mirror takes the patch, the band count and the width (task P03):
``patch_windows(z, patch=, window=)`` gathers any patch into any window layout (today's
4 and 7 by default), and ``pe_mirror`` reads the band count and the width from the
weight shapes ([dim, C p^2]). ``pe_mirror(..., sequential_sums=True)`` sums each
projection dot product strictly left to right in double, the order of EE's
matrixMultiply (Milestone 5), instead of torch's float64 matmul (the default, unchanged).
"""

from __future__ import annotations

import numpy as np
import torch

from reference.swin_block.ee_mirror import _tensor, layer_norm_formula, sequential_sum
from reference.swin_block.pytorch_swin_block import WINDOW, to_window_layout
from reference.swin_encoder.ee_mirror_encoder import encoder_mirror
from reference.swin_image_encoder.pytorch_image_encoder import PATCH, patches, to_nchw


def _wanted(names, key: str) -> bool:
    return names is None or key in names


def sequential_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a @ b (float64) with every dot product summed strictly left to right, each product
    formed first: the order of EE's matrixMultiply (Milestone 5). Products of float32
    values are exact in double, so a fused multiply-add would give the same sums."""
    out = a[..., :, 0:1] * b[..., 0:1, :]
    for k in range(1, a.shape[-1]):
        out = out + a[..., :, k:k + 1] * b[..., k:k + 1, :]
    return out


def pe_mirror(pe: dict, patch_windows, exact: bool = False, *,
              sequential_sums: bool = False) -> dict[str, np.ndarray]:
    """Patch embedding of patch_windows [nW, N, C p^2] (float32 values; today's
    [nW, 49, 32]): pe_proj, x [nW, N, dim]. The band count and the width come from
    proj.weight [dim, C p^2]. ``sequential_sums``: the projection's dot products summed
    left to right (``sequential_matmul``) rather than by torch's matmul."""
    p = {k: _tensor(v) for k, v in pe.items()}
    r32 = (lambda t: t) if exact else (lambda t: t.float().double())
    h = _tensor(np.asarray(patch_windows, dtype=np.float32))
    weight_t = p["proj.weight"].T
    product = sequential_matmul(h, weight_t) if sequential_sums else h @ weight_t
    proj = r32(product + p["proj.bias"])
    x = layer_norm_formula(proj, p["norm.weight"], p["norm.bias"], r32, total=sequential_sum)
    return {"pe_proj": proj.numpy(), "x": x.numpy()}


def patch_windows(z: np.ndarray, *, patch: int = PATCH, window: int = WINDOW) -> np.ndarray:
    """[H, W, C] -> [nW1, w^2, C p^2] in window layout (moved values; today's
    [nW1, 49, 32])."""
    return to_window_layout(patches(to_nchw(np.asarray(z, dtype=np.float32)), patch)[0].numpy(), window)


def image_encoder_mirror(parameters: dict, pe: dict, z: np.ndarray, exact: bool = False,
                         names=None, stop: str | None = None) -> dict[str, np.ndarray]:
    """Same names and layouts as pytorch_image_encoder.image_encoder_reference."""
    height, width, _ = z.shape
    windows = patch_windows(z)
    result = {"pe_patches": windows.astype(np.float64)} if _wanted(names, "pe_patches") else {}
    if _wanted(names, "input"):
        result["input"] = np.asarray(z, dtype=np.float64)
    pe_t = pe_mirror(pe, windows, exact)
    result.update({k: v for k, v in pe_t.items() if _wanted(names, k)})
    if stop == "x":
        return result
    rows, cols = height // 28, width // 28
    wanted = None if names is None else sorted(set(names) - {"pe_patches", "pe_proj", "input"})
    encoder = encoder_mirror(parameters, pe_t["x"], rows, cols, exact, names=wanted)
    result.update({k: v for k, v in encoder.items() if k != "x" or _wanted(names, "x")})
    return result


def matmul_is_sequential(rows: int = 2000, length: int = 32, seed: int = 0) -> bool:
    """torch's float64 matmul == strictly sequential accumulation (EE's order) for
    dot products of `length` float32 values (checked offline)."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((rows, length)).astype(np.float32).astype(np.float64)
    b = rng.standard_normal((length, 7)).astype(np.float32).astype(np.float64)
    fast = (torch.from_numpy(a) @ torch.from_numpy(b)).numpy()
    seq = np.zeros((rows, 7))
    for k in range(length):
        seq = seq + a[:, k:k + 1] * b[k:k + 1, :]
    return bool(np.array_equal(fast, seq))
