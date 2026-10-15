"""A float64 mirror of the Earth Engine SW-MSA block and W-MSA + SW-MSA pair,
with float32 roundings exactly where the EE implementation rounds.

Milestone 2's Mirror (swin_block/reference/ee_mirror.py) already mirrors
LayerNorm, the linear layers, GELU, the residuals and Milestone 1's softmax;
its attention takes the shifted-window mask as a separate float32 addition
after the relative-position bias.  The layout steps (cyclic shift, window
re-partition, reverse) move values without arithmetic, so they are exact
NumPy permutations here, and the mask is built with the same formula as the
EE construction (earth_engine/shifted_mask.py), independently of timm.

    EE - torch32 = (EE - mirror) + (mirror - torch32)

The first term should be exactly 0; the second is PyTorch's float32 kernel
rounding (LayerNorm, GEMM, softmax and GELU round internally).

The window and shift default to Swin-T's 7 and 3; ``window`` and ``shift`` set others.
"""

from __future__ import annotations

import numpy as np
import torch

from reference.swin_block.ee_mirror import Mirror, _tensor
from reference.swin_pair.pytorch_shifted_swin import (
    MASK_VALUE, SHIFT, WINDOW, shift_windows, unshift_windows)


def mask_numpy(rows: int, cols: int, window: int = WINDOW, shift: int = SHIFT) -> np.ndarray:
    """The EE construction in NumPy: [nW, N, N] float32 (window order I * cols + J)."""
    t = np.arange(window * window)
    a, b = t // window, t % window
    rows_differ = (a[:, None] >= window - shift) != (a[None, :] >= window - shift)
    cols_differ = (b[:, None] >= window - shift) != (b[None, :] >= window - shift)
    i, j = np.divmod(np.arange(rows * cols), cols)
    last_row = (i == rows - 1)[:, None, None]
    last_col = (j == cols - 1)[:, None, None]
    different = (last_row & rows_differ) | (last_col & cols_differ)
    return np.where(different, np.float32(MASK_VALUE), np.float32(0)).astype(np.float32)


class ShiftedMirror(Mirror):
    """Block B (SW-MSA) on a rows x cols-window raster, in window layout (window 7 and
    shift 3 by default)."""

    def __init__(self, parameters, rows: int, cols: int, exact: bool = False, *, window: int = WINDOW,
                 shift: int = SHIFT):
        super().__init__(parameters, exact, window=window)
        self.rows, self.cols = rows, cols
        self.shift_size = shift
        self.mask = mask_numpy(rows, cols, window, shift)

    def shift(self, windows: torch.Tensor) -> torch.Tensor:
        """Exact permutation of a float64 [nW, w*w, C] tensor (no rounding)."""
        return torch.from_numpy(shift_windows(windows.numpy(), self.rows, self.cols, self.shift_size,
                                              self.window))

    def unshift(self, windows: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(unshift_windows(windows.numpy(), self.rows, self.cols, self.shift_size,
                                                self.window))

    def shifted_forward(self, x) -> dict[str, np.ndarray]:
        """Same tensor names, shapes and layouts as pytorch_shifted_swin.shifted_reference().

        Intermediates stay float64 tensors so that exact=True never rounds.
        """
        x = _tensor(x)
        n_windows = x.shape[0]
        t = {"x": x, "ln1": self.layer_norm(x, 1)}
        t["shifted"] = self.shift(t["ln1"])
        t["qkv"] = self.linear(t["shifted"], "attn.qkv")
        heads = t["qkv"].reshape(n_windows, self.tokens, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        t["q"], t["k"], t["v"] = heads.unbind(0)
        t.update(self.attention(t["q"], t["k"], t["v"], self.mask))
        t["mask"] = _tensor(self.mask)
        t["concat"] = t["attn_v"].transpose(1, 2).reshape(n_windows, self.tokens, self.dim)
        t["proj"] = self.linear(t["concat"], "attn.proj")
        t["reverse"] = t["proj"]                 # window_reverse: identity in window layout
        t["unshifted"] = self.unshift(t["proj"])
        t["x1"] = self.add(x, t["unshifted"])
        t["ln2"] = self.layer_norm(t["x1"], 2)
        t["fc1"] = self.linear(t["ln2"], "mlp.fc1")
        t["act"] = self.gelu(t["fc1"])
        t["fc2"] = self.linear(t["act"], "mlp.fc2")
        t["out"] = self.add(t["x1"], t["fc2"])
        return {name: value.numpy() for name, value in t.items()}


def shifted_mirror_forward(parameters, x, rows: int, cols: int, exact: bool = False, *, window: int = WINDOW,
                           shift: int = SHIFT):
    return ShiftedMirror(parameters, rows, cols, exact, window=window, shift=shift).shifted_forward(x)


def pair_mirror_forward(parameters_a, parameters_b, x, rows: int, cols: int,
                        exact: bool = False, *, window: int = WINDOW, shift: int = SHIFT) -> dict[str, np.ndarray]:
    """Keys "a_<name>" (Milestone 2's mirror) and "b_<name>", as pair_reference()."""
    a = Mirror(parameters_a, exact, window=window).forward(np.asarray(x, dtype=np.float32))
    # Block A's output is float32-valued unless exact; keep it unrounded either way.
    b = shifted_mirror_forward(parameters_b, torch.from_numpy(a["out"]), rows, cols, exact, window=window,
                               shift=shift)
    result = {f"a_{name}": value for name, value in a.items()}
    result.update({f"b_{name}": value for name, value in b.items()})
    return result
