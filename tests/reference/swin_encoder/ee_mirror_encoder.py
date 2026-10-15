"""Phase 13: a float64 mirror of the Earth Engine encoder, with float32 roundings
exactly where the EE implementation rounds.

    blocks          Milestone 2's Mirror (W-MSA) and Milestone 3's ShiftedMirror
                    (SW-MSA): width and head count from the parameters, so the
                    same code mirrors dim 96 / 3 heads ... dim 768 / 24 heads
    Patch Merging   Milestone 4's PatchMergeMirror: exact gather, LayerNorm(4C)
                    and the bias-free reduction, each rounded once
    final norm      Milestone 2's LayerNorm formula over 768 channels, rounded once

    EE - torch32 = (EE - mirror) + (mirror - torch32)

The first term should be exactly 0; the second is PyTorch's float32 kernel
rounding.  With exact=True nothing is rounded: the float64 reference of the
same formulas.  Intermediates are kept as float64 tensors between blocks, so
exact=True never rounds.

Accumulation order.  EE's arrayReduce(sum) adds strictly left to right and its
matrixMultiply accumulates each dot product sequentially in double (both
established live).  torch's float64 matmul is already sequential, but its sum
is vectorized, which changes the last double bits of most row sums and, at
rare near-ties, a float32 result (found at one of 2.4 M LayerNorm(768)
elements).  Every sum here (LayerNorm mean and variance, softmax denominator)
is therefore sequential (sequential_sums); Milestones 2-4's mirrors keep
torch's sums.
"""

from __future__ import annotations

import numpy as np
import torch

from reference.swin_block.ee_mirror import Mirror, _tensor, layer_norm_formula, sequential_sum
from reference.swin_encoder.pytorch_swin_encoder import DEPTHS, STAGES, stage_shifted, stage_windows
from reference.swin_hierarchy.ee_mirror_hierarchy import PatchMergeMirror
from reference.swin_pair.ee_mirror_shifted import ShiftedMirror


class SequentialMirror(Mirror):
    """W-MSA block mirror with EE's left-to-right sums."""
    sequential_sums = True


class SequentialShiftedMirror(ShiftedMirror):
    """SW-MSA block mirror with EE's left-to-right sums."""
    sequential_sums = True


class SequentialPatchMergeMirror(PatchMergeMirror):
    """Patch Merging mirror whose LayerNorm(4C) sums left to right."""

    def norm(self, cat) -> torch.Tensor:
        return layer_norm_formula(_tensor(cat), self.p["norm.weight"], self.p["norm.bias"],
                                  self.r32, total=sequential_sum)


def _wanted(names, key: str) -> bool:
    return names is None or key in names


def stage_mirror(parameters: dict, s: int, h, rows: int, cols: int, exact: bool = False,
                 blocks: int | None = None, prefix: str = "", names=None) -> tuple:
    """The first `blocks` blocks of stage s on windows h [rows*cols, 49, dim]
    (float64 container).  Returns (output tensor, {prefix b{i}_<name>: array})."""
    shifted = stage_shifted(rows, cols)
    h = h if isinstance(h, torch.Tensor) else torch.from_numpy(np.asarray(h, dtype=np.float64))
    tensors = {}
    for b in range(DEPTHS[s - 1] if blocks is None else blocks):
        p = parameters[f"s{s}b{b}"]
        if shifted and b % 2:
            result = SequentialShiftedMirror(p, rows, cols, exact).shifted_forward(h)
        else:
            result = SequentialMirror(p, exact).forward(h)
        tensors.update({f"{prefix}b{b}_{k}": v for k, v in result.items()
                        if _wanted(names, f"{prefix}b{b}_{k}")})
        h = torch.from_numpy(result["out"])
    return h, tensors


def norm_mirror(parameters: dict, x, exact: bool = False) -> np.ndarray:
    p = {k: _tensor(v) for k, v in parameters["norm"].items()}
    r32 = (lambda t: t) if exact else (lambda t: t.float().double())
    x = x if isinstance(x, torch.Tensor) else torch.from_numpy(np.asarray(x, dtype=np.float64))
    return layer_norm_formula(x, p["weight"], p["bias"], r32, total=sequential_sum).numpy()


def merge_mirror(parameters: dict, n: int, h, rows: int, cols: int, exact: bool = False) -> dict:
    """Patch Merging n of stage-n windows h [rows*cols, 49, C] (rows x cols stage-n windows)."""
    return SequentialPatchMergeMirror(parameters[f"pm{n}"], exact).forward(np.asarray(h), rows, cols)


def encoder_mirror(parameters: dict, x_windows: np.ndarray, rows: int, cols: int,
                   exact: bool = False, names=None) -> dict[str, np.ndarray]:
    """Same names and layouts as pytorch_swin_encoder.encoder_reference().

    x_windows: stage-1 input [rows*cols, 49, 96]; rows, cols: stage-1 windows.
    """
    windows = stage_windows(rows, cols)
    result = {"x": np.asarray(x_windows, dtype=np.float64)} if _wanted(names, "x") else {}
    h = torch.from_numpy(np.asarray(x_windows, dtype=np.float32).astype(np.float64))
    for s in range(1, STAGES + 1):
        if s > 1:
            pm = merge_mirror(parameters, s - 1, h.numpy(), *windows[s - 2], exact)
            result.update({f"pm{s - 1}_{k}": v for k, v in pm.items()
                           if _wanted(names, f"pm{s - 1}_{k}")})
            h = torch.from_numpy(pm["out"])
        h, tensors = stage_mirror(parameters, s, h, *windows[s - 1], exact, prefix=f"s{s}",
                                  names=names)
        result.update(tensors)
        if _wanted(names, f"s{s}_out"):
            result[f"s{s}_out"] = h.numpy()
    if _wanted(names, "norm"):
        result["norm"] = norm_mirror(parameters, h, exact)
    return result
