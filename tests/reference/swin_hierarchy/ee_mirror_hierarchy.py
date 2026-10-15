"""Phase 13: a float64 mirror of the Earth Engine hierarchy, with float32 roundings
exactly where the EE implementation rounds.

    stage-1 pair    Milestone 3's pair mirror (dim 96, 3 heads)
    Patch Merging   gather: an exact NumPy permutation (no rounding);
                    LayerNorm(384): Milestone 2's formula over the last axis,
                    double inside, one float32 rounding;
                    reduction: one double product, one float32 rounding, no bias
    stage-2 pair    the same Milestone 2/3 mirrors, whose width and head count
                    now come from the parameters (dim 192, 6 heads)

    EE - torch32 = (EE - mirror) + (mirror - torch32)

The first term should be exactly 0; the second is PyTorch's float32 kernel
rounding.  With exact=True nothing is rounded: the float64 reference of the
same formulas, used for per-component checks on given float32 inputs.

The window defaults to Swin-T's 7 (``window`` sets another; the shift is
window // 2, timm's stage rule).  ``PatchMergeMirror.sequential_sums`` sums the
LayerNorm(4C) left to right, Earth Engine's order (off by default, as in
Milestone 4).
"""

from __future__ import annotations

import numpy as np
import torch

from reference.swin_block.ee_mirror import Mirror, _tensor, layer_norm_formula, sequential_sum
from reference.swin_block.pytorch_swin_block import WINDOW
from reference.swin_hierarchy.pytorch_hierarchy import gather_windows
from reference.swin_pair.ee_mirror_shifted import pair_mirror_forward, shifted_mirror_forward


class PatchMergeMirror:
    sequential_sums = False

    def __init__(self, parameters: dict, exact: bool = False):
        self.p = {k: _tensor(v) for k, v in parameters.items()}
        self.exact = exact

    def r32(self, t: torch.Tensor) -> torch.Tensor:
        return t if self.exact else t.float().double()

    def norm(self, cat) -> torch.Tensor:
        total = sequential_sum if self.sequential_sums else None
        return layer_norm_formula(_tensor(cat), self.p["norm.weight"], self.p["norm.bias"], self.r32,
                                  total=total)

    def reduction(self, norm) -> torch.Tensor:
        product = _tensor(norm) @ self.p["reduction.weight"].T
        if "reduction.bias" in self.p:
            product = product + self.p["reduction.bias"]
        return self.r32(product)

    def forward(self, windows: np.ndarray, rows: int, cols: int, *, window: int = WINDOW) -> dict[str, np.ndarray]:
        """Stage-1 windows [rows*cols, w*w, C] (float64 container) -> pm_* tensors."""
        windows = np.asarray(windows, dtype=np.float64)
        channels = windows.shape[-1]
        cat = gather_windows(windows, rows, cols, window)
        t = {f"x{g}": cat[..., g * channels:(g + 1) * channels] for g in range(4)}
        t["cat"] = cat
        norm = self.norm(torch.from_numpy(cat))
        t["norm"] = norm.numpy()
        t["out"] = self.reduction(norm).numpy()
        return t


def hierarchy_mirror(parameters: dict, x_windows: np.ndarray, rows: int, cols: int,
                     exact: bool = False, names=None, *, window: int = WINDOW) -> dict[str, np.ndarray]:
    """Same names and layouts as pytorch_hierarchy.hierarchy_reference().

    x_windows: stage-1 input [rows*cols, w*w, 96]; rows, cols: stage-1 windows.
    """
    shift = window // 2
    s1 = pair_mirror_forward(parameters["s1a"], parameters["s1b"], x_windows, rows, cols, exact,
                             window=window, shift=shift)
    pm = PatchMergeMirror(parameters["pm"], exact).forward(s1["b_out"], rows, cols, window=window)
    # Unrounded containers either way: exact=True must never round here.
    a2 = Mirror(parameters["s2a"], exact, window=window).forward(torch.from_numpy(pm["out"]))
    b2 = shifted_mirror_forward(parameters["s2b"], torch.from_numpy(a2["out"]), rows // 2,
                                cols // 2, exact, window=window, shift=shift)
    result = {"x": np.asarray(x_windows, dtype=np.float64)}
    result.update({f"s1{k}": v for k, v in s1.items()})          # a_ -> s1a_, b_ -> s1b_
    result.update({f"pm_{k}": v for k, v in pm.items()})
    result.update({f"s2a_{k}": v for k, v in a2.items()})
    result.update({f"s2b_{k}": v for k, v in b2.items()})
    if names is not None:
        result = {name: result[name] for name in names}
    return result
