"""Small NumPy and PyTorch helpers the ported decoder tests use, copied unchanged (apart from
their imports) from research modules archived at the tag ``research-archive-2026-10-12``:

- ``pixel_ids_numpy``: ``upernet/earth_engine/feature_interface.py``;
- ``fold``: ``upernet/earth_engine/normalization.py`` (BatchNorm folded into a conv weight;
  the decoder runs explicit BatchNorm instead, and this shows why);
- ``grouped_probe_inputs``: ``upernet/earth_engine/validate_fusion.py`` (the Phase 16 fixture
  that separates the fusion formulations);
- ``torch_resize_matrix``: ``upernet/reference/m12_architecture_contract.py`` (PyTorch's own
  bilinear resize matrix, read off one-hot inputs).
"""

from __future__ import annotations

import numpy as np
import torch

from reference.upernet import smp_fusion_reference as R

BN_EPS = 1e-5                     # torch.nn.BatchNorm2d's default, as in SMP's decoder


def pixel_ids_numpy(height: int, width: int, channels: int, offset: int = 0) -> np.ndarray:
    y, x, c = np.meshgrid(np.arange(height), np.arange(width), np.arange(channels), indexing="ij")
    return ((y * width + x) * channels + c + offset).astype(np.float32)


def fold(weight: np.ndarray, gamma, beta, mean, var, eps: float = BN_EPS) -> tuple:
    """(W', b') float32: BatchNorm folded into a bias-free conv weight [Cout, ...]."""
    alpha = np.asarray(gamma, np.float64) / np.sqrt(np.asarray(var, np.float64) + eps)
    w = np.asarray(weight, np.float64) * alpha.reshape((-1,) + (1,) * (np.ndim(weight) - 1))
    b = np.asarray(beta, np.float64) - np.asarray(mean, np.float64) * alpha
    return w.astype(np.float32), b.astype(np.float32)


def grouped_probe_inputs():
    """A one-window fixture (7 x 7 tokens, four groups of two channels) whose fusion conv
    reveals each formulation's rounding.  Output channel 0 at the centre token is
    2^40 (group 1) + 1 (group 1) - 2^40 (group 2); channel 1 is 2^53 + 1 (group 3) + 1 -
    2^53 (group 4).  Monolithic (one sequential double dot): 1, 0; A (groups rounded to
    float32): 0, 0; B (group dots in double, summed in double): 1, 1."""
    x = np.zeros((7, 7, 8), np.float32)
    x[3, 3, [0, 1, 2]] = [2.0 ** 20, 1.0, -(2.0 ** 20)]
    x[3, 3, [4, 5, 6, 7]] = [2.0 ** 27, 1.0, 1.0, -(2.0 ** 27)]
    w = np.zeros((2, 8, 3, 3), np.float32)
    w[0, [0, 1, 2], 1, 1] = [2.0 ** 20, 1.0, 2.0 ** 20]
    w[1, [4, 5, 6, 7], 1, 1] = [2.0 ** 26, 1.0, 1.0, 2.0 ** 26]
    params = {R.WEIGHT: w, f"{R.PREFIX}.1.weight": np.ones(2, np.float32),
              f"{R.PREFIX}.1.bias": np.zeros(2, np.float32),
              f"{R.PREFIX}.1.running_mean": np.zeros(2, np.float32),
              f"{R.PREFIX}.1.running_var": np.ones(2, np.float32)}
    return x, w, params


def torch_resize_matrix(n_in: int, n_out: int, align_corners: bool) -> np.ndarray:
    import torch.nn.functional as F
    eye = torch.eye(n_in, dtype=torch.float32).reshape(n_in, 1, n_in, 1)
    kw = {"scale_factor": (n_out // n_in, 1)} if align_corners else {"size": (n_out, 1)}
    y = F.interpolate(eye, mode="bilinear", align_corners=align_corners, **kw)
    return y[:, 0, :, 0].T.double().numpy()
