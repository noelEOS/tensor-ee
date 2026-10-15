"""Unregistered full-model planning inputs; construction only, never a forward.

Weights are the exact seeded SMP constructor state, with its initialization
unchanged. Dyadic boundary arrays are representative request-size inputs, not
evaluated PyTorch or Earth Engine model activations. Nothing registers a live
fixture, initializes Earth Engine, downloads, or submits a task.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json

import numpy as np

from tensor_ee.models.segformer_b0.configuration import SEGFORMER_B0, STAGE_DIMS, check_parameters
from .core import dyadic

SEED = 20261014


def seeded_state(seed=SEED):
    """SMP 0.5.0 Segformer mit_b0, two inputs/classes, eval; no model evaluation."""
    import torch
    import segmentation_models_pytorch as smp

    if importlib.metadata.version("segmentation-models-pytorch") != "0.5.0":
        raise ValueError("this plan requires the inspected SMP 0.5.0 reference")
    torch.set_num_threads(2)
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        with torch.random.fork_rng(), torch.device("cpu"):
            torch.manual_seed(seed)
            model = smp.Segformer("mit_b0", encoder_weights=None, in_channels=2, classes=2).eval()
        state = {name: tensor.detach().cpu().numpy().copy() for name, tensor in model.state_dict().items()}
    finally:
        torch.set_default_dtype(previous_dtype)
    check_parameters(state)
    return state


def representative_boundaries(seed=SEED):
    """Distinct deterministic finite float32 HWC literals, not model outputs."""
    shapes = {"input": (224, 224, 2)}
    shapes.update({f"C{i}": (grid.height, grid.width, dim)
                   for i, (grid, dim) in enumerate(zip(SEGFORMER_B0.stage_grids, STAGE_DIMS), 1)})
    shapes["S1B0"] = (56, 56, 32)
    return {name: dyadic(seed, f"request_boundary/{name}", shape) for name, shape in shapes.items()}


def reject_negative_zero(arrays):
    """Reject uncertain decimal transport of negative zero; never alter inputs."""
    for name, value in arrays.items():
        value = np.asarray(value)
        if np.issubdtype(value.dtype, np.floating) and np.any((value == 0) & np.signbit(value)):
            raise ValueError(f"{name}: negative zero is unsupported by this transport plan")


def array_provenance(arrays):
    """Metadata and IEEE-byte fingerprints only; no arrays in the report."""
    return {name: {"shape": list(value.shape), "dtype": str(value.dtype), "values": int(value.size),
                   "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
                   "range": [float(value.min()), float(value.max())]}
            for name, value in arrays.items()}


def state_sha256(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(json.dumps([name, list(value.shape), str(value.dtype)], separators=(",", ":")).encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()
