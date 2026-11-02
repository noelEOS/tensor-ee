"""ViT-B/16 encoder target, built by hand from existing library blocks.

Six input bands on exactly 224x224 pixels, patch 16, 196 spatial rows in W14
and the timm row-0 class token in W1 on the coarse grid. Position addition,
pre-norm attention, residuals and exact GELU follow timm 1.0.30's unfused ViT;
all norms use eps 1e-6 and attention scales Q before the score product.
The Prithvi encoder wiring/backend are shared, with explicit settings and names.
No decoder, trained weights, converter, cloud calls or model-accuracy claim.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import numbers

import numpy as np

from ...core.tensor import Raster, ServerArray
from ..prithvi_sen1floods11.model import (
    BANDS, INPUT_GRID, TOKEN_GRID, EncoderBackend, reshape_parameter, compose as _compose,
)
from .configuration import PREFIX, VIT_B16, ViTConfig, parameter_shapes, stage_names


def validate_parameters(parameters, config=VIT_B16):
    """Require finite float32 native arrays or native/flattened ServerArray shapes.

    The unused timm classifier head.weight/head.bias may be present and is ignored.
    Decoder parameters must be filtered out by the caller. No server values are read.
    """
    if not isinstance(parameters, Mapping):
        raise TypeError("parameters must map state names to NumPy arrays or ServerArrays")
    shapes = parameter_shapes(config)
    missing = set(shapes) - parameters.keys()
    extra = parameters.keys() - shapes.keys() - {PREFIX + ".head.weight", PREFIX + ".head.bias"}
    if missing or extra:
        raise ValueError(f"encoder parameter names differ: missing={sorted(missing)}, extra={sorted(extra)}")
    for name, shape in shapes.items():
        value = parameters[name]
        if not isinstance(value, (np.ndarray, ServerArray)):
            raise ValueError(f"{name} must be a NumPy array or ServerArray")
        received = tuple(value.shape)
        allowed = {shape}
        if isinstance(value, ServerArray):
            if any(isinstance(n, bool) or not isinstance(n, numbers.Integral) or n < 1 for n in received):
                raise ValueError(f"{name} has invalid server dimensions")
            allowed.add((shape[0], math.prod(shape[1:])))
            if len(shape) == 1:
                allowed.add((1, shape[0]))
        if received not in allowed:
            raise ValueError(f"{name} must have native shape {shape} or its flattened asset matrix")
        if isinstance(value, np.ndarray) and (value.dtype != np.float32 or not np.isfinite(value).all()):
            raise ValueError(f"{name} must contain finite float32 values")


class LibraryBackend(EncoderBackend):
    """The shared encoder backend with ViT's names and query-first scaling."""

    def __init__(self, ee, parameters, config=VIT_B16):
        validate_parameters(parameters, config)
        state = {name: reshape_parameter(parameters[name], shape) for name, shape in parameter_shapes(config).items()}
        super().__init__(ee, state, config, prefix=PREFIX, scale_query=True)

    def patch(self, raster):
        if not isinstance(raster, Raster) or (raster.grid.height, raster.grid.width) != (224, 224):
            raise ValueError("input must be a six-band Raster on exactly 224 by 224 pixels")
        return super().patch(raster)


def compose(backend, inputs, config=VIT_B16, *, start="input", stop="norm"):
    """Return every traversed boundary, including each raw block pair and final norm.

    ``start`` may be a saved patch/block pair. The class row is timm sequence row 0;
    spatial rows follow in row-major order. SMP's four default DPT outputs are raw
    block2, block5, block8, block11 (forward_intermediates defaults to norm=False).
    """
    names = stage_names(config)
    if start not in names or stop not in names or names.index(start) > names.index(stop):
        raise ValueError("start/stop must name ordered encoder boundaries")
    return _compose(backend, inputs, config, start=start, stop=stop,
                    encoder_prefix=PREFIX, norm_eps=1e-6)


@dataclass(frozen=True)
class ViTConstants:
    config: ViTConfig
    backend: LibraryBackend


def constants(ee, parameters, *, config=VIT_B16):
    """Build the encoder's coefficients ahead of spatial composition."""
    return ViTConstants(config, LibraryBackend(ee, parameters, config))


def forward(ee, raster, coefficients: ViTConstants, *, stop="norm"):
    """Build all encoder outputs through stop, keeping spatial/class pairs separate."""
    if coefficients.backend.ee is not ee:
        raise ValueError("constants and forward must use the same Earth Engine module")
    return compose(coefficients.backend, raster, coefficients.config, stop=stop)
