"""Composable float32 raster operations with the compiler's tensor semantics.

Start with ``from_image``; every operation returns a finite ``Raster``. These
functions construct deferred EE graphs and never initialize or evaluate EE.

Each function is a facade over the library block of the same operation in
``tensor_ee.blocks.raster`` (the objects ``compile_model`` calls), plus the
float32 materialization ``.toFloat()`` of its output. ``policy=`` is passed to
the block, which accepts only ``float32_operands``.
"""

from typing import Sequence

import ee
from torch import nn

from .blocks.raster import activations as _activations, conv as _conv, merge as _merge
from .blocks.raster import pooling as _pooling, resample as _resample
from .compiler import Grid, Raster
from .core.precision import FLOAT32_OPERANDS
from .layouts.raster import _float32, from_image  # moved to the library; re-exported here


def conv2d(raster: Raster, module: nn.Conv2d,
           parameters: tuple[ee.Array, ee.Array | None] | None = None, *,
           clip_footprints: bool = False, policy: str = FLOAT32_OPERANDS) -> Raster:
    """Apply a float32 Conv2d's coefficients, optionally supplied as EE arrays."""
    return _float32(_conv.conv2d(
        raster, module, parameters, clip_footprints=clip_footprints, policy=policy))


def pool2d(raster: Raster, module: nn.AvgPool2d | nn.MaxPool2d, *,
           clip_footprints: bool = False, policy: str = FLOAT32_OPERANDS) -> Raster:
    """Apply PyTorch average or max pooling at a supported configuration
    (``tensor_ee.blocks.raster.pooling.pool_config``)."""
    if not isinstance(module, (nn.AvgPool2d, nn.MaxPool2d)):
        raise TypeError("module must be AvgPool2d or MaxPool2d")
    return _float32(_pooling.pool2d(raster, module, clip_footprints=clip_footprints, policy=policy))


def nearest_upsample(raster: Raster, scale_factor: int, *,
                     clip_footprints: bool = False, policy: str = FLOAT32_OPERANDS) -> Raster:
    """Repeat pixels by an equal positive integer factor on both axes."""
    return _float32(_resample.nearest_upsample(
        raster, scale_factor, clip_footprints=clip_footprints, policy=policy))


def concatenate(rasters: Sequence[Raster], *, clip_footprints: bool = False,
                policy: str = FLOAT32_OPERANDS) -> Raster:
    """Join channels in order; every raster must have exactly the same grid."""
    return _float32(_merge.concatenate(rasters, clip_footprints=clip_footprints, policy=policy))


def residual_add(left: Raster, right: Raster, *, clip_footprints: bool = False,
                 policy: str = FLOAT32_OPERANDS) -> Raster:
    """Add equal-size tensors without broadcasting."""
    return _float32(_merge.residual_add(left, right, clip_footprints=clip_footprints, policy=policy))


def relu(raster: Raster, *, clip_footprints: bool = False,
         policy: str = FLOAT32_OPERANDS) -> Raster:
    """Replace negative values with zero."""
    return _float32(_activations.relu(raster, clip_footprints=clip_footprints, policy=policy))


def sigmoid(raster: Raster, *, clip_footprints: bool = False,
            policy: str = FLOAT32_OPERANDS) -> Raster:
    """Apply a sigmoid using the compiler's overflow-resistant expression."""
    return _float32(_activations.sigmoid(raster, clip_footprints=clip_footprints, policy=policy))



def leaky_relu(raster: Raster, negative_slope: float = 0.01, *,
               clip_footprints: bool = False, policy: str = FLOAT32_OPERANDS) -> Raster:
    """Scale negative values by a static finite float32 slope, preserving masks."""
    return _float32(_activations.leaky_relu(
        raster, negative_slope, clip_footprints=clip_footprints, policy=policy))



def silu(raster: Raster, *, clip_footprints: bool = False,
         policy: str = FLOAT32_OPERANDS) -> Raster:
    """Apply x*sigmoid(x) with an overflow-resistant sigmoid, preserving masks."""
    return _float32(_activations.silu(raster, clip_footprints=clip_footprints, policy=policy))


def gelu(raster: Raster, *, approximate: str = "none",
         clip_footprints: bool = False, policy: str = FLOAT32_OPERANDS) -> Raster:
    """Apply exact erf GELU; approximate='tanh' is unsupported."""
    return _float32(_activations.gelu(
        raster, approximate=approximate, clip_footprints=clip_footprints, policy=policy))


def tanh(raster: Raster, *, clip_footprints: bool = False,
         policy: str = FLOAT32_OPERANDS) -> Raster:
    """Apply hyperbolic tangent, retaining channel order and computational masks."""
    return _float32(_activations.tanh(raster, clip_footprints=clip_footprints, policy=policy))


def eval_dropout(raster: Raster, *, p: float = 0.5,
                 clip_footprints: bool = False, policy: str = FLOAT32_OPERANDS) -> Raster:
    """Evaluation dropout identity with finite real 0<=p<=1; no random mask."""
    return _float32(_activations.eval_dropout(raster, p=p, clip_footprints=clip_footprints, policy=policy))



def softmax(raster: Raster, *, clip_footprints: bool = False,
            policy: str = FLOAT32_OPERANDS) -> Raster:
    """Stable channel probabilities with a common finite mask across all scores."""
    return _float32(_activations.softmax(raster, clip_footprints=clip_footprints, policy=policy))
