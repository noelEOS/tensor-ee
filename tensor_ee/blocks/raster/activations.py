"""R pointwise activations, the common score mask and channel softmax.

Moved whole from ``tensor_ee.compiler``, where they were private:
``_relu``, ``_sigmoid``, ``_leaky_relu``, ``_silu``, ``_gelu``, ``_tanh``,
``_eval_dropout``, ``_common_score_mask`` and ``_softmax`` are here ``relu``,
``sigmoid``, ``leaky_relu``, ``silu``, ``gelu``, ``tanh``, ``eval_dropout``,
``common_score_mask`` and ``softmax``. Their helpers moved with them:
``_sigmoid_image``, ``_real_option``, ``_FLOAT32_MAX``, ``_exact_gelu`` and
``_dropout_probability``. The old names are re-exports of these objects.

Precision policy ``float32_operands`` (contract section 7): scalars are exact
binary64 (``_ee_scalar``); softmax keeps its ``toDouble()`` score widening; the
node output is materialized with ``.toFloat()`` by the caller
(``compile_model`` or the ``tensor_ee.primitives`` facade), not here.

``_finish`` and ``_ee_scalar`` are module-level bindings that these functions
look up (tests/test_pointwise_operators.py patches them here).
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

import ee
import torch

from ...core.errors import UnsupportedOperation
from ...core.grid import Grid
from ...core.metadata import ANY_SETTINGS, BlockMetadata, Locality, Validated, validate_metadata
from ...core.precision import FLOAT32_OPERANDS, validate_policy
from ...core.precision import exact_scalar as _ee_scalar
from ...core.tensor import Raster
from ...layouts.raster import finish as _finish
from .conv import _no_coefficients


def relu(raster: Raster, *, clip_footprints: bool = False, policy: str = FLOAT32_OPERANDS) -> Raster:
    validate_policy(policy, allowed=FLOAT32_OPERANDS)
    return _finish(raster.image.max(0), raster.bands, raster.grid, clip_footprints)


def _sigmoid_image(image: ee.Image) -> ee.Image:
    # Stable algebraic form avoids exp overflow for very negative logits.
    magnitude = image.abs().multiply(-1).exp()
    positive = magnitude.add(1).pow(-1)
    negative = magnitude.divide(magnitude.add(1))
    return positive.where(image.lt(0), negative)


def sigmoid(raster: Raster, *, clip_footprints: bool = False, policy: str = FLOAT32_OPERANDS) -> Raster:
    validate_policy(policy, allowed=FLOAT32_OPERANDS)
    return _finish(_sigmoid_image(raster.image), raster.bands, raster.grid, clip_footprints)


_FLOAT32_MAX = float(torch.finfo(torch.float32).max)


def _real_option(value: Any, name: str, *, float32: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise UnsupportedOperation(f"{name} must be a static finite real scalar; received {value!r}")
    try:
        number = float(value)
    except (OverflowError, ValueError) as error:
        raise UnsupportedOperation(f"{name} must be finite; received {value!r}") from error
    if not math.isfinite(number):
        raise UnsupportedOperation(f"{name} must be finite; received {value!r}")
    if float32:
        if abs(value) > _FLOAT32_MAX:
            raise UnsupportedOperation(f"{name} must be representable in float32; received {value!r}")
        rounded = torch.tensor(number, dtype=torch.float32).item()
        if value != 0 and rounded == 0:
            raise UnsupportedOperation(f"{name} must be representable in float32; received {value!r}")
        number = rounded
    return number


def _exact_gelu(approximate: Any) -> None:
    if not isinstance(approximate, str) or approximate != "none":
        raise UnsupportedOperation(f"approximate must be 'none'; received {approximate!r}")


def _dropout_probability(p: Any) -> float:
    probability = _real_option(p, "p")
    if not 0 <= p <= 1:
        raise UnsupportedOperation(f"p must satisfy 0 <= p <= 1; received {p!r}")
    return probability


def leaky_relu(raster: Raster, negative_slope: float = 0.01,
               *, clip_footprints: bool = False, policy: str = FLOAT32_OPERANDS) -> Raster:
    validate_policy(policy, allowed=FLOAT32_OPERANDS)
    slope = _real_option(negative_slope, "negative_slope", float32=True)
    image = raster.image.where(raster.image.lt(0), raster.image.multiply(_ee_scalar(slope)))
    return _finish(image, raster.bands, raster.grid, clip_footprints)


def silu(raster: Raster, *, clip_footprints: bool = False, policy: str = FLOAT32_OPERANDS) -> Raster:
    validate_policy(policy, allowed=FLOAT32_OPERANDS)
    # Both sigmoid branches use exp(-abs(x)); no positive exponent can overflow.
    sigmoid = _sigmoid_image(raster.image)
    return _finish(raster.image.multiply(sigmoid), raster.bands, raster.grid, clip_footprints)


def gelu(raster: Raster, *, approximate: str = "none",
         clip_footprints: bool = False, policy: str = FLOAT32_OPERANDS) -> Raster:
    validate_policy(policy, allowed=FLOAT32_OPERANDS)
    _exact_gelu(approximate)
    # Established Swin expression: double work, one float32 materialization at
    # the public/compiler boundary. No intermediate x*x or positive exponent.
    image = raster.image.multiply(0.5).multiply(
        raster.image.multiply(math.sqrt(0.5)).erf().add(1))
    return _finish(image, raster.bands, raster.grid, clip_footprints)


def tanh(raster: Raster, *, clip_footprints: bool = False, policy: str = FLOAT32_OPERANDS) -> Raster:
    validate_policy(policy, allowed=FLOAT32_OPERANDS)
    return _finish(raster.image.tanh(), raster.bands, raster.grid, clip_footprints)


def eval_dropout(raster: Raster, *, p: float = 0.5,
                 clip_footprints: bool = False, policy: str = FLOAT32_OPERANDS) -> Raster:
    validate_policy(policy, allowed=FLOAT32_OPERANDS)
    _dropout_probability(p)
    return _finish(raster.image, raster.bands, raster.grid, clip_footprints)


def common_score_mask(image: ee.Image, grid: Grid, *, policy: str = FLOAT32_OPERANDS) -> ee.Image:
    """Intersect every score mask and finite float32 predicate before reduction.

    This is the existing _valid_logits predicate in a recipe-free core path.
    Parsing FLOAT32_MAX avoids EE's integral-valued literal clipping.
    """
    validate_policy(policy, allowed=FLOAT32_OPERANDS)
    masks = image.mask().reduce(ee.Reducer.min()).gt(0)
    finite = image.eq(image).And(image.abs().lte(_ee_scalar(_FLOAT32_MAX))).reduce(ee.Reducer.min())
    return masks.And(finite).unmask(0, sameFootprint=False).And(grid.mask())


def softmax(raster: Raster, *, clip_footprints: bool = False, policy: str = FLOAT32_OPERANDS) -> Raster:
    validate_policy(policy, allowed=FLOAT32_OPERANDS)
    if (not raster.bands or any(not isinstance(name, str) or not name for name in raster.bands)
            or len(set(raster.bands)) != len(raster.bands)):
        raise UnsupportedOperation("softmax requires nonempty unique channel names")
    image = raster.image.select(list(raster.bands))
    image = image.updateMask(common_score_mask(image, raster.grid)).toDouble()
    maximum = image.reduce(ee.Reducer.max())
    exponential = image.subtract(maximum).exp()
    normalized = exponential.divide(exponential.reduce(ee.Reducer.sum()))
    return _finish(normalized, raster.bands, raster.grid, clip_footprints)


# ---------------------------------------------------------------------------
# Locality and cost metadata (contract section 9)
# ---------------------------------------------------------------------------
_SWEEPS = ("docs/evidence/t02-operators.json", "docs/evidence/t05-primary.json")

_UNCLIPPED = Validated(({"clip_footprints": (False,)},))
"""The blocks only the T02, T03 and T05 sweeps cover: their slopes, probabilities and inputs
varied, but ``clip_footprints`` never left False. The ResNet18 runs, which the other R blocks
cite, used True (resnet18_full64_float32_policy.json)."""


def _pointwise(name: str, evidence: tuple[str, ...], validated: Validated = ANY_SETTINGS) -> BlockMetadata:
    return BlockMetadata(name=name, layout_in="R", layout_out="R", locality=Locality.POINTWISE,
                         policy=FLOAT32_OPERANDS, embedded_constants=_no_coefficients,
                         asset_arrays=_no_coefficients, status="Ready", evidence=evidence, validated=validated)


METADATA = validate_metadata({
    "relu": _pointwise("relu", ("docs/history/cnn/results/parity.json", "docs/history/cnn/results/resnet18_full64_float32_policy.json",
                                "docs/history/cnn/results/segwater_js_parity_1x1.json")),
    "sigmoid": _pointwise("sigmoid", ("docs/history/cnn/results/parity.json", "docs/history/cnn/results/resnet18_full64_float32_policy.json")),
    "leaky_relu": _pointwise("leaky_relu", _SWEEPS, _UNCLIPPED),
    "silu": _pointwise("silu", _SWEEPS, _UNCLIPPED),
    "gelu": _pointwise("gelu", _SWEEPS, _UNCLIPPED),
    "tanh": _pointwise("tanh", _SWEEPS, _UNCLIPPED),
    "eval_dropout": _pointwise("eval_dropout", _SWEEPS, _UNCLIPPED),
    # A single-band 0/1 validity mask on the raster's grid.
    "common_score_mask": _pointwise("common_score_mask", ("docs/evidence/t03-outputs.json",)),
    "softmax": _pointwise("softmax", ("docs/evidence/t03-outputs.json", "docs/evidence/t05-primary.json"), _UNCLIPPED),
})
