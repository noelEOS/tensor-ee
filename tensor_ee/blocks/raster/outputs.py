"""Recipe-free segmentation outputs on explicit finite raster grids.

Scores/probabilities are float32. Classes are the ordered raw-logit argmax;
thresholds compare widened probabilities against a retained binary64 scalar.
These functions build deferred graphs without cloud inspection or initialization.

Moved whole from ``tensor_ee.outputs`` (``segmentation_outputs``,
``binary_threshold``, ``SegmentationOutputs`` and their helpers). The old names
are re-exports of these objects. The probabilities are this module's softmax
block materialized with ``.toFloat()``, exactly as the public
``tensor_ee.softmax`` primitive the old module called.

Precision policy ``float32_operands`` (contract section 7): the threshold
widens the float32 probabilities ``toDouble()`` and compares them with an exact
binary64 scalar (``_ee_scalar``). The generic defaults (positive class 1,
threshold 0.5) are API defaults, not the SegWater recipe (contract section 11).
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
import struct
from typing import Sequence

import ee

from ...core.errors import UnsupportedOperation
from ...core.metadata import ANY_SETTINGS, BlockMetadata, Locality, validate_metadata
from ...core.precision import FLOAT32_OPERANDS, validate_policy
from ...core.precision import exact_scalar as _ee_scalar
from ...core.tensor import Raster
from ...layouts.raster import _float32, finish as _finish
from .activations import _real_option, common_score_mask as _common_score_mask, softmax as _softmax
from .conv import _no_coefficients


@dataclass(frozen=True)
class SegmentationOutputs:
    logits: Raster
    probabilities: Raster
    classes: Raster


def _band_names(names: Sequence[str], count: int, option: str) -> tuple[str, ...]:
    if isinstance(names, str):
        raise UnsupportedOperation(f"{option} must be a sequence of {count} names")
    try:
        result = tuple(names)
    except TypeError as error:
        raise UnsupportedOperation(f"{option} must be a sequence of {count} names") from error
    if (len(result) != count or any(not isinstance(name, str) or not name for name in result)
            or len(set(result)) != count):
        raise UnsupportedOperation(f"{option} must contain exactly {count} nonempty unique names")
    return result


def _scores(raster: Raster, *, binary: bool = False) -> ee.Image:
    if not isinstance(raster, Raster):
        raise TypeError("scores must be a Raster")
    count = len(raster.bands)
    if count < 2 or (binary and count != 2):
        requirement = "exactly two" if binary else "at least two"
        raise UnsupportedOperation(f"scores require {requirement} channels")
    _band_names(raster.bands, count, "score bands")
    return raster.image.select(list(raster.bands))


def _check_literal_validity(image: ee.Image) -> None:
    """Inspect only constant literals and rename/cast/reproject wrappers.

    No generic graph evaluator is attempted. Computed literals, arbitrary graph
    operations and remote assets remain the caller's binary/finiteness assertion.
    A float cast is applied to the literal values before checking the effective
    binary payload; masks/other operations stop this narrow inspection.
    """
    wrappers = {"Image.rename": "input", "Image.toFloat": "value",
                "Image.toDouble": "value", "Image.reproject": "image"}
    float_cast = False
    while isinstance(image, ee.Image) and image.func is not None:
        signature = image.func.getSignature()
        name = signature.get("name")
        if name not in wrappers:
            break
        float_cast = float_cast or name == "Image.toFloat"
        image = image.args[wrappers[name]]
    if not isinstance(image, ee.Image) or image.func is None:
        return
    if image.func.getSignature().get("name") != "Image.constant":
        return
    value = image.args.get("value")
    if isinstance(value, Real):
        values = (value,)
    elif isinstance(value, (list, tuple)) and all(isinstance(v, Real) for v in value):
        values = tuple(value)
    else:
        return
    if len(values) != 1:
        raise UnsupportedOperation("local validity must have one band matching its Raster metadata")
    if float_cast:
        try:
            values = tuple(struct.unpack("f", struct.pack("f", float(v)))[0] for v in values)
        except (OverflowError, ValueError) as error:
            raise UnsupportedOperation("local validity must contain finite binary values") from error
    if any(v not in (0, 1) for v in values):
        raise UnsupportedOperation("local validity must contain finite binary values")


def _validity_mask(validity: Raster | None, raster: Raster) -> ee.Image | None:
    if validity is None:
        return None
    if not isinstance(validity, Raster):
        raise TypeError("validity must be a Raster")
    if validity.grid != raster.grid:
        raise UnsupportedOperation("validity must use the identical output grid")
    _band_names(validity.bands, 1, "validity bands")
    _check_literal_validity(validity.image)
    image = validity.image.select(list(validity.bands))
    return image.eq(1).And(image.mask().gt(0)).unmask(0, sameFootprint=False)


def _masked_scores(raster: Raster, image: ee.Image, validity: Raster | None) -> Raster:
    mask = _common_score_mask(image, raster.grid)
    explicit = _validity_mask(validity, raster)
    if explicit is not None:
        mask = mask.And(explicit)
    return Raster(image.updateMask(mask), raster.bands, raster.grid)


def segmentation_outputs(logits: Raster, *, logit_bands: Sequence[str] | None = None,
                         probability_bands: Sequence[str] | None = None,
                         class_band: str = "class_index",
                         validity: Raster | None = None,
                         policy: str = FLOAT32_OPERANDS) -> SegmentationOutputs:
    """Retain raw logits and return float32 probabilities/int32 logit argmax.

    Ordered channel indices define classes; exact logit ties choose the first.
    Processed outputs share all score masks/finiteness, finite grid and optional
    explicit validity. Raw logits retain their original values and masks.
    """
    validate_policy(policy, allowed=FLOAT32_OPERANDS)
    image = _scores(logits)
    count = len(logits.bands)
    logit_names = logits.bands if logit_bands is None else _band_names(logit_bands, count, "logit_bands")
    probability_names = (_band_names(probability_bands, count, "probability_bands") if probability_bands is not None
                         else tuple(f"probability_{i}" for i in range(count)))
    class_name = _band_names((class_band,), 1, "class_band")
    raw = logits if logit_bands is None else Raster(image.rename(list(logit_names)), logit_names, logits.grid)
    masked = _masked_scores(logits, image, validity)
    # The softmax block materialized as float32, as tensor_ee.softmax does.
    probabilities = _float32(_softmax(masked))
    probabilities = Raster(probabilities.image.rename(list(probability_names)), probability_names, logits.grid)
    # Do not derive the winner from rounded probabilities: [0,1e-8] can round
    # to [.5,.5]. arrayArgmax returns the first maximum of the ordered logits.
    winner = masked.image.toArray().arrayArgmax().arrayGet([0]).toInt32()
    classes = _finish(winner, class_name, logits.grid)
    return SegmentationOutputs(raw, probabilities, classes)


def _threshold_option(threshold: float) -> float:
    if isinstance(threshold, bool) or not isinstance(threshold, Real):
        raise UnsupportedOperation("threshold must be a static finite real scalar")
    if not 0 <= threshold <= 1:
        raise UnsupportedOperation("threshold must satisfy 0 <= threshold <= 1")
    number = _real_option(threshold, "threshold")
    if threshold != 0 and number == 0:
        raise UnsupportedOperation("threshold must be representable in binary64 without underflow to zero")
    return number


def binary_threshold(probabilities: Raster, *, positive_class: int = 1,
                     threshold: float = 0.5, output_band: str = "class_mask",
                     validity: Raster | None = None,
                     policy: str = FLOAT32_OPERANDS) -> Raster:
    """Inclusive uint8 threshold on one of exactly two normalized channels.

    Widen float32 probabilities losslessly and retain the finite binary64
    threshold, so probability .5 is below threshold .50000001. Invalid pixels
    are excluded rather than filled with class zero. Caller supplies normalized
    probabilities and asserts binary/finiteness for computed/remote validity.
    """
    validate_policy(policy, allowed=FLOAT32_OPERANDS)
    image = _scores(probabilities, binary=True)
    if isinstance(positive_class, bool) or not isinstance(positive_class, Integral) or positive_class not in (0, 1):
        raise UnsupportedOperation("positive_class must be the static index 0 or 1")
    value = _threshold_option(threshold)
    name = _band_names((output_band,), 1, "output_band")
    masked = _masked_scores(probabilities, image, validity)
    decision = masked.image.select(probabilities.bands[int(positive_class)]).toDouble().gte(_ee_scalar(value)).toUint8()
    return _finish(decision, name, probabilities.grid)


# ---------------------------------------------------------------------------
# Locality and cost metadata (contract section 9)
# ---------------------------------------------------------------------------
METADATA = validate_metadata({
    # Logits, float32 probabilities and an int32 class band, all on the input grid.
    # Validated: any accepted call (T03 varied the validity input; band names are labels).
    "segmentation_outputs": BlockMetadata(
        name="segmentation_outputs", layout_in="R", layout_out="R", locality=Locality.POINTWISE,
        policy=FLOAT32_OPERANDS, embedded_constants=_no_coefficients,
        asset_arrays=_no_coefficients, status="Ready",
        evidence=("docs/evidence/t03-outputs.json", "docs/evidence/t05-primary.json"),
        validated=ANY_SETTINGS),
    # Validated: any accepted call (T03: thresholds 0, 0.5, 1 and 0.50000001, both classes).
    "binary_threshold": BlockMetadata(
        name="binary_threshold", layout_in="R", layout_out="R", locality=Locality.POINTWISE,
        policy=FLOAT32_OPERANDS, embedded_constants=_no_coefficients,
        asset_arrays=_no_coefficients, status="Ready",
        evidence=("docs/evidence/t03-outputs.json",),
        validated=ANY_SETTINGS),
})
