"""Audited SegWater v3 VV/VH preprocessing, in NumPy and native Earth Engine.

The v3 inference source casts dB inputs to float32, rounds onto the q55s005
code lattice, clamps codes to [1, 65535], decodes them, then standardizes.
Code 1 decodes to approximately -54.95 dB; code 0 is reserved for NoData.
There is no additional ``db < -55 -> -55`` override in that source.

NumPy ``round``/``rint`` uses ties to even. The EE implementation spells out
that rule and casts after each arithmetic operation, rather than depending on
EE ``round`` or silently promoting the chain to double precision.

Source masks have no counterpart in the reference's ordinary ``rasterio.read``
arrays. Callers must choose how to handle them: preserve each channel's mask,
or fill each masked channel (including outside its footprint) with its training
mean *before* re-quantization. The latter produces small nonzero z-values and
is also the reference recipe for out-of-bounds context padding. A numeric dB
value of zero or -55 is never guessed to be NoData.
"""

from __future__ import annotations

from typing import Literal, Sequence

import ee
import numpy as np


BANDS = ("VV", "VH")
MEANS = (-15.435106, -24.856323)
STDS = (6.517790, 9.117346)
OFFSET_DB = np.float32(55.0)
STEP_DB = np.float32(0.05)
MIN_CODE = 1
MAX_CODE = 65535
MaskedPolicy = Literal["preserve", "training_mean"]


def _check_policy(masked_policy: str) -> None:
    if masked_policy not in ("preserve", "training_mean"):
        raise ValueError("masked_policy must be 'preserve' or 'training_mean'")


def _channel_constants(shape: tuple[int, ...], channel_axis: int):
    if not shape:
        raise ValueError("Expected VV/VH channels, not a scalar")
    if not isinstance(channel_axis, int) or not -len(shape) <= channel_axis < len(shape):
        raise ValueError("channel_axis must identify an input dimension")
    channel_axis %= len(shape)
    if shape[channel_axis] != 2:
        raise ValueError("Expected exactly two channels in VV, VH order")
    broadcast_shape = [1] * len(shape)
    broadcast_shape[channel_axis] = 2
    means = np.asarray(MEANS, dtype=np.float32).reshape(broadcast_shape)
    stds = np.asarray(STDS, dtype=np.float32).reshape(broadcast_shape)
    return means, stds


def quantize_db_numpy(db) -> np.ndarray:
    """Return float32 decoded q55s005 dB using the audited operation order.

    This low-level numeric reference has the same nonfinite behavior as NumPy
    inference: NaN propagates and infinities clamp to the endpoint codes. Masks
    should be handled by ``preprocess_db_numpy`` before calling this function.
    """
    data = np.asarray(db, dtype=np.float32)
    with np.errstate(invalid="ignore", over="ignore"):
        shifted = np.add(data, OFFSET_DB, dtype=np.float32)
        scaled = np.divide(shifted, STEP_DB, dtype=np.float32)
        code = np.clip(np.rint(scaled), MIN_CODE, MAX_CODE).astype(np.float32)
        decoded = np.multiply(code, STEP_DB, dtype=np.float32)
        return np.subtract(decoded, OFFSET_DB, dtype=np.float32)


def preprocess_db_numpy(db, *, channel_axis: int = 0,
                        masked_policy: MaskedPolicy = "preserve"):
    """Quantize and standardize VV/VH, preserving float32 intermediate results.

    Accepts CHW by default, or NCHW with ``channel_axis=1``. A masked input
    preserves its per-channel mask under ``preserve`` and returns a MaskedArray;
    ``training_mean`` fills masked entries in dB first and returns an ndarray.
    Ordinary arrays retain ordinary numeric values, including any raw NoData
    sentinel: their provenance cannot be reconstructed from values alone.
    """
    _check_policy(masked_policy)
    masked = np.ma.isMaskedArray(db)
    data = np.asarray(np.ma.getdata(db), dtype=np.float32)
    means, stds = _channel_constants(data.shape, channel_axis)
    mask = np.ma.getmaskarray(db) if masked else None
    if masked and masked_policy == "training_mean":
        data = np.where(mask, means, data).astype(np.float32)
    quantized = quantize_db_numpy(data)
    centered = np.subtract(quantized, means, dtype=np.float32)
    result = np.divide(centered, stds, dtype=np.float32)
    if masked and masked_policy == "preserve":
        return np.ma.array(result, mask=mask, copy=False)
    return result


def training_mean_padding_numpy(shape: Sequence[int], *, channel_axis: int = 0) -> np.ndarray:
    """Return v3 context padding, re-quantized rather than exactly zero."""
    shape = tuple(shape)
    if any(not isinstance(value, int) or value <= 0 for value in shape):
        raise ValueError("Padding dimensions must be positive integers")
    means, _ = _channel_constants(shape, channel_axis)
    return preprocess_db_numpy(np.broadcast_to(means, shape), channel_axis=channel_axis)


def round_ties_to_even_ee(value: ee.Image) -> ee.Image:
    """Numerical ties-even rounding for finite float32 values.

    A rounded negative zero may lose its sign; the immediately following
    positive-code clamp makes that irrelevant to q55s005 preprocessing.
    """
    value = value.toFloat()
    lower = value.floor().toFloat()
    fraction = value.subtract(lower).toFloat()
    odd_lower = lower.mod(2).neq(0)
    increment = fraction.gt(0.5).Or(fraction.eq(0.5).And(odd_lower))
    return lower.add(increment).toFloat()


def quantize_db_ee(db: ee.Image) -> ee.Image:
    """TensorEE q55s005 decode; masks and input band order are retained."""
    offset = ee.Image.constant(float(OFFSET_DB)).toFloat()
    step = ee.Image.constant(float(STEP_DB)).toFloat()
    shifted = db.toFloat().add(offset).toFloat()
    scaled = shifted.divide(step).toFloat()
    code = round_ties_to_even_ee(scaled).clamp(MIN_CODE, MAX_CODE).toFloat()
    decoded = code.multiply(step).toFloat()
    return decoded.subtract(offset).toFloat()


def preprocess_db_ee(image: ee.Image, input_bands: Sequence[str] = BANDS, *,
                     masked_policy: MaskedPolicy = "preserve") -> ee.Image:
    """Return float32 standardized input bands in the supplied VV/VH order.

    ``input_bands`` identifies VV then VH, even when the source uses other names.
    The result retains those names and the source projection. No request,
    initialization, download, or reprojection happens here. With
    ``training_mean``, the filled footprint is global; the caller must impose
    its finite inference grid after this preprocessing step.
    """
    _check_policy(masked_policy)
    input_bands = tuple(input_bands)
    if len(input_bands) != 2 or len(set(input_bands)) != 2:
        raise ValueError("input_bands must contain distinct VV and VH names in that order")
    data = image.select(list(input_bands)).toFloat()
    means = ee.Image.constant(np.asarray(MEANS, dtype=np.float32).tolist()).rename(
        list(input_bands)).toFloat()
    stds = ee.Image.constant(np.asarray(STDS, dtype=np.float32).tolist()).rename(
        list(input_bands)).toFloat()
    if masked_policy == "training_mean":
        data = data.unmask(means, sameFootprint=False).toFloat()
    centered = quantize_db_ee(data).subtract(means).toFloat()
    return centered.divide(stds).toFloat().rename(list(input_bands))


def preprocessing_metadata(masked_policy: MaskedPolicy = "preserve") -> dict:
    """Machine-readable numeric and mask contract for experiment provenance."""
    _check_policy(masked_policy)
    padding = training_mean_padding_numpy((2, 1, 1))[:, 0, 0]
    return {
        "lineage": "dataset_v3",
        "input_band_order": list(BANDS),
        "input_units": "dB",
        "encoding": "q55s005",
        "means_decimal": list(MEANS),
        "stds_decimal": list(STDS),
        "means_float32": np.asarray(MEANS, dtype=np.float32).tolist(),
        "stds_float32": np.asarray(STDS, dtype=np.float32).tolist(),
        "step_float32": float(STEP_DB),
        "offset_db": float(OFFSET_DB),
        "valid_code_range": [MIN_CODE, MAX_CODE],
        "decoded_code1_float32_db": float(quantize_db_numpy(-55)),
        "decoded_code65535_float32_db": float(quantize_db_numpy(4000)),
        "rounding": "ties_to_even_after_float32_add_and_divide",
        "arithmetic": "float32_cast_after_each_operation",
        "below_minus55_override": False,
        "masked_policy": masked_policy,
        "numeric_nodata_sentinel_inference": False,
        "training_mean_padding_z": padding.tolist(),
    }
