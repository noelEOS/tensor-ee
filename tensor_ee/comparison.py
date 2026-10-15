"""Local exact-pixel segmentation comparison, independent of EE requests.

Callers retrieve ordered value and raw mask channels on explicit grids. Recipe
preprocessing/source validity and remote metadata checks stay with the caller.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from numbers import Real
from typing import Any

import numpy as np
import torch
from torch import nn

from .compiler import Grid
from .outputs import _threshold_option


@dataclass(frozen=True)
class PixelTensor:
    """Retrieved CHW values, independent boolean masks, ordered names and grid.

    This is local array metadata, not a remote raster inspector. Masked payloads
    are ignored; unmasked NaN/Inf is rejected. Arrays are not copied or cast.
    """

    values: np.ndarray
    mask: np.ndarray
    bands: tuple[str, ...]
    grid: Grid

    def __post_init__(self) -> None:
        if not isinstance(self.grid, Grid):
            raise TypeError('grid must be a Grid')
        if (not isinstance(self.bands, tuple) or not self.bands
                or any(not isinstance(x, str) or not x for x in self.bands)
                or len(set(self.bands)) != len(self.bands)):
            raise ValueError('bands must be an ordered tuple of unique nonempty names')
        shape = (len(self.bands), self.grid.height, self.grid.width)
        if not isinstance(self.values, np.ndarray) or self.values.shape != shape:
            raise ValueError(f'values must be a CHW array with shape {shape}')
        if self.values.dtype.kind not in 'fiu':
            raise ValueError('values must have a real numeric dtype')
        if not isinstance(self.mask, np.ndarray) or self.mask.shape != shape or self.mask.dtype != np.bool_:
            raise ValueError(f'mask must be an independent boolean CHW array with shape {shape}')
        if not np.isfinite(self.values[self.mask]).all():
            raise ValueError('unmasked values must be finite')


def _mask(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != np.bool_:
        raise ValueError(f'{name} must be a boolean array with shape {shape}')
    return value


def _metrics(actual: np.ndarray, reference: np.ndarray, mask: np.ndarray, tolerance: float) -> dict[str, Any]:
    count = int(mask.sum())
    if count == 0:
        return dict(valid_count=0, max_absolute_error=None, mae=None, rmse=None,
                    max_relative_error_floor_1e_3=None, tolerance_pass=False)
    a, r = actual[mask].astype(np.float64), reference[mask].astype(np.float64)
    error = np.abs(a - r)
    if not np.isfinite(error).all():
        return dict(valid_count=count, max_absolute_error=None, mae=None, rmse=None,
                    max_relative_error_floor_1e_3=None, tolerance_pass=False)
    return dict(valid_count=count, max_absolute_error=float(error.max()), mae=float(error.mean()),
                rmse=float(np.sqrt(np.mean(error * error))),
                max_relative_error_floor_1e_3=float((error / np.maximum(np.abs(r), 1e-3)).max()),
                tolerance_pass=bool(np.all(error <= tolerance)))


def _identity(tensor: PixelTensor) -> dict[str, Any]:
    return dict(shape=list(tensor.values.shape), dtype=str(tensor.values.dtype), bands=list(tensor.bands),
                grid=asdict(tensor.grid), valid_values=int(tensor.mask.sum()),
                values_sha256=hashlib.sha256(np.ascontiguousarray(tensor.values).tobytes()).hexdigest(),
                mask_sha256=hashlib.sha256(np.ascontiguousarray(tensor.mask).tobytes()).hexdigest())


def compare_segmentation(model: nn.Module, prepared: PixelTensor, logits: PixelTensor,
                         probabilities: PixelTensor, classes: PixelTensor, *,
                         expected_prepared: PixelTensor, expected_logit_mask: np.ndarray,
                         expected_output_mask: np.ndarray,
                         pixel_coordinates: tuple[np.ndarray, np.ndarray],
                         absolute_tolerance: float = 1e-4, binary_mask: PixelTensor | None = None,
                         positive_class: int = 1, threshold: float = 0.5) -> dict[str, Any]:
    """Compare retrieved outputs with CPU PyTorch on the actual retrieved input.

    The prepared tensor must be dense finite float32, after the caller's fill.
    Independent expected masks, rather than observed output masks, define the
    comparison denominator. Mask disagreements always fail. Coordinates are
    input-grid tensor indices (x rightward, y downward). Output grids/names are
    caller assertions validated before retrieval; channels are never sorted.
    Returns JSON-safe measurements; no EE call, model coercion or accuracy claim.
    """
    tensors = (prepared, logits, probabilities, classes, expected_prepared)
    if any(not isinstance(x, PixelTensor) for x in tensors):
        raise TypeError('comparison inputs must be PixelTensor instances')
    if any(x.values.dtype != np.float32 for x in (prepared, expected_prepared, logits, probabilities)):
        raise ValueError('prepared, logits and probabilities must use float32')
    if classes.values.dtype != np.int32 or len(classes.bands) != 1:
        raise ValueError('classes must be a one-band int32 tensor')
    if prepared.grid != expected_prepared.grid or prepared.bands != expected_prepared.bands:
        raise ValueError('expected prepared input must have identical grid and ordered bands')
    if logits.grid != probabilities.grid or logits.grid != classes.grid:
        raise ValueError('output tensors must use identical grids')
    if len(logits.bands) < 2 or len(probabilities.bands) != len(logits.bands):
        raise ValueError('logits/probabilities must have the same ordered class count, at least two')
    if not prepared.mask.all() or not expected_prepared.mask.all():
        raise ValueError('prepared input must be dense after explicit model-domain fill')
    if (isinstance(absolute_tolerance, bool) or not isinstance(absolute_tolerance, Real)
            or not np.isfinite(absolute_tolerance) or absolute_tolerance < 0):
        raise ValueError('absolute_tolerance must be a finite nonnegative real')
    tolerance = float(absolute_tolerance)
    expected_logit_mask = _mask(expected_logit_mask, logits.values.shape, 'expected_logit_mask')
    output_shape = (logits.grid.height, logits.grid.width)
    expected_output_mask = _mask(expected_output_mask, output_shape, 'expected_output_mask')
    if np.any(expected_output_mask & ~expected_logit_mask.all(axis=0)):
        raise ValueError('expected output validity must exclude any masked score channel')
    if (not isinstance(pixel_coordinates, tuple) or len(pixel_coordinates) != 2
            or any(not isinstance(x, np.ndarray) or x.shape != prepared.values.shape[1:] for x in pixel_coordinates)):
        raise ValueError('pixel_coordinates must be input-grid (pixel_x, pixel_y) arrays')
    if not isinstance(model, nn.Module):
        raise TypeError('model must be a torch.nn.Module')
    if any(x.training for x in model.modules()):
        raise ValueError('reference model and submodules must already be in eval mode')
    if any(x.device.type != 'cpu' or (x.is_floating_point() and x.dtype != torch.float32)
           for x in (*model.parameters(), *model.buffers())):
        raise ValueError('reference model must already use CPU float32 floating tensors')

    failures = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    y, x = np.indices(prepared.values.shape[1:], dtype=np.float32)
    require(np.array_equal(pixel_coordinates[0], x) and np.array_equal(pixel_coordinates[1], y),
            'retrieved pixel coordinates disagree with input grid orientation')
    bit_mismatch = prepared.values.view(np.uint32) != expected_prepared.values.view(np.uint32)
    require(not bit_mismatch.any(), 'retrieved prepared input differs from expected IEEE float32 bits')
    # Validate every expected output location; observed holes cannot shrink it.
    logit_mask_mismatch = int(np.count_nonzero(logits.mask != expected_logit_mask))
    probability_mask_mismatch = int(np.count_nonzero(probabilities.mask != expected_output_mask[None]))
    class_mask_mismatch = int(np.count_nonzero(classes.mask[0] != expected_output_mask))
    require(logit_mask_mismatch == 0, 'raw logit masks differ from independent expectation')
    require(probability_mask_mismatch == 0, 'probability masks differ from independent expectation')
    require(class_mask_mismatch == 0, 'class masks differ from independent expectation')
    with torch.inference_mode():
        reference = model(torch.from_numpy(np.array(prepared.values, copy=True, order="C")).unsqueeze(0))
        if (not isinstance(reference, torch.Tensor) or reference.dtype != torch.float32
                or reference.device.type != 'cpu' or tuple(reference.shape) != (1, *logits.values.shape)):
            raise ValueError('reference must return one matching CPU float32 NCHW logit tensor')
        ref_logits = reference[0].numpy()
        ref_probabilities = reference.softmax(dim=1)[0].numpy()
        ref_classes = reference.argmax(dim=1)[0].numpy().astype(np.int32)
    require(np.isfinite(ref_logits[expected_logit_mask]).all(), 'reference logits are nonfinite on expected valid scores')
    require(np.isfinite(ref_probabilities[:, expected_output_mask]).all(), 'reference probabilities are nonfinite on expected valid outputs')
    logit_metrics = _metrics(logits.values, ref_logits, expected_logit_mask, tolerance)
    probability_metrics = _metrics(probabilities.values, ref_probabilities,
                                   np.broadcast_to(expected_output_mask, probabilities.values.shape), tolerance)
    require(logit_metrics['tolerance_pass'], 'raw logits exceed absolute tolerance or have no comparable values')
    require(probability_metrics['tolerance_pass'], 'probabilities exceed absolute tolerance or have no comparable values')
    decisions = int(np.count_nonzero((classes.values[0] != ref_classes) & expected_output_mask))
    require(decisions == 0, 'ordered raw-logit argmax decisions disagree')
    observed_argmax = logits.values.argmax(axis=0).astype(np.int32)
    require(np.array_equal(classes.values[0][expected_output_mask], observed_argmax[expected_output_mask]),
            'retrieved classes disagree with retrieved ordered raw logits')
    valid_probabilities = probabilities.values[:, expected_output_mask]
    range_pass = bool(np.isfinite(valid_probabilities).all() and np.all((valid_probabilities >= 0) & (valid_probabilities <= 1)))
    require(range_pass, 'retrieved probabilities are outside finite [0,1]')
    sum_error = float(np.abs(valid_probabilities.astype(np.float64).sum(axis=0) - 1).max()) if expected_output_mask.any() else None
    require(sum_error is not None and sum_error <= 2e-6, 'retrieved probability sums exceed tolerance')
    near_margin = []
    if np.isfinite(ref_logits[:, expected_output_mask]).all():
        ordered = np.sort(ref_logits.astype(np.float64), axis=0)
        margin = ordered[-1] - ordered[-2]
        locations = np.argwhere(expected_output_mask & (margin <= 2 * tolerance))
        for row, column in locations:
            near_margin.append(dict(row=int(row), column=int(column), reference_margin=float(margin[row, column]),
                                    reference_class=int(ref_classes[row, column]), observed_class=int(classes.values[0, row, column])))
    binary = None
    if binary_mask is not None:
        if (not isinstance(binary_mask, PixelTensor) or len(logits.bands) != 2
                or binary_mask.grid != logits.grid or len(binary_mask.bands) != 1
                or binary_mask.values.dtype != np.uint8):
            raise ValueError('binary mask must be one-band uint8 on the binary output grid')
        if isinstance(positive_class, bool) or not isinstance(positive_class, (int, np.integer)) or positive_class not in (0, 1):
            raise ValueError('positive_class must be 0 or 1')
        scalar = _threshold_option(threshold)
        mask_mismatches = int(np.count_nonzero(binary_mask.mask[0] != expected_output_mask))
        decisions_binary = int(np.count_nonzero((binary_mask.values[0] != (ref_probabilities[positive_class].astype(np.float64) >= scalar)) & expected_output_mask))
        own = probabilities.values[positive_class].astype(np.float64) >= scalar
        own_mismatches = int(np.count_nonzero((binary_mask.values[0] != own) & expected_output_mask))
        require(mask_mismatches == 0 and decisions_binary == 0 and own_mismatches == 0, 'binary masks/threshold decisions disagree')
        near_threshold = np.argwhere(expected_output_mask & (np.abs(ref_probabilities[positive_class].astype(np.float64) - scalar) <= tolerance))
        binary = dict(positive_class=int(positive_class), threshold=scalar, mask_mismatches=mask_mismatches,
                      reference_disagreements=decisions_binary, retrieved_probability_disagreements=own_mismatches,
                      near_threshold_pixels=[dict(row=int(row), column=int(column),
                          reference_probability=float(ref_probabilities[positive_class,row,column]),
                          observed_probability=float(probabilities.values[positive_class,row,column]))
                          for row,column in near_threshold])
    return dict(schema_version=1, status='pass' if not failures else 'fail', failures=failures,
                absolute_tolerance=tolerance, reference_dtype='float32', execution_dtype='float32',
                prepared=_identity(prepared), logits=_identity(logits), probabilities=_identity(probabilities),
                classes=_identity(classes), prepared_bit_mismatches=int(bit_mismatch.sum()),
                prepared_bit_mismatch_locations_chw=np.argwhere(bit_mismatch).tolist(),
                mask_mismatches=dict(logits=logit_mask_mismatch, probabilities=probability_mask_mismatch, classes=class_mask_mismatch),
                valid_output_pixels=int(expected_output_mask.sum()), logit_errors=logit_metrics,
                probability_errors=probability_metrics, probability_range_pass=range_pass,
                max_probability_sum_error=sum_error, probability_sum_tolerance=2e-6, class_disagreements=decisions,
                near_margin_pixels=near_margin, binary_threshold=binary, accuracy='not_measured')
