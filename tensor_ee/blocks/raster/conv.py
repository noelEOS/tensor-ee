"""R convolution: PyTorch ``nn.Conv2d`` as ordered neighbourhood bands and
``matrixMultiply``.

Moved whole from ``tensor_ee.compiler`` (``conv2d``, ``_pair``, ``_names``) and
``tensor_ee.semantics`` (``conv_options`` with the coefficient checks it uses:
``_coefficient_field``, ``finite_coefficients``). The old names are re-exports
of these objects. The errors they raise, ``SemanticUnknown`` and ``reject``
(here ``_reject``), are in ``tensor_ee.core.errors``.

Supported settings (contract section 8): odd kernels with same padding, stride
1 or 2, dilation 1, groups 1, zero padding, finite float32 coefficients. Other
strides stay rejected.

Precision policy ``float32_operands`` (contract section 7): the weight and bias
arrays are cast ``.float()`` before ``matrixMultiply``. The node output is
materialized with ``.toFloat()`` by the caller (``compile_model`` or the
``tensor_ee.primitives`` facade), not here.
"""

from __future__ import annotations

from typing import Any

import ee
import torch
from torch import nn

from ...core.errors import SemanticUnknown, UnsupportedOperation, reject as _reject
from ...core.metadata import ANY_SETTINGS, BlockMetadata, Locality, validate_metadata
from ...core.precision import FLOAT32_OPERANDS, validate_policy
from ...core.tensor import Raster
# Library converters and coefficient transport. These module-level bindings are
# the names conv2d looks up (tests patch _inline_float32_array here).
from ...layouts.raster import project as _project, stride_sample as _stride_sample
from ...layouts.raster import zero_extended as _zero_extended
from ...weights.constants import inline_float32_array as _inline_float32_array


def _pair(value: Any) -> tuple[Any, Any]:
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise UnsupportedOperation(f"Expected a scalar or pair, received {value!r}")
        return tuple(value)
    return value, value


def _names(count: int) -> tuple[str, ...]:
    return tuple(f"c{i}" for i in range(count))


def finite_coefficients(module: nn.Module) -> None:
    """Inspect local coefficients only, never an input-shaped activation."""
    for name, value in (*module.named_parameters(recurse=False), *module.named_buffers(recurse=False)):
        if not value.is_floating_point():
            continue
        if value.dtype != torch.float32:
            _reject("Only float32 model parameters and buffers are supported; convert with model.float()", "model_dtype")
        try:
            finite = bool(torch.isfinite(value.detach()).all().item())
        except Exception as error:
            raise SemanticUnknown(f"Cannot inspect local coefficient {name}: {type(error).__name__}: {error}") from error
        if not finite:
            _reject(f"Local coefficient {name} must contain only finite tensors", "nonfinite_coefficients")


def _coefficient_field(value: Any, name: str, shape: tuple[int, ...]) -> None:
    # Computational tensors require float32 even though the report's global
    # floating-dtype Boolean deliberately ignores integer bookkeeping.
    if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
        _reject(f"Only float32 computational coefficient {name} is supported", "model_dtype")
    try:
        actual = tuple(value.shape)
    except Exception as error:
        raise SemanticUnknown(f"Cannot inspect local coefficient {name} shape: {error}") from error
    if actual != shape:
        _reject(f"Local coefficient {name} shape must be {shape}, received {actual}", "coefficient_shape")


def conv_options(module: nn.Conv2d) -> tuple[tuple[int, int], int]:
    kh, kw = _pair(module.kernel_size)
    if kh % 2 != 1 or kw % 2 != 1:
        _reject("Conv2d kernels must have odd height and width")
    padding = (kh // 2, kw // 2) if module.padding == "same" else module.padding
    if padding != (kh // 2, kw // 2):
        _reject("Conv2d must use same padding (kernel_size // 2)")
    stride = _pair(module.stride)
    if (stride not in ((1, 1), (2, 2)) or _pair(module.dilation) != (1, 1)
            or module.groups != 1 or module.padding_mode != "zeros"):
        _reject("Conv2d requires stride=1 or 2, dilation=1, groups=1, padding_mode='zeros'")
    _coefficient_field(module.weight, "Conv2d.weight", (module.out_channels, module.in_channels, kh, kw))
    if module.bias is not None:
        _coefficient_field(module.bias, "Conv2d.bias", (module.out_channels,))
    finite_coefficients(module)
    return (kh, kw), stride[0]


def conv2d(raster: Raster, module: nn.Conv2d,
           parameters: tuple[ee.Array, ee.Array | None] | None = None,
           *, execution_dtype: str = "float32", clip_footprints: bool = False,
           policy: str = FLOAT32_OPERANDS) -> Raster:
    """PyTorch cross-correlation via ordered neighborhood bands and matrixMultiply."""
    validate_policy(policy, allowed=FLOAT32_OPERANDS)
    if execution_dtype != "float32":
        raise ValueError("execution_dtype must be float32 (maximum supported precision)")
    (kh, kw), stride_value = conv_options(module)
    stride = (stride_value, stride_value)
    if module.in_channels != len(raster.bands):
        raise UnsupportedOperation("Conv2d input channel count does not match raster")
    kernel = ee.Kernel.fixed(kw, kh, [[1] * kw for _ in range(kh)], kw // 2, kh // 2, False)
    # Never rely on the EE-generated default band ordering. Torch's flattened
    # OIHW weights require channels, then rows, then columns. EE suffixes use
    # offsets relative to the kernel origin, with x before y.
    patch_bands = [f"{band}_{dx}_{dy}" for band in raster.bands
                   for dy in range(-(kh // 2), kh // 2 + 1)
                   for dx in range(-(kw // 2), kw // 2 + 1)]
    patch = (_zero_extended(raster).neighborhoodToBands(kernel)
             .select(patch_bands).toArray().toArray(1))
    if parameters is None:
        weights = _inline_float32_array(module.weight.detach().cpu().float()
                           .reshape(module.out_channels, -1).tolist())
        bias = (None if module.bias is None else
                _inline_float32_array([[value] for value in module.bias.detach().cpu().float().tolist()]))
    else:
        weights, bias = parameters
    if execution_dtype == "float32":
        # Keep matrix operands in float32 rather than promoting the entire
        # activation neighborhood to double through the frozen weights.
        weights = weights.float()
        if bias is not None:
            bias = bias.float()
    result = ee.Image(weights).matrixMultiply(patch)
    if bias is not None:
        result = result.add(ee.Image(bias))
    bands = _names(module.out_channels)
    result = result.arrayProject([0]).arrayFlatten([list(bands)])
    # Make the dense convolution's sampling grid explicit before decimation.
    # Stride one already receives that projection in _finish; avoid a second
    # redundant reproject in the long residual/decoder graph.
    if stride[0] > 1:
        result = _project(result, raster.grid, clip_footprints)
    return _stride_sample(result, bands, raster.grid, stride[0], clip_footprints)


# ---------------------------------------------------------------------------
# Locality and cost metadata (contract section 9)
# ---------------------------------------------------------------------------
def _conv_coefficients(module: nn.Conv2d) -> int:
    """Values embedded in the request when the coefficients are inline:
    ``out * in * kh * kw`` weights plus ``out`` biases."""
    kh, kw = _pair(module.kernel_size)
    count = module.out_channels * module.in_channels * kh * kw
    return count + (module.out_channels if module.bias is not None else 0)


def _conv_asset_arrays(module: nn.Conv2d) -> int:
    """Server arrays read when the coefficients are supplied as EE arrays:
    the weight matrix, plus the bias column."""
    return 1 + (module.bias is not None)


def _no_coefficients(*args, **kwargs) -> int:
    """Cost of an R block without weights: it embeds no coefficients and reads
    no asset arrays. Scalar options (a slope, a threshold, the float32 maximum)
    are graph literals, not coefficients, and are not counted."""
    return 0


METADATA = validate_metadata({
    # Radius k // 2 for a k x k kernel (receptive_field._window_axis with
    # padding k // 2). The static field records the largest validated kernel,
    # 7 (radius 3); kernels 1 and 3 have radius 0 and 1.
    # Validated: any accepted call. Kernels 1, 3 and 7 at strides 1 and 2, and
    # clip_footprints both ways (True in resnet18_full64_float32_policy.json), were run;
    # the unqualified "Ready" covers them, and other forms are refused.
    "conv2d": BlockMetadata(
        name="conv2d", layout_in="R", layout_out="R", locality=Locality.KERNEL,
        policy=FLOAT32_OPERANDS, embedded_constants=_conv_coefficients,
        asset_arrays=_conv_asset_arrays, radius=3, status="Ready",
        evidence=("docs/history/cnn/results/parity.json", "docs/history/cnn/results/primitives_float32.json",
                  "docs/history/cnn/results/resnet_primitives.json", "docs/history/cnn/results/resnet18_full64_float32_policy.json",
                  "docs/history/cnn/results/segwater_js_parity_1x1.json", "docs/history/cnn/results/tiled_resnet18_2x2_parity.json"),
        validated=ANY_SETTINGS),
})
