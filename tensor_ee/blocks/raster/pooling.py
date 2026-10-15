"""R pooling: PyTorch ``nn.AvgPool2d`` and ``nn.MaxPool2d`` at a small set of
common CNN configurations.

Moved whole from ``tensor_ee.compiler`` (``pool2d``, both branches) and
``tensor_ee.semantics`` (``pool_options``). The old names are re-exports of
these objects. Task P02 added the configurations beyond the two the reference
models use (``pool_config``).

Supported settings (contract section 8), all with ``ceil_mode=False``,
max-pool ``dilation=1`` and ``return_indices=False``, and average-pool
``divisor_override=None``:

- **aligned** average or max, kernel = stride = k, padding 0, for k in
  (2, 3, 4), on input sizes divisible by k: ``reduceResolution`` onto the
  k-times coarser grid;
- **max 3 x 3, padding 1, stride 1 or 2**: a 3 x 3 neighbourhood maximum on
  the input grid (then stride sampling for stride 2). The input is masked
  outside its grid and Earth Engine reducers skip masked pixels, so the padding
  never wins: PyTorch's negative-infinity padding;
- **average 3 x 3, stride 1, padding 1**, under both PyTorch divisor rules:
  ``count_include_pad=True`` divides the in-bounds sum by 9 (zero padding);
  ``count_include_pad=False`` takes the mean of the in-bounds pixels only.

Every other configuration is rejected with the list above. Each supported
configuration has a P02 fixture (``raster/pool2d/*`` in
``tools/fixtures/raster.py``) that compares it with PyTorch: bit for bit for
max pooling, within 1e-4 for averages.

Precision policy ``float32_operands`` (contract section 7). There are no
coefficients. The node output is materialized with ``.toFloat()`` by the caller
(``compile_model`` or the ``tensor_ee.primitives`` facade).
"""

from __future__ import annotations

from dataclasses import dataclass

import ee
from torch import nn

from ...core.errors import UnsupportedOperation
from ...core.metadata import BlockMetadata, Locality, Validated, validate_metadata
from ...core.precision import FLOAT32_OPERANDS, validate_policy
from ...core.tensor import Raster
from ...layouts.raster import finish as _finish, project as _project
from ...layouts.raster import stride_sample as _stride_sample
from ...core.errors import reject as _reject
from .conv import _no_coefficients, _pair

ALIGNED_KERNELS = (2, 3, 4)
"""Kernel sizes k of the aligned k x k pools (stride k, padding 0)."""

SUPPORTED = ("aligned avg/max k x k with stride k and padding 0 for k in (2, 3, 4) on sizes divisible "
             "by k; max 3x3 padding 1 with stride 1 or 2; avg 3x3 stride 1 padding 1")
"""The supported configurations, as the rejection message names them."""


@dataclass(frozen=True)
class PoolConfig:
    """One supported pooling configuration, normalized from the PyTorch module.

    ``reducer`` is ``"max"`` or ``"mean"``; ``kernel``, ``stride`` and
    ``padding`` are the equal per-axis values; ``count_include_pad`` is the
    average pool's divisor rule (``None`` for max pooling).
    """

    reducer: str
    kernel: int
    stride: int
    padding: int
    count_include_pad: bool | None = None

    @property
    def aligned(self) -> bool:
        """An aligned k x k pool: kernel = stride, no padding."""
        return self.padding == 0

    def output_size(self, size: int) -> int:
        """PyTorch's output size on one axis (``ceil_mode=False``)."""
        return (size + 2 * self.padding - self.kernel) // self.stride + 1


def aligned_size_message(kernel: int) -> str:
    """The rejection of an input size that an aligned k x k pool does not divide."""
    if kernel == 2:
        return "This experiment requires even spatial dimensions before pooling"
    return f"Aligned {kernel}x{kernel} pooling requires spatial dimensions divisible by {kernel}"


def pool_config(module: nn.AvgPool2d | nn.MaxPool2d) -> PoolConfig:
    """Validate the module's options and return its supported configuration."""
    stride = module.stride if module.stride is not None else module.kernel_size
    if module.ceil_mode is not False:
        _reject("Pooling requires ceil_mode=False")
    if isinstance(module, nn.MaxPool2d):
        if _pair(module.dilation) != (1, 1) or module.return_indices is not False:
            _reject("MaxPool2d requires dilation=1 and return_indices=False")
        reducer, include = "max", None
    elif module.divisor_override is not None or not isinstance(module.count_include_pad, bool):
        _reject("AvgPool2d requires divisor_override=None and static count_include_pad")
    else:
        reducer, include = "mean", module.count_include_pad
    kernel, strides, padding = _pair(module.kernel_size), _pair(stride), _pair(module.padding)
    for k in ALIGNED_KERNELS:
        if kernel == (k, k) and strides == (k, k) and padding == (0, 0):
            return PoolConfig(reducer, k, k, 0, include)
    if kernel == (3, 3) and padding == (1, 1):
        if strides == (1, 1) or (strides == (2, 2) and reducer == "max"):
            return PoolConfig(reducer, 3, int(strides[0]), 1, include)
    _reject(f"Pooling supports {SUPPORTED}, without ceil_mode; received {type(module).__name__}"
            f"(kernel_size={module.kernel_size!r}, stride={stride!r}, padding={module.padding!r})")


def pool_options(module: nn.AvgPool2d | nn.MaxPool2d) -> bool:
    """Validate the module and return whether it pools a padded 3x3 neighbourhood
    (``pool_config(module).padding``), not an aligned k x k block."""
    return not pool_config(module).aligned


def pool2d(raster: Raster, module: nn.AvgPool2d | nn.MaxPool2d,
           *, clip_footprints: bool = False, policy: str = FLOAT32_OPERANDS) -> Raster:
    """PyTorch average or max pooling at a supported configuration (``pool_config``)."""
    validate_policy(policy, allowed=FLOAT32_OPERANDS)
    config = pool_config(module)
    if not config.aligned:
        kernel = ee.Kernel.square(1, "pixels", False)
        if config.reducer == "max":
            # Masked padding contributes no value to max, just as -infinity.
            result = raster.image.reduceNeighborhood(ee.Reducer.max(), kernel, skipMasked=False)
        elif config.count_include_pad:
            # count_include_pad=True: the in-bounds sum over the full window of 9
            # (PyTorch's zero padding counts in the divisor).
            result = raster.image.reduceNeighborhood(ee.Reducer.sum(), kernel, skipMasked=False).divide(9)
        else:
            # count_include_pad=False: the mean of the in-bounds pixels only.
            result = raster.image.reduceNeighborhood(ee.Reducer.mean(), kernel, skipMasked=False)
        if config.stride == 1:
            # The output is on the input grid; _stride_sample(..., 1) is _finish,
            # which pins it there (as conv2d does at stride 1).
            return _stride_sample(result, raster.bands, raster.grid, 1, clip_footprints)
        result = _project(result, raster.grid, clip_footprints)
        return _stride_sample(result, raster.bands, raster.grid, config.stride, clip_footprints)
    k = config.kernel
    reducer = ee.Reducer.max() if config.reducer == "max" else ee.Reducer.mean()
    if raster.grid.width % k or raster.grid.height % k:
        raise UnsupportedOperation(aligned_size_message(k))
    grid = raster.grid.resized(k, raster.grid.width // k, raster.grid.height // k)
    # maxPixels (k + 2)^2 keeps today's 16 at k = 2: the k x k block plus a
    # one-pixel ring. It bounds the input pixels per output pixel; with
    # bestEffort=False a larger need fails the request rather than changing values.
    result = raster.image.reduceResolution(reducer, bestEffort=False, maxPixels=(k + 2) ** 2)
    result = _project(result, grid, clip_footprints)
    return _finish(result, raster.bands, grid, clip_footprints)


# ---------------------------------------------------------------------------
# Locality and cost metadata (contract section 9)
# ---------------------------------------------------------------------------
METADATA = validate_metadata({
    # receptive_field._window_axis: output pixel i of an aligned k x k pool reads
    # input pixels [k i, k i + k - 1], and a padded 3x3 pool with stride s reads
    # [s i - 1, s i + 1]. The radius bounds the offset from s i: k - 1 = 3 at the
    # largest aligned kernel, 4; 1 for the padded pools.
    # Validated, in PoolConfig's terms (``pool_config(module)``; "mean" is AvgPool2d): the
    # status's three families. count_include_pad does not change an unpadded pool, so the
    # aligned pools do not name it.
    "pool2d": BlockMetadata(
        name="pool2d", layout_in="R", layout_out="R", locality=Locality.KERNEL,
        policy=FLOAT32_OPERANDS, embedded_constants=_no_coefficients,
        asset_arrays=_no_coefficients, radius=3,
        status=("Ready — aligned avg/max k = 2, 3, 4; max 3×3 stride 1 or 2, padding 1; "
                "avg 3×3 stride 1 padding 1, both divisor rules"),
        evidence=("docs/evidence/live-fixtures/20261013T065257Z-p02.json", "docs/evidence/p02-raster-offline.json",
                  "docs/history/cnn/results/parity.json", "docs/evidence/t05-primary.json",
                  "docs/history/cnn/results/resnet_primitives.json", "docs/history/cnn/results/resnet18_full64_float32_policy.json",
                  "docs/history/cnn/results/segwater_js_parity_1x1.json"),
        validated=Validated((
            *({"reducer": ("mean", "max"), "kernel": (k,), "stride": (k,), "padding": (0,)} for k in ALIGNED_KERNELS),
            {"reducer": ("max",), "kernel": (3,), "stride": (1, 2), "padding": (1,)},
            {"reducer": ("mean",), "kernel": (3,), "stride": (1,), "padding": (1,), "count_include_pad": (True, False)},
        ), sources=(PoolConfig,))),
})
