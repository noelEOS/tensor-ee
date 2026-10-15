"""Recipe-free segmentation outputs on explicit finite raster grids.

Scores/probabilities are float32. Classes are the ordered raw-logit argmax;
thresholds compare widened probabilities against a retained binary64 scalar.
These functions build deferred graphs without cloud inspection or initialization.

The definitions moved to the library (L03), ``tensor_ee.blocks.raster.outputs``;
every name here is a re-export of the same object. (The private helper re-exports that
nothing calls any more were removed at L06b.)
"""
from __future__ import annotations

from .compiler import Raster, UnsupportedOperation  # noqa: F401
from .primitives import softmax  # noqa: F401
from .blocks.raster.outputs import (SegmentationOutputs, _threshold_option,  # noqa: F401
                                    binary_threshold, segmentation_outputs)
