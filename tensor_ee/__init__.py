"""Experimental PyTorch-compatible building blocks for native Earth Engine."""

from .comparison import PixelTensor, compare_segmentation
from .compatibility import CompatibilityReport, check_compatibility
from .compiler import CompiledModel, Grid, Raster, UnsupportedOperation, WeightProvider, compile_model
from .primitives import (concatenate, conv2d, eval_dropout, from_image, gelu, leaky_relu, nearest_upsample,
                         pool2d, relu, residual_add, sigmoid, silu, softmax, tanh)
from .outputs import SegmentationOutputs, binary_threshold, segmentation_outputs
from .weight_assets import FeatureCollectionWeightProvider, effective_convolutions, export_weights
from .weight_mapping import MatchedWeightProvider

__all__ = [
    "PixelTensor", "compare_segmentation",
    "CompatibilityReport", "check_compatibility",
    "CompiledModel", "Grid", "Raster", "UnsupportedOperation", "WeightProvider",
    "compile_model", "from_image", "conv2d", "pool2d", "nearest_upsample",
    "concatenate", "residual_add", "relu", "sigmoid", "leaky_relu", "silu", "gelu", "tanh", "eval_dropout", "softmax",
    "SegmentationOutputs", "segmentation_outputs", "binary_threshold",
    "FeatureCollectionWeightProvider", "MatchedWeightProvider",
    "effective_convolutions", "export_weights",
]
