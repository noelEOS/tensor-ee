"""Raster-layout (R) blocks of the native EE library: the CNN path's compute blocks.

- ``conv``: ``conv2d``, ``grouped_conv2d``;
- ``pooling``: ``pool2d``;
- ``resample``: ``nearest_upsample``, ``BilinearConstants``, ``bilinear_upsample``;
- ``normalization``: ``BatchNormConstants``, ``batch_norm``;
- ``merge``: ``concatenate``, ``residual_add``;
- ``activations``: ``relu``, ``sigmoid``, ``leaky_relu``, ``silu``, ``gelu``, ``tanh``,
  ``relu6``, ``hardsigmoid``, ``hardswish``, ``eval_dropout``, ``common_score_mask``,
  ``softmax``;
- ``outputs``: ``segmentation_outputs``, ``binary_threshold``.

Each module declares ``METADATA`` (locality, policy, cost and status per block,
docs/library-contract.md section 9). Every block takes a keyword-only
``policy=`` that defaults to, and only accepts, ``float32_operands`` (section 7).

These are the blocks themselves. ``compile_model`` materializes each node with
``.toFloat()``, and the public ``tensor_ee`` primitives wrap the same blocks
with that materialization; neither cast is inside a block.

Modules here import only the standard library, ``ee``, NumPy, torch (the blocks
take PyTorch modules and options), ``tensor_ee.core``, ``tensor_ee.layouts``
and ``tensor_ee.weights``. Nothing here imports a research folder, the
converter (``compiler``, ``semantics``, ...) or the models.
"""

from .conv import conv2d, grouped_conv2d
from .pooling import pool2d
from .resample import BilinearConstants, bilinear_upsample, nearest_upsample
from .normalization import BatchNormConstants, batch_norm
from .merge import concatenate, residual_add
from .activations import (common_score_mask, eval_dropout, gelu, hardsigmoid, hardswish, leaky_relu,
                          relu, relu6, sigmoid, silu, softmax, tanh)
from .outputs import SegmentationOutputs, binary_threshold, segmentation_outputs

__all__ = ["conv2d", "pool2d", "nearest_upsample", "concatenate", "residual_add",
           "grouped_conv2d",
           "BilinearConstants", "bilinear_upsample",
           "BatchNormConstants", "batch_norm",
           "relu", "sigmoid", "leaky_relu", "silu", "gelu", "tanh", "relu6", "hardsigmoid",
           "hardswish", "eval_dropout", "common_score_mask", "softmax",
           "SegmentationOutputs", "segmentation_outputs", "binary_threshold"]
