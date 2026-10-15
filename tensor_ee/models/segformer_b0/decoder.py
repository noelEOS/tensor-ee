"""SMP SegFormer decoder and the fixed model's explicit raster interface.

Project C4,C3,C2,C1, resize each to C1 with align_corners=False, concatenate
in that order, apply bias-free Conv1x1 -> explicit evaluation BatchNorm -> ReLU,
then biased head Conv1x1 -> four-times bilinear (align_corners=True). The output
is two raw logit bands; there is no sensor preprocessing or probability recipe.
The two composites, ``project_pyramid`` and ``fuse_head``, moved to the library at
X01a (``tensor_ee/blocks/window/segformer.py``); ``project_pyramid`` is re-exported
here, and their constants builders followed at X02a. The generic input relabel and inverse output relabel moved at X02a
(``tensor_ee/layouts/boundaries.py``: ``relabel_to_windows``,
``relabel_from_windows``). This module keeps the fixed configuration and its checks.

NumPy coefficients are checked locally; ServerArrays and input raster values
must already be finite float32, and input pixels must all be valid. These server
preconditions cannot be proved without evaluation. Input masks are retained;
the interface never silently fills missing input pixels. No constructor or
graph builder initializes Earth Engine or makes a client evaluation call.

The fixed chip/global attention are EXTENT and chip-edge dependent. This model
has offline graph construction, selected live small-composition evidence, and
full staged live agreement on one seeded chip/state under the 1e-4 absolute
target (docs/evidence/segformer_b0-fullmodel-live.json). This covers complete
logits and the source-grid inverse relabel; it carries no accuracy or
monolithic-runtime claim.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...blocks.window.normalization import BatchNormConstants
from ...blocks.window.segformer import (_check_feature, _linear_constants, _parameter,  # noqa: F401  (re-exported)
                                        fuse_head, fuse_head_constants, project_pyramid,
                                        project_pyramid_constants)
from ...core.errors import reject
from ...core.grid import to_grid
from ...core.ops import WindowOps
from ...core.tensor import Raster, WindowTensor
from ...layouts.boundaries import relabel_from_windows, relabel_to_windows
from .configuration import (CHIP_EDGE_DEPENDENT, LOCALITY, SEGFORMER_B0, STAGE_DIMS,  # noqa: F401
                            SegformerB0Config, check_parameters)
from .encoder import encoder, encoder_constants


@dataclass
class DecoderConstants:
    projections: tuple
    fusion: object
    batch_norm: BatchNormConstants
    head: object
    config: SegformerB0Config = SEGFORMER_B0


def decoder_constants(ops, parameters, *, config=SEGFORMER_B0):
    if config != SEGFORMER_B0:
        reject("only the fixed SegFormer-B0 configuration is supported")
    projections = project_pyramid_constants(ops, parameters, [f"decoder.mlp_stage.{i}.linear" for i in range(4)],
                                            tuple(reversed(STAGE_DIMS)), 256, layout=config.layout)
    fusion, bn, head = fuse_head_constants(ops, parameters,
                                           ("decoder.fuse_stage.0", "decoder.fuse_stage.1", "segmentation_head.0"),
                                           1024, 256, 2, eps=1e-5, layout=config.layout)
    return DecoderConstants(projections, fusion, bn, head, config)


def decoder(ops, features, c: DecoderConstants) -> dict:
    """C1..C4 -> W7 logits on the fixed model input grid, graph construction only."""
    if c.config != SEGFORMER_B0 or len(features) != 4:
        reject("decoder requires the fixed SegFormer-B0 configuration and four features")
    for feature, dim, grid in zip(features, STAGE_DIMS, c.config.stage_grids):
        _check_feature(feature, channels=dim, grid=grid)
    pyramid = project_pyramid(ops, features, c.projections)
    head = fuse_head(ops, pyramid["out"], c.fusion, c.batch_norm, c.head, channels=256, classes=2)
    if head["out"].grid != c.config.input_grid:
        reject("head grid must return to the fixed model input grid")
    return {"pyramid": pyramid, **head}


@dataclass
class ModelConstants:
    encoder: object
    decoder: DecoderConstants
    config: SegformerB0Config = SEGFORMER_B0


def model_constants(ee, parameters, *, config=SEGFORMER_B0) -> ModelConstants:
    """Validate the complete original SMP state and build all constants ahead.

    Accept NumPy float32 values or caller-provided ServerArrays. The BN counter
    is checked by the schema and intentionally creates no arithmetic constant.
    """
    if config != SEGFORMER_B0:
        reject("only the fixed SegFormer-B0 configuration is supported")
    check_parameters(parameters)
    ops = WindowOps(ee)
    return ModelConstants(encoder_constants(ops, parameters, config=config),
                          decoder_constants(ops, parameters, config=config), config)


def input_windows(ee, raster: Raster, *, config=SEGFORMER_B0) -> WindowTensor:
    """Explicitly relabel a valid float32 input onto the abstract model coordinates.

    The source can be an external north-up 224 x 224 grid; channel labels/order
    must be exactly b0,b1. No implicit zero fill, sensor transform or normalization.
    Fully valid finite float32 server values are an explicit caller precondition.
    """
    if config != SEGFORMER_B0 or not isinstance(raster, Raster) or raster.bands != config.bands:
        reject("input must be a Raster with ordered bands ('b0', 'b1') at the fixed configuration")
    source = to_grid(raster.grid)
    if (source.height, source.width) != (224, 224):
        reject("input must have exactly 224 x 224 pixels")
    # The external interface permits a north-up georeferenced chip, relabeled (not resampled).
    return relabel_to_windows(ee, raster, config.input_grid, window=config.window)


def segformer_b0(ee, raster: Raster, c: ModelConstants, *, training=False) -> dict:
    """Construct a fixed batch-one raw-logit graph, returning stage boundaries.

    This function never executes the model. Logits are returned on the original
    source chip grid, through an explicit inverse coordinate relabel.
    """
    if training is not False or c.config != SEGFORMER_B0:
        reject("SegFormer-B0 supports the fixed evaluation configuration only")
    if c.encoder.config != c.config or c.decoder.config != c.config:
        reject("model constants use inconsistent configurations")
    ops = WindowOps(ee)
    x = input_windows(ee, raster, config=c.config)
    encoded = encoder(ops, x, c.encoder)
    decoded = decoder(ops, [encoded[f"C{s}"] for s in range(1, 5)], c.decoder)
    return {"encoder": encoded, "decoder": decoded,
            "logits": relabel_from_windows(ee, decoded["out"], raster.grid, c.config.output_bands)}
