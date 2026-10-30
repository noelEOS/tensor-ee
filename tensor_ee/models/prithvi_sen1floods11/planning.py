"""Offline capacity estimates and actual request sizing for explicit paired cuts.

This hand-composed model has no converter graph. These helpers use the planner's
Limits and the same request_size function, without claiming an optimizer-generated
plan. Separate class/spatial exports recompute shared work. Nothing is submitted.

At default dimensions, capacity estimates are 97,602,818 float state values,
122,783,330 logical constant values with shared matrices counted once, and
145.795 billion MACs per crop. The 3 x 3 head contributes about 60.897% of
MACs. Proposed cuts save paired spatial/class images after each encoder block
and features after each upsampling and before the head. Conservative separate
branch costs fit 24 million constants per encoder block; saved boundaries have
at most 768 bands. The largest activation has 602,112 values ([196,3072]),
or 4,816,896 bytes in double. The largest coefficient matrix holds 2,359,296
values. These array counts do not establish peak memory; request sizes depend
on the chosen table manifest.
"""
from __future__ import annotations

from collections.abc import Mapping
import math

from ...blocks.window.linear import linear_values
from ...blocks.window.normalization import layer_norm_values, batch_norm_values
from ...blocks.window.conv import conv3x3_values
from ...core.tensor import ServerArray, WindowTensor, Raster, band_names
from ...layouts.boundaries import boundary_raster, boundary_windows, saved_image
from ...converter.plan import Limits
from ...converter.request import request_size
from .configuration import PRITHVI, stage_names, parameter_shapes
from .model import INPUT_GRID, TOKEN_GRID, BANDS, compose, constants


def constant_costs(config=PRITHVI):
    """Logical evaluated coefficients; share each matrix across the two row branches."""
    dim = config.width
    projections = ((dim, 3 * dim), (dim, dim), (dim, 4 * dim), (4 * dim, dim))
    block = sum(linear_values(cin, cout, layout=197) for cin, cout in projections)
    block += 2 * layer_norm_values(dim, layout=197)
    conservative = sum(
        linear_values(cin, cout, layout=196) + linear_values(cin, cout, layout=1)
        for cin, cout in projections
    )
    conservative += 2 * (
        layer_norm_values(dim, layout=196) + layer_norm_values(dim, layout=1)
    )
    costs = {
        "patch": linear_values(1536, dim, layout=196) + 198 * dim,
        **{f"block{index}": block for index in range(config.depth)},
        "norm": layer_norm_values(dim, layout=197),
        "upsample28": (
            linear_values(dim, 4 * dim, layout=196) + layer_norm_values(dim, layout=49)
        ),
        "upsample56": linear_values(dim, 4 * dim, layout=49),
        "upsample112": (
            linear_values(dim, 4 * dim, layout=49) + layer_norm_values(dim, layout=49)
        ),
        "upsample224": linear_values(dim, 4 * dim, layout=49),
        "logits": (
            conv3x3_values(dim, config.decoder)
            + batch_norm_values(config.decoder, layout=49)
            + linear_values(config.decoder, 2, layout=49)
        ),
    }
    return {
        "stages": costs,
        "total": sum(costs.values()),
        "encoder_block_conservative": conservative,
        "float_state_values": sum(math.prod(shape) for shape in parameter_shapes(config).values()),
    }


def cut_proposal(config=PRITHVI, *, limits=None):
    """One block/cut is conservative; request measurements and runtime are still required."""
    limits = Limits() if limits is None else limits
    cost = constant_costs(config)
    cuts = []
    for name, values in cost["stages"].items():
        paired = name == "patch" or name.startswith("block") or name == "norm"
        conservative = cost["encoder_block_conservative"] if name.startswith("block") else values
        channels = 2 if name == "logits" else config.width
        cuts.append({
            "name": name,
            "constants": values,
            "conservative_constants": conservative,
            "constant_limit_satisfied": conservative <= limits.constants,
            "boundary_bands": [channels, channels] if paired else [channels],
            "band_limit_satisfied": channels <= limits.bands,
            "exports": 2 if paired else 1,
            "request_bytes": None,
            "runtime_measured": False,
            "extent_nodes": 0,
            "chip_edge_dependent": False,
            "crop_alignment": 224,
        })
    return {
        "kind": "explicit_cut_proposal",
        "limits": {
            "constants": limits.constants, "request_bytes": limits.request_bytes,
            "bands": limits.bands, "extent": limits.extent,
        },
        "largest_image_array": {
            "shape": [196, 4 * config.width], "values": 196 * 4 * config.width,
            "double_bytes": 8 * 196 * 4 * config.width,
        },
        "whole_model_constants": cost["total"],
        "whole_model_constant_limit_satisfied": cost["total"] <= limits.constants,
        "stages": cuts,
        "measured": False,
    }


def stage_rasters(ee, coefficients, *, config=PRITHVI, namespace="projects/example/assets/prithvi"):
    """Each explicit stage's exported images, built from the previous stage's saved assets.

    Yields ``(stage, [(name, raster), ...])`` in model order. The patch setup, each encoder
    block and the final norm export a spatial and a class image (``<stage>_spatial``,
    ``<stage>_class``); each upsampling and the logits export one. Stage inputs are read
    from ``<namespace>_<name>`` (the crop itself from ``<namespace>_input``), so a stage can
    be built only once its predecessor's assets exist. Builds expressions only.
    """
    previous = "input"

    def saved(name, grid, channels, window):
        image = saved_image(ee, namespace + "_" + name, band_names(channels), grid)
        return boundary_windows(ee, image, grid, channels, window=window)

    for stage in stage_names(config)[1:]:
        if previous == "input":
            image = saved_image(ee, namespace + "_input", BANDS, INPUT_GRID)
            incoming = Raster(image, BANDS, INPUT_GRID)
        elif previous == "patch" or previous.startswith("block") or previous == "norm":
            incoming = (
                saved(previous + "_spatial", TOKEN_GRID, config.width, 14),
                saved(previous + "_class", TOKEN_GRID.coarse(14), config.width, 1),
            )
        else:
            side = int(previous[8:])
            grid = INPUT_GRID.coarse(224 // side)
            incoming = saved(previous, grid, config.width, 7)
        output = compose(
            coefficients.backend, incoming, config, start=previous, stop=stage,
        )[stage]
        outputs = output if isinstance(output, tuple) else (output,)
        images = []
        for index, item in enumerate(outputs):
            raster = boundary_raster(ee, item) if isinstance(item, WindowTensor) else item
            suffix = ("_spatial", "_class")[index] if len(outputs) == 2 else ""
            images.append((stage + suffix, raster))
        yield stage, images
        previous = stage


def measured_cuts(ee, parameters, *, config=PRITHVI,
                  namespace="projects/example/assets/prithvi", limits=None):
    """Size every explicit stage with asset-backed weights and saved image inputs.

    Parameters must be ServerArrays; exact sizes depend on their actual table
    expressions and namespace. The result is an explicit measured proposal, not
    the converter's optimizer. EE signatures must already be available; this
    function initializes nothing, creates no asset and sends no request.
    """
    required = parameter_shapes(config)
    valid = isinstance(parameters, Mapping) and all(
        isinstance(parameters.get(key), ServerArray) for key in required
    )
    if not valid:
        raise ValueError("request sizing requires ServerArray inference parameters")
    limits = Limits() if limits is None else limits
    coefficients = constants(ee, parameters, config=config)
    records = []
    proposal = cut_proposal(config, limits=limits)
    stages = stage_rasters(ee, coefficients, config=config, namespace=namespace)
    for (stage, images), base in zip(stages, proposal["stages"]):
        sizes = [request_size(raster.image, raster.grid, asset_id=namespace + "_" + name,
                              description="prithvi_" + name)
                 for name, raster in images]
        records.append({
            **base,
            "request_bytes": sizes,
            "request_limit_satisfied": all(size <= limits.request_bytes for size in sizes),
        })
    within_limits = all(
        record["constant_limit_satisfied"] and record["band_limit_satisfied"]
        and record["request_limit_satisfied"]
        for record in records
    )
    return {**proposal, "stages": records, "measured": True, "within_limits": within_limits}
