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
from .model import BANDS, compose, constants, region_grid


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


def stage_cuts(cuts=None, *, config=PRITHVI, start="input"):
    """The validated saved boundaries a staged run exports, after ``start``.

    ``None`` means every boundary after ``start``: one stage per boundary, as in
    ``measured_cuts``. Otherwise the cuts must name boundaries after ``start`` in
    strictly increasing model order; a list need not end at the logits.
    """
    names = stage_names(config)
    if start not in names or start == names[-1]:
        raise ValueError(f"start must name a boundary before {names[-1]!r}: {start!r}")
    if cuts is None:
        return names[names.index(start) + 1:]
    cuts = tuple(cuts)
    unknown = [cut for cut in cuts if cut not in names]
    if not cuts or unknown:
        raise ValueError(f"cuts must name saved boundaries {names[1:]}: {unknown or 'none given'}")
    order = [names.index(start)] + [names.index(cut) for cut in cuts]
    if any(later <= earlier for earlier, later in zip(order, order[1:])):
        raise ValueError(f"cuts must follow {start!r} in strictly increasing model order: {cuts}")
    return cuts


def stage_rasters(ee, coefficients, *, config=PRITHVI, namespace="projects/example/assets/prithvi",
                  cuts=None, start="input", source_namespace=None, crops=(1, 1)):
    """Each explicit stage's exported images, built from the previous stage's saved assets.

    Yields ``(stage, [(name, raster), ...])`` in model order, one stage per cut of
    ``stage_cuts(cuts, start=start)``; by default one per saved boundary. A stage computes
    every boundary between the previous cut and its own in one expression, and exports
    only its own: encoder boundaries (patch, blocks, norm) as a spatial and a class image
    (``<stage>_spatial``, ``<stage>_class``), each upsampling and the logits as one. The
    two images of an encoder stage are separate exports, so each recomputes the whole
    stage. Stage inputs are read from ``<namespace>_<name>``; the first stage reads
    ``start``'s from ``<source_namespace>_<name>`` instead when one is given (the crop
    itself is ``_input``), so a stage can be built only once its predecessor's assets
    exist. ``crops`` = (rows, cols) runs a region of whole crops on ``region_grid``: the
    same stages, each boundary on that grid; (1, 1) is the single crop. Builds expressions
    only.
    """
    cuts = stage_cuts(cuts, config=config, start=start)
    input_grid = region_grid(*crops)
    token_grid = input_grid.coarse(16)
    previous = start
    prefix = namespace if source_namespace is None else source_namespace

    def saved(name, grid, channels, window):
        image = saved_image(ee, prefix + "_" + name, band_names(channels), grid)
        return boundary_windows(ee, image, grid, channels, window=window)

    for stage in cuts:
        if previous == "input":
            image = saved_image(ee, prefix + "_input", BANDS, input_grid)
            incoming = Raster(image, BANDS, input_grid)
        elif previous == "patch" or previous.startswith("block") or previous == "norm":
            incoming = (
                saved(previous + "_spatial", token_grid, config.width, 14),
                saved(previous + "_class", token_grid.coarse(14), config.width, 1),
            )
        else:
            side = int(previous[8:])
            grid = input_grid.coarse(224 // side)
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
        prefix = namespace


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
