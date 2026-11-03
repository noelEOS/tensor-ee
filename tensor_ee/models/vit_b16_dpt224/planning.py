"""Offline measured hand cuts for DPT; no assets, tasks or cloud calls are created.

Encoder cuts reuse the merged encoder planner, stopping before its unused norm.
Four branches save reassembled features, then four fusion stages and one head.
This is an explicit proposal, not the converter optimizer or an execution claim.
"""
from __future__ import annotations

from collections.abc import Mapping
import math

from ...blocks.window.conv import conv3x3_values
from ...blocks.window.linear import linear_values
from ...blocks.window.normalization import batch_norm_values
from ...blocks.window.resample import window_resize_values
from ...converter.plan import Limits
from ...converter.request import deep_recursion, request_size
from ...core.tensor import ServerArray, band_names
from ...layouts.boundaries import boundary_raster, boundary_windows, saved_image
from ..vit_b16_encoder224 import model as encoder
from ..vit_b16_encoder224 import planning as encoder_plan
from .configuration import DPT_B16, boundary_spec, parameter_shapes
from .model import INPUT_GRID, TOKEN_GRID, constants, readout, reassemble, fusion, head


def decoder_costs(config=DPT_B16):
    """Per-cut constants from existing block cost functions, not state size alone."""
    dim, width = config.encoder.width, config.fusion
    costs = {}
    for i, mid in enumerate(config.intermediate):
        rows = (49, 49, 196, 49)[i]
        cost = linear_values(2 * dim, dim, layout=196) + linear_values(dim, mid, layout=196)
        if i < 2:
            cost += linear_values(mid, (16, 4)[i] * mid, layout=196)
        elif i == 3:
            cost += conv3x3_values(mid, mid, bias=True, layout=49)
        costs[f"reassemble{i}"] = cost + conv3x3_values(mid, width, layout=rows)
    for i, side in enumerate((7, 14, 28, 56)):
        window, out_window = (7, 14, 7, 7)[i], (14, 7, None, None)[i]
        output_rows = (window if out_window is None else out_window) ** 2
        costs[f"fusion{i}"] = (4 * conv3x3_values(width, width, layout=window ** 2)
            + 4 * batch_norm_values(width, layout=window ** 2)
            + window_resize_values(side, side, 2, align_corners=True, window=window, out_window=out_window)
            + linear_values(width, width, layout=output_rows))
    costs["logits"] = (conv3x3_values(width, width, layout=49) + batch_norm_values(width, layout=49)
        + linear_values(width, 2, layout=49)
        + window_resize_values(112, 112, 2, align_corners=True, window=7))
    return costs


def stage_rasters(ee, coefficients, *, namespace):
    """Yield (cut, saved inputs, output rasters), matching the measured proposal."""
    config, backend = coefficients.config, coefficients.backend
    cuts = ("patch", *(f"block{i}" for i in range(config.encoder.depth)))
    enc = encoder.ViTConstants(config.encoder, backend)
    previous = "input"
    for stop, images in encoder_plan.stage_rasters(ee, enc, namespace=namespace, cuts=cuts):
        inputs = ("input",) if previous == "input" else (previous + "_spatial", previous + "_class")
        yield stop, inputs, images
        previous = stop

    def saved(name, side, channels, window):
        grid = INPUT_GRID.coarse(224 // side)
        image = saved_image(ee, namespace + "_" + name, band_names(channels), grid)
        return boundary_windows(ee, image, grid, channels, window=window)

    def boundary(name):
        return saved(name, *boundary_spec(name, config))

    for i, index in enumerate(config.indices):
        stem = f"block{index}"
        names = (stem + "_spatial", stem + "_class")
        pair = (saved(names[0], 14, config.encoder.width, 14),
                saved(names[1], 1, config.encoder.width, 1))
        name = f"reassemble{i}"
        output = reassemble(backend, readout(backend, pair, i), i)
        yield name, names, [(name, boundary_raster(ee, output))]
    for i in range(4):
        feature = f"reassemble{3-i}"
        previous = None if i == 0 else boundary(f"fusion{i-1}")
        inputs = (feature,) if i == 0 else (feature, f"fusion{i-1}")
        name = f"fusion{i}"
        output = fusion(backend, boundary(feature), previous, i)
        yield name, inputs, [(name, boundary_raster(ee, output))]
    output = backend.dense(backend.resize(head(backend, boundary("fusion3"))))
    yield "logits", ("fusion3",), [("logits", output)]


def measured_cuts(ee, parameters, *, config=DPT_B16,
                  namespace="projects/example/assets/vit_b16_dpt224", limits=None):
    """Measure table-backed export payloads and saved bands for every explicit cut.

    Exact sizes depend on the supplied ServerArray expressions; real uploaded
    manifests must be remeasured. The returned tasks/assets describe a future run.
    Weight uploads, input upload and comparison readbacks are listed separately
    and require a concrete manifest and runner before approval can be requested.
    """
    shapes = parameter_shapes(config)
    if not isinstance(parameters, Mapping) or not all(isinstance(parameters.get(n), ServerArray) for n in shapes):
        raise ValueError("request sizing requires ServerArray DPT parameters")
    limits = Limits() if limits is None else limits
    enc_cost = encoder_plan.constant_costs(config.encoder)
    costs = decoder_costs(config)
    records = []
    with deep_recursion():
        coefficients = constants(ee, parameters, config=config)
        for stop, inputs, images in stage_rasters(ee, coefficients, namespace=namespace):
            is_block = stop.startswith("block")
            logical = enc_cost["stages"][stop] if stop == "patch" or is_block else costs[stop]
            conservative = enc_cost["encoder_block_conservative"] if is_block else logical
            sizes = {name: request_size(raster.image, raster.grid, asset_id=namespace + "_" + name,
                                        description="vit_b16_dpt224_" + name) for name, raster in images}
            bands = [len(raster.bands) for _, raster in images]
            records.append({"stop": stop, "inputs": list(inputs), "outputs": list(sizes),
                "constants": logical, "conservative_constants": conservative,
                "boundary_bands": bands, "request_bytes": sizes, "exports": len(images),
                "extent_nodes": int(is_block), "tileable": False,
                "within_limits": conservative <= limits.constants and max(bands) <= limits.bands
                    and max(sizes.values()) <= limits.request_bytes
                    and (limits.extent is None or int(is_block) <= limits.extent)})
    return {"kind": "explicit_measured_dpt_cuts", "submitted": False,
        "limits": {"constants": limits.constants, "bands": limits.bands,
                   "request_bytes": limits.request_bytes, "extent": limits.extent},
        "state_values": sum(math.prod(shape) for shape in shapes.values()),
        "logical_constants": sum(stage["constants"] for stage in records),
        "stages": records, "inference_tasks": sum(stage["exports"] for stage in records),
        "inference_assets": sum(stage["exports"] for stage in records),
        "weight_table_assets": len(shapes), "input_assets": 1,
        "weight_upload_requests": None, "input_upload_requests": None, "readback_requests": None,
        "within_limits": all(stage["within_limits"] for stage in records),
        "runtime_measured": False, "eecu_seconds": None,
        "compute_estimate": "approximately 40 G MACs per crop; not an EECU estimate"}
