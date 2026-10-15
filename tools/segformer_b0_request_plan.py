"""Measure real SegFormer request bodies offline, never run or submit the model.

Examples (the lead runs expensive probes sequentially, threads capped at two)::

    PYTHONPATH=. python tools/segformer_b0_request_plan.py --weights inline --inputs inline
    PYTHONPATH=. python tools/segformer_b0_request_plan.py --weights asset-reference --inputs asset-reference

The first mode embeds exact seeded SMP state arrays and concrete representative
boundary literals. The second uses the existing v2 weight provider and explicit
placeholder raster IDs. Their exact bytes depend on these ID strings and the
installed client signatures; no placeholders are live/existing assets. A real
manifest/namespace/config must be remeasured before any submission.

The primary path splits stage 1 after encoder.block1.0, persisting S1B0 before
block1.1 and encoder.norm1. The failed coarse stage 1 and monolithic candidates
remain measurable. Weight-table bodies default to 8 MB; image/compute bodies
retain the 2 MB cap. Split-stage1 verification has 204 requests, coarse has 202.

The report contains sizes, hashes, manifests and provenance, never raw arrays.
It separates computePixels bodies, prepared image/table export bodies, expression
bytes, and response values. Private client preparation helpers serialize the
actual export envelope without constructing Task instances or calling data APIs.
No model forward, GraphEvaluator, upload, asset read, task, credentials or network.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "2"

import ee
import numpy as np
from googleapiclient.model import JsonModel

from tensor_ee._swin_parameters import records_and_locations, with_manifest
from tensor_ee.converter.plan import asset_name
from tensor_ee.converter.request import deep_recursion, export_request, request_size
from tensor_ee.layouts.boundaries import (boundary_raster, boundary_windows, grid_view, pin_boundary,
                                          relabel_from_windows, saved_image)
from tensor_ee.core.ops import WindowOps
from tensor_ee.core.grid import Grid, TokenGrid
from tensor_ee.blocks.window.linear import METADATA as LINEAR_METADATA
from tensor_ee.blocks.window.normalization import METADATA as NORM_METADATA
from tensor_ee.core.tensor import Raster, band_names
from tensor_ee.models.segformer_b0.configuration import COUNTER, SEGFORMER_B0, STAGE_DIMS, constant_costs
from tensor_ee.models.segformer_b0.decoder import decoder, input_windows, model_constants, segformer_b0
from tensor_ee.models.segformer_b0.encoder import block, channel_norm, encoder, overlap_embedding
from tensor_ee.weight_assets import collection_from_records, upload_chunks
from tools.fixtures.core import inline_raster, value_rows
from tools.fixtures.segformer_b0_fullmodel import (
    SEED, array_provenance, reject_negative_zero, representative_boundaries, seeded_state, state_sha256,
)
from tools.live_fixtures import npy_header_length, offline_earth_engine

WIRE_LIMIT = 2_000_000
SERVICE_PAYLOAD_LIMIT = 10_000_000
MAX_ARRAY_VALUES = 1024 * 256
PLACEHOLDER_NAMESPACE = "projects/example-project/assets/segformer_b0_request_plan"
REQUEST_ID = "00000000-0000-4000-8000-000000000000"
CUTS = ("monolithic", "stage1", "stage1a", "stage1b", "stage2", "stage3", "stage4", "decoder")
COARSE_CUTS = ("stage1", "stage2", "stage3", "stage4", "decoder")
SPLIT_STAGE1_CUTS = ("stage1a", "stage1b", "stage2", "stage3", "stage4", "decoder")
WEIGHT_WIRE_LIMIT = 8_000_000
SOURCE_GRID = Grid("EPSG:3857", (2.5, 0, 1000, 0, -2.5, 2000), 224, 224)


deep_serialization = deep_recursion
"""The approved capstone-style allowance (a recursion limit of at least 20,000), restored on exit:
the library's ``tensor_ee.converter.request.deep_recursion`` (X08), kept under this name for its
callers."""


def _fingerprint(body):
    # Validate JSON finiteness before invoking the installed SDK's serializer.
    json.dumps(body, allow_nan=False)
    wire = JsonModel().serialize(body).encode("utf-8")
    compact = json.dumps(body, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return {"bytes": len(wire), "sha256": hashlib.sha256(wire).hexdigest(),
            "compact_bytes": len(compact), "compact_sha256": hashlib.sha256(compact).hexdigest()}


def _encode_config(config):
    """Match data.py's expression substitution without invoking its transport (the table
    exports; an image export is the library's ``export_request``)."""
    request = dict(config)
    with deep_serialization():
        request["expression"] = ee.serializer.encode(request["expression"], for_cloud_api=True)
    request["requestId"] = REQUEST_ID
    return request


def _table_request(rows, namespace, index):
    collection = collection_from_records(rows)
    # Pure client preparation: no Export.table.toAsset or Task creation.
    config = ee.batch._prepare_table_export_config(
        collection, {"description": f"native_unet_weights_part_{index:04d}",
                     "assetId": f"{namespace}_part_{index:04d}", "overwrite": False},
        ee.batch.Task.ExportDestination.ASSET)
    return _encode_config(config)


def plan_weight_shards(records, namespace, max_wire_bytes=WIRE_LIMIT):
    """Reuse upload_chunks, then enforce the *complete* exportTable wire limit.

    upload_chunks limits collection expressions only. A measured over-limit
    envelope is split at a row boundary and measured again; no row is omitted.
    Destinations use the eventual sequential shard index, never a guessed size.
    """
    if not 100_000 <= max_wire_bytes <= 8_000_000:
        raise ValueError("weight wire limit must be between 100,000 and 8,000,000")
    shards = []

    def append(rows):
        index = len(shards)
        body = _table_request(rows, namespace, index)
        measured = _fingerprint(body)
        if measured["bytes"] > max_wire_bytes:
            if len(rows) < 2:
                raise ValueError("one weight row exceeds the complete exportTable wire limit")
            center = len(rows) // 2
            append(rows[:center])
            append(rows[center:])
            return
        expression = _fingerprint(body["expression"])
        shards.append({"index": index, "asset_id": f"{namespace}_part_{index:04d}",
                       "rows": len(rows), "layers": sorted({r["properties"]["layer"] for r in rows}),
                       "collection_payload_bytes": expression["bytes"], "export_table": measured,
                       "within_wire_limit": True})

    for rows, _collection_bytes in upload_chunks(records, max_bytes=max_wire_bytes):
        append(rows)
    return shards


def planned_manifest(records, shards, namespace, state_digest):
    """A provider-compatible placeholder manifest, explicitly not uploaded."""
    layers = {r.key: {"shape": list(r.shape), "dtype": r.dtype, "bias": r.bias is not None,
                      "source": r.source, "sha256": r.sha256, "assets": []} for r in records}
    for shard in shards:
        for key in shard["layers"]:
            layers[key]["assets"].append(shard["asset_id"])
    listing = json.dumps(sorted((r.key, r.sha256) for r in records), separators=(",", ":"))
    return {"format": "native_ee_featurecollection_weights_v2", "status": "ready",
            "planning_only": True, "assets_exist": False, "asset_namespace": namespace,
            "source_checkpoint_sha256": state_digest,
            "manifest_hash": hashlib.sha256(listing.encode()).hexdigest(), "layers": layers,
            "shards": [dict(shard, status="planned_not_submitted") for shard in shards]}


def asset_state(state, manifest, digest):
    """Reuse the existing generic v2 record/location transport, preserving names."""
    computational = {name: value for name, value in state.items() if name != COUNTER}
    server = with_manifest({}, {}, computational, manifest, digest)[2]
    return server | {COUNTER: state[COUNTER]}


def _boundary_spec(name):
    if name == "input":
        return SOURCE_GRID, 2, ("b0", "b1")
    if name == "S1B0":
        return SEGFORMER_B0.stage_grids[0], STAGE_DIMS[0], tuple(band_names(STAGE_DIMS[0]))
    s = int(name[1:])
    return SEGFORMER_B0.stage_grids[s - 1], STAGE_DIMS[s - 1], tuple(band_names(STAGE_DIMS[s - 1]))


def _boundary_image(name, source, arrays, namespace):
    # The boundary conventions (saved_image, pin_boundary, grid_view, boundary_windows and
    # boundary_raster) moved to tensor_ee/layouts/boundaries.py (X02a); the asset names and
    # literals stay here.
    grid, channels, names = _boundary_spec(name)
    if source == "asset-reference":
        return saved_image(ee, f"{namespace}_{name}", names, grid)
    # Bound literal-band construction depth, without repeating boundary
    # values or shrinking their IEEE precision to manufacture a small body.
    values = arrays[name]
    view = _grid_view(grid)
    parts = [inline_raster(ee, view.projection(ee), values[:, :, first:first + 32], names[first:first + 32])
             for first in range(0, channels, 32)]
    return pin_boundary(ee, ee.Image.cat(parts), grid)


_grid_view = grid_view


def _boundary_window(name, source, arrays, namespace):
    grid, channels, _ = _boundary_spec(name)
    return boundary_windows(ee, _boundary_image(name, source, arrays, namespace), grid, channels, window=7)


def build_cut(cut, constants, source, arrays, namespace):
    """Native stage boundaries, no whole-model local numerical evaluation."""
    ops = WindowOps(ee)
    if cut == "monolithic":
        raster = Raster(_boundary_image("input", source, arrays, namespace), ("b0", "b1"), SOURCE_GRID)
        return segformer_b0(ee, raster, constants)["logits"], ["input"], "logits"
    if cut in ("stage1a", "stage1b"):
        c = constants.encoder
        if cut == "stage1a":
            raster = Raster(_boundary_image("input", source, arrays, namespace), ("b0", "b1"), SOURCE_GRID)
            x = input_windows(ee, raster)
            x = overlap_embedding(ops, x, c.embeddings[0])["out"]
            out = block(ops, x, c.blocks[0][0])["out"]
            required, destination = ["input"], "S1B0"
        else:
            x = _boundary_window("S1B0", source, arrays, namespace)
            x = block(ops, x, c.blocks[0][1])["out"]
            out = channel_norm(ops, x, c.norms[0])
            required, destination = ["S1B0"], "C1"
        return boundary_raster(ee, out), required, destination
    if cut.startswith("stage"):
        stage = int(cut[5:])
        if stage == 1:
            name = "input"
            raster = Raster(_boundary_image(name, source, arrays, namespace), ("b0", "b1"), SOURCE_GRID)
            x = input_windows(ee, raster)
        else:
            name = f"C{stage - 1}"
            x = _boundary_window(name, source, arrays, namespace)
        out = encoder(ops, x, constants.encoder, start=stage, stop=stage)[f"C{stage}"]
        return boundary_raster(ee, out), [name], f"C{stage}"
    if cut == "decoder":
        names = [f"C{s}" for s in range(1, 5)]
        features = [_boundary_window(name, source, arrays, namespace) for name in names]
        out = decoder(ops, features, constants.decoder)["out"]
        return relabel_from_windows(ee, out, SOURCE_GRID, ("c0", "c1")), names, "logits"
    raise ValueError(f"unknown cut {cut}")


def _with_validity(image, names):
    values = image.select(list(names)).rename(list(names))
    validity = values.mask().reduce(ee.Reducer.min()).rename("valid").unmask(0, False).toFloat()
    return values.unmask(0, False).addBands(validity), list(names) + ["valid"]


def measure_compute_pixels(image, grid, names):
    """Exact offline NPY request, including validity; no submission."""
    grid = _grid_view(grid)
    image, bands = _with_validity(image, names)
    with deep_serialization():
        expression = ee.serializer.encode(image, for_cloud_api=True)
        graph_json = image.serialize()
    body = {"expression": expression, "fileFormat": "NPY", "grid": grid.pixel_grid()}
    wire = _fingerprint(body)
    values = grid.height * grid.width * len(bands)
    header = npy_header_length(bands, grid.height, grid.width)
    violations = []
    for condition, reason in ((wire["bytes"] > WIRE_LIMIT, "request exceeds 2,000,000-byte fixture budget"),
                              (grid.height > 56 or grid.width > 56, "grid exceeds W7 fixture 56x56-side budget"),
                              (values > 1_000_000, "response exceeds 1,000,000 values"),
                              (header > 9000, "NPY header exceeds 9,000 bytes"),
                              (len(bands) > 1024, "response exceeds 1,024 bands")):
        if condition:
            violations.append(reason)
    return {"wire": wire, "expression": _fingerprint(expression),
            "image_serialize": {"bytes": len(graph_json.encode("utf-8")),
                                "sha256": hashlib.sha256(graph_json.encode("utf-8")).hexdigest()},
            "grid": grid.pixel_grid(), "band_names": bands,
            "bands": len(bands), "values": values, "float32_response_bytes": 4 * values,
            "npy_header_bytes": header, "standing_fixture_violations": violations,
            "within_2mb": wire["bytes"] <= WIRE_LIMIT,
            "within_service_payload": wire["bytes"] <= SERVICE_PAYLOAD_LIMIT,
            "expanded_verification_limits_pass": wire["bytes"] <= WIRE_LIMIT and values <= 1_000_000 and
            header <= 9000 and len(bands) <= 1024}


def measure_image_export(image, grid, namespace, destination):
    """Actual client-prepared ExportImageRequest; no Task object or data API.

    Since X08 the request is the library's (``tensor_ee.converter.request``): ``export_request``
    builds it, to ``f"{namespace}_{destination}"`` described as ``f"segformer_b0_{destination}"``
    with ``REQUEST_ID``, and its bytes are the library's request sizer, ``request_size``, which
    this tool calls (one implementation; the planner's default measurement). A destination with
    ``.`` (a block graph's module-path edge id, as X04's Swin plans pass) is not an asset name, and
    the library refuses it; it is measured under the planner's ``asset_name``, the same-length name
    with ``_`` for ``.``, as the planner's default measurement does, so the bytes are unchanged."""
    grid = _grid_view(grid)
    name = asset_name(destination)
    target = {"asset_id": f"{namespace}_{name}", "description": f"segformer_b0_{name}", "request_id": REQUEST_ID}
    body = export_request(image, grid, **target)
    wire = _fingerprint(body)
    size = request_size(image, grid, **target)
    if wire["bytes"] != size:
        raise AssertionError(f"the installed serializer gives {wire['bytes']} bytes, the library's sizer {size}")
    return {"wire": wire, "expression": _fingerprint(body["expression"]),
            "grid": grid.pixel_grid(), "pyramiding_policy": "sample", "overwrite": False,
            "within_2mb": wire["bytes"] <= WIRE_LIMIT,
            "within_service_payload": wire["bytes"] <= SERVICE_PAYLOAD_LIMIT}


def array_segments(values, max_values=MAX_ARRAY_VALUES):
    """Flat contiguous coverage, each segment <=1024 rows x <=256 columns."""
    if isinstance(values, bool) or not isinstance(values, (int, np.integer)) or values < 1:
        raise ValueError("array value count must be positive integer")
    if isinstance(max_values, bool) or not isinstance(max_values, int) or not 1 <= max_values <= MAX_ARRAY_VALUES:
        raise ValueError("segment maximum must be positive integer <=262144")
    result = []
    start = 0
    while start < values:
        count = min(max_values, values - start)
        columns = next((c for c in range(min(256, count), 0, -1) if count % c == 0 and count // c <= 1024), None)
        if columns is None:
            raise ValueError(f"no <=1024x256 view covers {count} values")
        result.append({"start": start, "end": start + count, "rows": count // columns, "columns": columns})
        start += count
    assert result[0]["start"] == 0 and result[-1]["end"] == values
    assert all(a["end"] == b["start"] for a, b in zip(result, result[1:]))
    assert sum(v["end"] - v["start"] for v in result) == values
    return result


def primary_cuts(primary_plan):
    if primary_plan not in ("coarse", "split-stage1"):
        raise ValueError("primary plan must be coarse or split-stage1")
    return COARSE_CUTS if primary_plan == "coarse" else SPLIT_STAGE1_CUTS


def primary_producers(primary_plan):
    result = {f"C{i}": f"stage{i}" for i in range(1, 5)}
    if primary_plan == "split-stage1":
        result.update(S1B0="stage1a", C1="stage1b")
    primary_cuts(primary_plan)
    return result


def cut_constant_costs(cut):
    """Reachable evaluated constants/source arrays from actual block METADATA."""
    costs = constant_costs()
    if cut == "monolithic":
        return costs["total"]
    if cut == "decoder":
        return {key: costs["decoder_head"][key] + costs["resize"][key] for key in ("values", "asset_arrays")}
    if cut in ("stage1a", "stage1b"):
        layout = SEGFORMER_B0.layout
        lin, norm = LINEAR_METADATA["linear"], NORM_METADATA["layer_norm"]
        projection = {"values": lin.embedded_constants(98, 32, bias=True, layout=layout),
                      "asset_arrays": lin.asset_arrays(bias=True)}
        affine = {"values": norm.embedded_constants(32, layout=layout), "asset_arrays": norm.asset_arrays()}
        # Original stage = projection + patch norm + two blocks + final norm.
        block_cost = {}
        for key in ("values", "asset_arrays"):
            twice_block = costs["encoder_stage1"][key] - projection[key] - 2 * affine[key]
            assert twice_block % 2 == 0
            block_cost[key] = twice_block // 2
        return {key: block_cost[key] + affine[key] + (projection[key] if cut == "stage1a" else 0)
                for key in ("values", "asset_arrays")}
    return costs[f"encoder_stage{int(cut[5:])}"]


def verification_plan(state, server_state, namespace, stage_producers, primary_plan="split-stage1"):
    """Source coverage plus direct/saved model boundaries on the chosen path."""
    requests = []
    oversized = []
    for name, expected in state.items():
        if name == COUNTER:
            continue
        if expected.size > MAX_ARRAY_VALUES:
            oversized.append(name)
        flat = server_state[name].array.reshape([int(expected.size), 1])
        for segment in array_segments(expected.size):
            array = flat.slice(0, segment["start"], segment["end"]).reshape([segment["rows"], segment["columns"]])
            image, pixel_grid = value_rows(ee, array, segment["rows"], segment["columns"])
            grid = TokenGrid(segment["rows"], 1)
            assert grid.pixel_grid() == pixel_grid
            measured = measure_compute_pixels(image, grid, band_names(segment["columns"]))
            requests.append({"kind": "coefficient", "name": name, "original_shape": list(expected.shape),
                             "comparison": "IEEE float32 bits, reconstructed flat original order",
                             "segment": segment, "measurement": measured})
    if oversized != ["encoder.patch_embed4.proj.weight"]:
        raise AssertionError(f"fixed reference oversized-array census differs: {oversized}")
    coefficient_requests = len(requests)
    image_shapes = ["input"] + (["S1B0"] if primary_plan == "split-stage1" else []) + ["C1", "C2", "C3", "C4", "logits"]
    required_producers = primary_producers(primary_plan)
    assert set(stage_producers) == set(required_producers)
    for name in image_shapes:
        grid, channels, bands = ((SOURCE_GRID, 2, ("c0", "c1")) if name == "logits"
                                else _boundary_spec(name))
        grid = _grid_view(grid)
        image = ee.Image(f"{namespace}_{name}").select(list(bands)).reproject(grid.projection(ee))
        if name in required_producers:
            producer = stage_producers[name]
            requests.append({"kind": "stage_producer", "name": name,
                             "shape": [grid.height, grid.width, channels], "requires": producer["requires"],
                             "comparison": "unchanged 1e-4 absolute target versus PyTorch; saved asset must match recorded producer bits",
                             "failure_policy": "interactive producer failure stops; no automatic fallback",
                             "reference_hook": "encoder.block1.0 complete output before block1.1"
                             if name == "S1B0" else f"encoder.norm{name[1:]}",
                             "measurement": producer["compute_pixels"]})
        requests.append({"kind": "saved_image", "name": name, "shape": [grid.height, grid.width, channels],
                         "comparison": "input: IEEE float32 bits; saved model boundaries: recorded producer bits; logits: 1e-4 absolute target",
                         "measurement": measure_compute_pixels(image, grid, bands)})
    return {"requests": requests, "coefficient_requests": coefficient_requests,
            "direct_stage_producer_requests": len(stage_producers), "saved_image_requests": len(image_shapes),
            "image_requests": len(image_shapes) + len(stage_producers),
            "total_requests": len(requests), "primary_plan": primary_plan, "computational_source_arrays": len(state) - 1,
            "covered_source_values": sum(v.size for name, v in state.items() if name != COUNTER),
            "all_expanded_limits_pass": all(r["measurement"]["expanded_verification_limits_pass"] for r in requests),
            "approval": "Expanded full-model verification plan; not the standing 40-request/small-grid fixture budget"}


def measure(*, seed=SEED, weights="both", inputs="both", cuts=CUTS,
            namespace=PLACEHOLDER_NAMESPACE, weight_wire_limit=WEIGHT_WIRE_LIMIT, verification=True,
            primary_plan="split-stage1"):
    """Heavy offline probe. The lead runs it; there is deliberately no live mode."""
    if weights not in ("inline", "asset-reference", "both") or inputs not in ("inline", "asset-reference", "both"):
        raise ValueError("sources must be inline, asset-reference or both")
    if any(cut not in CUTS for cut in cuts) or not cuts:
        raise ValueError(f"cuts must be a nonempty subset of {CUTS}")
    if not namespace.startswith("projects/") or "/assets/" not in namespace:
        raise ValueError("placeholder namespace must have projects/PROJECT/assets/PREFIX form")
    planned_cuts = primary_cuts(primary_plan)
    producer_cuts = primary_producers(primary_plan)
    with offline_earth_engine() as attempts:
        state = seeded_state(seed)
        boundaries = representative_boundaries(seed)
        reject_negative_zero(state)
        reject_negative_zero(boundaries)
        computational = {name: value for name, value in state.items() if name != COUNTER}
        records, _locations = records_and_locations({}, {}, computational)
        record_list = list(records.values())
        digest = state_sha256(state)
        shards = plan_weight_shards(record_list, namespace + "_weights", weight_wire_limit)
        manifest = planned_manifest(record_list, shards, namespace + "_weights", digest)
        server = asset_state(state, manifest, digest)
        candidates = []
        stage_producers = {}
        weight_modes = ("inline", "asset-reference") if weights == "both" else (weights,)
        input_modes = ("inline", "asset-reference") if inputs == "both" else (inputs,)
        for weight_source in weight_modes:
            constants = model_constants(ee, state if weight_source == "inline" else server)
            for input_source in input_modes:
                for cut in cuts:
                    output, required, destination = build_cut(cut, constants, input_source, boundaries, namespace)
                    candidate = {"cut": cut, "weight_source": weight_source, "boundary_source": input_source,
                                       "requires": required, "destination": destination,
                                       "input_provenance": "representative finite float32 literals, not evaluated EE outputs"
                                       if input_source == "inline" else "placeholder image references; values not read",
                                       "shape": [output.grid.height, output.grid.width, len(output.bands)],
                                       "constant_costs": cut_constant_costs(cut),
                                       "compute_pixels": measure_compute_pixels(output.image, output.grid, output.bands),
                                       "image_export": measure_image_export(output.image, output.grid, namespace, destination)}
                    candidates.append(candidate)
                    if weight_source == input_source == "asset-reference" and producer_cuts.get(destination) == cut:
                        stage_producers[destination] = candidate
                    del output
                    gc.collect()
            del constants
        # The input export embeds actual deterministic b0/b1 data, unlike a
        # placeholder boundary. No input asset is created or read by this tool.
        image = _boundary_image("input", "inline", boundaries, namespace)
        input_export = measure_image_export(image, SOURCE_GRID, namespace, "input")
        if verification and set(stage_producers) != set(producer_cuts):
            constants = model_constants(ee, server)
            for name, cut in producer_cuts.items():
                if name not in stage_producers:
                    output, required, _ = build_cut(cut, constants, "asset-reference", boundaries, namespace)
                    stage_producers[name] = {"requires": required,
                                            "compute_pixels": measure_compute_pixels(output.image, output.grid, output.bands)}
            del constants
        readbacks = verification_plan(state, server, namespace, stage_producers, primary_plan) if verification else None
    if attempts:
        raise AssertionError(f"refused cloud attempts: {attempts}")
    asset_stages = [r for r in candidates if r["weight_source"] == r["boundary_source"] == "asset-reference"
                    and r["cut"] != "monolithic"]
    needs_finer = [r["cut"] for r in asset_stages if not r["image_export"]["within_2mb"] or not r["compute_pixels"]["within_2mb"]]
    primary_asset_requests = [r for r in asset_stages if r["cut"] in planned_cuts]
    primary_failures = [r["cut"] for r in primary_asset_requests
                        if not r["image_export"]["within_2mb"] or not r["compute_pixels"]["within_2mb"]]
    all_asset_cuts = {r["cut"] for r in primary_asset_requests} == set(planned_cuts)
    return {"schema_version": 1, "kind": "segformer_b0_fullmodel_offline_request_plan",
            "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
            "versions": {p: importlib.metadata.version(p) for p in
                         ("earthengine-api", "numpy", "torch", "timm", "segmentation-models-pytorch")},
            "python": sys.version.split()[0], "platform": platform.platform(), "seed": seed,
            "constructor": "smp.Segformer('mit_b0',encoder_weights=None,in_channels=2,classes=2).eval()",
            "initialization": "exact seeded constructor; no perturbation, quantization or replaced parameters",
            "state_sha256": digest, "state": array_provenance(state),
            "boundary_arrays": array_provenance(boundaries),
            "boundary_generator": "tools.fixtures.core.dyadic / SplitMix64, distinct named streams",
            "boundary_limitation": "Boundary literals are representative sizing data, not evaluated model features",
            "placeholder_namespace": namespace,
            "placeholder_limitation": "Exact measurements for these IDs; conditional until actual IDs/config/manifest are remeasured",
            "wire_encoding": "installed googleapiclient.model.JsonModel.serialize UTF-8; compact bytes/hash also recorded; 36-character dummy requestId included in exports",
            "wire_scope": "Cloud API body; excludes project URL and optional transport-added workloadTag",
            "serialization_recursion_limit": 20000, "recursion_limit_restored": True,
            "source_grid": _grid_view(SOURCE_GRID).pixel_grid(),
            "negative_zero_preflight": "rejected, never canonicalized or perturbed",
            "constant_costs": constant_costs(), "weight_manifest": manifest,
            "weight_records": len(records), "weight_shards": len(shards),
            "weight_feature_rows": sum(s["rows"] for s in shards), "weight_wire_limit": weight_wire_limit,
            "input_export": input_export, "candidates": candidates, "verification": readbacks,
            "primary_plan": primary_plan, "primary_cuts": list(planned_cuts),
            "proposed_primary_assets_and_batch_submissions": len(shards) + 1 + len(planned_cuts),
            "primary_count_scope": "K weight-table shards + one deterministic input image + each primary cut output",
            "primary_wire_limits_pass": all_asset_cuts and input_export["within_2mb"] and not primary_failures,
            "primary_image_compute_wire_limit": WIRE_LIMIT,
            "primary_cuts_exceeding_image_compute_wire_limit": primary_failures,
            "input_export_requires_budget_or_transport_revision": not input_export["within_2mb"],
            "finer_cuts_required_by_measured_asset_requests": needs_finer,
            "asset_coarse_cuts_measured": [r["cut"] for r in asset_stages if r["cut"] in COARSE_CUTS],
            "asset_cuts_measured": [r["cut"] for r in asset_stages],
            "all_five_asset_coarse_cuts_measured": set(COARSE_CUTS).issubset({r["cut"] for r in asset_stages}),
            "all_primary_asset_cuts_measured": all_asset_cuts,
            "finer_cut_capabilities": "existing overlap_embedding, block and channel_norm boundaries; no spatial model tiling",
            "model_activations_evaluated": False, "cloud_calls": 0, "assets_created": 0, "tasks_created": 0,
            "accuracy_claim": False, "live_capacity_claim": False, "logit_absolute_target": 1e-4}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--weights", choices=("inline", "asset-reference", "both"), default="both")
    parser.add_argument("--inputs", choices=("inline", "asset-reference", "both"), default="both")
    parser.add_argument("--cuts", default=",".join(CUTS), help="comma-separated monolithic,stage1,stage1a,stage1b,stage2..stage4,decoder")
    parser.add_argument("--namespace", default=PLACEHOLDER_NAMESPACE, help="placeholder ID prefix; no asset is accessed")
    parser.add_argument("--weight-wire-limit", type=int, default=WEIGHT_WIRE_LIMIT)
    parser.add_argument("--primary-plan", choices=("coarse", "split-stage1"), default="split-stage1")
    parser.add_argument("--no-verification", action="store_true", help="omit expanded verification (204 split-stage1 / 202 coarse requests)")
    parser.add_argument("--output", type=Path, help="save metadata/measurements JSON; never raw arrays")
    args = parser.parse_args()
    result = measure(seed=args.seed, weights=args.weights, inputs=args.inputs,
                     cuts=tuple(args.cuts.split(",")), namespace=args.namespace,
                     weight_wire_limit=args.weight_wire_limit, verification=not args.no_verification,
                     primary_plan=args.primary_plan)
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
