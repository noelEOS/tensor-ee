"""Prepare and execute only the approved, bounded SegFormer-B0 full-model plan.

The maintainer invokes prepare (graph serialization), preflight (local references), and
run (ADC/live). Imports perform none of those actions. All raw arrays, immutable
request envelopes and the write-ahead ledger live below ignored release_runs/.
A stopped run is never automatically resumed, retried, cancelled or repacked.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import uuid

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_variable] = "2"

import ee
import numpy as np

from tools.segformer_b0_live_guard import Guard, CAPS, BudgetStop, fingerprint, wire

ROOT = Path(__file__).resolve().parents[1]
CUTS = ("stage1a", "stage1b", "stage2", "stage3", "stage4", "decoder")
OUTPUTS = ("S1B0", "C1", "C2", "C3", "C4", "logits")
SHAPES = {"input": (224, 224, 2), "S1B0": (56, 56, 32), "C1": (56, 56, 32),
          "C2": (28, 28, 64), "C3": (14, 14, 160), "C4": (7, 7, 256), "logits": (224, 224, 2)}
TOLERANCE = 1e-4
APPROVED_PLAN_SHA256 = "4b1820bd17001ac80d4e6189a83d97f747f3bf90ff1e604aae456de558c7b34d"


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def save_json(path, value):
    """Atomic private evidence write; retain dictionary order for wire fingerprints."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def read_json(path):
    return json.loads(Path(path).read_text())


def private_directory(directory):
    path = Path(directory).resolve()
    if ROOT / "release_runs" not in path.parents:
        raise ValueError("private state must be a subdirectory of ignored release_runs")
    return path


def versions():
    return {p: importlib.metadata.version(p) for p in
            ("earthengine-api", "numpy", "torch", "segmentation-models-pytorch")}


def provenance():
    return {"revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
            "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
            "command": list(sys.argv)}


def source_hashes():
    paths = sorted((ROOT / "tensor_ee").rglob("*.py"))
    paths += [ROOT / p for p in ("tools/segformer_b0_validate.py", "tools/segformer_b0_live_guard.py",
        "tools/segformer_b0_request_plan.py", "tools/fixtures/core.py", "tools/fixtures/segformer_b0_fullmodel.py",
        "tools/fixtures/segformer_b0_validation.py", "tests/reference/segformer_b0/full_model.py",
        "tests/reference/segformer_b0/model.py", "tests/reference/segformer_b0/blocks.py")]
    return {str(p.relative_to(ROOT)): sha_file(p) for p in paths}


def load_npz(path):
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name].copy() for name in source.files}


def check_hwc(value, shape):
    if not isinstance(value, np.ndarray) or value.dtype != np.float32 or value.shape != tuple(shape) or not np.isfinite(value).all():
        raise ValueError(f"expected finite float32 array with shape {shape}")


def compare(actual, expected, *, exact=False):
    """Fixed denominator; failures never discard missing/nonfinite pixels."""
    check_hwc(expected, expected.shape)
    if not isinstance(actual, np.ndarray) or actual.dtype != np.float32 or actual.shape != expected.shape:
        raise ValidationFailure("numerical comparison shape/dtype differs", {"expected_values": int(expected.size),
            "expected_shape": list(expected.shape), "actual_shape": list(getattr(actual, "shape", ()))})
    if not np.isfinite(actual).all():
        raise ValidationFailure("nonfinite numerical comparison", {"expected_values": int(expected.size),
            "nonfinite": int(np.count_nonzero(~np.isfinite(actual)))})
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    absolute = np.abs(difference)
    worst = tuple(int(i) for i in np.unravel_index(np.argmax(absolute), expected.shape))
    bits = actual.view(np.uint32) != expected.view(np.uint32)
    result = {"values": int(expected.size), "max_abs": float(absolute.max()),
              "mae": float(absolute.mean()), "rmse": float(np.sqrt(np.mean(difference * difference))),
              "over_target": int(np.count_nonzero(absolute > TOLERANCE)), "nonfinite": 0,
              "worst": list(worst), "actual_worst": float(actual[worst]), "expected_worst": float(expected[worst]),
              "differing_bits": int(np.count_nonzero(bits)), "actual_sha256": array_sha(actual),
              "expected_sha256": array_sha(expected), "target": 0 if exact else TOLERANCE,
              "passed": bool(not bits.any() if exact else absolute.max() <= TOLERANCE)}
    if not result["passed"]:
        raise ValidationFailure("exact transport mismatch" if exact else "largest absolute error exceeds1e-4", result)
    return result


class ValidationFailure(RuntimeError):
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details


def read_pixels(raw, spec):
    """Decode NPY without pickle; fail missing fields, masks, shape or precision."""
    pixels = np.load(io.BytesIO(raw), allow_pickle=False, max_header_size=9000) if isinstance(raw, bytes) else raw
    d = spec["body"]["grid"]["dimensions"]
    shape = (d["height"], d["width"])
    bands = spec["bands"]
    if not isinstance(pixels, np.ndarray) or pixels.shape != shape or tuple(pixels.dtype.names or ()) != tuple(bands):
        raise ValidationFailure("response shape/ordered bands differs from prepared read")
    if any(pixels.dtype.fields[b][0] != np.dtype(np.float32) for b in bands):
        raise ValidationFailure("response is not float32")
    validity = pixels["valid"]
    if not np.isfinite(validity).all() or not np.all(validity == 1):
        raise ValidationFailure("missing/invalid pixels", {"expected_pixels": int(validity.size),
            "invalid_pixels": int(np.count_nonzero(validity != 1)), "nonfinite_validity": int(np.count_nonzero(~np.isfinite(validity)))})
    values = np.stack([pixels[b] for b in bands if b != "valid"], axis=-1)
    if not np.isfinite(values).all():
        raise ValidationFailure("nonfinite response values", {"expected_values": int(values.size),
            "nonfinite_values": int(np.count_nonzero(~np.isfinite(values)))})
    return np.ascontiguousarray(values)


def check_image_metadata(metadata, spec):
    bands = metadata.get("bands", [])
    expected_names = spec["bands"]
    grid = spec["grid"]
    if [b.get("id") for b in bands] != expected_names:
        raise ValidationFailure("saved image band order differs")
    for band in bands:
        g = band.get("grid", {})
        affine = g.get("affineTransform", {})
        want = grid["affineTransform"]
        if (band.get("dataType", {}).get("precision") != "FLOAT" or g.get("crsCode") != grid["crsCode"] or
                g.get("dimensions") != grid["dimensions"] or
                any(float(affine.get(k, 0)) != float(want.get(k, 0)) for k in
                    ("scaleX", "shearX", "translateX", "shearY", "scaleY", "translateY"))):
            raise ValidationFailure("saved image dtype/grid/affine differs")
    if metadata.get("type") != "IMAGE":
        raise ValidationFailure("saved boundary is not an image asset")


def image_diagnostics(actual, expected):
    result = compare(actual, expected)
    if actual.shape[-1] == 2:
        margins = np.abs(expected[..., 0].astype(np.float64) - expected[..., 1])
        result.update(argmax_disagreements=int(np.count_nonzero(actual.argmax(-1) != expected.argmax(-1))),
                      near_margin_pixels=int(np.count_nonzero(margins <= 2 * TOLERANCE)))
    return result


def _request(directory, label, category, body, **metadata):
    measured = fingerprint(body)
    cap = 8_000_000 if category == "table" else 2_000_000
    if measured["bytes"] > cap:
        raise ValueError(f"{label}: complete request exceeds approved body cap")
    relative = f"requests/{label}.json"
    save_json(directory / relative, body)
    return {"label": label, "category": category, "file": relative,
            "file_sha256": sha_file(directory / relative), "wire": measured, **metadata}


def _image_body(image, grid, namespace, destination):
    from tools import segformer_b0_request_plan as P
    grid = P._grid_view(grid)
    config = ee.batch._prepare_image_export_config(image,
        {"description": f"segformer_b0_{destination}", "assetId": f"{namespace}_{destination}",
         "region": grid.region(ee), "crs": grid.crs, "crsTransform": list(grid.transform),
         "maxPixels": grid.height * grid.width, "pyramidingPolicy": {".default": "sample"}, "overwrite": False},
        ee.batch.Task.ExportDestination.ASSET)
    return P._encode_config(config)


def _pixels_body(image, grid, names):
    from tools import segformer_b0_request_plan as P
    grid = P._grid_view(grid)
    image, bands = P._with_validity(image, names)
    with P.deep_serialization():
        expression = ee.serializer.encode(image, for_cloud_api=True)
    return {"expression": expression, "fileFormat": "NPY", "grid": grid.pixel_grid()}, bands


def _pixel_request(directory, label, image, grid, names, **metadata):
    from tools.live_fixtures import npy_header_length
    body, bands = _pixels_body(image, grid, names)
    d = body["grid"]["dimensions"]
    coefficient = metadata.get("kind") == "coefficient"
    if ((d["height"] > (1024 if coefficient else 224)) or d["width"] > (1 if coefficient else 224) or
            d["height"] * d["width"] * len(bands) > 1_000_000 or len(bands) > (257 if coefficient else 1024) or
            npy_header_length(bands, d["height"], d["width"]) > 9000):
        raise ValueError("readback grid/value/band/header exceeds approved limits")
    return _request(directory, label, "pixels", body, bands=bands, **metadata)


def _value_body(value):
    from tools import segformer_b0_request_plan as P
    with P.deep_serialization():
        return {"expression": ee.serializer.encode(value, for_cloud_api=True)}


def _approval_match(body, measured):
    if fingerprint(body) != {k: measured[k] for k in ("bytes", "sha256")}:
        raise ValueError("prepared dummy-UUID envelope differs from the exact approved sizing record")


def _cache_proof(state, references, report):
    from tools.fixtures.segformer_b0_fullmodel import state_sha256
    if (not report.get("passed") or not report.get("expanded_traversal_bitwise_equal") or
            not report.get("isolated_on_original_boundaries_bitwise_equal") or report.get("state_sha256") != state_sha256(state) or
            report.get("state_and_input_unchanged_after_checks") is not True or report.get("cloud_calls") != 0 or
            report.get("input_provenance", {}).get("input", {}).get("sha256") != array_sha(references["input"])):
        raise ValueError("cached numerical preflight lacks matching state/input/expanded audit")
    if report.get("target") != TOLERANCE or set(references) != {"input", *OUTPUTS}:
        raise ValueError("cached numerical target/output census differs")
    for name in ("input",) + OUTPUTS:
        check_hwc(references[name], SHAPES[name])
        provenance = report.get("reference_provenance", {}).get(name, {})
        if (provenance.get("sha256") != array_sha(references[name]) or provenance.get("shape") != list(SHAPES[name]) or
                provenance.get("dtype") != "float32" or provenance.get("values") != int(np.prod(SHAPES[name]))):
            raise ValueError("cached reference bytes/shape/precision differ from its recorded proof")
    for group, names in (("full", OUTPUTS), ("isolated", CUTS)):
        if set(report.get(group, {})) != set(names):
            raise ValueError("cached numerical comparison census differs")
        for name in names:
            output = name if group == "full" else OUTPUTS[CUTS.index(name)]
            result = report[group][name]
            if (result.get("passed") is not True or any(not isinstance(result.get(k), (int, float)) or
                    not np.isfinite(result[k]) or result[k] < 0 for k in ("max_abs", "mae", "rmse")) or
                    result.get("max_abs", float("inf")) > TOLERANCE or
                    result.get("over_target") != 0 or result.get("nonfinite") != 0 or
                    result.get("values") != int(np.prod(SHAPES[output]))):
                raise ValueError("cached numerical comparison did not pass at its fixed denominator")
    required_sources = {"tools/fixtures/segformer_b0_validation.py", "tools/fixtures/segformer_b0_fullmodel.py",
        "tests/reference/segformer_b0/full_model.py", "tests/reference/segformer_b0/model.py", "tests/reference/segformer_b0/blocks.py"}
    recorded_sources = report.get("source_files_sha256", {})
    if (required_sources != set(recorded_sources) or report.get("versions") != versions() or
            any(sha_file(ROOT / path) != digest for path, digest in recorded_sources.items())):
        raise ValueError("cached numerical preflight environment/source differs")


def prepare(directory, approved_plan_path, *, reference_cache=None):
    """Maintainer-only offline construction. No forwards, ADC, API call or Task object."""
    from tools import segformer_b0_request_plan as P
    from tools.fixtures.segformer_b0_fullmodel import (seeded_state, representative_boundaries,
        state_sha256, reject_negative_zero, array_provenance)
    from tensor_ee._swin_parameters import records_and_locations
    from tensor_ee.models.segformer_b0.configuration import COUNTER, check_parameters
    from tensor_ee.models.segformer_b0.decoder import model_constants
    from tensor_ee.weights.assets import feature_records
    from tools.fixtures.core import value_rows
    from tools.live_fixtures import offline_earth_engine
    directory = private_directory(directory)
    if sha_file(approved_plan_path) != APPROVED_PLAN_SHA256:
        raise ValueError("sizing record is not the untouched user-approved raw plan; changed scope needs new approval")
    if (directory / "prepared.json").exists():
        raise ValueError("preparation already exists; never regenerate submission UUIDs")
    approved = read_json(approved_plan_path)
    namespace = approved["placeholder_namespace"]
    project = namespace.split("/")[1]
    if (approved.get("primary_plan") != "split-stage1" or approved.get("weight_shards") != 9 or
            approved.get("proposed_primary_assets_and_batch_submissions") != 16 or
            approved.get("verification", {}).get("total_requests") != 204 or
            approved.get("weight_wire_limit") != 8_000_000 or "YOUR_EE_PROJECT" in namespace):
        raise ValueError("only the concrete user-approved split-stage1/9-shard plan is executable")
    cache = Path(reference_cache).resolve() if reference_cache else None
    state = load_npz(cache / "state.npz") if cache else seeded_state(P.SEED)
    arrays = representative_boundaries(P.SEED)
    check_parameters(state)
    reject_negative_zero(state)
    reject_negative_zero({"input": arrays["input"]})
    digest = state_sha256(state)
    if digest != approved["state_sha256"] or array_sha(arrays["input"]) != approved["boundary_arrays"]["input"]["sha256"]:
        raise ValueError("seeded state/input differs from the approved plan")
    directory.mkdir(parents=True, exist_ok=True)
    if not (directory / "state.npz").exists():
        np.savez(directory / "state.npz", **state)
    elif state_sha256(load_npz(directory / "state.npz")) != digest:
        raise ValueError("existing state cache differs")
    np.save(directory / "input.npy", arrays["input"], allow_pickle=False)
    if cache:
        refs = load_npz(cache / "references.npz")
        proof = read_json(cache / "numerical-preflight-raw.json")
        _cache_proof(state, refs, proof)
        if proof.get("references_npz_sha256") != sha_file(cache / "references.npz"):
            raise ValueError("cached reference NPZ container changed")
        if array_sha(refs["input"]) != array_sha(arrays["input"]):
            raise ValueError("reference cache input differs from approved input")
        for name in ("references.npz", "numerical-preflight-raw.json"):
            if cache / name != directory / name:
                shutil.copyfile(cache / name, directory / name)
    records, _ = records_and_locations({}, {}, {k: v for k, v in state.items() if k != COUNTER})
    rows = list(feature_records(list(records.values())))
    if len(records) != 128 or len(rows) != 15682:
        raise ValueError("coefficient source census differs")
    requests, image_specs, shards = [], {}, approved["weight_manifest"]["shards"]
    offset = 0
    producer_specs = {}
    with offline_earth_engine():
        for shard in shards:
            count, index = shard["rows"], shard["index"]
            selected = rows[offset:offset + count]
            if len(selected) != count or sorted({r["properties"]["layer"] for r in selected}) != shard["layers"]:
                raise ValueError("approved coefficient shard rows/layers differ")
            body = P._table_request(selected, namespace + "_weights", index)
            _approval_match(body, shard["export_table"])
            request_id = str(uuid.uuid4())
            body["requestId"] = request_id
            asset = shard["asset_id"]
            requests.append(_request(directory, f"weight_{index:04d}", "table", body,
                                     request_id=request_id, asset=asset, rows=count, layers=shard["layers"]))
            requests.append(_request(directory, f"metadata_weight_{index:04d}", "value",
                                     _value_body(ee.FeatureCollection(asset).size()), asset=asset, expected_rows=count))
            offset += count
        if offset != len(rows):
            raise ValueError("coefficient shard coverage is incomplete")
        manifest = P.planned_manifest(list(records.values()), shards, namespace + "_weights", digest)
        if manifest["layers"] != approved["weight_manifest"]["layers"] or manifest["manifest_hash"] != approved["weight_manifest"]["manifest_hash"]:
            raise ValueError("approved manifest mapping/fingerprints differ")
        server = P.asset_state(state, manifest, digest)
        constants = model_constants(ee, server)
        selected_candidates = {candidate["cut"]: candidate for candidate in approved["candidates"]
            if candidate["weight_source"] == candidate["boundary_source"] == "asset-reference"}
        for cut in ("input",) + CUTS:
            if cut == "input":
                image, grid, names = P._boundary_image("input", "inline", arrays, namespace), P.SOURCE_GRID, ("b0", "b1")
                required, destination, expected_export = [], "input", approved["input_export"]["wire"]
            else:
                out, required, destination = P.build_cut(cut, constants, "asset-reference", arrays, namespace)
                image, grid, names = out.image, out.grid, out.bands
                expected_export = selected_candidates[cut]["image_export"]["wire"]
            body = _image_body(image, grid, namespace, destination)
            _approval_match(body, expected_export)
            request_id = str(uuid.uuid4())
            body["requestId"] = request_id
            asset = f"{namespace}_{destination}"
            request = _request(directory, f"export_{destination}", "image", body, request_id=request_id,
                               asset=asset, output=destination, cut=cut, requires=required)
            requests.append(request)
            view_grid = P._grid_view(grid)
            image_specs[destination] = {"asset": asset, "grid": view_grid.pixel_grid(), "bands": list(names),
                                       "shape": list(SHAPES[destination])}
            stored = ee.Image(asset)
            metadata = ee.Dictionary({"bands": stored.bandNames(), "types": stored.bandTypes(),
                "crs": stored.projection().crs(), "projection_matches": stored.projection().transform().compareTo(
                    view_grid.projection(ee).transform()).eq(0)})
            requests.append(_request(directory, f"metadata_{destination}", "value", _value_body(metadata), output=destination, asset=asset))
            if cut not in ("input", "decoder"):
                producer = _pixel_request(directory, f"producer_{destination}", image, grid, names,
                                          kind="producer", output=destination, requires=required, cut=cut)
                _approval_match(read_json(directory / producer["file"]), selected_candidates[cut]["compute_pixels"]["wire"])
                requests.append(producer)
                producer_specs[destination] = producer
            saved = _pixel_request(directory, f"saved_{destination}", stored.select(list(names)).reproject(view_grid.projection(ee)),
                                   grid, names, kind="saved", output=destination)
            requests.append(saved)
            gc.collect()
        for name, expected in state.items():
            if name == COUNTER:
                continue
            flat = server[name].array.reshape([int(expected.size), 1])
            for part, segment in enumerate(P.array_segments(expected.size)):
                value = flat.slice(0, segment["start"], segment["end"]).reshape([segment["rows"], segment["columns"]])
                image, _ = value_rows(ee, value, segment["rows"], segment["columns"])
                grid = P.TokenGrid(segment["rows"], 1)
                request = _pixel_request(directory, f"coefficient_{len([r for r in requests if r.get('kind') == 'coefficient']):03d}",
                    image, grid, P.band_names(segment["columns"]), kind="coefficient", name=name, part=part,
                    original_shape=list(expected.shape), segment=segment)
                requests.append(request)
        del constants
    # Every pixel envelope must equal its approved counterpart; the UUID change
    # affects only exports. Metadata reads are newly prepared within the approved16 cap.
    approved_reads = {(r["kind"], r["name"], r.get("segment", {}).get("start")): r["measurement"]["wire"]
                      for r in approved["verification"]["requests"]}
    for request in requests:
        if request["category"] != "pixels":
            continue
        kind = {"coefficient": "coefficient", "producer": "stage_producer", "saved": "saved_image"}[request["kind"]]
        key = (kind, request.get("name", request.get("output")), request.get("segment", {}).get("start"))
        _approval_match(read_json(directory / request["file"]), approved_reads[key])
    metadata = {"schema_version": 1, "kind": "segformer_b0_approved_preparation", "project": project,
        "namespace": namespace, "seed": P.SEED, "state_sha256": digest, "input_sha256": array_sha(arrays["input"]),
        "approved_plan_sha256": sha_file(approved_plan_path), "requests": requests, "image_specs": image_specs,
        "planned_manifest": manifest, "versions": versions(), "source_files_sha256": source_hashes(),
        "artifacts": {name: sha_file(directory / name) for name in ("state.npz", "input.npy")},
        "wire_caps": {"table": 8_000_000, "image": 2_000_000, "pixels": 2_000_000},
        "runtime_workload_tag": "absent", "scope": "exact user-approved16batch/204pixel split; no automatic fallback",
        "provenance": provenance()}
    if cache:
        metadata["artifacts"].update({name: sha_file(directory / name) for name in ("references.npz", "numerical-preflight-raw.json")})
    validate_prepared(directory, metadata)
    save_json(directory / "prepared.json", metadata)
    save_json(directory / "state.json", {"state": "PREPARED", "tasks": {}, "checks": {}, "ledger": {}, "started_unix": None})
    return metadata


def validate_prepared(directory, prepared):
    """Pure validation of immutable files, request census and approved limits."""
    directory = Path(directory)
    requests = prepared["requests"]
    if prepared.get("approved_plan_sha256") != APPROVED_PLAN_SHA256:
        raise ValueError("prepared requests lack the exact user-approved sizing anchor")
    counts = {kind: sum(r["category"] == kind for r in requests) for kind in ("table", "image", "pixels", "value")}
    if counts != {"table": 9, "image": 7, "pixels": 204, "value": 16}:
        raise ValueError(f"prepared request census differs: {counts}")
    if len({r["label"] for r in requests}) != len(requests):
        raise ValueError("prepared labels are duplicated")
    ids, assets, coefficient_views = [], [], []
    for request in requests:
        path = directory / request["file"]
        if path.resolve().parent != (directory / "requests").resolve() or sha_file(path) != request["file_sha256"]:
            raise ValueError("prepared request path/content changed")
        body = read_json(path)
        if fingerprint(body) != request["wire"]:
            raise ValueError("prepared complete wire body fingerprint differs")
        category = request["category"]
        if request["wire"]["bytes"] > (8_000_000 if category == "table" else 2_000_000):
            raise ValueError("prepared wire cap exceeded")
        if category in ("table", "image"):
            rid = request["request_id"]
            if str(uuid.UUID(rid)) != rid or body.get("requestId") != rid:
                raise ValueError("export UUID differs")
            destination = body.get("assetExportOptions", {}).get("earthEngineDestination", {}).get("name")
            if destination != request["asset"] or not destination.startswith(f"projects/{prepared['project']}/assets/"):
                raise ValueError("export destination differs")
            if body.get("assetExportOptions", {}).get("earthEngineDestination", {}).get("overwrite") is not False:
                raise ValueError("asset overwrite is forbidden")
            ids.append(rid)
            assets.append(destination)
        if request.get("kind") == "coefficient":
            coefficient_views.append(request)
        if category == "pixels":
            from tools.live_fixtures import npy_header_length
            d = body["grid"]["dimensions"]
            bands = request["bands"]
            coefficient = request.get("kind") == "coefficient"
            if (body.get("fileFormat") != "NPY" or bands[-1] != "valid" or len(set(bands)) != len(bands) or
                    d["height"] > (1024 if coefficient else 224) or d["width"] > (1 if coefficient else 224) or
                    d["height"] < 1 or d["width"] < 1 or len(bands) > (257 if coefficient else 1024) or
                    d["height"] * d["width"] * len(bands) > 1_000_000 or
                    npy_header_length(bands, d["height"], d["width"]) > 9000):
                raise ValueError("prepared readback limits differ")
    if len(set(ids)) != 16 or len(set(assets)) != 16:
        raise ValueError("submission UUIDs and destinations must be unique")
    if len(coefficient_views) != 192 or len({r["name"] for r in coefficient_views}) != 191:
        raise ValueError("source-array/view coverage differs")
    for name in {r["name"] for r in coefficient_views}:
        pieces = sorted((r for r in coefficient_views if r["name"] == name), key=lambda r: r["segment"]["start"])
        expected_size = int(np.prod(pieces[0]["original_shape"]))
        at = 0
        for part in pieces:
            segment = part["segment"]
            if (segment["start"] != at or segment["end"] <= at or
                    segment["end"] - at != segment["rows"] * segment["columns"] or
                    part["original_shape"] != pieces[0]["original_shape"]):
                raise ValueError("source view coverage/order/shape differs")
            at = segment["end"]
        if at != expected_size:
            raise ValueError("source view coverage is incomplete")
    if not {"state.npz", "input.npy"} <= set(prepared["artifacts"]):
        raise ValueError("prepared source artifacts are missing")
    for name, digest in prepared["artifacts"].items():
        if Path(name).name != name:
            raise ValueError("prepared artifact path is not a private basename")
        if sha_file(directory / name) != digest:
            raise ValueError(f"prepared raw artifact changed: {name}")
    if prepared["versions"] != versions() or prepared["source_files_sha256"] != source_hashes():
        raise ValueError("prepared source or installed versions changed")
    return counts


def preflight(directory):
    """Maintainer-only numerical preflight; reuse a source/state-pinned reference cache."""
    directory = private_directory(directory)
    prepared = read_json(directory / "prepared.json")
    validate_prepared(directory, prepared)
    state = load_npz(directory / "state.npz")
    x = np.load(directory / "input.npy", allow_pickle=False)
    if "references.npz" in prepared["artifacts"]:
        full = load_npz(directory / "references.npz")
        report = read_json(directory / "numerical-preflight-raw.json")
        _cache_proof(state, full, report)
    else:
        from tools.fixtures.segformer_b0_validation import full_reference, isolated_reference
        from reference.segformer_b0.full_model import mirror_full, mirror_cut, CUT_INPUTS
        from tools.fixtures.segformer_b0_fullmodel import state_sha256, array_provenance
        from tools.live_fixtures import offline_earth_engine
        before_state, before_input = state_sha256(state), array_sha(x)
        with offline_earth_engine():
            full = full_reference(state, x)
            mirrored = mirror_full(state, x)
            checks = {name: compare(mirrored[name], full[name]) for name in OUTPUTS}
            isolated = {}
            for cut, output in zip(CUTS, OUTPUTS):
                incoming = {n: x if n == "input" else full[n] for n in CUT_INPUTS[cut]}
                reference = isolated_reference(cut, state, incoming)
                compare(reference, full[output], exact=True)
                isolated[cut] = compare(mirror_cut(cut, state, incoming), reference)
        if before_state != state_sha256(state) or before_input != array_sha(x):
            raise ValueError("references/mirrors changed source state or input")
        np.savez(directory / "references.npz", input=x, **full)
        proof_paths = ("tools/fixtures/segformer_b0_validation.py", "tools/fixtures/segformer_b0_fullmodel.py",
            "tests/reference/segformer_b0/full_model.py", "tests/reference/segformer_b0/model.py", "tests/reference/segformer_b0/blocks.py")
        report = {"passed": True, "full": checks, "isolated": isolated, "target": TOLERANCE,
            "expanded_traversal_bitwise_equal": True, "isolated_on_original_boundaries_bitwise_equal": True,
            "state_and_input_unchanged_after_checks": True, "state_sha256": before_state,
            "input_provenance": array_provenance({"input": x}), "reference_provenance": array_provenance({"input": x, **full}),
            "source_files_sha256": {p: sha_file(ROOT / p) for p in proof_paths}, "versions": versions(),
            "references_npz_sha256": sha_file(directory / "references.npz"), "cloud_calls": 0}
        save_json(directory / "numerical-preflight-raw.json", report)
        _cache_proof(state, {"input": x, **full}, report)
    result = {"passed": True, "prepared_sha256": sha_file(directory / "prepared.json"),
              "reference_sha256": sha_file(directory / "references.npz"), "reference_checks": report,
              "state_sha256": prepared["state_sha256"], "input_sha256": prepared["input_sha256"],
              "source_files_sha256": prepared["source_files_sha256"], "versions": prepared["versions"]}
    save_json(directory / "preflight.json", result)
    return result


def _gates(directory, gates_path):
    gates = read_json(gates_path)
    preflight_report = read_json(directory / "preflight.json")
    prepared = read_json(directory / "prepared.json")
    if (gates.get("g1") != 45 or any(gates.get(k) != "passed" for k in ("pytest", "catalogue", "segformer_offline", "review")) or
            gates.get("prepared_sha256") != sha_file(directory / "prepared.json") or
            gates.get("preflight_sha256") != sha_file(directory / "preflight.json") or not preflight_report.get("passed") or
            preflight_report.get("prepared_sha256") != sha_file(directory / "prepared.json") or
            preflight_report.get("reference_sha256") != sha_file(directory / "references.npz") or
            preflight_report.get("state_sha256") != prepared["state_sha256"] or
            preflight_report.get("input_sha256") != prepared["input_sha256"] or
            preflight_report.get("versions") != prepared["versions"] or
            preflight_report.get("source_files_sha256") != prepared["source_files_sha256"]):
        raise ValueError("maintainer release gates/source-pinned preflight are not complete for these exact envelopes")
    _cache_proof(load_npz(directory / "state.npz"), load_npz(directory / "references.npz"), preflight_report["reference_checks"])


def _not_found(error):
    """Recognize an actual HTTP404 through SDK translation, never a text guess."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if getattr(getattr(error, "resp", None), "status", None) == 404:
            return True
        error = error.__cause__ or error.__context__
    return False


@contextlib.contextmanager
def ipv4_first():
    original = socket.getaddrinfo
    def addresses(*args, **kwargs):
        return sorted(original(*args, **kwargs), key=lambda item: item[0] != socket.AF_INET)
    from unittest import mock
    with mock.patch.object(socket, "getaddrinfo", addresses):
        yield


class Execution:
    """Write-ahead state and strictly sequential interactive execution."""
    def __init__(self, directory, prepared, *, clock=time.time, sleep=time.sleep):
        self.directory, self.prepared = Path(directory), prepared
        self.clock, self.sleep = clock, sleep
        self.state = read_json(self.directory / "state.json")
        self.requests = {r["label"]: r for r in prepared["requests"]}
        approved = {kind: {} for kind in ("table", "image", "pixels", "value")}
        for request in prepared["requests"]:
            approved[request["category"]][request["wire"]["sha256"]] = request
        assets = {r["asset"] for r in prepared["requests"] if r["category"] in ("table", "image")}
        self.guard = Guard(prepared["project"], approved, assets, save=self.save_ledger)

    def save(self):
        save_json(self.directory / "state.json", self.state)
        save_json(self.directory / "evidence-raw.json", self.state)

    def save_ledger(self, ledger):
        self.state["ledger"] = ledger
        self.save()

    def body(self, label):
        request = self.requests[label]
        path = self.directory / request["file"]
        if sha_file(path) != request["file_sha256"]:
            raise ValueError("request changed after preparation")
        body = read_json(path)
        if fingerprint(body) != request["wire"]:
            raise ValueError("wire body changed after preparation")
        return body

    def submit(self, label):
        request = self.requests[label]
        if label in self.state["tasks"]:
            raise ValueError("submission already recorded; no automatic retry/resume")
        active = [t for t in self.state["tasks"].values() if t["state"] in ("SUBMITTING", "SUBMITTED", "RUNNING", "UNKNOWN", "POLLING_CAP")]
        if ((request["category"] == "table" and (len(active) >= 2 or any(t["kind"] != "table" for t in active))) or
                (request["category"] == "image" and active)):
            raise ValueError("approved active-task concurrency would be exceeded")
        body = self.body(label)
        record = {"state": "SUBMITTING", "request_id": request["request_id"], "asset": request["asset"],
                  "body_sha256": request["wire"]["sha256"], "submitted_unix": self.clock(),
                  "description": body.get("description"), "kind": request["category"]}
        self.state["tasks"][label] = record
        self.save()                         # durable intent BEFORE the Cloud call
        try:
            function = ee.data.exportTable if request["category"] == "table" else ee.data.exportImage
            operation = function(request["request_id"], body)
            record.update(operation=operation, response_received=True)
            self.save()
            self.guard.add_operation(operation.get("name"))
        except BaseException as error:
            returned = getattr(error, "returned_result", None)
            if hasattr(error, "returned_result"):
                record.update(operation=returned, response_received=True)
                self.save()
            if isinstance(returned, dict) and isinstance(returned.get("name"), str):
                try:
                    self.guard.add_operation(returned["name"])
                except BaseException:
                    pass
                else:
                    record.update(state="SUBMITTED", operation_name=returned["name"], operation=returned,
                                  error=f"{type(error).__name__}: {error}")
                    self.save()
                    raise
            record.update(state="UNKNOWN", error=f"{type(error).__name__}: {error}")
            self.save()
            raise
        record.update(state="SUBMITTED", operation_name=operation["name"], operation=operation)
        self.save()
        return label

    def poll(self, labels):
        active = list(labels)
        # Do not poll a task more frequently than30seconds, including its first poll.
        while active:
            self.guard.check_time()
            now = self.clock()
            wait = min(max(0, self.state["tasks"][name].get("last_poll_unix",
                        self.state["tasks"][name]["submitted_unix"]) + 30 - now) for name in active)
            if wait:
                self.sleep(wait)
            for name in list(active):
                record = self.state["tasks"][name]
                now = self.clock()
                if now - record["submitted_unix"] >= 1800:
                    record["state"] = "POLLING_CAP"
                    self.save()
                    raise BudgetStop("per-task30-minute observation cap reached")
                if now - record.get("last_poll_unix", record["submitted_unix"]) < 30:
                    continue
                self.guard.request_seconds = min(300., 1800. - (now - record["submitted_unix"]))
                try:
                    operation = ee.data.getOperation(record["operation_name"])
                except BudgetStop as error:
                    if hasattr(error, "returned_result"):
                        record.update(last_operation=error.returned_result, last_poll_unix=self.clock(), state="POLLING_CAP")
                        self.save()
                    raise
                finally:
                    self.guard.request_seconds = 300.
                record["last_poll_unix"] = self.clock()
                record["last_operation"] = operation
                self.save()
                if operation.get("name") != record["operation_name"]:
                    raise ValidationFailure("status response operation identity differs")
                if self.clock() - record["submitted_unix"] >= 1800:
                    record["state"] = "POLLING_CAP"
                    self.save()
                    raise BudgetStop("task response returned after its30-minute observation cap")
                self.guard.check_time()
                if operation.get("done"):
                    if operation.get("error"):
                        record.update(state="FAILED", error=operation["error"])
                        self.save()
                        raise ValidationFailure("batch export failed", operation)
                    record["state"] = "COMPLETED"
                    active.remove(name)
                else:
                    record["state"] = "RUNNING"
                self.save()
        return labels

    def collision_checks(self):
        for asset in sorted(self.guard.assets):
            try:
                result = ee.data.getAsset(asset)
            except BaseException as error:
                if not _not_found(error):
                    raise
            else:
                raise ValidationFailure("destination exists or returned invalid metadata; onlyHTTP404 proves absence", {"asset": asset})
            self.state["checks"].setdefault("collisions", []).append({"asset": asset, "absent": True})
            self.save()

    def pixels(self, label):
        begun = time.monotonic()
        body = self.body(label)
        call = ee.data._get_cloud_projects_raw().image().computePixels(project=f"projects/{self.prepared['project']}", body=body)
        raw = ee.data._execute_cloud_call(call, num_retries=0)
        npy_path = self.directory / "responses" / f"{label}.npy"
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(raw, bytes):
            npy_path.write_bytes(raw)
        else:
            np.save(npy_path, raw, allow_pickle=False)
        try:
            value = read_pixels(raw, {"body": body, "bands": self.requests[label]["bands"]})
        except BaseException as error:
            self.state["checks"].setdefault("responses", {})[label] = {"sha256": sha_file(npy_path),
                "passed": False, "error": str(error), "details": getattr(error, "details", None),
                "wall_seconds": time.monotonic() - begun}
            self.save()
            raise
        self.state["checks"].setdefault("responses", {})[label] = {"sha256": sha_file(npy_path),
            "values_sha256": array_sha(value), "wall_seconds": time.monotonic() - begun}
        self.save()
        return value

    def numerical_check(self, output, name, fn):
        try:
            result = fn()
        except ValidationFailure as error:
            self.state["checks"].setdefault("numerical", {}).setdefault(output, {})[name] = {
                "passed": False, "error": str(error), "details": error.details}
            self.save()
            raise
        self.state["checks"].setdefault("numerical", {}).setdefault(output, {})[name] = result
        self.save()
        return result

    def value(self, label):
        body = self.body(label)
        call = ee.data._get_cloud_projects().value().compute(project=f"projects/{self.prepared['project']}", body=body, prettyPrint=False)
        return ee.data._execute_cloud_call(call, num_retries=0)["result"]

    def completed_metadata(self, label):
        request = self.requests[label]
        asset = ee.data.getAsset(request["asset"])
        if asset.get("name") != request["asset"]:
            raise ValidationFailure("completion metadata destination differs")
        if request["category"] == "table":
            if asset.get("type") != "TABLE":
                raise ValidationFailure("weight destination is not a table asset")
            rows = self.value("metadata_" + label)
            if type(rows) is not int or rows != request["rows"]:
                raise ValidationFailure("completed weight shard row count differs")
            check = {"rows": rows, "asset": asset}
        else:
            output = request["output"]
            spec = self.prepared["image_specs"][output]
            check_image_metadata(asset, spec)
            metadata = self.value(f"metadata_{output}")
            if (metadata.get("bands") != spec["bands"] or metadata.get("crs") != spec["grid"]["crsCode"] or
                    metadata.get("projection_matches") not in (True, 1) or set(metadata.get("types", {})) != set(spec["bands"]) or
                    any(v.get("precision") != "float" for v in metadata.get("types", {}).values())):
                raise ValidationFailure("saved-image computed metadata differs")
            check = {"asset": asset, "metadata": metadata}
        self.state["checks"].setdefault("asset_metadata", {})[label] = check
        self.save()
        return check

    def coefficients(self, state):
        from tensor_ee.models.segformer_b0.configuration import COUNTER
        views = [r for r in self.prepared["requests"] if r.get("kind") == "coefficient"]
        restored = {}
        for name, expected in state.items():
            if name == COUNTER:
                continue
            pieces = sorted((r for r in views if r["name"] == name), key=lambda r: r["segment"]["start"])
            flat = np.concatenate([self.pixels(r["label"]).reshape(-1) for r in pieces])
            actual = flat.reshape(expected.shape)
            if actual.dtype != np.float32 or not np.array_equal(actual.view(np.uint32), expected.view(np.uint32)):
                raise ValidationFailure("coefficient transport bits differ", {"name": name,
                    "actual_sha256": array_sha(actual), "expected_sha256": array_sha(expected),
                    "differing_bits": int(np.count_nonzero(actual.view(np.uint32) != expected.view(np.uint32)))})
            restored[name] = actual
            self.state["checks"].setdefault("coefficients", {})[name] = {"shape": list(actual.shape),
                "values": int(actual.size), "sha256": array_sha(actual), "differing_bits": 0}
            self.save()
        if len(restored) != 191:
            raise ValidationFailure("computational coefficient readback is incomplete")
        np.savez(self.directory / "coefficients-readback.npz", **restored)

    def execute(self):
        if self.state.get("state") != "PREPARED" or self.state["tasks"]:
            raise ValueError("run has already started; explicit user steering is required to recover")
        state = load_npz(self.directory / "state.npz")
        full = load_npz(self.directory / "references.npz")
        actual = {}
        self.state.update(state="RUNNING", started_unix=self.clock(), prepared_sha256=sha_file(self.directory / "prepared.json"),
                          provenance=provenance())
        self.save()
        self.collision_checks()
        self.poll([self.submit("export_input")])
        self.completed_metadata("export_input")
        actual["input"] = self.pixels("saved_input")
        self.state["checks"]["input_transport"] = compare(actual["input"], full["input"], exact=True)
        self.save()
        tables = [r["label"] for r in self.prepared["requests"] if r["category"] == "table"]
        for first in range(0, len(tables), 2):
            active = [self.submit(label) for label in tables[first:first + 2]]
            self.poll(active)
            for label in active:
                self.completed_metadata(label)
        manifest = json.loads(json.dumps(self.prepared["planned_manifest"]))
        manifest.update(planning_only=False, assets_exist=True)
        for shard in manifest["shards"]:
            label = f"weight_{shard['index']:04d}"
            shard.update(status="COMPLETED", operation_name=self.state["tasks"][label]["operation_name"],
                         request_id=self.requests[label]["request_id"], actual_export_table=self.requests[label]["wire"])
        save_json(self.directory / "manifest-ready.json", manifest)
        self.coefficients(state)            # all source arrays verified before model requests
        from tools.fixtures.segformer_b0_validation import isolated_reference
        from reference.segformer_b0.full_model import mirror_cut, CUT_INPUTS
        for cut, output in zip(CUTS, OUTPUTS):
            incoming = {name: actual[name] for name in CUT_INPUTS[cut]}
            isolated = isolated_reference(cut, state, incoming)
            mirrored = mirror_cut(cut, state, incoming)
            self.numerical_check(output, "mirror_vs_isolated", lambda: compare(mirrored, isolated))
            np.save(self.directory / f"isolated-{output}.npy", isolated, allow_pickle=False)
            np.save(self.directory / f"mirror-{output}.npy", mirrored, allow_pickle=False)
            if cut != "decoder":
                producer = self.pixels(f"producer_{output}")
                self.numerical_check(output, "producer_isolated", lambda: image_diagnostics(producer, isolated))
                self.numerical_check(output, "producer_accumulated", lambda: image_diagnostics(producer, full[output]))
                self.numerical_check(output, "producer_mirror", lambda: compare(producer, mirrored))
            else:
                producer = None
            self.poll([self.submit(f"export_{output}")])
            self.completed_metadata(f"export_{output}")
            actual[output] = self.pixels(f"saved_{output}")
            if producer is not None:
                self.numerical_check(output, "producer_saved_transport", lambda: compare(actual[output], producer, exact=True))
            self.numerical_check(output, "saved_isolated", lambda: image_diagnostics(actual[output], isolated))
            self.numerical_check(output, "saved_accumulated", lambda: image_diagnostics(actual[output], full[output]))
            self.numerical_check(output, "saved_mirror", lambda: compare(actual[output], mirrored))
        expected_counts = dict(CAPS) | {"algorithms": self.guard.counts["algorithms"], "status": self.guard.counts["status"]}
        if self.guard.counts != expected_counts:
            raise ValidationFailure("completed run call census differs", self.guard.record())
        self.guard.check_time()
        self.state.update(state="PASSED", finished_unix=self.clock(), wall_seconds=time.monotonic() - self.guard.started, accuracy_claim=False,
            scope="one seeded full chip/state; full staged agreement and observed transport, no monolithic feasibility claim")
        self.save()
        return self.state


def run(directory, gates_path):
    """Maintainer-only ADC execution after complete offline gates and independent review."""
    directory = private_directory(directory)
    prepared = read_json(directory / "prepared.json")
    validate_prepared(directory, prepared)
    _gates(directory, gates_path)
    execution = Execution(directory, prepared)
    if execution.state.get("state") != "PREPARED" or execution.state["tasks"]:
        raise ValueError("no automatic resume or replay of a started run")
    try:
        with ipv4_first(), execution.guard.installed():
            ee.data.setMaxRetries(0)
            ee.data.resetWorkloadTag(resetDefault=True)
            execution.guard.initializing = True
            try:
                import google.auth
                credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/earthengine",
                    "https://www.googleapis.com/auth/cloud-platform"])
                ee.Initialize(credentials=credentials, project=prepared["project"])
            finally:
                execution.guard.initializing = False
            ee.data.setMaxRetries(0)
            ee.data.resetWorkloadTag(resetDefault=True)
            return execution.execute()
    except BaseException as error:
        execution.state.update(state="STOPPED", error=f"{type(error).__name__}: {error}",
            failure_details=getattr(error, "details", None), stopped_unix=time.time(),
            wall_seconds=time.monotonic() - execution.guard.started,
            note="no automatic retry/resume/cancel; submitted tasks may continue")
        execution.save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "prepare", "preflight", "run", "status", "verify"), required=True)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--plan", type=Path, help="untouched private actual-namespace sizing JSON")
    parser.add_argument("--reference-cache", type=Path, help="maintainer-produced state/references/numerical-preflight-raw cache")
    parser.add_argument("--gates", type=Path, help="maintainer gate attestation pinned to prepared/preflight SHA")
    args = parser.parse_args()
    if args.mode == "plan":
        directory = private_directory(args.state)
        prepared = read_json(directory / "prepared.json")
        print(json.dumps({"counts": validate_prepared(directory, prepared), "caps": CAPS,
            "largest_wire_bytes": {kind: max(r["wire"]["bytes"] for r in prepared["requests"] if r["category"] == kind)
                for kind in ("table", "image", "pixels", "value")}}, indent=2))
    elif args.mode == "status":
        state = read_json(private_directory(args.state) / "state.json")
        print(json.dumps({"state": state["state"], "tasks": state["tasks"], "ledger": state.get("ledger", {})}, indent=2))
    elif args.mode == "verify":
        print(json.dumps(verify_local(args.state), indent=2))
    elif args.mode == "prepare":
        if not args.plan:
            parser.error("prepare requires --plan")
        result = prepare(args.state, args.plan, reference_cache=args.reference_cache)
        print(json.dumps({"state": "PREPARED", "counts": validate_prepared(args.state, result)}, indent=2))
    elif args.mode == "preflight":
        print(json.dumps({"passed": preflight(args.state)["passed"]}))
    else:
        if not args.gates:
            parser.error("run requires source-pinned --gates")
        print(json.dumps({"state": run(args.state, args.gates)["state"]}))


def verify_local(directory):
    """Audit retained raw evidence locally; never poll, compute or resume."""
    directory = private_directory(directory)
    prepared = read_json(directory / "prepared.json")
    validate_prepared(directory, prepared)
    state = read_json(directory / "state.json")
    if state.get("state") != "PASSED":
        raise ValidationFailure("live execution has not recorded a complete pass")
    responses = state["checks"].get("responses", {})
    if len(responses) != 204 or len(state["checks"].get("coefficients", {})) != 191:
        raise ValidationFailure("retained readback/source-array census is incomplete")
    for request in prepared["requests"]:
        if request["category"] != "pixels":
            continue
        path = directory / "responses" / f"{request['label']}.npy"
        if sha_file(path) != responses[request["label"]]["sha256"]:
            raise ValidationFailure("retained raw response checksum differs")
        value = read_pixels(path.read_bytes(), {"body": read_json(directory / request["file"]), "bands": request["bands"]})
        if array_sha(value) != responses[request["label"]]["values_sha256"]:
            raise ValidationFailure("retained response values differ")
    return {"passed": True, "computePixels_artifacts": 204, "source_arrays": 191,
            "state_sha256": prepared["state_sha256"], "input_sha256": prepared["input_sha256"],
            "scope": "local retained-evidence audit; no new cloud work"}


if __name__ == "__main__":
    main()
