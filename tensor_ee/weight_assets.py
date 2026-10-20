"""Store arbitrary frozen convolution tensors in Earth Engine table assets.

Each feature has a harmless point geometry and properties ``layer``, ``row``,
``weights_json`` (one JSON string of a numeric chunk), and, when present, ``bias``.  EE table exports
reject null geometries and list-valued properties.  Chunks are sorted by an explicit integer index;
no inference weight is downloaded or embedded in an asset-backed graph.

Uploads are independently measured, bounded request graphs.  Nine significant
figures preserve every finite float32 value on restoration through
``ee.Array.float()``. Float32 is the maximum supported model precision.
"""
from __future__ import annotations

import contextlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Iterator, Mapping, Sequence

import ee
import numpy as np
import torch
from torch import nn
from torch.fx import Node, symbolic_trace

from .compiler import UnsupportedOperation, _ROOT_MODULES
from .semantics import semantic_call, diagnostic_convolution
# The asset format and its provider moved to the library; these names are
# re-exports of the same objects (one FeatureCollectionWeightProvider class).
from .weights.assets import (  # noqa: F401
    ConvolutionWeights, FeatureCollectionWeightProvider, convolution_weights, feature_records)


def effective_convolutions(model: nn.Module) -> list[ConvolutionWeights]:
    """Collect exactly the final graph's conv tensors, including BN folding.

    Names follow FX node keys, as ``compile_model`` does.  A convolution with a
    sole BatchNorm consumer is replaced by its folded convolution, keyed by the
    BN node.  Original conv diagnostics are deliberately absent from the asset.
    """
    if any(module.training for module in model.modules()):
        raise ValueError("Set model.eval() before collecting inference weights")
    if isinstance(model, _ROOT_MODULES):
        model = nn.Sequential(model)
    traced = symbolic_trace(model)
    result = []
    for node in traced.graph.nodes:
        if node.op in ("placeholder", "output"):
            continue
        try:
            call = semantic_call(node, traced)
        except UnsupportedOperation as error:
            raise UnsupportedOperation(f"At FX node {node.name} ({node.target}): {error}") from error
        if call.kind == "conv" and not diagnostic_convolution(node, traced):
            result.append(convolution_weights(node.name, call.module, str(node.target)))
        elif call.kind == "batchnorm":
            source = str(call.options["predecessor"].target) + "+" + str(node.target)
            result.append(convolution_weights(node.name, call.module, source))
    if not result:
        raise ValueError("Model contains no deployable convolutions")
    return result


def float32_wire_values(values: np.ndarray) -> list[float]:
    """Short decimal values which round back to the exact source float32s."""
    return [float(format(float(value), ".9g")) for value in values]


def collection_from_records(records: Sequence[Mapping[str, Any]]) -> ee.FeatureCollection:
    return ee.FeatureCollection([ee.Feature(dict(record)) for record in records])


def _split_measured(records: list[dict[str, Any]], max_bytes: int) -> Iterator[tuple[list[dict[str, Any]], int]]:
    actual = len(ee.serializer.toJSON(collection_from_records(records)).encode("utf-8"))
    if actual <= max_bytes:
        yield records, actual
    elif len(records) == 1:
        raise ValueError(f"One row requires {actual} serialized bytes, exceeding {max_bytes}")
    else:
        center = len(records) // 2
        yield from _split_measured(records[:center], max_bytes)
        yield from _split_measured(records[center:], max_bytes)


def upload_chunks(parameters: Sequence[ConvolutionWeights], max_bytes: int = 8_000_000
                  ) -> Iterator[tuple[list[dict[str, Any]], int]]:
    """Pack rows conservatively, then verify each actual EE graph byte count."""
    if not 100_000 <= max_bytes <= 8_000_000:
        raise ValueError("max_bytes must be between 100,000 and 8,000,000")
    records = []
    rough_bytes = 0
    # A literal FC is converted into API feature expressions.  Reserving 10%
    # and 32 KB covers that overhead; the exact serializer check is decisive.
    target = int(max_bytes * .9) - 32_000
    for record in feature_records(parameters):
        size = len(json.dumps(record, separators=(",", ":")).encode()) + 512
        if records and rough_bytes + size > target:
            yield from _split_measured(records, max_bytes)
            records, rough_bytes = [], 0
        records.append(record)
        rough_bytes += size
    if records:
        yield from _split_measured(records, max_bytes)


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class _RecordedTask:
    """A task an earlier run started, known only by the ID its manifest recorded: polled
    with ``ee.data.getTaskStatus``."""

    def __init__(self, task_id: str):
        self.id = task_id

    def status(self) -> dict[str, Any]:
        return ee.data.getTaskStatus([self.id])[0]


def _layer_metadata(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {key: {k: v for k, v in meta.items() if k != "assets"} for key, meta in manifest["layers"].items()}


def _asset_exists(asset_id: str) -> bool:
    try:
        ee.data.getAsset(asset_id)
    except ee.EEException:
        return False
    return True


def _resume(fresh: dict[str, Any], manifest_path: Path, iterator) -> tuple[dict[str, Any], list, list]:
    """The manifest an interrupted ``export_weights`` left, reconciled for these parameters:
    (manifest, tasks to poll, shards to dispatch again). Refused when it belongs to other
    parameters or another chunking, or when a shard FAILED or was CANCELLED."""
    manifest = json.loads(Path(manifest_path).read_text())
    for name in ("format", "asset_namespace", "max_collection_payload_bytes", "parameter_count"):
        if manifest.get(name) != fresh[name]:
            raise ValueError(f"the manifest's {name} differs: it is not an upload of these parameters")
    if _layer_metadata(manifest) != _layer_metadata(fresh):
        raise ValueError("the manifest's layers differ: it is not an upload of these parameters")
    if manifest.get("status") not in ("uploading", "ready"):
        raise ValueError(f"the upload is {manifest.get('status')!r}: a failed shard is not resumed")
    polling, redispatch = [], []
    for shard in manifest["shards"]:
        try:
            records, byte_count = next(iterator)
        except StopIteration:
            raise ValueError("the manifest records more shards than these parameters make") from None
        layers = sorted({record["properties"]["layer"] for record in records})
        expected = {"asset_id": f"{fresh['asset_namespace']}_part_{shard['index']:04d}", "rows": len(records),
                    "layers": layers, "collection_payload_bytes": byte_count}
        if any(shard.get(k) != v for k, v in expected.items()):
            raise ValueError(f"shard {shard['index']} differs from the chunking of these parameters")
        state = shard.get("status")
        if state == "COMPLETED":
            continue
        if state in ("FAILED", "CANCELLED"):
            raise ValueError(f"shard {shard['asset_id']} is {state}: it is not resumed")
        if shard.get("task_id"):                       # started: poll it to the end
            polling.append((_RecordedTask(shard["task_id"]), shard))
        elif _asset_exists(shard["asset_id"]):         # recorded "starting", but its export finished
            shard.update(status="COMPLETED", finished_unix=time.time(), resumed="asset found")
        else:                                          # never started: dispatch it again
            redispatch.append((records, byte_count, shard))
    return manifest, polling, redispatch


def export_weights(parameters: Sequence[ConvolutionWeights], asset_namespace: str,
                   manifest_path: Path, *, max_bytes: int = 8_000_000,
                   max_active: int = 2, poll_seconds: float = 15,
                   on_update: Callable[[dict[str, Any]], None] | None = None,
                   resume: bool = False) -> dict[str, Any]:
    """Upload FC shards, waiting for completion with bounded batch concurrency.

    ``asset_namespace`` is a new unique prefix under projects/.../assets.  This
    function never overwrites or deletes assets.  It updates its local manifest
    immediately after each dispatched task and completion, so interrupted runs
    remain auditable.  A failed shard is reported and prevents a ready manifest.

    ``resume=True`` continues the upload that ``manifest_path`` records, for the same
    parameters and chunking (refused otherwise): it polls the recorded tasks to their
    end; a shard recorded "starting" without a task ID is complete if its asset exists
    (one ``getAsset``) and dispatched again if not; the remaining shards follow from the
    same deterministic chunking; then the manifest is finalized. A manifest that is
    already ready is returned as it is. Without ``resume`` the manifest starts afresh.
    """
    if max_active not in (1, 2):
        raise ValueError("Asset upload uses at most two simultaneous exports")
    if not 1 <= poll_seconds <= 30:
        raise ValueError("poll_seconds must be between 1 and 30")
    keys = [parameter.key for parameter in parameters]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate convolution keys")
    manifest: dict[str, Any] = {
        "format": "native_ee_featurecollection_weights_v2", "status": "uploading",
        "asset_namespace": asset_namespace, "started_unix": time.time(),
        "serialization": "JSON strings of numeric chunks; 9 significant decimal digits for float32; float() restoration",
        "dummy_geometry": "Point(0, 0); EE table exports require non-null geometry",
        "max_collection_payload_bytes": max_bytes, "max_active_tasks": max_active,
        "parameter_count": sum(parameter.weights.size + (0 if parameter.bias is None else parameter.bias.size)
                               for parameter in parameters),
        "layers": {parameter.key: {"shape": list(parameter.shape), "dtype": parameter.dtype,
                                   "bias": parameter.bias is not None, "source": parameter.source,
                                   "sha256": parameter.sha256, "assets": []}
                   for parameter in parameters}, "shards": [],
    }
    iterator = iter(upload_chunks(parameters, max_bytes))
    active: list[tuple[Any, dict[str, Any]]] = []
    pending: list[tuple[list[dict[str, Any]], int, dict[str, Any]]] = []
    exhausted = False
    if resume:
        manifest, active, pending = _resume(manifest, Path(manifest_path), iterator)
        if manifest["status"] == "ready":
            return manifest

    def save() -> None:
        _write_manifest(manifest_path, manifest)
        if on_update:
            on_update(manifest)

    def dispatch(records: list[dict[str, Any]], byte_count: int, shard: dict[str, Any] | None) -> None:
        if shard is None:                               # the next shard of the chunking
            index = len(manifest["shards"])
            asset_id = f"{asset_namespace}_part_{index:04d}"
            layers = sorted({record["properties"]["layer"] for record in records})
            shard = {"index": index, "asset_id": asset_id, "rows": len(records),
                     "layers": layers, "collection_payload_bytes": byte_count,
                     "status": "starting", "started_unix": time.time()}
            manifest["shards"].append(shard)
            for key in layers:
                manifest["layers"][key]["assets"].append(asset_id)
        else:                                           # a recorded shard that never started
            shard.update(status="starting", started_unix=time.time())
        save()
        task = ee.batch.Export.table.toAsset(
            collection_from_records(records), description=f"native_unet_weights_part_{shard['index']:04d}",
            assetId=shard["asset_id"], overwrite=False)
        try:
            task.start()
        except Exception as error:
            shard.update(status="FAILED", error=f"{type(error).__name__}: {error}")
            manifest.update(status="failed", finished_unix=time.time())
            # Record the failure, but never let the save's callback (a polling cap, say)
            # replace the start() error.
            with contextlib.suppress(Exception):
                save()
            raise
        shard.update(task_id=task.id, status="READY")
        active.append((task, shard))
        save()

    save()
    while active or pending or not exhausted:
        while (pending or not exhausted) and len(active) < max_active:
            if pending:
                dispatch(*pending.pop(0))
                continue
            try:
                records, byte_count = next(iterator)
            except StopIteration:
                exhausted = True
                break
            dispatch(records, byte_count, None)
        any_finished = False
        for task, shard in list(active):
            status = task.status()
            shard["task_status"] = status
            shard["status"] = status["state"]
            if status["state"] in ("COMPLETED", "FAILED", "CANCELLED"):
                shard["finished_unix"] = time.time()
                active.remove((task, shard))
                any_finished = True
                if status["state"] != "COMPLETED":
                    manifest.update(status="failed", finished_unix=time.time())
                    save()
                    raise RuntimeError(f"Weight shard {shard['asset_id']} failed: {status}")
        save()
        if active and not any_finished:
            time.sleep(poll_seconds)
    manifest.update(status="ready", finished_unix=time.time(),
                    collection_payload_bytes=sum(shard["collection_payload_bytes"] for shard in manifest["shards"]))
    save()
    return manifest


def probe_asset_roundtrip(asset_namespace: str, result_path: Path) -> dict[str, Any]:
    """Live all-element test, including a standard ResNet18 length-4,608 row."""
    rng = np.random.default_rng(47)
    weights = rng.normal(0, .025, (3, 4608)).astype(np.float32)
    bias = np.array([-.03, .0, .03], dtype=np.float32)
    parameter = ConvolutionWeights("probe", weights, bias, "float32", "synthetic_probe")
    manifest_path = result_path.with_name(result_path.stem + "_manifest.json")
    start = time.perf_counter()
    manifest = export_weights([parameter], asset_namespace, manifest_path, poll_seconds=5)
    provider = FeatureCollectionWeightProvider(manifest)
    actual, actual_bias = provider.arrays("probe")
    expected = ee.Array(weights.astype(float).tolist())
    expected_bias = ee.Array(bias.astype(float).reshape(-1, 1).tolist())
    collection = ee.FeatureCollection(manifest["shards"][0]["asset_id"])
    verification = ee.Dictionary({
        "shape": actual.length(), "feature_count": collection.size(), "row_count": collection.filter(ee.Filter.eq("chunk", 0)).size(),
        "max_weight_absolute_error": actual.subtract(expected).abs().reduce(ee.Reducer.max(), [0, 1]),
        "max_bias_absolute_error": actual_bias.subtract(expected_bias).abs().reduce(ee.Reducer.max(), [0, 1]),
    })
    check_start = time.perf_counter()
    raw = verification.getInfo()
    report = {"status": "pass" if (raw["max_weight_absolute_error"] == [[0]] and raw["max_bias_absolute_error"] == [[0]]
                                  and raw["shape"] == [3, 4608] and raw["row_count"] == 3) else "fail",
              "manifest_path": str(manifest_path), "asset_ids": [shard["asset_id"] for shard in manifest["shards"]],
              "row_length": 4608, "source_sha256": parameter.sha256, "verification": raw,
              "verification_seconds": time.perf_counter() - check_start,
              "total_seconds": time.perf_counter() - start,
              "verification_request_bytes": len(ee.serializer.toJSON(verification).encode("utf-8")),
              "serialization": manifest["serialization"]}
    _write_manifest(result_path, report)
    return report


def verify_asset_rows(parameters: Sequence[ConvolutionWeights], manifest: Mapping[str, Any] | str | Path,
                      result_path: Path, *, rows_per_layer: int = 3,
                      on_update: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Check deterministic rows of every server-reconstructed matrix exactly.

    This is a storage and ordering check, separate from model output parity.
    Only scalar errors, shapes, and timing are returned.  The exact source
    fixtures are embedded for selected rows, keeping each query comfortably
    below the request payload cap.  It does not claim a full-asset checksum.
    """
    if rows_per_layer < 1:
        raise ValueError("rows_per_layer must be positive")
    provider = FeatureCollectionWeightProvider(manifest)
    report: dict[str, Any] = {"status": "running", "started_unix": time.time(),
                              "comparison": "deterministic sampled rows from every layer; native server reconstruction",
                              "layers": []}
    _write_manifest(result_path, report)
    for parameter in parameters:
        metadata = provider.manifest["layers"][parameter.key]
        if parameter.sha256 != metadata["sha256"]:
            raise ValueError(f"Source fingerprint mismatch for {parameter.key}")
        weights, bias = provider.arrays(parameter.key)
        indices = sorted(set(int(value) for value in np.linspace(0, parameter.shape[0] - 1,
                                                                  min(rows_per_layer, parameter.shape[0]))))
        row_checks = {}
        for index in indices:
            expected = ee.Array(parameter.weights[index:index + 1].astype(float).tolist())
            difference = weights.slice(0, index, index + 1).subtract(expected).abs()
            row_checks[f"weight_error_row_{index}"] = difference.reduce(ee.Reducer.max(), [0, 1]).get([0, 0])
            if bias is not None:
                row_checks[f"bias_error_row_{index}"] = bias.get([index, 0]).subtract(float(parameter.bias[index])).abs()
        row_checks["shape"] = weights.length()
        request = ee.Dictionary(row_checks)
        started = time.perf_counter()
        raw = request.getInfo()
        errors = [value for key, value in raw.items() if key != "shape"]
        entry = {"key": parameter.key, "rows": indices, "shape": list(parameter.shape),
                 "status": "pass" if raw["shape"] == list(parameter.shape) and all(value == 0 for value in errors) else "fail",
                 "max_absolute_error": max(errors), "evaluation_seconds": time.perf_counter() - started,
                 "request_bytes": len(ee.serializer.toJSON(request).encode()), "server_metrics": raw}
        report["layers"].append(entry)
        _write_manifest(result_path, report)
        if on_update:
            on_update(entry)
    report.update(status="pass" if all(entry["status"] == "pass" for entry in report["layers"]) else "fail",
                  finished_unix=time.time())
    _write_manifest(result_path, report)
    return report
