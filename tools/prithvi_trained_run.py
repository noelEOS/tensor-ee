"""Run the Prithvi Sen1Floods11 crop live in explicit stages, with asset-backed weights.

Run from the checkout with ``PYTHONPATH=.``. The weights are read from a completed
``native_ee_featurecollection_weights_v2`` manifest (``tensor_ee.weight_assets.export_weights``).
Each stage of ``planning.stage_rasters`` is exported to ``<namespace>_<name>`` and read back by
the next stage, so a run is a sequence of batch exports, resumable from ``<run-dir>/state.json``.

    python tools/prithvi_trained_run.py plan --manifest M
    python tools/prithvi_trained_run.py run --manifest M --run-dir D --project P --namespace N \\
        [--until STAGE]
    python tools/prithvi_trained_run.py compare --weights W.npz --run-dir D --project P

``plan`` measures every stage's export request with the manifest's real table expressions and
makes no cloud call. ``run`` first exports the input crop (a seeded dyadic crop, exact in
float32), then each stage in order, at most two exports at a time (a stage's spatial and class
images), waiting for each stage before building the next. It never overwrites an asset.
``compare`` fetches the input and the logits once each and compares the logits with the
plain-torch reference on the fetched input, with the given ``.npz`` state, and writes the
record to ``<run-dir>/compare.json``. The input is synthetic: this checks the arithmetic, not
accuracy.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUT_SEED = 2024
STAGE_LIMIT_BYTES = 9_500_000
"""Export requests above this are refused before submission (the measured stage budget)."""


def _parameters(ee, manifest):
    from tensor_ee.core.tensor import ServerArray
    from tensor_ee.models.prithvi_sen1floods11.configuration import parameter_shapes
    from tensor_ee.weights.assets import FeatureCollectionWeightProvider

    provider = FeatureCollectionWeightProvider(manifest)
    parameters = {}
    for name, shape in parameter_shapes().items():
        recorded = tuple(provider.manifest["layers"][name]["shape"])
        expected = (1, shape[0]) if len(shape) == 1 else (shape[0], int(np.prod(shape[1:])))
        if recorded != expected:
            raise ValueError(f"{name}: manifest shape {recorded}, expected {expected}")
        weight, _ = provider.arrays(name)
        if len(shape) == 1:
            weight = weight.reshape([shape[0]])
            parameters[name] = ServerArray(weight, tuple(shape))
        else:
            parameters[name] = ServerArray(weight, expected)
    return parameters


def input_crop() -> np.ndarray:
    """The run's input: [224, 224, 6] float32 values on a dyadic grid in [-2, 2)."""
    from tools.fixtures.core import dyadic

    return dyadic(INPUT_SEED, "prithvi-trained-input", (224, 224, 6), scale=2.0, bits=12)


def _input_raster(ee):
    from tools.fixtures.core import inline_raster
    from tensor_ee.core.tensor import Raster
    from tensor_ee.models.prithvi_sen1floods11.model import BANDS, INPUT_GRID

    image = inline_raster(ee, INPUT_GRID.projection(ee), input_crop(), BANDS)
    return Raster(image, BANDS, INPUT_GRID)


def _export(ee, raster, asset_id: str, description: str):
    grid = raster.grid
    task = ee.batch.Export.image.toAsset(
        raster.image.select(list(raster.bands)).toFloat(), description=description, assetId=asset_id,
        region=grid.region(ee), crs=grid.crs, crsTransform=list(grid.transform),
        maxPixels=grid.width * grid.height, pyramidingPolicy={".default": "sample"})
    return task


def _request_bytes(ee, task) -> int:
    request = dict(task.config)
    request["expression"] = ee.serializer.encode(request["expression"], for_cloud_api=True)
    return len(json.dumps(request, separators=(",", ":")).encode())


def _stages(ee, manifest, namespace):
    from tensor_ee.models.prithvi_sen1floods11.model import constants
    from tensor_ee.models.prithvi_sen1floods11.planning import stage_rasters

    coefficients = constants(ee, _parameters(ee, manifest))
    return stage_rasters(ee, coefficients, namespace=namespace)


def plan(manifest) -> dict:
    """Every stage's export request size with the real table expressions; no cloud call."""
    import ee
    from tools.live_fixtures import offline_earth_engine

    namespace = "projects/example/assets/prithvi"
    with offline_earth_engine() as attempts:
        stages = [{"stage": "input", "exports": [
            {"name": "input", "request_bytes": _request_bytes(ee, _export(
                ee, _input_raster(ee), namespace + "_input", "prithvi_input"))}]}]
        for stage, images in _stages(ee, manifest, namespace):
            stages.append({"stage": stage, "exports": [
                {"name": name, "request_bytes": _request_bytes(ee, _export(
                    ee, raster, namespace + "_" + name, "prithvi_" + name))}
                for name, raster in images]})
        if attempts:
            raise AssertionError(f"refused cloud attempts: {attempts}")
    sizes = [export["request_bytes"] for stage in stages for export in stage["exports"]]
    return {"stages": stages, "exports": len(sizes), "largest_request_bytes": max(sizes),
            "within_limit": max(sizes) <= STAGE_LIMIT_BYTES, "calls_to_earth_engine": 0}


def _initialize(project: str):
    import ee
    import google.auth

    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/earthengine",
                                                 "https://www.googleapis.com/auth/cloud-platform"])
    ee.Initialize(credentials=credentials, project=project)
    return ee


def _save(path: Path, state: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=1))
    temporary.replace(path)


def _wait(ee, records: list[dict], state: dict, path: Path, poll: float = 30) -> None:
    while True:
        pending = [r for r in records if r["status"] not in ("COMPLETED", "FAILED", "CANCELLED")]
        for record in pending:
            status = ee.data.getTaskStatus(record["task_id"])[0]
            record["status"] = status["state"]
            if status["state"] in ("COMPLETED", "FAILED", "CANCELLED"):
                record["finished_unix"] = time.time()
                record["task_status"] = {k: status.get(k) for k in ("state", "error_message",
                                                                    "batch_eecu_usage_seconds")}
        _save(path, state)
        failed = [r for r in records if r["status"] in ("FAILED", "CANCELLED")]
        if failed:
            raise RuntimeError(f"export failed: {failed}")
        if all(r["status"] == "COMPLETED" for r in records):
            return
        time.sleep(poll)


def run(manifest, run_dir: Path, project: str, namespace: str, until: str | None) -> dict:
    """Export the input, then each stage in order; resume what ``state.json`` records."""
    ee = _initialize(project)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "state.json"
    state = json.loads(path.read_text()) if path.exists() else {
        "namespace": namespace, "input_seed": INPUT_SEED, "stages": {}}
    if state["namespace"] != namespace:
        raise ValueError("the run directory records another namespace")

    def stage_exports(stage, images):
        done = state["stages"].get(stage)
        if done and all(r["status"] == "COMPLETED" for r in done):
            return
        records = done or []
        if not records:
            for name, raster in images:
                task = _export(ee, raster, namespace + "_" + name, "prithvi_" + name)
                size = _request_bytes(ee, task)
                if size > STAGE_LIMIT_BYTES:
                    raise ValueError(f"{name}: request {size} bytes exceeds {STAGE_LIMIT_BYTES}")
                records.append({"name": name, "asset_id": namespace + "_" + name,
                                "request_bytes": size, "status": "SUBMITTING"})
                state["stages"][stage] = records
                _save(path, state)
                task.start()
                records[-1].update(task_id=task.id, status="READY", started_unix=time.time())
                _save(path, state)
        print(time.strftime("%H:%M:%S"), stage, "submitted", [r["name"] for r in records], flush=True)
        _wait(ee, records, state, path)
        print(time.strftime("%H:%M:%S"), stage, "completed",
              [r["task_status"].get("batch_eecu_usage_seconds") for r in records], flush=True)

    stage_exports("input", [("input", _input_raster(ee))])
    if until == "input":
        return state
    for stage, images in _stages(ee, manifest, namespace):
        stage_exports(stage, images)
        if stage == until:
            break
    return state


def _fetch(ee, asset_id: str, bands, grid) -> np.ndarray:
    array = ee.data.computePixels({"expression": ee.Image(asset_id).select(list(bands)),
                                   "fileFormat": "NUMPY_NDARRAY", "grid": grid.pixel_grid()})
    return np.stack([array[band] for band in bands], axis=-1).astype(np.float32)


def compare(weights: Path, run_dir: Path, project: str) -> dict:
    """Fetch the input and logits once each; compare the logits with the torch reference."""
    sys.path.insert(0, str(ROOT / "tests"))
    from reference.prithvi_sen1floods11.reference import forward_torch
    from tensor_ee.core.tensor import band_names
    from tensor_ee.models.prithvi_sen1floods11.configuration import PRITHVI, parameter_shapes
    from tensor_ee.models.prithvi_sen1floods11.model import BANDS, INPUT_GRID

    state = json.loads((run_dir / "state.json").read_text())
    if not all(r["status"] == "COMPLETED" for r in state["stages"].get("logits", [{"status": None}])):
        raise ValueError("the logits stage has not completed")
    ee = _initialize(project)
    namespace = state["namespace"]
    fetched_input = _fetch(ee, namespace + "_input", BANDS, INPUT_GRID)
    logits = _fetch(ee, namespace + "_logits", band_names(2), INPUT_GRID)
    with np.load(weights, allow_pickle=False) as f:
        parameters = {name: np.asarray(f[name], np.float32) for name in parameter_shapes()}
    expected = forward_torch(fetched_input, parameters, heads=PRITHVI.heads, depth=PRITHVI.depth)
    difference = np.abs(logits.astype(np.float64) - expected.astype(np.float64))
    record = {
        "input_equals_local": bool(np.array_equal(fetched_input.view(np.uint32), input_crop().view(np.uint32))),
        "max_abs": float(difference.max()), "over_1e-4": int((difference > 1e-4).sum()),
        "argmax_flips": int((logits.argmax(-1) != expected.argmax(-1)).sum()),
        "pixels": int(logits.shape[0] * logits.shape[1]),
        "logit_range": [float(expected.min()), float(expected.max())],
        "eecu_seconds": sum((r.get("task_status") or {}).get("batch_eecu_usage_seconds") or 0
                            for records in state["stages"].values() for r in records),
        "exports": sum(len(records) for records in state["stages"].values()),
    }
    _save(run_dir / "compare.json", record)
    return record


def main(argv=None) -> None:
    from tensor_ee.converter.request import RECURSION_LIMIT

    # The patch projection gathers 1536 columns; this affects client serialization only.
    sys.setrecursionlimit(max(sys.getrecursionlimit(), RECURSION_LIMIT))
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--manifest", type=Path, required=True)
    r = sub.add_parser("run")
    for name in ("--manifest", "--run-dir"):
        r.add_argument(name, type=Path, required=True)
    r.add_argument("--project", required=True)
    r.add_argument("--namespace", required=True)
    r.add_argument("--until")
    c = sub.add_parser("compare")
    for name in ("--weights", "--run-dir"):
        c.add_argument(name, type=Path, required=True)
    c.add_argument("--project", required=True)
    args = parser.parse_args(argv)
    if args.command == "plan":
        result = plan(json.loads(args.manifest.read_text()))
    elif args.command == "run":
        result = run(json.loads(args.manifest.read_text()), args.run_dir, args.project,
                     args.namespace, args.until)
    else:
        result = compare(args.weights, args.run_dir, args.project)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
