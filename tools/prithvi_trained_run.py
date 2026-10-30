"""Run the Prithvi Sen1Floods11 crop live in explicit stages, with asset-backed weights.

Run from the checkout with ``PYTHONPATH=.``. The weights are read from a completed
``native_ee_featurecollection_weights_v2`` manifest (``tensor_ee.weight_assets.export_weights``).
Each stage of ``planning.stage_rasters`` is exported to ``<namespace>_<name>`` and read back by
the next stage, so a run is a sequence of batch exports, resumable from ``<run-dir>/state.json``.

    python tools/prithvi_trained_run.py plan --manifest M [--cuts C1,C2,..] [--start S]
    python tools/prithvi_trained_run.py run --manifest M --run-dir D --project P --namespace N \\
        [--cuts C1,C2,..] [--start S] [--source-namespace SN] [--until STAGE]
    python tools/prithvi_trained_run.py compare --weights W.npz --run-dir D --project P
    python tools/prithvi_trained_run.py identity --run-dir D --project P --reference-namespace R \
        [--crop I,J]

A region: ``plan`` and ``run`` take ``--crops RxC`` (default 1x1, the single crop),
``--input-seeds S,..`` (one dyadic input seed per crop, row-major; default 2024, 2025, ..),
``--reuse-input A`` (read the crop whose seed is 2024 from the existing input asset A
instead of exporting it) and ``--shard-crops K`` (K x K crops per export shard, default 1).

``--cuts`` names the saved boundaries the run exports (``planning.stage_cuts``); by default
every boundary, one stage each. A stage computes everything since the previous cut in one
expression, so fewer cuts mean fewer, longer stages. ``--start`` names the boundary the first
stage reads, from ``<source-namespace>_<start>`` when given (another run's saved assets),
otherwise from ``<namespace>_<start>``.

``plan`` measures every stage's export request with the manifest's real table expressions and
makes no cloud call. ``run`` first exports the input crop (a seeded dyadic crop, exact in
float32) unless it reads its start from a source namespace or starts after the input, then
each stage in order, at most two exports at a time (a stage's spatial and class images),
waiting for each stage before building the next. It never overwrites an asset, and refuses to
resume a run directory with other cuts, start or source. ``compare`` fetches the input (from
the source namespace when the run has one) and the logits once each and compares the logits
with the plain-torch reference on the fetched input, with the given ``.npz`` state, and writes
the record to ``<run-dir>/compare.json``. The input is synthetic: this checks the arithmetic,
not accuracy. ``identity`` compares each image the run saved with the same-named image of
another run, one ``reduceRegion`` request per image, and writes the largest absolute
difference and the compared pixel count to ``<run-dir>/identity.json``.

A region of R x C crops runs the same stages on ``model.region_grid(R, C)``. Its input is
assembled first: each crop's seeded dyadic input is exported on its own (an inline crop is
3.23 MB of request, so two would not fit one), and one export places the crops side by side
(``changeProj`` translations and a mosaic: values moved, no arithmetic). Every region export
sets ``shardSize`` to K x K crops' extent on its own grid, so a shard's memory does not
grow with the region; single-crop exports keep Earth Engine's default, as before.
``compare`` then checks each crop against the plain-torch reference on its own and reports
the largest absolute error per crop and on the seam ring (the pixels next to an interior
crop edge). ``identity --crop I,J`` compares crop (I, J) of each saved image with a
single-crop run's same-named image, translated into place.
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


def input_crop(seed: int = INPUT_SEED) -> np.ndarray:
    """A crop's input: [224, 224, 6] float32 values on a dyadic grid in [-2, 2)."""
    from tools.fixtures.core import dyadic

    return dyadic(seed, "prithvi-trained-input", (224, 224, 6), scale=2.0, bits=12)


def _input_raster(ee, seed: int = INPUT_SEED):
    from tools.fixtures.core import inline_raster
    from tensor_ee.core.tensor import Raster
    from tensor_ee.models.prithvi_sen1floods11.model import BANDS, INPUT_GRID

    image = inline_raster(ee, INPUT_GRID.projection(ee), input_crop(seed), BANDS)
    return Raster(image, BANDS, INPUT_GRID)


def _export(ee, raster, asset_id: str, description: str, shard: int | None = None):
    grid = raster.grid
    options = {} if shard is None else {"shardSize": shard}
    task = ee.batch.Export.image.toAsset(
        raster.image.select(list(raster.bands)).toFloat(), description=description, assetId=asset_id,
        region=grid.region(ee), crs=grid.crs, crsTransform=list(grid.transform),
        maxPixels=grid.width * grid.height, pyramidingPolicy={".default": "sample"}, **options)
    return task


def is_region(run_layout: dict) -> bool:
    return tuple(run_layout["crops"]) != (1, 1)


def shard_size(grid, run_layout: dict) -> int | None:
    """A region export's shardSize: K x K crops' extent on its grid; None for a single crop."""
    if not is_region(run_layout):
        return None
    rows, _ = run_layout["crops"]
    return grid.height // rows * run_layout["shard_crops"]


def crop_name(row: int, col: int) -> str:
    return f"input_r{row}c{col}"


def input_stages(ee, run_layout: dict, namespace: str):
    """A run's input exports: ``[(stage, [(name, raster, shard), ...]), ...]``.

    One crop: the seeded crop itself (``input``). A region: each crop's seeded input on its
    own (``input_crops``; a crop reused from ``reuse_input`` is not exported), then the
    region (``input``): each crop relabeled to its place and mosaicked, on the region grid.
    """
    from tensor_ee.core.tensor import Raster
    from tensor_ee.layouts.boundaries import pin_boundary
    from tensor_ee.models.prithvi_sen1floods11.model import BANDS, CROP, INPUT_GRID, region_grid

    if not is_region(run_layout):
        return [("input", [("input", _input_raster(ee), None)])]
    rows, cols = run_layout["crops"]
    seeds, reuse = run_layout["input_seeds"], run_layout["reuse_input"]
    sources, exports = [], []
    for index, seed in enumerate(seeds):
        row, col = divmod(index, cols)
        if reuse is not None and seed == INPUT_SEED:
            sources.append(reuse)
        else:
            exports.append((crop_name(row, col), _input_raster(ee, seed), None))
            sources.append(namespace + "_" + crop_name(row, col))
    placed = []
    for index, asset_id in enumerate(sources):
        row, col = divmod(index, cols)
        image = ee.Image(asset_id).select(list(BANDS)).reproject(INPUT_GRID.projection(ee))
        placed.append(image.changeProj(INPUT_GRID.projection(ee),
                                       INPUT_GRID.projection(ee, -CROP * col, -CROP * row)))
    grid = region_grid(rows, cols)
    region = Raster(pin_boundary(ee, ee.ImageCollection(placed).mosaic(), grid), BANDS, grid)
    stages = [("input_crops", exports)] if exports else []
    return stages + [("input", [("input", region, shard_size(grid, run_layout))])]


def _request_bytes(ee, task) -> int:
    request = dict(task.config)
    request["expression"] = ee.serializer.encode(request["expression"], for_cloud_api=True)
    return len(json.dumps(request, separators=(",", ":")).encode())


def _stages(ee, manifest, namespace, layout):
    """``(stage, [(name, raster, shard), ...])`` for every model stage of the layout."""
    from tensor_ee.models.prithvi_sen1floods11.model import constants
    from tensor_ee.models.prithvi_sen1floods11.planning import stage_rasters

    coefficients = constants(ee, _parameters(ee, manifest))
    for stage, images in stage_rasters(ee, coefficients, namespace=namespace, cuts=layout["cuts"],
                                       start=layout["start"], source_namespace=layout["source_namespace"],
                                       crops=tuple(layout["crops"])):
        yield stage, [(name, raster, shard_size(raster.grid, layout)) for name, raster in images]


def layout(cuts=None, start: str = "input", source_namespace: str | None = None, crops=(1, 1),
           input_seeds=None, reuse_input: str | None = None, shard_crops: int = 1) -> dict:
    """The run's validated cut list, start boundary, source namespace and region, as state
    records them. A region's input seeds default to 2024, 2025, .. in row-major crop order."""
    from tensor_ee.models.prithvi_sen1floods11.model import region_grid
    from tensor_ee.models.prithvi_sen1floods11.planning import stage_cuts

    crops = tuple(crops)
    region_grid(*crops)                                   # positive whole numbers of crops
    count = crops[0] * crops[1]
    if crops == (1, 1):
        if input_seeds is not None or reuse_input is not None or shard_crops != 1:
            raise ValueError("--input-seeds, --reuse-input and --shard-crops are for a region")
    else:
        seeds = [INPUT_SEED + k for k in range(count)] if input_seeds is None else list(input_seeds)
        if len(seeds) != count or len(set(seeds)) != count or not all(
                isinstance(s, int) and not isinstance(s, bool) for s in seeds):
            raise ValueError(f"a {crops[0]}x{crops[1]} region needs {count} distinct integer input seeds")
        if reuse_input is not None and INPUT_SEED not in seeds:
            raise ValueError(f"--reuse-input replaces the crop of seed {INPUT_SEED}, which is not in the seeds")
        if isinstance(shard_crops, bool) or not isinstance(shard_crops, int) or shard_crops < 1:
            raise ValueError("--shard-crops must be a positive integer")
        input_seeds = seeds
    return {"cuts": list(stage_cuts(cuts, start=start)), "start": start,
            "source_namespace": source_namespace, "crops": list(crops), "input_seeds": input_seeds,
            "reuse_input": reuse_input, "shard_crops": shard_crops}


def exports_input(run_layout: dict) -> bool:
    """A run exports its own input crop unless it reads its start from elsewhere."""
    return run_layout["start"] == "input" and run_layout["source_namespace"] is None


def plan(manifest, run_layout: dict | None = None) -> dict:
    """Every stage's export request size with the real table expressions; no cloud call."""
    import ee
    from tools.live_fixtures import offline_earth_engine

    run_layout = layout() if run_layout is None else run_layout
    namespace = "projects/example/assets/prithvi"
    def sized(stage, images):
        return {"stage": stage, "exports": [
            {"name": name, "request_bytes": _request_bytes(ee, _export(
                ee, raster, namespace + "_" + name, "prithvi_" + name, shard)),
             **({} if shard is None else {"shard_size": shard})}
            for name, raster, shard in images]}

    with offline_earth_engine() as attempts:
        stages = [sized(stage, images) for stage, images in input_stages(ee, run_layout, namespace)
                  ] if exports_input(run_layout) else []
        stages += [sized(stage, images) for stage, images in _stages(ee, manifest, namespace, run_layout)]
        if attempts:
            raise AssertionError(f"refused cloud attempts: {attempts}")
    sizes = [export["request_bytes"] for stage in stages for export in stage["exports"]]
    return {**run_layout, "stages": stages, "sequential_stages": len(stages),
            "exports": len(sizes), "largest_request_bytes": max(sizes),
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


def _recorded_layout(state: dict) -> dict:
    """The layout a state records; runs recorded before cuts existed used every boundary,
    and runs recorded before regions existed one crop."""
    return {"cuts": state.get("cuts", list(layout()["cuts"])), "start": state.get("start", "input"),
            "source_namespace": state.get("source_namespace"), "crops": state.get("crops", [1, 1]),
            "input_seeds": state.get("input_seeds"), "reuse_input": state.get("reuse_input"),
            "shard_crops": state.get("shard_crops", 1)}


def _open_state(path: Path, namespace: str, run_layout: dict) -> dict:
    """The run directory's state, new or resumed; a resume must match namespace and layout."""
    state = json.loads(path.read_text()) if path.exists() else {
        "namespace": namespace, "input_seed": INPUT_SEED, **run_layout, "stages": {}}
    if state["namespace"] != namespace:
        raise ValueError("the run directory records another namespace")
    if _recorded_layout(state) != run_layout:
        raise ValueError(f"the run directory records another layout: {_recorded_layout(state)}")
    return state


def run(manifest, run_dir: Path, project: str, namespace: str, until: str | None,
        run_layout: dict | None = None) -> dict:
    """Export the input, then each stage in order; resume what ``state.json`` records."""
    run_layout = layout() if run_layout is None else run_layout
    if until not in (None, "input", *run_layout["cuts"]):
        raise ValueError(f"--until must name one of the cuts {run_layout['cuts']}")
    if until == "input" and not exports_input(run_layout):
        raise ValueError("this run does not export its input")
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "state.json"
    state = _open_state(path, namespace, run_layout)
    ee = _initialize(project)

    def stage_exports(stage, images):
        done = state["stages"].get(stage)
        if done and all(r["status"] == "COMPLETED" for r in done):
            return
        records = done or []
        if not records:
            for name, raster, shard in images:
                task = _export(ee, raster, namespace + "_" + name, "prithvi_" + name, shard)
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

    if exports_input(run_layout):
        for stage, images in input_stages(ee, run_layout, namespace):
            stage_exports(stage, images)
        if until == "input":
            return state
    for stage, images in _stages(ee, manifest, namespace, run_layout):
        stage_exports(stage, images)
        if stage == until:
            break
    return state


def _fetch(ee, asset_id: str, bands, grid) -> np.ndarray:
    array = ee.data.computePixels({"expression": ee.Image(asset_id).select(list(bands)),
                                   "fileFormat": "NUMPY_NDARRAY", "grid": grid.pixel_grid()})
    return np.stack([array[band] for band in bands], axis=-1).astype(np.float32)


def seam_ring(height: int, width: int, crop: int = 224) -> np.ndarray:
    """[H, W] True on the pixels next to an interior crop edge (where a padding error shows)."""
    def axis(n):
        index = np.arange(n)
        edge = (index % crop == 0) | (index % crop == crop - 1)
        return edge & (index != 0) & (index != n - 1)
    return axis(height)[:, None] | axis(width)[None, :]


def crop_comparison(fetched_input, logits, reference, seeds, crops) -> dict:
    """Each crop's logits against ``reference(crop_input)`` (the plain-torch forward of that
    crop alone), with the fetched input checked against the crop's seed; no cloud call."""
    rows, cols = crops
    expected = np.empty(logits.shape, np.float64)
    per_crop = {}
    for index, seed in enumerate(seeds):
        row, col = divmod(index, cols)
        window = (slice(224 * row, 224 * row + 224), slice(224 * col, 224 * col + 224))
        crop_input = fetched_input[window]
        expected[window] = reference(crop_input)
        error = np.abs(logits[window].astype(np.float64) - expected[window])
        per_crop[f"{row},{col}"] = {
            "seed": seed, "input_equals_local": bool(np.array_equal(
                crop_input.view(np.uint32), input_crop(seed).view(np.uint32))),
            "max_abs": float(error.max()), "argmax_flips": int(
                (logits[window].argmax(-1) != expected[window].argmax(-1)).sum())}
    difference = np.abs(logits.astype(np.float64) - expected)
    ring = seam_ring(*logits.shape[:2])
    return {
        "input_equals_local": all(c["input_equals_local"] for c in per_crop.values()),
        "max_abs": float(difference.max()), "over_1e-4": int((difference > 1e-4).sum()),
        "argmax_flips": int((logits.argmax(-1) != expected.argmax(-1)).sum()),
        "pixels": int(logits.shape[0] * logits.shape[1]),
        "logit_range": [float(expected.min()), float(expected.max())],
        **({} if (rows, cols) == (1, 1) else {
            "crops": per_crop, "seam_max_abs": float(difference[ring].max()),
            "seam_pixels": int(ring.sum())}),
    }


def compare(weights: Path, run_dir: Path, project: str) -> dict:
    """Fetch the input and logits once each; compare each crop's logits with the torch
    reference on that crop's fetched input."""
    sys.path.insert(0, str(ROOT / "tests"))
    from reference.prithvi_sen1floods11.reference import forward_torch
    from tensor_ee.core.tensor import band_names
    from tensor_ee.models.prithvi_sen1floods11.configuration import PRITHVI, parameter_shapes
    from tensor_ee.models.prithvi_sen1floods11.model import BANDS, region_grid

    state = json.loads((run_dir / "state.json").read_text())
    if not all(r["status"] == "COMPLETED" for r in state["stages"].get("logits", [{"status": None}])):
        raise ValueError("the logits stage has not completed")
    run_layout = _recorded_layout(state)
    crops = tuple(run_layout["crops"])
    seeds = run_layout["input_seeds"] or [state.get("input_seed", INPUT_SEED)]
    grid = region_grid(*crops)
    ee = _initialize(project)
    namespace = state["namespace"]
    source = state.get("source_namespace") or namespace
    fetched_input = _fetch(ee, source + "_input", BANDS, grid)
    logits = _fetch(ee, namespace + "_logits", band_names(2), grid)
    with np.load(weights, allow_pickle=False) as f:
        parameters = {name: np.asarray(f[name], np.float32) for name in parameter_shapes()}
    record = crop_comparison(
        fetched_input, logits,
        lambda crop: forward_torch(crop, parameters, heads=PRITHVI.heads, depth=PRITHVI.depth),
        seeds, crops)
    record.update(
        eecu_seconds=sum((r.get("task_status") or {}).get("batch_eecu_usage_seconds") or 0
                         for records in state["stages"].values() for r in records),
        exports=sum(len(records) for records in state["stages"].values()))
    _save(run_dir / "compare.json", record)
    return record


def boundary(name: str, config=None, crops=(1, 1)):
    """A saved image's bands and grid, from its name (``block3_class``, ``upsample56``, ..),
    for a single crop or a region of ``crops``."""
    from tensor_ee.core.tensor import band_names
    from tensor_ee.models.prithvi_sen1floods11.configuration import PRITHVI
    from tensor_ee.models.prithvi_sen1floods11.model import BANDS, region_grid

    config = PRITHVI if config is None else config
    input_grid = region_grid(*crops)
    token_grid = input_grid.coarse(16)
    if name == "input":
        return BANDS, input_grid
    if name == "logits":
        return tuple(band_names(2)), input_grid
    if name.endswith("_spatial"):
        return tuple(band_names(config.width)), token_grid
    if name.endswith("_class"):
        return tuple(band_names(config.width)), token_grid.coarse(14)
    if name.startswith("upsample"):
        return tuple(band_names(config.width)), input_grid.coarse(224 // int(name[8:]))
    raise ValueError(f"not a saved image: {name!r}")


def difference(ee, asset_id: str, reference_id: str, bands, grid, *, reference_grid=None,
               offset=(0, 0)):
    """One request's expression: the largest |a - b| over all bands and pixels, and the count.

    With ``offset`` (rows, cols) of pixels, the reference, an image on ``reference_grid``,
    is relabeled that far down and right first (a single crop compared with a region's
    crop); the count is then the reference's pixels."""
    a = ee.Image(asset_id).select(list(bands))
    b = ee.Image(reference_id).select(list(bands))
    if tuple(offset) != (0, 0):
        rows, cols = offset
        b = b.changeProj(reference_grid.projection(ee), reference_grid.projection(ee, -cols, -rows))
    largest = a.subtract(b).abs().reduce(ee.Reducer.max()).rename("difference")
    return largest.reduceRegion(
        reducer=ee.Reducer.max().combine(ee.Reducer.count(), sharedInputs=True),
        geometry=grid.region(ee), crs=grid.crs, crsTransform=list(grid.transform),
        maxPixels=grid.width * grid.height)


def _identity_records(state: dict, crop=None):
    """The run's crops and its completed model-stage records; a crop must be in the region."""
    crops = tuple(_recorded_layout(state)["crops"])
    if crop is not None and not (0 <= crop[0] < crops[0] and 0 <= crop[1] < crops[1]):
        raise ValueError(f"crop {crop} is outside the {crops[0]}x{crops[1]} region")
    records = [r for stage, rs in state["stages"].items() if stage not in ("input", "input_crops")
               for r in rs]
    if not records or not all(r["status"] == "COMPLETED" for r in records):
        raise ValueError("the run has no completed stages, or some have not completed")
    return crops, records


def identity_requests(ee, state: dict, reference_namespace: str, crop=None) -> list[tuple]:
    """``(name, asset_id, reference_id, expression, expected_pixels)`` per saved model image.

    Without ``crop`` the reference run has the same crops. With ``crop`` = (I, J) the
    reference is a single-crop run, compared with crop (I, J) of each image."""
    crops, records = _identity_records(state, crop)
    requests = []
    for record in records:
        bands, grid = boundary(record["name"], crops=crops)
        reference_id = reference_namespace + "_" + record["name"]
        if crop is None:
            expression, expected = difference(ee, record["asset_id"], reference_id, bands, grid), \
                grid.width * grid.height
        else:
            _, single = boundary(record["name"])
            offset = (crop[0] * single.height, crop[1] * single.width)
            expression = difference(ee, record["asset_id"], reference_id, bands, grid,
                                    reference_grid=single, offset=offset)
            expected = single.width * single.height
        requests.append((record["name"], record["asset_id"], reference_id, expression, expected))
    return requests


def identity(run_dir: Path, project: str, reference_namespace: str, crop=None) -> dict:
    """Compare every image the run saved with the reference run's same-named image, or,
    with ``crop``, crop (I, J) of it with a single-crop run's image."""
    state = json.loads((run_dir / "state.json").read_text())
    _identity_records(state, crop)                        # refusals before any sign-in
    ee = _initialize(project)
    images = {}
    for name, asset_id, reference_id, expression, expected in identity_requests(
            ee, state, reference_namespace, crop):
        result = expression.getInfo()
        images[name] = {
            "asset_id": asset_id, "reference_id": reference_id,
            "max_abs": result["difference_max"], "pixels": result["difference_count"],
            "expected_pixels": expected,
        }
    report = {"crop": None if crop is None else list(crop), "images": images, "identical": all(
        image["max_abs"] == 0 and image["pixels"] == image["expected_pixels"]
        for image in images.values())}
    _save(run_dir / "identity.json", report)
    return report


def _layout_arguments(parser) -> None:
    parser.add_argument("--cuts", type=lambda text: tuple(text.split(",")),
                        help="comma-separated saved boundaries to export (default: every one)")
    parser.add_argument("--start", default="input", help="the boundary the first stage reads")
    parser.add_argument("--crops", type=_pair("x"), default=(1, 1),
                        help="RxC crops of 224 x 224 in a region (default 1x1: one crop)")
    parser.add_argument("--input-seeds", type=lambda text: [int(s) for s in text.split(",")],
                        help="a region's input seed per crop, row-major (default 2024, 2025, ..)")
    parser.add_argument("--reuse-input", help="read the crop of seed 2024 from this input asset")
    parser.add_argument("--shard-crops", type=int, default=1, help="K x K crops per export shard")


def _pair(separator: str):
    def parse(text: str) -> tuple[int, int]:
        first, second = text.split(separator)
        return int(first), int(second)
    return parse


def _layout_of(args) -> dict:
    return layout(args.cuts, args.start, getattr(args, "source_namespace", None), args.crops,
                  args.input_seeds, args.reuse_input, args.shard_crops)


def main(argv=None) -> None:
    from tensor_ee.converter.request import RECURSION_LIMIT

    # The patch projection gathers 1536 columns; this affects client serialization only.
    sys.setrecursionlimit(max(sys.getrecursionlimit(), RECURSION_LIMIT))
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--manifest", type=Path, required=True)
    _layout_arguments(p)
    r = sub.add_parser("run")
    for name in ("--manifest", "--run-dir"):
        r.add_argument(name, type=Path, required=True)
    r.add_argument("--project", required=True)
    r.add_argument("--namespace", required=True)
    _layout_arguments(r)
    r.add_argument("--source-namespace", help="read the start boundary from this run's assets")
    r.add_argument("--until")
    c = sub.add_parser("compare")
    for name in ("--weights", "--run-dir"):
        c.add_argument(name, type=Path, required=True)
    c.add_argument("--project", required=True)
    i = sub.add_parser("identity")
    i.add_argument("--run-dir", type=Path, required=True)
    i.add_argument("--project", required=True)
    i.add_argument("--reference-namespace", required=True)
    i.add_argument("--crop", type=_pair(","), help="I,J: compare this crop with a single-crop run")
    args = parser.parse_args(argv)
    if args.command == "plan":
        result = plan(json.loads(args.manifest.read_text()), _layout_of(args))
    elif args.command == "run":
        result = run(json.loads(args.manifest.read_text()), args.run_dir, args.project,
                     args.namespace, args.until, _layout_of(args))
    elif args.command == "identity":
        result = identity(args.run_dir, args.project, args.reference_namespace, args.crop)
    else:
        result = compare(args.weights, args.run_dir, args.project)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
