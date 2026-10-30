"""Prepare one real Sentinel-2 L1C crop for the staged Prithvi Sen1Floods11 run, and check it.

Run from the checkout with ``PYTHONPATH=.``. The crop is a 224 x 224 window of a chip grid,
given as arguments (nothing about a particular scene lives in this repository):
``--crs`` and ``--chip-transform a,0,x0,0,e,y0`` (the chip's north-up affine, pixel corners)
and ``--offset ROW,COL`` (the window's first pixel in the chip).

    python tools/prithvi_real_input.py discover --date YYYY-MM-DD --crs C --chip-transform T \\
        --offset R,C --project P
    python tools/prithvi_real_input.py prepare --image ID --crs C --chip-transform T --offset R,C \\
        --run-dir D --namespace N --project P [--plan]
    python tools/prithvi_real_input.py check-input --image ID --crs C --chip-transform T \\
        --offset R,C --run-dir D --project P
    python tools/prithvi_real_input.py map --run-dir D --label LABEL.npy --offset R,C

- ``discover`` (one request) lists the ``COPERNICUS/S2_HARMONIZED`` images of that day whose
  footprint contains the crop, with ``PRODUCT_ID``, ``MGRS_TILE`` and
  ``CLOUDY_PIXEL_PERCENTAGE``; pass the chosen one's id (``COPERNICUS/S2_HARMONIZED/<index>``)
  to the others.
- ``prepare`` (one batch export) writes ``<namespace>_input``: the six source bands,
  zero-filled, ``standardize``d (``models.prithvi_sen1floods11.preprocessing``), relabeled
  onto the model's ``INPUT_GRID``. ``--plan`` builds the export offline and prints its
  request size instead. The staged run then reads it as its input:
  ``prithvi_trained_run.py run --start input --source-namespace <namespace>``.
- ``compare --real-input`` (in ``prithvi_trained_run.py``, two requests) checks the logits
  against PyTorch on the fetched input and keeps ``input.npy`` and ``logits.npy``.
- ``check-input`` (one request) fetches the raw bands, zero-filled and relabeled exactly as
  ``prepare`` feeds them to ``standardize``, with the QA60 band, and compares the fetched
  prepared input (``input.npy``) with (a) the block's NumPy mirror and (b) the source's
  recipe (NumPy float32 ``x * 1e-4``, then ``sub_(mean).div_(std)`` on a float32 tensor),
  both bit for bit, and counts zero-filled pixels and QA60 cloud flags.
- ``map`` (no request) writes the predicted water mask (logits argmax), and with a label
  (an ``int16`` ``.npy`` of the chip: -1 no data, 0 not water, 1 water) the water IoU and
  accuracy over labeled pixels, and a PNG of RGB, prediction, label and agreement. It is
  one crop: no accuracy claim beyond it.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COLLECTION = "COPERNICUS/S2_HARMONIZED"
CROP = 224
PROPERTIES = ("system:index", "PRODUCT_ID", "MGRS_TILE", "CLOUDY_PIXEL_PERCENTAGE")


def crop_grid(crs: str, chip_transform, offset):
    """The 224 x 224 window at ``offset`` (row, col) of a chip grid, as a north-up ``Grid``."""
    from tensor_ee.core.grid import Grid

    a, b, x0, d, e, y0 = (float(v) for v in chip_transform)
    row, col = (int(v) for v in offset)
    if row < 0 or col < 0:
        raise ValueError("the offset must be nonnegative")
    return Grid(crs, (a, b, x0 + col * a, d, e, y0 + row * e), CROP, CROP)


def discover_expression(ee, date: str, grid):
    """One request: the day's L1C images whose footprint contains the crop, as a list of
    [index, product id, tile, cloud percentage]."""
    start = ee.Date(date)
    images = (ee.ImageCollection(COLLECTION).filterDate(start, start.advance(1, "day"))
              .filter(ee.Filter.contains(leftField=".geo", rightValue=grid.geometry())))
    return images.reduceColumns(ee.Reducer.toList(len(PROPERTIES)), list(PROPERTIES)).get("list")


def _image(ee, image_id: str):
    if not image_id.startswith(COLLECTION + "/"):
        raise ValueError(f"--image must be an image of {COLLECTION}: {COLLECTION}/<system:index>")
    return ee.Image(image_id)


def prepared_input(ee, image_id: str, grid):
    """The run's input: prepared on the crop grid, relabeled onto ``INPUT_GRID``."""
    from tensor_ee.models.prithvi_sen1floods11.preprocessing import prepare, to_input_grid

    return to_input_grid(ee, prepare(_image(ee, image_id), grid))


def raw_input(ee, image_id: str, grid):
    """What ``standardize`` reads (the zero-filled six bands), plus QA60, on ``INPUT_GRID``:
    bands ``b0``..``b5`` and ``qa60``."""
    from tensor_ee.core.tensor import Raster
    from tensor_ee.layouts.raster import from_image
    from tensor_ee.models.prithvi_sen1floods11.preprocessing import raw_bands, to_input_grid

    image = _image(ee, image_id)
    raw = to_input_grid(ee, raw_bands(image, grid))
    qa = to_input_grid(ee, from_image(image, ("QA60",), grid))
    bands = raw.bands + ("qa60",)
    return Raster(raw.image.addBands(qa.image.rename(["qa60"])), bands, raw.grid)


def _rt():
    from tools import prithvi_trained_run as RUN

    return RUN


def prepare(image_id: str, grid, run_dir: Path, namespace: str, project: str | None, plan_only: bool) -> dict:
    """Export ``<namespace>_input`` (or, with ``plan_only``, size its request offline)."""
    RUN = _rt()
    asset_id = namespace + "_input"
    if plan_only:
        import ee
        from tools.live_fixtures import offline_earth_engine

        with offline_earth_engine() as attempts:
            task = RUN._export(ee, prepared_input(ee, image_id, grid), asset_id, "prithvi_real_input")
            size = RUN._request_bytes(ee, task)
        if attempts:
            raise AssertionError(f"refused cloud attempts: {attempts}")
        return {"asset_id": asset_id, "request_bytes": size, "within_limit": size <= RUN.STAGE_LIMIT_BYTES,
                "calls_to_earth_engine": 0}
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "prepare.json"
    record = {"image": image_id, "crs": grid.crs, "transform": list(grid.transform),
              "asset_id": asset_id, "status": "SUBMITTING"}
    if path.exists():
        previous = json.loads(path.read_text())
        if {k: previous[k] for k in ("image", "crs", "transform", "asset_id")} != \
                {k: record[k] for k in ("image", "crs", "transform", "asset_id")}:
            raise ValueError("the run directory records another prepared input")
        if previous["status"] == "COMPLETED":
            return previous
        record = previous
    ee = RUN._initialize(project)
    if "task_id" not in record:
        task = RUN._export(ee, prepared_input(ee, image_id, grid), asset_id, "prithvi_real_input")
        record["request_bytes"] = RUN._request_bytes(ee, task)
        if record["request_bytes"] > RUN.STAGE_LIMIT_BYTES:
            raise ValueError(f"request {record['request_bytes']} bytes exceeds {RUN.STAGE_LIMIT_BYTES}")
        RUN._save(path, record)
        task.start()
        record.update(task_id=task.id, status="READY")
        RUN._save(path, record)
    RUN._wait(ee, [record], record, path)
    return record


def source_recipe(raw: np.ndarray) -> np.ndarray:
    """The source's float32 preparation of zero-filled raw bands: NumPy ``x * 1e-4`` (a
    float32 array times a Python float), then ``tensor.sub_(mean).div_(std)``."""
    import torch
    from tensor_ee.models.prithvi_sen1floods11.preprocessing import MEANS, SCALE, STDS

    x = np.asarray(raw, np.float32) * SCALE
    tensor = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).clone()
    with torch.no_grad():
        tensor.sub_(torch.as_tensor(MEANS, dtype=torch.float32)[:, None, None])
        tensor.div_(torch.as_tensor(STDS, dtype=torch.float32)[:, None, None])
    return tensor.permute(1, 2, 0).contiguous().numpy()


def input_check(raw: np.ndarray, qa60: np.ndarray, prepared: np.ndarray) -> dict:
    """The fetched prepared input against the block's mirror and the source recipe, both
    applied to the fetched raw bands; no cloud call."""
    sys.path.insert(0, str(ROOT / "tests"))
    from reference.raster_standardize import standardize
    from tensor_ee.models.prithvi_sen1floods11.preprocessing import MEANS, SCALE, STDS

    mirror = standardize(raw, np.array(MEANS, np.float32), np.array(STDS, np.float32), SCALE)
    recipe = source_recipe(raw)

    def agreement(expected):
        return {"bit_equal": bool(np.array_equal(prepared.view(np.uint32), expected.view(np.uint32))),
                "values_differing": int((prepared.view(np.uint32) != expected.view(np.uint32)).sum()),
                "max_abs": float(np.abs(prepared.astype(np.float64) - expected.astype(np.float64)).max())}

    clouds = qa60.astype(np.int64)
    return {"mirror": agreement(mirror), "source_recipe": agreement(recipe),
            "zero_filled_per_band": [int(v) for v in (raw == 0).sum(axis=(0, 1))],
            "qa60_opaque_fraction": float(((clouds >> 10) & 1).mean()),
            "qa60_cirrus_fraction": float(((clouds >> 11) & 1).mean()),
            "pixels": int(raw.shape[0] * raw.shape[1])}


def check_input(image_id: str, grid, run_dir: Path, project: str) -> dict:
    """Fetch the raw bands once (``raw.npy``) and compare with ``input.npy`` from
    ``compare --real-input``."""
    RUN = _rt()
    prepared_path = run_dir / "input.npy"
    if not prepared_path.exists():
        raise ValueError("run prithvi_trained_run.py compare --real-input first: it saves input.npy")
    raw_path = run_dir / "raw.npy"
    if not raw_path.exists():
        ee = RUN._initialize(project)
        raster = raw_input(ee, image_id, grid)
        np.save(raw_path, RUN._fetch(ee, raster.image, raster.bands, raster.grid))
    fetched = np.load(raw_path)
    record = input_check(fetched[..., :6], fetched[..., 6], np.load(prepared_path))
    RUN._save(run_dir / "check_input.json", record)
    return record


def iou(prediction: np.ndarray, label: np.ndarray) -> dict:
    """Water IoU and accuracy over labeled pixels (label -1 is no data)."""
    valid = label != -1
    water, truth = prediction[valid] == 1, label[valid] == 1
    union = int((water | truth).sum())
    return {"water_iou": float((water & truth).sum() / union) if union else None,
            "accuracy": float((water == truth).mean()) if valid.any() else None,
            "labeled_pixels": int(valid.sum()), "predicted_water": float(water.mean()) if valid.any() else None,
            "labeled_water": float(truth.mean()) if valid.any() else None}


def _png(path: Path, rgb: np.ndarray) -> None:
    """An 8-bit RGB PNG, with the standard library only."""
    height, width, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[row].astype(np.uint8).tobytes() for row in range(height))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def map_panels(raw: np.ndarray | None, prediction: np.ndarray, label: np.ndarray | None) -> np.ndarray:
    """RGB (B4, B3, B2, DN 0..3000), prediction, label and agreement, side by side."""
    blue, black, grey = np.array([30, 90, 220]), np.array([0, 0, 0]), np.array([128, 128, 128])
    panels = []
    if raw is not None:
        panels.append(np.clip(raw[..., [2, 1, 0]] / 3000 * 255, 0, 255))
    panels.append(np.where((prediction == 1)[..., None], blue, black))
    if label is not None:
        panels.append(np.where((label == 1)[..., None], blue, np.where((label == -1)[..., None], grey, black)))
        agree = np.select([(label == -1), (prediction == 1) & (label == 1), (prediction == 1) & (label == 0),
                           (prediction == 0) & (label == 1)],
                          [1, 2, 3, 4], 0)[..., None]
        panels.append(np.select([agree == 1, agree == 2, agree == 3, agree == 4],
                                [grey, np.array([40, 180, 60]), np.array([220, 50, 50]), np.array([240, 160, 30])],
                                np.array([20, 20, 20])))
    return np.concatenate([p.astype(np.uint8) for p in panels], axis=1)


def flood_map(run_dir: Path, label_path: Path | None, offset) -> dict:
    """The water mask and, with a label, its IoU on the crop; writes map.json and map.png."""
    logits = np.load(run_dir / "logits.npy")
    prediction = logits.argmax(-1)
    label = None
    if label_path is not None:
        chip = np.load(label_path)
        row, col = (int(v) for v in offset)
        label = chip[row:row + CROP, col:col + CROP].astype(np.int64)
        if label.shape != prediction.shape:
            raise ValueError(f"the label window at {offset} is {label.shape}, not {prediction.shape}")
    raw_path = run_dir / "raw.npy"
    raw = np.load(raw_path)[..., :6] if raw_path.exists() else None
    record = {"crop": [CROP, CROP], "predicted_water": float((prediction == 1).mean()),
              **({} if label is None else {"label": iou(prediction, label)}),
              "claim": "one 224 x 224 crop, a single forward pass, no slide inference"}
    _png(run_dir / "map.png", map_panels(raw, prediction, label))
    _rt()._save(run_dir / "map.json", record)
    return record


def _numbers(count: int, kind=float):
    def parse(text: str):
        values = [kind(v) for v in text.split(",")]
        if len(values) != count:
            raise argparse.ArgumentTypeError(f"expected {count} comma-separated values")
        return values
    return parse


def main(argv=None) -> None:
    from tensor_ee.converter.request import RECURSION_LIMIT

    sys.setrecursionlimit(max(sys.getrecursionlimit(), RECURSION_LIMIT))
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def crop_arguments(p):
        p.add_argument("--crs", required=True)
        p.add_argument("--chip-transform", type=_numbers(6), required=True)
        p.add_argument("--offset", type=_numbers(2, int), required=True)

    d = sub.add_parser("discover")
    crop_arguments(d)
    d.add_argument("--date", required=True)
    d.add_argument("--project", required=True)
    p = sub.add_parser("prepare")
    crop_arguments(p)
    p.add_argument("--image", required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--namespace", required=True)
    p.add_argument("--project")
    p.add_argument("--plan", action="store_true", help="size the export offline; no cloud call")
    c = sub.add_parser("check-input")
    crop_arguments(c)
    c.add_argument("--image", required=True)
    c.add_argument("--run-dir", type=Path, required=True)
    c.add_argument("--project", required=True)
    m = sub.add_parser("map")
    m.add_argument("--run-dir", type=Path, required=True)
    m.add_argument("--label", type=Path)
    m.add_argument("--offset", type=_numbers(2, int), required=True)
    args = parser.parse_args(argv)
    if args.command == "map":
        result = flood_map(args.run_dir, args.label, args.offset)
    else:
        grid = crop_grid(args.crs, args.chip_transform, args.offset)
        if args.command == "discover":
            ee = _rt()._initialize(args.project)
            result = {"images": [dict(zip(PROPERTIES, row)) for row in discover_expression(ee, args.date, grid).getInfo()]}
        elif args.command == "prepare":
            if not args.plan and not args.project:
                raise SystemExit("prepare needs --project unless --plan")
            result = prepare(args.image, grid, args.run_dir, args.namespace, args.project, args.plan)
        else:
            result = check_input(args.image, grid, args.run_dir, args.project)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
