"""Serve the SegWater browser demo; all model forward operations run in Earth Engine.

Run ``python -m demos.browser.server --project YOUR_PROJECT``, then open the URL it
prints, ``http://127.0.0.1:8765/#token=...`` (``--model swin-t``: port 8766); the token in
the fragment authorizes runs and never reaches the server. This loopback-only UI
controller builds native EE graphs with the ``tensor_ee`` library, submits batch
exports and serves their status. It never runs a local model forward pass or
downloads inference input arrays. ``--offline`` serves the same UI against a local
fixture with no Earth Engine requests (``demos/browser/offline.py``).

Ported from the archived demo (``demo/server.py`` at the tag
``research-archive-2026-10-12``): the orchestration is the same, the research
imports are replaced by the library and ``demos.browser.exports``, and the
command line has no personal defaults.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import sys
import threading
import time
from urllib.parse import urlsplit
import uuid

from tensor_ee import Grid

from demos.browser.exports import (atomic_write, digest, initialize, native_operation_names,
                                   prepared_image_request, submit)
from demos.browser.geometry import aoi_bounds, date_range, grid_from_projected_bounds, utm_crs, validate_aoi

ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).parent / "static"
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED"}
PROBABILITY_PALETTE = ["f7fcff", "c6dbef", "6baed6", "2171b5", "08306b"]
# Documented defaults, relative to the repository root. Each one is overridable.
DEFAULTS = {
    "unet": dict(port=8765, max_tiles=4, state_dir="demos/browser/state",
                 checkpoint="checkpoints/unet_resnet18_s42_step6000_last.pth",
                 manifest="data/manifests/segwater_r18_weight_manifest.json"),
    "swin-t": dict(port=8766, max_tiles=1, state_dir="demos/browser/state/swin-t",
                   checkpoint="downloads/m12_ckpt/upernet_tu-swin_tiny_patch4_window7_224_s42_step6000_last.pth",
                   manifest="data/manifests/m12_trained_weights_manifest.json"),
}
REQUEST_BUDGET_BYTES = 9_500_000
EXAMPLE_MESSAGE = "Completed example · a previous result, not a new run."
GENERIC_ERROR = "The request failed; see the server console."


def page_url(port: int, token: str) -> str:
    """The URL the reader opens: the session token travels in the fragment, which the browser
    never sends to the server; the page reads it and sends it as X-Demo-Token on every POST."""
    return f"http://127.0.0.1:{port}/#token={token}"


def rectangle(west, south, east, north):
    return dict(type="Polygon", coordinates=[[[west, south], [east, south], [east, north],
                                              [west, north], [west, south]]])


def grid_metadata(grid):
    return dict(crs=grid.crs, transform=list(grid.transform), width=grid.width, height=grid.height,
                tile_count=(grid.width//224)*(grid.height//224), area_km2=grid.width*grid.height/10000)


def load_example(path, model):
    """Read a completed result to show as a clearly labelled previous example.

    Accepts a completed job record of this demo (``<state-dir>/<job>/job.json``),
    or one of the two verified history records
    (docs/history/cnn/results/segwater_native_mosaic_2x2.json for the U-Net,
    docs/history/cnn/results/swin_demo_example_checks.json for Swin-T). The asset must be readable by the demo's project. Nothing is
    fetched here; the map layers are requested when the example is opened.
    """
    record = json.loads(Path(path).read_text())
    if not isinstance(record, dict):
        raise ValueError("The example must be a JSON object.")
    if "identity" in record and "export" in record:        # the U-Net mosaic history record
        config = record["identity"]["config"]
        record = dict(model="unet", status=record.get("status"), asset_id=record["export"]["asset_id"],
                      scene_id=config["scene_id"], grid={k: config[k] for k in ("crs", "transform", "width", "height")},
                      elapsed_seconds=record["export"].get("elapsed_seconds"),
                      ee_runtime_seconds=record["export"].get("ee_runtime_seconds"))
    if record.get("status", "pass") != "pass" or record.get("state", "COMPLETED") != "COMPLETED":
        raise ValueError("The example must be a completed, passing result.")
    if record.get("offline"):
        raise ValueError("An offline fixture job cannot be shown as an example.")
    if record.get("model", model) != model:
        raise ValueError(f"The example is a {record.get('model')} result; this demo runs {model}.")
    grid = record.get("grid")
    if (not isinstance(record.get("asset_id"), str) or not isinstance(record.get("scene_id"), str)
            or not isinstance(grid, dict) or not {"crs", "transform", "width", "height"} <= set(grid)):
        raise ValueError("The example needs asset_id, scene_id and grid (crs, transform, width, height).")
    Grid(grid["crs"], tuple(grid["transform"]), grid["width"], grid["height"])
    keys = ("asset_id", "scene_id", "grid", "aoi", "scene", "label", "elapsed_seconds", "ee_runtime_seconds")
    return {key: record[key] for key in keys if record.get(key) is not None}


class Demo:
    model = "unet"
    asset_prefix = "segwater_demo"
    offline = False

    def __init__(self, args, ee):
        self.args, self.ee = args, ee
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.RLock()
        self.ee_lock = threading.RLock()
        self.stop_event = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ee-batch")
        self.bundle = None
        self.jobs = {}
        self.searches = {}
        self.map_cache = {}
        example = getattr(args, "example", None)
        self.example_record = load_example(example, self.model) if example else None
        self.args.state_dir.mkdir(parents=True, exist_ok=True)
        for path in self.args.state_dir.glob("*/job.json"):
            job = json.loads(path.read_text())
            # A live server never adopts an offline fixture job (it would submit it), and
            # an offline server never adopts a live one.
            if (job.get("project") == args.project and job.get("model", "unet") == self.model
                    and bool(job.get("offline")) == self.offline):
                self.jobs[job["job_id"]] = job

    def resume_jobs(self):
        """Resume only after this process successfully binds its UI port."""
        for job in sorted(self.jobs.values(), key=lambda j: j["created_unix"]):
            if job["state"] not in TERMINAL:
                self.executor.submit(self._worker, job["job_id"])

    def _path(self, job):
        return self.args.state_dir / job["job_id"] / "job.json"

    def _save(self, job, **fields):
        with self.lock:
            job.update(fields)
            atomic_write(self._path(job), job)

    def _example_title(self):
        return (self.example_record or {}).get("label", "Completed example")

    def config(self):
        with self.lock:
            jobs = sorted(self.jobs.values(), key=lambda j: j["created_unix"])
            active = next((j["job_id"] for j in jobs if j["state"] not in TERMINAL), None)
        # The session token is not here: the page reads it from the URL fragment the server
        # printed (page_url), which the browser never sends; any local process can GET this.
        return dict(project=self.args.project, max_tiles=self.args.max_tiles, model=self.model,
                    offline=self.offline, default_aoi=rectangle(120.902, 23.849, 120.918, 23.863),
                    default_dates=dict(start="2024-01-01", end="2024-01-07"),
                    example_available=self.example_record is not None, example_title=self._example_title(),
                    active_job_id=active, last_job_id=jobs[-1]["job_id"] if jobs else None)

    # Earth Engine metadata requests: each is one getInfo call, kept in its own method
    # so that offline mode can answer it locally (demos/browser/offline.py).
    def _projected_ring(self, aoi, crs):
        """The AOI's bounding ring in ``crs``, as Earth Engine projects it."""
        geometry = self.ee.Geometry(aoi, proj="EPSG:4326", geodesic=False)
        with self.ee_lock:
            return geometry.bounds(maxError=1, proj=crs).coordinates().getInfo()[0]

    def _geographic_ring(self, grid):
        """The grid's bounding ring in longitude/latitude."""
        with self.ee_lock:
            return grid.geometry().bounds(maxError=1, proj="EPSG:4326").coordinates().getInfo()[0]


    def _url(self, image, visualization):
        with self.ee_lock:
            return image.getMapId(visualization)["tile_fetcher"].url_format

    def _area(self, value):
        aoi = validate_aoi(value)
        crs = utm_crs(aoi)
        geometry = self.ee.Geometry(aoi, proj="EPSG:4326", geodesic=False)
        coordinates = self._projected_ring(aoi, crs)
        xs, ys = zip(*coordinates)
        grid = grid_from_projected_bounds((min(xs), min(ys), max(xs), max(ys)), crs,
                                          max_tiles=self.args.max_tiles)
        if max(xs)-min(xs) < 10 or max(ys)-min(ys) < 10:
            raise ValueError("Draw an area at least 10 metres wide and high.")
        return aoi, geometry, grid

    @staticmethod
    def _search_key(aoi):
        return json.dumps(aoi, sort_keys=True, separators=(",", ":"))

    def _scene_rows(self, geometry, grid, start, end):
        """Up to 51 VV/VH IW scenes intersecting the area, with their coverage of the grid."""
        ee = self.ee
        with self.ee_lock:
            collection = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(geometry)
                          .filterDate(start, end).filter(ee.Filter.eq("instrumentMode", "IW"))
                          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
                          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
                          .filter(ee.Filter.eq("resolution_meters", 10)).sort("system:time_start"))

            def feature(value):
                image = ee.Image(value)
                valid = (image.select(["VV", "VH"]).mask().reduce(ee.Reducer.min()).gt(0)
                         .unmask(0, sameFootprint=False).rename("valid"))
                fraction = valid.reduceRegion(ee.Reducer.mean(), geometry=geometry, crs=grid.crs,
                                              crsTransform=list(grid.transform),
                                              maxPixels=grid.width*grid.height, tileScale=2).get("valid")
                return ee.Feature(None, dict(id=ee.String("COPERNICUS/S1_GRD/").cat(image.get("system:index")),
                                             time=image.get("system:time_start"),
                                             orbit=image.get("orbitProperties_pass"),
                                             relative_orbit=image.get("relativeOrbitNumber_start"),
                                             coverage_fraction=fraction))
            return ee.FeatureCollection(collection.limit(51).toList(51).map(feature)).getInfo()["features"]

    def scenes(self, body):
        start, end = date_range(body.get("start"), body.get("end"))
        aoi, geometry, grid = self._area(body.get("aoi"))
        rows = self._scene_rows(geometry, grid, start, end)
        found = []
        for row in rows[:50]:
            item = row["properties"]
            item["date"] = datetime.fromtimestamp(item.pop("time")/1000, timezone.utc).isoformat().replace("+00:00", "Z")
            item["coverage_fraction"] = item.get("coverage_fraction") or 0
            found.append(item)
        with self.lock:
            self.searches[self._search_key(aoi)] = {scene["id"]: scene for scene in found}
        return dict(scenes=found, grid=grid_metadata(grid), truncated=len(rows) > 50)

    def _selected(self, body):
        aoi = validate_aoi(body.get("aoi"))
        scene_id = body.get("scene_id")
        with self.lock:
            scene = self.searches.get(self._search_key(aoi), {}).get(scene_id)
        if scene is None:
            raise ValueError("Search available scenes for this area, then select one from the list.")
        return aoi, scene

    def preview(self, body):
        aoi, scene = self._selected(body)
        geom = self.ee.Geometry(aoi, proj="EPSG:4326", geodesic=False)
        url = self._url(self.ee.Image(scene["id"]).select("VV").clip(geom), dict(min=-25, max=0))
        west, south, east, north = aoi_bounds(aoi)
        return dict(layers=dict(source=url), bounds=[[south, west], [north, east]])

    def run(self, body):
        aoi, scene = self._selected(body)
        if scene["coverage_fraction"] < .999:
            raise ValueError("Choose a scene that fully covers the drawn area, or draw a smaller area.")
        aoi, _, grid = self._area(aoi)
        with self.lock:
            if any(j["state"] not in TERMINAL for j in self.jobs.values()):
                raise ValueError("An inference job is already running. Wait for it to finish.")
            job_id = uuid.uuid4().hex
            job = dict(job_id=job_id, state="BUILDING", message="Preparing native Earth Engine inference…",
                       project=self.args.project, model=self.model, aoi=aoi, scene_id=scene["id"], scene=scene,
                       grid=grid_metadata(grid), created_unix=time.time(),
                       asset_id=f"projects/{self.args.project}/assets/{self.asset_prefix}_{job_id}")
            if self.offline:
                job["offline"] = True
            self.jobs[job_id] = job
            self._save(job)
            self.executor.submit(self._worker, job_id)
        return dict(job_id=job_id, state=job["state"])

    def _prepare(self, job):
        import torch
        from tensor_ee import MatchedWeightProvider
        from tensor_ee.segwater_inference import predict_segwater
        from tensor_ee.segwater_pytorch import load_segwater_checkpoint
        torch.set_num_threads(2)
        if self.bundle is None:
            self.bundle = load_segwater_checkpoint(self.args.checkpoint)
        data = job["grid"]
        grid = Grid(data["crs"], tuple(data["transform"]), data["width"], data["height"])
        ee = self.ee
        with self.ee_lock:
            provider = MatchedWeightProvider(self.bundle.compilable_model, self.args.manifest)
            prediction = predict_segwater(ee.Image(job["scene_id"]), grid, self.bundle.compilable_model, provider)
            image = prediction.image.clip(ee.Geometry(job["aoi"], proj="EPSG:4326", geodesic=False))
            image = image.set(dict(source_scene=job["scene_id"], checkpoint_sha256=self.bundle.metadata["source_sha256"],
                                   inference="native_ee_fixed224_float32", aoi=json.dumps(job["aoi"])))
            task = ee.batch.Export.image.toAsset(image, description=f"segwater_demo_{job['job_id']}",
                                                 assetId=job["asset_id"], region=grid.geometry(), crs=grid.crs,
                                                 crsTransform=list(grid.transform), maxPixels=grid.width*grid.height*5,
                                                 pyramidingPolicy={".default": "sample"})
            config = prepared_image_request(ee, task)
            operations = native_operation_names(config["expression"])
            request_bytes = len(json.dumps(config, separators=(",", ":"), allow_nan=False).encode())
            if request_bytes >= REQUEST_BUDGET_BYTES:
                raise ValueError("The inference request is too large. Draw a smaller area.")
            request_id = ee.data.newTaskId()[0]
        if provider.fallback_keys or len(provider.used_asset_keys) != 31:
            raise ValueError("Checkpoint and all deployed weight layers must match exactly.")
        path = self._path(job).with_name("request.json")
        atomic_write(path, config)
        self._save(job, checkpoint_sha256=self.bundle.metadata["source_sha256"],
                   manifest_sha256=digest(self.args.manifest), predictor=prediction.metadata(),
                   native_operations=operations, request_bytes=request_bytes,
                   export=dict(request_id=request_id, task_id=request_id, asset_id=job["asset_id"],
                               request_file=str(path.resolve()), request_sha256=digest(path),
                               prepared_unix=time.time(), state="PREPARED"))

    def _worker(self, job_id):
        job = self.jobs[job_id]
        try:
            if "export" not in job:
                self._prepare(job)
            if job["export"]["state"] in ("PREPARED", "SUBMITTING") and not Path(job["export"]["request_file"]).is_file():
                raise ValueError("The prepared request file is missing. Start a new run.")
            while job["export"]["state"] in ("PREPARED", "SUBMITTING"):
                if self.stop_event.is_set():
                    return
                try:
                    with self.ee_lock:
                        submit(self.ee, job, self._path(job))
                except ValueError:
                    # A changed request is never submitted.
                    raise
                except Exception:
                    # A lost response does not prove submission failed. Retry
                    # the persisted idempotency key, never create a new task.
                    self._save(job, state="RECONNECTING", message="Reconnecting to Earth Engine submission…")
                    if self.stop_event.wait(10):
                        return
            while not self.stop_event.is_set():
                try:
                    with self.ee_lock:
                        status, = self.ee.data.getTaskStatus(job["export"]["task_id"])
                except Exception:
                    self._save(job, state="RECONNECTING", message="Connection interrupted. Retrying Earth Engine status…")
                    if self.stop_event.wait(10):
                        return
                    continue
                state = status["state"]
                job["export"].update(state=state, task_status=status)
                message = {"READY": "Queued in Earth Engine…", "RUNNING": "Earth Engine is computing the water map…",
                           "COMPLETED": "Water map ready.",
                           "FAILED": status.get("error_message", "Earth Engine export failed."),
                           "CANCELLED": "The Earth Engine job was cancelled."}.get(state, state)
                if state == "UNKNOWN":
                    self._save(job, state="RECONNECTING", message="Waiting for Earth Engine to locate the saved task…")
                    if self.stop_event.wait(10):
                        return
                    continue
                self._save(job, state=state, message=message)
                if state in TERMINAL:
                    start = status.get("start_timestamp_ms")
                    end = status.get("update_timestamp_ms")
                    self._save(job, finished_unix=time.time(),
                               ee_runtime_seconds=(end-start)/1000 if start and end else None,
                               batch_eecu_seconds=status.get("batch_eecu_usage_seconds"))
                    return
                if self.stop_event.wait(5):
                    return
        except Exception as error:
            # The page sees the demo's own messages (ValueError); anything else is detail for
            # the console only, since Earth Engine and Python errors can name local paths.
            message = str(error) if isinstance(error, ValueError) else GENERIC_ERROR
            self._save(job, state="FAILED", message=message, finished_unix=time.time())
            print(f"Job {job_id}: {type(error).__name__}: {error}", flush=True)

    def _layers(self, job):
        key = job["job_id"]
        cached = self.map_cache.get(key)
        if cached and time.time()-cached[0] < 1800:
            return cached[1]
        ee = self.ee
        geom = ee.Geometry(job["aoi"], proj="EPSG:4326", geodesic=False)
        result = ee.Image(job["asset_id"]).clip(geom)
        layers = dict(probability=self._url(result.select("water_probability"),
                                            dict(min=0, max=1, palette=PROBABILITY_PALETTE)),
                      water=self._url(result.select("water_mask").selfMask(), dict(min=0, max=1, palette=["2171b5"])),
                      source=self._url(ee.Image(job["scene_id"]).select("VV").clip(geom), dict(min=-25, max=0)))
        self.map_cache[key] = (time.time(), layers)
        return layers

    def public_job(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                raise ValueError("That demo job was not found.")
            job = dict(self.jobs[job_id])
        west, south, east, north = aoi_bounds(job["aoi"])
        result = {k: job.get(k) for k in ["job_id", "state", "message", "asset_id", "scene_id", "scene", "grid", "aoi",
                                          "ee_runtime_seconds", "batch_eecu_seconds"]}
        result.update(elapsed_seconds=(job.get("finished_unix") or time.time())-job["created_unix"],
                      bounds=[[south, west], [north, east]])
        if job["state"] == "COMPLETED":
            try:
                result["layers"] = self._layers(job)
            except Exception as error:
                print(f"Job {job['job_id']}: map layers failed: {type(error).__name__}: {error}", flush=True)
                result["layer_error"] = "Map display failed; see the server console."
                result["message"] = "Inference completed, but map display failed. Refresh to retry."
        return result

    def example(self):
        record = self.example_record
        if record is None:
            raise ValueError("No completed example is configured. Start the demo with --example.")
        config = record["grid"]
        grid = Grid(config["crs"], tuple(config["transform"]), config["width"], config["height"])
        ring = self._geographic_ring(grid)
        xs, ys = zip(*ring)
        aoi = record.get("aoi") or rectangle(min(xs), min(ys), max(xs), max(ys))
        # This is a clearly labeled previous result, never a substitute for a run.
        job = dict(job_id="example", asset_id=record["asset_id"], aoi=aoi, scene_id=record["scene_id"])
        return dict(**job, state="COMPLETED", message=EXAMPLE_MESSAGE, example_title=self._example_title(),
                    scene=record.get("scene") or dict(id=record["scene_id"]),
                    grid=grid_metadata(grid), bounds=[[min(ys), min(xs)], [max(ys), max(xs)]],
                    elapsed_seconds=record.get("elapsed_seconds"),
                    ee_runtime_seconds=record.get("ee_runtime_seconds"), layers=self._layers(job))


def handler_for(app, port):
    class Handler(BaseHTTPRequestHandler):
        server_version = "SegWaterDemo/1"          # no Python version in the Server header
        sys_version = ""

        def version_string(self):
            return self.server_version

        def _send(self, status, value, kind="application/json"):
            data = json.dumps(value, allow_nan=False).encode() if kind == "application/json" else value
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Content-Security-Policy", "frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(data)

        def _failed(self, error):
            """400 with the demo's own message (ValueError), or a generic one; details to the console."""
            if isinstance(error, ValueError):
                self._send(400, dict(error=str(error)))
            else:
                print(f"{getattr(self, 'command', '?')} {self.path}: {type(error).__name__}: {error}", flush=True)
                self._send(400, dict(error=GENERIC_ERROR))

        def _local(self):
            if self.headers.get("Host") not in (f"127.0.0.1:{port}", f"localhost:{port}"):
                raise PermissionError("This demo accepts loopback requests only.")

        def do_GET(self):
            try:
                self._local()
                path = urlsplit(self.path).path
                files = {"/": ("index.html", "text/html; charset=utf-8"),
                         "/app.js": ("app.js", "application/javascript"), "/style.css": ("style.css", "text/css")}
                if path in files:
                    name, kind = files[path]
                    self._send(200, (STATIC/name).read_bytes(), kind)
                elif path == "/api/config":
                    self._send(200, app.config())
                elif path == "/api/example":
                    self._send(200, app.example())
                elif path.startswith("/api/jobs/"):
                    self._send(200, app.public_job(path.rsplit("/", 1)[1]))
                else:
                    self._send(404, dict(error="Not found"))
            except PermissionError as error:
                self._send(403, dict(error=str(error)))
            except Exception as error:
                self._failed(error)

        def do_POST(self):
            try:
                self._local()
                origin = self.headers.get("Origin")
                if origin and origin not in (f"http://127.0.0.1:{port}", f"http://localhost:{port}"):
                    raise PermissionError("Request origin is not this local demo.")
                if not secrets.compare_digest(self.headers.get("X-Demo-Token", ""), app.token):
                    raise PermissionError("Reload the demo page before submitting a request.")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    raise ValueError("Expected a small JSON request.")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("Expected a JSON object.")
                routes = {"/api/scenes": app.scenes, "/api/preview": app.preview, "/api/run": app.run}
                route = routes.get(urlsplit(self.path).path)
                if route is None:
                    self._send(404, dict(error="Not found"))
                    return
                self._send(200, route(body))
            except PermissionError as error:
                self._send(403, dict(error=str(error)))
            except Exception as error:
                self._failed(error)

    return Handler


def parse_args(argv=None):
    """Parse and complete the command line; the defaults are documented in DEMO.md."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", required=True,
                        help="Earth Engine project that runs the exports and owns the result assets.")
    parser.add_argument("--model", choices=["unet", "swin-t"], default="unet")
    parser.add_argument("--port", type=int, help="Loopback port (default 8765; 8766 for swin-t).")
    parser.add_argument("--max-tiles", type=int, choices=[1, 2, 3, 4],
                        help="Chip cap for the snapped area (default 4; swin-t supports exactly 1).")
    parser.add_argument("--checkpoint", type=Path, help="Local checkpoint (default: see DEMO.md).")
    parser.add_argument("--manifest", type=Path, help="Weight-asset manifest (default: data/manifests/).")
    parser.add_argument("--state-dir", type=Path, help="Job records and prepared requests "
                                                       "(default demos/browser/state, or its swin-t/ for swin-t).")
    parser.add_argument("--example", type=Path, help="Completed result to offer as a labelled example "
                                                     "(a job.json of this demo or a verified history record).")
    parser.add_argument("--offline", action="store_true",
                        help="Serve the UI against a local fixture: no Earth Engine requests (see DEMO.md).")
    args = parser.parse_args(argv)
    defaults = DEFAULTS[args.model]
    if args.model == "swin-t" and args.max_tiles not in (None, 1):
        parser.error("The Swin-T/UPerNet demo supports exactly one 224×224 chip.")
    args.port = defaults["port"] if args.port is None else args.port
    args.max_tiles = args.max_tiles or defaults["max_tiles"]
    # Offline mode keeps its state apart: a new temporary directory unless --state-dir is given.
    for name in ("checkpoint", "manifest") if args.offline else ("checkpoint", "manifest", "state_dir"):
        if getattr(args, name) is None:
            setattr(args, name, ROOT / defaults[name])
    if not 1024 <= args.port <= 65535:
        parser.error("Use a port between 1024 and 65535.")
    return args


def controller_class(model):
    if model == "swin-t":
        from demos.browser.swin import SwinDemo
        return SwinDemo
    return Demo


def serve(app, port):
    """Bind loopback, resume unfinished jobs, and serve until interrupted."""
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), handler_for(app, port))
    except Exception:
        app.executor.shutdown(wait=False, cancel_futures=True)
        raise
    app.resume_jobs()
    mode = " · OFFLINE FIXTURE, no Earth Engine requests" if app.offline else ""
    print(f"SegWater {app.model} demo (project {app.args.project}){mode}", flush=True)
    print(f"Open this URL (its #token part stays in the browser and authorizes runs):\n  {page_url(port, app.token)}",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping UI; submitted Earth Engine exports continue. Restart to resume status.", flush=True)
    finally:
        app.stop_event.set()
        app.executor.shutdown(wait=False, cancel_futures=True)
        server.server_close()


def main(argv=None):
    args = parse_args(argv)
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))
    if args.offline:
        from demos.browser.offline import main as offline_main
        return offline_main(args)
    ee = initialize(args.project)
    serve(controller_class(args.model)(args, ee), args.port)


if __name__ == "__main__":
    main()
