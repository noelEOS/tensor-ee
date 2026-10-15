"""One-chip Swin-T/UPerNet controller on the library's three-stage Swin workflow.

The three exports have independent persisted request IDs. Local code loads and
checks checkpoint coefficients, constructs graphs, and polls EE; it never runs
the model locally. The sampling lattice is anchored to the verified catalogue
chip of the Sun Moon Lake scene (Milestone 12, ``docs/history/upernet/``).

Each stage graph comes from ``tensor_ee.swin_workflow.build_stage``, given a
per-job resource declaration (``resources.json``, validated by
``read_resources``): the named ``segwater_v3_db`` input recipe on the selected
scene and the ``water`` output recipe. The weights are the trained assets listed
in the manifest, loaded through ``tensor_ee.swin``. Ported from the archived demo
(``demo/swin.py`` at the tag ``research-archive-2026-10-12``); its export
requests are byte-identical (docs/evidence/l06c-demo-graph-identity.json).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

from tensor_ee import Grid
from tensor_ee.core.grid import to_feature_grid
from tensor_ee.swin import load_swin_t224_parameters
from tensor_ee.swin_workflow import build_stage, read_resources

from demos.browser.exports import atomic_write, digest, native_operation_names, prepared_image_request, submit
from demos.browser.geometry import grid_from_projected_bounds, utm_crs, validate_aoi
from demos.browser.server import REQUEST_BUDGET_BYTES, Demo, EXAMPLE_MESSAGE, PROBABILITY_PALETTE, grid_metadata, rectangle

# The verified Sentinel-1 scene's native grid (EE band metadata of VV and VH) and the
# Milestone 12 catalogue chip on it (archived swin_image_encoder/scene.py).
SCENE_ID = ("COPERNICUS/S1_GRD/"
            "S1A_IW_GRDH_1SDV_20240103T100134_20240103T100159_051941_06469E_B0BD")
CRS = "EPSG:32651"
BANDS = ("VV", "VH")
PIXEL = 10.0
NATIVE_ORIGIN = (169540.95335071225, 2780488.8405420803)
CATALOGUE_COLUMN, CATALOGUE_ROW = 11814, 13965      # native pixel of Sun Moon Lake (120.915 E, 23.865 N)
CATALOGUE_ORIGIN = (NATIVE_ORIGIN[0] + PIXEL * CATALOGUE_COLUMN, NATIVE_ORIGIN[1] - PIXEL * CATALOGUE_ROW)
CATALOGUE_TRANSFORM = (PIXEL, 0.0, CATALOGUE_ORIGIN[0], 0.0, -PIXEL, CATALOGUE_ORIGIN[1])
CHECKPOINT_SHA256 = "5c7a7c44d29b3c75bd69f59809fb1b63a039313bec021d897ff3876d2cea42a5"
PRODUCTION = "grouped_double"

STAGES = (("A", "Encoder stages 1–3"), ("B", "Encoder stage 4"), ("C", "UPerNet and water map"))
OUTPUT_BANDS = ["logit_0", "logit_1", "water_probability", "water_mask", "coverage"]


def single_chip_grid(bounds):
    """Snap projected bounds to the tested scene's 224-pixel lattice, at most one chip."""
    x0, y0 = CATALOGUE_ORIGIN
    xmin, ymin, xmax, ymax = bounds
    relative = grid_from_projected_bounds((xmin-x0, ymin-y0, xmax-x0, ymax-y0), CRS, 1)
    a, b, x, d, e, y = relative.transform
    return Grid(CRS, (a, b, x+x0, d, e, y+y0), 224, 224)


def native_offset(grid):
    """(column, row) of a chip on the verified scene's native pixel grid; ValueError if unaligned."""
    if grid.crs != CRS or (grid.width, grid.height) != (224, 224):
        raise ValueError("Swin-T requires one 224×224 chip on EPSG:32651.")
    a, b, x0, d, e, y0 = grid.transform
    if (a, b, d, e) != (10, 0, 0, -10):
        raise ValueError("Swin-T requires the tested north-up 10 m grid.")
    col = (x0-NATIVE_ORIGIN[0])/10
    row = (NATIVE_ORIGIN[1]-y0)/10
    if not math.isclose(col, round(col), abs_tol=1e-7, rel_tol=0) or not math.isclose(row, round(row), abs_tol=1e-7, rel_tol=0):
        raise ValueError("The chip must align to the tested Sentinel-1 sampling grid.")
    return round(col), round(row)


def grid_for_job(job):
    data = job["grid"]
    return Grid(data["crs"], tuple(data["transform"]), data["width"], data["height"])


def water_layers(ee, logits):
    """Display only: stable two-class softmax, water is class 1; ties follow softmax ≥ 0.5.

    The export graph's water recipe is ``tensor_ee.swin_workflow.build_stage``'s
    (output mode ``water``); this recomputes the same bands from saved logits, so
    that a logits-only example asset can be displayed too.
    """
    logits = logits.select(["logit_0", "logit_1"]).toFloat()
    maximum = logits.reduce(ee.Reducer.max())
    exp = logits.subtract(maximum).exp()
    probability = exp.select("logit_1").divide(exp.reduce(ee.Reducer.sum())).toFloat().rename("water_probability")
    return logits.addBands(probability).addBands(probability.gte(.5).toFloat().rename("water_mask"))


class SwinDemo(Demo):
    model = "swin-t"
    asset_prefix = "swint_upernet_demo"

    def _geographic_bounds(self, grid):
        xs, ys = zip(*self._geographic_ring(grid))
        return rectangle(min(xs), min(ys), max(xs), max(ys))

    def _chip_geometry(self, grid):
        """The chip outline in longitude/latitude (one getInfo call)."""
        with self.ee_lock:
            return grid.geometry().transform("EPSG:4326", maxError=1).getInfo()

    def config(self):
        config = super().config()
        x0, y0 = CATALOGUE_ORIGIN
        # Inset the geographic drawing: the WGS84 bounding rectangle of a UTM
        # chip protrudes past its edges and would otherwise snap to four chips.
        inset = Grid(CRS, (10, 0, x0+100, 0, -10, y0-100), 204, 204)
        config.update(max_tiles=1, default_aoi=self._geographic_bounds(inset),
                      grid_note="One 224×224 chip · 10 m · Sun Moon Lake sampling grid",
                      sampling_crs=CRS)
        return config

    def _area(self, value):
        aoi = validate_aoi(value)
        if utm_crs(aoi) != CRS:
            raise ValueError("This demo uses Taiwan’s UTM zone 51N. Use the lake preset or draw within 120°–126°E north of the equator.")
        coordinates = self._projected_ring(aoi, CRS)
        xs, ys = zip(*coordinates)
        if max(xs)-min(xs) < 10 or max(ys)-min(ys) < 10:
            raise ValueError("Draw an area at least 10 metres wide and high.")
        grid = single_chip_grid((min(xs), min(ys), max(xs), max(ys)))
        # Search/coverage checks cover the entire model input, not just the drawing.
        return aoi, grid.geometry(), grid

    def scenes(self, body):
        result = super().scenes(body)
        data = result["grid"]
        grid = Grid(data["crs"], tuple(data["transform"]), 224, 224)
        result["grid"]["chip_aoi"] = self._chip_geometry(grid)
        return result

    def preview(self, body):
        aoi, selected = self._selected(body)
        _, geometry, grid = self._area(aoi)
        source = self.ee.Image(selected["id"]).select("VV").reproject(grid.projection()).clip(geometry)
        bounds = self._geographic_bounds(grid)["coordinates"][0]
        return dict(layers=dict(source=self._url(source, dict(min=-25, max=0))),
                    bounds=[[bounds[0][1], bounds[0][0]], [bounds[2][1], bounds[2][0]]])

    def _parameters(self, job):
        manifest_sha = digest(self.args.manifest)
        if job.get("manifest_sha256", manifest_sha) != manifest_sha:
            raise ValueError("The weight manifest changed during this run. Restore the original manifest and start a new run.")
        if self.bundle is None:
            manifest = json.loads(Path(self.args.manifest).read_text())
            parameters = load_swin_t224_parameters(self.args.checkpoint, expected_checkpoint_sha256=CHECKPOINT_SHA256,
                                                   manifest=manifest)
            self.bundle = (parameters, manifest_sha, CHECKPOINT_SHA256)
        if self.bundle[1] != manifest_sha:
            raise ValueError("The weight manifest changed. Restart the demo after restoring it.")
        self._save(job, manifest_sha256=manifest_sha, checkpoint_sha256=self.bundle[2],
                   formulation=PRODUCTION, inference="native_ee_swint_upernet_independent224")
        return self.bundle[0]

    def _resources(self, job):
        """Write and validate this job's resource declaration for ``swin_workflow``.

        Input: the selected scene through the named ``segwater_v3_db`` recipe.
        Output: the ``water`` recipe. Destinations: ``<asset>_A``, ``<asset>_B``
        and the final ``<asset>``, all in the demo's project.
        """
        grid = job["grid"]
        declaration = dict(
            schema_version=1, project=self.args.project,
            checkpoint=str(Path(self.args.checkpoint).resolve()), checkpoint_sha256=job["checkpoint_sha256"],
            manifest=str(Path(self.args.manifest).resolve()), manifest_file_sha256=job["manifest_sha256"],
            input_bands=list(BANDS), output_bands=OUTPUT_BANDS[:2],
            grid={key: grid[key] for key in ("crs", "transform", "width", "height")},
            input=dict(mode="segwater_v3_db", asset_id=job["scene_id"]), output=dict(mode="water"),
            destinations={name: job["asset_id"] if name == "C" else f"{job['asset_id']}_{name}" for name, _ in STAGES})
        path = self._path(job).with_name("resources.json")
        atomic_write(path, declaration)
        return read_resources(path)

    def _stage_graph(self, job, index, params):
        raster = build_stage(self._resources(job), STAGES[index][0], parameters=params)
        return raster.image, to_feature_grid(raster.grid)

    def _prepare_stage(self, job, index):
        ee = self.ee
        with self.ee_lock:
            image, grid = self._stage_graph(job, index, self._parameters(job))
            name, label = STAGES[index]
            asset = job["asset_id"] if index == 2 else f"{job['asset_id']}_{name}"
            image = image.set(dict(source_scene=job["scene_id"], checkpoint_sha256=job["checkpoint_sha256"],
                                   inference=job["inference"], model=self.model, stage=name,
                                   formulation=PRODUCTION, manifest_sha256=job["manifest_sha256"]))
            task = ee.batch.Export.image.toAsset(image, description=f"{self.asset_prefix}_{job['job_id']}_{name}",
                                                 assetId=asset, region=grid.region(ee), crs=grid.crs,
                                                 crsTransform=grid.transform, maxPixels=grid.width*grid.height,
                                                 pyramidingPolicy={".default": "sample"})
            config = prepared_image_request(ee, task)
            operations = native_operation_names(config["expression"])
            size = len(json.dumps(config, separators=(",", ":"), allow_nan=False).encode())
            if size >= REQUEST_BUDGET_BYTES:
                raise ValueError(f"Stage {name} exceeds the EE request budget.")
            request_id = ee.data.newTaskId()[0]
        path = self._path(job).with_name(f"request_{name}.json")
        atomic_write(path, config)
        stage = dict(name=name, label=label, native_operations=operations, request_bytes=size,
                     export=dict(request_id=request_id, task_id=request_id, asset_id=asset,
                                 request_file=str(path.resolve()), request_sha256=digest(path),
                                 prepared_unix=time.time(), state="PREPARED"))
        with self.lock:
            job["stages"].append(stage)
            self._save(job, export=stage["export"])

    def _worker(self, job_id):
        job = self.jobs[job_id]
        try:
            job.setdefault("stages", [])
            for index, (name, label) in enumerate(STAGES):
                if self.stop_event.is_set():
                    return
                if len(job["stages"]) > index and job["stages"][index]["export"]["state"] == "COMPLETED":
                    continue
                self._save(job, stage_index=index, stage_name=name, stage_label=label,
                           state="BUILDING", message=f"Stage {index+1}/3 · Preparing {label.lower()}…")
                if len(job["stages"]) <= index:
                    self._prepare_stage(job, index)
                stage = job["stages"][index]
                self._save(job, export=stage["export"])
                if stage["export"]["state"] in ("PREPARED", "SUBMITTING") and not Path(stage["export"]["request_file"]).is_file():
                    raise ValueError(f"Stage {name}: the prepared request file is missing. Start a new run.")
                while stage["export"]["state"] in ("PREPARED", "SUBMITTING"):
                    if self.stop_event.is_set():
                        return
                    try:
                        with self.ee_lock:
                            submit(self.ee, job, self._path(job))
                    except ValueError:
                        raise
                    except Exception:
                        self._save(job, state="RECONNECTING", message=f"Stage {index+1}/3 · Reconnecting to EE submission…")
                        if self.stop_event.wait(10):
                            return
                while not self.stop_event.is_set():
                    try:
                        with self.ee_lock:
                            status, = self.ee.data.getTaskStatus(stage["export"]["task_id"])
                    except Exception:
                        self._save(job, state="RECONNECTING", message=f"Stage {index+1}/3 · Reconnecting to EE status…")
                        if self.stop_event.wait(10):
                            return
                        continue
                    state = status["state"]
                    with self.lock:
                        stage["export"].update(state=state, task_status=status)
                    if state == "COMPLETED":
                        start, end = status.get("start_timestamp_ms"), status.get("update_timestamp_ms")
                        stage.update(ee_runtime_seconds=(end-start)/1000 if start and end else None,
                                     batch_eecu_seconds=status.get("batch_eecu_usage_seconds"))
                        self._save(job)
                        break
                    if state in ("FAILED", "CANCELLED"):
                        self._save(job, state=state, finished_unix=time.time(),
                                   message=f"Stage {index+1}/3 · {label}: {status.get('error_message', state.lower())}")
                        return
                    if state not in ("READY", "RUNNING"):
                        raise ValueError(f"Stage {name}: unexpected EE task state {state}.")
                    verb = "Queued for" if state == "READY" else "Computing"
                    self._save(job, state=state, message=f"Stage {index+1}/3 · {verb} {label.lower()}…")
                    if self.stop_event.wait(5):
                        return
                if self.stop_event.is_set():
                    return
            self._save(job, state="COMPLETED", finished_unix=time.time(),
                       message="All 3 EE stages completed. One 224×224 water map is ready.",
                       ee_runtime_seconds=sum(s.get("ee_runtime_seconds") or 0 for s in job["stages"]),
                       batch_eecu_seconds=sum(s.get("batch_eecu_seconds") or 0 for s in job["stages"]))
        except Exception as error:
            self._save(job, state="FAILED", message=str(error), finished_unix=time.time())
            print(f"Swin-T job {job_id}: {type(error).__name__}: {error}", flush=True)

    def _layers(self, job):
        cached = self.map_cache.get(job["job_id"])
        if cached and time.time()-cached[0] < 1800:
            return cached[1]
        ee = self.ee
        grid = grid_for_job(job)
        with self.ee_lock:
            geometry = grid.geometry()
            result = water_layers(ee, ee.Image(job["asset_id"])).clip(geometry)
            source = ee.Image(job["scene_id"]).select("VV").reproject(grid.projection()).clip(geometry)
            layers = dict(probability=self._url(result.select("water_probability"),
                                                dict(min=0, max=1, palette=PROBABILITY_PALETTE)),
                          water=self._url(result.select("water_mask").selfMask(), dict(min=0, max=1, palette=["2171b5"])),
                          source=self._url(source, dict(min=-25, max=0)))
        self.map_cache[job["job_id"]] = (time.time(), layers)
        return layers

    def public_job(self, job_id):
        result = super().public_job(job_id)
        job = self.jobs[job_id]
        aoi = self._geographic_bounds(grid_for_job(job))["coordinates"][0]
        result.update(model=self.model, stage_index=job.get("stage_index"), stage_label=job.get("stage_label"),
                      bounds=[[aoi[0][1], aoi[0][0]], [aoi[2][1], aoi[2][0]]],
                      stages=[dict(name=s["name"], label=s["label"], state=s["export"]["state"],
                                   asset_id=s["export"]["asset_id"]) for s in job.get("stages", [])])
        return result

    def example(self):
        record = self.example_record
        if record is None:
            raise ValueError("No completed example is configured. Start the demo with --example.")
        config = record["grid"]
        grid = Grid(config["crs"], tuple(config["transform"]), config["width"], config["height"])
        ring = self._geographic_bounds(grid)["coordinates"][0]
        aoi = record.get("aoi") or self.config()["default_aoi"]
        job = dict(job_id="swin-example", model=self.model, asset_id=record["asset_id"], aoi=aoi,
                   scene_id=record["scene_id"], grid=grid_metadata(grid))
        # This is a clearly labeled previous result, never a substitute for a run.
        return dict(**job, state="COMPLETED", message=EXAMPLE_MESSAGE, example_title=self._example_title(),
                    scene=record.get("scene") or dict(id=record["scene_id"]),
                    elapsed_seconds=record.get("elapsed_seconds"), ee_runtime_seconds=record.get("ee_runtime_seconds"),
                    bounds=[[ring[0][1], ring[0][0]], [ring[2][1], ring[2][0]]], layers=self._layers(job))
