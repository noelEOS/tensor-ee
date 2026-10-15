"""``--offline``: the browser demo against a local fixture, with no Earth Engine requests.

Use it to drive the UI end to end without a cloud project or network access to
Earth Engine::

    python -m demos.browser.server --offline --project offline-check \\
        --checkpoint /path/to/checkpoints/unet_resnet18_s42_step6000_last.pth

The controllers, the HTTP handler, the job state machine and the browser code are
the real ones. Graph construction is real too: the export request of every job is
built with the library against the offline signatures of
``ee.apitestcase.ApiTestCase`` and saved next to the job, exactly as a live run
prepares it (this is why the checkpoint is needed). Everything that would reach
Earth Engine is answered locally instead:

- AOI projection and grid outlines use a local transverse Mercator (UTM)
  approximation, not Earth Engine's projection;
- the scene search returns fixture scenes dated 2024-01-03, 2024-01-05 (both full
  coverage) and 2024-01-06 (partial coverage), when the date range includes them;
- map layers are flat, labelled fixture tiles, not model output;
- submission and status come from a simulated task queue (READY, then RUNNING,
  then COMPLETED after about ten seconds), persisted in the state directory so that
  a restart resumes it.

Every other Earth Engine entry point raises. Offline jobs are marked
``"offline": true``; a live server never adopts them, and an offline server never
adopts live jobs. Without ``--state-dir`` the state goes to a new temporary
directory. Without ``--example`` a fixture example (clearly labelled) is offered.
The page still loads Leaflet and the OpenStreetMap basemap from the internet.
"""
from __future__ import annotations

import contextlib
from datetime import datetime
import json
import math
from pathlib import Path
import tempfile
import threading
import time
from unittest import mock
from urllib.parse import quote
import uuid

from demos.browser.exports import atomic_write
from demos.browser.server import Demo, controller_class, serve
from demos.browser.swin import CATALOGUE_TRANSFORM, SCENE_ID, SwinDemo

# ---------------------------------------------------------------------------
# Local UTM (WGS84 transverse Mercator, Krüger series to third order; about a
# millimetre within a zone). Offline approximation only: live runs use EE.
# ---------------------------------------------------------------------------
_A, _F = 6378137.0, 1 / 298.257223563
_N = _F / (2 - _F)
_RECTIFYING = _A / (1 + _N) * (1 + _N**2 / 4 + _N**4 / 64)
_ALPHA = (_N/2 - 2*_N**2/3 + 5*_N**3/16, 13*_N**2/48 - 3*_N**3/5, 61*_N**3/240)
_BETA = (_N/2 - 2*_N**2/3 + 37*_N**3/96, _N**2/48 + _N**3/15, 17*_N**3/480)
_DELTA = (2*_N - 2*_N**2/3 - 2*_N**3, 7*_N**2/3 - 8*_N**3/5, 56*_N**3/15)
_K0, _E0 = 0.9996, 500000.0


def _zone(crs):
    code = int(crs.split(":")[1])
    return code % 100, code // 100 == 327


def utm_forward(longitude, latitude, crs):
    """(x, y) metres in the UTM ``crs`` (EPSG:326zz or 327zz)."""
    zone, south = _zone(crs)
    phi, lam = math.radians(latitude), math.radians(longitude - (zone * 6 - 183))
    c = 2 * math.sqrt(_N) / (1 + _N)
    t = math.sinh(math.atanh(math.sin(phi)) - c * math.atanh(c * math.sin(phi)))
    xi, eta = math.atan2(t, math.cos(lam)), math.atanh(math.sin(lam) / math.sqrt(1 + t * t))
    x = eta + sum(a * math.cos(2*j*xi) * math.sinh(2*j*eta) for j, a in enumerate(_ALPHA, 1))
    y = xi + sum(a * math.sin(2*j*xi) * math.cosh(2*j*eta) for j, a in enumerate(_ALPHA, 1))
    return _E0 + _K0 * _RECTIFYING * x, (10_000_000.0 if south else 0.0) + _K0 * _RECTIFYING * y


def utm_inverse(x, y, crs):
    """(longitude, latitude) of UTM metres in ``crs``."""
    zone, south = _zone(crs)
    xi = (y - (10_000_000.0 if south else 0.0)) / (_K0 * _RECTIFYING)
    eta = (x - _E0) / (_K0 * _RECTIFYING)
    xi1 = xi - sum(b * math.sin(2*j*xi) * math.cosh(2*j*eta) for j, b in enumerate(_BETA, 1))
    eta1 = eta - sum(b * math.cos(2*j*xi) * math.sinh(2*j*eta) for j, b in enumerate(_BETA, 1))
    chi = math.asin(math.sin(xi1) / math.cosh(eta1))
    phi = chi + sum(d * math.sin(2*j*chi) for j, d in enumerate(_DELTA, 1))
    lam = math.atan2(math.sinh(eta1), math.cos(xi1))
    return math.degrees(lam) + zone * 6 - 183, math.degrees(phi)


def _edges(west, south, east, north, steps=8):
    """Points along a rectangle's four edges (its corners included)."""
    for k in range(steps + 1):
        u = k / steps
        yield from ((west + (east - west) * u, south), (west + (east - west) * u, north),
                    (west, south + (north - south) * u), (east, south + (north - south) * u))


def _ring(points):
    xs, ys = zip(*points)
    return [[min(xs), min(ys)], [max(xs), min(ys)], [max(xs), max(ys)], [min(xs), max(ys)], [min(xs), min(ys)]]


def _grid_extent(grid):
    a, _, c, _, e, f = grid.transform
    return c, f + e * grid.height, c + a * grid.width, f


# ---------------------------------------------------------------------------
# Fixture scenes and tiles
# ---------------------------------------------------------------------------
def _ms(text):
    return int(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp() * 1000)


# The two acquisitions of the verified runs (docs/history/cnn/results/demo_live_results.json),
# and one partial-coverage fixture to show how the UI refuses it.
FIXTURE_SCENES = (
    dict(id=SCENE_ID, time=_ms("2024-01-03T10:01:34Z"), orbit="ASCENDING", relative_orbit=69, coverage_fraction=1),
    dict(id="COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20240105T215301_20240105T215326_051977_0647E8_D7EE",
         time=_ms("2024-01-05T21:53:01Z"), orbit="DESCENDING", relative_orbit=105, coverage_fraction=1),
    dict(id="COPERNICUS/S1_GRD/OFFLINE_FIXTURE_PARTIAL_COVERAGE_20240106", time=_ms("2024-01-06T10:01:34Z"),
         orbit="ASCENDING", relative_orbit=69, coverage_fraction=0.62),
)


def fixture_tile(visualization):
    """A flat, labelled 256-pixel tile as a data URI (no {x}/{y}/{z}: every tile is the same)."""
    palette = visualization.get("palette")
    if visualization.get("min") == -25:
        fill, label = "#8a8a8a", "radar VV"
    elif palette and len(palette) == 1:
        fill, label = f"#{palette[0]}", "water mask"
    else:
        fill, label = "#6baed6", "probability"
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256">'
           f'<rect width="256" height="256" fill="{fill}"/>'
           f'<path d="M0 256L256 0" stroke="#fff" stroke-opacity="0.35" stroke-width="2"/>'
           f'<text x="10" y="24" font-family="sans-serif" font-size="14" fill="#fff">offline fixture · {label}</text>'
           f'</svg>')
    return "data:image/svg+xml," + quote(svg, safe="")


# ---------------------------------------------------------------------------
# Simulated Earth Engine task queue
# ---------------------------------------------------------------------------
class SimulatedTasks:
    """``newTaskId``/``exportImage``/``getTaskStatus`` with a READY → RUNNING → COMPLETED timeline.

    Submission is idempotent on the request ID, as in Earth Engine. The queue is saved
    in ``<state-dir>/offline_tasks.json`` so that a restarted fixture resumes it.
    """

    def __init__(self, project, path, ready_seconds=3.0, running_seconds=7.0):
        self.project, self.path = project, Path(path)
        self.ready_seconds, self.running_seconds = ready_seconds, running_seconds
        self.lock = threading.Lock()
        self.tasks = {}
        if self.path.is_file():
            self.tasks = json.loads(self.path.read_text())

    def new_task_id(self, count=1):
        return [uuid.uuid4().hex.upper()[:24] for _ in range(count)]

    def export_image(self, request_id, params):
        with self.lock:
            task = self.tasks.get(request_id)
            if task is None:
                task = dict(id=uuid.uuid4().hex.upper()[:24], description=params.get("description"),
                            created_ms=int(time.time() * 1000))
                self.tasks[request_id] = task
                atomic_write(self.path, self.tasks)
        return dict(name=f"projects/{self.project}/operations/{task['id']}")

    def task_status(self, task_ids):
        ids = [task_ids] if isinstance(task_ids, str) else list(task_ids)
        now = int(time.time() * 1000)
        statuses = []
        with self.lock:
            for task_id in ids:
                task = next((t for t in self.tasks.values() if t["id"] == task_id), None)
                if task is None:
                    statuses.append(dict(id=task_id, state="UNKNOWN"))
                    continue
                age = (now - task["created_ms"]) / 1000
                created, started = task["created_ms"], task["created_ms"] + int(self.ready_seconds * 1000)
                status = dict(id=task_id, description=task["description"], task_type="EXPORT_IMAGE",
                              creation_timestamp_ms=created, update_timestamp_ms=now,
                              name=f"projects/{self.project}/operations/{task_id}")
                if age < self.ready_seconds:
                    status.update(state="READY")
                elif age < self.ready_seconds + self.running_seconds:
                    status.update(state="RUNNING", start_timestamp_ms=started)
                else:
                    finished = started + int(self.running_seconds * 1000)
                    status.update(state="COMPLETED", start_timestamp_ms=started, update_timestamp_ms=finished,
                                  batch_eecu_usage_seconds=round(self.running_seconds * 1.5, 3))
                statuses.append(status)
        return statuses


# ---------------------------------------------------------------------------
# Offline controllers: only the Earth Engine metadata requests differ
# ---------------------------------------------------------------------------
class OfflineMixin:
    offline = True

    def _projected_ring(self, aoi, crs):
        west, south = aoi["coordinates"][0][0]
        east, north = aoi["coordinates"][0][2]
        return _ring(utm_forward(lon, lat, crs) for lon, lat in _edges(west, south, east, north))

    def _geographic_ring(self, grid):
        return _ring(utm_inverse(x, y, grid.crs) for x, y in _edges(*_grid_extent(grid)))

    def _scene_rows(self, geometry, grid, start, end):
        first, last = _ms(f"{start}T00:00:00Z"), _ms(f"{end}T00:00:00Z")
        return [dict(type="Feature", geometry=None, id=str(k), properties=dict(scene))
                for k, scene in enumerate(s for s in FIXTURE_SCENES if first <= s["time"] < last)]

    def _url(self, image, visualization):
        return fixture_tile(visualization)


class OfflineDemo(OfflineMixin, Demo):
    pass


class OfflineSwinDemo(OfflineMixin, SwinDemo):
    def _chip_geometry(self, grid):
        west, south, east, north = _grid_extent(grid)
        corners = [(west, south), (east, south), (east, north), (west, north), (west, south)]
        return dict(type="Polygon", coordinates=[[list(utm_inverse(x, y, grid.crs)) for x, y in corners]])


def fixture_example(model, project):
    """A labelled fixture result on the preset area (its layers are fixture tiles)."""
    if model == "swin-t":
        grid = dict(crs="EPSG:32651", transform=list(CATALOGUE_TRANSFORM), width=224, height=224)
    else:
        grid = dict(crs="EPSG:32651", transform=[10, 0, 284480, 0, -10, 2640960], width=448, height=224)
    return dict(asset_id=f"projects/{project}/assets/offline_fixture_example", scene_id=SCENE_ID, grid=grid,
                label="Offline fixture example", elapsed_seconds=75.0, ee_runtime_seconds=58.0)


_REFUSED = ("_execute_cloud_call", "getAsset", "getInfo", "listAssets", "computePixels", "computeImages",
            "computeFeatures", "computeValue", "getMapId", "getDownloadId", "getThumbId", "getTableDownloadId",
            "startProcessing", "listOperations", "getOperation", "cancelOperation", "updateAsset", "deleteAsset")


class CloudCall(RuntimeError):
    """An Earth Engine cloud entry point was reached in offline mode."""


@contextlib.contextmanager
def offline_earth_engine(tasks):
    """ApiTestCase signatures; simulated tasks; every other cloud entry point raises."""
    import ee
    from ee.apitestcase import ApiTestCase

    case = ApiTestCase()
    case.setUp()

    def refuse(name):
        def call(*args, **kwargs):
            raise CloudCall(f"offline mode: Earth Engine call refused: {name}")
        return call

    try:
        with contextlib.ExitStack() as stack:
            for name in _REFUSED:
                if hasattr(ee.data, name):
                    stack.enter_context(mock.patch.object(ee.data, name, refuse(name)))
            for name in ("Initialize", "Authenticate"):
                stack.enter_context(mock.patch.object(ee, name, refuse(name)))
            stack.enter_context(mock.patch.object(ee.data, "newTaskId", tasks.new_task_id))
            stack.enter_context(mock.patch.object(ee.data, "exportImage", tasks.export_image))
            stack.enter_context(mock.patch.object(ee.data, "getTaskStatus", tasks.task_status))
            yield ee
    finally:
        case.tearDown()


def build(args, ee, tasks=None):
    """The offline controller for ``args`` (its fixture example unless ``--example`` is given)."""
    cls = OfflineSwinDemo if controller_class(args.model) is SwinDemo else OfflineDemo
    app = cls(args, ee)
    if app.example_record is None:
        app.example_record = fixture_example(args.model, args.project)
    return app


def main(args):
    """Serve the fixture (called by ``demos.browser.server --offline``)."""
    if args.state_dir is None:
        args.state_dir = Path(tempfile.mkdtemp(prefix="browser-demo-offline-"))
    for name in ("checkpoint", "manifest"):
        if not Path(getattr(args, name)).is_file():
            print(f"Warning: --{name} {getattr(args, name)} is missing; runs will fail while preparing "
                  "their graphs. Pass the real file to exercise graph construction.", flush=True)
    print(f"Offline state directory: {args.state_dir}", flush=True)
    tasks = SimulatedTasks(args.project, Path(args.state_dir) / "offline_tasks.json")
    with offline_earth_engine(tasks) as ee:
        serve(build(args, ee, tasks), args.port)
