"""Live fixture harness (task P00): build, plan, check and run small inline block fixtures.

    PYTHONPATH=. python tools/live_fixtures.py --list
    PYTHONPATH=. python tools/live_fixtures.py --plan calibration
    PYTHONPATH=. python tools/live_fixtures.py --offline calibration
    PYTHONPATH=. python tools/live_fixtures.py --run calibration --project YOUR_EE_PROJECT

A fixture (``tools/fixtures``) builds one block's Earth Engine graph on a small input that
lives in the graph itself, with NumPy weights embedded in the request, and reads its
outputs back as float32 bands on small computePixels grids. A selection is fixture ids,
``fnmatch`` patterns (``window/*``), ``calibration``, ``all``, or a lane's named selection
(``SELECTIONS`` in its fixture module, such as ``p02``).

- ``--list``: every fixture, its block, settings, policy and request count.
- ``--plan``: every request a run would make (grid, bands, values, expected bytes, request
  bytes, graph SHA-256, the blocks' embedded-constant count from their ``METADATA``) and
  the totals against the budget. Zero cloud calls: graphs are built against the offline
  signatures of ``ee.apitestcase.ApiTestCase`` with every cloud entry point refused.
- ``--offline``: ``--plan``'s graph building, plus every PyTorch reference and NumPy
  mirror, comparing mirror with PyTorch (the G3-offline mechanism). Zero cloud calls.
- ``--run``: the live run. It refuses to start when the plan exceeds the budget or a
  reference fails, initializes Earth Engine with Application Default Credentials for
  ``--project``, rebuilds every graph and requires it to equal the planned one, executes
  each request once with ``ee.data.computePixels`` (``NUMPY_NDARRAY``, no client retries;
  the HTTP layer may re-send after a credential refresh or a dropped connection), compares
  with PyTorch and the mirror, and writes one evidence JSON under
  ``docs/evidence/live-fixtures/``. ``ee.data`` is wrapped for the whole run: any export,
  asset, task or other non-compute call fails before it is sent, and the transport lets
  only ``algorithms.list`` (initialization, at most 2) and ``image.computePixels`` through.

The budget (R4) is enforced in code: at most 40 requests per invocation; per request a grid
of at most 64 x 64 pixels (raster lane) or 8 x 8 windows (window lane), at most 1,000,000
output values and at most 2,000,000 bytes of request; zero assets and zero tasks; requests
are sent one at a time (the limit is 4 in flight). ``--budget key=value,...`` may lower a
limit, never raise it. A request's NPY response header (``npy_header_length``) also stays
within ``NPY_HEADER_LIMIT``, below the 10,000 bytes ``numpy.load`` accepts by default: readouts
on one grid are split over more requests when they would exceed it, and ``--plan`` refuses a
readout that exceeds it alone.

Exit codes: 0 pass; 1 a fixture failed or the run stopped after a cloud call (evidence is
written); 2 refused before any cloud call (budget, plan or local reference).
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime
import hashlib
import importlib.metadata
import inspect
import io
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from typing import Any, Callable, Iterator
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "tests", ROOT):            # ROOT first: the checkout's tensor_ee and tools
    if str(_path) in sys.path:
        sys.path.remove(str(_path))
    sys.path.insert(0, str(_path))

import ee  # noqa: E402
import numpy as np  # noqa: E402

import tensor_ee  # noqa: E402

if not Path(tensor_ee.__file__).resolve().is_relative_to(ROOT):
    raise SystemExit(f"tensor_ee must be imported from this checkout ({ROOT.name}), not "
                     f"{Path(tensor_ee.__file__).parent.name}")

from tools.fixtures import registry, select  # noqa: E402
from tools.fixtures.core import Fixture, FixtureData, Readout, jsonable  # noqa: E402

SCHEMA_VERSION = 1
EVIDENCE_DIR = ROOT / "docs" / "evidence" / "live-fixtures"
VERSION_PACKAGES = ("earthengine-api", "torch", "timm", "segmentation-models-pytorch", "numpy")
MAX_BANDS = 1024
"""computePixels refuses more than 1,024 bands (Milestone 9, section 14)."""
FIRST_MISMATCHES = 10
NUMPY_MAX_HEADER = inspect.signature(np.load).parameters["max_header_size"].default
"""NumPy's default ``max_header_size`` (10,000): the client loads a ``NUMPY_NDARRAY`` response
with ``numpy.load`` and these defaults, and refuses a longer NPY header (P01's live run)."""
NPY_HEADER_LIMIT = 9_000
"""The longest NPY header a request may have: NumPy's limit with a 10% margin. A request's
header grows with its band names (one structured float32 field per band)."""


# ---------------------------------------------------------------------------
# Budget (R4)
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Budget:
    """Per-invocation limits. Assets and tasks are always zero."""

    requests: int = 40
    raster_side: int = 64          # pixels per side of a raster-lane request grid
    window_side: int = 8           # windows per side of a window-lane request grid
    values: int = 1_000_000        # output values per request
    request_bytes: int = 2_000_000  # serialized request per request
    in_flight: int = 4             # requests in flight (this harness sends one at a time)
    assets: int = 0
    tasks: int = 0

    def record(self) -> dict:
        return dataclasses.asdict(self)


DEFAULT_BUDGET = Budget()
LOWERABLE = ("requests", "raster_side", "window_side", "values", "request_bytes", "in_flight")


def parse_budget(text: str | None, base: Budget = DEFAULT_BUDGET) -> Budget:
    """``key=value[,key=value...]``: each value a positive integer no larger than the default.
    Raising a limit is a lead change recorded in the plan, not a command-line option."""
    if not text:
        return base
    changes = {}
    for item in text.split(","):
        key, sep, value = item.partition("=")
        key = key.strip()
        if not sep or key not in LOWERABLE:
            raise ValueError(f"budget items are key=value with key one of {', '.join(LOWERABLE)}; "
                             f"received {item!r}")
        if not re.fullmatch(r"[0-9_]+", value.strip()):
            raise ValueError(f"budget {key} must be a positive integer; received {value!r}")
        number = int(value.strip())
        limit = getattr(DEFAULT_BUDGET, key)
        if number < 1:
            raise ValueError(f"budget {key} must be at least 1")
        if number > limit:
            raise ValueError(f"budget {key}={number} would raise the limit {limit}; only the lead may "
                             "raise a limit, by changing the plan")
        changes[key] = number
    return dataclasses.replace(base, **changes)


# ---------------------------------------------------------------------------
# Errors and the offline Earth Engine (zero cloud calls)
# ---------------------------------------------------------------------------
class CloudCall(AssertionError):
    """An Earth Engine cloud entry point was reached while building offline."""


class RefusedCall(RuntimeError):
    """A call that a live fixture run never makes (export, asset, task, other compute)."""


class BudgetExceeded(RuntimeError):
    """More requests than planned, or a plan over the budget."""


class Refused(RuntimeError):
    """The run was refused before any cloud call; ``reasons`` says why."""

    def __init__(self, reasons: list[str]):
        super().__init__("; ".join(reasons))
        self.reasons = reasons


_REFUSED_DATA_CALLS = ("_execute_cloud_call", "getAsset", "getInfo", "listAssets", "computePixels",
                       "computeImages", "computeFeatures", "computeValue", "getMapId",
                       "getDownloadId", "getThumbId", "getTableDownloadId", "startProcessing",
                       "newTaskId", "getTaskStatus", "listOperations", "getOperation",
                       "exportImage", "exportTable", "exportVideo", "exportMap", "exportClassifier",
                       "createAsset", "createFolder", "copyAsset", "renameAsset", "deleteAsset",
                       "startIngestion", "startTableIngestion", "cancelTask", "cancelOperation")


@contextlib.contextmanager
def offline_earth_engine() -> Iterator[list[str]]:
    """ApiTestCase signatures, with every cloud entry point refusing to run (as
    ``tools/graph_identity.py``). Yields the attempted calls; they must stay empty."""
    from ee.apitestcase import ApiTestCase

    # ApiTestCase.InitializeApi replaces ee.data._install_cloud_api_resource with a no-op
    # and its tearDown does not restore it; it also leaves its stand-in algorithms loaded.
    # Without the restore and reset below, a later real ee.Initialize in this process
    # installs no Cloud API resource, and every live request fails "not initialized".
    install_cloud_api_resource = ee.data._install_cloud_api_resource
    case = ApiTestCase()
    case.setUp()
    attempts: list[str] = []

    def refuse(name):
        def call(*args, **kwargs):
            attempts.append(name)
            raise CloudCall(f"cloud call attempted while building offline: {name}")
        return call

    try:
        with contextlib.ExitStack() as stack:
            for name in _REFUSED_DATA_CALLS:
                if hasattr(ee.data, name):
                    stack.enter_context(mock.patch.object(ee.data, name, refuse(name)))
            for name in ("Initialize", "Authenticate"):
                stack.enter_context(mock.patch.object(ee, name, refuse(name)))
            yield attempts
    finally:
        case.tearDown()
        ee.data._install_cloud_api_resource = install_cloud_api_resource
        ee.Reset()
    if attempts:
        raise CloudCall(f"cloud calls attempted: {attempts}")


# ---------------------------------------------------------------------------
# Requests: readouts grouped by grid
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Request:
    fixture: str
    index: int
    lane: str
    grid: dict
    readouts: list
    image: Any
    bands: list
    values: int
    payload_bytes: int
    request_bytes: int
    graph_sha256: str
    graph_bytes: int
    embedded_constants: int
    inline_values: int
    npy_header: int = 0

    @property
    def width(self) -> int:
        return int(self.grid["dimensions"]["width"])

    @property
    def height(self) -> int:
        return int(self.grid["dimensions"]["height"])

    def params(self) -> dict:
        """The computePixels parameters the run sends."""
        return {"expression": self.image, "fileFormat": "NUMPY_NDARRAY", "grid": self.grid}

    def record(self) -> dict:
        return {"index": self.index, "grid": self.grid, "width": self.width, "height": self.height,
                "readouts": [{"name": r.name, "kind": r.kind, "channels": r.channels} for r in self.readouts],
                "bands": len(self.bands), "values": self.values, "payload_bytes": self.payload_bytes,
                "request_bytes": self.request_bytes, "graph_sha256": self.graph_sha256,
                "graph_bytes": self.graph_bytes, "embedded_constants": self.embedded_constants,
                "inline_values": self.inline_values}


def _grid_key(grid: dict) -> str:
    return json.dumps(grid, sort_keys=True)


def npy_header_length(bands, height: int, width: int) -> int:
    """The NPY header length that ``numpy.load`` checks against ``max_header_size`` for a
    ``NUMPY_NDARRAY`` response: the header ``numpy.lib.format`` writes (version 1.0, or 2.0
    when 1.0 cannot hold it) for a structured float32 array with one field per band on a
    ``height`` x ``width`` grid. Computed exactly, from the band names and the dtype."""
    dtype = np.dtype([(name, "<f4") for name in bands])
    header = {"descr": np.lib.format.dtype_to_descr(dtype), "fortran_order": False,
              "shape": (int(height), int(width))}
    buffer = io.BytesIO()
    try:
        np.lib.format.write_array_header_1_0(buffer, header)
        prefix = 10                  # magic (6), version (2), header length (2)
    except ValueError:
        buffer = io.BytesIO()
        np.lib.format.write_array_header_2_0(buffer, header)
        prefix = 12                  # magic (6), version (2), header length (4)
    return len(buffer.getvalue()) - prefix


def _readout_bands(r: Readout) -> list[str]:
    return [f"{r.name}_{c}" for c in range(r.channels)] + [f"{r.name}_valid"]


def _pack(group: list, height: int, width: int) -> list[list]:
    """Readouts on one grid, in order, as few requests as keep every NPY header within
    ``NPY_HEADER_LIMIT``. A readout and its validity band stay in one request. A readout
    over the limit alone gets a request of its own, which the budget check refuses."""
    chunks, chunk, bands = [], [], []
    for r in group:
        mine = _readout_bands(r)
        if chunk and npy_header_length(bands + mine, height, width) > NPY_HEADER_LIMIT:
            chunks.append(chunk)
            chunk, bands = [], []
        chunk.append(r)
        bands += mine
    chunks.append(chunk)
    return chunks


def build_requests(fixture: Fixture, data: FixtureData) -> list[Request]:
    """The fixture's readouts as computePixels requests: readouts on the same grid share one
    request, unless together their response's NPY header would exceed ``NPY_HEADER_LIMIT``;
    then they are split, in order, over more requests (``_pack``). Each readout contributes
    its C value bands ``<name>_<c>`` (masked pixels unmasked to 0) and one validity band
    ``<name>_valid`` (1 where every band is valid)."""
    readouts = fixture.build(ee, data)
    names = [r.name for r in readouts]
    if len(set(names)) != len(names):
        raise ValueError(f"{fixture.id}: readout names must be unique; received {names}")
    groups: dict[str, list[Readout]] = {}
    for readout in readouts:
        if readout.kind == "input" and readout.source not in data.inputs:
            raise ValueError(f"{fixture.id}: readout {readout.name} echoes unknown input {readout.source}")
        groups.setdefault(_grid_key(readout.grid), []).append(readout)
    constants = int(sum(fixture.constants(data).values()))
    chunks = [chunk for group in groups.values() for chunk in _pack(group, *group[0].shape[:2])]
    requests = []
    for index, group in enumerate(chunks):
        parts, bands = [], []
        for r in group:
            value_names = [f"{r.name}_{c}" for c in range(r.channels)]
            parts.append(r.image.rename(value_names).unmask(0, False))
            parts.append(r.image.mask().reduce(ee.Reducer.min()).rename(f"{r.name}_valid")
                         .unmask(0, False).toFloat())
            bands += value_names + [f"{r.name}_valid"]
        image = ee.Image.cat(parts)
        text = image.serialize()
        grid = dict(group[0].grid)
        body = {"expression": ee.serializer.encode(image), "fileFormat": "NPY", "grid": grid}
        height, width = group[0].shape[:2]
        values = width * height * len(bands)
        requests.append(Request(
            fixture=fixture.id, index=index, lane=fixture.lane, grid=grid, readouts=group, image=image,
            bands=bands, values=values, payload_bytes=4 * values,
            request_bytes=len(json.dumps(body).encode()), graph_sha256=hashlib.sha256(text.encode()).hexdigest(),
            graph_bytes=len(text.encode()),
            embedded_constants=constants if any(r.kind == "output" for r in group) else 0,
            inline_values=data.inline_values, npy_header=npy_header_length(bands, height, width)))
    return requests


# ---------------------------------------------------------------------------
# Plan (R3, R4)
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Plan:
    fixtures: list
    data: dict
    requests: dict
    budget: Budget
    violations: list

    @property
    def all_requests(self) -> list[Request]:
        return [r for f in self.fixtures for r in self.requests[f.id]]

    def totals(self) -> dict:
        requests = self.all_requests
        return {"fixtures": len(self.fixtures), "requests": len(requests),
                "values": sum(r.values for r in requests),
                "payload_bytes": sum(r.payload_bytes for r in requests),
                "request_bytes": sum(r.request_bytes for r in requests),
                "max_values": max((r.values for r in requests), default=0),
                "max_request_bytes": max((r.request_bytes for r in requests), default=0),
                "max_bands": max((len(r.bands) for r in requests), default=0),
                "embedded_constants": sum(r.embedded_constants for r in requests),
                "inline_values": sum(r.inline_values for r in requests),
                "assets": 0, "tasks": 0, "in_flight": 1}

    def record(self) -> dict:
        return {"budget": self.budget.record(), "totals": self.totals(), "violations": list(self.violations),
                "fixtures": [{"id": f.id, "block": f.block, "requests": [r.record() for r in self.requests[f.id]],
                              "inputs_sha256": self.data[f.id].inputs_sha256,
                              "weights_sha256": self.data[f.id].weights_sha256}
                             for f in self.fixtures]}


def side_limit(lane: str, budget: Budget, window: int) -> int:
    """Largest request grid side: pixels (raster lane) or windows x window (window lane)."""
    return budget.raster_side if lane == "raster" else budget.window_side * window


def check_budget(fixtures, requests: dict, budget: Budget) -> list[str]:
    violations = []
    for fixture in fixtures:
        built = requests[fixture.id]
        if len(built) != fixture.requests:
            violations.append(f"{fixture.id}: declares {fixture.requests} requests but builds {len(built)}")
        limit = side_limit(fixture.lane, budget, fixture.window)
        for r in built:
            where = f"{fixture.id} request {r.index}"
            if r.width > limit or r.height > limit:
                violations.append(f"{where}: grid {r.width} x {r.height} exceeds {limit} per side "
                                  f"({fixture.lane} lane)")
            if r.values > budget.values:
                violations.append(f"{where}: {r.values:,} output values exceed {budget.values:,}")
            if r.request_bytes > budget.request_bytes:
                violations.append(f"{where}: {r.request_bytes:,} request bytes exceed {budget.request_bytes:,}")
            if len(r.bands) > MAX_BANDS:
                violations.append(f"{where}: {len(r.bands)} bands exceed computePixels' {MAX_BANDS}")
            if r.npy_header > NPY_HEADER_LIMIT:
                alone = (f"; readout {r.readouts[0].name} alone ({r.readouts[0].channels} channels) is over "
                         "the limit, so it cannot be split: give it fewer channels"
                         if len(r.readouts) == 1 else "")
                violations.append(f"{where}: its NPY response header would be {r.npy_header:,} bytes, over "
                                  f"{NPY_HEADER_LIMIT:,} (numpy.load refuses more than {NUMPY_MAX_HEADER:,}){alone}")
    total = sum(len(v) for v in requests.values())
    if total > budget.requests:
        violations.append(f"{total} requests exceed the budget of {budget.requests}")
    if budget.assets or budget.tasks:
        violations.append("assets and tasks must be zero")
    return violations


def make_plan(fixtures: list[Fixture], budget: Budget = DEFAULT_BUDGET) -> Plan:
    """Build every graph offline (zero cloud calls) and check the budget. Fixture ids must be
    unique: the plan is keyed by id, so a repeated fixture would be counted once."""
    ids = [f.id for f in fixtures]
    repeated = sorted({i for i in ids if ids.count(i) > 1})
    if repeated:
        raise ValueError(f"duplicate fixture ids: {repeated}")
    data = {f.id: f.data(f.seed) for f in fixtures}
    with offline_earth_engine():
        requests = {f.id: build_requests(f, data[f.id]) for f in fixtures}
    return Plan(fixtures, data, requests, budget, check_budget(fixtures, requests, budget))


# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------
def _location(index, shape) -> dict:
    row, col, channel = (int(i) for i in np.unravel_index(int(index), shape))
    return {"row": row, "col": col, "channel": channel}


def compare_reference(actual: np.ndarray, expected: np.ndarray, tolerance: float) -> dict:
    """Largest absolute difference from the reference and the number of values over
    ``tolerance`` (a non-finite value counts as over). A tolerance of 0 means equality: the
    float32 bits are compared, so -0.0 against +0.0 counts as over."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        return {"pass": False, "error": f"shape {list(actual.shape)} != reference {list(expected.shape)}"}
    a, e = actual.astype(np.float64), expected.astype(np.float64)
    finite = np.isfinite(a) & np.isfinite(e)
    diff = np.where(finite, np.abs(a - e), np.inf)
    if tolerance == 0:
        bits = actual.astype(np.float32).view(np.uint32) != expected.astype(np.float32).view(np.uint32)
        over_mask = bits | ~finite
        # The worst value: the largest difference among the differing ones (a sign of zero
        # differs with a difference of 0).
        worst = int(np.argmax(np.where(over_mask, diff, -1.0))) if diff.size else 0
    else:
        over_mask = diff > tolerance
        worst = int(np.argmax(diff)) if diff.size else 0
    over = int(np.count_nonzero(over_mask))
    result = {"tolerance": tolerance, "comparison": "bits" if tolerance == 0 else "absolute",
              "values": int(diff.size), "max_abs": float(diff.max()) if diff.size else 0.0,
              "over_tolerance": over, "pass": over == 0}
    if diff.size:
        result["worst"] = {**_location(worst, diff.shape), "actual": float(a.reshape(-1)[worst]),
                           "expected": float(e.reshape(-1)[worst])}
    return result


def compare_mirror(actual: np.ndarray, mirror: np.ndarray, exact: bool, bound: float | None) -> dict:
    """Values whose float32 bits differ from the mirror, and the largest difference. An exact
    mirror passes only with zero differing values; another one within its recorded bound."""
    actual, mirror = np.asarray(actual), np.asarray(mirror)
    if actual.shape != mirror.shape:
        return {"pass": False, "error": f"shape {list(actual.shape)} != mirror {list(mirror.shape)}"}
    a32, m32 = actual.astype(np.float32), mirror.astype(np.float32)
    differ = a32.view(np.uint32) != m32.view(np.uint32)
    diff = np.abs(a32.astype(np.float64) - m32.astype(np.float64))
    diff = np.where(np.isnan(diff), np.inf, diff)
    count = int(np.count_nonzero(differ))
    max_abs = float(diff.max()) if diff.size else 0.0
    first = [{**_location(i, differ.shape), "actual": float(a32.reshape(-1)[i]), "mirror": float(m32.reshape(-1)[i])}
             for i in np.flatnonzero(differ)[:FIRST_MISMATCHES]]
    passed = count == 0 if exact else max_abs <= bound
    return {"exact": exact, "bound": bound, "values": int(differ.size), "differing": count,
            "max_abs": max_abs, "first": first, "pass": bool(passed)}


# ---------------------------------------------------------------------------
# Local references and mirrors (no Earth Engine)
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Local:
    reference: dict | None = None
    mirror: dict | None = None
    errors: list = dataclasses.field(default_factory=list)
    seconds: float = 0.0


def _message(error: BaseException) -> str:
    """An exception as text, without the user's home folder or the checkout's location."""
    text = f"{type(error).__name__}: {error}"
    for path, label in ((str(ROOT), "<checkout>"), (str(Path.home()), "~")):
        text = text.replace(path, label)
    return text


def evaluate_local(fixture: Fixture, data: FixtureData) -> Local:
    """The PyTorch reference and the NumPy mirror of every output readout, inside
    ``offline_earth_engine()``: a reference or mirror that reaches Earth Engine is refused
    (and recorded as an error), whether or not the client is initialized."""
    local = Local()
    start = time.perf_counter()
    try:
        with offline_earth_engine():
            try:
                local.reference = {k: np.asarray(v, np.float32) for k, v in fixture.reference(data).items()}
            except Exception as error:  # recorded, refuses the run
                local.errors.append(f"reference: {_message(error)}")
            if fixture.mirror is not None:
                try:
                    local.mirror = {k: np.asarray(v, np.float32) for k, v in fixture.mirror.fn(data).items()}
                except Exception as error:
                    local.errors.append(f"mirror: {_message(error)}")
    except CloudCall as error:
        local.errors.append(f"Earth Engine call refused: {_message(error)}")
    local.seconds = time.perf_counter() - start
    return local


def expected_outputs(fixture: Fixture, requests: list[Request], local: Local) -> dict:
    """Every output readout's reference, checking that it exists and has the readout's shape."""
    shapes = {r.name: r.shape for q in requests for r in q.readouts if r.kind == "output"}
    for name, shape in shapes.items():
        if local.reference is None or name not in local.reference:
            raise KeyError(f"{fixture.id}: no reference for output {name}")
        if local.reference[name].shape != shape:
            raise ValueError(f"{fixture.id}: reference {name} has shape {local.reference[name].shape}, "
                             f"readout {shape}")
        if local.mirror is not None and name in local.mirror and local.mirror[name].shape != shape:
            raise ValueError(f"{fixture.id}: mirror {name} has shape {local.mirror[name].shape}, readout {shape}")
    return shapes


def offline_comparison(fixture: Fixture, requests: list[Request], local: Local) -> dict:
    """Mirror against PyTorch for every output (G3-offline)."""
    outputs = {}
    for name in expected_outputs(fixture, requests, local):
        reference = local.reference[name]
        if local.mirror is not None and name in local.mirror:
            outputs[name] = {"mirror_vs_pytorch": compare_reference(local.mirror[name], reference, fixture.tolerance)}
        else:
            outputs[name] = {"mirror_vs_pytorch": None, "note": "no mirror"}
    return outputs


def _status(passed: bool) -> str:
    return "passed" if passed else "failed"


def offline(fixtures: list[Fixture], budget: Budget = DEFAULT_BUDGET) -> dict:
    """``--offline``: build every graph and run every reference and mirror; zero cloud calls."""
    plan = make_plan(fixtures, budget)
    results = []
    for fixture in fixtures:
        data, requests = plan.data[fixture.id], plan.requests[fixture.id]
        local = evaluate_local(fixture, data)
        entry = {"spec": fixture.record(), "inputs_sha256": data.inputs_sha256,
                 "weights_sha256": data.weights_sha256, "requests": [r.record() for r in requests],
                 "errors": list(local.errors), "seconds": round(local.seconds, 3)}
        if not local.errors:
            try:
                entry["outputs"] = offline_comparison(fixture, requests, local)
            except (KeyError, ValueError) as error:
                entry["errors"].append(_message(error))
        compared = [o["mirror_vs_pytorch"]["pass"] for o in entry.get("outputs", {}).values()
                    if o["mirror_vs_pytorch"] is not None]
        entry["status"] = _status(not entry["errors"] and all(compared))
        results.append(entry)
    return {"schema_version": SCHEMA_VERSION, "kind": "offline_fixture_check", "cloud_calls": 0,
            "status": _status(all(r["status"] == "passed" for r in results)),
            **provenance(), "plan": {"totals": plan.totals(), "violations": plan.violations},
            "fixtures": results}


# ---------------------------------------------------------------------------
# The live guard (R3): exports, assets and tasks fail before they are sent
# ---------------------------------------------------------------------------
COMPUTE_METHOD = "earthengine.projects.image.computePixels"
ALLOWED_METHODS = {"earthengine.projects.algorithms.list": "initialization", COMPUTE_METHOD: "compute"}
"""The only Cloud API methods a run sends: the client's algorithm list at initialization,
and computePixels."""

CLIENT_FUNCTIONS = frozenset({
    "computePixels", "getAlgorithms", "initialize", "is_initialized", "reset", "get_persistent_credentials",
    "getWorkloadTag", "setWorkloadTag", "resetWorkloadTag", "setDefaultWorkloadTag", "workloadTagContext",
    "setDeadline", "setMaxRetries", "getUserAgent", "setUserAgent", "setCloudApiUserProject",
    "setCloudApiKey", "authorizeHttp", "convert_asset_id_to_asset_name", "profiling"})
"""Public ``ee.data`` functions left callable during a run: computePixels, and client
plumbing that sends nothing itself. Every other public function (exports, assets, tasks,
operations, downloads, map tiles, thumbnails, other computations) is refused."""


def refused_functions() -> list[str]:
    """Every public name in ``ee.data`` bound to a callable that is not a class, outside
    ``CLIENT_FUNCTIONS``, whatever is bound to it now: a function, a bound method or any
    other callable stand-in (such as ``ApiTestCase``'s) is refused too."""
    return sorted(name for name, value in vars(ee.data).items()
                  if not name.startswith("_") and callable(value) and not isinstance(value, type)
                  and name not in CLIENT_FUNCTIONS)


MAX_ALGORITHMS_LIST = 2
"""``algorithms.list`` calls allowed per run: ``ee.Initialize`` makes one (no pagination)."""


@dataclasses.dataclass
class Guard:
    """Wraps ``ee.data`` for a live run and records every call it lets through or refuses."""

    max_compute: int
    requests: int = 0               # computePixels calls made by the harness
    compute: int = 0                # computePixels requests that reached the transport
    initialization: int = 0         # algorithms.list requests that reached the transport
    sent: list = dataclasses.field(default_factory=list)
    refused: list = dataclasses.field(default_factory=list)

    def compute_pixels(self, params: dict):
        """One planned request: ``ee.data.computePixels``, refused beyond the plan's count."""
        if self.requests >= self.max_compute:
            self.refused.append("computePixels")
            raise BudgetExceeded(f"more than the {self.max_compute} planned requests")
        self.requests += 1
        return ee.data.computePixels(params)

    @contextlib.contextmanager
    def transport(self) -> Iterator["Guard"]:
        """Every Cloud API request passes ``ee.data._execute_cloud_call``: allow only
        ``ALLOWED_METHODS``, at most ``max_compute`` computePixels and
        ``MAX_ALGORITHMS_LIST`` algorithm lists, each executed once by the client
        (``num_retries=0``). The HTTP layer below may still re-send a request after a
        credential refresh or a dropped connection."""
        original = ee.data._execute_cloud_call

        def execute(call, num_retries=None):
            method = getattr(call, "methodId", None)
            kind = ALLOWED_METHODS.get(method)
            if kind is None:
                self.refused.append(str(method))
                raise RefusedCall(f"{method} is not allowed during a live fixture run")
            if kind == "compute":
                if self.compute >= self.max_compute:
                    self.refused.append(str(method))
                    raise BudgetExceeded(f"computePixels beyond the {self.max_compute} planned requests")
                self.compute += 1
            else:
                if self.initialization >= MAX_ALGORITHMS_LIST:
                    self.refused.append(str(method))
                    raise BudgetExceeded(f"algorithms.list beyond {MAX_ALGORITHMS_LIST} calls")
                self.initialization += 1
            self.sent.append(method)
            return original(call, num_retries=0)       # executed once by the client

        with mock.patch.object(ee.data, "_execute_cloud_call", execute):
            yield self

    @contextlib.contextmanager
    def functions(self) -> Iterator["Guard"]:
        """Refuse every public ``ee.data`` function outside ``CLIENT_FUNCTIONS``."""
        def refuse(name):
            def call(*args, **kwargs):
                self.refused.append(name)
                raise RefusedCall(f"ee.data.{name} is not allowed during a live fixture run "
                                  "(no exports, assets or tasks)")
            return call

        with contextlib.ExitStack() as stack:
            for name in refused_functions():
                stack.enter_context(mock.patch.object(ee.data, name, refuse(name)))
            yield self

    def record(self) -> dict:
        return {"requests": self.requests, "computePixels": self.sent.count(COMPUTE_METHOD),
                "algorithms_list": self.sent.count("earthengine.projects.algorithms.list"),
                "refused": list(self.refused)}


@contextlib.contextmanager
def live_session(project: str, credentials: str = "adc") -> Iterator[None]:
    """Initialize Earth Engine with Application Default Credentials for ``project``."""
    if credentials != "adc":
        raise ValueError("only --credentials adc is supported")
    import google.auth

    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/earthengine",
                                           "https://www.googleapis.com/auth/cloud-platform"])
    ee.Initialize(credentials=creds, project=project)
    yield


# ---------------------------------------------------------------------------
# Live evaluation
# ---------------------------------------------------------------------------
def read_bands(pixels, request: Request, readout: Readout) -> tuple[np.ndarray, dict]:
    """A readout's [H, W, C] float32 values from a NUMPY_NDARRAY response, with its checks."""
    names = [f"{readout.name}_{c}" for c in range(readout.channels)]
    fields = pixels.dtype.names or ()
    missing = [n for n in names + [f"{readout.name}_valid"] if n not in fields]
    if missing:
        raise KeyError(f"response lacks bands {missing}")
    if pixels.shape != (request.height, request.width):
        raise ValueError(f"response shape {pixels.shape} != grid {(request.height, request.width)}")
    dtypes = sorted({str(pixels.dtype.fields[n][0]) for n in names})
    values = np.stack([np.asarray(pixels[n]) for n in names], axis=-1)
    valid = np.asarray(pixels[f"{readout.name}_valid"])
    checks = {"dtypes": dtypes, "float32": dtypes == ["float32"],
              "invalid_pixels": int(np.count_nonzero(valid != 1))}
    return values.astype(np.float32), checks


def compare_live(fixture: Fixture, data: FixtureData, readout: Readout, values: np.ndarray, local: Local) -> dict:
    """One readout: against PyTorch (and the mirror), or, for an input echo, bit for bit."""
    if readout.kind == "input":
        expected = data.inputs[readout.source]
        return {"input": compare_mirror(values, expected, True, None)}
    result = {"pytorch": compare_reference(values, local.reference[readout.name], fixture.tolerance)}
    if local.mirror is not None and readout.name in local.mirror:
        result["mirror"] = compare_mirror(values, local.mirror[readout.name], fixture.mirror.exact,
                                          fixture.mirror.bound)
    return result


def simulate_response(request: Request, expected: dict) -> np.ndarray:
    """The structured array computePixels would return if Earth Engine produced ``expected``
    ({readout name: [H, W, C]}) with every pixel valid. For offline tests of ``--run``."""
    dtype = [(name, "<f4") for name in request.bands]
    out = np.zeros((request.height, request.width), dtype=dtype)
    for r in request.readouts:
        values = np.asarray(expected[r.name], np.float32)
        for c in range(r.channels):
            out[f"{r.name}_{c}"] = values[..., c]
        out[f"{r.name}_valid"] = 1.0
    return out


def provenance() -> dict:
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                                  check=True, timeout=60).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    versions = {"python": platform.python_version()}
    for package in VERSION_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    status = git("status", "--porcelain")
    return {"revision": git("rev-parse", "HEAD"), "dirty": None if status is None else bool(status),
            "tensor_ee": Path(tensor_ee.__file__).resolve().relative_to(ROOT).as_posix(),
            "versions": versions, "platform": {"system": platform.system(), "machine": platform.machine()}}


def _utc() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def run(fixtures: list[Fixture], *, project: str, budget: Budget = DEFAULT_BUDGET, credentials: str = "adc",
        session: Callable[[str, str], Any] | None = None, command: str | None = None) -> dict:
    """``--run``. Raises ``Refused`` before any cloud call when the plan exceeds the budget
    or a reference or mirror fails; otherwise returns the evidence record."""
    if not project or not isinstance(project, str):
        raise Refused(["--run needs --project"])
    started, clock = _utc(), time.perf_counter()
    try:
        plan = make_plan(fixtures, budget)
    except (ValueError, KeyError) as error:          # duplicate ids, malformed readouts
        raise Refused([_message(error)]) from error
    if plan.violations:
        raise Refused(plan.violations)
    locals_ = {}
    reasons = []
    for fixture in fixtures:
        local = evaluate_local(fixture, plan.data[fixture.id])
        if not local.errors:
            try:
                expected_outputs(fixture, plan.requests[fixture.id], local)
            except (KeyError, ValueError) as error:
                local.errors.append(_message(error))
        reasons += [f"{fixture.id}: {e}" for e in local.errors]
        locals_[fixture.id] = local
    if reasons:
        raise Refused(reasons)

    planned = len(plan.all_requests)
    guard = Guard(max_compute=planned)
    record = {"schema_version": SCHEMA_VERSION, "kind": "live_fixture_run", "status": "aborted",
              "command": command, "selection": [f.id for f in fixtures], "project": project,
              "credentials": credentials, **provenance(), "started_utc": started.isoformat(timespec="seconds"),
              "budget": budget.record(), "plan": plan.totals(), "max_in_flight": 1, "errors": [], "fixtures": []}
    entries = {}
    for fixture in fixtures:
        data = plan.data[fixture.id]
        entries[fixture.id] = {
            "spec": fixture.record(), "inputs_sha256": data.inputs_sha256, "weights_sha256": data.weights_sha256,
            "status": "not run", "wall_seconds": 0.0, "failures": [],
            "offline": offline_comparison(fixture, plan.requests[fixture.id], locals_[fixture.id]),
            "requests": [r.record() for r in plan.requests[fixture.id]], "outputs": {}}
    record["fixtures"] = list(entries.values())
    try:
        with guard.transport(), (session or live_session)(project, credentials), guard.functions():
            live = {}
            for fixture in fixtures:                    # every graph, before any request
                live[fixture.id] = build_requests(fixture, plan.data[fixture.id])
                built = [r.graph_sha256 for r in live[fixture.id]]
                if built != [r.graph_sha256 for r in plan.requests[fixture.id]]:
                    raise Refused([f"{fixture.id}: the graphs built after initialization differ from the plan"])
            for fixture in fixtures:
                _run_fixture(fixture, plan.data[fixture.id], live[fixture.id], locals_[fixture.id],
                             entries[fixture.id], guard)
        record["status"] = _status(all(e["status"] == "passed" for e in entries.values()))
    except Exception as error:                          # recorded; the run stops
        record["errors"].append(_message(error))
        record["status"] = "aborted"
    record["cloud_calls"] = guard.record()
    record["finished_utc"] = _utc().isoformat(timespec="seconds")
    record["wall_seconds"] = round(time.perf_counter() - clock, 3)
    return record


def _run_fixture(fixture: Fixture, data: FixtureData, requests: list[Request], local: Local, entry: dict,
                 guard: Guard) -> None:
    start = time.perf_counter()
    for request, request_record in zip(requests, entry["requests"]):
        t0 = time.perf_counter()
        try:
            pixels = guard.compute_pixels(request.params())
        except Exception as error:
            request_record.update(status="error", error=_message(error),
                                  wall_seconds=round(time.perf_counter() - t0, 3))
            entry["failures"].append(f"request {request.index}: {_message(error)}")
            if isinstance(error, (RefusedCall, BudgetExceeded)):
                raise
            continue
        request_record.update(status="ok", wall_seconds=round(time.perf_counter() - t0, 3))
        for readout in request.readouts:
            try:
                values, checks = read_bands(pixels, request, readout)
            except (KeyError, ValueError) as error:
                entry["outputs"][readout.name] = {"kind": readout.kind, "request": request.index,
                                                  "error": _message(error)}
                entry["failures"].append(f"{readout.name}: {_message(error)}")
                continue
            result = {"kind": readout.kind, "request": request.index, "shape": list(values.shape), **checks,
                      **compare_live(fixture, data, readout, values, local)}
            entry["outputs"][readout.name] = result
            if not checks["float32"]:
                entry["failures"].append(f"{readout.name}: bands are {checks['dtypes']}, not float32")
            if checks["invalid_pixels"]:
                entry["failures"].append(f"{readout.name}: {checks['invalid_pixels']} masked pixels")
            for which in ("input", "pytorch", "mirror"):
                if which in result and not result[which]["pass"]:
                    entry["failures"].append(_failure(readout.name, which, result[which]))
    expected = {r.name for q in requests for r in q.readouts}
    missing = sorted(expected - set(entry["outputs"]))
    if missing:
        entry["failures"].append(f"outputs not compared: {missing}")
    entry["wall_seconds"] = round(time.perf_counter() - start, 3)
    entry["status"] = _status(not entry["failures"])


def _failure(name: str, which: str, result: dict) -> str:
    if "error" in result:
        return f"{name} vs {which}: {result['error']}"
    if which == "pytorch":
        w = result["worst"]
        count = (f"{result['over_tolerance']} values differ bit for bit (tolerance 0)" if result["tolerance"] == 0
                 else f"{result['over_tolerance']} values over {result['tolerance']:g}")
        return (f"{name} vs PyTorch: {count}; largest {result['max_abs']:.3g} at row {w['row']}, col {w['col']}, "
                f"channel {w['channel']} (EE {w['actual']!r}, PyTorch {w['expected']!r})")
    first = result["first"][0] if result["first"] else None
    where = (f"; first at row {first['row']}, col {first['col']}, channel {first['channel']} "
             f"(EE {first['actual']!r}, expected {first['mirror']!r})") if first else ""
    label = "the input" if which == "input" else "the mirror"
    return f"{name} vs {label}: {result['differing']} values differ, largest {result['max_abs']:.3g}{where}"


def write_evidence(record: dict, output: Path | None, label: str) -> Path:
    """One JSON per run under ``docs/evidence/live-fixtures/`` (or ``output``)."""
    if output is None:
        stamp = datetime.datetime.fromisoformat(record["started_utc"]).strftime("%Y%m%dT%H%M%SZ")
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", label).strip("-") or "selection"
        output = EVIDENCE_DIR / f"{stamp}-{safe}.json"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(jsonable(record), indent=2, allow_nan=False) + "\n")
    return output


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def _settings_text(settings) -> str:
    return ", ".join(f"{k}={v}" for k, v in settings.items())


def list_text(fixtures: list[Fixture]) -> str:
    lines = [f"{len(fixtures)} fixtures"]
    for f in fixtures:
        mirror = "none" if f.mirror is None else ("exact" if f.mirror.exact else f"bound {f.mirror.bound:g}")
        lines.append(f"  {f.id}: {f.block} [{_settings_text(f.settings)}] policy {f.policy}; "
                     f"{f.requests} request{'s' * (f.requests != 1)}; mirror {mirror}"
                     f"{'; calibration' if f.calibration else ''}")
    return "\n".join(lines)


def plan_text(plan: Plan) -> str:
    lines = []
    for f in plan.fixtures:
        lines.append(f"{f.id} ({f.block}, {f.policy}); inputs {plan.data[f.id].inputs_sha256[:12]}, "
                     f"weights {plan.data[f.id].weights_sha256[:12]}")
        for r in plan.requests[f.id]:
            names = ", ".join(f"{x.name}[{x.channels}]" for x in r.readouts)
            lines.append(f"  request {r.index}: {r.height} x {r.width} grid (rows x columns), {len(r.bands)} bands ({names}); "
                         f"{r.values:,} values, {r.payload_bytes:,} bytes expected; request {r.request_bytes:,} "
                         f"bytes; graph {r.graph_sha256[:12]}; embedded constants {r.embedded_constants:,} "
                         f"(METADATA) + {r.inline_values:,} input values; NPY header {r.npy_header:,} bytes")
    t, b = plan.totals(), plan.budget
    largest_header = max((r.npy_header for r in plan.all_requests), default=0)
    lines.append(f"Totals: {t['requests']} requests (budget {b.requests}); largest request {t['max_values']:,} "
                 f"values (budget {b.values:,}) and {t['max_request_bytes']:,} bytes (budget {b.request_bytes:,}); "
                 f"largest NPY header {largest_header:,} bytes (limit {NPY_HEADER_LIMIT:,}); "
                 f"{t['values']:,} values and {t['payload_bytes']:,} bytes expected in all; "
                 f"{t['request_bytes']:,} request bytes in all; assets 0, tasks 0, in flight 1 (limit {b.in_flight})")
    lines.append("Within budget." if not plan.violations else
                 "OVER BUDGET:\n" + "\n".join(f"  {v}" for v in plan.violations))
    return "\n".join(lines)


def offline_text(report: dict) -> str:
    lines = []
    for entry in report["fixtures"]:
        lines.append(f"{entry['spec']['id']}: {entry['status']}")
        for error in entry["errors"]:
            lines.append(f"  error: {error}")
        for name, output in entry.get("outputs", {}).items():
            c = output["mirror_vs_pytorch"]
            lines.append(f"  {name}: " + ("no mirror (reference only)" if c is None else
                                          f"mirror vs PyTorch max {c['max_abs']:.3g}, {c['over_tolerance']} over "
                                          f"{c['tolerance']:g}"))
    lines.append(f"Offline check {report['status']} (zero cloud calls).")
    return "\n".join(lines)


def run_text(record: dict, path: Path) -> str:
    lines = [f"Run {record['status']}; evidence {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}"]
    for entry in record["fixtures"]:
        lines.append(f"  {entry['spec']['id']}: {entry['status']}")
        for failure in entry["failures"]:
            lines.append(f"    {failure}")
    for error in record.get("errors", []):
        lines.append(f"  error: {error}")
    calls = record.get("cloud_calls", {})
    lines.append(f"requests: {calls.get('requests', 0)}; refused calls: {calls.get('refused', [])}")
    return "\n".join(lines)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog="A SELECTION is fixture ids, fnmatch patterns (window/*), calibration, all, or a "
               "lane's named selection (SELECTIONS in tools/fixtures/<lane>.py, such as p02); "
               "commas also separate items.")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true", help="list every fixture")
    mode.add_argument("--plan", nargs="+", metavar="SELECTION", help="plan a run (zero cloud calls)")
    mode.add_argument("--offline", nargs="+", metavar="SELECTION",
                      help="build graphs and compare mirror with PyTorch (zero cloud calls)")
    mode.add_argument("--run", nargs="+", metavar="SELECTION", help="run live (needs --project)")
    p.add_argument("--project", help="Earth Engine project for --run (recorded in the evidence)")
    p.add_argument("--credentials", choices=("adc",), default="adc",
                   help="Application Default Credentials (the only option)")
    p.add_argument("--budget", help="lower limits: key=value[,...] with keys " + ", ".join(LOWERABLE))
    p.add_argument("--json", action="store_true", help="print --list or --plan as JSON")
    p.add_argument("--output", type=Path, help="evidence path for --run or --offline")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        budget = parse_budget(args.budget)
    except ValueError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    if args.list:
        fixtures = list(registry().values())
        print(json.dumps([f.record() for f in fixtures], indent=2) if args.json else list_text(fixtures))
        return 0
    tokens = args.plan or args.offline or args.run
    try:
        fixtures = select(tokens)
    except KeyError as error:
        print(f"refused: {error.args[0]}", file=sys.stderr)
        return 2
    if args.plan:
        plan = make_plan(fixtures, budget)
        print(json.dumps(jsonable(plan.record()), indent=2) if args.json else plan_text(plan))
        return 1 if plan.violations else 0
    if args.offline:
        report = offline(fixtures, budget)
        print(offline_text(report))
        if args.output:
            write = Path(args.output)
            write.parent.mkdir(parents=True, exist_ok=True)
            write.write_text(json.dumps(jsonable(report), indent=2, allow_nan=False) + "\n")
        return 0 if report["status"] == "passed" else 1
    if not args.project:
        print("refused: --run needs --project", file=sys.stderr)
        return 2
    command = "python tools/live_fixtures.py --run " + " ".join(tokens) + f" --project {args.project}" + (
        f" --budget {args.budget}" if args.budget else "")
    try:
        record = run(fixtures, project=args.project, budget=budget, credentials=args.credentials, command=command)
    except Refused as refused:
        print("refused before any cloud call:\n" + "\n".join(f"  {r}" for r in refused.reasons), file=sys.stderr)
        return 2
    path = write_evidence(record, args.output, "_".join(tokens))
    print(run_text(record, path))
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
