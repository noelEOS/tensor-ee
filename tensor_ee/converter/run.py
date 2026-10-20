"""The runner: a plan's stages executed in Earth Engine for one chip, resumably.

After ``convert`` and ``plan``, the
runner executes a plan for one chip: its stages in order, each boundary a stage writes exported
to an image asset that the stages after it read. It generalizes the two runners that ran live,
``tensor_ee/swin_workflow.py`` (Swin-T's stages A, B and C) and the maintainer's
``tools/segformer_b0_validate.py`` (SegFormer-B0's six cuts), and keeps what they proved:
requests prepared offline and pinned before anything is submitted, one request ID fixed per
request, the intent written before each submission, no resubmission, bounded polling, and each
saved asset's bands and grid checked before a stage reads it.

- ``prepare(directory, plan, graph, *, parameters, manifest, inputs, namespace, caps)``,
  offline. Builds every export request with the builder (``build_stage``) and the one
  export-request builder (``request.export_request``), and saves them with the run's files
  (below) before anything is submitted. Each request's bytes and SHA-256 are recorded, and a
  request over the plan's ``request_bytes`` limit, or a boundary over its ``bands`` limit, is
  refused. ``parameters`` binds every ``state_dict`` name the graph names to a ``ServerArray``
  read from the tables of ``manifest``, a ready table-asset manifest
  (``native_ee_featurecollection_weights_v2``, every shard ``COMPLETED``); each request's table
  reads must be among the manifest's tables. ``inputs`` gives each graph input as an image asset
  ID, which the first stage reads, or as an ``ee.Image`` the caller builds, which the run exports
  first, to ``f"{namespace}_{name}"``, so that every stage reads saved images, as the planner
  measured them. Each boundary is saved to ``f"{namespace}_{plan.names[edge]}"`` and its task
  described as ``f"{graph.name}_{plan.names[edge]}"``. No cloud call, no ``ee.data`` call.
- ``run(directory, *, data=None, caps=None)``: the explicit call that submits work, on a
  prepared run that has not started. ``resume(directory, ...)``: the same execution, on a run
  that started and stopped, in this process or another. ``data`` is the Earth Engine client
  (``ee.data`` by default, initialized by the caller, with the namespace's project); the runner
  calls only its ``exportImage``, ``getOperation`` and ``getAsset``, and its local
  ``setMaxRetries(0)`` (no client retries, set by ``run``, ``resume`` and ``recover`` alike)
  and ``getWorkloadTag``.
- ``status(directory)``: the run's progress, read from its files, without a cloud call.
- ``recover(directory, export, *, operation=None, absent=False)``: the caller's decision on an
  export whose submission outcome is unknown.
- ``CallGuard(caps, *, data=None, path=None)``: for the steps of a live run outside the runner
  (a weight upload with ``export_weights``, readbacks), a context manager that counts the calls
  of the named client functions before each call and refuses one past its cap
  (``CallCapReached``, the call not made), with the counts optionally kept in a file.

**One process per run directory.** ``run``, ``resume`` and ``recover`` hold the directory's lock
file (``run.lock``, created exclusively, with the process's ID and start time) and remove it
when they return or stop; a lock present when one starts is refused (``RunLocked``), naming the
process, so that two processes never write one state file. A process that died without
unwinding leaves its lock: the caller checks that no process holds the directory and removes
the file by hand. Never copy a prepared directory to run it twice: the copy has the same
request IDs and destinations.

**Execution.** Exports run one at a time, in order: the graph inputs to export, then each
stage's boundaries. Once per run, before the first submission, every table of the manifest is
checked to exist (``getAsset``), and every input asset to be an image with the input's bands on
the input's grid (CRS, dimensions, transform). Then for each export: its saved request is
checked against its recorded SHA-256; its destination must not exist (``getAsset`` answers HTTP
404, read from the error's status, never from its text); the intent (``SUBMITTING``) is written
before ``exportImage`` is called with the request's fixed ID; the operation it returns is
recorded (``SUBMITTED``); it is polled with ``getOperation``, sleeping ``caps.interval`` before
each call, until it is done; an error ends the run (``FAILED``); success (``SUCCEEDED``) is
followed by ``getAsset`` on the new asset, whose type, band names in order, float precision,
CRS, dimensions and affine transform must be those exported (``VERIFIED``) before the next
export is submitted. The run stops on the first failure, with its reason in the state file.

**Resume, never resubmit.** Every transition is written to the state file before the next call,
so after an interruption at any point a new process continues from the file: a ``VERIFIED``
export is skipped; one ``SUCCEEDED`` is verified; one ``SUBMITTED`` or ``RUNNING`` is polled,
not resubmitted; one ``PREPARED`` is submitted. One left ``SUBMITTING`` (the intent written, the
operation unknown: the call raised, its response named no operation, or the process stopped
before the response was written) stops the run and asks the caller (``DecisionNeeded``): they
find out whether its task exists and call ``recover`` with the operation's name (checked once
against the export's description, type and destination) or with ``absent=True``, which returns
the export to ``PREPARED``; it is then submitted with the same request ID, so Earth Engine starts
at most one operation for it (the API's documented request-ID semantics, which the tests exercise
only against a fake). A failed task is never retried: ``FAILED`` is final, and a new run is
prepared. Operations are pinned to the run's project, the namespace's: ``exportImage``'s answer
and ``recover``'s operation must be ``projects/PROJECT/operations/ID``. Every status response
must name the export's operation, and its description and type, when the response states them,
must be the export's. One that does not stops the run and changes nothing: the operation polled
is ``exports.<label>.operation`` in ``state.json``; the caller checks it against the task list
and resumes, or, if the recorded name is wrong (it came from ``recover``), corrects it there by
hand while no process holds the run; ``resume`` checks it again on its first poll.

**Caps** (``Caps``) are inputs: prepare records them, and ``run`` or ``resume`` may replace
them, which the history records. ``tasks``, ``status_calls`` and ``asset_calls`` bound the
``exportImage``, ``getOperation`` and ``getAsset`` calls of the whole run; the counts are in the
state file and add up across resumes, and each call is counted before it is made. ``seconds``
bounds one ``run`` or ``resume`` call's time, ``interval`` the wait before each status call; the
total polling of a run is bounded by ``status_calls``. A cap reached stops the run
(``STOPPED``); a running task is left running, never cancelled. The ``exportImage`` count and
the intent are one write, so a count never exists without its intent.

**The directory** (git-ignored run storage, chosen by the caller), written atomically
(fsynced, renamed, the directory fsynced):
- ``prepared.json``, written once: the graph's, plan's and manifest's identities (SHA-256 and a
  copy of each: ``graph.json``, ``plan.json``, ``manifest.json``), the namespace, the inputs, the
  caps, and every export: its label (the boundary's name), stage, edge, destination,
  description, request ID, request (file, bytes, SHA-256, and the bytes the plan measured), the
  assets it reads, and the bands and grid its asset must have;
- ``requests/<label>.json``: each request, exactly the bytes ``exportImage`` sends;
- ``state.json``: ``prepared.json``'s SHA-256, the run's ``status`` (``PREPARED``, ``RUNNING``,
  ``STOPPED``, ``FAILED`` or ``COMPLETED``) and ``reason``, the caps, the call counts, the assets
  checked, each export's state (``PREPARED``, ``SUBMITTING``, ``SUBMITTED``, ``RUNNING``,
  ``SUCCEEDED``, ``VERIFIED`` or ``FAILED``) with its operation, polls, last status and saved
  asset, and the history of events;
- ``run.lock``, while a process holds the directory.

**One chip.** A plan's stages run on the one chip the graph's input grid describes. An area of
many chips would need one prepared directory per chip (or one state file with a chip index),
the chips' tasks run concurrently under shared caps, and the halo and tiling rules of
``plan_tiles`` for any stage that can be tiled; the planner never tiles a stage with an EXTENT or
chip-edge dependent node, so such a stage runs per chip.

Imports and constructors make no call; ``prepare`` and ``status`` make none; ``run``,
``resume`` and ``recover`` call only the client they are given, and ``CallGuard`` only wraps it.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Mapping
from urllib.parse import parse_qs, urlparse
import uuid

import ee

from ..core.errors import reject
from ..core.tensor import Raster, ServerArray, band_names
from ..layouts.boundaries import grid_view
from .build import build_stage
from .graph import BlockGraph, canonical, grid_to_json
from .plan import Plan
from .request import ASSET_ID, DESCRIPTION, export_request, request_reads, wire

FORMAT = "native_ee.run"             # a format identifier: historical, never renamed (CONTRIBUTING)
VERSION = 1
MANIFEST_FORMAT = "native_ee_featurecollection_weights_v2"
STATES = ("PREPARED", "SUBMITTING", "SUBMITTED", "RUNNING", "SUCCEEDED", "VERIFIED", "FAILED")
"""An export's states, in the order a successful export passes through them (``FAILED`` aside)."""
NAME = re.compile(r"[A-Za-z0-9_-]+")
CALLS = ("exportImage", "getOperation", "getAsset")
LOCK = "run.lock"
"""The lock file a process holds in a run directory while it runs, resumes or recovers it."""


class RunStopped(RuntimeError):
    """The run stopped, with ``reason`` in its state file. ``status`` is ``STOPPED`` (the caller
    may act, then resume) or ``FAILED`` (final); ``export`` names the export it stopped at."""

    def __init__(self, message: str, *, status: str, export: str | None = None):
        super().__init__(message)
        self.status, self.export = status, export


class DecisionNeeded(RunStopped):
    """An export's submission outcome is unknown: the caller decides with ``recover``."""


class RunLocked(RuntimeError):
    """Another process holds the run directory (its lock file is present); nothing was read or
    written."""


class CallCapReached(BaseException):
    """``CallGuard`` refused a call past its cap; the call was not made. A ``BaseException``, as
    ``KeyboardInterrupt`` is, so that it passes through a caller's ``except Exception``: a step
    such as ``export_weights`` records a failed shard there, where a refused call started nothing
    and is resumed."""


@dataclass(frozen=True)
class Caps:
    """A run's caps (see the module docstring): ``tasks`` (``exportImage`` calls),
    ``status_calls`` (``getOperation``) and ``asset_calls`` (``getAsset``) for the whole run;
    ``seconds``, one ``run`` or ``resume`` call's time; ``interval``, the seconds slept before
    each status call (10 to 600)."""

    tasks: int
    status_calls: int
    asset_calls: int
    seconds: float
    interval: float = 30.0

    def __post_init__(self):
        for name in ("tasks", "status_calls", "asset_calls"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"caps.{name} must be a positive integer; received {value!r}")
        for name in ("seconds", "interval"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"caps.{name} must be a positive number of seconds; received {value!r}")
        if not 10 <= self.interval <= 600:
            raise ValueError(f"caps.interval must be 10 to 600 seconds; received {self.interval!r}")

    def to_json(self) -> dict:
        return {"tasks": self.tasks, "status_calls": self.status_calls, "asset_calls": self.asset_calls,
                "seconds": self.seconds, "interval": self.interval}

    @classmethod
    def from_json(cls, data: Mapping) -> "Caps":
        return cls(data["tasks"], data["status_calls"], data["asset_calls"], data["seconds"], data["interval"])


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------
def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _document(value) -> str:
    return json.dumps(value, indent=1, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"


def _write(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` durably: a temporary file, fsynced, renamed over ``path``, and
    the directory fsynced, so a reader sees the old file or the new one, whole."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="ascii") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextlib.contextmanager
def _locked(directory, call: str, clock):
    """Hold ``directory``'s lock file for the block: created exclusively (``O_EXCL``) with this
    process's ID and start time, removed when the block ends, normally or by an exception. A
    lock already present is refused (``RunLocked``): another process holds the directory, or one
    died without unwinding (killed, or out of power) and left it."""
    path = Path(directory) / LOCK
    holder = {"pid": os.getpid(), "started_unix": clock(), "call": call, "token": str(uuid.uuid4())}
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        try:
            held = json.loads(path.read_text(encoding="ascii"))
        except (OSError, ValueError):
            held = {}
        raise RunLocked(f"{path} is held by process {held.get('pid', '?')} ({held.get('call', '?')}, started at unix "
                        f"{held.get('started_unix', '?')}). One process runs a run directory at a time. If no process "
                        f"holds it (check that process {held.get('pid', '?')} is not running, on the machine that "
                        f"started it), remove {path} by hand, then resume.") from None
    with os.fdopen(descriptor, "w", encoding="ascii") as stream:
        stream.write(_document(holder))
        stream.flush()
        os.fsync(stream.fileno())
    try:
        yield
    finally:
        try:
            if json.loads(path.read_text(encoding="ascii")).get("token") == holder["token"]:
                path.unlink()
        except (OSError, ValueError):
            pass


class CallGuard:
    """Counts and caps the calls of the named functions of an Earth Engine client, for the steps
    of a live run outside the runner (a weight upload with ``export_weights``, readbacks).

    ``CallGuard(caps, *, data=None, path=None)``: ``caps`` maps function names of ``data``
    (``ee.data`` by default) to the most calls allowed. Inside ``with CallGuard(...) as guard``,
    each named function is replaced by one that counts the call before making it, and refuses it
    (``CallCapReached``, the call not made) once its count has reached its cap; the functions are
    restored when the block ends. Functions not named are not touched, so the caller names every
    cloud function the step calls. With ``path``, the counts are read from that file when the
    guard starts and written to it (atomically) before each call, so that they add up across
    processes, as a run's counts do. ``guard.counts`` and ``guard.report()`` give them."""

    def __init__(self, caps: Mapping[str, int], *, data=None, path=None):
        if not isinstance(caps, Mapping) or not caps:
            raise ValueError("caps map one or more function names to the most calls allowed")
        for name, cap in caps.items():
            if not isinstance(name, str) or isinstance(cap, bool) or not isinstance(cap, int) or cap < 0:
                raise ValueError(f"cap of {name!r} must be a non-negative integer; received {cap!r}")
        self.caps, self.data = dict(caps), ee.data if data is None else data
        self.path = None if path is None else Path(path)
        for name in self.caps:
            if not callable(getattr(self.data, name, None)):
                raise ValueError(f"{name!r} is not a function of the client")
        self.counts = {name: 0 for name in self.caps}
        if self.path is not None and self.path.exists():
            saved = json.loads(self.path.read_text(encoding="ascii")).get("counts", {})
            if set(saved) - set(self.caps):
                raise ValueError(f"{self.path} counts calls of {sorted(set(saved) - set(self.caps))}, which these caps "
                                 "do not name")
            self.counts.update(saved)
        self._originals = {}

    def report(self) -> dict:
        return {"caps": dict(self.caps), "counts": dict(self.counts)}

    def _counted(self, name: str, function):
        def call(*args, **kwargs):
            if self.counts[name] >= self.caps[name]:
                raise CallCapReached(f"the cap of {self.caps[name]} {name} calls is reached; the call was not made")
            self.counts[name] += 1
            if self.path is not None:
                _write(self.path, _document(self.report()))
            return function(*args, **kwargs)
        return call

    def __enter__(self) -> "CallGuard":
        if self._originals:
            raise RuntimeError("this guard is already in use")
        for name in self.caps:
            self._originals[name] = getattr(self.data, name)
            setattr(self.data, name, self._counted(name, self._originals[name]))
        return self

    def __exit__(self, *exc) -> None:
        for name, function in self._originals.items():
            setattr(self.data, name, function)
        self._originals = {}


def _load(directory: Path) -> tuple[dict, dict]:
    """The run's prepared record and state, refusing files that are not this format's, or a
    state that belongs to another preparation."""
    text = (directory / "prepared.json").read_text(encoding="ascii")
    prepared, state = json.loads(text), json.loads((directory / "state.json").read_text(encoding="ascii"))
    for name, document in (("prepared.json", prepared), ("state.json", state)):
        if document.get("format") != FORMAT or document.get("version") != VERSION:
            raise ValueError(f"{directory / name} is not a {FORMAT} version {VERSION} file")
    if state.get("prepared_sha256") != _sha256(text.encode("ascii")):
        raise ValueError(f"{directory / 'state.json'} belongs to another preparation: prepared.json has changed")
    return prepared, state


# ---------------------------------------------------------------------------
# Checks of what Earth Engine returns
# ---------------------------------------------------------------------------
def _not_found(error: BaseException) -> bool:
    """True when ``error`` came from an HTTP 404: the response status in the exception chain
    (the client raises its error while handling the HTTP error), never the message's text."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if getattr(getattr(error, "resp", None), "status", None) == 404:
            return True
        error = error.__cause__ or error.__context__
    return False


def _destination(uri: str) -> str | None:
    """The asset ID of an operation's destination URI (``swin_workflow``'s reading)."""
    if uri.startswith("projects/"):
        return uri
    parsed = urlparse(uri)
    if parsed.scheme == "https" and parsed.netloc == "code.earthengine.google.com":
        values = parse_qs(parsed.query).get("asset", [])
        if len(values) == 1:
            return values[0]
    return None


def _project(namespace: str) -> str:
    """The Cloud project of an asset namespace ``projects/PROJECT/assets/...``."""
    return namespace.split("/")[1]


def _name_problem(name, project: str) -> str | None:
    """Why ``name`` is not an operation of ``project``, or None."""
    if not isinstance(name, str) or not re.fullmatch(re.escape(f"projects/{project}/operations/") + r"[A-Za-z0-9_-]+",
                                                     name):
        return f"{str(name)[:200]!r} is not an operation of the run's project, projects/{project}/operations/ID"
    return None


def _operation_problem(operation, name: str, export: Mapping, *, destinations: bool = False) -> str | None:
    """Why ``operation`` is not the task of ``export``, or None."""
    if not isinstance(operation, Mapping) or operation.get("name") != name:
        return f"the status response for {name} names another operation: {str(operation)[:200]}"
    metadata = operation.get("metadata") or {}
    if metadata.get("description") not in (None, export["description"]):
        return (f"operation {name} is the task {metadata['description']!r}, not export {export['label']!r}'s "
                f"{export['description']!r}")
    if metadata.get("type") not in (None, "EXPORT_IMAGE"):
        return f"operation {name} is a {metadata['type']} task, not an image export"
    if destinations:
        found = [_destination(uri) for uri in metadata.get("destinationUris", [])]
        if found and found != [export["asset"]]:
            return f"operation {name} exports to {found}, not to {export['asset']}"
    return None


def _grid_problems(bands, grid: Mapping) -> list[str]:
    """How the grids of ``bands`` (``getAsset``'s band records) differ from ``grid`` (a pixel
    grid: CRS, dimensions, affine transform), at the first band that differs."""
    problems = []
    for band in bands:
        actual = band.get("grid") or {}
        affine = actual.get("affineTransform") or {}
        dimensions = actual.get("dimensions") or {}
        if actual.get("crsCode") != grid["crsCode"]:
            problems.append(f"band {band['id']} is in {actual.get('crsCode')!r}, not {grid['crsCode']!r}")
        if {k: int(dimensions.get(k, 0)) for k in ("width", "height")} != grid["dimensions"]:
            problems.append(f"band {band['id']} is {dimensions}, not {grid['dimensions']}")
        if any(float(affine.get(k, 0)) != float(v) for k, v in grid["affineTransform"].items()):
            problems.append(f"band {band['id']} has the transform {affine}, not {grid['affineTransform']}")
        if problems:
            break
    return problems


def _image_problems(asset: Mapping, expect: Mapping) -> list[str]:
    """How the saved image ``asset`` (``getAsset``'s answer) differs from what was exported."""
    if asset.get("type") != "IMAGE":
        return [f"it is a {asset.get('type')!r} asset, not an image"]
    bands = asset.get("bands") or []
    ids = [band.get("id") for band in bands]
    if ids != expect["bands"]:
        return [f"its bands are {ids[:8]}{' ...' if len(ids) > 8 else ''} ({len(ids)}), where "
                f"{expect['bands'][:8]}{' ...' if len(expect['bands']) > 8 else ''} ({len(expect['bands'])}) "
                "were exported"]
    for band in bands:
        if (band.get("dataType") or {}).get("precision") != "FLOAT":
            return [f"band {band['id']} is {(band.get('dataType') or {}).get('precision')!r}, not FLOAT"]
    return _grid_problems(bands, expect["grid"])


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------
def _check_plan(plan: Plan, graph: BlockGraph) -> None:
    if not isinstance(plan, Plan):
        raise TypeError(f"plan must be a Plan (tensor_ee.converter.plan); received {type(plan).__name__}")
    if not isinstance(graph, BlockGraph):
        raise TypeError(f"graph must be a BlockGraph; received {type(graph).__name__}")
    if (plan.graph, plan.graph_sha256) != (graph.name, graph.sha256):
        raise ValueError(f"the plan is of graph {plan.graph!r} ({plan.graph_sha256[:16]}...), not of "
                         f"{graph.name!r} ({graph.sha256[:16]}...)")
    saved = set(graph.inputs)
    for planned in plan.stages:
        stage = planned.stage
        unsaved = [e for e in stage.inputs if e not in saved]
        if unsaved:
            raise ValueError(f"plan stage {stage.name!r} reads {unsaved}, which no earlier stage writes")
        if saved & set(stage.outputs):
            raise ValueError(f"plan stage {stage.name!r} writes {sorted(saved & set(stage.outputs))} again")
        saved |= set(stage.outputs)
    if set(graph.outputs) - saved:
        raise ValueError(f"the plan writes no {sorted(set(graph.outputs) - saved)}")


def _check_manifest(manifest) -> dict:
    """The identity of a ready table-asset manifest: its SHA-256 and tables."""
    if not isinstance(manifest, Mapping):
        raise TypeError(f"manifest must be a mapping (a ready weight manifest); received {type(manifest).__name__}")
    if manifest.get("format") != MANIFEST_FORMAT or manifest.get("status") != "ready":
        raise ValueError(f"the weights must be a ready {MANIFEST_FORMAT} manifest; received format "
                         f"{manifest.get('format')!r}, status {manifest.get('status')!r}")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("the manifest records no table shards")
    if manifest.get("planning_only") or manifest.get("assets_exist") is False or \
            any(not isinstance(s, Mapping) or s.get("status") != "COMPLETED" for s in shards):
        raise ValueError("every shard of the manifest must be COMPLETED: a planned manifest's tables do not exist")
    tables = sorted({s.get("asset_id") for s in shards})
    if not all(isinstance(t, str) and ASSET_ID.fullmatch(t) for t in tables) or len(tables) != len(shards):
        raise ValueError(f"the manifest's shards must name distinct explicit table asset IDs; received {tables}")
    layers = manifest.get("layers")
    if not isinstance(layers, Mapping) or any(
            not isinstance(m, Mapping) or not set(m.get("assets") or [None]) <= set(tables) for m in layers.values()):
        raise ValueError("every layer of the manifest must be read from its shards' tables")
    return {"sha256": _sha256(canonical(manifest).encode("ascii")), "tables": tables,
            "manifest_hash": manifest.get("manifest_hash"),
            "source_checkpoint_sha256": manifest.get("source_checkpoint_sha256")}


def _check_parameters(graph: BlockGraph, parameters) -> None:
    if not isinstance(parameters, Mapping):
        raise TypeError("parameters map state_dict names to ServerArrays")
    for node in graph.nodes:
        for name in node.parameters.values():
            if name not in parameters:
                reject(f"node {node.id!r}: parameter {name!r} is not bound")
            if not isinstance(parameters[name], ServerArray):
                reject(f"{name}: the run reads its weights from the manifest's tables; bind a ServerArray, "
                       f"not {type(parameters[name]).__name__}")


def _check_inputs(graph: BlockGraph, inputs) -> dict:
    if not isinstance(inputs, Mapping) or set(inputs) != set(graph.inputs):
        raise ValueError(f"inputs map each graph input {list(graph.inputs)} to an image asset ID or an ee.Image; "
                         f"received {sorted(inputs) if isinstance(inputs, Mapping) else type(inputs).__name__}")
    for edge, value in inputs.items():
        if graph.edges[edge].layout != "R":
            raise ValueError(f"graph input {edge!r} is not a raster")
        if isinstance(value, str):
            if not ASSET_ID.fullmatch(value):
                raise ValueError(f"input {edge!r}: {value!r} is not an explicit projects/PROJECT/assets/NAME ID")
        elif not isinstance(value, ee.Image):
            raise TypeError(f"input {edge!r} must be an image asset ID or an ee.Image; received {type(value).__name__}")
    return dict(inputs)


def _same_grid(a, b) -> bool:
    return canonical(grid_to_json(a)) == canonical(grid_to_json(b))


def _expect(edge) -> dict:
    """The bands and grid an export of ``edge`` saves: an R edge's bands, a W edge's ``c0`` ..,
    on the grid ``grid_view`` gives (as the request exports it)."""
    bands = list(edge.bands) if edge.layout == "R" else band_names(edge.channels)
    return {"bands": bands, "precision": "FLOAT", "grid": grid_view(edge.grid).pixel_grid()}


def prepare(directory, plan: Plan, graph: BlockGraph, *, parameters: Mapping, manifest: Mapping,
            inputs: Mapping, namespace: str, caps: Caps) -> dict:
    """Prepare a run of ``plan`` in ``directory``, a new or empty directory (see the module
    docstring). Offline: builds and saves every request, makes no call. Returns the prepared
    record (``prepared.json``)."""
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(f"{directory} is not empty: a run is prepared once, in a new directory, so that its "
                         "request IDs are fixed once")
    _check_plan(plan, graph)
    manifest_identity = _check_manifest(manifest)
    _check_parameters(graph, parameters)
    inputs = _check_inputs(graph, inputs)
    if not isinstance(namespace, str) or not ASSET_ID.fullmatch(namespace):
        raise ValueError(f"namespace must be an explicit projects/PROJECT/assets/PREFIX; received {namespace!r}")
    if not isinstance(caps, Caps):
        raise TypeError(f"caps must be a Caps; received {type(caps).__name__}")

    def name(edge: str) -> str:
        value = plan.names.get(edge)
        if not isinstance(value, str) or not NAME.fullmatch(value):
            raise ValueError(f"boundary {edge!r} is named {value!r} in the plan: an asset name is letters, digits, "
                             "'_' and '-' (plan(..., names=...))")
        return value

    # Every export, in order (the graph inputs to export, then each stage's boundaries), and the
    # asset each saved edge is read from: all checked before anything is built or written.
    order = [(None, edge, None) for edge, value in inputs.items() if not isinstance(value, str)]
    order += [(planned, edge, planned.requests.get(edge)) for planned in plan.stages for edge in planned.stage.outputs]
    asset_of = {edge: value if isinstance(value, str) else f"{namespace}_{name(edge)}"
                for edge, value in inputs.items()}
    asset_of.update({edge: f"{namespace}_{name(edge)}" for planned, edge, _ in order if planned is not None})
    destinations = [asset_of[edge] for _, edge, _ in order]
    read = set(manifest_identity["tables"]) | {value for value in inputs.values() if isinstance(value, str)}
    clashes = set(destinations) & read
    if len(set(destinations)) != len(destinations) or clashes:
        raise ValueError(f"the run's destinations must be new and distinct: {sorted(clashes) or destinations}")
    if caps.tasks < len(order):
        raise ValueError(f"caps.tasks is {caps.tasks}, fewer than the run's {len(order)} exports")
    limits, exports, texts = plan.limits, [], {}

    def add(planned, edge: str, raster: Raster, planned_bytes: int | None) -> None:
        label, described = name(edge), graph.edges[edge]
        expect = _expect(described)
        if not _same_grid(raster.grid, described.grid) or list(raster.bands) != expect["bands"]:
            raise RuntimeError(f"internal: the raster built for {edge!r} is not its edge")
        if len(expect["bands"]) > limits.bands:
            reject(f"boundary {edge!r}: {len(expect['bands']):,} bands, over the plan's limit of {limits.bands:,}",
                   code="over_limit")
        asset, description = asset_of[edge], f"{graph.name}_{label}"
        if not DESCRIPTION.fullmatch(description):
            raise ValueError(f"the task description {description!r} is not 1 to 100 letters, digits, '_' or '-'")
        request = export_request(raster.image, raster.grid, asset_id=asset, description=description,
                                 request_id=str(uuid.uuid4()))
        body = wire(request)
        if len(body) > limits.request_bytes:
            reject(f"the request exporting {edge!r} is {len(body):,} bytes, over the plan's limit of "
                   f"{limits.request_bytes:,}", code="over_limit")
        reads = request_reads(request)
        if planned is not None:
            stage_reads = sorted({asset_of[e] for e in planned.stage.inputs})
            if not set(reads["images"]) <= set(stage_reads):
                raise RuntimeError(f"internal: the request exporting {edge!r} reads images {reads['images']}, not "
                                   f"its stage's inputs {stage_reads}")
        unknown = sorted(set(reads["tables"]) - set(manifest_identity["tables"]))
        if unknown:
            reject(f"the request exporting {edge!r} reads tables {unknown}, which are not the manifest's: its "
                   "parameters are not the manifest's weights", code="foreign_tables")
        file = f"requests/{label}.json"
        texts[file] = body.decode("ascii")
        exports.append({"label": label, "stage": None if planned is None else planned.stage.name, "edge": edge,
                        "asset": asset, "description": description, "request_id": request["requestId"],
                        "request": {"file": file, "bytes": len(body), "sha256": _sha256(body),
                                    "planned_bytes": planned_bytes},
                        "reads": reads, "expect": expect})

    built = {}
    for planned, edge, planned_bytes in order:
        if planned is None:
            described = graph.edges[edge]
            raster = Raster(inputs[edge], described.bands, described.grid)
        else:
            if planned.stage.name not in built:
                built[planned.stage.name] = build_stage(graph, planned.stage, parameters,
                                                        {e: asset_of[e] for e in planned.stage.inputs})
            raster = built[planned.stage.name][edge]
        add(planned, edge, raster, planned_bytes)

    texts.update({"graph.json": graph.dumps(indent=1) + "\n", "plan.json": plan.dumps(indent=1) + "\n",
                  "manifest.json": _document(manifest)})
    directory.mkdir(parents=True, exist_ok=True)
    for file, text in texts.items():
        _write(directory / file, text)
    sha = {file: _sha256(texts[file].encode("ascii")) for file in ("graph.json", "plan.json", "manifest.json")}
    prepared = {
        "format": FORMAT, "version": VERSION,
        "graph": {"name": graph.name, "sha256": graph.sha256, "file": "graph.json", "file_sha256": sha["graph.json"]},
        "plan": {"sha256": plan.sha256, "file": "plan.json", "file_sha256": sha["plan.json"],
                 "limits": plan.limits.to_json(), "stages": [planned.stage.to_json() for planned in plan.stages]},
        "manifest": {**manifest_identity, "file": "manifest.json", "file_sha256": sha["manifest.json"]},
        "namespace": namespace,
        "inputs": {edge: {"asset": asset_of[edge], "source": "asset" if isinstance(value, str) else "export",
                          "bands": list(graph.edges[edge].bands), "grid": _expect(graph.edges[edge])["grid"]}
                   for edge, value in inputs.items()},
        "caps": caps.to_json(), "exports": exports, "versions": {"earthengine-api": ee.__version__}}
    text = _document(prepared)
    _write(directory / "prepared.json", text)
    state = {"format": FORMAT, "version": VERSION, "prepared_sha256": _sha256(text.encode("ascii")),
             "status": "PREPARED", "reason": None, "caps": caps.to_json(), "calls": {call: 0 for call in CALLS},
             "checked": {}, "exports": {export["label"]: {"state": "PREPARED"} for export in exports},
             "history": [{"unix": time.time(), "event": "prepared", "exports": len(exports)}]}
    _write(directory / "state.json", _document(state))
    return prepared


# ---------------------------------------------------------------------------
# run, resume, recover
# ---------------------------------------------------------------------------
class _Runner:
    """One process's execution of a prepared run, writing every transition to its state file."""

    def __init__(self, directory, data, caps, clock, sleep):
        self.directory = Path(directory)
        self.prepared, self.state = _load(self.directory)
        self.exports = {export["label"]: export for export in self.prepared["exports"]}
        self.data = ee.data if data is None else data
        self.data.setMaxRetries(0)                   # no client retries: a stop is the caller's to decide
        self.project = _project(self.prepared["namespace"])
        self.clock, self.sleep = clock, sleep
        self.started = clock()
        if caps is not None:
            if not isinstance(caps, Caps):
                raise TypeError(f"caps must be a Caps; received {type(caps).__name__}")
            if caps.to_json() != self.state["caps"]:
                self.log("caps", previous=self.state["caps"], caps=caps.to_json())
                self.state["caps"] = caps.to_json()
                self.save()
        self.caps = Caps.from_json(self.state["caps"])

    # -- the state file -------------------------------------------------------
    def save(self) -> None:
        _write(self.directory / "state.json", _document(self.state))

    def log(self, event: str, export: str | None = None, **detail) -> None:
        entry = {"unix": self.clock(), "event": event, **detail}
        if export is not None:
            entry["export"] = export
        self.state["history"].append(entry)

    def stop(self, reason: str, *, export: str | None = None, failed: bool = False, decision: bool = False,
             cause: BaseException | None = None):
        status = "FAILED" if failed else "STOPPED"
        self.state.update(status=status, reason=reason)
        self.log("failed" if failed else "stopped", export, reason=reason)
        self.save()
        error = (DecisionNeeded if decision else RunStopped)(reason, status=status, export=export)
        raise error from cause

    def spend(self, call: str, cap: int, why: str, export: str | None) -> None:
        """Count one ``call`` before it is made, or stop at its cap."""
        if self.state["calls"][call] >= cap:
            self.stop(f"the cap of {cap} {call} calls is reached {why}", export=export)
        self.state["calls"][call] += 1
        self.save()

    def out_of_time(self, wait: float = 0) -> bool:
        return self.clock() + wait - self.started > self.caps.seconds

    # -- calls ----------------------------------------------------------------
    def get_asset(self, asset: str, why: str, export: str | None = None) -> dict | None:
        """``getAsset``'s answer, or None when the asset does not exist (HTTP 404)."""
        self.spend("getAsset", self.caps.asset_calls, why, export)
        try:
            return self.data.getAsset(asset)
        except Exception as error:
            if _not_found(error):
                return None
            self.stop(f"getAsset({asset}) failed {why}: {type(error).__name__}: {error}", export=export, cause=error)

    def request(self, export: Mapping) -> dict:
        """The saved request body; the run stops if it is not the one prepared."""
        label, raw = export["label"], (self.directory / export["request"]["file"]).read_bytes()
        if _sha256(raw) != export["request"]["sha256"]:
            self.stop(f"export {label!r}: its request file has changed since it was prepared; it is not submitted",
                      export=label)
        body = json.loads(raw)
        destination = body.get("assetExportOptions", {}).get("earthEngineDestination", {})
        if (wire(body) != raw or body.get("requestId") != export["request_id"]
                or destination.get("name") != export["asset"] or destination.get("overwrite") is not False):
            self.stop(f"export {label!r}: its request is not the one prepared; it is not submitted", export=label)
        return body

    # -- the steps ------------------------------------------------------------
    def check_reads(self) -> None:
        """Once per run: every table of the manifest is a table; every input asset an image with
        the input's bands, each on the input's grid (CRS, dimensions and transform), as a saved
        boundary is checked."""
        wanted = [(table, None) for table in self.prepared["manifest"]["tables"]]
        wanted += [(asset, record) for record in self.prepared["inputs"].values() if record["source"] == "asset"
                   for asset in (record["asset"],)]
        for asset, record in wanted:
            if asset in self.state["checked"]:
                continue
            found = self.get_asset(asset, "checking the assets the run reads")
            if found is None:
                self.stop(f"{asset}, which the run reads, does not exist")
            if record is None and found.get("type") != "TABLE":
                self.stop(f"{asset} is a {found.get('type')!r} asset, not the weight table the manifest names")
            if record is not None:
                bands = {band.get("id"): band for band in found.get("bands") or []}
                if found.get("type") != "IMAGE" or not set(record["bands"]) <= set(bands):
                    self.stop(f"input {asset} is not an image with the bands {record['bands']} (it has "
                              f"{list(bands)[:8]})")
                problems = _grid_problems([bands[b] for b in record["bands"]], record["grid"])
                if problems:
                    self.stop(f"input {asset} is not on the input's grid: {'; '.join(problems)}")
            self.state["checked"][asset] = {"type": found.get("type"), "unix": self.clock()}
            self.save()

    def submit(self, export: Mapping, record: dict) -> None:
        label = export["label"]
        if self.out_of_time():
            self.stop(f"the time cap of {self.caps.seconds} s is reached before export {label!r} is submitted",
                      export=label)
        body = self.request(export)
        if self.data.getWorkloadTag():
            self.stop("a workload tag is set in the Earth Engine session: exportImage would add it to the prepared "
                      "request; reset it (ee.data.resetWorkloadTag(resetDefault=True))", export=label)
        if self.state["calls"]["exportImage"] >= self.caps.tasks:
            self.stop(f"the cap of {self.caps.tasks} exportImage calls is reached before export {label!r} is "
                      "submitted", export=label)
        if self.get_asset(export["asset"], f"checking export {label!r}'s destination", label) is not None:
            self.stop(f"export {label!r}: its destination {export['asset']} exists; the run never overwrites an asset",
                      export=label)
        # The call's count and the intent in one write, before the call: neither exists without the other.
        self.state["calls"]["exportImage"] += 1
        record.update(state="SUBMITTING", intent_unix=self.clock())
        self.log("submitting", label, request_id=export["request_id"])
        self.save()
        try:
            operation = self.data.exportImage(export["request_id"], body)
        except BaseException as error:
            record["error"] = f"{type(error).__name__}: {error}"
            reason = (f"export {label!r}: exportImage raised {type(error).__name__}: {error}; whether its task exists "
                      f"is unknown. Find out (request ID {export['request_id']}, description "
                      f"{export['description']!r}, destination {export['asset']}), then call recover()")
            if not isinstance(error, Exception):     # interrupted: record it, let the interruption through
                self.state.update(status="STOPPED", reason=reason)
                self.log("stopped", label, reason=reason)
                self.save()
                raise
            self.stop(reason, export=label, decision=True, cause=error)
        name = operation.get("name") if isinstance(operation, Mapping) else None
        problem = (f"the response names no operation of the run's project: {str(operation)[:200]}"
                   if _name_problem(name, self.project) else _operation_problem(operation, name, export))
        if problem:
            record["error"] = problem
            self.stop(f"export {label!r}: {problem}; whether its task exists is unknown: find out, then call recover()",
                      export=label, decision=True)
        record.update(state="SUBMITTED", operation=name, submitted_unix=self.clock(), last_operation=operation)
        self.log("submitted", label, operation=name)
        self.save()

    def poll(self, export: Mapping, record: dict) -> None:
        label, name = export["label"], record["operation"]
        while True:
            if self.out_of_time(self.caps.interval):
                self.stop(f"the time cap of {self.caps.seconds} s is reached while export {label!r} runs; its task "
                          f"{name} is left running, not cancelled: resume() polls it", export=label)
            if self.state["calls"]["getOperation"] >= self.caps.status_calls:
                self.stop(f"the cap of {self.caps.status_calls} getOperation calls is reached while export {label!r} "
                          f"runs; its task {name} is left running, not cancelled", export=label)
            self.sleep(self.caps.interval)
            self.spend("getOperation", self.caps.status_calls, f"while export {label!r} runs", label)
            try:
                operation = self.data.getOperation(name)
            except Exception as error:
                self.stop(f"getOperation({name}) failed: {type(error).__name__}: {error}; the task is not resubmitted: "
                          "resume() polls it again", export=label, cause=error)
            problem = _operation_problem(operation, name, export)
            if problem:
                self.stop(f"export {label!r}: {problem}. The operation polled is exports.{label}.operation in "
                          "state.json: check it against the task list, then resume (or correct it there by hand, "
                          "while no process holds the run)", export=label)
            metadata = operation.get("metadata") or {}
            record.update(polls=record.get("polls", 0) + 1, last_status=metadata.get("state"), last_operation=operation)
            if not operation.get("done"):
                record["state"] = "RUNNING"
                self.save()
                continue
            error = operation.get("error") or (metadata.get("state") if metadata.get("state") in ("FAILED", "CANCELLED")
                                               else None)
            if error:
                record.update(state="FAILED", error=error, finished_unix=self.clock())
                self.stop(f"export {label!r}: its task {name} failed: {error}; a failed export is never retried: "
                          "prepare a new run", export=label, failed=True)
            record.update(state="SUCCEEDED", finished_unix=self.clock())
            self.log("succeeded", label, eecu_seconds=metadata.get("batchEecuUsageSeconds"))
            self.save()
            return

    def verify(self, export: Mapping, record: dict) -> None:
        label = export["label"]
        asset = self.get_asset(export["asset"], f"checking export {label!r}'s saved asset", label)
        problems = (["its task succeeded, but the asset does not exist"] if asset is None
                    else _image_problems(asset, export["expect"]))
        if problems:
            record.update(state="FAILED", error="; ".join(problems))
            self.stop(f"export {label!r}: {export['asset']}: {'; '.join(problems)}", export=label, failed=True)
        record.update(state="VERIFIED", asset={"name": asset.get("name"), "type": asset.get("type"),
                                               "bands": len(asset.get("bands") or []),
                                               "sizeBytes": asset.get("sizeBytes"),
                                               "updateTime": asset.get("updateTime")})
        self.log("verified", label)
        self.save()

    def execute(self, event: str) -> dict:
        self.state.update(status="RUNNING", reason=None)
        self.log(event)
        self.save()
        records = self.state["exports"]
        if any(records[label]["state"] == "PREPARED" for label in self.exports):
            self.check_reads()
        for label, export in self.exports.items():
            record = records[label]
            if record["state"] == "FAILED":
                self.stop(f"export {label!r} failed: {record.get('error')}", export=label, failed=True)
            if record["state"] == "SUBMITTING":
                self.stop(f"export {label!r} was being submitted (request ID {export['request_id']}) and whether its "
                          f"task exists is unknown ({record.get('error', 'no response was recorded')}): find out, "
                          "then call recover()", export=label, decision=True)
            if record["state"] == "PREPARED":
                self.submit(export, record)
            if record["state"] in ("SUBMITTED", "RUNNING"):
                self.poll(export, record)
            if record["state"] == "SUCCEEDED":
                self.verify(export, record)
        self.state.update(status="COMPLETED", reason=None)
        self.log("completed")
        self.save()
        return self.state


def run(directory, *, data=None, caps: Caps | None = None, clock=time.time, sleep=time.sleep) -> dict:
    """Start the prepared run in ``directory`` (see the module docstring); returns the final
    state. Refuses a run that has started: ``resume`` continues it. Raises ``RunStopped`` (or
    ``DecisionNeeded``) when it stops, with the reason in the state file, and ``RunLocked`` when
    another process holds the directory."""
    with _locked(directory, "run", clock):
        runner = _Runner(directory, data, caps, clock, sleep)
        if runner.state["status"] != "PREPARED":
            raise ValueError(f"the run in {directory} has started ({runner.state['status']}): resume() continues it")
        return runner.execute("started")


def resume(directory, *, data=None, caps: Caps | None = None, clock=time.time, sleep=time.sleep) -> dict:
    """Continue the run in ``directory`` from its state file, never resubmitting an export whose
    task exists (see the module docstring); returns the final state. A completed run is returned
    as it is; a failed one is refused."""
    with _locked(directory, "resume", clock):
        runner = _Runner(directory, data, caps, clock, sleep)
        if runner.state["status"] == "COMPLETED":
            return runner.state
        if runner.state["status"] == "FAILED":
            raise RunStopped(f"the run failed and is never resumed: {runner.state['reason']}", status="FAILED")
        return runner.execute("resumed")


def recover(directory, export: str, *, operation: str | None = None, absent: bool = False, data=None,
            clock=time.time) -> dict:
    """The caller's decision on an export left ``SUBMITTING`` (its submission outcome unknown):
    ``operation``, the name of the task the caller found, which one ``getOperation`` checks
    against the export's description, type and destination before it is recorded, so that
    ``resume`` polls it; or ``absent=True``: the caller found no task, and ``resume`` submits the
    export again, with the same request ID. The operation must be of the run's project (the
    namespace's). Returns ``status``."""
    with _locked(directory, "recover", clock):
        runner = _Runner(directory, data, None, clock, None)
        record, prepared = runner.state["exports"].get(export), runner.exports.get(export)
        if record is None:
            raise ValueError(f"no export {export!r}; exports: {list(runner.exports)}")
        if record["state"] != "SUBMITTING":
            raise ValueError(f"export {export!r} is {record['state']}: only an export whose submission outcome is "
                             "unknown (SUBMITTING) is recovered")
        if (operation is None) == (not absent):
            raise ValueError("recover takes the operation found, or absent=True, not both")
        if absent:
            record.update(state="PREPARED", unresolved=record.pop("error", None))
            runner.log("recovered", export, decision="absent: no task; it is submitted again with the same request ID")
        else:
            problem = _name_problem(operation, runner.project)
            if problem:
                raise ValueError(f"operation {problem}; nothing was recorded")
            runner.spend("getOperation", runner.caps.status_calls, f"recovering export {export!r}", export)
            try:
                found = runner.data.getOperation(operation)
            except Exception as error:
                raise RunStopped(f"getOperation({operation}) failed: {type(error).__name__}: {error}; nothing was "
                                 "recorded", status=runner.state["status"], export=export) from error
            problem = _operation_problem(found, operation, prepared, destinations=True)
            if problem:
                raise ValueError(f"export {export!r}: {problem}; nothing was recorded")
            record.update(state="SUBMITTED", operation=operation, recovered=True, last_operation=found,
                          unresolved=record.pop("error", None))
            runner.log("recovered", export, decision="operation", operation=operation)
        runner.state.update(status="STOPPED", reason=f"export {export!r} recovered: resume() continues the run")
        runner.save()
    return status(directory)


def status(directory) -> dict:
    """The run's progress, from its files (no call): its status and reason, the calls made and
    their caps, each export's state, and what the caller does next."""
    directory = Path(directory)
    prepared, state = _load(directory)
    exports = []
    for export in prepared["exports"]:
        record = state["exports"][export["label"]]
        exports.append({"label": export["label"], "stage": export["stage"], "edge": export["edge"],
                        "asset": export["asset"], "state": record["state"], "operation": record.get("operation"),
                        "polls": record.get("polls", 0), "request_bytes": export["request"]["bytes"],
                        "error": record.get("error")})
    waiting = [e["label"] for e in exports if e["state"] == "SUBMITTING"]
    following = {"PREPARED": "run()", "COMPLETED": "nothing: the run is complete",
                 "FAILED": "nothing: the run failed and is never resumed; prepare a new run"}
    after = (f"recover() export {waiting[0]!r}, then resume()" if waiting
             else following.get(state["status"], "resume()"))
    return {"status": state["status"], "reason": state["reason"], "calls": dict(state["calls"]),
            "caps": dict(state["caps"]), "exports": exports, "next": after}
