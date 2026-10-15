"""X06: the minimal staged runner (``tensor_ee/converter/run.py``) and the export-request builder
(``tensor_ee/converter/request.py``); docs/tasks/X06-runner.md.

- **R2, one export-request builder.** ``export_request`` gives, byte for byte and hash for
  hash, the request the SegFormer sizing tool measures (``measure_image_export``) for the four
  stages of the default plan and for the seeded input's export; ``wire`` is the API client's
  ``JsonModel.serialize``; the builder needs no raised recursion limit from its caller.
- **R1, prepare offline.** With no cloud call, every export request is built and saved with
  its bytes and SHA-256 before anything is submitted (for SegFormer-B0 with X00's parameters,
  the bytes the planner measured); a request over the plan's limit, a planned (not uploaded)
  manifest, NumPy weights, tables outside the manifest, unusable names, too few task caps
  and a used directory are refused.
- **R3, run explicitly.** Against a fake ``ee.data`` (exports, operations, assets): each
  destination is checked absent by its HTTP 404, the intent is written before the export,
  the operation is polled at the cap's interval, the saved asset's bands and grid are
  checked; caps, workload tags, failures and changed files stop the run with the reason in
  the state file; nothing is ever retried.
- **R4, resume, never resubmit.** The process is interrupted at every write of the state file
  and at every call (before and after the call takes effect), and a new process resumes:
  each export starts exactly one task, and an unknown submission stops for the caller's
  decision (``recover``).

The fast tests use the hand-written SegFormer-B0 block graph with a planner and builder that
measure and build placeholder images (no model), as X05's tests do; the SegFormer-B0 tests
(R1 and R2 with real stages) need SMP 0.5.0 and timm and skip by name without them.
Offline only: graphs are built against ``ee.apitestcase`` signatures with every cloud entry
point refusing, the runner talks only to the fake, and no model forward runs.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace
import uuid

import ee
from googleapiclient.errors import HttpError
from googleapiclient.model import JsonModel
import httplib2
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference.segformer_b0.block_graph import BOUNDARIES, SOURCE_GRID, segformer_b0_block_graph  # noqa: E402
from tensor_ee.converter import plan as PLAN, run as RUN  # noqa: E402
from tensor_ee.converter.plan import Limits, plan  # noqa: E402
from tensor_ee.converter.request import export_request, request_reads, wire  # noqa: E402
from tensor_ee.converter.run import (Caps, DecisionNeeded, RunStopped, prepare, recover, resume, run,  # noqa: E402
                                     status)
from tensor_ee.core.errors import UnsupportedOperation  # noqa: E402
from tensor_ee.core.grid import TokenGrid  # noqa: E402
from tensor_ee.core.tensor import Raster, ServerArray, band_names  # noqa: E402
from tools.live_fixtures import offline_earth_engine  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GRAPH = segformer_b0_block_graph()
NS = "projects/example-project/assets/x06_run"
TABLES = tuple(f"{NS}_weights_part_{i:04d}" for i in range(3))
# Calls are counted as attempts, so an interrupted export submitted again counts twice: room for that.
CAPS = Caps(tasks=8, status_calls=40, asset_calls=30, seconds=3600, interval=30)


def _placeholders(graph=GRAPH) -> dict:
    return {name: ServerArray(None, (1,)) for node in graph.nodes for name in node.parameters.values()}


def _manifest(tables=TABLES) -> dict:
    """A ready table-asset manifest, as ``export_weights`` leaves it."""
    return {"format": "native_ee_featurecollection_weights_v2", "status": "ready", "asset_namespace": NS + "_weights",
            "layers": {"layer": {"shape": [1, 1], "dtype": "float32", "bias": False, "assets": list(tables)}},
            "shards": [{"index": i, "asset_id": t, "status": "COMPLETED"} for i, t in enumerate(tables)]}


# ---------------------------------------------------------------------------
# A planner and builder without a model: placeholder images that read what a stage reads
# ---------------------------------------------------------------------------
def _plan_nodes(graph, stage, parameters, assets):
    nodes = graph.stage_nodes(stage.inputs, stage.outputs)
    return {edge: nodes for edge in stage.outputs}


def _fake_plan(limits=Limits(), graph=GRAPH):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(PLAN, "build_stage", _plan_nodes)
        return plan(graph, limits, parameters=_placeholders(graph), measure=lambda nodes, name: 1000 * len(nodes),
                    namespace=NS, names=BOUNDARIES)


def _placeholder_stage(tables=TABLES):
    """Stands in for ``build_stage``: each output an image of its edge's bands and grid that
    reads the stage's saved inputs and one weight table."""
    def build_stage(graph, stage, parameters, assets):
        assert set(assets) == set(stage.inputs) and all(isinstance(parameters[n], ServerArray)
                                                       for n in parameters)
        read = ee.Image.cat([ee.Image(assets[e]).select([0]) for e in stage.inputs]).reduce(ee.Reducer.sum())
        weights = ee.Image(ee.FeatureCollection(tables[len(stage.inputs) % len(tables)]).size())
        out = {}
        for edge in stage.outputs:
            described = graph.edges[edge]
            bands = tuple(described.bands) if described.layout == "R" else tuple(band_names(described.channels))
            image = ee.Image.constant([0.0] * len(bands)).rename(list(bands)).add(read).add(weights)
            out[edge] = Raster(image, bands, described.grid)
        return out
    return build_stage


def _input_image():
    return ee.Image.constant([0.25, -0.5]).rename(["b0", "b1"]).toFloat()


def _prepare(directory, the_plan, *, graph=GRAPH, inputs=None, caps=CAPS, manifest=None, parameters=None,
             builder=None, namespace=NS):
    with offline_earth_engine() as attempts, pytest.MonkeyPatch.context() as patch:
        patch.setattr(RUN, "build_stage", builder or _placeholder_stage())
        prepared = prepare(directory, the_plan, graph,
                           parameters=_placeholders(graph) if parameters is None else parameters,
                           manifest=_manifest() if manifest is None else manifest,
                           inputs={"input": _input_image()} if inputs is None else inputs, namespace=namespace,
                           caps=caps)
    assert attempts == []
    return prepared


@pytest.fixture(scope="module")
def default_plan():
    return _fake_plan()


@pytest.fixture(scope="module")
def template(tmp_path_factory, default_plan):
    directory = tmp_path_factory.mktemp("x06") / "prepared"
    _prepare(directory, default_plan)
    return directory


@pytest.fixture(scope="module")
def template5(tmp_path_factory, default_plan):
    """Prepared with the live plan's task cap: exactly one exportImage call per export."""
    directory = tmp_path_factory.mktemp("x06_tasks5") / "prepared"
    _prepare(directory, default_plan, caps=replace(CAPS, tasks=5))
    return directory


@pytest.fixture
def prepared(template, tmp_path):
    directory = tmp_path / "run"
    shutil.copytree(template, directory)
    return directory


# ---------------------------------------------------------------------------
# A fake Earth Engine client: exports, operations and assets
# ---------------------------------------------------------------------------
class Crash(BaseException):
    """The process stops here (not an Exception: nothing handles it as an error)."""


def _missing(asset_id):
    """getAsset's error for a missing asset, raised as the client raises it: while handling the
    HTTP error, whose status says 404."""
    try:
        raise HttpError(httplib2.Response({"status": 404}), b'{"error": {"message": "not found"}}')
    except HttpError:
        raise ee.EEException(f"Asset '{asset_id}' does not exist or doesn't allow this operation.")


def _saved(asset, edge, graph=GRAPH):
    """The metadata Earth Engine reports for an export of ``edge``, written from the edge (zero
    affine terms omitted, as the API omits them)."""
    described = graph.edges[edge]
    bands = list(described.bands) if described.layout == "R" else band_names(described.channels)
    grid = described.grid
    if isinstance(grid, TokenGrid):
        scale, x0, y0 = grid.scale, 0, 0
    else:
        scale, _, x0, _, _, y0 = grid.transform
    affine = {k: v for k, v in {"scaleX": scale, "shearX": 0, "translateX": x0, "shearY": 0, "scaleY": -scale,
                                "translateY": y0}.items() if v}
    band_grid = {"dimensions": {"width": grid.width, "height": grid.height}, "affineTransform": affine,
                 "crsCode": grid.crs}
    return {"type": "IMAGE", "name": asset, "sizeBytes": "4096", "updateTime": "2026-10-15T00:00:00Z",
            "bands": [{"id": b, "dataType": {"precision": "FLOAT"}, "grid": copy.deepcopy(band_grid),
                       "pyramidingPolicy": "SAMPLE"}
                      for b in bands]}


class FakeData:
    """``ee.data`` for the runner: ``exportImage`` starts an operation (at most one per request
    ID, as Earth Engine does); ``getOperation`` finishes it after ``polls`` calls and saves its
    asset; ``getAsset`` answers from the assets saved, or raises the 404 error. ``crash`` stops
    the process at the n-th call, before or after the call takes effect."""

    def __init__(self, directory, *, polls=2, tables=TABLES, extra=None, project="example-project"):
        prepared = json.loads((Path(directory) / "prepared.json").read_text())
        self.outputs = {e["asset"]: _saved(e["asset"], e["edge"]) for e in prepared["exports"]}
        self.assets = {t: {"type": "TABLE", "name": t} for t in tables} | dict(extra or {})
        self.polls, self.project = polls, project
        self.operations, self.by_request, self.calls, self.resubmitted = {}, {}, [], []
        self.fail, self.wrong, self.errors = {}, {}, {}
        self.crash = None              # (call number, "before" or "after")
        self.workload_tag, self.retries = "", None

    def _call(self, method, phase):
        if phase == "before":
            self.calls.append(method)
        if self.crash == (len(self.calls), phase):
            raise Crash(f"{method} {phase}")
        if phase == "before" and len(self.calls) in self.errors:
            raise self.errors.pop(len(self.calls))

    def exportImage(self, request_id, body):
        self._call("exportImage", "before")
        assert body["requestId"] == request_id
        if request_id in self.by_request:
            self.resubmitted.append(request_id)
            name = self.by_request[request_id]
        else:
            name = f"projects/{self.project}/operations/OP{len(self.operations):04d}"
            asset = body["assetExportOptions"]["earthEngineDestination"]["name"]
            self.operations[name] = {"asset": asset, "description": body["description"], "polls": 0}
            self.by_request[request_id] = name
        self._call("exportImage", "after")
        return {"name": name, "metadata": {"state": "PENDING", "description": body["description"],
                                           "type": "EXPORT_IMAGE"}, "done": False}

    def getOperation(self, name):
        self._call("getOperation", "before")
        operation = self.operations[name]
        operation["polls"] += 1
        metadata = {"state": "RUNNING", "description": operation["description"], "type": "EXPORT_IMAGE",
                    "destinationUris": [f"https://code.earthengine.google.com/?asset={operation['asset']}"]}
        result = {"name": name, "metadata": metadata, "done": False}
        if operation["polls"] >= self.polls:
            result["done"] = True
            if operation["asset"] in self.fail:
                metadata["state"] = "FAILED"
                result["error"] = {"code": 3, "message": self.fail[operation["asset"]]}
            else:
                metadata.update(state="SUCCEEDED", batchEecuUsageSeconds=1.5)
                self.assets[operation["asset"]] = self.wrong.get(operation["asset"], self.outputs[operation["asset"]])
        self._call("getOperation", "after")
        return result

    def getAsset(self, asset_id):
        self._call("getAsset", "before")
        found = copy.deepcopy(self.assets.get(asset_id))
        self._call("getAsset", "after")
        if found is None:
            _missing(asset_id)
        return found

    def getWorkloadTag(self):
        return self.workload_tag

    def setMaxRetries(self, count):
        self.retries = count


class Clock:
    def __init__(self):
        self.now, self.slept = 1_800_000_000.0, []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


def _go(function, directory, fake, clock, **kwargs):
    return function(directory, data=fake, clock=clock, sleep=clock.sleep, **kwargs)


def _state(directory) -> dict:
    return json.loads((Path(directory) / "state.json").read_text())


def _finish(directory, fake, clock) -> list[str]:
    """Resume until the run completes, deciding as a careful caller would: an unknown
    submission is recovered with its task's operation when one exists, else as absent.
    Returns the decisions made."""
    decisions, caps = [], None
    for _ in range(20):
        assert not (Path(directory) / RUN.LOCK).exists()          # released by every stop and crash
        try:
            _go(resume, directory, fake, clock, caps=caps)
            return decisions
        except DecisionNeeded as stop:
            request_id = next(e["request_id"] for e in json.loads((Path(directory) / "prepared.json").read_text())
                              ["exports"] if e["label"] == stop.export)
            if request_id in fake.by_request:
                recover(directory, stop.export, operation=fake.by_request[request_id], data=fake, clock=clock)
                decisions.append(f"{stop.export}: operation")
            else:
                recover(directory, stop.export, absent=True, data=fake, clock=clock)
                decisions.append(f"{stop.export}: absent")
        except RunStopped as stop:
            # A resubmission (after "absent") is past a task cap with no room: the user raises it.
            assert "exportImage calls is reached" in str(stop), stop
            recorded = Caps.from_json(_state(directory)["caps"])
            caps = replace(recorded, tasks=recorded.tasks + 1)
            decisions.append(f"{stop.export}: tasks raised")
    raise AssertionError("the run did not complete")


def _one_task_each(directory, fake):
    prepared = json.loads((Path(directory) / "prepared.json").read_text())
    assert sorted(fake.by_request) == sorted(e["request_id"] for e in prepared["exports"])
    assert len(set(fake.by_request.values())) == len(prepared["exports"]) and fake.resubmitted == []
    assert set(fake.outputs) <= set(fake.assets)


# ---------------------------------------------------------------------------
# R2: one export-request builder
# ---------------------------------------------------------------------------
def test_the_builder_writes_the_export_configuration_and_the_request_id_last():
    with offline_earth_engine() as attempts:
        image = _input_image()
        request = export_request(image, SOURCE_GRID, asset_id=f"{NS}_input", description="segformer_b0_input",
                                 request_id="00000000-0000-4000-8000-000000000000")
    assert attempts == []
    assert list(request) == ["expression", "description", "assetExportOptions", "maxPixels", "requestId"]
    assert request["assetExportOptions"] == {"earthEngineDestination": {"name": f"{NS}_input", "overwrite": False},
                                             "pyramidingPolicy": "SAMPLE"}
    assert request["maxPixels"] == {"value": "50176"} and request["description"] == "segformer_b0_input"
    assert wire(request) == JsonModel().serialize(request).encode("utf-8")
    assert request_reads(request) == {"images": [], "tables": []}
    for bad in ({"asset_id": "users/someone/x"}, {"asset_id": f"{NS}.input"}, {"description": "a b"},
                {"description": "x" * 101}, {"request_id": "not-a-uuid"},
                {"request_id": "00000000-0000-4000-8000-00000000000A"}):
        arguments = {"asset_id": f"{NS}_input", "description": "d",
                     "request_id": "00000000-0000-4000-8000-000000000000"}
        with offline_earth_engine(), pytest.raises(ValueError):
            export_request(_input_image(), SOURCE_GRID, **(arguments | bad))


def test_request_reads_lists_the_assets_an_expression_loads():
    with offline_earth_engine():
        image = ee.Image(f"{NS}_C1").select([0]).add(ee.Image(f"{NS}_C1").select([1]))
        image = image.add(ee.Image(ee.FeatureCollection(TABLES[1]).size())).add(ee.ImageCollection(f"{NS}_ic").mosaic())
        request = export_request(image, TokenGrid(56, 56, 10.0), asset_id=f"{NS}_out", description="out",
                                 request_id="00000000-0000-4000-8000-000000000000")
    assert request_reads(request) == {"images": [f"{NS}_C1", f"{NS}_ic"], "tables": [TABLES[1]]}


# ---------------------------------------------------------------------------
# R1: prepare, offline
# ---------------------------------------------------------------------------
def test_prepare_saves_every_request_and_the_state_before_anything_is_submitted(template, default_plan):
    prepared = json.loads((template / "prepared.json").read_text())
    state = _state(template)
    assert prepared["format"] == "native_ee.run" and prepared["version"] == 1
    assert prepared["graph"]["sha256"] == GRAPH.sha256 and prepared["plan"]["sha256"] == default_plan.sha256
    assert [(e["label"], e["stage"]) for e in prepared["exports"]] == [
        ("input", None), ("C1", "stage1"), ("C2", "stage2"), ("C3", "stage3"), ("logits", "stage4")]
    source_grid = {"crsCode": "EPSG:3857", "dimensions": {"width": 224, "height": 224},
                   "affineTransform": {"scaleX": 2.5, "shearX": 0, "translateX": 1000, "shearY": 0, "scaleY": -2.5,
                                       "translateY": 2000}}
    assert prepared["inputs"] == {"input": {"asset": f"{NS}_input", "source": "export", "bands": ["b0", "b1"],
                                            "grid": source_grid}}
    for export in prepared["exports"]:
        raw = (template / export["request"]["file"]).read_bytes()
        body = json.loads(raw)
        assert export["request"]["bytes"] == len(raw) and export["request"]["sha256"] == hashlib.sha256(raw).hexdigest()
        assert wire(body) == raw and body["requestId"] == export["request_id"]
        assert body["assetExportOptions"]["earthEngineDestination"] == {"name": export["asset"], "overwrite": False}
        assert export["asset"] == f"{NS}_{export['label']}"
        assert export["description"] == f"segformer_b0_{export['label']}"
    reads = {e["label"]: e["reads"]["images"] for e in prepared["exports"]}
    assert reads == {"input": [], "C1": [f"{NS}_input"], "C2": [f"{NS}_C1"], "C3": [f"{NS}_C2"],
                     "logits": [f"{NS}_C1", f"{NS}_C2", f"{NS}_C3"]}
    expect = {e["label"]: e["expect"] for e in prepared["exports"]}
    assert expect["C2"]["bands"] == band_names(64) and expect["C2"]["grid"]["dimensions"] == {"width": 28, "height": 28}
    assert expect["logits"]["grid"]["affineTransform"]["translateX"] == 1000
    assert len({e["request_id"] for e in prepared["exports"]}) == 5
    assert state["status"] == "PREPARED" and state["calls"] == {"exportImage": 0, "getOperation": 0, "getAsset": 0}
    assert all(record == {"state": "PREPARED"} for record in state["exports"].values())
    assert status(template)["next"] == "run()"
    assert (template / "graph.json").read_text() == GRAPH.dumps(indent=1) + "\n"


def test_prepare_with_an_input_asset_exports_no_input(tmp_path, default_plan):
    source = "projects/example-project/assets/chips/chip_0001"
    prepared = _prepare(tmp_path / "run", default_plan, inputs={"input": source})
    assert [e["label"] for e in prepared["exports"]] == ["C1", "C2", "C3", "logits"]
    assert prepared["exports"][0]["reads"]["images"] == [source]
    assert prepared["inputs"]["input"] == {"asset": source, "source": "asset", "bands": ["b0", "b1"],
                                           "grid": prepared["exports"][-1]["expect"]["grid"]}   # the source grid


def test_prepare_refuses_what_it_cannot_run(tmp_path, default_plan):
    def refused(error, match, **kwargs):
        directory = tmp_path / f"case{len(list(tmp_path.iterdir()))}"
        with pytest.raises(error, match=match):
            _prepare(directory, kwargs.pop("the_plan", default_plan), **kwargs)
        assert not (directory / "state.json").exists() and not (directory / "prepared.json").exists()

    planned = _manifest() | {"planning_only": True}
    planned["shards"] = [dict(s, status="planned_not_submitted") for s in planned["shards"]]
    refused(ValueError, "COMPLETED", manifest=planned)
    refused(ValueError, "ready", manifest=_manifest() | {"status": "uploading"})
    import numpy as np
    weights = _placeholders() | {"encoder.norm1.weight": np.zeros(32, np.float32)}
    refused(UnsupportedOperation, "bind a ServerArray", parameters=weights)
    unbound = _placeholders()
    del unbound["encoder.norm1.weight"]
    refused(UnsupportedOperation, "is not bound", parameters=unbound)
    refused(UnsupportedOperation, "not the manifest's",
            builder=_placeholder_stage(("projects/example-project/assets/other",)))
    refused(UnsupportedOperation, "over the plan's limit",
            the_plan=replace(default_plan, limits=Limits(request_bytes=1000)))
    refused(ValueError, "fewer than the run's 5 exports", caps=replace(CAPS, tasks=4))
    refused(ValueError, "inputs map each graph input", inputs={"chip": "projects/p/assets/x"})
    refused(ValueError, "explicit projects", inputs={"input": "users/someone/chip"})
    refused(ValueError, "namespace", namespace="projects/example-project/assets/x06.run")
    refused(ValueError, "new and distinct", inputs={"input": f"{NS}_C1"})
    dotted = replace(default_plan, names={**default_plan.names, "encoder.norm2": "encoder.norm2"})
    refused(ValueError, "an asset name is letters", the_plan=dotted)
    other = segformer_b0_block_graph(replace(SOURCE_GRID, transform=(2.5, 0, 0, 0, -2.5, 0)))
    refused(ValueError, "the plan is of graph", graph=other)
    used = tmp_path / "used"
    used.mkdir()
    (used / "notes.txt").write_text("x")
    with pytest.raises(ValueError, match="not empty"):
        _prepare(used, default_plan)


def test_caps_are_checked():
    assert Caps(1, 1, 1, 1.0, 10) == Caps.from_json(Caps(1, 1, 1, 1.0, 10).to_json())
    for bad in ({"tasks": 0}, {"status_calls": 1.5}, {"asset_calls": True}, {"seconds": 0}, {"interval": 5},
                {"interval": 601}, {"seconds": float("inf")}):
        with pytest.raises(ValueError):
            replace(CAPS, **bad)


# ---------------------------------------------------------------------------
# R3: run, explicitly
# ---------------------------------------------------------------------------
def test_a_run_submits_polls_and_checks_each_export_in_order(prepared):
    fake, clock = FakeData(prepared), Clock()
    state = _go(run, prepared, fake, clock)
    assert state["status"] == "COMPLETED" and fake.retries == 0
    assert all(r["state"] == "VERIFIED" for r in state["exports"].values())
    _one_task_each(prepared, fake)
    tables, per_export = ["getAsset"] * 3, ["getAsset", "exportImage", "getOperation", "getOperation", "getAsset"]
    assert fake.calls == tables + per_export * 5
    assert state["calls"] == {"exportImage": 5, "getOperation": 10, "getAsset": 13}
    assert clock.slept == [30] * 10
    order = [(h["event"], h.get("export")) for h in state["history"] if h["event"] in ("submitting", "verified")]
    assert order == [(event, label) for label in ("input", "C1", "C2", "C3", "logits")
                     for event in ("submitting", "verified")]
    assert state["exports"]["C1"]["asset"]["bands"] == 32 and set(state["checked"]) == set(TABLES)
    assert status(prepared)["next"] == "nothing: the run is complete"
    assert _go(resume, prepared, fake, clock)["status"] == "COMPLETED" and len(fake.calls) == 28
    with pytest.raises(ValueError, match="has started"):
        _go(run, prepared, fake, clock)


def test_the_intent_is_written_before_the_export_is_called(prepared):
    fake, clock = FakeData(prepared), Clock()
    seen = []
    original = fake.exportImage

    def export(request_id, body):
        record = _state(prepared)["exports"][body["description"].removeprefix("segformer_b0_")]
        seen.append((record["state"], _state(prepared)["calls"]["exportImage"]))
        return original(request_id, body)

    fake.exportImage = export
    _go(run, prepared, fake, clock)
    assert seen == [("SUBMITTING", k) for k in range(1, 6)]


def test_a_multi_export_stage_and_an_input_asset(tmp_path):
    the_plan = _fake_plan(Limits(request_bytes=9_000, extent=None))
    assert the_plan.stages[0].stage.outputs == ("encoder.norm1", "encoder.norm2")
    source = "projects/example-project/assets/chips/chip_0001"
    directory = tmp_path / "run"
    prepared = _prepare(directory, the_plan, inputs={"input": source})
    assert [(e["label"], e["stage"]) for e in prepared["exports"]] == [
        ("C1", "stage1"), ("C2", "stage1"), ("C3", "stage2"), ("logits", "stage3")]
    fake, clock = FakeData(directory, extra={source: _saved(source, "input")}), Clock()
    assert _go(run, directory, fake, clock)["status"] == "COMPLETED"
    assert fake.calls[:4] == ["getAsset"] * 4
    _one_task_each(directory, fake)


@pytest.mark.parametrize("change, match", [
    (lambda m: m["bands"].pop(1), re.escape("not an image with the bands ['b0', 'b1']")),
    (lambda m: m["bands"][1]["grid"].update(crsCode="EPSG:32633"), "band b1 is in 'EPSG:32633'"),
    (lambda m: m["bands"][0]["grid"]["dimensions"].update(height=448), re.escape("is {'width': 224, 'height': 448}")),
    (lambda m: m["bands"][0]["grid"]["affineTransform"].update(translateY=2560.0), "band b0 has the transform"),
])
def test_an_input_asset_not_on_the_input_grid_stops_the_run(tmp_path, default_plan, change, match):
    source = "projects/example-project/assets/chips/chip_0001"
    directory = tmp_path / "run"
    _prepare(directory, default_plan, inputs={"input": source})
    wrong = _saved(source, "input")
    wrong["bands"].append(dict(copy.deepcopy(wrong["bands"][0]), id="extra"))     # another band is allowed
    change(wrong)
    fake = FakeData(directory, extra={source: wrong})
    with pytest.raises(RunStopped, match=match):
        _go(run, directory, fake, Clock())
    assert "exportImage" not in fake.calls and source not in _state(directory)["checked"]


def test_a_destination_that_exists_stops_before_submitting(prepared):
    fake, clock = FakeData(prepared, extra={f"{NS}_C2": {"type": "IMAGE"}}), Clock()
    with pytest.raises(RunStopped, match="destination .*_C2 exists") as stop:
        _go(run, prepared, fake, clock)
    assert stop.value.status == "STOPPED" and stop.value.export == "C2"
    state = _state(prepared)
    assert state["status"] == "STOPPED" and "exists" in state["reason"]
    assert state["exports"]["C2"]["state"] == "PREPARED"
    assert fake.calls.count("exportImage") == 2
    del fake.assets[f"{NS}_C2"]                       # the caller removes it, then resumes
    assert _go(resume, prepared, fake, clock)["status"] == "COMPLETED"
    _one_task_each(prepared, fake)


def test_a_failed_task_ends_the_run_and_is_never_retried(prepared):
    fake, clock = FakeData(prepared), Clock()
    fake.fail[f"{NS}_C3"] = "User memory limit exceeded."
    with pytest.raises(RunStopped, match="memory limit") as stop:
        _go(run, prepared, fake, clock)
    assert stop.value.status == "FAILED"
    state = _state(prepared)
    assert state["status"] == "FAILED" and state["exports"]["C3"]["state"] == "FAILED"
    assert state["exports"]["logits"]["state"] == "PREPARED"
    calls = len(fake.calls)
    with pytest.raises(RunStopped, match="never resumed"):
        _go(resume, prepared, fake, clock)
    assert len(fake.calls) == calls and fake.resubmitted == []
    assert status(prepared)["next"].startswith("nothing: the run failed")


@pytest.mark.parametrize("change, match", [
    (lambda m: m["bands"].pop(), "bands are"),
    (lambda m: m["bands"][3]["dataType"].update(precision="DOUBLE"), "not FLOAT"),
    (lambda m: m["bands"][0]["grid"].update(crsCode="EPSG:4326"), "EPSG:4326"),
    (lambda m: m["bands"][0]["grid"]["dimensions"].update(width=27), re.escape("is {'width': 27")),
    (lambda m: m["bands"][0]["grid"]["affineTransform"].update(translateX=5.0), "has the transform"),
    (lambda m: m.update(type="TABLE"), "not an image"),
])
def test_a_saved_asset_that_is_not_the_export_fails_the_run(prepared, change, match):
    fake, clock = FakeData(prepared), Clock()
    wrong = copy.deepcopy(fake.outputs[f"{NS}_C2"])
    change(wrong)
    fake.wrong[f"{NS}_C2"] = wrong
    with pytest.raises(RunStopped, match=match) as stop:
        _go(run, prepared, fake, clock)
    assert stop.value.status == "FAILED" and stop.value.export == "C2"
    assert _state(prepared)["exports"]["C2"]["state"] == "FAILED" and fake.calls.count("exportImage") == 3


def test_caps_stop_the_run_and_leave_the_task_running(prepared):
    fake, clock = FakeData(prepared, polls=4), Clock()
    with pytest.raises(RunStopped, match="time cap of 100 s is reached while export 'input' runs") as stop:
        _go(run, prepared, fake, clock, caps=replace(CAPS, seconds=100))
    assert stop.value.status == "STOPPED"
    assert fake.calls.count("getOperation") == 3 and _state(prepared)["exports"]["input"]["state"] == "RUNNING"
    with pytest.raises(RunStopped, match="cap of 6 getOperation calls"):
        _go(resume, prepared, fake, clock, caps=replace(CAPS, status_calls=6))
    assert _state(prepared)["calls"]["getOperation"] == 6
    with pytest.raises(RunStopped, match="cap of 6 getOperation calls"):    # the recorded caps hold
        _go(resume, prepared, fake, clock)
    state = _go(resume, prepared, fake, clock, caps=CAPS)
    assert state["status"] == "COMPLETED" and state["caps"] == CAPS.to_json()
    assert [h["caps"]["status_calls"] for h in state["history"] if h["event"] == "caps"] == [40, 6, 40]
    _one_task_each(prepared, fake)


def test_the_task_and_asset_caps_stop_before_the_call(prepared):
    fake, clock = FakeData(prepared), Clock()
    caps = replace(CAPS, asset_calls=4)
    with pytest.raises(RunStopped, match="cap of 4 getAsset calls"):
        _go(run, prepared, fake, clock, caps=caps)
    assert fake.calls.count("getAsset") == 4 and fake.calls.count("exportImage") == 1
    with pytest.raises(RunStopped, match="cap of 1 exportImage calls"):
        _go(resume, prepared, fake, clock, caps=replace(CAPS, tasks=1))
    assert fake.calls.count("exportImage") == 1


def test_a_workload_tag_or_a_changed_request_stops_before_any_export(prepared):
    fake, clock = FakeData(prepared), Clock()
    fake.workload_tag = "x06"
    with pytest.raises(RunStopped, match="workload tag"):
        _go(run, prepared, fake, clock)
    assert "exportImage" not in fake.calls
    fake.workload_tag = ""
    path = prepared / "requests" / "input.json"
    path.write_bytes(path.read_bytes().replace(b"segformer_b0_input", b"segformer_b0_INPUT"))
    with pytest.raises(RunStopped, match="request file has changed"):
        _go(resume, prepared, fake, clock)
    assert "exportImage" not in fake.calls and "request file has changed" in _state(prepared)["reason"]


def test_a_state_of_another_preparation_is_refused(prepared, template):
    text = (prepared / "prepared.json").read_text()
    (prepared / "prepared.json").write_text(text.replace('"tasks": 8', '"tasks": 9'))
    with pytest.raises(ValueError, match="another preparation"):
        status(prepared)
    with pytest.raises(ValueError, match="another preparation"):
        _go(run, prepared, FakeData(template), Clock())


def test_a_missing_table_stops_the_run_before_any_export(prepared):
    fake = FakeData(prepared, tables=TABLES[:2])
    with pytest.raises(RunStopped, match="part_0002, which the run reads, does not exist"):
        _go(run, prepared, fake, Clock())
    assert "exportImage" not in fake.calls


def test_a_status_error_stops_and_resume_polls_the_same_task(prepared):
    fake, clock = FakeData(prepared), Clock()
    fake.errors[6] = ee.EEException("The service is currently unavailable.")
    with pytest.raises(RunStopped, match="getOperation.* failed: .*unavailable") as stop:
        _go(run, prepared, fake, clock)
    assert stop.value.export == "input" and _state(prepared)["exports"]["input"]["state"] == "SUBMITTED"
    assert _go(resume, prepared, fake, clock)["status"] == "COMPLETED"
    _one_task_each(prepared, fake)


def test_a_non_404_asset_error_stops_without_submitting(prepared):
    fake, clock = FakeData(prepared), Clock()
    fake.errors[4] = ee.EEException("Asset 'x' does not exist or doesn't allow this operation.")   # no 404 status
    with pytest.raises(RunStopped, match="getAsset.* failed"):
        _go(run, prepared, fake, clock)
    assert "exportImage" not in fake.calls and _state(prepared)["exports"]["input"]["state"] == "PREPARED"


# ---------------------------------------------------------------------------
# R4: resume, never resubmit
# ---------------------------------------------------------------------------
def test_an_export_call_that_raises_stops_for_a_decision(prepared):
    fake, clock = FakeData(prepared), Clock()
    fake.crash = None
    fake.errors[5] = ee.EEException("Deadline exceeded.")          # the first exportImage, before it takes effect
    with pytest.raises(DecisionNeeded, match="whether its task exists is unknown") as stop:
        _go(run, prepared, fake, clock)
    assert stop.value.export == "input" and status(prepared)["next"] == "recover() export 'input', then resume()"
    record = _state(prepared)["exports"]["input"]
    assert record["state"] == "SUBMITTING" and "Deadline exceeded" in record["error"]
    with pytest.raises(DecisionNeeded):                               # resume alone still asks
        _go(resume, prepared, fake, clock)
    with pytest.raises(ValueError, match="not both"):
        recover(prepared, "input", data=fake, clock=clock)
    with pytest.raises(ValueError, match="only an export whose submission outcome is unknown"):
        recover(prepared, "C1", absent=True, data=fake, clock=clock)
    recover(prepared, "input", absent=True, data=fake, clock=clock)
    assert _go(resume, prepared, fake, clock)["status"] == "COMPLETED"
    _one_task_each(prepared, fake)
    assert fake.calls.count("exportImage") == 6


def test_recover_checks_the_operation_it_is_given(prepared):
    fake, clock = FakeData(prepared), Clock()
    fake.crash = (5, "after")                                          # the task starts; its response is lost
    with pytest.raises(Crash):
        _go(run, prepared, fake, clock)
    fake.crash = None
    other = "projects/example-project/operations/OTHER"
    fake.operations[other] = {"asset": f"{NS}_elsewhere", "description": "someone_else", "polls": 0}
    with pytest.raises(ValueError, match="is the task 'someone_else'"):
        recover(prepared, "input", operation=other, data=fake, clock=clock)
    for name in ("OP0000", "projects/other-project/operations/OP0000"):        # pinned to the run's project
        with pytest.raises(ValueError, match="not an operation of the run's project, "
                                             "projects/example-project/operations/ID"):
            recover(prepared, "input", operation=name, data=fake, clock=clock)
    assert _state(prepared)["exports"]["input"]["state"] == "SUBMITTING"
    recover(prepared, "input", operation=fake.by_request[next(iter(fake.by_request))], data=fake, clock=clock)
    assert _state(prepared)["exports"]["input"]["state"] == "SUBMITTED"
    assert _go(resume, prepared, fake, clock)["status"] == "COMPLETED"
    _one_task_each(prepared, fake)
    assert fake.calls.count("exportImage") == 5


def _writes(prepared_dir) -> int:
    """The state file's writes in an uninterrupted run."""
    count = [0]
    original = RUN._write

    def counting(path, text):
        count[0] += 1
        original(path, text)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(RUN, "_write", counting)
        _go(run, prepared_dir, FakeData(prepared_dir), Clock())
    return count[0]


def test_an_interruption_at_every_write_resumes_with_one_task_per_export(template5, tmp_path):
    """With the live plan's task cap, ``Caps(tasks=5)``: every point completes with no cap raised,
    since an export's count and intent are one write."""
    reference = tmp_path / "reference"
    shutil.copytree(template5, reference)
    writes = _writes(reference)
    assert writes == 53
    decided = set()
    for k in range(1, writes + 1):
        directory = tmp_path / f"write{k}"
        shutil.copytree(template5, directory)
        fake, clock, count = FakeData(directory), Clock(), [0]
        original = RUN._write

        def crashing(path, text):
            count[0] += 1
            if count[0] == k:
                raise Crash(f"write {k}")
            original(path, text)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(RUN, "_write", crashing)
            with pytest.raises(Crash):
                _go(run, directory, fake, clock)
        before = _state(directory)
        decisions = _finish(directory, fake, clock)
        _one_task_each(directory, fake)
        assert fake.calls.count("exportImage") == 5 and _state(directory)["calls"]["exportImage"] == 5
        # A decision is asked for exactly when the crash left an export SUBMITTING, and its task exists.
        submitting = [label for label, r in before["exports"].items() if r["state"] == "SUBMITTING"]
        assert decisions == [f"{label}: operation" for label in submitting]
        decided |= set(decisions)
        shutil.rmtree(directory)
    assert decided == {f"{label}: operation" for label in ("input", "C1", "C2", "C3", "logits")}


@pytest.mark.parametrize("phase", ["before", "after"])
def test_an_interruption_at_every_call_resumes_with_one_task_per_export(template5, tmp_path, phase):
    """With ``Caps(tasks=5)``: only an ``exportImage`` stopped before it took effect needs a second
    call, which the user decides twice: ``absent``, then one more task allowed for the last export."""
    reference = tmp_path / "reference"
    shutil.copytree(template5, reference)
    probe = FakeData(reference)
    _go(run, reference, probe, Clock())
    assert len(probe.calls) == 28
    labels = ("input", "C1", "C2", "C3", "logits")
    for k in range(1, len(probe.calls) + 1):
        directory = tmp_path / f"call{k}"
        shutil.copytree(template5, directory)
        fake, clock = FakeData(directory), Clock()
        fake.crash = (k, phase)
        with pytest.raises(Crash):
            _go(run, directory, fake, clock)
        fake.crash = None
        method = probe.calls[k - 1]
        decisions = _finish(directory, fake, clock)
        _one_task_each(directory, fake)
        if method != "exportImage":
            assert decisions == [] and fake.calls.count("exportImage") == 5
        elif phase == "after":
            label = labels[probe.calls[:k].count("exportImage") - 1]
            assert decisions == [f"{label}: operation"] and fake.calls.count("exportImage") == 5
        else:
            # Its task never started: submitted once more, with the same request ID.
            label = labels[probe.calls[:k].count("exportImage") - 1]
            assert decisions == [f"{label}: absent", "logits: tasks raised"]
            assert fake.calls.count("exportImage") == 6
        shutil.rmtree(directory)


def test_an_operation_of_another_project_is_not_recorded(prepared):
    fake, clock = FakeData(prepared, project="other-project"), Clock()
    with pytest.raises(DecisionNeeded, match="names no operation of the run's project"):
        _go(run, prepared, fake, clock)
    assert _state(prepared)["exports"]["input"]["state"] == "SUBMITTING"
    fake.retries = None
    with pytest.raises(ValueError, match="not an operation of the run's project"):
        recover(prepared, "input", operation=next(iter(fake.by_request.values())), data=fake, clock=clock)
    assert fake.retries == 0 and _state(prepared)["exports"]["input"]["state"] == "SUBMITTING"


# ---------------------------------------------------------------------------
# One process per run directory (fix round, P2-1)
# ---------------------------------------------------------------------------
def test_a_lock_left_by_a_dead_process_is_refused_until_removed_by_hand(prepared):
    lock = prepared / RUN.LOCK
    lock.write_text(json.dumps({"pid": 424242, "started_unix": 1.0, "call": "run", "token": "x"}))
    before = (prepared / "state.json").read_text()
    fake, clock = FakeData(prepared), Clock()
    for call in (lambda: _go(run, prepared, fake, clock), lambda: _go(resume, prepared, fake, clock),
                 lambda: recover(prepared, "input", absent=True, data=fake, clock=clock)):
        with pytest.raises(RUN.RunLocked, match=r"held by process 424242 \(run, .* remove .*run\.lock by hand"):
            call()
    assert fake.calls == [] and fake.retries is None and lock.exists()
    assert (prepared / "state.json").read_text() == before
    assert status(prepared)["status"] == "PREPARED"                   # reading takes no lock
    lock.unlink()                                                      # the caller, after checking
    assert _go(run, prepared, fake, clock)["status"] == "COMPLETED" and not lock.exists()


def test_the_lock_is_released_on_every_stop(prepared):
    fake, clock = FakeData(prepared, extra={f"{NS}_C2": {"type": "IMAGE"}}), Clock()
    lock = prepared / RUN.LOCK
    with pytest.raises(RunStopped, match="exists"):
        _go(run, prepared, fake, clock)
    assert not lock.exists()
    del fake.assets[f"{NS}_C2"]
    fake.errors[len(fake.calls) + 2] = ee.EEException("Deadline exceeded.")      # C2's exportImage
    with pytest.raises(DecisionNeeded):
        _go(resume, prepared, fake, clock)
    assert not lock.exists()
    recover(prepared, "C2", absent=True, data=fake, clock=clock)
    assert not lock.exists()
    fake.fail[f"{NS}_C3"] = "User memory limit exceeded."
    with pytest.raises(RunStopped, match="memory limit"):
        _go(resume, prepared, fake, clock)
    assert not lock.exists()
    with pytest.raises(RunStopped, match="never resumed"):
        _go(resume, prepared, fake, clock)
    assert not lock.exists()


def test_a_second_process_cannot_write_while_the_first_runs(prepared):
    """The review's scenario E: a second process resuming the directory while the first runs
    would overwrite the state file with stale counts. The lock refuses it."""
    fake, clock = FakeData(prepared), Clock()
    refused, export_image = [], fake.exportImage

    def export(request_id, body):
        before = (prepared / "state.json").read_text()
        with pytest.raises(RUN.RunLocked, match=f"held by process {os.getpid()} \\(run"):
            _go(resume, prepared, FakeData(prepared), Clock())
        assert (prepared / "state.json").read_text() == before
        refused.append(request_id)
        return export_image(request_id, body)

    fake.exportImage = export
    assert _go(run, prepared, fake, clock)["status"] == "COMPLETED" and len(refused) == 5
    assert _state(prepared)["calls"] == {"exportImage": 5, "getOperation": 10, "getAsset": 13}
    assert not (prepared / RUN.LOCK).exists()


# ---------------------------------------------------------------------------
# CallGuard: the calls of the steps outside the runner (fix round, P2-2)
# ---------------------------------------------------------------------------
def _client():
    made = []
    return SimpleNamespace(
        made=made, newTaskId=lambda count=1: ["local"],
        exportTable=lambda request_id, body: made.append(("exportTable", request_id)) or {"name": request_id},
        getAsset=lambda asset_id: made.append(("getAsset", asset_id)) or {"type": "TABLE"})


def test_a_call_guard_counts_before_each_call_and_refuses_past_its_cap(tmp_path):
    client = _client()
    export_table, get_asset = client.exportTable, client.getAsset
    path = tmp_path / "weights-calls.json"
    with RUN.CallGuard({"exportTable": 2, "getAsset": 0}, data=client, path=path) as guard:
        assert client.exportTable("a", {}) == {"name": "a"} and client.exportTable("b", {}) == {"name": "b"}
        assert json.loads(path.read_text())["counts"] == {"exportTable": 2, "getAsset": 0}
        with pytest.raises(RUN.CallCapReached, match="cap of 2 exportTable calls is reached; the call was not made"):
            client.exportTable("c", {})
        with pytest.raises(RUN.CallCapReached, match="cap of 0 getAsset"):
            client.getAsset("t")
        assert client.newTaskId() == ["local"]                             # not named: not counted
    assert guard.report() == {"caps": {"exportTable": 2, "getAsset": 0}, "counts": {"exportTable": 2, "getAsset": 0}}
    assert client.made == [("exportTable", "a"), ("exportTable", "b")]
    assert client.exportTable is export_table and client.getAsset is get_asset
    # The counts add up across processes through the file; the functions are restored after an error.
    with pytest.raises(RUN.CallCapReached):
        with RUN.CallGuard({"exportTable": 3, "getAsset": 1}, data=client, path=path):
            client.getAsset("t")
            client.exportTable("c", {})
            client.exportTable("d", {})
    assert json.loads(path.read_text())["counts"] == {"exportTable": 3, "getAsset": 1}
    assert client.exportTable is export_table and client.made[-1] == ("exportTable", "c")
    for caps, match in (({"exportTabel": 1}, "not a function of the client"), ({"exportTable": -1}, "non-negative"),
                        ({}, "one or more"), ({"getAsset": 1}, "do not name")):
        with pytest.raises(ValueError, match=match):
            RUN.CallGuard(caps, data=client, path=path)
    assert issubclass(RUN.CallCapReached, BaseException) and not issubclass(RUN.CallCapReached, Exception)


def test_a_call_guard_stops_export_weights_at_a_cap_and_its_resume_completes(tmp_path):
    """The live plan's weight upload: a cap reached inside ``export_weights`` starts no task and
    leaves its shard "starting", which ``resume=True`` checks with one ``getAsset`` and uploads."""
    import numpy as np
    from tensor_ee.weight_assets import ConvolutionWeights, export_weights
    rng = np.random.default_rng(6)
    records = [ConvolutionWeights(f"layer{i}", rng.uniform(-1, 1, (1, 4000)).astype(np.float32), None, "float32",
                                  f"layer{i}.weight") for i in range(3)]
    manifest_path, counts = tmp_path / "manifest.json", tmp_path / "weights-calls.json"
    operations, saved = {}, set()

    def export_table(request_id, body):
        name = f"projects/example-project/operations/T{len(operations)}"
        operations[name] = body["assetExportOptions"]["earthEngineDestination"]["name"]
        return {"name": name}

    def get_operation(name):
        saved.add(operations[name])
        return {"name": name, "done": True, "metadata": {"state": "SUCCEEDED", "type": "EXPORT_FEATURES"}}

    def get_asset(asset_id):
        if asset_id not in saved:
            _missing(asset_id)
        return {"type": "TABLE", "name": asset_id}

    caps = {"exportTable": 2, "getOperation": 10, "getTaskStatus": 10, "getAsset": 3}
    with offline_earth_engine(), pytest.MonkeyPatch.context() as patch:
        for name, function in (("exportTable", export_table), ("getOperation", get_operation),
                               ("getAsset", get_asset), ("getTaskStatus", lambda ids: []),
                               ("newTaskId", lambda count=1: [str(uuid.uuid4())])):
            patch.setattr(ee.data, name, function)
        with pytest.raises(RUN.CallCapReached, match="cap of 2 exportTable calls"):
            with RUN.CallGuard(caps, path=counts):
                export_weights(records, f"{NS}_weights", manifest_path, max_bytes=100_000, poll_seconds=1)
        manifest = json.loads(manifest_path.read_text())
        assert manifest["status"] == "uploading"
        assert [s["status"] for s in manifest["shards"]] == ["COMPLETED", "COMPLETED", "starting"]
        with RUN.CallGuard(dict(caps, exportTable=3), path=counts) as guard:
            manifest = export_weights(records, f"{NS}_weights", manifest_path, max_bytes=100_000, poll_seconds=1,
                                      resume=True)
    assert manifest["status"] == "ready" and len(operations) == 3
    assert guard.report()["counts"] == {"exportTable": 3, "getOperation": 3, "getTaskStatus": 0, "getAsset": 1}


def test_status_and_imports_make_no_call(prepared):
    code = (
        "import ee\n"
        "from unittest import mock\n"
        "def refuse(*a, **k):\n"
        "    raise SystemExit('cloud call')\n"
        "for name in ('exportImage', 'getOperation', 'getAsset', 'getTaskStatus', 'computePixels',\n"
        "             'computeValue', 'getInfo', 'listAssets', 'setMaxRetries'):\n"
        "    if hasattr(ee.data, name):\n"
        "        setattr(ee.data, name, refuse)\n"
        "ee.Initialize = refuse\n"
        "from tensor_ee.converter import run, request\n"
        "run.Caps(1, 1, 1, 1.0)\n"
        f"print(run.status({str(prepared)!r})['status'])\n")
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=300,
                            env={"PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": "2", "PATH": os.environ.get("PATH", ""),
                                 "HOME": os.environ.get("HOME", "")})
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "PREPARED"


# ---------------------------------------------------------------------------
# SegFormer-B0: the real stages (R1 and R2 against the sizing tool)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def segformer(tmp_path_factory):
    """X00's parameters (the seeded state's weights as ServerArrays of tables planned under the
    sizing tool's placeholder namespace), the default plan measured with the tool, and a run
    prepared with the real builder under that namespace, the seeded input exported first."""
    from tools import graph_identity as GI
    reason = GI.ext_unavailable("segformer_b0", ("segmentation_models_pytorch", "timm"))
    if reason:
        pytest.skip(f"SegFormer-B0 not built: {reason}")
    from tensor_ee._swin_parameters import records_and_locations
    from tensor_ee.models.segformer_b0.configuration import COUNTER
    from tools import segformer_b0_request_plan as P
    from tools.fixtures.segformer_b0_fullmodel import (SEED, reject_negative_zero, representative_boundaries,
                                                       seeded_state, state_sha256)
    ns = P.PLACEHOLDER_NAMESPACE
    with offline_earth_engine() as attempts:
        state = seeded_state(SEED)
        reject_negative_zero(state)
        digest = state_sha256(state)
        records = list(records_and_locations({}, {}, {k: v for k, v in state.items() if k != COUNTER})[0].values())
        shards = P.plan_weight_shards(records, ns + "_weights", P.WEIGHT_WIRE_LIMIT)
        planned = P.planned_manifest(records, shards, ns + "_weights", digest)
        server = P.asset_state(state, planned, digest)
        the_plan = plan(GRAPH, Limits(), parameters=server, namespace=ns, names=BOUNDARIES,
                        measure=lambda raster, name: P.measure_image_export(raster.image, raster.grid, ns,
                                                                            name)["wire"]["bytes"])
        image = P._boundary_image("input", "inline", representative_boundaries(SEED), ns)
        # The manifest after its upload: the same tables, every shard COMPLETED.
        ready = copy.deepcopy(planned)
        ready.update(planning_only=False, assets_exist=True)
        for shard in ready["shards"]:
            shard["status"] = "COMPLETED"
        directory = tmp_path_factory.mktemp("x06_segformer") / "run"
        prepared = prepare(directory, the_plan, GRAPH, parameters=server, manifest=ready, inputs={"input": image},
                           namespace=ns, caps=Caps(tasks=5, status_calls=40, asset_calls=30, seconds=3600))
    assert attempts == []
    return {"P": P, "ns": ns, "server": server, "plan": the_plan, "image": image, "directory": directory,
            "prepared": prepared, "tables": [s["asset_id"] for s in ready["shards"]]}


def test_segformer_the_builder_reproduces_the_tools_requests(segformer):
    P, ns, server, the_plan = segformer["P"], segformer["ns"], segformer["server"], segformer["plan"]
    from tensor_ee.converter.build import build_stage
    assert [(s.stage.inputs, s.stage.outputs) for s in the_plan.stages] == [
        (("input",), ("encoder.norm1",)), (("encoder.norm1",), ("encoder.norm2",)),
        (("encoder.norm2",), ("encoder.norm3",)), (("encoder.norm1", "encoder.norm2", "encoder.norm3"), ("logits",))]
    measured = {}
    with offline_earth_engine() as attempts:
        rasters = {"input": Raster(segformer["image"], ("b0", "b1"), SOURCE_GRID)}
        for planned in the_plan.stages:
            assets = {e: f"{ns}_{BOUNDARIES[e]}" for e in planned.stage.inputs}
            rasters.update(build_stage(GRAPH, planned.stage, server, assets))
        for edge in ("input", "encoder.norm1", "encoder.norm2", "encoder.norm3", "logits"):
            name, raster = BOUNDARIES[edge], rasters[edge]
            tool = P.measure_image_export(raster.image, raster.grid, ns, name)["wire"]
            request = export_request(raster.image, raster.grid, asset_id=f"{ns}_{name}",
                                     description=f"segformer_b0_{name}", request_id=P.REQUEST_ID)
            body = wire(request)
            assert body == JsonModel().serialize(request).encode("utf-8")
            assert (len(body), hashlib.sha256(body).hexdigest()) == (tool["bytes"], tool["sha256"]), name
            measured[name] = len(body)
    assert attempts == []
    assert measured == {"input": 1_222_596, "C1": 2_062_548, "C2": 1_166_058, "C3": 874_385, "logits": 464_261}
    assert [s.requests for s in the_plan.stages] == [{"encoder.norm1": 2_062_548}, {"encoder.norm2": 1_166_058},
                                                     {"encoder.norm3": 874_385}, {"logits": 464_261}]


def test_segformer_prepare_records_the_planned_requests(segformer, tmp_path):
    prepared, ns = segformer["prepared"], segformer["ns"]
    exports = {e["label"]: e for e in prepared["exports"]}
    assert list(exports) == ["input", "C1", "C2", "C3", "logits"]
    for label, export in exports.items():
        assert export["request"]["planned_bytes"] in (None, export["request"]["bytes"])
        expected = {"input": 1_222_596, "C1": 2_062_548, "C2": 1_166_058, "C3": 874_385, "logits": 464_261}[label]
        assert export["request"]["bytes"] == expected                   # a fresh UUID is as long as the tool's
    assert exports["input"]["reads"] == {"images": [], "tables": []}
    # Each stage reads the tables that hold its parameters, all of them the manifest's.
    reads = {label: [t.removeprefix(f"{ns}_weights_part_") for t in exports[label]["reads"]["tables"]]
             for label in ("C1", "C2", "C3", "logits")}
    assert reads == {"C1": ["0000", "0001"], "C2": ["0000", "0001"], "C3": ["0000", "0001", "0002", "0003", "0004"],
                     "logits": ["0000", "0004", "0005", "0006", "0007", "0008"]}
    assert exports["logits"]["reads"]["images"] == [f"{ns}_C1", f"{ns}_C2", f"{ns}_C3"]
    assert exports["C3"]["expect"]["bands"] == band_names(160)
    assert prepared["manifest"]["tables"] == sorted(segformer["tables"]) and len(segformer["tables"]) == 9
    directory = tmp_path / "run"
    shutil.copytree(segformer["directory"], directory)
    fake = FakeData(directory, tables=segformer["tables"])
    assert _go(run, directory, fake, Clock())["status"] == "COMPLETED"
    _one_task_each(directory, fake)


def test_segformer_the_builder_needs_no_raised_recursion_limit(segformer):
    P, ns = segformer["P"], segformer["ns"]
    from tensor_ee.converter.build import build_stage
    stage = segformer["plan"].stages[0].stage
    with offline_earth_engine():
        (raster,) = build_stage(GRAPH, stage, segformer["server"], {"input": f"{ns}_input"}).values()
        previous = sys.getrecursionlimit()
        sys.setrecursionlimit(1000)
        try:
            request = export_request(raster.image, raster.grid, asset_id=f"{ns}_C1", description="segformer_b0_C1",
                                     request_id=P.REQUEST_ID)
            assert sys.getrecursionlimit() == 1000
        finally:
            sys.setrecursionlimit(previous)
    assert len(wire(request)) == 2_062_548
