"""tools/raster_cost_run.py offline: the plan, the cases' wiring and arithmetic, the run's
discipline and guards against a fake Earth Engine client, compare against a fake fetch, and the
report's reading on synthetic states. No cloud call: graphs are built against the offline
signatures with every cloud entry point refusing, and the run talks only to the fake."""
from __future__ import annotations

import copy
import datetime
import json
from pathlib import Path
import re
import shutil

import ee
from googleapiclient.errors import HttpError
import httplib2
import numpy as np
import pytest
import torch

from tensor_ee.converter.request import request_reads
from tools import raster_cost_run as RUN
from tools.fixtures import select
from tools.fixtures.core import hwc, nchw
from tools.live_fixtures import offline_earth_engine

PROJECT = "example"
ROOT_ID = "projects/example/assets/cost/run1"
LIMITS = RUN.Limits(budget_eecu_seconds=1800, task_seconds=900, queue_seconds=1800, interval_seconds=10)
ALL = "base,long,screen_size"


@pytest.fixture(scope="module")
def planned():
    return RUN.plan(ALL)


@pytest.fixture(scope="module")
def template(tmp_path_factory):
    directory = tmp_path_factory.mktemp("cost") / "prepared"
    RUN.prepare(directory, project=PROJECT, asset_root=ROOT_ID, names=ALL, limits=LIMITS)
    return directory


@pytest.fixture
def prepared(template, tmp_path):
    directory = tmp_path / "run"
    shutil.copytree(template, directory)
    return directory


# ---------------------------------------------------------------------------
# The plan and the cases
# ---------------------------------------------------------------------------
def test_the_plan_is_offline_deterministic_and_within_the_limit(planned):
    assert planned["calls_to_earth_engine"] == 0
    assert RUN.plan(ALL) == planned                       # offline_earth_engine refuses every cloud call
    assert [task["leaf"] for task in planned["tasks"]] == [
        "input_h128", "control_2", "gated_1", "gated_2", "staged_8_s0", "staged_8_s1", "staged_8_s2", "staged_8_s3",
        "gated_4", "gated_4_replicate", "control_8", "gated_8", "gated_16", "input_h64", "gated_1_h64", "gated_8_h64"]
    assert [task["case"] for task in planned["tasks"]][:13] == ["1", "2", "3", "4", "5", "5", "5", "5", "6", "7", "8",
                                                                  "9", "E1"]
    assert planned["images"] == len(planned["tasks"]) == 16
    assert all(task["request_bytes"] <= RUN.STAGE_LIMIT_BYTES for task in planned["tasks"])
    assert planned["largest_request_bytes"] == max(task["request_bytes"] for task in planned["tasks"])
    assert planned["calls"]["run"]["exportImage"] == 16 and planned["calls"]["compare"]["computePixels"] == 16
    assert [task["leaf"] for task in RUN.plan("base")["tasks"]] == [task["leaf"] for task in planned["tasks"]][:12]
    text = RUN.plan_text(planned)
    assert all(task["request_sha256"] in text for task in planned["tasks"])


def test_selections_and_the_request_limit_are_refused_before_anything_is_built(monkeypatch):
    assert RUN.selection("all") == RUN.selection("screen_size,long,base") == ("base", "long", "screen_size")
    for bad in ("long", "screen_size,long", "nope", ""):
        with pytest.raises(ValueError):
            RUN.selection(bad)
    monkeypatch.setattr(RUN, "STAGE_LIMIT_BYTES", 1_000_000)          # the inline input is larger
    with pytest.raises(ValueError, match="input_h128: the request is"):
        RUN.plan("base")


def test_every_chain_reads_only_the_input_and_every_stage_only_its_predecessor(template):
    prepared = json.loads((template / "prepared.json").read_text())
    reads = {}
    for task in prepared["plan"]["tasks"]:
        body = json.loads((template / prepared["requests"][task["leaf"]]["file"]).read_text())
        reads[task["leaf"]] = request_reads(body)
        assert reads[task["leaf"]]["tables"] == []
        assert body["assetExportOptions"]["earthEngineDestination"] == {"name": f"{ROOT_ID}/{task['leaf']}",
                                                                         "overwrite": False}
    image = {leaf: [asset.rsplit("/", 1)[1] for asset in found["images"]] for leaf, found in reads.items()}
    assert image["input_h128"] == image["input_h64"] == []
    for leaf in ("control_2", "gated_1", "gated_2", "gated_4", "gated_4_replicate", "control_8", "gated_8",
                 "gated_16", "staged_8_s0"):
        assert image[leaf] == ["input_h128"], leaf
    assert image["gated_1_h64"] == image["gated_8_h64"] == ["input_h64"]
    for k in (1, 2, 3):
        assert image[f"staged_8_s{k}"] == [f"staged_8_s{k - 1}"]


def test_the_requests_are_the_planned_requests_at_the_runs_root(template):
    prepared = json.loads((template / "prepared.json").read_text())
    at_root = RUN.plan(ALL, asset_root=ROOT_ID, limits=LIMITS)
    assert prepared["plan"] == at_root
    ids = [prepared["requests"][task["leaf"]]["request_id"] for task in at_root["tasks"]]
    assert len(set(ids)) == len(ids) and RUN.SIZING_REQUEST_ID not in ids


def test_the_four_stages_compose_to_the_whole_chain_bit_for_bit():
    stages = RUN.stage_modules("staged_8")
    whole = RUN.chain_module("staged_8")
    assert len(stages) == 4 and all(len(stage.blocks) == 2 for stage in stages)
    for k, stage in enumerate(stages):
        for j, block in enumerate(stage.blocks):
            assert block is whole.blocks[2 * k + j]
    x = nchw(RUN.input_values(RUN._case("input_h128")))
    with torch.no_grad():
        expected, staged = whole(x), x
        for stage in stages:
            staged = stage(staged)
    assert torch.equal(staged, expected)
    out = hwc(expected)
    assert np.all(np.isfinite(out)) and 0.25 < float(np.abs(out).max()) < 8.0


def test_every_case_has_its_own_seed_unused_elsewhere():
    seeds = [case.seed for case in RUN.CASES]
    assert len(set(seeds)) == len(seeds) == 13
    fixtures = {fixture.seed for fixture in select(["all"])}
    assert not set(seeds) & fixtures
    assert all(seed < 2024 or seed > 2100 for seed in seeds)       # tools/prithvi_trained_run.py: 2024 upwards
    for case in RUN.CASES:
        if case.blocks:
            weights = RUN.chain_module(case.leaf).state_dict()
            assert len(weights) == case.blocks * (8 if case.gate else 4)


def test_the_long_chain_stays_of_order_one():
    x = nchw(RUN.input_values(RUN._case("input_h128")))
    with torch.no_grad():
        out = hwc(RUN.chain_module("gated_16")(x))
    assert np.all(np.isfinite(out)) and float(np.abs(out).max()) < 16.0


# ---------------------------------------------------------------------------
# A fake Earth Engine client
# ---------------------------------------------------------------------------
class Crash(BaseException):
    """The process stops here."""


class Clock:
    def __init__(self):
        self.now = 1_800_000_000.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def _time(unix: float) -> str:
    return datetime.datetime.fromtimestamp(unix, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _missing():
    try:
        raise HttpError(httplib2.Response({"status": 404}), b'{"error": {"message": "not found"}}')
    except HttpError:
        raise ee.EEException("Asset does not exist or doesn't allow this operation.")


class FakeData:
    """``ee.data`` for the run: one operation per request ID; an operation waits ``queue``
    seconds, runs ``run`` seconds and reports ``eecu`` (per leaf, from ``behaviour``), then
    saves its asset; ``getAsset`` answers from the assets saved or raises a 404."""

    def __init__(self, directory, clock, *, behaviour=None, folder=True, extra=None):
        prepared = json.loads((Path(directory) / "prepared.json").read_text())
        self.tasks = {task["asset"]: task for task in prepared["plan"]["tasks"]}
        self.clock, self.behaviour = clock, dict(behaviour or {})
        self.assets = ({ROOT_ID: {"type": "FOLDER", "name": ROOT_ID}} if folder else {}) | dict(extra or {})
        self.operations, self.by_request, self.resubmitted, self.calls = {}, {}, [], []
        self.exported, self.cancelled, self.crash, self.workload_tag = [], [], None, ""

    def _call(self, method, phase):
        if phase == "before":
            self.calls.append(method)
        if self.crash == (len(self.calls), phase):
            raise Crash(f"{method} {phase}")

    def _shape(self, leaf):
        return {"queue": 15, "run": 30, "eecu": 5.0, **self.behaviour.get(leaf, {})}

    def exportImage(self, request_id, body):
        self._call("exportImage", "before")
        for operation in self.operations.values():           # one task at a time
            assert operation["done"], "a task was submitted while another was in flight"
        if request_id in self.by_request:
            self.resubmitted.append(request_id)
        asset = body["assetExportOptions"]["earthEngineDestination"]["name"]
        name = f"projects/{PROJECT}/operations/OP{len(self.operations):04d}"
        self.operations[name] = {"asset": asset, "description": body["description"], "created": self.clock(),
                                 "done": False, "cancelled": False, "leaf": asset.rsplit("/", 1)[1]}
        self.by_request[request_id] = name
        self.exported.append(asset.rsplit("/", 1)[1])
        self._call("exportImage", "after")
        return {"name": name, "metadata": {"state": "PENDING", "description": body["description"],
                                           "type": "EXPORT_IMAGE"}, "done": False}

    def getOperation(self, name):
        self._call("getOperation", "before")
        operation = self.operations[name]
        shape = self._shape(operation["leaf"])
        created, now = operation["created"], self.clock()
        metadata = {"state": "PENDING", "description": operation["description"], "type": "EXPORT_IMAGE",
                    "createTime": _time(created), "updateTime": _time(now),
                    "destinationUris": [f"https://code.earthengine.google.com/?asset={operation['asset']}"]}
        result = {"name": name, "metadata": metadata, "done": False}
        start = created + shape["queue"]
        if shape.get("never_start"):
            start = float("inf")
        if operation["cancelled"]:
            metadata.update(state="CANCELLED")
            if start <= now:
                metadata["startTime"] = _time(start)
            result["done"] = True
            result["error"] = {"code": 1, "message": "Cancelled."}
        elif now >= start + shape["run"]:
            metadata.update(state="SUCCEEDED", startTime=_time(start), endTime=_time(start + shape["run"]))
            if shape["eecu"] is not None:
                metadata["batchEecuUsageSeconds"] = shape["eecu"]
            result["done"] = True
            if shape.get("fail"):
                metadata["state"] = "FAILED"
                result["error"] = {"code": 3, "message": shape["fail"]}
            else:
                self.assets[operation["asset"]] = self._saved(operation["asset"])
        elif now >= start:
            metadata.update(state="RUNNING", startTime=_time(start))
        operation["done"] = result["done"]
        self._call("getOperation", "after")
        return result

    def _saved(self, asset):
        expect = self.tasks[asset]["expect"]
        return {"type": "IMAGE", "name": asset, "sizeBytes": "4096",
                "bands": [{"id": band, "dataType": {"precision": "FLOAT"}, "grid": copy.deepcopy(expect["grid"])}
                          for band in expect["bands"]]}

    def cancelOperation(self, name):
        self._call("cancelOperation", "before")
        self.operations[name]["cancelled"] = True
        self.cancelled.append(name)

    def getAsset(self, asset_id):
        self._call("getAsset", "before")
        found = copy.deepcopy(self.assets.get(asset_id))
        if found is None:
            _missing()
        return found

    def createFolder(self, path):
        self._call("createFolder", "before")
        self.assets[path] = {"type": "FOLDER", "name": path}
        return self.assets[path]

    def getWorkloadTag(self):
        return self.workload_tag

    def setMaxRetries(self, count):
        self.retries = count


def _go(directory, fake, clock, names=ALL):
    return RUN.run(directory, project=PROJECT, asset_root=ROOT_ID, names=names, limits=LIMITS, data=fake, clock=clock,
                   sleep=clock.sleep)


def _state(directory) -> dict:
    return json.loads((Path(directory) / "state.json").read_text())


def _calls(directory) -> dict:
    return json.loads((Path(directory) / "calls.json").read_text())["counts"]


def _caps_hold(directory):
    plan = json.loads((Path(directory) / "prepared.json").read_text())["plan"]
    counts = _calls(directory)
    assert all(counts[name] <= cap for name, cap in plan["calls"]["run"].items()), counts


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def test_a_run_submits_one_task_at_a_time_in_order_and_records_the_cost(prepared):
    clock = Clock()
    fake = FakeData(prepared, clock)
    state = _go(prepared, fake, clock)
    leaves = [task["leaf"] for task in json.loads((prepared / "prepared.json").read_text())["plan"]["tasks"]]
    assert state["status"] == "COMPLETED" and fake.exported == leaves
    assert fake.resubmitted == [] and len(set(fake.by_request.values())) == len(leaves) and fake.retries == 0
    assert "createFolder" not in fake.calls and state["folder"] == "existed"
    for leaf in leaves:
        record = state["tasks"][leaf]
        assert record["state"] == "VERIFIED" and record["eecu_seconds"] == 5.0
        assert record["run_seconds"] == pytest.approx(30) and record["queue_seconds"] == pytest.approx(15)
    # each destination checked absent before its export and present after
    assert fake.calls[:3] == ["getAsset", "getAsset", "exportImage"]
    assert fake.calls.count("getAsset") == 1 + 2 * len(leaves)
    _caps_hold(prepared)
    assert _calls(prepared)["exportImage"] == len(leaves)
    assert _go(prepared, fake, clock)["status"] == "COMPLETED" and fake.exported == leaves   # nothing again


def test_a_missing_folder_is_created_once_and_a_non_folder_stops_the_run(prepared, tmp_path, template):
    clock = Clock()
    fake = FakeData(prepared, clock, folder=False)
    assert _go(prepared, fake, clock, )["folder"] == "created"
    assert fake.calls.count("createFolder") == 1 and fake.calls[:2] == ["getAsset", "createFolder"]
    other = tmp_path / "other"
    shutil.copytree(template, other)
    fake = FakeData(other, clock, folder=False, extra={ROOT_ID: {"type": "IMAGE"}})
    with pytest.raises(RUN.RunStopped, match="not a folder"):
        _go(other, fake, clock)
    assert fake.exported == [] and "createFolder" not in fake.calls


def test_an_existing_destination_stops_the_run_before_submission(prepared):
    clock = Clock()
    fake = FakeData(prepared, clock, extra={f"{ROOT_ID}/gated_1": {"type": "IMAGE"}})
    with pytest.raises(RUN.RunStopped, match="exists; the run never overwrites"):
        _go(prepared, fake, clock)
    assert fake.exported == ["input_h128", "control_2"]
    assert _state(prepared)["tasks"]["gated_1"]["state"] == "PREPARED"


@pytest.mark.parametrize("leaf, behaviour, limit", [
    ("gated_2", {"run": 10_000}, "wall"),
    ("control_2", {"never_start": True}, "queue"),
])
def test_a_wall_or_queue_trip_cancels_records_and_ends_the_run(prepared, leaf, behaviour, limit):
    clock = Clock()
    fake = FakeData(prepared, clock, behaviour={leaf: behaviour})
    with pytest.raises(RUN.RunStopped, match=f"over the {limit} limit") as stopped:
        _go(prepared, fake, clock)
    assert stopped.value.status == "GUARD"
    state = _state(prepared)
    record = state["tasks"][leaf]
    assert state["status"] == "GUARD" and record["state"] == "CANCELLED" and record["guard"]["limit"] == limit
    assert record["guard"]["elapsed_seconds"] > (LIMITS.task_seconds if limit == "wall" else LIMITS.queue_seconds)
    assert fake.cancelled == [record["operation"]] and record["last_status"] == "CANCELLED"
    assert fake.exported[-1] == leaf and fake.calls[-2:] == ["cancelOperation", "getOperation"]
    _caps_hold(prepared)
    calls = list(fake.calls)
    with pytest.raises(RUN.RunStopped, match="never resumed"):
        _go(prepared, fake, clock)
    assert fake.calls == calls                       # a guard-stopped run makes no further call


def test_the_wall_limit_counts_from_the_tasks_start_not_its_queue(prepared):
    clock = Clock()
    fake = FakeData(prepared, clock, behaviour={"gated_1": {"queue": 1700, "run": 850}})
    assert _go(prepared, fake, clock)["status"] == "COMPLETED"
    assert fake.cancelled == []


def test_the_budget_stops_further_submissions(prepared):
    clock = Clock()
    fake = FakeData(prepared, clock, behaviour={leaf: {"eecu": 700.0} for leaf in ("input_h128", "control_2",
                                                                                   "gated_1", "gated_2")})
    with pytest.raises(RUN.RunStopped, match="over the budget") as stopped:
        _go(prepared, fake, clock)
    assert stopped.value.status == "GUARD"
    assert fake.exported == ["input_h128", "control_2", "gated_1"]
    assert _state(prepared)["tasks"]["gated_2"]["state"] == "PREPARED"


def test_a_finished_task_without_its_cost_ends_the_run(prepared):
    clock = Clock()
    fake = FakeData(prepared, clock, behaviour={"input_h128": {"eecu": None}})
    with pytest.raises(RUN.RunStopped, match="without batchEecuUsageSeconds"):
        _go(prepared, fake, clock)
    assert fake.exported == ["input_h128"] and _state(prepared)["status"] == "GUARD"


@pytest.mark.parametrize("seconds, removed", [(225, False), (226, True)])
def test_the_long_chain_is_removed_unless_case_9_ran_within_a_quarter_of_the_wall_limit(prepared, seconds, removed):
    clock = Clock()
    fake = FakeData(prepared, clock, behaviour={"gated_8": {"run": seconds}})
    state = _go(prepared, fake, clock)
    assert state["status"] == "COMPLETED"
    assert ("gated_16" in fake.exported) is not removed
    assert (state["tasks"]["gated_16"]["state"] == "REMOVED") is removed
    assert fake.exported[-3:] == ["input_h64", "gated_1_h64", "gated_8_h64"]


def test_a_failed_task_ends_the_run_and_is_never_retried(prepared):
    clock = Clock()
    fake = FakeData(prepared, clock, behaviour={"gated_1": {"fail": "out of memory"}})
    with pytest.raises(RUN.RunStopped, match="out of memory") as stopped:
        _go(prepared, fake, clock)
    assert stopped.value.status == "FAILED"
    with pytest.raises(RUN.RunStopped, match="never resumed"):
        _go(prepared, fake, clock)
    assert fake.exported == ["input_h128", "control_2", "gated_1"]


def test_resume_continues_an_interrupted_run_without_a_second_submission(prepared):
    clock = Clock()
    fake = FakeData(prepared, clock)
    fake.crash = (14, "before")                           # a getOperation of the second task, while it runs
    with pytest.raises(Crash):
        _go(prepared, fake, clock)
    state = _state(prepared)
    assert fake.calls[-1] == "getOperation" and state["status"] == "RUNNING"
    assert state["tasks"]["input_h128"]["state"] == "VERIFIED" and state["tasks"]["control_2"]["state"] == "RUNNING"
    assert not (prepared / "run.lock").exists()
    fake.crash = None
    state = _go(prepared, fake, clock)
    assert state["status"] == "COMPLETED" and fake.resubmitted == []
    assert len(fake.exported) == len(set(fake.exported)) == 16
    _caps_hold(prepared)


def test_an_unknown_submission_is_never_submitted_again(prepared):
    clock = Clock()
    fake = FakeData(prepared, clock)
    fake.crash = (3, "after")                             # exportImage took effect; its answer was lost
    with pytest.raises(Crash):
        _go(prepared, fake, clock)
    assert _state(prepared)["tasks"]["input_h128"]["state"] == "SUBMITTING"
    fake.crash = None
    with pytest.raises(RUN.RunStopped, match="never submitted again"):
        _go(prepared, fake, clock)
    assert fake.exported == ["input_h128"] and fake.calls.count("exportImage") == 1


def test_a_call_past_its_cap_is_not_made(prepared):
    clock = Clock()
    fake = FakeData(prepared, clock)
    plan = json.loads((prepared / "prepared.json").read_text())["plan"]
    counts = {name: 0 for name in plan["calls"]["run"]}
    counts["exportImage"] = plan["calls"]["run"]["exportImage"]
    (prepared / "calls.json").write_text(json.dumps({"caps": plan["calls"]["run"], "counts": counts}))
    with pytest.raises(RUN.RunStopped, match="cap of 16 exportImage calls"):
        _go(prepared, fake, clock)
    assert "exportImage" not in fake.calls and _state(prepared)["tasks"]["input_h128"]["state"] == "PREPARED"


def test_other_arguments_a_foreign_root_and_a_changed_request_are_refused(prepared, tmp_path):
    clock = Clock()
    fake = FakeData(prepared, clock)
    with pytest.raises(ValueError, match="other arguments"):
        RUN.run(prepared, project=PROJECT, asset_root=ROOT_ID, names="base", limits=LIMITS, data=fake, clock=clock,
                sleep=clock.sleep)
    with pytest.raises(ValueError, match="not in the project"):
        RUN.prepare(tmp_path / "x", project="another", asset_root=ROOT_ID, names="base", limits=LIMITS)
    request = prepared / "requests" / "input_h128.json"
    request.write_text(request.read_text().replace('"overwrite": false', '"overwrite": true'))
    with pytest.raises(RUN.RunStopped, match="not the one prepared"):
        _go(prepared, fake, clock)
    assert fake.exported == []


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------
class FakePixels:
    """``computePixels`` answering each saved image with the values ``values`` gives it."""

    def __init__(self, values, bands):
        self.values, self.bands, self.fetched = values, bands, []

    def computePixels(self, params):
        graph = json.dumps(ee.serializer.encode(params["expression"], for_cloud_api=True))
        leaf = re.search(re.escape(ROOT_ID) + r"/([a-z0-9_]+)", graph)[1]
        assert params["fileFormat"] == "NUMPY_NDARRAY" and all(f'"{band}"' in graph for band in self.bands[leaf])
        self.fetched.append(leaf)
        values = self.values[leaf]
        array = np.empty(values.shape[:2], dtype=[(band, np.float32) for band in self.bands[leaf]])
        for c, band in enumerate(array.dtype.names):
            array[band] = values[:, :, c]
        return array


def _expected_values():
    values = {}
    for case in RUN.CASES:
        if not case.blocks:
            values[case.leaf] = RUN.input_values(case)
            continue
        x = nchw(values[case.reads])
        with torch.no_grad():
            for k, module in enumerate(RUN.stage_modules(case.leaf)):
                x = module(x)
                values[case.leaf if case.stages == 1 else f"{case.leaf}_s{k}"] = hwc(x)
    return values


def test_compare_fetches_each_saved_image_once_and_compares_with_pytorch(prepared):
    clock = Clock()
    _go(prepared, FakeData(prepared, clock), clock)
    values = _expected_values()
    bands = {task["leaf"]: task["expect"]["bands"]
             for task in json.loads((prepared / "prepared.json").read_text())["plan"]["tasks"]}
    assert bands["input_h128"][0] == "b0" and bands["gated_1"][0] == "c0"
    values["gated_8"] = values["gated_8"] + np.float32(2e-4)
    pixels = FakePixels(values, bands)
    with offline_earth_engine():
        record = RUN.compare(prepared, project=PROJECT, data=pixels, ee_module=ee)
    assert sorted(pixels.fetched) == sorted(values) and len(pixels.fetched) == 16
    assert record["tasks"]["input_h128"]["equals_local"] and record["tasks"]["staged_8_s3"]["max_abs"] == 0
    assert record["tasks"]["gated_8"]["passed"] is False and not record["passed"]
    assert all(r["passed"] for leaf, r in record["tasks"].items() if leaf != "gated_8")
    with offline_earth_engine():
        RUN.compare(prepared, project=PROJECT, data=pixels, ee_module=ee)
    assert len(pixels.fetched) == 16                      # kept: never fetched twice
    with pytest.raises(ValueError, match="project"):
        RUN.compare(prepared, project="another", data=pixels, ee_module=ee)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _synthetic(planned, costs, *, status="COMPLETED", walled=None, reason=None):
    tasks = {}
    for task in planned["tasks"]:
        cost = costs.get(task["role"])
        tasks[task["leaf"]] = {"state": "VERIFIED", "eecu_seconds": cost} if cost is not None else {"state": "PREPARED"}
    if walled:
        tasks[walled] = {"state": "CANCELLED", "guard": {"limit": "wall"}}
    return {"status": status, "reason": reason, "tasks": tasks}


def _power(p, *, a=10.0, g=5.0, replicate=None, long=True):
    """Costs with a fixed cost a, one control block 1, and G(n) = a + g n^p."""
    costs = {"input": 2.0, "C2": a + 2, "C8": a + 8, **{f"G{n}": a + g * n ** p for n in (1, 2, 4, 8, 16)},
             **{f"S{k}": a + g * 2 ** p for k in range(4)}}
    costs["G4r"] = costs["G4"] if replicate is None else replicate
    if not long:
        del costs["G16"]
    return costs


def test_report_reads_one_graph_against_stages(planned):
    super_linear = RUN.reading(planned, _synthetic(planned, _power(1.5)))
    q = super_linear["quantities"]
    assert q["a"] == pytest.approx(10) and q["b"] == pytest.approx(1) and q["N"] == 16 and q["d"] == 0
    assert q["p"] == pytest.approx(1.5) and q["S"] == pytest.approx(4 * (10 + 5 * 2 ** 1.5)) == q["4G(2)"]
    assert q["B_length"] == 2 and q["Q"] > 2 and super_linear["decision"] == "not one graph"
    linear = RUN.reading(planned, _synthetic(planned, _power(1.0)))
    assert linear["quantities"]["Q"] < 1 and linear["decision"] == "one graph"
    noisy = RUN.reading(planned, _synthetic(planned, _power(1.5, replicate=10 + 5 * 4 ** 1.5 + 5)))
    assert 2 < noisy["quantities"]["Q"] <= 2 * (1 + 3 * noisy["quantities"]["d"])
    assert noisy["decision"] == "one graph" and "noise band" in noisy["reasons"][0]
    assert RUN.reading(planned, _synthetic(planned, _power(1.5)), threshold=3)["decision"] == "one graph"


def test_report_decides_every_guard_stop_and_says_incomplete_otherwise(planned):
    walled = RUN.reading(planned, _synthetic(planned, {"input": 2.0, "C2": 12.0}, status="GUARD", walled="gated_1"))
    assert walled["decision"] == "not one graph" and "wall limit" in walled["reasons"][0]
    budget = RUN.reading(planned, _synthetic(planned, _power(1.5, long=False), status="GUARD", reason="budget"))
    assert budget["quantities"]["N"] == 8 and budget["quantities"]["p"] == pytest.approx(1.5)
    assert budget["decision"] in ("one graph", "not one graph") and budget["reasons"][-1].startswith("the run ended")
    queued = RUN.reading(planned, _synthetic(planned, {"input": 2.0}, status="GUARD", reason="queue"))
    assert queued["decision"] == "one graph" and "C2" in queued["missing"] and "anything else" in queued["reasons"][0]
    flat = _power(1.0)
    flat.update(G4=9.0, G8=9.0, G16=9.0, G4r=9.0)                     # G(n) - a <= 0: p cannot be computed
    unreadable = RUN.reading(planned, _synthetic(planned, flat))
    assert unreadable["decision"] == "one graph" and "p" not in unreadable["quantities"]
    assert any("p is not computable" in reason for reason in unreadable["reasons"])
    for status in ("PREPARED", "RUNNING", "STOPPED"):
        assert RUN.reading(planned, _synthetic(planned, {}, status=status))["decision"] == "incomplete"
    assert RUN.reading(planned, _synthetic(planned, {}, status="FAILED"))["decision"].startswith("none")


def test_report_of_a_run_names_assets_by_leaf_only(prepared):
    clock = Clock()
    _go(prepared, FakeData(prepared, clock), clock)
    record = RUN.report(prepared)
    text = json.dumps(record)
    assert ROOT_ID not in text and "operations/" not in text and record["project"] == PROJECT
    assert record["eecu_seconds"] == 5.0 * 16 and record["reading"]["status"] == "COMPLETED"
    assert [row["leaf"] for row in record["tasks"]][0] == "input_h128"
