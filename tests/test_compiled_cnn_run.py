"""tools/compiled_cnn_run.py and tools/fixtures/cnn_reference.py offline: the deterministic weights
and their calibration, the exact table plan, the upload through the library's own uploader, each
outcome of the model's run, compare and the report, against fake Earth Engine clients. No cloud
call: graphs are built against the offline signatures with every cloud entry point refusing. The
tool's logic runs on a small local CNN recipe; the SMP recipe's tests skip without SMP and run no
forward of the model."""
from __future__ import annotations

import contextlib
import copy
import datetime
import importlib.util
import json
import os
from pathlib import Path
import shutil
from unittest import mock

import ee
from googleapiclient.errors import HttpError
import httplib2
import numpy as np
import pytest
import torch
from torch import nn

from tools import compiled_cnn_run as RUN
from tools.fixtures import cnn_reference as REF
from tools.live_fixtures import offline_earth_engine

PROJECT = "example"
ROOT_ID = "projects/example/assets/cnn/run1"
PARENT = "projects/example/assets/cnn"
TINY = "tiny_cnn_2band_16"
SMP = "smp_unet_efficientnet_b3_8band_256"
HAS_SMP = importlib.util.find_spec("segmentation_models_pytorch") is not None
LIMITS = RUN.Limits(budget_eecu_seconds=600, task_seconds=900, queue_seconds=1800, interval_seconds=10)


def _tiny():
    return nn.Sequential(nn.Conv2d(2, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
                         nn.Conv2d(16, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 3, 1)).eval()


@pytest.fixture(scope="module", autouse=True)
def tiny_recipe():
    RUN.RECIPES[TINY] = RUN.Recipe(TINY, "a small local CNN for the tests", _tiny, 2, 16,
                                   (("3.", "kaiming_relu"), ("6.", "xavier")), None, 100_000)
    yield
    del RUN.RECIPES[TINY]
    RUN.reference_model.cache_clear()
    RUN.compiled_module.cache_clear()


@pytest.fixture(scope="module")
def planned(tiny_recipe):
    return RUN.plan(TINY)


@pytest.fixture(scope="module")
def template(tiny_recipe, tmp_path_factory):
    directory = tmp_path_factory.mktemp("cnn") / "prepared"
    RUN.prepare(directory, recipe=TINY, project=PROJECT, asset_root=ROOT_ID)
    return directory


@pytest.fixture
def prepared(template, tmp_path):
    directory = tmp_path / "run"
    shutil.copytree(template, directory)
    return directory


# ---------------------------------------------------------------------------
# The deterministic weights
# ---------------------------------------------------------------------------
def test_the_weights_follow_their_native_scales_and_are_reproducible():
    first, second = _tiny(), _tiny()
    rules = REF.initialize(first, (("3.", "kaiming_relu"), ("6.", "xavier")))
    REF.initialize(second, (("3.", "kaiming_relu"), ("6.", "xavier")))
    for (name, a), (_, b) in zip(first.named_parameters(), second.named_parameters()):
        assert torch.equal(a, b), name
    assert rules["0.weight"]["kind"] == rules["0.bias"]["kind"] == "torch_default"
    assert rules["0.weight"]["std"] == pytest.approx((1 / (3 * 18)) ** .5)
    assert rules["3.weight"]["std"] == pytest.approx((2 / 144) ** .5) and rules["3.bias"]["constant"] == 0.0
    assert rules["6.weight"]["std"] == pytest.approx((2 / (64 + 3)) ** .5)
    assert rules["1.weight"]["constant"] == 1.0 and rules["1.bias"]["constant"] == 0.0
    weight = first[3].weight.detach().numpy()
    assert abs(weight.std() - rules["3.weight"]["std"]) < 0.1 * rules["3.weight"]["std"]
    with pytest.raises(ValueError, match="no declared native initializer"):
        REF.initialize(nn.Sequential(nn.Linear(2, 2)))


def test_the_calibration_is_fixed_changes_only_running_statistics_and_keeps_dropout_off():
    def model():
        module = nn.Sequential(nn.Conv2d(2, 4, 3, padding=1), nn.BatchNorm2d(4), nn.Dropout(0.5), nn.ReLU()).eval()
        REF.initialize(module)
        return module

    first, second = model(), model()
    before = copy.deepcopy(dict(first.named_parameters()))
    REF.calibrate(first, 2, 16)
    REF.calibrate(second, 2, 16)
    assert not first.training and all(not module.training for module in first.modules())
    assert all(torch.equal(before[name], value) for name, value in first.named_parameters())
    assert torch.equal(first[1].running_mean, second[1].running_mean)          # drop-out stayed off
    assert torch.equal(first[1].running_var, second[1].running_var)
    assert first[1].num_batches_tracked.item() == REF.CALIBRATION_BATCHES and first[1].momentum == 0.1
    images = np.concatenate([REF.calibration_batch(k, 2, 16) for k in range(REF.CALIBRATION_BATCHES)])
    x = torch.from_numpy(images).permute(0, 3, 1, 2)
    with torch.no_grad():
        conv = first[0](x)
    np.testing.assert_allclose(first[1].running_mean.numpy(), conv.mean((0, 2, 3)).numpy(), atol=1e-6)
    chip = REF.evaluation_input(2, 16)
    assert chip.dtype == np.float32 and not any(np.array_equal(chip, batch) for batch in images)


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------
def test_the_plan_is_exact_deterministic_and_capped(planned):
    from tensor_ee.weight_assets import upload_chunks

    assert planned["calls_to_earth_engine"] == 0
    with offline_earth_engine():
        chunks = list(upload_chunks(RUN.parameters(TINY), 100_000))
    shards = planned["tables"]["shards"]
    assert planned["tables"]["count"] == len(chunks) == len(shards) >= 2
    assert [s["collection_payload_bytes"] for s in shards] == [count for _, count in chunks]
    assert [s["rows"] for s in shards] == [len(records) for records, _ in chunks]
    assert [s["asset_id"] for s in shards] == [f"{RUN.PLAN_ROOT}/weights_part_{k:04d}" for k in range(len(chunks))]
    # determinism across two builds from scratch
    RUN.reference_model.cache_clear()
    RUN.compiled_module.cache_clear()
    assert RUN.plan(TINY) == planned
    assert planned["model"]["clip_footprints"] is True and planned["model"]["counts"]["clip"] > 1
    assert all(planned[part]["bytes"] <= RUN.STAGE_LIMIT_BYTES for part in ("input", "model"))
    calls = planned["calls"]
    assert calls["upload"]["exportTable"] == len(shards) + RUN.CALL_MARGIN and calls["upload"]["exportImage"] == 1
    assert calls["upload"]["computeValue"] == planned["tables"]["layers"] + RUN.CALL_MARGIN
    assert calls["run"]["exportImage"] == 1 and calls["run"]["cancelOperation"] == 1
    assert calls["compare"] == {"computePixels": 2 + RUN.CALL_MARGIN}
    text = RUN.plan_text(planned)
    assert all(s["asset_id"].rsplit("/", 1)[1] in text for s in shards)
    assert planned["model"]["sha256"] in text and planned["input"]["sha256"] in text
    assert "every validated block record names False" in text and "setDeadline" in text
    assert all(name in text for name in RUN.OUTCOMES)
    assert planned["ceiling"]["model_task"] == pytest.approx(3.3 * (900 + 10 + 60))


def test_the_requests_read_only_the_run_root(template):
    from tensor_ee.converter.request import request_reads

    prepared = json.loads((template / "prepared.json").read_text())
    shards = json.loads((template / "planned_manifest.json").read_text())["shards"]
    model = json.loads((template / prepared["requests"]["model"]["file"]).read_text())
    data = json.loads((template / prepared["requests"]["input"]["file"]).read_text())
    assert request_reads(data) == {"images": [], "tables": []}
    assert request_reads(model) == {"images": [f"{ROOT_ID}/input"],
                                    "tables": sorted(shard["asset_id"] for shard in shards)}


# ---------------------------------------------------------------------------
# Fake Earth Engine clients
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


def _time(unix):
    return datetime.datetime.fromtimestamp(unix, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _missing():
    try:
        raise HttpError(httplib2.Response({"status": 404}), b'{"error": {"message": "not found"}}')
    except HttpError:
        raise ee.EEException("Asset does not exist or doesn't allow this operation.")


class FakeEarthEngine:
    """The calls of ``upload`` (on ``ee.data`` itself, as the uploader uses it), ``run`` and
    ``compare``. ``tables``: per table index, "ok", "fail" or "crash" (the process stops at the
    first status read); ``model``: "ok", "complex", "error", "lost", "fail", "wall", "queue", the
    logits' behaviour; ``exports``: the same per export leaf of a staged run; ``poll_errors``:
    per export leaf, the numbers of the status reads (from 1) that raise."""

    def __init__(self, directory, clock, *, tables=None, model="ok", folders=(PARENT, ROOT_ID), extra=None,
                 exports=None, poll_errors=None):
        prepared = json.loads((Path(directory) / "prepared.json").read_text())
        self.plan = prepared["plan"]
        self.clock, self.tables, self.model = clock, dict(tables or {}), model
        self.behaviours = {"logits": model, **dict(exports or {})}
        self.poll_errors = dict(poll_errors or {})
        self.assets = {level: {"type": "FOLDER"} for level in folders} | dict(extra or {})
        self.calls, self.operations, self.exported = [], {}, []
        self.crashed = False

    # -- ee.data functions ----------------------------------------------------------
    def getAsset(self, asset_id):
        self.calls.append("getAsset")
        found = copy.deepcopy(self.assets.get(asset_id))
        if found is None:
            _missing()
        return found

    def createFolder(self, path):
        self.calls.append("createFolder")
        self.assets[path] = {"type": "FOLDER"}

    def exportTable(self, request_id, config):
        self.calls.append("exportTable")
        asset = config["assetExportOptions"]["earthEngineDestination"]["name"]
        index = int(asset.rsplit("_", 1)[1])
        name = f"projects/{PROJECT}/operations/T{index}"
        self.operations[name] = {"kind": "table", "asset": asset, "index": index, "reads": 0,
                                 "description": config["description"]}
        self.exported.append(asset.rsplit("/", 1)[1])
        return {"name": name, "metadata": {"state": "PENDING"}, "done": False}

    def exportImage(self, request_id, body):
        self.calls.append("exportImage")
        asset = body["assetExportOptions"]["earthEngineDestination"]["name"]
        leaf = asset.rsplit("/", 1)[1]
        self.exported.append(leaf)
        behaviour = self.behaviours.get(leaf, "ok")
        if behaviour == "complex":
            raise ee.EEException("Computation is too complex.")
        if behaviour == "error":
            raise ee.EEException("Collection.loadTable: Collection asset not found.")
        if behaviour == "lost":
            raise Crash("the call did not return")
        name = f"projects/{PROJECT}/operations/I{len(self.operations)}"
        self.operations[name] = {"kind": leaf, "asset": asset, "created": self.clock(), "reads": 0,
                                 "description": body["description"], "cancelled": False}
        return {"name": name, "metadata": {"state": "PENDING", "description": body["description"],
                                           "type": "EXPORT_IMAGE"}, "done": False}

    def _part(self, asset):
        if asset.endswith("/input"):
            return self.plan["input"]
        if self.plan.get("mode") == "staged":
            return next(e for e in self.plan["exports"] if e["asset"] == asset)
        return self.plan["model"]

    def _saved(self, asset):
        part = self._part(asset)
        return {"type": "IMAGE", "name": asset, "sizeBytes": "4096",
                "bands": [{"id": band, "dataType": {"precision": "FLOAT"}, "grid": copy.deepcopy(part["expect"]["grid"])}
                          for band in part["expect"]["bands"]]}

    def getOperation(self, name):
        self.calls.append("getOperation")
        operation = self.operations[name]
        operation["reads"] += 1
        if operation["kind"] == "table":
            behaviour = self.tables.get(operation["index"], "ok")
            if behaviour == "crash" and not self.crashed:
                self.crashed = True
                raise Crash("the process stops while polling a table")
            metadata = {"state": "SUCCEEDED", "batchEecuUsageSeconds": 0.25, "description": operation["description"]}
            if behaviour == "fail":
                metadata["state"] = "FAILED"
                return {"name": name, "metadata": metadata, "done": True, "error": {"message": "table export failed"}}
            self.assets[operation["asset"]] = {"type": "TABLE"}
            return {"name": name, "metadata": metadata, "done": True}
        if operation["reads"] in self.poll_errors.get(operation["kind"], ()):
            raise ee.EEException(f"Deadline exceeded on status read {operation['reads']}")
        created, now = operation["created"], self.clock()
        metadata = {"state": "PENDING", "description": operation["description"], "type": "EXPORT_IMAGE",
                    "createTime": _time(created), "updateTime": _time(now),
                    "destinationUris": [f"https://code.earthengine.google.com/?asset={operation['asset']}"]}
        result = {"name": name, "metadata": metadata, "done": False}
        behaviour = self.behaviours.get(operation["kind"], "ok")
        if behaviour == "crash" and not self.crashed and operation["reads"] == 1:
            self.crashed = True
            raise Crash("the process stops while polling an export")
        start = created + 15 if behaviour != "queue" else float("inf")
        run_seconds = 10_000 if behaviour == "wall" else 30
        if operation["cancelled"]:
            metadata.update(state="CANCELLED", batchEecuUsageSeconds=3.0)
            if start <= now:
                metadata["startTime"] = _time(start)
            result.update(done=True, error={"code": 1, "message": "Cancelled."})
        elif now >= start + run_seconds:
            metadata.update(state="SUCCEEDED", startTime=_time(start), endTime=_time(start + run_seconds),
                            batchEecuUsageSeconds=12.5 if operation["kind"] == "logits" else 0.5)
            result["done"] = True
            if behaviour == "fail":
                metadata["state"] = "FAILED"
                result["error"] = {"code": 3, "message": "User memory limit exceeded."}
            else:
                self.assets[operation["asset"]] = self._saved(operation["asset"])
        elif now >= start:
            metadata.update(state="RUNNING", startTime=_time(start))
        return result

    def getTaskStatus(self, ids):
        self.calls.append("getTaskStatus")
        out = []
        for task_id in ids:
            operation = self.operations[f"projects/{PROJECT}/operations/{task_id}"]
            self.assets[operation["asset"]] = {"type": "TABLE"}
            out.append({"id": task_id, "state": "COMPLETED", "batch_eecu_usage_seconds": 0.25})
        return out

    def cancelOperation(self, name):
        self.calls.append("cancelOperation")
        self.operations[name]["cancelled"] = True

    def computePixels(self, params):
        self.calls.append("computePixels")
        graph = json.dumps(ee.serializer.encode(params["expression"], for_cloud_api=True))
        values = self.values["input" if "/input" in graph else "logits"]
        final = self.plan["exports"][-1] if self.plan.get("mode") == "staged" else self.plan["model"]
        names = self.plan["input"]["expect"]["bands"] if "/input" in graph else final["expect"]["bands"]
        array = np.empty(values.shape[:2], dtype=[(band, np.float32) for band in names])
        for c, band in enumerate(names):
            array[band] = values[:, :, c]
        return array

    def getWorkloadTag(self):
        return ""

    def newTaskId(self, count=1):
        return [f"00000000-0000-4000-8000-{len(self.operations) * 10 + k:012d}" for k in range(count)]

    def setMaxRetries(self, count):
        self.retries = count

    def setDeadline(self, milliseconds):
        self.deadline = milliseconds


UPLOAD_CALLS = ("getAsset", "createFolder", "exportTable", "getOperation", "getTaskStatus", "exportImage")


@contextlib.contextmanager
def _on_ee_data(fake):
    """``fake``'s functions on ``ee.data`` itself (the uploader calls ``ee.data`` and ``ee.batch``),
    inside the offline signatures; restored in order on exit."""
    with offline_earth_engine(), contextlib.ExitStack() as stack:
        for name in UPLOAD_CALLS + ("getWorkloadTag", "newTaskId"):
            stack.enter_context(mock.patch.object(ee.data, name, getattr(fake, name)))
        yield ee.data


def _passed(params, manifest, path, **kwargs):
    report = {"status": "pass", "layers": [{"key": p.key, "status": "pass"} for p in params]}
    Path(path).write_text(json.dumps(report))
    return report


def _failed(params, manifest, path, **kwargs):
    return {"status": "fail"}


def _upload(directory, fake, clock, verify=_passed, recipe=TINY):
    with _on_ee_data(fake) as data:
        return RUN.upload(directory, recipe=recipe, project=PROJECT, asset_root=ROOT_ID, data=data, clock=clock,
                          sleep=clock.sleep, verify_asset_rows=verify)


def _run(directory, fake, clock, limits=LIMITS):
    return RUN.run(directory, project=PROJECT, limits=limits, data=fake, clock=clock, sleep=clock.sleep)


def _state(directory):
    return json.loads((Path(directory) / "state.json").read_text())


def _caps_hold(directory, command):
    plan = json.loads((Path(directory) / "prepared.json").read_text())["plan"]
    counts = json.loads((Path(directory) / f"calls_{command}.json").read_text())["counts"]
    assert all(counts[name] <= cap for name, cap in plan["calls"][command].items()), counts


# ---------------------------------------------------------------------------
# upload
# ---------------------------------------------------------------------------
def test_an_upload_completes_verifies_and_matches_the_plan(prepared):
    clock = Clock()
    fake = FakeEarthEngine(prepared, clock)
    state = _upload(prepared, fake, clock)
    plan = json.loads((prepared / "prepared.json").read_text())["plan"]
    tables = plan["tables"]["count"]
    assert state["upload"]["state"] == "VERIFIED" and state["upload"]["input"]["state"] == "VERIFIED"
    assert fake.exported == [f"weights_part_{k:04d}" for k in range(tables)] + ["input"]
    assert fake.calls[:2] == ["getAsset", "getAsset"] and fake.calls.count("exportTable") == tables
    assert state["upload"]["eecu_seconds"] == pytest.approx(0.25 * tables + 0.5)
    manifest = json.loads((prepared / "weights_manifest.json").read_text())
    assert manifest["status"] == "ready" and all(shard["task_id"] for shard in manifest["shards"])
    # the planned model request is the one built from the uploaded manifest
    with offline_earth_engine():
        body, _ = RUN.model_request(TINY, ROOT_ID, manifest)
    planned_manifest = json.loads((prepared / "planned_manifest.json").read_text())["manifest"]
    assert RUN.provider_view(manifest) == RUN.provider_view(planned_manifest)
    import hashlib
    assert hashlib.sha256(body).hexdigest() == plan["model"]["sha256"]
    _caps_hold(prepared, "upload")
    calls = list(fake.calls)
    assert _upload(prepared, fake, clock)["upload"]["state"] == "VERIFIED"
    assert fake.calls.count("exportTable") == tables and fake.calls.count("exportImage") == 1 == calls.count(
        "exportImage")


def test_an_interrupted_upload_resumes_without_a_second_export(prepared):
    clock = Clock()
    fake = FakeEarthEngine(prepared, clock, tables={0: "crash"})
    with pytest.raises(Crash):
        _upload(prepared, fake, clock)
    assert _state(prepared)["upload"]["state"] == "UPLOADING" and not (prepared / "run.lock").exists()
    state = _upload(prepared, fake, clock)
    tables = json.loads((prepared / "prepared.json").read_text())["plan"]["tables"]["count"]
    assert state["upload"]["state"] == "VERIFIED"
    assert sorted(fake.exported[:-1]) == [f"weights_part_{k:04d}" for k in range(tables)]   # each table once
    assert "getTaskStatus" in fake.calls
    _caps_hold(prepared, "upload")


def test_a_failed_table_stops_everything(prepared):
    clock = Clock()
    fake = FakeEarthEngine(prepared, clock, tables={1: "fail"})
    with pytest.raises(RUN.RunStopped, match="a failed table is never resumed"):
        _upload(prepared, fake, clock)
    assert _state(prepared)["upload"]["state"] == "FAILED" and "input" not in fake.exported
    with pytest.raises(RUN.RunStopped, match="never resumed"):
        _upload(prepared, fake, clock)
    with pytest.raises(RUN.RunStopped, match="run upload first"):
        _run(prepared, fake, clock)
    assert "logits" not in fake.exported


def test_an_existing_destination_or_other_weights_stop_the_upload_before_any_export(prepared, template, tmp_path):
    clock = Clock()
    fake = FakeEarthEngine(prepared, clock, extra={f"{ROOT_ID}/input": {"type": "IMAGE"}})
    with pytest.raises(RUN.RunStopped, match="exists; nothing is submitted"):
        _upload(prepared, fake, clock)
    assert fake.exported == []
    other = tmp_path / "other"
    shutil.copytree(template, other)
    planned = json.loads((other / "planned_manifest.json").read_text())
    first = next(iter(planned["manifest"]["layers"]))
    planned["manifest"]["layers"][first]["sha256"] = "0" * 64
    (other / "planned_manifest.json").write_text(json.dumps(planned))
    fake = FakeEarthEngine(other, clock)
    with pytest.raises(RUN.RunStopped, match="differ from the prepared manifest's fingerprints"):
        _upload(other, fake, clock)
    assert fake.exported == []


def test_a_failed_row_check_stops_the_upload(prepared):
    clock = Clock()
    fake = FakeEarthEngine(prepared, clock)
    with pytest.raises(RUN.RunStopped, match="the row check failed"):
        _upload(prepared, fake, clock, verify=_failed)
    assert _state(prepared)["upload"]["state"] == "FAILED" and "input" not in fake.exported


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
@pytest.fixture
def uploaded(prepared):
    clock = Clock()
    _upload(prepared, FakeEarthEngine(prepared, clock), clock)
    return prepared


def test_run_refuses_before_a_verified_upload(prepared):
    clock = Clock()
    with pytest.raises(RUN.RunStopped, match="not complete and verified"):
        _run(prepared, FakeEarthEngine(prepared, clock), clock)


@pytest.mark.parametrize("model, outcome", [
    ("ok", "completed"), ("complex", "refused_too_complex"), ("error", "submission_error"),
    ("fail", "task_failed"), ("wall", "stopped_at_limit"), ("queue", "stopped_at_limit"),
])
def test_each_outcome_of_the_run(uploaded, model, outcome):
    clock = Clock()
    fake = FakeEarthEngine(uploaded, clock, model=model)
    state = _run(uploaded, fake, clock)
    record = state["model"]
    assert record["outcome"] == outcome and fake.exported == ["logits"] and fake.deadline == 900_000
    assert fake.retries == 0
    if model in ("wall", "queue"):
        assert record["limit"] == model and fake.calls.count("cancelOperation") == 1 and record["cancelled"]
    if model == "complex":
        assert "Computation is too complex." in record["error"]
    if model == "fail":
        assert record["eecu_seconds"] == 12.5 and "memory" in str(record["error"])
    if model == "ok":
        assert record["state"] == "VERIFIED" and record["eecu_seconds"] == 12.5
    _caps_hold(uploaded, "run")
    calls = list(fake.calls)
    assert _run(uploaded, fake, clock)["model"]["outcome"] == outcome and fake.calls == calls   # never again
    text = RUN.report(uploaded)["outcome"]
    assert text["outcome"] == outcome


def test_a_lost_answer_is_never_submitted_again(uploaded):
    clock = Clock()
    fake = FakeEarthEngine(uploaded, clock, model="lost")
    with pytest.raises(Crash):
        _run(uploaded, fake, clock)
    assert _state(uploaded)["model"]["state"] == "SUBMITTING"
    assert RUN.report(uploaded)["outcome"]["outcome"] == "answer_lost"
    with pytest.raises(RUN.RunStopped, match="never submitted again"):
        _run(uploaded, fake, clock)
    assert fake.exported == ["logits"] and fake.calls.count("exportImage") == 1


def test_the_budget_stops_before_the_submission(uploaded):
    clock = Clock()
    fake = FakeEarthEngine(uploaded, clock)
    tight = RUN.Limits(budget_eecu_seconds=0.1, task_seconds=900, queue_seconds=1800, interval_seconds=10)
    state = _run(uploaded, fake, clock, tight)
    assert state["model"]["outcome"] == "stopped_at_limit" and state["model"]["limit"] == "budget"
    assert fake.exported == [] and "nothing submitted" in RUN.report(uploaded)["outcome"]["text"]


def test_an_existing_logits_destination_stops_before_submission(uploaded):
    clock = Clock()
    fake = FakeEarthEngine(uploaded, clock, extra={f"{ROOT_ID}/logits": {"type": "IMAGE"}})
    with pytest.raises(RUN.RunStopped, match="exists"):
        _run(uploaded, fake, clock)
    assert fake.exported == [] and _state(uploaded)["model"]["state"] == "PREPARED"


def _failed_polls(state):
    return [entry for entry in state["history"] if entry["event"] == "poll failed"]


@pytest.mark.parametrize("failing, attempts", [
    ({1}, [1]), ({1, 2, 3}, [1, 2, 3]), ({1, 2, 3, 5, 6, 7}, [1, 2, 3, 1, 2, 3]),   # an answer resets the count
])
def test_a_failed_status_read_is_tried_again(uploaded, failing, attempts):
    clock = Clock()
    fake = FakeEarthEngine(uploaded, clock, poll_errors={"logits": failing})
    state = _run(uploaded, fake, clock)
    assert state["model"]["outcome"] == "completed" and state["model"]["state"] == "VERIFIED"
    failed = _failed_polls(state)
    assert [entry["attempt"] for entry in failed] == attempts and len(failed) == len(failing)
    assert all(entry["export"] == "model" and entry["error"].startswith("EEException: Deadline exceeded on status read")
               for entry in failed)
    assert fake.exported == ["logits"] and fake.calls.count("exportImage") == 1
    _caps_hold(uploaded, "run")


def test_four_failed_status_reads_in_a_row_stop_the_run_until_it_is_run_again(uploaded):
    clock = Clock()
    fake = FakeEarthEngine(uploaded, clock, poll_errors={"logits": {1, 2, 3, 4}})
    with pytest.raises(RUN.RunStopped, match="the task is not resubmitted: run again to poll it"):
        _run(uploaded, fake, clock)
    state = _state(uploaded)
    assert state["reason"] == ("getOperation failed: EEException: Deadline exceeded on status read 4; the task is not "
                               "resubmitted: run again to poll it")
    assert state["model"]["state"] == "SUBMITTED" and "outcome" not in state["model"]
    assert [entry["attempt"] for entry in _failed_polls(state)] == [1, 2, 3, 4]
    assert fake.calls.count("getOperation") == 4 and fake.calls.count("exportImage") == 1
    state = _run(uploaded, fake, clock)                                            # polls the same task
    assert state["model"]["outcome"] == "completed" and fake.exported == ["logits"]
    assert fake.calls.count("exportImage") == 1 and len(_failed_polls(state)) == 4
    _caps_hold(uploaded, "run")


def test_status_reads_tried_again_count_against_the_cap(uploaded):
    clock = Clock()
    fake = FakeEarthEngine(uploaded, clock, model="queue", poll_errors={"logits": set(range(1, 100_000, 2))})
    patient = RUN.Limits(budget_eecu_seconds=600, task_seconds=900, queue_seconds=1e9, interval_seconds=10)
    with pytest.raises(RUN.RunStopped, match="the cap of .* getOperation calls is reached"):
        _run(uploaded, fake, clock, patient)
    cap = json.loads((uploaded / "prepared.json").read_text())["plan"]["calls"]["run"]["getOperation"]
    assert fake.calls.count("getOperation") == cap and len(_failed_polls(_state(uploaded))) == (cap + 1) // 2
    assert fake.calls.count("exportImage") == 1 and fake.calls.count("cancelOperation") == 0
    _caps_hold(uploaded, "run")


# ---------------------------------------------------------------------------
# compare and report
# ---------------------------------------------------------------------------
def _reference():
    chip = RUN.evaluation_chip(TINY)
    with torch.no_grad():
        return RUN.reference_model(TINY)(torch.from_numpy(chip).permute(2, 0, 1)[None])[0].permute(1, 2, 0).numpy()


@pytest.mark.parametrize("shift, passed", [(0.0, True), (np.float32(2e-4), False)])
def test_compare_passes_and_fails_and_the_report_is_scrubbed(uploaded, shift, passed):
    clock = Clock()
    fake = FakeEarthEngine(uploaded, clock)
    _run(uploaded, fake, clock)
    reference = _reference()
    fake.values = {"input": RUN.evaluation_chip(TINY), "logits": reference + shift}
    with offline_earth_engine():
        record = RUN.compare(uploaded, project=PROJECT, data=fake, ee_module=ee)
    assert record["passed"] is passed and record["input_equals_local"] and fake.calls.count("computePixels") == 2
    assert record["max_abs"] == pytest.approx(float(shift), abs=1e-6) and record["argmax_agreement"] == 1.0
    with offline_earth_engine():
        RUN.compare(uploaded, project=PROJECT, data=fake, ee_module=ee)
    assert fake.calls.count("computePixels") == 2                                  # fetched once
    report = RUN.report(uploaded)
    assert report["outcome"]["outcome"] == "completed"
    assert ("compare passed" if passed else "compare failed") in report["outcome"]["text"]
    text = json.dumps(report)
    assert ROOT_ID not in text and "operations/" not in text and "projects/" not in text


def test_compare_refuses_without_a_completed_run(uploaded):
    with pytest.raises(RUN.RunStopped, match="nothing to compare"):
        RUN.compare(uploaded, project=PROJECT, data=None, ee_module=ee)


def test_the_command_line_plans_and_reports_offline(tmp_path, capsys, prepared):
    output = tmp_path / "plan.json"
    assert RUN.main(["plan", "--recipe", TINY, "--json", "--output", str(output)]) == 0
    assert json.loads(capsys.readouterr().out) == json.loads(output.read_text())
    with pytest.raises(SystemExit):                      # run takes no default for its limits
        RUN.main(["run", "--run-dir", str(prepared), "--project", PROJECT])
    assert RUN.main(["report", "--run-dir", str(prepared)]) == 0
    assert json.loads(capsys.readouterr().out)["outcome"]["outcome"] is None


# ---------------------------------------------------------------------------
# The SMP recipe (no forward of the model)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not HAS_SMP, reason="segmentation-models-pytorch is not installed")
def test_every_smp_parameter_has_a_declared_native_rule():
    model = RUN.RECIPES[SMP].build()
    modules = dict(model.named_modules())
    kinds = {}
    for name, _ in model.named_parameters():
        parent, field = name.rsplit(".", 1)
        row = REF.native_rule(name, modules[parent], field, RUN.RECIPES[SMP].schemes)
        kinds.setdefault(name.split(".", 1)[0], set()).add(row["kind"])
    assert kinds == {"encoder": {"torch_default", "batch_norm"}, "decoder": {"kaiming_relu", "batch_norm"},
                     "segmentation_head": {"xavier"}}


def test_a_recipe_without_its_package_says_so(monkeypatch):
    import builtins

    real = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name.startswith("segmentation_models_pytorch"):
            raise ImportError("no module")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    with pytest.raises(RUN.RecipeUnavailable, match="segmentation-models-pytorch"):
        RUN.RECIPES[SMP].build()


# ---------------------------------------------------------------------------
# Staged runs: a tiny U-Net whose skip crosses the cut
# ---------------------------------------------------------------------------
UNET = "tiny_unet_2band_16"
STAGED_ROOT = "projects/example/assets/cnn/staged1"


class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.a, self.bn_a = nn.Conv2d(2, 8, 3, padding=1), nn.BatchNorm2d(8)
        self.b, self.bn_b = nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16)
        self.c, self.out = nn.Conv2d(16, 16, 3, padding=1), nn.Conv2d(24, 3, 1)

    def forward(self, x):
        a = torch.relu(self.bn_a(self.a(x)))
        b = torch.relu(self.bn_b(self.b(a)))
        up = nn.functional.interpolate(self.c(b), scale_factor=2, mode="nearest")
        return self.out(torch.cat([up, a], dim=1))


@pytest.fixture(scope="module", autouse=True)
def unet_recipe():
    RUN.RECIPES[UNET] = RUN.Recipe(UNET, "a tiny U-Net for the tests", lambda: TinyUNet().eval(), 2, 16, (), None,
                                   100_000, cut_lists={"skip": ("relu", "relu_1"),
                                                       "unread": ("relu", "c", "relu_1")})
    yield
    del RUN.RECIPES[UNET]


@pytest.fixture(scope="module")
def unet_upload(unet_recipe, tmp_path_factory):
    """An earlier one-graph upload of the tiny U-Net, verified, to adopt; and its fake's assets."""
    directory = tmp_path_factory.mktemp("unet") / "upload"
    RUN.prepare(directory, recipe=UNET, project=PROJECT, asset_root=ROOT_ID)
    clock = Clock()
    fake = FakeEarthEngine(directory, clock)
    _upload(directory, fake, clock, recipe=UNET)
    return directory, dict(fake.assets)


@pytest.fixture
def staged(unet_upload, tmp_path):
    upload, _ = unet_upload
    directory = tmp_path / "staged"
    RUN.prepare_staged(directory, recipe=UNET, cuts="skip", project=PROJECT, asset_root=STAGED_ROOT, adopt=upload,
                       path_bound=100_000, limits=LIMITS)
    return directory


def _staged_fake(directory, unet_upload, **kwargs):
    fake = FakeEarthEngine(directory, Clock(), folders=(PARENT,), **kwargs)
    fake.assets.update(unet_upload[1])
    return fake


def test_a_staged_plan_reads_what_each_export_depends_on(unet_recipe):
    record = RUN.staged_plan(UNET, "skip", asset_root=STAGED_ROOT)
    exports = record["exports"]
    assert [e["leaf"] for e in exports] == ["stage01", "stage02", "logits"]
    assert [e["node"] for e in exports] == ["relu", "relu_1", "output"]
    assert [e["reads"] for e in exports] == [["input"], ["stage01"], ["stage01", "stage02"]]   # the skip
    assert exports[0]["grid"]["height"] == 16 and exports[1]["grid"]["height"] == 8
    assert all(e["counts"]["clip"] > 0 and e["counts"]["longest_path"] > 0 for e in exports)
    assert record["path_bound"] is None and RUN.staged_plan(UNET, "skip", asset_root=STAGED_ROOT) == record
    text = RUN.staged_text(record)
    assert "none set: this plan measures and cannot be prepared" in text
    assert all(e["sha256"] in text for e in exports) and "stage_refused_too_complex" in text
    assert record["calls"]["run"]["exportImage"] == 3 and "exportTable" not in record["calls"]["run"]


def test_the_staged_plan_refuses_what_it_can_check_offline(unet_recipe, monkeypatch):
    longest = max(e["counts"]["longest_path"] for e in RUN.staged_plan(UNET, "skip", asset_root=STAGED_ROOT)["exports"])
    with pytest.raises(ValueError, match="over the longest-path bound"):
        RUN.staged_plan(UNET, "skip", asset_root=STAGED_ROOT, path_bound=longest - 1)
    assert RUN.staged_plan(UNET, "skip", asset_root=STAGED_ROOT, path_bound=longest)["path_bound"] == longest
    with pytest.raises(ValueError, match="'relu_1' is read by no later export"):
        RUN.staged_plan(UNET, "unread", asset_root=STAGED_ROOT)
    with pytest.raises(ValueError, match="unknown cut list"):
        RUN.staged_plan(UNET, "nope", asset_root=STAGED_ROOT)
    # the check that a saved node cuts off what it should is live: a wrong dependency set trips it
    monkeypatch.setattr(RUN, "frontier", lambda graph_module, target, saved: {"input"})
    with pytest.raises(ValueError, match="does not cut off what it should"):
        RUN.staged_plan(UNET, "skip", asset_root=STAGED_ROOT)


def test_adoption_reproduces_the_manifest_or_refuses(unet_upload, tmp_path):
    upload, _ = unet_upload
    adopted = RUN.adoption(UNET, upload)
    assert adopted["root"] == ROOT_ID and adopted["manifest_sha256"] == json.loads(
        (upload / "prepared.json").read_text())["plan"]["manifest_sha256"]
    other = tmp_path / "other"
    shutil.copytree(upload, other)
    manifest = json.loads((other / "weights_manifest.json").read_text())
    first = next(iter(manifest["layers"]))
    manifest["layers"][first]["sha256"] = "0" * 64
    (other / "weights_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="does not reproduce"):
        RUN.adoption(UNET, other)
    with pytest.raises(ValueError, match="does not reproduce"):
        RUN.prepare_staged(tmp_path / "never", recipe=UNET, cuts="skip", project=PROJECT, asset_root=STAGED_ROOT,
                           adopt=other, path_bound=100_000)
    assert not (tmp_path / "never").exists()
    with pytest.raises(ValueError, match="needs a longest-path bound"):
        RUN.prepare_staged(tmp_path / "unbounded", recipe=UNET, cuts="skip", project=PROJECT,
                           asset_root=STAGED_ROOT, adopt=upload, path_bound=None)


def test_a_staged_run_adopts_with_get_asset_only_runs_in_order_and_compares(staged, unet_upload):
    fake = _staged_fake(staged, unet_upload)
    state = _run(staged, fake, fake.clock)
    assert state["outcome"] == {"name": "completed", "export": None}
    assert fake.exported == ["stage01", "stage02", "logits"] and "exportTable" not in fake.calls
    tables = json.loads((staged / "prepared.json").read_text())["plan"]["tables"]["count"]
    # the output folder levels (one created), then each adopted table and the input: getAsset only
    assert fake.calls[:3 + tables + 1] == ["getAsset", "getAsset", "createFolder"] + ["getAsset"] * (tables + 1)
    assert all(state["exports"][leaf]["state"] == "VERIFIED" for leaf in ("stage01", "stage02", "logits"))
    _caps_hold(staged, "run")
    calls = list(fake.calls)
    assert _run(staged, fake, fake.clock)["outcome"]["name"] == "completed" and fake.calls == calls
    chip = RUN.evaluation_chip(UNET)
    with torch.no_grad():
        reference = RUN.reference_model(UNET)(torch.from_numpy(chip).permute(2, 0, 1)[None])[0].permute(1, 2, 0).numpy()
    fake.values = {"input": chip, "logits": reference}
    with offline_earth_engine():
        assert RUN.compare(staged, project=PROJECT, data=fake, ee_module=ee)["passed"]
    record = RUN.report(staged)
    assert record["outcome"]["outcome"] == "completed" and "compare passed" in record["outcome"]["text"]
    assert "argmax equal at 256 of 256 pixels" in record["outcome"]["text"]
    assert [row["eecu_seconds"] for row in record["exports"]] == [0.5, 0.5, 12.5]
    text = json.dumps(record)
    assert ROOT_ID not in text and STAGED_ROOT not in text and "operations/" not in text and "projects/" not in text


def test_a_missing_adopted_table_stops_before_any_export(staged, unet_upload):
    fake = _staged_fake(staged, unet_upload)
    table = json.loads((staged / "prepared.json").read_text())["plan"]["tables"]["assets"][0]
    del fake.assets[table]
    with pytest.raises(RUN.RunStopped, match="does not exist; nothing is submitted"):
        _run(staged, fake, fake.clock)
    assert fake.exported == [] and "exportImage" not in fake.calls


def test_a_refused_middle_export_ends_the_run_and_keeps_the_earlier_asset(staged, unet_upload):
    fake = _staged_fake(staged, unet_upload, exports={"stage02": "complex"})
    state = _run(staged, fake, fake.clock)
    assert state["outcome"]["name"] == "stage_refused_too_complex" and state["outcome"]["export"] == "stage02"
    assert fake.exported == ["stage01", "stage02"] and state["exports"]["stage01"]["state"] == "VERIFIED"
    assert f"{STAGED_ROOT}/stage01" in fake.assets
    outcome = RUN.report(staged)["outcome"]
    assert outcome["outcome"] == "stage_refused_too_complex" and outcome["text"].startswith("stage02 refused")
    with offline_earth_engine(), pytest.raises(RUN.RunStopped, match="nothing to compare"):
        RUN.compare(staged, project=PROJECT, data=fake, ee_module=ee)


def test_an_interrupted_staged_run_resumes_without_a_second_submission(staged, unet_upload):
    fake = _staged_fake(staged, unet_upload, exports={"stage02": "crash"})
    with pytest.raises(Crash):
        _run(staged, fake, fake.clock)
    assert _state(staged)["exports"]["stage02"]["state"] == "SUBMITTED"
    state = _run(staged, fake, fake.clock)
    assert state["outcome"]["name"] == "completed" and fake.exported == ["stage01", "stage02", "logits"]


def test_a_lost_answer_is_never_submitted_again(staged, unet_upload):
    fake = _staged_fake(staged, unet_upload, exports={"stage02": "lost"})
    with pytest.raises(Crash):
        _run(staged, fake, fake.clock)
    with pytest.raises(RUN.RunStopped, match="never submitted again"):
        _run(staged, fake, fake.clock)
    assert fake.exported == ["stage01", "stage02"]
    assert RUN.report(staged)["outcome"]["outcome"] == "stage_answer_lost"


@pytest.mark.parametrize("behaviour, outcome", [("fail", "stage_task_failed"), ("wall", "stage_stopped_at_limit"),
                                                ("error", "stage_submission_error")])
def test_each_staged_ending_is_named_with_its_export(staged, unet_upload, behaviour, outcome):
    fake = _staged_fake(staged, unet_upload, exports={"stage01": behaviour})
    state = _run(staged, fake, fake.clock)
    assert state["outcome"]["name"] == outcome and state["outcome"]["export"] == "stage01"
    assert fake.exported == ["stage01"]


def test_the_budget_is_checked_before_every_staged_submission(staged, unet_upload):
    fake = _staged_fake(staged, unet_upload)
    tight = RUN.Limits(budget_eecu_seconds=0.4, task_seconds=900, queue_seconds=1800, interval_seconds=10)
    state = _run(staged, fake, fake.clock, tight)
    assert state["outcome"] == {"name": "stage_stopped_at_limit", "export": "stage02", "limit": "budget"}
    assert fake.exported == ["stage01"]


# ---------------------------------------------------------------------------
# The trained recipe (the released Fields of The World v3.1 checkpoint)
# ---------------------------------------------------------------------------
FTW = "ftw_v31_unet_efficientnet_b3_8band_256"
RESOURCES = os.environ.get("TENSOR_EE_RESOURCES")
HAS_FTW = bool(RESOURCES) and (Path(RESOURCES) / RUN.FTW_CHECKPOINT).is_file()


@pytest.fixture
def resources():
    """Restore the tool's resources (and forget the models built) after a test changes them."""
    yield RUN.set_resources
    RUN.set_resources(None)


def test_the_argmax_agreement_is_reported_as_a_count():
    text = RUN._argmax_text({"argmax_agreement": 65_535 / 65_536, "argmax_agreeing": 65_535, "pixels": 65_536})
    assert text == "argmax equal at 65,535 of 65,536 pixels"
    assert "1.0000" not in RUN._argmax_text({"argmax_agreement": 65_535 / 65_536})


def test_the_trained_recipe_input_is_the_evaluation_chip_plus_one_half():
    chip = RUN.evaluation_chip(FTW)
    base = REF.evaluation_input(8, 256)
    assert chip.dtype == np.float32 and chip.shape == (256, 256, 8)
    assert np.array_equal(chip.astype(np.float64), base.astype(np.float64) + 0.5)          # exact
    assert chip.min() >= 0 and chip.max() < 1 and not (np.signbit(chip) & (chip == 0)).any()
    assert np.array_equal(RUN.evaluation_chip(SMP), base)                                  # the random recipe's own


@pytest.mark.skipif(not HAS_SMP, reason="segmentation-models-pytorch is not installed")
def test_the_trained_recipe_is_unavailable_without_its_file(resources, tmp_path):
    resources(None)
    with pytest.raises(RUN.RecipeUnavailable, match="none is given"):
        RUN.reference_model(FTW)
    resources(tmp_path)
    with pytest.raises(RUN.RecipeUnavailable, match="is not in the resources checkout"):
        RUN.reference_model(FTW)


def _fake_checkpoint(directory: Path, state: dict) -> Path:
    path = directory / RUN.FTW_CHECKPOINT
    path.parent.mkdir(parents=True)
    torch.save({"state_dict": state}, path)
    return path


@pytest.mark.skipif(not HAS_SMP, reason="segmentation-models-pytorch is not installed")
def test_a_wrong_or_malformed_weights_file_is_refused(resources, tmp_path, monkeypatch):
    import hashlib

    resources(tmp_path)
    path = _fake_checkpoint(tmp_path, {"model.encoder._conv_stem.weight": torch.zeros(1)})
    with pytest.raises(ValueError, match="not the released"):
        RUN.reference_model(FTW)
    # with the digest pinned to this file: a missing key fails the strict load
    monkeypatch.setattr(RUN, "FTW_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    resources(tmp_path)
    with pytest.raises(RuntimeError, match="Missing key"):
        RUN.reference_model(FTW)
    other = tmp_path / "other"
    path = _fake_checkpoint(other, {"encoder._conv_stem.weight": torch.zeros(1)})
    monkeypatch.setattr(RUN, "FTW_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    resources(other)
    with pytest.raises(ValueError, match="under 'model.'"):
        RUN.reference_model(FTW)


@pytest.mark.skipif(not HAS_FTW, reason="the Fields of The World v3.1 checkpoint is not in TENSOR_EE_RESOURCES")
def test_the_released_checkpoint_loads_strictly_without_calibration(resources):
    resources(RESOURCES)
    model = RUN.reference_model(FTW)
    checkpoint = torch.load(Path(RESOURCES) / RUN.FTW_CHECKPOINT, map_location="cpu", weights_only=True)["state_dict"]
    state = model.state_dict()
    assert set(state) == {key[len("model."):] for key in checkpoint}
    assert all(torch.equal(state[key[len("model."):]], value) for key, value in checkpoint.items())
    assert not model.training
