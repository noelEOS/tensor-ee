"""tools/compiled_cnn_run.py and tools/fixtures/cnn_reference.py offline: the deterministic weights
and their calibration, the exact table plan, the upload through the library's own uploader, each
outcome of the model's run, compare and the report, and an imagery input (the scene lookup, its
export, its check, the gate before a model export, compare and the map), against fake Earth Engine
clients. No cloud call: graphs are built against the offline signatures with every cloud entry
point refusing. The tool's logic runs on small local CNN recipes, with a made-up collection and
made-up image IDs; the SMP recipe's tests skip without SMP and run no forward of the model."""
from __future__ import annotations

import contextlib
import copy
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
from unittest import mock
import zlib

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

    values, valid, dtype, described = None, None, np.float32, None

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
        checked = '"valid"' in graph                        # check-input's readout: the bands and a validity band
        array = np.empty(values.shape[:2], dtype=[(band, self.dtype) for band in names]
                         + ([("valid", np.float32)] if checked else []))
        for c, band in enumerate(names):
            array[band] = values[:, :, c]
        if checked:
            array["valid"] = np.ones(values.shape[:2], np.float32) if self.valid is None else self.valid
        return array

    def computeValue(self, expression):
        self.calls.append("computeValue")
        self.expression = json.dumps(ee.serializer.encode(expression, for_cloud_api=True))
        return copy.deepcopy(self.described)

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


def _upload(directory, fake, clock, verify=_passed, recipe=TINY, source=None):
    with _on_ee_data(fake) as data:
        return RUN.upload(directory, recipe=recipe, project=PROJECT, asset_root=ROOT_ID, data=data, clock=clock,
                          sleep=clock.sleep, verify_asset_rows=verify, source=source)


def _run(directory, fake, clock, limits=LIMITS, until=None):
    return RUN.run(directory, project=PROJECT, limits=limits, data=fake, clock=clock, sleep=clock.sleep, until=until)


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
    assert record["rule"] == "float32_tolerance" and record["reference_max_abs_float64"] < 1e-5   # the tiny model
    assert record["max_abs_float64"] == pytest.approx(float(shift), abs=1e-5)
    assert record["float32_step"] == float(np.spacing(np.float32(record["largest_reference"])))
    with offline_earth_engine():
        RUN.compare(uploaded, project=PROJECT, data=fake, ee_module=ee)
    assert fake.calls.count("computePixels") == 2                                  # fetched once
    report = RUN.report(uploaded)
    assert report["outcome"]["outcome"] == "completed"
    assert ("compare passed" if passed else "compare failed") in report["outcome"]["text"]
    text = json.dumps(report)
    assert ROOT_ID not in text and "operations/" not in text and "projects/" not in text


STEP = 2.0 ** -13                                       # 1.22e-4: over the tolerance, exact in float64


@pytest.mark.parametrize("own, off, rule, within", [
    (0.0, 2 * STEP, "float32_tolerance", False),         # float32 and float64 PyTorch agree: 1e-4 against float32
    (STEP / 2, -3 * STEP / 8, "float32_tolerance", False),   # nearer float64 than float32 is; float32 inside 1e-4
    (STEP / 2, STEP / 4, "float32_tolerance", True),
    (2 * STEP, -STEP, "float64_ordering", True),         # 3 steps from float32, 1 from float64: float32 is 2 away
    (2 * STEP, 0.0, "float64_ordering", True),
    (2 * STEP, 2 * STEP, "float64_ordering", True),      # float32 PyTorch itself: no larger than its own
    (2 * STEP, -3 * STEP, "float64_ordering", False),
    (2 * STEP, 3 * STEP, "float64_ordering", False),
])
def test_the_check_is_chosen_by_the_two_references_and_never_by_the_logits(own, off, rule, within):
    """``own``: float32 PyTorch above float64 PyTorch at one value; ``off``: the logits above float64 there."""
    double = np.arange(48, dtype=np.float64).reshape(4, 4, 3) / 8
    single, logits = double.astype(np.float32), double.astype(np.float32)
    single[1, 2, 0] += np.float32(own)
    logits[1, 2, 0] += np.float32(off)
    facts = RUN.judged(logits, single, double)
    assert (facts["rule"], facts["within"]) == (rule, within)
    assert facts["reference_max_abs_float64"] == own and facts["max_abs_float64"] == abs(off)
    assert facts["max_abs"] == abs(off - own) and facts["over_tolerance"] == int(abs(off - own) > 1e-4)
    assert facts["reference_over_tolerance_float64"] == int(own > 1e-4) and facts["pixels"] == 16
    assert facts["largest_reference"] == float(np.abs(single).max())
    # the same references with other logits: the same check
    assert RUN.judged(single, single, double)["rule"] == RUN.judged(logits + 1, single, double)["rule"] == rule


def test_compare_judges_against_float64_where_float32_pytorch_is_itself_outside(uploaded):
    clock = Clock()
    fake = FakeEarthEngine(uploaded, clock)
    _run(uploaded, fake, clock)
    reference = _reference()
    double = reference.astype(np.float64)
    double[0, 0, 0] -= 2 * STEP                           # float32 PyTorch 2.4e-4 above float64 at one value
    logits = reference.copy()
    logits[0, 0, 0] = np.float32(double[0, 0, 0] - STEP)  # 1.2e-4 from float64, 3.7e-4 from float32
    fake.values = {"input": RUN.evaluation_chip(TINY), "logits": logits}
    with offline_earth_engine():
        record = RUN.compare(uploaded, project=PROJECT, data=fake, ee_module=ee, reference64=double)
    assert record["rule"] == "float64_ordering" and record["passed"] and record["over_tolerance"] == 1
    assert record["max_abs_float64"] < record["reference_max_abs_float64"] < record["max_abs"]
    assert record["reference_max_abs_float64"] == pytest.approx(2 * STEP, abs=1e-6)
    text = RUN.report(uploaded)["outcome"]["text"]
    assert "compare passed (the check: no farther from float64 PyTorch than float32 PyTorch" in text
    with offline_earth_engine():                           # the same logits, references that agree: 1e-4 against float32
        record = RUN.compare(uploaded, project=PROJECT, data=fake, ee_module=ee)
    assert record["rule"] == "float32_tolerance" and not record["passed"] and fake.calls.count("computePixels") == 2
    assert "compare failed (the check: within 0.0001 of float32 PyTorch)" in RUN.report(uploaded)["outcome"]["text"]


def test_a_comparison_recorded_before_the_float64_check_prints_as_it_did():
    old = {"passed": True, "max_abs": 3.43e-05, "over_tolerance": 0, "tolerance": 1e-4, "largest_reference": 35.69,
           "input_equals_local": True, "argmax_agreeing": 65536, "pixels": 65536}
    assert RUN._compare_text(old) == ("compare passed: largest absolute error 3.43e-05 (0 over 0.0001; largest "
                                      "reference 35.7; input equal True; argmax equal at 65,536 of 65,536 pixels)")


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
    def __init__(self, bands=2):
        super().__init__()
        self.a, self.bn_a = nn.Conv2d(bands, 8, 3, padding=1), nn.BatchNorm2d(8)
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


# ---------------------------------------------------------------------------
# An imagery input: a made-up collection, made-up image IDs, a made-up grid
# ---------------------------------------------------------------------------
IMG = "tiny_cnn_imagery_6band_16"
UNET_IMG = "tiny_unet_imagery_6band_16"
COLLECTION = "EXAMPLE/COLLECTION"
IMAGES = (f"{COLLECTION}/MADE_UP_A", f"{COLLECTION}/MADE_UP_B")
GRID_ARGUMENTS = {"crs": "EPSG:32601", "origin": (500000.0, 5000000.0), "pixel_size": 10.0}
COMMAND_LINE = ["--images", *IMAGES, "--crs", "EPSG:32601", "--origin", "500000,5000000", "--pixel-size", "10"]
IMAGERY_ROOT = "projects/example/assets/cnn/imagery1"
PLAN_KEYS = ["asset_root", "bands", "calls", "calls_to_earth_engine", "ceiling", "chip", "description", "folders",
             "format", "input", "limits", "manifest_sha256", "model", "outcomes", "placeholder_root", "recipe",
             "request_limit_bytes", "submit_deadline_seconds", "tables", "version"]
STAGED_KEYS = ["adopted", "asset_root", "calls", "calls_to_earth_engine", "ceiling", "cuts", "description", "exports",
               "folders", "format", "input", "limits", "manifest_sha256", "mode", "outcomes", "path_bound",
               "placeholder_root", "recipe", "request_limit_bytes", "submit_deadline_seconds", "tables", "tables_root",
               "version"]
"""The keys of a plan made with the arguments the tool had before it took imagery."""


def _tiny6():
    return nn.Sequential(nn.Conv2d(6, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.Conv2d(16, 3, 1)).eval()


@pytest.fixture(scope="module", autouse=True)
def imagery_recipes():
    convention = RUN.Imagery(COLLECTION, ("B4", "B3", "B2"), 2, 100, "a small local CNN on two made-up images",
                             true_colour=("B4", "B3", "B2"))
    RUN.RECIPES[IMG] = RUN.Recipe(IMG, "a small local CNN for the tests", _tiny6, 6, 16, (), None, 100_000,
                                  imagery=convention)
    RUN.RECIPES[UNET_IMG] = RUN.Recipe(UNET_IMG, "a tiny U-Net for the tests", lambda: TinyUNet(6).eval(), 6, 16, (),
                                       None, 100_000, cut_lists={"skip": ("relu", "relu_1")}, imagery=convention)
    yield
    del RUN.RECIPES[IMG], RUN.RECIPES[UNET_IMG]
    RUN.reference_model.cache_clear()
    RUN.compiled_module.cache_clear()


def _source(recipe=IMG):
    return RUN.imagery_input(recipe, IMAGES, **GRID_ARGUMENTS)


def _pixels():
    """A made-up fetched input: integer values over the divisor, float32, [16, 16, 6]."""
    values = np.arange(16 * 16 * 6, dtype=np.int64).reshape(16, 16, 6) * 7919 % 251
    return (values / 100).astype(np.float32)


def _check(directory, fake):
    with offline_earth_engine():
        return RUN.check_input(directory, project=PROJECT, data=fake, ee_module=ee)


def _renames(request):
    """{image ID: (the bands selected of it, the names they are given)} of an input request."""
    values = request["expression"]["values"]
    found = {}

    def resolve(node):
        while isinstance(node, dict) and "valueReference" in node:
            node = values[node["valueReference"]]
        return node

    def walk(node):
        call = (resolve(node) or {}).get("functionInvocationValue") if isinstance(resolve(node), dict) else None
        if not call:
            return
        if call["functionName"] == "Image.rename":
            select = resolve(call["arguments"]["input"])["functionInvocationValue"]
            load = resolve(select["arguments"]["input"])["functionInvocationValue"]
            found[resolve(load["arguments"]["id"])["constantValue"]] = (
                resolve(select["arguments"]["bandSelectors"])["constantValue"],
                resolve(call["arguments"]["names"])["constantValue"])
        for argument in call["arguments"].values():
            walk(argument)

    walk(values[request["expression"]["result"]])
    return found


@pytest.fixture(scope="module")
def imagery_template(imagery_recipes, tmp_path_factory):
    directory = tmp_path_factory.mktemp("imagery") / "prepared"
    RUN.prepare(directory, recipe=IMG, project=PROJECT, asset_root=ROOT_ID, source=_source())
    return directory


@pytest.fixture
def imagery_prepared(imagery_template, tmp_path):
    directory = tmp_path / "imagery"
    shutil.copytree(imagery_template, directory)
    return directory


@pytest.fixture
def imagery_uploaded(imagery_prepared):
    """A one-graph run whose input is imagery, its tables and input image saved: (directory, fake, clock)."""
    clock = Clock()
    fake = FakeEarthEngine(imagery_prepared, clock)
    _upload(imagery_prepared, fake, clock, recipe=IMG, source=_source())
    return imagery_prepared, fake, clock


# -- the arguments ---------------------------------------------------------------------
def test_image_arguments_need_a_recipe_with_the_convention_and_all_four():
    assert RUN.imagery_input(TINY) is None and RUN.imagery_input(IMG) is None        # as without them
    source = _source()
    assert source.to_json() == {"images": list(IMAGES), "crs": "EPSG:32601", "origin": [500000.0, 5000000.0],
                                "pixel_size": 10.0}
    assert RUN.imagery_input(IMG, **source.to_json()) == source
    for arguments, message in [
        ((TINY, IMAGES), "declares no imagery input convention"),
        ((IMG, IMAGES[:1]), "reads 2 images of EXAMPLE/COLLECTION; received 1"),
        ((IMG, (IMAGES[0], "OTHER/COLLECTION/MADE_UP")), "must be images of EXAMPLE/COLLECTION"),
        ((IMG, (IMAGES[0], COLLECTION + "/has a space")), "must be images of EXAMPLE/COLLECTION"),
    ]:
        with pytest.raises(ValueError, match=message):
            RUN.imagery_input(*arguments, **GRID_ARGUMENTS)
    with pytest.raises(ValueError, match="declares no imagery input convention"):
        RUN.imagery_input(TINY, crs="EPSG:32601")
    with pytest.raises(ValueError, match="together"):
        RUN.imagery_input(IMG, IMAGES, crs="EPSG:32601")
    with pytest.raises(ValueError, match="CRS code"):
        RUN.imagery_input(IMG, IMAGES, **dict(GRID_ARGUMENTS, crs="not a code"))
    for changed in ({"pixel_size": 0.0}, {"origin": (float("nan"), 0.0)}):
        with pytest.raises(ValueError, match="finite and --pixel-size positive"):
            RUN.imagery_input(IMG, IMAGES, **dict(GRID_ARGUMENTS, **changed))
    with pytest.raises(ValueError, match="declares no imagery input convention"):   # another recipe's input
        RUN.plan(TINY, source=source)
    with pytest.raises(ValueError, match="declares no imagery input convention"):
        RUN.main(["plan", "--recipe", TINY, *COMMAND_LINE])
    with pytest.raises(ValueError, match="together"):
        RUN.main(["plan", "--recipe", IMG, "--crs", "EPSG:32601"])


def test_todays_arguments_plan_and_prepare_as_before(planned, template, staged):
    from tensor_ee.converter.request import SIZING_REQUEST_ID, export_request, wire
    from tensor_ee.core.grid import Grid
    from tools.fixtures.core import inline_raster

    assert sorted(planned) == PLAN_KEYS and list(planned["calls"]) == ["upload", "run", "compare"]
    assert sorted(planned["input"]) == ["asset", "bytes", "description", "expect", "sha256"]
    assert planned["description"] == "a small local CNN for the tests"
    assert planned["calls"]["compare"] == {"computePixels": 2 + RUN.CALL_MARGIN}
    affine = '{"scaleX": 10, "shearX": 0, "translateX": 1000, "shearY": 0, "scaleY": -10, "translateY": 2000}'
    assert json.dumps(planned["input"]["expect"]["grid"]["affineTransform"]) == affine       # integers, as before
    with offline_earth_engine():                           # the input request is the chip inline on the fixed grid
        grid = Grid("EPSG:3857", (10, 0, 1000, 0, -10, 2000), 16, 16)
        image = inline_raster(ee, grid.projection(), RUN.evaluation_chip(TINY), ["b0", "b1"])
        body = wire(export_request(image, grid, asset_id=f"{RUN.PLAN_ROOT}/input", description="compiled_cnn_input",
                                   request_id=SIZING_REQUEST_ID))
    assert (len(body), hashlib.sha256(body).hexdigest()) == (planned["input"]["bytes"], planned["input"]["sha256"])
    prepared = json.loads((template / "prepared.json").read_text())
    assert prepared["arguments"] == {"recipe": TINY, "project": PROJECT, "asset_root": ROOT_ID}
    assert sorted(_state(template)) == ["format", "history", "model", "prepared_sha256", "reason", "upload", "version"]
    staged_prepared = json.loads((staged / "prepared.json").read_text())
    record = staged_prepared["plan"]
    assert sorted(record) == STAGED_KEYS and sorted(record["calls"]) == ["compare", "run"]
    assert sorted(record["input"]) == ["asset", "expect"] and record["input"]["asset"] == f"{ROOT_ID}/input"
    assert json.dumps(record["input"]["expect"]["grid"]["affineTransform"], sort_keys=True) == json.dumps(
        json.loads(affine), sort_keys=True)
    assert sorted(staged_prepared["arguments"]) == ["adopt", "asset_root", "cuts", "path_bound", "project", "recipe"]
    assert sorted(_state(staged)) == ["adopt", "exports", "format", "history", "mode", "outcome", "prepared_sha256",
                                      "reason", "version"]
    assert "imagery" not in RUN.plan(IMG) and RUN.plan(IMG)["description"] == "a small local CNN for the tests"


# -- the plan --------------------------------------------------------------------------
def test_an_imagery_plan_reads_the_pinned_images_and_names_every_call(imagery_template):
    from tensor_ee.converter.request import request_reads

    prepared = json.loads((imagery_template / "prepared.json").read_text())
    plan = prepared["plan"]
    assert prepared["arguments"]["imagery"] == _source().to_json() and plan["calls_to_earth_engine"] == 0
    assert sorted(set(plan) - set(PLAN_KEYS)) == ["imagery"]
    assert plan["description"] == "a small local CNN on two made-up images"
    imagery = plan["imagery"]
    assert (imagery["collection"], imagery["bands"], imagery["divisor"]) == (COLLECTION, ["B4", "B3", "B2"], 100)
    assert imagery["images"] == list(IMAGES)
    assert imagery["grid"] == {"crs": "EPSG:32601", "origin": [500000.0, 5000000.0], "pixel_size": 10.0,
                               "transform": [10.0, 0.0, 500000.0, 0.0, -10.0, 5000000.0], "height": 16, "width": 16}
    grid = {"dimensions": {"width": 16, "height": 16}, "crsCode": "EPSG:32601",
            "affineTransform": {"scaleX": 10.0, "shearX": 0.0, "translateX": 500000.0, "shearY": 0.0, "scaleY": -10.0,
                                "translateY": 5000000.0}}
    assert plan["input"]["expect"]["grid"] == grid == plan["model"]["expect"]["grid"]
    request = json.loads((imagery_template / prepared["requests"]["input"]["file"]).read_text())
    assert request_reads(request) == {"images": sorted(IMAGES), "tables": []}
    assert _renames(request) == {IMAGES[0]: (["B4", "B3", "B2"], ["b0", "b1", "b2"]),        # the first image first
                                 IMAGES[1]: (["B4", "B3", "B2"], ["b3", "b4", "b5"])}
    text = json.dumps(request)
    assert text.count('"Image.divide"') == 1 and text.count('"Image.toFloat"') == 1
    assert not any(name in text for name in ("Image.unmask", "Image.resample", "Image.reduceResolution"))
    assert RUN.declared_float32(request) and plan["input"]["float32_declared"] is True
    model = json.loads((imagery_template / prepared["requests"]["model"]["file"]).read_text())
    assert request_reads(model)["images"] == [f"{ROOT_ID}/input"]
    assert sorted(plan["calls"]) == ["check_input", "compare", "discover", "run", "upload"]
    assert plan["calls"]["discover"] == {"computeValue": 1}
    assert plan["calls"]["check_input"] == plan["calls"]["compare"] == {"computePixels": 1 + RUN.CALL_MARGIN}
    numerical = RUN.plan(IMG, asset_root=ROOT_ID)
    assert plan["calls"]["upload"] == numerical["calls"]["upload"] and plan["calls"]["run"] == numerical["calls"]["run"]
    assert plan["imagery"]["readback"]["bands"] == [f"b{k}" for k in range(6)] + ["valid"]
    assert plan["imagery"]["readback"]["response_bytes"] == 4 * 7 * 16 * 16
    printed = RUN.plan_text(plan)
    assert all(image in printed for image in IMAGES) and plan["input"]["sha256"] in printed
    assert plan["imagery"]["readback"]["graph_sha256"] in printed and "EPSG:32601" in printed
    assert "check-input" in printed and "Calls, discover" in printed and "inline" not in printed
    again = RUN.plan(IMG, asset_root=ROOT_ID, source=_source())
    assert again == plan                                                                     # exact, deterministic
    assert list(again["calls"]) == ["discover", "upload", "check_input", "run", "compare"]   # in the run's order


def test_an_input_that_is_not_cast_to_float32_last_is_refused_offline(monkeypatch):
    source = _source()
    with offline_earth_engine():
        cast = json.loads(RUN.input_request(IMG, ROOT_ID, source))
    assert RUN.declared_float32(cast)
    built = RUN.imagery_image
    monkeypatch.setattr(RUN, "imagery_image", lambda *arguments: built(*arguments).multiply(1))
    with offline_earth_engine(), pytest.raises(ValueError, match="not cast to float32 last"):
        RUN.input_request(IMG, ROOT_ID, source)
    with pytest.raises(ValueError, match="not cast to float32 last"):
        RUN.plan(IMG, source=source)
    # on the request as sent: the same body with the cast taken out of it
    values = cast["expression"]["values"]
    text = json.dumps(cast)
    assert text.count('"functionName": "Image.toFloat"') == 1

    def uncast(node):
        if isinstance(node, dict):
            call = node.get("functionInvocationValue")
            if call and call["functionName"] == "Image.toFloat":
                return uncast(call["arguments"]["value"])
            return {key: uncast(value) for key, value in node.items()}
        return [uncast(value) for value in node] if isinstance(node, list) else node

    assert not RUN.declared_float32(dict(cast, expression=dict(cast["expression"], values=uncast(values))))


# -- one graph: upload, check-input, the gate, compare, the report -------------------------
@pytest.mark.parametrize("shift, passed", [(0.0, True), (np.float32(2e-4), False)])
def test_an_imagery_run_needs_its_input_check_and_compares_on_the_fetched_input(imagery_prepared, shift, passed):
    clock = Clock()
    fake = FakeEarthEngine(imagery_prepared, clock)
    state = _upload(imagery_prepared, fake, clock, recipe=IMG, source=_source())
    assert state["upload"]["state"] == "VERIFIED" and state["upload"]["input"]["state"] == "VERIFIED"
    assert fake.exported[-1] == "input" and fake.calls.count("exportImage") == 1
    with pytest.raises(ValueError, match="other arguments"):                       # without the images: another run
        _upload(imagery_prepared, fake, clock, recipe=IMG)
    with pytest.raises(RUN.RunStopped, match="no check record: run check-input; no model export is submitted"):
        _run(imagery_prepared, fake, clock)
    assert "logits" not in fake.exported and _state(imagery_prepared)["model"]["state"] == "PREPARED"
    pixels = _pixels()
    fake.values = {"input": pixels}
    record = _check(imagery_prepared, fake)
    assert record["passed"] and record["masked_pixels"] == 0 and record["pixels"] == 256
    assert (record["shape"], record["dtype"], record["float32"], record["finite"]) == ([16, 16, 6], "float32", True, True)
    assert record["sha256"] == hashlib.sha256(pixels.tobytes()).hexdigest()
    assert record["bands"] == [{"band": f"b{c}", "min": float(pixels[..., c].min()), "max": float(pixels[..., c].max())}
                               for c in range(6)]
    assert record["images"] == list(IMAGES) and record["grid"]["crs"] == "EPSG:32601"
    reference = RUN.forward(IMG, pixels)
    double = RUN.forward(IMG, pixels, double=True)
    assert reference.dtype == np.float32 and double.dtype == np.float64 and reference.shape == (16, 16, 3)
    facts = record["reference"]
    assert facts["largest_logit"] == float(np.abs(reference).max()) and sum(facts["class_counts"]) == 256
    assert facts["float32_vs_float64_largest_absolute"] == float(np.abs(reference.astype(np.float64) - double).max())
    assert facts["class_counts"] == [int(n) for n in np.bincount(reference.argmax(-1).ravel(), minlength=3)]
    assert fake.calls.count("computePixels") == 1
    assert _check(imagery_prepared, fake) == record and fake.calls.count("computePixels") == 1      # fetched once
    state = _run(imagery_prepared, fake, clock)
    assert state["model"]["outcome"] == "completed" and fake.exported[-1] == "logits"
    fake.values["logits"] = reference + shift
    with offline_earth_engine():
        compared = RUN.compare(imagery_prepared, project=PROJECT, data=fake, ee_module=ee)
    assert compared["passed"] is passed and compared["max_abs"] == pytest.approx(float(shift), abs=1e-6)
    assert compared["input_equals_local"] is None                                  # no local chip: not reported true
    assert compared["input"]["source"] == "imagery" and compared["input"]["sha256"] == record["sha256"]
    assert compared["input"]["reference_equals_check"] is True and compared["rule"] == "float32_tolerance"
    assert compared["input"]["masked_pixels"] == 0 and sum(compared["class_counts"]["earth_engine"]) == 256
    assert fake.calls.count("computePixels") == 2            # the check's fetch and the logits: no second input fetch
    _caps_hold(imagery_prepared, "check_input")
    _caps_hold(imagery_prepared, "compare")
    report = RUN.report(imagery_prepared)
    assert report["imagery"]["images"] == list(IMAGES) and report["imagery"]["grid"]["origin"] == [500000.0, 5000000.0]
    assert report["input_check"]["sha256"] == record["sha256"] and report["compare"]["input_equals_local"] is None
    outcome = report["outcome"]["text"]
    assert ("compare passed" if passed else "compare failed") in outcome
    assert "input imagery, the reference computed on the fetched array" in outcome and "input equal" not in outcome
    text = json.dumps(report)
    assert ROOT_ID not in text and "operations/" not in text and "projects/" not in text


def test_a_masked_pixel_stops_the_run_before_the_model(imagery_uploaded):
    directory, fake, clock = imagery_uploaded
    fake.values = {"input": _pixels()}
    fake.valid = np.ones((16, 16), np.float32)
    fake.valid[3, 5] = 0
    record = _check(directory, fake)
    assert record["masked_pixels"] == 1 and record["passed"] is False and record["reference"] is None
    with pytest.raises(RUN.RunStopped, match="the imagery input has 1 masked pixels; no model export is submitted"):
        _run(directory, fake, clock)
    assert "logits" not in fake.exported and fake.calls.count("exportImage") == 1          # the input's only
    assert _state(directory)["model"]["state"] == "PREPARED"
    report = RUN.report(directory)
    assert report["input_check"]["masked_pixels"] == 1 and report["outcome"]["outcome"] is None


@pytest.mark.parametrize("fault", ["float64", "not finite"])
def test_a_fetch_that_is_not_finite_float32_does_not_pass(imagery_uploaded, fault):
    directory, fake, clock = imagery_uploaded
    pixels = _pixels()
    if fault == "float64":
        fake.dtype = np.float64
    else:
        pixels[2, 2, 2] = np.nan
    fake.values = {"input": pixels}
    record = _check(directory, fake)
    assert record["passed"] is False and record["masked_pixels"] == 0 and record["reference"] is None
    assert (record["dtype"], record["finite"]) == (("float64", True) if fault == "float64" else ("float32", False))
    with pytest.raises(RUN.RunStopped, match="the input check did not pass"):
        _run(directory, fake, clock)
    assert "logits" not in fake.exported


def test_the_input_check_record_must_be_of_this_run_and_its_fetched_array(imagery_uploaded, uploaded):
    directory, fake, clock = imagery_uploaded
    fake.values = {"input": _pixels()}
    assert _check(directory, fake)["passed"]
    np.save(directory / "fetched" / "input.npy", _pixels() + np.float32(1))                # another array
    with pytest.raises(RUN.RunStopped, match="not of this run's prepared input and its fetched array"):
        _run(directory, fake, clock)
    np.save(directory / "fetched" / "input.npy", _pixels())
    record = json.loads((directory / "input_check.json").read_text())
    (directory / "input_check.json").write_text(json.dumps(dict(record, prepared_sha256="0" * 64)))
    with pytest.raises(RUN.RunStopped, match="not of this run's prepared input and its fetched array"):
        _run(directory, fake, clock)
    assert "logits" not in fake.exported and fake.calls.count("computePixels") == 1
    with pytest.raises(ValueError, match="numerical chip, not imagery"):                   # nothing to check
        RUN.check_input(uploaded, project=PROJECT, data=fake, ee_module=ee)


def test_check_input_waits_for_the_saved_input(imagery_prepared):
    fake = FakeEarthEngine(imagery_prepared, Clock())
    with pytest.raises(RUN.RunStopped, match="not saved and verified: nothing to check"):
        RUN.check_input(imagery_prepared, project=PROJECT, data=fake, ee_module=ee)
    assert fake.calls == []


def test_a_refused_input_export_is_never_submitted_again_by_upload(prepared):
    clock = Clock()
    fake = FakeEarthEngine(prepared, clock, exports={"input": "error"})
    with pytest.raises(RUN.RunStopped, match="the input export was refused"):
        _upload(prepared, fake, clock)
    with pytest.raises(RUN.RunStopped, match="the input export failed and is never submitted again"):
        _upload(prepared, fake, clock)
    assert fake.exported.count("input") == 1 and _state(prepared)["upload"]["state"] == "ROWS_VERIFIED"
    with pytest.raises(RUN.RunStopped, match="run upload first"):
        _run(prepared, fake, clock)


# -- staged: the tables adopted, the input the run's own --------------------------------------
@pytest.fixture(scope="module")
def unet_imagery_upload(imagery_recipes, tmp_path_factory):
    """An earlier upload of the imagery U-Net recipe with its numerical chip, verified: a run
    with an imagery input adopts its tables, never its input image. With its fake's assets."""
    directory = tmp_path_factory.mktemp("unet_imagery") / "upload"
    RUN.prepare(directory, recipe=UNET_IMG, project=PROJECT, asset_root=ROOT_ID)
    clock = Clock()
    fake = FakeEarthEngine(directory, clock)
    _upload(directory, fake, clock, recipe=UNET_IMG)
    return directory, dict(fake.assets)


@pytest.fixture
def staged_imagery(unet_imagery_upload, tmp_path):
    directory = tmp_path / "staged_imagery"
    RUN.prepare_staged(directory, recipe=UNET_IMG, cuts="skip", project=PROJECT, asset_root=IMAGERY_ROOT,
                       adopt=unet_imagery_upload[0], path_bound=100_000, limits=LIMITS, source=_source(UNET_IMG))
    return directory


def _imagery_fake(directory, unet_imagery_upload, **kwargs):
    fake = FakeEarthEngine(directory, Clock(), folders=(PARENT,), **kwargs)
    fake.assets.update(unet_imagery_upload[1])
    del fake.assets[f"{ROOT_ID}/input"]                    # the adopted upload's input image: never read
    return fake


def test_a_staged_imagery_plan_adopts_the_tables_and_exports_its_own_input(staged_imagery, unet_imagery_upload):
    from tensor_ee.converter.request import request_reads

    prepared = json.loads((staged_imagery / "prepared.json").read_text())
    plan = prepared["plan"]
    assert sorted(set(plan) - set(STAGED_KEYS)) == ["imagery"] and prepared["arguments"]["imagery"] == _source(
        UNET_IMG).to_json()
    assert plan["tables_root"] == ROOT_ID and plan["adopted"] and plan["input"]["asset"] == f"{IMAGERY_ROOT}/input"
    assert all(asset.startswith(f"{ROOT_ID}/weights_part_") for asset in plan["tables"]["assets"])
    assert [e["reads"] for e in plan["exports"]] == [["input"], ["stage01"], ["stage01", "stage02"]]
    assert plan["exports"][0]["grid"]["transform"] == [10.0, 0.0, 500000.0, 0.0, -10.0, 5000000.0]
    requests = {leaf: json.loads((staged_imagery / record["file"]).read_text())
                for leaf, record in prepared["requests"].items()}
    assert sorted(requests) == ["input", "logits", "stage01", "stage02"]
    assert request_reads(requests["input"]) == {"images": sorted(IMAGES), "tables": []}
    assert request_reads(requests["stage01"])["images"] == [f"{IMAGERY_ROOT}/input"]
    assert plan["input"]["float32_declared"] is True and plan["input"]["sha256"] == hashlib.sha256(json.dumps(
        dict(requests["input"], requestId=RUN.SIZING_REQUEST_ID)).encode()).hexdigest()
    assert _state(staged_imagery)["input"] == {"state": "PREPARED"}
    numerical = RUN.staged_plan(UNET_IMG, "skip", asset_root=IMAGERY_ROOT, adopt=unet_imagery_upload[0], limits=LIMITS)
    assert sorted(numerical) == STAGED_KEYS and numerical["input"]["asset"] == f"{ROOT_ID}/input"
    assert sorted(plan["calls"]) == ["check_input", "compare", "discover", "run"]
    assert list(numerical["calls"]) == ["run", "compare"]
    run, base = plan["calls"]["run"], numerical["calls"]["run"]
    assert run["exportImage"] == base["exportImage"] + 1 == 4 and run["getAsset"] == base["getAsset"] + 1
    assert run["getOperation"] == base["getOperation"] + RUN.INPUT_TASK_SECONDS // 10 + 2
    assert plan["calls"]["compare"] == {"computePixels": 1 + RUN.CALL_MARGIN}
    printed = RUN.staged_text(plan)
    assert "its input image is not read" in printed and all(image in printed for image in IMAGES)
    assert "run --until input" in printed and "Tasks: 4 image exports; assets: 4 images" in printed
    assert "Tasks: 3 image exports; assets: 3 images" in RUN.staged_text(numerical)


def test_adoption_with_an_imagery_input_takes_the_tables_only(unet_imagery_upload, tmp_path):
    upload, _ = unet_imagery_upload
    partial = tmp_path / "partial"                         # its tables row-checked, its input never saved
    shutil.copytree(upload, partial)
    state = _state(partial)
    state["upload"]["state"] = "ROWS_VERIFIED"
    del state["upload"]["input"]
    (partial / "state.json").write_text(json.dumps(state))
    assert RUN.adoption(UNET_IMG, partial, tables_only=True)["root"] == ROOT_ID
    with pytest.raises(ValueError, match="not verified with its input image"):
        RUN.adoption(UNET_IMG, partial)
    state["upload"]["state"] = "UPLOADING"
    (partial / "state.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="its tables are not uploaded and row-checked"):
        RUN.adoption(UNET_IMG, partial, tables_only=True)
    # an upload whose own input is imagery: its tables only, for a run given its own images and grid
    imagery = tmp_path / "imagery_upload"
    RUN.prepare(imagery, recipe=UNET_IMG, project=PROJECT, asset_root=ROOT_ID, source=_source(UNET_IMG))
    clock = Clock()
    _upload(imagery, FakeEarthEngine(imagery, clock), clock, recipe=UNET_IMG, source=_source(UNET_IMG))
    assert RUN.adoption(UNET_IMG, imagery, tables_only=True)["root"] == ROOT_ID
    with pytest.raises(ValueError, match="its input image is imagery; only its tables are adopted"):
        RUN.staged_plan(UNET_IMG, "skip", asset_root=IMAGERY_ROOT, adopt=imagery)


def test_a_staged_imagery_run_saves_its_input_waits_for_the_check_and_compares(staged_imagery, unet_imagery_upload):
    fake = _imagery_fake(staged_imagery, unet_imagery_upload)
    state = _run(staged_imagery, fake, fake.clock, until="input")
    assert state["outcome"] is None and state["adopt"]["state"] == "VERIFIED"
    assert state["input"]["state"] == "VERIFIED" and fake.exported == ["input"]
    tables = json.loads((staged_imagery / "prepared.json").read_text())["plan"]["tables"]["count"]
    # the folder levels (one created), each adopted table, then the input's destination: absent
    assert fake.calls[:3 + tables + 2] == ["getAsset", "getAsset", "createFolder"] + ["getAsset"] * (tables + 1) + [
        "exportImage"]
    calls = list(fake.calls)
    assert _run(staged_imagery, fake, fake.clock, until="input")["input"]["state"] == "VERIFIED"
    assert fake.calls == calls                                                             # nothing again
    with pytest.raises(RUN.RunStopped, match="no check record: run check-input; no model export is submitted"):
        _run(staged_imagery, fake, fake.clock)
    assert fake.exported == ["input"]
    assert "the input image VERIFIED, 0 of 3 exports verified" in RUN.report(staged_imagery)["outcome"]["text"]
    pixels = _pixels()
    fake.values = {"input": pixels}
    record = _check(staged_imagery, fake)
    assert record["passed"] and record["sha256"] == hashlib.sha256(pixels.tobytes()).hexdigest()
    state = _run(staged_imagery, fake, fake.clock)
    assert state["outcome"] == {"name": "completed", "export": None}
    assert fake.exported == ["input", "stage01", "stage02", "logits"] and "exportTable" not in fake.calls
    _caps_hold(staged_imagery, "run")
    reference = RUN.forward(UNET_IMG, pixels)
    fake.values["logits"] = reference
    with offline_earth_engine():
        compared = RUN.compare(staged_imagery, project=PROJECT, data=fake, ee_module=ee)
    assert compared["passed"] and compared["input_equals_local"] is None and compared["max_abs"] == 0.0
    assert fake.calls.count("computePixels") == 2
    report = RUN.report(staged_imagery)
    assert report["outcome"]["outcome"] == "completed" and "input imagery" in report["outcome"]["text"]
    assert report["imagery"]["images"] == list(IMAGES) and report["imagery"]["grid"]["crs"] == "EPSG:32601"
    assert report["input"]["state"] == "VERIFIED" and report["input"]["eecu_seconds"] == 0.5
    assert report["input_check"]["sha256"] == record["sha256"] and report["compare"]["input"]["source"] == "imagery"
    text = json.dumps(report)
    assert ROOT_ID not in text and IMAGERY_ROOT not in text and "operations/" not in text and "projects/" not in text


@pytest.mark.parametrize("behaviour, first, again", [
    ("lost", Crash, "its answer was lost; it is never submitted again"),
    ("error", RUN.RunStopped, "the input export failed and is never submitted again"),
    ("fail", RUN.RunStopped, "the input export failed and is never submitted again"),
])
def test_the_input_export_is_submitted_once_whatever_its_end(staged_imagery, unet_imagery_upload, behaviour, first,
                                                             again):
    fake = _imagery_fake(staged_imagery, unet_imagery_upload, exports={"input": behaviour})
    with pytest.raises(first):
        _run(staged_imagery, fake, fake.clock, until="input")
    assert _state(staged_imagery)["input"]["state"] == {"lost": "SUBMITTING"}.get(behaviour, "FAILED")
    for _ in range(2):
        with pytest.raises(RUN.RunStopped, match=again):
            _run(staged_imagery, fake, fake.clock)
    assert fake.exported == ["input"] and fake.calls.count("exportImage") == 1


def test_an_interrupted_input_export_resumes_without_a_second_submission(staged_imagery, unet_imagery_upload):
    fake = _imagery_fake(staged_imagery, unet_imagery_upload, exports={"input": "crash"})
    with pytest.raises(Crash):
        _run(staged_imagery, fake, fake.clock, until="input")
    assert _state(staged_imagery)["input"]["state"] == "SUBMITTED"
    state = _run(staged_imagery, fake, fake.clock, until="input")
    assert state["input"]["state"] == "VERIFIED" and fake.exported == ["input"]
    assert fake.calls.count("exportImage") == 1
    _caps_hold(staged_imagery, "run")


def test_an_existing_input_destination_stops_before_its_export(staged_imagery, unet_imagery_upload):
    fake = _imagery_fake(staged_imagery, unet_imagery_upload, extra={f"{IMAGERY_ROOT}/input": {"type": "IMAGE"}})
    with pytest.raises(RUN.RunStopped, match="exists; nothing is submitted"):
        _run(staged_imagery, fake, fake.clock, until="input")
    assert fake.exported == [] and _state(staged_imagery)["input"]["state"] == "PREPARED"


def test_the_input_exports_cost_counts_toward_the_staged_budget(staged_imagery, unet_imagery_upload):
    fake = _imagery_fake(staged_imagery, unet_imagery_upload)
    tight = RUN.Limits(budget_eecu_seconds=0.4, task_seconds=900, queue_seconds=1800, interval_seconds=10)
    state = _run(staged_imagery, fake, fake.clock, tight)
    assert state["outcome"] == {"name": "stage_stopped_at_limit", "export": "stage01", "limit": "budget"}
    assert fake.exported == ["input"] and state["exports"]["stage01"]["spent_before"] == 0.5


def test_until_input_is_for_a_staged_run_whose_input_is_imagery(staged, unet_upload, uploaded):
    fake = _staged_fake(staged, unet_upload)
    with pytest.raises(ValueError, match="--until takes 'input', for a staged run whose input is imagery"):
        _run(staged, fake, fake.clock, until="input")
    with pytest.raises(ValueError, match="--until is for a staged run whose input is imagery"):
        _run(uploaded, FakeEarthEngine(uploaded, Clock()), Clock(), until="input")
    assert fake.calls == [] and not (staged / "run.lock").exists()


# -- discover ------------------------------------------------------------------------------
class FakeLookup:
    """``computeValue`` only: what describing the day's images returns."""

    def __init__(self, described):
        self.described, self.expressions = described, []

    def computeValue(self, expression):
        self.expressions.append(json.dumps(ee.serializer.encode(expression, for_cloud_api=True)))
        return copy.deepcopy(self.described)


def test_discover_lists_the_days_images_in_one_request():
    pixel_type = {"type": "PixelType", "precision": "int", "min": 0, "max": 65535}
    described = {"type": "ImageCollection", "bands": [], "features": [
        {"type": "Image", "id": IMAGES[0], "version": 1,
         "bands": [{"id": "B4", "data_type": pixel_type, "dimensions": [100, 100], "crs": "EPSG:32601",
                    "crs_transform": [10, 0, 499980, 0, -10, 5000040]}],
         "properties": {"system:index": "MADE_UP_A", "PRODUCT_ID": "MADE_UP_PRODUCT", "MGRS_TILE": "00XXX",
                        "CLOUDY_PIXEL_PERCENTAGE": 1.5, "PROCESSING_BASELINE": "00.00", "ANOTHER": 1}},
        {"type": "Image", "bands": [{"id": "B4", "crs": "EPSG:32601", "crs_transform": [10, 0, 499985, 0, -10, 5000040]}],
         "properties": {"system:index": "MADE_UP_B"}},                                 # half a pixel off; few properties
        {"type": "Image", "id": f"{COLLECTION}/MADE_UP_C", "bands": [], "properties": {}}]}
    fake = FakeLookup(described)
    with offline_earth_engine():
        record = RUN.discover(IMG, "2099-01-01", data=fake, ee_module=ee, **GRID_ARGUMENTS)
    assert len(fake.expressions) == 1 and record["calls"] == {"caps": {"computeValue": 1}, "counts": {"computeValue": 1}}
    for text in (f'"{COLLECTION}"', "Filter.contains", "2099-01-01", '["B4"]', "Collection.limit", "500000.0"):
        assert text in fake.expressions[0], text
    assert (record["collection"], record["date"], record["band"], record["truncated"]) == (COLLECTION, "2099-01-01",
                                                                                         "B4", False)
    first, second, third = record["images"]
    assert first == {"id": IMAGES[0], "system:index": "MADE_UP_A", "PRODUCT_ID": "MADE_UP_PRODUCT", "MGRS_TILE": "00XXX",
                     "CLOUDY_PIXEL_PERCENTAGE": 1.5, "PROCESSING_BASELINE": "00.00", "band": "B4", "crs": "EPSG:32601",
                     "transform": [10, 0, 499980, 0, -10, 5000040], "data_type": pixel_type,
                     "chip_on_band_pixels": True}
    assert second["id"] == IMAGES[1] and second["PRODUCT_ID"] is None and second["PROCESSING_BASELINE"] is None
    assert second["chip_on_band_pixels"] is False and third["chip_on_band_pixels"] is None and third["crs"] is None
    for arguments, message in [((TINY, "2099-01-01"), "declares no imagery input convention"),
                               ((IMG, "2099-1-1"), "--date must be a day"), ((IMG, "2099-02-30"), "--date must be a day")]:
        with pytest.raises(ValueError, match=message):
            RUN.discover(*arguments, data=fake, ee_module=ee, **GRID_ARGUMENTS)
    assert len(fake.expressions) == 1                                                  # refused before any request


# -- map -----------------------------------------------------------------------------------
def test_the_map_draws_each_image_and_the_classes_offline(imagery_uploaded, uploaded):
    directory, fake, _ = imagery_uploaded
    with pytest.raises(ValueError, match="run check-input first"):
        RUN.chip_map(directory)
    pixels = _pixels()
    fake.values = {"input": pixels}
    _check(directory, fake)
    record = RUN.chip_map(directory)
    reference = RUN.forward(IMG, pixels)
    assert record["panels"] == [*IMAGES, "predicted classes"] and "PyTorch" in record["prediction"]
    assert record["class_counts"] == [int(n) for n in np.bincount(reference.argmax(-1).ravel(), minlength=3)]
    assert sum(record["class_counts"]) == record["pixels"] == 256 and "not an accuracy claim" in record["claim"]
    png = (directory / "map.png").read_bytes()
    assert png.startswith(b"\x89PNG\r\n\x1a\n") and struct.unpack(">II", png[16:24]) == (3 * 16, 16)
    rows = zlib.decompress(png[png.index(b"IDAT") + 4:png.index(b"IEND") - 8])
    first = np.frombuffer(rows[1:1 + 3 * 48], np.uint8).reshape(48, 3)                     # the first row of pixels

    def colour(values):
        return np.clip(values.astype(np.float64) * 255, 0, 255).astype(np.uint8)

    assert np.array_equal(first[0], colour(pixels[0, 0, :3])) and np.array_equal(first[16], colour(pixels[0, 0, 3:]))
    assert tuple(first[32]) == RUN.PALETTE[reference.argmax(-1)[0, 0]]
    np.save(directory / "fetched" / "logits.npy", np.roll(reference, 1, axis=-1))          # as compare keeps them
    other = RUN.chip_map(directory, directory / "other.png")
    assert "fetched by compare" in other["prediction"] and (directory / "other.png").exists()
    assert other["class_counts"] == [record["class_counts"][k] for k in (2, 0, 1)]
    with pytest.raises(ValueError, match="numerical chip, not imagery"):
        RUN.chip_map(uploaded)


def test_the_command_line_plans_an_imagery_input_offline(capsys):
    assert RUN.main(["plan", "--recipe", IMG, "--json", *COMMAND_LINE]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["imagery"]["images"] == list(IMAGES)
    assert record == json.loads(json.dumps(RUN.plan(IMG, source=_source())))
    assert RUN.main(["plan", "--recipe", UNET_IMG, "--cuts", "skip", *COMMAND_LINE]) == 0
    printed = capsys.readouterr().out
    assert "run --until input" in printed and all(image in printed for image in IMAGES)


class SignedIn:
    """What the command line's sign-in returns here: the ``ee`` module with a fake client as its data."""

    def __init__(self, data):
        self.data = data

    def __getattr__(self, name):
        return getattr(ee, name)


def test_the_command_line_runs_each_imagery_command_through_its_sign_in(unet_imagery_upload, tmp_path, monkeypatch,
                                                                        capsys):
    directory, clock, fakes, signed = tmp_path / "run", Clock(), {}, []
    run = RUN.run
    monkeypatch.setattr(RUN, "run", lambda path, **arguments: run(path, clock=clock, sleep=clock.sleep, **arguments))

    def command(arguments, lookup=None):
        """``main(arguments)``; the sign-in gives the offline signatures and a fake client, made on
        first use (the run directory is prepared before any sign-in)."""
        with contextlib.ExitStack() as stack:
            def initialize(project):
                signed.append(arguments[0])
                stack.enter_context(offline_earth_engine())
                if lookup is None and "client" not in fakes:
                    fakes["client"] = _imagery_fake(directory, unet_imagery_upload)
                    fakes["client"].clock = clock
                return SignedIn(lookup or fakes["client"])

            monkeypatch.setattr(RUN, "_initialize", initialize)
            code = RUN.main(arguments)
        return code, capsys.readouterr().out

    lookup = FakeLookup({"type": "ImageCollection", "features": [
        {"type": "Image", "id": IMAGES[0], "bands": [{"id": "B4", "crs": "EPSG:32601",
                                                      "crs_transform": [10, 0, 500000, 0, -10, 5000000]}],
         "properties": {"system:index": "MADE_UP_A"}}]})
    grid = COMMAND_LINE[3:]
    code, out = command(["discover", "--recipe", IMG, "--date", "2099-01-01", "--project", PROJECT, *grid,
                         "--output", str(tmp_path / "discover.json")], lookup)
    assert code == 0 and json.loads(out)["images"][0]["chip_on_band_pixels"] is True and len(lookup.expressions) == 1
    assert json.loads((tmp_path / "discover.json").read_text())["images"][0]["id"] == IMAGES[0]
    with pytest.raises(ValueError, match="declares no imagery input convention"):         # before any sign-in
        command(["discover", "--recipe", TINY, "--date", "2099-01-01", "--project", PROJECT, *grid], lookup)
    assert signed == ["discover"]
    limits = ["--budget-eecu-seconds", "600", "--task-seconds", "900", "--queue-seconds", "1800"]
    staged = ["run", "--run-dir", str(directory), "--project", PROJECT, "--cuts", "skip", "--recipe", UNET_IMG,
              "--asset-root", IMAGERY_ROOT, "--adopt", str(unet_imagery_upload[0]), "--path-bound", "100000", *limits,
              *COMMAND_LINE]
    code, out = command([*staged, "--until", "input"])
    assert code == 0 and json.loads(out) == {"outcome": None, "input": "VERIFIED"}
    fake = fakes["client"]
    assert fake.exported == ["input"]
    code, out = command(staged)                                                    # no check record yet
    assert code == 1 and out.startswith("STOPPED: the imagery input has no check record") and fake.exported == ["input"]
    with pytest.raises(ValueError, match="another input than these images and grid"):
        command([*staged[:-1], "20"])                                              # another pixel size
    pixels = _pixels()
    fake.values = {"input": pixels}
    code, out = command(["check-input", "--run-dir", str(directory), "--project", PROJECT])
    assert code == 0 and json.loads(out)["passed"] is True and fake.calls.count("computePixels") == 1
    code, out = command(["map", "--run-dir", str(directory)])
    assert code == 0 and "PyTorch" in json.loads(out)["prediction"] and (directory / "map.png").exists()
    code, out = command(["run", "--run-dir", str(directory), "--project", PROJECT, *limits])
    assert code == 0 and json.loads(out) == {"outcome": {"name": "completed", "export": None}, "input": "VERIFIED"}
    assert fake.exported == ["input", "stage01", "stage02", "logits"]
    fake.values["logits"] = RUN.forward(UNET_IMG, pixels)
    code, out = command(["compare", "--run-dir", str(directory), "--project", PROJECT])
    assert code == 0 and json.loads(out)["input"]["source"] == "imagery" and fake.calls.count("computePixels") == 2
    code, out = command(["report", "--run-dir", str(directory)])
    text = json.loads(out)["outcome"]["text"]
    assert code == 0 and "13.5 EECU-seconds over 3 exports; the input export 0.5 more; compare passed" in text
    assert signed == ["discover", "run", "run", "check-input", "run", "compare"]           # map and report: none


def test_the_command_line_refuses_what_it_can_before_any_sign_in(uploaded, staged, monkeypatch):
    def initialize(project):
        raise AssertionError("signed in")

    monkeypatch.setattr(RUN, "_initialize", initialize)
    limits = ["--budget-eecu-seconds", "600", "--task-seconds", "900", "--queue-seconds", "1800"]
    with pytest.raises(ValueError, match="numerical chip, not imagery"):
        RUN.main(["check-input", "--run-dir", str(uploaded), "--project", PROJECT])
    with pytest.raises(ValueError, match="--until input is for a staged run whose input is imagery"):
        RUN.main(["run", "--run-dir", str(staged), "--project", PROJECT, *limits, "--until", "input"])
    with pytest.raises(ValueError, match="declares no imagery input convention"):
        RUN.main(["run", "--run-dir", str(uploaded), "--project", PROJECT, *limits, *COMMAND_LINE])
    with pytest.raises(ValueError, match="--date must be a day"):
        RUN.main(["discover", "--recipe", IMG, "--date", "yesterday", "--project", PROJECT, *COMMAND_LINE[3:]])
