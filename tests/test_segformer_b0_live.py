"""Focused offline guard/runner failure contracts; no forwards or EE calls."""
from __future__ import annotations

import gc
import hashlib
import json
from types import SimpleNamespace
from unittest import mock
import uuid

import ee
from googleapiclient.http import HttpRequest
import numpy as np
import pytest

from tools import segformer_b0_live_guard as G
from tools import segformer_b0_validate as V

PROJECT = "test-project"
ASSET = f"projects/{PROJECT}/assets/segformer_test_input"


class Call:
    def __init__(self, kind="image", body=None, *, resource=None, method=None):
        methods = {v: k for k, v in G.METHODS.items()}
        self.methodId = methods[kind]
        suffix = {"image": "image:export", "table": "table:export", "pixels": "image:computePixels",
                  "value": "value:compute", "asset": "assets/segformer_test_input", "algorithms": "algorithms",
                  "status": "operations/operation_1"}[kind]
        self.uri = f"https://earthengine.googleapis.com/v1/{resource or f'projects/{PROJECT}/{suffix}'}?alt=json"
        self.method = method or ("POST" if kind in ("image", "table", "pixels", "value") else "GET")
        self.body = G.wire(body) if body is not None else None


def export_body(asset=ASSET, overwrite=False):
    return {"requestId": str(uuid.uuid4()), "description": "segformer_b0_input", "expression": {},
            "assetExportOptions": {"earthEngineDestination": {"name": asset, "overwrite": overwrite}}}


def guard_for(body=None, kind="image", caps=None):
    body = export_body() if body is None else body
    measured = G.fingerprint(body)
    approved = {kind: {measured["sha256"]: {"label": "prepared", "request_id": body.get("requestId")}}}
    return G.Guard(PROJECT, approved, {ASSET}, caps=caps or dict(G.CAPS)), body


def test_guard_accepts_exact_body_and_refuses_replay():
    guard, body = guard_for()
    guard.authorize(Call(body=body))
    assert guard.counts["batch"] == guard.counts["image"] == 1
    with pytest.raises(G.RefusedLiveCall, match="already"):
        guard.authorize(Call(body=body))


@pytest.mark.parametrize("change", ["body", "project", "host", "scheme", "resource", "verb", "query", "method"])
def test_guard_rejects_actual_request_drift_before_send(change):
    guard, body = guard_for()
    call = Call(body=body)
    if change == "body":
        call.body = call.body + b" "
    elif change == "project":
        call.uri = call.uri.replace(PROJECT, "other-project")
    elif change == "host":
        call.uri = call.uri.replace("earthengine.googleapis.com", "example.com")
    elif change == "scheme":
        call.uri = call.uri.replace("https:", "http:")
    elif change == "resource":
        call.uri = call.uri.replace("image:export", "table:export")
    elif change == "verb":
        call.method = "GET"
    elif change == "query":
        call.uri += "&fields=secret"
    else:
        call.methodId = "earthengine.projects.assets.delete"
    with pytest.raises(G.RefusedLiveCall):
        guard.authorize(call)
    assert guard.counts["batch"] == 0


def test_nested_overwrite_refused_even_with_allowlisted_body():
    guard, body = guard_for(export_body(overwrite=True))
    with pytest.raises(G.RefusedLiveCall, match="overwrite"):
        guard.authorize(Call(body=body))


@pytest.mark.parametrize("kind", list(G.CAPS))
def test_every_guard_cap_can_only_lower_and_is_enforced(kind):
    caps = dict(G.CAPS)
    caps[kind] = 0
    actual_kind = "image" if kind == "batch" else kind
    body = export_body() if actual_kind in ("table", "image") else {"expression": {}}
    guard, body = guard_for(body, actual_kind, caps)
    guard.initializing = True
    if actual_kind == "status":
        guard.add_operation(f"projects/{PROJECT}/operations/operation_1")
    with pytest.raises(G.BudgetStop):
        guard.authorize(Call(actual_kind, body if actual_kind in ("table", "image", "pixels", "value") else None))
    larger = dict(G.CAPS)
    larger[kind] += 1
    with pytest.raises(ValueError):
        G.Guard(PROJECT, {}, {ASSET}, caps=larger)


def test_status_and_assets_require_explicit_exact_names():
    guard, _ = guard_for()
    with pytest.raises(G.RefusedLiveCall, match="recorded"):
        guard.authorize(Call("status"))
    with pytest.raises(G.RefusedLiveCall, match="asset"):
        guard.authorize(Call("asset", resource=f"projects/{PROJECT}/assets/other"))
    guard.add_operation(f"projects/{PROJECT}/operations/operation_1")
    guard.authorize(Call("status"))
    with pytest.raises(G.RefusedLiveCall):
        guard.add_operation(None)


def test_weaksets_do_not_confuse_reused_request_object_ids():
    guard, _ = guard_for()
    for _ in range(20):
        call = Call("asset")
        guard.authorize(call)
        guard._executed.add(call)
        del call
        gc.collect()
        assert not guard._authorized and not guard._executed
    assert guard.counts["asset"] == 20


def test_public_get_task_status_and_download_bypasses_are_blocked():
    guard, _ = guard_for()
    with guard.installed():
        for name in ("getTaskStatus", "getDownloadId", "deleteAsset", "computePixels", "computeValue"):
            with pytest.raises(G.RefusedLiveCall, match=name):
                getattr(ee.data, name)({})


def test_direct_http_execute_cannot_bypass_method_guard(monkeypatch):
    guard, _ = guard_for()
    sends = []
    monkeypatch.setattr(HttpRequest, "execute", lambda request, **kwargs: sends.append(kwargs) or {})
    request = HttpRequest(None, None, "https://earthengine.googleapis.com/v1/projects/test-project/assets/x",
                          method="DELETE", methodId="earthengine.projects.assets.delete")
    with guard.installed(), pytest.raises(G.RefusedLiveCall):
        request.execute(num_retries=5)
    assert not sends


def test_transport_and_http_execute_force_retry_zero(monkeypatch):
    guard, body = guard_for()
    retries = []
    monkeypatch.setattr(HttpRequest, "execute", lambda request, **kwargs: retries.append(kwargs["num_retries"]) or {})
    monkeypatch.setattr(ee.data, "_execute_cloud_call", lambda call, num_retries=None: call.execute(num_retries=num_retries))
    call = HttpRequest(None, None, f"https://earthengine.googleapis.com/v1/projects/{PROJECT}/image:export",
                       method="POST", body=G.wire(body), methodId="earthengine.projects.image.export")
    with guard.installed():
        ee.data._execute_cloud_call(call, num_retries=9)
        with pytest.raises(G.RefusedLiveCall):
            call.execute(num_retries=9)
    assert retries == [0]
    assert guard.counts["image"] == 1
    assert guard.record()["transport_behavior"]["physical_http_attempts"].startswith("not instrumented")


def test_global_time_limit_stops_before_cloud_call(monkeypatch):
    guard, body = guard_for()
    monkeypatch.setattr(G.time, "monotonic", lambda: guard.started + 7201)
    with pytest.raises(G.BudgetStop):
        guard.authorize(Call(body=body))


def execution(tmp_path):
    body = export_body()
    relative = "requests/export_input.json"
    V.save_json(tmp_path / relative, body)
    request = {"label": "export_input", "category": "image", "request_id": body["requestId"], "asset": ASSET,
               "file": relative, "file_sha256": V.sha_file(tmp_path / relative), "wire": G.fingerprint(body), "output": "input"}
    V.save_json(tmp_path / "state.json", {"state": "PREPARED", "tasks": {}, "checks": {}, "ledger": {}})
    return V.Execution(tmp_path, {"project": PROJECT, "requests": [request]})


@pytest.mark.parametrize("result", ["exception", "missing-name", "wrong-project"])
def test_submission_intent_is_durable_before_ambiguous_outcome_and_never_retried(tmp_path, monkeypatch, result):
    runner = execution(tmp_path)
    calls = []
    def send(rid, body):
        saved = V.read_json(tmp_path / "state.json")["tasks"]["export_input"]
        assert saved["state"] == "SUBMITTING" and saved["request_id"] == rid
        calls.append(rid)
        if result == "exception":
            raise RuntimeError("response lost")
        return {} if result == "missing-name" else {"name": "projects/other/operations/x"}
    monkeypatch.setattr(ee.data, "exportImage", send)
    with pytest.raises((RuntimeError, G.RefusedLiveCall)):
        runner.submit("export_input")
    assert V.read_json(tmp_path / "state.json")["tasks"]["export_input"]["state"] == "UNKNOWN"
    saved = V.read_json(tmp_path / "state.json")["tasks"]["export_input"]
    if result != "exception":
        assert saved["response_received"]
        assert saved["operation"] == ({} if result == "missing-name" else {"name": "projects/other/operations/x"})
    with pytest.raises(ValueError, match="already"):
        runner.submit("export_input")
    assert len(calls) == 1


def test_request_tampering_fails_before_submission(tmp_path, monkeypatch):
    runner = execution(tmp_path)
    with (tmp_path / "requests/export_input.json").open("a") as stream:
        stream.write(" ")
    sends = []
    monkeypatch.setattr(ee.data, "exportImage", lambda *a: sends.append(a))
    with pytest.raises(ValueError, match="changed"):
        runner.submit("export_input")
    assert not sends and not runner.state["tasks"]


def test_concurrency_is_enforced_before_new_submission(tmp_path):
    runner = execution(tmp_path)
    runner.state["tasks"]["other"] = {"state": "RUNNING", "kind": "table"}
    with pytest.raises(ValueError, match="concurrency"):
        runner.submit("export_input")


def test_poll_uses_recorded_operation_and30second_spacing(tmp_path, monkeypatch):
    runner = execution(tmp_path)
    now = [0.0]
    runner.clock = lambda: now[0]
    runner.sleep = lambda delay: now.__setitem__(0, now[0] + delay)
    operation = f"projects/{PROJECT}/operations/operation_1"
    runner.state["tasks"]["job"] = {"state": "SUBMITTED", "submitted_unix": 0., "operation_name": operation}
    observed = []
    monkeypatch.setattr(ee.data, "getOperation", lambda name: observed.append((name, now[0])) or {"name": name, "done": True})
    runner.poll(["job"])
    assert observed == [(operation, 30.)]
    assert runner.state["tasks"]["job"]["state"] == "COMPLETED"


def test_poll_timeout_retains_task_identity(tmp_path, monkeypatch):
    runner = execution(tmp_path)
    runner.clock = lambda: 1800.
    runner.state["tasks"]["job"] = {"state": "SUBMITTED", "submitted_unix": 0.,
        "operation_name": f"projects/{PROJECT}/operations/operation_1"}
    monkeypatch.setattr(ee.data, "getOperation", lambda name: pytest.fail("expired task polled"))
    with pytest.raises(G.BudgetStop):
        runner.poll(["job"])
    assert V.read_json(tmp_path / "state.json")["tasks"]["job"]["state"] == "POLLING_CAP"


def pixel_spec():
    return {"body": {"grid": {"dimensions": {"height": 2, "width": 2}}}, "bands": ["c0", "c1", "valid"]}


def structured():
    data = np.ones((2, 2), dtype=[("c0", "<f4"), ("c1", "<f4"), ("valid", "<f4")])
    return data


@pytest.mark.parametrize("wrong", ["dtype", "bands", "shape", "mask", "nonfinite", "mask-nonfinite"])
def test_readbacks_refuse_partial_invalid_or_retyped_pixels(wrong):
    pixels = structured()
    if wrong == "dtype":
        pixels = pixels.astype([(n, "<f8") for n in pixels.dtype.names])
    elif wrong == "bands":
        pixels = pixels[["c1", "c0", "valid"]]
    elif wrong == "shape":
        pixels = pixels[:1]
    elif wrong == "mask":
        pixels["valid"][0, 0] = 0
    elif wrong == "nonfinite":
        pixels["c0"][0, 0] = np.nan
    else:
        pixels["valid"][0, 0] = np.nan
    with pytest.raises(V.ValidationFailure):
        V.read_pixels(pixels, pixel_spec())


def test_exact_gate_preserves_signed_zero_and_absolute_target():
    expected = np.zeros((1, 1, 2), np.float32)
    actual = expected.copy()
    actual[0, 0, 0] = -0.0
    with pytest.raises(V.ValidationFailure, match="transport"):
        V.compare(actual, expected, exact=True)
    actual[0, 0, 0] = np.nextafter(np.float32(1e-4), np.float32(np.inf))
    with pytest.raises(V.ValidationFailure) as failure:
        V.compare(actual, expected)
    assert failure.value.details["max_abs"] > 1e-4 and failure.value.details["values"] == 2


def test_numerical_evidence_is_persisted_on_failure(tmp_path):
    runner = execution(tmp_path)
    bad = np.ones((1, 1, 1), np.float32)
    with pytest.raises(V.ValidationFailure):
        runner.numerical_check("C1", "producer_accumulated", lambda: V.compare(bad, np.zeros_like(bad)))
    saved = V.read_json(tmp_path / "state.json")["checks"]["numerical"]["C1"]["producer_accumulated"]
    assert not saved["passed"] and saved["details"]["max_abs"] == 1


def test_changed_approval_record_refused_before_model_or_ee(tmp_path, monkeypatch):
    directory = V.ROOT / "release_runs" / "unit-never-created"
    plan = tmp_path / "plan.json"
    V.save_json(plan, {"weight_shards": 9, "proposed_primary_assets_and_batch_submissions": 16})
    monkeypatch.setattr(ee, "Initialize", lambda *a, **k: pytest.fail("EE initialization"))
    with pytest.raises(ValueError, match="untouched"):
        V.prepare(directory, plan)
    assert not directory.exists()


def image_metadata():
    affine = {"scaleX": 2.5, "shearX": 0, "translateX": 1000, "shearY": 0, "scaleY": -2.5, "translateY": 2000}
    grid = {"crsCode": "EPSG:3857", "dimensions": {"height": 2, "width": 2}, "affineTransform": affine}
    metadata = {"name": ASSET, "type": "IMAGE", "bands": [{"id": n, "dataType": {"precision": "FLOAT"}, "grid": grid} for n in ("b0", "b1")]}
    return metadata, {"bands": ["b0", "b1"], "grid": grid}


@pytest.mark.parametrize("wrong", ["order", "dtype", "grid", "affine"])
def test_asset_metadata_checks_every_band_grid(wrong):
    metadata, spec = image_metadata()
    metadata = json.loads(json.dumps(metadata))
    if wrong == "order":
        metadata["bands"].reverse()
    elif wrong == "dtype":
        metadata["bands"][1]["dataType"]["precision"] = "DOUBLE"
    elif wrong == "grid":
        metadata["bands"][1]["grid"]["dimensions"]["width"] = 3
    else:
        metadata["bands"][1]["grid"]["affineTransform"]["translateX"] += 1
    with pytest.raises(V.ValidationFailure):
        V.check_image_metadata(metadata, spec)


def test_computed_projection_check_accepts_api_wkt_boolean_not_affine_list(tmp_path, monkeypatch):
    runner = execution(tmp_path)
    metadata, spec = image_metadata()
    runner.prepared["image_specs"] = {"input": spec}
    monkeypatch.setattr(ee.data, "getAsset", lambda name: metadata)
    runner.value = lambda label: {"bands": ["b0", "b1"], "types": {n: {"precision": "float"} for n in ("b0", "b1")},
                                  "crs": "EPSG:3857", "projection_matches": True}
    assert runner.completed_metadata("export_input")["metadata"]["projection_matches"]
    runner.value = lambda label: {"bands": ["b0", "b1"], "types": {n: {"precision": "float"} for n in ("b0", "b1")},
                                  "crs": "EPSG:3857", "transform": "PROJCS[WKT string]"}
    with pytest.raises(V.ValidationFailure):
        runner.completed_metadata("export_input")


def cached_proof():
    pytest.importorskip("segmentation_models_pytorch")      # V.versions() records SMP's version
    from tools.fixtures.segformer_b0_fullmodel import state_sha256, array_provenance
    state = {"dummy": np.ones(1, np.float32)}
    refs = {name: np.zeros(shape, np.float32) for name, shape in V.SHAPES.items()}
    sources = ("tools/fixtures/segformer_b0_validation.py", "tools/fixtures/segformer_b0_fullmodel.py",
        "tests/reference/segformer_b0/full_model.py", "tests/reference/segformer_b0/model.py", "tests/reference/segformer_b0/blocks.py")
    report = {"passed": True, "expanded_traversal_bitwise_equal": True, "isolated_on_original_boundaries_bitwise_equal": True,
        "state_and_input_unchanged_after_checks": True, "cloud_calls": 0, "target": 1e-4,
        "state_sha256": state_sha256(state), "input_provenance": array_provenance({"input": refs["input"]}),
        "reference_provenance": array_provenance(refs), "versions": V.versions(),
        "source_files_sha256": {p: V.sha_file(V.ROOT / p) for p in sources}}
    metric = lambda name: {"passed": True, "max_abs": 0., "mae": 0., "rmse": 0., "over_target": 0, "nonfinite": 0,
                           "values": int(np.prod(V.SHAPES[name]))}
    report["full"] = {n: metric(n) for n in V.OUTPUTS}
    report["isolated"] = {cut: metric(name) for cut, name in zip(V.CUTS, V.OUTPUTS)}
    return state, refs, report


@pytest.mark.parametrize("wrong", ["empty-sources", "missing-source", "ref-bytes", "isolated-failure", "count", "target", "nonfinite", "nan-metric"])
def test_cached_reference_proof_rejects_missing_or_tampered_evidence(wrong):
    state, refs, report = cached_proof()
    V._cache_proof(state, refs, report)
    if wrong == "empty-sources":
        report["source_files_sha256"] = {}
    elif wrong == "missing-source":
        report["source_files_sha256"].pop("tests/reference/segformer_b0/blocks.py")
    elif wrong == "ref-bytes":
        refs["C1"][0, 0, 0] = 1
    elif wrong == "isolated-failure":
        report["isolated"]["stage1a"]["passed"] = False
    elif wrong == "count":
        report["full"]["C4"]["values"] -= 1
    elif wrong == "target":
        report["target"] = 1e-3
    elif wrong == "nonfinite":
        report["isolated"]["decoder"]["nonfinite"] = 1
    else:
        report["full"]["C1"]["max_abs"] = float("nan")
    with pytest.raises(ValueError):
        V._cache_proof(state, refs, report)


def test_atomic_write_fsyncs_file_and_parent_directory(tmp_path, monkeypatch):
    original = V.os.fsync
    seen = []
    monkeypatch.setattr(V.os, "fsync", lambda fd: seen.append(fd) or original(fd))
    V.save_json(tmp_path / "state.json", {"state": "SUBMITTING"})
    assert len(seen) == 2


def test_post_transport_global_deadline_retains_known_export_operation(tmp_path, monkeypatch):
    runner = execution(tmp_path)
    name = f"projects/{PROJECT}/operations/operation_1"
    def returned_after_limit(*args):
        error = G.BudgetStop("deadline reached after response")
        error.returned_result = {"name": name}
        raise error
    monkeypatch.setattr(ee.data, "exportImage", returned_after_limit)
    with pytest.raises(G.BudgetStop):
        runner.submit("export_input")
    task = V.read_json(tmp_path / "state.json")["tasks"]["export_input"]
    assert task["state"] == "SUBMITTED" and task["operation_name"] == name


def test_post_transport_checks_global_deadline(monkeypatch):
    guard, body = guard_for()
    current = [guard.started]
    monkeypatch.setattr(G.time, "monotonic", lambda: current[0])
    def slow_transport(call, num_retries=None):
        current[0] += 7201
        return {"name": f"projects/{PROJECT}/operations/operation_1"}
    monkeypatch.setattr(ee.data, "_execute_cloud_call", slow_transport)
    with guard.installed(), pytest.raises(G.BudgetStop) as error:
        ee.data._execute_cloud_call(Call(body=body))
    assert error.value.returned_result["name"].endswith("operation_1")


def test_returned_status_after_task_deadline_cannot_be_accepted_completed(tmp_path, monkeypatch):
    runner = execution(tmp_path)
    now = [0.]
    runner.clock = lambda: now[0]
    runner.sleep = lambda delay: now.__setitem__(0, now[0] + delay)
    name = f"projects/{PROJECT}/operations/operation_1"
    runner.state["tasks"]["job"] = {"state": "SUBMITTED", "submitted_unix": 0., "operation_name": name}
    def slow_status(operation):
        now[0] = 1801.
        return {"name": operation, "done": True}
    monkeypatch.setattr(ee.data, "getOperation", slow_status)
    with pytest.raises(G.BudgetStop):
        runner.poll(["job"])
    assert runner.state["tasks"]["job"]["state"] == "POLLING_CAP"
    assert runner.state["tasks"]["job"]["last_operation"]["done"]


def test_http_timeout_is_bounded_and_restored_without_sdk_rebuild(monkeypatch):
    guard, body = guard_for()
    guard.request_seconds = 12.
    http = SimpleNamespace(http=SimpleNamespace(_timeout=None))
    seen = []
    monkeypatch.setattr(HttpRequest, "execute", lambda call, **kwargs: seen.append(call.http.http._timeout) or {})
    request = HttpRequest(http, None, f"https://earthengine.googleapis.com/v1/projects/{PROJECT}/image:export",
                          method="POST", body=G.wire(body), methodId="earthengine.projects.image.export")
    monkeypatch.setattr(ee.data, "setDeadline", lambda *a, **k: pytest.fail("SDK discovery rebuilt"))
    with guard.installed():
        request.execute()
    assert seen == [12.] and http.http._timeout is None


@pytest.mark.parametrize("response", [None, {}, {"name": "wrong"}])
def test_collision_absence_requires_actual404(tmp_path, monkeypatch, response):
    runner = execution(tmp_path)
    monkeypatch.setattr(ee.data, "getAsset", lambda asset: response)
    with pytest.raises(V.ValidationFailure):
        runner.collision_checks()


def test_completion_metadata_requires_exact_asset_name(tmp_path, monkeypatch):
    runner = execution(tmp_path)
    metadata, spec = image_metadata()
    runner.prepared["image_specs"] = {"input": spec}
    metadata["name"] = ASSET + "_wrong"
    monkeypatch.setattr(ee.data, "getAsset", lambda asset: metadata)
    with pytest.raises(V.ValidationFailure, match="destination"):
        runner.completed_metadata("export_input")


def test_wrong_status_response_is_saved_before_identity_refusal(tmp_path, monkeypatch):
    runner = execution(tmp_path)
    now = [0.]
    runner.clock = lambda: now[0]
    runner.sleep = lambda delay: now.__setitem__(0, now[0] + delay)
    runner.state["tasks"]["job"] = {"state": "SUBMITTED", "submitted_unix": 0.,
        "operation_name": f"projects/{PROJECT}/operations/operation_1"}
    wrong = {"name": "projects/other/operations/supplied", "done": True}
    monkeypatch.setattr(ee.data, "getOperation", lambda name: wrong)
    with pytest.raises(V.ValidationFailure, match="identity"):
        runner.poll(["job"])
    assert V.read_json(tmp_path / "state.json")["tasks"]["job"]["last_operation"] == wrong


def test_mocked_complete_lifecycle_checks_metadata_and_actual_incoming_boundaries(tmp_path, monkeypatch):
    """Exercise sequence/census with array stand-ins; no model or cloud execution."""
    from tools.fixtures import segformer_b0_validation as R
    from reference.segformer_b0 import full_model as M
    state = {"dummy": np.ones(1, np.float32)}
    full = {name: np.zeros(shape, np.float32) for name, shape in V.SHAPES.items()}
    actual = {"input": full["input"].copy()}
    for i, name in enumerate(V.OUTPUTS, 1):
        actual[name] = np.full(V.SHAPES[name], np.float32(i * 1e-7), np.float32)
    np.savez(tmp_path / "state.npz", **state)
    np.savez(tmp_path / "references.npz", **full)
    V.save_json(tmp_path / "prepared.json", {"test": "immutable stand-in"})
    requests = []
    for i in range(9):
        requests.append({"label": f"weight_{i:04d}", "category": "table", "asset": ASSET + f"_w{i}",
                         "wire": {"sha256": f"table{i}"}})
    for name in ("input",) + V.OUTPUTS:
        requests.append({"label": f"export_{name}", "category": "image", "asset": ASSET + "_" + name,
                         "wire": {"sha256": name}})
    V.save_json(tmp_path / "state.json", {"state": "PREPARED", "tasks": {}, "checks": {}, "ledger": {}})
    runner = V.Execution(tmp_path, {"project": PROJECT, "requests": requests, "state_sha256": "state", "input_sha256": "input"})
    checked, seen_inputs = set(), []
    def collisions():
        runner.guard.counts["asset"] += 16
    def submit(label):
        kind = "table" if label.startswith("weight_") else "image"
        runner.guard.counts[kind] += 1
        runner.guard.counts["batch"] += 1
        runner.state["tasks"][label] = {"state": "SUBMITTED", "operation_name": f"projects/{PROJECT}/operations/{label}"}
        return label
    def poll(labels):
        for label in labels:
            runner.state["tasks"][label]["state"] = "COMPLETED"
            runner.guard.counts["status"] += 1
        return labels
    def metadata(label):
        assert runner.state["tasks"][label]["state"] == "COMPLETED"
        runner.guard.counts["asset"] += 1
        runner.guard.counts["value"] += 1
        if label.startswith("export_"):
            checked.add(label[len("export_"):])
    def pixels(label):
        runner.guard.counts["pixels"] += 1
        source, name = label.split("_", 1)
        if source == "saved":
            assert name in checked
        else:
            cut = V.CUTS[V.OUTPUTS.index(name)]
            assert set(M.CUT_INPUTS[cut]) <= checked
        return actual[name].copy()
    def coefficients(source):
        assert "input" in checked and all(t["state"] == "COMPLETED" for t in runner.state["tasks"].values())
        runner.guard.counts["pixels"] += 192
        runner.state["checks"]["coefficients"] = {f"source{i}": {"differing_bits": 0} for i in range(191)}
    def reference(cut, source, incoming):
        assert set(incoming) == set(M.CUT_INPUTS[cut])
        for name, values in incoming.items():
            np.testing.assert_array_equal(values, actual[name])
        seen_inputs.append(cut)
        return actual[V.OUTPUTS[V.CUTS.index(cut)]].copy()
    monkeypatch.setattr(R, "isolated_reference", reference)
    monkeypatch.setattr(M, "mirror_cut", lambda cut, source, incoming: actual[V.OUTPUTS[V.CUTS.index(cut)]].copy())
    runner.collision_checks, runner.submit, runner.poll = collisions, submit, poll
    runner.completed_metadata, runner.pixels, runner.coefficients = metadata, pixels, coefficients
    runner.prepared["planned_manifest"] = {"shards": [{"index": i} for i in range(9)]}
    for request in requests:
        request["request_id"] = str(uuid.uuid4())
    result = runner.execute()
    assert result["state"] == "PASSED" and seen_inputs == list(V.CUTS)
    assert runner.guard.counts == dict(G.CAPS) | {"algorithms": 0, "status": 16}
    assert set(result["checks"]["numerical"]) == set(V.OUTPUTS)
    for name in V.OUTPUTS[:-1]:
        assert result["checks"]["numerical"][name]["producer_saved_transport"]["differing_bits"] == 0
        assert result["checks"]["numerical"][name]["saved_accumulated"]["max_abs"] <= 1e-4
