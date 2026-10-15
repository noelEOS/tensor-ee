"""C1: the capstone run tooling (tools/capstone.py), offline.

No network and no Earth Engine call: ``--plan`` runs under ``offline_earth_engine``, and the
live steps run against mocked ``ee.data`` calls inside ``ApiTestCase`` sessions. The
reference forward runs at batch 1 only; no NumPy evaluator or whole-model mirror runs.

- **R1.** The reference is SMP's own UPerNet at the capstone, seeded and checked (timm's
  shifts and windows, the explicit attention path, no dropout), and Milestone 12's
  ``expanded`` check passes.
- **R3, R4.** The upload records equal the state dict; the shards cover every record; the
  plan builds the export from a stand-in manifest through ``with_manifest``; the graph is
  deterministic from the seed and the manifest; the evaluated constants use the
  configuration and equal the configured constants' own counts.
- **R5, R7.** A mocked verification passes on the reference's own logits and fails on a
  logit one ulp past the tolerance, a one-ulp validity or mask value, a transposed output
  or a swapped class; the live steps refuse out of order; the polling cap stops polling.
- **R6.** The plan document equals ``--plan``.
"""
from __future__ import annotations

import contextlib
import importlib.util
from importlib.metadata import version as package_version
import json
import math
from pathlib import Path
import platform
import re
from unittest import mock

import ee
from ee.apitestcase import ApiTestCase
import numpy as np
import pytest
import torch

torch.set_num_threads(2)

ROOT = Path(__file__).resolve().parents[1]
HAVE_REFERENCES = all(importlib.util.find_spec(m) is not None for m in ("timm", "segmentation_models_pytorch"))
pytestmark = pytest.mark.skipif(not HAVE_REFERENCES, reason="the reference needs timm and SMP")

from tools import capstone as CAP  # noqa: E402
from tensor_ee import _swin_parameters as P  # noqa: E402

PLAN_DOC = ROOT / "docs" / "evidence" / "c1-capstone-plan.md"
PROJECT = re.search(r"tools/capstone\.py --plan --project ([\w-]+)", PLAN_DOC.read_text()).group(1)
# The public record replaced the original project ID by a placeholder (S03); the recorded request
# bytes were computed with the original. The request names the project once per asset reference,
# so its size is affine in the name's length; the test infers that length from the recorded bytes.
RECORDED_REQUEST_BYTES = int(re.search(r"The request is ([\d,]+) bytes", PLAN_DOC.read_text()).group(1).replace(",", ""))


@pytest.fixture(scope="module")
def models():
    return CAP.reference_model(torch.float32), CAP.reference_model(torch.float64)


@pytest.fixture(scope="module")
def plan(models):
    return CAP.make_plan(PROJECT, models=models)


@pytest.fixture(scope="module")
def local(models):
    maps = CAP.local_parameters(models[0])
    return maps, CAP.records_of(maps)


# ---------------------------------------------------------------------------
# R1: the reference
# ---------------------------------------------------------------------------
def test_the_reference_is_smps_upernet_at_the_capstone(models):
    m32, m64 = models
    assert type(m32).__name__ == "UPerNet" and not m32.training and not m64.training
    features = m32.encoder.model
    shifts = [[b.shift_size[0] for b in getattr(features, f"layers_{s}").blocks] for s in range(4)]
    assert shifts == CAP.CONFIG.block_shifts(8, 8) == [[0, 4], [0, 4], [0, 4], [0, 0]]
    assert all(b.window_size == (8, 8) and not b.attn.fused_attn
               for s in range(4) for b in getattr(features, f"layers_{s}").blocks)
    assert m32.decoder.fusion_block[0].in_channels == 256 and m32.segmentation_head[0].out_channels == 3
    assert type(m32.segmentation_head[1]).__name__ == "UpsamplingBilinear2d"
    assert m32.segmentation_head[1].scale_factor == 4 and m32.segmentation_head[1].align_corners
    assert sum(v.numel() for k, v in m32.state_dict().items() if not k.endswith("num_batches_tracked")) == 5_875_791
    # The same weights in both precisions.
    for (k, a), (_, b) in zip(m32.state_dict().items(), m64.state_dict().items()):
        assert torch.equal(a.double(), b.double()) if a.is_floating_point() else torch.equal(a, b), k


def test_the_seed_fixes_the_weights_and_every_parameter_is_exercised(models):
    again = CAP.reference_model(torch.float32)
    assert CAP.state_sha256(again.state_dict()) == CAP.state_sha256(models[0].state_dict())
    other = CAP.reference_model(torch.float32, seed=CAP.SEED + 7)
    assert CAP.state_sha256(other.state_dict()) != CAP.state_sha256(models[0].state_dict())
    state = models[0].state_dict()
    for name, value in state.items():
        if name.endswith("num_batches_tracked"):
            continue
        assert value.unique().numel() > 1, name                  # no constant tensor is left
    assert all(state[k].min() > 0 for k in state if k.endswith("running_var"))


def test_the_weights_need_no_platform_random_generator(models, monkeypatch):
    """``initial_state`` draws from the harness's integer hash only: with torch's and NumPy's
    random generators refused, the same weights come out (so CI gets them too)."""
    def refuse(*args, **kwargs):
        raise AssertionError("a platform random generator was used")

    model = models[0]
    for name in ("rand", "randn", "randint", "normal"):
        monkeypatch.setattr(torch, name, refuse)
    monkeypatch.setattr(np.random, "default_rng", refuse)
    values = CAP.initial_state(model)
    state = model.state_dict()
    assert set(values) == {k for k, v in state.items() if v.is_floating_point()}
    assert all(np.array_equal(values[k], state[k].numpy()) for k in values)


def test_the_weights_have_the_default_initializations_scale(models):
    """Each weight is uniform with the bound or the variance of timm's and SMP's initialization."""
    from fractions import Fraction

    state = {k: v.double().numpy() for k, v in models[0].state_dict().items() if v.is_floating_point()}
    checked = 0
    for name, value in state.items():
        shape = value.shape
        if name.startswith("encoder.") and len(shape) == 2:              # Linear weights, position tables
            bound = CAP._sqrt(CAP._std(Fraction(1, 50)))                  # timm: std 0.02
        elif name == "encoder.model.patch_embed.proj.weight":
            bound = CAP._sqrt(Fraction(1, 48))                            # PyTorch's Conv2d default
        elif name.startswith("decoder.") and len(shape) == 4:
            bound = CAP._sqrt(Fraction(6, int(np.prod(shape[1:]))))       # SMP's Kaiming uniform
        elif name == "segmentation_head.0.weight":
            bound = CAP._sqrt(Fraction(6, 64 + 3))                        # SMP's Xavier uniform
        else:
            continue
        assert np.abs(value).max() <= bound, name
        if value.size >= 2_000:
            assert abs(value.std() / (bound / 3 ** 0.5) - 1) < 0.05, name
        checked += 1
    assert checked > 50


def test_the_reference_check_and_precision(plan):
    r = plan["reference"]
    assert r["expanded_equals_forward"] and r["tensors_checked"] == 21
    assert r["float32_vs_float64_max_abs"] < CAP.TOLERANCE / 5
    assert -10 < r["logit_min"] < r["logit_max"] < 10


# ---------------------------------------------------------------------------
# R3: records, shards and the manifest
# ---------------------------------------------------------------------------
def test_the_upload_records_equal_the_state_dict(models, local):
    maps, records = local
    by_key = {r.key: r for r in records}
    _, locations = P.records_and_locations(*maps)
    state = models[0].state_dict()
    checked = 0
    for (group, name), (key, part, shape) in locations.items():
        if group == "patch_embed":
            source = f"encoder.model.patch_embed.{name}"
        elif group == "decoder":
            source = name
        else:
            source = "encoder.model." + P.encoder_prefix(group).replace("layers.", "layers_", 1) + name
        tensor = state[source].numpy()
        record = by_key[key]
        values = record.bias if part == "bias" else record.weights
        assert np.array_equal(values.reshape(-1), tensor.reshape(-1)), (group, name)
        checked += 1
    assert checked == sum(1 for k in state if not k.endswith("num_batches_tracked")) == 199 - 12
    assert len(records) == 153


def test_the_shards_cover_every_record(plan, local):
    _, records = local
    covered = {}
    for shard in plan["shards"]:
        for key in shard["layers"]:
            covered.setdefault(key, []).append(shard["index"])
    assert set(covered) == {r.key for r in records}
    rows = sum(r.shape[0] * math.ceil(r.shape[1] / 4096) for r in records)
    assert sum(s["rows"] for s in plan["shards"]) == rows == 16_411
    assert all(s["collection_payload_bytes"] <= 8_000_000 for s in plan["shards"])
    assert [s["asset_id"] for s in plan["shards"]] == [f"{plan['destinations']['weights']}_part_{i:04d}"
                                                       for i in range(plan["shard_count"])]


def test_the_names_carry_the_listing_hash(local):
    _, records = local
    names = CAP.names("some-project", records)
    assert names["weights"] == f"projects/some-project/assets/native_ee_capstone_c1_{CAP.listing_hash(records)[:12]}_w"
    assert names["result"].endswith("_logits") and names["result_description"] in names["result"]
    with pytest.raises(ValueError):
        CAP.names("a/b", records)


# ---------------------------------------------------------------------------
# R2, R4: the grid, the graph, the request and the constants
# ---------------------------------------------------------------------------
def test_the_grid_contains_the_m12_chip_at_a_whole_metre_origin(plan):
    assert plan["contains_m12_chip"] and CAP.GRID.transform[2] == 287520 and CAP.GRID.transform[5] == 2641000
    assert (CAP.GRID.width, CAP.GRID.height, CAP.GRID.crs) == (256, 256, "EPSG:32651")
    assert CAP.MODEL_GRID == CAP.CONFIG.model_grid(256, 256)


@pytest.fixture(scope="module")
def stand_in(models, local):
    maps, records = local
    digest = CAP.state_sha256(models[0].state_dict())
    destinations = CAP.names(PROJECT, records)
    with CAP.LF.offline_earth_engine():
        shards = CAP.shard_plan(records, destinations["weights"])
    return maps, records, digest, destinations, CAP.stand_in_manifest(records, shards, destinations["weights"], digest)


@pytest.fixture
def known(monkeypatch, models, stand_in):
    """The live steps' local work (the model, its records, the shard plan) from the fixtures,
    instead of being rebuilt at every call: they are tested above."""
    maps, records, digest, destinations, manifest = stand_in
    shards = [{k: v for k, v in s.items() if k != "status"} for s in manifest["shards"]]
    monkeypatch.setattr(CAP, "_local", lambda project: (models[0], digest, maps, records, destinations))
    monkeypatch.setattr(CAP, "shard_plan", lambda records_, namespace: [dict(s) for s in shards])
    monkeypatch.setattr(CAP.LF, "live_session", _session)
    return stand_in


def test_the_graph_is_deterministic_from_the_seed_and_the_manifest(plan, stand_in):
    maps, records, digest, destinations, manifest = stand_in
    shas = []
    with CAP.LF.offline_earth_engine() as attempts:
        for _ in range(2):
            server = P.with_manifest(*maps, manifest, digest)[:3]
            image, _ = CAP.model_graph(ee, server)
            shas.append(CAP.export_request(ee, image, destinations)["sha256"])
    assert shas[0] == shas[1] == plan["export"]["request_sha256"] and attempts == []
    # Another manifest (other asset IDs) gives another graph.
    moved = json.loads(json.dumps(manifest).replace("_part_0003", "_part_0099"))
    with CAP.LF.offline_earth_engine():
        image, _ = CAP.model_graph(ee, P.with_manifest(*maps, moved, digest)[:3])
        assert CAP.export_request(ee, image, destinations)["sha256"] != shas[0]


def test_the_export_request(plan):
    e = plan["export"]
    assert e["request_bytes"] < CAP.REQUEST_LIMIT and e["max_pixels"] == 65_536
    assert plan["task_count"] == plan["shard_count"] + 1 == 14 and plan["max_active_tasks"] == 2
    assert plan["tasks"][-1]["destination"] == plan["destinations"]["result"]
    assert plan["output_bands"] == ["logit_0", "logit_1", "logit_2", "valid"]


def test_the_evaluated_constants_use_the_configuration(plan):
    c = plan["evaluated_constants"]
    assert c["cross_checks_equal"] and all(a == b for a, b in c["cross_checks"].values())
    tables = c["parts"]["window_pool"] + c["parts"]["window_matrix_resize"] + c["parts"]["window_resize"]
    assert plan["formula_constants"]["total"] == c["total"] - tables       # the encoder's independent check
    assert c["total_with_those"] < c["cap"]
    # At window 7 (the cost functions' default) the count would differ: the configuration was used.
    from tensor_ee.blocks.window import normalization as WN
    assert WN.METADATA["layer_norm"].embedded_constants(48) != WN.METADATA["layer_norm"].embedded_constants(
        48, layout=CAP.CONFIG.layout)


def test_plan_makes_no_cloud_call(plan, capsys):
    assert plan["cloud_calls"] == 0
    text = CAP.plan_text(plan)
    assert text.endswith("Within limits.")


# ---------------------------------------------------------------------------
# R5, R7: verification, mocked
# ---------------------------------------------------------------------------
def _structured(values: np.ndarray, names) -> np.ndarray:
    out = np.zeros(values.shape[:2], dtype=[(n, "<f4") for n in names])
    for i, n in enumerate(names):
        out[n] = values[..., i]
    return out


def _metadata() -> dict:
    a, b, c, d, e, f = CAP.GRID.transform
    grid = {"crsCode": CAP.GRID.crs, "dimensions": {"width": 256, "height": 256},
            "affineTransform": {"scaleX": a, "translateX": c, "scaleY": e, "translateY": f}}
    return {"type": "IMAGE", "bands": [{"id": n, "dataType": {"precision": "FLOAT"}, "grid": grid}
                                       for n in CAP.OUTPUT_BANDS]}


@pytest.fixture(scope="module")
def answers(models):
    x = CAP.stand_in_input()
    _, logits = CAP.reference_record(x, models)
    ones = np.ones(x.shape[:2] + (1,), np.float32)
    inputs = _structured(np.concatenate([x, ones], -1), list(CAP.CONFIG.bands) + ["mask"])
    outputs = np.concatenate([logits, ones, ones], -1)
    return inputs, outputs, logits


def _verify(models, inputs, outputs):
    return CAP.verify_arrays(inputs, _structured(outputs, list(CAP.OUTPUT_BANDS) + ["mask"]), _metadata(),
                             models=models)


def test_a_verification_of_the_references_own_logits_passes(models, answers):
    inputs, outputs, _ = answers
    record = _verify(models, inputs, outputs)
    assert record["status"] == "passed", record["checks"]
    assert record["comparison"]["max_abs"] == 0 and record["comparison"]["argmax_agreement"] == 65_536
    assert record["comparison"]["bit_exact_values"] == record["comparison"]["values"] == 3 * 65_536
    assert record["reference"]["expanded_equals_forward"]


def test_the_tolerance_boundary_is_exact_to_one_ulp(models, answers):
    """The logit target is 1e-4: the largest float32 within it passes, the next fails."""
    inputs, outputs, logits = answers
    r = logits[100, 37, 1]
    up = np.float32(np.inf)
    inside = np.float32(np.float64(r) + CAP.TOLERANCE)                     # then the largest one within it
    while np.float64(inside) - np.float64(r) > CAP.TOLERANCE:
        inside = np.nextafter(inside, -up)
    while np.float64(np.nextafter(inside, up)) - np.float64(r) <= CAP.TOLERANCE:
        inside = np.nextafter(inside, up)
    passing, failing = outputs.copy(), outputs.copy()
    passing[100, 37, 1] = inside
    failing[100, 37, 1] = np.nextafter(inside, np.float32(np.inf))
    assert _verify(models, inputs, passing)["status"] == "passed"
    record = _verify(models, inputs, failing)
    assert record["status"] == "failed" and record["comparison"]["over_tolerance"] == 1
    assert record["comparison"]["worst"] == {"row": 100, "col": 37, "class": 1,
                                             "earth_engine": float(failing[100, 37, 1]), "reference": float(r)}


@pytest.mark.parametrize("band", [3, 4])
def test_a_one_ulp_validity_or_mask_value_fails(models, answers, band):
    inputs, outputs, _ = answers
    bad = outputs.copy()
    bad[5, 9, band] = np.nextafter(np.float32(1), np.float32(0))            # 0.99999994
    record = _verify(models, inputs, bad)
    assert record["status"] == "failed" and record["comparison"]["pass"]
    assert not record["checks"]["valid_band_all_one" if band == 3 else "output_unmasked"]


def test_a_transposed_output_or_a_swapped_class_fails(models, answers):
    inputs, outputs, _ = answers
    transposed = outputs.copy()
    transposed[..., :3] = outputs[..., :3].transpose(1, 0, 2)
    swapped = outputs.copy()
    swapped[..., [0, 1]] = outputs[..., [1, 0]]
    for bad in (transposed, swapped):
        record = _verify(models, inputs, bad)
        assert record["status"] == "failed"
        assert record["comparison"]["max_abs"] > 0.1 and record["comparison"]["argmax_agreement"] < 65_536


def test_a_wrong_grid_or_band_order_fails(models, answers):
    inputs, outputs, _ = answers
    metadata = _metadata()
    metadata["bands"] = metadata["bands"][::-1]
    record = CAP.verify_arrays(inputs, _structured(outputs, list(CAP.OUTPUT_BANDS) + ["mask"]), metadata,
                               models=models)
    assert record["status"] == "failed" and not record["checks"]["band_order_ok"]
    metadata = _metadata()
    metadata["bands"][0]["grid"]["affineTransform"]["translateX"] = 287530
    record = CAP.verify_arrays(inputs, _structured(outputs, list(CAP.OUTPUT_BANDS) + ["mask"]), metadata,
                               models=models)
    assert record["status"] == "failed" and not record["checks"]["grid_ok"]


# ---------------------------------------------------------------------------
# The live steps, mocked: order, refusals and the polling cap
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def _session(project, credentials="adc"):
    case = ApiTestCase()
    case.setUp()
    try:
        yield
    finally:
        case.tearDown()


def _counts(valid: int):
    return mock.patch.object(CAP, "input_counts",
                             return_value=mock.Mock(getInfo=lambda: {"pixels": 65_536, "valid": valid}))


def test_check_input_must_pass_before_any_upload(tmp_path, monkeypatch):
    monkeypatch.setattr(CAP.LF, "live_session", _session)
    with _counts(65_535):
        record = CAP.check_input("p", tmp_path)
    assert not record["passed"]
    with pytest.raises(ValueError, match="check-input"):
        CAP.upload("p", tmp_path)
    with _counts(65_536):
        assert CAP.check_input("p", tmp_path)["passed"]
    with CAP.LF.offline_earth_engine():                       # the real read: one reduceRegion
        image = CAP.input_counts(ee)
        assert "reduceRegion" in image.serialize() and CAP.SCENE.split("/")[-1] in image.serialize()


def test_upload_writes_the_manifest_and_records_a_failure(tmp_path, known):
    maps, records, digest, destinations, manifest = known
    state = CAP.load_state(tmp_path, PROJECT)
    state["steps"]["check_input"] = {"passed": True}
    CAP.save_state(tmp_path, state)

    calls = []

    def fake_export(records_, namespace, path, **kwargs):
        assert namespace == destinations["weights"] and kwargs["max_active"] == 2 and kwargs["poll_seconds"] == 30
        calls.append(kwargs["resume"])
        done = dict(manifest, status="ready")
        done.pop("manifest_hash"), done.pop("source_checkpoint_sha256")
        Path(path).write_text(json.dumps(done))
        return done

    with mock.patch.object(CAP, "export_weights", fake_export):
        record = CAP.upload(PROJECT, tmp_path)
        assert record == {"state": "ready", "shards": 13, "matches_plan": True, "unix": record["unix"]}
        written = json.loads((tmp_path / "manifest.json").read_text())
        assert written["manifest_hash"] == CAP.listing_hash(records) and written["source_checkpoint_sha256"] == digest
        assert CAP.upload(PROJECT, tmp_path) == record           # a completed upload is not repeated
    assert calls == [False]
    # A failing export is recorded, not left as UPLOADING; the next --upload resumes it.
    other = tmp_path / "other"
    CAP.save_state(other, state)
    with mock.patch.object(CAP, "export_weights", side_effect=CAP.PollingCap("cap")), pytest.raises(CAP.PollingCap):
        CAP.upload(PROJECT, other)
    assert CAP.load_state(other, PROJECT)["steps"]["upload"]["state"] == "POLLING_CAP"
    (other / "manifest.json").write_text(json.dumps(dict(manifest, status="uploading")))
    with mock.patch.object(CAP, "export_weights", fake_export):
        assert CAP.upload(PROJECT, other)["state"] == "ready"
    assert calls == [False, True]


def test_the_upload_cap_counts_from_each_invocation(tmp_path, known):
    """A resumed upload polls a task for up to 30 minutes from the resume, not from the task's
    start in an earlier run."""
    state = CAP.load_state(tmp_path, PROJECT)
    state["steps"]["check_input"] = {"passed": True}
    CAP.save_state(tmp_path, state)
    seen = []

    def fake_export(records_, namespace, path, *, on_update, **kwargs):
        old = {"shards": [{"asset_id": "a", "status": "RUNNING", "started_unix": 0.0}]}
        on_update(old)                                   # 10 h after its start, at the resume: polled on
        seen.append("polled")
        now[0] += CAP.POLL_CAP_SECONDS + 1
        on_update(old)                                   # 30 minutes into this invocation: the cap

    now = [36_000.0]
    with mock.patch.object(CAP, "export_weights", fake_export), pytest.raises(CAP.PollingCap):
        CAP.upload(PROJECT, tmp_path, clock=lambda: now[0])
    assert seen == ["polled"]


def test_prepare_submit_once_status_cap_and_verify(tmp_path, known, models, answers):
    maps, records, digest, destinations, manifest = known
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with mock.patch.object(CAP, "_local", return_value=(models[0], digest, maps, records, destinations)), \
            mock.patch.object(ee.data, "newTaskId", return_value=["request-1"]):
        prepared = CAP.prepare(PROJECT, tmp_path)
    assert prepared["state"] == "PREPARED" and prepared["destination"] == destinations["result"]
    assert prepared["request_id"] == "request-1"
    assert prepared["request_sha256"] == prepared["planned_in_session_sha256"]     # the pass criterion
    assert prepared["offline_plan_sha256"] and prepared["matches_document"]       # ApiTestCase here, so equal
    with pytest.raises(ValueError, match="not recorded COMPLETED"):
        CAP.verify(PROJECT, tmp_path, tmp_path / "evidence.json", models=models)
    # A lost response: --submit again re-sends the same request ID, and never makes a new one.
    operation = {"name": f"projects/{PROJECT}/operations/ABC123"}
    answers_ = iter([ConnectionError("lost response"), operation])

    def export_image(request_id, request):
        answer = next(answers_)
        if isinstance(answer, Exception):
            raise answer
        return answer

    with mock.patch.object(ee.data, "newTaskId", side_effect=AssertionError("a new request ID")), \
            mock.patch.object(ee.data, "exportImage", side_effect=export_image) as export:
        with pytest.raises(ConnectionError):
            CAP.submit(PROJECT, tmp_path)
        assert CAP.load_state(tmp_path, PROJECT)["steps"]["submit"]["state"] == "UNKNOWN"
        submitted = CAP.submit(PROJECT, tmp_path)
    assert submitted["state"] == "SUBMITTED" and submitted["task_id"] == "ABC123" and submitted["attempts"] == 2
    assert [call.args[0] for call in export.call_args_list] == ["request-1", "request-1"]
    with pytest.raises(ValueError, match="already SUBMITTED"):
        CAP.submit(PROJECT, tmp_path)
    with pytest.raises(ValueError, match="submission is recorded"):
        CAP.prepare(PROJECT, tmp_path)
    clock = iter(range(0, 10_000_000, 600))
    start = next(clock)
    state = CAP.load_state(tmp_path, PROJECT)
    state["steps"]["submit"]["unix"] = start
    CAP.save_state(tmp_path, state)
    with mock.patch.object(ee.data, "getTaskStatus", return_value=[{"state": "RUNNING"}]), \
            pytest.raises(CAP.PollingCap):
        CAP.status(PROJECT, tmp_path, sleep=lambda s: None, clock=lambda: next(clock))
    with mock.patch.object(ee.data, "getTaskStatus", return_value=[{"state": "COMPLETED", "id": "ABC123"}]):
        assert CAP.status(PROJECT, tmp_path, sleep=lambda s: None)["state"] == "COMPLETED"
    inputs, outputs, _ = answers
    responses = iter([inputs, _structured(outputs, list(CAP.OUTPUT_BANDS) + ["mask"])])
    with mock.patch.object(ee.data, "getAsset", return_value=_metadata()), \
            mock.patch.object(ee.data, "computePixels", side_effect=lambda params: next(responses)):
        record = CAP.verify(PROJECT, tmp_path, tmp_path / "evidence.json", models=models)
    assert record["status"] == "passed" and json.loads((tmp_path / "evidence.json").read_text())["status"] == "passed"


def test_submit_refuses_a_changed_request(tmp_path):
    state = CAP.load_state(tmp_path, "p")
    (tmp_path / "request.json").write_text("{}")
    state["steps"]["prepare"] = {"request_file": "request.json", "request_sha256": "0" * 64, "request_id": "r"}
    CAP.save_state(tmp_path, state)
    with pytest.raises(ValueError, match="changed"):
        CAP.submit("p", tmp_path)


def test_the_live_modes_need_a_state_outside_the_repository():
    with pytest.raises(SystemExit):
        CAP.main(["--check-input", "--project", "p"])
    with pytest.raises(SystemExit):
        CAP.main(["--check-input", "--project", "p", "--state", str(ROOT / "scratch")])


# ---------------------------------------------------------------------------
# R6: the plan document equals --plan
# ---------------------------------------------------------------------------
def _number(text: str) -> int:
    return int(text.replace(",", ""))


def test_the_plan_document_matches_the_plan(plan, models):
    text = PLAN_DOC.read_text()
    environment = re.search(r"earthengine-api (\S+), torch (\S+),", text)
    # The weights, their hashes and the asset names are the same everywhere (initial_state);
    # serialized sizes depend on the client, and float results on torch and the platform.
    same_client = package_version("earthengine-api") == environment.group(1)
    same_torch = (package_version("torch") == environment.group(2)
                  and platform.machine() in text and platform.system() in text)
    rows = re.findall(r"^\| (\d+) \| table \| `(native_unet_weights_part_\d{4})` \| `(\S+)` \| ([\d,]+) \| ([\d,]+) \|$",
                      text, re.M)
    assert len(rows) == plan["shard_count"] == 13
    for (number, description, destination, count, size), shard in zip(rows, plan["shards"]):
        assert int(number) == shard["index"] + 1 and description == shard["description"]
        assert _number(count) == shard["rows"] and f"projects/{PROJECT}/assets/{destination}" == shard["asset_id"]
        if same_client:
            assert _number(size) == shard["collection_payload_bytes"]
    for value in (plan["state_dict_sha256"], plan["manifest_hash"]):
        assert value in text
    c = plan["evaluated_constants"]
    for value in (c["total"], c["total_with_those"], *c["parts"].values(), plan["formula_constants"]["total"],
                  plan["formula_constants"]["weights"], plan["formula_constants"]["per_row_values"],
                  plan["parameters"]):
        assert f"{value:,}" in text, value
    assert f"{100 * c['fraction_of_cap']:.1f} %" in text
    for read in plan["reads"]:
        if "payload_bytes" in read:
            assert f"{read['payload_bytes']:,} bytes" in text
    assert f"{plan['task_count']} tasks" in text and f"{plan['records']} v2 weight records" in text
    if same_client:
        assert f"{plan['collection_payload_bytes']:,}" in text
        # The request names the project in its asset references, so its size depends on the
        # project name's length: a plan for a bare name of the original's length reproduces the
        # recorded bytes exactly. The recorded SHA-256 was computed with the original ID and is
        # the archive's; today's placeholder plan cannot reproduce it.
        # Bare lower-case names: the serializer orders strings, so a name's characters (not only
        # its length) can move the reference ids; a run of one letter sorts like the original.
        bare, longer = CAP.make_plan("a" * len(PROJECT), models=models), CAP.make_plan("a" * (len(PROJECT) + 1), models=models)
        per_character = longer["export"]["request_bytes"] - bare["export"]["request_bytes"]
        assert per_character > 0
        excess = bare["export"]["request_bytes"] - RECORDED_REQUEST_BYTES
        assert excess % per_character == 0
        original_length = len(PROJECT) - excess // per_character
        assert 0 < original_length < 64
        same_length = CAP.make_plan("a" * original_length, models=models)
        assert same_length["export"]["request_bytes"] == RECORDED_REQUEST_BYTES
        assert f"{same_length['export']['request_bytes']:,} bytes" in text
        assert f"{same_length['reads'][0]['request_bytes']:,}-byte expression" in text
        # The recorded SHA-256 was computed with the original ID: the document says so, and it is
        # the archive's, not today's placeholder plan's.
        assert "the recorded SHA-256 is the archive's" in text
        assert plan["export"]["request_sha256"] != same_length["export"]["request_sha256"]
    if same_torch:
        r = plan["reference"]
        assert f"{r['float32_vs_float64_max_abs']:.3g}".replace("e-0", "e-") in text
        assert f"{r['logit_min']:.4f}".replace("-", "−") in text and f"{r['logit_max']:.4f}" in text
    assert re.search(r"tools/capstone\.py --verify --project [\w-]+ --state", text)
