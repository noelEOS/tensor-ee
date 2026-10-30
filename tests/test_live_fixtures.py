"""P00: the live fixture harness (``tools/live_fixtures.py``), offline.

No network and no Earth Engine call: graphs are built against the offline signatures of
``ee.apitestcase.ApiTestCase``. ``--run`` is exercised with ``computePixels`` replaced by a
fake that answers from the mirrors, and once through the real ``ee.data.computePixels``
with a Cloud API resource built from earthengine-api's bundled test discovery document and a
fake HTTP transport. Tests that need timm or SMP (the references) skip without them.
"""
from __future__ import annotations

import contextlib
import dataclasses
import hashlib
from importlib.metadata import version as package_version
import importlib.util
import inspect
import io
import json
from pathlib import Path
import re
import socket
import sys

import ee
from ee.apitestcase import ApiTestCase
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("p00_live_fixtures", ROOT / "tools" / "live_fixtures.py")
LF = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = LF                     # dataclasses resolve annotations through sys.modules
_SPEC.loader.exec_module(LF)

from tools.fixtures import calibration, core, registry, select  # noqa: E402  (path set by the harness)

_CATALOGUE_SPEC = importlib.util.spec_from_file_location("p00_block_catalogue", ROOT / "tools" / "block_catalogue.py")
CATALOGUE = importlib.util.module_from_spec(_CATALOGUE_SPEC)
sys.modules[_CATALOGUE_SPEC.name] = CATALOGUE
_CATALOGUE_SPEC.loader.exec_module(CATALOGUE)

HAVE_REFERENCES = all(importlib.util.find_spec(m) is not None for m in ("timm", "segmentation_models_pytorch"))
needs_references = pytest.mark.skipif(not HAVE_REFERENCES, reason="the references need timm and SMP")

# SHA-256 of each calibration fixture's inputs and weights: the same seed gives the same
# bytes on every machine (integer arithmetic only; no RNG library).
DATA_SHA256 = {
    "raster/conv2d/k3": ("359a449fc44f3b59135b6ab6a49ef597997540c231af3391b704db7301941859",
                         "700b94506fa64f12a47ff06e13fcb5aeed61cca886f71fbf6ec8711f1cc6e7b4"),
    "raster/pool2d/avg-k2-s2": ("4baee82d66dc784102de3d22adff79796001eef123908243a35afddb997142de",
                                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
    "raster/pool2d/max-k3-s2-p1": ("ad3434668f58c2d7e862d47f7a4547dd6e96ba9c8105ca55a5754bea73ebc4ec",
                                   "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
    "raster/nearest_upsample/f2": ("eb028add8c366298d5de27cf2f44db2e6895291d838463c37a22bc22f75b70bb",
                                   "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
    "window/relative_position_bias/w7": ("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                                         "d2fa3bcc88ff11f6160f9e539f47508d9efd4b82b76980650f50edc363bc7c4f"),
    "window/attention/w7-s3": ("3015ba99cad4e6bd829d83ae36707675ec5f6b6b1eda782fdb005f807797a8c4",
                               "d7d93b90dcf44776c43bde30f02d190a761cbf5fe96b1156635d0b9e547dbfd7"),
    "window/layer_norm2d/w7-eps1e-6": ("500d795694996ce463e5213f00fca7106666c33cc674e141d72b5d3cc65046b5",
                                       "a10dd390ae4b2b9c66a789c09b4e9d0494c3e92dabcf76471b25bdec56054874"),
    "window/patch_merge/w7": ("ea8f6f45b61a3ef030f39a30086db0196a6e64f349a51c6a8d404f8cdd485d22",
                              "3fb01e86f33580aa2d1f99f1127bb407549f07066e0c7f450f82dfb58aee2a79"),
    "window/conv3x3_window/k3-w7": ("dec7d72c707ff2ad0edf27ff5392585187a127c4e10b080c30b830cd12f7228f",
                                    "1ed4bceb0bcc41a8b0ae31e979253c3a1f597dc4dd62b27421d7176670d594a5"),
    "window/group_product/g4-w7": ("09de39684361bb12c636fbea39cc4cca63661048d05b360a830cee8ade5f179f",
                                   "86157be195669ff78bfb53c0400263771822fda50e8a9cb61cb59b5f995ec191"),
    "window/window_pool/bins1236-w7": ("9fdb811a2b7ca84ba2f75dd9dcd07703345a7a90cd514064ddb40724ad532347",
                                       "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
    "window/patch_embed/p4-vvvh-d96": ("e211633e21866178739a3c1a11c0d4cf692f6bdd7f4d46860433536c7274c66c",
                                       "bcaa6fde6eb0a232427f46d94940da17519f2f48620dda473aca8bd3c0a228a4"),
}

# The 14 Fixed blocks Phase 2 promotes (plan, Phase 2 task board).
PHASE2_FIXED = {"relative_position_bias", "window_attention", "shifted_window_mask", "layer_norm2d",
                "patch_merge", "gather_windows", "merge_windows", "conv3x3_window", "nearest_upsample",
                "pool2d", "patch_embed", "stride_columns", "window_pool", "group_product"}

# New blocks stay outside Phase 2's historical calibration census. The depthwise
# block passed its separate SegFormer G3-live gate, and the bounded activations and the
# standalone raster BatchNorm, raster bilinear upsampling and raster grouped convolution
# theirs, and row concatenation, window depth-to-space and class-token attention
# theirs; raster input standardization is offline only.
OFFLINE_ONLY_FIXED: set[str] = {"standardize"}
COMPOSITION_FIXTURES = {"window/concat_rows/prithvi-crop-logits"}
"""Runnable composition checks excluded from block coverage and evidence counts."""

SEGFORMER_G3_BLOCKS = {"depthwise_conv3x3_window"}
HARD_ACTIVATION_G3_BLOCKS = {"relu6", "hardsigmoid", "hardswish"}
BATCH_NORM_G3_BLOCKS = {"batch_norm"}
BILINEAR_G3_BLOCKS = {"bilinear_upsample"}
GROUPED_CONV_G3_BLOCKS = {"grouped_conv2d"}
PRITHVI_G3_BLOCKS = {"concat_rows", "window_depth_to_space", "class_token_attention"}

# The SegFormer composites promoted from the model (X01a): Fixed at the settings the
# SegFormer-B0 live runs cover, outside Phase 2's calibration census.
SEGFORMER_FIXED = {"channel_norm", "decimate", "overlap_embedding", "spatial_reduction",
                   "shared_key_attention", "global_attention", "mix_ffn", "mit_block", "project_pyramid",
                   "fuse_head"}

# The Swin and UPerNet composites promoted from the Swin-T224 model (X01b): Fixed at the
# settings the Swin-T224 and capstone live runs cover, outside Phase 2's calibration census.
SWIN_UPERNET_FIXED = {"swin_mlp_branch", "swin_block", "shifted_swin_block", "swin_stage", "swin_patch_embedding",
                      "swin_encoder", "psp_layer_norm", "psp_branch", "psp_forward", "fpn_layer_norm", "fpn_lateral",
                      "fpn_upsample", "fpn_merge", "fpn_conv_block", "fpn_forward", "fusion_resize_to_c1",
                      "fusion_resize_level", "fusion_concat", "fusion_bn_relu", "fusion_grouped_double",
                      "fusion_forward", "head_conv", "head_upsample", "head_forward", "head_logits", "head_dense"}

DISCOVERY = Path(ee.__file__).parent / "tests" / "cloud_api_discovery_document.json"


# ---------------------------------------------------------------------------
# Fixtures of the tests
# ---------------------------------------------------------------------------
@pytest.fixture
def no_network(monkeypatch):
    """Any socket connection fails the test (and is recorded)."""
    attempts = []

    def refuse(*args, **kwargs):
        attempts.append(args)
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    return attempts


@pytest.fixture(scope="module")
def plan():
    return LF.make_plan(select(["calibration"]))


@pytest.fixture(scope="module")
def local(plan):
    if not HAVE_REFERENCES:
        pytest.skip("the references need timm and SMP")
    return {f.id: LF.evaluate_local(f, plan.data[f.id]) for f in plan.fixtures}


@contextlib.contextmanager
def offline_session(project, credentials):
    """Stands in for ``live_session``: the offline signatures, no initialization."""
    case = ApiTestCase()
    case.setUp()
    try:
        yield
    finally:
        case.tearDown()


def simulated(plan, local) -> dict:
    """{graph SHA-256: the response Earth Engine would give if it equalled the mirror (the
    reference where there is no mirror; the input itself for an input echo)}."""
    responses = {}
    for f in plan.fixtures:
        data, values = plan.data[f.id], local[f.id]
        for request in plan.requests[f.id]:
            expected = {}
            for r in request.readouts:
                if r.kind == "input":
                    expected[r.name] = data.inputs[r.source]
                elif values.mirror is not None and r.name in values.mirror:
                    expected[r.name] = values.mirror[r.name]
                else:
                    expected[r.name] = values.reference[r.name]
            responses[request.graph_sha256] = LF.simulate_response(request, expected)
    return responses


def fake_compute_pixels(responses: dict, calls: list):
    def computePixels(params):
        assert params["fileFormat"] == "NUMPY_NDARRAY"
        sha = hashlib.sha256(params["expression"].serialize().encode()).hexdigest()
        calls.append(sha)
        return responses[sha].copy()
    return computePixels


def _request_of(plan, fixture_id: str, readout: str):
    return next(r for r in plan.requests[fixture_id] if any(x.name == readout for x in r.readouts))


# ---------------------------------------------------------------------------
# R7: --plan and --offline make zero cloud calls
# ---------------------------------------------------------------------------
def test_plan_makes_zero_cloud_calls_and_fits_the_budget(no_network, capsys):
    plan = LF.make_plan(select(["calibration"]))
    assert not plan.violations
    totals = plan.totals()
    assert totals["requests"] == sum(f.requests for f in plan.fixtures) == 18
    assert totals["max_values"] <= LF.DEFAULT_BUDGET.values
    assert totals["max_request_bytes"] <= LF.DEFAULT_BUDGET.request_bytes
    assert totals["assets"] == totals["tasks"] == 0
    assert LF.main(["--plan", "calibration"]) == 0
    assert "Within budget." in capsys.readouterr().out
    assert no_network == []


def test_offline_building_refuses_every_cloud_entry_point():
    with pytest.raises(LF.CloudCall):
        with LF.offline_earth_engine():
            ee.data.computePixels({})
    with pytest.raises(LF.CloudCall):
        with LF.offline_earth_engine():
            ee.Image(1).getInfo()


@needs_references
def test_offline_makes_zero_cloud_calls_and_every_mirror_is_within_the_target(no_network, tmp_path):
    report = LF.offline(select(["calibration"]))
    assert report["status"] == "passed" and report["cloud_calls"] == 0
    compared = 0
    for entry in report["fixtures"]:
        assert entry["status"] == "passed", entry
        for output in entry["outputs"].values():
            if output["mirror_vs_pytorch"] is not None:
                compared += 1
                assert output["mirror_vs_pytorch"]["max_abs"] <= core.TOLERANCE
    assert compared == 13                   # every window-lane output; the raster lane has no mirror
    assert LF.main(["--offline", "calibration", "--output", str(tmp_path / "offline.json")]) == 0
    assert json.loads((tmp_path / "offline.json").read_text())["status"] == "passed"
    assert no_network == []


# ---------------------------------------------------------------------------
# R7: determinism
# ---------------------------------------------------------------------------
def test_fixture_data_is_deterministic():
    assert set(DATA_SHA256) == set(calibration())
    for fid, (inputs, weights) in DATA_SHA256.items():
        f = registry()[fid]
        first, second = f.data(f.seed), f.data(f.seed)
        assert (first.inputs_sha256, first.weights_sha256) == (second.inputs_sha256, second.weights_sha256)
        assert (first.inputs_sha256, first.weights_sha256) == (inputs, weights), fid
        other = f.data(f.seed + 1)
        assert (other.inputs_sha256, other.weights_sha256) != (inputs, weights) or not (first.inputs or first.weights)


def test_graphs_are_stable_across_builds(plan):
    again = LF.make_plan(select(["calibration"]))
    assert [r.graph_sha256 for r in plan.all_requests] == [r.graph_sha256 for r in again.all_requests]
    assert [r.request_bytes for r in plan.all_requests] == [r.request_bytes for r in again.all_requests]


def test_dyadic_values_are_exact_and_inline_values_are_checked():
    values = core.dyadic(7, "probe", (4096,))
    assert values.dtype == np.float32 and np.abs(values).max() <= 1
    assert np.array_equal(values, core.dyadic(7, "probe", (4096,)))
    assert not np.array_equal(values, core.dyadic(7, "other", (4096,)))
    assert len(np.unique(values)) > 4000
    with pytest.raises(ValueError):
        core.dyadic(1, "x", (3,), scale=0.3)
    with pytest.raises(AssertionError):
        core.dyadic(1, "x", (64,), scale=0.25, offset=1.0)        # 26 significant bits
    core.check_inline_values(np.array([0.0, 1.5, -2.0 ** -20, 3e15], np.float32))
    for bad in (np.array([-0.0], np.float32), np.array([np.nan], np.float32), np.array([2.0 ** 60], np.float32),
                np.array([1e30], np.float32), np.array([0.5], np.float64)):
        with pytest.raises((ValueError, TypeError)):
            core.check_inline_values(bad)


# ---------------------------------------------------------------------------
# R7: a run against a mocked computePixels
# ---------------------------------------------------------------------------
@needs_references
def test_a_run_answered_by_the_mirrors_passes(plan, local, monkeypatch, no_network, tmp_path):
    calls = []
    monkeypatch.setattr(ee.data, "computePixels", fake_compute_pixels(simulated(plan, local), calls))
    record = LF.run(plan.fixtures, project="test-project", session=offline_session, command="test")
    assert record["status"] == "passed", [e["failures"] for e in record["fixtures"]]
    assert record["cloud_calls"]["requests"] == len(calls) == 18
    assert record["cloud_calls"]["refused"] == []
    for entry in record["fixtures"]:
        assert entry["status"] == "passed"
        for name, output in entry["outputs"].items():
            if output["kind"] == "input":
                assert output["input"]["differing"] == 0
                continue
            assert output["pytorch"]["over_tolerance"] == 0
            assert output["float32"] and output["invalid_pixels"] == 0
            if entry["spec"]["mirror"] is not None:
                assert output["mirror"]["differing"] == 0 and output["mirror"]["exact"]
    path = LF.write_evidence(record, tmp_path / "run.json", "calibration")
    text = path.read_text()
    assert json.loads(text)["kind"] == "live_fixture_run"
    assert str(Path.home()) not in text and str(ROOT) not in text
    assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text)
    assert set(json.loads(text)["versions"]) == {"python", *LF.VERSION_PACKAGES}
    assert no_network == []


@needs_references
def test_one_changed_value_fails_the_run_and_is_reported(plan, local, monkeypatch):
    responses = simulated(plan, local)
    attention = _request_of(plan, "window/attention/w7-s3", "concat")
    responses[attention.graph_sha256]["concat_2"][3, 5] += np.float32(1e-3)
    conv = _request_of(plan, "raster/conv2d/k3", "out")
    responses[conv.graph_sha256]["out_1"][4, 7] = np.nextafter(responses[conv.graph_sha256]["out_1"][4, 7],
                                                               np.float32(np.inf))
    pool = _request_of(plan, "window/window_pool/bins1236-w7", "input")
    responses[pool.graph_sha256]["input_3"][6, 0] = np.nextafter(responses[pool.graph_sha256]["input_3"][6, 0],
                                                                 np.float32(-np.inf))
    monkeypatch.setattr(ee.data, "computePixels", fake_compute_pixels(responses, []))
    record = LF.run(plan.fixtures, project="test-project", session=offline_session)
    assert record["status"] == "failed"
    by_id = {e["spec"]["id"]: e for e in record["fixtures"]}
    failed = {fid for fid, e in by_id.items() if e["status"] == "failed"}
    assert failed == {"window/attention/w7-s3", "window/window_pool/bins1236-w7"}
    concat = by_id["window/attention/w7-s3"]["outputs"]["concat"]
    assert concat["pytorch"]["over_tolerance"] == 1
    assert {k: concat["pytorch"]["worst"][k] for k in ("row", "col", "channel")} == {"row": 3, "col": 5, "channel": 2}
    assert concat["mirror"]["differing"] == 1
    assert concat["mirror"]["first"][0]["row"] == 3 and concat["mirror"]["first"][0]["col"] == 5
    assert any("row 3, col 5, channel 2" in f for f in by_id["window/attention/w7-s3"]["failures"])
    echo = by_id["window/window_pool/bins1236-w7"]["outputs"]["input"]["input"]
    assert echo["differing"] == 1 and (echo["first"][0]["row"], echo["first"][0]["channel"]) == (6, 3)
    # One ulp is within 1e-4 of PyTorch, and the raster lane has no mirror: still a pass.
    out = by_id["raster/conv2d/k3"]["outputs"]["out"]
    assert by_id["raster/conv2d/k3"]["status"] == "passed" and out["pytorch"]["max_abs"] > 0


# ---------------------------------------------------------------------------
# Fix round: selection and copy fixtures demand equality; the max pool tests its padding
# ---------------------------------------------------------------------------
def test_selection_and_copy_fixtures_demand_bit_equality():
    tolerances = {fid: registry()[fid].tolerance for fid in calibration() if fid.startswith("raster/")}
    assert tolerances == {"raster/conv2d/k3": 1e-4, "raster/pool2d/avg-k2-s2": 1e-4,
                          "raster/pool2d/max-k3-s2-p1": 0.0, "raster/nearest_upsample/f2": 0.0}
    a = np.array([[[0.0], [1.5]]], np.float32)
    assert LF.compare_reference(a, a, 0.0)["pass"]
    signed = LF.compare_reference(np.array([[[-0.0], [1.5]]], np.float32), a, 0.0)
    assert signed["over_tolerance"] == 1 and not signed["pass"] and signed["comparison"] == "bits"
    assert (signed["worst"]["row"], signed["worst"]["col"]) == (0, 0)
    ulp = a.copy()
    ulp[0, 1, 0] = np.nextafter(ulp[0, 1, 0], np.float32(np.inf))
    one = LF.compare_reference(ulp, a, 0.0)
    assert one["over_tolerance"] == 1 and one["worst"]["col"] == 1
    assert LF.compare_reference(ulp, a, 1e-4)["pass"]              # the absolute form keeps 1e-4
    assert LF.compare_reference(np.array([[[-0.0], [1.5]]], np.float32), a, 1e-4)["pass"]


@needs_references
def test_the_max_pool_fixture_catches_a_pad_that_contributes_and_a_one_ulp_copy(plan, local, monkeypatch):
    import torch
    import torch.nn.functional as F

    fixture = registry()["raster/pool2d/max-k3-s2-p1"]
    x = plan.data[fixture.id].inputs["x"]
    assert x.max() < 0                                              # all negative, never 0
    zero_padded = core.hwc(torch.nn.MaxPool2d(3, 2, 0)(F.pad(core.nchw(x), (1, 1, 1, 1), value=0.0)))
    reference = local[fixture.id].reference["out"]
    assert np.count_nonzero(zero_padded != reference) > 0
    responses = simulated(plan, local)
    pool = _request_of(plan, fixture.id, "out")
    for c in range(2):
        responses[pool.graph_sha256][f"out_{c}"] = zero_padded[..., c]
    nearest = _request_of(plan, "raster/nearest_upsample/f2", "out")
    responses[nearest.graph_sha256]["out_0"][9, 2] = np.nextafter(responses[nearest.graph_sha256]["out_0"][9, 2],
                                                                  np.float32(np.inf))
    monkeypatch.setattr(ee.data, "computePixels", fake_compute_pixels(responses, []))
    selected = [registry()[fixture.id], registry()["raster/nearest_upsample/f2"]]
    by_id = {e["spec"]["id"]: e for e in LF.run(selected, project="p", session=offline_session)["fixtures"]}
    pooled = by_id[fixture.id]["outputs"]["out"]["pytorch"]
    assert by_id[fixture.id]["status"] == "failed"
    assert pooled["over_tolerance"] == int(np.count_nonzero(zero_padded != reference))
    copied = by_id["raster/nearest_upsample/f2"]
    assert copied["status"] == "failed" and copied["outputs"]["out"]["pytorch"]["over_tolerance"] == 1
    assert any("differ bit for bit" in f and "row 9, col 2, channel 0" in f for f in copied["failures"])


@needs_references
def test_a_live_graph_that_differs_from_the_plan_stops_the_run(plan, monkeypatch):
    fixture = registry()["raster/nearest_upsample/f2"]
    builds = []

    def build(ee_module, data):
        builds.append(1)
        readouts = fixture.build(ee_module, data)
        if len(builds) > 1:                       # after initialization: another graph
            readouts[0] = dataclasses.replace(readouts[0], image=readouts[0].image.add(0))
        return readouts

    calls = []
    monkeypatch.setattr(ee.data, "computePixels", fake_compute_pixels({}, calls))
    record = LF.run([dataclasses.replace(fixture, build=build)], project="p", session=offline_session)
    assert record["status"] == "aborted" and calls == []
    assert "differ from the plan" in record["errors"][0]


# ---------------------------------------------------------------------------
# R7: a plan over budget is refused before any call
# ---------------------------------------------------------------------------
def _never(*args, **kwargs):
    raise AssertionError("must not be reached")


@pytest.mark.parametrize("budget", ["requests=17", "values=30000", "request_bytes=200000", "window_side=1",
                                    "raster_side=8"])
def test_a_plan_over_budget_is_refused_before_any_call(budget, monkeypatch, capsys):
    monkeypatch.setattr(ee.data, "computePixels", _never)
    monkeypatch.setattr(LF, "live_session", _never)
    with pytest.raises(LF.Refused) as refused:
        LF.run(select(["calibration"]), project="p", budget=LF.parse_budget(budget))
    assert refused.value.reasons
    assert LF.main(["--run", "calibration", "--project", "p", "--budget", budget]) == 2
    assert "refused before any cloud call" in capsys.readouterr().err
    assert LF.main(["--plan", "calibration", "--budget", budget]) == 1


def test_a_fixture_that_builds_other_requests_than_it_declares_is_refused(monkeypatch):
    monkeypatch.setattr(LF, "live_session", _never)
    fixture = dataclasses.replace(registry()["raster/pool2d/avg-k2-s2"], requests=1)
    with pytest.raises(LF.Refused, match="declares 1 requests but builds 2"):
        LF.run([fixture], project="p")


def test_a_failing_reference_is_refused_before_any_call(monkeypatch):
    monkeypatch.setattr(LF, "live_session", _never)

    def broken(data):
        raise RuntimeError("no reference")

    fixture = dataclasses.replace(registry()["raster/conv2d/k3"], reference=broken)
    with pytest.raises(LF.Refused, match="no reference"):
        LF.run([fixture], project="p")


def test_a_reference_that_calls_earth_engine_is_refused_even_after_initialization(monkeypatch):
    calls = []

    def reaches_earth_engine(data):
        calls.append(ee.Image(1).getInfo())                   # ApiTestCase would answer this
        return {}

    fixture = dataclasses.replace(registry()["raster/conv2d/k3"], reference=reaches_earth_engine)
    with offline_session("p", "adc"):                          # an initialized client
        local = LF.evaluate_local(fixture, fixture.data(fixture.seed))
    assert calls == [] and any("CloudCall" in e for e in local.errors)
    monkeypatch.setattr(LF, "live_session", _never)
    with pytest.raises(LF.Refused, match="CloudCall"):
        LF.run([fixture], project="p")


def test_duplicate_fixture_ids_are_refused(monkeypatch):
    monkeypatch.setattr(LF, "live_session", _never)
    fixture = registry()["raster/conv2d/k3"]
    with pytest.raises(ValueError, match="duplicate fixture ids"):
        LF.make_plan([fixture, fixture])
    with pytest.raises(LF.Refused, match="duplicate fixture ids"):
        LF.run([fixture] * 41, project="p")


def test_a_window_lane_fixture_declares_the_window_it_is_budgeted_by():
    fixture = registry()["window/layer_norm2d/w7-eps1e-6"]
    with pytest.raises(ValueError, match="must equal settings"):
        dataclasses.replace(fixture, window=9)
    with pytest.raises(ValueError, match="must equal settings"):
        dataclasses.replace(fixture, settings={"eps": 1e-6})
    assert all(f.window == f.settings["window"] for f in registry().values() if f.lane == "window")
    assert LF.side_limit("window", LF.DEFAULT_BUDGET, 7) == 56
    assert LF.side_limit("raster", LF.DEFAULT_BUDGET, 7) == 64


def test_a_mirror_bound_cannot_exceed_the_tolerance():
    fn = registry()["window/layer_norm2d/w7-eps1e-6"].mirror.fn
    assert core.MirrorSpec(fn, False, "test", bound=1e-4).bound == 1e-4
    for kwargs in ({"bound": 2e-4}, {"bound": 1.0}, {"bound": None}, {"bound": -1e-6}):
        with pytest.raises(ValueError):
            core.MirrorSpec(fn, False, "test", **kwargs)
    with pytest.raises(ValueError):
        core.MirrorSpec(fn, True, "test", bound=1e-6)
    fixture = registry()["window/layer_norm2d/w7-eps1e-6"]
    with pytest.raises(ValueError, match="bound may not exceed"):
        dataclasses.replace(fixture, tolerance=1e-5, mirror=core.MirrorSpec(fn, False, "test", bound=1e-4))
    assert dataclasses.replace(fixture, tolerance=1e-5,
                               mirror=core.MirrorSpec(fn, False, "test", bound=1e-5)).mirror.bound == 1e-5


def test_the_budget_may_be_lowered_never_raised(capsys):
    assert LF.parse_budget("requests=10, values=5") == dataclasses.replace(LF.DEFAULT_BUDGET, requests=10, values=5)
    for text in ("requests=41", "values=1000001", "request_bytes=2000001", "in_flight=5", "raster_side=65",
                 "window_side=9", "assets=1", "tasks=1", "requests=0", "requests=-1", "requests", "speed=2"):
        with pytest.raises(ValueError):
            LF.parse_budget(text)
    assert LF.main(["--plan", "calibration", "--budget", "requests=41"]) == 2
    assert "only the lead may raise" in capsys.readouterr().err
    guard = LF.Guard(max_compute=1)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ee.data, "computePixels", lambda params: "pixels")
        assert guard.compute_pixels({}) == "pixels"
        with pytest.raises(LF.BudgetExceeded):
            guard.compute_pixels({})


# ---------------------------------------------------------------------------
# R7: exports, assets and tasks are refused
# ---------------------------------------------------------------------------
class FakeHttp:
    """An httplib2-like transport: records every request and answers with ``content``."""

    def __init__(self, content: bytes = b""):
        self.content = content
        self.requests = []

    def request(self, uri, method="GET", body=None, headers=None, redirections=None, connection_type=None):
        import httplib2

        self.requests.append({"uri": uri, "method": method, "body": body})
        return httplib2.Response({"status": 200, "content-type": "application/octet-stream"}), self.content


@contextlib.contextmanager
def document_resource(http: FakeHttp):
    """ApiTestCase signatures plus a Cloud API resource built from the bundled discovery
    document on ``http``: the real ee.data request path, with nothing sent anywhere."""
    if not DISCOVERY.exists():
        pytest.skip("earthengine-api's test discovery document is not installed")
    from ee import _cloud_api_utils

    document = json.loads(DISCOVERY.read_text())
    case = ApiTestCase()
    case.setUp()
    state = ee.data._get_state()
    try:
        state.cloud_api_resource = _cloud_api_utils.build_cloud_resource_from_document(document, http_transport=http)
        state.cloud_api_resource_raw = _cloud_api_utils.build_cloud_resource_from_document(
            document, http_transport=http, raw=True)
        yield state.cloud_api_resource_raw
    finally:
        case.tearDown()


def test_every_export_asset_and_task_function_is_refused():
    refused = set(LF.refused_functions())
    public = {name for name, value in vars(ee.data).items()
              if not name.startswith("_") and inspect.isfunction(value) and value.__module__ == "ee.data"}
    pattern = re.compile(r"export|asset|task|operation|ingestion|folder|bucket|download|thumb|map|tile|"
                         r"acl|iam|copy|rename|delete|cancel|create|update|list|compute|getinfo", re.I)
    assert {name for name in public if pattern.search(name)} - {"computePixels", "convert_asset_id_to_asset_name"} <= refused
    assert "computePixels" not in refused and "getAlgorithms" not in refused
    guard = LF.Guard(max_compute=1)
    with guard.functions():
        for name in sorted(refused):
            with pytest.raises(LF.RefusedCall):
                getattr(ee.data, name)()
    assert sorted(guard.refused) == sorted(refused)


def test_callable_stand_ins_bound_to_ee_data_names_are_refused():
    with offline_session("p", "adc"):                            # ApiTestCase binds bound methods
        names = set(LF.refused_functions())
        assert {"getMapId", "getDownloadId", "getTableDownloadId", "computeValue"} <= names
        assert not inspect.isfunction(ee.data.getMapId)               # a bound method
        guard = LF.Guard(max_compute=0)
        with guard.functions():
            for name in ("getMapId", "getDownloadId", "getTableDownloadId", "computeValue"):
                with pytest.raises(LF.RefusedCall):
                    getattr(ee.data, name)({})
    assert set(LF.refused_functions()) == names                   # the same names in a clean client
    assert not {"Any", "Callable", "Iterator", "Sequence"} & names   # classes are not refused


def test_the_transport_caps_algorithm_lists_and_never_retries(monkeypatch):
    import types

    seen = []
    monkeypatch.setattr(ee.data, "_execute_cloud_call",
                        lambda call, num_retries=None: seen.append((call.methodId, num_retries)) or {})
    listing = types.SimpleNamespace(methodId="earthengine.projects.algorithms.list")
    compute = types.SimpleNamespace(methodId=LF.COMPUTE_METHOD)
    guard = LF.Guard(max_compute=1)
    with guard.transport():
        for _ in range(LF.MAX_ALGORITHMS_LIST):
            ee.data._execute_cloud_call(listing, num_retries=5)
        with pytest.raises(LF.BudgetExceeded):
            ee.data._execute_cloud_call(listing)
        ee.data._execute_cloud_call(compute, num_retries=3)
        with pytest.raises(LF.BudgetExceeded):
            ee.data._execute_cloud_call(compute)
    assert LF.MAX_ALGORITHMS_LIST == 2
    assert seen == [("earthengine.projects.algorithms.list", 0)] * 2 + [(LF.COMPUTE_METHOD, 0)]
    assert guard.record()["algorithms_list"] == 2 and guard.record()["computePixels"] == 1


def test_exports_assets_and_tasks_fail_before_they_are_sent():
    http = FakeHttp()
    with document_resource(http) as resource:
        guard = LF.Guard(max_compute=1)
        with guard.transport(), guard.functions():
            image = ee.Image.constant(1)
            attempts = [
                lambda: ee.batch.Export.image.toAsset(image, description="p00", assetId="projects/my-project/assets/x").start(),
                lambda: ee.batch.Export.table.toDrive(ee.FeatureCollection([]), description="p00").start(),
                lambda: ee.data.exportImage("id", {}),
                lambda: ee.data.createAsset({"type": "FOLDER"}, "projects/my-project/assets/f"),
                lambda: ee.data.deleteAsset("projects/my-project/assets/x"),
                lambda: ee.data.newTaskId(),
                lambda: ee.data.getTaskStatus("T"),
                lambda: ee.data.startIngestion("id", {}),
                lambda: ee.data.listAssets({"parent": "projects/my-project/assets"}),
                lambda: image.getInfo(),
                # Below the functions: a raw Cloud API export request reaches the transport guard.
                lambda: ee.data._execute_cloud_call(
                    resource.projects().image().export(project="projects/my-project", body={})),
                lambda: ee.data._execute_cloud_call(
                    resource.projects().table().export(project="projects/my-project", body={})),
                lambda: ee.data._execute_cloud_call(
                    resource.projects().value().compute(project="projects/my-project", body={})),
            ]
            for attempt in attempts:
                with pytest.raises(LF.RefusedCall):
                    attempt()
        assert http.requests == []                        # nothing was sent
        assert {"earthengine.projects.image.export", "earthengine.projects.table.export",
                "earthengine.projects.value.compute"} <= set(guard.refused)


def test_compute_pixels_passes_the_guard_once_with_the_planned_request(plan):
    request = plan.requests["window/window_pool/bins1236-w7"][0]
    expected = {r.name: np.ones(r.shape, np.float32) for r in request.readouts}
    buffer = io.BytesIO()
    np.save(buffer, LF.simulate_response(request, expected), allow_pickle=False)
    http = FakeHttp(buffer.getvalue())
    with document_resource(http):
        live = LF.build_requests(registry()["window/window_pool/bins1236-w7"],
                                 plan.data["window/window_pool/bins1236-w7"])[0]
        guard = LF.Guard(max_compute=1)
        with guard.transport(), guard.functions():
            pixels = guard.compute_pixels(live.params())
            with pytest.raises(LF.BudgetExceeded):
                guard.compute_pixels(live.params())
    assert live.graph_sha256 == request.graph_sha256
    assert len(http.requests) == 1 and http.requests[0]["method"] == "POST"
    assert "/projects/my-project/image:computePixels" in http.requests[0]["uri"]
    assert len(http.requests[0]["body"].encode()) == request.request_bytes   # the plan's request bytes
    assert pixels.dtype.names == tuple(request.bands) and pixels.shape == (request.height, request.width)
    assert guard.record() == {"requests": 1, "computePixels": 1, "algorithms_list": 0, "refused": ["computePixels"]}


# ---------------------------------------------------------------------------
# R7: every fixture names a catalogue block and settings that block accepts
# ---------------------------------------------------------------------------
def _catalogue() -> dict:
    lanes = {"R": "raster", "W": "window"}
    return {(lanes[layout], entry.name): entry for layout, _, entry in CATALOGUE.collect_blocks()}


def _defined_in_tensor_ee(fn) -> bool:
    return getattr(fn, "__module__", "").startswith("tensor_ee.")


def test_every_fixture_names_a_catalogue_block_and_accepted_settings():
    from tensor_ee.core.precision import VARIANTS

    catalogue = _catalogue()
    for fixture in registry().values():
        entry = catalogue.get((fixture.lane, fixture.block))
        assert entry is not None, f"{fixture.id}: {fixture.block} is not a {fixture.lane}-lane block"
        for part in fixture.parts:
            assert (fixture.lane, part) in catalogue, f"{fixture.id}: part {part}"
        assert fixture.policy == entry.policy or VARIANTS.get(fixture.policy) == entry.policy, fixture.id
        assert fixture.accepts, fixture.id
        for key, value in fixture.settings.items():
            owners = [fn for fn in fixture.accepts if key in inspect.signature(fn).parameters]
            assert owners, f"{fixture.id}: no accepted callable takes the setting {key}"
            if fixture.calibration:            # today's settings: the library defaults
                for fn in filter(_defined_in_tensor_ee, owners):
                    default = inspect.signature(fn).parameters[key].default
                    if key == "shift" and default is None:     # the shift defaults to window // 2
                        default = fixture.settings["window"] // 2
                    if default is not inspect.Parameter.empty:
                        assert core.jsonable(value) == core.jsonable(default), (fixture.id, key, value, default)


def test_the_calibration_covers_every_fixed_block_of_phase_2(plan):
    catalogue = _catalogue()
    fixed = {name for (_, name), entry in catalogue.items() if entry.status.startswith("Fixed")}
    assert fixed <= PHASE2_FIXED | OFFLINE_ONLY_FIXED | SEGFORMER_FIXED | SWIN_UPERNET_FIXED
    offline = {name for (_, name), entry in catalogue.items()
               if entry.status.startswith("Fixed — offline only:")}
    assert offline == OFFLINE_ONLY_FIXED
    new_covered = {name for f in select(["segformer_b0", "hard_activations", "batch_norm", "bilinear", "grouped_conv", "class_token", "prithvi_depth_to_space", "prithvi_settings", "prithvi_model", "standardize"])
                   if f.id not in COMPOSITION_FIXTURES and f.mirror is not None and not f.calibration
                   for name in (f.block, *f.parts)}
    assert (SEGFORMER_G3_BLOCKS | HARD_ACTIVATION_G3_BLOCKS | BATCH_NORM_G3_BLOCKS | BILINEAR_G3_BLOCKS
            | GROUPED_CONV_G3_BLOCKS | PRITHVI_G3_BLOCKS | OFFLINE_ONLY_FIXED <= new_covered)
    covered = {name for f in plan.fixtures if f.id not in COMPOSITION_FIXTURES
               for name in (f.block, *f.parts)}
    assert PHASE2_FIXED <= covered
    lanes = {f.lane for f in plan.fixtures}
    assert lanes == {"raster", "window"}
    assert all(f.tolerance <= core.TOLERANCE for f in plan.fixtures)


def test_composition_fixture_does_not_claim_block_coverage():
    fixtures = select(["prithvi_model"])
    assert {fixture.id for fixture in fixtures} == COMPOSITION_FIXTURES
    assert all(not fixture.calibration and fixture.parts == () for fixture in fixtures)


def test_fixture_records_reject_a_larger_tolerance_and_malformed_ids():
    fixture = registry()["raster/conv2d/k3"]
    with pytest.raises(ValueError):
        dataclasses.replace(fixture, tolerance=2e-4)
    with pytest.raises(ValueError):
        dataclasses.replace(fixture, id="conv2d")
    with pytest.raises(ValueError):
        dataclasses.replace(fixture, id="window/conv2d/k3")
    assert dataclasses.replace(fixture, tolerance=0.0).tolerance == 0.0
    with pytest.raises(KeyError):
        select(["window/no-such-fixture"])
    assert [f.id for f in select(["window/patch*,raster/conv2d/k3"])] == [
        "raster/conv2d/k3", "window/patch_merge/w7", "window/patch_embed/p4-vvvh-d96",
        "window/patch_merge/w5", "window/patch_merge/w8", "window/patch_merge/w12",
        "window/patch_embed/p2-b3-d48-w5", "window/patch_embed/p2-b1-d128-w8", "window/patch_embed/p2-b2-d96-w12",
        "window/patch_embed/p4-b4-d128-w12", "window/patch_embed/p8-b1-d96-w7", "window/patch_embed/p8-b4-d48-w8",
        "window/patch_embed/prithvi-1536-768-w14"]


def test_comparisons_count_bits_and_non_finite_values():
    a = np.zeros((2, 2, 1), np.float32)
    b = a.copy()
    b[1, 0, 0] = -0.0
    result = LF.compare_mirror(a, b, True, None)
    assert result["differing"] == 1 and not result["pass"] and result["max_abs"] == 0.0
    c = a.copy()
    c[0, 1, 0] = np.nan
    reference = LF.compare_reference(c, a, 1e-4)
    assert reference["over_tolerance"] == 1 and not reference["pass"]
    assert json.loads(json.dumps(core.jsonable(reference), allow_nan=False))["max_abs"] == "inf"
    assert LF.compare_mirror(a, a + np.float32(1e-7), False, 1e-6)["pass"]


# ---------------------------------------------------------------------------
# R8: the calibration plan's numbers are --plan's
# ---------------------------------------------------------------------------
PLAN_DOC = ROOT / "docs" / "evidence" / "p00-calibration-plan.md"
ROW = re.compile(r"^\| `(?P<id>(raster|window)/[^`]+)` \| (?P<index>\d+) \| (?P<rows>\d+) × (?P<cols>\d+) \| "
                 r"(?P<bands>[\d,]+) \| (?P<values>[\d,]+) \| (?P<payload>[\d,]+) \| (?P<request>[\d,]+) \| "
                 r"(?P<constants>[\d,]+) \| (?P<inline>[\d,]+) \|")


def _number(text: str) -> int:
    return int(text.replace(",", ""))


def test_the_calibration_plan_document_matches_the_plan(plan):
    text = PLAN_DOC.read_text()
    recorded = re.search(r"earthengine-api\s+(\d+\.\d+\.\d+)", text).group(1)
    rows = {(m["id"], int(m["index"])): m for m in map(ROW.match, text.splitlines()) if m}
    requests = {(r.fixture, r.index): r for r in plan.all_requests}
    assert set(rows) == set(requests)
    for key, r in requests.items():
        m = rows[key]
        assert (_number(m["rows"]), _number(m["cols"])) == (r.height, r.width), key
        assert (_number(m["bands"]), _number(m["values"]), _number(m["payload"])) == (
            len(r.bands), r.values, r.payload_bytes), key
        assert (_number(m["constants"]), _number(m["inline"])) == (r.embedded_constants, r.inline_values), key
        if package_version("earthengine-api") == recorded:
            assert _number(m["request"]) == r.request_bytes, key
    totals = plan.totals()
    for label, value in (("requests", totals["requests"]), ("values", totals["values"]),
                         ("payload", totals["payload_bytes"]), ("request bytes", totals["request_bytes"])):
        assert f"{value:,}" in text, label
    assert re.search(r"tools/live_fixtures\.py --run calibration --project [\w-]+", text)


def test_the_offline_guard_leaves_a_later_live_initialize_working():
    """ApiTestCase.InitializeApi stubs ee.data._install_cloud_api_resource and its tearDown
    does not restore it: a later real ee.Initialize then installs no Cloud API resource
    (the first calibration attempt failed "not initialized" with zero requests sent)."""
    import ee
    from ee import _state

    original = ee.data._install_cloud_api_resource
    with LF.offline_earth_engine():
        assert ee.data._install_cloud_api_resource is not original
    assert ee.data._install_cloud_api_resource is original
    assert not ee.ApiFunction._api                     # the stand-in algorithms are gone
    installed = []
    ee.data._install_cloud_api_resource = lambda: installed.append(True)
    try:
        ee.data.initialize(credentials=None, cloud_api_key="fake", project="demo-project")
        assert installed == [True]                      # a real initialize installs the resource
    finally:
        ee.data._install_cloud_api_resource = original
        ee.Reset()
