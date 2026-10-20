"""Offline construction and grid contracts of the own-model examples."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import ee
import pytest
import torch
from torch import nn

smp = pytest.importorskip("segmentation_models_pytorch")

from examples import own_model as example
from examples import upload_model_weights as uploader
from tensor_ee import MatchedWeightProvider, check_compatibility, effective_convolutions
from tensor_ee.converter import compile_tiled_model
from tensor_ee.converter.request import deep_recursion, request_reads, export_request, SIZING_REQUEST_ID
from tensor_ee.receptive_field import analyze_receptive_field
from tools.live_fixtures import offline_earth_engine


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def model_and_plan():
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(example.SEED)
        return example.adapt_for_region(example.make_demo_model(), core_size=example.DEMO_CORE_SIZE)


def documented_report(architecture="shallow_demo"):
    guide = (ROOT / "docs/own-model.md").read_text()
    return next(report for block in re.findall(r"```json\n(.*?)\n```", guide, re.S)
                if (report := json.loads(block))["architecture"] == architecture)


def test_prepare_command_is_offline_and_matches_guide():
    # Run the real CLI in a fresh process with network transport and task submission
    # refused. ApiTestCase itself installs signatures before graph construction.
    command = """
from unittest.mock import patch
import ee
from examples import own_model

def refuse(*args, **kwargs):
    raise AssertionError('unexpected remote operation')

make_model = own_model.make_model
def demo_only(**settings):
    assert settings == {'encoder_depth': 3, 'decoder_channels': (16, 8, 4)}
    return make_model(**settings)

with (patch('requests.sessions.Session.request', refuse),
      patch.object(ee.data, '_execute_cloud_call', refuse),
      patch.object(ee.batch.Task, 'start', refuse),
      patch.object(own_model, 'initialize', refuse),
      patch.object(own_model, 'make_model', demo_only),
      patch.object(ee, 'Authenticate', refuse)):
    own_model.main(['--prepare'])
"""
    environment = {**os.environ, "PYTHONPATH": str(ROOT), "HF_HUB_OFFLINE": "1",
                   "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}
    result = subprocess.run([sys.executable, "-c", command], cwd=ROOT, env=environment,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert {key: report[key] for key in documented_report()} == documented_report()
    assert 0 < report["graph_bytes"] < 2_000_000


def test_example_equals_direct_tiled_compilation(model_and_plan):
    model, plan = model_and_plan
    manifest = example.planned_manifest(model)
    with offline_earth_engine() as calls, deep_recursion():
        image = ee.Image.constant([0, 0]).rename(list(example.BANDS))
        provider = MatchedWeightProvider(model, manifest)
        actual, payload = example.build_prediction(model, provider, image, plan)
        source = image.select(list(example.BANDS)).toFloat().reproject(plan.global_grid.projection())
        expected = compile_tiled_model(model, source, example.BANDS, plan,
                                       weight_provider=MatchedWeightProvider(model, manifest))
        assert actual.image.serialize() == expected.image.serialize()
        assert actual.coverage.serialize() == expected.coverage.serialize()
        assert len(actual.tiles) == len(expected.tiles) == 2
        report = example.demo_graph_report(actual, payload)
        assert {key: report[key] for key in documented_report()} == documented_report()
        assert not provider.fallback_keys
        assert provider.used_asset_keys == set(manifest["layers"])
        request = export_request(payload, plan.roi_grid,
                                 asset_id="projects/example-project/assets/own_prediction",
                                 description="own_model", request_id=SIZING_REQUEST_ID)
        reads = request_reads(request)
        assert reads["tables"] == ["projects/example-project/assets/own_model_weights_part_0000"]
    assert not calls


def test_documented_halo_covers_structural_support(model_and_plan):
    model, plan = model_and_plan
    support = analyze_receptive_field(model)
    documented = documented_report()
    assert plan.halo == tuple(documented["halo"]) == (support.safe_halo,) * 2
    assert plan.alignment == documented["alignment"] == support.alignment
    for tile in plan.tiles:
        for axis, start, end, left, right in (
                (support.x, tile.core.x, tile.core.right, tile.context.x, tile.context.right),
                (support.y, tile.core.y, tile.core.bottom, tile.context.y, tile.context.bottom)):
            for index in range(start, end):
                lower, upper = axis.bounds(index)
                assert left <= lower <= upper < right
        shape = (1, len(example.BANDS), tile.context.height, tile.context.width)
        assert check_compatibility(model, shape).status == "supported"
    global_shape = (1, len(example.BANDS), plan.global_grid.height, plan.global_grid.width)
    assert check_compatibility(model, global_shape).status == "unsupported"


def test_standard_checkpoint_loader_and_support_without_tiled_graph(tmp_path, monkeypatch):
    torch.set_num_threads(2)

    def refuse(*args, **kwargs):
        pytest.fail("Standard offline checks must not build a tiled graph or adapt large contexts")

    monkeypatch.setattr(example, "compile_tiled_model", refuse)
    monkeypatch.setattr(example, "adapt_for_region", refuse)
    # Independent standard SMP constructor: its full checkpoint must load unchanged.
    raw = smp.Unet("resnet34", encoder_weights=None, in_channels=2, classes=2).eval()
    with torch.no_grad():
        raw.segmentation_head[0].bias.copy_(torch.tensor([-0.25, 0.75]))
    checkpoint = tmp_path / "state.pth"
    torch.save(raw.state_dict(), checkpoint)
    del raw
    adapted = example.load_model(checkpoint)
    assert torch.equal(adapted.segmentation_head.get_submodule("0").bias,
                       torch.tensor([-0.25, 0.75]))
    assert tuple(adapted.tensor_ee_input_shape) == (2, 64, 64)
    support = analyze_receptive_field(adapted)
    assert (support.safe_halo, support.alignment) == (544, 32)
    for core, context in ((64, 1152), (1024, 2112)):
        plan = example.make_plan(support, core_size=core)
        assert len(plan.tiles) == 2
        assert {(t.context.height, t.context.width) for t in plan.tiles} == {(context, context)}
        for tile in plan.tiles:
            for axis, start, end, left, right in (
                    (support.x, tile.core.x, tile.core.right, tile.context.x, tile.context.right),
                    (support.y, tile.core.y, tile.core.bottom, tile.context.y, tile.context.bottom)):
                for index in range(start, end):
                    lower, upper = axis.bounds(index)
                    assert left <= lower <= upper < right
    assert example.plan_report(plan, architecture="standard_unet_resnet34") == {
        key: value for key, value in documented_report("standard_unet_resnet34").items()
        if key != "status"}
    torch.save({}, checkpoint)
    with pytest.raises(RuntimeError, match="Missing key"):
        example.load_model(checkpoint)


def test_checkpoint_loader_accepts_another_architecture_factory(tmp_path):
    torch.set_num_threads(2)
    raw = example.make_demo_model()
    checkpoint = tmp_path / "demo.pth"
    torch.save(raw.state_dict(), checkpoint)
    del raw
    adapted = example.load_model(checkpoint, model_factory=example.make_demo_model, core_size=64)
    assert tuple(adapted.tensor_ee_input_shape) == (2, 288, 288)
    support = analyze_receptive_field(adapted)
    assert (support.safe_halo, support.alignment) == (112, 8)


@pytest.mark.parametrize("core", [None, 64])
def test_standard_plan_mode_never_builds_an_inference_graph(core, monkeypatch, capsys):
    def refuse(*args, **kwargs):
        pytest.fail("--plan must only adapt the 64x64 probe and compute rectangles")

    for name in ("build_prediction", "compile_tiled_model", "planned_manifest",
                 "demo_graph_report", "offline_signatures", "adapt_for_region", "initialize"):
        monkeypatch.setattr(example, name, refuse)
    with offline_earth_engine() as calls:
        example.main(["--plan"] + ([] if core is None else ["--core-size", str(core)]))
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "offline_plan_only"
    assert "graph_bytes" not in report
    if core is None:
        assert report == documented_report("standard_unet_resnet34")
    else:
        assert report["context_shapes_nchw"] == [[1, 2, 1152, 1152]]
        assert report["global_dimensions_wh"] == [1216, 1152]
        assert report["roi_dimensions_wh"] == [128, 64]
        assert report["halo"] == [544, 544] and report["alignment"] == 32
    assert not calls


@pytest.mark.parametrize("destination", ["asset", "drive"])
def test_export_preserves_roi_grid_without_submission(model_and_plan, monkeypatch, destination):
    _, plan = model_and_plan
    recorded = {}

    def capture(**kwargs):
        recorded.update(kwargs)
        return "unstarted"

    with offline_earth_engine() as calls:
        payload = ee.Image.constant([0] * 6)
        name = "toAsset" if destination == "asset" else "toDrive"
        monkeypatch.setattr(ee.batch.Export.image, name, capture)
        options = ({"asset_id": "projects/example-project/assets/output"} if destination == "asset"
                   else {"drive_folder": "example-folder"})
        assert example.export_task(payload, plan, **options) == "unstarted"
        assert recorded["crs"] == plan.roi_grid.crs
        assert recorded["crsTransform"] == list(plan.roi_grid.transform)
        assert recorded["region"].serialize() == plan.roi_grid.geometry().serialize()
        assert recorded["maxPixels"] == plan.roi.pixels
        assert "scale" not in recorded and "dimensions" not in recorded
        if destination == "asset":
            assert recorded["overwrite"] is False
            assert recorded["pyramidingPolicy"] == {".default": "sample"}
    assert not calls


def test_generic_upload_uses_checkpoint_factory_and_folded_weights(tmp_path, monkeypatch):
    model = nn.Sequential(nn.Conv2d(2, 2, 3, padding=1), nn.BatchNorm2d(2)).eval()
    checkpoint = tmp_path / "state.pth"
    torch.save(model.state_dict(), checkpoint)
    module = ModuleType("local_checkpoint_factory")

    def load(path):
        assert path == checkpoint
        loaded = nn.Sequential(nn.Conv2d(2, 2, 3, padding=1), nn.BatchNorm2d(2)).eval()
        loaded.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"), strict=True)
        return loaded

    module.load = load
    monkeypatch.setitem(sys.modules, module.__name__, module)
    initialized = []
    monkeypatch.setattr(uploader, "initialize", lambda *args: initialized.append(args))
    uploads = []
    monkeypatch.setattr(uploader, "export_weights", lambda *args, **kwargs: uploads.append(args))
    manifest = tmp_path / "manifest.json"
    uploader.main(["--project", "example-project", "--checkpoint", str(checkpoint),
                   "--factory", "local_checkpoint_factory:load", "--manifest", str(manifest)])
    assert initialized == [("example-project", "ee")]
    parameters, namespace, path = uploads[0]
    assert [p.sha256 for p in parameters] == [p.sha256 for p in effective_convolutions(model)]
    assert len(parameters) == 1 and "+" in parameters[0].source
    assert namespace.startswith("projects/example-project/assets/tensor_ee_weights_")
    assert path == manifest
    manifest.write_text("{}")
    with pytest.raises(SystemExit):
        uploader.main(["--project", "example-project", "--checkpoint", str(checkpoint),
                       "--factory", "local_checkpoint_factory:load", "--manifest", str(manifest)])
    assert len(uploads) == 1


def test_live_mode_wires_checkpoint_manifest_and_one_submission(model_and_plan, tmp_path, monkeypatch):
    model, _ = model_and_plan
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(example.planned_manifest(model)))
    checkpoint = tmp_path / "model.pth"
    loads, initializations, starts = [], [], []
    monkeypatch.setattr(example, "load_model", lambda path, **kwargs: loads.append((path, kwargs)) or model)
    monkeypatch.setattr(example, "initialize", lambda *args: initializations.append(args))
    task = SimpleNamespace(id="offline-task", start=lambda: starts.append(True))
    monkeypatch.setattr(example, "export_task", lambda *args, **kwargs: task)
    with offline_earth_engine() as calls:
        example.main(["--project", "example-project", "--checkpoint", str(checkpoint),
                      "--manifest", str(manifest), "--image", "projects/example-project/assets/input",
                      "--asset-id", "projects/example-project/assets/output", "--core-size", "64"])
    assert loads == [(checkpoint, {"core_size": 64})]
    assert initializations == [("example-project", "ee")]
    assert starts == [True]
    assert not calls
