"""Asset schema, exact float32 storage, graph construction, and provenance."""
import json

import ee
from ee.apitestcase import ApiTestCase
import numpy as np
import pytest
import torch
from torch import nn

from tensor_ee.weight_assets import (ConvolutionWeights, FeatureCollectionWeightProvider,
                                    collection_from_records, convolution_weights,
                                    effective_convolutions, feature_records,
                                    float32_wire_values, upload_chunks)


def _algorithms(value):
    if isinstance(value, dict):
        result = [value["functionName"]] if "functionName" in value else []
        return result + [name for child in value.values() for name in _algorithms(child)]
    if isinstance(value, list):
        return [name for child in value for name in _algorithms(child)]
    return []


def _manifest(parameter, assets=("projects/test/assets/weights_a", "projects/test/assets/weights_b")):
    return {"format": "native_ee_featurecollection_weights_v2", "status": "ready", "layers": {
        parameter.key: {"shape": list(parameter.shape), "dtype": parameter.dtype,
                        "bias": parameter.bias is not None, "sha256": parameter.sha256,
                        "assets": list(assets)}}}


def test_short_float32_decimal_encoding_restores_exact_values():
    rng = np.random.default_rng(130)
    numbers = rng.standard_normal(10000).astype(np.float32)
    extremes = np.array([np.finfo(np.float32).max, np.finfo(np.float32).tiny,
                         np.nextafter(np.float32(0), np.float32(1)), -0., 0.], dtype=np.float32)
    source = np.concatenate([numbers, extremes])
    restored = np.asarray(float32_wire_values(source), dtype=np.float32)
    np.testing.assert_array_equal(restored.view(np.uint32), source.view(np.uint32))


def test_json_chunks_reconstruct_matrix_rows_and_bias():
    rng = np.random.default_rng(14)
    weights = rng.normal(0, .02, (3, 6912)).astype(np.float32)
    bias = np.array([-.05, .01, .11], dtype=np.float32)
    parameter = ConvolutionWeights("large_conv", weights, bias, "float32", "fixture")
    records = list(feature_records([parameter]))
    assert len(records) == 6
    assert [r["properties"]["index"] for r in records] == list(range(6))
    assert all(r["geometry"]["type"] == "Point" for r in records)
    assert max(len(r["properties"]["weights_json"].encode()) for r in records) < 90_000
    flattened = [number for record in records for number in json.loads(record["properties"]["weights_json"])]
    restored = np.asarray(flattened, dtype=np.float32).reshape(weights.shape)
    np.testing.assert_array_equal(restored.view(np.uint32), weights.view(np.uint32))
    restored_bias = np.array([r["properties"]["bias"] for r in records if r["properties"]["chunk"] == 0], np.float32)
    np.testing.assert_array_equal(restored_bias, bias)


def test_collects_folded_parameters_and_omits_unused_raw_convolution():
    model = nn.Sequential(nn.Conv2d(2, 3, 3, padding=1, bias=False), nn.BatchNorm2d(3),
                          nn.ReLU(), nn.Conv2d(3, 1, 1)).eval()
    with torch.no_grad():
        model[1].weight.copy_(torch.tensor([.7, -1.2, 1.9]))
        model[1].running_mean.copy_(torch.tensor([-.1, .4, .2]))
        model[1].running_var.copy_(torch.tensor([.8, 1.3, .6]))
    parameters = effective_convolutions(model)
    assert [parameter.key for parameter in parameters] == ["_1", "_3"]
    fused = torch.nn.utils.fuse_conv_bn_eval(model[0], model[1])
    np.testing.assert_array_equal(parameters[0].weights, fused.weight.detach().numpy().reshape(3, -1))
    np.testing.assert_array_equal(parameters[0].bias, fused.bias.detach().numpy())
    assert parameters[0].source == "0+1"


def test_weight_collection_requires_finite_eval_tensors():
    model = nn.Sequential(nn.Conv2d(2, 3, 3, padding=1))
    with pytest.raises(ValueError, match="model.eval"):
        effective_convolutions(model)
    model.eval()
    with pytest.raises(ValueError, match="Only float32"):
        effective_convolutions(model.double())
    model.float()
    with torch.no_grad():
        model[0].weight[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        effective_convolutions(model)


class TestWeightAssets(ApiTestCase):
    def setUp(self):
        super().setUp()
        self.module = nn.Conv2d(2, 3, 3, padding=1).eval()
        self.parameter = convolution_weights("conv", self.module)

    def test_graph_references_assets_and_decodes_inside_earth_engine(self):
        provider = FeatureCollectionWeightProvider(_manifest(self.parameter))
        weights, bias = provider("conv", self.module)
        names = _algorithms(json.loads(weights.serialize()))
        for name in ("Collection.loadTable", "Collection.merge", "Collection.filter",
                     "AggregateFeatureCollection.array", "String.decodeJSON", "List.flatten", "Array.reshape",
                     "Array.float"):
            self.assertIn(name, names)
        self.assertNotIn("Array.double", names)
        self.assertNotIn("predict", " ".join(names).lower())
        self.assertIsNotNone(bias)
        self.assertEqual(provider.used_asset_keys, {"conv"})
        self.assertEqual(provider.fallback_keys, set())
        self.assertLess(len(weights.serialize()), 6000)

    def test_manifest_rejects_changed_fixture_even_after_cached_use(self):
        provider = FeatureCollectionWeightProvider(_manifest(self.parameter))
        provider("conv", self.module)
        with torch.no_grad():
            self.module.weight.add_(.01)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            provider("conv", self.module)

    def test_shape_and_dtype_metadata_are_checked(self):
        for name, value in (("shape", [3, 19]), ("dtype", "float64")):
            manifest = _manifest(self.parameter)
            manifest["layers"]["conv"][name] = value
            provider = FeatureCollectionWeightProvider(manifest)
            with self.assertRaisesRegex(ValueError, "shape, dtype"):
                provider("conv", self.module)

    def test_missing_weights_fail_without_explicit_fallback(self):
        provider = FeatureCollectionWeightProvider(_manifest(self.parameter))
        with self.assertRaisesRegex(KeyError, "absent"):
            provider("unfused_diagnostic", self.module)
        provider = FeatureCollectionWeightProvider(_manifest(self.parameter), fallback=True)
        provider("unfused_diagnostic", self.module)
        self.assertEqual(provider.fallback_keys, {"unfused_diagnostic"})

    def test_upload_chunks_obey_actual_graph_size(self):
        rng = np.random.default_rng(8)
        parameter = ConvolutionWeights("large", rng.normal(0, .04, (9, 5000)).astype(np.float32),
                                       None, "float32", "fixture")
        chunks = list(upload_chunks([parameter], max_bytes=100_000))
        self.assertGreater(len(chunks), 1)
        self.assertEqual(sum(len(records) for records, _ in chunks), 18)
        for records, size in chunks:
            self.assertLessEqual(size, 100_000)
            self.assertEqual(size, len(ee.serializer.toJSON(collection_from_records(records)).encode()))

    def test_incomplete_manifest_is_rejected(self):
        manifest = _manifest(self.parameter)
        manifest["status"] = "uploading"
        with self.assertRaisesRegex(ValueError, "completed"):
            FeatureCollectionWeightProvider(manifest)


# ---------------------------------------------------------------------------
# export_weights: a fresh upload, and resuming an interrupted one (mocked tasks)
# ---------------------------------------------------------------------------
class Crash(Exception):
    """The process dying at a chosen save."""


class FakeEarthEngine:
    """Table exports that start with sequential IDs and finish after ``polls`` status calls;
    ``getTaskStatus`` and ``getAsset`` answer from the same record."""

    def __init__(self, polls: int = 1):
        self.polls, self.started, self.calls, self.assets = polls, [], {}, set()
        self.clock = 1_000.0

    def time(self):
        self.clock += 1.0
        return self.clock

    def to_asset(self, collection, description, assetId, overwrite):
        assert overwrite is False
        engine = self

        class Task:
            id = None

            def start(self):
                self.id = f"T{len(engine.started):03d}"
                engine.started.append((self.id, assetId, description))

            def status(self):
                return engine.status(self.id)
        task = Task()
        task.asset_id = assetId
        return task

    def status(self, task_id):
        self.calls[task_id] = self.calls.get(task_id, 0) + 1
        if self.calls[task_id] < self.polls:
            return {"state": "RUNNING", "id": task_id}
        asset = next(a for i, a, _ in self.started if i == task_id)
        self.assets.add(asset)
        return {"state": "COMPLETED", "id": task_id}

    def get_task_status(self, ids):
        return [self.status(ids[0])]

    def get_asset(self, asset_id):
        if asset_id not in self.assets:
            raise ee.EEException(f"Asset '{asset_id}' not found")
        return {"name": asset_id}


@pytest.fixture
def offline():
    case = ApiTestCase()
    case.setUp()
    try:
        yield
    finally:
        case.tearDown()


def _upload_parameters():
    rng = np.random.default_rng(8)
    return [ConvolutionWeights(f"large{i}", rng.normal(0, .04, (6, 5000)).astype(np.float32),
                               rng.normal(0, .04, 6).astype(np.float32) if i else None, "float32", f"fixture{i}")
            for i in range(3)]


def _run(fake, path, *, on_update=None, resume=False, parameters=None):
    import tensor_ee.weight_assets as WA
    from unittest import mock

    with mock.patch.object(ee.batch.Export.table, "toAsset", fake.to_asset), \
            mock.patch.object(ee.data, "getTaskStatus", fake.get_task_status), \
            mock.patch.object(ee.data, "getAsset", fake.get_asset), \
            mock.patch.object(WA.time, "time", fake.time), mock.patch.object(WA.time, "sleep", lambda s: None):
        return WA.export_weights(parameters or _upload_parameters(), "projects/test/assets/w", path,
                                 max_bytes=100_000, poll_seconds=1, on_update=on_update, resume=resume)


def _crash_at(predicate):
    def hook(manifest):
        if predicate(manifest):
            raise Crash()
    return hook


def test_a_fresh_upload_keeps_its_manifest_fields_and_order(offline, tmp_path):
    fake = FakeEarthEngine(polls=2)
    manifest = _run(fake, tmp_path / "m.json")
    assert list(manifest) == ["format", "status", "asset_namespace", "started_unix", "serialization",
                              "dummy_geometry", "max_collection_payload_bytes", "max_active_tasks",
                              "parameter_count", "layers", "shards", "finished_unix", "collection_payload_bytes"]
    assert manifest["status"] == "ready" and len(manifest["shards"]) > 2
    shard = manifest["shards"][0]
    assert list(shard) == ["index", "asset_id", "rows", "layers", "collection_payload_bytes", "status",
                           "started_unix", "task_id", "task_status", "finished_unix"]
    assert [a for _, a, _ in fake.started] == [s["asset_id"] for s in manifest["shards"]]
    assert json.loads((tmp_path / "m.json").read_text()) == manifest


@pytest.mark.parametrize("moment", ["before start", "after start", "mid-poll", "all complete"])
def test_an_interrupted_upload_resumes_to_the_same_manifest(offline, tmp_path, moment):
    reference = _run(FakeEarthEngine(polls=2), tmp_path / "reference.json")
    fake, path = FakeEarthEngine(polls=2), tmp_path / "m.json"
    hooks = {
        "before start": _crash_at(lambda m: len(m["shards"]) == 2 and m["shards"][1]["status"] == "starting"),
        "after start": _crash_at(lambda m: len(m["shards"]) == 2 and m["shards"][1]["status"] == "READY"),
        "mid-poll": _crash_at(lambda m: any(s["status"] == "RUNNING" for s in m["shards"])),
        "all complete": _crash_at(lambda m: m["shards"] and all(s["status"] == "COMPLETED" for s in m["shards"])
                                  and len(m["shards"]) == len(reference["shards"]) and m["status"] == "uploading"),
    }
    with pytest.raises(Crash):
        _run(fake, path, on_update=hooks[moment])
    interrupted = json.loads(path.read_text())
    assert interrupted["status"] == "uploading"
    started = len(fake.started)
    manifest = _run(fake, path, resume=True)
    assert manifest["status"] == "ready"
    assert [s["asset_id"] for s in manifest["shards"]] == [s["asset_id"] for s in reference["shards"]]
    assert all(s["status"] == "COMPLETED" for s in manifest["shards"])
    assert {k: v["assets"] for k, v in manifest["layers"].items()} == {k: v["assets"]
                                                                        for k, v in reference["layers"].items()}
    dispatched = [a for _, a, _ in fake.started]
    assert sorted(dispatched) == sorted(set(dispatched)) == [s["asset_id"] for s in reference["shards"]]
    if moment == "all complete":
        assert len(fake.started) == started                # nothing dispatched again
    if moment == "before start":
        assert "task_id" not in interrupted["shards"][1]  # recorded "starting", dispatched on resume
    if moment in ("after start", "mid-poll"):
        recorded = {s["task_id"] for s in interrupted["shards"] if s.get("task_id")}
        assert recorded <= set(fake.calls)                 # the recorded tasks were polled by ID
    assert _run(fake, path, resume=True) == manifest      # a ready manifest is returned as it is


def test_a_starting_shard_whose_asset_exists_is_complete(offline, tmp_path):
    fake, path = FakeEarthEngine(polls=1), tmp_path / "m.json"
    with pytest.raises(Crash):
        _run(fake, path, on_update=_crash_at(lambda m: len(m["shards"]) == 1 and m["shards"][0]["status"] == "starting"))
    interrupted = json.loads(path.read_text())
    fake.assets.add(interrupted["shards"][0]["asset_id"])  # its export did finish
    manifest = _run(fake, path, resume=True)
    assert manifest["status"] == "ready" and manifest["shards"][0]["resumed"] == "asset found"
    assert interrupted["shards"][0]["asset_id"] not in [a for _, a, _ in fake.started]


def test_resume_refuses_other_parameters_and_failed_shards(offline, tmp_path):
    fake, path = FakeEarthEngine(polls=2), tmp_path / "m.json"
    with pytest.raises(Crash):
        _run(fake, path, on_update=_crash_at(lambda m: any(s["status"] == "RUNNING" for s in m["shards"])))
    other = _upload_parameters()[:2]
    with pytest.raises(ValueError, match="not an upload of these parameters"):
        _run(fake, path, resume=True, parameters=other)
    manifest = json.loads(path.read_text())
    manifest["shards"][0]["status"] = "FAILED"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="FAILED"):
        _run(fake, path, resume=True)


def test_a_start_failure_is_not_masked_by_the_save_callback(offline, tmp_path):
    import tensor_ee.weight_assets as WA
    from unittest import mock

    class Failing:
        id = None

        def start(self):
            raise RuntimeError("quota")

    def cap(manifest):
        if manifest["status"] == "failed":
            raise TimeoutError("polling cap")

    with mock.patch.object(ee.batch.Export.table, "toAsset", lambda *a, **k: Failing()), \
            pytest.raises(RuntimeError, match="quota"):
        WA.export_weights(_upload_parameters(), "projects/test/assets/w", tmp_path / "m.json", max_bytes=100_000,
                          poll_seconds=1, on_update=cap)
    manifest = json.loads((tmp_path / "m.json").read_text())
    assert manifest["status"] == "failed" and manifest["shards"][0]["error"] == "RuntimeError: quota"
