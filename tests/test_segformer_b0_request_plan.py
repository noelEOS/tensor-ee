"""Focused request-planning checks; no reference forward or transport."""

import hashlib
import json
import math
import sys
from types import SimpleNamespace

import ee
import numpy as np
import pytest
from googleapiclient.model import JsonModel

from tensor_ee.layouts import boundaries
from tensor_ee.models.segformer_b0.configuration import COUNTER, parameter_shapes
from tensor_ee.weights.assets import ConvolutionWeights
from tools import segformer_b0_request_plan as plan
from tools.fixtures.segformer_b0_fullmodel import reject_negative_zero
from tools.live_fixtures import offline_earth_engine


def test_flat_views_cover_fixed_source_schema_once():
    schema = {name: shape for name, shape in parameter_shapes().items() if name != COUNTER}
    counts = {}
    for name, shape in schema.items():
        size = math.prod(shape)
        views = plan.array_segments(size)
        counts[name] = len(views)
        assert views[0]["start"] == 0 and views[-1]["end"] == size
        assert all(left["end"] == right["start"] for left, right in zip(views, views[1:]))
        assert sum(v["rows"] * v["columns"] for v in views) == size
        assert all(0 < v["rows"] <= 1024 and 0 < v["columns"] <= 256 for v in views)
    assert len(schema) == 191
    assert sum(counts.values()) == 192
    assert {name for name, count in counts.items() if count > 1} == {"encoder.patch_embed4.proj.weight"}
    assert plan.array_segments(368640) == [
        {"start": 0, "end": 262144, "rows": 1024, "columns": 256},
        {"start": 262144, "end": 368640, "rows": 416, "columns": 256},
    ]


@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_flat_views_refuse_bad_counts(bad):
    with pytest.raises(ValueError):
        plan.array_segments(bad)


def test_sdk_wire_fingerprint_is_installed_serializer():
    body = {"assetId": "projects/example-project/assets/placeholder", "note": "\N{SNOWMAN}", "x": [1, 2.0]}
    serialized = JsonModel().serialize(body).encode("utf-8")
    measured = plan._fingerprint(body)
    assert measured["bytes"] == len(serialized)
    assert measured["sha256"] == hashlib.sha256(serialized).hexdigest()
    assert measured["compact_bytes"] == len(json.dumps(body, separators=(",", ":")).encode())
    with pytest.raises(ValueError):
        plan._fingerprint({"bad": float("nan")})


def test_temporary_recursion_allowance_restores_on_failure():
    before = sys.getrecursionlimit()
    with pytest.raises(RuntimeError):
        with plan.deep_serialization():
            assert sys.getrecursionlimit() >= 20000
            raise RuntimeError("deliberate")
    assert sys.getrecursionlimit() == before


def test_negative_zero_is_refused_without_mutating_bits():
    values = np.array([0.0, -0.0, 1.0], dtype=np.float32)
    before = values.view(np.uint32).copy()
    with pytest.raises(ValueError, match="negative zero"):
        reject_negative_zero({"coefficient": values})
    np.testing.assert_array_equal(values.view(np.uint32), before)
    reject_negative_zero({"positive_zero": values[[0, 2]], "counter": np.array(0, dtype=np.int64)})


def test_external_grid_and_pure_request_envelopes():
    with offline_earth_engine() as attempts:
        grid = plan._grid_view(plan.SOURCE_GRID)
        image = ee.Image.constant([1.0, 2.0]).rename(["b0", "b1"]).toFloat()
        compute = plan.measure_compute_pixels(image, plan.SOURCE_GRID, ("b0", "b1"))
        export = plan.measure_image_export(image, plan.SOURCE_GRID, plan.PLACEHOLDER_NAMESPACE, "input")
        assert compute["grid"] == export["grid"] == grid.pixel_grid()
        assert compute["grid"]["affineTransform"] == {
            "scaleX": 2.5, "shearX": 0, "translateX": 1000,
            "shearY": 0, "scaleY": -2.5, "translateY": 2000,
        }
        assert compute["band_names"] == ["b0", "b1", "valid"]
        assert compute["values"] == 224 * 224 * 3
        assert compute["expanded_verification_limits_pass"]
        assert not compute["standing_fixture_violations"] == []
        assert export["pyramiding_policy"] == "sample" and export["overwrite"] is False
        assert export["within_2mb"]
    assert attempts == []


def test_complete_table_wire_and_provider_shapes():
    # Exercise transport mapping itself, before model transposes/repetitions.
    state = {"conv.weight": np.arange(6, dtype=np.float32).reshape(2, 3),
             "conv.bias": np.array([1, 2], dtype=np.float32),
             "norm.weight": np.array([1, 2, 3], dtype=np.float32),
             COUNTER: np.array(0, dtype=np.int64)}
    with offline_earth_engine() as attempts:
        records, _ = plan.records_and_locations({}, {}, {k: v for k, v in state.items() if k != COUNTER})
        values = list(records.values())
        assert all(isinstance(v, ConvolutionWeights) for v in values)
        shards = plan.plan_weight_shards(values, plan.PLACEHOLDER_NAMESPACE + "_weights")
        assert len(shards) == 1 and shards[0]["rows"] == 3
        assert shards[0]["export_table"]["bytes"] > shards[0]["collection_payload_bytes"]
        digest = plan.state_sha256(state)
        manifest = plan.planned_manifest(values, shards, plan.PLACEHOLDER_NAMESPACE + "_weights", digest)
        server = plan.asset_state(state, manifest, digest)
        assert server["conv.weight"].shape == (2, 3)
        assert server["conv.bias"].shape == (2,)
        assert server["norm.weight"].shape == (3,)
        assert manifest["planning_only"] and not manifest["assets_exist"]
        assert manifest["shards"][0]["status"] == "planned_not_submitted"
    assert attempts == []


def test_weight_shards_split_complete_envelope_without_losing_rows(monkeypatch):
    rows = [{"properties": {"layer": "tensor", "row": i}} for i in range(8)]
    seen = []
    monkeypatch.setattr(plan, "upload_chunks", lambda *args, **kwargs: iter([(rows, 1)]))

    def request(part, namespace, index):
        seen.append((index, [v["properties"]["row"] for v in part]))
        return {"expression": {"rows": [v["properties"]["row"] for v in part]},
                "assetId": f"{namespace}_part_{index:04d}", "requestId": plan.REQUEST_ID,
                "envelope_padding": "x" * (24000 * len(part))}

    monkeypatch.setattr(plan, "_table_request", request)
    shards = plan.plan_weight_shards([], "projects/example/assets/weights", 100000)
    assert [s["rows"] for s in shards] == [4, 4]
    assert [s["index"] for s in shards] == [0, 1]
    assert all(s["export_table"]["bytes"] <= 100000 for s in shards)
    assert seen == [(0, list(range(8))), (0, list(range(4))), (1, list(range(4, 8)))]


def test_stage1_split_executes_exact_existing_function_sequence(monkeypatch):
    """Spy actual cut wiring without evaluating any native/model arithmetic."""
    calls = []
    ops, input_feature = object(), object()
    grid = plan.SEGFORMER_B0.stage_grids[0]
    features = {name: SimpleNamespace(grid=grid, channels=32, window=7, image=name)
                for name in ("embedding", "block0", "saved", "block1", "norm1")}
    constants = SimpleNamespace(encoder=SimpleNamespace(
        embeddings=("embedding_constants",), blocks=(("block0_constants", "block1_constants"),),
        norms=("norm1_constants",)))
    monkeypatch.setattr(plan, "WindowOps", lambda ee_arg: ops)
    monkeypatch.setattr(plan, "_boundary_image", lambda name, *args: f"{name}_image")

    def input_windows(ee_arg, raster):
        assert raster.grid is plan.SOURCE_GRID and raster.bands == ("b0", "b1")
        calls.append(("input_windows", raster.image))
        return input_feature

    def embedding(actual_ops, feature, coefficients):
        assert actual_ops is ops and feature is input_feature
        calls.append(("overlap_embedding", coefficients))
        return {"out": features["embedding"]}

    def block(actual_ops, feature, coefficients):
        assert actual_ops is ops
        calls.append(("block", feature.image, coefficients))
        if coefficients == "block0_constants":
            assert feature is features["embedding"]
            return {"out": features["block0"]}
        assert coefficients == "block1_constants" and feature is features["saved"]
        return {"out": features["block1"]}

    def norm(actual_ops, feature, coefficients):
        assert actual_ops is ops and feature is features["block1"]
        calls.append(("channel_norm", coefficients))
        return features["norm1"]

    def boundary(name, *args):
        calls.append(("boundary_window", name))
        assert name == "S1B0"
        return features["saved"]

    def dense(ee_arg, actual_grid, image, channels, *, window):
        assert actual_grid == grid and channels == 32 and window == 7
        calls.append(("to_dense", image))
        return image

    monkeypatch.setattr(plan, "input_windows", input_windows)
    monkeypatch.setattr(plan, "overlap_embedding", embedding)
    monkeypatch.setattr(plan, "block", block)
    monkeypatch.setattr(plan, "channel_norm", norm)
    monkeypatch.setattr(plan, "_boundary_window", boundary)
    # build_cut's stage outputs call boundary_raster (X02a), which looks to_dense up there.
    monkeypatch.setattr(boundaries, "to_dense", dense)
    prefix, required, destination = plan.build_cut("stage1a", constants, "asset-reference", {}, "placeholder")
    assert required == ["input"] and destination == "S1B0"
    assert prefix.grid == grid and prefix.image == "block0" and len(prefix.bands) == 32
    assert calls == [("input_windows", "input_image"), ("overlap_embedding", "embedding_constants"),
                     ("block", "embedding", "block0_constants"), ("to_dense", "block0")]
    calls.clear()
    suffix, required, destination = plan.build_cut("stage1b", constants, "asset-reference", {}, "placeholder")
    assert required == ["S1B0"] and destination == "C1"
    assert suffix.grid == grid and suffix.image == "norm1" and len(suffix.bands) == 32
    assert calls == [("boundary_window", "S1B0"), ("block", "saved", "block1_constants"),
                     ("channel_norm", "norm1_constants"), ("to_dense", "norm1")]


def test_split_cut_costs_partition_original_stage_and_plan_counts():
    prefix, suffix, original = (plan.cut_constant_costs(name) for name in ("stage1a", "stage1b", "stage1"))
    assert {key: prefix[key] + suffix[key] for key in original} == original
    assert plan.primary_producers("split-stage1")["S1B0"] == "stage1a"
    assert plan.primary_producers("split-stage1")["C1"] == "stage1b"
    assert len(plan.primary_cuts("split-stage1")) == 6
    assert 192 + 2 + 2 * len(plan.primary_producers("split-stage1")) == 204
    assert 192 + 2 + 2 * len(plan.primary_producers("coarse")) == 202
    grid, channels, names = plan._boundary_spec("S1B0")
    assert (grid.height, grid.width, channels) == (56, 56, 32)
    assert names == tuple(f"c{i}" for i in range(32))
