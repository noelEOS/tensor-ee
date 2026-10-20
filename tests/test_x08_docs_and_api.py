"""X08: the converter layer's public API, the request sizer and the guide (docs/converter.md).

- **Exports.** ``tensor_ee.converter.__all__`` keeps its ten names and adds the Phase 3 API,
  each name the same object as at its module. ``convert``, ``build`` and ``plan`` stay at their
  module paths: ``from tensor_ee.converter import build, plan`` still returns the submodules.
- **R2, the request sizer.** ``request_size`` is ``len(wire(export_request(...)))``. On
  SegFormer-B0's default plan with X00's parameters and the sizing tool's placeholder namespace,
  the input export and the four stages are 1,222,596, 2,062,548, 1,166,058, 874,385 and
  464,261 bytes, the tool's measurements, which now call the sizer. Under a namespace as long as
  X06's run namespace (the record redacts the project ID, not its length), each request is the
  record's ``request_bytes`` exactly: 11 bytes fewer per asset reference (1, 4, 4, 7 and 10).
- **The planner's default measurement** is the sizer, naming each export as ``prepare`` does:
  the default plan is X05's (plan SHA-256 ``6d8fe1fd…``), and ``prepare`` records the planned
  bytes for every stage export. A module-path name is sized under its same-length asset name.
- **The guide.** Its Python blocks, run in order in a fresh interpreter, complete offline.

The SegFormer-B0 checks need SMP 0.5.0 and timm, and skip by name without them, as G1-ext does.
Offline only: graphs are built against ``ee.apitestcase`` signatures with every cloud entry point
refusing; nothing is evaluated, and a model's forward code runs only under fake-tensor tracing.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import ee
from googleapiclient.model import JsonModel
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference.segformer_b0.block_graph import BOUNDARIES, SOURCE_GRID, segformer_b0_block_graph  # noqa: E402
import tensor_ee.converter as converter  # noqa: E402
from tensor_ee.converter.graph import BlockGraph  # noqa: E402
from tensor_ee.converter.plan import Limits, asset_name, plan  # noqa: E402
from tensor_ee.converter.request import (SIZING_REQUEST_ID, deep_recursion, export_request,  # noqa: E402
                                         request_size, wire)
from tensor_ee.converter.run import Caps, prepare  # noqa: E402
from tensor_ee.core.tensor import Raster, ServerArray  # noqa: E402
from tools import graph_identity as GI  # noqa: E402
from tools import segformer_b0_request_plan as P  # noqa: E402
from tools.live_fixtures import offline_earth_engine  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GRAPH = segformer_b0_block_graph()
NS = P.PLACEHOLDER_NAMESPACE
RECORD = json.loads((ROOT / "docs" / "evidence" / "x06-live.json").read_text())
GUIDE = ROOT / "docs" / "converter.md"
EDGES = {"input": "input", "C1": "encoder.norm1", "C2": "encoder.norm2", "C3": "encoder.norm3", "logits": "logits"}
"""The default plan's five exports, as X06 labels them: the input export and the four stages'."""
PLACEHOLDER_BYTES = {"input": 1_222_596, "C1": 2_062_548, "C2": 1_166_058, "C3": 874_385, "logits": 464_261}
"""The brief's committed figures: the sizing tool's measurements with its placeholder namespace."""
SMP_REASON = GI.ext_unavailable("segformer_b0", ("segmentation_models_pytorch", "timm"))
needs_smp = pytest.mark.skipif(SMP_REASON is not None, reason=f"SegFormer-B0 not built: {SMP_REASON}")

OLD_EXPORTS = {"compile_model": "tensor_ee.compiler", "batchnorm_predecessor": "tensor_ee.semantics",
               "PixelWindow": "tensor_ee.tiling", "plan_tiles": "tensor_ee.tiling",
               "compile_tiled_model": "tensor_ee.tiling", "analyze_receptive_field": "tensor_ee.receptive_field",
               "effective_convolutions": "tensor_ee.weight_assets",
               "MatchedWeightProvider": "tensor_ee.weight_mapping", "load_onnx_model": "tensor_ee.onnx_import",
               "check_compatibility": "tensor_ee.compatibility"}
PHASE3_EXPORTS = {"Conversion": "tensor_ee.converter.convert", "ConversionReport": "tensor_ee.converter.convert",
                  "NodeCheck": "tensor_ee.converter.convert", "BlockGraph": "tensor_ee.converter.graph",
                  "Edge": "tensor_ee.converter.graph", "Node": "tensor_ee.converter.graph",
                  "Stage": "tensor_ee.converter.graph", "build_stage": "tensor_ee.converter.build",
                  "Limits": "tensor_ee.converter.plan", "Plan": "tensor_ee.converter.plan",
                  "PlannedStage": "tensor_ee.converter.plan", "request_size": "tensor_ee.converter.request"}


# ---------------------------------------------------------------------------
# The public API
# ---------------------------------------------------------------------------
def test_the_converter_namespace_exports_the_phase3_api_and_keeps_its_names():
    assert converter.__all__[:len(OLD_EXPORTS)] == list(OLD_EXPORTS)
    assert sorted(converter.__all__) == sorted({**OLD_EXPORTS, **PHASE3_EXPORTS})
    for name, module in {**OLD_EXPORTS, **PHASE3_EXPORTS}.items():
        assert getattr(converter, name) is getattr(importlib.import_module(module), name), name
        assert getattr(converter, name).__module__ == module, name


def test_the_functions_named_like_their_modules_stay_at_their_module_paths():
    from tensor_ee.converter import build, convert, plan as planner, request, run
    for module in (build, convert, planner, request, run):
        assert type(module).__name__ == "module"
    assert not {"convert", "build", "plan", "run"} & set(converter.__all__)
    assert callable(convert.convert) and callable(build.build) and callable(planner.plan) and callable(run.run)


def test_the_tool_keeps_its_names_as_the_librarys():
    assert P.deep_serialization is deep_recursion
    assert P.REQUEST_ID == SIZING_REQUEST_ID


# ---------------------------------------------------------------------------
# The sizer, offline on a small image
# ---------------------------------------------------------------------------
def _small_image():
    return ee.Image.constant([1.0, 2.0]).rename(["b0", "b1"]).toFloat()


def test_the_sizer_is_the_length_of_the_request_the_runner_submits():
    with offline_earth_engine() as attempts:
        image = _small_image()
        target = {"asset_id": f"{NS}_input", "description": "segformer_b0_input"}
        request = export_request(image, SOURCE_GRID, request_id=SIZING_REQUEST_ID, **target)
        assert request_size(image, SOURCE_GRID, **target) == len(wire(request))
        assert len(wire(request)) == len(JsonModel().serialize(request).encode("utf-8"))
        other = "123e4567-e89b-12d3-a456-426614174000"     # any canonical UUID: 36 characters
        assert request_size(image, SOURCE_GRID, request_id=other, **target) == len(wire(request))
        with pytest.raises(ValueError, match="asset_id"):
            request_size(image, SOURCE_GRID, asset_id=f"{NS}_encoder.norm1", description="segformer_b0_C1")
        with pytest.raises(ValueError, match="description"):
            request_size(image, SOURCE_GRID, asset_id=f"{NS}_C1", description="segformer_b0_encoder.norm1")
    assert attempts == []


def test_the_tool_measures_with_the_sizer_and_a_dotted_name_as_its_asset_name():
    with offline_earth_engine() as attempts:
        image = _small_image()
        tool = P.measure_image_export(image, SOURCE_GRID, NS, "input")["wire"]
        assert tool["bytes"] == request_size(image, SOURCE_GRID, asset_id=f"{NS}_input",
                                             description="segformer_b0_input")
        body = wire(export_request(image, SOURCE_GRID, asset_id=f"{NS}_input", description="segformer_b0_input",
                                   request_id=P.REQUEST_ID))
        assert tool["sha256"] == hashlib.sha256(body).hexdigest()
        # A module-path name is measured under its same-length asset name: the same bytes.
        assert asset_name("encoder.model.layers_0.blocks.1") == "encoder_model_layers_0_blocks_1"
        dotted = P.measure_image_export(image, SOURCE_GRID, NS, "a.b.c")["wire"]["bytes"]
        assert dotted == request_size(image, SOURCE_GRID, asset_id=f"{NS}_a_b_c", description="segformer_b0_a_b_c")
        assert dotted == tool["bytes"]                      # "a.b.c" is as long as "input"
    assert attempts == []


# ---------------------------------------------------------------------------
# The planner's default measurement: refusals before any build
# ---------------------------------------------------------------------------
def test_the_default_measurement_refuses_a_namespace_or_graph_name_it_cannot_name_assets_with(monkeypatch):
    from tensor_ee.converter import plan as planner
    monkeypatch.setattr(planner, "build_stage", lambda *args: pytest.fail("built"))
    placeholders = {name: ServerArray(None, (1,)) for node in GRAPH.nodes for name in node.parameters.values()}
    with pytest.raises(ValueError, match="namespace must be an explicit projects/PROJECT/assets/PREFIX"):
        plan(GRAPH, parameters=placeholders, namespace="segformer_b0", names=BOUNDARIES)
    data = GRAPH.to_json()
    data["name"] = "segformer b0"
    with pytest.raises(ValueError, match="the graph's name must be letters"):
        plan(BlockGraph.from_json(data), parameters=placeholders, namespace=NS, names=BOUNDARIES)
    with pytest.raises(TypeError, match="measure"):
        plan(GRAPH, parameters=placeholders, namespace=NS, measure=2_000_000)


# ---------------------------------------------------------------------------
# R2 on SegFormer-B0: the sizer against the tool, both namespaces, the planner, prepare
# ---------------------------------------------------------------------------
def _run_length_namespace() -> tuple[str, int]:
    """A namespace as long as X06's run namespace, and the bytes it saves per asset reference,
    both read from the record: its project ID is redacted (``YOUR_EE_PROJECT``), not its
    length. The stand-in project is a placeholder of that length."""
    exports = RECORD["run"]["exports"]
    deltas = {(e["plan_bytes_with_placeholder_namespace"] - e["request_bytes"]) / e["namespace_references"]
              for e in exports}
    assert len(deltas) == 1
    delta = int(deltas.pop())
    prefix = RECORD["namespace"].split("/assets/", 1)[1]
    length = len(NS) - delta - len(f"projects//assets/{prefix}")
    project = "example-" + "x" * (length - len("example-"))
    namespace = f"projects/{project}/assets/{prefix}"
    assert len(namespace) == len(NS) - delta and "YOUR_EE_PROJECT" not in namespace
    return namespace, delta


def _parameters(state, namespace):
    """X00's parameters: the seeded state's weights as ServerArrays read from tables planned (not
    uploaded) under ``namespace`` (``tools/graph_identity.py``, ``_segformer_outputs``)."""
    from tensor_ee._swin_parameters import records_and_locations
    from tensor_ee.models.segformer_b0.configuration import COUNTER
    from tools.fixtures.segformer_b0_fullmodel import state_sha256
    digest = state_sha256(state)
    records = list(records_and_locations({}, {}, {k: v for k, v in state.items() if k != COUNTER})[0].values())
    shards = P.plan_weight_shards(records, namespace + "_weights", P.WEIGHT_WIRE_LIMIT)
    manifest = P.planned_manifest(records, shards, namespace + "_weights", digest)
    return P.asset_state(state, manifest, digest), manifest


def _exports(state, namespace, the_plan):
    """The default plan's five export images under ``namespace``: the seeded input, as X06's run
    exported it, and each stage's boundary built from saved inputs."""
    from tools.fixtures.segformer_b0_fullmodel import SEED, representative_boundaries
    server, manifest = _parameters(state, namespace)
    image = P._boundary_image("input", "inline", representative_boundaries(SEED), namespace)
    rasters = {"input": Raster(image, ("b0", "b1"), SOURCE_GRID)}
    for planned in the_plan.stages:
        assets = {edge: f"{namespace}_{BOUNDARIES[edge]}" for edge in planned.stage.inputs}
        rasters.update(converter.build_stage(GRAPH, planned.stage, server, assets))
    return {label: rasters[edge] for label, edge in EDGES.items()}, server, manifest, image


@pytest.fixture(scope="module")
def segformer():
    if SMP_REASON:
        pytest.skip(f"SegFormer-B0 not built: {SMP_REASON}")
    from tools.fixtures.segformer_b0_fullmodel import SEED, reject_negative_zero, seeded_state
    run_namespace, delta = _run_length_namespace()
    with offline_earth_engine() as attempts:
        state = seeded_state(SEED)
        reject_negative_zero(state)
        server, manifest = _parameters(state, NS)
        default = plan(GRAPH, Limits(), parameters=server, namespace=NS, names=BOUNDARIES)
        rasters, _, _, image = _exports(state, NS, default)
        placeholder, tool, references = {}, {}, {}
        for label, raster in rasters.items():
            target = {"asset_id": f"{NS}_{label}", "description": f"segformer_b0_{label}"}
            placeholder[label] = request_size(raster.image, raster.grid, **target)
            tool[label] = P.measure_image_export(raster.image, raster.grid, NS, label)["wire"]["bytes"]
            body = wire(export_request(raster.image, raster.grid, request_id=SIZING_REQUEST_ID, **target))
            references[label] = body.count(NS.encode())
        run_rasters = _exports(state, run_namespace, default)[0]
        run = {label: request_size(raster.image, raster.grid, asset_id=f"{run_namespace}_{label}",
                                   description=f"segformer_b0_{label}") for label, raster in run_rasters.items()}
    assert attempts == []
    return {"state": state, "server": server, "manifest": manifest, "plan": default, "image": image,
            "placeholder": placeholder, "tool": tool, "references": references, "run": run, "delta": delta}


@needs_smp
def test_the_sizer_reproduces_the_tools_bytes_on_the_segformer_stages(segformer):
    assert segformer["placeholder"] == PLACEHOLDER_BYTES
    assert segformer["tool"] == PLACEHOLDER_BYTES


@needs_smp
def test_under_the_run_namespace_each_request_is_the_records_bytes(segformer):
    """11 bytes fewer per asset reference, as x06-live.json records, and the bytes the run sent."""
    exports = {e["label"]: e for e in RECORD["run"]["exports"]}
    assert list(exports) == list(EDGES)
    assert segformer["delta"] == 11
    for label, export in exports.items():
        assert export["plan_bytes_with_placeholder_namespace"] == segformer["placeholder"][label], label
        assert export["namespace_references"] == segformer["references"][label], label
        assert segformer["run"][label] == segformer["placeholder"][label] - 11 * segformer["references"][label]
        assert segformer["run"][label] == export["request_bytes"], label
    assert [segformer["references"][label] for label in EDGES] == [1, 4, 4, 7, 10]


@needs_smp
def test_the_planners_default_measurement_is_the_sizer(segformer):
    default = segformer["plan"]
    assert [(s.stage.inputs, s.stage.outputs) for s in default.stages] == [
        (("input",), ("encoder.norm1",)), (("encoder.norm1",), ("encoder.norm2",)),
        (("encoder.norm2",), ("encoder.norm3",)), (("encoder.norm1", "encoder.norm2", "encoder.norm3"), ("logits",))]
    assert [dict(s.requests) for s in default.stages] == [
        {"encoder.norm1": 2_062_548}, {"encoder.norm2": 1_166_058}, {"encoder.norm3": 874_385}, {"logits": 464_261}]
    # X05's default plan (fix round), and the one x06-live.json records for the placeholder namespace.
    assert default.sha256 == "6d8fe1fdb95bfb73a1062e9e8cfd35a393bd20f6cb50fd87b903663c6ee48234"
    assert default.sha256 == RECORD["plan_result"]["placeholder_namespace_sha256"]
    assert default.single_graph["requests"] == {"logits": 4_247_156}


@needs_smp
def test_prepare_records_the_planned_bytes_for_every_stage_export(segformer, tmp_path):
    ready = copy.deepcopy(segformer["manifest"])
    ready.update(planning_only=False, assets_exist=True)
    for shard in ready["shards"]:
        shard["status"] = "COMPLETED"
    with offline_earth_engine() as attempts:
        prepared = prepare(tmp_path / "run", segformer["plan"], GRAPH, parameters=segformer["server"],
                           manifest=ready, inputs={"input": segformer["image"]}, namespace=NS,
                           caps=Caps(tasks=5, status_calls=40, asset_calls=30, seconds=3600))
    assert attempts == []
    exports = {e["label"]: e["request"] for e in prepared["exports"]}
    assert {label: request["bytes"] for label, request in exports.items()} == PLACEHOLDER_BYTES
    assert [exports[label]["planned_bytes"] for label in ("C1", "C2", "C3", "logits")] == \
        [PLACEHOLDER_BYTES[label] for label in ("C1", "C2", "C3", "logits")]


# ---------------------------------------------------------------------------
# The guide
# ---------------------------------------------------------------------------
PYTHON_BLOCK = re.compile(r"^```python\n(.*?)^```", re.S | re.M)


def test_the_guide_has_python_blocks_that_import_only_from_the_checkout():
    blocks = PYTHON_BLOCK.findall(GUIDE.read_text(encoding="utf-8"))
    assert len(blocks) >= 5
    text = "\n".join(blocks)
    assert "ee.Initialize" not in text and "ee.Authenticate" not in text
    assert "/Users/" not in GUIDE.read_text(encoding="utf-8")


@pytest.mark.skipif(SMP_REASON is not None, reason=f"the guide's models not built: {SMP_REASON}")
def test_the_guides_python_blocks_run_in_order_offline():
    """The blocks as one session in a fresh interpreter from the checkout root, threads capped."""
    script = "\n".join(PYTHON_BLOCK.findall(GUIDE.read_text(encoding="utf-8")))
    environment = dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                       OPENBLAS_NUM_THREADS="2", VECLIB_MAXIMUM_THREADS="2")
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, env=environment, capture_output=True,
                            text=True, timeout=600)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
