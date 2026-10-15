"""X02a: the block graph, its operations and its builder (docs/phase3-design.md, section 3).

- **R1, the representation.** The hand-written SegFormer-B0 block graph
  (``tests/reference/segformer_b0/block_graph.py``) is the committed JSON; JSON round trips give
  an equal graph and the same hash; the graph names every SMP parameter once and holds no
  value; validation refuses unknown operations, missing inputs, cycles and edges that disagree
  with their operations; every operation is one library function, held by reference.
- **R2, cost.** Node costs come from the blocks' METADATA cost functions without building
  constants, and sum to the model's ``constant_costs()`` (6,620,078 values, 191 asset arrays)
  and to the sizing tool's cost of every cut.
- **R3, the builder.** It builds constants once per node, refuses unbound parameters and
  inputs that are not their edges, and never makes a cloud call.
- **R4, stage boundaries.** The model and the sizing tool call the conventions moved to
  ``tensor_ee/layouts/boundaries.py``, and the model does not load the converter; the model and
  the registry build the decoder's constants with the same library builders.
- **R5, the gate.** Built with the parameters and placeholder namespace X00 uses, the graph and
  its six stages give the seven G1-ext SegFormer-B0 targets byte for byte
  (``docs/evidence/x00-g1ext-golden-graphs.json``). These need SMP 0.5.0 and timm, and skip by
  name without them, as G1-ext does. Where the seeded state's raw inputs differ under a version
  difference, judge names each target a skip (X09); a changed graph's miss is then still shown
  against the unchanged graph's stage built from the same inputs.

Offline only: graphs are built against ``ee.apitestcase`` signatures with every cloud entry
point refusing; nothing is evaluated and no model forward runs.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import importlib
import inspect
import json
from pathlib import Path
import subprocess
import sys

import ee
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference.segformer_b0.block_graph import (BOUNDARIES, MODEL_GRID, SOURCE_GRID, STAGES,  # noqa: E402
                                                segformer_b0_block_graph)
from tensor_ee.blocks.window import segformer as S  # noqa: E402
from tensor_ee.converter import build as BUILD, operations  # noqa: E402
from tensor_ee.converter.graph import (FORMAT, VERSION, BlockGraph, Edge, Node, Stage, canonical,  # noqa: E402
                                      grid_to_json)
from tensor_ee.converter.operations import OPERATIONS  # noqa: E402
from tensor_ee.core.errors import GridMismatch, UnsupportedOperation  # noqa: E402
from tensor_ee.core.grid import TokenGrid  # noqa: E402
from tensor_ee.core.tensor import Raster, ServerArray, WindowTensor  # noqa: E402
from tensor_ee.layouts import boundaries as B  # noqa: E402
from tensor_ee.models.segformer_b0 import decoder as D  # noqa: E402
from tensor_ee.models.segformer_b0.configuration import COUNTER, constant_costs, parameter_shapes  # noqa: E402
from tools import graph_identity as GI  # noqa: E402
from tools import segformer_b0_request_plan as plan  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_JSON = ROOT / "tests" / "reference" / "segformer_b0" / "block_graph.json"
SHA256 = "568c92c33a639257c43578d3c7773f91f98497d4c12191319ff2565a23b0f39a"
GRAPH = segformer_b0_block_graph()
GOLDEN = json.loads(GI.GOLDEN_EXT.read_text())


def _server_state() -> dict:
    """Zero ServerArray placeholders of every parameter's server shape, and the BN counter."""
    state = {COUNTER: np.array(0, dtype=np.int64)}
    for name, shape in parameter_shapes().items():
        if name != COUNTER:
            target = (shape[0], int(np.prod(shape[1:]))) if len(shape) == 4 else shape
            state[name] = ServerArray(ee.Array(ee.List.repeat(0., int(np.prod(shape)))).reshape(list(target)).float(),
                                      target)
    return state


def _input():
    return {"input": Raster(ee.Image.constant([0., 0.]).toFloat().rename(["b0", "b1"]), ("b0", "b1"), SOURCE_GRID)}


def _changed(change) -> dict:
    data = copy.deepcopy(GRAPH.to_json())
    change(data, {n["id"]: n for n in data["nodes"]})
    return data


# ---------------------------------------------------------------------------
# R1: the representation
# ---------------------------------------------------------------------------
def test_the_hand_written_graph_is_the_committed_json():
    assert FIXTURE_JSON.read_text() == GRAPH.dumps(indent=1) + "\n"
    assert BlockGraph.loads(FIXTURE_JSON.read_text()) == GRAPH
    assert GRAPH.sha256 == SHA256
    expected = ["input_windows"]
    for s in range(1, 5):
        expected += [f"encoder.patch_embed{s}", f"encoder.block{s}.0", f"encoder.block{s}.1", f"encoder.norm{s}"]
    assert [n.id for n in GRAPH.nodes] == expected + ["decoder.pyramid", "decoder.head", "logits"]
    assert GRAPH.inputs == ("input",) and GRAPH.outputs == ("logits",)


def test_json_round_trip_is_equal_with_the_same_hash():
    data = GRAPH.to_json()
    assert data["format"] == FORMAT and data["version"] == VERSION
    for text in (GRAPH.dumps(), GRAPH.dumps(indent=1)):
        again = BlockGraph.loads(text)
        assert again == GRAPH and again.sha256 == GRAPH.sha256 and again.dumps() == GRAPH.dumps()
    # The order nodes are given in does not matter: they are kept in one canonical order.
    shuffled = BlockGraph(GRAPH.name, GRAPH.inputs, GRAPH.outputs, dict(reversed(list(GRAPH.edges.items()))),
                          tuple(reversed(GRAPH.nodes)))
    assert shuffled == GRAPH and shuffled.dumps() == GRAPH.dumps()
    assert segformer_b0_block_graph() == GRAPH and hash(segformer_b0_block_graph()) == hash(GRAPH)
    # A validated graph cannot change under its hash.
    for mapping in (GRAPH.edges, GRAPH.nodes[0].settings, GRAPH.nodes[1].parameters):
        with pytest.raises(TypeError):
            mapping["x"] = None
    with pytest.raises(ValueError, match="version 1"):
        BlockGraph.from_json({**data, "version": 2})


def test_the_graph_names_every_parameter_once_and_holds_no_value():
    names = [name for node in GRAPH.nodes for name in node.parameters.values()]
    assert len(names) == len(set(names)) == 191
    assert set(names) == set(parameter_shapes()) - {COUNTER}

    def scalars(value):
        if isinstance(value, dict):
            return [s for v in value.values() for s in scalars(v)]
        if isinstance(value, list):
            return [s for v in value for s in scalars(v)]
        return [value]
    # Settings and edges are a few small literals; no array of coefficients is anywhere.
    assert len(scalars(GRAPH.to_json())) < 1500
    assert all(isinstance(v, str) for node in GRAPH.to_json()["nodes"] for v in node["parameters"].values())


def test_int_and_float_grid_literals_are_different_edges():
    assert Edge("W", TokenGrid(56, 56, 10), 32, 7) != Edge("W", TokenGrid(56, 56, 10.0), 32, 7)
    data = _changed(lambda d, n: d["edges"]["encoder.norm1"]["grid"].update(scale=10))
    with pytest.raises(GridMismatch, match="encoder.norm1"):
        BlockGraph.from_json(data)


def _unknown(d, n):
    n["encoder.norm1"]["op"] = "layer_norm_by_another_name"


def _missing_input(d, n):
    n["encoder.patch_embed2"]["inputs"] = ["encoder.nowhere"]


def _cycle(d, n):
    n["encoder.patch_embed1"]["inputs"] = ["encoder.block1.0"]


def _extra_edge(d, n):
    d["edges"]["spare"] = d["edges"]["encoder.norm1"]


def _written_twice(d, n):
    n["encoder.block1.1"]["output"] = "encoder.block1.0"


REFUSALS = {
    "unknown operation": (_unknown, UnsupportedOperation, "unknown operation"),
    "missing input": (_missing_input, ValueError, "neither a graph input nor a node's output"),
    "cycle": (_cycle, ValueError, "cycle"),
    "undescribed or unwritten edge": (_extra_edge, ValueError, "neither a graph input nor a node's output"),
    "edge written twice": (_written_twice, ValueError, "written twice"),
    "graph input written by a node": (lambda d, n: d["inputs"].append("encoder.norm4"), ValueError,
                                      "written twice"),
    "layout": (lambda d, n: d["edges"]["encoder.block1.0"].update(layout="R", window=None,
                                                                  bands=[f"c{i}" for i in range(32)]),
               GridMismatch, "encoder.block1.0"),
    "grid": (lambda d, n: d["edges"]["encoder.block2.0"]["grid"].update(height=35), GridMismatch, "block2.0"),
    "channels": (lambda d, n: d["edges"]["encoder.block1.0"].update(channels=33), GridMismatch, "block1.0"),
    "window": (lambda d, n: d["edges"]["encoder.norm3"].update(window=8), GridMismatch, "norm3"),
    "consumer": (lambda d, n: n["encoder.patch_embed2"]["settings"].update(cin=64), GridMismatch, "64 channels"),
    "setting value": (lambda d, n: n["encoder.block3.0"]["settings"].update(ratio=3), UnsupportedOperation,
                      "SR ratio"),
    "setting name": (lambda d, n: n["encoder.norm2"]["settings"].update(elementwise_affine=True),
                     UnsupportedOperation, "settings are"),
    "parameter role": (lambda d, n: n["encoder.norm2"]["parameters"].pop("bias"), UnsupportedOperation,
                       "reads parameters"),
    "shared parameter": (lambda d, n: n["encoder.norm2"]["parameters"].update(bias="encoder.norm1.bias"),
                         UnsupportedOperation, "bound by nodes"),
    "arity": (lambda d, n: n["decoder.pyramid"]["inputs"].append("encoder.norm1"), ValueError, "reads 4 edges"),
    "mapping setting": (lambda d, n: n["input_windows"]["settings"].update(window={"size": 7}), ValueError,
                        "'input_windows' setting window: a mapping setting must be a grid"),
    "indivisible grid": (lambda d, n: n["encoder.block2.0"]["settings"].update(ratio=8), GridMismatch,
                         "node 'encoder.block2.0': mit_block: a 28 x 28 grid does not divide by 8"),
    "malformed grid": (lambda d, n: n["logits"]["settings"]["grid"].pop("transform"), ValueError,
                       "'logits' setting grid: malformed grid"),
}


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_validation_refuses(case):
    change, error, message = REFUSALS[case]
    with pytest.raises(error, match=message):
        BlockGraph.from_json(_changed(change))


def test_an_unread_edge_is_refused():
    data = _changed(lambda d, n: d["outputs"].append("encoder.norm4"))
    assert BlockGraph.from_json(data).outputs == ("logits", "encoder.norm4")
    data["nodes"] = [n for n in data["nodes"] if n["id"] != "logits"]
    del data["edges"]["logits"]
    data["outputs"] = ["encoder.norm4"]
    with pytest.raises(ValueError, match=r"\['decoder.head'\] are never read"):
        BlockGraph.from_json(data)


def test_every_operation_is_one_library_function_held_by_reference():
    """Every entry, SegFormer's and those later lanes add (X04): one window block or a boundary
    relabel, with its block's METADATA; constants exactly where it reads parameters."""
    for name, op in OPERATIONS.items():
        assert op.name == name
        module = importlib.import_module(op.function.__module__)
        assert getattr(module, name) is op.function
        boundaries = op.function.__module__ == "tensor_ee.layouts.boundaries"
        assert boundaries or op.function.__module__.startswith("tensor_ee.blocks.window.")
        assert op.metadata is (None if boundaries else module.METADATA[name])
        assert (op.constants is None) == (op.roles is operations._no_roles)
    # Explicit: no operation is looked up by name at run time.
    assert "getattr" not in inspect.getsource(operations) and "import_module" not in inspect.getsource(operations)
    assert {n.op for n in GRAPH.nodes} <= set(OPERATIONS)


def test_the_model_and_the_registry_build_the_decoder_constants_with_the_library_builders():
    hand_built = {"_linear_constants", "_parameter", "conv1x1_constants", "BatchNormConstants", "LinearConstants"}
    for function, builder in ((D.decoder_constants, "project_pyramid_constants"),
                              (D.decoder_constants, "fuse_head_constants"),
                              (operations._pyramid_constants, "project_pyramid_constants"),
                              (operations._head_constants, "fuse_head_constants")):
        names = set(function.__code__.co_names)
        assert builder in names and not hand_built & names, function.__qualname__
    assert D.project_pyramid_constants is S.project_pyramid_constants
    assert D.fuse_head_constants is S.fuse_head_constants
    for builder in (S.project_pyramid_constants, S.fuse_head_constants):
        assert builder.__module__ == "tensor_ee.blocks.window.segformer"
        with pytest.raises(UnsupportedOperation):
            builder(None, {}, "decoder", (256,), 256, 2, eps=1e-5) if builder is S.fuse_head_constants \
                else builder(None, {}, "decoder", (256, 160, 64, 32), 256)


# ---------------------------------------------------------------------------
# R2: cost
# ---------------------------------------------------------------------------
def test_costs_equal_the_models_and_the_sizing_tools():
    costs = constant_costs()
    assert GRAPH.cost() == costs["total"] == {"values": 6_620_078, "asset_arrays": 191}
    for stage in STAGES:
        assert GRAPH.cost(stage) == plan.cut_constant_costs(stage.name), stage.name
    by_id = {n.id: GRAPH.node_cost(n) for n in GRAPH.nodes}
    for s in range(1, 5):
        parts = [by_id[i] for i in by_id if i.startswith((f"encoder.patch_embed{s}", f"encoder.block{s}.",
                                                          f"encoder.norm{s}"))]
        assert {k: sum(p[k] for p in parts) for k in ("values", "asset_arrays")} == costs[f"encoder_stage{s}"]
    decoder = [by_id["decoder.pyramid"], by_id["decoder.head"]]
    assert {k: sum(p[k] for p in decoder) for k in ("values", "asset_arrays")} == {
        k: costs["decoder_head"][k] + costs["resize"][k] for k in ("values", "asset_arrays")}
    assert by_id["input_windows"] == by_id["logits"] == {"values": 0, "asset_arrays": 0}


def test_costs_build_no_constants(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("cost built an Earth Engine object")
    for name in ("Array", "Image", "List", "Number"):
        monkeypatch.setattr(ee, name, refuse)
    assert GRAPH.cost()["values"] == 6_620_078


# ---------------------------------------------------------------------------
# R3: the builder
# ---------------------------------------------------------------------------
def test_build_constructs_constants_once_per_node_and_makes_no_cloud_call(monkeypatch):
    calls = []
    for name, op in OPERATIONS.items():
        if op.constants is not None:
            def counted(ops, parameters, prefix, settings, inputs, _build=op.constants):
                calls.append(prefix)
                return _build(ops, parameters, prefix, settings, inputs)
            monkeypatch.setitem(OPERATIONS, name, replace(op, constants=counted))
    with GI.offline_earth_engine() as attempts:
        outputs = BUILD.build(GRAPH, _server_state(), _input())
    assert attempts == []
    assert sorted(calls) == sorted(n.id for n in GRAPH.nodes if n.op not in ("relabel_to_windows",
                                                                             "relabel_from_windows"))
    (logits,) = outputs.values()
    assert set(outputs) == {"logits"} and logits.bands == ("c0", "c1") and logits.grid == SOURCE_GRID


def test_build_refuses_unbound_parameters_and_foreign_inputs():
    with GI.offline_earth_engine():
        state = _server_state()
        del state["encoder.block2.1.attn.kv.bias"]
        with pytest.raises(UnsupportedOperation, match="encoder.block2.1.attn.kv.bias"):
            BUILD.build(GRAPH, state, _input())
        state = _server_state()
        state["encoder.norm1.weight"] = [1.0] * 32
        with pytest.raises(UnsupportedOperation, match="encoder.norm1.weight"):
            BUILD.build(GRAPH, state, _input())
        stage = STAGES[3]                                               # stage2: C1 -> C2
        wrong = WindowTensor(ee.Image(0), TokenGrid(56, 56, 10), 32, 7)    # int scale: another grid
        with pytest.raises(GridMismatch, match="encoder.norm1"):
            BUILD.build(GRAPH, state, {"encoder.norm1": wrong}, outputs=stage.outputs)
        with pytest.raises(GridMismatch, match="input"):
            BUILD.build(GRAPH, state, {"input": Raster(ee.Image(0), ("b1", "b0"), SOURCE_GRID)})
        with pytest.raises(ValueError, match="assets name"):
            BUILD.build_stage(GRAPH, stage, state, {"encoder.norm2": "projects/p/assets/x_C2"})


@pytest.mark.parametrize("inputs, outputs, message", [
    (("encoder.norm2",), ("logits",), "needs graph input 'input'"),
    (("input", "encoder.norm1"), ("encoder.norm1",), "does not read"),
    (("input", "encoder.norm2"), ("encoder.norm1",), "not needed"),
    (("input",), ("nowhere",), "not edges"),
])
def test_stage_nodes_refuse_inconsistent_boundaries(inputs, outputs, message):
    with pytest.raises(ValueError, match=message):
        GRAPH.stage_nodes(inputs, outputs)


def test_stage_nodes_are_the_cuts():
    nodes = {stage.name: [n.id for n in GRAPH.stage_nodes(stage.inputs, stage.outputs)] for stage in STAGES}
    assert nodes["stage1a"] == ["input_windows", "encoder.patch_embed1", "encoder.block1.0"]
    assert nodes["stage1b"] == ["encoder.block1.1", "encoder.norm1"]
    assert nodes["stage3"] == ["encoder.patch_embed3", "encoder.block3.0", "encoder.block3.1", "encoder.norm3"]
    assert nodes["decoder"] == ["decoder.pyramid", "decoder.head", "logits"]
    assert nodes["monolithic"] == [n.id for n in GRAPH.nodes]
    assert set(BOUNDARIES) == {e for stage in STAGES for e in stage.inputs + stage.outputs}


# ---------------------------------------------------------------------------
# R4: stage boundaries, one implementation
# ---------------------------------------------------------------------------
def test_the_old_call_sites_use_the_moved_conventions():
    for name in ("grid_view", "pin_boundary", "saved_image", "boundary_windows", "boundary_raster",
                 "relabel_to_windows", "relabel_from_windows"):
        assert getattr(B, name).__module__ == "tensor_ee.layouts.boundaries", name
    assert plan._grid_view is B.grid_view
    assert D.relabel_to_windows is B.relabel_to_windows and D.relabel_from_windows is B.relabel_from_windows
    uses = {D.input_windows: {"relabel_to_windows"}, D.segformer_b0: {"relabel_from_windows"},
            plan._boundary_image: {"saved_image", "pin_boundary"}, plan._boundary_window: {"boundary_windows"},
            plan.build_cut: {"boundary_raster", "relabel_from_windows"}}
    for function, names in uses.items():
        assert names <= set(function.__code__.co_names), function.__name__
    for function in (D.input_windows, D.segformer_b0, plan._boundary_image, plan._boundary_window, plan.build_cut):
        assert not {"changeProj", "from_dense", "to_dense", "reproject", "updateMask"} & set(
            function.__code__.co_names), function.__name__
    assert not hasattr(plan, "to_dense")


def test_the_model_does_not_load_the_converter():
    probe = ("import sys, tensor_ee.models.segformer_b0.decoder as D; "
             "print(sorted(m for m in sys.modules if m == 'tensor_ee.converter' or m.startswith('tensor_ee.converter.')));"
             "print(D.relabel_to_windows.__module__)")
    result = subprocess.run([sys.executable, "-c", probe], cwd=ROOT, capture_output=True, text=True, timeout=300,
                            env={**__import__("os").environ, "PYTHONPATH": str(ROOT)})
    assert result.returncode == 0, result.stderr
    assert result.stdout.split("\n")[:2] == ["[]", "tensor_ee.layouts.boundaries"]


def test_relabels_refuse_other_sizes_and_band_counts():
    with GI.offline_earth_engine():
        raster = Raster(ee.Image.constant([0., 0.]).rename(["b0", "b1"]), ("b0", "b1"), SOURCE_GRID)
        with pytest.raises(UnsupportedOperation, match="224 x 224 raster"):
            B.relabel_to_windows(ee, raster, TokenGrid(56, 56, 10.0))
        windows = B.relabel_to_windows(ee, raster, MODEL_GRID)
        assert (windows.grid, windows.channels, windows.window) == (MODEL_GRID, 2, 7)
        with pytest.raises(UnsupportedOperation, match="3 band names for 2 channels"):
            B.relabel_from_windows(ee, windows, SOURCE_GRID, ("c0", "c1", "c2"))
        logits = B.relabel_from_windows(ee, windows, SOURCE_GRID, ("c0", "c1"))
        assert logits.bands == ("c0", "c1") and logits.grid == SOURCE_GRID


# ---------------------------------------------------------------------------
# R5: the seven G1-ext SegFormer-B0 targets, built by the builder
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def x00_parameters():
    """The seeded state's weights as X00 reads them: ServerArrays of tables planned under the
    placeholder namespace (``tools/graph_identity.py``, ``_segformer_outputs``)."""
    reason = GI.ext_unavailable("segformer_b0", ("segmentation_models_pytorch", "timm"))
    if reason:
        pytest.skip(f"SegFormer-B0 targets not built: {reason}")
    from tensor_ee._swin_parameters import records_and_locations
    from tools.fixtures.segformer_b0_fullmodel import SEED, reject_negative_zero, seeded_state, state_sha256
    namespace = plan.PLACEHOLDER_NAMESPACE
    with GI.offline_earth_engine() as attempts:
        state = seeded_state(SEED)
        reject_negative_zero(state)
        digest = state_sha256(state)
        records = list(records_and_locations({}, {}, {k: v for k, v in state.items() if k != COUNTER})[0].values())
        shards = plan.plan_weight_shards(records, namespace + "_weights", plan.WEIGHT_WIRE_LIMIT)
        manifest = plan.planned_manifest(records, shards, namespace + "_weights", digest)
        server = plan.asset_state(state, manifest, digest)
    assert attempts == []
    return server, {"sha256": digest, "tensors": len(state)}


@pytest.mark.parametrize("stage", STAGES, ids=[stage.name for stage in STAGES])
def test_the_builder_gives_the_g1ext_segformer_targets(stage, x00_parameters):
    server, inputs = x00_parameters
    assets = {edge: f"{plan.PLACEHOLDER_NAMESPACE}_{BOUNDARIES[edge]}" for edge in stage.inputs}
    with GI.offline_earth_engine() as attempts:
        (output,) = BUILD.build_stage(GRAPH, stage, server, assets).values()
        with plan.deep_serialization():
            record = GI.raster_record(output, inputs=inputs)
    assert attempts == []
    name = f"segformer_b0/{stage.name}"
    verdict, message = GI.judge(name, GOLDEN["graphs"][name], {"kind": "fixture", **record}, GOLDEN["environment"])
    if verdict == "skip":
        pytest.skip(message)
    assert verdict == "pass", message
    assert record["sha256"] == GOLDEN["graphs"][name]["sha256"]


def test_a_changed_block_graph_misses_its_target(x00_parameters):
    """The gate bites. The changed graph's stage is judged against the unchanged graph's, built
    here from the same inputs (a fail in any environment), and against the golden target: a fail
    wherever judge can judge it. Where the seeded state's raw inputs differ from the golden ones
    under a version difference (the hosted CI's versions), judge names that comparison a skip,
    and the first check still shows the miss."""
    server, inputs = x00_parameters
    changed = BlockGraph.from_json(_changed(lambda d, n: n["encoder.norm1"]["settings"].update(eps=1e-5)))
    stage = STAGES[2]                                                    # stage1b: S1B0 -> C1
    records = {}
    with GI.offline_earth_engine():
        assets = {"encoder.block1.0": f"{plan.PLACEHOLDER_NAMESPACE}_S1B0"}
        for key, graph in (("unchanged", GRAPH), ("changed", changed)):
            (output,) = BUILD.build_stage(graph, stage, server, assets).values()
            with plan.deep_serialization():
                records[key] = {"kind": "fixture", **GI.raster_record(output, inputs=inputs)}
    name = "segformer_b0/stage1b"
    assert changed.sha256 != GRAPH.sha256
    verdict, message = GI.judge(name, records["unchanged"], records["changed"], GOLDEN["environment"])
    assert verdict == "fail" and records["changed"]["sha256"] != records["unchanged"]["sha256"], message
    verdict, message = GI.judge(name, GOLDEN["graphs"][name], records["changed"], GOLDEN["environment"])
    if verdict == "skip":
        assert inputs != GOLDEN["graphs"][name]["inputs"] and GI.environment_differences(GOLDEN["environment"])
    else:
        assert verdict == "fail", message
    assert records["changed"]["sha256"] != GOLDEN["graphs"][name]["sha256"]


def test_the_stages_are_the_golden_cuts():
    assert tuple(stage.name for stage in STAGES) == GI.SEGFORMER_CUTS
    assert Stage("x", ["input"], ["logits"]) == Stage("x", ("input",), ("logits",))
    assert canonical(grid_to_json(SOURCE_GRID)) == canonical(grid_to_json(plan.SOURCE_GRID))
    assert GRAPH.edges["input"].bands == ("b0", "b1")
    assert all(isinstance(n, Node) for n in GRAPH.nodes)
