"""X05: the planner, version 1 (``tensor_ee/converter/plan.py``; docs/phase3-design.md, 3.4).

- **R1, limits are inputs.** ``Limits`` defaults (24,000,000 constants, 10,000,000 request
  bytes, 1,024 bands, 2 EXTENT nodes per stage) and refusals; a boundary that cannot be saved
  as an image is never cut; NumPy weights are refused.
- **R2, rules and explanations.** Fewest stages, then fewest saved boundaries, values and
  uneven constants; required cuts; every stage carries its cost, measured bytes, EXTENT nodes
  and why it ends; a graph no plan fits is refused with code ``no_plan``. Deterministic,
  versioned JSON for stages and plans, ``Stage``'s equality unchanged.
- **R4, cost of planning.** With a counting fake builder: only the stages the search needs
  are built, never one containing a stage measured over the cap.
- **Schedule (fix round, P2-1).** The planner's own as-late-as-possible order: a side branch
  sits by its reader, so a plan does not depend on how node ids are spelled.
- **R3, the planner gate**, on the hand-written SegFormer-B0 block graph with X00's
  parameters and the planner's default measurement, the library's request sizer (X08; the
  sizing tool's ``measure_image_export`` calls it): the 2 MB cap alone gives the stage 1
  split and three stages; with the EXTENT bound of 2 read off the hand plan, five stages (stage
  4 and the decoder merge: C4 was a one-cut-per-encoder-stage convention, not a limit); with
  ``cuts`` requiring C4, all six; the default limits give four stages and report the single
  graph within the hard limits; the planned stages build the G1-ext targets byte for byte.
  These need SMP 0.5.0 and timm, and skip by name without them, as G1-ext does. The targets are
  judged by G1's version policy (X09): where the seeded state's raw inputs differ under a version
  difference, that check is a named skip.

Offline only: graphs are built against ``ee.apitestcase`` signatures with every cloud entry
point refusing; nothing is evaluated and no model forward runs.
"""
from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference.segformer_b0.block_graph import BOUNDARIES, STAGES, segformer_b0_block_graph  # noqa: E402
from tensor_ee.converter import build as BUILD, plan as PLAN  # noqa: E402
from tensor_ee.converter.graph import BlockGraph, Stage, _Canonical, canonical  # noqa: E402
from tensor_ee.converter.plan import Limits, Plan, PlannedStage, boundary_refusal, plan, schedule  # noqa: E402
from tensor_ee.core.errors import UnsupportedOperation  # noqa: E402
from tensor_ee.core.tensor import ServerArray  # noqa: E402
from tools import graph_identity as GI  # noqa: E402
from tools import segformer_b0_request_plan as P  # noqa: E402

GRAPH = segformer_b0_block_graph()
NS = P.PLACEHOLDER_NAMESPACE
GOLDEN = json.loads(GI.GOLDEN_EXT.read_text())
HAND = Limits(request_bytes=2_000_000, extent=2)
HAND_STAGES = {(stage.inputs, stage.outputs): stage.name for stage in STAGES}


def _io(planned) -> tuple:
    return planned.stage.inputs, planned.stage.outputs


# ---------------------------------------------------------------------------
# A counting fake builder: request bytes from a per-node table, no Earth Engine
# ---------------------------------------------------------------------------
# Rough per-node request bytes, shaped like the measured ones (stage 1 alone over 2 MB).
NODE_BYTES = {"relabel_to_windows": 50_000, "overlap_embedding": 100_000, "channel_norm": 5_000,
              "project_pyramid": 100_000, "fuse_head": 100_000, "relabel_from_windows": 10_000}
BLOCK_BYTES = {1: 950_000, 2: 500_000, 3: 400_000, 4: 100_000}


def _node_bytes(node) -> int:
    if node.op == "mit_block":
        return BLOCK_BYTES[int(node.id.removeprefix("encoder.block")[0])]
    return NODE_BYTES[node.op]


class FakeBuilds:
    """Stands in for ``build_stage``: records each stage built and returns, per output, the
    stage's node ids, which ``measure`` turns into bytes."""

    def __init__(self):
        self.built = []

    def build_stage(self, graph, stage, parameters, assets):
        assert set(assets) == set(stage.inputs)
        assert all(assets[e] == f"{NS}_{BOUNDARIES.get(e, e)}" for e in assets)
        nodes = graph.stage_nodes(stage.inputs, stage.outputs)
        self.built.append(tuple(n.id for n in nodes))
        return {edge: nodes for edge in stage.outputs}

    @staticmethod
    def measure(nodes, name) -> int:
        return sum(_node_bytes(node) for node in nodes)


def _placeholders(graph=GRAPH) -> dict:
    """A ServerArray for every parameter the graph binds: no array, no value (never built)."""
    return {name: ServerArray(None, (1,)) for node in graph.nodes for name in node.parameters.values()}


@pytest.fixture
def fake(monkeypatch):
    builds = FakeBuilds()
    monkeypatch.setattr(PLAN, "build_stage", builds.build_stage)
    return builds


def _plan(limits=Limits(), graph=GRAPH, **kwargs) -> Plan:
    kwargs = {"parameters": _placeholders(graph), "measure": FakeBuilds.measure, "namespace": NS,
              "names": BOUNDARIES, **kwargs}
    return plan(graph, limits, **kwargs)


def _with_lateral(name: str) -> BlockGraph:
    """The hand-written graph with a ``channel_norm`` lateral of C1, named ``name``, feeding the
    decoder in place of C1 (the review's P2-1 case)."""
    data = copy.deepcopy(GRAPH.to_json())
    data["edges"][name] = copy.deepcopy(data["edges"]["encoder.norm1"])
    data["nodes"].append({"id": name, "op": "channel_norm", "inputs": ["encoder.norm1"], "output": name,
                          "settings": {"channels": 32, "eps": 1e-6},
                          "parameters": {"weight": f"{name}.weight", "bias": f"{name}.bias"}})
    next(n for n in data["nodes"] if n["id"] == "decoder.pyramid")["inputs"][0] = name
    return BlockGraph.from_json(data)


# ---------------------------------------------------------------------------
# R1: limits and refusals
# ---------------------------------------------------------------------------
def test_limits_have_the_defaults_and_refuse_bad_values():
    assert Limits() == Limits(24_000_000, 10_000_000, 1024, 2, "table_assets")
    assert Limits().to_json()["transport"] == "table_assets"
    assert Limits.from_json(Limits(extent=None).to_json()) == Limits(extent=None)
    for bad in ({"constants": 0}, {"request_bytes": 2.0e6}, {"bands": True}, {"extent": -1}, {"extent": 2.0},
                {"transport": "inline"}):
        with pytest.raises(ValueError):
            Limits(**bad)
    assert Limits(extent=0).extent == 0


def test_a_boundary_is_an_image_of_at_most_the_band_limit():
    edges = GRAPH.edges
    assert boundary_refusal(edges["encoder.norm4"], Limits()) is None
    assert boundary_refusal(edges["decoder.pyramid"], Limits()) is None          # 1,024 bands
    assert "1,024 bands, over the limit of 1,000" in boundary_refusal(edges["decoder.pyramid"], Limits(bands=1000))


def test_refuses_numpy_and_unbound_weights_and_bad_names(fake):
    parameters = _placeholders()
    parameters["encoder.norm1.weight"] = np.zeros(32, np.float32)
    with pytest.raises(UnsupportedOperation, match="bind a ServerArray, not ndarray"):
        _plan(parameters=parameters)
    del parameters["encoder.norm1.weight"]
    with pytest.raises(UnsupportedOperation, match="'encoder.norm1.weight' is not bound"):
        _plan(parameters=parameters)
    with pytest.raises(ValueError, match="share a boundary name"):
        _plan(names={"encoder.norm1": "C1", "encoder.norm2": "C1"})
    with pytest.raises(ValueError, match="not edges"):
        _plan(names={"C9": "C9"})
    for cuts in (("input",), ("nope",), ("encoder.norm1", "encoder.norm1")):
        with pytest.raises(ValueError, match="required cuts"):
            _plan(cuts=cuts)
    with pytest.raises(TypeError, match="measure"):
        _plan(measure="bytes")                 # None is the library's request sizer (X08)
    assert fake.built == []


def test_a_graph_whose_input_is_not_an_image_or_that_no_plan_fits_is_refused(fake):
    with pytest.raises(UnsupportedOperation, match="graph input 'input' cannot be a saved image") as error:
        _plan(Limits(bands=1))
    assert error.value.code == "no_plan"
    # 16 bands: only the 2-channel edges after the relabel and the head can be cut, so every
    # plan holds all eight blocks in one stage.
    with pytest.raises(UnsupportedOperation, match="no plan of graph 'segformer_b0'") as error:
        _plan(Limits(bands=16, extent=2))
    assert error.value.code == "no_plan" and "32 bands, over the limit of 16" in str(error.value)
    single = _plan(Limits(bands=16, extent=None))
    assert [_io(s) for s in single.stages] == [(("input",), ("logits",))]


# ---------------------------------------------------------------------------
# R2: rules, explanations, JSON
# ---------------------------------------------------------------------------
def test_default_limits_cut_at_the_extent_bound_where_the_decoder_reads(fake):
    result = _plan()
    assert [_io(s) for s in result.stages] == [
        (("input",), ("encoder.norm1",)), (("encoder.norm1",), ("encoder.norm2",)),
        (("encoder.norm2",), ("encoder.norm3",)), (("encoder.norm1", "encoder.norm2", "encoder.norm3"), ("logits",))]
    assert [len(s.extent) for s in result.stages] == [2, 2, 2, 2]
    assert [s.ends for s in result.stages[:3]] == [("merged with stage2: 4 EXTENT nodes, over the bound of 2",),
                                                   ("merged with stage3: 4 EXTENT nodes, over the bound of 2",),
                                                   ("merged with stage4: 4 EXTENT nodes, over the bound of 2",)]
    c1 = result.stages[0].boundaries["encoder.norm1"]
    assert c1["readers"] == ("stage2: encoder.patch_embed2", "stage4: decoder.pyramid")    # read-only: tuples
    assert c1["reasons"] == ("stage2 and stage4 read it: one saved image serves each",       # its own reasons
                             "read at a resolution change: encoder.patch_embed2 maps 56 x 56 to 28 x 28")
    assert result.stages[2].boundaries["encoder.norm3"]["reasons"][0] == "the next stage, stage4, reads it"
    assert result.search["score"]["stages"] == 4 and result.search["score"]["boundaries"] == 3
    assert [s.exports for s in result.stages] == [1, 1, 1, 1]
    single = result.single_graph
    assert single["within_hard_limits"] and not single["within"]["extent"]
    assert single["extent"]["count"] == 8 and single["cost"] == {"values": 6_620_078, "asset_arrays": 191}
    assert "its recomputation cost is not modeled" in single["notes"][0]


def test_no_bound_gives_the_single_graph(fake):
    result = _plan(Limits(extent=None))
    assert [_io(s) for s in result.stages] == [(("input",), ("logits",))]
    assert result.stages[0].ends == ("it writes the graph's outputs",) and result.search["builds"] == 1


def test_a_required_cut_is_kept_and_explained(fake):
    result = _plan(cuts=("encoder.norm4",))
    assert [_io(s) for s in result.stages][-2:] == [
        (("encoder.norm3",), ("encoder.norm4",)),
        (("encoder.norm1", "encoder.norm2", "encoder.norm3", "encoder.norm4"), ("logits",))]
    reasons = result.stages[-2].boundaries["encoder.norm4"]["reasons"]
    assert reasons[:2] == ("required by the caller", "the next stage, stage5, reads it")
    assert result.stages[-2].ends == ("merged with stage5: it crosses the required cut at 'encoder.norm4'",)
    assert result.cuts == ("encoder.norm4",)


def test_several_exports_are_counted(fake):
    result = _plan(Limits(extent=3))            # stage 2 writes C2 for the decoder and block3.0 for stage 3
    assert [s.exports for s in result.stages] == [1, 2, 1]
    assert result.stages[1].to_json()["exports"] == 2
    assert result.stages[1].boundaries["encoder.norm2"]["reasons"][0] == "the next stage, stage3, reads it"


def test_the_schedule_places_a_side_branch_by_its_reader_not_its_spelling(fake):
    """P2-1: a lateral of C1 feeding the decoder sits just before it, whatever its id, so the
    plans do not depend on the spelling. The canonical hashing order would put ``a.lateral1``
    right after C1, inside encoder stage 2, and ``z.lateral1`` elsewhere."""
    assert schedule(GRAPH) == GRAPH.nodes                    # SegFormer-B0: a chain, one multi-input node
    plans = {}
    for name in ("a.lateral1", "z.lateral1"):
        graph = _with_lateral(name)
        order = [node.id for node in schedule(graph)]
        assert order[-5:] == ["encoder.norm4", name, "decoder.pyramid", "decoder.head", "logits"]
        for key, limits, cuts in (("default", Limits(), ()), ("hand", HAND, ()), ("hand_c4", HAND, ("encoder.norm4",))):
            result = _plan(limits, graph=graph, cuts=cuts)
            assert result.search["order"] == tuple(order)
            plans[name, key] = ([tuple(tuple("lateral" if e == name else e for e in side) for side in _io(s))
                                 for s in result.stages], dict(result.search["score"]))
    for key in ("default", "hand", "hand_c4"):
        assert plans["a.lateral1", key] == plans["z.lateral1", key]
    stages, score = plans["a.lateral1", "default"]
    assert stages == [(("input",), ("encoder.norm1",)), (("encoder.norm1",), ("encoder.norm2",)),
                      (("encoder.norm2",), ("encoder.norm3",)),
                      (("encoder.norm1", "encoder.norm2", "encoder.norm3"), ("logits",))]
    assert (score["stages"], score["boundaries"]) == (4, 3)


def test_every_stage_explains_itself_and_whole_chip_stages_are_not_tileable(fake):
    for result in (_plan(), _plan(HAND), _plan(HAND, cuts=("encoder.norm4",))):
        for planned in result.stages:
            assert planned.ends and planned.requests and set(planned.requests) == set(planned.stage.outputs)
            assert planned.cost == GRAPH.cost(planned.stage)
            assert planned.tileable is (not planned.extent and not planned.chip_edge_dependent)
            assert not planned.tileable                     # every stage holds a MiT block or the head
            assert all(b["reasons"] for b in planned.boundaries.values())
    open_stage = PlannedStage(Stage("s", ("input",), ("input_windows",)), ("input_windows",), {}, (), (), {"x": 1},
                              ("e",), {})
    assert open_stage.tileable


def test_plan_json_is_deterministic_versioned_and_checked(fake):
    first, second = _plan(HAND), _plan(HAND)
    assert first == second and first.sha256 == second.sha256 and hash(first) == hash(second)
    assert first.dumps() == canonical(first.to_json()) and "seconds" not in first.dumps()
    assert isinstance(first.seconds, float) and first.seconds >= 0
    text = first.dumps(indent=1)
    assert Plan.loads(text) == first and Plan.loads(text).sha256 == first.sha256
    data = first.to_json()
    assert (data["format"], data["version"]) == ("native_ee.plan", 1)
    assert sorted(data) == ["cuts", "format", "graph", "limits", "naming", "search", "single_graph", "stages",
                            "version"]                                       # data only, no method prose
    assert data["limits"]["transport"] == "table_assets"
    tampered = copy.deepcopy(data)
    tampered["stages"][0]["tileable"] = True
    with pytest.raises(ValueError, match="derived field"):
        Plan.from_json(tampered)
    with pytest.raises(ValueError, match="native_ee.plan version 1"):
        Plan.from_json({**data, "version": 2})
    with pytest.raises(TypeError):
        first.stages[0].boundaries["encoder.block1.0"]["name"] = "x"             # read-only
    assert isinstance(first.stages[0].boundaries["encoder.block1.0"]["reasons"], tuple)


def test_stage_json_round_trips_and_its_equality_is_unchanged():
    for stage in STAGES:
        assert Stage.from_json(json.loads(json.dumps(stage.to_json()))) == stage
        assert hash(Stage.from_json(stage.to_json())) == hash(stage)
    assert Stage("x", ["input"], ["logits"]) == Stage("x", ("input",), ("logits",))
    assert is_dataclass(Stage) and [f.name for f in fields(Stage)] == ["name", "inputs", "outputs"]
    assert not issubclass(Stage, _Canonical)


# ---------------------------------------------------------------------------
# R4: only what the search needs is built
# ---------------------------------------------------------------------------
def test_measures_lazily_and_never_builds_a_stage_containing_one_over_the_cap(fake):
    result = _plan(HAND)
    position = {node.id: k for k, node in enumerate(GRAPH.nodes)}
    spans = [(position[ids[0]], position[ids[-1]]) for ids in fake.built]
    assert len(fake.built) == result.search["builds"] == 7
    measured = result.search["measured"]
    assert [m["within"] for m in measured] == [False, False, True, True, True, True, True]
    over = []
    for (first, last), record in zip(spans, measured):
        assert not any(first <= a and b <= last for a, b in over)     # never one containing an over-cap stage
        if not record["within"]:
            over.append((first, last))
    assert len(set(spans)) == len(spans)                    # each candidate built once
    # The same stages as the measured plan below, from the same rules.
    assert [_io(s) for s in result.stages][:4] == [(s.inputs, s.outputs) for s in STAGES[1:5]]
    assert result.single_graph["requests"] is None and result.single_graph["notes"][0].startswith(
        "not built: it contains the stage ['input'] -> ['encoder.norm1'] measured 2,055,000 request bytes")


# ---------------------------------------------------------------------------
# R3: the planner gate on SegFormer-B0, with the library's request sizer
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def x00_parameters():
    """The seeded state's weights as X00 reads them: ServerArrays of tables planned under the
    placeholder namespace (``tools/graph_identity.py``, ``_segformer_outputs``)."""
    reason = GI.ext_unavailable("segformer_b0", ("segmentation_models_pytorch", "timm"))
    if reason:
        pytest.skip(f"SegFormer-B0 plans not measured: {reason}")
    from tensor_ee._swin_parameters import records_and_locations
    from tensor_ee.models.segformer_b0.configuration import COUNTER
    from tools.fixtures.segformer_b0_fullmodel import SEED, reject_negative_zero, seeded_state, state_sha256
    with GI.offline_earth_engine() as attempts:
        state = seeded_state(SEED)
        reject_negative_zero(state)
        digest = state_sha256(state)
        records = list(records_and_locations({}, {}, {k: v for k, v in state.items() if k != COUNTER})[0].values())
        shards = P.plan_weight_shards(records, NS + "_weights", P.WEIGHT_WIRE_LIMIT)
        manifest = P.planned_manifest(records, shards, NS + "_weights", digest)
        server = P.asset_state(state, manifest, digest)
    assert attempts == []
    return server, {"sha256": digest, "tensors": len(state)}


@pytest.fixture(scope="module")
def plans(x00_parameters):
    server, _ = x00_parameters
    runs = {"cap_only": (Limits(request_bytes=2_000_000, extent=None), ()), "hand": (HAND, ()),
            "hand_c4": (HAND, ("encoder.norm4",)), "default": (Limits(), ()), "unbounded": (Limits(extent=None), ())}
    with GI.offline_earth_engine() as attempts:
        # The planner's default measurement: the library's request sizer (X08), which the
        # sizing tool's measure_image_export also calls.
        out = {key: plan(GRAPH, limits, parameters=server, namespace=NS, names=BOUNDARIES, cuts=cuts)
               for key, (limits, cuts) in runs.items()}
    assert attempts == []
    return out


def test_r3_the_cap_alone_reproduces_the_stage1_split_only(plans):
    """The baseline: only the 2 MB cap was a limit of the hand plan; its other cuts (C1-C4)
    were one cut per encoder stage by convention, so the cap alone gives three stages."""
    result = plans["cap_only"]
    assert [_io(s) for s in result.stages] == [
        (("input",), ("encoder.block1.0",)), (("encoder.block1.0",), ("encoder.norm1", "encoder.block2.0")),
        (("encoder.norm1", "encoder.block2.0"), ("logits",))]
    assert HAND_STAGES.get(_io(result.stages[0])) == "stage1a"
    assert [s.exports for s in result.stages] == [1, 2, 1] and len(result.stages[2].extent) == 5
    assert all(s.request_bytes <= 2_000_000 for s in result.stages)


def test_r3_the_hand_limits_give_the_hand_stages_but_merge_stage4_and_the_decoder(plans):
    result = plans["hand"]
    assert [HAND_STAGES.get(_io(s)) for s in result.stages] == ["stage1a", "stage1b", "stage2", "stage3", None]
    assert _io(result.stages[4]) == (("encoder.norm1", "encoder.norm2", "encoder.norm3"), ("logits",))
    assert all(s.request_bytes <= 2_000_000 for s in result.stages)
    assert len(result.stages[4].extent) == 2 and result.stages[4].cost["values"] == 3_834_478
    assert result.stages[0].ends[0].startswith(
        "merged with stage2: the stage ['input'] -> ['encoder.norm1'] measured ")
    assert result.search["builds"] == 7 and not result.single_graph["within_hard_limits"]


def test_r3_the_hand_limits_with_the_c4_cut_give_all_six_hand_stages(plans):
    result = plans["hand_c4"]
    assert [_io(s) for s in result.stages] == [(s.inputs, s.outputs) for s in STAGES[1:]]
    assert result.stages[4].boundaries["encoder.norm4"]["reasons"][0] == "required by the caller"


def test_r3_the_default_limits_give_fewer_stages_and_report_the_single_graph(plans):
    result = plans["default"]
    assert [_io(s) for s in result.stages] == [
        (("input",), ("encoder.norm1",)), (("encoder.norm1",), ("encoder.norm2",)),
        (("encoder.norm2",), ("encoder.norm3",)), (("encoder.norm1", "encoder.norm2", "encoder.norm3"), ("logits",))]
    single = result.single_graph
    assert single["within_hard_limits"] and single["within"] == {"constants": True, "request_bytes": True,
                                                                 "extent": False}
    assert 2_000_000 < single["requests"]["logits"] <= 10_000_000 and single["extent"]["count"] == 8
    assert result.search["builds"] == 5
    assert [HAND_STAGES.get(_io(s)) for s in plans["unbounded"].stages] == ["monolithic"]


def judge_targets(records: dict) -> None:
    """Every record against its G1-ext target by G1's version policy (tools/graph_identity.judge),
    all judged before any verdict acts: any fail fails; otherwise, when judge names some a skip
    (their raw inputs differ under a version difference), the test is a named skip; otherwise
    every record equals its target. In the golden environment judge never skips."""
    verdicts = {name: GI.judge(name, GOLDEN["graphs"][name], record, GOLDEN["environment"])
                for name, record in records.items()}
    failed = [message for verdict, message in verdicts.values() if verdict == "fail"]
    assert not failed, "\n".join(failed)
    drifted = sorted(name for name, (verdict, _) in verdicts.items() if verdict == "skip")
    if drifted:
        pytest.skip(f"G1-ext targets not judged here: {', '.join(drifted)}. {verdicts[drifted[0]][1]}")
    assert all(verdict == "pass" for verdict, _ in verdicts.values())
    for name, record in records.items():
        assert record["sha256"] == GOLDEN["graphs"][name]["sha256"], name


def test_planned_stages_build_the_g1ext_targets(plans, x00_parameters):
    server, inputs = x00_parameters
    planned = [s for key in ("hand", "hand_c4", "unbounded") for s in plans[key].stages if _io(s) in HAND_STAGES]
    targets = {HAND_STAGES[_io(s)]: s.stage for s in planned}
    assert sorted(targets) == sorted(GI.SEGFORMER_CUTS)
    records = {}
    with GI.offline_earth_engine() as attempts:
        for target, stage in targets.items():
            assets = {edge: f"{NS}_{BOUNDARIES[edge]}" for edge in stage.inputs}
            (output,) = BUILD.build_stage(GRAPH, stage, server, assets).values()
            with P.deep_serialization():
                records[f"segformer_b0/{target}"] = {"kind": "fixture", **GI.raster_record(output, inputs=inputs)}
    assert attempts == []
    judge_targets(records)
