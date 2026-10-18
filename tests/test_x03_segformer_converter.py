"""X03: the converter for SMP SegFormer (``tensor_ee/converter/``: ``export``, ``recognizers``,
``convert``; docs/phase3-design.md, section 3.3).

- **R1, trace.** ``torch.export`` with fake tensors at one input shape; a model in training mode
  (or one submodule) is refused before anything is traced.
- **R2, modules, not operations.** ATen calls are grouped by the recognized module on their
  ``nn_module_stack``, matched by qualified class name; the converter layer loads neither SMP nor
  timm. A module no recognizer names is refused, naming it.
- **R3, closed glue.** Between recognized modules only layout bookkeeping is accepted, checked on
  the traced shapes as moves of named axes; anything else is refused, naming the operation and
  its module (a small torch model with a test-local recognizer table, and SMP's encoder with an
  arithmetic step added).
- **R4, converts versus validated.** Every node's ``check_settings`` answer, rendered per nearest
  configuration; ``mit_b1`` converts and is reported outside its records; what the library's
  composites fix is refused, naming the attribute; ``mit_b0`` at 512 x 512 is refused for its
  keys and values.
- **R5, the gate.** SMP 0.5.0 ``mit_b0`` at 224 x 224, with the hand-written graph's source grid,
  converts to exactly ``tests/reference/segformer_b0/block_graph.json`` (same text, same hash);
  built with the builder, the stages of X05's plans of the converted graph give the seven G1-ext
  SegFormer-B0 targets byte for byte. Judged by G1's version policy (X09): where the seeded
  state's raw inputs differ under a version difference, the gate is a named skip, and the 10 m
  source's stages are still judged against the hand-written graph's built from the same inputs.
- **Fix round.** Forward code: each unit's calls match the committed fingerprints of the
  validated trace (six over mit_b0's 18 units, the same for mit_b1), and the review's eight
  forward patches, with no attribute changed, are each refused by default and reported outside
  under ``forward_mismatch="report"``. The token grid is abstract (scale 2.5 by default), so a
  10 m source gives the validated interior stage graphs. The report gives the installed versions.
  ``_resolve``'s ``getitem`` branch is tested directly.

The SMP tests need SMP 0.5.0 and timm, and skip by name without them, as G1-ext does. Offline
only: models are constructed with random weights (``encoder_weights=None``) and traced with fake
tensors, which runs the models' forward code without computing on data; no cloud call is made.
"""
from __future__ import annotations

import json
import operator
import os
from pathlib import Path
import re
import subprocess
import sys
import types

import pytest
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference.segformer_b0.block_graph import BOUNDARIES, SOURCE_GRID, STAGES  # noqa: E402
from tensor_ee.converter import build as BUILD  # noqa: E402
from tensor_ee.converter.convert import INPUT, LOGITS, convert  # noqa: E402
from tensor_ee.converter.export import (GLUE, NCHW, Unit, _Grouping, container, group,  # noqa: E402
                                       operation_name, qualified_name, trace)
from tensor_ee.converter.graph import BlockGraph, canonical, grid_to_json  # noqa: E402
from tensor_ee.converter.plan import Limits, plan  # noqa: E402
from tensor_ee.converter.recognizers import FINGERPRINTS, RECOGNIZERS, TRACED_WITH  # noqa: E402
from tensor_ee.core.errors import UnsupportedOperation  # noqa: E402
from tensor_ee.core.grid import Grid, TokenGrid  # noqa: E402
from tools import graph_identity as GI  # noqa: E402
from tools import segformer_b0_request_plan as P  # noqa: E402

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[1]
FIXTURE_JSON = ROOT / "tests" / "reference" / "segformer_b0" / "block_graph.json"
SHA256 = "568c92c33a639257c43578d3c7773f91f98497d4c12191319ff2565a23b0f39a"
SHAPE = (1, 2, 224, 224)
NS = P.PLACEHOLDER_NAMESPACE
GOLDEN = json.loads(GI.GOLDEN_EXT.read_text())
HAND = Limits(request_bytes=2_000_000, extent=2)
HAND_STAGES = {(stage.inputs, stage.outputs): stage.name for stage in STAGES}
SMP_REASON = GI.ext_unavailable("segformer_b0", ("segmentation_models_pytorch", "timm"))
needs_smp = pytest.mark.skipif(SMP_REASON is not None, reason=f"SMP SegFormer not converted: {SMP_REASON}")
ENCODER = "segmentation_models_pytorch.encoders.mix_transformer.MixVisionTransformerEncoder"


def segformer(encoder="mit_b0", **kwargs):
    """SMP's Segformer with random weights, in evaluation mode unless ``train=True``."""
    import segmentation_models_pytorch as smp
    train = kwargs.pop("train", False)
    torch.manual_seed(0)
    model = smp.Segformer(encoder, encoder_weights=None, in_channels=2, classes=kwargs.pop("classes", 2), **kwargs)
    return model if train else model.eval()


@pytest.fixture(scope="module")
def b0():
    if SMP_REASON:
        pytest.skip(f"SMP SegFormer not converted: {SMP_REASON}")
    return convert(segformer(), SHAPE, SOURCE_GRID, name="segformer_b0")


@pytest.fixture(scope="module")
def b1():
    if SMP_REASON:
        pytest.skip(f"SMP SegFormer not converted: {SMP_REASON}")
    return convert(segformer("mit_b1"), SHAPE, SOURCE_GRID, name="segformer_b1")


# ---------------------------------------------------------------------------
# R5: the gate
# ---------------------------------------------------------------------------
def test_mit_b0_converts_to_exactly_the_hand_written_block_graph(b0):
    text = FIXTURE_JSON.read_text()
    assert b0.graph == BlockGraph.loads(text)
    assert b0.graph.sha256 == SHA256
    assert b0.graph.dumps(indent=1) + "\n" == text


OTHER_GRID = Grid("EPSG:32633", (10.0, 0, 500000.0, 0, -10.0, 4200000.0), 224, 224)
"""A 10 m source in another CRS and place."""


@pytest.fixture(scope="module")
def b0_10m():
    if SMP_REASON:
        pytest.skip(f"SMP SegFormer not converted: {SMP_REASON}")
    return convert(segformer(), SHAPE, OTHER_GRID, name="segformer_b0")


def test_the_source_grid_is_written_from_the_one_argument(b0_10m):
    """The source grid is the input edge, the inverse relabel's setting and the logits edge; the
    token grid is abstract (2.5 unless ``scale`` says otherwise), so nothing else depends on it."""
    graph = b0_10m.graph
    data, hand = graph.to_json(), json.loads(FIXTURE_JSON.read_text())
    assert data["edges"][INPUT] == {**hand["edges"][INPUT], "grid": grid_to_json(OTHER_GRID)}
    assert data["edges"][LOGITS] == {**hand["edges"][LOGITS], "grid": grid_to_json(OTHER_GRID)}
    logits = next(n for n in data["nodes"] if n["id"] == LOGITS)
    assert logits["settings"]["grid"] == grid_to_json(OTHER_GRID)
    assert canonical(data).count(canonical(grid_to_json(OTHER_GRID))) == 3
    assert next(n for n in graph.nodes if n.id == "input_windows").settings["grid"] == TokenGrid(224, 224, 2.5)
    # Another source, of any pixel size, converts to the hand-written graph with its grid replaced.
    assert canonical(data).replace(canonical(grid_to_json(OTHER_GRID)), canonical(grid_to_json(SOURCE_GRID))) == \
        canonical(hand)
    named = convert(segformer(), SHAPE, OTHER_GRID, bands=("vv", "vh")).graph
    assert named.edges[INPUT].bands == ("vv", "vh") and named.name == "segformer"


@needs_smp
def test_the_token_grid_scale_is_a_keyword():
    graph = convert(segformer(), SHAPE, SOURCE_GRID, scale=10).graph
    assert graph.edges["input_windows"].grid == TokenGrid(224, 224, 10.0)
    assert graph.edges["encoder.norm4"].grid == TokenGrid(7, 7, 320.0)
    with pytest.raises(ValueError, match="scale is a positive finite number"):
        convert(segformer(), SHAPE, SOURCE_GRID, scale=0.0)


@pytest.fixture(scope="module")
def x00_parameters():
    """The seeded state's weights as X00 reads them: ServerArrays of tables planned under the
    placeholder namespace (``tools/graph_identity.py``, ``_segformer_outputs``)."""
    if SMP_REASON:
        pytest.skip(f"SegFormer-B0 targets not built: {SMP_REASON}")
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


def _tool_measure(raster, name) -> int:
    """The sizing tool's complete ExportImageRequest body, as its measure() records it."""
    return P.measure_image_export(raster.image, raster.grid, NS, name)["wire"]["bytes"]


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


def _stage_record(graph, stage, server, inputs) -> dict:
    """The builder's stage of ``graph`` as G1-ext records a target, its inputs saved under the
    placeholder namespace."""
    assets = {edge: f"{NS}_{BOUNDARIES[edge]}" for edge in stage.inputs}
    (output,) = BUILD.build_stage(graph, stage, server, assets).values()
    with P.deep_serialization():
        return {"kind": "fixture", **GI.raster_record(output, inputs=inputs)}


def test_the_converted_graph_builds_the_seven_g1ext_targets(b0, x00_parameters):
    """X05's plans of the converted graph (the hand cap with C4 required: the six cuts; no EXTENT
    bound: the single graph), each stage built with the builder, judged against G1-ext."""
    server, inputs = x00_parameters
    with GI.offline_earth_engine() as attempts:
        plans = [plan(b0.graph, limits, parameters=server, measure=_tool_measure, namespace=NS, names=BOUNDARIES,
                      cuts=cuts) for limits, cuts in ((HAND, ("encoder.norm4",)), (Limits(extent=None), ()))]
        stages = {HAND_STAGES[(s.stage.inputs, s.stage.outputs)]: s.stage for p in plans for s in p.stages}
        assert sorted(stages) == sorted(GI.SEGFORMER_CUTS)
        records = {f"segformer_b0/{target}": _stage_record(b0.graph, stage, server, inputs)
                   for target, stage in stages.items()}
    assert attempts == []
    judge_targets(records)


TEN_METRE = {"monolithic": "fail", "stage1a": "fail", "stage1b": "pass", "stage2": "pass", "stage3": "pass",
             "stage4": "pass", "decoder": "fail"}
"""Each stage of the 10 m source's graph against the validated stage: equal, or not."""


def test_a_10m_source_gives_the_validated_interior_stage_graphs(b0, b0_10m, x00_parameters):
    """The token grid is abstract: the stages that neither read the input nor write the logits are
    the validated ones whatever the source; the three that do read or write it on the source grid
    (its saved input's pinning and the relabels' projections), so they differ. Judged against the
    hand-written graph's stages (``b0``) built here from the same inputs, in any environment; and
    against the G1-ext targets, which judge names skips where the seeded state's raw inputs differ
    under a version difference (the hosted CI's versions), and judges exactly elsewhere."""
    server, inputs = x00_parameters
    verdicts, here = {}, {}
    with GI.offline_earth_engine() as attempts:
        for stage in STAGES:
            record = _stage_record(b0_10m.graph, stage, server, inputs)
            name = f"segformer_b0/{stage.name}"
            verdicts[stage.name] = GI.judge(name, GOLDEN["graphs"][name], record, GOLDEN["environment"])[0]
            here[stage.name] = GI.judge(name, _stage_record(b0.graph, stage, server, inputs), record,
                                        GOLDEN["environment"])[0]
            if stage.name in ("monolithic", "stage1a", "decoder"):
                assert record["sha256"] != GOLDEN["graphs"][name]["sha256"]
    assert attempts == []
    assert here == TEN_METRE
    if "skip" in verdicts.values():
        assert set(verdicts.values()) == {"skip"} and inputs != GOLDEN["graphs"]["segformer_b0/stage1b"]["inputs"]
    else:
        assert verdicts == TEN_METRE


# ---------------------------------------------------------------------------
# R4: the report
# ---------------------------------------------------------------------------
def test_the_mit_b0_report_has_every_node_inside_its_record(b0):
    report = b0.report
    assert report.inside and report.outside == ()
    checked = [node for node in report.nodes if node.check is not None]
    assert [node.id for node in report.nodes] == [node.id for node in b0.graph.nodes]
    assert len(checked) == 18 and all(node.validated for node in checked)
    assert {node.id: node.validated for node in report.nodes if node.check is None} == {"input_windows": None,
                                                                                         LOGITS: None}
    head = next(node for node in report.nodes if node.id == "decoder.head")
    assert dict(head.settings) == {"channels": 256, "classes": 2, "eps": 1e-5, "in_channels": 1024, "height": 56,
                                   "width": 56, "window": 7}
    assert head.modules == (("decoder.fuse_stage", "segmentation_models_pytorch.base.modules.Conv2dReLU"),
                            ("segmentation_head", "segmentation_models_pytorch.base.heads.SegmentationHead"))
    pyramid = next(node for node in report.nodes if node.id == "decoder.pyramid")
    assert pyramid.settings["in_channels"] == (256, 160, 64, 32)
    assert report.glue == (("aten.contiguous.default", "encoder", ENCODER),) * 4
    assert report.unused == (("aten.empty.memory_format", "encoder", ENCODER),)
    lines = report.lines()
    assert lines[0] == "segformer_b0: 20 nodes; 18 with a validated record, 18 inside, 0 outside; 2 layout operations"
    assert "glue: 4 x aten.contiguous.default in encoder (MixVisionTransformerEncoder)" in lines


def test_mit_b1_converts_and_reports_the_settings_outside_the_records(b1):
    report, graph = b1.report, b1.graph
    assert not report.inside
    inside = {"encoder.norm1", "decoder.head"}                  # width 64, and the head: 1024 -> 256 -> 2
    assert set(report.outside) == {node.id for node in report.nodes if node.check is not None} - inside
    assert len(report.outside) == 16
    nodes = {node.id: node for node in report.nodes}
    block = nodes["encoder.block1.0"]
    assert block.check.outside == ("dim",) and block.check.unstated == ()
    assert block.verdict() == ("outside its validated record: no configuration holds dim 64 with heads 1, ratio 8, "
                               "window 7; nearest: {dim 32; heads 1; ratio 8; window 7}: dim 64 is not 32")
    # Ties: outside is rendered per nearest configuration, not as their union.
    embedding = nodes["encoder.patch_embed3"]
    assert embedding.check.outside == ("cin", "cout") and len(embedding.nearest()) == 3
    assert embedding.nearest()[1] == ("{cin 64; cout 160; kernel 3; stride 2; window 7}: cin 128 is not 64; "
                                      "cout 320 is not 160")
    assert nodes["decoder.pyramid"].check.outside == ("in_channels",)
    assert nodes["encoder.norm2"].nearest() == ("{channels 32 or 64 or 160 or 256; eps 1e-05 or 1e-06; window 7}: "
                                                 "channels 128 is not 32 or 64 or 160 or 256",)
    # Settings are plain Python, as the records hold them.
    assert all(type(node.settings["eps"]) is float for node in report.nodes if "eps" in node.settings)
    assert graph.cost()["asset_arrays"] == 191 and graph.edges["encoder.norm4"].channels == 512


@needs_smp
def test_other_unvalidated_settings_are_reported_not_refused():
    report = convert(segformer(classes=3, decoder_segmentation_channels=128), SHAPE, SOURCE_GRID).report
    nodes = {node.id: node for node in report.nodes}
    assert nodes["decoder.pyramid"].check.outside == ("channels",)
    assert nodes["decoder.head"].check.outside == ("channels", "classes", "in_channels")
    assert set(report.outside) == {"decoder.pyramid", "decoder.head"}
    assert report.graph == "segformer"


# ---------------------------------------------------------------------------
# R4: refused, what the library cannot express
# ---------------------------------------------------------------------------
@needs_smp
def test_mit_b0_at_512_is_refused_for_its_keys_and_values():
    grid = Grid("EPSG:3857", (2.5, 0, 1000, 0, -2.5, 2000), 512, 512)
    with pytest.raises(UnsupportedOperation, match=r"encoder\.block1\.0 .*spatial-reduction ratio 8 turns its "
                       r"128 x 128 map into 16 x 16 keys and values.*one 7 x 7 window.*needs a new attention block") \
            as refusal:
        convert(segformer(), (1, 2, 512, 512), grid)
    assert refusal.value.code == "unsupported_setting"


def _set_block_eps(model):
    model.encoder.block1[0].norm1.eps = 1e-5


def _gelu_tanh(model):
    model.encoder.block2[1].mlp.act = nn.GELU(approximate="tanh").eval()


def _qk_scale(model):
    model.encoder.block3[0].attn.scale = 0.1


def _embedding_eps(model):
    model.encoder.patch_embed2.norm.eps = 1e-6


@needs_smp
@pytest.mark.parametrize("change, message", [
    (_set_block_eps, r"encoder\.block1\.0\.norm1 \(.*LayerNorm\): eps 1e-05, where the library's composite "
                     r"fixes 1e-06"),
    (_gelu_tanh, r"encoder\.block2\.1\.mlp\.act \(torch\.nn\.modules\.activation\.GELU\): approximate 'tanh'"),
    (_qk_scale, r"encoder\.block3\.0\.attn \(.*Attention\): scale 0\.1, where the library scales scores by head width"),
    (_embedding_eps, r"encoder\.patch_embed2\.norm .*eps 1e-06, where the library's composite fixes 1e-05"),
], ids=["block norm eps", "tanh GELU", "attention scale", "embedding norm eps"])
def test_attributes_the_composites_fix_are_refused_naming_them(change, message):
    model = segformer()
    change(model)
    with pytest.raises(UnsupportedOperation, match=message) as refusal:
        convert(model, SHAPE, SOURCE_GRID)
    assert refusal.value.code == "unsupported_setting" and "cannot express it" in str(refusal.value)


@needs_smp
def test_heads_the_decoder_lacks_are_refused():
    with pytest.raises(UnsupportedOperation, match="the model returns 2 values; a converted model returns one"):
        convert(segformer(aux_params={"classes": 2}), SHAPE, SOURCE_GRID)       # masks and labels
    with pytest.raises(UnsupportedOperation, match=r"segmentation_head\.1 .*fuse_head upsamples 4 times"):
        convert(segformer(upsampling=2), SHAPE, SOURCE_GRID)


# ---------------------------------------------------------------------------
# R1: training mode
# ---------------------------------------------------------------------------
@needs_smp
def test_a_model_in_training_mode_is_refused_before_tracing(monkeypatch):
    def never(*args, **kwargs):
        raise AssertionError("traced")

    monkeypatch.setattr(torch.export, "export", never)
    with pytest.raises(UnsupportedOperation, match=r"set model.eval\(\) before converting: \d+ modules") as refusal:
        convert(segformer(train=True), SHAPE, SOURCE_GRID)
    assert refusal.value.code == "training_mode"
    model = segformer()
    model.encoder.block1[1].drop_path.train()
    with pytest.raises(UnsupportedOperation,
                       match=r"1 modules are in training mode \(first \['encoder\.block1\.1\.drop_path'\]"):
        convert(model, SHAPE, SOURCE_GRID)


# ---------------------------------------------------------------------------
# R2 and R3: unrecognized modules and glue on SMP's SegFormer
# ---------------------------------------------------------------------------
@needs_smp
def test_an_unrecognized_module_is_refused_naming_it():
    with pytest.raises(UnsupportedOperation, match=r"module '1' \(torch\.nn\.modules\.activation\.ReLU\) is not a "
                                                   r"recognized module \(it runs aten\.relu\.default\)") as refusal:
        convert(nn.Sequential(segformer(), nn.ReLU()).eval(), SHAPE, SOURCE_GRID)
    assert refusal.value.code == "unrecognized_module"
    model = segformer()
    model.encoder.block1[1] = nn.Conv2d(32, 32, 1).eval()
    with pytest.raises(UnsupportedOperation, match=r"module 'encoder\.block1\.1' \(torch\.nn\.modules\.conv\.Conv2d\)"):
        convert(model, SHAPE, SOURCE_GRID)


@needs_smp
def test_glue_outside_the_closed_set_is_refused_naming_it():
    """SMP's encoder, a container, with an arithmetic step between two recognized modules."""
    model = segformer()
    original = model.encoder.forward

    def forward(self, x):
        features = original(x)
        return features[:-1] + [features[-1] * 2.0]

    model.encoder.forward = types.MethodType(forward, model.encoder)
    with pytest.raises(UnsupportedOperation, match=r"aten\.mul\.Tensor in module 'encoder' \(.*"
                       r"MixVisionTransformerEncoder\) is not layout bookkeeping") as refusal:
        convert(model, SHAPE, SOURCE_GRID)
    assert refusal.value.code == "unsupported_glue"


def test_the_converter_loads_neither_smp_nor_timm():
    code = ("import sys; import tensor_ee.converter.convert; "
            "print(sorted(m for m in sys.modules if m.split('.')[0] in ('segmentation_models_pytorch', 'timm')))")
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=300,
                            env={**os.environ, "PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": "2"})
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
    assert sorted(name.rsplit(".", 1)[-1] for name in RECOGNIZERS) == [
        "Block", "LayerNorm", "MixVisionTransformerEncoder", "OverlapPatchEmbed", "Segformer", "Sequential"]


# ---------------------------------------------------------------------------
# R3: glue on a small torch model, with a test-local recognizer table
# ---------------------------------------------------------------------------
class Pair(nn.Module):
    """Two 1 x 1 convolutions with ``glue`` between them, in the container's own forward."""

    def __init__(self, glue):
        super().__init__()
        self.a, self.b, self.glue = nn.Conv2d(2, 4, 1), nn.Conv2d(4, 4, 1), glue

    def forward(self, x):
        return self.b(self.glue(self.a(x)))


def _conv_unit(module, path, traced):
    return (Unit(path, "conv", {}, {"weight": f"{path}.weight", "bias": f"{path}.bias"}, owns=(path,), inputs=(path,),
                 window=1),)


TABLE = {qualified_name(Pair(None)): container, "torch.nn.modules.conv.Conv2d": _conv_unit}


def _group(glue):
    return group(trace(Pair(glue).eval(), (1, 2, 8, 8)), TABLE)


@pytest.mark.parametrize("glue", [
    lambda y: y,
    lambda y: y.permute(0, 2, 3, 1).permute(0, 3, 1, 2),
    lambda y: y.flatten(2).transpose(1, 2).transpose(1, 2).reshape(1, 4, 8, 8),
    lambda y: y.view(1, -1, 64).view(1, 4, 8, -1).contiguous(),
    lambda y: y.reshape(4, 8, 8).reshape(1, 4, 64).view(1, 4, 8, 8),
], ids=["none", "permute and back", "tokens and back", "views with -1", "batch axis moved"])
def test_layout_bookkeeping_between_modules_is_accepted(glue):
    grouped = _group(glue)
    assert dict(grouped.reads) == {"a": ("input",), "b": ("a",)} and grouped.output == "b"
    assert {operation for operation, _, _ in grouped.glue} <= {operation_name(target) for target in GLUE}


@pytest.mark.parametrize("glue, message", [
    (lambda y: y * 2, r"aten\.mul\.Tensor in module '<root>' \(.*Pair\) is not layout bookkeeping between recognized"),
    (lambda y: y.flip(-1), r"aten\.flip\.default in module '<root>'"),
    (lambda y: y.transpose(2, 3), r"b \(conv\) reads input 0 as \(N, C, W, H\), where its module reads \(N, C, H, W\)"),
    (lambda y: y.reshape(1, 2, 2, 64).reshape(1, 4, 8, 8), r"aten\.reshape\.default .* reshapes \(N, C, H, W\) "
                                                           r"\(1, 4, 8, 8\) to \(1, 2, 2, 64\), splitting an axis"),
    (lambda y: y.permute(0, 2, 3, 1).reshape(1, 4, 8, 8), r"splitting an axis"),
], ids=["arithmetic", "flip", "rows and columns swapped", "channels split", "channels-last read as channels-first"])
def test_anything_else_between_modules_is_refused_naming_it(glue, message):
    with pytest.raises(UnsupportedOperation, match=message) as refusal:
        _group(glue)
    assert refusal.value.code == "unsupported_glue"


def test_calls_whose_results_reach_no_output_are_dropped_and_listed():
    def unused(y):
        torch.ones(3)
        y.sum()
        return y

    grouped = _group(unused)
    assert dict(grouped.reads) == {"a": ("input",), "b": ("a",)}
    assert [operation for operation, _, _ in grouped.unused] == ["aten.ones.default", "aten.sum.default"]


def test_a_unit_reads_exactly_the_parameters_its_node_binds():
    def wrong(module, path, traced):
        return (Unit(path, "conv", {}, {"weight": f"{path}.weight"}, owns=(path,), inputs=(path,), window=1),)

    with pytest.raises(UnsupportedOperation, match=r"a \(conv\): its module reads \['a.bias'\] that the node does "
                                                   r"not bind"):
        group(trace(Pair(lambda y: y).eval(), (1, 2, 8, 8)), {**TABLE, "torch.nn.modules.conv.Conv2d": wrong})


@pytest.mark.parametrize("shape", [(2, 2, 8, 8), (2, 8, 8), (1, 2, 8.0, 8)])
def test_one_chip_is_traced(shape):
    with pytest.raises(UnsupportedOperation, match=r"input_shape is one chip"):
        trace(Pair(lambda y: y).eval(), shape)


class _OneModule:
    """The two things ``_resolve`` asks of a trace: the model input, and each call's module."""

    def __init__(self, model_input):
        self.input = model_input

    def path(self, node):
        return "m"

    def stack(self, node):
        return (("", "root"), ("m", "multi"))


def test_getitem_glue_selects_one_output_of_a_recognized_module():
    """``_resolve``'s ``operator.getitem`` branch, directly: torch.export places a multi-output
    call's selections beside it, inside its unit, so no traced model reaches it (defensive)."""
    graph = torch.fx.Graph()

    def call(target, args, *shapes):
        node = graph.call_function(target, args)
        values = tuple(torch.empty(shape, device="meta") for shape in shapes)
        node.meta["val"] = values[0] if len(values) == 1 else values
        return node

    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((1, 4, 8, 8), device="meta")
    pair = call(torch.ops.aten.max.dim, (x, 1, True), (1, 1, 8, 8), (1, 1, 8, 8))
    first, again, second = (call(operator.getitem, (pair, i), (1, 1, 8, 8)) for i in (0, 0, 1))
    stray = call(operator.getitem, (x, 0), (4, 8, 8))
    grouping = _Grouping(_OneModule(x), {})
    unit = Unit("m", "max", {}, {}, owns=("m",), inputs=("m",), window=1)
    grouping.unit_of[pair] = unit
    outputs = {"m": pair}
    source, axes = grouping._resolve(first, {}, outputs)
    assert source == "m" and axes.canonical() == ((), (), ("H",), ("W",)) and axes.dims == tuple((a,) for a in NCHW)
    assert grouping._resolve(again, {}, outputs)[0] == "m"                  # the same element again
    with pytest.raises(UnsupportedOperation, match="not one recognized module's output"):
        grouping._resolve(second, {}, outputs)                              # a second output of the unit
    with pytest.raises(UnsupportedOperation, match="not one recognized module's output"):
        grouping._resolve(stray, {}, outputs)                               # not a unit's output


# ---------------------------------------------------------------------------
# Fix round: forward code
# ---------------------------------------------------------------------------
@needs_smp
@pytest.mark.parametrize("encoder", ["mit_b0", "mit_b1"])
def test_the_committed_fingerprints_are_those_of_the_validated_trace(encoder):
    """How ``FINGERPRINTS`` were made: the six distinct fingerprints of mit_b0's 18 units (and
    mit_b1's), each unit matching its recognizer kind's."""
    grouped = group(trace(segformer(encoder), SHAPE), RECOGNIZERS)
    assert len(grouped.units) == 18 and grouped.differences == {}
    assert set(grouped.fingerprints.values()) == set(FINGERPRINTS.values()) and len(FINGERPRINTS) == 6
    for unit in grouped.units:
        assert grouped.fingerprints[unit.id] == unit.forward.calls and unit.forward.source == "SMP 0.5.0"
    kinds = {unit.id: next(k for k, calls in FINGERPRINTS.items() if calls == unit.forward.calls)
             for unit in grouped.units}
    assert kinds["encoder.block1.0"] == "mit_block" and kinds["encoder.block4.1"] == "mit_block_without_reduction"
    assert sum(kind == "overlap_embedding" for kind in kinds.values()) == 4


def _decoder(align_corners=False, mode="bilinear", reverse=False):
    """SMP 0.5.0's ``SegformerDecoder.forward``, with one change."""
    def forward(self, features):
        target_size = [dim // 4 for dim in features[0].shape[2:]]
        features = features[2:] if features[1].size(1) == 0 else features[1:]
        features = features[::-1]
        resized = []
        for i, mlp_layer in enumerate(self.mlp_stage):
            options = {"align_corners": align_corners} if mode == "bilinear" else {}
            resized.append(F.interpolate(mlp_layer(features[i]), size=target_size, mode=mode, **options))
        return self.fuse_stage(torch.cat(resized[::-1] if reverse else resized, dim=1))
    return forward


def _block(subtract=False, post_norm=False):
    """SMP 0.5.0's ``Block.forward``, with one change."""
    def forward(self, x):
        batch_size, _, height, width = x.shape
        x = x.flatten(2).transpose(1, 2)
        if post_norm:
            x = self.norm1(x + self.drop_path(self.attn(x, height, width)))
            x = self.norm2(x + self.drop_path(self.mlp(x, height, width)))
        else:
            attention = self.drop_path(self.attn(self.norm1(x), height, width))
            x = x - attention if subtract else x + attention
            x = x + self.drop_path(self.mlp(self.norm2(x), height, width))
        return x.transpose(1, 2).view(batch_size, -1, height, width)
    return forward


def _softmax_over_queries(self, x, height, width):
    """SMP 0.5.0's ``Attention.forward`` with the softmax over dim -2."""
    batch_size, N, C = x.shape
    q = self.q(x).reshape(batch_size, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    if self.sr_ratio > 1:
        x_ = x.permute(0, 2, 1).reshape(batch_size, C, height, width)
        x_ = self.norm(self.sr(x_).reshape(batch_size, C, -1).permute(0, 2, 1))
    else:
        x_ = x
    kv = self.kv(x_).reshape(batch_size, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    k, v = kv[0], kv[1]
    attn = self.attn_drop(((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-2))
    return self.proj_drop(self.proj((attn @ v).transpose(1, 2).reshape(batch_size, N, C)))


def _gelu_first(self, x, height, width):
    """SMP 0.5.0's ``Mlp.forward`` with GELU before the depthwise convolution."""
    x = self.dwconv(self.act(self.fc1(x)), height, width)
    return self.drop(self.fc2(self.drop(x)))


def _sigmoid(self, x):
    return torch.sigmoid(x)


FORWARD_PATCHES = {
    # name: (module to patch, its new forward, the unit, its class, the first difference)
    "interpolate align_corners=True": (
        lambda m: m.decoder, _decoder(align_corners=True), "decoder.pyramid", "SegformerDecoder",
        r"aten\.upsample_bilinear2d\.vec\(align_corners=True\) where the validated trace has "
        r"aten\.upsample_bilinear2d\.vec\(align_corners=False\)"),
    "interpolate nearest": (lambda m: m.decoder, _decoder(mode="nearest"), "decoder.pyramid", "SegformerDecoder",
                            r"call 5 is aten\.upsample_nearest2d"),
    "concatenation reversed": (lambda m: m.decoder, _decoder(reverse=True), "decoder.pyramid", "SegformerDecoder",
                               r"the concatenation's input 0 comes from \['decoder\.mlp_stage\.3'\], where the "
                               r"validated trace has decoder\.mlp_stage\.0"),
    "attention residual subtracted": (lambda m: m.encoder.block2[0], _block(subtract=True), "encoder.block2.0",
                                      "Block", r"aten\.sub\.Tensor where the validated trace has aten\.add\.Tensor"),
    "post-norm": (lambda m: m.encoder.block1[1], _block(post_norm=True), "encoder.block1.1", "Block",
                  r"call 2 is aten\.linear\.default where the validated trace has aten\.layer_norm\.default"),
    "softmax over dim -2": (lambda m: m.encoder.block3[0].attn, _softmax_over_queries, "encoder.block3.0", "Block",
                            r"aten\.softmax\.int\(dim=-2\) where the validated trace has aten\.softmax\.int\(dim=-1\)"),
    "GELU before the depthwise convolution": (lambda m: m.encoder.block4[1].mlp, _gelu_first, "encoder.block4.1",
                                              "Block", r"aten\.gelu\.default\(approximate='none'\) where the "
                                              r"validated trace has aten\.transpose\.int"),
    "head activation sigmoid": (lambda m: m.segmentation_head[2], _sigmoid, "decoder.head", "SegmentationHead",
                                r"call 5 is aten\.sigmoid\.default, after the validated trace's last call"),
}


def _patched(name):
    model = segformer()
    module, forward = FORWARD_PATCHES[name][:2]
    target = module(model)
    target.forward = types.MethodType(forward, target)
    return model


@needs_smp
@pytest.mark.parametrize("name", list(FORWARD_PATCHES))
def test_a_changed_forward_is_refused_by_default(name):
    """The review's eight forward patches, no attribute changed: each refused, naming the unit,
    its class and the first differing call."""
    _, _, unit, kind, difference = FORWARD_PATCHES[name]
    with pytest.raises(UnsupportedOperation, match=rf"^{re.escape(unit)} \(.*{kind}\).*forward code differs from the "
                       rf"validated SMP 0\.5\.0 trace: .*{difference}") as refusal:
        convert(_patched(name), SHAPE, SOURCE_GRID)
    assert refusal.value.code == "forward_code"


@needs_smp
@pytest.mark.parametrize("name", list(FORWARD_PATCHES))
def test_a_changed_forward_converts_outside_its_record_on_request(name):
    _, _, unit, _, difference = FORWARD_PATCHES[name]
    report = convert(_patched(name), SHAPE, SOURCE_GRID, forward_mismatch="report").report
    assert report.outside == (unit,) and not report.inside
    node = next(node for node in report.nodes if node.id == unit)
    assert node.check.inside and node.validated is False
    assert node.forward.startswith("forward code differs from the validated SMP 0.5.0 trace: ")
    assert node.verdict().startswith("outside its validated record: forward code differs from the validated SMP 0.5.0 "
                                     "trace")


@needs_smp
def test_forward_mismatch_takes_refuse_or_report():
    with pytest.raises(ValueError, match="forward_mismatch is one of"):
        convert(segformer(), SHAPE, SOURCE_GRID, forward_mismatch="ignore")


def test_the_report_gives_the_installed_versions_for_information(b0):
    import importlib.metadata
    report = b0.report
    assert dict(report.traced_with) == TRACED_WITH == {"segmentation-models-pytorch": "0.5.0", "torch": "2.14.0"}
    assert dict(report.versions) == {name: importlib.metadata.version(name) for name in TRACED_WITH}
    assert report.lines()[-1].startswith("versions, for information: segmentation-models-pytorch ")
