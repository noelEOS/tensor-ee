"""X04: the converter for timm Swin and SMP UPerNet (``tensor_ee/converter/``: ``export``,
``recognizers``, ``operations``, ``convert``; docs/tasks/X04-swin-upernet-converter.md).

- **R1, granularity.** One node per timm block and SMP decoder module: ``patch_embed``,
  ``swin_block`` / ``shifted_swin_block`` per ``SwinTransformerBlock``, ``patch_merge`` per
  ``PatchMerging``; the PSP, each FPN level's norm, lateral, merge and 3 x 3 block, the fusion,
  the head and its dense form. Only the PSP is ``EXTENT``.
- **R2, recognizers**, by qualified class name, reading and checking the attributes that decide
  the arithmetic; timm's NHWC features are checked glue. The graph lies on the model's abstract
  token grid (no relabel of a raster onto one exists in the library).
- **R2b, forward code.** Each kind's unit matches the fingerprint of the validated timm 1.0.30 /
  SMP 0.5.0 trace, with the axes its calls name; dataflow checks pin the cyclic shift, the PSP's
  and the fusion's concatenation order and each FPN merge's state and lateral. Nine forward
  patches, no attribute changed, are refused by default and reported (or refused by shapes)
  under ``forward_mismatch="report"``. timm's relative-position index and shifted-window mask
  are checked, not bound (``Unit.derived``). Fix round: every convolution's stride, padding,
  dilation and groups and every adaptive pooling's output size are checked against the module
  running it, in both families; the review's four integer patches are refused (and reported
  under ``report``), and other PSP bins are reported against the PSP's record.
- **R3, the fusion split** is the library's (its one formulation), stated as a setting and, in
  the report, as not read from the model.
- **R4, the gate.** The converted nodes, built from the inputs and parameters the Swin-T224
  adapter passes (captured as X00 captures its targets), give all twelve Swin G1-ext targets
  byte for byte.
- **R5, reports and refusals.** Swin-T224 is inside every record, and so is the capstone's
  configuration; a wider Swin converts and is reported outside; what the library cannot express
  and training mode are refused.
- **R6, the planner on Swin** (reported, not gated): X05's planner under Swin's hand limits.

The tests need SMP 0.5.0 and timm, and skip by name without them, as G1-ext does. Offline only:
models are constructed with random weights (``encoder_weights=None``) and traced with fake
tensors; no forward pass runs, nothing is evaluated, and no cloud call is made.
"""
from __future__ import annotations

import contextlib
import functools
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import types
from unittest import mock

import pytest
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference.segformer_b0.block_graph import SOURCE_GRID  # noqa: E402
from tensor_ee._swin_parameters import ENC_PREFIX, PE_PREFIX, checkpoint_shapes, encoder_prefix  # noqa: E402
from tensor_ee.blocks.window import swin as SWIN  # noqa: E402
from tensor_ee.blocks.window import upernet as UPER  # noqa: E402
from tensor_ee.blocks.window.swin_upernet_config import SWIN_T224  # noqa: E402
from tensor_ee.converter import build as BUILD  # noqa: E402
from tensor_ee.converter.convert import INPUT, LOGITS, convert  # noqa: E402
from tensor_ee.converter.export import NHWC, group, trace  # noqa: E402
from tensor_ee.converter.graph import BlockGraph  # noqa: E402
from tensor_ee.converter.operations import OPERATIONS  # noqa: E402
from tensor_ee.converter.plan import Limits, plan  # noqa: E402
from tensor_ee.converter.recognizers import (ALL_RECOGNIZERS, FINGERPRINTS, RECOGNIZERS,  # noqa: E402
                                             SWIN_FINGERPRINTS, SWIN_RECOGNIZERS, SWIN_TRACED_WITH)
from tensor_ee.core.errors import GridMismatch, UnsupportedOperation  # noqa: E402
from tensor_ee.core.grid import Grid, TokenGrid  # noqa: E402
from tensor_ee.core.metadata import Locality  # noqa: E402
from tensor_ee.core.tensor import Raster, WindowTensor  # noqa: E402
from tools import graph_identity as GI  # noqa: E402
from tools import segformer_b0_request_plan as P  # noqa: E402

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[1]
SHAPE = (1, 2, 224, 224)
ENCODER = "tu-swin_tiny_patch4_window7_224"
GOLDEN = json.loads(GI.GOLDEN_EXT.read_text())
NS = P.PLACEHOLDER_NAMESPACE
SMP_REASON = GI.ext_unavailable("segformer_b0", ("segmentation_models_pytorch", "timm"))
needs_smp = pytest.mark.skipif(SMP_REASON is not None, reason=f"SMP UPerNet not converted: {SMP_REASON}")
DEPTHS = SWIN_T224.depths
C = {k: f"encoder.model.layers_{k - 1}.blocks.{DEPTHS[k - 1] - 1}" for k in (1, 2, 3, 4)}
"""The edges C1 .. C4: each stage's last block."""
NEW_OPERATIONS = ("patch_embed", "swin_block", "shifted_swin_block", "patch_merge", "psp_forward", "fpn_layer_norm",
                  "fpn_lateral", "fpn_merge", "fpn_conv_block", "fusion_forward", "head_logits", "head_dense")


def upernet(train: bool = False, **kwargs):
    """SMP's UPerNet with a timm Swin-T encoder and random weights, in evaluation mode unless
    ``train``; ``kwargs`` reach SMP (and through it timm)."""
    import segmentation_models_pytorch as smp
    torch.manual_seed(0)
    arguments = {"encoder_name": ENCODER, "encoder_weights": None, "in_channels": 2, "classes": 2, **kwargs}
    model = smp.UPerNet(**arguments)
    return model if train else model.eval()


@pytest.fixture(scope="module")
def swin():
    if SMP_REASON:
        pytest.skip(f"SMP UPerNet not converted: {SMP_REASON}")
    return convert(upernet(), SHAPE, bands=("VV", "VH"), name="swin_t224")


def flat_parameters(parameters) -> dict:
    """The adapter's three maps (``SwinTParameters``) under the SMP state_dict names a converted
    graph binds (``_swin_parameters.checkpoint_shapes``' names)."""
    flat = {}
    for group_name, values in parameters.encoder.items():
        prefix = ENC_PREFIX + encoder_prefix(group_name).replace("layers.", "layers_", 1)
        flat.update({prefix + name: value for name, value in values.items()})
    flat.update({PE_PREFIX + name: value for name, value in parameters.patch_embed.items()})
    flat.update(parameters.decoder)
    return flat


# ---------------------------------------------------------------------------
# R1 and R2: the nodes, the interface, the parameters and the cost
# ---------------------------------------------------------------------------
def test_swin_t224_is_one_node_per_timm_block_and_smp_module(swin):
    graph = swin.graph
    ops = [node.op for node in graph.nodes]
    assert {op: ops.count(op) for op in NEW_OPERATIONS} == {
        "patch_embed": 1, "swin_block": 7, "shifted_swin_block": 5, "patch_merge": 3, "psp_forward": 1,
        "fpn_layer_norm": 3, "fpn_lateral": 3, "fpn_merge": 3, "fpn_conv_block": 3, "fusion_forward": 1,
        "head_logits": 1, "head_dense": 1}
    assert len(graph.nodes) == 32 and len(graph.edges) == 33
    nodes = {node.id: node for node in graph.nodes}
    # Stage 4 is one window: timm runs both its blocks unshifted, as the library's stage rule does.
    assert [nodes[f"encoder.model.layers_{s}.blocks.{b}"].op for s in range(4) for b in range(DEPTHS[s])] == [
        "swin_block", "shifted_swin_block"] * 5 + ["swin_block", "swin_block"]
    assert dict(nodes["encoder.model.patch_embed"].settings) == {"dim": 96, "patch": 4, "window": 7}
    assert dict(nodes["encoder.model.layers_2.blocks.3"].settings) == {"dim": 384, "heads": 12, "shift": 3}
    assert dict(nodes["encoder.model.layers_3.downsample"].settings) == {"bias": False, "channels": 384}
    assert dict(nodes["decoder.psp"].settings) == {"decoder": 256, "sizes": (1, 2, 3, 6),
                                                   "stage_dims": (96, 192, 384, 768)}
    assert nodes["decoder.fusion"].inputs == ("decoder.fpn_conv_blocks.2", "decoder.fpn_conv_blocks.1",
                                              "decoder.fpn_conv_blocks.0", "decoder.psp")
    assert nodes["decoder.fpn_lateral_blocks.1"].inputs == ("decoder.fpn_lateral_blocks.0",
                                                            "decoder.fpn_lateral_blocks.1.conv_norm_relu")
    assert dict(nodes["segmentation_head"].settings) == {"classes": 2, "decoder": 256, "head_factor": 4}
    assert [graph.edges[C[k]].grid for k in (1, 2, 3, 4)] == [TokenGrid(56, 56, 10.0), TokenGrid(28, 28, 20.0),
                                                            TokenGrid(14, 14, 40.0), TokenGrid(7, 7, 80.0)]
    # Granularity (R1): only the PSP pools the whole extent; the SW-MSA blocks and the head are
    # chip-edge dependent, so no stage holding them is tiled.
    metadata = {node.id: OPERATIONS[node.op].metadata for node in graph.nodes}
    assert [i for i, m in metadata.items() if m.locality is Locality.EXTENT] == ["decoder.psp"]
    assert sorted(i for i, m in metadata.items() if m.chip_edge_dependent) == sorted(
        [f"encoder.model.layers_{s}.blocks.{b}" for s in range(3) for b in range(1, DEPTHS[s], 2)]
        + ["segmentation_head"])


def test_the_graph_lies_on_the_models_abstract_token_grid(swin):
    """A Swin starts from a raster: the library relabels a raster onto a token grid only as windows,
    so the graph starts from the model's input on its token grid and ends there (the recipe
    relabels, contract section 11)."""
    graph = swin.graph
    assert graph.inputs == (INPUT,) and graph.outputs == (LOGITS,)
    assert graph.edges[INPUT].to_json() == {"layout": "R", "grid": {"kind": "token", "height": 224, "width": 224,
                                                                      "scale": 2.5, "crs": "EPSG:3857"},
                                            "channels": 2, "window": None, "bands": ["VV", "VH"]}
    assert graph.edges[LOGITS].layout == "R" and graph.edges[LOGITS].bands == ("c0", "c1")
    assert graph.edges[LOGITS].grid == TokenGrid(224, 224, 2.5)
    assert next(node for node in graph.nodes if node.id == LOGITS).op == "head_dense"
    assert graph.name == "swin_t224" and BlockGraph.loads(graph.dumps()) == graph
    default = convert(upernet(), SHAPE).graph
    assert default.name == "upernet" and default.edges[INPUT].bands == ("b0", "b1")
    with pytest.raises(UnsupportedOperation, match="relabels a raster onto a token grid only as windows"):
        convert(upernet(), SHAPE, Grid("EPSG:3857", (20, 0, 1000, 0, -20, 2000), 224, 224))
    assert convert(upernet(), SHAPE, scale=10).graph.edges[C[4]].grid == TokenGrid(7, 7, 320.0)


@needs_smp
def test_parameters_are_the_state_dict_names_each_bound_once(swin):
    bound = [name for node in swin.graph.nodes for name in node.parameters.values()]
    state = upernet().state_dict()
    assert len(bound) == len(set(bound)) == 239
    assert set(bound) == {name for name in state if not name.endswith("num_batches_tracked")}
    assert set(bound) == {name for name in checkpoint_shapes() if not name.endswith("num_batches_tracked")}
    # timm's relative-position indices and masks are non-persistent buffers: the library derives them.
    assert not any("relative_position_index" in name or "attn_mask" in name for name in bound)


def test_the_cost_is_the_composites_cost_functions(swin):
    """Per node from the blocks' METADATA, summed: the Swin-T224 model's composite costs."""
    expected = (SWIN.swin_patch_embedding_values() + SWIN.swin_encoder_values(norm=False)
                + UPER.fpn_forward_values(7, 7) + UPER.fusion_forward_values(7, 7) + UPER.head_forward_values(56, 56))
    arrays = (SWIN.swin_patch_embedding_arrays() + SWIN.swin_encoder_arrays(norm=False) + UPER.fpn_forward_arrays()
              + UPER.fusion_forward_arrays() + UPER.head_forward_arrays())
    assert swin.graph.cost() == {"values": expected, "asset_arrays": arrays} == {"values": 40_364_976,
                                                                                 "asset_arrays": 239}


def test_the_registry_gains_one_library_function_per_operation():
    """Additive: twelve entries, each one library function held by reference, with its block's
    METADATA; SegFormer's seven are the X02a entries."""
    import importlib
    assert list(OPERATIONS)[:7] == ["relabel_to_windows", "overlap_embedding", "mit_block", "channel_norm",
                                    "project_pyramid", "fuse_head", "relabel_from_windows"]
    assert tuple(OPERATIONS)[7:] == NEW_OPERATIONS
    for name in NEW_OPERATIONS:
        op = OPERATIONS[name]
        module = importlib.import_module(op.function.__module__)
        assert op.name == name and getattr(module, name) is op.function
        assert op.function.__module__.rsplit(".", 1)[-1] in ("patch", "swin", "upernet")
        assert op.metadata is module.METADATA[name]
        assert (op.constants is None) == (name in ("fpn_merge", "head_dense"))


def test_the_recognizer_tables_and_the_converter_load_neither_smp_nor_timm():
    code = ("import sys; import tensor_ee.converter.convert; "
            "print(sorted(m for m in sys.modules if m.split('.')[0] in ('segmentation_models_pytorch', 'timm')))")
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=300,
                            env={**os.environ, "PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": "2"})
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
    assert sorted(name.rsplit(".", 1)[-1] for name in SWIN_RECOGNIZERS) == [
        "FeatureListNet", "PatchEmbed", "PatchMerging", "Sequential", "SwinTransformerBlock", "SwinTransformerStage",
        "TimmUniversalEncoder", "UPerNet"]
    assert ALL_RECOGNIZERS == {**RECOGNIZERS, **SWIN_RECOGNIZERS} and len(RECOGNIZERS) == 6


# ---------------------------------------------------------------------------
# R2b: forward code
# ---------------------------------------------------------------------------
@needs_smp
def test_the_fingerprints_are_the_validated_traces():
    """How ``SWIN_FINGERPRINTS`` were made: the eleven distinct fingerprints (with axes) of the
    Swin-T224/UPerNet's 31 units, one per operation."""
    grouped = group(trace(upernet(), SHAPE), ALL_RECOGNIZERS)
    assert len(grouped.units) == 31
    kinds = {unit.op: set() for unit in grouped.units}
    for unit in grouped.units:
        kinds[unit.op].add(grouped.fingerprints[unit.id])
        assert unit.forward.calls == SWIN_FINGERPRINTS[unit.op] and unit.forward.axes
    assert {op: len(prints) for op, prints in kinds.items()} == {op: 1 for op in SWIN_FINGERPRINTS}
    assert {op: next(iter(prints)) for op, prints in kinds.items()} == SWIN_FINGERPRINTS
    assert len(FINGERPRINTS) == 6 and not set(FINGERPRINTS) & set(SWIN_FINGERPRINTS)
    # timm's features are channels-last: the patch embedding writes NHWC, which the blocks read.
    units = {unit.op: unit for unit in grouped.units}
    assert units["patch_embed"].writes == NHWC and units["swin_block"].axes == NHWC
    assert units["psp_forward"].axes == units["head_logits"].writes == ("N", "C", "H", "W")


def _block_attn(back, forward):
    """timm 1.0.30's ``SwinTransformerBlock._attn`` with other cyclic shifts: forward code only.
    timm is imported when the patched block runs (only in tests that need SMP and timm), so this
    module collects without timm."""

    def _attn(self, x):
        from timm.models.swin_transformer import window_partition, window_reverse
        _, height, width, channels = x.shape
        shifted = torch.roll(x, shifts=back, dims=(1, 2))
        pad_h = (self.window_size[0] - height % self.window_size[0]) % self.window_size[0]
        pad_w = (self.window_size[1] - width % self.window_size[1]) % self.window_size[1]
        shifted = F.pad(shifted, (0, 0, 0, pad_w, 0, pad_h))
        _, hp, wp, _ = shifted.shape
        windows = window_partition(shifted, self.window_size).view(-1, self.window_area, channels)
        out = self.attn(windows, mask=self.attn_mask).view(-1, self.window_size[0], self.window_size[1], channels)
        shifted = window_reverse(out, self.window_size, hp, wp)[:, :height, :width, :].contiguous()
        return torch.roll(shifted, shifts=forward, dims=(1, 2))
    return _attn


def _merge_transposed(self, x):
    """``PatchMerging.forward`` with its 2 x 2 groups in another order."""
    batch, height, width, channels = x.shape
    x = F.pad(x, (0, 0, 0, width % 2, 0, height % 2))
    _, height, width, _ = x.shape
    x = x.reshape(batch, height // 2, 2, width // 2, 2, channels).permute(0, 1, 3, 2, 4, 5).flatten(3)
    return self.reduction(self.norm(x))


def _psp_bins_reversed(self, feature):
    _, _, height, width = feature.shape
    parts = [F.interpolate(block(feature), size=(height, width), mode="bilinear", align_corners=False)
             for block in self.blocks]
    return self.out_conv(torch.cat([feature] + parts[::-1], dim=1))


def _decoder(order=(3, 2, 1, 0), psp_state=False):
    """``UPerNetDecoder.forward`` with its concatenation in ``order`` of the resized features
    (psp, P3, P2, P1; SMP's is reversed: 3, 2, 1, 0), or every FPN merge reading the PSP."""

    def forward(self, features):
        features = features[2:]
        for i, norm in enumerate(self.feature_norms):
            features[i] = norm(features[i])
        fpn = [self.psp(features[-1])]
        for i, block in enumerate(self.fpn_lateral_blocks):
            fpn.append(block(fpn[0] if psp_state else fpn[-1], features[:-1][::-1][i]))
        for i, conv in enumerate(self.fpn_conv_blocks, start=1):
            fpn[i] = conv(fpn[i])
        resized = [F.interpolate(f, size=fpn[-1].shape[2:], mode="bilinear", align_corners=False) for f in fpn]
        return self.fusion_block(torch.cat([resized[i] for i in order], dim=1))
    return forward


FORWARD_PATCHES = {
    # name: (the patched module, its forward or _attn, the unit, its class, the first difference,
    #        under report: None (outside, that unit only) or (exception, message))
    "cyclic shift reversed": (lambda m: m.encoder.model.layers_0.blocks[1], ("_attn", _block_attn((3, 3), (-3, -3))),
                              "encoder.model.layers_0.blocks.1", "SwinTransformerBlock",
                              r"its cyclic shifts are \[\(3, 3\), \(-3, -3\)\], where the validated trace rolls by "
                              r"\[\(-3, -3\), \(3, 3\)\]", None),
    "cyclic shift by 2": (lambda m: m.encoder.model.layers_2.blocks[3], ("_attn", _block_attn((-2, -2), (2, 2))),
                          "encoder.model.layers_2.blocks.3", "SwinTransformerBlock",
                          r"its cyclic shifts are \[\(-2, -2\), \(2, 2\)\]", None),
    "residual subtracted": (lambda m: m.encoder.model.layers_1.blocks[0],
                            ("forward", lambda self, x: (lambda y: y + self.mlp(self.norm2(y)))(
                                x - self._attn(self.norm1(x)))),
                            "encoder.model.layers_1.blocks.0", "SwinTransformerBlock",
                            r"call \d+ is aten\.sub\.Tensor where the validated trace has aten\.add\.Tensor", None),
    "softmax over dim -2": (lambda m: m.encoder.model.layers_2.blocks[0].attn.softmax,
                            ("forward", lambda self, x: torch.softmax(x, dim=-2)),
                            "encoder.model.layers_2.blocks.0", "SwinTransformerBlock",
                            r"aten\.softmax\.int\(dim=-2\) where the validated trace has aten\.softmax\.int\(dim=-1\)",
                            None),
    "merge groups transposed": (lambda m: m.encoder.model.layers_1.downsample, ("forward", _merge_transposed),
                                "encoder.model.layers_1.downsample", "PatchMerging",
                                r"call 2 is aten\.permute\.default\(dims=\[0, 1, 3, 2, 4, 5\]\) where the validated "
                                r"trace has aten\.permute\.default\(dims=\[0, 1, 3, 4, 2, 5\]\)", None),
    "PSP bins reversed": (lambda m: m.decoder.psp, ("forward", _psp_bins_reversed), "decoder.psp", "PSPModule",
                          r"the concatenation's input 1 comes from 'decoder\.psp\.blocks\.3\.1\.2', where the "
                          r"validated trace has decoder\.psp\.blocks\.0", None),
    "fusion P2 and P3 swapped": (lambda m: m.decoder, ("forward", _decoder(order=(3, 1, 2, 0))), "decoder.fusion",
                                 "UPerNetDecoder", r"the concatenation's input 1 comes from "
                                 r"'decoder\.fpn_conv_blocks\.0\.2', where the validated trace has "
                                 r"decoder\.fpn_conv_blocks\.1", None),
    "FPN state from the PSP": (lambda m: m.decoder, ("forward", _decoder(psp_state=True)),
                               "decoder.fpn_lateral_blocks.1", "FPNLateralBlock",
                               r"its state comes from 'decoder\.psp\.out_conv\.2', where the validated trace has "
                               r"decoder\.fpn_lateral_blocks\.0",
                               (GridMismatch, "fpn_merge upsamples its state twice onto its lateral's grid")),
    "head align_corners=False": (lambda m: m.segmentation_head[1],
                                 ("forward", lambda self, x: F.interpolate(x, scale_factor=4.0, mode="bilinear",
                                                                           align_corners=False)),
                                 "segmentation_head", "SegmentationHead",
                                 r"aten\.upsample_bilinear2d\.vec\(align_corners=False\) where the validated trace has "
                                 r"aten\.upsample_bilinear2d\.vec\(align_corners=True\)", None),
}


def _patched(name):
    model = upernet()
    module, (attribute, function) = FORWARD_PATCHES[name][:2]
    target = module(model)
    setattr(target, attribute, types.MethodType(function, target))
    return model


@needs_smp
@pytest.mark.parametrize("name", list(FORWARD_PATCHES))
def test_a_changed_forward_is_refused_by_default(name):
    """Nine forward patches, no attribute changed: each refused, naming the unit, its class and
    the first difference (the fingerprint's, with axes, or a dataflow check's)."""
    _, _, unit, kind, difference, _ = FORWARD_PATCHES[name]
    with pytest.raises(UnsupportedOperation, match=rf"^{re.escape(unit)} \(.*{kind}\).*forward code differs from the "
                       rf"validated timm 1\.0\.30 and SMP 0\.5\.0 trace: .*{difference}") as refusal:
        convert(_patched(name), SHAPE)
    assert refusal.value.code == "forward_code"


@needs_smp
@pytest.mark.parametrize("name", list(FORWARD_PATCHES))
def test_a_changed_forward_converts_outside_its_record_on_request(name):
    _, _, unit, _, _, refused = FORWARD_PATCHES[name]
    if refused is not None:
        error, message = refused
        with pytest.raises(error, match=message):
            convert(_patched(name), SHAPE, forward_mismatch="report")
        return
    report = convert(_patched(name), SHAPE, forward_mismatch="report").report
    assert report.outside == (unit,) and not report.inside
    node = next(node for node in report.nodes if node.id == unit)
    assert node.check.inside and node.validated is False
    assert node.verdict().startswith("outside its validated record: forward code differs from the validated timm "
                                     "1.0.30 and SMP 0.5.0 trace: ")


@needs_smp
def test_derived_buffers_are_non_persistent_buffers_the_calls_read():
    """A unit leaves unbound only the non-persistent buffers its recognizer checked; a stored
    parameter declared derived, or a derived buffer left undeclared, is refused."""
    from dataclasses import replace
    from tensor_ee.converter import recognizers as R
    traced = trace(upernet(), SHAPE)

    def declaring(extra):
        def recognize(module, path, traced):
            (unit,) = R.swin_transformer_block(module, path, traced)
            return (replace(unit, derived=extra(unit.derived, path)),)
        return {**ALL_RECOGNIZERS, R.SWIN_BLOCK: recognize}

    stored = declaring(lambda derived, path: derived + (f"{path}.norm1.weight",))
    with pytest.raises(UnsupportedOperation, match=r"derived buffers are non-persistent buffers its calls read; "
                       r"\['encoder\.model\.layers_\d\.blocks\.\d\.norm1\.weight'\] are stored") as refusal:
        group(traced, stored)
    assert refusal.value.code == "unsupported_setting"
    undeclared = declaring(lambda derived, path: ())
    with pytest.raises(UnsupportedOperation, match=r"its module reads \['encoder\.model\.layers_\d\.blocks\.\d\.attn\."
                                                   r"relative_position_index'.*\] that the node does not bind"):
        group(traced, undeclared)


@needs_smp
def test_the_fusion_converts_its_levels_in_the_patched_order_under_report():
    """The fusion's inputs are found through its concatenation, so a patched order converts as the
    patched model computes: P1, P3, P2, PSP."""
    graph = convert(_patched("fusion P2 and P3 swapped"), SHAPE, forward_mismatch="report").graph
    fusion = next(node for node in graph.nodes if node.id == "decoder.fusion")
    assert fusion.inputs == ("decoder.fpn_conv_blocks.2", "decoder.fpn_conv_blocks.0", "decoder.fpn_conv_blocks.1",
                             "decoder.psp")


# ---------------------------------------------------------------------------
# R3: the fusion split
# ---------------------------------------------------------------------------
def test_the_fusion_split_is_the_librarys_and_the_report_says_so(swin):
    model = upernet()
    assert tuple(model.decoder.fusion_block[0].weight.shape) == (256, 1024, 3, 3)     # one convolution
    fusion = next(node for node in swin.graph.nodes if node.id == "decoder.fusion")
    assert dict(fusion.settings) == {"decoder": 256, "groups": 4}
    check = next(node for node in swin.report.nodes if node.id == "decoder.fusion")
    assert set(check.choices) == {"groups"} and check.validated
    assert check.line().endswith(
        "inside its validated record; groups 4 is not read from the model: PyTorch's fusion is one 3 x 3 "
        "convolution; the library computes it as four input-channel groups, one per pyramid level, summed in "
        "double and rounded once (formulation B, its one formulation, chosen in Milestone 10 for its numerics), "
        "not for capacity. The split is the library's, not a property of the model")
    assert all(not node.choices for node in swin.report.nodes if node.id != "decoder.fusion")


# ---------------------------------------------------------------------------
# Fix round (P2-1): the integers that decide a call's arithmetic, against its module
# ---------------------------------------------------------------------------
def _conv_forward(padding, dilation):
    """``Conv2d.forward`` with another padding and dilation in code; the module's attributes stay."""
    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, padding, dilation, self.groups)
    return forward


INTEGER_PATCHES = {
    # name: (the patched module, its forward, the unit, the difference)
    "PSP branch pooled to 2 x 2": (lambda m: m.decoder.psp.blocks[0][0], lambda self, x: F.adaptive_avg_pool2d(x, 2),
                                   "decoder.psp", r"aten\.adaptive_avg_pool2d\.default in module "
                                   r"'decoder\.psp\.blocks\.0\.0' \(.*AdaptiveAvgPool2d\) takes output_size \(2, 2\), "
                                   r"where the module holds \(1, 1\)"),
    "PSP 1 x 1 conv padded by 1": (lambda m: m.decoder.psp.blocks[0][1][0], _conv_forward(1, 1), "decoder.psp",
                                   r"'decoder\.psp\.blocks\.0\.1\.0' \(.*Conv2d\) takes padding \(1, 1\), where the "
                                   r"module holds \(0, 0\)"),
    "FPN 3 x 3 dilated": (lambda m: m.decoder.fpn_conv_blocks[0][0], _conv_forward(2, 2), "decoder.fpn_conv_blocks.0",
                          r"'decoder\.fpn_conv_blocks\.0\.0' \(.*Conv2d\) takes padding \(2, 2\), where the module "
                          r"holds \(1, 1\)"),
    "fusion dilated": (lambda m: m.decoder.fusion_block[0], _conv_forward(2, 2), "decoder.fusion",
                       r"'decoder\.fusion_block\.0' \(.*Conv2d\) takes padding \(2, 2\), where the module holds "
                       r"\(1, 1\)"),
}
"""The review's integer patches that the structure check closes. Its fifth, a patch merging that
regroups its 2 x 2 neighbourhoods by other reshape sizes with the same axes, stays a declared
limitation: a reshape's sizes vary with the map (``export``'s docstring)."""


def _integer_patched(name):
    model = upernet()
    module, forward = INTEGER_PATCHES[name][:2]
    target = module(model)
    target.forward = types.MethodType(forward, target)
    return model


@needs_smp
@pytest.mark.parametrize("name", list(INTEGER_PATCHES))
def test_an_integer_changed_in_code_is_refused_by_default(name):
    _, _, unit, difference = INTEGER_PATCHES[name]
    with pytest.raises(UnsupportedOperation, match=rf"^{re.escape(unit)} \(.*forward code differs from the validated "
                       rf"timm 1\.0\.30 and SMP 0\.5\.0 trace: its .*{difference}") as refusal:
        convert(_integer_patched(name), SHAPE)
    assert refusal.value.code == "forward_code"


@needs_smp
@pytest.mark.parametrize("name", list(INTEGER_PATCHES))
def test_an_integer_changed_in_code_converts_outside_its_record_on_request(name):
    unit = INTEGER_PATCHES[name][2]
    report = convert(_integer_patched(name), SHAPE, forward_mismatch="report").report
    assert report.outside == (unit,)
    node = next(node for node in report.nodes if node.id == unit)
    assert node.check.inside and node.validated is False


@needs_smp
def test_other_bins_are_reported_against_the_psps_record_not_refused():
    """A PSP whose modules pool to other bins converts: its ``sizes`` setting holds them, the bins'
    dataflow check agrees, and the report puts the node outside its record on ``sizes``."""
    model = _replace("decoder.psp.blocks.3.0", nn.AdaptiveAvgPool2d(4))(upernet())
    conversion = convert(model, SHAPE)
    psp = next(node for node in conversion.report.nodes if node.id == "decoder.psp")
    assert next(node for node in conversion.graph.nodes if node.id == "decoder.psp").settings["sizes"] == (1, 2, 3, 4)
    assert conversion.report.outside == ("decoder.psp",) and psp.forward is None and psp.check.outside == ("sizes",)


def test_an_fpn_nodes_constants_are_the_librarys_with_its_shape_checks(swin):
    """P2-3: one implementation. ``fpn_constants`` builds every level with ``fpn_level_constants``,
    which the registry calls for one node's part, so the node refuses a parameter of the wrong
    shape as the library does."""
    import inspect
    from tensor_ee.converter import operations
    assert "fpn_level_constants(" in inspect.getsource(UPER.fpn_constants)
    assert "fpn_level_constants(" in inspect.getsource(operations._fpn_constants)
    assert operations.fpn_level_constants is UPER.fpn_level_constants
    with GI.offline_earth_engine() as attempts:
        import ee
        from tensor_ee.core.tensor import ServerArray
        flat = flat_parameters(GI._swin_placeholder_parameters())
        flat["decoder.feature_norms.2.weight"] = ServerArray(ee.Array([0.0] * 100), (100,))
        c3 = WindowTensor(ee.Image(0), swin.graph.edges[C[3]].grid, 384, 7)
        with pytest.raises(ValueError, match=r"fpn level 3: decoder\.feature_norms\.2\.weight has shape \(100,\); "
                                             r"the configuration needs \(384,\)"):
            BUILD.build(swin.graph, flat, {C[3]: c3}, outputs=("decoder.feature_norms.2",))
    assert attempts == []


@needs_smp
def test_the_structure_check_covers_segformer_too():
    """SMP SegFormer's Mix-FFN depthwise convolution dilated in code, its module unchanged."""
    import segmentation_models_pytorch as smp
    torch.manual_seed(0)
    model = smp.Segformer("mit_b0", encoder_weights=None, in_channels=2, classes=2).eval()
    target = model.encoder.block1[0].mlp.dwconv.dwconv
    target.forward = types.MethodType(_conv_forward(2, 2), target)
    with pytest.raises(UnsupportedOperation, match=r"^encoder\.block1\.0 \(mit_block.*forward code differs from the "
                       r"validated SMP 0\.5\.0 trace: its aten\.conv2d\.default in module "
                       r"'encoder\.block1\.0\.mlp\.dwconv\.dwconv' \(.*Conv2d\) takes padding \(2, 2\), where the "
                       r"module holds \(1, 1\)") as refusal:
        convert(model, SHAPE, SOURCE_GRID)
    assert refusal.value.code == "forward_code"


# ---------------------------------------------------------------------------
# R4: the gate
# ---------------------------------------------------------------------------
def adapter_calls():
    """G1's ``swin_adapter`` stages built as X00 builds its Swin targets (``build_swin_targets``),
    with every composite the adapter calls recorded with its arguments and output; run inside the
    offline context. Returns (calls by composite, the adapter's parameters)."""
    import ee
    from tensor_ee import swin
    calls = {}

    def recorder(name, function):
        def wrapper(*args, **kwargs):
            out = function(*args, **kwargs)
            calls.setdefault(name, []).append((args, kwargs, out))
            return out
        return wrapper

    grid = Grid("EPSG:3857", GI.SWIN_GRID_TRANSFORM, 224, 224)
    parameters = GI._swin_placeholder_parameters()
    bands = ("second", "first")
    adapter = swin.SwinT224Adapter(input_bands=bands, output_bands=("land", "water"))

    def raster(values):
        return Raster(ee.Image.constant(values).rename(list(bands)), bands, grid)

    with contextlib.ExitStack() as hooks:
        for module, name, _ in GI._swin_hooks(swin):
            hooks.enter_context(mock.patch.object(module, name, recorder(name, getattr(module, name))))
        GI._swin_stages(adapter, raster([0, 0]), raster([1, 0]), grid, parameters, "test")
    return calls, parameters


def _windows(image, grid, channels):
    return WindowTensor(image, grid, channels, 7)


def _targets(calls, graph):
    """Each Swin G1-ext target: (the edges the converted graph reads, built from the tensors the
    adapter passed to that composite, and the edge whose image is the target)."""
    (pe, _, _), = calls["patch_embed"]
    (encoder_a, _, _), (encoder_b, _, _) = calls["encoder"]
    (fpn, _, _), = calls["fpn_forward"]
    (fusion, _, _), = calls["fusion_forward"]
    (head, _, _), = calls["head_forward"]
    (dense, _, _), = calls["head_dense"]
    stage1 = {"encoder.model.patch_embed": _windows(encoder_a[1], graph.edges["encoder.model.patch_embed"].grid, 96)}
    feats = {C[k]: _windows(fpn[1][k][0], fpn[1][k][1], SWIN_T224.stage_dims[k - 1]) for k in (1, 2, 3, 4)}
    pyramid = dict(zip(("decoder.fpn_conv_blocks.2", "decoder.fpn_conv_blocks.1", "decoder.fpn_conv_blocks.0",
                        "decoder.psp"), (fusion[1][n] for n in ("P1", "P2", "P3", "psp_relu"))))
    return {
        "A/patch_embed": ({INPUT: Raster(pe[1], ("VV", "VH"), pe[2])}, "encoder.model.patch_embed"),
        "A/encoder_stage1": (stage1, C[1]),
        "A/encoder_stage2": (stage1, C[2]),
        "A/encoder_stage3": (stage1, C[3]),
        "B/encoder_stage4": ({C[3]: _windows(encoder_b[1], graph.edges[C[3]].grid, 384)}, C[4]),
        "C/psp": ({C[4]: feats[C[4]]}, "decoder.psp"),
        "C/fpn_P3": ({C[k]: feats[C[k]] for k in (3, 4)}, "decoder.fpn_conv_blocks.0"),
        "C/fpn_P2": ({C[k]: feats[C[k]] for k in (2, 3, 4)}, "decoder.fpn_conv_blocks.1"),
        "C/fpn_P1": (feats, "decoder.fpn_conv_blocks.2"),
        "C/fusion": (pyramid, "decoder.fusion"),
        "C/head_logits": ({"decoder.fusion": _windows(head[1].image, head[1].grid, 256)}, "segmentation_head"),
        "C/head_dense": ({"segmentation_head": _windows(dense[1], dense[2], 2)}, LOGITS),
    }


SPANS = {"A/patch_embed": 1, "A/encoder_stage1": 2, "A/encoder_stage2": 5, "A/encoder_stage3": 12,
         "B/encoder_stage4": 3, "C/psp": 1, "C/fpn_P3": 5, "C/fpn_P2": 8, "C/fpn_P1": 11, "C/fusion": 1,
         "C/head_logits": 1, "C/head_dense": 1}
"""The converted nodes each target spans, from the inputs the adapter passed to its composite: an
encoder stage is its blocks (and from stage 2 its patch merging) after the previous stages'; P_k
is the PSP and, for each level from 3 down to k, the norm, lateral and merge, with level k's
3 x 3 block."""


def _built(graph, parameters, inputs, output):
    """The builder's tensor for ``output``, the W edges' window image itself rather than their
    dense boundary form (the targets are the composites' outputs)."""
    with mock.patch.object(BUILD, "boundary_raster", lambda ee, x: x):
        return BUILD.build(graph, parameters, inputs, outputs=(output,))[output].image


def test_the_converted_nodes_build_the_twelve_swin_g1ext_targets(swin):
    graph, built = swin.graph, {}
    with GI.offline_earth_engine() as attempts:
        calls, parameters = adapter_calls()
        flat = flat_parameters(parameters)
        targets = _targets(calls, graph)
        for target, (inputs, output) in targets.items():
            assert len(graph.stage_nodes(tuple(inputs), (output,))) == SPANS[target]
            built[target] = {"kind": "fixture", **GI.graph_record(_built(graph, flat, inputs, output))}
    assert attempts == []
    assert sorted(built) == sorted(name.split("/", 1)[1] for name in GOLDEN["graphs"] if name.startswith("swin_t224"))
    for target, record in built.items():
        name = f"swin_t224/{target}"
        verdict, message = GI.judge(name, GOLDEN["graphs"][name], record, GOLDEN["environment"])
        assert verdict == "pass", message
        golden = GOLDEN["graphs"][name]
        assert (record["sha256"], record["bytes"]) == (golden["sha256"], golden["bytes"])


@needs_smp
def test_a_changed_node_misses_its_target(swin):
    """The gate bites: the converted graph with one shift changed builds another stage 1."""
    data = swin.graph.to_json()
    node = next(n for n in data["nodes"] if n["id"] == "encoder.model.layers_0.blocks.1")
    node["settings"]["shift"] = 2
    changed = BlockGraph.from_json(data)
    with GI.offline_earth_engine() as attempts:
        calls, parameters = adapter_calls()
        inputs, output = _targets(calls, changed)["A/encoder_stage1"]
        record = GI.graph_record(_built(changed, flat_parameters(parameters), inputs, output))
    assert attempts == []
    assert record["sha256"] != GOLDEN["graphs"]["swin_t224/A/encoder_stage1"]["sha256"]


# ---------------------------------------------------------------------------
# R5: reports and refusals
# ---------------------------------------------------------------------------
def test_swin_t224_is_inside_every_record(swin):
    report = swin.report
    assert report.inside and report.outside == ()
    lines = report.lines()
    assert lines[0] == "swin_t224: 32 nodes; 32 with a validated record, 32 inside, 0 outside; 0 layout operations"
    assert ("encoder.model.layers_0.blocks.1 = shifted_swin_block from encoder.model.layers_0.blocks.1 "
            "(SwinTransformerBlock) [dim 96, heads 3, shift 3, window 7]: inside its validated record") in lines
    assert ("decoder.psp = psp_forward from decoder.feature_norms.3 (LayerNorm2d), decoder.psp (PSPModule) [decoder "
            "256, height 7, sizes (1, 2, 3, 6), stage_dims (96, 192, 384, 768), width 7, window 7]: inside its "
            "validated record") in lines
    assert "logits = head_dense [classes 2, window 7]: inside its validated record" in lines
    assert lines[-4:-1] == ("glue: 4 x aten.permute.default in encoder (TimmUniversalEncoder)",
                            "glue: 4 x aten.contiguous.default in encoder (TimmUniversalEncoder)",
                            "unused: 1 x aten.empty.memory_format in encoder (TimmUniversalEncoder): its result "
                            "reaches no output")
    assert dict(report.traced_with) == SWIN_TRACED_WITH
    assert lines[-1].startswith("versions, for information: timm ")


CAPSTONE = dict(in_channels=3, classes=3, decoder_channels=64, img_size=(256, 256), embed_dim=48, depths=(2, 2, 2, 2),
                num_heads=(2, 4, 8, 16), window_size=8)
"""C1's capstone (tools/capstone.py, ``REFERENCE_ARGUMENTS``): window 8, widths 48 to 384, three
bands and classes, decoder width 64, at 256 x 256."""


@needs_smp
def test_the_capstone_configuration_converts_inside_every_record():
    conversion = convert(upernet(**CAPSTONE), (1, 3, 256, 256))
    assert conversion.report.inside and len(conversion.graph.nodes) == 28
    nodes = {node.id: node for node in conversion.graph.nodes}
    assert conversion.graph.edges["encoder.model.patch_embed"].window == 8
    assert [nodes[f"encoder.model.layers_{s}.blocks.1"].op for s in range(4)] == ["shifted_swin_block"] * 3 + [
        "swin_block"]
    assert dict(nodes["encoder.model.layers_0.blocks.1"].settings) == {"dim": 48, "heads": 2, "shift": 4}
    assert dict(nodes["segmentation_head"].settings) == {"classes": 3, "decoder": 64, "head_factor": 4}


@pytest.fixture(scope="module")
def wider():
    if SMP_REASON:
        pytest.skip(f"SMP UPerNet not converted: {SMP_REASON}")
    return convert(upernet(embed_dim=64, num_heads=(2, 4, 8, 16)), SHAPE)


def test_a_wider_swin_converts_and_is_reported_outside(wider):
    """Widths 64 to 512 (heads 2 to 16): the same structure, outside the records that name widths."""
    report = wider.report
    assert not report.inside and len(report.nodes) == 32
    inside = {node.id for node in report.nodes if node.validated}
    assert inside == {"encoder.model.layers_1.downsample", "encoder.model.layers_2.downsample",
                      "encoder.model.layers_3.downsample", "decoder.fpn_lateral_blocks.0",
                      "decoder.fpn_lateral_blocks.1", "decoder.fpn_lateral_blocks.2", "decoder.fpn_conv_blocks.0",
                      "decoder.fpn_conv_blocks.1", "decoder.fpn_conv_blocks.2", "decoder.fusion",
                      "segmentation_head", LOGITS}
    lines = report.lines()
    assert lines[0] == "upernet: 32 nodes; 32 with a validated record, 12 inside, 20 outside; 0 layout operations"
    block = next(node for node in report.nodes if node.id == "encoder.model.layers_0.blocks.0")
    assert block.verdict() == (
        "outside its validated record: no configuration holds dim 64, heads 2, window 7; nearest: "
        "{dim 96; heads 3; window 7}: dim 64 is not 96; heads 2 is not 3 | "
        "{dim 192; heads 6; window 7}: dim 64 is not 192; heads 2 is not 6 | "
        "{dim 384; heads 12; window 7}: dim 64 is not 384; heads 2 is not 12 | "
        "{dim 768; heads 24; window 7}: dim 64 is not 768; heads 2 is not 24 | "
        "{dim 48; heads 2; window 8}: dim 64 is not 48; window 7 is not 8")
    embed = next(node for node in report.nodes if node.id == "encoder.model.patch_embed")
    assert embed.check.outside == ("dim",)
    psp = next(node for node in report.nodes if node.id == "decoder.psp")
    assert psp.check.outside == ("stage_dims",)


def _replace(path, module):
    """``model`` with the submodule at ``path`` replaced by ``module``, in evaluation mode."""
    def edit(model):
        parent, name = path.rsplit(".", 1)
        setattr(model.get_submodule(parent), name, module.eval())
        return model
    return edit


SETTING = "unsupported_setting"
"""A recognizer's refusal: an attribute the library's blocks fix, or a map they cannot hold."""
OPTION = "unsupported_option"
"""A refusal of the registry's shape rule, or of the trace (``reject``'s default code)."""
REFUSALS = {
    # name: (the model, the input shape, the message, the code)
    "qkv without bias": (lambda: upernet(qkv_bias=False), SHAPE,
                         r"encoder\.model\.layers_0\.blocks\.0\.attn\.qkv \(.*Linear\): a biased 96 -> 288 projection",
                         SETTING),
    "MLP ratio 2": (lambda: upernet(mlp_ratio=2.0), SHAPE, r"mlp\.fc1 \(.*Linear\): a biased 96 -> 384 projection",
                    SETTING),
    "norms at eps 1e-6": (lambda: upernet(norm_layer=functools.partial(nn.LayerNorm, eps=1e-6)), SHAPE,
                          r"eps 1e-06, where the library's composite fixes 1e-05", SETTING),
    "heads narrower than the width": (lambda: upernet(head_dim=16), SHAPE,
                                      r"attn \(.*WindowAttention\): scale 0\.25, where the library scales scores by "
                                      r"head width \*\* -0\.5", SETTING),
    "tanh GELU": (lambda: _replace("encoder.model.layers_1.blocks.0.mlp.act", nn.GELU(approximate="tanh"))(upernet()),
                  SHAPE, r"act \(.*GELU\): approximate 'tanh', where the library's GELU is exact", SETTING),
    "patch embedding without its norm": (
        lambda: _replace("encoder.model.patch_embed.norm", nn.Identity())(upernet()), SHAPE,
        r"encoder\.model\.patch_embed\.norm: expected torch\.nn\.modules\.normalization\.LayerNorm; found "
        r"torch\.nn\.modules\.linear\.Identity", SETTING),
    "decoder without BatchNorm": (lambda: upernet(decoder_use_norm=False), SHAPE,
                                  r"decoder\.psp\.blocks\.0\.1\.0 \(.*Conv2d\): .*bias True \(not False\)", SETTING),
    "head activation": (lambda: upernet(activation="sigmoid"), SHAPE,
                        r"segmentation_head\.2\.activation: expected torch\.nn\.modules\.linear\.Identity", SETTING),
    "auxiliary head": (lambda: upernet(aux_params={"classes": 2}), SHAPE, r"the model returns 2 values", OPTION),
    "window 7 at 256": (lambda: upernet(img_size=(256, 256)), (1, 2, 256, 256),
                        r"its 64 x 64 map does not partition into 7 x 7 windows", SETTING),
    "window 8 at 224": (lambda: upernet(window_size=8), SHAPE, r"the model's Swin blocks run windows \[\(7, 7\), "
                                                               r"\(8, 8\)\]", SETTING),
    "stage 4 shifted in one window": (lambda: upernet(always_partition=True), SHAPE,
                                      r"node 'encoder\.model\.layers_3\.blocks\.1': shifted_swin_block: SW-MSA needs "
                                      r"at least two windows along each axis; the map has 1 x 1", OPTION),
}


@needs_smp
@pytest.mark.parametrize("name", list(REFUSALS))
def test_what_the_library_cannot_express_is_refused(name):
    model, shape, message, code = REFUSALS[name]
    with pytest.raises(UnsupportedOperation, match=message) as refusal:
        convert(model(), shape)
    assert refusal.value.code == code


@needs_smp
def test_training_mode_and_unrecognized_modules_are_refused():
    with pytest.raises(UnsupportedOperation, match="set model.eval") as refusal:
        convert(upernet(train=True), SHAPE)
    assert refusal.value.code == "training_mode"
    with pytest.raises(UnsupportedOperation, match=r"module '1' \(torch\.nn\.modules\.activation\.Softmax\) is not a "
                                                   r"recognized module") as refusal:
        convert(nn.Sequential(upernet(), nn.Softmax(dim=1)).eval(), SHAPE)
    assert refusal.value.code == "unrecognized_module"


# ---------------------------------------------------------------------------
# R6: the planner on Swin (reported, not gated)
# ---------------------------------------------------------------------------
def _tool_measure(raster, name) -> int:
    """The sizing tool's complete ExportImageRequest body (X05's measurement)."""
    return P.measure_image_export(raster.image, raster.grid, NS, name)["wire"]["bytes"]


SWIN_HAND = Limits(constants=24_000_000, request_bytes=9_500_000, extent=1)
"""Swin's hand limits: Milestone 7's 24,000,000-constant budget, the Swin workflow's 9.5 MB
request cap, and the EXTENT bound the hand stages hold at this grain (A and B 0, C 1: the PSP)."""
A_WRITES = (C[1], C[2], C[3])
"""Stage A's boundaries, cut edges: C1, C2 and C3 (the adapter packs them into one asset, a recipe
convention; the planner exports one image per edge). B reads C3 and writes C4; C reads C1-C4."""


@pytest.fixture(scope="module")
def plans(swin):
    """X05's planner on the converted graph, with G1's placeholder server arrays (table-asset
    weights, the only transport it measures) and the sizing tool's measurement: under Swin's hand
    limits, with the hand plan's cut edges required, and under the defaults."""
    with GI.offline_earth_engine() as attempts:
        server = flat_parameters(GI._swin_placeholder_parameters())
        runs = {"hand": plan(swin.graph, SWIN_HAND, parameters=server, measure=_tool_measure, namespace=NS),
                "hand, C3 and C4 required": plan(swin.graph, SWIN_HAND, parameters=server, measure=_tool_measure,
                                                 namespace=NS, cuts=(C[3], C[4])),
                "defaults": plan(swin.graph, Limits(), parameters=server, measure=_tool_measure, namespace=NS)}
    assert attempts == []
    return runs


def _summary(planned):
    return (planned.stage.inputs, planned.stage.outputs, len(planned.nodes), dict(planned.cost), planned.extent,
            dict(planned.requests))


def test_the_planner_cuts_swin_t224_in_two_inside_stage_4(plans):
    """Under Swin's hand limits the fewest stages are two, cut after stage 4's first block: A and
    B together hold 30.5 M constants and B with C 26.3 M (Milestone 7's failed 26.38 M), both over
    the 24 M budget, while stage 4's two blocks fit one each side. Requests are far under the cap
    with table-asset weights."""
    hand = plans["hand"]
    s4b0 = "encoder.model.layers_3.blocks.0"
    assert [_summary(s) for s in hand.stages] == [
        ((INPUT,), (C[1], C[2], C[3], s4b0), 15, {"values": 22_972_578, "asset_arrays": 156}, (),
         {C[1]: 126_457, C[2]: 204_642, C[3]: 498_353, s4b0: 604_553}),
        ((C[1], C[2], C[3], s4b0), (LOGITS,), 17, {"values": 17_392_398, "asset_arrays": 83}, ("decoder.psp",),
         {LOGITS: 583_651})]
    assert hand.stages[0].ends == ("merged with stage2: 40,364,976 embedded constants, over the budget of 24,000,000",)
    assert hand.single_graph["within"] == {"constants": False, "request_bytes": None, "extent": True}
    assert hand.search["builds"] == 2
    # The defaults (10 MB, EXTENT bound 2) give the same two stages: neither limit binds here.
    assert [s.stage for s in plans["defaults"].stages] == [s.stage for s in hand.stages]


def test_with_the_hand_plans_cut_edges_required_the_planner_gives_a_b_and_c(plans):
    stages = plans["hand, C3 and C4 required"].stages
    assert [(s.stage.inputs, s.stage.outputs) for s in stages] == [
        ((INPUT,), A_WRITES), ((C[3],), (C[4],)), ((C[1], C[2], C[3], C[4]), (LOGITS,))]
    assert [s.nodes[0] for s in stages] == ["encoder.model.patch_embed", "encoder.model.layers_3.downsample",
                                            "decoder.psp"]
    assert [dict(s.cost) for s in stages] == [{"values": 14_071_242, "asset_arrays": 140},
                                              {"values": 16_472_496, "asset_arrays": 29},
                                              {"values": 9_821_238, "asset_arrays": 70}]
    assert [s.extent for s in stages] == [(), (), ("decoder.psp",)]
    assert stages[0].ends == ("merged with stage2: 30,543,738 embedded constants, over the budget of 24,000,000",
                              "merged with stage2: it crosses the required cut at 'encoder.model.layers_2.blocks.5'")
    assert stages[1].ends == ("merged with stage3: 26,293,734 embedded constants, over the budget of 24,000,000",
                              "merged with stage3: it crosses the required cut at 'encoder.model.layers_3.blocks.1'")
