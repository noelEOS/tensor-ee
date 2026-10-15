"""C0: the configurable Swin-UPerNet composite and record builder, offline.

No network and no Earth Engine call: graphs are built against ``ee.apitestcase.ApiTestCase``
signatures, with stand-in ``ServerArray`` weights (server-side constants, no values in the
request).

- **Defaults.** ``SwinUPerNetConfig()`` is ``SWIN_T224``; the module constants and rule
  functions keep Swin-T224's values. (The golden graphs, ``tools/graph_identity.py``, are the
  decisive check that the Swin-T224 graphs are unchanged.)
- **Refusals.** A value no composite can serve is refused when the configuration is built,
  with the reason; the composites refuse constants, tensors and parameters of another
  configuration.
- **Records.** Swin-T224's 189 records are byte-identical to the base revision's (``ed88e3f``),
  and their names, shapes and bias flags are the trained manifest's; any configuration's
  records follow its schema, whose names and shapes are timm's and SMP's own.
- **The capstone configuration** (docs/tasks/C0-capstone.md): 3 bands on 256 x 256, patch 4,
  width 48, window 8, depths (2, 2, 2, 2), heads (2, 4, 8, 16), decoder width 64, 3 classes.
  Its whole graph builds: the token grids and shapes of every stage, timm's shift at every
  block, the PSP's 8 x 8 extent and the 3-band 256 x 256 output.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import ee
from ee.apitestcase import ApiTestCase
import numpy as np
import pytest

from tensor_ee import _swin_parameters as P
from tensor_ee.blocks.window import swin as SW
from tensor_ee.blocks.window import upernet as UP
from tensor_ee.core.errors import GridMismatch, UnsupportedOperation
from tensor_ee.core.grid import TokenGrid
from tensor_ee.core.ops import BlockImageOps, WindowOps
from tensor_ee.core.tensor import Feature, WindowTensor
from tensor_ee.models.swin_t224 import configuration as C
from tensor_ee.models.swin_t224 import decoder as D
from tensor_ee.models.swin_t224 import encoder as E
from tensor_ee.weights.constants import RowMismatch, ServerArray, WindowLayout

ROOT = Path(__file__).resolve().parents[1]
HAVE_REFERENCES = all(importlib.util.find_spec(m) is not None for m in ("timm", "segmentation_models_pytorch"))
needs_references = pytest.mark.skipif(not HAVE_REFERENCES, reason="the references need timm and SMP")

CAPSTONE = C.SwinUPerNetConfig(window=8, patch=4, bands=("VV", "VH", "VVminusVH"), embed_dim=48,
                               depths=(2, 2, 2, 2), num_heads=(2, 4, 8, 16), stage_dims=(48, 96, 192, 384),
                               decoder=64, groups=4, sizes=(1, 2, 3, 6), classes=3, head_factor=4)
"""The capstone's configuration (the brief's lead decision); C1's band names may differ."""

SIDE = 256


@pytest.fixture
def offline_ee():
    case = ApiTestCase()
    case.setUp()
    try:
        yield ee
    finally:
        case.tearDown()


@pytest.fixture
def deep_graphs():
    """The whole model is one graph: its serialization recurses deeper than Python's default
    limit, so raise it, as ``tools/graph_identity.py`` does (20,000), for the test only."""
    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(limit, 20_000))
    try:
        yield
    finally:
        sys.setrecursionlimit(limit)


# ---------------------------------------------------------------------------
# Defaults: Swin-T224
# ---------------------------------------------------------------------------
def test_the_default_configuration_is_swin_t224():
    config = C.SwinUPerNetConfig()
    assert config == C.SWIN_T224
    assert (config.window, config.shift, config.patch, config.bands, config.embed_dim) == (7, 3, 4, ("VV", "VH"), 96)
    assert (config.depths, config.num_heads, config.stage_dims) == ((2, 2, 6, 2), (3, 6, 12, 24), (96, 192, 384, 768))
    assert (config.decoder, config.groups, config.sizes, config.classes, config.head_factor) == (256, 4, (1, 2, 3, 6), 2, 4)
    assert (config.strides, config.channels, config.norm_dim, config.fused, config.psp_concat) == (
        (4, 8, 16, 32), (96, 192, 384, 768), 768, 1024, 1792)
    assert config.layout == C.WINDOW_LAYOUT == WindowLayout(7) and config.window_multiple == 8
    # The module constants are Swin-T224's, as before C0.
    assert (C.WINDOW, C.SHIFT, C.DEPTHS, C.NUM_HEADS, C.STAGE_DIMS, C.STAGES, C.NORM_DIM) == (
        7, 3, (2, 2, 6, 2), (3, 6, 12, 24), (96, 192, 384, 768), 4, 768)
    assert (C.STRIDES, C.CHANNELS, C.DECODER, C.GROUPS, C.FUSED) == ((4, 8, 16, 32), (96, 192, 384, 768), 256, 4, 1024)
    assert C.LEVEL_CHANNELS == {1: 96, 2: 192, 3: 384, 4: 768}
    assert (D.C4_CHANNELS, D.CLASSES, D.HEAD_FACTOR) == (768, 2, 4)
    assert C.block_shifts(8, 8) == [[0, 3], [0, 3], [0, 3, 0, 3, 0, 3], [0, 0]]
    assert C.tensor_channels("fusion_concat") == 1024 and C.tensor_channels("P1") == 256
    assert C.SWIN_T224.model_grid(224, 224) == TokenGrid(224, 224, 2.5)
    assert C.SWIN_T224.input_windows(224, 224) == (8, 8)


def test_every_composite_defaults_to_swin_t224():
    import inspect

    functions = [E.stage_blocks, E.merge_constants, E.patch_embedding_constants, E.patch_embedding,
                 E.encoder_grids, D.psp_constants, D.fpn_constants, D.fusion_constants, D.pyramid_features,
                 D.head_constants, D.head_forward, D.head_logits, D.head_dense, C.stage_windows, C.block_shifts,
                 C.check_windows, C.tensor_channels, P.parameter_shapes, P.checkpoint_shapes, P.validate_state,
                 P.parameters_from_state, P.validate_parameters, P.parameter_records]
    for function in functions:
        assert inspect.signature(function).parameters["config"].default is C.SWIN_T224, function.__name__
    for cls in (E.EncoderConstants, E.EncoderGrids, D.PSPConstants, D.FPNConstants, D.FusionConstants):
        assert {f.name: f.default for f in dataclasses.fields(cls)}["config"] is C.SWIN_T224, cls.__name__


# ---------------------------------------------------------------------------
# Refusals: values no composite can serve
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("change,message", [
    ({"window": 1}, "window must be at least 2"),
    ({"window": 0}, "window must be a positive integer"),
    ({"window": 7.0}, "window must be a positive integer"),
    ({"patch": 0}, "patch must be a positive integer"),
    ({"bands": ()}, "non-empty"),
    ({"bands": "VV"}, "not one string"),
    ({"bands": ("VV", "VV")}, "distinct"),
    ({"bands": ("VV", "V.H")}, "regular expressions"),
    ({"embed_dim": 0}, "embed_dim must be a positive integer"),
    ({"depths": (2, 2, 2)}, "four feature levels"),
    ({"num_heads": (3, 6, 12, 24, 48)}, "four feature levels"),
    ({"depths": (2, 0, 2, 2)}, "positive integer"),
    ({"stage_dims": (64, 192, 384, 768)}, "stage_dims\\[0\\] must equal embed_dim"),
    ({"num_heads": (3, 6, 12, 25)}, "not a multiple of its 25 heads"),
    ({"groups": 8}, "groups must be 4"),
    ({"groups": 2}, "groups must be 4"),
    ({"sizes": (1, 2, 2, 6)}, "distinct"),
    ({"sizes": (1, 0, 3, 6)}, "positive integers"),
    ({"sizes": ()}, "positive integers"),
    ({"decoder": 0}, "decoder must be a positive integer"),
    ({"classes": 0}, "classes must be a positive integer"),
    ({"head_factor": 0}, "head_factor must be a positive integer"),
])
def test_unserveable_values_are_refused_with_the_reason(change, message):
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(C.SWIN_T224, **change)


def test_sequences_are_normalized_and_the_window_rule_follows_the_stages():
    config = dataclasses.replace(CAPSTONE, depths=[2, 2, 2, 2], bands=["VV", "VH", "VVminusVH"], sizes=[1, 2, 3, 6])
    assert config == CAPSTONE and isinstance(config.depths, tuple) and isinstance(config.bands, tuple)
    assert CAPSTONE.input_windows(SIDE, SIDE) == (8, 8)
    for side in (224, 255, 128):
        with pytest.raises(ValueError):
            CAPSTONE.input_windows(side, side)
    with pytest.raises(ValueError, match="multiples of 8"):
        CAPSTONE.check_windows(4, 4)
    with pytest.raises(ValueError, match="single axis"):
        CAPSTONE.check_windows(8, 16)             # stage 4: 1 x 2 windows


# ---------------------------------------------------------------------------
# Records: Swin-T224's 189 are unchanged; any configuration's follow its schema
# ---------------------------------------------------------------------------
BASE_RECORDS_SHA256 = "383cb5f79334aa0425e17cbfb91ef2fe57ea914b38dc64843a38a995d9a701c6"
"""The sorted (key, shape, bias, dtype, source, SHA-256) listing of Swin-T224's 189 records
from ``_seeded`` parameters, computed at the base revision ``ed88e3f`` (before C0)."""
BASE_SCHEMA_SHA256 = "1159361850b67231bd0a0ee93be4f9e328c0b9a062b7b7309021276dc3b57de6"
"""``parameter_shapes()`` and ``checkpoint_shapes()`` in order, with the records' order, at
``ed88e3f``."""


def _seeded(config, seed: int = 7):
    from tools.fixtures.core import dyadic

    enc, pe, dec, _ = P.parameter_shapes(config)
    return ({g: {n: dyadic(seed, f"{g}/{n}", s) for n, s in p.items()} for g, p in enc.items()},
            {n: dyadic(seed, f"pe/{n}", s) for n, s in pe.items()},
            {n: dyadic(seed, f"dec/{n}", s) for n, s in dec.items()})


def _listing(records) -> str:
    rows = sorted([r.key, list(r.shape), r.bias is not None, r.dtype, r.source, r.sha256] for r in records)
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def test_swin_t224_records_are_byte_identical_to_the_base_revision():
    from tensor_ee.swin import SwinTParameters
    from tensor_ee.swin_workflow import weight_records

    enc, pe, dec = _seeded(C.SWIN_T224)
    records = P.parameter_records(enc, pe, dec)
    assert len(records) == 189 and _listing(records) == BASE_RECORDS_SHA256
    workflow = weight_records(SwinTParameters(enc, pe, dec, "0" * 64))
    assert [(r.key, r.sha256) for r in records] == [(r.key, r.sha256) for r in workflow]
    shapes = P.parameter_shapes()
    schema = {"enc": [[g, [[n, list(s)] for n, s in p.items()]] for g, p in shapes[0].items()],
              "pe": [[n, list(s)] for n, s in shapes[1].items()],
              "dec": [[n, list(s)] for n, s in shapes[2].items()],
              "counters": [[n, list(s)] for n, s in shapes[3].items()],
              "checkpoint": [[n, list(s)] for n, s in P.checkpoint_shapes().items()],
              "record_order": [r.key for r in records]}
    assert hashlib.sha256(json.dumps(schema, separators=(",", ":")).encode()).hexdigest() == BASE_SCHEMA_SHA256


def test_swin_t224_records_match_the_trained_manifest():
    """An independent reference: the trained weights' v2 manifest (Milestone 12) names the
    same 189 records with the same shapes, bias flags, dtypes and sources."""
    manifest = json.loads((ROOT / "data/manifests/m12_trained_weights_manifest.json").read_text())
    assert manifest["format"] == "native_ee_featurecollection_weights_v2"
    records = P.parameter_records(*_seeded(C.SWIN_T224))
    assert {r.key: (list(r.shape), r.bias is not None, r.dtype, r.source) for r in records} == {
        k: (v["shape"], v["bias"], v["dtype"], v["source"]) for k, v in manifest["layers"].items()}


def test_capstone_records_follow_its_schema_and_load_through_the_manifest(offline_ee):
    enc, pe, dec = _seeded(CAPSTONE)
    records = P.parameter_records(enc, pe, dec, CAPSTONE)
    by_key = {r.key: r for r in records}
    assert len(records) == len(by_key) == 153                  # 189 less 4 blocks of 9 records
    assert by_key["patch_embed.proj"].shape == (48, 48) and by_key["patch_embed.proj"].bias is not None
    assert by_key["layers.0.blocks.1.attn.relative_position_bias_table"].shape == (225, 2)
    assert by_key["layers.3.downsample.reduction"].shape == (384, 768)
    assert by_key["decoder.psp.out_conv.0"].shape == (64, 640 * 9)
    assert by_key["decoder.fusion_block.0"].shape == (64, 256 * 9)
    assert by_key["segmentation_head.0"].shape == (3, 64) and by_key["segmentation_head.0"].bias.shape == (3,)
    count = sum(v.size for m in (*enc.values(), pe, dec) for v in m.values())
    # 5,124,588 encoder parameters (timm's count without its final norm and classifier) and
    # 751,203 in the decoder and head.
    assert sum(r.weights.size + (0 if r.bias is None else r.bias.size) for r in records) == count == 5_875_791
    # A stand-in v2 manifest (no assets read): the loader hands back every tensor's shape.
    listing = json.dumps(sorted((r.key, r.sha256) for r in records), separators=(",", ":"))
    manifest = {"format": "native_ee_featurecollection_weights_v2", "status": "ready",
                "source_checkpoint_sha256": "0" * 64, "manifest_hash": hashlib.sha256(listing.encode()).hexdigest(),
                "layers": {r.key: {"shape": list(r.shape), "dtype": r.dtype, "bias": r.bias is not None,
                                   "sha256": r.sha256, "assets": ["projects/c0-test/assets/w_part_0000"]}
                           for r in records}}
    s_enc, s_pe, s_dec, _ = P.with_manifest(enc, pe, dec, manifest, "0" * 64)
    P.validate_parameters(s_enc, s_pe, s_dec, CAPSTONE)
    with pytest.raises(UnsupportedOperation, match="groups differ"):
        P.validate_parameters(s_enc, s_pe, s_dec)                 # not Swin-T224's
    with pytest.raises(UnsupportedOperation, match="local NumPy"):
        P.parameter_records(s_enc, s_pe, s_dec, CAPSTONE)


@needs_references
def test_the_schema_is_timms_and_smps_own():
    """Every name and shape of the capstone's schema is the timm encoder's or SMP's decoder
    and head's: timm 1.0.30 SwinTransformer (its final norm and classifier excluded, as SMP's
    FeatureListNet drops them) and SMP 0.5.0 UPerNetDecoder and SegmentationHead."""
    from segmentation_models_pytorch.base import SegmentationHead
    from segmentation_models_pytorch.decoders.upernet.decoder import UPerNetDecoder
    from timm.models.swin_transformer import SwinTransformer

    for config, side in ((CAPSTONE, SIDE), (C.SWIN_T224, 224)):
        enc, pe, dec, counters = P.parameter_shapes(config)
        timm = SwinTransformer(img_size=side, patch_size=config.patch, in_chans=config.in_chans,
                               embed_dim=config.embed_dim, depths=config.depths, num_heads=config.num_heads,
                               window_size=config.window)
        expected = {k: tuple(v.shape) for k, v in timm.state_dict().items() if not k.startswith(("head.", "norm."))}
        ours = {P.encoder_prefix(g) + n: s for g, p in enc.items() for n, s in p.items()}
        ours.update({f"patch_embed.{n}": s for n, s in pe.items()})
        ours["patch_embed.proj.weight"] = (config.embed_dim, config.in_chans, config.patch, config.patch)
        assert ours == expected
        decoder = UPerNetDecoder((config.in_chans, 0) + config.stage_dims, decoder_channels=config.decoder)
        head = SegmentationHead(config.decoder, config.classes, kernel_size=1, upsampling=config.head_factor)
        smp = {f"decoder.{k}": tuple(v.shape) for k, v in decoder.state_dict().items()}
        smp.update({f"segmentation_head.{k}": tuple(v.shape) for k, v in head.state_dict().items()})
        assert {**dec, **counters} == smp


@needs_references
def test_the_capstone_shifts_are_timms():
    from timm.models.swin_transformer import SwinTransformer

    timm = SwinTransformer(img_size=SIDE, patch_size=4, in_chans=3, embed_dim=48, depths=(2, 2, 2, 2),
                           num_heads=(2, 4, 8, 16), window_size=8)
    assert all(b.window_size == (8, 8) for layer in timm.layers for b in layer.blocks)
    shifts = [[b.shift_size[0] for b in layer.blocks] for layer in timm.layers]
    assert all(b.shift_size[0] == b.shift_size[1] for layer in timm.layers for b in layer.blocks)
    assert shifts == CAPSTONE.block_shifts(*CAPSTONE.input_windows(SIDE, SIDE)) == [[0, 4], [0, 4], [0, 4], [0, 0]]


# ---------------------------------------------------------------------------
# The capstone's graph, offline
# ---------------------------------------------------------------------------
def _server(shape, value: int) -> ServerArray:
    stored = (shape[0], int(np.prod(shape[1:]))) if len(shape) == 4 else tuple(shape)
    array = ee.Array(ee.List.repeat(value, int(np.prod(shape)))).reshape(list(stored)).float()
    return ServerArray(array, stored)


def _stand_ins(config):
    """Stand-in weights: one server-side constant array per tensor (distinct values), stored
    as the weight assets store them (convolutions [Cout, Cin k k])."""
    enc, pe, dec, _ = P.parameter_shapes(config)
    count = iter(range(1, 100_000))
    return ({g: {n: _server(s, next(count)) for n, s in p.items()} for g, p in enc.items()},
            {n: _server(s, next(count)) for n, s in pe.items()},
            {n: _server(s, next(count)) for n, s in dec.items()})


def build_capstone(config=CAPSTONE, side=SIDE):
    """The whole forward as one graph: patch embedding, the four stages, the PSP, FPN,
    fusion and head, then the dense logits on the model grid."""
    enc, pe, dec = _stand_ins(config)
    ops = BlockImageOps(ee)
    grid = config.model_grid(side, side)
    x = ee.Image("projects/c0-test/assets/input").select(list(config.bands))
    embedded = E.patch_embedding(ops, x, grid, E.patch_embedding_constants(ops, pe, config=config), config=config)
    g = E.encoder_grids(*config.input_windows(side, side), config=config)
    constants = E.EncoderConstants([E.stage_blocks(ops, enc, s, config=config) for s in range(1, 5)],
                                   [E.merge_constants(ops, enc, n, config=config) for n in range(1, 4)],
                                   None, None, config)
    t = E.encoder(ops, embedded["x"], constants, g)
    feats = {k: (t[f"s{k}_out"], g.tokens[k - 1]) for k in range(1, 5)}
    fpn = D.fpn_constants(ee, dec, config=config)
    tensors, grids = D.fpn_forward(ee, feats, fpn)
    fusion = D.fusion_constants(ee, dec, config=config)
    fused, fused_grids = D.fusion_forward(ee, D.pyramid_features(tensors, grids, config=config), fusion)
    fused_tensor = WindowTensor(fused["double_relu"], fused_grids["double_relu"], config.decoder,
                                window=config.window)
    head_constants = D.head_constants(ee, dec, config=config)
    head, head_grids = D.head_forward(ee, fused_tensor, head_constants, config=config)
    logits = D.head_logits(ee, fused_tensor, head_constants, config=config)
    assert (logits.grid, logits.channels, logits.window) == (head_grids["logits"], config.classes, config.window)
    dense = D.head_dense(ee, logits, config=config)
    return {"grid": grid, "embedded": embedded, "grids": g, "constants": constants, "encoder": t, "fpn": fpn,
            "tensors": tensors, "tensor_grids": grids, "fusion": fusion, "fused": fused, "fused_grids": fused_grids,
            "head": head, "head_grids": head_grids, "dense": dense}


def test_the_capstone_graph_builds_with_the_configured_shapes_shifts_extent_and_output(offline_ee, deep_graphs):
    calls = {"shift": [], "unshift": [], "mask": [], "attention": [], "merge": [], "pool": [], "dense": []}

    def spy(name, module, function):
        real = getattr(module, function)

        def wrapper(*args, **kwargs):
            calls[name].append((args, kwargs))
            return real(*args, **kwargs)
        return patch.object(module, function, wrapper)

    with spy("shift", SW, "shift_windows"), spy("unshift", SW, "unshift_windows"), \
            spy("mask", SW, "shifted_window_mask"), spy("attention", SW, "window_attention"), \
            spy("merge", SW, "patch_merge"), spy("pool", UP, "window_pool"), spy("dense", UP, "to_dense"), \
            patch.object(ee.data, "computePixels", side_effect=AssertionError("evaluation")), \
            patch.object(ee.data, "getAsset", side_effect=AssertionError("asset read")):
        built = build_capstone()
    g, t = built["grids"], built["encoder"]

    # Token grids at every stage: 64, 32, 16, 8 tokens (10 to 80 m); 8, 4, 2, 1 windows of 8.
    assert built["grid"] == TokenGrid(256, 256, 2.5)
    assert [(tk.height, tk.width, tk.scale) for tk in g.tokens] == [(64, 64, 10.0), (32, 32, 20.0),
                                                                     (16, 16, 40.0), (8, 8, 80.0)]
    assert [(c.height, c.width) for c in g.coarse] == [(8, 8), (4, 4), (2, 2), (1, 1)]
    assert [g.shifted(s) for s in range(1, 5)] == [True, True, True, False]
    # Shapes: every block's width, heads and window; every merge; the patch embedding.
    blocks = built["constants"].blocks
    assert [[(b.dim, b.heads, b.window, b.qkv.rows) for b in stage] for stage in blocks] == [
        [(48, 2, 8, 64)] * 2, [(96, 4, 8, 64)] * 2, [(192, 8, 8, 64)] * 2, [(384, 16, 8, 64)] * 2]
    assert [(m.in_dim, m.reduction.out_features, m.rows) for m in built["constants"].merges] == [
        (48, 96, 64), (96, 192, 64), (192, 384, 64)]
    assert [kwargs["window"] for _, kwargs in calls["merge"]] == [8, 8, 8]
    # timm's shift at every block: SW-MSA exactly at timm's shifted blocks, shift 4 at window 8.
    sw = [[f"s{s}b{b}_shifted" in t for b in range(2)] for s in range(1, 5)]
    assert sw == [[False, True], [False, True], [False, True], [False, False]]
    assert [[4 if shifted else 0 for shifted in stage] for stage in sw] == CAPSTONE.block_shifts(8, 8)
    assert [a[4:] for a, _ in calls["shift"]] == [a[4:] for a, _ in calls["unshift"]] == [(4, 8)] * 3
    assert [(k["window"], k["shift"]) for _, k in calls["mask"]] == [(8, 4)] * 3
    assert [k["window"] for _, k in calls["attention"]] == [8] * 8
    # The PSP pools C4's 8 x 8 token extent (one window) to the bins (1, 2, 3, 6).
    (args, kwargs), = calls["pool"]
    assert (args[2], args[3], kwargs["sizes"], kwargs["window"]) == (g.tokens[3], 384, (1, 2, 3, 6), 8)
    plan = built["tensors"]["pooled"].plan
    assert (plan.height, plan.width, plan.sizes, plan.count) == (8, 8, (1, 2, 3, 6), 50)
    assert built["tensor_grids"]["psp_relu"] == g.tokens[3]
    # The FPN and fusion land on the stage grids, then C1's; the PSP and fusion widths.
    assert [built["tensor_grids"][f"F{k}"] for k in (3, 2, 1)] == [g.tokens[2], g.tokens[1], g.tokens[0]]
    assert set(built["fused_grids"].values()) == {g.tokens[0]}
    assert built["fpn"].psp.out_conv.cin == 640 and built["fpn"].psp.out_conv.cout == 64
    assert [w.cin for w in built["fusion"].groups] == [64] * 4
    # The output: 3 bands on the 256 x 256 model grid (x4, align_corners=True), window 8.
    assert built["head_grids"]["logits"] == TokenGrid(256, 256, 2.5)
    (args, kwargs), = calls["dense"]
    assert (args[1], args[3], kwargs["window"]) == (TokenGrid(256, 256, 2.5), 3, 8)
    serialized = built["dense"].serialize()
    assert '"c2"' in serialized and '"c3"' not in serialized


def test_the_capstone_graph_is_deterministic(offline_ee, deep_graphs):
    first, second = build_capstone()["dense"].serialize(), build_capstone()["dense"].serialize()
    assert hashlib.sha256(first.encode()).hexdigest() == hashlib.sha256(second.encode()).hexdigest()


# ---------------------------------------------------------------------------
# The composites refuse what belongs to another configuration
# ---------------------------------------------------------------------------
def test_the_encoder_refuses_constants_and_grids_of_another_configuration(offline_ee):
    enc, pe, dec = _stand_ins(CAPSTONE)
    t224 = _stand_ins(C.SWIN_T224)
    ops = WindowOps(ee)
    x = ee.Image("projects/c0-test/assets/x")
    g = E.encoder_grids(8, 8, config=CAPSTONE)
    blocks = [E.stage_blocks(ops, enc, s, config=CAPSTONE) for s in range(1, 5)]
    merges = [E.merge_constants(ops, enc, n, config=CAPSTONE) for n in range(1, 4)]
    with pytest.raises(ValueError, match="encoder grids were built for"):
        E.encoder(ops, x, E.EncoderConstants(blocks, merges, None, None, CAPSTONE), E.encoder_grids(8, 8))
    with pytest.raises(ValueError, match="encoder grids were built for"):
        E.encoder(ops, x, E.EncoderConstants(blocks, merges, None, None), g)
    window7 = [[dataclasses.replace(b, window=7) for b in blocks[0]]] + blocks[1:]
    with pytest.raises(ValueError, match="built for window 7"):
        E.encoder(ops, x, E.EncoderConstants(window7, merges, None, None, CAPSTONE), g)
    with pytest.raises(ValueError, match="s1b0"):
        E.stage_blocks(ops, t224[0], 1, config=CAPSTONE)               # Swin-T224's widths
    with pytest.raises(ValueError, match="pm1"):
        E.merge_constants(ops, t224[0], 1, config=CAPSTONE)
    with pytest.raises(ValueError, match="patch_embed"):
        E.patch_embedding_constants(ops, t224[1], config=CAPSTONE)
    with pytest.raises(ValueError, match="stage 1 has 2 blocks"):
        E.stage_blocks(ops, enc, 1, 3, config=CAPSTONE)
    with pytest.raises(ValueError, match="model grid"):
        E.patch_embedding(ops, x, TokenGrid(256, 256), E.patch_embedding_constants(ops, pe, config=CAPSTONE),
                          config=CAPSTONE)


def test_the_decoder_refuses_tensors_and_parameters_of_another_configuration(offline_ee):
    _, _, dec = _stand_ins(CAPSTONE)
    _, _, t224 = _stand_ins(C.SWIN_T224)
    tokens = TokenGrid(64, 64)
    for builder in (D.psp_constants, D.fpn_constants, D.head_constants):
        with pytest.raises(ValueError, match="the configuration needs"):
            builder(ee, t224, config=CAPSTONE)
    fusion = D.fusion_constants(ee, dec, config=CAPSTONE)
    window7 = {name: Feature(ee.Image(f"projects/c0-test/assets/{name}"), tokens.coarse(2 ** k), 64)
               for k, name in enumerate(C.PYRAMID)}
    with pytest.raises(GridMismatch, match="window-7 tensor in a decoder configured for window 8"):
        D.fusion_forward(ee, window7, fusion)
    parts = [WindowTensor(ee.Image(f"projects/c0-test/assets/p{g}"), tokens, 32, window=8) for g in range(4)]
    with pytest.raises(ValueError, match="fusion group 0: a part of 32 channels"):
        D.grouped_double(ee, parts, fusion)
    head = D.head_constants(ee, dec, config=CAPSTONE)
    fused = WindowTensor(ee.Image("projects/c0-test/assets/fused"), tokens, 64, window=8)
    with pytest.raises(GridMismatch):
        D.head_forward(ee, fused, head)                                # Swin-T224's window 7
    with pytest.raises(ValueError, match="needs 64 -> 2"):
        D.head_forward(ee, Feature(fused.image, tokens.coarse(1), 64), head)
    psp = D.psp_constants(ee, dec, config=CAPSTONE)
    fpn = D.fpn_constants(ee, dec, config=CAPSTONE)
    with pytest.raises(ValueError, match="PSP constants were built for"):
        D.fpn_forward(ee, {}, dataclasses.replace(fpn, psp=dataclasses.replace(psp, config=C.SWIN_T224)))
    with pytest.raises(RowMismatch):
        D.head_conv(ee, fused.image, head)                             # 64 rows against window 7's 49
    logits = WindowTensor(ee.Image("projects/c0-test/assets/logits"), TokenGrid(256, 256, 2.5), 3, window=8)
    with pytest.raises(GridMismatch):
        D.head_dense(ee, logits)                                       # Swin-T224's window 7
    with pytest.raises(ValueError, match="from the tensor"):
        D.head_dense(ee, logits, TokenGrid(64, 64), config=CAPSTONE)
    with pytest.raises(ValueError, match="needs the token grid"):
        D.head_dense(ee, logits.image, config=CAPSTONE)
