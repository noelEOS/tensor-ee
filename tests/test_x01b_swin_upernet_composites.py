"""X01b: the Swin and UPerNet composites live in ``tensor_ee.blocks.window.swin`` and ``upernet``,
and ``SwinUPerNetConfig`` with its rules in ``swin_upernet_config``.

One implementation: the Swin-T224 model modules re-export the library's objects, under their
former names where the library's name is Swin- or fusion-specific. Library names are unique
across the window blocks. The composites' cost functions sum their parts' METADATA cost
functions; they are checked against an independent count over the parameter schema (2-D and
4-D weights by size, 1-D vectors times their rows), the configured constants' own
``embedded_values``, the research budgets at Swin-T224, the capstone plan's recorded counts
(docs/evidence/c1-capstone-plan.md), and the assets an offline graph reads.

Offline only: graphs are built against ``ee.apitestcase.ApiTestCase`` signatures with
placeholder ``ServerArray``s; nothing is evaluated and no model forward runs. The whole-model
graph identity (G1, G1-ext, the fixture requests and the capstone graph) is checked by the
task's baseline script and the gates, not here.
"""
from __future__ import annotations

import inspect
import re

import ee
from ee.apitestcase import ApiTestCase
import numpy as np
import pytest

from tensor_ee import swin as ADAPTER
from tensor_ee.blocks.window import pooling as WO
from tensor_ee.blocks.window import resample as WR
from tensor_ee.blocks.window import swin as SW
from tensor_ee.blocks.window import swin_upernet_config as K
from tensor_ee.blocks.window import upernet as U
from tensor_ee.blocks.window.linear import WINDOW_LAYOUT
from tensor_ee.core.metadata import COMPOSITE, NO_ARITHMETIC, Locality
from tensor_ee.core.ops import BlockImageOps
from tensor_ee.core.precision import GROUPED_DOUBLE
from tensor_ee.core.tensor import WindowTensor
from tensor_ee.models.swin_t224 import configuration as C
from tensor_ee.models.swin_t224 import decoder as D
from tensor_ee.models.swin_t224 import encoder as E
from tensor_ee.weights.constants import ServerArray

CAPSTONE = K.SwinUPerNetConfig(window=8, patch=4, bands=("VV", "VH", "VVminusVH"), embed_dim=48,
                               depths=(2, 2, 2, 2), num_heads=(2, 4, 8, 16), stage_dims=(48, 96, 192, 384),
                               decoder=64, classes=3)
"""The C1 capstone (docs/tasks/C0-capstone.md): 256 x 256, window 8, decoder width 64, 3 classes."""

KEPT = {E: ("DIM", "HEADS", "EXTENSION_MODES", "swin_block", "shifted_swin_block", "swin_stage"),
        D: ("C4_CHANNELS", "PSP_NORM", "PSP_OUT", "psp_branch_prefix", "batch_norm_args", "PSPConstants",
            "psp_constants", "psp_layer_norm", "psp_branch", "psp_forward", "FPNConstants", "fpn_bn_constants",
            "fpn_constants", "fpn_layer_norm", "fpn_lateral", "fpn_upsample", "fpn_merge", "fpn_conv_block",
            "fpn_forward", "FUSION_ORDER", "FORMULATIONS", "FusionConstants", "fusion_constants",
            "pyramid_features", "fusion_resize_factor", "fusion_resize_to_c1", "fusion_resize_level",
            "fusion_concat", "fusion_bn_relu", "fusion_forward", "HEAD", "HEAD_WEIGHT", "HEAD_BIAS", "CLASSES",
            "HEAD_FACTOR", "head_constants", "head_conv", "head_upsample", "head_forward", "head_logits",
            "head_dense"),
        C: ("LEVELS", "PREFIX", "WEIGHT", "PYRAMID", "RESIZED", "DOUBLE", "STAGE_COUNT", "TOKEN_SCALE", "MLP_RATIO",
            "SwinUPerNetConfig", "SWIN_T224", "check_shapes", "stage_shifted", "bn_args", "level_tensors",
            "block_index", "lateral_prefix", "conv_prefix", "norm_prefix")}
"""Names the model re-exports unchanged: the model module -> its names in the library."""
RENAMED = {E: {"BlockConstants": "SwinBlockConstants", "block_constants": "swin_block_constants",
               "mlp_branch": "swin_mlp_branch", "stage_blocks": "swin_stage_blocks",
               "merge_constants": "swin_merge_constants", "patch_embedding_constants": "swin_patch_embedding_constants",
               "patch_embedding": "swin_patch_embedding", "EncoderConstants": "SwinEncoderConstants",
               "EncoderGrids": "SwinEncoderGrids", "encoder_grids": "swin_encoder_grids", "encoder": "swin_encoder"},
           D: {"grouped_double": "fusion_grouped_double"}}
"""The model's names -> the library's (one object each)."""
SWIN_COMPOSITES = ("swin_mlp_branch", "swin_block", "shifted_swin_block", "swin_stage", "swin_patch_embedding",
                   "swin_encoder")
UPERNET_COMPOSITES = ("psp_layer_norm", "psp_branch", "psp_forward", "fpn_layer_norm", "fpn_lateral", "fpn_upsample",
                      "fpn_merge", "fpn_conv_block", "fpn_forward", "fusion_resize_to_c1", "fusion_resize_level",
                      "fusion_concat", "fusion_bn_relu", "fusion_grouped_double", "fusion_forward", "head_conv",
                      "head_upsample", "head_forward", "head_logits", "head_dense")


@pytest.fixture
def offline_ee():
    case = ApiTestCase()
    case.setUp()
    try:
        yield ee
    finally:
        case.tearDown()


def _library(module):
    return SW if module is E else U if module is D else K


# ---------------------------------------------------------------------------
# One implementation, unique names
# ---------------------------------------------------------------------------
def test_the_model_modules_re_export_the_library_objects():
    for module, names in KEPT.items():
        library = _library(module)
        for name in names:
            assert getattr(module, name) is getattr(library, name), (module.__name__, name)
    for module, names in RENAMED.items():
        library = _library(module)
        for model_name, library_name in names.items():
            assert getattr(module, model_name) is getattr(library, library_name), model_name
            assert not hasattr(library, model_name), model_name
    for library in (SW, U, K):
        for name, value in vars(library).items():
            if (inspect.isfunction(value) or inspect.isclass(value)) and value.__module__.startswith("tensor_ee.models"):
                raise AssertionError(f"{library.__name__}.{name} comes from the models layer")
    defined = {name for library in (SW, U, K) for name, value in vars(library).items()
               if (inspect.isfunction(value) or inspect.isclass(value)) and value.__module__ == library.__name__}
    assert {"SwinUPerNetConfig", "swin_encoder", "fpn_forward", "fusion_grouped_double", "stage_shifted"} <= defined
    # The adapter's lookups resolve to the same objects.
    assert ADAPTER.E is E and ADAPTER.D is D and ADAPTER.E.encoder is SW.swin_encoder
    assert ADAPTER.D.fpn_forward is U.fpn_forward and ADAPTER.D.head_dense is U.head_dense


def test_swin_t224_keeps_its_module_constants_and_rule_wrappers():
    """The fixed Swin-T224 contract stays with the model: its constants and the rule functions
    that take a ``config`` keyword."""
    for name in ("stage_windows", "block_shifts", "check_windows", "tensor_channels"):
        assert getattr(C, name).__module__ == C.__name__, name
        assert inspect.signature(getattr(C, name)).parameters["config"].default is K.SWIN_T224
    expected = {"WINDOW": 7, "SHIFT": 3, "DEPTHS": (2, 2, 6, 2), "NUM_HEADS": (3, 6, 12, 24),
                "STAGE_DIMS": (96, 192, 384, 768), "STAGES": 4, "NORM_DIM": 768, "STRIDES": (4, 8, 16, 32),
                "CHANNELS": (96, 192, 384, 768), "DECODER": 256, "LEVEL_CHANNELS": {1: 96, 2: 192, 3: 384, 4: 768},
                "GROUPS": 4, "FUSED": 1024}
    assert {name: getattr(C, name) for name in expected} == expected
    assert C.WINDOW_LAYOUT == WINDOW_LAYOUT
    assert not any(hasattr(K, name) for name in (*expected, "WINDOW_LAYOUT", "stage_windows", "tensor_channels"))
    assert (SW.DIM, SW.HEADS, U.C4_CHANNELS, U.CLASSES, U.HEAD_FACTOR) == (96, 3, 768, 2, 4)


def test_public_names_are_unique_across_the_window_blocks():
    """The catalogue and tests/test_live_fixtures.py key blocks by name, so no two window-block
    modules define the same public function or class, and no block name repeats."""
    import importlib
    import pkgutil

    import tensor_ee.blocks.window as package
    modules = [importlib.import_module(f"{package.__name__}.{info.name}") for info in pkgutil.iter_modules(package.__path__)]
    owners = {}
    for module in modules:
        for name, value in vars(module).items():
            if (not name.startswith("_") and (inspect.isfunction(value) or inspect.isclass(value))
                    and value.__module__ == module.__name__):
                owners.setdefault(name, []).append(module.__name__)
    assert {name: where for name, where in owners.items() if len(where) > 1} == {}
    blocks = [name for module in modules for name in vars(module).get("METADATA", {})]
    assert len(blocks) == len(set(blocks))
    assert not {"encoder", "block", "block_constants", "BlockConstants", "grouped_double"} & set(owners)


def test_the_g1_ext_hooks_patch_where_the_composites_are_looked_up():
    """tools/graph_identity.py patches the adapter's PE, E and D, and psp_forward in the
    library's upernet module, where fpn_forward looks it up (X01b)."""
    from tools import graph_identity as GI
    hooks = {name: module for module, name, _ in GI._swin_hooks(ADAPTER)}
    assert hooks["psp_forward"] is U and U.fpn_forward.__globals__ is vars(U)
    assert hooks["encoder"] is E and hooks["fpn_forward"] is D and hooks["head_dense"] is D
    assert all(hasattr(module, name) for name, module in hooks.items())


# ---------------------------------------------------------------------------
# Signatures and metadata
# ---------------------------------------------------------------------------
def _form(positional, keyword=()):
    """(name, kind, default) per parameter: ``positional`` names (or (name, default) pairs), then
    keyword-only (name, default) pairs."""
    empty, plain = inspect.Parameter.empty, inspect.Parameter.POSITIONAL_OR_KEYWORD
    out = [(p, plain, empty) if isinstance(p, str) else (p[0], plain, p[1]) for p in positional]
    return out + [(name, inspect.Parameter.KEYWORD_ONLY, default) for name, default in keyword]


def test_moved_signatures_are_unchanged():
    """The parameters the adapter, the capstone tool and the tests pass keep their names, kinds,
    order and defaults (the renamed composites' under their library names)."""
    t224, config = K.SWIN_T224, ("config", K.SWIN_T224)
    expected = {
        SW: {"swin_block_constants": _form(("ops", "parameters"), (("layout", WINDOW_LAYOUT),)),
             "swin_mlp_branch": _form(("ops", "x1", "c")),
             "swin_block": _form(("ops", "x", "c")),
             "shifted_swin_block": _form(("ops", "x", "c", "coarse"), (("mask", None), ("shift", None))),
             "swin_stage": _form(("ops", "x", "blocks", "coarse"), (("shifted", True), ("shift", None))),
             "swin_stage_blocks": _form(("ops", "parameters", "s", ("count", None)), (config,)),
             "swin_merge_constants": _form(("ops", "parameters", "n"), (config,)),
             "swin_patch_embedding_constants": _form(("ops", "parameters"), (config,)),
             "swin_patch_embedding": _form(("ops", "x", "grid", "c"), (config,)),
             "swin_encoder_grids": _form(("rows", "cols"), (config,)),
             "swin_encoder": _form(("ops", "x", "c", "g"), (("start", 1), ("stop", 4),
                                                             ("extension_mode", "per_block")))},
        U: {"psp_constants": _form(("ee", "params"), (config,)),
            "psp_layer_norm": _form(("ee", "x", "c")),
            "psp_branch": _form(("ee", "pooled_s", "c", "i")),
            "psp_forward": _form(("ee", "x", "tokens", "c")),
            "fpn_bn_constants": _form(("img", "params", "prefix"), (("rows", WINDOW_LAYOUT),)),
            "fpn_constants": _form(("ee", "params"), (config,)),
            "fpn_layer_norm": _form(("ee", "x", "c", "level")),
            "fpn_lateral": _form(("ee", "n", "c", "level")),
            "fpn_upsample": _form(("ee", "state")),
            "fpn_merge": _form(("ee", "state", "lat")),
            "fpn_conv_block": _form(("ee", "f", "c", "level", "tokens")),
            "fpn_forward": _form(("ee", "feats", "c")),
            "fusion_constants": _form(("ee", "params", ("formulations", (GROUPED_DOUBLE,)), ("groups", None)),
                                      (config,)),
            "pyramid_features": _form(("t", "grids", ("channels", None)), (config,)),
            "fusion_resize_to_c1": _form(("ee", "pyramid")),
            "fusion_resize_level": _form(("ee", "f", "target")),
            "fusion_concat": _form(("resized",)),
            "fusion_bn_relu": _form(("ee", "x", "c", "prefix")),
            "fusion_grouped_double": _form(("ee", "parts", "c"), (("policy", GROUPED_DOUBLE),)),
            "fusion_forward": _form(("ee", "pyramid", "c", ("formulations", (GROUPED_DOUBLE,)))),
            "head_constants": _form(("ee", "params", ("rows", None)), (config,)),
            "head_conv": _form(("ee", "x", "c"), (("layout", WINDOW_LAYOUT),)),
            "head_upsample": _form(("ee", "head", ("factor", 4))),
            "head_forward": _form(("ee", "fused", "c"), (config,)),
            "head_logits": _form(("ee", "fused", "c"), (config,)),
            "head_dense": _form(("ee", "image", ("tokens", None), ("channels", None)), (config,))},
    }
    for module, forms in expected.items():
        actual = {name: [(p.name, p.kind, p.default) for p in inspect.signature(getattr(module, name)).parameters.values()]
                  for name in forms}
        assert actual == forms, module.__name__
    assert SW.SwinEncoderConstants.__dataclass_fields__["config"].default is t224
    assert SW.SwinBlockConstants.__dataclass_fields__["window"].default == 7


def test_metadata_covers_the_composites():
    assert tuple(SW.METADATA) == SWIN_COMPOSITES and tuple(U.METADATA) == UPERNET_COMPOSITES
    for module in (SW, U):
        for name, entry in module.METADATA.items():
            assert entry.status.startswith("Fixed — Swin-T224 and the capstone: "), name
            assert entry.evidence[:3] == ("docs/evidence/t06-rehearsal.json",
                                          "docs/history/upernet/notes/milestone12_findings.md",
                                          "docs/evidence/c1-capstone.json"), name
            assert entry.evidence[-1].startswith("parts: "), name
            parameters = inspect.signature(getattr(module, name)).parameters
            assert ("policy" in parameters) == (entry.policy not in (COMPOSITE, NO_ARITHMETIC)), name
    every = {**SW.METADATA, **U.METADATA}
    assert {n for n, e in every.items() if e.policy == NO_ARITHMETIC} == {"fusion_concat", "head_dense"}
    assert every["fusion_grouped_double"].policy == GROUPED_DOUBLE
    assert {n for n, e in every.items() if e.locality is Locality.EXTENT} == {
        "swin_stage", "swin_encoder", "psp_forward", "fpn_forward"}
    assert {n for n, e in every.items() if e.chip_edge_dependent} == {
        "shifted_swin_block", "swin_stage", "swin_encoder", "head_upsample", "head_forward", "head_logits"}
    assert (every["shifted_swin_block"].locality, every["shifted_swin_block"].radius) == (Locality.WINDOW_RING, 4)
    assert (every["swin_patch_embedding"].locality, every["swin_patch_embedding"].radius) == (Locality.KERNEL, 2)
    layouts = {n: (e.layout_in, e.layout_out) for n, e in every.items() if (e.layout_in, e.layout_out) != ("W", "W")}
    assert layouts == {"swin_patch_embedding": ("R", "W"), "psp_branch": ("V", "V"), "head_dense": ("W", "R")}


@pytest.mark.parametrize("window", [7, 8])
def test_the_sw_msa_ring_is_w_minus_half_w_tokens(window):
    """Why ``shifted_swin_block`` has radius 4: timm rolls by -s (s = w // 2), attends within the
    shifted windows and rolls back by +s, so output window k reads the tokens of the shifted
    windows that overlap it, w - s tokens before it and s after it (one axis; both alike)."""
    shift = window // 2
    reach = 0
    for k in range(1, 4):                               # an interior window
        for x in range(k * window, (k + 1) * window):
            j = (x - shift) // window                   # the shifted window holding output token x
            sources = range(j * window + shift, (j + 1) * window + shift)
            reach = max(reach, k * window - min(sources), max(sources) - (k + 1) * window + 1)
    assert reach == window - shift == SW.METADATA["shifted_swin_block"].radius


# ---------------------------------------------------------------------------
# Cost functions
# ---------------------------------------------------------------------------
def _schema_values(config, *, names=None) -> int:
    """An independent count over the parameter schema: every 2-D or 4-D weight by its size, every
    1-D vector (norm, bias, BatchNorm) times its rows: w^2, or s^2 for the PSP's pooled branches.
    ``names``: the decoder names to count (all decoder names when not given)."""
    enc, pe, dec, _ = config.parameter_shapes()
    rows = config.window ** 2

    def count(name, shape):
        if len(shape) > 1:
            return int(np.prod(shape))
        branch = re.match(r"decoder\.psp\.blocks\.(\d+)\.", name)
        return shape[0] * (config.sizes[int(branch.group(1))] ** 2 if branch else rows)
    if names == "encoder":
        return sum(count(n, s) for group in enc.values() for n, s in group.items())
    if names == "patch_embed":
        return sum(count(n, s) for n, s in pe.items())
    return sum(count(n, s) for n, s in dec.items() if names is None or n in names)


def _tables(config, h4, h1) -> dict:
    """The coefficient tables, from the parts' cost functions: pooling, the PSP's matrix resizes,
    the FPN's x2 resizes, the fusion's x2, x4, x8 resizes and the head's align_corners=True x4."""
    resize = WR.METADATA["window_resize"].embedded_constants
    w = config.window
    return {"pool": WO.METADATA["window_pool"].embedded_constants(h4, h4, sizes=config.sizes),
            "matrix": sum(WR.METADATA["window_matrix_resize"].embedded_constants((s, s), h4, h4) for s in config.sizes),
            "fpn": sum(resize(h4 * 2 ** (3 - k), h4 * 2 ** (3 - k), 2, window=w) for k in (3, 2, 1)),
            "fusion": sum(resize((8 // f) * h4, (8 // f) * h4, f, window=w) for f in (2, 4, 8)),
            "head": resize(h1, h1, config.head_factor, align_corners=True, window=w)}


@pytest.mark.parametrize("config, h4, h1", [(K.SWIN_T224, 7, 56), (CAPSTONE, 8, 64)])
def test_composite_costs_equal_an_independent_count_over_the_schema(config, h4, h1):
    tables = _tables(config, h4, h1)
    assert SW.swin_encoder_values(config=config, norm=False) == _schema_values(config, names="encoder")
    assert SW.swin_patch_embedding_values(config=config) == _schema_values(config, names="patch_embed")
    norm = SW.swin_encoder_values(config=config) - SW.swin_encoder_values(config=config, norm=False)
    assert norm == 2 * config.window ** 2 * config.norm_dim
    decoder = (U.fpn_forward_values(h4, h4, config=config) + U.fusion_forward_values(h4, h4, config=config)
               + U.head_forward_values(h1, h1, config=config))
    assert decoder == _schema_values(config) + sum(tables.values())
    # Per composite, against the same count of its own parameters.
    psp = [n for n in config.parameter_shapes()[2] if n.startswith(("decoder.psp.", "decoder.feature_norms.3."))]
    assert U.psp_forward_values(h4, h4, config=config) == (_schema_values(config, names=psp) + tables["pool"]
                                                           + tables["matrix"])
    head = ("segmentation_head.0.weight", "segmentation_head.0.bias")
    assert U.head_forward_values(h1, h1, config=config) == _schema_values(config, names=head) + tables["head"]
    fusion = [n for n in config.parameter_shapes()[2] if n.startswith(K.PREFIX)]
    assert U.fusion_forward_values(h4, h4, config=config) == _schema_values(config, names=fusion) + tables["fusion"]


@pytest.mark.parametrize("config, h4, h1", [(K.SWIN_T224, 7, 56), (CAPSTONE, 8, 64)])
def test_composite_arrays_equal_the_parameter_count(config, h4, h1):
    """Every parameter tensor is read once: 239 for Swin-T224, 187 for the capstone (the fusion's
    four groups slice one weight array)."""
    enc, pe, dec, _ = config.parameter_shapes()
    arrays = (SW.swin_patch_embedding_arrays() + SW.swin_encoder_arrays(config=config, norm=False)
              + U.fpn_forward_arrays(config=config) + U.fusion_forward_arrays() + U.head_forward_arrays())
    assert arrays == sum(map(len, enc.values())) + len(pe) + len(dec) == {7: 239, 8: 187}[config.window]
    assert SW.swin_block_arrays() == SW.shifted_swin_block_arrays() == len(enc["s1b0"]) == 13


def test_composite_costs_reproduce_the_capstone_plan():
    """docs/evidence/c1-capstone-plan.md, "Evaluated constants": patch embedding 11,520, encoder
    6,470,844, PSP 545,280, FPN 273,408, fusion 163,840, head 384, tables 850, 3,200 and 9,740;
    total 7,479,066."""
    tables = _tables(CAPSTONE, 8, 64)
    assert (tables["pool"], tables["matrix"], tables["fpn"] + tables["fusion"] + tables["head"]) == (850, 3_200, 9_740)
    assert SW.swin_patch_embedding_values(config=CAPSTONE) == 11_520
    assert SW.swin_encoder_values(config=CAPSTONE, norm=False) == 6_470_844
    psp = U.psp_forward_values(8, 8, config=CAPSTONE)
    assert psp == 545_280 + 850 + 3_200
    assert U.fpn_forward_values(8, 8, config=CAPSTONE) == psp + 273_408 + tables["fpn"]
    assert U.fusion_forward_values(8, 8, config=CAPSTONE) == 163_840 + tables["fusion"]
    assert U.head_forward_values(64, 64, config=CAPSTONE) == 384 + tables["head"]
    total = (SW.swin_patch_embedding_values(config=CAPSTONE) + SW.swin_encoder_values(config=CAPSTONE, norm=False)
             + U.fpn_forward_values(8, 8, config=CAPSTONE) + U.fusion_forward_values(8, 8, config=CAPSTONE)
             + U.head_forward_values(64, 64, config=CAPSTONE))
    assert total == 7_479_066


def test_decoder_costs_equal_the_research_budgets_at_swin_t224():
    """The research budget helpers' counts at C4 7 x 7 and C1 56 x 56 (upernet/earth_engine
    ``budget``, recorded in tests/test_window_blocks.py's BUDGETS)."""
    psp = {"c4_layernorm2d": 75264, "pool_tables": 750, "branch_conv_1x1": 786432, "branch_batchnorm": 51200,
           "branch_resize_tables": 2450, "out_conv_3x3": 4128768, "out_batchnorm": 50176}
    fpn = {"c1_c3_layernorm2d": 65856, "fpn_lateral_conv_1x1": 172032, "fpn_lateral_batchnorm": 150528,
           "fpn_resize_tables": 1288, "fpn_conv_3x3": 1769472, "fpn_conv_batchnorm": 150528}
    fusion = {"fusion_resize_tables": 2316, "fusion_conv_3x3": 2359296, "fusion_batchnorm": 50176}
    head = {"head_conv_1x1": 512, "head_bias_rows": 98, "head_upsample_tables": 4096}
    assert U.psp_forward_values(7, 7) == sum(psp.values())
    assert U.fpn_forward_values(7, 7) == sum(psp.values()) + sum(fpn.values())
    assert U.fusion_forward_values(7, 7) == sum(fusion.values())
    assert U.head_forward_values(56, 56) == sum(head.values())
    assert U.psp_layer_norm_values() == psp["c4_layernorm2d"]
    assert sum(U.psp_branch_values(i) for i in range(4)) == psp["branch_conv_1x1"] + psp["branch_batchnorm"]
    assert sum(U.fpn_layer_norm_values(k) for k in (3, 2, 1)) == fpn["c1_c3_layernorm2d"]
    assert U.head_conv_values() == head["head_conv_1x1"] + head["head_bias_rows"]
    assert U.fusion_grouped_double_values() == fusion["fusion_conv_3x3"] + fusion["fusion_batchnorm"]


def _server_parameters(config, prefix: str):
    """ServerArray placeholders of the configuration's encoder and decoder, one asset each,
    convolutions flattened as the asset format stores them."""
    enc, pe, dec, _ = config.parameter_shapes()

    def server(name, shape):
        stored = (shape[0], int(np.prod(shape[1:]))) if len(shape) == 4 else tuple(shape)
        array = ee.Array(ee.FeatureCollection(f"projects/{prefix}/assets/{name}").first().get("v"))
        return ServerArray(array, stored)
    return ({g: {k: server(f"{g}.{k}", s) for k, s in p.items()} for g, p in enc.items()},
            {k: server(f"patch_embed.{k}", s) for k, s in pe.items()},
            {k: server(k, s) for k, s in dec.items()})


@pytest.mark.parametrize("config, h4", [(K.SWIN_T224, 7), (CAPSTONE, 8)])
def test_decoder_costs_equal_the_configured_constants(offline_ee, config, h4):
    """Without the tables, the decoder's costs are the configured constants' own
    ``embedded_values`` (built from placeholders: nothing is embedded or evaluated)."""
    _, _, dec = _server_parameters(config, "x01b-constants")
    tables = _tables(config, h4, 8 * h4)
    fpn, fusion = U.fpn_constants(ee, dec, config=config), U.fusion_constants(ee, dec, config=config)
    assert U.psp_forward_values(h4, h4, config=config) - tables["pool"] - tables["matrix"] == fpn.psp.embedded_values
    assert (U.fpn_forward_values(h4, h4, config=config) - tables["pool"] - tables["matrix"] - tables["fpn"]
            == fpn.embedded_values)
    assert U.fusion_forward_values(h4, h4, config=config) - tables["fusion"] == fusion.embedded_values


def _assets(*values) -> set:
    return {name for value in values for name in re.findall(r"projects/x01b-graph/assets/([\w.]+)", value.serialize())}


def test_asset_array_counts_match_the_graphs(offline_ee):
    """Composites built from placeholders read exactly the counted server arrays (Swin-T224)."""
    enc, pe, dec = _server_parameters(K.SWIN_T224, "x01b-graph")
    ops = BlockImageOps(ee)
    grids = SW.swin_encoder_grids(8, 8)
    s4 = SW.swin_stage_blocks(ops, enc, 4)
    x4 = ee.Image("projects/x01b-input/assets/c4")
    assert len(_assets(SW.swin_mlp_branch(ops, x4, s4[0])["out"])) == SW.swin_mlp_branch_arrays()
    assert len(_assets(SW.swin_block(ops, x4, s4[0])["out"])) == SW.swin_block_arrays()
    x2 = ee.Image("projects/x01b-input/assets/c2")
    shifted = SW.shifted_swin_block(ops, x2, SW.swin_stage_blocks(ops, enc, 2)[1], grids.coarse[1])["out"]
    assert len(_assets(shifted)) == SW.shifted_swin_block_arrays()
    assert len(_assets(SW.swin_stage(ops, x4, s4, grids.coarse[3], shifted=False)["out"])) == SW.swin_stage_arrays(
        depth=2)
    c = SW.SwinEncoderConstants([[]] * 3 + [s4], [None] * 2 + [SW.swin_merge_constants(ops, enc, 3)], None, None)
    x3 = ee.Image("projects/x01b-input/assets/c3")
    assert len(_assets(SW.swin_encoder(ops, x3, c, grids, start=4, stop=4)["s4_out"])) == SW.swin_encoder_arrays(
        start=4, norm=False)
    image = ee.Image("projects/x01b-input/assets/image").rename(["VV", "VH"])
    grid = K.SWIN_T224.model_grid(224, 224)
    embedded = SW.swin_patch_embedding(ops, image, grid, SW.swin_patch_embedding_constants(ops, pe))["x"]
    assert len(_assets(embedded)) == SW.swin_patch_embedding_arrays()

    tokens = grids.tokens                                   # C1..C4: 56, 28, 14, 7 tokens
    feats = {k: (ee.Image(f"projects/x01b-input/assets/c{k}"), tokens[k - 1]) for k in range(1, 5)}
    t, g = U.fpn_forward(ee, feats, U.fpn_constants(ee, dec))
    assert len(_assets(t["psp_relu"])) == U.psp_forward_arrays() == 27
    assert len(_assets(t["branch_2_relu"])) == U.psp_layer_norm_arrays() + U.psp_branch_arrays()
    assert len(_assets(t["P1"], t["P2"], t["P3"], t["psp_relu"])) == U.fpn_forward_arrays() == 63
    pyramid = {name: WindowTensor(ee.Image(f"projects/x01b-input/assets/{name}"), tokens[k], 256)
               for k, name in enumerate(K.PYRAMID)}
    fused, fg = U.fusion_forward(ee, pyramid, U.fusion_constants(ee, dec))
    assert len(_assets(fused["double_relu"])) == U.fusion_forward_arrays() == 5
    fused_tensor = WindowTensor(ee.Image("projects/x01b-input/assets/fused"), fg["double_relu"], 256)
    head, hg = U.head_forward(ee, fused_tensor, U.head_constants(ee, dec))
    assert len(_assets(head["logits"])) == U.head_forward_arrays() == 2
    assert len(_assets(U.head_logits(ee, fused_tensor, U.head_constants(ee, dec)).image)) == U.head_forward_arrays()
    assert len(_assets(U.head_dense(ee, head["logits"], hg["logits"]))) == 2
