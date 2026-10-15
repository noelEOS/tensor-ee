"""X01a: the SegFormer composites live in ``tensor_ee.blocks.window.segformer``.

One implementation: the model modules re-export the library's objects. The composites' cost
functions sum their parts' METADATA cost functions; they are checked against the fixed
model's independent ``constant_costs()``, against the sizes of constants built with a NumPy
operand adapter, and against the assets an offline graph reads. The fuse/BatchNorm/ReLU/head
composite emits the expressions ``decoder()`` built inline before the move.

Offline only: graphs are built against ``ee.apitestcase.ApiTestCase`` signatures, small
widths only; nothing is evaluated and no model forward runs. The whole-model graph identity
(the seven SegFormer graphs and the fixture requests) is checked by the task's baseline
script and G1-ext, not here.
"""
from __future__ import annotations

import inspect
from pathlib import Path
import re
import sys

import ee
from ee.apitestcase import ApiTestCase
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference.segformer_b0 import blocks as M
from tensor_ee.blocks.window import segformer as S
from tensor_ee.blocks.window.linear import conv1x1, conv1x1_constants
from tensor_ee.blocks.window.activations import relu
from tensor_ee.blocks.window.normalization import BatchNormConstants, batch_norm
from tensor_ee.blocks.window.resample import window_resize
from tensor_ee.core.errors import UnsupportedOperation
from tensor_ee.core.grid import TokenGrid
from tensor_ee.core.metadata import COMPOSITE, NO_ARITHMETIC, Locality
from tensor_ee.core.ops import ValueOps, WindowOps
from tensor_ee.core.precision import DOUBLE_ROUND_ONCE
from tensor_ee.core.tensor import WindowTensor
from tensor_ee.models.segformer_b0 import configuration as CFG, decoder as D, encoder as E
from tensor_ee.weights.constants import ServerArray, WindowLayout

MOVED_ENCODER = ("LAYOUT", "NormConstants", "norm_constants", "channel_norm", "OverlapConstants",
                 "overlap_constants", "decimate", "overlap_embedding", "AttentionConstants",
                 "attention_constants", "spatial_reduction", "shared_key_attention", "global_attention",
                 "MixFFNConstants", "mix_ffn_constants", "mix_ffn",
                 "_parameter", "_linear_constants", "_conv_columns", "_check_feature")
RENAMED = {"BlockConstants": "MiTBlockConstants", "block_constants": "mit_block_constants", "block": "mit_block"}
"""The MiT block's model names -> its library names (one object each)."""
COMPOSITES = ("channel_norm", "decimate", "overlap_embedding", "spatial_reduction", "shared_key_attention",
              "global_attention", "mix_ffn", "mit_block", "project_pyramid", "fuse_head")


@pytest.fixture
def offline_ee():
    case = ApiTestCase()
    case.setUp()
    try:
        yield ee
    finally:
        case.tearDown()


def _w(shape, seed):
    """Integer-valued float32 weights (no platform-dependent RNG)."""
    n = int(np.prod(shape))
    return (((np.arange(n) * (2 * seed + 7)) % 23 - 11) / 8).astype(np.float32).reshape(shape)


def _numpy_parameters(shapes: dict) -> dict:
    return {name: _w(shape, seed) for seed, (name, shape) in enumerate(sorted(shapes.items()), 1)}


def _server_parameters(shapes: dict) -> dict:
    """ServerArray placeholders, convolution weights flattened as the asset format stores them."""
    def server(name, shape):
        stored = (shape[0], int(np.prod(shape[1:]))) if len(shape) == 4 else shape
        array = ee.Array(ee.FeatureCollection(f"projects/x01a-assets/assets/{name}").first().get("v"))
        return ServerArray(array, stored)
    return {name: server(name, shape) for name, shape in shapes.items()}


def _norm(prefix, dim):
    return {f"{prefix}.weight": (dim,), f"{prefix}.bias": (dim,)}


def _linear(prefix, cin, cout, kernel=None):
    weight = (cout, cin) if kernel is None else (cout, cin, kernel, kernel)
    return {f"{prefix}.weight": weight, f"{prefix}.bias": (cout,)}


def _overlap_shapes(cin, cout, kernel):
    return _linear("pe.proj", cin, cout, kernel) | _norm("pe.norm", cout)


def _attention_shapes(prefix, dim, ratio):
    shapes = _linear(f"{prefix}.q", dim, dim) | _linear(f"{prefix}.kv", dim, 2 * dim) | _linear(f"{prefix}.proj", dim, dim)
    if ratio > 1:
        shapes |= _linear(f"{prefix}.sr", dim, dim, ratio) | _norm(f"{prefix}.norm", dim)
    return shapes


def _mix_ffn_shapes(prefix, dim):
    return (_linear(f"{prefix}.fc1", dim, 4 * dim) | _linear(f"{prefix}.dwconv.dwconv", 1, 4 * dim, 3)
            | _linear(f"{prefix}.fc2", 4 * dim, dim))


def _block_shapes(dim, ratio):
    return (_norm("b.norm1", dim) | _attention_shapes("b.attn", dim, ratio) | _norm("b.norm2", dim)
            | _mix_ffn_shapes("b.mlp", dim))


def _built_values(constants, seen=None) -> int:
    """Values held by constants built with the NumPy operand adapter (each array once)."""
    seen = set() if seen is None else seen
    if id(constants) in seen:
        return 0
    seen.add(id(constants))
    if isinstance(constants, M.Operand):
        return constants.values.size
    if isinstance(constants, (list, tuple)):
        return sum(_built_values(part, seen) for part in constants)
    if hasattr(constants, "__dict__"):
        return sum(_built_values(part, seen) for part in vars(constants).values())
    return 0


def _assets(*values) -> set:
    return {name for value in values for name in re.findall(r"projects/x01a-assets/assets/([\w.]+)", value.serialize())}


# ---------------------------------------------------------------------------
# One implementation
# ---------------------------------------------------------------------------
def test_the_model_modules_re_export_the_library_objects():
    for name in MOVED_ENCODER:
        assert getattr(E, name) is getattr(S, name), name
    for model_name, library_name in RENAMED.items():
        assert getattr(E, model_name) is getattr(S, library_name), model_name
        assert not hasattr(S, model_name), model_name
    assert D.project_pyramid is S.project_pyramid and D.fuse_head is S.fuse_head
    assert D._linear_constants is S._linear_constants and D._parameter is S._parameter
    assert D._check_feature is S._check_feature
    assert CFG._server_shape is S._server_shape
    for name in (*MOVED_ENCODER[1:], *RENAMED.values(), "project_pyramid", "fuse_head", "_server_shape"):
        value = getattr(S, name)
        assert value.__module__ == "tensor_ee.blocks.window.segformer", name
    # The fixed configuration stays with the model.
    for module, names in ((E, ("EncoderConstants", "encoder_constants", "encoder")),
                          (D, ("DecoderConstants", "decoder_constants", "decoder", "ModelConstants",
                               "model_constants", "input_windows", "segformer_b0"))):
        for name in names:
            assert getattr(module, name).__module__ == module.__name__, name
            assert not hasattr(S, name), name
    assert not hasattr(S, "SEGFORMER_B0") and not hasattr(S, "check_parameters")


def _form(positional, keyword=()):
    """(name, kind, default) per parameter: ``positional`` names without defaults, then
    keyword-only (name, default) pairs."""
    empty = inspect.Parameter.empty
    return ([(name, inspect.Parameter.POSITIONAL_OR_KEYWORD, empty) for name in positional]
            + [(name, inspect.Parameter.KEYWORD_ONLY, default) for name, default in keyword])


def test_moved_signatures_are_unchanged():
    """The parameters the fixture harness and the tools read keep their names, kinds, order and
    defaults (the MiT block's under its library name)."""
    empty, layout = inspect.Parameter.empty, ("layout", S.LAYOUT)
    constants, block = ("ops", "parameters", "prefix"), ("ops", "x", "c")
    expected = {
        "norm_constants": _form((*constants, "channels"), (("eps", empty), layout)),
        "channel_norm": _form(block),
        "overlap_constants": _form((*constants, "cin", "cout"), (("kernel", empty), ("stride", empty), layout)),
        "decimate": _form(("ee", "x", "factor")),
        "overlap_embedding": _form(block),
        "attention_constants": _form((*constants, "dim", "heads", "ratio"), (layout,)),
        "spatial_reduction": _form(block),
        "shared_key_attention": _form(("ops", "q", "kv", "heads")),
        "global_attention": _form(block),
        "mix_ffn_constants": _form((*constants, "dim"), (layout,)),
        "mix_ffn": _form(block),
        "mit_block_constants": _form((*constants, "dim", "heads", "ratio")),
        "mit_block": _form(block),
        "project_pyramid": _form(("ops", "features", "projections")),
        "fuse_head": _form(("ops", "x", "fusion", "norm", "head"), (("channels", empty), ("classes", empty))),
    }
    actual = {name: [(p.name, p.kind, p.default) for p in inspect.signature(getattr(S, name)).parameters.values()]
              for name in expected}
    assert actual == expected
    assert S.LAYOUT == WindowLayout(7)


def test_metadata_covers_the_composites():
    assert tuple(S.METADATA) == COMPOSITES
    for name, entry in S.METADATA.items():
        assert entry.layout_in == entry.layout_out == "W"
        assert entry.policy == (NO_ARITHMETIC if name == "decimate" else COMPOSITE)
        assert "policy" not in inspect.signature(getattr(S, name)).parameters
        assert entry.status.startswith("Fixed — SegFormer-B0 (mit_b0) at 224 × 224: ")
        assert entry.evidence[:2] == ("docs/evidence/segformer_b0-fullmodel-live.json",
                                      "docs/evidence/segformer_b0-live.json")
    extent = {name for name, e in S.METADATA.items() if e.locality is Locality.EXTENT}
    assert extent == {"spatial_reduction", "shared_key_attention", "global_attention", "mit_block"}
    # The head's four-times align_corners=True resize depends on the map's size, so it is
    # chip-edge dependent although it reads only a ring of one window.
    assert {name for name, e in S.METADATA.items() if e.chip_edge_dependent} == {
        "shared_key_attention", "global_attention", "mit_block", "fuse_head"}
    assert (S.METADATA["overlap_embedding"].locality, S.METADATA["overlap_embedding"].radius) == (Locality.KERNEL, 3)
    for name in ("mix_ffn", "project_pyramid", "fuse_head"):
        assert (S.METADATA[name].locality, S.METADATA[name].radius) == (Locality.WINDOW_RING, 1), name


def test_the_head_resize_depends_on_the_map_size():
    """Why ``fuse_head`` is chip-edge dependent: its four-times ``align_corners=True`` weights
    change with the map's size, so a tile with halos cannot reproduce them; the pyramid's
    ``align_corners=False`` weights repeat the same few rows at any size."""
    from tensor_ee.blocks.window.resample import bilinear_weights

    def patterns(n, align_corners):
        weights = np.asarray(bilinear_weights(n, 4 * n, align_corners=align_corners))
        return {tuple(row[row != 0].tolist()) for row in weights}

    assert patterns(56, False) == patterns(28, False) and len(patterns(56, False)) == 5
    assert len(patterns(56, True)) == 223 and patterns(56, True) != patterns(28, True)


# ---------------------------------------------------------------------------
# Cost functions
# ---------------------------------------------------------------------------
def test_composite_costs_reproduce_the_fixed_model_constant_costs():
    """Summed over SegFormer-B0, the composites' costs equal ``constant_costs()``, an
    independent count over the parts' METADATA (and the 191 state arrays)."""
    costs = CFG.constant_costs()
    totals = {"values": 0, "asset_arrays": 0}
    for i, (dim, ratio) in enumerate(zip(CFG.STAGE_DIMS, CFG.SR_RATIOS)):
        cin, kernel = (2, 7) if i == 0 else (CFG.STAGE_DIMS[i - 1], 3)
        stage = {"values": (S.overlap_embedding_values(cin, dim, kernel=kernel)
                            + CFG.DEPTHS[i] * S.mit_block_values(dim, ratio) + S.channel_norm_values(dim)),
                 "asset_arrays": (S.overlap_embedding_arrays() + CFG.DEPTHS[i] * S.mit_block_arrays(ratio=ratio)
                                  + S.channel_norm_arrays())}
        assert stage == costs[f"encoder_stage{i + 1}"], i + 1
        for key in totals:
            totals[key] += stage[key]
    c1 = CFG.SEGFORMER_B0.stage_grids[0]
    decoder = {"values": (S.project_pyramid_values(CFG.STAGE_DIMS, 256, c1.height, c1.width)
                          + S.fuse_head_values(1024, 256, 2, c1.height, c1.width)),
               "asset_arrays": S.project_pyramid_arrays() + S.fuse_head_arrays()}
    assert decoder == {"values": costs["decoder_head"]["values"] + costs["resize"]["values"],
                       "asset_arrays": costs["decoder_head"]["asset_arrays"]}
    for key in totals:
        totals[key] += decoder[key]
    assert totals == costs["total"] == {"values": 6620078, "asset_arrays": 191}
    assert totals["asset_arrays"] == len(CFG.parameter_shapes()) - 1          # all but the BN counter


@pytest.mark.parametrize("ratio", [1, 2, 4, 8])
def test_cost_functions_equal_the_built_constants(ratio):
    """Constants built with the NumPy operand adapter hold exactly the values the cost
    functions count (each per-channel constant repeated to 49 rows)."""
    ops, dim = ValueOps(M.NUMPY_EE), 8
    p = _numpy_parameters(_block_shapes(dim, ratio))
    attention = S.attention_constants(ops, p, "b.attn", dim, 2, ratio)
    assert _built_values(attention) == S.global_attention_values(dim, ratio)
    assert _built_values([attention.sr, attention.sr_norm]) == S.spatial_reduction_values(dim, ratio)
    assert _built_values(S.mit_block_constants(ops, p, "b", dim, 2, ratio)) == S.mit_block_values(dim, ratio)
    assert _built_values(S.mix_ffn_constants(ops, p, "b.mlp", dim)) == S.mix_ffn_values(dim)
    for eps in (1e-5, 1e-6):
        assert _built_values(S.norm_constants(ops, p, "b.norm1", dim, eps=eps)) == S.channel_norm_values(dim)
    for cin, cout, kernel, stride in ((2, 8, 7, 4), (8, 16, 3, 2)):
        q = _numpy_parameters(_overlap_shapes(cin, cout, kernel))
        built = S.overlap_constants(ops, q, "pe", cin, cout, kernel=kernel, stride=stride)
        assert _built_values(built) == S.overlap_embedding_values(cin, cout, kernel=kernel)


def test_decoder_costs_equal_the_built_constants_and_resize_tables():
    ops, widths, channels = ValueOps(M.NUMPY_EE), (2, 3, 4, 5), 8
    projections = [S._linear_constants(ops, _numpy_parameters(_linear("p", w, channels)), "p", w, channels, S.LAYOUT)
                   for w in widths]
    resize = S._PARTS["window_resize"].embedded_constants
    tables = sum(resize(56 // 2 ** k, 56 // 2 ** k, 2 ** k, align_corners=False, window=7) for k in (1, 2, 3))
    assert _built_values(projections) + tables == S.project_pyramid_values(widths, channels, 56, 56)
    fusion = conv1x1_constants(ops, _w((channels, 4 * channels, 1, 1), 1), None, S.LAYOUT)
    norm = BatchNormConstants(ops, _w((channels,), 2), _w((channels,), 3), _w((channels,), 4),
                              np.abs(_w((channels,), 5)) + 1, rows=S.LAYOUT)
    head = conv1x1_constants(ops, _w((2, channels, 1, 1), 6), _w((2,), 7), S.LAYOUT)
    table = resize(56, 56, 4, align_corners=True, window=7)
    assert _built_values([fusion, norm, head]) + table == S.fuse_head_values(4 * channels, channels, 2, 56, 56)


def test_asset_array_counts_match_the_graphs(offline_ee):
    """Built with ServerArray placeholders, each composite's graph reads exactly the arrays its
    ``*_arrays`` function counts."""
    ops = WindowOps(ee)
    image = ee.Image("projects/x01a-test/assets/windows")
    for cin, cout, kernel, stride, side in ((2, 8, 7, 4, 28), (8, 16, 3, 2, 14)):
        c = S.overlap_constants(ops, _server_parameters(_overlap_shapes(cin, cout, kernel)), "pe", cin, cout,
                                kernel=kernel, stride=stride)
        out = S.overlap_embedding(ops, WindowTensor(image, TokenGrid(side, side, 2.5), cin, 7), c)["out"]
        assert len(_assets(out.image)) == S.overlap_embedding_arrays() == 4
    for ratio in (1, 2):
        x = WindowTensor(image, TokenGrid(7 * ratio, 7 * ratio, 2.5), 8, 7)
        c = S.mit_block_constants(ops, _server_parameters(_block_shapes(8, ratio)), "b", 8, 2, ratio)
        # Each composite on the raw input, so that no upstream part's arrays are counted.
        assert len(_assets(S.mit_block(ops, x, c)["out"].image)) == S.mit_block_arrays(ratio=ratio)
        attention = S.global_attention(ops, x, c.attention)["out"].image
        assert len(_assets(attention)) == S.global_attention_arrays(ratio=ratio)
        assert len(_assets(S.mix_ffn(ops, x, c.mlp)["out"].image)) == S.mix_ffn_arrays() == 6
        reduced = S.spatial_reduction(ops, x, c.attention)["out"].image
        assert len(_assets(reduced)) == S.spatial_reduction_arrays(ratio=ratio) == (0 if ratio == 1 else 4)
        assert len(_assets(S.channel_norm(ops, x, c.norm1).image)) == S.channel_norm_arrays() == 2
    grids = CFG.SEGFORMER_B0.stage_grids
    features = [WindowTensor(ee.Image(f"projects/x01a-test/assets/c{i}"), grid, 2, 7) for i, grid in enumerate(grids)]
    projections = [S._linear_constants(ops, _server_parameters(_linear(f"p{i}", 2, 4)), f"p{i}", 2, 4, S.LAYOUT)
                   for i in range(4)]
    joined = S.project_pyramid(ops, features, projections)["out"]
    assert len(_assets(joined.image)) == S.project_pyramid_arrays() == 8
    shapes = {"f.weight": (4, 16, 1, 1), "n.weight": (4,), "n.bias": (4,), "n.mean": (4,), "n.var": (4,),
              "h.weight": (2, 4, 1, 1), "h.bias": (2,)}
    q = _server_parameters(shapes)
    fusion = conv1x1_constants(ops, q["f.weight"], None, S.LAYOUT)
    norm = BatchNormConstants(ops, q["n.weight"], q["n.bias"], q["n.mean"], q["n.var"], rows=S.LAYOUT)
    head = conv1x1_constants(ops, q["h.weight"], q["h.bias"], S.LAYOUT)
    out = S.fuse_head(ops, joined, fusion, norm, head, channels=4, classes=2)["out"]
    assert len(_assets(out.image)) == S.project_pyramid_arrays() + S.fuse_head_arrays() == 15
    for name in ("shared_key_attention", "decimate"):
        assert S.METADATA[name].embedded_constants() == S.METADATA[name].asset_arrays() == 0


# ---------------------------------------------------------------------------
# fuse_head is decoder()'s former inline sequence
# ---------------------------------------------------------------------------
def test_fuse_head_emits_the_expressions_decoder_built_inline(offline_ee):
    ops = WindowOps(ee)
    grid = CFG.SEGFORMER_B0.stage_grids[0]
    joined = WindowTensor(ee.Image("projects/x01a-test/assets/joined"), grid, 16, 7)
    fusion = conv1x1_constants(ops, _w((4, 16, 1, 1), 1), None, S.LAYOUT)
    norm = BatchNormConstants(ops, _w((4,), 2), _w((4,), 3), _w((4,), 4), np.abs(_w((4,), 5)) + 1,
                              rows=S.LAYOUT, eps=1e-5)
    head = conv1x1_constants(ops, _w((2, 4, 1, 1), 6), _w((2,), 7), S.LAYOUT)
    out = S.fuse_head(ops, joined, fusion, norm, head, channels=4, classes=2)
    # decoder()'s body before X01a (widths 256 and 2 there; 4 and 2 here).
    fused = joined.like(conv1x1(ops, joined.image, fusion, layout=joined, policy=DOUBLE_ROUND_ONCE), 4)
    normalized = fused.like(batch_norm(ops, fused.image, norm, layout=fused, policy=DOUBLE_ROUND_ONCE))
    activated = normalized.like(relu(ops, normalized.image))
    logits = activated.like(conv1x1(ops, activated.image, head, layout=activated, policy=DOUBLE_ROUND_ONCE), 2)
    image, fine = window_resize(ops.ee, logits.image, logits.grid, 2, 4, align_corners=True,
                                window=7, policy=DOUBLE_ROUND_ONCE)
    expected = {"fusion": fused, "batch_norm": normalized, "relu": activated, "head": logits,
                "out": WindowTensor(image, fine, 2, 7)}
    assert list(out) == list(expected)
    for name, tensor in expected.items():
        assert (out[name].grid, out[name].channels, out[name].window) == (tensor.grid, tensor.channels, tensor.window)
        assert out[name].image.serialize() == tensor.image.serialize(), name
    assert out["out"].grid == CFG.SEGFORMER_B0.input_grid


# ---------------------------------------------------------------------------
# Refusals (the existing ones keep their behaviour; the head's is new)
# ---------------------------------------------------------------------------
def test_unsupported_forms_are_refused(offline_ee):
    ops = ValueOps(M.NUMPY_EE)
    p = _numpy_parameters(_block_shapes(8, 2) | _overlap_shapes(2, 8, 7))
    for kernel, stride in ((7, 2), (3, 4), (5, 3), (7.0, 4), (True, 4)):
        with pytest.raises(UnsupportedOperation):
            S.overlap_constants(ops, p, "pe", 2, 8, kernel=kernel, stride=stride)
    for heads, ratio in ((3, 2), (0, 2), (2, 3), (2, 16), (2, True)):
        with pytest.raises(UnsupportedOperation):
            S.attention_constants(ops, p, "b.attn", 8, heads, ratio)
    with pytest.raises(UnsupportedOperation):
        S.norm_constants(ops, p, "b.norm1", 8, eps=1e-4)
    with pytest.raises(UnsupportedOperation):                    # wrong shape for the named state
        S.norm_constants(ops, p, "b.norm1", 4, eps=1e-6)
    c = S.attention_constants(ops, p, "b.attn", 8, 2, 2)
    image = ee.Image("projects/x01a-test/assets/windows")
    with pytest.raises(UnsupportedOperation):                    # reduced keys are two windows wide
        S.spatial_reduction(WindowOps(ee), WindowTensor(image, TokenGrid(28, 14, 2.5), 8, 7), c)
    with pytest.raises(UnsupportedOperation):                    # not W7
        S.channel_norm(WindowOps(ee), WindowTensor(image, TokenGrid(16, 16, 2.5), 8, 8),
                       S.norm_constants(ops, p, "b.norm1", 8, eps=1e-6))
    with pytest.raises(UnsupportedOperation):
        S.decimate(ee, WindowTensor(image, TokenGrid(42, 42, 2.5), 8, 7), 3)
    q = WindowTensor(image, TokenGrid(14, 14, 2.5), 8, 7)
    with pytest.raises(UnsupportedOperation):                    # heads must divide the width
        S.shared_key_attention(WindowOps(ee), q, WindowTensor(image, TokenGrid(7, 7, 5.0), 16, 7), 3)
    with pytest.raises(UnsupportedOperation):                    # keys and values must be one window
        S.shared_key_attention(WindowOps(ee), q, WindowTensor(image, TokenGrid(14, 14, 2.5), 16, 7), 2)
    w = WindowOps(ee)
    fusion = conv1x1_constants(w, _w((4, 16, 1, 1), 1), None, S.LAYOUT)
    head = conv1x1_constants(w, _w((2, 4, 1, 1), 6), _w((2,), 7), S.LAYOUT)
    joined = WindowTensor(image, CFG.SEGFORMER_B0.stage_grids[0], 16, 7)
    for channels, classes in ((8, 2), (4, 3)):
        with pytest.raises(UnsupportedOperation):
            S.fuse_head(w, joined, fusion, None, head, channels=channels, classes=classes)
    with pytest.raises(UnsupportedOperation):
        S.overlap_embedding_values(2, 8, kernel=5)
    with pytest.raises(UnsupportedOperation):
        S.project_pyramid_values((2, 3, 4), 8, 56, 56)
    with pytest.raises(UnsupportedOperation):
        S.project_pyramid(w, [joined] * 3, [fusion] * 3)


def test_the_model_keeps_its_fixed_configuration_refusals():
    """The fixed-configuration checks stay in the model wrappers."""
    with pytest.raises(UnsupportedOperation):
        E.encoder_constants(None, {}, config=object())
    with pytest.raises(UnsupportedOperation):
        D.decoder_constants(None, {}, config=object())
    with pytest.raises(UnsupportedOperation):
        D.decoder(None, [], D.DecoderConstants((), None, None, None))
