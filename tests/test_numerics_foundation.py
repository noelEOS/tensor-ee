"""L02: the ops interface, precision policies, metadata schema and coefficient
transport live in the library; old tensor_ee names re-export them (the research names
were archived with the research code, L06b).

Offline only: graphs are built against ``ee.apitestcase.ApiTestCase`` signatures
and compared by their serialization. No Earth Engine initialization or request.
"""
from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import ee
from ee.apitestcase import ApiTestCase
import numpy as np
import pytest

from recorded_paths import current
import tensor_ee
from tensor_ee import compiler
from tensor_ee.core import errors, metadata as M, ops as O, precision as P
from tensor_ee.core.grid import TokenGrid
from tensor_ee.core.tensor import Feature, WindowTensor
from tensor_ee.layouts import window as LW
from tensor_ee.weights import assets as A, constants as K

ROOT = Path(__file__).resolve().parents[1]

# (old module, old name, path the name must resolve at, name there, defining module).
# Every old name is a re-export of the one library object.
MOVED = [
    # Precision helpers (core/precision.py)
    ("tensor_ee.compiler", "_ee_scalar", "tensor_ee.core.precision", "exact_scalar", "tensor_ee.core.precision"),
    # Ops (core/ops.py), hierarchy kept
    ("tensor_ee.core.ops", "ValueOps", "tensor_ee.core.ops", "BlockArrayOps", "tensor_ee.core.ops"),
    ("tensor_ee.core.ops", "WindowOps", "tensor_ee.core.ops", "BlockImageOps", "tensor_ee.core.ops"),
    ("tensor_ee.swin", "BlockImageOps", "tensor_ee.core.ops", "WindowOps", "tensor_ee.core.ops"),
    # Coefficient transport (weights/constants.py, the public namespace). ServerArray
    # is the V carrier (core/tensor.py); constant_value is defined in core/precision.py,
    # below the ops whose constant() uses it.
    ("tensor_ee.swin", "ServerArray", "tensor_ee.weights.constants", "ServerArray", "tensor_ee.core.tensor"),
    ("tensor_ee._swin_parameters", "ServerArray", "tensor_ee.weights.constants", "ServerArray",
     "tensor_ee.core.tensor"),
    ("tensor_ee.core.precision", "ServerArray", "tensor_ee.core.tensor", "ServerArray", "tensor_ee.core.tensor"),
    ("tensor_ee.weights.constants", "encode_float32", "tensor_ee.core.precision", "encode_float32",
     "tensor_ee.core.precision"),
    ("tensor_ee.compiler", "_inline_float32_array", "tensor_ee.weights.constants", "inline_float32_array",
     "tensor_ee.weights.constants"),
    # Asset format (weights/assets.py)
    ("tensor_ee.weight_assets", "ConvolutionWeights", "tensor_ee.weights.assets", "ConvolutionWeights",
     "tensor_ee.weights.assets"),
    ("tensor_ee.weight_assets", "convolution_weights", "tensor_ee.weights.assets", "convolution_weights",
     "tensor_ee.weights.assets"),
    ("tensor_ee.weight_assets", "feature_records", "tensor_ee.weights.assets", "feature_records",
     "tensor_ee.weights.assets"),
    ("tensor_ee.weight_assets", "FeatureCollectionWeightProvider", "tensor_ee.weights.assets",
     "FeatureCollectionWeightProvider", "tensor_ee.weights.assets"),
    ("tensor_ee", "FeatureCollectionWeightProvider", "tensor_ee.weights.assets",
     "FeatureCollectionWeightProvider", "tensor_ee.weights.assets"),
    ("tensor_ee._swin_parameters", "FeatureCollectionWeightProvider", "tensor_ee.weights.assets",
     "FeatureCollectionWeightProvider", "tensor_ee.weights.assets"),
    ("tensor_ee._swin_parameters", "ConvolutionWeights", "tensor_ee.weights.assets", "ConvolutionWeights",
     "tensor_ee.weights.assets"),
    ("tensor_ee.weight_mapping", "FeatureCollectionWeightProvider", "tensor_ee.weights.assets",
     "FeatureCollectionWeightProvider", "tensor_ee.weights.assets"),
]

LIBRARY_MODULES = ("tensor_ee.core.ops", "tensor_ee.core.precision", "tensor_ee.core.metadata",
                   "tensor_ee.weights", "tensor_ee.weights.constants", "tensor_ee.weights.assets",
                   "tensor_ee.blocks")


def _attr(module: str, name: str):
    return getattr(importlib.import_module(module), name)


@pytest.fixture
def offline_ee():
    case = ApiTestCase()
    case.setUp()
    try:
        yield ee
    finally:
        case.tearDown()


def _same(a, b) -> bool:
    return a.serialize() == b.serialize()


# ---------------------------------------------------------------------------
# One implementation: every old name is the library object
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("old_module, old_name, new_module, new_name, defined_in", MOVED,
                         ids=[f"{m}.{n}" for m, n, *_ in MOVED])
def test_old_name_is_the_library_object(old_module, old_name, new_module, new_name, defined_in):
    old, new = _attr(old_module, old_name), _attr(new_module, new_name)
    assert old is new
    # Defined once, in the library: there is no second implementation.
    assert new.__module__ == defined_in


def test_every_map_entry_owned_by_l02_is_at_its_target():
    module_map = json.loads((ROOT / "docs/evidence/l00-module-map.json").read_text())
    entries = [e for e in module_map["entries"] if e["owner"] == "L02"]
    assert len(entries) == 12
    for entry in entries:
        path, name = entry["target"].split("::")
        path = current(path)                         # the record predates the rename
        if entry["id"] == "swin_block/earth_engine/ops.py:encode_float32":
            # Lead change (L01 decision): core, not weights, so layouts may use it.
            assert path == "tensor_ee/weights/constants.py"
            path = "tensor_ee/core/precision.py"
        target = _attr(path[:-3].replace("/", "."), name)
        old_path, old_name = entry["id"].split(":")
        old_path = current(old_path)
        if old_path.startswith("tensor_ee/"):          # research old paths are archived (L06b)
            assert _attr(old_path[:-3].replace("/", "."), old_name) is target, entry["id"]


def test_feature_collection_weight_provider_is_one_class_patchable_from_swin_parameters():
    # tests/test_swin_adapter.py patches arrays on the class reached through
    # tensor_ee._swin_parameters; with_manifest instantiates the same class.
    from tensor_ee import _swin_parameters as SP
    assert SP.FeatureCollectionWeightProvider is A.FeatureCollectionWeightProvider
    assert "FeatureCollectionWeightProvider" in SP.with_manifest.__code__.co_names
    assert SP.with_manifest.__globals__ is vars(SP)
    with patch.object(SP.FeatureCollectionWeightProvider, "arrays", side_effect=AssertionError("patched")) as mocked:
        assert A.FeatureCollectionWeightProvider.arrays is mocked
        assert tensor_ee.FeatureCollectionWeightProvider.arrays is mocked


def test_compiler_functions_look_up_their_patched_bindings():
    # test_pointwise_operators.py monkeypatches activations._ee_scalar and
    # test_inline_coefficients.py patches conv._inline_float32_array (retargeted
    # by L03); the moved functions resolve the names in their modules' globals.
    from tensor_ee.blocks.raster import activations, conv
    assert compiler._ee_scalar is P.exact_scalar is activations._ee_scalar
    assert compiler._inline_float32_array is K.inline_float32_array is conv._inline_float32_array
    for helper in (compiler._leaky_relu, compiler._common_score_mask):
        assert helper.__globals__ is vars(activations)
        assert "_ee_scalar" in helper.__code__.co_names
    assert compiler.conv2d.__globals__ is vars(conv)
    assert "_inline_float32_array" in compiler.conv2d.__code__.co_names


def test_patching_compiler_inline_array_still_intercepts_conv2d(offline_ee):
    import torch
    from torch import nn
    model = nn.Conv2d(1, 1, 1).eval()
    with torch.no_grad():
        model.weight.fill_(0.5)
        model.bias.fill_(0.25)
    grid = tensor_ee.Grid("EPSG:3857", (10, 0, 100, 0, -10, 200), 1, 1)
    source = ee.Image.constant(0).rename("a")
    with patch("tensor_ee.blocks.raster.conv._inline_float32_array", side_effect=RuntimeError("intercepted")):
        with pytest.raises(RuntimeError, match="intercepted"):
            tensor_ee.compile_model(model, source, ("a",), grid)


def test_public_api_names_resolve_to_the_same_objects():
    assert len(tensor_ee.__all__) == 31
    for name in tensor_ee.__all__:
        assert getattr(tensor_ee, name) is not None
    from tensor_ee import swin
    assert swin.__all__ == ["ServerArray", "SwinTParameters", "load_swin_t224_parameters",
                            "load_swin_t224_model", "SwinT224Adapter"]
    for name in (*swin.__all__, "_SKIP_BANDS", "_C4_BANDS"):
        assert getattr(swin, name) is not None
    assert swin.ServerArray is K.ServerArray is P.ServerArray


# ---------------------------------------------------------------------------
# Ops: hierarchy unchanged; ValueOps/WindowOps are the full implementations
# ---------------------------------------------------------------------------
# Methods each class defined itself at the base revision (1f7a364).
M1_METHODS = {"__init__", "matmul", "transpose", "row_max", "row_sum", "repeat_cols", "exp", "maximum",
              "divide_by_sqrt", "scale", "slice_rows", "cat_rows", "slice_cols", "cat_cols"}
BLOCK_METHODS = {"constant", "wrap", "to_double", "to_float", "row_total", "matmul_double", "repeat_rows"}
LAYOUT_METHODS = {"reshape", "slice", "cat", "transpose"}   # added in L02
# The union in contract section 5, plus the in-window layout methods.
UNION = {"matmul", "matmul_double", "transpose", "row_max", "row_sum", "row_total", "repeat_cols",
         "repeat_rows", "exp", "maximum", "divide_by_sqrt", "scale", "slice_rows", "slice_cols",
         "cat_rows", "cat_cols", "constant", "wrap", "to_double", "to_float",
         "reshape", "slice", "cat"}


def _own(cls) -> set:
    return {name for name, value in vars(cls).items() if inspect.isfunction(value)}


def test_ops_class_hierarchy_is_unchanged():
    assert O.BlockArrayOps.__mro__ == (O.BlockArrayOps, O.ArrayOps, object)
    assert O.BlockImageOps.__mro__ == (O.BlockImageOps, O.ImageOps, object)
    assert O.ValueOps is O.BlockArrayOps and O.WindowOps is O.BlockImageOps
    assert (O.ValueOps.__name__, O.WindowOps.__name__) == ("BlockArrayOps", "BlockImageOps")
    # The parents moved verbatim; the subclasses gained only the layout methods.
    assert _own(O.ArrayOps) == M1_METHODS and _own(O.ImageOps) == M1_METHODS
    assert _own(O.BlockArrayOps) == BLOCK_METHODS | LAYOUT_METHODS
    assert _own(O.BlockImageOps) == BLOCK_METHODS | LAYOUT_METHODS


def test_value_and_window_ops_are_the_full_implementations():
    for cls in (O.ValueOps, O.WindowOps):
        assert all(callable(getattr(cls, name, None)) for name in UNION), cls
    # The Milestone 1 classes keep exactly their own surface: duck-typing on
    # which methods an ops object has gives the same answers as before.
    for cls in (O.ArrayOps, O.ImageOps):
        assert {n for n in UNION if hasattr(cls, n)} == M1_METHODS - {"__init__"}
    protocol = {n for n, v in vars(O.Ops).items() if inspect.isfunction(v) and not n.startswith("__")}
    assert protocol == UNION


# ---------------------------------------------------------------------------
# The new in-window layout methods emit today's expressions
# ---------------------------------------------------------------------------
def test_each_in_window_expression_has_one_implementation():
    # The W reshape is defined once in core; layouts.window.reshape (still defined
    # in layouts.window) and WindowOps.reshape both call it.
    assert "image_reshape" in LW.reshape.__code__.co_names and LW.reshape.__module__ == "tensor_ee.layouts.window"
    assert "image_reshape" in O.WindowOps.reshape.__code__.co_names
    # slice/cat and slice_rows/cols/cat_rows/cols share one helper per path.
    uses = {"_value_slice": (O.ArrayOps.slice_rows, O.ArrayOps.slice_cols, O.ValueOps.slice),
            "_value_cat": (O.ArrayOps.cat_rows, O.ArrayOps.cat_cols, O.ValueOps.cat),
            "_image_slice": (O.ImageOps.slice_rows, O.ImageOps.slice_cols, O.WindowOps.slice),
            "_image_cat": (O.ImageOps.cat_rows, O.ImageOps.cat_cols, O.WindowOps.cat)}
    for helper, methods in uses.items():
        for method in methods:
            assert helper in method.__code__.co_names, (helper, method)
    # Only the helpers emit these EE calls in core/ops.py.
    tree = ast.parse((ROOT / "tensor_ee/core/ops.py").read_text())
    emitters = {}
    for function in ast.walk(tree):
        if isinstance(function, ast.FunctionDef):
            for node in ast.walk(function):
                if isinstance(node, ast.Attribute) and node.attr in ("arrayReshape", "arraySlice", "arrayCat", "cat"):
                    emitters.setdefault(node.attr, set()).add(function.name)
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "slice" and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "a"):
                    emitters.setdefault("slice", set()).add(function.name)
    assert emitters == {"arrayReshape": {"image_reshape"}, "arraySlice": {"_image_slice"},
                        "arrayCat": {"_image_cat"}, "cat": {"_value_cat"}, "slice": {"_value_slice"}}


def _window_image(ee, rows=3, cols=2):
    return ee.Image.constant(1).toArray().arrayRepeat(0, rows).arrayRepeat(1, cols)


def test_window_layout_methods_emit_todays_expressions(offline_ee):
    ops = O.WindowOps(ee)
    a, b, c = (_window_image(ee).add(i) for i in range(3))
    # reshape is layouts.window.reshape (= upernet.earth_engine.layout.reshape).
    for shape in ([1, 6], [3, 2, 1], [6]):
        assert _same(ops.reshape(a, shape), LW.reshape(ee, a, shape))
        assert _same(ops.reshape(a, shape), a.arrayReshape(ee.Image(ee.Array(shape)), len(shape)))
    assert _same(ops.slice(a, 1, 0, 1), a.arraySlice(1, 0, 1))
    assert _same(ops.slice(a, 0, 1, 3), O.ImageOps(ee).slice_rows(a, 1, 3))
    start = ee.Image.constant(1).toInt()
    assert _same(ops.slice(a, 0, start, start.add(1)), a.arraySlice(0, start, start.add(1)))
    assert _same(ops.cat([a, b, c], 1), a.arrayCat(b, 1).arrayCat(c, 1))
    assert _same(ops.cat([a, b, c], 0), O.ImageOps(ee).cat_rows([a, b, c]))
    assert ops.cat([a], 1) is a
    # Without axes: ImageOps' matrix transpose, a no-argument arrayTranspose().
    assert _same(ops.transpose(a), a.arrayTranspose())
    assert _same(ops.transpose(a), O.ImageOps(ee).transpose(a))
    assert not _same(ops.transpose(a), a.arrayTranspose(0, 1))   # defaults are not passed
    # With axes: resize.py's L.reshape(...).arrayTranspose(0, 1).
    x = _window_image(ee, 63, 4)
    expected = LW.reshape(ee, x, [7, 9, 4]).arrayTranspose(0, 1)
    assert _same(ops.transpose(ops.reshape(x, [7, 9, 4]), 0, 1), expected)
    with pytest.raises(TypeError):
        ops.transpose(a, 0)


def test_value_layout_methods_emit_todays_expressions(offline_ee):
    ops = O.ValueOps(ee)
    a = ee.Array([[1, 2, 3], [4, 5, 6]]).float()
    b = a.multiply(2)
    assert _same(ops.reshape(a, [3, 2]), a.reshape([3, 2]))
    assert _same(ops.slice(a, 1, 0, 2), a.slice(1, 0, 2))
    assert _same(ops.slice(a, 0, 0, 1), O.ArrayOps(ee).slice_rows(a, 0, 1))
    assert _same(ops.cat([a, b], 0), ee.Array.cat([a, b], 0))
    assert _same(ops.cat([a, b], 1), O.ArrayOps(ee).cat_cols([a, b]))
    assert ops.cat([a], 0) is a
    assert _same(ops.transpose(a), a.transpose())
    assert _same(ops.transpose(a), O.ArrayOps(ee).transpose(a))
    assert not _same(ops.transpose(a), a.transpose(0, 1))   # defaults are not passed
    # conv2d.py's server-side weight permutation, written with the ops.
    cout, cin, k = 4, 2, 3
    w = ee.Array(ee.List.repeat(0, cout * cin * k * k)).reshape([cout, cin * k * k]).float()
    expected = (w.reshape([cout, cin, k * k]).transpose(1, 2).reshape([cout, k * k * cin]).transpose())
    actual = ops.transpose(ops.reshape(ops.transpose(ops.reshape(w, [cout, cin, k * k]), 1, 2),
                                       [cout, k * k * cin]))
    assert _same(actual, expected)


def test_existing_ops_methods_are_unchanged(offline_ee):
    # A spot check that the moved classes still emit their established casts.
    v, w = O.ValueOps(ee), O.WindowOps(ee)
    a = ee.Array([[1.5, 2.0]]).float()
    image = _window_image(ee)
    assert _same(v.matmul(a, a.transpose()), a.matrixMultiply(a.transpose()).float())
    assert _same(v.matmul_double(a, a.transpose()), a.matrixMultiply(a.transpose()))
    assert _same(w.row_sum(image), image.arrayReduce(ee.Reducer.sum(), [1]).toFloat())
    assert _same(w.row_total(image), image.arrayReduce(ee.Reducer.sum(), [1]))
    assert _same(w.constant(np.ones(3, np.float32), repeat_rows=49),
                 ee.Image(P.constant_value(ee, np.ones(3, np.float32), 49)))


# ---------------------------------------------------------------------------
# Layout-derived row repetition
# ---------------------------------------------------------------------------
def test_window_tensor_rows_is_a_read_only_property():
    grid = TokenGrid(14, 14)
    tensor = WindowTensor("image", grid, 8)
    assert tensor.rows == 49 and WindowTensor("image", grid, 8, window=1).rows == 1
    assert Feature("image", grid, 8).rows == 49 and Feature("image", grid, 8, "dense_row").rows == 1
    assert isinstance(vars(WindowTensor)["rows"], property)
    # Not a field: repr, fields and equality are unchanged.
    assert [f.name for f in dataclasses.fields(WindowTensor)] == ["image", "grid", "channels", "window"]
    assert [f.name for f in dataclasses.fields(Feature)] == ["image", "grid", "channels", "window", "layout"]
    assert "rows" not in repr(tensor) and tensor == WindowTensor("image", grid, 8)
    with pytest.raises(AttributeError):
        tensor.rows = 1
    assert K.constant_rows(tensor) == tensor.rows


def test_rows_helper_gives_49_1_and_the_value_row_count():
    grid = TokenGrid(14, 14)
    assert K.constant_rows(WindowTensor("image", grid, 8)) == 49
    assert K.constant_rows(WindowTensor("image", grid, 8, window=1)) == 1
    assert K.constant_rows(Feature("image", grid, 8)) == 49
    assert K.constant_rows(Feature("image", grid, 8, "dense_row")) == 1
    assert K.constant_rows(K.WindowLayout()) == 49 and K.WindowLayout().window == 7
    assert K.WindowLayout(1).rows == 1
    assert K.constant_rows(K.ValueLayout(36)) == 36
    assert K.constant_rows(P.ServerArray("array", (4, 8))) == 4
    assert K.constant_rows(np.zeros((9, 3), np.float32)) == 9
    assert K.layout_of(WindowTensor("image", grid, 8)) == K.WindowLayout(7)
    # The count comes from a layout, never from a literal or a vector.
    for bad in (49, None, "W", P.ServerArray("array", (8,)), np.zeros(8)):
        with pytest.raises((TypeError, ValueError)):
            K.constant_rows(bad)
    for bad in (0, -1, True, 2.0):
        with pytest.raises(ValueError):
            K.WindowLayout(bad)
        with pytest.raises(ValueError):
            K.ValueLayout(bad)


def test_row_constants_serialize_as_todays_repeat_rows(offline_ee):
    vector = np.asarray([0.5, -1.25, 3.0], np.float32)
    server = P.ServerArray(ee.Array([0.5, -1.25, 3.0]).float(), (3,))
    for values in (vector, server):
        window = K.row_constant(O.WindowOps(ee), values, K.WindowLayout())
        assert window.rows == 49
        assert _same(window.value, O.WindowOps(ee).constant(values, repeat_rows=49))
        assert _same(window.value, ee.Image(P.constant_value(ee, values, repeat_rows=49)))
        dense = K.row_constant(O.WindowOps(ee), values, Feature("image", TokenGrid(14, 14), 3, "dense_row"))
        assert dense.rows == 1 and _same(dense.value, O.WindowOps(ee).constant(values, repeat_rows=1))
        value = K.row_constant(O.ValueOps(ee), values, K.ValueLayout(4))
        assert value.rows == 4 and _same(value.value, P.constant_value(ee, values, repeat_rows=4))


def test_constants_built_for_one_layout_are_refused_by_another(offline_ee):
    grid = TokenGrid(14, 14)
    vector = np.ones(3, np.float32)
    built = K.row_constant(O.WindowOps(ee), vector, K.WindowLayout(7))
    assert built.check(WindowTensor("image", grid, 3)) is built.value
    K.check_rows(built, Feature("image", grid, 3))
    for other in (K.WindowLayout(1), Feature("image", grid, 3, "dense_row"), K.ValueLayout(4)):
        with pytest.raises(K.RowMismatch):
            built.check(other)
    assert issubclass(K.RowMismatch, errors.GridMismatch) and issubclass(K.RowMismatch, ValueError)
    # Builders that already record their rows (BatchNormConstants) are checked the same way.
    from tensor_ee.blocks.window.normalization import BatchNormConstants
    bn = BatchNormConstants(O.WindowOps(ee), vector, vector, vector, vector, rows=K.constant_rows(K.WindowLayout()))
    K.check_rows(bn, WindowTensor("image", grid, 3))
    with pytest.raises(K.RowMismatch):
        K.check_rows(bn, WindowTensor("image", grid, 3, window=1))
    with pytest.raises(K.RowMismatch):
        K.check_rows(object(), K.WindowLayout())


def test_inline_float32_array_keeps_its_expression(offline_ee):
    values = [[0.25, -2.0], [float(np.float32(2**63)), 1.0]]
    graph = K.inline_float32_array(values).serialize()
    assert graph.count("Number.parse") == 1
    assert _same(K.inline_float32_array([[0.25, -2.0]]), ee.Array([[0.25, -2.0]]))


# ---------------------------------------------------------------------------
# Precision policies
# ---------------------------------------------------------------------------
def test_policy_names_validate():
    assert P.POLICIES == ("float32_operands", "round_each_op", "double_round_once")
    assert P.VARIANTS == {"grouped_double": "double_round_once"}
    assert (P.FLOAT32_OPERANDS, P.ROUND_EACH_OP, P.DOUBLE_ROUND_ONCE, P.GROUPED_DOUBLE) == (
        "float32_operands", "round_each_op", "double_round_once", "grouped_double")
    for name in (*P.POLICIES, *P.VARIANTS):
        assert P.validate_policy(name) == name
    for bad in ("float32", "double", "", None, 1, ("round_each_op",)):
        with pytest.raises(ValueError):
            P.validate_policy(bad)
    assert P.validate_policy("double_round_once", allowed=("double_round_once",)) == "double_round_once"
    assert P.validate_policy("grouped_double", allowed=("double_round_once", "grouped_double")) == "grouped_double"
    with pytest.raises(errors.UnsupportedOperation, match="not implemented"):
        P.validate_policy("round_each_op", allowed="double_round_once")
    with pytest.raises(ValueError, match="unknown"):
        P.validate_policy("round_each_op", allowed=("round_each_op", "bogus"))


# ---------------------------------------------------------------------------
# Metadata schema
# ---------------------------------------------------------------------------
def test_metadata_schema_matches_the_contract():
    assert {item.name: item.value for item in M.Locality} == {
        "POINTWISE": "pointwise", "KERNEL": "kxk", "WINDOW": "window",
        "WINDOW_RING": "window+ring", "EXTENT": "extent"}
    fields = dataclasses.fields(M.BlockMetadata)
    # X02b appended validated and per_setting (contract "As built (Phase 3)").
    assert [f.name for f in fields] == ["name", "layout_in", "layout_out", "locality", "policy",
                                        "embedded_constants", "asset_arrays", "radius",
                                        "chip_edge_dependent", "status", "evidence", "validated", "per_setting"]
    defaults = {f.name: f.default for f in fields if f.default is not dataclasses.MISSING}
    assert defaults == {"radius": None, "chip_edge_dependent": False, "status": "Fixed", "evidence": (),
                        "validated": None, "per_setting": None}
    entry = M.BlockMetadata("relu", "R", "R", M.Locality.POINTWISE, "float32_operands", lambda: 0, lambda: 0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.name = "other"


def _entry(**changes):
    base = dict(name="layer_norm", layout_in="W", layout_out="W", locality=M.Locality.POINTWISE,
                policy="double_round_once", embedded_constants=lambda channels: 2 * 49 * channels,
                asset_arrays=lambda: 2, validated=M.ANY_SETTINGS)
    base.update(changes)
    return M.BlockMetadata(**base)


def test_metadata_accepts_valid_entries():
    valid = {
        "layer_norm": _entry(),
        "conv3x3_window": _entry(name="conv3x3_window", locality=M.Locality.WINDOW_RING, radius=1,
                                 status="Fixed — window 7", evidence=("M10",)),
        "conv2d": _entry(name="conv2d", layout_in="R", layout_out="R", locality=M.Locality.KERNEL,
                         radius=1, policy="float32_operands", status="Ready"),
        "fusion": _entry(name="fusion", policy="grouped_double", locality=M.Locality.WINDOW_RING, radius=1),
        "psp": _entry(name="psp", layout_out="V", locality=M.Locality.EXTENT, chip_edge_dependent=True),
        # Lead rules: no-arithmetic blocks are "none", composites "composite",
        # and W add is double_round_once.
        "relu": _entry(name="relu", policy="none", embedded_constants=lambda: 0, asset_arrays=lambda: 0),
        "concat": _entry(name="concat", policy=M.NO_ARITHMETIC),
        "window_attention": _entry(name="window_attention", policy=M.COMPOSITE, locality=M.Locality.WINDOW,
                                   evidence=("linear: double_round_once", "attention core: round_each_op")),
        "add": _entry(name="add", policy="double_round_once"),
        # X02b: a joint record, a per-setting property, and an offline-only block with no record.
        "window_resize": _entry(name="window_resize", locality=M.Locality.WINDOW_RING, radius=1,
                                chip_edge_dependent=True,
                                validated=M.Validated(({"window": (7, 8), "align_corners": (False,)},
                                                       {"window": (7,), "align_corners": (True,)})),
                                per_setting=lambda *, align_corners=False: {"chip_edge_dependent": align_corners}),
        "new_block": _entry(name="new_block", status="Fixed — offline only: window 7", validated=M.NOT_LIVE),
    }
    assert M.validate_metadata(valid) is valid
    assert M.MARKERS == ("none", "composite")


def test_metadata_markers_are_not_block_policies():
    # validate_policy checks a block's policy= argument and never accepts the markers.
    for marker in M.MARKERS:
        with pytest.raises(ValueError, match="unknown"):
            P.validate_policy(marker)


@pytest.mark.parametrize("key, entry", [
    ("other", _entry()),                                            # key != name
    ("layer_norm", _entry(layout_in="D")),                          # unknown layout
    ("layer_norm", _entry(layout_out="raster")),
    ("layer_norm", _entry(locality="pointwise")),                   # not a Locality
    ("layer_norm", _entry(policy="float64")),                       # unknown policy
    ("layer_norm", _entry(locality=M.Locality.KERNEL)),             # KERNEL needs a radius
    ("layer_norm", _entry(locality=M.Locality.WINDOW_RING, radius=0)),
    ("layer_norm", _entry(radius=1)),                               # POINTWISE takes no radius
    ("layer_norm", _entry(embedded_constants=96)),                  # costs are callables
    ("layer_norm", _entry(chip_edge_dependent="no")),
    ("layer_norm", _entry(status="Validated")),
    ("layer_norm", _entry(evidence=["M2"])),
    ("layer_norm", "not metadata"),
    ("layer_norm", _entry(validated=None)),                         # X02b: the record is required
    ("layer_norm", _entry(validated={"eps": (1e-5,)})),             # a Validated, not a mapping
    ("layer_norm", _entry(validated=M.NOT_LIVE)),                   # Ready with nothing live
    ("layer_norm", _entry(status="Fixed — offline only: eps 1e-5")),   # offline only, yet a record
    ("layer_norm", _entry(per_setting=4)),                          # per_setting is callable
    ("layer_norm", _entry(per_setting=lambda eps: {"radius": 1})),  # its settings have defaults
])
def test_metadata_rejects_invalid_entries(key, entry):
    with pytest.raises((TypeError, ValueError)):
        M.validate_metadata({key: entry})


def test_metadata_rejects_an_empty_or_non_mapping():
    for bad in ({}, [("layer_norm", _entry())], None):
        with pytest.raises(TypeError):
            M.validate_metadata(bad)


# ---------------------------------------------------------------------------
# Layer rule: core and weights import only what their layer may, no exceptions
# ---------------------------------------------------------------------------
THIRD_PARTY = ("ee", "numpy")
LAYER_ALLOWED = {"tensor_ee/core": ("tensor_ee.core",),
                 "tensor_ee/weights": ("tensor_ee.core", "tensor_ee.layouts")}
FORBIDDEN = ("ee_attention", "swin_", "upernet", "torch", "tensor_ee.blocks", "tensor_ee.converter",
             "tensor_ee.compiler", "tensor_ee.semantics", "tensor_ee.tiling", "tensor_ee.receptive_field",
             "tensor_ee.onnx_import", "tensor_ee.compatibility", "tensor_ee.weight_assets",
             "tensor_ee.weight_mapping", "tensor_ee.primitives", "tensor_ee.outputs", "tensor_ee.swin",
             "tensor_ee._swin_parameters", "tensor_ee.models", "tensor_ee.segwater")


def _type_checking_nodes(tree) -> set:
    """ids of nodes inside ``if TYPE_CHECKING:`` blocks (never executed at run time)."""
    inside = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and (
                (isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING")
                or (isinstance(node.test, ast.Attribute) and node.test.attr == "TYPE_CHECKING")):
            for statement in node.body:
                inside.update(id(child) for child in ast.walk(statement))
    return inside


def _imports(path: Path, *, type_checking: bool = False):
    """Every runtime import in the file (module level and inside functions),
    resolved; with ``type_checking=True``, only those under ``if TYPE_CHECKING:``."""
    package = ".".join(path.relative_to(ROOT).with_suffix("").parts[:-1])
    tree = ast.parse(path.read_text())
    guarded = _type_checking_nodes(tree)
    for node in ast.walk(tree):
        if (id(node) in guarded) != type_checking:
            continue
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parent = package.split(".")[: len(package.split(".")) - node.level + 1]
                base = ".".join(parent + ([base] if base else []))
            yield base


def _within(name: str, prefixes) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)


@pytest.mark.parametrize("layer", sorted(LAYER_ALLOWED))
def test_core_and_weights_respect_the_layer_rule_with_no_exceptions(layer):
    files = sorted((ROOT / layer).glob("*.py"))
    expected = {"tensor_ee/core": {"__init__.py", "grid.py", "tensor.py", "errors.py", "ops.py",
                                   "precision.py", "metadata.py"},
                "tensor_ee/weights": {"__init__.py", "constants.py", "assets.py"}}[layer]
    assert {p.name for p in files} >= expected
    violations = []
    for path in files:
        for name in _imports(path):
            top = name.split(".")[0]
            allowed = (top in sys.stdlib_module_names or top == "__future__" or _within(name, THIRD_PARTY)
                       or _within(name, LAYER_ALLOWED[layer]))
            if name.startswith(FORBIDDEN) or not allowed:
                violations.append((str(path.relative_to(ROOT)), name))
    assert violations == []
    # Annotation-only imports: exactly torch.nn in weights/assets.py, never at run time.
    guarded = {(str(path.relative_to(ROOT)), name) for path in files for name in _imports(path, type_checking=True)}
    assert guarded == ({("tensor_ee/weights/assets.py", "torch")} if layer == "tensor_ee/weights" else set())


def test_asset_annotations_resolve_without_a_runtime_torch_import():
    import typing
    import torch
    assert not hasattr(A, "nn") and A.TYPE_CHECKING is False
    hints = typing.get_type_hints(A.convolution_weights, localns={"nn": torch.nn})
    assert hints["module"] is torch.nn.Conv2d
    hints = typing.get_type_hints(A.FeatureCollectionWeightProvider.__call__, localns={"nn": torch.nn})
    assert hints["module"] is torch.nn.Conv2d


def test_layouts_import_the_float32_encoder_from_core():
    # The one L01 layer exception is gone: embedded_raster imports encode_float32
    # from tensor_ee.core.precision, and no layouts module imports a research folder.
    for path in sorted((ROOT / "tensor_ee/layouts").glob("*.py")):
        names = list(_imports(path))
        assert not [n for n in names if _within(n, ("ee_attention", "swin_block", "swin_pair", "swin_hierarchy",
                                                    "swin_encoder", "swin_image_encoder", "upernet"))], path
    assert "tensor_ee.core.precision" in set(_imports(ROOT / "tensor_ee/layouts/raster.py"))
    assert "\nKNOWN_EXCEPTIONS = set()\n" in (ROOT / "tests/test_core_layouts.py").read_text()


def test_blocks_package_is_empty():
    # Created by L02 so that L03 (blocks/raster/) and L04 (blocks/window/) never
    # both create it: no imports and no definitions (a docstring at most).
    body = ast.parse((ROOT / "tensor_ee/blocks/__init__.py").read_text()).body
    assert all(isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) for node in body)
    assert importlib.import_module("tensor_ee.blocks").__path__


def test_importing_the_library_loads_no_research_module():
    code = ("import sys; import " + ", ".join(LIBRARY_MODULES) + "\n"
            "bad = sorted(m for m in sys.modules if m.split('.')[0] in "
            "{'ee_attention', 'upernet'} or m.startswith('swin_'))\n"
            "assert not bad, bad")
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True,
                   env={**os.environ, "PYTHONPATH": str(ROOT)})
