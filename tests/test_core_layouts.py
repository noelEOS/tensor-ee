"""L01: core types and layout converters live in the library; old tensor_ee names re-export them.

The research names that re-exported them were archived with the research code (L06b).

Offline only: graphs are built against ``ee.apitestcase.ApiTestCase`` signatures
and compared by their serialization. No Earth Engine initialization or request.
"""
from __future__ import annotations

import ast
import importlib
import inspect
import os
from pathlib import Path
import subprocess
import sys

import ee
from ee.apitestcase import ApiTestCase
import numpy as np
import pytest

from recorded_paths import current
import tensor_ee
from tensor_ee import compiler
from tensor_ee.core import errors, grid as G, tensor as T
from tensor_ee.layouts import raster as R, shift as S, window as W

ROOT = Path(__file__).resolve().parents[1]

# (old module, old name, library module, library name): the old tensor_ee names of the
# L01 moves. Every old name is a re-export of the one library object. The research old
# names went with the research code (L06b, archive tag research-archive-2026-10-12).
MOVED = [
    # Grids (core/grid.py)
    ("tensor_ee.compiler", "Grid", "tensor_ee.core.grid", "Grid"),
    # Tensors and errors (core/tensor.py, core/errors.py)
    ("tensor_ee.compiler", "Raster", "tensor_ee.core.tensor", "Raster"),
    ("tensor_ee.compiler", "UnsupportedOperation", "tensor_ee.core.errors", "UnsupportedOperation"),
    # R converters (layouts/raster.py)
    ("tensor_ee.compiler", "_finish", "tensor_ee.layouts.raster", "finish"),
    ("tensor_ee.primitives", "from_image", "tensor_ee.layouts.raster", "from_image"),
    ("tensor_ee.primitives", "_float32", "tensor_ee.layouts.raster", "_float32"),
]

LIBRARY_MODULES = ("tensor_ee.core.grid", "tensor_ee.core.tensor", "tensor_ee.core.errors",
                   "tensor_ee.layouts.raster", "tensor_ee.layouts.window", "tensor_ee.layouts.shift")


def _attr(module: str, name: str):
    return getattr(importlib.import_module(module), name)


@pytest.mark.parametrize("old_module, old_name, new_module, new_name", MOVED,
                         ids=[f"{m}.{n}" for m, n, _, _ in MOVED])
def test_old_name_is_the_library_object(old_module, old_name, new_module, new_name):
    old, new = _attr(old_module, old_name), _attr(new_module, new_name)
    assert old is new
    if inspect.isfunction(new) or inspect.isclass(new):
        # Defined once, in the library: there is no second implementation.
        assert new.__module__ == new_module


def test_every_map_entry_owned_by_l01_is_at_its_target():
    import json
    module_map = json.loads((ROOT / "docs/evidence/l00-module-map.json").read_text())
    entries = [e for e in module_map["entries"] if e["owner"] == "L01"]
    assert len(entries) == 35
    for entry in entries:
        path, name = entry["target"].split("::")
        path = current(path)                         # the record predates the rename
        assert path in {f"{m.replace('.', '/')}.py" for m in LIBRARY_MODULES}
        target = _attr(path[:-3].replace("/", "."), name)
        if name == "WindowTensor":   # the decoder's Feature generalizes to WindowTensor
            assert issubclass(T.Feature, target)


def test_public_api_names_resolve_to_the_same_objects():
    assert len(tensor_ee.__all__) == 31
    for name in tensor_ee.__all__:
        assert getattr(tensor_ee, name) is not None
    assert tensor_ee.Grid is G.Grid and tensor_ee.Raster is T.Raster
    assert tensor_ee.from_image is R.from_image
    assert tensor_ee.UnsupportedOperation is errors.UnsupportedOperation
    assert issubclass(compiler.CompiledModel, T.Raster) and compiler.CompiledModel.__module__ == "tensor_ee.compiler"
    from tensor_ee import swin
    assert swin.__all__ == ["ServerArray", "SwinTParameters", "load_swin_t224_parameters",
                            "load_swin_t224_model", "SwinT224Adapter"]
    for name in (*swin.__all__, "_SKIP_BANDS", "_C4_BANDS"):
        assert getattr(swin, name) is not None


def test_one_unsupported_operation_class_everywhere():
    for module_name in ("tensor_ee", "tensor_ee.compiler", "tensor_ee.semantics", "tensor_ee.compatibility",
                        "tensor_ee.outputs", "tensor_ee.weight_assets", "tensor_ee.swin"):
        module = importlib.import_module(module_name)
        assert module.UnsupportedOperation is errors.UnsupportedOperation, module_name


def test_compiler_functions_look_up_its_finish_binding():
    # tests/test_pointwise_operators.py patches activations._finish (retargeted
    # by L03); the moved activations resolve the name in their module's globals.
    from tensor_ee.blocks.raster import activations
    assert compiler._finish is R.finish and activations._finish is R.finish
    for helper in (compiler._leaky_relu, compiler._silu, compiler._gelu, compiler._tanh,
                   compiler._eval_dropout, compiler._relu, compiler._softmax):
        assert helper.__globals__ is vars(activations)
        assert "_finish" in helper.__code__.co_names


# ---------------------------------------------------------------------------
# Defaults equal today's values
# ---------------------------------------------------------------------------
def _defaults(function) -> dict:
    return {name: p.default for name, p in inspect.signature(function).parameters.items()
            if p.default is not inspect.Parameter.empty}


def test_defaults_equal_todays_values():
    assert T.WINDOW == 7 and W.RING == 1 and S.SHIFT == 3
    assert S.STAGE_HALO == 1 and S.HIERARCHY_HALO == 3
    # L06b removed the research wraps and extension methods (WRAPS, METHODS) and the pin choices.
    assert not hasattr(S, "WRAPS") and not hasattr(S, "METHODS")
    assert _defaults(T.WindowTensor) == {"window": 7}
    assert _defaults(T.Feature) == {"layout": "window"}
    assert _defaults(G.TokenGrid) == {"scale": 10.0, "crs": "EPSG:3857"}
    assert _defaults(W.to_windows) == {"materialize": True, "window": 7, "mask": False}
    assert _defaults(W.from_dense) == {"window": 7}
    assert _defaults(W.to_dense) == {"window": 7}
    assert _defaults(W.zero_outside) == {"window": 7}
    assert _defaults(W.neighbours) == {"window": 7, "crop": None}       # crop: one padding domain
    assert _defaults(W.halo) == {"window": 7, "ring": 1, "crop": None}
    assert _defaults(W.taps) == {"k": 3, "window": 7}
    assert _defaults(W.im2col) == {"k": 3, "window": 7}
    assert _defaults(W.stride_samples) == {"materialize": True}
    assert _defaults(S.roll_axis) == {"window": 7, "pin": True}
    assert _defaults(S.roll_windows) == {"window": 7}
    # The shift defaults to window // 2 (P01 fix round): 3 at the default window.
    assert _defaults(S.shift_windows) == {"shift": None, "window": 7}
    assert _defaults(S.unshift_windows) == {"shift": None, "window": 7}
    assert S._resolve_shift(None, T.WINDOW) == S.SHIFT == T.WINDOW // 2
    assert _defaults(S.cyclic_neighbor) == {}
    assert _defaults(S.periodic_extension) == {"halo": 3}
    assert _defaults(R.stride_sample_phase) == {"a": 0, "b": 0}
    assert _defaults(R.finish) == {"clip_footprints": False}


# ---------------------------------------------------------------------------
# Passing the defaults explicitly serializes identically to omitting them
# ---------------------------------------------------------------------------
@pytest.fixture
def offline_ee():
    case = ApiTestCase()
    case.setUp()
    try:
        yield ee
    finally:
        case.tearDown()


def _same(*values):
    serialized = {value.serialize() for value in values}
    assert len(serialized) == 1


def test_explicit_defaults_serialize_identically(offline_ee):
    tokens = G.TokenGrid(28, 28, 2.5)
    coarse = tokens.coarse(7)
    windows = ee.Image("projects/l01-test/assets/windows")
    dense = ee.Image("projects/l01-test/assets/dense")
    rows = W.as_row(ee, dense.select(T.band_names(8)), 8)

    _same(W.to_windows(ee, rows, tokens), W.to_windows(ee, rows, tokens, True, window=7, mask=False))
    # The consolidated to_windows: layout's dense form is the masked rows form (L00 probe).
    _same(W.from_dense(ee, tokens, dense, 8), W.from_dense(ee, tokens, dense, 8, window=7),
          W.to_windows(ee, rows, tokens, mask=True),
          W.to_windows(ee, rows, tokens).updateMask(S.domain_mask(ee, coarse)))
    _same(W.to_dense(ee, tokens, windows, 8), W.to_dense(ee, tokens, windows, 8, window=7),
          W.depth_to_space(ee, tokens, windows, 7, 8).toFloat())
    _same(W.zero_outside(ee, windows, coarse, 8), W.zero_outside(ee, windows, coarse, 8, window=7))
    default, explicit = W.neighbours(ee, windows, coarse, 8), W.neighbours(ee, windows, coarse, 8, window=7)
    assert default.keys() == explicit.keys()
    for key in default:
        _same(default[key], explicit[key])
    halo = W.halo(ee, windows, coarse, 8)
    _same(halo, W.halo(ee, windows, coarse, 8, window=7, ring=1))
    for a, b in zip(W.taps(ee, halo, 8), W.taps(ee, halo, 8, 3, window=7), strict=True):
        _same(a, b)
    _same(W.im2col(ee, halo, 8), W.im2col(ee, halo, 8, 3, window=7))

    _same(S.shift_windows(ee, windows, coarse, 8), S.shift_windows(ee, windows, coarse, 8, 3, 7))
    _same(S.unshift_windows(ee, windows, coarse, 8), S.unshift_windows(ee, windows, coarse, 8, shift=3, window=7))
    _same(S.roll_axis(ee, windows, coarse, 8, 3, "y"), S.roll_axis(ee, windows, coarse, 8, 3, "y", window=7))
    _same(S.periodic_extension(ee, windows, coarse), S.periodic_extension(ee, windows, coarse, halo=3))
    _same(S.translate(ee, windows, coarse), S.translate(ee, windows, coarse, 0, 0))

    feature_grid = G.FeatureGrid("EPSG:32651", 500000.0, 1600000.0, 40.0, 28, 28)
    _same(R.stride_sample_phase(ee, dense, feature_grid, 2), R.stride_sample_phase(ee, dense, feature_grid, 2, 0, 0))
    grid = G.Grid("EPSG:3857", (20, 0, 1000, 0, -20, 2000), 4, 4)
    _same(R.finish(dense, ("a",), grid).image, R.finish(dense, ("a",), grid, False).image)


def test_non_default_parameters_reach_the_graph(offline_ee):
    # The parameters are live (not ignored), though values other than the
    # defaults are not validated.
    tokens = G.TokenGrid(30, 30, 2.5)
    coarse = tokens.coarse(5)
    windows = ee.Image("projects/l01-test/assets/windows")
    assert W.halo(ee, windows, coarse, 8, window=5).serialize() != W.halo(ee, windows, coarse, 8).serialize()
    assert (S.shift_windows(ee, windows, coarse, 8, 2, 5).serialize()
            != S.shift_windows(ee, windows, coarse, 8).serialize())


# ---------------------------------------------------------------------------
# Grids: constructors, equality, literals, conversions
# ---------------------------------------------------------------------------
def test_grid_kinds_keep_their_constructors_and_equality():
    assert list(inspect.signature(G.Grid).parameters) == ["crs", "transform", "width", "height"]
    assert list(inspect.signature(G.TokenGrid).parameters) == ["height", "width", "scale", "crs"]
    assert list(inspect.signature(G.FeatureGrid).parameters) == ["crs", "x0", "y0", "scale", "height", "width"]
    assert G.Grid("EPSG:3857", [2.5, 0, 0, 0, -2.5, 0], 28, 28).transform == (2.5, 0, 0, 0, -2.5, 0)
    assert G.Grid("EPSG:3857", (2.5, 0, 0, 0, -2.5, 0), 28, 28) == G.Grid("EPSG:3857", (2.5, 0.0, 0.0, 0.0, -2.5, 0.0), 28, 28)
    assert G.TokenGrid(28, 28) == G.TokenGrid(28, 28, 10.0, "EPSG:3857")
    assert G.TokenGrid(28, 28).transform == [10.0, 0, 0, 0, -10.0, 0]
    assert G.FeatureGrid("EPSG:3857", 0, 0, 2.5, 28, 28).transform == [2.5, 0.0, 0, 0.0, -2.5, 0]
    # Distinct kinds never compare equal, even on the same pixels.
    token = G.TokenGrid(28, 28, 2.5)
    assert token != G.to_feature_grid(token) and token != G.to_grid(token)
    with pytest.raises(ValueError):
        G.Grid("EPSG:3857", (2.5, 0, 0, 0, 2.5, 0), 28, 28)
    for kind in (G.Grid("EPSG:3857", (2.5, 0, 0, 0, -2.5, 0), 28, 28), token,
                 G.FeatureGrid("EPSG:3857", 0, 0, 2.5, 28, 28)):
        assert isinstance(kind, G.GridProtocol)


def test_grid_literals_are_kept_byte_for_byte(offline_ee):
    grid = G.Grid("EPSG:3857", (2.5, 0, 0, 0, -2.5, 0), 28, 28)
    token = G.TokenGrid(28, 28, 2.5)
    feature = G.FeatureGrid("EPSG:3857", 0, 0, 2.5, 28, 28)
    _same(grid.projection(), ee.Projection("EPSG:3857", [2.5, 0, 0, 0, -2.5, 0]))
    _same(token.projection(ee), ee.Projection("EPSG:3857", [2.5, 0, -0.0, 0, -2.5, 0.0]))
    _same(feature.projection(ee), ee.Projection("EPSG:3857", [2.5, 0, 0.0, 0, -2.5, 0.0]))
    assert len({grid.projection().serialize(), token.projection(ee).serialize(),
                feature.projection(ee).serialize()}) == 3
    # Grid.mask() is unfloored; the TokenGrid/FeatureGrid domains floor.
    assert "floor" not in grid.mask().serialize() and "floor" in token.domain(ee).serialize()
    _same(grid.domain(ee), grid.mask())


def test_grid_protocol_members():
    grid = G.Grid("EPSG:3857", (2.5, 0, 100, 0, -2.5, 200), 28, 28)
    assert grid.scale == 2.5
    assert grid.coarse(7) == grid.resized(7, 4, 4)
    assert grid.fine(2) == grid.resized(1 / 2, 56, 56)
    with pytest.raises(ValueError):
        grid.coarse(3)
    with pytest.raises(ValueError):
        G.Grid("EPSG:3857", (2.5, 0, 0, 0, -5, 0), 28, 28).scale
    token = G.TokenGrid(28, 28, 2.5)
    # Exactly the grid the research resize built for grids without ``fine`` (archived).
    assert token.fine(4) == G.TokenGrid(28 * 4, 28 * 4, 2.5 / 4, "EPSG:3857")
    assert token.resized(7, 4, 4) == token.coarse(7)
    feature = G.FeatureGrid("EPSG:32651", 10.0, 20.0, 40.0, 28, 28)
    assert feature.resized(7, 4, 4) == feature.coarse(7)
    assert not hasattr(token, "x0")   # adaptive_pool reads x0/y0 with a 0.0 fallback


def test_feature_grid_fine_bounds_and_region(offline_ee):
    # Before L06b only research code ran these; they stay covered offline.
    feature = G.FeatureGrid("EPSG:32651", 10.0, 20.0, 40.0, 28, 14)
    assert feature.fine(2) == G.FeatureGrid("EPSG:32651", 10.0, 20.0, 20.0, 56, 28)
    assert feature.fine(2) == feature.resized(1 / 2, 28, 56) and feature.fine(2).coarse(2) == feature
    assert feature.bounds == [10.0, 20.0 - 40.0 * 28, 10.0 + 40.0 * 14, 20.0]
    expected = ee.Geometry.Rectangle([10.0, -1100.0, 570.0, 20.0], "EPSG:32651", False)
    assert feature.region(ee).serialize() == expected.serialize()


def test_grid_conversions_round_trip():
    grid = G.Grid("EPSG:32651", (40, 0, 500000, 0, -40, 1600000), 56, 56)
    feature = G.to_feature_grid(grid)
    assert feature == G.FeatureGrid("EPSG:32651", 500000, 1600000, 40, 56, 56)
    assert G.to_grid(feature) == grid
    token = G.TokenGrid(56, 56, 2.5)
    assert G.to_token_grid(G.to_grid(token)) == token
    assert G.to_token_grid(G.to_feature_grid(token)) == token
    assert G.to_feature_grid(token) == G.FeatureGrid("EPSG:3857", 0.0, 0.0, 2.5, 56, 56)
    assert G.to_feature_grid(G.to_grid(feature)) == feature
    assert G.to_grid(grid) is grid and G.to_feature_grid(feature) is feature
    assert G.to_token_grid(token) is token
    with pytest.raises(ValueError, match="origin"):
        G.to_token_grid(grid)
    with pytest.raises(ValueError, match="square"):
        G.to_feature_grid(G.Grid("EPSG:3857", (2.5, 0, 0, 0, -5, 0), 4, 4))
    with pytest.raises(TypeError):
        G.to_grid(object())


def test_model_grid_coupling_is_asserted():
    # The adapter's 2.5 m model grid coarsened by the patch is the encoder's
    # default-scale stage-1 TokenGrid (swin_encoder.encoder_grids).
    model = G.TokenGrid(224, 224, 2.5)
    assert G.check_model_grid(model) is model
    assert model.coarse(4) == G.TokenGrid(56, 56)
    with pytest.raises(ValueError):
        G.check_model_grid(G.TokenGrid(224, 224, 5.0))
    with pytest.raises(ValueError):
        G.check_model_grid(G.TokenGrid(224, 224, 2.5, "EPSG:32651"))
    with pytest.raises(TypeError):
        G.check_model_grid(G.to_feature_grid(model))


# ---------------------------------------------------------------------------
# W tensor
# ---------------------------------------------------------------------------
def test_window_tensor_and_feature():
    grid = G.TokenGrid(14, 14)
    tensor = T.WindowTensor("image", grid, 8)
    assert tensor.window == 7 and tensor.like("other").window == 7
    assert T.WindowTensor("image", grid, 8, 1).window == 1
    for bad in (0, -7, 7.0, True):
        with pytest.raises(ValueError):
            T.WindowTensor("image", grid, 8, bad)
    # Feature keeps its constructor (positional layout) and layout values.
    assert list(inspect.signature(T.Feature).parameters) == ["image", "grid", "channels", "layout"]
    assert T.LAYOUTS == ("window", "dense_bands", "dense_row")
    assert issubclass(T.Feature, T.WindowTensor)
    window, bands, row = (T.Feature("image", grid, 8, layout) for layout in T.LAYOUTS)
    assert (window.window, bands.window, row.window) == (7, 1, 1)
    assert T.Feature("image", grid, 8) == window
    like = bands.like("other", 4)
    assert type(like) is T.Feature and (like.layout, like.channels, like.window) == ("dense_bands", 4, 1)
    with pytest.raises(ValueError):
        T.Feature("image", grid, 8, "tokens")


def test_same_grid_refuses_grid_layout_and_window_mismatch():
    grid = G.TokenGrid(14, 14)
    T.same_grid(T.WindowTensor("a", grid, 8), T.WindowTensor("b", grid, 8), T.Feature("c", grid, 8))
    with pytest.raises(errors.GridMismatch):
        T.same_grid(T.WindowTensor("a", grid, 8), T.WindowTensor("b", grid, 8, 1))
    with pytest.raises(errors.GridMismatch):
        T.same_grid(T.WindowTensor("a", grid, 8), T.WindowTensor("b", G.TokenGrid(14, 14, 20.0), 8))
    with pytest.raises(errors.GridMismatch):
        T.same_grid(T.Feature("a", grid, 8, "dense_bands"), T.Feature("b", grid, 8, "dense_row"))
    from tensor_ee.blocks.window import merge as WM
    with pytest.raises(errors.GridMismatch):
        WM.add(T.Feature("a", grid, 8), T.Feature("b", grid, 8, "dense_row"))


# ---------------------------------------------------------------------------
# Layer rule: core and layouts import no research module, blocks or converter
# ---------------------------------------------------------------------------
FORBIDDEN = ("ee_attention", "swin_", "upernet", "tensor_ee.blocks", "tensor_ee.converter",
             "tensor_ee.compiler", "tensor_ee.semantics", "tensor_ee.tiling", "tensor_ee.receptive_field",
             "tensor_ee.onnx_import", "tensor_ee.compatibility", "tensor_ee.weight_assets",
             "tensor_ee.weight_mapping", "tensor_ee.primitives", "tensor_ee.outputs", "tensor_ee.swin",
             "tensor_ee._swin_parameters", "tensor_ee.models", "tensor_ee.segwater")
ALLOWED = ("ee", "numpy", "__future__", "dataclasses", "enum", "inspect", "math", "numbers", "typing",
           "tensor_ee.core", "tensor_ee.layouts")
# inspect and numbers (standard library) joined at X02b: core/metadata.py checks a validated
# record's setting names against the block's signature (lead decision, X02b integration).
# No layer exception remains: L02 gave embedded_raster's float32 encoder a core
# home (tensor_ee.core.precision).
KNOWN_EXCEPTIONS = set()


def _imports(path: Path):
    package = ".".join(path.relative_to(ROOT).with_suffix("").parts[:-1])
    tree = ast.parse(path.read_text())
    scopes = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for child in ast.walk(node):
                scopes.setdefault(id(child), node.name)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parent = package.split(".")[: len(package.split(".")) - node.level + 1]
                base = ".".join(parent + ([base] if base else []))
            names = [base]
        else:
            continue
        for name in names:
            yield name, scopes.get(id(node))


def test_core_and_layouts_import_no_research_blocks_or_converter():
    found, exceptions = [], set()
    files = sorted((ROOT / "tensor_ee/core").glob("*.py")) + sorted((ROOT / "tensor_ee/layouts").glob("*.py"))
    assert {p.name for p in files} >= {"__init__.py", "grid.py", "tensor.py", "errors.py",
                                       "raster.py", "window.py", "shift.py"}
    for path in files:
        for name, scope in _imports(path):
            relative = str(path.relative_to(ROOT))
            if (relative, scope, name) in KNOWN_EXCEPTIONS:
                exceptions.add((relative, scope, name))
                continue
            assert not name.startswith(FORBIDDEN), (relative, name)
            if not any(name == allowed or name.startswith(allowed + ".") for allowed in ALLOWED):
                found.append((relative, name))
    assert not found
    assert exceptions == KNOWN_EXCEPTIONS


def _environment() -> dict:
    return {**os.environ, "PYTHONPATH": str(ROOT)}


def test_importing_the_library_loads_no_research_module():
    code = ("import sys; import " + ", ".join(LIBRARY_MODULES) + "\n"
            "bad = sorted(m for m in sys.modules if m.split('.')[0] in "
            "{'ee_attention', 'upernet'} or m.startswith('swin_'))\n"
            "assert not bad, bad")
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, env=_environment())
