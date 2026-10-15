"""L05: both reference models are built only from public library modules.

- The Swin-T224/UPerNet composites live in ``tensor_ee.models.swin_t224``
  (configuration, encoder, decoder), at the targets the L00 module map names; the research
  modules that re-exported them were archived at L06b.
- Their constants builders take layouts instead of row counts, and the composites pass
  ``layout=`` to the blocks: the explicit layout forms serialize exactly as the old
  row-count forms did.
- No module under ``tensor_ee`` imports a research folder, statically (AST, including
  function-local imports) or dynamically (a fresh interpreter builds the three Swin
  stages and loads none).
- ``tensor_ee/models.py`` is the ``tensor_ee.models`` package (D3); ``import tensor_ee``
  does not load ``tensor_ee.models.swin_t224``.
- The adapter keeps its literal ``TokenGrid(224, 224, 2.5)`` and checks it.

Offline only: graphs are built against ``ee.apitestcase.ApiTestCase`` signatures and
compared by their serialization. No Earth Engine initialization or request.
"""
from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import ee
from ee.apitestcase import ApiTestCase
import numpy as np
import pytest

from recorded_paths import current
from tensor_ee import _swin_parameters as SP
from tensor_ee import swin as S
from tensor_ee.blocks.window import merge as WM
from tensor_ee.blocks.window.attention import relative_position_bias
from tensor_ee.blocks.window.conv import Conv3x3Constants
from tensor_ee.blocks.window.linear import LinearConstants, conv1x1_constants
from tensor_ee.blocks.window.normalization import BatchNormConstants
from tensor_ee.blocks.window.pooling import SIZES
from tensor_ee.core.errors import UnsupportedOperation
from tensor_ee.core.grid import TokenGrid, check_model_grid
from tensor_ee.core.ops import ValueOps, WindowOps
from tensor_ee.core.tensor import Feature, WindowTensor
from tensor_ee.models import fixtures as FX
from tensor_ee.models.swin_t224 import configuration as C
from tensor_ee.models.swin_t224 import decoder as D
from tensor_ee.models.swin_t224 import encoder as E
from tensor_ee.weights.constants import RowConstant, RowMismatch, ServerArray, WindowLayout, row_constant

ROOT = Path(__file__).resolve().parents[1]
LIB = "tensor_ee.models.swin_t224"
RESEARCH = ("ee_attention", "swin_", "upernet")

# ---------------------------------------------------------------------------
# The L05 entries of the module map are at their library targets
# ---------------------------------------------------------------------------
def _attr(module: str, name: str):
    value = importlib.import_module(module)
    for part in name.split("."):
        value = getattr(value, part)
    return value


PROMOTED = {f"{LIB}.encoder": "tensor_ee.blocks.window.swin", f"{LIB}.decoder": "tensor_ee.blocks.window.upernet",
            f"{LIB}.configuration": "tensor_ee.blocks.window.swin_upernet_config"}
"""The swin_t224 model module of each L05 target -> the window block X01b promoted its
composites to; the model module re-exports them."""


def test_the_51_entries_are_at_their_targets():
    """Every L05 entry resolves at its library target and is defined there, or, for a composite
    moved to a ``swin_t224`` target, in the window block X01b promoted that module's composites
    to (the old research paths were archived at L06b)."""
    entries = [e for e in json.loads((ROOT / "docs/evidence/l00-module-map.json").read_text())["entries"]
               if e["owner"] == "L05"]
    assert len(entries) == 51
    for e in entries:
        path, names = e["target"].split("::")
        module = current(path)[:-3].replace("/", ".")   # the record predates the rename
        for name in names.split(", "):
            new = _attr(module, name)
            if inspect.isfunction(new) or inspect.isclass(new):
                promoted = e["action"] == "move" and PROMOTED.get(module) == new.__module__
                assert new.__module__ == module or promoted, e["id"]
            if e["action"] == "move":
                assert module.startswith(f"{LIB}."), e["id"]


# ---------------------------------------------------------------------------
# Offline fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def offline_ee():
    case = ApiTestCase()
    case.setUp()
    try:
        yield ee
    finally:
        case.tearDown()


def _server(shape, value: int) -> ServerArray:
    """A server-side constant array (no values in the request), stored as the assets are."""
    stored = (shape[0], int(np.prod(shape[1:]))) if len(shape) == 4 else tuple(shape)
    array = ee.Array(ee.List.repeat(value, int(np.prod(shape)))).reshape(list(stored)).float()
    return ServerArray(array, stored)


@pytest.fixture
def parameters(offline_ee):
    enc, pe, dec, _ = SP.parameter_shapes()
    count = iter(range(1, 10_000))
    return ({g: {k: _server(s, next(count)) for k, s in p.items()} for g, p in enc.items()},
            {k: _server(s, next(count)) for k, s in dec.items()})


def _sha(value) -> str:
    return hashlib.sha256(value.serialize().encode()).hexdigest()


def _graphs(tensors: dict) -> dict:
    return {k: _sha(v) for k, v in tensors.items() if hasattr(v, "serialize")}


# ---------------------------------------------------------------------------
# Layout forms serialize as the old row-count forms
# ---------------------------------------------------------------------------
def _row_count_block_constants(ops, p):
    """block_constants as it was before L05: repeat_rows=49 and LinearConstants(..., 49)."""
    return E.BlockConstants(
        norm1_weight=ops.constant(p["norm1.weight"], repeat_rows=49),
        norm1_bias=ops.constant(p["norm1.bias"], repeat_rows=49),
        qkv=LinearConstants(ops, p["attn.qkv.weight"], p["attn.qkv.bias"], 49),
        relative_bias=[ops.wrap(b) for b in relative_position_bias(ops.ee, p["attn.relative_position_bias_table"], 7)],
        proj=LinearConstants(ops, p["attn.proj.weight"], p["attn.proj.bias"], 49),
        norm2_weight=ops.constant(p["norm2.weight"], repeat_rows=49),
        norm2_bias=ops.constant(p["norm2.bias"], repeat_rows=49),
        fc1=LinearConstants(ops, p["mlp.fc1.weight"], p["mlp.fc1.bias"], 49),
        fc2=LinearConstants(ops, p["mlp.fc2.weight"], p["mlp.fc2.bias"], 49),
        dim=p["norm1.weight"].shape[0], heads=p["attn.relative_position_bias_table"].shape[1])


@pytest.mark.parametrize("group", ["s1b0", "s2b1", "s3b4"])
def test_block_constants_layout_forms_serialize_as_the_row_count_form(parameters, group):
    enc, _ = parameters
    ops, x = WindowOps(ee), ee.Image("projects/l05-test/assets/x")
    p = enc[group]
    old = _row_count_block_constants(ops, p)
    for new in (E.block_constants(ops, p), E.block_constants(ops, p, layout=WindowLayout(7))):
        assert isinstance(new.norm1_weight, RowConstant) and new.norm1_weight.rows == 49
        assert all(getattr(new, n).rows == 49 for n in ("qkv", "proj", "fc1", "fc2"))
        assert (new.dim, new.heads, new.head_dim) == (old.dim, old.heads, old.head_dim)
        for name in ("norm1_weight", "norm1_bias", "norm2_weight", "norm2_bias"):
            assert _sha(getattr(new, name).value) == _sha(getattr(old, name))
        assert [_sha(b) for b in new.relative_bias] == [_sha(b) for b in old.relative_bias]
        assert _graphs(E.swin_block(ops, x, new)) == _graphs(E.swin_block(ops, x, old))
        assert _graphs(E.mlp_branch(ops, x, new)) == _graphs(E.mlp_branch(ops, x, old))
    coarse = E.encoder_grids(8, 8).coarse[0]
    assert (_graphs(E.shifted_swin_block(ops, x, E.block_constants(ops, p), coarse))
            == _graphs(E.shifted_swin_block(ops, x, old, coarse)))


def test_block_constants_value_path_is_unchanged(parameters):
    enc, _ = parameters
    ops = ValueOps(ee)
    x = ee.Array(ee.List.repeat(0.5, 49 * 96)).reshape([49, 96]).float()
    old, new = _row_count_block_constants(ops, enc["s1b0"]), E.block_constants(ops, enc["s1b0"])
    assert _graphs(E.swin_block(ops, x, new)) == _graphs(E.swin_block(ops, x, old))


def test_encoder_final_norm_layout_form_serializes_as_before(parameters):
    enc, _ = parameters
    ops, x3 = WindowOps(ee), ee.Image("projects/l05-test/assets/s3")
    g = E.encoder_grids(8, 8)
    vector = np.linspace(-1, 1, 768).astype(np.float32)
    raw = (ops.constant(vector, repeat_rows=49), ops.constant(vector + 1, repeat_rows=49))
    rows = (row_constant(ops, vector, C.WINDOW_LAYOUT), row_constant(ops, vector + 1, C.WINDOW_LAYOUT))
    blocks, merges = [[]] * 3 + [E.stage_blocks(ops, enc, 4)], [None] * 2 + [E.merge_constants(ops, enc, 3)]
    out = [E.encoder(ops, x3, E.EncoderConstants(blocks, merges, *norm), g, start=4, stop=4,
                     extension_mode="per_block")["norm"] for norm in (raw, rows)]
    assert _sha(out[0]) == _sha(out[1])


def _row_count_decoder(ee_, dec):
    """The PSP, FPN, fusion and head constants as built before L05 (row counts)."""
    img, arr = WindowOps(ee_), ValueOps(ee_)

    def bn(ops, prefix, rows):
        return BatchNormConstants(ops, *D.batch_norm_args(dec, prefix), rows=rows)
    psp = D.PSPConstants(
        (img.constant(dec["decoder.feature_norms.3.weight"], repeat_rows=49),
         img.constant(dec["decoder.feature_norms.3.bias"], repeat_rows=49)),
        [conv1x1_constants(arr, dec[f"decoder.psp.blocks.{i}.1.0.weight"], None, s * s) for i, s in enumerate(SIZES)],
        [bn(arr, f"decoder.psp.blocks.{i}.1.1", s * s) for i, s in enumerate(SIZES)],
        Conv3x3Constants(img, dec["decoder.psp.out_conv.0.weight"]), bn(img, "decoder.psp.out_conv.1", 49))
    norm, lateral, lateral_bn, conv, conv_bn = {}, {}, {}, {}, {}
    for k in C.LEVELS:
        norm[k] = (img.constant(dec[f"{C.norm_prefix(k)}.weight"], repeat_rows=49),
                   img.constant(dec[f"{C.norm_prefix(k)}.bias"], repeat_rows=49))
        lateral[k] = conv1x1_constants(img, dec[f"{C.lateral_prefix(k)}.0.weight"], None, 49)
        lateral_bn[k] = bn(img, f"{C.lateral_prefix(k)}.1", 49)
        conv[k] = Conv3x3Constants(img, dec[f"{C.conv_prefix(k)}.0.weight"])
        conv_bn[k] = bn(img, f"{C.conv_prefix(k)}.1", 49)
    fpn = D.FPNConstants(psp, norm, lateral, lateral_bn, conv, conv_bn)
    groups = [Conv3x3Constants(img, w) for w in (D.group_weight(dec[C.WEIGHT], g, 4) for g in range(4))]
    fusion = D.FusionConstants(groups, BatchNormConstants(img, *C.bn_args(dec), rows=49))
    head = conv1x1_constants(img, dec[D.HEAD_WEIGHT], dec[D.HEAD_BIAS], 49)
    return fpn, fusion, head


def _features():
    return {k: (ee.Image(f"projects/l05-test/assets/c{k}"), S._model_grid(k)) for k in (1, 2, 3, 4)}


def test_decoder_layout_forms_serialize_as_the_row_count_forms(parameters):
    _, dec = parameters
    old_fpn, old_fusion, old_head = _row_count_decoder(ee, dec)
    new_fpn = D.fpn_constants(ee, dec)
    new_fusion = D.fusion_constants(ee, dec, ("grouped_double",))
    new_head = D.head_constants(ee, dec)
    assert all(isinstance(v, RowConstant) and v.rows == 49 for v in new_fpn.psp.norm)
    assert [c.rows for c in new_fpn.psp.branch_conv] == [b.rows for b in new_fpn.psp.branch_bn] == [1, 4, 9, 36]
    assert new_fpn.psp.out_bn.rows == new_fusion.bn.rows == new_head.rows == 49
    assert all(new_fpn.norm[k][0].rows == new_fpn.lateral[k].rows == new_fpn.lateral_bn[k].rows == 49
               for k in C.LEVELS)
    # The cost counts read the rows from the layout; the values are the base revision's.
    assert (new_fpn.psp.embedded_values, new_fpn.embedded_values) == (5_091_840, 7_400_256)
    assert new_fusion.embedded_values == 2_409_472
    assert _graphs(D.head_constants(ee, dec, rows=49).__dict__) == _graphs(new_head.__dict__)

    feats = _features()
    old_t, old_g = D.fpn_forward(ee, feats, old_fpn)
    new_t, new_g = D.fpn_forward(ee, feats, new_fpn)
    assert _graphs(new_t) == _graphs(old_t) and new_g == old_g
    pyramid = D.pyramid_features(new_t, new_g)
    old_f, old_fg = D.fusion_forward(ee, pyramid, old_fusion, ("grouped_double",))
    new_f, new_fg = D.fusion_forward(ee, pyramid, new_fusion, ("grouped_double",))
    assert _graphs(new_f) == _graphs(old_f) and new_fg == old_fg
    fused = Feature(new_f["double_relu"], new_fg["double_relu"], 256)
    assert _graphs(D.head_forward(ee, fused, new_head)[0]) == _graphs(D.head_forward(ee, fused, old_head)[0])
    pooled = ee.Array(ee.List.repeat(1, 9 * 768)).reshape([9, 768]).float()
    assert _graphs(D.psp_branch(ee, pooled, new_fpn.psp, 2)) == _graphs(D.psp_branch(ee, pooled, old_fpn.psp, 2))


def test_constants_built_for_another_layout_are_refused(parameters):
    enc, dec = parameters
    ops, x = WindowOps(ee), ee.Image("projects/l05-test/assets/x")
    p, one = enc["s1b0"], WindowLayout(1)
    good = E.block_constants(ops, p)
    coarse = E.encoder_grids(8, 8).coarse[0]
    ln, linear = (lambda n: row_constant(ops, p[n], one)), (lambda n: LinearConstants(ops, p[f"{n}.weight"],
                                                                                    p[f"{n}.bias"], one))
    composites = {"norm1_weight": (E.swin_block, E.shifted_swin_block), "proj": (E.swin_block, E.shifted_swin_block),
                  "norm2_bias": (E.mlp_branch,), "fc1": (E.mlp_branch,), "fc2": (E.mlp_branch, E.swin_block)}
    built = {"norm1_weight": ln("norm1.weight"), "proj": linear("attn.proj"), "norm2_bias": ln("norm2.bias"),
             "fc1": linear("mlp.fc1"), "fc2": linear("mlp.fc2")}
    for field, value in built.items():
        bad = dataclasses.replace(good, **{field: value})
        for composite in composites[field]:
            with pytest.raises(RowMismatch):
                composite(ops, x, bad, *((coarse,) if composite is E.shifted_swin_block else ()))
    with pytest.raises(RowMismatch):
        D.head_conv(ee, x, D.head_constants(ee, dec, rows=1))
    psp = D.psp_constants(ee, dec)
    swapped = D.PSPConstants(psp.norm, psp.branch_conv[::-1], psp.branch_bn[::-1], psp.out_conv, psp.out_bn)
    pooled = ee.Array(ee.List.repeat(1, 768)).reshape([1, 768]).float()
    with pytest.raises(RowMismatch):
        D.psp_branch(ee, pooled, swapped, 0)
    dense_norms = D.PSPConstants(tuple(row_constant(ops, np.ones(768, np.float32), WindowLayout(1))
                                       for _ in range(2)), *list(vars(psp).values())[1:])
    with pytest.raises(RowMismatch):
        D.psp_layer_norm(ee, x, dense_norms)
    fpn = D.fpn_constants(ee, dec)
    bad = D.FPNConstants(fpn.psp, fpn.norm, fpn.lateral, {3: BatchNormConstants(ops, *D.batch_norm_args(
        dec, f"{C.lateral_prefix(3)}.1"))}, fpn.conv, fpn.conv_bn)
    with pytest.raises(RowMismatch):
        D.fpn_lateral(ee, x, bad, 3)


# ---------------------------------------------------------------------------
# Policy, the concat grant and the research registries
# ---------------------------------------------------------------------------
def test_grouped_double_accepts_both_policies_and_adds_no_cast(parameters):
    _, dec = parameters
    c = D.fusion_constants(ee, dec, ("grouped_double",))
    tokens = S._model_grid(1)
    parts = [Feature(ee.Image(f"projects/l05-test/assets/p{g}"), tokens, 256) for g in range(4)]
    default = _graphs(D.grouped_double(ee, parts, c))
    assert default == _graphs(D.grouped_double(ee, parts, c, policy="grouped_double"))
    assert default == _graphs(D.grouped_double(ee, parts, c, policy="double_round_once"))
    for policy in ("round_each_op", "float32_operands"):
        with pytest.raises(UnsupportedOperation):
            D.grouped_double(ee, parts, c, policy=policy)
    with pytest.raises(ValueError):
        D.grouped_double(ee, parts, c, policy="nope")
    assert inspect.signature(D.grouped_double).parameters["policy"].default == "grouped_double"


def test_concat_accepts_a_bare_window_tensor(offline_ee):
    tokens = S._model_grid(1)
    a, b = ee.Image("projects/l05-test/assets/a"), ee.Image("projects/l05-test/assets/b")
    bare = WM.concat([WindowTensor(a, tokens, 3), WindowTensor(b, tokens, 4)])
    feature = WM.concat([Feature(a, tokens, 3), Feature(b, tokens, 4)])
    assert type(bare) is WindowTensor and bare.channels == feature.channels == 7
    assert _sha(bare.image) == _sha(feature.image)
    dense = WM.concat([Feature(a, tokens, 3, "dense_bands"), Feature(b, tokens, 4, "dense_bands")])
    assert dense.layout == "dense_bands" and _sha(dense.image) != _sha(feature.image)


def test_research_variants_are_removed_and_refused(parameters):
    """L06b removed the research registries and variants: only the production paths remain,
    and the research names are refused."""
    enc, dec = parameters
    for module, registry in ((E, "EXTENSION_HALOS"), (D, "PSP_POOLS"), (D, "FUSION_FORMULATIONS"),
                             (D, "HEAD_RESIZES")):
        assert not hasattr(module, registry), registry
    ops, x = WindowOps(ee), ee.Image("projects/l05-test/assets/x")
    g = E.encoder_grids(8, 8)
    c = E.EncoderConstants([E.stage_blocks(ops, enc, 1)], [], None, None)
    for mode in ("per_stage", "propagated"):
        with pytest.raises(ValueError, match="extension_mode"):
            E.encoder(ops, x, c, g, start=1, stop=1, extension_mode=mode)
    assert "s1_out" in E.encoder(ops, x, c, g, start=1, stop=1, extension_mode="per_block")   # production
    for option in ("stage_halos", "halo", "gather", "pin_stage_outputs", "pin_block_outputs", "extension"):
        assert option not in inspect.signature(E.encoder).parameters, option
    feats = _features()
    with pytest.raises(TypeError):
        D.psp_forward(ee, *feats[4], D.psp_constants(ee, dec), pool="global")
    t, grids = D.fpn_forward(ee, feats, D.fpn_constants(ee, dec))
    pyramid = D.pyramid_features(t, grids)
    for formulations in (("monolithic",), ("grouped_rounded",), ("unknown",), ("grouped_double", "monolithic")):
        with pytest.raises(ValueError, match="formulations"):
            D.fusion_constants(ee, dec, formulations)
        with pytest.raises(ValueError, match="formulations"):
            D.fusion_forward(ee, pyramid, D.fusion_constants(ee, dec), formulations)
    head = Feature(ee.Image("projects/l05-test/assets/h"), S._model_grid(1), 2)
    with pytest.raises(TypeError):
        D.head_upsample(ee, head, method="dense")


# ---------------------------------------------------------------------------
# Layer rule: no module under tensor_ee imports a research folder
# ---------------------------------------------------------------------------
def _imports(path: Path):
    """Every import in the file (module level, in functions, guarded), resolved."""
    package = ".".join(path.relative_to(ROOT).with_suffix("").parts[:-1])
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parent = package.split(".")[: len(package.split(".")) - node.level + 1]
                base = ".".join(parent + ([base] if base else []))
            yield base
        elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name in ("import_module", "__import__") and isinstance(node.args[0].value, str):
                yield node.args[0].value


def _within(name, prefixes):
    return any(name == p or name.startswith(p + ".") for p in prefixes)


def test_no_module_under_tensor_ee_imports_a_research_folder():
    files = sorted((ROOT / "tensor_ee").rglob("*.py"))
    assert ROOT / "tensor_ee/swin.py" in files and ROOT / "tensor_ee/models/swin_t224/decoder.py" in files
    found = [(str(p.relative_to(ROOT)), name) for p in files for name in _imports(p)
             if name.split(".")[0].startswith(RESEARCH)]
    assert found == []


MODEL_LAYER = ("tensor_ee/swin.py", "tensor_ee/_swin_parameters.py", "tensor_ee/swin_workflow.py",
               "tensor_ee/segwater_inference.py", "tensor_ee/segwater_preprocessing.py",
               "tensor_ee/segwater_pytorch.py")
THIRD_PARTY = ("ee", "numpy", "torch", "segmentation_models_pytorch", "google.auth")
LIBRARY = ("tensor_ee.core", "tensor_ee.layouts", "tensor_ee.weights", "tensor_ee.blocks", "tensor_ee.converter",
           "tensor_ee.compiler", "tensor_ee.semantics", "tensor_ee.tiling", "tensor_ee.receptive_field",
           "tensor_ee.onnx_import", "tensor_ee.compatibility", "tensor_ee.weight_assets", "tensor_ee.weight_mapping",
           "tensor_ee.models", "tensor_ee.swin", "tensor_ee._swin_parameters", "tensor_ee.segwater_preprocessing",
           "tensor_ee.segwater_pytorch", "tensor_ee.segwater_inference")


def test_reference_models_import_only_library_modules():
    files = sorted((ROOT / "tensor_ee/models").rglob("*.py")) + [ROOT / f for f in MODEL_LAYER]
    violations = [(str(p.relative_to(ROOT)), name) for p in files for name in _imports(p)
                  if not (name.split(".")[0] in sys.stdlib_module_names or name == "__future__"
                          or _within(name, THIRD_PARTY) or _within(name, LIBRARY))]
    assert violations == []
    # The library layers below the models never import them.
    lower = [p for d in ("core", "layouts", "weights", "blocks", "converter")
             for p in sorted((ROOT / "tensor_ee" / d).rglob("*.py"))]
    assert [(str(p.relative_to(ROOT)), n) for p in lower for n in _imports(p)
            if _within(n, ("tensor_ee.models", "tensor_ee.swin", "tensor_ee._swin_parameters"))] == []


FRESH = r'''
import sys
from pathlib import Path
import tensor_ee
assert Path(tensor_ee.__file__).resolve().is_relative_to(Path.cwd())
assert not any(m == "tensor_ee.models.swin_t224" or m.startswith("tensor_ee.models.swin_t224.") for m in sys.modules)
import tensor_ee.models
assert not any(m.startswith("tensor_ee.models.swin_t224") for m in sys.modules), "fixtures load the Swin model"
import tensor_ee.swin, tensor_ee.swin_workflow, tensor_ee._swin_parameters
import tensor_ee.segwater_inference, tensor_ee.segwater_preprocessing, tensor_ee.segwater_pytorch
import tensor_ee.models.swin_t224.configuration, tensor_ee.models.swin_t224.encoder, tensor_ee.models.swin_t224.decoder
import ee, numpy as np
from ee.apitestcase import ApiTestCase
case = ApiTestCase(); case.setUp()
from tensor_ee import Grid, Raster, _swin_parameters as P
from tensor_ee.swin import ServerArray, SwinT224Adapter, SwinTParameters
from tensor_ee.swin_workflow import build_stage, read_resources
enc, pe, dec, _ = P.parameter_shapes()
def arrays(schema):
    out = {}
    for k, shape in schema.items():
        stored = (shape[0], int(np.prod(shape[1:]))) if len(shape) == 4 else shape
        out[k] = ServerArray(ee.Array(ee.List.repeat(0, int(np.prod(shape)))).reshape(list(stored)).float(), stored)
    return out
parameters = SwinTParameters({g: arrays(p) for g, p in enc.items()}, arrays(pe), arrays(dec), "0" * 64)
grid = Grid("EPSG:3857", (20, 0, 1000, 0, -20, 2000), 224, 224)
raster = lambda v: Raster(ee.Image.constant(v).rename(["a", "b"]), ("a", "b"), grid)
adapter = SwinT224Adapter(input_bands=("a", "b"))
a = adapter.skips(raster([0, 0]), validity=raster([1, 0]), parameters=parameters)
b = adapter.c4(Raster(ee.Image("projects/x/assets/A"), a.bands, a.grid), input_grid=grid, parameters=parameters)
c = adapter.logits(Raster(ee.Image("projects/x/assets/A"), a.bands, a.grid),
                   Raster(ee.Image("projects/x/assets/B"), b.bands, b.grid), input_grid=grid, parameters=parameters)
assert all(len(r.image.serialize()) > 100_000 for r in (a, b, c))
research = sorted(m for m in sys.modules if m.split(".")[0].startswith(("ee_attention", "swin_", "upernet")))
assert not research, research
from tensor_ee.models.swin_t224 import decoder as D, encoder as E
assert not any(hasattr(m, r) for m, r in ((E, "EXTENSION_HALOS"), (D, "PSP_POOLS"),
                                          (D, "FUSION_FORMULATIONS"), (D, "HEAD_RESIZES")))
print("ok")
'''


def test_fresh_interpreter_builds_the_swin_stages_and_loads_no_research_module():
    result = subprocess.run([sys.executable, "-c", FRESH], cwd=ROOT, check=True, capture_output=True, text=True,
                            env={**os.environ, "PYTHONPATH": str(ROOT)}, timeout=300)
    assert result.stdout.strip() == "ok"


# ---------------------------------------------------------------------------
# Fixtures package (D3) and the adapter invariants
# ---------------------------------------------------------------------------
FIXTURES = ("SmallCNN", "ConvBlock", "TinyUNet", "PoolRoundTrip", "initialize_fixture", "parameter_count")


def test_fixture_names_import_from_tensor_ee_models_as_the_same_objects():
    import tensor_ee.models as models
    from tensor_ee.models import (ConvBlock, PoolRoundTrip, SmallCNN, TinyUNet,  # noqa: F401
                                  initialize_fixture, parameter_count)
    defined = {n for n, v in vars(FX).items() if (inspect.isclass(v) or inspect.isfunction(v))
               and v.__module__ == FX.__name__}
    assert defined == set(FIXTURES)
    for name in FIXTURES:
        assert getattr(models, name) is getattr(FX, name) is locals()[name]
    assert (ROOT / "tensor_ee/models/__init__.py").is_file() and not (ROOT / "tensor_ee/models.py").exists()
    assert not any(n.startswith("tensor_ee.models.swin_t224") for n in _imports(ROOT / "tensor_ee/models/__init__.py"))


def test_the_adapter_keeps_its_literal_model_grid_and_checks_it(offline_ee):
    source = inspect.getsource(S._model_grid)
    assert "check_model_grid(TokenGrid(224,224,2.5))" in source
    literal = TokenGrid(224, 224, 2.5)
    assert check_model_grid(literal) is literal
    grid = S._model_grid()
    assert grid == literal and (type(grid.height), type(grid.scale)) == (int, float)
    assert grid.projection(ee).serialize() == TokenGrid(224, 224, 2.5).projection(ee).serialize()
    for k, stride in enumerate(C.STRIDES, 1):
        assert S._model_grid(k) == literal.coarse(stride)
    # The caller's bands are renamed to the patch embedding's defaults; bands= is not passed.
    skips = inspect.getsource(S.SwinT224Adapter.skips)
    assert skips.count(".rename(list(PE.BANDS))") == 2 and "bands=" not in skips
    assert S.PE.BANDS == ("VV", "VH") and S.PE.__name__ == "tensor_ee.blocks.window.patch"
    assert [n for n in vars(S.SwinT224Adapter) if not n.startswith("_")] == ["skips", "c4", "logits"]


def test_swin_facade_names_are_the_same_objects():
    import tensor_ee.swin_workflow as W
    assert S.__all__ == ["ServerArray", "SwinTParameters", "load_swin_t224_parameters", "load_swin_t224_model",
                         "SwinT224Adapter"]
    assert S.ServerArray is SP.ServerArray
    assert S.SwinTParameters is SP.SwinTParameters and S.load_swin_t224_parameters is SP.load_swin_t224_parameters
    assert S.load_swin_t224_model is SP.load_swin_t224_model
    assert W.SwinT224Adapter is S.SwinT224Adapter and W.load_swin_t224_parameters is SP.load_swin_t224_parameters
    assert S._SKIP_BANDS[-1] == "valid_fraction" and len(S._SKIP_BANDS) == 96 + 192 + 384 + 1
    assert S._C4_BANDS == tuple(f"c{c}" for c in range(768))
