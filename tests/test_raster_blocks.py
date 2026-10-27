"""L03: the raster-layout (R) blocks live in tensor_ee/blocks/raster/; the old
compiler, semantics and outputs names re-export them; tensor_ee.converter
re-exports the converter entry points.

Offline only: graphs are built against ``ee.apitestcase.ApiTestCase`` signatures
and compared by their serialization. No Earth Engine initialization or request.
"""
from __future__ import annotations

import ast
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
import pytest
import torch
from torch import nn

from recorded_paths import current
import tensor_ee
import tensor_ee.converter
from tensor_ee import compiler, primitives
from tensor_ee.blocks.raster import activations as A, conv as CV, merge as MG, outputs as OUT
from tensor_ee.blocks.raster import pooling as PL, resample as RS
from tensor_ee.core import metadata as M, precision as P
from tensor_ee.core.errors import UnsupportedOperation
from tensor_ee.layouts import raster as LR
from tensor_ee.weights import constants as K

ROOT = Path(__file__).resolve().parents[1]
RASTER = "tensor_ee.blocks.raster"
BLOCK_MODULES = {"conv": CV, "pooling": PL, "resample": RS, "merge": MG, "activations": A, "outputs": OUT}

# (old module, old name, library module, library name): the 17 moved L03 map
# entries (16 functions; pool2d has two branches) and the helpers that moved with
# them. Every old name is a re-export of the one library object.
MOVED = [
    # Convolution (blocks/raster/conv.py)
    ("tensor_ee.compiler", "conv2d", f"{RASTER}.conv", "conv2d"),
    ("tensor_ee.compiler", "_pair", f"{RASTER}.conv", "_pair"),
    ("tensor_ee.compiler", "_names", f"{RASTER}.conv", "_names"),
    ("tensor_ee.semantics", "_pair", f"{RASTER}.conv", "_pair"),
    ("tensor_ee.semantics", "conv_options", f"{RASTER}.conv", "conv_options"),
    ("tensor_ee.semantics", "_coefficient_field", f"{RASTER}.conv", "_coefficient_field"),
    ("tensor_ee.semantics", "finite_coefficients", f"{RASTER}.conv", "finite_coefficients"),
    # Their errors (lead decision Q1): core/errors.py, one class and one helper.
    ("tensor_ee.semantics", "SemanticUnknown", "tensor_ee.core.errors", "SemanticUnknown"),
    ("tensor_ee.compatibility", "SemanticUnknown", "tensor_ee.core.errors", "SemanticUnknown"),
    (f"{RASTER}.conv", "SemanticUnknown", "tensor_ee.core.errors", "SemanticUnknown"),
    ("tensor_ee.semantics", "_reject", "tensor_ee.core.errors", "reject"),
    (f"{RASTER}.conv", "_reject", "tensor_ee.core.errors", "reject"),
    (f"{RASTER}.pooling", "_reject", "tensor_ee.core.errors", "reject"),
    (f"{RASTER}.resample", "_reject", "tensor_ee.core.errors", "reject"),
    # Pooling and resampling (blocks/raster/pooling.py, resample.py)
    ("tensor_ee.compiler", "pool2d", f"{RASTER}.pooling", "pool2d"),
    ("tensor_ee.semantics", "pool_options", f"{RASTER}.pooling", "pool_options"),
    ("tensor_ee.compiler", "nearest_upsample", f"{RASTER}.resample", "nearest_upsample"),
    ("tensor_ee.semantics", "nearest_factor", f"{RASTER}.resample", "nearest_factor"),
    # Merges (blocks/raster/merge.py)
    ("tensor_ee.compiler", "concatenate", f"{RASTER}.merge", "concatenate"),
    ("tensor_ee.compiler", "residual_add", f"{RASTER}.merge", "residual_add"),
    # Activations (blocks/raster/activations.py)
    ("tensor_ee.compiler", "_relu", f"{RASTER}.activations", "relu"),
    ("tensor_ee.compiler", "_sigmoid", f"{RASTER}.activations", "sigmoid"),
    ("tensor_ee.compiler", "_leaky_relu", f"{RASTER}.activations", "leaky_relu"),
    ("tensor_ee.compiler", "_silu", f"{RASTER}.activations", "silu"),
    ("tensor_ee.compiler", "_gelu", f"{RASTER}.activations", "gelu"),
    ("tensor_ee.compiler", "_tanh", f"{RASTER}.activations", "tanh"),
    ("tensor_ee.compiler", "_eval_dropout", f"{RASTER}.activations", "eval_dropout"),
    ("tensor_ee.compiler", "_common_score_mask", f"{RASTER}.activations", "common_score_mask"),
    ("tensor_ee.compiler", "_softmax", f"{RASTER}.activations", "softmax"),
    ("tensor_ee.compiler", "_real_option", f"{RASTER}.activations", "_real_option"),
    ("tensor_ee.compiler", "_FLOAT32_MAX", f"{RASTER}.activations", "_FLOAT32_MAX"),
    ("tensor_ee.compiler", "_exact_gelu", f"{RASTER}.activations", "_exact_gelu"),
    ("tensor_ee.compiler", "_dropout_probability", f"{RASTER}.activations", "_dropout_probability"),
    # Outputs (blocks/raster/outputs.py)
    ("tensor_ee.outputs", "SegmentationOutputs", f"{RASTER}.outputs", "SegmentationOutputs"),
    ("tensor_ee.outputs", "segmentation_outputs", f"{RASTER}.outputs", "segmentation_outputs"),
    ("tensor_ee.outputs", "binary_threshold", f"{RASTER}.outputs", "binary_threshold"),
    ("tensor_ee.outputs", "_threshold_option", f"{RASTER}.outputs", "_threshold_option"),
    ("tensor_ee", "SegmentationOutputs", f"{RASTER}.outputs", "SegmentationOutputs"),
    ("tensor_ee", "segmentation_outputs", f"{RASTER}.outputs", "segmentation_outputs"),
    ("tensor_ee", "binary_threshold", f"{RASTER}.outputs", "binary_threshold"),
]

# The public R blocks: the names of the moved map targets.
BLOCKS = {"conv2d": "conv", "pool2d": "pooling", "nearest_upsample": "resample",
          "concatenate": "merge", "residual_add": "merge",
          "relu": "activations", "sigmoid": "activations", "leaky_relu": "activations",
          "silu": "activations", "gelu": "activations", "tanh": "activations",
          "relu6": "activations", "hardsigmoid": "activations", "hardswish": "activations",
          "eval_dropout": "activations", "common_score_mask": "activations", "softmax": "activations",
          "segmentation_outputs": "outputs", "binary_threshold": "outputs"}

CONVERTER = {"compile_model": "tensor_ee.compiler", "batchnorm_predecessor": "tensor_ee.semantics",
             "PixelWindow": "tensor_ee.tiling", "plan_tiles": "tensor_ee.tiling",
             "compile_tiled_model": "tensor_ee.tiling",
             "analyze_receptive_field": "tensor_ee.receptive_field",
             "effective_convolutions": "tensor_ee.weight_assets",
             "MatchedWeightProvider": "tensor_ee.weight_mapping",
             "load_onnx_model": "tensor_ee.onnx_import",
             "check_compatibility": "tensor_ee.compatibility"}


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


def _graph(value) -> str | tuple:
    if isinstance(value, OUT.SegmentationOutputs):
        return tuple(_graph(part) for part in (value.logits, value.probabilities, value.classes))
    if isinstance(value, tensor_ee.Raster):
        return (value.image.serialize(), value.bands, value.grid)
    return value.serialize()


# ---------------------------------------------------------------------------
# One implementation: every old name is the library object
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("old_module, old_name, new_module, new_name", MOVED,
                         ids=[f"{m}.{n}" for m, n, _, _ in MOVED])
def test_old_name_is_the_library_object(old_module, old_name, new_module, new_name):
    old, new = _attr(old_module, old_name), _attr(new_module, new_name)
    assert old is new
    if inspect.isfunction(new) or inspect.isclass(new):
        # Defined once, in the library: there is no second implementation.
        assert new.__module__ == new_module


def test_every_map_entry_owned_by_l03_is_at_its_target():
    module_map = json.loads((ROOT / "docs/evidence/l00-module-map.json").read_text())
    entries = [e for e in module_map["entries"] if e["owner"] == "L03"]
    assert len(entries) == 26
    moved = [e for e in entries if e["action"] == "move"]
    stays = [e for e in entries if e["action"] != "move"]
    assert len(moved) == 17 and len(stays) == 9
    for entry in moved:
        path, name = entry["target"].split("::")
        path = current(path)                         # the record predates the rename
        assert path.startswith("tensor_ee/blocks/raster/"), entry["id"]
        target = _attr(path[:-3].replace("/", "."), name)
        old_path, old_name = entry["id"].split("#")[0].split(":")
        old_path = current(old_path)
        assert _attr(old_path[:-3].replace("/", "."), old_name) is target, entry["id"]
        assert BLOCKS[name] == path.split("/")[-1][:-3]
    # The moved names are the L03 blocks; the bounded activations came later, are not moves,
    # and are the only blocks the map does not list.
    assert {e["target"].split("::")[1] for e in moved} == set(BLOCKS) - {"relu6", "hardsigmoid", "hardswish"}
    for entry in stays:
        path, name = entry["target"].split("::")
        assert entry["id"].split(":")[0] == path   # the converter stays where it is
        target = _attr(current(path)[:-3].replace("/", "."), name)
        assert getattr(tensor_ee.converter, name) is target, entry["id"]


def test_package_reexports_its_blocks():
    import tensor_ee.blocks.raster as R
    for name in R.__all__:
        module = BLOCKS.get(name, "outputs")
        assert getattr(R, name) is getattr(BLOCK_MODULES[module], name)
    assert set(BLOCKS) <= set(R.__all__)


def test_public_api_names_resolve_to_the_same_objects():
    assert len(tensor_ee.__all__) == 31
    for name in tensor_ee.__all__:
        assert getattr(tensor_ee, name) is not None
    assert tensor_ee.compile_model is compiler.compile_model
    assert tensor_ee.segmentation_outputs is OUT.segmentation_outputs
    assert tensor_ee.binary_threshold is OUT.binary_threshold
    for name in ("conv2d", "pool2d", "nearest_upsample", "concatenate", "residual_add", "relu", "sigmoid",
                 "leaky_relu", "silu", "gelu", "tanh", "eval_dropout", "softmax", "from_image"):
        assert getattr(tensor_ee, name) is getattr(primitives, name)


# Parameters at the base revision (9336ad7): every signature is unchanged apart
# from a trailing keyword-only policy= defaulting to float32_operands.
POSITIONAL, KEYWORD = inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY
_EMPTY = inspect.Parameter.empty
BASE_PARAMETERS = {
    "conv2d": [("raster", POSITIONAL, _EMPTY), ("module", POSITIONAL, _EMPTY), ("parameters", POSITIONAL, None),
               ("execution_dtype", KEYWORD, "float32"), ("clip_footprints", KEYWORD, False)],
    "pool2d": [("raster", POSITIONAL, _EMPTY), ("module", POSITIONAL, _EMPTY), ("clip_footprints", KEYWORD, False)],
    "nearest_upsample": [("raster", POSITIONAL, _EMPTY), ("scale_factor", POSITIONAL, _EMPTY),
                         ("clip_footprints", KEYWORD, False)],
    "concatenate": [("rasters", POSITIONAL, _EMPTY), ("dim", POSITIONAL, 1), ("clip_footprints", KEYWORD, False)],
    "residual_add": [("left", POSITIONAL, _EMPTY), ("right", POSITIONAL, _EMPTY), ("clip_footprints", KEYWORD, False)],
    "relu": [("raster", POSITIONAL, _EMPTY), ("clip_footprints", KEYWORD, False)],
    "sigmoid": [("raster", POSITIONAL, _EMPTY), ("clip_footprints", KEYWORD, False)],
    "leaky_relu": [("raster", POSITIONAL, _EMPTY), ("negative_slope", POSITIONAL, 0.01),
                   ("clip_footprints", KEYWORD, False)],
    "silu": [("raster", POSITIONAL, _EMPTY), ("clip_footprints", KEYWORD, False)],
    "gelu": [("raster", POSITIONAL, _EMPTY), ("approximate", KEYWORD, "none"), ("clip_footprints", KEYWORD, False)],
    "tanh": [("raster", POSITIONAL, _EMPTY), ("clip_footprints", KEYWORD, False)],
    "relu6": [("raster", POSITIONAL, _EMPTY), ("clip_footprints", KEYWORD, False)],
    "hardsigmoid": [("raster", POSITIONAL, _EMPTY), ("clip_footprints", KEYWORD, False)],
    "hardswish": [("raster", POSITIONAL, _EMPTY), ("clip_footprints", KEYWORD, False)],
    "eval_dropout": [("raster", POSITIONAL, _EMPTY), ("p", KEYWORD, 0.5), ("clip_footprints", KEYWORD, False)],
    "common_score_mask": [("image", POSITIONAL, _EMPTY), ("grid", POSITIONAL, _EMPTY)],
    "softmax": [("raster", POSITIONAL, _EMPTY), ("clip_footprints", KEYWORD, False)],
    "segmentation_outputs": [("logits", POSITIONAL, _EMPTY), ("logit_bands", KEYWORD, None),
                             ("probability_bands", KEYWORD, None), ("class_band", KEYWORD, "class_index"),
                             ("validity", KEYWORD, None)],
    "binary_threshold": [("probabilities", POSITIONAL, _EMPTY), ("positive_class", KEYWORD, 1),
                         ("threshold", KEYWORD, 0.5), ("output_band", KEYWORD, "class_mask"),
                         ("validity", KEYWORD, None)],
}
FACADE_PARAMETERS = {  # the public primitives at the base revision, where they differ from the block
    "conv2d": [("raster", POSITIONAL, _EMPTY), ("module", POSITIONAL, _EMPTY), ("parameters", POSITIONAL, None),
               ("clip_footprints", KEYWORD, False)],
    "concatenate": [("rasters", POSITIONAL, _EMPTY), ("clip_footprints", KEYWORD, False)],
}
FACADES = ("conv2d", "pool2d", "nearest_upsample", "concatenate", "residual_add", "relu", "sigmoid",
           "leaky_relu", "silu", "gelu", "tanh", "eval_dropout", "softmax")


def _parameters(function):
    return [(p.name, p.kind, p.default) for p in inspect.signature(function).parameters.values()]


@pytest.mark.parametrize("name", sorted(BLOCKS))
def test_block_signature_adds_only_a_keyword_only_policy(name):
    block = getattr(BLOCK_MODULES[BLOCKS[name]], name)
    assert _parameters(block) == BASE_PARAMETERS[name] + [("policy", KEYWORD, P.FLOAT32_OPERANDS)]


@pytest.mark.parametrize("name", FACADES)
def test_facade_signature_adds_only_a_keyword_only_policy(name):
    expected = FACADE_PARAMETERS.get(name, BASE_PARAMETERS[name])
    assert _parameters(getattr(primitives, name)) == expected + [("policy", KEYWORD, P.FLOAT32_OPERANDS)]


# ---------------------------------------------------------------------------
# Policy: the default serializes as omitting it; other policies are refused
# ---------------------------------------------------------------------------
def _fixture():
    grid = tensor_ee.Grid("EPSG:3857", (10, 0, 100, 0, -10, 200), 4, 4)
    source = ee.Image.constant([1, -2, 3]).rename(["a", "b", "c"])
    three = primitives.from_image(source, ("a", "b", "c"), grid)
    two = primitives.from_image(source, ("a", "b"), grid)
    conv = nn.Conv2d(3, 2, 3, padding=1, stride=2).eval()
    with torch.no_grad():
        conv.weight.copy_(torch.arange(54, dtype=torch.float32).reshape(2, 3, 3, 3) / 64)
        conv.bias.copy_(torch.tensor([0.5, -0.25]))
    return grid, three, two, conv


CALLS = {   # block name -> call(function, fixture, **policy)
    "conv2d": lambda f, x, **kw: f(x[1], x[3], **kw),
    "pool2d": lambda f, x, **kw: (f(x[1], nn.AvgPool2d(2), **kw), f(x[1], nn.MaxPool2d(3, 2, 1), **kw)),
    "nearest_upsample": lambda f, x, **kw: f(x[1], 2, **kw),
    "concatenate": lambda f, x, **kw: f([x[1], x[2]], **kw),
    "residual_add": lambda f, x, **kw: f(x[1], x[1], **kw),
    "relu": lambda f, x, **kw: f(x[1], **kw),
    "sigmoid": lambda f, x, **kw: f(x[1], **kw),
    "leaky_relu": lambda f, x, **kw: f(x[1], 0.2, **kw),
    "silu": lambda f, x, **kw: f(x[1], **kw),
    "gelu": lambda f, x, **kw: f(x[1], **kw),
    "tanh": lambda f, x, **kw: f(x[1], **kw),
    "relu6": lambda f, x, **kw: f(x[1], **kw),
    "hardsigmoid": lambda f, x, **kw: f(x[1], **kw),
    "hardswish": lambda f, x, **kw: f(x[1], **kw),
    "eval_dropout": lambda f, x, **kw: f(x[1], p=0.25, **kw),
    "common_score_mask": lambda f, x, **kw: f(x[1].image, x[0], **kw),
    "softmax": lambda f, x, **kw: f(x[1], **kw),
    "segmentation_outputs": lambda f, x, **kw: f(x[1], **kw),
    "binary_threshold": lambda f, x, **kw: f(x[2], threshold=0.50000001, **kw),
}


def _serialized(value):
    return tuple(_graph(v) for v in value) if isinstance(value, tuple) else _graph(value)


@pytest.mark.parametrize("name", sorted(BLOCKS))
def test_block_policy_default_serializes_as_omitting_it_and_others_are_refused(offline_ee, name):
    block, x = getattr(BLOCK_MODULES[BLOCKS[name]], name), _fixture()
    call = CALLS[name]
    assert _serialized(call(block, x)) == _serialized(call(block, x, policy=P.FLOAT32_OPERANDS))
    for other in (P.ROUND_EACH_OP, P.DOUBLE_ROUND_ONCE, P.GROUPED_DOUBLE):
        with pytest.raises(UnsupportedOperation, match="not implemented"):
            call(block, x, policy=other)
    for bad in ("none", "composite", "float32", None):
        with pytest.raises(ValueError, match="unknown precision policy"):
            call(block, x, policy=bad)


@pytest.mark.parametrize("name", FACADES)
def test_facade_is_the_block_plus_float32_materialization(offline_ee, name):
    # The public primitive calls the library block and adds only .toFloat().
    block, facade, x = getattr(BLOCK_MODULES[BLOCKS[name]], name), getattr(primitives, name), _fixture()
    call = CALLS[name]
    expected = call(block, x)
    expected = tuple(LR._float32(r) for r in expected) if isinstance(expected, tuple) else LR._float32(expected)
    assert _serialized(call(facade, x)) == _serialized(expected)
    assert _serialized(call(facade, x)) == _serialized(call(facade, x, policy=P.FLOAT32_OPERANDS))
    with pytest.raises(UnsupportedOperation, match="not implemented"):
        call(facade, x, policy=P.DOUBLE_ROUND_ONCE)


def test_compile_model_keeps_the_per_node_float32_cast(offline_ee):
    grid, three, _, conv = _fixture()
    source = ee.Image.constant([1, -2, 3]).rename(["a", "b", "c"])
    compiled = tensor_ee.compile_model(nn.Sequential(conv, nn.LeakyReLU(0.2)).eval(), source, ("a", "b", "c"), grid)
    first = compiler._finish(source.select(["a", "b", "c"]).unmask(0, sameFootprint=False), ("c0", "c1", "c2"), grid)
    first = tensor_ee.Raster(first.image.toFloat(), first.bands, first.grid)
    second = CV.conv2d(first, conv)
    second = tensor_ee.Raster(second.image.toFloat(), second.bands, second.grid)
    third = A.leaky_relu(second, 0.2)
    assert compiled.image.serialize() == third.image.toFloat().serialize()


# ---------------------------------------------------------------------------
# Metadata: every public R block, locality from the inventory and receptive field
# ---------------------------------------------------------------------------
EXPECTED = {   # block -> (locality, radius, status)
    "conv2d": (M.Locality.KERNEL, 3, "Ready"),
    # P02: radius 3, the aligned 4 x 4 pool's offset k - 1 (tests/test_p02_raster_settings.py).
    "pool2d": (M.Locality.KERNEL, 3, "Ready — aligned avg/max k = 2, 3, 4; max 3×3 stride 1 or 2, padding 1; "
                                     "avg 3×3 stride 1 padding 1, both divisor rules"),
    "nearest_upsample": (M.Locality.POINTWISE, None, "Ready — factors 2, 3, 4"),
    **{name: (M.Locality.POINTWISE, None, "Ready")
       for name in ("concatenate", "residual_add", "relu", "sigmoid", "leaky_relu", "silu", "gelu", "tanh",
                    "relu6", "hardsigmoid", "hardswish",
                    "eval_dropout", "common_score_mask", "softmax", "segmentation_outputs", "binary_threshold")},
}


def test_every_public_block_has_valid_metadata_in_its_module():
    seen = {}
    for module_name, module in BLOCK_MODULES.items():
        assert M.validate_metadata(module.METADATA) is module.METADATA
        for name in module.METADATA:
            assert name not in seen
            seen[name] = module_name
            assert callable(getattr(module, name))
    assert seen == BLOCKS
    for name, (locality, radius, status) in EXPECTED.items():
        entry = getattr(BLOCK_MODULES[BLOCKS[name]], "METADATA")[name]
        assert (entry.layout_in, entry.layout_out, entry.policy) == ("R", "R", P.FLOAT32_OPERANDS)
        assert (entry.locality, entry.radius, entry.status) == (locality, radius, status), name
        assert "Fixed," not in entry.status and not entry.chip_edge_dependent
        assert entry.evidence and all((ROOT / path).is_file() for path in entry.evidence), name


def _inventory_locality(text: str) -> M.Locality:
    if text == "pointwise":
        return M.Locality.POINTWISE
    assert text.startswith("bounded k×k"), text
    return M.Locality.KERNEL


def test_metadata_locality_matches_the_inventory():
    module_map = json.loads((ROOT / "docs/evidence/l00-module-map.json").read_text())
    inventory = json.loads((ROOT / "docs/evidence/building-blocks-inventory.json").read_text())["blocks"]
    for entry in module_map["entries"]:
        if entry["owner"] != "L03" or entry["action"] != "move":
            continue
        record = inventory[entry["index"]]
        assert record["id"] == entry["id"]
        name = entry["target"].split("::")[1]
        metadata = getattr(BLOCK_MODULES[BLOCKS[name]], "METADATA")[name]
        assert metadata.locality == _inventory_locality(record["locality"]), entry["id"]


class _Cat(nn.Module):
    def forward(self, x):
        return torch.cat([x, x], 1)


class _Add(nn.Module):
    def forward(self, x):
        return x + x


def test_metadata_radius_follows_the_receptive_field_rules():
    from tensor_ee.receptive_field import analyze_receptive_field
    # Convolution: radius k // 2; the field records the largest validated kernel (7).
    for k in (1, 3, 7):
        report = analyze_receptive_field(nn.Conv2d(1, 1, k, padding=k // 2))
        assert report.x.left_radius == report.x.right_radius == k // 2 <= CV.METADATA["conv2d"].radius
    assert CV.METADATA["conv2d"].radius == 7 // 2
    # Both pool branches read at most 2 * radius + 1 input pixels per axis.
    for module in (nn.AvgPool2d(2), nn.MaxPool2d(2), nn.MaxPool2d(3, 2, 1)):
        support = analyze_receptive_field(module)
        assert max(support.x.maximum_support_width, support.y.maximum_support_width) <= 2 * PL.METADATA["pool2d"].radius + 1
    # Pointwise blocks read one input pixel.
    for module in (nn.Upsample(scale_factor=2), _Cat(), _Add(), nn.ReLU(), nn.Sigmoid(), nn.LeakyReLU(),
                   nn.SiLU(), nn.GELU(), nn.Tanh(), nn.Dropout(), nn.Softmax(dim=1)):
        support = analyze_receptive_field(module.eval())
        assert support.x.maximum_support_width == support.y.maximum_support_width == 1, module


def test_metadata_costs():
    conv = nn.Conv2d(3, 2, 3, padding=1)
    entry = CV.METADATA["conv2d"]
    assert entry.embedded_constants(conv) == 2 * 3 * 3 * 3 + 2 and entry.asset_arrays(conv) == 2
    unbiased = nn.Conv2d(3, 2, 1, bias=False)
    assert entry.embedded_constants(unbiased) == 6 and entry.asset_arrays(unbiased) == 1
    for name, module in BLOCKS.items():
        if name != "conv2d":
            metadata = getattr(BLOCK_MODULES[module], "METADATA")[name]
            assert metadata.embedded_constants() == metadata.asset_arrays() == 0


def test_conv2d_embedded_constant_count_matches_its_inline_graph(offline_ee):
    grid, three, _, conv = _fixture()
    values = []
    with patch(f"{RASTER}.conv._inline_float32_array",
               side_effect=lambda rows: values.extend(v for row in rows for v in row) or K.inline_float32_array(rows)):
        CV.conv2d(three, conv)
    assert len(values) == CV.METADATA["conv2d"].embedded_constants(conv)


# ---------------------------------------------------------------------------
# Patched names: the retargeted patches reach the blocks' own lookups
# ---------------------------------------------------------------------------
FINISH_USERS = ("relu", "sigmoid", "leaky_relu", "silu", "gelu", "tanh", "relu6", "hardsigmoid",
                "hardswish", "eval_dropout", "softmax")


def test_blocks_look_up_their_patched_bindings_in_their_own_modules():
    # tests/test_pointwise_operators.py patches activations._finish/_ee_scalar and
    # tests/test_inline_coefficients.py patches conv._inline_float32_array.
    assert A._finish is LR.finish and A._ee_scalar is P.exact_scalar
    assert CV._inline_float32_array is K.inline_float32_array
    for name in FINISH_USERS:
        function = getattr(A, name)
        assert function.__globals__ is vars(A) and "_finish" in function.__code__.co_names, name
    for name in ("leaky_relu", "common_score_mask", "relu6", "hardsigmoid", "hardswish"):
        function = getattr(A, name)
        assert function.__globals__ is vars(A) and "_ee_scalar" in function.__code__.co_names, name
    assert CV.conv2d.__globals__ is vars(CV) and "_inline_float32_array" in CV.conv2d.__code__.co_names
    # The old compiler bindings remain (old names), as the library objects.
    assert compiler._finish is LR.finish and compiler._ee_scalar is P.exact_scalar
    assert compiler._inline_float32_array is K.inline_float32_array


class _Reached(Exception):
    pass


def _raise_reached(*args, **kwargs):
    raise _Reached()


def _reached(monkeypatch, module, attribute) -> set:
    x = _fixture()
    monkeypatch.setattr(module, attribute, _raise_reached)
    reached = set()
    for name, call in CALLS.items():
        try:
            call(getattr(BLOCK_MODULES[BLOCKS[name]], name), x)
        except _Reached:
            reached.add(name)
    monkeypatch.undo()
    return reached


def test_what_each_retargeted_patch_reaches(offline_ee, monkeypatch):
    # _finish: the activations (and segmentation_outputs, through softmax). The
    # merges, pooling and resampling bind _finish in their own modules.
    assert _reached(monkeypatch, A, "_finish") == {*FINISH_USERS, "segmentation_outputs"}
    # _ee_scalar: leaky_relu's slope, the bounded activations' bounds and the common score
    # mask, so also softmax and both outputs (their masks). binary_threshold's own threshold
    # scalar is bound in the outputs module, as it was at the base revision.
    assert _reached(monkeypatch, A, "_ee_scalar") == {"leaky_relu", "common_score_mask", "softmax",
                                                      "relu6", "hardsigmoid", "hardswish",
                                                      "segmentation_outputs", "binary_threshold"}
    assert _reached(monkeypatch, CV, "_inline_float32_array") == {"conv2d"}


def test_patching_conv_inline_array_intercepts_compile_model(offline_ee):
    model = nn.Conv2d(1, 1, 1).eval()
    with torch.no_grad():
        model.weight.fill_(0.5)
        model.bias.fill_(0.25)
    grid = tensor_ee.Grid("EPSG:3857", (10, 0, 100, 0, -10, 200), 1, 1)
    source = ee.Image.constant(0).rename("a")
    with patch(f"{RASTER}.conv._inline_float32_array", side_effect=RuntimeError("intercepted")):
        with pytest.raises(RuntimeError, match="intercepted"):
            tensor_ee.compile_model(model, source, ("a",), grid)


def test_compile_model_resolves_symbolic_trace_from_the_compiler():
    # tests/test_compatibility.py patches tensor_ee.compiler.symbolic_trace.
    assert compiler.compile_model.__globals__ is vars(compiler)
    assert "symbolic_trace" in compiler.compile_model.__code__.co_names


# ---------------------------------------------------------------------------
# Layer rule: blocks/raster imports only core, layouts and weights
# ---------------------------------------------------------------------------
THIRD_PARTY = ("ee", "numpy", "torch")   # the R blocks take PyTorch modules and options
LAYERS = ("tensor_ee.core", "tensor_ee.layouts", "tensor_ee.weights", "tensor_ee.blocks.raster")
FORBIDDEN = ("ee_attention", "swin_", "upernet", "tensor_ee.converter", "tensor_ee.compiler",
             "tensor_ee.semantics", "tensor_ee.tiling", "tensor_ee.receptive_field", "tensor_ee.onnx_import",
             "tensor_ee.compatibility", "tensor_ee.weight_assets", "tensor_ee.weight_mapping",
             "tensor_ee.primitives", "tensor_ee.outputs", "tensor_ee.comparison", "tensor_ee.swin",
             "tensor_ee._swin_parameters", "tensor_ee.models", "tensor_ee.resnet_models",
             "tensor_ee.segwater", "tensor_ee.examples", "tensor_ee.blocks.window")


def _imports(path: Path):
    """Every import in the file, at module level, in functions or guarded,
    resolved to an absolute module name."""
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


def _within(name: str, prefixes) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)


def test_raster_blocks_respect_the_layer_rule():
    files = sorted((ROOT / "tensor_ee/blocks/raster").glob("*.py"))
    assert {p.name for p in files} == {"__init__.py", "conv.py", "pooling.py", "resample.py", "merge.py",
                                       "activations.py", "outputs.py"}
    violations = []
    for path in files:
        for name in _imports(path):
            top = name.split(".")[0]
            allowed = (top in sys.stdlib_module_names or top == "__future__" or _within(name, THIRD_PARTY)
                       or _within(name, LAYERS))
            if name.startswith(FORBIDDEN) or not allowed:
                violations.append((str(path.relative_to(ROOT)), name))
    assert violations == []


def test_the_converter_imports_the_blocks_not_the_reverse():
    for module in ("compiler", "semantics"):
        names = set(_imports(ROOT / f"tensor_ee/{module}.py"))
        assert any(_within(name, ("tensor_ee.blocks.raster",)) for name in names), module


def test_importing_the_raster_blocks_loads_no_research_module():
    code = ("import sys; import tensor_ee.blocks.raster, tensor_ee.converter\n"
            "bad = sorted(m for m in sys.modules if m.split('.')[0] in "
            "{'ee_attention', 'upernet'} or m.startswith('swin_'))\n"
            "assert not bad, bad")
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True,
                   env={**os.environ, "PYTHONPATH": str(ROOT)})


def test_blocks_parent_package_is_still_empty():
    body = ast.parse((ROOT / "tensor_ee/blocks/__init__.py").read_text()).body
    assert all(isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) for node in body)


# ---------------------------------------------------------------------------
# The converter namespace (decision D2)
# ---------------------------------------------------------------------------
def test_converter_namespace_reexports_the_converter_entry_points():
    # The ten Phase 1 names are still exported, each the same object as at its module. The
    # whole __all__, with Phase 3's names, is pinned in tests/test_x08_docs_and_api.py.
    import tensor_ee.converter as converter
    assert set(CONVERTER) <= set(converter.__all__)
    for name, module in CONVERTER.items():
        assert getattr(converter, name) is _attr(module, name), name
        assert getattr(converter, name).__module__ == module, name
