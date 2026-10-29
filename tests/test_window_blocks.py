"""L04: the window-layout (W) and value (V) blocks live in ``tensor_ee.blocks.window``.

The research modules that re-exported them were archived (L06b); settings are parameters
with today's values as defaults; constants are built for a layout and refused by another; every block
declares its precision policy and its locality and cost metadata.

Offline only: graphs are built against ``ee.apitestcase.ApiTestCase`` signatures and
compared by their serialization. No Earth Engine initialization or request.
"""
from __future__ import annotations

import ast
import importlib
import inspect
import os
from pathlib import Path
import re
import subprocess
import sys
import types

import ee
from ee.apitestcase import ApiTestCase
import numpy as np
import pytest

from recorded_paths import current
import tensor_ee
from tensor_ee.blocks.window import activations as WA
from tensor_ee.blocks.window import attention as WT
from tensor_ee.blocks.window import conv as WC
from tensor_ee.blocks.window import linear as WL
from tensor_ee.blocks.window import merge as WM
from tensor_ee.blocks.window import normalization as WN
from tensor_ee.blocks.window import patch as WP
from tensor_ee.blocks.window import pooling as WO
from tensor_ee.blocks.window import resample as WR
from tensor_ee.blocks.window import segformer as WS
from tensor_ee.blocks.window import swin as WSW
from tensor_ee.blocks.window import swin_upernet_config as WCF
from tensor_ee.blocks.window import upernet as WU
from tensor_ee.core import errors
from tensor_ee.core import metadata as M
from tensor_ee.core import precision as P
from tensor_ee.core.grid import TokenGrid
from tensor_ee.core.metadata import Locality
from tensor_ee.core.ops import ValueOps, WindowOps
from tensor_ee.core.tensor import Feature, WindowTensor
from tensor_ee.weights.constants import RowMismatch, ServerArray, ValueLayout, WindowLayout, row_constant

ROOT = Path(__file__).resolve().parents[1]
LIB = "tensor_ee.blocks.window"
MODULES = {"linear": WL, "activations": WA, "normalization": WN, "attention": WT, "conv": WC,
           "resample": WR, "pooling": WO, "patch": WP, "merge": WM, "segformer": WS, "swin": WSW, "upernet": WU}
SETTINGS_MODULES = {"swin_upernet_config": WCF}
"""Modules of the package that hold settings and no block, so no ``METADATA`` (X02b's census:
every module is in ``MODULES`` or here)."""

# ---------------------------------------------------------------------------
# The 41 L04 entries of the module map are at their library targets
# ---------------------------------------------------------------------------
def _attr(module: str, name: str):
    return getattr(importlib.import_module(module), name)


def test_the_41_entries_are_accounted_for():
    """Every L04 entry of the module map resolves at its library target and is defined there
    (the old research paths were archived at L06b)."""
    import json
    entries = [e for e in json.loads((ROOT / "docs/evidence/l00-module-map.json").read_text())["entries"]
               if e["owner"] == "L04"]
    assert len(entries) == 41
    for e in entries:
        module, name = e["target"].split("::")
        module = current(module)                     # the record predates the rename
        assert module.startswith("tensor_ee/blocks/window/")
        value = _attr(module[:-3].replace("/", "."), name)
        assert callable(value)
        if inspect.isfunction(value) or inspect.isclass(value):
            assert value.__module__ == module[:-3].replace("/", "."), e["id"]


def test_layer_norm2d_is_the_single_entry_point(offline_ee):
    """The decoder's feature norms (fpn_layer_norm, psp_layer_norm) call layer_norm2d with their
    prebuilt constants; the three serialize identically (the L00 layer_norm2d probes)."""
    from tensor_ee.models.swin_t224 import decoder as D
    for wrapper in (D.fpn_layer_norm, D.psp_layer_norm):
        assert "N.layer_norm2d(" in inspect.getsource(wrapper)
    ops, x = WindowOps(ee), ee.Image("projects/l04-test/assets/windows")
    gamma, beta = _w((96,), 1), _w((96,), 2)
    built = (ops.constant(gamma, repeat_rows=49), ops.constant(beta, repeat_rows=49))
    expected = WN.layer_norm2d(ops, x, gamma, beta, dim=96, rows=49).serialize()
    # The wrappers read the width and rows from the constants' configuration (C0).
    assert D.fpn_layer_norm(ee, x, types.SimpleNamespace(norm={1: built}, config=D.SWIN_T224), 1).serialize() == expected
    assert WN.layer_norm2d(ops, x, *built, dim=96).serialize() == expected
    assert WN.layer_norm(ops, x, *built, dim=96, eps=1e-6).serialize() == expected
    gamma4, beta4 = _w((768,), 3), _w((768,), 4)
    built4 = (ops.constant(gamma4, repeat_rows=49), ops.constant(beta4, repeat_rows=49))
    assert (D.psp_layer_norm(ee, x, types.SimpleNamespace(norm=built4, config=D.SWIN_T224)).serialize()
            == WN.layer_norm2d(ops, x, gamma4, beta4, dim=768, rows=WindowLayout()).serialize())


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


def _w(shape, seed):
    """Integer-valued float32 weights (no platform-dependent RNG)."""
    n = int(np.prod(shape))
    return (((np.arange(n) * (2 * seed + 7)) % 23 - 11) / 8).astype(np.float32).reshape(shape)


def _server(name, shape):
    return ServerArray(ee.Array(ee.FeatureCollection(f"projects/l04-assets/assets/{name}").first().get("v")),
                       tuple(shape))


def _serialize(value):
    if isinstance(value, dict):
        return {k: _serialize(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_serialize(v) for v in value]
    if isinstance(value, np.ndarray):
        return ("array", value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, (WindowTensor,)):
        return (_serialize(value.image), value.grid, value.channels, value.window)
    if hasattr(value, "pooled"):                           # pooling.Pooled
        return (_serialize(value.sums), _serialize(value.pooled), value.plan)
    if isinstance(value, ee.ComputedObject):
        return value.serialize()
    return value


def _block_parameters():
    p = {
        "norm1.weight": _w((96,), 8), "norm1.bias": _w((96,), 9),
        "attn.qkv.weight": _w((288, 96), 1), "attn.qkv.bias": _w((288,), 2),
        "attn.relative_position_bias_table": _w((169, 3), 6),
        "attn.proj.weight": _w((96, 96), 10), "attn.proj.bias": _w((96,), 11),
        "norm2.weight": _w((96,), 12), "norm2.bias": _w((96,), 13),
        "mlp.fc1.weight": _w((384, 96), 14), "mlp.fc1.bias": _w((384,), 15),
        "mlp.fc2.weight": _w((96, 384), 16), "mlp.fc2.bias": _w((96,), 17)}
    return p


PE = {"proj.weight": _w((96, 32), 21), "proj.bias": _w((96,), 22),
      "norm.weight": _w((96,), 23), "norm.bias": _w((96,), 24)}
PM = {"norm.weight": _w((384,), 25), "norm.bias": _w((384,), 26), "reduction.weight": _w((192, 384), 27)}


def _context():
    """Shared offline inputs (call inside offline_ee)."""
    from tensor_ee.models.swin_t224 import encoder as SB
    c = types.SimpleNamespace()
    c.W, c.V = WindowOps(ee), ValueOps(ee)
    c.img = ee.Image("projects/l04-test/assets/windows")
    c.img2 = ee.Image("projects/l04-test/assets/windows2")
    c.val = ee.Array(ee.FeatureCollection("projects/l04-test/assets/value").first().get("v"))
    c.tokens = TokenGrid(28, 28, 2.5)
    c.coarse = c.tokens.coarse(7)
    c.grid = TokenGrid(112, 112, 2.5)
    c.input = ee.Image("projects/l04-test/assets/input")
    c.lin = WL.LinearConstants(c.W, _w((288, 96), 1), _w((288,), 2), 49)
    c.bn = WN.BatchNormConstants(c.W, _w((96,), 3), _w((96,), 4), _w((96,), 5), np.abs(_w((96,), 6)) + 1, rows=49)
    c.c33 = WC.Conv3x3Constants(c.W, _w((8, 6, 3, 3), 18), _w((8,), 19))
    c.block = SB.block_constants(c.W, _block_parameters())
    c.part = Feature(c.img, c.tokens, 4)
    c.gc = WC.Conv3x3Constants(c.W, WC.group_weight(_w((8, 16, 3, 3), 20), 1))
    c.pe = WP.patch_embed_constants(c.W, PE)
    c.pm = WP.patch_merge_constants(c.W, PM)
    return c


# ---------------------------------------------------------------------------
# Explicit defaults serialize identically to omitting them; old call forms work
# ---------------------------------------------------------------------------
DEFAULT_CASES = {
    "linear": (lambda c: WL.linear(c.W, c.img, c.lin),
               lambda c: WL.linear(c.W, c.img, c.lin, policy="double_round_once", layout=WindowLayout(7))),
    "LinearConstants rows": (lambda c: WL.LinearConstants(c.W, _w((288, 96), 1), _w((288,), 2), 49).bias,
                             lambda c: WL.LinearConstants(c.W, _w((288, 96), 1), _w((288,), 2), WindowLayout()).bias),
    "LinearConstants tensor": (lambda c: WL.LinearConstants(c.W, _w((288, 96), 1), _w((288,), 2), 49).bias,
                               lambda c: WL.LinearConstants(c.W, _w((288, 96), 1), _w((288,), 2),
                                                            WindowTensor(c.img, c.tokens, 96)).bias),
    "conv1x1 V rows": (lambda c: WL.conv1x1(c.V, c.val, WL.conv1x1_constants(c.V, _w((256, 96, 1, 1), 3), _w((256,), 4), 9)),
                       lambda c: WL.conv1x1(c.V, c.val, WL.conv1x1_constants(c.V, _w((256, 96, 1, 1), 3), _w((256,), 4),
                                                                              ValueLayout(9)), layout=ValueLayout(9))),
    "gelu": (lambda c: WA.gelu(c.W, c.img), lambda c: WA.gelu(c.W, c.img, policy="double_round_once")),
    "layer_norm": (lambda c: WN.layer_norm(c.W, c.img, c.lin.bias, c.lin.bias, dim=96),
                   lambda c: WN.layer_norm(c.W, c.img, c.lin.bias, c.lin.bias, dim=96, eps=1e-5,
                                           policy="double_round_once")),
    "layer_norm RowConstant": (
        lambda c: WN.layer_norm(c.W, c.img, c.W.constant(_w((96,), 1), repeat_rows=49),
                                c.W.constant(_w((96,), 2), repeat_rows=49), dim=96),
        lambda c: WN.layer_norm(c.W, c.img, row_constant(c.W, _w((96,), 1), WindowLayout()),
                                row_constant(c.W, _w((96,), 2), WindowLayout()), dim=96, layout=WindowLayout())),
    "layer_norm2d dense": (lambda c: WN.layer_norm2d(c.W, c.img, _w((96,), 1), _w((96,), 2), dim=96),
                           lambda c: WN.layer_norm2d(c.W, c.img, _w((96,), 1), _w((96,), 2), dim=96, rows=1,
                                                     eps=1e-6, policy="double_round_once")),
    "layer_norm2d dense layout": (lambda c: WN.layer_norm2d(c.W, c.img, _w((96,), 1), _w((96,), 2), dim=96),
                                  lambda c: WN.layer_norm2d(c.W, c.img, _w((96,), 1), _w((96,), 2), dim=96,
                                                            rows=WindowLayout(1))),
    "layer_norm2d window": (lambda c: WN.layer_norm2d(c.W, c.img, _w((96,), 1), _w((96,), 2), dim=96, rows=49),
                            lambda c: WN.layer_norm2d(c.W, c.img, _w((96,), 1), _w((96,), 2), dim=96,
                                                      rows=WindowLayout(7))),
    "layer_norm2d prebuilt RowConstant": (
        lambda c: WN.layer_norm2d(c.W, c.img, _w((96,), 1), _w((96,), 2), dim=96, rows=49),
        lambda c: WN.layer_norm2d(c.W, c.img, row_constant(c.W, _w((96,), 1), WindowLayout()),
                                  row_constant(c.W, _w((96,), 2), WindowLayout()), dim=96, rows=WindowLayout())),
    "BatchNormConstants rows": (
        lambda c: WN.batch_norm(c.W, c.img, WN.BatchNormConstants(c.W, _w((96,), 3), _w((96,), 4), _w((96,), 5), _w((96,), 6) ** 2 + 1)),
        lambda c: WN.batch_norm(c.W, c.img, WN.BatchNormConstants(c.W, _w((96,), 3), _w((96,), 4), _w((96,), 5), _w((96,), 6) ** 2 + 1,
                                                                  rows=WindowLayout(1), eps=1e-5),
                                policy="double_round_once", layout=WindowLayout(1))),
    "BatchNormConstants int rows": (
        lambda c: WN.batch_norm(c.W, c.img, WN.BatchNormConstants(c.W, _w((96,), 3), _w((96,), 4), _w((96,), 5), _w((96,), 6) ** 2 + 1)),
        lambda c: WN.batch_norm(c.W, c.img, WN.BatchNormConstants(c.W, _w((96,), 3), _w((96,), 4), _w((96,), 5), _w((96,), 6) ** 2 + 1, 1))),
    "naive_attention": (lambda c: WT.naive_attention(c.V, c.val, c.val, c.val, tokens=49, dim=32),
                        lambda c: WT.naive_attention(c.V, c.val, c.val, c.val, tokens=49, dim=32,
                                                     policy="round_each_op")),
    "relative_position_index": (lambda c: WT.relative_position_index(ee), lambda c: WT.relative_position_index(ee, 7)),
    "relative_position_bias": (lambda c: WT.relative_position_bias(ee, _w((169, 3), 6)),
                               lambda c: WT.relative_position_bias(ee, _w((169, 3), 6), 7, policy="double_round_once")),
    "split_heads": (lambda c: WT.split_heads(c.W, c.img), lambda c: WT.split_heads(c.W, c.img, 96, 3)),
    "window_attention": (lambda c: WT.window_attention(c.W, c.img, c.block),
                         lambda c: WT.window_attention(c.W, c.img, c.block, None, window=7)),
    "shifted_window_mask": (lambda c: WT.shifted_window_mask(ee, c.coarse),
                            lambda c: WT.shifted_window_mask(ee, c.coarse, 7, 3, -100.0, policy="double_round_once")),
    "Conv3x3Constants": (lambda c: WC.Conv3x3Constants(c.W, _w((8, 6, 3, 3), 18), _w((8,), 19)).bias,
                         lambda c: WC.Conv3x3Constants(c.W, _w((8, 6, 3, 3), 18), _w((8,), 19), 49, 3).bias),
    "Conv3x3Constants layout": (lambda c: WC.Conv3x3Constants(c.W, _w((8, 6, 3, 3), 18), _w((8,), 19)).bias,
                                lambda c: WC.Conv3x3Constants(c.W, _w((8, 6, 3, 3), 18), _w((8,), 19),
                                                              rows=WindowLayout(7)).bias),
    "conv3x3_window": (lambda c: WC.conv3x3_window(c.W, c.img, c.c33, c.coarse),
                       lambda c: WC.conv3x3_window(c.W, c.img, c.c33, c.coarse, window=7,
                                                   policy="double_round_once")),
    "group_weight": (lambda c: WC.group_weight(_w((8, 16, 3, 3), 20), 1),
                     lambda c: WC.group_weight(_w((8, 16, 3, 3), 20), 1, 4)),
    "group_product": (lambda c: WC.group_product(ee, c.part, c.gc),
                      lambda c: WC.group_product(ee, c.part, c.gc, policy="double_round_once")),
    "group_product grouped_double": (lambda c: WC.group_product(ee, c.part, c.gc),
                                     lambda c: WC.group_product(ee, c.part, c.gc, policy="grouped_double")),
    "local_matrices": (lambda c: WR.local_matrices(56, 4, True), lambda c: WR.local_matrices(56, 4, True, window=7)),
    "window_token_order": (lambda c: WR.window_token_order(14, 21), lambda c: WR.window_token_order(14, 21, window=7)),
    "window_matrix_weights": (lambda c: WR.window_matrix_weights((3, 3), (14, 14)),
                              lambda c: WR.window_matrix_weights((3, 3), (14, 14), window=7)),
    "window_resize": (lambda c: WR.window_resize(ee, c.img, c.tokens, 4, 2),
                      lambda c: WR.window_resize(ee, c.img, c.tokens, 4, 2, False, window=7,
                                                 policy="double_round_once")),
    "window_matrix_resize": (lambda c: WR.window_matrix_resize(ee, c.val, (2, 2), c.tokens),
                             lambda c: WR.window_matrix_resize(ee, c.val, (2, 2), c.tokens, (28, 28), 0, 0, False,
                                                               window=7, policy="double_round_once")),
    "pool_rows": (lambda c: WO.pool_rows(14, 21), lambda c: WO.pool_rows(14, 21, (1, 2, 3, 6))),
    "plan_for": (lambda c: WO.plan_for(c.tokens), lambda c: WO.plan_for(c.tokens, sizes=(1, 2, 3, 6))),
    "window_pool": (lambda c: WO.window_pool(ee, c.img, c.tokens, 8),
                    lambda c: WO.window_pool(ee, c.img, c.tokens, 8, None, sizes=(1, 2, 3, 6), window=7,
                                             policy="double_round_once")),
    "patch_embed_constants": (lambda c: WP.patch_embed_constants(c.W, PE).proj.bias,
                              lambda c: WP.patch_embed_constants(c.W, PE, 49).proj.bias),
    "patch_embed_constants layout": (lambda c: WP.patch_embed_constants(c.W, PE, 1).norm_weight,
                                     lambda c: WP.patch_embed_constants(c.W, PE, WindowLayout(1)).norm_weight),
    "stride_columns": (lambda c: WP.stride_columns(ee, c.input, c.grid),
                       lambda c: WP.stride_columns(ee, c.input, c.grid, patch=4, bands=("VV", "VH"))),
    "patch_embed": (lambda c: WP.patch_embed(c.W, c.input, c.grid, c.pe),
                    lambda c: WP.patch_embed(c.W, c.input, c.grid, c.pe, patch=4,
                                             bands=("VV", "VH"), window=7, policy="double_round_once")),
    "valid_fraction": (lambda c: WP.valid_fraction(ee, c.input, c.grid),
                       lambda c: WP.valid_fraction(ee, c.input, c.grid, patch=4, bands=("VV", "VH"))),
    "patch_merge_constants": (lambda c: WP.patch_merge_constants(c.W, PM).norm_weight,
                              lambda c: WP.patch_merge_constants(c.W, PM, WindowLayout(7)).norm_weight),
    "patch_merge_constants int": (lambda c: WP.patch_merge_constants(c.W, PM).norm_weight,
                                  lambda c: WP.patch_merge_constants(c.W, PM, 49).norm_weight),
    "merge_windows": (lambda c: WP.merge_windows(ee, WP.gather_windows(ee, c.img, c.coarse), 96),
                      lambda c: WP.merge_windows(ee, WP.gather_windows(ee, c.img, c.coarse), 96, window=7)),
    "patch_merge": (lambda c: WP.patch_merge(c.W, c.img, c.pm, c.coarse),
                    lambda c: WP.patch_merge(c.W, c.img, c.pm, c.coarse, window=7,
                                             policy="double_round_once")),
    "add": (lambda c: WM.add(c.part, Feature(c.img2, c.tokens, 4)),
            lambda c: WM.add(c.part, Feature(c.img2, c.tokens, 4), policy="double_round_once")),
}


@pytest.mark.parametrize("case", sorted(DEFAULT_CASES))
def test_explicit_defaults_serialize_identically(offline_ee, case):
    c = _context()
    omitted, explicit = DEFAULT_CASES[case]
    a, b = _serialize(omitted(c)), _serialize(explicit(c))
    assert a == b


@pytest.mark.parametrize("dim,heads", [(5, 2), (0, 2), (-2, 2), (6, 0), (6, -2),
                                         (True, 1), (6, True), (6.0, 2), (6, 2.0)])
def test_split_heads_refuses_sizes_that_drop_or_empty_channels(dim, heads):
    # Refuse before slicing: floor division used to silently omit dim % heads channels.
    with pytest.raises(errors.UnsupportedOperation, match="positive integers|multiple of the head count"):
        WT.split_heads(None, None, dim=dim, heads=heads)


@pytest.mark.parametrize("integer", [int, np.int64])
def test_split_heads_preserves_every_qkv_channel(integer):
    qkv = np.arange(36, dtype=np.float32).reshape(2, 18)
    ops = types.SimpleNamespace(slice_cols=lambda value, start, end: value[:, start:end])
    heads = WT.split_heads(ops, qkv, dim=integer(6), heads=integer(3))
    assert len(heads) == 9
    for index, name in enumerate(("q", "k", "v")):
        np.testing.assert_array_equal(np.concatenate([heads[f"{name}_h{head}"] for head in range(3)], axis=1),
                                      qkv[:, index * 6:(index + 1) * 6])


def test_builders_record_layout_rows(offline_ee):
    c = _context()
    W = c.W
    assert WL.LinearConstants(W, _w((4, 3), 1), _w((4,), 2), WindowLayout()).rows == 49
    assert WL.LinearConstants(W, _w((4, 3), 1), _w((4,), 2), WindowTensor(c.img, c.tokens, 3)).rows == 49
    assert WL.LinearConstants(W, _w((4, 3), 1), None, ValueLayout(9)).rows == 9
    assert WL.conv1x1_constants(W, _w((4, 3, 1, 1), 1), None, 4).rows == 4
    assert WC.Conv3x3Constants(W, _w((2, 2, 3, 3), 1)).rows == 49
    assert WN.BatchNormConstants(W, *[_w((4,), s) for s in range(4)]).rows == 1
    assert WN.BatchNormConstants(W, *[_w((4,), s) for s in range(4)], rows=Feature(c.img, c.tokens, 4)).rows == 49
    assert WP.patch_embed_constants(W, PE).rows == 49 and WP.patch_embed_constants(W, PE, WindowLayout(1)).rows == 1
    assert WP.patch_merge_constants(W, PM).rows == 49 and WP.patch_merge_constants(W, PM, 1).rows == 1
    for bad in (0, -1, True, None, "49"):
        with pytest.raises((TypeError, ValueError)):
            WL.LinearConstants(W, _w((4, 3), 1), _w((4,), 2), bad)


def test_embedding_width_comes_from_the_weight_shape(offline_ee):
    c = _context()
    pe = {"proj.weight": _w((48, 32), 1), "proj.bias": _w((48,), 2),
          "norm.weight": _w((48,), 3), "norm.bias": _w((48,), 4)}
    t = WP.patch_embed(c.W, c.input, c.grid, WP.patch_embed_constants(c.W, pe))
    expected = WN.layer_norm(c.W, t["pe_proj"], WP.patch_embed_constants(c.W, pe).norm_weight,
                             WP.patch_embed_constants(c.W, pe).norm_bias, dim=48)
    from tensor_ee.layouts.shift import domain_mask
    assert t["x"].serialize() == expected.updateMask(domain_mask(ee, c.grid.coarse(4).coarse(7))).serialize()


def test_input_band_names_reach_the_graph(offline_ee):
    c = _context()
    renamed = WP.stride_columns(ee, c.input, c.grid, bands=("HH", "HV")).serialize()
    assert '"HH_00"' in renamed and '"VV_00"' not in renamed
    assert '"VV_00"' in WP.stride_columns(ee, c.input, c.grid).serialize()


def test_the_research_gathers_and_placement_are_gone():
    """L06b removed the gather registries, the gather and placement parameters and the
    unpinned gather: patch embedding gathers by stride, merging gathers whole windows."""
    assert not hasattr(WP, "COLUMN_GATHERS") and not hasattr(WP, "MERGE_GATHERS")
    assert set(inspect.signature(WP.patch_embed).parameters) == {"ops", "x", "grid", "c", "patch", "bands",
                                                                 "window", "policy", "norm"}
    assert set(inspect.signature(WP.patch_merge).parameters) == {"ops", "x", "c", "coarse1", "window", "policy"}
    assert set(inspect.signature(WP.gather_windows).parameters) == {"ee", "x", "coarse1"}


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------
POLICY_CALLS = {
    "linear": lambda c, p: WL.linear(c.W, c.img, c.lin, policy=p),
    "conv1x1": lambda c, p: WL.conv1x1(c.W, c.img, c.lin, policy=p),
    "gelu": lambda c, p: WA.gelu(c.W, c.img, policy=p),
    "layer_norm": lambda c, p: WN.layer_norm(c.W, c.img, c.lin.bias, c.lin.bias, dim=96, policy=p),
    "layer_norm2d": lambda c, p: WN.layer_norm2d(c.W, c.img, _w((96,), 1), _w((96,), 2), dim=96, policy=p),
    "batch_norm": lambda c, p: WN.batch_norm(c.W, c.img, c.bn, policy=p),
    "naive_attention": lambda c, p: WT.naive_attention(c.V, c.val, c.val, c.val, tokens=49, dim=32, policy=p),
    "relative_position_bias": lambda c, p: WT.relative_position_bias(ee, _w((169, 3), 6), policy=p),
    "shifted_window_mask": lambda c, p: WT.shifted_window_mask(ee, c.coarse, policy=p),
    "conv3x3_window": lambda c, p: WC.conv3x3_window(c.W, c.img, c.c33, c.coarse, policy=p),
    "depthwise_conv3x3_window": lambda c, p: WC.depthwise_conv3x3_window(
        c.W, c.img, WC.DepthwiseConv3x3Constants(c.W, _w((4, 1, 3, 3), 28), _w((4,), 29)),
        c.coarse, policy=p),
    "group_product": lambda c, p: WC.group_product(ee, c.part, c.gc, policy=p),
    "window_resize": lambda c, p: WR.window_resize(ee, c.img, c.tokens, 4, 2, policy=p),
    "window_matrix_resize": lambda c, p: WR.window_matrix_resize(ee, c.val, (2, 2), c.tokens, policy=p),
    "window_pool": lambda c, p: WO.window_pool(ee, c.img, c.tokens, 8, policy=p),
    "patch_embed": lambda c, p: WP.patch_embed(c.W, c.input, c.grid, c.pe, policy=p),
    "patch_merge": lambda c, p: WP.patch_merge(c.W, c.img, c.pm, c.coarse, policy=p),
    "add": lambda c, p: WM.add(c.part, Feature(c.img2, c.tokens, 4), policy=p),
    "fusion_grouped_double": lambda c, p: WU.fusion_grouped_double(
        ee, [c.part] * 4, WU.FusionConstants([WC.Conv3x3Constants(c.W, WC.group_weight(_w((8, 16, 3, 3), 20), g, 4))
                                              for g in range(4)], c.bn), policy=p),
}
ALLOWED = {name: {P.DOUBLE_ROUND_ONCE} for name in POLICY_CALLS}
ALLOWED["naive_attention"] = {P.ROUND_EACH_OP}
ALLOWED["group_product"] = {P.DOUBLE_ROUND_ONCE, P.GROUPED_DOUBLE}
ALLOWED["fusion_grouped_double"] = {P.DOUBLE_ROUND_ONCE, P.GROUPED_DOUBLE}


@pytest.mark.parametrize("name", sorted(POLICY_CALLS))
def test_policy_defaults_work_and_others_are_refused(offline_ee, name):
    c = _context()
    for policy in sorted(ALLOWED[name]):
        POLICY_CALLS[name](c, policy)
    for policy in sorted(set(P.POLICIES) | set(P.VARIANTS)):
        if policy not in ALLOWED[name]:
            with pytest.raises(errors.UnsupportedOperation):
                POLICY_CALLS[name](c, policy)
    for bad in ("float32", "none", "composite", None, 3):
        with pytest.raises(ValueError):
            POLICY_CALLS[name](c, bad)


def test_policy_parameters_match_the_metadata():
    for module in MODULES.values():
        for name, entry in module.METADATA.items():
            parameters = inspect.signature(getattr(module, name)).parameters
            if entry.policy in M.MARKERS:
                assert "policy" not in parameters, name           # relu, concat, layout moves, composite
            else:
                parameter = parameters["policy"]
                assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, name
                assert parameter.default == entry.policy, name
    assert set(POLICY_CALLS) == {name for module in MODULES.values() for name, e in module.METADATA.items()
                                 if e.policy not in M.MARKERS}


# ---------------------------------------------------------------------------
# Constants built for one layout are refused by another
# ---------------------------------------------------------------------------
def test_constants_built_for_one_layout_are_refused_by_another(offline_ee):
    c = _context()
    W = c.W
    one = WL.LinearConstants(W, _w((288, 96), 1), _w((288,), 2), WindowLayout(1))
    WL.linear(W, c.img, one, layout=WindowLayout(1))
    for layout in (WindowLayout(7), WindowTensor(c.img, c.tokens, 96), 49, ValueLayout(4)):
        with pytest.raises(RowMismatch):
            WL.linear(W, c.img, one, layout=layout)
        with pytest.raises(RowMismatch):
            WL.conv1x1(W, c.img, one, layout=layout)
    with pytest.raises(RowMismatch):
        WN.batch_norm(W, c.img, c.bn, layout=WindowLayout(1))
    with pytest.raises(RowMismatch):
        WN.layer_norm(W, c.img, row_constant(W, _w((96,), 1), WindowLayout()),
                      row_constant(W, _w((96,), 2), WindowLayout()), dim=96, layout=ValueLayout(4))
    with pytest.raises(RowMismatch):
        WN.layer_norm2d(W, c.img, row_constant(W, _w((96,), 1), WindowLayout()),
                        row_constant(W, _w((96,), 2), WindowLayout()), dim=96)          # rows=1 by default
    dense = WC.Conv3x3Constants(W, _w((8, 6, 3, 3), 18), _w((8,), 19), rows=1)
    with pytest.raises(RowMismatch):
        WC.conv3x3_window(W, c.img, dense, c.coarse)
    with pytest.raises(RowMismatch):
        WC.conv3x3_window(W, c.img, c.c33, c.coarse, window=5)
    with pytest.raises(RowMismatch):
        WC.group_product(ee, Feature(c.img, c.tokens, 4, "dense_row"), c.gc)
    with pytest.raises(RowMismatch):
        WP.patch_embed(W, c.input, c.grid, WP.patch_embed_constants(W, PE, 1))
    with pytest.raises(RowMismatch):
        WP.patch_merge(W, c.img, WP.patch_merge_constants(W, PM, WindowLayout(1)), c.coarse)
    block = types.SimpleNamespace(**{**vars(c.block), "qkv": one})
    with pytest.raises(RowMismatch):
        WT.window_attention(W, c.img, block)
    with pytest.raises(RowMismatch):
        WT.window_attention(W, c.img, c.block, window=5)
    assert issubclass(RowMismatch, ValueError)          # the old patch_embed refusal was a ValueError


# ---------------------------------------------------------------------------
# Metadata: locality, policy and status per the inventory
# ---------------------------------------------------------------------------
# (block): (locality, radius, chip_edge_dependent, policy, status, inventory locality).
# The inventory column (docs/building-blocks.md, section 3; the JSON's for helpers it does
# not list) maps: "pointwise"/"per token" -> POINTWISE; "window" -> WINDOW; "window+ring"
# and the halo-based "3x3" -> WINDOW_RING, ring 1; "4x4" and "2x2 windows" (the k x k
# class) -> KERNEL, radius k // 2 as for the R convolution; "extent" -> EXTENT; "window;
# depends on chip edge" -> WINDOW, chip_edge_dependent. Since P03, patch_embed and
# stride_columns ("4x4" in the inventory, at patch 4) have radius 4: half the largest
# validated patch, 8.
DRO, REO = P.DOUBLE_ROUND_ONCE, P.ROUND_EACH_OP
EXPECTED_METADATA = {
    "window_depth_to_space": (Locality.WINDOW, None, False, M.NO_ARITHMETIC,
        "Fixed — offline only: factor 2; windows 7, 14 to window 7", "window"),
    "concat_rows": (Locality.POINTWISE, None, False, M.NO_ARITHMETIC,
                    "Fixed — offline only: one class row followed by spatial rows", "per crop pixel"),
    "depthwise_conv3x3_window": (Locality.WINDOW_RING, 1, False, DRO,
        "Ready — 3 × 3, stride1, padding1, dilation1, multiplier1; window7; D4 optional bias, D1024 biased; D16 biased in C4/D16 Mix-FFN", "3x3 depthwise"),
    "linear": (Locality.POINTWISE, None, False, DRO, "Ready", "per token"),
    "conv1x1": (Locality.POINTWISE, None, False, DRO, "Ready", "pointwise"),
    "gelu": (Locality.POINTWISE, None, False, DRO, "Ready", "pointwise"),
    "relu": (Locality.POINTWISE, None, False, M.NO_ARITHMETIC, "Ready", "pointwise"),
    "layer_norm": (Locality.POINTWISE, None, False, DRO, "Ready", "per token"),
    "layer_norm2d": (Locality.POINTWISE, None, False, DRO, "Ready — windows 5, 7, 8, 12; eps 1e-6", "per token"),
    "batch_norm": (Locality.POINTWISE, None, False, DRO, "Ready", "pointwise"),
    "naive_attention": (Locality.WINDOW, None, False, REO, "Ready", "window"),
    "split_heads": (Locality.POINTWISE, None, False, M.NO_ARITHMETIC, "Ready", "per token"),
    "relative_position_bias": (Locality.POINTWISE, None, False, DRO, "Ready — windows 5, 7, 8, 12",
                               "— (non-spatial V)"),
    "window_attention": (Locality.WINDOW, None, False, M.COMPOSITE, "Ready — windows 5, 7, 8, 12", "window"),
    "shifted_window_mask": (Locality.WINDOW, None, True, DRO, "Ready — windows 5, 7, 8, 12; shift ⌊w/2⌋",
                            "window; depends on chip edge"),
    "conv3x3_window": (Locality.WINDOW_RING, 1, False, DRO, "Ready — 3 × 3 kernel, stride 1; windows 5, 7, 8, 12",
                       "3x3 (halo)"),
    "group_product": (Locality.WINDOW_RING, 1, False, DRO,
                      "Ready — 1, 2, 4, 8 groups at windows 7, 12; 4 groups at window 8", "3x3 (halo)"),
    # Chip-edge dependent over its validated settings (X02b): exactly with align_corners=True
    # (per_setting), whose weights depend on the map's size.
    "window_resize": (Locality.WINDOW_RING, 1, True, DRO,
                      "Ready — windows 7, 8; factors 2, 4, 8 (align_corners=False), 4 (align_corners=True)",
                      "window+ring"),
    "window_matrix_resize": (Locality.EXTENT, None, False, DRO,
                             "Ready — windows 7, 8; sources 1, 2, 3, 6 onto one window (align_corners=False)",
                             "extent"),
    # The square extents P04 ran (lead decision at X02b's fix round; "extents 5–14" before).
    "window_pool": (Locality.EXTENT, None, False, DRO,
                    "Ready — extents 5, 7, 8, 12 and 14 tokens with bins (1, 2, 3, 6); "
                    "extents 8 and 12 with bins (1, 2, 4, 8)", "extent"),
    "gather": (Locality.EXTENT, None, False, M.NO_ARITHMETIC, "Ready", "extent"),
    "patch_embed": (Locality.KERNEL, 4, False, DRO,
                    "Ready — patches 2, 4, 8; 1–4 bands; widths 48, 96, 128; windows 5, 7, 8, 12", "4x4"),
    "stride_columns": (Locality.KERNEL, 4, False, M.NO_ARITHMETIC, "Ready — patches 2, 4, 8; 1–4 bands", "4x4"),
    "patch_merge": (Locality.KERNEL, 1, False, DRO, "Ready — windows 5, 7, 8, 12", "2x2 windows"),
    "gather_windows": (Locality.KERNEL, 1, False, M.NO_ARITHMETIC, "Ready — windows 5, 7, 8, 12", "2x2 windows"),
    "merge_windows": (Locality.WINDOW, None, False, M.NO_ARITHMETIC, "Ready — windows 5, 7, 8, 12",
                      "window (2x2 block)"),
    "add": (Locality.POINTWISE, None, False, DRO, "Ready", "pointwise"),
    "concat": (Locality.POINTWISE, None, False, M.NO_ARITHMETIC, "Ready", "pointwise"),
}

# The SegFormer composites (X01a), promoted from the model, which the inventory predates. Their
# locality is their parts' widest: the global attention and the MiT block depend on the whole
# chip (EXTENT, chip-edge dependent), as does the shared-key core; the reduction makes the whole
# chip one window (EXTENT, like adaptive pooling); the overlap embedding is k x k, radius 3
# (its 7 x 7 kernel), decimation k x k, radius 2 (factor 4, as stride_columns counts a patch);
# the Mix-FFN, the pyramid and the head reach a ring of 1 (the depthwise convolution, the resize).
# The head is also chip-edge dependent: its align_corners=True weights depend on the map's size.
SEGFORMER = "Fixed — SegFormer-B0 (mit_b0) at 224 × 224: "
SEGFORMER_MODEL = "widths 32, 64, 160, 256; heads 1, 2, 5, 8; reduction ratios 8, 4, 2, 1; window 7"
EXPECTED_METADATA.update({
    "channel_norm": (Locality.POINTWISE, None, False, M.COMPOSITE,
                     SEGFORMER + "widths 32, 64, 160, 256; eps 1e-5, 1e-6; window 7", "per token"),
    "decimate": (Locality.KERNEL, 2, False, M.NO_ARITHMETIC,
                 SEGFORMER + "factor 4 at width 32, factor 2 at widths 64, 160, 256; window 7", "stride sample"),
    "overlap_embedding": (Locality.KERNEL, 3, False, M.COMPOSITE,
                          SEGFORMER + "k7/s4 2 → 32; k3/s2 32 → 64, 64 → 160, 160 → 256; window 7", "7x7"),
    "spatial_reduction": (Locality.EXTENT, None, False, M.COMPOSITE,
                          SEGFORMER + "widths 32, 64, 160, 256 at reduction ratios 8, 4, 2, 1; window 7", "extent"),
    "shared_key_attention": (Locality.EXTENT, None, True, M.COMPOSITE,
                             SEGFORMER + "widths 32, 64, 160, 256; heads 1, 2, 5, 8; one 7 × 7 key window",
                             "extent; depends on chip edge"),
    "global_attention": (Locality.EXTENT, None, True, M.COMPOSITE, SEGFORMER + SEGFORMER_MODEL,
                         "extent; depends on chip edge"),
    "mix_ffn": (Locality.WINDOW_RING, 1, False, M.COMPOSITE,
                SEGFORMER + "widths 32, 64, 160, 256, hidden 4 × width; window 7", "3x3 (halo)"),
    "mit_block": (Locality.EXTENT, None, True, M.COMPOSITE, SEGFORMER + SEGFORMER_MODEL,
                  "extent; depends on chip edge"),
    "project_pyramid": (Locality.WINDOW_RING, 1, False, M.COMPOSITE,
                        SEGFORMER + "widths 32, 64, 160, 256 to 256 on C1 56 × 56; window 7", "window+ring"),
    "fuse_head": (Locality.WINDOW_RING, 1, True, M.COMPOSITE,
                  SEGFORMER + "1024 → 256 channels → 2 classes, 56 × 56 to 224 × 224; window 7",
                  "window+ring; depends on chip edge"),
})

# The Swin and UPerNet composites (X01b), promoted from the Swin-T224 model; the inventory column
# is the research composite's. The SW-MSA block reads w - w // 2 tokens of its neighbours (4 at
# windows 7 and 8) and its mask depends on the chip edge; the stage and the encoder extend each
# SW-MSA block's input periodically (EXTENT, chip-edge dependent). The PSP and the FPN pool C4's
# whole extent (EXTENT, not chip-edge dependent, like window_pool); the FPN's blocks, the fusion
# and the head reach a ring of 1; the head's align_corners=True resize is chip-edge dependent.
SWIN = "Fixed — Swin-T224 and the capstone: "
SWIN_WIDTHS = ("widths 96, 192, 384, 768, heads 3, 6, 12, 24 at window 7; "
               "widths 48, 96, 192, 384, heads 2, 4, 8, 16 at window 8")
UPERNET_DECODER = "decoder widths 256 (window 7) and 64 (window 8)"
UPERNET_HEAD = "256 → 2 and 64 → 3 classes, × 4 from C1's 56 × 56 and 64 × 64 tokens"
RING, CMP = Locality.WINDOW_RING, M.COMPOSITE
EXPECTED_METADATA.update({
    "swin_mlp_branch": (Locality.POINTWISE, None, False, CMP,
                        SWIN + "widths 96, 192, 384, 768 at window 7 and 48, 96, 192, 384 at window 8; hidden 4 × width",
                        "pointwise (per token row)"),
    "swin_block": (Locality.WINDOW, None, False, CMP, SWIN + SWIN_WIDTHS, "window-local"),
    # Stages 1-3 only: stage 4 is one window in both models, never shifted (lead decision, X02b).
    "shifted_swin_block": (RING, 4, True, CMP,
                           SWIN + "widths 96, 192, 384, heads 3, 6, 12 at window 7; widths 48, 96, 192, heads 2, 4, 8 "
                                  "at window 8; shift ⌊w/2⌋; not stage 4 (one window, unshifted)", "window+ring"),
    "swin_stage": (Locality.EXTENT, None, True, CMP, SWIN + SWIN_WIDTHS + "; depths 2, 2, 6, 2 and 2, 2, 2, 2",
                   "window+ring (depth//2 windows per stage)"),
    "swin_patch_embedding": (Locality.KERNEL, 2, False, CMP,
                             SWIN + "patch 4; 2 bands → width 96 at window 7, 3 bands → width 48 at window 8",
                             "bounded k x k (k=4 px) then window-local"),
    "swin_encoder": (Locality.EXTENT, None, True, CMP,
                     SWIN + "8 × 8 stage-1 windows of 7 (stages 1–3, then 4) and of 8 (stages 1–4), "
                            "without the final norm", "extent-global"),
    "psp_layer_norm": (Locality.POINTWISE, None, False, CMP, SWIN + "C4 widths 768 (window 7) and 384 (window 8)",
                       "pointwise"),
    "psp_branch": (Locality.POINTWISE, None, False, CMP,
                   SWIN + "bins 1, 2, 3, 6; C4 widths 768 and 384 to " + UPERNET_DECODER, "pointwise"),
    "psp_forward": (Locality.EXTENT, None, False, CMP,
                    SWIN + "bins (1, 2, 3, 6) on C4's 7 × 7 and 8 × 8 tokens; C4 widths 768 and 384 to "
                    + UPERNET_DECODER, "extent-global"),
    "fpn_layer_norm": (Locality.POINTWISE, None, False, CMP,
                       SWIN + "C1–C3 widths 96, 192, 384 (window 7) and 48, 96, 192 (window 8)", "pointwise"),
    "fpn_lateral": (Locality.POINTWISE, None, False, CMP,
                    SWIN + "C1–C3 widths 96, 192, 384 and 48, 96, 192 to " + UPERNET_DECODER, "pointwise"),
    "fpn_upsample": (RING, 1, False, CMP, SWIN + "× 2 (align_corners=False) at " + UPERNET_DECODER, "window+ring"),
    "fpn_merge": (RING, 1, False, CMP, SWIN + "× 2 (align_corners=False) and the addition at " + UPERNET_DECODER,
                  "window+ring"),
    "fpn_conv_block": (RING, 1, False, CMP, SWIN + UPERNET_DECODER, "bounded k×k (k=3)"),
    "fpn_forward": (Locality.EXTENT, None, False, CMP,
                    SWIN + "C4's 7 × 7 and 8 × 8 tokens; C1–C4 widths 96, 192, 384, 768 and 48, 96, 192, 384 to "
                    + UPERNET_DECODER, "extent-global"),
    "fusion_resize_to_c1": (RING, 1, False, CMP,
                            SWIN + "P2, P3 and the PSP output onto C1's 56 × 56 and 64 × 64 tokens at "
                            + UPERNET_DECODER, "window+ring"),
    "fusion_resize_level": (RING, 1, False, CMP,
                            SWIN + "factors 1, 2, 4, 8 (align_corners=False) at " + UPERNET_DECODER, "window+ring"),
    "fusion_concat": (Locality.POINTWISE, None, False, M.NO_ARITHMETIC, SWIN + "four levels at " + UPERNET_DECODER,
                      "pointwise"),
    "fusion_bn_relu": (Locality.POINTWISE, None, False, CMP, SWIN + UPERNET_DECODER, "pointwise"),
    "fusion_grouped_double": (RING, 1, False, P.GROUPED_DOUBLE,
                              SWIN + "4 groups, 1024 → 256 (window 7) and 256 → 64 (window 8)", "bounded k×k (k=3)"),
    # Radius 2 in source-grid tokens (X02b, from X01b's review): the resize's ring and the
    # convolution's.
    "fusion_forward": (RING, 2, False, CMP,
                       SWIN + "4 groups on C1's 56 × 56 and 64 × 64 tokens at " + UPERNET_DECODER, "window+ring"),
    "head_conv": (Locality.POINTWISE, None, False, CMP,
                  SWIN + "256 → 2 classes (window 7) and 64 → 3 classes (window 8)", "pointwise"),
    "head_upsample": (RING, 1, True, CMP,
                      SWIN + "× 4 (align_corners=True) from C1's 56 × 56 (window 7) and 64 × 64 (window 8) tokens",
                      "window+ring"),
    "head_forward": (RING, 1, True, CMP, SWIN + UPERNET_HEAD, "window+ring"),
    "head_logits": (RING, 1, True, CMP, SWIN + UPERNET_HEAD, "window+ring"),
    "head_dense": (Locality.WINDOW, None, False, M.NO_ARITHMETIC,
                   SWIN + "2 and 3 classes on 224 × 224 (window 7) and 256 × 256 (window 8) tokens", "window-local"),
})

# Public names that are not blocks: constant builders, client-side coefficient and grid
# helpers, cost functions and the layout helpers. valid_fraction is a validity diagnostic
# whose mean stays double (no float32 rounding), so it carries no block policy.
NOT_BLOCKS = {
    "linear": {"LinearConstants", "conv1x1_constants", "as_layout", "builder_rows", "no_constants",
               "linear_values", "linear_arrays"},
    "activations": set(),
    "normalization": {"BatchNormConstants", "layer_norm_values", "layer_norm_arrays", "batch_norm_values",
                      "batch_norm_arrays"},
    "attention": {"relative_position_index", "coarse_projection", "window_coordinates", "region_boundaries",
                  "relative_bias_values", "relative_bias_arrays", "window_attention_values",
                  "window_attention_arrays"},
    "conv": {"DepthwiseConv3x3Constants", "depthwise_conv3x3_values", "depthwise_conv3x3_arrays", "Conv3x3Constants", "group_weight", "conv3x3_values", "conv3x3_arrays", "group_product_values",
             "group_product_arrays"},
    "resample": {"window_depth_to_space_values", "window_depth_to_space_arrays", "bilinear_weights", "model_bilinear_weights", "float32_exact", "local_matrices",
                 "window_token_order", "output_weights", "window_matrix_weights", "window_resize_values",
                 "window_matrix_resize_values", "window_resize_edge"},
    "pooling": {"bins", "onehot_bins", "check_rule", "pool_rows", "offsets", "axis_membership", "PoolPlan",
                "plan_for", "window_membership", "window_partials", "Pooled", "window_pool_values"},
    "patch": {"PatchEmbedConstants", "patch_embed_constants", "valid_fraction", "PatchMergeConstants",
              "patch_merge_constants", "stage2_coarse", "patch_embed_values", "patch_embed_arrays",
              "patch_merge_values", "patch_merge_arrays", "patch_locality"},
    "merge": {"concat_rows_values", "concat_rows_arrays"},
    "segformer": {"NormConstants", "norm_constants", "OverlapConstants", "overlap_constants",
                  "AttentionConstants", "attention_constants", "MixFFNConstants", "mix_ffn_constants",
                  "MiTBlockConstants", "mit_block_constants",
                  "channel_norm_values", "channel_norm_arrays", "overlap_embedding_values",
                  "overlap_embedding_arrays", "spatial_reduction_values", "spatial_reduction_arrays",
                  "global_attention_values", "global_attention_arrays", "mix_ffn_values", "mix_ffn_arrays",
                  "mit_block_values", "mit_block_arrays", "project_pyramid_values", "project_pyramid_arrays",
                  "fuse_head_values", "fuse_head_arrays", "project_pyramid_constants", "fuse_head_constants"},
    "swin": {"SwinBlockConstants", "swin_block_constants", "SwinEncoderConstants", "swin_stage_blocks",
             "swin_merge_constants", "swin_patch_embedding_constants", "SwinEncoderGrids", "swin_encoder_grids",
             "swin_mlp_branch_values", "swin_mlp_branch_arrays", "swin_block_values", "swin_block_arrays",
             "shifted_swin_block_values", "shifted_swin_block_arrays", "swin_stage_values", "swin_stage_arrays",
             "swin_patch_embedding_values", "swin_patch_embedding_arrays", "swin_encoder_values",
             "swin_encoder_arrays"},
    "upernet": {"psp_branch_prefix", "batch_norm_args", "PSPConstants", "psp_constants", "FPNConstants",
                "fpn_bn_constants", "fpn_level_constants", "fpn_constants", "FusionConstants", "fusion_constants",
                "pyramid_features",
                "fusion_resize_factor", "head_constants",
                "psp_layer_norm_values", "psp_layer_norm_arrays", "psp_branch_values", "psp_branch_arrays",
                "psp_forward_values", "psp_forward_arrays", "fpn_layer_norm_values", "fpn_layer_norm_arrays",
                "fpn_lateral_values", "fpn_lateral_arrays", "fpn_upsample_values", "fpn_upsample_arrays",
                "fpn_merge_values", "fpn_merge_arrays", "fpn_conv_block_values", "fpn_conv_block_arrays",
                "fpn_forward_values", "fpn_forward_arrays", "fusion_resize_level_values",
                "fusion_resize_level_arrays", "fusion_resize_to_c1_values", "fusion_resize_to_c1_arrays",
                "fusion_bn_relu_values", "fusion_bn_relu_arrays", "fusion_grouped_double_values",
                "fusion_grouped_double_arrays", "fusion_forward_values", "fusion_forward_arrays",
                "head_conv_values", "head_conv_arrays", "head_upsample_values", "head_upsample_arrays",
                "head_forward_values", "head_forward_arrays"},
    "swin_upernet_config": {"SwinUPerNetConfig", "check_shapes", "stage_shifted", "bn_args", "level_tensors",
                            "block_index", "lateral_prefix", "conv_prefix", "norm_prefix"},
}


def _defined(module) -> set[str]:
    return {name for name, value in vars(module).items()
            if not name.startswith("_") and (inspect.isfunction(value) or inspect.isclass(value))
            and value.__module__ == module.__name__}


def test_every_module_of_the_package_is_in_the_census():
    """Every module is a block module (``MODULES``, with ``METADATA``) or a settings module with
    no block, whose public names are all listed as not blocks (X02b, from X01b's review)."""
    files = {path.stem for path in (ROOT / "tensor_ee/blocks/window").glob("*.py")} - {"__init__"}
    assert files == set(MODULES) | set(SETTINGS_MODULES)
    assert not set(MODULES) & set(SETTINGS_MODULES)
    for key, module in SETTINGS_MODULES.items():
        assert not hasattr(module, "METADATA"), key
        assert _defined(module) == NOT_BLOCKS[key], key


def test_every_public_block_has_valid_metadata_matching_the_inventory():
    seen = set()
    for key, module in MODULES.items():
        assert M.validate_metadata(module.METADATA) is module.METADATA
        public = {name for name, value in vars(module).items()
                  if not name.startswith("_") and (inspect.isfunction(value) or inspect.isclass(value))
                  and value.__module__ == module.__name__}
        assert set(module.METADATA) | NOT_BLOCKS[key] == public, key
        assert not set(module.METADATA) & NOT_BLOCKS[key]
        for name, entry in module.METADATA.items():
            locality, radius, edge, policy, status, _ = EXPECTED_METADATA[name]
            assert (entry.locality, entry.radius, entry.chip_edge_dependent, entry.policy, entry.status) == (
                locality, radius, edge, policy, status), name
            assert entry.status.startswith(("Ready", "Fixed — ")) and "Fixed," not in entry.status
            assert entry.layout_in in M.LAYOUTS and entry.layout_out in M.LAYOUTS
            assert entry.evidence and all((ROOT / ref.split("#")[0]).exists() for ref in entry.evidence
                                          if "/" in ref and " " not in ref), name
            seen.add(name)
    assert seen == set(EXPECTED_METADATA)


def test_cost_functions_are_defined_in_the_library():
    for module in MODULES.values():
        for entry in module.METADATA.values():
            assert isinstance(entry, M.BlockMetadata)
            assert entry.embedded_constants.__module__.startswith(LIB)
            assert entry.asset_arrays.__module__.startswith(LIB)


# ---------------------------------------------------------------------------
# Cost functions: library formulas vs the research budget counts
# ---------------------------------------------------------------------------
def _cost(name):
    for module in MODULES.values():
        if name in module.METADATA:
            return module.METADATA[name].embedded_constants
    raise KeyError(name)


# Production: the 224 x 224 Swin-T224 chip (C1 56, C2 28, C3 14, C4 7 tokens); and the
# budget helpers' defaults (the 896 crop of Milestones 7-11: C4 28).
CONFIGS = [(7, 7), (28, 28)]
WIDTHS = {1: 96, 2: 192, 3: 384, 4: 768}
DECODER, SIZES = 256, (1, 2, 3, 6)

# The research budget helpers' embedded-value counts (upernet/earth_engine/{psp,fpn,fusion,
# segmentation_head,topology}.py ``budget``/``subsystem_constants``, written independently of
# the library cost functions), recorded at the archive tag research-archive-2026-10-12.
BUDGETS = {
    "psp": {(7, 7): {"c4_layernorm2d": 75264, "pool_tables": 750, "branch_conv_1x1": 786432,
                     "branch_batchnorm": 51200, "branch_resize_tables": 2450, "out_conv_3x3": 4128768,
                     "out_batchnorm": 50176},
            (28, 28): {"c4_layernorm2d": 75264, "pool_tables": 2850, "branch_conv_1x1": 786432,
                       "branch_batchnorm": 51200, "branch_resize_tables": 39200, "out_conv_3x3": 4128768,
                       "out_batchnorm": 50176}},
    "fpn": {(7, 7): {"c1_c3_layernorm2d": 65856, "fpn_lateral_conv_1x1": 172032, "fpn_lateral_batchnorm": 150528,
                     "fpn_resize_tables": 1288, "fpn_conv_3x3": 1769472, "fpn_conv_batchnorm": 150528},
            (28, 28): {"c1_c3_layernorm2d": 65856, "fpn_lateral_conv_1x1": 172032,
                       "fpn_lateral_batchnorm": 150528, "fpn_resize_tables": 1624, "fpn_conv_3x3": 1769472,
                       "fpn_conv_batchnorm": 150528}},
    "fusion": {(7, 7): {"fusion_resize_tables": 2316, "fusion_conv_3x3": 2359296, "fusion_batchnorm": 50176},
               (28, 28): {"fusion_resize_tables": 2712, "fusion_conv_3x3": 2359296, "fusion_batchnorm": 50176}},
    "head": {(56, 56): {"head_conv_1x1": 512, "head_bias_rows": 98, "head_upsample_tables": 4096},
             (224, 224): {"head_conv_1x1": 512, "head_bias_rows": 98, "head_upsample_tables": 16384}},
    "topology": {(h, h): {"feature_norms": 141120, "psp_pooled_1x1": 837632,
                          "psp_resize_tables": {7: 2450, 28: 39200}[h], "psp_out_conv_3x3": 4178944,
                          "fpn_lateral_1x1": 322560, "fpn_conv_3x3": 1920000, "fusion_3x3": 2409472,
                          "resize_weights": 1890, "segmentation_head": 610} for h in (7, 28)},
    "head_budget": {"head_conv_1x1": 512, "head_bias": 98},
    "M6_PE": 17184,
    "FINAL_NORM": 75264,
}


@pytest.mark.parametrize("h, w", CONFIGS)
def test_psp_costs_equal_its_budget(h, w):
    budget = BUDGETS["psp"][(h, w)]
    assert budget["c4_layernorm2d"] == _cost("layer_norm2d")(768)
    assert budget["pool_tables"] == _cost("window_pool")(h, w)
    assert budget["branch_conv_1x1"] == sum(_cost("conv1x1")(768, DECODER, bias=False, layout=ValueLayout(s * s))
                                            for s in SIZES)
    assert budget["branch_batchnorm"] == sum(_cost("batch_norm")(DECODER, layout=ValueLayout(s * s)) for s in SIZES)
    assert budget["branch_resize_tables"] == sum(_cost("window_matrix_resize")((s, s), h, w) for s in SIZES)
    assert budget["out_conv_3x3"] == _cost("conv3x3_window")(768 + len(SIZES) * DECODER, DECODER)
    assert budget["out_batchnorm"] == _cost("batch_norm")(DECODER)
    assert WO.PoolPlan(h, w).embedded_values == _cost("window_pool")(h, w)


@pytest.mark.parametrize("h, w", CONFIGS)
def test_fpn_costs_equal_its_budget(h, w):
    budget = BUDGETS["fpn"][(h, w)]
    levels = (3, 2, 1)
    assert budget["c1_c3_layernorm2d"] == sum(_cost("layer_norm2d")(WIDTHS[k]) for k in levels)
    assert budget["fpn_lateral_conv_1x1"] == sum(_cost("conv1x1")(WIDTHS[k], DECODER, bias=False) for k in levels)
    assert budget["fpn_lateral_batchnorm"] == 3 * _cost("batch_norm")(DECODER)
    assert budget["fpn_resize_tables"] == sum(
        _cost("window_resize")(h * 2 ** (3 - k), w * 2 ** (3 - k), 2) for k in levels)
    assert budget["fpn_conv_3x3"] == 3 * _cost("conv3x3_window")(DECODER, DECODER)
    assert budget["fpn_conv_batchnorm"] == 3 * _cost("batch_norm")(DECODER)


@pytest.mark.parametrize("h, w", CONFIGS)
def test_fusion_costs_equal_its_budget(h, w):
    budget = BUDGETS["fusion"][(h, w)]
    assert budget["fusion_resize_tables"] == sum(
        _cost("window_resize")((8 // f) * h, (8 // f) * w, f) for f in (2, 4, 8))
    assert budget["fusion_conv_3x3"] == 4 * _cost("group_product")(DECODER, DECODER)
    assert budget["fusion_batchnorm"] == _cost("batch_norm")(DECODER)


@pytest.mark.parametrize("h1, w1", [(56, 56), (224, 224)])
def test_head_costs_equal_its_budget(h1, w1):
    budget = BUDGETS["head"][(h1, w1)]
    assert budget["head_conv_1x1"] + budget["head_bias_rows"] == _cost("conv1x1")(DECODER, 2, bias=True)
    assert budget["head_upsample_tables"] == _cost("window_resize")(h1, w1, 4, align_corners=True)


@pytest.mark.parametrize("h, w", CONFIGS)
def test_topology_costs_equal_its_subsystems_except_the_resize_approximation(h, w):
    sub = BUDGETS["topology"][(h, w)]
    bn = _cost("batch_norm")
    assert sub["feature_norms"] == sum(_cost("layer_norm2d")(c) for c in WIDTHS.values())
    assert sub["psp_pooled_1x1"] == sum(_cost("conv1x1")(768, DECODER, bias=False)
                                        + bn(DECODER, layout=ValueLayout(s * s)) for s in SIZES)
    assert sub["psp_resize_tables"] == sum(_cost("window_matrix_resize")((s, s), h, w) for s in SIZES)
    assert sub["psp_out_conv_3x3"] == _cost("conv3x3_window")(768 + 4 * DECODER, DECODER) + bn(DECODER)
    assert sub["fpn_lateral_1x1"] == sum(_cost("conv1x1")(WIDTHS[k], DECODER, bias=False) + bn(DECODER)
                                         for k in (1, 2, 3))
    assert sub["fpn_conv_3x3"] == 3 * (_cost("conv3x3_window")(DECODER, DECODER) + bn(DECODER))
    assert sub["fusion_3x3"] == 4 * _cost("group_product")(DECODER, DECODER) + bn(DECODER)
    assert sub["segmentation_head"] == _cost("conv1x1")(DECODER, 2, bias=True)   # without the x4 tables
    assert BUDGETS["M6_PE"] == _cost("patch_embed")(96)
    assert BUDGETS["FINAL_NORM"] == WN.layer_norm_values(768)
    # Reported difference: topology's "resize_weights" is a fixed approximation
    # (3 * 2 * (7 * 9 * 5) = 1,890) of the FPN x2 and fusion tables; the library counts
    # them exactly (the same tables fpn.budget and fusion.budget count).
    fpn_tables = sum(_cost("window_resize")(h * 2 ** (3 - k), w * 2 ** (3 - k), 2) for k in (3, 2, 1))
    fusion_tables = sum(_cost("window_resize")((8 // f) * h, (8 // f) * w, f) for f in (2, 4, 8))
    assert sub["resize_weights"] == 1_890
    assert (fpn_tables, fusion_tables) == {(7, 7): (1_288, 2_316), (28, 28): (1_624, 2_712)}[(h, w)]


def test_head_budget_equals_the_head_conv_cost():
    head = BUDGETS["head_budget"]
    assert head["head_conv_1x1"] + head["head_bias"] == _cost("conv1x1")(DECODER, 2, bias=True)


def test_builder_embedded_values_agree_with_the_cost_functions(offline_ee):
    W = WindowOps(ee)
    bn = WN.BatchNormConstants(W, *[_w((256,), s) for s in range(4)], rows=49)
    assert bn.embedded_values == _cost("batch_norm")(256)
    conv = WC.Conv3x3Constants(W, _w((8, 6, 3, 3), 1))
    assert conv.embedded_values == _cost("conv3x3_window")(6, 8)
    # Reported difference: Conv3x3Constants.embedded_values counts a bias as Cout values
    # (before the server-side repeat); the library cost, like the budgets, counts rows x Cout.
    biased = WC.Conv3x3Constants(W, _w((8, 6, 3, 3), 1), _w((8,), 2))
    assert biased.embedded_values == _cost("conv3x3_window")(6, 8) + 8
    assert _cost("conv3x3_window")(6, 8, bias=True) == _cost("conv3x3_window")(6, 8) + 49 * 8


def _assets(value) -> set:
    return set(re.findall(r"projects/l04-assets/assets/(\w+)", value.serialize()))


def test_asset_array_counts_match_the_graphs(offline_ee):
    W = WindowOps(ee)
    img = ee.Image("projects/l04-test/assets/windows")
    tokens = TokenGrid(28, 28, 2.5)
    lin = WL.LinearConstants(W, _server("w", (288, 96)), _server("b", (288,)), 49)
    assert len(_assets(WL.linear(W, img, lin))) == WL.METADATA["linear"].asset_arrays()
    nob = WL.conv1x1_constants(W, _server("w", (256, 96)), None, 49)
    assert len(_assets(WL.conv1x1(W, img, nob))) == WL.METADATA["conv1x1"].asset_arrays(bias=False)
    ln = WN.layer_norm2d(W, img, _server("g", (96,)), _server("be", (96,)), dim=96, rows=49)
    assert len(_assets(ln)) == WN.METADATA["layer_norm2d"].asset_arrays()
    bn = WN.BatchNormConstants(W, *[_server(n, (96,)) for n in ("g", "b", "m", "v")], rows=49)
    assert len(_assets(WN.batch_norm(W, img, bn))) == WN.METADATA["batch_norm"].asset_arrays()
    c33 = WC.Conv3x3Constants(W, _server("w33", (8, 54)))
    assert len(_assets(WC.conv3x3_window(W, img, c33, tokens.coarse(7)))) == WC.METADATA["conv3x3_window"].asset_arrays()
    gc = WC.Conv3x3Constants(W, WC.group_weight(_server("wf", (8, 144)), 2))
    assert len(_assets(WC.group_product(ee, Feature(img, tokens, 4), gc))) == WC.METADATA["group_product"].asset_arrays()
    table = WT.relative_position_bias(ee, _server("table", (169, 3)))
    assert len(_assets(table[0])) == WT.METADATA["relative_position_bias"].asset_arrays()
    from tensor_ee.models.swin_t224 import encoder as SB
    params = {k: _server(k.replace(".", "_"), np.shape(v)) for k, v in _block_parameters().items()}
    block = SB.block_constants(W, params)
    assert len(_assets(WT.window_attention(W, img, block)["concat"])) == WT.METADATA["window_attention"].asset_arrays()
    pe = WP.patch_embed_constants(W, {k: _server(k.replace(".", "_"), np.shape(v)) for k, v in PE.items()})
    x = WP.patch_embed(W, ee.Image("projects/l04-test/assets/input"), TokenGrid(112, 112, 2.5), pe)["x"]
    assert len(_assets(x)) == WP.METADATA["patch_embed"].asset_arrays()
    pm = WP.patch_merge_constants(W, {k: _server(k.replace(".", "_"), np.shape(v)) for k, v in PM.items()})
    assert len(_assets(WP.patch_merge(W, img, pm, tokens.coarse(7))["out"])) == WP.METADATA["patch_merge"].asset_arrays()
    for name, value in (("window_resize", WR.window_resize(ee, img, tokens, 4, 2)[0]),
                        ("window_pool", WO.window_pool(ee, img, tokens, 8).pooled),
                        ("naive_attention", WT.naive_attention(W, img, img, img, tokens=49, dim=32)["output"]),
                        ("shifted_window_mask", WT.shifted_window_mask(ee, tokens.coarse(7)))):
        assert len(_assets(value)) == 0 == _metadata(name).asset_arrays()


def _metadata(name):
    for module in MODULES.values():
        if name in module.METADATA:
            return module.METADATA[name]
    raise KeyError(name)


# ---------------------------------------------------------------------------
# Layer rule
# ---------------------------------------------------------------------------
STDLIB_OR_THIRD = ("ee", "numpy")
TORCH_READERS = {"tensor_ee/blocks/window/resample.py", "tensor_ee/blocks/window/pooling.py"}
LAYER_ALLOWED = ("tensor_ee.core", "tensor_ee.layouts", "tensor_ee.weights", "tensor_ee.blocks.window")


def _imports(path: Path):
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


def _within(name, prefixes):
    return any(name == p or name.startswith(p + ".") for p in prefixes)


def test_window_blocks_respect_the_layer_rule():
    files = sorted((ROOT / "tensor_ee/blocks/window").glob("*.py"))
    assert {p.name for p in files} == {"__init__.py", "linear.py", "activations.py", "normalization.py",
                                       "attention.py", "conv.py", "resample.py", "pooling.py", "patch.py",
                                       "merge.py", "segformer.py", "swin.py", "upernet.py",
                                       "swin_upernet_config.py"}
    violations, torch_users = [], set()
    for path in files:
        relative = str(path.relative_to(ROOT))
        for name in _imports(path):
            top = name.split(".")[0]
            if top == "torch":
                torch_users.add(relative)        # PyTorch's own resize/pooling coefficients only
                continue
            if not (top in sys.stdlib_module_names or top == "__future__" or _within(name, STDLIB_OR_THIRD)
                    or _within(name, LAYER_ALLOWED)):
                violations.append((relative, name))
    assert violations == []
    assert torch_users == TORCH_READERS


FORBIDDEN_FOR_WINDOW_BLOCKS = ("ee_attention", "upernet", "swin_", "tensor_ee.blocks.raster", "tensor_ee.converter",
                               "tensor_ee.models", "tensor_ee.swin", "tensor_ee._swin_parameters")


def test_window_blocks_import_no_raster_block_converter_model_or_research():
    # The static layer rule, stated explicitly: no blocks/window module imports these.
    found = []
    for path in sorted((ROOT / "tensor_ee/blocks/window").glob("*.py")):
        for name in _imports(path):
            if name.startswith(("ee_attention", "upernet", "swin_")) or _within(name, (
                    "tensor_ee.blocks.raster", "tensor_ee.converter", "tensor_ee.models", "tensor_ee.swin",
                    "tensor_ee._swin_parameters", "tensor_ee.compiler", "tensor_ee.semantics",
                    "tensor_ee.primitives", "tensor_ee.outputs")):
                found.append((str(path.relative_to(ROOT)), name))
    assert found == []


def test_importing_the_window_blocks_loads_no_research_module():
    # Importing any tensor_ee submodule runs tensor_ee/__init__.py, which loads the public
    # CNN API (and, from L03 on, tensor_ee.blocks.raster). So the baseline is a plain
    # `import tensor_ee`; the window blocks must add nothing forbidden to it, and the
    # baseline itself must load no research module.
    code = (
        "import sys\n"
        f"forbidden = {FORBIDDEN_FOR_WINDOW_BLOCKS!r}\n"
        "def bad(names, prefixes):\n"
        "    return sorted({m for m in names for p in prefixes\n"
        "                   if (m.startswith(p) if p.endswith('_') else m == p or m.startswith(p + '.'))})\n"
        "import tensor_ee\n"
        "baseline = set(sys.modules)\n"
        "research = bad(baseline, ('ee_attention', 'upernet', 'swin_'))\n"
        "assert not research, ('import tensor_ee loads research modules', research)\n"
        f"assert not bad(baseline, ({LIB!r},)), 'import tensor_ee already loads the window blocks'\n"
        "import " + ", ".join(f"{LIB}.{m}" for m in (*MODULES, *SETTINGS_MODULES)) + "\n"
        "added = set(sys.modules) - baseline\n"
        "wrong = bad(added, forbidden)\n"
        "assert not wrong, ('the window blocks load', wrong)\n")
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, env={**os.environ, "PYTHONPATH": str(ROOT)})


# ---------------------------------------------------------------------------
# Public API unchanged
# ---------------------------------------------------------------------------
def test_public_api_is_unchanged():
    assert len(tensor_ee.__all__) == 31 and all(hasattr(tensor_ee, name) for name in tensor_ee.__all__)
    from tensor_ee import swin
    assert swin.__all__ == ["ServerArray", "SwinTParameters", "load_swin_t224_parameters",
                            "load_swin_t224_model", "SwinT224Adapter"]
    assert swin._SKIP_BANDS and swin._C4_BANDS
    assert swin.PE is WP and swin.PE.patch_embed is WP.patch_embed and swin.PE.BANDS == ("VV", "VH")
