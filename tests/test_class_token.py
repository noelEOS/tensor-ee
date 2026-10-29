"""Class-row transport and split-query attention without cloud calls."""
from __future__ import annotations

import ee
from ee.apitestcase import ApiTestCase
import numpy as np
import pytest
import torch

from tensor_ee.blocks.window.attention import (
    class_token_attention, class_token_attention_values, class_token_attention_arrays,
    naive_attention, split_heads, METADATA as ATTENTION_METADATA,
)
from tensor_ee.blocks.window.merge import concat_rows, concat_rows_arrays, concat_rows_values, METADATA
from tensor_ee.core.metadata import NOT_LIVE
from tensor_ee.core.ops import ValueOps, WindowOps
from tensor_ee.core.grid import TokenGrid
from tensor_ee.core.tensor import WindowTensor
from reference.class_token import class_token_attention as mirror_attention, concat_rows as mirror_concat, expand_qkv
from reference.segformer_b0.blocks import NUMPY_EE, Operand
from tools.fixtures.class_token import fixtures


def test_row_order_and_bits():
    first = np.array([[0.0, -0.0, 11]], np.float32)
    rest = np.arange(588, dtype=np.float32).reshape(196, 3)
    actual = concat_rows(ValueOps(NUMPY_EE), [Operand(first), Operand(rest)],
                         rows=(1, 196), channels=3).values
    expected = mirror_concat([first, rest])
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    assert METADATA["concat_rows"].validated == NOT_LIVE
    assert concat_rows_values(rows=(1, 196), channels=3) == concat_rows_arrays() == 0


@pytest.mark.parametrize("parts,rows,channels", [([], (), 3), ([1], (1, 2), 3),
    ([1], (0,), 3), ([1], (True,), 3), ([1], (1,), False), ([1], (1,), 0), ([1], (1.0,), 3)])
def test_refuses_invalid_shape_description(parts, rows, channels):
    with pytest.raises(ValueError):
        concat_rows(None, parts, rows=rows, channels=channels)


def test_window_row_concat_serializes_without_cast_or_arithmetic():
    case = ApiTestCase()
    case.setUp()
    try:
        ops = WindowOps(ee)
        cls, patch = ee.Image("class"), ee.Image("spatial")
        actual = concat_rows(ops, [cls, patch], rows=(1, 196), channels=64)
        assert actual.serialize() == cls.arrayCat(patch, 0).serialize()
    finally:
        case.tearDown()
        ee.Reset()


def test_both_query_parts_use_class_key_and_value():
    torch.set_num_threads(2)
    rng = np.random.default_rng(53)
    dim, heads = 24, 3
    patch = rng.normal(0, 0.5, (196, dim * 3)).astype(np.float32)
    cls = rng.normal(0, 0.5, (1, dim * 3)).astype(np.float32)
    ops = ValueOps(NUMPY_EE)
    spatial, class_output = class_token_attention(
        ops, Operand(patch), Operand(cls), dim=dim, heads=heads, window=14,
    )
    actual = {"patch": spatial.values, "cls": class_output.values}
    expected = mirror_attention(patch, cls, dim=dim, heads=heads)
    qkv = torch.from_numpy(np.concatenate([cls, patch])).reshape(197, 3, heads, 8).permute(1, 2, 0, 3)
    q, k, v = qkv.unbind(0)
    reference = ((((q @ k.transpose(-2, -1)) * (8 ** -0.5)).softmax(-1) @ v)
                 .permute(1, 0, 2).reshape(197, dim).numpy())
    for name, torch_part in (("cls", reference[:1]), ("patch", reference[1:])):
        output = actual[name]
        np.testing.assert_allclose(output, expected[name], rtol=0, atol=1e-7)
        np.testing.assert_allclose(output, torch_part, rtol=0, atol=1e-4)
    changed = cls.copy()
    changed[:, 2 * dim:] += 1
    with_changed_class = mirror_attention(patch, changed, dim=dim, heads=heads)
    assert np.max(np.abs(with_changed_class["patch"] - expected["patch"])) > 1e-4
    assert np.max(np.abs(with_changed_class["cls"] - expected["cls"])) > 1e-4


@pytest.mark.parametrize("fixture", fixtures(), ids=lambda f: f.id)
def test_fixture_mirror_against_plain_torch(fixture):
    data = fixture.data(fixture.seed)
    expected = fixture.reference(data)
    actual = fixture.mirror.fn(data)
    for key in expected:
        if fixture.tolerance == 0:
            np.testing.assert_array_equal(expected[key].view(np.uint32), actual[key].view(np.uint32))
        else:
            np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=fixture.tolerance)


def test_compact_heads_rotate_every_qkv_component():
    base = np.arange(192, dtype=np.float32).reshape(1, 192)
    expanded = expand_qkv(base, 12, 64)
    assert expanded.shape == (1, 2304)
    for component in range(3):
        for h in range(12):
            for channel in range(64):
                assert expanded[0, component * 768 + h * 64 + channel] == (
                    component * 64 + (channel + h) % 64)


def test_class_attention_has_no_coefficients_and_no_live_record():
    entry = ATTENTION_METADATA["class_token_attention"]
    assert entry.validated == NOT_LIVE
    assert class_token_attention_values() == class_token_attention_arrays() == 0


@pytest.mark.parametrize("window,dim,heads", [
    (False, 12, 3), (0, 12, 3), (3.0, 12, 3),
    (3, False, 3), (3, 0, 3), (3, 12, 0), (3, 13, 3),
])
def test_class_attention_refuses_invalid_dimensions(window, dim, heads):
    with pytest.raises(ValueError):
        class_token_attention(None, None, None, dim=dim, heads=heads, window=window)


def test_class_attention_preserves_the_two_query_graphs():
    case = ApiTestCase()
    case.setUp()
    try:
        ops = WindowOps(ee)
        spatial, cls = ee.Image("spatial_qkv"), ee.Image("class_qkv")
        actual = class_token_attention(ops, spatial, cls, dim=12, heads=3, window=3)
        spatial_heads = split_heads(ops, spatial, 12, 3)
        class_heads = split_heads(ops, cls, 12, 3)
        expected = [[], []]
        for head in range(3):
            keys, values = [
                concat_rows(ops, [class_heads[f"{name}_h{head}"], spatial_heads[f"{name}_h{head}"]],
                            rows=(1, 9), channels=4)
                for name in ("k", "v")
            ]
            for index, part in enumerate((spatial_heads, class_heads)):
                result = naive_attention(ops, part[f"q_h{head}"], keys, values,
                                         tokens=10, dim=4, score_scale=0.5)
                expected[index].append(result["output"])
        assert [value.serialize() for value in actual] == [
            ops.cat_cols(parts).serialize() for parts in expected]
    finally:
        case.tearDown()
        ee.Reset()


@pytest.mark.parametrize("prefix,eps,expected", [
    ("neck.fpn1.1.ln", 1e-5, "spatial"),
    ("neck.fpn2.1.ln", 1e-5, "spatial"),
    ("backbone.norm", 1e-6, "token"),
])
def test_model_norm_dispatch_uses_module_path(monkeypatch, prefix, eps, expected):
    from tensor_ee.models.prithvi_sen1floods11 import model

    backend = model.LibraryBackend.__new__(model.LibraryBackend)
    backend.ops = object()
    backend.norms = {(prefix, 49): (object(), object())}
    tensor = WindowTensor(object(), TokenGrid(7, 7), 4, 7)
    calls = []

    def token(*args, **kwargs):
        calls.append(("token", kwargs["eps"]))
        return "token"

    def spatial(*args, **kwargs):
        calls.append(("spatial", kwargs["eps"]))
        return "spatial"

    monkeypatch.setattr(model, "layer_norm", token)
    monkeypatch.setattr(model, "layer_norm2d", spatial)
    assert backend.norm(tensor, prefix, eps).image == expected
    assert calls == [(expected, eps)]
