"""Class-row transport and split-query attention without cloud calls."""
from __future__ import annotations

import ee
from ee.apitestcase import ApiTestCase
import numpy as np
import pytest
import torch

from tensor_ee.blocks.window.attention import naive_attention, split_heads
from tensor_ee.blocks.window.merge import concat_rows, concat_rows_arrays, concat_rows_values, METADATA
from tensor_ee.core.metadata import NOT_LIVE
from tensor_ee.core.ops import ValueOps, WindowOps
from reference.class_token import attention, concat_rows as mirror_concat, expand_qkv
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
    ph = split_heads(ops, Operand(patch), dim, heads)
    ch = split_heads(ops, Operand(cls), dim, heads)
    actual = {"patch": [], "cls": []}
    for h in range(heads):
        k, v = [concat_rows(ops, [ch[f"{n}_h{h}"], ph[f"{n}_h{h}"]], rows=(1, 196), channels=8)
                for n in ("k", "v")]
        for part, q in (("patch", ph[f"q_h{h}"]), ("cls", ch[f"q_h{h}"])):
            actual[part].append(naive_attention(ops, q, k, v, tokens=197, dim=8,
                                                 score_scale=8 ** -0.5)["output"].values)
    expected = attention(patch, cls, dim=dim, heads=heads)
    qkv = torch.from_numpy(np.concatenate([cls, patch])).reshape(197, 3, heads, 8).permute(1, 2, 0, 3)
    q, k, v = qkv.unbind(0)
    reference = ((((q @ k.transpose(-2, -1)) * (8 ** -0.5)).softmax(-1) @ v)
                 .permute(1, 0, 2).reshape(197, dim).numpy())
    for name, torch_part in (("cls", reference[:1]), ("patch", reference[1:])):
        output = np.concatenate(actual[name], 1)
        np.testing.assert_allclose(output, expected[name], rtol=0, atol=1e-7)
        np.testing.assert_allclose(output, torch_part, rtol=0, atol=1e-4)
    changed = cls.copy()
    changed[:, 2 * dim:] += 1
    with_changed_class = attention(patch, changed, dim=dim, heads=heads)
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
