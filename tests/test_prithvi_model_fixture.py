"""Composition fixture coverage and sensitivity to parameter/index permutations."""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from reference.prithvi_sen1floods11.reference import forward_torch
from tensor_ee.models.prithvi_sen1floods11.configuration import parameter_shapes
from tools.fixtures import prithvi_model as F


class _Array:
    """Evaluate only the coefficient-construction expressions used by the fixture."""

    def __init__(self, value):
        self.values = np.asarray(value.values if isinstance(value, _Array) else value)

    def float(self):
        return _Array(self.values.astype(np.float32))

    def double(self):
        return _Array(self.values.astype(np.float64))

    def reshape(self, shape):
        return _Array(self.values.reshape(shape))

    def repeat(self, axis, count):
        return _Array(np.concatenate([self.values] * count, axis=axis))

    def divide(self, value):
        return _Array(self.values / value)

    def add(self, value):
        return _Array(self.values + value)

    def multiply(self, value):
        return _Array(self.values * value.values)


@pytest.fixture(scope="module")
def crop():
    fixture = F.fixtures()[0]
    data = fixture.data(fixture.seed)
    image, state = F._expanded(data)
    baseline = forward_torch(image, state, heads=3, depth=12)[2::4, 2::4]
    return fixture, data, image, state, baseline


def test_composition_coefficients_distinguish_all_rows_and_columns(crop):
    _, _, image, state, _ = crop
    assert np.unique(image.reshape(-1, 6), axis=1).shape[1] == 6
    assert np.unique(image.reshape(-1, 6), axis=0).shape[0] == 224 * 224
    for name, values in state.items():
        if name == "backbone.pos_embed":
            matrix = values[0]
            assert np.unique(matrix, axis=0).shape[0] == 197
            assert np.unique(matrix, axis=1).shape[1] == matrix.shape[1]
        elif values.ndim == 1:
            assert np.unique(values).size == values.size, name
        else:
            matrix = values.reshape(values.shape[0], -1)
            assert np.unique(matrix, axis=0).shape[0] == matrix.shape[0], name
            assert np.unique(matrix, axis=1).shape[1] == matrix.shape[1], name


def test_composition_server_coefficients_equal_reference_coefficients(crop):
    _, data, _, state, _ = crop
    ee = SimpleNamespace(Array=_Array,
                         List=SimpleNamespace(sequence=lambda start, end: range(start, end + 1)))
    server = F._server_state(ee, data)
    for name, values in state.items():
        actual = server[name].array.values.reshape(values.shape)
        np.testing.assert_array_equal(actual.view(np.uint32), values.view(np.uint32), err_msg=name)


def _matrix_cases():
    return [(name, axis) for name, shape in parameter_shapes(F._config()).items()
            if len(shape) > 1 and name not in ("backbone.pos_embed", "backbone.cls_token")
            for axis in (0, 1)]


def _delta(crop, name, values):
    _, _, image, state, baseline = crop
    altered = dict(state)
    altered[name] = values
    result = forward_torch(image, altered, heads=3, depth=12)[2::4, 2::4]
    return float(np.max(np.abs(result.astype(np.float64) - baseline.astype(np.float64))))


@pytest.mark.parametrize("name,axis", _matrix_cases(),
                         ids=lambda item: str(item))
def test_composition_fixture_detects_weight_index_permutations(crop, name, axis):
    fixture, _, _, state, _ = crop
    values = state[name]
    matrix = values.reshape(values.shape[0], -1)
    size = matrix.shape[axis]
    # Preserve row/column index residues to expose symmetric channel-group mistakes.
    period = math.gcd(size, 6 if axis == 0 else 8)
    shift = period if size > period else 1
    changed = np.roll(matrix, shift, axis=axis).reshape(values.shape).copy()
    delta = _delta(crop, name, changed)
    print(f"sensitivity {name} {'rows' if axis == 0 else 'columns'} max_abs={delta:.17g}")
    assert delta > fixture.tolerance


@pytest.mark.parametrize("axis,shift,label", [(1, 1, "band"), (2, 1, "ky"), (3, 8, "kx-plus-eight")])
def test_composition_fixture_detects_patch_column_order(crop, axis, shift, label):
    fixture, _, _, state, _ = crop
    name = "backbone.patch_embed.proj.weight"
    source = state[name]
    matrix = source.reshape(F._config().width, 6, 16, 16)
    # Permuting these kernel columns is the same error as permuting gathered input columns.
    changed = np.roll(matrix, shift, axis=axis).reshape(source.shape).copy()
    delta = _delta(crop, name, changed)
    print(f"sensitivity patch {label} max_abs={delta:.17g}")
    assert delta > fixture.tolerance


@pytest.mark.parametrize("left,right,label", [(0, 1, "class-spatial"), (1, 9, "spatial-mod-eight")])
def test_composition_fixture_detects_positional_token_order(crop, left, right, label):
    fixture, _, _, state, _ = crop
    name = "backbone.pos_embed"
    order = np.arange(197)
    order[left], order[right] = order[right], order[left]
    changed = state[name][:, order, :].copy()
    delta = _delta(crop, name, changed)
    print(f"sensitivity position {label} max_abs={delta:.17g}")
    assert delta > fixture.tolerance



def test_composition_fixture_detects_same_phase_patch_token_order(crop):
    fixture, _, image, state, baseline = crop
    order = np.arange(14)
    order[0], order[2] = order[2], order[0]
    changed = image.reshape(14, 16, 224, 6)[order].reshape(224, 224, 6).copy()
    expected = forward_torch(changed, state, heads=3, depth=12)[2::4, 2::4]
    delta = float(np.max(np.abs(expected.astype(np.float64) - baseline.astype(np.float64))))
    print(f"sensitivity patch-token same-phase-rows max_abs={delta:.17g}")
    assert delta > fixture.tolerance


def test_composition_fixture_keeps_compatibility_field_without_block_parts():
    fixture = F.fixtures()[0]
    assert fixture.id == F.COMPOSITION_FIXTURE_ID
    assert fixture.block == "concat_rows"
    assert fixture.parts == ()
    assert "not individual-block evidence" in fixture.notes
    assert "never submitted" not in F.__doc__
