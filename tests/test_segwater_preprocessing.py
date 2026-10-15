"""V3 lattice, float32 arithmetic, padding and native graph contract checks."""

import json

import ee
from ee.apitestcase import ApiTestCase
import numpy as np
import pytest

from tensor_ee.segwater_preprocessing import (
    MEANS, STDS, preprocess_db_ee, preprocess_db_numpy, preprocessing_metadata,
    quantize_db_ee, quantize_db_numpy, round_ties_to_even_ee,
    training_mean_padding_numpy,
)


def source_formula(data):
    """Literal operation order from the audited run_inference.py function."""
    data = data.astype(np.float32, copy=False)
    offset_db = np.float32(55.0)
    step_db = np.float32(0.05)
    code = np.clip(np.round((data + offset_db) / step_db), 1, 65535)
    data = code * step_db - offset_db
    means = np.array(MEANS, dtype=np.float32)
    stds = np.array(STDS, dtype=np.float32)
    return (data - means[:, None, None]) / stds[:, None, None]


def test_v3_constants_and_code1_floor_are_exact_float32_values():
    actual = quantize_db_numpy([-100, -63.5, -55, -54.99, -54.95, 4000])
    expected = np.array([-54.95000076293945] * 5 + [3221.75], np.float32)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32
    metadata = preprocessing_metadata()
    assert metadata["means_decimal"] == [-15.435106, -24.856323]
    assert metadata["stds_decimal"] == [6.517790, 9.117346]
    assert metadata["below_minus55_override"] is False


def test_exact_float32_half_steps_use_even_codes():
    db = np.array([-54.875, -29.975000381469727, -29.924999237060547,
                   -4.974998474121094, -4.924999237060547], np.float32)
    scaled = (db + np.float32(55)) / np.float32(0.05)
    np.testing.assert_array_equal(scaled, [2.5, 500.5, 501.5, 1000.5, 1001.5])
    even_codes = np.array([2, 500, 502, 1000, 1002], np.float32)
    expected = even_codes * np.float32(0.05) - np.float32(55)
    np.testing.assert_array_equal(quantize_db_numpy(db), expected)


def test_sub_floor_policy_has_no_extra_minus55_override():
    below = np.nextafter(np.float32(-55), np.float32(-np.inf))
    above = np.nextafter(np.float32(-55), np.float32(np.inf))
    np.testing.assert_array_equal(quantize_db_numpy([below, -55, above]),
                                  np.full(3, np.float32(-54.95)))


def test_arithmetic_matches_audited_source_bit_for_bit_over_edges_and_random_db():
    rng = np.random.default_rng(912)
    data = rng.uniform(-100, 35, size=(2, 57, 23)).astype(np.float64)
    data[:, 0, :8] = [-100, -55, -54.875, -29.975000381469727,
                      -29.924999237060547, 0, 3221.72509765625, 4000]
    actual = preprocess_db_numpy(data)
    np.testing.assert_array_equal(actual.view(np.uint32), source_formula(data).view(np.uint32))
    assert actual.dtype == np.float32


def test_channel_axis_supports_nchw_without_swapping_vv_and_vh():
    chw = np.array([[[-20, -10]], [[-30, -25]]], np.float32)
    nchw = np.stack([chw, chw + np.float32(2)])
    actual = preprocess_db_numpy(nchw, channel_axis=1)
    expected = np.stack([source_formula(chw), source_formula(chw + np.float32(2))])
    np.testing.assert_array_equal(actual, expected)


def test_training_mean_padding_is_requantized_and_not_exactly_zero():
    actual = training_mean_padding_numpy((2, 3, 4))
    expected_z = np.array([-0.002285204827785492, 0.000693497946485877], np.float32)
    np.testing.assert_array_equal(actual, np.broadcast_to(expected_z[:, None, None], actual.shape))
    assert np.all(np.abs(actual) < np.float32(0.004))
    assert np.all(actual != 0)


def test_mask_preservation_and_training_mean_fill_are_explicit_per_channel():
    values = np.array([[[-20, np.nan, 0]], [[-30, -25, -55]]], np.float32)
    mask = np.array([[[False, True, False]], [[False, False, True]]])
    data = np.ma.array(values, mask=mask)
    preserved = preprocess_db_numpy(data)
    assert np.ma.isMaskedArray(preserved)
    np.testing.assert_array_equal(preserved.mask, mask)
    filled = preprocess_db_numpy(data, masked_policy="training_mean")
    means = np.array(MEANS, np.float32)[:, None, None]
    np.testing.assert_array_equal(filled, source_formula(np.where(mask, means, values)))
    assert not np.ma.isMaskedArray(filled)
    # Numeric 0 dB remains an observed pixel; neither 0 nor -55 implies NoData.
    assert preserved[0, 0, 2] == preprocess_db_numpy(np.zeros((2, 1, 1), np.float32))[0, 0, 0]


def test_low_level_nonfinite_reference_behavior_matches_numpy_source():
    result = quantize_db_numpy([np.nan, -np.inf, np.inf])
    assert np.isnan(result[0])
    np.testing.assert_array_equal(result[1:], [-54.95000076293945, 3221.75])


@pytest.mark.parametrize("data,kwargs", [
    (1.0, {}), (np.ones((3, 2, 2)), {}), (np.ones((2, 2)), {"channel_axis": 2}),
    (np.ones((2, 2)), {"masked_policy": "zero"}),
])
def test_invalid_numpy_contract_is_rejected(data, kwargs):
    with pytest.raises(ValueError):
        preprocess_db_numpy(data, **kwargs)


@pytest.mark.parametrize("shape", [(2, 0, 1), (2, -1, 1), (2, 1.5, 1), (3, 1, 1)])
def test_invalid_padding_contract_is_rejected(shape):
    with pytest.raises(ValueError):
        training_mean_padding_numpy(shape)


def algorithm_names(value):
    if isinstance(value, dict):
        names = [value["functionName"]] if "functionName" in value else []
        return names + [name for item in value.values() for name in algorithm_names(item)]
    if isinstance(value, list):
        return [name for item in value for name in algorithm_names(item)]
    return []


class TestSegWaterGraph(ApiTestCase):
    def setUp(self):
        super().setUp()
        self.image = ee.Image.constant([-20, -30, 99]).rename(["vv", "vh", "extra"])

    def test_graph_has_explicit_float32_ties_to_even_and_channel_selection(self):
        result = preprocess_db_ee(self.image, ("vv", "vh"))
        names = algorithm_names(json.loads(result.serialize()))
        for name in ("Image.select", "Image.floor", "Image.mod", "Image.clamp",
                     "Image.toFloat", "Image.subtract", "Image.divide"):
            self.assertIn(name, names)
        self.assertNotIn("Image.round", names)
        self.assertNotIn("Image.unmask", names)
        self.assertFalse(any("predict" in name.lower() for name in names))

    def test_training_mean_fill_happens_before_quantization_and_expands_footprint(self):
        serialized = preprocess_db_ee(self.image, ("vv", "vh"),
                                      masked_policy="training_mean").serialize()
        self.assertIn("Image.unmask", serialized)
        self.assertIn('"sameFootprint"', serialized)
        self.assertIn('"constantValue": false', serialized)

    def test_standalone_quantizer_and_rounding_are_native_images(self):
        self.assertIn("Image.clamp", quantize_db_ee(self.image).serialize())
        self.assertIn("Image.mod", round_ties_to_even_ee(self.image).serialize())

    def test_invalid_band_and_mask_policies_are_rejected(self):
        for bands in (("vv",), ("vv", "vv"), ("vv", "vh", "extra")):
            with self.assertRaises(ValueError):
                preprocess_db_ee(self.image, bands)
        with self.assertRaises(ValueError):
            preprocess_db_ee(self.image, masked_policy="zero")
