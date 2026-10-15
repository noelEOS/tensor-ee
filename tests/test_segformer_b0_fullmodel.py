"""Lightweight primitive/contract checks; full-chip preflight is lead-run."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference.segformer_b0 import full_model as mirror
from tools.fixtures import segformer_b0_validation as reference


def test_independent_schema_names_and_shapes():
    assert len(mirror.expected_state_shapes()) == 192
    assert mirror.expected_state_shapes()["encoder.patch_embed4.proj.weight"] == (256, 160, 3, 3)
    assert mirror.expected_state_shapes()[mirror.COUNTER] == ()
    assert set(mirror.CUT_INPUTS) == {"stage1a", "stage1b", "stage2", "stage3", "stage4", "decoder"}


def test_independent_schema_matches_fixed_native_contract():
    pytest.importorskip("torch")
    from tensor_ee.models.segformer_b0.configuration import parameter_shapes

    assert mirror.expected_state_shapes() == parameter_shapes()


@pytest.mark.parametrize("cut,name", [("stage1a", "input"), ("stage1b", "S1B0"), ("stage2", "C1")])
def test_incoming_arrays_refuse_wrong_shape_dtype_nonfinite_and_names(cut, name):
    with pytest.raises(ValueError):
        mirror.validate_cut_inputs(cut, {name: np.zeros((1, 1, 1), np.float32)})
    with pytest.raises(ValueError):
        mirror.validate_cut_inputs(cut, {name: np.zeros(mirror.SHAPES[name], np.float64)})
    values = np.zeros(mirror.SHAPES[name], np.float32)
    values.flat[0] = np.nan
    with pytest.raises(ValueError):
        mirror.validate_cut_inputs(cut, {name: values})
    with pytest.raises(ValueError):
        mirror.validate_cut_inputs(cut, {})
    with pytest.raises(ValueError):
        mirror.validate_cut_inputs("different_model", {name: values})


def test_missing_state_refused_before_reference_construction():
    with pytest.raises(ValueError, match="inference names"):
        reference.make_reference({})
    with pytest.raises(ValueError):
        mirror.mirror_full({}, np.zeros((1, 1, 2), np.float32))


def test_batch_norm_nontrivial_statistics_matches_torch_without_state_mutation():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    rng = np.random.default_rng(20261014)
    x = rng.normal(size=(3, 4, 5)).astype(np.float32)
    gamma = rng.normal(size=5).astype(np.float32)
    beta = rng.normal(size=5).astype(np.float32)
    mean = rng.normal(size=5).astype(np.float32)
    variance = rng.uniform(0.01, 1.5, size=5).astype(np.float32)
    arrays = [x, gamma, beta, mean, variance]
    snapshots = [a.view(np.uint32).copy() for a in arrays]
    expected = F.batch_norm(torch.from_numpy(x).permute(2, 0, 1)[None],
                            torch.from_numpy(mean), torch.from_numpy(variance),
                            torch.from_numpy(gamma), torch.from_numpy(beta), training=False, eps=1e-5)
    actual = mirror.batch_norm(x, gamma, beta, mean, variance)
    assert actual.dtype == np.float32
    assert np.max(np.abs(actual.astype(np.float64) - reference._hwc(expected))) <= 1e-4
    for array, snapshot in zip(arrays, snapshots):
        np.testing.assert_array_equal(array.view(np.uint32), snapshot)


@pytest.mark.parametrize("n,factor,align", [(7, 2, False), (7, 4, False), (7, 8, False), (56, 4, True)])
def test_independent_coordinate_coefficients_match_torch_one_hot(n, factor, align):
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    dtype = torch.float32 if align else torch.float64
    expected = F.interpolate(torch.eye(n, dtype=dtype)[:, None, None, :], size=(1, n * factor),
                             mode="bilinear", align_corners=align)[:, 0, 0].T.double().numpy()
    low, high, w0, w1 = mirror._axis_coefficients(n, n * factor, align)
    actual = np.zeros((n * factor, n), np.float64)
    rows = np.arange(n * factor)
    np.add.at(actual, (rows, low), w0)
    np.add.at(actual, (rows, high), w1)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("align", [False, True])
def test_independent_resize_matches_small_torch_map_and_preserves_corners(align):
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    x = np.random.default_rng(20261014).uniform(-2, 2, size=(7, 7, 3)).astype(np.float32)
    expected = F.interpolate(torch.from_numpy(x).permute(2, 0, 1)[None], size=(28, 28),
                             mode="bilinear", align_corners=align)
    actual = mirror.bilinear_resize(x, (28, 28), align_corners=align)
    assert np.max(np.abs(actual.astype(np.float64) - reference._hwc(expected))) <= 1e-4
    np.testing.assert_array_equal(actual[0, 0], x[0, 0])
    np.testing.assert_array_equal(actual[-1, -1], x[-1, -1])


def test_residual_rounding_and_bitwise_traversal_gate():
    a = np.array([2 ** 24, -2 ** 24, 1], np.float32)
    b = np.array([1, 3, -1], np.float32)
    np.testing.assert_array_equal(mirror.residual(a, b).view(np.uint32), (a + b).view(np.uint32))
    original = {"C1": np.array([0.0, 1.0], np.float32)}
    reference._assert_same_bits(original, {"C1": original["C1"].copy()})
    with pytest.raises(AssertionError, match="bitwise"):
        reference._assert_same_bits(original, {"C1": np.array([-0.0, 1.0], np.float32)})


def test_expanded_decoder_calls_installed_modules_in_original_order(monkeypatch):
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    calls = []
    interpolate = F.interpolate

    def resize(x, *, size, mode, align_corners):
        calls.append(("resize", size, mode, align_corners))
        return interpolate(x, size=size, mode=mode, align_corners=align_corners)

    class Module:
        def __init__(self, name):
            self.name = name

        def __call__(self, x):
            calls.append((self.name, float(x.flatten()[0])))
            return x

    model = SimpleNamespace(
        decoder=SimpleNamespace(mlp_stage=[Module(f"projection{i}") for i in range(4)],
                                fuse_stage=[Module("fuse_conv"), Module("bn"), Module("relu")]),
        segmentation_head=[Module("head_conv"), Module("head_resize"), Module("activation")])
    inputs = {f"C{i}": torch.full((1, 2, 2, 2), float(i)) for i in range(1, 5)}
    monkeypatch.setattr(F, "interpolate", resize)
    actual = reference._expanded_decoder(model, inputs)
    assert calls[:8] == sum(([ (f"projection{i}", float(4 - i)),
                              ("resize", (56, 56), "bilinear", False)] for i in range(4)), [])
    assert [name for name, _ in calls[8:]] == ["fuse_conv", "bn", "relu", "head_conv", "head_resize", "activation"]
    assert actual.shape == (1, 8, 56, 56)
    assert [float(actual[0, i * 2, 0, 0]) for i in range(4)] == [4, 3, 2, 1]


def test_mirror_split_routes_complete_block_outputs_and_final_norm(monkeypatch):
    calls = []
    state = {"encoder.norm1.weight": "gamma", "encoder.norm1.bias": "beta"}
    monkeypatch.setattr(mirror, "validate_state", lambda actual: None)
    monkeypatch.setattr(mirror, "validate_cut_inputs", lambda cut, inputs: inputs)

    def overlap(x, actual_state, prefix, **settings):
        calls.append(("overlap", x, prefix, settings))
        return {"out": "embedded"}

    def block(x, actual_state, prefix, **settings):
        calls.append(("block", x, prefix, settings))
        return "block_output"

    def norm(x, gamma, beta, eps):
        calls.append(("norm", x, gamma, beta, eps))
        return "normalized"

    monkeypatch.setattr(mirror, "overlap", overlap)
    monkeypatch.setattr(mirror, "transformer_block", block)
    monkeypatch.setattr(mirror, "layer_norm", norm)
    assert mirror.mirror_cut("stage1a", state, {"input": "original"}) == "block_output"
    assert calls == [("overlap", "original", "encoder.patch_embed1", {"kernel": 7, "stride": 4}),
                     ("block", "embedded", "encoder.block1.0", {"heads": 1, "ratio": 8})]
    calls.clear()
    assert mirror.mirror_cut("stage1b", state, {"S1B0": "actual_saved"}) == "normalized"
    assert calls == [("block", "actual_saved", "encoder.block1.1", {"heads": 1, "ratio": 8}),
                     ("norm", "block_output", "gamma", "beta", 1e-6)]


def test_resize_rounds_after_both_axes_once():
    # Approved false-alignment factor2 yields 0.75/0.25 on each axis here.
    # Rounding the y intermediate loses each 0.25; only a final cast keeps it.
    source = np.array([[100000000, -300000000], [1, 1]], dtype=np.float32)[:, :, None]
    actual = mirror.bilinear_resize(source, (4, 4), align_corners=False)
    assert actual[1, 1, 0] == np.float32(0.25)
    vertical = (source[0].astype(np.float64) * 0.75 + source[1].astype(np.float64) * 0.25).astype(np.float32)
    wrongly_rounded = np.float32(vertical[0, 0] * np.float64(0.75) + vertical[1, 0] * np.float64(0.25))
    assert wrongly_rounded == 0 and actual[1, 1, 0] != wrongly_rounded
