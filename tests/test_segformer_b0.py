"""SegFormer primitives: offline graphs, arithmetic, refusals and transport.

No Earth Engine calls or whole-model forward. Spatial halo/im2col are the
accepted existing library operators; arithmetic adapter tests supply independent
F.unfold neighbourhoods and execute the new block's actual arithmetic/casts.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import ee
from ee.apitestcase import ApiTestCase
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference.segformer_b0 import blocks as M
from tensor_ee.blocks.window import attention as A, conv as C, segformer as S
from tensor_ee.core.errors import UnsupportedOperation
from tensor_ee.core.grid import TokenGrid
from tensor_ee.core.ops import ValueOps, WindowOps
from tensor_ee.weights.constants import RowMismatch, ServerArray, WindowLayout


@pytest.fixture
def offline_ee():
    api = ApiTestCase()
    api.setUp()
    try:
        yield ee
    finally:
        api.tearDown()


def _numpy_depthwise(monkeypatch, x, weight, bias=None, server=False):
    """Execute the library's arithmetic on independent PyTorch-unfold tap rows."""
    torch = pytest.importorskip("torch")
    F = torch.nn.functional
    ops = ValueOps(M.NUMPY_EE)
    if server:
        weight = ServerArray(M.Operand(weight.reshape(weight.shape[0], 9)), (weight.shape[0], 9))
        if bias is not None:
            bias = ServerArray(M.Operand(bias), bias.shape)
    c = C.DepthwiseConv3x3Constants(ops, weight, bias)
    channels = x.shape[-1]
    padded = np.pad(x, ((1, 1), (1, 1), (0, 0)))
    monkeypatch.setattr(C, "halo", lambda ee, at, coarse, channels, **kwargs: padded[at[0]:at[0] + 9, at[1]:at[1] + 9])

    def columns(ee, tile, channels, k, **kwargs):
        tensor = torch.from_numpy(tile.copy()).permute(2, 0, 1)[None]
        # F.unfold uses (c,ky,kx); the library's existing im2col uses (ky,kx,c).
        rows = F.unfold(tensor, 3)[0].T.reshape(49, channels, 9).permute(0, 2, 1).reshape(49, 9 * channels)
        return M.Operand(rows.numpy())

    monkeypatch.setattr(C, "im2col", columns)
    out = np.empty_like(x)
    for y in range(0, x.shape[0], 7):
        for col in range(0, x.shape[1], 7):
            out[y:y + 7, col:col + 7] = C.depthwise_conv3x3_window(ops, (y, col), c, None).values.reshape(7, 7, channels)
    return out


@pytest.mark.parametrize("tap", range(9))
@pytest.mark.parametrize("server", [False, True])
def test_depthwise_token_channel_tap_ids_zero_edges_and_window_seams(monkeypatch, tap, server):
    x = np.arange(14 * 14 * 4, dtype=np.float32).reshape(14, 14, 4) + 1
    weight = np.zeros((4, 1, 3, 3), np.float32)
    weight[:, 0, tap // 3, tap % 3] = np.arange(1, 5, dtype=np.float32)
    bias = np.arange(4, dtype=np.float32)
    out = _numpy_depthwise(monkeypatch, x, weight, bias, server)
    padded = np.pad(x, ((1, 1), (1, 1), (0, 0)))
    expected = padded[tap // 3:tap // 3 + 14, tap % 3:tap % 3 + 14] * np.arange(1, 5, dtype=np.float32) + bias
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("channels", [4, 16, 128, 256, 640, 1024])
def test_depthwise_random_arithmetic_matches_mirror_and_pytorch(monkeypatch, bias, channels):
    from tools.fixtures.segformer_b0 import _dw_data, _dw_reference
    data = _dw_data(20261014, channels=channels, bias=bias)
    x, weight, b = data.inputs["x"], data.weights["weight"], data.weights.get("bias")
    out = _numpy_depthwise(monkeypatch, x, weight, b)
    np.testing.assert_array_equal(out, M.depthwise_conv3x3(x, weight, b))
    reference = _dw_reference(data)
    expected = reference["out"] if channels <= 128 else np.concatenate([reference[f"out_{s}"] for s in range(0, channels, 128)], axis=2)
    assert np.max(np.abs(out.astype(np.float64) - expected)) <= 1e-4


def test_depthwise_border_impulses_are_not_periodic(monkeypatch):
    x = np.zeros((14, 14, 4), np.float32)
    x[0, 0] = [1, 2, 3, 4]
    weight = np.ones((4, 1, 3, 3), np.float32)
    out = _numpy_depthwise(monkeypatch, x, weight)
    np.testing.assert_array_equal(out[:2, :2], np.broadcast_to(x[0, 0], (2, 2, 4)))
    assert not out[-1].any() and not out[:, -1].any()
    np.testing.assert_array_equal(out, M.depthwise_conv3x3(x, weight))


def test_depthwise_single_output_rounding_retains_cancellation_residue(monkeypatch):
    x = np.zeros((7, 7, 4), np.float32)
    # Ordered tap products: large, small, negative large; float32 partial sums lose small.
    x[0, 0], x[0, 1], x[0, 2] = 2 ** 24, 0.25, -(2 ** 24)
    out = _numpy_depthwise(monkeypatch, x, np.ones((4, 1, 3, 3), np.float32))
    np.testing.assert_array_equal(out[1, 1], np.full(4, 0.25, np.float32))
    assert np.float32(np.float32(2 ** 24) + np.float32(0.25)) + np.float32(-(2 ** 24)) == 0


def test_depthwise_double_products_retain_a_float32_product_rounding_residue(monkeypatch):
    x = np.zeros((7, 7, 4), np.float32)
    weight = np.zeros((4, 1, 3, 3), np.float32)
    x[0, 0] = 1 + 2 ** -23
    x[0, 1] = -(1 + 2 ** -22)
    weight[:, 0, 0, 0] = 1 + 2 ** -23
    weight[:, 0, 0, 1] = 1
    out = _numpy_depthwise(monkeypatch, x, weight)
    np.testing.assert_array_equal(out[1, 1], np.full(4, 2 ** -46, np.float32))
    # Dtype-faithful adapter: removing both explicit double casts loses this
    # exact product residue before the double tap reduction can recover it.
    with monkeypatch.context() as altered:
        altered.setattr(ValueOps, "to_double", lambda self, value: value)
        broken = _numpy_depthwise(altered, x, weight)
    np.testing.assert_array_equal(broken[1, 1], np.zeros(4, np.float32))


@pytest.mark.parametrize("option,value", [("k", 1), ("stride", 2), ("padding", 0), ("dilation", 2),
                                           ("multiplier", 2), ("k", True), ("stride", 1.0)])
def test_depthwise_unsupported_settings_are_refused(option, value):
    with pytest.raises(UnsupportedOperation):
        C.DepthwiseConv3x3Constants(ValueOps(M.NUMPY_EE), np.zeros((4, 1, 3, 3)), **{option: value})


@pytest.mark.parametrize("shape", [(4, 2, 3, 3), (4, 1, 1, 1), (4, 9), (0, 1, 3, 3), (4,)])
def test_depthwise_malformed_numpy_weights_are_refused(shape):
    with pytest.raises(UnsupportedOperation):
        C.DepthwiseConv3x3Constants(ValueOps(M.NUMPY_EE), np.zeros(shape))


@pytest.mark.parametrize("shape", [(4, 8), (4, 1, 3, 3), (0, 9)])
def test_depthwise_malformed_server_weights_are_refused(shape):
    with pytest.raises(UnsupportedOperation):
        C.DepthwiseConv3x3Constants(ValueOps(M.NUMPY_EE), ServerArray(M.Operand(np.zeros(shape)), shape))


@pytest.mark.parametrize("shape", [(3,), (4, 1), ()])
def test_depthwise_malformed_bias_is_refused(shape):
    with pytest.raises(UnsupportedOperation):
        C.DepthwiseConv3x3Constants(ValueOps(M.NUMPY_EE), np.zeros((4, 1, 3, 3)), np.zeros(shape))


def test_depthwise_rows_and_policies_refused_before_graph_arithmetic():
    ops = ValueOps(M.NUMPY_EE)
    with pytest.raises(ValueError):
        C.DepthwiseConv3x3Constants(ops, np.zeros((4, 1, 3, 3)), rows=0)
    with pytest.raises(UnsupportedOperation):
        C.DepthwiseConv3x3Constants(ops, np.zeros((4, 1, 3, 3)), policy="round_each_op")
    c = C.DepthwiseConv3x3Constants(ops, np.zeros((4, 1, 3, 3)), rows=25)
    with pytest.raises(RowMismatch):
        C.depthwise_conv3x3_window(ops, None, c, None)
    with pytest.raises(UnsupportedOperation):
        C.depthwise_conv3x3_window(ops, None, c, None, policy="round_each_op")
    c.policy = "round_each_op"
    with pytest.raises(UnsupportedOperation):
        C.depthwise_conv3x3_window(ops, None, c, None)


@pytest.mark.parametrize("bias", [False, True])
def test_depthwise_cost_metadata_and_asset_forms(offline_ee, bias):
    ops = WindowOps(ee)
    weight = ServerArray(ee.Array(np.ones((4, 9)).tolist()).float(), (4, 9))
    b = ServerArray(ee.Array([0.] * 4).float(), (4,)) if bias else None
    c = C.DepthwiseConv3x3Constants(ops, weight, b)
    graph = C.depthwise_conv3x3_window(ops, ee.Image(ee.Array(np.zeros((49, 4)).tolist())), c, TokenGrid(7, 7).coarse(7)).serialize()
    def names(node):
        if isinstance(node, dict):
            if "functionName" in node:
                yield node["functionName"]
            for child in node.values():
                yield from names(child)
        elif isinstance(node, list):
            for child in node:
                yield from names(child)
    functions = list(names(json.loads(graph)))
    assert functions.count("Image.multiply") == 1
    assert functions.count("Image.arrayReduce") == 1
    assert "Image.matrixMultiply" not in functions
    entry = C.METADATA["depthwise_conv3x3_window"]
    assert c.embedded_values == entry.embedded_constants(4, bias=bias) == 49 * 4 * (10 if bias else 9)
    assert entry.asset_arrays(bias=bias) == (2 if bias else 1)
    assert entry.embedded_constants(1024) == 501760


@pytest.mark.parametrize("dim", [7, 32])
def test_attention_post_product_order_is_observable_and_matches_pytorch(dim):
    import torch
    from tools.fixtures.core import dyadic
    ops = ValueOps(M.NUMPY_EE)
    q, k, v = [dyadic(20261014, n, (14 if n == "q" else 7, dim)) for n in ("q", "k", "v")]
    factor = dim ** -0.5
    result = A.naive_attention(ops, *[M.Operand(x) for x in (q, k, v)], tokens=7, dim=dim, score_scale=factor)
    mirror = M.attention(q, k, v, score_scale=factor)
    for name in ("scaled", "attention", "output"):
        np.testing.assert_array_equal(result[name].values, mirror[name])
    scores = torch.from_numpy(q) @ torch.from_numpy(k).T
    scaled = scores * factor
    expected = scaled.softmax(-1) @ torch.from_numpy(v)
    assert np.max(np.abs(result["output"].values - expected.numpy())) <= 1e-4
    # Injections: moving multiply to Q or replacing by divide changes the score bits.
    prescaled = A.naive_attention(ops, *[M.Operand(x) for x in (q, k, v)], tokens=7, dim=dim, query_scale=factor)
    divided = A.naive_attention(ops, *[M.Operand(x) for x in (q, k, v)], tokens=7, dim=dim)
    assert not np.array_equal(result["scaled"].values, prescaled["scaled"].values)
    assert not np.array_equal(result["scaled"].values, divided["scaled"].values)


def test_attention_score_scale_default_graph_unchanged_and_conflict_refused(offline_ee):
    ops = ValueOps(ee)
    q = ee.Array(np.ones((14, 32)).tolist()).float()
    k = ee.Array(np.ones((7, 32)).tolist()).float()
    default = A.naive_attention(ops, q, k, k, tokens=7, dim=32)
    explicit = A.naive_attention(ops, q, k, k, tokens=7, dim=32, score_scale=None)
    assert default.keys() == explicit.keys()
    assert all(default[n].serialize() == explicit[n].serialize() for n in default)
    with pytest.raises(UnsupportedOperation):
        A.naive_attention(ops, q, k, k, tokens=7, dim=32, query_scale=0.5, score_scale=0.5)


@pytest.mark.parametrize("shape", [(True, 9), (4.5, 9), (4, 9.0), (4, True), ("4", 9)])
def test_depthwise_server_weight_dimensions_must_be_positive_integers(shape):
    with pytest.raises(UnsupportedOperation):
        C.DepthwiseConv3x3Constants(ValueOps(M.NUMPY_EE), ServerArray(None, shape))


@pytest.mark.parametrize("shape", [(True,), (4.0,), ("4",)])
def test_depthwise_server_bias_dimensions_must_be_positive_integers(shape):
    with pytest.raises(UnsupportedOperation):
        C.DepthwiseConv3x3Constants(ValueOps(M.NUMPY_EE), np.zeros((4, 1, 3, 3)), ServerArray(None, shape))


@pytest.mark.parametrize("which", ["weight", "bias"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 1e100])
def test_depthwise_nonfinite_float32_coefficients_are_explicitly_refused(which, value):
    weight, bias = np.ones((4, 1, 3, 3), np.float64), np.zeros(4, np.float64)
    if which == "weight":
        weight[0, 0, 0, 0] = value
    else:
        bias[0] = value
    with pytest.raises(UnsupportedOperation):
        C.DepthwiseConv3x3Constants(ValueOps(M.NUMPY_EE), weight, bias)


@pytest.fixture(scope="module")
def smp_parameters():
    """Exact random reference construction only: never call forward/predict."""
    torch = pytest.importorskip("torch")
    smp = pytest.importorskip("segmentation_models_pytorch")
    torch.set_num_threads(2)
    with torch.random.fork_rng():
        torch.manual_seed(20261014)
        model = smp.Segformer("mit_b0", encoder_weights=None, in_channels=2, classes=2).eval()
    return {name: value.detach().cpu().numpy() for name, value in model.state_dict().items()}


def test_fixed_schema_matches_installed_smp_state_and_metadata_costs(smp_parameters):
    from tensor_ee.models.segformer_b0.configuration import parameter_shapes, check_parameters, constant_costs

    assert {name: value.shape for name, value in smp_parameters.items()} == parameter_shapes()
    assert len(parameter_shapes()) == 192
    check_parameters(smp_parameters)
    costs = constant_costs()
    assert [costs[f"encoder_stage{s}"]["values"] for s in range(1, 5)] == [342240, 614720, 1828640, 3333888]
    assert costs["total"] == {"values": 6620078, "asset_arrays": 191}


@pytest.mark.parametrize("change", [{"height": 256}, {"width": 256}, {"batch_size": 2}, {"classes": 3},
                                    {"bands": ("VV", "VH")}, {"bands": ("b1", "b0")}, {"bands": ["b0", "b1"]},
                                    {"window": 8}, {"training": True}, {"activation": "softmax"},
                                    {"height": 224.0}, {"batch_size": True}])
def test_fixed_model_refuses_all_configuration_variants(change):
    from tensor_ee.models.segformer_b0.configuration import SegformerB0Config

    with pytest.raises(UnsupportedOperation):
        SegformerB0Config(**change)


@pytest.mark.parametrize("fault", ["missing", "extra", "dtype", "shape", "nonfinite", "variance", "counter"])
def test_complete_state_refusals(smp_parameters, fault):
    from tensor_ee.models.segformer_b0.configuration import COUNTER, check_parameters

    state = dict(smp_parameters)
    name = "encoder.block4.1.attn.q.weight"
    if fault == "missing":
        del state[name]
    elif fault == "extra":
        state["encoder.position_embedding"] = np.zeros(1, np.float32)
    elif fault == "dtype":
        state[name] = state[name].astype(np.float64)
    elif fault == "shape":
        state[name] = state[name][:-1]
    elif fault == "nonfinite":
        state[name] = state[name].copy()
        state[name][0, 0] = np.nan
    elif fault == "variance":
        state["decoder.fuse_stage.1.running_var"] = -np.ones(256, np.float32)
    elif fault == "counter":
        state[COUNTER] = np.array(0., np.float32)
    with pytest.raises(UnsupportedOperation):
        check_parameters(state)


@pytest.mark.parametrize("shape", [None, 32, (True, 98), (32., 98), (32, 98.), (32, False), (32, 97)])
def test_model_server_shape_refusals(smp_parameters, shape):
    from tensor_ee.models.segformer_b0.configuration import check_parameters

    state = dict(smp_parameters)
    state["encoder.patch_embed1.proj.weight"] = ServerArray(None, shape)
    with pytest.raises(UnsupportedOperation):
        check_parameters(state)


@pytest.mark.parametrize("eps", [1e-5, 1e-6])
def test_norm_composition_actual_library_arithmetic_and_rows(eps):
    from tensor_ee.models.segformer_b0.encoder import norm_constants, channel_norm
    from tensor_ee.core.tensor import WindowTensor
    from tools.fixtures.segformer_b0 import _norm_data
    from reference.segformer_b0.model import layer_norm

    data = _norm_data(20261015)
    x = data.inputs["x"][:7, :7].reshape(49, 4)
    ops = ValueOps(M.NUMPY_EE)
    c = norm_constants(ops, data.weights, "norm", 4, eps=eps)
    out = channel_norm(ops, WindowTensor(M.Operand(x), TokenGrid(7, 7), 4), c)
    expected = layer_norm(x, data.weights["norm.weight"], data.weights["norm.bias"], eps)
    np.testing.assert_array_equal(out.image.values, expected)
    wrong_rows = norm_constants(ops, data.weights, "norm", 4, eps=eps, layout=WindowLayout(5))
    with pytest.raises(RowMismatch):
        channel_norm(ops, WindowTensor(M.Operand(x), TokenGrid(7, 7), 4), wrong_rows)


def test_global_kv_shared_by_four_windows_uses_actual_attention_arithmetic(monkeypatch):
    from tensor_ee.models.segformer_b0 import encoder as E
    from tensor_ee.core.tensor import WindowTensor
    from tools.fixtures.core import to_window_layout, from_window_layout
    from tools.fixtures.segformer_b0 import _shared_data, _shared_reference
    from reference.segformer_b0.model import shared_keys

    data = _shared_data(20261015)
    ops = ValueOps(M.NUMPY_EE)
    kv = WindowTensor(M.Operand(data.inputs["kv"].reshape(49, 128)), TokenGrid(7, 7), 128)
    monkeypatch.setattr(S, "gather", lambda ee, image, coarse: image.reshape([1, 1, 49, 128]))
    queries = to_window_layout(data.inputs["q"]).reshape(4, 49, 64)
    outputs = []
    for query in queries:
        result = E.shared_key_attention(ops, WindowTensor(M.Operand(query), TokenGrid(14, 14), 64), kv, 2)
        outputs.append(result["out"].image.values)
    actual = from_window_layout(np.stack(outputs).reshape(2, 2, 49, 64), 14, 14)
    np.testing.assert_array_equal(actual, shared_keys(data.inputs["q"], data.inputs["kv"], 2)["out"])
    expected = _shared_reference(data)["out"]
    assert np.max(np.abs(actual.astype(np.float64) - expected)) <= 1e-4
    # Changing a single global value affects every query window, not just its
    # spatial neighbour. No local per-window K/V substitution can satisfy this.
    changed = data.inputs["kv"].copy()
    changed[0, 0, 64:] += 1
    injected = shared_keys(data.inputs["q"], changed, 2)["out"]
    for y, x in ((0, 0), (0, 7), (7, 0), (7, 7)):
        assert not np.array_equal(actual[y:y + 7, x:x + 7], injected[y:y + 7, x:x + 7])


@pytest.mark.parametrize("factor", [2, 4])
def test_phase_sampler_projection_centres_select_exact_integer_token_ids(factor):
    from tensor_ee.core.grid import to_feature_grid
    from tensor_ee.layouts.raster import stride_sample_phase

    fine = to_feature_grid(TokenGrid(28, 28, 2.5))
    projections, relabels = [], []

    class Projection:
        def __init__(self, crs, transform):
            self.crs, self.transform = crs, transform

    class Image:
        def reproject(self, projection):
            projections.append(projection)
            return self

        def changeProj(self, before, after):
            relabels.append((before, after))
            return self

    from types import SimpleNamespace
    fake_ee = SimpleNamespace(Projection=Projection)
    stride_sample_phase(fake_ee, Image(), fine, factor, a=0, b=0)
    sample = projections[-1].transform
    centers = sample[2] + (np.arange(28 // factor) + 0.5) * sample[0]
    indices = (centers / fine.scale - 0.5).astype(int)
    np.testing.assert_array_equal(indices, np.arange(0, 28, factor))
    ids = np.arange(28 * 28).reshape(28, 28)
    np.testing.assert_array_equal(ids[np.ix_(indices, indices)], ids[::factor, ::factor])
    assert relabels[0][1].transform == fine.coarse(factor).transform


def test_decimation_passes_feature_grid_phase_zero_and_canonical_relabel(monkeypatch, offline_ee):
    from tensor_ee.models.segformer_b0 import encoder as E
    from tensor_ee.core.grid import FeatureGrid
    from tensor_ee.core.tensor import WindowTensor

    records = []
    image = ee.Image(ee.Array([[0., 0.]] * 49))
    monkeypatch.setattr(S, "to_dense", lambda *args, **kwargs: image)

    def sample(module, dense, fine, factor, a, b):
        assert isinstance(fine, FeatureGrid)
        records.append((fine, factor, a, b))
        return image

    monkeypatch.setattr(S, "stride_sample_phase", sample)
    monkeypatch.setattr(S, "from_dense", lambda module, grid, dense, channels, **kwargs: image)
    x = WindowTensor(image, TokenGrid(28, 28, 2.5), 2)
    out = E.decimate(ee, x, 4)
    assert records[0][1:] == (4, 0, 0)
    assert out.grid == TokenGrid(7, 7, 10)
    with pytest.raises(UnsupportedOperation):
        E.decimate(ee, x, 4.)


def test_overlap_kernel_weight_transport_keeps_channel_and_tap_ids(offline_ee):
    from tensor_ee.models.segformer_b0.encoder import _conv_columns
    from tensor_ee.core.ops import ValueOps

    w = np.arange(4 * 2 * 49, dtype=np.float32).reshape(4, 2, 7, 7)
    ops = ValueOps(M.NUMPY_EE)
    expected = w.transpose(0, 2, 3, 1).reshape(4, 98)
    np.testing.assert_array_equal(_conv_columns(ops, w, 2, 4, 7, tap_major=True), expected)
    server = ServerArray(M.Operand(w.reshape(4, 98)), (4, 98))
    converted = _conv_columns(ops, server, 2, 4, 7, tap_major=True)
    np.testing.assert_array_equal(converted.array.values, expected)
    np.testing.assert_array_equal(_conv_columns(ops, w, 2, 4, 7, tap_major=False), w.reshape(4, 98))


@pytest.mark.parametrize("kernel,stride", [(7, 4), (3, 2)])
@pytest.mark.parametrize("server", [False, True])
def test_overlap_actual_library_arithmetic_and_decimation_order(monkeypatch, kernel, stride, server):
    import torch
    from tensor_ee.models.segformer_b0 import encoder as E
    from tensor_ee.core.tensor import WindowTensor
    from tools.fixtures.core import to_window_layout, from_window_layout
    from tools.fixtures.segformer_b0 import _overlap_data, _overlap_reference

    data = _overlap_data(20261015, kernel=kernel, stride=stride)
    source = data.inputs["x"]
    grid = TokenGrid(*source.shape[:2], 2.5)
    ops = ValueOps(M.NUMPY_EE)
    p = dict(data.weights)
    if server:
        p = {name: ServerArray(M.Operand(value.reshape(value.shape[0], -1) if value.ndim == 4 else value),
                                (value.shape[0], int(np.prod(value.shape[1:]))) if value.ndim == 4 else value.shape)
             for name, value in p.items()}
    c = E.overlap_constants(ops, p, "embedding", 2, 4, kernel=kernel, stride=stride)

    def neighbourhood(module, image, coarse, channels, **kwargs):
        assert channels == 2
        assert kwargs.get("ring", 1) == kernel // 2
        return np.pad(source, ((kernel // 2, kernel // 2), (kernel // 2, kernel // 2), (0, 0)))

    def columns(module, padded, channels, k, **kwargs):
        tensor = torch.from_numpy(padded.copy()).permute(2, 0, 1)[None]
        rows = torch.nn.functional.unfold(tensor, k)[0].T.reshape(-1, channels, k * k)
        rows = rows.permute(0, 2, 1).reshape(*source.shape[:2], channels * k * k).numpy()
        return M.Operand(to_window_layout(rows))

    monkeypatch.setattr(S, "halo", neighbourhood)
    monkeypatch.setattr(S, "im2col", columns)
    monkeypatch.setattr(C, "halo", neighbourhood)
    monkeypatch.setattr(C, "im2col", columns)
    decimation_calls = []

    def sample(module, projected, factor):
        decimation_calls.append((projected.channels, factor))
        dense = from_window_layout(projected.image.values, *source.shape[:2])
        sampled = dense[::factor, ::factor]
        return WindowTensor(M.Operand(sampled.reshape(49, 4)), grid.coarse(factor), 4)

    monkeypatch.setattr(S, "decimate", sample)
    x = WindowTensor(M.Operand(to_window_layout(source)), grid, 2)
    out = E.overlap_embedding(ops, x, c)
    assert decimation_calls == [(4, stride)]  # convolution/bias precede decimation
    reference = _overlap_reference(data, kernel=kernel, stride=stride)
    for name in ("projection", "out"):
        assert np.max(np.abs(out[name].image.values.reshape(7, 7, 4).astype(np.float64) - reference[name])) <= 1e-4


@pytest.mark.parametrize("dim,ratio", [(32, 8), (64, 4), (160, 2)])
def test_grouped_sra_join_order_ids_and_actual_projection_norm(monkeypatch, dim, ratio):
    import torch
    from tensor_ee.models.segformer_b0 import encoder as E
    from tensor_ee.core.tensor import WindowTensor
    from tools.fixtures.segformer_b0 import _reduction_data, _reduction_constants, _reduction_reference

    data = _reduction_data(20261015, dim=dim, ratio=ratio)
    ops = ValueOps(M.NUMPY_EE)
    source = data.inputs["x"]
    columns = torch.nn.functional.unfold(torch.from_numpy(source).permute(2, 0, 1)[None], ratio, stride=ratio)[0].T.numpy()
    bands = []
    monkeypatch.setattr(S, "to_dense", lambda *args, **kwargs: source)

    def select(module, dense, grid, *, patch, bands: list):
        assert patch == ratio
        channels = [int(name[1:]) for name in bands]
        selected = columns.reshape(49, dim, ratio * ratio)[:, channels].reshape(49, -1)
        joined_bands.extend(bands)
        assert len(bands) * patch * patch <= 128
        return M.Operand(selected)

    joined_bands = bands
    monkeypatch.setattr(S, "stride_columns", select)
    monkeypatch.setattr(S, "to_windows", lambda module, rows, grid, **kwargs: rows)
    out = E.spatial_reduction(ops, WindowTensor(None, TokenGrid(*source.shape[:2]), dim),
                              _reduction_constants(ops, data, dim=dim, ratio=ratio))
    assert joined_bands == [f"c{i}" for i in range(dim)]
    reference = _reduction_reference(data, dim=dim, ratio=ratio)
    np.testing.assert_array_equal(out["projection"].image.values.reshape(7, 7, dim)[..., :8], reference["projection"])
    assert np.max(np.abs(out["out"].image.values.reshape(7, 7, dim).astype(np.float64) - reference["out"])) <= 1e-4


def test_mix_ffn_actual_arithmetic_keeps_depthwise_before_exact_gelu(monkeypatch):
    import torch
    from tensor_ee.models.segformer_b0 import encoder as E
    from tensor_ee.core.tensor import WindowTensor
    from tools.fixtures.segformer_b0 import _ffn_data
    from reference.segformer_b0.model import mix_ffn

    data = _ffn_data(20261015)
    x = data.inputs["x"][:7, :7].copy()
    ops = ValueOps(M.NUMPY_EE)
    c = E.mix_ffn_constants(ops, data.weights, "mlp", 4)

    def halo(module, image, coarse, channels, **kwargs):
        return np.pad(image.values.reshape(7, 7, channels), ((1, 1), (1, 1), (0, 0)))

    def columns(module, padded, channels, k, **kwargs):
        values = torch.nn.functional.unfold(torch.from_numpy(padded.copy()).permute(2, 0, 1)[None], 3)[0].T
        return M.Operand(values.reshape(49, channels, 9).permute(0, 2, 1).reshape(49, 9 * channels).numpy())

    monkeypatch.setattr(C, "halo", halo)
    monkeypatch.setattr(C, "im2col", columns)
    out = E.mix_ffn(ops, WindowTensor(M.Operand(x.reshape(49, 4)), TokenGrid(7, 7), 4), c)
    mirror = mix_ffn(x, data.weights, "mlp")
    for name in mirror:
        shape = (7, 7, out[name].channels)
        assert np.max(np.abs(out[name].image.values.reshape(shape).astype(np.float64) - mirror[name])) <= 1e-4


def test_decoder_pyramid_integer_ids_require_deepest_first_and_both_alignments():
    from tools.fixtures.segformer_b0 import _pyramid_data, _pyramid_reference, _pyramid_mirror
    from reference.segformer_b0.model import resize

    data = _pyramid_data(20261015)
    reference, mirror = _pyramid_reference(data), _pyramid_mirror(data)
    np.testing.assert_array_equal(mirror["out"], reference["out"])
    for i, name in enumerate(("C4", "C3", "C2", "C1")):
        np.testing.assert_array_equal(mirror["out"][..., 2 * i:2 * (i + 1)], mirror[f"{name}_to_C1"])
    assert not np.array_equal(mirror["out"], np.concatenate([mirror[f"C{i}_to_C1"] for i in range(1, 5)], -1))
    x = np.arange(14 * 14 * 2, dtype=np.float32).reshape(14, 14, 2)
    assert not np.array_equal(resize(x, 4, align_corners=True), resize(x, 4, align_corners=False))


def test_actual_decoder_wiring_is_deepest_first_with_false_then_true_resizing(monkeypatch):
    from types import SimpleNamespace
    from tensor_ee.models.segformer_b0 import decoder as D
    from tensor_ee.models.segformer_b0.configuration import SEGFORMER_B0, STAGE_DIMS
    from tensor_ee.core.tensor import WindowTensor

    calls = []

    class Image:
        def __init__(self, label):
            self.label = label

        def arrayCat(self, other, axis):
            assert axis == 1
            return Image(self.label + "," + other.label)

    ops = SimpleNamespace(ee=None)

    def project(at, image, constant, **kwargs):
        assert at is ops
        calls.append(("project", image.label))
        return Image(image.label)

    def resize(module, image, grid, channels, factor, align_corners, **kwargs):
        calls.append(("resize", image.label, factor, align_corners))
        return image, grid.fine(factor)

    def conv(at, image, constant, **kwargs):
        calls.append(("conv", image.label, constant.out_features))
        return Image("head" if constant.out_features == 2 else "fused")

    def relu(at, image):
        assert at is ops and image.label == "bn"
        calls.append(("relu", image.label))
        return Image("relu")

    monkeypatch.setattr(S, "linear", project)
    monkeypatch.setattr(S, "window_resize", resize)
    monkeypatch.setattr(S, "conv1x1", conv)
    monkeypatch.setattr(S, "batch_norm", lambda at, image, constant, **kwargs: Image("bn"))
    monkeypatch.setattr(S, "relu", relu)
    features = [WindowTensor(Image(f"C{s}"), grid, dim) for s, (grid, dim) in
                enumerate(zip(SEGFORMER_B0.stage_grids, STAGE_DIMS), 1)]
    projections = tuple(SimpleNamespace(in_features=dim, out_features=256) for dim in reversed(STAGE_DIMS))
    constants = D.DecoderConstants(projections, SimpleNamespace(out_features=256), None,
                                   SimpleNamespace(out_features=2))
    out = D.decoder(ops, features, constants)
    assert [call for call in calls if call[0] == "project"] == [("project", name) for name in ("C4", "C3", "C2", "C1")]
    assert out["pyramid"]["out"].image.label == "C4,C3,C2,C1"
    assert [call for call in calls if call[0] == "resize"] == [("resize", "C4", 8, False),
                                                             ("resize", "C3", 4, False),
                                                             ("resize", "C2", 2, False),
                                                             ("resize", "head", 4, True)]
    assert out["out"].grid == SEGFORMER_B0.input_grid


@pytest.mark.parametrize("i,j", [(i, j) for i in range(4) for j in range(4)])
def test_actual_ring3_halo_and_k7_im2col_integer_ids_at_chip_edges_and_seams(monkeypatch, i, j):
    """Execute actual layout slicing/joining with a tiny value-only image adapter.

    Neighbour windows are supplied explicitly; the previously validated spatial
    neighbour transport is not reimplemented or numerically evaluated here.
    """
    from types import SimpleNamespace
    import torch
    from tensor_ee.layouts import window as W

    class ArrayImage:
        def __init__(self, values):
            self.values = np.asarray(values)

        def arraySlice(self, axis, start, end):
            indices = [slice(None)] * self.values.ndim
            indices[axis] = slice(start, end)
            return ArrayImage(self.values[tuple(indices)])

        def arrayCat(self, other, axis):
            return ArrayImage(np.concatenate([self.values, other.values], axis))

        def arrayReshape(self, shape, dimensions):
            target = shape.values.astype(int).tolist()
            assert len(target) == dimensions
            return ArrayImage(self.values.reshape(target))

    fake_ee = SimpleNamespace(Array=np.asarray, Image=ArrayImage)
    source = np.arange(28 * 28 * 2, dtype=np.float32).reshape(28, 28, 2) + 1
    neighbours = {}
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            y, x = 7 * (i + dy), 7 * (j + dx)
            tile = source[y:y + 7, x:x + 7] if 0 <= y < 28 and 0 <= x < 28 else np.zeros((7, 7, 2), np.float32)
            neighbours[dy, dx] = ArrayImage(tile)
    monkeypatch.setattr(W, "neighbours", lambda *args, **kwargs: neighbours)
    actual_halo = W.halo(fake_ee, None, TokenGrid(4, 4), 2, window=7, ring=3)
    padded = np.pad(source, ((3, 3), (3, 3), (0, 0)))
    np.testing.assert_array_equal(actual_halo.values, padded[7 * i:7 * i + 13, 7 * j:7 * j + 13])
    actual_columns = W.im2col(fake_ee, actual_halo, 2, 7, window=7).values
    # Independent PyTorch unfolding has (c,ky,kx); transpose only that layout
    # into the library's (ky,kx,c), then select the tested fine-grid window.
    rows = torch.nn.functional.unfold(torch.from_numpy(source).permute(2, 0, 1)[None], 7, padding=3)[0].T
    rows = rows.reshape(28, 28, 2, 49).permute(0, 1, 3, 2).reshape(28, 28, 98).numpy()
    np.testing.assert_array_equal(actual_columns, rows[7 * i:7 * (i + 1), 7 * j:7 * (j + 1)].reshape(49, 98))


def test_model_interface_refuses_wrong_inputs_and_retains_source_mask(offline_ee):
    from tensor_ee.models.segformer_b0.decoder import input_windows
    from tensor_ee.core.tensor import Raster
    from tensor_ee.core.grid import Grid

    grid = Grid("EPSG:3857", (10., 0., 1000., 0., -10., 2000.), 224, 224)
    image = ee.Image.constant([1., 2.]).rename(["b0", "b1"])
    windows = input_windows(ee, Raster(image, ("b0", "b1"), grid))
    assert windows.grid == TokenGrid(224, 224, 2.5)
    serialized = windows.image.serialize()
    assert "Image.changeProj" in serialized and "Image.unmask" not in serialized
    with pytest.raises(UnsupportedOperation):
        input_windows(ee, Raster(image, ("b1", "b0"), grid))
    with pytest.raises(UnsupportedOperation):
        input_windows(ee, Raster(image, ("b0", "b1"), TokenGrid(256, 256)))


def test_fixed_complete_graph_builds_without_forward_and_with_dynamic_server_state(offline_ee, smp_parameters):
    from tensor_ee.models.segformer_b0.configuration import COUNTER, parameter_shapes
    from tensor_ee.models.segformer_b0.decoder import model_constants, segformer_b0
    from tensor_ee.core.tensor import Raster

    state = {}
    for name, shape in parameter_shapes().items():
        if name == COUNTER:
            state[name] = smp_parameters[name]
            continue
        target = (shape[0], int(np.prod(shape[1:]))) if len(shape) == 4 else shape
        # Dynamic server expressions: no coefficients, selectors or fake provider
        # records are repeated per query window. Values are not evaluated here.
        source = ee.Array(ee.List.repeat(0., int(np.prod(shape)))).reshape(list(target)).float()
        state[name] = ServerArray(source, target)
    constants = model_constants(ee, state)
    raster = Raster(ee.Image.constant([0., 0.]).toFloat().rename(["b0", "b1"]),
                    ("b0", "b1"), TokenGrid(224, 224, 2.5))
    numpy_constants = model_constants(ee, smp_parameters)
    numpy_out = segformer_b0(ee, raster, numpy_constants)
    assert numpy_out["logits"].bands == ("c0", "c1")
    assert numpy_out["logits"].grid.width == numpy_out["logits"].grid.height == 224
    # NumPy OIHW builders and all wiring are exercised, without serializing
    # the full random coefficient payload or evaluating any model activation.
    out = segformer_b0(ee, raster, constants)
    assert out["logits"].bands == ("c0", "c1")
    assert out["logits"].grid.width == out["logits"].grid.height == 224
    # Only full-model serialization uses the accepted capstone context. Building
    # the graph above needs no limit change; fixture serialization keeps defaults.
    previous = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(max(previous, 20000))
        graph = out["logits"].image.serialize()
    finally:
        sys.setrecursionlimit(previous)
    assert "Image.sampleRectangle" in graph and "Image.changeProj" in graph
    assert "Image.arrayRepeat" in graph
    with pytest.raises(UnsupportedOperation):
        segformer_b0(ee, raster, constants, training=True)
