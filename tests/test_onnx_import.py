"""Exact checkpoint tensors, supported ONNX semantics, and fixed-grid contract."""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.fx import symbolic_trace

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper, numpy_helper

from tensor_ee.onnx_import import UnsupportedOnnx, load_onnx_model, onnx_metadata
from tensor_ee.receptive_field import analyze_receptive_field
from tensor_ee.weight_assets import effective_convolutions


CHECKPOINT = Path(__file__).resolve().parents[1] / "checkpoints" / "segwater_r18_s42_fp32_b32.onnx"


def _fixture():
    weight = (np.arange(36, dtype=np.float32).reshape(2, 2, 3, 3) - 17.5) / 32
    bias = np.asarray([-.3, .2], dtype=np.float32)
    nodes = [
        helper.make_node("Conv", ["input", "weights", "bias"], ["features"],
                         kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[2, 2]),
        helper.make_node("Shape", ["features"], ["shape"]),
    ]
    for name, values in (("starts", [0]), ("ends", [2]), ("axes", [0]), ("spatial", [224, 224])):
        nodes.append(helper.make_node("Constant", [], [name],
                                      value=numpy_helper.from_array(np.asarray(values, dtype=np.int64))))
    nodes.extend([
        helper.make_node("Slice", ["shape", "starts", "ends", "axes"], ["nc"]),
        helper.make_node("Concat", ["nc", "spatial"], ["sizes"], axis=0),
        helper.make_node("Resize", ["features", "", "", "sizes"], ["logits"],
                         mode="nearest", coordinate_transformation_mode="asymmetric", nearest_mode="floor"),
    ])
    graph = helper.make_graph(nodes, "small_import_contract_fixture",
                              [helper.make_tensor_value_info("input", TensorProto.FLOAT, [32, 2, 224, 224])],
                              [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [32, 2, 224, 224])],
                              [numpy_helper.from_array(weight, "weights"), numpy_helper.from_array(bias, "bias")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def _save(tmp_path, model):
    path = tmp_path / "model.onnx"
    onnx.save(model, path)
    return path


def _set_attribute(node, name, value):
    retained = [attribute for attribute in node.attribute if attribute.name != name]
    del node.attribute[:]
    node.attribute.extend(retained)
    node.attribute.append(helper.make_attribute(name, value))


@pytest.fixture(scope="module")
def checkpoint_model():
    if not CHECKPOINT.exists():
        pytest.skip("The user-supplied trained checkpoint is not distributed with source")
    return load_onnx_model(CHECKPOINT)


def test_checkpoint_import_retains_every_frozen_float32_weight_and_bias(checkpoint_model):
    source = onnx.load(CHECKPOINT)
    initializers = {value.name: numpy_helper.to_array(value) for value in source.graph.initializer}
    source_convolutions = [node for node in source.graph.node if node.op_type == "Conv"]
    assert len(source_convolutions) == 31
    for index, original in enumerate(source_convolutions):
        module = checkpoint_model.get_submodule(f"conv_{index:03d}")
        assert isinstance(module, nn.Conv2d)
        np.testing.assert_array_equal(module.weight.detach().numpy().view(np.uint32),
                                      initializers[original.input[1]].view(np.uint32))
        np.testing.assert_array_equal(module.bias.detach().numpy().view(np.uint32),
                                      initializers[original.input[2]].view(np.uint32))
    assert sum(parameter.numel() for parameter in checkpoint_model.parameters()) == 14_319_426
    assert all(parameter.dtype == torch.float32 and not parameter.requires_grad
               for parameter in checkpoint_model.parameters())
    assert not any(isinstance(module, nn.BatchNorm2d) for module in checkpoint_model.modules())


def test_checkpoint_keys_and_metadata_survive_the_compiler_trace(checkpoint_model):
    metadata = onnx_metadata(checkpoint_model)
    assert metadata["source_sha256"] == "c34c75563208c184a5a8edc735d8643d0d41bf857ae8ce253440eeb1dacd3a94"
    assert checkpoint_model.tensor_ee_input_shape == (2, 224, 224)
    traced = symbolic_trace(checkpoint_model)
    assert [node.name for node in traced.graph.nodes] == [node.name for node in checkpoint_model.graph.nodes]
    parameters = effective_convolutions(checkpoint_model)
    assert [parameter.key for parameter in parameters] == [f"conv_{index:03d}" for index in range(31)]
    assert metadata["output_fx_node"] == "conv_030"
    assert metadata["onnx_tensor_to_fx"]["logits"] == "conv_030"
    assert metadata["input_shape"] == metadata["output_shape"] == [32, 2, 224, 224]
    assert metadata["spatial_generalization_supported"] is False
    assert len(metadata["resizes"]) == 5
    assert [resize["output_spatial_shape"] for resize in metadata["resizes"]] == [
        [14, 14], [28, 28], [56, 56], [112, 112], [224, 224]]
    metadata["resizes"].clear()
    assert len(onnx_metadata(checkpoint_model)["resizes"]) == 5


def test_checkpoint_support_is_derived_from_imported_graph(checkpoint_model):
    report = analyze_receptive_field(checkpoint_model)
    assert report.alignment == report.period == 32
    assert report.safe_halo == 288
    assert report.halo_for_any_phase == 320
    assert report.x.maximum_support_width == 563


@pytest.mark.parametrize("shape,dtype", [
    ((1, 2, 225, 224), torch.float32),
    ((1, 2, 224, 223), torch.float32),
    ((1, 3, 224, 224), torch.float32),
    ((2, 2, 224, 224), torch.float32),
    ((2, 224, 224), torch.float32),
    ((1, 2, 224, 224), torch.float64),
])
def test_eager_contract_rejects_unsupported_sizes_batches_and_dtype(checkpoint_model, shape, dtype):
    with pytest.raises(ValueError, match="contract|float32"):
        checkpoint_model(torch.zeros(shape, dtype=dtype))


def test_import_does_not_initialize_or_advance_a_random_model(tmp_path):
    path = _save(tmp_path, _fixture())
    before = torch.random.get_rng_state().clone()
    load_onnx_model(path)
    assert torch.equal(before, torch.random.get_rng_state())


def test_validated_shape_resize_matches_original_onnxruntime_batch32(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    path = _save(tmp_path, _fixture())
    model = load_onnx_model(path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    original = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    x, y = np.meshgrid(np.arange(224), np.arange(224))
    batch = np.zeros((32, 2, 224, 224), dtype=np.float32)
    batch[0] = np.stack([(3 * x + 5 * y) % 7 - 3, (7 * x + 3 * y) % 11 - 5]) / 8
    expected = original.run(["logits"], {"input": batch})[0]
    with torch.inference_mode():
        actual = model(torch.from_numpy(batch[:1])).numpy()
    np.testing.assert_allclose(actual, expected[:1], rtol=0, atol=2e-6)
    # Asymmetric/floor integer expansion repeats each source pixel exactly.
    np.testing.assert_array_equal(actual[:, :, 0::2, 0::2], actual[:, :, 1::2, 1::2])


@pytest.mark.parametrize("attribute,value", [
    ("mode", "linear"), ("coordinate_transformation_mode", "half_pixel"),
    ("nearest_mode", "round_prefer_floor"), ("exclude_outside", 1),
    ("extrapolation_value", 1.),
])
def test_rejects_resize_semantics_instead_of_silently_substituting_them(tmp_path, attribute, value):
    model = _fixture()
    _set_attribute(model.graph.node[-1], attribute, value)
    with pytest.raises(UnsupportedOnnx, match="nearest/asymmetric/floor"):
        load_onnx_model(_save(tmp_path, model))


def test_rejects_frozen_resize_batch_and_arbitrary_spatial_size(tmp_path):
    for values in ([32, 2, 224, 224], [32, 2, 225, 224]):
        model = _fixture()
        model.graph.node[-1].input[3] = "literal_sizes"
        model.graph.node.insert(-1, helper.make_node("Constant", [], ["literal_sizes"],
                                                    value=numpy_helper.from_array(np.asarray(values, np.int64))))
        with pytest.raises(UnsupportedOnnx, match="preserve input N/C"):
            load_onnx_model(_save(tmp_path, model))


@pytest.mark.parametrize("attribute,value", [
    ("pads", [1, 0, 1, 2]), ("strides", [3, 3]), ("dilations", [2, 2]),
    ("group", 2), ("auto_pad", "SAME_UPPER"), ("kernel_shape", [1, 1]),
])
def test_rejects_convolution_semantics_outside_the_native_subset(tmp_path, attribute, value):
    model = _fixture()
    _set_attribute(model.graph.node[0], attribute, value)
    with pytest.raises(UnsupportedOnnx, match="Conv requires odd same padding"):
        load_onnx_model(_save(tmp_path, model))


def test_rejects_external_tensor_files_before_loading_them(tmp_path):
    model = _fixture()
    path = tmp_path / "external.onnx"
    onnx.save_model(model, path, save_as_external_data=True, all_tensors_to_one_file=True,
                    location="external_weights.bin", size_threshold=0)
    assert any(value.data_location == TensorProto.EXTERNAL for value in onnx.load(path, load_external_data=False).graph.initializer)
    # The provenance rejection is independent of the external file's existence.
    (tmp_path / "external_weights.bin").unlink()
    with pytest.raises(UnsupportedOnnx, match="External tensor"):
        load_onnx_model(path)


def _external(tensor, location):
    """Mark a TensorProto as stored in an external file (no file is written)."""
    tensor.data_location = TensorProto.EXTERNAL
    tensor.ClearField("raw_data")
    for key, value in (("location", location), ("offset", "0"), ("length", "16")):
        entry = tensor.external_data.add()
        entry.key, entry.value = key, value


def test_rejects_an_external_constant_before_opening_any_file(tmp_path, monkeypatch):
    """S02 E-01: a Constant node's tensor stored outside the file passes the checker and would be
    read relative to the working directory by numpy_helper.to_array; it is refused first."""
    model = _fixture()
    constant = next(node for node in model.graph.node if node.op_type == "Constant" and node.output[0] == "spatial")
    (tmp_path / "local-file.bin").write_bytes(bytes(range(32)))     # any readable local file
    _external(constant.attribute[0].t, "local-file.bin")
    assert not any(value.data_location == TensorProto.EXTERNAL for value in model.graph.initializer)
    monkeypatch.chdir(tmp_path)
    onnx.checker.check_model(model)          # the checker accepts it (the file exists): the guard cannot rely on it
    path = _save(tmp_path, model)

    def never(*args, **kwargs):
        raise AssertionError("an external tensor file was opened")
    monkeypatch.setattr(onnx.external_data_helper, "load_external_data_for_tensor", never)
    with pytest.raises(UnsupportedOnnx, match="External tensor.*attribute 'value'"):
        load_onnx_model(path)


def test_rejects_an_external_tensor_inside_a_subgraph(tmp_path, monkeypatch):
    model = _fixture()
    inner = helper.make_graph([helper.make_node("Constant", [], ["c"], value=numpy_helper.from_array(np.zeros(2, np.int64)))],
                              "branch", [], [helper.make_tensor_value_info("c", TensorProto.INT64, [2])])
    _external(inner.node[0].attribute[0].t, "does-not-exist.bin")
    model.graph.node.append(helper.make_node("If", ["flag"], ["chosen"], then_branch=inner, else_branch=inner))
    path = _save(tmp_path, model)
    monkeypatch.setattr(onnx.external_data_helper, "load_external_data_for_tensor",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("opened")))
    with pytest.raises(UnsupportedOnnx, match="External tensor"):
        load_onnx_model(path)


def _external_sparse(location):
    """A SparseTensorProto whose values are stored in an external file (no file is written)."""
    values = numpy_helper.from_array(np.asarray([1, 2], dtype=np.int64), "sparse_values")
    indices = numpy_helper.from_array(np.asarray([0, 3], dtype=np.int64), "sparse_indices")
    _external(values, location)
    return helper.make_sparse_tensor(values, indices, [4])


def _never_stat_or_load(monkeypatch):
    """The guard must refuse before the checker stats the location and before any load."""
    monkeypatch.setattr(onnx.external_data_helper, "load_external_data_for_tensor",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("opened")))
    original = onnx.checker.check_model
    monkeypatch.setattr(onnx.checker, "check_model",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("the checker ran before the guard")))
    return original


def test_rejects_an_external_sparse_constant_before_the_checker(tmp_path, monkeypatch):
    """S03 review P2-1: a Constant's sparse_value stored outside the file."""
    model = _fixture()
    model.graph.node.append(helper.make_node("Constant", [], ["sparse"], sparse_value=_external_sparse("does-not-exist.bin")))
    path = _save(tmp_path, model)
    _never_stat_or_load(monkeypatch)
    with pytest.raises(UnsupportedOnnx, match="External tensor.*sparse_value.*values"):
        load_onnx_model(path)


def test_rejects_an_external_sparse_initializer_inside_a_subgraph(tmp_path, monkeypatch):
    model = _fixture()
    inner = helper.make_graph([], "branch", [], [helper.make_tensor_value_info("c", TensorProto.INT64, [4])],
                              sparse_initializer=[_external_sparse("does-not-exist.bin")])
    model.graph.node.append(helper.make_node("If", ["flag"], ["chosen"], then_branch=inner, else_branch=inner))
    path = _save(tmp_path, model)
    _never_stat_or_load(monkeypatch)
    with pytest.raises(UnsupportedOnnx, match="External tensor.*sparse initializer"):
        load_onnx_model(path)


def test_rejects_a_file_above_the_size_limit(tmp_path, monkeypatch):
    from tensor_ee import onnx_import
    path = _save(tmp_path, _fixture())
    monkeypatch.setattr(onnx_import, "MAX_ONNX_BYTES", path.stat().st_size - 1)
    with pytest.raises(UnsupportedOnnx, match="exceeds"):
        load_onnx_model(path)


def test_rejects_nonfinite_or_nonfloat32_initializers(tmp_path):
    for weights in (np.zeros((2, 2, 3, 3), dtype=np.float64),
                    np.full((2, 2, 3, 3), np.nan, dtype=np.float32)):
        model = _fixture()
        model.graph.initializer[0].CopyFrom(numpy_helper.from_array(weights, "weights"))
        with pytest.raises(UnsupportedOnnx, match="finite float32"):
            load_onnx_model(_save(tmp_path, model))


def test_rejects_other_opsets_and_spatial_generalization(tmp_path):
    model = _fixture()
    model.opset_import[0].version = 18
    with pytest.raises(UnsupportedOnnx, match="opset 17"):
        load_onnx_model(_save(tmp_path, model))
    model = _fixture()
    model.graph.input[0].type.tensor_type.shape.dim[2].dim_value = 256
    with pytest.raises(UnsupportedOnnx, match="fixed NCHW"):
        load_onnx_model(_save(tmp_path, model))
