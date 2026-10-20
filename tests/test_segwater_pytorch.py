"""Direct original SMP checkpoint loading, fixed224 parity, and offline FX/EE compatibility."""

import hashlib
import json
import operator
from pathlib import Path

import ee
from ee.apitestcase import ApiTestCase
import pytest
import torch
from torch import nn
import torch.nn.functional as F
from torch.fx import symbolic_trace

smp = pytest.importorskip("segmentation_models_pytorch")

from tensor_ee.compiler import Grid, UnsupportedOperation, compile_model
from tensor_ee.segwater_pytorch import adapt_for_earth_engine, load_segwater_checkpoint
from tensor_ee.weight_assets import effective_convolutions


@pytest.fixture(scope="module", autouse=True)
def bounded_torch_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def genuine_checkpoint(tmp_path_factory):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4281)
        original = smp.Unet(encoder_name="resnet18", encoder_weights=None,
                            in_channels=2, classes=2).eval()
    generator = torch.Generator().manual_seed(4282)
    with torch.no_grad():
        for module in original.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.running_mean.copy_(torch.randn(module.running_mean.shape, generator=generator) * .1)
                module.running_var.copy_(torch.rand(module.running_var.shape, generator=generator) + .5)
                module.weight.copy_(torch.rand(module.weight.shape, generator=generator) + .5)
                module.bias.copy_(torch.randn(module.bias.shape, generator=generator) * .05)
                module.num_batches_tracked.fill_(7)
    path = tmp_path_factory.mktemp("segwater-pytorch") / "random_genuine_smp.pth"
    torch.save({"model_state_dict": original.state_dict(), "step": 6000,
                "val_miou": .625, "optimizer_state_dict": {}}, path)
    return path, original


@pytest.fixture(scope="module")
def imported(genuine_checkpoint):
    return load_segwater_checkpoint(genuine_checkpoint[0])


def test_strict_roundtrip_preserves_original_smp_state_and_separates_adapter(genuine_checkpoint, imported):
    path, original = genuine_checkpoint
    reference, adapter = imported.reference_model, imported.compilable_model
    assert isinstance(reference, smp.Unet)
    assert reference is not adapter
    assert reference.encoder is not adapter.encoder
    assert list(reference.state_dict()) == list(original.state_dict()) == list(adapter.state_dict())
    for key, value in original.state_dict().items():
        assert torch.equal(reference.state_dict()[key], value)
        assert torch.equal(adapter.state_dict()[key], value)
    assert any(isinstance(module, nn.ReLU) and module.inplace for module in reference.modules())
    assert not any(isinstance(module, nn.ReLU) and module.inplace for module in adapter.modules())
    assert sum(isinstance(module, nn.BatchNorm2d) for module in reference.modules()) == 30
    assert sum(isinstance(module, nn.Conv2d) for module in reference.modules()) == 31
    assert reference.segmentation_head[0].weight.shape == (2, 16, 3, 3)
    assert all(value.dtype == torch.float32 and not value.requires_grad for value in reference.parameters())
    assert not any(module.training for module in reference.modules())
    assert imported.metadata["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert imported.metadata["source_bytes"] == path.stat().st_size
    assert imported.metadata["step"] == 6000
    assert imported.metadata["val_miou"] == .625
    assert imported.metadata["onnx_required"] is False
    assert imported.metadata["reference_batchnorm_fused"] is False
    json.dumps(imported.metadata)


@pytest.mark.parametrize("case", ["random", "boundary_impulse"])
def test_fixed224_adapter_equals_the_unmodified_genuine_smp_model(imported, case):
    tensor = torch.randn((1, 2, 224, 224), generator=torch.Generator().manual_seed(4283))
    if case == "boundary_impulse":
        tensor.zero_()
        tensor[0, :, 0, 0] = torch.tensor([4., -4.])
        tensor[0, :, -1, -1] = torch.tensor([-3., 3.])
    with torch.inference_mode():
        expected = imported.reference_model(tensor)
        actual = imported.compilable_model(tensor)
    assert actual.shape == (1, 2, 224, 224)
    assert actual.dtype == torch.float32
    assert torch.isfinite(actual).all()
    assert torch.equal(actual, expected)


def test_loading_uses_safe_cpu_deserialization_and_preserves_rng(genuine_checkpoint, monkeypatch):
    actual_load = torch.load
    calls = []
    def load(*args, **kwargs):
        calls.append(kwargs)
        return actual_load(*args, **kwargs)
    monkeypatch.setattr(torch, "load", load)
    before = torch.random.get_rng_state().clone()
    load_segwater_checkpoint(genuine_checkpoint[0])
    assert calls == [{"map_location": "cpu", "weights_only": True}]
    assert torch.equal(before, torch.random.get_rng_state())


def test_provenance_uses_the_loaded_bytes_if_the_checkpoint_changes(tmp_path, genuine_checkpoint, monkeypatch):
    path = tmp_path / "mutable.pth"
    source_data = genuine_checkpoint[0].read_bytes()
    path.write_bytes(source_data)
    actual_load = torch.load

    def load_then_change_file(*args, **kwargs):
        checkpoint = actual_load(*args, **kwargs)
        path.write_bytes(b"changed after deserialization")
        return checkpoint

    monkeypatch.setattr(torch, "load", load_then_change_file)
    imported = load_segwater_checkpoint(path)
    assert imported.metadata["source_sha256"] == hashlib.sha256(source_data).hexdigest()
    assert imported.metadata["source_bytes"] == len(source_data)
    assert torch.equal(imported.reference_model.encoder.conv1.weight,
                       genuine_checkpoint[1].encoder.conv1.weight)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_rejects_checkpoint_float_dtypes_before_any_implicit_conversion(tmp_path, dtype):
    path = tmp_path / "wrong-dtype.pth"
    torch.save({"encoder.conv1.weight": torch.ones((64, 2, 7, 7), dtype=dtype)}, path)
    with pytest.raises(ValueError, match="must use float32"):
        load_segwater_checkpoint(path)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_rejects_nonfinite_checkpoint_coefficients(tmp_path, invalid):
    path = tmp_path / "nonfinite.pth"
    torch.save({"encoder.conv1.weight": torch.full((64, 2, 7, 7), invalid)}, path)
    with pytest.raises(ValueError, match="finite values"):
        load_segwater_checkpoint(path)


def test_missing_checkpoint_key_is_a_strict_loading_failure(tmp_path, genuine_checkpoint):
    state = dict(genuine_checkpoint[1].state_dict())
    del state["segmentation_head.0.weight"]
    path = tmp_path / "missing-key.pth"
    torch.save(state, path)
    with pytest.raises(ValueError, match="Strict SegWater checkpoint state loading failed") as error:
        load_segwater_checkpoint(path)
    assert 'Missing key(s) in state_dict: "segmentation_head.0.weight"' in str(error.value)


@pytest.mark.parametrize("shape,dtype", [
    ((1, 2, 225, 224), torch.float32),
    ((1, 3, 224, 224), torch.float32),
    ((2, 2, 224, 224), torch.float32),
    ((1, 2, 224, 224), torch.float64),
])
def test_adapter_enforces_fixed_eager_input_contract(imported, shape, dtype):
    with pytest.raises(ValueError, match="contract|float32"):
        imported.compilable_model(torch.zeros(shape, dtype=dtype))


def test_adapter_rejects_a_different_smp_resize_recipe(genuine_checkpoint):
    reference = genuine_checkpoint[1]
    previous = reference.decoder.blocks[0].interpolation_mode
    try:
        reference.decoder.blocks[0].interpolation_mode = "bilinear"
        with pytest.raises(ValueError, match="nearest decoding"):
            adapt_for_earth_engine(reference)
    finally:
        reference.decoder.blocks[0].interpolation_mode = previous


def test_trace_and_effective_convolutions_use_the_existing_compiler_subset(imported):
    graph = symbolic_trace(imported.compilable_model)
    assert sum(node.op == "call_function" and node.target is operator.add for node in graph.graph.nodes) == 8
    assert sum(node.op == "call_function" and node.target is torch.cat for node in graph.graph.nodes) == 4
    resizes = [node for node in graph.graph.nodes if node.op == "call_function" and node.target is F.interpolate]
    assert len(resizes) == 5
    assert all(node.kwargs["scale_factor"] == 2 and node.kwargs["size"] is None
               and node.kwargs["mode"] == "nearest" for node in resizes)
    assert not any(node.op in ("get_attr", "call_method") for node in graph.graph.nodes)
    parameters = effective_convolutions(imported.compilable_model)
    assert len(parameters) == 31
    assert all(parameter.dtype == "float32" and parameter.bias is not None for parameter in parameters)
    assert sum(parameter.weights.size + parameter.bias.size for parameter in parameters) == 14_319_426
    assert parameters[0].source == "encoder.conv1+encoder.bn1"
    assert parameters[-1].source == "segmentation_head.0"


def test_offline_compiler_folds_all_batchnorms_and_keeps_the_224_grid(imported):
    offline = ApiTestCase()
    offline.setUp()
    try:
        grid = Grid("EPSG:32651", (10, 0, 500000, 0, -10, 2600000), 224, 224)
        image = ee.Image.constant([1, -2]).rename(["VV", "VH"])
        requested = []
        def provider(key, module):
            requested.append(key)
            row_length = module.weight.numel() // module.out_channels
            weight = ee.Array([[0.0]]).float().repeat(0, module.out_channels).repeat(1, row_length)
            bias = ee.Array([[0.0]]).float().repeat(0, module.out_channels) if module.bias is not None else None
            return weight, bias
        compiled = compile_model(imported.compilable_model, image, ["VV", "VH"], grid,
                                 weight_provider=provider, include_unfused_nodes=False)
        assert compiled.grid == grid
        assert compiled.bands == ("c0", "c1")
        assert len(compiled.folded_batchnorms) == 30
        assert requested == [parameter.key for parameter in effective_convolutions(imported.compilable_model)]
        assert len(requested) == 31
        with pytest.raises(UnsupportedOperation, match="fixed CHW input"):
            compile_model(imported.compilable_model, image, ["VV", "VH"],
                          Grid(grid.crs, grid.transform, 256, 256), weight_provider=provider)
    finally:
        offline.tearDown()
