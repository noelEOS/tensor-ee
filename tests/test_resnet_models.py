"""Architecture and fixture checks; live EE parity is an experiment runner."""

import operator

import pytest
import torch
from torch import nn
from torch.fx import Interpreter, symbolic_trace

from reference.cnn import capture_torch, synthetic_tensor
from tensor_ee.models import parameter_count
from tensor_ee.resnet_models import (BasicBlock, ResNet18Encoder, ResNet18UNet,
                                     initialize_resnet_fixture)


def test_standard_encoder_matches_torchvision_resnet18():
    """Load identical tensors into official torchvision and compare every skip."""
    torchvision = pytest.importorskip("torchvision")
    encoder = initialize_resnet_fixture(ResNet18Encoder(in_channels=3), seed=1800)
    official = torchvision.models.resnet18(weights=None).eval()
    official.fc = nn.Identity()
    official.load_state_dict(encoder.state_dict(), strict=True)
    tensor = synthetic_tensor(64, 64, channels=3)
    with torch.no_grad():
        expected_stem = official.relu(official.bn1(official.conv1(tensor)))
        expected_layer1 = official.layer1(official.maxpool(expected_stem))
        expected_layer2 = official.layer2(expected_layer1)
        expected_layer3 = official.layer3(expected_layer2)
        expected_layer4 = official.layer4(expected_layer3)
        actual = encoder(tensor)
    for expected, actual_skip in zip(
            (expected_stem, expected_layer1, expected_layer2,
             expected_layer3, expected_layer4), actual, strict=True):
        assert torch.equal(expected, actual_skip)


def test_resnet18_topology_and_parameter_counts():
    model = ResNet18UNet(in_channels=2, base_channels=64)
    encoder = model.encoder
    assert [len(getattr(encoder, f"layer{i}")) for i in range(1, 5)] == [2] * 4
    assert len([m for m in encoder.modules() if isinstance(m, BasicBlock)]) == 8
    assert encoder.conv1.kernel_size == (7, 7)
    assert encoder.conv1.stride == (2, 2)
    assert encoder.conv1.padding == (3, 3)
    assert (encoder.maxpool.kernel_size, encoder.maxpool.stride,
            encoder.maxpool.padding) == (3, 2, 1)
    assert len([m for m in encoder.modules() if isinstance(m, nn.Conv2d)]) == 20
    assert len([m for m in model.modules() if isinstance(m, nn.Conv2d)]) == 31
    assert len([m for m in model.modules() if isinstance(m, nn.Upsample)]) == 5
    assert parameter_count(encoder) == 11_173_376
    assert parameter_count(model) == 14_324_945
    assert not any(isinstance(m, nn.ReLU) and m.inplace for m in model.modules())


@pytest.mark.parametrize("base_channels", [8, 64])
def test_resnet_fixture_signed_bn_and_nonconstant_outputs(base_channels):
    model = initialize_resnet_fixture(ResNet18UNet(base_channels=base_channels))
    output, tensors = capture_torch(model, synthetic_tensor(32, 32))
    assert output.shape == (1, 1, 32, 32)
    assert torch.isfinite(output).all()
    assert output.std(unbiased=False) > 1e-4
    assert tensors["head"].std(unbiased=False) > 1e-4
    assert torch.allclose(output, tensors["head"].sigmoid())
    graph = symbolic_trace(model)
    assert sum(n.target is operator.add for n in graph.graph.nodes) == 8
    assert sum(n.target is torch.cat for n in graph.graph.nodes) == 4
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and module.weight.numel() >= 10:
            assert module.weight.min() < 0 < module.weight.max()
        elif isinstance(module, nn.BatchNorm2d):
            assert module.running_mean.abs().max() > 0
            assert torch.any(module.running_var != 1)
            assert torch.any(module.weight != 1)
            assert module.bias.abs().max() > 0


def test_resnet_fixture_decoder_uses_deep_and_skip_paths():
    class WithoutSkips(Interpreter):
        def call_function(self, target, args, kwargs):
            if target is torch.cat:
                args = ([args[0][0], torch.zeros_like(args[0][1])],)
            return super().call_function(target, args, kwargs)

    model = initialize_resnet_fixture(ResNet18UNet(base_channels=8))
    tensor = synthetic_tensor(32, 32)
    with torch.no_grad():
        output = model(tensor)
        without_skips = WithoutSkips(symbolic_trace(model)).run(tensor)
        handle = model.encoder.layer4.register_forward_hook(
            lambda module, args, result: torch.zeros_like(result))
        try:
            without_deep = model(tensor)
        finally:
            handle.remove()
    assert (output - without_skips).abs().max() > 1e-3
    assert (output - without_deep).abs().max() > 1e-4


def test_resnet_fixture_is_reproducible():
    model = initialize_resnet_fixture(ResNet18UNet(base_channels=8))
    duplicate = initialize_resnet_fixture(ResNet18UNet(base_channels=8))
    assert all(torch.equal(value, duplicate.state_dict()[key])
               for key, value in model.state_dict().items())
