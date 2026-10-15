import torch
from torch import nn
from torch.fx import Interpreter, symbolic_trace

from reference.cnn import capture_torch, cases, synthetic_tensor
from tensor_ee.models import TinyUNet, initialize_fixture, parameter_count


def test_fixtures_are_deterministic_and_nonconstant():
    for name, model, channels in cases():
        tensor = synthetic_tensor(16, 16, channels)
        output, nodes = capture_torch(model, tensor)
        assert output.shape[-2:] == (16, 16), name
        assert torch.isfinite(output).all(), name
        assert output.std(unbiased=False) > 1e-4, name
        assert torch.equal(output, model(tensor)), name
        if "unet" in name or name == "small_cnn":
            assert "head" in nodes
            assert torch.allclose(output, nodes["head"].sigmoid())


def test_spatial_weights_distinguish_kernel_orientation():
    model = initialize_fixture(nn.Conv2d(2, 3, 3, padding=1), 1)
    assert not torch.equal(model.weight, model.weight.flip(-1))
    assert not torch.equal(model.weight, model.weight.flip(-2))
    assert not torch.equal(model.weight, model.weight.transpose(-1, -2))
    copy = initialize_fixture(nn.Conv2d(2, 3, 3, padding=1), 1)
    assert torch.equal(model.weight, copy.weight)


def test_unet_skip_connections_and_parameter_budget():
    model = initialize_fixture(TinyUNet(base_channels=8, levels=2), 5)
    output, nodes = capture_torch(model, synthetic_tensor(16, 16))
    assert 10_000 < parameter_count(model) < 100_000
    concatenations = [tensor for name, tensor in nodes.items() if name.startswith("cat")]
    assert len(concatenations) == 2
    assert sorted(tensor.shape[1] for tensor in concatenations) == [24, 48]
    assert output.shape == (1, 1, 16, 16)


def test_pool_roundtrip_has_two_by_two_blocks():
    for name, model, channels in cases():
        if "pool_nearest" not in name:
            continue
        output = model(synthetic_tensor(16, 16, channels))
        assert torch.equal(output[:, :, ::2, ::2], output[:, :, 1::2, ::2])
        assert torch.equal(output[:, :, ::2, ::2], output[:, :, ::2, 1::2])


def test_unet_fixture_uses_both_skip_and_bottleneck_paths():
    class WithoutSkips(Interpreter):
        def call_function(self, target, args, kwargs):
            if target is torch.cat:
                args = ([args[0][0], torch.zeros_like(args[0][1])],)
            return super().call_function(target, args, kwargs)

    model = initialize_fixture(TinyUNet(base_channels=8, levels=2), 5)
    tensor = synthetic_tensor(16, 16)
    with torch.no_grad():
        reference = model(tensor)
        without_skips = WithoutSkips(symbolic_trace(model)).run(tensor)
        hook = model.bottom.relu2.register_forward_hook(
            lambda module, args, output: torch.zeros_like(output))
        try:
            without_bottleneck = model(tensor)
        finally:
            hook.remove()
    assert (reference - without_skips).abs().max() > 1e-3
    assert (reference - without_bottleneck).abs().max() > 1e-3
