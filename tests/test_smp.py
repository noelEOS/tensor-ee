"""SMP adapter contracts, offline graph construction and opt-in coverage."""
from copy import deepcopy
import json
import os
from types import MethodType

import ee
from ee.apitestcase import ApiTestCase
import pytest
import torch
from torch import nn

smp = pytest.importorskip("segmentation_models_pytorch")

from tensor_ee import (Grid, MatchedWeightProvider, UnsupportedOperation,
                       check_compatibility, compile_model, effective_convolutions)
from tensor_ee.segwater_pytorch import Fixed224SMPUnet, build_smp_resnet18_unet
from tensor_ee.smp import adapt_smp_model
from tensor_ee.converter.request import deep_recursion


class _Encoder(nn.Module):
    output_stride = 1

    def forward(self, x):
        return [x]


class _Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 2, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, features):
        return self.relu(self.conv(features[0]))


def tiny_smp(cls=None):
    # Real SMP base forward and class validation, with small test components.
    cls = smp.Unet if cls is None else cls
    model = cls.__new__(cls)
    nn.Module.__init__(model)
    model.encoder, model.decoder = _Encoder(), _Decoder()
    model.segmentation_head = nn.Identity()
    model.classification_head = None
    return model.eval()


def placeholder_weights(key, module):
    """Compact symbolic arrays; no learned coefficients serialized in graphs."""
    rows = module.out_channels
    cols = module.weight.numel() // rows
    weights = ee.Array(ee.List.repeat(ee.List.repeat(0, cols), rows))
    bias = None if module.bias is None else ee.Array(ee.List.repeat([0], rows))
    return weights, bias


@pytest.fixture
def offline():
    case = ApiTestCase()
    case.setUp()
    try:
        yield
    finally:
        case.tearDown()


def build(model, shape, **kwargs):
    _, channels, height, width = shape
    grid = Grid("EPSG:3857", (10, 0, 0, 0, -10, 0), width, height)
    bands = tuple(f"b{i}" for i in range(channels))
    return compile_model(model, ee.Image.constant([0] * channels).rename(bands), bands, grid,
                         weight_provider=placeholder_weights, **kwargs)


@pytest.mark.parametrize("architecture", ["Unet", "UnetPlusPlus"])
def test_standard_resnet18_at64(architecture, offline):
    torch.set_num_threads(2)
    model = getattr(smp, architecture)("resnet18", encoder_weights=None, in_channels=2, classes=2).eval()
    adapted = adapt_smp_model(model, input_shape=(1, 2, 64, 64))
    assert adapted.tensor_ee_smp_metadata["torch_equal"]
    assert adapted.tensor_ee_smp_metadata["max_abs"] == 0
    assert check_compatibility(adapted, (1, 2, 64, 64)).status == "supported"
    compiled = build(adapted, (1, 2, 64, 64), include_unfused_nodes=False)
    assert (len(compiled.bands), compiled.grid.height, compiled.grid.width) == (2, 64, 64)


def test_refusal_names_real_arithmetic(offline):
    model = smp.Linknet("resnet18", encoder_weights=None, in_channels=2, classes=2).eval()
    adapted = adapt_smp_model(model, input_shape=(1, 2, 64, 64))
    report = check_compatibility(adapted, (1, 2, 64, 64)).to_dict()
    assert report["status"] == "unsupported" and report["trace"]["status"] == "success"
    assert any("ConvTranspose2d" in b["message"] for b in report["blockers"])
    with pytest.raises(UnsupportedOperation, match="ConvTranspose2d"):
        build(adapted, (1, 2, 64, 64))


def test_source_rng_and_input_contract_preserved(offline):
    model = tiny_smp()
    state = deepcopy(model.state_dict())
    rng = torch.get_rng_state().clone()
    adapted = adapt_smp_model(model, input_shape=(1, 2, 8, 12))
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(value, state[key]) for key, value in model.state_dict().items())
    assert model.decoder.relu.inplace and all(p.requires_grad for p in model.parameters())
    assert not any(p.requires_grad for p in adapted.parameters())
    assert not any(m.training for m in adapted.modules())
    assert adapted.decoder.relu.inplace is False
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(model.parameters(), adapted.parameters()))
    json.dumps(adapted.tensor_ee_smp_metadata, allow_nan=False)
    x = torch.arange(192, dtype=torch.float32).reshape(1, 2, 8, 12) / 128
    assert torch.equal(model(x), adapted(x=x))
    for tensor in (torch.zeros(2, 2, 8, 12), torch.zeros(1, 2, 8, 8), x.double()):
        with pytest.raises(ValueError, match="fixed NCHW|float32"):
            adapted(tensor)
    with pytest.raises(ValueError, match="one input"):
        adapted(x, x=x)
    assert check_compatibility(adapted, (1, 2, 8, 8)).status == "unsupported"
    with pytest.raises(UnsupportedOperation, match="fixed CHW"):
        build(adapted, (1, 2, 8, 8))
    # The specialized module still works with existing weight collection/matching.
    weights = effective_convolutions(adapted)
    assert len(weights) == 1
    parameter = weights[0]
    manifest = {"format": "native_ee_featurecollection_weights_v2", "status": "ready", "layers": {
        parameter.key: {"shape": list(parameter.shape), "dtype": "float32", "bias": True,
                        "sha256": parameter.sha256, "assets": ["unused"]}}}
    provider = MatchedWeightProvider(adapted, manifest)
    assert provider.mapping == {parameter.key: parameter.key}


@pytest.mark.parametrize("shape", [(2, 2, 8, 8), (1, 2, 0, 8), (True, 2, 8, 8), [1, 2, 8, 8], (1, 2, 8)])
def test_invalid_input_shapes(shape):
    with pytest.raises(ValueError, match="batch-one NCHW"):
        adapt_smp_model(tiny_smp(), input_shape=shape)


@pytest.mark.parametrize("case", ["training", "dtype", "nonfinite", "head", "hook", "forward", "version"])
def test_model_contract_refusals(case, monkeypatch):
    model = tiny_smp()
    if case == "training":
        model.decoder.train()
    elif case == "dtype":
        model.double()
    elif case == "nonfinite":
        with torch.no_grad():
            model.decoder.conv.weight.fill_(float("nan"))
    elif case == "head":
        model.classification_head = nn.Identity()
    elif case == "hook":
        model.decoder.register_forward_pre_hook(lambda m, a: None)
    elif case == "forward":
        model.forward = MethodType(lambda self, x: x, model)
    else:
        monkeypatch.setattr(smp, "__version__", "0.0.0")
    patterns = dict(training="eval", dtype="float32", nonfinite="finite", head="classification_head",
                    hook="hooks", forward="forward overrides", version="requires SMP 0.5.0")
    with pytest.raises(ValueError, match=patterns[case]):
        adapt_smp_model(model, input_shape=(1, 2, 8, 8))


@pytest.mark.parametrize("cls", [smp.Segformer, smp.DPT])
def test_attention_and_forward_override_classes_refused(cls):
    with pytest.raises(ValueError, match="tensor_ee.converter.convert|DPT"):
        adapt_smp_model(tiny_smp(cls), input_shape=(1, 2, 8, 8))


@pytest.mark.parametrize("cls", [smp.Unet, smp.UnetPlusPlus, smp.MAnet, smp.Linknet, smp.FPN,
                                 smp.PSPNet, smp.PAN, smp.DeepLabV3, smp.DeepLabV3Plus, smp.UPerNet])
def test_base_forward_classes_are_eligible_without_an_encoder_allowlist(cls):
    adapted = adapt_smp_model(tiny_smp(cls), input_shape=(1, 2, 8, 8))
    assert adapted.tensor_ee_smp_metadata["torch_equal"]


@pytest.mark.parametrize("head, status", [(nn.Sigmoid, "supported"), (nn.Softplus, "unsupported")])
def test_segmentation_head_arithmetic_is_preserved(head, status):
    model = tiny_smp()
    model.segmentation_head = head().eval()
    adapted = adapt_smp_model(model, input_shape=(1, 2, 8, 8))
    report = check_compatibility(adapted, (1, 2, 8, 8)).to_dict()
    assert report["status"] == status
    if status == "unsupported":
        assert any("Softplus" in b["message"] for b in report["blockers"])


def test_mismatch_and_trace_failure_preserve_rng():
    class ChangedHead(nn.Module):
        inplace = True

        def forward(self, x):
            return x + (1 if self.inplace else 2)

    class Untraceable(nn.Module):
        def forward(self, features):
            return features[0] if features[0].sum() > 0 else -features[0]

    for component, pattern in ((ChangedHead(), "not bit-equal.*max_abs="),
                               (Untraceable(), "SMP FX trace failed")):
        model = tiny_smp()
        if isinstance(component, ChangedHead):
            model.segmentation_head = component.eval()
        else:
            model.decoder = component.eval()
        rng = torch.get_rng_state().clone()
        with pytest.raises(ValueError, match=pattern):
            adapt_smp_model(model, input_shape=(1, 2, 8, 8))
        assert torch.equal(rng, torch.get_rng_state())


def test_segwater_recipe_graph_identity(offline):
    torch.set_num_threads(2)
    raw = build_smp_resnet18_unet()
    generic = adapt_smp_model(raw, input_shape=(1, 2, 224, 224))
    fixed = Fixed224SMPUnet(raw)
    generic_weights, fixed_weights = effective_convolutions(generic), effective_convolutions(fixed)
    assert len(generic_weights) == len(fixed_weights) == 31
    assert [(x.key, x.source, x.sha256) for x in generic_weights] == [(x.key, x.source, x.sha256) for x in fixed_weights]
    for diagnostics in (False, True):
        for clip in (False, True):
            kwargs = dict(include_unfused_nodes=diagnostics, clip_footprints=clip)
            a, b = build(generic, (1, 2, 224, 224), **kwargs), build(fixed, (1, 2, 224, 224), **kwargs)
            with deep_recursion():
                assert a.image.serialize() == b.image.serialize()
            assert tuple(a.nodes) == tuple(b.nodes)
            assert a.folded_batchnorms == b.folded_batchnorms
    print("SegWater recipe: 4/4 graph identities; 31/31 effective convolution keys, sources and hashes")


def _grouped_encoder_build(encoder):
    shape = (1, 2, 64, 64)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(73)
        model = smp.Unet(encoder, encoder_weights=None, in_channels=2, classes=2).eval()
        adapted = adapt_smp_model(model, input_shape=shape)
    report = check_compatibility(adapted, shape).to_dict()
    assert report["status"] == "supported", report["blockers"]
    compiled = build(adapted, shape, include_unfused_nodes=False)
    assert compiled.bands == ("c0", "c1")
    assert (compiled.grid.height, compiled.grid.width) == (64, 64)
    assert any(module.groups > 1 for module in adapted.modules() if isinstance(module, nn.Conv2d))
    weights = effective_convolutions(adapted)
    assert len(weights) > 0 and all(w.weights.shape[0] > 0 and w.weights.shape[1] > 0 for w in weights)
    meta = adapted.tensor_ee_smp_metadata
    assert meta["torch_equal"] and meta["max_abs"] == 0
    return meta


@pytest.mark.parametrize("encoder", ["resnext50_32x4d", "mobilenet_v2"])
def test_grouped_encoders_at64(encoder, offline):
    meta = _grouped_encoder_build(encoder)
    print(f"Unet/{encoder}/64: supported, builds, torch adapter equal={meta['torch_equal']}, max_abs={meta['max_abs']}")


@pytest.mark.skipif(os.environ.get("TENSOR_EE_SMP_MATRIX") != "1", reason="set TENSOR_EE_SMP_MATRIX=1 for full SMP coverage")
def test_full_smp_matrix(offline):
    torch.set_num_threads(2)
    for architecture in ("Unet", "UnetPlusPlus"):
        for encoder in ("resnet18", "resnet34", "resnet50", "vgg11_bn"):
            model = getattr(smp, architecture)(encoder, encoder_weights=None, in_channels=2, classes=2).eval()
            for side in (64, 128, 224):
                shape = (1, 2, side, side)
                adapted = adapt_smp_model(model, input_shape=shape)
                report = check_compatibility(adapted, shape).to_dict()
                assert report["status"] == "supported", report["blockers"]
                build(adapted, shape, include_unfused_nodes=False)
                meta = adapted.tensor_ee_smp_metadata
                assert meta["torch_equal"] and meta["max_abs"] == 0
                print(f"{architecture}/{encoder}/{side}: supported, builds, equal=True, max_abs={meta['max_abs']}")
                del adapted
            del model
    for encoder in ("resnext50_32x4d", "mobilenet_v2"):
        meta = _grouped_encoder_build(encoder)
        print(f"Unet/{encoder}/64: supported, builds, torch adapter max_abs={meta['max_abs']}")
    negatives = [("Linknet", "resnet18", "ConvTranspose2d"), ("FPN", "resnet18", "GroupNorm"),
                 ("DeepLabV3Plus", "resnet18", "dilation"), ("Unet", "densenet121", "BatchNorm")]
    for architecture, encoder, blocker in negatives:
        model = getattr(smp, architecture)(encoder, encoder_weights=None, in_channels=2, classes=2).eval()
        adapted = adapt_smp_model(model, input_shape=(1, 2, 64, 64))
        report = check_compatibility(adapted, (1, 2, 64, 64)).to_dict()
        assert report["status"] == "unsupported" and report["trace"]["status"] == "success"
        assert any(blocker in x["message"] for x in report["blockers"]), report["blockers"]
        with pytest.raises(UnsupportedOperation):
            build(adapted, (1, 2, 64, 64))
        meta = adapted.tensor_ee_smp_metadata
        print(f"{architecture}/{encoder}: refused ({blocker}), equal={meta['torch_equal']}, max_abs={meta['max_abs']}")
        del adapted, model
    for architecture in ("Unet", "UnetPlusPlus"):
        model = getattr(smp, architecture)("resnet18", encoder_weights=None, in_channels=3, classes=3,
                                           decoder_channels=(16, 8, 8, 4, 4)).eval()
        adapted = adapt_smp_model(model, input_shape=(1, 3, 64, 96))
        assert check_compatibility(adapted, (1, 3, 64, 96)).status == "supported"
        build(adapted, (1, 3, 64, 96), include_unfused_nodes=False)
        print(f"{architecture}/resnet18/3ch/3classes/64x96: supported, builds, equal=True, max_abs=0.0")
