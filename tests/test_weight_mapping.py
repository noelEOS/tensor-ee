"""Exact coefficient identity is required before existing assets are reused."""
from dataclasses import replace

import pytest
import torch
from torch import nn

from tensor_ee.weight_assets import effective_convolutions
from tensor_ee.weight_mapping import MatchedWeightProvider


def fixture():
    model = nn.Sequential(nn.Conv2d(2, 3, 3, padding=1), nn.BatchNorm2d(3),
                          nn.ReLU(), nn.Conv2d(3, 2, 1)).eval()
    layers = {}
    for i, parameter in enumerate(effective_convolutions(model)):
        parameter = replace(parameter, key=f"saved_{i}")
        layers[parameter.key] = {"shape": list(parameter.shape), "dtype": "float32",
                                 "bias": True, "sha256": parameter.sha256, "assets": ["unused"]}
    return model, {"format": "native_ee_featurecollection_weights_v2", "status": "ready", "layers": layers}


def test_mapping_uses_all_fused_bytes_and_rechecks_compiler_coefficients(monkeypatch):
    model, manifest = fixture()
    mapped = MatchedWeightProvider(model, manifest)
    assert mapped.mapping == {"_1": "saved_0", "_3": "saved_1"}
    monkeypatch.setattr(mapped.provider, "arrays", lambda key: (key, None))
    assert mapped("_3", model[3]) == ("saved_1", None)
    assert mapped.used_asset_keys == {"saved_1"}
    assert not mapped.fallback_keys
    with torch.no_grad():
        model[3].bias[0] += 1e-3
    with pytest.raises(ValueError, match="fingerprint"):
        mapped("_3", model[3])


def test_same_shape_different_coefficient_or_missing_layer_cannot_reuse_assets():
    model, manifest = fixture()
    with torch.no_grad():
        model[0].weight[0, 0, 0, 0] += 1e-3
    with pytest.raises(ValueError, match="exact coefficient"):
        MatchedWeightProvider(model, manifest)
    manifest["layers"].pop("saved_0")
    with pytest.raises(ValueError, match="same number"):
        MatchedWeightProvider(model, manifest)
