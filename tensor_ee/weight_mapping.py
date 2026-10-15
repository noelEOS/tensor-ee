"""Reuse assets only after matching every effective float32 coefficient hash."""
from dataclasses import replace

from tensor_ee.weight_assets import FeatureCollectionWeightProvider, effective_convolutions


class MatchedWeightProvider:
    """Map FX names to asset names without trusting traversal order or shapes alone.

    Fingerprints cover all weight and bias bytes. The underlying provider also
    checks the actual convolution passed by the compiler on every call.
    """

    def __init__(self, model, manifest):
        self.provider = FeatureCollectionWeightProvider(manifest, fallback=False)
        self.manifest = self.provider.manifest
        layers = self.manifest["layers"]
        self.mapping = {}
        self.matches = []
        parameters = effective_convolutions(model)
        if len(parameters) != len(layers):
            raise ValueError("Model and asset manifest must contain the same number of convolutions")
        for parameter in parameters:
            matches = []
            for key, metadata in layers.items():
                if (list(parameter.shape) == metadata["shape"]
                        and parameter.dtype == metadata["dtype"]
                        and (parameter.bias is not None) == metadata["bias"]
                        and replace(parameter, key=key).sha256 == metadata["sha256"]):
                    matches.append(key)
            # Same-key matching also supports models with duplicate tensors.
            key = parameter.key if parameter.key in matches else matches[0] if len(matches) == 1 else None
            if key is None or key in self.mapping.values():
                raise ValueError(f"No unique exact coefficient match for {parameter.key!r}")
            self.mapping[parameter.key] = key
            self.matches.append({"fx_key": parameter.key, "source": parameter.source,
                                 "asset_key": key, "sha256": layers[key]["sha256"],
                                 "shape": list(parameter.shape), "dtype": parameter.dtype})
        if set(self.mapping.values()) != set(layers):
            raise ValueError("Asset mapping must cover every convolution exactly once")

    @property
    def used_asset_keys(self):
        return self.provider.used_asset_keys

    @property
    def fallback_keys(self):
        return self.provider.fallback_keys

    def __call__(self, key, module):
        return self.provider(self.mapping[key], module)
