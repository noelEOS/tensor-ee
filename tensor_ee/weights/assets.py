"""The FeatureCollection weight-asset format and its provider.

Moved from ``tensor_ee.weight_assets``: ``ConvolutionWeights``,
``convolution_weights``, ``feature_records`` and
``FeatureCollectionWeightProvider``. The old names are re-exports of these
objects, and ``FeatureCollectionWeightProvider`` is one class object everywhere
(``tensor_ee``, ``tensor_ee.weight_assets``, ``tensor_ee._swin_parameters``).

Each feature has a harmless point geometry and properties ``layer``, ``row``,
``chunk``, ``index``, ``weights_json`` (one JSON string of a numeric chunk), and,
when present, ``bias``. Chunks are sorted by an explicit integer index; no
inference weight is downloaded or embedded in an asset-backed graph. Nine
significant figures preserve every finite float32 value on restoration through
``ee.Array.float()``. Float32 is the maximum supported model precision.

Convolution modules are read by duck typing (``weight``, ``bias``,
``out_channels``). This module does not import torch at run time: ``torch.nn``
is imported only under ``TYPE_CHECKING``, so the ``nn.Conv2d`` annotations
resolve for type checkers and linters and are never evaluated
(``from __future__ import annotations``).
BatchNorm folding (``effective_convolutions``) and the export and upload
runners stay in ``tensor_ee.weight_assets`` with the converter (decision D6).
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Sequence

import ee
import numpy as np

if TYPE_CHECKING:  # annotations only; no runtime torch import in the weights layer
    from torch import nn


@dataclass(frozen=True)
class ConvolutionWeights:
    key: str
    weights: np.ndarray
    bias: np.ndarray | None
    dtype: str
    source: str

    @property
    def shape(self) -> tuple[int, int]:
        return self.weights.shape

    @property
    def sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(json.dumps([self.key, list(self.shape), self.dtype], separators=(",", ":")).encode())
        digest.update(np.ascontiguousarray(self.weights).tobytes())
        if self.bias is not None:
            digest.update(np.ascontiguousarray(self.bias).tobytes())
        return digest.hexdigest()


def convolution_weights(key: str, module: nn.Conv2d, source: str | None = None) -> ConvolutionWeights:
    dtype = str(module.weight.dtype).replace("torch.", "")
    if dtype != "float32":
        raise ValueError(f"Only float32 asset tensors are supported, received {dtype}")
    weights = module.weight.detach().cpu().numpy().reshape(module.out_channels, -1).copy()
    bias = None if module.bias is None else module.bias.detach().cpu().numpy().copy()
    if not np.isfinite(weights).all() or (bias is not None and not np.isfinite(bias).all()):
        raise ValueError("Weight assets require finite tensors")
    return ConvolutionWeights(key, weights, bias, dtype, source or key)


def feature_records(parameters: Sequence[ConvolutionWeights], chunk_values: int = 4096) -> Iterator[dict[str, Any]]:
    if not 1 <= chunk_values <= 4096:
        raise ValueError("chunk_values must be between 1 and 4096")
    for parameter in parameters:
        chunks_per_row = math.ceil(parameter.shape[1] / chunk_values)
        if parameter.dtype != "float32":
            raise ValueError("Only float32 asset tensors are supported")
        precision = ".9g"
        for row_index, row in enumerate(parameter.weights):
            for chunk_index in range(chunks_per_row):
                part = row[chunk_index * chunk_values:(chunk_index + 1) * chunk_values]
                encoded = "[" + ",".join(format(float(value), precision) for value in part) + "]"
                # EE table-asset string values are bounded; 4096 values leaves
                # ample room even for signs and scientific notation.
                if len(encoded.encode()) > 90_000:
                    raise ValueError("Encoded weight chunk exceeds conservative string limit")
                properties = {"layer": parameter.key, "row": row_index, "chunk": chunk_index,
                              "index": row_index * chunks_per_row + chunk_index, "weights_json": encoded}
                if parameter.bias is not None and chunk_index == 0:
                    properties["bias"] = float(format(float(parameter.bias[row_index]), precision))
                yield {"type": "Feature", "geometry": {"type": "Point", "coordinates": [0, 0]},
                       "properties": properties}


class FeatureCollectionWeightProvider:
    """Reconstruct a convolution directly from persistent EE FC asset rows.

    Local modules are checked against manifest SHA-256 fingerprints once per
    node.  Asset values stay in the EE graph; this class performs no EE requests.
    Missing diagnostic convolution keys may use local arrays only via explicit ``fallback=True``.
    Folded convolutions in the final graph must exist in the manifest.
    """
    def __init__(self, manifest: Mapping[str, Any] | str | Path, *, fallback: bool = False):
        if isinstance(manifest, (str, Path)):
            manifest = json.loads(Path(manifest).read_text())
        if manifest.get("format") != "native_ee_featurecollection_weights_v2" or manifest.get("status") != "ready":
            raise ValueError("A completed compatible weight manifest is required")
        self.manifest = manifest
        self.fallback = fallback
        self.used_asset_keys: set[str] = set()
        self.fallback_keys: set[str] = set()
        self._cache: dict[str, tuple[ee.Array, ee.Array | None]] = {}

    @classmethod
    def from_manifest(cls, path: str | Path, *, fallback: bool = False):
        return cls(path, fallback=fallback)

    def arrays(self, key: str) -> tuple[ee.Array, ee.Array | None]:
        metadata = self.manifest["layers"][key]
        if metadata["dtype"] != "float32":
            raise ValueError("Only float32 asset tensors are supported")
        assets = metadata["assets"]
        if not assets:
            raise ValueError(f"No assets recorded for {key}")
        collection = ee.FeatureCollection(assets[0]).filter(ee.Filter.eq("layer", key))
        for asset_id in assets[1:]:
            collection = collection.merge(ee.FeatureCollection(asset_id).filter(ee.Filter.eq("layer", key)))
        ordered = collection.sort("index")
        flat_values = ordered.aggregate_array("weights_json").map(lambda value: ee.String(value).decodeJSON()).flatten()
        weight = ee.Array(flat_values).reshape(metadata["shape"])
        bias_rows = collection.filter(ee.Filter.eq("chunk", 0)).sort("row")
        bias = ee.Array(bias_rows.aggregate_array("bias")).reshape([metadata["shape"][0], 1]) if metadata["bias"] else None
        if metadata["dtype"] == "float32":
            weight = weight.float()
            if bias is not None:
                bias = bias.float()
        return weight, bias

    def __call__(self, key: str, module: nn.Conv2d) -> tuple[ee.Array, ee.Array | None]:
        parameter = convolution_weights(key, module)
        metadata = self.manifest["layers"].get(key)
        if metadata is None:
            if not self.fallback:
                raise KeyError(f"Convolution {key!r} is absent from asset manifest")
            self.fallback_keys.add(key)
            weights = ee.Array(parameter.weights.astype(float).tolist()).float()
            bias = ee.Array(parameter.bias.astype(float).reshape(-1, 1).tolist()).float() if parameter.bias is not None else None
        else:
            if list(parameter.shape) != metadata["shape"] or parameter.dtype != metadata["dtype"] or (parameter.bias is not None) != metadata["bias"]:
                raise ValueError(f"Weight shape, dtype, or bias mismatch for {key!r}")
            if parameter.sha256 != metadata["sha256"]:
                raise ValueError(f"Weight fingerprint mismatch for {key!r}; this manifest belongs to another fixture")
            self.used_asset_keys.add(key)
            if key in self._cache:
                return self._cache[key]
            weights, bias = self.arrays(key)
        self._cache[key] = weights, bias
        return weights, bias
