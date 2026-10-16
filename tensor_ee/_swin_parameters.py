"""Local accounting and exact asset record mapping for the Swin-UPerNet composites.

The fixed Swin-T224 adapter (``tensor_ee.swin``) and workflow (``tensor_ee.swin_workflow``)
use ``SwinTParameters`` and the loaders here, at Swin-T224's settings. The schema, the
checkpoint mapping, the parameter checks and the weight records also take a
``SwinUPerNetConfig`` (``tensor_ee.models.swin_t224.configuration``; default ``SWIN_T224``):
``parameter_records`` yields the ``ConvolutionWeights`` records of any configuration, under
the timm and SMP names the composites' loader (``with_manifest``) reads, in the v2 manifest
format (``native_ee_featurecollection_weights_v2``) that ``export_weights`` writes.
Swin-T224's 189 records are unchanged."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import re
from typing import Any, Literal, Mapping

import numpy as np
import torch

from .core.errors import UnsupportedOperation
from .models.swin_t224.configuration import DEPTHS, SWIN_T224, SwinUPerNetConfig, block_shifts
from .weights.assets import ConvolutionWeights, FeatureCollectionWeightProvider
from .weights.constants import ServerArray

ENC_PREFIX = "encoder.model."
PE_PREFIX = ENC_PREFIX + "patch_embed."
ENCODER_NAME = "tu-swin_tiny_patch4_window7_224"


def parameter_shapes(config: SwinUPerNetConfig = SWIN_T224):
    """Complete inference schema in the historical three-map layouts (encoder groups, patch
    embedding, decoder, BatchNorm counters): ``config.parameter_shapes()``; Swin-T224's by
    default."""
    return config.parameter_shapes()


def encoder_prefix(group):
    if group.startswith("pm"):
        return f"layers.{int(group[2:])}.downsample."
    s, b = group[1:].split("b")
    return f"layers.{int(s)-1}.blocks.{b}."


def checkpoint_shapes(config: SwinUPerNetConfig = SWIN_T224):
    """The SMP UPerNet checkpoint's tensor shapes (``encoder.model.layers_{s}...``,
    ``encoder.model.patch_embed.*`` with the Conv2d weight [embed_dim, bands, patch, patch],
    the decoder and head, and the BatchNorm counters)."""
    enc, pe, dec, counters = parameter_shapes(config)
    out = {ENC_PREFIX + encoder_prefix(g).replace("layers.", "layers_", 1) + n: shape
           for g, params in enc.items() for n, shape in params.items()}
    conv = (config.embed_dim, config.in_chans, config.patch, config.patch)
    out.update({PE_PREFIX+n: (conv if n == "proj.weight" else shape) for n, shape in pe.items()})
    out.update(dec)
    out.update(counters)
    return out


def _sha(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise UnsupportedOperation(f"{name} must be a lowercase SHA-256 digest")


def _model(config: SwinUPerNetConfig) -> str:
    """The model's name in messages: "fixed Swin-T" (Swin-T224, the historical messages) or
    "configured Swin-UPerNet"."""
    return "fixed Swin-T" if config == SWIN_T224 else "configured Swin-UPerNet"


def validate_state(state, config: SwinUPerNetConfig = SWIN_T224):
    shapes = checkpoint_shapes(config)
    if not isinstance(state, Mapping) or set(state) != set(shapes):
        got = set(state) if isinstance(state, Mapping) else set()
        raise UnsupportedOperation(f"{_model(config)} checkpoint keys differ: missing={sorted(set(shapes)-got)}, extra={sorted(got-set(shapes))}")
    for k, shape in shapes.items():
        v = state[k]
        if not isinstance(v, torch.Tensor) or tuple(v.shape) != shape:
            raise UnsupportedOperation(f"{k}: expected tensor shape {shape}")
        if k.endswith("num_batches_tracked"):
            if v.dtype != torch.int64 or v.item() < 0:
                raise UnsupportedOperation(f"{k}: expected a nonnegative int64 BatchNorm counter")
        elif v.dtype != torch.float32 or not torch.isfinite(v).all():
            raise UnsupportedOperation(f"{k}: expected finite float32 coefficients")


def read_state(checkpoint: Path, expected_checkpoint_sha256: str):
    _sha(expected_checkpoint_sha256, "expected_checkpoint_sha256")
    # An open file can still be changed in place. Hash and deserialize one
    # immutable snapshot so the verified digest binds the returned coefficients.
    source_data = Path(checkpoint).read_bytes()
    if hashlib.sha256(source_data).hexdigest() != expected_checkpoint_sha256:
        raise UnsupportedOperation("checkpoint SHA-256 differs from expected_checkpoint_sha256")
    checkpoint_data = torch.load(io.BytesIO(source_data), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint_data, Mapping) or "model_state_dict" not in checkpoint_data:
        raise UnsupportedOperation("expected checkpoint with model_state_dict")
    state = checkpoint_data["model_state_dict"]
    validate_state(state)
    return state


def parameters_from_state(state, config: SwinUPerNetConfig = SWIN_T224):
    """Re-key only, preserving all coefficient values and original flatten order. The
    patch embedding's Conv2d weight becomes [embed_dim, bands x patch^2], flattened
    (c, ky, kx)."""
    validate_state(state, config)
    enc_shapes, _, _, _ = parameter_shapes(config)
    enc, pe, dec = {g: {} for g in enc_shapes}, {}, {}
    for k, tensor in state.items():
        if k.endswith("num_batches_tracked"):
            continue
        v = tensor.detach().cpu().numpy().copy()
        if k.startswith(PE_PREFIX):
            n = k[len(PE_PREFIX):]
            pe[n] = v.reshape(config.embed_dim, config.patch_columns) if n == "proj.weight" else v
        elif k.startswith(ENC_PREFIX):
            stage, rest = k[len(ENC_PREFIX+"layers_"):].split(".", 1)
            if rest.startswith("blocks."):
                b, n = rest[len("blocks."):].split(".", 1)
                enc[f"s{int(stage)+1}b{b}"][n] = v
            else:
                enc[f"pm{int(stage)}"][rest[len("downsample."):]] = v
        else:
            dec[k] = v
    return enc, pe, dec


def _validate_map(values, shapes, name, model="fixed Swin-T"):
    if not isinstance(values, Mapping) or set(values) != set(shapes):
        raise UnsupportedOperation(f"{name}: {model} parameter names differ")
    for k, shape in shapes.items():
        value = values[k]
        if isinstance(value, ServerArray):
            # Decoder asset convolutions use the historical flattened [Cout,Cin*k*k] record.
            accepted = ((shape[0], int(np.prod(shape[1:]))),) if len(shape) == 4 else (shape,)
            if tuple(value.shape) not in accepted:
                raise UnsupportedOperation(f"{name}.{k}: expected shape {shape}")
        elif not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise UnsupportedOperation(f"{name}.{k}: expected finite float32 array of shape {shape}")


def validate_parameters(encoder, patch, decoder, config: SwinUPerNetConfig = SWIN_T224) -> None:
    """The parameter checks: the three maps hold exactly the configuration's names (the
    encoder groups without a final norm), each a finite float32 NumPy array of its schema
    shape or a ``ServerArray`` of that shape (a convolution as [Cout, Cin k k])."""
    enc, pe, dec, _ = parameter_shapes(config)
    model = _model(config)
    if not isinstance(encoder, Mapping) or set(encoder) != set(enc):
        raise UnsupportedOperation(f"encoder: {model} groups differ (raw C4 has no final norm)")
    for g in enc:
        _validate_map(encoder[g], enc[g], g, model)
    _validate_map(patch, pe, "patch_embed", model)
    _validate_map(decoder, dec, "decoder", model)


@dataclass(frozen=True)
class SwinTParameters:
    """Validated Swin-T224 parameters: the fixed adapter's and workflow's input."""

    encoder: Mapping[str, Mapping[str, np.ndarray | ServerArray]]
    patch_embed: Mapping[str, np.ndarray | ServerArray]
    decoder: Mapping[str, np.ndarray | ServerArray]
    checkpoint_sha256: str
    manifest_sha256: str | None = None

    def __post_init__(self):
        self.validate()

    def validate(self):
        _sha(self.checkpoint_sha256, "checkpoint_sha256")
        if self.manifest_sha256 is not None:
            _sha(self.manifest_sha256, "manifest_sha256")
        validate_parameters(self.encoder, self.patch_embed, self.decoder, SWIN_T224)


def records_and_locations(encoder, patch, decoder):
    """The existing row/chunk/record format, with each source tensor's read location."""
    records, locations = {}, {}
    groups = [(g, p, encoder_prefix(g)) for g, p in encoder.items()]
    groups += [("patch_embed", patch, "patch_embed."), ("decoder", decoder, "")]
    for group, params, prefix in groups:
        for n, value in params.items():
            base = n.removesuffix(".bias")
            weight_name = base + ".weight"
            if n.endswith(".bias") and weight_name in params and params[weight_name].ndim >= 2:
                key, part = prefix+base, "bias"
            elif value.ndim == 1:
                key, part = prefix+n, "vector"
            elif n.endswith(".weight"):
                key, part = prefix+n.removesuffix(".weight"), "weight"
            else:
                key, part = prefix+n, "matrix"
            locations[(group,n)] = (key, part, value.shape)
            if part == "bias":
                continue
            matrix = value.reshape(1,-1) if part == "vector" else value.reshape(value.shape[0],-1)
            bias = params.get(n.removesuffix(".weight")+".bias") if part == "weight" else None
            if key in records:
                raise UnsupportedOperation(f"duplicate parameter record {key}")
            records[key] = ConvolutionWeights(key, matrix.copy(), None if bias is None else bias.copy(), "float32", prefix+n)
    return records, locations


def parameter_records(encoder, patch, decoder, config: SwinUPerNetConfig = SWIN_T224) -> list[ConvolutionWeights]:
    """The ``ConvolutionWeights`` records of a configuration's local parameters, for
    ``export_weights`` (a v2 manifest) and ``with_manifest`` (the composites' loader).

    The three maps are the historical layouts (``parameters_from_state``): encoder groups
    ``s{s}b{b}`` / ``pm{n}`` with timm names, the patch embedding, and the decoder and head
    with SMP names. They must pass ``validate_parameters`` for ``config`` and be NumPy arrays
    (server arrays cannot be re-exported). The records are ``records_and_locations``'s, in
    the maps' order: for Swin-T224 exactly ``swin_workflow.weight_records``'s 189. No
    requests occur."""
    validate_parameters(encoder, patch, decoder, config)
    maps = [*encoder.values(), patch, decoder]
    if any(not isinstance(value, np.ndarray) for values in maps for value in values.values()):
        raise UnsupportedOperation("parameter_records requires local NumPy parameters, without a manifest")
    records, _ = records_and_locations(encoder, patch, decoder)
    return list(records.values())


def with_manifest(encoder, patch, decoder, manifest, checkpoint_sha256):
    if not isinstance(manifest, Mapping) or manifest.get("source_checkpoint_sha256") != checkpoint_sha256:
        raise UnsupportedOperation("manifest source_checkpoint_sha256 differs from the checkpoint")
    if manifest.get("format") != "native_ee_featurecollection_weights_v2" or manifest.get("status") != "ready":
        raise UnsupportedOperation("a completed native_ee_featurecollection_weights_v2 manifest is required")
    records, locations = records_and_locations(encoder, patch, decoder)
    layers = manifest.get("layers")
    if not isinstance(layers, Mapping) or set(layers) != set(records):
        raise UnsupportedOperation("manifest must account for every fixed Swin-T record exactly once")
    for key, record in records.items():
        meta = layers[key]
        if not isinstance(meta, Mapping) or (meta.get("shape") != list(record.shape) or meta.get("dtype") != "float32"
            or meta.get("bias") is not (record.bias is not None) or meta.get("sha256") != record.sha256):
            raise UnsupportedOperation(f"manifest record {key}: shape/dtype/bias/fingerprint differs from checkpoint")
        assets = meta.get("assets")
        if not isinstance(assets, list) or not assets or any(not isinstance(v,str) or not v for v in assets) or len(set(assets)) != len(assets):
            raise UnsupportedOperation(f"manifest record {key}: explicit unique asset IDs are required")
    listing = json.dumps(sorted((key, r.sha256) for key,r in records.items()), separators=(",",":"))
    if manifest.get("manifest_hash") != hashlib.sha256(listing.encode()).hexdigest():
        raise UnsupportedOperation("manifest_hash differs from the exact parameter record listing")
    provider = FeatureCollectionWeightProvider(manifest)
    arrays = {k: provider.arrays(k) for k in records}
    output = {g:{} for g in encoder}
    output.update(patch_embed={},decoder={})
    for (g,n),(key,part,shape) in locations.items():
        weight, bias = arrays[key]
        if part == "bias":
            array = bias.reshape(list(shape))
        elif part == "vector":
            array = weight.reshape(list(shape))
        else:
            array = weight
            shape = tuple(records[key].shape)
        output[g][n] = ServerArray(array,shape)
    sha = hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    return {g:output[g] for g in encoder},output['patch_embed'],output['decoder'],sha


def load_swin_t224_parameters(checkpoint: Path, *, expected_checkpoint_sha256: str,
                             manifest: Mapping[str, Any] | None = None) -> SwinTParameters:
    state = read_state(checkpoint, expected_checkpoint_sha256)
    enc, pe, dec = parameters_from_state(state)
    manifest_sha = None
    if manifest is not None:
        enc, pe, dec, manifest_sha = with_manifest(enc, pe, dec, manifest, expected_checkpoint_sha256)
    return SwinTParameters(enc, pe, dec, expected_checkpoint_sha256, manifest_sha)


def build_model(state, attention="default"):
    """Strict local reference model; no download, EE initialization, or forward pass."""
    if attention not in ("default","explicit"):
        raise UnsupportedOperation("attention must be 'default' or 'explicit'")
    validate_state(state)
    import segmentation_models_pytorch as smp
    # Construction must not round the checkpoint through an ambient half dtype
    # or place the reference on an ambient accelerator default device.
    with torch.device("cpu"):
        model = smp.UPerNet(encoder_name=ENCODER_NAME, encoder_weights=None, in_channels=2, classes=2).float()
    model.load_state_dict(state, strict=True)
    model = model.eval()
    shifts = block_shifts(8,8)
    for s in range(4):
        stage = getattr(model.encoder.model,f"layers_{s}")
        if len(stage.blocks) != DEPTHS[s]:
            raise UnsupportedOperation("installed model does not implement the fixed Swin-T depths")
        for block, shift in zip(stage.blocks,shifts[s]):
            if block.shift_size != (shift,shift) or block.window_size != (7,7):
                raise UnsupportedOperation("installed model window/shift differs from fixed Swin-T224")
            if (not isinstance(block.mlp.act,torch.nn.GELU) or block.mlp.act.approximate != 'none'
                or block.norm1.eps != 1e-5 or block.norm2.eps != 1e-5):
                raise UnsupportedOperation("installed model GELU/LayerNorm differs from fixed Swin-T224")
            if attention == "explicit":
                block.attn.fused_attn = False
    return model


def load_swin_t224_model(checkpoint: Path, *, expected_checkpoint_sha256: str,
                        attention: Literal["default", "explicit"] = "default") -> torch.nn.Module:
    if attention not in ("default", "explicit"):
        raise UnsupportedOperation("attention must be 'default' or 'explicit'")
    return build_model(read_state(checkpoint,expected_checkpoint_sha256),attention)
