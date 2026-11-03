"""Final model-reference design: named SplitMix64 counters, native initializer scales.

This changes numerical references only, never _fixed_weights or gate fixtures.
Sources read before declaring this recipe:
* timm 1.0.30 models/vision_transformer.py: VisionTransformer.init_weights and
  init_weights_vit_timm; layers/patch_embed.py leaves Conv2d at torch defaults.
* SMP 0.5.0 base/model.py:initialize and base/initialization.py: decoder Conv2d
  Kaiming/ReLU, decoder Linear Xavier, head Conv2d Xavier; transpose convs untouched.
* torch 2.14 nn/modules/conv.py:_ConvNd.reset_parameters, normalization.py and
  batchnorm.py: constant norm affine; nn/init.py:_calculate_fan_in_and_fan_out.

key = uint64(crc32(name.encode('utf-8'))), zero-extended from 32 bits.
For flat index i, mix key + (i+1)*0x9E3779B97F4A7C15 with SplitMix64's
xor-shift/multiply finalizer, wrapping modulo 2**64 at every operation.
Take top 24 bits u; d=(u+0.5)/2**23-1 is an exact float32 dyadic in (-1,1).
Compute d*sqrt(3)*std_native in float64, then cast once to float32.
Constants stay constants. This matches native standard deviations, not native
Gaussian distribution shapes, sampled states, or trained model accuracy.
"""
from __future__ import annotations

import math
import zlib

import numpy as np


KNOWN_HASHES = (
    ("encoder.model.cls_token", 0, 0xCD87E5C0B58DE326),
    ("encoder.model.pos_embed", 104729, 0x13A2A11C9196E8E0),
    ("encoder.model.blocks.0.attn.qkv.weight", 17, 0x17FAA7E86949FDCB),
    ("decoder.reassemble_blocks.0.upsample.weight", 511, 0x370E7DB34C6240B7),
    ("segmentation_head.head.4.weight", 42, 0x7172BD62BC39F701),
)


def hash_bits(name, indices):
    """Counter-addressable SplitMix64; no RNG state, seed, or tensor-order dependence."""
    index = np.asarray(indices, dtype=np.uint64)
    key = np.uint64(zlib.crc32(name.encode("utf-8")))
    with np.errstate(over="ignore"):
        z = key + (index + np.uint64(1)) * np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def check_hashes():
    """Known answers derived independently with scalar Python masked integers."""
    for name, index, expected in KNOWN_HASHES:
        assert int(hash_bits(name, [index])[0]) == expected, (name, index)


def _native_rule(name, module, field):
    """The pinned source rule for each tensor, without reading its sampled values."""
    import torch

    row = {"name": name, "fan_in": None, "initializer_fan_in": None,
           "fan_out": None, "std": 0.0, "constant": None}
    if isinstance(module, (torch.nn.LayerNorm, torch.nn.BatchNorm2d)):
        assert field in ("weight", "bias"), name
        row.update(constant=1.0 if field == "weight" else 0.0,
                   native_initializer="ones_" if field == "weight" else "zeros_",
                   native_source="torch 2.14 norm reset_parameters; SMP 0.5.0 decoder norm affine")
        return row
    if field in ("cls_token", "pos_embed"):
        row.update(std=1e-6 if field == "cls_token" else .02,
                   native_initializer="normal_(std=1e-6)" if field == "cls_token" else "trunc_normal_(std=0.02,a=-2,b=2)",
                   native_source="timm 1.0.30 VisionTransformer.init_weights")
        return row
    assert field in ("weight", "bias"), name
    if isinstance(module, torch.nn.Linear):
        fan_in, fan_out = module.in_features, module.out_features
    elif isinstance(module, torch.nn.Conv2d):
        area = math.prod(module.kernel_size)
        fan_in, fan_out = module.in_channels // module.groups * area, module.out_channels * area
    elif isinstance(module, torch.nn.ConvTranspose2d):
        assert module.groups == 1 and module.kernel_size == module.stride and module.dilation == (1, 1), name
        # torch initializes stored [Cin,Cout,k,k] with Cout*k*k as fan_in.
        # At each non-overlapping output, the actual sum instead has Cin inputs.
        fan_in = module.out_channels * math.prod(module.kernel_size)
        fan_out = module.in_channels * math.prod(module.kernel_size)
        row["fan_in"] = module.in_channels
    else:
        raise ValueError(f"no declared native initializer for {name}")
    row.update(initializer_fan_in=fan_in, fan_out=fan_out)
    if row["fan_in"] is None:
        row["fan_in"] = fan_in
    if isinstance(module, torch.nn.ConvTranspose2d) or (
            name.startswith("encoder.") and isinstance(module, torch.nn.Conv2d)):
        row.update(std=math.sqrt(1.0 / (3 * fan_in)),
                   native_initializer="kaiming_uniform_(a=sqrt(5))" if field == "weight" else "uniform_(-1/sqrt(fan_in),1/sqrt(fan_in))",
                   native_source="torch 2.14 _ConvNd.reset_parameters; no timm/SMP override")
    elif field == "bias":
        row.update(constant=0.0, native_initializer="zeros_",
                   native_source="timm 1.0.30 init_weights_vit_timm / SMP 0.5.0 base.initialization")
    elif name.startswith("encoder.") and isinstance(module, torch.nn.Linear):
        row.update(std=.02, native_initializer="trunc_normal_(std=0.02,a=-2,b=2)",
                   native_source="timm 1.0.30 init_weights_vit_timm")
    elif name.startswith("decoder.") and isinstance(module, torch.nn.Conv2d):
        row.update(std=math.sqrt(2.0 / fan_in), native_initializer="kaiming_uniform_(mode=fan_in,nonlinearity=relu)",
                   native_source="SMP 0.5.0 base.initialization.initialize_decoder")
    elif ((name.startswith("decoder.") and isinstance(module, torch.nn.Linear)) or
          (name.startswith("segmentation_head.") and isinstance(module, torch.nn.Conv2d))):
        row.update(std=math.sqrt(2.0 / (fan_in + fan_out)), native_initializer="xavier_uniform_(gain=1)",
                   native_source="SMP 0.5.0 base.initialization.initialize_decoder/initialize_head")
    else:
        raise ValueError(f"no declared native initializer for {name}")
    return row


def tensor_values(name, shape, rule):
    if rule["constant"] is not None:
        return np.full(shape, rule["constant"], dtype=np.float32)
    u = (hash_bits(name, np.arange(math.prod(shape), dtype=np.uint64)) >> np.uint64(40)).astype(np.float64)
    dyadic = (u + .5) / 2**23 - 1
    assert np.array_equal(dyadic, dyadic.astype(np.float32).astype(np.float64))
    # No rescaling by sampled RMS: the source's declared std is fixed beforehand.
    return (dyadic * (math.sqrt(3.0) * rule["std"])).astype(np.float32).reshape(shape)


def initialize(model):
    """Build every tensor twice and require bit identity before any calibration."""
    import torch

    check_hashes()
    modules, report = dict(model.named_modules()), {}
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parent, field = name.rsplit(".", 1)
            assert parameter.dtype == torch.float32
            row = _native_rule(name, modules[parent], field)
            values = tensor_values(name, tuple(parameter.shape), row)
            second = tensor_values(name, tuple(parameter.shape), row)
            assert np.array_equal(values.view(np.uint32), second.view(np.uint32)), name
            assert np.isfinite(values).all(), name
            parameter.copy_(torch.from_numpy(values))
            assert np.array_equal(parameter.detach().numpy().view(np.uint32), second.view(np.uint32)), name
            row.update(key_crc32=zlib.crc32(name.encode("utf-8")), shape=list(parameter.shape),
                       second_build_bit_identical=True, exact_values_asserted=True)
            report[name] = row
    assert report.keys() == dict(model.named_parameters()).keys()
    return report
