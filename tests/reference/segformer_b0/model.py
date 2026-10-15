"""Independent small NumPy primitive/composite mirrors; no whole-model forward.

All inputs are HWC float32. Linear/convolution/LN/GELU use double work and one
output rounding; attention uses the independent round_each_op mirror. Bilinear
coefficients are read from PyTorch one-hot interpolation, not tensor_ee. These
mirrors intentionally expose no encoder or complete-model evaluation function.
"""

from __future__ import annotations

import math

import numpy as np

from .blocks import attention, depthwise_conv3x3


def linear(x, weight, bias):
    out = x.astype(np.float64) @ weight.astype(np.float64).T
    return (out if bias is None else out + bias.astype(np.float64)).astype(np.float32)


def layer_norm(x, gamma, beta, eps):
    values = x.astype(np.float64)
    mean = values.mean(-1, keepdims=True)
    centered = values - mean
    variance = (centered * centered).mean(-1, keepdims=True)
    return (centered / np.sqrt(variance + eps) * gamma.astype(np.float64) + beta.astype(np.float64)).astype(np.float32)


def convolution(x, weight, bias, *, padding, stride):
    """Full-channel OIHW cross-correlation, independently indexed patches."""
    cout, cin, kernel, _ = weight.shape
    height = (x.shape[0] + 2 * padding - kernel) // stride + 1
    width = (x.shape[1] + 2 * padding - kernel) // stride + 1
    padded = np.pad(x.astype(np.float64), ((padding, padding), (padding, padding), (0, 0)))
    result = np.zeros((height, width, cout), np.float64)
    for c in range(cin):
        for ky in range(kernel):
            for kx in range(kernel):
                tap = padded[ky:ky + height * stride:stride, kx:kx + width * stride:stride, c]
                result += tap[:, :, None] * weight[:, c, ky, kx].astype(np.float64)
    return (result + bias.astype(np.float64)).astype(np.float32)


def overlap(x, parameters, prefix, *, kernel, stride):
    projection = convolution(x, parameters[f"{prefix}.proj.weight"], parameters[f"{prefix}.proj.bias"],
                             padding=kernel // 2, stride=stride)
    normalized = layer_norm(projection, parameters[f"{prefix}.norm.weight"],
                             parameters[f"{prefix}.norm.bias"], 1e-5)
    return {"projection": projection, "out": normalized}


def reduction(x, parameters, prefix, *, ratio):
    projection = convolution(x, parameters[f"{prefix}.sr.weight"], parameters[f"{prefix}.sr.bias"],
                             padding=0, stride=ratio)
    normalized = layer_norm(projection, parameters[f"{prefix}.norm.weight"],
                             parameters[f"{prefix}.norm.bias"], 1e-5)
    return {"projection": projection, "out": normalized}


def shared_keys(query, kv, heads):
    dim, channels = query.shape[-1] // heads, query.shape[-1]
    q, keys = query.reshape(-1, channels), kv.reshape(49, 2 * channels)
    result, outputs = {}, []
    for h in range(heads):
        out = attention(q[:, h * dim:(h + 1) * dim], keys[:, h * dim:(h + 1) * dim],
                        keys[:, channels + h * dim:channels + (h + 1) * dim], score_scale=dim ** -0.5)
        result[f"scaled_h{h}"] = out["scaled"].reshape(*query.shape[:2], 49)
        result[f"attention_h{h}"] = out["attention"].reshape(*query.shape[:2], 49)
        outputs.append(out["output"])
    result["out"] = np.concatenate(outputs, -1).reshape(query.shape)
    return result


def mix_ffn(x, parameters, prefix):
    first = linear(x, parameters[f"{prefix}.fc1.weight"], parameters[f"{prefix}.fc1.bias"])
    depthwise = depthwise_conv3x3(first, parameters[f"{prefix}.dwconv.dwconv.weight"],
                                parameters[f"{prefix}.dwconv.dwconv.bias"])
    values = depthwise.astype(np.float64)
    activated = (values * 0.5 * (1 + np.vectorize(math.erf)(values / math.sqrt(2)))).astype(np.float32)
    output = linear(activated, parameters[f"{prefix}.fc2.weight"], parameters[f"{prefix}.fc2.bias"])
    return {"fc1": first, "depthwise": depthwise, "gelu": activated, "out": output}


def resize(x, factor, *, align_corners):
    """Separable double resize using independent PyTorch interpolation coefficients."""
    import torch
    import torch.nn.functional as F

    matrices = []
    for n in x.shape[:2]:
        eye = torch.eye(n, dtype=torch.float32 if align_corners else torch.float64)
        with torch.no_grad():
            matrix = F.interpolate(eye[:, None, None], size=(1, factor * n), mode="bilinear",
                                   align_corners=align_corners)[:, 0, 0].T.double().numpy()
        matrices.append(matrix)
    return np.einsum("yi,ijc,xj->yxc", matrices[0], x.astype(np.float64), matrices[1]).astype(np.float32)


def pyramid(features, parameters):
    projected, parts = {}, []
    for i, feature in enumerate(reversed(features)):
        prefix = f"decoder.mlp_stage.{i}.linear"
        value = linear(feature, parameters[f"{prefix}.weight"], parameters[f"{prefix}.bias"])
        factor = 2 ** (3 - i)
        resized = value if factor == 1 else resize(value, factor, align_corners=False)
        projected[f"C{4 - i}_to_C1"] = resized
        parts.append(resized)
    projected["out"] = np.concatenate(parts, -1)
    return projected
