"""SegFormer B0 primitive fixtures, offline-only evidence and a bounded live plan.

Depthwise mirrors use double sums but make no exact EE reduction-order claim.
Attention mirrors likewise use a1e-4 bound; exp/reduction order is not certified.
D1024 output and input echoes are each split into128-channel readouts for NPY
headers and request-band limits. No whole-model forward is performed here.
"""
from __future__ import annotations

from functools import partial

import numpy as np

from .core import Fixture, FixtureData, MirrorSpec, Readout, dyadic, hwc, nchw, value_rows
from .window import _dense, _read, _tokens, _windows

HISTORY = "docs/tasks/codex-segformer_b0.md"


def _dw_data(seed, *, channels, bias, edge=False):
    side = 14 if channels == 4 else 7
    x = dyadic(seed, "x", (side, side, channels))
    if edge:
        x[:] = 0
        x[0, 0] = np.arange(1, channels + 1, dtype=np.float32)
        x[6, 6] = np.arange(1, channels + 1, dtype=np.float32) * 2
        x[-1, -1] = np.arange(1, channels + 1, dtype=np.float32) * 3
    weights = {"weight": dyadic(seed, "weight", (channels, 1, 3, 3), scale=0.25)}
    if bias:
        weights["bias"] = dyadic(seed, "bias", (channels,), scale=0.125)
    return FixtureData({"x": x}, weights)


def _dw_build(ee, data):
    from tensor_ee.blocks.window.conv import DepthwiseConv3x3Constants, depthwise_conv3x3_window
    from tensor_ee.core.ops import WindowOps
    from tensor_ee.weights.constants import WindowLayout

    channels = data.inputs["x"].shape[-1]
    side = data.inputs["x"].shape[0]
    grid = _tokens(side // 7, side // 7)
    # Bound Image.cat depth: the standard raster helper creates one addBands per
    # channel, so build128-channel source chunks before joining the eight images.
    dense_parts = [_dense(ee, grid, data.inputs["x"][:, :, start:start + 128]).rename(
        [f"c{i}" for i in range(start, min(start + 128, channels))])
        for start in range(0, channels, 128)]
    dense = ee.Image.cat(dense_parts)
    ops = WindowOps(ee)
    c = DepthwiseConv3x3Constants(ops, data.weights["weight"], data.weights.get("bias"), WindowLayout(7))
    out = depthwise_conv3x3_window(ops, _windows(ee, grid, dense, channels), c, grid.coarse(7))
    result = []
    for start in range(0, channels, 128):
        end = min(start + 128, channels)
        name = "out" if channels <= 128 else f"out_{start}"
        result.append(_read(name, ee, grid, ops.slice_cols(out, start, end), end - start))
    # Each input split has its own source name in FixtureData to preserve exact echoes.
    for start in range(0, channels, 128):
        end = min(start + 128, channels)
        names = [f"c{i}" for i in range(start, end)]
        source = "x" if channels <= 128 else f"x_{start}"
        result.append(Readout("input" if channels <= 128 else f"input_{start}", dense.select(names),
                              end - start, grid.pixel_grid(), "input", source))
    return result


def _split_data(fn, seed):
    data = fn(seed)
    channels = data.inputs["x"].shape[-1]
    if channels <= 128:
        return data
    # Full x remains the graph source; split echoes need source arrays but avoid counting
    # duplicate input values: build reconstructs x from the split inputs below.
    return FixtureData({f"x_{s}": data.inputs["x"][:, :, s:s + 128] for s in range(0, channels, 128)}, data.weights)


def _join_dw_data(data):
    if "x" in data.inputs:
        return data
    x = np.concatenate([data.inputs[f"x_{s}"] for s in range(0, 1024, 128)], axis=2)
    return FixtureData({"x": x}, data.weights)


def _dw_build_split(ee, data):
    return _dw_build(ee, _join_dw_data(data))


def _dw_outputs(out):
    if out.shape[-1] <= 128:
        return {"out": out}
    return {f"out_{s}": out[:, :, s:s + 128] for s in range(0, out.shape[-1], 128)}


def _dw_reference(data):
    import torch
    import torch.nn.functional as F

    data = _join_dw_data(data)
    channels = data.inputs["x"].shape[-1]
    with torch.no_grad():
        out = F.conv2d(nchw(data.inputs["x"]), torch.from_numpy(data.weights["weight"]),
                       None if "bias" not in data.weights else torch.from_numpy(data.weights["bias"]),
                       padding=1, groups=channels)
    return _dw_outputs(hwc(out))


def _dw_mirror(data):
    from reference.segformer_b0.blocks import depthwise_conv3x3

    data = _join_dw_data(data)
    return _dw_outputs(depthwise_conv3x3(data.inputs["x"], data.weights["weight"], data.weights.get("bias")))


def _attention_data(seed, *, dim):
    return FixtureData({"q": dyadic(seed, "q", (14, 1, dim)),
                        "k": dyadic(seed, "k", (7, 1, dim)),
                        "v": dyadic(seed, "v", (7, 1, dim))}, {})


def _attention_build(ee, data, *, score_scale, window=7):
    if window != 7:
        raise ValueError("this rectangular V fixture uses window7 for its budget")
    from tensor_ee.blocks.window.attention import naive_attention
    from tensor_ee.core.ops import ValueOps
    from tensor_ee.core.precision import constant_value

    ops = ValueOps(ee)
    q, k, v = [constant_value(ee, data.inputs[n][:, 0]) for n in ("q", "k", "v")]
    dim = data.inputs["q"].shape[-1]
    out = naive_attention(ops, q, k, v, tokens=7, dim=dim, score_scale=score_scale)
    reads = []
    for name in ("scaled", "attention", "output"):
        cols = dim if name == "output" else 7
        image, grid = value_rows(ee, out[name], 14, cols)
        reads.append(Readout(name, image, cols, grid))
    for name, value in (("q", q), ("k", k), ("v", v)):
        rows = data.inputs[name].shape[0]
        image, grid = value_rows(ee, value, rows, dim)
        reads.append(Readout(f"input_{name}", image, dim, grid, "input", name))
    return reads


def _attention_reference(data, *, score_scale):
    import math
    import torch

    q, k, v = [torch.from_numpy(data.inputs[n][:, 0]) for n in ("q", "k", "v")]
    with torch.no_grad():
        scores = q @ k.T
        scaled = scores / math.sqrt(q.shape[1]) if score_scale is None else scores * score_scale
        probs = scaled.softmax(-1)
        return {"scaled": scaled.numpy()[:, None], "attention": probs.numpy()[:, None],
                "output": (probs @ v).numpy()[:, None]}


def _attention_mirror(data, *, score_scale):
    from reference.segformer_b0.blocks import attention

    out = attention(*[data.inputs[n][:, 0] for n in ("q", "k", "v")], score_scale=score_scale)
    return {name: out[name][:, None] for name in ("scaled", "attention", "output")}


def _dw_cost(channels, bias, data):
    from tensor_ee.blocks.window.conv import METADATA
    return {"depthwise_conv3x3_window": METADATA["depthwise_conv3x3_window"].embedded_constants(channels, bias=bias)}


def fixtures():
    from tensor_ee.blocks.window.attention import naive_attention
    from tensor_ee.blocks.window.conv import DepthwiseConv3x3Constants, depthwise_conv3x3_window

    result = []
    for channels, bias, edge in ((4, True, False), (4, False, False), (4, True, True), (1024, True, False)):
        label = f"w7-d{channels}-{'bias' if bias else 'nobias'}" + ("-edge" if edge else "")
        base = partial(_dw_data, channels=channels, bias=bias, edge=edge)
        result.append(Fixture(
            id=f"window/depthwise_conv3x3_window/{label}", lane="window", block="depthwise_conv3x3_window",
            settings={"window": 7, "k": 3, "stride": 1, "padding": 1, "dilation": 1, "multiplier": 1},
            policy="double_round_once", shapes={"channels": channels, "input": [14 if channels == 4 else 7]*2+[channels]},
            seed=20261014, data=partial(_split_data, base), build=_dw_build_split, reference=_dw_reference,
            reference_source="torch.nn.functional.conv2d(groups=D,padding=1)",
            mirror=MirrorSpec(_dw_mirror, False, "reference.segformer_b0.blocks.depthwise_conv3x3", bound=1e-4),
            requests=8 if channels == 1024 else 1, constants=partial(_dw_cost, channels, bias),
            history=HISTORY, accepts=(DepthwiseConv3x3Constants, depthwise_conv3x3_window),
            notes="Double reduction order not certified; bounded mirror. Split128-channel readouts at D1024."))
    for dim, scaled in ((32, True), (7, True), (32, False)):
        factor = dim ** -0.5 if scaled else None
        result.append(Fixture(
            id=f"window/naive_attention/segformer-d{dim}-{'score' if scaled else 'default'}",
            lane="window", block="naive_attention", settings={"window": 7, "tokens": 7, "dim": dim, "score_scale": factor},
            policy="round_each_op", shapes={"query": [14, dim], "key": [7, dim]}, seed=20261014,
            data=partial(_attention_data, dim=dim), build=partial(_attention_build, score_scale=factor),
            reference=partial(_attention_reference, score_scale=factor),
            reference_source="torch matmul, post-product scalar multiply, softmax, matmul",
            mirror=MirrorSpec(partial(_attention_mirror, score_scale=factor), False,
                              "reference.segformer_b0.blocks.attention", bound=1e-4),
            requests=2, constants=lambda data: {"naive_attention": 0}, history=HISTORY,
            accepts=(naive_attention, _attention_build),
            notes="Rectangular Q14/K7; dim7 distinguishes multiply by reciprocal from divide by root."))
    return result


SELECTIONS = {"segformer_b0": lambda: tuple(f.id for f in fixtures())}


# Small model compositions, appended to the primitive selection. No complete
# model reference/mirror is evaluated. The actual C32/C64/C160 SRA widths use
# sparse integer channel/tap selectors to keep each request below 2 MB.
def _feature(ee, data, source, grid):
    from tensor_ee.core.tensor import WindowTensor

    values = data.inputs[source]
    dense = ee.Image.cat([_dense(ee, grid, values[:, :, first:first + 32]).rename(
        [f"c{i}" for i in range(first, min(first + 32, values.shape[-1]))])
        for first in range(0, values.shape[-1], 32)])
    return WindowTensor(_windows(ee, grid, dense, values.shape[-1]), grid, values.shape[-1], 7), dense


def _echo_source(dense, grid, channels, source="x"):
    return Readout(f"input_{source}", dense, channels, grid.pixel_grid(), "input", source)


def _overlap_data(seed, *, kernel, stride):
    side = 28 if stride == 4 else 14
    return FixtureData({"x": dyadic(seed, "x", (side, side, 2), bits=12)},
                       {"embedding.proj.weight": dyadic(seed, "weight", (4, 2, kernel, kernel), scale=0.03125),
                        "embedding.proj.bias": dyadic(seed, "bias", (4,), scale=0.125),
                        "embedding.norm.weight": dyadic(seed, "gamma", (4,), offset=1, scale=0.25, bits=20),
                        "embedding.norm.bias": dyadic(seed, "beta", (4,), scale=0.125)})


def _overlap_build(ee, data, *, kernel, stride, window=7):
    from tensor_ee.core.grid import TokenGrid
    from tensor_ee.core.ops import WindowOps
    from tensor_ee.models.segformer_b0.encoder import overlap_constants, overlap_embedding

    if window != 7:
        raise ValueError("composition fixtures require W7")
    grid = TokenGrid(*data.inputs["x"].shape[:2], 2.5)
    x, dense = _feature(ee, data, "x", grid)
    ops = WindowOps(ee)
    c = overlap_constants(ops, data.weights, "embedding", 2, 4, kernel=kernel, stride=stride)
    out = overlap_embedding(ops, x, c)
    reads = [_read(name, ee, out[name].grid, out[name].image, 4) for name in ("projection", "out")]
    return reads + [_echo_source(dense, grid, 2)]


def _overlap_reference(data, *, kernel, stride):
    import torch
    import torch.nn.functional as F

    p = data.weights
    with torch.no_grad():
        projected = F.conv2d(nchw(data.inputs["x"]), torch.from_numpy(p["embedding.proj.weight"]),
                             torch.from_numpy(p["embedding.proj.bias"]), stride=stride, padding=kernel // 2)
        normalized = F.layer_norm(projected.permute(0, 2, 3, 1), (4,), torch.from_numpy(p["embedding.norm.weight"]),
                                  torch.from_numpy(p["embedding.norm.bias"]), eps=1e-5)
    return {"projection": hwc(projected), "out": normalized[0].numpy()}


def _overlap_mirror(data, *, kernel, stride):
    from reference.segformer_b0.model import overlap

    return overlap(data.inputs["x"], data.weights, "embedding", kernel=kernel, stride=stride)


def _overlap_cost(data, *, kernel):
    from tensor_ee.blocks.window.conv import METADATA as C
    from tensor_ee.blocks.window.linear import METADATA as L
    from tensor_ee.blocks.window.normalization import METADATA as N

    if kernel == 7:
        values = {"linear": L["linear"].embedded_constants(2 * kernel * kernel, 4)}
    else:
        values = {"conv3x3_window": C["conv3x3_window"].embedded_constants(2, 4, bias=True)}
    return values | {"layer_norm": N["layer_norm"].embedded_constants(4)}


def _reduction_data(seed, *, dim, ratio):
    side = 7 * ratio
    x = np.arange(side * side * dim, dtype=np.float32).reshape(side, side, dim)
    weight = np.zeros((dim, dim, ratio, ratio), np.float32)
    for out in range(dim):
        weight[out, (out * 3 + 1) % dim, out % ratio, (out // ratio) % ratio] = 1
    return FixtureData({"x": x}, {"attention.sr.weight": weight,
                                  "attention.sr.bias": np.arange(dim, dtype=np.float32) * 0.25,
                                  "attention.norm.weight": np.ones(dim, np.float32),
                                  "attention.norm.bias": np.zeros(dim, np.float32)})


def _reduction_constants(ops, data, *, dim, ratio):
    from tensor_ee.blocks.window.linear import LinearConstants
    from tensor_ee.models.segformer_b0.encoder import AttentionConstants, norm_constants

    p = data.weights
    sr = LinearConstants(ops, p["attention.sr.weight"].reshape(dim, dim * ratio * ratio), p["attention.sr.bias"], 49)
    return AttentionConstants(None, None, None, sr,
                              norm_constants(ops, p, "attention.norm", dim, eps=1e-5), dim, 1, ratio)


def _reduction_build(ee, data, *, dim, ratio, window=7):
    from tensor_ee.core.grid import TokenGrid
    from tensor_ee.core.ops import WindowOps
    from tensor_ee.models.segformer_b0.encoder import spatial_reduction

    if window != 7:
        raise ValueError("composition fixtures require W7")
    grid = TokenGrid(*data.inputs["x"].shape[:2])
    x, dense = _feature(ee, data, "x", grid)
    ops = WindowOps(ee)
    out = spatial_reduction(ops, x, _reduction_constants(ops, data, dim=dim, ratio=ratio))
    selected = ops.slice_cols(out["projection"].image, 0, 8)
    return [_read("projection", ee, out["projection"].grid, selected, 8),
            _read("out", ee, out["out"].grid, out["out"].image, dim), _echo_source(dense, grid, dim)]


def _reduction_reference(data, *, dim, ratio):
    import torch
    import torch.nn.functional as F

    p = data.weights
    with torch.no_grad():
        projected = F.conv2d(nchw(data.inputs["x"]), torch.from_numpy(p["attention.sr.weight"]),
                             torch.from_numpy(p["attention.sr.bias"]), stride=ratio)
        out = F.layer_norm(projected.permute(0, 2, 3, 1), (dim,), torch.from_numpy(p["attention.norm.weight"]),
                           torch.from_numpy(p["attention.norm.bias"]), eps=1e-5)
    return {"projection": hwc(projected)[..., :8], "out": out[0].numpy()}


def _reduction_mirror(data, *, dim, ratio):
    from reference.segformer_b0.model import reduction

    out = reduction(data.inputs["x"], data.weights, "attention", ratio=ratio)
    out["projection"] = out["projection"][..., :8]
    return out


def _reduction_cost(data, *, dim, ratio):
    from tensor_ee.blocks.window.linear import METADATA as L
    from tensor_ee.blocks.window.normalization import METADATA as N

    return {"linear": L["linear"].embedded_constants(dim * ratio * ratio, dim),
            "layer_norm": N["layer_norm"].embedded_constants(dim), "stride_columns": 0}


def _shared_data(seed):
    return FixtureData({"q": dyadic(seed, "q", (14, 14, 64), scale=0.25),
                        "kv": dyadic(seed, "kv", (7, 7, 128), scale=0.25)}, {})


def _shared_build(ee, data, *, heads=2, window=7):
    from tensor_ee.core.ops import WindowOps
    from tensor_ee.models.segformer_b0.encoder import shared_key_attention

    if window != 7:
        raise ValueError("composition fixtures require W7")
    q, dense_q = _feature(ee, data, "q", _tokens(2, 2))
    kv, dense_kv = _feature(ee, data, "kv", _tokens(1, 1))
    out = shared_key_attention(WindowOps(ee), q, kv, heads)
    reads = [_read("out", ee, q.grid, out["out"].image, 64)]
    for h in range(1):
        for name in ("scaled", "attention"):
            reads.append(_read(f"{name}_h{h}", ee, q.grid, out[f"h{h}_{name}"], 49))
    return reads + [_echo_source(dense_q, q.grid, 64, "q"), _echo_source(dense_kv, kv.grid, 128, "kv")]


def _shared_reference(data, *, heads=2):
    import torch

    query = data.inputs["q"]
    q = torch.from_numpy(query.reshape(-1, 64))
    kv = torch.from_numpy(data.inputs["kv"].reshape(49, 128))
    result, outputs = {}, []
    with torch.no_grad():
        for h in range(heads):
            score = (q[:, 32 * h:32 * (h + 1)] @ kv[:, 32 * h:32 * (h + 1)].T) * (32 ** -0.5)
            probability = score.softmax(-1)
            result[f"scaled_h{h}"] = score.numpy().reshape(14, 14, 49)
            result[f"attention_h{h}"] = probability.numpy().reshape(14, 14, 49)
            outputs.append((probability @ kv[:, 64 + 32 * h:64 + 32 * (h + 1)]).numpy())
    result["out"] = np.concatenate(outputs, -1).reshape(query.shape)
    return {name: value for name, value in result.items() if name == "out" or name.endswith("h0")}


def _shared_mirror(data, *, heads=2):
    from reference.segformer_b0.model import shared_keys

    result = shared_keys(data.inputs["q"], data.inputs["kv"], heads)
    return {name: value for name, value in result.items() if name == "out" or name.endswith("h0")}


def _norm_data(seed):
    return FixtureData({"x": dyadic(seed, "x", (14, 14, 4), scale=0.0009765625)},
                       {"norm.weight": dyadic(seed, "gamma", (4,), scale=0.25, offset=1, bits=20),
                        "norm.bias": dyadic(seed, "beta", (4,), scale=0.125)})


def _norm_build(ee, data, *, window=7):
    from tensor_ee.core.ops import WindowOps
    from tensor_ee.models.segformer_b0.encoder import channel_norm, norm_constants

    if window != 7:
        raise ValueError("composition fixtures require W7")
    grid = _tokens(2, 2)
    x, dense = _feature(ee, data, "x", grid)
    ops = WindowOps(ee)
    reads = []
    for name, eps in (("eps5", 1e-5), ("eps6", 1e-6)):
        out = channel_norm(ops, x, norm_constants(ops, data.weights, "norm", 4, eps=eps))
        reads.append(_read(name, ee, grid, out.image, 4))
    return reads + [_echo_source(dense, grid, 4)]


def _norm_reference(data):
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        return {name: F.layer_norm(torch.from_numpy(data.inputs["x"]), (4,),
                                   torch.from_numpy(data.weights["norm.weight"]),
                                   torch.from_numpy(data.weights["norm.bias"]), eps=eps).numpy()
                for name, eps in (("eps5", 1e-5), ("eps6", 1e-6))}


def _norm_mirror(data):
    from reference.segformer_b0.model import layer_norm

    return {name: layer_norm(data.inputs["x"], data.weights["norm.weight"], data.weights["norm.bias"], eps)
            for name, eps in (("eps5", 1e-5), ("eps6", 1e-6))}


def _norm_cost(data):
    from tensor_ee.blocks.window.normalization import METADATA

    return {"layer_norm": METADATA["layer_norm"].embedded_constants(4),
            "layer_norm2d": METADATA["layer_norm2d"].embedded_constants(4)}


def _ffn_data(seed):
    shapes = {"fc1.weight": (16, 4), "fc1.bias": (16,), "dwconv.dwconv.weight": (16, 1, 3, 3),
              "dwconv.dwconv.bias": (16,), "fc2.weight": (4, 16), "fc2.bias": (4,)}
    return FixtureData({"x": dyadic(seed, "x", (14, 14, 4))},
                       {f"mlp.{name}": dyadic(seed, name, shape, scale=0.125) for name, shape in shapes.items()})


def _ffn_build(ee, data, *, window=7):
    from tensor_ee.core.ops import WindowOps
    from tensor_ee.models.segformer_b0.encoder import mix_ffn, mix_ffn_constants

    if window != 7:
        raise ValueError("composition fixtures require W7")
    grid = _tokens(2, 2)
    x, dense = _feature(ee, data, "x", grid)
    ops = WindowOps(ee)
    out = mix_ffn(ops, x, mix_ffn_constants(ops, data.weights, "mlp", 4))
    return [_read(name, ee, grid, feature.image, feature.channels) for name, feature in out.items()] + [_echo_source(dense, grid, 4)]


def _ffn_reference(data):
    import torch
    import torch.nn.functional as F

    p = data.weights
    with torch.no_grad():
        first = F.linear(torch.from_numpy(data.inputs["x"]), torch.from_numpy(p["mlp.fc1.weight"]),
                          torch.from_numpy(p["mlp.fc1.bias"]))
        dw = F.conv2d(first.permute(2, 0, 1)[None], torch.from_numpy(p["mlp.dwconv.dwconv.weight"]),
                       torch.from_numpy(p["mlp.dwconv.dwconv.bias"]), padding=1, groups=16)
        activated = F.gelu(dw, approximate="none")
        out = F.linear(activated.permute(0, 2, 3, 1), torch.from_numpy(p["mlp.fc2.weight"]), torch.from_numpy(p["mlp.fc2.bias"]))
    return {"fc1": first.numpy(), "depthwise": hwc(dw), "gelu": hwc(activated), "out": out[0].numpy()}


def _ffn_mirror(data):
    from reference.segformer_b0.model import mix_ffn

    return mix_ffn(data.inputs["x"], data.weights, "mlp")


def _ffn_cost(data):
    from tensor_ee.blocks.window.conv import METADATA as C
    from tensor_ee.blocks.window.linear import METADATA as L

    return {"linear": L["linear"].embedded_constants(4, 16) + L["linear"].embedded_constants(16, 4),
            "depthwise_conv3x3_window": C["depthwise_conv3x3_window"].embedded_constants(16), "gelu": 0}


def _pyramid_data(seed):
    inputs = {}
    weights = {}
    for i, side in enumerate((56, 28, 14, 7), 1):
        inputs[f"C{i}"] = (np.arange(side * side * 2, dtype=np.float32).reshape(side, side, 2) + i * 8192)
    for i in range(4):
        prefix = f"decoder.mlp_stage.{i}.linear"
        weights[f"{prefix}.weight"] = np.eye(2, dtype=np.float32)
        weights[f"{prefix}.bias"] = np.array([i * 0.25, i * -0.25], np.float32)
    return FixtureData(inputs, weights)


def _pyramid_build(ee, data, *, window=7):
    from tensor_ee.core.ops import WindowOps
    from tensor_ee.models.segformer_b0.decoder import project_pyramid
    from tensor_ee.models.segformer_b0.encoder import _linear_constants

    if window != 7:
        raise ValueError("composition fixtures require W7")
    ops = WindowOps(ee)
    features, reads = [], []
    for i in range(1, 5):
        side = data.inputs[f"C{i}"].shape[0]
        grid = _tokens(side // 7, side // 7)
        # Physical spacings must agree after each resize, as in the real stages.
        grid = grid.resized(2 ** (i - 1), side, side)
        feature, dense = _feature(ee, data, f"C{i}", grid)
        features.append(feature)
        reads.append(_echo_source(dense, grid, 2, f"C{i}"))
    projections = [_linear_constants(ops, data.weights, f"decoder.mlp_stage.{i}.linear", 2, 2, 49) for i in range(4)]
    out = project_pyramid(ops, features, projections)
    return [_read(name, ee, feature.grid, feature.image, feature.channels)
            for name, feature in out.items() if "to_C1" in name or name == "out"] + reads


def _pyramid_reference(data):
    import torch
    import torch.nn.functional as F

    result, parts = {}, []
    with torch.no_grad():
        for i in range(4):
            name = f"C{4 - i}"
            p = f"decoder.mlp_stage.{i}.linear"
            projected = F.linear(torch.from_numpy(data.inputs[name]), torch.from_numpy(data.weights[f"{p}.weight"]),
                                torch.from_numpy(data.weights[f"{p}.bias"]))
            resized = F.interpolate(projected.permute(2, 0, 1)[None], size=(56, 56), mode="bilinear", align_corners=False)
            result[f"{name}_to_C1"] = hwc(resized)
            parts.append(hwc(resized))
    result["out"] = np.concatenate(parts, -1)
    return result


def _pyramid_mirror(data):
    from reference.segformer_b0.model import pyramid

    return pyramid([data.inputs[f"C{i}"] for i in range(1, 5)], data.weights)


def _pyramid_cost(data):
    from tensor_ee.blocks.window.linear import METADATA as L
    from tensor_ee.blocks.window.resample import METADATA as R

    return {"linear": 4 * L["linear"].embedded_constants(2, 2),
            "window_resize": sum(R["window_resize"].embedded_constants(side, side, factor, window=7)
                                 for side, factor in ((7, 8), (14, 4), (28, 2))), "concat": 0}


HEAD_CORNERS = ((0, 0), (0, 168), (168, 0), (168, 168))


def _head_data(seed):
    return FixtureData({"x": dyadic(seed, "x", (56, 56, 4), bits=12)},
                       {"weight": dyadic(seed, "head", (2, 4, 1, 1), scale=0.25),
                        "bias": dyadic(seed, "bias", (2,), scale=0.125)})


def _head_build(ee, data, *, factor=4, align_corners=True, window=7):
    from tensor_ee.blocks.window.linear import conv1x1_constants, conv1x1
    from tensor_ee.blocks.window.resample import window_resize
    from tensor_ee.core.grid import FeatureGrid
    from tensor_ee.core.ops import WindowOps
    from tensor_ee.layouts.window import to_dense

    if window != 7:
        raise ValueError("composition fixtures require W7")
    grid = _tokens(8, 8)
    x, dense = _feature(ee, data, "x", grid)
    ops = WindowOps(ee)
    c = conv1x1_constants(ops, data.weights["weight"], data.weights["bias"], 49)
    head = conv1x1(ops, x.image, c, layout=x)
    resized, fine = window_resize(ee, head, grid, 2, factor, align_corners=align_corners, window=window)
    fine_dense = to_dense(ee, fine, resized, 2, window=window)
    reads = [_read("head", ee, grid, head, 2), _echo_source(dense, grid, 4)]
    for i, (y, col) in enumerate(HEAD_CORNERS):
        block_grid = FeatureGrid(fine.crs, col * fine.scale, -y * fine.scale, fine.scale, 56, 56)
        reads.append(Readout(f"corner{i}", fine_dense, 2, block_grid.pixel_grid()))
    return reads


def _head_reference(data, *, factor=4, align_corners=True):
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        head = F.conv2d(nchw(data.inputs["x"]), torch.from_numpy(data.weights["weight"]), torch.from_numpy(data.weights["bias"]))
        out = hwc(F.interpolate(head, scale_factor=factor, mode="bilinear", align_corners=align_corners))
    return {"head": hwc(head)} | {f"corner{i}": out[y:y + 56, x:x + 56] for i, (y, x) in enumerate(HEAD_CORNERS)}


def _head_mirror(data, *, factor=4, align_corners=True):
    from reference.segformer_b0.model import linear, resize

    head = linear(data.inputs["x"], data.weights["weight"].reshape(2, 4), data.weights["bias"])
    out = resize(head, factor, align_corners=align_corners)
    return {"head": head} | {f"corner{i}": out[y:y + 56, x:x + 56] for i, (y, x) in enumerate(HEAD_CORNERS)}


def _head_cost(data):
    from tensor_ee.blocks.window.linear import METADATA as L
    from tensor_ee.blocks.window.resample import METADATA as R

    return {"conv1x1": L["conv1x1"].embedded_constants(4, 2),
            "window_resize": R["window_resize"].embedded_constants(56, 56, 4, align_corners=True, window=7)}


_primitive_fixtures = fixtures


def fixtures():
    from tensor_ee.models.segformer_b0.encoder import overlap_constants, attention_constants, shared_key_attention
    from tensor_ee.blocks.window.resample import window_resize

    result = _primitive_fixtures()

    def record(label, block, settings, shape, data, build, reference, mirror, requests, constants, parts, accepts, source):
        function = ("overlap" if "overlap" in label else "reduction" if "-sr-" in label else
                    "shared_keys" if "global-kv" in label else "layer_norm" if "norm-eps" in label else
                    "mix_ffn" if "mix-ffn" in label else "pyramid" if "deepest-first" in label else "resize")
        result.append(Fixture(id=f"window/{block}/{label}", lane="window", block=block,
                              settings={"window": 7} | settings, policy="round_each_op" if block == "naive_attention" else
                              "none" if block == "concat" else "double_round_once", shapes=shape, seed=20261015,
                              data=data, build=build, reference=reference, reference_source=source,
                              mirror=MirrorSpec(mirror, False, f"reference.segformer_b0.model.{function}", bound=1e-4),
                              requests=requests, constants=constants, history=HISTORY, parts=parts,
                              accepts=accepts, notes=("Small standalone composition, no complete-model forward or live claim. "
                                                      "Projection/resize components use double_round_once; concat only moves values."
                                                      if block == "concat" else
                                                      "Small standalone composition, no complete-model forward or live claim.")))

    for kernel, stride in ((7, 4), (3, 2)):
        args = {"kernel": kernel, "stride": stride}
        record(f"segformer-overlap-k{kernel}-s{stride}", "linear" if kernel == 7 else "conv3x3_window", args,
               {"channels": 2, "output": 4}, partial(_overlap_data, **args), partial(_overlap_build, **args),
               partial(_overlap_reference, **args), partial(_overlap_mirror, **args), 2,
               partial(_overlap_cost, kernel=kernel), ("layer_norm",),
               (overlap_constants, _overlap_build), "F.conv2d(k,stride,padding=k//2), F.layer_norm(eps=1e-5)")
    for dim, ratio in ((32, 8), (64, 4), (160, 2)):
        args = {"dim": dim, "ratio": ratio}
        record(f"segformer-sr-r{ratio}-c{dim}-ids", "linear", args, {"channels": dim, "source": 7 * ratio},
               partial(_reduction_data, **args), partial(_reduction_build, **args), partial(_reduction_reference, **args),
               partial(_reduction_mirror, **args), 2, partial(_reduction_cost, **args), ("stride_columns", "layer_norm"),
               (attention_constants, _reduction_build), "F.conv2d(unpadded,stride=r), F.layer_norm(eps=1e-5); unique channel/tap IDs")
    record("segformer-global-kv-4-query-windows", "naive_attention", {"heads": 2}, {"query": [14, 14, 64], "kv": [7, 7, 128]},
           _shared_data, _shared_build, _shared_reference, _shared_mirror, 2, lambda data: {"gather": 0, "naive_attention": 0},
           ("gather",), (shared_key_attention, _shared_build), "torch Q196 @ global K49, scale32**-0.5, softmax, V49")
    record("segformer-two-norm-eps", "layer_norm", {}, {"input": [14, 14, 4], "eps": [1e-5, 1e-6]},
           _norm_data, _norm_build, _norm_reference, _norm_mirror, 1, _norm_cost,
           ("layer_norm2d",), (_norm_build,), "F.layer_norm eps1e-5/1e-6 on low-variance input")
    record("segformer-mix-ffn", "linear", {}, {"input": [14, 14, 4], "hidden": 16},
           _ffn_data, _ffn_build, _ffn_reference, _ffn_mirror, 1, _ffn_cost,
           ("depthwise_conv3x3_window", "gelu"), (_ffn_build,), "F.linear -> F.conv2d(groups=16,padding=1) -> exact F.gelu -> F.linear")
    record("segformer-deepest-first", "concat", {}, {"levels": [56, 28, 14, 7], "channels": 2},
           _pyramid_data, _pyramid_build, _pyramid_reference, _pyramid_mirror, 4, _pyramid_cost,
           ("linear", "window_resize"), (_pyramid_build,), "F.linear, F.interpolate(56,align_corners=False), C4/C3/C2/C1 cat")
    args = {"factor": 4, "align_corners": True}
    record("segformer-head-56-to-224-corners", "window_resize", args, {"input": [56, 56, 4], "output": [224, 224, 2]},
           _head_data, partial(_head_build, **args), partial(_head_reference, **args), partial(_head_mirror, **args),
           5, _head_cost, ("conv1x1",), (window_resize, _head_build), "F.conv2d(1x1,bias), F.interpolate(scale4,align_corners=True)")
    return result
