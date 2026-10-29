"""Class-row concatenation and attention over one crop window.

The full-width attention fixture rotates a compact random input across heads;
all 197 keys and every class/spatial query participate. No asset is accessed.
"""
from __future__ import annotations

from functools import partial

import numpy as np

from .core import Fixture, FixtureData, MirrorSpec, Readout, dyadic, inline_raster


def _data(seed, *, window, head_dim):
    return FixtureData({"x": dyadic(seed, "patch", (window, window, 3 * head_dim), scale=0.5),
                        "c": dyadic(seed, "class", (1, 1, 3 * head_dim), scale=0.5)}, {})


def _sources(ee, data, window):
    from tensor_ee.core.grid import TokenGrid
    from tensor_ee.layouts.window import from_dense

    tokens = TokenGrid(window, window)
    coarse = tokens.coarse(window)
    columns = data.inputs["x"].shape[-1]
    names = [f"c{i}" for i in range(columns)]
    chunks = []
    for first in range(0, columns, 64):
        end = min(first + 64, columns)
        chunks.append(inline_raster(ee, tokens.projection(ee), data.inputs["x"][:, :, first:end],
                                    names[first:end]))
    dense = ee.Image.cat(chunks)
    cls_dense = inline_raster(ee, coarse.projection(ee), data.inputs["c"], names)
    patches = from_dense(ee, tokens, dense, columns, window=window)
    cls = cls_dense.toArray().arrayReshape(ee.Image(ee.Array([1, columns])), 2)
    return tokens, coarse, dense, cls_dense, patches, cls


def _readouts(ee, data, window, tokens, coarse, dense, cls_dense, patch_out, cls_out, channels):
    from tensor_ee.core.ops import WindowOps
    from tensor_ee.layouts.window import flatten_row, to_dense

    ops = WindowOps(ee)
    result = []
    for first in range(0, channels, 128):
        end = min(first + 128, channels)
        patch = to_dense(ee, tokens, ops.slice_cols(patch_out, first, end), end - first, window=window)
        result.append(Readout(f"p{first // 128}", patch, end - first, tokens.pixel_grid()))
    for first in range(0, channels, 128):
        end = min(first + 128, channels)
        cls = flatten_row(ops.slice_cols(cls_out, first, end), end - first)
        result.append(Readout(f"c{first // 128}", cls, end - first, coarse.pixel_grid()))
    columns = data.inputs["x"].shape[-1]
    result.extend([Readout("x", dense, columns, tokens.pixel_grid(), "input", "x"),
                   Readout("c", cls_dense, columns, coarse.pixel_grid(), "input", "c")])
    return result


def _expand(ops, base, heads, head_dim):
    pieces = []
    for s in range(3):
        part = ops.slice_cols(base, s * head_dim, (s + 1) * head_dim)
        for h in range(heads):
            offset = h % head_dim
            pieces.append(part if not offset else ops.cat_cols([
                ops.slice_cols(part, offset, head_dim), ops.slice_cols(part, 0, offset)]))
    return ops.cat_cols(pieces)


def _attention_build(ee, data, *, window, heads, head_dim):
    from tensor_ee.blocks.window.attention import class_token_attention
    from tensor_ee.core.ops import WindowOps

    tokens, coarse, dense, cls_dense, patch_base, cls_base = _sources(ee, data, window)
    ops = WindowOps(ee)
    dim = heads * head_dim
    spatial, cls = class_token_attention(
        ops, _expand(ops, patch_base, heads, head_dim), _expand(ops, cls_base, heads, head_dim),
        dim=dim, heads=heads, window=window,
    )
    return _readouts(ee, data, window, tokens, coarse, dense, cls_dense,
                     spatial, cls, dim)


def _outputs(patch, cls, window):
    channels = patch.shape[-1]
    return {**{f"p{first // 128}": patch[:, first:first + 128].reshape(window, window, -1)
               for first in range(0, channels, 128)},
            **{f"c{first // 128}": cls[:, first:first + 128].reshape(1, 1, -1)
               for first in range(0, channels, 128)}}


def _reference(data, *, window, heads, head_dim):
    import torch

    torch.set_num_threads(2)
    dim = heads * head_dim
    def expanded(name):
        base = torch.from_numpy(data.inputs[name].reshape(-1, 3 * head_dim))
        return torch.cat([torch.cat([torch.roll(base[:, s * head_dim:(s + 1) * head_dim],
                                                shifts=-h, dims=1) for h in range(heads)], 1)
                          for s in range(3)], 1)

    patch, cls = [expanded(name) for name in ("x", "c")]
    # Plain torch reference: all queries run together in their original row order.
    all_rows = torch.cat([cls, patch], 0).reshape(-1, 3, heads, head_dim).permute(1, 2, 0, 3)
    q, k, v = all_rows.unbind(0)
    with torch.no_grad():
        out = (((q @ k.transpose(-2, -1)) * (head_dim ** -0.5)).softmax(-1) @ v)
        out = out.permute(1, 0, 2).reshape(-1, dim).numpy()
    return _outputs(out[1:], out[:1], window)


def _mirror(data, *, window, heads, head_dim):
    from reference.class_token import class_token_attention, expand_qkv

    patch, cls = [expand_qkv(data.inputs[name].reshape(-1, 3 * head_dim), heads, head_dim)
                  for name in ("x", "c")]
    out = class_token_attention(patch, cls, dim=heads * head_dim, heads=heads)
    return _outputs(out["patch"], out["cls"], window)


def _concat_build(ee, data, *, window):
    from tensor_ee.blocks.window.merge import concat_rows
    from tensor_ee.core.ops import WindowOps

    tokens, coarse, dense, cls_dense, patch, cls = _sources(ee, data, window)
    ops = WindowOps(ee)
    channels = data.inputs["x"].shape[-1]
    joined = concat_rows(ops, [cls, patch], rows=(1, window ** 2), channels=channels)
    return _readouts(ee, data, window, tokens, coarse, dense, cls_dense,
                     ops.slice_rows(joined, 1, window ** 2 + 1), ops.slice_rows(joined, 0, 1), channels)


def _concat_reference(data, *, window):
    import torch

    channels = data.inputs["x"].shape[-1]
    joined = torch.cat([torch.from_numpy(data.inputs["c"].reshape(1, channels)),
                        torch.from_numpy(data.inputs["x"].reshape(-1, channels))], 0).numpy()
    return _outputs(joined[1:], joined[:1], window)


def _concat_mirror(data, *, window):
    from reference.class_token import concat_rows

    channels = data.inputs["x"].shape[-1]
    joined = concat_rows([data.inputs["c"].reshape(1, channels), data.inputs["x"].reshape(-1, channels)])
    return _outputs(joined[1:], joined[:1], window)


def fixtures():
    from tensor_ee.blocks.window.merge import concat_rows
    from tensor_ee.blocks.window.merge import METADATA
    from tensor_ee.blocks.window.attention import class_token_attention, METADATA as ATTENTION_METADATA
    from tensor_ee.weights.constants import WindowLayout

    result = [Fixture(
        id="window/concat_rows/class-spatial-w14", lane="window", block="concat_rows",
        settings={"window": 14, "rows": (1, 196), "channels": 3}, policy="none",
        shapes={"spatial": [196, 3], "class": [1, 3]}, seed=224,
        data=partial(_data, window=14, head_dim=1), build=partial(_concat_build, window=14),
        reference=partial(_concat_reference, window=14), reference_source="torch.cat(dim=0)",
        mirror=MirrorSpec(partial(_concat_mirror, window=14), True, "reference.class_token.concat_rows"),
        requests=2, constants=lambda data: {"concat_rows": METADATA["concat_rows"].embedded_constants(
            rows=(1, 196), channels=3)}, history="docs/evidence/live-fixtures/20261029T174042Z-class_token.json (passed; Earth Engine equal to the mirror)",
        accepts=(concat_rows, WindowLayout),
        tolerance=0, window=14)]
    for window, heads, head_dim, requests in ((3, 3, 4, 2), (14, 12, 64, 6)):
        args = dict(window=window, heads=heads, head_dim=head_dim)
        result.append(Fixture(
            id=f"window/class_token_attention/w{window}-h{heads}-d{head_dim}", lane="window",
            block="class_token_attention", parts=("naive_attention", "split_heads", "concat_rows"),
            settings={"window": window, "dim": heads * head_dim, "heads": heads}, policy="composite",
            shapes={"keys": window ** 2 + 1, "heads": heads, "head_dim": head_dim}, seed=225,
            data=partial(_data, window=window, head_dim=head_dim), build=partial(_attention_build, **args),
            reference=partial(_reference, **args), reference_source="plain torch post-QK scaling and softmax",
            mirror=MirrorSpec(partial(_mirror, **args), False, "reference.class_token.class_token_attention", bound=1e-4),
            requests=requests, constants=lambda data: {"class_token_attention":
                ATTENTION_METADATA["class_token_attention"].embedded_constants()},
            history="docs/evidence/live-fixtures/20261029T174042Z-class_token.json (passed; Earth Engine equal to the mirror)", accepts=(class_token_attention,), window=window,
            notes="Both class and spatial queries update; compact random inputs rotate across all heads."))
    return result


SELECTIONS = {"class_token": lambda: tuple(f.id for f in fixtures())}
