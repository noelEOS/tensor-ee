"""timm 1.x query scaling, with compact projected QKV and fixed selection weights.

One request per fixture: first/last channel of every head, every spatial query,
and the class result repeated onto the spatial grid. The class input is the
first spatial row. All keys participate; unsampled output channels are covered
by the full-width offline test. There are no pretrained weights or downloads.
"""
from __future__ import annotations

from functools import partial

import numpy as np

from .class_token import _expand
from .core import Fixture, FixtureData, MirrorSpec, Readout, dyadic, inline_raster

CASES = ((14, 12, 64), (3, 3, 6))


def _data(seed, *, window, head_dim):
    return FixtureData({"x": dyadic(seed, "qkv", (window, window, 3 * head_dim), scale=0.5)}, {})


def _columns(heads, head_dim):
    return [h * head_dim + c for h in range(heads) for c in (0, head_dim - 1)]


def _outputs(patch, cls, *, window, heads, head_dim):
    columns = _columns(heads, head_dim)
    return {"patch": patch[:, columns].reshape(window, window, -1),
            "class": np.tile(cls[:, columns], (window * window, 1)).reshape(window, window, -1)}


def _build(ee, data, *, window, heads, head_dim):
    from tensor_ee.blocks.window.attention import class_token_attention
    from tensor_ee.core.grid import TokenGrid
    from tensor_ee.core.ops import WindowOps
    from tensor_ee.layouts.window import from_dense, to_dense

    grid, ops = TokenGrid(window, window), WindowOps(ee)
    channels = 3 * head_dim
    # Small band chunks avoid deep inline expression recursion.
    dense = ee.Image.cat([inline_raster(
        ee, grid.projection(ee), data.inputs["x"][..., first:first + 64],
        tuple(f"c{i}" for i in range(first, min(first + 64, channels))))
        for first in range(0, channels, 64)])
    base = from_dense(ee, grid, dense, channels, window=window)
    patch_qkv = _expand(ops, base, heads, head_dim)
    cls_qkv = ops.slice_rows(patch_qkv, 0, 1)
    patch, cls = class_token_attention(ops, patch_qkv, cls_qkv, dim=heads * head_dim,
                                      heads=heads, window=window, scale_query=True)
    def sampled(value):
        return ops.cat_cols([ops.slice_cols(value, col, col + 1) for col in _columns(heads, head_dim)])

    patch = sampled(patch)
    cls = ops.repeat_rows(sampled(cls), window * window)
    count = 2 * heads
    return [Readout("patch", to_dense(ee, grid, patch, count, window=window), count, grid.pixel_grid()),
            Readout("class", to_dense(ee, grid, cls, count, window=window), count, grid.pixel_grid()),
            Readout("input", dense, channels, grid.pixel_grid(), "input", "x")]


def _reference_full(data, *, window, heads, head_dim):
    """Actual timm Attention.forward, fixed sparse QKV selection and identity proj.

    The input's first 3*d columns carry the compact QKV seed; the remaining
    columns are zero. A fixed 0/1 weight selects and rotates those columns to
    reproduce expand_qkv, so the two projections introduce no rounding error.
    Attention is instantiated directly (no pretrained model is loaded).
    """
    import torch
    from timm.models.vision_transformer import Attention

    torch.set_num_threads(2)
    dim = heads * head_dim
    model = Attention(dim, num_heads=heads, qkv_bias=False, proj_bias=False).eval()
    model.fused_attn = False
    base = data.inputs["x"].reshape(window * window, 3 * head_dim)
    rows = np.concatenate([base[:1], base])
    x = torch.zeros(1, len(rows), dim)
    x[0, :, :3 * head_dim] = torch.from_numpy(rows)
    with torch.no_grad():
        model.qkv.weight.zero_()
        for s in range(3):
            for h in range(heads):
                for c in range(head_dim):
                    model.qkv.weight[s * dim + h * head_dim + c,
                                     s * head_dim + (c + h) % head_dim] = 1
        model.proj.weight.copy_(torch.eye(dim))
        output = model(x).numpy()[0]
    return output[1:], output[:1]


def _reference(data, **args):
    return _outputs(*_reference_full(data, **args), **args)


def _mirror_full(data, *, window, heads, head_dim, scale_query=True):
    from reference.class_token import class_token_attention, expand_qkv

    base = data.inputs["x"].reshape(window * window, 3 * head_dim)
    qkv = expand_qkv(base, heads, head_dim)
    out = class_token_attention(qkv, qkv[:1], dim=heads * head_dim, heads=heads,
                                scale_query=scale_query)
    return out["patch"], out["cls"]


def _mirror(data, **args):
    return _outputs(*_mirror_full(data, **args), **args)


def fixtures():
    from tensor_ee.blocks.window.attention import class_token_attention, class_token_attention_values

    result = []
    for window, heads, head_dim in CASES:
        args = dict(window=window, heads=heads, head_dim=head_dim)
        result.append(Fixture(
            id=f"window/class_token_attention/query-scale-w{window}-h{heads}-d{head_dim}",
            lane="window", block="class_token_attention",
            parts=("naive_attention", "split_heads", "concat_rows"),
            settings={"window": window, "dim": heads * head_dim, "heads": heads, "scale_query": True},
            policy="composite", shapes={"keys": window**2 + 1, "heads": heads, "head_dim": head_dim},
            seed=9300 + head_dim, data=partial(_data, window=window, head_dim=head_dim),
            build=partial(_build, **args), reference=partial(_reference, **args),
            reference_source="timm 1.0.30 vision_transformer.Attention (fused_attn=False; fixed 0/1 QKV and identity proj)",
            mirror=MirrorSpec(partial(_mirror, **args), False,
                              "reference.class_token.class_token_attention", bound=1e-4),
            requests=1, constants=lambda data: {"class_token_attention": class_token_attention_values()},
            history="New setting; offline evidence only, not validated live.",
            accepts=(class_token_attention,), window=window,
            notes="First/last channel per head, all spatial queries; class output repeated on spatial grid. Class input is first spatial row; full output checked offline.",
        ))
    return result


SELECTIONS = {"class_token_query_scale": lambda: tuple(f.id for f in fixtures())}
