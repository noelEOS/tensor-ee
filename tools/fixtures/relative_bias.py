"""Opt-in structured relative-position gather, including full Swin blocks.

The bias uses the existing timm reference and NumPy mirror. Full-block fixtures
check both the original and optimized constants, with and without shifted masks,
against the same PyTorch model and expose their difference separately.
"""
from functools import partial

import numpy as np

from .core import Fixture, FixtureData, MirrorSpec, dyadic
from . import window as lane


def _bias_build(ee, data, *, window):
    from tensor_ee.blocks.window.attention import relative_position_bias
    from .core import Readout

    n = window * window
    biases = relative_position_bias(ee, data.weights["table"], window, gather="slices")
    readouts = []
    for h, bias in enumerate(biases):
        image, grid = lane._value_grid(ee, bias, window, n)
        readouts.append(Readout(f"bias_h{h}", image, n, grid))
    return readouts


def _block_data(seed):
    dim, heads, hidden = 16, 2, 64
    shapes = {
        "norm1.weight": (dim,), "norm1.bias": (dim,),
        "norm2.weight": (dim,), "norm2.bias": (dim,),
        "attn.qkv.weight": (3 * dim, dim), "attn.qkv.bias": (3 * dim,),
        "attn.proj.weight": (dim, dim), "attn.proj.bias": (dim,),
        "attn.relative_position_bias_table": (169, heads),
        "mlp.fc1.weight": (hidden, dim), "mlp.fc1.bias": (hidden,),
        "mlp.fc2.weight": (dim, hidden), "mlp.fc2.bias": (dim,),
    }
    weights = {name: dyadic(seed, name, shape, scale=0.125) for name, shape in shapes.items()}
    for name in ("norm1.weight", "norm2.weight"):
        weights[name] = dyadic(seed, name, (dim,), scale=0.25, offset=1, bits=16)
    return FixtureData(inputs={"x": dyadic(seed, "x", (14, 14, dim))}, weights=weights)


def _block_build(ee, data, *, shifted, exact_only=False):
    from tensor_ee.core.ops import WindowOps
    from tensor_ee.blocks.window.swin import swin_block_constants, swin_block, shifted_swin_block
    from tensor_ee.layouts.window import as_row, to_windows
    from tensor_ee.core.tensor import band_names

    ops, tokens, dim = WindowOps(ee), lane._tokens(2, 2, 7), 16
    dense = lane._dense(ee, tokens, data.inputs["x"], periodic=True)
    x = to_windows(ee, as_row(ee, dense.select(band_names(dim)), dim), tokens)
    values = {}
    for name, gather in (("original", "one_hot"), ("optimized", "slices")):
        c = swin_block_constants(ops, data.weights, relative_bias_gather=gather)
        result = (shifted_swin_block(ops, x, c, tokens.coarse(7)) if shifted else swin_block(ops, x, c))
        values[name] = result["out"]
    values["difference"] = values["optimized"].subtract(values["original"])
    if exact_only:
        values = {"difference": values["difference"]}
    return [lane._read(name, ee, tokens, value, dim) for name, value in values.items()] + [lane._echo(dense, tokens, dim)]


def _expected(output):
    output = np.asarray(output, dtype=np.float32)
    return {"original": output, "optimized": output, "difference": np.zeros_like(output)}


def _block_reference(data, *, shifted, exact_only=False):
    import torch
    from timm.models.swin_transformer import SwinTransformerBlock

    with lane._seeded():
        block = SwinTransformerBlock(dim=16, input_resolution=(14, 14), num_heads=2,
                                     window_size=7, shift_size=3 if shifted else 0).eval()
    block.attn.fused_attn = False
    block.load_state_dict({name: torch.from_numpy(value) for name, value in data.weights.items()})
    with torch.no_grad():
        output = block(torch.from_numpy(data.inputs["x"])[None])[0].numpy()
    result = _expected(output)
    return {"difference": result["difference"]} if exact_only else result


def _block_mirror(data, *, shifted, exact_only=False):
    from reference.swin_block.ee_mirror import Mirror
    from reference.swin_pair.ee_mirror_shifted import ShiftedMirror
    from reference.swin_block.pytorch_swin_block import to_window_layout, from_window_layout

    x = to_window_layout(data.inputs["x"])
    mirror = ShiftedMirror(data.weights, 2, 2) if shifted else Mirror(data.weights)
    mirror.sequential_sums = True
    result = mirror.shifted_forward(x) if shifted else mirror.forward(x)
    expected = _expected(from_window_layout(result["out"], 14, 14))
    return {"difference": expected["difference"]} if exact_only else expected


def fixtures():
    from tensor_ee.blocks.window.attention import relative_position_bias, relative_bias_values
    from tensor_ee.blocks.window.swin import swin_block_constants, METADATA
    from tensor_ee.weights.constants import WindowLayout

    result = []
    for w in (7, 12, 14):
        result.append(Fixture(
            id=f"window/relative_position_bias/slices-w{w}", lane="window", block="relative_position_bias",
            settings={"window": w, "gather": "slices"}, policy="double_round_once",
            shapes={"table": [(2 * w - 1) ** 2, 2], "heads": 2, "output": [2, w*w, w*w]},
            seed=20261103, data=partial(lane._bias_data, window=w), build=partial(_bias_build, window=w),
            reference=partial(lane._bias_reference, window=w, grid=True),
            reference_source="timm WindowAttention._get_rel_pos_bias, fixed dyadic table",
            mirror=MirrorSpec(partial(lane._bias_mirror, window=w, grid=True), True,
                              "reference.swin_block.ee_mirror.Mirror.bias"),
            requests=2 if w == 14 else 1,
            constants=lambda data, w=w: {"relative_position_bias": relative_bias_values(2, window=w)},
            history="docs/swin.md", accepts=(relative_position_bias,), window=w,
            tolerance=0, notes="Every entry of both heads; optional gather, original status unchanged."))
    for shifted in (False, True):
        block = "shifted_swin_block" if shifted else "swin_block"
        result.append(Fixture(
            id=f"window/{block}/bias-slices", lane="window", block=block,
            parts=("relative_position_bias", "window_attention", "linear", "layer_norm", "gelu"),
            settings={"window": 7, "relative_bias_gather": "slices"}, policy="composite",
            shapes={"tokens": [14, 14], "dim": 16, "heads": 2, "window": 7, "shifted": shifted},
            seed=20261103, data=_block_data, build=partial(_block_build, shifted=shifted),
            reference=partial(_block_reference, shifted=shifted),
            reference_source="timm SwinTransformerBlock, fixed dyadic parameters, non-fused attention",
            mirror=MirrorSpec(partial(_block_mirror, shifted=shifted), True,
                              "reference.swin_block.Mirror / swin_pair.ShiftedMirror, sequential sums"),
            requests=1, constants=lambda data, block=block: {
                block: 2 * METADATA[block].embedded_constants(16, 2, window=7)},
            history="docs/swin.md", accepts=(swin_block_constants, WindowLayout), window=7,
            notes="Both full-block outputs and optimized-minus-original. The difference must be exactly zero."))
        from dataclasses import replace

        result.append(replace(
            result[-1], id=f"window/{block}/bias-slices-exact", tolerance=0,
            build=partial(_block_build, shifted=shifted, exact_only=True),
            reference=partial(_block_reference, shifted=shifted, exact_only=True),
            mirror=MirrorSpec(partial(_block_mirror, shifted=shifted, exact_only=True), True,
                              "exact difference of the original and optimized full-block outputs"),
            notes="Strict zero-tolerance equality of the two EE block outputs, independently of PyTorch rounding."))
    return result
