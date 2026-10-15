"""Independent NumPy mirror of the approved fixed SegFormer-B0 arithmetic.

HWC float32 activations, exact OIHW state arrays, batch one, 224 x 224 input.
Existing independent primitives supply LN, projections, overlapping convolution,
SRA, global attention and MixFFN. Residuals round once; decoder BN and separable
bilinear interpolation are derived here, without native blocks or a graph
evaluator. There is no PyTorch/EE import or model execution in this module.

This mirror is a numerical hypothesis under the unchanged 1e-4 absolute target.
Implementing it is not evidence that a full forward or live model has passed.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .model import layer_norm, linear, mix_ffn, overlap, reduction, shared_keys

DIMS = (32, 64, 160, 256)
HEADS = (1, 2, 5, 8)
RATIOS = (8, 4, 2, 1)
SIDES = (56, 28, 14, 7)
COUNTER = "decoder.fuse_stage.1.num_batches_tracked"
SHAPES = {"input": (224, 224, 2), "S1B0": (56, 56, 32), "logits": (224, 224, 2)}
SHAPES.update({f"C{s}": (side, side, dim) for s, (side, dim) in enumerate(zip(SIDES, DIMS), 1)})
CUT_INPUTS = {"stage1a": ("input",), "stage1b": ("S1B0",),
              "stage2": ("C1",), "stage3": ("C2",), "stage4": ("C3",),
              "decoder": ("C1", "C2", "C3", "C4")}


def expected_state_shapes():
    """Independently declared SMP mit_b0 inference state schema."""
    shapes = {}

    def affine(prefix, dim):
        shapes[prefix + ".weight"] = (dim,)
        shapes[prefix + ".bias"] = (dim,)

    def projection(prefix, cin, cout, kernel=None, bias=True):
        shapes[prefix + ".weight"] = (cout, cin) if kernel is None else (cout, cin, kernel, kernel)
        if bias:
            shapes[prefix + ".bias"] = (cout,)

    for i, (dim, ratio) in enumerate(zip(DIMS, RATIOS)):
        stage = i + 1
        projection(f"encoder.patch_embed{stage}.proj", 2 if i == 0 else DIMS[i - 1], dim, 7 if i == 0 else 3)
        affine(f"encoder.patch_embed{stage}.norm", dim)
        for b in range(2):
            prefix = f"encoder.block{stage}.{b}"
            affine(prefix + ".norm1", dim)
            affine(prefix + ".norm2", dim)
            projection(prefix + ".attn.q", dim, dim)
            projection(prefix + ".attn.kv", dim, 2 * dim)
            projection(prefix + ".attn.proj", dim, dim)
            if ratio > 1:
                projection(prefix + ".attn.sr", dim, dim, ratio)
                affine(prefix + ".attn.norm", dim)
            projection(prefix + ".mlp.fc1", dim, 4 * dim)
            projection(prefix + ".mlp.dwconv.dwconv", 1, 4 * dim, 3)
            projection(prefix + ".mlp.fc2", 4 * dim, dim)
        affine(f"encoder.norm{stage}", dim)
    for i, dim in enumerate(reversed(DIMS)):
        projection(f"decoder.mlp_stage.{i}.linear", dim, 256)
    projection("decoder.fuse_stage.0", 1024, 256, 1, bias=False)
    affine("decoder.fuse_stage.1", 256)
    shapes["decoder.fuse_stage.1.running_mean"] = (256,)
    shapes["decoder.fuse_stage.1.running_var"] = (256,)
    shapes[COUNTER] = ()
    projection("segmentation_head.0", 256, 2, 1)
    return shapes


def validate_state(state):
    shapes = expected_state_shapes()
    if not isinstance(state, Mapping) or set(state) != set(shapes):
        raise ValueError("state must contain exactly the fixed SMP SegFormer-B0 inference names")
    for name, shape in shapes.items():
        value = state[name]
        if not isinstance(value, np.ndarray) or value.shape != shape:
            raise ValueError(f"{name}: expected NumPy shape {shape}")
        if name == COUNTER:
            if not np.issubdtype(value.dtype, np.integer) or value.item() < 0:
                raise ValueError(f"{name}: expected nonnegative integer scalar")
        elif value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f"{name}: expected finite float32 coefficients")
    if np.any(state["decoder.fuse_stage.1.running_var"] < 0):
        raise ValueError("BatchNorm running variance must be nonnegative")


def validate_hwc(value, name):
    if (not isinstance(value, np.ndarray) or value.dtype != np.float32 or value.shape != SHAPES[name]
            or not np.isfinite(value).all()):
        raise ValueError(f"{name}: expected finite float32 HWC shape {SHAPES[name]}")
    return value


def validate_cut_inputs(cut, inputs):
    if cut not in CUT_INPUTS:
        raise ValueError(f"unsupported cut {cut!r}")
    required = CUT_INPUTS[cut]
    if not isinstance(inputs, Mapping) or set(inputs) != set(required):
        raise ValueError(f"{cut}: expected exactly incoming arrays {required}")
    return {name: validate_hwc(inputs[name], name) for name in required}


def residual(a, b):
    return (a.astype(np.float64) + b.astype(np.float64)).astype(np.float32)


def transformer_block(x, state, prefix, *, heads, ratio):
    normalized = layer_norm(x, state[prefix + ".norm1.weight"], state[prefix + ".norm1.bias"], 1e-6)
    p = prefix + ".attn"
    query = linear(normalized, state[p + ".q.weight"], state[p + ".q.bias"])
    source = reduction(normalized, state, p, ratio=ratio)["out"] if ratio > 1 else normalized
    kv = linear(source, state[p + ".kv.weight"], state[p + ".kv.bias"])
    attention = shared_keys(query, kv, heads)["out"]
    first = residual(x, linear(attention, state[p + ".proj.weight"], state[p + ".proj.bias"]))
    normalized = layer_norm(first, state[prefix + ".norm2.weight"], state[prefix + ".norm2.bias"], 1e-6)
    return residual(first, mix_ffn(normalized, state, prefix + ".mlp")["out"])


def _stage(x, state, stage):
    i = stage - 1
    x = overlap(x, state, f"encoder.patch_embed{stage}", kernel=7 if stage == 1 else 3,
                stride=4 if stage == 1 else 2)["out"]
    first = None
    for b in range(2):
        x = transformer_block(x, state, f"encoder.block{stage}.{b}", heads=HEADS[i], ratio=RATIOS[i])
        if b == 0:
            first = x
    return layer_norm(x, state[f"encoder.norm{stage}.weight"], state[f"encoder.norm{stage}.bias"], 1e-6), first


def batch_norm(x, gamma, beta, mean, variance, *, eps=1e-5):
    """Explicit inference BN, double work and one final float32 rounding."""
    values = (x.astype(np.float64) - mean.astype(np.float64)) / np.sqrt(variance.astype(np.float64) + eps)
    return (values * gamma.astype(np.float64) + beta.astype(np.float64)).astype(np.float32)


def _axis_coefficients(n, output, align_corners):
    # align=True uses the source float32 model's coordinate arithmetic. The
    # half-pixel false-alignment coefficients at factors2/4/8 are exact dyadics.
    dtype = np.float32 if align_corners else np.float64
    scale = dtype((n - 1) / (output - 1) if align_corners and output > 1 else n / output)
    positions = np.arange(output, dtype=dtype)
    source = positions * scale if align_corners else (positions + dtype(0.5)) * scale - dtype(0.5)
    source = np.clip(source, dtype(0), dtype(n - 1))
    low = np.floor(source).astype(np.int64)
    high = np.minimum(low + 1, n - 1)
    fraction = source - low.astype(dtype)
    return low, high, (dtype(1) - fraction).astype(np.float64), fraction.astype(np.float64)


def bilinear_resize(x, output_shape, *, align_corners):
    """Independent coordinate rule; y product then x product, round once."""
    height, width = output_shape
    if height < 1 or width < 1:
        raise ValueError("resize output dimensions must be positive")
    ly, hy, wy0, wy1 = _axis_coefficients(x.shape[0], height, align_corners)
    lx, hx, wx0, wx1 = _axis_coefficients(x.shape[1], width, align_corners)
    source = x.astype(np.float64)
    vertical = source[ly] * wy0[:, None, None] + source[hy] * wy1[:, None, None]
    result = vertical[:, lx] * wx0[None, :, None] + vertical[:, hx] * wx1[None, :, None]
    return result.astype(np.float32)


def decoder_head(features, state):
    projected = []
    for i, name in enumerate(("C4", "C3", "C2", "C1")):
        prefix = f"decoder.mlp_stage.{i}.linear"
        value = linear(features[name], state[prefix + ".weight"], state[prefix + ".bias"])
        projected.append(value if i == 3 else bilinear_resize(value, (56, 56), align_corners=False))
    fused = linear(np.concatenate(projected, axis=-1), state["decoder.fuse_stage.0.weight"].reshape(256, 1024), None)
    p = "decoder.fuse_stage.1"
    normalized = batch_norm(fused, state[p + ".weight"], state[p + ".bias"],
                            state[p + ".running_mean"], state[p + ".running_var"])
    activated = np.maximum(normalized, np.float32(0))
    head = linear(activated, state["segmentation_head.0.weight"].reshape(2, 256), state["segmentation_head.0.bias"])
    return bilinear_resize(head, (224, 224), align_corners=True)


def mirror_full(state, input):
    """Root-run complete NumPy numerical mirror; never an EE graph evaluator."""
    validate_state(state)
    x = validate_hwc(input, "input")
    result = {}
    for stage in range(1, 5):
        x, first = _stage(x, state, stage)
        if stage == 1:
            result["S1B0"] = first
        result[f"C{stage}"] = x
    result["logits"] = decoder_head(result, state)
    return result


def mirror_cut(cut, state, inputs):
    """Numerical output for one selected cut on its actual incoming arrays."""
    validate_state(state)
    inputs = validate_cut_inputs(cut, inputs)
    if cut == "stage1a":
        x = overlap(inputs["input"], state, "encoder.patch_embed1", kernel=7, stride=4)["out"]
        return transformer_block(x, state, "encoder.block1.0", heads=1, ratio=8)
    if cut == "stage1b":
        x = transformer_block(inputs["S1B0"], state, "encoder.block1.1", heads=1, ratio=8)
        return layer_norm(x, state["encoder.norm1.weight"], state["encoder.norm1.bias"], 1e-6)
    if cut == "decoder":
        return decoder_head(inputs, state)
    stage = int(cut[-1])
    return _stage(inputs[f"C{stage - 1}"], state, stage)[0]
