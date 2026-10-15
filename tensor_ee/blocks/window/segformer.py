"""SegFormer's composites in window layout: the channel norm, the overlapping patch
embedding, efficient attention with global keys and values, the Mix-FFN, the MiT block, and
the all-MLP decoder's pyramid projection and fused head.

Moved (X01a) from ``tensor_ee/models/segformer_b0/encoder.py`` (``NormConstants``,
``norm_constants``, ``channel_norm``, ``OverlapConstants``, ``overlap_constants``,
``decimate``, ``overlap_embedding``, ``AttentionConstants``, ``attention_constants``,
``spatial_reduction``, ``shared_key_attention``, ``global_attention``, ``MixFFNConstants``,
``mix_ffn_constants``, ``mix_ffn``, and the MiT block ``BlockConstants``, ``block_constants``
and ``block``, renamed here ``MiTBlockConstants``, ``mit_block_constants`` and ``mit_block``,
with their helpers) and ``decoder.py`` (``project_pyramid``, and the fuse, BatchNorm, ReLU and
head sequence of ``decoder()``, now ``fuse_head``). The fixed SegFormer-B0 model keeps its
configuration and checks there and re-exports these objects under their former names; there
is one implementation. The decoder composites' constants builders, ``project_pyramid_constants``
and ``fuse_head_constants``, came from ``decoder_constants`` at X02a, so that the model and the
block-graph builder (``tensor_ee/converter/``) share them.

Parameters keep SMP 0.5.0 state_dict names under a ``prefix``: NumPy float32 arrays of the
original shapes, or ``ServerArray``s, a convolution's as [out, in k k] in PyTorch's (c, ky, kx)
flatten order. Every spatial feature is W7: one [49, C] float32 array per 7 x 7 window
(``WindowTensor``, window 7) of a ``TokenGrid`` divisible by seven. A token is a pixel of
that grid. In PyTorch's terms (``mit_b0``, ``SegformerDecoder``)::

    channel_norm          LayerNorm over channels, eps 1e-5 (layer_norm) or 1e-6 (layer_norm2d)
    overlap_embedding     LN(Conv2d(x; k, stride s, padding k // 2)), (k, s) = (7, 4) or (3, 2)
    spatial_reduction     LN(Conv2d(x; r, stride r)) for r > 1; x itself for r = 1
    shared_key_attention  per head h of width d: softmax(q_h k_h^T * d^-0.5) v_h, concatenated
    global_attention      proj(shared_key_attention(q(x), kv(spatial_reduction(x))))
    mix_ffn               fc2(GELU(DWConv3x3(fc1(x)))), hidden width 4 C
    mit_block             x1 = x + attn(LN1(x));  out = x1 + mlp(LN2(x1))
    project_pyramid       cat(up(Linear(C4)), up(Linear(C3)), up(Linear(C2)), Linear(C1))
    fuse_head             up4(Conv1x1_head(ReLU(BN(Conv1x1_fuse(x)))))

The overlap embedding convolves at stride 1 on the fine grid (k 7: a ring-3 halo, im2col and
``linear``; k 3: ``conv3x3_window``), then ``decimate`` samples phase (0, 0): pixel (i, j) of
the result is pixel (s i, s j) of the stride-1 map, exactly PyTorch's strided output. The
reduction's grid must be exactly one 7 x 7 window (a 7 r x 7 r chip): its 49 keys and values
are gathered once (``gather``) and shared by every query window, so the attention is global
without a per-window key or value. The pyramid's resizes are bilinear with
``align_corners=False`` (factors 8, 4, 2; C1 unchanged), the head's is four-times bilinear
with ``align_corners=True``.

**Policy** ``composite``: the norms, projections, convolutions, GELU, BatchNorm, residual
additions and resizes run in ``double_round_once``, the attention core in ``round_each_op``
with SMP's post-QK scale (``score_scale``), ReLU and concatenation without arithmetic.
``decimate`` has no arithmetic (``none``).

**Locality.** The global attention, its shared-key core and the MiT block read the whole chip
(``EXTENT``), and tiling with halos cannot reproduce them (chip-edge dependent); the
reduction makes the whole chip one window (``EXTENT``). The head's ``align_corners=True``
weights depend on the map's size (they are not shift-invariant), so ``fuse_head`` is
chip-edge dependent too, though it reads only a ring of one window.

**Rows.** Constants are built ahead for a layout (``LAYOUT``, ``WindowLayout(7)``: 49 rows),
pre-repeated to its rows because Earth Engine arrays do not broadcast, and the blocks refuse
constants built for another row count (client-side).

**Refused** (``reject``): a feature that is not W7 on a grid divisible by seven; norm epsilons
other than 1e-5 and 1e-6; overlap kernel and stride other than 7/4 and 3/2; decimation
factors other than 2 and 4; heads that do not divide the width; reduction ratios other than 1,
2, 4 and 8; a reduced map that is not exactly one 7 x 7 window; a pyramid that is not four
levels at C1, C1/2, C1/4 and C1/8; head constants of other widths than ``channels`` and
``classes``.

**Validated.** Live only inside the fixed SegFormer-B0 (``mit_b0``) at 224 x 224, two bands
and two classes: widths 32, 64, 160, 256, heads 1, 2, 5, 8, reduction ratios 8, 4, 2, 1,
window 7 and decoder width 256, by staged agreement on one seeded chip and state under the
1e-4 absolute target (docs/evidence/segformer_b0-fullmodel-live.json), and as small
compositions (docs/evidence/segformer_b0-live.json). Other settings the builders accept,
such as the small fixture widths, are not validated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...core.errors import reject
from ...core.grid import TokenGrid, to_feature_grid, to_token_grid
from ...core.metadata import COMPOSITE, NO_ARITHMETIC, BlockMetadata, Locality, Validated, validate_metadata
from ...core.ops import ValueOps
from ...core.precision import DOUBLE_ROUND_ONCE, ROUND_EACH_OP
from ...core.tensor import WindowTensor, band_names
from ...layouts.raster import stride_sample_phase
from ...layouts.window import from_dense, halo, im2col, to_dense, to_windows
from ...weights.constants import ServerArray, WindowLayout, row_constant
from .activations import gelu, relu
from .attention import naive_attention
from .conv import METADATA as CONV_METADATA
from .conv import Conv3x3Constants, DepthwiseConv3x3Constants, conv3x3_window, depthwise_conv3x3_window
from .linear import METADATA as LINEAR_METADATA
from .linear import LinearConstants, conv1x1, conv1x1_constants, linear, no_constants
from .merge import add, concat
from .normalization import METADATA as NORM_METADATA
from .normalization import BatchNormConstants, batch_norm, layer_norm, layer_norm2d
from .patch import stride_columns
from .pooling import gather
from .resample import METADATA as RESIZE_METADATA
from .resample import window_resize

LAYOUT = WindowLayout(7)
"""The layout every SegFormer composite runs in: window 7, 49 rows per constant."""


def _server_shape(value, name):
    if not isinstance(value.shape, (tuple, list)):
        reject(f"{name}: ServerArray shape must be a tuple or list of dimensions")
    if any(isinstance(d, (bool, np.bool_)) or not isinstance(d, (int, np.integer)) or d < 1
           for d in value.shape):
        reject(f"{name}: ServerArray dimensions must be positive non-bool integers")
    return tuple(int(d) for d in value.shape)


def _parameter(parameters, name, shape):
    value = parameters[name]
    if isinstance(value, ServerArray):
        expected = (shape[0], int(np.prod(shape[1:]))) if len(shape) == 4 else shape
        if _server_shape(value, name) != expected:
            reject(f"{name}: ServerArray shape {value.shape}, expected {expected}")
    elif (not isinstance(value, np.ndarray) or value.shape != shape or
          value.dtype != np.float32 or not np.isfinite(value).all()):
        reject(f"{name}: expected finite float32 NumPy shape {shape}")
    return value


def _linear_constants(ops, p, name, cin, cout, layout):
    return LinearConstants(ops, _parameter(p, f"{name}.weight", (cout, cin)),
                           _parameter(p, f"{name}.bias", (cout,)), layout)


def _conv_columns(ops, weight, cin, cout, kernel, *, tap_major):
    """OIHW -> Linear [out,columns], moving coefficients without arithmetic.

    halo/im2col has (ky,kx,c) columns; stride_columns has (c,ky,kx).
    Server convolution weights arrive flattened in the latter PyTorch order.
    """
    if isinstance(weight, ServerArray):
        if not tap_major:
            return weight
        arr = ValueOps(ops.ee)
        w = arr.reshape(weight.array, [cout, cin, kernel * kernel])
        w = arr.reshape(arr.transpose(w, 1, 2), [cout, kernel * kernel * cin])
        return ServerArray(w, (cout, kernel * kernel * cin))
    if tap_major:
        return weight.transpose(0, 2, 3, 1).reshape(cout, kernel * kernel * cin)
    return weight.reshape(cout, kernel * kernel * cin)


def _check_feature(x, *, channels=None, grid=None):
    if not isinstance(x, WindowTensor) or x.window != 7 or getattr(x, "layout", "window") != "window":
        reject("SegFormer compositions require W7 spatial features")
    if not isinstance(x.grid, TokenGrid) or x.grid.height % 7 or x.grid.width % 7:
        reject("features require a TokenGrid divisible by seven")
    if channels is not None and x.channels != channels:
        reject(f"feature has {x.channels} channels; expected {channels}")
    if grid is not None and x.grid != grid:
        reject(f"feature grid {x.grid} differs from expected {grid}")


# ---------------------------------------------------------------------------
# Channel norm
# ---------------------------------------------------------------------------
@dataclass
class NormConstants:
    gamma: object
    beta: object
    channels: int
    eps: float


def norm_constants(ops, parameters, prefix, channels, *, eps, layout=LAYOUT):
    """Affine channel constants pre-repeated to the spatial layout's rows."""
    if eps not in (1e-5, 1e-6):
        reject("SegFormer norms use epsilon 1e-5 or 1e-6")
    return NormConstants(row_constant(ops, _parameter(parameters, f"{prefix}.weight", (channels,)), layout),
                         row_constant(ops, _parameter(parameters, f"{prefix}.bias", (channels,)), layout),
                         channels, eps)


def channel_norm(ops, x, c: NormConstants):
    """LayerNorm over the channels of a W7 feature: ``layer_norm`` at eps 1e-5 (the
    embedding and reduction norms), ``layer_norm2d`` at eps 1e-6 (the block and stage norms)."""
    _check_feature(x, channels=c.channels)
    if c.eps == 1e-5:
        image = layer_norm(ops, x.image, c.gamma, c.beta, dim=c.channels,
                           layout=x, eps=1e-5, policy=DOUBLE_ROUND_ONCE)
    elif c.eps == 1e-6:
        image = layer_norm2d(ops, x.image, c.gamma, c.beta, dim=c.channels,
                             rows=x, eps=1e-6, policy=DOUBLE_ROUND_ONCE)
    else:
        reject("malformed SegFormer norm epsilon")
    return x.like(image)


# ---------------------------------------------------------------------------
# Overlapping patch embedding
# ---------------------------------------------------------------------------
@dataclass
class OverlapConstants:
    projection: object
    norm: NormConstants
    cin: int
    cout: int
    kernel: int
    stride: int


def overlap_constants(ops, parameters, prefix, cin, cout, *, kernel, stride, layout=LAYOUT):
    """Biased overlapping embedding: supported k7/s4/p3 and k3/s2/p1 only.

    This composition builder can use small fixture widths; the fixed model
    constructor separately enforces every full-model state shape.
    """
    if (any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in (kernel, stride))
            or (kernel, stride) not in ((7, 4), (3, 2))):
        reject("overlap embeddings support k7/s4/p3 or k3/s2/p1 only")
    weight = _parameter(parameters, f"{prefix}.proj.weight", (cout, cin, kernel, kernel))
    bias = _parameter(parameters, f"{prefix}.proj.bias", (cout,))
    projection = (LinearConstants(ops, _conv_columns(ops, weight, cin, cout, kernel, tap_major=True), bias, layout)
                  if kernel == 7 else Conv3x3Constants(ops, weight, bias, layout))
    return OverlapConstants(projection, norm_constants(ops, parameters, f"{prefix}.norm", cout,
                                                     eps=1e-5, layout=layout), cin, cout, kernel, stride)


def decimate(ee, x, factor):
    """Exact phase-(0,0) sample and explicit canonical relabel, with no arithmetic.

    FeatureGrid supplies x0/y0 to stride_sample_phase; TokenGrid is never passed
    directly to that sampler. Its pixel (i,j) is source (factor*i,factor*j).
    """
    _check_feature(x)
    if isinstance(factor, (bool, np.bool_)) or not isinstance(factor, (int, np.integer)) or factor not in (2, 4):
        reject("SegFormer decimation factors are two and four")
    destination = x.grid.coarse(factor)
    if destination.height % 7 or destination.width % 7:
        reject("decimated map must partition into complete W7 windows")
    fine = to_feature_grid(x.grid)
    coarse = fine.coarse(factor)
    canonical = to_token_grid(coarse)
    dense = to_dense(ee, x.grid, x.image, x.channels, window=7)
    sampled = stride_sample_phase(ee, dense, fine, factor, a=0, b=0)
    relabeled = sampled.changeProj(coarse.projection(ee), canonical.projection(ee))
    image = from_dense(ee, canonical, relabeled, x.channels, window=7)
    return WindowTensor(image, canonical, x.channels, 7)


def overlap_embedding(ops, x, c: OverlapConstants) -> dict:
    """The stride-1 convolution, ``decimate`` at the stride, then ``channel_norm`` (eps 1e-5)."""
    _check_feature(x, channels=c.cin)
    coarse = x.grid.coarse(7)
    if c.kernel == 7 and c.stride == 4:
        columns = im2col(ops.ee, halo(ops.ee, x.image, coarse, c.cin, window=7, ring=3),
                         c.cin, 7, window=7)
        dense = linear(ops, columns, c.projection, layout=x, policy=DOUBLE_ROUND_ONCE)
    elif c.kernel == 3 and c.stride == 2:
        columns = None
        dense = conv3x3_window(ops, x.image, c.projection, coarse, window=7, policy=DOUBLE_ROUND_ONCE)
    else:
        reject("malformed overlap constants")
    dense = x.like(dense, c.cout)
    projected = decimate(ops.ee, dense, c.stride)
    return {"columns": columns, "dense_projection": dense, "projection": projected,
            "out": channel_norm(ops, projected, c.norm)}


# ---------------------------------------------------------------------------
# Efficient attention: spatial reduction and global keys and values
# ---------------------------------------------------------------------------
@dataclass
class AttentionConstants:
    q: LinearConstants
    kv: LinearConstants
    projection: LinearConstants
    sr: object
    sr_norm: object
    dim: int
    heads: int
    ratio: int


def attention_constants(ops, parameters, prefix, dim, heads, ratio, *, layout=LAYOUT):
    if (isinstance(heads, (bool, np.bool_)) or not isinstance(heads, (int, np.integer)) or heads < 1 or dim % heads or
            isinstance(ratio, (bool, np.bool_)) or not isinstance(ratio, (int, np.integer)) or ratio not in (1, 2, 4, 8)):
        reject("attention needs positive dividing heads and SR ratio 1/2/4/8")
    sr, norm = None, None
    if ratio > 1:
        weight = _parameter(parameters, f"{prefix}.sr.weight", (dim, dim, ratio, ratio))
        sr = LinearConstants(ops, _conv_columns(ops, weight, dim, dim, ratio, tap_major=False),
                             _parameter(parameters, f"{prefix}.sr.bias", (dim,)), layout)
        norm = norm_constants(ops, parameters, f"{prefix}.norm", dim, eps=1e-5, layout=layout)
    return AttentionConstants(_linear_constants(ops, parameters, f"{prefix}.q", dim, dim, layout),
                              _linear_constants(ops, parameters, f"{prefix}.kv", dim, 2 * dim, layout),
                              _linear_constants(ops, parameters, f"{prefix}.proj", dim, dim, layout),
                              sr, norm, dim, heads, ratio)


def spatial_reduction(ops, x, c: AttentionConstants) -> dict:
    """Learned unpadded full-channel convolution of the normalized query input."""
    _check_feature(x, channels=c.dim)
    reduced_grid = x.grid.coarse(c.ratio)
    if (reduced_grid.height, reduced_grid.width) != (7, 7):
        reject("global reduced keys must cover exactly one 7 x 7 window")
    if c.ratio == 1:
        if c.sr is not None or c.sr_norm is not None:
            reject("ratio-one attention has no SR convolution or normalization")
        return {"columns": None, "projection": x, "out": x}
    dense = to_dense(ops.ee, x.grid, x.image, x.channels, window=7)
    # Keep Python's ordinary EE serialization recursion limit meaningful. Each
    # existing stride_columns call gathers a bounded channel group; concatenating
    # its value-only rows preserves (c,ky,kx) order and changes no arithmetic.
    names = band_names(x.channels)
    group = max(1, 128 // (c.ratio * c.ratio))
    parts = [stride_columns(ops.ee, dense, x.grid, patch=c.ratio, bands=names[first:first + group])
             for first in range(0, x.channels, group)]
    columns = ops.cat_cols(parts)
    windows = to_windows(ops.ee, columns, reduced_grid, window=7, mask=True)
    projection = linear(ops, windows, c.sr, layout=x, policy=DOUBLE_ROUND_ONCE)
    feature = WindowTensor(projection, reduced_grid, x.channels, 7)
    return {"columns": columns, "projection": feature, "out": channel_norm(ops, feature, c.sr_norm)}


def shared_key_attention(ops, q, kv, heads: int) -> dict:
    """Global V49 K/V shared dynamically by all W7 query windows.

    q is W7 C channels; kv is the single reduced W7 window with 2C channels.
    Each head computes [49,49] temporary scores, never a spatial tensor whose
    channels or rows misrepresent the layout. Result images are pinned to q.
    """
    _check_feature(q)
    _check_feature(kv, channels=2 * q.channels)
    if ((kv.grid.height, kv.grid.width) != (7, 7) or isinstance(heads, (bool, np.bool_)) or
            not isinstance(heads, (int, np.integer)) or heads < 1 or q.channels % heads):
        reject("shared keys need one W7 window and heads dividing the query width")
    arr = ValueOps(ops.ee)
    global_kv = arr.reshape(gather(ops.ee, kv.image, kv.grid.coarse(7)), [49, kv.channels])
    dim = q.channels // heads
    result, outputs = {"kv_global": global_kv}, []
    for h in range(heads):
        query = ops.slice_cols(q.image, h * dim, (h + 1) * dim)
        key = ops.wrap(arr.slice_cols(global_kv, h * dim, (h + 1) * dim))
        value = ops.wrap(arr.slice_cols(global_kv, q.channels + h * dim, q.channels + (h + 1) * dim))
        key = key.reproject(q.grid.coarse(7).projection(ops.ee))
        value = value.reproject(q.grid.coarse(7).projection(ops.ee))
        head = naive_attention(ops, query, key, value, tokens=49, dim=dim,
                               score_scale=dim ** -0.5, policy=ROUND_EACH_OP)
        result.update({f"h{h}_{name}": tensor for name, tensor in head.items()})
        outputs.append(head["output"])
    combined = ops.cat_cols(outputs).reproject(q.grid.coarse(7).projection(ops.ee))
    result["out"] = q.like(combined)
    return result


def global_attention(ops, x, c: AttentionConstants) -> dict:
    """SMP's efficient self-attention: the query projection, ``spatial_reduction``, the key and
    value projection, ``shared_key_attention`` and the output projection."""
    _check_feature(x, channels=c.dim)
    q = x.like(linear(ops, x.image, c.q, layout=x, policy=DOUBLE_ROUND_ONCE))
    reduction = spatial_reduction(ops, x, c)
    reduced = reduction["out"]
    kv = reduced.like(linear(ops, reduced.image, c.kv, layout=reduced, policy=DOUBLE_ROUND_ONCE), 2 * c.dim)
    result = shared_key_attention(ops, q, kv, c.heads)
    combined = result["out"]
    projected = x.like(linear(ops, combined.image, c.projection, layout=x, policy=DOUBLE_ROUND_ONCE))
    result.update({"q": q, "sr_projection": reduction["projection"], "sr_norm": reduced,
                   "kv": kv, "concat": combined, "out": projected})
    return result


# ---------------------------------------------------------------------------
# Mix-FFN and the MiT block
# ---------------------------------------------------------------------------
@dataclass
class MixFFNConstants:
    fc1: LinearConstants
    depthwise: DepthwiseConv3x3Constants
    fc2: LinearConstants
    dim: int


def mix_ffn_constants(ops, parameters, prefix, dim, *, layout=LAYOUT):
    width = 4 * dim
    dw = f"{prefix}.dwconv.dwconv"
    depthwise = DepthwiseConv3x3Constants(ops, _parameter(parameters, f"{dw}.weight", (width, 1, 3, 3)),
                                        _parameter(parameters, f"{dw}.bias", (width,)), rows=layout,
                                        policy=DOUBLE_ROUND_ONCE)
    return MixFFNConstants(_linear_constants(ops, parameters, f"{prefix}.fc1", dim, width, layout),
                           depthwise, _linear_constants(ops, parameters, f"{prefix}.fc2", width, dim, layout), dim)


def mix_ffn(ops, x, c: MixFFNConstants) -> dict:
    """fc1, the depthwise 3 x 3 convolution (zero padded at the chip edge), exact GELU, fc2."""
    _check_feature(x, channels=c.dim)
    first = x.like(linear(ops, x.image, c.fc1, layout=x, policy=DOUBLE_ROUND_ONCE), 4 * c.dim)
    depthwise = first.like(depthwise_conv3x3_window(ops, first.image, c.depthwise, first.grid.coarse(7),
                                                 window=7, policy=DOUBLE_ROUND_ONCE))
    activated = depthwise.like(gelu(ops, depthwise.image, policy=DOUBLE_ROUND_ONCE))
    out = x.like(linear(ops, activated.image, c.fc2, layout=activated, policy=DOUBLE_ROUND_ONCE))
    return {"fc1": first, "depthwise": depthwise, "gelu": activated, "out": out}


@dataclass
class MiTBlockConstants:
    norm1: NormConstants
    attention: AttentionConstants
    norm2: NormConstants
    mlp: MixFFNConstants


def mit_block_constants(ops, parameters, prefix, dim, heads, ratio):
    return MiTBlockConstants(norm_constants(ops, parameters, f"{prefix}.norm1", dim, eps=1e-6),
                             attention_constants(ops, parameters, f"{prefix}.attn", dim, heads, ratio),
                             norm_constants(ops, parameters, f"{prefix}.norm2", dim, eps=1e-6),
                             mix_ffn_constants(ops, parameters, f"{prefix}.mlp", dim))


def mit_block(ops, x, c: MiTBlockConstants) -> dict:
    """One MiT block: ``channel_norm``, ``global_attention`` and a residual addition, then
    ``channel_norm``, ``mix_ffn`` and a residual addition."""
    norm1 = channel_norm(ops, x, c.norm1)
    attention = global_attention(ops, norm1, c.attention)
    first = add(x, attention["out"], policy=DOUBLE_ROUND_ONCE)
    norm2 = channel_norm(ops, first, c.norm2)
    mlp = mix_ffn(ops, norm2, c.mlp)
    return {"norm1": norm1, "attention": attention, "first_residual": first,
            "norm2": norm2, "mlp": mlp, "out": add(first, mlp["out"], policy=DOUBLE_ROUND_ONCE)}


# ---------------------------------------------------------------------------
# All-MLP decoder
# ---------------------------------------------------------------------------
def project_pyramid_constants(ops, parameters, prefixes, in_channels, channels: int, *, layout=LAYOUT) -> tuple:
    """``project_pyramid``'s four biased projections, deepest first as it takes them: projection
    ``i`` is the [``channels``, ``in_channels[i]``] ``weight`` and the ``bias`` under ``prefixes[i]``
    (SMP: ``decoder.mlp_stage.{i}.linear``, C4 first)."""
    if isinstance(prefixes, str) or len(prefixes) != 4 or len(in_channels) != 4:
        reject("SegFormer pyramid requires four levels and four projections")
    return tuple(_linear_constants(ops, parameters, prefix, cin, channels, layout)
                 for prefix, cin in zip(prefixes, in_channels))


def project_pyramid(ops, features, projections) -> dict:
    """Project/resize/concat four W7 levels; projections are deepest-first.

    Small fixture widths are permitted here; decoder() separately enforces the
    fixed full-model shape contract. Equal-size interpolation is an identity.
    """
    if len(features) != 4 or len(projections) != 4:
        reject("SegFormer pyramid requires four levels and four projections")
    target = features[0].grid
    result, parts = {}, []
    for i, (feature, projection) in enumerate(zip(reversed(features), projections)):
        _check_feature(feature, channels=projection.in_features)
        factor = 2 ** (3 - i)
        if feature.grid.fine(factor) != target:
            reject("pyramid feature grids must be C1, C1/2, C1/4, C1/8")
        image = linear(ops, feature.image, projection, layout=feature, policy=DOUBLE_ROUND_ONCE)
        projected = feature.like(image, projection.out_features)
        result[f"C{4 - i}_projected"] = projected
        if factor > 1:
            image, fine = window_resize(ops.ee, image, feature.grid, projected.channels, factor,
                                        align_corners=False, window=7, policy=DOUBLE_ROUND_ONCE)
        else:
            fine = feature.grid
        resized = WindowTensor(image, fine, projected.channels, 7)
        result[f"C{4 - i}_to_C1"] = resized
        parts.append(resized)
    result["out"] = concat(parts)
    return result


def fuse_head_constants(ops, parameters, prefixes, in_channels: int, channels: int, classes: int, *,
                        eps: float, layout=LAYOUT) -> tuple:
    """``fuse_head``'s constants ``(fusion, norm, head)`` from the three layers ``prefixes`` names
    (SMP: ``decoder.fuse_stage.0``, ``decoder.fuse_stage.1``, ``segmentation_head.0``): the bias-free
    fuse ``weight`` [``channels``, ``in_channels``, 1, 1]; the BatchNorm's ``weight``, ``bias``,
    ``running_mean`` and ``running_var`` with ``eps``; the head ``weight`` [``classes``, ``channels``,
    1, 1] and its ``bias``."""
    if isinstance(prefixes, str) or len(prefixes) != 3:
        reject("the fused head reads three layers: the fusion, the BatchNorm and the head")
    fusion, norm, head = prefixes
    return (conv1x1_constants(ops, _parameter(parameters, f"{fusion}.weight", (channels, in_channels, 1, 1)),
                              None, layout),
            BatchNormConstants(ops, *(_parameter(parameters, f"{norm}.{name}", (channels,))
                                      for name in ("weight", "bias", "running_mean", "running_var")),
                               rows=layout, eps=eps),
            conv1x1_constants(ops, _parameter(parameters, f"{head}.weight", (classes, channels, 1, 1)),
                              _parameter(parameters, f"{head}.bias", (classes,)), layout))


def fuse_head(ops, x, fusion, norm, head, *, channels: int, classes: int) -> dict:
    """The fused pyramid ``x`` -> W7 logits four times finer: the bias-free fuse ``conv1x1``
    (``fusion``, to ``channels``), inference ``batch_norm`` (``norm``), ``relu``, the biased head
    ``conv1x1`` (``head``, to ``classes``) and four-times ``window_resize`` with
    ``align_corners=True``. Graph construction only; constants of other widths are refused."""
    if fusion.out_features != channels or head.out_features != classes:
        reject(f"fuse and head constants must have {channels} and {classes} outputs")
    fused = x.like(conv1x1(ops, x.image, fusion, layout=x, policy=DOUBLE_ROUND_ONCE), channels)
    normalized = fused.like(batch_norm(ops, fused.image, norm, layout=fused, policy=DOUBLE_ROUND_ONCE))
    activated = normalized.like(relu(ops, normalized.image))
    logits = activated.like(conv1x1(ops, activated.image, head, layout=activated, policy=DOUBLE_ROUND_ONCE),
                            classes)
    image, grid = window_resize(ops.ee, logits.image, logits.grid, classes, 4, align_corners=True,
                                window=7, policy=DOUBLE_ROUND_ONCE)
    return {"fusion": fused, "batch_norm": normalized, "relu": activated, "head": logits,
            "out": WindowTensor(image, grid, classes, 7)}


# ---------------------------------------------------------------------------
# Cost functions (contract section 9): sums of the parts' METADATA cost functions
# ---------------------------------------------------------------------------
_PARTS = {**LINEAR_METADATA, **NORM_METADATA, **CONV_METADATA, **RESIZE_METADATA}
"""The parts' entries by block name: a composite's cost is the sum of its parts' costs."""


def channel_norm_values(channels: int, *, layout=LAYOUT) -> int:
    """gamma and beta [rows, C] (``layer_norm`` and ``layer_norm2d`` count alike)."""
    return _PARTS["layer_norm"].embedded_constants(channels, layout=layout)


def channel_norm_arrays() -> int:
    """Server arrays read: gamma and beta."""
    return _PARTS["layer_norm"].asset_arrays()


def overlap_embedding_values(cin: int, cout: int, *, kernel: int, layout=LAYOUT) -> int:
    """The k x k projection with its bias rows (``linear`` on k k Cin columns at k 7,
    ``conv3x3_window`` at k 3), and the norm."""
    if kernel not in (7, 3):
        reject("overlap embeddings support k7/s4/p3 or k3/s2/p1 only")
    projection = (_PARTS["linear"].embedded_constants(kernel * kernel * cin, cout, bias=True, layout=layout)
                  if kernel == 7 else
                  _PARTS["conv3x3_window"].embedded_constants(cin, cout, bias=True, layout=layout))
    return projection + channel_norm_values(cout, layout=layout)


def overlap_embedding_arrays() -> int:
    """Server arrays read: the projection's weight and bias, and the norm's gamma and beta."""
    return _PARTS["linear"].asset_arrays(bias=True) + channel_norm_arrays()


def spatial_reduction_values(dim: int, ratio: int, *, layout=LAYOUT) -> int:
    """The r x r reduction as a ``linear`` on r r C columns with its bias rows, and its norm;
    nothing at ratio 1."""
    if ratio == 1:
        return 0
    return (_PARTS["linear"].embedded_constants(ratio * ratio * dim, dim, bias=True, layout=layout)
            + channel_norm_values(dim, layout=layout))


def spatial_reduction_arrays(*, ratio: int) -> int:
    """Server arrays read: the reduction's weight and bias and its norm's; none at ratio 1."""
    return 0 if ratio == 1 else _PARTS["linear"].asset_arrays(bias=True) + channel_norm_arrays()


def global_attention_values(dim: int, ratio: int, *, layout=LAYOUT) -> int:
    """The q [C, C], kv [C, 2C] and output [C, C] projections with their bias rows, and the
    spatial reduction (``shared_key_attention`` embeds nothing; the heads cost nothing)."""
    linear_values = _PARTS["linear"].embedded_constants
    return (linear_values(dim, dim, layout=layout) + linear_values(dim, 2 * dim, layout=layout)
            + linear_values(dim, dim, layout=layout) + spatial_reduction_values(dim, ratio, layout=layout))


def global_attention_arrays(*, ratio: int) -> int:
    """Server arrays read: the three projections' weights and biases, and the reduction's."""
    return 3 * _PARTS["linear"].asset_arrays(bias=True) + spatial_reduction_arrays(ratio=ratio)


def mix_ffn_values(dim: int, *, layout=LAYOUT) -> int:
    """fc1 [C, 4C], the biased depthwise convolution of width 4C, and fc2 [4C, C]."""
    linear_values = _PARTS["linear"].embedded_constants
    return (linear_values(dim, 4 * dim, layout=layout)
            + _PARTS["depthwise_conv3x3_window"].embedded_constants(4 * dim, bias=True, layout=layout)
            + linear_values(4 * dim, dim, layout=layout))


def mix_ffn_arrays() -> int:
    """Server arrays read: fc1, the depthwise convolution and fc2, each a weight and a bias."""
    return (2 * _PARTS["linear"].asset_arrays(bias=True)
            + _PARTS["depthwise_conv3x3_window"].asset_arrays(bias=True))


def mit_block_values(dim: int, ratio: int, *, layout=LAYOUT) -> int:
    """Two channel norms, the global attention and the Mix-FFN (the additions embed nothing)."""
    return (2 * channel_norm_values(dim, layout=layout) + global_attention_values(dim, ratio, layout=layout)
            + mix_ffn_values(dim, layout=layout))


def mit_block_arrays(*, ratio: int) -> int:
    """Server arrays read: two norms, the global attention and the Mix-FFN."""
    return 2 * channel_norm_arrays() + global_attention_arrays(ratio=ratio) + mix_ffn_arrays()


def project_pyramid_values(in_channels, channels: int, height: int, width: int, *, layout=LAYOUT) -> int:
    """Four projections, ``in_channels`` (the C1, C2, C3, C4 widths) to ``channels``, with their
    bias rows, and the ``align_corners=False`` resize tables of C2, C3 and C4 (height / 2^k by
    width / 2^k tokens, factor 2^k) onto C1's ``height`` x ``width`` tokens."""
    if len(in_channels) != 4:
        reject("SegFormer pyramid requires four levels and four projections")
    projections = sum(_PARTS["linear"].embedded_constants(c, channels, layout=layout) for c in in_channels)
    return projections + sum(_PARTS["window_resize"].embedded_constants(height // 2 ** k, width // 2 ** k, 2 ** k,
                                                                        align_corners=False, window=7)
                             for k in (1, 2, 3))


def project_pyramid_arrays() -> int:
    """Server arrays read: the four projections' weights and biases (resize tables are inline)."""
    return 4 * _PARTS["linear"].asset_arrays(bias=True)


def fuse_head_values(in_channels: int, channels: int, classes: int, height: int, width: int, *,
                     layout=LAYOUT) -> int:
    """The bias-free fuse [in, channels], the BatchNorm rows, the biased head [channels, classes]
    and the four-times ``align_corners=True`` resize table of a ``height`` x ``width`` token map."""
    return (_PARTS["conv1x1"].embedded_constants(in_channels, channels, bias=False, layout=layout)
            + _PARTS["batch_norm"].embedded_constants(channels, layout=layout)
            + _PARTS["conv1x1"].embedded_constants(channels, classes, bias=True, layout=layout)
            + _PARTS["window_resize"].embedded_constants(height, width, 4, align_corners=True, window=7))


def fuse_head_arrays() -> int:
    """Server arrays read: the fuse weight, the four BatchNorm vectors, the head weight and bias."""
    return (_PARTS["conv1x1"].asset_arrays(bias=False) + _PARTS["batch_norm"].asset_arrays()
            + _PARTS["conv1x1"].asset_arrays(bias=True))


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
# Validated live only inside the fixed SegFormer-B0: the whole model's staged agreement, and
# small compositions (overlap, reduction, global keys, norms, Mix-FFN, pyramid, head).
_STATUS = "Fixed — SegFormer-B0 (mit_b0) at 224 × 224: "
_MODEL = "widths 32, 64, 160, 256; heads 1, 2, 5, 8; reduction ratios 8, 4, 2, 1; window 7"
_EVIDENCE = ("docs/evidence/segformer_b0-fullmodel-live.json", "docs/evidence/segformer_b0-live.json")

# Validated settings (X02b): SegFormer-B0's, named by the constants builders' parameters (the
# block graph's settings, tensor_ee/converter/operations.py) and joined per stage.
_WIDTHS = (32, 64, 160, 256)
_STAGES = tuple({"dim": (dim,), "heads": (heads,), "ratio": (ratio,), "window": (7,)}
                for dim, heads, ratio in zip(_WIDTHS, (1, 2, 5, 8), (8, 4, 2, 1)))
_EMBEDDINGS = tuple({"cin": (cin,), "cout": (cout,), "kernel": (kernel,), "stride": (stride,), "window": (7,)}
                    for cin, cout, kernel, stride in ((2, 32, 7, 4), (32, 64, 3, 2), (64, 160, 3, 2), (160, 256, 3, 2)))

METADATA = validate_metadata({
    "channel_norm": BlockMetadata(
        name="channel_norm", layout_in="W", layout_out="W", locality=Locality.POINTWISE,
        policy=COMPOSITE, embedded_constants=channel_norm_values, asset_arrays=channel_norm_arrays,
        status=_STATUS + "widths 32, 64, 160, 256; eps 1e-5, 1e-6; window 7",
        evidence=_EVIDENCE + ("parts: layer_norm (eps 1e-5) or layer_norm2d (eps 1e-6), double_round_once",),
        validated=Validated(({"channels": _WIDTHS, "eps": (1e-5, 1e-6), "window": (7,)},), sources=(norm_constants,))),
    "decimate": BlockMetadata(
        # A stride-f sample reads within each output token's f x f footprint: k x k, radius
        # f // 2 at the largest factor (4), as stride_columns counts its patch.
        name="decimate", layout_in="W", layout_out="W", locality=Locality.KERNEL, radius=2,
        policy=NO_ARITHMETIC, embedded_constants=no_constants, asset_arrays=no_constants,
        status=_STATUS + "factor 4 at width 32, factor 2 at widths 64, 160, 256; window 7",
        evidence=_EVIDENCE,
        # decimate moves values and takes no width: the widths say where the model ran each factor.
        validated=Validated(({"factor": (2, 4), "window": (7,)},))),
    "overlap_embedding": BlockMetadata(
        name="overlap_embedding", layout_in="W", layout_out="W", locality=Locality.KERNEL, radius=3,
        policy=COMPOSITE, embedded_constants=overlap_embedding_values, asset_arrays=overlap_embedding_arrays,
        status=_STATUS + "k7/s4 2 → 32; k3/s2 32 → 64, 64 → 160, 160 → 256; window 7",
        evidence=_EVIDENCE + ("parts: linear or conv3x3_window and layer_norm (double_round_once), decimate (none)",),
        validated=Validated(_EMBEDDINGS, sources=(overlap_constants,))),
    "spatial_reduction": BlockMetadata(
        name="spatial_reduction", layout_in="W", layout_out="W", locality=Locality.EXTENT,
        policy=COMPOSITE, embedded_constants=spatial_reduction_values, asset_arrays=spatial_reduction_arrays,
        status=_STATUS + "widths 32, 64, 160, 256 at reduction ratios 8, 4, 2, 1; window 7",
        evidence=_EVIDENCE + ("parts: stride_columns (none), linear and layer_norm (double_round_once)",),
        validated=Validated(tuple({name: stage[name] for name in ("dim", "ratio", "window")} for stage in _STAGES),
                            sources=(attention_constants,))),
    "shared_key_attention": BlockMetadata(
        name="shared_key_attention", layout_in="W", layout_out="W", locality=Locality.EXTENT,
        policy=COMPOSITE, embedded_constants=no_constants, asset_arrays=no_constants,
        chip_edge_dependent=True,
        status=_STATUS + "widths 32, 64, 160, 256; heads 1, 2, 5, 8; one 7 × 7 key window",
        evidence=_EVIDENCE + ("parts: gather (none), naive_attention (round_each_op)",),
        validated=Validated(tuple({name: stage[name] for name in ("dim", "heads", "window")} for stage in _STAGES),
                            sources=(attention_constants,))),
    "global_attention": BlockMetadata(
        name="global_attention", layout_in="W", layout_out="W", locality=Locality.EXTENT,
        policy=COMPOSITE, embedded_constants=global_attention_values, asset_arrays=global_attention_arrays,
        chip_edge_dependent=True, status=_STATUS + _MODEL,
        evidence=_EVIDENCE + ("parts: linear (double_round_once), spatial_reduction, shared_key_attention",),
        validated=Validated(_STAGES, sources=(attention_constants,))),
    "mix_ffn": BlockMetadata(
        name="mix_ffn", layout_in="W", layout_out="W", locality=Locality.WINDOW_RING, radius=1,
        policy=COMPOSITE, embedded_constants=mix_ffn_values, asset_arrays=mix_ffn_arrays,
        status=_STATUS + "widths 32, 64, 160, 256, hidden 4 × width; window 7",
        evidence=_EVIDENCE + ("parts: linear, depthwise_conv3x3_window, gelu (double_round_once)",),
        validated=Validated(({"dim": _WIDTHS, "window": (7,)},), sources=(mix_ffn_constants,))),
    "mit_block": BlockMetadata(
        name="mit_block", layout_in="W", layout_out="W", locality=Locality.EXTENT,
        policy=COMPOSITE, embedded_constants=mit_block_values, asset_arrays=mit_block_arrays,
        chip_edge_dependent=True, status=_STATUS + _MODEL,
        evidence=_EVIDENCE + ("parts: channel_norm, global_attention, mix_ffn, add (double_round_once)",),
        validated=Validated(_STAGES, sources=(mit_block_constants,))),
    "project_pyramid": BlockMetadata(
        name="project_pyramid", layout_in="W", layout_out="W", locality=Locality.WINDOW_RING, radius=1,
        policy=COMPOSITE, embedded_constants=project_pyramid_values, asset_arrays=project_pyramid_arrays,
        status=_STATUS + "widths 32, 64, 160, 256 to 256 on C1 56 × 56; window 7",
        evidence=_EVIDENCE + ("parts: linear and window_resize (double_round_once), concat (none)",),
        # in_channels deepest first, as project_pyramid_constants takes them; height x width is C1's.
        validated=Validated(({"in_channels": (_WIDTHS[::-1],), "channels": (256,), "height": (56,), "width": (56,),
                              "window": (7,)},), sources=(project_pyramid_constants,))),
    "fuse_head": BlockMetadata(
        # Chip-edge dependent: the four-times resize's align_corners=True weights depend on the
        # map's size and are not shift-invariant, so tiling with halos cannot reproduce them.
        name="fuse_head", layout_in="W", layout_out="W", locality=Locality.WINDOW_RING, radius=1,
        policy=COMPOSITE, embedded_constants=fuse_head_values, asset_arrays=fuse_head_arrays,
        chip_edge_dependent=True,
        status=_STATUS + "1024 → 256 channels → 2 classes, 56 × 56 to 224 × 224; window 7",
        evidence=_EVIDENCE + ("parts: conv1x1, batch_norm and window_resize (double_round_once), relu (none)",),
        # The BatchNorm's eps 1e-5 never varied; height x width is the input's (C1).
        validated=Validated(({"in_channels": (1024,), "channels": (256,), "classes": (2,), "eps": (1e-5,),
                              "height": (56,), "width": (56,), "window": (7,)},), sources=(fuse_head_constants,))),
})
