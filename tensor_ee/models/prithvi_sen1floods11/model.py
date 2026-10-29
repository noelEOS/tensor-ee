"""One Prithvi Sen1Floods11 crop, composed only from library operations.

The six-band input is already scaled and normalized. Every input pixel must be
finite float32 and fully valid; this graph-only interface cannot verify remote
values or masks, and does not fill missing pixels. Output is two 224x224 logits;
slide inference is outside this interface. Position values are supplied parameters,
never regenerated here. Both the 196 spatial queries and class query attend to all
197 keys through all encoder blocks. Encoder arrays use W14; after the first
upsampling the neck/head use W7. Inference dropout and the final same-size bilinear
resize are identities. The 3x3 decoder convolution has no bias; BN eps is 1e-5.

Source arithmetic: HLS geospatial_fm.py at
https://github.com/NASA-IMPACT/hls-foundation-os/blob/3b6d401f3b4527059af0e44bd640225285e1933d/geospatial_fm/geospatial_fm.py
and timm Attention/Block at
https://github.com/huggingface/pytorch-image-models/blob/7096b52a613eefb4f6d8107366611c8983478b19/timm/models/vision_transformer.py
The encoder uses post-product score scaling, exact GELU, LN eps1e-5; neck LN eps1e-6.
Parameter names follow those sources and the MMSegmentation FCNHead hierarchy.

Capacity estimates at default width: 97,602,818 float state values, 122,783,330
logical constant values with shared matrices counted once; 145.795G MACs/crop.
The 3x3 head contributes 60.897% of MACs. Proposed saved cuts are paired spatial
and class images after each encoder block, then after each upsampling and before
the head. Conservative separate-branch costs fit 24M constants per encoder block;
every saved boundary has at most768 bands. The largest image operand is
[196,3072], 602,112 values or4,816,896 bytes widened to double; the largest
parameter matrix holds2,359,296 values. These are array counts, not peak memory. Request bytes need actual sizing with
the chosen table manifest. No whole-model live or runtime-capacity claim is made.
Region inference is refused: each call covers one complete independent crop and
keeps its final convolution's zero-padding separate from neighboring crops.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import numbers

import numpy as np

from ...core.grid import TokenGrid, to_grid
from ...core.ops import WindowOps
from ...core.tensor import Raster, ServerArray, WindowTensor, band_names
from ...weights.constants import ValueLayout, WindowLayout, row_constant
from ...blocks.window.activations import gelu, relu
from ...blocks.window.attention import split_heads, naive_attention
from ...blocks.window.conv import Conv3x3Constants, conv3x3_window
from ...blocks.window.linear import LinearConstants, linear, conv1x1
from ...blocks.window.merge import add, concat_rows
from ...blocks.window.normalization import (
    BatchNormConstants, batch_norm, layer_norm, layer_norm2d,
)
from ...blocks.window.patch import patch_embed_constants, patch_embed
from ...blocks.window.resample import window_depth_to_space
from ...blocks.window.upernet import head_dense
from .configuration import PRITHVI, PrithviConfig, parameter_shapes, stage_names


INPUT_GRID = TokenGrid(224, 224, scale=1.0)
TOKEN_GRID = INPUT_GRID.coarse(16)
BANDS = tuple(f"b{index}" for index in range(6))
COUNTER = "decode_head.convs.0.bn.num_batches_tracked"


def validate_parameters(parameters, config=PRITHVI):
    """Require inference names/shapes; permit flattened ServerArray asset matrices.

    NumPy tensors retain native state shapes. ServerArrays may use native shapes
    or [first_axis, product(other_axes)]; vectors may also be [1,C] or [C,1].
    Auxiliary tensors and the BN counter are ignored. Server values are not read;
    their finite-float32 contract belongs to the asset provider.
    """
    if not isinstance(parameters, Mapping):
        raise TypeError("parameters must map state names to NumPy arrays or ServerArrays")
    shapes = parameter_shapes(config)
    missing = set(shapes) - set(parameters)
    extra = {
        key for key in parameters
        if key not in shapes and key != COUNTER and not key.startswith("auxiliary_head.")
    }
    if missing or extra:
        raise ValueError(
            f"inference parameter names differ: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    for name, shape in shapes.items():
        value = parameters[name]
        if not isinstance(value, (np.ndarray, ServerArray)):
            raise ValueError(f"{name} must be a NumPy array or ServerArray")
        received = tuple(value.shape)
        allowed = {shape}
        if isinstance(value, ServerArray):
            invalid = any(
                isinstance(size, bool) or not isinstance(size, numbers.Integral) or size < 1
                for size in received
            )
            if invalid:
                raise ValueError(f"{name} has invalid server dimensions")
            allowed.add((shape[0], math.prod(shape[1:])))
            if len(shape) == 1:
                allowed.add((1, shape[0]))
        if received not in allowed:
            raise ValueError(f"{name} must have native shape {shape} or its flattened asset matrix")
        if isinstance(value, np.ndarray):
            if value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f"{name} must contain finite float32 values")
    variance = parameters["decode_head.convs.0.bn.running_var"]
    if isinstance(variance, np.ndarray) and np.any(variance < 0):
        raise ValueError("BatchNorm running variance must be nonnegative")


def _reshape(value, shape):
    if isinstance(value, ServerArray):
        return ServerArray(value.array.reshape(list(shape)), tuple(shape))
    return value.reshape(shape)


def _upsampling_parameters(weight, bias):
    """ConvTranspose [in,out,2,2] -> linear [4*out,in], position-major columns."""
    cin, cout = weight.shape[:2]
    if isinstance(weight, ServerArray):
        packed = (weight.array.reshape([cin, cout, 4]).transpose(1, 2)
                  .reshape([cin, 4 * cout]).transpose())
        weight = ServerArray(packed, (4 * cout, cin))
    else:
        weight = weight.transpose(0, 2, 3, 1).reshape(cin, 4 * cout).T
    if isinstance(bias, ServerArray):
        repeated = bias.array.reshape([1, cout]).repeat(0, 4).reshape([4 * cout])
        bias = ServerArray(repeated, (4 * cout,))
    else:
        bias = np.tile(bias, 4)
    return weight, bias


def compose(backend, inputs, config=PRITHVI, *, start="input", stop="logits"):
    """Execute the same model wiring with a library or numerical backend.

    ``start`` names the saved boundary supplied by ``inputs``. Encoder boundaries
    are pairs (spatial,class); neck boundaries are spatial values. ``stop`` is
    returned in a dictionary alongside every intermediate boundary traversed.
    The backend supplies primitives, coefficients, and layout representations.
    """
    names = stage_names(config)
    if start not in names or stop not in names or names.index(stop) < names.index(start):
        raise ValueError("start/stop must name ordered model boundaries")
    current = inputs
    result = {start: current}
    for name in names[names.index(start) + 1:names.index(stop) + 1]:
        if name == "patch":
            current = backend.position(backend.patch(current))
        elif name.startswith("block"):
            prefix = f"backbone.blocks.{int(name[5:])}"
            spatial, cls = current
            qkv = []
            for value in (spatial, cls):
                normalized = backend.norm(value, prefix + ".norm1", 1e-5)
                qkv.append(backend.linear(normalized, prefix + ".attn.qkv"))
            attention = backend.attention(qkv[0], qkv[1], config.heads)
            first = [
                backend.add(value, backend.linear(attended, prefix + ".attn.proj"))
                for value, attended in zip((spatial, cls), attention)
            ]
            outputs = []
            for value in first:
                normalized = backend.norm(value, prefix + ".norm2", 1e-5)
                hidden = backend.linear(normalized, prefix + ".mlp.fc1")
                activated = backend.gelu(hidden)
                projected = backend.linear(activated, prefix + ".mlp.fc2")
                outputs.append(backend.add(value, projected))
            current = tuple(outputs)
        elif name == "norm":
            current = tuple(backend.norm(value, "backbone.norm", 1e-5) for value in current)
        elif name.startswith("upsample"):
            side = int(name[8:])
            if side == 28:
                current = current[0]
            prefix = "neck.fpn1" if side <= 56 else "neck.fpn2"
            index = 0 if side in (28, 112) else 3
            current = backend.upsample(current, f"{prefix}.{index}")
            if index == 0:
                current = backend.gelu(backend.norm(current, prefix + ".1.ln", 1e-6))
        elif name == "logits":
            current = backend.conv3x3(current, "decode_head.convs.0.conv")
            current = backend.relu(backend.batch_norm(current, "decode_head.convs.0.bn"))
            current = backend.head_dense(backend.linear(current, "decode_head.conv_seg"))
        result[name] = current
    return result


class LibraryBackend:
    """Coefficients are all built here, before any spatial forward is composed."""

    def __init__(self, ee, parameters, config=PRITHVI):
        validate_parameters(parameters, config)
        self.ee, self.ops, self.config = ee, WindowOps(ee), config
        self.norms = {}
        self.linears = {}
        dim = config.width
        state = {
            name: _reshape(parameters[name], shape)
            for name, shape in parameter_shapes(config).items()
        }
        patch = {
            "proj.weight": _reshape(state["backbone.patch_embed.proj.weight"], (dim, 1536)),
            "proj.bias": state["backbone.patch_embed.proj.bias"],
        }
        self.patch_constants = patch_embed_constants(
            self.ops, patch, rows=WindowLayout(14), norm=False,
        )
        position = state["backbone.pos_embed"]
        if isinstance(position, ServerArray):
            spatial = ServerArray(position.array.slice(1, 1, 197).reshape([196, dim]), (196, dim))
            first = ServerArray(position.array.slice(1, 0, 1).reshape([1, dim]), (1, dim))
        else:
            spatial, first = position[0, 1:], position[0, :1]
        self.position_spatial = self.ops.constant(spatial)
        class_grid = TOKEN_GRID.coarse(14)
        cls = WindowTensor(
            self.ops.constant(_reshape(state["backbone.cls_token"], (1, dim))),
            class_grid, dim, 1,
        )
        self.position_class = add(cls, cls.like(self.ops.constant(first))).image

        def norm(prefix, rows):
            gamma = row_constant(self.ops, state[prefix + ".weight"], ValueLayout(rows))
            beta = row_constant(self.ops, state[prefix + ".bias"], ValueLayout(rows))
            self.norms[prefix, rows] = gamma, beta

        def projection(prefix, rows, shape=None):
            weight = state[prefix + ".weight"]
            if shape is not None:
                weight = _reshape(weight, shape)
            self.linears[prefix, rows] = LinearConstants(
                self.ops, weight, state[prefix + ".bias"], rows,
            )

        for index in range(config.depth):
            prefix = f"backbone.blocks.{index}"
            for rows in (196, 1):
                norm(prefix + ".norm1", rows)
                norm(prefix + ".norm2", rows)
                for suffix in ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"):
                    projection(prefix + "." + suffix, rows)
        norm("backbone.norm", 196)
        norm("backbone.norm", 1)
        for prefix in ("neck.fpn1", "neck.fpn2"):
            norm(prefix + ".1.ln", 49)
            for index in (0, 3):
                rows = 196 if prefix == "neck.fpn1" and index == 0 else 49
                name = f"{prefix}.{index}"
                weight, bias = _upsampling_parameters(
                    state[name + ".weight"], state[name + ".bias"],
                )
                self.linears[name, rows] = LinearConstants(self.ops, weight, bias, rows)
        weight = state["decode_head.convs.0.conv.weight"]
        if isinstance(weight, ServerArray):
            weight = _reshape(weight, (config.decoder, 9 * dim))
        self.conv_constants = Conv3x3Constants(self.ops, weight, rows=49)
        prefix = "decode_head.convs.0.bn"
        self.bn_constants = BatchNormConstants(
            self.ops, state[prefix + ".weight"], state[prefix + ".bias"],
            state[prefix + ".running_mean"], state[prefix + ".running_var"],
            rows=49, eps=1e-5,
        )
        projection("decode_head.conv_seg", 49, (2, config.decoder))

    def patch(self, raster):
        valid = (
            isinstance(raster, Raster)
            and (raster.grid.height, raster.grid.width) == (224, 224)
            and len(raster.bands) == 6
        )
        if not valid:
            raise ValueError("input must be a complete 224 by 224 six-band Raster")
        source = to_grid(raster.grid)
        image = raster.image.select(list(raster.bands)).rename(BANDS).reproject(source.projection())
        image = image.changeProj(source.projection(), INPUT_GRID.projection(self.ee))
        image = image.updateMask(INPUT_GRID.domain(self.ee))
        output = patch_embed(
            self.ops, image, INPUT_GRID, self.patch_constants,
            patch=16, bands=BANDS, window=14, norm=False,
        )["x"]
        return WindowTensor(output, TOKEN_GRID, self.config.width, 14)

    def position(self, spatial):
        output = add(spatial, spatial.like(self.position_spatial))
        grid = spatial.grid.coarse(spatial.window)
        image = self.position_class.reproject(grid.projection(self.ee))
        image = image.updateMask(grid.domain(self.ee))
        cls = WindowTensor(image, grid, spatial.channels, 1)
        return output, cls

    def norm(self, tensor, prefix, eps):
        gamma, beta = self.norms[prefix, tensor.rows]
        layout = WindowLayout(tensor.window)
        if eps == 1e-6:
            output = layer_norm2d(
                self.ops, tensor.image, gamma, beta, dim=tensor.channels, rows=layout, eps=eps,
            )
        else:
            output = layer_norm(
                self.ops, tensor.image, gamma, beta, dim=tensor.channels, eps=eps, layout=layout,
            )
        return tensor.like(output)

    def linear(self, tensor, prefix):
        layer = self.linears[prefix, tensor.rows]
        function = conv1x1 if prefix == "decode_head.conv_seg" else linear
        output = function(self.ops, tensor.image, layer, layout=WindowLayout(tensor.window))
        return tensor.like(output, layer.out_features)

    def attention(self, spatial, cls, heads):
        dim = self.config.width
        valid = (
            spatial.window == 14 and cls.window == 1
            and spatial.channels == 3 * dim and cls.channels == 3 * dim
            and spatial.grid.coarse(14) == cls.grid
        )
        if not valid:
            raise ValueError("attention requires W14 spatial and W1 class rows on the same crop grid")
        spatial_heads = split_heads(self.ops, spatial.image, dim, heads)
        class_heads = split_heads(self.ops, cls.image, dim, heads)
        outputs = [[], []]
        for head in range(heads):
            keys = concat_rows(
                self.ops, [class_heads[f"k_h{head}"], spatial_heads[f"k_h{head}"]],
                rows=(1, 196), channels=dim // heads,
            )
            values = concat_rows(
                self.ops, [class_heads[f"v_h{head}"], spatial_heads[f"v_h{head}"]],
                rows=(1, 196), channels=dim // heads,
            )
            for index, part in enumerate((spatial_heads, class_heads)):
                output = naive_attention(
                    self.ops, part[f"q_h{head}"], keys, values,
                    tokens=197, dim=dim // heads, score_scale=(dim // heads) ** -0.5,
                )["output"]
                outputs[index].append(output)
        return (
            spatial.like(self.ops.cat_cols(outputs[0]), dim),
            cls.like(self.ops.cat_cols(outputs[1]), dim),
        )

    def upsample(self, tensor, prefix):
        projected = self.linear(tensor, prefix)
        output, grid = window_depth_to_space(
            self.ee, projected.image, tensor.grid, self.config.width,
            window=tensor.window, out_window=7,
        )
        return WindowTensor(output, grid, self.config.width, 7)

    def conv3x3(self, tensor, prefix):
        output = conv3x3_window(
            self.ops, tensor.image, self.conv_constants,
            tensor.grid.coarse(tensor.window), window=tensor.window,
        )
        return tensor.like(output, self.config.decoder)

    def batch_norm(self, tensor, prefix):
        output = batch_norm(
            self.ops, tensor.image, self.bn_constants, layout=WindowLayout(tensor.window),
        )
        return tensor.like(output)

    def add(self, first, second):
        return add(first, second)

    def gelu(self, tensor):
        return tensor.like(gelu(self.ops, tensor.image))

    def relu(self, tensor):
        return tensor.like(relu(self.ops, tensor.image))

    def head_dense(self, tensor):
        image = head_dense(self.ee, tensor.image, tensor.grid, tensor.channels)
        return Raster(image, tuple(band_names(2)), tensor.grid)


@dataclass(frozen=True)
class PrithviConstants:
    config: PrithviConfig
    backend: LibraryBackend


def constants(ee, parameters, *, config=PRITHVI):
    """Build and check all coefficient constants from NumPy arrays or ServerArrays."""
    return PrithviConstants(config, LibraryBackend(ee, parameters, config))


def forward(ee, raster, coefficients: PrithviConstants, *, stop="logits"):
    """Build one complete-crop graph, preserving its georeferenced output grid."""
    if coefficients.backend.ee is not ee:
        raise ValueError("constants and forward must use the same Earth Engine module")
    result = compose(coefficients.backend, raster, coefficients.config, stop=stop)
    if "logits" in result:
        output = result["logits"]
        target = to_grid(raster.grid)
        image = output.image.changeProj(output.grid.projection(ee), target.projection())
        image = image.updateMask(target.domain(ee))
        result["logits"] = Raster(image, output.bands, target)
    return result
