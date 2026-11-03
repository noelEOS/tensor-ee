"""Hand-composed SMP 0.5.0 DPT, using the existing ViT encoder and library blocks.

The input is a fully valid finite six-band 224x224 float32 crop, already prepared
by its caller. Raw encoder pairs at uniformly sampled blocks feed cat readouts.
W14(14) reassembles to W7(56), W7(28), W14(14), W7(7). Fusion runs coarse to
fine: W7(7) -> W14(14) -> W7(28) -> W7(56) -> W7(112). The head resizes its
two logits to W7(224), then moves them to dense bands. Every resize is x2 with
align_corners=True. BN epsilon is 1e-5, exact GELU, dropout is identity in eval.

All arithmetic is in existing blocks with their existing policies. Parameters
use SMP state_dict names as native NumPy float32 arrays or flattened ServerArrays.
No new block, preprocessing, converter, cloud call or model-accuracy claim.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import numbers

import numpy as np

from ...blocks.window.activations import relu
from ...blocks.window.conv import Conv3x3Constants, conv3x3_window
from ...blocks.window.linear import LinearConstants, conv1x1_constants
from ...blocks.window.merge import concat
from ...blocks.window.normalization import BatchNormConstants, batch_norm
from ...blocks.window.resample import window_depth_to_space, window_resize
from ...core.tensor import Raster, ServerArray, WindowTensor, band_names
from ...layouts.window import to_dense
from ...weights.constants import WindowLayout
from ..prithvi_sen1floods11.model import _upsampling_parameters, reshape_parameter
from ..vit_b16_encoder224 import model as encoder
from ..vit_b16_encoder224.configuration import parameter_shapes as encoder_shapes
from .configuration import DPT_B16, DPTConfig, parameter_shapes

BANDS, INPUT_GRID, TOKEN_GRID = encoder.BANDS, encoder.INPUT_GRID, encoder.TOKEN_GRID


def validate_parameters(parameters, config=DPT_B16):
    """Require finite float32 inference tensors; ignore only known unused tensors."""
    if not isinstance(parameters, Mapping):
        raise TypeError("parameters must map SMP state names to NumPy arrays or ServerArrays")
    shapes = parameter_shapes(config)
    counters = {name.rsplit(".", 1)[0] + ".num_batches_tracked"
                for name in shapes if name.endswith(".running_var")}
    ignored = counters | {"encoder.model.head.weight", "encoder.model.head.bias"}
    missing, extra = shapes.keys() - parameters.keys(), parameters.keys() - shapes.keys() - ignored
    if missing or extra:
        raise ValueError(f"DPT parameter names differ: missing={sorted(missing)}, extra={sorted(extra)}")
    for name, shape in shapes.items():
        value = parameters[name]
        if not isinstance(value, (np.ndarray, ServerArray)):
            raise ValueError(f"{name} must be a NumPy array or ServerArray")
        received = tuple(value.shape)
        allowed = {shape}
        if isinstance(value, ServerArray):
            if any(isinstance(n, bool) or not isinstance(n, numbers.Integral) or n < 1 for n in received):
                raise ValueError(f"{name} has invalid server dimensions")
            allowed.add((shape[0], math.prod(shape[1:])))
            if len(shape) == 1:
                allowed.add((1, shape[0]))
        if received not in allowed:
            raise ValueError(f"{name} must have native shape {shape} or its flattened asset matrix")
        if isinstance(value, np.ndarray):
            if value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f"{name} must contain finite float32 values")
            if name.endswith("running_var") and np.any(value < 0):
                raise ValueError("BatchNorm running variance must be nonnegative")


def readout(backend, pair, index):
    """Spatial columns first, repeated class columns second, then linear and GELU."""
    joined = backend.cat_readout(*pair)
    return backend.gelu(backend.linear(joined, f"decoder.projection_blocks.{index}.project.0"))


def reassemble(backend, tensor, index):
    prefix = f"decoder.reassemble_blocks.{index}"
    tensor = backend.linear(tensor, prefix + ".project_to_out_channel")
    if index < 2:
        tensor = backend.upsample(tensor, prefix + ".upsample", (4, 2)[index])
    elif index == 3:
        tensor = backend.conv(tensor, prefix + ".upsample", stride=2)
    return backend.conv(tensor, prefix + ".project_to_feature_dim")


def residual(backend, tensor, prefix):
    skip = tensor
    for j in (1, 2):
        tensor = backend.conv(backend.relu(tensor), prefix + f".conv_{j}")
        tensor = backend.bn(tensor, prefix + f".batch_norm_{j}")
    return backend.add(tensor, skip)


def fusion(backend, tensor, previous, index):
    prefix = f"decoder.fusion_blocks.{index}"
    tensor = residual(backend, tensor, prefix + ".residual_conv_block1")
    if previous is not None:
        tensor = backend.add(tensor, previous)
    tensor = residual(backend, tensor, prefix + ".residual_conv_block2")
    tensor = backend.resize(tensor, out_window=(14, 7, None, None)[index])
    return backend.linear(tensor, prefix + ".project")


def head(backend, tensor):
    tensor = backend.conv(tensor, "segmentation_head.head.0")
    tensor = backend.relu(backend.bn(tensor, "segmentation_head.head.1"))
    return backend.linear(tensor, "segmentation_head.head.4")


def decoder(backend, pairs, config=DPT_B16):
    """Expose every readout, reassemble, fusion, pre-resize head and dense logit."""
    pairs = tuple(pairs)
    if len(pairs) != 4:
        raise ValueError("DPT decoder requires exactly four spatial/class pairs")
    outputs = {}
    for i, pair in enumerate(pairs):
        outputs[f"readout{i}"] = readout(backend, pair, i)
        outputs[f"reassemble{i}"] = reassemble(backend, outputs[f"readout{i}"], i)
    previous = None
    for i in range(4):
        previous = fusion(backend, outputs[f"reassemble{3-i}"], previous, i)
        outputs[f"fusion{i}"] = previous
    outputs["head"] = head(backend, previous)
    outputs["logits"] = backend.dense(backend.resize(outputs["head"]))
    return outputs


def compose(backend, inputs, config=DPT_B16):
    """Reuse the merged encoder, selecting its four raw pairs without final norm."""
    encoded = encoder.compose(backend, inputs, config.encoder, stop=f"block{config.encoder.depth-1}")
    pairs = [encoded[f"block{i}"] for i in config.indices]
    return {**{f"encoder{i}": pair for i, pair in enumerate(pairs)},
            **decoder(backend, pairs, config)}


class LibraryBackend(encoder.LibraryBackend):
    """Shared ViT backend plus coefficients for existing decoder blocks, built ahead."""

    def __init__(self, ee, parameters, config=DPT_B16):
        validate_parameters(parameters, config)
        super().__init__(ee, {name: parameters[name] for name in encoder_shapes(config.encoder)}, config.encoder)
        state = {name: reshape_parameter(parameters[name], shape)
                 for name, shape in parameter_shapes(config).items() if not name.startswith("encoder.")}
        self.convs, self.bns = {}, {}
        for name, weight in state.items():
            if not name.endswith(".weight"):
                continue
            prefix = name.removesuffix(".weight")
            bias = state.get(prefix + ".bias")
            if prefix.startswith("decoder.projection_blocks."):
                self.linears[prefix, 196] = LinearConstants(self.ops, weight, bias, 196)
            elif "upsample" in prefix and weight.shape[-1] in (2, 4):
                factor = weight.shape[-1]
                packed, repeated = _upsampling_parameters(weight, bias, factor=factor)
                self.linears[prefix, 196] = LinearConstants(self.ops, packed, repeated, 196)
            elif len(weight.shape) == 4:
                if prefix.startswith("decoder.reassemble_blocks."):
                    i = int(prefix.split(".")[2])
                    rows = 196 if (prefix.endswith("project_to_out_channel") or i == 2) else 49
                elif prefix.startswith("decoder.fusion_blocks.1."):
                    rows = 49 if prefix.endswith(".project") else 196
                else:
                    rows = 196 if prefix == "decoder.fusion_blocks.0.project" else 49
                if weight.shape[-1] == 1:
                    weight = reshape_parameter(weight, (weight.shape[0], math.prod(weight.shape[1:])))
                    self.linears[prefix, rows] = conv1x1_constants(self.ops, weight, bias, rows)
                else:
                    if isinstance(weight, ServerArray):
                        weight = reshape_parameter(weight, (weight.shape[0], math.prod(weight.shape[1:])))
                    self.convs[prefix] = Conv3x3Constants(self.ops, weight, bias, rows=rows)
            elif prefix + ".running_var" in state:
                rows = 196 if prefix.startswith("decoder.fusion_blocks.1.") else 49
                self.bns[prefix] = BatchNormConstants(self.ops, weight, bias,
                    state[prefix + ".running_mean"], state[prefix + ".running_var"], rows=rows, eps=1e-5)

    def cat_readout(self, spatial, cls):
        if (spatial.window != 14 or cls.window != 1 or cls.channels != spatial.channels
                or cls.grid != spatial.grid.coarse(14)):
            raise ValueError("cat readout requires matching W14 spatial and W1 class rows")
        repeated = spatial.like(self.ops.repeat_rows(cls.image, spatial.rows))
        return concat([spatial, repeated])

    def upsample(self, tensor, prefix, factor):
        projected = self.linear(tensor, prefix)
        output, grid = window_depth_to_space(self.ee, projected.image, tensor.grid,
            projected.channels // (factor * factor), window=tensor.window, out_window=7, factor=factor)
        return WindowTensor(output, grid, projected.channels // (factor * factor), 7)

    def conv(self, tensor, prefix, *, stride=1):
        c = self.convs[prefix]
        output = conv3x3_window(self.ops, tensor.image, c, tensor.grid.coarse(tensor.window),
                               window=tensor.window, stride=stride)
        grid = tensor.grid if stride == 1 else tensor.grid.coarse(stride)
        return WindowTensor(output, grid, c.cout, tensor.window // stride)

    def bn(self, tensor, prefix):
        return tensor.like(batch_norm(self.ops, tensor.image, self.bns[prefix], layout=WindowLayout(tensor.window)))

    def relu(self, tensor):
        return tensor.like(relu(self.ops, tensor.image))

    def resize(self, tensor, *, out_window=None):
        output, grid = window_resize(self.ee, tensor.image, tensor.grid, tensor.channels, 2,
                                    align_corners=True, window=tensor.window, out_window=out_window)
        return WindowTensor(output, grid, tensor.channels, tensor.window if out_window is None else out_window)

    def dense(self, tensor):
        image = to_dense(self.ee, tensor.grid, tensor.image, tensor.channels, window=tensor.window)
        return Raster(image, tuple(band_names(2)), tensor.grid)


@dataclass(frozen=True)
class DPTConstants:
    config: DPTConfig
    backend: LibraryBackend


def constants(ee, parameters, *, config=DPT_B16):
    return DPTConstants(config, LibraryBackend(ee, parameters, config))


def forward(ee, raster, coefficients: DPTConstants):
    """Build all comparison boundaries; logits are dense on the abstract input grid."""
    if coefficients.backend.ee is not ee:
        raise ValueError("constants and forward must use the same Earth Engine module")
    return compose(coefficients.backend, raster, coefficients.config)
