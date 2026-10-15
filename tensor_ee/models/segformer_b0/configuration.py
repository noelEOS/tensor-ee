"""The fixed 224 x 224, two-band, two-logit SMP 0.5.0 mit_b0 contract.

The schema keeps SMP state_dict names. NumPy weights have their original shapes
and float32 dtype; convolution ServerArrays use [out, flattened input] storage,
in PyTorch (channel, ky, kx) order. The BatchNorm tracking counter is recorded but
never used in inference arithmetic. Server value validity cannot be proved
client-side: finite float32 weights and a fully valid finite float32 input are
caller preconditions. No constructor evaluates or initializes Earth Engine.

This model and its global attention depend on the complete fixed chip (EXTENT,
chip_edge_dependent=True). Full staged live agreement on one seeded chip/state
is recorded in docs/evidence/segformer_b0-fullmodel-live.json under the 1e-4
absolute target. Model accuracy and monolithic runtime are unassessed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...blocks.window.conv import METADATA as CONV_METADATA
from ...blocks.window.linear import METADATA as LINEAR_METADATA
from ...blocks.window.normalization import METADATA as NORM_METADATA
from ...blocks.window.resample import METADATA as RESIZE_METADATA
from ...blocks.window.segformer import _server_shape  # moved to the library (X01a)
from ...core.errors import reject
from ...core.grid import TokenGrid
from ...core.metadata import Locality
from ...weights.constants import ServerArray, WindowLayout

STAGE_DIMS = (32, 64, 160, 256)
NUM_HEADS = (1, 2, 5, 8)
SR_RATIOS = (8, 4, 2, 1)
DEPTHS = (2, 2, 2, 2)
COUNTER = "decoder.fuse_stage.1.num_batches_tracked"
LOCALITY = Locality.EXTENT
CHIP_EDGE_DEPENDENT = True


@dataclass(frozen=True)
class SegformerB0Config:
    """Only the approved fixed configuration is accepted, including band labels."""

    height: int = 224
    width: int = 224
    batch_size: int = 1
    bands: tuple = ("b0", "b1")
    classes: int = 2
    window: int = 7
    training: bool = False
    activation: object = None

    def __post_init__(self):
        for name, expected in (("height", 224), ("width", 224), ("batch_size", 1),
                               ("classes", 2), ("window", 7)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value != expected:
                reject(f"SegFormer-B0 requires {name}={expected}; received {value!r}")
        if self.bands != ("b0", "b1") or not isinstance(self.bands, tuple):
            reject("SegFormer-B0 requires ordered bands ('b0', 'b1')")
        if self.training is not False or self.activation is not None:
            reject("SegFormer-B0 supports evaluation mode and raw logits only")

    @property
    def layout(self) -> WindowLayout:
        return WindowLayout(self.window)

    @property
    def input_grid(self) -> TokenGrid:
        return TokenGrid(self.height, self.width, 2.5)

    @property
    def stage_grids(self) -> tuple:
        return tuple(self.input_grid.coarse(4 * 2 ** s) for s in range(4))

    @property
    def output_bands(self) -> tuple:
        return ("c0", "c1")


SEGFORMER_B0 = SegformerB0Config()


def parameter_shapes() -> dict[str, tuple]:
    """All 192 original SMP state_dict entries, including the unused BN counter."""
    shapes = {}

    def affine(prefix, width):
        shapes[f"{prefix}.weight"] = (width,)
        shapes[f"{prefix}.bias"] = (width,)

    def projection(prefix, out_width, in_width, kernel=None, bias=True):
        shapes[f"{prefix}.weight"] = ((out_width, in_width) if kernel is None else
                                     (out_width, in_width, kernel, kernel))
        if bias:
            shapes[f"{prefix}.bias"] = (out_width,)

    for s, (dim, ratio) in enumerate(zip(STAGE_DIMS, SR_RATIOS), 1):
        projection(f"encoder.patch_embed{s}.proj", dim, 2 if s == 1 else STAGE_DIMS[s - 2],
                   7 if s == 1 else 3)
        affine(f"encoder.patch_embed{s}.norm", dim)
        for b in range(DEPTHS[s - 1]):
            p = f"encoder.block{s}.{b}"
            affine(f"{p}.norm1", dim)
            affine(f"{p}.norm2", dim)
            projection(f"{p}.attn.q", dim, dim)
            projection(f"{p}.attn.kv", 2 * dim, dim)
            projection(f"{p}.attn.proj", dim, dim)
            if ratio > 1:
                projection(f"{p}.attn.sr", dim, dim, ratio)
                affine(f"{p}.attn.norm", dim)
            projection(f"{p}.mlp.fc1", 4 * dim, dim)
            projection(f"{p}.mlp.dwconv.dwconv", 4 * dim, 1, 3)
            projection(f"{p}.mlp.fc2", dim, 4 * dim)
        affine(f"encoder.norm{s}", dim)
    for i, dim in enumerate(reversed(STAGE_DIMS)):
        projection(f"decoder.mlp_stage.{i}.linear", 256, dim)
    projection("decoder.fuse_stage.0", 256, 1024, 1, bias=False)
    affine("decoder.fuse_stage.1", 256)
    shapes["decoder.fuse_stage.1.running_mean"] = (256,)
    shapes["decoder.fuse_stage.1.running_var"] = (256,)
    shapes[COUNTER] = ()
    projection("segmentation_head.0", 2, 256, 1)
    return shapes


def check_parameters(parameters: dict) -> None:
    """Refuse missing/extra names, invalid shapes, and invalid NumPy dtype/values.

    ServerArray shapes are checked, but their values are a caller precondition.
    Four-dimensional convolution weights are flattened for ServerArray storage;
    vectors/matrices keep their original shapes. The counter is an integer scalar.
    """
    shapes = parameter_shapes()
    missing, extra = set(shapes) - set(parameters), set(parameters) - set(shapes)
    if missing or extra:
        reject(f"SegFormer-B0 state names differ: missing={sorted(missing)}, extra={sorted(extra)}")
    for name, shape in shapes.items():
        value = parameters[name]
        if isinstance(value, ServerArray):
            expected = (shape[0], int(np.prod(shape[1:]))) if len(shape) == 4 else shape
            if _server_shape(value, name) != expected:
                reject(f"{name}: ServerArray shape {value.shape}, expected {expected}")
            continue
        if not isinstance(value, np.ndarray) or value.shape != shape:
            reject(f"{name}: expected a NumPy array with shape {shape}")
        if name == COUNTER:
            if not np.issubdtype(value.dtype, np.integer) or value.item() < 0:
                reject(f"{name}: expected a nonnegative integer scalar")
        elif value.dtype != np.float32 or not np.isfinite(value).all():
            reject(f"{name}: expected finite float32 values")
    variance = parameters["decoder.fuse_stage.1.running_var"]
    if not isinstance(variance, ServerArray) and np.any(variance < 0):
        reject("BatchNorm running variance must be nonnegative")


def constant_costs(config: SegformerB0Config = SEGFORMER_B0) -> dict:
    """Evaluated constants and source-array reads, calling actual block METADATA.

    Values count repeated arrays once per layer, shared by spatial windows.
    Server arrays count source state entries; repeats create no provider records.
    Input activation literals are excluded. No model activation is evaluated.
    """
    if config != SEGFORMER_B0:
        reject("only the fixed SegFormer-B0 configuration is supported")
    layout = config.layout
    result = {}

    def linear(a, b, bias=True):
        m = LINEAR_METADATA["linear"]
        return (m.embedded_constants(a, b, bias=bias, layout=layout), m.asset_arrays(bias=bias))

    def norm(c):
        m = NORM_METADATA["layer_norm"]
        return m.embedded_constants(c, layout=layout), m.asset_arrays()

    def total(parts):
        return {"values": sum(p[0] for p in parts), "asset_arrays": sum(p[1] for p in parts)}

    for i, (dim, ratio) in enumerate(zip(STAGE_DIMS, SR_RATIOS)):
        cin = 2 if i == 0 else STAGE_DIMS[i - 1]
        if i == 0:
            embedding = linear(49 * cin, dim)
        else:
            m = CONV_METADATA["conv3x3_window"]
            embedding = (m.embedded_constants(cin, dim, bias=True, layout=layout), m.asset_arrays(bias=True))
        parts = [embedding, norm(dim), norm(dim)]
        for _ in range(DEPTHS[i]):
            parts += [norm(dim), norm(dim), linear(dim, dim), linear(dim, 2 * dim),
                      linear(dim, dim), linear(dim, 4 * dim), linear(4 * dim, dim)]
            m = CONV_METADATA["depthwise_conv3x3_window"]
            parts.append((m.embedded_constants(4 * dim, bias=True, layout=layout), m.asset_arrays(bias=True)))
            if ratio > 1:
                parts += [linear(ratio * ratio * dim, dim), norm(dim)]
        result[f"encoder_stage{i + 1}"] = total(parts)
    decoder = [linear(dim, 256) for dim in reversed(STAGE_DIMS)] + [linear(1024, 256, False), linear(256, 2)]
    m = NORM_METADATA["batch_norm"]
    decoder.append((m.embedded_constants(256, layout=layout), m.asset_arrays()))
    result["decoder_head"] = total(decoder)
    m = RESIZE_METADATA["window_resize"]
    resizes = [m.embedded_constants(g.height, g.width, 2 ** (3 - i), align_corners=False, window=7)
               for i, g in enumerate(reversed(config.stage_grids)) if i < 3]
    g = config.stage_grids[0]
    resizes.append(m.embedded_constants(g.height, g.width, 4, align_corners=True, window=7))
    result["resize"] = {"values": sum(resizes), "asset_arrays": 0}
    result["total"] = {name: sum(component[name] for component in result.values())
                       for name in ("values", "asset_arrays")}
    return result
