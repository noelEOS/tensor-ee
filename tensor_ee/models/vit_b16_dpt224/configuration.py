"""SMP 0.5.0 DPT's fixed 224/16 geometry, cat readout and state names."""
from __future__ import annotations

from dataclasses import dataclass
import numbers

from ..vit_b16_encoder224.configuration import VIT_B16, ViTConfig, parameter_shapes as encoder_shapes


@dataclass(frozen=True)
class DPTConfig:
    """Default ViT-B/16 DPT; smaller widths are numerical twins, not trained models."""

    encoder: ViTConfig = VIT_B16
    intermediate: tuple[int, ...] = (256, 512, 1024, 1024)
    fusion: int = 256
    readout: str = "cat"

    def __post_init__(self):
        if not isinstance(self.encoder, ViTConfig) or self.encoder.depth < 4:
            raise ValueError("DPT requires a ViT encoder with at least four blocks")
        if self.readout != "cat":
            raise ValueError('only readout="cat" is supported')
        widths = tuple(self.intermediate)
        if len(widths) != 4:
            raise ValueError("DPT requires four intermediate widths")
        for value in (*widths, self.fusion):
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
                raise ValueError("decoder widths must be positive integers")
        object.__setattr__(self, "intermediate", tuple(int(n) for n in widths))
        object.__setattr__(self, "fusion", int(self.fusion))

    @property
    def indices(self):
        """SMP's uniform four-output sampling, zero-based and without final norm."""
        return tuple(self.encoder.depth * i // 4 - 1 for i in range(1, 5))


DPT_B16 = DPTConfig()
REDUCED = DPTConfig(ViTConfig(width=64, heads=1, depth=4), (4, 8, 16, 16), 4)


def parameter_shapes(config=DPT_B16):
    """Native SMP inference shapes; BN counters and the unused classifier are omitted."""
    shapes = encoder_shapes(config.encoder)
    dim, fusion = config.encoder.width, config.fusion

    def conv(prefix, cin, cout, kernel, bias=True):
        shapes[prefix + ".weight"] = (cout, cin, kernel, kernel)
        if bias:
            shapes[prefix + ".bias"] = (cout,)

    def bn(prefix):
        for suffix in ("weight", "bias", "running_mean", "running_var"):
            shapes[prefix + "." + suffix] = (fusion,)

    for i, mid in enumerate(config.intermediate):
        readout = f"decoder.projection_blocks.{i}.project.0"
        shapes[readout + ".weight"] = (dim, 2 * dim)
        shapes[readout + ".bias"] = (dim,)
        prefix = f"decoder.reassemble_blocks.{i}"
        conv(prefix + ".project_to_out_channel", dim, mid, 1)
        if i < 2:
            factor = (4, 2)[i]
            shapes[prefix + ".upsample.weight"] = (mid, mid, factor, factor)
            shapes[prefix + ".upsample.bias"] = (mid,)
        elif i == 3:
            conv(prefix + ".upsample", mid, mid, 3)
        conv(prefix + ".project_to_feature_dim", mid, fusion, 3, False)
    for i in range(4):
        prefix = f"decoder.fusion_blocks.{i}"
        for residual in (1, 2):
            block = prefix + f".residual_conv_block{residual}"
            for j in (1, 2):
                conv(block + f".conv_{j}", fusion, fusion, 3, False)
                bn(block + f".batch_norm_{j}")
        conv(prefix + ".project", fusion, fusion, 1)
    conv("segmentation_head.head.0", fusion, fusion, 3, False)
    bn("segmentation_head.head.1")
    conv("segmentation_head.head.4", fusion, 2, 1)
    return shapes


def boundary_spec(name, config=DPT_B16):
    """(token side, channels, window) for decoder boundaries, on the input's extent."""
    if name.startswith("readout"):
        return 14, config.encoder.width, 14
    if name.startswith("reassemble"):
        i = int(name[-1])
        return (56, 28, 14, 7)[i], config.fusion, (7, 7, 14, 7)[i]
    if name.startswith("fusion"):
        i = int(name[-1])
        return (14, 28, 56, 112)[i], config.fusion, (14, 7, 7, 7)[i]
    if name == "head":
        return 112, 2, 7
    if name == "logits":
        return 224, 2, 7
    raise ValueError(f"unknown decoder boundary {name!r}")
