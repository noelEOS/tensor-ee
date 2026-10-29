"""Fixed crop geometry and source state names for the Prithvi flood model."""
from __future__ import annotations

from dataclasses import dataclass
import numbers


@dataclass(frozen=True)
class PrithviConfig:
    """One 224-pixel crop, six input bands, patch 16, four upsamplings, two logits."""

    width: int = 768
    heads: int = 12
    depth: int = 12
    decoder: int = 256

    def __post_init__(self):
        for name in ("width", "heads", "depth", "decoder"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")


PRITHVI = PrithviConfig()


def parameter_shapes(config=PRITHVI):
    """Inference state shapes, following the source module hierarchy; no auxiliary head."""
    dim = config.width
    shapes = {
        "backbone.patch_embed.proj.weight": (dim, 6, 1, 16, 16),
        "backbone.patch_embed.proj.bias": (dim,),
        "backbone.pos_embed": (1, 197, dim),
        "backbone.cls_token": (1, 1, dim),
    }

    def affine(name, channels):
        shapes[name + ".weight"] = (channels,)
        shapes[name + ".bias"] = (channels,)

    for index in range(config.depth):
        prefix = f"backbone.blocks.{index}"
        affine(prefix + ".norm1", dim)
        affine(prefix + ".norm2", dim)
        projections = (
            ("attn.qkv", dim, 3 * dim),
            ("attn.proj", dim, dim),
            ("mlp.fc1", dim, 4 * dim),
            ("mlp.fc2", 4 * dim, dim),
        )
        for name, cin, cout in projections:
            shapes[f"{prefix}.{name}.weight"] = (cout, cin)
            shapes[f"{prefix}.{name}.bias"] = (cout,)
    affine("backbone.norm", dim)
    for prefix in ("neck.fpn1", "neck.fpn2"):
        for index in (0, 3):
            shapes[f"{prefix}.{index}.weight"] = (dim, dim, 2, 2)
            shapes[f"{prefix}.{index}.bias"] = (dim,)
        affine(prefix + ".1.ln", dim)
    shapes["decode_head.convs.0.conv.weight"] = (config.decoder, dim, 3, 3)
    batch_norm = "decode_head.convs.0.bn"
    affine(batch_norm, config.decoder)
    shapes[batch_norm + ".running_mean"] = (config.decoder,)
    shapes[batch_norm + ".running_var"] = (config.decoder,)
    shapes["decode_head.conv_seg.weight"] = (2, config.decoder, 1, 1)
    shapes["decode_head.conv_seg.bias"] = (2,)
    return shapes


def stage_names(config=PRITHVI):
    """Saved boundaries in execution order; encoder values are spatial/class pairs."""
    return (
        "input", "patch", *(f"block{index}" for index in range(config.depth)),
        "norm", "upsample28", "upsample56", "upsample112", "upsample224", "logits",
    )
