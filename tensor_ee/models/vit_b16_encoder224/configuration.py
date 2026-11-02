"""Fixed six-band 224/16 geometry and SMP DPT encoder parameter names."""
from __future__ import annotations

from dataclasses import dataclass
import numbers

PREFIX = "encoder.model"
DPT_INDICES = (2, 5, 8, 11)
"""SMP 0.5.0 TimmViTEncoder's default four outputs (zero-based block indices)."""


@dataclass(frozen=True)
class ViTConfig:
    """ViT-B defaults; reduced twins retain head dimension 64 and the 14x14 grid."""

    width: int = 768
    heads: int = 12
    depth: int = 12

    def __post_init__(self):
        for name in ("width", "heads", "depth"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        if self.width != 64 * self.heads:
            raise ValueError("head dimension must be 64")
        if self.depth > 12:
            raise ValueError("depth must be at most 12")


VIT_B16 = ViTConfig()
REDUCED = ViTConfig(width=128, heads=2, depth=2)


def stage_names(config=VIT_B16):
    """Encoder boundaries in execution order; all but input are (spatial, class)."""
    return ("input", "patch", *(f"block{i}" for i in range(config.depth)), "norm")


def parameter_shapes(config=VIT_B16):
    """DPT's encoder.model.* inference parameters, excluding the unused classifier."""
    dim = config.width
    shapes = {
        f"{PREFIX}.patch_embed.proj.weight": (dim, 6, 16, 16),
        f"{PREFIX}.patch_embed.proj.bias": (dim,),
        f"{PREFIX}.cls_token": (1, 1, dim),
        f"{PREFIX}.pos_embed": (1, 197, dim),
    }

    def affine(name):
        shapes[name + ".weight"] = (dim,)
        shapes[name + ".bias"] = (dim,)

    for i in range(config.depth):
        prefix = f"{PREFIX}.blocks.{i}"
        affine(prefix + ".norm1")
        affine(prefix + ".norm2")
        for name, cin, cout in (("attn.qkv", dim, 3 * dim), ("attn.proj", dim, dim),
                                ("mlp.fc1", dim, 4 * dim), ("mlp.fc2", 4 * dim, dim)):
            shapes[f"{prefix}.{name}.weight"] = (cout, cin)
            shapes[f"{prefix}.{name}.bias"] = (cout,)
    affine(PREFIX + ".norm")
    return shapes
