"""Phases 1-2: PyTorch/timm reference for Stage 1 -> Patch Merging -> Stage 2 of Swin-T.

The trusted implementation is timm's own ``SwinTransformer`` (Swin-T widths:
embed_dim 96, num_heads (3, 6), window 7, MLP ratio 4) restricted to its first
two stages, ``model.layers[0]`` and ``model.layers[1]``, so the stage-2
construction (Patch Merging, widths, heads, shifts) is timm's, not ours:

    layers[0]  SwinTransformerStage(96 -> 96, depth 2, downsample=False, 3 heads)
               block 0 W-MSA (shift 0), block 1 SW-MSA (shift 3)
    layers[1]  SwinTransformerStage(96 -> 192, depth 2, downsample=True, 6 heads)
               downsample = PatchMerging(96 -> 192):
                   x      = pad(x) to even H, W                  (zeros, bottom/right)
                   merged = x.reshape(B, H/2, 2, W/2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
                   norm   = LayerNorm(384)(merged)
                   out    = Linear(384, 192, bias=False)(norm)
               block 0 W-MSA, block 1 SW-MSA at dim 192, MLP 768

The patch embedding is not used: the hierarchy's input is a stage-1 token
raster (Milestones 2-3's generated tokens).  All blocks use timm's non-fused
attention path (the original Swin order of operations).

``hierarchy_reference`` runs the hierarchy step by step with timm's own
modules (Milestone 2's explicit W-MSA forward, Milestone 3's explicit SW-MSA
forward, ``explicit_patch_merge``) and raises unless every stage output is
bit-identical to timm's: ``layers[0](x)``, ``layers[1].downsample``,
``layers[1]`` and ``model.layers(x)``.  The patch-merge gather is timm's own
module code (the module with its norm and reduction replaced by identities),
not a re-implementation.

Tensor names and layouts (comparisons key on these):
  x                        stage-1 window layout [nW1, 49, 96]
  s1a_<name>, s1b_<name>   Milestone 2 / 3 names, stage-1 window layouts
                           (s1b's shifted tensors in the shifted layout)
  pm_x0 .. pm_x3           the four gathered token groups, stage-2 window layout [nW2, 49, 96]
  pm_cat                   their concatenation [nW2, 49, 384]
  pm_norm, pm_out          LayerNorm(384) [nW2, 49, 384], reduction [nW2, 49, 192]
  s2a_<name>, s2b_<name>   stage-2 blocks, stage-2 window layouts (s2b shifted as in M3)
Window (I, J) is index I * kw + J; row 7a + b is token (7I + a, 7J + b) of
that stage's own token raster.

Moved from ``swin_hierarchy/reference/pytorch_hierarchy.py`` at L06b (test support). Its command-line writer of the
research data files was archived with the research code.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch

from reference.swin_block import pytorch_swin_block as m2
from reference.swin_block.pytorch_swin_block import (
    LN_EPS, MLP_RATIO, WINDOW, from_window_layout, raster_input, to_window_layout)
from reference.swin_pair import pytorch_shifted_swin as m3

SHIFT = WINDOW // 2
EMBED_DIM, DEPTHS, NUM_HEADS = 96, (2, 2), (3, 6)
STAGE_DIMS = {1: 96, 2: 192}
GROUPS = ("s1a", "s1b", "pm", "s2a", "s2b")
# Stage-1 blocks keep Milestone 3's parameters (seeds 0 and 1).
SEEDS = {"s1a": 0, "s1b": 1, "s2a": 2, "s2b": 3, "pm": 4}
# timm state_dict prefix of each group inside SwinTransformer.
TIMM_PREFIX = {"s1a": "layers.0.blocks.0.", "s1b": "layers.0.blocks.1.",
               "pm": "layers.1.downsample.", "s2a": "layers.1.blocks.0.",
               "s2b": "layers.1.blocks.1."}


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
def block_parameter_spec(dim: int, hidden: int | None = None) -> dict:
    """Milestone 2's deliberately non-trivial spec, generic in the width.

    At dim 96 it is swin_block's PARAMETER_SPEC exactly.  The qkv scale falls
    as 1/sqrt(dim) so attention logits keep a standard deviation of about 2 at
    any width (a peaked, not saturated, softmax).
    """
    hidden = hidden or dim * MLP_RATIO
    return {
        "norm1.weight": (0.2, 1.0),
        "norm1.bias": (0.2, 0.0),
        "attn.relative_position_bias_table": (1.0, 0.0),
        "attn.qkv.weight": (0.15 * math.sqrt(96 / dim), 0.0),
        "attn.qkv.bias": (0.1, 0.0),
        "attn.proj.weight": (1 / math.sqrt(dim), 0.0),
        "attn.proj.bias": (0.05, 0.0),
        "norm2.weight": (0.2, 1.0),
        "norm2.bias": (0.2, 0.0),
        "mlp.fc1.weight": (1 / math.sqrt(dim), 0.0),
        "mlp.fc1.bias": (0.1, 0.0),
        "mlp.fc2.weight": (1 / math.sqrt(hidden), 0.0),
        "mlp.fc2.bias": (0.05, 0.0),
    }


def patch_merge_parameter_spec(dim: int) -> dict:
    """LayerNorm(4 dim) gamma/beta deliberately not 1/0; reduction ~ 1/sqrt(4 dim)."""
    return {"norm.weight": (0.2, 1.0), "norm.bias": (0.2, 0.0),
            "reduction.weight": (1 / math.sqrt(4 * dim), 0.0)}


def generate_parameters(template: dict, spec: dict, seed: int) -> dict[str, np.ndarray]:
    """offset + std * N(0, 1), tensor i from torch.Generator(seed * 1000 + 100 + i)
    (Milestone 2's scheme), shapes from a timm state_dict."""
    if set(template) != set(spec):
        raise ValueError(f"unexpected timm parameters: {sorted(template)}")
    parameters = {}
    for index, (name, (std, offset)) in enumerate(spec.items()):
        generator = torch.Generator().manual_seed(seed * 1000 + 100 + index)
        values = torch.randn(tuple(template[name].shape), generator=generator, dtype=torch.float32)
        parameters[name] = (values * std + offset).numpy()
    return parameters


def _group_template(model, group: str) -> dict:
    prefix = TIMM_PREFIX[group]
    return {k[len(prefix):]: v for k, v in model.state_dict().items() if k.startswith(prefix)}


def hierarchy_parameters() -> dict[str, dict[str, np.ndarray]]:
    """{group: {block-local timm name: float32}} for s1a, s1b, pm, s2a, s2b."""
    model = make_model((28, 28))
    parameters = {"s1a": m2.make_parameters(SEEDS["s1a"]), "s1b": m2.make_parameters(SEEDS["s1b"])}
    for group in ("s2a", "s2b"):
        parameters[group] = generate_parameters(_group_template(model, group),
                                                block_parameter_spec(STAGE_DIMS[2]), SEEDS[group])
    parameters["pm"] = generate_parameters(_group_template(model, "pm"),
                                           patch_merge_parameter_spec(STAGE_DIMS[1]), SEEDS["pm"])
    return {group: parameters[group] for group in GROUPS}


def timm_state(parameters: dict) -> dict[str, np.ndarray]:
    """Full SwinTransformer state_dict names -> arrays (also the asset keys)."""
    return {TIMM_PREFIX[group] + name: value for group in GROUPS
            for name, value in parameters[group].items()}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def check_resolution(height: int, width: int, window: int = WINDOW):
    """Stage-1 token raster accepted by the window-layout hierarchy (window 7 by default)."""
    for extent in (height, width):
        if extent % (2 * window):
            raise ValueError(f"stage-1 extents must be an even number of {window}-token windows "
                             "(Patch Merging maps 2 x 2 stage-1 windows to one stage-2 window)")
        if extent < 4 * window:
            raise ValueError("stage 2 needs more than one window per side, or timm uses shift 0")


def _configure(block, shift: int, window: int = WINDOW):
    block.attn.fused_attn = False  # pin the explicit (original Swin) path
    if block.shift_size != (shift, shift) or block.window_size != (window, window):
        raise ValueError(f"expected window {window} and shift {shift}, got {block.window_size} and "
                         f"{block.shift_size}")
    if not isinstance(block.mlp.act, torch.nn.GELU) or block.mlp.act.approximate != "none":
        raise ValueError("expected exact (erf) GELU")
    if block.norm1.eps != LN_EPS or block.norm2.eps != LN_EPS:
        raise ValueError("expected LayerNorm eps 1e-5")


def make_model(resolution, parameters: dict | None = None, dtype=torch.float32, *, window: int = WINDOW):
    """timm SwinTransformer with Swin-T's first two stages for a stage-1 token
    raster of `resolution` (the patch embedding is constructed but not used); window 7 by
    default, shift window // 2 (timm's stage rule)."""
    from timm.models.swin_transformer import PatchMerging, SwinTransformer
    height, width = resolution
    check_resolution(height, width, window)
    model = SwinTransformer(img_size=(4 * height, 4 * width), patch_size=4, in_chans=3,
                            num_classes=0, embed_dim=EMBED_DIM, depths=DEPTHS,
                            num_heads=NUM_HEADS, window_size=window, mlp_ratio=MLP_RATIO,
                            qkv_bias=True)
    stage1, stage2 = model.layers
    if not isinstance(stage1.downsample, torch.nn.Identity):
        raise ValueError("stage 1 must not downsample")
    if not isinstance(stage2.downsample, PatchMerging):
        raise ValueError("stage 2 must start with Patch Merging")
    for stage in (stage1, stage2):
        _configure(stage.blocks[0], 0, window)
        _configure(stage.blocks[1], window // 2, window)
    if parameters is not None:
        state = {k[len("layers."):]: torch.from_numpy(np.asarray(v))
                 for k, v in timm_state(parameters).items()}
        model.layers.load_state_dict(state, strict=True)
    model = model.to(dtype).eval()
    for stage in model.layers:
        stage.blocks[1].attn_mask = stage.blocks[1].get_attn_mask(device=torch.device("cpu"),
                                                                  dtype=dtype)
    return model


# ---------------------------------------------------------------------------
# Patch Merging
# ---------------------------------------------------------------------------
def timm_gather(patch_merging, x: torch.Tensor) -> torch.Tensor:
    """timm PatchMerging.forward without its norm and reduction: timm's own
    padding, reshape, permute and flatten.  x [B, H, W, C] -> [B, H', W', 4C]."""
    gather = copy.deepcopy(patch_merging)
    gather.norm = torch.nn.Identity()
    gather.reduction = torch.nn.Identity()
    with torch.no_grad():
        return gather(x)


def explicit_patch_merge(patch_merging, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Rasters [B, H/2, W/2, *]: the four groups, their concatenation, LN, reduction."""
    channels = x.shape[-1]
    merged = timm_gather(patch_merging, x)
    with torch.no_grad():
        norm = patch_merging.norm(merged)
        out = patch_merging.reduction(norm)
        reference = patch_merging(x)
    if not torch.equal(out, reference):
        raise AssertionError("explicit patch merge differs from timm PatchMerging")
    result = {f"x{g}": merged[..., g * channels:(g + 1) * channels] for g in range(4)}
    result.update(cat=merged, norm=norm, out=out)
    return result


def patch_merge_semantics() -> dict:
    """Phase 2: what timm's PatchMerging does, established by experiment.

    Every claim below is computed from timm's module on unique token IDs,
    not read from a diagram.
    """
    from timm.models.swin_transformer import PatchMerging
    model = make_model((28, 28))
    pm = model.layers[1].downsample
    out = {"timm_class": f"{type(pm).__module__}.{type(pm).__name__}",
           "norm": {"type": type(pm.norm).__name__, "normalized_shape": list(pm.norm.normalized_shape),
                    "eps": pm.norm.eps, "elementwise_affine": pm.norm.elementwise_affine},
           "reduction": {"type": type(pm.reduction).__name__, "in_features": pm.reduction.in_features,
                         "out_features": pm.reduction.out_features,
                         "bias": pm.reduction.bias is not None}}
    # 1-2, 7: unique IDs (id = y W + x, y, x) through timm's gather.
    height, width = 6, 8
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    ids = np.stack([y * width + x, y, x], -1).astype(np.float32)
    merged = timm_gather(PatchMerging(dim=3), torch.from_numpy(ids)[None])[0].numpy()
    groups = []
    for g in range(4):
        part = merged[..., 3 * g:3 * g + 3]
        i, j = np.meshgrid(np.arange(height // 2), np.arange(width // 2), indexing="ij")
        dy, dx = part[..., 1] - 2 * i, part[..., 2] - 2 * j
        if not (np.unique(dy).size == 1 and np.unique(dx).size == 1):
            raise AssertionError("group is not one fixed 2x2 offset")
        if not np.array_equal(part[..., 0], part[..., 1] * width + part[..., 2]):
            raise AssertionError("channels within a group are not the source token's, in order")
        groups.append({"group": g, "channels": [3 * g, 3 * g + 2], "dy": int(dy.flat[0]),
                       "dx": int(dx.flat[0])})
    out["groups"] = groups
    out["groups_formula"] = ("merged[i, j] = cat(x[2i, 2j], x[2i+1, 2j], x[2i, 2j+1], "
                             "x[2i+1, 2j+1]): x0 = x[0::2, 0::2], x1 = x[1::2, 0::2], "
                             "x2 = x[0::2, 1::2], x3 = x[1::2, 1::2]")
    out["channel_order_within_group"] = "source token channels 0..C-1 in order (verified on IDs)"
    out["output_spatial_order"] = "merged token (i, j) covers source tokens (2i + dy, 2j + dx), row-major"
    # 6: padding, on an odd raster with IDs + 1 so that padded zeros are visible.
    odd = timm_gather(PatchMerging(dim=3), torch.from_numpy(ids[:5, :7] + 1)[None])[0].numpy()
    zero = (odd.reshape(3, 4, 4, 3) == 0).all(-1)               # [i, j, group]
    out["padding"] = {
        "input": [5, 7], "output": list(odd.shape[:2]),
        "zero_groups_last_row": sorted({g for g in range(4) if zero[2, :, g].all()}),
        "zero_groups_last_col": sorted({g for g in range(4) if zero[:, 3, g].all()}),
        "rule": "zeros appended at the bottom row and right column (F.pad (0, 0, 0, W % 2, 0, H % 2))",
        "used_here": "no: every stage-1 raster here has an even number of windows per side"}
    # 3-5: order of operations, LayerNorm width, bias: the explicit composition
    # reduction(norm(gather(x))) equals the module bit for bit (float32 and float64).
    rng = np.random.default_rng(0)
    probe = torch.from_numpy(rng.standard_normal((1, 4, 6, 96)).astype(np.float32))
    for dtype in (torch.float32, torch.float64):
        module = copy.deepcopy(pm).to(dtype)
        explicit_patch_merge(module, probe.to(dtype))   # raises if not bit-identical
    out["order"] = "gather -> LayerNorm(4C) -> Linear(4C, 2C, bias=False); no activation, no residual"
    out["explicit_composition_bit_identical"] = {"float32": True, "float64": True}
    out["float32_rounding_points"] = ("gather: none (values moved); LayerNorm: one float32 output "
                                      "(PyTorch accumulates in float32 inside); Linear: one float32 "
                                      "output (float32 GEMM accumulation)")
    return out


# ---------------------------------------------------------------------------
# Explicit forward of the hierarchy
# ---------------------------------------------------------------------------
def _raster(windows: torch.Tensor, height: int, width: int, window: int = WINDOW) -> torch.Tensor:
    return torch.from_numpy(from_window_layout(windows.numpy(), height, width, window))[None]


def _layout(raster: torch.Tensor, window: int = WINDOW) -> np.ndarray:
    return to_window_layout(raster[0].numpy(), window)


def hierarchy_reference(parameters, x: np.ndarray, dtype=torch.float32,
                        names=None, *, window: int = WINDOW) -> dict[str, np.ndarray]:
    """Every intermediate of Stage-1 pair -> Patch Merging -> Stage-2 pair for the
    stage-1 token raster x [H, W, 96], computed in `dtype` (see module docstring).

    Raises unless each stage output is bit-identical to timm's own modules.
    names, when given, keeps only those tensors.
    """
    height, width, _ = x.shape
    model = make_model((height, width), parameters, dtype, window=window)
    stage1, stage2 = model.layers
    tokens = torch.from_numpy(np.ascontiguousarray(x))[None].to(dtype)
    a1 = m2.explicit_forward(stage1.blocks[0], tokens)
    b1 = m3.explicit_shifted_forward(stage1.blocks[1], _raster(a1["out"], height, width, window))
    s1_out = _raster(b1["out"], height, width, window)
    pm = explicit_patch_merge(stage2.downsample, s1_out)
    a2 = m2.explicit_forward(stage2.blocks[0], pm["out"])
    b2 = m3.explicit_shifted_forward(stage2.blocks[1], _raster(a2["out"], height // 2, width // 2, window))
    out = _raster(b2["out"], height // 2, width // 2, window)
    with torch.no_grad():
        timm_s1 = stage1(tokens)
        timm_pm = stage2.downsample(timm_s1)
        timm_s2 = stage2(timm_s1)
        timm_all = model.layers(tokens)
    for label, mine, theirs in (("stage 1", s1_out, timm_s1), ("patch merging", pm["out"], timm_pm),
                                ("stage 2", out, timm_s2), ("model.layers", out, timm_all)):
        if not torch.equal(mine, theirs):
            raise AssertionError(f"explicit {label} differs from timm ({dtype})")
    result = {"x": to_window_layout(x, window)}
    result.update({f"s1a_{k}": v.numpy() for k, v in a1.items()})
    result.update({f"s1b_{k}": v.numpy() for k, v in b1.items()})
    result.update({f"pm_{k}": _layout(v, window) for k, v in pm.items()})
    result.update({f"s2a_{k}": v.numpy() for k, v in a2.items()})
    result.update({f"s2b_{k}": v.numpy() for k, v in b2.items()})
    if names is not None:
        result = {name: result[name] for name in names}
    return result


def gather_windows(windows: np.ndarray, rows: int, cols: int, window: int = WINDOW) -> np.ndarray:
    """Exact patch-merge gather in window layout via the fine raster (NumPy):
    stage-1 windows [rows*cols, w*w, C] -> stage-2 windows [rows*cols/4, w*w, 4C] (w = 7 by
    default), groups in timm's order x[0::2, 0::2], x[1::2, 0::2], x[0::2, 1::2], x[1::2, 1::2]."""
    raster = from_window_layout(windows, rows * window, cols * window, window)
    groups = [raster[dy::2, dx::2] for dy, dx in ((0, 0), (1, 0), (0, 1), (1, 1))]
    return to_window_layout(np.concatenate(groups, axis=-1), window)


def stage2_token_ids(rows: int, cols: int, window: int = WINDOW) -> np.ndarray:
    """Stage-1 token IDs (y W + x, y, x) gathered by timm's PatchMerging: [nW2, w*w, 12]."""
    from timm.models.swin_transformer import PatchMerging
    ids = m3.token_ids(rows, cols, window)
    return _layout(timm_gather(PatchMerging(dim=3), torch.from_numpy(ids)[None]), window)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_REFERENCES: dict = {}


def references(rows: int, cols: int, names=None, dtypes=("torch32", "float64")) -> dict:
    """{"torch32" / "float64": tensors} for the generated input of rows x cols
    stage-1 windows (memoized per name set)."""
    key = (rows, cols, None if names is None else tuple(names), tuple(dtypes))
    if key not in _REFERENCES:
        parameters = hierarchy_parameters()
        x = raster_input(rows * WINDOW, cols * WINDOW)
        torch_dtype = {"torch32": torch.float32, "float64": torch.float64}
        _REFERENCES[key] = {d: hierarchy_reference(parameters, x, torch_dtype[d], names)
                            for d in dtypes}
    return _REFERENCES[key]
