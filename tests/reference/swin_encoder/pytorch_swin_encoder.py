"""Phase 1: PyTorch/timm reference for the complete Swin-T encoder from stage-1 tokens.

The trusted implementation is timm's own ``SwinTransformer`` with Swin-T's
configuration (embed_dim 96, depths (2, 2, 6, 2), num_heads (3, 6, 12, 24),
window 7, MLP ratio 4).  The patch embedding is constructed but not used: the
encoder's input is a stage-1 token raster, and its output is timm's
``forward_features`` minus the patch embedding:

    model.norm(model.layers(tokens))

    layers[0]  stage 1   dim  96,  3 heads, MLP  384, 2 blocks (W-MSA, SW-MSA)
    layers[1]  PatchMerging( 96 ->  192): LayerNorm( 384), Linear( 384 ->  192, no bias)
               stage 2   dim 192,  6 heads, MLP  768, 2 blocks
    layers[2]  PatchMerging(192 ->  384): LayerNorm( 768), Linear( 768 ->  384, no bias)
               stage 3   dim 384, 12 heads, MLP 1536, 6 blocks (W, SW, W, SW, W, SW)
    layers[3]  PatchMerging(384 ->  768): LayerNorm(1536), Linear(1536 ->  768, no bias)
               stage 4   dim 768, 24 heads, MLP 3072, 2 blocks
    norm       LayerNorm(768)

timm suppresses the shift of a block whose stage resolution is not larger than
the window (``_calc_window_shift``: resolution <= 7 -> shift 0, window =
resolution).  With 7 x 7-token windows that is a stage with one window per
side; such blocks are W-MSA here too (``block_shifts``).  A stage with one
window along only one axis would shift along the other axis only; that mixed
case is rejected (``check_resolution``).

``encoder_reference`` runs the encoder step by step with timm's own modules
(Milestone 2's explicit W-MSA forward, Milestone 3's explicit SW-MSA forward,
Milestone 4's explicit Patch Merging) and raises unless every stage output is
bit-identical to timm's ``layers[s]`` and the final output to
``model.norm(model.layers(tokens))``, so every compared tensor is timm's own.

Tensor names (window layouts as in Milestones 2-4; window (I, J) of a stage
is index I * kw + J of that stage's own window grid):
  x                     stage-1 input [nW1, 49, 96]
  s{s}b{b}_<name>       block b of stage s: Milestone 2 names (W-MSA) or
                        Milestone 3 names (SW-MSA, shifted tensors in the
                        shifted layout)
  pm{n}_x0 .. _x3       Patch Merging into stage n + 1: the four gathered groups [., 49, C]
  pm{n}_cat, _norm      their concatenation and LayerNorm(4C) [., 49, 4C]
  pm{n}_out             reduction [., 49, 2C]
  s{s}_out              stage s output (= its last block's out)
  norm                  final LayerNorm(768): the normalized encoder output [nW4, 49, 768]

Parameters: Milestone 4's groups keep their values (seeds 0-4), so the first
two stages are Milestone 4's hierarchy bit for bit; stages 3-4, Patch Merging 2-3
and the final norm use the same deliberately non-trivial spec, generic in width.

Moved from ``swin_encoder/reference/pytorch_swin_encoder.py`` at L06b (test support). Its command-line writer of the
research data files was archived with the research code.
"""

from __future__ import annotations

from tensor_ee.models.swin_t224.configuration import (stage_windows, stage_shifted, block_shifts,  # noqa: F401
                                                      check_windows)


import numpy as np
import torch

from reference import synthetic
from reference.swin_block import pytorch_swin_block as m2
from reference.swin_block.pytorch_swin_block import (
    LN_EPS, MLP_RATIO, WINDOW, X_SCALE, X_SEED, from_window_layout, to_window_layout)
from reference.swin_hierarchy import pytorch_hierarchy as m4
from reference.swin_hierarchy.pytorch_hierarchy import (
    block_parameter_spec, explicit_patch_merge, generate_parameters, patch_merge_parameter_spec)
from reference.swin_pair import pytorch_shifted_swin as m3

SHIFT = WINDOW // 2
EMBED_DIM = 96
DEPTHS = (2, 2, 6, 2)
NUM_HEADS = (3, 6, 12, 24)
STAGE_DIMS = (96, 192, 384, 768)
STAGES = len(DEPTHS)
NORM_DIM = STAGE_DIMS[-1]

def check_resolution(height: int, width: int):
    """Moved from swin_encoder/earth_engine/configuration.py (archived) at L06b."""
    if height % WINDOW or width % WINDOW:
        raise ValueError("stage-1 extents must be multiples of the 7-token window")
    check_windows(height // WINDOW, width // WINDOW)


def _forward_groups() -> tuple:
    groups = []
    for s, depth in enumerate(DEPTHS, 1):
        if s > 1:
            groups.append(f"pm{s - 1}")
        groups += [f"s{s}b{b}" for b in range(depth)]
    return tuple(groups + ["norm"])


GROUPS = _forward_groups()                    # forward order
# Milestone 4's groups keep their seeds (and values); the rest are new.
M4_GROUPS = {"s1b0": "s1a", "s1b1": "s1b", "pm1": "pm", "s2b0": "s2a", "s2b1": "s2b"}
SEEDS = {"s1b0": 0, "s1b1": 1, "s2b0": 2, "s2b1": 3, "pm1": 4, "pm2": 5,
         **{f"s3b{b}": 6 + b for b in range(6)}, "pm3": 12, "s4b0": 13, "s4b1": 14, "norm": 15}
NORM_SPEC = {"weight": (0.2, 1.0), "bias": (0.2, 0.0)}
# Generated stage inputs (isolated stage / merge / norm checks): seed per width.
INPUT_SEEDS = {1: X_SEED, 2: 12, 3: 13, 4: 14}
INPUT_SCALE = X_SCALE
MERGE_TENSORS = ("x0", "x1", "x2", "x3", "cat", "norm", "out")


def stage_of(group: str) -> int:
    return int(group[1])


def timm_prefix(group: str) -> str:
    """Full SwinTransformer state_dict prefix of a parameter group."""
    if group == "norm":
        return "norm."
    if group.startswith("pm"):
        return f"layers.{int(group[2:])}.downsample."
    s, b = group[1:].split("b")
    return f"layers.{int(s) - 1}.blocks.{int(b)}."


TIMM_PREFIX = {group: timm_prefix(group) for group in GROUPS}


# ---------------------------------------------------------------------------
# Resolutions and shifts
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
def _template(model, group: str) -> dict:
    prefix = TIMM_PREFIX[group]
    return {k[len(prefix):]: v for k, v in model.state_dict().items()
            if k.startswith(prefix) and not k.endswith(("relative_position_index", "attn_mask"))}


_PARAMETERS: dict = {}


def encoder_parameters() -> dict[str, dict[str, np.ndarray]]:
    """{group: {group-local timm name: float32}} for every encoder group (memoized)."""
    if not _PARAMETERS:
        model = make_model((8 * WINDOW, 8 * WINDOW))
        m4_parameters = m4.hierarchy_parameters()
        parameters = {group: m4_parameters[old] for group, old in M4_GROUPS.items()}
        for group in GROUPS:
            if group in parameters:
                continue
            template = _template(model, group)
            if group == "norm":
                spec = NORM_SPEC
            elif group.startswith("pm"):
                spec = patch_merge_parameter_spec(STAGE_DIMS[int(group[2:]) - 1])
            else:
                spec = block_parameter_spec(STAGE_DIMS[stage_of(group) - 1])
            parameters[group] = generate_parameters(template, spec, SEEDS[group])
        _PARAMETERS.update({group: parameters[group] for group in GROUPS})
    return _PARAMETERS


def timm_state(parameters: dict) -> dict[str, np.ndarray]:
    """Full SwinTransformer state_dict names -> arrays (also the asset keys)."""
    return {TIMM_PREFIX[group] + name: value for group in GROUPS
            for name, value in parameters[group].items()}


def parameter_count(parameters: dict, groups=GROUPS) -> int:
    return int(sum(v.size for g in groups for v in parameters[g].values()))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def _configure(block, shift: int):
    block.attn.fused_attn = False  # pin the explicit (original Swin) path
    if block.shift_size != (shift, shift) or block.window_size != (WINDOW, WINDOW):
        raise ValueError(f"expected shift {shift}, window 7; got {block.shift_size}, "
                         f"{block.window_size}")
    if not isinstance(block.mlp.act, torch.nn.GELU) or block.mlp.act.approximate != "none":
        raise ValueError("expected exact (erf) GELU")
    if block.norm1.eps != LN_EPS or block.norm2.eps != LN_EPS:
        raise ValueError("expected LayerNorm eps 1e-5")


def _refresh_masks(blocks, dtype):
    for block in blocks:
        if block.shift_size != (0, 0):
            block.attn_mask = block.get_attn_mask(device=torch.device("cpu"), dtype=dtype)


def make_model(resolution, parameters: dict | None = None, dtype=torch.float32, in_chans: int = 3):
    """timm SwinTransformer (Swin-T) for a stage-1 token raster of `resolution`.
    in_chans sets the (unloaded) patch embedding's input channels (Milestone 6: 2)."""
    from timm.models.swin_transformer import PatchMerging, SwinTransformer
    height, width = resolution
    check_resolution(height, width)
    model = SwinTransformer(img_size=(4 * height, 4 * width), patch_size=4, in_chans=in_chans,
                            num_classes=0, embed_dim=EMBED_DIM, depths=DEPTHS,
                            num_heads=NUM_HEADS, window_size=WINDOW, mlp_ratio=MLP_RATIO,
                            qkv_bias=True)
    shifts = block_shifts(height // WINDOW, width // WINDOW)
    for s, stage in enumerate(model.layers):
        expected = torch.nn.Identity if s == 0 else PatchMerging
        if not isinstance(stage.downsample, expected):
            raise ValueError(f"stage {s + 1}: unexpected downsample {type(stage.downsample)}")
        if len(stage.blocks) != DEPTHS[s] or stage.blocks[0].attn.num_heads != NUM_HEADS[s]:
            raise ValueError(f"stage {s + 1}: unexpected depth or heads")
        for block, shift in zip(stage.blocks, shifts[s]):
            _configure(block, shift)
    if not isinstance(model.norm, torch.nn.LayerNorm) or model.norm.eps != LN_EPS:
        raise ValueError("expected a final LayerNorm(768), eps 1e-5")
    if parameters is not None:
        state = {k: torch.from_numpy(np.asarray(v)) for k, v in timm_state(parameters).items()}
        layers = {k[len("layers."):]: v for k, v in state.items() if k.startswith("layers.")}
        model.layers.load_state_dict(layers, strict=False)
        missing = set(model.layers.state_dict()) - set(layers)
        if any(not k.endswith(("relative_position_index", "attn_mask")) for k in missing):
            raise ValueError(f"parameters missing: {sorted(missing)[:5]}")
        model.norm.load_state_dict({k[len("norm."):]: v for k, v in state.items()
                                    if k.startswith("norm.")}, strict=True)
    model = model.to(dtype).eval()
    for stage in model.layers:
        _refresh_masks(stage.blocks, dtype)
    return model


def make_stage(s: int, rows: int, cols: int, parameters: dict | None = None,
               dtype=torch.float32):
    """timm SwinTransformerStage s (1-4) whose blocks run on rows x cols windows,
    constructed exactly as SwinTransformer constructs layers[s - 1] (stages > 1
    start with their Patch Merging, at twice the resolution)."""
    from timm.models.swin_transformer import SwinTransformerStage
    shifted = stage_shifted(rows, cols)
    scale = 2 if s > 1 else 1
    stage = SwinTransformerStage(
        dim=STAGE_DIMS[s - 2] if s > 1 else EMBED_DIM, out_dim=STAGE_DIMS[s - 1],
        input_resolution=(rows * WINDOW * scale, cols * WINDOW * scale), depth=DEPTHS[s - 1],
        downsample=s > 1, num_heads=NUM_HEADS[s - 1], window_size=WINDOW, mlp_ratio=MLP_RATIO,
        qkv_bias=True)
    for b, block in enumerate(stage.blocks):
        _configure(block, SHIFT if b % 2 and shifted else 0)
    if parameters is not None:
        state = {}
        for group in GROUPS:
            if group.startswith(f"s{s}b"):
                state.update({f"blocks.{group.split('b')[1]}.{k}": torch.from_numpy(np.asarray(v))
                              for k, v in parameters[group].items()})
            elif s > 1 and group == f"pm{s - 1}":
                state.update({f"downsample.{k}": torch.from_numpy(np.asarray(v))
                              for k, v in parameters[group].items()})
        stage.load_state_dict(state, strict=False)
        missing = set(stage.state_dict()) - set(state)
        if any(not k.endswith(("relative_position_index", "attn_mask")) for k in missing):
            raise ValueError(f"parameters missing: {sorted(missing)[:5]}")
    stage = stage.to(dtype).eval()
    _refresh_masks(stage.blocks, dtype)
    return stage


# ---------------------------------------------------------------------------
# Explicit forward
# ---------------------------------------------------------------------------
def _raster(windows: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return torch.from_numpy(from_window_layout(windows.numpy(), height, width))[None]


def _layout(raster: torch.Tensor) -> np.ndarray:
    return to_window_layout(raster[0].numpy())


def _wanted(names, key: str) -> bool:
    return names is None or key in names


def _block_forward(block, h: torch.Tensor) -> dict[str, torch.Tensor]:
    """Milestone 2's (shift 0) or Milestone 3's (shift 3) explicit forward."""
    if block.shift_size == (0, 0):
        return m2.explicit_forward(block, h)
    return m3.explicit_shifted_forward(block, h)


def run_blocks(blocks, h: torch.Tensor, prefix: str, names=None) -> tuple[torch.Tensor, dict]:
    """Blocks on raster h [1, H, W, C] step by step; every block output is
    checked bit for bit against timm's block(h).  Returns (out raster, tensors)."""
    _, height, width, _ = h.shape
    tensors = {}
    for b, block in enumerate(blocks):
        result = _block_forward(block, h)
        out = _raster(result["out"], height, width)
        with torch.no_grad():
            if not torch.equal(out, block(h)):
                raise AssertionError(f"explicit {prefix}b{b} differs from timm")
        for k, v in result.items():
            if _wanted(names, f"{prefix}b{b}_{k}"):
                tensors[f"{prefix}b{b}_{k}"] = v.numpy()
        h = out
    return h, tensors


def merge_forward(patch_merging, h: torch.Tensor, prefix: str, names=None):
    """Milestone 4's explicit Patch Merging (raises unless equal to timm's module)."""
    pm = explicit_patch_merge(patch_merging, h)
    tensors = {f"{prefix}_{k}": _layout(v) for k, v in pm.items()
               if _wanted(names, f"{prefix}_{k}")}
    return pm["out"], tensors


def encoder_reference(parameters, x: np.ndarray, dtype=torch.float32,
                      names=None) -> dict[str, np.ndarray]:
    """Every named intermediate of the encoder for the stage-1 token raster
    x [H, W, 96], computed in `dtype` (see the module docstring).

    Raises unless each block, each stage and the final output are
    bit-identical to timm's own modules.  names, when given, keeps only those
    tensors (every block is still run explicitly and checked).
    """
    height, width, _ = x.shape
    model = make_model((height, width), parameters, dtype)
    tokens = torch.from_numpy(np.ascontiguousarray(x))[None].to(dtype)
    result = {"x": to_window_layout(x)} if _wanted(names, "x") else {}
    h = tokens
    for s, stage in enumerate(model.layers, 1):
        stage_input = h
        if s > 1:
            h, tensors = merge_forward(stage.downsample, h, f"pm{s - 1}", names)
            result.update(tensors)
        h, tensors = run_blocks(stage.blocks, h, f"s{s}", names)
        result.update(tensors)
        with torch.no_grad():
            if not torch.equal(h, stage(stage_input)):
                raise AssertionError(f"explicit stage {s} differs from timm ({dtype})")
        if _wanted(names, f"s{s}_out"):
            result[f"s{s}_out"] = _layout(h)
    with torch.no_grad():
        final = model.norm(h)
        timm_final = model.norm(model.layers(tokens))
    if not torch.equal(final, timm_final):
        raise AssertionError(f"explicit encoder differs from model.norm(model.layers(x)) ({dtype})")
    if _wanted(names, "norm"):
        result["norm"] = _layout(final)
    return result


# ---------------------------------------------------------------------------
# Isolated components on generated inputs (Phases 3-8)
# ---------------------------------------------------------------------------
def stage_input(s: int, rows: int, cols: int) -> np.ndarray:
    """Generated tokens [7 rows, 7 cols, dim_s] at stage s's width (seed per
    width): exactly what EE generates on the server (window_layout_raster)."""
    return synthetic.generated_raster_numpy(rows * WINDOW, cols * WINDOW, STAGE_DIMS[s - 1],
                                         INPUT_SEEDS[s], INPUT_SCALE)


def stage_reference(parameters, s: int, x: np.ndarray, dtype=torch.float32, blocks: int | None = None,
                    names=None) -> dict[str, np.ndarray]:
    """The first `blocks` blocks of stage s on the stage-s token raster x [H, W, dim_s]:
    keys b{b}_<name>.  The full stage is also checked against timm's
    stage.blocks(x)."""
    height, width, _ = x.shape
    stage = make_stage(s, height // WINDOW, width // WINDOW, parameters, dtype)
    blocks = len(stage.blocks) if blocks is None else blocks
    h = torch.from_numpy(np.ascontiguousarray(x))[None].to(dtype)
    out, tensors = run_blocks(list(stage.blocks)[:blocks], h, "", names)
    if blocks == len(stage.blocks):
        with torch.no_grad():
            if not torch.equal(out, stage.blocks(h)):
                raise AssertionError(f"explicit stage {s} blocks differ from timm")
    return tensors


def merge_reference(parameters, n: int, x: np.ndarray, dtype=torch.float32) -> dict[str, np.ndarray]:
    """Patch Merging n (stage n -> n + 1) of the stage-n token raster x [H, W, C]:
    keys x0..x3, cat, norm, out in stage-(n + 1) window layout."""
    height, width, _ = x.shape
    stage = make_stage(n + 1, height // (2 * WINDOW), width // (2 * WINDOW), parameters, dtype)
    h = torch.from_numpy(np.ascontiguousarray(x))[None].to(dtype)
    _, tensors = merge_forward(stage.downsample, h, "pm", None)
    return {k[len("pm_"):]: v for k, v in tensors.items()}


def norm_reference(parameters, x: np.ndarray, dtype=torch.float32) -> np.ndarray:
    """timm's final LayerNorm(768) of tokens x [..., 768] (window layout in, out)."""
    norm = torch.nn.LayerNorm(NORM_DIM, eps=LN_EPS)
    norm.load_state_dict({k: torch.from_numpy(v) for k, v in parameters["norm"].items()})
    norm = norm.to(dtype).eval()
    with torch.no_grad():
        return norm(torch.from_numpy(np.ascontiguousarray(x)).to(dtype)).numpy()


def channel_token_ids(height: int, width: int, channels: int) -> np.ndarray:
    """[H, W, C] float32 unique (token, channel) IDs (y W + x) C + c, exact
    integers below 2^24 (layout tests at any width)."""
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    ids = (y * width + x)[..., None] * channels + np.arange(channels)
    if ids.max() >= 2 ** 24:
        raise ValueError("IDs would not be exact in float32")
    return ids.astype(np.float32)


def merged_channel_ids(height: int, width: int, channels: int) -> np.ndarray:
    """timm's PatchMerging gather (its own module code) of channel_token_ids:
    [nW', 49, 4C] in the merged stage's window layout."""
    from timm.models.swin_transformer import PatchMerging
    ids = torch.from_numpy(channel_token_ids(height, width, channels))[None]
    return _layout(m4.timm_gather(PatchMerging(dim=channels), ids))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def raster_input(height: int, width: int) -> np.ndarray:
    """Milestone 2's generated stage-1 tokens [H, W, 96] (not periodic)."""
    return m2.raster_input(height, width)


_REFERENCES: dict = {}


def references(rows: int, cols: int, names=None, dtypes=("torch32", "float64")) -> dict:
    """{"torch32" / "float64": tensors} of the encoder on the generated input of
    rows x cols stage-1 windows (memoized per name set)."""
    key = (rows, cols, None if names is None else tuple(sorted(names)), tuple(dtypes))
    if key not in _REFERENCES:
        parameters = encoder_parameters()
        x = raster_input(rows * WINDOW, cols * WINDOW)
        torch_dtype = {"torch32": torch.float32, "float64": torch.float64}
        _REFERENCES[key] = {d: encoder_reference(parameters, x, torch_dtype[d], names)
                            for d in dtypes}
    return _REFERENCES[key]
