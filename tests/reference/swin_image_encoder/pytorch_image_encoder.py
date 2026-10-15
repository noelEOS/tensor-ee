"""Phases 1-3: the PyTorch/timm reference from a Sentinel-1 VV/VH raster to encoder features.

    raw         VV, VH dB from EE (double)                                [H, W, 2] float64
    input       SegWater v3 preprocessing (tensor_ee.segwater_preprocessing), float32
                after every operation: cast -> q55s005 quantization -> (q - mean) / std
    patch_embed timm PatchEmbed: Conv2d(2, 96, kernel 4, stride 4, bias) -> NHWC -> LayerNorm(96)
    encoder     Milestone 5: model.norm(model.layers(tokens))

i.e. timm's ``SwinTransformer(in_chans=2, img_size=(H, W)).forward_features(input)``.
The preprocessing is a fixed, documented transform (the audited SegWater v3 recipe,
proven bit-exact between NumPy and EE in the SegWater work); there is no trained
checkpoint behind it.  The encoder parameters are Milestone 5's seeded set; the
patch embedding gets the same kind of deliberately non-trivial seeded parameters
(group "pe", seed 16).

``image_encoder_reference`` runs the patch embedding with timm's own submodules
(proj, then norm), feeds its tokens to Milestone 5's explicit encoder (every block
and stage checked against timm) and raises unless the final output is
bit-identical to ``model.forward_features(input)``.

The standalone patch embedding (``make_patch_embed``, ``patches``,
``explicit_patch_embed``, ``timm_pe_state``, ``check_patch_embed``) and the token-ID
helpers (``pixel_ids``, ``one_hot_extraction``, ``timm_patch_ids``) take the patch size,
the input channels and the embedding width (task P03), with today's 4, 2 and 96 as
defaults. Nothing here reads the library.

Tensor names (window layouts as in Milestones 2-5; window (I, J) is I * kw + J):
  raw          [H, W, 2] dB (float64)
  input        [H, W, 2] preprocessed model input (float32 values)
  pe_patches   [nW1, 49, 32] the 4 x 4 x 2 patch of each token, flattened (c, ky, kx)
               (Conv2d weight order); moved values only
  pe_proj      [nW1, 49, 96] Conv2d output (bias included), tokens in NHWC order
  x            [nW1, 49, 96] LayerNorm(96) of pe_proj: the stage-1 tokens
  ... and every Milestone 5 name (s{s}b{b}_*, pm{n}_*, s{s}_out, norm).

Moved from ``swin_image_encoder/reference/pytorch_image_encoder.py`` at L06b (test support). Its command-line writer of the
research data files was archived with the research code.
"""

from __future__ import annotations


import numpy as np
import torch

from reference.swin_block.pytorch_swin_block import LN_EPS, from_window_layout, to_window_layout
from reference.swin_encoder import pytorch_swin_encoder as m5
from reference.swin_hierarchy.pytorch_hierarchy import generate_parameters

PATCH = 4                                             # Swin-T patch (was swin_image_encoder.scene)
IN_CHANS = 2                                          # VV, VH
EMBED_DIM = m5.EMBED_DIM
PATCH_VALUES = IN_CHANS * PATCH * PATCH               # 32
PE_SEED = 16
# proj.weight is kept flattened [96, 32] in (c, ky, kx) order, the order of
# Conv2d.weight.reshape(96, -1) and of the EE patch columns; timm gets [96, 2, 4, 4].
PE_SPEC = {"proj.weight": (1 / np.sqrt(PATCH_VALUES), 0.0), "proj.bias": (0.1, 0.0),
           "norm.weight": (0.2, 1.0), "norm.bias": (0.2, 0.0)}
PE_SHAPES = {"proj.weight": (EMBED_DIM, PATCH_VALUES), "proj.bias": (EMBED_DIM,),
             "norm.weight": (EMBED_DIM,), "norm.bias": (EMBED_DIM,)}
PE_TENSORS = ("pe_patches", "pe_proj", "x")


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
_PE: dict = {}


def pe_parameters() -> dict[str, np.ndarray]:
    """{"proj.weight" [96, 32], "proj.bias", "norm.weight", "norm.bias"} float32 (memoized)."""
    if not _PE:
        template = {name: torch.empty(shape) for name, shape in PE_SHAPES.items()}
        _PE.update(generate_parameters(template, PE_SPEC, PE_SEED))
    return _PE


def timm_pe_state(pe: dict, *, in_chans: int = IN_CHANS, patch: int = PATCH) -> dict[str, torch.Tensor]:
    """timm PatchEmbed state_dict: proj.weight [dim, C p^2], flattened (c, ky, kx), reshaped
    to Conv2d's [dim, C, p, p] (today's [96, 2, 4, 4])."""
    state = {k: torch.from_numpy(np.asarray(v)) for k, v in pe.items()}
    dim = state["proj.weight"].shape[0]
    state["proj.weight"] = state["proj.weight"].reshape(dim, in_chans, patch, patch)
    return state


def parameter_count(pe: dict) -> int:
    return int(sum(np.asarray(v).size for v in pe.values()))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def check_patch_embed(pe_module, *, patch: int = PATCH, in_chans: int = IN_CHANS,
                      embed_dim: int = EMBED_DIM):
    """The configuration Milestone 6 implements, at the given patch, input channels and
    width (today's 4, 2 and 96): Conv2d with kernel = stride = patch and a bias, then
    LayerNorm(embed_dim, eps 1e-5) on NHWC tokens (raises otherwise)."""
    from timm.layers.format import Format
    proj, norm = pe_module.proj, pe_module.norm
    expected = dict(in_channels=in_chans, out_channels=embed_dim, kernel_size=(patch, patch),
                    stride=(patch, patch), padding=(0, 0), dilation=(1, 1), groups=1)
    got = {k: getattr(proj, k) for k in expected}
    if not isinstance(proj, torch.nn.Conv2d) or got != expected or proj.bias is None:
        raise ValueError(f"unexpected patch projection {proj}")
    if not isinstance(norm, torch.nn.LayerNorm) or norm.eps != LN_EPS \
            or not norm.elementwise_affine or norm.normalized_shape != (embed_dim,):
        raise ValueError(f"unexpected patch norm {norm}")
    if pe_module.output_fmt != Format.NHWC or pe_module.flatten or pe_module.dynamic_img_pad:
        raise ValueError("expected NHWC output, no flatten, no dynamic padding")


def make_image_model(height: int, width: int, parameters: dict | None = None,
                     pe: dict | None = None, dtype=torch.float32):
    """timm SwinTransformer(in_chans=2, img_size=(height, width)) with Milestone 5's
    encoder parameters and the patch embedding `pe` (input pixels height x width)."""
    if height % PATCH or width % PATCH:
        raise ValueError("input extents must be multiples of the 4-px patch")
    model = m5.make_model((height // PATCH, width // PATCH), parameters, dtype, in_chans=IN_CHANS)
    check_patch_embed(model.patch_embed)
    if getattr(model, "absolute_pos_embed", None) is not None:
        raise ValueError("Swin-T has no absolute position embedding")
    if pe is not None:
        model.patch_embed.load_state_dict({k: v.to(dtype) for k, v in timm_pe_state(pe).items()})
    return model.eval()


def make_patch_embed(height: int, width: int, pe: dict | None = None, dtype=torch.float32,
                     embed_dim: int = EMBED_DIM, *, patch: int = PATCH, in_chans: int = IN_CHANS):
    """A standalone timm PatchEmbed exactly as SwinTransformer builds it:
    ``PatchEmbed(img_size=(height, width), patch_size=patch, in_chans=in_chans,
    embed_dim=embed_dim, norm_layer=nn.LayerNorm, output_fmt="NHWC")``, with the weights
    ``pe`` when given (proj.weight flattened [embed_dim, in_chans patch^2])."""
    from timm.layers import PatchEmbed
    module = PatchEmbed(img_size=(height, width), patch_size=patch, in_chans=in_chans,
                        embed_dim=embed_dim, norm_layer=torch.nn.LayerNorm, output_fmt="NHWC")
    if pe is not None:
        module.load_state_dict(timm_pe_state(pe, in_chans=in_chans, patch=patch))
    return module.to(dtype).eval()


# ---------------------------------------------------------------------------
# Explicit forward
# ---------------------------------------------------------------------------
def to_nchw(raster: np.ndarray, dtype=torch.float32) -> torch.Tensor:
    """[H, W, C] -> [1, C, H, W]."""
    return torch.from_numpy(np.ascontiguousarray(raster)).permute(2, 0, 1)[None].to(dtype)


def patches(x: torch.Tensor, patch: int = PATCH) -> torch.Tensor:
    """[1, C, H, W] -> [1, H/p, W/p, C p^2]: each token's patch flattened (c, ky, kx), the
    Conv2d weight order (unfold; values are moved, not computed)."""
    _, _, height, width = x.shape
    columns = torch.nn.functional.unfold(x, kernel_size=patch, stride=patch)
    return columns.transpose(1, 2).reshape(1, height // patch, width // patch, -1)


def explicit_patch_embed(pe_module, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """timm's own proj and norm, step by step ([1, h, w, .] NHWC); raises unless the
    result is bit-identical to pe_module(x). The patch is the module's own."""
    patch_y, patch_x = pe_module.patch_size
    if patch_y != patch_x:
        raise ValueError(f"square patches only; received {pe_module.patch_size}")
    with torch.no_grad():
        proj = pe_module.proj(x).permute(0, 2, 3, 1)
        tokens = pe_module.norm(proj)
        if not torch.equal(tokens, pe_module(x)):
            raise AssertionError("explicit patch embedding differs from timm")
    return {"pe_patches": patches(x, patch_y), "pe_proj": proj, "x": tokens}


def _wanted(names, key: str) -> bool:
    return names is None or key in names


def image_encoder_reference(parameters: dict, pe: dict, z: np.ndarray, dtype=torch.float32,
                            names=None, stop: str | None = None) -> dict[str, np.ndarray]:
    """Every named tensor of the forward pass from the preprocessed input z [H, W, 2].

    stop="x" ends after the patch embedding.  Otherwise the encoder runs through
    Milestone 5's explicit forward and the final output is checked bit for bit
    against model.forward_features(input).
    """
    height, width, _ = z.shape
    model = make_image_model(height, width, parameters, pe, dtype)
    x = to_nchw(z, dtype)
    pe_t = explicit_patch_embed(model.patch_embed, x)
    result = {k: to_window_layout(v[0].numpy()) for k, v in pe_t.items() if _wanted(names, k)}
    if _wanted(names, "input"):
        result["input"] = np.asarray(z, dtype=np.float64)
    if stop == "x":
        return result
    tokens = pe_t["x"][0].numpy()
    wanted = None if names is None else sorted(set(names) - set(PE_TENSORS) - {"input"} | {"norm"})
    encoder = m5.encoder_reference(parameters, tokens, dtype, wanted)
    with torch.no_grad():
        final = model.forward_features(x)
    explicit = from_window_layout(encoder["norm"], height // 32, width // 32)
    if not torch.equal(torch.from_numpy(explicit)[None], final):
        raise AssertionError(f"explicit forward differs from forward_features ({dtype})")
    result.update({k: v for k, v in encoder.items() if _wanted(names, k) and k != "x"})
    return result


# ---------------------------------------------------------------------------
# Phase 1: patch-embedding semantics, determined experimentally
# ---------------------------------------------------------------------------
def integer_fixture(height: int, width: int, seed: int = 0, channels: int = IN_CHANS) -> np.ndarray:
    """[H, W, C] small integers: every float32 product and sum is exact."""
    rng = np.random.default_rng(seed)
    return rng.integers(-8, 9, size=(height, width, channels)).astype(np.float32)


def one_hot_extraction(z: np.ndarray, dtype=torch.float64, *, patch: int = PATCH) -> np.ndarray:
    """timm's own Conv2d with one-hot weights (output k = patch element k, zero bias):
    [H/p, W/p, C p^2] for z [H, W, C] (today's [H/4, W/4, 32]).  Exact for any input (one
    product by 1, sums of zeros)."""
    height, width, channels = z.shape
    values = channels * patch * patch
    module = make_patch_embed(height, width, embed_dim=values, dtype=dtype, patch=patch, in_chans=channels)
    with torch.no_grad():
        module.proj.weight.copy_(torch.eye(values, dtype=dtype).reshape(values, channels, patch, patch))
        module.proj.bias.zero_()
        return module.proj(to_nchw(z, dtype)).permute(0, 2, 3, 1)[0].numpy()


def pixel_ids(height: int, width: int, channels: int = IN_CHANS) -> np.ndarray:
    """[H, W, C] float32 unique (pixel, channel) IDs (y W + x) C + c, exact below 2^24."""
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    ids = (y * width + x)[..., None] * channels + np.arange(channels)
    if ids.max() >= 2 ** 24:
        raise ValueError("IDs would not be exact in float32")
    return ids.astype(np.float32)


def timm_patch_ids(height: int, width: int, *, patch: int = PATCH, channels: int = IN_CHANS,
                   window: int = 7) -> np.ndarray:
    """[nW1, w^2, C p^2] (today's [nW1, 49, 32]): the pixel IDs each stage-1 token's patch
    holds, in timm's flatten order, extracted by timm's own Conv2d (one-hot weights), in
    window layout."""
    return to_window_layout(one_hot_extraction(pixel_ids(height, width, channels), patch=patch), window)


# ---------------------------------------------------------------------------
# Preprocessing twin in torch (the NumPy reference is tensor_ee.segwater_preprocessing)
# ---------------------------------------------------------------------------
def preprocess_db_torch(db: np.ndarray) -> np.ndarray:
    """SegWater v3 in torch float32, op for op ([..., 2] VV, VH; no masks)."""
    from tensor_ee import segwater_preprocessing as sp
    x = torch.from_numpy(np.asarray(db, dtype=np.float64)).float()
    offset, step = torch.tensor(float(sp.OFFSET_DB)), torch.tensor(float(sp.STEP_DB))
    code = torch.clamp(torch.round((x + offset) / step), sp.MIN_CODE, sp.MAX_CODE)
    decoded = code * step - offset
    means = torch.tensor(sp.MEANS, dtype=torch.float32)
    stds = torch.tensor(sp.STDS, dtype=torch.float32)
    return ((decoded - means) / stds).numpy()


def preprocess(raw: np.ndarray, mask: np.ndarray | None = None,
               policy: str = "training_mean") -> np.ndarray:
    """The model input [H, W, 2] float32 from raw dB [H, W, 2] (+ validity mask [H, W, 2],
    True = valid): SegWater v3 with masked values filled by the training mean in dB
    (policy "training_mean", production) or with z = 0 ("zero")."""
    from tensor_ee.segwater_preprocessing import preprocess_db_numpy
    data = np.asarray(raw, dtype=np.float64)
    if mask is not None:
        data = np.ma.array(data, mask=~np.asarray(mask, dtype=bool))
    if policy == "zero":                 # z of masked values exactly 0 (evaluated, not production)
        out = preprocess_db_numpy(data, channel_axis=-1, masked_policy="preserve")
        return np.ma.filled(out, 0).astype(np.float32)
    out = preprocess_db_numpy(data, channel_axis=-1, masked_policy=policy)
    return np.ma.getdata(out).astype(np.float32)


# ---------------------------------------------------------------------------
