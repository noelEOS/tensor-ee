"""Milestone 10, Phases 1-3 (Tests 1-3): the installed SMP UPerNet's multi-scale fusion stage, read
off the installed code and run operation by operation.

The authoritative code is segmentation_models_pytorch 0.5.0's decoders/upernet/decoder.py,
`UPerNetDecoder.forward` after the FPN conv blocks (Milestone 9's recursion leaves
fpn_features = [PSP, P3, P2, P1]):

    target_size = fpn_features[-1].shape[2:]                    # P1's size: C1 = H/4
    resized = [F.interpolate(f, size=target_size, mode="bilinear", align_corners=False)
               for f in fpn_features]                           # PSP 8x, P3 4x, P2 2x, P1 1x
    stacked = torch.cat(resized[::-1], dim=1)                   # [P1, P2, P3, PSP]: 1024
    output  = fusion_block(stacked)                             # Conv2dReLU 3x3 1024 -> 256

    fusion_block = Conv2d(1024, 256, 3, padding=1, bias=False) -> BatchNorm2d(256, eps 1e-5)
                   -> ReLU(inplace)

The decoder's forward returns `output`: the fused feature IS UPerNetDecoder's output (the
segmentation head follows in the model, not in the decoder).

Names (NHWC NumPy outside this module):

    P1, P2, P3, psp_relu        the four post-ReLU pyramid outputs (Milestone 9)
    P1_to_C1 .. PSP_to_C1       each resized to C1 (P1's call is a same-size interpolate)
    fusion_concat               cat([P1_to_C1, P2_to_C1, P3_to_C1, PSP_to_C1], channels)
    fusion_conv, fusion_bn      fusion_block[0], fusion_block[1]
    fusion_relu                 fusion_block's output = the decoder's output

`contract` records the structure programmatically (source lines, configuration, the four
recorded interpolation calls, the [P1, P2, P3, PSP] order with unique level / channel IDs
at every position, P1's same-size resize as a bitwise identity); `expanded` runs the
installed submodules one at a time and raises unless every tensor equals the decoder's own
forward bit for bit (the fusion input and the decoder output included).

The grouped decomposition (Phase 14): the fusion weight [256, 1024, 3, 3] has bias=False and
its input channels are four consecutive 256-channel groups; W_g = weight[:, 256 g : 256 (g+1)]
and, in the asset's flattened (c, ky, kx) order, W_g is the contiguous column slice
[2304 g, 2304 (g + 1)).
"""

from __future__ import annotations

from tensor_ee.models.swin_t224.configuration import tensor_channels, bn_args

import inspect

import numpy as np
import torch
import torch.nn.functional as F

from reference.upernet import smp_feature_interface as S
from reference.upernet import smp_fpn_reference as FR

DECODER = FR.DECODER
PREFIX = "decoder.fusion_block"
WEIGHT = f"{PREFIX}.0.weight"
GROUPS = 4
FUSED = GROUPS * DECODER                                    # 1024 input channels
PYRAMID = ("P1", "P2", "P3", "psp_relu")                    # concatenation order
RESIZED = {"P1": "P1_to_C1", "P2": "P2_to_C1", "P3": "P3_to_C1", "psp_relu": "PSP_to_C1"}
STEPS = ("fusion_conv", "fusion_bn", "fusion_relu")
TENSORS = list(PYRAMID) + [RESIZED[n] for n in PYRAMID] + ["fusion_concat", *STEPS]
FUSION_TENSORS = [RESIZED[n] for n in PYRAMID] + ["fusion_concat", *STEPS]

model = FR.model
nchw = FR.nchw
nhwc = FR.nhwc


def group_weight(weight, g: int, groups: int = GROUPS) -> np.ndarray:
    """W_g [Cout, Cin / groups, 3, 3]: input-channel group g of ``groups`` contiguous groups
    (the fusion weight at 4 groups: [256, 256, 3, 3], groups P1, P2, P3, PSP). ``groups``
    must divide Cin, and 0 <= g < groups (task P04 parameterized the count)."""
    cin = np.shape(weight)[1]
    if groups < 1 or cin % groups or not 0 <= g < groups:
        raise ValueError(f"group {g} of {groups} does not split {cin} input channels evenly")
    width = cin // groups
    return np.asarray(weight)[:, g * width:(g + 1) * width]


# ---------------------------------------------------------------------------
# The fusion stage on given pyramid tensors (Phase 1)
# ---------------------------------------------------------------------------
def fusion_stage(decoder, pyramid: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The installed fusion calls on {P1, P2, P3, psp_relu} (NCHW), operation by operation:
    the decoder's resize loop (in its own order PSP, P3, P2, P1), the reversed concat, the
    fusion block's Conv, BN and ReLU.  Raises unless the expanded block equals the module."""
    with torch.no_grad():
        target = pyramid["P1"].shape[2:]
        resized = {name: F.interpolate(pyramid[name], size=target, mode="bilinear", align_corners=False)
                   for name in ("psp_relu", "P3", "P2", "P1")}
        cat = torch.cat([resized[name] for name in PYRAMID], dim=1)
        block = decoder.fusion_block
        conv = block[0](cat)
        bn = block[1](conv)
        relu = torch.relu(bn)                               # SMP's ReLU is in place
        if not torch.equal(relu, block(cat)):
            raise AssertionError("expanded fusion Conv2dReLU differs from fusion_block")
    out = {RESIZED[name]: resized[name] for name in PYRAMID}
    out.update(fusion_concat=cat, fusion_conv=conv, fusion_bn=bn, fusion_relu=relu)
    return out


def expanded(decoder, feats: list[torch.Tensor], check: bool = True) -> dict[str, torch.Tensor]:
    """Every PSP, FPN and fusion intermediate from the installed submodules (NCHW).  feats:
    raw C1..C4 (NCHW-contiguous).  check: one hooked run of the decoder's own forward, to
    the end; raises unless every PSP / FPN tensor (Milestone 9's checks), the fusion
    block's input and the decoder's output equal the expanded ones bit for bit."""
    out = FR.expanded(decoder, feats, check=False)
    out.update(fusion_stage(decoder, {name: out[name] for name in PYRAMID}))
    if check:
        cap = FR.module_forward(decoder, feats, stop_before_fusion=False)
        FR.check_against_module(out, cap)
        if not torch.equal(out["fusion_concat"], cap["fusion_in"]):
            raise AssertionError("expanded fusion_concat differs from the fusion block's input")
        if not torch.equal(out["fusion_relu"], cap["output"]):
            raise AssertionError("expanded fusion output differs from UPerNetDecoder.forward")
    return out


def fusion_tensors(feats: list[np.ndarray], dtype=torch.float32, check: bool = True,
                   keep=None) -> dict:
    """{name: NHWC array} of the installed decoder (PSP, FPN, fusion) on raw C1..C4."""
    t = expanded(model(dtype).decoder, [nchw(np.asarray(f), dtype) for f in feats], check)
    keep = TENSORS if keep is None else keep
    return {k: nhwc(v) for k, v in t.items() if k in keep}


def fusion_on_pyramid(pyramid: dict[str, np.ndarray], dtype=torch.float32) -> dict:
    """SMP's fusion stage alone on given pyramid values (NHWC), e.g. exactly the values EE's
    FPN produced: the fusion's own float32 error, without the FPN's."""
    t = fusion_stage(model(dtype).decoder, {n: nchw(np.asarray(pyramid[n]), dtype) for n in PYRAMID})
    return {k: nhwc(v) for k, v in t.items()}


# ---------------------------------------------------------------------------
# Unique-ID fixtures (Phase 2, Test 2)
# ---------------------------------------------------------------------------
class ChannelPSP(torch.nn.Module):
    """A PSP stand-in for ID fixtures only: channel c of its output is base + c everywhere."""

    def __init__(self, base: float):
        super().__init__()
        self.base = base

    def forward(self, feature):
        _, _, h, w = feature.shape
        ids = self.base + torch.arange(DECODER, dtype=feature.dtype)
        return ids[None, :, None, None].expand(1, DECODER, h, w).contiguous()


LEVEL_IDS = {"P1": 1000.0, "P2": 2000.0, "P3": 3000.0, "psp_relu": 4000.0}


def expected_ids() -> np.ndarray:
    """Channel j of the fusion input under the ID fixture: 1000 (j // 256 + 1) + j % 256."""
    j = np.arange(FUSED)
    return 1000.0 * (j // DECODER + 1) + j % DECODER


def id_fixture(decoder, input_size: int = 224) -> dict:
    """Tests 1-2 with unique level / channel IDs.  Copies the decoder with PSP = 4000 + c and
    conv blocks P3 = 3000 + c, P2 = 2000 + c, P1 = 1000 + c (Milestone 9's id_decoder: zero
    conv weights, BatchNorm mean 0 and bias = ID: exact), runs its own forward and reads the
    fusion block's input at every position of all 1,024 channels."""
    d = FR.id_decoder(decoder, conv_base=(LEVEL_IDS["P3"], LEVEL_IDS["P2"], LEVEL_IDS["P1"]))
    d.psp = ChannelPSP(LEVEL_IDS["psp_relu"])
    cap = FR.module_forward(d, FR.id_features(input_size))
    fusion = cap["fusion_in"][0]                                       # [1024, H, W]
    want = torch.from_numpy(expected_ids()).float()[:, None, None].expand_as(fusion)
    segments = []
    for g, name in enumerate(PYRAMID):
        part = fusion[g * DECODER:(g + 1) * DECODER]
        segments.append({"channels": [g * DECODER, (g + 1) * DECODER], "source": name,
                         "ids_exact": bool(torch.equal(part, want[g * DECODER:(g + 1) * DECODER])),
                         "first": float(part[0, 0, 0]), "last": float(part[-1, -1, -1])})
    return {"fusion_input_shape": list(cap["fusion_in"].shape),
            "every_channel_every_position_exact": bool(torch.equal(fusion, want)),
            "positions_checked": int(fusion.shape[1] * fusion.shape[2]),
            "segments": segments, "call_order": cap["order"]}


def p1_identity(p1: torch.Tensor) -> dict:
    """P1's own resize: F.interpolate(P1, size=P1's size, bilinear, align_corners=False)."""
    with torch.no_grad():
        out = F.interpolate(p1, size=p1.shape[2:], mode="bilinear", align_corners=False)
    int_type = torch.int64 if p1.dtype == torch.float64 else torch.int32
    return {"values_equal": bool(torch.equal(out, p1)),
            "bits_equal": bool(torch.equal(out.view(int_type), p1.view(int_type))),
            "same_storage": out.data_ptr() == p1.data_ptr(),
            "negative_zeros": int((p1.view(int_type) == torch.iinfo(int_type).min).sum())}


# ---------------------------------------------------------------------------
# The contract (Phase 1, Tests 1-3)
# ---------------------------------------------------------------------------
def fusion_source() -> dict:
    """The fusion lines of UPerNetDecoder.forward and the fusion block's construction."""
    from segmentation_models_pytorch.decoders.upernet import decoder as D
    forward, fstart = inspect.getsourcelines(D.UPerNetDecoder.forward)
    init, istart = inspect.getsourcelines(D.UPerNetDecoder.__init__)
    first = next(i for i, line in enumerate(forward) if "Resize all FPN features" in line)
    last = next(i for i, line in enumerate(forward) if "self.fusion_block(" in line)
    ifirst = next(i for i, line in enumerate(init) if "num_blocks_to_fuse =" in line)
    ilast = next(i for i in range(ifirst, len(init)) if init[i].strip() == ")")
    return {"file": inspect.getsourcefile(D.UPerNetDecoder),
            "forward_fusion_lines": [fstart + first, fstart + last + 1],
            "forward_fusion": "".join(forward[first:last + 2]),
            "init_fusion_lines": [istart + ifirst, istart + ilast],
            "init_fusion": "".join(init[ifirst:ilast + 1])}


def real_forward_check(seed: int = 0, size: int = 224) -> dict:
    """In a real UPerNet forward (random input), the decoder's output equals the expanded
    fusion_relu of the features it received, the head consumes exactly it, and P1's
    same-size resize is a bitwise identity."""
    m = model(torch.float32)
    captured = {}
    pre = m.decoder.register_forward_pre_hook(
        lambda mod, args: captured.__setitem__("features", [f.detach().clone() if f is not None
                                                            else None for f in args[0]]))
    head = m.segmentation_head.register_forward_pre_hook(
        lambda mod, args: captured.__setitem__("head_in", args[0].detach().clone()))
    x = torch.from_numpy(np.random.default_rng(seed).standard_normal(
        (1, S.IN_CHANNELS, size, size)).astype(np.float32))
    try:
        with torch.no_grad():
            m(x)
    finally:
        pre.remove()
        head.remove()
    t = expanded(m.decoder, captured["features"][2:])                  # raises on any mismatch
    return {"input_size": size, "expanded_equals_decoder_output": True,
            "head_input_is_fusion_relu": bool(torch.equal(captured["head_in"], t["fusion_relu"])),
            "fusion_output_shape": list(t["fusion_relu"].shape),
            "p1_identity": p1_identity(t["P1"])}


def contract() -> dict:
    import segmentation_models_pytorch as smp
    m = model(torch.float32)
    d = m.decoder
    feats = [torch.zeros(1, FR.CHANNELS[k], 224 // S.FEATURE_STRIDES[k - 1],
                         224 // S.FEATURE_STRIDES[k - 1]) for k in (1, 2, 3, 4)]
    with S.recorded_interpolations() as calls, torch.no_grad():
        d([None, None, *feats])
    source = fusion_source()
    fusion_calls = [c for c in calls if c["target"] == [56, 56] and c["caller"].startswith("decoder.py")
                    and c["caller"].endswith("(forward)") and int(c["caller"].split(":")[1].split()[0])
                    >= source["forward_fusion_lines"][0]]
    weight = S.decoder_parameters()[WEIGHT]
    flat = weight.reshape(weight.shape[0], -1)
    contiguous = all(np.array_equal(flat[:, g * 2304:(g + 1) * 2304],
                                    group_weight(weight, g).reshape(DECODER, -1)) for g in range(GROUPS))
    return {
        "versions": {"segmentation_models_pytorch": smp.__version__, "torch": torch.__version__},
        "source": source,
        "fusion_block": {"class": type(d.fusion_block).__name__, **FR.block_config(d.fusion_block)},
        "fusion_interpolations": fusion_calls,
        "concat": {"call": "torch.cat(resized_fpn_features[::-1], dim=1)",
                   "resize_loop_order": ["psp_relu", "P3", "P2", "P1"], "concat_order": list(PYRAMID)},
        "ids": id_fixture(d),
        "real_forward": real_forward_check(),
        "parameters": {name: list(v.shape) for name, v in S.decoder_parameters().items()
                       if name.startswith(PREFIX)},
        "group_weight_slices_contiguous": contiguous,
    }


# ---------------------------------------------------------------------------
# References (Phases 1, 10)
# ---------------------------------------------------------------------------
REFS = ("torch_float32", "float64", "ee_mirror", "torch_float32_same_input",
        "torch_float32_fusion_input")


class LazyConcat(dict):
    """A reference dict whose fusion_concat is the concatenation of the resized levels,
    built on first access."""

    def __missing__(self, key):
        if key != "fusion_concat" or not all(RESIZED[n] in self for n in PYRAMID):
            raise KeyError(key)
        value = np.concatenate([np.asarray(self[RESIZED[n]]) for n in PYRAMID], axis=-1)
        self[key] = value
        return value

    def __contains__(self, key):
        return dict.__contains__(self, key) or (
            key == "fusion_concat" and all(dict.__contains__(self, RESIZED[n]) for n in PYRAMID))


def references(feats: dict[str, list[np.ndarray]] | list[np.ndarray],
               refs=REFS, mirror_pyramid: dict | None = None) -> dict:
    """{reference: {tensor: NHWC}} for the pyramid and fusion tensors:

      torch_float32               SMP float32 on its own features
      float64                     SMP float64 on the float64 features
      ee_mirror                   the mirror (Milestone 9's FPN mirror, then ee_mirror_fusion)
                                  on the values EE reads
      torch_float32_same_input    SMP float32 on the values EE reads (C1-C4)
      torch_float32_fusion_input  SMP's fusion stage alone, float32, on the mirror's pyramid
                                  (= EE's P1, P2, P3, PSP bit for bit): the fusion's own error

    mirror_pyramid: the mirror's {P1, P2, P3, psp_relu} if already known.  fusion_concat is not
    kept: it is the four resized levels side by side, rebuilt on access.  (The research's
    on-disk cache of the large crops' references was archived at L06b.)"""
    from reference.upernet import ee_mirror_fpn as MF
    from reference.upernet import ee_mirror_fusion as MU
    if not isinstance(feats, dict):
        feats = {"torch_float32": feats, "float64": feats, "ee_mirror": feats}
    out: dict = {}
    decoder = S.decoder_parameters()
    pyramid = None
    for ref in refs:
        if ref == "torch_float32":
            t = fusion_tensors(feats["torch_float32"], torch.float32)
        elif ref == "float64":
            t = fusion_tensors([np.asarray(f, np.float64) for f in feats["float64"]], torch.float64)
        elif ref == "torch_float32_same_input":
            t = fusion_tensors([np.asarray(f, np.float32) for f in feats["ee_mirror"]], torch.float32)
        else:
            if pyramid is None:
                pyramid = mirror_pyramid
                if pyramid is None:
                    full = MF.fpn(feats["ee_mirror"], decoder)
                    pyramid = {n: full[n] for n in PYRAMID}
                    del full
            if ref == "ee_mirror":
                t = {**{n: np.asarray(pyramid[n], np.float64) for n in PYRAMID},
                     **MU.fusion(pyramid, decoder)}
            else:
                t = {**{n: np.asarray(pyramid[n], np.float32) for n in PYRAMID},
                     **fusion_on_pyramid(pyramid, torch.float32)}
        t.pop("fusion_concat", None)
        out[ref] = LazyConcat(t)
    return out
