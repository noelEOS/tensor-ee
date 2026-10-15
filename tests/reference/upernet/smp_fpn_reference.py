"""Milestone 9, Phases 1-2, 23 (Tests 1-2): the installed SMP UPerNet's top-down FPN pathway,
read off the installed code and run operation by operation.

The authoritative code is segmentation_models_pytorch 0.5.0's decoders/upernet/decoder.py,
`UPerNetDecoder.forward` and `FPNLateralBlock`.  As installed (not as usually drawn):

    features = [LayerNorm2d_k(C_k) for k = 1..4]               # feature_norms[0..3]
    psp_out  = psp(features[-1])                                # Milestone 8
    lateral  = features[:-1][::-1]                              # [C3_norm, C2_norm, C1_norm]
    fpn = [psp_out]
    for i, block in enumerate(fpn_lateral_blocks):              # i = 0, 1, 2  ->  C3, C2, C1
        fpn.append(block(fpn[-1], lateral[i]))                  #   up(state) + lateral(C)
    for i, conv_block in enumerate(fpn_conv_blocks, start=1):   # after the whole recursion
        fpn[i] = conv_block(fpn[i])                             #   P = Conv-BN-ReLU 3x3 (F)

    FPNLateralBlock(state, lateral):  l = Conv2dReLU 1x1(lateral)          (Conv, BN, ReLU)
                                      u = F.interpolate(state, size=l.shape[2:],
                                                        mode="bilinear", align_corners=False)
                                      return u + l

So the state handed down the pyramid is the pre-convolution sum F_k = l_k + up(F_{k+1})
(F_4 = the PSP output), and each 3 x 3 Conv2dReLU is applied to its F_k afterwards:
P_k = conv_k(F_k) never feeds another level.  The usual diagram (P3 -> up -> + lateral(C2))
is not what the installed code computes.

Names (NHWC NumPy outside this module), per level k = 3, 2, 1 (lateral / conv block 3 - k):

    C{k}_norm   feature_norms[k - 1](C_k)
    lat{k}_conv, lat{k}_bn, lat{k}       fpn_lateral_blocks[3 - k].conv_norm_relu steps
    up{k}       the state resized to C_k (PSP output for k = 3, F_{k+1} otherwise)
    F{k}        up{k} + lat{k} = fpn_lateral_blocks[3 - k] output (the next level's state)
    P{k}_conv, P{k}_bn, P{k}             fpn_conv_blocks[3 - k] steps (P{k} = after ReLU)

`contract` establishes the structure programmatically (source, configurations, recorded
interpolation calls, call order, the feature -> block mapping and the recursion state with
unique IDs, the fusion order M10 consumes); `expanded` runs the installed submodules one
at a time and raises unless every F, every P and the PSP output equal the module's own
forward bit for bit, up to the input of the fusion block.
"""

from __future__ import annotations

from tensor_ee.models.swin_t224.configuration import lateral_prefix, conv_prefix, norm_prefix

from tensor_ee.models.swin_t224.configuration import level_tensors

import copy
import inspect

import numpy as np
import torch
import torch.nn.functional as F

from reference.upernet import smp_feature_interface as S
from reference.upernet import smp_psp_reference as R

DECODER = 256
LEVELS = (3, 2, 1)                                      # traversal order of the recursion
CHANNELS = dict(zip((1, 2, 3, 4), S.FEATURE_CHANNELS))


PSP_TENSORS = [t for t in R.TENSORS if t != "C4_raw"]
TENSORS = (["C1_raw", "C2_raw", "C3_raw", "C4_raw"] + PSP_TENSORS
           + [t for k in LEVELS for t in level_tensors(k)])


def tensor_channels(name: str) -> int:
    if name.endswith("_raw") or (name.endswith("_norm") and name[0] == "C"):
        return CHANNELS[int(name[1])]
    if name == "psp_concat":
        return 1792
    if name.startswith("pool_"):
        return 768
    return DECODER


model = R.model
nchw = R.nchw
nhwc = R.nhwc


# ---------------------------------------------------------------------------
# The installed module's own forward, observed (Phase 1)
# ---------------------------------------------------------------------------
class _BeforeFusion(Exception):
    """Raised by the fusion block's pre-hook: the forward stops at its input."""


def module_forward(decoder, feats: list[torch.Tensor], stop_before_fusion: bool = True) -> dict:
    """Run UPerNetDecoder.forward([_, _, C1, C2, C3, C4]) with hooks; returns what every FPN
    module received and returned, the call order, the recorded interpolations and the
    fusion block's input (the forward stops there unless stop_before_fusion is False)."""
    cap = {"order": [], "lateral_state": {}, "lateral_in": {}, "lateral_out": {},
           "conv_in": {}, "conv_out": {}, "norm_out": {}}
    hooks = []

    def keep(t):
        return t.detach().clone()

    def record(label, **items):
        """A forward hook that records and returns None (a returned value would replace
        the module's output)."""
        def hook(module, args, output=None):
            cap["order"].append(label)
            for key, (slot, pick) in items.items():
                value = keep(pick(args, output))
                if slot is None:
                    cap[key] = value
                else:
                    cap[key][slot] = value
        return hook

    def pre(key, slot):
        def hook(module, args):
            cap[key][slot] = keep(args[0])
        return hook

    for k, norm in enumerate(decoder.feature_norms):
        hooks.append(norm.register_forward_hook(
            record(f"feature_norms.{k}", norm_out=(k, lambda a, o: o))))
    hooks.append(decoder.psp.register_forward_hook(
        record("psp", psp_out=(None, lambda a, o: o))))
    for i, block in enumerate(decoder.fpn_lateral_blocks):
        hooks.append(block.register_forward_hook(record(
            f"fpn_lateral_blocks.{i}", lateral_state=(i, lambda a, o: a[0]),
            lateral_in=(i, lambda a, o: a[1]), lateral_out=(i, lambda a, o: o))))
    for i, block in enumerate(decoder.fpn_conv_blocks):
        hooks.append(block.register_forward_pre_hook(pre("conv_in", i)))
        hooks.append(block.register_forward_hook(
            record(f"fpn_conv_blocks.{i}", conv_out=(i, lambda a, o: o))))

    def fusion_hook(m, a):
        cap["order"].append("fusion_block")
        cap["fusion_in"] = keep(a[0])
        if stop_before_fusion:
            raise _BeforeFusion

    hooks.append(decoder.fusion_block.register_forward_pre_hook(fusion_hook))
    try:
        with S.recorded_interpolations() as calls, torch.no_grad():
            try:
                cap["output"] = decoder([None, None, *feats])
            except _BeforeFusion:
                pass
    finally:
        for h in hooks:
            h.remove()
    cap["interpolations"] = calls
    return cap


# ---------------------------------------------------------------------------
# The expanded forward (Phase 1)
# ---------------------------------------------------------------------------
def expanded(decoder, feats: list[torch.Tensor], check: bool = True) -> dict[str, torch.Tensor]:
    """Every PSP and FPN intermediate from the installed submodules (NCHW).  feats: raw
    C1..C4 as the encoder delivers them (NCHW-contiguous).  Raises unless each step equals
    its module, and (check) unless the PSP output, every F and every P equal what the
    decoder's own forward computes, bit for bit."""
    out: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        norms = [decoder.feature_norms[k](f) for k, f in enumerate(feats)]
        out.update(R.expanded(decoder, feats[3]))           # PSP: raises unless == PSPModule
        if not torch.equal(out["C4_norm"], norms[3]):
            raise AssertionError("PSP input differs from feature_norms[3](C4)")
        for k in (1, 2, 3):
            out[f"C{k}_raw"], out[f"C{k}_norm"] = feats[k - 1], norms[k - 1]
        state = out["psp_relu"]
        for i, block in enumerate(decoder.fpn_lateral_blocks):
            k = LEVELS[i]
            lateral = norms[k - 1]
            cnr = block.conv_norm_relu
            conv = cnr[0](lateral)
            bn = cnr[1](conv)
            relu = torch.relu(bn)                           # SMP's ReLU is in place
            if not torch.equal(relu, cnr(lateral)):
                raise AssertionError(f"lateral block {i}: expanded Conv2dReLU differs")
            up = F.interpolate(state, size=relu.shape[2:], mode="bilinear", align_corners=False)
            fused = up + relu                               # state_feature + lateral_feature
            if not torch.equal(fused, block(state, lateral)):
                raise AssertionError(f"lateral block {i}: expanded FPNLateralBlock differs")
            out.update({f"lat{k}_conv": conv, f"lat{k}_bn": bn, f"lat{k}": relu,
                        f"up{k}": up, f"F{k}": fused})
            state = fused                                   # the pre-conv sum goes down
        for i, block in enumerate(decoder.fpn_conv_blocks):
            k = LEVELS[i]
            conv = block[0](out[f"F{k}"])
            bn = block[1](conv)
            relu = torch.relu(bn)
            if not torch.equal(relu, block(out[f"F{k}"])):
                raise AssertionError(f"conv block {i}: expanded Conv2dReLU differs")
            out.update({f"P{k}_conv": conv, f"P{k}_bn": bn, f"P{k}": relu})
    if check:
        check_against_module(out, module_forward(decoder, feats))
    return out


def check_against_module(out: dict, cap: dict):
    """Raises unless the expanded PSP output, every F, every P, every lateral input and the
    state entering each lateral block equal what module_forward captured, bit for bit."""
    pairs = [("psp_relu", cap["psp_out"])]
    for i, k in enumerate(LEVELS):
        pairs += [(f"F{k}", cap["lateral_out"][i]), (f"F{k}", cap["conv_in"][i]),
                  (f"P{k}", cap["conv_out"][i]), (f"C{k}_norm", cap["lateral_in"][i])]
    for name, got in pairs:
        if not torch.equal(out[name], got):
            raise AssertionError(f"expanded {name} differs from UPerNetDecoder.forward")
    states = [cap["lateral_state"][i] for i in range(3)]
    for want, got in zip(("psp_relu", "F3", "F2"), states):
        if not torch.equal(out[want], got):
            raise AssertionError(f"state into the next lateral block is not {want}")


def fpn_tensors(feats: list[np.ndarray], dtype=torch.float32, check: bool = True) -> dict:
    """{name: NHWC array} of the installed decoder up to the fusion input on raw C1..C4."""
    t = expanded(model(dtype).decoder, [nchw(np.asarray(f), dtype) for f in feats], check)
    return {k: nhwc(v) for k, v in t.items()}


# ---------------------------------------------------------------------------
# Unique-ID fixtures (Phase 2, Test 2; Phase 23)
# ---------------------------------------------------------------------------
class ConstantPSP(torch.nn.Module):
    """A PSP stand-in for ID fixtures only: a constant 256-channel map on C4's grid."""

    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, feature):
        _, _, h, w = feature.shape
        return torch.full((1, DECODER, h, w), self.value, dtype=feature.dtype)


def id_decoder(decoder, psp_value: float = 1.0, lateral_values=(10.0, 100.0, 1000.0),
               conv_base=(10000.0, 20000.0, 30000.0)):
    """A copy of the decoder whose FPN modules emit known constants: feature norms are
    identities, the PSP returns psp_value, lateral block i returns lateral_values[i] in
    every channel, conv block i returns conv_base[i] + channel (zero conv weights,
    BatchNorm mean 0 and bias = ID: (0 - 0) / s * gamma + bias, exact)."""
    d = copy.deepcopy(decoder).float().eval()
    d.feature_norms = torch.nn.ModuleList([torch.nn.Identity() for _ in range(4)])
    d.psp = ConstantPSP(psp_value)
    with torch.no_grad():
        for i, block in enumerate(d.fpn_lateral_blocks):
            block.conv_norm_relu[0].weight.zero_()
            block.conv_norm_relu[1].running_mean.zero_()
            block.conv_norm_relu[1].bias.fill_(lateral_values[i])
        for i, block in enumerate(d.fpn_conv_blocks):
            block[0].weight.zero_()
            block[1].running_mean.zero_()
            block[1].bias.copy_(conv_base[i] + torch.arange(DECODER, dtype=torch.float32))
    return d


def id_features(input_size: int = 224) -> list[torch.Tensor]:
    """C_k[c, i, j] = k * 10^6 + c (constant planes; exact in float32)."""
    out = []
    for k in (1, 2, 3, 4):
        h = input_size // S.FEATURE_STRIDES[k - 1]
        ids = k * 1_000_000 + torch.arange(CHANNELS[k], dtype=torch.float32)
        out.append(ids[None, :, None, None].expand(1, CHANNELS[k], h, h).contiguous())
    return out


def id_fixture(decoder) -> dict:
    """Tests 1-2 / Phase 23 with unique IDs.  PSP = 1, lateral blocks 10 / 100 / 1000:

      state into lateral block i, if the state is the pre-conv sum F:  1, 11, 111
      conv block inputs F3, F2, F1:                                    11, 111, 1111
      (were the state P instead: 1, 10000 + c, 20000 + c - visibly different)

    and the fusion block's input segments carry the conv blocks' IDs (P1 = 30000 + c,
    P2 = 20000 + c, P3 = 10000 + c) and the PSP's 1, which pins M10's order."""
    feats = id_features()
    d = id_decoder(decoder)
    cap = module_forward(d, feats)

    def constant(t) -> float | None:
        v = t.flatten()[0].item()
        return v if torch.equal(t, torch.full_like(t, v)) else None

    def channel_ids(t) -> list[float] | None:
        per = t[0, :, :1, :1].expand_as(t[0])
        return t[0, :, 0, 0].tolist() if torch.equal(t[0], per) else None

    mapping = []
    for i in range(3):
        got = cap["lateral_in"][i]
        source = [k for k in (1, 2, 3, 4) if got.shape == feats[k - 1].shape
                  and torch.equal(got, feats[k - 1])]
        mapping.append({"block": i, "lateral_input_is": [f"C{k}" for k in source],
                        "channels": got.shape[1], "size": list(got.shape[2:]),
                        "channel_ids_exact": bool(source)})
    fusion = cap["fusion_in"]
    segments = []
    for s, (label, base) in enumerate((("P1 (conv block 2)", 30000.0), ("P2 (conv block 1)", 20000.0),
                                       ("P3 (conv block 0)", 10000.0), ("PSP", None))):
        part = fusion[:, s * DECODER:(s + 1) * DECODER]
        ids = channel_ids(part)
        want = [1.0] * DECODER if base is None else [base + c for c in range(DECODER)]
        segments.append({"channels": [s * DECODER, (s + 1) * DECODER], "source": label,
                         "ids_exact": ids == want})
    return {
        "call_order": cap["order"],
        "lateral_mapping": mapping,
        "state_into_lateral_blocks": [constant(cap["lateral_state"][i]) for i in range(3)],
        "lateral_block_outputs": [constant(cap["lateral_out"][i]) for i in range(3)],
        "conv_block_inputs": [constant(cap["conv_in"][i]) for i in range(3)],
        "conv_block_sizes": [list(cap["conv_in"][i].shape[2:]) for i in range(3)],
        "fusion_input_channels": fusion.shape[1],
        "fusion_segments": segments,
        "fusion_size": list(fusion.shape[2:]),
    }


# ---------------------------------------------------------------------------
# The contract (Phase 1, Tests 1-2; Phase 23)
# ---------------------------------------------------------------------------
def real_forward_check(seed: int = 0, size: int = 224) -> dict:
    """In a real UPerNet forward (random input), the decoder's hooked FPN tensors equal the
    expanded forward of the features it received, and the fusion input equals
    cat(interpolate([P1, P2, P3, PSP] -> C1, bilinear, align_corners=False))."""
    m = model(torch.float32)
    captured = {}
    pre = m.decoder.register_forward_pre_hook(
        lambda mod, args: captured.__setitem__("features", [f.detach().clone() if f is not None
                                                            else None for f in args[0]]))
    x = torch.from_numpy(np.random.default_rng(seed).standard_normal(
        (1, S.IN_CHANNELS, size, size)).astype(np.float32))
    fusion = {}
    hook = m.decoder.fusion_block.register_forward_pre_hook(
        lambda mod, a: fusion.__setitem__("input", a[0].detach().clone()))
    try:
        with torch.no_grad():
            m(x)
    finally:
        pre.remove()
        hook.remove()
    feats = captured["features"][2:]
    t = expanded(m.decoder, feats)                       # raises on any mismatch
    c1 = t["P1"].shape[2:]
    with torch.no_grad():
        resized = [F.interpolate(t[n], size=c1, mode="bilinear", align_corners=False)
                   for n in ("psp_relu", "P3", "P2", "P1")]
    expected = torch.cat(resized[::-1], dim=1)
    return {"feature_shapes": [list(f.shape) for f in feats],
            "expanded_equals_module_forward": True,
            "fusion_input_equals_cat_resized_reversed": bool(torch.equal(fusion["input"], expected)),
            "p1_identity_resize_exact": bool(torch.equal(resized[3], t["P1"])),
            "fusion_input_shape": list(fusion["input"].shape)}


def block_config(block) -> dict:
    return R.conv_block_config(block)


def contract() -> dict:
    import segmentation_models_pytorch as smp
    from segmentation_models_pytorch.decoders.upernet import decoder as D
    m = model(torch.float32)
    d = m.decoder
    forward, fstart = inspect.getsourcelines(D.UPerNetDecoder.forward)
    init, istart = inspect.getsourcelines(D.UPerNetDecoder.__init__)
    lateral, lstart = inspect.getsourcelines(D.FPNLateralBlock)
    feats = [torch.zeros(1, CHANNELS[k], 224 // S.FEATURE_STRIDES[k - 1],
                         224 // S.FEATURE_STRIDES[k - 1]) for k in (1, 2, 3, 4)]
    cap = module_forward(d, feats, stop_before_fusion=False)
    return {
        "versions": {"segmentation_models_pytorch": smp.__version__, "torch": torch.__version__},
        "source": {"file": inspect.getsourcefile(D.UPerNetDecoder),
                   "decoder_forward_lines": [fstart, fstart + len(forward) - 1],
                   "decoder_init_lines": [istart, istart + len(init) - 1],
                   "lateral_block_lines": [lstart, lstart + len(lateral) - 1],
                   "decoder_forward": "".join(forward), "lateral_block": "".join(lateral)},
        "lateral_blocks": [{"class": type(b).__name__, **block_config(b.conv_norm_relu)}
                           for b in d.fpn_lateral_blocks],
        "conv_blocks": [block_config(b) for b in d.fpn_conv_blocks],
        "feature_norms": [{"class": type(n).__name__, "channels": n.normalized_shape[0],
                           "eps": n.eps, "affine": n.elementwise_affine} for n in d.feature_norms],
        "call_order": cap["order"],
        "interpolations": cap["interpolations"],
        "ids": id_fixture(d),
        "real_forward": real_forward_check(),
        "parameters": {name: list(v.shape) for name, v in S.decoder_parameters().items()
                       if name.startswith(("decoder.fpn_", "decoder.feature_norms."))},
    }


# ---------------------------------------------------------------------------
# Inputs and references (Phases 1, 10, 12)
# ---------------------------------------------------------------------------
SYNTHETIC_SEEDS = {1: 91, 2: 92, 3: 93, 4: 81}       # C4 = Milestone 8's synthetic C4


def synthetic_features(h4: int, scale: float = 3.0) -> list[np.ndarray]:
    """Deterministic C1..C4 for a C4 of h4 x h4 tokens: the project's hash values (multiples
    of 1/2048 in [-1, 1)) x scale, generated identically in EE (sa.window_layout_raster)."""
    from reference import synthetic
    out = []
    for k in (1, 2, 3, 4):
        h = h4 * 2 ** (4 - k)
        out.append((synthetic.generated_raster_numpy(h, h, CHANNELS[k], SYNTHETIC_SEEDS[k])
                    * np.float32(scale)).astype(np.float32))
    return out


def references(feats: dict[str, list[np.ndarray]] | list[np.ndarray],
               refs=("torch_float32", "float64", "ee_mirror", "torch_float32_same_input"),
               keep=None) -> dict:
    """{reference: {tensor: NHWC}}: torch_float32 (the installed modules on the float32
    features), float64 (in float64 on the float64 features), ee_mirror (the mirror on the
    mirror's features = the values EE reads), torch_float32_same_input (the installed
    float32 modules on the mirror's features: the decoder's own error, without the
    encoder's); keep: the tensor names to keep (default all).  (The research's on-disk cache of
    the large crops' references was archived at L06b.)"""
    from reference.upernet import ee_mirror_fpn as MF
    if not isinstance(feats, dict):
        feats = {"torch_float32": feats, "float64": feats, "ee_mirror": feats}
    out: dict = {}
    for ref in refs:
        if ref == "torch_float32":
            t = fpn_tensors(feats["torch_float32"], torch.float32)
        elif ref == "float64":
            t = fpn_tensors([np.asarray(f, np.float64) for f in feats["float64"]], torch.float64)
        elif ref == "ee_mirror":
            t = MF.fpn(feats["ee_mirror"], S.decoder_parameters())
        else:
            t = fpn_tensors([np.asarray(f, np.float32) for f in feats["ee_mirror"]], torch.float32)
        if keep is not None:
            t = {name: value for name, value in t.items() if name in keep}
        out[ref] = t
    return out
