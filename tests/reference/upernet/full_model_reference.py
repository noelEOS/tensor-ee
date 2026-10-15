"""Milestone 11, Phases 1, 13-14: the installed SMP Swin-T + UPerNet forward, end to end.

The authority is `smp.UPerNet.forward` itself (segmentation_models_pytorch 0.5.0,
base/model.py `SegmentationModel.forward`: features = encoder(x); decoder_output =
decoder(features); masks = segmentation_head(decoder_output); return masks), run on the
SegWater v3-preprocessed Sentinel-1 crop with Milestones 5-6's encoder parameters and
Milestone 7's seeded decoder + head (smp_feature_interface.make_upernet).

`capture` runs that forward once with hooks and keeps the tensors it actually computes
(their own strides and memory formats, Milestone 8's lesson): the input, the patch
embedding, C1-C4 at the decoder boundary, the four feature norms, the PSP output, the FPN
states F3 / F2 / F1 and leaves P3 / P2 / P1, the fusion block's input and output, the
head's input, its Conv2d output, its UpsamplingBilinear2d output, its activation output
and the model's return value.  `expanded` recomputes the whole forward from the installed
submodules one operation at a time (Milestone 10's expanded decoder, then the head written
out: Conv2d, F.interpolate(scale_factor=4, bilinear, align_corners=True), identity) and
raises unless every tensor equals the captured one bit for bit, the logits included.

Moved from ``upernet/reference/full_model_reference.py`` at L06b (test support, ``capture``
and ``expanded`` only). The per-crop references it cached under the research
``results/raw/reference_cache/<crop>_head/<reference>/`` were archived with the research code:

    torch_float32               the captured float32 forward (authoritative)
    float64                     the same model in float64 (its own float64 align_corners
                                coefficients, which differ from the float32 model's)
    ee_mirror                   the EE mirror: Milestone 10's mirror pyramid / fusion on
                                the C1-C4 values EE reads, then ee_mirror_head
    torch_float32_same_input    SMP float32 decoder + head on the C1-C4 values EE reads
    torch_float32_head_input    SMP's float32 head alone on EE's fused feature (= the
                                mirror's, bit for bit): the head's own float32 error
    float64_head_input          the head on EE's fused feature in float64 with the float32
                                model's coefficients, unrounded: EE's arithmetic error alone
"""

from __future__ import annotations


import torch
import torch.nn.functional as F

from reference.upernet import smp_fpn_reference as FR
from reference.upernet import smp_fusion_reference as R


# ---------------------------------------------------------------------------
# The real forward, observed
# ---------------------------------------------------------------------------
def capture(model, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """model(x) with hooks: {name: the tensor the forward computed (detached clone, its
    strides kept)} for CAPTURED, plus "layouts": {name: (shape, stride, contiguous NCHW,
    contiguous channels-last)}."""
    cap: dict = {}
    hooks = []

    def keep(name, value):
        cap[name] = value.detach().clone(memory_format=torch.preserve_format)

    def after(name, pick=lambda a, o: o):
        return lambda module, args, output: keep(name, pick(args, output))

    d, head = model.decoder, model.segmentation_head
    hooks.append(model.encoder.model.patch_embed.register_forward_hook(after("patch_embed")))

    def encoder_hook(module, args, output):
        for k in range(4):
            keep(f"C{k + 1}_raw", output[k + 2])

    hooks.append(model.encoder.register_forward_hook(encoder_hook))
    for k, norm in enumerate(d.feature_norms):
        hooks.append(norm.register_forward_hook(after(f"C{k + 1}_norm")))
    hooks.append(d.psp.register_forward_hook(after("psp_relu")))
    for i, k in enumerate(FR.LEVELS):
        hooks.append(d.fpn_lateral_blocks[i].register_forward_hook(after(f"F{k}")))
        hooks.append(d.fpn_conv_blocks[i].register_forward_hook(after(f"P{k}")))
    hooks.append(d.fusion_block.register_forward_pre_hook(lambda m, a: keep("fusion_concat", a[0])))
    hooks.append(d.fusion_block.register_forward_hook(after("fusion_relu")))
    hooks.append(d.register_forward_hook(after("decoder_output")))
    hooks.append(head[0].register_forward_pre_hook(lambda m, a: keep("head_input", a[0])))
    hooks.append(head[0].register_forward_hook(after("head_conv")))
    hooks.append(head[1].register_forward_hook(after("head_upsampled")))
    hooks.append(head[2].register_forward_hook(after("head_activation")))
    try:
        with torch.no_grad():
            keep("input", x)
            keep("logits", model(x))
    finally:
        for h in hooks:
            h.remove()
    cap["layouts"] = {n: {"shape": list(t.shape), "stride": list(t.stride()),
                          "contiguous": t.is_contiguous(),
                          "channels_last": t.ndim == 4 and t.is_contiguous(memory_format=torch.channels_last)}
                      for n, t in cap.items() if isinstance(t, torch.Tensor)}
    return cap


def head_steps(head, fused: torch.Tensor) -> dict[str, torch.Tensor]:
    """The installed head written out: Conv2d (with bias), the upsample as the explicit
    F.interpolate(scale_factor=4, bilinear, align_corners=True) call, the activation.
    Raises unless each written step equals its module and the whole equals the head."""
    with torch.no_grad():
        conv = head[0](fused)
        up = F.interpolate(conv, scale_factor=head[1].scale_factor, mode="bilinear", align_corners=True)
        if not torch.equal(up, head[1](conv)):
            raise AssertionError("F.interpolate(align_corners=True) differs from the head's upsampler")
        act = head[2](up)
        if not torch.equal(act, up):
            raise AssertionError("the head's activation is not the identity")
        if not torch.equal(act, head(fused)):
            raise AssertionError("expanded head differs from SegmentationHead.forward")
    return {"head_conv": conv, "head_upsampled": up, "head_activation": act}


def expanded(model, x: torch.Tensor, cap: dict | None = None) -> dict[str, torch.Tensor]:
    """The whole forward from the installed submodules; raises unless every tensor equals the
    captured forward (cap; captured here when None) bit for bit, the logits included."""
    cap = capture(model, x) if cap is None else cap
    with torch.no_grad():
        feats = model.encoder(x)
    out = {f"C{k + 1}_raw": feats[k + 2] for k in range(4)}
    out.update(R.expanded(model.decoder, feats[2:], check=False))
    out.update(head_steps(model.segmentation_head, out["fusion_relu"]))
    out["logits"] = out["head_activation"]
    checks = [n for n in ("C1_raw", "C2_raw", "C3_raw", "C4_raw", "C1_norm", "C2_norm", "C3_norm",
                          "C4_norm", "psp_relu", "F3", "F2", "F1", "P3", "P2", "P1", "fusion_concat",
                          "fusion_relu", "head_conv", "head_upsampled", "head_activation", "logits")]
    for name in checks:
        if not torch.equal(out[name], cap[name]):
            raise AssertionError(f"expanded {name} differs from the captured model.forward")
    for a, b in (("decoder_output", "fusion_relu"), ("head_input", "fusion_relu"),
                 ("logits", "head_activation")):
        if not torch.equal(cap[a], cap[b]):
            raise AssertionError(f"captured {a} != {b}")
    return out


# ---------------------------------------------------------------------------
# Head-only references
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Cached references
# ---------------------------------------------------------------------------
