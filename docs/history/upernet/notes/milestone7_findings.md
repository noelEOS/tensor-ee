# Findings: the UPerNet decoder foundation in native Earth Engine

Live experiments on 5-6 October 2026: Earth Engine Python API 1.7.45, PyTorch 2.14.0,
timm 1.0.30, segmentation_models_pytorch 0.5.0, NumPy 2.5.3, Cloud project
`ORIGINAL_PROJECT`, application-default credentials. Every Earth Engine tensor was computed
in EE graphs from EE assets (Milestone 6's real Sentinel-1 encoder outputs, this
milestone's feature and weight assets) or from fixtures embedded in the request; tensors
were downloaded only to compare them.

Evidence: numerical_comparison_m7.csv (development archive: `docs/history/upernet/results/numerical_comparison_m7.csv`)
(summary: numerical_summary_m7.md (development archive: `docs/history/upernet/results/numerical_summary_m7.md`)),
benchmark_results_m7.csv (development archive: `docs/history/upernet/results/benchmark_results_m7.csv`) (summary:
benchmark_summary_m7.md (development archive: `docs/history/upernet/results/benchmark_summary_m7.md`)), per-request records in
results/raw/ (archive `upernet/results/raw/` at `research-archive-2026-10-12`) (`validation_*`, `bench/`, `feature_assets.json`,
`feature_asset_sizes.json`, `constant_budget.json`, `logs/`), the contract in
data/manifest.json (archive `upernet/data/manifest.json` at `research-archive-2026-10-12`), Phase 2 statistics in
data/interface_stats.json (archive `upernet/data/interface_stats.json` at `research-archive-2026-10-12`), the weight assets in
data/weights_manifest.json (archive `upernet/data/weights_manifest.json` at `research-archive-2026-10-12`).

**Research questions.** (Q1) Can the M6 encoder's outputs be exposed exactly as the
installed UPerNet consumes them? (Q2) Do the decoder's primitives reproduce the reference
in native EE at realistic widths and scales? (Q3) How must C1-C4 travel between EE tasks
so that the future decoder stays under EE's limits? **Yes, yes, and: task A' writes C1,
C2, C3 into one asset, task B' writes C4, and the decoder needs its own task C.**

- **The contract, from the installed code.** SMP's Swin-T encoder is timm's
  `FeatureListNet` (`patch_embed`, `layers_0`..`layers_3`; timm's final `norm` is not in
  it): C1-C4 are the four stage outputs, bit-identical to Milestone 6's `s1_out`..`s4_out`.
  **C4 is the raw stage-4 output, not the final LayerNorm** (they differ by up to 7.6).
  UPerNet then applies its own `LayerNorm2d(eps=1e-6)` to each.
- **Real features, every element.** The canonical 896 × 896 Sentinel-1 crop's C1-C4
  (9.0 M values) are bit-identical to the EE mirror and within **1.55e-5** of SMP's
  float32 tensors at the decoder boundary; EE's LayerNorm2d of each is within **5.7e-6**
  of SMP's `feature_norms`, bit-identical to the mirror. Grids and footprints exact.
- **Every decoder primitive matches.** 1 × 1 and 3 × 3 Conv2d (six 3 × 3 formulations;
  kernel, channel and zero-padding order exact on unique IDs), LayerNorm2d at 96-768,
  inference BatchNorm (explicit and folded), ReLU, bilinear resize
  (`align_corners=False`; four formulations, every edge), addition and concatenation:
  every float result bit-identical to the mirror, ≤ 8.1e-6 from PyTorch float32.
- **The decoder should compute in window layout.** Dense per-pixel arrays cost 3-6× the
  EECU·s of Milestone 5's window layout for the same matrix products, and turning wide
  per-pixel arrays back into bands exhausts interactive memory at C1 scale. A
  window-native 3 × 3 convolution (halo from the 8 neighbouring windows) costs
  102-168 EECU·s for 256 → 256 at the canonical C1 scale; window-layout bilinear resize
  4-8. Dense georeferenced rasters are the format *between* tasks.
- **Transport.** Exporting C1, C2 and C3 in one task costs what Milestone 6's task A cost
  for C3 alone (308 vs 309 EECU·s); C4 follows from it (62 EECU·s). The decoder
  (9.23 M parameters, 9.85 M embedded values) cannot join task B (16.57 M): together
  26.4 M, over the 24-26 M cap, and the real task-B' graph carrying the decoder's
  constants fails in batch with "Encoded object too large."; the decoder's constants alone
  pass. Its fusion conv (1024 → 256 at C1) is batch-only (602 EECU·s, im2col).

**Milestone verdict.** **Minimum, strong and full success are all met.** The completed Milestone 6 encoder
delivers the exact C1-C4 that SMP's UPerNet consumes (contract read from the installed
code and proven bit for bit), entirely in Earth Engine, georeferenced and numerically
equal to SMP within 1.55e-5 (prepared: 5.7e-6). Every fundamental decoder primitive
reproduces PyTorch at the decoder's real widths and scales, with bit-identical agreement
to the EE mirror and exact layout operations; asset-backed decoder parameters round-trip
bit for bit; a representative chain of the primitives on the real C4 and C3 matches the
installed modules within 1.48e-5; and the A' → B' → C topology is demonstrated live, with
the decoder's constant budget measured against the cap rather than estimated. Nothing
leaves Earth Engine for inference.

## Validation matrix (Phase 23)

Against PyTorch / SMP float32 (layout rows: against the exact expected values); every row passes (max ≤ 1e-4 for floating point, exact for layout).

| operation | MAE | RMSE | max abs | EE vs mirror |
|---|---:|---:|---:|---|
| C1 decoder-interface layout (task A'/B' asset vs Milestone 6's export, window → dense → window) | 0 | 0 | 0 | exact |
| C2 decoder-interface layout (task A'/B' asset vs Milestone 6's export, window → dense → window) | 0 | 0 | 0 | exact |
| C3 decoder-interface layout (task A'/B' asset vs Milestone 6's export, window → dense → window) | 0 | 0 | 0 | exact |
| C4 decoder-interface layout (task A'/B' asset vs Milestone 6's export, window → dense → window) | 0 | 0 | 0 | exact |
| C1 feature normalization (real, LayerNorm2d) | 2.4e-07 | 3.2e-07 | 3.9e-06 | bit-identical |
| C2 feature normalization (real, LayerNorm2d) | 3.5e-07 | 4.5e-07 | 3.9e-06 | bit-identical |
| C3 feature normalization (real, LayerNorm2d) | 4.8e-07 | 6.2e-07 | 4.9e-06 | bit-identical |
| C4 feature normalization (real, LayerNorm2d) | 6.5e-07 | 8.4e-07 | 5.7e-06 | bit-identical |
| 1×1 Conv channel order (unique IDs) | 0 | 0 | 0 | exact |
| 1×1 Conv 96 → 256 | 1.2e-07 | 1.8e-07 | 1.2e-06 | bit-identical |
| 1×1 Conv 192 → 256 | 1.7e-07 | 2.5e-07 | 2.1e-06 | bit-identical |
| 1×1 Conv 384 → 256 | 2.5e-07 | 3.5e-07 | 2.5e-06 | bit-identical |
| 1×1 Conv 768 → 256 | 3.5e-07 | 5e-07 | 4.5e-06 | bit-identical |
| 3×3 Conv ordering (unique IDs, 6 formulations) | 0 | 0 | 0 | exact |
| 3×3 Conv tiny 2 → 3 (window im2col) | 8.7e-08 | 1.4e-07 | 9.5e-07 | bit-identical |
| 3×3 Conv 256 → 256 (window im2col) | 5.5e-07 | 7.9e-07 | 8.1e-06 | bit-identical |
| BatchNorm (inference) | 3.7e-08 | 6.9e-08 | 9.5e-07 | bit-identical |
| Conv + BatchNorm | 5.5e-07 | 8.1e-07 | 7.6e-06 | bit-identical |
| folded Conv' (+ ReLU) | 2.8e-07 | 5.8e-07 | 7.9e-06 | bit-identical |
| activation (ReLU, edge values) | 0 | 0 | 0 | exact |
| bilinear, window 8× (7×7 → 56×56) | 1.1e-08 | 2.9e-08 | 4.8e-07 | bit-identical |
| bilinear, real C4 → C3 (chain u) | 2.6e-07 | 4.8e-07 | 4.5e-06 | bit-identical |
| channel concatenation (4 × 256, IDs) | 0 | 0 | 0 | exact |
| elementwise addition (256) | 0 | 0 | 0 | bit-identical |
| **representative primitive chain (real C4, C3 → p3)** | 4.9e-07 | 9.8e-07 | 1.5e-05 | bit-identical |

## Answers

| # | Question | Answer |
|---|---|---|
| 1 | What exact four tensors does the Swin-T encoder provide to UPerNet? | **timm `FeatureListNet` outputs `layers_0`..`layers_3`**, NHWC permuted to NCHW by SMP's `TimmUniversalEncoder`: the outputs of stages 1-4 (C2-C4 after their stage's Patch Merging and blocks), bit-identical to Milestone 6's `s1_out`..`s4_out`. The encoder list also holds the input and an empty 1/2-scale placeholder, which UPerNet skips (§1). |
| 2 | Raw Stage-4 output or the M6 final normalized output? | **The raw stage-4 output.** `FeatureListNet` has no `norm`; C4 ≠ final norm (up to 7.59 apart). UPerNet normalizes C4 itself with `LayerNorm2d(768, eps=1e-6)` and its own parameters (§1). |
| 3 | Are C1-C4 shapes/channels as expected? | **Yes**: (1, 96, H/4, W/4), (1, 192, H/8, W/8), (1, 384, H/16, W/16), (1, 768, H/32, W/32), float32, contiguous NCHW (§1). |
| 4 | Are the real-image decoder-facing values correct? | **Yes**: canonical 896 crop, every element (9.0 M), bit-identical to the mirror, ≤ 1.55e-5 from SMP float32 (C3), from both Milestone 6's assets and this milestone's transport (§2, §5). |
| 5 | Are the four EE grids aligned correctly? | **Yes**: EPSG:32651, the crop's origin, 40/80/160/320 m; asset metadata of every band exact; 40/40 native-pixel footprints per feature (§2). |
| 6 | Is a window→dense conversion needed beyond M6's export path? | **No.** Milestone 6's export (depth_to_space + relabel) and read-back (49 exact stride samples) are the only conversions; unique IDs survive window → dense → window exactly (§2). |
| 7 | Where should the permanent window → raster transition occur? | **Nowhere inside a task.** The decoder computes in window layout too (cheaper for every primitive; dense per-pixel arrays cost 3-6× and wide-array flattening exhausts memory); dense georeferenced rasters are the at-rest format, converted once at each task boundary and at the final output (§4). |
| 8 | What feature normalization does UPerNet use? | **`LayerNorm2d(C, eps=1e-6)`** per feature: channels-last `F.layer_norm` over C, biased variance, affine (§1). |
| 9 | Can the existing LayerNorm primitive be reused? | **Yes, unchanged** (Milestone 2's `layer_norm` with eps 1e-6, rows 49) (§2). |
| 10 | Maximum feature-normalization error? | **5.72e-6** (real C4 prepared, vs SMP float32; 8.2e-7 vs float64); fixtures ≤ 9.5e-7; bit-identical to the mirror everywhere (§2). |
| 11 | Can 1×1 Conv2d reuse the Linear primitive? | **Yes**: Conv2d weight [Cout, Cin, 1, 1] → [Cout, Cin] → Milestone 2's `linear` (W^T [Cin, Cout]); channel order exact (permutation IDs); ≤ 4.5e-6 at 768 → 256 (§3). |
| 12 | Can 3×3 Conv2d be implemented efficiently in native EE? | **Yes, in window layout**: the 9 × 9 halo from the 8 neighbouring windows, then im2col or 9 tap products. 256 → 256 at the canonical C1 scale (224² tokens): 102-168 EECU·s, 638 MiB, interactive (§3, §4). |
| 13 | Maximum Conv2d error? | **8.11e-6** (256 → 256 fixture, vs float32; 1.3e-7 vs float64); chain 1.05e-5 for the real chain's 3 × 3 conv (1.48e-5 after its BatchNorm) (§3, §9). |
| 14 | How is PyTorch padding reproduced exactly? | **Explicit zeros**: outside the raster the window layout holds exact zero arrays (built on the server), which fill the halo ring; dense formulations `unmask(0)` before gathering. Unique IDs prove every border position (§3). |
| 15 | Most efficient 3×3 formulation tested? | **Window layout, im2col**: im2col and taps are equal interactively at C1 (102-168 EECU·s), but only im2col scales well in batch (448²: 586 vs 1,458; the 1024-channel fusion conv: 602 vs 1,045); window layout is ~1.6× cheaper than stride-sampling a dense im2col into windows, ~3.4× than 9 dense tap products, ~5.7× than `neighborhoodToArray`; `Image.convolve` needs Cin × Cout kernels (65,536 at 256 → 256) (§4). |
| 16 | What normalization/activation blocks does the decoder use? | **`Conv2dReLU`: Conv2d(bias=False) → BatchNorm2d(eps 1e-5) → ReLU(inplace)** for every PSP, FPN and fusion conv; LayerNorm2d on the inputs; the head is Conv2d(256 → 2, bias) → bilinear ×4 (align_corners=True) (§1). |
| 17 | Can inference BatchNorm be folded safely? | **Yes, within float32 rounding, not bit for bit**: folded vs explicit EE 2.4e-7 max (75% bit-equal), both ≤ 7.9e-6 from PyTorch; folding inside EE from the unfolded assets gives the client fold's bits (§3). |
| 18 | Does folding materially improve EE execution? | **No.** Conv + BN + ReLU 256 → 256 at the canonical C1 scale (asset weights, 2 runs each): explicit 133-187 EECU·s, folded (inside EE) 145-149, the same 638 MiB peak (the halo assembly) and graph (37.6 vs 37.3 KB). Folding saves 37,632 embedded values per block, 0.34 M for the decoder (3.4% of task C), and costs bit-equality with the checkpoint's explicit form. Keep BatchNorm explicit; fold only if the constant budget ever binds (§3, §9). |
| 19 | Can EE reproduce UPerNet's bilinear convention exactly? | **Yes.** `align_corners=False` is pixel-centre interpolation on a shared origin; the window formulation (PyTorch's own weights) is bit-identical to the exact value rounded once, every fixture and edge; EE's own `resample("bilinear")` matched too on every fixture and on the real UTM grids (bit-identical to the window formulation at all 802,816 values of the chain's C4 → C3 resize) (§3). |
| 20 | If native resampling differs, what explicit implementation is required? | The production path is explicit anyway: **window-layout resize with PyTorch's weight matrices** (per-axis [7, 9] blocks on the 9 × 9 halo, two small products), plus a **matrix** form for arbitrary sizes (PSP: 1/2/3/6 → H/32) (§3). |
| 21 | Do resizes land on the exact geospatial target grids? | **Yes**: the C4 → C3 resize of the chain has the C3 grid's projection exactly, 40/40 footprints, and equals PyTorch's interpolation of the real tensors (4.5e-6, bit-identical to the mirror) (§9). |
| 22 | What dominates decoder-primitive memory? | **The halo assembly of 3 × 3 convolutions** (zero-padded window arrays gathered from 8 neighbours: 638 MiB for 256 channels at the canonical C1 grid), scaling with channels × windows, so the 1024-channel fusion conv is the memory peak; in dense layouts, flattening wide per-pixel arrays (≥ 601 MiB already at 56²). Decoded weights add ~64 bytes per value (566 MiB for all 9.23 M) (§4, §9). |
| 23 | What dominates decoder-primitive compute? | **The 3 × 3 convolutions, above all the fusion conv**: 118 of 163 GMAC per canonical crop (73%), 602 EECU·s in batch (im2col; 1,045 with taps), ~1.6× the whole encoder; FPN 3 × 3 at C1 next (~100-170). LayerNorm2d, 1 × 1, resize, addition, concatenation and feature reads are 1-10 EECU·s each (§9). |
| 24 | How expensive is exposing/persisting C1-C4? | **Task A' 307.6 EECU·s for all three skips** (= Milestone 6's task A for C3 alone), one 30.7 MB, 673-band asset; **task B' 61.7 EECU·s** for C4; separate exports would cost 512 + 68. Reading them back costs 7.2 EECU·s (all four, one request) (§5). |
| 25 | What feature-asset/task topology should M8 use? | **A' (skips asset) → B' (C4 asset) → C (decoder)**, all EE, all exact (§5, §6). |
| 26 | Decoder parameters per subsystem? | Feature norms 2,880; PSP pooled 1 × 1 790,528; PSP 3 × 3 4,129,792; FPN lateral 175,104; FPN 3 × 3 1,772,544; fusion 2,360,320; head 514: **9,231,682** (§1). |
| 27 | Could the decoder fit in Task B under the cap? | **No**: task B' + decoder embeds 26.43 M (26.09 M folded) and, run live, the real task B' graph carrying them fails with "Encoded object too large." (26.38 M), while the decoder's constants alone (9.81 M) and task B' + PSP (21.73 M) pass (§6). |
| 28 | How many decoder tasks are likely required? | **One** (task C: 9.85 M embedded, 41% of the cap; ~545 MiB decoded weights) (§6). |
| 29 | Do asset-backed decoder parameters round-trip correctly? | **Yes, all 9,231,682 values bit for bit** (19 table assets, 131 MB payload, 27.5 min upload); one request reads them all in 13.0 EECU·s, 4.1 s, 566 MiB (§9). |
| 30 | Does the mirror explain the numerical differences? | **Yes, completely**: every compared EE float tensor (fixtures, real features, transport, chain) is bit-identical to the mirror, so EE − PyTorch is PyTorch's float32 rounding (§3, §9). |
| 31 | Largest realistic decoder feature map run interactively? | **The canonical C1 grid, 224 × 224 × 256 (1,024 windows)**, through a 256 → 256 3 × 3 conv (638 MiB); 448 × 448 (the 1792 crop's C1) exceeds interactive memory and runs in batch (586 EECU·s, im2col) (§4, §9). |
| 32 | Are required primitives batch-only at canonical scale? | **Yes, one**: the fusion conv (1024 → 256, 3 × 3 at the C1 grid) exceeds interactive memory at the canonical size and runs in batch (602 EECU·s with im2col, 1,045 with taps); every other primitive runs interactively at the canonical size. At the 1792 crop's C1 grid (448²) the 256 → 256 conv is batch-only too (586 EECU·s) (§9). |
| 33 | Is graph complexity manageable? | **Yes**: primitives 10-60 KB and 12-80 invocations; task A' 842 KB / 766, task B' 360 KB / 379; the chain ≤ 202 KB per request; no "too complex" anywhere (§9). |
| 34 | How should valid-fraction metadata be carried forward? | **As a separate band of the skips asset** (C1 grid), coarser grids by exact means; exact on both crops; never a decoder input; to be propagated to the outputs as QC (§7). |
| 35 | How much local receptive field do the primitives add? | 3 × 3 at C_k: ±4/8/16/32 px; 2 × bilinear: one source pixel (±8/16/32 px); pointwise 0; the chain: ±48 px, but across a stage-4 window boundary, so the aligned real-context halo grows **672 → 896 px**; PSP pooling is global (§8). |
| 36 | EE semantics that could invalidate PSP/FPN composition? | Implicit resampling of misaligned operands (guarded); one grid per export (packing); nearest reprojection ambiguous at coarse pixel centres (exact stride samples instead); `resample` renormalizes over masks (define domains explicitly); PSP's global pooling makes tiling non-equivalent; JSON integral constants parse as int64 (§10). |
| 37 | Should the project proceed to M8 (PSP)? | **Yes** (§11). |

## 1. The installed model and its encoder → decoder contract (Phases 1-3, 6; Tests 1-2)

Everything below was established by running the installed packages
(reference/smp_feature_interface.py (archive `upernet/reference/smp_feature_interface.py` at `research-archive-2026-10-12`), recorded in
data/manifest.json (archive `upernet/data/manifest.json` at `research-archive-2026-10-12`); tests in
test_upernet_contract.py (archive `tests/test_upernet_contract.py` at `research-archive-2026-10-12`)). The repository had no
UPerNet yet, so the model is built the way the project builds its SegWater U-Net:

    smp.UPerNet("tu-swin_tiny_patch4_window7_224", encoder_weights=None, encoder_depth=5,
                decoder_channels=256, decoder_use_norm="batchnorm", in_channels=2,
                classes=2, upsampling=4, img_size=(H, W))

| item | installed source (smp 0.5.0, timm 1.0.30, torch 2.14.0) |
|---|---|
| UPerNet | `segmentation_models_pytorch/decoders/upernet/{model,decoder}.py` |
| encoder wrapper | `encoders/timm_universal.py` `TimmUniversalEncoder` (transformer style: reductions 4, 8, 16, 32) |
| timm module | `timm/models/_features.py` `FeatureListNet`: `patch_embed`, `layers_0` .. `layers_3`; **no `norm`** |
| encoder list | `[x (1,2,H,W), empty (1,0,H/2,W/2), C1, C2, C3, C4]`; the decoder takes `features[2:]` |
| C1 / C2 / C3 / C4 | outputs of `layers_0..3` (feature_info modules `layers.0..3`), NHWC permuted to NCHW (`.contiguous()`), float32 |
| shapes | (1, 96, H/4, W/4), (1, 192, H/8, W/8), (1, 384, H/16, W/16), (1, 768, H/32, W/32) |
| feature normalization | `LayerNorm2d(C, eps=1e-6)` per feature (SMP's own class: permute to NHWC, `F.layer_norm` over C, permute back; affine) |
| conv blocks | `Conv2dReLU` = Conv2d(bias=False) → BatchNorm2d(eps 1e-5) → ReLU(inplace) everywhere (PSP 4 × 1 × 1 768→256 and 3 × 3 1792→256; FPN lateral 1 × 1 384/192/96→256; FPN 3 × 3 256→256 × 3; fusion 3 × 3 1024→256) |
| interpolation | every decoder resize is `F.interpolate(size=…, mode="bilinear", align_corners=False)`: PSP 1/2/3/6 → H/32 (4 calls), FPN 2 × (3), fusion to H/4 (4: 8 ×, 4 ×, 2 ×, 1 ×) |
| segmentation head | Conv2d(256 → 2, 1 × 1, bias) → `UpsamplingBilinear2d(scale_factor=4)` (**align_corners=True**, a different convention, out of scope) → identity activation |
| normalization classes | BatchNorm2d, LayerNorm2d (nothing else) |

**Where each feature is taken.** timm's stage k ≥ 2 begins with its Patch Merging, so
`layers_{k-1}` returns the stage-k blocks' output: C1 is after stage 1 (no merge), C2-C4
after Patch Merging k-1 *and* stage k's blocks. Swin-T has no per-stage norm; the wrapper
only permutes. Checked numerically, not read off: on a random 224² input and on the real
canonical crop, in float32 and float64, each C_k equals Milestone 6's explicit encoder
output `s{k}_out` **bit for bit**, and C4 differs from the final norm by up to 7.59. A
pre-hook on the decoder captures exactly the list it receives; forward hooks on
`feature_norms` capture C1-C4 prepared, and a manual call of the norms reproduces them
bit for bit.

**The real tensors (Phase 2), canonical crop, SMP float32 forward**
(data/interface_stats.json (archive `upernet/data/interface_stats.json` at `research-archive-2026-10-12`)):

| feature | shape | raw min / max / mean / std | prepared min / max / mean / std |
|---|---|---|---|
| C1 | 224 × 224 × 96 | −10.81 / 10.21 / −0.127 / 2.44 | −4.67 / 4.83 / 0.001 / 0.99 |
| C2 | 112 × 112 × 192 | −10.59 / 10.37 / 0.313 / 2.63 | −5.85 / 4.01 / 0.004 / 1.02 |
| C3 | 56 × 56 × 384 | −15.23 / 14.23 / −0.372 / 4.16 | −4.13 / 3.88 / −0.004 / 1.02 |
| C4 | 28 × 28 × 768 | −11.12 / 9.39 / 0.019 / 2.57 | −4.87 / 5.29 / −0.011 / 1.05 |

**The contract (Phase 3)**, canonical crop (`canonical_896`, origin
(283200.95335071225, 2645318.8405420803) in EPSG:32651):

| feature | source tensor | channels | grid | EE CRS / transform | pixel (0, 0) centre |
|---|---|---:|---|---|---|
| C1 | `layers_0` = s1_out | 96 | H/4 × W/4 = 224² | EPSG:32651, [40, 0, x0, 0, −40, y0] | (x0 + 20, y0 − 20) |
| C2 | `layers_1` = s2_out | 192 | H/8 × W/8 = 112² | EPSG:32651, [80, 0, x0, 0, −80, y0] | (x0 + 40, y0 − 40) |
| C3 | `layers_2` = s3_out | 384 | H/16 × W/16 = 56² | EPSG:32651, [160, 0, x0, 0, −160, y0] | (x0 + 80, y0 − 80) |
| C4 | `layers_3` = s4_out | 768 | H/32 × W/32 = 28² | EPSG:32651, [320, 0, x0, 0, −320, y0] | (x0 + 160, y0 − 160) |

Layout: NCHW at the decoder; in EE a dense georeferenced raster has bands c0..c{C−1}
(channel order = PyTorch's channel axis) and pixel (row, col) = NCHW (h, w); in compute
the same values sit in window layout (§4). Normalization state: raw (UPerNet normalizes).
Mask policy: dense, unmasked inside the crop; values outside the grid are never read
implicitly (convolutions pad with explicit zeros, resizes clamp through their weights).
Validity metadata: a separate `valid_fraction` band (§6), never a model input.

**Parameters by subsystem (Phase 6):**

| subsystem | conv weights | BatchNorm / LayerNorm values | total |
|---|---:|---:|---:|
| feature norms (LayerNorm2d × 4) | — | 2,880 | 2,880 |
| PSP pooled 1 × 1 (768 → 256) × 4 | 786,432 | 4,096 | 790,528 |
| PSP output 3 × 3 (1792 → 256) | 4,128,768 | 1,024 | 4,129,792 |
| FPN lateral 1 × 1 (384/192/96 → 256) | 172,032 | 3,072 | 175,104 |
| FPN 3 × 3 (256 → 256) × 3 | 1,769,472 | 3,072 | 1,772,544 |
| fusion 3 × 3 (1024 → 256) | 2,359,296 | 1,024 | 2,360,320 |
| segmentation head (256 → 2) | 512 | 2 (bias) | 514 |
| **decoder + head** | **9,216,512** | 15,170 (6,144 running stats) | **9,231,682** |

One third of Swin-T's encoder (27.5 M); 45% of it is the PSP output conv, 26% the fusion conv.

## 2. Real decoder-facing features (Phases 4, 7, 19; Tests 3-6)

validate.py (archive `upernet/earth_engine/validate.py` at `research-archive-2026-10-12`) `interface`: Milestone 6's canonical EE assets
(the real Sentinel-1 encoder outputs) against SMP's tensors at the decoder boundary, every
element; then EE's LayerNorm2d of each (read back into window layout) against SMP's
`feature_norms` in the real forward pass.

| tensor | elements | MAE | RMSE | max \|EE − torch32\| | max \|EE − float64\| | EE vs mirror |
|---|---:|---:|---:|---:|---:|---|
| C1 (decoder input) | 4,816,896 | 5.9e-7 | 7.6e-7 | 7.87e-6 | 2.0e-6 | bit-identical |
| C2 | 2,408,448 | 9.1e-7 | 1.2e-6 | 8.23e-6 | 1.6e-6 | bit-identical |
| C3 | 1,204,224 | 2.0e-6 | 2.5e-6 | 1.55e-5 | 3.2e-6 | bit-identical |
| C4 | 602,112 | 1.7e-6 | 2.1e-6 | 1.14e-5 | 1.4e-6 | bit-identical |
| C1 prepared (LayerNorm2d 96) | 4,816,896 | 2.4e-7 | 3.2e-7 | 3.87e-6 | 1.0e-6 | bit-identical |
| C2 prepared (192) | 2,408,448 | 3.5e-7 | 4.5e-7 | 3.93e-6 | 7.4e-7 | bit-identical |
| C3 prepared (384) | 1,204,224 | 4.8e-7 | 6.2e-7 | 4.89e-6 | 9.1e-7 | bit-identical |
| C4 prepared (768) | 602,112 | 6.5e-7 | 8.4e-7 | 5.72e-6 | 8.2e-7 | bit-identical |

Asset metadata (every band: CRS, transform, dimensions, name, float) equals the contract
grids, and 40 of 40 random 10 m native pixel centres per feature sample the feature pixel
(row // stride, col // stride). **LayerNorm2d is Milestone 2's `layer_norm` unchanged with
eps = 1e-6**: biased variance over the channel axis, gamma and beta affine, statistics and
normalization in double, one float32 rounding; PyTorch rounds inside (its float32 kernel
differs from float64 by up to 4.8e-7 on its own). On fixtures at C = 96, 192, 384, 768
(7 × 7 windows, offset channels): ≤ 9.5e-7 from torch float32, ≤ 2.2e-7 from float64,
bit-identical to the mirror.

**Window → dense (Phase 4).** No new conversion was needed: Milestone 6 already turns
each stage output from window layout into a georeferenced token raster once
(depth_to_space, relabel) when it exports it, and reads it back with 49 exact stride
samples. Both are pure moves: unique (token, channel) IDs go window → dense → window
unchanged (live Test 3), and every transported tensor is bit-identical to the mirror.

## 3. The primitives (Phases 7-16, 18; Tests 6-16)

Fixtures in validate.py (archive `upernet/earth_engine/validate.py` at `research-archive-2026-10-12`) `fixtures`, against PyTorch / SMP
modules (reference/pytorch_decoder_primitives.py (archive `upernet/reference/pytorch_decoder_primitives.py` at `research-archive-2026-10-12`))
and the mirror (reference/ee_mirror_decoder.py (archive `upernet/reference/ee_mirror_decoder.py` at `research-archive-2026-10-12`)):

| operation | elements | max \|EE − torch32\| | max \|EE − float64\| | EE vs mirror |
|---|---:|---:|---:|---|
| 3 × 3 kernel / channel / padding order (unique IDs, one-hot weights), 6 formulations | 3 × 630 + 3 × 5,292 | 0 | — | exact |
| 3 × 3 conv tiny, Cin 2 → 3 + bias, 6 formulations | 90 / 882 | 9.5e-7 | 2.3e-7 | bit-identical (6/6) |
| 3 × 3 conv 256 → 256 (FPN width), 4 formulations | 50,176 | 8.11e-6 | 1.3e-7 | bit-identical (4/4) |
| 1 × 1 channel order (permutation, IDs) | 4,704 | 0 | — | exact |
| 1 × 1 conv 96 / 192 / 384 / 768 → 256 | 12,544 each | 1.2e-6 / 2.2e-6 / 2.5e-6 / 4.5e-6 | 1.2e-7 | bit-identical |
| 1 × 1 conv 256 → 256; head 256 → 2 + bias | 12,544 / 98 | 2.2e-6 / 9.5e-7 | 1.2e-7 | bit-identical |
| LayerNorm2d 96 / 192 / 384 / 768 | 4,704 … 37,632 | ≤ 9.5e-7 | ≤ 2.2e-7 | bit-identical |
| inference BatchNorm2d(256) | 50,176 | 9.5e-7 | 2.3e-7 | bit-identical |
| Conv → BN → ReLU (SMP `Conv2dReLU` 256 → 256, step by step) | 50,176 | 7.15e-6 / 7.63e-6 / 7.63e-6 | 2.8e-7 | bit-identical |
| the same block with BN folded | 50,176 | 7.87e-6 | 2.4e-7 | bit-identical; vs explicit EE 2.4e-7 (75% equal) |
| ReLU on ±0, subnormals, ±6.1e37 | 10 | 0 | — | exact (values; sign of zero below) |
| bilinear, dense / native, 2 × 2 → 4 × 4, 3 × 3 → 6 × 6, odd 5 × 5 → 10 × 10, 3 × 5 → 6 × 10, 4 ×, 8 × | 96 … 2,304 | 1.2e-7 | 6.0e-8 | bit-identical (18/18) |
| bilinear, window, 7 × 14 → 14 × 28, 21 × 21 → 42 × 42, 4 ×, 8 × | 3,136 … 25,088 | 4.8e-7 | 1.2e-7 | bit-identical (4/4) |
| bilinear, matrix, 3 × 5 → 7 × 11, 2 × 2 → 4 × 4, PSP 6 / 3 / 1 → 28 / 28 / 7 | 96 … 4,704 | 7.6e-6 | 2.2e-6 | bit-identical (5/5) |
| addition (window; dense) | 50,176; 1,568 | 0 | — | bit-identical |
| concatenation 4 × 256 (fusion order) / dense, unique IDs | 200,704; 2,352 | 0 | — | exact |

Every floating-point EE result is bit-identical to the mirror, so every EE − PyTorch
difference is PyTorch's float32 rounding: EE accumulates each dot product sequentially in
double and rounds once (torch's float64 matmul reproduces that order bit for bit, re-checked
at lengths 2,304 and 9,216), torch's float32 kernels round inside.

**Convolution semantics (Q14).** PyTorch's 3 × 3, padding 1 is an im2col product: the
zero-padded neighbourhood in (ky, kx, c) order times the weight permuted
(o, c, ky, kx) → (ky, kx, c, o), applied once to the constant (on the server for asset
weights). Zero padding is explicit: outside the raster the window layout holds exact
zero arrays built on the server; dense formulations `unmask(0)` before gathering. One-hot
weights extract exactly PyTorch's own im2col, every tap, channel and border position.

**BatchNorm.** Explicit inference BN is `(x − mean) / sqrt(var + eps) · γ + β` in double,
rounded once. Folding (W′ = W · γ/√(var+eps), b′ = β − mean · γ/√(var+eps), rounded to
float32 once, the content of a folded checkpoint) is numerically equivalent but not bit
for bit: folded vs explicit EE 2.4e-7 max, both ≤ 7.9e-6 from PyTorch. Folding inside EE
from the unfolded assets (`fold_server`) gives the same float32 bits as folding in NumPy
(live Test 11).

**ReLU and signed zero.** `max(x, 0)` reproduces every value, subnormals and ±6.1e37
included. EE maps −0.0 to +0.0; PyTorch's own ReLU maps −0.0 to +0.0 in its 8-wide
vectorized loop and keeps −0.0 in the scalar tail, so its sign of zero depends on the
element's position: equal as numbers, no effect downstream. NaN does not occur (finite
inputs). A fixture pitfall found on the way: EE parses integral-valued JSON numbers as
int64, so an embedded −0.0 arrives as +0 and |v| ≥ 2⁶³ saturates to ±2⁶³; the edge
values were therefore computed inside EE. Model weights are unaffected (none is integral
and ≥ 2⁶³; ±0 weights give equal sums).

**Bilinear (Q19-Q21).** UPerNet's `align_corners=False` rule samples output pixel o at
source coordinate (o + ½)·n_in/n_out − ½ (clamped at 0; upper neighbour clamped to the
last pixel): the centre of output pixel o in source-pixel units. On a shared origin that
*is* pixel-centre interpolation on the georeferenced grids, so tensor and geospatial
conventions coincide. The EE formulations use PyTorch's own weights, read off
`F.interpolate` with one-hot inputs (equal to the written-out rule): for integer factors
they are dyadic, products with float32 values are exact in double, and EE's result is the
exact bilinear value rounded once. Coordinate ramps (row, column channels) prove the
source coordinate of every output pixel exactly (clamped edges included) on every fixture.

- **window** (production): the parent window's 9 × 9 halo, read at each child window, and
  two small products with the child's [7, 9] weight blocks (slices of PyTorch's matrices;
  the out-of-raster ring tokens get weight 0, which *is* the edge clamp).
- **dense**: replicate-edge ring + 3 × 3 exact translations + per-phase weights on scalar
  bands.
- **matrix** (any sizes, small sources: PSP): out(Y, X) = (M_y[Y] ⊗ M_x[X]) · S with
  PyTorch's float32 1-D weights (non-dyadic for 3 → 28 etc., which is why PyTorch's own
  error is larger there: 7.6e-6).
- **native**: EE's `resample("bilinear")` onto the target grid. **Bit-identical to the
  exact rule on every fixture, edges included, with or without a replicate ring**: with
  the source masked outside its grid, EE renormalizes the weights over the valid
  neighbours, which for integer factors is PyTorch's clamp. On the real UTM grids (the chain's C4 → C3 resize, 320 m → 160 m, EPSG:32651 coordinates near 2.6e6) native, dense and window agree bit for bit (§9).

**Addition and concatenation.** float32 + float32 is correctly rounded by both. EE adds
images on different grids without complaint (the coarse operand is nearest-resampled
onto the request grid: the negative control reproduces nearest replication exactly), so
decoder tensors carry their grid (`Feature`) and mismatched operands are refused.
Concatenation follows `torch.cat(dim=1)`, first operand's channels first; the fusion order
is `[C1-level, C2-level, C3-level, PSP]` (`resized_fpn_features[::-1]`).

## 4. Representation study: dense rasters or window layout? (Phase 10, representation study)

Synthetic inputs and server-generated weights at the canonical feature scales (one
interactive request each, checksum of the output; probe_representation.py (archive `upernet/benchmarks/probe_representation.py` at `research-archive-2026-10-12`)):

| operation (canonical scale) | dense per-pixel arrays | block (dense im2col → windows) | **window layout** |
|---|---|---|---|
| 1 × 1 96 → 256, C1 (224²) | 24-34 EECU·s, 276 MiB | — | **7.2-7.3 EECU·s, 157 MiB** |
| … and back to 256 scalar bands | ✗ memory limit (601 MiB already at 56²) | — | not needed |
| 3 × 3 256 → 256, C3 (56²) | 42 (`neighborhoodToArray`), 29 (9 taps) | 11.9 | 13.1 (im2col), 34.5 (taps) |
| 3 × 3 256 → 256, C1 (224²) | 727 (`neighborhoodToArray`, 948 s), 437 (taps) | 186-236, 1,002 MiB | **102-168, 638 MiB** |
| 3 × 3 256 → 256, 448² (1792 crop) | — | — | ✗ interactive memory (batch: 586 EECU·s im2col, 1,458 taps) |
| bilinear 2 ×, 256 ch., to C1 | 51.7 (dense bands) | — | **8.2-8.4** (native: 11.0-11.3) |
| bilinear 8 ×, 256 ch., C4 → C1 | — | — | **4.2-6.3** |
| `Image.convolve`, 16 → 16 at 56² | 1.0 (256 kernels, 178 KB request) | — | 0.5 (taps) |

EECU·s of identical requests vary up to ~1.5× (repeats shown as ranges). Per-pixel arrays
pay a per-pixel overhead that window layout amortizes over 49 tokens (Milestone 1 saw
15-25× for attention), and `arrayProject`/`arrayFlatten` of wide arrays holds the array tile
once per output band (601 MiB for 256 bands at only 56²). `Image.convolve` is exact (it
accumulates in double in kernel order: bit-identical to its mirror) but needs one kernel
per (input, output) channel pair, 65,536 for a 256 → 256 conv and 2.36 M for the fusion
conv, far past EE's 10 MiB request limit: not realistic at decoder widths.

**Decision (Q7).** The decoder computes in Milestone 5's window layout, the encoder's own
layout, and does not switch to dense rasters: no switch is needed to do spatial
operations (halos from the 8 neighbouring windows give 3 × 3 convolutions and bilinear
resizes their one-token ring), and every decoder primitive is cheaper there. Dense
georeferenced rasters (C float32 bands) are the format *at rest*: what a task exports,
what the next reads back (one exact conversion at each task boundary, as in Milestone 6),
and the final output. Adaptive average pooling (PSP) is a global linear map of the C4
map: in window layout a reduction over the 16 windows with fixed averaging weights, in
dense layout `reduceRegion` per bin; layout-agnostic, to be decided in Milestone 8.

## 5. Transporting C1-C4: the task topology (Phases 5, 25; Tests 4, 18)

Three facts constrain it. EE batch embeds at most 24-26 M evaluated constants per graph
(Milestone 5). Separate batch tasks share no work (Milestone 5), so exporting C1, C2 and C3
as separate tasks recomputes the encoder head each time. And **an export writes one
grid**: probed with a 160 m and a 320 m band in one image, the asset came back with both
on 160 m (the second band resampled onto the first band's grid).

**Production topology** (earth_engine/feature_assets.py (archive `upernet/earth_engine/feature_assets.py` at `research-archive-2026-10-12`)):

    task A'  S1 scene -> preprocessing -> patch embedding -> stages 1-3 -> ONE "skips" asset on
             the C1 grid (40 m): c1_* (96), c2_* (192, each 80 m pixel on its 2 x 2 C1 pixels),
             c3_* (384, on its 4 x 4), valid_fraction                          673 bands
    task B'  skips asset -> C3 (exact stride sample) -> Patch Merging 3 -> stage 4 -> "C4" asset
             (320 m, 768 bands); Milestone 6's tail without the final norm
    task C   (Milestone 8+) the decoder, reading C1-C3 from the skips asset, C4 from the C4 asset

Writing a coarse value at 40 m is nearest resampling at 40 m pixel centres, which never lie
on an 80 or 160 m pixel edge: exact. Reading back never resamples a finer grid onto a
coarser one by nearest (a 160 m pixel centre falls exactly on a 40 m pixel corner);
C2 and C3 are exact stride samples (one fine pixel per coarse pixel), then relabelled.

| canonical crop (896²) | EECU·s | run s | assets | stored bytes | bands | graph |
|---|---:|---:|---|---:|---:|---|
| **task A' (C1 + C2 + C3 + validity, one task)** | **307.6** | 294 | 1 | 30.7 MB | 673 | 842 KB, 766 invocations |
| **task B' (C4 from the skips asset)** | **61.7** | 123 | 1 | 1.96 MB | 768 | 360 KB, 379 invocations |
| Milestone 6 task A (C3 only) | 309.4 | 146 | 1 | 3.8 MB | 384 | — |
| Milestone 6 separate exports C1, C2 (head recomputed) | 83.1 + 119.9 | 40 + 87 | 2 | 15.8 + 7.7 MB | 96, 192 | — |
| Milestone 6 task B (C4 from the C3 asset) | 67.7 | 131 | 1 | 1.96 MB | 768 | — |

Persisting all three skips costs **nothing extra in EECU·s** (307.6 vs 309.4 for C3 alone),
against 512 for three separate tasks; the 4 × / 16 × replicated C2 / C3 add 12% storage
(30.7 vs 27.3 MB, compressed) and double the run time (bytes written). Read-back
is cheap: C1, C2, C3 and C4 into window layout in one request, 7.2 EECU·s, 2.8 s, 74 MiB (1.2-2.5 EECU·s each).

**Every transported element is exact (Test 18)**: C1, C2, C3 read from the skips asset
and C4 from the task-B' asset are bit-identical to the mirror, to Milestone 6's separately
exported assets, and within 1.55e-5 of SMP (the table of §2 again, 9.0 M values); the
asset grids equal the contract. The swath-edge crop's skips (49% valid) are bit-identical
to the mirror too (C3 within 1.26e-5 of timm).

**Topology B (persist a subset) and C (split the decoder), evaluated.** Persisting only C3
(Milestone 6's task A) and recomputing C1 and C2 inside the decoder task would add the
stage-1-2 encoder to task C: +1.21 M embedded values (stages 1-2, Patch Merging 1 and the
patch embedding) and stage 1-2 compute on every decoder run (Milestone 5: 80-159 EECU·s
for stage 1 in batch plus 73-93 for stage 2, i.e. ~150-250 EECU·s), plus the Sentinel-1
read and preprocessing, to save nothing: task A' already persists C1 and C2 for 0 extra
EECU·s and +12% storage. Splitting the decoder (PSP into task B', FPN + fusion in task C)
fits the cap (21.7 M, verified live, §6) but leaves 2.3 M of margin and still needs a
PSP-output asset. **A' → B' → C dominates both.**

## 6. The decoder's constant budget and where it can run (Phase 6; Test 19)

Embedded values in window layout (earth_engine/topology.py (archive `upernet/earth_engine/topology.py` at `research-archive-2026-10-12`);
row-repeated vectors × 49):

| graph | explicit BatchNorm | folded BatchNorm | below 24 M? |
|---|---:|---:|---|
| task A' (encoder head + patch embedding) | 14.27 M | 14.27 M | ✓ |
| task B' (PM 3 + stage 4; Milestone 5's tail without the final norm) | 16.57 M | 16.57 M | ✓ |
| **task B' + whole decoder** | **26.43 M** | **26.09 M** | **✗** |
| task B' + PSP (+ C4 LayerNorm2d) | 21.71 M | 21.63 M | ✓ (2.3 M margin) |
| **task C: whole decoder** | **9.85 M** | **9.51 M** | **✓** (14 M margin) |
| task C without PSP (FPN + fusion) | 4.72 M | 4.46 M | ✓ |

**Live, in batch** (topology.probe (archive `upernet/earth_engine/topology.py` at `research-archive-2026-10-12`); the constants built by
this milestone's EE constructors from the decoder's assets, explicit BatchNorm; results in
`results/raw/constant_budget.json`):

| probe | embedded values | result |
|---|---:|---|
| decoder constants alone (one-pixel reduceRegion, a "task C" load) | 9.81 M | ✓ 10.4 EECU·s |
| the real task B' graph (C4 export from the skips asset) + PSP and feature-norm constants | 21.73 M | ✓ 208.5 EECU·s, 340 s |
| the real task B' graph + **all** decoder constants | 26.38 M | **✗ "Encoded object too large."** |

The constants are added to task B's output as `0 × (their sum)`: values unchanged, every
constant embedded. The cap measured in Milestone 5 with synthetic constants holds for the
real graphs to within the bracket.

So the decoder cannot live in task B: one more task is inevitable (Q27-Q28), and one is
enough (the whole decoder embeds 9.85 M, 41% of the cap; its decoded weights need
~545 MiB at Milestone 5's ~59 bytes per value). Moving PSP into task B would fit, but
leaves only 2.3 M of margin and gains nothing (PSP's output would have to be exported
anyway); the clean split is **A' | B' | C**.

## 7. Validity metadata (Phase 20)

The skips asset carries Milestone 6's stage-1 valid fraction (k / 32 valid patch values
per C1 pixel) as band `valid_fraction`; the decoder task derives C2-C4 fractions as exact
means of the 4, 16, 64 C1 fractions beneath (equal footprints; sums of k / 32 are exact in
float32). It is metadata, never a decoder input. Live on both crops, every pixel exact
against NumPy's mask fractions:

| crop | C1 | C2 | C3 | C4 |
|---|---|---|---|---|
| canonical_896 | 50,176 pixels, all 1 | 12,544, all 1 | 3,136, all 1 | 784, all 1 |
| edge_224 (swath edge, 49% valid) | 3,136; 1,609 partial or empty | 784; 410 | 196; 107 | 49; 29 |

For Milestone 8+: carry the C1 fraction to the decoder output grid (H/4, then H) and
derive a receptive-field-weighted confidence (§8): a pixel's prediction depends on fill
wherever its dependency reaches a partially valid token, far beyond its own footprint.

## 8. Receptive field of the decoder primitives (Phase 21)

reference/receptive_field.py (archive `upernet/reference/receptive_field.py` at `research-archive-2026-10-12`) composes exact per-axis
index maps (pointwise ops: the pixel itself; k × k conv: ± k//2 pixels with zero padding;
bilinear: PyTorch's own two source indices; adaptive pooling: every pixel) along every
path, then maps C_k pixels to input pixels, optionally through the encoder's
window-granular envelope (Milestone 5: 1, 3, 15, 23 stage-1 windows for C1-C4). Reach
beyond an output pixel's own footprint, input pixels, interior of a long axis:

| operation | reach (each side) | beyond its 224-px aligned block |
|---|---:|---:|
| pointwise (LayerNorm2d, 1 × 1, BN, ReLU, add, concat) | 0 | 0 |
| 3 × 3 conv at C1 / C2 / C3 / C4 | 4 / 8 / 16 / 32 | same |
| bilinear 2 × C2→C1 / C3→C2 / C4→C3 | 8 / 16 / 32 (one source pixel) | same |
| bilinear 8 × C4→C1 | 44 | 32 |
| adaptive average pooling on C4 (PSP) | **the whole map** | unbounded |
| representative chain, decoder only | 48 | 32 |
| C3 / C4 through the encoder (Milestone 6's halo) | 516 / 836 | **420 / 644** |
| **representative chain through the encoder** | 884 | **868 → 896-px aligned halo** |

The decoder's local reach is small, but it crosses the boundary of the 224-px stage-4
window, and the neighbouring window brings its whole encoder envelope: **any spatial
decoder op raises the aligned real-context halo from 672 px (encoder) to 896 px**. PSP's
adaptive pooling then makes every output depend on the whole input raster: tiled
inference with PSP is not equivalent to whole-raster inference for any finite halo.
Tiles must either be the unit of inference (the model sees a tile, as in training) or
PSP's pooled statistics must be computed once over the full area and shared; Milestone 8
must choose. The method updates mechanically as decoder stages are added.

## 9. The representative primitive chain on real features, and asset-backed parameters (Phases 14, 17, 22; Tests 14, 17, 20)

**Chain** (validate.py (archive `upernet/earth_engine/validate.py` at `research-archive-2026-10-12`) `chain`; reference: the installed
model's own modules, pytorch_decoder_primitives.chain (archive `upernet/reference/pytorch_decoder_primitives.py` at `research-archive-2026-10-12`)).
UPerNet's first FPN step with PSP replaced by a stand-in, on the real C4 (task B' asset)
and C3 (task A' skips asset), every parameter read from the decoder's assets:

    n4 = feature_norms[3](C4)            s4 = psp.blocks[0][1](n4)   1x1 Conv-BN-ReLU 768 -> 256
    n3 = feature_norms[2](C3)            l3 = fpn_lateral_blocks[0]   1x1 Conv-BN-ReLU 384 -> 256
    u  = bilinear(s4 -> C3 grid)         f3 = u + l3                 (SMP's FPNLateralBlock)
    p3 = fpn_conv_blocks[0](f3)          3x3 Conv-BN-ReLU 256 -> 256

| tensor | grid | elements | MAE | RMSE | max \|EE − torch32\| | max \|EE − float64\| | EE vs mirror |
|---|---|---:|---:|---:|---:|---:|---|
| n4 (LayerNorm2d 768) | C4 | 602,112 | 6.5e-7 | 8.4e-7 | 5.72e-6 | 8.2e-7 | bit-identical |
| s4 (1 × 1 + BN + ReLU) | C4 | 200,704 | 3.9e-7 | 7.3e-7 | 6.20e-6 | 5.5e-7 | bit-identical |
| n3 (LayerNorm2d 384) | C3 | 1,204,224 | 4.8e-7 | 6.2e-7 | 4.89e-6 | 9.1e-7 | bit-identical |
| l3 (lateral 1 × 1 + BN + ReLU) | C3 | 802,816 | 3.2e-7 | 5.7e-7 | 5.72e-6 | 7.2e-7 | bit-identical |
| u (bilinear C4 → C3) | C3 | 802,816 | 2.6e-7 | 4.8e-7 | 4.53e-6 | 4.6e-7 | bit-identical |
| f3 (u + l3) | C3 | 802,816 | 5.1e-7 | 7.5e-7 | 6.91e-6 | 1.1e-6 | bit-identical |
| p3 conv (3 × 3) | C3 | 802,816 | 9.6e-7 | 1.3e-6 | 1.05e-5 | 5.5e-7 | bit-identical |
| p3 BN | C3 | 802,816 | 9.9e-7 | 1.4e-6 | 1.48e-5 | 8.1e-7 | bit-identical |
| **p3 (ReLU): the chain output** | C3 | 802,816 | 4.9e-7 | 9.8e-7 | **1.48e-5** | 8.1e-7 | **bit-identical** |

Every intermediate of the real-image chain is bit-identical to the mirror and the output
is within 1.48e-5 of SMP's own modules (target 1e-4). This also validates the
asset-backed path end to end: the 3 × 3 weights are stored in PyTorch's (c, ky, kx)
flattening and permuted to the im2col order inside EE.

**Geospatial landing (Test 14).** The resize output relabelled to the crop grid has
projection EPSG:32651 [160, 0, 283200.95…, 0, −160, 2645318.84…] = C3's grid exactly
(56 × 56), 40 / 40 random native pixel centres sample the right C3 pixel, and its values
equal PyTorch's `F.interpolate` of the real s4 (4.5e-6, bit-identical to the mirror).
On the real UTM grids EE's native `resample("bilinear")` (320 m → 160 m) and the dense
formulation are **bit-identical to the window formulation** at all 802,816 values, no
pixel masked: large georeferenced coordinates do not perturb EE's bilinear weights here.

Cost of the chain from the assets: four interactive row-block requests of ≤ 202 KB,
48 EECU·s in total (each block recomputes its halo), ≤ 107 MiB peak.

**Asset-backed decoder parameters (Phase 17, Test 17)**
(earth_engine/decoder_weights.py (archive `upernet/earth_engine/decoder_weights.py` at `research-archive-2026-10-12`)), Milestones
4-6's table format unchanged:

| measure | value |
|---|---|
| parameters | 9,231,682 (69 records: 13 conv weights flattened [Cout, Cin·k·k] in (c, ky, kx) order, the head's bias, 52 BatchNorm and 8 LayerNorm2d vectors as [1, C]) |
| table assets / features / payload | 19 / 4,410 rows / 131.4 MB (≤ 8 MB each) |
| upload | 1,647 s (27.5 min; 2 concurrent export tasks) |
| bit-exact read-back | **all 9,231,682 values: float32 bits equal** (70 tensors) |
| read cost (all values, one request) | 13.0 EECU·s, 4.1 s, 566 MiB reported peak (~64 bytes per decoded value) |
| embedded-value contribution | 9.85 M in window layout (explicit BN), 9.51 M folded (§6) |

**Cost of the primitives (Phase 24)** (benchmark_decoder_primitives.py (archive `upernet/benchmarks/benchmark_decoder_primitives.py` at `research-archive-2026-10-12`);
window layout, asset weights, synthetic inputs at the canonical feature grids; interactive
profiler totals unless batch):

| primitive (canonical scale) | EECU·s | wall s | peak MiB | request / invocations | embedded |
|---|---:|---:|---:|---|---:|
| LayerNorm2d C1 (224², 96) / C4 (28², 768) | 7.3 / 1.6 | 2.6 / 2.2 | 255 / 47 | 14 KB / 18 | 9 k / 75 k |
| 1 × 1 conv 96 → 256 at C1 / 768 → 256 at C4 | 10.0 / 1.6 | 5.6 / 1.3 | 157 / 29 | 10 KB / 10 | 25 k / 197 k |
| 3 × 3 conv 256 → 256 at C3 (56²): im2col / taps | 10.6 / 10.9 | 6.2 / 3.7 | 64 / 46 | 26-29 KB / 29 | 590 k |
| 3 × 3 conv 256 → 256 at C1 (224²): im2col / taps (interactive) | 139 / 102 | 93 / 32 | 638 | 26-29 KB / 30 | 590 k |
| … batch (taps, fresh seed) | 137 | run 110 | — | — | 590 k |
| … at 448² (the 1792 crop's C1), interactive | ✗ memory | — | — | — | — |
| … at 448², batch: im2col / taps | **586** / 1,458 | run 230 / 1,723 | — | 28-31 KB | 590 k |
| Conv + BN + ReLU at C1: explicit / folded (2 runs each) | 133-187 / 145-149 | 67-100 / 69-75 | 638 | 38 / 37 KB | 640 k / 602 k |
| **fusion 3 × 3, 1024 → 256 at C1**: interactive | ✗ memory | — | — | 31 KB / 30 | 2.36 M |
| … batch (taps) / (im2col) | 1,045 / **602** | run 536 / 309 | — | 31 KB | 2.36 M |
| bilinear 2 × C2 → C1 / 8 × C4 → C1 (256 ch.) | 7.8 / 5.8 | 5.2 / 3.1 | 159 / 19 | 27-30 KB / 31 | ~1 k |
| addition / concatenation 4 × 256 at C1 | 27.8 / 43.4 | 12 / 18 | 835 / 1,670 | 10-13 KB | 0 |
| read C1 / C2 / C3 / C4 from the assets into window layout | 1.2 / 1.7 / 2.5 / 1.2 | ≤ 1.2 | 3-74 | 43-49 KB | — |
| read all four | 7.2 | 2.8 | 74 | 224 KB / 286 | — |
| read all 9.23 M decoder parameters | 13.0 | 4.1 | 566 | 138 KB | — |

Addition and concatenation are pure moves or one rounding: their rows measure the
generation of the synthetic 256-channel inputs (the profile's peak is the hash, `Image.mod`,
16 and 32 invocations); on real data the chain's `f3 = u + l3` request cost no more than
`u` and `l3` alone (21.3 vs 11.2 + 13.3 EECU·s). One batch repeat was answered from EE's
cache (an identical graph to an interactive request: 0.9 EECU·s, marked in the CSV);
repeats use fresh seeds.

**Where the decoder's cost will be.** Multiply-accumulates per canonical crop: fusion
3 × 3 118.4 G (73%), FPN 3 × 3 38.8 G (24%; 29.6 G of it at C1), PSP output 3 × 3 3.2 G,
laterals 2.2 G, PSP 1 × 1 and head < 0.05 G: 162.7 G, ~2.3× the Swin-T encoder's ~72 G.
Window-layout 3 × 3 convolutions run at 0.18-0.29 GMAC per EECU·s at 256 channels (the
encoder's whole two-task pass: ~0.2), the 1024-channel fusion conv at 0.20 with im2col
(0.11 with taps): **the fusion conv alone costs ~600 EECU·s in batch, ~1.6× the whole
encoder (task A' + B', 369)**, and the full decoder is projected at ~850-1,000 EECU·s per
canonical crop (fusion ~600, FPN 3 × 3 ~180, the rest ~50-100 including PSP). Memory is bounded
by the halo assembly (`Image.unmask` of the zero-padded window arrays: 638 MiB for 1,024
windows × 256 channels), which grows with channels × windows: the 1024-channel fusion conv
and the 448² C1 grid exceed interactive memory and run in batch.

## 10. Earth Engine behaviour found or confirmed (probed live)

1. **An export writes one grid.** `Export.image.toAsset` resamples every band onto the
   first band's grid (a 320 m band beside a 160 m band came back on 160 m), although
   assets store a grid per band. Multi-scale features in one task need packing.
2. **Binary operations resample silently.** Adding a C3-grid image to a C2-grid image
   evaluates on the request's grid with the coarse operand nearest-resampled: no error.
3. **Nearest reprojection between nested grids is ambiguous at coarse pixel centres**
   (a 160 m centre lies on a 40 m pixel corner); exact stride samples avoid it. Writing a
   coarse grid onto a finer one is unambiguous (fine centres never lie on coarse edges).
4. **`resample("bilinear")` renormalizes over masked neighbours**: with a source masked
   outside its grid this reproduces PyTorch's edge clamp for integer factors, and EE
   computes the bilinear value in double: bit-identical to the exact rule on every fixture and on the real UTM grids (bit-identical to the window formulation at all 802,816 values of the chain's C4 → C3 resize).
5. **Wide per-pixel arrays are expensive to flatten**: `arrayProject` / `arrayFlatten` of
   a [1, 256] array per pixel peaked at 601 MiB on a 56 × 56 grid and exceeded the
   interactive limit on 224 × 224; per-pixel matrix products cost 3-6× the EECU·s of the
   same products on [49, C] window arrays.
6. **EE parses integral-valued JSON numbers as int64**: an embedded −0.0 becomes +0 and
   |v| ≥ 2⁶³ saturates to ±2⁶³ (computed values are unaffected). `max(−0.0, 0)` is +0.0.
7. **`Image.convolve`** accumulates in double in kernel order (exact, as Milestone 6 found)
   but needs one kernel per (input, output) channel pair.
8. As before: matrix products accumulate sequentially in double, translations and stride
   samples are exact relabels, float32 assets carry tensors bit for bit, decoded asset
   arrays cost ~64 bytes per value (566 MiB for the decoder's 9.23 M).

## 11. Recommendation: proceed to Milestone 8

**Proceed.** On each criterion:

- **Exact SMP feature contract**: established from the installed code and proven bit for
  bit (C4 = raw stage 4; LayerNorm2d eps 1e-6 inside UPerNet).
- **Real-image feature correctness and grids**: 9.0 M values within 1.55e-5 of SMP,
  bit-identical to the mirror; grids and footprints exact.
- **Feature transport**: task A' (one skips asset, no extra EECU·s) and task B' (C4),
  exact, read back for 7 EECU·s.
- **Decoder budget**: the decoder needs its own task; it fits there with 14 M values of
  margin (9.85 M of 24 M), confirmed live in both directions.
- **Normalization, Conv2d, bilinear, addition, concatenation, parameters**: all exact to
  the mirror, ≤ 1.5e-5 from PyTorch on real features; the resize lands on the target grid.
- **Memory, EECU, graphs**: every primitive but the fusion conv runs interactively at the
  canonical size; graphs ≤ 202 KB; the decoder's cost will be dominated by its 3 × 3
  convolutions, the fusion conv above all (602 EECU·s in batch, im2col).

For Milestone 8 (the Pyramid Pooling Module on C4):

1. **Build it in task C**, from the C4 asset, in window layout: LayerNorm2d(768), adaptive
   average pooling to 1, 2, 3, 6 (a fixed linear averaging map over C4's 16 windows; check
   PyTorch's overlapping bin rule for 28 / 6 and its float32 summation order), 1 × 1
   Conv-BN-ReLU on each pooled map, the **matrix** resize back to 28 × 28 with PyTorch's
   float32 weights (validated here for 1/3/6 → 28), concatenation (768 + 4 × 256 = 1,792
   channels, PyTorch's order), then the 3 × 3 1792 → 256 Conv-BN-ReLU (window im2col;
   4.1 M weights, 3.2 GMAC: cheap at C4's 16 windows).
2. **Decide PSP's tiling semantics before map production**: pooling is global, so a
   tiled decoder is not equivalent to whole-raster inference for any halo (§8).
3. Keep BatchNorm explicit, the mirror and unique-ID fixtures for every new operation,
   and fresh seeds for repeated benchmarks (EE answers identical graphs from cache).
4. Plan the fusion conv (Milestone 10) as a batch operation with window im2col (602 EECU·s here), and consider splitting it
   by input groups (the four 256-channel levels: a sum of four 256 → 256 convolutions,
   each with a quarter of the halo memory) if batch memory binds at larger crops.

## 12. Caveats

- **Seeded weights.** The decoder's 9.23 M parameters are deterministic seeded values
  (non-trivial BatchNorm statistics included), the encoder's are Milestones 5-6's: the
  milestone proves execution fidelity, not segmentation quality. A trained checkpoint maps
  into the same asset records (conv weights flattened [Cout, Cin k k]).
- **`classes = 2`** follows the project's SegWater model; the head is not built here.
- **One scene, one canonical crop** for the real-image interface (plus the swath-edge crop
  for transport and validity); the 1792 crop's C1 scale was exercised in batch only.
- **Profiler semantics and variance.** "Peak" is EE's per-algorithm accounting, not RAM;
  identical requests varied up to ~1.5× in EECU·s (ranges are repeats).
- **Assets.** Feature assets (`upernet/features/`), weight assets
  (`upernet/m7_decoder_weights_*`) and probe / benchmark outputs (`upernet/bench`,
  `ee_attention` batch tables) remain in project `ORIGINAL_PROJECT`; delete probes when done.
