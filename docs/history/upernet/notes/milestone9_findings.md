# Findings: UPerNet's top-down FPN pathway in native Earth Engine

Live experiments on 6 October 2026: Earth Engine Python API 1.7.45, PyTorch 2.14.0,
timm 1.0.30, segmentation_models_pytorch 0.5.0, NumPy 2.5.3, Cloud project
`ORIGINAL_PROJECT`, application-default credentials. Every Earth Engine tensor was computed in
EE graphs from EE assets (task A''s skips assets, the C4 assets of Milestones 6-8,
Milestone 7's decoder weight assets) or from fixtures generated on the server; tensors were
downloaded only to compare them. No FPN level is computed client-side.

Evidence: numerical_comparison_m9.csv (development archive: `docs/history/upernet/results/numerical_comparison_m9.csv`)
(summary with the Phase 22 matrix: numerical_summary_m9.md (development archive: `docs/history/upernet/results/numerical_summary_m9.md`)),
benchmark_results_m9.csv (development archive: `docs/history/upernet/results/benchmark_results_m9.csv`) (summary:
benchmark_summary_m9.md (development archive: `docs/history/upernet/results/benchmark_summary_m9.md`)), per-request records in
results/raw/ (archive `upernet/results/raw/` at `research-archive-2026-10-12`) (`validation_m9_*`, `bench/m9_*`, `m9_fpn_contract.json`,
`m9_stats_*.json`, `feature_assets.json`, `logs/m9_*`). Required Tests 1-22: offline in
test_upernet_fpn_contract.py (archive `tests/test_upernet_fpn_contract.py` at `research-archive-2026-10-12`),
test_upernet_fpn.py (archive `tests/test_upernet_fpn.py` at `research-archive-2026-10-12`),
test_upernet_fpn_budget.py (archive `tests/test_upernet_fpn_budget.py` at `research-archive-2026-10-12`), live in
test_upernet_fpn_live.py (archive `tests/test_upernet_fpn_live.py` at `research-archive-2026-10-12`) (`EE_LIVE=1`: 14 passed); the full
validation is validate_fpn.py (archive `upernet/earth_engine/validate_fpn.py` at `research-archive-2026-10-12`): 773 comparison rows (584
measured against an EE tensor), 0 failing, all 163 EE-vs-mirror comparisons bit-identical,
14 negative controls that differ as they must.

**Research questions.** (Q1) Can the installed SMP UPerNet's full top-down FPN recursion be
reproduced operation by operation in native EE? (Q2) Do the composed LayerNorm2d → lateral
Conv-BN-ReLU → bilinear resize → addition → 3 × 3 Conv-BN-ReLU steps stay within 1e-4?
(Q3) Do all three merges land exactly on the C3 / C2 / C1 grids with no implicit
reprojection? (Q4) Does PSP + FPN fit comfortably in task C? **Yes, to the installed code,
which is not the recursion the milestone brief draws; yes, 2.5e-5 at worst, every tensor
bit-identical to the mirror; yes, exactly; and yes: 7.44 M embedded values (31% of the
~24 M cap; 9.85 M with M10's fusion constants, run live), 259-288 EECU·s per canonical
task-C export of the whole pyramid, which even runs interactively.**

## The headline: the installed recursion hands down the pre-convolution sum

The brief draws the usual FPN (P3 → bilinear × 2 → + lateral(C2) → 3 × 3 → P2 …), with the
3 × 3 convolution inside the recursion. The installed code
(`segmentation_models_pytorch/decoders/upernet/decoder.py`, `UPerNetDecoder.forward`,
lines 184-197) does something else:

```python
fpn_features = [psp_out]
for i, block in enumerate(self.fpn_lateral_blocks):     # i = 0, 1, 2 -> C3, C2, C1
    state_feature = fpn_features[-1]                     # the previous block's OUTPUT
    fpn_features.append(block(state_feature, lateral_features[i]))   # up(state) + lateral
for i, conv_block in enumerate(self.fpn_conv_blocks, start=1):     # AFTER the loop
    fpn_features[i] = conv_block(fpn_features[i])
```

so, with F4 = the PSP output,

    F_k = up(F_{k+1}) + lateral_k(C_k)          k = 3, 2, 1   (the state handed down)
    P_k = Conv-BN-ReLU 3×3 (F_k)                 applied after the whole recursion

**P3, P2 and P1 are independent leaves**: no P feeds another level; F2 = lat2 + up(F3), not
lat2 + up(P3). This is proved, not read: with the PSP replaced by the constant 1 and the
lateral blocks emitting 10 / 100 / 1000, the state entering lateral blocks 0, 1, 2 is 1, 11,
111 and the conv blocks receive 11, 111, 1111 (were P handed down, block 1 would see
10000 + c); EE's own recursion returns the same 11 / 111 / 1111 (§ 1, § 2). Everything
here follows the installed code; the brief's "P3 → C2 resize" and "P2 → C1 resize" are
**F3 → C2** and **F2 → C1**. Consequences used below: a 3 × 3 error never propagates to
another level (§ 4); a request or export of P1 alone evaluates none of the P2 / P3
convolutions, so every whole-pyramid measurement materializes all four outputs (§ 8).

**Milestone verdict.** **Minimum, strong and full success are all met.** The complete
real-image path, C1-C4 assets → LayerNorm2d → PSP → C3 merge → P3 → C2 merge → P2 → C1
merge → P1, runs entirely inside Earth Engine in task C (interactively and as one batch
export of the whole pyramid) and matches the installed SMP UPerNet within **2.53e-5** at
every final output (P3 1.96e-5, P2 2.00e-5, P1 2.53e-5; the 1792 crop 2.67e-5). Every
intermediate of the synthetic and canonical runs (the PSP output and 27 FPN tensors) was
compared, the 1792 crop at every final output, and every compared tensor is bit-identical to
the controlled mirror. Grids,
footprints, ordering, ID fixtures and padding are exact; the guard refuses every misaligned
operand; task C stays at a third of EE's constant cap. Nothing leaves Earth Engine for
inference.

## Validation matrix (Phase 22)

Generated from the CSV by `python -m upernet.benchmarks.summarize --milestone 9` ("same
input": SMP float32 on the C1-C4 values EE reads). The brief's "P3 → C2" / "P2 → C1" rows are
F3 → C2 / F2 → C1 in the installed code; "P3 ReLU" and "P3" are one tensor.

| operation | MAE | RMSE | max abs | max abs (same input) | max vs float64 | EE vs mirror |
|---|---:|---:|---:|---:|---:|---|
| PSP output (in the task-C graph) | 6.44e-07 | 1.3e-06 | 1.57e-05 | 1.81e-05 | 5.13e-07 | bit-identical |
| C3 LayerNorm2d | 4.82e-07 | 6.23e-07 | 4.89e-06 | 9.54e-07 | 9.11e-07 | bit-identical |
| C3 lateral Conv | 5.78e-07 | 7.37e-07 | 4.77e-06 | 4.29e-06 | 5.64e-07 | bit-identical |
| C3 lateral BN | 5.92e-07 | 7.74e-07 | 6.44e-06 | 5.01e-06 | 7.51e-07 | bit-identical |
| C3 lateral ReLU | 3.21e-07 | 5.7e-07 | 5.72e-06 | 5.01e-06 | 7.18e-07 | bit-identical |
| PSP → C3 resize | 4.61e-07 | 9.21e-07 | 1.26e-05 | 1.29e-05 | 4.06e-07 | bit-identical |
| C3 sum F3 | 6.77e-07 | 1.08e-06 | 1.26e-05 | 1.29e-05 | 7.36e-07 | bit-identical |
| P3 Conv | 1.12e-06 | 1.45e-06 | 1.34e-05 | 1.24e-05 | 5.61e-07 | bit-identical |
| P3 BN | 1.16e-06 | 1.55e-06 | 1.96e-05 | 1.67e-05 | 8.17e-07 | bit-identical |
| **P3** (ReLU) | 5.86e-07 | 1.13e-06 | 1.96e-05 | 1.67e-05 | 8.17e-07 | bit-identical |
| C2 LayerNorm2d | 3.48e-07 | 4.54e-07 | 3.93e-06 | 9.54e-07 | 7.39e-07 | bit-identical |
| C2 lateral Conv | 4.13e-07 | 5.3e-07 | 5.01e-06 | 3.34e-06 | 7.43e-07 | bit-identical |
| C2 lateral BN | 4.16e-07 | 5.45e-07 | 6.56e-06 | 5.01e-06 | 9.02e-07 | bit-identical |
| C2 lateral ReLU | 1.97e-07 | 3.74e-07 | 4.56e-06 | 3.1e-06 | 9.02e-07 | bit-identical |
| F3 → C2 resize | 5.69e-07 | 9.44e-07 | 1.26e-05 | 1.29e-05 | 7.03e-07 | bit-identical |
| C2 sum F2 | 6.6e-07 | 1.02e-06 | 1.24e-05 | 1.29e-05 | 9.06e-07 | bit-identical |
| P2 Conv | 1.28e-06 | 1.7e-06 | 1.86e-05 | 1.91e-05 | 6.4e-07 | bit-identical |
| P2 BN | 1.31e-06 | 1.8e-06 | 2e-05 | 2.05e-05 | 9.36e-07 | bit-identical |
| **P2** (ReLU) | 6.19e-07 | 1.24e-06 | 2e-05 | 2.05e-05 | 9.36e-07 | bit-identical |
| C1 LayerNorm2d | 2.42e-07 | 3.18e-07 | 3.87e-06 | 9.54e-07 | 1.01e-06 | bit-identical |
| C1 lateral Conv | 2.82e-07 | 3.61e-07 | 3.93e-06 | 2.38e-06 | 1.03e-06 | bit-identical |
| C1 lateral BN | 2.9e-07 | 3.8e-07 | 4.41e-06 | 3.81e-06 | 1.16e-06 | bit-identical |
| C1 lateral ReLU | 1.53e-07 | 2.78e-07 | 3.7e-06 | 3.81e-06 | 1.08e-06 | bit-identical |
| F2 → C1 resize | 6.04e-07 | 9.55e-07 | 1.24e-05 | 1.29e-05 | 8.91e-07 | bit-identical |
| C1 sum F1 | 6.61e-07 | 9.96e-07 | 1.24e-05 | 1.29e-05 | 1.19e-06 | bit-identical |
| P1 Conv | 1.5e-06 | 2.01e-06 | 2.15e-05 | 2.29e-05 | 7.76e-07 | bit-identical |
| P1 BN | 1.52e-06 | 2.09e-06 | 2.53e-05 | 2.6e-05 | 1.2e-06 | bit-identical |
| **P1** (ReLU) | 7.74e-07 | 1.47e-06 | 2.53e-05 | 2.38e-05 | 1.15e-06 | bit-identical |
| **complete FPN synthetic (C4 14²)** P3 | 4.55e-07 | 8.69e-07 | 1.12e-05 | 1.12e-05 | 4.47e-07 | bit-identical |
| **complete FPN synthetic (C4 14²)** P2 | 4.69e-07 | 9.72e-07 | 1.72e-05 | 1.72e-05 | 5.91e-07 | bit-identical |
| **complete FPN synthetic (C4 14²)** P1 | 6.22e-07 | 1.24e-06 | 2.24e-05 | 2.24e-05 | 7.93e-07 | bit-identical |
| **complete FPN real canonical (C1 224²)** P3 | 5.86e-07 | 1.13e-06 | 1.96e-05 | 1.67e-05 | 8.17e-07 | bit-identical |
| **complete FPN real canonical (C1 224²)** P2 | 6.19e-07 | 1.24e-06 | 2e-05 | 2.05e-05 | 9.36e-07 | bit-identical |
| **complete FPN real canonical (C1 224²)** P1 | 7.74e-07 | 1.47e-06 | 2.53e-05 | 2.38e-05 | 1.15e-06 | bit-identical |
| **complete FPN real large (C1 448², batch)** P3 | 6.06e-07 | 1.17e-06 | 1.76e-05 | 1.72e-05 | 8.57e-07 | bit-identical |
| **complete FPN real large (C1 448², batch)** P2 | 6.41e-07 | 1.27e-06 | 2.19e-05 | 2.1e-05 | 9.9e-07 | bit-identical |
| **complete FPN real large (C1 448², batch)** P1 | 7.89e-07 | 1.5e-06 | 2.67e-05 | 3e-05 | 1.1e-06 | bit-identical |

Layout and grid rows (exact required, all exact): call order, lateral mapping and recursion
state with unique IDs (PyTorch and EE), fusion order, im2col IDs at 56² / 112² / 224²
(every tap, every edge), weight-row permutations, ramps through the composed resizes (vs
PyTorch, mirror and written rule), resize → lateral grid equality at every level, 10
projections, 8 footprint checks (40 / 40 each), window layout = georeferenced raster, the
1,024-band pyramid asset and every P asset on its grid, pyramid / P1 asset = interactive
output bit for bit, 1,948,992 FPN parameters bit for bit.

## Answers

| # | Question | Answer |
|---|---|---|
| 1 | Exact installed SMP modules of the top-down FPN? | `UPerNetDecoder.feature_norms[0..2]` (LayerNorm2d, eps 1e-6), `fpn_lateral_blocks[0..2]` (`FPNLateralBlock`: Conv2dReLU 1×1 C→256 (Conv bias=False, BN eps 1e-5, ReLU) + `F.interpolate` to the lateral's size + sum), `fpn_conv_blocks[0..2]` (Conv2dReLU 3×3 256→256, padding 1); the PSP output enters as the first state (§1). |
| 2 | Exact feature traversal order? | C3, C2, C1 (`features[:-1][::-1]`): all three lateral blocks run first (state = previous block's output, the **pre-conv sum**), then the three conv blocks (§1). |
| 3 | Which lateral block maps to C3, C2, C1? | Block 0 ← C3 (384), 1 ← C2 (192), 2 ← C1 (96); conv block i serves the same level; proved with unique IDs (§1). |
| 4 | Interpolation convention at each level? | `F.interpolate(state, size=lateral.shape[2:], mode="bilinear", align_corners=False)`, exactly 2× here (28→56→112→224); EE's window resize reproduces PyTorch's weights, source-coordinate rule and edge clamp bit for bit (ramps, §2). |
| 5 | Can the C3 lateral block reproduce SMP? | **Yes**: LN 4.9e-6, conv 4.8e-6, BN 6.4e-6, ReLU 5.7e-6 vs SMP float32; bit-identical to the mirror (§2). |
| 6 | Can PSP resize exactly to C3? | **Yes**: on exactly the C3 token grid (guard) and georeferenced grid; values bit-identical to the mirror; 1.26e-5 vs SMP, inherited from the PSP output (1.57e-5) (§2, §4). |
| 7 | Can the first addition reproduce SMP? | **Yes**: F3 bit-identical to the mirror, 1.26e-5 vs SMP; the guard refuses an un-resized or offset operand (negative controls) (§2). |
| 8 | Can the P3 3×3 Conv-BN-ReLU reproduce SMP? | **Yes**: conv 1.34e-5, BN 1.96e-5, ReLU 1.96e-5; bit-identical to the mirror (§2-3). |
| 9 | Final P3 error? | **1.96e-5** vs SMP float32 (8.2e-7 vs float64); 1792 crop 1.76e-5 (§3). |
| 10 | Can the C2 lateral / top-down step reproduce SMP? | **Yes**: lateral ≤ 6.6e-6, F3→C2 resize 1.26e-5, F2 1.24e-5, conv / BN 1.86 / 2.00e-5; bit-identical to the mirror (§2-3). |
| 11 | Final P2 error? | **2.00e-5** (9.4e-7 vs float64); 1792 crop 2.19e-5 (§3). |
| 12 | Can the C1 lateral / top-down step reproduce SMP? | **Yes**: lateral ≤ 4.4e-6, F2→C1 resize 1.24e-5, F1 1.24e-5, conv / BN 2.15 / 2.53e-5; bit-identical (§2-3). |
| 13 | Final P1 error? | **2.53e-5** (1.15e-6 vs float64; same input 2.38e-5); 1792 crop 2.67e-5 (§3, §11). |
| 14 | Does the full PSP→P3→P2→P1 recursion reproduce SMP? | **Yes**, on synthetic C4 14², the canonical crop (interactive and batch) and the 1792 crop (batch): every output ≤ 2.67e-5 (§3). |
| 15 | Maximum error at any intermediate? | **2.53e-5** (canonical P1_bn / P1); 2.67e-5 / 3.0e-5 (same input) for the 1792 crop's P1 (§3). |
| 16 | Does error materially accumulate from C4 to C1? | **No**: the state's error is flat (PSP 1.57e-5 → F3 1.26e-5 → F2 1.24e-5 → F1 1.24e-5); each P adds one float32 conv on SMP's side (~3e-6 relative at every level); EE − float64 grows only with the magnitudes (5e-7 → 1.2e-6) (§3). |
| 17 | Every EE intermediate bit-identical to the mirror? | **Yes**: 163 / 163 (synthetic, canonical, batch exports, 1792 crop); no first divergence exists (§3). |
| 18 | Do P3 / P2 / P1 occupy exactly the C3 / C2 / C1 grids? | **Yes**: EPSG:32651, 160 / 80 / 40 m, the crop's origin, 56² / 112² / 224²; projections, 40 / 40 footprints, assets (§4, §12). |
| 19 | Are all zero-padding boundaries exact? | **Yes**: ID im2col exact at every edge and corner of all three grids; real border regions bit-identical to the mirror and ≤ 1.2e-5 from SMP; replicate / reflect padding would differ by up to 4.9 on the border (§5). |
| 20 | Local decoder halo the FPN adds inside the tile? | P1: 1 token at C1, C2, C3 and 2 at C4 (92 input pixels each side); P3 80, P2 88; all inside the 8 neighbouring windows; global with the PSP's pooling (Option A: no encoder halo) (§5). |
| 21 | Which FPN level dominates memory? | **C1**: its 3×3 block peaks at 1,657 MiB interactively (C2 504, C3 171, PSP 314) (§7). |
| 22 | Which level dominates compute? | **C1**: its 3×3 block is ~125 of the level's 143 EECU·s and about half of the whole canonical graph (258); C1 > C2 > C3 measured (§7). |
| 23 | Does the canonical full FPN run interactively? | **Yes**, as a checksum of all four outputs: 258 EECU·s, 63 s, 1,657 MiB peak; fetching the outputs needs row blocks (§8). |
| 24 | Does it run in batch? | **Yes**: the whole pyramid in one task-C export (1,024 bands on the C1 grid), P1 alone, and the 1792 crop's P3 / P2 / P1 (§8, §11). |
| 25 | Complete PSP+FPN task-C EECU cost? | **259-288 EECU·s** (canonical, fresh graphs, batch); 155 for a run whose sub-graphs EE had cached; P1 alone 135-245 (§8). |
| 26 | Its peak-memory profile? | Interactive peaks: PSP 314 MiB, + P3 610, + P2 610, + P1 1,657 MiB (the C1 3×3 halo assembly) (§8). |
| 27 | Embedded values of PSP + FPN? | **7,443,930** at C4 28² (7.56 M at 56²): PSP 5.13 M, FPN 3×3 1.92 M, laterals 0.32 M, C1-C3 norms 66 k, resize tables 1.6 k (§9). |
| 28 | Constant headroom for M10? | **16.6 M** below ~24 M now; **14.1 M** (59%) after the fusion block (+2.41 M) and head; the PSP+FPN graph **with the fusion constants (9.85 M) ran in batch** (§9). |
| 29 | Task-C graph size / invocations? | Pyramid export 869 KB / 476 invocations (P1 alone 730 KB / 455; 1792 P1 1.54 MB / 454); interactive full graph 755 KB / 480 (§8). |
| 30 | Does selective parameter loading help? | **Marginally**: reading only the 5 PSP+FPN groups (6.87 M values) vs all 7 (9.23 M): same EECU·s (11.7 vs 11.5), 426 vs 566 MiB peak, 5.7 vs 7.2 s; in a task graph only referenced arrays are read anyway (§10). |
| 31 | Can the 1792 crop run through the full FPN? | **Yes, in batch**: P3 / P2 / P1 at 100 / 310 / 603 EECU·s, within 2.67e-5 of SMP, bit-identical to the mirror; its task A' cost 4,282 EECU·s (§11). |
| 32 | Is window layout still clearly preferred? | **Yes**: every level runs in it, the whole canonical graph is interactive, and nothing measured contradicts Milestone 7's 3-5× advantage over dense formulations (not re-benchmarked) (§7). |
| 33 | What exact four tensors and order must M10 fuse? | `[P1, P2, P3, PSP]` (channels 0-255, 256-511, 512-767, 768-1023), each post-ReLU, resized bilinear (align_corners=False) to C1's 224² / 40 m grid (P1's resize an exact identity) (§13). |
| 34 | New EE behaviours that could invalidate M10? | **None invalidating**; operational ones: computePixels ≤ 1,024 bands; batch reuses cached sub-graphs (cost runs need fresh graphs); point sampling of deep graphs exceeds memory; per-pixel probe constants cost per export pixel; multi-shard exports recompute halos (§14). |
| 35 | Proceed to M10? | **Yes** (§15). |

## 1. The installed FPN and its contract (Phases 1-2, 23; Tests 1-2)

Read off and run from the installed packages
(reference/smp_fpn_reference.py (archive `upernet/reference/smp_fpn_reference.py` at `research-archive-2026-10-12`) `contract`, recorded in
`results/raw/m9_fpn_contract.json`; tests in
test_upernet_fpn_contract.py (archive `tests/test_upernet_fpn_contract.py` at `research-archive-2026-10-12`)):

| item | installed (smp 0.5.0, torch 2.14.0) |
|---|---|
| modules | `UPerNetDecoder.feature_norms[0..2]` (LayerNorm2d 96 / 192 / 384, eps 1e-6), `fpn_lateral_blocks[0..2]` (`FPNLateralBlock`, decoder.py lines 67-94), `fpn_conv_blocks[0..2]` (`Conv2dReLU`); forward lines 163-211 |
| lateral block | `conv_norm_relu` = Conv2d C_k → 256, 1 × 1, **bias=False** → BatchNorm2d(256, eps 1e-5) → ReLU(inplace); then `F.interpolate(state, size=lateral.shape[2:], mode="bilinear", align_corners=False)`; returns `state + lateral` |
| conv block | Conv2d 256 → 256, 3 × 3, padding 1 (zeros), stride 1, **bias=False** → BatchNorm2d(256, eps 1e-5) → ReLU(inplace) |
| loop order | `fpn_lateral_blocks.0`, `.1`, `.2`, then `fpn_conv_blocks.0`, `.1`, `.2` (hooked call order, after `feature_norms.0-3` and `psp`; then `fusion_block`) |
| feature order | `features[:-1][::-1]` = [C3_norm, C2_norm, C1_norm]: **block 0 ← C3 (384), block 1 ← C2 (192), block 2 ← C1 (96)** (unique IDs k·10⁶ + c, every channel in order) |
| PSP insertion | `fpn_features = [psp(features[-1])]`: the PSP output is the first state and the top pyramid level |
| resize rule | `size` = the **lateral's** spatial size (not `scale_factor=2`): 7 → 14 → 28 → 56 at a 224 input, 28 → 56 → 112 → 224 at 896; exactly 2× for every crop of this project (multiples of 224 px) |
| interpolation | bilinear, `align_corners=False`, `antialias=False` (recorded calls at decoder.py:89) |
| normalization / activation | LayerNorm2d before the lateral conv; BatchNorm (explicit, inference) and ReLU in both block types; the sum F is **not** activated (it is a sum of two ReLU outputs, ≥ 0) |
| stored outputs | `fpn_features = [PSP, P3, P2, P1]` after the conv blocks; fusion input `cat(interpolate(fpn_features → C1 size)[::-1])` = **[P1, P2, P3, PSP]** (§ 13) |

**Expanded forward (Phase 1).** `expanded` runs the installed submodules one at a time and
raises unless (i) each lateral block's Conv-BN-ReLU and full output, each conv block and the
PSP (Milestone 8's expansion) equal their modules, and (ii) the PSP output, every F (lateral
block outputs and conv block inputs), every P, every lateral input and the state entering
each lateral block equal what the decoder's **own** `forward` computes (hooks, the forward
stopped at the fusion block's input). It holds bit for bit in float32 and float64 at C4 7²,
14², 28² and in a real UPerNet forward from a random 224² input, whose fusion input also
equals `cat(interpolate([P1, P2, P3, PSP] → C1)[::-1])` exactly (P1's own resize is an exact
identity). References for every phase are slices of this expanded forward on the real
features (never rebuilt from NHWC arrays: the memory-format trap of Milestone 8).

**Statistics (Phase 1)**, SMP float32 on the canonical crop (`results/raw/m9_stats_canonical_896.json`;
synthetic beside it):

| tensor | shape | min | max | mean | std |
|---|---|---:|---:|---:|---:|
| PSP output | 28 × 28 × 256 | 0 | 3.67 | 0.36 | 0.52 |
| C3_norm / lat3 / up3 | 56² × 384 / 256 / 256 | −4.13 / 0 / 0 | 3.88 / 4.25 / 3.52 | −0.004 / 0.52 / 0.36 | 1.02 / 0.75 / 0.52 |
| F3 / P3_conv / P3 | 56² × 256 | 0 / −4.44 / 0 | 5.29 / 4.73 / 6.31 | 0.88 / 0.02 / 0.60 | 0.90 / 1.32 / 0.91 |
| C2_norm / lat2 / up2 | 112² × 192 / 256 / 256 | −5.85 / 0 / 0 | 4.01 / 4.42 / 5.25 | 0.004 / 0.37 / 0.88 | 1.02 / 0.61 / 0.89 |
| F2 / P2_conv / P2 | 112² × 256 | 0 / −5.92 / 0 | 6.58 / 6.44 / 7.27 | 1.26 / −0.09 / 0.62 | 1.06 / 1.63 / 1.01 |
| C1_norm / lat1 / up1 | 224² × 96 / 256 / 256 | −4.67 / 0 / 0 | 4.83 / 5.38 / 6.47 | 0.001 / 0.41 / 1.26 | 0.99 / 0.59 / 1.05 |
| F1 / P1_conv / P1 | 224² × 256 | 0 / −6.69 / 0 | 8.39 / 8.60 / 8.17 | 1.67 / 0.03 / 0.88 | 1.18 / 2.04 / 1.22 |

All float32. Because every term of F is a ReLU output, F grows down the pyramid (mean 0.88 →
1.26 → 1.67, max 5.3 → 6.6 → 8.4): F1 ≈ lat1 + up(lat2 + up(lat3 + up(PSP))).

## 2. The steps in EE (Phases 3-9; Tests 3-5, 8-9, 11-12)

earth_engine/fpn.py (archive `upernet/earth_engine/fpn.py` at `research-archive-2026-10-12`) composes Milestone 7's primitives unchanged, in
window layout ([49, C] per window of each level's own window grid):

| step | EE | rounding |
|---|---|---|
| LayerNorm2d | Milestone 2's `layer_norm`, eps 1e-6, γ β rows 49 from the assets | sequential double sums, one float32 rounding |
| lateral 1 × 1 Conv-BN-ReLU | Milestone 2's `linear` (W^T [C_k, 256] from the asset, no bias), explicit inference BatchNorm, ReLU | one rounding per op |
| resize | `resize.window_resize`, factor 2: the parent window's 9 × 9 halo read at each child window, two products with PyTorch's own dyadic weights ([7, 9] per axis) | exact in double, one rounding |
| addition | `feature_ops.add` with the grid guard: the resize's output grid must **equal** the lateral's (dimensions, scale, CRS) or the add is refused | one rounding |
| 3 × 3 Conv-BN-ReLU | window im2col (9 × 9 halo from the 8 neighbouring windows, zeros outside the raster, [49, 2304] × W^T [2304, 256] permuted on the server), explicit BatchNorm, ReLU | one rounding per op |

**Ordering and layout, exact (fixtures, `m9_ids`, 66 rows, all exact).** Window im2col of
256-channel ID maps at the three FPN grids (56², 112², 224²: 8², 16², 32² windows), all 9
taps × channels 0, 1, 127, 128, 254, 255, equals PyTorch's zero-padded im2col at every
position (edges and corners included); the server-permuted W^T rows of the three conv
blocks equal `weight.permute(2, 3, 1, 0)`. A coordinate ramp (row, column, 1) at C4 28²
pushed through EE's three composed 2× resizes with laterals (0, 0, 10^(4−k)) equals
PyTorch's chained `F.interpolate`, the mirror and the written rule
src = max((o + ½)/2 − ½, 0) (edge clamp) bit for bit at every level, and its ID channel reads
11 / 111 / 1111: **EE hands the pre-conv sum down, as SMP does.** Each resize lands on
exactly the next level's token grid (TokenGrid equality; the guard accepted every add).

**Real features (canonical 896 crop; C1-C3 from task A''s skips asset, C4 from task B''s,
read into window layout by exact stride samples; `m9_fpn_canonical_896`).** Every element of
every tensor below was fetched from the task-C graph (PSP in-graph) on its georeferenced
crop grid and compared with SMP float32 on its own features, SMP float32 on the exact values
EE reads ("same input"), SMP float64, and the mirror:

the Phase 22 matrix above (every operation, canonical crop).

The lateral blocks are within 5.7e-6 of SMP; the resizes, the sums and the 3 × 3 blocks
are dominated by the error the state already carries from the PSP (1.57e-5 in, F 1.26e-5
out): the "same input" column shows the same picture, so the residual is SMP's own float32
arithmetic, not the encoder's. **Every tensor is bit-identical to the mirror.**

**Grid guard negative controls (Test 5; `m9_negative_canonical_896`).** (a) The PSP output
added to lat3 without the resize: the guard refuses (`window on TokenGrid(56, 56, 40.0) vs
window on TokenGrid(28, 28, 80.0)`); EE itself evaluates the unguarded add on the C3 window
grid by nearest resampling of the C4-window image, silently: max error 1.98 against SMP's F3
(78% of values wrong). (b) The lateral relabelled onto a C3 grid offset by 112 m (0.7 pixel)
east: refused (`x0 = 283312.95 vs 283200.95`); unguarded, EE resamples it nearest onto the
true grid — a one-column shift of the lateral and a fully masked first column (56 pixels).
Neither is an error in EE; both are why every FPN addition goes through the guard.

## 3. Composition, mirror and error accumulation (Phases 10-12; Tests 7, 10, 13-16)

**The controlled mirror** (reference/ee_mirror_fpn.py (archive `upernet/reference/ee_mirror_fpn.py` at `research-archive-2026-10-12`))
fixes only the composition order (lateral, resize, add per level; then the three 3 × 3
blocks) on Milestone 7's primitives and Milestone 8's PSP. Its 3 × 3 convolution runs in row
blocks of ≤ 3,136 output pixels (the shape Milestone 8 validated) and equals the single
product bit for bit at 112² and 224²; torch's float64 matmul accumulates strictly
sequentially at [3,136 × 2,304] and [784 × 16,128] (checked). **Every EE floating-point
tensor of the milestone is bit-identical to it** — synthetic (46 / 46), canonical (85 / 85),
batch exports (5 / 5) and the 1792 crop (21 / 21): 163 / 163. There is no first divergence to report.

**Accumulation (Phase 12).** Max |EE − reference| at each boundary, canonical crop:

| boundary | vs SMP float32 | vs SMP float32, same input | vs float64 | SMP float32 vs float64 | max \|value\| | EE vs mirror |
|---|---:|---:|---:|---:|---:|---|
| PSP output (C4) | 1.57e-5 | 1.81e-5 | 5.1e-7 | 1.57e-5 | 3.67 | bit-identical |
| F3 (C3) | 1.26e-5 | 1.29e-5 | 7.4e-7 | 1.26e-5 | 5.29 | bit-identical |
| P3 | 1.96e-5 | 1.67e-5 | 8.2e-7 | 1.94e-5 | 6.31 | bit-identical |
| F2 (C2) | 1.24e-5 | 1.29e-5 | 9.1e-7 | 1.29e-5 | 6.58 | bit-identical |
| P2 | 2.00e-5 | 2.05e-5 | 9.4e-7 | 2.00e-5 | 7.27 | bit-identical |
| F1 (C1) | 1.24e-5 | 1.29e-5 | 1.19e-6 | 1.28e-5 | 8.39 | bit-identical |
| P1 | 2.53e-5 | 2.38e-5 | 1.15e-6 | 2.49e-5 | 8.17 | bit-identical |

The error carried by the state does not grow from C4 to C1 (1.57e-5 → 1.26e-5 → 1.24e-5 →
1.24e-5): each resize is exact and each addition rounds once, so the recursion adds nothing
measurable to what the PSP brings. Each P adds one 2,304-term float32 dot product on SMP's
side (P3 / P2 / P1: 1.96 / 2.00 / 2.53e-5); relative to the map's magnitude that is ~3e-6 at
every level (P1's slightly larger absolute value follows its larger values). EE − float64
grows from 5e-7 to 1.2e-6 with the magnitudes, two orders below the target. **No material
accumulation: worst element anywhere 2.53e-5 (P1_bn / P1), 4× inside 1e-4.** SMP float32
differs from float64 by the same amounts as EE does from SMP float32, so the residual is
PyTorch's float32 arithmetic.

**Every input (Phase 10; Tests 14-15, 21).** The same composition on three inputs, every
final pyramid output compared element by element:

| input | C1 grid | P3 / P2 / P1 vs SMP float32 | P1 same input | P1 vs float64 | EE vs mirror |
|---|---|---|---:|---:|---|
| synthetic C1-C4 (hash × 3), C4 14² | 112², abstract | 1.12e-5 / 1.72e-5 / 2.24e-5 | (= own) | 7.9e-7 | 46 / 46 bit-identical (every FPN tensor) |
| **real canonical 896 crop**, interactive | 224², 40 m | 1.96e-5 / 2.00e-5 / 2.53e-5 | 2.38e-5 | 1.15e-6 | 85 / 85 (every tensor, 4 references) |
| real 1792 crop (C1 448²), batch exports | 448², 40 m | 1.76e-5 / 2.19e-5 / 2.67e-5 | 3.00e-5 | 1.10e-6 | 21 / 21 |

For the 1792 crop task A' was run here (§ 11): its C3 equals Milestone 6's stage-3 asset and
mirror bit for bit, and the "same input" references take EE's C1-C4 as downloaded from the
assets. The synthetic fixture's hash values are dyadic, so its float32 errors are smaller;
the real crops are the evidence.

## 4. Geospatial alignment (Phase 13; Test 17)

Every tensor of the canonical run was fetched as a georeferenced raster (window layout →
Milestone 4's depth_to_space → relabel onto the crop's grid) and compared there:

| tensor | grid (EPSG:32651) | transform | dims | projection check | 40 random native pixels → feature pixel |
|---|---|---|---|---|---|
| PSP output | C4, 320 m | [320, 0, 283200.953…, 0, −320, 2645318.840…] | 28 × 28 | exact | 40 / 40 |
| up3, F3, P3 | C3, 160 m | [160, 0, 283200.953…, 0, −160, 2645318.840…] | 56 × 56 | exact (all three) | 40 / 40 (up3, P3) |
| up2, F2, P2 | C2, 80 m | [80, …] | 112 × 112 | exact | 40 / 40 (up2, P2) |
| up1, F1, P1 | C1, 40 m | [40, …] | 224 × 224 | exact | 40 / 40 (up1; P1 on its task-C asset) |

The crop's origin is shared by all four grids, so pixel (0, 0) of C_k is centred at
(x0 + s/2, y0 − s/2) and its extent is s · n. Pixel centres coincide with PyTorch's
align_corners=False sample positions (Milestone 7), which is why the 2× resize **is** the
geospatial resize from the coarser grid to the finer one: the ramp fixtures prove the
source-coordinate rule, and the window-layout result lands on the next level's token grid
by construction (the guard checks equality before every add). Window layout = georeferenced
raster exactly (P3 whole; P1 window row 13 = rows 91-97). **No EE resampling participates**
(relabels and exact stride samples only); the one place an implicit resample could occur —
the addition — refuses mismatched operands (§ 2). Sampling the in-graph P1 at 40 scattered
native points exceeds interactive memory even five points per request (each point makes EE
evaluate the whole chain around its tile); P1's footprint check therefore samples the
exported P1 asset, which § 12 proves equal to the interactive raster bit for bit.

## 5. Halo and zero padding (Phases 14-15; Test 18)

**Local halo** (receptive_field.py (archive `upernet/reference/receptive_field.py` at `research-archive-2026-10-12`) `fpn_report`: exact
index maps of every op, the PSP's global pooling excluded): an interior output reaches
beyond its own footprint by

| tensor | C1 | C2 | C3 | C4 | input pixels each side |
|---|---:|---:|---:|---:|---:|
| PSP output | | | | 1 | 32 |
| F3 / P3 | | | 0 / 1 | 2 / 2 | 64 / 80 |
| F2 / P2 | | 0 / 1 | 1 | 2 | 80 / 88 |
| F1 / P1 | 0 / 1 | 1 | 1 | 2 | 88 / **92** |

(tokens of each grid). Each 3 × 3 adds one token at its own scale; each 2× resize reads at
most one parent token beyond the child's parent (PyTorch's i0, i1); the C4 reach of 2 is
the PSP's own 3 × 3 plus the C3 resize. In window layout all of it is inside the 8
neighbouring windows: the 3 × 3 blocks read a 1-token ring (9 × 9 halo), the resizes the
parent's 1-token ring. With the PSP's pooling the dependency is global (Milestone 8): under
**Option A** the tile is the model input, so no encoder-style halo is added; these halos
only describe the correct zero-padding / neighbourhood semantics inside the tile.

**Zero padding, exact.** (i) The ID im2col fixtures prove the window conv reads zeros
outside the raster at every edge and corner of all three grids. (ii) On real data every
border region (top, bottom, left, right rows without corners, the four corners) and the
interior of up / F / P at every level is bit-identical to the mirror, and within 1.2e-5
(borders) / 2.5e-5 (interior) of SMP: the borders are no worse than the interior. (iii)
Negative control: the mirror's conv with **replicate** or **reflect** padding differs from
EE's P_conv on the border by up to 3.6-4.9 (canonical) and equals it exactly in the interior
— the padding rule is observable, and EE's is PyTorch's zeros. Resizes use PyTorch's edge
clamp (ramp fixtures, edges exact).

## 7. Cost by level (Phase 16)

benchmark_fpn.py (archive `upernet/benchmarks/benchmark_fpn.py` at `research-archive-2026-10-12`) `levels`: each level alone, cumulative
prefixes, one profiled interactive request each (checksum of the prefix's tensors). The
state is read from a persisted asset (C3: Milestone 8's PSP asset; C2: F3; C1: F2, exported
here): an operation's cost does not depend on its input values. Every request is salted
(+ constant × 0 on the inputs) so no graph coincides with an earlier one (§ 14).

| prefix | C3 (56²) EECU·s / peak MiB | C2 (112²) | C1 (224²) |
|---|---:|---:|---:|
| read C_k + state | 2.8 / 74 | 3.3 / 37 | 4.1 / 18 |
| + LayerNorm2d | 3.9 / 74 | 6.4 / 37 | 6.3 / 39 |
| + 1 × 1 Conv-BN-ReLU | 4.8 / 74 | 6.3 / 37 | 15.0 / 39 |
| + 2× bilinear resize of the state | 5.5 / 74 | 14.5 / 37 | 19.9 / 39 |
| + addition (F_k) | 7.3 / 74 | 14.5 / 37 | 17.7 / 39 |
| **+ 3 × 3 Conv-BN-ReLU (P_k)** | **16.8 / 171** | **40.3 / 504** | **142.9 / 1,657** (247 s wall) |

Graphs: 97-170 KB, 144-198 invocations. **C1 > C2 > C3**, measured: each level's 3 × 3 block
is most of its cost (≈ 9, 26, 125 EECU·s) and all of its memory (the 9 × 9 halo assembly of
256-channel windows, growing with the windows a request touches); laterals, resizes and the
additions cost a few EECU·s (identical requests vary up to ~1.5×). Asset reads are 3-4
EECU·s per level.

## 8. The complete PSP + FPN task C (Phase 17; Test 20)

**Interactive** (`full`, the whole graph from the four feature assets; every pyramid output
so far enters the checksum — P1, P2, P3 are leaves, so a checksum of P1 alone would never
evaluate the P2 / P3 convolutions):

| prefix | EECU·s | wall s | peak MiB | request / invocations | embedded |
|---|---:|---:|---:|---|---:|
| PSP | 36.7 | 18.5 | 314 | 439 KB / 149 | 5.20 M |
| + P3 | 82.3 | 29.2 | 610 | 577 KB / 267 | 5.95 M |
| + P2 | 132.2 | 28.7 | 610 | 668 KB / 377 | 6.70 M |
| **+ P1 (complete)** | **258.1** | **63.4** | **1,657** | 755 KB / 480 | 7.44 M |

**The complete canonical PSP + FPN runs interactively** as one request. Downloading its
outputs does not fit one request (the validation fetched 7-row blocks, 4-20 s each).

**Batch** (task C exports from the assets; `fpn_pipeline`):

| export | EECU·s | run s | request / invocations | notes |
|---|---:|---:|---|---|
| **pyramid** (PSP, P3, P2, P1; 1,024 bands on the C1 grid), fresh r1 / r2 | **288 / 259** | 364 / 338 | 869 KB / 476 | one graph evaluating every level |
| pyramid, first run | 155 | 214 | 869 KB / 475 | after the F3 / F2 / P1 exports: sub-graphs likely served from EE's cache |
| P1 alone (milestone asset, 256 bands, 40 m), first / fresh r1 / r2 | 245 / 135 / 169 | 352 / 129 / 201 | 730 KB / 455 | no P2 / P3 convolution in the graph |
| F3 / F2 states (benchmark inputs) | 22 / 28 | 68 / 34 | 561 / 636 KB | |
| pyramid + fusion constants × 0 (budget probe) | 4,723 | 4,290 | 882 KB / 483 | success; cost is the probe's own per-pixel checksum (§ 9) |

Batch EECU·s vary by up to ~1.8× between identical-cost runs (P1: 135-245); the canonical
PSP + FPN task C costs ~260-290 EECU·s fresh, about half of Milestone 7's measured fusion
conv alone (602 EECU·s), which M10 adds.

## 9. Task-C constant budget (Phase 18; Test 19)

Embedded values, counted from the constructors (`fpn.budget`; explicit BatchNorm, window
layout), and raw parameters:

| component | raw parameters | embedded (C4 28²) |
|---|---:|---:|
| C1-C3 LayerNorm2d | 1,344 | 65,856 |
| PSP (C4 LayerNorm2d, branches, resize tables, 1792 → 256 3 × 3, BN) | 4,921,856 | 5,133,890 |
| FPN lateral 1 × 1 (672 × 256) + BN | 175,104 | 322,560 |
| FPN 2× resize tables (3 resizes × 2 axes) | — | 1,624 |
| FPN 3 × 3 (3 × 2,304 × 256) + BN | 1,772,544 | 1,920,000 |
| **PSP + FPN (task C before fusion)** | **6,870,848** | **7,443,930** (56²: 7,564,442) |
| + fusion 3 × 3 1024 → 256 + BN (M10) | 2,360,320 | 2,409,472 |
| + segmentation head | 514 | ~600 |
| whole decoder | 9,231,682 | 9.85 M (Milestone 7) |

**7.44 M is 31% of the ~24 M batch cap** (Milestone 5 placed the failure between 24 and
26 M), leaving 16.6 M; with M10's fusion block and head the whole decoder is 9.85 M (41%),
leaving **14.1 M**. **Live:** the canonical pyramid export carrying, in addition, the fusion
block's constants (× 0; 9,853,402 embedded in total) **succeeds in batch**. Its 4,723 EECU·s
are an artefact of the probe, not of the FPN: the probe's checksum reduces all 2.4 M fusion
constants at every exported pixel (50,176 pixels on the C1 grid; Milestones 7-8 ran the same
probe on 784 C4 pixels). Future probes should put the checksum on a one-pixel grid.
Decoded parameter memory at Milestone 7's ~64 bytes per value: ~450 MiB for PSP + FPN (the
parameter-read request below peaks at 426 MiB).

## 10. Asset reads (Phase 19)

No second copy of any parameter: task C reads Milestone 7's decoder assets, groups
`feature_norms`, `psp_pooled_1x1`, `psp_out_conv_3x3`, `fpn_lateral_1x1`, `fpn_conv_3x3`; the
FPN's 21 tensors (1,948,992 values) read back bit for bit, SHA-256 equal to Milestone 7's
manifest. A checksum of every parameter read:

| groups | values | EECU·s | wall s | peak MiB | invocations |
|---|---:|---:|---:|---:|---:|
| PSP + FPN (5 groups) | 6,870,848 | 11.7 | 5.7 | 426 | 80 |
| all 7 decoder groups | 9,231,682 | 11.5 | 7.2 | 566 | 88 |

Selective loading saves ~25% of the read's memory and ~20% of its wall time, not EECU·s;
inside a task graph only the arrays the graph references are read at all, so selectivity is
automatic. The reads are ~4% of a task-C export's EECU·s.

## 11. The 1792 crop (Phase 20; Test 21)

C1 448², C2 224², C3 112², C4 56². Task A' was run for this crop (`m9_skips_large_1792`, one
673-band asset on the 448² C1 grid): **4,282 EECU·s** (run 1,047 s), 3.6× Milestone 6's
task A for the same crop (1,184, stage 3 only on the 112² C3 grid). Its C3 equals Milestone
6's stage-3 asset and mirror bit for bit, so the cost is not a different computation. Likely
cause (not tested): the 448² export is computed in 256-pixel shards, each re-evaluating the
encoder head with its window-granular halo, whereas the canonical 224² export fits one shard
(Milestone 7 measured A' = task A there). `Export.image.toAsset(shardSize=…)` is the
untested remedy; this matters for production tiling, not for the FPN.

Task C exports (C4 = Milestone 6's asset):

| output | EECU·s | run s | request / invocations | vs SMP float32 | same input | vs float64 | mirror |
|---|---:|---:|---|---:|---:|---:|---|
| P3 (112², 160 m) | 100 | 90 | 1.39 MB / 270 | 1.76e-5 | 1.72e-5 | 8.6e-7 | bit-identical |
| P2 (224², 80 m) | 310 | 377 | 1.47 MB / 365 | 2.19e-5 | 2.10e-5 | 9.9e-7 | bit-identical |
| P1 (448², 40 m) | 603 | 351 | 1.54 MB / 454 | 2.67e-5 | 3.00e-5 | 1.1e-6 | bit-identical |

Every band of each asset on its crop grid; every element of P1 (51.4 M values), P2, P3
compared. P1 at 448² is batch-only (Milestone 7: the 3 × 3 at 448² exceeds interactive
memory). EECU·s scale ~4.5× from canonical P1 (135-245) to 1792 P1 (603) for 4× the pixels.

## 12. Task-C output prototype (Phase 21; Test 22)

`fpn_pipeline.export_level(P1)`: the canonical P1 as a georeferenced asset, 256 float32 bands
`c0..c255`, EPSG:32651, [40, 0, 283200.953…, 0, −40, 2645318.840…], 224 × 224 (every band
checked); every element equal to the interactive output, to the pyramid asset's P1 bands and
to the mirror (bit for bit), 2.53e-5 from SMP float32; its 40 random native-pixel footprints
land on (row // 4, col // 4). The pyramid asset `m9_pyramid_canonical_896` holds all four
levels (P2, P3, PSP as exact nearest copies on the C1 grid, recovered by stride sampling,
each equal to its interactive output bit for bit). These exports are demonstrations; the
production decoder keeps the pyramid inside task C through fusion and the head.

## 13. What M10 consumes (Phase 23)

From the installed code (lines 199-210) and the ID fixture (`m9_fpn_contract.json`):

| list position | tensor | channels in the 1,024 | from | grid before the resize | state |
|---|---|---|---|---|---|
| 0 | P1 | 0-255 | `fpn_conv_blocks[2]` | C1, 40 m (resize to its own size: exact identity) | post-ReLU |
| 1 | P2 | 256-511 | `fpn_conv_blocks[1]` | C2, 80 m → C1 | post-ReLU |
| 2 | P3 | 512-767 | `fpn_conv_blocks[0]` | C3, 160 m → C1 | post-ReLU |
| 3 | PSP output | 768-1023 | `psp.out_conv` | C4, 320 m → C1 | post-ReLU |

`resized = [F.interpolate(f, size=P1.shape[2:], mode="bilinear", align_corners=False) for f
in [PSP, P3, P2, P1]]`, `torch.cat(resized[::-1], dim=1)` → `fusion_block` (3 × 3 Conv-BN-ReLU
1024 → 256). The resizes are 8×, 4×, 2× and 1× (integer factors: Milestone 7's window resize
applies, the 1× is a copy). Neither F nor any pre-activation tensor reaches the fusion.

## 14. Earth Engine behaviour found or confirmed

1. **computePixels refuses more than 1,024 bands** per request ("Number of bands (2432) must
   be less than or equal to 1024"); `Export.image.toAsset` wrote a 1,024-band asset.
2. **Batch reuses cached work across tasks.** The first pyramid export (155 EECU·s), run
   after exports of overlapping graphs, cost about half of two fresh ones (288, 259).
   Milestone 7 saw identical graphs answered from cache; this suggests sub-graphs too. Cost
   measurements here use salted inputs (+ constant × 0: same values, new graph).
3. **Point sampling of a deep graph** (`reduceRegions` at 40 scattered points of the
   in-graph P1) exceeds interactive memory even five points at a time: each point evaluates
   the whole chain around its tile. Footprints of deep outputs are checked on their exports.
4. **Constants evaluated per pixel cost per pixel**: the budget probe's checksum image cost
   4,723 EECU·s on the C1 grid (§ 9).
5. **Large exports appear to recompute per shard** (task A' at 1792: 3.6× the expected cost;
   hypothesis, § 11).
6. Operational: one validation process hung during EE initialization (network; killed and
   rerun); interactive block fetches ran 4 at a time without errors.

None affects the FPN's values or invalidates M10: the fusion stage resizes the four levels
with Milestone 7's window resize (integer factors 8, 4, 2, 1), concatenates in window layout
and runs one window-im2col 3 × 3 (Milestone 7: 602 EECU·s in batch at C1, batch-only).

## 15. Recommendation

**Proceed to Milestone 10 (resize P2, P3, PSP to the C1 grid, concatenate the four
256-channel levels, run the 1024 → 256 fusion block): yes.** Against each criterion:

| criterion | evidence |
|---|---|
| FPN numerical correctness | P3 / P2 / P1 1.96 / 2.00 / 2.53e-5 (target 1e-4); every one of 28 intermediates compared (§3) |
| P3 / P2 / P1 geospatial alignment | grids, transforms, footprints, assets exact; guard on every add (§4) |
| error accumulation | none: the state's error flat C4 → C1; ≈ 3e-6 relative per P (§3) |
| controlled-mirror agreement | 163 / 163 bit-identical (§3) |
| C1-level memory | 1,657 MiB interactive peak (the C1 3 × 3); runs interactively at 224² (§7) |
| C1-level 3 × 3 cost | ~125-143 EECU·s interactive, half the canonical graph (§7) |
| task-C constant budget | 7.44 M (31%); 9.85 M with fusion, live in batch; 14.1 M headroom after M10 (§9) |
| task-C graph size | 869 KB / 476 invocations for the whole pyramid (§8) |
| interactive feasibility | complete canonical PSP + FPN: 258 EECU·s, 63 s (§8) |
| batch feasibility | pyramid 259-288 EECU·s; 1792 crop P1 603 (§8, §11) |
| larger-crop scaling | 1792 crop exact to 2.67e-5, ~4.5× canonical cost for 4× pixels (§11) |
| window-layout viability | confirmed at all three levels (§7) |
| remaining constant headroom | 14.1 M (59% of the cap) after fusion and head (§9) |

For M10: consume `[P1, P2, P3, PSP]` from `fpn.forward` inside the same task-C graph (no
persisted pyramid); resize with `resize.window_resize` (8×, 4×, 2×; P1 as is), concatenate
with `feature_ops.concat` (grid guard), and expect the fusion 3 × 3 to be batch-only at the
canonical size (Milestone 7) — the whole decoder then ~0.9 k EECU·s per canonical tile.
Check `shardSize` for exports larger than 256 px per side (§ 11).

## 16. Caveats

- **Seeded weights.** The decoder's parameters are Milestone 7's deterministic seeded values
  (non-trivial BatchNorm statistics); execution fidelity is proven for them, and a trained
  checkpoint's values change magnitudes, not the semantics.
- **One scene.** Real features are Milestone 6's Sun Moon Lake crops (canonical 896, 1792).
- **Variance.** Interactive EECU·s vary up to ~1.5× between identical requests, batch up to
  ~1.8×; benchmark rows are single runs except the fresh batch repeats. "Peak" is EE's
  per-algorithm accounting, not RAM.
- **Interactive headroom is thin.** The standalone C1 3 × 3 request took 247 s of wall time
  against EE's 300 s interactive deadline (the complete-graph request took 63 s: variance);
  "runs interactively" at the canonical size is real but not comfortable, and M10's fusion
  conv on top of it will be batch-only.
- **Embedded values in the batch rows** of `benchmark_results_m9.csv` are the whole PSP + FPN
  budget (7,443,930) for every export, an upper bound: the F3 / F2 exports embed far less (no
  3 × 3 FPN block), P1 alone lacks the P2 / P3 blocks.
- **Costs spent beyond the FPN itself**: task A' for the 1792 crop (4,282 EECU·s) and the
  budget probe (4,723 EECU·s, an artefact of the probe's design).
- **Assets** remain in project `ORIGINAL_PROJECT`: `upernet/fpn/m9_*` (canonical P1, F3, F2,
  pyramid, probe, bench repeats; 1792 P1-P3) and `upernet/features/m9_skips_large_1792_*`;
  delete the probe and bench exports when done.

## Tests → evidence

| test | where |
|---|---|
| 1 installed FPN contract | `test_upernet_fpn_contract.py` (source, configs, call order, interpolation calls), `m9_contract` rows |
| 2 lateral mapping with unique IDs | `test_lateral_mapping_with_unique_ids`, `test_recursion_hands_down_the_pre_conv_sum`; EE: `m9_ids` F_k ID channel |
| 3 C3 lateral block | `m9_fpn_canonical_896` C3_norm, lat3_* rows; live test 03 |
| 4 PSP → C3 resize | up3 rows, `up3_lands_on_C3`, ramps, up3 projection / footprint; live test 04 |
| 5 C3 addition + guard | F3 rows; `m9_negative_canonical_896` (both refusals, unguarded EE results); live test 05 |
| 6 P3 3 × 3 block | P3_conv / P3_bn / P3 rows; im2col IDs at C3; live tests 06-07 |
| 7 complete P3 step | P3 vs SMP (canonical, synthetic, 1792) |
| 8 C2 lateral block | C2_norm, lat2_* rows; live tests 08-10 |
| 9 F3 → C2 resize | up2 rows, ramps, grid rows |
| 10 complete P2 step | P2 rows (three inputs) |
| 11 C1 lateral block | C1_norm, lat1_* rows; live tests 11-13 |
| 12 F2 → C1 resize | up1 rows, ramps, grid rows |
| 13 complete P1 step | P1 rows (three inputs) |
| 14 full synthetic FPN | `m9_fpn_synthetic_14` (46 / 46 mirror); live test 14 |
| 15 full real canonical FPN | `m9_fpn_canonical_896` (85 / 85 mirror); live test 15 |
| 16 controlled mirror | every `ee_mirror` row (163 / 163); `test_upernet_fpn.py` mirror tests |
| 17 geospatial pyramid grids | projection, footprint, window-vs-georeferenced rows; asset grid rows; live test 17 |
| 18 boundary zero padding | im2col IDs (every edge), boundary_* rows, padding negative controls; live test 18 |
| 19 task-C constant budget | `test_upernet_fpn_budget.py`; `task_C_budget` probe row; live test 19 |
| 20 canonical task-C batch run | pyramid export rows (`m9_task_c_canonical_896`), bench rows; live test 20 |
| 21 large crop batch run | `m9_fpn_large_1792`; live test 21 |
| 22 P1 milestone export | `P1_asset` rows (grid, = interactive = pyramid = mirror); live test 22 |

Offline: the 22 Milestone 9 tests pass (125 with Milestones 7-8's). Live (`EE_LIVE=1`): 14 / 14 pass (`logs/m9_live_tests.log`).
