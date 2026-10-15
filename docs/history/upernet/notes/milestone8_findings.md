# Findings: UPerNet's Pyramid Pooling Module in native Earth Engine

Live experiments on 6 October 2026: Earth Engine Python API 1.7.45, PyTorch 2.14.0,
timm 1.0.30, segmentation_models_pytorch 0.5.0, NumPy 2.5.3, Cloud project
`ORIGINAL_PROJECT`, application-default credentials. Every Earth Engine tensor was computed in
EE graphs from EE assets (the C4 assets of Milestones 6-7 and this milestone, Milestone 7's
decoder weight assets) or from fixtures generated on the server or embedded in the request;
tensors were downloaded only to compare them. Pooling is never computed client-side.

Evidence: numerical_comparison_m8.csv (development archive: `docs/history/upernet/results/numerical_comparison_m8.csv`)
(summary with the Phase 22 matrix: numerical_summary_m8.md (development archive: `docs/history/upernet/results/numerical_summary_m8.md`)),
benchmark_results_m8.csv (development archive: `docs/history/upernet/results/benchmark_results_m8.csv`) (summary:
benchmark_summary_m8.md (development archive: `docs/history/upernet/results/benchmark_summary_m8.md`)), per-request records in
results/raw/ (archive `upernet/results/raw/` at `research-archive-2026-10-12`) (`validation_m8_*_runs.json`, `bench/m8_*`,
`m8_pool_bins.json`, `psp_semantics.json`, `m8_semantics_ee.json`, `m8_stats_*.json`,
`m8_edge_valid_fraction.json`, `feature_assets.json`, `logs/m8_*`). Required Tests 1-18:
offline in test_upernet_psp_contract.py (archive `tests/test_upernet_psp_contract.py` at `research-archive-2026-10-12`),
test_upernet_adaptive_pool.py (archive `tests/test_upernet_adaptive_pool.py` at `research-archive-2026-10-12`),
test_upernet_psp_budget.py (archive `tests/test_upernet_psp_budget.py` at `research-archive-2026-10-12`),
test_upernet_psp_semantics.py (archive `tests/test_upernet_psp_semantics.py` at `research-archive-2026-10-12`) (all pass, with
Milestone 7's), live in test_upernet_psp_live.py (archive `tests/test_upernet_psp_live.py` at `research-archive-2026-10-12`)
(`EE_LIVE=1`: 14 passed, `logs/m8_live_tests.log`); the full validation is
validate_psp.py (archive `upernet/earth_engine/validate_psp.py` at `research-archive-2026-10-12`): 1,359 comparison rows, 0 failing, all 200
EE-vs-mirror comparisons bit-identical.

**Research questions.** (Q1) Can the installed SMP UPerNet's PSP be reproduced operation by
operation in native EE? (Q2) Can EE reproduce PyTorch's `AdaptiveAvgPool2d` exactly for 1, 2,
3 and the non-divisible 6? (Q3) What does PSP's global pooling mean for geospatial
inference? (Q4) Does the PSP fit comfortably in task C? **Yes; yes, bin for bin and
bit-identical to an exact float64 model of EE; a finite tile is a different model input
than the larger area (measured), and the exact whole-area alternative is cheap; and yes,
with 5.1 M of the ~24 M embedded values.**

- **The installed PSP, read off the code.** `PSPModule` (smp 0.5.0
  `decoders/upernet/decoder.py`): `AdaptiveAvgPool2d(s)` → `Conv2dReLU` 1×1 768→256 for
  s = 1, 2, 3, 6 in that order, `F.interpolate(bilinear, align_corners=False)` to C4's size,
  `torch.cat([C4_norm, b1, b2, b3, b6])` (1,792 channels, proven with unique channel IDs),
  `Conv2dReLU` 3×3 1792→256 (padding 1, no bias). An expanded forward is bit-identical to the
  module, and a real UPerNet forward feeds the PSP exactly `feature_norms[3](C4)`.
- **Adaptive pooling, exactly.** PyTorch's bins are [⌊iN/M⌋, ⌈(i+1)N/M⌉): 28 → 6 is
  [0,5) [4,10) [9,14) [14,19) [18,24) [23,28), overlapping at rows/columns 4, 9, 18, 23.
  One-hot × ID fixtures recover the exact source set of all 50 bins at 7, 14, 28 and 56 from
  EE; pooled values are bit-identical to the mirror everywhere and to the exactly rounded
  mean on ordinary data; real C4: ≤ 1.9e-6 from SMP float32.
- **Window layout throughout.** Pooling is a per-window 0/1 membership product, a
  deterministic `sampleRectangle` gather of the window partials, a sequential sum over
  windows and one division by the bin area (1.8 EECU·s at the canonical size); the branches
  run on the pooled value; Milestone 7's matrix resize, concatenation and window im2col
  conv do the rest. Dense formulations cost more (pooling 1.6-2.5×, the 3×3 conv 2×).
- **The complete PSP on real imagery.** Canonical C4 (28 × 28 × 768, the task-B' asset):
  every intermediate bit-identical to the mirror, the PSP output within **1.57e-5** of SMP
  float32 (5.1e-7 of float64), on exactly the C4 grid; the 1792 crop (56²) 1.76e-5, the
  swath edge 1.54e-5. The residual is SMP's own float32 arithmetic (EE is bit-identical to
  the mirror everywhere), concentrated in the 16,128-term dot products of the output conv.
  Task C exports the 256-band output asset on the C4 grid in batch (17 EECU·s at 28², 65 at
  56²), every element equal to the interactive output.
- **Global semantics.** PSP of a 448-px tile differs from the 1792-px area's PSP over the
  same tokens by 0.29 mean relative L2 (encoder held fixed): the border ring by 0.55 (zero
  padding), every interior token by ~0.19, not decaying with distance (global pooling).
  Persisting the whole area's 50 × 256 pooled branch maps and decoding tiles with a
  one-window C4 ring reproduces the whole-area PSP **bit for bit**, live in EE (three tiles,
  interior and corner); the persisted maps are a 32 KB asset (5.2 EECU·s to export).

**Milestone verdict.** **Minimum, strong and full success are all met.** EE reproduces
the four adaptive-average pools with exact bin membership (overlapping 28 → 3 and 28 → 6
included, 56² too) and bit-identical floating point to its mirror; each complete branch
(pool → 1 × 1 Conv-BN-ReLU → bilinear resize) reproduces the installed SMP modules; and the
complete real-image PSP (raw C4 → LayerNorm2d → four branches → concat 1,792 → 3 × 3
Conv-BN-ReLU) runs entirely in EE in task C and produces 28 × 28 × 256 georeferenced
features on the C4 grid within **1.57e-5** of SMP (target 1e-4), with every intermediate
compared. Pooling membership and every layout operation are exact. PSP's global semantics
are characterized on real nested crops and decided (§ 11, § 14), with the exact
whole-area alternative proven live. Nothing leaves Earth Engine for inference.

## Validation matrix (Phase 22)

Real canonical C4 (the task-B' asset) through the EE PSP, every element of every
intermediate, against SMP float32 on its own forward ("same input": SMP float32 on the C4
values EE reads), float64, and the mirror; then the complete PSP on every input. Generated
from the CSV by `python -m upernet.benchmarks.summarize --milestone 8`.

| operation | MAE | RMSE | max abs | max abs (same input) | max vs float64 | EE vs mirror |
|---|---:|---:|---:|---:|---:|---|
| C4 LayerNorm2d | 6.49e-07 | 8.42e-07 | 5.72e-06 | 9.54e-07 | 8.16e-07 | bit-identical |
| pool 1×1 | 9.36e-08 | 1.29e-07 | 7.15e-07 | 2.38e-07 | 1.19e-07 | bit-identical |
| pool 2×2 | 1.86e-07 | 2.64e-07 | 1.91e-06 | 1.91e-06 | 1.56e-07 | bit-identical |
| pool 3×3 | 1.77e-07 | 2.4e-07 | 1.91e-06 | 2.15e-06 | 1.78e-07 | bit-identical |
| pool 6×6 | 2.06e-07 | 2.69e-07 | 1.67e-06 | 7.15e-07 | 2.73e-07 | bit-identical |
| branch1 Conv | 1.14e-07 | 1.49e-07 | 3.58e-07 | 3.58e-07 | 9.43e-08 | bit-identical |
| branch1 BN | 1.28e-07 | 1.69e-07 | 4.77e-07 | 7.15e-07 | 1.26e-07 | bit-identical |
| branch1 ReLU | 6.38e-08 | 1.21e-07 | 4.77e-07 | 7.15e-07 | 1.15e-07 | bit-identical |
| branch1 resize | 6.48e-08 | 1.23e-07 | 7.15e-07 | 9.54e-07 | 1.15e-07 | bit-identical |
| branch2 Conv | 4.47e-07 | 6.12e-07 | 4.29e-06 | 3.34e-06 | 1.48e-07 | bit-identical |
| branch2 BN | 4.49e-07 | 6.15e-07 | 4.05e-06 | 3.1e-06 | 2.74e-07 | bit-identical |
| branch2 ReLU | 2.48e-07 | 4.49e-07 | 2.62e-06 | 2.38e-06 | 2.19e-07 | bit-identical |
| branch2 resize | 2.02e-07 | 3.73e-07 | 2.62e-06 | 2.62e-06 | 3.63e-07 | bit-identical |
| branch3 Conv | 4.14e-07 | 5.62e-07 | 3.1e-06 | 3.1e-06 | 2.01e-07 | bit-identical |
| branch3 BN | 4.16e-07 | 5.75e-07 | 3.46e-06 | 2.98e-06 | 2.38e-07 | bit-identical |
| branch3 ReLU | 2.15e-07 | 4.15e-07 | 3.46e-06 | 2.98e-06 | 2.38e-07 | bit-identical |
| branch3 resize | 1.72e-07 | 3.29e-07 | 3.46e-06 | 3.1e-06 | 3.48e-07 | bit-identical |
| branch6 Conv | 4.26e-07 | 5.74e-07 | 3.93e-06 | 3.58e-06 | 2.03e-07 | bit-identical |
| branch6 BN | 4.39e-07 | 6.11e-07 | 4.29e-06 | 5.25e-06 | 4.1e-07 | bit-identical |
| branch6 ReLU | 2.11e-07 | 4.06e-07 | 3.7e-06 | 2.98e-06 | 2.73e-07 | bit-identical |
| branch6 resize | 1.61e-07 | 3.06e-07 | 3.46e-06 | 2.62e-06 | 4.8e-07 | bit-identical |
| PSP concat | 3.64e-07 | 5.95e-07 | 5.72e-06 | 3.1e-06 | 8.16e-07 | bit-identical |
| PSP output Conv 1792→256 | 1.3e-06 | 1.81e-06 | 2.31e-05 | 2.12e-05 | 3.41e-07 | bit-identical |
| PSP output BN | 1.29e-06 | 1.82e-06 | 2.5e-05 | 2.29e-05 | 5.15e-07 | bit-identical |
| PSP output ReLU | 6.44e-07 | 1.3e-06 | 1.57e-05 | 1.81e-05 | 5.13e-07 | bit-identical |
| **complete PSP synthetic (28²)** | 5.54e-07 | 1.16e-06 | 1.67e-05 | 1.67e-05 | 2.76e-07 | bit-identical |
| **complete PSP real canonical (28²)** | 6.44e-07 | 1.3e-06 | 1.57e-05 | 1.81e-05 | 5.13e-07 | bit-identical |
| **complete PSP large real crop (56²)** | 6.74e-07 | 1.34e-06 | 1.76e-05 | 1.6e-05 | 5.56e-07 | bit-identical |
| **complete PSP swath edge (7²)** | 5.19e-07 | 1.08e-06 | 1.54e-05 | 1.36e-05 | 3.77e-07 | bit-identical |

Layout and membership rows (exact required, all exact): pooling bins at 7², 14², 28², 56²
(200 cells, every member set; window / global / dense on ID sums), 28 → 6 overlap cells,
resize coordinate ramps (8 shapes, bit-identical to the mirror), PSP concatenation IDs
(1,404,928 values), output-conv im2col IDs (77,616 values) and weight rows, georeferenced
output = window output (200,704 values), asset grids, 40 / 40 footprints, parameters
(4,921,856 values bit for bit).

## Answers

| # | Question | Answer |
|---|---|---|
| 1 | What exact installed SMP modules form the PSP path? | `UPerNetDecoder.feature_norms[3]` (LayerNorm2d 768, eps 1e-6) then `PSPModule` (smp 0.5.0 `decoders/upernet/decoder.py`): four `Sequential(AdaptiveAvgPool2d(s), Conv2dReLU(768→256, 1×1))`, `F.interpolate(bilinear, align_corners=False)`, `torch.cat`, `Conv2dReLU(1792→256, 3×3, padding 1)`; `Conv2dReLU` = Conv2d(bias=False) → BatchNorm2d(eps 1e-5) → ReLU (§1). |
| 2 | Exact branch order? | **1, 2, 3, 6**; concatenation `[C4_norm, b1, b2, b3, b6]` (channels 0-767, 768-1023, 1024-1279, 1280-1535, 1536-1791), proven with unique channel IDs in SMP and in EE (§1, §5). |
| 3 | Exact adaptive-pooling output sizes? | 1 × 1, 2 × 2, 3 × 3, 6 × 6 of the C4 map, whatever its size (7² … 56² tested) (§2). |
| 4 | Exact PyTorch bin rule? | Output i of an N → M axis averages inputs **[⌊iN/M⌋, ⌈(i+1)N/M⌉)**; a cell averages over its own kh × kw; read off the kernel with one-hot inputs and equal to the written rule for every tested shape (§2). |
| 5 | How does 28 → 6 overlap? | Bins [0,5) [4,10) [9,14) [14,19) [18,24) [23,28): sizes 5, 6, 5, 5, 6, 5; rows/columns 4, 9, 18, 23 belong to two bins; cells hold 25, 30 or 36 pixels (28 → 3 overlaps at 9, 18) (§2). |
| 6 | Can adaptive pooling be reproduced exactly in native EE? | **Yes.** Every bin's exact member set from EE at 7², 14², 28², 56²; values bit-identical to the mirror and, on ordinary data, to the correctly rounded mean (§2, §3). |
| 7 | What summation/rounding order is required? | EE: double sums, sequential over each window's tokens, then sequentially over row-major windows; ÷ area in double; one float32 rounding (proved with order-revealing fixtures). PyTorch (channels-last, as SMP feeds it): sequential row-major float32 sums, one float32 ÷ area; size 1: `torch.mean`. The difference is PyTorch's float32 rounding (§3). |
| 8 | Maximum adaptive-pooling error? | Real C4: **1.9e-6** (canonical) / **3.6e-6** (56²) vs SMP float32, ≤ 2.7e-7 vs float64; synthetic: 0 (dyadic values); bit-identical to the mirror everywhere (§3). |
| 9 | Preferred EE pooling representation? | **Window layout**: per-window 0/1 membership product, `sampleRectangle` gather of the partials, sequential sum, ÷ area; the pooled maps as an `ee.Array` value. 1.8 EECU·s at 28² (dense `reduceRegions` 4.4; global gather 1.3 but O(H W) tables) (§4). |
| 10 | Can all four pooled Conv-BN-ReLU branches reproduce SMP? | **Yes**: all 12 intermediates bit-identical to the mirror, ≤ 4.3e-6 (canonical) / 5.2e-6 (56²) from SMP float32 on real C4 (§5). |
| 11 | Can all four branch resizes reproduce SMP? | **Yes**: coordinate ramps prove PyTorch's source-coordinate rule and edge clamp for 1/2/3/6 → 28 and → 56; 256-channel resizes ≤ 3.5e-6, bit-identical to the mirror (§5). |
| 12 | Is PSP concatenation order exact? | **Yes**, 1,404,928 unique IDs (§5). |
| 13 | Can the 1792 → 256 3 × 3 Conv-BN-ReLU run at canonical scale? | **Yes**, interactively (28-42 EECU·s, 314 MiB) and in batch (task C 17.3 EECU·s), with Milestone 7's window im2col; im2col order exact on IDs (§5, §9). |
| 14 | Its numerical error? | conv 2.3e-5, BN 2.5e-5, ReLU 1.6e-5 vs SMP float32; ≤ 5.2e-7 vs float64; bit-identical to the mirror (§5). |
| 15 | Does the complete PSP reproduce SMP? | **Yes**, on synthetic 14² / 28², real canonical, swath edge and the real 1792 crop (56²), every intermediate: ≤ 1.76e-5 at the output, ≤ 2.5e-5 anywhere (§6). |
| 16 | Final canonical PSP max absolute error? | **1.57e-5** vs SMP float32 (5.1e-7 vs float64; every element; bit-identical to the mirror) (§6). |
| 17 | Does the PSP output occupy exactly the C4 grid? | **Yes**: EPSG:32651, 320 m, the crop's origin, 28 × 28; 40 / 40 footprints; georeferenced values = window output bit for bit; both task-C assets (28², 56²) have all 256 bands on the C4 grid (§6, §10). |
| 18 | Does the mirror explain all residual differences? | **Yes**: every EE tensor bit-identical to the mirror, so EE − SMP is SMP's float32 arithmetic (§12). |
| 19 | PSP parameters and embedded values? | 4,921,856 parameters (C4 LayerNorm2d included); **5.13 M** embedded values at 28² (5.25 M at 56²) (§7). |
| 20 | Does PSP fit comfortably inside task C? | **Yes**: 5.2 M with every feature norm, 22% of the ~24 M cap; task C exports succeed at 28² and 56², and with all four feature norms' constants added (5.28 M) in batch (§7, §10). |
| 21 | What dominates PSP memory? | The 3 × 3 output conv's halo assembly of 1,792-channel windows: 314 MiB (28²), 322 MiB (56²); pooling ≤ 123 MiB (the 56² gather) (§9). |
| 22 | What dominates PSP compute? | The 1792 → 256 3 × 3 conv: ~85-90% of the PSP's EECU·s (§9). |
| 23 | Does canonical PSP run interactively? | **Yes**: 28-45 EECU·s, 12-25 s (§9). |
| 24 | Does 56 × 56 C4 PSP run interactively or batch-only? | **Both**: interactively 97-103 EECU·s, ~55 s, 322 MiB; batch 64.7 EECU·s (§9). |
| 25 | Any reason to abandon window-layout decoder compute? | **No**: window pooling is cheaper than dense `reduceRegions` (1.8 vs 4.4 EECU·s) and immune to masks; the window im2col output conv costs half the dense formulation (42 vs 85 EECU·s) (§4). |
| 26 | How does PSP output change for the same interior inside different crop extents? | 448 tile vs 1792 area: mean relative L2 0.32 end to end, 0.29 with the encoder fixed (cosine 0.94); 896 vs 1792: 0.13. Border ring ~0.55 (zero padding); interior ~0.19 (448) / ~0.06 (896) everywhere, not decaying with distance (§11). |
| 27 | Is finite-tile PSP materially different from larger-area PSP? | **Yes**, for any halo: the pooled statistics change by 37-47% (448 vs 1792) and that change reaches every token (§11). |
| 28 | Can whole-area pooled summaries be computed once and reused by tiled decoder tasks? | **Yes, exactly**: 50 × 256 branch maps + the tile's C4 with a one-window ring + absolute resize rows reproduce the whole-area PSP bit for bit, live on three tiles of the 1792 area (interior and corner) (§11). |
| 29 | What does such persistence cost? | Export 5.2 EECU·s (31 s run), a 32 KB asset, read 0.1 EECU·s; a tile task then needs no whole-area C4 (118 vs 172 invocations, 280 vs 566 MiB for a 448 tile) (§11). |
| 30 | How should PSP be defined for later map production? | **Option A: the tile is the model input**, with tiles equal to the training chip size (the canonical 896 crop until a checkpoint exists); the PSP ends the halo-for-exactness approach; Option C (persisted pooled branches) is the proven exact alternative if a model is ever trained to pool over areas; Option B is not recommended (§14). |
| 31 | Does valid-fraction metadata need new handling at PSP? | **As metadata only**, plus one global value: after pooling every token depends on the map's (or bin's) mean validity, so tiles should carry their C4 mean valid fraction as QC (§8). |
| 32 | New EE behaviours that could invalidate M9 FPN composition? | **None**: the new behaviours (array reducers, `sampleRectangle`, value-space order, PyTorch's stride-dependent kernel) concern pooling only; FPN reuses Milestone 7's window primitives on the PSP's window-layout output (§13). |
| 33 | Proceed to M9 (the FPN top-down pathway)? | **Yes**: pooling exact, branches and full PSP within 1.8e-5 and bit-identical to the mirror, C4 grid exact, 22% of the cap, interactive at 28² and 56², window layout confirmed, semantics decided (§14). |

## 1. The installed PSP and its contract (Phases 1, 9; Tests 1, 8)

Read and run from the installed packages
(reference/smp_psp_reference.py (archive `upernet/reference/smp_psp_reference.py` at `research-archive-2026-10-12`) `contract`; tests in
test_upernet_psp_contract.py (archive `tests/test_upernet_psp_contract.py` at `research-archive-2026-10-12`)):

| item | installed (smp 0.5.0, torch 2.14.0) |
|---|---|
| module | `segmentation_models_pytorch/decoders/upernet/decoder.py`, class `PSPModule` (lines 10-54), built by `UPerNetDecoder.__init__` with in 768, out 256, sizes (1, 2, 3, 6); called as `self.psp(features[-1])` after `feature_norms` |
| classes | `AdaptiveAvgPool2d`, `Conv2dReLU` (= `Sequential(Conv2d, BatchNorm2d, ReLU)`), `F.interpolate`, `torch.cat`; nothing else |
| pool sizes | **1, 2, 3, 6**, in this order (`blocks[0..3]`) |
| branch block | Conv2d 768 → 256, 1 × 1, stride 1, padding 0, **bias=False**; BatchNorm2d(256, eps 1e-5, affine, running stats); ReLU(inplace=True) |
| resize | `F.interpolate(pooled, size=(H, W), mode="bilinear", align_corners=False)`: four calls, 1/2/3/6 → H × W (recorded at run time) |
| concatenation | `torch.cat([feature, up_1, up_2, up_3, up_6], dim=1)`: channels 0-767 C4_norm, 768-1023 branch 1, 1024-1279 branch 2, 1280-1535 branch 3, 1536-1791 branch 6 (**1,792**; unique-ID run) |
| output conv | `Conv2dReLU`: Conv2d 1792 → 256, 3 × 3, padding 1 (zeros), **bias=False**; BatchNorm2d(256, eps 1e-5); ReLU |
| input | `feature_norms[3](C4)`: LayerNorm2d(768, eps 1e-6) of the raw stage-4 output, whose output is a **channels-last** view (strides (H W C, 1, W C, C)) |

**Expanded forward.** `expanded` runs the module's own submodules one at a time
(pool, conv, BN, ReLU, resize per branch; cat; conv, BN, ReLU) and raises unless each branch
equals `block(feature)` and the result equals `psp(feature)` bit for bit, in float32 and
float64, at 7², 14², 28² and 56² (SMP's ReLU is in place, so the expansion applies
`torch.relu` to a kept tensor). In a real UPerNet forward a pre-hook proves the PSP
receives exactly `feature_norms[3](C4)`, channels-last, and a forward hook that it returns
the expanded output. **Concatenation order (Test 8):** in a copy of the PSP whose feature
carries c + 1 in channel c and whose branch k outputs 1000 (k + 1) + j (zero conv weights,
BatchNorm bias = ID), the tensor entering `out_conv` holds exactly the IDs in the order
above. Parameters: 4 × (196,608 + 4 × 256) + (4,128,768 + 4 × 256) + 2 × 768 =
**4,921,856** (C4 LayerNorm2d included).

**Statistics (Phase 1),** SMP float32, real canonical crop (all intermediates in
`results/raw/m8_stats_canonical_896.json`, synthetic fixtures beside it):

| tensor | shape | min | max | mean | std |
|---|---|---:|---:|---:|---:|
| C4_raw | 28 × 28 × 768 | −11.12 | 9.39 | 0.019 | 2.57 |
| C4_norm | 28 × 28 × 768 | −4.87 | 5.29 | −0.011 | 1.05 |
| pool_1 / pool_6 | 1 × 1 / 6 × 6 × 768 | −3.46 / −4.54 | 3.98 / 4.33 | −0.011 | 1.01 / 1.04 |
| branch_6_relu / branch_6_up | 6 × 6 / 28 × 28 × 256 | 0 | 3.46 / 3.34 | 0.40 | 0.62 / 0.61 |
| psp_concat | 28 × 28 × 1792 | −4.87 | 5.29 | 0.24 | 0.87 |
| psp_conv / psp_bn | 28 × 28 × 256 | −3.47 / −3.82 | 2.82 / 3.67 | −0.046 / −0.010 | 0.86 / 0.92 |
| **PSP_output** (psp_relu) | 28 × 28 × 256 | 0 | 3.67 | 0.36 | 0.52 |

All float32.

## 2. Adaptive average pooling: the bins (Phases 2, 5; Tests 2-3)

**The rule.** ATen's integer `start_index` / `end_index`: along an axis of N inputs pooled
to M outputs, output i averages inputs **[⌊iN/M⌋, ⌈(i+1)N/M⌉)**, and a 2-D cell is the product
of its row and column bins averaged over **its own area** kh × kw. The rule is written out
(`adaptive_pool_reference.bins`) and independently read off `F.adaptive_avg_pool2d` with
one-hot rows (`onehot_bins`: members and the weight 1 / count); the two agree for every
N ∈ {7, 14, 28, 56} and M ∈ {1, 2, 3, 6} (offline Test 2).

| N → M | bins | overlaps |
|---|---|---|
| 28 → 1 | [0,28) | — |
| 28 → 2 | [0,14) [14,28) | — |
| 28 → 3 | [0,10) [9,19) [18,28) (10 each) | 9, 18 |
| **28 → 6** | **[0,5) [4,10) [9,14) [14,19) [18,24) [23,28)** (5, 6, 5, 5, 6, 5) | **4, 9, 18, 23** |
| 56 → 3 | [0,19) [18,38) [37,56) (19, 20, 19) | 18, 37 |
| 56 → 6 | [0,10) [9,19) [18,28) [28,38) [37,47) [46,56) (10 each) | 9, 18, 37, 46 |
| 14 → 6 | [0,3) [2,5) [4,7) [7,10) [9,12) [11,14) | 2, 4, 9, 11 |
| 7 → 6 | [0,2) [1,3) [2,4) [3,5) [4,6) [5,7) | 1, 2, 3, 4, 5 |

So 28 → 6 is **not** a partition: 2-D cells hold 25, 30 or 36 pixels, the rows and columns
4, 9, 18, 23 feed two bins each, and pixel (4, 9) feeds four cells. Non-overlapping
averaging would be wrong for 3 and 6 at 28 (and at 56); only 1 and 2 partition.

**From EE, every bin (Tests 2-3).** The fixture token t = iW + j carries t + 1 in channel t
and 0 elsewhere (generated on the server in window layout; 784 channels at 28², 3,136 in
four chunks at 56²). EE's pooled *sums* are then (t + 1) exactly where t is in the bin: the
exact source set of every output cell, with no possible cancellation. At 7², 14², 28² and
56² all 50 stacked bins (sizes 1, 2, 3, 6) equal the membership × ID matrix exactly, and
the pooled means equal float32((t + 1) / area) exactly; from EE's sums alone, every cell's
row start/end, column start/end, count and member set equal PyTorch's rule
(`results/raw/m8_pool_bins.json`, 200 cells). A row/column/ID/count fixture gives exact
sums in all three EE formulations (window, global, dense), and the dense formulation's
counts equal the bin areas.

## 3. Floating-point semantics of the pooling (Phases 3, 6, 12; Test 4)

**PyTorch (characterized against emulations,
adaptive_pool_reference.characterize (archive `upernet/reference/adaptive_pool_reference.py` at `research-archive-2026-10-12`)).** The
float32 CPU kernel depends on the output size and on the input's memory format, which
PyTorch infers from the strides:

| case | float32 semantics (bit-identical emulation, 7² … 56², 64-768 channels) |
|---|---|
| s = 1 | `mean(-1, -2)`: torch's own vectorized reduction order of the sum, then one division by H W |
| s > 1, **channels-last** input (what SMP feeds the PSP) | per cell a float32 sum, sequential in row-major (row, column) order, vectorized across channels, then **one** division by kh kw |
| s > 1, NCHW-contiguous input | the same sum, divided by kh, then by kw |

A trap found on the way: `torch.from_numpy(x).permute(2, 0, 1)[None]` (an NHWC array
viewed as NCHW) reports `is_contiguous(memory_format=channels_last) == True`, yet PyTorch
dispatches it to the NCHW kernel (its batch stride is C, not H W C), and
`.contiguous(memory_format=channels_last)` returns it unchanged. A reference that
rebuilds SMP's input that way silently uses the other division. The references here
build the input with SMP's exact strides (`as_nchw`), and the PSP references run the
real `feature_norms[3]` output.

**EE.** The EE pooling accumulates in double and rounds once:
1. per window, the [50, 49] 0/1 membership times the window's [49, 768] tokens
   (`matrixMultiply`: sequential over the window's tokens, 7a + b);
2. the window partials gathered row-major and summed over the window axis (`ee.Array.reduce`
   on the gathered value: sequential, window I n_w + J);
3. ÷ kh kw in double (exact integers; 1/area weights would round before the sum), one
   float32 rounding.

**The order, established empirically.** Double sums of ≤ 3,136 float32 values of ordinary
range are exact, so on ordinary data every order gives the same float32 mean (the
mirror's window, global and row-major orders and math.fsum agree bit for bit on the
synthetic fixtures). Order-revealing fixtures therefore put ±2⁴⁰-scale pairs that cancel
inside every bin of one pool size (in each bin's exclusive interior), so the low bits of the
N(0, 1) values absorbed by the running sum depend on the order. On 8 fixtures (28² and 56²,
s = 1, 2, 3, 6; embedded in the request) EE's window formulation equals the mirror's
window-partial order on **100%** of outputs and EE's global formulation the single
sequential (window, token) dot on 100%, while every other order (row-major, exact sum, the
other formulation) matches at most 71%. So value-space `ee.Array.reduce(sum)` and
`matrixMultiply` sum sequentially in double, as Milestone 5 found for images.

**Errors (Phase 6).** Every EE pooled map is bit-identical to the mirror (window, global
and dense formulations; 14², 28², 56² synthetic; real C4 below). Against PyTorch:

| input | max \|EE − torch32\| | max \|EE − float64\| | EE vs mirror |
|---|---:|---:|---|
| synthetic C4 14² / 28² / 56² (hash × 3: dyadic values, PyTorch's float32 sums are exact too) | 0 | ≤ 8e-8 | bit-identical |
| real canonical C4_norm, pool 1 / 2 / 3 / 6 | 7.2e-7 / 1.9e-6 / 1.9e-6 / 1.7e-6 | ≤ 2.7e-7 | bit-identical |
| … same input (SMP float32 on EE's exact C4) | 2.4e-7 / 1.9e-6 / 2.2e-6 / 7.2e-7 | | |
| real 1792 C4_norm (56²), pool 1 / 2 / 3 / 6 | 4.8e-7 / 3.6e-6 / 2.6e-6 / 1.4e-6 | ≤ 2.2e-7 | bit-identical |

EE's pooled maps are the correctly rounded means wherever the double sum is exact (all
ordinary data); PyTorch's differ by its float32 accumulation and division. Target 1e-4: met
by ~50×.

## 4. Representation: where and how to pool (Phases 4, 25)

Three formulations, all exact on the ID fixtures and bit-identical to their mirrors
(adaptive_pool.py (archive `upernet/earth_engine/adaptive_pool.py` at `research-archive-2026-10-12`)):

- **window (A + B, production).** Per window the [50, 49] membership is built on the server
  from per-axis tables Ry [50, H], Rx [50, W] (50 (H + W) embedded values; any H, W that are
  multiples of 7), times the window's tokens: 50 partial sums. `Reducer.sum` rejects
  array inputs ("Reducer input must be a number"), so the 16 (or 64) window partials are
  gathered with `sampleRectangle` (row-major, deterministic; it *fails* on a masked window
  instead of dropping it) and summed in value space.
- **global.** The window-layout tokens themselves gathered into one [n 49, 768] value, one
  [50, n 49] product with the global 0/1 membership (50 n 49 embedded values: 39 k at 28²,
  157 k at 56²).
- **dense (C, baseline).** C4_norm as 768 scalar bands, `reduceRegions` over the 50 bin
  rectangles with an unweighted sum and a count; mean = sum / count, so EE masks enter the
  denominator (§ 8).

Cost: one profiled interactive request each (C4 read + LayerNorm2d + all 50 bins, a
checksum of the pooled value):

| formulation | canonical 28² EECU·s / peak MiB / request | 1792 crop 56² EECU·s / peak / request |
|---|---|---|
| (C4 read + LayerNorm2d only) | 1.14 / 5 / 54 KB | 3.02 / 19 / 54 KB |
| **window** | **1.79** / 32 / 65 KB | **3.45** / 123 / 67 KB |
| global | 1.33 / 19 / 252 KB | 5.05 / 75 / 840 KB |
| dense reduceRegions | 4.40 / 16 / 96 KB | 5.46 / 64 / 96 KB |

Pooling costs 0.2-0.7 EECU·s on top of the LayerNorm in window form (identical repeats of
these requests vary up to ~1.5×). The global form needs a table that grows with the map
(request 13× larger at 56²); the dense form costs 2.5× (canonical) and needs a window → dense
conversion of the 768-channel map, then hands back a value anyway. **The window formulation
is preferred**: it starts from the decoder's layout, its constants grow with H + W, its
order is fixed, and it cannot drop masked data silently. The pooled maps then live as an
`ee.Array` value: the branch convolutions run once on [s², 768] instead of per pixel, and
Milestone 7's matrix resize takes exactly such a value.

**The output conv (Phase 25).** The whole PSP up to the concatenation plus the 1792 → 256
3 × 3 conv, canonical scale: **window im2col 42.2 EECU·s**, 314 MiB; dense
`neighborhoodToArray` on 1,792 dense bands 84.6 EECU·s, 309 MiB. PSP gives no reason to
leave window layout.

## 5. Branches, resizes, concatenation and the output conv (Phases 7-10; Tests 6-10)

**Pooled branches (Phase 7).** Each pooled [s², 768] value goes through Milestone 2's
`linear` (W^T from the asset, no bias), Milestone 7's explicit inference BatchNorm (rows
s²) and ReLU, run through `BlockArrayOps` on the value: each runs once per request, not per
pixel. All 12 intermediates (conv, BN, ReLU × 4 sizes) bit-identical to the mirror on every
fixture and the real crops; real canonical: ≤ 4.3e-6 from SMP float32 (branch 2's conv),
≤ 4.1e-7 from float64. BatchNorm stays explicit: the PSP's 5.1 M embedded values leave no
constant-budget reason to fold (§ 7).

**Branch resizes (Phase 8, Test 7).** `resize.window_matrix_resize` is Milestone 7's matrix
formulation emitted in window layout: the [n 49, s²] rows of PyTorch's own weights
(products of its float32 1-D weights, exact in double) in window-token order, 49 rows per
window, times the [s², 256] branch value. Coordinate ramps (channels = source row and
column) through 1/2/3/6 → 28 and → 56 return PyTorch's clamped source coordinate
(o + ½) s / H − ½ at every output pixel (edges included; ≤ 2.7e-7 from the formula, ≤ 4.8e-7
from `F.interpolate` in float32), bit-identical to the mirror; full 256-channel resizes on
real C4 ≤ 3.5e-6 from SMP. The resized maps land on the C4 token grid by construction (the
tokens of the input window grid), and the concatenation's grid guard checks it. For
Option C the rows are taken at **absolute** positions of the whole-area grid (§ 11).

**Concatenation (Phase 9, Test 8).** Milestone 7's `feature_ops.concat` (arrayCat on the
channel axis, grid guard) on IDs (row-major token × 1792 + c + 1, 5 parts of 768 / 256 /
256 / 256 / 256): 1,404,928 values, exact, in SMP's order.

**The 1792 → 256 conv (Phase 10, Test 9).** Milestone 7's window im2col unchanged: the 9 × 9
halo of the [49, 1792] concat from the 8 neighbouring windows (zeros outside the raster),
[49, 16,128] in (ky, kx, c) order, one product with W^T [16,128, 256] permuted on the
server from the asset's (c, ky, kx) rows. On the ID concat, all 9 taps × 11 channels at
every segment boundary (0, 1, 767, 768, 1023, 1024, 1279, 1280, 1535, 1536, 1791) equal
PyTorch's zero-padded im2col at every position of the 28² map (77,616 values, exact),
and the server-permuted weight rows for the same 99 (tap, channel) pairs equal
`weight.permute(2, 3, 1, 0)` bit for bit. Numerics on real C4: conv 2.3e-5, BN 2.5e-5,
ReLU 1.6e-5 from SMP float32 (3.4e-7 / 5.2e-7 / 5.1e-7 from float64), bit-identical to the
mirror: PyTorch's error is its float32 accumulation of 16,128-term dot products.

## 6. The complete PSP (Phases 11, 13, 14, 17; Tests 10-13, 15)

validate_psp.py (archive `upernet/earth_engine/validate_psp.py` at `research-archive-2026-10-12`), every intermediate fetched and compared
(references: SMP float32 on its own C4, SMP float64, the mirror, and SMP float32 on the exact
C4 values EE reads):

| input | grid | tensors | max \|EE − torch32\| (output) | worst intermediate | max \|EE − float64\| | EE vs mirror |
|---|---|---:|---:|---:|---:|---|
| synthetic 14² (2 × 2 windows) | abstract | 25 | 1.50e-5 | 2.07e-5 (psp_bn) | 3.7e-7 | 25 / 25 bit-identical |
| synthetic 28² (canonical) | abstract | 25 | 1.67e-5 | 1.86e-5 (psp_bn) | 2.8e-7 | 25 / 25 |
| **real canonical 896 crop** (task B' asset) | EPSG:32651, 320 m | 26 | **1.57e-5** | 2.50e-5 (psp_bn) | 5.2e-7 | **26 / 26** |
| real swath-edge 224 crop (7², dense fill) | EPSG:32651, 320 m | 26 | 1.54e-5 | 1.54e-5 | 7.3e-7 | 26 / 26 |
| real 1792 crop (56², Milestone 6 asset) | EPSG:32651, 320 m | 25 (all but the concat) | 1.76e-5 | 2.05e-5 (psp_bn) | 7.7e-7 | 25 / 25 |

The "same input" reference shows the PSP's own share: on the canonical crop SMP float32 on
EE's exact C4 is 1.8e-5 from EE at the output, so the residual is PyTorch's float32
arithmetic inside the PSP, not the encoder's.

**Geospatial correctness (Phase 14, Test 13).** The output in window layout → dense 256-band
raster (Milestone 4's depth_to_space) → relabelled onto the crop's C4 grid: projection
EPSG:32651, transform [320, 0, 283200.953…, 0, −320, 2645318.840…], 28 × 28 = C4's grid
exactly; 40 / 40 random 10 m native pixel centres sample the C4 pixel (row // 32,
col // 32); `computePixels` on that grid returns the window-layout output bit for bit. The
same holds for the 1792 crop (56 × 56, transform [320, 0, 278720.953…, 0, −320,
2649798.840…]), fetched in blocks of 7 rows: one `computePixels` request for the whole 56²
output exceeds interactive memory ("User memory limit exceeded."; each block recomputes only
its windows' halos), while row-of-windows requests run at ≤ 720 MiB. No
EE resampling is involved (every step is a relabel or an exact stride sample; Milestone 7's
grid guards refuse mismatched operands).

## 7. Constant budget, parameters and assets (Phases 15, 23; Test 14)

Embedded values of the PSP graph in window layout (explicit BatchNorm), counted from the
constructors (`psp.budget`) and recorded with every task-C export:

| component | raw parameters | embedded at 28² | embedded at 56² |
|---|---:|---:|---:|
| C4 LayerNorm2d (rows 49) | 1,536 | 75,264 | 75,264 |
| pooling tables Ry, Rx (+ areas) | — | 2,850 | 5,650 |
| branch 1 × 1 convs (4 × 768 × 256) | 786,432 | 786,432 | 786,432 |
| branch BatchNorms (rows s²) | 4,096 | 51,200 | 51,200 |
| branch resize tables | — | 39,200 | 156,800 |
| output 3 × 3 conv | 4,128,768 | 4,128,768 | 4,128,768 |
| output BatchNorm (rows 49) | 1,024 | 50,176 | 50,176 |
| **PSP** | **4,921,856** | **5,133,890** | **5,254,290** |
| + C1-C3 LayerNorm2d (task C's other inputs) | 1,344 | 65,856 | 65,856 |

Task C's whole decoder was 9.85 M in Milestone 7; the PSP is 52% of it, and the PSP with
every feature norm is 5.2 M, **22% of the ~24 M cap**. Decoded parameters at Milestone 7's
~64 bytes per value: ~315 MiB. Graphs: task C canonical 471 KB / 149 invocations, 56² 1.28 MB
/ 148 (the resize tables are embedded as JSON doubles: 39 k / 157 k values). **Live**: the
canonical task-C export carrying, in addition, the constants of all four LayerNorm2d
(× 0; 5.28 M embedded in total) succeeds in batch (27.2 EECU·s, 80 s run), as does every
task-C export here; Milestone 7 placed the failure between 24 and 26 M.

**Parameters (Phase 23).** No second copy was uploaded: the PSP reads Milestone 7's decoder
assets (groups `feature_norms`, `psp_pooled_1x1`, `psp_out_conv_3x3`). Its 27 tensors (four
1 × 1 weights [256, 768] and their 16 BatchNorm vectors, the 3 × 3 weight [256, 16,128] in
PyTorch's (c, ky, kx) flattening and its 4 BatchNorm vectors, C4's LayerNorm2d gamma and
beta) were read back live: **all 4,921,856 values bit-identical to the float32 originals**,
and the originals' SHA-256 digests equal Milestone 7's manifest. The 3 × 3 weight is
permuted to the im2col row order on the server (§5).

## 8. Validity and masks (Phase 21; Test 16)

The swath-edge crop (224 × 224 px, 49% valid input) is dense after Milestone 6's
training-mean fill; its C4 (7 × 7, one window) went through the PSP in EE and matched SMP
on the same dense tensor (1.54e-5, bit-identical to the mirror, § 6). Its C4 valid fractions
(exact means of the skips asset's C1 fractions) are 0 on 20 tokens, partial on 9, 1 on 20:
**49.5% of the map's input is fill.**

No mask reaches the pooling: the window and global formulations divide by the constant bin
areas and never read masks, and the gather *refuses* a masked window ("Image.sampleRectangle:
Fully masked pixels … with no default value set"), live. The hazard is real for the dense
alternative: masking the 29 not-fully-valid C4 tokens and pooling with `reduceRegions`
shrinks the counts (global bin 49 → 20) and moves the pooled means by up to 3.76 from
PyTorch's dense-tensor result (negative control).

**Interpreting valid_fraction after global pooling.** PSP mixes the whole map into every
token, so a token's validity is no longer local: its PSP output depends on (i) its own
3 × 3 neighbourhood of C4 (local validity, as for every decoder op) and (ii) the pooled
statistics, whose input validity is the **map-level** fraction (49.5% here; per pool bin, the
mean valid fraction over that bin). Recommendation: keep `valid_fraction` metadata-only
(the reference never sees it), and carry a second, global QC value per inference tile, the
mean valid fraction of the tile's C4 (or per pooled bin), so that outputs of tiles
dominated by fill are flagged even where the local fraction is 1.

## 9. Cost (Phase 16)

Cumulative prefixes on the real C4 assets, one profiled interactive request each (checksum
of the prefix's last tensor; identical requests vary up to ~1.5×):

| prefix | canonical 28² EECU·s | peak MiB | embedded | 1792 crop 56² EECU·s | peak MiB |
|---|---:|---:|---:|---:|---:|
| C4 read (window layout) | 1.14 | 3 | 0 | 1.54 | 9 |
| + LayerNorm2d | 1.32 | 5 | 75 k | 3.52 | 19 |
| + pool 1 | 1.27 | 22 | 75 k | 2.45 | 85 |
| + pool 2 | 1.38 | 23 | 76 k | 2.82 | 88 |
| + pool 3 | 2.22 | 25 | 76 k | 4.80 | 95 |
| + pool 6 | 1.65 | 32 | 78 k | 2.61 | 123 |
| + branch convs | 3.12 | 83 | 865 k | 3.80 | 123 |
| + branch BN / ReLU | 3.32 | 48 | 916 k | 5.67 | 123 |
| + branch resizes | 3.57 | 86 | 955 k | 8.61 | 123 |
| + concat | 4.65 | 50 | 955 k | 10.20 | 123 |
| **+ output 3 × 3 conv** | **27.86** | **314** | 5.08 M | **103.3** | **322** |
| + output BN / ReLU (complete PSP) | 37.85 | 314 | 5.13 M | 97.3 | 322 |
| task C in batch (C4 asset → PSP asset) | **17.3** (run 34 s, queue 3 s) | — | 5.13 M | **64.7** (run 88 s, queue 4 s) | — |

1. **The 1792 → 256 3 × 3 conv dominates** compute (~85-90% of the PSP's EECU·s) and memory
   (the halo assembly of 1,792-channel windows: 314-322 MiB).
2. **Adaptive pooling is negligible**: its prefixes sit within the ~1.5× noise of identical
   requests (pool 6 even below the LayerNorm prefix at 56²); the pooling benchmark proper
   (§ 4) puts all 50 bins at 0.4-0.7 EECU·s over the LayerNorm, at 28² and 56².
3. **The branch resizes are negligible**: ~0.3 EECU·s at 28², ~3 at 56² (their tables grow
   with H W s²; still small).
4. **The canonical PSP runs interactively** (28-45 EECU·s, 12-25 s, 314 MiB).
5. **The 56² PSP still runs interactively** (97-103 EECU·s, ~55 s, 322 MiB: memory follows
   the windows a request computes, not the map) and in batch (65 EECU·s). Batch is not
   required at these sizes; it is for production exports, and EECU·s grow ~4× with 4× the
   tokens.

## 10. Task C prototype (Phase 24; Test 18)

psp_pipeline.export_psp (archive `upernet/earth_engine/psp_pipeline.py` at `research-archive-2026-10-12`): C4 asset → window layout (49
exact stride samples) → LayerNorm2d → PSP → window → dense 256-band raster → relabel onto
the crop's C4 grid → `Export.image.toAsset` on exactly that grid.

| crop | EECU·s | run s | queue s | asset | request / invocations | embedded | exported values |
|---|---:|---:|---:|---|---|---:|---|
| canonical 896 (28²) | **17.3** | 34 | 3.1 | 351 KB, 256 bands, C4 grid | 471 KB / 149 | 5.13 M | 200,704: 1.57e-5 from SMP float32, bit-identical to the mirror and to the interactive output |
| 1792 (56²) | **64.7** | 88 | 4.0 | 1.27 MB, 256 bands, C4 grid | 1.28 MB / 148 | 5.25 M | 802,816: 1.76e-5 from SMP float32, bit-identical to the mirror and to the interactive output |
| canonical + all feature-norm constants | 27.2 | 80 | 8.0 | (the same values) | 486 KB / 163 | 5.28 M | budget probe (§7) |

Every band of each asset has CRS EPSG:32651, the crop's C4 transform and 28 × 28 (56 × 56)
dimensions. Batch is cheaper than the interactive checksum of the same graph (17 vs 28-38
EECU·s at 28²): one evaluation instead of a profiled interactive request. This export
demonstrates task C; production UPerNet keeps the PSP output inside task C for the FPN.

## 11. PSP is global: finite tile vs map production (Phases 18-20; Test 17)

Adaptive average pooling over the whole C4 map makes every PSP output depend on the whole
map supplied to the module. Measured on Milestone 6's nested real crops, centred on Sun
Moon Lake with offsets of whole stage-4 windows (small 448 px ⊂ canonical 896 ⊂ large
1792; C4 offsets (7, 7), (21, 21), (14, 14) tokens), over the smaller crop's tokens
(reference/psp_semantics.py (archive `upernet/reference/psp_semantics.py` at `research-archive-2026-10-12`), SMP float32 and the mirror;
`results/raw/psp_semantics.json`; the same comparisons on the EE outputs of every crop's own
C4 and of the sliced 1792 C4, `m8_semantics_ee.json`, agree with this table to five digits,
e.g. 0.28892 / 0.28892 and cosine 0.94144 / 0.94144 for the sliced 448 tile):

| comparison (tile vs the larger area over the tile) | mean rel. L2 per token | median | mean cosine (min) | max \|Δ\| | mean \|Δ\| (output mean \|·\| 0.37) |
|---|---:|---:|---:|---:|---:|
| C4 itself: 448 crop vs 1792 crop (encoder edge effects) | 0.139 | 0.119 | 0.986 (0.930) | 3.29 | 0.28 (C4 mean 2.04) |
| C4 itself: 896 vs 1792 | 0.015 | 0.009 | 0.9998 (0.997) | 0.63 | 0.032 |
| **end to end**: PSP of the 448 crop's own C4 vs the 1792 crop's PSP | **0.320** | 0.266 | 0.934 (0.701) | 1.95 | 0.119 |
| end to end: 448 in 896 | 0.281 | 0.202 | 0.943 (0.703) | 1.94 | 0.104 |
| end to end: 896 in 1792 | 0.129 | 0.069 | 0.977 (0.722) | 2.05 | 0.048 |
| **PSP alone** (encoder fixed): PSP of the 1792 C4 sliced to 448 vs the 1792 PSP | **0.289** | 0.228 | 0.941 (0.705) | 2.03 | 0.108 |
| PSP alone: sliced to 896 | 0.128 | 0.067 | 0.977 (0.721) | 2.05 | 0.048 |

By distance to the tile edge (PSP alone; ring 0 = the tile's border tokens):

| tile | ring 0 | ring 1 | ring 2 | ring 3 | ring 4 | ring 5 | ring 6 | … ring 13 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 448 in 1792 (14²) | 0.551 | 0.193 | 0.194 | 0.192 | 0.198 | 0.197 | 0.197 | — |
| 896 in 1792 (28²) | 0.528 | 0.058 | 0.059 | 0.060 | 0.060 | 0.061 | 0.061 | 0.101 |

**Decomposition** (mirror; the 448 tile): (a) *padding* — the whole area's pooled branches
but the tile's own zero padding at its edge: nonzero **only on ring 0** (0.53 mean relative
L2 there, exactly 0 inside); (b) *pooling* — tile-pooled vs whole-area-pooled branches with
the same padding: **0.19-0.20 on every ring**, the global part. The pooled statistics
themselves move by 37% (1 × 1) to 47% (6 × 6) relative L2 between the 448 tile and the 1792
area (11-19% for the 896 tile). So the difference does not decay with distance from the
tile edge, and no finite halo removes it: **finite-tile PSP is materially different from
larger-area PSP** (Q26-27), roughly in proportion to how much the tile's statistics differ
from the area's (here 0.06 for an 896 tile, 0.19 for a 448 tile, inside the ring). These
magnitudes are for seeded weights; a trained model's sensitivity will differ, but the
structure (exact-zero beyond ring 0 for padding, global offset for pooling) is the model's.

**Option C is exact.** PSP output at a token = f(its 3 × 3 neighbourhood of C4_norm and of
the four resized branch maps), and the branch maps are 50 × 256 values computed from the
whole area. So a tile of the whole-area PSP needs only (i) the whole area's ReLU'd branch
maps, (ii) the tile's C4 with a one-token halo of real neighbours (zeros beyond the area),
(iii) the whole-area grid's resize rows at the tile's absolute positions. The mirror
reproduces the whole-area PSP over both nested tiles and over non-window-aligned tiles
**bit for bit** (offline test). In EE (psp_semantics.py (archive `upernet/earth_engine/psp_semantics.py` at `research-archive-2026-10-12`)
`tile_forward`): the tile plus one stage-4 window of real C4 around it (clipped to the area;
Milestone 7's halo assembly then supplies the neighbours unchanged), branches read from the
persisted asset, `window_matrix_resize` with absolute rows. **Live: three tiles of the 1792
area's PSP, all bit-identical to the whole-area EE output over the same tokens**: the 448
region (14², 50,176 values), the 896 region (28², 200,704) and the (0, 0) corner tile (14²,
whose ring is clipped by the area's edge, so the area's own zero padding is exercised).

**Pooled-statistics persistence (Phase 20).** The four ReLU'd branch maps of the 1792 area
(1 + 4 + 9 + 36 = 50 rows × 256) exported as a 50 × 1 pixel, 256-band float32 image asset
(`psp_pipeline.export_branches`): **5.2 EECU·s**, 31 s run, 3.6 s queue.

| measure | value |
|---|---|
| asset | 50 × 1 pixels × 256 float32 bands, **32,214 bytes** |
| export | 5.2 EECU·s, run 31 s, queue 3.6 s (graph 136 KB / 121 invocations) |
| read (`sampleRectangle` → [50, 256] value + checksum) | **0.10 EECU·s**, 0.4 s |
| exactness | read-back = in-graph branch maps = mirror, all 12,800 values bit for bit |
| tile task (448 tile of the 1792 area, checksum) with persisted maps | 31.6 EECU·s, 280 MiB, 483 KB / 118 invocations |
| … pooling the whole area's C4 inside the tile's graph instead | 33.4 EECU·s, 566 MiB, 534 KB / 172 invocations |

Persistence makes a tile task independent of the area's size: it reads 12,800 values
instead of the whole area's C4 (and halves the tile request's peak memory here); the
pooling itself is cheap at 56², so the saving grows with the area, not visible in EECU·s at
this size. It is exploratory: no permanent task boundary is added (Option A needs none).

## 12. The controlled-rounding mirror (Phase 12)

reference/ee_mirror_psp.py (archive `upernet/reference/ee_mirror_psp.py` at `research-archive-2026-10-12`) extends Milestone 7's mirror
with what the PSP adds: pooling in EE's order (per-window sequential double sums, then a
sequential sum over row-major windows, ÷ area, one float32 rounding; the global, row-major
and exact orders for comparison), the four branches in value space, the window matrix
resize (per output token a sequential double dot over the s² source pixels), the
concatenation and the window-im2col output conv, and `psp_tile` for Option C. **Every EE
floating-point tensor of this milestone is bit-identical to it**: pooled maps (synthetic,
order-revealing, real), all branch intermediates, resizes, concatenations, output convs,
BatchNorms and ReLUs on synthetic (50 / 50), canonical (26 / 26), edge (26 / 26) and 1792
(25 / 25) inputs. There is no first divergence to report; EE − SMP is SMP's float32 arithmetic.

## 13. Earth Engine and PyTorch behaviour found or confirmed (probed live)

1. **`Reducer.sum` rejects array inputs** ("Reducer input must be a number"), so array
   pixels cannot be summed across a region with `reduceRegion`. `sampleRectangle` returns
   array-valued pixels as one nested value in row-major pixel order (deterministic), and
   **fails on masked pixels** unless a default is given; `Reducer.toList` returns them too
   (order not guaranteed).
2. **Value-space array arithmetic matches image arithmetic**: `ee.Array.reduce(sum)` over
   an axis and `ee.Array.matrixMultiply` on computed values accumulate sequentially in
   double (order-revealing fixtures), as Milestone 5 found for array images.
3. **Computed values compose with images in batch.** A value gathered from an image
   (`sampleRectangle`), reduced and turned back into a constant image runs inside
   `Export.image.toAsset` (the gather probe, task C at 28² and 56², the branch export).
   Whether EE counts the gathered intermediate (602 k values at 28², 2.4 M at 56²) against
   the embedded-value cap was not isolated; even if it does, 56²'s 2.4 M + 5.3 M stays far
   below ~24 M.
4. **`ee.Array(list)` refuses a list of Arrays** ("Unrecognized type … found type Array");
   convert with `toList()` first (dense pooling's assembly).
5. **`reduceRegions` with an EE mask changes the denominator** of a mean (and an unweighted
   sum's count): masks must never reach pooling (they do not in window layout).
6. **PyTorch selects the pooling kernel from strides.** SMP's PSP input is channels-last
   (sequential float32 sum, one division by the area); an NCHW-contiguous input divides by
   kh then kw; an NHWC array viewed as NCHW without a copy reports channels-last contiguity
   but takes the NCHW kernel. Size 1 is `torch.mean`. References must reproduce the real
   memory format, not only the values.
7. **EE's interactive cache** answers identical graphs (Milestone 7): benchmark repeats need
   fresh inputs; first runs are recorded here.
8. **Interactive memory follows the windows a request computes**: the complete 56² PSP as one
   checksum (97-103 EECU·s, 322 MiB) or row by row (≤ 720 MiB with conv, BN and ReLU
   fetched together) runs, but the whole 56² × 256 output as one `computePixels` download
   does not; batch export has no such limit (64.7 EECU·s).

None of these invalidates the FPN composition of Milestone 9 (Q32): the FPN's resizes are
Milestone 7's window resizes on window layout, its additions and lateral convs per pixel,
and its only global input is the PSP output, already a window-layout map on the C4 grid.
The one semantic choice the FPN inherits is § 11's.

## 14. Recommendation

**PSP semantics for M9/M10 and map production (Q30): Option A, the tile is the model
input.** PSP pools over exactly the C4 map the model is given, as in fixed-size training
and in the project's own SegWater practice (independent 224 × 224 chips). Concretely:

- the inference tile must equal the training chip size of whatever checkpoint is used (no
  trained Swin-UPerNet exists yet; until one does, the canonical 896-px crop stays the
  experimental unit of M9/M10);
- **the PSP ends the "halo for exactness" programme of Milestones 6-7.** Their 672- / 896-px
  aligned halos made a tile's features equal whole-scene features; § 11 shows that no finite
  halo can do that past the PSP (the interior offset does not decay with distance). Under
  Option A the tile defines the model at the encoder too, with no halo, like SegWater's
  chips; seams between tiles are a map-production question (overlap and blending) for
  later milestones, not a numerical one;
- **Option C stays available and is proven**: when a model is trained to pool over larger
  areas, the whole area's pooled branch maps (50 × 256 values, 5.2 EECU·s to persist)
  plus tiles with a one-window C4 ring reproduce the whole-area PSP bit for bit (live, interior
  and corner tiles). Option B (pool over the requested AOI) is Option C's semantics with predictions
  that change with the AOI and a pooling cost that grows with it; it is not recommended.
- No new permanent task boundary: the PSP lives in task C (5.1 M embedded values), and
  its output is not persisted in production (the task-C export here is a demonstration).

**Proceed to Milestone 9 (Q33): yes.** On each criterion the spec names:

| criterion | evidence |
|---|---|
| adaptive-pooling correctness | every bin's exact member set at 7² … 56²; bit-identical to the mirror; ≤ 1.9e-6 from SMP (§2, §3) |
| PSP branch correctness | 12 branch intermediates + 4 resizes bit-identical to the mirror, ≤ 4.3e-6 (§5) |
| full PSP numerical error | canonical 1.57e-5 (target 1e-4), 1792 1.76e-5, edge 1.54e-5; residual = SMP's float32 arithmetic (§6, §12) |
| C4 geospatial alignment | projection, transform, dimensions, footprints and values on the C4 grid exact; task-C asset on the C4 grid (§6, §10) |
| task-C constant budget | 5.13 M (5.2 M with all feature norms), 22% of the cap, live at 28² and 56² (§7) |
| memory | ≤ 322 MiB interactive at 56² (the output conv's halo) (§9) |
| EECU cost | 17 / 65 EECU·s in batch at 28² / 56², ~85-90% in the output conv (§9) |
| graph complexity | 471 KB / 149 invocations (28²), 1.28 MB / 148 (56²) (§7) |
| window-layout viability | confirmed: pooling and the output conv are cheaper in window layout (§4) |
| large-crop scaling | 56² interactive and batch, pooling geometry rebuilt per shape and exact (§2, §6, §9) |
| PSP global / tile semantics | characterized and decided (Option A), with the exact alternative (Option C) proven (§11) |
| pooled-statistics persistence | 50 × 256 values, 32 KB, exact round trip, 0.1 EECU·s to read (§11) |

For Milestone 9: the PSP output is a window-layout [49, 256] map on the C4 grid, ready for
FPN step 1 (`fpn_lateral_blocks[0]` on C3 + Milestone 7's 2× window resize of the PSP
output, the step Milestone 7's representative chain already validated with a stand-in);
keep the mirror, the unique-ID fixtures and fresh inputs for benchmarks; the FPN 3 × 3
convs at C1 are Milestone 7's measured cost driver.

## 15. Caveats

- **Seeded weights.** The decoder's parameters are Milestone 7's deterministic seeded
  values (non-trivial BatchNorm statistics), the encoder's Milestones 5-6's. The milestone
  proves execution fidelity and measures semantics for these weights; a trained model's
  sensitivity to tile extent will differ in magnitude (not in structure).
- **One scene.** The real C4 maps are Milestone 6's Sun Moon Lake crops (canonical 896, the
  nested 448 / 1792, the swath edge 224).
- **Profiler semantics and variance.** "Peak" is EE's per-algorithm accounting, not RAM;
  identical requests vary up to ~1.5× in EECU·s; benchmark rows are single first runs.
- **Synthetic fixtures are dyadic.** The hash-valued synthetic C4 makes even PyTorch's
  float32 pooling exact; pooling numerics are therefore judged on real C4 and the
  order-revealing fixtures.
- **Assets.** C4 (`upernet/features/m8_c4_*`), PSP outputs and branch maps
  (`upernet/psp/`) and the gather probe (`upernet/bench/m8_probe_gather_batch_56_*`) remain in
  project `ORIGINAL_PROJECT`; delete the probes when done.
