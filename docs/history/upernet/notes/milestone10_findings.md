# Findings: UPerNet's multi-scale fusion stage in native Earth Engine

Live experiments on 7-8 October 2026: Earth Engine Python API 1.7.45, PyTorch 2.14.0,
timm 1.0.30, segmentation_models_pytorch 0.5.0, NumPy 2.5.3, Cloud project
`ORIGINAL_PROJECT`, application-default credentials. Every Earth Engine tensor was computed in
EE graphs from EE assets (task A''s skips assets, the C4 assets of Milestones 6-8,
Milestone 9's pyramid asset, Milestone 7's decoder weight assets) or from fixtures generated
on the server; tensors were downloaded only to compare them. No resize, concatenation or
convolution of the fusion stage is computed client-side.

Deliverables follow the brief's suggested layout, except that its three benchmark modules
(`benchmark_fusion`, `_formulations`, `_scale`) are one module,
benchmark_fusion.py (archive `upernet/benchmarks/benchmark_fusion.py` at `research-archive-2026-10-12`) (prefixes, formulations, fresh batch
repeats; the scale and shardSize runs are `validate_fusion --stages large`), as in Milestones 8-9.

Evidence: numerical_comparison_m10.csv (development archive: `docs/history/upernet/results/numerical_comparison_m10.csv`)
(summary with the Phase 27 matrix: numerical_summary_m10.md (development archive: `docs/history/upernet/results/numerical_summary_m10.md`)),
benchmark_results_m10.csv (development archive: `docs/history/upernet/results/benchmark_results_m10.csv`) (summary:
benchmark_summary_m10.md (development archive: `docs/history/upernet/results/benchmark_summary_m10.md`)), per-request records in
results/raw/ (archive `upernet/results/raw/` at `research-archive-2026-10-12`) (`validation_m10_*`, `bench/m10_*`, `m10_fusion_contract.json`,
`m10_stats_*.json`, `feature_assets.json`, `logs/m10_*`). Required Tests 1-22: offline in
test_upernet_fusion_contract.py (archive `tests/test_upernet_fusion_contract.py` at `research-archive-2026-10-12`),
test_upernet_fusion.py (archive `tests/test_upernet_fusion.py` at `research-archive-2026-10-12`),
test_upernet_fusion_budget.py (archive `tests/test_upernet_fusion_budget.py` at `research-archive-2026-10-12`),
test_upernet_fusion_optimized.py (archive `tests/test_upernet_fusion_optimized.py` at `research-archive-2026-10-12`) (36 passed), live in
test_upernet_fusion_live.py (archive `tests/test_upernet_fusion_live.py` at `research-archive-2026-10-12`) (`EE_LIVE=1`: 14 passed);
the full validation is validate_fusion.py (archive `upernet/earth_engine/validate_fusion.py` at `research-archive-2026-10-12`): **883 rows
(706 measured against an EE tensor), 0 failing; all 193 EE-vs-mirror comparisons
bit-identical (819.7 M elements); 10 negative controls that differ as they must.**

**Research questions.** (Q1) Can the installed SMP resize → concatenate → Conv-BN-ReLU
fusion be reproduced operation by operation in native EE? (Q2) Does the real-image fusion
output stay within 1e-4 of the installed SMP UPerNet? (Q3) Does the 1024 → 256, 3 × 3 C1-scale
bottleneck run reliably in task C at canonical size, and what dominates its memory and
EECU·s? (Q4) Can the convolution's linearity over its four 256-channel groups give an
equivalent cheaper formulation without silently changing rounding? (Q5) Does the complete
feature reads → PSP → FPN → fusion graph run as one task C with headroom for the head?
**Yes; yes — 2.38e-5 at worst on the canonical crop (2.69e-5 on the 1792 crop), every EE
tensor bit-identical to the controlled mirror; yes, in batch and even interactively — memory
is the zero-padded 9 × 9 halo assembly of the three resized levels, compute the [49, 9216]
im2col and the 9,216-term products in about equal parts; yes — keeping the four group products in double
and rounding once (formulation B) equals the monolithic convolution after BatchNorm and ReLU
in every case measured at a cost inside the monolithic one's spread, while four ordinary
rounded convolutions (A) are not equal — neither saves material memory (3-11%), because the
halos of the resized levels dominate however the channels are split; and yes — 9.86 M embedded values (41% of the
~24 M cap), 14.1 M left for the head, which needs 610.**

## The headline: the fusion is exact, and EE can sum the groups without an extra rounding

The installed code resizes every pyramid level to P1's size with
`F.interpolate(bilinear, align_corners=False)` — P1's own call is a **bitwise copy** in
PyTorch — concatenates `[P1, P2, P3, PSP]` and applies one `Conv2dReLU` 3 × 3, 1024 → 256,
bias-free; the result **is** the decoder's output (the head consumes it). EE reproduces it
with Milestone 7's primitives unchanged (window resize 2×, 4×, 8×; guarded `arrayCat`;
window im2col; explicit BatchNorm; ReLU). Every EE tensor of every input — synthetic, the
real edge and canonical crops, interactively and in batch, and the 1792 crop — is
**bit-identical to the controlled mirror**; the fused feature is within **2.38e-5** of SMP
float32 on the canonical crop. SMP's own float32 fusion is 2.4e-5 from float64; EE is
6.2e-7 from float64.

Because the convolution is bias-free, it is the sum of four 256 → 256 convolutions over the
P1, P2, P3 and PSP channel groups. In finite precision **how** they are summed matters, and
EE offers both answers (probed, § 4):

- **A — four ordinary convolutions** (each rounded to float32, three float32 additions):
  49.1% of the canonical conv outputs equal the monolithic ones (≤ 4.8e-7 apart; after ReLU
  79.6%, ≤ 7.2e-7); within 2.43e-5 of SMP.
- **B — the four group products kept in double, added in double, rounded once**: EE's
  `matrixMultiply` result stays double until `toFloat` (a 2^40 + 1 − 2^40 fixture spanning
  two groups reads 1 with B, 0 with A). Its double sum is a re-association of the monolithic
  9,216-term dot product (a 2^53 + 1 + 1 − 2^53 fixture reads 1 with B, 0 monolithic), yet
  after the single rounding it differs from the monolithic conv in **2 of 12,845,056**
  canonical outputs (1 ulp, 4.7e-10) and in **none** after BatchNorm or ReLU — on the
  canonical, edge and synthetic maps, and in the mirror for every group order tried.

**Milestone verdict.** **Minimum, strong and full success are all met.** The entire task-C
decoder body so far — C1-C4 assets → feature norms → PSP → FPN → multi-scale resize →
concat → fusion Conv-BN-ReLU — runs inside Earth Engine as one batch graph at canonical size
and produces the georeferenced 224 × 224 × 256 fused feature on the C1 grid within 2.38e-5
of the installed SMP reference, bit-identical to the mirror and to the same tensor computed
interactively. **Semantic reference: the monolithic 1024 → 256 im2col convolution.
Production: formulation B** (double group partials, one rounding), chosen on its numerics:
the reference's output after BatchNorm and ReLU in every measured case, at a cost no worse
than the reference's (complete canonical task C 669 / 693 vs 567 / 878 / 1,121 EECU·s, a
difference the ~2× batch spread cannot resolve) and with the same memory (§ 4-6). It is not a
memory optimization; the monolithic path remains the semantic reference and check.

## Validation matrix (Phase 27)

Generated from the CSV by `python -m upernet.benchmarks.summarize --milestone 10` ("same
input": SMP float32 on the C1-C4 values EE reads; "fusion input": SMP's fusion stage alone
on EE's pyramid values, the fusion's own float32 error; last column: elements equal to EE's
monolithic tensor).

| operation | experiment | MAE | RMSE | max abs | same input | fusion input | max vs float64 | EE vs mirror | vs EE monolithic |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| fusion order, level IDs (all channels, all positions) | m10_ids | — | — | — | — | — | — | exact | — |
| concat order, positional IDs 56² × 1024 | m10_ids | — | — | — | — | — | — | exact | — |
| concat order, 224² × 1024 (server count) | m10_ids | — | — | — | — | — | — | exact | — |
| fusion im2col ordering (1,024 channels, every tap) | m10_ids | — | — | — | — | — | — | exact | — |
| fusion weight-row ordering (monolithic) | m10_ids | — | — | — | — | — | — | exact | — |
| group weight rows (A / B), groups 1-4 | m10_ids | — | — | — | — | — | — | exact | — |
| P2 / P3 / PSP → C1 ramps (source rule, edges) | m10_ids | — | — | — | — | — | — | exact | — |
| P1 identity (P1_to_C1) | m10_fusion_canonical_896 | 7.74e-07 | 1.47e-06 | 2.53e-05 | 2.38e-05 | 0 | 1.15e-06 | bit-identical | — |
| P2 → C1 resize (2×) | m10_fusion_canonical_896 | 4.8e-07 | 9.25e-07 | 1.29e-05 | 1.34e-05 | 9.54e-07 | 8.23e-07 | bit-identical | — |
| P3 → C1 resize (4×) | m10_fusion_canonical_896 | 4.64e-07 | 8.72e-07 | 1.62e-05 | 1.38e-05 | 9.54e-07 | 8.3e-07 | bit-identical | — |
| PSP → C1 resize (8×) | m10_fusion_canonical_896 | 4.77e-07 | 9.58e-07 | 1.41e-05 | 1.65e-05 | 4.77e-07 | 5.01e-07 | bit-identical | — |
| fusion concat (1,024 channels) | m10_fusion_canonical_896 | 5.49e-07 | 1.08e-06 | 2.53e-05 | 2.38e-05 | 9.54e-07 | 1.15e-06 | bit-identical | — |
| baseline fusion Conv | m10_task_c_canonical_896 | 1.51e-06 | 2.02e-06 | 1.93e-05 | 1.97e-05 | 2.19e-05 | 4.97e-07 | bit-identical | — |
| baseline fusion BN | m10_task_c_canonical_896 | 1.52e-06 | 2.1e-06 | 2.38e-05 | 2.57e-05 | 2.24e-05 | 7.38e-07 | bit-identical | — |
| **baseline fusion ReLU** (output) | m10_task_c_canonical_896 | 7.03e-07 | 1.42e-06 | 2.38e-05 | 2.57e-05 | 2.24e-05 | 6.21e-07 | bit-identical | — |
| grouped (A) Y1 / Y2 / Y3 / Y4 | m10_formulations_canonical_896 | 6.9 / 4.5 / 4.3 / 4.1e-07 | — | 6.68 / 5.01 / 4.41 / 3.16e-06 | — | — | 3.1 / 2.5 / 2.1 / 1.4e-07 | bit-identical | — |
| grouped (A) sum | m10_formulations_canonical_896 | 1.51e-06 | 2.02e-06 | 1.96e-05 | 1.97e-05 | 2.19e-05 | 7.03e-07 | bit-identical | 49.0946% equal |
| grouped (A) BN | m10_formulations_canonical_896 | 1.52e-06 | 2.1e-06 | 2.43e-05 | 2.57e-05 | 2.24e-05 | 1.2e-06 | bit-identical | 56.3676% equal |
| grouped (A) ReLU | m10_formulations_canonical_896 | 7.03e-07 | 1.42e-06 | 2.43e-05 | 2.57e-05 | 2.24e-05 | 8.12e-07 | bit-identical | 79.5602% equal |
| double-partial (B) sum | m10_formulations_canonical_896 | 1.51e-06 | 2.02e-06 | 1.93e-05 | 1.97e-05 | 2.19e-05 | 4.97e-07 | bit-identical | all but 2 of 12,845,056 equal |
| double-partial (B) BN | m10_formulations_canonical_896 | 1.52e-06 | 2.1e-06 | 2.38e-05 | 2.57e-05 | 2.24e-05 | 7.38e-07 | bit-identical | bit-identical |
| double-partial (B) ReLU | m10_formulations_canonical_896 | 7.03e-07 | 1.42e-06 | 2.38e-05 | 2.57e-05 | 2.24e-05 | 6.21e-07 | bit-identical | bit-identical |
| **complete fusion synthetic (C1 112²)** | m10_fusion_synthetic_14 | 6.84e-07 | 1.36e-06 | 2.24e-05 | 2.24e-05 | 2.05e-05 | 4.41e-07 | bit-identical | — |
| **complete fusion real edge (C1 56²)** | m10_fusion_edge_224 | 6.31e-07 | 1.27e-06 | 1.72e-05 | 1.62e-05 | 1.5e-05 | 5.27e-07 | bit-identical | — |
| **complete fusion real canonical (C1 224², interactive)** | m10_fusion_canonical_896 | 7.03e-07 | 1.42e-06 | 2.38e-05 | 2.57e-05 | 2.24e-05 | 6.21e-07 | bit-identical | — |
| **complete fusion real canonical (C1 224², batch task C)** | m10_task_c_canonical_896 | 7.03e-07 | 1.42e-06 | 2.38e-05 | 2.57e-05 | 2.24e-05 | 6.21e-07 | bit-identical | — |
| **complete fusion real large (C1 448², batch)** | m10_fusion_large_1792 | 7.09e-07 | 1.43e-06 | 2.69e-05 | 2.38e-05 | — | 6.31e-07 | bit-identical | — |

Pure layout, index and grid rows are exact; every floating-point row is within 2.7e-5 of SMP
float32 (target 1e-4) and bit-identical to the mirror.

## Answers

| # | Question | Answer |
|---|---|---|
| 1 | Exact installed SMP operations of the fusion stage? | `UPerNetDecoder.forward` lines 199-211: `target_size = fpn_features[-1].shape[2:]` (P1's), `F.interpolate(f, size=target_size, mode="bilinear", align_corners=False)` for f in [PSP, P3, P2, P1], `torch.cat(resized[::-1], dim=1)`, `fusion_block` = `Conv2dReLU` (lines 154-161): Conv2d 1024 → 256, 3 × 3, padding 1 (zeros), **bias=False** → BatchNorm2d(256, eps 1e-5) → ReLU(inplace). Its output is the decoder's output (§ 1). |
| 2 | Exact input tensor order? | **[P1, P2, P3, PSP]**: the reverse of the resize loop's order (PSP, P3, P2, P1) (§ 1). |
| 3 | Are the channel ranges [P1, P2, P3, PSP] exact? | **Yes**: 0-255, 256-511, 512-767, 768-1023, by unique IDs at every channel and position, in PyTorch (3,136 positions) and through EE's own resizes and concatenation; positional IDs 56² × 1024 fetched and 224² × 1024 counted on the server: 0 mismatches of 51,380,224 (§ 1-2). |
| 4 | Is P1's same-size resize an exact identity? | **Yes**: PyTorch returns a bitwise copy (new storage, signed zeros kept) for random and canonical P1, float32 and float64; EE passes P1 through, so P1_to_C1 == P1 == mirror, 0 from SMP's own fusion on EE's P1 (§ 1-2). |
| 5 | Can P2 resize exactly to C1? | **Yes**: window resize 2×, bit-identical to the mirror; ramps equal PyTorch, mirror and the written rule (edges included); 1.29e-5 from SMP (inherited from P2), ≤ 9.5e-7 = 1 ulp from SMP's resize of EE's P2 (§ 2). |
| 6 | Can P3 resize exactly to C1? | **Yes**: 4×, bit-identical, ramps exact; 1.62e-5 from SMP, ≤ 9.5e-7 on the same input (§ 2). |
| 7 | Can PSP resize exactly to C1? | **Yes**: 8× — Milestone 7's primitive in the real PSP / FPN context — bit-identical, ramps exact; 1.41e-5 from SMP, ≤ 4.8e-7 on the same input (§ 2). |
| 8 | Do all four occupy exactly the C1 grid before concatenation? | **Yes**: each resize returns the token grid it lands on and the guard requires equality; georeferenced projections EPSG:32651, 40 m, the crop's origin, 224 × 224 for all four and the concatenation; un-resized P2 / P3 / PSP, a 0.7-pixel offset, wrong dimensions and a non-integer factor are refused (§ 2). |
| 9 | Is the 1,024-channel concatenation exact? | **Yes**: IDs above; the real canonical concatenation (51.4 M values) bit-identical to the mirror (§ 2). |
| 10 | Is fusion im2col ordering exact? | **Yes**: a 1,024-channel ID map at C1 56²: 9 taps × channels {0, 1, 255, 256, 511, 512, 767, 768, 1023}, (ky, kx, c) order, zeros at every edge and corner (§ 3). |
| 11 | Is fusion weight-row ordering exact? | **Yes**: the server-permuted W^T [9216, 256] rows equal `weight.permute(2, 3, 1, 0)`; each group's W_g^T is the contiguous asset column slice [2304 g, 2304 (g + 1)) and equals both its own permutation and the monolithic rows of its channels (§ 3). |
| 12 | Can the baseline 1024 → 256 conv reproduce SMP? | **Yes**: bit-identical to the mirror on every input; 9,216-term dots accumulate sequentially in double (cancellation fixtures) (§ 3). |
| 13 | Fusion conv max abs error? | **1.93e-5** canonical (same input 1.97e-5; SMP's fusion alone on EE's pyramid 2.19e-5); 2.03e-5 synthetic, 1.72e-5 edge, 2.19e-5 1792 (§ 3). |
| 14 | BN and ReLU errors? | **2.38e-5** and **2.38e-5** canonical (1792: ReLU 2.69e-5) (§ 3). |
| 15 | Does the complete baseline fusion output reproduce SMP within 1e-4? | **Yes**: 2.38e-5 canonical (interactive and batch), 2.69e-5 at 1792 (§ 3, § 8-9). |
| 16 | Is every EE tensor bit-identical to the controlled mirror? | **Yes**: 193 / 193 comparisons, 819.7 M elements (§ 3). |
| 17 | Does the canonical baseline fusion run interactively? | **Yes, in both forms tried**: the in-graph task C fetched in 7-row blocks (32 requests, every element of conv, BN, ReLU), and a checksum of the whole map's fusion (pyramid asset → conv + BN + ReLU) in one request (781 EECU·s, 696 s, 1,690 MiB). Milestone 7's whole-map interactive memory failure was not reproduced; batch stays the production mode (§ 5). |
| 18 | Does it run reliably in batch? | **Yes**: every task-C export succeeded (canonical: 2 validation and 4 fresh complete task C runs, 10 fusion-only; 1792: 2) (§ 6). |
| 19 | Its EECU cost? | Complete canonical task C (assets → fused feature): **567 / 878 / 1,121 EECU·s** (three fresh graphs, ~2× run-to-run spread); conv + BN + ReLU export 952; the fusion alone (from the pyramid asset) 667-756; interactively, the whole-map conv 862 (§ 5-6). |
| 20 | Its peak memory? | **1,690 MiB** (EE's per-algorithm peak) for the whole canonical map, reached at the halo prefix and unchanged through im2col, conv, BN and ReLU; 317 / 528 / 951 MiB over 4 / 8 / 16 of the 32 window rows (§ 5). |
| 21 | What dominates fusion memory? | **The 9 × 9 halo assembly of the resized levels** ("Image.unmask computing pixels": every window's 8 neighbours must be resized from their parents' halos too): the PSP group's halo alone peaks at 563 MiB over the whole map, P1's (not resized) below 46 MiB; linear in the windows a request evaluates (~53 MiB per window row). Weights and products ~140 MiB, the resized inputs + concatenation 174 MiB; im2col and the product add no peak (§ 5). |
| 22 | What dominates fusion compute? | **The [49, 9216] im2col assembly and the 9,216-term product, about equally**: whole-map prefixes concat 42 → halo 70 → im2col 441 → conv 862 EECU·s (8 rows: 25 → 20 → 59 → 159); resizes and concatenation ~5%; BN / ReLU small (their whole-map prefixes, 554 / 752, are within the interactive spread) (§ 5). |
| 23 | Does the four-group decomposition work? | **Yes**: Σ_g Conv(X_g, W_g) equals the monolithic conv to 3.2e-14 in double; EE's Y1-Y4 are bit-identical to the mirror and within 6.7e-6 of PyTorch's per-group convs (§ 4). |
| 24 | How far does ordinary grouped execution (A) differ from the monolithic EE baseline? | 49.1% of conv outputs equal, ≤ 4.8e-7 (BN 56.4%, ≤ 9.5e-7; ReLU 79.6%, ≤ 7.2e-7) (§ 4). |
| 25 | Does grouped execution stay within 1e-4 of SMP? | **Yes**: A 2.43e-5, B 2.38e-5 (§ 4). |
| 26 | How much memory does grouped execution save? | **Little**: 3-11% (282 / 493 / 915 / 1,636 vs 317 / 528 / 951 / 1,690 MiB over 4 / 8 / 16 / 32 window rows): the grouped formulations still assemble the halos of the three resized levels (~550 MiB each at the whole map), which is where the memory is (§ 5). |
| 27 | How much EECU does grouped execution save or add? | Complete task C: **B 669 / 693 vs monolithic 567 / 878 / 1,121 EECU·s** (B's two runs lie inside the monolithic spread, below its mean of 855, with consistently short runs, 320 / 354 s; not resolvable from these runs); fusion alone, batch: monolithic 667-756, A 339-669, B 569-723; interactive whole map: monolithic 781, A 417, B 729 (§ 5-6). |
| 28 | Can grouped partials be accumulated before the final float32 rounding? | **Yes**: `matrixMultiply` results are double until `toFloat`, double images add in double; the 2^40 + 1 − 2^40 probe reads 1 (B) vs 0 (A) (§ 4). |
| 29 | If so, bit-identical to the monolithic EE implementation? | **Not in double** (re-association: the 2^53 probe reads 1 vs 0; 97% of double sums differ by ≤ 3.2e-14), **but after the one rounding**: 2 of 12,845,056 conv outputs differ (1 ulp), 0 after BN and ReLU, on every map measured (§ 4). |
| 30 | Semantic reference? | **The monolithic im2col convolution** (one 9,216-term dot per output in the installed channel order) (§ 4). |
| 31 | Production implementation? | **Formulation B** (four double group products, summed in [P1, P2, P3, PSP] order, rounded once): chosen on numerics: equal to the reference after BN / ReLU where measured, cost no worse, memory the same (it skips the 1,024-channel concatenation, which buys nothing measurable: the resized levels' halos dominate); keep the monolithic path as the checked reference (§ 4-6). |
| 32 | Full PSP + FPN + fusion task-C EECU cost? | **567-1,121 EECU·s** monolithic, **669-693** with B (canonical, batch, fresh graphs) (§ 6). |
| 33 | Task-C embedded values after fusion? | **9,856,114** (PSP + FPN 7,443,930 + fusion 2,412,184: conv 2,359,296, BN 50,176, resize tables 2,712); 9,976,818 at C4 56² (§ 7). |
| 34 | Constant headroom for the segmentation head? | **14.1 M** below 24 M (41% used); the head needs 610 (W^T 512 + bias rows 98) (§ 7). |
| 35 | Does the complete task-C graph fit comfortably in batch? | **Yes**: 816-894 KB, 542-552 invocations, 41% of the constant cap (§ 6-7). |
| 36 | Does the fusion output occupy exactly the C1 grid? | **Yes**: EPSG:32651, [40, 0, 283200.953…, 0, −40, 2645318.840…], 224 × 224, every band of both assets; 40 / 40 footprints; window layout == georeferenced raster (§ 8). |
| 37 | Can the 1792 crop run through fusion? | **Yes, in batch**: 3,082-4,589 EECU·s, ReLU within 2.69e-5 of SMP, bit-identical to the mirror (§ 9). |
| 38 | Does shardSize materially affect large fusion exports? | **Yes**: `shardSize=448` (one shard) vs the default 256: **3,082 vs 4,589 EECU·s (−33%), 500 vs 1,344 s run (−63%)**, values identical (§ 9). |
| 39 | Is the fusion block the dominant cost of the entire model? | **Yes, the largest single block**: ~300-830 EECU·s of a 567-1,121 task C (PSP + FPN 259-288), against 370 for the whole encoder (A' 308 + B' 62); task C is ~70% of the canonical model (§ 10). |
| 40 | Proceed to M11? | **Yes** (§ 12). |

## 1. The installed fusion and its contract (Phases 1-3; Tests 1-3)

Read off and run from the installed packages
(reference/smp_fusion_reference.py (archive `upernet/reference/smp_fusion_reference.py` at `research-archive-2026-10-12`) `contract`,
recorded in `results/raw/m10_fusion_contract.json`; tests in
test_upernet_fusion_contract.py (archive `tests/test_upernet_fusion_contract.py` at `research-archive-2026-10-12`)):

| item | installed (smp 0.5.0, torch 2.14.0) |
|---|---|
| source | `decoders/upernet/decoder.py`, `UPerNetDecoder.forward` lines 199-211; `fusion_block` built at lines 154-161 (`num_blocks_to_fuse = len(fpn_conv_blocks) + 1`) |
| fusion input list | `fpn_features = [PSP, P3, P2, P1]` (Milestone 9's leaves, post-ReLU) |
| resize calls | four `F.interpolate(feature, size=target_size, mode="bilinear", align_corners=False)` (antialias False, no scale_factor), in the order PSP, P3, P2, P1; recorded at a 224 input: 7, 14, 28, 56 → 56 |
| target size | `fpn_features[-1].shape[2:]` = P1's = C1 = H/4 (224 × 224 for the canonical crop) |
| concatenation | `torch.cat(resized_fpn_features[::-1], dim=1)` = **[P1, P2, P3, PSP]** |
| fusion conv | Conv2d 1024 → 256, 3 × 3, stride 1, padding 1 (zeros), dilation 1, groups 1, **bias=False** |
| BatchNorm | BatchNorm2d(256, eps 1e-5), inference statistics (explicit in EE) |
| activation | ReLU(inplace) — its output is `UPerNetDecoder.forward`'s return value and the segmentation head's input (hooked) |

**Expanded forward (Phase 1).** `expanded` runs Milestone 9's expanded PSP / FPN, then the
fusion step by step (`fusion_stage`), and raises unless the expanded Conv-BN-ReLU equals
`fusion_block`, every PSP / FPN tensor equals Milestone 9's hooks, the concatenation equals
the fusion block's input and the ReLU output equals the decoder's own output — one hooked
run of the whole forward. It holds bit for bit in float32 and float64 at C4 7² and 14² and
in a real UPerNet forward from a random 224² input. **P1's resize** is a bitwise copy: on
random tensors (float32, float64, channels-last), on the canonical P1 and on a fixture with
three −0.0 entries (kept as −0.0), with new storage. References are slices of the expanded
forward on the real features; SMP's fusion alone on cached pyramids equals it bit for bit
(checked on the canonical crop), which the 1792 references use.

**Unique IDs (Test 2).** A copy of the decoder whose conv blocks emit P1 = 1000 + c, P2 =
2000 + c, P3 = 3000 + c and whose PSP emits 4000 + c, run through its own forward: channel j
of the fusion block's input is 1000 (j // 256 + 1) + j % 256 at all 3,136 positions.

**Statistics (Phase 1)**, SMP float32 on the canonical crop (`results/raw/m10_stats_canonical_896.json`;
synthetic and edge beside it):

| tensor | shape | min | max | mean | std |
|---|---|---:|---:|---:|---:|
| P1 / P2 / P3 / PSP | 224² / 112² / 56² / 28² × 256 | 0 | 8.17 / 7.27 / 6.31 / 3.67 | 0.875 / 0.622 / 0.596 / 0.360 | 1.22 / 1.01 / 0.91 / 0.52 |
| P1_to_C1 … PSP_to_C1 | 224² × 256 each | 0 | 8.17 / 6.89 / 6.25 / 3.63 | 0.875 / 0.622 / 0.596 / 0.360 | 1.22 / 1.00 / 0.91 / 0.52 |
| fusion_concat | 224² × 1024 | 0 | 8.17 | 0.613 | 0.965 |
| fusion_conv | 224² × 256 | −4.78 | 4.39 | −0.076 | 1.21 |
| fusion_bn | 224² × 256 | −5.61 | 4.37 | −0.086 | 1.30 |
| fusion_relu (output) | 224² × 256 | 0 | 4.37 | 0.458 | 0.73 |

All float32. Bilinear resizes preserve the means and smooth the extremes.

## 2. Resize, grid guard and concatenation in EE (Phases 3-8; Tests 3-8)

earth_engine/fusion.py (archive `upernet/earth_engine/fusion.py` at `research-archive-2026-10-12`) composes Milestone 7's primitives in
window layout ([49, C] per window of the C1 window grid):

| step | EE | rounding |
|---|---|---|
| P1_to_C1 | P1 itself (PyTorch's call is a bitwise copy) | none |
| P2 / P3 / PSP → C1 | `resize.window_resize` with the factor the grids imply (2, 4, 8): the parent window's 9 × 9 halo at every child window, two products with PyTorch's own dyadic weights ([7, 9] per axis) | exact in double, one rounding |
| guard | each resize returns the token grid it lands on; `feature_ops.concat` refuses unless all four are on P1's grid (dimensions, scale, CRS) | — |
| concatenation | `arrayCat` on axis 1 in [P1, P2, P3, PSP] order → [49, 1024] | values moved |

**Layout, exact (`m10_ids`).** Level IDs on their own grids (C4 7²) through EE's resizes and
concatenation: all 1,024 channels at all 3,136 C1 positions equal the PyTorch fixture.
Positional IDs (token × 1024 + j + 1) through the concatenation: every element at 56²
fetched, and at the canonical 224² × 1024 the difference from the expected [49, 1024] counted
on the server over all 1,024 windows: **0 of 51,380,224** elements differ. Ramps (row,
column, 1) on P2's, P3's and PSP's grids resized 2×, 4×, 8× to C1 equal PyTorch float32, the
mirror and the written rule src = max((o + ½)/f − ½, 0), i0, i1 = min(i0 + 1, n − 1) bit for
bit, edges and corners included; offline, every output row of the window-local weight
matrices equals that rule for factors 2, 4, 8 (dyadic, multiples of 1/2f).

**Real features.** On the canonical crop the four resized levels and the concatenation are
bit-identical to the mirror; their distance to SMP (1.29-1.62e-5) is the pyramid's own
(Milestone 9's P levels: ≤ 2.53e-5), and against SMP's resize of EE's own pyramid values
it is ≤ 9.5e-7 — one float32 ulp at those magnitudes: PyTorch's float32 bilinear kernel
rounds its intermediate products, EE (and the mirror) compute the exact bilinear value in
double and round once.

**Grid guard negative controls (Test 7; `m10_negative_canonical_896`).** Refused: the
concatenation with P2, P3 or the PSP output not resized (`window on TokenGrid(112, 112, 20)
vs window on TokenGrid(224, 224, 10)` …), P2_to_C1 relabelled 28 m (0.7 pixel) east, P2_to_C1
on a 217-column grid, and a level whose grid is not an integer fraction of C1's. Without the
guard EE computes something: (a) the un-resized P2 concatenated on the C1 window grid is
resampled (nearest) from the C2 window image — every C1 window receives the 49 P2 tokens of
the C2 window containing it: up to 3.53 from SMP's P2_to_C1, 51% of values wrong; (b) the
offset P2_to_C1 resampled onto the C1 grid: a one-column shift and a masked last column
(224 pixels). Neither is an error in EE; both are why the concatenation goes through the guard.

## 3. The baseline fusion convolution (Phases 9-11; Tests 9-13)

`fusion.monolithic`: Milestone 7's window im2col — the [9, 9, 1024] halo from the 8
neighbouring windows (zeros outside the raster) → [49, 9216] in (ky, kx, c) order → one
`matrixMultiply` with W^T [9216, 256] (permuted on the server from the asset's [256, 9216] in
PyTorch's (c, ky, kx) order) → `toFloat`; explicit BatchNorm (double inside, one rounding);
ReLU.

**Ordering, exact.** The im2col of a 1,024-channel ID map at C1 56² (8 × 8 windows): 9 taps
× channels 0, 1, 255, 256, 511, 512, 767, 768, 1023 equal PyTorch's zero-padded im2col at
every position (254,016 values; top, bottom, left, right, corners, interior). The server's
W^T rows at the same (tap, channel) picks equal `weight.permute(2, 3, 1, 0)` (20,736
values); the four group slices likewise (13,824 values each).

**Accumulation, probed (Phase 11).** On one pixel, a [1, 9216] × [9216, 2] product whose
column 0 holds 2^53, +1 (k 4607), +1 (k 4608), −2^53 (k 9215) and column 1 holds 2^40, +1,
−2^40 returns [0, 1] in value space and on an image: strictly sequential accumulation in
double (two halves or a pairwise tree would give 1 or 2 in column 0; float32 accumulation 0
in column 1). torch's float64 matmul is strictly sequential at the mirror's block shape
[3136 × 9216] × [9216 × 256] (checked), so the mirror's blocked conv reproduces EE's.

**Every input (Phase 10; Tests 11-12).**

| input | C1 grid | conv / BN / ReLU vs SMP float32 | ReLU same input | ReLU fusion input | ReLU vs float64 | EE vs mirror |
|---|---|---|---:|---:|---:|---|
| synthetic C1-C4 (C4 14²) | 112², abstract | 2.03 / 2.24 / 2.24e-5 | (= own) | 2.05e-5 | 4.4e-7 | bit-identical (every tensor, 3 formulations) |
| real edge_224 | 56², 40 m | 1.72 / 1.72 / 1.72e-5 | 1.62e-5 | 1.50e-5 | 5.3e-7 | bit-identical (every tensor, 3 formulations) |
| **real canonical 896**, interactive and batch | 224², 40 m | **1.93 / 2.38 / 2.38e-5** | 2.57e-5 | 2.24e-5 | 6.2e-7 | bit-identical |
| real 1792, batch | 448², 40 m | 2.19 / — / 2.69e-5 | 2.38e-5 | — | 6.3e-7 | bit-identical |

**Error composition (Phase 11).** EE − float64 is 5-6e-7 everywhere (two orders below the
target); SMP float32 − float64 is 2.4e-5 at the canonical output; SMP's fusion alone on EE's
pyramid differs from EE by 2.24e-5. The residual is SMP's float32 9,216-term convolution, not
EE: the pyramid's own error (P1 2.53e-5) does not grow through the fusion (output 2.38e-5).

## 4. The grouped formulations (Phases 14-18; Tests 14-16)

**Decomposition (Phase 14).** With bias=False, Conv([X1, X2, X3, X4], W) = Σ_g Conv(X_g, W_g),
X_g the resized levels in [P1, P2, P3, PSP] order and W_g = W[:, 256 g : 256 (g + 1)] — in
the asset's flattened (c, ky, kx) order the contiguous columns [2304 g, 2304 (g + 1)), sliced
on the server. In double the four group convs sum to the monolithic conv within 3.2e-14 (all
12.8 M canonical outputs).

**A — four ordinary convolutions (Phase 15).** `fusion.grouped_rounded`: each Y_g is
Milestone 7's `conv3x3_window` (a [9, 9, 256] halo, [49, 2304] im2col, one product, rounded),
then ((Y1 + Y2) + Y3) + Y4 with `feature_ops.add` (float32 + float32 → float32), BN, ReLU. Y1-Y4
are bit-identical to the mirror and within 6.7 / 5.0 / 4.4 / 3.2e-6 of PyTorch's per-group
convolutions. The sum differs from the monolithic EE conv in 50.9% of outputs (≤ 4.8e-7: up
to four partial roundings and three addition roundings, ≤ 2^-20 of the partials'
magnitude); after BN 56.4% equal (≤ 9.5e-7), after ReLU 79.6% (≤ 7.2e-7). Within 2.43e-5 of
SMP — acceptable, but not the baseline's numbers.

**B — double partials, one rounding (Phase 16).** `fusion.grouped_double`: each group's
[49, 2304] im2col times W_g^T with `matmul_double` and **no** `toFloat`; the four double
images added in [P1, P2, P3, PSP] order; one `toFloat`; BN; ReLU. The probe fixture (one
window, four groups of two channels, weights only at the centre tap) separates the three:

| centre output | channel 0: 2^40 + 1 (group 1), −2^40 (group 2) | channel 1: 2^53 + 1 (group 3), + 1 − 2^53 (group 4) |
|---|---:|---:|
| monolithic (one sequential double dot) | 1 | 0 |
| A (each group rounded to float32) | 0 | 0 |
| B (group dots in double, double sum) | 1 | 1 |

EE and the mirror agree on all six values: EE keeps `matrixMultiply` results in double until
`toFloat`, and adds double images in double. B is therefore **not** the monolithic double
sum — channel 1 shows the re-association (each group restarts its dot from zero) — and on the
canonical crop 97% of B's double sums differ from the monolithic ones, by ≤ 3.2e-14. The
final float32 rounding hides that difference except where the exact value lies within ~1e-14
of a rounding midpoint: **2 of 12,845,056 conv outputs** (1 ulp, 4.7e-10), **0 after BatchNorm,
0 after ReLU**; edge and synthetic maps: 0 / 1 conv outputs, 0 after BN / ReLU.

**Order study (Phase 17; `m10_order_study`, mirror, canonical, numerical characterization
only):**

| grouped sum | equal to the monolithic conv | max vs monolithic | max vs float64 |
|---|---:|---:|---:|
| A sequential P1+P2+P3+PSP (the installed order) | 49.09% | 4.8e-7 | 7.0e-7 |
| A sequential P1+P3+P2+PSP / PSP+P3+P2+P1 / P2+P1+PSP+P3 | 49.29 / 52.80 / 50.47% | 4.8e-7 | 6.7-6.8e-7 |
| A pairwise (P1+P2)+(P3+PSP) | 51.83% | 4.8e-7 | 6.8e-7 |
| A partials rounded, summed in double | 61.01% | 4.8e-7 | 6.2e-7 |
| B, every order and both trees | 99.99998% (2 elements) | 4.7e-10 | 5.0e-7 |

A's result depends on the order; B's does not (beyond the same two outputs). Production
keeps the installed channel order [P1, P2, P3, PSP] regardless.

**Comparison (Phase 18):**

| formulation | max vs SMP (ReLU) | vs baseline EE (conv / ReLU) | task C EECU·s (canonical, fresh) | fusion alone EECU·s | interactive peak, 4 / 8 / 16 / 32 window rows | interactive / batch |
|---|---:|---|---:|---:|---|---|
| **monolithic im2col** (reference) | 2.38e-5 | — | 567 / 878 / 1,121 | 739 / 667 / 756 | 317 / 528 / 951 / 1,690 MiB | both |
| A: 4 rounded group convs | 2.43e-5 | 49.1% / 79.6% equal, ≤ 7.2e-7 | — | 517 / 669 / 339 | 282 / 493 / 915 / 1,636 MiB | both |
| **B: 4 double group products, one rounding** (production) | 2.38e-5 | all but 2 / **bit-identical** | **693 / 669** | 569 / 642 / 723 | 282 / 475 / 915 / 1,637 MiB | both |

## 5. Memory and compute (Phases 12-13)

benchmark_fusion.py (archive `upernet/benchmarks/benchmark_fusion.py` at `research-archive-2026-10-12`), one profiled interactive request
per row: the fusion stage alone, fed from Milestone 9's pyramid asset (bit-identical to the
in-graph pyramid), every request salted. "Peak" is EE's largest per-algorithm "Peak Mem"
(the profile reports one row per algorithm), not RAM at an instant.

**Cumulative prefixes over the whole canonical map** (32 × 32 C1 windows):

| prefix | whole map: EECU·s | wall s | peak MiB | 8 of 32 window rows: EECU·s | wall s | peak MiB |
|---|---:|---:|---:|---:|---:|---:|
| read P1, P2, P3, PSP (pyramid asset → window layout) | 15 | 5 | 46 | 8 | 2 | 46 |
| + P2 → C1 (2×) | 24 | 9 | 46 | 11 | 3 | 46 |
| + P3 → C1 (4×) | 38 | 13 | 46 | 13 | 3 | 46 |
| + PSP → C1 (8×) | 45 | 10 | 46 | 19 | 4 | 46 |
| + concatenation [49, 1024] | 42 | 26 | 174 | 25 | 12 | 174 |
| + 9 × 9 halo (zero-padded, 8 neighbours) | 70 | 36 | 1,689 | 20 | 6 | 528 |
| + im2col [49, 9216] | 441 | 232 | 1,690 | 59 | 32 | 528 |
| + product with W^T [9216, 256] (conv) | 862 | 442 | 1,689 | 159 | 55 | 528 |
| + BatchNorm | 554 | 177 | 1,690 | 202 | 102 | 528 |
| + ReLU (fused output) | 752 | 624 | 1,690 | 273 | 164 | 528 |

**Memory vs extent** (each formulation's conv + BN + ReLU, checksum over the first k of the
32 window rows):

| formulation | 4 rows: EECU·s | peak MiB | 8 rows | peak | 16 rows | peak | 32 rows (whole map) | peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| monolithic conv + BN + ReLU | 86 | 317 | 217 | 528 | 288 | 951 | 781 | 1,690 |
| A: 4 rounded group convs + BN + ReLU | 69 | 282 | 177 | 493 | 327 | 915 | 417 | 1,636 |
| B: 4 double group products + BN + ReLU | 118 | 282 | 152 | 475 | 286 | 915 | 729 | 1,637 |
| one group's [9, 9, 256] halo | 3 | 46 | 5 | 46 | 8 | 46 | 10 | 46 |
| one group's [49, 2304] im2col | 11 | 46 | 26 | 46 | 70 | 46 | 182 | 46 |
| one group's conv (rounded) | 34 | 140 | 33 | 140 | 81 | 140 | 198 | 140 |
| PSP group (8× resized) alone: [9, 9, 256] halo | — | — | — | — | — | — | 13 | 563 |
| PSP group alone: conv (rounded) | — | — | — | — | — | — | 183 | 563 |

What dominates (Phase 13):

- **The halo assembly of the resized levels, not weights or channel width.** Every fusion
  request's peak is "Image.unmask computing pixels" — the zero-padding step that
  materializes each window's [49, C] array before the 9 × 9 gather from the 8 neighbours.
  For P1 (passed through, no resize) the halo stays below the 46 MiB asset-read floor even
  over the whole map; for the PSP group (8× resized) the halo alone peaks at **563 MiB**: each
  C1 window's neighbours must themselves be resized from their parents' 9 × 9 halos. The
  four-group concatenation's halo reaches 1,690 MiB; the grouped formulations' four halos
  1,636 MiB (~3 × 550 for P2, P3, PSP). Linear in the windows a request evaluates
  (≈ 53 MiB per window row of 32 windows).
- **The im2col and the product add compute, not peak**: the [49, 9216] im2col, the conv, BN
  and ReLU prefixes all stay at the halo's 1,690 MiB.
- **Weights**: reading the fusion block's 2.36 M parameters alone: 2.7 EECU·s, 139 MiB
  ("Algorithm Array"); one group's conv request peaks at 140 MiB (decoded W_g and products).
- **Four resized inputs and the concatenation**: 174 MiB, ~42 EECU·s.

Compute (Phase 13): for the whole map concat 42 → halo 70 → **im2col 441** → **conv 862
EECU·s** (8 rows: 25 → 20 → 59 → 159): the im2col assembly (9 shifted [49, 1024] slices
concatenated per window) costs about as much as the 9,216-term products; resizes and
concatenation are ~5%; the BN and ReLU prefixes (554, 752 for the whole map) are inside the
interactive spread (~1.7×), so their own cost is small. A cheaper formulation would have to
avoid the resized levels' neighbour halos — e.g. a window resize that emits each child
window's [9, 9] halo directly from the parent's halo — not split the channels; not tried here.

**Interactive feasibility (Phase 12).** Milestone 7 recorded the standalone canonical fusion
conv as an interactive memory failure; here the whole map's fusion ran interactively as one
checksum request (conv + BN + ReLU: 781 EECU·s, 696 s; the conv prefix 862 EECU·s, 442 s), and the in-graph task C (PSP + FPN +
fusion) was fetched interactively in 32 seven-row blocks. The profiled wall times (442-696 s)
exceed the 300 s interactive deadline Milestone 9 assumed, so either that figure is not the
limit for these requests or the wall time includes queueing — unresolved. Batch remains the
production mode: interactive requests near 1.7 GB and several minutes have no margin.

## 6. The complete task C (Phase 19; Tests 18, 20)

fusion_pipeline.py (archive `upernet/earth_engine/fusion_pipeline.py` at `research-archive-2026-10-12`) `export_fused`: C1-C3 from task A''s
skips asset, C4 from task B''s, → feature norms → PSP → FPN → fusion, **one graph**, exported on
the C1 grid. Every export succeeded:

| export (canonical) | EECU·s | run s | request / invocations | notes |
|---|---:|---:|---|---|
| task C monolithic → conv + BN + ReLU (768 bands) | 952 | 1,182 | 894 KB / 542 | values = interactive = mirror |
| task C monolithic → fused feature (256 bands), milestone asset | 567 | 234 | 816 KB / 543 | values = 768-band asset = interactive = mirror |
| task C monolithic → fused feature, fresh r1 / r2 | 1,121 / 878 | 2,247 / 588 | 820 KB / 543 | salted (no shared sub-graph) |
| **task C B → fused feature, fresh r1 / r2** | **693 / 669** | 354 / 320 | 856 KB / 552 | production formulation |
| fusion alone (pyramid asset): monolithic conv + BN + ReLU, run / r1 / r2 | 739 / 667 / 756 | 788 / 736 / 897 | 325 KB / 351 | |
| fusion alone: A sum + BN + ReLU, run / r1 / r2 | 517 / 669 / 339 | 703 / 912 / 343 | 363 KB / 373 | |
| fusion alone: A partials Y1-Y4 (1,024 bands) | 499 | 409 | 378 KB / 377 | |
| fusion alone: B sum + BN + ReLU, run / r1 / r2 | 569 / 642 / 723 | 756 / 1,187 / 1,178 | 362 KB / 369 | |

Identical graphs vary up to 2× in batch (monolithic task C: 567-1,121), so differences
under ~30% are not resolvable from three runs. The reading: B's two complete task C runs
(669, 693) lie inside the monolithic spread (567-1,121), below its mean (855), with
consistently short run times (320 / 354 s vs 234 / 588 / 2,247 s); neither A nor B is
measurably more or less expensive than the monolithic fusion. "Fusion alone" exports cost more than
the fusion's share of a task C: they read the 1,024-band pyramid asset by stride sampling and
export three tensors.

## 7. Task-C constant budget (Phase 20; Test 19)

Embedded values, counted from the constructors (`fusion.task_c_budget`; explicit BatchNorm,
window layout), and raw parameters:

| component | raw parameters | embedded (C4 28²) |
|---|---:|---:|
| feature norms + PSP + FPN (Milestone 9) | 6,870,848 | 7,443,930 |
| fusion 3 × 3 1024 → 256 (W^T [9216, 256]; grouped: the same values as four slices) | 2,359,296 | 2,359,296 |
| fusion BatchNorm (4 × 49 × 256) | 1,024 | 50,176 |
| fusion resize tables (2×, 4×, 8×; two axes) | — | 2,712 |
| **task C through fusion** | **9,231,168** | **9,856,114** (C4 56²: 9,976,818) |
| segmentation head (M11): Conv 256 → 2 (1 × 1) + bias rows | 514 | 610 |

**9.86 M is 41% of the ~24 M batch cap** (Milestone 5 placed the failure between 24 and
26 M): **14.1 M of headroom**, of which the head needs 610 — plus its × 4 resize tables,
hundreds of values. The cap is no constraint for M11. Every task-C export above embedded all
of it (the fusion constants are part of the graph, no probe needed). Graphs: 816-894 KB,
542-552 invocations.

## 8. Geospatial output, boundaries and the milestone asset (Phases 21-23, 26; Tests 17, 20-21)

| check | result |
|---|---|
| projections of P1…PSP_to_C1, the concatenation, the fused output (in graph) | EPSG:32651, [40, 0, 283200.953…, 0, −40, 2645318.840…] — exact |
| task-C assets (768 bands; milestone 256 bands c0..c255) | every band: EPSG:32651, that transform, 224 × 224, float32 (98.4 MB / 19.6 MB) |
| 40 random 10 m native pixel centres of the milestone asset | 40 / 40 → C1 pixel (row // 4, col // 4) |
| window layout vs georeferenced raster | window row 13 (32 windows) == rows 91-97, exact |
| no implicit resampling | relabels and exact stride samples only; P2 / P3 / PSP explicitly resized before the guarded concatenation |

**Boundaries (Phase 22).** Every region of the canonical fused output — top, bottom, left,
right (corners excluded), the four corners, the interior — is bit-identical to the mirror;
vs SMP the borders are no worse than the interior (ReLU: top 7.7e-6, bottom 1.1e-5, left
8.9e-6, right 8.0e-6, corners 3.1e-6, interior 2.38e-5; 1792 crop alike). Negative control:
the mirror's conv with replicate or reflect padding differs from EE's border by up to
2.73 / 3.07 and equals it exactly on the ring one pixel inside: the padding is observable and
EE's is PyTorch's zeros (Option A: outside the finite tile = 0; no neighbouring imagery).

**Every element (Phase 23).** All 12,845,056 values of conv, BN and ReLU were compared, from
the interactive in-graph fetch (32 blocks), the 768-band task-C asset and the milestone asset:
the three are identical bit for bit and equal the mirror.

**Milestone asset (Phase 26).** `projects/ORIGINAL_PROJECT/assets/upernet/fusion/m10_fused_canonical_896_bb9225`:
the fused decoder feature, 224 × 224 × 256, 40 m, float32, on the C1 grid — = in-graph = mirror,
2.38e-5 from SMP. A validation artifact: production keeps the feature inside task C and
continues to the head.

## 9. The 1792 crop and shardSize (Phases 24-25; Test 22)

Task C at the 1792 crop (C1 448²; C1-C3 from Milestone 9's task-A' asset, C4 = Milestone 6's),
exporting conv + ReLU (512 bands, 291 MB):

| export | EECU·s | run s | request / invocations | vs SMP (conv / ReLU) | same input | mirror |
|---|---:|---:|---|---|---:|---|
| default shards (256 px: 2 × 2 shards) | 4,589 | 1,344 | 1.68 MB / 543 | 2.19 / 2.69e-5 | 2.38e-5 | bit-identical |
| `shardSize=448` (one shard) | **3,082** | **500** | 1.68 MB / 544 | identical to the default export, every element | | bit-identical |

`Export.image.toAsset(shardSize=…)` is accepted and **materially changes the cost: −33%
EECU·s, −63% run time**, values unchanged. This supports Milestone 9's hypothesis: each
256-px shard re-evaluates its halo of the upstream graph — and here the PSP's global pooling —
so a tile exported as one shard computes it once. The 1792 task C costs 3.6-5.4× the
canonical one for 4× the pixels. For production tiles larger than 256 px per side, set
`shardSize` to the tile's C1 extent (448 here; a multiple of 7 keeps windows whole).

## 10. Performance summary (Phase 28)

Canonical crop, batch EECU·s (single runs vary up to 2×; ranges are all runs):

| part | EECU·s | source |
|---|---:|---|
| encoder task A' (S1 → C1, C2, C3) | 308 | Milestone 7 |
| encoder task B' (→ C4) | 62 | Milestone 7 |
| PSP + FPN (task C before fusion) | 259-288 | Milestone 9 (fresh pyramid exports) |
| fusion: resizes + concatenation | ~40 | interactive prefixes (§ 5) |
| fusion conv (monolithic) | ~300-830 | task C − PSP + FPN; interactive whole-map conv 862 |
| fusion BN + ReLU | small | prefixes (§ 5) |
| **complete task C, monolithic** | **567 / 878 / 1,121** | § 6 |
| **complete task C, B** | **669 / 693** | § 6 |

**The fusion block is now the largest single cost of the Swin-T + UPerNet model**: about
two thirds of task C (PSP + FPN ~270 of a 570-1,120 total), more than the whole encoder
(370), and task C is ~70% of the canonical model (~940-1,490 EECU·s end to end so far). That
matches the MAC count (the fusion conv is 118 of the decoder's 163 GMAC) and Milestone 7's
602 EECU·s projection.

## 11. Earth Engine behaviour found or confirmed

1. **`matrixMultiply` stays double until `toFloat`**, and adding double array images stays in
   double: partial products can be accumulated before one rounding (§ 4). EE sums a
   9,216-term dot strictly sequentially in double, in value space and on images.
2. **`shardSize` matters for large exports** (−33% EECU·s, −63% run time at 448², § 9).
3. **Batch costs vary ~2×** between identical (salted) graphs: 567-1,121 for the canonical
   task C; and only ~2 of a user's batch tasks run at once (the rest queue, up to ~30 min).
4. **Interactive throttling and hangs**: with ~20 concurrent heavy interactive requests even
   a trivial `getInfo` took 15-32 min; one request without a client timeout hung for 5.5 h
   after a network interruption. The validation reran with ≤ 7 concurrent requests and a
   900 s socket timeout (session-local), resuming from saved blocks.
5. The profiler reports one peak per algorithm; in every fusion request it is the
   zero-padding unmask of the halo assembly, whatever the formulation — benchmarking one
   group's halo alone (PSP 563 MiB, P1 < 46 MiB) is what attributes it to the resized levels (§ 5).
6. Unguarded binary operations and concatenations resample silently (nearest), as Milestone 9
   found for additions (§ 2).

None affects the fusion's values.

## 12. Recommendation

**Proceed to Milestone 11 (segmentation head 256 → 2, × 4 bilinear upsampling, the first
complete Sentinel-1 → logits forward in native EE): yes.** Against each criterion:

| criterion | evidence |
|---|---|
| fusion numerical correctness | output 2.38e-5 from SMP (1792: 2.69e-5); every intermediate compared; target 1e-4 (§ 3) |
| fusion geospatial alignment | grids, transforms, footprints, window layout, both assets exact; guard on the concatenation (§ 2, § 8) |
| monolithic baseline correctness | 193 / 193 bit-identical to the mirror; layout exact (§ 3) |
| grouped-equivalence results | A within tolerance, not equal; B equal after BN / ReLU, order-insensitive (§ 4) |
| memory reduction | small (3-11%): memory is the halo assembly of the resized levels, linear in the windows a request touches (§ 5) |
| EECU cost | task C 567-1,121 (monolithic), 669-693 (B); fusion the largest single block (§ 6, § 10) |
| task-C constant budget | 9.86 M (41%); 14.1 M headroom (§ 7) |
| task-C graph size | 816-894 KB, 542-552 invocations (§ 6) |
| canonical batch reliability | every export succeeded (§ 6) |
| large-crop scaling | 1792 exact; 3.6-5.4× canonical cost; shardSize −33% (§ 9) |
| remaining head headroom | the head needs 610 of 14.1 M (§ 7) |

For M11: continue from the fused feature inside the same task C (formulation B; the
monolithic path stays as the reference check); the head is a 1 × 1 Conv 256 → 2 **with
bias**, then `UpsamplingBilinear2d(scale_factor=4)` — **align_corners=True**, a different
convention from every resize so far — onto the 10 m input grid (16× the C1 pixels, 2
channels); export with `shardSize` covering the tile.

## 13. Caveats

- **Seeded weights.** The decoder's parameters are Milestone 7's deterministic seeded values;
  execution fidelity is proven for them. B's equality with the monolithic result after BN /
  ReLU is measured on four maps (canonical, edge, synthetic, mirror orders), not guaranteed:
  an output whose exact value lies within ~1e-14 of a float32 rounding midpoint can round
  differently (2 conv outputs here), and a trained checkpoint's values could expose more.
  Both stay within one float32 ulp of the reference.
- **One scene.** Real features are Milestone 6's Sun Moon Lake crops (edge 224, canonical
  896, 1792).
- **Variance.** Batch EECU·s vary up to 2× for identical graphs; interactive EECU·s up to
  ~1.7× (B at 4 window rows: 118 vs 69 for A). Cost conclusions are drawn on ranges, not
  single runs.
- **Peak memory** is EE's per-algorithm accounting, not RAM; attributing it between nodes of
  the same algorithm needed separate requests (§ 5). The single-group benchmarks cover P1 and
  the PSP group; P2 and P3 are inferred to lie between them.
- **Costs spent beyond the fusion itself**: the 1792 task C (two exports, 7,670 EECU·s), the
  repeated canonical task C runs (≈ 5,000), the interactive validation fetches and prefix
  benchmarks.
- **Assets** remain in project `ORIGINAL_PROJECT`: `upernet/fusion/m10_*` (canonical task C,
  milestone, fusion-alone formulations, fresh repeats, 1792 default and shard448); delete the
  benchmark exports when done.

## Tests → evidence

| test | where |
|---|---|
| 1 installed fusion contract | `test_upernet_fusion_contract.py`; `m10_contract` rows |
| 2 fusion level order (unique IDs) | `test_fusion_order_with_unique_ids`; EE `level_ids_concat`; live test 02 |
| 3 P1 identity | `test_p1_resize_is_a_bitwise_identity`; `P1_to_C1` rows; live test 03 |
| 4-6 P2 / P3 / PSP → C1 | ramps, `test_window_resize_weights_are_pytorchs_rule`, resized rows, projections; live tests 04-06 |
| 7 grid guard negative controls | `m10_negative_canonical_896`; live test 07 |
| 8 1,024-channel concat | positional IDs 56² and 224² (server), concat rows; live test 08 |
| 9 fusion im2col ordering | `im2col_ids_C1_1024` (every tap, every edge) |
| 10 weight-row permutation | `fusion_weight_rows`, `group*_weight_rows` |
| 11 baseline Conv-BN-ReLU (synthetic) | `m10_fusion_synthetic_14`; live tests 11-13 |
| 12 baseline on the real canonical pyramid | `m10_fusion_canonical_896`, `m10_task_c_canonical_896` |
| 13 controlled mirror | every `ee_mirror` row (193 / 193); `test_upernet_fusion.py` |
| 14 grouped decomposition | `grouped_Y1-4` rows; `test_decomposition_is_exact_in_real_arithmetic`; live test 14 |
| 15 grouped vs monolithic | `formulation_vs_monolithic` rows, order study; live test 15 |
| 16 high-precision grouped accumulation | probe rows (`m10_ids`), `double_*` rows; live test 16 |
| 17 boundary zero padding | `boundary_*` rows, padding controls, im2col IDs; live test 17 |
| 18 full task C | task-C export rows; live test 18 |
| 19 constant budget | `test_upernet_fusion_budget.py`; § 7; live test 19 |
| 20 geospatial fused-output grid | projection, asset-grid, footprint rows; live tests 20-21 |
| 21 canonical fused-output asset | milestone asset rows; live tests 20-21 |
| 22 larger crop | `m10_fusion_large_1792`; live test 22 |
