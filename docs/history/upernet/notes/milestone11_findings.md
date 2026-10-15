# Findings: the segmentation head and the complete Sentinel-1 → logits forward in native Earth Engine

Live experiments on 8 October 2026: Earth Engine Python API 1.7.45, PyTorch 2.14.0, timm 1.0.30,
segmentation_models_pytorch 0.5.0, NumPy 2.5.3, Cloud project `ORIGINAL_PROJECT`,
application-default credentials (the pinned Milestone 10 environment, re-inspected: unchanged).
Every Earth Engine tensor was computed in EE graphs from EE assets — the Sentinel-1 catalogue
image, task A′'s skips asset, task B′'s C4 asset, Milestone 10's fused-feature asset, Milestone 7's
decoder / head weight tables — or from fixtures generated on the server or embedded as
constants; tensors were downloaded only to compare them. No head convolution, interpolation or
raster placement is computed client-side.

Deliverables follow the brief's suggested layout, adapted to the repository: the
`align_corners=True` resize extends resize.py (archive `upernet/earth_engine/resize.py` at `research-archive-2026-10-12`) (every formulation takes
`align_corners`, default `False`, so Milestones 7-10 are untouched) instead of a separate
`resize_align_corners.py`; the three suggested benchmark modules are one,
benchmark_head.py (archive `upernet/benchmarks/benchmark_head.py` at `research-archive-2026-10-12`), whose batch rows come from
validate_head.py (archive `upernet/earth_engine/validate_head.py` at `research-archive-2026-10-12`)'s exports, as in Milestones 8-10.

Evidence: numerical_comparison_m11.csv (development archive: `docs/history/upernet/results/numerical_comparison_m11.csv`) (summary with
the Phase 21 matrix: numerical_summary_m11.md (development archive: `docs/history/upernet/results/numerical_summary_m11.md`)),
benchmark_results_m11.csv (development archive: `docs/history/upernet/results/benchmark_results_m11.csv`) (summary:
benchmark_summary_m11.md (development archive: `docs/history/upernet/results/benchmark_summary_m11.md`)), per-request records and
manifests in results/raw/ (archive `upernet/results/raw/` at `research-archive-2026-10-12`) (`validation_m11_*`, `m11_head_contract.json`,
`m11_forward_<crop>.json`, `m11_provenance_canonical_896_*.json`, `m11_stats_*.json`,
`feature_assets.json`, `logs/m11_*`). Required Tests 1-24: offline in
test_upernet_head_contract.py (archive `tests/test_upernet_head_contract.py` at `research-archive-2026-10-12`),
test_upernet_head.py (archive `tests/test_upernet_head.py` at `research-archive-2026-10-12`),
test_upernet_align_corners.py (archive `tests/test_upernet_align_corners.py` at `research-archive-2026-10-12`),
test_upernet_full_forward.py (archive `tests/test_upernet_full_forward.py` at `research-archive-2026-10-12`),
test_upernet_m11_budget.py (archive `tests/test_upernet_m11_budget.py` at `research-archive-2026-10-12`) (91 passed), live in
test_upernet_m11_live.py (archive `tests/test_upernet_m11_live.py` at `research-archive-2026-10-12`) (`EE_LIVE=1`: 9 passed); the full
validation is validate_head.py (archive `upernet/earth_engine/validate_head.py` at `research-archive-2026-10-12`): **578 rows (514 measured
against an EE tensor or a reference), 0 failing; all 139 EE-vs-mirror comparisons of the milestone
bit-identical (96.5 M elements; 153 / 153 with the error budget's restated Milestone 7-10 rows); 18
negative controls that differ as they must.**

**Research questions.** (1) What exactly is the installed head? (2) Does EE reproduce its
biased 256 → 2 convolution with Milestone 7's linear primitive and asset weights? (3) Does EE
reproduce the new `align_corners=True` interpolation, globally, edges included? (4) Can the
result sit on the exact 10 m Sentinel-1 grid without EE resampling? (5) Does the complete
A′ → B′ → C workflow reproduce `smp.UPerNet.forward`? (6) Does formulation B stay equivalent to
the monolithic fusion through the head? (7) What does the 10 m export cost, and which
`shardSize`? (8) Does the canonical workflow complete without a new task boundary?
**Yes on all eight, with one cost
finding. (1) Conv2d 256 → 2 with bias → `UpsamplingBilinear2d(4)` = `align_corners=True` → Identity:
`model.forward` returns raw logits; PyTorch evaluates `align_corners` in float32 in the float32
model, so its coefficients are not the textbook ones (≤ 2.7e-5 off). (2) Yes: Milestone 2's linear
with the asset weights, bit-identical to the mirror, bias added in double before the one rounding.
(3) Yes: Milestone 7's window resize with one global-coefficient matrix per child window and the
same one-token halo; every fixture from 2 → 8 to 448 → 1792 bit-identical to the mirror and ≤ 1 ulp
from PyTorch; EE's own bilinear cannot do it (0.31 off). (4) Yes: computed by index, then relabelled
onto the crop's native 10 m grid — no resampling, 48 / 48 footprints, every mask clear. (5) Yes:
5.54e-6 from SMP's `model.forward` over all 1,605,632 canonical logits (target 1e-4), through a
fresh chain of three tasks from the catalogue image. (6) Yes: bit-identical to the monolithic
reference at the fused feature, the 40 m and the 10 m logits. (7) Task C exported at 10 m costs
1,222-1,636 EECU·s (2 × 2 shards cheapest), ~2× its 40 m export: the head is 25, the 10 m tile
geometry re-evaluates the decoder; a task boundary at the 40 m logits halves it (measured, bit-
identical). (8) Yes: no new boundary was needed; one graph from the feature assets to the logits
runs at 896 and at 1792.**

## The headline

A real Sentinel-1 image now passes through the complete Swin-T + SMP UPerNet inside Earth
Engine, across three chained native tasks, to a georeferenced two-channel logit raster that
matches the installed PyTorch model: **every one of the 1,605,632 canonical logits is within
5.54e-6 of `smp.UPerNet.forward`** (RMSE 9.8e-7; the 1792 crop's 6.4 M logits within 5.93e-6), and
bit-identical to the controlled EE mirror.

What was new is small in code and subtle in numerics:

- **The head conv** is Milestone 2's linear with a bias; the bias joins the double dot before the
  one rounding (probed: 1 + 2⁻²⁴ + 2⁻²⁴ → 1 + 2⁻²³).
- **The final resize uses `align_corners=True`** — a global end-point map, src = o·(N − 1)/(M − 1),
  unlike every decoder resize so far. Two facts decide its implementation. First, PyTorch evaluates
  it *in the tensor's precision*, so the float32 model's coefficients deviate from the textbook ones
  by up to 1.4e-5 (2.7e-5 at 1792) while reading the same source pixels: EE embeds the float32
  kernel's own coefficients. Second, for an integer factor every output reads only its parent token
  and the parent's neighbours, so Milestone 7's window resize with its 9 × 9 halo stays exact; it
  just needs one coefficient matrix per child window (128 per axis at 896), sliced from the global
  map — a per-window template would be wrong (up to 4.6 off on fixtures).
- **Non-dyadic coefficients make EE's double arithmetic visible for the first time**; the probe
  shows `matrixMultiply` rounds each product and each sum separately (no FMA), and the mirror
  written that way matches EE on 100% of double values.
- **Tensor interpolation and georeferencing are separate steps.** The logits are computed by
  index and then placed, by a relabel, on the input crop's own 10 m grid. EE's geospatial bilinear
  (pixel-centre, = `align_corners=False`) is not the model's resize and is rejected (0.31 off).

The production formulation B equals the monolithic reference bit for bit through the head. The
error budget is flat to falling: the fused feature's 2.38e-5 shrinks to 6.2e-6 through the seeded
256 → 2 projection and to 5.5e-6 through the interpolation; EE stays ~20× closer to float64 than
SMP float32 once both use the float32 model's coefficients.

The one cost surprise: the head costs 25 EECU·s, but exporting task C at 10 m costs ~2× exporting
it at 40 m (1,222-1,617 vs 685), because the decoder is re-evaluated inside 10 m output tiles. The
one-graph task C — the brief's topology — runs reliably at 896 and 1792; a task boundary at the
40 m logits is measured, bit-identical and half the cost, and is the recommended production
variant for Milestone 12.

**Milestone verdict.** **Minimum, strong and full success are all met.** The fused-feature asset
passes through the biased head, the exact `align_corners=True` × 4 interpolation and onto the 10 m
grid within 4.8e-7 of SMP's head (minimum); both fusion formulations are validated through the
head and the whole task C — feature assets → norms → PSP → FPN → fusion → head → logits — runs as
one native batch graph at canonical size (and 1792) with complete numerical and geospatial
comparisons (strong); and a fresh chain catalogue image → task A′ → task B′ → task C produced the
canonical logits with no external inference and no client-side tensor in any task, bit-identical
to the production run and within 5.54e-6 of the installed `model.forward` over every logit, with
an auditable provenance manifest (full).

## Validation matrix (Phase 21)

Generated by `python -m upernet.benchmarks.summarize --milestone 11` from the CSV.  Primary reference per row as named (torch_float32 = SMP's captured model.forward on the preprocessed crop; torch_float32_head_input = SMP's head alone on EE's fused feature).  Layout / index / probe rows must be exact.  "max vs float64" at the 10 m logits is the float32 and float64 models' different align_corners coefficients (findings § 10): with the float32 model's coefficients the float64 logits are 2.4e-7 from EE.

| operation | experiment | reference | MAE | RMSE | max abs | max vs float64 | EE vs mirror | exact |
|---|---|---|---:|---:|---:|---:|---|---|
| head weight + bias asset read-back (514 values) | m11_weights | smp_state_dict | 0 | 0 | 0 | — | — | exact |
| head bias read-back | m11_weights | smp_state_dict | 0 | 0 | 0 | — | — | exact |
| head output-class / channel ordering (IDs) | m11_ids | class_channel_ids | 0 | 0 | 0 | — | bit-identical | exact |
| head one-hot channels, real W + b | m11_ids | W[:, j] + b | 0 | 0 | 0 | — | bit-identical | exact |
| bias added before the one rounding (probe) | m11_ids | one_rounding_expectation | 0 | 0 | 0 | — | bit-identical | exact |
| head Conv on synthetic fused feature (± inputs, real b) | m11_ids | torch_float32 | 3.15e-07 | 4.49e-07 | 2.38e-06 | — | bit-identical | pass |
| head Conv on synthetic fused feature (bias [3, -2]) | m11_ids | torch_float32 | 7.85e-07 | 1.04e-06 | 4.77e-06 | — | bit-identical | pass |
| head Conv on real canonical fused feature (40 m) | m11_isolated_canonical_896 | torch_float32_head_input | 5.11e-08 | 6.83e-08 | 5.36e-07 | 2.56e-07 | bit-identical | pass |
| 224 → 896 coordinate ramps (interpolated source row / column) | m11_ids | torch_float32 | 0 | 0 | 0 | — | bit-identical | exact |
| source-pixel membership 224 → 896 (one-hot combs) | m11_ids | exact | — | — | — | — | — | exact |
| 2 × 2 → 8 × 8 (matrix formulation) | m11_ids | torch_float32 | 4.48e-08 | 1.15e-07 | 4.77e-07 | — | bit-identical | pass |
| 3 × 3 → 12 × 12 | m11_ids | torch_float32 | 5.61e-08 | 1.21e-07 | 4.77e-07 | — | bit-identical | pass |
| 5 × 7 → 20 × 28 | m11_ids | torch_float32 | 5.1e-08 | 1.08e-07 | 9.54e-07 | — | bit-identical | pass |
| 1 × 5 → 4 × 20 (N = 1) | m11_ids | torch_float32 | 2.81e-08 | 7.18e-08 | 2.38e-07 | — | bit-identical | pass |
| 7 → 28 across one window (window formulation) | m11_ids | torch_float32 | 4.14e-08 | 9.01e-08 | 4.77e-07 | — | bit-identical | pass |
| 14 → 56 across window boundaries | m11_ids | torch_float32 | 5.13e-08 | 1.03e-07 | 9.54e-07 | — | bit-identical | pass |
| 21 → 84 (3 × 3 windows) | m11_ids | torch_float32 | 4.79e-08 | 9.9e-08 | 9.54e-07 | — | bit-identical | pass |
| 224 → 896 hash fixture (global coordinates, edges) | m11_ids | torch_float32 | 2.99e-08 | 5.88e-08 | 4.77e-07 | — | bit-identical | pass |
| 448 → 1792 hash fixture | m11_ids | torch_float32 | 2.94e-08 | 5.85e-08 | 4.77e-07 | — | bit-identical | pass |
| double-precision probe (no FMA) | m11_ids | mirror_double_no_fma | 0 | 0 | 0 | — | — | exact |
| Option B (dense) == window, 14 → 56 | m11_ids | ee_window | 0 | 0 | 0 | — | bit-identical | exact |
| native EE bilinear (negative control) | m11_native_canonical_896 | torch_float32_head_input | 0.0113 | 0.022 | 0.314 | — | — | differs (negative control) |
| native EE bilinear vs align_corners=False | m11_native_canonical_896 | torch_align_corners_false | 3.37e-08 | 4.58e-08 | 4.77e-07 | — | — | pass |
| head 40 m → logits 10 m on the same EE fused input | m11_isolated_canonical_896 | torch_float32_head_input | 3.41e-08 | 4.64e-08 | 4.77e-07 | 5.55e-06 | bit-identical | pass |
| monolithic fusion → head → 40 m logits (batch) | m11_task_c_canonical_896 | torch_float32 | 1.05e-06 | 1.32e-06 | 6.22e-06 | 2.56e-07 | bit-identical | pass |
| **monolithic fusion → head → logits (batch)** | m11_task_c_canonical_896 | torch_float32 | 7.78e-07 | 9.83e-07 | 5.54e-06 | 5.55e-06 | bit-identical | pass |
| monolithic vs B, fused feature (Milestone 10 task-C exports) | m11_task_c_canonical_896 | ee_monolithic_m10_milestone | 0 | 0 | 0 | — | — | exact |
| monolithic vs B, 40 m logits | m11_task_c_canonical_896 | ee_monolithic | 0 | 0 | 0 | — | — | exact |
| monolithic vs B, 10 m logits | m11_task_c_canonical_896 | ee_monolithic | 0 | 0 | 0 | — | — | exact |
| task-C batch logits vs the interactive isolated head | m11_task_c_canonical_896 | ee_isolated_interactive | 0 | 0 | 0 | — | — | exact |
| **complete EE 896 input → 896 × 896 × 2 logits (production B)** | m11_shards_canonical_896 | torch_float32 | 7.78e-07 | 9.83e-07 | 5.54e-06 | 5.55e-06 | bit-identical | pass |
| production exports, other shardSize (448 / default vs 896) | m11_shards_canonical_896 | ee_shard896_r0 | 0 | 0 | 0 | — | — | exact |
| production repeat, same shardSize (r1 vs r0) | m11_shards_canonical_896_r1 | ee_r0_same_shard | 0 | 0 | 0 | — | — | exact |
| **fresh A′ → B′ → C chain logits** | m11_chain_canonical_896 | torch_float32 | 7.78e-07 | 9.83e-07 | 5.54e-06 | 5.55e-06 | bit-identical | pass |
| chain logits == production export | m11_chain_canonical_896 | ee_production_shard448_r0 | 0 | 0 | 0 | — | — | exact |
| chain C4 == Milestone 7's task-B′ C4 | m11_chain_canonical_896 | ee_m7_task_B | 0 | 0 | 0 | — | — | exact |
| split topology: task C (40 m logits) + task D (× 4) | m11_split_canonical_896 | torch_float32 | 7.78e-07 | 9.83e-07 | 5.54e-06 | 5.55e-06 | bit-identical | pass |
| split == one-graph production, every logit | m11_split_canonical_896 | ee_one_graph_production | 0 | 0 | 0 | — | — | exact |
| swath-edge full forward (224 crop) | m11_edge_edge_224 | torch_float32 | 6.95e-07 | 8.82e-07 | 5.04e-06 | 2.27e-06 | bit-identical | pass |
| 1792 full forward (1792 × 1792 × 2) | m11_large_large_1792 | torch_float32 | 7.97e-07 | 1e-06 | 5.93e-06 | 2.65e-05 | bit-identical | pass |

Pure layout, index, probe and repeat rows are exact; every floating-point row is within 5.93e-6 of its
primary reference (target 1e-4) and bit-identical to the mirror.

## Answers

| # | Question | Answer |
|---|---|---|
| 1 | Which installed SMP class and source implement the head? | `segmentation_models_pytorch/base/heads.py` `SegmentationHead(nn.Sequential)` (lines 5-18), built in `decoders/upernet/model.py` `UPerNet.__init__` with `kernel_size=1, upsampling=4, activation=None`, called by `base/model.py` `SegmentationModel.forward` (`masks = self.segmentation_head(decoder_output)`) (§ 1). |
| 2 | Weight shape, bias, class order? | `segmentation_head.0.weight` [2, 256, 1, 1], `.bias` [2] (514 values, seeded, non-zero bias), asset record `segmentation_head.0` [2, 256] + bias; output channel k = class index k = weight row k (ID fixtures exact) (§ 1, § 3). |
| 3 | Is the activation Identity; does `model.forward` return logits? | **Yes**: `Activation(None)` = `nn.Identity`; the hooked forward returns the upsample's output unchanged: raw logits (§ 1). |
| 4 | Which upsampler and convention? | `nn.UpsamplingBilinear2d(scale_factor=4.0)` = `F.interpolate(scale_factor=4, mode="bilinear", align_corners=True)` (recorded call; bit-identical to the `size=` call, to a channels-last input; ≠ `align_corners=False`) (§ 1). |
| 5 | Source coordinates for 224 → 896 and 448 → 1792? | src = o·223/895 and o·447/1791 — **evaluated in float32** by the float32 model: src = f32(f32(223/895)·o), i0 = min(⌊src⌋, N − 1), l1 = src − i0, l0 = f32(1 − l1); coefficients ≤ 1.37e-5 / 2.71e-5 from the rational ones, same source pixels (§ 2). |
| 6 | Does EE's geospatial bilinear reproduce it? | **No**: `resample("bilinear")` + `reproject` is pixel-centre interpolation (= PyTorch `align_corners=False` within one ulp); vs `align_corners=True` up to 0.314 (interior), 0.141 (edges), equal only at the four corners (§ 4). |
| 7 | Best EE implementation across window boundaries? | Milestone 7's window resize with the global `align_corners=True` coefficients: per 7 × 7 output window, the parent C1 window's 9 × 9 halo and two [7, 9] matrices sliced from the global map at its absolute rows / columns (§ 4). |
| 8 | What halo / absolute-coordinate support does it need? | **The one-token halo**: for any integer factor src ∈ (o/f − (f − 1)/f, o/f), so outputs read only the parent token and its neighbours (proved, asserted per row, checked N = 7 … 448, f = 2, 4, 8); one coefficient matrix per child window (128 / 256 per axis at 896 / 1792) indexed by the absolute window position (§ 2, § 4). |
| 9 | Is the 256 → 2 biased conv correct and asset-backed? | **Yes**: Milestone 2's linear on [49, 256] windows with W, b read from Milestone 7's asset (bit-exact read-back); IDs, one-hot channels and the bias-rounding probe exact; bit-identical to the mirror (§ 3). |
| 10 | Its max error vs SMP float32 and float64? | Same fused input: 5.4e-7 vs SMP's head (float32), 3.0e-8 vs float64; in the full forward 6.2e-6 vs `model.forward` (inherited from the fused feature), 2.6e-7 vs float64 (§ 3, § 7, § 10). |
| 11 | Final resize error on synthetic fixtures? | ≤ 9.5e-7 (≤ 1 ulp at the fixtures' magnitudes) from PyTorch, bit-identical to the mirror, on every fixture from 2 × 2 → 8 × 8 to 448 → 1792 (floats, ramps, combs; N = 1 included) (§ 4). |
| 12 | Its error on the same real fused input? | 40 m → 10 m on Milestone 10's fused asset: **4.8e-7** from SMP's head (11.6% of logits equal), 5.5e-8 from float64 with the same coefficients (§ 7). |
| 13 | Does the mirror reproduce every new EE tensor bit for bit? | **Yes**: all 139 EE-vs-mirror comparisons of the milestone (96.5 M elements: fixtures, the isolated head, every task-C, shard, chain, split, edge and 1792 export); and at the double level the resize matches the no-FMA mirror on 100% of values (§ 5). |
| 14 | Does formulation B stay equivalent through the head? | **Yes, bit for bit**: fused feature (12.8 M), 40 m logits (100,352) and 10 m logits (1,605,632) all identical to the monolithic reference — measured on this map (§ 8). |
| 15 | Is the 896 × 896 × 2 tensor ordered like SMP's NCHW logits? | **Yes**: logits[row, col, k] = SMP's output[0, k, row, col] at every element (all comparisons index-aligned; window rows vs raster exact; footprints) (§ 6-8). |
| 16 | Exact native projection, origin, size, footprints? | **Yes**: EPSG:32651, [10, 0, 283200.953…, 0, −10, 2645318.840…], 896 × 896, every band of every asset; 48 / 48 footprints (40 random, 4 corners, 4 edge midpoints) → logits[row, col] (§ 6). |
| 17 | How are tensor end-point alignment and the EE pixel grid reconciled without a resample? | Two steps: compute SMP's values by index (the tensor map aligns end-point indices, not pixel centres), then **relabel** output (row, col) onto native pixel (row, col) of the input crop (depth_to_space + changeProj): values moved, never resampled; downstream must not resample again (§ 6). |
| 18 | Final canonical max error vs `model.forward`? | **5.54e-6** (MAE 7.8e-7, RMSE 9.8e-7) over all 1,605,632 logits; max relative 4.9e-4 with denominator max(\|SMP\|, 0.01) (§ 8, § 10). |
| 19 | Where is the largest error, and why? | At (row 803, col 237, class 0), interior: it interpolates almost entirely from C1 token (200, 59), where the 40 m logit error peaks (6.2e-6) — SMP's float32 error of the fused feature (2.38e-5 max) projected through the head weights (−6.14e-6 of −6.22e-6); the head's own rounding 3.7e-8 (§ 10). |
| 20 | Does error accumulate after fusion? | **No, it falls**: fused 2.38e-5 → head conv 6.2e-6 (the 256 → 2 projection averages it) → logits 5.5e-6 (the interpolation is a convex combination: max and RMS both drop) (§ 10). |
| 21 | One native batch graph from feature assets to logits? | **Yes**: at canonical (eight exports: three shard settings, both formulations, the chain; every one succeeded) and at 1792 (§ 8, § 11, § 13). |
| 22 | Is the real image traced through A′ → B′ → C with no external computation? | **Yes**: a fresh chain from the catalogue image (salted, recomputed): A′ 312, B′ 61, C 2,220 EECU·s; each task's graph loads only the catalogue image / previous assets / weight tables; logits bit-identical to the production run and the mirror, 5.54e-6 from `model.forward`; task and asset IDs in the provenance manifests (§ 9). |
| 23 | Task-C embedded values incl. interpolation tables? | **9,873,108** (through fusion 9,856,114 + head conv 610 + coefficient tables 16,384): 41.1% of ~24 M; 1792: 10,010,196 (§ 11). |
| 24 | Task-C graph size, memory, batch runtime / EECU? | 935 KB / 574 invocations (production); head peak 105 MiB interactively; batch **1,222-1,271 EECU·s at shardSize 448** (379-692 s), 1,602-1,617 at 896, 1,351 default; 1792: 7,102 (§ 11). |
| 25 | Fraction of task C that is head / interpolation / 10 m export? | Head conv + resize + raster ≈ 2-3 EECU·s (0.2%); with the 10 m write and its input read 25 (2%); but the 10 m output geometry adds ~530-930 (~45%) by re-evaluating the decoder per 10 m tile (§ 11). |
| 26 | How much does shard size matter; recommendation? | ~25% EECU·s (448: 1,222-1,271; 896: 1,602-1,617; default 256: 1,351) and 3-6× run time (default 2,075 s vs 337-692 s); **recommended `shardSize` = half the tile side (448 at 896, 896 at 1792)**; the split topology halves the cost (§ 11). |
| 27 | Is the 896² output repeatable and preserved by the asset round-trip? | **Yes**: eight canonical one-graph exports (five production at three shard settings incl. two repeats, two validation in both formulations, the chain) and the split are identical bit for bit, and equal the interactive in-graph tensors and the mirror (§ 8, § 9, § 11). |
| 28 | Do swath-edge / invalid-pixel cases keep the dense-input semantics? | **Yes**: edge 224 (~50% valid, training-mean fill) within 5.0e-6 of SMP, bit-identical to the mirror, every pixel of every band unmasked; QC (valid fraction = Milestone 6's exactly) is a separate asset (§ 12). |
| 29 | Can the 1792² case run; scaling? | **Yes, in batch**: 7,102 EECU·s, all 6,422,528 logits within 5.93e-6 of SMP, bit-identical to the mirror; 4.4-5.8× the canonical cost for 4× the pixels (§ 13). |
| 30 | New EE limits from writing full-resolution logits? | **None hit**: the 10 m two-band outputs, 0.9-1.8 MB graphs and 1-16 shards all ran; the cost of 10 m tiles (~2×) is the finding (§ 11, § 13). |
| 31 | Which fusion formulation is retained, why? | **B** (double group partials, one rounding) for production: bit-identical to the monolithic reference through the head, no cost penalty; monolithic stays the runnable reference (§ 8). |
| 32 | What is now proven about native-EE end-to-end inference? | § 15. |
| 33 | What remains unproven? | Trained behaviour and class meaning, calibration, other scenes, tile sizes and mosaicking, production cost (§ 15). |
| 34 | Proceed to Milestone 12? | **Yes**, with the prerequisites in § 15. |

## 1. The installed head and the reference (Phases 1, 5; Tests 1-3)

Read off and run from the installed packages
(reference/smp_segmentation_head_reference.py (archive `upernet/reference/smp_segmentation_head_reference.py` at `research-archive-2026-10-12`),
recorded in `results/raw/m11_head_contract.json`):

| item | installed (smp 0.5.0, torch 2.14.0) |
|---|---|
| head class | `base/heads.py` `SegmentationHead(nn.Sequential)` (lines 5-18): `Conv2d(in, out, kernel_size, padding=kernel_size // 2)` (bias: `nn.Conv2d`'s default `True`), `nn.UpsamplingBilinear2d(scale_factor=upsampling)` if `upsampling > 1`, `Activation(activation)` |
| construction | `decoders/upernet/model.py` `UPerNet.__init__`: `SegmentationHead(in_channels=decoder_channels, out_channels=classes, activation=activation, kernel_size=1, upsampling=upsampling)` |
| use | `base/model.py` `SegmentationModel.forward`: `masks = self.segmentation_head(decoder_output)`; returned as is (no classification head) |
| [0] conv | Conv2d 256 → 2, 1 × 1, stride 1, padding 0, dilation 1, groups 1, **bias=True** |
| [1] upsample | `UpsamplingBilinear2d(scale_factor=4.0, mode='bilinear')`: its `__init__` passes **`align_corners=True`**; `forward` = `F.interpolate(input, None, 4.0, "bilinear", True, recompute_scale_factor=None)`; recorded in a real forward: 56² → 224² (224 input), `size=None`, `antialias=False` |
| [2] activation | `Activation(None)` → `nn.Identity`: **`model.forward` returns raw logits** |
| parameters | `segmentation_head.0.weight` [2, 256, 1, 1], `.bias` [2]: **514 values**, SHA-256 equal to `data/manifest.json`; asset record `segmentation_head.0` ([2, 256], bias true) in Milestone 7's tables; seeded bias [0.00306, 0.02253] (non-zero) |
| model | in_channels 2, classes 2, decoder_channels 256, upsampling 4, eval mode (inference BatchNorm), no preprocessing wrapper |

**The real forward, observed (Phase 1).** reference/full_model_reference.py (archive `upernet/reference/full_model_reference.py` at `research-archive-2026-10-12`)
`capture` runs `model.forward` once with hooks and keeps every tensor it computes, with its own
strides: the input, the patch embedding, C1-C4 at the decoder, the four feature norms, the PSP
output, F3 / F2 / F1, P3 / P2 / P1, the fusion block's input and output, the head's input, its
Conv2d, its upsample, its activation and the return value. `expanded` recomputes the whole
forward from the installed submodules one operation at a time (Milestone 10's expanded decoder,
then the head written out as `head[0]`, the explicit `F.interpolate(scale_factor=4, bilinear,
align_corners=True)`, `head[2]`) and raises unless **every tensor equals the captured forward bit
for bit, the logits included** — at a random 224 input, and on the real edge, canonical and 1792
crops in float32 (canonical and edge in float64 too). The captured fused features equal
Milestone 10's cached references bit for bit (all three crops), so Milestones 7-10's
references are the real forward's tensors.

**Layouts.** The head's input (the fusion block's output) is NCHW-contiguous in the real
forward (stride [12845056, 50176, 224, 1] at 896); so are its conv output and the logits. The
references feed SMP's head NCHW-contiguous tensors accordingly (Milestone 8's lesson). The
upsample's output is the same for a channels-last copy of its input, for `F.interpolate(size=)`
and for `F.interpolate(scale_factor=4)`, bit for bit; `align_corners=False` differs.

**Statistics (Phase 1)**, SMP float32 on the canonical crop (`results/raw/m11_forward_canonical_896.json`):

| tensor | shape | min | max | mean | std |
|---|---|---:|---:|---:|---:|
| preprocessed input | 896² × 2 | −4.889 | 6.05 | 1.004 | 0.808 |
| C1 raw / C4 raw | 224² × 96 / 28² × 768 | −10.81 / −11.12 | 10.21 / 9.39 | −0.127 / 0.019 | 2.44 / 2.57 |
| P1 / PSP | 224² × 256 / 28² × 256 | 0 | 8.17 / 3.67 | 0.875 / 0.360 | 1.22 / 0.52 |
| fused feature | 224² × 256 | 0 | 4.374 | 0.458 | 0.731 |
| head conv (40 m logits) | 224² × 2 | −0.672 | 0.858 | 0.102 | 0.169 |
| **logits** (10 m) | 896² × 2 | −0.671 | 0.851 | 0.102 | 0.162 |

The logits are small (|z| < 0.86): the seeded head weights are N(0, 1/256), so the 256-channel
projection shrinks the fused feature's scale by ~16×. That also shrinks absolute errors (§ 10).

## 2. `align_corners=True`, exactly (Phases 3-4; Tests 6-8)

**The rule, read off PyTorch's kernel.** For an axis of N inputs and M outputs PyTorch evaluates
(ATen `area_pixel_compute_scale`, `area_pixel_compute_source_index`, `guard_index_and_lambda`):

    scale = (N − 1) / (M − 1)            (0 if M = 1; a copy if M = N)
    src   = scale · o
    i0    = min(⌊src⌋, N − 1),  i1 = i0 + 1 if i0 < N − 1 else i0
    l1    = clamp(src − i0, 0, 1),  l0 = 1 − l1

**in the tensor's own precision.** The float32 model computes `scale` and `src` in float32, so its
coefficients are *not* the textbook `o·(N − 1)/(M − 1)`:
pytorch_decoder_primitives.align_corners_formula (archive `upernet/reference/pytorch_decoder_primitives.py` at `research-archive-2026-10-12`)
writes the rule out in float32 and in exact rationals, and the float32 kernel's weights (read off
it with one-hot inputs) equal the written float32 formula at every output index of every size
tested — 1 → 4 (N = 1: a copy of the single row), 1 → 1, 4 → 1 (M = 1), 6 → 6 (copy), 2 → 8, 3 → 12,
5 → 20, 7 → 28, 14 → 56, 21 → 84, 56 → 224, **224 → 896**, **448 → 1792**:

| N → M | max \|float32 kernel − exact\| | float64 kernel − exact | float32 rows = float64 rows | membership changed | integer sources |
|---|---:|---:|---:|---:|---|
| 2 → 8 | 6.8e-8 | 1.1e-16 | 25% | 0 | 0, 7 |
| 7 → 28 | 2.6e-7 | 6.1e-16 | 14% | 0 | 0, 9, 18, 27 |
| 14 → 56 | 5.5e-7 | 6.8e-16 | 3.6% | 0 | 0, 55 |
| 56 → 224 | 3.2e-6 | 5.6e-15 | 0.9% | 0 | 0, 223 |
| **224 → 896** | **1.37e-5** | 2.5e-14 | 0.22% | **0** | 0, 895 |
| **448 → 1792** | **2.71e-5** | 4.8e-14 | 0.22% | **0** | 0, 597, 1194, 1791 |

The float32 coefficients deviate from the rational ones by up to 1.4e-5 (canonical) and 2.7e-5
(1792) — the float32 rounding of `scale` (≤ 2⁻²⁶ relative) times `o` up to 1791, plus the rounding
of `src` — but **no output reads a different pair of source pixels**: even at 448 → 1792's
interior integer sources (597 → 149, 1194 → 298) the float32 coordinate does not fall below the
integer. The float64 model's coefficients are the rational ones to ~1e-14, so **the float32 and
float64 models use different coefficients** (equal rows: 0.22%): § 10 separates this from
arithmetic error. EE must therefore embed the float32 model's own coefficients (the exact ones
alone would put up to 1.4e-5 × |Δz| into the logits). Milestone 8 made the same choice for the
PSP's non-dyadic resizes.

**Global, not per window.** `src(o) = o·(N − 1)/(M − 1)` is defined over the whole finite tensor.
For an integer factor f and o in child window I′ of parent token window I, `src` lies in
`(o/f − (f − 1)/f, o/f)`, so `i0 ≥ ⌊o/f⌋ − 1` and `i1 ≤ ⌊o/f⌋ + 1`: every output reads its parent
token and the parent's direct neighbours — inside the parent window's one-token ring, which is
Milestone 7's 9 × 9 halo, for every window and every supported size (proved here, asserted for
every output row by `resize.local_matrices`, and checked offline for N = 7 … 448 and f = 2, 4, 8).
A per-window 7 → 28 `align_corners=True` template is a different map: it resets the endpoints in
every window (§ 4).

**Endpoints.** Output 0 is source 0 exactly (l0 = 1); output M − 1 reads sources N − 2 / N − 1
with l1 = 1 − ε or exactly N − 1. The tensor's first / last pixels are interpolation endpoints;
nothing outside the finite tile is read (Milestone 8's Option A: the tile is the model input).

**PyTorch's arithmetic.** SMP's float32 upsample differs from "exact double with its own float32
coefficients, rounded once" by at most one float32 ulp (71-72% of outputs equal at 224 / 448
inputs): its kernel rounds the row interpolations to float32. EE's result is the latter
(§ 3-5), so EE vs SMP on the same 40 m logits is ≤ 1 ulp.

## 3. The biased 256 → 2 head convolution in EE (Phase 2; Tests 3-5)

segmentation_head.py (archive `upernet/earth_engine/segmentation_head.py` at `research-archive-2026-10-12`) `head_conv` is Milestone 2's linear
(Milestone 7's `conv1x1`), unchanged: each C1 window's [49, 256] × Wᵀ [256, 2] (`matrixMultiply`,
sequential double dots of exact float32 × float32 products), **+ the bias rows [49, 2] in double**,
one `toFloat`. W and b are the asset record `segmentation_head.0` (ServerArrays): no new upload.

| check (`m11_weights`, `m11_ids`) | result |
|---|---|
| W [2, 256], b [2], Wᵀ [256, 2] read back from the asset | **bit-identical** to the state dict (514 values); row k = class k |
| one-hot input channel j(token) with the real W, b | out = fl32(W[:, j] + b) at every token, exact; = PyTorch |
| class / channel IDs: W[k, c] = 1000 (k + 1) + c, b = [0.5, −0.25] | out_k = 1000 (k + 1) + j + b_k exactly: class order, channel order and weight flattening |
| zero / real / [3, −2] bias, ± and non-negative inputs (14² × 256) | bit-identical to the mirror; ≤ 4.8e-6 from PyTorch's float32 conv |
| bias-rounding probe: dot = 1 + 2⁻²⁴, b = 2⁻²⁴ | **1 + 2⁻²³**: the bias is added to the double dot before the one rounding (rounding first would give 1) |

**Where the bias goes.** In EE: `fl32(Σ_c W_kc x_c + b_k)`, the dot exact-ish in double, one rounding.
PyTorch's float32 conv equals `fl32(conv without bias + b)` in only 40% of outputs and EE's
expression in 11% (≤ 1.4e-6 apart on |N(0, 1)| inputs): it accumulates in float32 with the bias
inside its kernel. The difference is PyTorch's float32 accumulation, characterized, not an
ordering error (the ID fixtures are exact).

## 4. The final upsampling in EE: design study (Phase 5; Tests 6-9)

**A — window-native, absolute coordinates (production).** `resize.window_resize(..., factor=4,
align_corners=True)`: Milestone 7's window resize, now fed the global `align_corners=True`
coefficients. Each 7 × 7 output window at the 10 m scale (128 × 128 windows at 896²) reads its
parent C1 window's [9, 9, 2] halo (nearest, exact) and applies two [7, 9] matrices — rows
7I′ … 7I′ + 6 of the GLOBAL 224 → 896 matrix restricted to the parent's halo columns — indexed by
the child window's absolute row / column: **128 distinct matrices per axis at 896² (256 at
1792²), one per child window**, embedded as a [896, 9] (1792: [1792, 9]) stack and a child index
per axis. y first ([7, 9] × [9, 9·2]), then x ([7, 9] × [9, 7·2]), one rounding;
`depth_to_space` then gives 2 scalar bands on the input-pixel grid. The halo bound (§ 2) is what
makes the window layout exact across window boundaries: no wider halo is needed.

**B — explicit output-grid sampling.** `resize.dense_absolute_resize`: the 40 m logits as 2
scalar bands, the parent pixel and its 8 neighbours as exact translations read nearest at 10 m,
weights from [3, 896] per-axis tables indexed by the absolute output row / column (5,376 values).
The same y-then-x arithmetic.

**C — EE's geospatial bilinear (comparison).** `resample("bilinear")` + `reproject` onto the
10 m grid (`resize.native_resize`), with EE's masks or a replicate ring.

| formulation | vs SMP's head on the same input | vs mirror | EECU·s (interactive, whole map, 2 runs) | peak | constants |
|---|---:|---|---:|---:|---:|
| **A window** (production) | ≤ 4.8e-7 (1 ulp) | bit-identical | 6.8 / 9.5 | 105 MiB | 16,384 |
| B dense | ≤ 4.8e-7 | bit-identical; **= A bit for bit** | 15.6 / 11.8 | 114 MiB | 5,376 |
| C native `resample("bilinear")` | **0.314** (interior), 0.141 (top edge); corners equal | — | 9.7 / 6.7 | 114 MiB | 0 |

(EECU·s include the ~5.6 of reading the fused asset.) A and B are identical arithmetic in a
different layout; A is cheaper in the measured runs and keeps the window layout, so it is the
production resize (and the semantic reference, with the mirror). **C is rejected**: EE's
geospatial bilinear interpolates between pixel *centres* — it equals PyTorch's
`align_corners=False` resize of the same 40 m logits within 4.8e-7 (one ulp; masked or ringed
edges alike, Milestone 7's equivalence) — so it differs from `align_corners=True` by up to 0.31 in
the interior (the two source coordinates differ by a shift varying linearly from −3/8 to +3/8 of
a 40 m pixel, ±15 m, across the tile) and agrees only at the four corner pixels.

**Fixtures (Phase 4; `m11_ids`), every output position:**

| fixture | formulation | EE vs mirror | EE vs PyTorch `UpsamplingBilinear2d` | exact checks |
|---|---|---|---:|---|
| 2 × 2 → 8 × 8, 3 × 3 → 12 × 12, 5 × 7 → 20 × 28, 1 × 5 → 4 × 20 (N = 1), 7 × 7 → 28 × 28; floats and ramps | matrix | bit-identical | ≤ 9.5e-7 | — |
| 7 → 28 (one window), 14 → 56 (2 × 2 windows), 21 → 84 (3 × 3); ± floats | window | bit-identical | ≤ 9.5e-7 | B == A bit for bit |
| same sizes, coordinate ramps (row, column) | window | bit-identical | ≤ 9.5e-7 (21: ≤ 1.9e-6) | output = the written float32 source coordinate, exactly |
| same sizes, one-hot combs at rows 0, 6, 7, N − 1 / columns 1, 7, N/2, N − 2 | window | bit-identical | ≤ 6e-8 | support of every response = PyTorch's source membership |
| **224 → 896**: hash values (server-generated), ramps, combs at rows 0, 6, 7, 111, 112, 223 / columns 1, 13, 14, 216, 222, 223 | window | bit-identical, every edge / corner / interior region | ≤ 4.8e-7; ramps exact | membership exact |
| **448 → 1792**: hash values, combs at 0, 7, 223, 224, 447 / 6, 224, 440, 446, 447 | window | bit-identical, every region | ≤ 4.8e-7 | membership exact |

**Negative controls**: a per-window 7 → 28 template differs from the global map by up to 4.6 / 4.8
(224 / 448 hash fixtures; ~0.5% of values equal, at window endpoints) — identical only for a
single-window tensor, where the window is the tensor; the 224 → 896 coefficients applied to each
224-quadrant of the 448 fixture differ from the 448 → 1792 map by up to 3.1. The 1792 map is built
from the global 448 → 1792 coordinates, not a repeated template.

## 5. EE's arithmetic and the mirror (Phase 6; Test 19)

reference/ee_mirror_head.py (archive `upernet/reference/ee_mirror_head.py` at `research-archive-2026-10-12`) extends the Milestones 7-10 mirror
(float64, rounding to float32 where EE rounds): the head conv as Milestone 2's linear; the
upsample in EE's order — stage y as a matrix product (products of float32 coefficients and
float32 values are exact in double; at most two are non-zero, so order and fusing cannot matter),
stage x written out as two products and one addition, because there the products (float32
coefficient × double) round.

**The new numerical question** is that second stage: with non-dyadic coefficients the head's
resize is the first EE operation in this project whose double intermediates are inexact, so
"double inside, one rounding" no longer pins the result — whether EE fuses multiply and add
(FMA) does. **The probe** (`m11_ids`, `double_probe`): the window resize's unrounded double
result at 7 → 28 and 14 → 56 equals the mirror with every product and every sum rounded
separately in double **on 100% of values**, and differs from the exact-FMA variant in 18-23% of
them (by ≤ 8.9e-16): **EE's `matrixMultiply` does not fuse multiply-adds.** Option B's image
arithmetic (multiply, add) is the same, which is why A == B bit for bit.

Every EE tensor of the milestone is bit-identical to the mirror: **139 / 139 comparisons, 96.5 M
elements** (head-conv and resize fixtures, the isolated head, both task-C formulations, every
shard-study export, the chain, the split, the edge and 1792 crops). The earliest
divergence from SMP is therefore never an indexing error: on the same input, EE − SMP is SMP's
float32 kernel rounding (≤ 1 ulp at the upsample, ≤ 4.8e-6 at the conv).

## 6. Tensor interpolation vs geospatial placement (Phases 7-8; Tests 12-13)

SMP's final resize aligns tensor **end-point indices**: C1 token 0 → output pixel 0 and token 223
→ pixel 895. Geospatially, C1 token 0's centre is 20 m from the crop edge and native pixel 0's is
5 m; token 223's centre is 20 m from the far edge, pixel 895's 5 m. The tensor map is *not* a
resampling between those pixel-centre grids (that is § 4's option C, which differs by up to 0.31).

The reconciliation is two separate steps, and the second moves values only:

1. **Compute the exact PyTorch output by array index** (the window resize, § 4).
2. **Assign output (row, col) to native 10 m pixel (row, col) of the input crop**:
   `depth_to_space` (window layout → 2 bands on the abstract input grid), then a `changeProj`
   relabel onto the crop's native projection (`preprocessing.to_crop_grid(s=0)`), i.e. the same
   grid the input pixels came from. Nothing is shifted or warped to make the interpolation look
   geospatially centred.

The output is a georeferenced **prediction raster on the input pixel grid**, with SMP's
semantics; it is not a geospatially centred resampling of the 40 m logits. **Downstream mapping
must not resample it again** (and must not "correct" the end-point alignment).

| check | result |
|---|---|
| in-graph logits raster and every logits asset | EPSG:32651, [10, 0, 283200.953…, 0, −10, 2645318.840…] (canonical), 896 × 896 — exactly Milestone 6's native-aligned crop grid (`crop.transform()`, the C1 grid's origin at 10 m); 1792: [10, 0, 278720.953…, 0, −10, 2649798.840…], 1792 × 1792; edge 224: [10, 0, 188480.953…, 0, −10, 2641958.840…], 224 × 224 |
| 40 random native pixel centres + the 4 corner pixels + 4 edge midpoints | **48 / 48** return logits[row, col] (both classes) — in-graph raster, two production assets, the chain asset, the 1792 asset |
| window layout vs raster | 10 m window rows 0, 64, 127 (128 windows of [49, 2] each) == the raster's rows, exact |
| masks | every band unmasked over the whole crop (mask min 1, count = H·W) in each asset checked: two production exports, the chain, the edge crop, the 1792 crop |
| band names | `logit_0`, `logit_1` = SMP's class indices 0, 1 (no class meaning assumed) |

## 7. The head on Milestone 10's fused asset (Phase 10; Tests 10, 14)

The isolated head (`m11_isolated_canonical_896`): `m10_fused_canonical_896_bb9225` (the monolithic
fused feature, 224² × 256, bit-identical to the mirror) → [49, 256] windows → head conv → × 4 →
10 m raster, interactively; **every** low-res (100,352) and final (1,605,632) logit compared:

| tensor | vs SMP's head on the same fused feature | vs float64 (same input, float32 coefficients) | vs SMP `model.forward` | vs mirror |
|---|---:|---:|---:|---|
| head conv (40 m) | 5.4e-7 (7.9% equal) | 3.0e-8 | 6.2e-6 | bit-identical |
| logits (10 m) | **4.8e-7** (11.6% equal) | 5.5e-8 | 5.5e-6 | bit-identical |
| logits, Option B | 4.8e-7 | 5.5e-8 | 5.5e-6 | bit-identical (= A) |

Edges, corners and interior behave alike (top / bottom / left / right / corner / interior regions
all bit-identical to the mirror; vs SMP the borders are smaller, ≤ 2.7e-6, than the interior).

## 8. Task C through the head: monolithic and B (Phases 11, 13, 16; Tests 11, 16, 18)

logits_pipeline.py (archive `upernet/earth_engine/logits_pipeline.py` at `research-archive-2026-10-12`) `export_logits`: C1-C3 from task A′'s
skips asset, C4 from task B′'s, → feature norms → PSP → FPN → fusion → head conv → × 4 → 10 m
logits, **one batch graph** (`m11_task_c_canonical_896`). Two validation exports, one per
fusion formulation, each writing the 10 m logits and the 40 m logits (as exact nearest copies on
the 10 m grid: every 4 × 4 native block constant, checked):

| export | formulation | shardSize | EECU·s | run s | request / invocations | values |
|---|---|---|---:|---:|---|---|
| monolithic (reference) | monolithic im2col fusion | default 256 (4 × 4 shards) | 2,642 | 1,323 | 903 KB / 565 | = mirror, = the isolated head on the monolithic fused asset |
| **B (production)** | double group partials | 896 (one shard) | 1,636 | 883 | 939 KB / 589 | = mirror, = monolithic |

**Every element, both formulations** (identical rows: the two exports are bit-identical):

| tensor | vs SMP `model.forward` (MAE / RMSE / max) | same C1-C4 input | SMP head on EE's fused | float64 (same input, float32 coeff.) | mirror |
|---|---|---:|---:|---:|---|
| head conv, 40 m (100,352) | 1.05e-6 / 1.32e-6 / **6.22e-6** | 6.0e-6 | 5.4e-7 | 3.0e-8 | bit-identical |
| **logits, 10 m (1,605,632)** | 7.8e-7 / 9.8e-7 / **5.54e-6** | 5.6e-6 | 4.8e-7 | 5.5e-8 | bit-identical |

**Monolithic vs B through the head (Phase 11), measured element by element:**

| tensor | elements | B vs monolithic |
|---|---:|---|
| pre-head fused feature | 12,845,056 | **bit-identical** — Milestone 10's two fresh task-C B exports (`m10_fresh_taskc_double_canonical_896_r1`, `_r2`: the same A′ / B′ assets and the same task-C code path) vs its monolithic milestone asset; no new 40 m fused export was made in M11 |
| 40 m logits | 100,352 | **bit-identical** (the two M11 task-C exports) |
| 10 m logits | 1,605,632 | **bit-identical** |

B is therefore equivalent to the reference through the head on this map — as measured, not as a
guarantee for other weights (Milestone 10 § 13). The batch monolithic logits also equal the
interactive isolated head (§ 7) bit for bit: the graph's batch and interactive evaluations agree.
Both assets are on the exact 10 m crop grid (4 bands each); every boundary region is
bit-identical to the mirror and no worse than the interior vs SMP.

## 9. The complete A′ → B′ → C forward and its provenance (Phases 12-13; Tests 17-18)

A fresh chain from the catalogue image (`m11_chain_canonical_896`,
full_model_pipeline.run (archive `upernet/earth_engine/full_model_pipeline.py` at `research-archive-2026-10-12`)), salted at the Sentinel-1 source
(`+ 211 × 0`: the same pixels, a graph EE has not cached) so every task recomputes:

| task | graph loads (from its serialized expression) | EECU·s | run s | request / invocations | output |
|---|---|---:|---:|---|---|
| **A′** | `COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20240103T100134_…_B0BD` + 28 encoder / patch-embedding weight tables | 312 | 236 | 843 KB / 766 | skips asset (C1, C2, C3, valid fraction; C1 grid) |
| **B′** | A′'s skips asset + 31 encoder weight tables | 61 | 147 | 360 KB / 379 | raw C4 asset (C4 grid) |
| **C** | A′'s and B′'s assets + 18 decoder / head weight tables | 2,220 | 1,189 | 934 KB / 573 | **logits asset** (`logit_0`, `logit_1`, native 10 m, shardSize 448) |

(A′ and B′ cost what Milestone 7's did: 308 and 62. Task C's 2,220 is inside the ~2× batch spread of
the production runs, 1,222-1,617.)

| check | result |
|---|---|
| inputs of each task | only the catalogue image, the previous task's asset(s) and the weight tables: **no client-side tensor enters any task** (row `task_inputs`) |
| chain C4 vs Milestone 7's task-B′ C4 | **bit-identical** (602,112 values) |
| chain logits vs the production export (task C on Milestone 7's A′ / B′, shardSize 448) | **bit-identical** (1,605,632) |
| chain logits vs SMP `model.forward` | **5.54e-6** max (MAE 7.8e-7, RMSE 9.8e-7); bit-identical to the mirror |
| grid / footprints / masks | exact crop grid; 48 / 48 footprints; 802,816 / 802,816 pixels unmasked per band |

**Provenance.** `results/raw/m11_provenance_canonical_896_chain.json` (this chain) and
`…_production.json` (Milestone 7's task A′ `ZDCXXLQFHXV73GLXHQHBFMCR` → skips
`m7_skips_canonical_896_547f14`, task B′ `KAI3GWBRZGQLGP72MYZT2EKF` → C4 `m7_c4_canonical_896_c627cf`,
production task C `RH33MBFALZUZC5CXX6MEHHOR` → logits `m11_prod_canonical_896_shard448_r0_4e3816`)
record the source image ID, the crop grid / transform / size, the preprocessing spec and mask policy,
the weight manifests (decoder namespace, head record digest), every task ID and asset ID, the assets
and tables each task's graph loads, the fusion formulation, the head configuration, the package
versions and the comparison artifacts. The chain's task IDs: A′ `ED2LSAAC7WBHIUA5BT3NKAL4`, B′
`Z4V3TIGFQUJDGR2C7E2RENMT`, C `44CWTRBDUDPCHANAZCC3WV4Z`.

**The split topology, validated.** Task C → the 40 m logits on the C1 grid (685 EECU·s) → task D →
× 4 → 10 m (5 EECU·s): its logits equal the one-graph production logits **bit for bit**
(`m11_split_canonical_896`), at ~half the cost (§ 11).

## 10. Error budget (Phase 14)

Canonical crop, every element, max |EE − reference| (EE = the mirror bit for bit at every row):

| stage | EE − SMP float32 | EE − float64 | SMP float32 − float64 |
|---|---:|---:|---:|
| C1 / C2 / C3 / C4 raw (Milestone 7) | 7.9e-6 / 8.2e-6 / 1.6e-5 / 1.1e-5 | 2.0 / 1.6 / 3.2 / 1.4e-6 | ~1e-5 |
| C4 LayerNorm2d (Milestone 8) | 5.7e-6 | 8.2e-7 | |
| PSP output (Milestone 8-9) | 1.6e-5 | 5.1e-7 | |
| F3 / F2 / F1 (Milestone 9) | 1.3 / 1.2 / 1.2e-5 | 7.4 / 9.1 / 12e-7 | |
| P3 / P2 / P1 (Milestone 9) | 2.0 / 2.0 / 2.5e-5 | 8.2 / 9.4 / 12e-7 | |
| fusion conv / fused feature (Milestone 10) | 1.9e-5 / **2.38e-5** | 5.0e-7 / 6.2e-7 | 2.4e-5 |
| **head conv (40 m)** | **6.2e-6** | 2.6e-7 | 6.3e-6 |
| **logits (10 m)** | **5.5e-6** | 5.5e-6 (own coefficients) / **2.4e-7** (float32 coefficients) | 7.1e-6 / 5.6e-6 |

1. **The head shrinks the error** (2.38e-5 → 6.2e-6, ×0.26): the seeded 256 → 2 projection
   (weights N(0, 1/256)) averages 256 channel errors of mixed sign, and the bias adds nothing
   (it is exact in the double sum).
2. **The non-dyadic coefficients cost EE nothing measurable**: on the same 40 m logits EE is ≤ 1
   ulp (4.8e-7) from SMP and 5.5e-8 from exact float64 with the same coefficients. What they do
   change is the *model*: the float32 model's coefficients differ from the float64 model's by up
   to 1.4e-5, which moves the canonical logits by up to **5.5e-6** (float64 model with its own vs
   the float32 coefficients). That — not EE — is the whole of "EE vs float64" at the logits
   (5.5e-6); with the float32 model's coefficients the float64 model is **2.4e-7** from EE.
3. **The interpolation smooths**: each output is a convex combination (weights ≥ 0, summing to
   1 ± ulp) of two × two 40 m values, so the logits' max error (5.5e-6) is below the 40 m logits'
   (6.2e-6) and their RMS drops (9.8e-7 vs 1.3e-6).
4. **EE stays far closer to float64 than SMP float32**, as in Milestones 5-10: 2.4e-7 vs 5.6e-6 at
   the logits (matched coefficients), 2.6e-7 vs 6.3e-6 at the head conv.

No intermediate exceeds 1e-4; the largest error of the whole forward is the fused feature's
(2.38e-5), and the final logits sit at 5.54e-6 (canonical, at (803, 237, class 0); 1792: 5.93e-6 at
(180, 1595, class 1)) — the same maximum in every complete canonical run (task C both formulations,
production, chain, split; `results/raw/m11_final_report_canonical_896.json`).

## 11. Cost, constants and shardSize (Phases 15, 17-18; Tests 20-22)

**The head itself is cheap; writing at 10 m is not.** The head conv, the resize and the 10 m raster
add ~2-3 EECU·s to reading their input, and the isolated head exported to the 10 m grid costs 25
EECU·s; but task C exported at 10 m costs **1,222-1,636 EECU·s** with the best shard settings
(2,642 in the one monolithic default-shard run) — ~2× Milestone 10's task C at 40 m (669-693 for
B) — because the decoder is re-evaluated inside 10 m output tiles. Splitting the work at the 40 m
logits removes that: a task C exporting the 40 m logits (one C1 tile, like Milestone 10) costs 685
and a task D doing only the × 4 resize costs 5.

**Head prefixes** (`benchmark_head`, interactive, the whole canonical map from Milestone 10's
fused asset, each request salted; two runs):

| prefix | EECU·s | wall s | peak MiB | embedded |
|---|---:|---:|---:|---:|
| read fused asset → [49, 256] windows | 5.7 / 5.6 | 3.0 / 1.8 | 45 | 0 |
| + head conv 256 → 2 (bias) | 5.0 / 6.0 | 1.9 / 3.9 | 45 | 610 |
| + × 4 align_corners=True (window) | 6.4 / 7.6 | 4.1 / 5.0 | 105 | 16,994 |
| + depth_to_space + relabel → 10 m raster | 7.5 / 9.2 | 3.1 / 4.9 | 105 | 16,994 |

The resize's per-child-window coefficient rows are the peak (105 MiB); everything stays far below
the interactive limits.

**Batch exports (canonical unless noted):**

| export | what | shardSize (shards, derived) | EECU·s | run s | request / invocations |
|---|---|---|---:|---:|---|
| isolated head | fused asset → head → 10 m logits | default (4 × 4) / 896 (1) | 24.9 / 24.9 | 34 / 46 | 125 KB / 95 |
| **production task C (B)** | assets → … → 10 m logits (2 bands) | **448 (2 × 2)** | **1,271 / 1,222** | 692 / 379 | 935 KB / 574 |
| production task C (B) | same | 896 (1) | 1,602 / 1,617 | 388 / 337 | 935 KB / 574 |
| production task C (B) | same | default 256 (4 × 4) | 1,351 | 2,075 | 935 KB / 574 |
| validation task C (B) | + 40 m copies (4 bands) | 896 (1) | 1,636 | 883 | 939 KB / 589 |
| validation task C (monolithic) | + 40 m copies (4 bands) | default 256 (4 × 4) | 2,642 | 1,323 | 903 KB / 565 |
| split: task C | assets → … → 40 m logits on the C1 grid | default (one 224 tile) | **685** | 575 | 848 KB / 558 |
| split: task D | 40 m logits asset → × 4 → 10 m logits | 896 (1) | **5.2** | 44 | 115 KB / 93 |
| edge 224, task C (B) | + 40 m copies | default (1) | 54.9 | 75 | 600 KB / 588 |
| 1792, task C (B) | + 40 m copies | 896 (2 × 2) | 7,102 | 768 | 1.82 MB / 589 |

Shard counts are derived from the output size (896 px / 256 → 4 × 4 with partial shards; 448 → 2 × 2;
896 → 1); EE does not report them. Every canonical task-C export — five production exports (448, 896, default; 448 and 896 repeated)
and the two validation exports (B and monolithic); eight with the chain, nine with the split (§ 9)
— is identical bit for bit, every logit, so
`shardSize`, salts and the fusion formulation change cost, never values.

**shardSize at the 10 m output scale (Phase 18).** Measured on the production graph: **448 px (2 ×
2 shards) is the cheapest, 1,222-1,271 EECU·s in two runs**; one 896-px shard costs ~25% more
(1,602-1,617, but the shortest run, 337-388 s); the default 256 px is in between in EECU·s (1,351)
but takes 2,075 s, 3-6× longer. Recommendation: **`shardSize` = half the tile side** (448 for an
896 tile, 896 for 1792: window-aligned, 2 × 2 shards). Milestone 10's C1-scale result (one shard best
at 448²) does not transfer: at 10 m the shard is not the unit of upstream reuse.

**Why 10 m costs ~2× (Phase 15).** The isolated head at 10 m (25), the split's task C at 40 m (685
= Milestone 10's 669-693) and task D (5) together cost 715; the same work in one graph exporting
10 m pixels costs 1,222-1,617. Consistent explanation (not observable in EE's records): each 10 m
evaluation tile covers only part of the C1 window grid, and the decoder graph above it (resized
halos, im2col, the 9,216-term fusion products, the global PSP) is evaluated again for every
tile's footprint plus its halo, ~2× in total. The 40 m export evaluates the whole 224² C1 map in
one tile. **This is a measured reason for a task boundary at the 40 m logits — not adopted as
the M11 deliverable**, which is the one-graph task C the brief specifies (it runs, reliably, at
canonical and 1792); the split is bit-identical to it (§ 9) and recommended for production cost in
Milestone 12.

**Fraction of task C (Q25).** Of a production task C (1,222-1,271 EECU·s at 448): the head conv +
resize + raster themselves ≈ 2-3 EECU·s (0.2%); the head + 10 m write including its input read, 25
(2%); the 10 m evaluation geometry ≈ 530-930 (~45%); the rest is Milestone 10's decoder (685).

**Constants and graph (Phase 17).** Counted from the constructors:

| component | embedded values |
|---|---:|
| task C through fusion (Milestones 8-10) | 9,856,114 |
| head conv Wᵀ [256, 2] + bias rows 49 × 2 | 610 |
| × 4 coefficient tables, 224 → 896 (2 axes × ([896, 9] stack + 128 child indices)) | 16,384 |
| **task C total, canonical** | **9,873,108 (41.1% of ~24 M)** |
| task C total, 1792 (tables 2 × (1,792 × 9 + 256) = 32,768) | 10,010,196 (41.7%) |
| split task D (tables only) | 16,384 |

The coefficient tables grow linearly with the tile side (O(H + W), not O(H W)): 16,384 at 896, 32,768
at 1792 — negligible against the cap and the graph (production request 935 KB, 574 invocations; 1792:
1.82 MB). No new weight upload: the head's 514 values come from Milestone 7's asset record.

## 12. Swath edge, validity and QC (Phases 9, 19-20; Test 15)

The swath-edge crop `edge_224` (~50% valid; Milestone 6's training-mean fill) through task C
(B) in batch (`m11_edge_edge_224`, 54.9 EECU·s): logits within **5.0e-6** of SMP's `model.forward`
on the same filled input (40 m logits 5.8e-6), bit-identical to the mirror, every band of the
224 × 224 asset unmasked (50,176 / 50,176 pixels): no EE mask reached attention, pooling,
convolution or the interpolation — the network sees the dense filled input exactly as PyTorch
does, and the final interpolation reads only the finite tile (edges and corners included).

**QC stays separate.** The QC asset (`m11_qc_edge_224`, 0.2 EECU·s) holds `valid_fraction` (task
A′'s per-C1-token valid fraction as exact nearest copies on the 10 m grid: **equal to Milestone 6's
per-token fraction at every token**, mean 0.495) and the source's `valid_VV`, `valid_VH`; it is a
separate asset on the same 10 m grid, never a logit channel, and nothing in it alters a logit.
`argmax`, softmax and thresholds are post-processing outside `model.forward` (Identity) and are
not computed here: with seeded weights they would only suggest a skill the network does not have.

## 13. The 1792 crop (Phase 19; Test 23)

Task C (B) for the 1792 crop (`m11_large_large_1792`; C1-C3 from Milestone 9's task-A′ asset,
C4 = Milestone 6's): C1 fused 448² × 256 → 40 m logits 448² × 2 → **1792 × 1792 × 2 logits**,
**one batch graph**, `shardSize=896` (2 × 2 shards, the canonical recommendation's geometry):

| item | result |
|---|---|
| export | **success**: 7,102 EECU·s, 768 s run (422 s queued), 1.82 MB request; 4 bands (logits + 40 m copies) |
| logits vs SMP `model.forward` (every one of 6,422,528) | **5.93e-6** max (interior; borders ≤ 2.8e-6, corners 1.0e-6) |
| 40 m logits (401,408) | 6.59e-6 |
| vs SMP's head on EE's fused feature | 4.8e-7 (logits), 4.8e-7 (40 m) |
| vs float64 | 2.65e-5 (own coefficients: 448 → 1792's float32 coefficients deviate by up to 2.7e-5) / 5.6e-8 (same coefficients) |
| vs mirror | **bit-identical** (logits and 40 m logits) |
| coefficients | the 448 → 1792 map, built from the global 448 → 1792 coordinates: the 224 → 896 coefficients applied per 224-quadrant differ from the EE output by up to 0.225 (negative control) |
| grid / footprints / masks | EPSG:32651, [10, 0, 278720.953…, 0, −10, 2649798.840…], 1792 × 1792, every band; 48 / 48 footprints; 3,211,264 / 3,211,264 pixels unmasked in every band |
| checksums | Σ logits = 603,787.92, Σ logits² = 142,059.1 |

**Scaling.** 7,102 EECU·s is 4.4-5.8× the canonical one-graph export (1,222-1,636) for 4× the
pixels, and 2.3× Milestone 10's best 1792 task C at 40 m (3,082, one 448-px C1 shard): the same
10 m factor as at canonical (§ 11). No new EE limit appeared: the full-resolution two-band
output, its 1.82 MB graph and the 2 × 2 shards ran first time. Batch only; the brief's
interactive limit was not tested at this size.

## 14. Earth Engine behaviour found or confirmed

1. **`matrixMultiply` does not fuse multiply-adds**: its double products are rounded before the
   (sequential) sum, on arrays and on array images (§ 5). Combined with Milestone 10's finding
   that results stay double until `toFloat`, EE's arithmetic is now pinned down even where the
   intermediates are inexact.
2. **EE's `resample("bilinear")` is pixel-centre interpolation** (= `align_corners=False` within
   one ulp, edge clamp by mask renormalization or ring); it cannot express `align_corners=True`.
3. **Export cost depends on the output grid's scale, not just shardSize**: the same task C costs
   ~2× when it writes 10 m pixels instead of 40 m ones (1,222-1,617 vs 685 EECU·s), while the head
   itself costs 25; at 10 m, 2 × 2 shards (448 px) beat one shard and the default 256 (§ 11).
4. Concurrent exports into a new asset folder race on its creation ("Cannot overwrite asset");
   `export_image` now tolerates a folder created meanwhile.

## 15. What is proven, what is not, and Milestone 12

**Proven, precisely.** For the seeded Swin-T + SMP UPerNet (smp 0.5.0, timm 1.0.30) on one
Sentinel-1 scene, Earth Engine computes the complete model forward — SegWater v3 preprocessing,
patch embedding, four Swin stages, LayerNorm2d, PSP, the installed FPN recursion, the multi-scale
fusion, the biased segmentation head and its `align_corners=True` × 4 upsampling — in native EE
graphs, across three chained batch tasks that read only the catalogue image, the previous task's
asset and the weight tables; the result is a 10 m two-band raster on the input crop's own grid
whose every value is within 5.5e-6 (canonical) / 5.9e-6 (1792) of the installed PyTorch
`model.forward` in float32, and whose every intermediate equals a float64 model of EE's own
roundings bit for bit. The numerics are understood to the last bit: EE rounds once per PyTorch
operation, keeps matrix products in double until `toFloat`, rounds products and sums separately
(no FMA), and the remaining distance to SMP is SMP's own float32 accumulation.

**Not proven.** Trained-model behaviour: these are seeded weights; nothing here says the network
segments water, what class index 1 means, how the logits are calibrated, or how a trained
checkpoint's values interact with formulation B's (measured, not guaranteed) equality. Other
scenes, orbits, seasons and terrain; tiles of other sizes (the model is the finite tile — Milestone
8's Option A — so tile size and PSP pooling matter); geographic mosaicking of tiles (overlap,
seams, the end-point alignment at tile edges); production-scale cost and throughput; float64
deployments (different `align_corners` coefficients).

**Final decision.** *Can a real Sentinel-1 image be passed through the entire Swin-T + SMP UPerNet
model inside Google Earth Engine, across native EE tasks only, producing a georeferenced
two-channel segmentation-logit raster numerically equivalent to the installed PyTorch model?*
**Yes** — at canonical size in three chained tasks (A′ → B′ → C), within 5.54e-6 of the installed
`model.forward` over every one of 1,605,632 logits (target 1e-4), on the exact native 10 m grid, with
reproducible provenance; and at 1792² within 5.93e-6.

**Recommendation: proceed to Milestone 12 — trained-checkpoint ingestion and real segmentation
evaluation**, carrying forward:

```text
trained Swin-T + exact SMP UPerNet checkpoint (same smp / timm versions or a re-validated contract)
correct input preprocessing and model metadata (SegWater v3 statistics, band order, tile size,
    mask policy, img_size)
bit-exact mapping of every checkpoint tensor into EE assets (SHA-256 per record, read-back tests)
verification of class-index / logit meaning (which channel is water; softmax / argmax as
    post-processing only)
full-forward numerical regression on trained weights (this milestone's references and mirror; the
    monolithic fusion as the check on formulation B)
segmentation accuracy / transfer evaluation on held-out sites
runtime and geospatial production behaviour (the split C → 40 m / D → 10 m topology and
    shardSize = half the tile side; tiling, overlap and mosaicking without a second resample)
```

Do not read the seeded logits as flood or water predictions.

## 16. Caveats

- **Seeded weights.** The encoder, decoder and head are the deterministic seeded test weights of
  Milestones 5-7; execution fidelity is proven for them. The logits are small (|z| < 0.86) and
  carry no water-segmentation skill; class index 1 has no established meaning.
- **One scene.** Sun Moon Lake crops of one Sentinel-1 scene (edge 224, canonical 896, 1792).
- **Coefficients.** EE reproduces the float32 model's `align_corners=True` coefficients; a float64
  deployment of the same checkpoint would use slightly different ones (≤ 2.7e-5 at 448 → 1792).
- **Formulation B.** Its equality with the monolithic fusion after BN / ReLU — and therefore
  through the head — is measured on these maps, not guaranteed for a trained checkpoint (Milestone
  10, § 13); the monolithic path remains runnable as the reference.
- **Variance.** Batch EECU·s of identical graphs vary up to ~2× (Milestone 10); cost conclusions are
  drawn on ranges.
- **Assets** remain in `ORIGINAL_PROJECT`: `upernet/logits/m11_*` (validation, production, shard
  study, chain, 1792, QC) and `upernet/features/m11_chain_*`; delete the shard-study and
  isolated-benchmark exports when done.

## Tests → evidence

| test | where |
|---|---|
| 1 installed head contract | `test_upernet_head_contract.py`; `m11_contract` rows |
| 2 reference equivalence | `test_capture_and_expanded_forward_agree`, `test_real_forward_expanded_equals_model_forward`; `expanded_equals_model_forward_*` rows |
| 3 head parameter fidelity | `test_head_parameters_and_asset_record`; `m11_weights` |
| 4 channel / bias IDs | `test_class_and_channel_ids_are_exact`; `class_channel_ids`, `onehot_real_weights` |
| 5 head conv numerics | `test_head_conv_mirror_matches_pytorch`; `random_*_bias`, `bias_rounding_probe` |
| 6 exact align_corners mapping | `test_kernel_equals_written_float32_formula` (N = 1, M = 1 incl.); matrix fixtures |
| 7 global coordinates | `test_window_matrices_are_rows_of_the_global_map`; `hash_224_to_896`, per-window controls |
| 8 window boundaries | `test_one_token_halo_suffices`; combs at 6 \| 7, 111 \| 112, 223 \| 224 |
| 9 native EE resampling | `m11_native_canonical_896` (rejected) |
| 10 head on the real fused asset | `m11_isolated_canonical_896` |
| 11 monolithic vs B | `m11_task_c_canonical_896` `B_vs_monolithic` |
| 12 output grid | projection / footprint / asset-grid rows |
| 13 10 m output layout | window-row vs raster rows, footprints |
| 14 edge / corner interpolation | boundary rows (every experiment), corner footprints |
| 15 swath-edge masks | `m11_edge_edge_224` |
| 16 whole task C | `m11_task_c_canonical_896`, `m11_shards_canonical_896` |
| 17 A′ → B′ → C provenance | `m11_chain_canonical_896`, `m11_provenance_canonical_896_*.json` |
| 18 full model numeric check | every logits row vs `torch_float32` (captured `model.forward`) |
| 19 mirror | every `ee_mirror` row; `double_probe` |
| 20 budget | `test_upernet_m11_budget.py`; `m11_budget_canonical_896` |
| 21 repeat export | `m11_shards_canonical_896` `repeat` rows |
| 22 output-scale shard study | `m11_shards_canonical_896`, benchmark rows |
| 23 larger crop | `m11_large_large_1792` |
| 24 no trained-quality claim | `test_seeded_weights_are_labelled_in_exports`; export records' `seeded_weights` field; § 12, § 16 |
