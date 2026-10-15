# Findings: the complete Swin-T encoder in native Earth Engine

Live experiments on 4-5 October 2026: Earth Engine Python API 1.7.45, PyTorch
2.14.0, timm 1.0.30, NumPy 2.5.3, Cloud project `ORIGINAL_PROJECT`,
application-default credentials. Every forward pass ran entirely in Earth Engine
computation graphs: no Vertex AI, Cloud Run, model server or client-side tensor
computation. All 27,514,458 encoder parameters are read from EE table assets
inside the graph. EE-computed tensors were fetched only for comparison.

Evidence: numerical_comparison_m5.csv (development archive: `docs/history/swin_encoder/results/numerical_comparison_m5.csv`)
(summary: numerical_summary_m5.md (development archive: `docs/history/swin_encoder/results/numerical_summary_m5.md`)),
benchmark_results_m5.csv (development archive: `docs/history/swin_encoder/results/benchmark_results_m5.csv`) (summary:
benchmark_summary_m5.md (development archive: `docs/history/swin_encoder/results/benchmark_summary_m5.md`)), per-request
records with EE profiles in results/raw/ (archive `swin_encoder/results/raw/` at `research-archive-2026-10-12`) (`bench/`,
`validation_*`, `asset_read.json`, `graph_size.json`,
`probe_constant_limit.json`, `stage3_assets.json`), configuration and digests in
data/manifest.json (archive `swin_encoder/data/manifest.json` at `research-archive-2026-10-12`), the weight assets in
data/weights_manifest.json (archive `swin_encoder/data/weights_manifest.json` at `research-archive-2026-10-12`).

**Research question.** Can a complete modern hierarchical Swin-T encoder execute
entirely inside the Google Earth Engine computation graph while preserving
timm/PyTorch numerical behavior? **Yes, as two chained Earth Engine batch tasks.**
It cannot run as one graph.

- The complete encoder (stage 1 → PM 1 → stage 2 → PM 2 → six-block stage 3 →
  PM 3 → stage 4 → final LayerNorm) matches timm's `model.norm(model.layers(x))`
  on the canonical 224 × 224 stage-1 tokens to **5.48e-6 at every one of the
  602,112 output elements** (MAE 6.7e-7; target 1e-4). On 448 × 448 tokens the
  figure is 5.96e-6 (2.4 M elements).
- Every compared tensor is **bit-identical** to a float64 mirror that rounds where
  EE rounds and sums in EE's order. The PyTorch difference is all PyTorch's
  float32 rounding: at the canonical output EE is 8× closer to float64 than
  PyTorch is (7.0e-7 vs 5.7e-6).
- **One graph is impossible in batch**: EE batch embeds every evaluated constant
  (each asset-read weight array) in the image object it ships to workers, and
  caps that object at 24-26 M values, independent of the element type. Swin-T's
  weights alone are 27.4 M values (30.9 M with the row-repeated vectors). The
  single graph fails with "Encoded object too large." at any raster size. Split
  at the stage-3 output, it runs as task A (stages 1-3 → EE image asset, 305-363
  EECU·s) and task B (asset → PM 3 → stage 4 → norm, 53-68 EECU·s).
- **Interactive requests are bounded by the weights, not the raster**: decoded
  weights take ~59 bytes each of EE memory (1,641 MiB for the encoder), so the
  whole encoder exceeds the interactive limit even on an 8 × 8-window raster.
  With per-block extension each half of the canonical encoder fits one
  interactive request (head 1,061 MiB, tail 979 MiB), including the stage-1
  pair that Milestones 3-4 could not run interactively.
- **Stage 3 dominates compute; Milestone 4's per-stage extension rule does not
  scale to it.** A six-block stage needs a three-window ring. Re-extending
  before every SW-MSA block (per-block) keeps memory flat in depth and costs
  0.41-0.84× the EECU·s of per-stage.

**Milestone verdict.** Minimum, strong and full architectural success are all
met. **Full success is as a two-task native Earth Engine workflow** with one
persisted EE intermediate (the stage-3 output asset). That split is necessary
because of the platform's embedded-constant cap; it is not a computation or
memory boundary. Nothing leaves Earth Engine. One batch graph (Q29) is not
achievable with asset-read weights.

## Answers

| # | Question | Answer |
|---|---|---|
| 1 | Can Patch Merge 192 → 384 execute correctly in native EE? | **Yes.** Layout exact at every element on unique (token, channel) IDs at full width (16 × 16 → 8 × 8 stage-2 windows, 4.8 M elements). LayerNorm(768) and the 768 → 384 reduction are bit-identical to the mirror: 6.7e-6 vs PyTorch on the canonical grid, 5.0e-6 fed PyTorch's own inputs (§2). |
| 2 | Can Patch Merge 384 → 768 execute correctly? | **Yes.** Layout exact (8 × 8 → 4 × 4 stage-3 windows, 2.4 M elements). LayerNorm(1536) and the 1536 → 768 reduction are bit-identical to the mirror: 7.6e-6 on the canonical grid (§2). |
| 3 | Does the same Patch Merge implementation serve all three transitions? | **Yes, one function**, unchanged from Milestone 4, its width read from the parameters. Its layout was tested with full-width channel IDs at C = 96, 192 and 384 (§2). |
| 4 | Can dim = 384 / 12-head attention run correctly? | **Yes.** Every operation, fed PyTorch's own inputs, is bit-identical to the mirror: max 6.7e-6 (QKV), 7.6e-6 at masked logits (§3). |
| 5 | Can dim = 768 / 24-head attention run correctly? | **Yes.** Every operation is bit-identical to the mirror: max 1.05e-5 (QKV), 3.6e-6 attention × V (§3). |
| 6 | Can MLP widths 1536 and 3072 run correctly? | **Yes.** fc1 / GELU / fc2 bit-identical: stage 3 4.5e-6 / 4.8e-7 / 4.5e-6, stage 4 6.2e-6 / 4.8e-7 / 7.2e-6 vs PyTorch (§3). |
| 7 | Does the complete six-block Stage 3 match timm? | **Yes.** Every intermediate of all six blocks on 2 × 3 windows (132 tensors, 20.8 M elements) is bit-identical to the mirror, max 2.3e-5 at masked logits. On the canonical 8 × 8 the block outputs after 1, 2, 4 and 6 blocks are within 7.2e-6, 1.0e-5, 1.7e-5 and 2.1e-5, in both extension modes, and the stage verifies in batch (§3). |
| 8 | Does Stage 4 match timm? | **Yes.** Every intermediate of both blocks on 2 × 2 windows (all four mask types): bit-identical, output within 1.3e-5. Canonical 4 × 4: 1.3e-5, bit-identical (§3). |
| 9 | Does final LayerNorm match timm? | **Yes.** LayerNorm(768) alone: 7.2e-7, bit-identical; at the end of the canonical encoder 5.5e-6 (upstream error, renormalized) (§1, §3). |
| 10 | Does the entire four-stage encoder match timm? | **Yes.** Canonical 224 × 224 → 28 × 28: every element within 5.48e-6, bit-identical to the mirror in both extension modes. 8 × 8 and 16 × 16 windows: every compared intermediate of every block bit-identical, final output within 4.3e-6 and 5.0e-6. 64 × 64 windows: every element within 5.96e-6. The whole encoder always runs as head + tail over the stage-3 EE asset (§1, §4). |
| 11 | What is the final max absolute error? | **5.48e-6** (canonical, 602,112 elements; MAE 6.68e-7, RMSE 8.62e-7). EE − float64: 7.0e-7 (§1). |
| 12 | Does error materially accumulate with depth? | **No.** Max error vs PyTorch stays ~1e-5 at every boundary (stage-3 output 1.6e-5 after 10 blocks, stage 4 1.2e-5, final norm 5.5e-6). Patch Merging and the final LayerNorm renormalize the residual stream. At every boundary EE stays within PyTorch's own distance from float64 (§1). |
| 13 | Does EE remain bit-identical to the controlled-rounding mirror? | **Yes, everywhere: 661 of 661 compared tensors, 165.6 M elements**, once the mirror sums in EE's order. Milestone 2's mirror summed LayerNorm rows with torch's vectorized sum; at one of 2.4 M LayerNorm(768) elements that differs from EE's strictly left-to-right sum by one float32 ulp (1.8e-12). The Milestone 5 mirror sums sequentially (§1, §11). |
| 14 | Which stage dominates memory? | **Stage 1 for activations, stage 4 for parameters.** At the canonical size stage 1 has the largest per-component peak (1,080-1,460 MiB, 1,024 windows); its pair fits interactively only with per-block extension. Stage 4's activations are the smallest (16 windows), but its 15.4 M decoded weights alone take 904 MiB, as much as the larger activation peaks (§6). |
| 15 | Which stage dominates compute? | **Stage 3**: 244-301 EECU·s for its six blocks (interactive, canonical grid), against 78-93 for stage 2, 80-159 for stage 1 (batch) and ~78-105 for stage 4. Each block has the same linear-layer FLOPs at every stage (4× fewer windows at 2× the width), and stage 3 has six. Patch Merging (8-14 each) and the final norm (2) are minor (§6, §7). |
| 16 | Does Stage 3 depth introduce a new graph/memory limitation? | **Not with per-block extension**: memory is flat in depth (666, 509, 666, 734 MiB at 1, 2, 4, 6 blocks), EECU·s and the graph grow linearly (~45-50 EECU·s and ~520 invocations per block), and there is no complexity error or recomputation. **With per-stage extension, yes**: its three-window ring pushes stage 3 over the interactive limit from depth 4 (§7, §8). |
| 17 | Does Stage 4's high width introduce a new limitation? | **Through its parameters, not its activations.** Stage 4 holds 56% of the parameters (15.4 M): 904 MiB decoded, and the weights that push the single graph over the batch constant cap. Its activations are the smallest of any stage, and the stage-4 pair fits interactively with per-block extension (907 MiB) (§4, §5, §7). |
| 18 | How does memory change as spatial resolution drops and width rises? | **Activation memory per tensor halves per stage** (windows ÷ 4, width × 2): 4.8 M, 2.4 M, 1.2 M, 0.6 M values per [49, C] tensor at the canonical size. **Parameter memory grows ~4× per stage** (width²): 0.2 M, 1.0 M, 11.0 M, 15.4 M. By stage 3 the weights outweigh the activations (§6). |
| 19 | Does window layout remain the preferred representation through all stages? | **Yes.** All four stages and the three merges stay on coarse window grids (stage 4 is a 4 × 4-pixel raster); no fine token raster is formed anywhere (§2). |
| 20 | Are asset-backed weights still practical at full encoder size? | **Yes for storage and reading; they shape the execution.** Upload 55 min (57 tasks, 389 MB), all 27.5 M values bit-exact. Reading costs 43 EECU·s / 14 s per request. But decoded weights cost ~59 B each (interactive limit), and as evaluated constants they hit the batch embedding cap (§4, §5). |
| 21 | What is total asset-read overhead? | **43.4 EECU·s, 13.8 s, 1,641 MiB reported peak** for all 27.5 M parameters in one request (~1.6 EECU·s per million). Per stage: 1.1 / 3.0 / 19.0 / 17.3 EECU·s. Every head request reads ~12.4 M (~22 EECU·s), every tail request ~16.6 M (~19 EECU·s), included in the costs above (§5). |
| 22 | What is the serialized graph size? | **916 KB, 6,980 invocations, 69 distinct functions** for the single graph; head 657 KB, tail 323 KB. Constant in raster size (§10). |
| 23 | Does Earth Engine hit graph-complexity limits? | **No "Computation is too complex"** anywhere. The limit hit is a different one, the batch cap on embedded constants ("Encoded object too large."), measured at 24 M pass / 26 M fail (§4). |
| 24 | Which materialization points are beneficial? | **None beyond Milestones 3-4's.** Extra pins at every stage output (+13-64% EECU·s) or block output (+29-88%) only cost. Separate `reduceRegion`s in one task do not share work (boundary checksums ≈ 4× one head pass). The stage-3 boundary is persisted as an asset only because of the cap (§8). |
| 25 | Does per-stage periodic extension remain best for batch? | **No: per-block is better for batch and interactive.** Batch EECU·s (head): 83 vs 181 at 16 × 16 windows, 229 / 286 vs 363 / 382 / 555 at 32 × 32 (0.41-0.84×); propagated costs 472 at 16 × 16. Interactively per-block fits the stage-1 pair, the stage-4 pair, the canonical head and the canonical tail; per-stage fits none of them. The reason is stage 3's three-window per-stage ring; per-block computes no ring (§6, §8). |
| 26 | What is the complete encoder's Stage-1 dependency halo? | **23 stage-1 windows = 161 stage-1 tokens** on each side (1 + 2 + 3 × 4 + 8; 15 when stage 4 is a single window). Verified: propagated head rings of 14 vs 15 windows give an entirely masked vs a complete, correct stage-3 output. Every per-stage ring one window short (1/1/1/1, 1/1/2/1, 0/1/3/1, 1/0/3/1, stage-4 0) masks output windows; 1/1/3/1 is complete (§9). |
| 27 | Is interactive chunking still practical? | **No.** The halo (23 windows) covers the canonical raster (32), so every chunk computes the whole raster. Measured on the head (16 × 16 windows, four 8 × 8 chunks): every chunk correct, combined checksums exact, but no memory benefit. And per-block extension already fits each half of the canonical encoder in one interactive request (§6, §9). |
| 28 | What is its EECU overhead relative to batch? | **4.1× the whole-raster interactive request and 12× batch** (1,024 vs 252 vs 83 EECU·s for the head on 16 × 16 windows in four chunks). For the full encoder on larger rasters the stage-1 recompute factor is ((n + 46)/n)²: 2.95× for 64-window chunks, 15× for 16-window chunks (§9). |
| 29 | Can the canonical 224 × 224 → 28 × 28 encoder run as one batch graph? | **No.** The single graph embeds 30.9 M constant values (27.4 M of them weights), 5-7 M over the 24-26 M cap, and fails at any raster size. **As two chained batch tasks, yes**: 305 + 53 EECU·s (per-block), verified at every output element (§4). |
| 30 | What is the largest encoder input successfully tested? | **448 × 448 stage-1 tokens (64 × 64 windows) → 56 × 56 × 768**, two tasks, 930 + 141 EECU·s, every element verified (5.96e-6) (§1, §4). |
| 31 | Based on the evidence, should we proceed to M6? | **Yes**, with the two-task batch pipeline, per-block extension, asset-backed weights and the mirror kept for every new operation (§12). |


## 1. Numerical equivalence (Phases 1, 13-15, 24-25)

**Reference.** timm's own `SwinTransformer(embed_dim=96, depths=(2, 2, 6, 2),
num_heads=(3, 6, 12, 24), window_size=7, mlp_ratio=4)`, `fused_attn=False` on
every block, from stage-1 tokens to `model.norm(model.layers(x))`
(`forward_features` without the patch embedding). The explicit forward
(Milestone 2's W-MSA, Milestone 3's SW-MSA, Milestone 4's Patch Merging) exposes
every intermediate. It is asserted bit-identical, in float32 and float64, to
every timm block's output, to every `layers[s]`, and to
`model.norm(model.layers(x))`, so every compared tensor is timm's own.
Milestone 4's five parameter groups keep their seeds, and M5's first two stages
reproduce all 96 Milestone 4 reference tensors bit for bit. Stages 3-4, Patch
Merging 2-3 and the final norm use the same deliberately non-trivial spec,
generic in width (LayerNorm γ/β ≠ 1/0, qkv scale ∝ 1/√dim, so logits keep a
standard deviation of ~2.5 at every width). Inputs are Milestone 2's generated
tokens, **not periodic and masked outside the raster**.

**The final encoder output vs PyTorch float32** (every element):

| case | execution | elements | MAE | RMSE | max \|EE − torch32\| | max \|EE − float64\| | max \|torch32 − float64\| | EE vs mirror |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 8 × 8 → 1 × 1 windows (stage 4 unshifted) | head + tail, interactive | 37,632 | 6.8e-7 | 8.7e-7 | 4.3e-6 | 5.4e-7 | 4.4e-6 | bit-identical |
| 16 × 16 → 2 × 2 (every stage shifts) | head + tail, interactive | 150,528 | 6.8e-7 | 8.7e-7 | 5.0e-6 | 6.5e-7 | 5.3e-6 | bit-identical |
| **32 × 32 → 4 × 4 (224² → 28² tokens)** | **two batch tasks** | **602,112** | **6.7e-7** | **8.6e-7** | **5.48e-6** | **7.0e-7** | **5.7e-6** | **bit-identical** |
| 64 × 64 → 8 × 8 (448² → 56²) | two batch tasks | 2,408,448 | 6.6e-7 | 8.6e-7 | 5.96e-6 | 7.8e-7 | 6.0e-6 | bit-identical |

**Error through the encoder (Phase 25)**, max |EE − torch32| at each boundary
(every element; the canonical stage-1/2 boundaries by batch checksums, all
passing at relative ≤ 2.9e-8):

| boundary | 8 × 8 windows | 16 × 16 | 32 × 32 (canonical) | 64 × 64 |
|---|---:|---:|---:|---:|
| stage 1 output | 9.2e-6 | 9.2e-6 | checksum ✓ | — |
| Patch Merging 1 | 4.1e-6 | 4.8e-6 | checksum ✓ | — |
| stage 2 output | 1.0e-5 | 1.1e-5 | checksum ✓ | — |
| Patch Merging 2 | 6.0e-6 | 6.3e-6 | checksum ✓ | — |
| stage 3, 2 blocks | 9.5e-6 | 1.0e-5 | checksum ✓ | — |
| stage 3, 4 blocks | 1.3e-5 | 1.4e-5 | checksum ✓ | — |
| stage 3 output (10 blocks in) | 1.3e-5 | 1.6e-5 | **1.64e-5** | 1.72e-5 |
| Patch Merging 3 | 7.9e-6 | 7.9e-6 | checksum ✓ | — |
| stage 4 output | 1.1e-5 | 1.3e-5 | **1.24e-5** | 1.43e-5 |
| **final norm** | **4.3e-6** | **5.0e-6** | **5.48e-6** | **5.96e-6** |

**Divergence does not accumulate materially with depth.** The residual stream
grows within a stage (its standard deviation reaches ~4 after stage 3's six
blocks) and so does the absolute float32 rounding of PyTorch's kernels. But each
Patch Merging (LayerNorm + projection) resets the stream to unit scale, and the
final LayerNorm divides by it. At every boundary EE lies closer to float64 than
PyTorch does: 3.4e-6 vs 1.7e-5 at the canonical stage-3 output, 7.0e-7 vs 5.7e-6
at the output. The largest difference at any compared intermediate is 2.3e-5, at
stage-3 masked logits near −100 (three float32 ulps of 100); masked
probabilities are below 1e-38 and do not propagate.

**EE vs its specification.** Every compared tensor is bit-identical to the
mirror: **661 of 661 tensors, 165,607,456 elements**, including the canonical
outputs of both extension modes and 448 × 448. The layout tests (9.4 M ID
elements) and the 27.5 M weight values are exact as well. One correction was
needed. Milestone 2's mirror sums LayerNorm rows with torch's vectorized `sum`; EE's `arrayReduce`
adds strictly left to right (established live, §11). Both agree after float32
rounding except at rare near-ties, and one appeared here for the first time: one
of 2,408,448 elements of LayerNorm(768) in Patch Merging 2, off by one float32
ulp (1.8e-12 at a value of 2.7e-5). EE's value equals the sequential-sum
emulation exactly. The Milestone 5 mirror therefore sums in EE's order
(LayerNorm mean and variance, softmax denominators) and is bit-identical
everywhere. torch's float64 matmul already accumulates sequentially, so the
products needed no change. Milestones 2-4's mirrors are unchanged (the
sequential mode is opt-in).

**Test status (Tests 1-17).** Offline: 52 tests (`test_patch_merge_wide`,
`test_stage3`, `test_stage4`, `test_full_encoder`, `test_encoder_weights`) pass;
Milestones 2-4's 58 offline tests still pass. Live
(tests/test_swin_encoder_live.py (archive `tests/test_swin_encoder_live.py` at `research-archive-2026-10-12`), `EE_LIVE=1
EE_LIVE_BATCH=1`): **17 of 17 pass** (17 min). Test 17 runs a fresh canonical head
export and the tail as two batch tasks, verified against float64.

## 2. Patch Merging 192 → 384 and 384 → 768 (Phases 2-4)

**One implementation.** Milestone 4's `patch_merge` is used unchanged at all
three transitions; it reads C from the LayerNorm width (4C). timm's semantics
are asserted for the wider modules (tests/test_patch_merge_wide.py (archive `tests/test_patch_merge_wide.py` at `research-archive-2026-10-12`)):
LayerNorm(768) / LayerNorm(1536), eps 1e-5; `Linear(768, 384, bias=False)` /
`Linear(1536, 768, bias=False)`; gather → norm → reduction, bit-identical to the
module in float32 and float64.

**Layout (Tests 1-2).** Milestone 4 checked provenance with 3-channel token IDs,
which never exercises the C-dependent slab widths. Here every element carries a
unique (token, channel) ID, (y W + x) C + c, exact in float32 up to 2^24, at the
merge's full width, compared with timm's own `PatchMerging` gather:

| merge | input C | stage-n windows | elements | x0 … x3 and the 4C concatenation |
|---|---:|---|---:|---|
| PM 1 | 96 | 4 × 6, 8 × 8 | 225,792; 602,112 | exact |
| **PM 2** | **192** | 4 × 6, **16 × 16 (canonical)** | 451,584; **4,816,896** | **exact** |
| **PM 3** | **384** | 4 × 6, **8 × 8 (canonical)** | 903,168; **2,408,448** | **exact** |

One comparison checks source positions (2i + dy, 2j + dx), the group order
(dy fastest), the channel order within each group, window order and output token
order. The same comparison is emulated offline in NumPy, step for step, at
C = 96, 192 and 384 on four grids.

**Numerics (Tests 3-4)**, on generated stage-n tokens (exact float32 in EE and
NumPy), against timm's `layers[n].downsample`:

| tensor | PM 2, 4 × 6 → 2 × 3 | PM 2, 16 × 16 → 8 × 8 | PM 3, 4 × 6 → 2 × 3 | PM 3, 8 × 8 → 4 × 4 |
|---|---|---|---|---|
| groups, 4C concatenation | exact | exact | exact | exact |
| LayerNorm(4C) vs torch32 / float64 | 4.8e-7 / 1.2e-7 | 4.8e-7 / 1.2e-7 | 4.8e-7 / 1.2e-7 | 7.2e-7 / 1.2e-7 |
| reduction (output) vs torch32 / float64 | 5.0e-6 / 2.5e-7 | **6.7e-6 / 2.7e-7** | 6.2e-6 / 2.7e-7 | **7.6e-6 / 2.7e-7** |
| LayerNorm alone, fed PyTorch's concatenation | 4.8e-7 | — | 4.8e-7 | — |
| reduction alone, fed PyTorch's LayerNorm | 5.0e-6 | — | 6.2e-6 | — |
| EE vs mirror (every tensor) | bit-identical | bit-identical | bit-identical | bit-identical |

The reduction error grows only with the dot-product length (384 → 768 → 1536
terms) as PyTorch's float32 GEMM accumulates; EE accumulates in double and stays
within 2.7e-7 of float64.

**Cost** (interactive, cumulative prefixes on generated input at the canonical
grids; two runs; EE evaluates only what the requested tensor needs):

| prefix | PM 1 (32 × 32 → 16 × 16, 96 → 192) | PM 2 (16 × 16 → 8 × 8, 192 → 384) | PM 3 (8 × 8 → 4 × 4, 384 → 768) |
|---|---|---|---|
| input | 294 MiB / 4.6-6.4 EECU·s | 156 MiB / 3.4-3.6 | 87 MiB / 1.2-2.4 |
| + 2 × 2 window gather | 312 MiB / 4.7-5.2 | 175 MiB / 3.0-3.1 | 108 MiB / 1.9-2.2 |
| + 4C concatenation | 312 MiB / 9.1-9.5 | 175 MiB / 5.4-7.6 | 108 MiB / 3.0-4.1 |
| + LayerNorm(4C) | 312 MiB / 10-14 | 175 MiB / 4.3-5.5 | 108 MiB / 3.0-3.8 |
| + reduction: **Patch Merging** | **381 MiB / 12-14** | **304 MiB / 7.6-8.2** | **300 MiB / 9.2-13.8** |

PM 3, the largest merge, costs 9-14 EECU·s and 300 MiB, mostly its 1.18 M-parameter
reduction: read (≈ 2 EECU·s, ~70 MiB decoded) and applied as [49, 1536] ×
[1536, 768] per window. Graph size is ~10-30 KB per merge. The three merges are about
5% of the encoder's cost. Its memory falls with the stage because the
gathered input shrinks 2× per merge in values, while the reduction weights grow
4×.

## 3. Stages 3 and 4 (Phases 5-8)

**No stage-specific code.** `stage.py` runs any number of Milestone 2/3 blocks;
width, heads (head width 32) and MLP width come from the parameters. timm's
configuration is asserted: stage 3 shifts (0, 3) × 3, 12 heads, MLP 1536;
stage 4 shifts (0, 3), 24 heads, MLP 3072, and no shift when it is a single
window (8 × 8 stage-1 windows: timm sets the shift to 0 when the resolution does
not exceed the window; recorded, and both stage-4 blocks are W-MSA there in EE
too). The isolated stages are timm's `SwinTransformerStage` built exactly as
`SwinTransformer` builds `layers[2]` / `layers[3]`; their outputs equal the full
model's stage bit for bit.

**Every operation fed PyTorch's own float32 inputs** (2 × 2 windows, all four
mask types; max |EE − torch32|, every row bit-identical to the mirror):

| operation | stage 3, W-MSA | stage 3, SW-MSA | stage 4, W-MSA | stage 4, SW-MSA |
|---|---:|---:|---:|---:|
| LN1 | 4.8e-7 | 9.5e-7 | 4.8e-7 | 7.2e-7 |
| shift + re-partition | — | 0 | — | 0 |
| QKV (384 → 1152 / 768 → 2304) | 6.7e-6 | 5.7e-6 | 1.05e-5 | 8.1e-6 |
| Q × 32^−0.5 | 0 | 0 | 0 | 0 |
| Q Kᵀ | 2.9e-6 | 1.9e-6 | 2.9e-6 | 2.9e-6 |
| + relative-position bias (+ mask) | 2.9e-6 | 7.6e-6 (masked, ≈ −100) | 3.8e-6 | 7.6e-6 (masked) |
| softmax | 5.1e-7 | 5.7e-7 | 6.6e-7 | 6.0e-7 |
| attention × V | 2.6e-6 | 2.4e-6 | 3.6e-6 | 3.1e-6 |
| projection | 2.6e-6 | 3.1e-6 | 4.3e-6 | 4.8e-6 |
| reverse shift, residuals | 0 | 0 | 0 | 0 |
| LN2 | 9.5e-7 | 9.5e-7 | 7.2e-7 | 9.5e-7 |
| fc1 (→ 1536 / 3072) | 4.5e-6 | 3.8e-6 | 5.5e-6 | 6.2e-6 |
| exact GELU | 4.8e-7 | 4.8e-7 | 4.8e-7 | 4.8e-7 |
| fc2 | 4.1e-6 | 4.5e-6 | 7.2e-6 | 6.4e-6 |

The layout steps and the float32 residual additions are exact; the largest
per-operation differences are PyTorch's float32 GEMM rounding at the widest
projections (24 heads × 32 → 2304 outputs from 768 inputs) and the masked
logits near −100, where the float32 ulp is 7.6e-6.

**Composed stages** (masked generated input, per-stage extension, asset weights):

| test | elements | max vs torch32 | EE vs mirror |
|---|---:|---:|---|
| stage 3, all 6 blocks, **every intermediate** (2 × 3 windows; 132 tensors) | 20.8 M | 2.3e-5 (block-3 masked logits); outputs ≤ 1.6e-5 | **132 of 132 bit-identical** |
| stage 4, both blocks, **every intermediate** (2 × 2 windows; 44 tensors) | 9.3 M | 1.9e-5 (block-1 scores); output 1.3e-5 | **44 of 44 bit-identical** |
| stage 3 on the canonical 8 × 8: outputs after 1 / 2 / 4 / 6 blocks | 1.2 M per output | 7.2e-6 / 1.0e-5 / 1.7e-5 / 2.1e-5 (vs float64: 9.8e-7 / 1.6e-6 / 2.6e-6 / 3.4e-6) | bit-identical |
| stage 3 on 8 × 8, per-block extension, the same four outputs | 4.8 M | identical to per-stage | bit-identical |
| stage 4 on the canonical 4 × 4 | 602,112 per output | 1.3e-5 | bit-identical |
| stage 3 complete in batch (8 × 8) | checksums | rel 1.7e-10 / 3.5e-9 vs float64 | — |
| stage 4 complete in batch (4 × 4) | checksums | rel 1.4e-10 / 6.6e-9 | — |
| final LayerNorm(768) alone (4 × 4) | 602,112 | 7.2e-7 (1.2e-7 vs float64) | bit-identical |

Attention is not degenerate at any width: logit standard deviation 2.2-3.0,
mean row entropy 2.4-3.3 nats (of ln 49 = 3.9) through the twelve blocks.

## 4. One graph, two tasks: the batch embedded-constant cap (Phase 17)

**The single graph fails in batch at any raster size.** The whole encoder as one
checksum request: "Image.reduceRegion: Encoded object too large." at 32 × 32
stage-1 windows (69 EECU·s consumed before the error) and at 8 × 8 (84 EECU·s).
Exports of the final norm and the stage-4 output fail the same way, 18-26 s into
the run. The request is only 916 KB and 6,980 invocations, so this is not the
10 MiB payload limit or a complexity limit, and not memory either (8 × 8 windows
have negligible activations).

**Locating the boundary** (one-pixel batch `reduceRegion` of images that only sum
constant arrays; probe_constant_limit.py (archive `swin_encoder/benchmarks/probe_constant_limit.py` at `research-archive-2026-10-12`)):

| constants embedded in one image graph | float32 | double |
|---:|---|---|
| 12 M values | ✓ | — |
| 16 M | ✓ | ✓ |
| 24 M | ✓ | ✓ |
| 26 M | ✗ Encoded object too large | — |
| 28 M, 30 M, 32 M | ✗ | — |

| real graphs | embedded constants | batch |
|---|---:|---|
| stage-3 block alone (8 × 8 windows) | 2.04 M | ✓ 34 EECU·s |
| complete six-block stage 3 (8 × 8) | 12.26 M | ✓ 228 |
| stage-4 block alone (4 × 4) | 7.62 M | ✓ 45 |
| complete stage 4 (4 × 4) | 15.25 M | ✓ 78 |
| stages 1-3 at 32 × 32 (the head) | 14.25 M | ✓ 229-555 |
| PM 3 → stage 4 → norm from an asset (the tail) | 16.65 M | ✓ 53-68 |
| the whole encoder (8 × 8 or 32 × 32) | **30.9 M** | ✗ |

The cap is 24-26 M values and does not depend on the element type (24 M doubles
pass, as do 24 M floats). The failing probes also consumed 48-88 EECU·s each, so
EE evaluates the constants first and fails when it encodes them for its
workers. That fits EE's evaluation model: value arguments (here each
`ee.Array` read from a table asset, transposed, wrapped as a constant image) are
computed once on the coordinating node, then serialized into the image
computation sent to each tile worker. Milestone 4 embedded 1.63 M values; the
encoder embeds 30.9 M: 27.43 M weight values, 1.95 M bias values and 1.20 M
LayerNorm values (both row-repeated × 49, because EE array arithmetic does not
broadcast), and 0.33 M per-head position biases. Repeating the rows in image
space instead (`arrayRepeat` of a `[1, C]` constant) would still leave 27.8 M,
over the cap. **No bit-exact re-arrangement of embedded weights gets the whole
encoder under it.**

**The two-task pipeline** (earth_engine/pipeline.py (archive `swin_encoder/earth_engine/pipeline.py` at `research-archive-2026-10-12`)):

    task A  stage-1 tokens → stages 1-3 → Export.image.toAsset (56 × 56 px, 384 float32 bands)
    task B  ee.Image(asset) → window layout (49 exact stride samples) → PM 3 → stage 4 → norm

The boundary sits at the stage-3 output, which gives the "strong success"
grouping and puts both graphs well under the cap (14.25 M and 16.65 M). The
intermediate is an EE image asset: float32 values move exactly. A 768-band
probe and every pipeline run read back bit for bit; the tail's read-back
stage-3 input is bit-identical to the mirror at 8 × 8 windows and passes the
float64 checksum at 32 × 32 and 64 × 64. Nothing is downloaded, processed or
re-uploaded outside EE.

| raster (stage-1 windows) | mode | task A (head) EECU·s / run s | task B (tail) EECU·s / run s | verified |
|---|---|---|---|---|
| 8 × 8 | per_stage | 82 / 190 | 9.7 (checksums) | output bit-identical to the mirror |
| **32 × 32 (canonical)** | per_stage | 363 / 369 | 68 / 63 | every element, 5.48e-6 |
| **32 × 32 (canonical)** | **per_block** | **305 / 144** | **53 / 88** | every element, 5.48e-6 |
| 64 × 64 | per_block | 930 / 307 | 141 / 70 | every element, 5.96e-6 |

So the complete canonical encoder costs **~360 EECU·s** in two tasks (Milestone
4's two stages: 170-209). Queue times were 1.4-4.7 s; each task had one attempt.

## 5. Asset-backed parameters at full size (Phases 9-10)

**Storage** is Milestone 4's, unchanged: one feature per matrix row (chunks of
≤ 4,096 values), 9-significant-digit JSON strings, a SHA-256 per record, keyed
by the full timm state_dict name.

| measure | value |
|---|---|
| parameters | 27,514,458 (119 records; stage 3 10.96 M, stage 4 15.37 M) |
| table features | 43,172 |
| payload | 389 MB in 57 table assets (≤ 8 MB each, 2 concurrent export tasks) |
| upload | 3,325 s (55 min): median task run 44 s, the rest client-side shard packing |
| upload EECU·s | 0.29 (all 57 tasks) |
| bit-exact read-back (Test 15) | **all 27,514,458 values, 167 arrays: float32 bits equal** (658 s, 400 k values per request) |
| asset vs inline EE output | PM 2 (inline request 3.98 MB vs 22.7 KB) and final norm: identical bits |

No stage-3 or stage-4 block fits a request inline (1.8 M / 7.1 M parameters at
~13 bytes each vs the 10 MiB limit), so assets are mandatory, as Milestone 4
predicted.

**Read overhead** (all parameters of a part read, decoded and reduced in one
interactive request):

| part | parameters | EECU·s | wall s | reported peak ("Algorithm Array") |
|---|---:|---:|---:|---:|
| stage 1 | 224,694 | 1.1 | 0.8 | 24 MiB |
| stage 2 (+ PM 1) | 966,252 | 3.0 | 1.8 | 58 MiB |
| stage 3 (+ PM 2) | 10,955,400 | 19.0 | 6.6 | 666 MiB |
| stage 4 (+ PM 3, norm) | 15,368,112 | 17.3 | 9.5 | 904 MiB |
| **whole encoder** | **27,514,458** | **43.4** | **13.8** | **1,641 MiB** |

~1.6 EECU·s per million parameters per request, and **~59 bytes of reported EE
memory per decoded parameter**: the read path builds each tensor from a
JSON-decoded `ee.List` of boxed numbers before the `ee.Array`. That memory is
what limits interactive requests (§6), and the evaluated arrays are what the
batch cap counts (§4). The table layout is otherwise adequate: storage,
fingerprints, exactness and read time all scale.

## 6. Interactive execution and memory by stage (Phases 16, 19, 21)

"Peak" below is the EE profiler's largest per-algorithm "Peak Mem" of an
interactive request (image entries include their subgraph), deterministic for a
graph (identical across repeats); EECU·s ranges are two or three runs.

**Each component alone at its canonical resolution** (generated input at the
component's own width; asset weights; whole raster in one request):

| component | windows | dim | peak | EECU·s | interactive |
|---|---:|---:|---:|---:|---|
| stage-1 W-MSA block | 1,024 | 96 | 1,460 MiB | 49-66 | ✓ |
| stage 1 (pair), per-stage | 1,024 | 96 | — | — | ✗ memory limit, as in Milestones 3-4 |
| **stage 1 (pair), per-block** | 1,024 | 96 | **1,080 MiB** | **105** | **✓** (Milestones 3-4 never tried per-block) |
| Patch Merging 1 | 1,024 → 256 | 96 → 192 | 381 MiB | 12-14 | ✓ |
| stage-2 W-MSA block | 256 | 192 | 910 MiB | 26-42 | ✓ |
| stage 2 (pair), per-stage / per-block | 256 | 192 | 1,109 / 683 MiB | 78-93 / 73 | ✓ |
| Patch Merging 2 | 256 → 64 | 192 → 384 | 304 MiB | 8 | ✓ |
| stage-3 W-MSA block | 64 | 384 | 666 MiB | 26-35 | ✓ |
| stage 3 (6 blocks, per-block) | 64 | 384 | 734 MiB | 244-301 | ✓ |
| stage 3 (6 blocks, per-stage) | 64 | 384 | — | — | ✗ memory limit from depth 4 (3 of 3 runs) |
| Patch Merging 3 | 64 → 16 | 384 → 768 | 300 MiB | 9-14 | ✓ |
| stage-4 W-MSA block | 16 | 768 | 497 MiB | 38-43 | ✓ (~420 MiB of it decoded weights) |
| stage 4 (pair), per-stage | 16 | 768 | 907 MiB at failure | — | ✗ memory limit (2 of 2) |
| **stage 4 (pair), per-block** | 16 | 768 | **907 MiB** | **101-105** | **✓** (2 of 2) |
| final norm | 16 | 768 | 54 MiB | 1.5-1.7 | ✓ |
| all 27.5 M weights, read only | — | — | 1,641 MiB | 43 | ✓ |
| **whole encoder, 8 × 8 windows** | 64 | — | 1,641 MiB ("Algorithm Array") | — | **✗ memory limit** |

**Which stage dominates?** Memory is dominated by stage 1's *activations*
(1,024 windows: the largest per-component peak, 1,080-1,460 MiB) and by stage 4's
*weights* (904 MiB decoded, far more than its activations).
**Compute is dominated by stage 3**: 244-301 EECU·s for its six blocks against
78-93 for stage 2, 80-159 for stage 1 (batch) and ~78 for stage 4 (batch),
because each block has the same linear-layer FLOPs at every stage (4× fewer
windows at 2× the width) and stage 3 has six. Patch Merging (8-14 EECU·s each)
and the final norm (2) are minor. **Wall time** follows compute: stage 3's six
blocks take 2-3 min interactively.

**Memory as resolution drops and width rises** (canonical, values per tensor):

| stage | windows | dim | one [49, dim] tensor | MLP hidden tensor | parameters |
|---|---:|---:|---:|---:|---:|
| 1 | 1,024 | 96 | 4.8 M | 19.3 M | 0.22 M |
| 2 | 256 | 192 | 2.4 M | 9.6 M | 0.97 M (with PM 1) |
| 3 | 64 | 384 | 1.2 M | 4.8 M | 10.96 M (with PM 2) |
| 4 | 16 | 768 | 0.6 M | 2.4 M | 15.37 M (with PM 3, norm) |

Activations halve at every stage and stay manageable; parameters grow ~4× per
stage (width²) and overtake activations from stage 3 on. That inversion is why
the whole encoder exceeds interactive memory on any raster, and why the batch
constant cap binds.

**Largest whole-raster interactive inputs** (one request each, checksums
verified against float64):

| graph | 8 × 8 | 16 × 16 | 24 × 24 | 32 × 32 (canonical) |
|---|---|---|---|---|
| head (stages 1-3), per-stage | ✓ 812 MiB / 161 EECU·s | ✓ 940 MiB / 227 | ✗ memory limit | — |
| **head (stages 1-3), per-block** | ✓ 812 MiB / 187 | ✓ 812 MiB / 252 | ✓ 812 MiB / 363 | **✓ 1,061 MiB / 493 EECU·s / 239 s** |
| tail (PM 3 → stage 4 → norm, from the stage-3 asset), per-stage | ✓ 979 MiB / 51 | ✓ 979 MiB / 95 | — | ✗ memory limit |
| **tail, per-block** | — | ✓ 979 MiB / 105 | — | **✓ 979 MiB / 144 EECU·s / 81 s** |
| whole encoder (one graph) | ✗ memory limit (1,641 MiB decoded weights) | — | — | — |

**With per-block extension both halves of the canonical encoder run as single
interactive requests**; the head's peak is flat at 812 MiB (its decoded weights)
up to 24 × 24 windows. 32 × 32 is the largest raster tried with per-block
extension, not a measured ceiling (no larger raster was attempted); the
per-stage ceilings are measured (head: 16 × 16 succeeds, 24 × 24 fails). This reverses Milestones 3-4's finding that the stage-1
pair cannot run interactively on 32 × 32 windows. That finding holds for the
variants they used: periodic input, per-stage and propagated extension. All
three compute block A on a ring of windows around the raster, which pushes the
request over the limit. Only the whole encoder at once is out of reach,
because of its weights. Chaining the two halves interactively still needs the
stage-3 output persisted, which only a batch export does; interactively the
encoder is served as two requests over an existing intermediate.

## 7. Stage-3 depth and stage-4 width (Phases 6, 7, 20)

**Stage 3 alone on its canonical 8 × 8 windows**, interactive, one request per
depth (whole raster), and in batch:

| depth | per-block: peak / EECU·s / wall s | per-stage: peak / EECU·s | graph: KB / invocations | batch (per-stage) EECU·s |
|---:|---|---|---|---:|
| 1 | 666 MiB / 26-27 / 11-12 | 666 MiB / 30-35 | 92 / 723 | 34 |
| 2 | 509 MiB / 68-75 / 35-47 | 986 MiB / 71-109 | 173 / 1,334 | — |
| 4 | 666 MiB / 144-146 / 72 | ✗ memory limit (3 of 3) | 310 / 2,378 | — |
| 6 | 734 MiB / 244-301 / 117-166 | ✗ memory limit (3 of 3) | 447 / 3,425 | 228 |

With per-block extension, **memory is flat in depth** (each SW-MSA block
materializes and re-extends its own input, so the subgraph beneath any pixel is
one block deep). **EECU·s and wall time grow linearly** (~45-50 EECU·s per block),
and the graph grows linearly (~520 invocations per block) with no complexity
error. No unexpected recomputation appears. Six-block depth does not create a
different scaling problem. **The problem it does create is in the extension
rule**: per-stage extension needs a three-window ring for stage 3; the early
blocks then run on up to 3.06× the windows, and from depth 4 the request
exceeds the interactive limit. Correctness is unaffected: depth 1, 2, 4 and 6
are bit-identical to the mirror (per-stage, row requests) and the per-block
outputs equal the per-stage ones.

**Few windows × high width vs many windows × low width** (one W-MSA block per
stage at its canonical grid; identical linear-layer FLOPs per block):

| block | windows × dim | peak | EECU·s |
|---|---|---:|---:|
| stage 1 | 1,024 × 96 | 1,460 MiB | 49-66 |
| stage 2 | 256 × 192 | 910 MiB | 26-42 |
| stage 3 | 64 × 384 | 666 MiB | 26-35 |
| stage 4 | 16 × 768 | 497 MiB | 38-43 |

**Few windows × high width is easier on memory and similar in compute.** Peak
memory falls by 3× from stage 1 to stage 4 (per-pixel arrays are larger, but
there are 64× fewer pixels). EECU·s are flat from stage 2 on: attention FLOPs
halve per stage, while the stage-4 block reads 7.1 M weights per request
(~12 EECU·s of its ~40). At stage 4 the parameter read, not the computation, is
the growing term. Stage 4's 24 heads double the per-block graph (~1,000
invocations, attention written per head) but stay far from any graph limit.

## 8. Extension modes and materialization (Phases 11, 18)

**All modes are bit-identical** (EE vs EE, stage-3 output on 16 × 16 windows:
per_stage, per_block, per_stage with pinned stage outputs, per_stage with pinned
block outputs; and the canonical outputs of the two-task pipeline in per_stage
vs per_block mode). They differ only in cost.

**Milestone 4's per-stage rule needs a three-window ring at stage 3.** An SW-MSA
block reads one window of its own stage on each side, so a stage's output
depends on (number of SW-MSA blocks) windows of its input. That is 1, 1, 3, 1 for
Swin-T, and per_stage extends each stage input by exactly that. With one window
at stage 3, 16 of 16 stage-3 output windows come back masked (§9). The ring is
then computed through the stage: stage-3 block 0 on (k₃ + 6)² windows,
3.06× its 8 × 8 raster at the canonical size. per_block instead re-extends each
SW-MSA block's (materialized) input by one window copied from the opposite edge,
and computes no ring.

| head (stages 1-3), batch | per_block | per_stage | propagated |
|---|---:|---:|---:|
| 16 × 16 stage-1 windows, EECU·s | **83** | 181 | 472 |
| 32 × 32, EECU·s (repeats) | **229, 286** | 555, 382, (363 in the export) | — |
| 32 × 32, run s | 197, 285 (144 in the export) | 372, 161, (369) | — |
| stage-1 output alone at 32 × 32 | 80 | 159 | — |

per_block was cheaper in every pairing (0.41-0.84× the per_stage EECU·s), and
propagated (stage 1 on a 15-window ring) costs most. **For whole-raster and batch requests use per_block**
once a stage has more than one SW-MSA block. That reverses Milestone 4's
recommendation, which was measured on depth-2 stages only. Per-block extension
is also what makes the canonical halves fit interactively (§6): the stage-1 pair
(1,080 MiB), the stage-4 pair (907 MiB), the whole head (1,061 MiB) and the whole
tail (979 MiB) on their canonical grids. Per-stage fails all four. For spatial
chunks Milestone 4 preferred propagated extension. Here the halo covers the
raster, so chunks compute the whole raster in any mode, and per-block (no ring)
is the cheapest per chunk (§9).

**Materialization.** The pins Milestones 3-4 established (each SW-MSA block's
input, the LayerNorm feeding the shift, the projection feeding the reverse
shift, the Patch Merging input, the extension inputs) are sufficient. Extra
materialization only costs:

| head at 32 × 32, batch | EECU·s | run s |
|---|---:|---:|
| per_stage (Milestone 3-4 pins) | 382-555 | 161-372 |
| + every stage output pinned | 628 | 404 |
| + every block output pinned | 718 | 553 |

Separate `reduceRegion`s in one task do **not** share computation: one batch task
with checksums of the seven head boundaries cost 1,585 EECU·s (per_stage) and
2,182 (per_block), ~4-7× one head pass. Each boundary was evaluated with its own
subgraph. The only boundary worth persisting is the stage-3 output, and only
because of the cap (§4).

## 9. Dependency halo and chunking (Phases 12, 22)

**Analytically** (window granularity, which is what window layout computes):
window-local operations add nothing; an SW-MSA block reads ±1 window of its
stage; Patch Merging reads exactly the 2 × 2 windows beneath it; a stage-s window
spans 2^(s−1) stage-1 windows. So the halo is Σₛ 2^(s−1) × (SW-MSA blocks of stage s):

| stage | SW-MSA blocks | stage-1 windows |
|---|---:|---:|
| 1 | 1 | 1 |
| 2 | 1 | 2 |
| 3 | 3 | 12 |
| 4 | 1 | 8 |
| **encoder** | | **23 windows = 161 stage-1 tokens** (head: 15 = 105 tokens) |

(When stage 4 is a single window, as at 8 × 8 stage-1 windows, timm does not
shift it and the encoder halo is 15.) A window-validity model of the data flow
reproduces 15 and 23 exactly (tests/test_full_encoder.py (archive `tests/test_full_encoder.py` at `research-archive-2026-10-12`)).

**Experimentally:**

| test (live) | result |
|---|---|
| head, propagated ring 14 stage-1 windows (8 × 8 raster, batch) | stage-3 output **entirely masked** (count 0) |
| head, propagated ring **15** | **complete and correct** (checksums vs float64 7.4e-10) |
| per-stage rings 1/1/1/1 (16 × 16 raster) | 16 of 16 stage-3 output windows masked |
| per-stage rings 1/1/2/1 | 12 of 16 masked (only the interior 2 × 2 survive) |
| per-stage rings 0/1/3/1, 1/0/3/1 | 16 of 16 masked |
| per-stage rings **1/1/3/1** | **0 masked, output correct** |
| tail, stage-4 ring 0 | 4 of 4 final windows masked |

Each stage's ring is necessary, and the head's halo is exactly 15 windows; with
stage 4's 8, the encoder's is 23. (The full propagated test cannot run as one
graph, for the same cap.)

**Chunking.** The halo (23 windows on each side) exceeds the canonical raster
(32 windows), so every chunk of the canonical encoder depends on the whole
raster. And the whole encoder exceeds the interactive memory limit on its
decoded weights alone (§6). Chunked interactive execution of the complete
encoder is therefore not viable at this size. One experiment on the head
(stages 1-3, halo 15 windows), per-block extension, 16 × 16 stage-1 windows in
four 8 × 8-window chunks:

| chunk (stage-3 windows) | peak | EECU·s | wall s |
|---|---:|---:|---:|
| rows 0-2, cols 0-2 | 812 MiB | 259 | 150 |
| rows 0-2, cols 2-4 | 1,294 MiB | 222 | 99 |
| rows 2-4, cols 0-2 | 812 MiB | 260 | 140 |
| rows 2-4, cols 2-4 | 812 MiB | 283 | 174 |
| **4 chunks** | | **1,024** (checksums combine to float64 ✓) | 563 sequential |
| whole raster, one interactive request | 812 MiB | 252 | 130 |
| whole raster, batch | — | 83 | 154 |

Every chunk computes the whole raster (its halo covers it), so chunking cost
**4.1× the whole-raster interactive EECU·s and 12× batch**, with no memory
benefit. For larger rasters a chunk of n × n stage-1 windows needs stage 1 on
(n + 46)² windows (full encoder):

| raster | chunk | chunks | stage-1 windows per chunk | stage-1 recompute vs whole raster |
|---:|---:|---:|---:|---:|
| 64 | 32 | 4 | 4,096 (whole raster) | 4.0× |
| 128 | 64 | 4 | 12,100 | 2.95× |
| 256 | 64 | 16 | 12,100 | 2.95× |
| 256 | 32 | 64 | 6,084 | 5.94× |
| 256 | 16 | 256 | 3,844 | 15.0× |

Batch is clearly preferable for the complete encoder.

## 10. Graph size (Phase 23)

| request (asset parameters unless noted) | bytes | invocations after dedup | distinct functions | serialize s |
|---|---:|---:|---:|---:|
| M2 block, inline / asset | 1,498,720 / 55,461 | 345 / 451 | 44 / 54 | 1.38 / 0.01 |
| M3 pair, inline / asset | 2,987,146 / 99,031 | 588 / 785 | 51 / 61 | 2.96 / 0.02 |
| M4 hierarchy (4 or 32 windows) | 221,926 | 1,721 | 69 | 0.05 |
| M5 stage 3, 1 / 2 / 4 / 6 blocks | 91,645 / 172,990 / 309,644 / 446,559 | 723 / 1,334 / 2,378 / 3,425 | 59-69 | 0.02-0.12 |
| M5 stage 4, 1 / 2 blocks | 139,903 / 266,821 | 1,088 / 2,041 | 59 / 69 | 0.03 / 0.05 |
| M5 head (stages 1-3), 32 × 32 | 656,814 | 5,030 | 69 | 0.20 |
| M5 tail (PM 3 → norm, from the asset) | 323,025 | — | — | — |
| **M5 whole encoder, 8 × 8 or 32 × 32** | **904,064 / 915,606** | **6,900 / 6,980** | **69** | **0.26** |

The graph is constant in raster size and grows linearly with depth (~520 to ~1,000
invocations per block; a 24-head block is about twice a 12-head block, because
attention is written per head). With assets the parameters add only their read
expressions. No request was rejected as "too complex"; the cap that binds is
on embedded *values* (§4), not graph size.

## 11. Earth Engine behavior found or confirmed (probed live)

1. **Batch caps the evaluated constants embedded in one image graph at 24-26 M
   values**, independent of element type, and fails with "Encoded object too
   large." after evaluating them (48-88 EECU·s spent). Interactive requests hit
   the user memory limit first.
2. **Decoded asset-read arrays cost ~59 bytes per value** of reported memory
   ("Algorithm Array"), because each tensor is built from a JSON-decoded boxed
   `ee.List`; ~1.6 EECU·s per million values per request.
3. **`arrayReduce(sum)` sums strictly left to right**, and `matrixMultiply`
   accumulates each dot product sequentially in double. torch's float64 matmul
   agrees bit for bit with sequential accumulation (0 of 768 k dot products of
   length 384-3,072 differ), but torch's `sum` does not (80-97% of rows differ in
   the last double bits). One of 2.4 M LayerNorm(768) elements rounded
   differently to float32 as a result.
4. **Separate `reduceRegion`s in one batch task do not share work**: seven
   boundary checksums cost ~4-7× one head pass.
5. **Image assets carry float32 tensors exactly**: a 768-band token raster exports
   and reads back (`computePixels`, NPY) bit for bit. NumPy's default header
   limit rejects the 768-field structured array; raw NPY bytes with a larger
   `max_header_size` load it.
6. As in Milestones 2-4: constant array images are not replicated per pixel;
   stride sampling relabels exactly at any factor; translated views of
   un-materialized images re-evaluate their subgraph; EE answers an identical graph
   from its cache, so every repeat used a new checksum seed.

## 12. Recommendation: proceed to Milestone 6

**Proceed.** On each criterion:

- **Full encoder correctness**: every element of the canonical output within
  5.48e-6 of timm, bit-identical to EE's own specification; 448 × 448 tokens too.
- **Stage-3 depth scaling**: correct at every intermediate; graph and cost linear
  in depth; no new graph limit (§7). The depth requires per-block extension to
  stay cheap (§8).
- **Stage-4 width scaling**: correct at 24 heads and MLP 3072; activations are
  small; the width's cost is in the parameters (§5, §7).
- **Patch Merging scaling**: one implementation, exact layout at every width,
  ≤ 7.6e-6 at the widest.
- **Final numerical error**: ~1e-5 at every boundary, no accumulation with depth.
- **Batch reliability**: every batch task of the pipeline succeeded at the first
  attempt (8 × 8, 32 × 32 in both modes, 64 × 64).
- **Memory scaling**: activations halve per stage; with per-block extension each
  half of the canonical encoder fits one interactive request; the weights bound
  any request that holds all of them.
- **Graph complexity**: < 1 MB, constant in raster size, no complexity errors.
- **Asset-backed parameters**: exact and fast to read, but they set the execution
  model: two batch tasks, and head/tail for interactive work.
- **Periodic extension**: exact for arbitrary inputs; per-block for whole rasters.
- **Window layout**: preferred through all four stages.
- **EECU cost**: ~360 EECU·s for the canonical encoder (~0.35 EECU·s per stage-1
  window), ~1,070 at 448 × 448.

For Milestone 6 (patch embedding from Sentinel-1 imagery, end-to-end forward pass
to encoder features):

1. **Keep the two-task pipeline**, split at the stage-3 output. The patch
   embedding (4 × 4 × C_in → 96, ~5 k parameters) belongs to task A; both
   graphs stay far under the embedded-constant cap.
2. **Use per-block extension** for batch and whole-raster requests.
3. **Plan batch for the whole encoder; use interactive requests per half.** With
   per-block extension the canonical head and tail each fit one interactive
   request (and the head up to 32 × 32 stage-1 windows), but the whole encoder
   never does (its decoded weights alone exceed the limit), and chaining needs
   the stage-3 intermediate persisted by a batch export.
4. **Mind the receptive field.** The encoder output depends on ±161 stage-1 tokens
   (±644 input pixels at patch size 4). Tiling real scenes needs that much
   overlap, or the cyclic boundary of the canonical model must be replaced by
   the imagery's real neighbourhood (a modelling choice for Milestone 6).
5. **If one graph becomes a requirement**, the weights must reach the workers
   some other way than as embedded values (e.g. image assets read per tile);
   that is an asset-format redesign, now justified by measurement.
6. **Keep the mirror test** (with EE's sequential sums) for every new operation.

## 13. Caveats

- **Profiler semantics.** "Peak Mem" is EE's per-algorithm accounting, not RAM;
  peaks are compared only within one graph shape. "Ceiling" means the largest
  success and smallest failure observed.
- **Shared service.** EECU·s vary up to ~1.5× between identical runs (head at
  32 × 32 per-stage: 363-555). Mode comparisons were repeated; single
  observations are indicative.
- **The cap's exact value** is bracketed (24 M pass, 26 M fail) with synthetic
  constants. The real graphs fall on either side by a wide margin (≤ 16.7 M pass,
  30.9 M fail).
- **Inputs and parameters.** Generated hash tokens and seeded parameters, not a
  trained model or real imagery. Inputs are non-periodic and masked outside the
  raster. The cyclic boundary is timm's semantics for one image, not a physical
  neighbourhood.
- **Scope.** Swin-T encoder from stage-1 tokens: no patch embedding, no decoder,
  no head. Stage-1 window counts are multiples of 8 (timm's padding of other
  sizes is not implemented), and timm's per-axis shift (one axis a single
  window) is rejected.
- **Assets.** The weight assets (`swin_encoder/m5_weights_*`), the pipeline's
  stage-3 intermediates and exports (`swin_encoder/exports/`) and the checksum
  tables (`ee_attention_benchmarks/`) remain in project `ORIGINAL_PROJECT`; delete
  the exports and checksum tables when finished.
