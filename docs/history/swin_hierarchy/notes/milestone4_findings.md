# Findings: hierarchical Swin (Stage 1 → Patch Merging → Stage 2) in native Earth Engine

Live experiments on 4 October 2026: Earth Engine Python API 1.7.45, PyTorch
2.14.0, timm 1.0.30, NumPy 2.5.3, Cloud project `ORIGINAL_PROJECT`,
application-default credentials. Every forward pass ran entirely in EE
computation graphs: no Vertex AI, Cloud Run, model server, exported imagery or
client-side tensor computation. Stage-1 features are never exported or
re-imported; all 1,190,946 parameters are read from EE table assets inside the
graph. EE-computed tensors were fetched only for comparison.

Evidence: numerical_comparison_m4.csv (development archive: `docs/history/swin_hierarchy/results/numerical_comparison_m4.csv`)
(summary: numerical_summary_m4.md (development archive: `docs/history/swin_hierarchy/results/numerical_summary_m4.md`)),
benchmark_results_m4.csv (development archive: `docs/history/swin_hierarchy/results/benchmark_results_m4.csv`) (summary:
benchmark_summary_m4.md (development archive: `docs/history/swin_hierarchy/results/benchmark_summary_m4.md`)), per-request
records with full EE profiles in results/raw/ (archive `swin_hierarchy/results/raw/` at `research-archive-2026-10-12`), Patch
Merging semantics and reference digests in
data/manifest.json (archive `swin_hierarchy/data/manifest.json` at `research-archive-2026-10-12`), the weight assets in
data/weights_manifest.json (archive `swin_hierarchy/data/weights_manifest.json` at `research-archive-2026-10-12`).

**Research question.** Can native Earth Engine execute the hierarchical
transition between two genuine Swin stages, including exact Patch Merging,
the channel-width increase and a complete Stage-2 W-MSA/SW-MSA pair, while
preserving timm/PyTorch numerical behavior? **Yes.**

- Patch Merging is computed directly from 2 × 2 neighbouring stage-1 windows,
  with no fine token raster. Its token provenance and 4C order equal timm's
  at every token, including at the 224 × 224 target (1.31 M layout
  elements).
- The complete Stage-1 pair → Patch Merging → Stage-2 pair matches timm's
  `SwinTransformer.layers[0:2]` to **1.13e-5** over all 2,408,448 outputs of
  the canonical target (224 × 224 → 112 × 112 tokens). The target is 1e-4.
- At every intermediate of every block, EE is bit-identical to the float64
  mirror of its algorithm: **30.3 M elements, zero differences**.
- The canonical target runs as **one batch task** (170–209 EECU·s, 2–4 min).
  It also runs as 4 or 16 interactive chunks, or 32 interactive row requests.
  The largest single interactive request is 28 × 28 stage-1 windows
  (196 × 196 tokens).

**Milestone verdict.** Minimum, strong and full architectural success are all
met; full success is via **Option B (batch)**. The canonical target exceeds
the interactive per-request memory envelope in every mode. The stage-1 pair
alone already does, as in Milestone 3. That is a resource limit, not an
architectural one: the same graph succeeds in batch up to 448 × 448
stage-1 tokens, and in ≤ 256-window interactive chunks.

## Answers

| # | Question | Answer |
|---|---|---|
| 1 | Can timm Patch Merging be reproduced exactly in native EE? | **Yes.** The gather and 4C concatenation are exact (0 error, every element). LayerNorm(384) and the reduction are bit-identical to the mirror and within 3.6e-6 of PyTorch when fed PyTorch's own inputs (§2, §3). |
| 2 | Directly from four neighbouring Stage-1 windows, without fine-grid expansion? | **Yes.** Each stage-2 window is assembled from stage-1 windows (2P + a, 2Q + b). They are read by exact stride sampling of the coarse grid, then go through `arrayCat` / `arrayReshape` / `arraySlice` slabs to `[49, 384]`. No token raster is formed (§2). |
| 3 | Is the 4C token concatenation order exact? | **Yes, 100%.** Unique token IDs: each of the four gathered windows, each group x0 … x3 and the concatenation equal timm's own `PatchMerging` gather at every token. Tested on 4×4, 4×6, 6×4, 8×8 and 32×32 stage-1 windows (§2). |
| 4 | Does Patch Merge LayerNorm match timm? | **Yes.** LayerNorm(384), fed PyTorch's concatenation: max 7.2e-7 vs PyTorch, 2.3e-7 vs float64, bit-identical to the mirror. Composed in the hierarchy: 3.4e-6 (§3). |
| 5 | Does the 384→192 reduction match timm? | **Yes.** Fed PyTorch's LayerNorm output: max 3.6e-6 vs PyTorch, 2.3e-7 vs float64, bit-identical to the mirror. There is no bias and no addition, as in timm's `Linear(bias=False)` (§3). |
| 6 | Maximum Patch Merge output error? | **5.5e-6** at the 224 × 224 target (2.4 M elements; MAE 5.1e-7). 4.1e-6 on 4×6 and 8×8 windows, 3.6e-6 in isolation. EE vs float64: 7.8e-7 (§1). |
| 7 | Can Stage-2 width 192 use the same generic primitives? | **Yes, the same code.** The block reads its width and head count from the parameters. One code path runs dim 96 / 3 heads / MLP 384 and dim 192 / 6 heads / MLP 768. The generalization left 9 Milestone 2/3 request graphs byte-identical (§6). |
| 8 | Can 6-head W-MSA run correctly? | **Yes.** Every intermediate is bit-identical to the mirror. Max error vs PyTorch is 1.3e-5 composed (Q Kᵀ, 8×8 windows) and 4.3e-6 per component in isolation (§3). |
| 9 | Can 6-head SW-MSA run correctly? | **Yes.** The shift, mask and reverse shift are exact; the mask equals timm's buffer element by element. Max error is 1.5e-5 composed and 7.6e-6 in isolation. Both are at masked logits near −100, where the float32 ulp is 7.6e-6 (§3). |
| 10 | Does the full Stage-2 pair match timm? | **Yes.** Stage-2 output within 9.8e-6 (4×6 stage-1 windows), 1.04e-5 (8×8) and 1.13e-5 (32×32) of PyTorch (§1). |
| 11 | Does the complete Stage-1 → Patch Merge → Stage-2 hierarchy match timm? | **Yes.** The reference is asserted bit-identical to `layers[0]`, `layers[1].downsample`, `layers[1]` and `model.layers`. EE matches it at every intermediate on 4×6 and 8×8 stage-1 windows, and at Patch Merging and the output on 32×32 (§1). |
| 12 | Maximum final-output error? | **1.13e-5** (224 × 224 target, 2,408,448 outputs; MAE 1.5e-6, RMSE 1.8e-6). The largest error at any compared intermediate is 1.53e-5, at stage-2 masked logits (§1). |
| 13 | Does the controlled-rounding mirror remain bit-identical to EE? | **Yes, everywhere.** 30,313,948 EE-computed elements: Patch Merging and stage-2 components in isolation, every intermediate of the small and moderate hierarchies, and the target. Zero differences. EE is 4.5× closer to float64 than PyTorch at the target output (2.5e-6 vs 1.13e-5) (§1). |
| 14 | What dominates Patch Merge memory? | **The stage-1 input it gathers from**: the input itself reports 294 MiB at 1,024 stage-1 windows, its materialization for the gather (`Image.reproject`, the dominant entry) 312 MiB. The four-window gather and the slab re-layout add ~18 MiB; LayerNorm and the reduction bring the reported peak to 381 MiB (§4). |
| 15 | What dominates Stage-2 memory? | **The materialization of block A's output** that the SW-MSA shifts read: 853 → 1,237 MiB at 256 stage-2 windows. Its accounting includes all of block A. Within block A, attention + projection with 6 heads is the largest step (509 → 765 MiB) (§5). |
| 16 | Does the MLP remain the dominant memory consumer? | **For compute, yes; for memory, no longer in stage 2.** `erf` (exact GELU) is the largest compute term of the hierarchy (28% of EECU·s). In stage-2 block A the MLP adds +137 MiB vs +256 MiB for attention + projection. In Milestone 2's stage-1 block the MLP was the largest step (+529 MiB at 1,024 windows) (§5). |
| 17 | Does direct window-layout Patch Merging outperform fine-grid expansion? | **It is equal or better, and simpler.** Patch Merging alone at 1,024 windows, 3 runs each: same peak (312 / 381 MiB), EECU·s within noise (out: median 13.9 vs 14.4). The fine grid needs 1,498 vs 14 reproject invocations and 2.3× the request bytes, plus a fix for negative coordinates. In the hierarchy: same peak and ceiling, ~1.4× EECU·s (median 93 vs 65 at k = 16). Both are bit-identical (§4). |
| 18 | What does Patch Merging cost in EECU? | **~9–18 EECU·s** at 1,024 stage-1 windows (224² → 112², 3 runs), of which the gather and slab layout are ~1.5–6 above reading the input. That is under 10% of the hierarchy (209 EECU·s in batch) (§4). |
| 19 | What does the complete hierarchy cost? | **170–209 EECU·s** in batch at the canonical target (per-stage / propagated), 376 at 336 × 336 tokens, 642–753 at 448 × 448. Interactive: 117–123 EECU·s at 168 × 168, 170 at 196 × 196. ~1.1–2× Milestone 3's stage-1 pair alone (107–159 EECU·s in batch) (§7). |
| 20 | Largest hierarchy that runs interactively? | **28 × 28 stage-1 windows** (196 × 196 → 98 × 98 tokens) as one request with per-stage extension. 24 × 24 with propagated extension; 32 × 32 fails in both (§7). |
| 21 | Does the canonical 224 → 112 target run interactively? | **Not as one request**, in either extension mode: the whole-raster requests exceed the user memory limit. The stage-1 pair alone already does, as in Milestone 3. **As chunks, yes**: 4 × 256-window and 16 × 64-window chunks, and 32 row requests (16 for Patch Merging, 16 for the output) that fetched every element (§7). |
| 22 | If not, does it run in batch? | **Yes**, in both modes, verified against float64 (Test 14). Propagated: 209 EECU·s, 107 s run, 9 s queue. Per-stage: 170 EECU·s, 201 s run, 5 s queue. Also 2,304 and 4,096 stage-1 windows (376 and 642–753 EECU·s) (§7). |
| 23 | Is chunked interactive execution practical? | **Yes, with the propagated extension.** 4 chunks of 16 × 16 stage-1 windows: 299 EECU·s total, 114 s sequential, max 793 MiB. 16 chunks of 8 × 8: 605 EECU·s, 279 s. Chunk checksums combine to the whole-raster float64 checksum. With per-stage extension, chunks at the raster edge must also compute stage 1 at the opposite edge: corner chunks exceed the limit (0 of 4 at 16 × 16, 12 of 16 at 8 × 8) (§7). |
| 24 | What dependency halo is required? | **3 stage-1 windows (21 stage-1 tokens) on each side**, in either mode. Stage-2 SW-MSA reads ±1 stage-2 window (±2 stage-1), and stage-1 SW-MSA reads ±1 more. Demonstrated: with extension rings of 1 or 2 windows (propagated) every stage-2 output window is masked; with 3 the output is complete and correct. A 16 × 16 chunk recomputes stage-1 block A on 1.89× its windows, an 8 × 8 chunk on 3.06× (§3, §7). |
| 25 | Does explicit periodic extension of arbitrary inputs work correctly? | **Yes, exactly.** The ring around the raster equals `np.pad(wrap)` at every edge and corner, for inputs masked outside the raster and for unmasked ones. The shift and reverse shift of extended non-periodic IDs equal `torch.roll` at every edge and corner. The hierarchy from a masked non-periodic input is bit-identical to Milestone 3's periodic-input path. Negative controls fail as expected (§3). |
| 26 | What does periodic extension cost? | **The extension itself is small:** 1.6 EECU·s / 69 MiB at 256 windows, 6.4 / 277 MiB at 1,024 (mosaic, 3-window ring). `where` costs 648 MiB at 256 windows and exceeds the limit at 1,024. **The real cost is the ring that downstream stages compute.** Per-stage extension (1 + 1) computes stage-1 block A on (k + 2)² windows instead of (k + 6)² for propagated. Measured: 443 vs 665 MiB at k = 16 (3 of 3 runs), 933 vs 1,245 MiB at k = 24, EECU·s within noise, interactive ceiling 28 vs 24 (§3). |
| 27 | Do asset-backed weights reproduce the source parameters exactly? | **Yes.** All 1,190,946 values read back from the assets are bit-identical to the float32 sources. The stage-1 pair, Patch Merging and each stage-2 block computed with asset vs inline parameters give bit-identical EE outputs (§6). |
| 28 | How much does asset-backed storage reduce serialized request size? | **15.76 MB → 218 KB (72×).** Inline, the hierarchy cannot run at all: EE rejects it ("Request payload size exceeds the limit: 10485760 bytes"). Serialization drops from 18 s to 0.05 s, and request wall time from 24–43 s to 4–5 s per part. Reading the assets costs ~1–2 EECU·s per request; all 1.19 M parameters take 4.5 EECU·s and 2.1 s (§6). |
| 29 | Does graph complexity remain manageable? | **Yes.** 218 KB, 1,695 function invocations after deduplication, 69 distinct functions. It is constant in raster size (identical at 4 and 1,024 windows), with no "too complex" error in any run (§8). |
| 30 | Is window layout still the preferred representation after Patch Merging? | **Yes.** The hierarchical transition stays on coarse grids: stage-2 windows are 2 × 2 stage-1 windows. The fine-grid alternative is no cheaper anywhere and costs more in the hierarchy (§4). |
| 31 | Should we proceed to M5? | **Yes**, with batch as the canonical execution mode (per-stage extension), chunked interactive execution as the alternative (propagated extension), asset-backed weights, and the mirror test kept for each new operation (§10). |

## 1. Numerical equivalence (Phases 1, 8-13)

**Reference.** timm's own `SwinTransformer(embed_dim=96, depths=(2, 2),
num_heads=(3, 6), window_size=7, mlp_ratio=4)`, `layers[0]` then `layers[1]`
(Swin-T's first two stages; the patch embedding is not used).
`fused_attn=False` on every block. `layers[1]` starts with timm's
`PatchMerging(96 → 192)`, then a W-MSA and an SW-MSA block at dim 192 with 6
heads. The explicit forward exposes every intermediate. It is asserted
bit-identical to `layers[0](x)`, `layers[1].downsample`, `layers[1]` and
`model.layers(x)` in float32 and float64, so every compared tensor is timm's
own.

Stage-1 blocks keep Milestone 3's parameters (seeds 0, 1). Stage-2 blocks
(seeds 2, 3) use the same deliberately non-trivial spec, generalized in width
(qkv scale ∝ 1/√dim). Stage-2 attention is therefore neither flat nor
saturated: logits std 2.55, max probability 0.99, mean row entropy 2.2 nats.
Patch Merging (seed 4) has LayerNorm gamma/beta ≠ 1/0. Inputs are Milestone 2's
generated tokens: **not periodic, and masked outside the raster**, the
imagery-like production case. timm uses no shift when a stage has a single
window per side, so stage-1 rasters have an even number ≥ 4 of windows per
side. The 4 × 6 case gives stage 2 the 2 × 3 windows that contain all four
mask types.

**Final output vs PyTorch float32** (stage-by-stage tables:
numerical_summary_m4.md (development archive: `docs/history/swin_hierarchy/results/numerical_summary_m4.md`)):

| case | elements | MAE | RMSE | max abs error | max \|EE − float64\| | max \|torch32 − float64\| | EE vs mirror |
|---|---:|---:|---:|---:|---:|---:|---|
| 4×6 → 2×3 windows, every intermediate (96 tensors) | 56,448 | 1.5e-6 | 1.9e-6 | 9.8e-6 | 1.8e-6 | 1.1e-5 | bit-identical (10.9 M elements) |
| 8×8 → 4×4 windows, Patch Merging and stage 2 (54 tensors) | 150,528 | 1.5e-6 | 1.9e-6 | 1.0e-5 | 2.0e-6 | 1.1e-5 | bit-identical (11.2 M) |
| **32×32 → 16×16 (224² → 112² tokens)** | **2,408,448** | **1.5e-6** | **1.8e-6** | **1.13e-5** | **2.5e-6** | **1.13e-5** | **bit-identical (4.8 M)** |

**Hierarchy boundaries, 4 × 6 stage-1 windows** (all bit-identical to the mirror):

| tensor | MAE | RMSE | max abs error | max \|EE − float64\| |
|---|---:|---:|---:|---:|
| stage-1 W-MSA block output | 4.4e-7 | 5.8e-7 | 4.1e-6 | 1.1e-6 |
| stage-1 pair output | 8.2e-7 | 1.1e-6 | 6.0e-6 | 1.7e-6 |
| Patch Merging: 4C concatenation | 8.2e-7 | 1.1e-6 | 6.0e-6 | 1.7e-6 |
| Patch Merging: LayerNorm(384) | 4.1e-7 | 5.3e-7 | 3.4e-6 | 9.2e-7 |
| Patch Merging: reduction (output) | 5.1e-7 | 6.5e-7 | 4.1e-6 | 6.3e-7 |
| stage-2 W-MSA: Q Kᵀ (6 heads) | 1.8e-6 | 2.2e-6 | 1.0e-5 | 2.4e-6 |
| stage-2 W-MSA block output | 1.1e-6 | 1.4e-6 | 7.6e-6 | 1.4e-6 |
| stage-2 SW-MSA: cyclic shift + re-partition | 6.7e-7 | 8.6e-7 | 5.7e-6 | 9.6e-7 |
| stage-2 SW-MSA: attention mask | 0 | 0 | 0 | 0 |
| stage-2 SW-MSA: + mask (masked logits) | 2.3e-6 | 3.4e-6 | 1.5e-5 | 6.0e-6 |
| stage-2 SW-MSA: softmax | 3.3e-8 | 1.1e-7 | 1.9e-6 | 5.0e-7 |
| stage-2 SW-MSA: reverse shift | 8.7e-7 | 1.1e-6 | 6.5e-6 | 1.3e-6 |
| **hierarchy output** | 1.5e-6 | 1.9e-6 | 9.8e-6 | 1.8e-6 |

The layout steps carry their input's error unchanged: the 4C concatenation
row equals the stage-1 output's, the shift row equals LN1's, the reverse
shift row equals the projection's. The largest error anywhere is 1.53e-5, at
stage-2 masked logits near −100, which is two float32 ulps of 100. Masked
probabilities are below 1e-38, so it does not propagate. The first nonzero
difference is stage-1 LayerNorm 1 (4.8e-7), as in Milestones 2-3. Every later
difference is PyTorch's float32 kernel rounding: at every tensor EE lies
within PyTorch's own distance from float64 (Q13).

Extension modes. The small case was validated in both modes: first
propagated, then per-stage (production); both runs are bit-identical to the
mirror with identical errors, and the per-stage rows are the ones recorded.
The moderate case and the target's every-element check ran propagated.
The target is fetched in requests of one stage-2 window row, which are
spatial chunks; with per-stage extension the first row request exceeded the
memory limit (§3). The target in per-stage mode is verified by its batch
checksum (§7). The formulation variants (§3) are bit-identical to one
another, so every number above holds for both modes.

## 2. Patch Merging semantics and layout (Phases 2-5)

**timm's semantics, established by experiment** on unique token IDs through
timm's own `PatchMerging` (the module with its norm and reduction replaced by
identities, so the gather is timm's code, not a re-implementation):

| fact | finding |
|---|---|
| four source tokens | (2i, 2j), (2i + 1, 2j), (2i, 2j + 1), (2i + 1, 2j + 1) |
| concatenation order | x0 = x[0::2, 0::2], x1 = x[1::2, 0::2], x2 = x[0::2, 1::2], x3 = x[1::2, 1::2] (the dy offset varies fastest) |
| channels within a group | the source token's channels 0 … C−1, in order |
| order of operations | gather → LayerNorm → reduction; no activation, no residual |
| LayerNorm | `LayerNorm(384)`, eps 1e-5, affine |
| reduction | `Linear(384, 192, bias=False)` |
| padding | zeros at the bottom row and right column for odd H or W (groups x1/x3 and x2/x3 respectively). Not reached here: stage-1 rasters have an even number of windows |
| output spatial order | merged token (i, j) row-major over the H/2 × W/2 raster |
| output channel order | `Linear` output channels |
| float32 rounding | gather none; LayerNorm one float32 output; reduction one float32 output |

`reduction(norm(gather(x)))` equals `PatchMerging(x)` bit for bit in float32
and float64. These facts are recorded in data/manifest.json (archive `swin_hierarchy/data/manifest.json` at `research-archive-2026-10-12`)
and asserted by tests/test_patch_merge.py (archive `tests/test_patch_merge.py` at `research-archive-2026-10-12`).

**Window-layout construction.** A stage-2 window (P, Q) covers exactly the
14 × 14 stage-1 tokens of stage-1 windows A = (2P, 2Q), B = (2P, 2Q+1),
C = (2P+1, 2Q), D = (2P+1, 2Q+1):

1. gather each onto the stage-2 coarse grid (2× coarser) with Milestone 1's
   stride sampling: `reproject` to a projection whose pixel centres are the
   centres of stage-1 coarse pixels (2P + a, 2Q + b), then `changeProj` to the
   stage-2 grid (exact selection; the stage-1 raster is materialized once);
2. `[14, 14C]` block: A|B and C|D side by side in their `[7, 7C]` reshapes,
   stacked;
3. `[7, 28C]` reshape (row u, column dy·14C + x·C + c), one slice per row
   parity dy, `[49, 2C]` reshape (row 7u + v, column dx·C + c): already
   stage-2 window layout;
4. x0 … x3 are `[49, C]` column slices, concatenated in timm's order.

Seven is odd, so merged row u = 3 pairs stage-1 token row 6 (windows A, B)
with token row 7 (row 0 of windows C, D). The windows cannot be merged one
at a time; the 2 × 2 block is assembled first. No arithmetic touches a value.

**Tests (exact or fail;** numerical_summary_m4.md (development archive: `docs/history/swin_hierarchy/results/numerical_summary_m4.md`)**):**

| test | rasters (stage-1 windows) | result |
|---|---|---|
| 1 each gathered window is stage-1 window (2P + a, 2Q + b) | 4×4, 4×6, 6×4, 8×8, 32×32 | 100% exact |
| 2 every merged token's four source token IDs vs timm | same | 100% exact |
| 3 each group x0 … x3 and the 4C concatenation vs timm | same | 100% exact |
| 7 stage-2 window rows = stage-2 token raster (7P + u, 7Q + v) | same | 100% exact |
| B fine-grid gather = window gather (IDs and 96-channel floats) = NumPy `x[dy::2, dx::2]` | 4×4, 8×8 | bit-identical |

The live tests repeat Tests 1-13 at small sizes
(tests/test_swin_hierarchy_live.py (archive `tests/test_swin_hierarchy_live.py` at `research-archive-2026-10-12`)):
12 of 12 pass (107 s). Test 14 (the canonical target as one batch task,
`EE_LIVE_BATCH=1`, per-stage extension) passes too (208 s, verified against float64).

## 3. Periodic extension and the dependency halo (Phase 7, Test 8)

Milestone 3's shift reads the window across the raster edge from the
translated neighbour, which is torch.roll's wrap only for an input periodic
with the raster's window counts. Its inputs were generated periodic. Here the
cyclic domain is supplied explicitly:
`extended(I, J) = x(I mod kh, J mod kw)` on a ring of windows around the
raster. It is built by mosaicking the materialized input with copies
translated by multiples of the raster size (`changeProj`, exact), each clipped
to the strip it fills: rows, then columns. Inside the raster the input is
untouched, so Swin's semantics are unchanged.

**Correctness** (non-periodic token IDs from the raw window index: outside
the raster they name tokens that do not exist):

| raster (windows) | input / method | 3-window ring vs `np.pad(wrap)` (interior, 4 edges, 4 corners) | shift / reverse shift vs timm `torch.roll` (all, each edge, each corner) |
|---|---|---|---|
| 2×2, 4×4, 3×5, 4×6 | masked outside the raster, `mosaic` | all exact | all exact |
| same | unmasked, `mosaic` | all exact | all exact |
| same | unmasked, `where` | all exact | all exact |
| same | negative controls | `where` on the masked input: the ring is lost (masked) | shift without extension: wrong at the edges (73-88% of elements correct) |

`where` keeps masked input pixels masked (Milestone 3's probe 4), so it
cannot extend an input masked outside its footprint. `mosaic` takes each
strip only where its copy is valid and works for both.

**In the hierarchy** (4 × 6 stage-1 windows, EE vs EE, bit for bit): the
masked input with per-stage `mosaic` extension (production), the same with
propagated extension, Milestone 3's periodic input without any extension,
the unmasked input with `where` or `mosaic`, and the fine-grid Patch Merging
gather all give **identical Patch Merging and output bits**.

**Two ways to supply the wrap, both exact:**

- *propagated*: extend the stage-1 input once by the hierarchy's dependency
  halo. Window-local operations, the periodic-wrap shifts and Patch Merging
  map periodic inputs to periodic outputs (Patch Merging halves the period),
  so stage 2 needs no extension. But stage 1 is then evaluated on a
  three-window ring: block A on (k + 6)² windows, 1.41× at k = 32.
- *per_stage* (**production**): extend each stage's input by one window: the
  stage-1 input, then the Patch Merging output. Stage 1 is evaluated on
  (k + 2)² windows (1.13× at k = 32), and stage-2's wrap copies stage-2
  windows instead of recomputing stage 1. The re-extension's translated
  copies read Patch Merging outputs beyond the stage-1 ring, which are masked.
  They cost no measurable memory.

| k (stage-1 windows) | propagated: EECU·s / peak | per-stage: EECU·s / peak |
|---:|---:|---:|
| 16 | 93, 65, 65 / **665 MiB** (3 runs) | 55, 64, 55 / **443 MiB** (3 runs) |
| 20 | 123 / 933 MiB | 80 / 665 MiB |
| 24 | 117, 120 / 1,245 MiB | 96, 123 / 933 MiB |
| 28 | memory limit | **170 / 1,245 MiB ✓** |
| 32 | memory limit | memory limit |

The reported peak is deterministic (identical across runs) and 25-33% lower
per-stage; EECU·s differences are within run-to-run noise (batch at 32 × 32:
170 vs 209; at 64 × 64: 753 vs 642). **For chunks the order reverses.** A
chunk at the raster edge needs stage-2 windows across the edge. Per-stage,
those are re-extended copies of Patch Merging outputs at the *opposite* edge,
so the request must also compute stage 1 there. Propagated, they come from
stage 1 on the extended *input*, which is cheap to read across the edge:

| 32 × 32 stage-1 windows as chunks | propagated: succeeded / EECU·s / max peak | per-stage: succeeded / EECU·s / max peak |
|---|---|---|
| 4 chunks of 16 × 16 | 4/4 / 299 / 793 MiB | **0/4** (memory limit) |
| 16 chunks of 8 × 8 | 16/16 / 605 / 349 MiB | 12/16 / 781 / 748 MiB: interior chunks 349 MiB, edge chunks 748 MiB, the 4 corner chunks fail |
| 16 full-width rows of 2 × 32 (the target validation) | 32/32 (Patch Merging and output) | the first output row request fails (memory limit) |

So the extension mode follows the execution mode. **Whole-raster requests**
(batch, the canonical mode, and single interactive requests) use
**per-stage**: lower memory, ceiling 28 instead of 24. **Chunked interactive
requests** use **propagated**: every chunk stays local. Per-stage is the
library default (`hierarchical_stage.hierarchy`); the chunk runner uses
propagated. Both are bit-identical, and per-stage is also the mechanism the
milestone asked for: an arbitrary stage input, extended explicitly, then the
existing shift.

**Cost of the extension itself** (the extended input `x`, propagated
3-window ring): 1.6 EECU·s / 69 MiB at 256 windows, 6.4 / 277 MiB at 1,024
with `mosaic`. With `where`: 8.6 EECU·s / 648 MiB at 256 windows, memory limit
at 1,024. `where` evaluates every translated copy over the whole region;
the clipped mosaic does not.

**Dependency halo** (the receptive field, independent of the mode). A
stage-2 output window needs stage-2 inputs ±1 window: stage-1 outputs ±2
windows, stage-1 inputs ±3. Measured with the propagated extension on
4 × 6 stage-1 windows:

| extension ring (stage-1 windows) | stage-1 output | stage-2 output |
|---:|---|---|
| 1 | complete, exact | 6 of 6 windows masked |
| 2 | complete, exact | 6 of 6 windows masked |
| **3** | complete, exact | **complete, exact (9.8e-6)** |

(Every stage-2 window of a 2 × 3 grid touches the raster edge.) The halo
needs no code in chunked requests: EE evaluates exactly the windows a chunk
depends on.

## 4. Patch Merging cost; window layout vs fine grid (Phases 14-15)

Patch Merging alone on a cheap generated stage-1 input (periodic, no
extension), interactive, 1,024 stage-1 windows → 256 stage-2 windows
(cumulative prefixes; EE evaluates only what a request depends on):

| prefix | EECU·s | peak | request |
|---|---:|---:|---:|
| stage-1 input | 5.7 | 294 MiB | 14 KB |
| + gather of the 2 × 2 windows (side by side, [49, 384]) | 7.2 | 312 MiB | 18 KB |
| + slab re-layout to timm's 4C order | 12.1, 9.3, 13.7 | 312 MiB | 22 KB |
| + LayerNorm(384) | 12.6 | 312 MiB | 28 KB |
| + reduction 384 → 192: **Patch Merging** | 9.3, 13.9, 17.8 | 381 MiB | 30 KB |

At 64 and 256 stage-1 windows it costs 1.1 and 4.5 EECU·s (33 and
107 MiB). The dominant profiler entry is the materialized stage-1 input
(`Image.reproject`); the gather adds ~18 MiB, the one-window-per-pixel
`[49, 384]` → `[49, 192]` arithmetic the rest.

**Direct window layout (A) vs the fine token grid (B).** B converts the stage-1
windows to a 224 × 224 token raster (Milestone 1's `depth_to_space`), takes
the four stride-2 samples of the token raster on the 112 × 112 grid, and
re-partitions into stage-2 windows (Milestone 1's 49 stride-7 samples):

| measure (1,024 stage-1 windows) | A: window layout | B: fine grid |
|---|---|---|
| numerical result | — | bit-identical to A (IDs and floats; in the hierarchy too) |
| peak (concatenation / output) | 312 / 381 MiB | 312 / 381 MiB |
| EECU·s, concatenation (3 runs) | 12.1, 9.3, 13.7 | 17.7, 18.5, 11.8 |
| EECU·s, Patch Merging output (3 runs) | 9.3, 13.9, 17.8 | 21.6, 9.8, 14.4 |
| reproject invocations in the profile | 14 | 1,498 |
| request bytes | 30 KB | 71 KB |
| hierarchy at 256 windows: EECU·s / peak | 93, 65, 65 / 665 MiB | 68, 97, 93 / 665 MiB |
| hierarchy interactive ceiling | same | same |
| layout pitfalls | none | M1's `depth_to_space` index is negative above/left of the raster (EE's `mod` takes the dividend's sign); the ring windows came back masked until fixed |

Patch Merging is under 10% of the hierarchy, so the choice moves the whole by
little. The fine grid is never cheaper. It is ~1.4× the EECU·s inside the
hierarchy (medians), far more complex, and fragile at the ring. **Window
layout remains the preferred representation** (Q30).

## 5. Stage-2 cost (Phase 14)

Stage 2 on Patch Merging of the cheap input (cumulative prefixes), 256 and
64 stage-2 windows:

| prefix | EECU·s @256 | peak @256 | peak @64 |
|---|---:|---:|---:|
| Patch Merging output | 9.5 | 381 MiB | 107 MiB |
| + LN1 | 13.2 | 381 | 107 |
| + QKV (192 → 576) | 23.6 | 493 | 138 |
| + 6-head W-MSA (bias, softmax, × V) | 21.4 | 509 | 143 |
| + projection | 25.4 | 765 | 214 |
| + residual 1 | 30.3 | 890 | 249 |
| + LN2, MLP (768, exact GELU), residual 2: **W-MSA block** | 42.0 | 1,027 | 288 |
| + LN1, shift (materialized block-A output) | 44.6 | 997 | 308 |
| + 6-head masked SW-MSA | 58.2 | 1,005 | 310 |
| + projection, reverse shift, residual 1 | 67.8 | 1,237 | 407 |
| + MLP: **SW-MSA block = stage-2 pair** | 83.5 | 1,237 | 407 |

The stage-2 pair alone fits interactively at the full 16 × 16 stage-2 grid
(1,237 MiB). The dominant entry from the shift on is the materialization of
block A's output (`Image.reproject`), whose accounting includes block A, as in
Milestone 3. In stage-2 block A, the step to the projection (6 heads'
attention evaluated beneath it) adds the most memory (+256 MiB), the MLP
+137 MiB. In Milestone 2's stage-1 block the MLP step was the largest
(+529 MiB at 1,024 windows). Compute: `erf` is the largest term of the
stage-2 pair (23 EECU·s of 84), and of the whole hierarchy (28% of
EECU·s at 24 × 24 windows; fused float arithmetic 27%, materialization reads
19%).

Relative to stage 1: the stage-2 pair costs ~74 EECU·s at 256 stage-2
windows, against Milestone 3's 107-159 for the stage-1 pair at 1,024 windows
in batch. That is 4× fewer windows at 2× the width, 6 heads and a 4× wider
MLP per window.

## 6. Asset-backed parameters and generalization (Phases 6, 16)

**Storage.** The repository's `native_ee.weight_assets` was used unchanged:
one feature per matrix row, values as 9-significant-digit JSON strings, 3
private table assets under `projects/ORIGINAL_PROJECT/assets/swin_hierarchy/`,
6,070 features, 18.0 MB of payload, uploaded in 213 s. Each record holds one
timm tensor, keyed by its full `SwinTransformer` state_dict name. Linear layers
are stored with their bias (the reduction has none), LayerNorm vectors as
`[1, C]`, bias tables as `[169, heads]`. Inside the graph each tensor is
filtered, sorted, decoded and reshaped back to an `ee.Array`; every Milestone
2/3 constructor accepts it as a `ServerArray`. A SHA-256 fingerprint per
tensor refuses assets that hold other parameters.

**Exactness (Test 13).** All 1,190,946 values read back from the assets equal
the float32 sources bit for bit (55 arrays, fetched in full). Each part
computed with asset vs inline parameters gives bit-identical EE outputs:
the stage-1 pair, Patch Merging, and stage-2 blocks A and B.

**Request size and overhead:**

| request | inline | asset-backed |
|---|---:|---:|
| complete hierarchy | 15,761,279 B: **rejected** ("Request payload size exceeds the limit: 10485760 bytes", after a 111 s upload) | **217,787 B** |
| serialization (client) | 18 s | 0.05 s |
| stage-1 pair, 4 × 4 windows: wall / EECU·s | 2.98 MB: 24 s / 4.6 | 93 KB: 4.8 s / 5.7 |
| stage-1 pair, 16 × 16 windows: wall / EECU·s / peak | 41 s / 40.8 / 495 MiB | 16 s / 35.6 / 495 MiB |
| Patch Merging, 4 × 4 windows: wall | 1.0 MB: 8.2 s | 23 KB: 1.8 s |
| stage-2 block A / B, 2 × 2 windows: wall | 5.9 MB: 43 s each | 61 / 71 KB: 4.2 / 4.5 s |
| all 1.19 M parameters read and reduced, one request | — | 82 KB: 2.1 s, 4.5 EECU·s |

Inline parameters cost request wall time (upload and parsing, ~6.5 s per MB)
rather than EECU·s. Assets cost ~1-2 EECU·s per request and no memory. The
hierarchy depends on assets: it does not fit the request limit otherwise.

**Generalization (Phase 16).** The audit found stage-1 constants in
Milestone 2's `split_heads`, `window_attention`, `mlp_branch`, `swin_block`,
Milestone 3's `shifted_swin_block`, and both mirrors (`DIM`, `HEADS`,
`HEAD_DIM` imported from the stage-1 reference). Fixes:

- `BlockConstants` now carries `dim` and `heads`, read from the parameters
  (`norm1.weight`, the bias table's columns);
- `ServerArray` parameters are accepted by `constant_value`,
  `LinearConstants` and `relative_position_bias`;
- `LinearConstants(bias=None)` gives a linear layer with no addition;
- the mirrors take dim and heads from their parameters, and share one
  LayerNorm formula with Patch Merging.

Window size 7 and shift 3 stay fixed (Swin-T). 9 Milestone 2/3 request graphs
were serialized before and after: byte-identical. The mirrors are bit-identical
on the Milestone 3 fixture, and 47 Milestone 1-3 offline tests pass. The
32 × 32 window grids were never hard-coded: grids come from the raster.

## 7. Execution: interactive, batch, chunked (Phase 11, Test 14)

Each request returns checksums of one tensor over every window, verified
against float64. "Peak" is the EE profiler's largest per-algorithm "Peak Mem"
(image entries include their subgraph), not RAM.

**Hierarchy output vs raster size, interactive**:

| stage-1 windows (tokens) | stage-2 windows | propagated: wall s / EECU·s / peak | per-stage (production): wall s / EECU·s / peak |
|---|---:|---|---|
| 4×4 (28²) | 2×2 | 9 / 16 / 134 MiB | — |
| 8×8 (56²) | 4×4 | 17 / 34 / 266 MiB | — |
| 12×12 (84²) | 6×6 | 30 / 59 / 443 MiB | — |
| 16×16 (112²) | 8×8 | 21-46 / 65-93 / 665 MiB | 22-35 / 55-64 / 443 MiB |
| 20×20 (140²) | 10×10 | 58 / 123 / 933 MiB | 35 / 80 / 665 MiB |
| 24×24 (168²) | 12×12 | 36 / 117-120 / 1,245 MiB | 34-67 / 96-123 / 933 MiB |
| **28×28 (196²)** | 14×14 | ✗ memory limit | **89 / 170 / 1,245 MiB ✓** |
| 32×32 (224²) | 16×16 | ✗ memory limit | ✗ memory limit |

At 32 × 32 the whole-raster interactive request fails in every variant tried:
propagated and per-stage, periodic input, `where`, and the fine grid. The
stage-1 pair output alone fails there too (4 of 4 attempts here;
Milestone 3: 8 of 8). Stage-1 block A alone fits (1,553 MiB) and the stage-2 pair alone
fits (1,237 MiB).

**Batch export** (the same request, checksums only):

| stage-1 windows (tokens) | mode | queue s | run s | EECU·s | status |
|---|---|---:|---:|---:|---|
| **32×32 (224² → 112²)** | propagated | 9 | 107 | **209** | ✓ verified |
| 48×48 (336² → 168²) | propagated | 8 | 149 | 376 | ✓ |
| 64×64 (448² → 224²) | propagated | 8 | 574 | 642 | ✓ |
| 32×32 | per-stage | 5 | 201 | 170 | ✓ verified |
| 64×64 | per-stage | 9 | 466 | 753 | ✓ |

~0.15-0.2 EECU·s per stage-1 window. Milestone 3's stage-1 pair took 107-159
EECU·s at 1,024 windows; the hierarchy adds Patch Merging and stage 2 for
~1.1-2× that.

**Chunked interactive requests** at 32 × 32 stage-1 windows (output
checksums per chunk, summed):

| chunks | stage-1 windows each | mode | succeeded | max peak | EECU·s total | wall s total (sequential) | combined checksum | stage-1 block A recomputed |
|---:|---|---|---:|---:|---:|---:|---|---:|
| 4 | 16 × 16 (256) | propagated | 4/4 | 793 MiB | 299 | 114 | ✓ = whole-raster float64 | 1.89× |
| 16 | 8 × 8 (64) | propagated | 16/16 | 349 MiB | 605 | 279 | ✓ | 3.06× |
| 4 | 16 × 16 | per-stage | **0/4** (memory limit) | — | — | — | — | — |
| 16 | 8 × 8 | per-stage | 12/16 (corner chunks fail) | 748 MiB | 781 | 506 | — | — |
| 32 | 2 × 32 (one stage-2 window row) | propagated | 32/32 (the target validation: 16 Patch Merging + 16 output rows) | — | — | ~16-25 per request | every output element compared (§1) | |

The halo is the dependency halo (3 stage-1 windows), wrapped at the raster
edge. A chunk recomputes stage-1 block A on (n + 6)² windows and stage-1
block B, Patch Merging and stage-2 block A on (n + 4)²; stage-2 block B
runs on the chunk only. That recomputation is visible in EECU·s: 299 and 605
vs 209 in batch. These are propagated-mode chunks. Per-stage chunks also
compute stage 1 at the opposite raster edge (§3), which doubles the edge
chunks' peak and puts the corner chunks over the limit.

## 8. Graph size and composition

| request | bytes | invocations after dedup | distinct functions | serialize s |
|---|---:|---:|---:|---:|
| hierarchy, asset parameters, 4 or 1,024 windows | **217,787** | **1,695** | 69 | 0.05 |
| hierarchy, inline parameters | 15,761,279 | 1,274 | 58 | 18 |
| parameters inline: stage-1 block / Patch Merging / stage-2 block | 1.46 MB / 0.98 MB / 5.85 MB | — | — | — |

The asset reads account for the 421 additional invocations (57 filters,
55 aggregates and their decoding). The graph is constant in raster size
and in parameter count. No request was rejected as "too complex". The
hierarchy is the composition of the unchanged Milestone 2/3 subgraphs, two
`PatchMerging`-sized additions and two periodic extensions.

## 9. Earth Engine behavior found or confirmed (probed live)

1. Stride sampling (`reproject` to a projection whose pixel centres are
   chosen coarse-pixel centres, then `changeProj`) selects array-valued pixels
   exactly at any coarse factor, here 2 between window grids.
2. `Image.displace` rejects array-valued images ("Cannot use array-valued
   image here"); it cannot implement the cyclic wrap.
3. `where` keeps masked input pixels masked, so it cannot fill the ring of a
   footprint-masked input. It also evaluates every translated copy over the
   whole region (648 MiB vs 69 MiB at 256 windows). `mosaic` of copies
   clipped to their strips is correct for masked inputs and cheap.
4. Masked out-of-domain pixels appear to cost little: per-stage re-extension
   reads translated Patch Merging outputs that are masked beyond the stage-1
   ring, and measured cheaper (memory) than recomputing a wider ring.
5. EE's `mod` takes the dividend's sign (Milestone 3): any index
   computed with `mod` on pixel coordinates must be made non-negative before
   it is used on the ring. Milestone 1's `depth_to_space` was not.
6. Inline parameters cost request wall time (~6.5 s per MB: upload and
   parsing), not EECU·s. The payload limit is 10,485,760 bytes.
7. As in Milestones 2-3: tileScale does not split a raster that is one
   coarse tile; batch export lifts the interactive memory limit.

## 10. Recommendation: proceed to Milestone 5

**Proceed.** The evidence on each criterion:

- **Patch Merging correctness**: exact layout (token provenance, 4C order),
  bit-identical arithmetic, 5.5e-6 at the target.
- **Stage-2 correctness**: 6-head W-MSA and SW-MSA, MLP 768, every
  intermediate bit-identical to the specification.
- **Hierarchical composition**: one EE graph from stage-1 tokens to stage-2
  output, 1.13e-5 at the target; 30.3 M elements, zero mirror differences.
- **Window-layout efficiency**: the hierarchical transition is a 2 × 2
  coarse-window gather; the fine grid is never cheaper.
- **Asset-backed weights**: exact, 72× smaller requests, mandatory beyond
  ~0.79 M parameters.
- **Periodic extension**: exact for arbitrary (masked) inputs; per-stage
  extension costs a one-window ring per stage.
- **Memory scaling**: interactive up to 784 stage-1 windows (the same
  ceiling as Milestone 3's stage-1 pair alone).
- **Batch feasibility**: the target in ~2 min, 4× the target in ~10 min.
- **Chunked feasibility**: 256-window chunks with a 3-window halo, 1.4× batch
  EECU·s (propagated extension; per-stage extension is unsuited to chunks).
- **Graph growth**: constant in raster size and parameter count with assets
  (~60-90 KB of graph per block).
- **EECU cost**: ~0.2 EECU·s per stage-1 window for both stages.

For a multi-stage encoder (Swin-T stages 3-4: dims 384 and 768, depth 6 and 2):

1. **Keep weights in assets.** Swin-T stage 3 alone has 10.96 M parameters
   (9× this hierarchy's) and stage 4 15.37 M; inline requests are impossible
   beyond ~0.79 M (13.2 bytes per parameter against the 10 MiB limit).
2. **Plan batch as the canonical execution mode.** Each later stage has 4×
   fewer windows at 2× the width, so memory should not grow beyond stage 1's.
   But a deeper stage 3 (6 blocks) lengthens one graph, and interactive
   requests remain bounded by stage 1 at full resolution.
3. **Choose the extension mode by execution mode.** Per-stage for
   whole-raster and batch requests: each stage's input is extended by one
   window, so a stage's ring cost does not compound across stages. Propagated
   for spatial chunks: an edge chunk then never computes stage 1 at the
   opposite raster edge. A deeper encoder makes that choice matter more,
   because the propagated ring grows by 2^(s−1) stage-1 windows per stage.
4. **Chunk with the dependency halo:** each SW-MSA block at stage s adds
   2^(s−1) stage-1 windows. Deeper stages make chunks of the full encoder
   expensive (the receptive field grows), so batch is preferable there.
5. **Keep the mirror test for each new operation** (stage-3 Patch Merging
   384 → 768, deeper stages; the final norm when added).

## 11. Caveats

- **Profiler semantics.** "Peak Mem" is EE's per-algorithm accounting, not
  RAM; its attribution changes with materialization, so peaks are compared
  only within one graph shape. "Ceiling" means the largest success and
  smallest failure observed; EE does not document the per-request limit.
- **Shared service.** EECU·s and wall time vary up to ~1.5× between identical
  runs (hierarchy at 256 windows: 65-93 EECU·s). Peaks were identical across
  repeats. Single observations are indicative.
- **Inputs and parameters.** Generated hash tokens and seeded parameters,
  not a trained model or real imagery. The input is non-periodic and masked
  outside the raster, as imagery would be, but its values inside are
  synthetic.
- **Scope.** Swin-T stages 1-2 without patch embedding (the hierarchy starts
  from a stage-1 token raster). Stage-1 rasters must have an even number
  (≥ 4) of windows per side; timm's padding of odd rasters is documented but
  not implemented. No stage 3-4, final norm, decoder, or Sentinel-1 inference.
- **Assets.** The weight assets and the batch-checksum tables
  (`ee_attention_benchmarks/`) remain in project `ORIGINAL_PROJECT`. Delete the
  checksum tables when finished; the weight assets are needed to reproduce.
