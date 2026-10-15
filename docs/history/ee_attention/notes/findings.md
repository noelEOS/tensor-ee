# Findings: exact window attention in native Earth Engine

Live experiments on 29 September 2026: Earth Engine Python API 1.7.45, PyTorch
2.14.0, NumPy 2.5.3, Cloud project `ORIGINAL_PROJECT`, application-default
credentials. Every attention operation ran inside EE computation graphs; no
Vertex AI, Cloud Run, model server, or exported imagery. Verification fetched
EE-computed tensors for small cases and scalar checksums for stress runs.

Evidence: numerical_comparison.csv (development archive: `docs/history/ee_attention/results/numerical_comparison.csv`)
(all phases), benchmark_results.csv (development archive: `docs/history/ee_attention/results/benchmark_results.csv`) (every
stress run, successes and failures), per-run raw records with full EE profiles
in results/raw/ (archive `ee_attention/results/raw/` at `research-archive-2026-10-12`), and the live semantics probes in
capability_probes.json (archive `ee_attention/results/raw/capability_probes.json` at `research-archive-2026-10-12`).

## Answers

| # | Question | Answer |
|---|---|---|
| 1 | Can exact attention be reproduced in EE? | **Yes.** Naive and online-softmax (blockwise) attention run entirely in EE graphs, both on single windows (`ee.Array`) and on token rasters (array-valued pixels), and match PyTorch at every element and pixel tested (§1, §3). |
| 2 | Which EE primitives were required? | `matrixMultiply`, transpose, `reduce` (max, sum), `repeat`, `exp`, element-wise arithmetic and `max`, `slice`, `cat`, float casts; for rasters, array images (`toArray`, `arrayReshape`, `arrayProject`, `arrayFlatten`), `neighborhoodToArray`, per-pixel `arraySlice`, `pixelCoordinates`, `changeProj`/`reproject`; `ee.List.iterate` for a server-side loop (§2). |
| 3 | How closely does EE match PyTorch? | Max abs output error 8.3e-7 (d=32) and 4.2e-7 (d=64); MAE ≈4e-8, RMSE ≈6e-8; target was 1e-4. EE's `matrixMultiply` is correctly rounded, so the gap is PyTorch's float32 rounding at QKᵀ; EE is 4–8× closer to float64. EE's `matrixMultiply` and `exp` return double and must be cast back (§1). |
| 4 | Can local window attention be expressed with `neighborhoodToArray()`? | **Yes for sliding windows** (one gather per tensor, out-of-raster keys excluded). Swin's aligned windows need a (2w−1)² gather plus per-pixel `arraySlice`, or a space-to-depth window layout, which is the efficient route (§3). |
| 5 | Does blockwise online-softmax reduce memory-limit failures? | **No, not within an EE request.** In every interactive ladder (one window with three tilings including a server-side loop, six neighborhood configurations plus a reproduced fine boundary, window partition, window layout) blockwise failed at the same size as naive. EE's profiles show every intermediate still held when a request fails. What does move limits: fewer temporaries, batch export, smaller tiles for rasters with many small per-pixel arrays, and a better formulation (§4). |
| 6 | Does it change runtime? | It does not help: value-path tilings are within noise (EECU ×0.86–1.08); on rasters blockwise costs 5% (window layouts) to 26–32% (neighborhoods) more EECU (§5). |
| 7 | Largest problem tested successfully? | Interactive: one window of 7,056 tokens as arrays (7,744 with fewer temporaries); 4,096 tokens per window as an image pixel; 4,096 query pixels × 2,025 keys (neighborhoods); a Swin-T-stage-1-sized layer on 224×224 tokens. Batch: one window of 12,544 tokens; 4,096 query pixels × 5,041 keys at d=128 (§6). |
| 8 | Are graph-size or serialization limits important? | **Yes.** The request payload limit is 10,485,760 bytes (embedded Q/K/V hit it near 3×4,096×64 values), and a separate "Computation is too complex" limit, shared by batch, stopped the O(w²) stride-sample layout at w=64. Unrolled blocks grow the graph linearly; `iterate` keeps it constant (§7). |
| 9 | Is a small Swin-style encoder worth attempting next? | **Yes, in window layout.** A Swin-T stage-1 attention layer (224×224 tokens, window 7, 3 heads) ran interactively in ≈5 s; per-pixel forms needed tileScale 16 and 15–25× the compute. Blockwise attention is unnecessary at 49-token windows (§8). |

## 1. Numerical reproduction (Phases 1, 2, 4)

Fixtures: 7×7 window (N = 49 tokens), one head, d = 32 and 64, `torch.randn`
float32 Q, K, V with fixed seeds (data/manifest.json (archive `ee_attention/data/manifest.json` at `research-archive-2026-10-12`)).
EE received the exact float32 values as constants. Both the `ee.Array` value
path and a single-pixel array-image path were run; after one correction below
they are **bit-identical at every intermediate**.

| Output tensor, max abs error | vs PyTorch float32 | vs float64 | vs naive EE |
|---|---:|---:|---:|
| Naive, d=32 (MAE 4.0e-8, RMSE 6.4e-8 vs PyTorch) | 8.34e-7 | 1.26e-7 | — |
| Naive, d=64 (MAE 4.6e-8, RMSE 6.4e-8) | 4.17e-7 | 9.12e-8 | — |
| Blockwise, K/V blocks 0–15/16–31/32–48, d=32 | 7.15e-7 | 1.11e-7 | 1.19e-7 |
| Blockwise, same blocks, d=64 | 4.17e-7 | 1.08e-7 | 1.19e-7 |
| Blockwise, seven blocks of 7, d=32 / 64 | 7.15e-7 / 4.77e-7 | 1.75e-7 / 1.42e-7 | 1.79e-7 / 1.19e-7 |
| Blockwise, 49 single-token blocks, d=32 / 64 | 8.34e-7 / 4.77e-7 | 3.33e-7 / 4.99e-7 | 2.98e-7 / 4.77e-7 |

All 228 rows of the numerical comparison (174 array rows from Phases 2 and 4,
54 raster rows from Phase 3) pass the 1e-4 target; the largest attention-output
error anywhere is 8.34e-7. The online-softmax running maxima
are **bit-identical** to the naive row maxima in every blockwise case, and the
normalizers agree to float32 rounding.

**Where divergence enters.** It enters at QKᵀ and is PyTorch's rounding, not
EE's. EE's `matrixMultiply` returned exactly the float64 dot product rounded to
float32 for 100% of the 2,401 scores; PyTorch's float32 GEMM did so for 28.7%
(d=32) and 20.8% (d=64). At every stage EE is 4–8× closer to float64 than
PyTorch float32 is, and |EE − PyTorch| ≈ |PyTorch − float64|:

| d=32 stage | \|EE − float64\| | \|PyTorch32 − float64\| | \|EE − PyTorch32\| | magnitude |
|---|---:|---:|---:|---:|
| QKᵀ | 9.4e-7 | 3.4e-6 | 3.8e-6 | 22.7 |
| / √d | 2.2e-7 | 7.3e-7 | 7.2e-7 | 4.0 |
| softmax | 5.2e-8 | 2.0e-7 | 2.1e-7 | 0.40 |
| output | 1.3e-7 | 7.5e-7 | 8.3e-7 | 1.1 |

Precision details that matter for anyone reproducing this (all probed live):

- `matrixMultiply` and `exp` on float32 operands **return double**; a float32
  implementation must cast back (`.float()` / `.toFloat()`) after each. Sums
  also accumulate above float32 precision.
- `ee.Array` arithmetic with a non-integer Python number, or with
  `ee.Number(x).float()`, promotes to double; with an integer it stays float32.
- An image divisor `ee.Image.constant(√d).toFloat()` still divided by the
  unrounded double √d (78.5% of quotients differed in the last bit from IEEE
  float32 division). Supplying float32(√d) as the constant fixed it and made
  the image path bit-identical to the value path.

## 2. Earth Engine primitives used

- Array algebra: `matrixMultiply`, `transpose`/`arrayTranspose`,
  `reduce`/`arrayReduce` (max, sum), `repeat`/`arrayRepeat`, `exp`, `max`,
  `subtract`, `multiply`, `divide`, `add`, `slice`/`arraySlice`,
  `Array.cat`/`arrayCat`, `float`/`toFloat`.
- Array images: `ee.Image(ee.Array)`, `toArray(axis)`, `arrayReshape`,
  `arrayProject`, `arrayFlatten`, `arrayGet`.
- Spatial gathering: `neighborhoodToArray` with `Kernel.square` /
  `Kernel.fixed`, per-pixel `arraySlice(axis, startImage, endImage)`,
  `pixelCoordinates`, `changeProj` (exact integer shifts and space-to-depth
  relabeling), `reproject`, `updateMask` / `unmask`.
- Loops: unrolled Python-side graphs, and `ee.List.iterate` for a server-side
  block loop.
- Measurement: `reduceRegion`, `ee.data.computePixels`, `Export.table.toAsset`,
  and the hidden `Profile.getProfiles` (JSON format) for per-algorithm
  EECU-seconds and peak memory.

No softmax primitive exists; the numerically stable form is written out with
max, subtract, exp, sum and divide.

## 3. Attention on a token raster (Phase 3)

Tokens are pixels of a fine grid; each token's q, k and v are d float32 bands.
Heads split channels contiguously. Four exact formulations were built on the
same naive/blockwise code:

| Formulation | How a query meets its keys | Semantics | Per-pixel gather | Graph size vs w |
|---|---|---|---|---|
| **F2a** centered neighborhood | one `neighborhoodToArray` (w×w, centered) | sliding window (neighborhood attention) | N·d for K and for V | constant |
| **F2b** per-pixel Swin slice | `neighborhoodToArray` (2w−1)², then `arraySlice` with per-pixel start (w−1)−(y mod w), (w−1)−(x mod w) | Swin W-MSA (unshifted) | (2w−1)²·d ≈ 4N·d | constant |
| **F3** window partition | space-to-depth onto a w×-coarser grid: w² stride samples (`reproject` to a shifted coarse grid, then `changeProj`), or one top-left-anchored `neighborhoodToArray` plus one stride sample (`naive_nta`) | Swin W-MSA; one [N, N] matrix per window | none per query (samples); N·d at every fine pixel (nta) | O(w²) (samples); constant (nta) |
| **Window layout** | Q/K/V generated directly in window layout on the coarse grid | Swin W-MSA core without layout conversion | none | constant |

Every pixel of every case was compared with PyTorch (all 25 checks pass):

| Case | Max abs error vs PyTorch float32 / float64 |
|---|---|
| Phase 1 tensors as a 7×7 raster: F2b, F3 (both space-to-depth forms) | 8.34e-7 / 1.26e-7 — **bit-identical to the Phase 2 value path** |
| same raster, F3 blockwise (four K/V blocks) | 7.15e-7 / 1.73e-7 |
| same raster, F2a naive / blockwise-gather / blockwise-slice | 3.58e-7 / 2.98e-7 / 2.98e-7 (vs float64 ≤ 1.47e-7) |
| 21×21, 2 heads × 32, window 7 (nine windows, all border cases), all forms incl. window layout | ≤ 3.28e-7 / ≤ 1.30e-7 |
| 16×24, even window 8, 2 heads × 16 (Swin forms incl. window layout) | ≤ 2.38e-7 / ≤ 6.54e-8 |
| 13×19, window 5 (F2a forms) | ≤ 2.09e-7 / ≤ 1.18e-7 |

What had to be learned about `neighborhoodToArray` (each probed live):

- Axis 0 is y and axis 1 is x; bands concatenated with `toArray(2)` give
  [h, w, d], reshaped row-major to [N, d].
- Masked neighbors are filled with the default value, so out-of-raster keys
  must be **excluded**, not zero-padded: a gathered validity image adds a
  finite −1e30 to their scores (JSON cannot carry −∞; `exp` underflows to 0).
- The output mask equals the input's mask at the **center** pixel. After a
  row shift that center may lie outside the raster, masking a valid query:
  the first blockwise-gather run had errors of 1.24 until K/V were
  zero-extended with `unmask(0, false)`.
- `Kernel.fixed` silently clamps a focus outside the kernel, so an off-center
  block of window rows needs an exact integer shift via `changeProj`
  (`new(y,x) = old(y+dy, x+dx)`), which was verified live.
- `arraySlice` accepts per-pixel start/end images, which is what makes Swin
  partition windows expressible per pixel.

**Q4 answer:** yes for sliding windows — one `neighborhoodToArray` expresses
"token raster → w×w neighborhood → N tokens → attention → updated center
token" directly. Swin's aligned windows need either the (2w−1)² gather plus a
per-pixel slice (about 4× the gather) or a space-to-depth layout (F3), which
either avoids `neighborhoodToArray` (stride samples, graph O(w²)) or uses it
once with w²-fold redundant work (constant graph).

## 4. Does blockwise attention change Earth Engine's limits? (Phase 5)

Protocol. Sizes increase until a series' first failure. Within a comparison,
naive and blockwise use identical generated inputs, raster, tileScale and
execution mode (all part of each run's label). Every success is verified
against a float64 reference (value path: 12 sampled entries < 1e-5 plus
weighted and sum-of-squares checksums < 1e-6 relative; rasters: weighted and
sum-of-squares checksums < 1e-5 relative plus exact pixel counts); every
recorded success passes. Memory figures are **EE's own profiler numbers**:
"Peak Mem" per algorithm (for image tiles, inclusive of the subgraph computed
beneath it). They are not request-level RAM measurements, and EE does not
document its per-request memory limit.

EE-reported memory versus tokens per window (archive `ee_attention/results/memory_vs_size.svg` at `research-archive-2026-10-12`)

**Key comparison** (interactive unless noted; details in §4.1–4.5):

| | naive | blockwise (online softmax) |
|---|---|---|
| Correctness | exact; output ≤ 8.3e-7 from PyTorch float32 | exact; output ≤ 8.3e-7; row maxima bit-identical to naive |
| Largest single window, `ee.Array` (d=32) | N=7,056 | N=7,056 (all three tilings) |
| Largest window, array-valued pixel (d=32) | N=4,096 | N=4,096 |
| Largest neighborhood, 64×64 tokens (d=32) | w=45 (N=2,025), reproduced ×3 | w=45 (block gathers and sliced); one-row blocks fail at w=45 |
| Memory-limit errors | at every boundary above | at the same sizes, in every ladder |
| EE-reported peak | baseline | value path ×1.02–1.03; neighborhoods: block gathers ×1.12, sliced ×0.98; window layouts ×1.00–1.02 (medians) |
| Runtime (EECU·s) | baseline | value path ×0.86–1.08; neighborhoods ×1.26–1.32; window layouts ×1.05 |
| Graph size | constant in N (≈14 KB value path, ≈20 KB rasters) | +3–6 KB per unrolled block; constant with `iterate` |
| Batch export | lifts the interactive ceiling | lifts it equally (§4.5) |

### 4.1 One window as `ee.Array` values

| d, heads | naive | blockwise, K/V blocks of 1024 | blockwise, Q and K/V tiles of 2048 | blockwise, server-side `iterate` | lean naive |
|---|---|---|---|---|---|
| 32, 1 | ✓ 7,056 · ✗ 7,744 | ✓ 7,056 · ✗ 7,744 | ✓ 7,056 · ✗ 7,744 | ✓ 7,056 · ✗ 7,744 | ✓ 7,744 · ✗ 8,464 |
| 128, 1 | ✓ 4,096 · ✗ 9,216 | ✓ 4,096 · ✗ 9,216 | ✓ 4,096 · ✗ 9,216 | — | — |
| 32, 4 | ✓ 1,024 · ✗ 4,096 | ✓ 1,024 · ✗ 4,096 | ✓ 1,024 · ✗ 4,096 | — | — |

(tokens per window N; ✓ largest success, ✗ first failure, all `User memory limit
exceeded`.) The d=32 boundary was refined with windows of 64² to 92².

- **Direct evidence of retention.** In every failed request, EE's profile
  reports "Current Mem" equal to "Peak Mem" for every algorithm: all
  intermediates were still held when the request died (2.50–2.95 GiB summed
  across algorithms). The largest summed peak among successes was 2.50 GiB
  (naive, N=7,056).
- **Tiling re-partitions the same total.** At N=7,056 the summed peaks were
  2.50 GiB (naive) versus 2.35–2.36 GiB (all three blockwise forms); at the
  failing N the blockwise requests held the same ~2.5 GiB as naive.
- **The server-side loop behaves the same.** `ee.List.iterate` keeps its
  per-block arrays for the rest of the request too.
- **What moved the boundary:** fewer N×N temporaries. Folding 1/√d into Q and
  normalizing after P·V (28 instead of 44 bytes per score under the float32
  policy) cut the summed peak to 1.76 GiB at N=7,056 and succeeded at 7,744.

### 4.2 Token raster: centered neighborhoods (F2a)

64×64 tokens (4,096 query pixels), tileScale 1, interactive `reduceRegion`;
blocks of ⌈w/4⌉ window rows.

| d × heads | largest success, all three variants | first failure, all three | largest EE peak at last success: naive / block gathers / sliced blocks (GiB) |
|---|---|---|---|
| 32 × 1 | w=41 (N=1,681) | w=51 | 1.89 / 1.93 / 1.81 |
| 64 × 1 | w=31 (N=961) | w=35 | 2.04 / 2.11 / 1.98 |
| 96 × 1 | w=25 (N=625) | w=31 | 1.95 / 2.06 / 1.91 |
| 128 × 1 | w=21 (N=441) | w=25 | 1.83 / 1.97 / 1.79 |
| 32 × 2 | w=31 (N=961) | w=35 | 2.06 / 1.98 / 1.98 |
| 32 × 4 | w=21 (N=441) | w=25 | 1.82 / 1.87 / 1.87 |

- In all six coarse ladders, **naive and both blockwise variants fail at
  the same window**. The boundary follows the per-pixel
  gather N·d (successes up to N·d ≈ 54–62k, failures from ≈ 80k): heads only
  matter through total channels.
- The faithful FlashAttention analog (gathering K/V block by block after an
  exact row shift) has a **higher** EE-reported peak than naive (median +12%)
  and costs more compute (median +26% EECU·s); slicing blocks out of the full
  gather is within a few percent of naive.
- **Fine boundary, reproduced (d=32).** A finer ladder (w = 43…49) once let
  block gathers succeed at w=47 where naive and sliced blocks failed. Three
  interleaved repetitions of w=45/47/49/51 did not reproduce it:

  | window | naive | block gathers (⌈w/4⌉ rows) | sliced blocks | block gathers, 1 row per block |
  |---|---|---|---|---|
  | 45 (N=2,025) | ✓ ✓ ✓ | ✓ ✓ ✓ | ✓ ✓ ✓ | ✗ ✗ ✗ |
  | 47 (N=2,209) | ✗ ✗ ✗ | ✗ ✗ ✗ | ✗ ✗ ✗ | ✗ ✗ ✗ |
  | 49, 51 | ✗ | ✗ | ✗ | ✗ |

  The w=47 block-gather success was 1 of 4 attempts in total: noise at the
  limit, not a shifted boundary. The finest blocking is worse than none: it
  fails at w=45, where every other variant succeeds.
- Failed raster requests end after ≈0.03 EECU·s and carry no pixel-computation
  profile, so there is no raster counterpart to the value path's "Current Mem"
  evidence. The identical boundaries are *consistent with* the same behavior;
  they do not prove it.
- **Smaller aggregation tiles did not help here.** Re-running d=128, w=25
  with tileScale 4 and 16 failed the same way for naive and block gathers
  (per-pixel K/V arrays of ~640 KB). On the 224×224 Swin-like raster, where
  per-pixel arrays are ~37 KB but pixels are many, tileScale 16 did rescue
  every per-pixel form (§8).

### 4.3 Swin window partition (F3) and per-pixel Swin slice (F2b)

Fixed-raster families (side 64: w 8/16/32/64; side 56: w 7/14/28/56), d=32,
d=128 and 4 heads × 32; blockwise uses four K/V blocks.

| Formulation | Outcome |
|---|---|
| F3, w² stride samples (naive and blockwise) | ✓ w=32 (N=1,024) and w=28; ✗ w=56 `User memory limit exceeded` (2.43 GiB reported) and ✗ w=64 `Computation is too complex` (request ≈10.0 MB), for every d and heads |
| F3, one `neighborhoodToArray` (`naive_nta`) | ✓ every window, including w=56 (0.74–0.80 GiB) and w=64 (N=4,096; 1.26–1.28 GiB), with 22–33 KB requests |
| F2b per-pixel slice, d=32 | ✓ every window tried (w=32 on 64×64, w=28 on 56×56; 1.11 GiB at w=32) |

- The stride-sample layout, not attention, limits F3: ≈600 bytes and one
  reprojection per sample, w² samples per tensor.
- Where the N×N score matrix dominates (w=28–32), blockwise lowers the
  EE-reported peak by 8–37% (e.g. 0.45 → 0.28 GiB at w=32, d=32), but both
  variants still succeed and fail at the same windows. The saving most
  likely comes from assembling each K/V block from a subset of the stride
  samples (fewer bands per stacking node), not from attention itself: with
  no stacking, in window layout, blockwise does not lower the peak (§4.4).

### 4.4 Swin attention core in window layout

Q, K, V generated directly in window layout (one coarse pixel holds one
window's [N, d] arrays), one window per raster, d=32, interactive: the
image-pixel analog of the value-path test, without any layout conversion.

| Variant | largest success | first failure | largest EE peak at N=4,096 |
|---|---|---|---|
| naive | w=64 (N=4,096) | w=72 (N=5,184), memory | 1.31 GiB |
| blockwise, four K/V blocks generated block by block | w=64 (N=4,096) | w=72 (N=5,184), memory | 1.32 GiB |

Even here, where the [N, N] score matrix dominates and K/V blocks are
produced block by block, the blockwise form neither lowers the reported peak
(within +1% to +9% of naive at every size from N=256 to 4,096) nor admits a
larger window. The same single window fits a larger N on the
value path (7,056) than as an array-valued pixel (4,096).

### 4.5 Batch export

Every failure in the main interactive ladders (neighborhoods, window
partition, window layout) was replayed as a batch task (`Export.table.toAsset`
of the same verification scalars; no imagery), starting at the failing size
and climbing until a batch failure. Not replayed: the sliced-block d=32
series, the tileScale sweep, the Swin-like per-pixel runs, and the
fine-boundary repetitions.

| Replayed failure | Batch outcome (all verified) |
|---|---|
| Neighborhoods, d=128 / d=96 / 4×32 at w=25–31 (naive and both blockwise) | ✓ every variant at every replayed window |
| Window partition, stride samples, w=56 (memory), naive and blockwise; `naive_nta` 4×32 at w=56 and w=64 | ✓ |
| Window partition, stride samples, w=64 (`Computation is too complex`) | ✗ rejected at submission: batch shares the complexity limit |

Extended batch ladders, naive versus the FlashAttention analog (sliced
blocks, an ablation, were not extended):

| Series (batch) | Tested | naive | blockwise |
|---|---|---|---|
| Neighborhoods, 64×64 tokens, d=128 | w = 25, 31, 41, 51, 61, 71 | ✓ all (w=71 in 438 s) | ✓ all (w=71: N·d ≈ 645k per pixel, ~8× the interactive ceiling; 654 s) |
| Neighborhoods, 64×64 tokens, d=32 | w = 51, 61, 71 | ✓ all | ✓ all |
| Window layout, one window, d=32 | N = 5,184 … 12,544 (w = 72 … 112) | ✓ all | ✓ all |

No batch run of either variant hit a memory limit within these ranges, so
batch did not separate naive from blockwise either; it raised both ceilings
beyond the largest sizes tried. Batch tasks queued for 3 s to 4 min (median 5 s;
longer when three ran at once) and ran for 46–714 s (median 131 s).

Batch export therefore lifts the interactive memory ceiling substantially,
for naive and blockwise alike, but not the graph-complexity limit.

### 4.6 Why blockwise does not help inside one EE request

FlashAttention saves memory because its executor keeps one bounded buffer
per loop iteration and overwrites it: only the running statistics survive a
block. EE evaluates a dataflow graph whose every node output is materialized.
Within a request, the value-path profiles show those outputs had **not been
released** when the request failed (current memory equals peak memory for every
algorithm at failure), and the server-side `iterate` loop behaves the same.
Splitting K/V into blocks therefore only re-partitions the same total across
more nodes; the running state (m, l, O) and any shifted copies are extra.
On rasters the dominant term is not the score matrix at all but the per-pixel
K/V gather (N·d per query pixel), which the key-axis blocking cannot remove
unless every block's gather is freed, and the measured boundaries show no
such benefit. Blocking over keys never moved a boundary. What did:
removing temporaries (value path), batch execution, smaller aggregation tiles
when the per-pixel arrays are small and the pixels many (224×224 Swin-like
raster), and above all a formulation whose intermediates are smaller (window
layout instead of per-pixel gathers). The axis that can be blocked
profitably in EE is the query/pixel axis, through EE's own tiling, not the
key axis inside one graph.

Not tested here: splitting one attention computation across *separate*
requests or batch tasks, with the running state staged in EE assets between
them. That is the only EE analog of FlashAttention's bounded working set,
at the cost of one export per block.

## 5. Runtime

Wall-clock times come from a shared service and vary run to run; EECU·s
(EE's compute accounting) is the steadier measure. Medians of the
blockwise/naive ratio over matched successful runs:

| Formulation | Variant | EECU·s ratio | wall-clock ratio | EE peak ratio |
|---|---|---|---|---|
| value path | K/V blocks of 1024 | 0.92 | 0.99 | 1.03 |
| value path | Q and K/V tiles of 2048 | 0.86 | 0.77 | 1.02 |
| value path | server-side `iterate` | 1.08 | 0.80 | 1.03 |
| value path | lean naive (not blockwise) | 0.72 | 0.50 | 1.00 |
| neighborhood rasters | block gathers | 1.26 | 0.81 | 1.12 |
| neighborhood rasters | sliced blocks | 1.32 | 0.99 | 0.98 |
| window partition | K/V blocks | 1.05 | 0.99 | 1.00 |
| window layout | K/V blocks | 1.05 | 0.66 | 1.02 |
| window partition | one `neighborhoodToArray` vs stride samples (both naive) | 0.64 | 0.25 | 1.20 |

Blockwise does not buy speed: at best it is within noise on the value path;
on rasters it costs 5% (window layouts) to 26–32% (neighborhoods) more
compute. Interactive requests near the boundaries took roughly 10–75 s;
batch tasks queued for a median 5 s and ran for 46–714 s (median 131 s).

## 6. Largest problems that succeeded

All verified against float64 references.

| Setting | Largest success |
|---|---|
| One window as `ee.Array` values, interactive | N=7,056 tokens with the specified float32 formulation (49.8M scores, d=32); N=7,744 with the lean formulation (60.0M scores); N=4,096 at d=128 |
| One window as an array-valued pixel (window layout), interactive | N=4,096 (d=32) |
| Neighborhoods on 64×64 tokens, interactive | 4,096 queries × 2,025 keys (w=45, d=32); 4,096 × 441 at d=128; 4,096 × 961 at 64 channels (d=64, or 2 heads × 32) |
| Window partition on 64×64 tokens, interactive | one window of N=4,096 (w=64) via one `neighborhoodToArray` |
| Swin-stage-1-like layer, interactive | 224×224 tokens = 1,024 windows × 49 tokens, 3 heads × 32, window partition at default tiles; per-pixel forms at tileScale 16 |
| Batch export | one window of N=12,544 tokens as an array-valued pixel (157M scores; naive and blockwise); neighborhoods of 4,096 queries × 5,041 keys at d=128 (w=71; naive and block gathers); window partition N=3,136 with stride samples and N=4,096 with one `neighborhoodToArray` (4 heads) |

## 7. Graph size and serialization limits

- **Request payload: 10,485,760 bytes**, stated by EE ("Request payload size
  exceeds the limit"). Embedding Q, K, V as JSON constants hits it between
  N=4,096 (d=64, 10,419,723 bytes, passed with 66 KB to spare) and N=8,281
  (18.6 MB). Embedded requests also cost ≈6 s of wall time per MB (62 s at
  10.4 MB versus 6–9 s for the same attention with server-generated inputs).
  Weights or tokens should therefore come from assets or be generated on the
  server, never inlined at scale.
- **"Computation is too complex."** A separate graph-complexity limit: the
  w² stride-sample space-to-depth at w=64 (≈16k reprojection chains, 10.0 MB)
  was rejected although it was under the payload limit.
- **Growth rates.** Unrolled blockwise graphs grow linearly with the number of
  blocks: ≈3.1 KB per block with embedded K/V (Phase 4: 100.5 KB for one block,
  250.6 KB for 49), ≈6.3 KB per block when each block also carries its own K/V
  generator (Phase 5: 34.0 KB for 4 blocks, 65.3 KB for 9; 97.2 KB for 25 Q×K/V
  tiles; naive 13.9 KB); the `iterate` form stays constant (24–31 KB); stride-sample
  space-to-depth grows by ≈600 bytes per sample and per tensor (O(w²));
  the one-`neighborhoodToArray` layout and the generated window layout are
  constant (≈22–33 KB). The client serializer deduplicates identical
  subexpressions by content hash, which keeps shared blockwise state from
  being re-serialized.
- **Client recursion.** Deep graphs need a raised Python recursion limit for
  serialization (set to 50,000 here, as in the earlier ResNet18 work).

## 8. Is a small Swin-style encoder worth attempting next?

**Yes, in window layout.** A Swin-T stage-1-sized attention layer
(224×224 tokens, an 896×896 image at 4×4 patches; window 7; 3 heads × 32; 1,024
windows) was run in every formulation, interactively, and verified:

| Formulation | tileScale 1 | tileScale 4 | tileScale 16 |
|---|---|---|---|
| F3 window partition, naive / blockwise / one `neighborhoodToArray` | ✓ 4.8 / 5.1 / 15.3 s; 35 / 37 / 54 EECU·s; 1.18–1.22 GiB | — | — |
| F2b per-pixel Swin slice | ✗ memory | ✗ memory | ✓ 18.9 s, 766 EECU·s, 0.16 GiB |
| F2a neighborhoods, naive / block gathers / sliced blocks | ✗ memory (all) | ✗ memory (all) | ✓ 11–13 s, 526–875 EECU·s, 0.16–0.27 GiB |

Window partition does the layer at default tiles with **15–25× less compute**
than the per-pixel forms, which need tileScale 16. With Swin's 49-token
windows the score matrices are tiny, so blockwise attention buys nothing
here; the attention core alone holds up to N=4,096 per window (§4.4). All
primitives a full block needs exist in EE (`matrixMultiply` for QKV and
output projections, `arrayReduce`, `sqrt`/`rsqrt` for LayerNorm, `erf` for
exact GELU).

A small encoder would still have to build and validate:

- the QKV and output projections, relative-position bias (a constant
  [heads, N, N] add), LayerNorm and the MLP;
- shifted windows: the same partition on a grid offset by w/2, with the
  out-of-raster keys excluded, reproducing Swin's masked cyclic shift;
- patch merging (2×2 space-to-depth plus a linear layer);
- keeping tokens in window layout across blocks. The depth-to-space used
  here for checking copies each window's [N, d] output into every pixel of
  the window, which is fine for w=7 but should not sit between blocks.

The risks are graph growth over many blocks (the 10 MiB request limit and
"Computation is too complex"), the extra float32 casts EE's double-returning
`matrixMultiply`/`exp` require, and the interactive memory ceiling. As in the
earlier U-Net work, batch export gives more headroom (§4.5).

## 9. Caveats

- **Profiler semantics.** "Peak Mem" is EE's per-algorithm accounting across
  all invocations in a request; image-tile entries include their subgraph.
  Numbers compare variants under the same accounting; they are not RAM
  measurements. Failed raster requests report no pixel-computation profile.
- **Generated inputs.** The stress inputs come from one integer hash with the
  seed as an offset, so on the value path Q row i equals 3× K row i−2,771
  (and V row j equals K row j+2,771) modulo 65,536: beyond N=2,772 about 40%
  of queries have an exactly proportional key, which makes those attention
  rows peakier. Memory, runtime and graph size do not depend on this, and
  every check compares against references on identical inputs. No such
  alignment occurs within ±100 pixels on the rasters.
- **Shared service.** Timings are single observations on a shared service;
  EE may cache. A few small correctness requests overlapped the first raster
  ladder; comparisons therefore use medians and EECU·s.
- **Scope.** One attention block with given Q, K, V; no projections, relative
  position bias, shifted windows, LayerNorm or MLP. float32 throughout.
