# Findings: a W-MSA + SW-MSA Swin block pair in native Earth Engine

Live experiments on 3–4 October 2026: Earth Engine Python API 1.7.45, PyTorch
2.14.0, timm 1.0.30, NumPy 2.5.3, Cloud project `ORIGINAL_PROJECT`,
application-default credentials. Every forward pass ran entirely in EE
computation graphs: no Vertex AI, Cloud Run, model server, exported imagery or
client-side tensor computation. EE-computed tensors were fetched only for
comparison.

Evidence: numerical_comparison_m3.csv (development archive: `docs/history/swin_pair/results/numerical_comparison_m3.csv`)
(summary: numerical_summary_m3.md (development archive: `docs/history/swin_pair/results/numerical_summary_m3.md`)),
benchmark_results_m3.csv (development archive: `docs/history/swin_pair/results/benchmark_results_m3.csv`) (summary:
benchmark_summary_m3.md (development archive: `docs/history/swin_pair/results/benchmark_summary_m3.md`)), per-request
records with full EE profiles in results/raw/ (archive `swin_pair/results/raw/` at `research-archive-2026-10-12`).

**Research question.** Can Earth Engine execute a genuine consecutive
W-MSA → SW-MSA Swin block pair while preserving PyTorch/timm numerical
behavior? It must include the cyclic shift, shifted-window re-partition,
attention mask, reverse shift, LayerNorm, MLP and residuals. **Yes.**

- The pair matches timm's `SwinTransformerStage(depth=2)` to **9.2e-6** over
  all 4,816,896 outputs of the 224×224-token target (target 1e-4).
- Every intermediate is bit-identical to the float64 mirror of the EE
  algorithm. That covers 22.1 M EE-computed elements with zero differences.
- Shift, re-partition, reverse shift and mask are exact: 2.85 M layout and
  mask elements, all identical to timm's.
- The pair runs interactively up to 784 windows (196×196 tokens; 3 of 3 runs).
  The full 1,024-window target exceeds EE's interactive memory limit as one
  request, but it runs as four 256-window interactive requests. It also runs
  as one **batch** task, and batch ran up to 9,216 windows.

**Milestone verdict.** The minimum, strong and full-architectural success
criteria are all met; full architectural success is via **Option B (batch)**.
Interactive success at 224×224 holds only through ≤256-window chunked requests.
SW-MSA is supported by the native EE computation model. The full target
exceeds the *interactive per-request* memory envelope; that is an operational
limit, not an architectural one.

## Answers

| # | Question | Answer |
|---|---|---|
| 1 | Can exact SW-MSA execute entirely inside EE? | **Yes.** Shift → re-partition → masked 3-head attention with relative bias → reverse partition → reverse shift → projection → residual → LN → MLP → residual runs in one EE graph. Tested on multi-window rasters of 2×2 to 32×32 windows, square and rectangular (§1, §2). |
| 2 | Can the cyclic shift be reproduced exactly? | **Yes, bit for bit.** Unique integer token IDs land exactly where `window_partition(torch.roll(x, (-3, -3)))` puts them. This holds at every token of rasters of 2×2, 3×3, 4×4, 3×5 and 5×2 windows, including the wrap at both edges (§2). |
| 3 | Can shifted windows be assembled efficiently from window-layout tensors? | **Yes.** The roll separates into row and column rolls. Each is one translated neighbour, two contiguous `arraySlice`s and one `arrayCat` (the column step via a `[7, 7·C]` reshape). With its input materialized once and a periodic domain for the wrap, LN1 + shift costs ~6% more memory than LN1 alone, which is exactly the one-window halo (§3, §5). |
| 4 | Is the timm attention mask reproduced exactly? | **Yes, element by element.** The mask is built in EE from two `[49, 49]` patterns and two per-window flags, with no payload. It equals timm's `attn_mask` buffer at every element for 2×2, 2×3, 4×4, 3×5 and 32×32 windows (2.46 M elements at the target), sign bits included (§2). |
| 5 | Does reverse partition + reverse shift restore the token layout? | **Yes, exactly.** Reverse shift of timm's own shifted layout gives back the IDs. Shift → (identity) → reverse shift is the identity on integer IDs and on 96-channel float tokens, bit for bit, on every raster tested. Window reverse is the identity in window layout (§2). |
| 6 | Does a complete SW-MSA block numerically match timm? | **Yes.** Block B alone on 4×4 windows: every intermediate of every window compared. Final output within **3.9e-6** of PyTorch float32; largest stage 7.6e-6 (masked logits). Every isolated component on 2×3 windows is within 7.6e-6, and the layout steps have 0 error (§1). |
| 7 | Does a W-MSA + SW-MSA pair numerically match timm? | **Yes.** 4×4 windows, every intermediate of both blocks: pair output within **5.8e-6**. Rectangular 3×5 windows: 5.8e-6. 224×224 target: **9.2e-6** over all 4.8 M outputs (MAE 8.1e-7, RMSE 1.0e-6) (§1). |
| 8 | What is the maximum final-output error? | **9.18e-6** (target pair output vs PyTorch float32). The largest error at any compared intermediate is 9.3e-6, at block B's Q Kᵀ (4×4 windows) (§1). |
| 9 | Where does the first numerical divergence appear? | **Block A's LayerNorm 1** (4.8e-7), as in Milestone 2. Every later difference is PyTorch's float32 rounding (Q10). No layout operation introduces any difference: the isolated shift, reverse shift and residuals have 0 error (§1). |
| 10 | Does the EE mirror remain bit-identical? | **Yes, everywhere.** 22,141,336 EE-computed elements across isolated components, block B, the pair (three variants), the rectangular pair and the full target: zero differences. EE is 3.7× closer to float64 than PyTorch at the target output (2.4e-6 vs 8.9e-6) (§1). |
| 11 | How much extra memory does SW-MSA introduce relative to W-MSA? | **The shift itself: ~6%** (the halo): 87 vs 78 MiB at 256 windows, 312 vs 294 MiB at 1,024 (LN1 + shift vs LN1). **The block: none that limits it.** The SW-MSA block alone fits interactively at 1,024 windows (3 of 3 runs), like the W-MSA block, at the same EECU·s (49–71 vs 50–74). Reported peaks are not comparable across these two graphs (§6). |
| 12 | Which shifted-window operation dominates memory? | **None of the layout operations.** They add the halo only. In the pair, the dominant profiled entry is the materialization of block A's output, which includes block A's whole subgraph and its MLP. Without that materialization the translations dominate catastrophically: EE re-evaluates the upstream subgraph per translated view (memory limit at 256 windows) (§3, §6). |
| 13 | Which operation dominates compute? | **`erf`** (exact GELU): 27–31% of EECU·s in the W-MSA block, the SW-MSA block and the pair. Then fused element-wise arithmetic (`Image.float`/`Image.add`), then materialization reads (`Image.reproject`, 13–14% in the pair). `exp` (softmax) is ~3% (§6). |
| 14 | Does the MLP remain the largest memory consumer? | **Yes.** In block B alone, LN2 + MLP + residual is the largest step (761 → 1,148 MiB at 1,024 windows). In the pair, the dominant entry (block A's materialization) is itself dominated by block A's MLP (§6). |
| 15 | What is the largest raster that runs interactively? | **784 windows (196×196 tokens)**: 3 of 3 runs, 1,330 MiB reported peak, 70–148 EECU·s. The ladder was 4, 16, 64, 144, 256, 400, 576 and 784 windows, all verified (§4). |
| 16 | Does the 224×224-token target run interactively? | **Not as one request.** 1,024 windows exceeds the memory limit in all 8 attempts: default (twice), tileScale 16, LN after shift, and with block A's residual 1 materialized (4×). **As ≤256-window requests, yes:** 4 of 4 chunks, and 32 one-window-row requests that fetched every output element (§4, §7). |
| 17 | If not, does it run in batch? | **Yes.** The same graph, exported as a table of checksums: 2 of 2 runs verified against float64. 107 and 159 EECU·s, 64 and 140 s run, 7–8 s queue (§4). |
| 18 | What is the batch headroom? | **At least 9×.** Batch succeeded at 2,304 (225 EECU·s), 4,096 (387) and **9,216 windows** (672×672 tokens, 753 EECU·s, 280 s run), the largest tried. All verified; ~0.08–0.16 EECU·s per window (§4). |
| 19 | Is chunking into ≤256-window regions with halos practical interactively? | **Yes.** 4 chunks of 256 windows: all succeeded, 495 MiB peak each, 140 EECU·s total, 144 s sequential wall. 16 chunks of 64 windows: 184 EECU·s, 429 s. The chunk checksums combine to the whole-raster float64 checksums. The halo needs no code: EE evaluates exactly the windows a chunk depends on, wrapped cyclically by the periodic domain (§4). |
| 20 | Does the serialized graph stay below EE's limits? | **Yes.** **2.99 MB**, identical at 4 and 1,024 windows. 588 function invocations after deduplication (one W-MSA block: 345); 3.4 s to serialize; no "too complex" errors. 98% is the 224,694 inline parameters (1.46 MB per block); the limit is 10 MiB (§8). |
| 21 | Is window layout still the preferred representation? | **Yes.** Every operation stays on the coarse grid of `[49, 96]` arrays. The shift is a two-neighbour slab gather costing one halo; window reverse is free; no fine-grid expansion or per-token gather is ever formed. The mask is per window from two shared patterns (§3). |
| 22 | Should the project proceed to Milestone 4, a multi-block Swin stage? | **Yes**, with batch or ≤256-window chunked requests as the execution mode at stage-1 scale, and asset-backed weights as the first engineering step (§9). |

## 1. Numerical equivalence (Phases 1, 5–10)

**Reference.** timm's `SwinTransformerStage(dim=96, out_dim=96, depth=2,
downsample=False, num_heads=3, window_size=7, mlp_ratio=4)`: block 0 shift 0,
block 1 shift 3, `fused_attn=False` on both (scale, `q kᵀ`, `+ bias`, `+ mask`,
softmax, `× v`). Block A has Milestone 2's parameters (seed 0); block B the
same deliberately non-trivial spec with seed 1. The explicit forwards expose
every intermediate. They are asserted bit-identical to `block(x)` and
`stage(x)` in float32 and float64, so every compared intermediate is timm's own.

timm silently uses shift 0 when the resolution does not exceed the window
(`_calc_window_shift`). A single 7×7 window therefore cannot test SW-MSA, and
every shifted case here has at least 2 windows per side. The 2×3 raster
contains all four mask types (interior, last row, last column, corner). The
3×3 raster has an interior window whose shifted window draws on four
neighbours without wrapping.

Inputs are Milestone 2's integer-hash tokens, generated on the server in
window layout. Attention statistics in block B: logits std 2.6, max
probability 0.99, mean row entropy 1.9 nats. 23% of logits are masked;
masked probabilities are float32 denormals up to 2e-41 or exactly 0.

**Final output vs PyTorch float32** (full per-stage tables:
numerical_summary_m3.md (development archive: `docs/history/swin_pair/results/numerical_summary_m3.md`)):

| case | elements | MAE | RMSE | max abs error | max \|EE − float64\| | max \|torch32 − float64\| |
|---|---:|---:|---:|---:|---:|---:|
| block B alone, 4×4 windows | 75,264 | 4.6e-7 | 6.0e-7 | 3.9e-6 | 1.3e-6 | 4.0e-6 |
| pair, 4×4 windows | 75,264 | 8.2e-7 | 1.1e-6 | 5.8e-6 | 1.5e-6 | 6.0e-6 |
| pair, 3×5 windows (rectangular) | 70,560 | 8.2e-7 | 1.1e-6 | 5.8e-6 | — | — |
| **pair, 32×32 windows (224×224 tokens)** | **4,816,896** | **8.1e-7** | **1.0e-6** | **9.2e-6** | **2.4e-6** | **8.9e-6** |

**Block B, every stage, inside the pair (4×4 windows)**:

| stage | MAE | RMSE | max abs error | EE vs mirror |
|---|---:|---:|---:|---|
| LN1 | 2.8e-7 | 3.7e-7 | 3.1e-6 | bit-identical |
| cyclic shift + window repartition | 2.8e-7 | 3.7e-7 | 3.1e-6 | bit-identical |
| Q / K / V | 4.8e-7 | 6.2e-7 | 3.9e-6 | bit-identical |
| Q Kᵀ | 1.1e-6 | 1.4e-6 | 9.3e-6 | bit-identical |
| + relative-position bias | 1.1e-6 | 1.4e-6 | 9.3e-6 | bit-identical |
| shifted-window mask | 0 | 0 | 0 | bit-identical |
| + shifted-window mask | 1.1e-6 | 1.8e-6 | 9.3e-6 | bit-identical |
| softmax | 1.7e-8 | 6.3e-8 | 1.3e-6 | bit-identical |
| attention × V | 5.4e-7 | 7.3e-7 | 7.3e-6 | bit-identical |
| attention projection | 5.8e-7 | 7.6e-7 | 4.8e-6 | bit-identical |
| reverse partition (identity) | 5.8e-7 | 7.6e-7 | 4.8e-6 | bit-identical |
| reverse shift | 5.8e-7 | 7.6e-7 | 4.8e-6 | bit-identical |
| residual 1 | 7.4e-7 | 9.5e-7 | 6.1e-6 | bit-identical |
| LN2 / fc1 / GELU / fc2 | ≤ 4.1e-7 | ≤ 5.3e-7 | ≤ 3.6e-6 | bit-identical |
| **pair output** | 8.2e-7 | 1.1e-6 | 5.8e-6 | bit-identical |

The layout rows carry the error of their input unchanged: the shift row
equals LN1's, the reverse-shift row equals the projection's. In isolation,
fed PyTorch's own float32 tensors, the shift, reverse shift and both residual
additions have **0 error**, and so does Q × 32^-½.

The "+ mask" stage reaches 7.6e-6 even in isolation. That is one float32 ulp
at masked logits near −100 (ulp(100) = 7.6e-6), whose probabilities are below
1e-38. It does not propagate: softmax error stays ≤ 1.3e-6.

## 2. Layout operations and the mask (Tests 1–5)

Every check below is exact or fails (numerical_summary_m3.md (development archive: `docs/history/swin_pair/results/numerical_summary_m3.md`)):

| test | rasters (windows) | result |
|---|---|---|
| 1 cyclic shift of unique token IDs (y·W + x, y, x) vs timm `window_partition(torch.roll(·, (−3, −3)))` | 2×2, 3×3, 4×4, 3×5, 5×2 | 100% exact |
| 1 the row roll alone vs `torch.roll(·, (−3, 0))` | same | 100% exact |
| 2 EE reverse shift of timm's shifted IDs == IDs | same | 100% exact |
| 2 EE shift → reverse shift == IDs | same | 100% exact |
| 3 shift → partition → reverse partition → reverse shift == x, 96-channel float tokens | same | bit-identical |
| 4 EE mask vs timm `attn_mask` | 2×2, 2×3, 4×4, 3×5, 32×32 | element-identical (2.46 M at the target), sign bits identical |
| 5 attention per head, fed timm's shifted Q/K/V and the EE mask | 2×3 | bit-identical to the mirror; ≤ 7.6e-6 vs PyTorch |

The live tests (tests/test_swin_pair_live.py (archive `tests/test_swin_pair_live.py` at `research-archive-2026-10-12`))
repeat Tests 1–8 at small sizes (8 of 8 pass, 107 s). Test 8 also checks that
the interior shifted window of a 3×3 raster holds tokens from exactly the
four windows (1,1), (1,2), (2,1), (2,2).

**Mask construction.** timm labels the shifted raster with 9 regions and
masks token pairs from different regions. Only the last window row and
column straddle region boundaries, split at in-window row/column 4. So for
window (I, J):

    different = max(R · [I = kh−1], C · [J = kw−1])
    mask = 0 − 100 · different

R and C are fixed `[49, 49]` 0/1 patterns ("row slabs differ", "column slabs
differ"), built once from integer sequences. Writing `0 − 100·d` rather than
`−100·d` guarantees +0.0 for allowed pairs whatever the operand types. In
IEEE, 0.0 × −100 is −0.0: numerically harmless, but a sign-bit difference
from timm. EE's integer-typed operands gave +0 either way (probed).

## 3. The shift in window layout (Phases 2, 3, 6)

Shifted window (I, J) holds tokens ((7I + a + 3) mod H, (7J + b + 3) mod W).
Rows a = 0–3 come from rows 3–6 of window row I, and rows 4–6 from rows 0–2
of window row I+1; columns work the same way. In a `[49, C]` window, token-row
slabs are contiguous row ranges, so the row roll is:

    cat( x[21:49], below[0:21] )                axis 0

The column roll is the same on the `[7, 7C]` reshape, where token-column slabs
are contiguous column ranges. The reverse shift uses the opposite offset
(rows 28–48 of the window above, then rows 0–27 of its own). No
`neighborhoodToArray` is used; it rejects array-valued pixels ("Input band
type is not scalar", probed). The neighbour comes from `changeProj` with an
integer offset, which is an exact relabelling of the coarse grid.

**Two decisions made the shift cheap; both were measured** (16×16 windows,
shift of block A's output, all verified):

| formulation | EECU·s | peak MiB | result |
|---|---:|---:|---|
| periodic domain, input materialized (**chosen**) | 15.7 | 336 | ✓ |
| periodic, input and intermediate materialized | 18.8 | 336 | ✓ |
| periodic, nothing materialized | — | — | memory limit |
| `where` wrap (exact domain), input materialized | 44.7 | 1,128 | ✓ |
| `where` wrap, nothing materialized | — | — | memory limit |
| `mosaic` of footprint-clipped near/far copies, materialized | 69.0 | 299 | ✓ |
| *(block A output alone, for scale: 16.5 EECU·s, 412 MiB)* | | | |

1. **Materialize the shift's input.** A translated reference to an
   un-materialized image makes EE evaluate its whole subgraph again on the
   translated region; four translations of a W-MSA block exceed the memory
   limit at 256 windows. `reproject` to the coarse grid makes all
   translations read the same computed tiles.
2. **Periodic domain for the cyclic wrap.** Selecting the copy translated by
   −(k−1) on the edge row makes EE compute the upstream block over the
   translated, mostly out-of-domain region (`where`: 2.8× EECU, 3.4× peak);
   footprint clipping does not prevent it (`mosaic`: 4.4× EECU). Instead the
   input is generated periodic with the raster's window counts: window (I, J)
   = window (I mod kh, J mod kw). Window-local operations preserve this
   (W-MSA, LN, MLP; the mask uses I mod kh), so the +1 neighbour of the last
   row *is* row 0. It is recomputed as a one-window halo, (k+1)²/k² = 6%
   extra at 1,024 windows.

   The period is global, so a chunked request is still correct at the raster
   edge (§4).

   **This is a precondition timm does not have.** The inputs here are
   periodic by construction (generated tokens). A general stage input, such
   as a patch embedding of imagery, needs an explicit periodic extension
   first. This milestone neither implemented nor measured one. The cost
   measured above for far translations applies to an expensive upstream
   image; for a cheap input it is expected to be small, but that is untested.

Window reverse is the identity in this representation: the shifted windows
already are the window layout of the shifted raster. Only the reverse shift
is a real operation.

## 4. Performance: interactive, batch, chunked (Phase 11)

Each request returns checksums of one tensor over every window. Each success
was verified against float64. "Peak" is the EE profiler's largest
per-algorithm "Peak Mem" (image entries include their subgraph), not RAM.

**Pair output vs raster size, interactive** (default composition):

| windows | tokens | wall s | EECU·s | peak | status |
|---:|---:|---:|---:|---:|---|
| 4 | 14×14 | 23 | 2.6 | 32 MiB | ✓ |
| 16 | 28×28 | 34 | 7.1 | 64 MiB | ✓ |
| 64 | 56×56 | 27 | 9.8 | 163 MiB | ✓ |
| 144 | 84×84 | 39 | 33 | 306 MiB | ✓ |
| **256** | **112×112** | **29–46** | **31–42** | **495 MiB** | ✓ 3/3 |
| 400 | 140×140 | 33 | 38 | 728 MiB | ✓ |
| 576 | 168×168 | 77 | 100 | 1,007 MiB | ✓ |
| **784** | **196×196** | **45–168** | **70–148** | **1,330 MiB** | ✓ 3/3 |
| 1,024 | 224×224 | — | — | — | ✗ memory limit, 8 of 8 attempts (default ×2, tileScale 16, `pairopt`, `pairpin` ×4) |

The reported peak grows ~1.7 MiB per window. By that rate 1,024 windows would
need ~1,740 MiB; the W-MSA block alone reports 1,553 MiB at 1,024 windows and
fits. Small requests carry ~20 s of fixed handling for the 3 MB of
parameters (4 windows: 23 s wall for 2.6 EECU·s).

**Batch export** of the same request (checksums only):

| windows | tokens | queue s | run s | EECU·s | status |
|---:|---:|---:|---:|---:|---|
| 1,024 | 224×224 | 7.3 | 64 | 107 | ✓ |
| 1,024 | 224×224 (new seed) | 8.2 | 140 | 159 | ✓ |
| 2,304 | 336×336 | 4.2 | 114 | 225 | ✓ |
| 4,096 | 448×448 | 6.3 | 159 | 387 | ✓ |
| 9,216 | 672×672 | 5.1 | 280 | 753 | ✓ |

For comparison, Milestone 2's single block took 52.6 EECU·s in batch at
1,024 windows; the pair takes 2.0–3.0× that.

**Chunked interactive requests** at 1,024 windows (pair output, checksums per
chunk, summed):

| chunks | windows each | succeeded | max peak | EECU·s total | wall s total (sequential) | combined checksum |
|---:|---:|---:|---:|---:|---:|---|
| 4 | 256 (16×16) | 4/4 | 495 MiB | 140 | 144 | ✓ = whole-raster float64 |
| 16 | 64 (8×8) | 16/16 | 163 MiB | 184 | 429 | ✓ |
| 32 | 32 (one window row) | 32/32 | — | — | 928 (median 29 per request), each returning 150 k numbers | every element compared (§1) |

Each chunk request is the same graph over a sub-rectangle. EE evaluates block
B on the chunk plus a one-window halo and block A on a two-window halo,
wrapped cyclically at the raster edge by the periodic domain. The halo cost is
visible: 256-window chunks recompute block A on (16+2)²/16² = 1.27× the
windows (140 vs 107–159 EECU·s in batch), 64-window chunks on 1.56×.

## 5. Where SW-MSA cost comes from (Phase 12)

Cumulative prefixes, each the request for that tensor (EE evaluates only
what a request depends on). SW-MSA block alone vs W-MSA block, both on the
generated input:

| prefix | W-MSA peak @256 | @1,024 | SW-MSA peak @256 | @1,024 | SW-MSA EECU·s @1,024 |
|---|---:|---:|---:|---:|---:|
| input | — | — | 78 | 294 | 5.7 |
| + LN1 | 78 | 294 | 78 | 294 | 8.3 |
| + shift / re-partition | — | — | **87** | **312** | 7.7 |
| + QKV | 186 | 699 | 149 | 563 | 13.5 |
| + attention (bias, mask, softmax, ×V) | 194 | 730 | 158 | 594 | 27.7 |
| + projection | — | — | 265 | 999 | 23.9 |
| + reverse shift | — | — | 177 | 630 | 24.6 |
| + residual 1 | 272 | 1,024 | 218 | 761 | 20.0 |
| + LN2, MLP, residual 2 | 412 | 1,553 | 305 | 1,148 | 49–71 (3 runs) |

(MiB. The W-MSA block took 50–74 EECU·s at 1,024 over 3 runs; prefix EECU·s
vary by ±50% between runs, so single prefix differences under ~2× are not
meaningful.)

1. **Memory added by the shifted re-partition:** the halo, +9 MiB at 256 and
   +18 MiB at 1,024 windows (+6%).
2. **Compute added:** within run-to-run noise (+1.9 EECU·s at 256, −0.6 at
   1,024).
3. **Does the 2×2 gather dominate?** No, once its input is materialized;
   without that, yes, fatally (§3).
4. **Does masking add measurable cost?** No: identical peaks with and without
   the mask (158/594 MiB at the attention prefix, 305/1,148 MiB for the
   block); EECU·s within noise.
5. **Is the MLP still the dominant memory term?** Yes: LN2 + MLP is the
   largest step of block B, and `erf` is the largest compute term (27–31%).
6. **Do repartition temporaries stay live?** No evidence: past the shift, the
   SW-MSA prefixes report *lower* peaks than the same W-MSA prefixes. A
   materialized tile of `[49, 96]` float32 windows is ~19 MB at 1,024
   windows, and image entries below a materialization no longer include the
   upstream subgraph. The reverse shift's materialization also lowers the
   reported peak (999 → 630 MiB).
7. **Can layout temporaries be eliminated earlier?** One can (§7).

In the pair, the profile's dominant entry is `Image.reproject`: the
materialization of block A's output, whose accounting includes block A
(1,330 MiB at 784 windows). Reported peaks are therefore *not* comparable
across graphs with different materializations. The comparable limit is the
interactive ceiling: one block (W-MSA or SW-MSA) fits at 1,024 windows; the
pair fits at 784 and not at 1,024.

## 6. Optimization after correctness (Phase 13)

All three variants below are bit-identical to the mirror at every
intermediate (pair, 4×4 windows):

| variant | what changes | 256 windows | 784 windows | 1,024 windows |
|---|---|---|---|---|
| default | timm order: Shift(LN1(x)); materialized block input, shift inputs | 495 MiB ✓ | 1,330 MiB ✓ | ✗ |
| `pairopt` | LN1(Shift(x)): LN is per token, so bit-identical; one materialization fewer | 495 MiB ✓ | 1,330 MiB ✓ | ✗ |
| `pairpin` | + block A's residual 1 materialized before its MLP | 515 MiB ✓ | 1,370 MiB ✓ | ✗ (4 of 4) |

- Materializing block A's residual 1 lowers block A's *own* reported peak by
  24–27% (412 → 299 MiB at 256; 1,553 → 1,128 MiB at 1,024 windows). It does
  not move the pair's ceiling. The default composition is therefore kept;
  both options are available (`shift_first`, `pin_a_x1`).
- No explicit casts were added; the mask is the only new constant-sized array
  per window. It is built from shared value constants, not a per-window
  payload.
- tileScale 16 does not help (single coarse tile; materialized tiles have
  their own fixed tiling).
- Weights remain inline: 2.99 MB is under the 10 MiB limit. The repository's
  asset-backed weight store (`native_ee/weight_assets.py`) was not exercised
  in this milestone.

## 7. Graph size and composition

| request | bytes | invocations after dedup | distinct functions | serialize s |
|---|---:|---:|---:|---:|
| W-MSA block (block A), periodic input | 1,498,720 | 345 | 44 | 1.4 |
| SW-MSA block (block B) | 1,508,333 | 411 | 51 | 1.5 |
| **pair**, 4 or 1,024 windows | **2,987,146** | **588** | 51 | 3.4 |
| parameters of one block alone | 1,460,525 | — | — | — |

(Milestone 2 recorded 339 invocations for the same block with its
non-periodic input; the periodic index adds the difference.) The shifted
machinery (two shifts, mask, periodic index) adds 9.6 KB and 66 invocations
to a block. The parameters (224,694 for the pair) are 98% of the
bytes. A request grows by ~1.5 MB per block of this width, so ~6 blocks of
Swin-T stage-1 width is the inline ceiling.

## 8. Earth Engine behavior found or confirmed (probed live)

probes.json (archive `swin_pair/results/raw/probes.json` at `research-archive-2026-10-12`) and the formulation benchmarks:

1. `changeProj` by an integer number of coarse pixels relabels array-valued
   pixels exactly. `Image.translate(0, 1, 'pixels')` moves toward smaller row
   indices (y up) and is not used.
2. A translated view of an un-materialized image re-evaluates its subgraph;
   `reproject` to the coarse grid makes translations share computed tiles.
3. `where` evaluates both branches over the whole region. Far-translated
   branches therefore make EE compute upstream images over translated,
   out-of-domain regions. Footprint-clipped images in a `mosaic` are still
   computed (4.4× EECU·s).
4. `clip` combined with `where` loses the wrapped windows: `where` leaves
   masked input pixels masked. Observed as a failed checksum in the first
   probe.
5. `neighborhoodToArray` rejects array-valued pixels.
6. EE's `mod` takes the dividend's sign (−1 mod 8 = −1). The periodic index
   is ((I mod k) + k) mod k.
7. EE keeps float32 denormals: exp(−100) → 3.78e-44, as NumPy. Masked
   probabilities are therefore compared bit for bit too.
8. As in Milestone 2: tileScale does not split a raster that is one coarse
   tile; batch export lifts the interactive memory limit.

## 9. Recommendation: proceed to Milestone 4

**Proceed.** The evidence on each criterion:

- **Numerical correctness**: 9.2e-6 at the target. Bit-identical to the
  specification at every one of 22.1 M compared elements; no new error
  source.
- **Shift / repartition correctness**: exact, on IDs and floats, square and
  rectangular, through chunked requests.
- **Memory growth**: the shift costs one halo. A pair costs ~1.3× a single
  block's interactive envelope (ceiling 784–1,024 windows vs ≥1,024).
  Materialization points let blocks be evaluated as stages rather than one
  ever-growing subgraph.
- **Interactive feasibility**: up to 784 windows per request; the 224×224
  target as ≤256-window chunks.
- **Batch feasibility**: the target in ~1.5–2.5 min, and 9× the target in
  under 5 min.
- **Graph growth**: linear in parameters (~1.5 MB per block), constant in
  raster size; the shifted machinery is 66 invocations.
- **Runtime / EECU**: ~0.1 EECU·s per window per pair in batch, about twice
  a single block.
- **Layout efficiency**: window layout carries SW-MSA with no fine-grid
  expansion.

Swin-T's stage 1 has depth 2, so this pair is that whole stage, without patch
embedding. For a deeper stage (Swin-T stage 3: 6 blocks; Swin-B: 18):

1. **Move weights to table assets first.** Inline parameters reach the
   10 MiB payload limit at ~6 stage-1-width blocks. Later Swin-T stages have
   4×, 16× and 64× the parameters per block (dim 192, 384, 768).
2. **Plan execution as batch, or chunks with an N-window halo for N pairs.**
   A stage of more pairs at stage-1 resolution will not fit one interactive
   request. Each SW-MSA block extends the dependency halo by one window per
   side.
3. **Keep the periodic-domain convention, and implement and measure the
   periodic extension of a general stage input.** That extension is the one
   precondition of this implementation that timm does not have, and it is
   untested here (§3).
4. **Keep the mirror test for each new operation** (patch merging next).

## 10. Caveats

- **Profiler semantics.** "Peak Mem" is EE's per-algorithm accounting, not
  RAM. Its attribution changes with materialization, so peaks are compared
  only within one graph shape. EE does not document the per-request limit;
  "ceiling" means the largest success and smallest failure observed.
- **Shared service.** EECU·s and wall time vary 2–4× between identical runs
  (784 windows: 70–148 EECU·s, 45–168 s); single observations are
  indicative.
- **Inputs.** Generated hash tokens and seeded parameters, not a trained
  model or real imagery. The input is periodic by construction. Inside the
  raster it equals Milestone 2's input bit for bit.
- **Scope.** One W-MSA + SW-MSA pair at Swin-T stage-1 width. No patch
  embedding, patch merging, deeper stages or decoder.
