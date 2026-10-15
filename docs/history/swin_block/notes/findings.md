# Findings: one Swin Transformer block in native Earth Engine

Live experiments on 1 October 2026: Earth Engine Python API 1.7.45, PyTorch
2.14.0, timm 1.0.30, NumPy 2.5.3, Cloud project `ORIGINAL_PROJECT`,
application-default credentials. The block's forward pass ran entirely in EE
computation graphs; no Vertex AI, Cloud Run, model server, exported imagery or
client-side tensor computation. EE-computed tensors were fetched only for
comparison.

Evidence: numerical_comparison.csv (development archive: `docs/history/swin_block/results/numerical_comparison.csv`) and
its summary numerical_summary.md (development archive: `docs/history/swin_block/results/numerical_summary.md`);
benchmark_results.csv (development archive: `docs/history/swin_block/results/benchmark_results.csv`) and
benchmark_summary.md (development archive: `docs/history/swin_block/results/benchmark_summary.md`); per-request records
with full EE profiles in results/raw/ (archive `swin_block/results/raw/` at `research-archive-2026-10-12`).

**Research question.** Can a genuine Swin Transformer block, not merely its
attention kernel, be compiled into native Earth Engine operations while
preserving PyTorch numerical behavior and remaining practical at Swin-T
stage-1 scale? **Yes.** One non-shifted Swin-T stage-1 block (dim 96, 3 heads
× 32, window 7, MLP 384, exact GELU) runs over a 224×224-token raster (1,024
windows) in one interactive EE request. All 4,816,896 outputs are within
5.8e-6 of PyTorch float32 (target 1e-4). The implementation is bit-identical to
its own float64-with-float32-rounding specification at every intermediate.
It runs about 27% below EE's interactive memory ceiling, and the MLP is the
dominant memory consumer.

## Answers

| # | Question | Answer |
|---|---|---|
| 1 | Can a complete Swin block execute entirely inside EE? | **Yes.** LN → QKV → 7×7 W-MSA with relative-position bias → projection → residual → LN → MLP (exact GELU) → residual runs in one EE graph, on one window (`ee.Array` and array-image paths), on 16 windows, and on the 224×224 raster (§1, §4). |
| 2 | Does the final output numerically match PyTorch? | **Yes**, at every element compared: single window, 16-window raster (every intermediate), and all 1,024 windows of the target (final output, plus whole-raster checksums) (§1). |
| 3 | What is the maximum error? | Final output vs PyTorch float32: **3.2e-6** (single window), 4.1e-6 (16 windows), **5.8e-6** over all 4.8 M target outputs (MAE 4.4e-7, RMSE 5.7e-7; 7.8e-7 of the output range). Largest among the compared intermediates: 5.5e-6 (attention × V, 16 windows; at the target only the output was compared) (§1). |
| 4 | Where does the largest divergence appear? | At the attention logits Q Kᵀ and what follows (Q Kᵀ, + bias, × V: 3.8e-6 on the single window), where magnitudes are largest (logits up to 15.6). The first nonzero divergence is LayerNorm 1 (7.2e-7). Every divergence is PyTorch's float32 rounding, not EE's (§2). |
| 5 | Float32 differences vs implementation differences? | **All float32, no implementation error.** EE is bit-identical to a float64 mirror of the EE algorithm that rounds to float32 where EE does, at every intermediate of every window checked (7.4 M EE-computed elements). EE is 3–9× closer to float64 than PyTorch float32 is (target output: 1.9e-6 vs 6.2e-6) (§2). |
| 6 | Can LayerNorm be implemented efficiently in EE? | **Yes.** Two `arrayReduce` sums, `arrayRepeat`, `sqrt`, element-wise arithmetic, one float32 rounding. At 1,024 windows it adds ~1.3 EECU·s and no measurable EE-reported peak memory; it is 3–5× closer to float64 than PyTorch's LayerNorm (§3). |
| 7 | Can QKV and MLP projections share one linear primitive? | **Yes.** One function (`linear.py`: `matrixMultiply` of the `[49, in]` window by a constant Wᵀ image, plus a row-repeated bias, rounded once) serves QKV 96→288, projection 96→96, fc1 96→384 and fc2 384→96 (§3). |
| 8 | Does relative-position bias reproduce PyTorch exactly? | **Yes, bit for bit.** The index is built inside EE from integer sequences and equals timm's `relative_position_index` exactly. The gather, a one-hot `[2401, 169]` × table matrix product, gives a `[3, 49, 49]` bias identical to timm's on both paths (§3). |
| 9 | How large is the serialized graph? | **1.50 MB, constant in raster size** (identical at 1 and 1,024 windows): 339 function invocations after the client's deduplication, 97% of the bytes being the 112,347 parameters (shortest-decimal encoding; 2.44 MB with exact float64 decimals). The limit is 10 MiB (§5). |
| 10 | Interactive runtime and EECU cost? | Target, one request: **44–52 EECU·s** (one outlier 86) and **24–261 s wall** (median ~82 s) over 8 runs. 256 windows: 12.6 EECU·s, 16 s. A 32-window request returning all 150k of its outputs: ~14 s. Batch export: 52.6 EECU·s, 46 s run + 6 s queue at the target; 2,304 windows in 93 s, 4,096 windows in 109 s (§4). |
| 11 | Which operation dominates memory? | **The MLP**: its `[49, 384]` hidden activations, kept in double inside fc1 and GELU. It raises the EE-reported peak at 1,024 windows from 1.07 GB (attention branch) to 1.63 GB, and alone needs 1.22 GB. Next is the QKV projection (+0.43 GB); attention itself (3 × `[49, 49]`) adds only ~0.03 GB. By compute, `erf` (exact GELU) is 31% of all EECU·s (§4). |
| 12 | Can the full block run over the 224×224 target raster? | **Yes, interactively, in one request** (8 of 8 attempts at tileScale 1, 4 and 16, all verified), and every output element was verified through 32 smaller requests. It is near the interactive ceiling: 1,296 windows (252×252) fail, and tileScale does not help. Batch export has headroom: 4,096 windows (448×448) succeed (§4). |
| 13 | Does window layout remain effective? | **Yes.** The whole block is per-token row operations plus per-window attention, so it runs on a 32×32 grid of `[49, 96]` arrays with no gather, neighborhood or conversion. The cost: the 32×32-pixel grid is a single EE tile, so tileScale cannot split it and memory grows with the windows per request (§4, §6). |
| 14 | Graph-complexity problems when composing? | **None.** 339 invocations, no "Computation is too complex", 1.1 s to serialize. The composition risk was memory: the first implementation (explicit double casts, Milestone 1's per-pixel checksum weights) failed at the target. Removing the redundant casts alone was not enough (2 of 2 failures); with lighter verification weights as well, the target fits (§5, §6). |
| 15 | Proceed to shifted-window attention? | **Yes, with the memory budget as the first constraint.** Correctness and graph size are not limiting. SW-MSA adds a cyclic re-partition across neighboring windows and masks, more temporaries on a target already within ~27% of the interactive ceiling, so plan from the start for ≤256-window requests or batch, and asset-backed weights (§7). |

## 1. Numerical equivalence (Phases 1–10)

**Reference.** timm's `SwinTransformerBlock(dim=96, input_resolution, num_heads=3,
window_size=7, shift_size=0, mlp_ratio=4)` with `fused_attn=False`. That is the
original Swin order: `q * scale`, `q @ kᵀ`, `+ bias`, softmax, `@ v`; timm only
takes the fused path when `TIMM_FUSED_ATTN > 1`, and it is pinned off. A
step-by-step forward exposes every intermediate and is asserted bit-identical
to `block(x)` in float32 and float64, so the intermediates are timm's own.
Parameters are seeded and chosen so that mistakes cannot hide: LayerNorm γ ≈ 1 ± 0.2
and β ≈ ±0.2, an O(1) relative-position table, and attention logits with std 2.5
(max probability 0.995, mean row entropy 2.05 nats vs 3.89 uniform). The
single window uses `torch.randn` tokens; rasters use Milestone 1's integer-hash
tokens (×2), generated on the server directly in window layout and
bit-identical to NumPy.

**Final block output vs PyTorch float32** (full per-stage tables in
numerical_summary.md (development archive: `docs/history/swin_block/results/numerical_summary.md`)):

| Case | elements | MAE | RMSE | max abs error | max \|EE − float64\| | max \|torch32 − float64\| |
|---|---:|---:|---:|---:|---:|---:|
| 1 window (value path = pixel path, bit for bit) | 4,704 | 4.6e-7 | 5.9e-7 | 3.2e-6 | 7.9e-7 | 3.0e-6 |
| 4×4 windows, 28×28 tokens | 75,264 | 4.4e-7 | 5.7e-7 | 4.1e-6 | — | — |
| 32×32 windows, 224×224 tokens | 4,816,896 | 4.4e-7 | 5.7e-7 | 5.8e-6 | 1.9e-6 | 6.2e-6 |

**Every stage, single window** (EE value path; the pixel path is identical):

| stage | MAE | RMSE | max abs error | max \|EE − float64\| | max \|torch32 − float64\| |
|---|---:|---:|---:|---:|---:|
| LayerNorm 1 | 4.5e-8 | 7.8e-8 | 7.2e-7 | 1.2e-7 | 6.5e-7 |
| Q | 2.1e-7 | 2.9e-7 | 1.7e-6 | 2.6e-7 | 1.7e-6 |
| K | 2.0e-7 | 2.9e-7 | 2.9e-6 | 3.0e-7 | 2.8e-6 |
| V | 2.1e-7 | 2.9e-7 | 1.9e-6 | 2.5e-7 | 1.8e-6 |
| attention logits Q Kᵀ (after Q × 32^-0.5) | 5.3e-7 | 6.8e-7 | 3.8e-6 | 9.0e-7 | 3.8e-6 |
| + relative position bias | 5.3e-7 | 6.9e-7 | 3.8e-6 | 8.7e-7 | 3.9e-6 |
| softmax | 9.8e-9 | 3.7e-8 | 7.0e-7 | 1.7e-7 | 5.9e-7 |
| attention × V | 3.0e-7 | 4.3e-7 | 3.8e-6 | 8.6e-7 | 3.0e-6 |
| output projection | 3.4e-7 | 4.5e-7 | 2.3e-6 | 7.5e-7 | 2.1e-6 |
| residual 1 | 3.4e-7 | 4.5e-7 | 2.3e-6 | 7.5e-7 | 2.1e-6 |
| LayerNorm 2 | 2.6e-7 | 3.5e-7 | 2.2e-6 | 6.5e-7 | 2.0e-6 |
| MLP fc1 | 3.1e-7 | 4.0e-7 | 2.3e-6 | 5.6e-7 | 2.3e-6 |
| GELU | 1.8e-7 | 2.9e-7 | 2.4e-6 | 6.2e-7 | 2.3e-6 |
| MLP fc2 | 2.9e-7 | 3.8e-7 | 1.9e-6 | 3.7e-7 | 1.8e-6 |
| **final block output** | 4.6e-7 | 5.9e-7 | **3.2e-6** | 7.9e-7 | 3.0e-6 |

On the 16-window raster every one of these stages was compared at every
element of every window (max 5.5e-6, at attention × V); on the target, the
final output of every window, plus whole-raster checksums (weighted sum and
sum of squares within 1e-8 relative of float64, exact window count).

## 2. Where the error comes from

Each component was also run **in isolation**, fed PyTorch's own float32 inputs,
on both paths. Error does not grow across the block: every isolated component
is within 2.4e-6 of PyTorch, and the composed block's worst stage is 3.8e-6.
The pure layout steps are exact. Splitting PyTorch's `qkv` into heads at
columns `s·96 + 32h + j` reproduces its `reshape(N, 3, 3, 32).permute(…)` q,
k, v bit for bit, and concatenating its heads reproduces its
`transpose(1, 2).reshape` bit for bit. Q × 32^-0.5 and both residual adds are
bit-identical to PyTorch: single IEEE float32 operations. PyTorch's
`tensor * python_float` rounds the scalar to float32 first (checked), so EE
multiplies by float32(32^-0.5) too.

To separate implementation error from rounding, a float64 **mirror** of the
EE algorithm (ee_mirror.py (archive `swin_block/reference/ee_mirror.py` at `research-archive-2026-10-12`)) rounds to float32
exactly where the EE graph does: once at the end of each LayerNorm, Linear and
GELU, and at each step of Milestone 1's softmax. Result:

- **EE equals the mirror bit for bit** at every intermediate, isolated and
  composed, value and pixel path, every window: 7.4 million EE-computed
  elements, zero differences. So EE's `matrixMultiply`, reductions, `exp`,
  `sqrt` and `erf`, and the whole composition, execute exactly the specified
  arithmetic; double accumulation order never changed a float32 result.
- **Every EE−PyTorch difference is PyTorch's float32 kernel rounding.**
  PyTorch rounds inside LayerNorm (float32 statistics), inside its float32
  GEMM accumulation, inside softmax, and in its vectorized float32 `erf`. Its
  LayerNorm and GELU outputs were correctly rounded for only 41% and 35% of
  elements in a check on random inputs. EE's policy, double inside each PyTorch operation and
  one float32 rounding at its boundary, lands 3–9× closer to float64 at
  every stage (2.8× at the output projection, 9× at K). EE−PyTorch therefore
  ≈ PyTorch−float64, the same conclusion as Milestone 1 at Q Kᵀ, now for every operation of a block.

Precision behavior relied on (all probed live; see §6): `matrixMultiply`, sum
reductions, `sqrt`, `erf` and `exp` return double for float32 inputs; float32
± × ÷ float32 stays float32 (one correct IEEE rounding); a non-integer Python
constant promotes to double; EE's `erf` is accurate to double precision
(≤ 8e-16 on [−6, 6]), so exact GELU is reproducible.

## 3. Components

| Component | EE formulation | Validation |
|---|---|---|
| LayerNorm | mean = Σx/96 (`arrayReduce` sum, double), centered, var = Σ(x−μ)²/96 (biased), (x−μ)/√(var+1e-5)·γ+β, one float32 rounding; γ, β as `[49, 96]` row-repeated constants (no broadcasting in EE array arithmetic) | isolated: 7.2e-7 vs torch32, 1.2e-7 vs float64 |
| Linear (QKV, proj, fc1, fc2) | `X[49,in].matrixMultiply(Wᵀ[in,out])` (double) + bias rows, one rounding | isolated: ≤ 2.4e-6 vs torch32, ≤ 2.4e-7 vs float64 |
| QKV split | `arraySlice` columns s·96 + 32h … +31 | exact vs timm's reshape/permute |
| Attention | Milestone 1's `naive_attention`, extended with `query_scale` (Swin's scale-before-matmul; default behavior unchanged) and its existing additive `key_bias` used for the relative-position bias | isolated: ≤ 1.9e-6; softmax 3.6e-7 |
| Relative-position bias | index from `ee.List.sequence` arithmetic; gather = one-hot `eq` matrix × table (`matrixMultiply`), computed once as an `ee.Array` value | index exact; bias bit-identical to timm |
| Head concat | `arrayCat` along columns | exact vs timm's transpose/reshape |
| GELU | exact: x·0.5·(1 + erf(x·√½)) in double, one rounding | isolated: 4.8e-7 vs torch32, 1.2e-7 vs float64 |
| Residuals | float32 + float32 | bit-identical to PyTorch |

Parameters are embedded as the shortest decimals that round to each float32
value and restored with `ee.Array(...).float()`. Every parameter was fetched
back bit-exact, and on the image path (constant − exact float64 value) was
exactly 0, which rules out the trap Milestone 1 found with
`Image.constant(x).toFloat()`.

## 4. Performance (Phase 11)

Each benchmark request returns checksums of one tensor over all windows;
every success was verified against float64. k = windows per side; the target
is k = 32. EE-reported memory is the profiler's per-algorithm "Peak Mem"
(image entries include their subgraph), not RAM.

**Full block versus raster size** (interactive, default tiles):

| windows | tokens | wall s | EECU·s | EE peak | status |
|---:|---:|---:|---:|---:|---|
| 1 | 7×7 | 11.1 | 1.0 | 11 MB | ✓ |
| 4 | 14×14 | 11.1 | 0.9 | 13 MB | ✓ |
| 16 | 28×28 | 11.4 | 1.7 | 37 MB | ✓ |
| 64 | 56×56 | 14.5 | 4.5 | 121 MB | ✓ |
| 256 | 112×112 | 15.7 | 12.6 | 432 MB | ✓ |
| **1,024** | **224×224** | **24–261 (median ~82)** | **44–52** (one 86) | **1.63 GB** | ✓ 8/8 |
| 1,296 | 252×252 | 24.6 | — | — | ✗ memory |

**Batch export** of the same request (`Export.table.toAsset` of the
checksums; EECU·s is the task's `batch_eecu_usage_seconds`):

| windows | tokens | queue s | run s | EECU·s | status |
|---:|---:|---:|---:|---:|---|
| 1,024 | 224×224 | 6.5 | 45.7 | 0.04 | ✓ answered from EE's cache: the identical graph had run interactively; not a compute measurement |
| 1,024 | 224×224 (new checksum seed, so a new graph) | 5.9 | 46.4 | 52.6 | ✓ |
| 2,304 | 336×336 | 4.1 | 92.9 | 91.2 | ✓ |
| 4,096 | 448×448 | 4.5 | 109.3 | 200.4 | ✓ |

Batch removes the interactive ceiling for this block (4× the target
succeeded, the largest size tried) at about the same EECU·s per window, with
~46 s minimum run time even for a cached result.

**Where cost goes** (EE-reported peak and EECU·s, by block prefix; each
prefix is the request for that tensor, since EE evaluates only what a request
depends on):

| prefix (cumulative) | peak @ 256 windows | peak @ 1,024 | EECU·s @ 1,024 | request |
|---|---:|---:|---:|---:|
| generated input x | 82 MB | 308 MB | 5.7 | 14 KB |
| + LayerNorm 1 | 82 MB | 308 MB | 7.0 | 19 KB |
| + QKV projection | 195 MB | 733 MB | 15.0 | 377 KB |
| + 3-head W-MSA (bias, softmax, ×V, concat) | 203 MB | 765 MB | 15.2 | 398 KB |
| + output projection, residual 1 | 285 MB | 1,073 MB | 29.5 | 519 KB |
| + LN2, MLP, residual 2 (full block) | 432 MB | 1,628 MB | 44–52 | 1,498 KB |
| MLP branch alone (x + MLP(LN(x))) | 325 MB | 1,224 MB | 34.0 | 993 KB |

- **Memory:** the MLP dominates. Its `[49, 384]` hidden activations are 4× the
  token width and are held in double through fc1 and GELU. Attention adds
  little, because 7×7 windows give `[49, 49]` score matrices. Memory grows
  linearly at ~1.6 MB per window, and the 1.63 GB at 1,024 windows sits below
  a failure at 1,296.
- **Compute:** by algorithm at the target, `erf` 31% (the exact GELU, ~19 M
  evaluations), then the fused arithmetic attributed to `Image.toFloat` (28%)
  and `Image.add` (26%); `exp` (softmax) 3%.
- **Wall time** at the target ranged 24–261 s over 8 runs while EECU·s stayed
  at 44–52 (one 86). That is consistent with the single 32×32-pixel tile being
  computed largely serially plus variable service-side delay near the memory
  ceiling; the profile does not show which. Small requests carry a fixed ~10 s of request handling for the 1.5 MB
  of parameters (1 window: 11 s wall for 1 EECU·s; without the parameters,
  0.5 s). Splitting the target into 32 requests of 32 windows (one row
  of windows each) gave ~14 s each (median), including returning 150k numbers.
- **tileScale** 2–16 changed nothing: the same peak (1.63 GB) at tileScale 1,
  4 and 16, and the first implementation failed at every tileScale. On a
  32×32-pixel coarse grid the computation is one tile; Milestone 1's
  tileScale rescue applied to fine 224×224-pixel grids with many tiles.

## 5. Graph size and composition

- Request: **1.50 MB, constant in raster size** (k = 1 and k = 32 differ by
  4 bytes). 339 function invocations after the client's content-hash
  deduplication, 44 distinct algorithms, 0.04 s to build and 1.1 s to
  serialize. No "Computation is too complex"; the 10 MiB payload limit leaves
  ~6× headroom for one block.
- 97% of the bytes are the 112,347 parameters. The shortest-decimal float32
  encoding saves 40% (1.46 MB vs 2.44 MB for exact float64 decimals) and
  restores every value exactly. A full Swin-T (28 M parameters) cannot be
  inlined. The repository's table-asset weight store
  (`native_ee/weight_assets.py`) is the route for multiple blocks.
- Lazy evaluation keeps composition cheap to measure: a request for an
  intermediate serializes and computes only its subgraph (QKV prefix: 377 KB).

## 6. Earth Engine semantics found or confirmed (probed live)

1. **No broadcasting** in array-image arithmetic: `[49, 96] + [1, 96]` fails
   ("Binary array lengths should be equal"). Every per-feature parameter is a
   row-repeated `[49, C]` constant.
2. Result types: `matrixMultiply`, `arrayReduce(sum)`, `sqrt`, `erf`, `exp` →
   double; float32 ± × ÷ float32 → float32; any non-integer Python constant →
   double. `arrayReduce(mean)` *reports* float, so LayerNorm uses sum ÷ 96;
   reported band types were not relied on.
3. `erf` is double-precision accurate (≤ 7.8e-16 absolute vs `math.erf`).
4. **Explicit casts materialize copies.** An explicit `toDouble()` before
   LayerNorm and GELU added nothing numerically, because promotion already
   happens, but removing it removed a profiled `Image.double` entry. The results
   stayed bit-identical. Together with lighter verification weights (item 9),
   summed peak fell 23% and EECU·s 32% at 256 windows, and the target went from
   failing to fitting. Attribution at the target: the cast-free block with
   Milestone 1's checksum weights still failed (2 of 2), so the lighter
   verification was necessary; the casts' share alone was not isolated.
5. **Constant array images are not replicated per pixel**: a `[96, 384]`
   constant evaluated over 1,024 pixels peaks at 2.3 MB.
6. Shortest-decimal float32 constants round-trip exactly through
   `ee.Array(...).float()` (all 112,347 parameters).
7. `computePixels` cannot return array-valued pixels ("Cannot export
   array-valued tiles to NumPy"); `Image.sample()` on the coarse grid returns
   whole `[49, C]` windows with their coordinates.
8. tileScale does not reduce memory for a raster that is a single tile (§4).
9. The verification weights matter at this scale. Milestone 1's per-pixel hash
   weights cost as much as generating the input (~157 MB at 1,024 windows), so
   checksums here use a constant `[49, C]` weight array times a per-window
   scalar (`checksums.py`).
10. PyTorch side: `tensor * python_float` rounds the scalar to float32;
    float32 LayerNorm and GELU are not correctly rounded (41% / 35% of
    elements on random inputs).

## 7. Next: shifted windows

Correctness and graph size are not what will limit the next milestone; memory
headroom and tile granularity are. SW-MSA on window layout needs:

- a cyclic shift by 3 tokens, which in window layout is a re-partition: each
  shifted window takes rows and columns from up to four neighboring coarse
  pixels. That is a gather on the coarse grid (e.g. a 2×2 `neighborhoodToArray`
  plus per-token `arraySlice`/`arrayCat`), the conversion this milestone
  avoided;
- the SW-MSA mask (−100 between tokens from different source regions, per
  timm), applied via the existing additive `key_bias`;
- the reverse shift after attention.

At 224×224 the unshifted block already reports 1.63 GB, and the interactive
ceiling lies below the ~2.1 GB a 1,296-window request would need. A shifted
block will therefore not fit in one interactive request at that size without
changes. Options, in order of cost:

- batch export (4,096 windows of this block succeeded);
- ≤256-window requests (the block is window-independent until the shift; with
  a shift, regions need a one-window halo);
- moving parameters into assets to remove the 1.5 MB payload;
- trimming double temporaries in the MLP (e.g. folding the bias into the matrix
  product).

Keep the exactness test. The mirror comparison found no differences here, but
it is what makes "bit-identical except for PyTorch's own rounding" a checkable
claim for each new operation.

## 8. Caveats

- **Profiler semantics.** "Peak Mem" is EE's per-algorithm accounting, not
  RAM; image entries include their subgraph. EE does not document the
  per-request memory limit; "near the ceiling" means 1,024 windows succeed and
  1,296 fail.
- **Shared service.** Wall times are single observations and vary widely at
  the target (24–261 s for the same request); EECU·s is the steadier measure.
- **Inputs.** Rasters use generated hash tokens (uniform on [−2, 2) in steps
  of 2/2048), not features from a trained patch embedding; parameters are
  seeded, not pretrained. Input generation itself costs ~0.31 GB and 5.7
  EECU·s at the target and is included in every block measurement.
- **Scope.** One non-shifted block, input already in window layout. No shifted
  windows, patch embedding, patch merging, multiple blocks or decoder.
