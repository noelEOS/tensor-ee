# Findings: a Sentinel-1 image through the Swin-T encoder in native Earth Engine

Live experiments on 5 October 2026: Earth Engine Python API 1.7.45, PyTorch 2.14.0,
timm 1.0.30, NumPy 2.5.3, Cloud project `ORIGINAL_PROJECT`, application-default
credentials. Every forward pass ran entirely in Earth Engine computation graphs: the
Sentinel-1 scene is read from the EE catalogue inside the graph, preprocessed,
patch-embedded and encoded there; the only intermediate is an EE image asset. No
Vertex AI, Cloud Run, model server or client-side tensor computation. Inputs and
tensors were fetched only for comparison.

Evidence: numerical_comparison_m6.csv (development archive: `docs/history/swin_image_encoder/results/numerical_comparison_m6.csv`)
(summary: numerical_summary_m6.md (development archive: `docs/history/swin_image_encoder/results/numerical_summary_m6.md`)),
benchmark_results_m6.csv (development archive: `docs/history/swin_image_encoder/results/benchmark_results_m6.csv`) (summary:
benchmark_summary_m6.md (development archive: `docs/history/swin_image_encoder/results/benchmark_summary_m6.md`)), per-request records
with EE profiles in results/raw/ (archive `swin_image_encoder/results/raw/` at `research-archive-2026-10-12`) (`validation_*`, `bench/`,
`stage3_assets.json`, `graph_size.json`), the inputs' provenance in
data/image_manifest.json (archive `swin_image_encoder/data/image_manifest.json` at `research-archive-2026-10-12`), the Phase 1 semantics and
parameter digests in data/manifest.json (archive `swin_image_encoder/data/manifest.json` at `research-archive-2026-10-12`), the
patch-embedding assets in data/weights_manifest.json (archive `swin_image_encoder/data/weights_manifest.json` at `research-archive-2026-10-12`).

**Research question.** Can a real Earth Engine satellite image be transformed into
Swin-T patch tokens and passed through the complete native EE encoder while
preserving the numerical behaviour of the equivalent PyTorch/timm forward pass?
**Yes.**

- A Sentinel-1 GRD scene (VV, VH) is read on its own native 10 m grid, preprocessed
  (SegWater v3), patch-embedded (Conv2d 2 → 96, 4 × 4, + LayerNorm) and encoded by
  Milestone 5's unchanged encoder, as two chained EE batch tasks. On the canonical
  896 × 896 crop the 28 × 28 × 768 features are within **5.72e-6 of timm float32 at
  every one of their 602,112 elements** (6.4e-7 of float64; target 1e-4).
- **EE starts from exactly the reference's input.** The preprocessed tensor EE feeds
  the patch embedding equals NumPy's preprocessing of the raw dB fetched separately on
  the native grid, bit for bit, on all four crops (11.4 M values). Patch membership,
  flatten order and window layout are exact at every element. From there every
  compared tensor is bit-identical to the float64 mirror that rounds where EE rounds.
- **Going from the image to stage-1 tokens is cheap**: 7.3 EECU·s in batch at 896²,
  2.4% of task A (309 EECU·s, which is Milestone 5's synthetic task A within its own
  run-to-run spread). The patch embedding adds 3,360 parameters (17,184 embedded
  values) and leaves Milestone 5's two-task split unchanged.
- **Outputs are georeferenced**: every stage grid is the crop's own grid at 40, 80,
  160, 320 m; asset metadata, token footprints and relabels are verified.
- **The cyclic wrap is timm's, and it can be removed**: with a 672-px real-imagery
  halo the core's features are provably independent of the wrap (bit-identical under
  perturbation of the outer ring). Computed from the 448-px crop alone, the cyclic wrap
  changes every token of the same core (up to 1.85 at feature scale ~1, 12% median
  per token); 224 px of real context already reduce that to ≤ 0.5%.

**Milestone verdict.** Minimum, strong and full architectural success are all met:
the canonical 896 × 896 × 2 Sentinel-1 crop runs through native preprocessing,
native patch embedding, the complete four-stage encoder and the final LayerNorm in
the validated two-task workflow, to 28 × 28 × 768 features within 5.72e-6 of timm.
Nothing leaves Earth Engine for inference.

## Answers

| # | Question | Answer |
|---|---|---|
| 1 | What are timm's exact Patch Embedding semantics here? | **Conv2d(2 → 96, kernel 4, stride 4, padding 0, bias) then LayerNorm(96, eps 1e-5, affine)**; NCHW in (VV = 0, VH = 1), NHWC out (token (i, j) = pixels 4i..4i+3, 4j..4j+3); patch flattened (c, ky, kx); no position embedding; non-divisible extents rejected (`strict_img_size`); float32 roundings after the input, the conv (bias fused) and the LayerNorm. Established by running timm, ordering proven exactly with integer/one-hot fixtures (§1). |
| 2 | Can exact 4 × 4 patch extraction be implemented natively in EE? | **Yes**, two ways: 16 exact stride samples, or one `neighborhoodToArray`; both exact at every element of 448² and 896² crops (§3). |
| 3 | Is input pixel/channel ordering exact? | **Yes**: unique (pixel, channel) IDs built on the native georeferenced grid come out of the production chain equal to timm's own Conv2d extraction (3.2 M values, both gathers) (§3). |
| 4 | Can the 32 → 96 projection be reproduced numerically? | **Yes**: bit-identical to the mirror, ≤ 2.4e-6 from timm float32, ≤ 3.9e-7 from float64, on every token of the real crops (§4). |
| 5 | Is PatchEmbed normalization present, and does it match? | **Present** (LayerNorm(96), `patch_norm=True`); bit-identical to the mirror, ≤ 1.67e-6 from timm (§4). |
| 6 | Can a real S1 EE image be preprocessed identically to the reference? | **Yes**: SegWater v3 in EE equals NumPy and torch bit for bit on a 94-case tie/clamp fixture and on 11.4 M real values (§2). |
| 7 | Are the exact same source pixels used by EE and PyTorch? | **Yes**: crops on the native grid (no resampling); EE's preprocessed input equals the reference's at every pixel of all four crops. EE's raw doubles jitter by 1-2 ulp between requests; the float32 cast absorbs it (margin 4×) (§2). |
| 8 | Does Patch Embedding output match PyTorch? | **Yes**: 1.67e-6 (canonical, 4.8 M values), bit-identical to the mirror (§4). |
| 9 | Can Patch Embed output go directly into the Stage-1 window layout? | **Yes**: the gather writes [1, 32] rows on the token grid, 49 stride samples write [49, 32] windows, the projection and LayerNorm run per window: Milestone 5's stage-1 input with no fine token raster (§3). |
| 10 | Does the complete real-image → encoder forward pass match timm? | **Yes**: 448, 896, 1792 and the swath-edge crop, every boundary, as `SwinTransformer(in_chans=2).forward_features` (§5). |
| 11 | What is the final max absolute error? | **5.72e-6** on the canonical 28 × 28 × 768 features (602,112 elements; MAE 6.4e-7, RMSE 8.3e-7, max relative to max 1.3e-6); 6.08e-6 at 1792 (§5). |
| 12 | Where does the first numerical divergence appear? | **At the patch embedding's Conv2d, and only in PyTorch**: input, preprocessing, gather and partition are exact; EE is bit-identical to the mirror at every tensor, so EE − timm is PyTorch's float32 rounding (§5). |
| 13 | Does the controlled-rounding mirror remain bit-identical to EE? | **Yes, everywhere: 98 of 98 compared tensors, 56.8 M elements** (448, 896, 1792 and edge crops; patch embedding to final norm, every stage-4 intermediate of the 448 crop, and the convolution formulation) (§4, §5). |
| 14 | Does error accumulate differently on real imagery? | **No**: ~1e-5 at every boundary, renormalized by merges and the final norm, as on synthetic tokens; real tokens are smoother (slightly larger early residual stream, slightly more peaked stage-4 attention) (§5). |
| 15 | Are the Stage-1/2/3/4 grids geospatially aligned? | **Yes**: EPSG:32651, the crop's origin, 40/80/160/320 m; asset metadata exact, 40/40 random pixels per stage map to the right token, relabels exact (§6). |
| 16 | How are input masks handled? | **Explicitly**: masked values are filled with the training mean in dB before quantization (deterministic, timm sees the same tensor); a per-token valid fraction is computed separately. EE's implicit masks would delete the whole output by stage 2 (§7). |
| 17 | How are partially valid patches handled? | **Embedded like any patch**, their masked values at the training mean; their valid fraction (k/32) flags them (49 partial tokens on the edge crop, exact) (§7). |
| 18 | Does cyclic boundary handling reproduce timm exactly? | **Yes**: Milestone 5's per-block extension on real crops; every tensor equals timm within float32 rounding and the mirror bit for bit (§5, §8). |
| 19 | Can a real-imagery halo remove cyclic-edge effects for interior inference? | **Yes**: with a 672-px halo the core is provably wrap-free (bit-identical under perturbation of the outer ring); the cyclic 448 crop differs from it by up to 1.85 (12% median per token), the 896 crop (224 px of context) by ≤ 0.022 (§8). |
| 20 | How large is the real-image context requirement? | **644 px = 6.44 km** each side at 10 m (23 stage-1 windows); **672 px = 6.72 km** aligned to stage-4 windows; in practice 224 px (2.24 km) gets within 0.5% (§8). |
| 21 | What does preprocessing cost? | **~3-6 EECU·s** interactively at 896² (reading the crop ~1) (§9). |
| 22 | What does Patch Embedding cost? | **7.3 EECU·s in batch** at 896² (gather + partition + projection + LayerNorm, including the read and preprocessing); 9-12 interactively; 0.05 EECU·s to read its parameters (§9). |
| 23 | What fraction of task A is Patch Embedding vs the transformer? | **~2.5-3%** (7.3 of ~260 EECU·s); task A from real imagery costs the same as Milestone 5's from synthetic tokens (258 vs 264 mean) (§9). |
| 24 | Does the two-task M5 split remain sufficient? | **Yes**, unchanged: task A (input → stage 3 → georeferenced asset), task B (→ PM 3 → stage 4 → norm), every run successful (§9). |
| 25 | Do the graphs stay below the embedded-constant cap? | **Yes**: task A 14.27 M (+17,184 for the patch embedding), task B 16.65 M, cap 24-26 M; graphs 728 KB / 360 KB (§9). |
| 26 | What is the largest real-image crop processed? | **1792 × 1792 px (17.92 km) → 448² stage-1 tokens → 56 × 56 × 768**, within 6.08e-6 of timm (§5, §9). |
| 27 | Does performance scale like M5's synthetic tests? | **Yes**: roughly with the stage-1 window count (×3.8 from 896 to 1792 vs Milestone 5's ×3.0, under concurrent load); same interactive peak as Milestone 5 (§9). |
| 28 | What is the preferred Patch Embedding formulation? | **Stride gather + window-layout projection**: cheapest in batch (7.3 vs 12.6 EECU·s for `neighborhoodToArray`, ~13× less than a direct `convolve` formulation, which is also bit-exact), 10× less memory than per-token projection, asset-backed (§9). |
| 29 | Does the complete pipeline remain entirely inside Earth Engine? | **Yes**: the scene is read from the EE catalogue inside task A's graph; preprocessing, patch embedding and the encoder are EE operations; the only intermediate is an EE image asset; downloads were for validation only. |
| 30 | Should the project proceed to M7? | **Yes** (§11). |

## 1. timm's patch embedding, determined experimentally (Phase 1)

Every statement below was established by running timm 1.0.30
(reference/pytorch_image_encoder.py (archive `swin_image_encoder/reference/pytorch_image_encoder.py` at `research-archive-2026-10-12`)
`patch_embed_semantics`, recorded in data/manifest.json (archive `swin_image_encoder/data/manifest.json` at `research-archive-2026-10-12`); tests
in tests/test_patch_embed.py (archive `tests/test_patch_embed.py` at `research-archive-2026-10-12`)):

| property | `SwinTransformer(in_chans=2, embed_dim=96, patch_size=4)` |
|---|---|
| module | `PatchEmbed(proj=Conv2d(2, 96, kernel_size=(4, 4), stride=(4, 4)), norm=LayerNorm((96,), eps=1e-05))` |
| kernel / stride / padding / dilation / groups | 4 × 4 / 4 × 4 / 0 / 1 / 1 |
| bias | yes (`proj.bias` [96]) |
| normalization | LayerNorm(96), eps 1e-5, affine weight and bias (`patch_norm=True`) |
| input layout | NCHW [1, 2, H, W]; channel 0 = VV, 1 = VH |
| output layout | NHWC [1, H/4, W/4, 96] (`output_fmt='NHWC'`, no flatten): token (i, j) at row i, column j |
| flatten order of a patch | (c, ky, kx): Conv2d `weight.reshape(96, -1)` order; token (i, j) reads input rows 4i..4i+3, columns 4j..4j+3 |
| position embedding | none (`absolute_pos_embed` is None) |
| extents not divisible by 4 | rejected: `strict_img_size=True` asserts H, W equal `img_size`; with `strict_img_size=False` it asserts divisibility; only `dynamic_img_pad=True` (off) would zero-pad bottom/right |
| float32 rounding points | the float32 input; the Conv2d including its bias (one kernel: in 63% of elements its result differs from conv-then-add-bias); the LayerNorm |

The ordering facts are proven exactly, not approximately: with an integer input and
one-hot weights, timm's own Conv2d extracts element k = (c, ky, kx) of every patch,
equal to `unfold`; with integer weights too, `proj(x)` equals the im2col product
`patches @ W^T + b` bit for bit. So the Conv2d is exactly an im2col matrix product,
which is how EE computes it. The explicit forward runs timm's `proj` and `norm`
submodules step by step (bit-identical to `patch_embed(x)`), then Milestone 5's
explicit encoder, and raises unless the final output equals
`model.forward_features(x)` bit for bit, at every crop size and in float32 and
float64.

The patch embedding's parameters are seeded like Milestone 5's (group `pe`, seed 16):
weight ~ N(0, 1/32) (unit-variance output for unit-variance input), bias ~ N(0, 0.1²),
LayerNorm γ ~ 1 + N(0, 0.2²), β ~ N(0, 0.2²). The encoder keeps Milestone 5's
parameters; no trained checkpoint is involved.

## 2. Exactly the same input (Phases 3, 4, 23; Tests 6, 7)

**Scene.** `COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20240103T100134_20240103T100159_051941_06469E_B0BD`:
Sentinel-1A, IW, GRD high resolution, dual polarization VV + VH, ascending pass,
relative orbit 69, slice 2, acquired 2024-01-03 10:01:34 UTC (the Sun Moon Lake
scene of the SegWater real-chip experiments). EE stores VV and VH in dB as **double**,
on EPSG:32651 with crs_transform [10, 0, 169540.95335071225, 0, -10, 2780488.8405420803]
(28,431 × 21,709 px).

**Crops** (scene.py (archive `swin_image_encoder/scene.py` at `research-archive-2026-10-12`), data/image_manifest.json (archive `swin_image_encoder/data/image_manifest.json` at `research-archive-2026-10-12`)),
all on the native grid translated by whole pixels:

| crop | native col, row | size | stage-1 windows | role |
|---|---|---|---|---|
| small_448 | 11590, 13741 | 448 × 448 | 16 × 16 | small end-to-end (every stage shifts) |
| canonical_896 | 11366, 13517 | 896 × 896 | 32 × 32 | the M6 target |
| large_1792 | 10918, 13069 | 1792 × 1792 | 64 × 64 | largest; small_448 + 672-px halo |
| edge_224 | 1894, 13853 | 224 × 224 | 8 × 8 | straddles the western swath edge (49% valid) |

The first three are nested around Sun Moon Lake with offsets that are multiples of
224 px (one stage-4 window), so every Swin window of a smaller crop is a window of the
larger ones. The SegWater chips used an integer-metre origin (286030, 2640860), 0.9 px
off the native pixel edges; nearest-neighbour sampling still copied native pixels one
to one there, but the geolocation was off by ~9 m. Every M6 grid is native-aligned.

**No resampling, explicit relabel.** Inside the graph the crop is read with
`reproject(crop grid)` (same CRS, same scale, integer offset: a one-to-one copy of
native pixels), preprocessed there, and relabelled with
`changeProj(crop grid, abstract grid)` onto the EPSG:3857, origin-(0, 0) grid Milestones
1-5 compute on, at 2.5 units per pixel (so a 4 × 4 patch is one 10-unit token of
Milestone 5's token grid). Milestone 1 had only proven same-CRS relabelling; the
first live probe of M6 verified the cross-CRS relabel exactly (pixel IDs and S1
values).

**Preprocessing** (Phase 3): the audited SegWater v3 recipe
(native_ee/segwater_preprocessing.py (archive `native_ee/segwater_preprocessing.py` at `research-archive-2026-10-12`)),
chosen as a fixed documented transform (there is no trained checkpoint behind it):

    x  = float32(dB)                          (EE: toFloat of the stored double)
    q  = clamp(round_half_even((x + 55) / 0.05), 1, 65535)    float32 after each op
    dB'= q * 0.05 - 55
    z  = (dB' - mean_c) / std_c               mean = (-15.435106, -24.856323),
                                              std  = (6.517790, 9.117346), float32

EE casts to float32 after every operation, so each step is a correctly rounded
float32 operation, identical to NumPy's and torch's float32 arithmetic.

| check (live) | values | result |
|---|---:|---|
| numeric fixture (94 dB cases: q55s005 ties and neighbours, clamps, random) × 2 channels through EE vs NumPy and vs torch | 188 | **bit-identical** to both |
| small_448: EE's preprocessed input (fetched on the abstract model grid) vs NumPy preprocessing of the raw dB fetched on the native crop grid | 401,408 | **bit-identical** |
| canonical_896 | 1,605,632 | **bit-identical** |
| large_1792 | 6,422,528 | **bit-identical** |
| edge_224 (masked pixels filled, training-mean policy) | 100,352 | **bit-identical** |

One comparison checks the source pixels, the grid, the relabel and the preprocessing
together: **EE and PyTorch start from the same input values at the same pixel
positions.**

**A new EE behaviour: the stored dB values jitter.** Two identical `computePixels`
requests of the raw bands return doubles that differ by 1-2 ulp (≤ 1.42e-14 dB) in
2.1-4.5% of the pixels (all four crops; the first probe showed the same between a
direct read and a relabelled read). The jitter vanishes at the float32 cast the
preprocessing starts with: the float32 values were equal in every repeat, and the
closest valid value to a float32 rounding midpoint is 5.7e-14 dB away (canonical and
large; 3.1e-13 small, 8.7e-12 edge), 4× the largest jitter observed. So the model
input is deterministic for these crops, but the margin is finite: a value within
1.4e-14 dB of a float32 midpoint could round either way between requests. The
validation therefore compares EE's own preprocessed input with the reference's, not
just the raw values.

## 3. Patch extraction and window layout (Phases 5, 6, 8; Tests 1, 2, 5)

earth_engine/patch_embed.py (archive `swin_image_encoder/earth_engine/patch_embed.py` at `research-archive-2026-10-12`) gathers each token's
4 × 4 × 2 patch as a [1, 32] row in (c, ky, kx) order on Milestone 5's token grid,
then partitions the rows into [49, 32] windows (49 exact stride samples), so the
patch embedding writes Milestone 5's stage-1 window layout directly; no fine token
raster of the embedding is ever formed. Two gathers:

- **stride** (production): 16 exact stride samples of the input (Milestone 1's
  reproject + changeProj space-to-depth at factor 4), sample ky·4 + kx holding input
  pixel (4i + ky, 4j + kx);
- **neighborhood**: one `neighborhoodToArray` with a 4 × 4 kernel anchored at its
  top-left, one corner stride sample.

**Patch membership and order** (unique IDs (y·W + x)·2 + c built from
`pixelCoordinates` of the crop's *native georeferenced* grid, through the production
reproject → relabel → gather → partition, against timm's own Conv2d with one-hot
weights in window layout):

| crop | gather | values | result |
|---|---|---:|---|
| small_448 (16 × 16 windows) | stride | 401,408 | **exact** |
| small_448 | neighborhood | 401,408 | **exact** |
| canonical_896 (32 × 32 windows) | stride | 1,605,632 | **exact** |
| canonical_896 | neighborhood | 1,605,632 | **exact** |

Every element of every window: the source pixel of each patch element (top-left,
interior, final row, final column and bottom-right patches included), the channel
order, the (c, ky, kx) flatten order, the token order within each window (7a + b)
and the window order (I·kw + J). The same comparison runs offline as a step-for-step
NumPy emulation of both gathers on three grids
(tests/test_image_windows.py (archive `tests/test_image_windows.py` at `research-archive-2026-10-12`)). Projecting the
window arrays is bit-identical to projecting tokens and partitioning afterwards
(offline, and live: §9).

## 4. Patch-embedding numerics and weights (Phases 7, 9, 10; Tests 3, 4, 8, 12, 13)

**Projection and LayerNorm** on the real crops, every token, asset parameters:

| crop | tensor | elements | MAE | max \|EE − torch32\| | max \|EE − float64\| | max \|torch32 − float64\| | EE vs mirror |
|---|---|---:|---:|---:|---:|---:|---|
| small_448 | patches | 401,408 | 0 | 0 | 0 | 0 | bit-identical |
| small_448 | Conv2d + bias | 1,204,224 | 7.3e-8 | 1.91e-6 | 3.7e-7 | 1.90e-6 | bit-identical |
| small_448 | LayerNorm(96) (stage-1 tokens) | 1,204,224 | 9.1e-8 | 1.67e-6 | 4.0e-7 | 1.71e-6 | bit-identical |
| canonical_896 | patches | 1,605,632 | 0 | 0 | 0 | 0 | bit-identical |
| canonical_896 | Conv2d + bias | 4,816,896 | 8.4e-8 | 2.38e-6 | 3.9e-7 | 2.17e-6 | bit-identical |
| canonical_896 | LayerNorm(96) | 4,816,896 | 9.3e-8 | 1.67e-6 | 4.2e-7 | 1.71e-6 | bit-identical |

Max error 2.4e-6, far below 1e-4, and all of it PyTorch's float32 Conv2d: EE
accumulates the 32 products in double (sequentially, which torch's float64 matmul
reproduces; checked offline) and rounds once, so it sits within one float32 rounding
of float64. **PatchEmbed normalization is present and matches** (Q5).

**Mirror** (reference/ee_mirror_image_encoder.py (archive `swin_image_encoder/reference/ee_mirror_image_encoder.py` at `research-archive-2026-10-12`)):
the preprocessing is float32 after every operation (the NumPy reference itself); the
projection is a float64 dot product of 32 float32 products plus bias, rounded once;
the LayerNorm is Milestone 2's formula with EE's left-to-right sums, rounded once;
the window partition moves values. Then Milestone 5's mirror. Bit-identical at
every compared tensor (§5).

**Asset-backed parameters** (earth_engine/image_weights.py (archive `swin_image_encoder/earth_engine/image_weights.py` at `research-archive-2026-10-12`)):
the 3,360 parameters as Milestone 4-5 records (`patch_embed.proj` [96, 32] + bias,
`patch_embed.norm.weight`, `.bias` [1, 96]) in one table asset of 77,677 bytes;
upload 70 s (one export task); read back **bit for bit** (float32 bit patterns of all
3,360 values). Reading them costs 0.05 EECU·s and 0.8 s; the asset-backed patch
embedding graph is 75,402 bytes, the inline one 112,915. The asset-backed and inline graphs give the
same stage-1 tokens (max difference 0 over the 448 crop). The conv weight is stored
flattened (the im2col order) and reshaped to [96, 2, 4, 4] only for timm, because
the shared record code treats 2-D weights as linear layers.

## 5. End to end: Sentinel-1 crop → encoder features (Phases 11-14; Tests 9, 14, 15)

Every run is the production path: S1 scene → crop grid → preprocessing → relabel →
patch embedding (stride gather, window placement, asset parameters) → Milestone 5's
encoder with per-block extension, as task A (→ georeferenced stage-3 asset) and task
B (asset → PM 3 → stage 4 → norm). References: timm float32 and float64 and the EE
mirror, all from the reference's own preprocessing of the downloaded raw dB.

**Small crop (448 × 448, 16 × 16 stage-1 windows, every stage shifts), every element
of every boundary and block output**, fetched interactively (head) and from the
task-A asset (tail, which also fetches every stage-4 intermediate):

| tensor | elements | MAE | max \|EE − torch32\| | max \|EE − float64\| | EE vs mirror |
|---|---:|---:|---:|---:|---|
| patches | 401,408 | 0 | 0 | 0 | bit-identical |
| patch embedding (stage-1 tokens) | 1,204,224 | 9.1e-8 | 1.67e-6 | 4.1e-7 | bit-identical |
| stage 1, block 0 (W-MSA) | 1,204,224 | 4.0e-7 | 4.77e-6 | 1.6e-6 | bit-identical |
| stage 1 output (SW-MSA) | 1,204,224 | 6.2e-7 | 5.72e-6 | 2.6e-6 | bit-identical |
| Patch Merging 1 | 602,112 | 4.0e-7 | 4.53e-6 | 7.1e-7 | bit-identical |
| stage 2, block 0 | 602,112 | 7.4e-7 | 6.44e-6 | 1.3e-6 | bit-identical |
| stage 2 output | 602,112 | 9.9e-7 | 7.87e-6 | 1.6e-6 | bit-identical |
| Patch Merging 2 | 301,056 | 5.6e-7 | 5.72e-6 | 5.9e-7 | bit-identical |
| stage 3, blocks 0 / 1 / 2 / 3 / 4 | 301,056 each | 1.0e-6 … 2.0e-6 | 8.34e-6 / 9.78e-6 / 1.06e-5 / 1.22e-5 / 1.31e-5 | 1.2e-6 … 2.6e-6 | bit-identical (5 of 5) |
| stage 3 output (head; and read back from the task-A asset by the tail) | 301,056 | 2.2e-6 | 1.35e-5 | 2.8e-6 | bit-identical (both) |
| Patch Merging 3 | 150,528 | 7.8e-7 | 7.63e-6 | 5.2e-7 | bit-identical |
| stage 4, every intermediate of both blocks (44 tensors: LN, QKV, per-head scores, biased and masked logits, softmax, attention, projection, shifts, MLP) | 9,264,724 | — | 2.29e-5 (masked logits) | — | bit-identical (44 of 44) |
| stage 4, block 0 | 150,528 | 1.3e-6 | 1.10e-5 | 1.1e-6 | bit-identical |
| stage 4 output | 150,528 | 1.8e-6 | 1.27e-5 | 1.5e-6 | bit-identical |
| **final norm** | **150,528** | **7.0e-7** | **5.13e-6** | **6.3e-7** | **bit-identical** |

The head ran as interactive requests (one or two tensors per request; stage-3 requests
recompute the head, 40-180 EECU·s each), the tail from task A's georeferenced stage-3
asset (100 EECU·s). Two stage-4 tensors of the shifted block ([49, 3072] MLP hidden
and its fc2) intermittently exceeded EE's interactive memory limit even one row at a
time (the decoded stage-4 weights alone take ~900 MiB); they succeeded on retry (the
driver can also fetch such tensors by batch export).

**Canonical crop (896 × 896 → 28 × 28 × 768), two batch tasks, every element of
the exported georeferenced rasters:**

| boundary | elements | MAE | RMSE | max \|EE − torch32\| | max rel. to max \|torch32\| | max \|EE − float64\| | max \|torch32 − float64\| | EE vs mirror |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| preprocessed input | 1,605,632 | 0 | 0 | 0 | 0 | 0 | 0 | bit-identical |
| patch embedding (stage-1 tokens) | 4,816,896 | 9.3e-8 | 1.4e-7 | 1.67e-6 | 3.2e-7 | 4.2e-7 | 1.71e-6 | bit-identical |
| stage 1 output | 4,816,896 | 5.9e-7 | 7.6e-7 | 7.87e-6 | 7.3e-7 | 2.0e-6 | 8.4e-6 | bit-identical |
| stage 2 output | 2,408,448 | 9.1e-7 | 1.2e-6 | 8.23e-6 | 7.8e-7 | 1.6e-6 | 8.2e-6 | bit-identical |
| stage 3 output (task A asset) | 1,204,224 | 2.0e-6 | 2.5e-6 | 1.55e-5 | 1.0e-6 | 3.2e-6 | 1.4e-5 | bit-identical |
| stage 4 output (task B) | 602,112 | 1.7e-6 | 2.1e-6 | 1.14e-5 | 1.0e-6 | 1.4e-6 | 1.2e-5 | bit-identical |
| **final norm (encoder features)** | **602,112** | **6.4e-7** | **8.3e-7** | **5.72e-6** | **1.3e-6** | **6.4e-7** | **5.4e-6** | **bit-identical** |

Weighted checksums and sums of squares of every exported tensor agree with float64
to < 1e-5 relative (Milestone 1's normalization) and with the mirror exactly. The
stage-1 and stage-2 outputs were exported by their own batch tasks from the same
head graph (83 and 120 EECU·s).

**Large crop (1792 × 1792 → 56 × 56 × 768), two batch tasks:**

| boundary | elements | MAE | max \|EE − torch32\| | max \|EE − float64\| | EE vs mirror |
|---|---:|---:|---:|---:|---|
| stage 3 output (task A asset) | 4,816,896 | 1.9e-6 | 1.43e-5 | 3.3e-6 | bit-identical |
| stage 4 output | 2,408,448 | 1.6e-6 | 1.43e-5 | 1.5e-6 | bit-identical |
| **final norm** | **2,408,448** | **6.2e-7** | **6.08e-6** | **7.2e-7** | **bit-identical** |

**Swath-edge crop (224 × 224, 49% valid, training-mean fill)**: every boundary from
the patch embedding to the final norm bit-identical to the mirror, final norm within
4.53e-6 of timm (§7).

**Error through the encoder on real imagery vs Milestone 5's synthetic tokens**
(max |EE − torch32|; Milestone 5's canonical stage-1/2 boundaries were checked by
checksums only):

| boundary | M6 448 crop | M5 16 × 16 windows | M6 896 crop | M5 32 × 32 | M6 1792 crop | M5 64 × 64 |
|---|---:|---:|---:|---:|---:|---:|
| stage-1 tokens (patch embedding) | 1.67e-6 | — (input) | 1.67e-6 | — | — | — |
| stage 1 output | 5.72e-6 | 9.2e-6 | 7.87e-6 | checksum ✓ | — | — |
| Patch Merging 1 | 4.53e-6 | 4.8e-6 | — | checksum ✓ | — | — |
| stage 2 output | 7.87e-6 | 1.1e-5 | 8.23e-6 | checksum ✓ | — | — |
| Patch Merging 2 | 5.72e-6 | 6.3e-6 | — | checksum ✓ | — | — |
| stage 3 output | 1.35e-5 | 1.6e-5 | 1.55e-5 | 1.64e-5 | 1.43e-5 | 1.72e-5 |
| Patch Merging 3 | 7.63e-6 | 7.9e-6 | — | checksum ✓ | — | — |
| stage 4 output | 1.27e-5 | 1.3e-5 | 1.14e-5 | 1.24e-5 | 1.43e-5 | 1.43e-5 |
| **final norm** | **5.13e-6** | **5.0e-6** | **5.72e-6** | **5.48e-6** | **6.08e-6** | **5.96e-6** |

**Where does divergence first appear (Phase 14)?** Not in the input: the source
pixels, the preprocessing, the patch gather and the window partition are exact
(0 difference from the reference at every element). The first tensor that differs
from timm float32 is the patch embedding's Conv2d output (2.4e-6), and that
difference is PyTorch's: EE is bit-identical to the mirror there and at every later
tensor (EE − torch32 = (EE − mirror) + (mirror − torch32), first term 0). EE stays
closer to float64 than PyTorch at every boundary (final norm 6.4e-7 vs 5.4e-6).

**Does real imagery change the error behaviour (Q14)?** No. Real S1 inputs make
smoother, spatially correlated tokens: the residual stream is a little larger in
stages 1-2 (standard deviation 2.4 vs 2.0 at the stage-1 output, 2.6 vs 2.2 at stage
2), identical by stage 4 (2.57 vs 2.58), and stage-4 attention is slightly more peaked
(mean row entropy 3.08 / 2.58 nats vs 3.27 / 2.71 for the two blocks). The error
profile is the same as on Milestone 5's hash tokens: ~1e-5 at every boundary,
renormalized by each Patch Merging and the final LayerNorm, no accumulation with
depth or raster size.

## 6. Georeferenced outputs (Phase 15; Test 11)

Every exported tensor is a token raster on the crop's own grid: EPSG:32651, the
crop's origin (283200.95335071225, 2645318.8405420803 for the canonical crop), and
4 · 2^(s−1) input pixels per token:

| tensor | grid | asset metadata (crs, transform, dimensions, bands) | 40 random native pixels → token (row // f, col // f) |
|---|---|---|---|
| stage-1 output | 40 m, 224 × 224, 96 bands | exact | 40 / 40 |
| stage-2 output | 80 m, 112 × 112, 192 bands | exact | 40 / 40 |
| stage-3 output (task A asset) | 160 m, 56 × 56, 384 bands | exact | 40 / 40 |
| stage-4 output | 320 m, 28 × 28, 768 bands | exact | 40 / 40 |
| final norm (encoder features) | 320 m, 28 × 28, 768 bands | exact | 40 / 40 |

The footprint check samples each exported asset at the centres of random native
pixels of the crop (in EPSG:32651) and requires the value of token
(row // f, col // f) of the downloaded raster: stage-4 token (i, j) covers exactly
input pixels 32i..32i+31, 32j..32j+31, i.e. native rows r0 + 32i.. and columns
c0 + 32j... The relabel itself is exact at every stage (token IDs on the abstract
stage-s grid relabelled onto the crop's grid: 50,176 / 12,544 / 3,136 / 784 IDs, all
equal), and the inbound relabel is covered by the ID test of §3, which builds its IDs
on the georeferenced native grid. Effective resolutions: input 10 m; stage 1 40 m
(×4); stage 2 80 m (×8); stage 3 160 m (×16); stage 4 320 m (×32).

The encoder still computes on Milestone 5's abstract grids; only the boundaries are
relabelled (input → model grid, model grid → crop grid, and the stage-3 asset back
in task B), each an exact `changeProj` between grids of equal shape. Task A's
stage-3 asset is georeferenced, so the intermediate can be displayed or reused like
any EE image.

## 7. Masks and scene edges (Phase 16; Test 10)

The `edge_224` crop straddles the scene's western swath edge (49.45% of its pixels
valid; VV and VH masks are binary and identical): 1,527 stage-1 tokens are fully
valid, 49 partially valid, 1,560 fully masked; 27 of its 64 stage-1 windows contain
no masked pixel. timm has no masks: it takes a dense tensor. Three policies were run
through the native encoder:

| policy | what EE does | encoder vs timm on the same dense tensor |
|---|---|---|
| **training_mean** (production) | fill each masked channel with its training mean in dB *before* quantization (SegWater v3's recipe for missing context): z = (−0.00229, +0.00069) | every boundary through the final norm: max 4.53e-6 vs torch32, **bit-identical to the mirror** (11 tensors) |
| zero | preserve, then z = 0 exactly | head through stage 3: max 1.22e-5, bit-identical (8 tensors) |
| preserve (EE's own masks) | a token with any masked value masks its window array (`toArray`), shifted windows spread the hole, Patch Merging needs all four windows | unmasked windows: 27 / 64 at the stage-1 tokens and after the W-MSA block, **9 / 64** after the SW-MSA block, **0** from Patch Merging 1 on: the whole output is masked |

So EE's implicit mask propagation is not a usable semantics: one masked pixel removes
its 28 × 28-pixel window, the shifted windows remove their neighbours, and by stage
2 nothing is left. The production path is dense and deterministic: masked inputs
are filled by the training-mean recipe before the model, timm sees the same filled
tensor, and a separate per-token **valid fraction** (the fraction of the 32 patch
values that were valid; `patch_embed.valid_fraction`) records which tokens rest on
real data. It is exact live (every token of the edge crop: 1 for fully valid, 0 for
fully masked, k/32 for the 49 partial tokens). Partially valid patches are therefore
embedded like any other patch, with their masked values at the training mean;
downstream code (Milestone 7) can down-weight or mask outputs whose receptive field
is mostly fill.

The fill value matters. Training-mean and zero fill differ by up to 0.048 at the
stage-1 tokens, only in the 37 windows that contain masked pixels (0 in the 27 fully
valid windows), but by stage 2 the shifted windows and the cyclic wrap have carried
the difference to every window (up to 0.18 at the stage-3 output, whose values reach
13). Fill is a modelling decision with a visible effect, not an implementation
detail; M6 fixes it to the documented SegWater recipe. The canonical crops contain no
masked pixel at all (validity 1.0000), and AOI boundaries are not masks: tokens
outside the crop are masked and replaced by the timm cyclic wrap (§8).

## 8. Cyclic boundary and real-context halo (Phases 17-18)

**The primary path keeps timm's semantics exactly.** timm processes one finite image:
its shifted windows roll the tensor, so tokens at the left edge attend (masked) to
tokens rolled in from the right edge, and every stage-s window at the edge of the
crop reads windows from the opposite side. Milestone 5's per-block periodic
extension reproduces this for any input; on real crops it does too (every compared
tensor equals timm within float32 rounding and the mirror bit for bit, §5). This is
a **model boundary convention, not geography**: pixels at the western edge of the
crop are not neighbours of those at its eastern edge. M6 does not replace it in the
correctness experiments.

**Separately: can real neighbouring imagery remove it for an interior target?**
The encoder output's dependency halo is 23 stage-1 windows = 161 tokens =
**644 input pixels = 6.44 km** on each side at 10 m (Milestone 5's analysis; window
granularity, every stage shifting). Swin windows must also coincide, so a halo must
be a multiple of one stage-4 window (224 px): the smallest aligned halo is **672 px
(6.72 km)**. The nested crops were built for this: `small_448` is the target core,
`canonical_896` surrounds it with 224 px of real imagery, `large_1792` with 672 px.

| test (float64 reference, final norm of the 448 core) | result |
|---|---|
| large_1792 with its outer 28-px ring (one window, outside the core's halo) negated | core **bit-identical** (every one of 150,528 values); every other stage-4 window changes |
| large_1792 with its outer 56-px ring (two windows: 644 px from the core edge) negated | core changes (max 3.3e-6), in the four token rings nearest its edge: the halo is tight |

So inside the 1792 crop the core's features depend only on real imagery within
644 px: they are what an infinitely large image would give, independent of the wrap
and of anything beyond the halo. Compared with the cyclic finite-crop features of
the same 448 × 448 pixels:

| core features computed from | max \|Δ\| vs the real-context core | mean \|Δ\| | median / max per-token relative L2 | min cosine | max \|Δ\| by stage-4 token distance to the core edge (0 … 6) |
|---|---:|---:|---:|---:|---|
| the 448 crop alone (cyclic at its own edges) | **1.85** | 0.11 | 12% / 38% | 0.93 | 1.58, 1.67, 1.73, 1.86, 0.30, 0.25, 0.22 |
| the 896 crop (224 px of real context, cyclic beyond) | 0.022 | 0.0012 | 0.14% / 0.48% | 0.99999 | 0.022, 0.022, 0.020, 0.008, 0.001, 0.001, 0.001 |

(Encoder features: standard deviation 1.02, max 4.4.) Without context, the cyclic
wrap changes **every** token of a 448 crop, its centre included, because the
dependency halo (644 px) exceeds half the crop (224 px): edge tokens by up to 40% of
their norm. One stage-4 window (224 px) of real context already removes almost all
of it (≤ 0.5%), well inside the formal halo: the effective receptive field decays
much faster than the window-granularity dependency bound. The same comparison on EE's
own outputs (the 448 crop's task B, the canonical and large crops' exported final
norms) gives the same numbers to within 1e-6 (1.85 and 0.022; EE's real-context core
is within 5.9e-7 of float64), so this is a property of the model, measured natively.

**Answer (Q19, Q20).** Yes: supply real imagery around the target. A 672-px
(6.72 km) aligned halo makes the interior features provably wrap-free; 224 px
(2.24 km) already brings them within 0.5% of that. For tiled map production the
cost is the halo's area: a 448 core inside 1792 computes 16× its pixels; a
4,480 × 4,480 tile (16 × 16 stage-4 windows) inside its 672-px halo computes 1.69×.

## 9. Cost, formulations and the two-task split (Phases 19-22)

**From the image to stage-1 tokens** (canonical crop, interactive, one request per
cumulative prefix, two repeats; EE profiler totals):

| prefix | EECU·s (r0, r1) | peak | request bytes |
|---|---|---:|---:|
| read S1 VV/VH on the crop grid, relabel | 0.8, 1.4 | 14 MiB | 2.4 KB |
| + SegWater v3 preprocessing | 4.1, 7.0 | 14 MiB | 7.5 KB |
| + patch gather ([1, 32] rows, 16 stride samples) | 3.0, 5.9 | 14 MiB | 29 KB |
| + window partition ([49, 32] windows) | 9.4, 6.6 | 17 MiB | 66 KB |
| + projection (Conv2d as [49, 32] × [32, 96] + bias) | 9.8, 7.4 | 17 MiB | 70 KB |
| + LayerNorm(96): the stage-1 tokens | 4.8, 8.3 | 137 MiB | 77 KB |

Each prefix costs 1-10 EECU·s and the differences are within the service's
run-to-run noise: the whole path from the S1 scene to the stage-1 window layout is
**~5-12 EECU·s interactively and 7.3 EECU·s in batch** (29 s run). Preprocessing
(~3-6 EECU·s) and the gather/partition (~3-5) are the measurable parts; the
32 → 96 projection and the LayerNorm are negligible. Reading the 3,360 patch-embedding
parameters costs 0.05 EECU·s.

**Patch-embedding formulations** (all numerically equivalent; full patch embedding at
the canonical crop):

| gather | projection | interactive EECU·s (r0, r1) | peak | request | batch EECU·s | equal to the mirror |
|---|---|---|---:|---:|---:|---|
| **stride** (16 stride samples) | **window** ([49, 32] per window) | **11.6, 9.1** | **137 MiB** | 77 KB | **7.3** | bit-identical (production; §4) |
| stride | token ([1, 32] per token, then partition) | 11.9, 10.5 | 1,385 MiB | 77 KB | — | same values (bit-identical by construction) |
| neighborhood (4 × 4 `neighborhoodToArray`) | window | 9.7, 12.3 | 137 MiB | 58 KB | 12.6 | IDs exact (§3) |
| neighborhood | token | 13.6, 10.1 | 1,400 MiB | 58 KB | — | — |
| direct convolution (96 × 2 `Image.convolve`) | token | 130, 141 | 1,398 MiB | 260 KB (inline kernels) | — | **bit-identical** on all 1,204,224 projection values of the 448 crop |

`Image.convolve` accumulates in double in row-major kernel order (probed: equal to a
sequential double sum), so even the convolution formulation reproduces the mirror; it
is ~13× the EECU·s because it evaluates every input pixel (16-fold redundant at stride
4) for 192 kernel products, and its kernels must be inline. Projecting per token on
the fine grid costs 10× the memory of projecting window arrays. **Preferred: stride
gather, window placement**: cheapest in batch (7.3 vs 12.6 EECU·s), smallest memory,
asset-backed weights, and it writes Milestone 5's window layout directly.

**Task A from real imagery vs Milestone 5's task A from synthetic tokens** (batch,
canonical size, per-block, both exporting the stage-3 output; the same session,
alternating):

| input | EECU·s | run s |
|---|---|---|
| Milestone 5: generated stage-1 tokens (224 × 224 × 96) | 256.8, 259.0 | 180, 133 |
| Milestone 6: Sentinel-1 crop → preprocessing → patch embedding | 252.8, 229.0, 309.4 (validation run) | 150, 267, 146 |

Real-imagery task A costs the same as synthetic task A (means 258 vs 264 EECU·s;
Milestone 5 measured 229-305 for its own per-block head). **The patch embedding is
~2.5-3% of task A** (7.3 of ~260 EECU·s) and ~2% of the whole encoder; the transformer
is the rest. As one interactive request (checksum of
the stage-3 output, canonical size) the two heads are indistinguishable: 462 vs
464 EECU·s, and the same EE profiler peak, **1,061 MiB** (to within 40 bytes;
`Image.reproject computing pixels`), which Milestone 5 traced to the decoded weights;
the real-image request's wall time (863 s vs 214 s) reflects other M6 runs sharing
the project at the time, not its own cost.

**The two-task split still holds (Phase 21).** Embedded constants (EE batch's cap:
24 M pass, 26 M fail):

| graph | embedded values | status |
|---|---:|---|
| task A (Milestone 5 head 14.25 M + patch embedding 17,184) | 14.27 M | ✓ every run |
| task B (PM 3 → stage 4 → norm, unchanged) | 16.65 M | ✓ every run |

The patch embedding adds W^T (3,072 values) plus its bias and LayerNorm γ, β with rows
repeated × 49 (3 × 4,704): 0.12% of task A's budget. No new boundary is needed.
Serialized graphs stay small and constant in the crop size: task A 728 KB (732
function invocations) at 448, 896 and 1792 (Milestone 5's head: 659 KB), of which the
input and patch embedding are 75 KB; task B 360 KB (Milestone 5: 323 KB). No
"Computation is too complex" anywhere.

**Scaling with the crop (Phase 22; the same two tasks, per-block):**

| crop | stage-1 windows | task A EECU·s / run s | task B EECU·s / run s | M5 synthetic task A / task B (same size) |
|---|---:|---|---|---|
| edge_224 | 8 × 8 | 41 / 45 | (interactive) | 82 (per-stage) / 9.7 |
| small_448 | 16 × 16 | 100 / 132 | (interactive) | 83 / — |
| canonical_896 | 32 × 32 | 309 / 146 | 47 (norm), 68 (stage 4) | 305 / 53 |
| large_1792 | 64 × 64 | 1,184 / 929 | 180 (norm), 201 (stage 4) | 930 / 141 |

Cost scales like Milestone 5's: roughly with the number of stage-1 windows (×3.8 per
4× area from 896 to 1792; Milestone 5: ×3.0). The 1792 task A queued 339 s and ran
929 s while other M6 tasks shared the project; at that size task A cost 1.27× and
task B (180 + 201 EECU·s for two separate exports) more than Milestone 5's 141 —
within the service variance seen in Milestone 5 (up to 1.5×) plus concurrency, and
no new limit appeared. **Largest real-image crop processed: 1792 × 1792 pixels
(17.92 km), 448 × 448 stage-1 tokens → 56 × 56 × 768 features**, the same size as
Milestone 5's largest synthetic input.

## 10. Earth Engine behaviour found or confirmed (probed live)

1. **`changeProj` relabels exactly across coordinate systems**: native EPSG:32651
   pixels (row, col) → EPSG:3857 abstract pixels (row, col) and back, for IDs and for
   Sentinel-1 values, at every stage scale (Milestone 1 had shown it within one CRS).
2. **Sentinel-1 GRD dB values are not bit-deterministic between requests**: 2-4.5% of
   the doubles differ by 1-2 ulp (≤ 1.42e-14 dB) between identical requests. A float32
   cast absorbs it here (closest value to a float32 midpoint: 5.7e-14 dB), but a model
   that consumed the doubles would not be reproducible to the bit.
3. **Reading a crop on a native-aligned grid copies pixels exactly**; an integer-metre
   grid (as SegWater used) also copies one to one at the same scale but shifts the
   geolocation by up to half a pixel.
4. **EE's implicit masks are destructive in window layout**: `toArray` masks a whole
   array pixel if any input is masked; through shifted windows and Patch Merging a
   49%-valid crop loses its entire output by stage 2.
5. **`Image.convolve` accumulates in double, in kernel row-major order** (equal to a
   sequential double sum), so a convolution formulation is exact too, but it evaluates
   every input pixel: 13× the EECU·s of the strided gather.
6. **Per-token projection on the fine grid costs memory**: projecting [1, 32] rows on
   the 224 × 224 token grid peaks at 1.45 GB (EE profiler), 10× the window-layout
   projection (137 MB), for the same EECU·s.
7. As in Milestones 1-5: stride sampling relabels exactly; EE caches identical
   graphs (repeats use a new scale or seed); batch caps embedded constants
   (unchanged split, §9).

## 11. Recommendation: proceed to Milestone 7

**Proceed.** On each criterion:

- **Real-image input fidelity**: EE's model input equals the reference's bit for bit
  on every crop (11.4 M values), read on the scene's native grid with no resampling;
  the source's double-precision jitter is measured and absorbed by the float32 cast.
- **Patch embedding**: timm's semantics established experimentally; patch membership
  and ordering exact (3.2 M ID values, two gathers); projection and LayerNorm
  bit-identical to the mirror, ≤ 2.4e-6 from timm.
- **Encoder correctness and numerical error**: unchanged from Milestone 5 on real
  data: every boundary of the canonical crop within 1.6e-5 of timm, the features
  within 5.7e-6; every compared tensor bit-identical to the mirror.
- **Geospatial alignment**: outputs are georeferenced on 40-320 m grids derived from
  the native grid; metadata, footprints and relabels verified.
- **Mask behaviour**: explicit and deterministic (training-mean fill + valid
  fraction); EE's implicit masks are shown to be unusable.
- **Boundary/context behaviour**: timm's cyclic wrap is reproduced exactly; a
  644-px (aligned: 672-px) real-imagery halo makes interior features wrap-free,
  which is how tiled inference should feed a decoder.
- **Two-task pipeline, batch scaling, memory, EECU, graph limits**: the split is
  unchanged (patch embedding adds 17 k embedded values to task A's 14.25 M); task A
  costs the same from real imagery as from synthetic tokens; the 1792 crop runs in
  the same two tasks; graphs stay below 1 MB.

For Milestone 7 (a native segmentation decoder on these features):

1. **Feed the decoder from task B's outputs and task A's intermediate.** The stage-1
   to stage-4 features are georeferenced; a U-Net-style decoder needs the skip
   features of stages 1-3, which task A already computes. Decide early whether the
   decoder lives in task B (reading the stage-3 asset plus re-exported skips) or in a
   task C; the embedded-constant cap (24-26 M) is the binding constraint, not
   compute.
2. **Use the real-context halo for map production.** For tiles of a larger area,
   compute the encoder on the tile plus 672 px (aligned to stage-4 windows) and keep
   the core: then each core's features are independent of the tile edges and of the
   cyclic wrap.
3. **Carry the valid fraction** to the decoder's output to flag predictions whose
   receptive field is mostly fill.
4. **Train or load real weights before interpreting outputs.** M6's features are
   numerically exact but semantically arbitrary (seeded parameters); M7's
   segmentation quality needs a trained Swin-T + decoder checkpoint, mapped into the
   same asset format (the patch embedding's 2-D storage of the conv weight extends to
   any decoder conv).
5. **Keep the mirror and the exact-input check** for every new operation.

## 12. Caveats

- **Seeded weights.** The encoder and the patch embedding use deterministic seeded
  parameters (Milestone 5's and seed 16), not a trained checkpoint: M6 proves
  execution correctness from real imagery, not features that mean anything about
  water. The preprocessing is SegWater v3's, chosen as a fixed documented transform.
- **One scene.** All crops come from one Sentinel-1 scene (ascending, 3 January
  2024, Taiwan). Other scenes are other images of the same collection on other UTM
  grids; nothing in the code is scene-specific except the constants in `scene.py`.
- **Input determinism** rests on a measured margin (§2): EE's doubles jitter, the
  float32 cast absorbs it on these crops.
- **Profiler semantics and variance.** "Peak" is EE's per-algorithm accounting, not
  RAM; EECU·s of identical runs vary up to ~1.5× on the shared service (Milestone 5),
  so single observations are indicative.
- **Interrupted runs, resumed.** The 448 head and tail fetches ran in three attempts
  (a DNS outage, then the intermittent stage-4 memory limit; logs
  `results/raw/validate_small_{head,tail,tail2,tail3}.log`); fetched tensors were
  cached and every tensor was compared once, complete. The 1792 task A finished in EE
  after its polling client was stopped by a local 30-min limit; its record was taken
  from the EE task list (`results/raw/stage3_assets.json`, note field) and task B
  ran from that asset (`validate_large.log`).
- **Validation downloads.** Each crop's raw dB was downloaded once for the reference
  (and twice for the determinism check); tensors were fetched or exported only for
  comparison. Inference reads nothing from outside EE.
- **Assets.** The patch-embedding assets (`swin_image_encoder/m6_pe_weights_*`), the
  task-A stage-3 assets (`results/raw/stage3_assets.json`) and the validation and
  benchmark exports (`swin_image_encoder/exports/`) remain in project
  `ORIGINAL_PROJECT`; delete the exports when finished.
