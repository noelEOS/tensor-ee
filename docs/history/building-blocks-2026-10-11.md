# Building-block inventory

> **Historical record.** This is the pre-refactor inventory of commit `24a6291`, which
> sized Phase 1 of the library plan (development archive: `docs/library-plan.md`). It was `docs/building-blocks.md`
> until L07. The [current block catalogue](../building-blocks.md) supersedes it. Its code
> links are pinned to `24a6291`, where the cited lines are; its text is unchanged.

Snapshot of commit `24a6291` (2026-10-11). This is a read-only review of the source
and of the recorded evidence (milestone notes, `results/`, `docs/evidence/`). Nothing
was re-run for it.

**Purpose.** List every reusable piece of Earth Engine neural-network code in the repo
and record, for each one:
- how it represents tensors;
- what each output pixel depends on;
- where it has been validated;
- which production path uses it.

That is the information needed to size the refactor toward a library of EE building
blocks plus two reference models (the CNN U-Net and Swin-T/UPerNet) built on it.

The raw per-block records are in
[evidence/building-blocks-inventory.json](../evidence/building-blocks-inventory.json):
239 entries, and every `file:line` was checked against the source. This page curates
them. Its **Status** column supersedes the `reader_tier` field in the JSON.

## 1. Summary

### What is there

| Where the block is used | Blocks |
|---|---:|
| CNN path: reachable from `compile_model` or `native_ee.primitives` | 44 |
| Swin-T/UPerNet path: reachable from `SwinT224Adapter` | 120 |
| Research only: alternatives, superseded variants, Code Editor JavaScript copies | 75 |
| **Total** | **239** |

The curated tables in section 3 have 69 rows; some rows group closely related
functions. Each row gets one of four statuses:

| Status | Rows | Meaning |
|---|---:|---|
| Ready | 35 | Takes its shape parameters as arguments; validated live at more than one setting a new model would change |
| Fixed | 22 | Validated live, but only at the two models' settings, or with those settings hard-coded |
| Model | 7 | Architecture composite: belongs in a reference model, built from library blocks |
| Research | 5 | Off every production path; each row groups several variants |

"Ready" only covers the settings that were actually varied: mostly grid size, channel
width and depth. For every window-layout (W) block, window size was never one of them
(section 8). A Ready W block is ready for other grids and widths, not yet for other
window sizes.

**The two production paths share almost nothing.** The Swin adapter reuses only:
- `Grid` and `Raster`;
- the weight-asset format (`ConvolutionWeights`, `FeatureCollectionWeightProvider`);
- the SegWater dB preprocessing.

They share no compute operation (native_ee/swin.py:12 (archive `native_ee/swin.py:12` at `24a6291`),
_swin_parameters.py:15 (archive `native_ee/_swin_parameters.py:15` at `24a6291`)). The repo currently
shows that two separate EE stacks work, not that one library supports two
architectures.

**The numerics are not the problem.** Almost every production block:
- has been validated live against PyTorch float32, typically to 1e-5 or better;
- is bit-identical to its NumPy mirror.

The refactor is structural. It should not need new numerical methods.

### The work, in six parts

1. **Unify representations.** The code has three grid types, two tensor wrappers and
   four tensor layouts (R, D, W, V; section 2). The bridges between layouts
   (`to_windows`/`to_dense`, `_to_model`/`_to_coordinate`) only work for 7×7 windows
   and 224-pixel chips. Pick the first-class layouts, probably R for convolutional
   models and W for windowed attention, and give them general, tested bridges.
2. **Remove duplicates.** About 20 computations exist two to four times (section 4).
   Examples:
   - Conv2d in two layouts with incompatible weight orders;
   - BatchNorm folded in one path and explicit in the other;
   - three entry points for LayerNorm;
   - a JavaScript runtime that re-implements nine Python blocks by hand.
3. **Parameterize constants.** These are hard-coded independently in several places
   (section 5):
   - `WINDOW = 7` in 12 runtime modules;
   - `DECODER = 256` in 4;
   - `MASK_VALUE` 3 times, with two different values.

   EE arrays don't broadcast, so per-channel constants are pre-repeated to 49 rows when
   weights are loaded. The window size is therefore baked in at that point, not passed
   at call time.
4. **Pick one default per block.** Production uses:
   - one of six 3×3 convolution methods;
   - two of six resize paths;
   - one of three pooling paths;
   - one of three fusion formulations.

   The rest are research variants (section 6).
5. **Declare locality and cost.** CNN blocks depend on a bounded neighbourhood and
   already have exact receptive-field and halo analysis. Swin's cyclic shift and the
   UPerNet PSP depend on the whole chip. The limit on embedded constants per graph
   forces staged exports. Each block needs a declared locality class and an
   embedded-constant cost before a converter can plan graphs for it (section 7).
6. **Build a live regression suite.** CI runs offline only. For the Swin/UPerNet
   blocks, no automated test exercises EE numerics; the evidence is the recorded
   milestone runs. No Swin block has been run with window ≠ 7, shift ≠ 3, patch ≠ 4
   or input channels ≠ 2 (section 8).

Packaging needs some cleanup too: several shipped modules lazily import unshipped ones
(section 9).

### Suggested order

1. Settle the layouts and their bridges.
2. Consolidate each duplicate onto the variant production already uses.
3. Parameterize window, shift, patch and channels.
4. Add small live fixtures per block.
5. Rebuild both reference models on the public blocks only. The proof is that the
   existing gates still pass unchanged:
   - Swin: at most 1.633e-5 from ONNX on the 32 M12 chips, with zero changed
     predictions;
   - CNN: the ResNet18 U-Net and SegWater parity records.
6. Only then extend `compile_model` to the blocks that have reached Ready.

## 2. How to read the tables

**Representation**

| Code | Meaning |
|---|---|
| R | Raster: an `ee.Image` with one band per channel (`c0…cN`) on a georeferenced `Grid` (compiler.py:83 (archive `native_ee/compiler.py:83` at `24a6291`)). The CNN path's layout. |
| D | Dense token grid: one band per channel on an abstract `TokenGrid` with origin (0, 0) in EPSG:3857 (spatial_attention.py:48 (archive `ee_attention/earth_engine/spatial_attention.py:48` at `24a6291`)). |
| W | Window layout: one `[49, C]` float32 array per pixel of the 7× coarser window grid. Token `t = 7a + b` is pixel `(7I + a, 7J + b)` (layout.py:1 (archive `upernet/earth_engine/layout.py:1` at `24a6291`)). Every Swin/UPerNet compute block uses it. |
| V | An `ee.Array` value rather than an image: weights, and small maps such as the PSP's pooled `[s², C]`. |

**Locality** (what one output pixel depends on):
- **pointwise**: only the same pixel;
- **k×k**: a bounded neighbourhood;
- **window**: its own 7×7 window;
- **window+ring**: its window plus a one-token ring from the 8 neighbouring windows;
- **extent**: the whole finite chip or tile, as with global pooling or cyclic shift.

**Status**

| Status | Meaning |
|---|---|
| Ready | Takes its shape parameters as arguments and has been validated live at more than one setting a new model would change |
| Fixed | Validated live, but only at the settings the two models use, or with those settings hard-coded. The limiting setting is named after the dash. |
| Model | Architecture composite, such as the UPerNet FPN or a Swin stage. It belongs in a reference model, built from library blocks. |
| Research | Not on any production path |

**Path**: CNN, Swin, Both, or — (none). "Validated" means recorded live EE runs unless
it says *offline*. Errors are the maximum |EE − PyTorch float32|.

## 3. Inventory by function

### 3.1 Grids, tensors and the ops abstraction

| Block | Code | Repr | Validated | Status | Path |
|---|---|---|---|---|---|
| `Grid`: georeferenced north-up grid | compiler.py:42 (archive `native_ee/compiler.py:42` at `24a6291`) | — | Several CRSs, 10–30 m, 16×12 to 608² | Ready | Both |
| `Raster`: tensor value type | compiler.py:83 (archive `native_ee/compiler.py:83` at `24a6291`) | R | Every CNN run; Swin adapter input/output | Ready | Both |
| `TokenGrid`: abstract token/window grid | spatial_attention.py:48 (archive `ee_attention/earth_engine/spatial_attention.py:48` at `24a6291`) | — | 7² to 448² tokens | Fixed — origin (0,0), default scale 10 | Swin |
| `FeatureGrid`: georeferenced feature grid | feature_interface.py:47 (archive `upernet/earth_engine/feature_interface.py:47` at `24a6291`) | — | Every decoder milestone | Ready | Swin |
| `Feature`: tensor + grid + layout; refuses operands on different grids | feature_ops.py:25 (archive `upernet/earth_engine/feature_ops.py:25` at `24a6291`) | W, D | Negative controls refused | Ready | Swin |
| `ServerArray`, `constant_value`, `encode_float32`: float32 weights as EE arrays | ops.py:33 (archive `swin_block/earth_engine/ops.py:33` at `24a6291`) | V | 27.5 M asset weights; 112,347 embedded values read back bit-exact | Ready | Swin |
| `ArrayOps`/`ImageOps`, `BlockArrayOps`/`BlockImageOps`: the same maths on V or W | array_ops.py:21 (archive `ee_attention/earth_engine/array_ops.py:21` at `24a6291`), ops.py:88 (archive `swin_block/earth_engine/ops.py:88` at `24a6291`) | V, W | N = 49, d = 32–768 | Ready | Swin |

The ops abstraction is only half polymorphic. Compute blocks take `ops`, but every
layout transform takes `ee` directly and only works on images.

### 3.2 Layout transforms and bridges

| Block | Code | Repr | Locality | Validated | Status | Path |
|---|---|---|---|---|---|---|
| `from_image`: select bands, zero-fill, place on grid | primitives.py:20 (archive `native_ee/primitives.py:20` at `24a6291`) | → R | pointwise | *offline*: graph identical to `compile_model`'s input | Ready | CNN |
| `_finish`, `_zero_extended`: pin each layer to its grid; zero padding | compiler.py:116 (archive `native_ee/compiler.py:116` at `24a6291`) | R | pointwise | Every CNN run, border pixels included | Ready | CNN |
| `_stride_sample`: stride-s decimation | compiler.py:129 (archive `native_ee/compiler.py:129` at `24a6291`) | R | pointwise | Stride 2 only, 16² and 15×17 | Fixed — stride 2 | CNN |
| `_to_model` / `_to_coordinate`: relabel caller grid ↔ abstract model grid | swin.py:73 (archive `native_ee/swin.py:73` at `24a6291`) | R ↔ D | pointwise | 224² only (T01, T06) | Fixed — 224, 2.5 m | Swin |
| `to_windows` (D → W) | layout.py:107 (archive `upernet/earth_engine/layout.py:107` at `24a6291`) | D → W | window | C1–C4 round trip exact | Fixed — window 7 | Swin |
| `depth_to_space` / `to_dense` (W → D) | patch_merge.py:150 (archive `swin_hierarchy/earth_engine/patch_merge.py:150` at `24a6291`), layout.py:101 (archive `upernet/earth_engine/layout.py:101` at `24a6291`) | W → D | window | C = 96–768; 8²–64² windows | Fixed — window 7 | Swin |
| `stride_samples`, `stack_tokens`, `as_row`, `flatten_row`: space-to-depth into per-pixel arrays | spatial_attention.py:255 (archive `ee_attention/earth_engine/spatial_attention.py:255` at `24a6291`) | D ↔ V rows | window | `stride_samples` at factors 2 (patch merge), 4 (patch embed), 7 (`to_windows`) and 8 (Milestone 1); `stack_tokens` at 49 tokens in production, 64 in Milestone 1 | Ready | Swin |
| `stride_sample` (georeferenced) | feature_transport.py:5 (archive `upernet/earth_engine/feature_transport.py:5` at `24a6291`) | D | pointwise | Factors 2 and 4 | Ready | Swin |
| `halo`, `neighbours`, `zero_outside`: window plus zero-padded one-token ring | layout.py:61 (archive `upernet/earth_engine/layout.py:61` at `24a6291`) | W | window+ring | ID fixtures exact at every edge of C1–C3 | Fixed — window 7, ring 1 | Swin |
| `im2col`, `taps` | layout.py:89 (archive `upernet/earth_engine/layout.py:89` at `24a6291`) | W | window | IDs exact at 1,024 and 1,792 channels | Fixed — kernel 3 | Swin |
| `translate`, `materialize`: exact integer shift, pin to grid | shift.py:66 (archive `swin_pair/earth_engine/shift.py:66` at `24a6291`) | W | shift | 2×2 to 32×32 window grids | Ready | Swin |
| `periodic_extension`: wrap ring of windows | periodic_extension.py:92 (archive `swin_hierarchy/earth_engine/periodic_extension.py:92` at `24a6291`) | W | extent | Halo 1 and 3; 2×2 to 64×64 windows. Works for any window size by construction, but only run at window 7 | Ready | Swin |
| `domain_mask` | periodic_extension.py:113 (archive `swin_hierarchy/earth_engine/periodic_extension.py:113` at `24a6291`) | — | pointwise | Every masked run | Ready | Swin |
| `embedded_raster`: NumPy fixture → image | feature_interface.py:165 (archive `upernet/earth_engine/feature_interface.py:165` at `24a6291`) | → D | pointwise | Every M7–M12 fixture | Ready | Swin |

### 3.3 Convolution

| Block | Code | Repr | Locality | Validated | Status | Path |
|---|---|---|---|---|---|---|
| `conv2d`: odd kernel, same padding, stride 1 or 2 | compiler.py:164 (archive `native_ee/compiler.py:164` at `24a6291`) | R | k×k | Kernels 1, 3, 7; strides 1, 2; up to 768 → 256; ResNet18 U-Net 2.55e-7; trained SegWater | Ready | CNN |
| `conv1x1` (= `linear`) | conv2d.py:52 (archive `upernet/earth_engine/conv2d.py:52` at `24a6291`) | W, V | pointwise | 96–768 → 256; 256 → 2 + bias; ≤ 4.5e-6 | Ready | Swin |
| `conv3x3_window` (`window_im2col`) | conv2d.py:92 (archive `upernet/earth_engine/conv2d.py:92` at `24a6291`) | W | 3×3 | 2 → 3, 256 → 256, 1792 → 256 on 7² to 448²; ≤ 2.3e-5 | Fixed — kernel 3, stride 1 | Swin |
| `group_product` / `grouped_double`: 3×3 conv split into input-channel groups, double-precision partials | fusion.py:160 (archive `upernet/earth_engine/fusion.py:160` at `24a6291`) | W | 3×3 | 1024 → 256 at C1 56² and 224² | Fixed — 4 groups | Swin |
| `patch_embed`: Conv2d with kernel = stride = 4, as im2col + LayerNorm | patch_embed.py:113 (archive `swin_image_encoder/earth_engine/patch_embed.py:113` at `24a6291`) | R → W | 4×4 | Crops 224–1792; ≤ 1.7e-6 | Fixed — patch 4, bands VV/VH, 96 channels | Swin |
| `window_taps`, `conv3x3_dense`, `conv_convolve` | conv2d.py:129 (archive `upernet/earth_engine/conv2d.py:129` at `24a6291`) | W, D | 3×3 | Comparison studies only | Research | — |

### 3.4 Normalization

| Block | Code | Repr | Locality | Validated | Status | Path |
|---|---|---|---|---|---|---|
| BatchNorm folded into the preceding conv | semantics.py:164 (archive `native_ee/semantics.py:164` at `24a6291`) | R | pointwise | 30 folds each in ResNet18 U-Net and trained SegWater | Ready | CNN |
| `batch_norm` (kept explicit) | normalization.py:63 (archive `upernet/earth_engine/normalization.py:63` at `24a6291`) | W, V | pointwise | 256 channels; rows 1–49 | Ready | Swin |
| `layer_norm` (over channels) | layer_norm.py:23 (archive `swin_block/earth_engine/layer_norm.py:23` at `24a6291`) | W, V | per token | Dims 96–1536 | Ready | Swin |
| `layer_norm2d` | normalization.py:37 (archive `upernet/earth_engine/normalization.py:37` at `24a6291`) | W | per token | Validators only | Research | — |

### 3.5 Activations, merges and outputs

| Block | Code | Repr | Validated | Status | Path |
|---|---|---|---|---|---|
| ReLU, Sigmoid, LeakyReLU, SiLU, exact GELU, Tanh, eval Dropout, channel Softmax | compiler.py:268 (archive `native_ee/compiler.py:268` at `24a6291`)–501 (archive `native_ee/compiler.py:501` at `24a6291`) | R | Value sweeps including ±1e30 and masks; the 3-class nonwater CNN | Ready | CNN |
| `gelu` | linear.py:45 (archive `swin_block/earth_engine/linear.py:45` at `24a6291`) | W, V | Widths 384–3072; 4.8e-7 | Ready | Swin |
| `relu` | normalization.py:68 (archive `upernet/earth_engine/normalization.py:68` at `24a6291`) | W, V | ±0, subnormals | Ready | Swin |
| `concatenate`, `residual_add` | compiler.py:244 (archive `native_ee/compiler.py:244` at `24a6291`) | R | Every U-Net skip and residual | Ready | CNN |
| `concat`, `add` | feature_ops.py:59 (archive `upernet/earth_engine/feature_ops.py:59` at `24a6291`) | W, D | Up to 51 M IDs exact | Ready | Swin |
| `segmentation_outputs`, `binary_threshold` | outputs.py:113 (archive `native_ee/outputs.py:113` at `24a6291`) | R | 65 cases, 2 and 3 classes, masks | Ready | CNN |

### 3.6 Linear layers and attention

| Block | Code | Repr | Locality | Validated | Status | Path |
|---|---|---|---|---|---|---|
| `linear` + `LinearConstants` | linear.py:39 (archive `swin_block/earth_engine/linear.py:39` at `24a6291`) | W, V | per token | 96 → 288 up to 768 → 2304; ≤ 1.05e-5 | Ready | Swin |
| `naive_attention`: softmax(q·kᵀ·scale + bias + mask)·v | naive_attention.py:29 (archive `ee_attention/earth_engine/naive_attention.py:29` at `24a6291`) | V per window | window | N = 49 in production (head dim 32, heads 3–24); head dim 64 and N = 25 and 64 through the Milestone 1 wrapper `attend` | Ready | Swin |
| `split_heads` | swin_block.py:104 (archive `swin_block/earth_engine/swin_block.py:104` at `24a6291`) | W | per token | Dims 96–768, heads 3–24 | Ready | Swin |
| `relative_position_index` / `relative_position_bias` | relative_position.py:25 (archive `swin_block/earth_engine/relative_position.py:25` at `24a6291`) | V | — | Window 7 only; heads 3–24 | Fixed — window 7 | Swin |
| `window_attention`: W-MSA/SW-MSA up to the output projection | swin_block.py:116 (archive `swin_block/earth_engine/swin_block.py:116` at `24a6291`) | W | window | 1 to 1,024 windows; dims 96–768 | Fixed — 49 tokens | Swin |
| `shifted_window_mask` | shifted_mask.py:46 (archive `swin_pair/earth_engine/shifted_mask.py:46` at `24a6291`) | V | window; depends on chip edge | 2×2 to 32×32 windows | Fixed — window 7, shift 3 | Swin |
| `shift_windows`, `unshift_windows`, `roll_windows`: cyclic shift in W | shift.py:151 (archive `swin_pair/earth_engine/shift.py:151` at `24a6291`) | W | window+ring, cyclic | 2×2 to 5×2 grids; C = 3–768 | Fixed — window 7, shift 3 | Swin |
| `blockwise_attention` (online softmax), `neighborhood_attention`, `window_partition_attention`, `partition_slice_attention`, `attend` | blockwise_attention.py:53 (archive `ee_attention/earth_engine/blockwise_attention.py:53` at `24a6291`), spatial_attention.py:216 (archive `ee_attention/earth_engine/spatial_attention.py:216` at `24a6291`) | D, V | k×k, window | Milestone 1 study. These are wrappers and alternatives; the attention core they call in their naive mode is `naive_attention` above | Research | — |

### 3.7 Resampling

| Block | Code | Repr | Locality | Validated | Status | Path |
|---|---|---|---|---|---|---|
| `nearest_upsample` | compiler.py:234 (archive `native_ee/compiler.py:234` at `24a6291`) | R | pointwise | Factor 2 only | Fixed — factor 2 | CNN |
| `window_resize`: bilinear ×f, align_corners False or True | resize.py:136 (archive `upernet/earth_engine/resize.py:136` at `24a6291`) | W | window+ring | Factors 2/4/8, both modes, 7 to 1792 px; fixtures ≤ 9.5e-7 | Ready | Swin |
| `window_matrix_resize`: bilinear from a small V map onto W | resize.py:293 (archive `upernet/earth_engine/resize.py:293` at `24a6291`) | V → W | extent | PSP bins 1/2/3/6 → 7² to 56² | Ready | Swin |
| `bilinear_weights`, `local_matrices`: PyTorch-exact coefficients | _coefficients.py:78 (archive `upernet/earth_engine/_coefficients.py:78` at `24a6291`), resize.py:83 (archive `upernet/earth_engine/resize.py:83` at `24a6291`) | — | — | *offline* against PyTorch, ≥ 7 shape pairs | Ready | Swin |
| `patch_merge`: 2×2 space-to-depth + LayerNorm + Linear | patch_merge.py:128 (archive `swin_hierarchy/earth_engine/patch_merge.py:128` at `24a6291`) | W | 2×2 windows | C = 96–384; 4×6 to 32×32 windows; ≤ 7.6e-6 | Fixed — window 7 | Swin |
| `dense_resize`, `dense_absolute_resize`, `matrix_resize`, `native_resize` | resize.py:182 (archive `upernet/earth_engine/resize.py:182` at `24a6291`) | D, V | 3×3, extent | Comparisons and negative controls | Research | — |

### 3.8 Pooling

| Block | Code | Repr | Locality | Validated | Status | Path |
|---|---|---|---|---|---|---|
| Average/max pool 2×2, stride 2 | compiler.py:220 (archive `native_ee/compiler.py:220` at `24a6291`) | R | 2×2 | 16², TinyUNet, nonwater CNN | Fixed — 2×2, even sizes | CNN |
| Max pool 3×3, stride 2, padding 1 (ResNet stem) | compiler.py:214 (archive `native_ee/compiler.py:214` at `24a6291`) | R | 3×3 | 16², 15×17, ResNet18, SegWater | Fixed — one configuration | CNN |
| `window_pool` + `pool_rows`: AdaptiveAvgPool2d, all bins at once | adaptive_pool.py:131 (archive `upernet/earth_engine/adaptive_pool.py:131` at `24a6291`), _coefficients.py:34 (archive `upernet/earth_engine/_coefficients.py:34` at `24a6291`) | W → V | extent | Grids 7² to 56²; ≤ 3.6e-6 | Fixed — bins (1,2,3,6) | Swin |
| `global_pool`, `dense_pool` | adaptive_pool.py:141 (archive `upernet/earth_engine/adaptive_pool.py:141` at `24a6291`) | W, D | extent | Comparisons | Research | — |

### 3.9 Composite blocks (the model layer)

| Block | Code | Repr | Locality | Validated | Status | Path |
|---|---|---|---|---|---|---|
| `compile_model`: traced PyTorch → EE | compiler.py:540 (archive `native_ee/compiler.py:540` at `24a6291`) | R | model's receptive field | SmallCNN, TinyUNet, ResNet18 U-Net, trained SegWater, nonwater CNN | Ready | CNN |
| `compile_tiled_model`, `plan_tiles`, `analyze_receptive_field` | tiling.py:211 (archive `native_ee/tiling.py:211` at `24a6291`), receptive_field.py:193 (archive `native_ee/receptive_field.py:193` at `24a6291`) | R | halo ≥ receptive field | TinyUNet; ResNet18 2×2 cores, 3.05e-7 | Ready | CNN |
| `swin_block`, `shifted_swin_block`, `swin_stage`, `encoder` | swin_block.py:150 (archive `swin_block/earth_engine/swin_block.py:150` at `24a6291`), stage.py:45 (archive `swin_encoder/earth_engine/stage.py:45` at `24a6291`), swin_encoder.py:155 (archive `swin_encoder/earth_engine/swin_encoder.py:155` at `24a6291`) | W | extent | 1 to 1,024 windows; dims 96–768; crops 224–1792 | Model | Swin |
| PSP `branch`, `forward` | psp.py:91 (archive `upernet/earth_engine/psp.py:91` at `24a6291`) | W, V | extent | C4 7² to 56²; ≤ 1.8e-5 | Model | Swin |
| FPN `lateral`, `merge`, `conv_block`, `forward` | fpn.py:101 (archive `upernet/earth_engine/fpn.py:101` at `24a6291`) | W | window+ring | C4 14² to 56²; ≤ 2.7e-5 | Model | Swin |
| Fusion `resize_level`, `forward` | fusion.py:130 (archive `upernet/earth_engine/fusion.py:130` at `24a6291`) | W | window+ring | C1 56² and 224² | Model | Swin |
| Segmentation head `head_conv`, `upsample`, `forward` | segmentation_head.py:74 (archive `upernet/earth_engine/segmentation_head.py:74` at `24a6291`) | W | window+ring | Crops 896, 1792; M12 32 chips ≤ 1.633e-5 | Model | Swin |
| `SwinT224Adapter.skips` / `.c4` / `.logits` | swin.py:132 (archive `native_ee/swin.py:132` at `24a6291`) | R | extent | 224² chips: T01 ≤ 1.46e-5, T06 5.96e-6, no changed predictions | Model | Swin |

The UPerNet composites work on any grid, but the architecture is fixed. They look up
weights by SMP `state_dict` names and take channel widths from module constants
(psp.py:43 (archive `upernet/earth_engine/psp.py:43` at `24a6291`),
_decoder_contract.py:5 (archive `upernet/earth_engine/_decoder_contract.py:5` at `24a6291`)).

### 3.10 Weights, preprocessing and model I/O

| Block | Code | Validated | Status | Path |
|---|---|---|---|---|
| `ConvolutionWeights`, `feature_records`, `FeatureCollectionWeightProvider`: weight tables as EE assets | weight_assets.py:261 (archive `native_ee/weight_assets.py:261` at `24a6291`) | ResNet18, SegWater, Swin M12 | Ready | Both |
| `MatchedWeightProvider`: match traced layers to the manifest by fingerprint | weight_mapping.py:7 (archive `native_ee/weight_mapping.py:7` at `24a6291`) | Trained SegWater | Ready | CNN |
| `load_swin_t224_parameters` and helpers: checkpoint → validated `SwinTParameters` | _swin_parameters.py:245 (archive `native_ee/_swin_parameters.py:245` at `24a6291`) | T01 | Model | Swin |
| `load_onnx_model`: ONNX → the converter's subset | onnx_import.py:93 (archive `native_ee/onnx_import.py:93` at `24a6291`) | SegWater ONNX chips | Fixed — opset 17, input [N,2,224,224] | CNN |
| `preprocess_db_ee`, `quantize_db_ee`, `round_ties_to_even_ee`: SegWater Sentinel-1 dB recipe | segwater_preprocessing.py:133 (archive `native_ee/segwater_preprocessing.py:133` at `24a6291`) | Zero error against NumPy | Fixed — SegWater constants | Both |
| `plan_segwater_tiles`, `_valid_logits`, `_water_probability` | segwater_inference.py:87 (archive `native_ee/segwater_inference.py:87` at `24a6291`) | SegWater mosaics | Fixed — copies of generic blocks | CNN |

### 3.11 Code Editor runtime (JavaScript)

`code_editor/segwater.js` (archive `code_editor/segwater.js` at `24a6291`) re-implements by hand:
- grid helpers;
- stride sampling;
- the weight reader;
- convolution, ReLU, 3×3 max pool, nearest upsampling, add and concat;
- SegWater preprocessing and water probability.

The model descriptor is generated from Python, but the operation bodies are not. Live
parity runs show 0.0 difference from Python EE on SegWater
(`results/segwater_js_parity_*.json`), and those runs are the only numerical link
between the two. The JavaScript covers only SegWater's operations and accepts inputs
whose sides are multiples of 224, up to 4 chips. It is not part of the installed
package.

## 4. Duplicates to unify

| Computation | Copies | Production uses | Note |
|---|---|---|---|
| Grid type | `compiler.Grid`, `TokenGrid`, `FeatureGrid`, plus about 7 grid factories | All three | `TokenGrid` assumes origin (0,0), so the adapter relabels in and out (swin.py:73 (archive `native_ee/swin.py:73` at `24a6291`)) |
| Tensor wrapper | `Raster`, `Feature` | Both | |
| Conv2d | `compiler.conv2d` (R); `conv3x3_window` (W); `fusion.group_product` (inline copy); `conv1x1` = `linear`; JavaScript `convolution` | All but the JS | Weight orders are not interchangeable. The compiler flattens channel-outermost `(c, ky, kx)` (compiler.py:179 (archive `native_ee/compiler.py:179` at `24a6291`)); the window path uses `(ky, kx, c)`. The compiler casts matrix operands to float32; the decoder accumulates in double and rounds once. release-contract.md:146 (development archive: `docs/release-contract.md`) requires keeping those two policies distinct. |
| BatchNorm | Folded (semantics.py:196 (archive `native_ee/semantics.py:196` at `24a6291`)); explicit (normalization.py:63 (archive `upernet/earth_engine/normalization.py:63` at `24a6291`)); `normalization.fold`; descriptor folding in `experiments/export_segwater_js.py` | Folded on CNN, explicit on Swin | Milestone 7 kept decoder BN explicit to stay bit-exact with its mirror |
| LayerNorm2d entry points | `fpn.layer_norm`, `psp.layer_norm`, `normalization.layer_norm2d` (unused), all wrapping `swin_block` `layer_norm` | The first two | |
| ReLU | compiler.py:268 (archive `native_ee/compiler.py:268` at `24a6291`), normalization.py:68 (archive `upernet/earth_engine/normalization.py:68` at `24a6291`), `feature_ops.relu` (unreferenced), inline in `psp.py` | Two | |
| GELU | compiler.py:469 (archive `native_ee/compiler.py:469` at `24a6291`), linear.py:45 (archive `swin_block/earth_engine/linear.py:45` at `24a6291`) | Both | Same formula |
| Stride sampling | compiler.py:129 (archive `native_ee/compiler.py:129` at `24a6291`), feature_transport.py:5 (archive `upernet/earth_engine/feature_transport.py:5` at `24a6291`), spatial_attention.py:255 (archive `ee_attention/earth_engine/spatial_attention.py:255` at `24a6291`) | All three | |
| D → W | `layout.to_windows` (production), `patch_embed.to_windows`, `swin_encoder.stage3_from_asset`, `image_encoder.stage3_from_asset` | First two | |
| W → D (`depth_to_space`) | patch_merge.py:150 (archive `swin_hierarchy/earth_engine/patch_merge.py:150` at `24a6291`) (production), spatial_attention.py:374 (archive `ee_attention/earth_engine/spatial_attention.py:374` at `24a6291`), plus 3 wrappers | First | The `spatial_attention` copy returns the wrong window index on the ring above and left of the raster: EE's `mod` keeps the sign of the dividend (patch_merge.py:151 (archive `swin_hierarchy/earth_engine/patch_merge.py:151` at `24a6291`)) |
| Integer translate | shift.py:66 (archive `swin_pair/earth_engine/shift.py:66` at `24a6291`), feature_interface.py:160 (archive `upernet/earth_engine/feature_interface.py:160` at `24a6291`), resize.py:169 (archive `upernet/earth_engine/resize.py:169` at `24a6291`) | All three | |
| Projection builders, domain mask, `embedded_raster`, `valid_fraction` | 3, 2, 2 and 2 copies | Mixed | |
| `bilinear_weights` | _coefficients.py:78 (archive `upernet/earth_engine/_coefficients.py:78` at `24a6291`), resize.py:67 (archive `upernet/earth_engine/resize.py:67` at `24a6291`) | First | |
| PSP bins | `_coefficients.SIZES`, `topology.PSP_SIZES` | First | |
| Weight readers | `_swin_parameters.records_and_locations` re-implements `swin_hierarchy` `weights.asset_records`; `server_parameters` exists in 4 research modules | `_swin_parameters` | |
| Tilers | `plan_tiles`, `plan_segwater_tiles` (and the JS chip loop) | Both | |
| Validity and probability | `_common_score_mask` ↔ `_valid_logits`; `_water_probability` repeats the sigmoid formula; `FLOAT32_MAX` defined 3 times | Both | |
| Traced-graph validation | `semantics.semantic_call`; `export_segwater_js.build_model_descriptor` re-implements its conv, nearest, BN and max-pool checks | First | |
| Research composites | `swin_block_pair`, `hierarchical_stage`, `isolated_stage`, `image_head`/`image_tail`, plus one NumPy mirror per milestone | None | Archive with the milestone notes |

## 5. Hard-coded constants

| Constant | Definition sites |
|---|---|
| `WINDOW = 7` | Defined independently in 12 runtime modules: `swin_encoder/earth_engine/{configuration:2, rasters:22, swin_encoder:57}`, `swin_hierarchy/earth_engine/{hierarchical_stage:47, patch_merge:57, rasters:17}`, `swin_image_encoder/earth_engine/patch_embed:43`, `swin_pair/earth_engine/{coordinates:4, rasters:17, shift:58}`, `upernet/earth_engine/{_coefficients:6, layout:32}`. Aliased in `conv2d`, `resize` and `adaptive_pool`. Imported "upward" by `swin_block.py:47` from the later `swin_encoder` package. Literal `7` in swin.py:153 (archive `native_ee/swin.py:153` at `24a6291`) and `175`; default argument in `shifted_mask.py:46`. |
| 49 tokens | `TOKENS` in 7 modules. More importantly, per-channel constants (bias, LayerNorm, BatchNorm) are pre-repeated to 49 rows when built (`LinearConstants`, `BlockConstants`, `PatchMergeConstants`, `PatchEmbedConstants`, `Conv3x3Constants`), because EE arrays don't broadcast. |
| Shift 3 | `configuration.py:3`, `shifted_swin_block.py:36`, defaults in `shift.py:164,170` and `shifted_mask.py:46` |
| `MASK_VALUE` | `-1e30` at `ee_attention/earth_engine/common.py:32` and `spatial_attention.py:39`; `-100.0` at `shifted_mask.py:29` (timm's value, the one production uses) |
| Chip size 224 | Literals in `native_ee/swin.py` (with 2.5 m, 56, 7 and 768); `CHIP_SIZE` in `segwater_inference.py:35`; input `[N,2,224,224]` in `onnx_import.py:127` |
| Decoder width 256 | `_decoder_contract.py:2`, `psp.py:42`, `segmentation_head.py:56`, `topology.py:43` |
| Stage widths (96, 192, 384, 768) | `feature_interface.py:41` (tuple), `_decoder_contract.py:4` (dict), `STAGE_DIMS` in `configuration.py`; `C4_CHANNELS = 768` in `psp.py:41` |
| Swin-T geometry | `DEPTHS (2,2,6,2)`, heads, `STAGES = 4` in `configuration.py`; `PATCH = 4`, `EMBED_DIM = 96`, `BANDS = ('VV','VH')` in `patch_embed.py:41-45`. The sensor band names are hard-coded in a shipped block, and the adapter renames caller bands to match. |
| Head | `CLASSES = 2`, `FACTOR = 4` in `segmentation_head.py:55-57` |
| Epsilons | `LN_EPS = 1e-5` (`layer_norm.py:20`), `BN_EPS = 1e-5`, `FEATURE_EPS = 1e-6` (`normalization.py:33-34`) |
| `TokenGrid` defaults | Scale 10 and origin (0,0). The adapter's 2.5 m input scale only works because of this coupling, and the adapter does not assert it. |
| SegWater recipe | `MEANS`, `STDS`, `OFFSET_DB`, `STEP_DB` in `segwater_preprocessing.py:28-34`, repeated as rounded decimals in `segwater.js` |

## 6. Production choices and research variants

| Choice | Production uses | Also in the tree |
|---|---|---|
| 3×3 convolution | `window_im2col` (default argument) | `window_taps` (unreachable: needs per-tap constants that production never builds), `dense_nbr`, `dense_bands`, `dense_taps`, `convolve` |
| Fusion convolution | `grouped_double` (swin.py:189 (archive `native_ee/swin.py:189` at `24a6291`)) | `monolithic`, `grouped_rounded` |
| Bilinear resize | `window_resize` for FPN ×2, fusion ×2/4/8 and the head ×4 (align_corners True); `window_matrix_resize` for PSP | `dense_resize`, `dense_absolute_resize`, `matrix_resize`, `native_resize` (negative control) |
| Adaptive pooling | `window_pool` | `global_pool`, `dense_pool` |
| PSP semantics | Pools only the tile's own C4 | Option C, `psp_semantics.tile_forward`: matches whole-area pooling exactly on three tiles of a 1792 area; not shipped |
| Attention | `naive_attention` via `window_attention` | `blockwise_attention`, neighbourhood, window-partition and partition-slice forms |
| Shift wrap / extension | Shift `periodic` on a ring built by `mosaic` extension, halo 1, `per_block` | `where`; `per_stage` and `propagated` extension modes |
| Patch embedding | `gather='stride'`, `placement='window'` | `neighborhood` gather, `token` placement, `Image.convolve` (benchmark) |
| Pinning | `pin='input'`, `pin_block_input=True`, `patch_merge` `pin=True` | `each`, `none`, stage/block output pins |
| Weights | Asset `ServerArray` or embedded NumPy (both used) | — |

Most production choices are reached through default arguments rather than explicit
call-site values. A library should make them explicit.

## 7. Locality and execution model

**CNN path.** Every block depends on a bounded neighbourhood.
`receptive_field.py` (archive `native_ee/receptive_field.py` at `24a6291`) computes the exact receptive
field and halo, and `tiling.py` (archive `native_ee/tiling.py` at `24a6291`) tiles arbitrary areas with
results equal to full-extent inference. This is what lets `compile_model` scale to any
area.

**Swin/UPerNet path.** Several pieces depend on the whole chip:

| Piece | Why | Measured effect |
|---|---|---|
| Cyclic shift + periodic extension | `torch.roll` semantics wrap within the chip, not across geography | Encoder output for a 448 crop vs the same area inside a 1792 crop: mean absolute difference 0.111 (numerical_summary_m6.md:263 (development archive: `docs/history/swin_image_encoder/results/numerical_summary_m6.md`)) |
| PSP adaptive pooling | Pools the tile's own C4 | 448 tile vs 1792 area: relative L2 0.29–0.32 ([milestone8_findings.md:150](upernet/notes/milestone8_findings.md)) |
| `window_pool` gather, `window_matrix_resize` | Pull values off the server through `sampleRectangle` | Fails on a masked window |

That is the model's own semantics, not a bug. A Swin model is defined per chip, so
tiling a large raster is not the same as running timm on it. A library has to label
such blocks so that nobody tiles them as if they were local.

**Graph limits force stage boundaries:**
- A batch graph can embed at most about 24–26 M constant values ("Encoded object too
  large"). The encoder embeds 30.9 M, so it is split after stage 3
  (swin_encoder.py:30 (archive `swin_encoder/earth_engine/swin_encoder.py:30` at `24a6291`)).
- The 10 MiB request limit forces weights to be read from assets.
- The release contract requires three separate graphs, exported in order A → B → C
  (release-contract.md:165 (development archive: `docs/release-contract.md`)).

For a converter to plan these cuts, each block needs two declared properties:
1. Its locality class. The CNN side already has this through `receptive_field`; the
   W-layout blocks have nothing equivalent.
2. Its embedded-constant cost.

## 8. Test coverage

- **CI runs offline only.** `tests.yml` (archive `.github/workflows/tests.yml` at `24a6291`) sets
  `EE_LIVE=0`, and the 11 live test files skip unless `EE_LIVE=1`.
- **CNN path:** offline tests check graph construction and semantics. The numerical
  evidence is in recorded live runs (`results/*.json`, `docs/evidence/t02`, `t03`,
  `t05`).
- **Swin/UPerNet path:** offline tests exercise NumPy/torch mirrors (`ee_mirror_*`) and
  coefficient helpers. `test_swin_adapter.py` only builds the three graphs. No offline
  test runs the EE attention, block or decoder code numerically, which is unavoidable
  because that needs the EE server. The numerical evidence is the milestone runs
  (M1–M12, T01, T06).
- **Untested settings:**
  - No Swin block has been run, live or offline, with window ≠ 7, shift ≠ 3, patch ≠ 4,
    input channels ≠ 2 or embedding ≠ 96. The exception is the Milestone 1 attention
    forms, which also ran windows 5 and 8.
  - On the CNN side, only nearest factor 2, stride 2 and the two pooling
    configurations have been evaluated.
- **What a library needs:** small, cheap live fixtures per block, run on demand as a
  regression suite. The milestone validators (`*/earth_engine/validate*.py`) are the
  raw material for this.

## 9. Packaging and dead code

- **Shipped modules lazily import unshipped ones.** Calling these functions from the
  installed wheel would raise `ImportError`; none of them is on a production path:
  - `psp`/`fpn`/`segmentation_head.server_parameters` → `decoder_weights`;
  - `fusion.head_budget` → `upernet.reference`;
  - `fusion.task_c_budget` → `topology`;
  - `feature_interface.crop_grid`/`input_grid` → `swin_image_encoder.scene`;
  - several `spatial_attention` functions → `ee_attention.earth_engine.common` and
    `ee_attention.reference`;
  - `swin_encoder.stage_tokens` → `swin_encoder.reference`.
- **Shipped but never called by production:**
  - `blockwise_attention.py`;
  - the `spatial_attention` research composites;
  - `encoder_head`/`encoder_tail`, `isolated_stage` and the extra extension modes in
    `swin_encoder.py`;
  - `patch_merge.fine_grid_gather`;
  - the neighbourhood gather and token placement in `patch_embed`;
  - the `mosaic`/`where` branches of `cyclic_neighbor`.
- **Dead code:**
  - `compiler._interpolate` has no callers;
  - the `window_taps` path cannot be reached;
  - `feature_ops.relu` is never referenced;
  - the 1024-channel concat built in `fusion.forward` is only a guard and is never
    evaluated under `grouped_double`.
- **Layering inversion:** `swin_block.py` imports `WINDOW` from the later
  `swin_encoder` package.
