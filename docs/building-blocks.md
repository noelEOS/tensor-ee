# Block catalogue

The `tensor_ee` library is a set of Earth Engine building blocks. This page lists
them with their validated status. The tables between the `GENERATED` markers are
written by [`tools/block_catalogue.py`](../tools/block_catalogue.py) from the code:

- the block tables come from the `METADATA` mapping in each block module
  ([contract §9](library-contract.md#9-locality-and-cost-metadata));
- the API tables come from the layers' public names.

Do not edit those tables by hand. Change the code, then run
`python tools/block_catalogue.py`. `python tools/block_catalogue.py --check`, run by
[`tests/test_docs.py`](../tests/test_docs.py), fails when the page is out of date.

The [inventory of the code before the library refactor](history/building-blocks-2026-10-11.md),
at the development archive's commit `24a6291`, which sized that refactor, is kept as history.
This page supersedes it.

## How to read the status

Each block's status was set in its `METADATA` when the refactor moved it into the
library. It starts from the inventory's status of the same code; the decisions on
individual blocks are in the development plan's decision log, in the private development
archive.

| Status | Meaning |
|---|---|
| **Ready** | Takes its shape parameters as arguments, and was validated live at more than one setting that a new model would change |
| **Fixed** | Evidence is limited to the settings after the dash, for example "Fixed — window 7". "Fixed — offline only" explicitly has no live validation. |

- **Validated** means the block ran in recorded Earth Engine runs whose outputs were
  compared with a PyTorch float32 reference. The evidence column links those records;
  the inventory's section 2 defines the terms.
- **The library refactor changed no computation.** Every production graph serializes
  byte-identically to the hash recorded before the refactor (the graph-identity gate, G1, in
  [CONTRIBUTING](../CONTRIBUTING.md#which-gate-applies);
  [45 graphs](evidence/l00-golden-graphs.json)), so the recorded validations still
  apply. Graph identity is not itself a numerical result.
- **The validated settings are machine-readable.** Each block's `validated`
  record, in the [table below](#validated-settings), is its status as data: the
  settings the status names, at its values and joined as it joins them; a setting
  the evidence never varied, at its one value; and nothing for a setting the evidence
  varied, which an unqualified "Ready" covers. Composites record the settings of the
  models they ran in. `tensor_ee.core.metadata.check_settings` answers whether a
  call's settings are inside a record, and which are outside it
  ([contract, as built for the converter](library-contract.md#as-built-phase-3)).
- **A parameter is not evidence.** Window, shift, patch and the other settings are
  parameters whose defaults are validated values; another value is validated only
  where a record holds it. The settings validation that followed the refactor varied the
  window lane's windows (5, 7, 8, 12), the patch, the decoder's extents, bins and groups,
  and the raster lane's factors and pooling forms (development archive:
  `docs/library-plan.md`).
- A status changes only with new evidence: a live comparison with a PyTorch reference
  (gate G3 in [CONTRIBUTING](../CONTRIBUTING.md#which-gate-applies)).

**What later validation added.** The settings validation promoted the original 41 blocks: the window
lane is validated live at windows 5, 7, 8 and 12 with shift ⌊w/2⌋, the patch embedding at
patches 2, 4 and 8, the decoder blocks at other extents, bins and group counts, and the
raster lane at nearest factors 2 to 4 and more pooling forms. A setting outside a block's
status is not validated. The window depthwise 3×3 convolution passed
[a live check](evidence/segformer_b0-live.json) at window7: D4 with/without bias, D1024 biased,
and D16 biased within the C4/D16 Mix-FFN fixture. Its catalogue entry names this coverage.
SegFormer-B0 uses it ([model notes](model.md#segformer-b0)). The 36 **Fixed** blocks are
composites promoted from the models when the converter was built: SegFormer-B0's (10) and Swin-T/UPerNet's
(26), each validated live only inside the models it ran in, at their settings.

## Reading the columns

**Layouts** ([contract §3](library-contract.md#3-tensor-types)):
- **R**, raster: an `ee.Image` with one band per channel on a georeferenced `Grid`
  (`Raster`). The CNN path's layout.
- **W**, window: one `[window², C]` float32 array per pixel of the window grid
  (`WindowTensor`, window 7 by default). Token `t = window·a + b` of window `(I, J)`
  is pixel `(window·I + a, window·J + b)`. The Swin-T/UPerNet path's layout.
- **V**, value: an `ee.Array` with no grid, for weights (`ServerArray`) and small
  maps such as the PSP's pooled `[s², C]`.

**Locality**, what one output value depends on: `pointwise` (the same pixel or
token), `k×k` (a neighbourhood of the given radius), `window` (its own window),
`window + ring` (its window and a ring of the given width from the 8 neighbouring
windows) and `extent` (the whole chip or tile). An `extent` block, or one marked
**chip edge**, depends on where the chip ends, so tiling a large area is not the same
as running the model on it.

**Policy**, where float32 rounding happens
([contract §7](library-contract.md#7-precision-policies)):
- `float32_operands`: the CNN path. Weight arrays are cast to float32 before the
  matrix product, and every node's output is materialized as float32.
- `round_each_op`: each Earth Engine operation's double result is rounded at once
  (the attention core).
- `double_round_once`: one float32 rounding per PyTorch operation, with the work in
  double. `grouped_double` is its fusion variant.
- `none`: no arithmetic, and never a cast. `composite`: the parts carry their own
  policies.

The two paths keep separate policies on purpose; each policy has one implementation.
Every block with its own arithmetic takes an explicit `policy=` that defaults to its
validated policy; a composite such as `window_attention` takes none, and its parts keep
their own.

## Blocks

<!-- BEGIN GENERATED: blocks (tools/block_catalogue.py; edit the code, then rerun) -->
87 blocks: 22 in `tensor_ee/blocks/raster` and 65 in `tensor_ee/blocks/window`.

| Layout in → out | Ready | Fixed | Total |
|---|---:|---:|---:|
| R → R | 22 | 0 | 22 |
| R → W | 2 | 1 | 3 |
| W → R | 0 | 1 | 1 |
| W → W | 20 | 36 | 56 |
| W → V | 2 | 0 | 2 |
| V → W | 1 | 0 | 1 |
| V → V | 1 | 1 | 2 |
| **All** | **48** | **39** | **87** |

### Raster-layout (R) blocks

`tensor_ee/blocks/raster`: the CNN path's blocks. `compile_model` and the top-level `tensor_ee` primitives call them.

| Block | Layout | Locality | Policy | Chip edge | Status | Evidence |
|---|---|---|---|---|---|---|
| [`relu`](../tensor_ee/blocks/raster/activations.py) | R → R | pointwise | `float32_operands` | no | Ready | [cnn/results/parity.json](history/cnn/results/parity.json); [cnn/results/resnet18_full64_float32_policy.json](history/cnn/results/resnet18_full64_float32_policy.json); [cnn/results/segwater_js_parity_1x1.json](history/cnn/results/segwater_js_parity_1x1.json) |
| [`sigmoid`](../tensor_ee/blocks/raster/activations.py) | R → R | pointwise | `float32_operands` | no | Ready | [cnn/results/parity.json](history/cnn/results/parity.json); [cnn/results/resnet18_full64_float32_policy.json](history/cnn/results/resnet18_full64_float32_policy.json) |
| [`leaky_relu`](../tensor_ee/blocks/raster/activations.py) | R → R | pointwise | `float32_operands` | no | Ready | [evidence/t02-operators.json](evidence/t02-operators.json); [evidence/t05-primary.json](evidence/t05-primary.json) |
| [`silu`](../tensor_ee/blocks/raster/activations.py) | R → R | pointwise | `float32_operands` | no | Ready | [evidence/t02-operators.json](evidence/t02-operators.json); [evidence/t05-primary.json](evidence/t05-primary.json) |
| [`gelu`](../tensor_ee/blocks/raster/activations.py) | R → R | pointwise | `float32_operands` | no | Ready | [evidence/t02-operators.json](evidence/t02-operators.json); [evidence/t05-primary.json](evidence/t05-primary.json) |
| [`tanh`](../tensor_ee/blocks/raster/activations.py) | R → R | pointwise | `float32_operands` | no | Ready | [evidence/t02-operators.json](evidence/t02-operators.json); [evidence/t05-primary.json](evidence/t05-primary.json) |
| [`relu6`](../tensor_ee/blocks/raster/activations.py) | R → R | pointwise | `float32_operands` | no | Ready | [evidence/raster-hard-activations-offline.json](evidence/raster-hard-activations-offline.json); [evidence/live-fixtures/20261027T163041Z-hard_activations.json](evidence/live-fixtures/20261027T163041Z-hard_activations.json) |
| [`hardsigmoid`](../tensor_ee/blocks/raster/activations.py) | R → R | pointwise | `float32_operands` | no | Ready | [evidence/raster-hard-activations-offline.json](evidence/raster-hard-activations-offline.json); [evidence/live-fixtures/20261027T163041Z-hard_activations.json](evidence/live-fixtures/20261027T163041Z-hard_activations.json) |
| [`hardswish`](../tensor_ee/blocks/raster/activations.py) | R → R | pointwise | `float32_operands` | no | Ready | [evidence/raster-hard-activations-offline.json](evidence/raster-hard-activations-offline.json); [evidence/live-fixtures/20261027T163041Z-hard_activations.json](evidence/live-fixtures/20261027T163041Z-hard_activations.json) |
| [`eval_dropout`](../tensor_ee/blocks/raster/activations.py) | R → R | pointwise | `float32_operands` | no | Ready | [evidence/t02-operators.json](evidence/t02-operators.json); [evidence/t05-primary.json](evidence/t05-primary.json) |
| [`common_score_mask`](../tensor_ee/blocks/raster/activations.py) | R → R | pointwise | `float32_operands` | no | Ready | [evidence/t03-outputs.json](evidence/t03-outputs.json) |
| [`softmax`](../tensor_ee/blocks/raster/activations.py) | R → R | pointwise | `float32_operands` | no | Ready | [evidence/t03-outputs.json](evidence/t03-outputs.json); [evidence/t05-primary.json](evidence/t05-primary.json) |
| [`grouped_conv2d`](../tensor_ee/blocks/raster/conv.py) | R → R | k×k, radius 2 | `float32_operands` | no | Ready — depthwise 3×3 stride 1 or 2, 5×5 stride 1; 3×3 stride 1 with 2 channels per group or channel multiplier 2 | [evidence/raster-grouped-conv-offline.json](evidence/raster-grouped-conv-offline.json); [evidence/live-fixtures/20261029T125301Z-grouped_conv.json](evidence/live-fixtures/20261029T125301Z-grouped_conv.json) |
| [`conv2d`](../tensor_ee/blocks/raster/conv.py) | R → R | k×k, radius 3 | `float32_operands` | no | Ready | [cnn/results/parity.json](history/cnn/results/parity.json); [cnn/results/primitives_float32.json](history/cnn/results/primitives_float32.json); [cnn/results/resnet_primitives.json](history/cnn/results/resnet_primitives.json); [cnn/results/resnet18_full64_float32_policy.json](history/cnn/results/resnet18_full64_float32_policy.json); [cnn/results/segwater_js_parity_1x1.json](history/cnn/results/segwater_js_parity_1x1.json); [cnn/results/tiled_resnet18_2x2_parity.json](history/cnn/results/tiled_resnet18_2x2_parity.json) |
| [`concatenate`](../tensor_ee/blocks/raster/merge.py) | R → R | pointwise | `float32_operands` | no | Ready | [cnn/results/parity.json](history/cnn/results/parity.json); [cnn/results/resnet18_full64_float32_policy.json](history/cnn/results/resnet18_full64_float32_policy.json); [cnn/results/segwater_js_parity_1x1.json](history/cnn/results/segwater_js_parity_1x1.json) |
| [`residual_add`](../tensor_ee/blocks/raster/merge.py) | R → R | pointwise | `float32_operands` | no | Ready | [cnn/results/resnet_primitives.json](history/cnn/results/resnet_primitives.json); [cnn/results/resnet18_full64_float32_policy.json](history/cnn/results/resnet18_full64_float32_policy.json); [cnn/results/segwater_js_parity_1x1.json](history/cnn/results/segwater_js_parity_1x1.json) |
| [`batch_norm`](../tensor_ee/blocks/raster/normalization.py) | R → R | pointwise | `float32_operands` | no | Ready | [evidence/raster-batch-norm-offline.json](evidence/raster-batch-norm-offline.json); [evidence/live-fixtures/20261029T070703Z-batch_norm.json](evidence/live-fixtures/20261029T070703Z-batch_norm.json) |
| [`segmentation_outputs`](../tensor_ee/blocks/raster/outputs.py) | R → R | pointwise | `float32_operands` | no | Ready | [evidence/t03-outputs.json](evidence/t03-outputs.json); [evidence/t05-primary.json](evidence/t05-primary.json) |
| [`binary_threshold`](../tensor_ee/blocks/raster/outputs.py) | R → R | pointwise | `float32_operands` | no | Ready | [evidence/t03-outputs.json](evidence/t03-outputs.json) |
| [`pool2d`](../tensor_ee/blocks/raster/pooling.py) | R → R | k×k, radius 3 | `float32_operands` | no | Ready — aligned avg/max k = 2, 3, 4; max 3×3 stride 1 or 2, padding 1; avg 3×3 stride 1 padding 1, both divisor rules | [evidence/live-fixtures/20261013T065257Z-p02.json](evidence/live-fixtures/20261013T065257Z-p02.json); [evidence/p02-raster-offline.json](evidence/p02-raster-offline.json); [cnn/results/parity.json](history/cnn/results/parity.json); [evidence/t05-primary.json](evidence/t05-primary.json); [cnn/results/resnet_primitives.json](history/cnn/results/resnet_primitives.json); [cnn/results/resnet18_full64_float32_policy.json](history/cnn/results/resnet18_full64_float32_policy.json); [cnn/results/segwater_js_parity_1x1.json](history/cnn/results/segwater_js_parity_1x1.json) |
| [`bilinear_upsample`](../tensor_ee/blocks/raster/resample.py) | R → R | k×k, radius 1 | `float32_operands` | yes (by setting) | Ready — factors 2, 3, 4 (align_corners=False), 2, 4 (align_corners=True) | [evidence/raster-bilinear-offline.json](evidence/raster-bilinear-offline.json); [evidence/live-fixtures/20261029T094157Z-bilinear.json](evidence/live-fixtures/20261029T094157Z-bilinear.json) |
| [`nearest_upsample`](../tensor_ee/blocks/raster/resample.py) | R → R | pointwise | `float32_operands` | no | Ready — factors 2, 3, 4 | [evidence/live-fixtures/20261013T065257Z-p02.json](evidence/live-fixtures/20261013T065257Z-p02.json); [evidence/p02-raster-offline.json](evidence/p02-raster-offline.json); [cnn/results/parity.json](history/cnn/results/parity.json); [cnn/results/resnet18_full64_float32_policy.json](history/cnn/results/resnet18_full64_float32_policy.json); [cnn/results/segwater_js_parity_1x1.json](history/cnn/results/segwater_js_parity_1x1.json) |

### Window-layout (W) and value (V) blocks

`tensor_ee/blocks/window`: the window lane's blocks, and the SegFormer-B0 and Swin-T/UPerNet composites.

| Block | Layout | Locality | Policy | Chip edge | Status | Evidence |
|---|---|---|---|---|---|---|
| [`gelu`](../tensor_ee/blocks/window/activations.py) | W → W | pointwise | `double_round_once` | no | Ready | [building-blocks-2026-10-11.md#35-activations-merges-and-outputs](history/building-blocks-2026-10-11.md#35-activations-merges-and-outputs); [swin_block/notes/findings.md](history/swin_block/notes/findings.md) |
| [`relu`](../tensor_ee/blocks/window/activations.py) | W → W | pointwise | none (no arithmetic) | no | Ready | [building-blocks-2026-10-11.md#35-activations-merges-and-outputs](history/building-blocks-2026-10-11.md#35-activations-merges-and-outputs); [upernet/notes/milestone7_findings.md](history/upernet/notes/milestone7_findings.md) |
| [`class_token_attention`](../tensor_ee/blocks/window/attention.py) | W → W | window | composite | no | Fixed — offline only: windows 3, 14; heads 3, 12; head widths 4, 64 | [evidence/class-token-offline.json](evidence/class-token-offline.json) |
| [`naive_attention`](../tensor_ee/blocks/window/attention.py) | W → W | window | `round_each_op` | no | Ready | [building-blocks-2026-10-11.md#36-linear-layers-and-attention](history/building-blocks-2026-10-11.md#36-linear-layers-and-attention); [ee_attention/notes/findings.md](history/ee_attention/notes/findings.md); [evidence/segformer_b0-live.json](evidence/segformer_b0-live.json) |
| [`split_heads`](../tensor_ee/blocks/window/attention.py) | W → W | pointwise | none (no arithmetic) | no | Ready | [building-blocks-2026-10-11.md#36-linear-layers-and-attention](history/building-blocks-2026-10-11.md#36-linear-layers-and-attention); [swin_block/notes/findings.md](history/swin_block/notes/findings.md) |
| [`relative_position_bias`](../tensor_ee/blocks/window/attention.py) | V → V | pointwise | `double_round_once` | no | Ready — windows 5, 7, 8, 12 | [building-blocks-2026-10-11.md#36-linear-layers-and-attention](history/building-blocks-2026-10-11.md#36-linear-layers-and-attention); [swin_block/notes/findings.md](history/swin_block/notes/findings.md); [evidence/p01-offline.json](evidence/p01-offline.json); [evidence/live-fixtures/20261013T074301Z-p01a.json](evidence/live-fixtures/20261013T074301Z-p01a.json); [evidence/live-fixtures/20261013T074330Z-p01b.json](evidence/live-fixtures/20261013T074330Z-p01b.json); [evidence/live-fixtures/20261013T075542Z-window-relative_position_bias-w12.json](evidence/live-fixtures/20261013T075542Z-window-relative_position_bias-w12.json) |
| [`window_attention`](../tensor_ee/blocks/window/attention.py) | W → W | window | composite | no | Ready — windows 5, 7, 8, 12 | [building-blocks-2026-10-11.md#36-linear-layers-and-attention](history/building-blocks-2026-10-11.md#36-linear-layers-and-attention); *parts: linear (double_round_once), naive_attention (round_each_op)*; [swin_block/notes/findings.md](history/swin_block/notes/findings.md); [swin_pair/notes/findings.md](history/swin_pair/notes/findings.md); [evidence/p01-offline.json](evidence/p01-offline.json); [evidence/live-fixtures/20261013T074301Z-p01a.json](evidence/live-fixtures/20261013T074301Z-p01a.json); [evidence/live-fixtures/20261013T074330Z-p01b.json](evidence/live-fixtures/20261013T074330Z-p01b.json); [evidence/live-fixtures/20261013T075542Z-window-relative_position_bias-w12.json](evidence/live-fixtures/20261013T075542Z-window-relative_position_bias-w12.json) |
| [`shifted_window_mask`](../tensor_ee/blocks/window/attention.py) | W → W | window | `double_round_once` | yes | Ready — windows 5, 7, 8, 12; shift ⌊w/2⌋ | [building-blocks-2026-10-11.md#36-linear-layers-and-attention](history/building-blocks-2026-10-11.md#36-linear-layers-and-attention); [swin_pair/notes/findings.md](history/swin_pair/notes/findings.md); [evidence/p01-offline.json](evidence/p01-offline.json); [evidence/live-fixtures/20261013T074301Z-p01a.json](evidence/live-fixtures/20261013T074301Z-p01a.json); [evidence/live-fixtures/20261013T074330Z-p01b.json](evidence/live-fixtures/20261013T074330Z-p01b.json); [evidence/live-fixtures/20261013T075542Z-window-relative_position_bias-w12.json](evidence/live-fixtures/20261013T075542Z-window-relative_position_bias-w12.json) |
| [`depthwise_conv3x3_window`](../tensor_ee/blocks/window/conv.py) | W → W | window + ring of 1 | `double_round_once` | no | Ready — 3 × 3, stride1, padding1, dilation1, multiplier1; window7; D4 optional bias, D1024 biased; D16 biased in C4/D16 Mix-FFN | [evidence/segformer_b0-offline.json](evidence/segformer_b0-offline.json); [evidence/segformer_b0-live.json](evidence/segformer_b0-live.json) |
| [`conv3x3_window`](../tensor_ee/blocks/window/conv.py) | W → W | window + ring of 1 | `double_round_once` | no | Ready — 3 × 3 kernel, stride 1; windows 5, 7, 8, 12 | [building-blocks-2026-10-11.md#33-convolution](history/building-blocks-2026-10-11.md#33-convolution); [upernet/notes/milestone7_findings.md](history/upernet/notes/milestone7_findings.md); [evidence/p01-offline.json](evidence/p01-offline.json); [evidence/live-fixtures/20261013T074301Z-p01a.json](evidence/live-fixtures/20261013T074301Z-p01a.json); [evidence/live-fixtures/20261013T074330Z-p01b.json](evidence/live-fixtures/20261013T074330Z-p01b.json); [evidence/live-fixtures/20261013T075542Z-window-relative_position_bias-w12.json](evidence/live-fixtures/20261013T075542Z-window-relative_position_bias-w12.json); [evidence/live-fixtures/20261013T115558Z-p04.json](evidence/live-fixtures/20261013T115558Z-p04.json) |
| [`group_product`](../tensor_ee/blocks/window/conv.py) | W → W | window + ring of 1 | `double_round_once` | no | Ready — 1, 2, 4, 8 groups at windows 7, 12; 4 groups at window 8 | [building-blocks-2026-10-11.md#33-convolution](history/building-blocks-2026-10-11.md#33-convolution); *variant: grouped_double*; [upernet/notes/milestone10_findings.md](history/upernet/notes/milestone10_findings.md); [evidence/p04-offline.json](evidence/p04-offline.json); [evidence/live-fixtures/20261013T115558Z-p04.json](evidence/live-fixtures/20261013T115558Z-p04.json); [evidence/c0b-offline.json](evidence/c0b-offline.json); [evidence/live-fixtures/20261013T154410Z-c0b.json](evidence/live-fixtures/20261013T154410Z-c0b.json) |
| [`linear`](../tensor_ee/blocks/window/linear.py) | W → W | pointwise | `double_round_once` | no | Ready | [building-blocks-2026-10-11.md#36-linear-layers-and-attention](history/building-blocks-2026-10-11.md#36-linear-layers-and-attention); [swin_block/notes/findings.md](history/swin_block/notes/findings.md) |
| [`conv1x1`](../tensor_ee/blocks/window/linear.py) | W → W | pointwise | `double_round_once` | no | Ready | [building-blocks-2026-10-11.md#33-convolution](history/building-blocks-2026-10-11.md#33-convolution); [upernet/notes/milestone7_findings.md](history/upernet/notes/milestone7_findings.md) |
| [`concat_rows`](../tensor_ee/blocks/window/merge.py) | W → W | pointwise | none (no arithmetic) | no | Fixed — offline only: one class row followed by spatial rows | [evidence/class-token-offline.json](evidence/class-token-offline.json) |
| [`add`](../tensor_ee/blocks/window/merge.py) | W → W | pointwise | `double_round_once` | no | Ready | [building-blocks-2026-10-11.md#35-activations-merges-and-outputs](history/building-blocks-2026-10-11.md#35-activations-merges-and-outputs); [upernet/notes/milestone9_findings.md](history/upernet/notes/milestone9_findings.md) |
| [`concat`](../tensor_ee/blocks/window/merge.py) | W → W | pointwise | none (no arithmetic) | no | Ready | [building-blocks-2026-10-11.md#35-activations-merges-and-outputs](history/building-blocks-2026-10-11.md#35-activations-merges-and-outputs); [upernet/notes/milestone8_findings.md](history/upernet/notes/milestone8_findings.md) |
| [`layer_norm`](../tensor_ee/blocks/window/normalization.py) | W → W | pointwise | `double_round_once` | no | Ready | [building-blocks-2026-10-11.md#34-normalization](history/building-blocks-2026-10-11.md#34-normalization); [swin_block/notes/findings.md](history/swin_block/notes/findings.md) |
| [`layer_norm2d`](../tensor_ee/blocks/window/normalization.py) | W → W | pointwise | `double_round_once` | no | Ready — windows 5, 7, 8, 12; eps 1e-6 | [building-blocks-2026-10-11.md#34-normalization](history/building-blocks-2026-10-11.md#34-normalization); [evidence/l00-consolidation-probes.json](evidence/l00-consolidation-probes.json); [upernet/notes/milestone8_findings.md](history/upernet/notes/milestone8_findings.md); [upernet/notes/milestone9_findings.md](history/upernet/notes/milestone9_findings.md); [evidence/p01-offline.json](evidence/p01-offline.json); [evidence/live-fixtures/20261013T074301Z-p01a.json](evidence/live-fixtures/20261013T074301Z-p01a.json); [evidence/live-fixtures/20261013T074330Z-p01b.json](evidence/live-fixtures/20261013T074330Z-p01b.json); [evidence/live-fixtures/20261013T075542Z-window-relative_position_bias-w12.json](evidence/live-fixtures/20261013T075542Z-window-relative_position_bias-w12.json) |
| [`batch_norm`](../tensor_ee/blocks/window/normalization.py) | W → W | pointwise | `double_round_once` | no | Ready | [building-blocks-2026-10-11.md#34-normalization](history/building-blocks-2026-10-11.md#34-normalization); [upernet/notes/milestone7_findings.md](history/upernet/notes/milestone7_findings.md) |
| [`patch_embed`](../tensor_ee/blocks/window/patch.py) | R → W | k×k, radius 4 (by setting) | `double_round_once` | no | Ready — patches 2, 4, 8; 1–4 bands; widths 48, 96, 128; windows 5, 7, 8, 12 | [building-blocks-2026-10-11.md#33-convolution](history/building-blocks-2026-10-11.md#33-convolution); [swin_image_encoder/notes/milestone6_findings.md](history/swin_image_encoder/notes/milestone6_findings.md); [evidence/p03-offline.json](evidence/p03-offline.json); [evidence/live-fixtures/20261013T120034Z-p03.json](evidence/live-fixtures/20261013T120034Z-p03.json) |
| [`stride_columns`](../tensor_ee/blocks/window/patch.py) | R → W | k×k, radius 4 (by setting) | none (no arithmetic) | no | Ready — patches 2, 4, 8; 1–4 bands | [building-blocks-2026-10-11.md#33-convolution](history/building-blocks-2026-10-11.md#33-convolution); [swin_image_encoder/notes/milestone6_findings.md](history/swin_image_encoder/notes/milestone6_findings.md); [evidence/p03-offline.json](evidence/p03-offline.json); [evidence/live-fixtures/20261013T120034Z-p03.json](evidence/live-fixtures/20261013T120034Z-p03.json) |
| [`patch_merge`](../tensor_ee/blocks/window/patch.py) | W → W | k×k, radius 1 | `double_round_once` | no | Ready — windows 5, 7, 8, 12 | [building-blocks-2026-10-11.md#37-resampling](history/building-blocks-2026-10-11.md#37-resampling); [swin_hierarchy/notes/milestone4_findings.md](history/swin_hierarchy/notes/milestone4_findings.md); [evidence/p01-offline.json](evidence/p01-offline.json); [evidence/live-fixtures/20261013T074301Z-p01a.json](evidence/live-fixtures/20261013T074301Z-p01a.json); [evidence/live-fixtures/20261013T074330Z-p01b.json](evidence/live-fixtures/20261013T074330Z-p01b.json); [evidence/live-fixtures/20261013T075542Z-window-relative_position_bias-w12.json](evidence/live-fixtures/20261013T075542Z-window-relative_position_bias-w12.json) |
| [`gather_windows`](../tensor_ee/blocks/window/patch.py) | W → W | k×k, radius 1 | none (no arithmetic) | no | Ready — windows 5, 7, 8, 12 | [building-blocks-2026-10-11.md#37-resampling](history/building-blocks-2026-10-11.md#37-resampling); [swin_hierarchy/notes/milestone4_findings.md](history/swin_hierarchy/notes/milestone4_findings.md); [evidence/p01-offline.json](evidence/p01-offline.json); [evidence/live-fixtures/20261013T074301Z-p01a.json](evidence/live-fixtures/20261013T074301Z-p01a.json); [evidence/live-fixtures/20261013T074330Z-p01b.json](evidence/live-fixtures/20261013T074330Z-p01b.json); [evidence/live-fixtures/20261013T075542Z-window-relative_position_bias-w12.json](evidence/live-fixtures/20261013T075542Z-window-relative_position_bias-w12.json) |
| [`merge_windows`](../tensor_ee/blocks/window/patch.py) | W → W | window | none (no arithmetic) | no | Ready — windows 5, 7, 8, 12 | [building-blocks-2026-10-11.md#37-resampling](history/building-blocks-2026-10-11.md#37-resampling); [swin_hierarchy/notes/milestone4_findings.md](history/swin_hierarchy/notes/milestone4_findings.md); [evidence/p01-offline.json](evidence/p01-offline.json); [evidence/live-fixtures/20261013T074301Z-p01a.json](evidence/live-fixtures/20261013T074301Z-p01a.json); [evidence/live-fixtures/20261013T074330Z-p01b.json](evidence/live-fixtures/20261013T074330Z-p01b.json); [evidence/live-fixtures/20261013T075542Z-window-relative_position_bias-w12.json](evidence/live-fixtures/20261013T075542Z-window-relative_position_bias-w12.json) |
| [`window_pool`](../tensor_ee/blocks/window/pooling.py) | W → V | extent | `double_round_once` | no | Ready — extents 5, 7, 8, 12 and 14 tokens with bins (1, 2, 3, 6); extents 8 and 12 with bins (1, 2, 4, 8) | [building-blocks-2026-10-11.md#38-pooling](history/building-blocks-2026-10-11.md#38-pooling); [upernet/notes/milestone8_findings.md](history/upernet/notes/milestone8_findings.md); [evidence/p04-offline.json](evidence/p04-offline.json); [evidence/live-fixtures/20261013T115558Z-p04.json](evidence/live-fixtures/20261013T115558Z-p04.json) |
| [`gather`](../tensor_ee/blocks/window/pooling.py) | W → V | extent | none (no arithmetic) | no | Ready | [building-blocks-2026-10-11.md#38-pooling](history/building-blocks-2026-10-11.md#38-pooling); [upernet/notes/milestone8_findings.md](history/upernet/notes/milestone8_findings.md) |
| [`window_depth_to_space`](../tensor_ee/blocks/window/resample.py) | W → W | window | none (no arithmetic) | no | Fixed — offline only: factor 2; windows 7, 14 to window 7 | [evidence/prithvi-depth-to-space-offline.json](evidence/prithvi-depth-to-space-offline.json) |
| [`window_resize`](../tensor_ee/blocks/window/resample.py) | W → W | window + ring of 1 | `double_round_once` | yes (by setting) | Ready — windows 7, 8; factors 2, 4, 8 (align_corners=False), 4 (align_corners=True) | [building-blocks-2026-10-11.md#37-resampling](history/building-blocks-2026-10-11.md#37-resampling); [upernet/notes/milestone7_findings.md](history/upernet/notes/milestone7_findings.md); [upernet/notes/milestone11_findings.md](history/upernet/notes/milestone11_findings.md); [evidence/c0b-offline.json](evidence/c0b-offline.json); [evidence/live-fixtures/20261013T154410Z-c0b.json](evidence/live-fixtures/20261013T154410Z-c0b.json) |
| [`window_matrix_resize`](../tensor_ee/blocks/window/resample.py) | V → W | extent | `double_round_once` | no | Ready — windows 7, 8; sources 1, 2, 3, 6 onto one window (align_corners=False) | [building-blocks-2026-10-11.md#37-resampling](history/building-blocks-2026-10-11.md#37-resampling); [upernet/notes/milestone8_findings.md](history/upernet/notes/milestone8_findings.md); [evidence/c0b-offline.json](evidence/c0b-offline.json); [evidence/live-fixtures/20261013T154410Z-c0b.json](evidence/live-fixtures/20261013T154410Z-c0b.json) |
| [`channel_norm`](../tensor_ee/blocks/window/segformer.py) | W → W | pointwise | composite | no | Fixed — SegFormer-B0 (mit_b0) at 224 × 224: widths 32, 64, 160, 256; eps 1e-5, 1e-6; window 7 | [evidence/segformer_b0-fullmodel-live.json](evidence/segformer_b0-fullmodel-live.json); [evidence/segformer_b0-live.json](evidence/segformer_b0-live.json); *parts: layer_norm (eps 1e-5) or layer_norm2d (eps 1e-6), double_round_once* |
| [`decimate`](../tensor_ee/blocks/window/segformer.py) | W → W | k×k, radius 2 | none (no arithmetic) | no | Fixed — SegFormer-B0 (mit_b0) at 224 × 224: factor 4 at width 32, factor 2 at widths 64, 160, 256; window 7 | [evidence/segformer_b0-fullmodel-live.json](evidence/segformer_b0-fullmodel-live.json); [evidence/segformer_b0-live.json](evidence/segformer_b0-live.json) |
| [`overlap_embedding`](../tensor_ee/blocks/window/segformer.py) | W → W | k×k, radius 3 | composite | no | Fixed — SegFormer-B0 (mit_b0) at 224 × 224: k7/s4 2 → 32; k3/s2 32 → 64, 64 → 160, 160 → 256; window 7 | [evidence/segformer_b0-fullmodel-live.json](evidence/segformer_b0-fullmodel-live.json); [evidence/segformer_b0-live.json](evidence/segformer_b0-live.json); *parts: linear or conv3x3_window and layer_norm (double_round_once), decimate (none)* |
| [`spatial_reduction`](../tensor_ee/blocks/window/segformer.py) | W → W | extent | composite | no | Fixed — SegFormer-B0 (mit_b0) at 224 × 224: widths 32, 64, 160, 256 at reduction ratios 8, 4, 2, 1; window 7 | [evidence/segformer_b0-fullmodel-live.json](evidence/segformer_b0-fullmodel-live.json); [evidence/segformer_b0-live.json](evidence/segformer_b0-live.json); *parts: stride_columns (none), linear and layer_norm (double_round_once)* |
| [`shared_key_attention`](../tensor_ee/blocks/window/segformer.py) | W → W | extent | composite | yes | Fixed — SegFormer-B0 (mit_b0) at 224 × 224: widths 32, 64, 160, 256; heads 1, 2, 5, 8; one 7 × 7 key window | [evidence/segformer_b0-fullmodel-live.json](evidence/segformer_b0-fullmodel-live.json); [evidence/segformer_b0-live.json](evidence/segformer_b0-live.json); *parts: gather (none), naive_attention (round_each_op)* |
| [`global_attention`](../tensor_ee/blocks/window/segformer.py) | W → W | extent | composite | yes | Fixed — SegFormer-B0 (mit_b0) at 224 × 224: widths 32, 64, 160, 256; heads 1, 2, 5, 8; reduction ratios 8, 4, 2, 1; window 7 | [evidence/segformer_b0-fullmodel-live.json](evidence/segformer_b0-fullmodel-live.json); [evidence/segformer_b0-live.json](evidence/segformer_b0-live.json); *parts: linear (double_round_once), spatial_reduction, shared_key_attention* |
| [`mix_ffn`](../tensor_ee/blocks/window/segformer.py) | W → W | window + ring of 1 | composite | no | Fixed — SegFormer-B0 (mit_b0) at 224 × 224: widths 32, 64, 160, 256, hidden 4 × width; window 7 | [evidence/segformer_b0-fullmodel-live.json](evidence/segformer_b0-fullmodel-live.json); [evidence/segformer_b0-live.json](evidence/segformer_b0-live.json); *parts: linear, depthwise_conv3x3_window, gelu (double_round_once)* |
| [`mit_block`](../tensor_ee/blocks/window/segformer.py) | W → W | extent | composite | yes | Fixed — SegFormer-B0 (mit_b0) at 224 × 224: widths 32, 64, 160, 256; heads 1, 2, 5, 8; reduction ratios 8, 4, 2, 1; window 7 | [evidence/segformer_b0-fullmodel-live.json](evidence/segformer_b0-fullmodel-live.json); [evidence/segformer_b0-live.json](evidence/segformer_b0-live.json); *parts: channel_norm, global_attention, mix_ffn, add (double_round_once)* |
| [`project_pyramid`](../tensor_ee/blocks/window/segformer.py) | W → W | window + ring of 1 | composite | no | Fixed — SegFormer-B0 (mit_b0) at 224 × 224: widths 32, 64, 160, 256 to 256 on C1 56 × 56; window 7 | [evidence/segformer_b0-fullmodel-live.json](evidence/segformer_b0-fullmodel-live.json); [evidence/segformer_b0-live.json](evidence/segformer_b0-live.json); *parts: linear and window_resize (double_round_once), concat (none)* |
| [`fuse_head`](../tensor_ee/blocks/window/segformer.py) | W → W | window + ring of 1 | composite | yes | Fixed — SegFormer-B0 (mit_b0) at 224 × 224: 1024 → 256 channels → 2 classes, 56 × 56 to 224 × 224; window 7 | [evidence/segformer_b0-fullmodel-live.json](evidence/segformer_b0-fullmodel-live.json); [evidence/segformer_b0-live.json](evidence/segformer_b0-live.json); *parts: conv1x1, batch_norm and window_resize (double_round_once), relu (none)* |
| [`swin_mlp_branch`](../tensor_ee/blocks/window/swin.py) | W → W | pointwise | composite | no | Fixed — Swin-T224 and the capstone: widths 96, 192, 384, 768 at window 7 and 48, 96, 192, 384 at window 8; hidden 4 × width | [evidence/t06-rehearsal.json](evidence/t06-rehearsal.json); [upernet/notes/milestone12_findings.md](history/upernet/notes/milestone12_findings.md); [evidence/c1-capstone.json](evidence/c1-capstone.json); [swin_block/notes/findings.md](history/swin_block/notes/findings.md); *parts: layer_norm, linear and gelu (double_round_once), a float32 addition* |
| [`swin_block`](../tensor_ee/blocks/window/swin.py) | W → W | window | composite | no | Fixed — Swin-T224 and the capstone: widths 96, 192, 384, 768, heads 3, 6, 12, 24 at window 7; widths 48, 96, 192, 384, heads 2, 4, 8, 16 at window 8 | [evidence/t06-rehearsal.json](evidence/t06-rehearsal.json); [upernet/notes/milestone12_findings.md](history/upernet/notes/milestone12_findings.md); [evidence/c1-capstone.json](evidence/c1-capstone.json); [swin_block/notes/findings.md](history/swin_block/notes/findings.md); *parts: layer_norm, window_attention, linear, swin_mlp_branch* |
| [`shifted_swin_block`](../tensor_ee/blocks/window/swin.py) | W → W | window + ring of 4 | composite | yes | Fixed — Swin-T224 and the capstone: widths 96, 192, 384, heads 3, 6, 12 at window 7; widths 48, 96, 192, heads 2, 4, 8 at window 8; shift ⌊w/2⌋; not stage 4 (one window, unshifted) | [evidence/t06-rehearsal.json](evidence/t06-rehearsal.json); [upernet/notes/milestone12_findings.md](history/upernet/notes/milestone12_findings.md); [evidence/c1-capstone.json](evidence/c1-capstone.json); [swin_pair/notes/findings.md](history/swin_pair/notes/findings.md); *parts: layer_norm, shift_windows (none), window_attention with shifted_window_mask, linear, swin_mlp_branch* |
| [`swin_stage`](../tensor_ee/blocks/window/swin.py) | W → W | extent | composite | yes | Fixed — Swin-T224 and the capstone: widths 96, 192, 384, 768, heads 3, 6, 12, 24 at window 7; widths 48, 96, 192, 384, heads 2, 4, 8, 16 at window 8; depths 2, 2, 6, 2 and 2, 2, 2, 2 | [evidence/t06-rehearsal.json](evidence/t06-rehearsal.json); [upernet/notes/milestone12_findings.md](history/upernet/notes/milestone12_findings.md); [evidence/c1-capstone.json](evidence/c1-capstone.json); [swin_encoder/notes/milestone5_findings.md](history/swin_encoder/notes/milestone5_findings.md); *parts: swin_block, periodic_extension (none), shifted_swin_block* |
| [`swin_patch_embedding`](../tensor_ee/blocks/window/swin.py) | R → W | k×k, radius 2 (by setting) | composite | no | Fixed — Swin-T224 and the capstone: patch 4; 2 bands → width 96 at window 7, 3 bands → width 48 at window 8 | [evidence/t06-rehearsal.json](evidence/t06-rehearsal.json); [upernet/notes/milestone12_findings.md](history/upernet/notes/milestone12_findings.md); [evidence/c1-capstone.json](evidence/c1-capstone.json); [swin_image_encoder/notes/milestone6_findings.md](history/swin_image_encoder/notes/milestone6_findings.md); *parts: patch_embed (double_round_once)* |
| [`swin_encoder`](../tensor_ee/blocks/window/swin.py) | W → W | extent | composite | yes | Fixed — Swin-T224 and the capstone: 8 × 8 stage-1 windows of 7 (stages 1–3, then 4) and of 8 (stages 1–4), without the final norm | [evidence/t06-rehearsal.json](evidence/t06-rehearsal.json); [upernet/notes/milestone12_findings.md](history/upernet/notes/milestone12_findings.md); [evidence/c1-capstone.json](evidence/c1-capstone.json); [swin_encoder/notes/milestone5_findings.md](history/swin_encoder/notes/milestone5_findings.md); *parts: patch_merge, swin_stage, layer_norm* |
| [`psp_layer_norm`](../tensor_ee/blocks/window/upernet.py) | W → W | pointwise | composite | no | Fixed — Swin-T224 and the capstone: C4 widths 768 (window 7) and 384 (window 8) | [evidence/t06-rehearsal.json](evidence/t06-rehearsal.json); [upernet/notes/milestone12_findings.md](history/upernet/notes/milestone12_findings.md); [evidence/c1-capstone.json](evidence/c1-capstone.json); [upernet/notes/milestone8_findings.md](history/upernet/notes/milestone8_findings.md); *parts: layer_norm2d (double_round_once)* |
| [`psp_branch`](../tensor_ee/blocks/window/upernet.py) | V → V | pointwise | composite | no | Fixed — Swin-T224 and the capstone: bins 1, 2, 3, 6; C4 widths 768 and 384 to decoder widths 256 (window 7) and 64 (window 8) | [evidence/t06-rehearsal.json](evidence/t06-rehearsal.json); [upernet/notes/milestone12_findings.md](history/upernet/notes/milestone12_findings.md); [evidence/c1-capstone.json](evidence/c1-capstone.json); [upernet/notes/milestone8_findings.md](history/upernet/notes/milestone8_findings.md); *parts: conv1x1 and batch_norm (double_round_once), ReLU (none)* |
| [`psp_forward`](../tensor_ee/blocks/window/upernet.py) | W → W | extent | composite | no | Fixed — Swin-T224 and the capstone: bins (1, 2, 3, 6) on C4's 7 × 7 and 8 × 8 tokens; C4 widths 768 and 384 to decoder widths 256 (window 7) and 64 (window 8) | [evidence/t06-rehearsal.json](evidence/t06-rehearsal.json); [upernet/notes/milestone12_findings.md](history/upernet/notes/milestone12_findings.md); [evidence/c1-capstone.json](evidence/c1-capstone.json); [upernet/notes/milestone8_findings.md](history/upernet/notes/milestone8_findings.md); *parts: psp_layer_norm, window_pool, psp_branch, window_matrix_resize, concat, conv3x3_window, batch_norm, relu* |
| [`fpn_layer_norm`](../tensor_ee/blocks/window/upernet.py) | W → W | pointwise | composite | no | Fixed — Swin-T224 and the capstone: C1–C3 widths 96, 192, 384 (window 7) and 48, 96, 192 (window 8) | [evidence/t06-rehearsal.json](evidence/t06-rehearsal.json); [upernet/notes/milestone12_findings.md](history/upernet/notes/milestone12_findings.md); [evidence/c1-capstone.json](evidence/c1-capstone.json); [upernet/notes/milestone9_findings.md](history/upernet/notes/milestone9_findings.md); *parts: layer_norm2d (double_round_once)* |
| [`fpn_lateral`](../tensor_ee/blocks/window/upernet.py) | W → W | pointwise | composite | no | Fixed — Swin-T224 and the capstone: C1–C3 widths 96, 192, 384 and 48, 96, 192 to decoder widths 256 (window 7) and 64 (window 8) | [evidence/t06-rehearsal.json](evidence/t06-rehearsal.json); [upernet/notes/milestone12_findings.md](history/upernet/notes/milestone12_findings.md); [evidence/c1-capstone.json](evidence/c1-capstone.json); [upernet/notes/milestone9_findings.md](history/upernet/notes/milestone9_findings.md); *parts: conv1x1 and batch_norm (double_round_once), relu (none)* |
| [`fpn_upsample`](../tensor_ee/blocks/window/upernet.py) | W → W | window + ring of 1 | composite | no | Fixed — Swin-T224 and the capstone: × 2 (align_corners=False) at decoder widths 256 (window 7) and 64 (window 8) | [evidence/t06-rehearsal.json](evidence/t06-rehearsal.json); [upernet/notes/milestone12_findings.md](history/upernet/notes/milestone12_findings.md); [evidence/c1-capstone.json](evidence/c1-capstone.json); [upernet/notes/milestone9_findings.md](history/upernet/notes/milestone9_findings.md); *parts: window_resize (double_round_once)* |
| [`fpn_merge`](../tensor_ee/blocks/window/upernet.py) | W → W | window + ring of 1 | composite | no | Fixed — Swin-T224 and the capstone: × 2 (align_corners=False) and the addition at decoder widths 256 (window 7) and 64 (window 8) | [evidence/t06-rehearsal.json](evidence/t06-rehearsal.json); [upernet/notes/milestone12_findings.md](history/upernet/notes/milestone12_findings.md); [evidence/c1-capstone.json](evidence/c1-capstone.json); [upernet/notes/milestone9_findings.md](history/upernet/notes/milestone9_findings.md); *parts: fpn_upsample, add (double_round_once)* |
| [`fpn_conv_block`](../tensor_ee/blocks/window/upernet.py) | W → W | window + ring of 1 | composite | no | Fixed — Swin-T224 and the capstone: decoder widths 256 (window 7) and 64 (window 8) | [evidence/t06-rehearsal.json](evidence/t06-rehearsal.json); [upernet/notes/milestone12_findings.md](history/upernet/notes/milestone12_findings.md); [evidence/c1-capstone.json](evidence/c1-capstone.json); [upernet/notes/milestone9_findings.md](history/upernet/notes/milestone9_findings.md); *parts: conv3x3_window and batch_norm (double_round_once), relu (none)* |
| [`fpn_forward`](../tensor_ee/blocks/window/upernet.py) | W → W | extent | composite | no | Fixed — Swin-T224 and the capstone: C4's 7 × 7 and 8 × 8 tokens; C1–C4 widths 96, 192, 384, 768 and 48, 96, 192, 384 to decoder widths 256 (window 7) and 64 (window 8) | [evidence/t06-rehearsal.json](evidence/t06-rehearsal.json); [upernet/notes/milestone12_findings.md](history/upernet/notes/milestone12_findings.md); [evidence/c1-capstone.json](evidence/c1-capstone.json); [upernet/notes/milestone9_findings.md](history/upernet/notes/milestone9_findings.md); *parts: psp_forward, fpn_layer_norm, fpn_lateral, fpn_merge, fpn_conv_block* |
| [`fusion_resize_to_c1`](../tensor_ee/blocks/window/upernet.py) | W → W | window + ring of 1 | composite | no | Fixed — Swin-T224 and the capstone: P2, P3 and the PSP output onto C1's 56 × 56 and 64 × 64 tokens at decoder widths 256 (window 7) and 64 (window 8) | [evidence/t06-rehearsal.json](evidence/t06-rehearsal.json); [upernet/notes/milestone12_findings.md](history/upernet/notes/milestone12_findings.md); [evidence/c1-capstone.json](evidence/c1-capstone.json); [upernet/notes/milestone10_findings.md](history/upernet/notes/milestone10_findings.md); *parts: fusion_resize_level* |
| [`fusion_resize_level`](../tensor_ee/blocks/window/upernet.py) | W → W | window + ring of 1 | composite | no | Fixed — Swin-T224 and the capstone: factors 1, 2, 4, 8 (align_corners=False) at decoder widths 256 (window 7) and 64 (window 8) | [evidence/t06-rehearsal.json](evidence/t06-rehearsal.json); [upernet/notes/milestone12_findings.md](history/upernet/notes/milestone12_findings.md); [evidence/c1-capstone.json](evidence/c1-capstone.json); [upernet/notes/milestone10_findings.md](history/upernet/notes/milestone10_findings.md); *parts: window_resize (double_round_once)* |
| [`fusion_concat`](../tensor_ee/blocks/window/upernet.py) | W → W | pointwise | none (no arithmetic) | no | Fixed — Swin-T224 and the capstone: four levels at decoder widths 256 (window 7) and 64 (window 8) | [evidence/t06-rehearsal.json](evidence/t06-rehearsal.json); [upernet/notes/milestone12_findings.md](history/upernet/notes/milestone12_findings.md); [evidence/c1-capstone.json](evidence/c1-capstone.json); [upernet/notes/milestone10_findings.md](history/upernet/notes/milestone10_findings.md); *parts: concat (none)* |
| [`fusion_bn_relu`](../tensor_ee/blocks/window/upernet.py) | W → W | pointwise | composite | no | Fixed — Swin-T224 and the capstone: decoder widths 256 (window 7) and 64 (window 8) | [evidence/t06-rehearsal.json](evidence/t06-rehearsal.json); [upernet/notes/milestone12_findings.md](history/upernet/notes/milestone12_findings.md); [evidence/c1-capstone.json](evidence/c1-capstone.json); [upernet/notes/milestone10_findings.md](history/upernet/notes/milestone10_findings.md); *parts: batch_norm (double_round_once), relu (none)* |
| [`fusion_grouped_double`](../tensor_ee/blocks/window/upernet.py) | W → W | window + ring of 1 | `grouped_double` | no | Fixed — Swin-T224 and the capstone: 4 groups, 1024 → 256 (window 7) and 256 → 64 (window 8) | [evidence/t06-rehearsal.json](evidence/t06-rehearsal.json); [upernet/notes/milestone12_findings.md](history/upernet/notes/milestone12_findings.md); [evidence/c1-capstone.json](evidence/c1-capstone.json); [upernet/notes/milestone10_findings.md](history/upernet/notes/milestone10_findings.md); *parts: group_product (grouped_double), fusion_bn_relu* |
| [`fusion_forward`](../tensor_ee/blocks/window/upernet.py) | W → W | window + ring of 2 | composite | no | Fixed — Swin-T224 and the capstone: 4 groups on C1's 56 × 56 and 64 × 64 tokens at decoder widths 256 (window 7) and 64 (window 8) | [evidence/t06-rehearsal.json](evidence/t06-rehearsal.json); [upernet/notes/milestone12_findings.md](history/upernet/notes/milestone12_findings.md); [evidence/c1-capstone.json](evidence/c1-capstone.json); [upernet/notes/milestone10_findings.md](history/upernet/notes/milestone10_findings.md); *parts: fusion_resize_to_c1, fusion_concat, fusion_grouped_double* |
| [`head_conv`](../tensor_ee/blocks/window/upernet.py) | W → W | pointwise | composite | no | Fixed — Swin-T224 and the capstone: 256 → 2 classes (window 7) and 64 → 3 classes (window 8) | [evidence/t06-rehearsal.json](evidence/t06-rehearsal.json); [upernet/notes/milestone12_findings.md](history/upernet/notes/milestone12_findings.md); [evidence/c1-capstone.json](evidence/c1-capstone.json); [upernet/notes/milestone11_findings.md](history/upernet/notes/milestone11_findings.md); *parts: conv1x1 (double_round_once)* |
| [`head_upsample`](../tensor_ee/blocks/window/upernet.py) | W → W | window + ring of 1 | composite | yes | Fixed — Swin-T224 and the capstone: × 4 (align_corners=True) from C1's 56 × 56 (window 7) and 64 × 64 (window 8) tokens | [evidence/t06-rehearsal.json](evidence/t06-rehearsal.json); [upernet/notes/milestone12_findings.md](history/upernet/notes/milestone12_findings.md); [evidence/c1-capstone.json](evidence/c1-capstone.json); [upernet/notes/milestone11_findings.md](history/upernet/notes/milestone11_findings.md); *parts: window_resize (double_round_once)* |
| [`head_forward`](../tensor_ee/blocks/window/upernet.py) | W → W | window + ring of 1 | composite | yes | Fixed — Swin-T224 and the capstone: 256 → 2 and 64 → 3 classes, × 4 from C1's 56 × 56 and 64 × 64 tokens | [evidence/t06-rehearsal.json](evidence/t06-rehearsal.json); [upernet/notes/milestone12_findings.md](history/upernet/notes/milestone12_findings.md); [evidence/c1-capstone.json](evidence/c1-capstone.json); [upernet/notes/milestone11_findings.md](history/upernet/notes/milestone11_findings.md); *parts: head_conv, head_upsample* |
| [`head_logits`](../tensor_ee/blocks/window/upernet.py) | W → W | window + ring of 1 | composite | yes | Fixed — Swin-T224 and the capstone: 256 → 2 and 64 → 3 classes, × 4 from C1's 56 × 56 and 64 × 64 tokens | [evidence/t06-rehearsal.json](evidence/t06-rehearsal.json); [upernet/notes/milestone12_findings.md](history/upernet/notes/milestone12_findings.md); [evidence/c1-capstone.json](evidence/c1-capstone.json); [upernet/notes/milestone11_findings.md](history/upernet/notes/milestone11_findings.md); *parts: head_forward* |
| [`head_dense`](../tensor_ee/blocks/window/upernet.py) | W → R | window | none (no arithmetic) | no | Fixed — Swin-T224 and the capstone: 2 and 3 classes on 224 × 224 (window 7) and 256 × 256 (window 8) tokens | [evidence/t06-rehearsal.json](evidence/t06-rehearsal.json); [upernet/notes/milestone12_findings.md](history/upernet/notes/milestone12_findings.md); [evidence/c1-capstone.json](evidence/c1-capstone.json); [upernet/notes/milestone11_findings.md](history/upernet/notes/milestone11_findings.md); *parts: to_dense (none)* |

### Evidence notes

- Every evidence path names an existing file.
- Entries in *italics* are notes in the metadata, not files.
- Locality and chip edge hold over every validated setting; *(by setting)* marks a property whose exact value at a call's settings is `per_setting`'s (`metadata_at`).

### Validated settings

Each block's `validated` record, the status in machine-readable form: 87 records. Each `{…}` is one configuration, whose values were validated together; a record is their union (∪). A setting no configuration names is not restricted. Names come from the block function, its cost functions and the sources listed.

| Block | Layout | Validated settings | Names from |
|---|---|---|---|
| [`relu`](../tensor_ee/blocks/raster/activations.py) | R → R | any accepted call | — |
| [`sigmoid`](../tensor_ee/blocks/raster/activations.py) | R → R | any accepted call | — |
| [`leaky_relu`](../tensor_ee/blocks/raster/activations.py) | R → R | {clip_footprints False} | — |
| [`silu`](../tensor_ee/blocks/raster/activations.py) | R → R | {clip_footprints False} | — |
| [`gelu`](../tensor_ee/blocks/raster/activations.py) | R → R | {clip_footprints False} | — |
| [`tanh`](../tensor_ee/blocks/raster/activations.py) | R → R | {clip_footprints False} | — |
| [`relu6`](../tensor_ee/blocks/raster/activations.py) | R → R | {clip_footprints False} | — |
| [`hardsigmoid`](../tensor_ee/blocks/raster/activations.py) | R → R | {clip_footprints False} | — |
| [`hardswish`](../tensor_ee/blocks/raster/activations.py) | R → R | {clip_footprints False} | — |
| [`eval_dropout`](../tensor_ee/blocks/raster/activations.py) | R → R | {clip_footprints False} | — |
| [`common_score_mask`](../tensor_ee/blocks/raster/activations.py) | R → R | any accepted call | — |
| [`softmax`](../tensor_ee/blocks/raster/activations.py) | R → R | {clip_footprints False} | — |
| [`grouped_conv2d`](../tensor_ee/blocks/raster/conv.py) | R → R | {clip_footprints False} | — |
| [`conv2d`](../tensor_ee/blocks/raster/conv.py) | R → R | any accepted call | — |
| [`concatenate`](../tensor_ee/blocks/raster/merge.py) | R → R | {dim 1} | — |
| [`residual_add`](../tensor_ee/blocks/raster/merge.py) | R → R | any accepted call | — |
| [`batch_norm`](../tensor_ee/blocks/raster/normalization.py) | R → R | {clip_footprints False; eps 1e-05} | `BatchNormConstants` |
| [`segmentation_outputs`](../tensor_ee/blocks/raster/outputs.py) | R → R | any accepted call | — |
| [`binary_threshold`](../tensor_ee/blocks/raster/outputs.py) | R → R | any accepted call | — |
| [`pool2d`](../tensor_ee/blocks/raster/pooling.py) | R → R | {kernel 2; padding 0; reducer 'mean' or 'max'; stride 2} ∪ {kernel 3; padding 0; reducer 'mean' or 'max'; stride 3} ∪ {kernel 4; padding 0; reducer 'mean' or 'max'; stride 4} ∪ {kernel 3; padding 1; reducer 'max'; stride 1 or 2} ∪ {count_include_pad True or False; kernel 3; padding 1; reducer 'mean'; stride 1} | `PoolConfig` |
| [`bilinear_upsample`](../tensor_ee/blocks/raster/resample.py) | R → R | {align_corners False; clip_footprints False; factor 2 or 3 or 4} ∪ {align_corners True; clip_footprints False; factor 2 or 4} | `BilinearConstants` |
| [`nearest_upsample`](../tensor_ee/blocks/raster/resample.py) | R → R | {scale_factor 2 or 3 or 4} | — |
| [`gelu`](../tensor_ee/blocks/window/activations.py) | W → W | any accepted call | — |
| [`relu`](../tensor_ee/blocks/window/activations.py) | W → W | any accepted call | — |
| [`class_token_attention`](../tensor_ee/blocks/window/attention.py) | W → W | none live | — |
| [`naive_attention`](../tensor_ee/blocks/window/attention.py) | W → W | any accepted call | — |
| [`split_heads`](../tensor_ee/blocks/window/attention.py) | W → W | any accepted call | — |
| [`relative_position_bias`](../tensor_ee/blocks/window/attention.py) | V → V | {window 5 or 7 or 8 or 12} | — |
| [`window_attention`](../tensor_ee/blocks/window/attention.py) | W → W | {window 5 or 7 or 8 or 12} | — |
| [`shifted_window_mask`](../tensor_ee/blocks/window/attention.py) | W → W | {shift 2 or None; value -100.0; window 5} ∪ {shift 3 or None; value -100.0; window 7} ∪ {shift 4 or None; value -100.0; window 8} ∪ {shift 6 or None; value -100.0; window 12} | — |
| [`depthwise_conv3x3_window`](../tensor_ee/blocks/window/conv.py) | W → W | {bias True or False; channels 4; dilation 1; k 3; multiplier 1; padding 1; stride 1; window 7} ∪ {bias True; channels 16 or 1024; dilation 1; k 3; multiplier 1; padding 1; stride 1; window 7} | `DepthwiseConv3x3Constants` |
| [`conv3x3_window`](../tensor_ee/blocks/window/conv.py) | W → W | {k 3; window 5 or 7 or 8 or 12} | `Conv3x3Constants` |
| [`group_product`](../tensor_ee/blocks/window/conv.py) | W → W | {groups 1 or 2 or 4 or 8; k 3; window 7 or 12} ∪ {groups 4; k 3; window 8} | `group_weight`, `Conv3x3Constants` |
| [`linear`](../tensor_ee/blocks/window/linear.py) | W → W | any accepted call | — |
| [`conv1x1`](../tensor_ee/blocks/window/linear.py) | W → W | any accepted call | — |
| [`concat_rows`](../tensor_ee/blocks/window/merge.py) | W → W | none live | — |
| [`add`](../tensor_ee/blocks/window/merge.py) | W → W | any accepted call | — |
| [`concat`](../tensor_ee/blocks/window/merge.py) | W → W | any accepted call | — |
| [`layer_norm`](../tensor_ee/blocks/window/normalization.py) | W → W | {eps 1e-05} | — |
| [`layer_norm2d`](../tensor_ee/blocks/window/normalization.py) | W → W | {eps 1e-06; window 5 or 7 or 8 or 12} | — |
| [`batch_norm`](../tensor_ee/blocks/window/normalization.py) | W → W | {eps 1e-05} | `BatchNormConstants` |
| [`patch_embed`](../tensor_ee/blocks/window/patch.py) | R → W | {channels 1 or 2 or 3 or 4; dim 48 or 96 or 128; patch 2 or 4 or 8; window 5 or 7 or 8 or 12} | — |
| [`stride_columns`](../tensor_ee/blocks/window/patch.py) | R → W | {channels 1 or 2 or 3 or 4; patch 2 or 4 or 8} | `patch_embed_values` |
| [`patch_merge`](../tensor_ee/blocks/window/patch.py) | W → W | {bias False; window 5 or 7 or 8 or 12} | — |
| [`gather_windows`](../tensor_ee/blocks/window/patch.py) | W → W | {window 5 or 7 or 8 or 12} | — |
| [`merge_windows`](../tensor_ee/blocks/window/patch.py) | W → W | {window 5 or 7 or 8 or 12} | — |
| [`window_pool`](../tensor_ee/blocks/window/pooling.py) | W → V | {height 5; sizes (1, 2, 3, 6); width 5} ∪ {height 7; sizes (1, 2, 3, 6); width 7} ∪ {height 8; sizes (1, 2, 3, 6); width 8} ∪ {height 12; sizes (1, 2, 3, 6); width 12} ∪ {height 14; sizes (1, 2, 3, 6); width 14} ∪ {height 8; sizes (1, 2, 4, 8); width 8} ∪ {height 12; sizes (1, 2, 4, 8); width 12} | — |
| [`gather`](../tensor_ee/blocks/window/pooling.py) | W → V | any accepted call | — |
| [`window_depth_to_space`](../tensor_ee/blocks/window/resample.py) | W → W | none live | — |
| [`window_resize`](../tensor_ee/blocks/window/resample.py) | W → W | {align_corners False; factor 2 or 4 or 8; window 7 or 8} ∪ {align_corners True; factor 4; window 7 or 8} | — |
| [`window_matrix_resize`](../tensor_ee/blocks/window/resample.py) | V → W | {align_corners False; col0 0; height 7; in_size (1, 1) or (2, 2) or (3, 3) or (6, 6); out_size None; row0 0; width 7; window 7} ∪ {align_corners False; col0 0; height 8; in_size (1, 1) or (2, 2) or (3, 3) or (6, 6); out_size None; row0 0; width 8; window 8} | — |
| [`channel_norm`](../tensor_ee/blocks/window/segformer.py) | W → W | {channels 32 or 64 or 160 or 256; eps 1e-05 or 1e-06; window 7} | `norm_constants` |
| [`decimate`](../tensor_ee/blocks/window/segformer.py) | W → W | {factor 2 or 4; window 7} | — |
| [`overlap_embedding`](../tensor_ee/blocks/window/segformer.py) | W → W | {cin 2; cout 32; kernel 7; stride 4; window 7} ∪ {cin 32; cout 64; kernel 3; stride 2; window 7} ∪ {cin 64; cout 160; kernel 3; stride 2; window 7} ∪ {cin 160; cout 256; kernel 3; stride 2; window 7} | `overlap_constants` |
| [`spatial_reduction`](../tensor_ee/blocks/window/segformer.py) | W → W | {dim 32; ratio 8; window 7} ∪ {dim 64; ratio 4; window 7} ∪ {dim 160; ratio 2; window 7} ∪ {dim 256; ratio 1; window 7} | `attention_constants` |
| [`shared_key_attention`](../tensor_ee/blocks/window/segformer.py) | W → W | {dim 32; heads 1; window 7} ∪ {dim 64; heads 2; window 7} ∪ {dim 160; heads 5; window 7} ∪ {dim 256; heads 8; window 7} | `attention_constants` |
| [`global_attention`](../tensor_ee/blocks/window/segformer.py) | W → W | {dim 32; heads 1; ratio 8; window 7} ∪ {dim 64; heads 2; ratio 4; window 7} ∪ {dim 160; heads 5; ratio 2; window 7} ∪ {dim 256; heads 8; ratio 1; window 7} | `attention_constants` |
| [`mix_ffn`](../tensor_ee/blocks/window/segformer.py) | W → W | {dim 32 or 64 or 160 or 256; window 7} | `mix_ffn_constants` |
| [`mit_block`](../tensor_ee/blocks/window/segformer.py) | W → W | {dim 32; heads 1; ratio 8; window 7} ∪ {dim 64; heads 2; ratio 4; window 7} ∪ {dim 160; heads 5; ratio 2; window 7} ∪ {dim 256; heads 8; ratio 1; window 7} | `mit_block_constants` |
| [`project_pyramid`](../tensor_ee/blocks/window/segformer.py) | W → W | {channels 256; height 56; in_channels (256, 160, 64, 32); width 56; window 7} | `project_pyramid_constants` |
| [`fuse_head`](../tensor_ee/blocks/window/segformer.py) | W → W | {channels 256; classes 2; eps 1e-05; height 56; in_channels 1024; width 56; window 7} | `fuse_head_constants` |
| [`swin_mlp_branch`](../tensor_ee/blocks/window/swin.py) | W → W | {dim 96 or 192 or 384 or 768; window 7} ∪ {dim 48 or 96 or 192 or 384; window 8} | — |
| [`swin_block`](../tensor_ee/blocks/window/swin.py) | W → W | {dim 96; heads 3; window 7} ∪ {dim 192; heads 6; window 7} ∪ {dim 384; heads 12; window 7} ∪ {dim 768; heads 24; window 7} ∪ {dim 48; heads 2; window 8} ∪ {dim 96; heads 4; window 8} ∪ {dim 192; heads 8; window 8} ∪ {dim 384; heads 16; window 8} | — |
| [`shifted_swin_block`](../tensor_ee/blocks/window/swin.py) | W → W | {dim 96; heads 3; shift 3 or None; window 7} ∪ {dim 192; heads 6; shift 3 or None; window 7} ∪ {dim 384; heads 12; shift 3 or None; window 7} ∪ {dim 48; heads 2; shift 4 or None; window 8} ∪ {dim 96; heads 4; shift 4 or None; window 8} ∪ {dim 192; heads 8; shift 4 or None; window 8} | — |
| [`swin_stage`](../tensor_ee/blocks/window/swin.py) | W → W | {depth 2; dim 96; heads 3; shift 3 or None; shifted True; window 7} ∪ {depth 2; dim 192; heads 6; shift 3 or None; shifted True; window 7} ∪ {depth 6; dim 384; heads 12; shift 3 or None; shifted True; window 7} ∪ {depth 2; dim 768; heads 24; shift 3 or None; shifted False; window 7} ∪ {depth 2; dim 48; heads 2; shift 4 or None; shifted True; window 8} ∪ {depth 2; dim 96; heads 4; shift 4 or None; shifted True; window 8} ∪ {depth 2; dim 192; heads 8; shift 4 or None; shifted True; window 8} ∪ {depth 2; dim 384; heads 16; shift 4 or None; shifted False; window 8} | — |
| [`swin_patch_embedding`](../tensor_ee/blocks/window/swin.py) | R → W | {channels 2; embed_dim 96; patch 4; window 7} ∪ {channels 3; embed_dim 48; patch 4; window 8} | `SwinUPerNetConfig`, `patch_embed_values` |
| [`swin_encoder`](../tensor_ee/blocks/window/swin.py) | W → W | {cols 8; depths (2, 2, 6, 2); norm False; num_heads (3, 6, 12, 24); rows 8; stage_dims (96, 192, 384, 768); start 1; stop 3; window 7} ∪ {cols 8; depths (2, 2, 6, 2); norm False; num_heads (3, 6, 12, 24); rows 8; stage_dims (96, 192, 384, 768); start 4; stop 4; window 7} ∪ {cols 8; depths (2, 2, 2, 2); norm False; num_heads (2, 4, 8, 16); rows 8; stage_dims (48, 96, 192, 384); start 1; stop 4; window 8} | `SwinUPerNetConfig`, `swin_encoder_grids` |
| [`psp_layer_norm`](../tensor_ee/blocks/window/upernet.py) | W → W | {stage_dims (96, 192, 384, 768); window 7} ∪ {stage_dims (48, 96, 192, 384); window 8} | `SwinUPerNetConfig` |
| [`psp_branch`](../tensor_ee/blocks/window/upernet.py) | V → V | {decoder 256; sizes (1, 2, 3, 6); stage_dims (96, 192, 384, 768); window 7} ∪ {decoder 64; sizes (1, 2, 3, 6); stage_dims (48, 96, 192, 384); window 8} | `SwinUPerNetConfig` |
| [`psp_forward`](../tensor_ee/blocks/window/upernet.py) | W → W | {decoder 256; height 7; sizes (1, 2, 3, 6); stage_dims (96, 192, 384, 768); width 7; window 7} ∪ {decoder 64; height 8; sizes (1, 2, 3, 6); stage_dims (48, 96, 192, 384); width 8; window 8} | `SwinUPerNetConfig` |
| [`fpn_layer_norm`](../tensor_ee/blocks/window/upernet.py) | W → W | {level 1 or 2 or 3; stage_dims (96, 192, 384, 768); window 7} ∪ {level 1 or 2 or 3; stage_dims (48, 96, 192, 384); window 8} | `SwinUPerNetConfig` |
| [`fpn_lateral`](../tensor_ee/blocks/window/upernet.py) | W → W | {decoder 256; level 1 or 2 or 3; stage_dims (96, 192, 384, 768); window 7} ∪ {decoder 64; level 1 or 2 or 3; stage_dims (48, 96, 192, 384); window 8} | `SwinUPerNetConfig` |
| [`fpn_upsample`](../tensor_ee/blocks/window/upernet.py) | W → W | {decoder 256; height 7; width 7; window 7} ∪ {decoder 256; height 14; width 14; window 7} ∪ {decoder 256; height 28; width 28; window 7} ∪ {decoder 64; height 8; width 8; window 8} ∪ {decoder 64; height 16; width 16; window 8} ∪ {decoder 64; height 32; width 32; window 8} | `SwinUPerNetConfig` |
| [`fpn_merge`](../tensor_ee/blocks/window/upernet.py) | W → W | {decoder 256; height 7; width 7; window 7} ∪ {decoder 256; height 14; width 14; window 7} ∪ {decoder 256; height 28; width 28; window 7} ∪ {decoder 64; height 8; width 8; window 8} ∪ {decoder 64; height 16; width 16; window 8} ∪ {decoder 64; height 32; width 32; window 8} | `SwinUPerNetConfig` |
| [`fpn_conv_block`](../tensor_ee/blocks/window/upernet.py) | W → W | {decoder 256; level 1 or 2 or 3; window 7} ∪ {decoder 64; level 1 or 2 or 3; window 8} | `SwinUPerNetConfig` |
| [`fpn_forward`](../tensor_ee/blocks/window/upernet.py) | W → W | {decoder 256; height 7; sizes (1, 2, 3, 6); stage_dims (96, 192, 384, 768); width 7; window 7} ∪ {decoder 64; height 8; sizes (1, 2, 3, 6); stage_dims (48, 96, 192, 384); width 8; window 8} | `SwinUPerNetConfig` |
| [`fusion_resize_to_c1`](../tensor_ee/blocks/window/upernet.py) | W → W | {decoder 256; height 7; width 7; window 7} ∪ {decoder 64; height 8; width 8; window 8} | `SwinUPerNetConfig` |
| [`fusion_resize_level`](../tensor_ee/blocks/window/upernet.py) | W → W | {decoder 256; factor 1; height 56; width 56; window 7} ∪ {decoder 256; factor 2; height 28; width 28; window 7} ∪ {decoder 256; factor 4; height 14; width 14; window 7} ∪ {decoder 256; factor 8; height 7; width 7; window 7} ∪ {decoder 64; factor 1; height 64; width 64; window 8} ∪ {decoder 64; factor 2; height 32; width 32; window 8} ∪ {decoder 64; factor 4; height 16; width 16; window 8} ∪ {decoder 64; factor 8; height 8; width 8; window 8} | `SwinUPerNetConfig` |
| [`fusion_concat`](../tensor_ee/blocks/window/upernet.py) | W → W | {decoder 256; window 7} ∪ {decoder 64; window 8} | `SwinUPerNetConfig` |
| [`fusion_bn_relu`](../tensor_ee/blocks/window/upernet.py) | W → W | {decoder 256; window 7} ∪ {decoder 64; window 8} | `SwinUPerNetConfig` |
| [`fusion_grouped_double`](../tensor_ee/blocks/window/upernet.py) | W → W | {decoder 256; groups 4; window 7} ∪ {decoder 64; groups 4; window 8} | `SwinUPerNetConfig` |
| [`fusion_forward`](../tensor_ee/blocks/window/upernet.py) | W → W | {decoder 256; groups 4; height 7; width 7; window 7} ∪ {decoder 64; groups 4; height 8; width 8; window 8} | `SwinUPerNetConfig` |
| [`head_conv`](../tensor_ee/blocks/window/upernet.py) | W → W | {classes 2; decoder 256; window 7} ∪ {classes 3; decoder 64; window 8} | `SwinUPerNetConfig` |
| [`head_upsample`](../tensor_ee/blocks/window/upernet.py) | W → W | {factor 4; height 56; width 56; window 7} ∪ {factor 4; height 64; width 64; window 8} | `SwinUPerNetConfig` |
| [`head_forward`](../tensor_ee/blocks/window/upernet.py) | W → W | {classes 2; decoder 256; head_factor 4; height 56; width 56; window 7} ∪ {classes 3; decoder 64; head_factor 4; height 64; width 64; window 8} | `SwinUPerNetConfig` |
| [`head_logits`](../tensor_ee/blocks/window/upernet.py) | W → W | {classes 2; decoder 256; head_factor 4; height 56; width 56; window 7} ∪ {classes 3; decoder 64; head_factor 4; height 64; width 64; window 8} | `SwinUPerNetConfig` |
| [`head_dense`](../tensor_ee/blocks/window/upernet.py) | W → R | {classes 2; window 7} ∪ {classes 3; window 8} | `SwinUPerNetConfig` |
<!-- END GENERATED: blocks -->

Each block's `METADATA` also holds two cost functions, `embedded_constants` and
`asset_arrays`: the values a block embeds in a request when its weights are NumPy
arrays, and the server arrays it reads when they are assets. The converter's planner uses
them, with each block's locality, to cut a graph into stages: a batch graph holds at most
24-26 M embedded values ([a research milestone's findings](history/swin_encoder/notes/milestone5_findings.md)),
and the planner's default budget is 24 M per stage ([converter guide](converter.md#3-plan)).

## Layers

The blocks sit in one layer of the library. Each layer imports only the layers before it:

| Layer | Package | Holds |
|---|---|---|
| Core | `tensor_ee.core` | grids, tensor types, the ops interface, precision policies, the metadata schema |
| Layouts | `tensor_ee.layouts` | converters between layouts, halos, stride sampling, shifts |
| Weights | `tensor_ee.weights` | coefficient transport: inline constants and table assets |
| Blocks | `tensor_ee.blocks.raster`, `tensor_ee.blocks.window` | the compute blocks, each with `METADATA` |
| Converter | `tensor_ee.converter` | `compile_model`, BatchNorm folding, tiling, receptive field, ONNX import; the attention-model converter, block graph, builder, planner and runner |
| Models | `tensor_ee.models`, `tensor_ee.swin`, `tensor_ee.segwater_*` | the two reference models and their recipes |

[`tests/test_library_boundaries.py`](../tests/test_library_boundaries.py) enforces the
rule. The [library contract](library-contract.md) defines it, and
[CONTRIBUTING](../CONTRIBUTING.md#layers-and-import-rules) lists what each layer may import.

## Weights

Weights reach Earth Engine in one of two ways
([`tensor_ee.weights`](../tensor_ee/weights/__init__.py)):
- **constants** in the request: NumPy float32 values encoded exactly
  (`inline_float32_array`, `constant_value`). Per-channel constants are repeated to
  the rows of the layout they serve, because Earth Engine arrays do not broadcast,
  and a block refuses constants built for another layout;
- **table assets**: coefficients exported once as Earth Engine FeatureCollection
  assets and read as `ServerArray` values (`FeatureCollectionWeightProvider`). A
  manifest records each record's assets and fingerprints. The two trained models'
  manifests are in [`data/manifests/`](../data/manifests/).

`export_weights` writes the assets, and `MatchedWeightProvider` matches a traced
model's folded coefficients to a manifest by fingerprint. Both are in the converter
layer, because they trace the PyTorch graph.

## Converter

The converter layer has two paths.

**CNNs, onto the R blocks.** `compile_model` traces a PyTorch model with `torch.fx`,
checks every operation against the supported forms, folds evaluation BatchNorm into an
unshared directly preceding convolution, and uses standalone pointwise BatchNorm for
other fixed inference statistics. It builds the graph from the R blocks. `check_compatibility`
reports the same rules locally, without Earth Engine. `analyze_receptive_field`,
`plan_tiles` and `compile_tiled_model` tile large areas with exact halos, and
`load_onnx_model` imports an ONNX model into the same subset. The supported forms and
numerical rules are in the [API guide](api.md).

**Attention models, onto the W blocks and composites**
([converter guide](converter.md)). `tensor_ee.converter.convert.convert` traces with
`torch.export`, maps recognized modules (the SegFormer of segmentation-models-pytorch, SMP;
SMP's UPerNet with timm's Swin) onto the blocks below through an explicit registry, and writes a weight-free
block graph with a conversion report: each node's settings checked against its block's
`validated` record, so a setting outside it is reported, not assumed. The planner cuts
the graph into stages from the blocks' cost functions and locality, the builder builds
a stage with the blocks' own constants builders, and the runner exports one chip's
stages in order. A converted graph is evidence only where it is byte-identical to a
live-validated one: the SegFormer-B0 and Swin-T/UPerNet conversions are (the converter
gate in [CONTRIBUTING](../CONTRIBUTING.md#which-gate-applies)). The hand-composed models stay as they are.

## The two reference models

Both are built only from public library modules, which
[`tests/test_library_boundaries.py`](../tests/test_library_boundaries.py) enforces.

- **SegWater ResNet18 U-Net.** `tensor_ee.segwater_inference.predict_segwater`
  compiles the trained model with `compile_model` on 224 × 224 chips, reads its 31
  folded convolutions ([record of their preservation](evidence/t05-preservation.json)) from table assets
  through `MatchedWeightProvider`, and applies the SegWater preprocessing and
  water-probability recipe ([model notes](model.md), [walkthrough](water-chip.md)).
- **Swin-T/UPerNet, fixed 224 × 224.** The encoder and decoder composites, with their
  timm and SMP parameter names, are window blocks (`tensor_ee/blocks/window/swin.py` and
  `upernet.py`, with `swin_upernet_config.py`). `tensor_ee.models.swin_t224` holds
  the fixed Swin-T224 configuration and re-exports them. The adapter
  `tensor_ee.swin.SwinT224Adapter` runs them as three saved-asset stages, A, B and C
  ([Swin guide](swin.md)).

The [model notes](model.md) list the evidence that validates each model, with SegFormer-B0
and a new 256 × 256 Swin-UPerNet, which are also built only from the blocks.

## Public functions without `METADATA`

The core, layouts, weights and converter layers hold types, converters and runners,
not blocks, so they have no `METADATA`; nor does a block package's settings module
(`swin_upernet_config`, the Swin-UPerNet configuration), listed here as "Blocks
(settings)", so every module of the block packages is in one table or the other. The
tables below are generated from the
package's public names: a module's `__all__` when it defines one, otherwise the
public functions and classes it defines. They are generated rather than copied from
[contract §10](library-contract.md#10-public-api-compatibility-phase-1) because §10 is
the refactor's compatibility list of older module paths, not the layer API, and a
generated list cannot drift from the code.

<!-- BEGIN GENERATED: api (tools/block_catalogue.py; edit the code, then rerun) -->
| Layer | Module | What it holds | Public names |
|---|---|---|---|
| Core | [`tensor_ee.core.errors`](../tensor_ee/core/errors.py) | Errors shared by every layer of the library. | `UnsupportedOperation`, `GridMismatch`, `SemanticUnknown`, `reject` |
| Core | [`tensor_ee.core.grid`](../tensor_ee/core/grid.py) | The three grid kinds behind one protocol, with explicit conversions. | `GridProtocol`, `Grid`, `TokenGrid`, `FeatureGrid`, `to_grid`, `to_feature_grid`, `to_token_grid`, `check_model_grid` |
| Core | [`tensor_ee.core.metadata`](../tensor_ee/core/metadata.py) | Locality, cost and validated-settings metadata for library blocks (docs/library-contract.md, section 9, and its "As built (Phase 3)" section). | `Locality`, `Between`, `Validated`, `SettingsCheck`, `check_settings`, `describe_validated`, `BlockMetadata`, `validate_metadata`, `metadata_at` |
| Core | [`tensor_ee.core.ops`](../tensor_ee/core/ops.py) | The ops interface: one set of array primitives, two implementations. | `Ops`, `image_reshape`, `ArrayOps`, `ImageOps`, `BlockArrayOps`, `BlockImageOps`, `ValueOps`, `WindowOps` |
| Core | [`tensor_ee.core.precision`](../tensor_ee/core/precision.py) | Precision policies, and how float32 values enter an Earth Engine graph. | `validate_policy`, `exact_scalar`, `encode_float32`, `constant_value` |
| Core | [`tensor_ee.core.tensor`](../tensor_ee/core/tensor.py) | Tensor carriers: R (`Raster`) and W (`WindowTensor`), plus `Feature`. | `band_names`, `ServerArray`, `Raster`, `WindowTensor`, `Feature`, `same_grid` |
| Layouts | [`tensor_ee.layouts.boundaries`](../tensor_ee/layouts/boundaries.py) | Stage boundaries: how saved images enter a stage's graph and how its outputs leave it. | `grid_view`, `pin_boundary`, `saved_image`, `boundary_windows`, `boundary_raster`, `relabel_to_windows`, `relabel_from_windows` |
| Layouts | [`tensor_ee.layouts.raster`](../tensor_ee/layouts/raster.py) | R-layout converters: pinning, zero extension, stride sampling, embedded fixtures. | `project`, `finish`, `zero_extended`, `stride_sample`, `from_image`, `stride_sample_phase`, `embedded_raster` |
| Layouts | [`tensor_ee.layouts.shift`](../tensor_ee/layouts/shift.py) | W-layout translation, Swin's cyclic shift, periodic extension and domain masks. | `timm_shift`, `projection`, `translate`, `materialize`, `cyclic_neighbor`, `roll_axis`, `roll_windows`, `shift_windows`, `unshift_windows`, `periodic_extension`, `domain_mask` |
| Layouts | [`tensor_ee.layouts.window`](../tensor_ee/layouts/window.py) | R <-> W converters, halos and im2col. | `as_row`, `flatten_row`, `stride_samples`, `stack_tokens`, `to_windows`, `from_dense`, `depth_to_space`, `to_dense`, `zeros`, `reshape`, `zero_outside`, `neighbours`, `halo`, `taps`, `im2col` |
| Weights | [`tensor_ee.weights.assets`](../tensor_ee/weights/assets.py) | The FeatureCollection weight-asset format and its provider. | `ConvolutionWeights`, `convolution_weights`, `feature_records`, `FeatureCollectionWeightProvider` |
| Weights | [`tensor_ee.weights.constants`](../tensor_ee/weights/constants.py) | Coefficient transport: how weights and per-channel constants enter a graph. | `inline_float32_array`, `RowMismatch`, `WindowLayout`, `ValueLayout`, `layout_of`, `constant_rows`, `RowConstant`, `row_constant`, `check_rows` |
| Blocks (settings) | [`tensor_ee.blocks.window.swin_upernet_config`](../tensor_ee/blocks/window/swin_upernet_config.py) | Swin-UPerNet settings: `SwinUPerNetConfig`, the UPerNet decoder contract and the Swin rules that the `swin` and `upernet` composites read. | `SwinUPerNetConfig`, `check_shapes`, `stage_shifted`, `bn_args`, `level_tensors`, `block_index`, `lateral_prefix`, `conv_prefix`, `norm_prefix` |

**Converter** (`tensor_ee.converter`, a re-export namespace; each name is the same object as at its own module):

| Name | Defined in |
|---|---|
| `compile_model` | [`tensor_ee.compiler`](../tensor_ee/compiler.py) |
| `batchnorm_predecessor` | [`tensor_ee.semantics`](../tensor_ee/semantics.py) |
| `PixelWindow` | [`tensor_ee.tiling`](../tensor_ee/tiling.py) |
| `plan_tiles` | [`tensor_ee.tiling`](../tensor_ee/tiling.py) |
| `compile_tiled_model` | [`tensor_ee.tiling`](../tensor_ee/tiling.py) |
| `analyze_receptive_field` | [`tensor_ee.receptive_field`](../tensor_ee/receptive_field.py) |
| `effective_convolutions` | [`tensor_ee.weight_assets`](../tensor_ee/weight_assets.py) |
| `MatchedWeightProvider` | [`tensor_ee.weight_mapping`](../tensor_ee/weight_mapping.py) |
| `load_onnx_model` | [`tensor_ee.onnx_import`](../tensor_ee/onnx_import.py) |
| `check_compatibility` | [`tensor_ee.compatibility`](../tensor_ee/compatibility.py) |
| `Conversion` | [`tensor_ee.converter.convert`](../tensor_ee/converter/convert.py) |
| `ConversionReport` | [`tensor_ee.converter.convert`](../tensor_ee/converter/convert.py) |
| `NodeCheck` | [`tensor_ee.converter.convert`](../tensor_ee/converter/convert.py) |
| `BlockGraph` | [`tensor_ee.converter.graph`](../tensor_ee/converter/graph.py) |
| `Edge` | [`tensor_ee.converter.graph`](../tensor_ee/converter/graph.py) |
| `Node` | [`tensor_ee.converter.graph`](../tensor_ee/converter/graph.py) |
| `Stage` | [`tensor_ee.converter.graph`](../tensor_ee/converter/graph.py) |
| `build_stage` | [`tensor_ee.converter.build`](../tensor_ee/converter/build.py) |
| `Limits` | [`tensor_ee.converter.plan`](../tensor_ee/converter/plan.py) |
| `Plan` | [`tensor_ee.converter.plan`](../tensor_ee/converter/plan.py) |
| `PlannedStage` | [`tensor_ee.converter.plan`](../tensor_ee/converter/plan.py) |
| `request_size` | [`tensor_ee.converter.request`](../tensor_ee/converter/request.py) |

**`tensor_ee`** (`__all__`, 31 names): `PixelTensor`, `compare_segmentation`, `CompatibilityReport`, `check_compatibility`, `CompiledModel`, `Grid`, `Raster`, `UnsupportedOperation`, `WeightProvider`, `compile_model`, `from_image`, `conv2d`, `pool2d`, `nearest_upsample`, `concatenate`, `residual_add`, `relu`, `sigmoid`, `leaky_relu`, `silu`, `gelu`, `tanh`, `eval_dropout`, `softmax`, `SegmentationOutputs`, `segmentation_outputs`, `binary_threshold`, `FeatureCollectionWeightProvider`, `MatchedWeightProvider`, `effective_convolutions`, `export_weights`.

**`tensor_ee.swin`** (`__all__`, 5 names): `ServerArray`, `SwinTParameters`, `load_swin_t224_parameters`, `load_swin_t224_model`, `SwinT224Adapter`.
<!-- END GENERATED: api -->

## No longer in the library

The refactor kept one default per block: the variant that production uses. The research
variants were removed when the research code was archived, at the private development
archive's tag `research-archive-2026-10-12` ([history](history/README.md)):

- the other 3 × 3 convolution methods (`window_taps` and the dense forms);
- the `monolithic` and `grouped_rounded` fusion formulations;
- the other resize paths (`dense_resize`, `dense_absolute_resize`, `matrix_resize`,
  `native_resize`) and pooling paths (`global_pool`, `dense_pool`);
- the attention variants (blockwise, neighbourhood, window-partition and
  partition-slice);
- the `per_stage` and `propagated` periodic-extension modes, the neighbourhood
  patch-embedding gather and the fine-grid patch-merge gather;
- the whole-area PSP ("Option C"), which a later change may add back as an optional mode.

Inventory §6 lists the production choice for each
([snapshot](history/building-blocks-2026-10-11.md#6-production-choices-and-research-variants)),
and the [milestone notes](history/README.md#what-is-here) record how each was measured.
