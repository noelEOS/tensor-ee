# Public API and numerical behavior

Import the generic CNN API from `tensor_ee`. Compilation, primitives and output
helpers construct deferred graphs. The compatibility report and exact-pixel
comparison operate locally. Initialize EE in your application before graph
construction; initialization, retrieval, uploads and exports are caller actions.
The explicitly named weight uploader starts tasks. To build graphs with no credentials,
for example to inspect them or to test, install the Earth Engine client's offline function
signatures with its test helper `ee.apitestcase.ApiTestCase` instead of initializing; this
works from an installed library, and
[Getting started](getting-started.md#2-run-the-offline-example) shows it. The helper is not
a documented client API, so it may change between client versions.

| Entry point | Purpose |
| --- | --- |
| `Grid(crs, transform, width, height)` | North-up finite grid; origin is the upper-left pixel corner |
| `Raster(image, bands, grid)` | EE image with explicitly ordered channel/grid metadata |
| `from_image(image, bands, grid)` | Select ordered channels, zero-fill model values, cast float32 |
| `conv2d(raster, module, parameters=None)` | Supported `nn.Conv2d`, optionally supplied EE weight/bias arrays |
| `pool2d(raster, module)` | Supported average/max pooling |
| `nearest_upsample(raster, factor)` | Repeat pixels by an equal positive integer factor |
| `concatenate(rasters, dim=1)`, `residual_add(left, right)` | Channel concatenation and tensor addition on identical grids |
| `relu`, `sigmoid`, `leaky_relu`, `silu`, `gelu`, `tanh` | Pointwise float32 activations |
| `eval_dropout(raster, p=...)` | Validated evaluation identity |
| `softmax(raster)` | Channel normalization with common finite score masks |
| `compile_model(model, image, input_bands, grid, ...)` | Translate supported eval-mode FX operations |
| `check_compatibility(model, input_shape, input_dtype=torch.float32)` | Local schema-versioned semantic/shape facts |
| `segmentation_outputs(logits, ...)` | Raw logits, float32 probabilities and int32 ordered classes |
| `binary_threshold(probabilities, ...)` | Inclusive uint8 binary class threshold |
| `PixelTensor`, `compare_segmentation` | Local retrieved-array/reference comparison |
| `UnsupportedOperation` | Unsupported model semantics; subclass of `ValueError` |

`CompiledModel` retains `Raster` fields plus `nodes`, `graph_module` and
`folded_batchnorms`. CNN intermediate channels are `c0`, `c1`, … in positional
order. No class identity is inferred by sorting band names.

## Spatial and numerical rules

Tensors are batch-one NCHW: channels map to bands, row/column to H/W, with rows
downward. Generic grids allow unequal x/y spacing, require +x/−y and no shear,
and bound every intermediate tensor. Input names must be unique and nonempty.
A Raster asserts its image's alignment and channel order; it does not inspect
remote assets. Known local metadata conflicts are rejected.

Prepare finite float32 inputs in the model's units before compilation. Capture
source validity before filling missing model-domain values. `from_image` and
compiler input selection fill missing values with zero inside the finite grid;
this computational fill does not establish output validity. Numeric zero or an
ordinary source value is never a nodata sentinel. Convolution zero-extends every
preceding finite tensor, including after biased layers. Max-pool padding instead
excludes exterior values, corresponding to negative infinity in PyTorch. A padded
average pool follows `count_include_pad`: `True` divides the in-bounds sum by the
full window (9), as PyTorch's zero padding does; `False` divides by the number of
in-bounds pixels.

Convolution uses odd same-padded kernels, stride 1/2, dilation 1 and zero padding
mode. Groups=1 retains the existing `conv2d` block. Groups greater than one route
to `grouped_conv2d`, including depthwise convolutions and channel multipliers;
groups must divide both channel counts. The grouped block is validated live for
depthwise 3 x 3 at strides 1/2 with optional bias, depthwise 5 x 5 at stride 1, and
3 x 3 at stride 1 with 2 channels per group or a channel multiplier of 2. Its products and reduction run in double,
then the compiler materializes float32. Grouped asset weights retain PyTorch's
`[out, in_per_group * kh * kw]` layout; unshared Conv→BN folding preserves groups.
Stride-two sampling preserves PyTorch's phase before relabeling
the output grid. Pooling (`nn.AvgPool2d`, `nn.MaxPool2d`) supports aligned
average or max pooling with kernel = stride = k and padding 0, for k = 2, 3 or 4,
on sizes divisible by k; 3×3 max pooling with padding 1 and stride 1 or 2; and
3×3 average pooling with stride 1 and padding 1, under either `count_include_pad`.
Pooling requires `ceil_mode=False`, max-pool `dilation=1` and average-pool
`divisor_override=None`; every other configuration is rejected. Nearest factors
are equal positive integers; factors 2, 3 and 4 are validated against PyTorch.
Bilinear `F.interpolate` and `nn.Upsample` (including `nn.UpsamplingBilinear2d`)
accept equal positive integer factors or a proven static size at such a factor,
with a static `align_corners` bool and `antialias=False`. `recompute_scale_factor`
may be None or a static bool for scale-based calls; size-based calls reject True,
as PyTorch does. Downsampling, non-integer or unequal factors, antialias and other
interpolation modes are refused. Coefficients come from PyTorch's existing kernel
reader with explicit float32 model dtype under both alignment rules; the block
guards their exact float32 representation.
The raster bilinear block is validated live at factors 2 and 4 under both alignment
rules and factor 3 with align_corners=False.
Constants stay inline and the output rounds once
to float32. `align_corners=True` depends on the whole source extent; its receptive
field reports local bounds but refuses a halo guarantee for tiled execution.
Merges require identical spatial grids, even when dimensions
happen to match; residual addition rejects broadcasting.

Model and submodules must already be in eval mode. Computational weights,
biases and BatchNorm statistics are finite float32 with declared shapes;
nonfloating bookkeeping counters are allowed. BatchNorm folds into an unshared
directly preceding convolution. Other inference BatchNorm2d calls use the pointwise
raster block with inline statistics, including after concatenations and shared
convolutions. Functional `F.batch_norm` accepts fixed `get_attr` buffers/parameters
for its statistics and optional affine coefficients, with literal `training=False`;
other attribute uses remain unsupported. Standalone BatchNorm is validated live at eps 1e-5.
Float32 accumulation/folding can change rounding;
compare against the original model on your actual prepared inputs. Public
primitives and compiler activations materialize float32. `clip_footprints=True`
is the existing optional clipping experiment with the same finite padding rules.

Input preparation for models trained on scaled, normalized imagery uses
`tensor_ee.blocks.raster.standardize`: per band `(x * scale - mean) / std`, each
operation rounded to float32 (`round_each_op`), so it reproduces a float32 pipeline
such as `x * 1e-4` followed by torchvision's `normalize` bit for bit. Zero-fill missing
pixels first (`from_image` does). It is not called by `compile_model`. Validated live
with six bands at scale 1e-4 and three bands without a scale.

Local convolution coefficients at or above +2⁶³ or below −2⁶³ use parsed EE
numbers inside the existing array constructor, avoiding observed integral-literal
clipping. Ordinary coefficients and asset-provider arrays retain their existing
graphs. Signed boundary neighbors, large weights and large biases were verified
with exact decoded float32 bits in [the T05 evidence](evidence/t05-primary.json).
This does not establish universal numerical bounds or audit arbitrary external
array constructors. Asset providers assert exact contents and float32 read paths.

## Exact operator forms

LeakyReLU accepts static finite slopes representable in float32, including zero
and negative values. SiLU and exact GELU are supported; GELU requires
`approximate="none"`. Dropout/Dropout2d modules are identity only in effective
eval mode. Functional dropout requires literal `training=False`, including at
p=0; enclosing `.eval()` does not override its default `training=True`.

Channel softmax accepts `nn.Softmax`, `F.softmax`, `torch.softmax` and Tensor
`.softmax` with dim 1 or −3 and dtype=None. It rejects implicit/other axes. The
third positional `F.softmax` argument is `_stacklevel`, whose only supported
value is the static default 3; its fourth argument is dtype. Tensor Tanh and
softmax are the new advertised method forms. Ordinary positional/keyword/default
binding is checked. In-place, `out=`, dynamic/unknown/extra options, tanh GELU
approximation and unsupported dropout families are rejected. Malformed Python
calls may raise `TypeError`; unsupported traced calls identify their FX node.

## Compatibility facts

```python
from tensor_ee import check_compatibility

report = check_compatibility(model, (1, 4, 12, 16))
facts = report.to_dict()
```

The schema version is 1. `supported` requires complete local semantics and
resolved output shape; `unsupported` takes precedence if any known blocker
exists; otherwise unresolved tracing/shape/data facts are `unknown`. Static
options and independent branches remain diagnosed after an unsupported node;
dependent shapes are null. Trace failures retain known input/eval/dtype/parameter
facts. No hidden `.eval()`/`.float()` coercion or input-shaped dummy tensor is
used. Ordinary FX tracing invokes user Python; this is not a sandbox for forwards.

Unique parameter counts exclude buffers. Storage counts effective Conv/BN
coefficients per FX invocation, omitting a diagnostic Conv replaced by folded BN;
repeated calls to one module count separately. The float32 byte estimate is four
times known coefficients, with incomplete/null states explicit. Encoded request
size, asset size, EE capacity, latency and cost are unmeasured. Every report's
`execution_evidence` is `not_measured`, independently of successful local tracing.

## Segmentation outputs and validity

```python
outputs = segmentation_outputs(
    logits, logit_bands=("score_a", "score_b", "score_c"),
    probability_bands=("p_a", "p_b", "p_c"), validity=output_validity,
)
```

At least two ordered classes are required. Raw logits retain their values and
computational masks, with optional renaming. Default probability names are
`probability_0..K-1`; default class name is `class_index`. Processed probabilities
and classes intersect all score-channel masks, finite float32 scores, the finite
grid, and optional one-band validity on that same grid. Only unmasked exact 1 is
valid; zero or masked validity excludes the pixel. Locally inspectable invalid
validity is rejected; computed/remote values remain caller preconditions.
Invalid outputs remain masked rather than becoming a class-zero decision.

Softmax subtracts the maximum, then does exp/sum/divide in double and materializes
float32. Classes use int32 **raw-logit argmax** before probability rounding, with
the lowest index winning an exact tie. `[0,1e-8]` can yield rounded [.5,.5]
probabilities while raw class 1 still wins.

`binary_threshold` requires exactly two probability channels, positive class 0
or 1, and a finite real threshold in [0,1]. It returns uint8 using inclusive >=.
Existing float32 probabilities widen losslessly to binary64; the supplied scalar
retains binary64 precision. Probability .5 is below threshold .50000001. Equal
two-class logits give argmax 0 but a class-1 threshold of 1 at .5. Existing water
recipes retain their own arithmetic, validity and float32 export bands.

## Compare exact retrieved pixels

`PixelTensor(values, mask, bands, grid)` holds local CHW arrays and independent
boolean CHW masks. It validates dimensions, ordered names and finite unmasked
values; masked NaN/Inf payloads remain excluded. It does not retrieve pixels or
infer metadata from an EE asset. For prepared inputs, logits and probabilities,
comparison requires float32; classes require one-band int32.

```python
report = compare_segmentation(
    model, prepared_pixels, logit_pixels, probability_pixels, class_pixels,
    expected_prepared=expected_prepared_pixels,
    expected_logit_mask=expected_raw_score_masks,   # boolean K×H×W
    expected_output_mask=expected_valid_pixels,    # boolean H×W
    pixel_coordinates=(pixel_x, pixel_y),          # input-grid indices
    absolute_tolerance=1e-4,
)
```

The model must already be on CPU in eval mode with float32 floating tensors.
It receives a copy of the actual retrieved dense prepared tensor, after exact
IEEE-bit comparison with the declared input. Pixel coordinates verify input row/
column orientation; remote band/grid/type checks are the caller's responsibility.
Logit/probability/class output grids must agree. Recipe source validity and
preprocessing checks stay with the caller, as demonstrated by the installed
[nonwater example](../tensor_ee/examples/nonwater.py).

Independent expected masks define numerical denominators. Missing observed
output pixels therefore fail rather than disappearing from the comparison.
The JSON-safe report records absolute/MAE/RMSE errors, relative diagnostics with
floor 1e-3, finite probability range/sums, exact class disagreements, input hashes
and near-margin locations. Relative diagnostics do not replace absolute gates.
An optional one-band uint8 `binary_mask`, `positive_class` and `threshold` use the
same binary64 >= rule and report near-threshold pixels. A pass is numerical
agreement on the supplied pixels, not accuracy or a universal error guarantee.

## Weight assets and fixed Swin

`effective_convolutions(model)` extracts validated effective float32 weights.
`export_weights(parameters, asset_namespace, manifest_path)` starts table export
tasks and waits for completion. Use a new namespace/manifest. `MatchedWeightProvider`
verifies local coefficient fingerprints and maps FX keys, including folded BN,
to the manifest. Use `include_unfused_nodes=False` to omit diagnostics. Matching
local identities does not independently checksum a remote asset. Asset reads can
avoid large embedded requests but do not remove EE computation limits.

The fixed optional [Swin-T/UPerNet interface](swin.md) is explicitly imported from
`tensor_ee.swin`; it is separate from the CNN compiler. Its installed orchestration
entry point preserves three saved-asset stages and explicit resources. The explicit
Prithvi crop interface below adds one hand-composed architecture. ONNX import
(`load_onnx_model`) is in the
converter layer (`tensor_ee.converter`), not in the top-level API; the
[block catalogue](building-blocks.md) lists the layers' public names.

Tiling (`plan_tiles`, `compile_tiled_model`) is also in `tensor_ee.converter`.
The [own-model guide](own-model.md) takes a trained SMP CNN through adaptation,
folded-weight upload, aligned context windows, tiled compilation and export.
It explains how a padded input domain keeps every context at the adapter's
fixed shape, and includes a runnable offline graph-construction example.

The historical
research runners are archived at the tag `research-archive-2026-10-12`
([history](history/README.md)).

## Class rows beside spatial windows

`tensor_ee.blocks.window.merge.concat_rows(ops, parts, *, rows, channels)`
concatenates raw array operands along the token axis without arithmetic or casts.
The caller supplies each part's positive row count and a shared channel count;
all parts must already represent the same crop pixel. The result is an
intermediate matrix, so its row count need not be a square window size.
Empty parts and invalid dimensions are refused. No coefficients are embedded or
read from assets. It is validated live.

A class token can therefore remain a one-row array beside a spatial window.
Concatenate class then spatial keys and values, and call the existing
`naive_attention` twice per head: once for the spatial queries and once for the
class query. Both calls read all keys and values. The class row must pass through
every normalization, projection and residual update.

`tensor_ee.blocks.window.attention.class_token_attention(ops, spatial_qkv,
cls_qkv, dim=..., heads=..., window=14)` provides this composition for every
head. Its settings are keyword-only. It takes projected QKV arrays, returns
the spatial and class outputs separately, and applies post-product score
scaling through the existing attention core. No coefficient constants are
introduced by the composite. It is validated live at window 3 with 3 heads of width 4
and at window 14 with 12 heads of width 64.

## Window depth-to-space

`tensor_ee.blocks.window.resample.window_depth_to_space(ee, x, grid, channels,
window=7, out_window=7, factor=2)` moves a projected window's position-major
columns `(dy, dx, channel)` to the corresponding fine-grid tokens. It returns
the array image and its fine token grid. The settings are keyword-only.

The supported window pairs are 14→7 and 7→7, at factor two, on a `TokenGrid`
with complete windows. Each child window selects its rows and columns directly
from the expanded parent array. Tensor values undergo no arithmetic or casts;
no dense channel raster is formed. The block has window locality, needs no
halo, and has no chip-edge dependence. It is validated live for both window pairs.
This supplies the value-moving part of a kernel-two, stride-two transposed
convolution after an ordinary linear projection.


## Patch embeddings without normalization

The window patch embedding and its constants builder take keyword-only
`norm=True`. This default preserves the normalized embedding and its existing
graphs. Set `norm=False` on both `patch_embed_constants` and `patch_embed` for a
biased patch projection with no following LayerNorm. The constants builder
then reads only projection weights and bias; the cost functions take the same
flag. A mismatched flag or non-boolean flag is refused. Input columns retain
Conv2d/one-frame Conv3d order `(band, ky, kx)`.

The `prithvi_settings` fixture selection covers patch16 with six bands and
768 outputs, projection widths 768/2304/3072, channel normalization at 768,
the bias-free 768-to-256 decoder convolution with BatchNorm, and the two-class
classifier. Compact random dyadic tiles are expanded identically on the server
and in the references, with every output channel read. These structured inputs
keep the unchanged request budget; they do not establish behavior with an
arbitrary dense coefficient matrix or server runtime memory. Every fixture
passed live, with Earth Engine equal to the NumPy mirror in every value. The
records of `linear`, `conv1x1`, `layer_norm2d` and `conv3x3_window` already
cover these settings. The patch embedding's record now names `norm`: `True` at
its historical settings, and `False` at patch 16 with six bands, width 768 and
window 14. Its metadata radius, and that of `stride_columns`, is 8, half the
largest validated patch.


## Prithvi Sen1Floods11

`tensor_ee.models.prithvi_sen1floods11` composes one 224 × 224 crop with six
already scaled and normalized bands into two logit bands. Import it explicitly;
`import tensor_ee` does not load it. It uses the window blocks directly,
independently of the converter and CNN compiler.

```python
from tensor_ee.models.prithvi_sen1floods11.model import constants, forward

coefficients = constants(ee, state)
logits = forward(ee, input_raster, coefficients)["logits"]
```

`state` supplies NumPy float32 arrays or `ServerArray` coefficients under the
source state names. `parameter_shapes()` declares the inventory. NumPy values
use native tensor shapes; server arrays also accept the existing asset format's
flattened matrices, and vector rows or columns. All constants are prepared
before composition. Position values are supplied, never generated here.
The unused auxiliary head and BatchNorm counter are ignored.
State names follow the source model's module hierarchy; they have not been checked against a released checkpoint.

The default configuration is width 768, 12 heads, 12 blocks, MLP width 3072,
and decoder width 256. `PrithviConfig` also supplies a reduced twin for the
offline comparisons. Post-product attention scaling follows the historical
source order. The class row remains a query, key and value through every
encoder block, alongside 196 spatial rows in W14. Four linear projections
plus exact depth-to-space moves form the neck, using W7 after the first
upsampling. The decoder's 3 × 3 convolution has no bias; BatchNorm epsilon is
1e-5. Neck channel normalization uses epsilon 1e-6. Dropout and same-size final
bilinear resize are evaluation identities.

The input must already be finite float32 with every pixel valid, in the
Raster's declared band order. The interface checks shape and band count;
it cannot inspect remote values or establish mask completeness. Scaling and
per-band normalization stay outside this model. An input of 224 R × 224 C
pixels is a region of R × C whole crops, anchored at its top-left pixel
(`region_grid(R, C)`). Each crop is computed as if it ran alone: the decoder's
3 × 3 convolution pads every crop edge with zeros (`conv3x3_window`'s `crop`,
used only when there is more than one crop, so one crop builds the same graph as
before). Slide inference, with overlapping crops and averaged logits, is
outside this interface.

`compose` supports explicit saved boundaries: spatial/class pairs after the
patch setup, each encoder block and final norm, then a spatial feature after
each upsampling. Both parts are required when resuming the encoder. The
`planning` module uses existing cost functions, the planner's `Limits`, and
`request_size` for explicit hand cuts. It does not claim an optimized converter
plan. `tools/prithvi_capacity.py` reproduces its offline request measurements.

Capacity estimates for one default crop are 97,602,818 float state values,
122,783,330 logical evaluated constants, and 145.795 billion MACs. The
bias-free decoder convolution accounts for about 60.897% of those MACs;
that fraction alone does not establish a need for grouped execution. The
largest image array is [196,3072], or 602,112 values. The proposed 19 stages
fit default constant and boundary limits; request sizes depend on the table
manifest. Symbolic table sizing does not establish server memory or runtime.

A reduced-width random-weight crop (width 24, 3 heads, 12 blocks, decoder 16)
was checked live: Earth Engine equals the NumPy mirror at every probed logit,
within 2.64e-07 of PyTorch. This checks the composition, not trained weights
or accuracy.

The released Sen1Floods11 fine-tune also runs at full width in Earth Engine.
`tools/prithvi_trained_run.py` serves its 97,602,818 values as 217 table assets
and exports each saved boundary of `planning.stage_rasters` in turn, 34 exports
in 19 stages, each stage reading only its predecessor's assets. On one synthetic
crop all 50,176 pixels' logits match the plain-torch reference within
4.11e-06 (target 1e-4), with 0 argmax disagreements; the run took 926 batch
EECU-seconds and 35 minutes ([record](evidence/prithvi_sen1floods11-trained-live.json)).
`plan` measures every export request offline first. The input is synthetic:
this checks the trained arithmetic, not flood-mapping accuracy.

`--cuts` exports fewer boundaries, so one stage covers several blocks;
`--start` and `--source-namespace` begin a run from another run's saved
boundary, and `identity` compares each saved image with another run's.
Request size does not bound the merge: Earth Engine refused stages of 4, 6
and 12 encoder blocks ("Encoded object too large") at 0.54 to 1.36 MB, while
2 blocks and the merged decoder ran. With
`--cuts patch,block1,block3,block5,block7,block9,block11,upsample56,logits`
the crop runs in 16 exports instead of 33 and 751 batch EECU-seconds instead
of 925, every saved image equal to the 19-stage run's, and the same logits
([record](evidence/prithvi_sen1floods11-fewer-stages-live.json)). Its wall
time, 35 minutes, did not fall: two stages waited 8 to 9 minutes in the batch
queue.

`--crops RxC` runs a region through the same stages, so the queue time is shared
by its crops. Its input is assembled first: each crop's seeded input is exported
on its own (`--input-seeds`; `--reuse-input` reads one crop from an existing
input asset), then one export places them side by side. Every region export sets
`shardSize` to one crop's extent on its grid (`--shard-crops K` for K × K).
`compare` checks each crop against the plain-torch reference on that crop alone
and reports the largest absolute error per crop and on the seam ring, the pixels
next to an interior crop edge; `identity --crop I,J` compares crop (I, J) with a
single-crop run. A region of 2 × 2 crops with the 2-block cut list ran live in
20 exports in 11 sequential stages, with the trained weights and synthetic
input. Crop (0, 0) was bit-identical to the single-crop run at all 16 saved
images, and every crop's logits match the plain-torch reference within
6.11e-06, 0 argmax flips, 2.50e-06 on the seam ring
([record](evidence/prithvi_sen1floods11-region-live.json)). Its stages took 54
minutes for the 4 crops, 13 minutes a crop, against 35 for one crop alone, and
3,154 batch EECU-seconds, against 751. The encoder stages cost about twice one
crop's EECU. The logits stage took 30 of the 54 minutes and 2,463 EECU-seconds,
but repeats of that stage cost 1,627 to 1,776 for the 4 crops and 413 to 673
for one crop alone: it scales about linearly with the crops, and its cost varies
by about a quarter between identical runs. One-crop shards finished it in 12
minutes, one shard for the whole region in 19
([probes](evidence/prithvi_sen1floods11-logits-probes-live.json)).

The plain-torch reference uses native convolutions and all 197 attention rows.
The numerical executor runs the actual library backend and arithmetic, with
independently checked NumPy spatial transports. Default tests compare every
logit of a reduced-width 224 crop. A separate opt-in checks full width. The
`prithvi_model` harness selection probes every fourth output pixel of the
complete reduced-width graph with structured random coefficients. It is a
composition check that can be planned and run independently of block validation.
The `prithvi_region` selection runs the same reduced model on 2 × 2 crops and
compares each crop with PyTorch on that crop alone, probing both sides of every
crop seam (3 requests; not yet run live).
No accuracy claim is made.
