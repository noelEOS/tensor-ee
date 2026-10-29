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
entry point preserves three saved-asset stages and explicit resources. Other
architectures and arbitrary transformers are not supported. ONNX import
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
read from assets. Its status remains `Fixed — offline only` pending promotion.

A class token can therefore remain a one-row array beside a spatial window.
Concatenate class then spatial keys and values, and call the existing
`naive_attention` twice per head: once for the spatial queries and once for the
class query. Both calls read all keys and values. The class row must pass through
every normalization, projection and residual update. For post-product scaling,
use `score_scale`; this preserves the existing attention policy.

## Window depth-to-space

`tensor_ee.blocks.window.resample.window_depth_to_space(ee, x, grid, channels,
window=7, out_window=7, factor=2)` moves a projected window's position-major
columns `(dy, dx, channel)` to the corresponding fine-grid tokens. It returns
the array image and its fine token grid. The settings are keyword-only.

The supported window pairs are 14→7 and 7→7, at factor two, on a `TokenGrid`
with complete windows. Each child window selects its rows and columns directly
from the expanded parent array. Tensor values undergo no arithmetic or casts;
no dense channel raster is formed. The block has window locality, needs no
halo, and has no chip-edge dependence. Its status remains
`Fixed — offline only: factor 2; windows 7, 14 to window 7` pending promotion.
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
arbitrary dense coefficient matrix or server runtime memory. Existing statuses
and validated records are unchanged. The preserved historical record does not
encode `norm`: `check_settings` alone therefore cannot distinguish its values
at legacy shapes. Historical evidence requires `norm=True`; `norm=False` has
only the new offline evidence. Prithvi settings are outside the historical
record on patch size, band count, width and window.
