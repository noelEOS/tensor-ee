# ViT-B/16 + DPT target at 224

`tensor_ee.models.vit_b16_dpt224` is the hand-composed target for the
[SMP DPT recognizer](converter.md#smp-050-dpt-at-224--224). It reuses
`vit_b16_encoder224` and existing library blocks and is not loaded by
`import tensor_ee`. The [converter gate](../tests/test_smp_dpt_converter.py)
checks byte-identical graphs at full and reduced width: graph identity only,
separate from the offline and reduced live numerical evidence below.
Weights are native NumPy float32 arrays or native/flattened `ServerArray`s,
using SMP 0.5.0's `state_dict` names. Constructors build graphs only.

The target is SMP 0.5.0
`DPT(encoder_name="tu-vit_base_patch16_224", encoder_weights=None,
in_channels=6, classes=2)`, with its default cat readout, intermediate widths
`(256, 512, 1024, 1024)` and fusion width 256. The untagged encoder name selects
the same architecture as SMP's default pretrained-tagged name; no pretrained
weights are loaded. The reference is evaluation mode with exact GELU and timm's
unfused attention. Inference dropout and output activation are identities.

The caller supplies one fully valid, finite, already prepared six-band 224 × 224
float32 crop in declared band order. No sensor normalization, masking strategy,
trained checkpoint, crop blending, or model-accuracy claim is included. The
model returns dense `c0,c1` logits on its abstract input grid; georeferenced
relabeling belongs to a caller's recipe.

## Features and layouts

SMP samples four zero-based block indices uniformly: `(2, 5, 8, 11)` for twelve
blocks. These are raw block outputs (`norm=False`), without the final encoder
norm. The class token remains timm sequence row 0. The merged encoder returns
each feature as `(spatial W14, class W1)` on matching physical window grids.
The unused encoder classifier and BatchNorm counters may be present in a state
mapping and are ignored; unsupported names/shapes are refused.

| Stage | Token side | Window | Channels (full width) | Library composition |
|---|---:|---:|---:|---|
| Encoder pairs | 14 / 1 | 14 / 1 | 768 | Reused encoder; 196 spatial rows plus one class row |
| Each cat readout | 14 | 14 | 768 | `repeat_rows`, `concat` spatial first, linear 1536 → 768, exact GELU |
| Reassemble 0 | 56 | 7 | 256 | 1×1 to 256; packed k4/s4 transpose conv; factor-4 depth-to-space; bias-free 3×3 |
| Reassemble 1 | 28 | 7 | 256 | 1×1 to 512; packed k2/s2 transpose conv; factor-2 depth-to-space; bias-free 3×3 |
| Reassemble 2 | 14 | 14 | 256 | 1×1 to 1024; identity resize; bias-free 3×3 |
| Reassemble 3 | 7 | 7 | 256 | 1×1 to 1024; biased 3×3 stride 2; bias-free 3×3 |
| Fusion 0 | 7 → 14 | 7 → 14 | 256 | Reassemble 3, two residual conv blocks, x2 resize, 1×1 |
| Fusion 1 | 14 → 28 | 14 → 7 | 256 | Reassemble 2 plus previous feature, same fusion |
| Fusion 2 | 28 → 56 | 7 → 7 | 256 | Reassemble 1 plus previous feature, same fusion |
| Fusion 3 | 56 → 112 | 7 → 7 | 256 | Reassemble 0 plus previous feature, same fusion |
| Head | 112 → 224 | 7 → 7 | 256 → 2 | Bias-free 3×3, BN, ReLU, eval dropout, biased 1×1, x2 resize |

Every fusion first applies residual block 1 to its own reassembled feature, then
adds the previous fusion output, then applies residual block 2. Each residual
block is ReLU → bias-free 3×3 → BN → ReLU → bias-free 3×3 → BN → skip addition.
BN epsilon is 1e-5. Resizes use `window_resize(factor=2, align_corners=True)`;
the two distinct-output-window settings implement the 7 → 14 → 28 bridge.
The W14 feature at 14 tokens avoids an extra layout conversion for fusion 1.
Zero padding uses each complete crop's extent. Global attention and the aligned
resize coefficients make this a fixed-crop target, not a halo-tileable model.

The transpose-convolution packing is Prithvi's existing
`_upsampling_parameters(..., factor=2)` generalized to factor 4, shared by both
models. Position-major `(dy,dx,Cout)` columns feed one existing linear and one
depth-to-space call. No per-channel projection or second model packer is added.
The [preservation record](evidence/dpt-prithvi-preservation.json) compares 68
Prithvi graphs before/after the change: 33 full-width forward boundaries, 33
full-width staged boundaries and two existing reduced-fixture requests.

## Fixed integer reference procedure

The final full-width and reduced numerical references overwrite every learned
SMP/timm parameter using a counter-based integer hash, with no library RNG values
retained. The generator and scales are fixed before comparison.

For a parameter's full `state_dict` name, the key is
`uint64(zlib.crc32(name.encode("utf-8")))` (zero-extended from 32 bits). For flat
index `i`, start at `key+(i+1)*0x9E3779B97F4A7C15`, then apply SplitMix64's
xor-shift/multiply finalizer (`>>30`, multiplier `0xBF58476D1CE4E5B9`; `>>27`,
multiplier `0x94D049BB133111EB`; final xor `>>31`). All arithmetic wraps modulo
`2**64` in NumPy uint64. Take its top 24 bits `u`; the value
`d=(u+0.5)/2**23-1` lies in `(-1,1)` and is exactly representable in float32.
Compute `d*sqrt(3)*std_native` in float64, then cast once to float32.
This matches the source initializer's standard deviation, with uniform dyadics
rather than Gaussian samples where the native initializer is Gaussian.
Constant native initializers stay constants. Every tensor is built twice and
asserted bit-identical; five independently derived name/index/hash answers
are hard-coded in the generator and asserted before initialization.

The scales below were read from the pinned installed sources before declaration:
`timm/models/vision_transformer.py` (`VisionTransformer.init_weights`,
`init_weights_vit_timm`), `timm/layers/patch_embed.py`, SMP's `base/model.py`
and `base/initialization.py`, and torch's `_ConvNd.reset_parameters`, norm
`reset_parameters`, and `nn.init` fan calculations. The record names each
parameter, native initializer and source, standard deviation, fan-in, shape,
and generator key. No observed tensor RMS or comparison error sets a scale.

| Parameters | Native initializer | Declared standard deviation |
|---|---|---|
| Encoder Linear weights, including unused classifier | timm 1.0.30 truncated normal | .02 |
| Position embedding | timm 1.0.30 truncated normal | .02 |
| Class token | timm 1.0.30 normal | 1e-6 |
| Encoder patch Conv2d weight and bias | torch 2.14 defaults | `sqrt(1/(3*fan_in))` |
| Decoder Conv2d weights | SMP 0.5.0 Kaiming uniform, ReLU, fan-in | `sqrt(2/fan_in)` |
| Decoder Linear weights | SMP 0.5.0 Xavier uniform | `sqrt(2/(fan_in+fan_out))` |
| Decoder ConvTranspose2d weight and bias | torch 2.14 defaults, untouched by SMP | `sqrt(1/(3*initializer_fan_in))` |
| Head Conv2d weights | SMP 0.5.0 Xavier uniform | `sqrt(2/(fan_in+fan_out))` |
| Norm affine / other biases | Native constants | weights 1; biases 0 |

Linear fan-in/out are its input/output widths. Conv2d uses
`fan_in=(Cin/groups)*kh*kw`, `fan_out=Cout*kh*kw`. For ConvTranspose2d,
torch's stored `[Cin,Cout,k,k]` gives `initializer_fan_in=Cout*kh*kw`;
the record distinguishes that initializer convention from `fan_in=Cin`, the
actual sum at each non-overlapping output. Torch's default convolution bias is
random uniform, so patch/transpose biases also use the hash at that std.

The two periodic-pattern records stay unchanged as history:
[unscaled](evidence/dpt-integer-offline.json), declared at `24e1b6f`, and
[fan-in-scaled](evidence/dpt-fan-in-offline.json), preserved at `ce619d9`.
The `_fixed_weights` pattern `((17*i+3)%31-15)/128` repeats every 31 values;
its matrices have rank at most 30. Diagnostic measurements on 768×768,
3072×768 and 256×2304 matrices found top singular values 5–10 times those of
random weights at the same RMS. This coherent structure made the full-width
activations snowball: fusion 3 reached 3.8e8 unscaled and 2.99e6 after fan-in
scaling. It is unsuitable for these model numerical references. `_fixed_weights`
itself remains unchanged for graph-gate fixtures. Native-init evidence also
remains unchanged. The hash recipe's calibration and evaluation inputs are
unchanged from `24e1b6f`.

BatchNorm running statistics are reset and calibrated once, using four batches of two six-band
224 × 224 inputs generated by `dyadic(14128, "bn-calibration-{batch}",
(2,224,224,6), scale=.5, bits=16)`, for batch indices 0 through 3.
The model runs in training mode only for this pass, under `torch.no_grad()`,
with BatchNorm `momentum=None` (cumulative averaging), then returns to evaluation
mode and restores momentum. Learned parameters are checked unchanged; only
running statistics and batch counters change. Head dropout follows all BatchNorms.
Evaluation uses the separate `dyadic(14127, "random-weight-input", (224,224,6),
scale=.5, bits=16)` input. The procedure is fixed before comparisons and is not
adjusted in response to results.

The comparison uses the exact calibrated float32 state and input, then
`model.double()` and the same input converted exactly to double. Head and logits
require largest absolute error at most 1e-4 against float32 PyTorch. For every
intermediate output, mirror error against float64 PyTorch must be no larger
than float32 PyTorch's own error. Records include per-output peaks, float32
spacing, all three absolute errors and argmax flips. This is offline mirror
agreement only; it establishes neither live agreement nor model accuracy.
The compact live fixture below keeps its original weights and seven pinned hashes.
The reduced inline converter gate uses native SMP literals on both sides; it
checks serialization identity, independently of these numerical references.

## Numerical scope

The independent SMP reference hooks its actual forward outputs: encoder pairs,
readouts, reassembled features, fusion outputs, head and final logits. The NumPy
mirror uses the library's established rounding hypotheses. The final
[hash-reference record](evidence/dpt-hash-offline.json), generated once on clean
declaration commit `6197082`, contains all 22 outputs at each width and every
parameter's native-scale initialization rule. Both widths passed the
[numerical comparison rules](../CONTRIBUTING.md#numerical-comparisons): head
and logits meet 1e-4 against float32 PyTorch, and all 20 intermediates have
mirror/float64 error no larger than float32 PyTorch's own. There is no expected
failure for this final reference. No weights, inputs or calibration were
adjusted after comparison. Every output at both widths has zero argmax flips in all three comparisons.

| Reference | Head error vs float32 | Logit error vs float32 | Head error vs float64 | Logit error vs float64 |
|---|---:|---:|---:|---:|
| Full width | 2.059340476989746e-5 | 2.0056962966918945e-5 | 4.209816711675529e-6 | 5.705644484210737e-6 |
| Reduced | 7.599592208862305e-6 | 7.286667823791504e-6 | 9.560554570825497e-6 | 1.1002035396723286e-5 |

At full width, the largest mirror/float32 error is 0.00018787384033203125
at fusion 3 (fusion 2 also exceeds 1e-4). These intermediate outputs are checked
against float64, not held to a 1e-4 float32 target. Fusion 3's mirror/float64
error is 4.644907568440715e-5 versus float32/float64 error
0.00018706553028202677; it is the largest mirror/float64 error in the model.
Its float32 peak is 51.930015563964844 with step 3.814697265625e-6.
Head/logit peaks are 3.6273183822631836/3.608076572418213.

At reduced width, fusion 3 has the largest mirror/float32 error,
5.7697296142578125e-5, and mirror/float64 error, 3.139735947632616e-5,
versus float32/float64 error 6.617691321508801e-5. Its float32 peak is
28.775081634521484 with step 1.9073486328125e-6. Head/logit peaks are
3.752713680267334/3.7479302883148193. The record retains peaks, float32
steps, three absolute errors and argmax flips for every output.

The earlier evidence remains unchanged as history:

- [Native-init dtype record](evidence/dpt-dtype-offline.json): full-width
  mirror/float32 error 0.00060272216796875 at fusion 3 and
  0.0003509521484375 at logits; activations reached 190.1188201904297.
- [Unscaled periodic-pattern record](evidence/dpt-integer-offline.json),
  declared at `24e1b6f`: head/logits met 1e-4, but fusion 3 reached
  381,309,472 (step 32), with mirror/float32 error 1000 and
  mirror/float64 error 105.84820801019669. This unrealistic activation
  regime caused the procedure to be superseded.
- [Fan-in-scaled periodic-pattern record](evidence/dpt-fan-in-offline.json),
  preserved at `ce619d9`: full-width head/logit errors
  0.0005536079406738281/0.0005426406860351562 against float32, with
  mirror/float64 errors 0.00010378112662046135/0.00010239660514788085.
  Fusion 3 still reached 2,992,211.5. That run stopped before reduced width.

The final hash recipe addresses the periodic matrices' coherent low-rank
structure described above. Offline mirror agreement establishes neither Earth
Engine execution nor model accuracy. Graph identity and the unchanged compact
reduced live evidence remain separate.

The reduced twin retains 224/16 geometry, four raw feature pairs and 64-dimensional
heads, with encoder width 64, one head and four blocks; intermediate widths
`(4,8,16,16)`, fusion width 4, two classes. SMP's real encoder wrapper, decoder
and head run with a real reduced timm `VisionTransformer` supplied by a test
factory (SMP reserves the `depth` keyword for its four output features).

`vit_dpt_reduced` is a separate composition fixture. Its integer-dyadic
coefficients reconstruct each matrix from at most four seed rows multiplied by
distinct dyadic row gains. These rank-at-most-four weights keep interactive
requests small; positional rows use one seed row times distinct dyadic row gains.
They do not validate unrestricted random weights. Input pixels
combine a periodic 16 × 16 six-band seed with a global pixel code. Readouts cover
all reassemble and fusion outputs; head/final outputs sample 56 × 56 locations
from the 112/224 maps. The input seed is echoed exactly. Offline tests compare
all pixels and check native NumPy, native ServerArray and flattened ServerArray
paths. Whole-model records are excluded from individual-block promotion.
The fixture's head BN bias has a positive dyadic offset to keep its ReLU active;
a sensitivity test requires nonconstant logits that change with the input.
The [reduced fixture's offline record](evidence/dpt-reduced-offline.json),
generated on clean `2d702c9`, passes all ten readouts. Its largest absolute
mirror-versus-PyTorch error is 1.862645149230957e-9 (fusion 3); head and logits
are each 9.313225746154785e-10. Its seven-request plan has a largest payload
of 1,944,662 bytes and at most 31,360 output values per request, with no assets
or tasks. This is separate evidence from the calibrated integer-reference check.

The [reduced G3-live run](evidence/live-fixtures/20261103T041356Z-vit_dpt_reduced.json)
passed exactly those seven requests on clean `24fdec8`, with zero assets and
tasks. Every readout was float32 and fully valid; the input echo was exact.
Earth Engine matched the NumPy mirror bit for bit at all ten output readouts.
The largest absolute EE-versus-SMP error was 1.862645149230957e-9 (fusion 3);
head and final sampled logits were each 9.313225746154785e-10. This validates
only the structured reduced twin and sampled readouts described above.

Post-run local comparison on identical input/weight hashes found zero argmax
flips at every readout, including final logits. This inference uses the live
record's EE-versus-mirror bit identity and the local mirror-versus-SMP decisions;
the live record itself remains exactly as the harness wrote it.

## Offline request sizing and deferred full-width live run

```sh
PYTHONPATH=. python tools/vit_b16_dpt224_plan.py
PYTHONPATH=. python tools/live_fixtures.py --plan vit_dpt_reduced
PYTHONPATH=. flock /tmp/tensor-ee-heavy.lock python tools/vit_b16_dpt224_numerics.py \
    --output docs/evidence/dpt-hash-offline.json
```

The numerical command writes the hash-initialized reference record. The tool
refuses to overwrite earlier evidence; the fixed procedure is not retuned or
rerun to obtain a passing result.

The planning command measures actual serialized export payloads against offline
EE signatures. It uses synthetic table reads, one per parameter; these create no
assets. Its explicit cuts are patch plus twelve encoder blocks (spatial and class
saved separately), four reassemble outputs, four fusion outputs and final logits:
22 stages, 35 inference exports/tasks and 35 saved inference images. Encoder
stages recompute once per component export. Reassembly reads its corresponding
saved raw encoder pair; fusion reads its own reassembled feature and the previous
fusion result. The plan records constants, output bands, inputs and request bytes
per export under 24M constants, 10MB/request, 1024 saved bands and at most two
global-attention blocks per stage. It measures the table-backed single-graph
payload separately. Request capacity does not predict runtime or memory.

The [measured proposal](evidence/dpt-full-width-plan.json), built on clean
`cc5a5c6`, has export payloads from 13,974 to 904,545 bytes per stage output;
all 22 cuts fit those limits. The single-graph table-backed export is 2,625,882
bytes with 149,172,200 logical constants and twelve global-attention blocks.

Full-width random weights require staged execution with weight tables and a saved
input, so they do not fit the zero-assets/zero-tasks `live_fixtures` path. The
proposal adds one input image and 271 synthetic weight tables, one per parameter;
actual sharding/upload requests, comparison readbacks and polling calls depend
on a concrete manifest and runner and are not yet budgeted. Nothing here uploads
weights or submits exports. The later recognizer/runner must remeasure real
transport, provide the final command and total call/task/asset budget, and obtain
maintainer approval before a full-width run. The supplied compute estimate is
approximately 40 G MACs per crop; EECU cost and runtime are unknown. The future
comparison must record largest absolute logit error and argmax flips. Full-width
live agreement and model accuracy remain unmeasured.

Full-width live validation is **outside this PR**, moved to a follow-up; that
criterion is not met here. The follow-up needs the staged runner, its 35 inference
export tasks, proposed 271 weight tables and input asset, concrete upload/readback
budgets, and an EECU estimate before maintainer approval. No full-width live run
or export has been submitted.
