# ViT-B/16 + DPT target at 224

`tensor_ee.models.vit_b16_dpt224` is the hand-composed target for a later SMP
DPT recognizer. It reuses `vit_b16_encoder224` and existing library blocks.
It adds no block or converter support and is not loaded by `import tensor_ee`.
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

## Numerical scope

The independent SMP reference hooks its actual forward outputs: encoder pairs,
readouts, reassembled features, fusion outputs, head and final logits. The NumPy
mirror uses the library's established rounding hypotheses. Tests compare every
pixel at reduced and full width under the unchanged absolute 1e-4 target and
report the largest absolute error per output and final-logit argmax flips.
Those checks are offline agreement, not Earth Engine execution or model accuracy.

The current full-width check **fails** the absolute target. With SMP's native
random initialization (seed 14126), the largest error is 0.00060272216796875
at fusion 3; final logits differ by 0.0003509521484375 with zero argmax flips.
The [failed per-output record](evidence/dpt-smp-full-width-failed.json) also
feeds exact SMP encoder pairs to the decoder mirror: the decoder alone still
fails. Disabling oneDNN reduces the error but
[still fails](evidence/dpt-smp-cpu-backend-failed.json) at final logits
(0.00017547607421875). These observations do not establish a cause or justify
a tolerance change. The full-width regression test remains failing pending a
maintainer decision; no full-width equivalence is claimed.

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

## Offline request sizing and deferred full-width live run

```sh
PYTHONPATH=. python tools/vit_b16_dpt224_plan.py
PYTHONPATH=. python tools/live_fixtures.py --plan vit_dpt_reduced
```

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

Full-width random weights require staged execution with weight tables and a saved
input, so they do not fit the zero-assets/zero-tasks `live_fixtures` path. The
proposal adds one input image and one synthetic weight table per parameter;
actual sharding/upload requests, comparison readbacks and polling calls depend
on a concrete manifest and runner and are not yet budgeted. Nothing here uploads
weights or submits exports. The later recognizer/runner must remeasure real
transport, provide the final command and total call/task/asset budget, and obtain
maintainer approval before a full-width run. The supplied compute estimate is
approximately 40 G MACs per crop; EECU cost and runtime are unknown. The future
comparison must record largest absolute logit error and argmax flips. Full-width
live agreement and model accuracy remain unmeasured.
