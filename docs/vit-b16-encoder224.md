# ViT-B/16 encoder target at 224

`tensor_ee.models.vit_b16_encoder224` is a hand-composed target for a later
converter recognizer. It is not imported by `import tensor_ee`. It uses existing
blocks, with no converter or new block. It validates offline composition and
numerical agreement on the recorded inputs; no encoder execution in Earth Engine,
converter equivalence, trained-model accuracy, or DPT decoder is claimed.

## Contract and usage

The default `ViTConfig` is six bands, exactly 224 × 224 pixels, patch 16, width
768, 12 heads of dimension 64, 12 blocks, MLP ratio 4, biases, exact GELU,
LayerNorm eps 1e-6, and unfused query-first attention scaling. Dropout, drop path,
layer scale and pre-norm before the blocks are absent/identity in evaluation.
The reduced twin has width 128, two heads and two blocks, keeping the same grid
and head dimension. Other widths/depths have no numerical evidence here.

Weights use **`encoder.model.*`**, as in the encoder portion of SMP 0.5.0 DPT
with `tu-vit_base_patch16_224`. Pass only those encoder keys from a DPT state;
the unused `encoder.model.head.weight` and `.bias` are accepted and ignored.
A standalone `TimmViTEncoder.state_dict()` has `model.*` keys: prepend `encoder.`.
A standalone timm model has unprefixed keys: prepend `encoder.model.`.
No model package is imported by the implementation.

```python
from tensor_ee.models.vit_b16_encoder224.model import constants, forward

# ee signatures must already be available; this code initializes nothing.
# state maps encoder.model.* names to NumPy float32 arrays or ServerArrays.
coefficients = constants(ee, state)
outputs = forward(ee, six_band_raster, coefficients)
spatial, cls = outputs["block2"]
normalized_spatial, normalized_cls = outputs["norm"]
```

Native NumPy shapes are required. ServerArrays also accept flattened matrices
`[first_axis, product(other_axes)]` and row/column vectors. Constants are prepared
before spatial composition. Every output is `(spatial, class)`:
W14 `[196,C]` on the 14 × 14 token grid, and W1 `[1,C]` on its coarse grid.
The class token is timm sequence row **0**, followed by 196 row-major spatial
rows. Position weights split as `pos_embed[0,1:]` and `pos_embed[0,:1]`;
class plus position is added with the library block, not pre-summed on the host.

`outputs` contains `input`, positioned `patch`, raw `block0` through `block11`,
and the final `norm`. SMP's four default DPT indices are **(2,5,8,11)**;
`forward_intermediates(norm=False)` returns raw block pairs, which differ from
the separate final-normalized pair. `compose(..., start="block2", stop="norm")`
accepts a saved pair and returns every later boundary. Input rasters are relabeled
to the fixed abstract model grid; output carriers stay on that grid.

## Offline evidence

[Recorded comparisons and request measurements](evidence/vit-b16-encoder224-offline.json)
use random-initialized timm 1.0.30, `pretrained=False`, `fused_attn=False`, seed 9316,
and deterministic dyadic input. Tests compare every spatial/class value, including
all blocks and the final norm, with absolute target 1e-4. Actual library arithmetic
through the eager spatial-transport harness equals the independent NumPy mirror
bit-for-bit, with flattened ServerArray coefficients. The reduced fixture also
checks native NumPy coefficients and compact server coefficient reconstruction.
This eager execution is an offline numerical hypothesis, not Earth Engine execution.

Largest absolute mirror-versus-timm errors for each output:

| Boundary | Reduced spatial | Reduced class | Full spatial | Full class |
|---|---:|---:|---:|---:|
| patch | 9.53674316e-07 | 0 | 1.01327896e-06 | 0 |
| block0 | 1.01327896e-06 | 5.96046448e-08 | 1.60932541e-06 | 7.74860382e-07 |
| block1 | 1.04308128e-06 | 9.49949026e-08 | 2.50339508e-06 | 1.07288361e-06 |
| block2 | — | — | 3.09944153e-06 | 1.43051147e-06 |
| block3 | — | — | 3.17394733e-06 | 1.66893005e-06 |
| block4 | — | — | 4.09781933e-06 | 1.81794167e-06 |
| block5 | — | — | 4.14252281e-06 | 2.14576721e-06 |
| block6 | — | — | 4.4927001e-06 | 2.2649765e-06 |
| block7 | — | — | 4.58955765e-06 | 2.86102295e-06 |
| block8 | — | — | 5.36441803e-06 | 2.98023224e-06 |
| block9 | — | — | 5.48362732e-06 | 3.33786011e-06 |
| block10 | — | — | 5.84125519e-06 | 3.87430191e-06 |
| block11 | — | — | 6.2584877e-06 | 4.05311584e-06 |
| norm | 5.48362732e-06 | 1.28149986e-06 | 4.41074371e-06 | 2.8014183e-06 |

The shared Prithvi `compose` and encoder backend keep their former defaults
(`backbone.*`, eps 1e-5, score scaling). The [preservation check](evidence/vit-prithvi-preservation.json)
compares all 33 default full-width boundary graphs with base `924a7c2`, including
all encoder pairs and decoder outputs. The existing reduced crop fixture's graphs
also retain the live record's SHA-256 values; pre/post crop and 2×2-region plans
are identical. Existing evidence records and block statuses are unchanged.

## Request capacity

`planning.measured_cuts` measures explicit cuts with caller-supplied ServerArrays,
saved input reads, and the library request sizer; it creates no assets or tasks.
The recorded measurements use synthetic table reads, one row-list table per
parameter. Their exact bytes apply only to those expressions and example namespace;
a real manifest must be measured again. Constants and boundary widths are exact.

| Explicit stage | Spatial request bytes | Class request bytes | Conservative constants |
|---|---:|---:|---:|
| input → patch | 904,540 | 13,968 | 1,482,240 |
| patch → block0 | 230,125 | 229,998 | 16,122,624 |
| block0 → block1 | 230,127 | 230,000 | 16,122,624 |
| block1 → block2 | 230,127 | 230,000 | 16,122,624 |
| block2 → block3 | 230,127 | 230,000 | 16,122,624 |
| block3 → block4 | 230,127 | 230,000 | 16,122,624 |
| block4 → block5 | 230,127 | 230,000 | 16,122,624 |
| block5 → block6 | 230,127 | 230,000 | 16,122,624 |
| block6 → block7 | 230,127 | 230,000 | 16,122,624 |
| block7 → block8 | 230,127 | 230,000 | 16,122,624 |
| block8 → block9 | 230,127 | 230,000 | 16,122,624 |
| block9 → block10 | 230,141 | 230,014 | 16,122,624 |
| block10 → block11 | 230,143 | 230,016 | 16,122,624 |
| block11 → norm | 163,065 | 18,357 | 302,592 |

Each component fits one request under the default 10 MB cap; each stage needs
two separate exports, each recomputing its stage. All boundaries have 768 bands,
under 1024; each block has 16,122,624 conservative constants, under 24 million.
The whole encoder has 86,388,480 state values and 110,321,664 logical constants;
a single whole-encoder stage fails the constant/extent limits. Cuts are a hand
proposal, not an optimized schedule or measured runtime. Global attention spans
the whole 224-pixel chip; attention stages cannot be tiled with halos. Peak memory
and EECU cost are unknown. No runner or exports are included or approved.

## Reduced live plan (approval pending)

`vit_encoder_reduced` selects `window/concat_rows/vit-encoder-reduced`.
Integer-only dyadic coefficient seeds expand to distinct matrix rows and columns;
position rows and the input's global pixel code distinguish tokens. All spatial
and class channels after both blocks and final norm are read, plus the input echo.
The fixture is composition evidence, excluded from individual-block coverage.

```sh
PYTHONPATH=. python tools/live_fixtures.py --offline vit_encoder_reduced --output /tmp/vit-encoder-offline.json
PYTHONPATH=. python tools/live_fixtures.py --plan vit_encoder_reduced
```

The plan is **5 requests, 0 assets, 0 tasks**, largest request **1,637,391 bytes**,
maximum 50,568 output values and NPY header 7,862 bytes, within the unchanged
interactive budget. Spatial readouts use 14×14; class readouts 1×1; the input echo
32×32. The full plan must be reviewed and approved before a live run. No
full-width live selection or staged/export run is proposed. A reduced live result
would validate only that reduced composition, not full ViT-B/16 execution or accuracy.

## Verification

Run `tests/test_vit_b16_encoder224.py`, plus the Prithvi composition/execution and
fixture-harness tests when changing shared wiring. Tests requiring timm or SMP use
`pytest.importorskip`, so core installations skip those reference comparisons.
Deep graph building and serialization use scoped `deep_recursion()` contexts.
The standing G1 and G1-ext gates protect existing targets; no goldens are rewritten.
