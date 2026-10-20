# SMP CNNs through compile_model

Use `tensor_ee.smp.adapt_smp_model` to prepare an existing, trained
segmentation-models-pytorch (SMP) CNN for the FX compiler:

```python
from tensor_ee.smp import adapt_smp_model
from tensor_ee import check_compatibility, compile_model

adapted = adapt_smp_model(trained_model.eval(), input_shape=(1, 2, 224, 224))
report = check_compatibility(adapted, (1, 2, 224, 224))
assert report.status == "supported", report.to_dict()
compiled = compile_model(adapted, image, bands, grid, include_unfused_nodes=False)
```

`trained_model`, `image`, ordered `bands`, and `grid` are supplied by the caller.
Construct models with `encoder_weights=None` and load an existing state dictionary;
the adapter never obtains weights or data. Install the existing `segwater` extra
for SMP 0.5.0 ([installation](getting-started.md#1-install)).

## Contract

```python
adapt_smp_model(model: torch.nn.Module, *,
                input_shape: tuple[int, int, int, int]) -> torch.fx.GraphModule
```

The input is positive, batch-one NCHW. All modules must be in evaluation mode,
all state must be on CPU, and floating parameters and buffers must be finite
float32. Complex state and module hooks (including global hooks) are refused.
The original model, its tensors, flags and CPU PyTorch RNG state are preserved.
The returned module owns independent, frozen parameters.

Eligible classes are `Unet`, `UnetPlusPlus`, `MAnet`, `Linknet`, `FPN`, `PSPNet`,
`PAN`, `DeepLabV3`, `DeepLabV3Plus`, and `UPerNet`, using the inherited
`SegmentationModel.forward`. There is no encoder-name compiler allowlist.
Eligibility means the forward can be adapted; actual arithmetic must still
satisfy `check_compatibility` and `compile_model`.

Any non-`None` `classification_head`, including one constructed with `aux_params`,
is refused because its additional output cannot be discarded. Segmentation-head
activations and channel counts are preserved. For example, sigmoid is supported;
an unsupported head operation remains a named compatibility blocker.

`Segformer` is refused with a pointer to `tensor_ee.converter.convert`, the
[attention conversion path](converter.md). Forward overrides, including DPT's,
are refused: DPT passes prefix tokens as well as features to its decoder.
Untraceable configurations raise `SMP FX trace failed` with the original error.
No diagnostic tracing retry, attention leaf rule, or data-dependent branch
specialization is used. Raw `check_compatibility` trace failures retain their
existing report behavior.

The adapter deep-copies the model and retains its encoder, decoder and
segmentation-head operations, omitting only the base forward's Python input-shape
check. Copied modules with an `inplace` flag use `False`. A deterministic,
nonconstant dyadic probe is evaluated by an untouched raw-model copy and the final
adapter at exactly the requested shape. The outputs must be finite and
`torch.equal`; any mismatch is refused with its largest absolute error.
This single-input check is not an all-input numerical proof.

`tensor_ee_input_shape` records CHW, with batch one enforced during eager calls;
both compilation and compatibility analysis check the same fixed input contract.
Read JSON-safe `tensor_ee_smp_metadata` for the architecture, checked shape,
output shape, SMP/PyTorch versions, probe description, `torch_equal`, and `max_abs`.
The adapter explicitly requires `SMP_VERSION = "0.5.0"`. Another version needs
review and new evidence. Keep the returned module fixed; replacing its state or
components requires adapting and probing again.

## Offline graph construction

This complete example uses random weights and Earth Engine's offline signatures.
It submits no work and needs no authentication. It uses a smaller decoder for
the example; the standard decoder is covered by the evidence below.

```python
import ee
from ee.apitestcase import ApiTestCase
import torch
import segmentation_models_pytorch as smp
from tensor_ee import Grid, check_compatibility, compile_model
from tensor_ee.smp import adapt_smp_model

torch.set_num_threads(2)
raw = smp.Unet("resnet18", encoder_weights=None, encoder_depth=3,
               decoder_channels=(8, 4, 2), in_channels=2, classes=2).eval()
shape = (1, 2, 64, 64)
adapted = adapt_smp_model(raw, input_shape=shape)
assert check_compatibility(adapted, shape).status == "supported"

api = ApiTestCase()
api.setUp()
try:
    bands = ("first", "second")
    image = ee.Image.constant([0, 0]).rename(bands)
    grid = Grid("EPSG:3857", (10, 0, 0, 0, -10, 0), 64, 64)
    compiled = compile_model(adapted, image, bands, grid,
                             include_unfused_nodes=False)
    assert compiled.bands == ("c0", "c1")
finally:
    api.tearDown()
```

Standard encoders have millions of coefficients. A successful build is not a
request-size or Earth Engine capacity guarantee. For an existing completed weight
manifest use `MatchedWeightProvider(adapted, manifest)` as `weight_provider`, with
`include_unfused_nodes=False`. The existing collector reads the specialized FX
module and verifies every effective Conv/BatchNorm coefficient. No change to
weight storage is required. The offline tests use compact symbolic arrays for
graph serialization and compare actual coefficient hashes separately.

## Static plumbing and exactness

Compiler, compatibility analysis and adapter share one pass in `semantics.py`.
It infers tensor metadata structurally, without running activations, then resolves:

- `getattr(tensor, "shape")` from proven NCHW metadata;
- literal integer indices and literal slices on those shape tuples;
- selections and subsequences from literal lists/tuples of proven tensor references;
- nearest interpolation to a static positive `(Hout, Wout)` when both output/input
  ratios are the same positive integer `k`.

Tensor-value indexing, dynamic or invalid indices, arbitrary attributes, unavailable
shapes, fractional/unequal/shrinking resize ratios, other interpolation modes and
unsupported options remain refused or unknown with their original node names.
Size together with `scale_factor`, and size with `recompute_scale_factor=True`,
are refused. Shape/container bookkeeping emits no raster operation or cast;
dead arithmetic is retained for support analysis.

An accepted size is lowered to the existing nearest block, through its existing
factor validator. That block accepts every positive integer factor; **2, 3 and 4
are the validated settings**. Factors outside that record have the same support
and evidence limits as the existing scale-factor path. No block, validated record,
precision policy, default, convolution folding rule, or cast placement changes.

For integer `k`, PyTorch nearest reads input index
`floor(i * Hin / Hout) = floor(i / k)` when `Hout = k * Hin`, matching the existing
block's repeated-pixel mapping. Offline tests compare size and scale forms and
the `floor(i/k)` indices for every factor 1 through 16 at spatial sizes
`1×1`, `2×3`, `7×9`, and `32×32`. The accepted factor domain is unbounded;
these finite tests support the identity argument rather than exhausting it.

## Evidence

On CPU Python 3.12.14, PyTorch 2.14.1+cpu, torchvision 0.29.1+cpu and SMP 0.5.0,
with random weights, two input channels and two classes:

| Architecture | Encoders | Input sides | Supported / builds / probe equality | max_abs |
|---|---|---|---|---:|
| Unet | resnet18, resnet34, resnet50, vgg11_bn | 64, 128, 224 | 12 / 12 / 12 | 0 |
| UnetPlusPlus | resnet18, resnet34, resnet50, vgg11_bn | 64, 128, 224 | 12 / 12 / 12 | 0 |

Both architectures also pass at `64×96`, three input channels, three classes and
decoder channels `(16, 8, 8, 4, 4)`: supported, build, equality true, max_abs 0.
The six negative pairs trace and pass the adapter probe (max_abs 0), then remain
unsupported with actual arithmetic blockers:

| Pair | Representative blocker |
|---|---|
| Linknet/resnet18 | ConvTranspose2d |
| FPN/resnet18 | GroupNorm; additional stack/reduction and bilinear calls |
| DeepLabV3Plus/resnet18 | Dilation; padding and adaptive pooling |
| Unet/resnext50_32x4d | Grouped convolution |
| Unet/densenet121 | BatchNorm placement |
| Unet/mobilenet_v2 | ReLU6; depthwise convolution |

Run the full matrix explicitly; the default suite skips it:

```sh
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
TENSOR_EE_SMP_MATRIX=1 PYTHONPATH=. python -m pytest -q -s \
    tests/test_smp.py::test_full_smp_matrix
```

The default suite retains Unet/resnet18 and UnetPlusPlus/resnet18 at 64, the
Linknet refusal, contract tests with small components, and the SegWater identity.
For a random model with the unchanged SegWater recipe at 224, the generic adapter
and `Fixed224SMPUnet` produce **4/4 byte-identical output graphs** across diagnostics
and clipping on/off. All **31/31** effective convolution keys, sources and
coefficient hashes match, as do tensor-node and folded-BatchNorm keys. Compact
symbolic weights keep the serialized graphs small; actual coefficients are
hashed independently. The existing SegWater adapter is unchanged.

Graph identity, the adaptation probe, offline block reference checks, live
agreement and model accuracy are separate claims. No full-model NumPy evaluator
or mirror is used here. Trained checkpoints are absent in the container, so the
nine trained golden graphs require the maintainer's environment. The unchanged
Python 3.12 baseline has four `t04` raw-input drift cases in the 36 fixture graphs
and seven SegFormer raw-input drift cases in the 19 extended records; these are
failed strict gates, not passes or permission to rewrite golden files.

### Maintainer live validation plan (not run)

Use Unet/resnet18 with the SegWater recipe and existing verified weights. Adapt at
`(1, 2, 64, 64)`. Prepare a nonconstant float32 dyadic input with boundary impulses
on a 64×64 grid and compare both complete logit channels against the unmodified
PyTorch model using exactly those retained input pixels. Verify their IEEE
float32 identity by echoing both input channels, and check expected input and
output validity independently, ordered bands, affine grid and orientation.

Record max_abs, MAE, RMSE, argmax disagreements and near-margin locations. The
absolute target remains **1e-4**; retain the known U-Net impulse failure if it
occurs. This establishes neither model accuracy nor generalization to other chips.

Request budget:

- `python tools/live_fixtures.py --plan raster/nearest_upsample/f2` plans **2**
  computation requests, largest **9,936 bytes**, within the existing budget.
- Both model logits, both input echoes and validity on one 64×64 grid estimate
  **1** additional computation request: **3 total**, plus at most **2**
  initialization calls. The maintainer must preflight the actual model payload
  against the existing 2,000,000-byte request limit and remaining budget.
- Reuse existing assets only: **0 new assets, 0 tasks**, serial requests,
  no automatic reruns. No live request was submitted for this evidence.

The offline nearest-factor-two fixture passes with zero cloud calls. It has a
PyTorch reference and no NumPy mirror, so mirror-vs-PyTorch max_abs is N/A.
