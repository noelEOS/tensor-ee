# Run your own trained CNN over a region

This path takes a **standard SMP Unet/resnet34** checkpoint, with default encoder
depth and decoder, to tiled Earth Engine inference using the [SMP adapter](smp-models.md).
The [example](../examples/own_model.py) defaults to that architecture for checkpoint
loading and live inference; its shallow model is a separate offline demo.

This tiled standard-model recipe has not been validated live. The guide claims
graph construction and the library's existing validation, not whole-model
numerical agreement or accuracy. Compatibility does not measure capacity,
latency or cost. Whole-model runs use **batch export**: the library's
[live check](smp-models.md#status) found that one whole trained U-Net in a single interactive request exceeds
Earth Engine's user memory limit, even at 64×64. Server capacity at the much
larger context sizes below is unmeasured; batch export is not a capacity guarantee.

## 1. Install and try the shallow offline demo

Run from a repository checkout; examples do not ship in the wheel. Install the
existing extras without obtaining weights or input data:

```sh
python -m pip install 'torch>=2.10' 'torchvision>=0.20' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test,resnet,segwater]'
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
PYTHONPATH=. python -m examples.own_model --prepare
```

`--prepare` explicitly selects `make_demo_model()`: an SMP Unet/resnet34 with
`encoder_depth=3`, `decoder_channels=(16, 8, 4)`, two input channels and two
logit channels. It uses random weights (`encoder_weights=None`) and synthetic
constant input. It does not load the reader's checkpoint. Expect these fields:

```json
{
  "status": "offline_graph_only",
  "architecture": "shallow_demo",
  "weight_assets_exist": false,
  "tile_count": 2,
  "context_shapes_nchw": [[1, 2, 288, 288]],
  "halo": [112, 112],
  "alignment": 8,
  "global_dimensions_wh": [352, 288],
  "roi_dimensions_wh": [128, 64]
}
```

These are **demo numbers**. `graph_bytes` gives the UTF-8 serialized six-band
expression size: positive and below 2 MB in the test, varying with the Earth
Engine client version. It is not an export request size or output pixel size.

The demo uses `MatchedWeightProvider` and `FeatureCollectionWeightProvider`:
only shapes, coefficient hashes and nonexistent placeholder table references
enter the graph, never millions of inline weights. Its in-memory manifest is
not a completed upload and is never saved for live reuse. Signatures come
from `ee.apitestcase.ApiTestCase`, as in
[Getting started](getting-started.md#2-run-the-offline-example). No authentication,
upload, export or retrieval occurs. Only this shallow demo's full tiled graph
is built or serialized by `--prepare` and the offline graph tests.

## 2. Load your standard trained checkpoint

`make_model()` uses the defaults of the following training architecture:

```python
import torch
import segmentation_models_pytorch as smp

torch.set_num_threads(2)
raw = smp.Unet("resnet34", encoder_weights=None, in_channels=2, classes=2).eval()
raw.load_state_dict(torch.load("YOUR_CHECKPOINT.pth", map_location="cpu",
                               weights_only=True), strict=True)
```

Expect an eval-mode CPU model with your trained weights, including BatchNorm
statistics. Supply a plain `state_dict` with finite float32 floating state;
extract it locally first if your checkpoint wraps it in another dictionary.
Architecture/key mismatches fail strictly; do not ignore missing keys.
The default activation is `None`, retaining raw logits for `segmentation_outputs`.

For another architecture, pass SMP settings to `make_model`, or pass a
zero-argument `model_factory` to `load_model`, for example:

```python
from functools import partial
from examples.own_model import make_model, load_model

other = load_model("YOUR_OTHER_CHECKPOINT.pth",
                   model_factory=partial(make_model, encoder_name="resnet18"))
```

The factory must reproduce the training architecture without downloading weights.
If channel counts differ, adjust bands and output helpers and repeat the checks. The
[SMP guide](smp-models.md#evidence) lists separately tested configurations.

## 3. Derive standard support and plan equal context windows

First adapt at 64×64 solely to check local semantics and compute support:

```python
from tensor_ee import check_compatibility
from tensor_ee.smp import adapt_smp_model
from tensor_ee.receptive_field import analyze_receptive_field
from examples.own_model import make_plan

probe = adapt_smp_model(raw, input_shape=(1, 2, 64, 64))
report = check_compatibility(probe, (1, 2, 64, 64))
assert report.status == "supported", report.to_dict()
support = analyze_receptive_field(probe)
print(support.safe_halo, support.alignment)  # 544 32
plan = make_plan(support, core_size=1024)
```

Expect `supported`, `probe.tensor_ee_smp_metadata["torch_equal"] == True`, and
`max_abs == 0`: the adapter preserves the source model and checks one deterministic
PyTorch probe. Single-input local equality is not live validation.
Support is a structural dependency envelope, including skip branches and nearest
sampling phases; small observed distant effects do not justify reducing it.

For the **standard model**, halo is **544** and alignment is **32**. A square
context has side `core + 2 * 544`, or **core + 1,088 pixels**:

| Standard core | Context | Input pixels per core output pixel |
|---|---|---:|
| 64×64 | 1,152×1,152 | 324 |
| 1,024×1,024 | 2,112×2,112 | about 4.25 |

Use large aligned cores to amortize the halo; the live command defaults to
1,024. Larger contexts still have unmeasured server capacity. To inspect either
plan cheaply, with random standard weights and no Earth Engine graph:

```sh
PYTHONPATH=. python -m examples.own_model --plan --core-size 64
PYTHONPATH=. python -m examples.own_model --plan
```

The default standard plan report contains:

```json
{
  "status": "offline_plan_only",
  "architecture": "standard_unet_resnet34",
  "tile_count": 2,
  "context_shapes_nchw": [[1, 2, 2112, 2112]],
  "halo": [544, 544],
  "alignment": 32,
  "global_dimensions_wh": [3136, 2112],
  "roi_dimensions_wh": [2048, 1024]
}
```

At core 64, expect two 1,152×1,152 contexts, a 1,216×1,152 global input and
128×64 ROI. Both commands stop after the small probe and rectangle planning;
neither adapts a large context nor builds/serializes a standard EE inference graph.

This path chooses **one plan with identical context shapes**. `plan_tiles`
clamps contexts at finite global boundaries, so arbitrary edge or partial cores
could have different sizes, which a fixed-shape adapter cannot accept.
`make_plan` instead places two complete cores in a larger finite input domain:
ROI origin `(halo, halo)`, ROI dimensions `(2 * core, core)`, and global dimensions
`ROI + 2 * halo` on each axis. It calls `plan_tiles(global_grid, core, halo,
alignment=support.alignment, roi=roi)` using `PixelWindow` for the ROI.
For core 1,024, contexts start at `(0, 0)` and `(1024, 0)` and both are 2,112×2,112.
All origins, sizes, core widths and halo widths are multiples of 32.

The illustrative grid is `Grid("EPSG:3857", (10, 0, 0, 0, -10, 0), width, height)`:
10 m pixels, upper-left pixel **corner**, +x and downward rows. Replace its CRS
and affine origin in `make_plan` with your area's north-up +x/−y grid without
shear. It does not select a remote dataset. For larger rectangles, retain a full
halo around the ROI and use dimensions divisible by the core size. Pad partial
last cores to complete cores and crop outputs afterwards. For irregular areas,
plan the aligned bounding rectangle and apply the geometry as an output mask.

Obtain real prepared source pixels over the **whole padded input**, not just the
ROI. This defines a different finite domain from an earlier cropped image;
adding exterior zeros does not in general preserve that image's edge predictions,
because biased intermediate tensors also have finite boundaries. More tiles
increase graph size and recomputation.

## 4. Upload folded weights once — live, billed to your project

The remaining commands are for the reader to run with an Earth Engine account
and a Cloud project registered for Earth Engine. Uploads, reads and inference
consume the reader's project resources; Drive export also needs Drive access.
They have not been run for this guide. Authenticate explicitly:

```sh
earthengine authenticate
```

Expect local credentials to be saved. With existing application-default
credentials, select `--credentials adc` instead on the commands below. Both
examples reuse [examples/_ee.py](../examples/_ee.py) for explicit initialization.

```sh
PYTHONPATH=. python -m examples.upload_model_weights \
  --project YOUR_EE_PROJECT --checkpoint YOUR_CHECKPOINT.pth \
  --factory examples.own_model:load_model --manifest YOUR_NEW_MANIFEST.json
```

`load_model` defaults to the standard model and adapts/checks it at 64×64 for
weight collection. Effective coefficients are independent of the spatial probe
shape; uploading does not require a large-context adaptation or inference graph.
The trusted local `module:callable` factory takes a checkpoint `Path` and returns
a compilable eval CNN; use your own loader for other checkpoint formats or
architectures. It is ordinary Python code, not a sandbox.

The uploader calls `effective_convolutions` for the compiler's exact Conv/BatchNorm
folding and `export_weights(parameters, new_namespace, manifest_path)` to upload
bounded, measured table shards and wait. Expect progress JSON, then
`Completed weight manifest: …`, with `"status": "ready"` and every shard
`COMPLETED`. At most two exports run concurrently; the shard count depends on
the coefficient values. A separate uploader preserves the existing SegWater
[upload_weights.py](../examples/upload_weights.py) usage and output.

Use a new manifest filename and namespace. After interruption inspect recorded
task IDs before another upload; `export_weights(..., resume=True)` resumes the
same parameters, namespace and chunking. New weights need a new upload and
manifest. Keep both checkpoints and manifests private and outside the repository.

`FeatureCollectionWeightProvider` reconstructs float32 arrays from table rows;
`MatchedWeightProvider` also matches every local effective coefficient hash to
an asset key. Both construct deferred expressions without retrieving the weights.
Local identity does not independently checksum remote contents. The demo's
placeholder manifest must never be substituted for a completed upload.

## 5. Adapt contexts, compile and export — live, billed to your project

Provide your prepared input image asset with ordered bands `("first", "second")`,
finite float32 values already in the model's training units, and the chosen grid.
Edit `BANDS` for your actual ordered channels. Reproduce training resampling,
offsets and scales first; `toFloat()` alone does not normalize values.

On the reader's live path, `load_model(checkpoint, core_size=1024)` probes small,
plans the region, then adapts/checks the actual **2,112×2,112 context**. It requires
support to agree between adaptations. Only this live workflow performs the large
local probe and standard graph construction. Use `--core-size` for another size.

Expect the final adapter to have fixed CHW `(2, 2112, 2112)`, compatibility
`supported`, and a successful local probe. Passing the 3,136×2,112 global grid to
`compile_model` with this adapter would fail. `build_prediction` instead calls
`compile_tiled_model` on each equal-sized context with the shared
`MatchedWeightProvider`, `include_unfused_nodes=False`, float32 execution and
clipped finite footprints. It checks alignment/artificial margins, retains
disjoint cores, and mosaics two logit bands on the global grid masked to the ROI.
No millions of inline coefficients or unfused diagnostic convolutions enter it.

`segmentation_outputs` makes `score_0`, `score_1`, float32 softmax `p_0`, `p_1`, and
int32 `class_index` from raw-logit argmax, lowest index winning ties.
`binary_threshold` adds uint8 `binary_class` using inclusive `p_1 >= 0.5`: a tie
means argmax class 0 but binary class 1. The six export bands share float32 dtype,
which preserves these discrete values exactly.

The recipe captures source masks before zero-fill and requires both input channels
at the output pixel for probability/decision validity. Missing neighbours become
computational zeros, not proof of a complete input neighbourhood. Supply stricter
output validity if training/evaluation requires every contributing source pixel.
Raw logits retain their computational masks separately; see the
[API validity rules](api.md#segmentation-outputs-and-validity).

Choose a **new** asset destination:

```sh
PYTHONPATH=. python -m examples.own_model \
  --project YOUR_EE_PROJECT --checkpoint YOUR_CHECKPOINT.pth \
  --manifest YOUR_NEW_MANIFEST.json \
  --image projects/YOUR_EE_PROJECT/assets/YOUR_PREPARED_INPUT \
  --asset-id projects/YOUR_EE_PROJECT/assets/YOUR_NEW_PREDICTION
```

For Drive, replace the destination with `--drive-folder YOUR_DRIVE_FOLDER`:

```sh
PYTHONPATH=. python -m examples.own_model \
  --project YOUR_EE_PROJECT --checkpoint YOUR_CHECKPOINT.pth \
  --manifest YOUR_NEW_MANIFEST.json \
  --image projects/YOUR_EE_PROJECT/assets/YOUR_PREPARED_INPUT \
  --drive-folder YOUR_DRIVE_FOLDER
```

Expect one submitted batch task ID, then wait for `COMPLETED` in Earth Engine Tasks;
submission is not completion. The command does not poll, retry or resubmit.
The task uses the ROI's translated affine transform, CRS, exact region and
`maxPixels=2048*1024` by default, with no conflicting `scale` or `dimensions`.
Asset export uses sample pyramiding and `overwrite=False`; Drive writes a GeoTIFF.
Expect the six bands over the 2,048×1,024 ROI, without the context margin.

If Earth Engine rejects the graph or exhausts resources, retain the failure and
reduce the planned work before another attempt. Before relying on predictions,
compare exact prepared inputs and retrieved logits with the CPU model using
[PixelTensor and compare_segmentation](api.md#compare-exact-retrieved-pixels),
and evaluate accuracy separately on your labelled data.

## Offline checks

```sh
PYTHONPATH=. python -m pytest -q tests/test_own_model.py
```

Checks verify the standard checkpoint defaults, 64×64 support calculation and cheap
context plans without compiling a standard tiled graph. The full graph, direct
`compile_tiled_model` identity, halo coverage, and `--prepare` report checks all use
the shallow demo. Remote transport is refused and no graph is evaluated in Earth Engine.
