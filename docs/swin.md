# Fixed Swin-T/UPerNet workflow

Install from this repository with `python -m pip install '.[swin]'`, then run the
commands below from any directory. The distribution remains
`tensor-ee`; the `swin` extra pins `segmentation-models-pytorch==0.5.0`
and `timm==1.0.30`. Core `tensor_ee` imports and CNN execution do not require that
extra. ONNX/ORT are optional reference tools, not part of EE Swin execution.

This path supports two ordered input channels, two output logits, and one
224 × 224 chip with north-up square pixels. It uses the fixed Swin-T patch-4,
window-7 architecture and UPerNet decoder. It is not a generic transformer
compiler. A different grid changes execution coordinates; it does not establish
trained-model performance on a different sensor or resolution.

## Supply the model and weight assets

The wheel contains code, not weights, personal manifests, prepared images or
historical asset registries. Supply an explicit local checkpoint. It is not yet public; it will be published under
CC BY 4.0 ([WEIGHTS-LICENSE.md](../WEIGHTS-LICENSE.md)). No command here fetches it.
The measured step-6000 checkpoint has SHA-256
`5c7a7c44d29b3c75bd69f59809fb1b63a039313bec021d897ff3876d2cea42a5`.
The loader checks the supplied hash before deserialization and rejects different
keys, shapes, dtypes or nonfinite coefficients.

EE inference also needs a completed `native_ee_featurecollection_weights_v2`
manifest for those coefficients, pointing to assets the caller can read. You may
supply a retained manifest after verifying its provenance and permissions. The
loader checks all 189 record fingerprints, shapes, dtypes, biases and the
source-checkpoint identity locally; it does not check remote values or readiness.
For the measured checkpoint these are 36,747,964 float32 inference values. Its
12 BatchNorm counters are checked but not used by the EE graph.

To create your own manifest, initialize EE explicitly and invoke the existing
weight export utility. This is a separate upload operation, not part of stage
preparation. Choose a fresh prefix inside an asset folder you have already
created and a local manifest path; the utility does not overwrite assets. It
waits for its bounded exports and returns a ready manifest only after every
recorded shard completes. The local binding helper adds the unchanged Swin
record-list hash and checkpoint identity, then validates every record against the
local coefficients. It does not promote an unfinished manifest to ready.

```python
import json
from pathlib import Path
import ee
import google.auth
from tensor_ee.swin import load_swin_t224_parameters
from tensor_ee.swin_workflow import weight_records, bind_weight_manifest
from tensor_ee.weight_assets import export_weights

checkpoint = Path('/your/model/step6000_last.pth')
expected_sha = '5c7a7c44d29b3c75bd69f59809fb1b63a039313bec021d897ff3876d2cea42a5'
local = load_swin_t224_parameters(checkpoint, expected_checkpoint_sha256=expected_sha)
records = weight_records(local)  # exact existing keys/chunks/coefficient precision
credentials, _ = google.auth.default(scopes=[
    'https://www.googleapis.com/auth/earthengine',
    'https://www.googleapis.com/auth/cloud-platform',
])
ee.Initialize(credentials=credentials, project='YOUR_PROJECT')
manifest_path = Path('/your/run/weights.json')
manifest = export_weights(records,
    'projects/YOUR_PROJECT/assets/YOUR_EXISTING_FOLDER/fresh_weights',
    manifest_path, max_active=2)
assert manifest['status'] == 'ready'
assert all(shard['status'] == 'COMPLETED' for shard in manifest['shards'])
manifest = bind_weight_manifest(local, manifest)
manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
```

`weight_records` requires local parameters loaded without a manifest; it cannot
re-export deferred server arrays. To inspect storage after an explicit upload,
`tensor_ee.weight_assets.verify_asset_rows(records, manifest, result_path)` makes
read-only EE requests for deterministic sample rows from every record. This is a
sampled storage/order check, not a full checksum or output parity test. Remote
asset content and permissions remain caller responsibilities.

## Declare inputs and destinations

Create `resources.json` with exactly these fields. Replace the project, input
image ID, local paths, grid origin and all three fresh destinations. Compute
`manifest_file_sha256` from the exact bound file bytes, for example
`hashlib.sha256(Path('/your/run/weights.json').read_bytes()).hexdigest()`.
Checkpoint and manifest paths can be absolute or relative to the resource file.
No source scene, project, grid, checkpoint directory or output prefix is assumed.

```json
{
  "schema_version": 1,
  "project": "your-project",
  "checkpoint": "/your/model/step6000_last.pth",
  "checkpoint_sha256": "5c7a7c44d29b3c75bd69f59809fb1b63a039313bec021d897ff3876d2cea42a5",
  "manifest": "/your/run/weights.json",
  "manifest_file_sha256": "REPLACE_WITH_64_LOWERCASE_HEX_FILE_SHA256",
  "input_bands": ["VV", "VH"],
  "output_bands": ["logit_0", "logit_1"],
  "grid": {
    "crs": "EPSG:32651",
    "transform": [10, 0, 287680.95335071225, 0, -10, 2640838.8405420803],
    "width": 224,
    "height": 224
  },
  "input": {
    "mode": "segwater_v3_db",
    "asset_id": "COPERNICUS/S1_GRD/YOUR_EXACT_ACQUISITION_ID"
  },
  "output": {"mode": "water"},
  "destinations": {
    "A": "projects/your-project/assets/YOUR_EXISTING_FOLDER/fresh_skips",
    "B": "projects/your-project/assets/YOUR_EXISTING_FOLDER/fresh_c4",
    "C": "projects/your-project/assets/YOUR_EXISTING_FOLDER/fresh_water"
  }
}
```

Input IDs may name an explicit catalogue image or caller asset. There is no
collection search or acquisition selection. Destinations must be distinct,
belong to the declared project, and differ from the inputs. Exports set
`overwrite=false`; create their containing folders yourself.

`segwater_v3_db` interprets the two input names positionally as VV then VH in dB.
It captures the original per-channel mask, fills each missing dB channel with
its training mean **before** quantization, and produces a dense model tensor.
The retained v3 operation order uses float32 after each operation:
`clip(round_even((db + 55) / 0.05), 1, 65535)`, decode `code * 0.05 - 55`, then
standardize with means `(-15.435106, -24.856323)` and standard deviations
`(6.517790, 9.117346)`. Filled channels therefore have small nonzero normalized
values. Numeric zero or −55 dB is not guessed to be nodata. This differs from
the U-Net recipe's exact-zero fill after normalization.

For already prepared tensors, replace `input` and `output` with:

```json
"input": {
  "mode": "prepared",
  "asset_id": "projects/your-project/assets/YOUR_PREPARED_IMAGE",
  "validity_asset_id": "projects/your-project/assets/YOUR_PER_CHANNEL_VALIDITY"
},
"output": {"mode": "logits"}
```

Both images must have the two declared input bands in the given order, aligned
to the declared grid. Prepared values assert dense finite float32 values with
normalization and fill already applied. Validity asserts separate per-channel
0/1 values, with masked values treated as zero. Stage preparation does not scan
remote pixels to verify these properties. Retrieve the exact prepared pixels
and validity when checking a new input. Dataset fp16 normalized inputs can be
widened to float32 without re-quantizing them; supply their validity separately.

`output.mode="logits"` returns only the two named raw scores. The named water
recipe also permits `output.mode="water"`, producing those logits followed by
`water_probability`, `water_mask`, and `coverage`. It preserves the existing
stable two-class softmax; water is class 1 and the threshold is inclusive
`probability >= 0.5`. These computational layers stay dense inside the chip.
`coverage` is the intersection of the original source channel masks; apply it
explicitly when displaying observed water pixels. It is not a receptive-field
validity calculation. Prepared-input mode has no inferred sensor or water
semantics and accepts only logits output.

## Prepare, submit and resume one stage

For an offline structural check, use a separate fresh local directory:

```bash
python -m tensor_ee.swin_workflow --config resources.json --state offline-check \
  --stage A --action prepare --offline
```

This uses the EE SDK's canned API signatures, checks the local checkpoint and
manifest, and writes the stage-A request without authentication or cloud calls.
Offline requests are marked and cannot be submitted. No numerical parity is
claimed by graph construction.

For actual execution, explicitly select credentials (`adc` for application
default credentials, or `ee` for `earthengine authenticate`) and a fresh state
directory. Preparation builds and measures the deferred request; it submits no
export. Inspect the printed bands/grid/destination and the saved request first.
Then submit that exact request and ask for status until it reports `COMPLETED`:

```bash
python -m tensor_ee.swin_workflow --config resources.json --state run \
  --stage A --action prepare --credentials adc
python -m tensor_ee.swin_workflow --config resources.json --state run \
  --stage A --action submit --credentials adc
python -m tensor_ee.swin_workflow --config resources.json --state run \
  --stage A --action status --credentials adc
```

Repeat these three commands with `--stage B` after A completes, then with
`--stage C` after A and B complete. Each status command performs one read; there
is no automatic polling or job loop. Keep one writer per state directory.

| Stage | Persisted grid | Ordered output |
| --- | --- | --- |
| A | Input grid ×4, 56 × 56 | `c1_0..95`, `c2_0..191`, `c3_0..383`, `valid_fraction` (673 bands) |
| B | Input grid ×32, 7 × 7 | `c0..767`, raw C4 (768 bands) |
| C | Original 224 × 224 input grid | Two configured logits, or the five water recipe bands |

A stores C2/C3 as exact nearest copies on the C1 grid. B reads saved A, samples
C3 at the existing phase and runs stage 4. C reads saved A and B, performs
feature norms, tile-local PSP, FPN, grouped-double fusion and the head. No stage
rebuilds its predecessors. `valid_fraction` is per-patch `k/32` across the 4×4×2
input validity values; it does not mask logits or represent output confidence.
Grid origins are upper-left pixel corners; stride grids preserve that corner.

`state.json` and `request_A.json`, `request_B.json`, `request_C.json` persist the
resource/checkpoint/manifest identity, ordered bands/grids, request sizes and
fingerprints, caller request UUIDs, and the returned EE operation/task IDs.
Changed configuration or request bytes stop resume. Preparation reuses an
already saved matching request. Saved predecessor requests must still match,
and their recorded operations must be `COMPLETED` before proceeding. This
assumes the completed caller assets remain unchanged and readable.

A submission error or lost response records `UNKNOWN` and stops. Repeating
`submit` never replays it. Inspect your EE tasks externally, identify the exact
operation, then bind it explicitly:

```bash
python -m tensor_ee.swin_workflow --config resources.json --state run \
  --stage A --action status --credentials adc \
  --operation-name projects/your-project/operations/EXACT_OPERATION_ID
```

Recovery checks project, export type, exact description and destination returned
by that one status read. It performs no task-list scan. A caller request UUID is
not an operation ID. `UNKNOWN`, failed or cancelled remote states stop; choose
new destinations and a new local run directory for a deliberate new execution.

## Use the adapter directly

For application code, initialize EE yourself, load the explicit checkpoint and
manifest, and supply prepared `Raster`s with matching grids and ordered bands:

```python
import json
from pathlib import Path
import ee
from tensor_ee import Grid, Raster
from tensor_ee.swin import SwinT224Adapter, load_swin_t224_parameters

# ee.Initialize(credentials=YOUR_CREDENTIALS, project=YOUR_PROJECT) first.
grid = Grid('EPSG:3857', (20, 0, 1000, 0, -20, 2000), 224, 224)
bands = ('first', 'second')
parameters = load_swin_t224_parameters(Path('/your/model/step6000_last.pth'),
    expected_checkpoint_sha256='5c7a7c44d29b3c75bd69f59809fb1b63a039313bec021d897ff3876d2cea42a5',
    manifest=json.loads(Path('/your/run/weights.json').read_text()))
adapter = SwinT224Adapter(input_bands=bands)
a = adapter.skips(Raster(ee.Image(YOUR_PREPARED_ID), bands, grid),
    validity=Raster(ee.Image(YOUR_VALIDITY_ID), bands, grid), parameters=parameters)
# Explicitly export a and wait for completion; then read that saved asset:
saved_a = Raster(ee.Image(YOUR_SAVED_A_ID), a.bands, a.grid)
b = adapter.c4(saved_a, input_grid=grid, parameters=parameters)
# Explicitly export b and wait; then read saved A/B:
saved_b = Raster(ee.Image(YOUR_SAVED_B_ID), b.bands, b.grid)
c = adapter.logits(saved_a, saved_b, input_grid=grid, parameters=parameters)
```

`load_swin_t224_model(checkpoint, expected_checkpoint_sha256=...)` provides the
fixed local float32 eval reference model. It does not fetch weights, forward the
model or initialize EE. Use identical prepared input pixels, channel order,
row/column orientation and grid when comparing it to exported raw logits; see
[API comparison guidance](api.md). For the retained ORT reference use its fixed
batch of 32, even when EE evaluates individual nominated chips.

## Evidence and limits

[T01 evidence](evidence/t01-live.json) records live three-stage regression on the
original catalogue chip, `hpo_val_00` and near-margin `hpo_val_04`: maximum
absolute logit errors were `1.30e-5`, `4.08e-6`, and `1.46e-5`, with zero argmax
or inclusive class-1 threshold disagreements. The target was `1e-4`; these are
numerical execution checks, not accuracy measurements. Retained exact inputs,
checkpoint/ORT identity and independent review are in that report and the
T01 handoff (development archive: `docs/tasks/01-swin-boundaries.md`).

That bounded run used six A/B exports with at most two active, then three C
pixel evaluations. A/B request bodies were about 0.32–0.80 MB; observed export
run times ranged from 103 to 250 seconds. Those are identified observations,
not throughput or cost guarantees. The commands here export C as an asset
rather than performing the validation runner's C pixel read; this packaging
lane ran no new cloud jobs. Requests are measured and rejected at 9.5 MB.

[Packaging evidence](evidence/t05-packaging.json) records isolated wheel imports,
exact real-checkpoint model loading, retained-manifest binding and all three
offline saved-asset graphs. It also records exact graph equality between this
water workflow and the historical demo for preprocessing, stage transport,
probability/tie behavior and coverage. At T05 the wheel included the verified T01
runtime closure and the public `tensor_ee` modules. Since L06 it contains only the
`tensor_ee` package, which now holds that runtime
([runtime imports](evidence/l06-runtime-imports.json)). Research runners, scenes,
registries, results and personal manifests are not installed.
