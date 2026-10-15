# C1 export plan: the capstone Swin-UPerNet, end to end

Status: **approved by the user on 2026-10-14 and run; passed.** The lead ran every step as written, on `a1bf50a`, in `ORIGINAL_PROJECT`:
- **Input check:** 65,536 of 65,536 valid.
- **Upload:** 13 table exports, all COMPLETED on the first attempt (94 to 135 s each, 13.3 min overall); the manifest matches the plan.
- **Prepare:** the live request equals the planned one and this document's hash (`0f584e83…`, 887,493 bytes).
- **Export:** task `UGT4OSVOQHPOJ764HM474TMD`, COMPLETED on the first attempt in about 112 s, 49.1 batch EECU-seconds.
- **Verify:** passed. The largest absolute logit difference from PyTorch is 9.78e-6 over 196,608 logits, with 65,536 of 65,536 argmax agreeing.

> **Public record (S03).** The `--project` value in this document was the original project ID; it reads `ORIGINAL_PROJECT` here. The request bytes and SHA-256 recorded below were computed with the original ID: `tests/test_capstone.py` reproduces the bytes with a bare project name of the original's length, which it infers from the recorded bytes, and the recorded SHA-256 is the archive's.

Evidence: [c1-capstone.json](c1-capstone.json). The weight manifest is `data/manifests/capstone_c1_weights_manifest.json`. The plan below is unchanged.

Every number below is the output of
`PYTHONPATH=. python tools/capstone.py --plan --project ORIGINAL_PROJECT` (zero cloud calls),
with earthengine-api 1.7.45, torch 2.14.0, timm 1.0.30, SMP 0.5.0, NumPy 2.5.3, Python
3.14.6, on Darwin arm64.
- **The same everywhere.** The random weights come from the harness's platform-independent
  integer generator, so their hashes, the shard layout and the asset names are the same on
  every platform.
- **Client-dependent.** Serialized sizes and the request's hash can change with the
  earthengine-api version.
- **Platform-dependent.** The reference's float results can change in their last digits
  with torch and the platform.

`tests/test_capstone.py` checks this document against `--plan`.

## Common setup

| Item | Value |
|---|---|
| Project | `ORIGINAL_PROJECT` |
| Credentials | Application Default Credentials, scopes `earthengine` and `cloud-platform` (`tools/live_fixtures.live_session`); nothing is saved |
| Network | The IPv4-first network shim the lead uses for Earth Engine calls |
| Code | `lib/c1-capstone-run` at the reviewed revision, from a checkout with `PYTHONPATH=.`; the environment above |
| Compute | `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2`; the tool sets `torch.set_num_threads(2)` |
| State directory | a new empty directory outside the repository, passed as `--state DIR` to every live command |
| Concurrency | One command at a time; at most 2 Earth Engine tasks active |

## The model

| Item | Value |
|---|---|
| Configuration | `SwinUPerNetConfig(window=8, patch=4, bands=("VV", "VH", "VVminusVH"), embed_dim=48, depths=(2, 2, 2, 2), num_heads=(2, 4, 8, 16), stage_dims=(48, 96, 192, 384), decoder=64, groups=4, sizes=(1, 2, 3, 6), classes=3, head_factor=4)` |
| Shifts | 0, 4 in stages 1–3; 0, 0 in stage 4 (one window), equal to timm's |
| Reference | SMP 0.5.0 `UPerNet("tu-swin_tiny_patch4_window7_224", encoder_depth=5, encoder_weights=None, decoder_channels=64, decoder_use_norm="batchnorm", in_channels=3, classes=3, upsampling=4, img_size=(256, 256), embed_dim=48, depths=(2, 2, 2, 2), num_heads=(2, 4, 8, 16), window_size=8, drop_rate=0, proj_drop_rate=0, attn_drop_rate=0, drop_path_rate=0)`, eval mode, timm's explicit attention path, as Milestone 12 built its reference |
| Weights | the default initialization's scales (timm's and SMP's), drawn from the harness's SplitMix64 hash with seed `2610`; then seeded perturbations of the parameters that it sets to constants (below) |
| State dict | 199 tensors, SHA-256 `2caabc42d86dc172dd1a46adc70d9836a85801c0c7d25db5415850429d60cd69` |
| Parameters | 5,875,791: 5,124,588 in the encoder and patch embedding, 751,203 in the decoder and head |
| Records | 153 v2 weight records; listing hash (`manifest_hash`) `ead153be2306d1c20050481ca781480c27e443d2e419f8df3f833fb593287ab9` |

**The weights** (`tools/capstone.py`, `initial_state`). Every tensor is uniform with the
bound or the variance of the default initialization:
- **Encoder.** timm's trunc-normal std 0.02 for every Linear weight and relative-position
  table. PyTorch's Conv2d default for the patch embedding: U(−1/√48, 1/√48) for its weight
  and its bias.
- **Decoder.** SMP's Kaiming uniform for the convolutions: U(−√(6/fan_in), √(6/fan_in)).
- **Head.** SMP's Xavier uniform: U(−√(6/67), √(6/67)).

The values are U on the 24-bit dyadic grid of [−1, 1), from the harness's SplitMix64 hash
of the seed and the tensor's name (`tools.fixtures.core.dyadic`). Each is multiplied by its
bound, whose square root is taken in integers (`math.isqrt`), then cast to float32. Every
operation is correctly rounded, with no platform random generator and no libm, so the
same weights come out on every platform.

**The perturbations** (a deviation from a literal "default initialization"). Default
initialization sets every bias to 0, every LayerNorm and BatchNorm scale and shift to 1
and 0, and every BatchNorm running mean and variance to 0 and 1. Those paths would then
not be tested: a wrong bias row or BatchNorm statistic would give the same logits. So, with
U of unit variance from the same generator:
- norm and BatchNorm scales are 1 + 0.1 U, and their shifts 0.05 U;
- BatchNorm running means are 0.1 U, and running variances 1 + 0.2 U (0.65 to 1.35);
- Linear and Conv2d biases get + 0.02 U.

Setting `REFERENCE_PERTURBATION` to zeros gives the literal default initialization's
constants; the hashes and asset names would then change.

**The reference on the stand-in input** (a deterministic input in the expected range,
`stand_in_input`):
- Logits run from −4.4095 to 3.9221.
- The reference's own float32 against float64 (same weights) differs by at most 9.2e-6
  (RMSE 1.89e-6), a tenth of the 1e-4 target.
- Milestone 12's check passes: the forward rebuilt from the installed submodules equals the
  model's own at all 21 captured tensors, bit for bit.

`--verify` records all three again on the real input.

## The input and the grid

| Item | Value |
|---|---|
| Scene | `COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20240103T100134_20240103T100159_051941_06469E_B0BD` (the Milestone 12 and demo scene) |
| Prepared bands | `VV`, `VH` (dB, as the collection stores them) and `VVminusVH` = VV − VH, each × 0.1, float32; nearest-neighbour onto the grid |
| Grid | `EPSG:32651`, 10 m, 256 × 256, transform `(10, 0, 287520, 0, -10, 2641000)`: a whole-metre origin over Sun Moon Lake |
| Milestone 12 chip | origin (287680.95, 2640838.84), 224 × 224: inside the grid, with 160.95 m to spare on the left, 159.05 m on the right, 161.16 m at the top and 158.84 m at the bottom |
| Model grid | the abstract `TokenGrid(256, 256, 2.5)` (`config.model_grid`), onto which the input is relabelled exactly (`reproject` + `changeProj`, as the Swin-T224 adapter does), and from which the logits are relabelled back |

No accuracy claim is made: the weights are random.

## The commands, in order

```sh
S=<a new empty directory outside the repository>
PYTHONPATH=. python tools/capstone.py --plan --project ORIGINAL_PROJECT
PYTHONPATH=. python tools/capstone.py --check-input --project ORIGINAL_PROJECT --state $S
PYTHONPATH=. python tools/capstone.py --upload --project ORIGINAL_PROJECT --state $S
PYTHONPATH=. python tools/capstone.py --prepare --project ORIGINAL_PROJECT --state $S
PYTHONPATH=. python tools/capstone.py --submit --project ORIGINAL_PROJECT --state $S
PYTHONPATH=. python tools/capstone.py --status --project ORIGINAL_PROJECT --state $S
PYTHONPATH=. python tools/capstone.py --verify --project ORIGINAL_PROJECT --state $S
```

Each live command initializes Earth Engine once. Initialization makes the client's own
reads: the Cloud API discovery document (fetched twice), one `algorithms.list`, and the
public catalogue of deprecated assets. That is six initializations in all, more only if a
step is re-run (below).

| Step | Earth Engine calls | Refuses when |
|---|---|---|
| `--check-input` | 1 `getInfo`: a `reduceRegion` count of all pixels and of valid prepared-input pixels on the grid (a 3,551-byte expression; returns 2 integers) | — (records the counts; exit 1 unless both are 65,536) |
| `--upload` | 13 table exports (`Export.table.toAsset`, `overwrite=False`), at most 2 active; `getTaskStatus` every 30 s for each active task. Re-run on an interrupted upload, it resumes: it polls the recorded tasks, makes 1 `getAsset` for a shard recorded "starting" without a task ID, and dispatches what never started | `--check-input` has not passed with 65,536 valid pixels; the state directory's manifest belongs to other weights or records a FAILED shard. A completed upload is not repeated |
| `--prepare` | none beyond initialization: the graph is built locally from the uploaded manifest; `ee.data.newTaskId` makes the request ID locally (a UUID) | the manifest is not the completed upload of these weights; the request differs from the one built in the same session from the planned stand-in manifest; the request is over 10,485,760 bytes; a submission is already recorded |
| `--submit` | 1 `exportImage` with the request ID `--prepare` saved | not prepared; already SUBMITTED; the saved request changed |
| `--status` | `getTaskStatus` every 30 s until the export is terminal | no submitted task |
| `--verify` | 1 `getAsset` (the result's metadata); 1 `computePixels` of the prepared input; 1 `computePixels` of the result | the export is not recorded COMPLETED |

**The request's pass criterion is the live one.** `--prepare` refuses unless the request
built from the uploaded manifest equals the one built, in the same live session, from the
planned stand-in manifest. It prints and records both, and also the SHA-256 that `--plan`
gives offline (the one below). That offline SHA-256 is built against ApiTestCase
signatures, so it may differ from the live one without meaning anything.

## Every task

**14 tasks: 13 table exports, then 1 image export.** At most 2 are active at once. The
image export starts only after every table export is COMPLETED and `--prepare` has passed.

| # | Kind | Description | Destination (`projects/ORIGINAL_PROJECT/assets/…`) | Rows | Collection bytes |
|---:|---|---|---|---:|---:|
| 1 | table | `native_unet_weights_part_0000` | `native_ee_capstone_c1_ead153be2306_w_part_0000` | 3,697 | 5,887,889 |
| 2 | table | `native_unet_weights_part_0001` | `native_ee_capstone_c1_ead153be2306_w_part_0001` | 1,978 | 6,488,761 |
| 3 | table | `native_unet_weights_part_0002` | `native_ee_capstone_c1_ead153be2306_w_part_0002` | 1,930 | 6,512,924 |
| 4 | table | `native_unet_weights_part_0003` | `native_ee_capstone_c1_ead153be2306_w_part_0003` | 956 | 6,825,065 |
| 5 | table | `native_unet_weights_part_0004` | `native_ee_capstone_c1_ead153be2306_w_part_0004` | 1,201 | 6,763,588 |
| 6 | table | `native_unet_weights_part_0005` | `native_ee_capstone_c1_ead153be2306_w_part_0005` | 1,202 | 6,766,693 |
| 7 | table | `native_unet_weights_part_0006` | `native_ee_capstone_c1_ead153be2306_w_part_0006` | 646 | 6,938,366 |
| 8 | table | `native_unet_weights_part_0007` | `native_ee_capstone_c1_ead153be2306_w_part_0007` | 928 | 6,844,287 |
| 9 | table | `native_unet_weights_part_0008` | `native_ee_capstone_c1_ead153be2306_w_part_0008` | 1,201 | 6,761,330 |
| 10 | table | `native_unet_weights_part_0009` | `native_ee_capstone_c1_ead153be2306_w_part_0009` | 1,201 | 6,764,957 |
| 11 | table | `native_unet_weights_part_0010` | `native_ee_capstone_c1_ead153be2306_w_part_0010` | 435 | 7,012,731 |
| 12 | table | `native_unet_weights_part_0011` | `native_ee_capstone_c1_ead153be2306_w_part_0011` | 511 | 6,938,639 |
| 13 | table | `native_unet_weights_part_0012` | `native_ee_capstone_c1_ead153be2306_w_part_0012` | 525 | 5,631,462 |
| 14 | image | `native_ee_capstone_c1_ead153be2306_logits` | `native_ee_capstone_c1_ead153be2306_logits` | — | — |

- **Weights.** 13 tables hold 16,411 feature rows and 86,136,692 collection bytes in all.
  Each shard is under `export_weights`' 8,000,000-byte limit. The shards are
  `upload_chunks`' measured partition: `--upload` writes the manifest with these asset IDs,
  and `--prepare` requires the graph built from it to equal the planned one.
- **The image.** `Export.image.toAsset` of the whole model on the grid: region the grid,
  `crs` `EPSG:32651`, `crsTransform` `[10, 0, 287520, 0, -10, 2641000]`,
  `maxPixels` 65,536, pyramiding `sample`, `overwrite=False`.
  - Bands: `logit_0`, `logit_1` and `logit_2` (float32), and `valid` (1 where all three
    prepared bands are valid).
  - The request is 887,493 bytes (offline SHA-256
    `0f584e83e1c5c1a07cec2501029a8bfba0b42e64b3265fe98a00bfb0a2afda70`), against Earth
    Engine's 10,485,760-byte limit.

**Evaluated constants** (the blocks' `METADATA` cost functions at window 8, 64 rows, and
bins (1, 2, 3, 6)):

| Part | Values |
|---|---:|
| Patch embedding | 11,520 |
| Encoder (8 blocks, 3 merges) | 6,470,844 |
| PSP | 545,280 |
| FPN | 273,408 |
| Fusion | 163,840 |
| Head | 384 |
| Coefficient tables: pooling, matrix resizes, window resizes | 850; 3,200; 9,740 |
| **Total** | **7,479,066** |
| Not in the cost functions, but in Milestone 5's count: the per-head relative-position biases (245,760) and the shifted-window masks' boundary arrays (24,576) | 270,336 |
| **Total, against the 24 M cap** | **7,749,402 (32.3 %)** |

- **Cross-checks.** The PSP's, the FPN's and the fusion's counts equal their configured
  constants' own `embedded_values` (545,280; 818,688 with the PSP; 163,840).
- **The brief's formula.** 5,849,724 weight values plus 26,067 per-row parameters × their
  rows, 1,615,552, gives 7,465,276. That is the total without the coefficient tables.
- **One graph.** The model fits one export, so nothing is staged.

## Every read

| Step | Read | Size |
|---|---|---|
| `--check-input` | `getInfo` of a `reduceRegion` count | a 3,551-byte expression; 2 integers back |
| `--upload`, resumed | `getAsset` of a shard recorded "starting" without a task ID | its metadata; at most once per such shard |
| `--verify` | `getAsset` of the result | its metadata (band names, types, grid) |
| `--verify` | `computePixels` (`NUMPY_NDARRAY`) of the prepared input: its 3 bands (masked pixels as 0) and their joint mask | 256 × 256 × 4 float32: 1,048,576 bytes |
| `--verify` | `computePixels` (`NUMPY_NDARRAY`) of the result: its 4 bands and their joint mask | 256 × 256 × 5 float32: 1,310,720 bytes |

Both `computePixels` reads exceed the standing budget's 64 × 64 raster cap, which is why
this plan needs approval.

## Polling caps

- Every active task is polled every 30 s: at most 60 polls in 30 minutes.
- **The cap is 30 minutes per task**, as in L06c: 30 minutes after a table export started,
  or after the image export was submitted.
  - Run `--status` right after `--submit`: its cap counts from the submission.
  - A resumed `--upload` counts each task's 30 minutes from the resume.
- At the cap the tool stops polling and records `POLLING_CAP` in the state directory. The
  Earth Engine task continues; nothing is cancelled without the user's approval. The lead
  records the task state and reports to the user.
- In all, at most 13 × 60 polls for the tables and 60 for the image, per invocation.

## What `--verify` checks

`--verify` runs the reference, in float32 and float64, on the input read back from Earth
Engine, and writes `docs/evidence/c1-capstone.json`. Pass criteria:
- **Logits.** The largest absolute difference from the reference is at most 1e-4 (the
  unchanged target). The record also gives the MAE, the mean and the RMSE.
- **Observed, not criteria.** The number of bit-identical values, and the argmax agreement
  over the 65,536 pixels. On the stand-in, classes 0 and 1 win 53,475 and 12,058 pixels and
  class 2 only 3, so a high agreement is weak evidence.
- **Input.** Float32, and fully valid and finite.
- **Result.**
  - Every band is float32, with the bands in order `logit_0`, `logit_1`, `logit_2`,
    `valid`.
  - It is unmasked, and `valid` is 1 everywhere.
  - The logits are finite.
  - The asset's grid is the planned one.
- **Reference.** Milestone 12's check passes on that input.

**`periodic_extension` at window 8.** The SW-MSA blocks' periodic extension runs live at
window 8 for the first time in this export (C0/C0b review), so `--verify` is its check.

**Head width 24.** The attention heads are 24 wide (48/2, 96/4, 192/8, 384/16), a value no
status names; `--verify` is its check too.

## What is left behind

- In `ORIGINAL_PROJECT`: 13 table assets (`…_w_part_0000` to `…_w_part_0012`) and 1 image
  asset (`…_logits`), under the prefix
  `projects/ORIGINAL_PROJECT/assets/native_ee_capstone_c1_ead153be2306`. They stay. The tool
  deletes nothing; deleting them is the user's decision.
- Locally: the state directory, holding `state.json`, the weight manifest
  `manifest.json` and the image request `request.json`.
- In the repository: `docs/evidence/c1-capstone.json`, which the lead commits with the
  results.

## Stop rules

- **A lost submit response** (`--submit` exits with an error and records UNKNOWN or
  SUBMITTING): re-run `--submit`. It re-sends the same request ID, which Earth Engine takes
  as the same task.
- **An interrupted upload** (a polling cap, a lost connection, a crash): re-run `--upload`.
  It resumes from the manifest in the state directory, and never re-exports a shard that
  started.
  - **Caveat.** If the process died in the moment between a table export's start and the
    save of its task ID, the shard is recorded "starting" with no task ID and no asset yet.
    The resume then dispatches it again, and of the two exports with the same asset ID the
    one that finishes second fails. If that is the recorded one, the upload stops on a
    FAILED shard (below).
- **A table export FAILED or CANCELLED:** the upload stops, its state records the error, and
  nothing is retried. The shard names are immutable, so a retry needs the user's decision.
- **A polling cap reached:** stop and report, as above.
- **The image export fails**, for example with "Encoded object too large." or a memory
  limit: report. If the graph does not fit one export, shrink the model; do not add staging
  (the brief).
- **`--verify` fails:** the record names the worst value by row, column and class, and the
  failed checks.
