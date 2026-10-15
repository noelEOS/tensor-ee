# X06 runner: SegFormer-B0 live plan

2026-10-15. **Approved by the user in advance, 2026-10-15, within this scope:** one chip; SMP
SegFormer-B0 converted by X03; the planner's default 4-stage plan; random weights uploaded as
table assets; the seeded input; comparison with the PyTorch reference
(decision log (development archive: `docs/library-plan.md`), X06 brief (development archive: `docs/tasks/X06-runner.md`), R5). Everything below
stays inside that scope; the last section lists what would need a new approval. Written by the
X06 implementer, who runs nothing live; the lead runs it. An addendum records what ran.

**14 tasks: 9 weight tables and 5 images (the seeded input's upload and the 4 stages); 14 new
assets.** The staged run's input asset was deleted on 2026-10-14, so the seeded input needs one
upload.

## Question

Phase 3 can now convert a model, plan its stages and run them. This run asks whether the whole
chain works in Earth Engine with nothing chosen by hand: SMP SegFormer-B0 converted by
[`convert`](../../tensor_ee/converter/convert.py) (X03), cut by [`plan`](../../tensor_ee/converter/plan.py)
(X05) under its default limits, and executed by the [runner](../../tensor_ee/converter/run.py)
(X06) for one chip, and whether its logits agree with PyTorch. A failure (memory, time, payload,
a check) is an answer, not a reason to retry.

## What runs

- **Model and weights.** SMP 0.5.0 `Segformer("mit_b0", encoder_weights=None, in_channels=2,
  classes=2).eval()`, its weights the seeded constructor state of the staged run
  (`tools/fixtures/segformer_b0_fullmodel.py`, `seeded_state(20261014)`, state SHA-256
  `3c7d9cb7c85c473bf54af1f8cf748d63b4d1243017614aaf6b0678867d357a23`): random weights, no
  download. This is a numerical check, not an accuracy assessment.
- **Input.** The staged run's seeded chip, `representative_boundaries(20261014)["input"]`
  (224 × 224 × 2 float32, SHA-256
  `46e1a72700e65f8bb533ee51f75f3a56487f5517b5856bb59b021546475e1ad1`), as literal bands `b0`, `b1`
  on the source grid EPSG:3857, affine `[2.5, 0, 1000, 0, -2.5, 2000]`, 224 × 224 (the sizing
  tool's `_boundary_image("input", "inline", ...)`). The runner exports it first, as the run's
  first image task, so that every stage reads saved images, as the planner measured them.
- **Graph.** `convert(model, (1, 2, 224, 224), SOURCE_GRID, name="segformer_b0")`. Its SHA-256 must
  be `568c92c33a639257c43578d3c7773f91f98497d4c12191319ff2565a23b0f39a`, the hand-written
  [`block_graph.json`](../../tests/reference/segformer_b0/block_graph.json) (X03's gate); otherwise
  stop.
- **Plan.** `plan(graph, Limits(), ...)`: the defaults (24,000,000 embedded constants, 10,000,000
  request bytes, 1,024 bands, at most 2 EXTENT nodes per stage), the weights read from the tables,
  the boundaries named `input`, `C1`, `C2`, `C3`, `logits`. It must give these four stages
  (X05's default plan, plan SHA-256 `6d8fe1fd...` with the placeholder namespace; another
  namespace changes only the recorded bytes); otherwise stop:

| Stage | Reads → writes | Embedded constants | EXTENT | Request bytes | Weight tables read | Evidence before this run |
|---|---|---:|---:|---:|---|---|
| stage1 | `input` → `encoder.norm1` (C1) | 342,240 | 2 | 2,062,548 | parts 0000, 0001 | **New**: Codex's coarse stage 1, measured but never submitted (over that plan's 2 MB cap, so it was split) |
| stage2 | C1 → `encoder.norm2` (C2) | 614,720 | 2 | 1,166,058 | parts 0000, 0001 | The G1-ext target `stage2`, which ran live in the staged run |
| stage3 | C2 → `encoder.norm3` (C3) | 1,828,640 | 2 | 874,385 | parts 0000 to 0004 | The G1-ext target `stage3`, which ran live in the staged run |
| stage4 | C1, C2, C3 → `logits` | 3,834,478 | 2 | 464,261 | parts 0000, 0004 to 0008 | **New**: the staged run's stage 4 and decoder in one graph |

The input export's request is 1,222,596 bytes. Every byte count is the complete
`ExportImageRequest`, measured by the library's export-request builder
([`request.py`](../../tensor_ee/converter/request.py)) with the sizing tool's placeholder namespace,
and equal to the tool's own measurement (`tests/test_x06_runner.py`). `prepare` measures them
again with the run's namespace and refuses any request over 10,000,000 bytes. Each stage is
whole-chip (EXTENT nodes), so the run is one export per boundary, never tiled.

## Sequence

All in `YOUR_EE_PROJECT`, namespace `projects/YOUR_EE_PROJECT/assets/x06_segformer_b0`, with
explicit application-default credentials (initialized with that project: the runner pins
operations to it), the IPv4-first resolver, a 900 s socket timeout, SDK retries off
(`ee.data.setMaxRetries(0)`; `run`, `resume` and `recover` set it too) and no workload tag. Every
process that initializes Earth Engine is recorded with its one initialization. Every live step
outside the runner runs under a `CallGuard` (`native_ee/converter/run.py`) with the caps below,
its counts kept in a file in the run storage (`path=`), so they add up across processes; a call
past a cap is refused, not made (`CallCapReached`).

**One process per run directory.** `run`, `resume` and `recover` hold the directory's lock file
(`run.lock`, with the process's ID and start time) and refuse to start while another holds it;
a lock left by a process that died is removed by hand only after checking that the process is
gone. The prepared directory is never copied to run it again: a copy has the same request IDs
and destinations.

1. **Offline.** Build the seeded state and input and check both fingerprints above; convert and
   check the graph's SHA-256; split the 128 weight records with the library's chunking
   (`upload_chunks`, 8,000,000 bytes) and the sizing tool's complete-body check
   (`plan_weight_shards`): it must give 9 shards, every complete `exportTable` body at most
   8,000,000 bytes (at most 6,936,189 offline, with `projects/example-project/assets/x06_segformer_b0`
   as the namespace); otherwise stop.
2. **Weights** (live), all under one guard, `CallGuard({"getAsset": 11, "exportTable": 9,
   "getOperation": 120, "getTaskStatus": 20}, path=...)` (`export_weights` has no caps of its
   own). Check that the 9 table destinations `..._weights_part_0000` to `_0008` do not exist
   (`getAsset`, HTTP 404), then upload them with [`export_weights`](../../tensor_ee/weight_assets.py)
   (`max_bytes=8_000_000`, `max_active=2`, `poll_seconds=30`): it polls its own tasks with
   `getOperation`. Its manifest is written after each task; after an interruption (or a cap),
   `export_weights(..., resume=True)` polls the tasks it recorded (`getTaskStatus`) and checks
   each shard recorded "starting" with one `getAsset`, uploading it if absent. A failed shard
   stops the run.
3. **Offline.** Bind the ready manifest to the state: add the state's SHA-256 as
   `source_checkpoint_sha256` and the record listing's `manifest_hash` (as `planned_manifest`
   computes it), then `with_manifest` gives the `ServerArray` of every `state_dict` name (the
   sizing tool's `asset_state`). Plan (above), then, in a separate process under
   `offline_earth_engine()`, `prepare(directory, plan, graph, parameters=..., manifest=...,
   inputs={"input": image}, namespace=..., caps=...)`: 5 exports, every request saved with its
   bytes and SHA-256 and checked to read only the manifest's tables and its stage's saved inputs.
   `status(directory)` is reviewed before step 4.
4. **The run** (live). `run(directory)`. Once, the 9 tables are checked to exist. Then for each
   of the 5 exports in order: the destination is checked absent (HTTP 404), the intent written,
   `exportImage` called once with the request ID fixed at preparation, the operation polled
   every 30 s, and the saved asset's bands (names in order, FLOAT), CRS, dimensions and affine
   transform checked before the next export is submitted.
5. **Readback** (live), under `CallGuard({"computePixels": 5}, path=...)` (naming the client
   function the readback calls), then the comparison, offline.

## Requests and caps

| Step | Calls | Cap |
|---|---|---|
| Initialize | `algorithms.list` | 1 per process (upload, run, readback, and each resume in a new process), each recorded |
| Weight destinations absent | `getAsset` | 9 |
| Weight upload | `exportTable`: 9 tasks, at most 2 active | 9 |
| Weight status | `getOperation` (`export_weights`' own tasks), every 30 s per active task; `getTaskStatus` (tasks recorded before an interruption) | 120; 20 |
| Weight resume | `getAsset`, one per shard recorded "starting" (at most one per interruption) | 2 |
| Finding out whether a task exists, if a submission's outcome is unknown | the task list (`listOperations`, under `CallGuard({"listOperations": 2})`) | 2 reads |
| The run: tables and destinations absent, then each saved asset | `getAsset` (9 + 5 + 5 = 19) | 24 (`Caps.asset_calls`) |
| The run: the input and the four stages | `exportImage`: 5 tasks, one at a time | 5 (`Caps.tasks`) |
| The run: status, and `recover`'s check of an operation | `getOperation`, every 30 s | 200 (`Caps.status_calls`), which bounds the run's total polling (100 minutes at 30 s); `Caps.seconds` (5,400) bounds each `run` or `resume` call, not the total |
| Readback: the saved input, C1, C2, C3 and logits, each with a validity band | `computePixels`, NPY, bodies measured offline first | 5; at most 150,528 values each |

At most 14 tasks (9 tables, 5 images), 14 new assets, 35 `getAsset`, 340 status reads, 2
task-list reads and 5 `computePixels`, with one initialization per process. One request in
flight. No retries, no fallback, no deletion, no cancellation, no other writes. A cap reached stops
the step with its reason recorded (the runner's state file, or the guard's counts); a task left
running is reported, not cancelled, and `resume` continues it under the same caps. Calls are
counted as they are attempted, so an export interrupted before its call took effect and submitted
again would exceed `tasks` 5: that is the user's decision (below), never automatic.

## Interruptions and failures

The runner writes each transition before its next call. After an interruption, `resume(directory)`
in a new process polls an export whose task exists and never resubmits it. If an export was being
submitted and its outcome is unknown (the call raised, or the process stopped before the response
was written), the run stops (`DecisionNeeded`) and the lead asks the user; `recover` then records
the task the lead found in the task list (an operation of `YOUR_EE_PROJECT`, checked against the
export's description, type and destination), or, if none exists, returns the export to be
submitted once more with the same request ID (Earth Engine starts at most one operation per
request ID). A status response that names another operation stops the run and changes nothing;
the operation polled is `exports.<label>.operation` in `state.json`, which the lead checks
against the task list. A failed task, or a saved asset whose bands or grid differ, ends the run
(`FAILED`): it is never retried, and a new run needs a new approval.

## Checks

For every readback: all 50,176 (224 × 224) or stage-grid pixels valid, every value finite float32.
- **Input transport:** the saved input equals the seeded array, 0 differing float32 bits.
- **C1, C2, C3** (the boundaries the stages save): against the PyTorch reference, the original SMP
  forward's `encoder.norm1` to `norm3` outputs on the same state and input, largest absolute error
  within **1e-4**, with MAE, RMSE and the count over the target; against the staged run's saved C1,
  C2 and C3, the differing bits (expected 0: C2 and C3 are the same graphs, and the single-graph
  run matched the staged run bit for bit).
- **Logits:** all 100,352 values against the PyTorch reference, largest absolute error within
  **1e-4**, MAE, RMSE, count over the target and argmax disagreements; against the staged run's and
  the single-graph run's logits, the differing bits (expected 0).

A difference in bits is recorded and explained; it does not change the 1e-4 test. The PyTorch
references are the staged run's cached arrays (its `references.npz`, SHA-256
`4a79bff7f81d8aa1ec94b0473fb1e1ffc2d5ebedb9e30efbf9b2a561a5f69295`, recorded in
[the staged run's record](segformer_b0-fullmodel-live.json)): the state and input are the same, so
no whole-model forward runs again. Recorded as a data point for cost calibration (X07, deferred):
each task's run time and batch EECU-seconds from its operation's metadata, beside the staged run's
(305 s, 91 EECU-seconds for six exports) and the single graph's (77.7 minutes, 5,870).

## Records

The run directory (prepared requests, state file, manifest, responses) stays in git-ignored
`release_runs/x06-segformer_b0/`. The public record, `x06-live.json`, replaces only the project
string, with a redaction manifest. This plan is the implementer's; the addendum below is the
lead's.

## Assets created and kept

Fourteen new assets under `projects/YOUR_EE_PROJECT/assets/`: the weight tables
`x06_segformer_b0_weights_part_0000` to `_0008`, and the images `x06_segformer_b0_input`, `_C1`,
`_C2`, `_C3` and `_logits`. They are kept after the run. Deleting them is the user's step, by a
separate instruction; nothing in this plan deletes, overwrites or cancels anything.

## Needs a new approval

Outside the approved scope, so not done under this plan:
- any export submitted a second time after an interruption (more than 5 image or 9 table
  `export` calls), any shard count other than 9, or caps raised above those in the table;
- a new run after a failed one, or a plan other than the default four stages (other limits,
  `cuts=`, the hand plan's six stages);
- another chip, other weights (trained, or another seed) or real imagery;
- an interactive `computePixels` of a stage or of the whole model (the single graph failed
  interactively), and any readback beyond the five above;
- any request over 10,000,000 bytes, or any change to a prepared request;
- deleting or overwriting any asset, and cost-calibration runs (X07).

## Addendum: the run (lead, 2026-10-15)

Run by the lead under the user's advance approval, at `b7c995b` with earthengine-api 1.7.45. The
run directory stays in git-ignored `release_runs/x06-segformer_b0/`; the public record is
[`x06-live.json`](x06-live.json), which replaces only the project string (56 times) and carries
its redaction manifest. Everything ran as planned, within every cap: no deviation, no
interruption, no resume, no second submission.

1. **Offline.** State `3c7d9cb7…`, input `46e1a727…`, graph `568c92c3…`, 128 records, 9 upload
   chunks and 9 planned shards. The largest complete `exportTable` body is 6,936,187 bytes with the
   run's namespace (6,936,189 with the placeholder, as planned), all 9 under 8,000,000. The
   driver's offline event reads `largest_body: 0` because it read the wrong key; the lead
   recomputed the figure offline after the run from the same records and chunking (noted in the
   record).
2. **Weights.** The 9 destinations were absent; the 9 tables uploaded, all `COMPLETED`, in 15
   minutes. Guard: `getAsset` 9 of 11, `exportTable` 9 of 9, `getOperation` 47 of 120,
   `getTaskStatus` 0 of 20.
3. **Prepare** (offline, in its own process). The plan gave the four stages above (plan SHA-256
   `f8a6b964…` with the run's namespace). The requests are 1,222,585, 2,062,504, 1,166,014,
   874,308 and 464,151 bytes. Each is the table's figure less 11 bytes per asset reference (1,
   4, 4, 7 and 10 references), because the sizing tool's placeholder namespace is 11 characters
   longer than the run's; the record checks this exactly for all five. `status`: `PREPARED`, no
   calls.
4. **Run.** `COMPLETED` in one `run` call, every saved asset verified (bands, CRS, dimensions,
   transform) before the next submission. Calls: `exportImage` 5 of 5, `getAsset` 19 of 24,
   `getOperation` 11 of 200.
5. **Readback.** `computePixels` 5 of 5. Each live expression equalled the one measured offline,
   and every response was decoded with all pixels valid and every value finite float32.

| Boundary | Values | Largest absolute error vs PyTorch | MAE | Over 1e-4 | Differing bits vs the staged run |
|---|---:|---:|---:|---:|---:|
| input | 100,352 | 0 (bit-exact, SHA-256 `46e1a727…`) | 0 | 0 | — |
| C1 | 100,352 | 1.67e-6 | 1.40e-7 | 0 | 0 |
| C2 | 50,176 | 3.10e-6 | 2.77e-7 | 0 | 0 |
| C3 | 31,360 | 3.58e-6 | 4.39e-7 | 0 | 0 |
| logits | 100,352 | 2.98e-6 | 4.83e-7 | 0 | 0 |

The logits have 0 argmax disagreements with PyTorch, and 0 differing bits against the
single-graph run's logits: SHA-256 `fade1379…` in all three runs.

**What this shows.** This is live agreement, within 1e-4, of the converted SegFormer-B0 run by
the planner's default plan and the library runner: one chip, seeded random weights, the seeded
input. It is bit-identical to the hand-staged and single-graph runs. Stage 1 (input → C1) and
stage 4 (C1, C2, C3 → logits) ran live for the first time. It says nothing about accuracy (the
weights are random), other chips or other settings.

**Cost** (a data point for X07, deferred):

| Run | Exports | Batch EECU-seconds | Task seconds |
|---|---:|---:|---:|
| This run, four stages | 4 | 92.2 | 216 |
| Hand-staged run | 6 | 91.1 | 305 |
| Single graph | 1 | 5,870 | 4,665 |

- **This run, by export:** input 0.40, C1 29.8, C2 11.4, C3 12.6, logits 38.4 EECU-seconds.
- **Elapsed:** 370 s from the first submission to the last success.
- **Comparison:** the hand-staged figures and this run's four stages both exclude the input's
  transport.

**Assets created and kept** (14): the weight tables `x06_segformer_b0_weights_part_0000` to
`_0008`, and the images `x06_segformer_b0_input`, `_C1`, `_C2`, `_C3` and `_logits`. Deleting
them is the user's step.
