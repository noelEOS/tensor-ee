# Browser demo

A local browser app for the two reference models. Draw a rectangle on the map, or
use the Sun Moon Lake preset, choose an available Sentinel-1 acquisition, and run
the model **inside Earth Engine**:

- the trained SegWater ResNet18 U-Net (the default);
- the trained Swin-T/UPerNet, on one 224 × 224 chip (`--model swin-t`).

Every model operation is a native Earth Engine operation. The local Python process
builds the graphs with the `tensor_ee` library, submits batch exports, polls them
and serves the UI. It never runs a model forward pass locally and never downloads
input pixels. The demo lives in [`demos/browser/`](demos/browser/) and imports only
public library modules. It is a checkout tool and is not part of the installed
package.

This page covers both models. It replaces the former `SWIN_DEMO.md`, which now
points here.

## Start

Requirements: the repository's Python environment with `.[segwater]` installed for
the U-Net (Swin-T needs only the core dependencies), Earth Engine access through
Application Default Credentials (`gcloud auth application-default login`), and the
local checkpoints. From the repository root:

```sh
PYTHONPATH=. python -m demos.browser.server --project YOUR_PROJECT
PYTHONPATH=. python -m demos.browser.server --project YOUR_PROJECT --model swin-t
```

Then open the URL the server prints: `http://127.0.0.1:8765/#token=…` (U-Net) or
`http://127.0.0.1:8766/#token=…` (Swin-T). The part after `#` is the session token: the
browser never sends it to the server, the page reads it and sends it with every request
that starts work. A page opened without it can show results but cannot search or run.
The server binds to loopback only. The page loads Leaflet and the OpenStreetMap basemap
from the internet; Earth Engine serves the radar and result tiles.

| Option | Meaning | Default |
|---|---|---|
| `--project` | Earth Engine project that runs the exports and owns the result assets. **Required**; there is no default project | — |
| `--model` | `unet` or `swin-t` | `unet` |
| `--port` | Local port, 1024–65535 | 8765; 8766 for Swin-T |
| `--max-tiles` | Chip cap for the snapped area, 1–4 | 4; Swin-T supports exactly 1 |
| `--checkpoint` | Local checkpoint | `checkpoints/unet_resnet18_s42_step6000_last.pth`; Swin-T `downloads/m12_ckpt/upernet_tu-swin_tiny_patch4_window7_224_s42_step6000_last.pth` |
| `--manifest` | Weight-asset manifest | `data/manifests/segwater_r18_weight_manifest.json`; Swin-T `data/manifests/m12_trained_weights_manifest.json` |
| `--state-dir` | Job records and prepared requests | `demos/browser/state/`; Swin-T `demos/browser/state/swin-t/` (ignored by Git) |
| `--example` | A completed result to offer as a labelled example (see below) | none |
| `--offline` | Serve the UI against a local fixture, with no Earth Engine request (see below) | off |

Default paths are relative to the repository root. The weight assets come only from
the manifests. The default manifests' assets are public, in the Earth Engine project
`tensor-ee`, under CC BY 4.0 ([WEIGHTS-LICENSE.md](WEIGHTS-LICENSE.md)): `--project` can be any project of your own that may create image
assets. The Swin-T checkpoint must have the verified
SHA-256 (`5c7a7c44…`); the demo checks it, and checks every checkpoint tensor
against the deployed weight records before building a graph. The U-Net must match
all 31 deployed convolution layers exactly.

Credentials stay in the Python process. They are never sent to the browser or
written to the job records.

## Offline mode

`--offline` serves the same UI, controllers and job state machine without any
Earth Engine request. Use it to try the interface, or to check the demo without a
cloud project:

```sh
PYTHONPATH=. python -m demos.browser.server --offline --project offline-check \
    --checkpoint /path/to/checkpoints/unet_resnet18_s42_step6000_last.pth
PYTHONPATH=. python -m demos.browser.server --offline --project offline-check --model swin-t \
    --checkpoint /path/to/downloads/m12_ckpt/upernet_tu-swin_tiny_patch4_window7_224_s42_step6000_last.pth
```

- **Graphs are real.** Each run builds and saves its export request with the
  library, against the offline signatures of `ee.apitestcase.ApiTestCase`, exactly
  as a live run prepares it. That is why the checkpoint is needed; without it a run
  fails while preparing, and the rest of the UI still works. The page then says only "The
  request failed; see the server console."; the console where the server runs shows the
  cause, a missing checkpoint file.
- **Everything else is a fixture.** AOI projections use a local UTM approximation.
  The scene search returns 2024-01-03 and 2024-01-05 (full coverage) and a
  2024-01-06 partial-coverage scene, when the dates include them. Map layers are
  flat tiles labelled "offline fixture". Submission and status come from a
  simulated queue: READY, then RUNNING, then COMPLETED after about ten seconds per
  export. A fixture example is offered unless `--example` is given.
- Every other Earth Engine call raises. The header reads "Offline fixture".
- State goes to a new temporary directory unless `--state-dir` is given; pass one
  to try restarting and resuming. Offline jobs are marked `"offline": true`: a live
  server never adopts them, and an offline server never adopts live jobs.

## Using the demo

1. Optionally, open the **completed example** to show a finished water map at
   once. It is labelled as a previous result, not a new run.
2. Click the preset (**Sun Moon Lake**, or **Lake chip** for Swin-T), or **Draw an
   area** and click two opposite corners.
3. Choose start and end dates and click **Find available scenes**. Both dates are
   inclusive, and a search covers at most 31 days. The list shows fully or partly
   covering VV/VH Sentinel-1 IW scenes (up to 50), with date, orbit direction and
   coverage of the area (Swin-T: of the whole chip).
4. Select a fully covering scene; the radar VV preview loads.
5. Click **Run native inference**. The job card reports preparation, queueing and
   Earth Engine execution (for Swin-T, per stage: 1/3, 2/3, 3/3).
6. When it completes, switch between **Probability**, **Water mask** and **Radar
   VV**, and adjust the opacity. The job card shows the result's asset ID.

A new inference is a batch export and takes minutes. The completed result is then
displayed as ordinary Earth Engine map tiles: panning the map does not rerun the
network.

## AOI and model behaviour

**Area.** A rectangle must lie within one WGS84 UTM zone and hemisphere, between
80°S and 84°N, span at most 0.1° in each direction and at least 10 m. Earth Engine
projects its bounds; the demo expands them to a stable 2240 m lattice (224 pixels
at 10 m), so nearby drawings keep the same chip origins.

**U-Net.** The expanded area may need at most `--max-tiles` chips (4 by default). A
small rectangle near a chip boundary can need several; an oversized drawing gets a
clear request to draw a smaller one. Every chip is an independent 224 × 224 model
input, computed with the full chip of raw scene context, so seams between chips are
possible; the exported result is clipped to the drawn rectangle. The model uses
float32, the VV/VH order with the v3 preprocessing, and folded BatchNorm
([`tensor_ee.segwater_inference`](tensor_ee/segwater_inference.py)).

**Swin-T/UPerNet.** Exactly one 224 × 224 chip at 10 m in UTM zone 51N
(120°–126°E, north of the equator). The lattice is anchored at the verified
catalogue chip of the last research milestone, in the Sun Moon Lake scene; a drawing that needs more
than one chip is refused, even a small one across a chip boundary. Inference and
display cover the entire chip, and the scene search shows its outline. Other
acquisitions are sampled onto that grid with Earth Engine's default nearest
neighbour. Three chained exports, built by
[`tensor_ee.swin_workflow`](tensor_ee/swin_workflow.py) from a per-job resource
declaration (`resources.json`: the `segwater_v3_db` input recipe and the `water`
output recipe):

| Stage | Earth Engine computation | Export grid |
|---|---|---|
| A | v3 preprocessing, patch embedding, Swin stages 1–3; C1/C2/C3 and the scene's validity | 56 × 56 at 40 m, 673 bands |
| B | Patch merging 3 and Swin stage 4; raw C4 | 7 × 7 at 320 m, 768 bands |
| C | Feature norms, PSP, FPN, `grouped_double` fusion, head, upsampling, softmax and mask | 224 × 224 at 10 m, 5 bands |

Both models export the float32 bands `logit_0`, `logit_1`, `water_probability`,
`water_mask` and `coverage`. Water is class 1, with probability ≥ 0.5. This is an
execution and presentation demo, not an accuracy assessment.

## Jobs and artifacts

- **One job at a time.** A second run while one is active is refused.
- **Persisted and resumable.** Each job keeps `job.json` and its prepared request
  (`request.json`, or `request_A/B/C.json` with `resources.json` for Swin-T) in the
  state directory, with the request's SHA-256 and a persisted request ID. Submission
  reuses that ID, so a lost response is retried without creating a second task. A
  request whose SHA-256 changed, or that is missing, fails the job before
  submission. Restarting with the same `--project` and `--state-dir` resumes
  unfinished jobs; Swin-T skips completed stages. Refreshing the browser never
  submits again. Keep one server per state directory.
- **Stopping.** Closing the browser or stopping Python leaves submitted exports
  running in Earth Engine; restart to pick up their status.
- **Assets.** Results stay in `projects/<project>/assets/` as `segwater_demo_<job>`,
  or `swint_upernet_demo_<job>_A`, `…_B` and `swint_upernet_demo_<job>`, with the
  source scene and checkpoint SHA-256 as image properties (Swin-T also the manifest
  SHA-256 and the stage). The demo deletes nothing.
- **Checks.** Every request is audited for nested operations (an external
  prediction call is refused) and must stay below 9.5 MB.
- **Security.** The server accepts only loopback `Host` headers. POST requests need
  the session token from the printed URL's fragment and, when sent, a same-origin
  `Origin`; bodies are JSON objects of at most 64 KB; responses refuse framing. The
  token is printed to the console only, so another website cannot obtain it; the demo
  still assumes a single-user machine, since any process that can read the console or
  the state directory could use it. It is not meant for public hosting.
- **Cost.** One run is one batch export in your project (at most `--max-tiles` chips at
  10 m; Swin-T runs three exports in sequence for its one chip), plus the metadata
  and map-tile requests the page makes. The recorded runs took a few minutes of batch
  time each ([U-Net](docs/evidence/l06c-live-run.json)); Earth Engine bills batch
  compute to the project you pass with `--project`.

## The completed example

The archived demo showed a verified previous result. Its asset belongs to a
particular project, so the new demo offers an example only when asked:

```sh
--example docs/history/cnn/results/segwater_native_mosaic_2x2.json    # U-Net, 2 x 2 chips, verified against PyTorch
--example docs/history/cnn/results/swin_demo_example_checks.json      # Swin-T catalogue chip
--example <state-dir>/<job>/job.json                                  # any completed job of this demo
```

The asset must be readable by `--project`. The example is always labelled "a
previous result, not a new run"; its layers are display tiles of the saved asset.

## How it is built

| File | Role |
|---|---|
| [`demos/browser/server.py`](demos/browser/server.py) | U-Net controller, HTTP handler, command line |
| [`demos/browser/swin.py`](demos/browser/swin.py) | Swin-T controller: one chip, three stages |
| [`demos/browser/geometry.py`](demos/browser/geometry.py) | AOI validation, chip lattice, dates (no EE calls) |
| [`demos/browser/exports.py`](demos/browser/exports.py) | Persisted requests, idempotent submission, EE initialization |
| [`demos/browser/offline.py`](demos/browser/offline.py) | `--offline` fixture |
| [`demos/browser/static/`](demos/browser/static/) | The page, its script and styles |

The orchestration is ported from the archived demo (`demo/` at the tag
`research-archive-2026-10-12`). When the library rebuilt the demo, its export requests for
fixed inputs were byte-identical to the archived demo's:
[record](docs/evidence/l06c-demo-graph-identity.json), from
its generator (development archive: `docs/evidence/l06c-demo-graph-identity.py`). That record predates
the move of the weight assets to the public project `tensor-ee`. The requests now name the public
assets, so they are no longer byte-identical to the archived demo's, and the record
describes the earlier manifests. The offline tests are
`tests/test_browser_demo_*.py`.

## History

The archived demo's verified live runs, browser checks and example checks are
recorded in [`docs/history/`](docs/history/README.md):
`docs/history/cnn/results/demo_live_results.json` (two U-Net runs, preset and drawn),
`demo_example_tiles.json`, `swin_demo_example_checks.json` and
`swin_demo_prepare_checks.json`. Their code is at the archive tag. The new demo's live
run, under the user-approved
[`docs/evidence/l06c-live-run-plan.md`](docs/evidence/l06c-live-run-plan.md), passed
for both models: [`docs/evidence/l06c-live-run.json`](docs/evidence/l06c-live-run.json).
It ran before the weight assets moved to `tensor-ee`, on the earlier ones. No browser-demo run
has used the public assets in `tensor-ee` yet; the Code Editor demo has
([CODE_EDITOR.md](CODE_EDITOR.md#how-this-was-checked)).

The scenes the demo searches and runs on are Sentinel-1 GRD images from the Earth
Engine catalogue: contains modified Copernicus Sentinel data (2024 for the presets).
