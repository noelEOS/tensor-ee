# L06c live-run plan: the rebuilt browser demo

Status: **approved and run; both runs passed.** The user approved this plan on
2026-10-13 for project `ORIGINAL_PROJECT`, and the lead ran it as written that day
(2026-10-12 16:56–17:08 UTC): 4 export tasks, all COMPLETED on the first attempt.
The results, the UI observations, the request cross-checks and the output
verification are in [l06c-live-run.json](l06c-live-run.json). The plan below is
unchanged from what the user approved.

The L06c check is "G1 + demo run check". The offline part is done: the new demo's
export requests are byte-identical to the archived demo's
([graph identity](l06c-demo-graph-identity.json)), and its two U-Net requests also
equal, byte for byte, the requests the archived demo submitted in its two verified
live runs (`live_request_crosscheck` in the same file). This plan adds one live run
per model through the new demo, to show that the rebuilt orchestration submits,
polls and displays end to end.

## Common setup

| Item | Value |
|---|---|
| Project | `<PROJECT>`: the project the user approves (the plan's validation rules name the one used for T01–T06) |
| Credentials | Application Default Credentials, scopes `earthengine` and `cloud-platform` (`demos/browser/exports.py::initialize`); nothing is sent to the browser or saved |
| Code | `lib/l06c-browser-demo` at the handoff revision, from a checkout with `PYTHONPATH=.` |
| U-Net checkpoint | `checkpoints/unet_resnet18_s42_step6000_last.pth`, SHA-256 `2afeaef8…2d74928` |
| Swin-T checkpoint | `downloads/m12_ckpt/upernet_tu-swin_tiny_patch4_window7_224_s42_step6000_last.pth`, SHA-256 `5c7a7c44…2cea42a5` (checked by the demo) |
| Weight manifests | `data/manifests/segwater_r18_weight_manifest.json` (`87c2f53f…f4b65a`), `data/manifests/m12_trained_weights_manifest.json` (`dbd3cdaf…32fc9a3`); `<PROJECT>` must be able to read the weight assets they list |
| State directory | a new empty directory per run, outside the repository, passed with `--state-dir` |
| Browser | the lead drives the UI on `http://127.0.0.1:<port>`; Leaflet and OpenStreetMap tiles load from the internet |

### Every Earth Engine request the demo makes

The server sends every request below; the browser itself only fetches the display
tiles of the map IDs the server returns. `ee.data.newTaskId` is local (a UUID) and
contacts nothing. The polling behaviour is the archived demo's, unchanged.

| Trigger | U-Net (`Demo`) | Swin-T (`SwinDemo`) |
|---|---|---|
| `GET /api/config` (each page load or "Retry connection") | none | 1 `getInfo` (preset outline, `swin.py` `config`) |
| `POST /api/scenes` (each **Find available scenes**) | 2 `getInfo`: AOI projection; scene search (up to 51 scenes, each with a server-side coverage `reduceRegion`) | 3 `getInfo`: AOI projection; scene search; chip outline |
| `POST /api/preview` (each scene selection; the search auto-selects the first fully covering scene) | 1 `getMapId` (radar VV) | 2 `getInfo` (AOI projection, chip outline bounds) + 1 `getMapId` |
| `POST /api/run` | 1 `getInfo` (AOI projection) | 1 `getInfo` (AOI projection) |
| Preparation and submission (server worker) | 1 `exportImage` | 1 `exportImage` per stage, 3 in sequence |
| Server status polling (worker) | 1 `getTaskStatus` every 5 s until terminal (every 10 s after a connection error or an `UNKNOWN` state) | 1 `getTaskStatus` every 5 s per stage (every 10 s after a connection error; an `UNKNOWN` state fails the job) |
| `GET /api/jobs/<id>` (the page polls every 5 s while the job is active, once on load, and on returning to the tab) | none while active; 3 `getMapId` once COMPLETED (probability, water mask, radar VV), cached in the server for 30 minutes | **1 `getInfo` on every poll** (result bounds, `swin.py` `public_job`), plus the same 3 `getMapId` once COMPLETED |
| `GET /api/example` (only with `--example`; not part of this plan) | 1 `getInfo` (outline) + 3 `getMapId` (cached 30 minutes) | 1 `getInfo` (outline), 1 more through `config` when the record has no AOI, + 3 `getMapId` (cached 30 minutes) |
| Map display | EE display tiles for the shown layer, as the map is panned and zoomed | the same |

Expected totals for the two runs below, as scripted (one search, one auto-selected
preview, one run, the result viewed once, no reload):

| Requests | Run 1: U-Net | Run 2: Swin-T |
|---|---|---|
| `getInfo` | 3 | 7, plus 1 per page poll while active: about one per 5 s of total job time, at most 360 per 30-minute stage and 1,080 at the caps |
| `getMapId` | 4 (1 preview, 3 result) | 4 (1 preview, 3 result) |
| `exportImage` (writes) | 1 | 3 |
| `getTaskStatus` | about one per 5 s of task time; at most 360 at the cap | the same per stage; at most 1,080 at the caps |

Each extra search adds 2 (U-Net) or 3 (Swin-T) `getInfo` plus its auto-preview; each
extra page load adds, for Swin-T, 1 `getInfo` (config) and 1 for the restored job. Every
request other than `exportImage` is a read; the scene search also computes coverage
on the server. Closing the page stops the page's polls; the server's status polls
continue until the export is terminal or the server is stopped.

## Run 1: U-Net, Sun Moon Lake preset (one export)

```sh
PYTHONPATH=. python -m demos.browser.server --project <PROJECT> \
    --checkpoint <resources>/checkpoints/unet_resnet18_s42_step6000_last.pth \
    --state-dir <scratch>/l06c-live-unet
```

1. Click **Sun Moon Lake**. Set **From** and **Until** to 2024-01-05 and click
   **Find available scenes**. Expected: the 2024-01-05 descending scene
   `…20240105T215301_20240105T215326_051977_0647E8_D7EE` at 100 % coverage, and
   "2 model chips · 448 × 224 px" (grid `EPSG:32651`, origin 284480, 2640960).
2. Select it (the radar preview loads), then **Run native inference** once.

| Item | Value |
|---|---|
| Export tasks | **1** (`Export.image.toAsset`, description `segwater_demo_<job>`) |
| Destination | `projects/<PROJECT>/assets/segwater_demo_<job>`: 448 × 224 at 10 m, five float32 bands `logit_0`, `logit_1`, `water_probability`, `water_mask`, `coverage`; the image is clipped to the preset rectangle |
| Request | ≈ 1.21 MB; 1,213,421 bytes when `<PROJECT>` has 13 characters (1 byte more or less per character); budget 9.5 MB |
| Reference | the verified archived run of the same inputs took 57.98 s of EE execution and 91.2 batch EECU-seconds (`docs/history/cnn/results/demo_live_results.json`); not a guarantee |
| Polling | the demo polls `getTaskStatus` every 5 s, and every 10 s after a connection error |
| Polling cap | 30 minutes after submission (at most 360 status polls). If the task is not terminal by then, stop the server with Ctrl+C (the EE task continues), record the task state and report to the user; nothing is cancelled without the user's approval |

## Run 2: Swin-T/UPerNet, one chip (three chained exports)

```sh
PYTHONPATH=. python -m demos.browser.server --model swin-t --project <PROJECT> \
    --checkpoint <resources>/downloads/m12_ckpt/upernet_tu-swin_tiny_patch4_window7_224_s42_step6000_last.pth \
    --state-dir <scratch>/l06c-live-swin
```

1. Click **Lake chip**. Set both dates to 2024-01-03 and click **Find available
   scenes**. Expected: `…20240103T100134_20240103T100159_051941_06469E_B0BD`
   (ascending, 100 % chip coverage) and the chip outline of grid `EPSG:32651`,
   origin (287680.95335071225, 2640838.8405420803), 224 × 224: the Milestone 12
   catalogue chip.
2. Select it, then **Run native inference** once.

| Stage | Destination (`projects/<PROJECT>/assets/…`) | Output | Request (with a 12-character project) |
|---|---|---|---|
| A | `swint_upernet_demo_<job>_A` | 56 × 56 at 40 m, 673 bands (C1–C3, valid fraction) | 796,265 bytes |
| B | `swint_upernet_demo_<job>_B` | 7 × 7 at 320 m, 768 bands (C4) | 324,521 bytes |
| C | `swint_upernet_demo_<job>` | 224 × 224 at 10 m, the same five bands as the U-Net | 558,768 bytes |

| Item | Value |
|---|---|
| Export tasks | **3**, in sequence: B is prepared and submitted only after A is COMPLETED, C only after B |
| Failure rule | a FAILED or CANCELLED stage stops the job; no later stage is submitted and nothing is retried automatically |
| Polling | every 5 s per stage (10 s after a connection error) |
| Polling cap | 30 minutes per stage, 90 minutes in total; past a cap, stop the server, record the stage states and report (restarting the server with the same `--state-dir` later resumes without resubmitting a stage) |

## Totals and what is left behind

- **4 export tasks** and **4 new image assets** in `<PROJECT>`, plus the reads listed
  above. The assets stay; the demo deletes nothing, and deleting them is the
  user's decision.
- Local job records and prepared requests in the two state directories.

## Pass criteria

- Both jobs reach COMPLETED through the UI. The U-Net job card goes through
  "Building", "Queued", "Running"; the Swin-T card through Stage 1/3, 2/3, 3/3.
- The completed result shows the probability, water mask and radar VV layers, and
  opacity works.
- Each prepared request file still matches its recorded SHA-256 (the demo checks
  this before every submission).
- Optional, offline afterwards: rebuild the U-Net request with the run's project
  and job ID, as `live_request_crosscheck` in `docs/evidence/l06c-demo-graph-identity.py`
  does, and compare the SHA-256 with the job's `request_sha256`.

These are execution and display checks. Numerical agreement with PyTorch is the
earlier, separate evidence (T06, M12); graph identity carries it to this demo.
