# P02 live plan: pooling configurations and nearest factors 3 and 4

Status: **not run.** The lead runs it on the reviewed revision, under the user's standing
budget for live fixture harness runs, before P02 is integrated. The implementer has not run
it.

The run checks the raster lane's new settings live against PyTorch: nearest upsampling at
factors 3 and 4, and `pool2d` at the configurations of
the P02 brief (development archive: `docs/tasks/P02-raster-factors-and-pooling.md`). Today's pooling and nearest
settings run again as regression checks. Every number below is the output of
`PYTHONPATH=. python tools/live_fixtures.py --plan p02`, with earthengine-api 1.7.45
(request bytes and graph hashes can change with the client's serialization;
`tests/test_p02_raster_settings.py` checks these tables against `--plan`).

## Common setup

| Item | Value |
|---|---|
| Project | `ORIGINAL_PROJECT`, as for the [P00 calibration](p00-calibration-plan.md) |
| Credentials | Application Default Credentials, scopes `earthengine` and `cloud-platform` (`live_session`); nothing is saved |
| Network | The IPv4-first network shim the lead uses for Earth Engine calls (plan, standing constraints) |
| Code | `lib/p02-raster-factors-and-pooling` at the reviewed revision, from a checkout with `PYTHONPATH=.`; earthengine-api 1.7.45, torch 2.14.0, NumPy 2.5.3 |
| Concurrency | One request at a time; no other harness invocation runs at once (P01's run included) |
| Inputs and weights | Inline: built in the graph from NumPy values (8,084 input values in all, summed over requests). No weights; no asset is read |

## The command

```sh
PYTHONPATH=. python tools/live_fixtures.py --plan p02
PYTHONPATH=. python tools/live_fixtures.py --run p02 --project ORIGINAL_PROJECT
```

`p02` is a named selection (`SELECTIONS` in `tools/fixtures/raster.py`): 14 fixtures, the
eleven new ones and three calibration fixtures. `--run` refuses to start (exit 2, before any
cloud call) if the plan exceeds the budget or a reference fails locally. It then
initializes Earth Engine, rebuilds the 25 graphs and stops unless each equals its planned
SHA-256 (below, first 12 hex digits), and sends the requests in the order listed.

## Every request

All are `ee.data.computePixels` with `NUMPY_NDARRAY`. "Values" are output values (pixels ×
bands, with one validity band per readout); "expected" is their float32 payload, without
the NPY header; "request" is the serialized request body; "inputs" is the inline input
values the graph embeds. No block here has coefficients, so the `METADATA` embedded-constant
count is 0 for every request. Grids are `Grid`s in EPSG:3857 with the upper-left corner at
(1000, 2000); the readout column gives each grid's pixel size. The new fixtures' grids are
not square, so a swapped axis changes the output shape.

| Fixture | Request | Grid (rows × columns) | Bands | Values | Expected bytes | Request bytes | Inputs | Readouts (channels; grid) | Graph |
|---|---:|---|---:|---:|---:|---:|---:|---|---|
| `raster/pool2d/avg-k2-s2` | 0 | 8 × 8 | 3 | 192 | 768 | 14,877 | 512 | out 2 (40 m) | `1e16cd636714` |
| `raster/pool2d/avg-k2-s2` | 1 | 16 × 16 | 3 | 768 | 3,072 | 12,313 | 512 | input 2 (20 m) | `a70404994d07` |
| `raster/pool2d/max-k3-s2-p1` | 0 | 8 × 8 | 3 | 192 | 768 | 17,073 | 512 | out 2 (40 m) | `dbd4a41c0594` |
| `raster/pool2d/max-k3-s2-p1` | 1 | 16 × 16 | 3 | 768 | 3,072 | 12,365 | 512 | input 2 (20 m) | `d7df8022f8af` |
| `raster/nearest_upsample/f2` | 0 | 16 × 16 | 3 | 768 | 3,072 | 9,936 | 128 | out 2 (20 m) | `6f2a9d5a8375` |
| `raster/nearest_upsample/f2` | 1 | 8 × 8 | 3 | 192 | 768 | 7,615 | 128 | input 2 (40 m) | `0f6382a738af` |
| `raster/pool2d/max-k2-s2` | 0 | 5 × 7 | 3 | 105 | 420 | 12,136 | 280 | out 2 (20 m) | `b5ccf549291d` |
| `raster/pool2d/max-k2-s2` | 1 | 10 × 14 | 3 | 420 | 1,680 | 9,520 | 280 | input 2 (10 m) | `aa325a241924` |
| `raster/pool2d/avg-k3-s3` | 0 | 4 × 5 | 3 | 60 | 240 | 13,123 | 360 | out 2 (30 m) | `2f61022b8aa3` |
| `raster/pool2d/avg-k3-s3` | 1 | 12 × 15 | 3 | 540 | 2,160 | 10,506 | 360 | input 2 (10 m) | `79d0e2021d0f` |
| `raster/pool2d/max-k3-s3` | 0 | 4 × 5 | 3 | 60 | 240 | 13,132 | 360 | out 2 (30 m) | `d28838214a8b` |
| `raster/pool2d/max-k3-s3` | 1 | 12 × 15 | 3 | 540 | 2,160 | 10,516 | 360 | input 2 (10 m) | `cf034abdf159` |
| `raster/pool2d/avg-k4-s4` | 0 | 4 × 3 | 3 | 36 | 144 | 13,417 | 384 | out 2 (40 m) | `322cbf4d0aa0` |
| `raster/pool2d/avg-k4-s4` | 1 | 16 × 12 | 3 | 576 | 2,304 | 10,800 | 384 | input 2 (10 m) | `1dd3a583aa09` |
| `raster/pool2d/max-k4-s4` | 0 | 4 × 3 | 3 | 36 | 144 | 13,427 | 384 | out 2 (40 m) | `e82e31a7d17e` |
| `raster/pool2d/max-k4-s4` | 1 | 16 × 12 | 3 | 576 | 2,304 | 10,811 | 384 | input 2 (10 m) | `92854815d06d` |
| `raster/pool2d/max-k3-s1-p1` | 0 | 12 × 15 | 6 | 1,080 | 4,320 | 12,727 | 360 | out 2, input 2 (10 m) | `986b7ada1719` |
| `raster/pool2d/avg-k3-s1-p1-incl` | 0 | 12 × 15 | 6 | 1,080 | 4,320 | 12,525 | 360 | out 2, input 2 (10 m) | `b327d8e49020` |
| `raster/pool2d/avg-k3-s1-p1-excl` | 0 | 12 × 15 | 6 | 1,080 | 4,320 | 12,300 | 360 | out 2, input 2 (10 m) | `4b60a9148795` |
| `raster/pool2d/max-k3-s2-p1-odd` | 0 | 7 × 8 | 3 | 168 | 672 | 15,754 | 390 | out 2 (20 m) | `70b45aeae8cf` |
| `raster/pool2d/max-k3-s2-p1-odd` | 1 | 13 × 15 | 3 | 585 | 2,340 | 10,929 | 390 | input 2 (10 m) | `eeceeb24524e` |
| `raster/nearest_upsample/f3` | 0 | 24 × 18 | 3 | 1,296 | 5,184 | 9,669 | 96 | out 2 (10 m) | `664ca3aac016` |
| `raster/nearest_upsample/f3` | 1 | 8 × 6 | 3 | 144 | 576 | 7,294 | 96 | input 2 (30 m) | `49cbcc8925a5` |
| `raster/nearest_upsample/f4` | 0 | 32 × 24 | 3 | 2,304 | 9,216 | 9,656 | 96 | out 2 (10 m) | `5641942165cd` |
| `raster/nearest_upsample/f4` | 1 | 8 × 6 | 3 | 144 | 576 | 7,281 | 96 | input 2 (40 m) | `d5eab056be8b` |

The first six rows are the calibration's, with the same graphs as in the
[calibration plan](p00-calibration-plan.md).

## Totals against the budget

| Item | Plan | Budget |
|---|---:|---:|
| Interactive requests | 25 | 40 |
| Largest grid, raster lane | 32 × 24 pixels | 64 × 64 |
| Largest request, output values | 2,304 | 1,000,000 |
| Largest request, serialized bytes | 17,073 | 2,000,000 |
| Largest request, bands | 6 | 1,024 (computePixels' limit) |
| All requests: values / expected bytes | 13,710 / 54,840 | — |
| All requests: request bytes | 289,702 | — |
| Embedded constants (`METADATA`) / inline input values, summed over requests | 0 / 8,084 | — |
| Assets read or written, tasks | 0, 0 | 0, 0 |
| Requests in flight | 1 | 4 |

Besides the 25 requests, `ee.Initialize` makes the client's own reads: the Cloud API
discovery document (fetched twice), one `algorithms.list`, and the public catalogue of
deprecated assets. The harness lets only `algorithms.list` (at most 2) and
`image.computePixels` (at most the 25 planned) reach the Earth Engine transport, and the
client executes each of them once (no retries). The HTTP layer below may still re-send a
request after a credential refresh or a dropped connection.

## What each fixture must show

There is no NumPy mirror of the raster path's Earth Engine arithmetic, so each fixture
compares with PyTorch only, as in the calibration. Max pooling and nearest upsampling
select or copy values: tolerance 0, compared bit for bit (one ulp fails, and so does -0.0
against +0.0). Averages: within 1e-4. Every input echo must equal the NumPy input bit for
bit. Inputs are exact float32 dyadic values; "signed" is [-1, 1), "negative" [-2, 0) and
never 0, "positive" [0.5, 1.5).

| Fixture | Reference | Input (rows × columns × 2) | Tolerance | What it shows | Recorded result |
|---|---|---|---:|---|---|
| `raster/pool2d/avg-k2-s2` | `nn.AvgPool2d(2, 2)` | 16 × 16, signed | 1e-4 | Regression: today's aligned average | [calibration](live-fixtures/20261013T033726Z-calibration.json): passed |
| `raster/pool2d/max-k3-s2-p1` | `nn.MaxPool2d(3, 2, 1)` | 16 × 16, negative | 0 | Regression: today's padded max; a pad that wins fails every border window | calibration: passed, exactly equal |
| `raster/nearest_upsample/f2` | `F.interpolate(scale_factor=2, mode="nearest")` | 8 × 8, signed | 0 | Regression: factor 2 | calibration: passed, exactly equal |
| `raster/pool2d/max-k2-s2` | `nn.MaxPool2d(2)` | 10 × 14, signed | 0 | Today's aligned max, first time as a live fixture | [primitives_float32.json](../history/cnn/results/primitives_float32.json), `max_pool_nearest`: pool max 0 |
| `raster/pool2d/avg-k3-s3` | `nn.AvgPool2d(3)` | 12 × 15, signed | 1e-4 | Aligned 3 × 3 average: `reduceResolution` onto the 30 m grid | none: new |
| `raster/pool2d/max-k3-s3` | `nn.MaxPool2d(3)` | 12 × 15, signed | 0 | Aligned 3 × 3 max: only the block's own 9 pixels can win | none: new |
| `raster/pool2d/avg-k4-s4` | `nn.AvgPool2d(4)` | 16 × 12, signed | 1e-4 | Aligned 4 × 4 average onto the 40 m grid | none: new |
| `raster/pool2d/max-k4-s4` | `nn.MaxPool2d(4)` | 16 × 12, signed | 0 | Aligned 4 × 4 max | none: new |
| `raster/pool2d/max-k3-s1-p1` | `nn.MaxPool2d(3, 1, 1)` | 12 × 15, negative | 0 | "Same" max pool: padding never wins. A zero pad would change all 50 border pixels of each channel | none: new |
| `raster/pool2d/avg-k3-s1-p1-incl` | `nn.AvgPool2d(3, 1, 1, count_include_pad=True)` | 12 × 15, positive | 1e-4 | Divisor 9 at every pixel. The other rule moves every border pixel by more than 0.16 | none: new |
| `raster/pool2d/avg-k3-s1-p1-excl` | `nn.AvgPool2d(3, 1, 1, count_include_pad=False)` | 12 × 15, positive | 1e-4 | Divisor = in-bounds count (4 at a corner, 6 on an edge, 9 inside) | none: new |
| `raster/pool2d/max-k3-s2-p1-odd` | `nn.MaxPool2d(3, 2, 1)` | 13 × 15, negative | 0 | Today's padded max on odd sizes: the last output row and column read the bottom and right padding, which the even calibration fixture never reaches. A zero pad would change all 26 border pixels of each channel of the 7 × 8 output | [resnet_primitives.json](../history/cnn/results/resnet_primitives.json), `max3_stride2_padded_negative_odd` (17 × 15): pass, 5.2e-8, from inputs that were not float32-exact |
| `raster/nearest_upsample/f3` | `F.interpolate(scale_factor=3, mode="nearest")` | 8 × 6, signed (30 m → 10 m) | 0 | Factor 3 | none: new |
| `raster/nearest_upsample/f4` | `F.interpolate(scale_factor=4, mode="nearest")` | 8 × 6, signed (40 m → 10 m) | 0 | Factor 4 | none: new |

`--offline p02` runs every reference with zero cloud calls and passes; offline,
`tests/test_p02_raster_settings.py` checks that these data catch a winning pad, the wrong
divisor and a one-ulp copy error through a mocked `computePixels`.

## What is left behind

- One evidence file, `docs/evidence/live-fixtures/<UTC time>-p02.json`, which the lead
  commits with the result.
- Nothing in Earth Engine: no asset is created, read or deleted, and no task is started.
  The harness refuses any such call before it is sent.

## Pass criteria and stop rules

- **Pass:** exit 0 and `"status": "passed"`: all 14 fixtures pass as above, and
  `cloud_calls` shows 25 requests and no refused call. P02's promotions then stand.
- **A fixture fails** (exit 1): the evidence names each failing value by output, row,
  column and channel. A failing configuration is removed from the block's status and
  rejected again by `pool_config` (or, for a nearest factor, left out of the status), as
  the brief requires; the lead reports it to the user. A failing regression fixture stops
  the promotion until it is explained. Nothing is retried automatically.
- **The run aborts** (a transient error, a graph that differs from the plan, a refused
  call): the evidence records why, and the lead reports it. A re-run is a new invocation,
  within the same standing budget.
