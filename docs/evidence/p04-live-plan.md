# P04 live plan: pooling extents and bins, fusion group counts

Status: **not run.** The lead runs it on the reviewed revision, under the user's standing
budget for live fixture harness runs, before P04 is integrated. The implementer has not run
it.

The run checks two decoder blocks live against PyTorch at settings the reference model does
not use: `window_pool` at extents of 5, 8, 12 and 14 tokens with the bins (1, 2, 3, 6), and
at 8 and 12 tokens with the bins (1, 2, 4, 8); and `group_product`, the fusion's grouped
double product, at 1, 2, 4 and 8 groups at windows 7 and 12, plus two order probes. The
calibration's two fixtures of these blocks run again as the window-7 regression. Every
number below is the output of `PYTHONPATH=. python tools/live_fixtures.py --plan p04`, with
earthengine-api 1.7.45 (request bytes and graph hashes can change with the client's
serialization; `tests/test_p04_decoder_blocks.py` checks these tables against `--plan`).

## Common setup

| Item | Value |
|---|---|
| Project | `ORIGINAL_PROJECT`, as for the [P00 calibration](p00-calibration-plan.md) |
| Credentials | Application Default Credentials, scopes `earthengine` and `cloud-platform` (`live_session`); nothing is saved |
| Network | The IPv4-first network shim the lead uses for Earth Engine calls (plan, standing constraints) |
| Code | `lib/p04-decoder-blocks` at the reviewed revision, from a checkout with `PYTHONPATH=.`; earthengine-api 1.7.45, torch 2.14.0, timm 1.0.30, SMP 0.5.0, NumPy 2.5.3 |
| Concurrency | One request at a time; no other harness invocation runs at once (P03's run included) |
| Inputs and weights | Inline: built in the graph from NumPy values and embedded constants; no asset is read |

## The command

One run, within one run's budget.

```sh
PYTHONPATH=. python tools/live_fixtures.py --plan p04
PYTHONPATH=. python tools/live_fixtures.py --run p04 --project ORIGINAL_PROJECT
```

`p04` is a named selection (`SELECTIONS` in `tools/fixtures/decoder.py`): 17 fixtures, the
fifteen new ones and two calibration fixtures. `--run` refuses to start (exit 2, before any
cloud call) if the plan exceeds the budget or a reference or mirror fails locally. It then
initializes Earth Engine, rebuilds the 24 graphs and stops unless each equals its planned
SHA-256 (below, first 12 hex digits), and sends the requests in the order listed.

## The selection

| Block | Fixtures | Settings | Requests |
|---|---|---|---:|
| `window_pool` | `window_pool/bins1236-w7` (calibration, the regression) | one 7 × 7 window; bins (1, 2, 3, 6) | 2 |
| `window_pool` | `e5-bins1236-w5`, `e8-bins1236-w8`, `e12-bins1236-w12` | one window of 5, 8 or 12 tokens; bins (1, 2, 3, 6) | 2 each |
| `window_pool` | `e14-bins1236-w7` | 14 × 14 tokens, 2 × 2 windows of 7; bins (1, 2, 3, 6) | 2 |
| `window_pool` | `e8-bins1248-w8`, `e12-bins1248-w12` | one window of 8 or 12 tokens; bins (1, 2, 4, 8) | 2 each |
| `group_product` | `group_product/g4-w7` (calibration, the regression) | 4 groups, window 7 | 1 |
| `group_product` | `g1-w7`, `g2-w7`, `g8-w7`, `g1-w12`, `g2-w12`, `g4-w12`, `g8-w12` | 1, 2, 4 or 8 groups of the 8 input channels; window 7 or 12; 2 × 2 windows | 1 each |
| `group_product` | `order-g4-w7`, `order-g8-w7` | 4 or 8 groups, window 7; one designed output channel | 1 each |

Each pool fixture reads the stacked [R, 4] pooled value (R = 50 or 85 bins) as a grid of its
rows (50 × 1 for the calibration's; R / 5 × 5 for the new ones) and its input on the token
grid: 2 requests. Each group fixture reads its output and its input on the same token grid:
1 request.

## Every request

All are `ee.data.computePixels` with `NUMPY_NDARRAY`. "Values" are output values (pixels ×
bands, with one validity band per readout); "expected" is their float32 payload, without the
NPY header; "request" is the serialized request body; "constants" is the blocks'
embedded-constant count from their `METADATA` cost functions; "inputs" is the inline input
values the graph embeds; "NPY header" is the length of the response's NPY header, as
`numpy.lib.format` writes it for that structured float32 array (`npy_header_length`). Grids
are abstract `TokenGrid`s (origin (0, 0), EPSG:3857, 10 m tokens).

### `p04` (24 requests)

| Fixture | Request | Grid (rows × columns) | Bands | Values | Expected bytes | Request bytes | Constants | Inputs | Readouts (channels) | Graph | NPY header |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|---:|
| `window/group_product/g4-w7` | 0 | 14 × 14 | 14 | 2,744 | 10,976 | 208,749 | 288 | 1,568 | out 4, input 8 | `1e45467eb92c` | 374 |
| `window/window_pool/bins1236-w7` | 0 | 50 × 1 | 5 | 250 | 1,000 | 57,845 | 750 | 196 | pooled 4 | `e286e4b6aa81` | 246 |
| `window/window_pool/bins1236-w7` | 1 | 7 × 7 | 5 | 245 | 980 | 7,506 | 0 | 196 | input 4 | `e819ca0ae071` | 182 |
| `window/window_pool/e5-bins1236-w5` | 0 | 10 × 5 | 5 | 250 | 1,000 | 39,279 | 550 | 100 | pooled 4 | `cdb9729e7f15` | 246 |
| `window/window_pool/e5-bins1236-w5` | 1 | 5 × 5 | 5 | 125 | 500 | 6,335 | 0 | 100 | input 4 | `d914b87ddf69` | 182 |
| `window/window_pool/e8-bins1236-w8` | 0 | 10 × 5 | 5 | 250 | 1,000 | 70,235 | 850 | 256 | pooled 4 | `4716fc4e85ca` | 246 |
| `window/window_pool/e8-bins1236-w8` | 1 | 8 × 8 | 5 | 320 | 1,280 | 8,257 | 0 | 256 | input 4 | `5d2232525f2a` | 182 |
| `window/window_pool/e12-bins1236-w12` | 0 | 10 × 5 | 5 | 250 | 1,000 | 133,501 | 1,250 | 576 | pooled 4 | `2f224b96a654` | 246 |
| `window/window_pool/e12-bins1236-w12` | 1 | 12 × 12 | 5 | 720 | 2,880 | 12,146 | 0 | 576 | input 4 | `62c847e3f25e` | 182 |
| `window/window_pool/e14-bins1236-w7` | 0 | 10 × 5 | 5 | 250 | 1,000 | 66,070 | 1,450 | 784 | pooled 4 | `dc3d8d602fc7` | 246 |
| `window/window_pool/e14-bins1236-w7` | 1 | 14 × 14 | 5 | 980 | 3,920 | 14,678 | 0 | 784 | input 4 | `321d264d02c5` | 182 |
| `window/window_pool/e8-bins1248-w8` | 0 | 17 × 5 | 5 | 425 | 1,700 | 73,170 | 1,445 | 256 | pooled 4 | `0d9641ff05db` | 246 |
| `window/window_pool/e8-bins1248-w8` | 1 | 8 × 8 | 5 | 320 | 1,280 | 8,251 | 0 | 256 | input 4 | `c7d0a2bf9bec` | 182 |
| `window/window_pool/e12-bins1248-w12` | 0 | 17 × 5 | 5 | 425 | 1,700 | 136,573 | 2,125 | 576 | pooled 4 | `343eee7434dc` | 246 |
| `window/window_pool/e12-bins1248-w12` | 1 | 12 × 12 | 5 | 720 | 2,880 | 12,184 | 0 | 576 | input 4 | `f2787e6eaa74` | 182 |
| `window/group_product/g1-w7` | 0 | 14 × 14 | 14 | 2,744 | 10,976 | 87,783 | 288 | 1,568 | out 4, input 8 | `7180585c7f8e` | 374 |
| `window/group_product/g2-w7` | 0 | 14 × 14 | 14 | 2,744 | 10,976 | 129,489 | 288 | 1,568 | out 4, input 8 | `a4dcdf9baca4` | 374 |
| `window/group_product/g8-w7` | 0 | 14 × 14 | 14 | 2,744 | 10,976 | 367,945 | 288 | 1,568 | out 4, input 8 | `2b0e06fb8f98` | 374 |
| `window/group_product/g1-w12` | 0 | 24 × 24 | 14 | 8,064 | 32,256 | 194,904 | 288 | 4,608 | out 4, input 8 | `7deba5b4f8b0` | 374 |
| `window/group_product/g2-w12` | 0 | 24 × 24 | 14 | 8,064 | 32,256 | 291,264 | 288 | 4,608 | out 4, input 8 | `0d1d258415f1` | 374 |
| `window/group_product/g4-w12` | 0 | 24 × 24 | 14 | 8,064 | 32,256 | 474,624 | 288 | 4,608 | out 4, input 8 | `adf1f70b9c05` | 374 |
| `window/group_product/g8-w12` | 0 | 24 × 24 | 14 | 8,064 | 32,256 | 841,751 | 288 | 4,608 | out 4, input 8 | `d68350509f19` | 374 |
| `window/group_product/order-g4-w7` | 0 | 14 × 14 | 11 | 2,156 | 8,624 | 206,810 | 72 | 1,568 | out 1, input 8 | `8146502724ee` | 310 |
| `window/group_product/order-g8-w7` | 0 | 14 × 14 | 11 | 2,156 | 8,624 | 365,659 | 72 | 1,568 | out 1, input 8 | `6059bc1d2db0` | 310 |

The first three rows are the calibration's requests, with the same graphs as the
[P00 calibration plan](p00-calibration-plan.md). A run sends fixtures in registry order, so
they come first.

## Totals against the budget

| Item | `p04` | Budget (P00, R4), per run |
|---|---:|---:|
| Fixtures | 17 | — |
| Interactive requests | 24 | 40 |
| Largest grid, window lane | 50 × 1 (window 7); 24 × 24 (window 12) | 8 × 8 windows: 40 at window 5, 56 at 7, 64 at 8, 96 at 12 |
| Largest request, output values | 8,064 | 1,000,000 |
| Largest request, serialized bytes | 841,751 | 2,000,000 |
| Largest request, bands | 14 | 1,024 (computePixels' limit) |
| Largest NPY response header (bytes) | 374 | 9,000 (`numpy.load` refuses more than 10,000) |
| All requests: values | 53,074 | — |
| All requests: expected bytes | 212,296 | — |
| All requests: request bytes | 3,815,008 | — |
| Embedded constants (`METADATA`) / inline input values | 10,868 / 33,328 | — |
| Assets read or written, tasks | 0, 0 | 0, 0 |
| Requests in flight | 1 | 4 |

Besides its requests, the run's `ee.Initialize` makes the client's own reads: the Cloud API
discovery document (fetched twice), one `algorithms.list`, and the public catalogue of
deprecated assets. The harness lets only `algorithms.list` (at most 2) and
`image.computePixels` (at most the planned number) reach the Earth Engine transport, and the
client executes each of them once (no retries). The HTTP layer below may still re-send a
request after a credential refresh or a dropped connection.

## What each fixture should show

Every mirror is exact (the reasons are in `tools/fixtures/decoder.py` and each fixture's
notes), so every output should equal its mirror bit for bit. `--offline p04` (zero cloud
calls) gives the expected distance from PyTorch:

| Fixtures | Mirror against PyTorch, largest |
|---|---|
| `window_pool`, all seven | 5.96e-8 |
| `group_product`, the eight group fixtures | 8.34e-7 |
| `group_product`, the two order probes | 8.34e-14 |

- **Pooling.** The new fixtures' mirrors take their bin tables from PyTorch's
  `adaptive_avg_pool2d` on one-hot index maps, not from the library. Uneven bins overlap: 5 → 2,
  3 and 6 (more bins than pixels at 6), 8 → 3 and 6, 14 → 3 and 6, and 12 → 8 (the
  calibration's 7 → 2, 3 and 6 too). 12 → (1, 2, 3, 6) and 8 → (1, 2, 4, 8) divide evenly.
  `e14-bins1236-w7` is the only pool on more than one window: it is the first library run of
  the gather (`sampleRectangle`) and the sum over 4 windows at a new extent.
- **Group products.** Every group product and group sum of these data is exact in double, so
  the result does not depend on the order of the sums, and one rounding follows.
- **Order probes.** One output channel whose weights make the group partials at token (7, 7)
  sum, in double and in group order, to a float32 value that another order gives only in
  specific cases (the swap of the first two partials at 4 groups; 192 of the 40,320 orders at
  8 groups, none of them the reversed order or a rotation), and that rounding each partial
  first does not give. Their double group sums are inexact at about a third of the tokens
  (64 and 81 of 196), though on this data only the probe token's float32 value depends on
  the order. So these two whole fixtures rest on one premise the others do not: Earth Engine
  adds the double partial images in the order the graph adds them (Milestone 10's
  formulation B). If a probe fails while the 8 group fixtures pass, the premise failed, not
  the block.

## What is left behind

- One evidence file, `docs/evidence/live-fixtures/<UTC time>-p04.json`, which the lead commits
  with the results.
- Nothing in Earth Engine: no asset is created, read or deleted, and no task is started. The
  harness refuses any such call before it is sent.

## Pass criteria and stop rules

- **Pass:** the run exits 0 with `"status": "passed"`: every fixture passes (every output
  float32 and valid, within 1e-4 of PyTorch, bit-identical to its exact mirror, and every
  input echo exact), and `cloud_calls` shows 24 requests and no refused call.
- **A fixture fails** (exit 1): the evidence names each failing value by output, row, column
  and channel. A setting that fails is left out of its block's promotion (brief, R5). Nothing
  is retried automatically.
- **A run aborts** (a transient error, a graph that differs from the plan, a refused call):
  the evidence records why. A re-run is a new invocation, within the same standing budget.
