# C0b live plan: window 8 for `group_product`, `window_resize` and `window_matrix_resize`

Status: **not run.** The lead runs it on the reviewed revision, under the user's standing
budget for live fixture harness runs. The implementer has not run it.

The capstone (brief (development archive: `docs/tasks/C0-capstone.md`)) runs three decoder blocks at window 8 that
had no window-8 evidence (the brief's status audit). The run checks them live against
`torch.nn.functional` at the capstone's settings:
- `group_product`: the fusion's 4 input-channel groups at window 8, against the unsplit
  `F.conv2d`;
- `window_resize`: every resize of the capstone, at its own token extents. The FPN's x2
  (8 → 16, 16 → 32, 32 → 64; the last is also the fusion's P2), the fusion's x4 (P3,
  16 → 64) and x8 (the PSP output, 8 → 64), all align_corners=False; and the head's x4 with
  align_corners=True (64 → 256);
- `window_matrix_resize`: the PSP's 1 × 1, 2 × 2, 3 × 3 and 6 × 6 branches onto C4's 8 × 8
  tokens, align_corners=False.

The calibration's `group_product` fixture runs again as the window-7 regression. Every
number below is the output of `PYTHONPATH=. python tools/live_fixtures.py --plan c0b`, with
earthengine-api 1.7.45 (request bytes and graph hashes can change with the client's
serialization; `tests/test_c0b_window8.py` checks these tables against `--plan`).

## Common setup

| Item | Value |
|---|---|
| Project | `ORIGINAL_PROJECT`, as for the [P00 calibration](p00-calibration-plan.md) |
| Credentials | Application Default Credentials, scopes `earthengine` and `cloud-platform` (`live_session`); nothing is saved |
| Network | The IPv4-first network shim the lead uses for Earth Engine calls (plan, standing constraints) |
| Code | `lib/c0-capstone-composite` at the reviewed revision, from a checkout with `PYTHONPATH=.`; earthengine-api 1.7.45, torch 2.14.0, timm 1.0.30, SMP 0.5.0, NumPy 2.5.3 |
| Concurrency | One request at a time; no other harness invocation runs at once |
| Inputs and weights | Inline: built in the graph from NumPy values and embedded constants; no asset is read |

## The command

One run, within one run's budget.

```sh
PYTHONPATH=. python tools/live_fixtures.py --plan c0b
PYTHONPATH=. python tools/live_fixtures.py --run c0b --project ORIGINAL_PROJECT
```

`c0b` is a named selection (`SELECTIONS` in `tools/fixtures/decoder.py`): 12 fixtures, the
eleven new ones and one calibration fixture. `--run` refuses to start (exit 2, before any
cloud call) if the plan exceeds the budget or a reference or mirror fails locally. It then
initializes Earth Engine, rebuilds the 25 graphs and stops unless each equals its planned
SHA-256 (below, first 12 hex digits), and sends the requests in the order listed.

## The selection

| Block | Fixtures | Settings | Requests |
|---|---|---|---:|
| `group_product` | `group_product/g4-w7` (calibration, the regression) | 4 groups, window 7 | 1 |
| `group_product` | `g4-w8` | 4 groups of the 8 input channels, window 8; 2 × 2 windows | 1 |
| `window_resize` | `f2-e8-w8`, `f2-e16-w8`, `f2-e32-w8` | x2, align_corners=False, on 8, 16 and 32 tokens (1, 2 × 2 and 4 × 4 windows) | 2 each |
| `window_resize` | `f4-e16-w8`, `f8-e8-w8` | x4 on 16 tokens and x8 on 8 tokens, align_corners=False | 2 each |
| `window_resize` | `ac-f4-e64-w8` | x4, align_corners=True, on 64 tokens (8 × 8 windows) | 5 |
| `window_matrix_resize` | `s1-e8-w8`, `s2-e8-w8`, `s3-e8-w8`, `s6-e8-w8` | an s × s source onto 8 × 8 tokens (one window), align_corners=False | 2 each |

- Each resize fixture reads its output on the fine token grid and its input on the coarse
  one: 2 requests. The head's 256 × 256 output exceeds a window-8 request's side of 64
  tokens (8 windows), so it is read as its four diagonal 64 × 64 blocks, at token offsets
  (0, 0), (64, 64), (128, 128) and (192, 192): 4 requests, plus the input. Its weights
  depend on the extent (align_corners=True), so the fixture runs at the capstone's own 64
  → 256. The blocks hold every child-window row and column, and so every window-local
  matrix of both axes.
- Each matrix fixture's source is a value [s², 2] (an embedded constant, as the library
  embeds weights). It reads the output on the 8 × 8 grid and echoes the source as an s × s
  grid of its rows: 2 requests.
- The group fixtures read their output and input on the same token grid: 1 request.

## Every request

All are `ee.data.computePixels` with `NUMPY_NDARRAY`. "Values" are output values (pixels ×
bands, with one validity band per readout); "expected" is their float32 payload, without the
NPY header; "request" is the serialized request body; "constants" is the blocks'
embedded-constant count from their `METADATA` cost functions; "inputs" is the input values
the graph embeds; "NPY header" is the length of the response's NPY header, as
`numpy.lib.format` writes it for that structured float32 array (`npy_header_length`). Grids
are abstract `TokenGrid`s (origin (0, 0), EPSG:3857; 10 m tokens on the inputs, finer on the
resized outputs: 5, 2.5 and 1.25 m); "offset" is a block's first token (row, column) on its
grid.

### `c0b` (25 requests)

| Fixture | Request | Grid (rows × columns) | Offset | Bands | Values | Expected bytes | Request bytes | Constants | Inputs | Readouts (channels) | Graph | NPY header |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|---:|
| `window/group_product/g4-w7` | 0 | 14 × 14 | — | 14 | 2,744 | 10,976 | 208,749 | 288 | 1,568 | out 4, input 8 | `1e45467eb92c` | 374 |
| `window/group_product/g4-w8` | 0 | 16 × 16 | — | 14 | 3,584 | 14,336 | 250,905 | 288 | 2,048 | out 4, input 8 | `a3523fa269d2` | 374 |
| `window/window_resize/f2-e8-w8` | 0 | 16 × 16 | — | 3 | 768 | 3,072 | 74,406 | 324 | 128 | out 2 | `f92f3bd68eb4` | 182 |
| `window/window_resize/f2-e8-w8` | 1 | 8 × 8 | — | 3 | 192 | 768 | 5,453 | 0 | 128 | input 2 | `6345eaacb66f` | 182 |
| `window/window_resize/f2-e16-w8` | 0 | 32 × 32 | — | 3 | 3,072 | 12,288 | 80,517 | 648 | 512 | out 2 | `f4c8bc2f7198` | 182 |
| `window/window_resize/f2-e16-w8` | 1 | 16 × 16 | — | 3 | 768 | 3,072 | 10,124 | 0 | 512 | input 2 | `2e861564f41d` | 182 |
| `window/window_resize/f2-e32-w8` | 0 | 64 × 64 | — | 3 | 12,288 | 49,152 | 99,136 | 656 | 2,048 | out 2 | `83d8bd281da9` | 182 |
| `window/window_resize/f2-e32-w8` | 1 | 32 × 32 | — | 3 | 3,072 | 12,288 | 28,845 | 0 | 2,048 | input 2 | `44207a90cd73` | 182 |
| `window/window_resize/f4-e16-w8` | 0 | 64 × 64 | — | 3 | 12,288 | 49,152 | 81,634 | 976 | 512 | out 2 | `0fb8df7615dd` | 182 |
| `window/window_resize/f4-e16-w8` | 1 | 16 × 16 | — | 3 | 768 | 3,072 | 10,111 | 0 | 512 | input 2 | `c9599f02d198` | 182 |
| `window/window_resize/f8-e8-w8` | 0 | 64 × 64 | — | 3 | 12,288 | 49,152 | 77,499 | 1,296 | 128 | out 2 | `29ade23c8fd2` | 182 |
| `window/window_resize/f8-e8-w8` | 1 | 8 × 8 | — | 3 | 192 | 768 | 5,433 | 0 | 128 | input 2 | `cd999f4798fb` | 182 |
| `window/window_resize/ac-f4-e64-w8` | 0 | 64 × 64 | — | 3 | 12,288 | 49,152 | 188,386 | 5,184 | 8,192 | out_b0 2 | `0ede2309c8cd` | 182 |
| `window/window_resize/ac-f4-e64-w8` | 1 | 64 × 64 | 64, 64 | 3 | 12,288 | 49,152 | 188,391 | 5,184 | 8,192 | out_b1 2 | `56b916b29812` | 182 |
| `window/window_resize/ac-f4-e64-w8` | 2 | 64 × 64 | 128, 128 | 3 | 12,288 | 49,152 | 188,391 | 5,184 | 8,192 | out_b2 2 | `4df9cba09895` | 182 |
| `window/window_resize/ac-f4-e64-w8` | 3 | 64 × 64 | 192, 192 | 3 | 12,288 | 49,152 | 188,391 | 5,184 | 8,192 | out_b3 2 | `b132629cc938` | 182 |
| `window/window_resize/ac-f4-e64-w8` | 4 | 64 × 64 | — | 3 | 12,288 | 49,152 | 103,457 | 0 | 8,192 | input 2 | `bbbc4f3e2363` | 182 |
| `window/window_matrix_resize/s1-e8-w8` | 0 | 8 × 8 | — | 3 | 192 | 768 | 10,447 | 64 | 2 | out 2 | `467cf44668a3` | 182 |
| `window/window_matrix_resize/s1-e8-w8` | 1 | 1 × 1 | — | 3 | 3 | 12 | 3,759 | 0 | 2 | input 2 | `93947b800db8` | 182 |
| `window/window_matrix_resize/s2-e8-w8` | 0 | 8 × 8 | — | 3 | 192 | 768 | 12,061 | 256 | 8 | out 2 | `e8926760fb2d` | 182 |
| `window/window_matrix_resize/s2-e8-w8` | 1 | 2 × 2 | — | 3 | 12 | 48 | 3,897 | 0 | 8 | input 2 | `bd2464f9087f` | 182 |
| `window/window_matrix_resize/s3-e8-w8` | 0 | 8 × 8 | — | 3 | 192 | 768 | 13,122 | 576 | 18 | out 2 | `c68437da4871` | 182 |
| `window/window_matrix_resize/s3-e8-w8` | 1 | 3 × 3 | — | 3 | 27 | 108 | 4,035 | 0 | 18 | input 2 | `4f2fa4704696` | 182 |
| `window/window_matrix_resize/s6-e8-w8` | 0 | 8 × 8 | — | 3 | 192 | 768 | 22,116 | 2,304 | 72 | out 2 | `9658b92ef707` | 182 |
| `window/window_matrix_resize/s6-e8-w8` | 1 | 6 × 6 | — | 3 | 108 | 432 | 4,725 | 0 | 72 | input 2 | `194e5608525b` | 182 |

The first row is the calibration's `group_product` request, with the same graph as the
[P00 calibration plan](p00-calibration-plan.md) and the [P04 plan](p04-live-plan.md). A run
sends fixtures in registry order, so it comes first.

## Totals against the budget

| Item | `c0b` | Budget (P00, R4), per run |
|---|---:|---:|
| Fixtures | 12 | — |
| Interactive requests | 25 | 40 |
| Largest grid, window lane | 14 × 14 (window 7); 64 × 64 (window 8) | 8 × 8 windows: 56 at window 7, 64 at 8 |
| Largest request, output values | 12,288 | 1,000,000 |
| Largest request, serialized bytes | 250,905 | 2,000,000 |
| Largest request, bands | 14 | 1,024 (computePixels' limit) |
| Largest NPY response header (bytes) | 374 | 9,000 (`numpy.load` refuses more than 10,000) |
| All requests: values | 114,382 | — |
| All requests: expected bytes | 457,528 | — |
| All requests: request bytes | 1,863,990 | — |
| Embedded constants (`METADATA`) / input values | 28,412 / 51,432 | — |
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
notes, and `tests/test_c0b_window8.py` checks each), so every output should equal its mirror
bit for bit. `--offline c0b` (zero cloud calls) gives the expected distance from PyTorch:

| Fixtures | Mirror against PyTorch, largest |
|---|---|
| `group_product`, `g4-w7` and `g4-w8` | 8.34e-7 and 7.15e-7 |
| `window_resize`, the five with align_corners=False | 1.19e-7 |
| `window_resize`, `ac-f4-e64-w8` (all four blocks) | 1.19e-7 |
| `window_matrix_resize`, all four | 5.96e-8 |

- **References and mirrors.** The references are `F.interpolate` as the capstone calls it
  (SMP's resizes with `size=`, the head's `UpsamplingBilinear2d` with `scale_factor=4`) and
  `F.conv2d` of the unsplit weight. The resize mirrors take PyTorch's weights read off
  `F.interpolate` on one-hot inputs, not the library's.
- **align_corners=False.** The inputs are multiples of 2^-23 and the weights of 1/16 or
  coarser, so every product and sum of Earth Engine's double arithmetic is exact, in any
  order, and one float32 rounding follows.
- **align_corners=True (the head).** The weights are PyTorch's float32 ones. The y stage's
  products are exact. The x stage multiplies float32 weights by doubles, so its result
  could depend on whether Earth Engine fuses the multiply and the add; Milestone 11 found it
  does not. On this data, the unfused result and both fused ones round to the same float32
  at every output, so the expected values do not depend on that.
- **Matrix resizes.** The 2-D weights are products of 1-D weights that are multiples of
  1/16, so every dot product is exact in double. A 1 × 1 source resizes to a constant.
- **Group products.** As in P04: every group product and group sum is exact in double.

## What is left behind

- One evidence file, `docs/evidence/live-fixtures/<UTC time>-c0b.json`, which the lead
  commits with the results.
- Nothing in Earth Engine: no asset is created, read or deleted, and no task is started. The
  harness refuses any such call before it is sent.

## Pass criteria and stop rules

- **Pass:** the run exits 0 with `"status": "passed"`: every fixture passes (every output
  float32 and valid, within 1e-4 of PyTorch, bit-identical to its exact mirror, and every
  input echo exact), and `cloud_calls` shows 25 requests and no refused call.
- **A fixture fails** (exit 1): the evidence names each failing value by output, row, column
  and channel. A setting that fails is left out of its block's promotion. Nothing is retried
  automatically.
- **A run aborts** (a transient error, a graph that differs from the plan, a refused call):
  the evidence records why. A re-run is a new invocation, within the same standing budget.
