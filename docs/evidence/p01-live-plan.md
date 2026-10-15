# P01 live plan: the window lane at windows 5, 8 and 12

Status: **run on 2026-10-13; all 24 fixtures passed.** The lead ran the selections under the user's standing budget, on the reviewed P01 revision. `p01a` passed 18 of 18 at `00751e8` ([record](live-fixtures/20261013T074301Z-p01a.json)). `p01b` passed 6 of 7 ([record](live-fixtures/20261013T074330Z-p01b.json)): the window-12 bias readout's NPY header was over numpy's default limit, a client-side refusal after Earth Engine had computed it. After the planner fix `07cf47e`, the bias fixture passed in 2 requests at `f2d245b` ([record](live-fixtures/20261013T075542Z-window-relative_position_bias-w12.json)). Every output is bit-identical to its mirror; the largest difference from PyTorch is 6.68e-6.

The runs check the window lane's Fixed blocks live at windows 5, 8 and 12 with shift
⌊w/2⌋, against timm with random weights, and repeat window 7 as the regression check. Every
number below is the output of `PYTHONPATH=. python tools/live_fixtures.py --plan p01a` and
`--plan p01b`, with earthengine-api 1.7.45 (request bytes can change with the client's
serialization; `tests/test_p01_window_settings.py` checks these tables against `--plan`).

**Revision after the first live run.** `p01a` passed (18 of 18, 21 requests). `p01b` passed 6 of 7:
`relative_position_bias/w12` failed on the client, because its response's NPY header was over
`numpy.load`'s default `max_header_size`. The planner now keeps every response header within
9,000 bytes (`NPY_HEADER_LIMIT`), and splits that fixture into two requests. Only that fixture
changed: `p01b` is now 8 requests, and the lead reruns it.

## Common setup

| Item | Value |
|---|---|
| Project | `ORIGINAL_PROJECT`, the project of the P00 calibration run (the plan's validation rules) |
| Credentials | Application Default Credentials, scopes `earthengine` and `cloud-platform` (`live_session`); nothing is saved |
| Network | The IPv4-first network shim the lead uses for Earth Engine calls (plan, standing constraints) |
| Code | `lib/p01-window-and-shift` at the reviewed revision, from a checkout with `PYTHONPATH=.`; earthengine-api 1.7.45, torch 2.14.0, timm 1.0.30, SMP 0.5.0 |
| Concurrency | One request at a time; one harness invocation at a time |
| Inputs and weights | Inline: built in the graph from NumPy values and embedded constants; no asset is read |

## The commands

Two runs, each within one run's budget. `p01a` holds windows 5, 7 and 8. `p01b` holds
window 12 alone, so that the likeliest live failure (below) cannot cost the other windows'
evidence. Inside `p01b`, the three fixtures that build window 12's relative-position gather
run last. `p01` is the union of the two (29 requests), for `--plan` and `--offline`.

```sh
PYTHONPATH=. python tools/live_fixtures.py --plan p01a
PYTHONPATH=. python tools/live_fixtures.py --run p01a --project ORIGINAL_PROJECT
PYTHONPATH=. python tools/live_fixtures.py --plan p01b
PYTHONPATH=. python tools/live_fixtures.py --run p01b --project ORIGINAL_PROJECT
```

Each `--run` refuses to start (exit 2, before any cloud call) if its plan exceeds the budget
or a reference or mirror fails locally. It then initializes Earth Engine, rebuilds its graphs
and stops unless each equals its planned SHA-256 (below, first 12 hex digits), and sends the
requests in the order listed.

## The selections

24 fixtures. The shift ⌊w/2⌋ is 2, 4 and 6 at windows 5, 8 and 12.

| Selection | Window | Fixtures | Requests |
|---|---|---|---:|
| `p01a` | 7 (regression) | The calibration fixtures of the blocks P01 promotes: `relative_position_bias/w7`, `attention/w7-s3`, `layer_norm2d/w7-eps1e-6`, `patch_merge/w7`, `conv3x3_window/k3-w7`; and `attention/w7-single`, new (the 224 chip's stage 4 is one 7 × 7 window) | 7 |
| `p01a` | 5, 8 | For each window: `layer_norm2d/w{w}-eps1e-6`, `patch_merge/w{w}` (2w × 2w tokens into one stage-2 window), `conv3x3_window/k3-w{w}`, `relative_position_bias/w{w}`, `attention/w{w}-s{s}` (W-MSA and SW-MSA on 2 × 2 windows, with the mask and the unshifted heads), `attention/w{w}-single` | 7 each |
| `p01b` | 12 | The same six fixtures, in the same order | 8 (the bias takes 2) |

## Every request

All are `ee.data.computePixels` with `NUMPY_NDARRAY`. "Values" are output values (pixels ×
bands, with one validity band per readout); "expected" is their float32 payload, without the
NPY header; "request" is the serialized request body; "constants" is the blocks'
embedded-constant count from their `METADATA` cost functions; "inputs" is the inline input
values the graph embeds; "NPY header" is the length of the response's NPY header, as
`numpy.lib.format` writes it for that structured float32 array (`npy_header_length`). Grids
are abstract `TokenGrid`s (origin (0, 0), EPSG:3857, 10 m tokens; 20 m for the stage-2
outputs of patch merging).

### `p01a` (21 requests)

| Fixture | Request | Grid (rows × columns) | Bands | Values | Expected bytes | Request bytes | Constants | Inputs | Readouts (channels) | Graph | NPY header |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|---:|
| `window/relative_position_bias/w7` | 0 | 49 × 1 | 100 | 4,900 | 19,600 | 16,269 | 338 | 0 | bias_h0 49, bias_h1 49 | `6fd9e92beff0` | 2,422 |
| `window/attention/w7-s3` | 0 | 14 × 14 | 84 | 16,464 | 65,856 | 133,029 | 3,458 | 3,136 | concat 16, mask 49, input 16 | `6e0dc7cbee71` | 1,846 |
| `window/layer_norm2d/w7-eps1e-6` | 0 | 14 × 14 | 18 | 3,528 | 14,112 | 71,809 | 784 | 1,568 | out 8, input 8 | `9b3865551a92` | 438 |
| `window/patch_merge/w7` | 0 | 7 × 7 | 83 | 4,067 | 16,268 | 93,669 | 3,648 | 1,568 | cat 32, norm 32, out 16 | `52d58538f73a` | 1,718 |
| `window/patch_merge/w7` | 1 | 14 × 14 | 9 | 1,764 | 7,056 | 26,738 | 0 | 1,568 | input 8 | `dc3e7658b7c1` | 310 |
| `window/conv3x3_window/k3-w7` | 0 | 14 × 14 | 10 | 1,960 | 7,840 | 73,920 | 340 | 784 | out 4, input 4 | `68438139db4f` | 310 |
| `window/attention/w7-single` | 0 | 7 × 7 | 34 | 1,666 | 6,664 | 92,896 | 3,458 | 784 | concat 16, input 16 | `254f20d321b3` | 822 |
| `window/layer_norm2d/w5-eps1e-6` | 0 | 10 × 10 | 18 | 1,800 | 7,200 | 45,050 | 400 | 800 | out 8, input 8 | `a12852e18afe` | 438 |
| `window/patch_merge/w5` | 0 | 5 × 5 | 83 | 2,075 | 8,300 | 66,265 | 2,112 | 800 | cat 32, norm 32, out 16 | `1358181c513e` | 1,718 |
| `window/patch_merge/w5` | 1 | 10 × 10 | 9 | 900 | 3,600 | 17,359 | 0 | 800 | input 8 | `862e769f3842` | 310 |
| `window/conv3x3_window/k3-w5` | 0 | 10 × 10 | 10 | 1,000 | 4,000 | 51,557 | 244 | 400 | out 4, input 4 | `bdcba1a2decf` | 310 |
| `window/relative_position_bias/w5` | 0 | 5 × 5 | 78 | 1,950 | 7,800 | 17,172 | 243 | 0 | bias_h0 25, bias_h1 25, bias_h2 25 | `336d259bff1c` | 1,910 |
| `window/attention/w5-s2` | 0 | 10 × 10 | 78 | 7,800 | 31,200 | 101,403 | 1,575 | 1,200 | wmsa 12, concat 12, unshifted 12, mask 25, input 12 | `3b8e9944f17f` | 1,718 |
| `window/attention/w5-single` | 0 | 5 × 5 | 26 | 650 | 2,600 | 64,469 | 1,575 | 300 | concat 12, input 12 | `611a05bf0400` | 630 |
| `window/layer_norm2d/w8-eps1e-6` | 0 | 16 × 16 | 18 | 4,608 | 18,432 | 88,161 | 1,024 | 2,048 | out 8, input 8 | `fba097be37b3` | 438 |
| `window/patch_merge/w8` | 0 | 8 × 8 | 83 | 5,312 | 21,248 | 110,492 | 4,608 | 2,048 | cat 32, norm 32, out 16 | `44099c3bf63d` | 1,718 |
| `window/patch_merge/w8` | 1 | 16 × 16 | 9 | 2,304 | 9,216 | 32,588 | 0 | 2,048 | input 8 | `2f111e727fd2` | 310 |
| `window/conv3x3_window/k3-w8` | 0 | 16 × 16 | 10 | 2,560 | 10,240 | 87,921 | 400 | 1,024 | out 4, input 4 | `aced5702c3b2` | 310 |
| `window/relative_position_bias/w8` | 0 | 8 × 8 | 195 | 12,480 | 49,920 | 24,634 | 675 | 0 | bias_h0 64, bias_h1 64, bias_h2 64 | `24ad02b6321c` | 4,598 |
| `window/attention/w8-s4` | 0 | 16 × 16 | 133 | 34,048 | 136,192 | 170,207 | 4,290 | 4,096 | wmsa 16, concat 16, unshifted 16, mask 64, input 16 | `8177814a94d0` | 2,870 |
| `window/attention/w8-single` | 0 | 8 × 8 | 34 | 2,176 | 8,704 | 108,232 | 4,290 | 1,024 | concat 16, input 16 | `192a03d9ac5e` | 822 |

The first six rows are the calibration's requests, with the same graphs.

### `p01b` (8 requests)

| Fixture | Request | Grid (rows × columns) | Bands | Values | Expected bytes | Request bytes | Constants | Inputs | Readouts (channels) | Graph | NPY header |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|---:|
| `window/layer_norm2d/w12-eps1e-6` | 0 | 24 × 24 | 18 | 10,368 | 41,472 | 177,721 | 2,304 | 4,608 | out 8, input 8 | `f4574e10b39f` | 438 |
| `window/patch_merge/w12` | 0 | 12 × 12 | 83 | 11,952 | 47,808 | 201,338 | 9,728 | 4,608 | cat 32, norm 32, out 16 | `2e8bccd32cfb` | 1,718 |
| `window/patch_merge/w12` | 1 | 24 × 24 | 9 | 5,184 | 20,736 | 63,845 | 0 | 4,608 | input 8 | `c3bdb511d7a4` | 310 |
| `window/conv3x3_window/k3-w12` | 0 | 24 × 24 | 10 | 5,760 | 23,040 | 162,470 | 720 | 2,304 | out 4, input 4 | `2290213a9633` | 310 |
| `window/relative_position_bias/w12` | 0 | 12 × 12 | 290 | 41,760 | 167,040 | 36,259 | 1,587 | 0 | bias_h0 144, bias_h1 144 | `12a9ab241182` | 6,838 |
| `window/relative_position_bias/w12` | 1 | 12 × 12 | 145 | 20,880 | 83,520 | 31,927 | 1,587 | 0 | bias_h2 144 | `0775d50323b7` | 3,510 |
| `window/attention/w12-s6` | 0 | 24 × 24 | 245 | 141,120 | 564,480 | 390,368 | 13,683 | 13,824 | wmsa 24, concat 24, unshifted 24, mask 144, input 24 | `1c94303aa123` | 5,238 |
| `window/attention/w12-single` | 0 | 12 × 12 | 50 | 7,200 | 28,800 | 232,514 | 13,683 | 3,456 | concat 24, input 24 | `fa964b1c564e` | 1,142 |

The window-12 bias reads its three heads in two requests (heads 0 and 1, then head 2): in one request their NPY response header would be 10,230 bytes, over the 10,000 that `numpy.load` accepts, which failed the first live run of `p01b` on the client.

## Totals against the budget

| Item | `p01a` | `p01b` | `p01` (both) | Budget (P00, R4), per run |
|---|---:|---:|---:|---:|
| Fixtures | 18 | 6 | 24 | — |
| Interactive requests | 21 | 8 | 29 | 40 |
| Largest grid, window lane | 49 × 1 (window 7); 16 × 16 (window 8) | 24 × 24 | — | 8 × 8 windows: 56 at window 7, 64 at 8, 96 at 12 |
| Largest request, output values | 34,048 | 141,120 | 141,120 | 1,000,000 |
| Largest request, serialized bytes | 170,207 | 390,368 | 390,368 | 2,000,000 |
| Largest request, bands | 195 | 290 | 290 | 1,024 (computePixels' limit) |
| Largest NPY response header (bytes) | 4,598 | 6,838 | 6,838 | 9,000 (`numpy.load` refuses more than 10,000) |
| All requests: values | 114,012 | 244,224 | 358,236 | — |
| All requests: expected bytes | 456,048 | 976,896 | 1,432,944 | — |
| All requests: request bytes | 1,493,840 | 1,296,442 | 2,790,282 | — |
| Embedded constants (`METADATA`) / inline input values | 33,462 / 25,996 | 43,292 / 33,408 | 76,754 / 59,404 | — |
| Assets read or written, tasks | 0, 0 | 0, 0 | 0, 0 | 0, 0 |
| Requests in flight | 1 | 1 | 1 | 4 |

Besides its requests, each run's `ee.Initialize` makes the client's own reads: the Cloud API
discovery document (fetched twice), one `algorithms.list`, and the public catalogue of
deprecated assets. The harness lets only `algorithms.list` (at most 2) and
`image.computePixels` (at most the planned number) reach the Earth Engine transport, and the
client executes each of them once (no retries). The HTTP layer below may still re-send a
request after a credential refresh or a dropped connection.

## What each fixture should show

Each must pass at 1e-4 against PyTorch, and bit for bit against its mirror: every mirror is
exact (the fixture's `notes` give the reason). Every input echo must equal the NumPy input
bit for bit. The last column is `--offline p01` (mirror against PyTorch, largest absolute
difference over the fixture's outputs, zero cloud calls).

| Fixture (per window) | Reference | Mirror (exact) | Mirror vs PyTorch offline, windows 5 / 8 / 12 |
|---|---|---|---|
| `relative_position_bias/w{w}` | timm `WindowAttention(window_size=w)._get_rel_pos_bias()`, 3 heads | `swin_block` `Mirror(window=w)` bias: a one-hot gather, exact whatever the order | 0 / 0 / 0 |
| `attention/w{w}-s{s}` | timm `SwinTransformerBlock(window_size=w, shift_size=s)` SW-MSA and `shift_size=0` W-MSA, non-fused, to the projection; the shifted block's `attn_mask`; the SW-MSA heads moved back by `window_reverse` and `torch.roll(+s)` | `swin_pair` `ShiftedMirror(window=w, shift=s)` (with its `unshift`) and `swin_block` `Mirror(window=w)`, sequential sums | wmsa 4.77e-7 / 1.43e-6 / 3.46e-6; concat and unshifted 5.96e-7 / 2.38e-6 / 6.68e-6; mask 0 / 0 / 0 |
| `attention/w{w}-single` | timm `SwinTransformerBlock(input_resolution=(w, w), shift_size=s)`: timm sets the shift to 0; its attention branch | `swin_block` `Mirror(window=w)`, sequential sums | 4.77e-7 / 1.19e-6 / 2.38e-6 |
| `layer_norm2d/w{w}-eps1e-6` | SMP `LayerNorm2d(8, eps=1e-6)` | `ee_mirror_decoder.layer_norm2d` | 3.58e-7 / 4.77e-7 / 4.77e-7 |
| `patch_merge/w{w}` | timm `PatchMerging(dim=8)`, step by step | `swin_hierarchy` `PatchMergeMirror(window=w)`, sequential sums | cat 0, norm 2.38e-7, out 4.77e-7 / 0, 3.58e-7, 4.77e-7 / 0, 3.58e-7, 7.15e-7 |
| `conv3x3_window/k3-w{w}` | `F.conv2d(padding=1)` | `ee_mirror_decoder.conv3x3` (im2col) | 1.79e-7 / 2.38e-7 / 4.77e-7 |

At window 7 the regression fixtures should reproduce the calibration run
([evidence](live-fixtures/20261013T033726Z-calibration.json)): bit-identical to their mirrors,
and 1.31e-6 at most from PyTorch. `attention/w7-single` is new: 9.54e-7 offline.

**One value may differ by summation order.** In `attention/w12-s6`, one W-MSA P·V value
(about −3.3e-4) lies within a reordering error bound of a float32 rounding boundary (the
P01 review's margin ratio 0.85). If the `wmsa` readout differs from the mirror in that one
value alone, it points to Earth Engine's summation order, not to a library error.

## The likeliest live failure

`relative_position_bias` gathers the table with a one-hot matrix of [N², (2w − 1)²] values,
built on the server from integer sequences, so neither the request size nor the embedded-constant
count shows it:

| Window | One-hot values |
|---|---:|
| 5 | 50,625 |
| 7 | 405,769 (production) |
| 8 | 921,600 |
| 12 | 10,969,344 |

At window 12 it is 27 times production's. Three window-12 fixtures build it
(`relative_position_bias/w12`, `attention/w12-s6`, `attention/w12-single`); they run last in
`p01b`, after the window-12 fixtures that do not. If they fail with a memory error, the
blocks keep "Fixed" at window 12, and the brief's rule applies: the reason is written in the
docstring and status. A gather that scales better would change the graph at other windows
only, and is left to a later task.

## What is left behind

- Two evidence files, `docs/evidence/live-fixtures/<UTC time>-p01a.json` and `...-p01b.json`,
  which the lead commits with the results.
- Nothing in Earth Engine: no asset is created, read or deleted, and no task is started. The
  harness refuses any such call before it is sent.

## Pass criteria and stop rules

- **Pass:** each run exits 0 with `"status": "passed"`: every fixture passes as above, and
  `cloud_calls` shows the planned requests (21 and 8) and no refused call.
- **A fixture fails** (exit 1): the evidence names each failing value by output, row, column
  and channel. A block that fails at a window loses that window from its promotion (brief, R6).
  Nothing is retried automatically.
- **A run aborts** (a transient error, a graph that differs from the plan, a refused call):
  the evidence records why. A re-run is a new invocation, within the same standing budget.
