# P03 live plan: the patch embedding at patches 2, 4 and 8

Status: **planned; not run.** The lead runs it under the user's standing budget, on the
reviewed P03 revision. The implementer does not run it.

The run checks `patch_embed` and `stride_columns` live at patches 2, 4 and 8, with 1 to 4
input bands under arbitrary names, at widths 48, 96 and 128, and in windows 5, 7, 8 and 12.
The references are timm's `PatchEmbed` with random weights. Today's patch 4, VV/VH, width 96
and window 7 is repeated as the regression check. Every number below is the output of
`PYTHONPATH=. python tools/live_fixtures.py --plan p03` with earthengine-api 1.7.45. Request
bytes can change with the client's serialization. `tests/test_p03_patch_embedding.py` checks
these tables against `--plan`.

## Common setup

| Item | Value |
|---|---|
| Project | `ORIGINAL_PROJECT`, the project of the P00 calibration run and the P01 and P02 runs (the plan's validation rules) |
| Credentials | Application Default Credentials, scopes `earthengine` and `cloud-platform` (`live_session`); nothing is saved |
| Network | The IPv4-first network shim the lead uses for Earth Engine calls (plan, standing constraints) |
| Code | `lib/p03-patch-embedding` at the reviewed revision, from a checkout with `PYTHONPATH=.`; earthengine-api 1.7.45, torch 2.14.0, timm 1.0.30 |
| Concurrency | One request at a time; one harness invocation at a time |
| Inputs and weights | Inline: built in the graph from NumPy values and embedded constants; no asset is read |

## The command

One run within one run's budget: 14 requests.

```sh
PYTHONPATH=. python tools/live_fixtures.py --plan p03
PYTHONPATH=. python tools/live_fixtures.py --run p03 --project ORIGINAL_PROJECT
```

`--run` refuses to start (exit 2, before any cloud call) if its plan exceeds the budget or a
reference or mirror fails locally. It then initializes Earth Engine and rebuilds its graphs.
It stops unless each graph equals its planned SHA-256 (below, first 12 hex digits). It sends
the requests in the order listed.

## The selection

Seven fixtures. Each value of each setting appears at least once.

| Fixture | Patch | Bands (the block's order) | Width | Window | Windows | Input (pixels) | Tokens |
|---|---:|---|---:|---:|---|---|---|
| `window/patch_embed/p4-vvvh-d96` (regression: P00's calibration fixture) | 4 | VV, VH | 96 | 7 | 2 × 2 | 56 × 56 | 14 × 14 |
| `window/patch_embed/p2-b3-d48-w5` | 2 | B4, B3, B2 (the raster stores B2, B3, B4) | 48 | 5 | 2 × 3 | 20 × 30 | 10 × 15 |
| `window/patch_embed/p2-b1-d128-w8` | 2 | b0 | 128 | 8 | 3 × 2 | 48 × 32 | 24 × 16 |
| `window/patch_embed/p2-b2-d96-w12` | 2 | HH, HV | 96 | 12 | 2 × 2 | 48 × 48 | 24 × 24 |
| `window/patch_embed/p4-b4-d128-w12` | 4 | b0, b1, b2, b3 | 128 | 12 | 2 × 2 | 96 × 96 | 24 × 24 |
| `window/patch_embed/p8-b1-d96-w7` | 8 | b0 | 96 | 7 | 1 × 1 | 56 × 56 | 7 × 7 |
| `window/patch_embed/p8-b4-d48-w8` | 8 | b0, b1, b2, b3 | 48 | 8 | 1 × 1 | 64 × 64 | 8 × 8 |

The input pixels are 2.5 m (the Swin-T model grid), so tokens are 5, 10 and 20 m at
patches 2, 4 and 8. The input echo is a window-lane request on the input grid, so its side
may not exceed 8 windows. Patch 8 therefore runs on one window, and patch 4 on 2 × 2.

## Every request

All are `ee.data.computePixels` with `NUMPY_NDARRAY`. "Values" are output values (pixels ×
bands, with one validity band per readout). "Expected" is their float32 payload, without the
NPY header. "Request" is the serialized request body. "Constants" is the blocks'
embedded-constant count from their `METADATA` cost functions. "Inputs" is the inline input
values the graph embeds. "NPY header" is the length of the response's NPY header, as
`numpy.lib.format` writes it for that structured float32 array (`npy_header_length`). Grids
are abstract `TokenGrid`s (origin (0, 0), EPSG:3857): request 0 of each fixture is its token
grid, request 1 its input grid.

| Fixture | Request | Grid (rows × columns) | Bands | Values | Expected bytes | Request bytes | Constants | Inputs | Readouts (channels) | Graph | NPY header |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|---:|
| `window/patch_embed/p4-vvvh-d96` | 0 | 14 × 14 | 194 | 38,024 | 152,096 | 194,299 | 17,184 | 6,272 | pe_proj 96, x 96 | `28d3b1ebab0d` | 3,958 |
| `window/patch_embed/p4-vvvh-d96` | 1 | 56 × 56 | 3 | 9,408 | 37,632 | 80,060 | 0 | 6,272 | input 2 | `aac111fe1e7e` | 182 |
| `window/patch_embed/p2-b3-d48-w5` | 0 | 10 × 15 | 111 | 16,650 | 66,600 | 77,703 | 4,176 | 1,800 | pe_columns 12, pe_proj 48, x 48 | `fee37de3f38c` | 2,358 |
| `window/patch_embed/p2-b3-d48-w5` | 1 | 20 × 30 | 4 | 2,400 | 9,600 | 26,489 | 0 | 1,800 | input 3 | `29323fddc057` | 182 |
| `window/patch_embed/p2-b1-d128-w8` | 0 | 24 × 16 | 263 | 100,992 | 403,968 | 103,853 | 25,088 | 1,536 | pe_columns 4, pe_proj 128, x 128 | `1d975e9ce499` | 5,430 |
| `window/patch_embed/p2-b1-d128-w8` | 1 | 48 × 32 | 2 | 3,072 | 12,288 | 21,956 | 0 | 1,536 | input 1 | `4ef340d51576` | 182 |
| `window/patch_embed/p2-b2-d96-w12` | 0 | 24 × 24 | 203 | 116,928 | 467,712 | 203,096 | 42,240 | 4,608 | pe_columns 8, pe_proj 96, x 96 | `10a11f33e12d` | 4,214 |
| `window/patch_embed/p2-b2-d96-w12` | 1 | 48 × 48 | 3 | 6,912 | 27,648 | 59,949 | 0 | 4,608 | input 2 | `cd999649e027` | 182 |
| `window/patch_embed/p4-b4-d128-w12` | 0 | 24 × 24 | 323 | 186,048 | 744,192 | 726,425 | 63,488 | 36,864 | pe_columns 64, pe_proj 128, x 128 | `978a67569a9b` | 6,966 |
| `window/patch_embed/p4-b4-d128-w12` | 1 | 96 × 96 | 5 | 46,080 | 184,320 | 452,906 | 0 | 36,864 | input 4 | `6f2f9cb5a7b4` | 182 |
| `window/patch_embed/p8-b1-d96-w7` | 0 | 7 × 7 | 259 | 12,691 | 50,764 | 237,885 | 20,256 | 3,136 | pe_columns 64, pe_proj 96, x 96 | `5895fee1692f` | 5,686 |
| `window/patch_embed/p8-b1-d96-w7` | 1 | 56 × 56 | 2 | 6,272 | 25,088 | 41,397 | 0 | 3,136 | input 1 | `c18e13b22286` | 182 |
| `window/patch_embed/p8-b4-d48-w8` | 0 | 8 × 8 | 355 | 22,720 | 90,880 | 572,584 | 21,504 | 16,384 | pe_columns 256, pe_proj 48, x 48 | `4fd5f431b0f6` | 8,886 |
| `window/patch_embed/p8-b4-d48-w8` | 1 | 64 × 64 | 5 | 20,480 | 81,920 | 204,429 | 0 | 16,384 | input 4 | `d43c3c6f81b1` | 182 |

The first two rows are the calibration's requests, with the same graphs.

## Totals against the budget

| Item | `p03` | Budget (P00, R4), per run |
|---|---:|---:|
| Fixtures | 7 | — |
| Interactive requests | 14 | 40 |
| Largest grid, window lane | 56 × 56 (window 7), 64 × 64 (window 8), 96 × 96 (window 12), 20 × 30 (window 5): each an input echo | 8 × 8 windows: 40 at window 5, 56 at 7, 64 at 8, 96 at 12 |
| Largest request, output values | 186,048 | 1,000,000 |
| Largest request, serialized bytes | 726,425 | 2,000,000 |
| Largest request, bands | 355 | 1,024 (computePixels' limit) |
| Largest NPY response header (bytes) | 8,886 (`p8-b4-d48-w8`, request 0) | 9,000 (`numpy.load` refuses more than 10,000) |
| All requests: values | 588,677 | — |
| All requests: expected bytes | 2,354,708 | — |
| All requests: request bytes | 3,003,031 | — |
| Embedded constants (`METADATA`) / inline input values | 193,936 / 141,200 | — |
| Assets read or written, tasks | 0, 0 | 0, 0 |
| Requests in flight | 1 | 4 |

The largest header is close to the limit: 355 bands, 256 of them `pe_columns`. The planner
computes headers exactly as `numpy.lib.format` writes them, and would split the request if it
were over 9,000 bytes.

Besides its requests, the run's `ee.Initialize` makes the client's own reads: the Cloud API
discovery document (fetched twice), one `algorithms.list`, and the public catalogue of
deprecated assets. The harness lets only `algorithms.list` (at most 2) and
`image.computePixels` (at most the planned number) reach the Earth Engine transport. The
client executes each of them once (no retries). The HTTP layer below may still re-send a
request after a credential refresh or a dropped connection.

## What each fixture should show

Each must pass at 1e-4 against PyTorch, and bit for bit against its mirror: every mirror is
exact (the fixture's `notes` give the reason). Every input echo must equal the NumPy input
bit for bit. The mirror-against-PyTorch columns are `--offline p03` (largest absolute
difference, zero cloud calls).

| Fixture | pe_columns | pe_proj | x | Certificates, `pe_proj` and LayerNorm means (bits, below 53) | `x` order ratio: smallest (values at most 1) |
|---|---:|---:|---:|---|---|
| `p4-vvvh-d96` (regression) | — | 3.58e-7 | 9.54e-7 | 49.9, 44.5 | 200 (0) |
| `p2-b3-d48-w5` | 0 | 1.19e-7 | 7.15e-7 | 48.6, 42.8 | 2,130 (0) |
| `p2-b1-d128-w8` | 0 | 5.96e-8 | 7.15e-7 | 47.7, 46.3 | 88.6 (0) |
| `p2-b2-d96-w12` | 0 | 1.19e-7 | 7.15e-7 | 48.4, 45.4 | 44.3 (0) |
| `p4-b4-d128-w12` | 0 | 9.54e-7 | 1.91e-6 | 50.6, 45.1 | 0.995 (1) |
| `p8-b1-d96-w7` | 0 | 7.15e-7 | 1.67e-6 | 50.5, 42.7 | 471 (0) |
| `p8-b4-d48-w8` | 0 | 2.38e-6 | 1.67e-6 | 52.3, 40.8 | 959 (0) |

- **References:** timm `PatchEmbed(img_size, patch_size, in_chans, embed_dim,
  norm_layer=nn.LayerNorm, output_fmt="NHWC")` with the fixture's random weights, run step
  by step. The columns come from timm's own Conv2d with one-hot weights, which must equal
  `F.unfold`'s patches.
- **Mirrors:** `F.unfold`'s patches for the columns. `pe_mirror(sequential_sums=True)` for
  the projection: each dot product summed left to right in double, the bias added in double,
  one rounding. Then LayerNorm with left-to-right sums in double and one rounding.
- **Summation order** (`tools/fixtures/patch.py::order_analysis`):
  - **Certificates.** Every `pe_proj` dot product (C·patch² products and the bias) and every
    LayerNorm mean sum (the float32 `pe_proj` of a token) has terms that are multiples of
    2^q, q the lowest set bit among them, with Σ|terms| below 2^(q+53). The column gives
    the widest, in bits. Every partial sum, in any order, is then a double, formed without
    rounding. So `pe_proj`, the means, the centred values and their squares are the same
    in every summation order.
  - **`x` order ratio.** Only the LayerNorm's variance sum depends on order. For each `x`
    value, its distance to the nearest float32 rounding boundary is divided by a first-order
    bound of how far any order of that sum can move it, with rounding slack. Above 1, every
    order rounds alike.

At patch 4, VV/VH, width 96 and window 7, the regression should reproduce the calibration run
([evidence](live-fixtures/20261013T033726Z-calibration.json)): bit-identical to its mirror,
3.58e-7 (`pe_proj`) and 9.54e-7 (`x`) from PyTorch.

**One value may differ by summation order.** In `p4-b4-d128-w12`, one `x` value lies within
the first-order reordering bound of the variance sum from a float32 rounding boundary. It is
about −0.7336123, at token row 12, column 16, channel 30 (window 3, token 4). The distance is
4.22e-15 and the bound 4.24e-15, a ratio of 0.995. The reversed, exact, NumPy, pairwise,
strided (2 to 16 lanes) and chunked (4 to 32) sums, and 2,000 random orders, all round it
alike. If `x` differs from the mirror in that one value alone, it points to Earth Engine's
summation order, not to a library error. P01's order-sensitive window-12 value (ratio 0.85)
passed live. No `pe_proj` value can differ by order.

## What is left behind

- One evidence file, `docs/evidence/live-fixtures/<UTC time>-p03.json`, which the lead
  commits with the results.
- Nothing in Earth Engine: no asset is created, read or deleted, and no task is started. The
  harness refuses any such call before it is sent.

## Pass criteria and stop rules

- **Pass:** the run exits 0 with `"status": "passed"`. Every fixture passes as above, and
  `cloud_calls` shows the 14 planned requests and no refused call.
- **A fixture fails** (exit 1): the evidence names each failing value by output, row, column
  and channel. A setting that fails live is left out of the blocks' statuses (brief, Outcome).
  Nothing is retried automatically.
- **A run aborts** (a transient error, a graph that differs from the plan, a refused call):
  the evidence records why. A re-run is a new invocation, within the same standing budget.
