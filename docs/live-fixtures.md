# Live fixtures

Settings validation promotes Fixed blocks by checking them at new settings. The live fixture harness,
[`tools/live_fixtures.py`](../tools/live_fixtures.py), is the machinery every lane uses:

- it builds one block's Earth Engine graph on a small input that lives in the graph;
- it evaluates the graph with a few interactive `computePixels` requests;
- it compares the result with PyTorch and with the block's NumPy mirror;
- it records the result as evidence, inside a request budget the user approved once.

The fixtures are registered in [`tools/fixtures/`](../tools/fixtures/).
The harness was calibrated at the library's validated defaults
([calibration plan](evidence/p00-calibration-plan.md)).

## Commands

Run from a checkout, with `PYTHONPATH=.`:

```sh
python tools/live_fixtures.py --list
python tools/live_fixtures.py --plan calibration
python tools/live_fixtures.py --offline calibration [--output path.json]
python tools/live_fixtures.py --run calibration --project YOUR_EE_PROJECT [--budget requests=10]
```

A selection is one or more fixture ids, `fnmatch` patterns such as `'window/*'`,
`calibration` (the fixtures at the library defaults), `all`, or a lane's named selection, such as `p02`. Commas
also separate items. A lane module names its selections in a `SELECTIONS` mapping: each
name maps to a sequence of fixture ids, or to a zero-argument callable that returns one.
A name may not be a fixture id, `all` or `calibration`.

| Mode | What it does | Cloud calls |
|---|---|---|
| `--list` | Every fixture: block, settings, policy, request count, mirror | none |
| `--plan` | Every request a run would make: grid, bands, output values, expected bytes, request bytes, graph SHA-256, and the embedded-constant count from the blocks' `METADATA` cost functions; then the totals against the budget. `--json` prints the same as JSON. Exits 1 when over budget | none |
| `--offline` | `--plan`'s graphs, plus every PyTorch reference and NumPy mirror, comparing mirror with PyTorch. This is the G3-offline mechanism | none |
| `--run` | The live run, described below. It writes one evidence JSON | interactive `computePixels` only |

`--plan` and `--offline` build graphs against the offline signatures of
`ee.apitestcase.ApiTestCase`, with every Earth Engine cloud entry point patched to raise,
as `tools/graph_identity.py` does.

Exit codes: 0 when everything passed; 1 when a fixture failed, or when the run stopped after a
cloud call (the evidence is still written); 2 when the run was refused before any cloud call.

## The fixture record

A fixture is a frozen record (`tools/fixtures/core.py::Fixture`):

| Field | Meaning |
|---|---|
| `id` | Stable and unique: `lane/block/setting`, for example `window/attention/w7-s3` |
| `lane`, `block`, `parts` | `raster` or `window`; the catalogue block it exercises, and other catalogue blocks it runs on the way |
| `settings` | The block settings, named as the parameters of the callables in `accepts` (the block, its constants builder, or the PyTorch module an R block takes) |
| `policy` | The precision policy the graph uses, as the [catalogue](building-blocks.md) gives it: `float32_operands`, `round_each_op`, `double_round_once`, the variant `grouped_double`, or `composite` |
| `shapes`, `seed` | Input shape, channels, heads and grid; the seed of the inputs and the random weights |
| `data(seed)` | The NumPy inputs ([H, W, C] float32, built in the graph) and weights (float32, embedded in the request) |
| `build(ee, data)` | The graph, as readouts: scalar float32 bands on a computePixels grid |
| `reference(data)` | The PyTorch output of every output readout: timm, SMP or `torch.nn.functional`, with the random weights |
| `mirror` | Optional: the NumPy mirror from `tests/reference`, and whether it models Earth Engine's rounding exactly (`exact`) or within a recorded `bound`, which may not exceed the fixture's tolerance |
| `tolerance` | 1e-4 absolute against PyTorch. A fixture may set a smaller one, never a larger one (refused by the record). A tolerance of 0 compares float32 bits, so -0.0 against +0.0 fails; selection and copy blocks use it |
| `window` | The window of the graph, for the window lane's request budget; it must equal `settings["window"]` (refused otherwise) |
| `requests` | The expected number of interactive requests; `--plan` refuses a fixture whose graphs need another number |
| `constants(data)` | The embedded-constant count of its blocks, from their `METADATA` cost functions |
| `history` | The recorded result the fixture should reproduce, with its file |

**Deterministic data.** Values come from a SplitMix64 integer hash of `(seed, name)`, mapped
to dyadic rationals that are exact in float32. Only integer arithmetic and exact float
operations are used, so the same seed gives the same bytes on every machine;
`tests/test_live_fixtures.py` pins their SHA-256.

**Inline inputs.** Each input is built in the graph from its NumPy values: one embedded
constant per channel (the shortest decimals that round back to the float32 values, then
`.float()`, as the library's `embedded_raster` does), read per pixel at the pixel
coordinates. The values are checked first: finite float32, never -0.0, and below 2^53 in
magnitude. The float32 guarantee itself is `encode_float32`'s assertion that its decimals
round back to the float32 values. A periodic variant takes the
index modulo the raster size, so a window layout built from it wraps exactly under Swin's
cyclic shift.

**Readouts.** A raster (R) output is read on its own grid. A window (W) output is read as
dense bands on its token grid with `to_dense`. A value (V) output, an `ee.Array`, is read
row by row on a one-column grid. Readouts only move values. Each readout also returns one
validity band, so a masked pixel fails instead of reading as 0. Readouts on the same grid
share a request. Every fixture also reads its inline input back (`input`), which must
equal the NumPy input bit for bit.

## The budget

The limits are enforced in code (`Budget` in the harness). The user's standing approval
covers runs within them.

| Limit | Default |
|---|---|
| Interactive requests per invocation | 40 |
| Request grid, raster lane | 64 × 64 pixels |
| Request grid, window lane | 8 × 8 windows (56 × 56 tokens at window 7) |
| Output values per request | 1,000,000 |
| Serialized request | 2,000,000 bytes |
| Assets, tasks | 0 |
| Requests in flight | 4 (the harness sends one at a time) |
| NPY response header per request | 9,000 bytes (numpy's `max_header_size` is 10,000); readouts on one grid are split across requests to stay within it |

`--budget key=value,...` lowers a limit (`requests`, `raster_side`, `window_side`,
`values`, `request_bytes`, `in_flight`) and refuses to raise one. Raising a limit is a maintainer
change, recorded in the plan with the user's approval. The plan also checks
computePixels' own limit of 1,024 bands per request.

## What `--run` does

1. Builds every graph offline and checks the budget. A plan over budget is refused before
   any cloud call (exit 2).
2. Runs every reference and mirror locally, with every Earth Engine entry point refused.
   A failure, or a reference that reaches Earth Engine, is refused before any cloud call.
3. Wraps `ee.data` for the rest of the run:
   - the transport (`ee.data._execute_cloud_call`) lets only two Cloud API methods
     through: `projects.algorithms.list` at initialization (at most 2) and
     `projects.image.computePixels` (at most the planned number). The client executes
     each once, without retries; the HTTP layer below may still re-send a request after a
     credential refresh or a dropped connection;
   - every other public callable in `ee.data` (exports, assets, tasks, operations,
     downloads, map tiles, thumbnails, other computations) is replaced by one that
     raises. A refused call stops the run before anything is sent.
4. Initializes Earth Engine with Application Default Credentials for `--project`
   (`--credentials adc`, the only option).
5. Rebuilds every graph and requires its SHA-256 to equal the planned one, so the
   requests sent are the requests planned.
6. Sends the requests one at a time with `ee.data.computePixels` (`NUMPY_NDARRAY`).
7. Compares each output with PyTorch (largest absolute difference, values over the
   tolerance, or values whose bits differ at tolerance 0) and with the mirror (values whose
   float32 bits differ, largest difference).
8. Writes `docs/evidence/live-fixtures/<UTC time>-<selection>.json` (or `--output`).

A fixture passes when every output is float32, every pixel is valid, every value is
within the tolerance of PyTorch, an exact mirror is equal bit for bit (or within its
bound), and its input echo equals the NumPy input bit for bit. Any failing fixture fails
the run.

## Reading the evidence

One JSON per run (`schema_version` 1, `kind` `live_fixture_run`):

- `status`: `passed`, `failed`, or `aborted` (the run stopped; `errors` says why);
- `revision` and `dirty`, `versions` (Python, earthengine-api, torch, timm, SMP, NumPy),
  `platform`, `project` as given, `command`, `budget`, `plan` (the totals);
- `cloud_calls`: the requests made, the computePixels and `algorithms.list` calls that
  reached the transport, and any refused call;
- `fixtures`, one entry each:
  - `spec`: the fixture record without its callables;
  - `inputs_sha256`, `weights_sha256`;
  - `requests`: grid, bands, values, bytes, graph SHA-256 and size, wall time, status;
  - `offline`: mirror against PyTorch, computed before the run;
  - `outputs`, per readout: `pytorch` (`max_abs`, `over_tolerance`, the `worst` value
    with its row, column and channel), `mirror` (`differing`, `max_abs`, the `first`
    mismatches), `input` for the echo, and the dtype and validity checks;
  - `status`, `failures` (each failing value named by output, row, column and channel),
    `wall_seconds`.

The record holds no credentials and no local paths. The project ID is inherent to live
evidence, as in [the browser-demo validation run](evidence/l06c-live-run.json).

`--offline --output` writes the same per-fixture fields without the live ones
(`kind` `offline_fixture_check`).

## Adding a fixture at a new setting

1. Add the fixture to its lane's module (`tools/fixtures/raster.py` or `window.py`) with
   `calibration=False`, the new `settings` named as the block's parameters, and
   `accepts` listing the callables that take them.
2. Pass the setting to the mirror. A mirror that cannot take it yet is parameterized in
   `tests/reference`, offline, first.
3. `--offline <id> --output docs/evidence/<lane>-<name>-offline.json`: the G3-offline
   evidence (mirror against PyTorch within 1e-4, zero cloud calls).
4. `--plan <id>` shows the requests; the maintainer runs `--run <id>` within the standing budget
   for the G3-live evidence.

`tests/test_live_fixtures.py` checks every registered fixture: its block and parts are in
the catalogue, its policy is the block's, and each setting is a parameter of a callable
in `accepts`. For a calibration fixture, each setting must also equal that parameter's
library default.

## Calibration fixtures

The library's validated defaults, in both layouts. Every original Fixed block appears as a
fixture's block or part. The [calibration plan](evidence/p00-calibration-plan.md) lists
the requests and the results each fixture should reproduce.

| Fixture | Block | Settings | Mirror |
|---|---|---|---|
| `raster/conv2d/k3` | `conv2d` | kernel 3, stride 1, padding 1 | none (PyTorch only) |
| `raster/pool2d/avg-k2-s2` | `pool2d` | average 2 × 2, stride 2 | none |
| `raster/pool2d/max-k3-s2-p1` | `pool2d` | max 3 × 3, stride 2, padding 1; all-negative inputs; tolerance 0 | none |
| `raster/nearest_upsample/f2` | `nearest_upsample` | factor 2; tolerance 0 | none |
| `window/relative_position_bias/w7` | `relative_position_bias` | window 7 | exact |
| `window/attention/w7-s3` | `window_attention` with `shifted_window_mask` | window 7, shift 3, mask −100 | exact |
| `window/layer_norm2d/w7-eps1e-6` | `layer_norm2d` | window 7, eps 1e-6 | exact |
| `window/patch_merge/w7` | `patch_merge` with `gather_windows`, `merge_windows` | window 7 | exact |
| `window/conv3x3_window/k3-w7` | `conv3x3_window` | kernel 3, window 7 | exact |
| `window/group_product/g4-w7` | `group_product` (grouped double) | 4 groups | exact |
| `window/window_pool/bins1236-w7` | `window_pool` | bins (1, 2, 3, 6), one window | exact |
| `window/patch_embed/p4-vvvh-d96` | `patch_embed` with `stride_columns` | patch 4, VV/VH, 96 channels | exact |

No NumPy mirror of the raster path's Earth Engine arithmetic exists in `tests/reference`,
so the raster fixtures compare with PyTorch only. Max pooling and nearest upsampling
select or copy values, so they must equal PyTorch bit for bit (tolerance 0); the recorded
runs found 0 difference there. The max pool's inputs are all negative, so a padding that
contributed (a zero) would change every border window.
