# P00 calibration run plan: the live fixture harness at today's settings

Status: **run on 2026-10-13; passed.** The lead ran it under the user's standing budget for live fixture harness runs. The first attempt sent nothing to Earth Engine (a client-initialization bug, fixed in `574fe2c`); the second passed all 12 fixtures with the 18 requests below. Records: [first attempt](live-fixtures/20261013T033224Z-calibration.json), [passing run](live-fixtures/20261013T033726Z-calibration.json). The plan below is unchanged.

The calibration tries no new setting. It checks that the harness reproduces results the
milestones already recorded: a harness that cannot reproduce known results cannot promote
anything. Every number below is the output of
`PYTHONPATH=. python tools/live_fixtures.py --plan calibration`, with earthengine-api
1.7.45 (request bytes can change with the client's serialization;
`tests/test_live_fixtures.py` checks these tables against `--plan`).

## Common setup

| Item | Value |
|---|---|
| Project | `ORIGINAL_PROJECT`, the project of T01–T06 and L06c (the plan's validation rules) |
| Credentials | Application Default Credentials, scopes `earthengine` and `cloud-platform` (`live_session`); nothing is saved |
| Network | The IPv4-first network shim the lead uses for Earth Engine calls (plan, standing constraints) |
| Code | `lib/p00-live-fixture-harness` at the accepted revision, from a checkout with `PYTHONPATH=.`; earthengine-api 1.7.45, torch 2.14.0, timm 1.0.30, SMP 0.5.0 |
| Concurrency | One request at a time; no other harness invocation runs at once |
| Inputs and weights | Inline: built in the graph from NumPy values (25,944 input values in all) and embedded constants; no asset is read |

## The command

```sh
PYTHONPATH=. python tools/live_fixtures.py --plan calibration
PYTHONPATH=. python tools/live_fixtures.py --run calibration --project ORIGINAL_PROJECT
```

`--run` refuses to start (exit 2, before any cloud call) if the plan exceeds the budget or a
reference or mirror fails locally. It then initializes Earth Engine, rebuilds the 18 graphs
and stops unless each equals its planned SHA-256 (below, first 12 hex digits), and sends
the requests in the order listed.

## Every request

All are `ee.data.computePixels` with `NUMPY_NDARRAY`. "Values" are output values (pixels ×
bands, with one validity band per readout); "expected" is their float32 payload, without
the NPY header; "request" is the serialized request body; "constants" is the blocks'
embedded-constant count from their `METADATA` cost functions; "inputs" is the inline
input values the graph embeds. Raster grids are 20 m or 40 m `Grid`s in EPSG:3857; window
grids are abstract `TokenGrid`s (origin (0, 0), EPSG:3857, 10 m tokens, 2.5 m input
pixels for the patch embedding).

| Fixture | Request | Grid (rows × columns) | Bands | Values | Expected bytes | Request bytes | Constants | Inputs | Readouts (channels) | Graph |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| `raster/conv2d/k3` | 0 | 16 × 16 | 7 | 1,792 | 7,168 | 17,829 | 57 | 512 | out 3, input 2 | `55b4876639f6` |
| `raster/pool2d/avg-k2-s2` | 0 | 8 × 8 | 3 | 192 | 768 | 14,877 | 0 | 512 | out 2 | `1e16cd636714` |
| `raster/pool2d/avg-k2-s2` | 1 | 16 × 16 | 3 | 768 | 3,072 | 12,313 | 0 | 512 | input 2 | `a70404994d07` |
| `raster/pool2d/max-k3-s2-p1` | 0 | 8 × 8 | 3 | 192 | 768 | 17,073 | 0 | 512 | out 2 | `dbd4a41c0594` |
| `raster/pool2d/max-k3-s2-p1` | 1 | 16 × 16 | 3 | 768 | 3,072 | 12,365 | 0 | 512 | input 2 | `d7df8022f8af` |
| `raster/nearest_upsample/f2` | 0 | 16 × 16 | 3 | 768 | 3,072 | 9,936 | 0 | 128 | out 2 | `6f2a9d5a8375` |
| `raster/nearest_upsample/f2` | 1 | 8 × 8 | 3 | 192 | 768 | 7,615 | 0 | 128 | input 2 | `0f6382a738af` |
| `window/relative_position_bias/w7` | 0 | 49 × 1 | 100 | 4,900 | 19,600 | 16,269 | 338 | 0 | bias_h0 49, bias_h1 49 | `6fd9e92beff0` |
| `window/attention/w7-s3` | 0 | 14 × 14 | 84 | 16,464 | 65,856 | 133,029 | 3,458 | 3,136 | concat 16, mask 49, input 16 | `6e0dc7cbee71` |
| `window/layer_norm2d/w7-eps1e-6` | 0 | 14 × 14 | 18 | 3,528 | 14,112 | 71,809 | 784 | 1,568 | out 8, input 8 | `9b3865551a92` |
| `window/patch_merge/w7` | 0 | 7 × 7 | 83 | 4,067 | 16,268 | 93,669 | 3,648 | 1,568 | cat 32, norm 32, out 16 | `52d58538f73a` |
| `window/patch_merge/w7` | 1 | 14 × 14 | 9 | 1,764 | 7,056 | 26,738 | 0 | 1,568 | input 8 | `dc3e7658b7c1` |
| `window/conv3x3_window/k3-w7` | 0 | 14 × 14 | 10 | 1,960 | 7,840 | 73,920 | 340 | 784 | out 4, input 4 | `68438139db4f` |
| `window/group_product/g4-w7` | 0 | 14 × 14 | 14 | 2,744 | 10,976 | 208,749 | 288 | 1,568 | out 4, input 8 | `1e45467eb92c` |
| `window/window_pool/bins1236-w7` | 0 | 50 × 1 | 5 | 250 | 1,000 | 57,845 | 750 | 196 | pooled 4 | `e286e4b6aa81` |
| `window/window_pool/bins1236-w7` | 1 | 7 × 7 | 5 | 245 | 980 | 7,506 | 0 | 196 | input 4 | `e819ca0ae071` |
| `window/patch_embed/p4-vvvh-d96` | 0 | 14 × 14 | 194 | 38,024 | 152,096 | 194,299 | 17,184 | 6,272 | pe_proj 96, x 96 | `28d3b1ebab0d` |
| `window/patch_embed/p4-vvvh-d96` | 1 | 56 × 56 | 3 | 9,408 | 37,632 | 80,060 | 0 | 6,272 | input 2 | `aac111fe1e7e` |

## Totals against the budget

| Item | Plan | Budget (R4) |
|---|---:|---:|
| Interactive requests | 18 | 40 |
| Largest grid, raster lane | 16 × 16 pixels | 64 × 64 |
| Largest grid, window lane | 56 × 56 (patch-embedding input echo); 50 × 1 | 56 × 56 (8 × 8 windows) |
| Largest request, output values | 38,024 | 1,000,000 |
| Largest request, serialized bytes | 208,749 | 2,000,000 |
| Largest request, bands | 194 | 1,024 (computePixels' limit) |
| All requests: values / expected bytes | 88,026 / 352,104 | — |
| All requests: request bytes | 1,055,901 | — |
| Embedded constants (`METADATA`) / inline input values, summed over requests | 26,847 / 25,944 | — |
| Assets read or written, tasks | 0, 0 | 0, 0 |
| Requests in flight | 1 | 4 |

Besides the 18 requests, `ee.Initialize` makes the client's own reads: the Cloud API
discovery document (fetched twice, for the two API resources), one `algorithms.list`, and
the public catalogue of deprecated assets. The harness lets only `algorithms.list` (at most
2) and `image.computePixels` (at most the 18 planned) reach the Earth Engine transport,
and the client executes each of them once (no retries). The HTTP layer below may still
re-send a request after a credential refresh or a dropped connection.

## What each fixture should reproduce

Each must pass at 1e-4 against PyTorch, and bit for bit against its mirror where the mirror
is exact. The max pool and the nearest upsampling select or copy values, so their tolerance
is 0: they must equal PyTorch bit for bit. Every input echo must equal the NumPy input bit
for bit. `--offline` already gives the mirror against PyTorch (largest difference over all
outputs, zero cloud calls).

| Fixture | Reference | Mirror (exact?) | Mirror vs PyTorch offline | Recorded result to reproduce |
|---|---|---|---:|---|
| `raster/conv2d/k3` | `nn.Conv2d(2, 3, 3, padding=1)` | none: no R mirror exists | — | [primitives_float32.json](../history/cnn/results/primitives_float32.json), `conv3_asymmetric` (Conv2d 2 → 3, 3 × 3, 16 × 16): pass, 3.28e-7 |
| `raster/pool2d/avg-k2-s2` | `nn.AvgPool2d(2, 2)` | none | — | [primitives_float32.json](../history/cnn/results/primitives_float32.json), `average_pool_nearest`: pool pass, 7.45e-9 |
| `raster/pool2d/max-k3-s2-p1` (all-negative inputs, tolerance 0) | `nn.MaxPool2d(3, 2, 1)` | none | — | [resnet_primitives.json](../history/cnn/results/resnet_primitives.json): `max3_stride2_padded_even`, pass, 0; `max3_stride2_padded_negative_odd` (negative inputs, 17 × 15), pass, 5.2e-8, from inputs that were not float32-exact (input error 5.96e-8) |
| `raster/nearest_upsample/f2` (tolerance 0) | `F.interpolate(scale_factor=2, mode="nearest")` | none | — | [primitives_float32.json](../history/cnn/results/primitives_float32.json), `max_pool_nearest`: up pass, 0 |
| `window/relative_position_bias/w7` | timm `WindowAttention._get_rel_pos_bias()` | `swin_block` `Mirror` (yes) | 0 | [swin_block findings](../history/swin_block/notes/findings.md), answer 8: equal to timm bit for bit |
| `window/attention/w7-s3` | timm `SwinTransformerBlock(shift_size=3)`, SW-MSA up to the projection; its `attn_mask` | `swin_pair` `ShiftedMirror`, sequential sums (yes) | 1.31e-6 | [swin_pair findings](../history/swin_pair/notes/findings.md), answer 10 and tests 4–5: mask element-identical to timm's; attention bit-identical to the mirror, ≤ 7.6e-6 vs PyTorch |
| `window/layer_norm2d/w7-eps1e-6` | SMP `LayerNorm2d(eps=1e-6)` | `ee_mirror_decoder.layer_norm2d` (yes) | 4.77e-7 | [Milestone 7](../history/upernet/notes/milestone7_findings.md): LayerNorm2d bit-identical to the mirror, ≤ 5.7e-6 vs SMP |
| `window/patch_merge/w7` | timm `PatchMerging(dim=8)`, step by step | `PatchMergeMirror`'s steps with sequential LayerNorm sums (yes) | 4.77e-7 | [Milestone 4](../history/swin_hierarchy/notes/milestone4_findings.md), answers 1, 4, 5, 13: gather exact; LayerNorm and reduction bit-identical to the mirror, ≤ 3.6e-6 vs PyTorch |
| `window/conv3x3_window/k3-w7` | `F.conv2d(padding=1)` | `ee_mirror_decoder.conv3x3`, window im2col (yes) | 3.58e-7 | [Milestone 7](../history/upernet/notes/milestone7_findings.md): 3 × 3 window im2col bit-identical to the mirror, ≤ 8.1e-6 vs PyTorch |
| `window/group_product/g4-w7` | `F.conv2d(padding=1)` of the unsplit weight | `ee_mirror_fusion.sum_partials` of `ee_mirror_fpn.conv3x3`, grouped double (yes) | 8.34e-7 | [Milestone 10](../history/upernet/notes/milestone10_findings.md), "double-partial (B) sum": bit-identical to the mirror, 1.93e-5 vs SMP |
| `window/window_pool/bins1236-w7` | SMP `PSPModule(sizes=(1, 2, 3, 6))` pooling blocks | `ee_mirror_psp.pool`, window order (yes) | 2.98e-8 | [Milestone 8](../history/upernet/notes/milestone8_findings.md), answers 6 and 8: bit-identical to the mirror, ≤ 3.6e-6 vs SMP |
| `window/patch_embed/p4-vvvh-d96` | timm `PatchEmbed(patch_size=4, in_chans=2, embed_dim=96)`, step by step | `pe_mirror` (yes) | 9.54e-7 | [Milestone 6](../history/swin_image_encoder/notes/milestone6_findings.md), answers 4, 5, 8: bit-identical to the mirror; ≤ 2.4e-6 (projection) and 1.67e-6 (LayerNorm) vs timm |

## What is left behind

- One evidence file, `docs/evidence/live-fixtures/<UTC time>-calibration.json`, which the
  lead commits with the result.
- Nothing in Earth Engine: no asset is created, read or deleted, and no task is started.
  The harness refuses any such call before it is sent.

## Pass criteria and stop rules

- **Pass:** exit 0 and `"status": "passed"`: every one of the 12 fixtures passes as above,
  and `cloud_calls` shows 18 requests and no refused call.
- **A fixture fails** (exit 1): the evidence names each failing value by output, row,
  column and channel. The lead reports it to the user before any Phase 2 lane relies on the
  harness. Nothing is retried automatically: the client executes each request once (the
  HTTP layer may re-send one after a credential refresh or a dropped connection).
- **The run aborts** (a transient error, a graph that differs from the plan, a refused
  call): the evidence records why, and the lead reports it. A re-run is a new invocation,
  within the same standing budget.

The P01–P04 briefs are written after the calibration, citing its evidence.
