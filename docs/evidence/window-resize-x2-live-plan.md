# Window resize x2 with aligned corners

This extends the evidence for the existing `window_resize` setting
`factor=2, align_corners=True`. The implementation already accepts it; the
arithmetic, casts and defaults are unchanged. The base is `924a7c2`.

## Layout choice

Use W7 for every map in **7 → 14 → 28 → 56 → 112 → 224**. Each map is divisible
by seven, and `window_resize` preserves its input window size at the output.
The coarsest 7×7 map rules out W14 as one common layout for this chain. W7
therefore needs no new window size and no changes of window size between
resizes. The future hand-composed DPT target must explicitly convert any
backbone W14 features to its decoder layout; this change adds no model.

The block keeps WINDOW_RING locality with radius one in source tokens.
With aligned corners, coefficients use absolute output positions and the
whole map's size, so the setting remains chip-edge dependent. The weights
are PyTorch's float32 one-hot interpolation coefficients, applied y then x
in double with one float32 rounding at the operation boundary.

## Evidence and scope

`tools/fixtures/resize_x2.py` registers one deterministic two-channel fixture
per arrow, under selection `resize_x2`. References call
`F.interpolate(scale_factor=2, mode="bilinear", align_corners=True)`.
The standalone NumPy mirror is extracted from the historical head mirror,
which delegates to it and keeps its API. It needs no optional model package.

Outputs up to 56×56 are read completely. The 112×112 and 224×224 outputs
are read as respectively two and four diagonal 56×56 blocks. These cover
every row and column coefficient pattern and both chip borders, but do not
cover every row/column pair. All source pixels are echoed: the 112×112
input uses four 56×56 tiles. Offline graph evaluation checks every planned
readout against the mirror bit for bit; full-output mirror comparisons
against PyTorch and wrong-corner/per-window negative controls run offline.
These checks establish no live agreement or model accuracy.

The exact-mirror flag follows the historical non-fused double x-stage
products and sum. A live failure would be recorded and investigated;
the tolerance stays 1e-4 and the run is never retried.

## Commands and live budget

Run offline from a clean implementation commit, then commit the generated
offline record before requesting live approval:

```sh
PYTHONPATH=. python tools/live_fixtures.py --offline resize_x2 --output docs/evidence/window-resize-x2-offline.json
PYTHONPATH=. python tools/live_fixtures.py --plan resize_x2
```

The plan is **17 interactive computePixels requests**, zero assets, zero
tasks, one request at a time. Every grid is at most 56×56, within eight
windows per side at W7. The fixtures are:

| Fixture | Source → output | Output requests | Input requests |
|---|---|---:|---:|
| `window/window_resize/ac-f2-e7-w7` | 7 → 14 | 1 | 1 |
| `window/window_resize/ac-f2-e14-w7` | 14 → 28 | 1 | 1 |
| `window/window_resize/ac-f2-e28-w7` | 28 → 56 | 1 | 1 |
| `window/window_resize/ac-f2-e56-w7` | 56 → 112 | 2 | 1 |
| `window/window_resize/ac-f2-e112-w7` | 112 → 224 | 4 | 4 |

After explicit approval of selection and request count, the harness runs
exactly once using ADC and `--budget requests=17`. It allows only the
planned computePixels calls and at most two algorithms-list calls for
initialization. No EECU estimate is available for this interactive plan.
The exact live record is committed before any settings are added to the
`validated` record or appended to the existing status text. Until then,
the x2 aligned-corners setting is outside that record.

Full-output mirror-vs-PyTorch maximum absolute errors on the fixture inputs:
7→14 and 14→28: 5.960464477539063e-8; 28→56, 56→112 and 112→224:
1.1920928955078125e-7. These are offline results under the unchanged 1e-4 target.

## Full offline plan output

```text
window/window_resize/ac-f2-e7-w7 (window_resize, double_round_once); inputs f706b8c58151, weights e3b0c44298fc
  request 0: 14 x 14 grid (rows x columns), 3 bands (out_0[2]); 588 values, 2,352 bytes expected; request 62,926 bytes; graph 5a56d1f2ff1d; embedded constants 256 (METADATA) + 98 input values; NPY header 182 bytes
  request 1: 7 x 7 grid (rows x columns), 3 bands (input_0_0[2]); 147 values, 588 bytes expected; request 5,072 bytes; graph bbfe941cf6d8; embedded constants 0 (METADATA) + 98 input values; NPY header 182 bytes
window/window_resize/ac-f2-e14-w7 (window_resize, double_round_once); inputs 4bb1d3121921, weights e3b0c44298fc
  request 0: 28 x 28 grid (rows x columns), 3 bands (out_0[2]); 2,352 values, 9,408 bytes expected; request 67,494 bytes; graph 7914b62d6787; embedded constants 512 (METADATA) + 392 input values; NPY header 182 bytes
  request 1: 14 x 14 grid (rows x columns), 3 bands (input_0_0[2]); 588 values, 2,352 bytes expected; request 8,668 bytes; graph 9ae48e1637a3; embedded constants 0 (METADATA) + 392 input values; NPY header 182 bytes
window/window_resize/ac-f2-e28-w7 (window_resize, double_round_once); inputs a9fae2b56e8e, weights e3b0c44298fc
  request 0: 56 x 56 grid (rows x columns), 3 bands (out_0[2]); 9,408 values, 37,632 bytes expected; request 83,471 bytes; graph d8eef4a5bb98; embedded constants 1,024 (METADATA) + 1,568 input values; NPY header 182 bytes
  request 1: 28 x 28 grid (rows x columns), 3 bands (input_0_0[2]); 2,352 values, 9,408 bytes expected; request 22,958 bytes; graph 0a0476cb81a5; embedded constants 0 (METADATA) + 1,568 input values; NPY header 182 bytes
window/window_resize/ac-f2-e56-w7 (window_resize, double_round_once); inputs b5cbc271ae51, weights e3b0c44298fc
  request 0: 56 x 56 grid (rows x columns), 3 bands (out_0[2]); 9,408 values, 37,632 bytes expected; request 144,161 bytes; graph 79f754d532fd; embedded constants 2,048 (METADATA) + 6,272 input values; NPY header 182 bytes
  request 1: 56 x 56 grid (rows x columns), 3 bands (out_56[2]); 9,408 values, 37,632 bytes expected; request 144,169 bytes; graph 41089ce071aa; embedded constants 2,048 (METADATA) + 6,272 input values; NPY header 182 bytes
  request 2: 56 x 56 grid (rows x columns), 3 bands (input_0_0[2]); 9,408 values, 37,632 bytes expected; request 80,205 bytes; graph 8c444ec146f0; embedded constants 0 (METADATA) + 6,272 input values; NPY header 182 bytes
window/window_resize/ac-f2-e112-w7 (window_resize, double_round_once); inputs 427dc9fcee48, weights e3b0c44298fc
  request 0: 56 x 56 grid (rows x columns), 3 bands (out_0[2]); 9,408 values, 37,632 bytes expected; request 379,660 bytes; graph c578b040f233; embedded constants 4,096 (METADATA) + 25,088 input values; NPY header 182 bytes
  request 1: 56 x 56 grid (rows x columns), 3 bands (out_56[2]); 9,408 values, 37,632 bytes expected; request 379,668 bytes; graph 8f978092c343; embedded constants 4,096 (METADATA) + 25,088 input values; NPY header 182 bytes
  request 2: 56 x 56 grid (rows x columns), 3 bands (out_112[2]); 9,408 values, 37,632 bytes expected; request 379,671 bytes; graph d7c31e576b8c; embedded constants 4,096 (METADATA) + 25,088 input values; NPY header 182 bytes
  request 3: 56 x 56 grid (rows x columns), 3 bands (out_168[2]); 9,408 values, 37,632 bytes expected; request 379,671 bytes; graph 0ed5be45b418; embedded constants 4,096 (METADATA) + 25,088 input values; NPY header 182 bytes
  request 4: 56 x 56 grid (rows x columns), 3 bands (input_0_0[2]); 9,408 values, 37,632 bytes expected; request 308,858 bytes; graph d22cfecc6a29; embedded constants 0 (METADATA) + 25,088 input values; NPY header 182 bytes
  request 5: 56 x 56 grid (rows x columns), 3 bands (input_0_56[2]); 9,408 values, 37,632 bytes expected; request 308,863 bytes; graph b9b6205ac7f1; embedded constants 0 (METADATA) + 25,088 input values; NPY header 182 bytes
  request 6: 56 x 56 grid (rows x columns), 3 bands (input_56_0[2]); 9,408 values, 37,632 bytes expected; request 308,864 bytes; graph 2bf3b037bcb0; embedded constants 0 (METADATA) + 25,088 input values; NPY header 182 bytes
  request 7: 56 x 56 grid (rows x columns), 3 bands (input_56_56[2]); 9,408 values, 37,632 bytes expected; request 308,869 bytes; graph 656ec861e69d; embedded constants 0 (METADATA) + 25,088 input values; NPY header 182 bytes
Totals: 17 requests (budget 40); largest request 9,408 values (budget 1,000,000) and 379,671 bytes (budget 2,000,000); largest NPY header 182 bytes (limit 9,000); 118,923 values and 475,692 bytes expected in all; 3,373,248 request bytes in all; assets 0, tasks 0, in flight 1 (limit 4)
Within budget.
```
