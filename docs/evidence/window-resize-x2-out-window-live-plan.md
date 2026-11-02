# Window resize x2 with a different output window

This extends `window_resize` with keyword-only `out_window=None`, appended
after the existing parameters. None keeps the input window and emits the
historical graph. The default has live W7 evidence in
[the completed resize_x2 run](live-fixtures/20261102T190649Z-resize_x2.json),
which is retained exactly and is never rerun.

## DPT layouts and arithmetic

The fusion chain is **W7(7) → W14(14) → W7(28) → W7(56) → W7(112) → W7(224)**.
The 14-token identity-reassemble branch uses the backbone's W14, and its
fusion add needs a matching W14 resized map. The 28-token factor-2
reassemble branch uses W7, as do the later maps. The two new fixtures cover
the first two arrows; the existing W7 evidence covers the remaining three.

`layouts.boundaries.relabel_from_windows`/`relabel_to_windows` do not fit
the bridge because the dense round trip multiplies the graph with a 49- or
196-sample gather instead of slicing the resize's existing coefficient table.

For source window w, output window o, factor f and output child index j,
the parent is floor(j*o/(f*w)). Requiring o to divide f*w keeps every
output window inside one expanded parent. Each axis's coefficient stack
has rows from the same global PyTorch float32 interpolation matrix, now
in o-row chunks of width w+2. The source halo stays w+2. The same two
double matrix products produce o×o rows, followed by the same single
float32 rounding. No cast is added, removed or moved. With None, all
row counts, coefficients, indices and graphs are unchanged.

The block retains WINDOW_RING locality, radius one in source tokens, and
`chip_edge_dependent=True` for aligned corners: sample positions depend
on the whole map size. Unsupported windows that cannot fit one parent
halo are refused. No new operation, dependency or model is added.

## Evidence and planned scope

The new selection `resize_x2_out_window` contains:

| Fixture | Source → output | Window → out_window | Requests |
|---|---|---|---:|
| `window/window_resize/ac-f2-e7-w7-out14` | 7 → 14 | 7 → 14 | 2 |
| `window/window_resize/ac-f2-e14-w14-out7` | 14 → 28 | 14 → 7 | 2 |

Both use deterministic two-channel dyadic inputs, echo every input and
read every output. References use
`F.interpolate(scale_factor=2, mode="bilinear", align_corners=True)`.
The shared standalone NumPy mirror keeps y-then-x double arithmetic
with one float32 rounding; it has no dependence on the library's
coefficients or the window setting. The tests evaluate both layouts and
multi-parent rectangular graphs against it bit for bit. They also
reconstruct every global coefficient row, check costs, compare default
graphs at historical settings, and refuse invalid output windows.

The [approved live run](live-fixtures/20261102T193505Z-resize_x2_out_window.json)
passed both complete-output fixtures at clean `f0c464f`, using four
computePixels calls and one algorithms-list call, with zero refused calls.
Every output was bit-identical to the NumPy mirror; the largest absolute
EE-vs-PyTorch difference was 5.960464477539063e-8 for each fixture.
The record was committed exactly as written before appending these two
window/output-window pairs to the existing status (verbatim prefix) and
`validated` record. Earlier configurations retain `out_window=None`.
Historical records are unchanged and report the new parameter as unstated.
This is numerical agreement on deterministic inputs, not model accuracy.

## Commands and budget

Generate the offline record on a clean implementation commit and commit
that evidence before submitting the full plan for approval:

```sh
PYTHONPATH=. python tools/live_fixtures.py --offline resize_x2_out_window --output docs/evidence/window-resize-x2-out-window-offline.json
PYTHONPATH=. python tools/live_fixtures.py --plan resize_x2_out_window
```

The plan needs **four interactive computePixels requests**, zero assets,
zero tasks and one request at a time. No EECU estimate is available.
This selection received separate approval and ran exactly once. The completed
W7 run was never rerun. No assets or tasks were created, and no retry or
harness change was needed.

The offline record was generated at clean `0daae63`. Both fixtures have maximum
absolute mirror-vs-PyTorch error 5.960464477539063e-8, with zero values over
1e-4 and zero cloud calls. This is offline agreement only.

## Full offline plan output

```text
window/window_resize/ac-f2-e7-w7-out14 (window_resize, double_round_once); inputs 54930c12aa87, weights e3b0c44298fc
  request 0: 14 x 14 grid (rows x columns), 3 bands (out_0[2]); 588 values, 2,352 bytes expected; request 61,412 bytes; graph a46201225899; embedded constants 254 (METADATA) + 98 input values; NPY header 182 bytes
  request 1: 7 x 7 grid (rows x columns), 3 bands (input_0_0[2]); 147 values, 588 bytes expected; request 5,083 bytes; graph 1b60952f2018; embedded constants 0 (METADATA) + 98 input values; NPY header 182 bytes
window/window_resize/ac-f2-e14-w14-out7 (window_resize, double_round_once); inputs 38d76abfc03b, weights e3b0c44298fc
  request 0: 28 x 28 grid (rows x columns), 3 bands (out_0[2]); 2,352 values, 9,408 bytes expected; request 177,018 bytes; graph 83c476e56beb; embedded constants 904 (METADATA) + 392 input values; NPY header 182 bytes
  request 1: 14 x 14 grid (rows x columns), 3 bands (input_0_0[2]); 588 values, 2,352 bytes expected; request 8,667 bytes; graph 85e314dda8b4; embedded constants 0 (METADATA) + 392 input values; NPY header 182 bytes
Totals: 4 requests (budget 40); largest request 2,352 values (budget 1,000,000) and 177,018 bytes (budget 2,000,000); largest NPY header 182 bytes (limit 9,000); 3,675 values and 14,700 bytes expected in all; 252,180 request bytes in all; assets 0, tasks 0, in flight 1 (limit 4)
Within budget.
```
