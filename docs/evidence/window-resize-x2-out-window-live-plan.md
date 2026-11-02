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

Mixed-window settings are **offline only**: the existing status is
unchanged, and every live configuration restricts `out_window` to None.
Historical records predate that parameter and now report it as unstated;
they are not edited or reinterpreted as validation of explicit output windows.

## Commands and budget

Generate the offline record on a clean implementation commit and commit
that evidence before submitting the full plan for approval:

```sh
PYTHONPATH=. python tools/live_fixtures.py --offline resize_x2_out_window --output docs/evidence/window-resize-x2-out-window-offline.json
PYTHONPATH=. python tools/live_fixtures.py --plan resize_x2_out_window
```

The plan needs **four interactive computePixels requests**, zero assets,
zero tasks and one request at a time. No EECU estimate is available.
The new selection requires separate live approval; the completed W7 run
does not authorize it. A failure or abort is recorded and never retried.
