# TensorEE library contract

Task L00 of the library plan (development archive: `docs/library-plan.md`). This page turns the plan's
architecture boundaries into rules that L01–L07 are dispatched and reviewed
against. It changes no runtime code. Base revision: `d976819` (`lib/phase1`).

Three artifacts go with it:
- the graph-identity harness, [`tools/graph_identity.py`](../tools/graph_identity.py);
- the golden hashes, [`evidence/l00-golden-graphs.json`](evidence/l00-golden-graphs.json),
  checked in CI by [`tests/test_graph_identity.py`](../tests/test_graph_identity.py);
- the module map, [`evidence/l00-module-map.json`](evidence/l00-module-map.json),
  which gives every one of the 239 inventory entries a destination, an owner and a gate.

Section 14 lists every place where this contract departs from, or has to
interpret, the plan, with the maintainer's decision on each.

## 1. The graph-identity gate

### What G1 means in practice

A Phase 1 change passes G1 when all of these hold, with zero cloud calls:

```
PYTHONPATH=. python tools/graph_identity.py \
    --resources <checkout> \
    --check docs/evidence/l00-golden-graphs.json        # exit 0: all 45 graphs equal
PYTHONPATH=. python -m pytest -q -ra                    # offline suite, incl. tests/test_graph_identity.py
```

- **Golden environment.** Make the `--check` run in the golden environment
  (section 1.4). It is strict: any difference fails, in any environment. The
  pytest applies the per-graph version policy instead.
- **`--resources`.** The harness only reads `checkpoints/` and
  `downloads/t05_integration/` from it. When it is given, every trained graph
  must be built and compared. A missing or wrong checkpoint exits 1, and so does
  a failed T05 continuity check.
- **Fixtures only.** Without `--resources`, `--check` compares the fixtures and
  prints an explicit "trained graphs NOT compared" line.
- **Source check.** The harness always imports the checkout that contains it. It
  asserts that `tensor_ee` and every research package were loaded from there, as
  the T05 script does. A worktree nested under `.claude/worktrees/` does not
  count as this checkout.

### 1.1 What the harness builds

- **Offline build.** Graphs are built against the offline signatures of
  `ee.apitestcase.ApiTestCase` (`setUp`, as in
  `docs/evidence/t05-trained-preservation.py`).
- **Cloud calls refused.** For the whole run, every Earth Engine cloud entry
  point is patched to raise: `ee.Initialize`, `ee.Authenticate`,
  `ee.data._execute_cloud_call`, `getAsset`, `computePixels`, `computeValue`,
  `startProcessing`, and the other `ee.data` calls in `_REFUSED_DATA_CALLS`.
  Any attempt aborts the run, so `cloud_calls: 0` is enforced, not just declared.
- **What a record holds.** Each record is the SHA-256 and byte count of
  `value.serialize().encode()`. Raster outputs also record band order and grid.
  Records built from a model also hold:
  - `inputs`: a fingerprint of the raw `state_dict` before any folding;
  - `coefficients`: the effective (BatchNorm-folded) convolution fingerprints;
  - node hashes and folded BatchNorm keys, where the graph has them.

| Group | Graphs | Entry point | Needs |
|---|---:|---|---|
| `trained` | 9 | **The 8 T05 graphs, built as T05 built them:** U-Net raw, prepared and prediction (`predict_segwater` + `MatchedWeightProvider`); Swin prepared, validity and stages A, B, C (`SwinT224Adapter` + manifest-bound parameters). Checkpoint and manifest SHA-256 values are asserted. Both manifests are read from this checkout, where they are tracked; the T05 script read byte-identical copies from the resource checkout. **Plus `trained/unet/prediction_448`,** added at L00: the trained recipe on 448 × 448, i.e. 4 chips through `plan_segwater_tiles` and the mosaic. | Checkpoints |
| `t04` | 6 | T04's CNN regression (NonWater, TinyUNet, conv+BN+softmax), each with and without unfused diagnostic nodes, built verbatim from `downloads/t05_integration/t04-graph-regression.py` | — |
| `inline_transport` | 2 | Inline coefficients: ordinary literals, and the `Number.parse` route for values beyond signed int64 (`tests/test_inline_coefficients.py`) | — |
| `public_examples` | 7 | `examples.building_blocks` (manual form asserted equal to the compiled form); the skip/pool/residual composition with `clip_footprints` False and True (manual form asserted equal); `examples.water_chip` prediction, its retained inputs and the export payload (`Export.image.toAsset` mocked) | — |
| `segwater_recipe` | 1 | `predict_segwater` on 448 × 448 (4 chips) with a fixed-weight 1 × 1 model: CI coverage of the SegWater tiler and mosaic without a checkpoint | — |
| `nonwater_example` | 2 | The installed `tensor_ee.examples.nonwater.build()` image and metadata | — |
| `resnet18_unet` | 2 | `ResNet18UNet(base_channels=8)` on 32 × 32: inline, and asset-backed through `MatchedWeightProvider` with `include_unfused_nodes=False` and `clip_footprints=True`, as `predict_segwater` calls it | — |
| `tiling` | 2 | `compile_tiled_model` on the `tests/test_tiling.py` plan, plus its `head` node | — |
| `outputs` | 4 | `segmentation_outputs` with explicit validity, and `binary_threshold` | — |
| `onnx_import` | 1 | `load_onnx_model` → `compile_model` on the importer's contract fixture | `onnx` (skipped cleanly without it) |
| `swin_adapter` | 4 | The three stages with placeholder `ServerArray`s, exactly as `tests/test_swin_adapter.py` builds them, plus stage A from local CHW arrays (`embedded_raster`) | — |
| `swin_workflow` | 5 | `swin_workflow.build_stage`: the named `segwater_v3_db` input with the water output (A, B, C), and the prepared-input, raw-logit mode (A, C) | — |

**The T05 "ordinary-inline variants" are the six `t04` variants.**
`docs/evidence/t05-preservation.json` (`ordinary_inline`) names those three cases
with diagnostics on and off. T05 produced `t05-inline-baseline.json` by
re-running T04's script, so this is not a second set.

The ResNet18, tiling, `segwater_recipe`, `outputs` and water-chip fixtures use
integer-arithmetic weights and BatchNorm statistics (`_fixed_weights`). torch's
CPU normal generator can take a platform-specific SIMD path, so a seeded `randn`
fixture need not be bit-identical across machines. The `t04` fixtures keep
T04's seeded initialization verbatim so that their continuity with T05 can be
checked.

### 1.2 Continuity (required by L00)

L00 ran at `9143b49`, whose runtime paths are identical to its base `d976819`. S01 regenerated
the golden file once the trained weight manifests had become the public ones in `tensor-ee`
(base `b204a2c`; S01 (development archive: `docs/tasks/S01-public-weight-assets.md`)). The asset IDs those manifests list
are part of five trained graphs. [S01's link record](evidence/s01-weights-link.json) shows that
each of the five is its earlier graph with the asset prefix substituted, byte for byte, and that
the other 40 graphs did not change.

| Check | Reference | Result |
|---|---|---|
| 8 trained graph hashes and byte counts | `downloads/t05_integration/trained-baseline.json` (also embedded as `T05_TRAINED`) | equal: 4 directly (`unet/raw`, `unet/prepared`, `swin/prepared`, `swin/validity`), and 4 through the link record (`unet/prediction`, `swin/A`, `B`, `C`), whose old hashes are T05's |
| L00's four-chip prediction (`unet/prediction_448`) | L00's golden hash (embedded as `L00_TRAINED`) | equal through the link record |
| U-Net coefficients (31), matched asset keys, checkpoint hashes | `trained-baseline.json` | equal |
| Weight manifest hashes | `trained-baseline.json`, through the link record | the tracked manifests are the record's public ones, and the record's tracked ones are T05's |
| 6 `t04` variants: image hash and bytes, bands, grid, folded BN, every node hash, effective weights | `downloads/t05_integration/t05-inline-baseline.json` (sha256 `a0591dc8…`) | equal |

L00's review round added `inputs`, two graphs and `coefficients` on 8 more records. Before and
after it, the 43 graph hashes, byte counts, bands, grids, node hashes, folded keys and
previously recorded coefficients are unchanged.

### 1.3 Golden file

`docs/evidence/l00-golden-graphs.json` has:
- `graphs`: 45 records, 9 `trained` and 36 `fixture`. 40 are L00's; the 5 that read the weight
  manifests' asset IDs have their S01 hashes;
- `continuity` and `trained_identities`, from the run above;
- `environment`: Python 3.14.6 (darwin-arm64), earthengine-api 1.7.45,
  torch 2.14.0, NumPy 2.5.3, onnx 1.23.0, timm 1.0.30, SMP 0.5.0,
  torchvision 0.29.0;
- `provenance`: S01's regeneration commit as the source revision, base `b204a2c`, no runtime path
  changed since base, the command, and `cloud_calls: 0`.

Comparisons use `graphs` only; provenance is informational.

**Regenerating the golden file is a lead decision.** It follows an approved G2
item, a separately investigated dependency upgrade, or a proven change of recorded inputs such
as S01's, and is never part of a refactor task. It also means editing harness and test provenance:
- the expectations in `tests/test_graph_identity.py`: the provenance base equals `BASE_REVISION`,
  no runtime path changed since base, and each trained graph's name and continuity route;
- `BASE_REVISION` and `L00_TRAINED` in the harness.

The golden file has been regenerated once since L00, at S01 (lead decision, 2026-10-15). To
regenerate, from the main checkout with everything committed, pass `--resources .`: the file
records the command line as typed.

```
PYTHONPATH=. python tools/graph_identity.py --resources . --output docs/evidence/l00-golden-graphs.json
```

### 1.4 Version-mismatch policy (decided here)

Hosted CI does not run in the golden environment: it uses Linux x86-64,
earthengine-api 1.7.47, torch 2.14.1+cpu, NumPy 2.4.6/2.5.4 and Python 3.11 and
3.12. `tests/test_graph_identity.py` applies this policy per graph
(`graph_identity.judge`):

1. **Equal record and equal raw inputs:** pass, in any environment.
2. **Raw inputs differ:** the pre-folding parameters and buffers differ, for
   example after seeded `randn` on another CPU.
   - With a version difference: named skip, because graph identity can only be
     judged with the golden inputs.
   - In the golden environment: fail, because changed fixture inputs are a code
     change.
3. **Effective coefficients differ with identical raw inputs:** always fail.
   This is a BatchNorm-folding or coefficient-extraction change, for example in
   converter code L03 edits.
4. **Graph fields differ with identical raw inputs and coefficients:** fail.
   - The message says whether earthengine-api, torch, NumPy or Python differ
     from the golden environment. If they do, it says the change cannot be
     attributed to a refactor and must be investigated separately (plan,
     validation rules).
   - If the maintainer decides to accept CI drift while that investigation runs,
     `TENSOR_EE_GRAPH_IDENTITY_ALLOW_ENV_DRIFT=1` turns *only this case* (with a
     version difference) into a named skip. It is not set in CI.

Reasons:
- Always comparing keeps CI's guard wherever serialization agrees.
- Fingerprinting raw inputs, not folded ones, means a folding regression cannot
  pass as RNG drift.
- Failing by default never hides a serialization change.
- The authoritative G1 decision remains the harness in the golden environment,
  which also covers the trained graphs.

**Observed on CI.** Hosted CI run 38193429268 on `e8400bc` passed all three
jobs (lead decision on Q3).
- Named skips (coefficient drift) in every job: only
  `t04/tiny_unet/{plain,diagnostics}` and
  `t04/conv_bn_pointwise_softmax/{plain,diagnostics}`.
- `onnx_import/conv_resize` is skipped in the two core jobs, which do not
  install onnx.
- Every other fixture graph serializes identically on Linux. That includes
  `examples/building_blocks`, `examples/composition/*` (seeded uniform
  initialization), `resnet18_unet/*` and `swin_adapter/*`.

The raw-input fingerprint was introduced after that run. The drift it measures
is the same: the `t04` seeded `randn` weights.

### 1.5 Why `tools/`

- Outside every shipped package: `[tool.setuptools.packages.find]` includes only
  `tensor_ee*` and the named research packages, and `tools/` has no `__init__.py`.
- Not a research folder, and not a test: it must run the trained graphs, which
  need git-ignored checkpoints that CI does not have.
- The test loads it by file path (`importlib`), so no top-level `tools` package
  name can clash.

### 1.6 What it does not establish

Graph identity is not numerical execution. G1 holds only because the 8 T05
trained graphs are byte-identical to the ones T06 validated live.
`trained/unet/prediction_448` was not run live; its golden hash is L00's. A G2
change needs its own NumPy-mirror or bounded live evidence. Graph identity, local
reference checks, live comparisons and model accuracy remain four separate
claims.

## 2. Package layout and dependency direction

### 2.1 Layers

| # | Layer | Package | Responsibility | May import |
|---|---|---|---|---|
| 1 | Core | `tensor_ee/core/` | `Grid` and grid views, tensor types (R, W, V), errors, the ops interface, precision policies, locality/cost schema | `ee`, NumPy |
| 2 | Layouts | `tensor_ee/layouts/` | Value-moving transforms: R pinning and zero extension, stride sampling, R↔W converters, halos and im2col, translation, cyclic shift, periodic extension, domain masks, embedded fixtures | core |
| 3 | Weights | `tensor_ee/weights/` | Coefficient transport: `ServerArray`, `encode_float32`, `constant_value`, inline float32 arrays, layout-derived row repetition; the asset format and `FeatureCollectionWeightProvider` | core, layouts |
| 4 | Blocks | `tensor_ee/blocks/raster/`, `tensor_ee/blocks/window/` | Compute blocks per layout: convolution, normalization, activations, linear, attention, resampling, pooling, merges, outputs, patch embed/merge | core, layouts, weights |
| 5 | Converter | `tensor_ee/compiler.py`, `semantics.py`, `tiling.py`, `receptive_field.py`, `onnx_import.py`, `compatibility.py`, `weight_assets.effective_convolutions`, `weight_mapping.py` (re-exported by `tensor_ee/converter/`) | FX tracing and `compile_model`, BatchNorm folding for export, tiling, receptive field, ONNX import | layers 1–4 |
| 6 | Models | `tensor_ee/models/` (Swin-T224/UPerNet composites, timm/SMP names), `tensor_ee/swin.py`, `_swin_parameters.py`, `swin_workflow.py`, `segwater_*.py` (SegWater recipe) | The two reference models and their recipes | layers 1–5 |

Outside the six layers, unchanged:
- `comparison.py`: local NumPy/torch comparison (`PixelTensor`,
  `compare_segmentation`); it imports `Grid` and `outputs._threshold_option`;
- `examples/` and `tensor_ee/examples/`;
- the fixture models `models.py` and `resnet_models.py`.

**Order deviation (D1).** The plan lists Blocks before Weights and Models before
Converter. The code requires the reverse in both places:
- every W block takes its constants from `constant_value`/`ServerArray`
  (linear.py:35 (archive `swin_block/earth_engine/linear.py:35` at `24a6291`),
  ops.py:67 (archive `swin_block/earth_engine/ops.py:67` at `24a6291`)), and the R
  convolution embeds coefficients with `_inline_float32_array`
  (compiler.py:185 (archive `native_ee/compiler.py:185` at `24a6291`));
- the SegWater U-Net recipe is built through `compile_model`
  (segwater_inference.py:30 (archive `native_ee/segwater_inference.py:30` at `24a6291`)).

`effective_convolutions` and `MatchedWeightProvider` trace FX graphs through
`semantics`, so they belong to the converter, not to Weights.

### 2.2 Research boundary

- No module in `core/`, `layouts/`, `weights/` or `blocks/` may import `ee_attention`,
  `swin_*` or `upernet`. This holds from the task that creates the module.
- All of `tensor_ee` is free of research imports by the end of L05.
  `tensor_ee/swin.py` and `_swin_parameters.py` still import research modules
  until then. L06 adds the enforcing test (`tests/test_library_boundaries.py`).
- When code moves out of a research module, that research module keeps its name
  by re-exporting from the library, and the moving task makes that edit. There is
  never a second implementation. Example:
  `tests/test_swin_adapter.py` asserts
  `tensor_ee.swin.ServerArray is swin_block.earth_engine.ops.ServerArray`.
- Research-only code stays where it is, unedited, and stops shipping in L06.

### 2.3 Old module paths keep working

Every current `tensor_ee` module path and name in section 10 stays importable
throughout Phase 1, either as the definition or as a re-export.

Some tests patch a module attribute and then call code that must look the name
up *in that module*. Those patches pin where a name is looked up. The table below
is from a grep of `tests/` for `patch(`, `patch.object` and
`monkeypatch.setattr` on `tensor_ee` and research-module attributes; patches of
`ee`, `torch`, NumPy and `experiments`/`demo` runner internals are left out.

**Lead decision on Q1.** A task that moves a patched symbol retargets that
test's patch to the new location, so the test keeps testing the same thing. This
is inside that task's ownership; see `file_ownership` in the module map.

| Patch target | Test | What must hold | Task that may move it |
|---|---|---|---|
| `tensor_ee.compiler.symbolic_trace` | `test_compatibility.py:274` | `compile_model` resolves `symbolic_trace` from `tensor_ee.compiler` | none (converter stays, D2) |
| `tensor_ee.compiler._inline_float32_array` | `test_inline_coefficients.py:34` | `conv2d`'s inline-coefficient path resolves it from the module the test patches | L02 moves the helper; the `compiler.py` binding still intercepts while `conv2d` is in `compiler.py`. L03 moves `conv2d` and retargets the patch |
| `compiler._finish`, `compiler._ee_scalar` (monkeypatched), then `compiler._leaky_relu`, `_silu`, `_gelu`, `_tanh`, `_eval_dropout` called | `test_pointwise_operators.py:285-288` | those activations resolve `_finish` and `_ee_scalar` from the module the test patches | L01 (`_finish` → `layouts/raster.py`) and L02 (`_ee_scalar` → `core/precision.py`) leave `compiler.py` bindings that still intercept while the activations live in `compiler.py`. L03 moves the activations to `blocks/raster/activations.py` and retargets both patches |
| `tensor_ee.segwater_inference.compile_model` | `test_segwater_inference.py:144,265` | `predict_segwater` resolves `compile_model` from its own module | none planned (recipe stays); L05 if it moves |
| `swin_workflow.load_swin_t224_parameters`, `swin_workflow.build_stage`, `swin_workflow.SwinT224Adapter` (`patch.object(W, …)`) | `test_swin_workflow.py:77-78,99,213` | `swin_workflow`'s CLI and `build_stage` resolve these from `tensor_ee.swin_workflow` | none planned; L05 if it moves |
| `segwater_pytorch.load_segwater_checkpoint` (monkeypatched) | `test_segwater_parity.py:228` | `experiments/segwater_parity.py` resolves it from `tensor_ee.segwater_pytorch` | none planned; L05 if it moves |
| `_swin_parameters.FeatureCollectionWeightProvider.arrays` (class attribute) | `test_swin_adapter.py:165,191,198` | `_swin_parameters.FeatureCollectionWeightProvider` is the same class object that `with_manifest` instantiates | L02, when it moves the asset format to `weights/assets.py`: one class, re-exported |
| `MatchedWeightProvider().provider.arrays` (instance attribute) | `test_weight_mapping.py:27` | `MatchedWeightProvider` keeps a `provider` attribute whose `arrays` it calls | L03 (`weight_mapping.py`) |

For these reasons the converter modules stay at their paths, and
`tensor_ee/converter/` is a re-export namespace in Phase 1 (D2).

`tensor_ee/models.py` (the PyTorch fixtures `SmallCNN`, `TinyUNet`, …) collides
with a `tensor_ee/models/` package. L05 converts it into
`tensor_ee/models/__init__.py`, which re-exports the fixtures from
`models/fixtures.py`, so `from tensor_ee.models import TinyUNet` keeps working
(D3).

## 3. Tensor types

| Layout | Carrier | Today | Contract |
|---|---|---|---|
| **R** raster | `Raster(image, bands, grid)` | `compiler.Raster` | Unchanged fields, constructor and equality. One band per channel on a `Grid`. The inventory's **D** (dense token grid) is R on an abstract grid. |
| **W** window | `WindowTensor(image, grid, channels, window=7)` | `feature_ops.Feature(image, grid, channels, layout)` | One `[window², C]` float32 array per pixel of `grid.coarse(window)`. Token `t = window·a + b` of window `(I, J)` is pixel `(window·I + a, window·J + b)` of `grid`. Feature's `dense_row` layout (one `[1, C]` array band per pixel) is W with `window=1`. The `same_grid` refusal (mixed grids, layouts or windows) is kept. |
| **V** value | `ServerArray(array, shape)` for parameters; `ee.Array` with a client-known shape for small maps (PSP pooled `[s², C]`) | `swin_block.ops.ServerArray` | Not spatial; no grid. |

- **Window size is a field of the W tensor.** Layout converters, constant
  builders and blocks read it from the tensor, not from a module constant. Its
  default is 7, so graphs are unchanged.
- **`Raster` keeps working.** `Raster`, `CompiledModel` and `CompiledTiledModel`
  keep their dataclass fields. R blocks keep taking and returning `Raster`.
  `Feature` remains a name for the W carrier, by alias or subclass, for research
  callers.

### 3.1 Grid consolidation

There are three grid types:
- `compiler.Grid(crs, transform, width, height)`: georeferenced; `projection()`
  takes no argument;
- `spatial_attention.TokenGrid(height, width, scale=10, crs="EPSG:3857")`:
  origin (0, 0); `projection(ee, dx, dy)`;
- `feature_interface.FeatureGrid(crs, x0, y0, scale, height, width)`:
  georeferenced; `projection(ee, dx, dy)`.

They describe the same thing, but they **do not serialize the same way**. The
`grid_projection`, `grid_literal_types` and `grid_domain_mask` probes all differ
([evidence/l00-consolidation-probes.json](evidence/l00-consolidation-probes.json)):
- int and float zeros in the transform list serialize differently;
- `-s·0` produces `-0.0`;
- `TokenGrid`/`FeatureGrid` domain masks floor the pixel coordinates, and
  `Grid.mask()` does not.

Contract for L01:
- One module, `tensor_ee/core/grid.py`, defines `Grid` and the two other grids
  as views or constructors over one protocol: `crs`, `height`, `width`,
  `scale`/`transform`, `projection`, domain mask, `coarse`, `fine`, `resized`.
- Explicit conversions between them, which the adapter's relabel uses.
- Each kind keeps its current constructor signature, its equality and its
  serialized projection/mask expressions **byte-identical**.
- Merging them into one class whose literals change is a G2 item: every Swin
  graph would change. It is deferred (D4).
- The `TokenGrid` defaults (scale 10, origin (0, 0), EPSG:3857) stay. The
  adapter's 2.5 m model grid relies on them; L01 adds an assertion that does
  not change the graph.

## 4. Converters between layouts

| Converter | Today | General already | Still window 7 (or fixed) |
|---|---|---|---|
| R pinning: `_project`, `_finish`, `_zero_extended`, `from_image` | `compiler.py` | yes | — |
| R stride sampling `_stride_sample` | `compiler.py:129` | stride `s` in code | only stride 2 evaluated |
| Phase stride sample (georeferenced) | `feature_transport.stride_sample` | factor and phase | — |
| Space-to-depth rows `stride_samples`, `stack_tokens`, `as_row`, `flatten_row` | `spatial_attention.py` | `window` argument | — |
| R → W `to_windows` | `layout.py:107`, `patch_embed.to_windows` | built on the general rows | `WINDOW`/`TOKENS` module constants |
| W → R `depth_to_space` (production copy), `to_dense` | `patch_merge.py:150`, `layout.py:101` | `window` argument | `to_dense` passes `WINDOW` |
| Halo `zero_outside`, `neighbours`, `halo`; `taps`, `im2col` | `layout.py` | — | window 7, ring 1, kernel 3 |
| `translate`, `materialize` | `shift.py:66` | yes | — |
| Cyclic shift `cyclic_neighbor`, `roll_windows`, `shift_windows`, `unshift_windows` | `shift.py` | — | window 7, shift 3 (defaults) |
| `periodic_extension`, `domain_mask` | `periodic_extension.py` | any halo by construction | only run at window 7 |
| Patch-merge gather `gather_windows`, `merge_windows` | `patch_merge.py` | — | window 7 |
| Adapter relabel `_to_model` / `_to_coordinate` | `swin.py:73` | — | 224 chip, 2.5 m (model-specific; stays in the model) |
| `embedded_raster` (fixtures) | `feature_interface.py:165` | yes | — |

- L01 makes the window, ring and shift of every converter in the right-hand
  column a parameter, defaulting to today's value.
- `to_windows` gains an optional domain mask. With it, the two production copies
  serialize identically (`to_windows` probe), so they become one function (G1).
- Window sizes other than 7 remain unvalidated until Phase 2 (section 8).

## 5. The ops interface

Today the interface is only half polymorphic:
- `ArrayOps`/`ImageOps` and `BlockArrayOps`/`BlockImageOps` give compute
  primitives two implementations: V (`ee.Array`) and W (array-valued `ee.Image`);
- every layout transform takes `ee` and works only on images.

Contract (L02, `tensor_ee/core/ops.py`):
- **One `Ops` protocol, two implementations:** `ValueOps` (V) and `WindowOps`
  (W). Together they offer the union of today's methods: `matmul`,
  `matmul_double`, `transpose`, `row_max`, `row_sum`, `row_total`,
  `repeat_cols`, `repeat_rows`, `exp`, `maximum`, `divide_by_sqrt`, `scale`,
  `slice_rows/cols`, `cat_rows/cols`, `constant`, `wrap`, `to_double` and
  `to_float`. The four current class names remain as aliases.
- **In-window layout moves become ops methods,** so a block is written once for
  V and W: `reshape` (`arrayReshape` / `reshape`), `slice` (`arraySlice` /
  `slice`), `cat` (`arrayCat` / `Array.cat`) and axis `transpose`
  (`arrayTranspose` / `transpose`). For W they must emit the expressions in use
  today; for example, `layout.reshape` is
  `image.arrayReshape(ee.Image(ee.Array(shape)), len(shape))`.
- **Spatial transforms stay R/W only.** Translation, halo, stride sampling and
  cyclic shift have no V meaning. They dispatch on the tensor type, not on `ee`.
- Every ops method keeps the exact expression and cast it emits today. A method
  that rounds (`matmul` → `.float()`) and one that does not (`matmul_double`)
  stay distinct names; see section 7.

## 6. Constant builders

EE arrays do not broadcast. Per-channel constants are therefore pre-repeated to
`[tokens, C]` when they are built: `LinearConstants`, `BlockConstants`,
`PatchMergeConstants`, `PatchEmbedConstants`, `Conv3x3Constants`,
`BatchNormConstants`, the FPN/PSP norm constants. All of them call
`constant_value(ee, values, repeat_rows=49)` (`TOKENS`).

Contract (L02 mechanism; L04/L05 call sites):
- **The layout is passed, not 49.** A builder receives the layout it serves
  (W with window `w` → `w²` rows; W with window 1 → 1 row; V → the value's row
  count). The row count is `layout.rows`, and the repeat expression
  `reshape([1, C]).repeat(0, rows)` is unchanged, so window 7 serializes exactly
  as today.
- Constants carry their row count. A block refuses constants built for another
  layout. This is a client-side check and adds nothing to the graph.
- `ServerArray` (asset-backed) and NumPy (embedded with `encode_float32`) stay
  the two sources, with today's expressions.
- When constants are built makes no difference to the graph: the `layer_norm2d`
  probe compares constants built ahead with constants built at call time and
  finds them identical.

## 7. Precision policies

There are three named policies. Every block takes `policy=` with its current
policy as the default. A policy is implemented once, by the ops and block code
that emit its casts. **Graph identity requires every model to keep its exact
cast placement:** no cast may be added, removed or moved by a Phase 1 task.

| Policy | Definition (cast placement) | Today |
|---|---|---|
| `float32_operands` | Weight and bias arrays are cast `.float()` before `matrixMultiply`, so EE accumulates float32 operands in double. Every converter node output is materialized with `.toFloat()` (`compile_model`, `execution_dtype="float32"`), and so is every public primitive's (`primitives._float32`). Scalars are exact binary64 (`_ee_scalar`). | compiler.py:191 (archive `native_ee/compiler.py:191` at `24a6291`), compiler.py:661 (archive `native_ee/compiler.py:661` at `24a6291`) |
| `round_each_op` | Each EE operation that EE returns in double is rounded immediately: `matmul(...).float()`, `exp().float()`, `row_sum().float()`. Scalars are pre-rounded to float32 (`scale`, `divide_by_sqrt`). | array_ops.py (archive `ee_attention/earth_engine/array_ops.py` at `research-archive-2026-10-12`) (Milestone 1) |
| `double_round_once` | One float32 rounding per PyTorch operation. Inside LayerNorm, Linear (with bias), GELU, explicit BatchNorm, 1×1/3×3 convolution, bilinear resize and adaptive pooling, the work is double (`matmul_double`, `row_total`, double constants). The result is rounded once at the operation boundary (`ops.to_float`). Variant `grouped_double`: the four input-group partials stay double and are summed in P1, P2, P3, PSP order before the single rounding. | ops.py:15 (archive `swin_block/earth_engine/ops.py:15` at `research-archive-2026-10-12`), fusion.py:182 (archive `upernet/earth_engine/fusion.py:182` at `24a6291`) |

Release-contract line 146 requires the CNN and Swin policies to stay distinct.
They do, and each is implemented once.

Each block's default policy:

| Blocks | Default |
|---|---|
| R: `conv2d`, `grouped_conv2d`, `pool2d`, `nearest_upsample`, `bilinear_upsample`, `concatenate`, `residual_add`, `batch_norm`, `relu`, `sigmoid`, `leaky_relu`, `silu`, `gelu`, `tanh`, `relu6`, `hardsigmoid`, `hardswish`, `eval_dropout`, `softmax`, `segmentation_outputs`, `binary_threshold` | `float32_operands` (softmax and outputs keep their `toDouble()` score widening) |
| W/V: `naive_attention` (scores, softmax, value product) | `round_each_op` |
| W/V: `linear`/`conv1x1`, `layer_norm`, `layer_norm2d`, `gelu`, `batch_norm`, `conv3x3_window`, `window_resize`, `window_matrix_resize`, `window_pool`, `patch_embed`, `patch_merge` | `double_round_once` |
| W: `group_product` (unrounded partial), fusion `grouped_double` | `double_round_once` / `grouped_double` |
| W: `window_attention` | composite: Linear in `double_round_once`, attention core in `round_each_op` |
| W: `add` | float32 + float32 rounded once (`.toFloat()`) |
| W: `relu`, `concat`, `concat_rows`, `window_depth_to_space`; every layout transform | no arithmetic, no cast; must not gain one |

GELU has one formula with two policies: the R block rounds at the node boundary,
the W block with `ops.to_float`. They are different blocks, not two
implementations of one policy.

## 8. Parameterized settings

Each setting becomes a parameter whose **default equals today's value**, so
graphs stay identical. **A parameter is not evidence that other values work.**
Until Phase 2 validates a value live against PyTorch (G3), the block's status
stays "Fixed" at today's setting, and the docstring says so.

| Setting | Today | Defined at (inventory section 5) | Becomes a parameter of | Owner |
|---|---|---|---|---|
| Window | 7 | `WINDOW` in 12 modules; literal 7 in `swin.py:153,175` | `WindowTensor.window` and every W converter, constant builder and block | L01, L02, L04 |
| Tokens per window | 49 | `TOKENS` in 7 modules; pre-repeated constants | derived: `window²` | L02, L04 |
| Shift | 3 | `configuration.py:3`, `shifted_swin_block.py:36`, `shift.py:164,170`, `shifted_mask.py:46` | cyclic shift, `shifted_window_mask` | L01, L04 |
| Ring / halo | 1 | implicit in `layout.HALO = WINDOW + 2`; periodic `halo=1` | `halo`, `periodic_extension` | L01 |
| W conv kernel | 3 | `taps`/`im2col` `k=3` | `conv3x3_window` (kernel argument) | L04 |
| Patch | 4 | `patch_embed.py:41` | `patch_embed` | L04 |
| Input channel names | `("VV", "VH")` | `patch_embed.py:42`; the names reach the graph | `patch_embed` | L04 (recipe-facing; section 11) |
| Embedding width | 96 | `patch_embed.py:45` | `patch_embed` (from weight shapes) | L04 |
| PSP bins | (1, 2, 3, 6) | `_coefficients.SIZES`, `topology.PSP_SIZES` | `pool_rows`, `window_pool`, PSP | L04, L05 |
| Mask value | −100.0 (timm; production) | `shifted_mask.py:29` (M1 uses −1e30 at `common.py:32`, `spatial_attention.py:39`; research) | `shifted_window_mask` | L04 |
| Encoder LayerNorm eps | 1e-5 | `layer_norm.py:20` | `layer_norm` | L04 |
| Feature LayerNorm eps | 1e-6 | `normalization.py:33` | `layer_norm2d` | L04 |
| BatchNorm eps | 1e-5 | `normalization.py:34` | `batch_norm` | L04 |
| Fusion groups | 4 | `_decoder_contract.GROUPS` | grouped convolution | L04, L05 |
| Decoder width | 256 | `_decoder_contract.py:2`, `psp.py:42`, `segmentation_head.py:56`, `topology.py:43` | model configuration | L05 |
| Stage widths, depths, heads | (96, 192, 384, 768), (2, 2, 6, 2), (3, 6, 12, 24) | `configuration.py`, `feature_interface.py:41`, `_decoder_contract.py:4` | model configuration | L05 |
| Classes, head factor | 2, 4 | `segmentation_head.py:55,57` | head | L05 |
| Chip, input spacing | 224, 2.5 m abstract | `swin.py` | adapter (fixed) | L05 |
| CNN stride, nearest factor | stride {1, 2}; nearest: any positive equal integer factor, only 2 validated live | `semantics.py`, `blocks/raster/resample.py::nearest_factor` | unchanged. Strides other than 1 and 2 are rejected. Nearest accepts any positive equal integer factor (as at the release), but its status stays "Fixed — factor 2" until Phase 2 validates or restricts other factors. Corrected by the maintainer at L03 acceptance | L03 |

## 9. Locality and cost metadata

**Schema** (L02, `tensor_ee/core/metadata.py`):

```python
class Locality(Enum):
    POINTWISE = "pointwise"      # the same pixel or token
    KERNEL = "kxk"               # bounded neighbourhood; radius in pixels
    WINDOW = "window"            # its own window
    WINDOW_RING = "window+ring"  # its window plus a ring from the 8 neighbours; ring width
    EXTENT = "extent"            # the whole chip/tile (global pooling, cyclic shift, PSP)

@dataclass(frozen=True)
class BlockMetadata:
    name: str
    layout_in: str                           # "R" | "W" | "V"
    layout_out: str
    locality: Locality
    policy: str                              # section 7
    embedded_constants: Callable[..., int]   # values embedded in the request for NumPy weights
    asset_arrays: Callable[..., int]         # server arrays read when weights are ServerArrays
    radius: int | None = None                # KERNEL radius or WINDOW_RING ring width
    chip_edge_dependent: bool = False        # e.g. shifted-window mask, periodic wrap
    status: str = "Fixed"                    # Ready | Fixed, with the validated settings
    evidence: tuple[str, ...] = ()           # milestone or evidence references
```

- **Where it lives.** Each block module declares a module-level `METADATA`
  mapping public block name → `BlockMetadata`, next to the code: L03 for R blocks
  (radius from the rules in `receptive_field.py`), L04 for W blocks. The
  embedded-constant functions start from the existing budget helpers
  (`*_budget` in `upernet/earth_engine/`). The inventory's locality column is
  the initial value.
- **Who reads it:**
  - a test that every public block has metadata and that its locality matches
    the inventory (L03/L04);
  - the README block table (L07);
  - in Phase 3, the converter's graph planner. It keeps the per-graph
    embedded-constant budget (about 24–26 M values) and refuses to tile EXTENT
    blocks as if they were local.
  - The CNN converter keeps using `analyze_receptive_field` for exact halos.
- **No metadata may change a graph.**

Deviation D5: the plan gives metadata to L04 only. The schema goes in L02 so
that L03 and L04 can annotate their blocks in parallel.

## 10. Public API compatibility (Phase 1)

These names keep working, with unchanged behaviour, through every Phase 1 task.

**`tensor_ee` top level (`__all__`, 31 names):**
- `PixelTensor`, `compare_segmentation`, `CompatibilityReport`,
  `check_compatibility`;
- `CompiledModel`, `Grid`, `Raster`, `UnsupportedOperation`, `WeightProvider`,
  `compile_model`;
- `from_image`, `conv2d`, `pool2d`, `nearest_upsample`, `concatenate`,
  `residual_add`, `relu`, `sigmoid`, `leaky_relu`, `silu`, `gelu`, `tanh`,
  `eval_dropout`, `softmax`;
- `SegmentationOutputs`, `segmentation_outputs`, `binary_threshold`;
- `FeatureCollectionWeightProvider`, `MatchedWeightProvider`,
  `effective_convolutions`, `export_weights`.

**`tensor_ee.swin` (`__all__`, 5 names):** `ServerArray`, `SwinTParameters`,
`load_swin_t224_parameters`, `load_swin_t224_model`, `SwinT224Adapter`. Also
`_SKIP_BANDS` and `_C4_BANDS`, which tests import. Importing `tensor_ee.swin`
and constructing an adapter must stay free of initialization and cloud calls
(`test_swin_adapter.py`).

**Module paths imported by the repository's tests, examples, demo, experiments
and evidence scripts.** This list comes from an AST scan of every tracked
`.py` file outside `tensor_ee/` plus `tensor_ee/examples`:

| Module | Names |
|---|---|
| `tensor_ee.compiler` | `Grid`, `Raster`, `CompiledModel`, `UnsupportedOperation`, `WeightProvider`, `compile_model`, `_inline_float32_array`, `_pointwise_call`, `_real_option`, `_dropout_probability`, `_FLOAT32_MAX`, `_eval_dropout`, `_gelu`, `_leaky_relu`, `_silu`, `_tanh` |
| `tensor_ee.primitives`, `tensor_ee.outputs` | the public functions; `outputs._threshold_option` |
| `tensor_ee.weight_assets` | `ConvolutionWeights`, `FeatureCollectionWeightProvider`, `convolution_weights`, `effective_convolutions`, `export_weights`, `feature_records`, `collection_from_records`, `float32_wire_values`, `upload_chunks`, `probe_asset_roundtrip`, `verify_asset_rows`, `_write_manifest` |
| `tensor_ee.weight_mapping` | `MatchedWeightProvider` |
| `tensor_ee.tiling` | `PixelWindow`, `plan_tiles`, `compile_tiled_model`, `seam_mask` |
| `tensor_ee.receptive_field` | `analyze_receptive_field`, `UnsupportedReceptiveField`, `ReceptiveFieldReport` |
| `tensor_ee.onnx_import` | `load_onnx_model`, `onnx_metadata`, `UnsupportedOnnx` |
| `tensor_ee.segwater_preprocessing` | `BANDS`, `MEANS`, `STDS`, `OFFSET_DB`, `STEP_DB`, `MIN_CODE`, `MAX_CODE`, `preprocess_db_ee`, `preprocess_db_numpy`, `preprocessing_metadata`, `quantize_db_ee`, `quantize_db_numpy`, `round_ties_to_even_ee`, `training_mean_padding_numpy` |
| `tensor_ee.segwater_inference` | `predict_segwater`, `plan_segwater_tiles`, `CHIP_SIZE`, `OUTPUT_BANDS` |
| `tensor_ee.segwater_pytorch` | `load_segwater_checkpoint`, `adapt_for_earth_engine`, `INPUT_SHAPE` |
| `tensor_ee.models`, `tensor_ee.resnet_models` | `SmallCNN`, `TinyUNet`, `PoolRoundTrip`, `initialize_fixture`, `parameter_count`; `ResNet18UNet`, `ResNet18Encoder`, `BasicBlock`, `initialize_resnet_fixture`, `fixture_metadata` |
| `tensor_ee._swin_parameters` | `parameter_shapes`, `checkpoint_shapes`, `read_state`, `parameters_from_state`, `records_and_locations`, `with_manifest`, `encoder_prefix`, `ENC_PREFIX`, `FeatureCollectionWeightProvider`, `build_model` |
| `tensor_ee.swin_workflow` | `read_resources`, `build_stage`, `prepare`, `submit`, `status`, `weight_records`, `bind_weight_manifest`, `main`, `_SKIP_BANDS`, `_C4_BANDS`, `_fingerprint`, `_predecessors`, `_save`, `_state` |
| `tensor_ee.examples.nonwater` | `build`, `evaluate`, `fixture`, `main`, `GRID`, `BANDS`, `_request_grid` |

The test patch targets in section 2.3 stay valid; a task that moves one of them
retargets the patch. The module command lines
`python -m tensor_ee.examples.nonwater` and `python -m tensor_ee.swin_workflow`
keep their arguments. Core imports must keep working without the Swin extras.

## 11. Recipe boundary

These stay named application recipes. They are model-layer code, never library
defaults:
- **SegWater v3 preprocessing** (`segwater_preprocessing.py`): band order VV
  then VH; `MEANS`, `STDS`, `OFFSET_DB`, `STEP_DB`, `MIN_CODE`/`MAX_CODE`; the
  float32 rounding after each step; ties-to-even quantization; the `preserve`
  and `training_mean` masked policies.
- **The SegWater U-Net prediction recipe** (`segwater_inference.py`):
  - 224 chips, `plan_segwater_tiles`, alignment 32;
  - masks intersected before normalization, zero fill after it;
  - `_valid_logits` and `FLOAT32_MAX`;
  - the stable two-class `_water_probability`;
  - class 1 = water, inclusive threshold 0.5, the output band names and coverage.
- **The SegWater checkpoint loader** (`segwater_pytorch.py`), and the Swin water
  output in `swin_workflow.build_stage`: softmax probability, the `≥ 0.5` tie
  rule and coverage.
- **The fixed Swin-T224 contract** (`tensor_ee/swin.py`): 224² north-up square
  grid; the abstract 2.5 m model grid and its relabel; two input and two output
  band names; stage band names; the A → B → C graph split.
- **The sensor band names in `patch_embed`.** `("VV", "VH")` reach the graph
  through `rename`. The block takes channel names as a parameter defaulting to
  them, and the adapter keeps passing them.
- **The Code Editor JavaScript** (`code_editor/`): a frozen SegWater-only demo,
  live-parity-checked as today.

The generic blocks stay generic. They do not absorb the recipe's
`_valid_logits` or `_water_probability`, whose expressions differ
(`score_validity` probe). `binary_threshold`'s public defaults (positive class
1, threshold 0.5) remain for API compatibility, documented as generic defaults
and not as the SegWater rule.

## 12. Consolidation candidates: G1 or G2

Evidence: `PYTHONPATH=. python tools/graph_identity.py --probes …` builds the
existing copies on the same inputs, offline, and compares their serialization
([evidence/l00-consolidation-probes.json](evidence/l00-consolidation-probes.json)).

| Candidate | Probe | Identical? | Decision | Gate |
|---|---|---|---|---|
| `depth_to_space`: `patch_merge` (production) vs `spatial_attention` (M1) | `depth_to_space` | no; the M1 copy's `mod` gives the wrong window index above and left of the raster | Keep the production copy as the library block; the M1 copy stays frozen in research | **G1** |
| `layout.to_dense` vs `depth_to_space` + `toFloat` | `depth_to_space_wrappers` | yes | One converter | G1 |
| LayerNorm2d: `fpn.layer_norm`, `psp.layer_norm`, `normalization.layer_norm2d` | `layer_norm2d`, `layer_norm2d_psp` | yes (constants built ahead or at call time) | `layer_norm2d` becomes the single entry point; the wrappers call it | **G1** |
| W ReLU: PSP inline `max(0)`, `normalization.relu`, `feature_ops.relu` | `window_relu` | yes | One W `relu` | G1 |
| R→W: `layout.to_windows` vs `patch_embed.to_windows` (+ mask) | `to_windows` | yes, with the optional mask | One `to_windows(mask=…)` | G1 |
| Grid projection: `Grid` / `TokenGrid` / `FeatureGrid` | `grid_projection`, `grid_literal_types` | no | One protocol, literals kept (section 3.1) | G1; one class would be G2, deferred |
| Domain mask: `Grid.mask`, `TokenGrid.domain`, `FeatureGrid.domain`, `domain_mask` | `grid_domain_mask` | no | Keep all four expressions | G1; a merge would be G2, deferred |
| Stride samplers: `compiler._stride_sample`, `feature_transport.stride_sample`, `spatial_attention.stride_samples` | `stride_sample_2` | no | Keep three named transforms | G1; a merge would be G2, deferred |
| Integer translate: `shift.translate` vs `feature_interface.translate` | `integer_translate` | no (the second pins first and uses FeatureGrid literals) | Keep both | G1; a merge would be G2, deferred |
| Score validity: `compiler._common_score_mask` vs SegWater `_valid_logits` | `score_validity` | no | Recipe keeps its own expression (section 11) | G1; a merge would be G2, deferred |
| Conv2d R vs W; BatchNorm folded vs explicit; GELU R vs W | — | different layouts and policies | Separate blocks per layout and policy | G1 |

**Expected G2 items in Phase 1: none.** Every consolidation that changes a
serialized expression is deferred, and every mapped action keeps the production
expression byte-identical. This follows from the Phase 1 exit criterion: every
harness graph must equal its L00 golden hash. A G2 item would therefore also
require a maintainer-approved regeneration of the golden file. If the maintainer wants any
deferred merge in Phase 1, it becomes an explicit G2 item, with the reason
recorded here and in the module map.

## 13. Module map summary

[`evidence/l00-module-map.json`](evidence/l00-module-map.json) has one record per
inventory entry, keyed by index 0–238 plus `id` and `line`. Each record gives:
- the destination, and the target `file::name`;
- the owner task;
- the gate and the action (move, consolidate, generalize, stays, freeze);
- the probe evidence for consolidation candidates.

The inventory has **237 unique ids for 239 entries**: `tensor_ee/swin.py:_to_model`
(indices 163 and 185) and `:_to_coordinate` (164 and 186) were each recorded by
two readers. Both copies are mapped to the same target.

| Destination | Entries |
|---|---:|
| library | 114 |
| model | 51 |
| research (frozen) | 74 |
| delete | 0 |
| **total** | **239** |

| Owner | Entries | Destination | Gate |
|---|---:|---|---|
| L01 core types and layout converters | 35 | library | G1 |
| L02 ops, precision, metadata schema, weights transport | 12 | library | G1 |
| L03 R blocks and converter layer | 26 | library | G1 |
| L04 W blocks with metadata | 41 | library | G1 |
| L05 reference models and recipes | 51 | model | G1 |
| L06 freeze research and the Code Editor demo; packaging | 74 | research (frozen) | G1 |
| **total** | **239** | | 239 G1, 0 G2 |

- Seventeen entries carry probe evidence (`consolidation`).
- Research (frozen) is 74, not the inventory's 75: the unused
  `normalization.layer_norm2d` becomes the single LayerNorm2d entry point.
- No inventory entry is deleted. Dead code outside the inventory goes to L06:
  - `compiler._interpolate` (no callers): delete;
  - `feature_ops.relu` and the unreachable `window_taps` path stay frozen in
    research;
  - the 1024-channel guard concatenation in `fusion.forward` is never part of
    the output graph. L05 may drop it only if the harness stays identical.

**File ownership** (`file_ownership` in the map) lists, for each task, the new
files it creates (each created by exactly one task) and the existing files it
edits:
- **New test files, one per task:**
  - L01: `tests/test_core_layouts.py`;
  - L02: `tests/test_numerics_foundation.py`;
  - L03: `tests/test_raster_blocks.py`;
  - L04: `tests/test_window_blocks.py`;
  - L05: `tests/test_reference_models.py`;
  - L06: `tests/test_library_boundaries.py`.
- **L03** creates `tensor_ee/blocks/raster/*`, `tensor_ee/converter/__init__.py`
  and its test file.
  - It edits only `tensor_ee` CNN and converter modules: `compiler`,
    `primitives`, `outputs`, `semantics`, `tiling`, `receptive_field`,
    `onnx_import`, `compatibility`, `weight_assets`, `weight_mapping`.
  - Under Q1 it also edits `tests/test_pointwise_operators.py` and
    `tests/test_inline_coefficients.py`, to retarget their patches.
- **L04** creates `tensor_ee/blocks/window/*` and its test file. It edits only
  research modules, to turn them into re-exports.
- **L05** may retarget `tests/test_swin_workflow.py`,
  `tests/test_segwater_inference.py` and `tests/test_segwater_parity.py`, only if
  it moves a patched symbol (Q1).
- **L06** creates `docs/evidence/l06-runtime-imports.json` and edits
  `tests/test_runtime_packaging.py` (Q2).
- **The two lanes share no file**, tests included, and
  `tests/test_graph_identity.py` checks this.
- Sequential tasks (L01 → L02 → L03/L04 → L05 → L06) may edit the same existing
  file in turn. Examples: `compiler.py` gets L01's `Grid`/`Raster` re-export,
  then L02's coefficient helpers, then L03's blocks; `patch_merge.py` and
  `patch_embed.py` are edited by L01, then L04.
- `blocks/__init__.py` is created empty by L02, so neither lane edits it.

## 14. Deviations, open questions and lead decisions

Lead decisions recorded 2026-10-12, after the independent L00 review.

| # | Item | Proposal | Lead decision |
|---|---|---|---|
| D1 | Layer order. The plan lists blocks before weights and models before converter; the code needs weights (coefficient transport) before blocks and the converter before models (section 2.1) | Adopt core → layouts → weights → blocks → converter → models | Accepted |
| D2 | Converter location. Tests patch `tensor_ee.compiler.symbolic_trace` and `tensor_ee.segwater_inference.compile_model` | The converter modules stay at their paths; `tensor_ee/converter/` is a re-export namespace in Phase 1 | Accepted |
| D3 | `tensor_ee/models.py` collides with a `tensor_ee/models/` package | L05 converts it into a package that re-exports the fixtures | Accepted |
| D4 | Grid "consolidation" | Phase 1 gives one module and protocol with each grid's serialized literals kept. A single class is G2 and deferred | Accepted |
| D5 | Metadata ownership | Schema in L02; R annotations in L03; W annotations in L04 | Accepted |
| D6 | The weights layer has no task on the board | L02 owns coefficient transport and the asset format. Effective-coefficient folding and matching stay with the converter (L03) | Accepted |
| D7 | Research files are edited (turned into re-exports) by the task that moves their code, not only by L06 | Needed so that there is never a second implementation, and so that the `ServerArray` identity test passes from L02 on | Accepted |
| D8 | No G2 items in Phase 1 (section 12) | Confirm, or name the merges to schedule as G2 | Confirmed: no G2 items in Phase 1 |
| Q1 | Tests patch symbols that later tasks move (section 2.3). After a move, such a test can pass while no longer testing anything | Retarget the patch, or keep the lookup where the test patches it | The task that moves a patched symbol retargets that test's patch to the new location so the test keeps testing the same thing. This is inside that task's ownership |
| Q2 | `test_runtime_packaging.py` pins `build_support.SWIN_MODULES` to `docs/evidence/t01-runtime-imports.json`. The Phase 1 exit criterion (research not shipped) changes the closure | Allow L06 to update packaging evidence and test | L06 adds a new evidence file (for example `docs/evidence/l06-runtime-imports.json`) and points `test_runtime_packaging.py` at it. `t01-runtime-imports.json` stays unchanged as history |
| Q3 | CI uses earthengine-api 1.7.47, torch 2.14.1+cpu and NumPy 2.4.6/2.5.4 on Linux; the golden file was made with 1.7.45, 2.14.0 and 2.5.3 on macOS arm64 | Run the test on hosted CI before integrating | Resolved by hosted CI run 38193429268 on `e8400bc`: all three jobs pass. The only named skips are the four seeded `t04` variants (coefficient drift) and, in the two core jobs, `onnx_import/conv_resize` (onnx not installed). Every other fixture graph serializes identically on Linux. No drift variable is set in CI |
| Q4 | The board's L04 scope says "with locality and cost metadata" and gate "G1 (any G2 item named in L00)" | With D5 and D8, L04 is plain G1 | L04 is plain G1 |
| R1 | Regenerating the golden file (reviewer note) | Lead decision only (section 1.3) | Recorded: regeneration is a lead decision and requires harness/test provenance edits; Phase 1 expects none after L00 |

## As built (L07)

Added at L07. The sections above are the accepted text and are not rewritten. This
section lists where the implementation now differs from them, as of the L07 base
(`710f429`).

| Accepted text | As built |
|---|---|
| §1.3 regenerates the golden file with `--probes docs/evidence/l00-consolidation-probes.json`; §12 cites `tools/graph_identity.py --probes` | `--probes` was retired at L06b with the research copies it compared. The record stays in [`evidence/l00-consolidation-probes.json`](evidence/l00-consolidation-probes.json), and the harness's `--help` says so |
| §13: `compiler._interpolate` (no callers): delete | Removed at L06 |
| §1 (source check), §2.2, §2.3, §4, §7 and §13 cite research modules (`ee_attention`, `swin_*`, `upernet`), `experiments/` and the old `demo/` as files in the checkout, frozen in place and re-exporting the library | Archived at L06b: the research folders, `experiments/`, `demo/` and `code_editor/` were removed from the working tree. They are at the tag `research-archive-2026-10-12`, and the milestone findings are in [`history/`](history/README.md). Links in this contract to removed files point there, or to `24a6291` where the cited line is |
| §1: the harness asserts that `tensor_ee` and every research package load from the checkout | It asserts this for `tensor_ee` and `examples`; the research names left the harness's package lists at L06b (lead decision Q2) |
| §1.1: the trained-weight manifests are read from where they are tracked | They moved, unchanged, to [`data/manifests/`](../data/manifests/) at L06b |
| §1.5: `packages.find` includes `tensor_ee*` and the named research packages | The wheel ships `tensor_ee*` only (L06) |
| §2.2: `tests/test_swin_adapter.py` asserts `tensor_ee.swin.ServerArray is swin_block.earth_engine.ops.ServerArray` | The research module is archived. `tests/test_numerics_foundation.py` asserts that the library's `ServerArray` names are one object |
| §2.3: `test_segwater_parity.py` patches `segwater_pytorch.load_segwater_checkpoint` | That test and `experiments/segwater_parity.py` were archived at L06b; the other patch targets were retargeted as Q1 decided |
| §9: the metadata feeds "the README block table (L07)" | It feeds the generated [block catalogue](building-blocks.md) (`tools/block_catalogue.py`); the README copies its count sentence (S05) |
| §11: the Code Editor JavaScript (`code_editor/`) stays a frozen, hand-written SegWater demo | Replaced at L06d by `demos/code_editor/`, which loads the SegWater graph the library builds from a generated template |

**§10 public names, compared with the code at the L07 base.**
- `tensor_ee.__all__` has the same 31 names, and `tensor_ee.swin.__all__` the same
  5; `_SKIP_BANDS` and `_C4_BANDS` remain.
- Every name in the §10 module table still imports from its module.
- `python -m tensor_ee.examples.nonwater` and `python -m tensor_ee.swin_workflow`
  keep their arguments: their `add_argument` calls are identical to those at
  `24a6291`.
- Differences: none removed. The names were listed because tests, examples, the demo,
  experiments and evidence scripts imported them; the demo and experiments are now
  archived. New public modules exist beyond §10: the layer packages
  `tensor_ee.core`, `tensor_ee.layouts`, `tensor_ee.weights`,
  `tensor_ee.blocks.raster`, `tensor_ee.blocks.window`, `tensor_ee.converter` and
  `tensor_ee.models.swin_t224` (§2.1). The [block catalogue](building-blocks.md) lists
  their public names.

## As built (Phase 2)

- **P02, raster lane** (2026-10-13): nearest upsampling is validated live at factors
  2, 3 and 4, and `pool2d` gained the configurations listed in its `METADATA` status:
  - aligned average or max, k = 2, 3, 4;
  - max 3 × 3 at stride 1 or 2 with padding 1;
  - average 3 × 3 at stride 1 with padding 1, under both `count_include_pad` rules.

  Evidence: [live run](evidence/live-fixtures/20261013T065257Z-p02.json). The §8 row
  "only 2 validated live" is superseded. Strides other than 1 and 2 stay rejected.
- **P01, window lane** (2026-10-13): the window lane is validated live at windows 5, 7,
  8 and 12, with shift ⌊w/2⌋. Eight blocks are promoted; `conv3x3_window` stays Fixed
  at kernel 3. The default shift of `shift_windows`, `unshift_windows` and
  `shifted_window_mask` now follows the window (`window // 2`), so window 7 still gives
  3. `timm_shift` gives timm's single-window rule. Evidence: live runs
  [p01a](evidence/live-fixtures/20261013T074301Z-p01a.json),
  [p01b](evidence/live-fixtures/20261013T074330Z-p01b.json) and
  [the window-12 bias rerun](evidence/live-fixtures/20261013T075542Z-window-relative_position_bias-w12.json).
  The §3 and §8 statements that other windows are unvalidated are superseded for these
  windows.
- **P04, decoder blocks** (2026-10-13): `window_pool` is validated live on token
  extents 5, 7, 8, 12 and 14 with bins (1, 2, 3, 6), and on extents 8 and 12 with bins
  (1, 2, 4, 8); its bin tables are checked against PyTorch's rule independently.
  `group_product` is validated at 1, 2, 4 and 8 groups, at windows 7 and 12.
  `conv3x3_window` is Ready on P01's windows, its 3 × 3 kernel being the block's
  identity. `grouped_double` in the Swin-T decoder now refuses a group count other
  than its constants'. Evidence:
  [live run](evidence/live-fixtures/20261013T115558Z-p04.json). The §8 rows "bins
  (1, 2, 3, 6)" and "fusion groups 4" are superseded for these settings.
- **P03, patch embedding** (2026-10-13): `patch_embed` and `stride_columns` are
  validated live at patches 2, 4 and 8, with 1 to 4 arbitrarily named bands, widths 48,
  96 and 128, and windows 5, 7, 8 and 12. Invalid patches, band lists and projection
  shapes are now refused. Evidence:
  [live run](evidence/live-fixtures/20261013T120034Z-p03.json). With P03, no catalogue
  block remains Fixed; §8's "patch 4", "input channel names" and "embedding width 96"
  rows are superseded for these settings.
- **Capstone C0 and C0b** (2026-10-13): `SwinUPerNetConfig` (in
  `tensor_ee/models/swin_t224/configuration.py`) carries every setting of the
  Swin-UPerNet composite and its weight records, with Swin-T224 as the default instance;
  20 silent defaults on the composite's path now read the configuration or are refused.
  `group_product` (4 groups), `window_resize` (factors 2, 4, 8 with `align_corners=False`,
  4 with `align_corners=True`) and `window_matrix_resize` (sources 1, 2, 3, 6) are
  validated live at window 8. Evidence:
  [live run](evidence/live-fixtures/20261013T154410Z-c0b.json).
- **Capstone C1** (2026-10-14): a Swin-UPerNet at `SwinUPerNetConfig(window=8, patch=4,
  3 bands, embed_dim=48, depths=(2, 2, 2, 2), heads=(2, 4, 8, 16), decoder=64, classes=3)`.
  It has random weights in 13 table assets (`export_weights`, now resumable) and runs as
  one batch export, matching PyTorch to 9.78e-6 in the largest logit difference. It is
  the first live use of decoder width 64, 3 classes, head width 24 and
  `periodic_extension` at window 8. Evidence: [c1-capstone.json](evidence/c1-capstone.json).


## As built (Phase 3)

Added at X02b (2026-10-14). The sections above are the accepted text and are not rewritten.

**§2.1, layers.** The table places the Swin-T224/UPerNet composites in the models layer. They
are blocks now: X01a promoted the SegFormer-B0 composites to `tensor_ee/blocks/window/segformer.py`,
and X01b the Swin and UPerNet composites to `swin.py` and `upernet.py`, with
`SwinUPerNetConfig` and `SWIN_T224` in `swin_upernet_config.py` (not in
`tensor_ee/models/swin_t224/configuration.py`, as "As built (Phase 2)" says). The model packages
keep the fixed configurations and re-export the composites; the converter layer reaches them.

**§9, validated settings** (design decision E5). `BlockMetadata` gains two fields, appended so
that positional construction is unchanged:

```python
    validated: Validated | None = None       # the status's settings, machine-readable (required)
    per_setting: Callable[..., Mapping] | None = None   # properties by setting
```

- **Form.** `Validated(configurations, sources=())` is the union of its configurations. A
  configuration maps setting names to a tuple of values (or `Between(low, high)`, a closed
  range) validated together as the status joins them: each value of each setting was validated
  live in the combinations the evidence ran, and the library's "Ready" rule extends it to the
  rest of the configuration. A record does not claim that every combination was run. Settings
  the status joins are separate configurations, so a record can say "these values together":
  `group_product` is `{groups 1, 2, 4, 8; window 7, 12} ∪ {groups 4; window 8}`,
  `shifted_window_mask` one configuration per window with its shift (`w // 2`, or `None`, which
  resolves to it). A setting no configuration names is not restricted. `ANY_SETTINGS`
  (`({},)`) validates every call the block accepts; `NOT_LIVE` (`()`) none, and
  `validate_metadata` requires it exactly for a "Fixed — offline only" status. Values are plain
  Python (numbers, bools, strings, `None`, tuples), so a report can print them, and a record is
  hashable.
- **Names.** A setting name is a parameter, by signature, of the block function, of its cost
  functions (`embedded_constants`, `asset_arrays`) or of the record's `sources`: its constants
  builder (`mit_block_constants`'s `dim`, `heads`, `ratio`, the block graph's settings), a
  constants or configuration class (`Conv3x3Constants`'s `k`, `SwinUPerNetConfig`'s fields) or
  `PoolConfig` for `pool2d`. Any block with a W layout may name `window`, the window of its
  window-layout tensors (`WindowLayout`'s parameter). Cost functions name what the planner
  computes from edges (`height`, `width`, `in_channels`), so a caller states those as it costs
  them.
- **What a record holds.** It follows the status: a setting the status names holds exactly its
  values, joined as the status joins them; a setting the status does not name holds its one
  value when the evidence never varied it (`layer_norm`'s eps 1e-5, `clip_footprints=False` for
  the R blocks only the T02–T05 sweeps cover); a setting the evidence varied is left out, which
  an unqualified "Ready" covers. Composites record the settings of the models they ran in,
  joined per stage (SegFormer-B0's widths with their heads and ratios; Swin-T224's and the
  capstone's). Labels (band and output names) are not settings, and a value the block refuses
  needs no record.
- **The query.** `check_settings(metadata, settings) -> SettingsCheck(inside, outside,
  unstated, nearest)`: `inside` when some configuration holds every setting it names at a
  stated value; otherwise, over the nearest configurations (fewest settings to change or
  state), the stated settings outside them and the settings they name that the call did not
  state. An unstated setting is reported, never assumed. Values compare as settings: lists as
  tuples, NumPy scalars as Python ones (a float32 as its exact double, so callers pass Python
  floats), and a bool never equals a number.
- **Consistency.** `tests/test_x02b_validated_settings.py` checks every record against its
  status: names by signature; setting by setting where a status phrase names the setting
  ("windows 5, 7, 8, 12", "factors", "patches", "heads", "stride", "bins", ...), the record's
  values equal the status's, and each clause's joint statement is configurations of the record;
  the status's other numbers pooled, both ways, each exception listed with its reason; the
  defaults inside the record; no setting of a recorded live fixture outside its block's and
  parts' records (inside them where the fixture states every setting); the hand-written
  SegFormer-B0 block graph inside every node's record. Within one clause's own numbers a joint
  over-claim can pass (the test's docstring gives an example).
- **Two statuses corrected toward the evidence** (lead decision, X02b fix round):
  `shifted_swin_block` names stages 1–3 only (stage 4 is one window in both models, so timm
  never shifts it), and `window_pool` names the square extents P04 ran: 5, 7, 8, 12 and 14
  tokens with bins (1, 2, 3, 6), 8 and 12 with bins (1, 2, 4, 8), where it said "extents 5–14".
  Every other status is unchanged.

**§9, properties that depend on a setting** (X02b R4). `radius` and `chip_edge_dependent` hold
over every validated setting: the largest radius, and chip-edge dependent when any validated
setting is. This was already the meaning of the radius of `conv2d` (kernel 7), `pool2d`
(k = 4) and `patch_embed` (patch 8), and stays exact where a property does not vary. Where it
varies, `per_setting` is a function of setting names (keyword, each with the block's default)
returning the exact `locality`, `radius` or `chip_edge_dependent`, and `metadata_at(entry,
settings)` returns the entry with them. Applied to:
- `window_resize`: chip-edge dependent with `align_corners=True` only (its weights depend on
  the map's size; X01a's review). Its static flag is now True, the bound over its validated
  settings; `metadata_at(entry, {"align_corners": False})` gives False.
- `patch_embed`, `stride_columns` and `swin_patch_embedding`: k × k, radius `patch // 2` (patch
  1 is pointwise), `patch_locality`; the static radii stay 4, 4 and 2.

`conv2d` (radius `k // 2`), `pool2d` and `decimate` (`factor // 2`) have the same form of
dependence and keep their bounds; no reader needs their exact values yet.

**Other metadata (X02b R5).** `fusion_forward`'s radius is 2 in source-grid tokens, the
library's convention (X01b's review). `swin_encoder`'s cost functions keep `norm=True` as their
default (no default changes, and a caller counts the norm as the difference); their docstrings
say why, and the record says `norm` False, as both models ran. The block catalogue lists every
module of the block packages, `swin_upernet_config` among the modules without `METADATA`, and
every block's record.

**§2.1, the converter layer** (X02a to X06 and X08, 2026-10-14 and 2026-10-15). The table's
converter row lists the CNN path. Phase 3 added the path for attention models, in
`tensor_ee/converter/`, importing layers 1 to 4, `torch` and `ee` (never
`segmentation_models_pytorch` or `timm`; [guide](converter.md)):

| Module | Task | Holds |
|---|---|---|
| `graph.py` | X02a; `Stage` JSON X05 | `BlockGraph`, `Edge`, `Node`, `Stage`: the weight-free block graph, validated on construction, as versioned deterministic JSON (`native_ee.block_graph`, version 1) hashed by its canonical text |
| `operations.py` | X02a; X04 | `OPERATIONS`, the registry (below) |
| `build.py` | X02a | `build`, `build_stage`, `saved_inputs`: a block graph or a stage to Earth Engine graphs, each node's constants built by the library's builder |
| `export.py` | X03; X04 | `trace` (`torch.export`, non-strict, evaluation mode and float32 only), `group` (units by the innermost recognized module), the closed glue set `GLUE`, `Unit`, `Through`, `Forward`, the call fingerprints and `STRUCTURE` |
| `recognizers.py` | X03; X04 | `RECOGNIZERS` (SMP SegFormer), `SWIN_RECOGNIZERS` (timm Swin, SMP UPerNet), `ALL_RECOGNIZERS`; the forward-code fingerprints `FINGERPRINTS` (6) and `SWIN_FINGERPRINTS` (11), with the versions they were traced with (`TRACED`) |
| `convert.py` | X03; X04 | `convert`, `Conversion`, `ConversionReport`, `NodeCheck`, `record_settings` |
| `plan.py` | X05; X08 | `plan`, `Limits`, `Plan`, `PlannedStage`, `schedule`, `boundary_refusal`, `asset_name`; plan JSON `native_ee.plan`, version 1 |
| `request.py` | X06; X08 | `export_request`, `wire`, `request_reads`, `deep_recursion`; the request sizer `request_size` (X08) |
| `run.py` | X06 | `prepare`, `run`, `resume`, `status`, `recover`, `Caps`, `CallGuard`; run files `native_ee.run`, version 1 |

The stage-boundary conventions (`grid_view`, `pin_boundary`, `saved_image`, `boundary_windows`,
`boundary_raster`, `relabel_to_windows`, `relabel_from_windows`) moved from the SegFormer sizing
tool and model to the layouts layer, `tensor_ee/layouts/boundaries.py` (X02a), so the model no
longer loads the converter.

**The registry** (`OPERATIONS`, 19 entries). Each entry holds one library function by reference
(never a name looked up at run time), its block's `METADATA` (none for the two relabels), its
settings and parameter roles, the output edge it infers (refusing what the function refuses), a
cost adapter over the block's `METADATA` cost functions, and the call and constants builder the
hand composites use: `overlap_embedding`, `mit_block`, `channel_norm`, `project_pyramid`,
`fuse_head`, `relabel_to_windows` and `relabel_from_windows` (X02a); `patch_embed`, `swin_block`,
`shifted_swin_block`, `patch_merge`, `psp_forward`, `fpn_layer_norm`, `fpn_lateral`, `fpn_merge`,
`fpn_conv_block`, `fusion_forward`, `head_logits` and `head_dense` (X04). An operation the library
lacks is a new block, not an entry. The library constants builders the registry needed were added
to the blocks, one implementation each: `project_pyramid_constants`, `fuse_head_constants` (X02a)
and `fpn_level_constants` (X04).

**Public names** (X08). `tensor_ee.converter.__all__` keeps its ten names and adds
`Conversion`, `ConversionReport`, `NodeCheck`, `BlockGraph`, `Edge`, `Node`, `Stage`,
`build_stage`, `Limits`, `Plan`, `PlannedStage` and `request_size`, each the same object as at its
module. `convert`, `build`, `plan` and the runner's `run` are public at their module paths and not
re-exported: a package attribute is either the submodule or the function, and binding the
function would change what `from tensor_ee.converter import build, plan` returns.

**Maintainer decisions during Phase 3.**
- **X00.** The G1-ext golden file holds 19 graphs (7 SegFormer-B0, 12 Swin composite-level),
  including the two extra Swin targets `patch_embed` and `head_dense`; an SMP version other than
  0.5.0 is a named skip; writing the file (`--write-ext`) is the maintainer's.
- **X01a.** `spatial_reduction` is `EXTENT`, `decimate` `KERNEL` radius 2, `shared_key_attention`
  and `fuse_head` chip-edge dependent; the library's MiT block is `mit_block`; the composites rest
  on the whole-model live evidence, with no fixtures of their own.
- **X01b.** SW-MSA is `WINDOW_RING` radius 4 and chip-edge dependent, the cyclic wrap belonging to
  the stage and the encoder (`EXTENT`); radii are in source-grid tokens; `asset_arrays` counts
  distinct arrays.
- **X02a.** The decoder's constants builders and the boundary conventions moved into the library
  (above); variadic operations (`concat`, `add`) need a schema change when they arrive.
- **X02b.** The two status corrections recorded above; at integration, the core layer's
  standard-library allow-list gained `inspect` and `numbers`, which `check_settings` uses.
- **X03.** A unit whose forward code differs from its validated trace is refused by default, and
  converted outside its record under `forward_mismatch="report"`; the token grid is abstract
  (scale 2.5 in `EPSG:3857` by default, `scale=`), so interior stage graphs do not depend on the
  source's pixel size.
- **X04.** Structural integers (convolution stride, padding, dilation and groups; adaptive pool
  sizes) are checked against the module running the call, not fingerprinted, because they vary
  with validated settings; integer reshape and pad sizes inside a unit are a declared limitation.
  The SW-MSA node's chip-edge flag stands in for its periodic wrap, and the planner's `EXTENT`
  count measures global reductions only. The fusion's four input-channel groups are the
  library's fixed split, chosen in Milestone 10 for its numerics, not a converter choice. The
  Swin and UPerNet recognizers are a table of their own (`SWIN_RECOGNIZERS`).
- **X05.** The planner gate is met by the five-stage plan under the hand plan's 2 MB cap: only
  Codex's stage-1 split came from a limit, the C1 to C4 cuts were one per encoder stage by
  convention, and `cuts=("encoder.norm4",)` gives all six. The `extent` default of 2 was read off
  that plan, so the cap-only plan (three stages) is the honest baseline; calibration is X07's,
  deferred. The measurement was injected, pending a library sizer (X08).
- **X06.** The user added the minimal runner to Phase 3 and approved its live run in advance; one
  process per run directory (a lock file), call caps outside the runner (`CallGuard`), operations
  pinned to the run's project. The live run passed ([record](evidence/x06-live.json)).
- **X08.** `request_size` is the planner's default measurement and the sizing tool's: the request
  the runner prepares, to `f"{namespace}_{name}"` described as `f"{graph.name}_{name}"`, a
  module-path name sized under its same-length asset name (`asset_name`). The exports above.
