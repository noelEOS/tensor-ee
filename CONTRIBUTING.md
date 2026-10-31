# Contributing

`tensor_ee` is a library of Earth Engine building blocks, plus two reference models
built only from it (see the [README](README.md)). Small, reproducible improvements
are welcome. New settings and new operations need live evidence; the
[converter guide](docs/converter.md) shows the path for attention models.
Contributions to the code use the repository's MIT licence.

## Set up and run the tests

```sh
python -m pip install -e '.[test]'
python -m pytest -q
# with the optional architecture, importer and model tests:
python -m pip install -e '.[test,resnet,onnx,segwater,swin]'
python -m pytest -q
```

The suite is offline. Graphs are built against the offline signatures of
`ee.apitestcase.ApiTestCase`; no test authenticates, initializes Earth Engine,
exports or downloads weights. Tests that need an optional package, Node.js or a
local checkpoint skip with a reason.

## Pull requests

Changes reach `main` through a pull request, and CI ([below](#continuous-integration)) must pass
before one is merged. Keep a pull request to one change, and report in it what
[AGENTS.md](AGENTS.md#in-the-pull-request) lists: the gate that applies and the checks you ran.

## Layers and import rules

Each layer imports only the standard library, its allowed third-party packages and
the layers above it in this table
([contract §2.1](docs/library-contract.md#21-layers)):

| Layer | Package | May also import |
|---|---|---|
| Core | `tensor_ee/core/` | `ee`, NumPy |
| Layouts | `tensor_ee/layouts/` | core; `ee`, NumPy |
| Weights | `tensor_ee/weights/` | core, layouts; `ee`, NumPy |
| R blocks | `tensor_ee/blocks/raster/` | core, layouts, weights; `ee`, NumPy, torch |
| W blocks | `tensor_ee/blocks/window/` | core, layouts, weights; `ee`, NumPy (torch only in `resample.py` and `pooling.py`, to read PyTorch's own coefficients) |
| Converter | `tensor_ee/converter/`, `compiler.py`, `semantics.py`, `tiling.py`, `receptive_field.py`, `onnx_import.py`, `compatibility.py`, `weight_assets.py`, `weight_mapping.py` | the layers above; torch, onnx |
| Models | `tensor_ee/models/`, `swin.py`, `_swin_parameters.py`, `swin_workflow.py`, `segwater_*.py` | the layers above; segmentation-models-pytorch, google-auth |

The R and W block packages never import each other. `import tensor_ee` does not load
the converter namespace or the Swin-T model. Library imports and constructors never
initialize Earth Engine or submit work; only explicitly invoked commands do.

[`tests/test_library_boundaries.py`](tests/test_library_boundaries.py) enforces these
rules by reading every import in `tensor_ee`, including imports inside functions, and
refuses dynamic imports. A new top-level `tensor_ee` module must be added to a group
in that test. No library module may import research code, which is archived at the
tag `research-archive-2026-10-12` ([history](docs/history/README.md)).

## Add a block

1. **Place the module.** An R block goes in `tensor_ee/blocks/raster/`, a W or V
   block in `tensor_ee/blocks/window/`, in the module of its family (convolution,
   pooling, normalization and so on). A block takes its settings as parameters.
   Defaults must be values that are validated; a parameter alone is not evidence
   that another value works.
2. **Take a precision policy.** A block with arithmetic takes a keyword-only
   `policy=` that defaults to its policy and is checked with
   `tensor_ee.core.precision.validate_policy`. Use the existing ops and casts; each
   policy has one implementation ([contract §7](docs/library-contract.md#7-precision-policies)).
   Never add, move or remove a cast in an existing block.
3. **Declare `METADATA`.** At module level, map each public block name to a
   `BlockMetadata` and pass the mapping through `validate_metadata`
   ([`tensor_ee/core/metadata.py`](tensor_ee/core/metadata.py)):
   - `layout_in`, `layout_out`: `R`, `W` or `V`;
   - `locality`, with `radius` for `KERNEL` and `WINDOW_RING`;
   - `policy`: a policy name, `"none"` for no arithmetic, or `"composite"`;
   - `embedded_constants` and `asset_arrays`: cost functions;
   - `chip_edge_dependent`;
   - `status`: `Ready` or `Fixed — <the validated settings>`;
   - `evidence`: paths under `docs/history/` or `docs/evidence/`, which must exist
     ([`tests/test_history_citations.py`](tests/test_history_citations.py));
   - `validated`, required: a `Validated` record of the settings the status names
     ([contract, as built for the converter](docs/library-contract.md#as-built-phase-3)), or
     `NOT_LIVE` for "Fixed — offline only". Name settings by the parameters of the block, its
     cost functions or its constants builder (the record's `sources`), and add the block to
     [`tests/test_x02b_validated_settings.py`](tests/test_x02b_validated_settings.py)'s
     allowance tables only with a reason;
   - `per_setting`, where a radius or chip-edge dependence depends on a setting
     (`window_resize`, `patch_embed`).
4. **Add offline tests.** Build the block's graph against the offline signatures and
   check the layout exactly (unique integer IDs work well), the refusals of
   unsupported forms, and the arithmetic against a NumPy mirror where one exists
   (`tests/reference/`). The lane tests list every block and its expected metadata:
   `EXPECTED` in [`tests/test_raster_blocks.py`](tests/test_raster_blocks.py),
   `EXPECTED_METADATA` and `NOT_BLOCKS` in
   [`tests/test_window_blocks.py`](tests/test_window_blocks.py). Add the new block
   there.
5. **Regenerate the catalogue.** Run `python tools/block_catalogue.py`, which
   rewrites the tables of [`docs/building-blocks.md`](docs/building-blocks.md).
   [`tests/test_docs.py`](tests/test_docs.py) fails while the page is out of date.
6. **Name the gate**, below, and record its evidence. A new setting of a block is checked
   with the [live fixture harness](docs/live-fixtures.md): add a fixture at that setting,
   record its `--offline` comparison of mirror and PyTorch, and its `--run` evidence.

## Add a recognizer

The converter (`tensor_ee.converter.convert`, [guide](docs/converter.md)) turns a PyTorch
model into a block graph through recognizers: functions that read a module and return the
block-graph nodes it is. A new architecture family, or a new module in a known one, needs
recognizers, written in `tensor_ee/converter/recognizers.py` with their tests:

1. **Reuse the library.** A recognizer maps a module onto a registry operation
   (`OPERATIONS` in [`operations.py`](tensor_ee/converter/operations.py)), each one library
   block or composite held by reference. Where the library has the block but no entry calls
   it, add an entry: the function, its block's `METADATA`, a cost adapter over that
   `METADATA`'s cost functions, the output edge it infers (refusing what the function
   refuses) and the library's own constants builder, never a second one.
2. **Match by qualified name.** Key the recognizer by the class name `nn_module_stack`
   records (`segmentation_models_pytorch.encoders.mix_transformer.Block`): the converter layer
   imports `torch`, never the model's package, which
   [`tests/test_library_boundaries.py`](tests/test_library_boundaries.py) and a
   fresh-interpreter test enforce. A recognizer takes `(module, path, traced)` and returns
   `Unit`s (`tensor_ee/converter/export.py`), or is a `container` whose own calls must be
   glue. Node ids are module paths and parameters `state_dict` names.
3. **Check the attributes that decide the arithmetic.** Read every setting from the module
   (widths, heads, window and shift, epsilons, the GELU form, kernel, stride and padding,
   bins). Refuse with `reject` (code `unsupported_setting`) what the library's block fixes,
   naming the module path and the attribute. Report, never refuse, a setting the block takes
   as a parameter: the conversion report checks it against the block's validated record. A
   setting not read from the model goes in `Unit.choices`, with why.
4. **Record a forward-code fingerprint.** Trace the validated model once (`export.trace`,
   `export.group`) and commit each recognizer kind's call fingerprint (`FINGERPRINTS`,
   `SWIN_FINGERPRINTS`), with the versions it was traced with (`TRACED`). A unit whose calls
   differ is refused by default (`forward_code`) and reported outside its record under
   `forward_mismatch="report"`. Convolution and adaptive-pooling integers are checked against
   the module running the call (`export.STRUCTURE`); integer reshape and pad sizes inside a
   unit are not fingerprinted, so state any that decide the arithmetic as a limitation.
5. **Add dataflow checks where argument order matters** (`Forward.dataflow`): a
   concatenation's input order, a cyclic shift's direction and amount, the state an FPN merge
   reads. A fingerprint sees the calls, not which tensor feeds which argument.
6. **Glue stays closed.** Between units only layout bookkeeping is accepted (`export.GLUE`),
   checked as moves of named axes. Add an operation there only where a trace needs it.
7. **Test, and name the gate.** Converting the validated model must give its hand-built,
   live-validated graphs byte for byte (the converter gate; adding G1-ext targets is a maintainer
   decision). Test that every node of the validated model is inside its record, that a
   variant outside it is reported, that each fixed attribute and each forward-code patch is
   refused (and reported under the opt-in), and that a retrace equals the committed
   fingerprints. A family with no live-validated graph converts with no numerical claim until
   it has live evidence (G3).

**When a new block is needed instead.** A recognizer only maps; it never computes. A new
block is needed when a module computes something no block does, when a block refuses the
module's setting or size (SegFormer's attention shares exactly one 7 × 7 window of reduced
keys and values, so `mit_b0` at 512 × 512 needs a new attention block), or when expressing
it would compose blocks in a way that multiplies the graph's size. Add it under
[Add a block](#add-a-block), with its own fixtures and gate, and then the registry entry and
the recognizer.

## Which gate applies

| Gate | When | Evidence |
|---|---|---|
| **G1**, identical graphs | Moves, renames, wrappers, parameters with unchanged defaults | Every graph in the harness serializes to its golden SHA-256, with zero cloud calls; the offline suite passes |
| **G1-ext**, the converter's targets | Any change to the blocks, composites, builder or converter | The 19 hand-built SegFormer-B0 and Swin-T/UPerNet graphs serialize to [their golden hashes](docs/evidence/x00-g1ext-golden-graphs.json) (`tools/graph_identity.py --check-ext`) |
| **Converter gate** | A recognizer for a model with live-validated graphs | The converted graph, built, is byte-identical to those graphs' G1-ext targets, and so inherits their evidence at their settings |
| **G2**, graphs change, results do not | Consolidating copies whose expressions differ | A stated reason for each changed graph, and an exact NumPy-mirror check or a bounded live check; the other graphs stay identical. Regenerating the golden file is the maintainer's decision |
| **G3**, new behaviour | New settings or operations | Live validation against a PyTorch reference under the unchanged 1e-4 target, under an export plan the maintainer approves |

A live run needs a written plan that the maintainer approves before any submission: the
fixtures, project, destinations, number of tasks and polling limits.

The G1 harness builds every production graph with zero cloud calls and compares it
with [`docs/evidence/l00-golden-graphs.json`](docs/evidence/l00-golden-graphs.json).
The trained graphs need a checkout that holds the git-ignored `checkpoints/` and
`downloads/`:

```sh
PYTHONPATH=. python tools/graph_identity.py --check docs/evidence/l00-golden-graphs.json \
    --resources /path/to/checkout
```

It must report every graph equal. Without `--resources` it compares the fixtures only
and says that the trained graphs were not compared;
[`tests/test_graph_identity.py`](tests/test_graph_identity.py) runs that part in CI.
Run it from a git checkout: it records the revision with `git`, so it fails in a source
archive without `.git`.

The converter's gate, G1-ext, compares the 19 hand-built SegFormer-B0 and Swin-T/UPerNet graphs
that the converter must rebuild (its SegFormer-B0 graphs need SMP 0.5.0 and timm, and are skipped by
name without them):

```sh
PYTHONPATH=. python tools/graph_identity.py --check-ext docs/evidence/x00-g1ext-golden-graphs.json
```

## Test markers and switches

- **`reference_environment`** (5 tests): historical bitwise observations of the
  recorded macOS arm64 environment. Four check that PyTorch's float64 matrix product
  accumulates sequentially at the mirror's block shapes (`test_upernet_fpn.py`,
  `test_upernet_primitives.py`, `test_upernet_fusion.py`, `test_patch_embed.py`), and
  one records PyTorch's ReLU signed-zero pattern (`test_upernet_primitives.py`). They
  run automatically only on macOS arm64 with torch 2.14.0, NumPy 2.5.3, timm 1.0.30
  and SMP 0.5.0 ([`tests/conftest.py`](tests/conftest.py)); elsewhere they skip. To
  run them anyway, use
  `python -m pytest -q -m reference_environment --run-reference-observations`;
  mismatches then fail normally.
- **`SWIN_T224_CHECKPOINT`**: the path of the verified Swin-T checkpoint. Two tests in
  `tests/test_swin_adapter.py` load it; without it they skip.
- **`TENSOR_EE_RESOURCES`** and **`TENSOR_EE_EE_JS_NODE_PATH`**: a checkout holding
  `checkpoints/`, and a `node_modules` folder with `@google/earthengine`, for the Code
  Editor demo's regeneration and Node round trip (`tests/test_code_editor_demo.py`).
- **`TENSOR_EE_GRAPH_IDENTITY_ALLOW_ENV_DRIFT=1`**: a maintainer decision only. It turns a
  graph difference seen with different dependency versions into a named skip
  ([contract §1.4](docs/library-contract.md#14-version-mismatch-policy-decided-here)).
  CI does not set it.
- CI sets `EE_LIVE`, `EE_LIVE_BATCH` and `EE_LIVE_LARGE` to `0`. No test in the
  current suite reads them; the live tests were archived with the research code.

The portable checks stay mandatory when their dependencies are present: model-parity
tolerances, coefficient checks, layout, resize and padding rules, ReLU semantics and
explicit sequential cancellation. The historical NumPy mirrors are not claimed to
reproduce Earth Engine's exact accumulation order on every CPU; PyTorch documents
[platform differences in reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html)
and [floating-point evaluation order](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html).

## Continuous integration

[GitHub Actions](.github/workflows/tests.yml) runs on every pull request and every push to
`main`, with no Earth Engine credentials. A new push to a pull request cancels that pull
request's run still in progress:
- **core**, on Python 3.11 and 3.12: CPU-only PyTorch, `.[test]`, the offline suite,
  then a wheel built, installed and imported outside the checkout;
- **optional-models**, on Python 3.11: CPU-only PyTorch and torchvision,
  `.[test,resnet,onnx,segwater,swin]`, the offline suite.

A pull request is merged only when all three jobs pass.

The checkout is shallow: no test needs the history. Private checkpoints and cached
research inputs are not uploaded to CI.

## Numerical comparisons

- Keep three claims apart: graph identity, numerical agreement on recorded inputs,
  and model accuracy. One does not establish another. Graph construction alone is
  not numerical equivalence. Describe live evidence separately from offline checks.
- Keep the 1e-4 absolute logit target and every recorded exception. Never relax a
  check silently or rewrite a historical result; the known U-Net impulse failure
  stays documented.
- Retain the actual prepared input pixels and their IEEE float32 identity, the
  ordered bands, the affine grid and coordinate orientation, and the raw mask
  channels.
- Declare the expected source and output validity independently. A missing observed
  output must fail, not shrink the denominator.
- Record the absolute error, MAE, RMSE, relative diagnostics, exact decisions and
  near-margin locations. Relative diagnostics never replace the absolute tolerance.
- `PixelTensor` and `compare_segmentation` handle local array checks; preprocessing
  and source-availability checks belong to the recipe ([API guide](docs/api.md)).

## Bug reports and credentials

For a bug, include the smallest model, the ordered input bands, the full grid,
package versions and the error. For a numerical discrepancy, also give the reference
dtype, padding, mask handling, and how the exact input pixels were obtained. Report a
security vulnerability privately, as [SECURITY.md](SECURITY.md) describes, never in a
public issue.

Never include credentials. Do not commit credentials, private project IDs, personal
paths, private asset IDs or checkpoints; use placeholders such as `YOUR_EE_PROJECT`.

## Historical identifiers

The project was developed as `native_ee` and renamed to TensorEE (`tensor_ee`) before
publication. Three kinds of name keep the old spelling on purpose, and must never be renamed:

- **Format and wire identifiers**, which are data, not code: the manifests' `format`
  (`native_ee_featurecollection_weights_v2`), the Code Editor module's `format`
  (`native_ee_segwater_code_editor_template_v1`), the converter's document formats
  (`native_ee.block_graph`, `native_ee.plan`, `native_ee.run`), the export image properties
  (`native_ee_fixed224_float32`, `native_ee_swint_upernet_independent224`,
  `native_ee_predict_segwater_code_editor`) and the window-8 Swin-UPerNet validation model's asset names. Renaming one
  would change a recorded hash or a live-validated request.
- **Recorded module paths** in the evidence records and the history notes
  (`native_ee/...`, `native_ee.*`): archive labels, left as recorded. A test that joins such a
  record to today's code maps the path through `tests/recorded_paths.py`, the one place that
  mapping lives.
- **Revisions, the archive tag and the name `native_ee`** cited in documents and records
  are those of the private development archive; they do not exist in this repository.

## What ships

The wheel contains only the `tensor_ee` package (`pyproject.toml` includes
`tensor_ee*`). The demos (`demos/`), the examples (`examples/`), `tools/`, the tests
and the documentation are checkout tools, not part of the installed package.
