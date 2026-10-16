# Instructions for coding agents

`tensor_ee` is a library of Earth Engine building blocks, with two reference models built
only from it: the SegWater ResNet18 U-Net and a fixed 224 × 224 Swin-T/UPerNet. Work here
extends the library, composes new models from it, and converts supported attention models
to it. The rules below apply to every change.

## Read first

These are binding. Where a task prompt and one of them disagree, stop and ask.

- [CONTRIBUTING.md](CONTRIBUTING.md): layers and import rules, "Add a block", "Add a
  recognizer", "Which gate applies", "Numerical comparisons", "Historical identifiers".
- [docs/library-contract.md](docs/library-contract.md) §2–§9, plus its "As built" sections.
- [docs/building-blocks.md](docs/building-blocks.md): every block's layout, locality, policy
  and validated status.
- [docs/live-fixtures.md](docs/live-fixtures.md): the fixture harness and how to add a fixture.
- [docs/converter.md](docs/converter.md): the converter for attention models, the planner and the
  runner, with what a conversion claims.
- Style exemplars: [linear.py](tensor_ee/blocks/window/linear.py),
  [normalization.py](tensor_ee/blocks/window/normalization.py), and the model composite in
  [tensor_ee/models/swin_t224/](tensor_ee/models/swin_t224/).

## Environment

- **Python.** Set up as [CONTRIBUTING](CONTRIBUTING.md#set-up-and-run-the-tests) describes, and
  run from the repository root with `PYTHONPATH=.`.
- **Branch.** Work on a branch cut from the latest `main`; changes reach `main` through a pull
  request.
- **Checkpoints.** The trained G1 graphs and a few tests read the git-ignored `checkpoints/`
  and `downloads/` from the checkout given as `--resources`; without them they skip by name.
  Never commit them.

## Checks

Run these on your final revision, and report each with its result:

```sh
PYTHONPATH=. python tools/graph_identity.py --check docs/evidence/l00-golden-graphs.json \
    --resources <checkout>             # G1: must report 45 of 45 graphs equal
PYTHONPATH=. python tools/graph_identity.py --check-ext docs/evidence/x00-g1ext-golden-graphs.json
                                       # G1-ext: must report 19 of 19 graphs equal
PYTHONPATH=. python -m pytest -q       # the offline suite
PYTHONPATH=. python tools/block_catalogue.py --check
PYTHONPATH=. python tools/live_fixtures.py --offline <selection> --output <path>   # G3-offline
PYTHONPATH=. python tools/live_fixtures.py --plan <selection>                      # within budget
```

Without the trained checkpoints, G1 compares the 36 fixture graphs and skips the 9 trained
graphs by name; say so when you report it.

Never report a check you did not run. If one fails, fix it or report it as failed with its
output.

## Library rules

- **G1 is standing.** The 45 golden graphs stay byte-identical at today's defaults, and the 19
  G1-ext graphs (the hand-built SegFormer-B0 and Swin-T/UPerNet graphs the converter rebuilds)
  with them. Run both gates before and after your change. Anything other than 45/45 and 19/19
  means stop and report. Never regenerate a golden file; that is the maintainer's decision.
- **Reuse before adding.** Check the catalogue first. A new value of an existing setting is a
  parameter whose default is today's value, plus a fixture: never a copy of the block. Two
  implementations of one operation are a defect. An operation the library does not have is a
  new block; do not emulate it with a composition that multiplies the graph's size, such as one
  call per channel.
- **Existing blocks keep their numerics.** Never add, move or remove a cast in an existing block.
  Never change an existing default or status, and never remove, rename or reorder a parameter.
  A new setting may be added as a keyword-only parameter whose default emits today's graph
  exactly, which G1 shows. There are three precision policies (contract §7); never add a fourth.
- **Layers.** Each layer imports only the layers before it
  ([`tests/test_library_boundaries.py`](tests/test_library_boundaries.py)). R and W blocks never
  import each other. Nothing in `tensor_ee/` imports from `tests/` or from timm. PyTorch
  references live in `tests/` and `tools/fixtures/` only.
- **Historical identifiers stay** (CONTRIBUTING, "Historical identifiers"): the format and wire
  identifiers spelled `native_ee_*` and `native_ee.*`, and the module paths the evidence
  records cite, are never renamed.
- **A new block** follows CONTRIBUTING's "Add a block", steps 1–6:
  - the family module in its layout's package, with its settings as parameters;
  - a keyword-only `policy=` checked with `validate_policy`;
  - arithmetic only through `tensor_ee.core.ops` and the existing cast helpers;
  - constants built ahead in a `*Constants` class and pre-repeated to the layout's rows, because
    Earth Engine arrays do not broadcast;
  - `*_values` and `*_arrays` cost functions;
  - a module docstring with the formula, the policy, the rows and what is validated;
  - `METADATA` passed through `validate_metadata`, with its required `validated` record of the
    settings validated live (`NOT_LIVE` with an offline-only status), and a `per_setting`
    function where a radius or chip-edge dependence depends on a setting;
  - unsupported forms refused with `reject` or `UnsupportedOperation`, never approximated;
  - a NumPy mirror in `tests/reference/`, a fixture in `tools/fixtures/`, its entry in the lane
    test's `EXPECTED` or `EXPECTED_METADATA`, and the catalogue regenerated.
- **Status.** Never write "Ready"; the maintainer promotes a block after G3-live. A new block's
  status is `Fixed — offline only: <the settings your offline evidence covers>`, and its
  docstring says it has not been validated live.
- **Locality.** A block whose output depends on the whole chip (global attention, global
  pooling) is `EXTENT`. Set `chip_edge_dependent` only when tiling with halos cannot reproduce
  the result, as with the shifted-window mask or a periodic wrap. A kernel's zero padding at
  the border is not chip-edge dependence: `KERNEL` or `WINDOW_RING` with its radius covers it,
  as for `conv3x3_window`.
- **A new model** of a family the converter recognizes is converted, not composed:
  `tensor_ee.converter.convert.convert` gives its block graph and a report of which settings are
  outside the validated records ([guide](docs/converter.md)). Any other new model goes in
  `tensor_ee/models/<slug>/`, composed from blocks by hand like `swin_t224`. It imports only
  library layers, uses the reference model's `state_dict` names, takes NumPy arrays or
  `ServerArray`s as weights, and is not loaded by `import tensor_ee`. `compile_model` stays the
  CNN path (`torch.fx` onto the R blocks): attention models go through the converter, which is
  extended with recognizers, never by routing them through `compile_model`.
- **A new recognizer** follows CONTRIBUTING's "Add a recognizer":
  - match the module by the qualified class name `nn_module_stack` records; the converter layer
    never imports the model's package (SMP, timm);
  - map it onto a registry operation that calls one library block or composite (`OPERATIONS`);
    a missing entry for an existing block holds the function by reference, its `METADATA`, a
    cost adapter and the library's constants builder;
  - check every attribute that decides the arithmetic: refuse (`reject`, `unsupported_setting`)
    what the block fixes, and leave settings the block takes as parameters to the report;
  - record the unit's forward-code fingerprint, traced once from the validated versions, with
    those versions (`FINGERPRINTS`, `TRACED`), refused by default on a mismatch and reported
    outside under `forward_mismatch="report"`;
  - add dataflow checks where argument order matters (concatenation order, a roll's direction
    and amount, which state a merge reads);
  - meet the converter gate: the validated model converts to graphs byte-identical to its
    live-validated targets.
  A new block is needed instead when the module computes something no block computes, or a
  block refuses its setting or size (SegFormer's keys and values beyond one 7 × 7 window): add
  the block first ("A new block" above), never arithmetic in the converter.

## Numerical claims

- Keep four claims apart: graph identity, offline mirror agreement, live agreement and model
  accuracy. One does not establish another.
- The 1e-4 absolute target is unchanged. Report the largest absolute error, never only a
  relative one. Never relax a check or rewrite a recorded result.

## Never

- Call Earth Engine: no `ee.Initialize` or `ee.Authenticate`, no `live_fixtures.py --run`, no
  exports, assets or tasks. Write a live plan instead; the maintainer runs it.
- Download anything: build references with random weights (`pretrained=False`,
  `encoder_weights=None`).
- Add a dependency.
- Commit credentials, project IDs, personal paths or asset IDs.

## In the pull request

Report:
- the base revision;
- changed files, grouped by layer;
- every command above with its result: the G1 and G1-ext counts, the pytest summary, the
  catalogue check, and the largest absolute mirror-vs-PyTorch error for each fixture;
- the live plan's request count, if there is one;
- limitations and open questions.
