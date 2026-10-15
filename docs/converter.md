# Converting attention models and planning stages

The converter turns a supported PyTorch attention model into Earth Engine graphs, without
composing the blocks by hand:

```
PyTorch model ──convert──▶ block graph ──plan──▶ stages ──build_stage──▶ Earth Engine graphs ──run──▶ assets
```

- `convert` traces the model with `torch.export` (fake tensors, no forward pass; tracing still runs
  the model's Python code, so convert only models whose source you trust), recognizes its
  modules and writes a **block graph**: library operations, their settings, their parameters by
  `state_dict` name, and the tensors between them. It also writes a **conversion report**: for
  every node, whether its settings are inside its block's validated record.
- `plan` cuts the block graph into **stages** that fit Earth Engine's limits, each ending in saved
  image boundaries, and explains every cut.
- `build_stage` turns a stage into Earth Engine graphs, its weights read from table assets.
- The runner, or a model's existing workflow, exports the stages in order.

This guide converts, offline, the two models the converter was built for: SegFormer-B0,
`Segformer("mit_b0")` of segmentation-models-pytorch (SMP) 0.5.0, and SMP's `UPerNet` with
timm's Swin-T encoder, both at 224 × 224 with two input bands and two classes. The converter's
design document and the task records of its development are in the private development archive
(`docs/phase3-design.md`, and `docs/tasks/X00-byte-identity-targets.md` to
`docs/tasks/X08-documentation-and-api.md`).

Two terms recur below. The **hand-built graphs** are the 19 SegFormer-B0 and Swin-T/UPerNet
graphs composed by hand from the blocks and run live before the converter existed; the
graph-identity gate G1-ext ([CONTRIBUTING](../CONTRIBUTING.md#which-gate-applies)) holds their
hashes. The **hand plan** is the six stages SegFormer-B0 was cut into by hand for its first
live run. C1 to C4 name the encoder's four feature levels.

## What a conversion claims

Four kinds of evidence apply, and they are kept apart:

- **Graph identity.** Converting SMP's `Segformer("mit_b0")` at 224 × 224 gives exactly the
  hand-written SegFormer-B0 block graph (SHA-256 `568c92c3…`), whose stages build the seven
  SegFormer hand-built graphs byte for byte; converting the Swin-T/UPerNet gives nodes that build
  all twelve Swin hand-built graphs byte for byte (their hashes are in
  [x00-g1ext-golden-graphs.json](evidence/x00-g1ext-golden-graphs.json); the task records,
  `docs/tasks/X03-segformer-converter.md` and `docs/tasks/X04-swin-upernet-converter.md`, are in
  the development archive). Those graphs ran live, so a converted graph inherits their evidence, **at the settings they ran**. A
  setting outside a block's validated record carries no evidence until it has its own: a
  conversion that succeeds says the library can express the model, not that it was validated.
- **Offline mirror agreement**: NumPy mirrors against PyTorch, per block, in the
  [live fixture harness](live-fixtures.md). The converter adds none.
- **Live agreement.** One run so far ([record of the run](evidence/x06-live.json); its task
  record, `docs/tasks/X06-runner.md`, is in the development archive): the converted SegFormer-B0, under the planner's default four
  stages, run by the library runner on one chip with random weights, agreed with PyTorch within the
  1e-4 absolute target at every saved boundary (largest absolute error 3.58e-6; logits 2.98e-6, 0
  argmax disagreements). C1–C3 and the logits are bit-identical to the hand-staged run, and the
  logits to the single-graph run, which exported nothing else. The four stages took 92.2 batch
  EECU-seconds.
- **Model accuracy**: not measured. The weights here are random.

## Before you start

Install the `swin` extra (SMP 0.5.0 and timm 1.0.30; [Getting started, Install](getting-started.md#1-install)).
The blocks below are one Python session: run them in order, from the checkout root with
`PYTHONPATH=.`, for example saved in order to one file. They make no cloud call and download
nothing (`encoder_weights=None`); nothing runs a model's forward pass.

Building an Earth Engine graph needs the API's function signatures. Offline, the checkout's
`tools.live_fixtures.offline_earth_engine()` installs them and refuses every cloud call, as the
tests do; it is in the checkout only, and from an installed library the Earth Engine client's
own test helper does the same ([Getting started](getting-started.md#2-run-the-offline-example)).
In a live session (`ee.Initialize(project=...)`) the same code builds the same graphs.

`convert`, `plan` and `build` are public at their module paths
(`tensor_ee.converter.convert.convert`, `tensor_ee.converter.plan.plan`,
`tensor_ee.converter.build.build`); the package `tensor_ee.converter` exports the rest, among them
`BlockGraph`, `build_stage`, `Limits`, `Plan` and `request_size`. A package attribute is either a
submodule or a function, so the three functions named like their modules stay there.

## 1. Convert

```python
import torch
import segmentation_models_pytorch as smp
from tensor_ee.core.grid import Grid
from tensor_ee.converter.convert import convert

torch.set_num_threads(2)
torch.manual_seed(20261014)              # the staged and X06 runs' random weights
segformer = smp.Segformer("mit_b0", encoder_weights=None, in_channels=2, classes=2).eval()
chip = Grid("EPSG:3857", (2.5, 0, 1000, 0, -2.5, 2000), 224, 224)   # the source chip

b0 = convert(segformer, (1, 2, 224, 224), chip, name="segformer_b0")
graph = b0.graph
assert graph.sha256 == "568c92c33a639257c43578d3c7773f91f98497d4c12191319ff2565a23b0f39a"
print(b0.report.text())
```

`convert(model, input_shape, source_grid)` takes one chip, `(1, channels, height, width)`, and a
model in evaluation mode. SegFormer's first blocks read windows, so its graph starts from the source
chip: the input `input` (bands `b0`, `b1` unless `bands=` names them) is relabeled onto the model's
token grid, and the logits are relabeled back onto the chip as bands `c0`, `c1`. The token grid is
abstract, the chip's height and width at scale 2.5 in `EPSG:3857` (`scale=`), so the graph between
the relabels does not depend on the source's pixel size or CRS. Node ids are module paths and
parameters `state_dict` names, so the graph can be compared with a hand-written one directly. It
holds no weight value: `graph.sha256` does not depend on the weights, and `graph.dumps(indent=1)`
is its versioned JSON.

```python
torch.manual_seed(0)
upernet = smp.UPerNet(encoder_name="tu-swin_tiny_patch4_window7_224", encoder_weights=None,
                      in_channels=2, classes=2).eval()
swin = convert(upernet, (1, 2, 224, 224), name="swin_t224", bands=("VV", "VH"))
assert swin.graph.sha256 == "d98671f13e7aefe1b7064794452c59b28cb132f90ae350bd5dd941c67d083e3e"
print(swin.report.lines()[0])    # swin_t224: 32 nodes; 32 with a validated record, 32 inside, ...
```

timm's Swin starts with a patch embedding that reads a raster, and the library relabels a raster
onto a token grid only as windows. So this graph starts from the model's own input on the abstract
token grid and ends there (`head_dense`: the logits as dense bands); `source_grid` is refused for
it. Relabeling a source image onto that grid and back is the recipe's, as the fixed adapter
`tensor_ee.swin.SwinT224Adapter` does it ([contract §11](library-contract.md#11-recipe-boundary)).
The graph has one node per timm block and SMP module: 32 nodes.

## 2. Read the report

The report's first line counts the nodes: those with a validated record, how many are inside it,
how many outside, and the layout operations (the relabels), which have no record. Each node's line
gives its operation, the modules it came from, its settings as its block's record names them, and a
verdict:

- **inside its validated record**: some configuration of the block's `validated` record
  ([catalogue](building-blocks.md#validated-settings)) holds every setting it names;
- **outside its validated record**: no configuration holds them. The line names the settings and,
  for each nearest configuration (fewest settings to change), what differs. It is reported, never
  refused: the node converts, and its numbers have no evidence;
- **a layout operation**: no record.

The report also lists settings not read from the model, with why (the UPerNet fusion's four
input-channel groups: the library computes the one 3 × 3 convolution as four groups summed in double
and rounded once, its one formulation, chosen in a research milestone for its numerics), the glue between
recognized modules (layout bookkeeping such as `contiguous` and `permute`), calls dropped because
nothing reads them, and, for information only, the installed versions beside those the forward-code
fingerprints were traced with.

```python
b1 = convert(smp.Segformer("mit_b1", encoder_weights=None, in_channels=2, classes=2).eval(),
             (1, 2, 224, 224), chip, name="segformer_b1")
print(b1.report.lines()[0])      # 18 with a validated record, 2 inside, 16 outside
print(b1.report.outside[:3])
print(next(node for node in b1.report.nodes if node.id == "encoder.block1.0").verdict())
```

`mit_b1` has SegFormer-B0's structure at other widths. It converts, and the report says that its
blocks are outside their records (`dim 64 is not 32`): the library can express it, and nothing
validates it.

**Refusals.** `convert` refuses (`UnsupportedOperation`, with a `code` and the module or call it
names) what the library cannot express, rather than approximating it:

- a model in training mode (`training_mode`): dropout and drop path are identities only in
  evaluation;
- a module no recognizer names (`unrecognized_module`), and glue outside the closed set of layout
  bookkeeping (`unsupported_glue`);
- an attribute the library fixes (`unsupported_setting`), such as a tanh GELU or a block norm at
  another epsilon, and a size it cannot compute: SegFormer at 512 × 512 reduces its keys and values
  to more than one 7 × 7 window, which needs a new attention block;
- **forward code** that differs from the validated trace (`forward_code`). Each recognizer kind
  carries a fingerprint of its unit's traced calls, recorded from SMP 0.5.0, timm 1.0.30 and torch
  2.14.0, with dataflow checks where argument order matters (concatenation order, the cyclic shift's
  direction) and every convolution's and adaptive pool's integers checked against the module running
  it. A unit whose code differs (a residual subtracted, a softmax over another axis) is refused,
  naming the unit and the first differing call. `convert(..., forward_mismatch="report")` converts it
  instead and marks that node outside its record: the opt-in for a harmless renaming after a torch
  upgrade, which you then judge.

```python
from tensor_ee.core.errors import UnsupportedOperation

for training, size in ((True, 224), (False, 512)):
    segformer.train(training)
    try:
        convert(segformer, (1, 2, size, size), Grid("EPSG:3857", (2.5, 0, 1000, 0, -2.5, 2000), size, size))
        raise AssertionError("converted")
    except UnsupportedOperation as refusal:
        print(refusal.code)      # training_mode, then unsupported_setting
segformer.eval()
```

## 3. Plan

`plan(graph, limits, *, parameters, namespace, names=None, cuts=(), measure=None)` cuts the graph
into stages, contiguous runs of an as-late-as-possible order of its nodes. Each stage reads saved
images and writes saved images: one batch export per boundary it writes, each on the whole chip.

**Limits** (`Limits`), each an input with a default:

| Limit | Default | Meaning |
|---|---:|---|
| `constants` | 24,000,000 | a stage's embedded constants, from its blocks' `METADATA` cost functions (the budget measured in two research milestones) |
| `request_bytes` | 10,000,000 | each export request of a stage, measured |
| `bands` | 1,024 | a saved boundary's bands |
| `extent` | 2 | `EXTENT` nodes (global reductions) in one stage; `None` for no bound |
| `transport` | `table_assets` | the only one: weights read from table assets, inputs from saved images |

The transport is how the requests are measured: every weight a `ServerArray` read from a table
asset, so NumPy weights are refused and the planner builds no weight value. `constants` counts the
values as if the weights were inline, the transport the 24 M budget was measured with, so with
table-asset weights it is conservative. `extent` bounds the work a stage recomputes: a global
reduction recomputes everything before it in its stage that is not saved, which is how the single
SegFormer-B0 graph cost about 64 times its staged run. The planner does nothing more about
recomputation; it is not modeled.

**Measuring.** By default the planner measures each candidate stage's requests with the library's
request sizer, `request_size`: `json.dumps` of the export request the runner submits, the bytes
Earth Engine receives. Each boundary is saved to `f"{namespace}_{name}"` and described as
`f"{graph.name}_{name}"`, as the runner names it, where `name` is `names[edge]` or the edge id. An
edge id is a module path, and an asset name cannot hold `.`, so such a name is sized under
`asset_name(name)` (`.` as `_`, the same length); to run a plan, name its boundaries with `names`.
It builds only the candidates it needs: metadata rules most out first.

The weights for planning are table-asset reads before any table exists. For SegFormer-B0 this
plans the tables offline with the checkout helpers the planner's tests and the live run used, so the requests are
those the live run sent (with its namespace: 11 bytes fewer per asset reference, the namespace
being 11 characters shorter). The library has no public function for this step yet.

```python
import numpy as np
from tensor_ee._swin_parameters import records_and_locations, with_manifest
from tensor_ee.converter import Limits, Plan
from tensor_ee.converter.plan import plan
from tools import segformer_b0_request_plan as sizing
from tools.fixtures.segformer_b0_fullmodel import state_sha256
from tools.live_fixtures import offline_earth_engine

namespace = "projects/example-project/assets/segformer_b0_request_plan"
names = {"input": "input", "encoder.block1.0": "S1B0", "encoder.norm1": "C1", "encoder.norm2": "C2",
         "encoder.norm3": "C3", "encoder.norm4": "C4", "logits": "logits"}
state = {name: tensor.numpy() for name, tensor in segformer.state_dict().items()}
weights = {name: value for name, value in state.items() if value.dtype == np.float32}
digest = state_sha256(state)

with offline_earth_engine():
    records = list(records_and_locations({}, {}, weights)[0].values())
    shards = sizing.plan_weight_shards(records, namespace + "_weights", sizing.WEIGHT_WIRE_LIMIT)
    manifest = sizing.planned_manifest(records, shards, namespace + "_weights", digest)
    tables = with_manifest({}, {}, weights, manifest, digest)[2]     # state_dict name -> ServerArray
    default = plan(graph, Limits(), parameters=tables, namespace=namespace, names=names)

for planned in default.stages:
    print(planned.stage.name, planned.stage.inputs, "->", planned.stage.outputs, dict(planned.cost),
          len(planned.extent), dict(planned.requests))
    print("   ", planned.ends)
assert default.sha256 == "6d8fe1fdb95bfb73a1062e9e8cfd35a393bd20f6cb50fd87b903663c6ee48234"
```

**Why these four stages.** The objective is, in order: the fewest stages, the fewest saved
boundaries, the fewest saved values, then the most even constants. Every SegFormer block is an
`EXTENT` node (its spatial reduction attends over the whole chip), and each encoder stage holds two,
so under the bound of 2 no stage can hold two encoder stages: each stage's `ends` says so ("merged
with stage2: 4 EXTENT nodes, over the bound of 2"). Within that, the cuts fall at the stage norms,
because the decoder reads C1, C2 and C3 anyway: a boundary saved there serves the next stage and the
decoder's, at no extra export. The last stage holds encoder stage 4 and the decoder. The requests
are 2,062,548, 1,166,058, 874,385 and 464,261 bytes, and the input's export 1,222,596.

```python
print(default.stages[0].boundaries["encoder.norm1"]["reasons"])
print(default.single_graph["within"], default.single_graph["notes"])
print(default.search["score"], default.search["builds"])
assert Plan.loads(default.dumps(indent=1)) == default
```

Each boundary carries its own reasons (who reads it; resolution changes at it). `single_graph`
reports the whole graph as one stage: 6,620,078 constants and a 4,247,156-byte request, within the
hard limits, but 8 `EXTENT` nodes, so the plan cuts it; its recomputation cost is not modeled (the
single-graph run took 77.7 minutes and 5,870 batch EECU-seconds, the six hand stages about 91).
`search` holds the schedule, the objective's terms and the candidates measured. A plan's JSON is
versioned and deterministic, and `Plan.loads` refuses a document its plan would not write.

**`cuts=`** makes the plan save edges you need, to compare them with a reference or to try other
cuts. Under the hand plan's 2 MB request cap the planner splits stage 1 after `encoder.block1.0`,
because `input` → C1 measures 2,062,548 bytes: the one cut of the hand plan's six that a limit required. Its
other cuts, one per encoder stage, were a convention, so the planner merges stage 4 with the decoder
until C4 is required:

```python
with offline_earth_engine():
    capped = plan(graph, Limits(request_bytes=2_000_000), parameters=tables, namespace=namespace, names=names)
    six = plan(graph, Limits(request_bytes=2_000_000), parameters=tables, namespace=namespace, names=names,
               cuts=("encoder.norm4",))
print([planned.stage.outputs for planned in capped.stages])     # five stages
print([planned.stage.outputs for planned in six.stages])        # the hand plan's six
```

**Swin-T/UPerNet** has about ten times SegFormer-B0's parameters (36.7 M), and planning its tables
offline would serialize all of them. This example binds placeholder weights instead, zero arrays
built on the server with each parameter's shape, as the graph-identity harness and the Swin converter's tests did: the cuts and
constants are exact, and the request bytes are the placeholders', not those of table reads.

```python
import math
import ee
from tensor_ee.converter.plan import asset_name
from tensor_ee.core.tensor import ServerArray

swin_state = upernet.state_dict()
swin_names = {edge: asset_name(edge) for edge in swin.graph.edges}
swin_limits = Limits(constants=24_000_000, request_bytes=9_500_000, extent=1)   # Swin's hand limits
c3, c4 = "encoder.model.layers_2.blocks.5", "encoder.model.layers_3.blocks.1"

with offline_earth_engine():
    zeros = {}
    for node in swin.graph.nodes:
        for name in node.parameters.values():
            shape = tuple(swin_state[name].shape)
            stored = shape if len(shape) < 2 else (shape[0], math.prod(shape[1:]))
            zeros[name] = ServerArray(ee.Array(ee.List.repeat(0, math.prod(shape))).reshape(list(stored)).float(),
                                      stored)
    swin_namespace = "projects/example-project/assets/swin_t224"
    two = plan(swin.graph, swin_limits, parameters=zeros, namespace=swin_namespace, names=swin_names)
    abc = plan(swin.graph, swin_limits, parameters=zeros, namespace=swin_namespace, names=swin_names,
               cuts=(c3, c4))
for planned in two.stages + abc.stages:
    print(planned.stage.name, len(planned.nodes), dict(planned.cost), planned.extent, planned.ends)
```

Under Swin's hand limits (a research milestone's 24 M constants, the Swin workflow's 9.5 MB request cap, one
`EXTENT` node: the PSP's pooling) the planner gives two stages, cut inside encoder stage 4: the
encoder through stage 3 with stage 4's merge and first block (22,972,578 constants), then the rest
(17,392,398), because A and B together hold 30,543,738 constants and B and C 26,293,734, both over
the budget. The default limits give the same two. With C3 and C4 required it gives the hand stages
A, B and C, compared by cut edges: the adapter packs A's three skips into one asset, a recipe
convention the converter does not reproduce. The shifted-window nodes and the head are chip-edge
dependent (the cyclic shift wraps the chip), so no stage holding them can be tiled
(`PlannedStage.tileable`); the planner never tiles.

## 4. Build

`build_stage(graph, stage, parameters, assets)` builds a stage's outputs, each input read from the
asset it was saved to: an R edge as its bands pinned to its grid, a W edge as dense bands `c0` ..
windowed on its token grid. A W output comes back as dense bands `c0` .. on its token grid, the image
a stage exports. `request_size` measures the export request; `export_request`
(`tensor_ee.converter.request`) builds it.

```python
from tensor_ee.converter import build_stage, request_size

stage2 = default.stages[1]
with offline_earth_engine():
    assets = {edge: f"{namespace}_{default.names[edge]}" for edge in stage2.stage.inputs}
    (c2,) = build_stage(graph, stage2.stage, tables, assets).values()
    size = request_size(c2.image, c2.grid, asset_id=f"{namespace}_C2", description="segformer_b0_C2")
print(len(c2.bands), c2.grid.height, c2.grid.width, size)       # 64 bands on 28 x 28 tokens; 1,166,058
assert size == stage2.requests["encoder.norm2"]
```

`tensor_ee.converter.build.build(graph, parameters, inputs, outputs=...)` builds from tensors you
hold instead of saved assets, and accepts NumPy weights (inline constants), which no plan measures.
The builder emits the expressions the hand-built composites emit: that is what the G1-ext gate
compares with the hand-built graphs.

## 5. Run

Running needs Earth Engine, a project, and, in this repository, a live plan the user approves
([CONTRIBUTING](../CONTRIBUTING.md#which-gate-applies)); nothing in this section is run here.

- **Weights.** Upload the records with `tensor_ee.export_weights` to table assets; its manifest is
  written after each task and resumes (`resume=True`). Bind the ready manifest to the state with
  `with_manifest`, as above, for the `ServerArray`s. The live run's plan, at its step 3, adds the state's SHA-256
  and the record listing's hash to the manifest first.
- **SegFormer-B0, the library runner** (`tensor_ee.converter.run`). `prepare(directory, plan, graph,
  parameters=..., manifest=..., inputs=..., namespace=..., caps=Caps(...))` is offline: it builds and
  saves every request with its bytes and SHA-256, checks that each reads only the manifest's tables and
  its stage's saved inputs, and refuses one over the plan's limits. `run(directory)` submits the
  exports in order, one at a time, each with a request ID fixed at preparation, the intent written
  before each submission, polled at a bounded interval, and each saved asset's bands and grid
  checked before the next stage reads it. `resume` continues after an interruption and never
  resubmits; `status` reads the state without a call; `recover` records your decision when a
  submission's outcome is unknown. `CallGuard` caps the client calls of the steps outside the runner
  (the upload, readbacks). This is the path [the live run's plan](evidence/x06-live-plan.md) ran, with
  its record in [x06-live.json](evidence/x06-live.json).
- **Swin-T/UPerNet, its workflow.** The fixed adapter and `tensor_ee.swin_workflow` run stages A, B
  and C, live-validated ([Swin guide](swin.md)). The runner can run a converted Swin plan's stages,
  but that has not run live.

The runner runs one chip. An area needs a prepared directory per chip, and the planner never tiles
a stage with an `EXTENT` or chip-edge dependent node, so such a stage runs per chip.

## 6. Extending the converter

Another architecture family needs recognizers that map its modules onto the library's blocks and
composites, and a block the library lacks is a new block, never converter arithmetic.
[CONTRIBUTING, "Add a recognizer"](../CONTRIBUTING.md#add-a-recognizer) gives the steps and when a
new block is needed instead.

## Limits

- Two families: SMP SegFormer (keys and values one 7 × 7 window after the spatial reduction, so
  `mit_b0` at 224 × 224) and SMP UPerNet with timm's Swin. Only the settings in each block's record
  are validated.
- One input and one output; batch execution only. The planner cuts by rules and metadata: interactive
  memory and recomputation cost are not modeled, and its `extent` default was read off one hand plan
  (calibrating costs is deferred).
- Fingerprints are as exact as the trace: another torch version that renames or decomposes an ATen
  call is refused until they are retraced, or reported outside under `forward_mismatch="report"`.
  Integer reshape and pad sizes inside a unit are not fingerprinted.
- No public function yet binds weights to tables for a plan before an upload (this guide uses the
  checkout helpers), and no relabel of a source raster onto a token grid exists for Swin's input.
- The runner runs one chip, and only SegFormer-B0 has run through it live.

## Checks

The converter gate and this guide's code, offline:

```sh
PYTHONPATH=. python tools/graph_identity.py --check-ext docs/evidence/x00-g1ext-golden-graphs.json
PYTHONPATH=. python -m pytest -q tests/test_x03_segformer_converter.py tests/test_x04_swin_upernet_converter.py \
    tests/test_x05_planner.py tests/test_x08_docs_and_api.py
```

The first rebuilds the 19 hand-built graphs and must report `G1-ext: 19 of 19 graphs equal`, in
about 40 seconds. The tests take two to three minutes and about 1 GB of memory with two
threads; with the `swin` extra installed, expect `143 passed`, and without SMP and timm the
tests that convert skip with a reason. `tests/test_x08_docs_and_api.py` runs this guide's
Python blocks in order in a fresh interpreter.
