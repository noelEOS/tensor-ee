# Getting started

Five steps, from a fresh copy to a converted model. Each says what you should see.

| Step | Needs |
|---|---|
| [1. Install](#1-install) | Python 3.11 or newer; git for a source checkout |
| [2. Run the offline example](#2-run-the-offline-example) | step 1; no Earth Engine account |
| [3. Run the U-Net in the Code Editor](#3-run-the-u-net-in-the-code-editor) | an Earth Engine account; no install |
| [4. Try the browser demo offline](#4-try-the-browser-demo-offline) | step 1, from source; no Earth Engine account |
| [5. Convert an attention model](#5-convert-an-attention-model) | step 1 with the `swin` extra |

Nothing here downloads a checkpoint, and only step 3 and the optional live commands contact
Earth Engine.

## 1. Install

Install the library from PyPI:

```sh
python -m pip install tensor-ee
```

pip ends with `Successfully installed`, followed by what it installed; the distribution
is `tensor-ee` 0.1.0. Allow several minutes: most of the download is PyTorch, about 600 MB
once installed on macOS. On Linux, pip's default PyTorch build also brings the CUDA
libraries, a few gigabytes; to skip them, install the CPU-only build before installing
TensorEE, as the continuous integration does:
`python -m pip install 'torch>=2.10' --index-url https://download.pytorch.org/whl/cpu`.

For the tests, tools, demos and checkout examples, install from source:

```sh
git clone https://github.com/noelEOS/tensor-ee.git
cd tensor-ee
python -m venv .venv
source .venv/bin/activate
python -m pip install '.[test]'
```

Or install the library alone, straight from the repository:

```sh
python -m pip install git+https://github.com/noelEOS/tensor-ee
```

Optional extras: `resnet` (torchvision), `onnx` (ONNX import), `segwater` (the
trained U-Net), `swin` (the trained Swin-T/UPerNet, and the attention models of
segmentation-models-pytorch, SMP, and timm that the converter reads) and `plot`.
From PyPI, name them as in `'tensor-ee[onnx]'`; from a clone, use `'.[test,onnx]'`.
Importing the library never authenticates, initializes Earth Engine or submits work.
The wheel contains only `tensor_ee`, including the offline example below; the tests,
tools, demos and top-level `examples/` run from the checkout.

## 2. Run the offline example

With no credentials and no cloud calls, build the Earth Engine graph of a small synthetic
CNN (four input bands, three classes, a 16 × 12 grid):

```sh
python -m tensor_ee.examples.nonwater --prepare
```

It prints a JSON report of about 330 lines within seconds. Expect, among its fields:

```text
  "status": "offline_graph_only",
  "bands": 42,
  "compute_queries": 2,
  "compatibility": {
    "schema_version": 1,
    "status": "supported",
```

`compatibility` lists every operation of the model as supported. The graph's `graph_sha256`
depends on your Earth Engine client's version, `source_file` is the absolute path of the
example on your machine (remove it before you share the report), and `fixture` is the
example's internal name. From a source checkout, run the example's offline tests:

```sh
python -m pytest -q tests/test_nonwater_example.py tests/test_public_examples.py
```

Expect `7 passed`. `tests/test_public_examples.py` checks that building a small network by
hand from the blocks and translating it with `compile_model` give the same graph. The whole
offline suite is for development: [CONTRIBUTING](../CONTRIBUTING.md#set-up-and-run-the-tests)
runs it, and says which tests skip without an optional package or a checkpoint.

**The API, offline.** This script translates a two-layer CNN into an Earth Engine graph with
no credentials. `ee.apitestcase.ApiTestCase`, a test helper shipped with the Earth Engine
Python client, installs the client's offline function signatures, as the example above does;
it is not a documented client API, so it may change between client versions.

```python
import ee
from ee.apitestcase import ApiTestCase
import torch
from torch import nn

from tensor_ee import Grid, check_compatibility, compile_model, segmentation_outputs

offline = ApiTestCase()
offline.setUp()            # the Earth Engine client's offline signatures: no credentials, no network

torch.manual_seed(0)
model = nn.Sequential(nn.Conv2d(2, 4, 3, padding=1), nn.ReLU(), nn.Conv2d(4, 3, 1)).eval()
grid = Grid("EPSG:32651", (10, 0, 286030, 0, -10, 2640860), 16, 16)   # 16 x 16 pixels at 10 m
image = ee.Image.constant([-12.0, -20.0]).rename(["VV", "VH"])        # any ee.Image with these bands

print(check_compatibility(model, (1, 2, 16, 16)).status)              # local; no Earth Engine
logits = compile_model(model, image, ("VV", "VH"), grid)
outputs = segmentation_outputs(logits)
print(logits.bands, outputs.classes.bands)
offline.tearDown()
```

Expect:

```text
supported
('c0', 'c1', 'c2') ('class_index',)
```

`outputs.probabilities.image` and `outputs.classes.image` are ordinary, deferred `ee.Image`
graphs. After `ee.Initialize(project="YOUR_EE_PROJECT")`, without the offline helper, the same
code builds the same graphs, ready to display or export. The [API guide](api.md) gives the
supported forms and numerical rules.

**With an Earth Engine-enabled Cloud project** (`earthengine authenticate`, or append
`--credentials adc` for application-default credentials), the example runs in Earth
Engine, and its result is compared with CPU PyTorch on the retrieved input:

```sh
python -m tensor_ee.examples.nonwater --project YOUR_EE_PROJECT --output runs/nonwater.json
python -m examples.building_blocks --project YOUR_EE_PROJECT --mode primitives
python -m examples.building_blocks --project YOUR_EE_PROJECT --mode compiler
```

The nonwater command makes two interactive requests, and no export or asset; it writes
its report and exits with status 0 when the comparison passes. Its recorded run measured a
largest logit error of 7.450580596923828e-09, a largest probability error of
2.9802322387695312e-08 and 0 class disagreements over 190 valid pixels
([record](evidence/t05-primary.json)). That shows execution, not accuracy.
The two `building_blocks` modes build the same small network by hand and with the
compiler, and print the same minimum and maximum of its output.

## 3. Run the U-Net in the Code Editor

No install and no checkpoint: the trained U-Net's weights are public Earth Engine table
assets, under CC BY 4.0 ([WEIGHTS-LICENSE.md](../WEIGHTS-LICENSE.md)), so you need only an Earth Engine account and a Cloud project
[registered for Earth Engine](https://developers.google.com/earth-engine/guides/access), where
you can create an image asset.

1. Open the [Earth Engine Code Editor](https://code.earthengine.google.com/) and select
   your Cloud project.
2. In the **Scripts** tab, create a repository named `segwater` (the name `inference.js`
   expects) and in it a script named `segwater`. Paste the whole of
   [`demos/code_editor/segwater.js`](../demos/code_editor/segwater.js) and save. The file is
   about 190 KB: on GitHub, use **Copy raw file** (or open **Raw**, select all and copy);
   from a clone, copy everything from an editor. A paste cut short shows up as a syntax error
   in `segwater`, or, when `inference` runs, as "The template data is damaged: re-copy
   segwater.js in full."
3. In the same repository, create a script named `inference` and paste
   [`demos/code_editor/inference.js`](../demos/code_editor/inference.js). In
   `require('users/YOUR_USERNAME/segwater:segwater')`, replace `YOUR_USERNAME` with your Code
   Editor user name: the **Scripts** tab lists your repository under **Owner** as
   `users/<name>/segwater`.
   Save. (If the path is wrong, **Run** stops with a Console error naming it, for example
   `Cannot find required repo: users/YOUR_USERNAME/segwater:segwater`.)
4. Click **Run**. The panel starts on the preset chip, Sun Moon Lake on 3 January 2024.
   Click **Find scenes**: the 2024-01-03 Sentinel-1 scene is selected and its radar
   preview appears.
5. Enter an asset root such as `projects/YOUR_PROJECT/assets` and click **Create
   inference task**. In **Tasks**, start the export and wait for **Completed**. It is one
   batch export of one 224 × 224 chip, billed to your project: the first recorded run took
   140 s and 64.9 batch EECU-seconds ([record](evidence/l06d-user-check.json)).
6. Click **Load completed asset**. The map shows the water probability, the water mask and
   the coverage as layers: Sun Moon Lake and its water mask, like the
   [README figure](../README.md).

[CODE_EDITOR.md](../CODE_EDITOR.md) explains each step, drawing your own rectangle, the
exported bands, and how the scripts were checked: run from an account with no role on
`tensor-ee`, the preset chip's export was identical, value for value, to an earlier verified
export ([record of that check](evidence/s01-user-check.json)).

## 4. Try the browser demo offline

The [browser demo](../DEMO.md) is a local web app: draw an area, choose a Sentinel-1 scene
and run either reference model in Earth Engine. `--offline` serves the same interface with
no Earth Engine request, from the repository root:

```sh
PYTHONPATH=. python -m demos.browser.server --offline --project offline-check
```

Without the trained checkpoint it first warns that `--checkpoint … is missing; runs will
fail while preparing their graphs`, then prints a line ending in `OFFLINE FIXTURE, no Earth
Engine requests` and the URL to open, `http://127.0.0.1:8765/#token=…`. The page loads its
map library and basemap from the internet. The header reads "Offline fixture"; a scene
search returns fixture scenes from January 2024, and a run fails while preparing, since
the checkpoint is not public. The page then says only "The request failed; see the server
console."; the console where the server runs shows the cause, a missing checkpoint file.
Stop the server with Ctrl+C.

A live run needs application-default credentials and the trained checkpoint, which is not
public; the U-Net's also needs the `segwater` extra ([DEMO.md](../DEMO.md#start)).

## 5. Convert an attention model

Install the `swin` extra (SMP 0.5.0 and timm 1.0.30): `python -m pip install 'tensor-ee[swin]'`.
Then, in Python:

```python
import torch
import segmentation_models_pytorch as smp
from tensor_ee.core.grid import Grid
from tensor_ee.converter.convert import convert

torch.manual_seed(20261014)
segformer = smp.Segformer("mit_b0", encoder_weights=None, in_channels=2, classes=2).eval()
chip = Grid("EPSG:3857", (2.5, 0, 1000, 0, -2.5, 2000), 224, 224)   # the source chip

b0 = convert(segformer, (1, 2, 224, 224), chip, name="segformer_b0")
print(b0.report.lines()[0])
```

Expect:

```text
segformer_b0: 20 nodes; 18 with a validated record, 18 inside, 0 outside; 2 layout operations
```

`convert` traces the model with `torch.export`, which runs its Python `forward` code locally
with fake tensors, without numerical inference on image data. Nothing is downloaded with
`encoder_weights=None`. The result is a block graph: library operations with their settings
and parameter names, and no weight values. Tracing is not a sandbox, so convert only models
whose source you trust. The report checks every node's
settings against its block's validated record: here all 18 nodes with a record are inside
it, and the two layout operations (the relabels onto the model's token grid and back) have
none. A setting outside a record is reported, not refused, and carries no evidence.

The [converter guide](converter.md) continues from here, offline: the full report, a
refusal, planning the graph into stages that fit Earth Engine's limits, building them, and
the runner.

## Where next

- [Models and their evidence](model.md): what was checked for each model, and how.
- [Block catalogue](building-blocks.md): every block's layout, policy and validated settings.
- [The U-Net walkthrough](water-chip.md) and the [Swin-T/UPerNet workflow](swin.md): the
  trained models from Python, with your own checkpoint and assets.
- [CONTRIBUTING](../CONTRIBUTING.md): adding a block or a recognizer, the gates, the tests.
