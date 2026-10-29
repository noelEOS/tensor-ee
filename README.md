# TensorEE

[![Offline tests](https://github.com/noelEOS/tensor-ee/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/noelEOS/tensor-ee/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/tensor-ee)](https://pypi.org/project/tensor-ee/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23524569.svg)](https://doi.org/10.5281/zenodo.23524569)

**Run supported PyTorch models inside Google Earth Engine, without deploying a
model-serving endpoint.** TensorEE expresses neural-network inference as Earth Engine
image and array operations. Larger models can run in stages, saving intermediate
features as Earth Engine assets.

![Earth Engine imagery passes to a deployed Vertex AI endpoint, or is evaluated directly with TensorEE's Earth Engine image and array operations](https://raw.githubusercontent.com/noelEOS/tensor-ee/main/docs/images/inference-location.svg)

Python builds the computation graph; Earth Engine evaluates the forward pass. Weights
are embedded constants or reads from EE table assets. Unsupported operations are refused,
not approximated. The [Vertex AI integration](https://developers.google.com/earth-engine/guides/ee-vertex-overview)
shown alongside it uses a separately deployed model endpoint.

## A real example

![Sentinel-1 radar imagery of Sun Moon Lake, with the water probability and mask computed by a trained U-Net inside Earth Engine](https://raw.githubusercontent.com/noelEOS/tensor-ee/main/docs/images/water_chip_224.png)

*A trained ResNet18 U-Net on one 224 × 224 Sentinel-1 chip: Sun Moon Lake, Taiwan,
3 January 2024. Prediction, not ground truth. [Validation](https://github.com/noelEOS/tensor-ee/blob/main/docs/validation.md) ·
[Walkthrough](https://github.com/noelEOS/tensor-ee/blob/main/docs/water-chip.md). Contains modified Copernicus Sentinel data (2024).*

## Get started

- **[Run the Code Editor demo](https://github.com/noelEOS/tensor-ee/blob/main/CODE_EDITOR.md).** No install or checkpoint. The trained
  U-Net reads public weights; you need an Earth Engine account and a
  [registered Cloud project](https://developers.google.com/earth-engine/guides/access)
  where you can create an image asset. The demo exports one chip, billed to your project.
- **[Try the offline example](https://github.com/noelEOS/tensor-ee/blob/main/README.md#install).** Build a small CNN's graph locally, with no
  credentials or cloud calls.
- **[Convert an attention model](https://github.com/noelEOS/tensor-ee/blob/main/docs/getting-started.md#5-convert-an-attention-model).**
  Start with random weights, inspect the conversion report, then plan the stages.

## What you can run

| Model or approach | Entry point | Availability |
|---|---|---|
| SegWater ResNet18 U-Net, 224 × 224 | [Code Editor demo](https://github.com/noelEOS/tensor-ee/blob/main/CODE_EDITOR.md); `compile_model` for CNNs | Trained EE weights are public; the Code Editor demo needs no checkpoint |
| Fixed Swin-T/UPerNet, 224 × 224 | [Three-stage workflow](https://github.com/noelEOS/tensor-ee/blob/main/docs/swin.md) | Trained EE weights are public; this workflow also needs a local checkpoint, not yet public |
| Prithvi Sen1Floods11, one 224 × 224 crop | [Hand-composed crop API](https://github.com/noelEOS/tensor-ee/blob/main/docs/api.md#prithvi-sen1floods11) | Random-weight checks; a reduced-width crop matches its mirror live. State names follow the source model's module hierarchy; they have not been checked against a released checkpoint. |
| SMP SegFormer-B0 or UPerNet with timm's Swin encoder | [Attention converter](https://github.com/noelEOS/tensor-ee/blob/main/docs/converter.md) | Conversion examples use random weights; the report identifies settings outside the validated records |

You can also compose a model from the [block catalogue](https://github.com/noelEOS/tensor-ee/blob/main/docs/building-blocks.md).
It has 87 blocks: 22 in `tensor_ee/blocks/raster` and 65 in `tensor_ee/blocks/window`.
The [browser demo](https://github.com/noelEOS/tensor-ee/blob/main/DEMO.md) runs either trained model with a local checkpoint;
its `--offline` mode previews the interface without one.

## Staged inference

Saving intermediate features lets a model run as several bounded EE graphs and avoids
recomputing earlier stages. In the fixed Swin workflow below, each step waits for its
predecessor's export to complete. The decoder reads both C4 and the earlier features.

![Swin inference in three execution steps: weights are read from EE table assets, intermediate features are saved as EE image assets, and both feature paths feed the decoder. Active steps are blue; completed steps and saved assets turn green with checkmarks.](https://raw.githubusercontent.com/noelEOS/tensor-ee/main/docs/images/swin-inference.gif)

*Illustrative sequence, not execution timing. [Static diagram](https://raw.githubusercontent.com/noelEOS/tensor-ee/main/docs/images/swin-staged-inference.png).*

Weights live in EE table assets; saved features and predictions live in EE image assets.
The checkpoint, manifest and task state are local files. This workflow needs no GCS bucket.
The [Swin guide](https://github.com/noelEOS/tensor-ee/blob/main/docs/swin.md) details the three steps, including the validity summary
saved with C1–C3. Other models can use different cuts: the [converter planner](https://github.com/noelEOS/tensor-ee/blob/main/docs/converter.md#3-plan)
chooses stages under explicit limits.

## Install

Python 3.11 or newer.

```sh
python -m pip install tensor-ee
```

Build a small CNN's graph offline, with no credentials or cloud calls:

```sh
python -m tensor_ee.examples.nonwater --prepare
```

Expect `"status": "offline_graph_only"` in the report.
[Getting started](https://github.com/noelEOS/tensor-ee/blob/main/docs/getting-started.md)
covers optional extras, PyTorch's download size and CPU-only installation, and the
steps for running in Earth Engine.

For the tests, tools, demos and checkout examples, install from source:

```sh
git clone https://github.com/noelEOS/tensor-ee.git
cd tensor-ee
python -m venv .venv && source .venv/bin/activate
python -m pip install '.[test]'
python -m pytest -q tests/test_nonwater_example.py
```

Expect `2 passed` from the example's tests.

## Evidence and limits

- **Experimental, with restricted support.** Only the settings in the
  [block catalogue](https://github.com/noelEOS/tensor-ee/blob/main/docs/building-blocks.md) have validation evidence. A conversion can
  report settings outside those records; converting them does not validate them.
- **Numerical agreement is not model accuracy.** The [model evidence](https://github.com/noelEOS/tensor-ee/blob/main/docs/model.md)
  records live comparisons with PyTorch and ONNX Runtime, offline graph identity,
  and the inputs each result covers. The absolute logit target is 1e-4; segmentation
  accuracy is not measured.
- **Known exception.** The U-Net's synthetic upper-left impulse chip has a largest
  absolute logit difference of 1.6510487e-4, above the target; probabilities and the
  water mask agree ([record](https://github.com/noelEOS/tensor-ee/blob/main/docs/history/cnn/SEGWATER_PYTORCH_RESULTS.md)).
- **EE limits still apply.** Inference uses EE computation and storage, with their costs
  and quotas. The figures make no speed or cost comparison with Vertex AI.
- **Weights and training data.** Both trained models' EE weights are public under
  [CC BY 4.0](https://github.com/noelEOS/tensor-ee/blob/main/WEIGHTS-LICENSE.md); their [manifests](https://github.com/noelEOS/tensor-ee/tree/main/data/manifests/) list the assets.
  Checkpoint files are not yet public. Training used Sentinel-1 chips with water labels
  derived from Sentinel-2; a fuller dataset description is not yet published
  ([provenance](https://github.com/noelEOS/tensor-ee/blob/main/docs/model.md#training-data-and-provenance)).

## Documentation

[Getting started](https://github.com/noelEOS/tensor-ee/blob/main/docs/getting-started.md) · [API guide](https://github.com/noelEOS/tensor-ee/blob/main/docs/api.md) ·
[Converter guide](https://github.com/noelEOS/tensor-ee/blob/main/docs/converter.md) · [Block catalogue](https://github.com/noelEOS/tensor-ee/blob/main/docs/building-blocks.md) ·
[All guides and evidence](https://github.com/noelEOS/tensor-ee/blob/main/docs/README.md) ·
[Changelog](https://github.com/noelEOS/tensor-ee/blob/main/CHANGELOG.md)

## Licence, citation and attribution

Code and documentation: [MIT](https://github.com/noelEOS/tensor-ee/blob/main/LICENSE). Trained weights: [CC BY 4.0](https://github.com/noelEOS/tensor-ee/blob/main/WEIGHTS-LICENSE.md),
including the checkpoints when published. Satellite data and third-party libraries keep
their own terms. The Sentinel-1 examples contain modified Copernicus Sentinel data (2024).
To cite TensorEE, use [CITATION.cff](https://github.com/noelEOS/tensor-ee/blob/main/CITATION.cff) or the DOI
[10.5281/zenodo.23524569](https://doi.org/10.5281/zenodo.23524569), which resolves to the latest
release; each release also has its own DOI on Zenodo. Record the version you used.

See [CONTRIBUTING](https://github.com/noelEOS/tensor-ee/blob/main/CONTRIBUTING.md) for contributions and [SECURITY.md](https://github.com/noelEOS/tensor-ee/blob/main/SECURITY.md) for
private vulnerability reports.
