# Start here

This page is for anyone with an Earth Engine account and some Python who wants to try
TensorEE. You do not need to know how the library is built, and you do not need any special
tools: a browser, and for the notebooks, Google Colab or Jupyter.

## What TensorEE does, and what it does not

TensorEE runs a trained neural network inside Earth Engine. You keep your imagery in Earth
Engine, and the model's arithmetic (convolutions, additions, activations) is written as
ordinary Earth Engine image and array operations, so Earth Engine computes the prediction
itself. There is no model server to deploy and no imagery to download. The weights are stored
as Earth Engine table assets. It works only for the kinds of model it supports: some
convolutional networks (CNNs) such as standard U-Nets, and a few attention models through a
converter. It refuses anything else with a message saying what is missing, rather than
running an approximation. It does not train models, and it does not tell you how accurate a
model is: it checks that Earth Engine computes the same numbers as PyTorch, which is a
different question from whether the predictions are right.

## Pick a path

### 1. See it work, with nothing to install

The [Code Editor demo](../CODE_EDITOR.md) runs a trained water-mapping U-Net on Sentinel-1
radar imagery. You paste two scripts into the Earth Engine Code Editor, click **Run**, and
export one 224 × 224 pixel prediction to your own project. It needs an Earth Engine account and
a Cloud project [registered for Earth Engine](https://developers.google.com/earth-engine/guides/access).
One export is billed to your project; the first recorded run took 140 s and 64.9 batch
EECU-seconds ([record](evidence/l06d-user-check.json)).

### 2. Use it from Python

[Notebook 01, quickstart](../notebooks/01_quickstart.ipynb)
([open in Colab](https://colab.research.google.com/github/noelEOS/tensor-ee/blob/main/notebooks/01_quickstart.ipynb))
does the same thing from a Python notebook: it installs TensorEE, signs in with your own Earth
Engine project, finds a Sentinel-1 scene at a place you pick, runs the same trained U-Net on
one chip there, and shows the radar image and the water map on an interactive map. It uses
only the public weights; you do not need the model's checkpoint file.

Every notebook first runs offline: it builds the Earth Engine computation without sending it
anywhere, so you can read and run it without an account. The steps that use your Earth Engine
project are switched off until you change `RUN_LIVE = False` to `RUN_LIVE = True` and put your
project ID in place of `YOUR_EE_PROJECT`. Before each of those steps the notebook says what it
will create and what it costs.

### 3. Bring your own model

- [Notebook 02, is my model supported?](../notebooks/02_is_my_model_supported.ipynb)
  ([open in Colab](https://colab.research.google.com/github/noelEOS/tensor-ee/blob/main/notebooks/02_is_my_model_supported.ipynb))
  needs no Earth Engine account at all. It loads a PyTorch model, checks it, and explains in
  plain words what is supported, what is refused and why, and what you can do next. It shows a
  model that passes and models that are refused, and has a cell where you put your own.
- [Notebook 03, bring your own CNN](../notebooks/03_bring_your_own_cnn.ipynb)
  ([open in Colab](https://colab.research.google.com/github/noelEOS/tensor-ee/blob/main/notebooks/03_bring_your_own_cnn.ipynb))
  follows a trained segmentation-models-pytorch U-Net from its checkpoint file to Earth
  Engine: check it, plan how to cut a region into tiles, upload its weights once, predict one
  tile, compare that tile with PyTorch, and export a region. Each step that is billed to your
  project is switched off until you turn it on.

When you want the details, the [Getting started](getting-started.md) page and the
[guides](README.md) are the reference.

## Words you will meet

- **Chip.** One fixed-size square of pixels that a model takes as input, for example
  224 × 224 pixels of Sentinel-1 at 10 m. The trained U-Net was trained on chips of that size.
- **Tile.** One piece of a larger region. A region too big for one run is cut into tiles that
  Earth Engine computes one by one, and the results are put back together.
- **Halo.** The extra border of input pixels around a tile that a model needs to compute the
  tile's edge correctly, because each output pixel depends on input pixels around it. The halo
  is computed but thrown away, so it is a cost: for a standard U-Net with a ResNet-34 encoder
  it is 544 pixels on each side, so a 1,024-pixel tile reads a 2,112-pixel square.
- **Weight asset.** An Earth Engine table asset that holds a model's numbers (its weights).
  You upload them once, and every prediction reads them from there. The trained U-Net's weight
  assets are public, so you can use them without uploading anything.
- **Export.** An Earth Engine batch task (`Export.image.toAsset` or `toDrive`) that computes an
  image and saves it as an asset or a file. Whole models run as exports, because one
  interactive request with a whole model exceeded Earth Engine's memory limit. An export runs
  in the background: you start it, wait for **Completed** in the Tasks list, then look at the
  result.
- **EECU cost.** Earth Engine measures computation in EECU-seconds (Earth Engine Compute
  Units), and bills commercial projects for them; see Google's
  [Earth Engine pricing](https://cloud.google.com/earth-engine/pricing). Batch exports and
  interactive requests are counted separately. The notebooks give the recorded cost where one exists, and say so when a cost has not
  been measured.

## What has been checked

TensorEE keeps four kinds of claim apart, and the notebooks do too:

- **Graph identity**: the Earth Engine computation is exactly, byte for byte, one that was
  checked before.
- **Offline agreement**: each building block, computed in NumPy, matches PyTorch.
- **Live agreement**: Earth Engine's result matches PyTorch on the same input, within an
  absolute difference of 1e-4 in the logits. For the trained U-Net the recorded single-chip
  run differed by at most 1.35e-5 ([validation](validation.md)).
- **Accuracy**: how often the predictions are right. This is not measured by TensorEE. You
  measure it on your own labelled data.

If a notebook gets a result you did not expect, or refuses a model you think should work,
[open an issue](https://github.com/noelEOS/tensor-ee/issues) with the message it printed.
