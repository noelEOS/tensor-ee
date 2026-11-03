# Start here

For anyone with an Earth Engine account and some Python who wants to try TensorEE. A browser
is enough: the notebooks run in Google Colab.

## What TensorEE does, and does not do

- **Does:** runs a trained neural network inside Earth Engine. The model's arithmetic becomes
  ordinary Earth Engine image and array operations, so Earth Engine computes the prediction
  where your imagery already is. No model server, no image download; the weights are Earth
  Engine table assets.
- **Supports:** some CNNs, such as standard U-Nets, and a few attention models through a
  converter. Anything else is refused with a message naming what is missing, never
  approximated.
- **Does not:** train models, or tell you how accurate one is. It checks that Earth Engine
  computes the same numbers as PyTorch, which is a different question.

## Pick a path

1. **See it work, nothing to install.** The [Code Editor demo](../CODE_EDITOR.md): paste two
   scripts into the Earth Engine Code Editor and run a trained water-mapping U-Net on one
   Sentinel-1 chip.
2. **Use it from Python.** [Notebook 01, quickstart](../notebooks/01_quickstart.ipynb)
   ([open in Colab](https://colab.research.google.com/github/noelEOS/tensor-ee/blob/main/notebooks/01_quickstart.ipynb)):
   enter your project, **Run all**, and get a water map at a place you pick. No model file
   needed.
3. **Bring your own model.**
   - [Notebook 02, is my model supported?](../notebooks/02_is_my_model_supported.ipynb)
     ([open in Colab](https://colab.research.google.com/github/noelEOS/tensor-ee/blob/main/notebooks/02_is_my_model_supported.ipynb)):
     offline, no account. Checks a PyTorch model and explains what is supported, what is
     refused and why.
   - [Notebook 03, bring your own CNN](../notebooks/03_bring_your_own_cnn.ipynb)
     ([open in Colab](https://colab.research.google.com/github/noelEOS/tensor-ee/blob/main/notebooks/03_bring_your_own_cnn.ipynb)):
     a trained segmentation-models-pytorch U-Net from checkpoint to an exported region. It runs
     offline first; each Earth Engine step has its own switch.

For details, see [Getting started](getting-started.md) and the [guides](README.md).

## What a run costs

- Runs use your project's Earth Engine quota (EECU time, Earth Engine's unit of computation).
- That is free on a noncommercial project; a commercial project pays its plan's rate. See
  [Earth Engine pricing](https://cloud.google.com/earth-engine/pricing).
- The quickstart's export used about 65 EECU-seconds (64.9 recorded,
  [record](evidence/l06d-user-check.json)), plus a few small requests. Notebook 03 says where
  a cost has not been measured.

## Words you will meet

- **Chip:** the fixed-size square a model takes as input, for example 224 × 224 pixels of
  Sentinel-1 at 10 m.
- **Tile:** one piece of a region too large for one run. Tiles are computed one by one and put
  back together.
- **Halo:** the extra border of input a tile needs so that its edges come out as the model
  would compute them on the whole image. It is read and then dropped: for a U-Net with a
  ResNet-34 encoder, 544 pixels per side, so a 1,024-pixel tile reads 2,112 × 2,112.
- **Weight asset:** an Earth Engine table asset holding a model's weights. Uploaded once,
  read by every prediction. The trained U-Net's are public.
- **Export:** an Earth Engine batch task that computes an image and saves it as an asset or a
  file. Whole models run as exports, because one interactive request with a whole model
  exceeded Earth Engine's memory limit.
- **EECU-seconds:** how Earth Engine measures computation time; batch exports and interactive
  requests are counted separately.

## What has been checked

- **Graph identity:** the Earth Engine computation is byte for byte one checked before.
- **Offline agreement:** each building block, computed in NumPy, matches PyTorch.
- **Live agreement:** Earth Engine's result matches PyTorch on the same input, within 1e-4 in
  the logits. For the trained U-Net's recorded chip: at most 1.35e-5
  ([validation](validation.md)).
- **Accuracy:** not measured by TensorEE; measure it on your own labelled data.

Something unexpected, or a model refused that you think should work?
[Open an issue](https://github.com/noelEOS/tensor-ee/issues) with the message it printed.
