# Changelog

## 0.2.0 (2026-11-12)

Second experimental release. It adds a foundation-model workflow, ViT and DPT conversion,
more raster blocks for standard CNN encoders, and a plain-language way in.

- **Prithvi-EO-1.0 Sen1Floods11.** `tensor_ee.models.prithvi_sen1floods11` composes the
  100M-parameter flood model from window blocks and runs it in staged exports, as one
  224 × 224 crop or a region of whole crops, from its released trained checkpoint. It was
  checked live against PyTorch on synthetic crops and on one real Sentinel-2 crop from the
  held-out Bolivia event: logits within 1.41e-5, with 0 argmax disagreements. Water IoU
  against that crop's hand label was 0.911, an example and not an accuracy score. A 2 × 2
  region of synthetic crops matched within 6.11e-6. Merged stages run a crop in 16 exports
  instead of 33.
- **ViT-B/16 and DPT conversion.** The converter recognizes timm ViT-B/16 encoders and SMP
  0.5.0 DPT models at 224 × 224, and converts them to graphs byte-identical to hand-composed
  targets. The numerical evidence is narrower than for the other models: random weights
  only; offline, the full-width head and logits are within 2.06e-5 of float32 PyTorch; live
  agreement covers reduced twins and individual blocks, not the full-width model.
- **Ten new blocks, 88 in all** (23 raster, 65 window). Raster: `relu6`, `hardsigmoid`,
  `hardswish`, `batch_norm`, `bilinear_upsample`, `grouped_conv2d` (grouped and depthwise)
  and `standardize`. Window: `class_token_attention`, `concat_rows` and
  `window_depth_to_space`. Each was validated live at the settings its catalogue row lists.
- **New settings on existing blocks,** each keyword-only, validated live, with today's
  default graph unchanged: stride 2 and per-crop padding for `conv3x3_window`, epsilon 1e-6
  for `layer_norm`, ×2 with aligned corners for `window_resize`, and an opt-in slice gather
  for `relative_position_bias`.
- **SMP adapter.** Unet on `resnext50_32x4d` and `mobilenet_v2` encoders now builds its graph
  offline through the grouped and depthwise convolution. Neither whole model has been
  checked in Earth Engine.
- **A way in.** A plain-language start guide and three Colab notebooks: a quickstart with the
  trained U-Net, an offline check of whether your own model is supported, and your own CNN
  from checkpoint to an exported region.
- **Numerical comparisons.** The 1e-4 absolute target against float32 PyTorch applies to
  block fixtures and to a model's final outputs. A model's intermediate outputs are checked
  against float64 PyTorch instead.
- **For contributors.** Gate fixtures are generated from integer arithmetic, so the
  graph-identity checks can be judged on Linux as well as on macOS arm64.

`pip install tensor-ee` provides the library, the converter and the models. The notebooks,
the tools that drive the Prithvi staged runs, the demos and the tests need a checkout.

Support and numerical evidence apply only to the documented settings, and numerical
agreement does not establish segmentation accuracy. The Prithvi checkpoint is a separate
third-party release under its own terms. The SegWater and Swin-T checkpoint files and the
full training dataset description are not yet public.

## 0.1.0 (2026-10-21)

First experimental release of TensorEE, for expressing supported PyTorch models as
Google Earth Engine image and array operations without a model-serving endpoint.

- A library of 78 raster, window and array building blocks, with explicit precision
  policies, locality, cost metadata and machine-readable validated settings.
- A restricted FX compiler for CNNs and an optional ONNX importer. Unsupported
  operations are refused rather than approximated.
- `tensor_ee.smp.adapt_smp_model`: standard segmentation-models-pytorch Unet and
  UnetPlusPlus models on resnet18/34/50 and vgg11_bn compile without a hand-written
  adapter. One trained Unet/resnet18 was checked live against PyTorch (largest
  logit error 1.195e-5). A guide takes your own trained Unet/resnet34 to a tiled,
  batch-exported region.
- An attention-model converter for SMP SegFormer-B0 and SMP UPerNet with timm's
  Swin-T encoder, with conversion reports, staged graph planning and a resumable
  export runner.
- SegWater ResNet18 U-Net and fixed 224 × 224 Swin-T/UPerNet reference workflows,
  asset-backed weight transport, tiled CNN inference and segmentation outputs.
- Offline examples, guides, graph-identity checks and recorded live comparisons.
  The checkout includes a Code Editor demo using public trained U-Net weights.

Support and numerical evidence apply only to the documented settings. The absolute
logit target is 1e-4; the U-Net's recorded synthetic upper-left impulse case reaches
1.6510487e-4. Numerical agreement does not establish segmentation accuracy.
Checkpoint files and the full training dataset description are not yet public.
Code and documentation use MIT; the trained weights use CC BY 4.0.
