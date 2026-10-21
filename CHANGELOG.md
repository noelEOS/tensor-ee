# Changelog

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
