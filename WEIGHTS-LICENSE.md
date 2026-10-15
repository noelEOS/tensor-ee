# Trained weights licence

The trained weights of TensorEE's two pretrained models, the SegWater ResNet18 U-Net and the
fixed 224 × 224 Swin-T/UPerNet, are licensed under the **Creative Commons Attribution 4.0
International licence (CC BY 4.0)**: <https://creativecommons.org/licenses/by/4.0/>.

## What it covers

- The 104 public Earth Engine table assets, in the project `tensor-ee`, that the weight
  manifests in [`data/manifests/`](data/manifests/) list: 29 for the U-Net
  ([manifest](data/manifests/segwater_r18_weight_manifest.json)) and 75 for the
  Swin-T/UPerNet ([manifest](data/manifests/m12_trained_weights_manifest.json)).
- The two models' `.pth` checkpoints, once they are published. They are not public yet.

## What it does not cover

- The code and documentation of this repository, which are under the [MIT licence](LICENSE).
- Satellite data and third-party libraries, which keep their own terms.

## Attribution

A suggested attribution:

> TensorEE trained weights © 2026 Noel Ivan Ulloa, licensed under CC BY 4.0. Contains modified
> Copernicus Sentinel data.

## Training data and provenance

Both models were trained by Noel Ivan Ulloa. Their inputs are Sentinel-1 chips of
224 × 224 pixels at 10 m, VV and VH in dB; the [model notes](docs/model.md) give each model's
input and preprocessing. Their water labels are derived from Sentinel-2. The training data are
Copernicus Sentinel-1 inputs and Sentinel-2-derived labels, so the weights contain modified
Copernicus Sentinel data. A fuller description of the dataset is not yet published.
