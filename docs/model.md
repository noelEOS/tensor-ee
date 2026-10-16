# Models and their evidence

Four models are built only from the library: two trained water models, the SegWater ResNet18
U-Net and a fixed 224 × 224 Swin-T/UPerNet (a Swin Transformer encoder with a UPerNet decoder),
and two with random weights: SegFormer-B0 from segmentation-models-pytorch (SMP), and a 256 ×
256 Swin-UPerNet in a configuration that had never run in Earth Engine. This page records how
each was checked, with the record of each run, and the trained models' checkpoints and
recipes.

Three kinds of evidence apply, and they are kept apart:
- **numerical agreement**: Earth Engine outputs compared with PyTorch on recorded
  inputs, under a 1e-4 absolute logit target;
- **graph identity**: serialized Earth Engine graphs compared byte for byte, offline;
- **model accuracy**: not measured here.

## SegWater ResNet18 U-Net

The U-Net runs through `compile_model` ([walkthrough](water-chip.md)). When the library was
packaged for release, its trained graphs were preserved exactly
([record of that check](evidence/t05-preservation.json)). In an independent live rehearsal
from a fresh installation, one 224 × 224 Sentinel-1 chip matched the original PyTorch model:
largest logit error 1.3455748558044434e-05, largest probability error
6.4373016357421875e-06, 0 argmax and 0 threshold disagreements over 50,176 pixels
([record of the rehearsal](evidence/t06-rehearsal.json)). The [README figure](../README.md) was rendered
from a verified export of this model on one Sun Moon Lake chip
([validation](validation.md)); its water mask is a model prediction, not ground truth.

**A known failure is kept visible.** On the U-Net's synthetic upper-left impulse
chip, the largest logit difference from the original PyTorch model is 1.6510487e-4,
above the 1e-4 target; probabilities and the water mask agree
([record](history/cnn/SEGWATER_PYTORCH_RESULTS.md)).

The [checkout single-chip example](water-chip.md) uses this ResNet18 U-Net, trained for
two-class Sentinel-1 water mapping.
Its original input is float32 `[1, 2, 224, 224]`, ordered VV then VH. Outputs are
two logits per pixel. Class 1 is water; the demonstration uses probability ≥ 0.5.

| Item | Value |
| --- | --- |
| Checkpoint filename | `unet_resnet18_s42_step6000_last.pth` |
| SHA256 | `2afeaef8086bd85dc41741126ef7af7dbf7462ce54802a5e5ff18e9162d74928` |
| Architecture library | `segmentation-models-pytorch==0.5.0` |
| Encoder | ResNet18, depth 5 |
| Decoder channels | 256, 128, 64, 32, 16 |
| Input | One 224 × 224 chip, VV/VH dB on a 10 m WGS84 UTM grid |
| Checkpoint download | Not yet public; to be published under CC BY 4.0 |
| Weight license | [CC BY 4.0](../WEIGHTS-LICENSE.md): the public table assets now, the checkpoint when published; not the code's MIT license |

The checkpoint file is not public, but its folded coefficients are: the 29 Earth Engine
table assets that the [weight manifest](../data/manifests/segwater_r18_weight_manifest.json)
lists are readable by all users, in the project `tensor-ee`, which is how the
[Code Editor demo](../CODE_EDITOR.md) runs without a checkpoint. The Swin-T/UPerNet's assets
are public in the same project. Both are licensed under CC BY 4.0
([WEIGHTS-LICENSE.md](../WEIGHTS-LICENSE.md)).

Checkpoint tensors are loaded strictly with `weights_only=True`. The adapter
copies the original model, uses nearest 2× decoding at the fixed resolutions
and disables in-place ReLU. The compiler folds evaluation BatchNorm into the
convolutions. The reference remains the original unfused SMP model.

Preprocessing reproduces the training pipeline's q55s005 quantization and
VV/VH standardization in float32, including ties-to-even rounding. The
implementation and numeric constants are in
[`segwater_preprocessing.py`](../tensor_ee/segwater_preprocessing.py).
Missing either source channel zero-fills both normalized channels and masks
the prediction at that pixel. Numeric zero and −55 dB are not nodata sentinels.

The checkpoint and original training data are not bundled in the repository. The
Sentinel-1 chips used here contain modified Copernicus Sentinel data (2024).
The training data's provenance is [below](#training-data-and-provenance). The current evidence establishes execution and numerical agreement
on specified chips, not general water-mapping accuracy. See
real-pixel parity (development archive: `docs/history/cnn/SEGWATER_REAL_PARITY.md`) and
the [boundary-impulse limitation](history/cnn/SEGWATER_PYTORCH_RESULTS.md).

## Fixed Swin-T/UPerNet recipe

**Evidence** ([Swin guide](swin.md)). The last research milestone
compared the trained model in Earth Engine with its ONNX Runtime float32 deployment on
32 chips: largest difference 1.633e-5, 0 argmax and 0 threshold disagreements
([findings](history/upernet/notes/milestone12_findings.md)). In the same independent live
rehearsal as the U-Net's, one chip matched the original PyTorch model: largest logit error
5.9604644775390625e-06, 0 argmax and 0 threshold disagreements
([record of the rehearsal](evidence/t06-rehearsal.json)).

The separately imported `tensor_ee.swin` supports one fixed two-input,
two-class 224 × 224 SMP UPerNet with encoder
`tu-swin_tiny_patch4_window7_224`. It is not a general transformer compiler.
Use the [installed three-stage walkthrough](swin.md) with explicit checkpoint,
manifest, project, input/validity assets, grid and output destinations.

| Item | Value |
| --- | --- |
| Checkpoint | `upernet_tu-swin_tiny_patch4_window7_224_s42_step6000_last.pth` |
| SHA256 | `5c7a7c44d29b3c75bd69f59809fb1b63a039313bec021d897ff3876d2cea42a5` |
| Inference tensors | 239 finite float32 tensors; 36,747,964 values |
| Bookkeeping | 12 int64 BatchNorm counters accounted for but unused |
| Dependencies | `segmentation-models-pytorch==0.5.0`, `timm==1.0.30` |
| Input | Two ordered prepared channels, exactly 224 × 224 |
| Weight license | [CC BY 4.0](../WEIGHTS-LICENSE.md): the public table assets now, the checkpoint when published (it is not bundled and not yet public) |

The named SegWater-v3 dB lane fills each missing channel with its training mean
**before** quantization, so its prepared fill can be slightly nonzero. It carries
original per-channel availability separately; source-intersection coverage does
not erase dense computational logits. The U-Net recipe above instead fills exact
model-domain zero after normalization. These choices are preserved separately.
The exact dataset lane uses finite fp16 z-scores widened to fp32 without another
quantization step. Both recipes retain water class 1 and inclusive probability
threshold 0.5.

The changed adapter's catalogue224, `hpo_val_00` and near-margin `hpo_val_04`
regression measured maximum raw-logit errors 1.30e-5, 4.08e-6 and 1.46e-5;
argmax and class-1 threshold disagreements were zero. Exact inputs, checkpoint/
ORT identities and separate saved stages are recorded in
[its record](evidence/t01-live.json). That record concerns the nominated
fixtures, not other sensors, resolutions, architectures or segmentation accuracy.

## Training data and provenance

Both models were trained on Sentinel-1 chips of 224 × 224 pixels at 10 m, VV and VH in dB;
the tables above give each model's input and preprocessing. Their water labels are derived
from Sentinel-2. The training data are Copernicus Sentinel-1 inputs and Sentinel-2-derived
labels, so the weights contain modified Copernicus Sentinel data. A fuller description of the
dataset is not yet published. The weights are licensed under CC BY 4.0, with the attribution
in [WEIGHTS-LICENSE.md](../WEIGHTS-LICENSE.md).

## Why that evidence still applies

The library refactor that followed changed no computation. Its graph-identity gate (G1,
[CONTRIBUTING](../CONTRIBUTING.md#which-gate-applies)) builds 45 production graphs (9 trained,
36 fixtures) offline and requires each to serialize byte-identically to the hash recorded
before the refactor ([golden hashes](evidence/l00-golden-graphs.json)); every step of the
refactor passed it. The trained graphs include the eight that the independent rehearsal
validated live, so that validation holds without a rerun.

**The rebuilt demo reproduced both models' outputs exactly** in a live run after the
refactor ([record of the run](evidence/l06c-live-run.json)): the U-Net result equals the
earlier demo's verified run (largest difference 0 in all five bands over 25,577
pixels), and the Swin-T logits equal those the last research milestone saved (largest
difference 0 over 50,176 pixels, 0 argmax disagreements).

## SegFormer-B0

SMP 0.5.0 `Segformer("mit_b0")` is built on the library: fixed
224 × 224, two input bands and two raw logits, composed from window blocks in
`tensor_ee/models/segformer_b0/`; its task record, `docs/tasks/codex-segformer_b0.md`, is in the
development archive. It added one block, the depthwise 3 × 3 convolution, and one attention setting,
`score_scale`. The [converter](converter.md) rebuilds the same block graph from SMP's model.

It is validated end to end in Earth Engine with random weights and a synthetic chip.
All 100,352 logits match SMP's PyTorch model within 2.9802322387695312e-06 (target
1e-4), with 0 argmax disagreements, and every stage output is bit-identical to an
independent NumPy mirror ([record of the run](evidence/segformer_b0-fullmodel-live.json)).
It was validated as six staged exports, with its weights in table assets. It also runs
as a single graph in one batch export, with bit-identical logits, but that took 77.7
minutes and 5,870 batch EECU-seconds, against about 5 minutes and 91 for the six
stages; as one interactive request it exceeds Earth Engine's memory
([record of the single-graph run](evidence/segformer_b0-single-graph-live.json)).
Trained weights, real imagery and accuracy are not yet tested.

## A new Swin-UPerNet composed from the blocks

To test the blocks at settings no earlier model used, a Swin-UPerNet that had never run in
Earth Engine was composed from them:
- 256 × 256 pixels, 3 Sentinel-1 bands, patch 4, width 48, window 8;
- decoder width 64, 3 classes;
- random weights stored as 13 table assets.

It ran end to end as one batch export. Against PyTorch on the input Earth Engine used,
the largest logit difference was 9.775161743164062e-06 over 196,608 logits, with 0 argmax
disagreements ([record of the run](evidence/c1-capstone.json)). This is numerical agreement
with random weights, not accuracy.

## Synthetic CNN example

`python -m tensor_ee.examples.nonwater` uses reproducible arithmetic weights and
inputs without any pretrained checkpoint. Its four-band/three-class 16 × 12
fixture measured logit/probability errors below 1e-4 with zero valid-pixel class
disagreements. [Exact evidence](evidence/t05-primary.json) separates this small
execution demonstration from trained-model measurements. It is not a water model
or an accuracy benchmark.
