# Direct PyTorch checkpoint reference

*A history record, kept as recorded. Its commands ran the research code (`experiments/`, in
the private development archive; the commands do not run from this repository), and the result
files it cites are in that archive too, cited here by path.*

The direct path is a PyTorch `.pth` checkpoint → an evaluation-mode SMP U-Net →
native Earth Engine operations. It does not require an ONNX export or runtime.
Direct loading and live EE execution are verified. The unchanged 1e-4 raw-logit
target passes on texture and fails on the boundary impulse.

## Checkpoint and reference

- Checkpoint: `unet_resnet18_s42_step6000_last.pth`, training step 6000.
- Drive location: `<drive path>`.
- SHA256: `2afeaef8086bd85dc41741126ef7af7dbf7462ce54802a5e5ff18e9162d74928`.
- Library: `segmentation-models-pytorch==0.5.0`; ResNet18 encoder, decoder widths
  256/128/64/32/16, two input channels, two raw-logit output channels.
- Inputs tested here: float32 `[1,2,224,224]`, VV then VH, using the existing v3 preprocessing.
- Reference: the original SMP model with its separate BatchNorm operations in
  evaluation mode. State loading is strict, with `torch.load(weights_only=True)`.
- Deployment: an independent copy uses equivalent nearest-neighbor 2× decoder
  resizing on this fixed grid and out-of-place ReLUs. The existing compiler
  folds BatchNorm into convolution weights. All inference is float32.

This proof remains scoped to 224×224 chips. It does not establish arbitrary
chip sizes, large-area throughput, or interactive map performance.

## What was verified

All 31 effective convolution fingerprints match the previously uploaded assets
exactly, covering all 14,319,426 fused weight and bias elements. The direct graph
loads 29 FeatureCollection assets and contains 31 neighborhood extractions and
31 matrix multiplications, with no external prediction operation. No weights
were retrained or re-uploaded.

The original SMP reference and the fixed224 adapter produce bit-identical logits
and probabilities on the tested checkpoint fixtures. The suite passes **171
tests and 7 subtests**, including strict loading, dtype rejection, graph support,
asset identity, and reporting gates. An additional offline run succeeded with
ONNX and ONNX Runtime package discovery disabled.

BatchNorm folding is mathematically equivalent but changes float32 rounding.
A local diagnostic deep-copied the original SMP model and folded its 30
BatchNorm layers using `torch.nn.utils.fuse_conv_bn_eval`. That folded SMP model
matched the previous imported model **bit-for-bit** on texture and impulse.
However, against the original unfused SMP model, the impulse reached
**2.6717782e-4** logit error and **1.0603666e-4** water-probability error. Both
exceed the unchanged 1e-4 target, while the impulse masks remain identical.
The texture stayed within tolerance, with one near-threshold mask difference
in this local folded-versus-unfolded comparison. See
the isolated local comparison (development archive: `docs/history/cnn/results/segwater_pth_migration_reference.json`).
This demonstrates that the distinction between folded and unfused evaluation
already matters inside PyTorch; it is not solely an EE or model-format issue.

## Live EE versus the original PyTorch checkpoint

Both jobs completed on 224×224 chips without memory errors. Each comparison
covered all 50,176 pixels, including both raw-logit channels. The reference was
the untouched evaluation-mode SMP model with unfused BatchNorm.

| Fixture | Maximum logit error | Maximum water-probability error | Mask differences | Overall at 1e-4 |
| --- | ---: | ---: | ---: | --- |
| Texture | 2.6166439e-5 | 1.1026859e-5 | 0 | Pass |
| Upper-left impulse | 1.6510487e-4 | 6.1631203e-5 | 0 | **Fail: logits** |

The probability comparisons both pass. Mask equality uses water probability
≥ 0.5. No tolerance was relaxed. These two synthetic chips establish numerical
behavior on the tested inputs, not universal parity or segmentation accuracy.

EE execution took 152.281 seconds for texture and 180.217 seconds for impulse;
wall time including submission and queueing was approximately 180 and 341 seconds.
These are single-run batch observations, not an interactive-performance benchmark.
The scalar export requests were 3,995,219 and 4,008,901 bytes respectively.

Evidence (development archive, under `docs/history/cnn/results/`): texture
(`segwater_r18_pth_synthetic_pattern.json`), impulse
(`segwater_r18_pth_synthetic_upper_left_impulse.json`), texture graph audit
(`segwater_r18_pth_synthetic_pattern_graph_audit.json`), impulse graph audit
(`segwater_r18_pth_synthetic_upper_left_impulse_graph_audit.json`).
These runs reused `results/segwater_r18_weight_manifest.json` after the full
coefficient identity check. To reproduce them, use that manifest in the commands
below rather than creating new assets.

The earlier real Sentinel-1 smoke test remains documented in the
trained-checkpoint report (development archive: `docs/history/cnn/SEGWATER_RESULTS.md`). It used the same effective
coefficients; this reference-change milestone reran the two synthetic comparisons,
and did not download imagery or perform a new local Sentinel-1 comparison.

## Reproduce without an intermediate model format

Place the checkpoint in `checkpoints/`, then install the model library:

```sh
.venv/bin/python -m pip install -e '.[test,segwater]'
.venv/bin/python -m experiments.segwater_parity \
  --checkpoint checkpoints/unet_resnet18_s42_step6000_last.pth \
  --offline-only
```

For a new deployment, upload the directly folded PyTorch coefficients once:

```sh
.venv/bin/python -m experiments.segwater_weights \
  --checkpoint checkpoints/unet_resnet18_s42_step6000_last.pth \
  --manifest results/segwater_r18_pth_weight_manifest.json \
  --project YOUR_EE_PROJECT
.venv/bin/python -m experiments.segwater_weights \
  --checkpoint checkpoints/unet_resnet18_s42_step6000_last.pth \
  --manifest results/segwater_r18_pth_weight_manifest.json \
  --project YOUR_EE_PROJECT --verify
.venv/bin/python -m experiments.segwater_parity \
  --checkpoint checkpoints/unet_resnet18_s42_step6000_last.pth \
  --manifest results/segwater_r18_pth_weight_manifest.json \
  --project YOUR_EE_PROJECT --tile-scale 1
```

Repeat the last command with `--case upper_left_impulse` for the boundary fixture,
or `--mode sentinel` for the real Sentinel-1 execution smoke test. Satellite
pixels stay inside EE; only scalar diagnostics return to the client. Synthetic
reference tensors are generated locally and embedded solely for numerical tests.

The runner accepts an existing manifest only when every effective convolution's
weight and bias bytes exactly match its fingerprint. It records the mapping
from PyTorch node names to asset layer names and rechecks coefficients during
compilation. This permits verified asset reuse without changing asset provenance.
