# Release preparation validation

The new `examples.water_chip` command was run against Earth Engine with the
documented pretrained checkpoint, the pinned Sun Moon Lake scene and exactly
one 224 × 224 grid. It reused the previously uploaded coefficients through
`MatchedWeightProvider`; no checkpoint tensors were re-uploaded for this check.
The batch image export completed successfully.

`examples.verify_water_chip` then downloaded the completed asset's exact float32
pixels and ran the original, unfused PyTorch model on the exported normalized
input. The machine-readable [comparison report](validation/water_chip.json)
records the grid, scene, checkpoint hash, array archive hash and library versions.

| Check | Result |
| --- | ---: |
| Valid pixels | 50,176 / 50,176 |
| Local versus EE preprocessing | Bitwise equal |
| Maximum absolute logit error | 1.3455748558e-5 |
| Maximum absolute water-probability error | 6.4373016357e-6 |
| Mask disagreements at probability ≥ 0.5 | 0 |
| Exported mask agrees with exported probability | Yes |
| Absolute tolerance for logits and probability | 1e-4 |

The README figure was rendered from this fresh export's downloaded arrays.
The asset, full task status and downloaded arrays remain local/private; readers
can create their own output by following [the walkthrough](water-chip.md).
That workflow does not need access to those assets.

This is numerical validation of one chip. It does not establish segmentation
accuracy, universal numerical bounds or larger-area performance. The earlier
synthetic boundary impulse exceeds the strict logit target and remains documented
in the [original report](history/cnn/SEGWATER_PYTORCH_RESULTS.md).

The offline suite also verifies that manual primitive composition produces the
same serialized graphs as the compiler, including skip connections, pooling,
residual addition and optional finite-footprint clipping. Package build and
installation outside the checkout were checked locally. Hosted GitHub CI has since
passed ([evidence](evidence/hosted-ci.json); the repository name and the Actions run URLs in that record are the private development archive's, not this repository's); a fresh reader's checkpoint download
remains unverified until publication.

At the release-preparation commit `2792599` (2026-10-01), the local suite reported
**554 passed, 18 subtests passed**; the suite has grown since. The two live building-block
quick-start modes also returned identical extrema: `c0_min=0.6043846011161804`,
`c0_max=0.6104217171669006`. These seeded random-network outputs demonstrate
execution and agreement between construction routes, not water classification.
