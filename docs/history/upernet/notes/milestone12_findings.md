# Findings: the trained step-6,000 Swin-T + UPerNet in native Earth Engine vs the local `fp32_b32.onnx`

Experiments ran on 8 October 2026 with Earth Engine Python API 1.7.45, PyTorch 2.14.0, timm 1.0.30,
segmentation_models_pytorch 0.5.0, NumPy 2.5.3, ONNX 1.23.0 and ONNX Runtime 1.30.0 (CPU, macOS
arm64), against Cloud project `ORIGINAL_PROJECT` with application-default credentials. The model
artifacts were fetched from Google Drive with rclone into `downloads/` (gitignored, never
committed); only their hashes and paths are recorded here.

Every Earth Engine tensor was computed by EE graphs from EE assets. The sources were the
Sentinel-1 catalogue image, the 32 exact-tensor input test assets, the trained weight tables, and
each tile's A′ and B′ assets. Tensors were downloaded only for comparison. No ONNX or PyTorch
output, intermediate feature or logit was uploaded.

**Evidence**

- m12_numerical_comparison.csv (development archive: `docs/history/upernet/results/m12_numerical_comparison.csv`): 241 rows. The 1e-4 gate
  (`passes_1e-4`) applies only to logit rows. Intermediate rows against PyTorch are diagnostics,
  and the raw encoder activations reach |x| ≈ 458. The mirror rows for the same tensors are
  bit-identical.
- m12_numerical_summary.json (archive `upernet/results/m12_numerical_summary.json` at `research-archive-2026-10-12`): aggregate, per chip, and
  margins.
- m12_decision_disagreements.csv (development archive: `docs/history/upernet/results/m12_decision_disagreements.csv`): header only, no
  disagreement.
- m12_model_manifest.json (archive `upernet/results/m12_model_manifest.json` at `research-archive-2026-10-12`) and
  m12_onnx_contract.md (development archive: `docs/history/upernet/notes/m12_onnx_contract.md`): Phase 1.
- m12_onnx_pytorch_reconciliation.json (archive `upernet/results/m12_onnx_pytorch_reconciliation.json` at `research-archive-2026-10-12`) and
  m12_onnx_pytorch_reconciliation.md (development archive: `docs/history/upernet/notes/m12_onnx_pytorch_reconciliation.md`): Phase 2.
- m12_input_manifest.json (archive `upernet/results/m12_input_manifest.json` at `research-archive-2026-10-12`) and
  m12_input_contract.md (development archive: `docs/history/upernet/notes/m12_input_contract.md`): Phase 3.
- m12_architecture_contract.json (archive `upernet/results/m12_architecture_contract.json` at `research-archive-2026-10-12`): Phase 4.
- m12_trained_weight_manifest.json (archive `upernet/results/m12_trained_weight_manifest.json` at `research-archive-2026-10-12`) and
  m12_weight_readback.json (archive `upernet/results/m12_weight_readback.json` at `research-archive-2026-10-12`): Phase 5.
- m12_ee_task_manifest.json (archive `upernet/results/m12_ee_task_manifest.json` at `research-archive-2026-10-12`): Phases 10–11.
- Per-stage records: `results/raw/m12_stages.json` and `m12_assets.json`, with logs in
  `results/raw/logs/m12_*`.

**Tests**

Offline: test_m12_onnx_contract.py (archive `tests/test_m12_onnx_contract.py` at `research-archive-2026-10-12`),
test_m12_step6000_weights.py (archive `tests/test_m12_step6000_weights.py` at `research-archive-2026-10-12`),
test_m12_input_parity.py (archive `tests/test_m12_input_parity.py` at `research-archive-2026-10-12`),
test_m12_small_swin_contract.py (archive `tests/test_m12_small_swin_contract.py` at `research-archive-2026-10-12`),
test_m12_trained_fusion.py (archive `tests/test_m12_trained_fusion.py` at `research-archive-2026-10-12`) and
test_m12_onnx_ee_parity.py (archive `tests/test_m12_onnx_ee_parity.py` at `research-archive-2026-10-12`): 36 passed. Live:
test_m12_live.py (archive `tests/test_m12_live.py` at `research-archive-2026-10-12`) with `EE_LIVE=1`: 4 passed. The existing suite
still passes: 1,017 passed, 132 skipped (live), run after this milestone's small, backward-compatible
edits.

## Verdict

**Native EE reproduces the specified local step-6,000 FP32 ONNX model.** On the 32 exact chips
of the reported ORT/PyTorch control (3,211,264 logits, 1,605,632 pixels):

| EE vs | max \|Δ\| | MAE | RMSE | argmax | threshold |
|---|---:|---:|---:|---:|---:|
| ONNX Runtime FP32 (primary) | **1.633e-5** | 4.71e-7 | 8.19e-7 | **0** | **0** |
| Trained PyTorch FP32 (secondary) | 8.34e-6 | 8.26e-7 | 1.09e-6 | 0 | 0 |
| Controlled EE mirror | **0, bit-identical** | 0 | 0 | 0 | 0 |
| *Control: PyTorch vs ORT* | *1.663e-5* | *9.70e-7* | *1.36e-6* | *0* | *0* |

The max |Δ| target was ≤ 1e-4. The argmax and threshold columns count disagreeing pixels (target 0).

Every chip individually passes, with max |Δ| between 1.67e-6 and 1.63e-5 against ORT and no
disagreeing pixel. For a genuine Sentinel-1 catalogue tile, EE's preprocessing produces the ORT
input bit for bit, and the output matches ORT within 1.30e-5 with no disagreeing pixel.

## Answers to the brief's 21 questions

**1. Which ONNX artifact was used?**
`<drive path>`:
149,802,158 bytes, SHA-256 `75a5ad2e1e2e506072b918d15f848483ad33c4c4f6bf07f1246f74d8aab967f8`,
IR 8, opset 17, exported by pytorch 2.11.0 (TorchScript).

**2. Was its source established as the step-6,000 last checkpoint, not `best.pth`?**
Yes. `export.csv` and the export script name `..._s42_step6000_last.pth`. All 203 ONNX initializers
are bit-identical to `fold_batchnorm` of that checkpoint (question 4). `best.pth`, a
`.rclonelink`, was never fetched, and no step-5,000 file was used.

**3. Which PyTorch checkpoint file supplied the weights?**
`…/20261006-033738_s42/upernet_tu-swin_tiny_patch4_window7_224_s42_step6000_last.pth`:
441,258,091 bytes, SHA-256 `5c7a7c44d29b3c75bd69f59809fb1b63a039313bec021d897ff3876d2cea42a5`,
`step` = 6000. Its `model_state_dict` has 251 keys:

- 239 inference tensors, 36,747,964 values;
- 12 BatchNorm `num_batches_tracked` counters, which eval-mode inference never reads.

It loads with `strict=True` into the export script's `smp.UPerNet(...)` line. The optimizer state
was not used.

**4. Are the ONNX graph and the PyTorch reconstruction numerically equivalent on identical inputs?**
Yes, in their weights and their outputs.

- **Weights.** Every one of the 203 initializers is a bit-identical transform of the checkpoint:
  - 24: `fuse_conv_bn_eval` of the 12 Conv-BN pairs;
  - 51: Linear weights transposed;
  - 12: relative-position bias tables gathered by their index;
  - 116: the checkpoint tensors unchanged.

  The 5 attention-mask constants equal timm's `attn_mask` buffers.
- **Outputs.** On the 32 chips, ORT vs PyTorch differs by at most 1.657e-5 with 0 disagreeing
  pixels under either rule.
- **Fold and pool effects.** The fold alone moves the logits by ≤ 4.8e-6, and the export's static
  PSP pool by ≤ 2.6e-6. The remaining ORT–PyTorch difference comes from runtime kernels.

**5. What is the ONNX input/output contract?**
The input is `input`, float32 `[32, 2, 224, 224]`, with every dimension fixed; ORT rejects B = 1,
B = 16 and 448 × 448. Channels are SegWater v3 z-scores, VV then VH. Preprocessing is external:
the input feeds the patch-embedding Conv directly.

The output is `logits`, float32 `[32, 2, 224, 224]` at runtime (the dimension metadata is symbolic).
These are **raw logits**, ordered by SMP class index. They come from the head's `align_corners`
Resize, and no Softmax or ArgMax follows. Runs are deterministic: bit-identical across runs,
sessions and thread counts. Each chip is batch-independent: in reversed order or among zero chips
its logits are bit-identical.

**6. What does local inference do to the input?**
The 32 chips are `Chips().batch(np.arange(32))`: rows 0–31 of `hpo_10pct/val.memmap`, channels
(vv_norm, vh_norm). Each value is a float16 z-score widened exactly to float32. The z-scores were
built from q55s005 dB (`dB = code·0.05 − 55`, code 0 = nodata) with the training moments
VV −15.435106 ± 6.517790 and VH −24.856323 ± 9.117346, then stored as float16. The memmap has no
mask, and every value is finite.

The local scene runner (`common.read_scene` / `run_inference.py`) works in float32:

1. Cast the dB to float32.
2. `code = clip(round((dB + 55) / 0.05), 1, 65535)`, rounding ties to even.
3. `dB = code · 0.05 − 55`.
4. `z = (dB − mean) / std`, with a float32 cast after every operation.

There is no float16 step there, so the memmap chips are a dataset artifact that the catalogue
recipe cannot produce. Masked pixels would be filled with the training mean in dB, but the
catalogue tile is fully valid.

**7. Does the EE model see exactly the ORT input tensor?**
Yes, in both lanes.

- **Exact-tensor lane.** All 32 input test assets read back bit-identical to the ORT tensors, with
  asset (row, col, band) = tensor (row, col, channel). On chip 0, the image the EE patch embedding
  actually reads on the model grid is also bit-identical.
- **Catalogue lane.** EE's preprocessed model input is bit-identical to `preprocess_db_numpy`
  applied to the downloaded raw dB, and to `read_scene`'s own lines. That tensor is the one ORT was
  given.
- **Row identity.** The chip rows were fetched from the Drive memmap. Its stored MD5 `a9b9a80b…`
  equals the build manifest's, and all 32 chips' label counts equal those recorded with the
  subset indices.

**8. What is the effective Swin stage-4 behaviour at 224?**
Stage 4 is a single 7 × 7 window. timm suppresses its shift (`_calc_window_shift`: resolution 7 ≤
window 7 gives shift 0), so both stage-4 blocks are W-MSA with no mask. The ONNX graph agrees:
mask constants exist only for stage 1 block 1, stage 2 block 1, and stage 3 blocks 1/3/5, none for
stage 4. EE uses Milestone 5's `block_shifts(8, 8)`, which gives `[[0,3],[0,3],[0,3,0,3,0,3],[0,0]]`.
Its stage-4 blocks are bit-identical to the mirror, and within 7.6e-5 of PyTorch where C4 reaches
|x| ≈ 96 (chip 0). All 15 timm Pad amounts are zero at 224.

**9. Do all trained weights map to EE assets with exact read-back?**
Yes.

- **Records.** 189 records (36,747,964 values) in the encoder (M5), patch-embedding (M6) and
  decoder (M7) formats, uploaded to the new immutable namespace
  `upernet/trained/swint_step6000_last_fp32_1f398d3256ba`: 75 shards, 516 MB, 1.7 h. The hash is
  over the sorted (record, SHA-256) list.
- **Read-back.** Every value was read back through the production read path
  (`FeatureCollectionWeightProvider.arrays` → getInfo, 310 requests). All 189 records are
  bit-identical.
- **No orphans.** None of the 239 checkpoint tensors is unmapped, and no record is unmapped.
  `num_batches_tracked` was not imported (eval never reads it). No final encoder norm was fabricated:
  the checkpoint has none, so the encoder now skips building one.
- **Untouched.** Milestone 4–11 seeded weights were not modified.

**10. What are the trained intermediate errors, and does EE equal the mirror?**
On chip 0, EE equals the mirror bit for bit at all 35 fetched tensors (82 of 82 EE-vs-mirror
comparisons across the milestone):

- the patch-embedding tokens;
- stage-1 W-MSA and SW-MSA;
- every Patch Merging;
- stage 2 and 3 blocks;
- stage 4's two unshifted blocks;
- C1–C4;
- the four feature norms;
- PSP;
- F3–F1 and P3–P1;
- the fusion conv, BN and ReLU, and the double sum, BN and ReLU;
- both head convs;
- the logits.

Against trained PyTorch, max |Δ| is 1.7e-6 to 2.1e-5 everywhere except the large-magnitude encoder
activations. C3 (|x| up to 458 on chip 0) is 2.3e-4 off and C4 (|x| up to 96) 7.6e-5; these are float32 ulps of large
values, and the feature norms bring them back to ≤ 2.1e-5. Across the network, no operation shows
a material disagreement.

**11. What are the errors of monolithic fusion and of formulation B?**
Both are bit-identical to their mirrors, and the fused feature is within 2.12e-5 of PyTorch on
chip 0. On chip 23, the one chip where the mirror predicted a post-ReLU difference, B differs from
monolithic at exactly 1 of 802,816 values by 1 ulp in each of the conv, BN and ReLU tensors.
The mirror predicted 1-ulp conv differences on chips 8, 11, 23, 29 and 31; on chips 0 and 23 the
measurements match the mirror exactly.

**12. Does B stay equivalent through BN, ReLU and the trained head?**
Yes. The head conv and the logits are bit-identical between B and monolithic:

- on chip 0 at every level;
- live, for the logits of all 5 chips where the fusion convs differ;
- in the mirror, for all 32 chips.

B is kept as the production formulation; the decision is measured, not assumed.

**13. Does the final `align_corners=True` 56→224 upsampling match?**
Yes. EE embeds exactly PyTorch's float32 coefficients for 56 → 224. ORT's own `align_corners`
coefficients differ from PyTorch's by up to 3.8e-6 at 146 entries, with the same support; that is
part of the ORT–PyTorch kernel difference. EE's output order is NCHW ↔ raster (row, col), and its
head conv and logits are bit-identical to the mirror.

**14. What are the per-chip and aggregate MAE, RMSE and max |Δ| of EE vs ORT?**
The aggregate is in the verdict table. Per chip, max |Δ| ranges from 1.67e-6 (`hpo_val_11`) to
1.63e-5 (`hpo_val_18`, (130, 212), class 0), MAE from 2.3e-7 to 1.25e-6, and RMSE from 3.1e-7 to
1.79e-6. Relative error uses a floor of 1e-3 on |reference|: it is ≤ 0.0111 overall and ≤ 0.0068 on
|reference| ≥ 1e-3. The per-chip table is in `m12_numerical_summary.json`.

**15. Are all class decisions equal?**
Yes: **0 of 1,605,632 pixels** disagree under argmax (ties go to class 0) and under the export
check's `softmax ≥ 0.5` rule. There are no exact ties.

ORT's margins |l₁ − l₀| come within EE's error range at only 2 pixels (< 2e-5). EE matches ORT's
class at both:

| Pixel | ORT margin | EE margin | PyTorch margin |
|---|---:|---:|---:|
| `hpo_val_04` (96, 140) | +9.54e-7 | +5.39e-6 | +5.93e-6 |
| `hpo_val_03` (156, 20) | −7.27e-6 | −6.32e-6 | −7.03e-6 |

Four pixels lie below 1e-4 and 37 below 1e-3. The mirror predicted all of this before any EECU was
spent.

**16. Was the reported 32-chip ORT/PyTorch control reproduced?**
Yes, on the same 32 inputs: rows 0–31 of the same memmap, as checked in question 7. ORT vs PyTorch
was 1.657e-5 here against the 1.68e-5 reported, with 0 % disagreement in both. This ORT runs on
arm64; the handoff's ran on the x86 export host, so the numbers agree closely but are not bitwise.

**17. Can a genuine Sentinel-1 catalogue scene be processed natively and compared on identical
inputs?**
Yes. The tile `m12_catalogue_224` is native column 11814, row 13965 of
`S1A_IW_GRDH_1SDV_20240103T100134_…_B0BD`, on EPSG:32651 at 10 m, fully valid. ORT and EE both assign 34,259 of its 50,176 pixels to class index 1 and 15,917 to
class index 0, so both decisions occur on the tile. This is a presence count of the logits' argmax,
not an accuracy measure. The chain was:

1. **A′:** catalogue → SegWater v3 preprocessing → stages 1–3, 38 EECU·s.
2. **B′:** stage 4, 16 EECU·s.
3. **C:** batch, formulation B, 43 EECU·s.

EE's input equals ORT's bit for bit. The logits vs ORT: max 1.30e-5, MAE 5.6e-7, RMSE 1.14e-6,
0 disagreeing pixels, minimum ORT margin 4.7e-4. They are within 5.96e-6 of PyTorch and
bit-identical to the mirror.

**18. Is the EE output on exactly the expected native grid, with no extra resampling?**
Yes. The asset bands are `logit_0` and `logit_1`, float32, EPSG:32651, with transform
`[10, 0, 287680.95…, 0, −10, 2640838.84…]` (the tile's own) and 224 × 224. Every pixel is unmasked.
The first pixel centre is (287685.95, 2640833.84) and the last is (289915.95, 2638603.84).
The logits come from the tensor-index upsample followed by `changeProj` relabelling. EE's bilinear
resampling is never used, and nothing is resampled downstream.

**19. Did all required tasks run without external model inference, and what documents them?**
Yes. The tasks were:

- 32 input-asset exports;
- 33 A′ and 33 B′ exports (32 chips plus the catalogue tile);
- 3 batch task C exports: chip 0 monolithic and B, and the catalogue tile with B.

All succeeded with no memory or encoded-object error. Graph loads, read from the serialized graphs,
show A′ loading only the input asset or the catalogue image plus the trained weight tables; B′ and
C load only the previous M12 assets plus weight tables.

For chips 1–31, task C's identical graph was evaluated by `computePixels` (22–91 s each) instead of
being exported. EE ran about one batch task at a time for this account, and this saved 31 tasks.
The interactive and batch results are bit-identical on chip 0 and on the catalogue tile, as they
were on Milestone 11's `edge_224`. Every task ID, asset ID and cost is in `m12_ee_task_manifest.json`.

**20. What limits or remaining inference-parity issues exist?**
None blocks parity. The limits are scope and reference platform:

- **Reference platform.** ORT ran on arm64. The bitwise ORT outputs are platform-specific, though
  the comparison standard is not.
- **Decision margin.** The exact-tensor chips include one pixel with an ORT margin of 9.5e-7, about
  17× smaller than EE's maximum deviation from ORT. EE agrees with ORT there, but a differently
  rounded runtime could flip such a pixel without any semantic error. The 0-disagreement result is
  measured, not structurally guaranteed.
- **Tile size.** Only the 224 × 224 invocation is supported, as for the ONNX model itself.
- **Wall time.** The batch queue's one-at-a-time execution made 32 independent chains take hours.
  The cost per tile is small: A′ 36–59, B′ 15–26 and C 43–81 EECU·s.

The local reference contract needs no change.

**21. Has native EE reproduced the specified local step-6,000 FP32 ONNX model?**
**Pass.** On exactly the same FP32 inputs, native EE's two-channel logits agree with the local
ONNX Runtime FP32 model within 1.63e-5, against a target of 1e-4. Every per-pixel class decision
agrees on all 32 nominated chips, and on a genuine Sentinel-1 catalogue tile processed end to end
in EE. The weights are the reconciled step-6,000 checkpoint, read back exactly, and every EE tensor
equals its controlled-rounding mirror.

## Milestone success levels

| Level | Status | Basis |
|---|---|---|
| **Minimum** | met | Exact ONNX and checkpoint, 203/203 initializers reconciled, all weights read back exactly, chip 0 through EE |
| **Strong** | met | Bit-identical inputs, ≤ 1e-4 on all 32 chips with 0 disagreements, intermediates explained (mirror bit-identical), B checked against monolithic |
| **Full** | met | The original 32-chip set, plus a real Sentinel-1 tile through EE preprocessing → A′ → B′ → C on its native grid, input bit-identical to ORT's, within tolerance with 0 disagreements |

## Decisions and deviations

- **Explicit BatchNorm in EE.** EE implements the checkpoint's arithmetic: explicit BatchNorm with
  running statistics, the Milestone 7–11 path. The ONNX graph holds the folded form. The fold's
  effect (≤ 4.8e-6) is part of the measured EE−ORT difference; the folded weights were not
  substituted to shrink it.
- **Interactive task C for chips 1–31.** The batch export is the production form. Equality between
  the two was measured on chip 0 and the catalogue tile.
- **Inputs on a carrier grid.** The exact-tensor inputs are clearly labelled EE test image assets on
  a carrier grid (canonical_896's top-left 224 tile of the scene's native grid). That location's
  Sentinel-1 pixels are never read in this lane.
- **Code changes (backward-compatible).**
  - The encoder appends its final norm only when the parameters have one.
  - `skips_image` / `image_head` accept an already-normalized input and a valid-fraction image.
  - `decoder_weights.server_parameters` takes the decoder tensors to verify against.
  - `run_chain` can stop after B′.
- **Optional C → D split.** Not used; one task-C graph runs from the A′/B′ assets to the 10 m logits.
- **Out of scope.** No accuracy, IoU, label or model-selection metric was computed. Labels were used
  only as a provenance cross-check of chip identity.
