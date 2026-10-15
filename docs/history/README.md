# Development history

This folder keeps the milestone notes and result summaries that the library cites: the
`METADATA` evidence and docstrings in `tensor_ee` name them, and
`tests/test_history_citations.py` checks that every cited file exists. One more record ships
because the README cites it: the U-Net's comparison with its original PyTorch checkpoint, which
holds the known impulse failure. They record what was
tried, measured and decided while the library was developed as `native_ee`; they are not part
of the library, and their numbers are left as recorded.

The rest of the research record stays in the private development archive: the research code
(at the archive's tag `research-archive-2026-10-12`), the pre-refactor code (at its commit
`24a6291`), the plans, the task briefs, the raw results and the milestone READMEs. Revisions,
the archive tag and paths cited in the notes below are the archive's; they do not exist in
this repository. A note cites archived code as ``archive `path:line` at `ref` ``.

## Milestones

The library grew out of a series of research milestones:

| Milestone | Result (largest difference from PyTorch float32; target 1e-4) |
|---|---|
| CNN: ResNet18 U-Net, SegWater, tiling | Full 14.32-million-parameter ResNet18 U-Net in batch: logit error 2.55e-7 ([summaries](#what-is-here)) |
| 1. Exact window attention | ≤ 8.3e-7 for one 7×7 window ([M1](ee_attention/notes/findings.md)) |
| 2. One Swin-T block | 5.8e-6 over the 224×224 raster ([M2](swin_block/notes/findings.md)) |
| 3. W-MSA + SW-MSA pair | 9.2e-6 over 224×224 tokens ([M3](swin_pair/notes/findings.md)) |
| 4. Stage 1 → patch merging → stage 2 | 1.13e-5 over 224 × 224 → 112 × 112 ([M4](swin_hierarchy/notes/milestone4_findings.md)) |
| 5. Complete Swin-T encoder | 5.5e-6 at every one of 602,112 elements ([M5](swin_encoder/notes/milestone5_findings.md)) |
| 6. Sentinel-1 image → encoder | 5.72e-6 at all 602,112 elements ([M6](swin_image_encoder/notes/milestone6_findings.md)) |
| 7–11. UPerNet decoder and complete forward | 5.54e-6 from `smp.UPerNet.forward` over 1,605,632 logits ([M7](upernet/notes/milestone7_findings.md)–[M11](upernet/notes/milestone11_findings.md)) |
| 12. Trained step-6,000 model | 1.633e-5 from ONNX Runtime on 32 chips, 0 argmax and 0 threshold disagreements ([M12](upernet/notes/milestone12_findings.md)) |

## What is here

| File | Content |
|---|---|
| `docs/history/building-blocks-2026-10-11.md` | The pre-refactor building-block inventory of `24a6291`, which sized the library refactor |
| `docs/history/ee_attention/notes/findings.md` | Milestone 1: exact window attention |
| `docs/history/swin_block/notes/findings.md` | Milestone 2: one Swin Transformer block |
| `docs/history/swin_pair/notes/findings.md` | Milestone 3: the W-MSA + SW-MSA block pair |
| `docs/history/swin_hierarchy/notes/milestone4_findings.md` | Milestone 4: stage 1, patch merging, stage 2 |
| `docs/history/swin_encoder/notes/milestone5_findings.md` | Milestone 5: the complete Swin-T encoder |
| `docs/history/swin_image_encoder/notes/milestone6_findings.md` | Milestone 6: a Sentinel-1 image through the encoder |
| `docs/history/upernet/notes/milestone7_findings.md` | Milestone 7: the UPerNet decoder foundation |
| `docs/history/upernet/notes/milestone8_findings.md` | Milestone 8: the PSP module |
| `docs/history/upernet/notes/milestone9_findings.md` | Milestone 9: the FPN |
| `docs/history/upernet/notes/milestone10_findings.md` | Milestone 10: the fusion block |
| `docs/history/upernet/notes/milestone11_findings.md` | Milestone 11: the segmentation head and the complete forward |
| `docs/history/upernet/notes/milestone12_findings.md` | Milestone 12: the trained step-6,000 model against ONNX Runtime |
| `docs/history/cnn/results/parity.json` | CNN: primitive parity summary |
| `docs/history/cnn/results/primitives_float32.json` | CNN: float32 primitive checks |
| `docs/history/cnn/results/resnet_primitives.json` | CNN: ResNet primitive checks |
| `docs/history/cnn/results/resnet18_full64_float32_policy.json` | CNN: the full ResNet18 U-Net under the float32 policy |
| `docs/history/cnn/results/tiled_resnet18_2x2_parity.json` | CNN: tiled inference parity (2 × 2) |
| `docs/history/cnn/results/segwater_js_parity_1x1.json` | CNN: the Code Editor module's parity with the library (1 × 1) |
| `docs/history/cnn/results/segwater_onnx_reference.json` | CNN: the ONNX Runtime reference of the SegWater checkpoint |
| `docs/history/cnn/results/segwater_native_mosaic_2x2.json` | CNN: the archived browser demo's verified 2 × 2 mosaic, offered by `demos/browser` as a labelled U-Net example |
| `docs/history/cnn/results/demo_live_results.json` | CNN: the archived browser demo's verified live result, offered by `demos/browser` as a labelled example |
| `docs/history/cnn/results/swin_demo_example_checks.json` | CNN: the archived Swin-T demo's verified example, offered by `demos/browser` as a labelled example |
| [`docs/history/cnn/SEGWATER_PYTORCH_RESULTS.md`](cnn/SEGWATER_PYTORCH_RESULTS.md) | CNN: the trained U-Net in Earth Engine against its original PyTorch checkpoint, with the synthetic impulse chip above the 1e-4 target; its Drive path reads `<drive path>`, and the result files it cites stay in the archive |

Each note keeps its original relative layout under its milestone folder, so the links between
the notes that ship still resolve. The CNN JSON summaries carry asset namespaces and task
identifiers of the original project as `projects/ORIGINAL_PROJECT/assets/…` placeholders.
