# Documentation

Start with [Getting started](getting-started.md). The guides below are the reader
documentation; the records after them hold the evidence the guides cite.

## Guides

| Guide | What it covers |
|---|---|
| [Getting started](getting-started.md) | Install, the offline example, both demos and a first conversion, with expected output |
| [Code Editor demo](../CODE_EDITOR.md) | The trained U-Net in the Earth Engine Code Editor: no install, no checkpoint |
| [Browser demo](../DEMO.md) | A local web app that runs either trained model in Earth Engine, with an offline mode |
| [Models and their evidence](model.md) | The four models built from the blocks, what each was checked against, and the trained checkpoints |
| [U-Net single-chip walkthrough](water-chip.md) | The trained U-Net from Python: upload its weights, export one chip, check it against PyTorch |
| [Release validation](validation.md) | The walkthrough's verified run, which the README figure was rendered from |
| [Swin-T/UPerNet workflow](swin.md) | The fixed trained Swin-T/UPerNet as three saved-asset stages |
| [API guide](api.md) | The CNN API: entry points, supported forms, numerical and validity rules |
| [Converter guide](converter.md) | Convert, report, plan, build and run SegFormer-B0 and the Swin-T/UPerNet |
| [Block catalogue](building-blocks.md) | Every block's layout, locality, precision policy, status and evidence; the layers and the public API |
| [Live fixtures](live-fixtures.md) | The harness that compares a block's Earth Engine result and its NumPy mirror with PyTorch |
| [Library contract](library-contract.md) | The rules the library is built to: layers, tensor types, policies, metadata, gates |
| [CONTRIBUTING](../CONTRIBUTING.md) | Development: tests, adding a block or a recognizer, the gates, CI |
| [SECURITY](../SECURITY.md) | Reporting a vulnerability, and the scope |
| [Weights licence](../WEIGHTS-LICENSE.md) | CC BY 4.0 for the trained weights: what it covers, the attribution, the provenance |

## Records

- [`evidence/`](evidence/): the records of live runs, offline checks and golden graph
  hashes that the guides and the catalogue cite. Each guide links the records for its
  claims.
- [`history/`](history/README.md): the milestone notes and result summaries the library
  cites, with the milestone results.
- [`validation/`](validation/water_chip.json): the walkthrough's comparison report.

Revisions, the archive tag and the name `native_ee` cited in documents and records are those
of the project's private development archive; they do not exist in this repository.
