# Security policy

## Reporting a vulnerability

Report a vulnerability privately, through GitHub's private vulnerability reporting:
open the repository's **Security** tab and choose **Report a vulnerability**
([direct link](https://github.com/noelEOS/tensor-ee/security/advisories/new)). Please do
not open a public issue for it.

Say what is affected (a module, a demo or a command), the commit you used, how to
reproduce it, and what an attacker gains. Never include credentials, project IDs or
private asset IDs; use placeholders such as `YOUR_EE_PROJECT`. There are no releases
yet, so fixes go to the latest commit on `main`.

## Scope

In scope:

- **The library**, the `tensor_ee` package, including the ONNX importer, the checkpoint
  loaders, the weight manifests and assets it reads, and the converter.
- **The demos**: the local browser demo ([DEMO.md](DEMO.md)) and the Code Editor scripts
  ([CODE_EDITOR.md](CODE_EDITOR.md)).

Out of scope: vulnerabilities in the dependencies (PyTorch, ONNX, the Earth Engine
client, SMP, timm, Leaflet) and in Earth Engine itself, which go to their own projects,
and the pretrained checkpoints, which this repository does not distribute.

## What the design already assumes

- Converting a model runs the model's Python code
  ([converter guide](docs/converter.md)): convert only models whose source you trust.
- The ONNX importer refuses tensors stored in external files, and the checkpoint loaders
  read with `torch.load(..., weights_only=True)`.
- The browser demo binds to loopback and assumes a single-user machine (DEMO.md,
  "Security").
