"""Local browser demo: draw an area, pick a Sentinel-1 scene, run a model in Earth Engine.

Start it with ``python -m demos.browser.server --project YOUR_PROJECT`` (see
``DEMO.md``). The SegWater ResNet18 U-Net is the default; ``--model swin-t`` runs
the fixed 224 x 224 Swin-T/UPerNet. Every model operation runs in Earth Engine:
this package builds the graphs with the library, submits batch exports and
serves their status. It never runs a local forward pass.

Modules:

- ``geometry``: AOI validation, the 2240 m chip lattice and inclusive dates
  (no Earth Engine calls);
- ``exports``: persisted export requests, idempotent submission and the
  Earth Engine initialization the server performs;
- ``server``: the U-Net controller, the loopback HTTP handler and the command line;
- ``swin``: the one-chip Swin-T/UPerNet controller (three chained exports);
- ``offline``: ``--offline`` mode, a local fixture with no Earth Engine requests.

Importing any of them neither initializes Earth Engine nor sends a request.
"""
