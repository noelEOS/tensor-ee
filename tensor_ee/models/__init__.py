"""Model layer of the native EE library (docs/library-contract.md, section 2.1).

- ``tensor_ee.models.fixtures``: the small deterministic PyTorch fixture models
  (formerly ``tensor_ee/models.py``; decision D3). Their names are re-exported
  here as the same objects, so ``from tensor_ee.models import TinyUNet`` keeps
  working.
- ``tensor_ee.models.swin_t224``: the fixed 224 x 224 Swin-T/UPerNet reference
  model's configuration, encoder and decoder composites, built only from the
  library blocks. Import it explicitly: neither ``import tensor_ee`` nor this
  package loads it.

The SegWater U-Net recipe stays in ``tensor_ee/segwater_*.py`` and the Swin-T224
adapter in ``tensor_ee/swin.py``.
"""

from .fixtures import ConvBlock, PoolRoundTrip, SmallCNN, TinyUNet, initialize_fixture, parameter_count  # noqa: F401
