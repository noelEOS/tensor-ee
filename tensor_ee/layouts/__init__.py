"""Layout layer of the native EE library: value-moving converters.

- ``raster``: R pinning and zero extension, stride sampling, embedded fixtures;
- ``window``: R <-> W converters, halos and im2col;
- ``shift``: integer translation, cyclic shift, periodic extension, domain masks;
- ``boundaries``: stage boundaries, a staged model's saved images in and dense images
  out, and a model's input relabel and inverse relabel.

These modules move values without arithmetic. They import only ``ee``, NumPy
and ``tensor_ee.core`` (docs/library-contract.md, section 2.1);
``raster.embedded_raster`` takes ``encode_float32`` from
``tensor_ee.core.precision``, and ``window.reshape`` uses the one W reshape
expression in ``tensor_ee.core.ops``.

Window, ring, kernel and shift arguments default to today's values (7, 1, 3, 3).
Only those values are validated; other values are not.
"""
