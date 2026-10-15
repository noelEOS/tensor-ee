"""Core layer of the native EE library: grids, tensor types (R, W and the V
carrier ``ServerArray``), errors, the ops interface, precision policies and the
metadata schema.

Modules here import only ``ee``, NumPy and the standard library
(docs/library-contract.md, section 2.1). Nothing here imports a research
folder, ``tensor_ee.blocks`` or the converter.
"""
