"""Weights layer of the native EE library: coefficient transport and the asset format.

- ``constants``: ``ServerArray``, ``constant_value``, ``inline_float32_array``
  and layout-derived row repetition (docs/library-contract.md, section 6);
- ``assets``: the FeatureCollection weight-asset format
  (``ConvolutionWeights``, ``convolution_weights``, ``feature_records``) and
  ``FeatureCollectionWeightProvider``.

Modules here import only the standard library, ``ee``, NumPy,
``tensor_ee.core`` and ``tensor_ee.layouts`` (section 2.1). Nothing here
imports a research folder, ``tensor_ee.blocks`` or the converter. BatchNorm
folding (``effective_convolutions``), matching (``MatchedWeightProvider``) and
the export and upload runners stay with the converter (decision D6).
"""
