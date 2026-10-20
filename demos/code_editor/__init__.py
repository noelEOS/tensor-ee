"""The SegWater Earth Engine Code Editor demo (Code Editor demo).

``build`` generates the Code Editor module ``segwater.js`` from the graph that the
``tensor_ee`` library builds; ``verify`` holds the offline comparison helpers used by the
evidence script and tests. Importing either module loads no Earth Engine client, no
PyTorch and no ``tensor_ee`` code; those are imported only when a graph is built.
The demo is not part of the ``tensor_ee`` wheel.
"""
