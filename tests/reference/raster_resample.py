"""Separable raster bilinear mirror using caller-supplied PyTorch coefficients.

The decoder's resize mirror fixes align_corners=False and builds its own
coefficients. This version takes both matrices so it also serves True without
another coefficient generator. Like the window block, it applies y then x in
double and rounds once to float32.
"""

import numpy as np


def bilinear_upsample(x, weights_y, weights_x):
    x = np.asarray(x, np.float32).astype(np.float64)
    my = np.asarray(weights_y, np.float64)
    mx = np.asarray(weights_x, np.float64)
    vertical = np.einsum("yi,ijc->yjc", my, x)
    return np.einsum("xj,yjc->yxc", mx, vertical).astype(np.float32)
