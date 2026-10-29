"""Direct child-window selection preserves order and float32 bit patterns."""
from types import SimpleNamespace

import ee
from ee.apitestcase import ApiTestCase
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from tensor_ee.blocks.window import resample as WR
from tensor_ee.core.grid import TokenGrid
from reference.window_depth_to_space import window_depth_to_space as mirror


class Scalar:
    """Integer coordinate expression, independent of tensor values."""
    def __init__(self, values):
        self.values = np.asarray(values)

    def mod(self, n):
        return Scalar(np.fmod(self.values,n))

    def add(self, n):
        return Scalar(self.values+n)

    def multiply(self, n):
        return Scalar(self.values*n)

    def toInt(self):
        return Scalar(self.values.astype(np.int32))


class LocalArray:
    """Image array axes after its two coarse-grid axes; nearest reproject/slice."""
    def __init__(self, values, scale):
        self.values = values
        self.scale = scale

    def arrayTranspose(self, a, b):
        return LocalArray(self.values.swapaxes(a+2,b+2),self.scale)

    def reproject(self, projection):
        factor = self.scale / projection.transform[0]
        assert factor == int(factor)
        values = self.values.repeat(int(factor),axis=0).repeat(int(factor),axis=1)
        return LocalArray(values,projection.transform[0])

    def arraySlice(self, axis, start, end):
        gh,gw = self.values.shape[:2]
        def selected(y,x):
            index = [slice(None)]*(self.values.ndim-2)
            index[axis] = slice(int(start.values[y,x]),int(end.values[y,x]))
            return self.values[y,x][tuple(index)]
        values = np.stack([np.stack([selected(y,x) for x in range(gw)]) for y in range(gh)])
        return LocalArray(values,self.scale)

    def updateMask(self, mask):
        assert mask.shape == self.values.shape[:2]
        assert mask.all()
        return self


def _windows(x, w):
    h,width,c = x.shape
    return x.reshape(h//w,w,width//w,w,c).transpose(0,2,1,3,4).reshape(h//w,width//w,w*w,c)


def _torch_reference(x):
    torch.set_num_threads(2)
    h,w,packed = x.shape
    tx = torch.from_numpy(x).reshape(h,w,2,2,packed//4).permute(4,2,3,0,1).reshape(1,packed,h,w)
    return F.pixel_shuffle(tx,2)[0].permute(1,2,0).numpy()


@pytest.mark.parametrize('gh,gw,w', [(1,1,14),(2,3,14),(1,1,7),(2,3,7),(3,2,7)])
def test_direct_window_selection_and_mirror_preserve_coordinate_order_and_zero_bits(monkeypatch,gh,gw,w):
    channels = 4
    x = np.arange(gh*gw*w*w*4*channels,dtype=np.float32).reshape(gh*w,gw*w,4*channels)
    x[0,0,0] = -0.0
    x[0,0,1] = 0.0
    local = LocalArray(_windows(x,w),w*2.5)
    subdivisions = 2*w//7

    class Coordinates:
        def floor(self):
            return self

        def select(self,axis):
            return Scalar(np.indices((gh*subdivisions,gw*subdivisions))[0 if axis=='y' else 1])

    fake = SimpleNamespace(
        Projection=lambda crs,transform:SimpleNamespace(crs=crs,transform=transform),
        Image=SimpleNamespace(pixelCoordinates=lambda projection:Coordinates()))

    def reshape(ee,image,shape):
        return LocalArray(image.values.reshape(*image.values.shape[:2],*shape),image.scale)

    monkeypatch.setattr(WR,'reshape',reshape)
    monkeypatch.setattr(WR,'domain_mask',lambda ee,grid:np.ones((grid.height,grid.width),dtype=bool))
    actual,fine = WR.window_depth_to_space(fake,local,TokenGrid(gh*w,gw*w,2.5),channels,window=w)
    assert fine == TokenGrid(gh*w*2,gw*w*2,1.25)
    expected = _torch_reference(x)
    np.testing.assert_array_equal(actual.values.view(np.uint32),_windows(expected,7).view(np.uint32))
    np.testing.assert_array_equal(mirror(x).view(np.uint32),expected.view(np.uint32))


@pytest.mark.parametrize('options', [dict(channels=0),dict(channels=True),dict(window=0),
    dict(window=2.5),dict(out_window=False),dict(factor=3),dict(factor=True),dict(window=8),
    dict(out_window=3),dict(out_window=1),dict(window=1),dict(out_window=14)])
def test_unsupported_dimensions_are_refused_before_graph_construction(options):
    kwargs = dict(channels=4,window=7,out_window=7,factor=2)
    kwargs.update(options)
    with pytest.raises(ValueError):
        WR.window_depth_to_space(None,None,TokenGrid(14,14,2.5),**kwargs)


def test_graph_constructs_without_tensor_value_arithmetic_or_dense_bands():
    case = ApiTestCase()
    case.setUp()
    try:
        for w in (7,14):
            result,fine = WR.window_depth_to_space(ee,ee.Image('source'),TokenGrid(14,14,2.5),4,window=w)
            text = result.serialize()
            assert fine == TokenGrid(28,28,1.25)
            assert 'Image.arrayTranspose' in text
            for forbidden in ('Image.toFloat','Image.toDouble','Image.matrixMultiply',
                              'Image.arrayFlatten','Image.toArray','Image.changeProj'):
                assert forbidden not in text
            assert text.count('Image.reproject') == 2
            assert 'Image.updateMask' in text
    finally:
        case.tearDown()


@pytest.mark.parametrize('grid',[None,object()])
def test_non_origin_grids_are_refused(grid):
    with pytest.raises(ValueError,match='TokenGrid'):
        WR.window_depth_to_space(None,None,grid,4)
