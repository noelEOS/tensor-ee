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


def _torch_reference(x, factor=2):
    torch.set_num_threads(2)
    h,w,packed = x.shape
    tx = torch.from_numpy(x).reshape(h,w,factor,factor,packed//(factor*factor)).permute(4,2,3,0,1).reshape(1,packed,h,w)
    return F.pixel_shuffle(tx,factor)[0].permute(1,2,0).numpy()


@pytest.mark.parametrize('factor', [2, 4])
@pytest.mark.parametrize('gh,gw,w', [(1,1,14),(2,3,14),(1,1,7),(2,3,7),(3,2,7)])
def test_direct_window_selection_and_mirror_preserve_coordinate_order_and_zero_bits(monkeypatch,gh,gw,w,factor):
    channels = 4
    x = np.arange(gh*gw*w*w*factor*factor*channels,dtype=np.float32).reshape(gh*w,gw*w,factor*factor*channels)
    x[0,0,0] = -0.0
    x[0,0,1] = 0.0
    local = LocalArray(_windows(x,w),w*2.5)
    subdivisions = factor*w//7

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
    actual,fine = WR.window_depth_to_space(fake,local,TokenGrid(gh*w,gw*w,2.5),channels,window=w,factor=factor)
    assert fine == TokenGrid(gh*w*factor,gw*w*factor,2.5/factor)
    expected = _torch_reference(x,factor)
    np.testing.assert_array_equal(actual.values.view(np.uint32),_windows(expected,7).view(np.uint32))
    np.testing.assert_array_equal(mirror(x,factor).view(np.uint32),expected.view(np.uint32))


@pytest.mark.parametrize('options', [dict(channels=0),dict(channels=True),dict(window=0),
    dict(window=2.5),dict(out_window=False),dict(factor=3),dict(factor=True),dict(factor=4.0),dict(factor=0),dict(window=8),
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
        for w, factor in ((7,2),(14,2),(7,4),(14,4)):
            result,fine = WR.window_depth_to_space(ee,ee.Image('source'),TokenGrid(14,14,2.5),4,window=w,factor=factor)
            from tensor_ee.converter.request import deep_recursion
            with deep_recursion():
                text = result.serialize()
            assert fine == TokenGrid(14*factor,14*factor,2.5/factor)
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


@pytest.mark.parametrize('factor', [0, 1, 3, True, 4.0])
def test_mirror_refuses_unsupported_factors(factor):
    with pytest.raises(ValueError, match='factor'):
        mirror(np.zeros((14, 14, 64), np.float32), factor=factor)


@pytest.mark.parametrize('cin,cout,bias', [(3,4,True), (3,5,False), (256,256,True)])
def test_linear_and_factor4_match_convtranspose_k4_s4(cin, cout, bias):
    from tensor_ee.blocks.window.linear import LinearConstants, linear
    from tensor_ee.core.ops import ValueOps
    from tensor_ee.weights.constants import WindowLayout
    from reference.segformer_b0.blocks import NUMPY_EE, Operand
    from tools.fixtures.core import dyadic, hwc, nchw

    # Full DPT width and token extent; deterministic dyadic inputs and weights.
    x = dyadic(104, 'x', (14, 14, cin), bits=10)
    weight = dyadic(104, 'weight', (cin, cout, 4, 4), scale=0.03125, bits=10)
    b = dyadic(104, 'bias', (cout,), scale=0.25, bits=10) if bias else None
    matrix = weight.transpose(2, 3, 1, 0).reshape(16*cout, cin)
    packed_bias = None if b is None else np.tile(b, 16)
    layout = WindowLayout(14)
    ops = ValueOps(NUMPY_EE)
    constants = LinearConstants(ops, matrix, packed_bias, layout)
    projected = linear(ops, Operand(x.reshape(196, cin)), constants, layout=layout).values
    assert projected.shape == (196, 16*cout)
    actual = mirror(projected.reshape(14, 14, 16*cout), factor=4)
    expected = hwc(F.conv_transpose2d(nchw(x), torch.from_numpy(weight),
                   None if b is None else torch.from_numpy(b), stride=4))
    max_abs = float(np.max(np.abs(actual.astype(np.float64) - expected)))
    assert max_abs <= 1e-4, max_abs


def test_factor4_fixtures_compare_whole_outputs_and_fit_the_live_budget():
    from tensor_ee.converter.request import deep_recursion
    from tools.fixtures import select
    from tools.live_fixtures import make_plan

    fixtures = select(['depth_to_space_factor4'])
    assert len(fixtures) == 2
    for fixture in fixtures:
        data = fixture.data(fixture.seed)
        actual = fixture.mirror.fn(data)['output']
        expected = fixture.reference(data)['output']
        assert actual.shape == expected.shape == (56, 56, 4)
        if fixture.tolerance == 0:
            np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        else:
            assert np.max(np.abs(actual.astype(np.float64) - expected)) <= 1e-4
    with deep_recursion():
        plan = make_plan(fixtures)
    assert sum(len(requests) for requests in plan.requests.values()) == 4
    assert not plan.violations


@pytest.mark.parametrize('window,digest', [
    (7, '63192fef677018a4f2a1c8e473bff071dfcf3f408aa679b7ae61bf612e904a4c'),
    (14, '227dfec0eae9993eb4c484eb19717840378ddf46022fd96576259915ed200bbf'),
])
def test_factor2_graphs_keep_the_base_revision_hashes(window, digest):
    import hashlib
    from tensor_ee.converter.request import deep_recursion
    from tools.live_fixtures import offline_earth_engine

    # Captured from 9c3de78, before allowing factor 4. G1 predates this block.
    with offline_earth_engine(), deep_recursion():
        result, _ = WR.window_depth_to_space(ee, ee.Image('source'), TokenGrid(14,14,2.5), 4, window=window)
        assert hashlib.sha256(result.serialize().encode()).hexdigest() == digest


@pytest.mark.parametrize('factor,window,out_window,inside', [
    (2,7,7,True), (2,14,7,True), (4,14,7,True),
    (4,7,7,False), (4,14,8,False), (3,14,7,False),
])
def test_validated_record_adds_only_the_live_factor4_window(factor, window, out_window, inside):
    from tensor_ee.core.metadata import check_settings

    metadata = WR.METADATA['window_depth_to_space']
    assert metadata.status.startswith('Ready — factor 2; windows 14, 7 to window 7; ')
    check = check_settings(metadata, dict(factor=factor, window=window, out_window=out_window))
    assert check.inside is inside
