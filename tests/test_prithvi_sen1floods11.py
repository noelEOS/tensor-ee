"""Crop wiring, source-state schema, asset shapes, and independent arithmetic."""
import numpy as np
import pytest
import ee

from tensor_ee.core.grid import Grid
from tensor_ee.core.tensor import Raster,ServerArray
from tensor_ee.models.prithvi_sen1floods11.configuration import PrithviConfig,parameter_shapes
from tensor_ee.models.prithvi_sen1floods11.model import constants,compose,forward,validate_parameters,_reshape,_upsampling_parameters
from tensor_ee.models.prithvi_sen1floods11.planning import constant_costs,cut_proposal
from reference.prithvi_sen1floods11.reference import NumpyBackend,expected_shapes,random_state,forward_torch
from tools.live_fixtures import offline_earth_engine


REDUCED=PrithviConfig(width=24,heads=3,depth=12,decoder=16)


def _input():
    return np.random.default_rng(81).normal(0,.5,(224,224,6)).astype(np.float32)


def _state(config=REDUCED):
    return random_state(config.width,config.depth,config.decoder)


def test_state_schema_is_independent_and_matches_source():
    assert parameter_shapes(REDUCED)==expected_shapes(24,12,16)
    assert 'decode_head.convs.0.conv.bias' not in parameter_shapes()
    assert constant_costs()['float_state_values']==97_602_818


@pytest.mark.parametrize('options',[{'width':False},{'width':25,'heads':3},{'depth':0},{'decoder':-1},{'heads':1.5}])
def test_configuration_refuses_unsupported_dimensions(options):
    with pytest.raises(ValueError):
        PrithviConfig(**options)


def test_shared_composition_matches_independent_torch():
    state=_state(); image=_input()
    actual=compose(NumpyBackend(state),image,REDUCED)
    expected=forward_torch(image,state,heads=3,depth=12,readouts=True)
    for name in ('encoder','class','logits'):
        mirror=(actual['norm'][0] if name=='encoder' else actual['norm'][1]) if name!='logits' else actual['logits']
        error=float(np.max(np.abs(mirror.astype(np.float64)-expected[name].astype(np.float64))))
        assert error<=1e-4,(name,error)
    for side in (28,56,112,224):
        name=f'upsample{side}'
        assert np.max(np.abs(actual[name].astype(np.float64)-expected[name].astype(np.float64)))<=1e-4


def test_saved_class_and_spatial_resume_preserves_all_later_blocks():
    state=_state(); backend=NumpyBackend(state)
    direct=compose(backend,_input(),REDUCED)
    resumed=compose(backend,direct['block5'],REDUCED,start='block5')
    np.testing.assert_array_equal(resumed['logits'],direct['logits'])
    spatial,cls=direct['block5']
    # Constant shifts would disappear in LN; change one class channel instead.
    altered=cls.copy(); altered[0,0]+=np.float32(1)
    changed=compose(backend,(spatial,altered),REDUCED,start='block5')
    assert np.any(changed['logits']!=direct['logits'])


def test_parameters_refuse_bad_shape_dtype_missing_nonfinite_and_variance():
    state=_state()
    cases=[]
    missing=state.copy(); missing.pop('backbone.cls_token'); cases.append(missing)
    malformed=state.copy(); malformed['backbone.cls_token']=np.zeros((24,),np.float32); cases.append(malformed)
    double=state.copy(); double['backbone.cls_token']=state['backbone.cls_token'].astype(np.float64); cases.append(double)
    nonfinite=state.copy(); nonfinite['backbone.cls_token']=np.full((1,1,24),np.nan,np.float32); cases.append(nonfinite)
    variance=state.copy(); variance['decode_head.convs.0.bn.running_var']=-np.ones(16,np.float32); cases.append(variance)
    for case in cases:
        with pytest.raises(ValueError):
            validate_parameters(case,REDUCED)


def _server_state(config):
    state={}
    for name,shape in parameter_shapes(config).items():
        # Distinct table references preserve names/shapes; never read a remote value.
        values=ee.FeatureCollection('projects/example/assets/'+name.replace('.','_')).aggregate_array('values').flatten()
        asset_shape=(shape[0],int(np.prod(shape[1:])))
        state[name]=ServerArray(ee.Array(values).reshape(list(asset_shape)).float(),asset_shape)
    return state


def test_asset_backed_model_graph_has_correct_geometry_and_refuses_partial_crops():
    with offline_earth_engine() as attempts:
        c=constants(ee,_server_state(REDUCED),config=REDUCED)
        grid=Grid('EPSG:3857',(10,0,100,0,-10,200),224,224)
        raster=Raster(ee.Image.constant([0]*6).toFloat().rename([f'b{i}' for i in range(6)]),tuple(f'b{i}' for i in range(6)),grid)
        outputs=forward(ee,raster,c)
        assert outputs['norm'][0].window==14
        assert outputs['norm'][1].window==1
        assert outputs['norm'][0].grid.coarse(14)==outputs['norm'][1].grid
        assert outputs['upsample28'].window==7
        assert outputs['upsample224'].grid.height==224
        assert outputs['logits'].grid==grid
        assert outputs['logits'].bands==('c0','c1')
        assert all(key in c.backend.linears for key in [('backbone.blocks.0.attn.qkv',196),('backbone.blocks.0.attn.qkv',1)])
        region=Raster(raster.image,raster.bands,Grid(grid.crs,grid.transform,448,224))
        assert forward(ee,region,c)['logits'].grid==region.grid      # two crops side by side
        for width,height in ((300,224),(224,200),(112,112)):
            partial=Raster(raster.image,raster.bands,Grid(grid.crs,grid.transform,width,height))
            with pytest.raises(ValueError,match='complete 224'):
                forward(ee,partial,c)
    assert not attempts


def test_cut_proposal_is_an_estimate_with_paired_encoder_boundaries():
    plan=cut_proposal()
    assert not plan['measured']
    assert not plan['whole_model_constant_limit_satisfied']
    assert all(stage['constant_limit_satisfied'] and stage['band_limit_satisfied'] for stage in plan['stages'])
    assert plan['stages'][1]['exports']==2
    assert plan['stages'][1]['request_bytes'] is None
    assert constant_costs()['total']==122_783_330



class _Array:
    """Only the server-side coefficient layout operations, evaluated eagerly."""
    def __init__(self,value):
        self.value=np.asarray(value)
    def reshape(self,shape):
        return _Array(self.value.reshape(shape))
    def transpose(self,axis1=None,axis2=None):
        return _Array(self.value.T if axis1 is None else self.value.swapaxes(axis1,axis2))
    def repeat(self,axis,count):
        repeats=[1]*self.value.ndim; repeats[axis]=count
        return _Array(np.tile(self.value,repeats))


@pytest.mark.parametrize('asset',[False,True])
def test_deconvolution_constants_native_and_flattened_assets_move_each_kernel_position(asset):
    weight=np.arange(3*5*2*2,dtype=np.float32).reshape(3,5,2,2)
    bias=np.arange(5,dtype=np.float32)-2
    x=np.arange(2*3*3,dtype=np.float32).reshape(2,3,3)
    if asset:
        flat=ServerArray(_Array(weight.reshape(3,20)),(3,20))
        source=_reshape(flat,weight.shape)
        b=ServerArray(_Array(bias.reshape(1,5)),(1,5))
        source_bias=_reshape(b,(5,))
        packed,repeated=_upsampling_parameters(source,source_bias)
        packed,repeated=packed.array.value,repeated.array.value
    else:
        packed,repeated=_upsampling_parameters(weight,bias)
    projected=x.astype(np.float64)@packed.astype(np.float64).T+repeated
    moved=projected.reshape(2,3,2,2,5).transpose(0,2,1,3,4).reshape(4,6,5)
    independent=np.empty((4,6,5),np.float64)
    for dy in (0,1):
        for dx in (0,1):
            independent[dy::2,dx::2]=x.astype(np.float64)@weight[:,:,dy,dx].astype(np.float64)+bias
    np.testing.assert_array_equal(moved,independent)
