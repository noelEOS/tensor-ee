"""Offline full-crop composition fixture with compact structured random weights.

The full224 forward is built and independently evaluated locally. The harness
reads every fourth logit location, 56x56, plus the32x32 periodic input seed. The
separate whole-model tests compare all224x224 logits. This selection is offline
only and is never submitted as a whole-model live request.
"""
from __future__ import annotations
import math
import numpy as np
from .core import Fixture,FixtureData,MirrorSpec,Readout,dyadic,inline_raster


def _config():
    from tensor_ee.models.prithvi_sen1floods11.configuration import PrithviConfig
    return PrithviConfig(width=24,heads=3,depth=12,decoder=16)


def _data(seed):
    from tensor_ee.models.prithvi_sen1floods11.configuration import parameter_shapes
    weights={}
    for name,shape in parameter_shapes(_config()).items():
        if len(shape)==1:
            tile=(min(8,shape[0]),)
            offset=1 if name.endswith('.weight') or name.endswith('running_var') else 0
            scale=.25 if offset else .03125
        else:
            rows=shape[0]; cols=math.prod(shape[1:])
            tile=(math.gcd(rows,6),math.gcd(cols,8))
            offset=0
            scale=2**-5 if cols>1000 else 2**-3
            if name in ('backbone.pos_embed','backbone.cls_token'):
                scale=.03125
        weights[name]=dyadic(seed,name,tile,scale=scale,offset=offset,bits=10)
    return FixtureData({'x':dyadic(seed,'input',(32,32,6),bits=10)},weights)


def _expanded(data):
    from tensor_ee.models.prithvi_sen1floods11.configuration import parameter_shapes
    state={}
    for name,shape in parameter_shapes(_config()).items():
        seed=data.weights[name]
        if len(shape)==1:
            state[name]=np.tile(seed,shape[0]//seed.shape[0])
        else:
            target=(shape[0],math.prod(shape[1:]))
            matrix=np.tile(seed,(target[0]//seed.shape[0],target[1]//seed.shape[1]))
            state[name]=matrix.reshape(shape)
    return np.tile(data.inputs['x'],(7,7,1)),state


def _server_state(ee,data):
    from tensor_ee.core.precision import encode_float32
    from tensor_ee.core.tensor import ServerArray
    from tensor_ee.models.prithvi_sen1floods11.configuration import parameter_shapes
    state={}
    for name,shape in parameter_shapes(_config()).items():
        seed=data.weights[name]
        value=ee.Array(encode_float32(seed)).float()
        if len(shape)==1:
            value=value.repeat(0,shape[0]//seed.shape[0])
            target=shape
        else:
            target=(shape[0],math.prod(shape[1:]))
            value=value.repeat(0,target[0]//seed.shape[0]).repeat(1,target[1]//seed.shape[1])
        state[name]=ServerArray(value,target)
    return state


def _build(ee,data):
    from tensor_ee.core.grid import TokenGrid,FeatureGrid
    from tensor_ee.core.tensor import Raster
    from tensor_ee.models.prithvi_sen1floods11.model import constants,forward
    from tensor_ee.converter.request import RECURSION_LIMIT
    import sys
    # Serialization depth only, matching the library sizer; no execution budget changes.
    sys.setrecursionlimit(max(sys.getrecursionlimit(),RECURSION_LIMIT))
    grid=TokenGrid(224,224,scale=1.0)
    seed_grid=TokenGrid(32,32,scale=1.0)
    names=tuple(f'b{i}' for i in range(6))
    source=inline_raster(ee,grid.projection(ee),data.inputs['x'],names,periodic=True)
    c=constants(ee,_server_state(ee,data),config=_config())
    result=forward(ee,Raster(source,names,grid),c)['logits']
    # Centers2.5,6.5,..222.5 coincide exactly with fine pixel2,6,..222.
    probes=FeatureGrid(grid.crs,.5,-.5,4.,56,56)
    return [Readout('logits',result.image,2,probes.pixel_grid()),
            Readout('input',source,6,seed_grid.pixel_grid(),'input','x')]


def _reference(data):
    from reference.prithvi_sen1floods11.reference import forward_torch
    image,state=_expanded(data)
    output=forward_torch(image,state,heads=3,depth=12)
    return {'logits':np.ascontiguousarray(output[2::4,2::4],dtype=np.float32)}


def _mirror(data):
    from reference.prithvi_sen1floods11.execution import execute, read
    image,state=_expanded(data)
    output=read(execute(image,state,_config(),asset_shapes=True)['logits'])
    return {'logits':np.ascontiguousarray(output[2::4,2::4],dtype=np.float32)}


def _cost(data):
    from tensor_ee.models.prithvi_sen1floods11.planning import constant_costs
    return {'logical_model_constants':constant_costs(_config())['total']}


def fixtures():
    from tensor_ee.models.prithvi_sen1floods11.configuration import PrithviConfig
    from tensor_ee.blocks.window.merge import concat_rows
    from tensor_ee.weights.constants import WindowLayout
    return [Fixture(id='window/concat_rows/prithvi-crop-logits',lane='window',block='concat_rows',
        settings={'window':14,'rows':(1,196),'channels':8,'width':24,'heads':3,'depth':12,'decoder':16},
        policy='none',shapes={'input':(224,224,6),'logits':(224,224,2),'readout':(56,56,2)},
        seed=8217,data=_data,build=_build,reference=_reference,reference_source='plain torch source-order Prithvi reference',
        mirror=MirrorSpec(_mirror,False,'tests/reference/prithvi_sen1floods11/execution.py',bound=1e-4),
        requests=2,constants=_cost,history='docs/api.md#prithvi-sen1floods11',
        parts=('patch_embed','linear','layer_norm','naive_attention','gelu','add','window_depth_to_space','layer_norm2d','conv3x3_window','batch_norm','relu','conv1x1','head_dense'),
        accepts=(concat_rows,PrithviConfig,WindowLayout),window=14,
        notes='Offline only: full224crop graph, structured random repeated dyadic coefficients; probe every fourth output pixel, all pixels in model tests. No full-model live run.')]


SELECTIONS = {'prithvi_model': lambda: tuple(f.id for f in fixtures())}
