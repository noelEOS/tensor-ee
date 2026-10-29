"""Plain PyTorch reference and independent NumPy primitive backend for one crop.

Position values are supplied in the state; no positional generator is included.
The NumPy backend models double-round-once projections and round-each-operation
attention. It uses float64 matrix products with float32 operation boundaries.
"""
from __future__ import annotations
import math
import numpy as np


def expected_shapes(width=24, depth=12, decoder_width=16):
    shapes = {
        'backbone.patch_embed.proj.weight': (width,6,1,16,16),
        'backbone.patch_embed.proj.bias': (width,),
        'backbone.pos_embed': (1,197,width),
        'backbone.cls_token': (1,1,width),
    }
    def affine(name,channels):
        shapes[name+'.weight']=(channels,); shapes[name+'.bias']=(channels,)
    for i in range(depth):
        base=f'backbone.blocks.{i}'
        affine(base+'.norm1',width); affine(base+'.norm2',width)
        for name,cin,cout in [('attn.qkv',width,3*width),('attn.proj',width,width),('mlp.fc1',width,4*width),('mlp.fc2',4*width,width)]:
            shapes[base+'.'+name+'.weight']=(cout,cin)
            shapes[base+'.'+name+'.bias']=(cout,)
    affine('backbone.norm',width)
    for name in ['neck.fpn1','neck.fpn2']:
        for index in [0,3]:
            shapes[f'{name}.{index}.weight']=(width,width,2,2)
            shapes[f'{name}.{index}.bias']=(width,)
        affine(name+'.1.ln',width)
    shapes['decode_head.convs.0.conv.weight']=(decoder_width,width,3,3)
    bn='decode_head.convs.0.bn'
    affine(bn,decoder_width)
    shapes[bn+'.running_mean']=(decoder_width,); shapes[bn+'.running_var']=(decoder_width,)
    shapes['decode_head.conv_seg.weight']=(2,decoder_width,1,1)
    shapes['decode_head.conv_seg.bias']=(2,)
    return shapes


def random_state(width=24,depth=12,decoder_width=16,seed=294):
    rng=np.random.default_rng(seed); state={}
    for name,shape in expected_shapes(width,depth,decoder_width).items():
        if name.endswith('running_var'):
            value=rng.uniform(.7,1.3,size=shape)
        elif name.endswith('running_mean'):
            value=rng.normal(0,.05,size=shape)
        elif name.endswith('.weight') and len(shape)==1:
            value=rng.uniform(.8,1.2,size=shape)
        elif name.endswith('.bias'):
            value=rng.normal(0,.01,size=shape)
        elif name in ['backbone.pos_embed','backbone.cls_token']:
            value=rng.normal(0,.02,size=shape)
        else:
            fan_in=np.prod(shape[1:])
            value=rng.normal(0,1/math.sqrt(fan_in),size=shape)
        state[name]=np.asarray(value,dtype=np.float32)
    return state


def forward_torch(x,state,*,heads=3,depth=12,readouts=False):
    """Independent timm-0.4.12 arithmetic, native Conv3d/ConvTranspose2d; HWC IO."""
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(2)
    p={name:torch.from_numpy(value) for name,value in state.items()}
    result={}
    def ln(value,name,eps):
        return F.layer_norm(value,(value.shape[-1],),p[name+'.weight'],p[name+'.bias'],eps)
    def linear(value,name):
        return F.linear(value,p[name+'.weight'],p[name+'.bias'])
    with torch.no_grad():
        image=torch.from_numpy(x).permute(2,0,1)[None,:,None]
        z=F.conv3d(image,p['backbone.patch_embed.proj.weight'],p['backbone.patch_embed.proj.bias'],stride=(1,16,16))
        z=z.flatten(2).transpose(1,2)
        z=z+p['backbone.pos_embed'][:,1:]
        cls=p['backbone.cls_token']+p['backbone.pos_embed'][:,:1]
        z=torch.cat([cls,z],dim=1)
        for i in range(depth):
            name=f'backbone.blocks.{i}'
            normalized=ln(z,name+'.norm1',1e-5)
            batch,rows,width=normalized.shape
            qkv=linear(normalized,name+'.attn.qkv').reshape(batch,rows,3,heads,width//heads).permute(2,0,3,1,4)
            q,k,v=qkv.unbind(0)
            scores=(q@k.transpose(-2,-1))*((width//heads)**-.5)
            values=(scores.softmax(dim=-1)@v).transpose(1,2).reshape(batch,rows,width)
            z=z+linear(values,name+'.attn.proj')
            z=z+linear(F.gelu(linear(ln(z,name+'.norm2',1e-5),name+'.mlp.fc1'),approximate='none'),name+'.mlp.fc2')
        z=ln(z,'backbone.norm',1e-5)
        result['encoder']=z[0,1:].reshape(14,14,-1).numpy().copy()
        result['class']=z[0,:1].numpy().copy()
        z=z[:,1:].transpose(1,2).reshape(1,-1,14,14)
        for name in ['neck.fpn1','neck.fpn2']:
            for index in [0,3]:
                z=F.conv_transpose2d(z,p[f'{name}.{index}.weight'],p[f'{name}.{index}.bias'],stride=2)
                if index==0:
                    z=ln(z.permute(0,2,3,1),name+'.1.ln',1e-6).permute(0,3,1,2)
                    z=F.gelu(z,approximate='none')
                result[f'upsample{z.shape[-1]}']=z[0].permute(1,2,0).numpy().copy() if readouts else None
        z=F.conv2d(z,p['decode_head.convs.0.conv.weight'],padding=1)
        bn='decode_head.convs.0.bn'
        z=F.batch_norm(z,p[bn+'.running_mean'],p[bn+'.running_var'],p[bn+'.weight'],p[bn+'.bias'],training=False,eps=1e-5)
        z=F.relu(z)
        z=F.conv2d(z,p['decode_head.conv_seg.weight'],p['decode_head.conv_seg.bias'])
        z=F.interpolate(z,size=(224,224),mode='bilinear',align_corners=False)
        logits=z[0].permute(1,2,0).numpy().copy()
    result['logits']=logits
    return result if readouts else logits


class PrimitiveBackend:
    """NumPy numerical hypotheses for the operations the model composes."""
    def patch(self,x,weight,bias):
        width=weight.shape[0]
        columns=x.reshape(14,16,14,16,6).transpose(0,2,4,1,3).reshape(14,14,1536)
        return self.linear(columns,weight.reshape(width,1536),bias)
    patch_input=patch

    def position(self,x,pos,cls):
        width=x.shape[-1]
        spatial=self.add(x,pos[0,1:].reshape(14,14,width))
        token=self.add(cls.reshape(1,width),pos[0,:1].reshape(1,width))
        return spatial,token

    def norm(self,x,gamma,beta,eps=1e-5):
        values=x.astype(np.float64)
        centered=values-values.mean(axis=-1,keepdims=True)
        var=np.mean(centered*centered,axis=-1,keepdims=True)
        return (centered/np.sqrt(var+eps)*gamma.astype(np.float64)+beta.astype(np.float64)).astype(np.float32)

    def linear(self,x,weight,bias=None):
        product=x.astype(np.float64)@weight.astype(np.float64).T
        return (product if bias is None else product+bias.astype(np.float64)).astype(np.float32)

    def attention(self,spatial,class_token,heads):
        channels=spatial.shape[-1]//3; head_dim=channels//heads
        qkv=np.concatenate([class_token.reshape(1,3*channels),spatial.reshape(-1,3*channels)],axis=0)
        parts=qkv.reshape(-1,3,heads,head_dim).transpose(1,2,0,3)
        qs,ks,vs=parts
        outputs=[]
        for q,k,v in zip(qs,ks,vs):
            scores=(q.astype(np.float64)@k.astype(np.float64).T).astype(np.float32)
            scaled=(scores.astype(np.float64)*np.float32(head_dim**-.5)).astype(np.float32)
            shifted=(scaled.astype(np.float64)-scaled.max(axis=-1,keepdims=True).astype(np.float64)).astype(np.float32)
            exp=np.exp(shifted.astype(np.float64)).astype(np.float32)
            total=exp.astype(np.float64).sum(axis=-1,keepdims=True).astype(np.float32)
            probabilities=(exp.astype(np.float64)/total.astype(np.float64)).astype(np.float32)
            outputs.append((probabilities.astype(np.float64)@v.astype(np.float64)).astype(np.float32))
        out=np.concatenate(outputs,axis=-1)
        return out[1:].reshape(*spatial.shape[:-1],channels),out[:1]

    def shuffle(self,x):
        h,w,packed=x.shape; channels=packed//4
        return x.reshape(h,w,2,2,channels).transpose(0,2,1,3,4).reshape(2*h,2*w,channels)

    def upsample(self,x,weight,bias,window=7):
        cin,cout=weight.shape[:2]
        packed=weight.transpose(0,2,3,1).reshape(cin,4*cout).T
        return self.shuffle(self.linear(x,packed,np.tile(bias,4)))

    def conv3x3(self,x,weight):
        h,w,_=x.shape; cout=weight.shape[0]
        padded=np.pad(x,((1,1),(1,1),(0,0)))
        kernel=weight.transpose(2,3,1,0).reshape(-1,cout).astype(np.float64)
        out=np.empty((h,w,cout),np.float32)
        # Form only one W7 row of windows at a time; avoid a 2.8 GB global im2col.
        for y in range(0,h,7):
            yh=min(y+7,h)
            rows=np.stack([padded[y+dy:yh+dy,dx:dx+w] for dy in range(3) for dx in range(3)],axis=-2).reshape((yh-y)*w,-1)
            out[y:yh]=(rows.astype(np.float64)@kernel).reshape(yh-y,w,cout).astype(np.float32)
        return out

    def batch_norm(self,x,gamma,beta,mean,var,eps=1e-5):
        values=(x.astype(np.float64)-mean.astype(np.float64))/np.sqrt(var.astype(np.float64)+eps)
        return (values*gamma.astype(np.float64)+beta.astype(np.float64)).astype(np.float32)

    def gelu(self,x):
        values=x.astype(np.float64)
        erf=np.vectorize(math.erf,otypes=[float])(values/math.sqrt(2))
        return (values*.5*(1+erf)).astype(np.float32)

    def add(self,a,b):
        return (a.astype(np.float64)+b.astype(np.float64)).astype(np.float32)

    def relu(self,x):
        return np.maximum(x,np.float32(0))

    def head_dense(self,x):
        return x



class NumpyBackend:
    """State-key adapter for the actual library composition, with NumPy primitives."""
    def __init__(self,parameters):
        self.parameters=parameters
        self.numeric=PrimitiveBackend()

    def patch(self,x):
        p=self.parameters
        return self.numeric.patch(x,p['backbone.patch_embed.proj.weight'],p['backbone.patch_embed.proj.bias'])

    def position(self,x):
        p=self.parameters
        return self.numeric.position(x,p['backbone.pos_embed'],p['backbone.cls_token'])

    def norm(self,x,name,eps):
        p=self.parameters
        return self.numeric.norm(x,p[name+'.weight'],p[name+'.bias'],eps)

    def linear(self,x,name):
        p=self.parameters
        weight=p[name+'.weight']
        if name=='decode_head.conv_seg':
            weight=weight.reshape(weight.shape[0],-1)
        return self.numeric.linear(x,weight,p[name+'.bias'])

    def attention(self,spatial,cls,heads):
        return self.numeric.attention(spatial,cls,heads)

    def upsample(self,x,name):
        p=self.parameters
        return self.numeric.upsample(x,p[name+'.weight'],p[name+'.bias'])

    def conv3x3(self,x,name):
        return self.numeric.conv3x3(x,self.parameters[name+'.weight'])

    def batch_norm(self,x,name):
        p=self.parameters
        return self.numeric.batch_norm(x,p[name+'.weight'],p[name+'.bias'],p[name+'.running_mean'],p[name+'.running_var'])

    def gelu(self,x):
        return self.numeric.gelu(x)

    def relu(self,x):
        return self.numeric.relu(x)

    def add(self,a,b):
        return self.numeric.add(a,b)

    def head_dense(self,x):
        return x
