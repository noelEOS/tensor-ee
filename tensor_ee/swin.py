"""Explicit, fixed 224² Swin-T/UPerNet execution in three deferred stages.

Import this module explicitly. Prepared values already include the caller's
normalization and missing-value fill; validity travels separately. Saved stages
are aligned Rasters supplied by the caller. No cloud inspection, initialization,
export, download, checkpoint fetch, or experiment registry is implicit here.

Built only from the library: the Swin-T encoder and UPerNet decoder composites of
``tensor_ee.models.swin_t224`` on the core, layout, weight and block modules. No research
folder is imported.
"""
from __future__ import annotations

import numpy as np

from .blocks.window import patch as PE
from .core.errors import UnsupportedOperation
from .core.grid import FeatureGrid, Grid, TokenGrid, check_model_grid
from .core.ops import BlockImageOps
from .core.tensor import Feature, Raster, band_names
from .layouts.raster import embedded_raster, stride_sample_phase
from .layouts.shift import domain_mask
from .layouts.window import depth_to_space, from_dense
from .models.swin_t224 import decoder as D
from .models.swin_t224 import encoder as E
from .models.swin_t224.configuration import CHANNELS, STRIDES
from .weights.constants import ServerArray
from ._swin_parameters import (
    SwinTParameters, load_swin_t224_model, load_swin_t224_parameters,
)

__all__ = ["ServerArray", "SwinTParameters", "load_swin_t224_parameters",
           "load_swin_t224_model", "SwinT224Adapter"]

_SKIP_BANDS = tuple([f"c{k}_{c}" for k, channels in enumerate(CHANNELS[:3], 1)
                     for c in range(channels)] + ["valid_fraction"])
_C4_BANDS = tuple(f"c{c}" for c in range(768))


def _names(names, label):
    if (not isinstance(names, tuple) or len(names) != 2
        or any(not isinstance(n,str) or not n for n in names) or len(set(names)) != 2):
        raise UnsupportedOperation(f"{label} must be exactly two unique nonempty band names")
    return names


def _parameters(parameters):
    if not isinstance(parameters, SwinTParameters):
        raise UnsupportedOperation("parameters must be validated SwinTParameters")
    parameters.validate()


def _input_grid(grid):
    if not isinstance(grid,Grid) or (grid.height,grid.width) != (224,224):
        raise UnsupportedOperation("Swin-T224 requires an explicit 224 x 224 Grid")
    a,b,_,d,e,_ = grid.transform
    if b != 0 or d != 0 or a != -e:
        raise UnsupportedOperation("Swin-T224 requires north-up square pixels without shear")


def _metadata(raster, bands, grid, label):
    if not isinstance(raster,Raster) or tuple(raster.bands) != bands or raster.grid != grid:
        raise UnsupportedOperation(f"{label}: grid/band order differs from the fixed stage contract")


def _feature_grid(grid):
    a,_,x,_,_,y = grid.transform
    return FeatureGrid(grid.crs,x,y,a,grid.height,grid.width)


def _model_grid(stage=0):
    # The literal grid serializes as the stage graphs expect; check_model_grid asserts, client-side
    # only, that 2.5 m x patch 4 is the encoder's default 10 m token grid.
    return check_model_grid(TokenGrid(224,224,2.5)).coarse(1 if stage == 0 else STRIDES[stage-1])


def _to_model(ee,image,grid,stage=0):
    coordinate_grid = grid if stage == 0 else grid.resized(STRIDES[stage-1],224//STRIDES[stage-1],224//STRIDES[stage-1])
    source = coordinate_grid.projection()
    return image.reproject(source).changeProj(source,_model_grid(stage).projection(ee))


def _to_coordinate(ee,image,input_grid,stage=0):
    target = input_grid if stage == 0 else input_grid.resized(STRIDES[stage-1],224//STRIDES[stage-1],224//STRIDES[stage-1])
    source = _model_grid(stage).projection(ee)
    return image.reproject(source).changeProj(source,target.projection()).updateMask(target.mask())


def _local_image(ee,raster,*,validity=False):
    """Validate arrays when locally inspectable; deferred images assert the same rules.

    Array fixtures use CHW (or batch-one NCHW). This never scans a remote image.
    Masked validity is zero; prepared arrays must be dense finite float32.
    """
    if not isinstance(raster.image,np.ndarray):
        image = ee.Image(raster.image).select(list(raster.bands))
        if validity:
            return image.unmask(0,False).toFloat().updateMask(raster.grid.mask())
        return image.toFloat().updateMask(raster.grid.mask())
    values = np.asanyarray(raster.image)
    if values.shape == (1,2,224,224):
        values = values[0]
    if values.shape != (2,224,224):
        raise UnsupportedOperation("local prepared/validity fixtures must be [2,224,224] or [1,2,224,224]")
    if validity:
        data = np.ma.getdata(values)
        mask = np.ma.getmaskarray(values)
        if not np.isfinite(data[~mask]).all() or not np.isin(data[~mask],[0,1]).all():
            raise UnsupportedOperation("local validity must be finite binary values per channel")
        values = np.ma.filled(values,0).astype(np.float32)
    elif values.dtype != np.float32 or np.ma.getmaskarray(values).any() or not np.isfinite(values).all():
        raise UnsupportedOperation("local prepared input must be dense finite float32")
    return embedded_raster(ee,_feature_grid(raster.grid),values.transpose(1,2,0)).rename(list(raster.bands))


def _feature_image(ee,skips,input_grid,k):
    image = skips.image.select([f"c{k}_{c}" for c in range(CHANNELS[k-1])],band_names(CHANNELS[k-1]))
    c1 = _feature_grid(input_grid.resized(4,56,56))
    if k == 1:
        return image.reproject(c1.projection(ee))
    return stride_sample_phase(ee,image,c1,STRIDES[k-1]//4)


def _windows(ee,image,input_grid,k):
    tokens = _model_grid(k)
    return from_dense(ee,tokens,_to_model(ee,image,input_grid,k),CHANNELS[k-1]),tokens


class SwinT224Adapter:
    """Fixed two-channel/two-class trained architecture, explicit execution context."""

    def __init__(self, *, input_bands: tuple[str,str], output_bands: tuple[str,str] = ("logit_0","logit_1")):
        self.input_bands = _names(input_bands,"input_bands")
        self.output_bands = _names(output_bands,"output_bands")

    def skips(self, prepared: Raster, *, validity: Raster, parameters: SwinTParameters) -> Raster:
        import ee
        if not isinstance(prepared, Raster):
            raise UnsupportedOperation("prepared must be a Raster")
        _input_grid(prepared.grid)
        _metadata(prepared,self.input_bands,prepared.grid,"prepared")
        _metadata(validity,self.input_bands,prepared.grid,"validity")
        _parameters(parameters)
        ops = BlockImageOps(ee)
        # Canonical names preserve caller channel positions, not sensor semantics.
        x = _to_model(ee,_local_image(ee,prepared).rename(list(PE.BANDS)),prepared.grid)
        valid = _to_model(ee,_local_image(ee,validity,validity=True).rename(list(PE.BANDS)),prepared.grid)
        pe = PE.patch_embed(ops,x,_model_grid(),PE.patch_embed_constants(ops,parameters.patch_embed))
        c = E.EncoderConstants(
            blocks=[E.stage_blocks(ops,parameters.encoder,s) for s in (1,2,3)],
            merges=[E.merge_constants(ops,parameters.encoder,n) for n in (1,2)],
            norm_weight=None,norm_bias=None)
        g = E.encoder_grids(8,8)
        tensors = E.encoder(ops,pe['x'],c,g,start=1,stop=3,extension_mode='per_block')
        parts=[]
        for k,channels in enumerate(CHANNELS[:3],1):
            dense = depth_to_space(ee,g.tokens[k-1],tensors[f's{k}_out'],7,channels).toFloat()
            geo = _to_coordinate(ee,dense,prepared.grid,k)
            parts.append(geo.select(band_names(channels),[f'c{k}_{c}' for c in range(channels)]))
        fraction = _to_coordinate(ee,PE.valid_fraction(ee,valid,_model_grid()),prepared.grid,1)
        # C2/C3 become exact nearest copies on the persisted C1 grid.
        grid = prepared.grid.resized(4,56,56)
        image = ee.Image.cat(parts+[fraction]).select(list(_SKIP_BANDS)).reproject(grid.projection()).updateMask(grid.mask())
        return Raster(image,_SKIP_BANDS,grid)

    def c4(self, skips: Raster, *, input_grid: Grid, parameters: SwinTParameters) -> Raster:
        import ee
        _input_grid(input_grid)
        _metadata(skips,_SKIP_BANDS,input_grid.resized(4,56,56),"skips")
        _parameters(parameters)
        ops = BlockImageOps(ee)
        g=E.encoder_grids(8,8)
        x3,_ = _windows(ee,_feature_image(ee,skips,input_grid,3),input_grid,3)
        x3=x3.updateMask(domain_mask(ee,g.coarse[2]))
        c=E.EncoderConstants(blocks=[[]]*3+[E.stage_blocks(ops,parameters.encoder,4)],
                             merges=[None]*2+[E.merge_constants(ops,parameters.encoder,3)],
                             norm_weight=None,norm_bias=None)
        out=E.encoder(ops,x3,c,g,start=4,stop=4,extension_mode='per_block')['s4_out']
        image = _to_coordinate(ee,depth_to_space(ee,g.tokens[3],out,7,768).toFloat(),input_grid,4)
        return Raster(image,_C4_BANDS,input_grid.resized(32,7,7))

    def logits(self, skips: Raster, c4: Raster, *, input_grid: Grid, parameters: SwinTParameters) -> Raster:
        import ee
        _input_grid(input_grid)
        _metadata(skips,_SKIP_BANDS,input_grid.resized(4,56,56),"skips")
        _metadata(c4,_C4_BANDS,input_grid.resized(32,7,7),"c4")
        _parameters(parameters)
        feats={k:_windows(ee,_feature_image(ee,skips,input_grid,k),input_grid,k) for k in (1,2,3)}
        feats[4]=_windows(ee,c4.image.select(list(_C4_BANDS)).reproject(c4.grid.projection()),input_grid,4)
        # Same forward wiring as the saved stage C: tile-local PSP, FPN,
        # grouped-double fusion in P1/P2/P3/PSP order, then the distinct head resize.
        tensors,grids=D.fpn_forward(ee,feats,D.fpn_constants(ee,parameters.decoder))
        fused,fg=D.fusion_forward(ee,D.pyramid_features(tensors,grids),D.fusion_constants(ee,parameters.decoder,('grouped_double',)),('grouped_double',))
        head,hg=D.head_forward(ee,Feature(fused['double_relu'],fg['double_relu'],256),D.head_constants(ee,parameters.decoder))
        dense=D.head_dense(ee,head['logits'],hg['logits']).select(band_names(2))
        image=_to_coordinate(ee,dense,input_grid).toFloat().rename(list(self.output_bands))
        return Raster(image,self.output_bands,input_grid)
