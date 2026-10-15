"""Two tiny synchronous T03 queries, explicit project, no EE assets or tasks.

--offline builds/serializes against API signatures; it is not numerical evidence.
Local arrays reject unmasked NaN/Inf. Separate graph-generated nonfinite probes
record actual source payloads/masks and require processed exclusion.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import ee
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import tensor_ee
from tensor_ee import Grid,Raster,binary_threshold,compile_model,segmentation_outputs,softmax
assert Path(tensor_ee.__file__).resolve().is_relative_to(ROOT),tensor_ee.__file__

GRID=Grid("EPSG:3857",(20,0,1000,0,-20,2000),6,2)
EXT_GRID=Grid(GRID.crs,GRID.transform,8,2)
PAIRS=np.array([[-1000,1000],[1000,-1000],[0,0],[1000,1000],[1e-6,0],[-1e-6,0]],np.float32)
TRIPLES=np.array([[1000,999,-1000],[-1000,-1000,-1000],[0,0,0],[5,5,-5],[1e-6,0,-1e-6],[80,0,-80]],np.float32)
MAX=np.float32(torch.finfo(torch.float32).max)
EXTRA=np.array([[1e30,-1e30],[-1e30,1e30],[MAX,MAX],[0,1e-8],[1e-8,0],[0,0],[0,0],[0,0]],np.float32)
THRESHOLDS=(0.,.5,1.,.50000001)


def fixture_values(rows):
    return np.repeat(rows.T[:,None,:],2,axis=1)


def validate_fixture(values,mask):
    if values.dtype!=np.float32 or values.ndim!=3 or mask.shape!=values.shape:
        raise ValueError("fixture must be float32 CHW with same-shape mask")
    if not np.isfinite(values[mask]).all():
        raise ValueError("unmasked fixture input must be finite")


class Functional(nn.Module):
    def forward(self,x):
        return F.softmax(x,1,3,None)


def raster_from_arrays(values,mask,grid,bands):
    validate_fixture(values,mask)
    coords=ee.Image.pixelCoordinates(grid.projection())
    x,y=coords.select("x").floor(),coords.select("y").floor()
    index=y.multiply(grid.width).add(x)
    parts=[]
    for channel,band in enumerate(bands):
        # Masked payloads are excluded and never serialized as nonfinite values.
        safe=np.where(mask[channel],values[channel],np.float32(0)).reshape(-1)
        mapping=[ee.Number.parse(str(float(v))) for v in safe]
        image=index.remap(list(range(len(safe))),mapping).toFloat()
        flags=index.remap(list(range(len(safe))),mask[channel].astype(np.uint8).reshape(-1).astype(int).tolist())
        parts.append(image.rename(band).updateMask(flags).updateMask(grid.mask()).reproject(grid.projection()))
    return Raster(ee.Image.cat(parts),tuple(bands),grid)


def build():
    parts,metadata,cases=[],{},{}
    def package(label,raster,role,expected=None,expected_mask=None,source=None):
        count=len(raster.bands)
        parts.append(raster.image.unmask(0,sameFootprint=False).rename([f"{label}_v{i}" for i in range(count)]))
        parts.append(raster.image.mask().unmask(0,sameFootprint=False).toUint8().rename([f"{label}_m{i}" for i in range(count)]))
        metadata[label]=ee.Dictionary({"bands":raster.image.bandNames(),"types":raster.image.bandTypes(),"projection":raster.image.projection()})
        cases[label]={"role":role,"bands":raster.bands,"grid":raster.grid,"expected":expected,"mask":expected_mask,"source":source}

    validity_values=np.ones((1,2,6),np.float32);validity_values[0,1,0]=0
    validity_mask=np.ones_like(validity_values,bool);validity_mask[0,1,5]=False
    validity=raster_from_arrays(validity_values,validity_mask,GRID,("application_valid",))
    package("validity",validity,"validity",validity_values,validity_mask)
    for name,rows,bands in (("binary",PAIRS,("z","a")),("three",TRIPLES,("z","m","a")),("extra",EXTRA,("z","a"))):
        grid=EXT_GRID if name=="extra" else GRID
        values=fixture_values(rows);dense=np.ones_like(values,bool)
        source=raster_from_arrays(values,dense,grid,bands)
        if name=="extra":
            coords=ee.Image.pixelCoordinates(grid.projection());x=coords.select("x").floor();y=coords.select("y").floor()
            infinity=ee.Image.constant(1000).toDouble().exp()
            nan=ee.Image.constant(-1).toDouble().sqrt()
            inf_cell=x.eq(6).And(y.eq(0));nan_cell=x.eq(7).And(y.eq(0))
            # where() would use the finite input if the invalid replacement is
            # masked. Leave a source hole and overlay the generated invalid
            # result, preserving either its payload or its excluded mask.
            first=source.image.select(bands[0]).updateMask(inf_cell.Not()).blend(infinity.updateMask(inf_cell)).rename(bands[0]).toFloat()
            second=source.image.select(bands[1]).updateMask(nan_cell.Not()).blend(nan.updateMask(nan_cell)).rename(bands[1]).toFloat()
            source=Raster(first.addBands(second),bands,grid)
        source_label=name+"_input"
        package(source_label,source,"probe_source" if name=="extra" else "source",values,dense)
        variants=[("dense",source,values,dense,None)]
        if name!="extra":
            reverse=tuple(reversed(bands))
            variants.append(("reversed",Raster(source.image.select(list(reverse)),reverse,grid),values[::-1].copy(),dense[::-1].copy(),None))
            masked=dense.copy();masked[:,1,1]=False;masked[0,1,3]=False
            masked_source=raster_from_arrays(values,masked,grid,bands)
            masked_label=name+"_masked_input"
            package(masked_label,masked_source,"source",values,masked)
            variants.append(("masked",masked_source,values,masked,validity))
            low=softmax(masked_source)
            low_expected=torch.from_numpy(values.copy()).softmax(0).numpy()
            low_mask=np.repeat(masked.all(0)[None,:,:],len(bands),axis=0)
            package(name+"_low",low,"probabilities",low_expected,low_mask)
            for suffix,input_raster,input_mask in (("dense",source,dense),("filled",masked_source,masked)):
                compiled=compile_model(Functional().eval(),input_raster.image,input_raster.bands,grid)
                prepared=np.where(input_mask,values,np.float32(0))
                package(name+"_compiled_"+suffix,compiled,"probabilities",torch.from_numpy(prepared).softmax(0).numpy(),dense)
        for variant,raster,ordered,raw_mask,explicit in variants:
            label=name+"_"+variant
            names=tuple(f"score_{i}" for i in range(len(bands))) if variant!="dense" else None
            probabilities=tuple(f"p_{i}" for i in range(len(bands))) if variant!="dense" else None
            outputs=segmentation_outputs(raster,logit_bands=names,probability_bands=probabilities,class_band="winner",validity=explicit)
            source_of_raw=(name+"_masked_input") if variant=="masked" else source_label
            package(label+"_raw",outputs.logits,"raw",ordered,raw_mask,source=source_of_raw)
            common=raw_mask.all(0)
            if explicit is not None:common=common & validity_mask[0] & (validity_values[0]==1)
            if name=="extra":common[0,6:8]=False
            reference=torch.from_numpy(ordered.copy()).softmax(0).numpy()
            probability_mask=np.repeat(common[None,:,:],len(bands),axis=0)
            package(label+"_p",outputs.probabilities,"probabilities",reference,probability_mask)
            package(label+"_class",outputs.classes,"classes",torch.from_numpy(ordered.copy()).argmax(0).numpy()[None,:,:].astype(np.int32),common[None,:,:])
            if len(bands)==2:
                # Masked binary decisions test their own explicit validity path,
                # starting from low-level softmax without application validity.
                feed=softmax(raster) if explicit is not None else outputs.probabilities
                for positive in (0,1):
                    for index,threshold in enumerate(THRESHOLDS):
                        result=binary_threshold(feed,positive_class=positive,threshold=threshold,output_band="decision",validity=explicit)
                        expected=(torch.from_numpy(reference[positive]).double()>=float(threshold)).numpy()[None,:,:].astype(np.uint8)
                        package(f"{label}_t{positive}_{index}",result,"threshold",expected,common[None,:,:])
    image=ee.Image.cat(parts)
    field_count=sum(2*len(c["bands"]) for c in cases.values())
    query_grid={"dimensions":{"width":10,"height":4},"crsCode":GRID.crs,"affineTransform":{"scaleX":20,"shearX":0,"translateX":980,"shearY":0,"scaleY":-20,"translateY":2020}}
    return image,ee.Dictionary(metadata),cases,query_grid,field_count


def read(pixels,label,kind,count):
    return np.stack([pixels[f"{label}_{kind}{i}"] for i in range(count)])


def errors(actual,expected,mask):
    delta=np.abs(actual[mask].astype(np.float64)-expected[mask].astype(np.float64))
    return {"max_absolute_error":float(delta.max()),"mae":float(delta.mean()),"rmse":float(np.sqrt((delta**2).mean())),
            "max_relative_error_floor_1e_3":float((delta/np.maximum(np.abs(expected[mask].astype(np.float64)),1e-3)).max())}


def run(args):
    # Validate all local fixture arrays before any EE initialization/request.
    for rows in (PAIRS,TRIPLES,EXTRA):
        values=fixture_values(rows);validate_fixture(values,np.ones_like(values,bool))
    if args.offline:
        from ee.apitestcase import ApiTestCase
        harness=ApiTestCase();harness.setUp()
    else:
        import google.auth
        credentials,_=google.auth.default(scopes=["https://www.googleapis.com/auth/earthengine","https://www.googleapis.com/auth/cloud-platform"])
        ee.Initialize(credentials=credentials,project=args.project)
    image,metadata,cases,query_grid,field_count=build()
    serialized=image.serialize();graph_bytes=len(serialized.encode())
    scalar_count=40*field_count
    if args.offline:
        print(json.dumps({"status":"offline_graph_only","source":tensor_ee.__file__,"graph_bytes":graph_bytes,"fields":field_count,"scalar_samples":scalar_count,"query_grid":query_grid,"cases":len(cases)}))
        harness.tearDown();return
    pixels=ee.data.computePixels({"expression":image,"fileFormat":"NUMPY_NDARRAY","grid":query_grid})
    raw_path=str(Path(args.output).with_suffix(".pixels.npy"));np.save(raw_path,pixels,allow_pickle=False)
    remote=metadata.getInfo()
    report={"schema_version":1,"status":"pending","revision":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
            "dirty":bool(subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip()),"source":tensor_ee.__file__,
            "project":args.project,"versions":{k:version(k) for k in ("earthengine-api","torch","numpy")},
            "graph_bytes":graph_bytes,"graph_sha256":hashlib.sha256(serialized.encode()).hexdigest(),"field_count":field_count,"scalar_samples":scalar_count,
            "queries":["computePixels NUMPY_NDARRAY 10x4", "Dictionary.getInfo output metadata"],"computation_queries":2,"ee_tasks":0,"ee_assets_created":0,
            "raw_pixel_response":raw_path,"raw_pixel_sha256":hashlib.sha256(Path(raw_path).read_bytes()).hexdigest(),
            "tolerance":{"probability_max_absolute":1e-6,"probability_sum":2e-6,"classes_thresholds":"exact"},
            "thresholds":THRESHOLDS,"threshold_reference":"float32 probabilities widened losslessly; .double() >= float(threshold)",
            "fixtures":{"pairs":PAIRS.astype(float).tolist(),"triples":TRIPLES.astype(float).tolist(),"extra":EXTRA.astype(float).tolist()},
            "fixture_sha256":{name:hashlib.sha256(rows.tobytes()).hexdigest() for name,rows in (("pairs",PAIRS),("triples",TRIPLES),("extra",EXTRA))},
            "expected_mask_recipe":{"score_masks":"whole row1,col1 and channel0 row1,col3 in masked variants","validity":"zero row1,col0; masked row1,col5","server_nonfinite_probes":"extra channel0 row0,col6 exp(1000); channel1 row0,col7 sqrt(-1)","outside":"one-pixel query border"},
            "results":{},"failures":[],"metadata":remote}
    def require(condition,message):
        if not condition:report["failures"].append(message)
    for label,case in cases.items():
        grid=case["grid"];count=len(case["bands"])
        full=read(pixels,label,"v",count);actual=full[:,1:grid.height+1,1:grid.width+1]
        masks=read(pixels,label,"m",count).astype(bool)
        expected_mask=np.zeros((count,4,10),bool);expected_mask[:,1:grid.height+1,1:grid.width+1]=case["mask"]
        info=remote[label]
        dtype="int32" if case["role"]=="classes" else "uint8" if case["role"]=="threshold" else "float32"
        stats={"role":case["role"],"bands":case["bands"],"grid":asdict(grid),"valid_count":int(masks.sum()),"expected_valid_count":int(expected_mask.sum()),
               "dtype":str(full.dtype),"dtype_match":str(full.dtype)==dtype,"band_order_match":info["bands"]==list(case["bands"]),
               "grid_match":info["projection"]["crs"]==grid.crs and info["projection"]["transform"]==list(grid.transform),
               "mask_match_including_border":bool(np.array_equal(masks,expected_mask)),
               "mask_mismatch_locations":np.argwhere(masks!=expected_mask).tolist()}
        if case["role"]=="probe_source":
            # Raw source probes can retain a nonfinite value or be masked by the
            # EE math operation. Record both, with no assertion of hidden payloads.
            known=np.ones_like(expected_mask,bool)
            known[0,1,7]=False;known[1,1,8]=False
            stats["mask_match_including_border"]=bool(np.array_equal(masks[known],expected_mask[known]))
            stats["probe_payloads"]={}
            for channel,column in ((0,6),(1,7)):
                value=actual[channel,0,column];unmasked=bool(masks[channel,1,column+1])
                invalid=(not unmasked) or not np.isfinite(value)
                stats["probe_payloads"][str(column)]={"value_repr":str(value),"bits":int(value.view(np.uint32)),"unmasked":unmasked,"invalid":bool(invalid)}
                require(invalid,f"{label}: server probe at col{column} did not produce invalid input")
        elif case["role"]=="raw" and label.startswith("extra"):
            source=read(pixels,case["source"],"v",count)
            source_masks=read(pixels,case["source"],"m",count).astype(bool)
            stats["mask_match_including_border"]=bool(np.array_equal(masks,source_masks))
            stats["raw_values_preserved"]=bool(np.array_equal(full[source_masks].view(np.uint32),source[source_masks].view(np.uint32)))
            require(stats["raw_values_preserved"],f"{label}: raw values changed")
        valid=case["mask"].copy()
        if label.startswith("extra") and case["role"] in ("raw","probe_source"):
            valid[:,0,6:8]=False
        if case["role"] in ("source","raw","probe_source","validity"):
            stats["input_bitwise_match"]=bool(np.array_equal(actual[valid].view(np.uint32),case["expected"][valid].view(np.uint32)))
            require(stats["input_bitwise_match"],f"{label}: declared input bits differ")
        elif case["role"]=="probabilities":
            stats.update(errors(actual,case["expected"],valid));stats["probability_tolerance_pass"]=stats["max_absolute_error"]<=1e-6
            stats["in_range"]=bool(((actual[valid]>=0)&(actual[valid]<=1)&np.isfinite(actual[valid])).all())
            sum_error=np.abs(actual.astype(np.float64).sum(0)[valid[0]]-1)
            stats["max_sum_error"]=float(sum_error.max());stats["sum_tolerance_pass"]=stats["max_sum_error"]<=2e-6
            for check in ("probability_tolerance_pass","in_range","sum_tolerance_pass"):require(stats[check],f"{label}: {check}")
        else:
            stats["decision_disagreements"]=int(np.count_nonzero(actual[valid]!=case["expected"][valid]))
            require(stats["decision_disagreements"]==0,f"{label}: decisions differ")
        precision="int" if dtype in ("int32","uint8") else "float"
        stats["metadata_precision_match"]=all(t["precision"]==precision for t in info["types"].values())
        for check in ("dtype_match","band_order_match","grid_match","mask_match_including_border","metadata_precision_match"):
            require(stats[check],f"{label}: {check}")
        report["results"][label]=stats
    report["status"]="passed" if not report["failures"] else "failed"
    Path(args.output).write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":report["status"],"output":args.output,"graph_bytes":graph_bytes,"fields":field_count,"scalar_samples":scalar_count,"failures":report["failures"],"max_probability_error":max(r.get("max_absolute_error",0) for r in report["results"].values()),"decision_disagreements":sum(r.get("decision_disagreements",0) for r in report["results"].values())}))
    if report["failures"]:raise SystemExit(1)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--project",required=True)
    parser.add_argument("--output",default="/tmp/t03-outputs-live.json");parser.add_argument("--offline",action="store_true")
    run(parser.parse_args())
