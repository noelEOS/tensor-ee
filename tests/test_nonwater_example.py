"""Installed nonwater workflow schema, guards, and zero-cloud preparation."""
import json
from unittest.mock import patch
import numpy as np
import torch
from ee.apitestcase import ApiTestCase

from tensor_ee.examples import nonwater


def response(fields):
    model, raw, source_mask, prepared = nonwater.fixture()
    with torch.inference_mode():
        output = model(torch.from_numpy(prepared)[None])
        logits = output[0].numpy()
        probabilities = output.softmax(1)[0].numpy()
        classes = output.argmax(1)[0].numpy().astype(np.int32)[None]
    output_mask = source_mask.all(0)
    values = dict(raw=np.where(source_mask,raw,np.float32(0)),prepared=prepared,
                  source_validity=source_mask.astype(np.float32),output_validity=output_mask[None].astype(np.float32),
                  logits=logits,probabilities=probabilities,classes=classes)
    masks = dict(raw=source_mask,prepared=np.ones_like(source_mask),source_validity=np.ones_like(source_mask),
                 output_validity=np.ones((1,12,16),bool),logits=np.ones((3,12,16),bool),
                 probabilities=np.broadcast_to(output_mask,(3,12,16)),classes=output_mask[None])
    schema = [(name,np.int32 if name.startswith('classes_v') else np.float32) for names in fields.values() for name in names]
    schema += [('pixel_x',np.float32),('pixel_y',np.float32)]
    pixels = np.zeros((14,18),dtype=schema)
    for label, array in values.items():
        for channel, name in enumerate(fields[label]):
            pixels[name][1:13,1:17] = array[channel]
        for channel, name in enumerate(fields[label+'_mask']):
            pixels[name][1:13,1:17] = masks[label][channel]
    pixels['pixel_y'],pixels['pixel_x'] = np.indices((14,18),dtype=np.float32)-1
    names = dict(raw=nonwater.BANDS, prepared=nonwater.BANDS,source_validity=nonwater.BANDS,
                 output_validity=('valid_output',),logits=('score_a','score_b','score_c'),
                 probabilities=('probability_a','probability_b','probability_c'),classes=('class_index',))
    metadata = {label:dict(bands=list(bands),types={name:dict(precision='int' if label=='classes' else 'float') for name in bands},
                   projection=dict(crs=nonwater.GRID.crs,transform=list(nonwater.GRID.transform))) for label,bands in names.items()}
    return pixels,metadata


class TestNonwaterExample(ApiTestCase):
    def test_graph_plan_and_independent_response_checks(self):
        with patch('ee.data.computePixels',side_effect=AssertionError('cloud')):
            image,metadata,fields,compatibility = nonwater.build()
            assert compatibility['status'] == 'supported' and compatibility['parameter_count'] == 573
            assert sum(len(x) for x in fields.values())+2 == 42
            assert len(image.serialize()) < 100000
            pixels,meta = response(fields)
            report = nonwater.evaluate(pixels,meta,fields)
        json.dumps(report,allow_nan=False)
        assert report['status'] == 'pass' and report['valid_output_pixels'] == 190
        pixels['prepared_v3'][6,8] = 1
        assert nonwater.evaluate(pixels,meta,fields)['status'] == 'fail'

    def test_masks_border_orientation_and_metadata_cannot_be_omitted(self):
        _,_,fields,_ = nonwater.build()
        pixels,meta = response(fields)
        pixels['logits_m0'][0,0] = 1
        assert nonwater.evaluate(pixels,meta,fields)['status'] == 'fail'
        pixels['logits_m0'][0,0] = 0
        pixels['probabilities_m0'][2,2] = 0
        assert nonwater.evaluate(pixels,meta,fields)['status'] == 'fail'
        pixels,meta = response(fields)
        pixels['source_validity_v3'][6,8] = 1
        assert nonwater.evaluate(pixels,meta,fields)['status'] == 'fail'
        pixels,meta = response(fields)
        del meta['classes']
        assert nonwater.evaluate(pixels,meta,fields)['status'] == 'fail'
        pixels,meta = response(fields)
        pixels['pixel_y'][:] = pixels['pixel_y'][::-1]
        assert nonwater.evaluate(pixels,meta,fields)['status'] == 'fail'
