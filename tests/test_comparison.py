"""Exact local inputs, independent masks and finite segmentation comparisons."""
from unittest.mock import patch
import json
import numpy as np
import pytest
import torch
from torch import nn

from tensor_ee import Grid, PixelTensor, compare_segmentation
from tensor_ee.examples.nonwater import fixture


GRID = Grid('EPSG:3857', (30, 0, 1200, 0, -30, 2400), 3, 2)


def tensors():
    model = nn.Identity().eval()
    values = np.array([[[0., 2., -2.], [1., 0., .1]], [[0., -2., 2.], [1., 1e-8, -.1]]], dtype=np.float32)
    dense = np.ones_like(values, bool)
    valid = np.ones((2, 3), bool)
    valid[0, 1] = False
    probabilities = torch.from_numpy(values).softmax(dim=0).numpy()
    classes = values.argmax(axis=0).astype(np.int32)[None]
    return model, PixelTensor(values.copy(), dense.copy(), ('a','b'), GRID), \
        PixelTensor(values.copy(), dense.copy(), ('score_a','score_b'), GRID), \
        PixelTensor(probabilities, np.broadcast_to(valid, values.shape).copy(), ('p_a','p_b'), GRID), \
        PixelTensor(classes, valid[None].copy(), ('class_index',), GRID), \
        dict(expected_prepared=PixelTensor(values.copy(), dense, ('a','b'), GRID), expected_logit_mask=dense,
             expected_output_mask=valid, pixel_coordinates=tuple(np.indices((2,3), dtype=np.float32)[::-1]))


def compare_case():
    model, prepared, logits, probabilities, classes, options = tensors()
    return compare_segmentation(model, prepared, logits, probabilities, classes, **options)


def test_exact_comparison_no_cloud_and_raw_logit_near_tie_identity():
    with patch('ee.Initialize', side_effect=AssertionError('cloud')), patch('ee.data.computePixels', side_effect=AssertionError('cloud')):
        report = compare_case()
    json.dumps(report, allow_nan=False)
    assert report['status'] == 'pass'
    assert report['prepared_bit_mismatches'] == report['class_disagreements'] == 0
    assert report['logit_errors']['max_absolute_error'] == report['probability_errors']['max_absolute_error'] == 0
    assert report['valid_output_pixels'] == 5
    assert any(x['row'] == 1 and x['column'] == 1 and x['reference_class'] == 1 for x in report['near_margin_pixels'])
    assert report['accuracy'] == 'not_measured'


def test_independent_expected_masks_cannot_hide_numerical_failures():
    model, prepared, logits, probabilities, classes, options = tensors()
    logits.mask[:, 1, 0] = False
    logits.values[:, 1, 0] = 100
    report = compare_segmentation(model, prepared, logits, probabilities, classes, **options)
    assert report['status'] == 'fail'
    assert report['mask_mismatches']['logits'] == 2
    assert report['logit_errors']['valid_count'] == 12
    assert report['logit_errors']['max_absolute_error'] == 99
    probabilities.mask[:, 1, 1] = False
    classes.mask[:, 1, 1] = False
    report = compare_segmentation(model, prepared, logits, probabilities, classes, **options)
    assert report['mask_mismatches']['probabilities'] == 2 and report['mask_mismatches']['classes'] == 1


def test_signed_zero_ieee_identity_and_orientation_are_verified():
    model, prepared, logits, probabilities, classes, options = tensors()
    prepared.values[0,0,0] = np.float32(-0.)
    report = compare_segmentation(model, prepared, logits, probabilities, classes, **options)
    assert report['status'] == 'fail' and report['prepared_bit_mismatches'] == 1
    options['pixel_coordinates'] = tuple(x[::-1] for x in options['pixel_coordinates'])
    report = compare_segmentation(model, prepared, logits, probabilities, classes, **options)
    assert any('orientation' in x for x in report['failures'])


def test_optional_binary_threshold_uses_double_scalar_and_exact_masks():
    model, prepared, logits, probabilities, classes, options = tensors()
    threshold = .50000001
    binary = (probabilities.values[1].astype(np.float64) >= threshold).astype(np.uint8)[None]
    mask = PixelTensor(binary, classes.mask.copy(), ('decision',), GRID)
    report = compare_segmentation(model, prepared, logits, probabilities, classes,
                                  binary_mask=mask, threshold=threshold, **options)
    assert report['status'] == 'pass'
    assert binary[0,0,0] == 0  # .5 is below the supplied double threshold
    mask.values[0,0,0] = 1
    assert compare_segmentation(model, prepared, logits, probabilities, classes,
        binary_mask=mask, threshold=threshold, **options)['status'] == 'fail'


def test_masked_nan_is_excluded_but_unmasked_invalid_payload_rejected():
    model, prepared, logits, probabilities, classes, options = tensors()
    probabilities.values[:,0,1] = np.nan
    probabilities = PixelTensor(probabilities.values, probabilities.mask, probabilities.bands, GRID)
    report = compare_segmentation(model, prepared, logits, probabilities, classes, **options)
    assert report['status'] == 'pass'
    json.dumps(report, allow_nan=False)
    with pytest.raises(ValueError, match='unmasked'):
        PixelTensor(probabilities.values, np.ones_like(probabilities.mask), probabilities.bands, GRID)
    with pytest.raises(ValueError, match='unmasked'):
        PixelTensor(np.full((1,2,3),np.inf,np.float32),np.ones((1,2,3),bool),('a',),GRID)


def test_no_coercion_or_reference_mutation_and_array_schema_rejections():
    model, prepared, logits, probabilities, classes, options = tensors()
    with pytest.raises(ValueError, match='eval'):
        compare_segmentation(model.train(), prepared, logits, probabilities, classes, **options)
    assert model.training
    model = nn.Conv2d(2,2,1).double().eval()
    with pytest.raises(ValueError, match='CPU float32'):
        compare_segmentation(model, prepared, logits, probabilities, classes, **options)
    assert model.weight.dtype == torch.float64
    with pytest.raises(ValueError, match='dense'):
        prepared.mask[0,0,0] = False
        compare_segmentation(nn.Identity().eval(), prepared, logits, probabilities, classes, **options)
    with pytest.raises(ValueError, match='float32'):
        prepared.mask[:] = True
        compare_segmentation(nn.Identity().eval(), PixelTensor(prepared.values.astype(np.float64), prepared.mask, prepared.bands,GRID), logits, probabilities, classes, **options)
    with pytest.raises(ValueError, match='ordered bands'):
        options['expected_prepared'] = PixelTensor(prepared.values, prepared.mask, ('b','a'), GRID)
        compare_segmentation(nn.Identity().eval(), prepared, logits, probabilities, classes, **options)
    with pytest.raises(ValueError, match='boolean'):
        PixelTensor(prepared.values, prepared.mask.astype(np.float32), prepared.bands,GRID)
    with pytest.raises(ValueError, match='unique'):
        PixelTensor(prepared.values, prepared.mask, ('a','a'),GRID)


def test_exact_nonwater_prescription_and_missing_fill():
    model, raw, source_mask, prepared = fixture()
    i = np.arange(4*12*16, dtype=np.int64)
    np.testing.assert_array_equal(raw.reshape(-1), ((13*i+5)%67-33).astype(np.float32)/np.float32(16))
    for parameter in model.parameters():
        i = torch.arange(parameter.numel())
        torch.testing.assert_close(parameter.reshape(-1), ((17*i+3)%31-15).float()/128, rtol=0, atol=0)
    assert not source_mask[:,0,0].any() and not source_mask[3,5,7]
    assert source_mask.sum() == 763
    assert not prepared[:,0,0].any() and prepared[3,5,7] == 0
    assert source_mask.all(axis=0).sum() == 190
    assert tuple(model(torch.from_numpy(prepared)[None]).shape) == (1,3,12,16)
