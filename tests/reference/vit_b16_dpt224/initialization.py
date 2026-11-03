"""Declared fan-in-scaled integer reference; no sampled or fitted coefficients.

For the _fixed_weights pattern p_i = ((17*i+3)%31-15)/128, define
R = sqrt(sum_i p_i**2 / N), q = round_half_up(-0.5*log2(fan_in)),
and k = round_half_up(q-log2(R)). Initialize p_i * 2**k. Token/position
parameters use q=-5. Half ties round upward, including negative exponents.
The achieved RMS is nearest 2**q in logarithmic distance among allowed scales;
it is not claimed to equal 2**q exactly. All exponent decisions use integer
comparisons of exact sums of squares, independent of reduction order or libm.
"""
from __future__ import annotations

import math

import numpy as np


def _round_rms_exponent(numerator, denominator):
    """round_half_up(0.5*log2(numerator/denominator)), exactly for positive ints."""
    assert numerator > 0 and denominator > 0
    # Locate floor(log2(numerator/denominator)) using integer bit lengths.
    exponent = numerator.bit_length() - denominator.bit_length()
    below = (numerator < denominator << exponent if exponent >= 0 else
             numerator << -exponent < denominator)
    if below:
        exponent -= 1
    lower = exponent // 2
    # The logarithmic midpoint is sqrt(2)*2**lower. Squared, it is rational.
    midpoint = 2 * lower + 1
    round_up = (numerator >= denominator << midpoint if midpoint >= 0 else
                numerator << -midpoint >= denominator)
    return lower + int(round_up)


def _pattern_square_sum(count):
    """Exact integer numerator: one period has 31 values and sum of squares 2480."""
    periods, remainder = divmod(count, 31)
    return periods * 2480 + sum(((17 * i + 3) % 31 - 15) ** 2 for i in range(remainder))


def _scaled_pattern(shape, *, fan_in=None, target_q=None):
    count = math.prod(shape)
    square_sum = _pattern_square_sum(count)
    q = _round_rms_exponent(1, fan_in) if target_q is None else target_q
    # k = round_half_up(q-log2(base_RMS)); q is an integer.
    k = q + _round_rms_exponent(count * 128**2, square_sum)
    integers = (17 * np.arange(count, dtype=np.int64) + 3) % 31 - 15
    pattern = integers.astype(np.float32) / np.float32(128)
    values = np.ldexp(pattern, k).reshape(shape)
    assert values.dtype == np.float32 and np.isfinite(values).all()
    return values, {"fan_in": fan_in, "q": q, "k": k,
                    "rms": math.ldexp(math.sqrt(square_sum / count) / 128, k),
                    "values": count, "pattern_square_sum": square_sum}


def initialize(model):
    """Overwrite and independently assert every learned value before calibration.

    Linear fan-in is in_features; Conv2d is Cin/groups * kh * kw. The only
    supported ConvTranspose2d is non-overlapping (kernel=stride, groups=1),
    whose actual per-output sum has Cin inputs. Norm affine weights are one;
    every bias is zero. Tokens and positions use the fixed target q=-5.
    """
    import torch

    modules = dict(model.named_modules())
    report = {}
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parent, field = name.rsplit(".", 1)
            module = modules[parent]
            assert parameter.dtype == torch.float32
            if field == "bias":
                parameter.zero_()
                expected = np.zeros(tuple(parameter.shape), np.float32)
                row = {"rule": "zero_bias", "rms": 0.0}
            elif isinstance(module, (torch.nn.LayerNorm, torch.nn.BatchNorm2d)):
                assert field == "weight", name
                parameter.fill_(1)
                expected = np.ones(tuple(parameter.shape), np.float32)
                row = {"rule": "unit_norm_affine", "rms": 1.0}
            else:
                fan_in, target_q = None, None
                if field in ("cls_token", "pos_embed"):
                    target_q = -5
                elif isinstance(module, torch.nn.Linear):
                    assert field == "weight", name
                    fan_in = module.in_features
                elif isinstance(module, torch.nn.Conv2d):
                    assert field == "weight", name
                    fan_in = module.in_channels // module.groups * math.prod(module.kernel_size)
                elif isinstance(module, torch.nn.ConvTranspose2d):
                    assert field == "weight" and module.groups == 1, name
                    assert module.kernel_size == module.stride and module.dilation == (1, 1), name
                    fan_in = module.in_channels
                else:
                    raise ValueError(f"no declared integer initialization for {name}")
                expected, row = _scaled_pattern(tuple(parameter.shape), fan_in=fan_in, target_q=target_q)
                # Independent torch integer formula, rather than copying expected.
                indices = torch.arange(parameter.numel(), dtype=torch.int64)
                pattern = ((17 * indices + 3) % 31 - 15).to(torch.float32) / 128
                parameter.copy_((pattern * (2.0 ** row["k"])).reshape(parameter.shape))
                row["rule"] = "fixed_q_pattern" if target_q is not None else "fan_in_pattern"
            assert np.array_equal(parameter.detach().numpy().view(np.uint32),
                                  expected.view(np.uint32)), name
            row["exact_values_asserted"] = True
            report[name] = row
    assert report.keys() == dict(model.named_parameters()).keys()
    return report
