"""numpy.ma.MaskedArray is refused, never read with its mask silently dropped.

A masked array is an ndarray, so run() and Kernel used to read its data and ignore the
mask: masked elements entered the result. As a number operand, NumPy turns a masked
scalar into nan with only a warning. Both are wrong answers rather than errors, so both
are TypeErrors now. Plain ndarrays, NumPy scalars and Python numbers are unchanged.
"""
import re

import numpy as np
import pytest

import coffee_ten as ct

MESSAGE = ("input x is a numpy.ma.MaskedArray, whose mask would be ignored; pass "
           "x.filled(fill_value) to choose the masked values, or numpy.asarray(x) "
           "to use the underlying data deliberately")


@pytest.fixture(params=["run", "kernel"])
def call(request, tmp_path):
    out = ct.input("x", (2, 3)) * 2.0
    if request.param == "run":
        return lambda **inputs: ct.run(out, **inputs)
    kernel = ct.compile(out, cache_dir=tmp_path)
    return lambda **inputs: kernel(**inputs)


@pytest.mark.parametrize("masked", [
    np.ma.array(np.ones((2, 3), np.float32), mask=[[0, 1, 0], [0, 0, 1]]),
    np.ma.array(np.ones((2, 3), np.float32)),  # no element masked: still refused
    np.ma.masked_invalid(np.array([[1, np.nan, 3], [4, 5, 6]], np.float32)),
    np.ma.array(np.ones((2, 3)), mask=True),  # also float64: the mask is reported first
], ids=["some-masked", "nothing-masked", "masked-invalid", "float64"])
def test_masked_inputs_are_type_errors(call, masked):
    with pytest.raises(TypeError, match="^" + re.escape(MESSAGE) + "$"):
        call(x=masked)


def test_the_suggested_conversions_work(call):
    masked = np.ma.array(np.arange(6, dtype=np.float32).reshape(2, 3), mask=[[0, 1, 0], [0, 0, 0]])
    filled = call(x=masked.filled(np.float32(0)))
    assert filled.tobytes() == (masked.filled(np.float32(0)) * np.float32(2)).tobytes()
    raw = call(x=np.asarray(masked))
    assert raw.tobytes() == (masked.data * np.float32(2)).tobytes()


def test_plain_ndarray_subclasses_are_still_accepted(call):
    class Tagged(np.ndarray):
        pass

    data = np.arange(6, dtype=np.float32).reshape(2, 3)
    assert call(x=data.view(Tagged)).tobytes() == (data * np.float32(2)).tobytes()


@pytest.mark.parametrize("value", [np.ma.masked, np.ma.array(2.0, mask=True), np.ma.array(2.0)],
                         ids=["masked-constant", "masked-0d", "unmasked-0d"])
def test_masked_number_operands_are_refused(value):
    x = ct.input("x", (3,))
    with pytest.raises(TypeError, match="mask would be ignored"):
        ct.maximum(x, value)
    with pytest.raises(TypeError, match="mask would be ignored"):
        ct.maximum(value, x)
    # Operators return NotImplemented, so Python then tries the MaskedArray's reflected
    # method, which fails; either way no Tensor is built.
    for build in (lambda: x + value, lambda: value * x):
        with pytest.raises((TypeError, AttributeError)):
            build()


@pytest.mark.parametrize("value", [2.0, 2, np.float32(2.0), np.float64(2.0)])
def test_plain_number_operands_are_unchanged(value):
    x = ct.input("x", (3,))
    data = np.array([1.0, 3.0, -1.0], np.float32)
    assert ct.run(ct.maximum(x, value), x=data).tolist() == [2.0, 3.0, 2.0]
    assert ct.run(x + value, x=data).tolist() == [3.0, 5.0, 1.0]
