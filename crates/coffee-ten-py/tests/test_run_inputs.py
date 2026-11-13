"""run() and Kernel.__call__ validate their keyword inputs the same way.

Each error names the input. A name the graph does not have is reported as
`unknown input` before any array is read, so a misspelled input with a wrong array is
not mistaken for a dtype or type problem of a real input.
"""
import re

import numpy as np
import pytest

import coffee_ten as ct


def graph():
    x, y, w = ct.input("x", (2, 3)), ct.input("y", (3,)), ct.input("w", (2, 3))
    return x + y * w


def ones(shape, dtype=np.float32):
    return np.ones(shape, dtype=dtype)


GOOD = {"x": ones((2, 3)), "y": ones(3), "w": ones((2, 3))}


@pytest.fixture(params=["run", "kernel"])
def call(request, tmp_path):
    out = graph()
    if request.param == "run":
        return lambda **inputs: ct.run(out, **inputs)
    kernel = ct.compile(out, cache_dir=tmp_path)
    return lambda **inputs: kernel(**inputs)


CASES = [
    # Missing inputs: the first one in declaration order.
    ({}, ValueError, "missing input x"),
    ({"x": ones((2, 3))}, ValueError, "missing input y"),
    ({"x": ones((2, 3)), "y": ones(3)}, ValueError, "missing input w"),
    # Unknown names win over every array problem, their own and others'.
    (dict(GOOD, z=ones(1)), ValueError, "unknown input z"),
    (dict(GOOD, z=ones(1, np.float64)), ValueError, "unknown input z"),
    (dict(GOOD, z=[1.0]), ValueError, "unknown input z"),
    ({"x": ones((2, 3), np.float64), "Y": ones(3)}, ValueError, "unknown input Y"),
    ({"x": ones((2, 3)), "out": ones((2, 3))}, ValueError, "unknown input out"),
    # Wrong shape or rank, named.
    (dict(GOOD, x=ones((3, 2))), ValueError, "input x has shape (3, 2), expected (2, 3)"),
    (dict(GOOD, x=ones(6)), ValueError, "input x has shape (6,), expected (2, 3)"),
    (dict(GOOD, y=np.array(1, np.float32)), ValueError, "input y has shape (), expected (3,)"),
    # Wrong dtype or not an ndarray: TypeError, no cast.
    (dict(GOOD, y=ones(3, np.float64)), TypeError, "input y must have dtype float32, got float64"),
    (dict(GOOD, y=ones(3, np.float16)), TypeError, "input y must have dtype float32, got float16"),
    (dict(GOOD, y=ones(3, np.int32)), TypeError, "input y must have dtype float32, got int32"),
    (dict(GOOD, y=np.ones(3, ">f4")), TypeError, "input y must have dtype float32, got >f4"),
    (dict(GOOD, w=[[1.0] * 3] * 2), TypeError, "input w must be a numpy.ndarray of dtype float32, got list"),
    (dict(GOOD, y=np.float32(1)), TypeError, "input y must be a numpy.ndarray of dtype float32, got float32"),
]


@pytest.mark.parametrize("inputs, error, message", CASES)
def test_each_input_error_names_the_input(call, inputs, error, message):
    with pytest.raises(error, match="^" + re.escape(message) + "$"):
        call(**inputs)


def test_run_and_kernel_raise_identical_errors(tmp_path):
    out = graph()
    kernel = ct.compile(out, cache_dir=tmp_path)
    for inputs, _, _ in CASES:
        errors = []
        for fn in (lambda **a: ct.run(out, **a), kernel):
            with pytest.raises(Exception) as caught:
                fn(**inputs)
            errors.append((type(caught.value), str(caught.value)))
        assert errors[0] == errors[1]


def test_valid_inputs_give_the_same_bits_as_numpy(call):
    rng = np.random.default_rng(24)
    inputs = {name: rng.standard_normal(a.shape).astype(np.float32) for name, a in GOOD.items()}
    expected = inputs["x"] + inputs["y"] * inputs["w"]
    result = call(**inputs)
    assert result.dtype == np.float32
    assert result.tobytes() == expected.tobytes()
