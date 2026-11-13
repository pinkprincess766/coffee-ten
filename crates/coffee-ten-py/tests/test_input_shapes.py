"""ct.input shape checks: every error names the input and, for a bad dimension, its index.

Exception types are unchanged; only the messages gained the input name and position.
Accepted shapes are unchanged too, including empty dimensions, scalars, NumPy integers and
bool (an int subclass in Python).
"""
import re

import numpy as np
import pytest

import coffee_ten as ct


@pytest.mark.parametrize("shape, index, value", [
    ((-1,), 0, -1),
    ((2, -3), 1, -3),
    ((4, 5, -9223372036854775808), 2, -9223372036854775808),
])
def test_negative_dimension_names_input_index_and_value(shape, index, value):
    message = f"input 'weights': negative dimension {value} at index {index}"
    with pytest.raises(ValueError, match="^" + re.escape(message) + "$"):
        ct.input("weights", shape)


@pytest.mark.parametrize("shape, index, shown", [
    ((2.5,), 0, "float 2.5"),
    ((3, 3.0), 1, "float 3.0"),
    (("3",), 0, "str '3'"),
    ((2, None), 1, "NoneType None"),
    ((np.float32(2),), 0, "float32 " + repr(np.float32(2))),
    (((2,),), 0, "tuple (2,)"),
])
def test_non_integer_dimension_is_a_type_error_naming_input_and_index(shape, index, shown):
    message = f"input 'x': dimension {index} must be an int, got {shown}"
    with pytest.raises(TypeError, match="^" + re.escape(message) + "$"):
        ct.input("x", shape)


@pytest.mark.parametrize("value", [2**63, 2**64, -(2**63) - 1])
def test_dimension_outside_64_bits_is_an_overflow_error_naming_input_and_index(value):
    message = f"input 'big': dimension 1 is out of range: {value}"
    with pytest.raises(OverflowError, match="^" + re.escape(message) + "$"):
        ct.input("big", (1, value))


def test_unallocatable_shape_names_the_input():
    shape = (2**40, 2**40)
    message = f"input 'huge': shape {shape} has too many elements"
    with pytest.raises(ValueError, match="^" + re.escape(message) + "$"):
        ct.input("huge", shape)


def test_first_bad_dimension_is_reported():
    with pytest.raises(TypeError, match="dimension 1 must be an int"):
        ct.input("x", (2, 1.5, -1))
    with pytest.raises(ValueError, match="negative dimension -1 at index 1"):
        ct.input("x", (2, -1, 1.5))


@pytest.mark.parametrize("shape, expected", [
    ((), ()),
    ((0,), (0,)),
    ((3, 0, 2), (3, 0, 2)),
    ((np.int64(3), np.uint8(4)), (3, 4)),
    ((True, 2), (1, 2)),
    ((2**31,), (2**31,)),
])
def test_accepted_shapes_are_unchanged(shape, expected):
    assert ct.input("x", shape).shape == expected


def test_shape_must_be_a_tuple():
    for shape in ([2, 3], 3, None):
        with pytest.raises(TypeError):
            ct.input("x", shape)
