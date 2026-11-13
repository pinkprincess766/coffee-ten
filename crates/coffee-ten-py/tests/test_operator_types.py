"""Operators and `maximum` with unsupported operand types raise TypeError on both sides.

Supported operands are unchanged: a Tensor, or a real number (Python int/float/bool,
NumPy scalar or 0-d array, Fraction, Decimal) that becomes a float32 constant exactly as
NumPy 2 casts a Python scalar against a float32 array. Everything else is a TypeError,
never a ValueError, a pyo3 PanicException or a silently built Tensor.
"""
import decimal
import fractions
import operator

import numpy as np
import pytest

import coffee_ten as ct

ARITHMETIC = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}
UNSUPPORTED_OPERATORS = {"//": operator.floordiv, "%": operator.mod, "**": operator.pow,
                         "<<": operator.lshift, "&": operator.and_, "<": operator.lt,
                         ">=": operator.ge}
BAD_OPERANDS = {
    "str": "2", "None": None, "list": [1.0, 2.0], "tuple": (1.0,), "dict": {},
    "complex": 1j, "ndarray": np.ones((2, 3), np.float32),
    "int beyond float": 10**400, "object": object(),
}
NUMBERS = {"int": 2, "float": 2.0, "bool": True, "np.float32": np.float32(2.0),
           "np.int64": np.int64(2), "Fraction": fractions.Fraction(1, 2),
           "Decimal": decimal.Decimal("1.5"), "0-d ndarray": np.array(2.0, np.float32)}


def x():
    return ct.input("x", (2, 3))


def x_flat():
    return ct.input("x", (3,))


def assert_type_error(build):
    with pytest.raises(TypeError) as caught:
        build()
    # PanicException derives from BaseException, not TypeError, so this also excludes it.
    assert type(caught.value) is TypeError


@pytest.mark.parametrize("name", ARITHMETIC)
@pytest.mark.parametrize("operand", BAD_OPERANDS, ids=str)
def test_arithmetic_with_unsupported_operands(name, operand):
    op, value = ARITHMETIC[name], BAD_OPERANDS[operand]
    assert_type_error(lambda: op(x(), value))
    assert_type_error(lambda: op(value, x()))


@pytest.mark.parametrize("operand", list(BAD_OPERANDS) + list(NUMBERS), ids=str)
def test_matmul_accepts_only_tensors(operand):
    value = {**BAD_OPERANDS, **NUMBERS}[operand]
    assert_type_error(lambda: x() @ value)
    assert_type_error(lambda: value @ x())
    assert_type_error(lambda: ct.matmul(x(), value))


@pytest.mark.parametrize("operand", BAD_OPERANDS, ids=str)
def test_maximum_with_unsupported_operands(operand):
    value = BAD_OPERANDS[operand]
    assert_type_error(lambda: ct.maximum(x(), value))
    assert_type_error(lambda: ct.maximum(value, x()))


@pytest.mark.parametrize("name", UNSUPPORTED_OPERATORS)
def test_operators_the_builder_does_not_define(name):
    op = UNSUPPORTED_OPERATORS[name]
    for other in (x(), 2.0):
        assert_type_error(lambda: op(x(), other))
    for unary in (operator.invert, abs, operator.pos):
        assert_type_error(lambda: unary(x()))


def test_unary_functions_take_only_tensors():
    for fn in (ct.relu, ct.exp, ct.sum, ct.max, ct.softmax):
        for value in (2.0, np.ones(3, np.float32), None, "x"):
            assert_type_error(lambda: fn(value))


@pytest.mark.parametrize("operand", NUMBERS, ids=str)
def test_supported_numbers_are_float32_constants_on_both_sides(operand):
    value = NUMBERS[operand]
    data = np.arange(6, dtype=np.float32).reshape(2, 3) - np.float32(2.5)
    f32 = np.float32(float(value))
    for name, op in ARITHMETIC.items():
        expected = op(data, f32), op(f32, data)
        left, right = ct.run(op(x(), value), x=data), ct.run(op(value, x()), x=data)
        assert left.tobytes() == expected[0].tobytes(), name
        assert right.tobytes() == expected[1].tobytes(), name
    assert ct.run(ct.maximum(x(), value), x=data).tobytes() == np.where(data > f32, data, f32).tobytes()


def test_out_of_float32_range_numbers_round_like_numpy():
    # A finite Python float beyond float32 becomes inf, as NumPy 2 casts it; tiny ones round
    # to zero. Only ints that do not fit a float64 are refused (TypeError above).
    data = np.ones(3, np.float32)
    with np.errstate(over="ignore"):
        for value in (1e300, -3.5e38, 1e-50, 2**70):
            expected = data + np.float32(value)
            assert ct.run(x_flat() + value, x=data).tobytes() == expected.tobytes()


def test_numpy_arrays_on_the_left_do_not_become_object_arrays():
    # __array_ufunc__ = None makes NumPy defer, so ndarray + Tensor is a TypeError.
    for op in ARITHMETIC.values():
        assert_type_error(lambda: op(np.ones((2, 3), np.float32), x()))
