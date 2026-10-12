"""Tests of exp, `/`, last-axis `ct.sum` and `ct.max`, and `ct.softmax`.

exp is not exactly rounded (libm in the interpreter, `expf` in the compiled kernel), so
the two paths are each compared with a float64 NumPy reference within a derived bound,
not with each other. Division, sum and max of hand values are exact.
"""

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings
from test_diff_numpy import f32_arrays, f32_values, gamma

import coffee_ten as ct

U = 2.0**-24
F32_TINY_ABS = 2.0**-149  # the spacing of float32 subnormals

SHAPES = [(1, 5), (4, 7), (3, 64)]
SHIFTS = [0.0, 1000.0]


def f32(values):
    return np.array(values, dtype=np.float32)


A_VALUES = f32([[1.0, 5.0, 3.0], [-2.0, -8.0, -4.0]])
B_VALUES = f32([[3.0, -0.5, 7.0], [0.1, 6.0, -9.0]])


# --- exp -----------------------------------------------------------------------


def test_exp_is_within_one_ulp_of_the_rounded_float64_result():
    # libm is within 1 ulp of the true value, so it is within 1 ulp of the nearest float32.
    x = f32([[-20.0, -1.5, -0.0], [0.0, 0.1, 3.0]])
    out = ct.run(ct.exp(ct.input("x", (2, 3))), x=x)
    assert out.dtype == np.float32
    assert out.shape == (2, 3)
    np.testing.assert_array_max_ulp(out, np.exp(x.astype(np.float64)).astype(np.float32), maxulp=1)
    assert out[0, 2] == 1.0 and out[1, 0] == 1.0


def test_exp_overflow_and_underflow_follow_ieee():
    out = ct.run(ct.exp(ct.input("x", (3,))), x=f32([1000.0, -1000.0, -np.inf]))
    np.testing.assert_array_equal(out, f32([np.inf, 0.0, 0.0]))


# --- division ------------------------------------------------------------------


def test_div_matches_numpy_exactly():
    A = ct.input("A", (2, 3))
    B = ct.input("B", (2, 3))
    out = ct.run(A / B, A=A_VALUES, B=B_VALUES)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, A_VALUES / B_VALUES)
    # 1 / 3 is not exact in float32, so the rounding is part of what is compared.
    assert np.any(out.astype(np.float64) != A_VALUES.astype(np.float64) / B_VALUES)


def test_div_by_a_number_and_a_number_by_a_tensor_keep_operand_order():
    A = ct.input("A", (2, 3))
    B = ct.input("B", (2, 3))
    np.testing.assert_array_equal(ct.run(A / 4.0, A=A_VALUES), A_VALUES / np.float32(4.0))
    np.testing.assert_array_equal(ct.run(A / 3, A=A_VALUES), A_VALUES / np.float32(3))
    np.testing.assert_array_equal(ct.run(1.0 / B, B=B_VALUES), np.float32(1.0) / B_VALUES)
    np.testing.assert_array_equal(ct.run(2 / B, B=B_VALUES), np.float32(2) / B_VALUES)
    np.testing.assert_array_equal(
        ct.run(np.float32(1.0) / B, B=B_VALUES), np.float32(1.0) / B_VALUES
    )
    # The constant is the float32 nearest to 0.1, as for `+` and `-`.
    np.testing.assert_array_equal(ct.run(A / 0.1, A=A_VALUES), A_VALUES / np.float32(0.1))


def test_div_broadcasts_and_by_zero_follows_ieee():
    a = f32([[1.0], [-2.0]])
    b = f32([4.0, 0.0, -0.0])
    out = ct.run(ct.input("a", (2, 1)) / ct.input("b", (3,)), a=a, b=b)
    with np.errstate(divide="ignore"):
        expected = a / b
    assert out.shape == (2, 3)
    np.testing.assert_array_equal(out, expected)
    assert np.isinf(out).sum() == 4


def test_div_errors():
    A = ct.input("A", (2, 3))
    assert (A / ct.input("b", (3,))).shape == (2, 3)
    with pytest.raises(ValueError, match="broadcast"):
        A / ct.input("c", (4,))
    with pytest.raises(TypeError):
        A / "x"
    with pytest.raises(TypeError):
        A_VALUES / A
    with pytest.raises(TypeError):
        A / A_VALUES


# --- sum and max over the last axis ---------------------------------------------


def test_sum_and_max_hand_values():
    A = ct.input("A", (2, 3))
    total = ct.sum(A)
    largest = ct.max(A)
    assert total.shape == (2, 1)
    assert largest.shape == (2, 1)
    out = ct.run(total, A=A_VALUES)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, f32([[9.0], [-14.0]]))
    np.testing.assert_array_equal(ct.run(largest, A=A_VALUES), f32([[5.0], [-2.0]]))


def test_sum_and_max_keep_the_last_axis_for_other_ranks():
    assert ct.sum(ct.input("v", (3,))).shape == (1,)
    assert ct.max(ct.input("v", (3,))).shape == (1,)
    assert ct.sum(ct.input("t", (2, 4, 3))).shape == (2, 4, 1)
    t = np.arange(24, dtype=np.float32).reshape(2, 4, 3) - 7.0
    np.testing.assert_array_equal(ct.run(ct.sum(ct.input("t", t.shape)), t=t), t.sum(-1, keepdims=True))
    np.testing.assert_array_equal(ct.run(ct.max(ct.input("t", t.shape)), t=t), t.max(-1, keepdims=True))
    np.testing.assert_array_equal(ct.run(ct.max(ct.input("v", (1,))), v=f32([-3.0])), f32([-3.0]))


def test_max_nan_differs_from_numpy_as_documented():
    # `acc > x ? acc : x`: a nan is dropped when another element follows it.
    A = ct.input("A", (2, 3))
    x = f32([[np.nan, 1.0, 2.0], [1.0, 2.0, np.nan]])
    out = ct.run(ct.max(A), A=x)
    assert out[0, 0] == 2.0
    assert np.isnan(out[1, 0])
    assert np.isnan(np.max(x, axis=-1)).all()


def test_sum_and_max_shape_errors_come_at_op_time():
    for reduce in (ct.sum, ct.max):
        with pytest.raises(ValueError, match=r"^cannot reduce shape \(\) over its last axis$"):
            reduce(ct.input("s", ()))
        with pytest.raises(ValueError, match=r"^cannot reduce shape \(2, 0\) over its last axis$"):
            reduce(ct.input("e", (2, 0)))
        with pytest.raises(TypeError):
            reduce(1.0)
        with pytest.raises(TypeError):
            reduce(A_VALUES)
    with pytest.raises(ValueError, match=r"^cannot reduce shape \(\) over its last axis$"):
        ct.softmax(ct.input("s", ()))
    with pytest.raises(TypeError):
        ct.exp(1.0)
    with pytest.raises(TypeError):
        ct.softmax(A_VALUES)


def test_sum_and_max_compile_like_the_interpreter(tmp_path):
    A = ct.input("A", (2, 3))
    for y in (ct.sum(A), ct.max(A), ct.exp(A) / ct.sum(A)):
        compiled = ct.compile(y, cache_dir=tmp_path)(A=A_VALUES)
        interpreted = ct.run(y, A=A_VALUES)
        assert compiled.shape == interpreted.shape
        if y.shape == (2, 3):  # exp is in it: compare within 2 ulp
            np.testing.assert_array_max_ulp(compiled, interpreted, maxulp=2)
        else:
            np.testing.assert_array_equal(compiled, interpreted)


# --- depth limit ----------------------------------------------------------------


def chain(depth):
    """An input shaped (2, 3) wrapped in `depth - 1` additions: an expression of that depth."""
    y = ct.input("x", (2, 3))
    for _ in range(depth - 1):
        y = y + 1.0
    return y


def test_depth_limit_applies_to_exp_div_sum_and_max():
    for build in (ct.exp, ct.sum, ct.max, lambda t: t / 2.0, lambda t: 2.0 / t):
        build(chain(999))  # level 1000 is allowed
        with pytest.raises(ValueError, match="nested more than 1000 levels"):
            build(chain(1000))
    # A level-1000 expression with a reduction on top still runs: x + 998 per element.
    out = ct.run(ct.sum(chain(999)), x=A_VALUES)
    np.testing.assert_array_equal(out, (A_VALUES + np.float32(998)).sum(-1, keepdims=True))


def test_softmax_adds_five_levels_and_the_limit_applies():
    # softmax(x) is max, sub, exp, sum, div: five levels above x.
    assert ct.softmax(chain(995)).shape == (2, 3)
    with pytest.raises(ValueError, match="nested more than 1000 levels"):
        ct.softmax(chain(996))


# --- softmax --------------------------------------------------------------------
#
# Reference: float64 softmax of the same float32 inputs. Bound per element, with
# y_i = x_i - max(x) (the exponent actually computed), u = 2**-24, n the row length:
#
#   * y_i = fl(x_i - m): one rounding, so the exponent has an absolute error of at most
#     u|y_i|, which is a relative error of about u|y_i| in exp(y_i);
#   * exp adds at most 1 ulp, which is at most 2u relative;
#   * so every numerator e_j carries a relative error eps_j <= u|y_j| + 2u;
#   * the sum S of n positive terms adds at most gamma_n relative (any order), on top of
#     the errors eps_j of its terms, whose weighted mean is at most max_j eps_j;
#   * the division adds at most 0.5 ulp = u relative.
#
# The numerator and the denominator are built from the same e_j, so to first order
#
#   |p_i_hat - p_i| / p_i <= eps_i + max_j eps_j + gamma_n + u
#                          = (u|y_i| + 2u) + (u max_j|y_j| + 2u) + gamma_n + u.
#
# (The denominator term needs the largest |y_j| of the row, not only |y_i|.) A term below
# the float32 normal range has an absolute error of up to 2**-149 instead of a relative
# one: in a numerator (divided by S >= 1, since e = exp(0) = 1 is in every row), in each
# of the n terms of S, and in a subnormal quotient (2**-150), so (n + 2) * 2**-149 is added.
# The tolerance is twice the sum, which covers the second-order terms.
#
# The row sum is simpler: the p_i_hat share the e_j and S, so sum_i p_i_hat = 1 plus
# the error of S (gamma_{n-1}) plus one division rounding (u) per term, about n*u in
# all; the test allows 2*n*u.


def softmax_reference_and_tolerance(x32):
    x = x32.astype(np.float64)
    n = x.shape[-1]
    y = np.abs(x - x.max(axis=-1, keepdims=True))
    e = np.exp(-y)
    p = e / e.sum(axis=-1, keepdims=True)
    rel = (U * y + 2 * U) + (U * y.max(axis=-1, keepdims=True) + 2 * U) + gamma(n) + U
    tol = 2 * (p * rel + (n + 2) * F32_TINY_ABS)
    return p, tol


def assert_softmax_within_bound(out, x32):
    assert out.dtype == np.float32
    assert out.shape == x32.shape
    ref, tol = softmax_reference_and_tolerance(x32)
    error = np.abs(out.astype(np.float64) - ref)
    assert np.all(error <= tol), f"max error/tol = {np.max(error / tol)}"
    n = x32.shape[-1]
    row_error = np.abs(out.astype(np.float64).sum(axis=-1) - 1.0)
    assert np.all(row_error <= 2 * n * U), f"row sum off by {row_error.max()}"


def softmax_inputs(shape, shift):
    # Rows with a spread of about 10, then shifted: the float32 values are what both the
    # kernel and the reference see.
    rng = np.random.default_rng(1234)
    return (rng.uniform(-5.0, 5.0, size=shape) + shift).astype(np.float32)


@pytest.mark.parametrize("shift", SHIFTS)
@pytest.mark.parametrize("shape", SHAPES)
def test_softmax_interpreted_is_within_the_bound(shape, shift):
    x = softmax_inputs(shape, shift)
    out = ct.run(ct.softmax(ct.input("x", shape)), x=x)
    assert_softmax_within_bound(out, x)


@pytest.mark.parametrize("shift", SHIFTS)
@pytest.mark.parametrize("shape", SHAPES)
def test_softmax_compiled_is_within_the_bound(shape, shift, tmp_path):
    x = softmax_inputs(shape, shift)
    kernel = ct.compile(ct.softmax(ct.input("x", shape)), cache_dir=tmp_path)
    assert_softmax_within_bound(kernel(x=x), x)


def test_exp_of_the_shifted_rows_overflows_so_the_max_subtraction_is_what_is_tested():
    x = softmax_inputs((3, 64), 1000.0)
    X = ct.input("x", x.shape)
    naive = ct.exp(X) / ct.sum(ct.exp(X))
    with np.errstate(all="ignore"):
        out = ct.run(naive, x=x)
    assert np.isnan(out).all()
    assert np.isinf(ct.run(ct.exp(X), x=x)).all()


def test_softmax_wide_rows_and_extreme_gaps(tmp_path):
    # A gap of 100 between the max and the min makes exp(-100) subnormal in float32.
    x = f32([[-50.0, 50.0, 0.0, 49.5], [50.0, 50.0, 50.0, 50.0], [-50.0, -50.0, 50.0, -50.0]])
    X = ct.input("x", x.shape)
    assert_softmax_within_bound(ct.run(ct.softmax(X), x=x), x)
    assert_softmax_within_bound(ct.compile(ct.softmax(X), cache_dir=tmp_path)(x=x), x)


def test_softmax_hand_values():
    # Equal entries give 1/n (a power of two here, so exact); a single entry gives 1.
    out = ct.run(ct.softmax(ct.input("x", (2, 4))), x=f32([[3.0] * 4, [-1.0] * 4]))
    np.testing.assert_array_equal(out, np.full((2, 4), 0.25, dtype=np.float32))
    out = ct.run(ct.softmax(ct.input("x", (2, 1))), x=f32([[3.0], [-1000.0]]))
    np.testing.assert_array_equal(out, np.ones((2, 1), dtype=np.float32))


def test_softmax_is_built_from_ops_and_shares_the_exponentials(tmp_path):
    kernel = ct.compile(ct.softmax(ct.input("x", (3, 7))), cache_dir=tmp_path)
    assert kernel.c_source.count("expf(") == 1
    assert kernel.c_source.count("ct_max(") == 2  # the helper and its one call
    assert ct.softmax(ct.input("x", (3, 7))).shape == (3, 7)
    assert ct.softmax(ct.input("v", (5,))).shape == (5,)
    assert ct.softmax(ct.input("t", (2, 3, 4))).shape == (2, 3, 4)


@st.composite
def softmax_cases(draw):
    rank = draw(st.integers(1, 3))
    leading = draw(st.lists(st.integers(1, 4), min_size=rank - 1, max_size=rank - 1))
    shape = (*leading, draw(st.integers(1, 32)))
    return draw(f32_arrays(shape, f32_values(50.0)))


@settings(derandomize=True, max_examples=50, deadline=None, database=None)
@given(x=softmax_cases())
def test_softmax_random_shapes_and_values_are_within_the_bound(x):
    out = ct.run(ct.softmax(ct.input("x", x.shape)), x=x)
    assert_softmax_within_bound(out, x)
