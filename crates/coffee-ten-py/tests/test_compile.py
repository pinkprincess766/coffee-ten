"""Tests of `ct.compile`: the compiled kernel against the reference interpreter `ct.run`.

Compiled code and the interpreter do the same f32 operations in the same order, so the
comparisons are bit for bit; only a reordering schedule is compared with a tolerance.
"""

import numpy as np
import pytest
from test_diff_numpy import gamma

import coffee_ten as ct


def assert_same_bits(actual, expected):
    assert actual.dtype == np.float32
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def a_values(m, k):
    # Values that do not add exactly in f32, some negative.
    return ((np.arange(m * k, dtype=np.float32) - 5.0) * 0.1 + 1.0 / 3.0).reshape(m, k)


def b_values(k, n):
    return ((np.arange(k * n, dtype=np.float32) - 2.0) * 0.7 - 0.25).reshape(k, n)


def matmul_expr(m, k, n):
    return ct.input("A", (m, k)) @ ct.input("B", (k, n))


def matmul_inputs(m, k, n):
    return {"A": a_values(m, k), "B": b_values(k, n)}


def test_matmul_matches_the_interpreter_bit_for_bit(tmp_path):
    y = matmul_expr(4, 3, 2)
    inputs = matmul_inputs(4, 3, 2)
    kernel = ct.compile(y, cache_dir=tmp_path)
    out = kernel(**inputs)
    assert (out < 0).any()
    assert_same_bits(out, ct.run(y, **inputs))
    assert "void kernel(" in kernel.c_source
    assert kernel.schedule == ""
    assert kernel.order_class == "preserving"


def test_relu_of_matmul_plus_bias_matches_the_interpreter_bit_for_bit(tmp_path):
    y = ct.relu(matmul_expr(4, 3, 2) + ct.input("bias", (2,)))
    inputs = {**matmul_inputs(4, 3, 2), "bias": np.array([-0.9, 0.3], dtype=np.float32)}
    kernel = ct.compile(y, cache_dir=tmp_path)
    out = kernel(**inputs)
    # relu clips some elements and keeps others.
    assert (out == 0).any() and (out > 0).any()
    assert_same_bits(out, ct.run(y, **inputs))
    assert "void kernel(" in kernel.c_source


def test_cache_dir_accepts_str_and_pathlike_and_has_a_default(tmp_path):
    y = matmul_expr(4, 3, 2)
    inputs = matmul_inputs(4, 3, 2)
    expected = ct.run(y, **inputs)
    for cache_dir in (str(tmp_path / "as_str"), tmp_path / "as_path", None):
        kernel = ct.compile(y, cache_dir=cache_dir)
        assert_same_bits(kernel(**inputs), expected)
    assert list((tmp_path / "as_str").iterdir())
    assert list((tmp_path / "as_path").iterdir())


# --- Order-preserving schedules -------------------------------------------------------


def test_split_of_an_elementwise_kernel_is_bit_exact(tmp_path):
    # 7 is not a multiple of 3, so the split loop needs a guard.
    y = ct.relu(ct.input("x", (7,)) * 1.1 - 0.3)
    x = np.linspace(-2.0, 2.0, 7, dtype=np.float32)
    kernel = ct.compile(y, schedule="split(0, i0, a, b, 3)", cache_dir=tmp_path)
    assert kernel.schedule == "split(0, i0, a, b, 3)"
    assert kernel.order_class == "preserving"
    assert_same_bits(kernel(x=x), ct.run(y, x=x))


@pytest.mark.parametrize(
    ("schedule", "shape"),
    [
        ("split(1, k, ko, ki, 2)", (4, 3, 2)),  # 3 is not a multiple of 2
        ("reorder(1, i, j, k)", (4, 3, 2)),
        ("vectorize(1, j)", (4, 5, 32)),
    ],
)
def test_order_preserving_schedules_are_bit_exact(tmp_path, schedule, shape):
    m, k, n = shape
    y = matmul_expr(m, k, n)
    inputs = matmul_inputs(m, k, n)
    kernel = ct.compile(y, schedule=schedule, cache_dir=tmp_path)
    assert kernel.order_class == "preserving"
    assert_same_bits(kernel(**inputs), ct.run(y, **inputs))


def test_schedule_is_reported_in_canonical_form(tmp_path):
    kernel = ct.compile(
        matmul_expr(4, 3, 2),
        schedule="  split( 1,k,ko , ki,2 ) ;reorder(1,i,j,ko,ki)",
        cache_dir=tmp_path,
    )
    assert kernel.schedule == "split(1, k, ko, ki, 2); reorder(1, i, j, ko, ki)"


# --- A reordering schedule ------------------------------------------------------------


def test_reordering_schedule_is_within_the_float64_bound(tmp_path):
    m, k, n = 4, 6, 3
    y = matmul_expr(m, k, n)
    inputs = matmul_inputs(m, k, n)
    schedule = "split(1, k, ko, ki, 2); reorder(1, i, ki, ko, j)"
    kernel = ct.compile(y, schedule=schedule, cache_dir=tmp_path)
    assert kernel.order_class == "reordering"
    out = kernel(**inputs)
    assert out.dtype == np.float32
    assert out.shape == (m, n)
    a, b = inputs["A"].astype(np.float64), inputs["B"].astype(np.float64)
    # Same bound as test_diff_numpy.py, without the bias term: the float32 sum of K
    # products in any order is within gamma_K * sum|a b| plus the subnormal term, doubled.
    tol = 2 * (gamma(k) * (np.abs(a) @ np.abs(b)) + k * 2.0**-149)
    assert np.all(np.abs(out.astype(np.float64) - a @ b) <= tol)


# --- Errors ---------------------------------------------------------------------------


def test_schedule_text_error_has_the_byte_offset(tmp_path):
    with pytest.raises(ValueError, match="byte"):
        ct.compile(matmul_expr(4, 3, 2), schedule="split(!", cache_dir=tmp_path)


def test_illegal_schedule_step_is_a_value_error(tmp_path):
    y = matmul_expr(4, 3, 2)
    schedule = "split(1, k, ko, ki, 2); reorder(1, i, ki)"  # not a permutation of the loops
    with pytest.raises(ValueError, match="step 1"):
        ct.compile(y, schedule=schedule, cache_dir=tmp_path)
    with pytest.raises(ValueError, match="nest"):
        ct.compile(y, schedule="split(9, k, ko, ki, 2)", cache_dir=tmp_path)


def test_graph_errors_are_value_errors(tmp_path):
    with pytest.raises(ValueError, match="duplicate input A"):
        ct.compile(ct.input("A", (2,)) + ct.input("A", (2,)), cache_dir=tmp_path)
    with pytest.raises(ValueError, match="size 0"):
        ct.compile(ct.input("A", (0, 2)) + 1.0, cache_dir=tmp_path)


def test_missing_clang_is_a_runtime_error(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(RuntimeError, match="clang"):
        ct.compile(matmul_expr(4, 3, 2), cache_dir=tmp_path / "cache")


def compiled_bias_kernel(tmp_path):
    y = ct.relu(ct.input("A", (2, 3)) + ct.input("bias", (3,)))
    return ct.compile(y, cache_dir=tmp_path)


def test_call_with_a_missing_input(tmp_path):
    kernel = compiled_bias_kernel(tmp_path)
    with pytest.raises(ValueError, match="missing input bias"):
        kernel(A=np.ones((2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="missing input A"):
        kernel()


def test_call_with_an_unknown_input(tmp_path):
    kernel = compiled_bias_kernel(tmp_path)
    ones = {"A": np.ones((2, 3), dtype=np.float32), "bias": np.ones(3, dtype=np.float32)}
    with pytest.raises(ValueError, match="unknown input Z"):
        kernel(**ones, Z=ones["A"])
    # The output is not an input.
    with pytest.raises(ValueError, match="unknown input out"):
        kernel(**ones, out=ones["A"])


def test_call_with_a_wrong_shape(tmp_path):
    kernel = compiled_bias_kernel(tmp_path)
    bias = np.ones(3, dtype=np.float32)
    with pytest.raises(ValueError, match=r"input A has shape \(3, 2\), expected \(2, 3\)"):
        kernel(A=np.ones((3, 2), dtype=np.float32), bias=bias)


def test_call_with_float64_is_a_type_error_not_a_cast(tmp_path):
    kernel = compiled_bias_kernel(tmp_path)
    bias = np.ones(3, dtype=np.float32)
    with pytest.raises(TypeError, match="input A must have dtype float32, got float64"):
        kernel(A=np.ones((2, 3)), bias=bias)
    with pytest.raises(TypeError, match="input A must be a numpy.ndarray.*got list"):
        kernel(A=[[1.0] * 3] * 2, bias=bias)


def test_calling_twice_gives_identical_bits(tmp_path):
    y = ct.relu(matmul_expr(4, 3, 2) + ct.input("bias", (2,)))
    inputs = {**matmul_inputs(4, 3, 2), "bias": np.array([-0.9, 0.3], dtype=np.float32)}
    kernel = ct.compile(y, cache_dir=tmp_path)
    first = kernel(**inputs)
    other = kernel(A=-inputs["A"], B=inputs["B"], bias=inputs["bias"])
    second = kernel(**inputs)
    assert not np.shares_memory(first, second)
    assert_same_bits(second, first)
    assert not np.array_equal(other, first)


def test_inputs_may_have_any_memory_layout_and_are_not_modified(tmp_path):
    y = matmul_expr(4, 3, 2)
    inputs = matmul_inputs(4, 3, 2)
    kernel = ct.compile(y, cache_dir=tmp_path)
    expected = kernel(**inputs)
    a_fortran = np.asfortranarray(inputs["A"])
    b_strided = np.repeat(inputs["B"], 2, axis=1)[:, ::2]  # non-contiguous view
    assert not b_strided.flags["C_CONTIGUOUS"]
    assert_same_bits(kernel(A=a_fortran, B=b_strided), expected)
    np.testing.assert_array_equal(a_fortran, inputs["A"])
