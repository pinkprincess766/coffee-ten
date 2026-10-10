"""User-path tests for `import coffee_ten as ct`.

Every elementwise f32 op is correctly rounded, so NumPy float32 results are exact
references and the comparisons are bit-for-bit (`assert_array_equal`).
"""

import numpy as np
import pytest

import coffee_ten as ct


def f32(values):
    return np.array(values, dtype=np.float32)


A_VALUES = f32([[-1.5, -0.0, 2.25], [0.5, -3.0, 4.0]])
BIAS_VALUES = f32([0.5, -0.0, -2.0])


def test_relu_of_sum_matches_numpy():
    # Column 1 is -0.0 + -0.0 = -0.0 before the relu.
    A = ct.input("A", (2, 3))
    bias = ct.input("bias", (3,))
    y = ct.relu(A + bias)
    out = ct.run(y, A=A_VALUES, bias=BIAS_VALUES)
    assert out.dtype == np.float32
    assert out.shape == (2, 3)
    np.testing.assert_array_equal(
        out, np.maximum(A_VALUES + BIAS_VALUES, np.float32(0))
    )


def test_relu_of_negative_zero_is_positive_zero():
    out = ct.run(ct.relu(ct.input("x", (2,))), x=f32([-0.0, 0.0]))
    assert not np.signbit(out).any()


def test_broadcast_column_times_row():
    a = f32([[1.5], [-2.0]])
    b = f32([3.0, 0.25, -1.0])
    y = ct.input("a", (2, 1)) * ct.input("b", (3,))
    assert y.shape == (2, 3)
    np.testing.assert_array_equal(ct.run(y, a=a, b=b), a * b)


def test_scalar_on_the_right():
    A = ct.input("A", (2, 3))
    two = np.float32(2.0)
    np.testing.assert_array_equal(ct.run(A * 2.0, A=A_VALUES), A_VALUES * two)
    np.testing.assert_array_equal(ct.run(A + 2, A=A_VALUES), A_VALUES + two)
    np.testing.assert_array_equal(ct.run(A - 2.0, A=A_VALUES), A_VALUES - two)


def test_scalar_on_the_left_keeps_operand_order():
    A = ct.input("A", (2, 3))
    one = np.float32(1.0)
    np.testing.assert_array_equal(ct.run(1.0 - A, A=A_VALUES), one - A_VALUES)
    np.testing.assert_array_equal(ct.run(1.0 + A, A=A_VALUES), one + A_VALUES)
    np.testing.assert_array_equal(ct.run(3 * A, A=A_VALUES), np.float32(3) * A_VALUES)
    np.testing.assert_array_equal(
        ct.run(np.float32(1.0) - A, A=A_VALUES), one - A_VALUES
    )


def test_python_float_is_rounded_to_f32_first():
    # 0.1 is not representable; the constant must be the f32 nearest to it.
    out = ct.run(ct.input("A", (2, 3)) + 0.1, A=A_VALUES)
    np.testing.assert_array_equal(out, A_VALUES + np.float32(0.1))


def test_tensor_minus_tensor_and_neg():
    A = ct.input("A", (2, 3))
    bias = ct.input("bias", (3,))
    np.testing.assert_array_equal(
        ct.run(A - bias, A=A_VALUES, bias=BIAS_VALUES), A_VALUES - BIAS_VALUES
    )
    out = ct.run(-A, A=A_VALUES)
    np.testing.assert_array_equal(out, -A_VALUES)
    assert np.signbit(out[0, 1]) == np.signbit(-A_VALUES[0, 1])


def test_maximum_of_two_tensors_and_with_a_scalar():
    a = f32([1.0, -2.0, 3.0, 0.0])
    b = f32([2.0, -3.0, 3.0, -1.0])
    A = ct.input("A", (4,))
    B = ct.input("B", (4,))
    np.testing.assert_array_equal(ct.run(ct.maximum(A, B), A=a, B=b), np.maximum(a, b))
    np.testing.assert_array_equal(
        ct.run(ct.maximum(A, 0.5), A=a), np.maximum(a, np.float32(0.5))
    )
    np.testing.assert_array_equal(
        ct.run(ct.maximum(0.5, A), A=a), np.maximum(np.float32(0.5), a)
    )


def test_same_tensor_twice_is_one_input():
    A = ct.input("A", (2, 3))
    y = A + A
    np.testing.assert_array_equal(ct.run(y, A=A_VALUES), A_VALUES + A_VALUES)
    with pytest.raises(ValueError, match="unknown input B"):
        ct.run(y, A=A_VALUES, B=A_VALUES)


def test_shapes_are_inferred_at_op_time():
    A = ct.input("A", (2, 3))
    assert A.shape == (2, 3)
    assert ct.relu(A + ct.input("bias", (3,))).shape == (2, 3)
    assert (ct.input("a", (2, 1)) * ct.input("b", (3,))).shape == (2, 3)
    assert (A * 2.0).shape == (2, 3)
    assert ct.maximum(1.0, 2.0).shape == ()
    assert ct.input("s", ()).shape == ()
    assert ct.input("e", (0, 3)).shape == (0, 3)


def test_scalar_and_empty_results():
    out = ct.run(ct.maximum(1.0, 2.0))
    assert out.shape == () and out.dtype == np.float32 and out == 2.0
    empty = ct.run(ct.input("e", (0, 3)) + 1.0, e=np.zeros((0, 3), dtype=np.float32))
    assert empty.shape == (0, 3) and empty.dtype == np.float32


def test_input_named_like_a_parameter_of_run():
    x = f32([1.0, 2.0])
    out = ct.run(ct.input("output", (2,)) + 1.0, output=x)
    np.testing.assert_array_equal(out, x + np.float32(1.0))


LAYOUTS = {
    "fortran": lambda a: np.asfortranarray(a),
    "transposed_view": lambda a: np.ascontiguousarray(a.T).T,
    "strided_view": lambda a: np.repeat(a, 2, axis=1)[:, ::2],
    "reversed_view": lambda a: a[::-1, ::-1].copy()[::-1, ::-1],
}


@pytest.mark.parametrize("layout", LAYOUTS)
def test_any_memory_layout_gives_the_same_result(layout):
    c_ordered = np.ascontiguousarray(A_VALUES)
    other = LAYOUTS[layout](c_ordered)
    np.testing.assert_array_equal(other, c_ordered)
    assert not other.flags.c_contiguous
    A = ct.input("A", (2, 3))
    bias = ct.input("bias", (3,))
    y = ct.relu(A - bias) * 2.0
    np.testing.assert_array_equal(
        ct.run(y, A=other, bias=BIAS_VALUES),
        ct.run(y, A=c_ordered, bias=BIAS_VALUES),
    )


def test_mismatched_shapes_raise_at_op_time():
    a = ct.input("a", (3,))
    b = ct.input("b", (4,))
    with pytest.raises(ValueError, match=r"cannot broadcast shapes \(3,\) and \(4,\)"):
        a + b
    with pytest.raises(ValueError, match="cannot broadcast"):
        ct.maximum(a, b)


def test_missing_input():
    A = ct.input("A", (2, 3))
    bias = ct.input("bias", (3,))
    with pytest.raises(ValueError, match="missing input bias"):
        ct.run(A + bias, A=A_VALUES)


def test_unknown_input():
    A = ct.input("A", (2, 3))
    with pytest.raises(ValueError, match="unknown input Z"):
        ct.run(A + 1.0, A=A_VALUES, Z=A_VALUES)


def test_wrong_input_shape():
    A = ct.input("A", (2, 3))
    with pytest.raises(ValueError, match=r"input A has shape \(3, 2\), expected \(2, 3\)"):
        ct.run(A + 1.0, A=np.ones((3, 2), dtype=np.float32))


def test_float64_input_is_a_type_error_not_a_cast():
    A = ct.input("A", (2, 3))
    with pytest.raises(TypeError, match="input A must have dtype float32, got float64"):
        ct.run(A + 1.0, A=A_VALUES.astype(np.float64))


def test_non_array_input_is_a_type_error():
    A = ct.input("A", (2,))
    with pytest.raises(TypeError, match="input A must be a numpy.ndarray.*got list"):
        ct.run(A + 1.0, A=[1.0, 2.0])
    with pytest.raises(TypeError, match="input A must be a numpy.ndarray.*got float32"):
        ct.run(A + 1.0, A=np.float32(1.0))


def test_two_different_inputs_with_one_name():
    first = ct.input("A", (2,))
    second = ct.input("A", (2,))
    x = f32([1.0, 2.0])
    with pytest.raises(ValueError, match="duplicate input A"):
        ct.run(first + second, A=x)


def test_bad_input_declarations():
    with pytest.raises(ValueError, match="negative dimension -1"):
        ct.input("A", (2, -1))
    with pytest.raises(TypeError):
        ct.input("A", [2, 3])
    with pytest.raises(TypeError):
        ct.input("A", (2.0, 3))


def test_unsupported_operands_are_type_errors():
    A = ct.input("A", (2, 3))
    with pytest.raises(TypeError):
        A + "x"
    with pytest.raises(TypeError):
        A_VALUES + A
    with pytest.raises(TypeError):
        ct.relu(1.0)
    with pytest.raises(TypeError):
        ct.run("not a tensor")


def test_expression_depth_limit_is_1000():
    x = f32([1.0, 2.0])
    y = ct.input("x", (2,))
    for _ in range(999):  # the input is level 1, so this makes level 1000
        y = y + 1.0
    out = ct.run(y, x=x)
    expected = x.copy()
    for _ in range(999):
        expected = expected + np.float32(1.0)
    np.testing.assert_array_equal(out, expected)
    with pytest.raises(ValueError, match="nested more than 1000 levels"):
        y + 1.0
    with pytest.raises(ValueError, match="nested more than 1000 levels"):
        ct.relu(y)


def test_matmul_hand_values():
    a = f32([[1, 2, 3], [4, 5, 6]])
    b = f32([[7, 8], [9, 10], [11, 12]])
    A = ct.input("A", (2, 3))
    B = ct.input("B", (3, 2))
    assert (A @ B).shape == (2, 2)
    expected = f32([[58, 64], [139, 154]])
    np.testing.assert_array_equal(ct.run(A @ B, A=a, B=b), expected)
    np.testing.assert_array_equal(ct.run(ct.matmul(A, B), A=a, B=b), expected)
    # Operand order matters: (3,2) @ (2,3) is a different (3,3) product.
    out = ct.run(B @ A, A=a, B=b)
    assert out.shape == (3, 3)
    np.testing.assert_array_equal(out, b @ a)


def test_matmul_shape_errors_use_the_graph_message():
    A = ct.input("A", (2, 3))
    with pytest.raises(ValueError, match=r"^cannot matmul shapes \(2, 3\) and \(4, 2\)$"):
        A @ ct.input("B", (4, 2))
    with pytest.raises(ValueError, match=r"^cannot matmul shapes \(2, 3\) and \(2, 3\)$"):
        A @ A
    with pytest.raises(ValueError, match=r"^cannot matmul shapes \(3,\) and \(3, 2\)$"):
        ct.input("v", (3,)) @ ct.input("B", (3, 2))
    with pytest.raises(ValueError, match=r"^cannot matmul shapes \(2, 3\) and \(3,\)$"):
        ct.matmul(A, ct.input("v", (3,)))
    with pytest.raises(ValueError, match=r"^cannot matmul shapes \(\) and \(2, 3\)$"):
        ct.matmul(ct.input("s", ()), A)


def test_matmul_with_a_non_tensor_is_a_type_error():
    A = ct.input("A", (2, 2))
    with pytest.raises(TypeError):
        A @ 2.0
    with pytest.raises(TypeError):
        2.0 @ A
    with pytest.raises(TypeError):
        A @ np.eye(2, dtype=np.float32)
    with pytest.raises(TypeError):
        np.eye(2, dtype=np.float32) @ A
    with pytest.raises(TypeError):
        ct.matmul(A, 2.0)


def test_matmul_depth_limit_is_1000():
    eye = np.eye(2, dtype=np.float32)
    w = ct.input("w", (2, 2))
    y = ct.input("x", (2, 2))
    for _ in range(999):  # the input is level 1, so this makes level 1000
        y = y @ w
    x = f32([[1, 2], [3, 4]])
    np.testing.assert_array_equal(ct.run(y, x=x, w=eye), x)
    with pytest.raises(ValueError, match="nested more than 1000 levels"):
        y @ w
    with pytest.raises(ValueError, match="nested more than 1000 levels"):
        ct.matmul(w, y)
