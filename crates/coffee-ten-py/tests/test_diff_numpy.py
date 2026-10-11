"""Differential tests of `import coffee_ten as ct` against NumPy, with Hypothesis.

Two generators: elementwise expression trees (compared exactly) and matmul with a
bias and relu epilogue (compared to a float64 reference within a derived bound).
The same generators also check the compiled path against `ct.run`, bit for bit, with
fewer examples because every example runs clang.
Settings are derandomized, so CI runs the same examples every time.
"""

import hypothesis.extra.numpy as hnp
import numpy as np
import pytest
from hypothesis import Phase, example, given, settings
from hypothesis import strategies as st

import coffee_ten as ct

SETTINGS = settings(derandomize=True, max_examples=200, deadline=None, database=None)
COMPILED_SETTINGS = settings(SETTINGS, max_examples=40)


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory):
    # Module scoped, so Hypothesis may use it (a function-scoped fixture is refused).
    return tmp_path_factory.mktemp("kernel-cache")


def assert_same_bits(actual, expected):
    assert actual.dtype == np.float32
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def f32_values(limit):
    # Hypothesis wants bounds that are float32 values: the largest one within the limit.
    bound = np.float32(limit)
    if float(bound) > limit:
        bound = np.nextafter(bound, np.float32(0))
    return st.floats(
        min_value=-float(bound),
        max_value=float(bound),
        width=32,
        allow_nan=False,
        allow_infinity=False,
    )


def f32_arrays(shape, elements):
    return hnp.arrays(np.float32, shape, elements=elements)


# --- Elementwise trees ---------------------------------------------------------
#
# A tree is ("in", i), ("const", c), (unary_name, x) or (binary_name, a, b).
# A constant is a bare Python float on the ct side, so it never sits alone under an
# op and never as both operands of one: there must be a Tensor to carry the op.

UNARY = ("neg", "relu")
BINARY = ("add", "sub", "mul", "max")
MAX_TREE_DEPTH = 3
ELEMENTWISE_LIMIT = 1e4


@st.composite
def elementwise_cases(draw):
    """(tree, arrays): a tree over `len(arrays)` inputs and their float32 values."""
    n_inputs = 0

    def node(depth, allow_const):
        nonlocal n_inputs
        kind = "leaf" if depth == 0 else draw(st.sampled_from(["leaf", "unary", "binary", "binary"]))
        if kind == "unary":
            return (draw(st.sampled_from(UNARY)), node(depth - 1, False))
        if kind == "binary":
            op = draw(st.sampled_from(BINARY))
            left = node(depth - 1, True)
            right = node(depth - 1, left[0] != "const")
            return (op, left, right)
        choices = ["new", "const"] if allow_const else ["new"]
        if n_inputs > 0:
            choices.append("old")
        choice = draw(st.sampled_from(choices))
        if choice == "const":
            return ("const", draw(f32_values(ELEMENTWISE_LIMIT)))
        if choice == "old":
            return ("in", draw(st.integers(0, n_inputs - 1)))
        n_inputs += 1
        return ("in", n_inputs - 1)

    tree = node(MAX_TREE_DEPTH, False)
    shapes = draw(
        hnp.mutually_broadcastable_shapes(
            num_shapes=n_inputs, min_dims=0, max_dims=3, min_side=1, max_side=4
        )
    ).input_shapes
    arrays = [draw(f32_arrays(shape, f32_values(ELEMENTWISE_LIMIT))) for shape in shapes]
    return tree, arrays


def build_ct(tree, inputs):
    tag = tree[0]
    if tag == "in":
        return inputs[tree[1]]
    if tag == "const":
        return tree[1]
    if tag == "neg":
        return -build_ct(tree[1], inputs)
    if tag == "relu":
        return ct.relu(build_ct(tree[1], inputs))
    a, b = build_ct(tree[1], inputs), build_ct(tree[2], inputs)
    if tag == "max":
        return ct.maximum(a, b)
    return {"add": a + b, "sub": a - b, "mul": a * b}[tag]


def build_np(tree, arrays):
    tag = tree[0]
    if tag == "in":
        return arrays[tree[1]]
    if tag == "const":
        return np.float32(tree[1])
    if tag == "neg":
        return -build_np(tree[1], arrays)
    if tag == "relu":
        return np.maximum(build_np(tree[1], arrays), np.float32(0))
    a, b = build_np(tree[1], arrays), build_np(tree[2], arrays)
    if tag == "max":
        return np.maximum(a, b)
    return {"add": a + b, "sub": a - b, "mul": a * b}[tag]


@SETTINGS
@given(case=elementwise_cases())
def test_elementwise_tree_matches_numpy_exactly(case):
    # Exact comparison is justified because |x| <= 1e4 and depth <= 3 keep every
    # intermediate below 1e32 (three multiplications at most), so no inf or NaN
    # appears. Then add, sub and mul are correctly rounded float32 operations in both
    # implementations, and neg is exact, so the results are the same bits. Our relu and
    # max differ from NumPy's only in the sign of a zero and on NaN; assert_array_equal
    # treats -0.0 == +0.0, and NaN cannot occur.
    tree, arrays = case
    inputs = [ct.input(f"x{i}", array.shape) for i, array in enumerate(arrays)]
    ours = ct.run(build_ct(tree, inputs), **{f"x{i}": a for i, a in enumerate(arrays)})
    expected = np.asarray(build_np(tree, arrays))
    assert expected.dtype == np.float32
    assert ours.dtype == np.float32
    assert ours.shape == expected.shape
    np.testing.assert_array_equal(ours, expected)


@COMPILED_SETTINGS
@given(case=elementwise_cases())
def test_compiled_elementwise_tree_equals_the_interpreter_bit_for_bit(case, cache_dir):
    # Both do the same f32 operations in the same order, so even the sign of a zero agrees.
    tree, arrays = case
    inputs = [ct.input(f"x{i}", array.shape) for i, array in enumerate(arrays)]
    y = build_ct(tree, inputs)
    values = {f"x{i}": a for i, a in enumerate(arrays)}
    assert_same_bits(ct.compile(y, cache_dir=cache_dir)(**values), ct.run(y, **values))


# --- Matmul with epilogue --------------------------------------------------------
#
# Tolerance: Higham, Accuracy and Stability of Numerical Algorithms, inner products.
# Our dot product is a left-to-right float32 sum of float32 products (not fused), for
# any order |fl(x.y) - x.y| <= gamma_n * sum|x_i y_i| with gamma_n = n*u / (1 - n*u),
# u = 2**-24. The bias is one more addition, the (K+1)-th term, so the factor is
# gamma_{K+1} and the sum of magnitudes gets |bias| added. A product that falls in the
# subnormal range has an absolute rounding error of at most 2**-150 instead of a
# relative one, so each of the K products adds up to 2**-150, bounded here by
# 2**-149. The whole bound is doubled as a margin. relu is 1-Lipschitz, so the same
# tolerance holds after it. The reference is float64, where a product of two float32
# values is exact and the remaining error is about 2**29 times smaller than the bound.

U = 2.0**-24
MATMUL_LIMIT = 1e3
TINY_LIMIT = 1e-30


def gamma(n):
    return n * U / (1 - n * U)


# Mostly ordinary values, sometimes tiny ones so that products underflow.
MATMUL_ELEMENTS = st.one_of(f32_values(MATMUL_LIMIT), f32_values(TINY_LIMIT))


@st.composite
def matmul_cases(draw):
    """(a, b, bias, relu): a (M, K), b (K, N), bias of shape (N,), (M, 1), () or None."""
    m, k, n = (draw(st.integers(1, 8)) for _ in range(3))
    a = draw(f32_arrays((m, k), MATMUL_ELEMENTS))
    b = draw(f32_arrays((k, n), MATMUL_ELEMENTS))
    bias_shape = draw(st.sampled_from([None, (n,), (m, 1), ()]))
    bias = None if bias_shape is None else draw(f32_arrays(bias_shape, MATMUL_ELEMENTS))
    return a, b, bias, draw(st.booleans())


def build_matmul_case(case):
    """Returns (expression, input values) of a matmul case."""
    a, b, bias, apply_relu = case
    out = ct.input("a", a.shape) @ ct.input("b", b.shape)
    values = {"a": a, "b": b}
    if bias is not None:
        out = out + ct.input("bias", bias.shape)
        values["bias"] = bias
    if apply_relu:
        out = ct.relu(out)
    return out, values


def evaluate_matmul_case(case):
    """Returns (ours, reference, tolerance), all with the shape (M, N)."""
    a, b, bias, apply_relu = case
    k = a.shape[1]
    out, values = build_matmul_case(case)
    ref = a.astype(np.float64) @ b.astype(np.float64)
    bound_scale = np.abs(a).astype(np.float64) @ np.abs(b).astype(np.float64)
    if bias is not None:
        ref = ref + bias.astype(np.float64)
        bound_scale = bound_scale + np.abs(bias).astype(np.float64)
    if apply_relu:
        ref = np.maximum(ref, 0.0)
    tol = 2 * (gamma(k + 1) * bound_scale + k * 2.0**-149)
    return ct.run(out, **values), ref, tol


def assert_matmul_case(case):
    ours, ref, tol = evaluate_matmul_case(case)
    assert ours.dtype == np.float32
    assert ours.shape == ref.shape
    assert np.all(np.abs(ours.astype(np.float64) - ref) <= tol)


@SETTINGS
@given(case=matmul_cases())
def test_matmul_with_epilogue_matches_float64_within_bound(case):
    assert_matmul_case(case)


@COMPILED_SETTINGS
@given(case=matmul_cases())
def test_compiled_matmul_with_epilogue_equals_the_interpreter_bit_for_bit(case, cache_dir):
    # Same canonical order of the K sum in both: acc = 0; acc += a[i, k] * b[k, j].
    out, values = build_matmul_case(case)
    assert_same_bits(ct.compile(out, cache_dir=cache_dir)(**values), ct.run(out, **values))


# K = 1 (a single product, no accumulation) and a bias of shape (); hand values.
@settings(phases=[Phase.explicit], database=None)
@given(case=matmul_cases())
@example(
    case=(
        np.array([[3.0], [-2.0]], dtype=np.float32),
        np.array([[4.0, -5.0, 0.5]], dtype=np.float32),
        np.array(1.0, dtype=np.float32),
        True,
    )
)
def test_matmul_k1_with_scalar_bias_regression(case):
    assert_matmul_case(case)
    ours, _, _ = evaluate_matmul_case(case)
    # [[12, -15, 1.5], [-8, 10, -1]] + 1, then relu.
    expected = np.array([[13.0, 0.0, 2.5], [0.0, 11.0, 0.0]], dtype=np.float32)
    np.testing.assert_array_equal(ours, expected)
