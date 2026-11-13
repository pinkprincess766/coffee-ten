"""Builder limits exactly at the boundary and one past it.

Depth: an input is level 1; every op is one more than its deepest operand; constants and
reused subtrees add nothing. Level 1000 (MAX_DEPTH) builds and runs, 1001 is a ValueError
at op time. Size: a shape is allowed while its float32 bytes fit in isize::MAX; one more
element, or a product that overflows usize, is a ValueError. Empty shapes are always allowed.
Lowering (loop_ir, compile) adds its own limit of 1000 tree nodes, where constants count and
a reused subtree counts at every use. compile() is further bounded by clang's bracket depth
(256), which a deep chain of nested C expressions reaches first. The limits themselves are
unchanged; these tests only pin where they apply.
"""
import struct

import numpy as np
import pytest

import coffee_ten as ct

DEPTH_ERROR = "^expression is nested more than 1000 levels deep$"
POINTER_BITS = struct.calcsize("P") * 8
MAX_BYTES = (1 << (POINTER_BITS - 1)) - 1  # isize::MAX
MAX_ELEMENTS = MAX_BYTES // 4


def chain(level, name="x", shape=(2,)):
    """A Tensor at exactly `level`: the input plus level - 1 additions of a constant."""
    y = ct.input(name, shape)
    for _ in range(level - 1):
        y = y + 1.0
    return y


def test_depth_is_one_more_than_the_deepest_operand_on_either_side():
    deep, shallow = chain(999), ct.input("s", (2,))
    for build in (lambda a, b: a + b, lambda a, b: a - b, lambda a, b: a * b,
                  lambda a, b: a / b, ct.maximum):
        build(deep, shallow)  # level 1000
        build(shallow, deep)
        with pytest.raises(ValueError, match=DEPTH_ERROR):
            build(chain(1000), shallow)
        with pytest.raises(ValueError, match=DEPTH_ERROR):
            build(shallow, chain(1000))


def test_constants_and_reuse_do_not_add_depth():
    y = chain(999)
    for _ in range(3):
        z = y + y  # one level above y, however often y is reused
    assert z.shape == (2,)
    with pytest.raises(ValueError, match=DEPTH_ERROR):
        z + z
    # Constant operands sit at level 1, so they never make an op deeper.
    ct.maximum(2.0, y)
    with pytest.raises(ValueError, match=DEPTH_ERROR):
        ct.maximum(2.0, z)


def test_matmul_depth_counts_the_deeper_side():
    deep = ct.input("x", (2, 2))
    for _ in range(998):
        deep = deep + 1.0
    w = ct.input("w", (2, 2))
    deep @ w  # level 1000
    w @ deep
    with pytest.raises(ValueError, match=DEPTH_ERROR):
        (deep + 1.0) @ w
    with pytest.raises(ValueError, match=DEPTH_ERROR):
        ct.matmul(w, deep + 1.0)


def test_softmax_needs_five_levels():
    # max, subtract, exp, sum, divide: softmax of a level-d Tensor is level d + 5.
    assert ct.softmax(chain(995)).shape == (2,)
    with pytest.raises(ValueError, match=DEPTH_ERROR):
        ct.softmax(chain(996))


def negations(level):
    """A level-`level` chain of `level - 1` negations: `level` nodes, no constants."""
    y = ct.input("x", (2,))
    for _ in range(level - 1):
        y = -y
    return y


def test_a_level_1000_expression_runs_and_lowers(tmp_path):
    data = np.array([0.5, -2.0], np.float32)
    # Interpreter: any level-1000 expression runs.
    expected = data.copy()
    for _ in range(999):
        expected = expected + np.float32(1.0)
    assert ct.run(chain(1000), x=data).tobytes() == expected.tobytes()
    # Lowering also has MAX_EXPR_NODES = 1000 tree nodes. A level-1000 chain without
    # constants has exactly 1000 nodes, so loop_ir accepts it.
    y = negations(1000)
    assert ct.loop_ir(y)
    assert ct.run(y, x=data).tobytes() == (-data).tobytes()  # 999 negations
    # compile() is not tried at level 1000: the C has one parenthesis level per negation,
    # and clang refuses more than 256 (-fbracket-depth). A level-200 chain compiles and
    # matches the interpreter bit for bit.
    y = negations(200)
    kernel = ct.compile(y, cache_dir=tmp_path)
    assert kernel(x=data).tobytes() == ct.run(y, x=data).tobytes() == (-data).tobytes()


def test_lowering_counts_tree_nodes_not_depth(tmp_path):
    # Each constant is a node, so the level-1000 chain of "+ 1.0" has 1999 nodes.
    message = "^expression has 1999 nodes, limit 1000$"
    with pytest.raises(ValueError, match=message):
        ct.loop_ir(chain(1000))
    with pytest.raises(ValueError, match=message):
        ct.compile(chain(1000), cache_dir=tmp_path)
    # A reused subtree is one level for the depth limit but is counted at every use by
    # lowering: doubling 8 times is 511 nodes, 9 times 1023, at only level 10.
    y = ct.input("x", (2,))
    for _ in range(8):
        y = y + y
    assert ct.loop_ir(y)
    with pytest.raises(ValueError, match="^expression has 1023 nodes, limit 1000$"):
        ct.loop_ir(y + y)
    data = np.array([1.0, -3.0], np.float32)
    assert ct.run(y + y, x=data).tobytes() == (data * np.float32(512)).tobytes()


def test_input_size_boundary():
    assert ct.input("edge", (MAX_ELEMENTS,)).shape == (MAX_ELEMENTS,)
    with pytest.raises(ValueError, match="has too many elements$"):
        ct.input("past", (MAX_ELEMENTS + 1,))
    # Split across two dimensions: the largest multiple of 7 that fits is allowed, one more
    # row is not.
    rows = MAX_ELEMENTS // 7
    ct.input("edge2", (rows, 7))
    with pytest.raises(ValueError, match="has too many elements$"):
        ct.input("past2", (rows + 1, 7))
    # A product that overflows usize before the byte count is checked.
    half = 1 << (POINTER_BITS // 2)
    with pytest.raises(ValueError, match="has too many elements$"):
        ct.input("wrap", (half, half))
    # Any zero dimension makes the shape empty, whatever the other extents.
    assert ct.input("empty", (half, half, 0)).shape == (half, half, 0)
    assert ct.input("empty-first", (0, MAX_ELEMENTS + 1)).shape == (0, MAX_ELEMENTS + 1)


def test_op_results_at_the_size_boundary():
    # Broadcasting (n, 1) with (1, m): allowed while n * m fits, refused one row past it.
    m = 8
    n = MAX_ELEMENTS // m
    assert (ct.input("a", (n, 1)) + ct.input("b", (1, m))).shape == (n, m)
    with pytest.raises(ValueError, match="has too many elements$"):
        ct.input("a", (n + 1, 1)) + ct.input("b", (1, m))
    # Matmul (n, k) @ (k, m) with the same result shapes.
    assert (ct.input("a", (n, 1)) @ ct.input("b", (1, m))).shape == (n, m)
    with pytest.raises(ValueError, match="has too many elements$"):
        ct.input("a", (n + 1, 1)) @ ct.input("b", (1, m))
