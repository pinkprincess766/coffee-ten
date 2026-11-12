"""N18: current-operation errors are raised before C emission or compiler startup."""
import math
import re
import struct

import numpy as np
import pytest

import coffee_ten as ct


@pytest.mark.parametrize("op", [ct.sum, ct.max, ct.softmax])
@pytest.mark.parametrize("shape,axis", [((2, 3), 2), ((2, 3), -3), ((), 0), ((2, 0, 3), 1)])
def test_axis_errors_are_builder_errors_without_a_compiler(op, shape, axis, monkeypatch):
    monkeypatch.setenv("PATH", "")
    with pytest.raises(ValueError, match="cannot reduce shape"):
        op(ct.input("x", shape), axis=axis)


def test_incompatible_dimensions_fail_before_compile(monkeypatch):
    monkeypatch.setenv("PATH", "")
    a = ct.input("a", (2, 3))
    with pytest.raises(ValueError, match=r"cannot matmul shapes \(2, 3\) and \(4, 2\)"):
        a @ ct.input("b", (4, 2))
    with pytest.raises(ValueError, match="cannot broadcast shapes"):
        a + ct.input("b", (4,))


def test_impossible_dense_shapes_fail_at_input_and_op_time(monkeypatch):
    monkeypatch.setenv("PATH", "")
    # Each input fits usize; the broadcast/matmul output has 2**pointer_bits elements.
    extent = 1 << (struct.calcsize("P") * 4)
    shape = (extent, extent)
    message = "shape " + re.escape(str(shape)) + " has too many elements"
    with pytest.raises(ValueError, match=message):
        ct.input("huge", shape)
    a = ct.input("a", (extent, 1))
    b = ct.input("b", (1, extent))
    with pytest.raises(ValueError, match=message):
        a @ b
    with pytest.raises(ValueError, match=message):
        a + b
    # Empty shapes and scalars retain their existing contracts; no allocation occurs.
    assert ct.input("empty", (extent, extent, 0)).shape == (extent, extent, 0)
    assert ct.input("scalar", ()).shape == ()


def test_duplicate_names_fail_before_c_or_cache_creation(tmp_path, monkeypatch):
    first, second = ct.input("x", (2,)), ct.input("x", (2,))
    output = first + second
    cache = tmp_path / "never-created"
    monkeypatch.setenv("PATH", "")
    with pytest.raises(ValueError, match="^duplicate input x$"):
        ct.loop_ir(output)
    with pytest.raises(ValueError, match="^duplicate input x$"):
        ct.compile(output, cache_dir=cache)
    assert not cache.exists()


def test_reusing_one_tensor_is_valid_and_shape_checks_keep_bits():
    x = ct.input("x", (2, 3))
    y = ct.sum(x + x, axis=-2)
    assert y.shape == (1, 3)
    assert ct.loop_ir(y).splitlines()[0].count("x: f32[2, 3]") == 1
    values = np.array([[1, -0.0, 3], [-1, -0.0, -3]], dtype=np.float32)
    out = ct.run(y, x=values)
    expected = np.zeros((1, 3), dtype=np.float32)
    for row in values + values:
        expected = expected + row
    np.testing.assert_array_equal(out.view(np.uint32), expected.view(np.uint32))


def test_unrepresentable_f32_buffers_fail_at_build_time(monkeypatch):
    monkeypatch.setenv("PATH", "")
    # Only metadata is constructed: no input arrays, C compiler or dense buffers.
    max_bytes = (1 << (struct.calcsize("P") * 8 - 1)) - 1
    rows = max_bytes // 4 + 1
    with pytest.raises(ValueError, match="has too many elements"):
        ct.input("huge", (rows,))
    a, b = ct.input("a", (rows, 0)), ct.input("b", (0, 1))
    with pytest.raises(ValueError, match="has too many elements"):
        a @ b
    extent = math.isqrt(max_bytes // 4) + 1
    left, right = ct.input("left", (extent, 1)), ct.input("right", (1, extent))
    for operation in (lambda: left @ right, lambda: left + right):
        with pytest.raises(ValueError, match="has too many elements"):
            operation()
    assert ct.input("empty", (rows, rows, 0)).shape == (rows, rows, 0)
    assert ct.input("scalar", ()).shape == ()
