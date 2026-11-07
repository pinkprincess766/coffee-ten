"""Read NumPy byte layouts without changing their float32 bit patterns."""

import numpy as np
import pytest

import coffee_ten as ct


def layout(kind):
    backing = np.arange(64, dtype=np.uint8)
    if kind == "positive_byte_strides":
        return np.ndarray((2, 3), dtype=np.float32, buffer=backing, strides=(5, 1))
    if kind == "negative_byte_strides":
        return np.ndarray((2, 3), dtype=np.float32, buffer=backing, offset=7,
                          strides=(-5, -1))
    if kind == "unaligned_c":
        return np.ndarray((2, 3), dtype=np.float32, buffer=backing, offset=1)
    if kind == "unaligned_f":
        return np.ndarray((2, 3), dtype=np.float32, buffer=backing, offset=1,
                          strides=(4, 8))
    if kind == "unaligned_special_bits":
        values = np.ndarray((2, 3), dtype=np.float32, buffer=backing, offset=1)
        values.view(np.uint32)[:] = [[0x80000000, 0, 0x7F800000],
                                    [0xFF800000, 0x7FC12345, 0x7FA12345]]
        return values
    raise AssertionError(kind)


@pytest.mark.parametrize("kind", ["positive_byte_strides", "negative_byte_strides",
                                  "unaligned_c", "unaligned_f", "unaligned_special_bits"])
@pytest.mark.parametrize("compiled", [False, True])
def test_input_byte_layout_preserves_bits(kind, compiled, tmp_path):
    values = layout(kind)
    output = ct.input("x", values.shape)
    expected = values.copy(order="C")
    if compiled:
        actual = ct.compile(output, cache_dir=tmp_path)(x=values)
    else:
        actual = ct.run(output, x=values)
    assert actual.shape == values.shape
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def test_empty_unaligned_array_needs_no_data_view():
    values = np.ndarray((0, 3), dtype=np.float32,
                        buffer=np.arange(5, dtype=np.uint8), offset=1)
    assert values.__array_interface__["data"][0] % 4 == 1
    actual = ct.run(ct.input("x", values.shape), x=values)
    assert actual.shape == (0, 3)
    assert actual.dtype == np.float32


def test_kernel_reports_empty_unaligned_input_shape(tmp_path):
    values = np.ndarray((0,), dtype=np.float32,
                        buffer=np.arange(5, dtype=np.uint8), offset=1)
    kernel = ct.compile(ct.input("x", (1,)), cache_dir=tmp_path)
    with pytest.raises(ValueError, match=r"input x has shape \(0,\), expected \(1,\)"):
        kernel(x=values)
