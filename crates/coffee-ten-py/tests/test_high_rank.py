"""NumPy 2 supports up to 64 axes; copying must not use a 32-axis view."""

import numpy as np
import pytest

import coffee_ten as ct


pytestmark = pytest.mark.skipif(
    np.lib.NumpyVersion(np.__version__) < "2.0.0",
    reason="NumPy 1 limits ndarrays to 32 axes",
)


def high_rank_values(rank, layout):
    bits = np.array([0x80000000, 0, 0x7F800000, 0xFF800000, 0x7FC12345, 0x7FA12345],
                    dtype=np.uint32)
    values = bits.view(np.float32).reshape((1,) * (rank - 2) + (2, 3))
    if layout == "transposed":
        values = values.swapaxes(-2, -1)
    elif layout == "reversed":
        values = values[..., ::-1]
    elif layout == "broadcast":
        values = np.broadcast_to(values[..., :1, :], values.shape)
    return values


@pytest.mark.parametrize("rank", [33, 64])
@pytest.mark.parametrize("layout", ["contiguous", "transposed", "reversed", "broadcast"])
@pytest.mark.parametrize("compiled", [False, True])
def test_high_rank_input_preserves_shape_and_bits(rank, layout, compiled, tmp_path):
    values = high_rank_values(rank, layout)
    output = ct.input("x", values.shape)
    if compiled:
        actual = ct.compile(output, cache_dir=tmp_path)(x=values)
    else:
        actual = ct.run(output, x=values)
    assert actual.shape == values.shape
    np.testing.assert_array_equal(actual.view(np.uint32), values.copy(order="C").view(np.uint32))


@pytest.mark.parametrize("rank", [33, 64])
def test_empty_high_rank_input_preserves_shape(rank):
    shape = (1,) * (rank - 1) + (0,)
    actual = ct.run(ct.input("x", shape), x=np.empty(shape, dtype=np.float32))
    assert actual.shape == shape
    assert actual.dtype == np.float32
