"""Axes of sum, max and softmax: negative spellings and every out-of-range form.

A negative axis counts from the end, as in NumPy, and must give bit-identical results to
the matching positive one. Every rejected axis is a ValueError raised at op time that
states the tensor's rank and the allowed axes, the absence of axes at rank 0, or that the
selected axis is empty.
"""
import re
import sys

import numpy as np
import pytest

import coffee_ten as ct

OPS = [ct.sum, ct.max, ct.softmax]
T = (np.arange(24, dtype=np.float32).reshape(2, 3, 4) * np.float32(0.37)) - np.float32(3.1)


@pytest.mark.parametrize("op", OPS, ids=lambda op: op.__name__)
@pytest.mark.parametrize("axis", range(3))
def test_negative_axes_match_their_positive_spelling_bit_for_bit(op, axis):
    x = ct.input("t", T.shape)
    positive, negative = op(x, axis=axis), op(x, axis=axis - 3)
    assert positive.shape == negative.shape
    assert ct.run(positive, t=T).tobytes() == ct.run(negative, t=T).tobytes()


@pytest.mark.parametrize("op", OPS, ids=lambda op: op.__name__)
def test_default_axis_is_the_last(op):
    x = ct.input("t", T.shape)
    assert ct.run(op(x), t=T).tobytes() == ct.run(op(x, axis=2), t=T).tobytes()


@pytest.mark.parametrize("op", OPS, ids=lambda op: op.__name__)
@pytest.mark.parametrize("shape, axis, spelled", [
    ((4,), 1, "axis 1"),
    ((4,), -2, "axis -2"),
    ((2, 3, 4), 3, "axis 3"),
    ((2, 3, 4), -4, "axis -4"),
    ((2, 3, 4), 2**40, f"axis {2**40}"),
    ((2, 3, 4), -sys.maxsize - 1, f"axis {-sys.maxsize - 1}"),
    ((2, 3, 4), sys.maxsize, f"axis {sys.maxsize}"),
])
def test_out_of_range_axes_state_the_rank_and_the_allowed_axes(op, shape, axis, spelled):
    rank = len(shape)
    message = f"cannot reduce shape {ct.input('x', shape).shape} over {spelled}: rank {rank} allows axes {-rank}..={rank - 1}"
    with pytest.raises(ValueError, match="^" + re.escape(message) + "$"):
        op(ct.input("x", shape), axis=axis)


@pytest.mark.parametrize("op", OPS, ids=lambda op: op.__name__)
@pytest.mark.parametrize("axis, spelled", [(-1, "its last axis"), (0, "axis 0"), (5, "axis 5")])
def test_rank_0_has_no_axes(op, axis, spelled):
    message = f"cannot reduce shape () over {spelled}: a rank 0 tensor has no axes"
    with pytest.raises(ValueError, match="^" + re.escape(message) + "$"):
        op(ct.input("s", ()), axis=axis)


@pytest.mark.parametrize("op", OPS, ids=lambda op: op.__name__)
def test_empty_selected_axis_is_named_and_other_empty_axes_are_kept(op):
    e = ct.input("e", (2, 0, 3))
    for axis, spelled in [(1, "axis 1"), (-2, "axis -2")]:
        message = f"cannot reduce shape (2, 0, 3) over {spelled}: that axis has size 0"
        with pytest.raises(ValueError, match="^" + re.escape(message) + "$"):
            op(e, axis=axis)
    assert op(e, axis=0).shape == ((2, 0, 3) if op is ct.softmax else (1, 0, 3))


@pytest.mark.parametrize("op", OPS, ids=lambda op: op.__name__)
def test_axis_must_be_an_integer(op):
    for axis in (1.0, "1", None):
        with pytest.raises(TypeError):
            op(ct.input("x", (2, 3)), axis=axis)
    with pytest.raises(OverflowError):
        op(ct.input("x", (2, 3)), axis=2**64)
