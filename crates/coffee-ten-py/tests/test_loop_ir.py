"""Tests of `ct.loop_ir` and `Kernel.loop_ir`: the loop IR text, with and without a schedule.

The text is the loop crate's `Display`. The exact form is pinned for one kernel; a schedule
is checked by the names and guard it adds.
"""

import pytest

import coffee_ten as ct

MATMUL_LOOP_IR = """\
kernel kernel(A: f32[4, 3], B: f32[3, 2], out: f32[4, 2])
for i in 0..4:
  for j in 0..2:
    out[i, j] = 0.0
for i in 0..4:
  for k in 0..3:
    for j in 0..2:
      out[i, j] = out[i, j] + (A[i, k] * B[k, j])
"""


def matmul_expr():
    return ct.input("A", (4, 3)) @ ct.input("B", (3, 2))


def test_matmul_loop_ir_is_exact():
    assert ct.loop_ir(matmul_expr()) == MATMUL_LOOP_IR


def test_split_names_the_outer_and_inner_loops_and_the_guard():
    text = ct.loop_ir(matmul_expr(), schedule="split(1, k, ko, ki, 2)")
    assert "for ko in 0..2:" in text
    assert "for ki in 0..2:" in text
    assert "if 2*ko + ki < 3:" in text


def test_compiled_kernel_stores_the_loop_ir_of_its_expression(tmp_path):
    y = ct.relu(matmul_expr() + ct.input("bias", (2,)))
    kernel = ct.compile(y, cache_dir=tmp_path)
    assert kernel.loop_ir == ct.loop_ir(y)


def test_compiled_kernel_stores_the_loop_ir_after_the_schedule(tmp_path):
    y = matmul_expr()
    schedule = "split(1, k, ko, ki, 2)"
    kernel = ct.compile(y, schedule=schedule, cache_dir=tmp_path)
    assert kernel.loop_ir == ct.loop_ir(y, schedule=schedule)
    assert "for ko in 0..2:" in kernel.loop_ir


def test_bad_schedule_text_is_a_value_error():
    with pytest.raises(ValueError, match="byte"):
        ct.loop_ir(matmul_expr(), schedule="split(!")


def test_illegal_schedule_step_is_a_value_error():
    schedule = "split(1, k, ko, ki, 2); reorder(1, i, ki)"  # not a permutation of the loops
    with pytest.raises(ValueError, match="step 1"):
        ct.loop_ir(matmul_expr(), schedule=schedule)
    with pytest.raises(ValueError, match="nest"):
        ct.loop_ir(matmul_expr(), schedule="split(9, k, ko, ki, 2)")


def test_graph_error_is_a_value_error():
    with pytest.raises(ValueError, match="duplicate input A"):
        ct.loop_ir(ct.input("A", (2,)) + ct.input("A", (2,)))


def test_loop_ir_writes_no_files_to_the_default_cache(tmp_path, monkeypatch):
    # The default cache is the temp dir's coffee-ten-cache. A kernel cached by an earlier
    # run would hide a write there, so TMPDIR points the default cache at an empty dir.
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    ct.loop_ir(matmul_expr(), schedule="split(1, k, ko, ki, 2)")
    assert not (tmp_path / "coffee-ten-cache").exists()
