"""N7: the same bounded corpus through the Python builder and native compilation."""
from pathlib import Path
import os
import sys
import time

import numpy as np
import pytest
import coffee_ten as ct

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tests" / "fuzz"))
from ct_fuzz_cases import Case, build, generate, numpy_inputs, read_cases, render  # noqa: E402

CORPUS = ROOT / "tests" / "fuzz" / "corpus-v1.tsv"


def test_corpus_regenerates_exactly_with_its_seed_and_bounds():
    assert CORPUS.read_text() == render()
    cases = read_cases(CORPUS.read_text())
    assert len(cases) == 32
    assert len({c.kind for c in cases}) == 4
    assert {c.pre for c in cases} == {0, 1, 2, 3}
    assert {c.post for c in cases} == {0, 1, 2, 3}
    for kind in {c.kind for c in cases}:
        assert {c.pre for c in cases if c.kind == kind} == {0, 1, 2, 3}
        assert {c.post for c in cases if c.kind == kind} == {0, 1, 2, 3}
    assert {c.axis for c in cases if c.kind == "reduce"} == {0, 1, 2}
    assert render(17, 8) == render(17, 8)
    assert render(17, 8) != render(18, 8)
    assert list(generate(0, 4))
    with pytest.raises(ValueError):
        list(generate(1, 129))
    with pytest.raises(ValueError):
        Case.parse(cases[0].row().replace("2,2,4", "9,2,4"))
    with pytest.raises(ValueError):
        read_cases(cases[0].row() + "\n" + cases[0].row())


def test_seeded_generated_programs_match_reference_bits(tmp_path):
    replay = os.environ.get("COFFEE_TEN_FUZZ_CASE")
    cases = [Case.parse(replay)] if replay else read_cases(CORPUS.read_text())
    start = time.monotonic()
    for case in cases:
        print(f"N7 replay TSV: {case.row()}")
        inputs = numpy_inputs(case)
        original_bits = {name: x.view(np.uint32).copy() for name, x in inputs.items()}
        assert all(x.size <= 64 and np.isfinite(x).all() for x in inputs.values())
        expression = build(case, ct)
        reference = ct.run(expression, **inputs)
        kernel = ct.compile(expression, cache_dir=tmp_path)
        actual = kernel(**inputs)
        assert kernel.order_class == "preserving"
        different = (actual.dtype != reference.dtype or actual.shape != reference.shape
                     or not np.array_equal(actual.view(np.uint32), reference.view(np.uint32)))
        if different:
            from ct_fuzz_minimize import python_mismatch, save_failure
            directory = Path(os.environ.get("COFFEE_TEN_FUZZ_FAILURE_DIR", tmp_path)) / f"python-{case.index}"
            try:
                save_failure(case, lambda candidate: python_mismatch(candidate, ct, tmp_path / "minimize-cache"),
                             directory, backend="python")
            except Exception as error:
                print(f"N9 minimizer stopped; original retained: {error}")
        assert actual.dtype == reference.dtype == np.float32
        assert actual.shape == reference.shape, case.row()
        np.testing.assert_array_equal(actual.view(np.uint32), reference.view(np.uint32),
                                      err_msg=f"replay TSV: {case.row()}\n{kernel.loop_ir}")
        for name, values in inputs.items():
            np.testing.assert_array_equal(values.view(np.uint32), original_bits[name],
                                          err_msg=f"input {name} changed: {case.row()}")
        assert time.monotonic() - start < 120, f"N7 CI time budget exceeded: {case.row()}"
