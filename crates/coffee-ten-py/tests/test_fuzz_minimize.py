"""N9 regression: an injected native-output error is detected and genuinely shrunk."""
from pathlib import Path
import json
import sys

import numpy as np
import pytest
import coffee_ten as ct

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tests" / "fuzz"))
from ct_fuzz_cases import generate, Case  # noqa: E402
from ct_fuzz_minimize import minimize, python_mismatch, save_failure, updated  # noqa: E402


def test_injected_output_mismatch_is_saved_and_minimized(tmp_path):
    case = updated(next(generate()), dims=(4, 4, 4), pre=1, post=0)
    def injected(candidate, actual):
        if candidate.kind == "matmul" and candidate.dims[1] > 1:
            actual = actual.copy()
            actual.flat[0] = np.finfo(np.float32).tiny
        return actual
    def fails(candidate):
        return python_mismatch(candidate, ct, tmp_path / "cache", injected)
    assert fails(case)
    best = save_failure(case, fails, tmp_path / "failure", backend="python-injected-test")
    assert best.seed == case.seed
    assert best.dims == (1, 2, 1)
    assert best.pre == best.post == best.axis == 0
    assert fails(best)
    assert not fails(updated(best, dims=(1, 1, 1)))
    report = json.loads((tmp_path / "failure" / "report.json").read_text())
    assert report["verified"] is True and report["status"] == "fixed_point"
    assert report["probes"] <= 64
    assert Case.parse((tmp_path / "failure" / "minimal.tsv").read_text()).row() == best.row()
    assert (tmp_path / "failure" / "original.tsv").read_text() == case.row() + "\n"


def test_budget_and_unreproduced_failure_are_explicit(tmp_path):
    case = next(generate())
    best, probes, status = minimize(case, lambda _: True, max_probes=2)
    assert best == case and probes == 2 and status == "probe_budget"
    save_failure(case, lambda _: True, tmp_path / "not-reproduced", backend="test", max_probes=2)
    with pytest.raises(ValueError, match="does not reproduce"):
        save_failure(case, lambda _: False, tmp_path / "not-reproduced", backend="test")
    assert (tmp_path / "not-reproduced" / "original.tsv").exists()
    assert not (tmp_path / "not-reproduced" / "minimal.tsv").exists()
    report = json.loads((tmp_path / "not-reproduced" / "report.json").read_text())
    assert report["status"] == "error" and report["verified"] is False
    with pytest.raises(ValueError, match="max_probes"):
        minimize(case, lambda _: True, max_probes=1)


def test_reduction_axes_are_removed_only_while_the_fault_reproduces(tmp_path):
    source = next(c for c in generate() if c.kind == "reduce")
    case = updated(source, dims=(4, 4, 4), axis=1, pre=1, post=0)
    def injected(candidate, actual):
        if len(candidate.dims) >= 2 and candidate.dims[candidate.axis] >= 2:
            actual = actual.copy()
            actual.flat[0] = np.finfo(np.float32).tiny
        return actual
    def fails(candidate):
        return python_mismatch(candidate, ct, tmp_path / "cache", injected)
    best, probes, status = minimize(case, fails)
    assert best.seed == case.seed
    assert best.dims == (2, 1) and best.axis == 0
    assert best.pre == best.post == 0
    assert probes <= 64 and status == "fixed_point"
    assert fails(best)
    assert not fails(updated(best, dims=(1, 1)))


def test_python_driver_retains_original_failure_after_automatic_shrink(tmp_path, monkeypatch):
    from test_seeded_fuzz import test_seeded_generated_programs_match_reference_bits as drive
    case = updated(next(generate()), dims=(4, 4, 4), pre=1, post=0)
    native_compile = ct.compile
    class Fault:
        def __init__(self, native):
            self.native = native
        def __getattr__(self, name):
            return getattr(self.native, name)
        def __call__(self, **inputs):
            actual = self.native(**inputs)
            if "A" in inputs and "C" not in inputs and inputs["A"].shape[1] > 1:
                actual = actual.copy()
                actual.flat[0] = np.finfo(np.float32).tiny
            return actual
    monkeypatch.setattr(ct, "compile", lambda *a, **k: Fault(native_compile(*a, **k)))
    monkeypatch.setenv("COFFEE_TEN_FUZZ_CASE", case.row())
    monkeypatch.setenv("COFFEE_TEN_FUZZ_FAILURE_DIR", str(tmp_path / "artifacts"))
    with pytest.raises(AssertionError):
        drive(tmp_path)
    artifact = tmp_path / "artifacts" / f"python-{case.index}"
    minimal = Case.parse((artifact / "minimal.tsv").read_text())
    assert minimal.dims == (1, 2, 1)
    assert minimal.seed == case.seed
    assert json.loads((artifact / "report.json").read_text())["verified"] is True
