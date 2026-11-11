"""N9: bounded greedy minimization of reproducible N7 output mismatches."""
from dataclasses import replace
from pathlib import Path
import argparse
import json
import os
import subprocess
import time

from ct_fuzz_cases import Case, build, input_digest, numpy_inputs, read_cases

ROOT = Path(__file__).resolve().parents[2]


def updated(case, **changes):
    candidate = replace(case, **changes, input_hash="")
    candidate = replace(candidate, input_hash=input_digest(candidate))
    return Case.parse(candidate.row())


def complexity(case):
    return (len(case.dims), sum(case.dims), case.pre, case.post, case.axis)


def smaller(case):
    candidates = []
    if case.kind in ("reduce", "elementwise") and len(case.dims) > 1:
        for axis in range(len(case.dims)):
            dims = case.dims[:axis] + case.dims[axis + 1:]
            kept_axis = min(case.axis - int(axis < case.axis), len(dims) - 1)
            candidates.append(updated(case, dims=dims, axis=kept_axis))
    for axis, dim in enumerate(case.dims):
        for value in sorted({1, max(1, dim // 2), dim - 1}):
            if 1 <= value < dim:
                dims = case.dims[:axis] + (value,) + case.dims[axis + 1:]
                candidates.append(updated(case, dims=dims))
    for field in ("pre", "post", "axis"):
        if getattr(case, field):
            candidates.append(updated(case, **{field: 0}))
    seen = set()
    for candidate in candidates:
        row = candidate.row()
        if row not in seen and complexity(candidate) < complexity(case):
            seen.add(row)
            yield candidate


def minimize(case, fails, *, max_probes=64, seconds=60):
    if not 2 <= max_probes <= 256 or not 0 < seconds <= 120:
        raise ValueError("max_probes must be 2..256 and seconds must be in (0,120]")
    start = time.monotonic()
    probes = 0
    def probe(candidate):
        nonlocal probes
        probes += 1
        return fails(candidate)
    if not probe(case):
        raise ValueError("initial mismatch does not reproduce")
    best = case
    status = "fixed_point"
    while True:
        accepted = False
        for candidate in smaller(best):
            if probes >= max_probes - 1:
                status = "probe_budget"
                break
            if time.monotonic() - start >= seconds:
                status = "time_budget"
                break
            if probe(candidate):
                best = candidate
                accepted = True
                break
        if not accepted:
            break
    # Reserve this independent final check even when no further shrinking is allowed.
    if not probe(best):
        raise ValueError("minimized mismatch no longer reproduces")
    return best, probes, status


def source_revision():
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True, timeout=5)
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def source_dirty_tracked():
    try:
        result = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                cwd=ROOT, capture_output=True, text=True, timeout=5)
        return bool(result.stdout.strip()) if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def save_failure(case, fails, directory, *, backend, max_probes=64, seconds=60):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Save the original before invoking any shrink oracle, including a failing one.
    (directory / "original.tsv").write_text(case.row() + "\n")
    (directory / "minimal.tsv").unlink(missing_ok=True)
    report = {"version": 1, "case_format": "coffee-ten-fuzz-v1", "backend": backend, "seed": f"{case.seed:016x}",
              "source_commit": source_revision(), "source_dirty_tracked": source_dirty_tracked(),
              "original": case.row()}
    try:
        best, probes, status = minimize(case, fails, max_probes=max_probes, seconds=seconds)
        (directory / "minimal.tsv").write_text(best.row() + "\n")
        report.update(minimal=best.row(), probes=probes, status=status, verified=True)
    except BaseException as error:
        # Native panic wrappers and user interruptions must retain error metadata too.
        # Re-raise unchanged after recording: the oracle failure is never swallowed.
        report.update(status="error", error=str(error), verified=False)
        raise
    finally:
        text = json.dumps(report, sort_keys=True, indent=2)
        (directory / "report.json").write_text(text + "\n")
        print(f"N9 artifact {directory}\n{text}", flush=True)
    return best


def python_mismatch(case, ct, cache_dir, inject=None):
    import numpy as np
    expression = build(case, ct)
    inputs = numpy_inputs(case)
    expected = ct.run(expression, **inputs)
    actual = ct.compile(expression, cache_dir=cache_dir)(**inputs)
    if inject is not None:
        actual = inject(case, actual)
    return (actual.shape != expected.shape or actual.dtype != expected.dtype
            or not np.array_equal(actual.view(np.uint32), expected.view(np.uint32)))


def rust_driver():
    result = subprocess.run(["cargo", "test", "-p", "coffee-ten-codegen-c", "--test",
                             "seeded_fuzz", "--no-run", "--locked", "--message-format=json"],
                            cwd=ROOT, capture_output=True, text=True, timeout=600, check=True)
    for line in result.stdout.splitlines():
        item = json.loads(line)
        if item.get("reason") == "compiler-artifact" and item.get("target", {}).get("name") == "seeded_fuzz" and item.get("executable"):
            return item["executable"]
    raise ValueError("cargo did not report the seeded_fuzz test executable")


def rust_mismatch(case, driver):
    env = dict(os.environ, COFFEE_TEN_FUZZ_CASE=case.row(), COFFEE_TEN_FUZZ_MINIMIZING="1")
    result = subprocess.run([str(Path(driver).resolve()), "--exact", "seeded_generated_programs_match_reference_bits", "--nocapture"],
                            cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
    if result.returncode == 0:
        return False
    text = result.stdout + result.stderr
    if "N7 output mismatch" in text and "assertion" in text:
        return True
    raise ValueError("Rust replay produced a different error, not an output-bit mismatch: " + text[-3000:])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--case", type=Path)
    source.add_argument("--row")
    parser.add_argument("--backend", choices=("python", "rust"), default="python")
    parser.add_argument("--rust-driver", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-probes", type=int, default=64)
    parser.add_argument("--seconds", type=float, default=60)
    args = parser.parse_args()
    cases = read_cases(args.case.read_text() if args.case else args.row)
    if len(cases) != 1:
        parser.error("provide exactly one replay case")
    if args.backend == "python":
        import coffee_ten as ct
        cache = args.output / "compile-cache"
        oracle = lambda case: python_mismatch(case, ct, cache)
    else:
        driver = args.rust_driver or rust_driver()
        oracle = lambda case: rust_mismatch(case, driver)
    try:
        save_failure(cases[0], oracle, args.output, backend=args.backend,
                     max_probes=args.max_probes, seconds=args.seconds)
    except Exception as error:
        parser.exit(1, f"minimization failed; original retained: {error}\n")
