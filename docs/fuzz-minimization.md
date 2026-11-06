# Failure capture and greedy shrinking (N9)

N9 operates on the N7 TSV grammar and only on reproducible output mismatches. It changes
test tooling, not the compiler, interpreter or checked-in corpus. Both N7 drivers retain
an original failure before attempting minimization. A shrink exception or timeout cannot
turn the original failing test into a pass.

Artifacts are `original.tsv`, `minimal.tsv` (only after successful final verification)
and `report.json`. The report contains grammar version, backend, original seed, source
commit/tracked-dirty state, both replay rows, probe count, status and verification flag.
Input fingerprints are recomputed after every structural change; seed, family and case
ID remain unchanged. Reusing an output directory clears any stale minimal replay first.
Shrinking creates a new test program and regenerates its inputs from that seed; it is not
an optimization claiming the same output as the original case. Its own interpreter result
is recomputed for every probe, and that program's accumulation order remains the reference.

## Algorithm and guarantees

For reductions/elementwise cases, try deleting dimensions and adjusting the reduction
axis so the result remains a valid case. Matrix dimensions stay rank 2 on each operand;
their m/k/n tuple cannot lose an axis. Try smaller dimension values (1, half, predecessor),
remove optional pre/post expressions, and simplify the selected axis to zero. Reject any
candidate outside the original grammar. Every accepted step strictly decreases the
complexity tuple `(len(dimensions), sum(dimensions), pre, post, axis)` and must still reproduce the
output mismatch when compiled and checked against the interpreter.

The search is greedy: restart from the new case after each accepted step. Default limits
are 64 oracle calls and a 60-second between-probe budget. The initial reproduction and a
reserved independent final reproduction both count toward the call limit. Completion
means a local fixed point under these moves or an explicitly reported probe/time budget;
it is not a claim of global minimality. A failed final check sets `verified: false` and
retains the original rather than publishing an unverified minimal replay.

## Backends and replay

```sh
# From the same source checkout, with its Python extension available:
python3 tests/fuzz/ct_fuzz_minimize.py --backend python --case original.tsv --output target/n9-repro
python3 tests/fuzz/ct_fuzz_minimize.py --backend rust --case original.tsv --output target/n9-repro-rust
```

The Python backend calls the existing builder, `ct.run` and `ct.compile`; it compares
shape/dtype and uint32 output bits. The Rust backend builds the N7 test executable once
(or accepts `--rust-driver`), then executes only its named comparison test with
`COFFEE_TEN_FUZZ_CASE`. A stable output-mismatch marker distinguishes numerical/shape
failures from unrelated setup/compilation errors. Probes set `COFFEE_TEN_FUZZ_MINIMIZING`
to prevent recursive minimization. A Rust probe has a 30-second subprocess timeout.
Neither oracle treats an arbitrary exception or process error as evidence that the
original mismatch persists.

Automatic Rust capture passes the already-running test executable, so it never invokes
Cargo recursively while a parent Cargo test is active. Python capture uses its existing
native extension. CI places files under ignored `target/fuzz-failures`; local drivers use
their temporary directories unless `COFFEE_TEN_FUZZ_FAILURE_DIR` is supplied. The complete
JSON report is also printed into the failure log, retaining both replay rows after a CI
runner is removed. No upload action or new dependency is introduced.

The existing CI step timeout still bounds the whole test command. Between-probe budgets
do not preempt a Python native compile already in progress. Compilation/setup failures,
input mutation failures, and N8 transformation-specific failures keep their original N7/N8
diagnostics; this minimizer does not relabel them as ordinary N7 output mismatches.

Regression tests inject a deliberately wrong value **after native execution**, only when
a matmul reduction has k>1. Actual compile/interpreter comparisons detect it and shrink
4x4 by 4x4 to 1x2 by 2x1 with the seed unchanged. Removing the last necessary k dimension
then makes the artificial fault disappear. Both Python and Rust subprocess paths exercise
this test; an additional native reduction test removes axes while preserving its injected
failure. Reserved `COFFEE_TEN_FUZZ_INJECT_FAULT` is used only by this test machinery, never
by production evaluation. No real compiler defect is claimed by these injected tests.
