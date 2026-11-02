# Interpreter measurements

Run on an otherwise idle Mac: `cargo run --release --locked -p coffee-ten-interp --example interp_bench`.
The harness uses only std::time and black_box. Each case includes graph evaluation and allocation,
with inputs and graph constructed outside timing, one untimed call and nine samples of at least
10 ms. Reports contain p25, median and p75 in microseconds for f32 and f64. No compiler kernel
is timed. These short local measurements are diagnostic, not controlled performance claims.

`02-before.csv` is the baseline before runtime changes. Numbered reports were captured after
that plan item, using its source before committing; the parent commit and Mac CPU are recorded
in comment lines. Compare the same case and dtype, accounting for the reported spread.


## Explicit parallel execution

`cargo run --release --locked -p coffee-ten-interp --example interp_bench -- --threads 4`
opts into interpreter workers. The default is 1, which creates no workers. The flag accepts
1..=256; it belongs only to this interpreter harness. Rust callers can opt into
`run_f32_with_threads` / `run_f64_with_threads`. Python `ct.run`, normal Rust `run_f32` /
`run_f64`, and the compiler measurement harness keep their default serial behavior.

Workers own disjoint output chunks; a single output's reduction never crosses workers.
Every sum and matrix product retains ascending reduction indices. Workers join before
an intermediate can be released. Thread setup happens per operation, so small kernels
and chains can be slower; the flag is an explicit experiment, not automatic tuning.
`19-after.csv` measures the default and `19-threads4.csv` measures four threads on the Mac.


Non-finite sums and matmuls take the original isolated scalar path, including in explicit
threaded execution. This preserves the checked oracle bits on the tested toolchains without
changing the bitwise test assertions or enabling reassociation. The finite-input scan is
part of the recorded timing. See SERIES.md for the NaN portability limits.
