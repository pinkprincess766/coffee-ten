# Interpreter measurements

Run on an otherwise idle Mac: `cargo run --release --locked -p coffee-ten-interp --example interp_bench`.
The harness uses only std::time and black_box. Each case includes graph evaluation and allocation,
with inputs and graph constructed outside timing, one untimed call and nine samples of at least
10 ms. Reports contain p25, median and p75 in microseconds for f32 and f64. No compiler kernel
is timed. These short local measurements are diagnostic, not controlled performance claims.

`02-before.csv` is the baseline before runtime changes. Numbered reports were captured after
that plan item, using its source before committing; the parent commit and Mac CPU are recorded
in comment lines. Compare the same case and dtype, accounting for the reported spread.
