# Interpreter series: local Mac measurements

Apple M5, release builds. All values below are f32 medians in microseconds from the
committed reports. Inputs and graphs are built outside timing; allocation/evaluation are
included. These are short diagnostic samples on one machine, not a controlled cross-machine
or statistical performance claim. Each raw report includes p25/p75, CPU and source parent.

| Case | Before | Final serial | Final 4 threads | Serial speedup |
|---|---:|---:|---:|---:|
| Column sum 512x512 | 237.50 | 120.77 | 125.17 | 1.97x |
| Middle-axis sum 8x64x128 | 22.20 | 20.47 | 48.37 | 1.08x |
| Equal-shape add | 118.23 | 30.66 | 60.88 | 3.86x |
| Right-scalar add | 116.94 | 30.48 | 59.88 | 3.84x |
| General broadcast | 478.72 | 476.39 | 222.43 | 1.00x |
| Matmul 128x128 | 876.81 | 280.73 | 75.99 | 3.12x |
| 32-node negation chain | 725.88 | 131.78 | 1190.49 | 5.51x |

## Limits and observed regressions

- Item 13 blocking is slower on the sampled 128x128 matmul than item 12: 187.10 -> 279.28 us. It is isolated in its own PR for review; these data do not justify a general speedup claim.
- Axis coalescing is explicit for reduction (the original implementation already used outer/k/inner products); general broadcasting already maintained offsets incrementally. Their reports do not demonstrate a separate speedup on the sampled case.
- Four threads add startup cost per operation. They help some larger operations but hurt short operations and the chain. The default creates no workers, and compiler benchmarks are unchanged.
- The sum sign-zero premise was corrected: NumPy 2.0.2 also returns +0 for the tested negative-zero sums.

## Reproduction

`cargo run --release --locked -p coffee-ten-interp --example interp_bench`

`cargo run --release --locked -p coffee-ten-interp --example interp_bench -- --threads 4`

Reports `04-after.csv` through `15-after.csv` were captured after every corresponding
plan item. `02-before.csv` is the baseline; `19-after.csv` and `19-threads4.csv` are the
final default and opt-in threaded paths. No 300 ms warmup or compiler measurements were added.

## Exceptional arithmetic

Ubuntu release CI caught a NaN sign-bit difference between the SIMD-friendly loop and
its scalar oracle. The bitwise assertions were retained. Sums and matrix products with
non-finite inputs use an isolated copy of the original scalar kernel, and the corresponding
independent test oracles are also isolated from inlining. The finite-input guard and its
scan are included in the updated Mac timings. The threaded path uses this scalar fallback
for exceptional inputs as well.

Rust does not promise portable or fully deterministic NaN arithmetic bit patterns;
these bitwise checks compare the two kernels on the same target/toolchain. See the
[Rust f32 documentation](https://doc.rust-lang.org/core/primitive.f32.html).
