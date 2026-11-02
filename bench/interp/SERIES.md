# Interpreter series: local Mac measurements

Apple M5, release builds. All values below are f32 medians in microseconds from the
committed reports. Inputs and graphs are built outside timing; allocation/evaluation are
included. These are short diagnostic samples on one machine, not a controlled cross-machine
or statistical performance claim. Each raw report includes p25/p75, CPU and source parent.

| Case | Before | Final serial | Final 4 threads | Serial speedup |
|---|---:|---:|---:|---:|
| Column sum 512x512 | 237.50 | 21.89 | 42.82 | 10.85x |
| Middle-axis sum 8x64x128 | 22.20 | 4.14 | 30.05 | 5.36x |
| Equal-shape add | 118.23 | 30.57 | 59.60 | 3.87x |
| Right-scalar add | 116.94 | 30.57 | 59.32 | 3.82x |
| General broadcast | 478.72 | 470.31 | 220.90 | 1.02x |
| Matmul 128x128 | 876.81 | 270.36 | 67.12 | 3.24x |
| 32-node negation chain | 725.88 | 132.15 | 1185.47 | 5.49x |

## Limits and observed regressions

- Item 13 blocking is slower on the sampled 128x128 matmul than item 12: 205.90 -> 282.52 us. It is isolated in its own PR for review; these data do not justify a general speedup claim.
- Axis coalescing is explicit for reduction (the original implementation already used outer/k/inner products); general broadcasting already maintained offsets incrementally. Their reports do not demonstrate a separate speedup on the sampled case.
- Four threads add startup cost per operation. They help some larger operations but hurt short operations and the chain. The default creates no workers, and compiler benchmarks are unchanged.
- The sum sign-zero premise was corrected: NumPy 2.0.2 also returns +0 for the tested negative-zero sums.

## Reproduction

`cargo run --release --locked -p coffee-ten-interp --example interp_bench`

`cargo run --release --locked -p coffee-ten-interp --example interp_bench -- --threads 4`

Reports `04-after.csv` through `15-after.csv` were captured after every corresponding
plan item. `02-before.csv` is the baseline; `19-after.csv` and `19-threads4.csv` are the
final default and opt-in threaded paths. No 300 ms warmup or compiler measurements were added.
