# Benchmarks

Single-thread f32 matmul on one machine, as schedules are added: naive → tiled →
tiled + vectorized, against the machine's own measured ceilings. Pre-MVP numbers from
two sessions; they describe this compiler on this Mac, not a comparison with BLAS.

## Conditions

| | |
|---|---|
| Machine | Mac17,4 (MacBook Air, Apple M5, 16 GB, fanless), macOS 27.0.1 |
| Power | AC, Low Power Mode off (checked by `bench/run.sh`), heavy apps closed |
| Compiler | Apple clang 21.0.0, `-std=c99 -O2 -ffp-contract=fast -fno-vectorize -fno-slp-vectorize` (FMA contraction allowed; only the schedule vectorizes) |
| Code | commit `ffce49d`, clean tree (`"dirty": false` in every record) |
| Sessions | S1 2026-10-15, 60 s pauses between series; S2 2026-10-16 on a cold Mac, 180 s pauses |
| Method | per (kernel, N): one warm-up round, then 30 rounds; in every round each variant runs once, in a seeded random order; a sample repeats the kernel for ≥ 10 ms; median and p25–p75 of GFLOP/s = 2·N³ / time; a variant is flagged `drift` when the medians of its first and last 5 rounds differ by > 5% |
| Raw data | `bench/results/2026-10-15-ffce49d-s1.jsonl`, `bench/results/2026-10-16-ffce49d.jsonl` |

Every variant was checked against the naive kernel within the inner-product error bound
before it was timed.

## Ceilings (`bench/peak`, one core)

| | S1 | S2 |
|---|---|---|
| FMA peak, 16 chains of float32x4 | 122.9 GFLOP/s | 122.0 GFLOP/s |
| Memory triad, 3 × 128 MiB | 105.3 GB/s | 107.9 GB/s |
| L2 triad, 3 × 1 MiB | 119.6 GB/s | 120.4 GB/s |

The roofline column below is S2's median over S2's FMA peak (122.0 GFLOP/s). Matmul's
arithmetic intensity is far above the ridge point (~1 FLOP/byte here), so the compute
ceiling is the one that applies.

## Matmul, `A @ B`

Schedules (nest 1 is the `i, k, j` update nest):
- **naive**: no schedule;
- **tiled**: `split(1, i, io, ii, 32); split(1, j, jo, ji, 64); reorder(1, io, jo, k, ii, ji)`;
- **tiled + vec**: tiled, then `vectorize(1, ji)`.

S2 median [p25–p75] in GFLOP/s, (% of peak), and S1's median. `†` marks a variant
flagged `drift`.

| N | naive | tiled | tiled + vec | S1: naive / tiled / tiled + vec |
|---|---|---|---|---|
| 64 | 6.60 [6.12–7.47] † | 7.69 [7.66–7.71] | **60.50** [60.08–60.67] (49.6%) | 7.41 / 7.67 / 60.29 |
| 128 | 7.08 [7.06–7.09] | 7.66 [7.64–7.68] | **59.44** [59.33–59.62] (48.7%) | 7.07 / 7.65 / 59.40 |
| 256 | 7.33 [7.31–7.34] | 7.69 [7.68–7.72] | **57.38** [57.29–57.47] (47.0%) | 7.26 / 7.64 / 57.14 |
| 512 | 7.69 [7.60–7.72] | 7.54 [7.51–7.58] | **45.63** [45.29–45.93] (37.4%) | 7.72 / 7.60 / 46.09 |
| 1024 | 7.83 [7.80–7.86] | 7.53 [7.45–7.55] † | **24.21** [24.02–24.39] (19.8%) | 7.41 † / 7.11 † / 23.81 † |

## `relu(A @ B + bias)`

Same schedules on the matmul nests; the epilogue is a separate O(N²) nest.

| N | naive | tiled | tiled + vec | S1: naive / tiled / tiled + vec |
|---|---|---|---|---|
| 64 | 7.56 [7.39–7.61] | 7.56 [7.54–7.60] | **53.62** [53.38–53.91] (43.9%) | 3.95 / 4.40 † / 25.99 — invalid, see below |
| 128 | 7.00 [6.95–7.02] | 7.58 [7.54–7.59] | **55.18** [54.79–55.47] (45.2%) | 4.14 † / 4.41 † / 29.33 † |
| 256 | 7.22 [7.16–7.25] | 7.65 [7.58–7.69] | **55.43** [54.79–55.67] (45.4%) | 7.26 / 7.68 / 55.48 |
| 512 | 7.65 [7.55–7.68] | 7.55 [7.50–7.57] | **50.56** [49.49–50.96] (41.4%) | 3.30 † / 3.11 † / 17.86 † |
| 1024 | 7.88 [7.87–7.88] | 7.59 [7.59–7.59] | **24.72** [24.55–24.86] (20.3%) | 3.95 † / 3.63 † / 12.93 † |

## What the numbers say

Only differences larger than both variants' p25–p75 and seen in both sessions count as
results; the rest is marked as an observation.

- **`vectorize` is the step that matters: ×7.8–8.4 over naive for N ≤ 256** (e.g. 7.08 →
  59.44 at N = 128; the drifted naive variant at N = 64 is left out), 47–50% of peak. Reproduced in S1 and S2 within 0.5%.
- **Tiling alone gives +3–8% for N ≤ 256 (+3.5% at 64 in S1, +8% at 128, +5% at 256) and nothing or slightly less for N ≥ 512**
  (7.69 vs 7.54 at 512, reproduced). With N ≤ 1024 the three f32 matrices take at most
  12 MiB and fit the 16 MiB L2 (assumed size), so there is little traffic to save; the
  single-core L2 triad (120 GB/s) is barely above memory (106 GB/s).
- **The vectorized kernel falls from 57 to 46 and 24 GFLOP/s at N = 512 and 1024**,
  reproduced in both sessions for matmul. Hypothesis, not tested: `k` is not tiled, so
  the inner `k, ii, ji` block streams a K × 64 slice of B (128 KiB at N = 512, 256 KiB at
  1024), which no longer fits the assumed 128 KiB L1. A schedule that also splits `k`
  is the next thing to measure; the schedule space (`coffee-ten-bench space`) contains
  such schedules.
- **Observation, not reproduced:** at N = 512 the vectorized `relu(A @ B + bias)` (50.6)
  is faster than the vectorized matmul (45.6), although it does strictly more work. The
  two kernels differ only in where the product is written (a temporary vs the output).
  S1's value for it is invalid, so this is one session; unexplained.
- **The naive kernel runs at ~7.5 GFLOP/s, 6% of peak**, flat in N: scalar code,
  one fused multiply-add per inner iteration, bounded by issue and load/store rather
  than by the memory hierarchy.

## Validity

- **S1's second half is not usable.** In S1 most `relu(A @ B + bias)` series and matmul at
  N = 1024 drifted; several ran at about half the speed of the same kernel elsewhere.
  S2, on a cold machine with 180 s pauses, repeated them cleanly except two flagged
  variants (matmul N = 64 naive, N = 1024 tiled).
- **The drift flag misses a whole slow series.** S1's `relu(A @ B + bias)` at N = 64
  ran at half speed in every round (likely a hot fanless chassis or a migration to an
  efficiency core) and was not flagged, because the flag only compares the first and
  last rounds. Such series show up only against a second session, which is why every
  number above is shown with both.
- **Not measured:** NumPy/Accelerate (it may use the matrix units; a comparison would not
  measure the compiler), multiple threads, N above 1024, other machines. L1 and L2
  sizes are assumed, not measured.
