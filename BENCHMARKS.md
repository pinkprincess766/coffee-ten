# Benchmarks

Single-thread f32 matmul on one machine, as schedules are added: naive → tiled →
tiled + vectorized → register tile, against the machine's own measured ceilings. Numbers
from three sessions; they describe this compiler on this Mac, not a comparison with BLAS.

## Conditions

| | |
|---|---|
| Machine | Mac17,4 (MacBook Air, Apple M5, 16 GB, fanless), macOS 27.0.1 |
| Power | AC, Low Power Mode off (checked by `bench/run.sh`), heavy apps closed |
| Compiler | Apple clang 21.0.0, `-std=c99 -O2 -ffp-contract=fast -fno-vectorize -fno-slp-vectorize` (FMA contraction allowed; only the schedule vectorizes) |
| Code | S1, S2: commit `ffce49d`; S3: commit `4928ffa` (adds `unroll` and row-pointer indexing, #42–#44); clean tree (`"dirty": false` in every record) |
| Sessions | S1 2026-10-15, 60 s pauses between series; S2 2026-10-16 on a cold Mac, 180 s pauses; S3 2026-10-18 on a cold Mac, 60 s pauses |
| Method | per (kernel, N): one warm-up round, then 30 rounds; in every round each variant runs once, in a seeded random order; a sample repeats the kernel for ≥ 10 ms; median and p25–p75 of GFLOP/s = 2·N³ / time; a variant is flagged `drift` when the medians of its first and last 5 rounds differ by > 5% |
| Raw data | `bench/results/2026-10-15-ffce49d-s1.jsonl`, `bench/results/2026-10-16-ffce49d.jsonl`, `bench/results/2026-10-18-4928ffa.jsonl`, `bench/results/2026-10-18-0a946ba-peak.jsonl` (P1: `bench/peak` alone, after the kernel fix #46), `bench/results/2026-10-19-bb9a027-peak.jsonl` (P2: `bench/peak` alone, with the scalar kernel #50), `bench/results/2026-10-19-hypothesis-a.jsonl` (hypothesis A, run 1, commit `c435f89`), `bench/results/2026-10-21-561ede3-peak.jsonl` (P3: `bench/peak` alone, with the FMA chain kernels #58) |

Every variant was checked against the naive kernel within the inner-product error bound
before it was timed.

## Ceilings (`bench/peak`, one core)

| | S1 | S2 | S3 |
|---|---|---|---|
| FMA peak, 16 chains of float32x4 | 122.9 GFLOP/s | 122.0 GFLOP/s | 129.2 GFLOP/s [117.7–130.9] |
| Memory triad, 3 × 128 MiB | 105.3 GB/s | 107.9 GB/s | 112.1 GB/s |
| L2 triad, 3 × 1 MiB | 119.6 GB/s | 120.4 GB/s | 128.1 GB/s |
| FMA with the accumulator in memory (`fma_mem_acc`) | — | — | 31.1 GFLOP/s, old kernel, see below; P1: **46.0** [45.9–46.1]; P2: **46.9** [46.8–47.0] |
| The same in scalar code (`fma_mem_acc_scalar`) | — | — | P2: **8.56** [8.54–8.58] GFLOP/s; P3: 8.47 [8.44–8.51] |
| One dependent scalar FMA chain (`fma_chain`) | — | — | P3: **2.605** [2.599–2.612] GFLOP/s (0.77 ns per FMA) |
| Independent chains of length 256, N = 256 matmul body (`fma_chain256`) | — | — | P3: **3.133** [3.124–3.144] GFLOP/s; overlap window W = 256 × (3.133 / 2.605 − 1) ≈ 52 FMAs |

The S1/S2 roofline columns below are S2's median over S2's FMA peak (122.0 GFLOP/s); the
register-tile section uses S3's median over S3's peak (129.2 GFLOP/s). Matmul's
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

## Register tile (S3)

`regtile`: `split(1, i, io, ii, 4); split(1, j, jo, ji, 16); reorder(1, io, jo, k, ii, ji);
vectorize(1, ji); unroll(1, ii); unroll(1, ji)`. With the tile loops unrolled, clang keeps
the 4×16 tile of C in 16 vector registers across `k` (checked in the optimized IR,
`crates/coffee-ten-codegen-c/tests/unroll.rs`). S3 median [p25–p75] in GFLOP/s, (% of S3's
peak); `tiled + vec` from the same session for comparison.

| N | matmul: tiled + vec | matmul: regtile | relu(A @ B + bias): tiled + vec | relu(A @ B + bias): regtile |
|---|---|---|---|---|
| 64 | 64.01 [63.77–64.57] | **121.84** [121.57–122.73] (94.3%) | 57.96 [57.81–58.07] | **100.77** [100.45–100.90] (78.0%) |
| 128 | 62.98 [61.58–63.30] | **122.77** [120.98–124.08] (95.0%) | 59.59 [59.40–59.66] | **111.29** [110.96–111.60] (86.1%) |
| 256 | 61.43 [61.13–61.53] | **123.72** [123.34–124.21] (95.7%) | 59.59 [59.35–59.66] | **116.06** [115.07–116.45] (89.8%) |
| 512 | 55.95 [55.74–56.10] | **121.81** [120.92–122.21] (94.3%) | 55.10 [54.85–55.40] | **117.67** [117.42–118.11] (91.1%) |
| 1024 | 26.73 [26.61–26.81] | **105.30** [104.49–106.51] (81.5%) | 26.43 [26.20–26.54] | **103.12** [101.54–103.92] (79.8%) |

S3's naive and tiled values (7.5–8.4 GFLOP/s, naive at N = 64 flagged `drift` for both
kernels, tiled `relu(A @ B + bias)` at 128 and 256 flagged) are in the raw file.

## What the numbers say

Only differences larger than both variants' p25–p75 and seen in both sessions count as
results; the rest is marked as an observation.

- **The register tile reaches 94–96% of the FMA peak for matmul at N ≤ 512** (121.8–123.7
  GFLOP/s), ×1.9–2.2 over `tiled + vec` in the same session, and 81% at N = 1024 (×3.9,
  where `tiled + vec` falls to 26.7). With the epilogue: 78–91%. One session (S3); the
  gaps are far larger than any p25–p75 range, but a second session is still to come.
- **`vectorize` is the step that matters among S1/S2's schedules: ×7.8–8.4 over naive for N ≤ 256** (e.g. 7.08 →
  59.44 at N = 128; the drifted naive variant at N = 64 is left out), 47–50% of peak. Reproduced in S1 and S2 within 0.5%.
  Why only half: C stays in memory, so every vector FMA also loads and stores C. clang
  keeps the 64-float slice of B in 16 registers across `ii` (it does not depend on `ii`),
  so the inner loop does about 2 memory operations per FMA, not 3 (`otool -tv` on the same
  nest written by hand: per 4 `fmla.4s`, 2 `ldp` and 2 `stp` of C). The register tile removes the C traffic.
- **Tiling alone gives +3–8% for N ≤ 256 (+3.5% at 64 in S1, +8% at 128, +5% at 256) and nothing or slightly less for N ≥ 512**
  (7.69 vs 7.54 at 512, reproduced). With N ≤ 1024 the three f32 matrices take at most
  12 MiB and fit the 16 MiB L2 (assumed size), so there is little traffic to save; the
  single-core L2 triad (120 GB/s) is barely above memory (106 GB/s).
- **The vectorized kernel falls from 57 to 46 and 24 GFLOP/s at N = 512 and 1024**,
  reproduced in both sessions for matmul. Hypothesis, not tested: `k` is not tiled, so
  the inner `k, ii, ji` block streams a K × 64 slice of B (128 KiB at N = 512, 256 KiB at
  1024), which no longer fits the assumed 128 KiB L1. A schedule that also splits `k`
  is the next thing to measure; the schedule space (`coffee-ten-bench space`) contains
  such schedules. Consistent with it: the register tile streams a K × 16 slice of B
  (64 KiB at N = 1024, inside the assumed L1), holds 121.8 at N = 512 and drops only to
  105.3 at 1024.
- **Observation, not reproduced:** at N = 512 the vectorized `relu(A @ B + bias)` (50.6)
  is faster than the vectorized matmul (45.6), although it does strictly more work. The
  two kernels differ only in where the product is written (a temporary vs the output).
  S1's value for it is invalid, so this is one session; unexplained. In S3 (different
  code: row-pointer indexing) it did not reappear: 55.1 vs 56.0.
- **The naive kernel runs at ~7.5 GFLOP/s, 6% of peak**, flat in N: scalar code,
  one fused multiply-add per inner iteration, bounded by issue and load/store rather
  than by the memory hierarchy.

## Hypothesis A (spec §7.1), run 1: not met

The hypothesis: the analytic model's top 10 contains a schedule at least 90% as fast as
the best one random search finds with 1000 measurements. Run 1 (commit `c435f89`, N = 256,
seed 1, machine P2):

1. 1000 random candidates of the 7334-schedule space, 5 samples each;
2. the model's top 10, same way;
3. the best of each group in one interleaved run, 30 rounds; the decision uses only that run.

| | schedule | median [p25–p75], GFLOP/s |
|---|---|---|
| best random (candidate 886 of 1000) | 4×16 register tile | **121.97** [121.83–122.27] |
| best of the model's top 10 (rank 2) | `split i 128, j 64, k 128; reorder io, ko, jo, ii, ki, ji; vectorize ji` | **75.86** [75.71–76.39] |

**Ratio 0.622: the criterion is not met.** Over the 1000 random candidates the model's
Spearman correlation with the measurements is 0.292, and its median error is ×2.10 (on
the 19 hand-picked S3 schedules it was 0.922 and ×1.07). Why, from the log:

- **`vectorize` on a short loop does nothing.** Under the benchmark flags clang fully
  unrolls a loop of 8 or 16 iterations before its loop vectorizer runs, so the pragma
  finds no loop and the code stays scalar. Vectorized random candidates by `ji` extent
  (measured median vs the model's median prediction): 8: 13.0 vs 80.5; 16: 14.9 vs 93.9;
  32: 24.1 vs 66.0; 64: 24.7 vs 67.8; 128: 20.6 vs 46.5 GFLOP/s. The model believes the
  schedule; the schedule does not do what it says. This is a code generator problem first.
- **The model's top 10 has no register tile.** It predicts schedules with `ii, ki, ji`
  innermost at the FMA peak (a 16-vector row of C held in registers across `ki`, never
  checked in clang); they measured 31–77. The six register tiles tie with them at the
  peak, and the tie is broken by the schedule text, which puts `ii, 128` before `ii, 2`.
- **Random search found a register tile only once** (candidate 886): 6 tiles among 7334
  candidates give 0.8 expected in 1000 draws.

Next: make `vectorize` vectorize short loops, check the C-row hoisting across `ki` in the
optimized IR, break ties at the cap by something other than the text, then repeat with a
new seed. Until then the hypothesis stays open; this run is kept as it is.

Since then the model has learned that a short vectorized loop is scalar and how to break
ties (#57), and it has an FMA latency ceiling for reductions (#59), calibrated by P3. Over the same 1000
candidates it now has Spearman 0.564 and a median error of ×1.53. These candidates were
used to find the fixes, so this is not an independent test; run 2 needs a new seed. The
largest remaining error is the vectorized candidates with `ji` ≥ 32: measured 21–25,
predicted 46–67 GFLOP/s.

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
- **S3 is not a third repeat of S1/S2.** Its code indexes matrices through row pointers
  (#44), and its `tiled + vec` values are 6–23% higher than S2's (e.g. 56.0 vs 45.6 at
  N = 512). Its FMA peak is also 6% higher (129.2 vs 122.0). The two effects cannot be
  separated from these runs.
- **The FMA peak has a slow lower tail in S3, P1 and P2**: p25 117.7 and min 58.2 in S3,
  p25 118.6 and min 59.6 in P1, p25 118.2 and min 60.1 in P2, against medians of 129.2,
  126.7 and 128.8 (P3: p25 127.2, min 85.8, median 130.6). S3's run overlapped a
  background build for its first seconds and an earlier version of this note blamed
  that; P1 ran with nothing else and shows the same tail, so the cause is open
  (frequency ramp at the start, or a migration to an efficiency core). The median is used.
- **S3's `fma_mem_acc` (31.1) is from a kernel that was limited by its own loop
  instructions** (one vector per iteration), below the 56–64 GFLOP/s of `tiled + vec`
  that it should bound. The kernel now does four vectors per iteration (#46); P1
  measured it at 46.0 GFLOP/s [45.9–46.1].
- **Not measured:** NumPy/Accelerate (it may use the matrix units; a comparison would not
  measure the compiler), multiple threads, N above 1024, other machines. L1 and L2
  sizes are assumed, not measured.
