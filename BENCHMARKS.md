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
| Method | per (kernel, N): a warm-up (S1–S3 and run 1: one round of one 10 ms sample per variant; from #61: each variant runs 300 ms), then 30 rounds; in every round each variant runs once, in a seeded random order; a sample repeats the kernel for ≥ 10 ms; median and p25–p75 of GFLOP/s = 2·N³ / time; a variant is flagged `drift` when the medians of its first and last 5 rounds differ by > 5% |
| Raw data | `bench/results/2026-10-15-ffce49d-s1.jsonl`, `bench/results/2026-10-16-ffce49d.jsonl`, `bench/results/2026-10-18-4928ffa.jsonl`, `bench/results/2026-10-18-0a946ba-peak.jsonl` (P1: `bench/peak` alone, after the kernel fix #46), `bench/results/2026-10-19-bb9a027-peak.jsonl` (P2: `bench/peak` alone, with the scalar kernel #50), `bench/results/2026-10-19-hypothesis-a.jsonl` (hypothesis A, run 1, commit `c435f89`), `bench/results/2026-10-21-561ede3-peak.jsonl` (P3: `bench/peak` alone, with the FMA chain kernels #58), `bench/results/2026-10-21-hypothesis-a-run2.jsonl` (hypothesis A, run 2, commit `702dcf1`), `bench/results/2026-10-22-hypothesis-a-run3.jsonl` (hypothesis A, run 3, commit `726cc1d`), `bench/results/2026-10-27-6556e03-peak.jsonl` (P4: `bench/peak` alone, with the store kernel #80), `bench/results/2026-10-27-4753578-peak.jsonl` (P5: `bench/peak` alone, with the unrolled scalar kernel #83), `bench/results/2026-10-28-827cd1d-peak.jsonl` (P6: `bench/peak` alone, with the column-reload kernel #88), `bench/results/2026-10-28-adcf09c-peak.jsonl` (P7: `bench/peak` alone, with the unrolled column-load kernel #91), `bench/results/2026-10-30-hypothesis-a-run4.jsonl` (hypothesis A, run 4, commit `736d82f`) |

Every variant was checked against the naive kernel within the inner-product error bound
before it was timed.

Plot of S3 (all four schedules, median with p25–p75): [`docs/bench-gflops.png`](docs/bench-gflops.png),
built from the committed raw data with `python3 bench/plot.py` (needs matplotlib; no new measurements).

Roofline of S3: [`docs/bench-roofline.png`](docs/bench-roofline.png), `python3 bench/roofline.py`
(same input, no new measurements; after Williams, Waterman, Patterson, CACM 2009, README [1]).
Ceilings are S3's own `bench/peak` medians from the same file: `fma_gflops` 129.2 GFLOP/s and
`triad_gbs` 112.1 GB/s (ridge 1.15 FLOP/byte), not datasheet numbers. Arithmetic intensity
of each point is counted on compulsory DRAM traffic, float32: A and B read once, C read and
written once, so AI = 2·N³ / (4 · 4·N²) = N / 8 FLOP/byte (8 at N = 64 to 128 at N = 1024),
the same for all four schedules. That is the minimum traffic, so it is an **upper bound** on
the real AI (a lower bound on traffic): the naive and tiled nests can reload B and C and
have a lower AI. The real per-variant traffic is not in the data (no counters), so the plot
does not claim it. The 4 bytes per element are the buffer type of every kernel: `float`,
`EntryFn = unsafe extern "C" fn(*const *mut f32)` in `crates/coffee-ten-codegen-c/src/compile.rs`.
At N ≤ 256 all three matrices fit in cache (48 KiB at N = 64), so DRAM is not the limit
there and the L2 triad (`triad_l2_gbs`, 128.1 GB/s in the same peak line) would be the
more accurate slope; it does not change the conclusion. All points sit right of the ridge,
which is why the compute ceiling is the one that applies.

## Ceilings (`bench/peak`, one core)

| | S1 | S2 | S3 |
|---|---|---|---|
| FMA peak, 16 chains of float32x4 | 122.9 GFLOP/s | 122.0 GFLOP/s | 129.2 GFLOP/s [117.7–130.9] |
| Memory triad, 3 × 128 MiB | 105.3 GB/s | 107.9 GB/s | 112.1 GB/s |
| L2 triad, 3 × 1 MiB | 119.6 GB/s | 120.4 GB/s | 128.1 GB/s |
| FMA with the accumulator in memory (`fma_mem_acc`) | — | — | 31.1 GFLOP/s, old kernel, see below; P1: **46.0** [45.9–46.1]; P2: **46.9** [46.8–47.0]; P4: 46.3 [45.6–46.8]; P6: 46.2 [46.1–46.2]; P7: 46.1 [46.05–46.11] |
| The same in scalar code (`fma_mem_acc_scalar`) | — | — | P2: **8.56** [8.54–8.58] GFLOP/s; P3: 8.47 [8.44–8.51]; P4: 8.55 [8.53–8.57]; P5: 8.50 [8.47–8.52]; P6: 8.49 [8.44–8.51]; P7: 8.38 [8.36–8.39] |
| The same, scalar, 16 elements per iteration as straight-line code (`fma_mem_acc_scalar_unrolled`) | — | — | P5: **23.00** [22.92–23.07] GFLOP/s; P6: 23.01 [23.00–23.04]; P7: 22.62 [22.60–22.63] |
| One dependent scalar FMA chain (`fma_chain`) | — | — | P3: **2.605** [2.599–2.612] GFLOP/s (0.77 ns per FMA); P4: 2.602 [2.598–2.605]; P6: 2.617 [2.614–2.621]; P7: 2.585 [2.562–2.601] |
| Independent chains of length 256, N = 256 matmul body (`fma_chain256`) | — | — | P3: **3.133** [3.124–3.144] GFLOP/s; overlap window W = 256 × (3.133 / 2.605 − 1) ≈ 52 FMAs; P4: 3.241 [3.238–3.243], W ≈ 63; P6: 3.107 [3.084–3.116], W ≈ 48; P7: 3.130 [3.116–3.138], W ≈ 54 |
| Every vector FMA result stored, nothing loaded (`fma_store`) | — | — | P4: **58.50** [58.30–58.61] GFLOP/s; P5: 58.21 [57.71–58.53]; P6: 57.94 [57.25–58.02]; P7: 56.73 [56.28–56.97] |
| Scalar matmul nest N = 256 with `ii` innermost and `k` outside it, C reloaded at a 1 KiB stride (`fma_col_reload`) | — | — | P6: **2.121** [2.117–2.123] GFLOP/s; by hand 7.7 with only C's row padded to 272 floats; P7: 2.112 [2.108–2.119] |
| The same matmul with `ii` = 16 unrolled under `ki`: C in registers, 16 loads down a column of A at a 1 KiB stride per `k` (`fma_col_unrolled`) | — | — | P7: **15.62** [15.53–15.72] GFLOP/s; by hand 20.0 with only A's row padded to 272 floats, 26.7 with A transposed |

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

**The per-candidate numbers of this run are biased low and are not usable to judge the
model.** `search` compiles each candidate and measures it at once, with one 10 ms warm-up
round and 5 samples of 10 ms. Right after a fresh compile a kernel runs at about half
speed for tens of milliseconds. One schedule (`split i 32, j 64; reorder jo, io, k, ii, ji;
vectorize ji`) gave 34–36 GFLOP/s, spread 21–41, when freshly compiled, and 58–60, tight,
from the cache. It was recorded in this run at 24.1, while the interleaved harness measures
60. A fresh 30-candidate search repeated the pattern: its best four vectorized picks
recorded 30–44 against 52–79 in the harness. The decision run above is not affected: 30
interleaved rounds of already compiled kernels. The Spearman figures computed against
these candidates (0.292, and later 0.564 after the model fixes #57, #59) are therefore not
evidence about the model. A time-based warm-up is the fix, then run 2 with a new seed.

## Hypothesis A (spec §7.1), run 2: met

Run 2 (commit `702dcf1`, N = 256, seed 2, machine P3). It is run 1's procedure with three
differences: the model knows short vectorized loops, FMA latency and tie-breaking (#57,
#59); `search` warms each candidate for 300 ms (#61); and the seed is new. Within each
random candidate's 5 samples the spread is now p75/p25 = 1.004 (run 1: 1.4–1.6).

| | schedule | median [p25–p75], GFLOP/s |
|---|---|---|
| best random (candidate 953 of 1000) | 8×8 register tile | **112.98** [112.52–113.04] |
| best of the model's top 10 (rank 1) | 4×16 register tile | **121.85** [121.40–121.92] |

**Ratio 1.079: the criterion is met.** The model's first measurement already gives its best
schedule. Random search stays at 88–90 GFLOP/s until candidate 953, the only register tile
among its 1000 draws. Over the 1000 random candidates the model has Spearman 0.571 and a
median error of ×1.37 (P3 machine, `model compare`). By group (measured vs predicted median):

| group | n | measured | predicted | pred / meas |
|---|---|---|---|---|
| vectorized, `ji` = 128 | 46 | 45.6 | 45.8 | 1.00 |
| scalar, `ji` innermost | 271 | 7.8 | 8.4 | 1.07 |
| scalar, `k`/`ki` innermost | 239 | 7.3 | 6.8 | 0.82 |
| scalar, `ii` innermost | 243 | 6.9 | 8.4 | 1.22 (wide) |
| "vectorized", `ji` = 8 or 16 (scalar code) | 101 | 22–29 | 18–21 | 0.69 |
| vectorized, `ji` = 32 or 64 | 98 | 55–60 | 76–106 | 1.43–1.50 |

The last column is the median of the per-record ratios predicted / measured, not the ratio of
the two medians next to it (the same metric as run 3 below). Model v1.2; the method is
reconstructed from the table's own numbers, not recomputed.

What this does and does not show:

- **The criterion is met because of the register tiles.** The model ranks them first, and random
  search meets one in about 0.8 of 1000 draws. Spec §7.1 named this risk in advance. The test
  says the model finds the region a 1000-draw random search mostly misses; it says less
  about the model's precision elsewhere. Spearman 0.571 says that.
- **This is the first clean measurement of the random candidates.** Run 1's per-candidate
  numbers were biased (see above). Most of what looked like a ×2.2–3.3 model error on
  vectorized schedules was that bias; the real error there is about ×1.45.
- **The model's fixes were found on run 1's data**, then frozen before run 2. Run 2's random
  sample is new (seed 2). The model's top 10 does not depend on the seed.
- **Not tested:** other N, other seeds beyond 1 and 2, other machines.

## Hypothesis A (spec §7.1), run 3: met, out of sample

Run 3 (commit `726cc1d`, N = 256, seed 3, machine P3) repeats run 2's procedure with model
v1.3 (#63). v1.3 knows which rolled loops clang fully unrolls, a rule found from the IR of
run 2's schedules. Run 3's 1000 random candidates are a new sample, so this is the
out-of-sample check of v1.3. Within-record spread p75/p25 = 1.004; no drift flags.

| | schedule | median [p25–p75], GFLOP/s |
|---|---|---|
| best random (candidate 284 of 1000) | `split i 8, j 64, k 64; reorder ko, jo, io, ii, ki, ji; vectorize ji` | **86.32** [83.11–86.64] |
| best of the model's top 10 (rank 1) | 4×16 register tile | **117.56** [113.12–118.02] |

**Ratio 1.362: the criterion is met.** This time random search drew no register tile in 1000
draws. Its best-so-far was 60.0 GFLOP/s after 16 candidates, 86.4 after 64 and 87.0 from 284
on. The model's rank 1 gave 118 at the first measurement. The decision run's quartiles are wider than in run 2
(p25 3–4% below the median), but the gap is far larger than that.

**The model's precision holds out of sample.** Over the 1000 random candidates, model v1.3 vs
measured:

| | Spearman | median error |
|---|---|---|
| run 2 (v1.2, before the rule) | 0.571 | ×1.37 |
| run 2 (v1.3, in sample) | 0.854 | ×1.24 |
| **run 3 (v1.3, new sample)** | **0.853** | **×1.26** |

By group (run 3, 1000 random records, P3 machine). The error is the **median of the ratios**
predicted / measured over the group's records, not the ratio of the two medians next to it:

| group | n | measured median | predicted median | median of pred / meas |
|---|---|---|---|---|
| vectorized, `ji` = 128 | 37 | 42.2 | 45.8 | 1.083 |
| vectorized, `ji` = 32/64 | 106 | 52.4 | 65.3 | 1.495 |
| "vectorized", `ji` ≤ 16 (scalar code) | 100 | 25.5 | 19.4 | 0.734 |
| scalar, `ji` innermost | 235 | 7.7 | 8.4 | 1.100 |
| scalar, `ii` innermost | 263 | 6.6 | 8.4 | 1.416 |
| scalar, `k`/`ki` innermost | 259 | 7.1 | 6.8 | 0.906 |

The whole table and Spearman 0.853 come from one command:
`coffee-ten-bench model compare --from bench/results/2026-10-22-hypothesis-a-run3.jsonl
--peak bench/results/2026-10-21-561ede3-peak.jsonl --group random` (without `--group`, all
1013 records: Spearman 0.858). The same groups were off by about the same
amount in run 2, so these are the model's stable errors, not noise of one sample.

## Model group errors after C4 (v1.4–v1.7)

C4 asked for the median of predicted / measured to lie in [0.83, 1.20] for each of the three
groups that were off in run 3, by fixing the model's formulas, not by fitting. Each fix is a
measured `bench/peak` point for a code shape the model had wrong, plus a rule that recognizes
that shape in the compiled nest:

| version | PR | what it adds | peak point |
|---|---|---|---|
| v1.4 | #82 | separate load and store ceilings for vector code | `fma_store` (P4) |
| v1.5 | #85 | a point for fully unrolled scalar bodies with unit stride | `fma_mem_acc_scalar_unrolled` (P5) |
| v1.6 | #90 | "reload": a rolled scalar loop walks C at a stride that is a multiple of 256 B and the next loop out reloads it | `fma_col_reload` (P6) |
| v1.7 | #93, #94 | "colload": an unrolled scalar body loads down a column at a stride that is a multiple of 256 B, one accumulator per copy | `fma_col_unrolled` (P7) |

The 256 B granule of v1.6 and v1.7 comes from hand sweeps of one array's row stride with the
code unchanged. A likely reading, not proven, is store-to-load matching and L1 bank conflicts
on addresses equal mod 256 B.

Model v1.7, 1000 random records, median of predicted / measured per group:

| group | run 3, P7 line | run 3, P6 line | run 2, P7 line | run 3 at the start (v1.3, P3) |
|---|---|---|---|---|
| vectorized, `ji` = 128 | 1.074 | 1.076 | 0.995 | 1.083 |
| **vectorized, `ji` = 32/64** | **1.171** | **1.191** | 1.031 | 1.495 |
| **"vectorized", `ji` ≤ 16 (scalar code)** | **1.147** | **1.155** | 1.136 | 0.734 |
| scalar, `ji` innermost | 1.150 | 1.165 | 1.112 | 1.100 |
| **scalar, `ii` innermost** | **1.122** | **1.127** | 1.116 | 1.416 |
| scalar, `k`/`ki` innermost | 0.910 | 0.886 | 0.877 | 0.906 |
| Spearman | 0.924 | 0.924 | 0.921 | 0.853 |

**C4 is met on run 3** with either peak line. The margin depends on which peak run calibrates
the machine: the vectorized 32/64 group is 1.171 with P7 and 1.191 with P6. Two caveats.
First, the rules were found on runs 2 and 3 and are checked on them, so this is in sample.
A new hypothesis A run with a new seed would be the out-of-sample check. Second, every point
is from one M5.

Run 3 with the P7 line: `coffee-ten-bench model compare --from
bench/results/2026-10-22-hypothesis-a-run3.jsonl --peak
bench/results/2026-10-28-adcf09c-peak.jsonl --group random` (without `--group`, all 1013
records: Spearman 0.927).

## Hypothesis A (spec §7.1), run 4: met; C4 out of sample: two of three groups

Run 4 (commit `736d82f`, N = 256, seed 5, machine P7) is the first run with the short-loop
codegen of C5 (#96): a vectorized loop of 8 or 16 iterations is now one vector iteration, and a
4×16 or 8×8 tile holds C in registers without `unroll` steps. The model that chose the top 10
was v1.7, not v1.8 (#97): the run was started from a checkout before #97 had merged. v1.7 and
v1.8 share rank 1 (the unrolled 4×16 tile), which was also the best of the ten, so the decision
does not depend on it. The comparison below uses v1.8, which models the new codegen.

| | schedule | median [p25–p75], GFLOP/s |
|---|---|---|
| best random (candidate 483 of 1000) | `split i 8, j 8, k 64; reorder io, ko, jo, ki, ii, ji; vectorize ji` (8×8 tile, no unroll steps) | **116.90** [116.05–117.17] |
| best of the model's top 10 (rank 1) | 4×16 register tile | **119.43** [118.14–120.00] |

**Ratio 1.022: the criterion is met, by a small margin.** With C5, random search meets register
tiles often: its best-so-far was 90.6 GFLOP/s after 16 candidates, 105.1 after 32, 114.4 after
256 and 119.4 from candidate 483 on. In run 3 it stayed at 86–87. The model's advantage now
comes from reaching the tile at the first measurement, not from finding something random
search cannot.

**C4 out of sample.** Model v1.8 against run 4's 1000 random records, P7 line (`model compare
--from bench/results/2026-10-30-hypothesis-a-run4.jsonl --peak
bench/results/2026-10-28-adcf09c-peak.jsonl --group random`):

| group | n | measured median | predicted median | median of pred / meas |
|---|---|---|---|---|
| vectorized, `ji` = 128 | 50 | 43.7 | 45.4 | 1.037 |
| **vectorized, `ji` = 32/64** | 96 | 55.0 | 56.7 | **1.109** |
| **"vectorized", `ji` ≤ 16** (vector code since C5) | 96 | 35.9 | 56.7 | **1.227** |
| scalar, `ji` innermost | 265 | 7.9 | 8.4 | 1.117 |
| **scalar, `ii` innermost** | 249 | 7.3 | 8.4 | **1.091** |
| scalar, `k`/`ki` innermost | 244 | 7.6 | 6.9 | 0.875 |

Spearman 0.915 (median error ×1.13). Two of C4's three groups hold out of sample. The `ji` ≤ 16
group misses at 1.227, and that is the group whose code changed with C5. Most of its error is
in the 40 store-bound rows that have a rolled `ii` directly above the short vector `ji`. In 33
of them `ii` is 16 to 128 with `k` or `ki` directly outside it, and they measure 12–40 GFLOP/s
(median per `ii`) against a predicted store ceiling of 56.7. This is the
column-reload shape of v1.6 (C's row is reloaded at the next `k` at a 1 KiB stride) in vector
code. The v1.6 rule covers only scalar code. Not yet measured in isolation.

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
- **A kernel runs slow for about 100 ms after its thread has been idle** (waiting for clang,
  or one of `run.sh`'s pauses): it starts at 25–50% of its steady rate (M5, N = 256, naive,
  vectorized and register-tile kernels alike). S1–S3 warmed each variant for one 10 ms
  sample only, so their first rounds ran slow. With 30 interleaved rounds this moves the
  medians little, but it is a likely cause of the `drift` flags on the first series after a
  pause (matmul N = 64 naive in S2 and S3; not verified). It ruined `search`'s 6 short
  samples per freshly compiled candidate (hypothesis A run 1). From #61 every variant runs
  300 ms before round 1.
- **Not measured:** NumPy/Accelerate (it may use the matrix units; a comparison would not
  measure the compiler), multiple threads, N above 1024, other machines. L1 and L2
  sizes are assumed, not measured.
