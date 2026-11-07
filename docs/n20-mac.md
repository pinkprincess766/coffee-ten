# N20: complete default-space acquisition on the Mac

Acquired 7 November 2026 from clean source commit
`d0dc132e208d25a38ed200fbdbe8c3a5cd219ee2`, release profile, Apple M5 / Mac17,4,
macOS 27.2, Apple clang 21.0.0 (`clang-2100.4.1.1`), Rust 1.98.0.
The full matmul N=256 V1 space was measured: 7334 candidates plus one naive reference.
Input/selection seed 1, five rounds, minimum 10 ms per round. The existing acquisition
policy is unchanged: 300 ms for the first variant of each phase, then 10 ms per candidate.
BENCH_FLAGS still allow FMA contraction; results use the existing verification bound.

## Data and reproducibility

All paths below are under `bench/results/` with prefix `2026-11-07-n20-v1-n256`:

- `.jsonl`: complete live measurement table, 7335 records / 36675 raw rounds.
- `.report.md`: unedited N11 top-10 report and seeded median bootstrap intervals.
- `.curves.csv`: N13 offline random-baseline curves, budgets 0..1000, policy seeds 1,2,3,5.
- `.replay.txt`: N3 replay of all 7334 candidates, seed 1.
- `.summary.json`, `.historical.csv`: descriptive totals and historical comparison.
- `.environment.json`: start environment, power settings, source state and acquisition times.

The adjacent `2026-11-07-n20-v1-generator-provenance.json` records the byte-identical
V1 generator block in all four historical source commits. The manifest lists SHA-256
and byte sizes of the published artifacts. Raw-table SHA-256:
`e4766ae970b3b0267bc0167b7b9fd3217b3ed706554ee911bf562ac2eb680ac2`.

N2 key anchors: space `252d60d1d76478c7`, source `ba0dd81f6b56e86d`,
graph `a97c1645b9dc930c`; the complete key is retained in every record.
The V1 generator block is byte-identical to commits `c435f89`, `702dcf1`, `726cc1d`,
`736d82f`. The first 1000 N20 schedule texts exactly match run 1's ordered sample.
Historical files did not record an N2 key, so no missing historical metadata is invented.

```sh
# Acquisition at the source commit above, on the matching Mac/toolchain:
cargo build --release -p coffee-ten-bench --locked
caffeinate -i target/release/coffee-ten-bench search --n 256 --budget 7334 \
  --seed 1 --samples 5 --min-sample-ms 10 --space v1 --log new-table.jsonl

# Analysis of the committed table (no kernel builds or new timings):
python3 bench/validate.py bench/results/2026-11-07-n20-v1-n256.jsonl
target/release/coffee-ten-bench report \
  bench/results/2026-11-07-n20-v1-n256.jsonl --top 10 > report.md
target/release/coffee-ten-bench curve --n 256 --budget 1000 \
  --samples 5 --min-sample-ms 10 --space v1 --seeds 1,2,3,5 \
  --replay bench/results/2026-11-07-n20-v1-n256.jsonl > curves.csv
```

Replay requires the exact source/profile/machine/toolchain key; use the acquisition
source commit when rebuilding a compatible executable. N11 report reading does not
require the current host to match the acquisition machine. Its generated Markdown was
reproduced byte for byte from the committed log.

## Observations

| Run / prefix | Input seed | Random candidates | Naive GFLOP/s | Best sequential random GFLOP/s |
| --- | ---: | ---: | ---: | ---: |
| A1 | 1 | 1000 | 7.812 | 108.841 |
| A2 | 2 | 1000 | 7.718 | 113.288 |
| A3 | 3 | 1000 | 7.425 | 87.022 |
| A4 | 5 | 1000 | 7.781 | 119.350 |
| N20 first 1000 | 1 | 1000 | 7.795 | 123.776 |
| N20 complete V1 | 1 | 7334 | 7.795 | 123.776 |

This table compares sequential five-round medians, not the historical 30-round final
decision records. It is descriptive: compiler builds differ, A1-A3 used macOS 27.0.1,
input seeds differ, and old logs have neither raw rounds nor full keys. No cross-version
ratio confidence interval or causal speedup claim follows. Historical model-top values
are retained separately in the CSV and are not compared as equal-budget searches.

The best observed candidate was first reached at step 732:

```text
split(1, i, io, ii, 32); split(1, j, jo, ji, 8); split(1, k, ko, ki, 8);
reorder(1, io, ko, jo, ii, ki, ji); vectorize(1, ji)
```

Median 123.775949 GFLOP/s; N4's nominal 95% percentile bootstrap interval is
[123.327680, 124.171798], 10000 resamples, seed 1. This interval describes those five
rounds conditional on the selected winner; it does not correct selection among 7334
candidates or describe repeat-process uncertainty. With only five rounds the existing
drift check is **not assessed**, despite the stored `drift:false` value.

Measured-round time totals 452.298467818 seconds. Compiler API work, including the initial
reference, totals 33174.493428760 seconds: this is summed overlapping work, not compiler
wall time. The table has 7333 variant API misses and two hits; the separate initial
reference call is included in the work total. The wall-clock interval from environment
capture start through the last record was 4337.782261 seconds, including metadata
capture/compilation/setup/validation/warmup/logging. These
are distinct quantities; `compilation_wall_ns` remains null.

The offline curves use one table with **input seed 1**. Policy seeds 2,3,5 change only
the traversal; they are not new live acquisitions or recreations of historical input
seeds 2,3,5. Stored costs are replay estimates, not future wall-time guarantees.

## Conditions and V2 scope

Chrome and Grok Bot were closed normally with Nikita's permission. No other user work
was running, per Nikita. The Mac was on AC, low-power mode off; no thermal/performance
warning was recorded at the start. Ordinary OS services remained. During timing there
were no builds, tests or source edits; only lightweight log-progress checks. This is a
record of acquisition conditions, not a claim of a noise-free machine.

`2026-11-07-n20-v2-n256-space.txt` is a **listing only**, not a V2 measurement. The
reported caches (64 KiB L1, 6 MiB L2, 128 B line) resolve factors
`[8,16,32,64,128,512]`; 512 is unusable at N=256, so the 7334 schedules equal V1's list.
V2 still has its own hash `cb87764f049c4fea` and key. No V2 performance result is claimed
or mixed into this default-space baseline.

## Validation

The stdlib schema checker accepted all 7335 rows. Additional checks required unique
candidate schedules, one common complete N2 key, five ordered rounds per row and every
round at least 10 ms. N3 validated the complete table, including exact raw-rate and
aggregate bits, costs, groups and candidate membership. Full-seed-1 replay and four N13
curves completed successfully. The report reproduced byte for byte. No dependencies,
search policy, arithmetic order or measurement flags changed for this acquisition.
