# Saved-run reports (N11)

```sh
cargo run --release -p coffee-ten-bench -- report run.jsonl --top 10 > report.md
```

This reads the log and writes Markdown to stdout. It compiles no kernels, acquires no
timings and changes no input files. Peak rows are checked as in `bench/validate.py`, then
left out of the tables. Offline replay curve rows are
rejected. N1 raw rounds and N2 protocol metadata are required: legacy aggregate-only logs
cannot supply confidence intervals or measured costs and produce an explicit error.

Each distinct protocol key gets a separate table, in first-seen order. This includes
different kernels, sizes, spaces, source builds, input seeds and sampling protocols.
Repeated schedules remain distinct observations; rounds are never pooled. Within each
table, rows rank by descending recorded lower median, with acquisition order breaking
ties. Search groups remain visible, including final decision rounds. This is descriptive
ranking of the supplied rows, not a new search decision.

The 95% percentile bootstrap uses N4, with defaults `--seed 1 --resamples 10000`.
Intervals cover uncertainty across the supplied rounds, not independent process restarts
or the effect of choosing the best observed candidate. `--top` defaults to 10; each table states
`Showing k of m records` so a truncated table is not mistaken for the whole run. Quick
logs carry a prominent smoke-data label. Drift flags are retained; with fewer than 10
rounds (two disjoint windows of 5) the check cannot run, and the Drift column says
`not evaluated` instead of `false`. Markdown table text is escaped, and an input-byte fingerprint identifies the exact log.

Time totals cover **all** rows of each protocol, including rows outside top-k. Measurement
time is the sum of raw round durations. Compilation is summed successful compiler API work,
including each recorded reference compilation; parallel work is not elapsed wall time.
Warmup/setup are excluded from measurement costs. The reader checks raw rate/timing,
aggregate bits, raw acquisition order, metadata consistency, drift and cost totals before
rendering. Malformed inputs fail without partial output.
Each line must hold exactly one JSON object (LF or CRLF endings; no byte-order mark); a
blank or whitespace-only line is an error, as in `bench/validate.py`. Errors cite the
physical line number.

The saved synthetic fixture `crates/coffee-ten-bench/tests/fixtures/report.jsonl` is
checked against a Markdown golden through both the API and the actual command in CI.
`report --compare` of `report-b.jsonl` against it is pinned the same way, independent
(`report-compare.md`) and paired (`report-compare-paired.md`).

## Comparing runs (N12)

```sh
cargo run --release -p coffee-ten-bench -- report --compare A.jsonl B.jsonl
```

Each file must contain exactly one N2 protocol key, and the keys must be equal in full.
The key must also be complete and closed: every field `ProtocolKey::to_json` writes is
required with its type (`cache` and `tile_factors` only for non-V1 spaces), and an unknown
field is rejected. Two files that both lack, say, `source_hash` are therefore not comparable,
even though their incomplete keys are equal. Errors name the offending field.
Mixed-key files are rejected rather than silently choosing a compatible subset. Both errors
list every differing key field with its two values as JSON, e.g.
`source_hash (A: "ffee…", B: "1111…")`. The first
row with the highest lower median is the winner of each file. The reported ratio is
`median(best A) / median(best B)`, with the 95% bootstrap interval and both schedules.

Separately acquired runs use **independent** resampling by default. Equal round indices
across files do not establish paired observations. Use `--paired` only when the acquisition
design actually paired the observations; this is an explicit caller assertion, printed in
the report. Paired mode uses N4's shared-index resampling and checks matching index sets.
`--seed` and `--resamples` apply to either mode. Neither interval corrects winner-selection
bias or captures variation across process restarts. No search/acquisition behavior changes.

### Example

The golden `crates/coffee-ten-bench/tests/fixtures/report-compare.md`, pinned byte for byte
through the API and the binary by `tests/report_compare_golden.rs`, is the output of:

```sh
cargo run --release -p coffee-ten-bench -- report --compare \
  crates/coffee-ten-bench/tests/fixtures/report-b.jsonl \
  crates/coffee-ten-bench/tests/fixtures/report.jsonl --resamples 257 --seed 42
```

```markdown
# Run comparison

A fingerprint: `081ee30d6c09ae9c`. B fingerprint: `607f6a4ff4823fc0`.

**Quick smoke data; not a performance measurement.**

Ratio: median(best A) / median(best B). 95% percentile bootstrap, 257 resamples, seed 42, independent resampling.

| Run | Best schedule | Median GFLOP/s | Rounds | Drift |
| --- | --- | ---: | ---: | --- |
| A | vectorize(1, j) | 8.000000 | 3 | not evaluated |
| B | vectorize(1, j) | 4.000000 | 3 | not evaluated |

Median ratio A/B: 2.000000; 95% CI [0.500000, 8.000000].

Intervals are conditional on the selected winners and supplied rounds; they do not adjust for winner selection or independent process restarts.
```

- The fingerprints identify the exact bytes of each input file, as in a single-run report.
- The bold smoke-data line appears when the shared protocol key has `quick: true`.
- **Run**: `A` is the first file, `B` the second.
- **Best schedule**: the winner of that file, the first row with the highest lower median;
  `(naive)` for the empty schedule.
- **Median GFLOP/s**: the winner's recorded lower median.
- **Rounds**: how many raw rounds the winner has; these are what the bootstrap resamples.
- **Drift**: the winner's drift flag, or `not evaluated` with fewer than 10 rounds, as in a
  single-run report.
- **Median ratio A/B**: `median(best A) / median(best B)` and its 95% interval. With
  `--paired` the same inputs give the golden `report-compare-paired.md`: the method reads
  `paired (asserted by caller) resampling` and the interval is `[2.000000, 2.000000]`,
  because the fixture's rounds are exactly proportional.

## Exit status

`report` and `curve` exit 0 on success, 2 when the command line is wrong (the message ends
with the usage), and 1 when the input is rejected (unreadable file, invalid record, protocol
mismatch, or a value such as `--top 0` that parses but is refused by the analysis).
