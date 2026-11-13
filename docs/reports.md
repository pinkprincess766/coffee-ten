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

## Exit status

`report` and `curve` exit 0 on success, 2 when the command line is wrong (the message ends
with the usage), and 1 when the input is rejected (unreadable file, invalid record, protocol
mismatch, or a value such as `--top 0` that parses but is refused by the analysis).

## Raw rounds: schema and errors

`report` (and `replay`, which shares `crates/coffee-ten-bench/src/rounds.rs`) reads the
`rounds` array of every measurement record. The schema is the one written today; this
section only lists what the reader checks. Each element is an object with exactly these
fields:

| Field | Type | Rule |
| --- | --- | --- |
| `round_index` | unsigned integer | equals the element's position, from 0 |
| `elapsed_ns` | unsigned integer | > 0 and at least the protocol's `min_sample_ms` |
| `iterations` | unsigned integer | > 0 |
| `gflops` | finite number | > 0 and bit for bit `2·n³ · iterations / (elapsed_ns / 1e9) / 1e9` |

Errors are prefixed `report: line N: `, where N is the physical line, and exit with status 1.
In the messages below `i` is the zero-based position of the round in the array.

| Violation | Message | Pinned by |
| --- | --- | --- |
| an extra field in a round | `round i: unknown field <name>` | `tests/report_closed_fields.rs` |
| `rounds` absent / not an array | `missing rounds` / `expected array` | — |
| length differs from `samples` or the protocol's `samples`, or is 0 | `wrong raw round count: <len> rounds, samples <s>, protocol samples <p>` | `tests/report_rounds.rs` |
| a round is not an object | `round i round_index: expected object` | — |
| a field is absent | `round i <field>: missing <field>` | — |
| `round_index`, `elapsed_ns` or `iterations` negative, fractional or not a number | `round i <field>: expected unsigned integer` | `tests/report_rounds.rs` |
| `gflops` null, a string, or not finite | `round i gflops: expected finite number` | `tests/report_rounds.rs` |
| `round_index` out of order | `invalid raw round order: round i has round_index <k>` | `tests/report_rounds.rs`, `rounds.rs` unit tests |
| `elapsed_ns` or `iterations` is 0 | `invalid raw round timing: round i needs positive elapsed_ns and iterations` | `rounds.rs` unit tests |
| `elapsed_ns` below `min_sample_ms` | `raw round is below the protocol minimum duration: round i took <ns> ns, minimum <ms> ms` | `rounds.rs` unit tests |
| `gflops` ≤ 0 or not the rate of its timing | `raw rate does not match timing: round i stores <x>, timing gives <y>` | `tests/report_rounds.rs`, `rounds.rs` unit tests |

Then the record's own fields must equal what its rounds give, checked in this order
(`rounds.rs`, `aggregates_cost_and_drift_are_checked_in_that_order`):
`median`, `p25`, `p75`, `min` (`<name> differs from raw rounds`), `measurement_ns`
(`inconsistent measurement cost: measurement_ns <stored>, raw rounds sum to <sum>`) and
`drift` (`inconsistent drift: stored flag differs from raw rounds`).

`bench/README.md` describes `gflops` as "number or null" on the writer side. The reader does
not accept `null`: such a round is refused with `expected finite number`.
