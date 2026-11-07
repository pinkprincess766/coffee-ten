# Saved-run reports (N11)

```sh
cargo run --release -p coffee-ten-bench -- report run.jsonl --top 10 > report.md
```

This reads the log and writes Markdown to stdout. It compiles no kernels, acquires no
timings and changes no input files. Peak rows are skipped. Offline replay curve rows are
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
or the effect of choosing the best observed candidate. `--top` defaults to 10. Quick
logs carry a prominent smoke-data label. Drift flags are retained. Markdown table text
is escaped, and an input-byte fingerprint identifies the exact log.

Time totals cover **all** rows of each protocol, including rows outside top-k. Measurement
time is the sum of raw round durations. Compilation is summed successful compiler API work,
including each recorded reference compilation; parallel work is not elapsed wall time.
Warmup/setup are excluded from measurement costs. The reader checks raw rate/timing,
aggregate bits, raw acquisition order, metadata consistency, drift and cost totals before
rendering. Malformed inputs fail without partial output.

The saved synthetic fixture `crates/coffee-ten-bench/tests/fixtures/report.jsonl` is
checked against a Markdown golden through both the API and the actual command in CI.
