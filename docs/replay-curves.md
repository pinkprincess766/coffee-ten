# Offline random-baseline curves (N13)

```sh
cargo run --release -p coffee-ten-bench -- curve \
  --n 64 --budget 200 --samples 5 --min-sample-ms 10 \
  --replay complete-table.jsonl --seeds 1,42,999 > curves.csv
```

Use the table's sampling settings and `--space v2` when appropriate. N3 validates the
complete table and exact N2 key before output: current machine/toolchain/source/profile
must match. This command performs no kernel compilation or measurements, and never
writes a log. A quick table requires `--quick`; it remains smoke data, identified in
the full protocol carried by each CSV row. `--log`, live search modes and `--seed` are
rejected; `--seeds` supplies the policy seeds (default `1`). Input generation seed stays
the one recorded in the table.

The baseline uses the **unchanged random-without-replacement sample** for each seed. CSV
contains every prefix, not just sparse checkpoints. Step zero is the naive reference;
steps 1 through budget count candidate measurements only, capped by space size. The
incumbent includes naive and retains the first winner on ties. This differs from N3's
candidate-only `replay_curve` summaries, which are unchanged.

Columns:

- `policy_seed`, `step`: curve identity and candidate count.
- `cumulative_measurement_ns`: stored round durations, with naive charged at step zero.
- `cumulative_compilation_ns`: stored compiler API work, including initial correctness
  reference and naive at step zero; summed work, not parallel wall time.
- `best_gflops`, `best_schedule`: best median observed so far, including naive.
- `table_hash`, `protocol`: input-byte fingerprint and complete N2 key.

CSV uses standard doubled-quote escaping for schedules and the embedded JSON key. Curves
are deterministic for a fixed table, settings and ordered seed list. They model the
stored table's costs, not newly observed time or a guarantee of future performance.
Different seeds are alternate traversals of one table, not independent hardware runs.

Tests pin prefixes, tie policy, costs and quoting on a synthetic replay, and exercise
multiple seeds against N3's complete tiny measured table without building new references.
