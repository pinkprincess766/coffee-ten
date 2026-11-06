# Measurement protocol (N2)

The default search, candidate-count `--budget`, candidate order, warmup and floating-point
arithmetic are unchanged. Search already builds/checks a group in parallel before timing
its candidates one at a time. This document describes existing behavior and adds accounting;
it does not authorize a different search or warmup policy.

## Compatibility key

Each new harness/search record contains `protocol`, an object with these exact fields:

| Field | Meaning |
| --- | --- |
| `version` | Schema/protocol version, currently 1 |
| `kernel`, `n`, `graph_hash` | Kernel kind, square size, hash of unscheduled lowered graph |
| `space`, `space_hash` | `v1` by default, opt-in `v2`; versioned fingerprint of ordered canonical matmul candidates and arithmetic classes (order affects seeded sampling) |
| `cache`, `tile_factors` | V2 only: resolved cache inputs or null fallback, plus proposed factor values; see [cache-tiles.md](cache-tiles.md) |
| `source_hash` | Fingerprint embedded at build time from Rust/TOML files under `crates`, root Cargo.toml/Cargo.lock, toolchain/config and peak C source |
| `cpu`, `machine`, `os`, `arch`, `target_os` | CPU brand, hardware model, OS version, compiled architecture and OS |
| `clang`, `flags`, `rustc` | Compiler versions and benchmark C flags, including FMA contraction policy |
| `harness_profile`, `rustflags` | Rust harness profile and hex bytes of CARGO_ENCODED_RUSTFLAGS (N3); debug/release mixing is rejected |
| `samples`, `min_sample_ms`, `quick` | Effective sampling settings; quick data are smoke data, not measurements |
| `input_seed` | Seed for input matrices and round ordering |
| `warmup_ms`, `candidate_warmup_ms` | Full warmup and subsequent-candidate warmup durations; quick mode has its own durations |

Hashes use domain-separated, length-delimited FNV-1a 64: reproducibility fingerprints,
not security hashes. The source fingerprint describes built inputs, independently of Git
commit/dirty metadata; a documentation-only commit leaves it unchanged. Full key equality
is required to combine tables or compare measurements under this protocol. Unknown CPU or
compiler identity is not sufficient evidence of compatibility. Old aggregate-only logs
remain readable by existing reports but cannot supply a complete replay table.

The key describes the build inputs listed above, not external-library binary provenance or
thermal state. Preserve original logs and environment information when publishing results.
V1 has no cache-input fields and makes no cache queries. V2 has its own hash domain and
full key even on fallback; mixing V1/V2 or differing V2 cache inputs fails replay validation.

## Budget and cost fields

`--budget M` still chooses M distinct random candidates, capped at the space size; the naive
reference does not count. For comparing strategies, retain both candidate count and cumulative
`measurement_ns` (the sum of raw `elapsed_ns`). Report compilation separately, never subtract
it from measurement time or silently include it in a measurement-only budget.

`compilation_ns` measures the successful `compile_with_flags` API call for this variant,
including C emission, cache lookup, a clang invocation on a miss, and library loading.
`cache_hit` comes from the actual cache branch; a concurrent call that invoked clang is a
miss even if another call produced the same library. It is not inferred from a later file
existence check. The initial naive correctness reference is an additional call, represented
once in `reference_compilation` on the first record of the run (or naive search group).
Sum its `elapsed_ns` with variant `compilation_ns` for compiler API work; keep cache states.
No failed compile gets a successful measurement record.

Compiler calls can overlap in the existing parallel build path, so their duration sum is
work, not elapsed wall time. `compilation_wall_ns` is reserved as `null` pending an agreed
wall-time accounting protocol. No scheduling policy is changed to populate it.

Warmup, validation, input allocation, process startup, schedule generation, logging and gaps
between phases are excluded from `measurement_ns` and are not total end-to-end runtime.
Raw samples keep the original rate formula and aggregate rules. A complete-table replay
charges stored costs as estimates: it does not recreate live thermal drift or order effects.
The table acquisition order and first/subsequent warmup asymmetry remain relevant.

## Offline replay (N3)

`search --replay table.jsonl --n N --budget M --seed POLICY_SEED` reads a complete
single-run random-search table: exactly one `naive` reference and one `random` record for
every candidate of the keyed space (`v1` by default; pass `--space v2` for V2 data).
An optional peak line is ignored. Partial,
duplicate, unknown-schedule, mixed-group and mixed-protocol tables fail before output.
Raw round counts, order, finite rates, timing-derived rates, aggregate bits and measurement
costs are checked. Old logs without N1/N2 data cannot be used as replay tables.

The current build, local CPU/platform/compiler identity and effective CLI sampling settings
must match the table's full key. Run on the same measured platform with the same built
sources; unknown identity is rejected. The input seed is taken from the table, while
`--seed` now selects candidates only. The same seeded `sample` function serves live random
search and replay. `--budget` remains a candidate count. `--quick`, `--samples` and
`--min-sample-ms` must match acquisition settings. N3 supports the random baseline;
`--local`, `--model-top` and `--from` are rejected with replay.

The offline branch runs before QoS, `Reference`, kernel compilation and measurement.
Environment commands such as `clang --version` still establish compatibility; no generated
C is built or executed. Selection has no new measurement noise. Stored values remain tied
to the acquisition run, including its warmup and ordering effects.

With `--log replay.jsonl`, replay emits a separate schema, never live measurement records.
Use a dedicated output file. It must differ from the input table. Every row is:

| Field | Meaning |
| --- | --- |
| `kind`, `offline` | `replay_curve`, `true` |
| `protocol` | Full validated N2 key of the acquisition table |
| `table_hash` | FNV-1a 64 fingerprint of exact input file bytes (including any peak line) |
| `policy_seed` | Candidate-selection seed, distinct from protocol.input_seed |
| `step` | One-based number of selected candidates |
| `schedule`, `gflops` | Selected canonical candidate and its stored median |
| `best_schedule`, `best_gflops` | Best stored median so far; a tie keeps the earlier selection |
| `charged_measurement_ns`, `charged_compilation_ns`, `cache_hit` | This candidate's stored costs/cache state |
| `cumulative_measurement_ns` | Sum of selected candidate sample times plus naive sample time once |
| `cumulative_compilation_ns` | Sum of selected compiler API durations plus naive variant and initial correctness reference calls once |

These are charged acquisition costs, not the CPU time spent running replay. In particular,
compiler duration sums remain work rather than parallel wall time. Output contains no new
wall-clock timestamps, so the same table/config produces byte-identical curve JSONL.

A toy full-space smoke run (not a publishable benchmark) can be captured and replayed with
one unchanged build:

```sh
cargo build -p coffee-ten-bench --release --locked
target/release/coffee-ten-bench search --n 2 --budget 999 --seed 17 --quick --log toy.jsonl
target/release/coffee-ten-bench search --n 2 --budget 3 --seed 42 --quick --replay toy.jsonl --log toy-replay.jsonl
```

Full-space Mac measurements are intentionally deferred until N1–N3 are merged. No toy
rates should be presented as a performance improvement or compared with hypothesis A.
