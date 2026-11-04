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
| `space`, `space_hash` | `v1`; fingerprint of ordered canonical matmul candidates and arithmetic classes (order affects seeded sampling) |
| `source_hash` | Fingerprint embedded at build time from Rust/TOML files under `crates`, root Cargo.toml/Cargo.lock, toolchain/config and peak C source |
| `cpu`, `machine`, `os`, `arch`, `target_os` | CPU brand, hardware model, OS version, compiled architecture and OS |
| `clang`, `flags`, `rustc` | Compiler versions and benchmark C flags, including FMA contraction policy |
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
