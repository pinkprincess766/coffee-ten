# Opt-in cache-derived tiles (N5)

`space --n N --space v2` lists the versioned matmul space without measuring anything.
`search --n N --budget M --space v2` samples that space with the existing seeded random
baseline. `search --replay table.jsonl --space v2` validates and uses a complete V2 table.
Omitting `--space` (or using `--space v1`) keeps the old space, order and seeded sample.

```sh
cargo run -p coffee-ten-bench --release -- space --n 1024 --space v2
```

## Hardware inputs and formula

Only V2 queries `sysctl -n` for `hw.l1dcachesize`, `hw.l2cachesize` and `hw.cachelinesize`
on macOS. Values are bytes. They are resolved once into the invocation's space config,
then reused for enumeration, every record's key, and replay validation. These are the
generic values reported by the OS, not measured capacities or a guarantee of which core
executes a benchmark. No calibration run or learned cost model is used.

The capacity-only proposal uses these explicit assumptions:

- f32 elements occupy 4 bytes; three square blocks model A, B and C;
- reserve half of each cache for other data/traffic;
- propose a cache-line width in floats, `line_bytes / 4`;
- for each cache capacity C, compute `floor(sqrt(floor(C / 24)))`, then round down to a
  power of two, provided it spans at least a cache line. This follows
  `3 * 4 * t^2 <= C/2`. Integer square root and shifts avoid floating-point variation.

This is a simple candidate-generation heuristic in the spirit of cache-blocked GEMM, not
a full BLIS packing/microkernel model or a claim of optimal tile sizes. The half-cache
assumption is fixed and has not been fitted to measurements.

Example used by the tests: L1 = 32 KiB, L2 = 2 MiB, line = 64 B proposes 16, 32, 256.
Unioning with the original `[8,16,32,64,128]` yields `[8,16,32,64,128,256]`.
Factors below 2 are ignored. All proposals are suggestions, not a cache-based filter.

Missing/unparseable/invalid inputs (including non-power-of-two line sizes, zero values or
L2 smaller than L1) fall back to the original values. Off macOS no cache query is made.
V2 prints the resolved factors and cache inputs, or an explicit fallback marker.

## Enumeration and compatibility

V1 enumeration is unchanged. V2 keeps all V1 candidates in their original order as a
prefix, then appends legal combinations containing at least one newly proposed factor.
The existing legality rules still apply: split factors must divide the extent and be
strictly smaller than it. Existing register-tile candidates remain present unchanged.
No candidate is scored, removed, or skipped based on cache capacity or measurements.
Some dimensions admit no new factors; their V2 candidate list can equal V1.

V1 `space_hash` retains its exact original hash domain and ordered contents. V2 uses a
separate domain, so it has a different hash even on fallback. Its protocol key additionally
contains `space: v2`, `cache` (`{l1_bytes,l2_bytes,line_bytes}` or `null`) and the resolved
`tile_factors` array. Different hardware inputs are different full keys even if they yield
the same schedules. Replay checks the locally resolved V2 config against the table.
Source fingerprints change with this implementation; the frozen regression assertion is
about the schedule-space hash and selection order, not the source hash.

V2 currently supports space listing, random search and random replay. Combining it with
model-top/local modes fails before kernel compilation; those rankers describe V1. Existing
V1 model modes continue to work. The general harness stays V1. `--budget`, warmup, round
order, compiler flags, accumulation order and the default random-search policy are unchanged.

Tests pin V1 hashes/counts and samples for several seeds to main `ee9393b` (before N5),
check formulas and fallback, enforce the V1 prefix and uniqueness of V2 schedules, compare
cache-derived splits on each axis with a scalar f32 reference bit for bit using TEST_FLAGS,
and round-trip a complete expanded toy table through live search/replay.
These are correctness checks, not new performance results. Full Mac campaigns remain N20.
