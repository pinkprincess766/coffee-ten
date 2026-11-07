# Compilation cache verification (N14)

The requested cache already exists in `coffee-ten-codegen-c/src/compile.rs`:
`compile_with_flags` emits C, fingerprints the source, the ordered and NUL-delimited
compiler flags, and the full `clang --version` output. It reuses the corresponding shared
library on a hit. The benchmark uses this API and the persistent temporary directory
`coffee-ten-bench-cache`; N2 already logs successful API time and `cache_hit` separately
for the variant and the initial correctness reference.

N14 therefore adds regression evidence, not a second cache or a selection policy.
The repeated-build test uses a fresh private directory, checks cold miss/cached hit,
identical library path, bytes and modification time, then executes both handles and
compares output bits. Source changes, flag changes, flag order/boundaries and compiler
version changes get different keys. Existing tests cover actual flag-specific libraries
and concurrent compilation. No measurements or correctness checks are skipped on a hit.

## Diagnostic on the Mac, 7 November 2026

Two consecutive release-harness quick calls on Apple M5 / Mac17,4, macOS 27.2,
Apple clang 21.0.0, matmul n=23, `vectorize(1, j)`, seed 1:

| Call | Variant compiler API ns | Hit | Reference compiler API ns | Hit |
| --- | ---: | --- | ---: | --- |
| First | 341700750 | false | 302804833 | false |
| Repeat | 137333 | true | 8226750 | true |

Raw records: `bench/cache/n14-cold.jsonl` and `bench/cache/n14-cached.jsonl`.
Both include all three timing rounds: the cache did not bypass measurements. These
are quick diagnostic records, **not kernel-performance measurements**. They were acquired
on a tracked-dirty tree containing this PR's cache-test changes, as recorded in the log;
the source fingerprint identifies that build. The cache implementation was unchanged.
The observed API-work decrease is evidence for this pair, not a CI timing threshold or
a promise of the same speedup on another host. CI asserts cache reuse and bit identity,
not a noisy wall-time inequality.

```sh
cargo run --release -p coffee-ten-bench -- --kernel matmul --n 23 \
  --variant 'vectorize(1, j)' --quick --log first.jsonl
cargo run --release -p coffee-ten-bench -- --kernel matmul --n 23 \
  --variant 'vectorize(1, j)' --quick --log repeat.jsonl
```

A pre-existing cache can make both calls hits; the unit regression creates its own empty
cache without deleting a user's cache. This is compiled-library reuse, not C9 schedule
selection caching.
