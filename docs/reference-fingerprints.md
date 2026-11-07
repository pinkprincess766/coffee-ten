# Frozen interpreter outputs (N16)

`coffee-ten-interp/tests/fixtures/reference-v1.tsv` pins all 32 N7 corpus outputs in
both f32 and f64, captured from the unchanged interpreter at main
`a9077973bef1dc582e5706a33fd4dbbcaec21b32`. Cases cover matmul, reductions, elementwise
expressions and chains, including signed-zero inputs. They exclude platform-libm `exp`
and NaN payloads. N16 runs only the reference; a compiler and interpreter changing in
the same way cannot make a moved reference baseline pass unnoticed.

The versioned FNV-1a fingerprint encodes `coffee-ten-reference-v1\0`, a NUL-terminated
dtype tag, rank, dimensions and element count as little-endian u64, then every output's
IEEE bits as little-endian u32/f32 or u64/f64. Shape, dtype and signed zero are retained.
This is a reproducibility fingerprint, not a security hash. Case IDs and fixture length
are checked, so removal or accidental reordering is rejected.

Independent sum and matmul canaries pin the sequential result of
`[1e20, 1, -1e20, 3]` to exactly 3 in both precisions. Pairwise grouping gives a different
result and fails the canary, even if a new kernel agrees with a reassociated interpreter.
Existing CI runs these integration tests in debug/release on Linux and macOS main runs.

```sh
cargo test -p coffee-ten-interp --test reference_fingerprints --locked
cargo test -p coffee-ten-interp --test reference_fingerprints --release --locked
```

There is no automatic fixture-update mode. Changes to the pinned outputs require a
**separate PR explaining each contract change**, the old/new bits, why changing the
reference is justified, and the affected cases. Do not regenerate these fingerprints
merely to make a failing implementation pass. Changing the N7 corpus or its grammar
also requires reviewing this baseline explicitly.
