# Schedule space (N19)

The default is **V1**. `--space v2` opts into additional candidates and a different N2
key. Enumeration never times, models, ranks or filters candidates by measured speed.
Random search samples without replacement from the ordered legal list. The exact list
order affects seeded sampling and is part of the space fingerprint.

```sh
cargo run --release -p coffee-ten-bench -- space --n 256
cargo run --release -p coffee-ten-bench -- space --n 256 --space v2
cargo run --release -p coffee-ten-bench -- space reduce \
  --shape 2,7,5 --axis 1 --op sum --space v2
```

## Matmul axes and tile choices

`A[M,K] @ B[K,N]` has an initialization nest (0) and an update nest (1). The lowered
update starts in `i,k,j` order: i selects an output row, j an output column, and k is the
reduction. Schedules target nest 1; initialization remains unchanged.

V1 chooses unsplit or a tile from `[8,16,32,64,128]` for each axis. A factor must divide
the axis extent and be strictly smaller than it. Split names are `io/ii`, `jo/ji` and
`ko/ki`; steps are written in i,j,k order. Enumeration nests choices in i,j,k order,
then emits all permutations of outer loops followed by all permutations of inner loops.
For each order it emits the scalar schedule, then its explicit vectorized variant when
the innermost loop is j or ji. These divisor-only candidates need no remainder guards.

Additional register tiles use heights `[2,4,8]`, widths `[8,16,32]`, four f32 lanes and
at most 24 vector accumulators (`ri*rj/4 <= 24`), with divisible extents. Their fixed
order is `io,jo,k,ii,ji`; ji is vectorized and ii/ji unrolled. No k split or extra
register-tile permutation is introduced. The 24-register limit is a fixed assumption,
not fitted to measurements.

Each schedule goes through the existing legality checker; refused schedules are counted
as dropped. Candidates retain their arithmetic `OrderClass`. Reversing split reduction
loops can change arithmetic order and is classified accordingly; accepting a candidate
does not label it bit-preserving automatically. Default enumeration and classes remain
the historical baseline. TEST_FLAGS disable FMA contraction; BENCH_FLAGS permit it and
the harness uses the existing error-bound contract.

## V2 proposals

N5 reads OS cache sizes once, unions the proposed factors with the legacy factors and
appends legal new combinations after the complete unchanged V1 prefix. Register tiles
are retained. Cache values propose sizes; they do not remove existing candidates or
choose a winner. Missing/invalid sysctl values fall back to legacy factors, with a
distinct V2 hash domain. See [cache-tiles.md](cache-tiles.md) for the exact formula,
fallback, resolved factors and key fields. Hardware-dependent V2 counts are deliberately
not presented as universal constants.

## Reduce listing

N6's separate `space reduce` command lists a naive candidate in V1. For a non-last
reduction axis, V2 adds `r` before the contiguous independent output axes, retaining
the increasing r sequence for every output. Rank-one/last-axis cases and singleton Max
without an update do not receive redundant reorderings. See
[reduce-space.md](reduce-space.md). This listing does not add reduce graphs to the
matmul search/replay driver or change the default lowerer.

## Fingerprints and compatibility

For matmul, FNV-1a 64 consumes the domain `coffee-ten-matmul-space-v1\0` (or v2), then
each canonical schedule in enumeration order: UTF-8 byte length as little-endian u64,
UTF-8 text, and one arithmetic-class byte (0 preserving, 1 reordering). This is a
reproducibility fingerprint, not a security hash. N2's full key additionally records
kernel/shape/source/toolchain/acquisition settings; V2 records resolved cache and factors.
Equal space hashes alone are insufficient for replay or run comparison.

These V1 anchors are checked both against the implementation and this document in CI:

| Space | N | Legal candidates | Space hash |
| --- | ---: | ---: | --- |
| v1 | 16 | 133 | f659b70a6a6219db |
| v1 | 256 | 7334 | 252d60d1d76478c7 |

The pre-N5 regression also pins sampled schedule fingerprints for seeds 0, 1, 17 and
u64::MAX. Changing defaults, enumeration order or class encoding requires an explained
baseline change; V2 additions must not update V1 anchors merely to satisfy a test.
Reduction listing uses its own shape/op-aware fingerprint, not this matmul hash.
