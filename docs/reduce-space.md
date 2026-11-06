# Outer reduction schedule (N6)

`coffee-ten-bench space reduce --shape 2,7,5 --axis 1 --op sum --space v2`
lists the naive schedule and `reorder(1, _i0, r, _i2)`. `--op max` uses the same
transformation when there is an update nest. Without V2, only the naive candidate is listed.
This is an additional schedule proposal, not a change to the default lowerer or search.

For a row-major reduction with kept dimensions before and after the reduced axis, the
default order is kept-prefix, kept-suffix, r. V2 proposes kept-prefix, r, kept-suffix.
The innermost kept dimension now walks adjacent elements rather than advancing the
reduction stride. For each output element, r still runs strictly in ascending order.
No partial sums, reassociation, vectorization directive or FMA-policy change is introduced.
`apply_classified` must classify the candidate as preserving. Its legality check is extended
only for accumulator-first Max updates with no other target-buffer reads, injective output
indices, and unchanged relative reduction-loop order. Swapping split reduction loops is
still refused for Max; its order-sensitive NaN/zero behavior is not treated as reassociation.

Only the update nest is reordered. Sum retains +0 initialization and visits every r;
Max retains its first-element initializer, accumulator-first comparison and r+1 indexing.
Max of an axis of extent one has no update nest. Rank-one and last-axis reductions
have no useful independent suffix and therefore have only the original candidate.

The returned schedule can be passed to existing `apply`/`apply_classified` or the Python
compile API, for example `ct.compile(ct.sum(x, axis=1), schedule="reorder(1, _i0, r, _i2)")`
for the example shape. The new `space::enumerate_reduce` API returns the lowered kernel
and candidates for Rust callers. Neither it nor the listing command measures anything.

Reduction listing hashes include their graph text, ordered schedules and space version in
a separate domain. They are listing fingerprints, not N2 matmul measurement-table keys.
The matmul benchmark/search/replay modes remain matmul-only; this PR does not create
synthetic reduction measurements or claim integration into their replay tables.
V1/V2 matmul candidate lists and frozen V1 hashes/seeded samples remain unchanged.

Tests check the emitted schedule, unchanged init/body and preserving classification.
End-to-end tests compile original/reordered kernels with TEST_FLAGS and compare both with
the f32 interpreter on all axes of rank-1/2/3 shapes, cancellation-sensitive sums, signed
zeros and NaN Max inputs, including singleton axes. Existing default-space regressions run
alongside them. No dependencies or numerical reference changes are needed.
