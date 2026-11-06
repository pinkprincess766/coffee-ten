# Seeded compiler checks (N7)

N7 generates bounded programs over the **existing** API, as confirmed by Nikita on
7 November 2026: matmul, axis reductions, unary/elementwise operations and broadcast.
There is no new public einsum API. This is the differential-testing foundation; preserving
metamorphic transformations (N8) and automatic failure minimization (N9) follow separately.

## Corpus and generator

`tests/fuzz/ct_fuzz_cases.py` is a stdlib-only standalone generator. The checked-in
`tests/fuzz/corpus-v1.tsv` uses seed `0xc0ffee01` and 32 cases, eight per family:

- matmul, optionally with computed operands, bias/ReLU or a broadcast reduction epilogue;
- Sum/Max and compositions such as `x - max(x, axis)`;
- elementwise expressions with a trailing broadcast operand, including safe division;
- matmul -> Sum -> matmul chains with shared intermediates and optional epilogues.

Families and pre/post-operation coverage are stratified (including seed-based rotation);
dimensions, input bits and most reduction axes are seeded. Rank-3 last-axis reductions
are pinned for coverage. This is not a uniform distribution over arbitrary expressions.
Each dimension is 1..4, each input has at most 64 elements, and graphs have at most
24 nodes. The generator accepts 1..128 cases for local use; the CI corpus stays fixed at 32.

Each TSV row contains case ID, u64 seed (hex), family, dimensions, axis, pre-operation,
post-operation and an input fingerprint. Pre-operations are identity, multiply by f32 0.1,
ReLU and negation. Exact family/post mappings live in the two test builders. Parsers reject
unknown families, invalid shapes/axes, out-of-bound choices and wrong input fingerprints.
Input values are finite f32 bit patterns with diverse signs/magnitudes/mantissas plus
signed zeros. Division uses `b*b + 1`, so its denominator stays positive and finite.
Exp/libm-specific comparisons and NaN arithmetic are outside this finite bitwise corpus;
their existing dedicated tests remain in place.

The test protocol uses xorshift64*: initial state is `seed * 0x9e3779b97f4a7c15` modulo
2^64 (zero seed substitutes that constant), shifts are right 12 / left 25 / right 27,
and output multiplies by `0x2545f4914f6cdd1d` modulo 2^64. Per-input streams mix the case
seed with input ordinal using `0xd1b54a32d192ed03`. The implementation specifies all bit
operations and does not use Python's global `random`, NumPy's RNG, or live-search RNG state.
FNV-1a 64 over length-delimited input names, shapes and little-endian f32 words is a
cross-language reproducibility fingerprint, not a security checksum.

Regenerate/check the manifest without building the extension:

```sh
python3 tests/fuzz/ct_fuzz_cases.py --check tests/fuzz/corpus-v1.tsv
python3 tests/fuzz/ct_fuzz_cases.py --seed 17 --count 8
```

## Execution and reproduction

Rust's test-only parser/builder is `crates/coffee-ten-codegen-c/tests/support/fuzz_case.rs`;
`tests/seeded_fuzz.rs` lowers each graph, compiles C with TEST_FLAGS and compares every
result bit with `run_f32`. Python's `test_seeded_fuzz.py` reads the same manifest, verifies
byte-exact regeneration, builds through `coffee_ten`, compiles and compares with `ct.run`
using uint32 views. Both sides verify input fingerprints before executing a case.
No tolerance replaces the bit comparison. No benchmark compiler flags, reference rules,
default schedule spaces, search behavior or project dependencies change.

```sh
cargo test -p coffee-ten-codegen-c --test seeded_fuzz --locked
python -m pytest crates/coffee-ten-py/tests/test_seeded_fuzz.py -q
```

Failures print a full replay TSV row, seed and, for numerical failures, lowered IR.
Set `COFFEE_TEN_FUZZ_CASE` to that complete TSV row to execute only the reproduced case
with either driver. The checked-in corpus integrity tests still check the whole manifest.
N9 persists and shrinks output mismatches as described in [fuzz-minimization.md](fuzz-minimization.md);
the checked-in corpus is not silently modified.
Keep the source commit with a replay artifact because its grammar/builders define semantics.

Drivers enforce a 120-second elapsed budget between cases. This is a workload guard, not
a preemptive timer inside clang. The enclosing Rust debug/release and Python test steps
have 10-minute CI timeouts, so a stuck subprocess cannot hold a test step indefinitely.
The existing workspace tests and Python test discovery include the corpus automatically;
no required checks are skipped and no new dependency or test plugin is installed by N7.
