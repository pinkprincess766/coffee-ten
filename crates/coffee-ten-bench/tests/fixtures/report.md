# Run report

Input fingerprint: `9214d54cb4c0647c`.

95% percentile bootstrap: 257 resamples, seed 42. Intervals describe supplied rounds, not process restarts or winner-selection uncertainty.

## Protocol 1: matmul, n=2

- space: v1
- space_hash: 0011223344556677
- source_hash: ffeeddccbbaa9988
- cpu: fixture CPU
- clang: fixture clang
- harness_profile: release

**Quick smoke data; not a performance measurement.**

Records: 3. Measurement: 56 ns. Compilation API work: 650 ns (includes reference; summed work, not wall time).

| Rank | Group | Schedule | Median GFLOP/s | 95% CI | Rounds | Drift |
| --- | --- | --- | ---: | --- | ---: | --- |
| 1 | random | vectorize(1, j) | 4.000000 | [2.000000, 8.000000] | 3 | false |
| 2 | random | split(1, j, jo, ji, 2) | 4.000000 | [2.000000, 8.000000] | 3 | false |
