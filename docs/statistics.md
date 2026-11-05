# Raw-round bootstrap (N4)

The public analysis API is in `coffee_ten_bench::stats`:

- `bootstrap_median(rounds, config)` estimates a candidate's median GFLOP/s.
- `bootstrap_paired_ratio(numerator, denominator, config)` estimates
  `median(numerator) / median(denominator)`.
- `BootstrapConfig { resamples, seed }` defaults to 10,000 resamples and analysis seed 1.
  This seed is separate from input generation and search selection.
- `BootstrapInterval` returns `estimate`, `lower`, `upper`, nominal `confidence_level`
  (0.95), original sample count, resample count and seed.

These functions consume N1 `RoundSample` values. They do not run kernels, write measurement
records, change acquisition aggregates, hide drift, or choose candidates. The CLI/report
integration is N11/N12; N4 supplies the reusable statistical API. Logs without raw rounds
cannot be reconstructed from their median/IQR.

## Exact algorithm

1. Require nonempty raw data with unique round IDs and finite positive GFLOP/s values.
   Sort by round ID before resampling so input row order does not change seeded output.
2. Each replicate draws `n` indices with replacement using a private instance of the
   existing xorshift64* RNG. Rejection sampling avoids plain-modulo range reduction bias.
   Existing search/harness RNG code and draws are unchanged.
3. For a median, compute the existing lower median, sorted element `(n-1)/2`.
   For a ratio, draw the **same index** for both matched round arrays, then divide their
   lower medians. The statistic is a ratio of medians, not a median of per-round ratios.
4. Sort the `B` bootstrap statistics. Return empirical lower quantiles at 2.5% and 97.5%:
   indices `floor((B-1)/40)` and `floor(39*(B-1)/40)`. No endpoint interpolation,
   arithmetic-mean median, BCa correction or estimate-based widening is applied.
   The point estimate remains the original lower median (or ratio of lower medians).

The percentile-bootstrap and common-index paired-resampling concepts are described in the
[SciPy bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html).
This implementation uses its own documented integer endpoint policy and existing project
RNG; SciPy is not a dependency and identical output to its defaults is not claimed.

## Pairing and interpretation

The paired API requires exactly matching round-index sets. Missing or duplicate IDs fail;
it never silently intersects, truncates or independently resamples the arrays. Nonfinite
or underflowed ratios fail rather than producing an unusable interval.

The caller must validate provenance and compatible protocol keys before comparing logs.
Equal counters alone do not establish paired observations: separate unrelated runs each
numbering their first round 0 are not automatically paired. Interleaved variants from the
same acquisition have shared round IDs; two runs require an explicitly paired design.
The API validates IDs/rates, not kernel timing fields or log provenance.

The interval describes resampling the supplied rounds, not between-process variability.
Few rounds, autocorrelation and thermal drift limit interpretation; the existing drift
flag must still be reported. A single observation or constant data produces a degenerate
interval, not evidence of zero physical measurement uncertainty. `--quick` data remain
smoke data. The nominal 95% level is not a finite-sample coverage guarantee.

Memory is `O(n + B)`; resamples reuse scratch storage. Identical data/config produce the
same output, including `[1, 1]` for identical paired arrays and `[c, c]` for exact positive
power-of-two proportional data. No measured results or search budgets change in N4.
