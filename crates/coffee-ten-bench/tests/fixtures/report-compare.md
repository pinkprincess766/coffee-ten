# Run comparison

A fingerprint: `dba5227762af97b9`. B fingerprint: `cd0e590f0c28fdc1`.

**Quick smoke data; not a performance measurement.**

Ratio: median(best A) / median(best B). 95% percentile bootstrap, 257 resamples, seed 42, independent resampling.

| Run | Best schedule | Median GFLOP/s | Rounds | Drift |
| --- | --- | ---: | ---: | --- |
| A | vectorize(1, j) | 8.000000 | 3 | not evaluated |
| B | vectorize(1, j) | 4.000000 | 3 | not evaluated |

Median ratio A/B: 2.000000; 95% CI [0.500000, 8.000000].

Intervals are conditional on the selected winners and supplied rounds; they do not adjust for winner selection or independent process restarts.
