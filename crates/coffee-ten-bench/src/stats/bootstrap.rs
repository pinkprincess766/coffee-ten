//! Read-only, seeded percentile bootstrap of the existing lower-median statistic.
use super::{Rng, summarize};
use crate::run::RoundSample;
use std::fmt;

/// Analysis randomness is independent of input generation and candidate selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BootstrapConfig {
    pub resamples: usize,
    pub seed: u64,
}

impl Default for BootstrapConfig {
    fn default() -> Self {
        Self {
            resamples: 10_000,
            seed: 1,
        }
    }
}

/// Nominal two-sided 95% percentile interval; endpoints are empirical lower quantiles.
/// This describes uncertainty across the supplied rounds, not across process restarts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BootstrapInterval {
    pub estimate: f64,
    pub lower: f64,
    pub upper: f64,
    pub confidence_level: f64,
    pub samples: usize,
    pub resamples: usize,
    pub seed: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BootstrapError {
    EmptyRounds,
    ZeroResamples,
    InvalidRate { round_index: usize },
    DuplicateRound { round_index: usize },
    UnmatchedRounds,
    InvalidRatio,
}

impl fmt::Display for BootstrapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyRounds => {
                write!(f, "bootstrap needs raw rounds; aggregates are insufficient")
            }
            Self::ZeroResamples => write!(f, "bootstrap resamples must be positive"),
            Self::InvalidRate { round_index } => {
                write!(
                    f,
                    "round {round_index} needs a finite positive GFLOP/s rate"
                )
            }
            Self::DuplicateRound { round_index } => write!(f, "duplicate round {round_index}"),
            Self::UnmatchedRounds => write!(f, "paired bootstrap needs identical round-index sets"),
            Self::InvalidRatio => write!(f, "median ratio must be finite and positive"),
        }
    }
}

impl std::error::Error for BootstrapError {}

fn ordered(rounds: &[RoundSample]) -> Result<Vec<&RoundSample>, BootstrapError> {
    if rounds.is_empty() {
        return Err(BootstrapError::EmptyRounds);
    }
    let mut rounds: Vec<_> = rounds.iter().collect();
    rounds.sort_unstable_by_key(|r| r.round_index);
    for (index, round) in rounds.iter().enumerate() {
        if !round.gflops.is_finite() || round.gflops <= 0.0 {
            return Err(BootstrapError::InvalidRate {
                round_index: round.round_index,
            });
        }
        if index > 0 && rounds[index - 1].round_index == round.round_index {
            return Err(BootstrapError::DuplicateRound {
                round_index: round.round_index,
            });
        }
    }
    Ok(rounds)
}

// Rejection avoids the modulo bias of a plain x % n. A private analysis RNG means no
// draws are consumed from the unchanged search/harness RNGs.
fn draw_index(rng: &mut Rng, n: usize) -> usize {
    let bound = n as u64;
    let threshold = bound.wrapping_neg() % bound;
    loop {
        let value = rng.next_u64();
        if value >= threshold {
            return (value % bound) as usize;
        }
    }
}

fn lower_median(values: &mut [f64]) -> f64 {
    let index = (values.len() - 1) / 2;
    *values.select_nth_unstable_by(index, f64::total_cmp).1
}

fn ratio(a: f64, b: f64) -> Result<f64, BootstrapError> {
    let value = a / b;
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(BootstrapError::InvalidRatio)
    }
}

fn interval(
    estimate: f64,
    samples: usize,
    config: BootstrapConfig,
    mut distribution: Vec<f64>,
) -> BootstrapInterval {
    distribution.sort_by(f64::total_cmp);
    let last = distribution.len() - 1;
    // floor((B-1)*p), p = 1/40 and 39/40. Integer arithmetic fixes endpoint policy
    // across platforms and avoids interpolation and usize multiplication overflow.
    BootstrapInterval {
        estimate,
        lower: distribution[last / 40],
        upper: distribution[(last as u128 * 39 / 40) as usize],
        confidence_level: 0.95,
        samples,
        resamples: config.resamples,
        seed: config.seed,
    }
}

/// Bootstrap the lower median of a candidate's raw GFLOP/s rounds.
/// Each replicate draws `n` rounds with replacement. Round IDs must be unique and
/// are sorted before sampling, so reordering the input does not change seeded output.
/// The input, acquisition statistics and drift flag are never changed.
pub fn bootstrap_median(
    rounds: &[RoundSample],
    config: BootstrapConfig,
) -> Result<BootstrapInterval, BootstrapError> {
    if config.resamples == 0 {
        return Err(BootstrapError::ZeroResamples);
    }
    let rounds = ordered(rounds)?;
    let rates: Vec<_> = rounds.iter().map(|r| r.gflops).collect();
    let estimate = summarize(&rates).expect("nonempty validated rounds").median;
    let mut rng = Rng::new(config.seed);
    let mut sample = vec![0.0; rates.len()];
    let mut distribution = Vec::with_capacity(config.resamples);
    for _ in 0..config.resamples {
        for rate in &mut sample {
            *rate = rates[draw_index(&mut rng, rates.len())];
        }
        distribution.push(lower_median(&mut sample));
    }
    Ok(interval(estimate, rates.len(), config, distribution))
}

/// Bootstrap `median(numerator) / median(denominator)`, drawing the same round IDs
/// for both sides of every replicate. This is not the median of per-round ratios.
/// Missing/duplicate IDs are errors; no intersection, truncation or independent draws.
/// The caller must establish that equal IDs represent genuinely paired observations
/// and validate log/protocol provenance; equal counters from unrelated runs do not do so.
pub fn bootstrap_paired_ratio(
    numerator: &[RoundSample],
    denominator: &[RoundSample],
    config: BootstrapConfig,
) -> Result<BootstrapInterval, BootstrapError> {
    if config.resamples == 0 {
        return Err(BootstrapError::ZeroResamples);
    }
    let a = ordered(numerator)?;
    let b = ordered(denominator)?;
    if a.len() != b.len()
        || a.iter()
            .zip(&b)
            .any(|(a, b)| a.round_index != b.round_index)
    {
        return Err(BootstrapError::UnmatchedRounds);
    }
    let a_rates: Vec<_> = a.iter().map(|r| r.gflops).collect();
    let b_rates: Vec<_> = b.iter().map(|r| r.gflops).collect();
    let estimate = ratio(
        summarize(&a_rates)
            .expect("nonempty validated rounds")
            .median,
        summarize(&b_rates)
            .expect("nonempty validated rounds")
            .median,
    )?;
    let mut rng = Rng::new(config.seed);
    let mut a_sample = vec![0.0; a.len()];
    let mut b_sample = vec![0.0; b.len()];
    let mut distribution = Vec::with_capacity(config.resamples);
    for _ in 0..config.resamples {
        for (a_rate, b_rate) in a_sample.iter_mut().zip(&mut b_sample) {
            let index = draw_index(&mut rng, a.len());
            *a_rate = a_rates[index];
            *b_rate = b_rates[index];
        }
        distribution.push(ratio(
            lower_median(&mut a_sample),
            lower_median(&mut b_sample),
        )?);
    }
    Ok(interval(estimate, a.len(), config, distribution))
}

/// Ratio of lower medians from independent runs. Equal round counters in separately
/// acquired files do not establish pairing: draw each side independently by default.
pub fn bootstrap_independent_ratio(
    numerator: &[RoundSample],
    denominator: &[RoundSample],
    config: BootstrapConfig,
) -> Result<BootstrapInterval, BootstrapError> {
    if config.resamples == 0 {
        return Err(BootstrapError::ZeroResamples);
    }
    let a = ordered(numerator)?;
    let b = ordered(denominator)?;
    let a_rates: Vec<_> = a.iter().map(|r| r.gflops).collect();
    let b_rates: Vec<_> = b.iter().map(|r| r.gflops).collect();
    let estimate = ratio(
        summarize(&a_rates).unwrap().median,
        summarize(&b_rates).unwrap().median,
    )?;
    let mut rng = Rng::new(config.seed);
    let mut a_sample = vec![0.0; a.len()];
    let mut b_sample = vec![0.0; b.len()];
    let mut distribution = Vec::with_capacity(config.resamples);
    for _ in 0..config.resamples {
        for value in &mut a_sample {
            *value = a_rates[draw_index(&mut rng, a.len())];
        }
        for value in &mut b_sample {
            *value = b_rates[draw_index(&mut rng, b.len())];
        }
        distribution.push(ratio(
            lower_median(&mut a_sample),
            lower_median(&mut b_sample),
        )?);
    }
    Ok(interval(estimate, a.len(), config, distribution))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rounds(rates: &[f64]) -> Vec<RoundSample> {
        rates
            .iter()
            .enumerate()
            .map(|(index, &gflops)| RoundSample {
                round_index: 10 + index * 3,
                elapsed_ns: 1,
                iterations: 1,
                gflops,
            })
            .collect()
    }

    #[test]
    fn constant_and_single_round_intervals_are_exact() {
        for rates in [vec![4.5; 30], vec![7.25]] {
            let result = bootstrap_median(&rounds(&rates), BootstrapConfig::default()).unwrap();
            assert_eq!(result.estimate.to_bits(), rates[0].to_bits());
            assert_eq!(result.lower.to_bits(), rates[0].to_bits());
            assert_eq!(result.upper.to_bits(), rates[0].to_bits());
            assert_eq!(result.confidence_level, 0.95);
            assert_eq!(result.samples, rates.len());
        }
    }

    #[test]
    fn equal_and_proportional_pairs_cannot_gain_spurious_ratio_uncertainty() {
        // A common per-round slowdown cancels when the same index is drawn on both sides.
        // Independent resampling would create a wide interval and fail this test.
        let a = rounds(&[1.0, 128.0, 2.0, 64.0, 4.0, 32.0, 8.0, 16.0]);
        for factor in [1.0, 2.0] {
            let mut b = a.clone();
            for round in &mut b {
                round.gflops *= factor;
            }
            b.reverse();
            let ci = bootstrap_paired_ratio(&b, &a, BootstrapConfig::default()).unwrap();
            assert_eq!((ci.estimate, ci.lower, ci.upper), (factor, factor, factor));
        }
    }

    #[test]
    fn ratio_is_of_medians_and_even_samples_keep_the_existing_lower_median() {
        let a = rounds(&[1.0, 4.0, 5.0]);
        let b = rounds(&[2.0, 3.0, 100.0]);
        let ci = bootstrap_paired_ratio(&a, &b, BootstrapConfig::default()).unwrap();
        assert_eq!(ci.estimate.to_bits(), (4.0f64 / 3.0).to_bits());
        assert_ne!(ci.estimate, 0.5); // median of the per-round ratios
        let ci = bootstrap_median(&rounds(&[9.0, 7.0]), BootstrapConfig::default()).unwrap();
        assert_eq!(ci.estimate, 7.0);
        assert_eq!((ci.lower, ci.upper), (7.0, 9.0));
    }

    #[test]
    fn seeded_output_is_reproducible_and_input_order_does_not_change_pairing() {
        let mut a = rounds(&[1.0, 9.0, 3.0, 7.0, 4.0]);
        let mut b = rounds(&[2.0, 5.0, 8.0, 6.0, 10.0]);
        let saved = (a.clone(), b.clone());
        let config = BootstrapConfig {
            resamples: 257,
            seed: 42,
        };
        let median = bootstrap_median(&a, config).unwrap();
        let ratio = bootstrap_paired_ratio(&a, &b, config).unwrap();
        // Independent Python stdlib median_low reference for these 257 draws/seed 42.
        // This also fixes the seeded endpoint contract across debug/release and CI hosts.
        assert_eq!((median.lower, median.upper), (1.0, 9.0));
        assert_eq!(ratio.lower.to_bits(), 0.375f64.to_bits());
        assert_eq!(ratio.upper.to_bits(), 1.8f64.to_bits());
        assert_eq!((ratio.resamples, ratio.seed, ratio.samples), (257, 42, 5));
        assert_eq!(median, bootstrap_median(&a, config).unwrap());
        assert_eq!(ratio, bootstrap_paired_ratio(&a, &b, config).unwrap());
        assert_eq!((a.clone(), b.clone()), saved);
        a.reverse();
        b.rotate_left(2);
        assert_eq!(median, bootstrap_median(&a, config).unwrap());
        assert_eq!(ratio, bootstrap_paired_ratio(&a, &b, config).unwrap());
    }

    #[test]
    fn rejects_invalid_raw_data_and_unmatched_pairs() {
        let config = BootstrapConfig::default();
        let valid = rounds(&[1.0, 2.0, 3.0]);
        assert_eq!(
            bootstrap_median(&[], config),
            Err(BootstrapError::EmptyRounds)
        );
        assert_eq!(
            bootstrap_median(
                &valid,
                BootstrapConfig {
                    resamples: 0,
                    seed: 1
                }
            ),
            Err(BootstrapError::ZeroResamples)
        );
        for value in [0.0, -0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut invalid = valid.clone();
            invalid[1].gflops = value;
            assert_eq!(
                bootstrap_median(&invalid, config),
                Err(BootstrapError::InvalidRate { round_index: 13 })
            );
            assert!(bootstrap_paired_ratio(&valid, &invalid, config).is_err());
        }
        let mut duplicate = valid.clone();
        duplicate[1].round_index = duplicate[0].round_index;
        assert_eq!(
            bootstrap_median(&duplicate, config),
            Err(BootstrapError::DuplicateRound { round_index: 10 })
        );
        assert_eq!(
            bootstrap_paired_ratio(&valid, &valid[..2], config),
            Err(BootstrapError::UnmatchedRounds)
        );
        let mut different = valid.clone();
        different[1].round_index += 1;
        assert_eq!(
            bootstrap_paired_ratio(&valid, &different, config),
            Err(BootstrapError::UnmatchedRounds)
        );
        for (a, b) in [(f64::MAX, f64::MIN_POSITIVE), (f64::MIN_POSITIVE, f64::MAX)] {
            assert_eq!(
                bootstrap_paired_ratio(&rounds(&[a]), &rounds(&[b]), config),
                Err(BootstrapError::InvalidRatio)
            );
        }
    }

    #[test]
    fn percentile_endpoints_use_documented_integer_ranks() {
        let distribution: Vec<_> = (0..=100).map(f64::from).collect();
        let ci = interval(
            50.0,
            5,
            BootstrapConfig {
                resamples: 101,
                seed: 1,
            },
            distribution,
        );
        assert_eq!((ci.lower, ci.upper), (2.0, 97.0));
    }

    #[test]
    fn percentile_ranks_at_small_and_boundary_resample_counts() {
        // Ranks worked out by hand from floor((B-1)/40) and floor((B-1)*39/40):
        // B = 1, 2: both 0. B = 39: 38*39 = 1482, /40 = 37.05. B = 40: 1521/40 = 38.025.
        // B = 41 is the first B whose lower rank leaves 0. B = 10_000 is the default.
        for (resamples, lower, upper) in [
            (1, 0, 0),
            (2, 0, 0),
            (39, 0, 37),
            (40, 0, 38),
            (41, 1, 39),
            (10_000, 249, 9_749),
        ] {
            // Reversed, so the ranks are taken after sorting, not from input order.
            let distribution: Vec<_> = (0..resamples).rev().map(|r| r as f64).collect();
            let config = BootstrapConfig { resamples, seed: 1 };
            let ci = interval(0.5, 3, config, distribution);
            assert_eq!(
                (ci.lower, ci.upper),
                (lower as f64, upper as f64),
                "B = {resamples}"
            );
            assert!(ci.lower <= ci.upper);
        }
    }

    #[test]
    fn one_or_two_resamples_give_an_ordered_interval_from_drawn_values() {
        let input = rounds(&[1.0, 2.0, 3.0, 5.0, 8.0]);
        for resamples in [1, 2] {
            let config = BootstrapConfig { resamples, seed: 7 };
            for ci in [
                bootstrap_median(&input, config).unwrap(),
                bootstrap_paired_ratio(&input, &input, config).unwrap(),
                bootstrap_independent_ratio(&input, &input, config).unwrap(),
            ] {
                // For B <= 2 both endpoints are rank 0, the smallest replicate. (The lower end
                // stays rank 0 up to B = 40; the upper end moves from B = 3 on.)
                assert_eq!(ci.lower, ci.upper, "B = {resamples}");
                assert_eq!(ci.resamples, resamples);
            }
            let ci = bootstrap_median(&input, config).unwrap();
            assert!([1.0, 2.0, 3.0, 5.0, 8.0].contains(&ci.lower));
        }
    }

    #[test]
    fn independent_runs_do_not_cancel_shared_counter_variation() {
        let a = rounds(&[1.0, 2.0, 16.0, 32.0, 128.0]);
        let config = BootstrapConfig {
            resamples: 1000,
            seed: 42,
        };
        let ci = bootstrap_independent_ratio(&a, &a, config).unwrap();
        assert_eq!(ci.estimate, 1.0);
        assert!(ci.lower < 1.0 && ci.upper > 1.0);
        assert_eq!(ci, bootstrap_independent_ratio(&a, &a, config).unwrap());
        assert_eq!(bootstrap_paired_ratio(&a, &a, config).unwrap().lower, 1.0);
        let exact =
            bootstrap_independent_ratio(&rounds(&[4.0; 3]), &rounds(&[2.0; 5]), config).unwrap();
        assert_eq!((exact.estimate, exact.lower, exact.upper), (2.0, 2.0, 2.0));
        assert!(bootstrap_independent_ratio(&[], &a, config).is_err());
        assert!(bootstrap_independent_ratio(&a, &rounds(&[0.0]), config).is_err());
    }
}
