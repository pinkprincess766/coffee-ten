//! Percentiles, the drift flag and the shuffle RNG.

/// Rounds at each end of a run whose medians the drift check compares.
pub const DRIFT_ROUNDS: usize = 5;
/// A variant whose median changes between the first and the last `DRIFT_ROUNDS` rounds by
/// more than this fraction is flagged (thermal throttling or another load).
pub const DRIFT_LIMIT: f64 = 0.05;
/// With fewer rounds the two windows would overlap, so the check is skipped.
pub const MIN_ROUNDS_FOR_DRIFT: usize = 2 * DRIFT_ROUNDS;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stats {
    pub median: f64,
    pub p25: f64,
    pub p75: f64,
    pub min: f64,
}

/// Sorted `r`, integer indices, no interpolation (as `bench/peak/peak.c`):
/// min = r[0], p25 = r[(n-1)/4], median = r[(n-1)/2], p75 = r[3*(n-1)/4].
/// `None` for an empty slice.
pub fn summarize(rates: &[f64]) -> Option<Stats> {
    if rates.is_empty() {
        return None;
    }
    let mut sorted = rates.to_vec();
    sorted.sort_by(f64::total_cmp);
    let last = sorted.len() - 1;
    Some(Stats {
        median: sorted[last / 2],
        p25: sorted[last / 4],
        p75: sorted[3 * last / 4],
        min: sorted[0],
    })
}

/// `rates` in round order. True if the median of the first `DRIFT_ROUNDS` and the median of
/// the last `DRIFT_ROUNDS` differ by more than `DRIFT_LIMIT`, relative to the first.
/// Always false with fewer than `MIN_ROUNDS_FOR_DRIFT` rounds.
pub fn drifted(rates: &[f64]) -> bool {
    if rates.len() < MIN_ROUNDS_FOR_DRIFT {
        return false;
    }
    let first = summarize(&rates[..DRIFT_ROUNDS]);
    let last = summarize(&rates[rates.len() - DRIFT_ROUNDS..]);
    let (Some(first), Some(last)) = (first, last) else {
        return false;
    };
    (last.median - first.median).abs() / first.median > DRIFT_LIMIT
}

/// xorshift64* (Vigna 2016). Not for anything but shuffling and test data.
#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Any seed works; 0 is replaced because the state of xorshift must not be 0.
    pub fn new(seed: u64) -> Rng {
        let state = if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            // An odd multiplier is a bijection, so a nonzero seed stays nonzero and small
            // seeds do not start with a nearly empty state.
            seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)
        };
        Rng { state }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in [-1, 1), 24 random bits.
    pub fn next_signed_unit_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32;
        bits as f32 / (1u32 << 23) as f32 - 1.0
    }

    /// Fisher-Yates. `% (i + 1)` is biased by at most 2^-50 for the lengths used here.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            items.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentiles_of_30_rates() {
        // 30..=1 reversed and interleaved, so the sort has work to do.
        let mut rates: Vec<f64> = (1..=30).map(f64::from).collect();
        rates.reverse();
        rates.swap(3, 20);
        let s = summarize(&rates).unwrap();
        // n = 30: indices 0, 29/4 = 7, 29/2 = 14, 87/4 = 21 of the sorted 1..=30.
        assert_eq!(s.min, 1.0);
        assert_eq!(s.p25, 8.0);
        assert_eq!(s.median, 15.0);
        assert_eq!(s.p75, 22.0);
    }

    #[test]
    fn percentiles_of_3_rates() {
        let s = summarize(&[30.0, 10.0, 20.0]).unwrap();
        // n = 3: indices 0, 2/4 = 0, 2/2 = 1, 6/4 = 1.
        assert_eq!(s.min, 10.0);
        assert_eq!(s.p25, 10.0);
        assert_eq!(s.median, 20.0);
        assert_eq!(s.p75, 20.0);
        assert_eq!(summarize(&[]), None);
    }

    /// 10 rounds: the first 5 have median 100, the last 5 have median `last`.
    fn rounds_with_last_median(last: f64) -> Vec<f64> {
        let mut rates = vec![100.0; DRIFT_ROUNDS];
        rates.extend([last; DRIFT_ROUNDS]);
        rates
    }

    #[test]
    fn drift_at_exactly_the_limit_is_not_flagged() {
        // 5 / 100 is the double nearest to 0.05, the same as the literal DRIFT_LIMIT.
        assert_eq!(DRIFT_LIMIT, 0.05);
        assert!(!drifted(&rounds_with_last_median(105.0)));
        assert!(!drifted(&rounds_with_last_median(95.0)));
    }

    #[test]
    fn drift_above_the_limit_is_flagged_up_and_down() {
        assert!(drifted(&rounds_with_last_median(105.01)));
        assert!(drifted(&rounds_with_last_median(94.99)));
        assert!(!drifted(&rounds_with_last_median(100.0)));
    }

    #[test]
    fn drift_uses_medians_not_single_rounds() {
        // One slow round in the last window does not move its median.
        let mut rates = rounds_with_last_median(100.0);
        rates[9] = 1.0;
        assert!(!drifted(&rates));
    }

    #[test]
    fn drift_needs_two_disjoint_windows() {
        let mut nine = vec![100.0; DRIFT_ROUNDS];
        nine.extend([1.0; DRIFT_ROUNDS - 1]);
        assert_eq!(nine.len(), MIN_ROUNDS_FOR_DRIFT - 1);
        assert!(!drifted(&nine));
        nine.push(1.0);
        assert!(drifted(&nine));
    }

    fn shuffled_orders(seed: u64) -> Vec<Vec<usize>> {
        let mut rng = Rng::new(seed);
        (0..10)
            .map(|_| {
                let mut order: Vec<usize> = (0..8).collect();
                rng.shuffle(&mut order);
                order
            })
            .collect()
    }

    #[test]
    fn shuffle_is_a_permutation() {
        for order in shuffled_orders(1) {
            let mut sorted = order.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, (0..8).collect::<Vec<_>>());
        }
    }

    #[test]
    fn shuffle_is_deterministic_per_seed_and_differs_between_seeds() {
        assert_eq!(shuffled_orders(1), shuffled_orders(1));
        assert_ne!(shuffled_orders(1), shuffled_orders(2));
        // The order changes from round to round.
        let orders = shuffled_orders(1);
        assert!(orders.iter().any(|order| *order != orders[0]));
    }

    #[test]
    fn rng_matches_the_xorshift64_star_definition() {
        // Reference computed by hand from the definition for the state after seeding.
        let mut rng = Rng::new(7);
        let mut x = 7u64.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        for _ in 0..3 {
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            assert_eq!(rng.next_u64(), x.wrapping_mul(0x2545_F491_4F6C_DD1D));
        }
        // Seed 0 does not stick at state 0.
        assert_ne!(Rng::new(0).next_u64(), 0);
    }

    #[test]
    fn signed_unit_stays_in_range() {
        let mut rng = Rng::new(3);
        for _ in 0..10_000 {
            let x = rng.next_signed_unit_f32();
            assert!((-1.0..1.0).contains(&x), "{x}");
        }
    }
}
