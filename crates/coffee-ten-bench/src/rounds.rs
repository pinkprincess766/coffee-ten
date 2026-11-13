//! One validator for saved raw rounds, shared by `report` and `replay` so their rules and
//! messages cannot drift apart. Read-only: nothing here measures or changes stored values.
use crate::{
    json::Value,
    run::RoundSample,
    stats::{Stats, drifted, summarize},
};
use std::time::Duration;

/// Raw rounds that passed every check, with the statistics recomputed from them.
#[derive(Debug)]
pub(crate) struct Rounds {
    pub rounds: Vec<RoundSample>,
    pub rates: Vec<f64>,
    pub stats: Stats,
    pub measurement_ns: u128,
}

/// One field of raw round `index`, with errors that name both.
fn field<T>(
    round: &Value,
    index: usize,
    name: &str,
    read: impl FnOnce(&Value) -> Result<T, String>,
) -> Result<T, String> {
    round
        .get(name)
        .and_then(read)
        .map_err(|e| format!("round {index} {name}: {e}"))
}

/// Checks acquisition order, positive timings, the protocol minimum duration and that each
/// stored rate is bit-for-bit the rate its timing gives for an `n`-cubed matmul. The caller
/// checks the round count against its own protocol first; `raw` must not be empty.
pub(crate) fn validate(raw: &[Value], n: usize, min_sample_ms: u64) -> Result<Rounds, String> {
    let minimum_ns = u128::from(min_sample_ms) * 1_000_000;
    let mut rounds = Vec::with_capacity(raw.len());
    let mut measurement_ns = 0u128;
    for (index, value) in raw.iter().enumerate() {
        let round = RoundSample {
            round_index: field(value, index, "round_index", Value::uint)?,
            elapsed_ns: field(value, index, "elapsed_ns", Value::uint)?,
            iterations: field(value, index, "iterations", Value::uint)?,
            gflops: field(value, index, "gflops", Value::number)?,
        };
        if round.round_index != index {
            return Err(format!(
                "invalid raw round order: round {index} has round_index {}",
                round.round_index
            ));
        }
        if round.elapsed_ns == 0 || round.iterations == 0 {
            return Err(format!(
                "invalid raw round timing: round {index} needs positive elapsed_ns and iterations"
            ));
        }
        if u128::from(round.elapsed_ns) < minimum_ns {
            return Err(format!(
                "raw round is below the protocol minimum duration: round {index} took {} ns, minimum {min_sample_ms} ms",
                round.elapsed_ns
            ));
        }
        // Same expression and evaluation order as the harness (`run::time_sample`).
        let flop_per_call = 2.0 * (n as f64).powi(3);
        let expected = flop_per_call * round.iterations as f64
            / Duration::from_nanos(round.elapsed_ns).as_secs_f64()
            / 1e9;
        if round.gflops <= 0.0 || round.gflops.to_bits() != expected.to_bits() {
            return Err(format!(
                "raw rate does not match timing: round {index} stores {:?}, timing gives {expected:?}",
                round.gflops
            ));
        }
        measurement_ns = measurement_ns
            .checked_add(u128::from(round.elapsed_ns))
            .ok_or("measurement cost overflow")?;
        rounds.push(round);
    }
    let rates: Vec<_> = rounds.iter().map(|r| r.gflops).collect();
    let stats = summarize(&rates).ok_or("empty rounds")?;
    Ok(Rounds {
        rounds,
        rates,
        stats,
        measurement_ns,
    })
}

/// The row's stored aggregates, cost and drift flag must equal those recomputed from its rounds.
pub(crate) fn check_aggregates(row: &Value, rounds: &Rounds) -> Result<(), String> {
    for (name, expected) in [
        ("median", rounds.stats.median),
        ("p25", rounds.stats.p25),
        ("p75", rounds.stats.p75),
        ("min", rounds.stats.min),
    ] {
        if row.get(name)?.number()?.to_bits() != expected.to_bits() {
            return Err(format!("{name} differs from raw rounds"));
        }
    }
    let stored_ns = row.get("measurement_ns")?.uint::<u128>()?;
    if stored_ns != rounds.measurement_ns {
        return Err(format!(
            "inconsistent measurement cost: measurement_ns {stored_ns}, raw rounds sum to {}",
            rounds.measurement_ns
        ));
    }
    if row.get("drift")?.boolean()? != drifted(&rounds.rates) {
        return Err("inconsistent drift: stored flag differs from raw rounds".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json;

    /// n = 2: 16 FLOP per call, so 1e6 calls in 8 ms is exactly 2 GFLOP/s.
    fn raw(rounds: &str) -> Vec<Value> {
        json::parse(&format!("[{rounds}]"))
            .unwrap()
            .array()
            .unwrap()
            .to_vec()
    }
    const GOOD: &str = "{\"round_index\":0,\"elapsed_ns\":8000000,\"iterations\":1000000,\"gflops\":2.0},{\"round_index\":1,\"elapsed_ns\":4000000,\"iterations\":1000000,\"gflops\":4.0}";

    #[test]
    fn valid_rounds_keep_order_rates_and_cost() {
        let checked = validate(&raw(GOOD), 2, 4).unwrap();
        assert_eq!(checked.rates, [2.0, 4.0]);
        assert_eq!(checked.measurement_ns, 12_000_000);
        assert_eq!(checked.stats.median, 2.0);
        assert_eq!(checked.rounds[1].round_index, 1);
    }

    #[test]
    fn each_rule_has_one_message_for_report_and_replay() {
        for (rounds, n, min_ms, expected) in [
            (
                GOOD.replace("\"round_index\":1", "\"round_index\":2"),
                2,
                1,
                "invalid raw round order: round 1 has round_index 2",
            ),
            (
                GOOD.replace("\"elapsed_ns\":4000000", "\"elapsed_ns\":0"),
                2,
                0,
                "invalid raw round timing: round 1 needs positive elapsed_ns and iterations",
            ),
            (
                GOOD.to_string(),
                2,
                5,
                "raw round is below the protocol minimum duration: round 1 took 4000000 ns, minimum 5 ms",
            ),
            (
                GOOD.to_string(),
                3,
                1,
                "raw rate does not match timing: round 0 stores 2.0, timing gives 6.75",
            ),
            (
                GOOD.replace("\"gflops\":4.0", "\"gflops\":null"),
                2,
                1,
                "round 1 gflops: expected finite number",
            ),
            (
                GOOD.replace(",\"iterations\":1000000,\"gflops\":4.0", ",\"gflops\":4.0"),
                2,
                1,
                "round 1 iterations: missing iterations",
            ),
        ] {
            assert_eq!(validate(&raw(&rounds), n, min_ms).unwrap_err(), expected);
        }
    }

    #[test]
    fn aggregates_cost_and_drift_are_checked_in_that_order() {
        let checked = validate(&raw(GOOD), 2, 1).unwrap();
        let row = |median: &str, ns: u64, drift: bool| {
            json::parse(&format!(
                "{{\"median\":{median},\"p25\":2.0,\"p75\":2.0,\"min\":2.0,\"measurement_ns\":{ns},\"drift\":{drift}}}"
            ))
            .unwrap()
        };
        assert_eq!(
            check_aggregates(&row("2.0", 12_000_000, false), &checked),
            Ok(())
        );
        assert_eq!(
            check_aggregates(&row("4.0", 1, true), &checked).unwrap_err(),
            "median differs from raw rounds"
        );
        assert_eq!(
            check_aggregates(&row("2.0", 1, true), &checked).unwrap_err(),
            "inconsistent measurement cost: measurement_ns 1, raw rounds sum to 12000000"
        );
        assert_eq!(
            check_aggregates(&row("2.0", 12_000_000, true), &checked).unwrap_err(),
            "inconsistent drift: stored flag differs from raw rounds"
        );
    }
}
