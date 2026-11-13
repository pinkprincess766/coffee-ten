//! Degenerate raw rounds in saved logs: none, one, zero or non-finite timings and rates.
//! The report must either compute exact statistics or refuse with the round named.
use coffee_ten_bench::{report, stats::BootstrapConfig};

const LOG: &str = include_str!("fixtures/report.jsonl");
const CONFIG: BootstrapConfig = BootstrapConfig {
    resamples: 50,
    seed: 9,
};
const ROUNDS: &str = "\"rounds\":[{\"round_index\":0,\"elapsed_ns\":16000000,\"iterations\":1000000,\"gflops\":1.0},{\"round_index\":1,\"elapsed_ns\":8000000,\"iterations\":1000000,\"gflops\":2.0},{\"round_index\":2,\"elapsed_ns\":4000000,\"iterations\":1000000,\"gflops\":4.0}]";

/// The fixture's naive row (line 2) with its raw rounds and other substrings replaced.
fn naive_row(rounds: &str, replacements: &[(&str, &str)]) -> String {
    let line = LOG.lines().nth(1).unwrap();
    assert!(line.contains(ROUNDS));
    let mut row = line.replace(ROUNDS, &format!("\"rounds\":[{rounds}]"));
    for (before, after) in replacements {
        assert!(row.contains(before), "{before}");
        row = row.replace(before, after);
    }
    format!("{row}\n")
}

fn error(text: &str) -> String {
    report::markdown(text, 1, CONFIG).unwrap_err().to_string()
}

#[test]
fn a_single_round_gives_exact_statistics_and_a_degenerate_interval() {
    let one = naive_row(
        "{\"round_index\":0,\"elapsed_ns\":8000000,\"iterations\":1000000,\"gflops\":2.0}",
        &[
            ("\"samples\":3", "\"samples\":1"),
            ("\"p25\":1.0", "\"p25\":2.0"),
            ("\"min\":1.0", "\"min\":2.0"),
            ("\"measurement_ns\":28000000", "\"measurement_ns\":8000000"),
        ],
    );
    let out = report::markdown(&one, 1, CONFIG).unwrap();
    assert!(
        out.contains("| 1 | naive | (naive) | 2.000000 | [2.000000, 2.000000] | 1 |"),
        "{out}"
    );
    assert!(out.contains("Records: 1. Measurement: 8000000 ns."));
    // Comparing a one-round run with itself is an exact ratio of 1 in both modes.
    for paired in [false, true] {
        let compared = report::compare(&one, &one, CONFIG, paired).unwrap();
        assert!(compared.contains("Median ratio A/B: 1.000000; 95% CI [1.000000, 1.000000]"));
    }
    // Aggregates must still match the single round bit for bit.
    let wrong = one.replace("\"p25\":2.0", "\"p25\":1.0");
    assert!(error(&wrong).contains("p25 differs from raw rounds"));
}

#[test]
fn empty_or_miscounted_rounds_are_refused_with_the_counts() {
    let empty = naive_row("", &[("\"samples\":3", "\"samples\":0")]);
    assert_eq!(
        error(&empty),
        "report: line 1: wrong raw round count: 0 rounds, samples 0, protocol samples 0"
    );
    // Two rounds while both the row and the protocol declare three.
    let short = naive_row(
        "{\"round_index\":0,\"elapsed_ns\":16000000,\"iterations\":1000000,\"gflops\":1.0},{\"round_index\":1,\"elapsed_ns\":8000000,\"iterations\":1000000,\"gflops\":2.0}",
        &[],
    );
    assert_eq!(
        error(&short),
        "report: line 1: wrong raw round count: 2 rounds, samples 3, protocol samples 3"
    );
}

#[test]
fn zero_timings_and_non_finite_rates_name_the_round() {
    let cases = [
        // (round 1 replacement, expected message)
        (
            "{\"round_index\":1,\"elapsed_ns\":0,\"iterations\":1000000,\"gflops\":2.0}",
            "round 1 needs positive elapsed_ns and iterations",
        ),
        (
            "{\"round_index\":1,\"elapsed_ns\":8000000,\"iterations\":0,\"gflops\":2.0}",
            "round 1 needs positive elapsed_ns and iterations",
        ),
        // NaN and infinities are written as null by the harness; neither is a rate.
        (
            "{\"round_index\":1,\"elapsed_ns\":8000000,\"iterations\":1000000,\"gflops\":null}",
            "round 1 gflops: expected finite number",
        ),
        (
            "{\"round_index\":1,\"elapsed_ns\":8000000,\"iterations\":1000000,\"gflops\":1e999}",
            "round 1 gflops: expected finite number",
        ),
        (
            "{\"round_index\":1,\"elapsed_ns\":-8000000,\"iterations\":1000000,\"gflops\":2.0}",
            "round 1 elapsed_ns: expected unsigned integer",
        ),
        (
            "{\"round_index\":1,\"elapsed_ns\":8000000,\"gflops\":2.0}",
            "round 1 iterations: missing iterations",
        ),
        (
            "{\"round_index\":2,\"elapsed_ns\":8000000,\"iterations\":1000000,\"gflops\":2.0}",
            "invalid raw round order: round 1 has round_index 2",
        ),
        // Zero and negative rates are finite but cannot come from a positive timing.
        (
            "{\"round_index\":1,\"elapsed_ns\":8000000,\"iterations\":1000000,\"gflops\":0.0}",
            "raw rate does not match timing",
        ),
        (
            "{\"round_index\":1,\"elapsed_ns\":8000000,\"iterations\":1000000,\"gflops\":-2.0}",
            "raw rate does not match timing",
        ),
    ];
    let good = "{\"round_index\":1,\"elapsed_ns\":8000000,\"iterations\":1000000,\"gflops\":2.0}";
    assert!(ROUNDS.contains(good));
    for (bad, expected) in cases {
        let text = LOG.replacen(good, bad, 1);
        assert_ne!(text, LOG);
        let err = error(&text);
        assert!(err.contains(expected), "{bad}: {err}");
        assert!(err.starts_with("report: line 2: "), "{err}");
    }
}

#[test]
fn rates_at_the_extremes_of_timing_stay_exact() {
    // One iteration over u64::MAX nanoseconds is a tiny but finite, positive rate; the
    // report recomputes it from the timing rather than trusting the stored value.
    let elapsed = u64::MAX;
    let rate = 2.0 * 8.0 / std::time::Duration::from_nanos(elapsed).as_secs_f64() / 1e9;
    assert!(rate.is_finite() && rate > 0.0);
    let row = naive_row(
        &format!(
            "{{\"round_index\":0,\"elapsed_ns\":{elapsed},\"iterations\":1,\"gflops\":{rate:?}}}"
        ),
        &[
            ("\"samples\":3", "\"samples\":1"),
            (
                "\"median\":2.0,\"p25\":1.0,\"p75\":2.0,\"min\":1.0",
                &format!("\"median\":{rate:?},\"p25\":{rate:?},\"p75\":{rate:?},\"min\":{rate:?}"),
            ),
            (
                "\"measurement_ns\":28000000",
                &format!("\"measurement_ns\":{elapsed}"),
            ),
        ],
    );
    let out = report::markdown(&row, 1, CONFIG).unwrap();
    assert!(
        out.contains(&format!("Measurement: {elapsed} ns.")),
        "{out}"
    );
}
