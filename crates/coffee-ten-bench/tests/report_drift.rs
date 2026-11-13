//! The Drift column of `report` and `report --compare`.
use coffee_ten_bench::{report, stats::BootstrapConfig};

const LOG: &str = include_str!("fixtures/report.jsonl");

/// The fixture's naive row rewritten to `rates.len()` rounds of the given rates (2.0 or 4.0
/// GFLOP/s, which the fixture's 8 ms and 4 ms rounds produce bit-exactly at n = 2).
fn naive_row_with_rates(rates: &[f64], drift: bool) -> String {
    let line = LOG.lines().nth(1).unwrap();
    let mut rounds = Vec::new();
    let mut measurement_ns = 0u64;
    for (index, rate) in rates.iter().enumerate() {
        let elapsed_ns = if *rate == 2.0 { 8_000_000 } else { 4_000_000 };
        assert!(*rate == 2.0 || *rate == 4.0);
        measurement_ns += elapsed_ns;
        rounds.push(format!(
            "{{\"round_index\":{index},\"elapsed_ns\":{elapsed_ns},\"iterations\":1000000,\"gflops\":{rate:?}}}"
        ));
    }
    let mut sorted = rates.to_vec();
    sorted.sort_by(f64::total_cmp);
    let last = sorted.len() - 1;
    let start = line.find("\"rounds\":[").unwrap();
    let end = start + line[start..].find(']').unwrap() + 1;
    let line = format!(
        "{}\"rounds\":[{}]{}",
        &line[..start],
        rounds.join(","),
        &line[end..]
    );
    let row = line
        .replace("\"samples\":3", &format!("\"samples\":{}", rates.len()))
        .replace(
            "\"median\":2.0,\"p25\":1.0,\"p75\":2.0,\"min\":1.0,\"drift\":false",
            &format!(
                "\"median\":{:?},\"p25\":{:?},\"p75\":{:?},\"min\":{:?},\"drift\":{drift}",
                sorted[last / 2],
                sorted[last / 4],
                sorted[3 * last / 4],
                sorted[0]
            ),
        )
        .replace(
            "\"measurement_ns\":28000000",
            &format!("\"measurement_ns\":{measurement_ns}"),
        );
    assert_ne!(row, line);
    format!("{row}\n")
}

#[test]
fn drift_column_says_not_evaluated_when_rounds_are_too_few() {
    let config = BootstrapConfig {
        resamples: 10,
        seed: 1,
    };
    // Three rounds: the drift check never ran, so `false` would be a claim nobody tested.
    let three = report::markdown(LOG, 2, config).unwrap();
    assert_eq!(three.matches("| 3 | not evaluated |").count(), 2);
    assert!(!three.contains("| false |"));
    let compared = report::compare(LOG, LOG, config, false).unwrap();
    assert_eq!(compared.matches("| 3 | not evaluated |").count(), 2);
    assert!(!compared.contains("| false |"));

    // Nine rounds, one short of two disjoint windows of five: still not evaluated.
    let nine = naive_row_with_rates(&[2.0; 9], false);
    assert!(
        report::markdown(&nine, 1, config)
            .unwrap()
            .contains("| 9 | not evaluated |")
    );

    // Ten rounds: the check ran, and both outcomes are printed as such.
    let steady = naive_row_with_rates(&[2.0; 10], false);
    let mut rising = vec![2.0; 5];
    rising.extend([4.0; 5]);
    let rising = naive_row_with_rates(&rising, true);
    assert!(
        report::markdown(&steady, 1, config)
            .unwrap()
            .contains("| 10 | false |")
    );
    assert!(
        report::markdown(&rising, 1, config)
            .unwrap()
            .contains("| 10 | true |")
    );
    let compared = report::compare(&rising, &steady, config, false).unwrap();
    assert!(compared.contains("| A | (naive) | 2.000000 | 10 | true |"));
    assert!(compared.contains("| B | (naive) | 2.000000 | 10 | false |"));
}
