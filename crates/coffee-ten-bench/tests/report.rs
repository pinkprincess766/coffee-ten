use coffee_ten_bench::{report, stats::BootstrapConfig};

const LOG: &str = include_str!("fixtures/report.jsonl");

#[test]
fn incomplete_or_malformed_protocols_cannot_establish_a_comparison() {
    let config = BootstrapConfig {
        resamples: 3,
        seed: 1,
    };
    for (field, value) in [
        ("source_hash", "\"ffeeddccbbaa9988\""),
        ("space_hash", "\"0011223344556677\""),
        ("arch", "\"aarch64\""),
        ("target_os", "\"macos\""),
        ("rustc", "\"fixture rustc\""),
        ("rustflags", "\"\""),
        ("harness_profile", "\"release\""),
        ("warmup_ms", "5"),
        ("candidate_warmup_ms", "5"),
    ] {
        let token = format!("\"{field}\":{value}");
        for bad in [
            LOG.replace(&format!("{token},"), "")
                .replace(&format!(",{token}"), ""),
            LOG.replace(&token, &format!("\"{field}\":null")),
        ] {
            assert_ne!(bad, LOG, "fixture must contain {field}");
            assert!(
                report::markdown(&bad, 2, config).is_err(),
                "accepted {field}"
            );
            assert!(
                report::compare(&bad, &bad, config, false).is_err(),
                "compared {field}"
            );
        }
    }
    let v2 = LOG.replace(
        "\"space\":\"v1\"",
        "\"space\":\"v2\",\"cache\":null,\"tile_factors\":[2,4]",
    );
    assert!(report::markdown(&v2, 2, config).is_ok());
    for bad in [
        v2.replace("\"cache\":null,", ""),
        v2.replace("\"tile_factors\":[2,4],", ""),
        v2.replace("[2,4]", "[4,2]"),
    ] {
        assert_ne!(bad, v2);
        assert!(report::compare(&bad, &bad, config, false).is_err());
    }
}

#[test]
fn rejects_rounds_shorter_than_the_declared_minimum() {
    // The shortest fixture round is exactly 2 ms: the lower bound is inclusive.
    let boundary = LOG.replace("\"min_sample_ms\":1", "\"min_sample_ms\":2");
    assert!(report::markdown(&boundary, 2, BootstrapConfig::default()).is_ok());
    for minimum in [3, u64::MAX] {
        let too_short = LOG.replace(
            "\"min_sample_ms\":1",
            &format!("\"min_sample_ms\":{minimum}"),
        );
        assert_ne!(too_short, LOG);
        assert!(report::markdown(&too_short, 2, BootstrapConfig::default()).is_err());
    }
}

#[test]
fn saved_log_matches_markdown_golden_and_binary_is_read_only() {
    let config = BootstrapConfig {
        resamples: 257,
        seed: 42,
    };
    let expected = include_str!("fixtures/report.md");
    assert_eq!(report::markdown(LOG, 2, config).unwrap(), expected);
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/report.jsonl");
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_coffee-ten-bench"))
        .args([
            "report",
            path,
            "--top",
            "2",
            "--resamples",
            "257",
            "--seed",
            "42",
        ])
        .output()
        .unwrap();
    assert!(output.status.success(), "{:?}", output);
    assert_eq!(String::from_utf8(output.stdout).unwrap(), expected);
    assert_eq!(std::fs::read_to_string(path).unwrap(), LOG);
}

#[test]
fn rejects_corrupt_raw_data_costs_and_unsupported_records() {
    let config = BootstrapConfig::default();
    for bad in [
        "".into(),
        "{\"kernel\":\"matmul\"}".into(),
        "{\"kind\":\"replay_curve\"}".into(),
        LOG.replace("\"round_index\":1", "\"round_index\":0"),
        LOG.replace("\"measurement_ns\":28000000", "\"measurement_ns\":28000001"),
        LOG.replace("\"verified\":true", "\"verified\":false"),
        LOG.replace("\"median\":2.0", "\"median\":3.0"),
        LOG.replace("\"gflops\":1.0", "\"gflops\":1.5"),
        LOG.replace(
            "\"compilation_ns\":100",
            "\"compilation_ns\":340282366920938463463374607431768211455",
        ),
    ] {
        assert_ne!(bad, LOG, "corruption must change the fixture");
        assert!(report::markdown(&bad, 2, config).is_err(), "accepted {bad}");
    }
    assert!(report::markdown(LOG, 0, config).is_err());
    assert!(
        report::markdown(
            LOG,
            2,
            BootstrapConfig {
                resamples: 0,
                seed: 1
            }
        )
        .is_err()
    );
}

#[test]
fn protocol_groups_do_not_pool_rounds_or_rank_incompatible_measurements() {
    let other = LOG.replace("ffeeddccbbaa9988", "1111111111111111");
    let out = report::markdown(&format!("{LOG}{other}"), 1, BootstrapConfig::default()).unwrap();
    assert_eq!(out.matches("## Protocol ").count(), 2);
    assert_eq!(
        out.matches("Records: 3. Measurement: 56000000 ns. Compilation API work: 650 ns")
            .count(),
        2
    );
    assert_eq!(out.matches("| 1 | random | vectorize(1, j)").count(), 2);
    // Stable tie order and top-k do not remove charges from rows outside the table.
    assert!(!out.contains("split(1, j"));
}

#[test]
fn compares_best_candidates_with_explicit_pairing_and_exact_protocol_match() {
    let config = BootstrapConfig {
        resamples: 1000,
        seed: 42,
    };
    let b = include_str!("fixtures/report-b.jsonl");
    let independent = report::compare(b, LOG, config, false).unwrap();
    assert!(independent.contains("Median ratio A/B: 2.000000"));
    assert!(independent.contains("independent resampling"));
    assert!(!independent.contains("95% CI [2.000000, 2.000000]"));
    assert_eq!(independent, report::compare(b, LOG, config, false).unwrap());
    let paired = report::compare(b, LOG, config, true).unwrap();
    assert!(paired.contains("95% CI [2.000000, 2.000000]"));
    assert!(paired.contains("paired (asserted by caller)"));
    assert!(paired.contains("| A | vectorize(1, j)"));
    for (before, after) in [
        ("ffeeddccbbaa9988", "1111111111111111"),
        ("0011223344556677", "1111111111111111"),
        ("fixture CPU", "another CPU"),
        (
            "\"harness_profile\":\"release\"",
            "\"harness_profile\":\"debug\"",
        ),
    ] {
        let changed = LOG.replace(before, after);
        assert_ne!(LOG, changed);
        assert!(
            report::compare(LOG, &changed, config, false)
                .unwrap_err()
                .to_string()
                .contains("protocol key mismatch")
        );
        assert!(report::compare(&format!("{LOG}{changed}"), LOG, config, false).is_err());
    }
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/report.jsonl");
    assert!(
        report::command(
            ["--compare", path, path, "--paired", "--resamples", "10"].map(str::to_string)
        )
        .unwrap()
        .contains("95% CI [1.000000, 1.000000]")
    );
    assert!(report::command(["--compare", path].map(str::to_string)).is_err());
    assert!(report::command(["--compare", path, path, "--top", "1"].map(str::to_string)).is_err());
}

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
