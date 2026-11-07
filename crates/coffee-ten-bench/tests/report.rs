use coffee_ten_bench::{report, stats::BootstrapConfig};

const LOG: &str = include_str!("fixtures/report.jsonl");

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
        LOG.replace("\"measurement_ns\":28", "\"measurement_ns\":29"),
        LOG.replace("\"verified\":true", "\"verified\":false"),
        LOG.replace("\"median\":1.9999999999999998", "\"median\":3.0"),
        LOG.replace("\"gflops\":0.9999999999999999", "\"gflops\":1.5"),
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
        out.matches("Records: 3. Measurement: 56 ns. Compilation API work: 650 ns")
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
