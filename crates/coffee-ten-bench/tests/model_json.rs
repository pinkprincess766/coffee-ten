use coffee_ten_bench::model::{compare_report, machine_from_results};

const RUN: &str = include_str!("../../../bench/results/2026-10-18-4928ffa.jsonl");

#[test]
fn present_optional_peak_measurements_must_have_numeric_medians() {
    const PEAK: &str = r#"{"kind":"peak","fma_gflops":{"median":100},"triad_gbs":{"median":90},"triad_l2_gbs":{"median":110}}"#;
    let baseline = machine_from_results(PEAK).unwrap();
    for (key, valid) in [
        ("fma_mem_acc_gflops", baseline.0.mem_acc_gflops),
        (
            "fma_mem_acc_scalar_gflops",
            baseline.0.mem_acc_scalar_gflops,
        ),
        (
            "fma_mem_acc_scalar_unrolled_gflops",
            baseline.0.mem_acc_scalar_unrolled_gflops,
        ),
        ("fma_col_reload_gflops", baseline.0.col_reload_gflops),
        ("fma_col_unrolled_gflops", baseline.0.col_unrolled_gflops),
        ("fma_store_gflops", baseline.0.store_gflops),
        ("fma_chain_gflops", baseline.0.fma_chain_gflops),
        ("fma_chain256_gflops", baseline.0.fma_chain256_gflops),
    ] {
        let prefix = &PEAK[..PEAK.len() - 1];
        let good = format!("{prefix},\"{key}\":{{\"median\":{valid}}}}}");
        assert_eq!(machine_from_results(&good).unwrap().0, baseline.0);
        for malformed in [
            "null",
            "{}",
            r#"{"median":null}"#,
            r#"{"median":"broken"}"#,
            r#"{"median":1e999}"#,
        ] {
            let bad = format!("{prefix},\"{key}\":{malformed}}}");
            let error = machine_from_results(&bad).expect_err(key).to_string();
            assert!(error.contains(key), "{error}");
        }
    }
}

#[test]
fn rejects_fractional_sizes_and_nonpositive_measured_rates() {
    let peak = RUN
        .lines()
        .find(|line| line.contains("\"kind\":\"peak\""))
        .unwrap();
    let mut accepted = Vec::new();
    for (n, median) in [("2.5", "1"), ("2", "0"), ("2", "-1")] {
        let record = format!(
            "{peak}\n{{\"kernel\":\"matmul\",\"n\":{n},\"median\":{median},\"schedule\":\"\",\"drift\":false}}\n"
        );
        if compare_report("fixture", &record, None).is_ok() {
            accepted.push((n, median));
        }
    }
    assert!(
        accepted.is_empty(),
        "accepted invalid numeric fields: {accepted:?}"
    );
}

#[test]
fn json_whitespace_cannot_reinclude_drifted_records() {
    let changed = RUN.replace("\"drift\":true", "\"drift\": true");
    assert_ne!(RUN, changed, "fixture must contain a drifted record");
    assert_eq!(
        compare_report("fixture", RUN, None).unwrap(),
        compare_report("fixture", &changed, None).unwrap()
    );
}

#[test]
fn peak_and_record_fields_accept_standard_json_spacing_and_escapes() {
    let changed = RUN.replace("\":", "\" \t: \t");
    assert_eq!(
        machine_from_results(RUN).unwrap(),
        machine_from_results(&changed).unwrap()
    );
    assert_eq!(
        compare_report("fixture", RUN, None).unwrap(),
        compare_report("fixture", &changed, None).unwrap()
    );
    let escaped = RUN
        .replace("\"kind\"", "\"\\u006bind\"")
        .replace("\"matmul\"", "\"mat\\u006dul\"");
    assert_eq!(
        compare_report("fixture", RUN, None).unwrap(),
        compare_report("fixture", &escaped, None).unwrap()
    );
}
