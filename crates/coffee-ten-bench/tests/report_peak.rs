//! Peak lines are checked as `bench/validate.py` checks them, then left out of the tables:
//! exactly the current fields, positive sizes, finite positive and ordered statistics.
use coffee_ten_bench::{report, stats::BootstrapConfig};

const LOG: &str = include_str!("fixtures/report.jsonl");
const CONFIG: BootstrapConfig = BootstrapConfig {
    resamples: 10,
    seed: 1,
};

/// The fixture with one replacement applied to its peak line (line 1) only.
fn with_peak(from: &str, to: &str) -> String {
    let mut lines: Vec<String> = LOG.lines().map(str::to_string).collect();
    assert!(lines[0].starts_with("{\"kind\":\"peak\""));
    assert!(lines[0].contains(from), "{from}");
    lines[0] = lines[0].replacen(from, to, 1);
    lines.join("\n") + "\n"
}

fn rejected(text: &str) -> String {
    report::markdown(text, 10, CONFIG).unwrap_err().to_string()
}

#[test]
fn the_fixture_and_real_saved_peak_lines_are_accepted() {
    report::markdown(LOG, 10, CONFIG).unwrap();
    // The newest saved peak line has every current field.
    let saved = include_str!("../../../bench/results/2026-10-28-adcf09c-peak.jsonl");
    let records: Vec<_> = LOG.lines().skip(1).collect();
    report::markdown(&format!("{}{}\n", saved, records.join("\n")), 10, CONFIG).unwrap();
}

#[test]
fn malformed_peak_lines_are_named() {
    let stats = "{\"median\":2.0,\"p25\":1.5,\"p75\":2.5,\"min\":1.0,\"n\":5}";
    for (from, to, message) in [
        (
            "{\"kind\":\"peak\",",
            "{\"kind\":\"peak\",\"fixture\":true,",
            "peak: unknown field fixture",
        ),
        (",\"mem_acc_n\":1", "", "peak: missing mem_acc_n"),
        (
            &format!("\"fma_gflops\":{stats},"),
            "",
            "peak: missing fma_gflops",
        ),
        (
            "\"triad_n\":1",
            "\"triad_n\":0",
            "peak triad_n: expected positive integer",
        ),
        (
            "\"triad_n\":1",
            "\"triad_n\":1.5",
            "peak triad_n: expected unsigned integer",
        ),
        (
            "\"fma_gflops\":{",
            "\"fma_gflops\":[{",
            "peak fma_gflops: expected object",
        ),
        (
            "\"fma_gflops\":{\"median\"",
            "\"fma_gflops\":{\"mean\":1.0,\"median\"",
            "peak fma_gflops: unknown field mean",
        ),
        (
            "\"n\":5",
            "\"n\":0",
            "peak fma_gflops n: expected positive integer",
        ),
        (",\"n\":5", "", "peak fma_gflops: missing n"),
        (
            "\"min\":1.0",
            "\"min\":0.0",
            "peak fma_gflops min: expected finite positive rate",
        ),
        (
            "\"min\":1.0",
            "\"min\":null",
            "peak fma_gflops min: expected finite number",
        ),
        (
            "\"median\":2.0",
            "\"median\":3.0",
            "peak fma_gflops: unordered statistics",
        ),
        (
            "\"p25\":1.5",
            "\"p25\":0.5",
            "peak fma_gflops: unordered statistics",
        ),
    ] {
        let mut text = with_peak(from, to);
        if to.ends_with("[{") {
            text = text.replacen("\"n\":5}", "\"n\":5}]", 1);
        }
        assert_eq!(
            rejected(&text),
            format!("report: line 1: {message}"),
            "{to}"
        );
    }
}

#[test]
fn a_peak_line_from_an_older_session_is_rejected_with_its_missing_field() {
    // 2026-10-28-827cd1d predates fma_col_unrolled_gflops; validate.py rejects it too.
    let old = include_str!("../../../bench/results/2026-10-28-827cd1d-peak.jsonl");
    let records: Vec<_> = LOG.lines().skip(1).collect();
    assert_eq!(
        rejected(&format!("{}{}\n", old, records.join("\n"))),
        "report: line 1: peak: missing fma_col_unrolled_gflops"
    );
}
