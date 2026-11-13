//! `report` checks record metadata it does not use (commit, date, dirty, order_class, group)
//! with the same rules as `bench/validate.py`, so both accept and reject the same logs.
use coffee_ten_bench::{report, stats::BootstrapConfig};

const LOG: &str = include_str!("fixtures/report.jsonl");
const CONFIG: BootstrapConfig = BootstrapConfig {
    resamples: 10,
    seed: 1,
};

/// The fixture with one replacement applied to the first measurement record (line 2) only.
fn with_line_2(from: &str, to: &str) -> String {
    let mut lines: Vec<String> = LOG.lines().map(str::to_string).collect();
    assert!(lines[1].contains(from), "{from}");
    lines[1] = lines[1].replacen(from, to, 1);
    lines.join("\n") + "\n"
}

fn rejected(text: &str) -> String {
    report::markdown(text, 10, CONFIG).unwrap_err().to_string()
}

#[test]
fn accepted_metadata_forms() {
    report::markdown(LOG, 10, CONFIG).unwrap();
    for (from, to) in [
        ("\"dirty\":false", "\"dirty\":null"),
        ("\"dirty\":false", "\"dirty\":true"),
        (
            "\"order_class\":\"preserving\"",
            "\"order_class\":\"reordering\"",
        ),
        ("\"commit\":\"fixture\"", "\"commit\":\"\""),
        (",\"group\":\"naive\"", ""),
    ] {
        report::markdown(&with_line_2(from, to), 10, CONFIG)
            .unwrap_or_else(|e| panic!("{to}: {e}"));
    }
    for group in ["random", "model", "local", "final"] {
        let text = LOG.replacen("\"group\":\"random\"", &format!("\"group\":\"{group}\""), 1);
        report::markdown(&text, 10, CONFIG).unwrap_or_else(|e| panic!("{group}: {e}"));
    }
}

#[test]
fn missing_metadata_is_named() {
    for field in ["commit", "date", "dirty", "order_class"] {
        let line = LOG.lines().nth(1).unwrap();
        let start = line.find(&format!("\"{field}\":")).unwrap();
        let end = start + line[start..].find(',').unwrap() + 1;
        let text = with_line_2(&line[start..end], "");
        assert_eq!(rejected(&text), format!("report: line 2: missing {field}"));
    }
}

#[test]
fn wrong_metadata_types_and_values_are_named() {
    for (from, to, message) in [
        (
            "\"commit\":\"fixture\"",
            "\"commit\":1",
            "commit: expected string",
        ),
        (
            "\"date\":\"fixture\"",
            "\"date\":null",
            "date: expected string",
        ),
        (
            "\"dirty\":false",
            "\"dirty\":\"no\"",
            "dirty: expected bool or null",
        ),
        (
            "\"dirty\":false",
            "\"dirty\":0",
            "dirty: expected bool or null",
        ),
        (
            "\"order_class\":\"preserving\"",
            "\"order_class\":\"sideways\"",
            "invalid order_class \"sideways\"",
        ),
        (
            "\"order_class\":\"preserving\"",
            "\"order_class\":1",
            "order_class: expected string",
        ),
        (
            "\"group\":\"naive\"",
            "\"group\":\"harness\"",
            "invalid group \"harness\"",
        ),
        (
            "\"group\":\"naive\"",
            "\"group\":7",
            "group: expected string",
        ),
        (
            "\"schedule\":\"\"",
            "\"schedule\":\"vectorize(1, j)\"",
            "naive schedule must be empty",
        ),
    ] {
        assert_eq!(
            rejected(&with_line_2(from, to)),
            format!("report: line 2: {message}"),
            "{to}"
        );
    }
}
