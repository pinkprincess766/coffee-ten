//! A record, its raw rounds and its reference compilation accept only the fields the
//! harness writes, as in `bench/validate.py`; an unknown field is named, not ignored.
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
fn unknown_fields_are_named_at_each_level() {
    let line = LOG.lines().nth(1).unwrap();
    let reference = line.contains("\"reference_compilation\":{");
    let mut cases = vec![
        (
            "{\"kernel\"",
            "{\"note\":1,\"kernel\"",
            "unknown record field note",
        ),
        (
            "{\"kernel\"",
            "{\"Kernel\":\"matmul\",\"kernel\"",
            "unknown record field Kernel",
        ),
        (
            "{\"round_index\":0,",
            "{\"round_index\":0,\"warm\":true,",
            "round 0: unknown field warm",
        ),
        (
            "{\"round_index\":1,",
            "{\"round_index\":1,\"x\":null,",
            "round 1: unknown field x",
        ),
    ];
    if reference {
        cases.push((
            "\"reference_compilation\":{",
            "\"reference_compilation\":{\"wall_ns\":1,",
            "unknown reference_compilation field wall_ns",
        ));
    }
    for (from, to, message) in cases {
        assert_eq!(
            rejected(&with_line_2(from, to)),
            format!("report: line 2: {message}"),
            "{to}"
        );
    }
}

#[test]
fn every_written_field_is_still_accepted() {
    // The fixture carries every written field: an object and a null reference compilation,
    // and the optional group, which may also be absent.
    report::markdown(LOG, 10, CONFIG).unwrap();
    report::markdown(&with_line_2(",\"group\":\"naive\"", ""), 10, CONFIG).unwrap();
}
