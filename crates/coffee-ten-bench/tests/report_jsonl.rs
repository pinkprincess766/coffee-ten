//! Line handling of the JSONL reader behind `report`: endings, blank lines (rejected, as in
//! `bench/validate.py`), truncation and
//! records that are not measurement objects.
use coffee_ten_bench::{report, stats::BootstrapConfig};

const LOG: &str = include_str!("fixtures/report.jsonl");
const CONFIG: BootstrapConfig = BootstrapConfig {
    resamples: 7,
    seed: 3,
};

/// The report without its first lines, which carry the input-byte fingerprint.
fn body(report: &str) -> &str {
    &report[report.find("## Protocol").unwrap()..]
}

fn error(text: &str) -> String {
    report::markdown(text, 2, CONFIG).unwrap_err().to_string()
}

#[test]
fn crlf_and_unterminated_files_read_like_the_plain_file() {
    let plain = report::markdown(LOG, 2, CONFIG).unwrap();
    let crlf = LOG.replace('\n', "\r\n");
    let unterminated = LOG.trim_end_matches('\n');
    for variant in [crlf.as_str(), unterminated] {
        assert_ne!(variant, LOG);
        let out = report::markdown(variant, 2, CONFIG).unwrap();
        // Same tables; only the fingerprint of the exact input bytes differs.
        assert_eq!(body(&out), body(&plain));
        assert_ne!(out, plain);
    }
}

#[test]
fn blank_and_whitespace_lines_are_errors_with_their_line_number() {
    let lines: Vec<_> = LOG.lines().collect();
    for (text, line) in [
        (format!("\n{LOG}"), 1),
        (format!("{}\n\n{}\n", lines[0], lines[1..].join("\n")), 2),
        (
            format!(
                "{}\n  \t\n{}\n",
                lines[..2].join("\n"),
                lines[2..].join("\n")
            ),
            3,
        ),
        (format!("{LOG}\n"), 5),
        (format!("{LOG} \r\n"), 5),
        (LOG.replace('\n', "\r\n").replacen("\r\n", "\r\n\r\n", 1), 2),
    ] {
        assert_eq!(
            error(&text),
            format!("report: line {line}: empty JSONL line")
        );
        let compared = report::compare(LOG, &text, CONFIG, false)
            .unwrap_err()
            .to_string();
        assert_eq!(compared, format!("report: line {line}: empty JSONL line"));
    }
}

#[test]
fn errors_report_the_physical_line_number() {
    let lines: Vec<_> = LOG.lines().collect();
    // A record cut off mid-write (an interrupted append) on the last line.
    let last = lines[3];
    let truncated = format!(
        "{}\n{}\n{}\n{}",
        lines[0],
        lines[1],
        lines[2],
        &last[..last.len() / 2]
    );
    assert!(
        error(&truncated).starts_with("report: line 4: "),
        "{}",
        error(&truncated)
    );
    // Skipped peak lines still count, so the number matches what an editor shows.
    let shifted = format!("{}\n{}\n{truncated}", lines[0], lines[0]);
    assert!(
        error(&shifted).starts_with("report: line 6: "),
        "{}",
        error(&shifted)
    );
    // The same holds for `--compare`, on either side.
    let compared = report::compare(LOG, &truncated, CONFIG, false)
        .unwrap_err()
        .to_string();
    assert!(compared.starts_with("report: line 4: "), "{compared}");
}

#[test]
fn every_line_must_be_one_json_object() {
    for line in ["[]", "1", "\"text\"", "null", "true"] {
        let text = format!("{LOG}{line}\n");
        assert_eq!(
            error(&text),
            "report: line 5: expected one JSON object per line",
            "{line}"
        );
    }
    // Two records glued onto one line are not silently split or half-read.
    let lines: Vec<_> = LOG.lines().collect();
    let glued = format!("{}\n{}{}\n", lines[0], lines[1], lines[2]);
    assert!(
        error(&glued).contains("trailing JSON data"),
        "{}",
        error(&glued)
    );
}

#[test]
fn record_kinds_are_strings_and_only_peak_rows_are_skipped() {
    assert_eq!(
        error(&format!("{LOG}{{\"kind\":1}}\n")),
        "report: line 5: kind: expected string"
    );
    for kind in ["replay_curve", "", "Peak"] {
        let text = format!("{LOG}{{\"kind\":\"{kind}\"}}\n");
        assert!(
            error(&text).contains("expected live measurements"),
            "{kind}"
        );
    }
    // A peak row is skipped whatever else it carries; a log of only peaks has no records.
    let peaks = "{\"kind\":\"peak\",\"anything\":[1,{}]}\n{\"kind\":\"peak\"}\n";
    assert_eq!(error(peaks), "report: no measurement records");
    let with_peaks = report::markdown(&format!("{peaks}{LOG}"), 2, CONFIG).unwrap();
    assert_eq!(
        body(&with_peaks),
        body(&report::markdown(LOG, 2, CONFIG).unwrap())
    );
}

#[test]
fn a_byte_order_mark_is_rejected_not_skipped() {
    let text = format!("\u{feff}{LOG}");
    assert!(
        error(&text).starts_with("report: line 1: invalid JSON value"),
        "{}",
        error(&text)
    );
}
