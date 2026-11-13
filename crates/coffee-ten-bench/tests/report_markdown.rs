//! Text copied from a log into the Markdown report cannot change the report's structure.
use coffee_ten_bench::{report, stats::BootstrapConfig};

const LOG: &str = include_str!("fixtures/report.jsonl");
const CONFIG: BootstrapConfig = BootstrapConfig {
    resamples: 10,
    seed: 1,
};
/// Everything Markdown or HTML could interpret, as a JSON string body.
const HOSTILE: &str = r#"a|b `c` <i>d</i> &amp; *e* _f_ [g](h) \\ ~~i~~ \nj"#;
/// What `HOSTILE` must look like in the report.
const ESCAPED: &str = "a&#124;b &#96;c&#96; &lt;i&gt;d&lt;/i&gt; &amp;amp; &#42;e&#42; &#95;f&#95; &#91;g&#93;(h) &#92; &#126;&#126;i&#126;&#126;  j";

fn hostile_log() -> String {
    // `group` is no longer free text (one of naive/random/model/local/final), so it cannot
    // carry hostile text; see `hostile_group_is_rejected_not_rendered`.
    let text = LOG
        .replace(
            "\"schedule\":\"vectorize(1, j)\"",
            &format!("\"schedule\":\"{HOSTILE}\""),
        )
        // cpu and kernel appear in both the row and its protocol; replace every copy.
        .replace("\"cpu\":\"fixture CPU\"", &format!("\"cpu\":\"{HOSTILE}\""))
        .replace(
            "\"kernel\":\"matmul\"",
            &format!("\"kernel\":\"{HOSTILE}\""),
        );
    assert_ne!(text, LOG);
    text
}

/// Table rows keep exactly their column separators.
fn assert_tables_intact(out: &str, columns: usize) {
    for line in out.lines().filter(|l| l.starts_with('|')) {
        assert_eq!(line.matches('|').count(), columns + 1, "{line}");
    }
}

#[test]
fn report_escapes_every_copied_field() {
    let out = report::markdown(&hostile_log(), 10, CONFIG).unwrap();
    assert!(
        out.contains(&format!("## Protocol 1: {ESCAPED}, n=2")),
        "{out}"
    );
    assert!(out.contains(&format!("- cpu: {ESCAPED}\n")), "{out}");
    // Group and schedule cells of the random row whose schedule was replaced.
    assert_eq!(
        out.matches(&format!("| random | {ESCAPED} |")).count(),
        1,
        "{out}"
    );
    assert_tables_intact(&out, 7);
    for raw in ["<i>", "`c`", "*e*", "_f_", "[g]", "~~i~~", "a|b"] {
        assert!(!out.contains(raw), "{raw} leaked: {out}");
    }
}

#[test]
fn hostile_group_is_rejected_not_rendered() {
    let text = LOG.replace("\"group\":\"random\"", &format!("\"group\":\"{HOSTILE}\""));
    let err = report::markdown(&text, 10, CONFIG).unwrap_err().to_string();
    assert!(err.contains("invalid group"), "{err}");
}

#[test]
fn compare_escapes_the_winning_schedule() {
    let text = hostile_log();
    let out = report::compare(&text, &text, CONFIG, true).unwrap();
    assert!(out.contains(&format!("| A | {ESCAPED} |")), "{out}");
    assert!(out.contains(&format!("| B | {ESCAPED} |")), "{out}");
    assert_tables_intact(&out, 5);
}

#[test]
fn ordinary_text_is_unchanged() {
    // Schedules, hashes and the fixture's metadata contain none of the escaped characters.
    let out = report::markdown(LOG, 10, CONFIG).unwrap();
    for plain in [
        "split(1, j, jo, ji, 2)",
        "vectorize(1, j)",
        "- cpu: fixture CPU",
        "- space_hash: 0011223344556677",
    ] {
        assert!(out.contains(plain), "{plain}");
    }
}
