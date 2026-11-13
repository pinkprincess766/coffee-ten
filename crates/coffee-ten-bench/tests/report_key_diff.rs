//! `report --compare` says which protocol fields differ, with both values.
use coffee_ten_bench::{report, stats::BootstrapConfig};

const LOG: &str = include_str!("fixtures/report.jsonl");
const CONFIG: BootstrapConfig = BootstrapConfig {
    resamples: 10,
    seed: 1,
};

fn compare_error(a: &str, b: &str) -> String {
    report::compare(a, b, CONFIG, false)
        .unwrap_err()
        .to_string()
}

#[test]
fn a_mismatch_lists_each_differing_field_with_both_values() {
    let one = LOG.replace(
        "\"source_hash\":\"ffeeddccbbaa9988\"",
        "\"source_hash\":\"1111111111111111\"",
    );
    assert_eq!(
        compare_error(LOG, &one),
        "report: protocol key mismatch between A and B: source_hash (A: \"ffeeddccbbaa9988\", B: \"1111111111111111\")"
    );
    // Several fields, sorted by name; numbers and strings keep their JSON form.
    let several = one
        .replace(
            "\"rustflags\":\"\"",
            "\"rustflags\":\"-C target-cpu=native\"",
        )
        .replace("\"warmup_ms\":5", "\"warmup_ms\":300");
    assert_eq!(
        compare_error(LOG, &several),
        "report: protocol key mismatch between A and B: rustflags (A: \"\", B: \"-C target-cpu=native\"); \
         source_hash (A: \"ffeeddccbbaa9988\", B: \"1111111111111111\"); warmup_ms (A: 5, B: 300)"
    );
    // Swapping the files swaps the labels.
    assert!(
        compare_error(&one, LOG)
            .ends_with("source_hash (A: \"1111111111111111\", B: \"ffeeddccbbaa9988\")")
    );
}

#[test]
fn values_are_quoted_so_odd_text_cannot_confuse_the_message() {
    let odd = LOG.replace(
        "\"cpu\":\"fixture CPU\"",
        "\"cpu\":\"a \\\"quoted\\\" (B: x); cpu\"",
    );
    // The row's own cpu copy must agree with its protocol, so change both.
    assert_ne!(odd, LOG);
    let err = compare_error(LOG, &odd);
    assert!(
        err.ends_with("cpu (A: \"fixture CPU\", B: \"a \\\"quoted\\\" (B: x); cpu\")"),
        "{err}"
    );
}

#[test]
fn a_file_with_two_keys_names_the_file_and_the_fields() {
    let other = LOG.replace("\"rustflags\":\"\"", "\"rustflags\":\"-g\"");
    let mixed = format!("{LOG}{other}");
    let expected = "comparison requires exactly one protocol per file; {} mixes keys: rustflags (first: \"\", later: \"-g\")";
    assert_eq!(
        compare_error(&mixed, LOG),
        format!("report: {}", expected.replace("{}", "A"))
    );
    assert_eq!(
        compare_error(LOG, &mixed),
        format!("report: {}", expected.replace("{}", "B"))
    );
}
