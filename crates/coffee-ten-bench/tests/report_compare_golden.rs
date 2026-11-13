//! `report --compare` output is pinned byte for byte, through the API and the real binary,
//! like the single-run golden in `report.rs`. A change to the comparison text, the ratio, the
//! seeded bootstrap or the table layout shows up here as a reviewed golden diff.
use coffee_ten_bench::{report, stats::BootstrapConfig};
use std::process::Command;

const A: &str = include_str!("fixtures/report-b.jsonl");
const B: &str = include_str!("fixtures/report.jsonl");
const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
const CONFIG: BootstrapConfig = BootstrapConfig {
    resamples: 257,
    seed: 42,
};

fn binary(paired: bool) -> String {
    let mut args = vec![
        "report".to_string(),
        "--compare".into(),
        format!("{DIR}/report-b.jsonl"),
        format!("{DIR}/report.jsonl"),
        "--resamples".into(),
        "257".into(),
        "--seed".into(),
        "42".into(),
    ];
    if paired {
        args.push("--paired".into());
    }
    let out = Command::new(env!("CARGO_BIN_EXE_coffee-ten-bench"))
        .args(&args)
        .output()
        .unwrap();
    assert!(out.status.success(), "{out:?}");
    assert!(out.stderr.is_empty());
    String::from_utf8(out.stdout).unwrap()
}

#[test]
fn independent_comparison_matches_its_golden() {
    let expected = include_str!("fixtures/report-compare.md");
    assert_eq!(report::compare(A, B, CONFIG, false).unwrap(), expected);
    assert_eq!(binary(false), expected);
}

#[test]
fn paired_comparison_matches_its_golden() {
    let expected = include_str!("fixtures/report-compare-paired.md");
    assert_eq!(report::compare(A, B, CONFIG, true).unwrap(), expected);
    assert_eq!(binary(true), expected);
}

#[test]
fn comparing_reads_the_inputs_without_changing_them() {
    binary(false);
    binary(true);
    assert_eq!(
        std::fs::read_to_string(format!("{DIR}/report-b.jsonl")).unwrap(),
        A
    );
    assert_eq!(
        std::fs::read_to_string(format!("{DIR}/report.jsonl")).unwrap(),
        B
    );
}
