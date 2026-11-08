use coffee_ten_bench::model::{compare_report, machine_from_results};

const RUN: &str = include_str!("../../../bench/results/2026-10-18-4928ffa.jsonl");

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
