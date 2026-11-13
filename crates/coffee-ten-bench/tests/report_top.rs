//! `--top` limits the ranked rows, never the costs, and the report says how many it shows.
use coffee_ten_bench::{report, stats::BootstrapConfig};

const LOG: &str = include_str!("fixtures/report.jsonl");
const CONFIG: BootstrapConfig = BootstrapConfig {
    resamples: 10,
    seed: 1,
};

fn table_rows(out: &str) -> usize {
    out.lines()
        .filter(|l| l.starts_with("| ") && !l.starts_with("| Rank") && !l.starts_with("| ---"))
        .count()
}

#[test]
fn shown_count_is_min_of_top_and_records() {
    for (top, shown) in [(1, 1), (2, 2), (3, 3), (4, 3), (usize::MAX, 3)] {
        let out = report::markdown(LOG, top, CONFIG).unwrap();
        assert!(
            out.contains(&format!("Showing {shown} of 3 records (--top {top}).")),
            "{top}: {out}"
        );
        assert_eq!(table_rows(&out), shown, "{top}");
        // Costs always cover all three records, whatever --top is.
        assert!(out.contains("Records: 3. Measurement: 56000000 ns. Compilation API work: 650 ns"));
    }
}

#[test]
fn each_protocol_table_counts_its_own_records() {
    let other = LOG.replace("ffeeddccbbaa9988", "1111111111111111");
    let lines: Vec<_> = other.lines().collect();
    // The second protocol gets only its first measurement row (plus the peak line).
    let one_row = format!("{}\n{}\n", lines[0], lines[1]);
    let out = report::markdown(&format!("{LOG}{one_row}"), 2, CONFIG).unwrap();
    assert!(out.contains("Showing 2 of 3 records (--top 2)."), "{out}");
    assert!(out.contains("Showing 1 of 1 records (--top 2)."), "{out}");
}
