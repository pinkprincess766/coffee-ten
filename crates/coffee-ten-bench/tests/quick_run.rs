//! The harness logic (not the binary) on a small matmul, in quick mode.

use coffee_ten_bench::cli::{QUICK_SAMPLES, parse_args};
use coffee_ten_bench::run;

#[test]
fn quick_run_checks_and_times_naive_and_vectorized_matmul() {
    let args = [
        "--kernel",
        "matmul",
        "--n",
        "16",
        "--variant",
        "",
        "--variant",
        "vectorize(1, j)",
        "--quick",
    ];
    let config = parse_args(args.iter().map(|a| a.to_string())).unwrap();
    // `run` aborts with `BenchError::Mismatch` before timing if a variant is wrong, so an
    // `Ok` means both variants passed the correctness check.
    let records = run(&config).unwrap();
    assert_eq!(records.len(), 2);
    assert_eq!(records[0].variant, "");
    assert_eq!(records[1].variant, "vectorize(1, j)");
    for record in &records {
        assert!(record.quick);
        assert_eq!(record.n, QUICK_SAMPLES);
        let s = record.stats;
        assert!(s.min > 0.0, "{record:?}");
        assert!(
            s.min <= s.p25 && s.p25 <= s.median && s.median <= s.p75,
            "{record:?}"
        );
        assert!(s.median.is_finite());
    }
}
