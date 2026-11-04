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
    assert!(records[0].reference_compilation.is_some());
    assert!(records[1].reference_compilation.is_none());
    for record in &records {
        assert!(record.compilation_ns > 0);
        assert_eq!(record.min_sample_ms, config.min_sample_ms);
        assert!(record.quick);
        assert_eq!(record.samples, QUICK_SAMPLES);
        assert_eq!(record.rounds.len(), record.samples);
        for (index, round) in record.rounds.iter().enumerate() {
            assert_eq!(round.round_index, index);
            assert!(round.elapsed_ns > 0 && round.iterations > 0);
            let expected = 2.0 * 16.0f64.powi(3) * round.iterations as f64
                / std::time::Duration::from_nanos(round.elapsed_ns).as_secs_f64()
                / 1e9;
            assert_eq!(round.gflops.to_bits(), expected.to_bits());
        }
        let rates: Vec<_> = record.rounds.iter().map(|r| r.gflops).collect();
        assert_eq!(
            coffee_ten_bench::stats::summarize(&rates)
                .unwrap()
                .median
                .to_bits(),
            record.stats.median.to_bits()
        );
        let s = record.stats;
        assert!(s.min > 0.0, "{record:?}");
        assert!(
            s.min <= s.p25 && s.p25 <= s.median && s.median <= s.p75,
            "{record:?}"
        );
        assert!(s.median.is_finite());
    }
}
