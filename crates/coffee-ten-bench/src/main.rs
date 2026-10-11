//! `coffee-ten-bench`: see the library docs and `USAGE`.

use coffee_ten_bench::cli::Config;
use coffee_ten_bench::meta::{self, Metadata};
use coffee_ten_bench::stats::{DRIFT_LIMIT, DRIFT_ROUNDS, MIN_ROUNDS_FOR_DRIFT};
use coffee_ten_bench::{Record, parse_args, run};
use std::process::ExitCode;

/// Raises the chance that this thread runs on a P-core; it is not a guarantee.
#[cfg(target_os = "macos")]
fn raise_qos() {
    /// `QOS_CLASS_USER_INTERACTIVE` of `<sys/qos.h>`.
    const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;

    // SAFETY: declares the libpthread function of `<pthread/qos.h>`, whose C signature is
    // `int pthread_set_qos_class_self_np(qos_class_t, int)`; `qos_class_t` is `unsigned int`.
    unsafe extern "C" {
        fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
    }
    // SAFETY: the call takes two integers, touches no memory of ours and only changes the
    // scheduling class of the calling thread; a nonzero result is reported, not trusted.
    let status = unsafe { pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0) };
    if status != 0 {
        eprintln!("warning: could not set the QoS class (status {status})");
    }
}

#[cfg(not(target_os = "macos"))]
fn raise_qos() {}

fn main() -> ExitCode {
    let config = match parse_args(std::env::args().skip(1)) {
        Ok(config) => config,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };
    raise_qos();
    let metadata = meta::collect(config.seed);
    print_header(&config, &metadata);
    match run(&config) {
        Ok(records) => {
            print_table(&records);
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn print_header(config: &Config, m: &Metadata) {
    if config.quick {
        println!("quick: true (3 samples, not a measurement)");
    }
    println!(
        "kernel {}  n {}  samples {}  min-sample {} ms  seed {}",
        config.kernel.name(),
        config.n,
        config.samples,
        config.min_sample_ms,
        m.seed
    );
    println!(
        "machine {}  cpu {}  macOS {}",
        m.hw_model, m.cpu_brand, m.os_version
    );
    println!("clang {}", m.clang);
    println!("flags {}  (FMA contraction allowed)", m.flags);
    println!("git {}  date {}", m.git_rev, m.date_utc);
}

fn print_table(records: &[Record]) {
    println!(
        "{:>3} {:>10} {:>10} {:>10} {:>10} {:>3} {:>6}  schedule",
        "#", "median", "p25", "p75", "min", "n", "drift"
    );
    for (index, r) in records.iter().enumerate() {
        // The check needs two disjoint windows of rounds; "-" means it did not run.
        let drift = match (r.n >= MIN_ROUNDS_FOR_DRIFT, r.drift) {
            (false, _) => "-",
            (true, false) => "no",
            (true, true) => "true",
        };
        let schedule = if r.variant.is_empty() {
            "(naive)"
        } else {
            &r.variant
        };
        println!(
            "{index:>3} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>3} {:>6}  {schedule}",
            r.stats.median, r.stats.p25, r.stats.p75, r.stats.min, r.n, drift
        );
        if r.drift {
            eprintln!(
                "warning: variant #{index} drifted by more than {:.0}% between the first and last {DRIFT_ROUNDS} rounds (throttling or other load?)",
                DRIFT_LIMIT * 100.0
            );
        }
    }
    println!("rates in GFLOP/s, 2*M*N*K per call");
}
