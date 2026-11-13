//! `coffee-ten-bench`: see the library docs and `USAGE`.

use coffee_ten_bench::cli::Config;
use coffee_ten_bench::log::append_lines;
use coffee_ten_bench::meta::{self, Metadata};
use coffee_ten_bench::run::record_json;
use coffee_ten_bench::space::{
    SpaceConfig, enumerate_matmul_for, parse_space_config_args, summary,
};
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

/// Exit status of an offline subcommand (`report`, `curve`): 2 for a wrong command line, as the
/// harness's own argument errors, and 1 when the input data are rejected, as `search --replay`.
fn exit_code(e: &coffee_ten_bench::BenchError) -> ExitCode {
    match e {
        coffee_ten_bench::BenchError::Usage(_) => ExitCode::from(2),
        _ => ExitCode::FAILURE,
    }
}

/// `coffee-ten-bench space --n <N>`: lists the schedule space, measures nothing.
fn space_main(args: impl Iterator<Item = String>) -> ExitCode {
    let mut args = args.peekable();
    if args.peek().map(String::as_str) == Some("reduce") {
        args.next();
        return match coffee_ten_bench::space::reduce_command(args) {
            Ok(text) => {
                print!("{text}");
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::from(2)
            }
        };
    }
    match parse_space_config_args(args) {
        Ok(config) => {
            let n = config.n;
            if config.space != SpaceConfig::V1 {
                print!("{}", config.space.description());
                println!(
                    "space hash {}",
                    coffee_ten_bench::protocol::space_hash_for(n, config.space)
                );
            }
            print!(
                "{}",
                summary(n, &enumerate_matmul_for(n, n, n, config.space))
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(2)
        }
    }
}

/// `coffee-ten-bench search --n <N> --budget <M>`: random search, one candidate at a time.
/// `--local` spends the budget on the model's hill climb instead.
fn search_main(args: impl Iterator<Item = String>) -> ExitCode {
    let config = match coffee_ten_bench::search::parse_search_args(args) {
        Ok(config) => config,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };
    if config.replay.is_some() {
        return match coffee_ten_bench::replay::run(&config, &meta::collect(config.seed)) {
            Ok(result) => {
                print!("{}", result.report());
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        };
    }
    raise_qos();
    let metadata = meta::collect(config.seed);
    match coffee_ten_bench::search::search(&config, &metadata) {
        Ok(result) => {
            print!(
                "{}",
                coffee_ten_bench::search::report(&config, &result, &metadata)
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

/// The usage for `--help`/`-h` anywhere on the command line, chosen by subcommand; `None`
/// without a help flag. `curve` parses its own arguments and is not covered here.
fn help_text(args: &[String]) -> Option<String> {
    if !args.iter().any(|a| a == "--help" || a == "-h") {
        return None;
    }
    Some(match args.first().map(String::as_str) {
        Some("report") => coffee_ten_bench::report::USAGE.to_string(),
        Some("search") => coffee_ten_bench::search::SEARCH_USAGE.to_string(),
        Some("space") => coffee_ten_bench::space::SPACE_USAGE.to_string(),
        Some("model") => coffee_ten_bench::model::USAGE.to_string(),
        Some("curve") => return None,
        _ => format!(
            "{}\nsubcommands: report, search, space, model, curve (each takes --help)",
            coffee_ten_bench::cli::USAGE
        ),
    })
}

fn main() -> ExitCode {
    let all: Vec<String> = std::env::args().skip(1).collect();
    if let Some(text) = help_text(&all) {
        println!("{text}");
        return ExitCode::SUCCESS;
    }
    let mut args = all.into_iter().peekable();
    if args.peek().map(String::as_str) == Some("curve") {
        args.next();
        return match coffee_ten_bench::curves::command(args) {
            Ok(text) => {
                print!("{text}");
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("error: {e}");
                exit_code(&e)
            }
        };
    }
    if args.peek().map(String::as_str) == Some("report") {
        args.next();
        return match coffee_ten_bench::report::command(args) {
            Ok(text) => {
                print!("{text}");
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("error: {e}");
                exit_code(&e)
            }
        };
    }
    if args.peek().map(String::as_str) == Some("space") {
        args.next();
        return space_main(args);
    }
    if args.peek().map(String::as_str) == Some("search") {
        args.next();
        return search_main(args);
    }
    if args.peek().map(String::as_str) == Some("model") {
        args.next();
        return coffee_ten_bench::model::cli_main(args);
    }
    let config = match parse_args(args) {
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
            if let Some(path) = &config.log {
                let lines: Vec<String> = records
                    .iter()
                    .map(|record| record_json(record, &metadata))
                    .collect();
                if let Err(e) = append_lines(path, &lines) {
                    eprintln!("error: cannot append to {}: {e}", path.display());
                    return ExitCode::FAILURE;
                }
            }
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
    let dirty = match m.git_dirty {
        Some(true) => " (dirty tree)",
        Some(false) => "",
        None => " (dirty: unknown)",
    };
    println!("git {}{dirty}  date {}", m.git_rev, m.date_utc);
}

fn print_table(records: &[Record]) {
    println!(
        "{:>3} {:>10} {:>10} {:>10} {:>10} {:>3} {:>6}  schedule",
        "#", "median", "p25", "p75", "min", "n", "drift"
    );
    for (index, r) in records.iter().enumerate() {
        // The check needs two disjoint windows of rounds; "-" means it did not run.
        let drift = match (r.samples >= MIN_ROUNDS_FOR_DRIFT, r.drift) {
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
            r.stats.median, r.stats.p25, r.stats.p75, r.stats.min, r.samples, drift
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
