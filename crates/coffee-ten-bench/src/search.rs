//! Random search over the matmul schedule space: the research baseline (spec §7.1).
//!
//! Samples `budget` distinct candidates of `matmul_space(n, n, n)` uniformly without
//! replacement and measures them one after another. The search is sequential by nature:
//! each candidate is compiled, checked and timed alone (`run::run` with one variant) before
//! the next one is measured, so the variants are not interleaved as in the harness. The
//! naive kernel is measured first as the reference and is not counted in the budget.

use crate::BenchError;
use crate::cli::{Config, DEFAULT_SEED, KernelKind, MAX_N, QUICK_MIN_SAMPLE_MS, QUICK_SAMPLES};
use crate::log::append_lines;
use crate::meta::Metadata;
use crate::run::{Record, record_json, run};
use crate::space::{Candidate, matmul_space};
use crate::stats::Rng;
use coffee_ten_loop::Schedule;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::str::FromStr;

/// Samples and minimum sample time when not given: a search measures many candidates, so it
/// takes fewer samples per candidate than the harness.
pub const DEFAULT_SAMPLES: usize = 5;
pub const DEFAULT_MIN_SAMPLE_MS: u64 = 10;
pub const SEARCH_USAGE: &str = "usage: coffee-ten-bench search --n <N> --budget <M> \
[--seed <SEED>] [--samples <S>] [--min-sample-ms <MS>] [--log <path>] [--quick]";

/// What a search measures; the same for every candidate.
#[derive(Clone, Debug, PartialEq)]
pub struct SearchConfig {
    /// Matrix size.
    pub n: usize,
    /// Number of candidates to measure, not counting the naive reference.
    pub budget: usize,
    pub seed: u64,
    pub samples: usize,
    pub min_sample_ms: u64,
    /// The run was `--quick`: not a measurement.
    pub quick: bool,
    /// JSONL file; each record is appended as soon as it is measured.
    pub log: Option<PathBuf>,
}

/// What a search measured.
#[derive(Clone, Debug, PartialEq)]
pub struct Search {
    /// Number of candidates in the whole space.
    pub space_size: usize,
    /// The naive kernel, the reference. It is not a candidate and not in the curve.
    pub naive: Record,
    /// The sampled candidates, in the order they were measured.
    pub records: Vec<Record>,
}

/// `budget` distinct candidates, uniform without replacement: a seeded shuffle of the whole
/// space, then its first `budget` entries. A budget at least the size gives the whole space
/// in shuffled order. Deterministic per seed.
pub fn sample(space: Vec<Candidate>, budget: usize, seed: u64) -> Vec<Candidate> {
    let mut pool = space;
    Rng::new(seed).shuffle(&mut pool);
    pool.truncate(budget);
    pool
}

/// Compiles, checks and times one schedule alone, through the harness's `run`.
fn measure(config: &SearchConfig, text: &str) -> Result<Record, BenchError> {
    let schedule = Schedule::parse(text).expect("invariant: candidate texts parse");
    let run_config = Config {
        kernel: KernelKind::Matmul,
        n: config.n,
        variants: vec![(text.to_string(), schedule)],
        samples: config.samples,
        min_sample_ms: config.min_sample_ms,
        seed: config.seed,
        quick: config.quick,
        log: None,
    };
    let mut records = run(&run_config)?;
    Ok(records
        .pop()
        .expect("invariant: run gives one record per variant"))
}

/// `measure`, then appends the record to the log if there is one.
fn measure_and_log(
    config: &SearchConfig,
    metadata: &Metadata,
    text: &str,
) -> Result<Record, BenchError> {
    let record = measure(config, text)?;
    if let Some(path) = &config.log {
        append_lines(path, &[record_json(&record, metadata)])
            .map_err(|e| BenchError::Setup(format!("cannot append to {}: {e}", path.display())))?;
    }
    Ok(record)
}

/// Measures the naive kernel, then the sampled candidates one after another. A candidate
/// that fails the check stops the search with its error; the records logged before it stay
/// in the log.
pub fn search(config: &SearchConfig, metadata: &Metadata) -> Result<Search, BenchError> {
    let space = matmul_space(config.n, config.n, config.n);
    let space_size = space.len();
    let naive = measure_and_log(config, metadata, "")?;
    let mut records = Vec::new();
    for candidate in sample(space, config.budget, config.seed) {
        records.push(measure_and_log(config, metadata, &candidate.schedule)?);
    }
    Ok(Search {
        space_size,
        naive,
        records,
    })
}

/// Index of the best median among the first `m` values, for m = 1..=len. A tie keeps the
/// earlier index.
pub fn best_index_so_far(medians: &[f64]) -> Vec<usize> {
    let mut best: Vec<usize> = Vec::with_capacity(medians.len());
    for (index, &median) in medians.iter().enumerate() {
        let keep = match best.last() {
            Some(&previous) if medians[previous] >= median => previous,
            _ => index,
        };
        best.push(keep);
    }
    best
}

/// The m at which the curve is printed: 1, 2, 4, ... below `measured`, then `measured`.
pub fn curve_points(measured: usize) -> Vec<usize> {
    let mut points: Vec<usize> = std::iter::successors(Some(1usize), |m| m.checked_mul(2))
        .take_while(|&m| m < measured)
        .collect();
    if measured > 0 {
        points.push(measured);
    }
    points
}

/// The report of `coffee-ten-bench search`: the reference, then the best-so-far curve as
/// `m  best_gflops  best_schedule`.
pub fn report(config: &SearchConfig, result: &Search, metadata: &Metadata) -> String {
    let mut text = String::new();
    let measured = result.records.len();
    if config.quick {
        let _ = writeln!(
            text,
            "quick: true ({} samples, {} ms, not a measurement)",
            config.samples, config.min_sample_ms
        );
    }
    let _ = writeln!(
        text,
        "matmul {n}x{n}x{n}  budget {}  seed {}  samples {}  min-sample {} ms",
        config.budget,
        config.seed,
        config.samples,
        config.min_sample_ms,
        n = config.n
    );
    let dirty = match metadata.git_dirty {
        Some(true) => " (dirty tree)",
        Some(false) => "",
        None => " (dirty: unknown)",
    };
    let _ = writeln!(
        text,
        "cpu {}  git {}{dirty}",
        metadata.cpu_brand, metadata.git_rev
    );
    let _ = writeln!(
        text,
        "space {} candidates, measured {measured}",
        result.space_size
    );
    let _ = writeln!(
        text,
        "naive {:.3} GFLOP/s (the reference; not a candidate, not in the curve)",
        result.naive.stats.median
    );
    let _ = writeln!(text, "{:>4}  {:>11}  best_schedule", "m", "best_gflops");
    let medians: Vec<f64> = result.records.iter().map(|r| r.stats.median).collect();
    let best = best_index_so_far(&medians);
    for m in curve_points(measured) {
        let record = &result.records[best[m - 1]];
        let _ = writeln!(
            text,
            "{m:>4}  {:>11.3}  {}",
            record.stats.median, record.schedule
        );
    }
    text
}

fn parse_number<T: FromStr>(flag: &str, text: &str) -> Result<T, BenchError> {
    text.parse().map_err(|_| {
        BenchError::Usage(format!(
            "{flag} needs a non-negative integer, got {text:?}\n{SEARCH_USAGE}"
        ))
    })
}

/// Parses the arguments after `search`. `--quick` replaces the samples and minimum sample
/// time with the quick values.
pub fn parse_search_args<I: IntoIterator<Item = String>>(
    args: I,
) -> Result<SearchConfig, BenchError> {
    let usage = |msg: String| BenchError::Usage(format!("{msg}\n{SEARCH_USAGE}"));
    let mut args = args.into_iter();
    let mut n = None;
    let mut budget = None;
    let mut seed = DEFAULT_SEED;
    let mut samples = DEFAULT_SAMPLES;
    let mut min_sample_ms = DEFAULT_MIN_SAMPLE_MS;
    let mut quick = false;
    let mut log = None;
    while let Some(flag) = args.next() {
        if flag == "--quick" {
            quick = true;
            continue;
        }
        let value = args
            .next()
            .ok_or_else(|| usage(format!("{flag} needs a value")))?;
        match flag.as_str() {
            "--n" => n = Some(parse_number::<usize>(&flag, &value)?),
            "--budget" => budget = Some(parse_number::<usize>(&flag, &value)?),
            "--seed" => seed = parse_number(&flag, &value)?,
            "--samples" => samples = parse_number(&flag, &value)?,
            "--min-sample-ms" => min_sample_ms = parse_number(&flag, &value)?,
            "--log" => log = Some(PathBuf::from(value)),
            _ => return Err(usage(format!("unknown argument {flag}"))),
        }
    }
    let n = n.ok_or_else(|| usage("--n is required".to_string()))?;
    if !(1..=MAX_N).contains(&n) {
        return Err(usage(format!("--n must be in 1..={MAX_N}, got {n}")));
    }
    let budget = budget.ok_or_else(|| usage("--budget is required".to_string()))?;
    if budget == 0 {
        return Err(usage("--budget must be at least 1".to_string()));
    }
    if quick {
        samples = QUICK_SAMPLES;
        min_sample_ms = QUICK_MIN_SAMPLE_MS;
    }
    if samples == 0 {
        return Err(usage("--samples must be at least 1".to_string()));
    }
    Ok(SearchConfig {
        n,
        budget,
        seed,
        samples,
        min_sample_ms,
        quick,
        log,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::meta::UNKNOWN;
    use std::collections::HashSet;
    use std::fs;

    fn texts(candidates: &[Candidate]) -> Vec<String> {
        candidates.iter().map(|c| c.schedule.clone()).collect()
    }

    #[test]
    fn sample_is_distinct_and_comes_from_the_space() {
        let space = matmul_space(16, 16, 16);
        let all: HashSet<String> = texts(&space).into_iter().collect();
        for seed in 1..=5 {
            // 100 of 128: sampling with replacement repeats a candidate almost surely.
            let picked = texts(&sample(space.clone(), 100, seed));
            let distinct: HashSet<String> = picked.iter().cloned().collect();
            assert_eq!(picked.len(), 100);
            assert_eq!(distinct.len(), 100, "seed {seed}");
            assert!(distinct.is_subset(&all));
        }
    }

    #[test]
    fn sample_is_deterministic_per_seed_and_differs_between_seeds() {
        let space = matmul_space(16, 16, 16);
        let first = texts(&sample(space.clone(), 10, 1));
        assert_eq!(first, texts(&sample(space.clone(), 10, 1)));
        assert_ne!(first, texts(&sample(space.clone(), 10, 2)));
    }

    #[test]
    fn a_budget_of_the_space_size_or_more_is_the_whole_space_shuffled() {
        let space = matmul_space(16, 16, 16);
        let size = space.len();
        assert_eq!(size, 128);
        let mut sorted_space = texts(&space);
        sorted_space.sort();
        for budget in [size, size + 1] {
            let picked = texts(&sample(space.clone(), budget, 1));
            let mut sorted_picked = picked.clone();
            sorted_picked.sort();
            assert_eq!(sorted_picked, sorted_space, "budget {budget}");
            assert_ne!(picked, texts(&space), "budget {budget} is not shuffled");
        }
    }

    #[test]
    fn best_so_far_is_the_running_maximum_with_ties_to_the_earlier() {
        let mut rng = Rng::new(5);
        for _ in 0..200 {
            let len = 1 + (rng.next_u64() % 40) as usize;
            // Multiples of 0.1 in 0..100: ties are common.
            let medians: Vec<f64> = (0..len)
                .map(|_| (rng.next_u64() % 1000) as f64 / 10.0)
                .collect();
            let best = best_index_so_far(&medians);
            assert_eq!(best.len(), len);
            let mut previous = f64::NEG_INFINITY;
            for m in 0..len {
                let running = medians[..=m]
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max);
                assert_eq!(medians[best[m]], running, "m = {}", m + 1);
                let first_of_max = medians[..=m].iter().position(|&x| x == running);
                assert_eq!(Some(best[m]), first_of_max, "m = {}", m + 1);
                assert!(medians[best[m]] >= previous, "m = {}", m + 1);
                previous = medians[best[m]];
            }
        }
    }

    #[test]
    fn curve_points_are_powers_of_two_then_the_count() {
        assert_eq!(curve_points(0), Vec::<usize>::new());
        assert_eq!(curve_points(1), [1]);
        assert_eq!(curve_points(3), [1, 2, 3]);
        assert_eq!(curve_points(4), [1, 2, 4]);
        assert_eq!(curve_points(50), [1, 2, 4, 8, 16, 32, 50]);
    }

    fn parse(args: &[&str]) -> Result<SearchConfig, BenchError> {
        parse_search_args(args.iter().map(|a| a.to_string()))
    }

    #[test]
    fn search_args_have_defaults_and_quick_overrides_samples() {
        let config = parse(&["--n", "256", "--budget", "50"]).unwrap();
        assert_eq!(
            config,
            SearchConfig {
                n: 256,
                budget: 50,
                seed: 1,
                samples: 5,
                min_sample_ms: 10,
                quick: false,
                log: None,
            }
        );
        let quick = parse(&["--n", "16", "--budget", "3", "--samples", "50", "--quick"]).unwrap();
        assert!(quick.quick);
        assert_eq!(quick.samples, QUICK_SAMPLES);
        assert_eq!(quick.min_sample_ms, QUICK_MIN_SAMPLE_MS);
    }

    #[test]
    fn search_args_limits_are_checked_on_both_sides() {
        let with = |n: &str, budget: &str| parse(&["--n", n, "--budget", budget]);
        assert!(with("0", "1").is_err());
        assert!(with("1", "1").is_ok());
        assert!(with(&MAX_N.to_string(), "1").is_ok());
        assert!(with(&(MAX_N + 1).to_string(), "1").is_err());
        assert!(with("16", "0").is_err());
        assert!(with("16", "1").is_ok());
        assert!(parse(&["--n", "16", "--budget", "3", "--samples", "0"]).is_err());
    }

    #[test]
    fn missing_and_unknown_search_arguments_are_errors() {
        assert!(parse(&["--n", "16"]).is_err());
        assert!(parse(&["--budget", "3"]).is_err());
        assert!(parse(&["--n", "x", "--budget", "3"]).is_err());
        assert!(parse(&["--n", "16", "--budget", "3", "--bogus", "1"]).is_err());
        assert!(parse(&["--n", "16", "--budget"]).is_err());
    }

    #[test]
    fn quick_search_on_16_measures_naive_and_three_candidates() {
        let path = std::env::temp_dir().join(format!(
            "coffee-ten-bench-search-{}.jsonl",
            std::process::id()
        ));
        let _ = fs::remove_file(&path);
        let config = SearchConfig {
            n: 16,
            budget: 3,
            seed: 1,
            samples: QUICK_SAMPLES,
            min_sample_ms: QUICK_MIN_SAMPLE_MS,
            quick: true,
            log: Some(path.clone()),
        };
        let metadata = Metadata {
            hw_model: UNKNOWN.to_string(),
            cpu_brand: UNKNOWN.to_string(),
            os_version: UNKNOWN.to_string(),
            clang: UNKNOWN.to_string(),
            flags: UNKNOWN.to_string(),
            git_rev: UNKNOWN.to_string(),
            git_dirty: None,
            date_utc: UNKNOWN.to_string(),
            seed: config.seed,
        };
        // `run` aborts with `BenchError::Mismatch` before timing a wrong variant, so an `Ok`
        // means every measured variant passed the check.
        let result = search(&config, &metadata).unwrap();
        let log_text = fs::read_to_string(&path).unwrap();
        fs::remove_file(&path).unwrap();

        assert_eq!(result.space_size, 128);
        assert_eq!(result.naive.schedule, "");
        assert_eq!(result.records.len(), 3);
        for record in std::iter::once(&result.naive).chain(&result.records) {
            assert!(record.verified && record.quick, "{record:?}");
            assert_eq!(record.samples, QUICK_SAMPLES);
        }
        // Naive first, then the three candidates: four lines.
        let lines: Vec<&str> = log_text.lines().collect();
        assert_eq!(lines.len(), 4);
        assert!(lines[0].contains("\"schedule\":\"\""), "{}", lines[0]);

        let text = report(&config, &result, &metadata);
        let rows: Vec<&str> = text
            .lines()
            .skip_while(|line| !line.contains("best_schedule"))
            .skip(1)
            .collect();
        let ms: Vec<&str> = rows
            .iter()
            .map(|row| row.split_whitespace().next().unwrap())
            .collect();
        assert_eq!(ms, ["1", "2", "3"], "{text}");
    }
}
