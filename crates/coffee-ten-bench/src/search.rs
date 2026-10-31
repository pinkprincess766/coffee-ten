//! Random search over the matmul schedule space: the research baseline (spec §7.1).
//!
//! Samples `budget` distinct candidates of `matmul_space(n, n, n)` uniformly without
//! replacement and measures them one after another. All candidates of a group are compiled
//! and checked first, in parallel, and then timed one at a time with no compilation in
//! between, each alone (one variant per `Reference::time`), so the variants are not
//! interleaved as in the harness. The naive kernel is measured first as the reference and is
//! not counted in the budget.
//!
//! With `--model-top K --from <results.jsonl>` it is also the experiment of spec §7.1: after
//! the random candidates it measures the analytic model's top K schedules the same way, then
//! runs the best random and the best model-top schedule together in one interleaved `run`,
//! and decides on that run alone.
//!
//! With `--local --from <results.jsonl>` the budget is spent on a hill climb instead of the
//! random sample: the third curve of spec §7.1. The model picks the next neighbor; a
//! measurement becomes the incumbent only when it is strictly faster. `--local` and
//! `--model-top` are separate experiments.
//!
//! Every log line is a `record_json` line plus a last field `"group"`: `"naive"` (the
//! reference), `"random"` (the sample), `"model"` (the model's top K, rank order),
//! `"local"` (the hill climb) or `"final"` (the interleaved decision run: best random,
//! then best model).

use crate::BenchError;
use crate::cli::{Config, DEFAULT_SEED, KernelKind, MAX_N, QUICK_MIN_SAMPLE_MS, QUICK_SAMPLES};
use crate::local::local_search;
use crate::log::append_lines;
use crate::meta::Metadata;
use crate::model::{Machine, Ranked, machine_from_results, rank_matmul, spearman};
use crate::run::{Record, Reference, candidate_warmup, full_warmup, record_json, run};
use crate::space::{Candidate, matmul_space, neighbors};
use crate::stats::{Rng, Stats, summarize};
use coffee_ten_loop::Schedule;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::str::FromStr;

/// Samples and minimum sample time when not given: a search measures many candidates, so it
/// takes fewer samples per candidate than the harness.
pub const DEFAULT_SAMPLES: usize = 5;
pub const DEFAULT_MIN_SAMPLE_MS: u64 = 10;
pub const SEARCH_USAGE: &str = "usage: coffee-ten-bench search --n <N> --budget <M> \
[--seed <SEED>] [--samples <S>] [--min-sample-ms <MS>] [--log <path>] [--quick] \
[--model-top <K> --from <results.jsonl>] [--local --from <results.jsonl>]";
/// Spec §7.1: the model's best must reach this fraction of the best random schedule.
pub const CRITERION_RATIO: f64 = 0.9;

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
    /// Also measure the model's top K schedules (K >= 1; cut to the space size).
    pub model_top: Option<usize>,
    /// Results file whose peak line is the model's machine. Set exactly with `model_top` or
    /// with `local`.
    pub from: Option<PathBuf>,
    /// Hill climb on the model's neighbors instead of the random sample.
    pub local: bool,
}

/// What a search measured.
#[derive(Clone, Debug, PartialEq)]
pub struct Search {
    /// Number of candidates in the whole space.
    pub space_size: usize,
    /// The naive kernel, the reference. It is not a candidate and not in the curve.
    pub naive: Record,
    /// The sampled candidates, in the order they were measured. With `--local`, the hill-climb
    /// measurements instead of the random sample.
    pub records: Vec<Record>,
    /// The model group; `None` without `--model-top`.
    pub model: Option<ModelGroup>,
}

/// The measurements of the model's top K and the decision run.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelGroup {
    pub machine: Machine,
    /// The model's predicted GFLOP/s of each of `Search::records`, same order.
    pub random_predicted: Vec<f64>,
    /// The model's top K in rank order, measured alone like the random ones. A schedule that
    /// is also in the random sample is measured again here.
    pub records: Vec<Record>,
    /// Best random and best model-top measured together; `None` if they are the same schedule.
    pub final_run: Option<FinalRun>,
}

/// The interleaved decision run (`"group":"final"`).
#[derive(Clone, Debug, PartialEq)]
pub struct FinalRun {
    pub random: Record,
    pub model: Record,
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

/// The harness config of the search's candidates; `variants` is filled per use.
fn bench_config(config: &SearchConfig) -> Config {
    Config {
        kernel: KernelKind::Matmul,
        n: config.n,
        variants: Vec::new(),
        samples: config.samples,
        min_sample_ms: config.min_sample_ms,
        seed: config.seed,
        quick: config.quick,
        log: None,
    }
}

/// Compiles and checks all `texts` first, in parallel and before any timing, then times them
/// one after another, each alone, and logs each record as soon as it is timed. No compilation
/// happens between two timings, so the thread stays busy: only the first candidate gets the
/// full warm-up, the others the short one (`run::CANDIDATE_WARMUP_MS`). A candidate that fails
/// to compile or fails the check stops the search with its error after the earlier ones are
/// logged.
fn measure_group(
    config: &SearchConfig,
    metadata: &Metadata,
    reference: &Reference,
    group: &str,
    texts: &[&str],
) -> Result<Vec<Record>, BenchError> {
    let candidates: Vec<(String, Schedule)> = texts
        .iter()
        .map(|&text| {
            let schedule = Schedule::parse(text).expect("invariant: candidate texts parse");
            (text.to_string(), schedule)
        })
        .collect();
    let bench = bench_config(config);
    let mut records = Vec::with_capacity(texts.len());
    for (index, built) in reference.build_all(&candidates).into_iter().enumerate() {
        let warmup = if index == 0 {
            full_warmup(config.quick)
        } else {
            candidate_warmup(config.quick)
        };
        let mut variant = [built?];
        let mut timed = reference.time(&bench, &mut variant, warmup)?;
        let record = timed
            .pop()
            .expect("invariant: time gives one record per variant");
        log_records(config, metadata, group, &[&record])?;
        records.push(record);
    }
    Ok(records)
}

/// The log line of `record`: `record_json` plus the last field `"group"`.
fn group_json(record: &Record, metadata: &Metadata, group: &str) -> String {
    let line = record_json(record, metadata);
    let body = line
        .strip_suffix('}')
        .expect("invariant: record_json gives an object");
    format!("{body},\"group\":\"{group}\"}}")
}

/// Appends the records to the log if there is one.
fn log_records(
    config: &SearchConfig,
    metadata: &Metadata,
    group: &str,
    records: &[&Record],
) -> Result<(), BenchError> {
    let Some(path) = &config.log else {
        return Ok(());
    };
    let lines: Vec<String> = records
        .iter()
        .map(|record| group_json(record, metadata, group))
        .collect();
    append_lines(path, &lines)
        .map_err(|e| BenchError::Setup(format!("cannot append to {}: {e}", path.display())))
}

/// One schedule alone, with the full warm-up: `measure_group` of one text.
fn measure_and_log(
    config: &SearchConfig,
    metadata: &Metadata,
    reference: &Reference,
    group: &str,
    text: &str,
) -> Result<Record, BenchError> {
    let mut records = measure_group(config, metadata, reference, group, &[text])?;
    Ok(records.pop().expect("invariant: one record per candidate"))
}

/// Index of the highest median among `records`; a tie keeps the earlier.
fn best_of(records: &[Record]) -> usize {
    let medians: Vec<f64> = records.iter().map(|r| r.stats.median).collect();
    *best_index_so_far(&medians)
        .last()
        .expect("invariant: at least one record")
}

/// `random` and `model` together in one interleaved `run`, as the harness measures variants.
fn measure_final(
    config: &SearchConfig,
    metadata: &Metadata,
    random: &Record,
    model: &Record,
) -> Result<FinalRun, BenchError> {
    let variants = [random, model]
        .iter()
        .map(|record| {
            let schedule =
                Schedule::parse(&record.schedule).expect("invariant: record schedules parse");
            (record.schedule.clone(), schedule)
        })
        .collect();
    // The decision run uses the harness's rounds (as in BENCHMARKS.md), not the search's
    // fewer samples per candidate; `--quick` keeps its own.
    let samples = if config.quick {
        config.samples
    } else {
        crate::cli::DEFAULT_SAMPLES
    };
    let run_config = Config {
        kernel: KernelKind::Matmul,
        n: config.n,
        variants,
        samples,
        min_sample_ms: config.min_sample_ms,
        seed: config.seed,
        quick: config.quick,
        log: None,
    };
    let mut records = run(&run_config)?.into_iter();
    let (Some(random), Some(model)) = (records.next(), records.next()) else {
        return Err(BenchError::Setup("the final run lost a record".to_string()));
    };
    log_records(config, metadata, "final", &[&random, &model])?;
    Ok(FinalRun { random, model })
}

/// The hill climb of `--local`: `budget` measurements, group `"local"`.
fn measure_local(
    config: &SearchConfig,
    metadata: &Metadata,
    reference: &Reference,
    space: &[Candidate],
    ranked: &[Ranked],
) -> Result<Vec<Record>, BenchError> {
    if ranked.len() != space.len() {
        return Err(BenchError::Setup(
            "the model did not rank the whole space".to_string(),
        ));
    }
    let index_of: HashMap<&str, usize> = space
        .iter()
        .enumerate()
        .map(|(index, candidate)| (candidate.schedule.as_str(), index))
        .collect();
    let mut rank_of = vec![usize::MAX; space.len()];
    for (rank, entry) in ranked.iter().enumerate() {
        let Some(&index) = index_of.get(entry.schedule.as_str()) else {
            return Err(BenchError::Setup(format!(
                "the model ranked a schedule that is not in the space: {}",
                entry.schedule
            )));
        };
        if rank_of[index] != usize::MAX {
            return Err(BenchError::Setup(
                "the model ranked a schedule twice".to_string(),
            ));
        }
        rank_of[index] = rank;
    }
    if rank_of.contains(&usize::MAX) {
        return Err(BenchError::Setup(
            "the model did not rank the whole space".to_string(),
        ));
    }
    let steps = neighbors(space, config.n, config.n, config.n);
    let mut records = Vec::new();
    local_search(&steps, &rank_of, config.budget, |index| {
        // Each step depends on the last median, so the hill climb cannot compile ahead.
        let record = measure_and_log(config, metadata, reference, "local", &space[index].schedule)?;
        let median = record.stats.median;
        records.push(record);
        Ok(median)
    })?;
    Ok(records)
}

/// Measures the naive kernel, then the sampled candidates one after another; with
/// `--model-top` then the model's top K and the final run. `--local` measures the hill
/// climb instead of the sample. A candidate that fails the check stops the search with
/// its error; the records logged before it stay in the log.
pub fn search(config: &SearchConfig, metadata: &Metadata) -> Result<Search, BenchError> {
    let space = matmul_space(config.n, config.n, config.n);
    let space_size = space.len();
    // Ranked before anything is measured: a bad `--from` fails at once.
    let model_input = match &config.from {
        Some(path) if config.model_top.is_some() || config.local => {
            let text = std::fs::read_to_string(path)
                .map_err(|e| BenchError::Setup(format!("cannot read {}: {e}", path.display())))?;
            // Only `.0`: the rest of the tuple is about which ceilings were measured.
            let machine = machine_from_results(&text)?.0;
            let ranked = rank_matmul(&machine, config.n)?;
            Some((machine, ranked))
        }
        _ => None,
    };
    // The naive output is computed once for the whole search.
    let reference = Reference::new(&bench_config(config))?;
    let naive = measure_and_log(config, metadata, &reference, "naive", "")?;
    if config.local {
        let (_, ranked) = model_input.expect("invariant: --local ranks before measuring");
        let records = measure_local(config, metadata, &reference, &space, &ranked)?;
        return Ok(Search {
            space_size,
            naive,
            records,
            model: None,
        });
    }
    let sampled = sample(space, config.budget, config.seed);
    let texts: Vec<&str> = sampled.iter().map(|c| c.schedule.as_str()).collect();
    let records = measure_group(config, metadata, &reference, "random", &texts)?;
    let mut model = None;
    if let Some((machine, ranked)) = model_input {
        let top = config
            .model_top
            .expect("invariant: the model group has --model-top");
        let predicted: HashMap<&str, f64> = ranked
            .iter()
            .map(|r| (r.schedule.as_str(), r.prediction.gflops))
            .collect();
        let random_predicted = records
            .iter()
            .map(|r| {
                predicted
                    .get(r.schedule.as_str())
                    .copied()
                    .expect("invariant: the ranking covers the space, texts are canonical")
            })
            .collect();
        let texts: Vec<&str> = ranked
            .iter()
            .take(top)
            .map(|entry| entry.schedule.as_str())
            .collect();
        let model_records = measure_group(config, metadata, &reference, "model", &texts)?;
        let best_random = &records[best_of(&records)];
        let best_model = &model_records[best_of(&model_records)];
        let final_run = if best_random.schedule == best_model.schedule {
            None
        } else {
            Some(measure_final(config, metadata, best_random, best_model)?)
        };
        model = Some(ModelGroup {
            machine,
            random_predicted,
            records: model_records,
            final_run,
        });
    }
    Ok(Search {
        space_size,
        naive,
        records,
        model,
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
    if config.local {
        let from = config
            .from
            .as_ref()
            .map_or(String::new(), |path| path.display().to_string());
        let _ = writeln!(
            text,
            "local search (spec 7.1): hill climb on one model step, restart from the next model \
             rank when that neighborhood is measured; machine from {from}"
        );
    }
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
    if let Some(model) = &result.model {
        write_model_section(&mut text, config, result, model);
    }
    text
}

fn stats_text(stats: &Stats) -> String {
    format!(
        "median {:.3} [p25 {:.3}, p75 {:.3}] GFLOP/s",
        stats.median, stats.p25, stats.p75
    )
}

/// The part of the report that is about `--model-top`: the model's curve, the best of each
/// group, the decision (from the final run alone) and how well the model predicts.
fn write_model_section(
    text: &mut String,
    config: &SearchConfig,
    result: &Search,
    model: &ModelGroup,
) {
    let machine = &model.machine;
    let _ = writeln!(
        text,
        "\nmodel: top {} of the analytic model (PREDICTION), machine from {}: peak {} GFLOP/s, \
         mem {} GB/s, L2 {} GB/s, mem_acc {} GFLOP/s",
        model.records.len(),
        config
            .from
            .as_ref()
            .map_or(String::new(), |p| p.display().to_string()),
        machine.peak_gflops,
        machine.mem_gbs,
        machine.l2_gbs,
        machine.mem_acc_gflops
    );
    let _ = writeln!(
        text,
        "a model-top schedule that was also in the random sample was measured again"
    );
    let medians: Vec<f64> = model.records.iter().map(|r| r.stats.median).collect();
    let best = best_index_so_far(&medians);
    let _ = writeln!(
        text,
        "{:>4}  {:>11}  best_schedule (model order, sequential medians)",
        "m", "best_gflops"
    );
    for m in curve_points(model.records.len()) {
        let record = &model.records[best[m - 1]];
        let _ = writeln!(
            text,
            "{m:>4}  {:>11.3}  {}",
            record.stats.median, record.schedule
        );
    }
    let best_random_index = best_of(&result.records);
    let best_model_index = best_of(&model.records);
    let best_random = &result.records[best_random_index];
    let best_model = &model.records[best_model_index];
    let _ = writeln!(
        text,
        "best random (m = {}): {}",
        best_random_index + 1,
        best_random.schedule
    );
    let _ = writeln!(
        text,
        "best model  (model rank {}): {}",
        best_model_index + 1,
        best_model.schedule
    );
    let _ = writeln!(
        text,
        "criterion (spec 7.1): the best model-top schedule reaches at least {:.0}% of the best \
         random schedule, compared in the final interleaved run",
        CRITERION_RATIO * 100.0
    );
    match &model.final_run {
        Some(final_run) => {
            let _ = writeln!(
                text,
                "final random  {}",
                stats_text(&final_run.random.stats)
            );
            let _ = writeln!(text, "final model   {}", stats_text(&final_run.model.stats));
            let ratio = final_run.model.stats.median / final_run.random.stats.median;
            let verdict = if ratio >= CRITERION_RATIO {
                "MET"
            } else {
                "NOT MET"
            };
            let _ = writeln!(
                text,
                "ratio = best-model median / best-random median = {ratio:.3}: criterion {verdict}"
            );
            let _ = writeln!(
                text,
                "(the curves use the sequential medians; the decision uses only the final run)"
            );
        }
        None => {
            let _ = writeln!(
                text,
                "best random and best model are the same schedule: no final run; ratio = 1.000: \
                 criterion MET"
            );
        }
    }
    let measured: Vec<f64> = result.records.iter().map(|r| r.stats.median).collect();
    let abs_log_ratios: Vec<f64> = model
        .random_predicted
        .iter()
        .zip(&measured)
        .map(|(p, m)| (p / m).ln().abs())
        .collect();
    let _ = write!(
        text,
        "model vs measured over the {} random candidates: ",
        measured.len()
    );
    match (
        spearman(&model.random_predicted, &measured),
        summarize(&abs_log_ratios),
    ) {
        (Some(rho), Some(stats)) => {
            let _ = writeln!(
                text,
                "Spearman rank correlation {rho:.3}; median |ln(predicted/measured)| {:.3} \
                 (a factor {:.2})",
                stats.median,
                stats.median.exp()
            );
        }
        _ => {
            let _ = writeln!(
                text,
                "Spearman rank correlation: undefined (too few or constant values)"
            );
        }
    }
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
    let mut model_top = None;
    let mut from = None;
    let mut local = false;
    while let Some(flag) = args.next() {
        if flag == "--quick" {
            quick = true;
            continue;
        }
        if flag == "--local" {
            local = true;
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
            "--model-top" => model_top = Some(parse_number::<usize>(&flag, &value)?),
            "--from" => from = Some(PathBuf::from(value)),
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
    if model_top == Some(0) {
        return Err(usage("--model-top must be at least 1".to_string()));
    }
    if local && model_top.is_some() {
        return Err(usage(
            "--local and --model-top are different experiments".to_string(),
        ));
    }
    if local && from.is_none() {
        return Err(usage(
            "--local needs --from (the machine of the model)".to_string(),
        ));
    }
    if !local && model_top.is_some() != from.is_some() {
        return Err(usage(
            "--model-top and --from (the machine of the model) go together".to_string(),
        ));
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
        model_top,
        from,
        local,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::meta::UNKNOWN;
    use crate::run::{RunInfo, make_record};
    use coffee_ten_loop::OrderClass;
    use std::collections::{HashMap, HashSet};
    use std::fs;

    fn test_metadata(seed: u64) -> Metadata {
        Metadata {
            hw_model: UNKNOWN.to_string(),
            cpu_brand: UNKNOWN.to_string(),
            os_version: UNKNOWN.to_string(),
            clang: UNKNOWN.to_string(),
            flags: UNKNOWN.to_string(),
            git_rev: UNKNOWN.to_string(),
            git_dirty: None,
            date_utc: UNKNOWN.to_string(),
            seed,
        }
    }

    /// Naive references built on this test's thread so far.
    fn references_built() -> usize {
        crate::run::REFERENCES_BUILT.with(std::cell::Cell::get)
    }

    fn texts(candidates: &[Candidate]) -> Vec<String> {
        candidates.iter().map(|c| c.schedule.clone()).collect()
    }

    #[test]
    fn sample_is_distinct_and_comes_from_the_space() {
        let space = matmul_space(16, 16, 16);
        let all: HashSet<String> = texts(&space).into_iter().collect();
        for seed in 1..=5 {
            // 100 of 133: sampling with replacement repeats a candidate almost surely.
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
        assert_eq!(size, 133);
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
                model_top: None,
                from: None,
                local: false,
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
            model_top: None,
            from: None,
            local: false,
        };
        let metadata = test_metadata(config.seed);
        // `run` aborts with `BenchError::Mismatch` before timing a wrong variant, so an `Ok`
        // means every measured variant passed the check.
        let result = search(&config, &metadata).unwrap();
        // The naive reference for the check: once for the naive record and 3 candidates.
        assert_eq!(references_built(), 1);
        let log_text = fs::read_to_string(&path).unwrap();
        fs::remove_file(&path).unwrap();

        assert_eq!(result.space_size, 133);
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
        assert!(!text.contains("criterion"), "{text}");
        assert!(lines[0].ends_with(",\"group\":\"naive\"}"), "{}", lines[0]);
        for line in &lines[1..] {
            assert!(line.ends_with(",\"group\":\"random\"}"), "{line}");
        }
    }

    #[test]
    fn a_failing_candidate_stops_the_group_after_the_earlier_ones_are_logged() {
        let log = std::env::temp_dir().join(format!(
            "coffee-ten-bench-search-failing-{}.jsonl",
            std::process::id()
        ));
        let _ = fs::remove_file(&log);
        let config = SearchConfig {
            n: 8,
            budget: 3,
            seed: 1,
            samples: QUICK_SAMPLES,
            min_sample_ms: QUICK_MIN_SAMPLE_MS,
            quick: true,
            log: Some(log.clone()),
            model_top: None,
            from: None,
            local: false,
        };
        let reference = Reference::new(&bench_config(&config)).unwrap();
        // The third one is fine, but it comes after the failing one.
        let texts = [
            "vectorize(1, j)",
            "vectorize(1, nope)",
            "split(1, j, jo, ji, 4)",
        ];
        let err =
            measure_group(&config, &test_metadata(1), &reference, "random", &texts).unwrap_err();
        let message = err.to_string();
        let log_text = fs::read_to_string(&log).unwrap();
        fs::remove_file(&log).unwrap();
        assert!(message.contains("cannot apply schedule"), "{message}");
        assert!(message.contains("vectorize(1, nope)"), "{message}");
        assert_eq!(log_text.lines().count(), 1, "{log_text}");
        assert!(
            log_text.contains("\"schedule\":\"vectorize(1, j)\""),
            "{log_text}"
        );
    }

    #[test]
    fn model_top_needs_from_and_a_positive_k() {
        let base = ["--n", "16", "--budget", "3"];
        let with = |extra: &[&str]| parse(&[&base[..], extra].concat());
        let config = with(&["--model-top", "10", "--from", "peak.jsonl"]).unwrap();
        assert_eq!(config.model_top, Some(10));
        assert_eq!(config.from, Some(PathBuf::from("peak.jsonl")));
        assert!(with(&["--model-top", "1", "--from", "p"]).is_ok());
        assert!(with(&["--model-top", "0", "--from", "p"]).is_err());
        assert!(with(&["--model-top", "x", "--from", "p"]).is_err());
        assert!(with(&["--model-top", "3"]).is_err());
        assert!(with(&["--from", "p"]).is_err());
        assert!(with(&["--model-top"]).is_err());
        assert!(with(&["--local"]).is_err());
        assert!(with(&["--local", "--from", "p"]).is_ok());
        assert!(with(&["--local", "--model-top", "2", "--from", "p"]).is_err());
        let local = with(&["--local", "--from", "peak.jsonl"]).unwrap();
        assert!(local.local);
        assert_eq!(local.from, Some(PathBuf::from("peak.jsonl")));
        assert_eq!(local.model_top, None);
    }

    #[test]
    fn group_is_the_last_field_and_the_other_fields_stay() {
        let record = hand_record("vectorize(1, j)", [1.0, 2.0, 3.0]);
        let metadata = test_metadata(1);
        let plain = record_json(&record, &metadata);
        let line = group_json(&record, &metadata, "final");
        assert_eq!(
            line,
            format!("{},\"group\":\"final\"}}", plain.strip_suffix('}').unwrap())
        );
        assert!(
            line.starts_with("{\"kernel\":\"matmul\",\"n\":8,"),
            "{line}"
        );
    }

    const README: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../bench/README.md");

    /// `(field, writer)` of the rows of the record table in bench/README.md, in table order:
    /// the rows whose first cell is `` `field` `` and whose last cell is `harness`, `meta` or
    /// `search`. The rows of the peak table (writer `peak.c`) are not read.
    fn documented_fields() -> Vec<(String, String)> {
        let text = fs::read_to_string(README).unwrap();
        let mut rows = Vec::new();
        for line in text.lines().filter(|line| line.starts_with("| `")) {
            let cells: Vec<&str> = line.trim_matches('|').split('|').map(str::trim).collect();
            let writer = cells[cells.len() - 1];
            if ["harness", "meta", "search"].contains(&writer) {
                rows.push((cells[0].trim_matches('`').to_string(), writer.to_string()));
            }
        }
        rows
    }

    /// The keys of the top-level object of a one-line JSON text, in order. Strings in values
    /// are skipped, so text that looks like a key cannot count as one.
    fn top_level_keys(line: &str) -> Vec<String> {
        let mut keys = Vec::new();
        let (mut depth, mut string_start, mut escaped) = (0usize, None, false);
        for (i, c) in line.char_indices() {
            match string_start {
                Some(_) if escaped => escaped = false,
                Some(_) if c == '\\' => escaped = true,
                Some(start) if c == '"' => {
                    if depth == 1 && line[i + 1..].starts_with(':') {
                        keys.push(line[start + 1..i].to_string());
                    }
                    string_start = None;
                }
                Some(_) => {}
                None => match c {
                    '"' => string_start = Some(i),
                    '{' | '[' => depth += 1,
                    '}' | ']' => depth -= 1,
                    _ => {}
                },
            }
        }
        keys
    }

    #[test]
    fn top_level_keys_skips_nested_objects_and_key_like_text_in_values() {
        let line = r#"{"a":1,"b":{"x":2},"c":"q\":\"y\":3","d":[{"z":1}],"e":null}"#;
        assert_eq!(top_level_keys(line), ["a", "b", "c", "d", "e"]);
    }

    /// bench/README.md documents the fields of a log line. The harness line (`main.rs`:
    /// `record_json`, `append_lines`) and the search line (`log_records`) must have exactly
    /// the documented keys, in the documented order. Add a field to a writer and to the README
    /// table together, in the order written.
    #[test]
    fn log_lines_have_exactly_the_keys_of_the_readme_table_in_its_order() {
        let documented = documented_fields();
        let search_rows = documented.iter().filter(|(_, w)| w == "search").count();
        assert_eq!(search_rows, 1, "{documented:?}");
        assert_eq!(documented.last().unwrap().1, "search", "{documented:?}");
        let all: Vec<&str> = documented.iter().map(|(f, _)| f.as_str()).collect();
        let harness = &all[..all.len() - 1];

        let path = std::env::temp_dir().join(format!(
            "coffee-ten-bench-schema-{}.jsonl",
            std::process::id()
        ));
        let _ = fs::remove_file(&path);
        let record = flat("vectorize(1, j)", 2.0);
        let metadata = test_metadata(1);
        append_lines(&path, &[record_json(&record, &metadata)]).unwrap();
        let (mut config, _) = hand_search(None);
        config.log = Some(path.clone());
        log_records(&config, &metadata, "final", &[&record]).unwrap();
        let log_text = fs::read_to_string(&path).unwrap();
        fs::remove_file(&path).unwrap();

        let lines: Vec<&str> = log_text.lines().collect();
        assert_eq!(lines.len(), 2, "{log_text}");
        assert_eq!(top_level_keys(lines[0]), harness, "{}", lines[0]);
        assert_eq!(top_level_keys(lines[1]), all, "{}", lines[1]);
    }

    /// A record with the given schedule text and three rates (median = the middle one).
    fn hand_record(schedule: &str, rates: [f64; 3]) -> Record {
        let info = RunInfo {
            kernel: "matmul",
            n: 8,
            graph_hash: 0,
            quick: false,
        };
        let mut record = make_record(
            &info,
            schedule,
            &Schedule::default(),
            OrderClass::Preserving,
            &rates,
        )
        .unwrap();
        record.schedule = schedule.to_string();
        record
    }

    fn flat(schedule: &str, median: f64) -> Record {
        hand_record(schedule, [median, median, median])
    }

    /// Random candidates measure 5, 15, 10, 20 and the model predicted 10, 20, 30, 40; the
    /// model's two measured 18 and 12 in the sequential search, but 15 against 20 in the
    /// final run. The decision must use the final run.
    fn hand_search(final_run: Option<FinalRun>) -> (SearchConfig, Search) {
        let config = SearchConfig {
            n: 8,
            budget: 4,
            seed: 1,
            samples: 3,
            min_sample_ms: 1,
            quick: false,
            log: None,
            model_top: Some(2),
            from: Some(PathBuf::from("peak.jsonl")),
            local: false,
        };
        let result = Search {
            space_size: 100,
            naive: flat("", 7.0),
            records: vec![
                flat("r1", 5.0),
                flat("r2", 15.0),
                flat("r3", 10.0),
                flat("r4", 20.0),
            ],
            model: Some(ModelGroup {
                machine: Machine::assumed_m5(),
                random_predicted: vec![10.0, 20.0, 30.0, 40.0],
                records: vec![flat("m1", 12.0), flat("m2", 18.0)],
                final_run,
            }),
        };
        (config, result)
    }

    #[test]
    fn the_ratio_is_from_the_final_run_not_the_sequential_medians() {
        let final_run = FinalRun {
            random: hand_record("r4", [19.0, 20.0, 21.0]),
            model: hand_record("m2", [14.0, 15.0, 16.0]),
        };
        let (config, result) = hand_search(Some(final_run));
        let text = report(&config, &result, &test_metadata(1));
        // 15 / 20 = 0.75; the sequential medians would give 18 / 20 = 0.9, which is met.
        assert!(text.contains("= 0.750: criterion NOT MET"), "{text}");
        assert!(!text.contains("0.900"), "{text}");
        assert!(
            text.contains("final random  median 20.000 [p25 19.000, p75 20.000]"),
            "{text}"
        );
        assert!(
            text.contains("final model   median 15.000 [p25 14.000, p75 15.000]"),
            "{text}"
        );
        assert!(text.contains("best random (m = 4): r4"), "{text}");
        assert!(text.contains("best model  (model rank 2): m2"), "{text}");
    }

    #[test]
    fn ratio_at_the_criterion_is_met() {
        let final_run = FinalRun {
            random: flat("r4", 20.0),
            model: flat("m2", 18.0),
        };
        let (config, result) = hand_search(Some(final_run));
        let text = report(&config, &result, &test_metadata(1));
        assert!(text.contains("= 0.900: criterion MET"), "{text}");
    }

    #[test]
    fn spearman_and_log_error_of_the_hand_case() {
        let (config, result) = hand_search(None);
        let text = report(&config, &result, &test_metadata(1));
        // Ranks of the measured [5, 15, 10, 20] are [1, 3, 2, 4], of the predicted
        // [1, 2, 3, 4]: sum d^2 = 2, rho = 1 - 6 * 2 / (4 * 15) = 0.8. The |ln(p/m)| are
        // ln 2, ln(4/3), ln 3, ln 2: sorted, the element at index 1 is ln 2 = 0.693.
        assert!(
            text.contains(
                "Spearman rank correlation 0.800; median |ln(predicted/measured)| 0.693 \
                 (a factor 2.00)"
            ),
            "{text}"
        );
        assert!(
            text.contains("no final run; ratio = 1.000: criterion MET"),
            "{text}"
        );
        // The model curve: m = 1, 2 over the two records in rank order.
        assert!(text.contains("   1       12.000  m1"), "{text}");
        assert!(text.contains("   2       18.000  m2"), "{text}");
    }

    /// A peak line as `bench/peak` writes it, with made-up numbers.
    const PEAK_LINE: &str = "{\"kind\":\"peak\",\"fma_gflops\":{\"median\":100.0},\
        \"triad_gbs\":{\"median\":100.0},\"triad_l2_gbs\":{\"median\":100.0}}";

    #[test]
    fn quick_local_search_on_16_measures_the_hill_climb_and_logs_it() {
        let dir = std::env::temp_dir();
        let pid = std::process::id();
        let log = dir.join(format!("coffee-ten-bench-search-local-{pid}.jsonl"));
        let peak = dir.join(format!("coffee-ten-bench-search-local-peak-{pid}.jsonl"));
        let _ = fs::remove_file(&log);
        fs::write(&peak, format!("{PEAK_LINE}\n")).unwrap();
        let config = SearchConfig {
            n: 16,
            budget: 4,
            seed: 1,
            samples: QUICK_SAMPLES,
            min_sample_ms: QUICK_MIN_SAMPLE_MS,
            quick: true,
            log: Some(log.clone()),
            model_top: None,
            from: Some(peak.clone()),
            local: true,
        };
        let metadata = test_metadata(config.seed);
        let result = search(&config, &metadata).unwrap();
        assert_eq!(references_built(), 1);
        let log_text = fs::read_to_string(&log).unwrap();
        fs::remove_file(&log).unwrap();
        fs::remove_file(&peak).unwrap();

        assert!(result.model.is_none());
        assert_eq!(result.records.len(), 4);
        let space = matmul_space(16, 16, 16);
        let machine = machine_from_results(PEAK_LINE).unwrap().0;
        let ranked = rank_matmul(&machine, 16).unwrap();
        assert_eq!(result.records[0].schedule, ranked[0].schedule);
        let index_of: HashMap<&str, usize> = space
            .iter()
            .enumerate()
            .map(|(index, candidate)| (candidate.schedule.as_str(), index))
            .collect();
        let mut rank_of = vec![0; space.len()];
        for (rank, entry) in ranked.iter().enumerate() {
            rank_of[index_of[entry.schedule.as_str()]] = rank;
        }
        let medians: HashMap<usize, f64> = result
            .records
            .iter()
            .map(|record| (index_of[record.schedule.as_str()], record.stats.median))
            .collect();
        let order = local_search(&neighbors(&space, 16, 16, 16), &rank_of, 4, |index| {
            Ok::<f64, &str>(medians[&index])
        })
        .unwrap();
        let got: Vec<usize> = result
            .records
            .iter()
            .map(|record| index_of[record.schedule.as_str()])
            .collect();
        assert_eq!(got, order);
        let distinct: HashSet<usize> = got.iter().copied().collect();
        assert_eq!(distinct.len(), got.len());

        let groups: Vec<&str> = log_text
            .lines()
            .map(|line| {
                let tail = line.rsplit_once("\"group\":\"").unwrap().1;
                tail.strip_suffix("\"}").unwrap()
            })
            .collect();
        assert_eq!(groups, ["naive", "local", "local", "local", "local"]);
        let text = report(&config, &result, &metadata);
        assert!(text.contains("local search (spec 7.1)"), "{text}");
        assert!(!text.contains("criterion (spec 7.1)"), "{text}");
    }

    #[test]
    fn quick_model_search_on_16_logs_every_group_and_decides_on_the_final_run() {
        let dir = std::env::temp_dir();
        let pid = std::process::id();
        let log = dir.join(format!("coffee-ten-bench-search-model-{pid}.jsonl"));
        let peak = dir.join(format!("coffee-ten-bench-search-peak-{pid}.jsonl"));
        let _ = fs::remove_file(&log);
        fs::write(&peak, format!("{PEAK_LINE}\n")).unwrap();
        let config = SearchConfig {
            n: 16,
            budget: 3,
            seed: 1,
            samples: QUICK_SAMPLES,
            min_sample_ms: QUICK_MIN_SAMPLE_MS,
            quick: true,
            log: Some(log.clone()),
            model_top: Some(2),
            from: Some(peak.clone()),
            local: false,
        };
        let metadata = test_metadata(config.seed);
        let result = search(&config, &metadata).unwrap();
        let log_text = fs::read_to_string(&log).unwrap();
        fs::remove_file(&log).unwrap();
        fs::remove_file(&peak).unwrap();

        let model = result.model.as_ref().unwrap();
        // One reference for the naive record, the 3 random and the 2 model candidates, one more
        // for the final run if there is one (it is a harness `run` of its own).
        assert_eq!(
            references_built(),
            1 + usize::from(model.final_run.is_some())
        );
        assert_eq!(result.records.len(), 3);
        assert_eq!(model.records.len(), 2);
        assert_eq!(model.random_predicted.len(), 3);
        assert!(model.random_predicted.iter().all(|p| *p > 0.0));
        // The model's top 2 are the first two of its ranking, whatever the random sample.
        let ranked = rank_matmul(&model.machine, 16).unwrap();
        for (record, entry) in model.records.iter().zip(&ranked) {
            assert_eq!(record.schedule, entry.schedule);
        }

        let groups: Vec<&str> = log_text
            .lines()
            .map(|line| {
                let tail = line.rsplit_once("\"group\":\"").unwrap().1;
                tail.strip_suffix("\"}").unwrap()
            })
            .collect();
        let same = result.records[best_of(&result.records)].schedule
            == model.records[best_of(&model.records)].schedule;
        let mut want = vec!["naive", "random", "random", "random", "model", "model"];
        if same {
            assert!(model.final_run.is_none());
        } else {
            assert!(model.final_run.is_some());
            want.extend(["final", "final"]);
        }
        assert_eq!(groups, want, "{log_text}");

        let text = report(&config, &result, &metadata);
        assert!(text.contains("criterion (spec 7.1)"), "{text}");
        assert!(
            text.contains("criterion MET") || text.contains("criterion NOT MET"),
            "{text}"
        );
        assert!(text.contains("Spearman rank correlation"), "{text}");
        if let Some(final_run) = &model.final_run {
            assert_eq!(
                final_run.random.schedule,
                result.records[best_of(&result.records)].schedule
            );
            assert_eq!(
                final_run.model.schedule,
                model.records[best_of(&model.records)].schedule
            );
        }
    }

    #[test]
    fn a_missing_results_file_fails_before_anything_is_measured() {
        let log = std::env::temp_dir().join(format!(
            "coffee-ten-bench-search-nofile-{}.jsonl",
            std::process::id()
        ));
        let _ = fs::remove_file(&log);
        let config = SearchConfig {
            n: 16,
            budget: 1,
            seed: 1,
            samples: QUICK_SAMPLES,
            min_sample_ms: QUICK_MIN_SAMPLE_MS,
            quick: true,
            log: Some(log.clone()),
            model_top: Some(1),
            from: Some(PathBuf::from("/nonexistent/peak.jsonl")),
            local: false,
        };
        assert!(search(&config, &test_metadata(1)).is_err());
        assert!(!log.exists());
    }
}
