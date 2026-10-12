//! Analytic cost model, version 0 (spec §7.1): cache traffic from the loop footprints plus a
//! roofline. Deterministic, no measurements; every constant is an assumption.
//!
//! Only the update nest (nest 1) of a matmul kernel is modelled; the init nest is ignored.
//! Guards are ignored (the schedule space has none). Loop extents are exact, so the
//! iteration counts are exact too.
//!
//! Footprint `F(p)` of loop level `p` is the bytes of distinct elements of the distinct
//! accesses (C, A, B; `C[i, j]` is one access though it is loaded and stored) touched by one
//! full run of loops `p..` (loop `p` and everything inside it). Per access and index dim,
//! the loops at or inside `p` give `min(product of extents, 1 + sum |coeff| * (extent - 1))`
//! distinct values over the terms of the index (exact for `split` indices such as
//! `32*io + ii`; loops outside `p` are fixed). The distinct values of the last dim are taken
//! as a contiguous run starting on a line boundary, so its bytes are rounded up to whole
//! lines; this is why a lone element costs a full line. Other dims multiply.
//!
//! Traffic past a cache of `S` bytes: the innermost level `p` with `F(p) > S`; then every
//! run of the loops inside `p` is reloaded: `F(p + 1) * (iterations of loops 0..=p)`. If no
//! level exceeds `S`, the traffic is `F(0)`, once. Each access is counted once, so a
//! written `C` is not charged for its write-back or write-allocate (as in `bench/peak`).

use crate::BenchError;
use crate::cli::MAX_N;
use crate::space::matmul_space;
use coffee_ten_ir::{Graph, Shape};
use coffee_ten_loop::{Access, Expr, Kernel, Loop, Schedule, apply, lower};
use std::cmp::Ordering;
use std::fmt::Write as _;

/// The matmul update nest; nest 0 is the init nest.
const UPDATE_NEST: usize = 1;
/// Bytes of one f32 element.
const ELEMENT_BYTES: usize = 4;
/// How many ranked schedules the CLI prints.
pub const TOP: usize = 10;

/// PLACEHOLDER, not measured: single-thread f32 peak in GFLOP/s. Stands for 4 FMA pipes x 4
/// lanes x 2 FLOP x about 3.2 GHz. Replace by the median of the `bench/peak` run.
pub const PLACEHOLDER_PEAK_GFLOPS: f64 = 100.0;
/// PLACEHOLDER, not measured: single-thread memory bandwidth in GB/s (1e9 bytes). Replace by
/// the STREAM triad median of the `bench/peak` run.
pub const PLACEHOLDER_MEM_GBS: f64 = 60.0;
/// PLACEHOLDER, not measured: bandwidth from L2 into L1 in GB/s. Nothing in `bench/peak`
/// measures it yet.
pub const PLACEHOLDER_L2_GBS: f64 = 200.0;

/// The machine the model predicts for. All fields are inputs, none is detected.
#[derive(Clone, Debug, PartialEq)]
pub struct Machine {
    /// Peak f32 GFLOP/s of vectorized code on one core.
    pub peak_gflops: f64,
    /// Memory bandwidth in GB/s (1e9 bytes) seen by one core.
    pub mem_gbs: f64,
    /// L2 to L1 bandwidth in GB/s (1e9 bytes) seen by one core.
    pub l2_gbs: f64,
    /// L1 data cache size in bytes.
    pub l1_bytes: usize,
    /// L2 cache size in bytes.
    pub l2_bytes: usize,
    /// Cache line size in bytes.
    pub line_bytes: usize,
    /// f32 lanes of one vector operation.
    pub lanes: usize,
}

impl Machine {
    /// An Apple M-series P-core as we assume it, with placeholder speeds. ASSUMED, not
    /// measured: L1D 128 KiB, L2 16 MiB, line 128 B, 4 f32 lanes (NEON). Peak and bandwidths
    /// are the `PLACEHOLDER_*` constants until the peak run.
    pub fn assumed_m5() -> Self {
        Self {
            peak_gflops: PLACEHOLDER_PEAK_GFLOPS,
            mem_gbs: PLACEHOLDER_MEM_GBS,
            l2_gbs: PLACEHOLDER_L2_GBS,
            l1_bytes: 128 * 1024,
            l2_bytes: 16 * 1024 * 1024,
            line_bytes: 128,
            lanes: 4,
        }
    }

    fn check(&self) -> Result<(), BenchError> {
        let speeds = [self.peak_gflops, self.mem_gbs, self.l2_gbs];
        let sizes = [self.l1_bytes, self.l2_bytes, self.line_bytes, self.lanes];
        if speeds.iter().all(|s| s.is_finite() && *s > 0.0) && sizes.iter().all(|s| *s > 0) {
            Ok(())
        } else {
            Err(BenchError::Setup(format!(
                "machine numbers must be positive and finite: {self:?}"
            )))
        }
    }
}

/// What the model says about one kernel.
#[derive(Clone, Debug, PartialEq)]
pub struct Prediction {
    pub time_s: f64,
    /// `flops / time_s / 1e9`.
    pub gflops: f64,
    /// Bytes moved from memory into L2 (traffic past L2).
    pub mem_bytes: usize,
    /// Bytes moved from L2 into L1 (traffic past L1).
    pub l2_bytes: usize,
    /// The roofline term that sets `time_s`: "compute", "memory" or "l2". A tie goes to the
    /// first of that order.
    pub bound: &'static str,
}

/// A schedule text and its prediction.
#[derive(Clone, Debug, PartialEq)]
pub struct Ranked {
    pub schedule: String,
    pub prediction: Prediction,
}

fn model_err(msg: String) -> BenchError {
    BenchError::Setup(msg)
}

/// Each access once, the store target first, then the loads in the order of the expression.
fn distinct_accesses(kernel_nest: &coffee_ten_loop::LoopNest) -> Vec<&Access> {
    fn loads<'a>(e: &'a Expr, out: &mut Vec<&'a Access>) {
        match e {
            Expr::Load(access) => out.push(access),
            Expr::Const(_) => {}
            Expr::Unary(_, x) => loads(x, out),
            Expr::Binary(_, a, b) => {
                loads(a, out);
                loads(b, out);
            }
        }
    }
    let mut all = vec![&kernel_nest.body.target];
    loads(&kernel_nest.body.value, &mut all);
    let mut unique: Vec<&Access> = Vec::new();
    for access in all {
        if !unique.contains(&access) {
            unique.push(access);
        }
    }
    unique
}

/// Distinct values of one index dim over the loops `from..` of `loops`.
fn distinct_values(
    index: &coffee_ten_loop::AffineExpr,
    loops: &[Loop],
    from: usize,
) -> Result<usize, BenchError> {
    let mut count = 1_usize;
    let mut span = 1_usize;
    for (name, coeff) in &index.terms {
        let pos = loops
            .iter()
            .position(|l| &l.name == name)
            .ok_or_else(|| model_err(format!("index names the unknown loop {name}")))?;
        if pos < from || *coeff == 0 {
            continue;
        }
        let extent = loops[pos].extent;
        count = count.saturating_mul(extent);
        let reach = usize::try_from(coeff.unsigned_abs())
            .unwrap_or(usize::MAX)
            .saturating_mul(extent - 1);
        span = span.saturating_add(reach);
    }
    Ok(count.min(span))
}

/// Bytes of the elements of `access` touched by the loops `from..`, last dim in whole lines.
fn access_bytes(
    access: &Access,
    loops: &[Loop],
    from: usize,
    line_bytes: usize,
) -> Result<usize, BenchError> {
    let mut bytes = 1_usize;
    let mut last_dim_values = 1_usize;
    for (dim, index) in access.index.iter().enumerate() {
        let values = distinct_values(index, loops, from)?;
        if dim + 1 == access.index.len() {
            last_dim_values = values;
        } else {
            bytes = bytes.saturating_mul(values);
        }
    }
    let run_bytes = last_dim_values.saturating_mul(ELEMENT_BYTES);
    let run_lines = run_bytes.div_ceil(line_bytes);
    Ok(bytes.saturating_mul(run_lines.saturating_mul(line_bytes)))
}

/// `result[p][a]`: bytes of access `a` (C, A, B for the matmul update) over the loops `p..`,
/// for `p` in `0..=loops.len()`; `p = loops.len()` is one run of the body.
fn level_bytes(
    nest: &coffee_ten_loop::LoopNest,
    line_bytes: usize,
) -> Result<Vec<Vec<usize>>, BenchError> {
    let accesses = distinct_accesses(nest);
    (0..=nest.loops.len())
        .map(|p| {
            accesses
                .iter()
                .map(|a| access_bytes(a, &nest.loops, p, line_bytes))
                .collect()
        })
        .collect()
}

/// Traffic past a cache of `cache_bytes`, from the footprints `totals[p]` of each level
/// `p` in `0..=extents.len()` and the extents of the loops (outermost first).
fn traffic_past(totals: &[usize], extents: &[usize], cache_bytes: usize) -> usize {
    let overflowing = (0..extents.len()).rev().find(|&p| totals[p] > cache_bytes);
    match overflowing {
        Some(p) => {
            let reloads = extents[..=p]
                .iter()
                .fold(1_usize, |acc, e| acc.saturating_mul(*e));
            totals[p + 1].saturating_mul(reloads)
        }
        None => totals[0],
    }
}

/// Predicts the update nest of `kernel` on `machine`. Errors if the machine numbers are not
/// positive and finite or the kernel has no nest 1 or an index names an unknown loop.
pub fn predict(machine: &Machine, kernel: &Kernel) -> Result<Prediction, BenchError> {
    machine.check()?;
    let nest = kernel
        .nests
        .get(UPDATE_NEST)
        .ok_or_else(|| model_err(format!("kernel has no nest {UPDATE_NEST}")))?;
    let extents: Vec<usize> = nest.loops.iter().map(|l| l.extent).collect();
    let totals: Vec<usize> = level_bytes(nest, machine.line_bytes)?
        .iter()
        .map(|per_access| per_access.iter().fold(0_usize, |a, b| a.saturating_add(*b)))
        .collect();
    let l2_bytes = traffic_past(&totals, &extents, machine.l1_bytes);
    let mem_bytes = traffic_past(&totals, &extents, machine.l2_bytes);

    // 2 FLOP (multiply, add) per iteration of the update nest.
    let flops = 2.0 * extents.iter().fold(1.0, |acc, e| acc * *e as f64);
    // ASSUMPTION: scalar code issues one lane per operation, so it runs at peak / lanes.
    let is_vectorized = nest.loops.last().is_some_and(|l| l.vectorize);
    let ceiling_gflops = if is_vectorized {
        machine.peak_gflops
    } else {
        machine.peak_gflops / machine.lanes as f64
    };
    let compute_s = flops / (ceiling_gflops * 1e9);
    let memory_s = mem_bytes as f64 / (machine.mem_gbs * 1e9);
    let l2_s = l2_bytes as f64 / (machine.l2_gbs * 1e9);
    let time_s = compute_s.max(memory_s).max(l2_s);
    let bound = if compute_s >= memory_s && compute_s >= l2_s {
        "compute"
    } else if memory_s >= l2_s {
        "memory"
    } else {
        "l2"
    };
    Ok(Prediction {
        time_s,
        gflops: flops / time_s / 1e9,
        mem_bytes,
        l2_bytes,
        bound,
    })
}

fn matmul_kernel(n: usize) -> Result<Kernel, BenchError> {
    let mut g = Graph::new();
    let shape = || Shape::new(vec![n, n]);
    let a = g
        .input("A", shape())
        .map_err(|e| model_err(e.to_string()))?;
    let b = g
        .input("B", shape())
        .map_err(|e| model_err(e.to_string()))?;
    let product = g.matmul(a, b).map_err(|e| model_err(e.to_string()))?;
    lower(&g, product, "model_kernel").map_err(|e| model_err(e.to_string()))
}

/// Faster prediction first; equal times by schedule text.
fn order(a: &Ranked, b: &Ranked) -> Ordering {
    a.prediction
        .time_s
        .total_cmp(&b.prediction.time_s)
        .then_with(|| a.schedule.cmp(&b.schedule))
}

/// Predicts every schedule of `matmul_space(n, n, n)` and sorts them, best first.
pub fn rank_matmul(machine: &Machine, n: usize) -> Result<Vec<Ranked>, BenchError> {
    machine.check()?;
    let naive = matmul_kernel(n)?;
    let mut ranked = Vec::new();
    for candidate in matmul_space(n, n, n) {
        let schedule =
            Schedule::parse(&candidate.schedule).map_err(|e| model_err(e.to_string()))?;
        let scheduled = apply(&naive, &schedule).map_err(|e| model_err(e.to_string()))?;
        let prediction = predict(machine, &scheduled)?;
        ranked.push(Ranked {
            schedule: candidate.schedule,
            prediction,
        });
    }
    ranked.sort_by(order);
    Ok(ranked)
}

const USAGE: &str =
    "usage: coffee-ten-bench model --n <N> [--peak <GFLOPS>] [--mem-gbs <GBS>] [--l2-gbs <GBS>]";

/// Parses the arguments after `model`.
pub fn parse_model_args<I: IntoIterator<Item = String>>(
    args: I,
) -> Result<(usize, Machine), BenchError> {
    let usage = |msg: String| BenchError::Usage(format!("{msg}\n{USAGE}"));
    let mut machine = Machine::assumed_m5();
    let mut n = None;
    let mut args = args.into_iter();
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| usage(format!("{flag} needs a value")))?;
        let speed = || match value.parse::<f64>() {
            Ok(v) if v.is_finite() && v > 0.0 => Ok(v),
            _ => Err(usage(format!(
                "{flag} needs a positive number, got {value:?}"
            ))),
        };
        match flag.as_str() {
            "--n" => {
                n = Some(value.parse::<usize>().map_err(|_| {
                    usage(format!("--n needs a non-negative integer, got {value:?}"))
                })?);
            }
            "--peak" => machine.peak_gflops = speed()?,
            "--mem-gbs" => machine.mem_gbs = speed()?,
            "--l2-gbs" => machine.l2_gbs = speed()?,
            _ => return Err(usage(format!("unknown argument {flag}"))),
        }
    }
    let n = n.ok_or_else(|| usage("--n is required".to_string()))?;
    if !(1..=MAX_N).contains(&n) {
        return Err(usage(format!("--n must be in 1..={MAX_N}, got {n}")));
    }
    Ok((n, machine))
}

/// The report of `coffee-ten-bench model`: the machine numbers and the top `TOP` schedules.
pub fn report(n: usize, machine: &Machine, ranked: &[Ranked]) -> String {
    let mut text = String::new();
    let _ = writeln!(
        text,
        "PREDICTION by the analytic model v0, not a measurement"
    );
    let _ = writeln!(
        text,
        "matmul {n}x{n}x{n}, update nest 1, {} candidates",
        ranked.len()
    );
    let _ = writeln!(
        text,
        "machine: peak {} GFLOP/s, mem {} GB/s, L2 {} GB/s (placeholders unless set on the command line); \
         L1 {} B, L2 {} B, line {} B, {} lanes (ASSUMED)",
        machine.peak_gflops,
        machine.mem_gbs,
        machine.l2_gbs,
        machine.l1_bytes,
        machine.l2_bytes,
        machine.line_bytes,
        machine.lanes
    );
    let _ = writeln!(
        text,
        "{:>3} {:>10} {:>8}  schedule",
        "#", "GFLOP/s", "bound"
    );
    for (index, r) in ranked.iter().take(TOP).enumerate() {
        let _ = writeln!(
            text,
            "{index:>3} {:>10.3} {:>8}  {}",
            r.prediction.gflops, r.prediction.bound, r.schedule
        );
    }
    text
}

/// `coffee-ten-bench model ...`: ranks the schedule space, measures nothing.
pub fn cli_main<I: IntoIterator<Item = String>>(args: I) -> std::process::ExitCode {
    let run = || -> Result<String, BenchError> {
        let (n, machine) = parse_model_args(args)?;
        Ok(report(n, &machine, &rank_matmul(&machine, n)?))
    };
    match run() {
        Ok(text) => {
            print!("{text}");
            std::process::ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            let is_usage = matches!(e, BenchError::Usage(_));
            std::process::ExitCode::from(if is_usage { 2 } else { 1 })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    const TILED: &str =
        "split(1, i, io, ii, 32); split(1, j, jo, ji, 64); reorder(1, io, jo, k, ii, ji)";

    fn naive(n: usize) -> Kernel {
        matmul_kernel(n).expect("lowers")
    }

    fn scheduled(n: usize, text: &str) -> Kernel {
        apply(&naive(n), &Schedule::parse(text).expect("parses")).expect("applies")
    }

    fn update_nest(kernel: &Kernel) -> &coffee_ten_loop::LoopNest {
        &kernel.nests[UPDATE_NEST]
    }

    /// Naive `i, k, j`, N = 64, line 128 B = 32 floats. Accesses in the order C, A, B.
    /// One row of 64 floats is 256 B = 2 whole lines, so rows cost no rounding; a lone
    /// element costs a full line (128 B).
    /// - body (nothing inside): each access 1 element -> 128 B.
    /// - loop j (64): C[i,j] 64 el = 256 B; A[i,k] 1 el -> 128 B; B[k,j] 256 B.
    /// - loop k (k, j): C 1 row x 256 = 256; A[i,k]: 1 row, 64 contiguous k -> 256; B: 64 rows
    ///   x 256 = 16384.
    /// - loop i (all): C 64 x 256 = 16384; A 64 rows x (64 el = 256 B) = 16384; B 16384.
    #[test]
    fn footprint_of_the_naive_nest_at_n_64_is_hand_computed() {
        let levels = level_bytes(update_nest(&naive(64)), 128).expect("analyses");
        assert_eq!(
            levels,
            vec![
                vec![16384, 16384, 16384],
                vec![256, 256, 16384],
                vec![256, 128, 256],
                vec![128, 128, 128],
            ]
        );
    }

    #[test]
    fn a_split_index_counts_distinct_values_not_the_product_of_extents() {
        // A[32*io + ii, k] over (ii) only has 32 values; over (io, ii) 1 + 32*(2-1) + 31 = 64.
        let kernel = scheduled(64, "split(1, i, io, ii, 32); reorder(1, io, k, j, ii)");
        let nest = update_nest(&kernel);
        let a = &distinct_accesses(nest)[1];
        assert_eq!(distinct_values(&a.index[0], &nest.loops, 0), Ok(64));
        assert_eq!(distinct_values(&a.index[0], &nest.loops, 1), Ok(32));
        assert_eq!(distinct_values(&a.index[0], &nest.loops, 3), Ok(32));
        assert_eq!(distinct_values(&a.index[0], &nest.loops, 4), Ok(1));
    }

    /// N = 64: the footprint of the whole nest is 3 x 64 x 64 x 4 = 49152 B, which is below
    /// L1 (131072 B) and L2, so both traffics are that footprint once.
    #[test]
    fn a_nest_that_fits_moves_its_footprint_once() {
        let p = predict(&Machine::assumed_m5(), &naive(64)).expect("predicts");
        assert_eq!(p.mem_bytes, 3 * 64 * 64 * 4);
        assert_eq!(p.l2_bytes, 3 * 64 * 64 * 4);
    }

    /// Naive `i, k, j`, N = 1024. A row is 4096 B = 32 lines.
    /// - loop j: C 4096 + A 128 + B 4096 = 8320.
    /// - loop k: C 4096 + A 4096 + B 1024 x 4096 = 4194304 -> 4202496, over L1 = 131072 (B
    ///   alone is 4 MiB). Loop j still fits.
    /// - L1 traffic = F(j) x (iterations of i, k) = 8320 x 1024 x 1024 = 8724152320.
    /// - all loops: 3 x 4 MiB = 12582912 B <= L2 = 16 MiB, so memory traffic is that once.
    #[test]
    fn the_naive_nest_at_n_1024_reloads_from_l2_but_not_from_memory() {
        let p = predict(&Machine::assumed_m5(), &naive(1024)).expect("predicts");
        assert_eq!(p.l2_bytes, 8_724_152_320);
        assert_eq!(p.mem_bytes, 12_582_912);
    }

    /// Tiled `io, jo, k, ii, ji` (32, 64), N = 1024: extents 32, 16, 1024, 32, 64.
    /// - loop ji: C 256 + A 128 + B 256 = 640.
    /// - loop ii: C 32 x 256 = 8192; A 32 x 128 = 4096; B 256 -> 12544.
    /// - loop k: C 8192; A 32 x 4096 = 131072; B 1024 x 256 = 262144 -> 401408 > L1.
    /// - L1 traffic = F(ii) x (io, jo, k iterations) = 12544 x 32 x 16 x 1024 = 6576668672,
    ///   which is less than the naive 8724152320.
    #[test]
    fn tiling_moves_less_from_l2_than_the_naive_nest() {
        let machine = Machine::assumed_m5();
        let tiled = predict(&machine, &scheduled(1024, TILED)).expect("predicts");
        let plain = predict(&machine, &naive(1024)).expect("predicts");
        assert_eq!(tiled.l2_bytes, 6_576_668_672);
        assert!(tiled.l2_bytes < plain.l2_bytes);
    }

    #[test]
    fn the_cache_size_decides_the_level_that_is_reloaded() {
        // Footprints 10, 100, 1000 for 3 loops of extents 4, 5, 6 (outer first), body 1.
        let totals = [1000, 100, 10, 1];
        let extents = [4, 5, 6];
        assert_eq!(traffic_past(&totals, &extents, 1000), 1000);
        assert_eq!(traffic_past(&totals, &extents, 999), 100 * 4);
        assert_eq!(traffic_past(&totals, &extents, 99), 10 * 4 * 5);
        assert_eq!(traffic_past(&totals, &extents, 9), 4 * 5 * 6);
    }

    #[test]
    fn vectorize_is_never_predicted_slower_over_the_n_64_space() {
        let machine = Machine::assumed_m5();
        let by_text: HashMap<String, Prediction> = rank_matmul(&machine, 64)
            .expect("ranks")
            .into_iter()
            .map(|r| (r.schedule, r.prediction))
            .collect();
        let mut pairs = 0;
        for (text, vectorized) in &by_text {
            let Some((base, _)) = text.rsplit_once("; vectorize(") else {
                continue;
            };
            let plain = by_text
                .get(base)
                .expect("the unvectorized twin is in the space");
            assert!(vectorized.time_s <= plain.time_s, "{text}");
            pairs += 1;
        }
        assert!(pairs > 100, "only {pairs} pairs");
    }

    #[test]
    fn scalar_code_runs_at_a_lane_th_of_peak() {
        let machine = Machine::assumed_m5();
        let plain = predict(&machine, &naive(64)).expect("predicts");
        let kernel = scheduled(64, "reorder(1, i, k, j); vectorize(1, j)");
        let vectorized = predict(&machine, &kernel).expect("predicts");
        assert_eq!(plain.bound, "compute");
        // 64^3 x 2 FLOP at 100 / 4 GFLOP/s, then at 100 GFLOP/s.
        assert!((plain.gflops - 25.0).abs() < 1e-9, "{}", plain.gflops);
        assert!(
            (vectorized.gflops - 100.0).abs() < 1e-9,
            "{}",
            vectorized.gflops
        );
    }

    #[test]
    fn a_slow_memory_makes_the_prediction_memory_bound() {
        let machine = Machine {
            mem_gbs: 0.001,
            ..Machine::assumed_m5()
        };
        let p = predict(&machine, &naive(64)).expect("predicts");
        assert_eq!(p.bound, "memory");
        // 49152 B at 1e6 B/s.
        assert!((p.time_s - 49152.0 / 1e6).abs() < 1e-12, "{}", p.time_s);
        let slow_l2 = Machine {
            l2_gbs: 0.001,
            mem_gbs: 1e6,
            ..Machine::assumed_m5()
        };
        assert_eq!(predict(&slow_l2, &naive(64)).expect("predicts").bound, "l2");
    }

    #[test]
    fn the_ranking_is_deterministic_and_ties_go_by_schedule_text() {
        let machine = Machine::assumed_m5();
        let first = rank_matmul(&machine, 64).expect("ranks");
        assert_eq!(first, rank_matmul(&machine, 64).expect("ranks"));
        let mut ties = 0;
        for pair in first.windows(2) {
            let (a, b) = (&pair[0], &pair[1]);
            assert!(a.prediction.time_s <= b.prediction.time_s);
            if a.prediction.time_s == b.prediction.time_s {
                assert!(a.schedule < b.schedule, "{} / {}", a.schedule, b.schedule);
                ties += 1;
            }
        }
        assert!(ties > 0, "the check never met a tie");
    }

    #[test]
    fn the_order_function_breaks_ties_by_text() {
        let p = predict(&Machine::assumed_m5(), &naive(8)).expect("predicts");
        let ranked = |schedule: &str| Ranked {
            schedule: schedule.to_string(),
            prediction: p.clone(),
        };
        assert_eq!(order(&ranked("a"), &ranked("b")), Ordering::Less);
        assert_eq!(order(&ranked("b"), &ranked("a")), Ordering::Greater);
        let faster = Ranked {
            prediction: Prediction {
                time_s: p.time_s / 2.0,
                ..p.clone()
            },
            ..ranked("z")
        };
        assert_eq!(order(&faster, &ranked("a")), Ordering::Less);
    }

    #[test]
    fn bad_machines_and_kernels_are_errors() {
        let zero_bw = Machine {
            mem_gbs: 0.0,
            ..Machine::assumed_m5()
        };
        assert!(predict(&zero_bw, &naive(8)).is_err());
        let nan = Machine {
            peak_gflops: f64::NAN,
            ..Machine::assumed_m5()
        };
        assert!(rank_matmul(&nan, 8).is_err());
        let no_lanes = Machine {
            lanes: 0,
            ..Machine::assumed_m5()
        };
        assert!(predict(&no_lanes, &naive(8)).is_err());
        let mut kernel = naive(8);
        kernel.nests.truncate(1);
        assert!(predict(&Machine::assumed_m5(), &kernel).is_err());
    }

    #[test]
    fn model_args_limits_are_checked_on_both_sides() {
        let parse = |args: &[&str]| parse_model_args(args.iter().map(|a| a.to_string()));
        let (n, machine) = parse(&["--n", "256"]).expect("parses");
        assert_eq!((n, machine), (256, Machine::assumed_m5()));
        assert!(parse(&["--n", "0"]).is_err());
        assert_eq!(parse(&["--n", "1"]).expect("parses").0, 1);
        assert_eq!(
            parse(&["--n", &MAX_N.to_string()]).expect("parses").0,
            MAX_N
        );
        assert!(parse(&["--n", &(MAX_N + 1).to_string()]).is_err());
        assert!(parse(&[]).is_err());
        assert!(parse(&["--n"]).is_err());
        let (_, set) = parse(&[
            "--n",
            "8",
            "--peak",
            "50",
            "--mem-gbs",
            "10",
            "--l2-gbs",
            "30",
        ])
        .expect("parses");
        assert_eq!(
            (set.peak_gflops, set.mem_gbs, set.l2_gbs),
            (50.0, 10.0, 30.0)
        );
        assert!(parse(&["--n", "8", "--peak", "0"]).is_err());
        assert!(parse(&["--n", "8", "--peak", "0.001"]).is_ok());
        assert!(parse(&["--n", "8", "--mem-gbs", "-1"]).is_err());
        assert!(parse(&["--n", "8", "--l2-gbs", "inf"]).is_err());
        assert!(parse(&["--n", "8", "--bogus", "1"]).is_err());
    }

    #[test]
    fn the_report_lists_the_top_ten() {
        let machine = Machine::assumed_m5();
        let text = report(16, &machine, &rank_matmul(&machine, 16).expect("ranks"));
        assert!(text.contains("not a measurement"), "{text}");
        assert_eq!(text.matches("reorder(1,").count(), TOP, "{text}");
    }
}
