//! Analytic cost model, version 1.1 (spec §7.1): cache traffic from the loop footprints plus a
//! roofline whose compute ceiling is the lower of the FMA and the load/store ceiling.
//! Deterministic; the machine numbers are inputs (assumed, or read from a results file).
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
//! Traffic past a cache of `S` bytes: the innermost level `p` with `F(p) > S`; then every run
//! of loop `p` streams its footprint once: `F(p) * (iterations of loops 0..p)`. Loops inside
//! `p` fit, so what an iteration of `p` shares with the next one stays cached: an access
//! invariant in `p`, and the lines of an access that moves along `p` by less than a line
//! (v1.1 reloaded `F(p + 1)` per iteration of `p`, which charged both again every time). If no
//! level exceeds `S`, the traffic is `F(0)`, once. Each access is counted once, so a
//! written `C` is not charged for its write-back or write-allocate (as in `bench/peak`).
//!
//! Compute ceiling, see `mem_ops_per_fma`: the update nest keeps `C[i, j]` in memory, so a
//! good vectorized schedule is bound by loads and stores, not by FMA issue (v0 missed this).
//! v1.1: accesses held in registers across a loop are paid once per outer iteration, see
//! `mem_ops_per_fma`.

use crate::BenchError;
use crate::cli::MAX_N;
use crate::space::matmul_space;
use crate::stats::summarize;
use coffee_ten_ir::{Graph, Shape};
use coffee_ten_loop::{Access, AffineExpr, Expr, Kernel, Loop, LoopNest, Schedule, apply, lower};
use std::cmp::Ordering;
use std::fmt::Write as _;

/// The matmul update nest; nest 0 is the init nest.
const UPDATE_NEST: usize = 1;
/// Vector registers one access may keep across a loop, see `mem_ops_per_fma`. ASSUMED, not
/// measured: NEON has 32; the same number as `MAX_ACCUMULATORS` in `space.rs` (private there),
/// which leaves 8 for the operands.
const REG_BUDGET: usize = 24;
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
/// PLACEHOLDER, not measured: GFLOP/s of the `fma_mem_acc` kernel of `bench/peak` (every vector
/// FMA loads and stores its accumulator in L1). Replace by the `fma_mem_acc_gflops` median.
pub const PLACEHOLDER_MEM_ACC_GFLOPS: f64 = 60.0;
/// PLACEHOLDER, not measured: GFLOP/s of the `fma_mem_acc_scalar` kernel of `bench/peak` (the
/// same pattern in scalar code). Set to the vector placeholder / lanes (4), which is what the
/// model used for scalar nests before this point was measured. Replace by the
/// `fma_mem_acc_scalar_gflops` median.
pub const PLACEHOLDER_MEM_ACC_SCALAR_GFLOPS: f64 = PLACEHOLDER_MEM_ACC_GFLOPS / 4.0;
/// Memory ops per FMA of the `fma_mem_acc` kernels: load b, load c, store c.
const MEM_ACC_KERNEL_MEM_OPS: f64 = 3.0;

/// The machine the model predicts for. All fields are inputs, none is detected.
#[derive(Clone, Debug, PartialEq)]
pub struct Machine {
    /// Peak f32 GFLOP/s of vectorized code on one core.
    pub peak_gflops: f64,
    /// Memory bandwidth in GB/s (1e9 bytes) seen by one core.
    pub mem_gbs: f64,
    /// L2 to L1 bandwidth in GB/s (1e9 bytes) seen by one core.
    pub l2_gbs: f64,
    /// GFLOP/s of vector FMAs that each load and store an accumulator in L1 (the
    /// `fma_mem_acc` kernel, `MEM_ACC_KERNEL_MEM_OPS` memory ops per FMA): the load/store
    /// calibration point.
    pub mem_acc_gflops: f64,
    /// GFLOP/s of the same pattern in scalar code (the `fma_mem_acc_scalar` kernel, the
    /// instruction shape of the naive inner loop): the load/store calibration point of scalar
    /// nests.
    pub mem_acc_scalar_gflops: f64,
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
            mem_acc_gflops: PLACEHOLDER_MEM_ACC_GFLOPS,
            mem_acc_scalar_gflops: PLACEHOLDER_MEM_ACC_SCALAR_GFLOPS,
            l1_bytes: 128 * 1024,
            l2_bytes: 16 * 1024 * 1024,
            line_bytes: 128,
            lanes: 4,
        }
    }

    fn check(&self) -> Result<(), BenchError> {
        let speeds = [
            self.peak_gflops,
            self.mem_gbs,
            self.l2_gbs,
            self.mem_acc_gflops,
            self.mem_acc_scalar_gflops,
        ];
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
    /// The roofline term that sets `time_s`: "compute" (FMA issue), "ldst" (loads and stores
    /// per FMA), "memory" or "l2". A tie goes to the first of "compute"/"ldst", then "memory".
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

/// `all` without repeats, first occurrences kept in order.
fn distinct(all: Vec<&Access>) -> Vec<&Access> {
    let mut unique: Vec<&Access> = Vec::new();
    for access in all {
        if !unique.contains(&access) {
            unique.push(access);
        }
    }
    unique
}

/// Each access once, the store target first, then the loads in the order of the expression.
fn distinct_accesses(kernel_nest: &LoopNest) -> Vec<&Access> {
    let mut all = vec![&kernel_nest.body.target];
    loads(&kernel_nest.body.value, &mut all);
    distinct(all)
}

/// Distinct values of one index dim over the loops `from..` of `loops`.
fn distinct_values(
    index: &coffee_ten_loop::AffineExpr,
    loops: &[Loop],
    from: usize,
) -> Result<usize, BenchError> {
    distinct_values_where(index, loops, |pos| pos >= from)
}

/// Distinct values of one index dim over the loops `pos` of `loops` with `keep(pos)`.
fn distinct_values_where(
    index: &coffee_ten_loop::AffineExpr,
    loops: &[Loop],
    keep: impl Fn(usize) -> bool,
) -> Result<usize, BenchError> {
    let mut count = 1_usize;
    let mut span = 1_usize;
    for (name, coeff) in &index.terms {
        let pos = loops
            .iter()
            .position(|l| &l.name == name)
            .ok_or_else(|| model_err(format!("index names the unknown loop {name}")))?;
        if !keep(pos) || *coeff == 0 {
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
            let runs = extents[..p]
                .iter()
                .fold(1_usize, |acc, e| acc.saturating_mul(*e));
            totals[p].saturating_mul(runs)
        }
        None => totals[0],
    }
}

/// Coefficient of the loop `name` in `index` (0 if it does not appear).
fn coeff_of(index: &AffineExpr, name: &str) -> i64 {
    index
        .terms
        .iter()
        .filter(|(term, _)| term == name)
        .map(|(_, coeff)| coeff)
        .sum()
}

/// Memory ops one execution of `access` costs per iteration of `inner`. 0 if no index depends
/// on `inner` (hoisted out of the loop). Otherwise 1 (a vector op if `inner` is vectorized,
/// then only for unit stride: `inner` has coefficient 1 in the last dim and 0 in the others;
/// else a scalar op), or `lanes` scalar ops for a strided or gathered access in a vectorized loop.
fn access_ops(access: &Access, inner: &Loop, lanes: usize) -> usize {
    let coeffs: Vec<i64> = access
        .index
        .iter()
        .map(|index| coeff_of(index, &inner.name))
        .collect();
    let Some((last, others)) = coeffs.split_last() else {
        return 0;
    };
    if coeffs.iter().all(|c| *c == 0) {
        return 0;
    }
    let unit_stride = *last == 1 && others.iter().all(|c| *c == 0);
    if inner.vectorize && !unit_stride {
        lanes
    } else {
        1
    }
}

/// Vector registers an access needs to hold the elements it touches over the loops `from..`
/// of `loops`: the distinct values over the loops that are not vectorized, times, if the
/// vectorized (last) loop is inside, 1 if the access is invariant in it, `extent / lanes`
/// rounded up for unit stride and `extent` for a strided access (scalars).
fn registers_over(
    access: &Access,
    loops: &[Loop],
    from: usize,
    lanes: usize,
) -> Result<usize, BenchError> {
    let mut registers = 1_usize;
    for index in &access.index {
        let values =
            distinct_values_where(index, loops, |pos| pos >= from && !loops[pos].vectorize)?;
        registers = registers.saturating_mul(values);
    }
    if let Some(v) = loops.last().filter(|l| l.vectorize)
        && from < loops.len()
    {
        // `access_ops` of the vectorized loop: 0 invariant, 1 unit stride, `lanes` strided.
        registers = registers.saturating_mul(match access_ops(access, v, lanes) {
            0 => 1,
            1 => v.extent.div_ceil(lanes),
            _ => v.extent,
        });
    }
    Ok(registers)
}

/// Memory ops `access` costs over the whole update nest, loads of it or stores of it (the
/// same count for either; `C[i, j]` has both). Counted as iterations of the loops that are not
/// held across: a loop `p` is held across if
/// - the access is invariant in `p`, and
/// - the elements it touches over the loops inside `p` fit `REG_BUDGET` registers, and
/// - every loop inside `p` that the access depends on is unrolled or vectorized (a register
///   cannot be indexed by a loop variable; tested for the `C` tile in
///   `crates/coffee-ten-codegen-c/tests/unroll.rs`, where clang keeps nothing in registers
///   without the unroll steps).
///
/// Each loop that is not held across multiplies the count by its iterations (the vectorized
/// loop by `extent / lanes` rounded up, times `lanes` for a strided access).
fn total_ops(access: &Access, loops: &[Loop], lanes: usize) -> Result<f64, BenchError> {
    let mut ops = 1.0_f64;
    let mut rolled_dependence_inside = false;
    for (p, l) in loops.iter().enumerate().rev() {
        let depends = access
            .index
            .iter()
            .any(|index| coeff_of(index, &l.name) != 0);
        let held = !depends
            && !rolled_dependence_inside
            && registers_over(access, loops, p + 1, lanes)? <= REG_BUDGET;
        if depends && !l.unroll && !l.vectorize {
            rolled_dependence_inside = true;
        }
        if held {
            continue;
        }
        ops *= if l.vectorize {
            l.extent.div_ceil(lanes) as f64 * access_ops(access, l, lanes).max(1) as f64
        } else {
            l.extent as f64
        };
    }
    Ok(ops)
}

/// `(memory ops, FMA ops)` over the whole update `nest`. FMA ops: the product of the extents,
/// the vectorized loop as `extent / lanes` (rounded up) vector FMAs. Memory ops: `total_ops`
/// of each distinct Load of the stored value plus of the store.
fn nest_ops(nest: &LoopNest, lanes: usize) -> Result<(f64, f64), BenchError> {
    if lanes == 0 {
        return Err(model_err("lanes must be positive".to_string()));
    }
    if nest.loops.is_empty() {
        return Err(model_err("the nest has no loops".to_string()));
    }
    let mut all_loads = Vec::new();
    loads(&nest.body.value, &mut all_loads);
    let mut mem_ops = total_ops(&nest.body.target, &nest.loops, lanes)?;
    for load in distinct(all_loads) {
        mem_ops += total_ops(load, &nest.loops, lanes)?;
    }
    let fma_ops = nest.loops.iter().fold(1.0, |acc, l| {
        acc * if l.vectorize {
            l.extent.div_ceil(lanes) as f64
        } else {
            l.extent as f64
        }
    });
    Ok((mem_ops, fma_ops))
}

/// Memory ops `m` per FMA op of the update `nest`: the total memory ops over the nest divided
/// by the total FMA ops (vector FMAs if the last loop is vectorized), after registers took
/// the hoistable accesses out of the loops. Model v1.1.
///
/// Rule, see `total_ops`: an access is held in registers across a loop that it does not depend
/// on, if what it touches inside that loop fits `REG_BUDGET` vector registers; its ops are then
/// paid once per iteration of the loops outside, not per iteration of the held loop. v1 only
/// knew this for the innermost loop (counting the hoisted access as 0, and the store of an
/// invariant target as 1). Now the cost of a hoisted access is its loads over the loops
/// outside, so v1 hand values change by a small amount, e.g. the naive `i, k, j` has `C` load,
/// `B` load, `C` store per FMA and `A[i, k]` once per `j` loop: m = 3 + 1 / (iterations of j),
/// at N = 64: 3.015625 scalar, 3.0625 vectorized (16 vector iterations), no longer exactly 3.
/// The store of an invariant target is held too: `i, j, k` has `A`, `B` per FMA and `C` load
/// and store once per `k` loop: m = 2 + 2 / N.
///
/// Hand values, `lanes` = 4, N = 64 (`K` = 64 iterations of `k`, one tile = one `io, jo`):
/// - Register tile 4x16, `split i by 4; split j by 16; reorder io, jo, k, ii, ji; vectorize ji;
///   unroll ii; unroll ji`. `C` over `ii, ji` needs 4 * 4 = 16 <= 24 registers and is held
///   across `k`: 16 loads and 16 stores per tile. `A[i, k]` per `k` and `ii`: 4 scalar loads
///   (held across `ji`). `B[k, j]` per `k`: 4 vector loads. FMAs 4 * 4 per `k`. Per tile
///   (4 + 4) * K + 32 = 544 ops over 16 * K = 1024 FMAs: m = 0.53125, which is 0.5 + 32 / (16
///   K); the 0.5 of "A 4, B 4, FMA 16" is the limit of large `K`.
/// - 2x8: A 2 K, B 2 K, C 4 + 4: 264 / 256 = 1.03125. 8x8: A 8 K, B 2 K, C 16 + 16: 672 / 1024
///   = 0.65625.
/// - `tiled+vectorize`, `split i by 32; split j by 64; reorder io, jo, k, ii, ji; vectorize
///   ji`: `B[k, j]` over `ji` needs 16 registers and is held across `ii` (16 vector loads per
///   `k`); `C` over `ii, ji` needs 32 * 16 = 512 > 24 and `ii` is not unrolled: not held, 16
///   loads and 16 stores per `k` and `ii`; `A[i, k]` is held across `ji`: 1 load per `k` and
///   `ii`. Per `k` and `ii`: 16 + 16 + 1 + 16 / 32 = 33.5 over 16 FMAs: m = 2.09375, not 3.
/// - Naive `i, k, j` vectorized at N = 256: `B`, `C` over `j` need 64 registers (not held, they
///   depend on `j` anyway), `A` is held across `j`: per `i, k`: 64 + 64 + 64 + 1 = 193 over 64
///   FMAs: m = 3.015625.
///
/// NOT modelled: register pressure beyond `REG_BUDGET` per access (all accesses together may
/// need more than the 32 registers) and spills. ASSUMED: `REG_BUDGET` equals the space limit
/// `MAX_ACCUMULATORS` (24), which keeps the register tiles in registers. The `fma_mem_acc`
/// calibration below is unchanged.
///
/// Load/store ceiling, TWO calibration points, each for its own code shape, not a fit of the
/// model. The `fma_mem_acc` kernel does vector FMAs with m = 3 (load b, load c, store c) at
/// `mem_acc_gflops` = G. Its vector FMA rate is G / (2 * lanes), so the core retires
/// 3 * G / (2 * lanes) memory ops per second, if memory ops are what limits it. A vectorized
/// nest with m ops per FMA then runs `G / (2 * lanes) * 3 / m` FMAs per second, which is, at
/// 2 * lanes FLOP per vector FMA, `G * 3 / m` GFLOP/s. A scalar nest (last loop not
/// vectorized) uses the other point: the `fma_mem_acc_scalar` kernel, the same pattern in
/// scalar code with the instruction shape of the naive inner loop, at `mem_acc_scalar_gflops`
/// = S; at 2 FLOP per scalar FMA it gives `S * 3 / m` GFLOP/s. S is NOT G / lanes: measured on
/// P2, G = 46.94 and S = 8.563, not 11.7. ASSUMED: loads and stores share one pool; for a
/// vectorized nest nothing else (address arithmetic, the loop branch) limits; the scalar point
/// already contains that overhead of the naive loop shape, and every scalar nest is assumed to
/// pay the same.
pub fn mem_ops_per_fma(nest: &LoopNest, lanes: usize) -> Result<f64, BenchError> {
    let (mem_ops, fma_ops) = nest_ops(nest, lanes)?;
    Ok(mem_ops / fma_ops)
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
    // The load/store ceiling is derived at `mem_ops_per_fma`.
    let is_vectorized = nest.loops.last().is_some_and(|l| l.vectorize);
    let lanes = machine.lanes as f64;
    let mem_ops = mem_ops_per_fma(nest, machine.lanes)?;
    let (fma_gflops, mem_acc_gflops) = if is_vectorized {
        (machine.peak_gflops, machine.mem_acc_gflops)
    } else {
        (machine.peak_gflops / lanes, machine.mem_acc_scalar_gflops)
    };
    let ldst_gflops = mem_acc_gflops * MEM_ACC_KERNEL_MEM_OPS / mem_ops;
    let ceiling_gflops = fma_gflops.min(ldst_gflops);
    let compute_s = flops / (ceiling_gflops * 1e9);
    let memory_s = mem_bytes as f64 / (machine.mem_gbs * 1e9);
    let l2_s = l2_bytes as f64 / (machine.l2_gbs * 1e9);
    let time_s = compute_s.max(memory_s).max(l2_s);
    let bound = if compute_s >= memory_s && compute_s >= l2_s {
        if ldst_gflops < fma_gflops {
            "ldst"
        } else {
            "compute"
        }
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

/// Predicts the `n x n x n` matmul under the schedule text `schedule` ("" = naive).
fn predict_schedule(machine: &Machine, n: usize, schedule: &str) -> Result<Prediction, BenchError> {
    let schedule = Schedule::parse(schedule).map_err(|e| model_err(e.to_string()))?;
    let scheduled = apply(&matmul_kernel(n)?, &schedule).map_err(|e| model_err(e.to_string()))?;
    predict(machine, &scheduled)
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

const USAGE: &str = "usage: coffee-ten-bench model --n <N> [--from <results.jsonl>] [--peak <GFLOPS>] \
     [--mem-gbs <GBS>] [--l2-gbs <GBS>]  (flags after --from override it)\n       \
     coffee-ten-bench model compare --from <results.jsonl>";

/// The text after `"key":` in a JSON line. The log lines are flat, plus the `{"median":..}`
/// objects of the peak line, so this finds the key and is not a JSON parser.
fn after_key<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let pattern = format!("\"{key}\":");
    let start = line.find(&pattern)? + pattern.len();
    Some(&line[start..])
}

/// The number after `"key":`; `None` for a missing key or `null`.
fn number_after(line: &str, key: &str) -> Option<f64> {
    let rest = after_key(line, key)?;
    rest[..rest.find([',', '}']).unwrap_or(rest.len())]
        .trim()
        .parse()
        .ok()
}

/// `"median"` inside the object `"key":{...}`.
fn median_after(line: &str, key: &str) -> Option<f64> {
    let rest = after_key(line, key)?;
    number_after(&rest[..rest.find('}')?], "median")
}

/// The string after `"key":`, with `\"`, `\\`, `\n`, `\t` unescaped (as `json_string` writes).
fn string_after(line: &str, key: &str) -> Option<String> {
    let mut chars = after_key(line, key)?.strip_prefix('"')?.chars();
    let mut out = String::new();
    loop {
        match chars.next()? {
            '"' => return Some(out),
            '\\' => out.push(match chars.next()? {
                'n' => '\n',
                't' => '\t',
                other => other,
            }),
            c => out.push(c),
        }
    }
}

/// The machine of the `"kind":"peak"` line of a results file, and whether its `mem_acc` and
/// `mem_acc_scalar` are measured. Without `fma_mem_acc_gflops` in the line, `mem_acc_gflops`
/// stays the placeholder. Without `fma_mem_acc_scalar_gflops`, `mem_acc_scalar_gflops` is
/// `mem_acc_gflops / lanes` (the placeholder, or the measured vector point over lanes, as the
/// model did before the scalar point existed).
pub fn machine_from_results(text: &str) -> Result<(Machine, bool, bool), BenchError> {
    let line = text
        .lines()
        .find(|line| line.contains("\"kind\":\"peak\""))
        .ok_or_else(|| model_err("no peak line (\"kind\":\"peak\") in the results".to_string()))?;
    let median = |key: &str| {
        median_after(line, key)
            .ok_or_else(|| model_err(format!("the peak line has no {key} median")))
    };
    let mut machine = Machine {
        peak_gflops: median("fma_gflops")?,
        mem_gbs: median("triad_gbs")?,
        l2_gbs: median("triad_l2_gbs")?,
        ..Machine::assumed_m5()
    };
    let mem_acc = median_after(line, "fma_mem_acc_gflops");
    if let Some(gflops) = mem_acc {
        machine.mem_acc_gflops = gflops;
    }
    let mem_acc_scalar = median_after(line, "fma_mem_acc_scalar_gflops");
    machine.mem_acc_scalar_gflops =
        mem_acc_scalar.unwrap_or(machine.mem_acc_gflops / machine.lanes as f64);
    machine.check()?;
    Ok((machine, mem_acc.is_some(), mem_acc_scalar.is_some()))
}

/// The header note on where the machine numbers of `--from path` came from.
fn from_note(path: &str, mem_acc_measured: bool, mem_acc_scalar_measured: bool) -> String {
    let mut note = format!(
        "peak, mem, L2{}{} from {path}",
        if mem_acc_measured { ", mem_acc" } else { "" },
        if mem_acc_scalar_measured {
            ", mem_acc_scalar"
        } else {
            ""
        }
    );
    if !mem_acc_measured {
        let _ = write!(
            note,
            "; mem_acc {PLACEHOLDER_MEM_ACC_GFLOPS} is a PLACEHOLDER \
             (no fma_mem_acc_gflops in the peak line)"
        );
    }
    if !mem_acc_scalar_measured {
        note.push_str(
            "; mem_acc_scalar is a PLACEHOLDER, mem_acc / lanes \
             (no fma_mem_acc_scalar_gflops in the peak line)",
        );
    }
    note
}

fn read_results(path: &str) -> Result<String, BenchError> {
    std::fs::read_to_string(path).map_err(|e| model_err(format!("cannot read {path}: {e}")))
}

/// Parses the arguments after `model`.
pub fn parse_model_args<I: IntoIterator<Item = String>>(
    args: I,
) -> Result<(usize, Machine, String), BenchError> {
    let usage = |msg: String| BenchError::Usage(format!("{msg}\n{USAGE}"));
    let mut machine = Machine::assumed_m5();
    let mut source = "placeholders unless set on the command line".to_string();
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
            "--from" => {
                let (from, vector, scalar) = machine_from_results(&read_results(&value)?)?;
                machine = from;
                source = from_note(&value, vector, scalar);
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
    Ok((n, machine, source))
}

/// The report of `coffee-ten-bench model`: the machine numbers and the top `TOP` schedules.
pub fn report(n: usize, machine: &Machine, source: &str, ranked: &[Ranked]) -> String {
    let mut text = String::new();
    let _ = writeln!(
        text,
        "PREDICTION by the analytic model v1.1, not a measurement"
    );
    let _ = writeln!(
        text,
        "matmul {n}x{n}x{n}, update nest 1, {} candidates",
        ranked.len()
    );
    let _ = writeln!(
        text,
        "machine: peak {} GFLOP/s, mem {} GB/s, L2 {} GB/s, mem_acc {} GFLOP/s, \
         mem_acc_scalar {} GFLOP/s ({source}); L1 {} B, L2 {} B, line {} B, {} lanes (ASSUMED)",
        machine.peak_gflops,
        machine.mem_gbs,
        machine.l2_gbs,
        machine.mem_acc_gflops,
        machine.mem_acc_scalar_gflops,
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

/// Ranks of `values` from 1, ties get the mean of the ranks they span.
fn ranks(values: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    let mut ranks = vec![0.0; values.len()];
    let mut start = 0;
    while start < order.len() {
        let mut end = start;
        while end + 1 < order.len() && values[order[end + 1]] == values[order[start]] {
            end += 1;
        }
        let mean_rank = (start + end) as f64 / 2.0 + 1.0;
        for &i in &order[start..=end] {
            ranks[i] = mean_rank;
        }
        start = end + 1;
    }
    ranks
}

/// Spearman's rank correlation: Pearson's of the tie-averaged ranks. `None` for fewer than
/// two pairs or if one side is constant (zero rank variance).
pub fn spearman(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.len() != y.len() || x.len() < 2 {
        return None;
    }
    let (rx, ry) = (ranks(x), ranks(y));
    let mean = (x.len() as f64 + 1.0) / 2.0;
    let cov: f64 = rx
        .iter()
        .zip(&ry)
        .map(|(a, b)| (a - mean) * (b - mean))
        .sum();
    let var = |r: &[f64]| r.iter().map(|a| (a - mean) * (a - mean)).sum::<f64>();
    let denom = (var(&rx) * var(&ry)).sqrt();
    (denom > 0.0).then(|| cov / denom)
}

/// `schedule` without the nest number and extra spaces; "(naive)" for "".
fn short_schedule(schedule: &str) -> String {
    if schedule.is_empty() {
        return "(naive)".to_string();
    }
    schedule
        .replace("(1, ", "(")
        .replace("; ", ";")
        .replace(", ", ",")
}

/// The report of `coffee-ten-bench model compare`: the model against the `matmul` records of
/// the results file `text` (`source` is its name) that are not flagged `drift`.
pub fn compare_report(source: &str, text: &str) -> Result<String, BenchError> {
    let (machine, mem_acc_measured, mem_acc_scalar_measured) = machine_from_results(text)?;
    let (mut drifted, mut other_kernels) = (0, 0);
    let (mut measured, mut predicted, mut rows) = (Vec::new(), Vec::new(), String::new());
    for (number, line) in text.lines().enumerate().map(|(i, line)| (i + 1, line)) {
        if line.trim().is_empty() || line.contains("\"kind\":\"peak\"") {
            continue;
        }
        let bad = |what: &str| model_err(format!("{source} line {number}: no {what}"));
        let kernel = string_after(line, "kernel").ok_or_else(|| bad("kernel"))?;
        // relu-matmul-bias has extra nests (the epilogue), which the model does not predict.
        if kernel != "matmul" {
            other_kernels += 1;
            continue;
        }
        let drift = after_key(line, "drift").ok_or_else(|| bad("drift"))?;
        if drift.starts_with("true") {
            drifted += 1;
            continue;
        }
        let n = number_after(line, "n").ok_or_else(|| bad("n"))? as usize;
        let schedule = string_after(line, "schedule").ok_or_else(|| bad("schedule"))?;
        let median = number_after(line, "median").ok_or_else(|| bad("median"))?;
        let prediction = predict_schedule(&machine, n, &schedule)?;
        let _ = writeln!(
            rows,
            "{n:>5} {median:>9.3} {:>9.3} {:>6.2} {:>7}  {}",
            prediction.gflops,
            prediction.gflops / median,
            prediction.bound,
            short_schedule(&schedule)
        );
        measured.push(median);
        predicted.push(prediction.gflops);
    }
    let abs_log_ratios: Vec<f64> = predicted
        .iter()
        .zip(&measured)
        .map(|(p, m)| (p / m).ln().abs())
        .collect();
    let mut out = String::new();
    let _ = writeln!(
        out,
        "analytic model v1.1 (PREDICTION) against the measurements in {source}"
    );
    let _ = writeln!(
        out,
        "machine: peak {} GFLOP/s, mem {} GB/s, L2 {} GB/s, mem_acc {} GFLOP/s, \
         mem_acc_scalar {} GFLOP/s ({})",
        machine.peak_gflops,
        machine.mem_gbs,
        machine.l2_gbs,
        machine.mem_acc_gflops,
        machine.mem_acc_scalar_gflops,
        from_note(source, mem_acc_measured, mem_acc_scalar_measured)
    );
    let _ = writeln!(
        out,
        "skipped: {drifted} matmul records flagged drift; {other_kernels} records of other \
         kernels (relu-matmul-bias: its extra nests are not modelled yet)"
    );
    let _ = writeln!(
        out,
        "{:>5} {:>9} {:>9} {:>6} {:>7}  schedule (nest number dropped)",
        "n", "measured", "predicted", "ratio", "bound"
    );
    out.push_str(&rows);
    let _ = writeln!(
        out,
        "rates in GFLOP/s; ratio = predicted / measured; {} records",
        measured.len()
    );
    match (spearman(&measured, &predicted), summarize(&abs_log_ratios)) {
        (Some(rho), Some(stats)) => {
            let _ = writeln!(
                out,
                "Spearman rank correlation {rho:.3}; median |ln(predicted/measured)| {:.3} \
                 (a factor {:.2})",
                stats.median,
                stats.median.exp()
            );
        }
        _ => {
            let _ = writeln!(
                out,
                "Spearman rank correlation: undefined (too few or constant values)"
            );
        }
    }
    Ok(out)
}

/// `coffee-ten-bench model ...`: ranks the schedule space, or with `compare` checks the model
/// against a results file; measures nothing.
pub fn cli_main<I: IntoIterator<Item = String>>(args: I) -> std::process::ExitCode {
    let mut args = args.into_iter().peekable();
    let compare = args.peek().map(String::as_str) == Some("compare");
    if compare {
        args.next();
    }
    let mut run = || -> Result<String, BenchError> {
        if compare {
            return match args.by_ref().collect::<Vec<_>>().as_slice() {
                [flag, path] if flag == "--from" => compare_report(path, &read_results(path)?),
                _ => Err(BenchError::Usage(format!(
                    "model compare needs exactly --from <results.jsonl>\n{USAGE}"
                ))),
            };
        }
        let (n, machine, source) = parse_model_args(args.by_ref())?;
        Ok(report(n, &machine, &source, &rank_matmul(&machine, n)?))
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
    /// - L1 traffic = F(k) x (iterations of i) = 4202496 x 1024 = 4303355904: per `i`, the row
    ///   of C stays in L1 across `k`, the row of A is read along `k` (a line serves 32 values of
    ///   `k`), and B is streamed whole. (v1.1 charged F(j) per `k`: 8724152320, 2x too much.)
    /// - all loops: 3 x 4 MiB = 12582912 B <= L2 = 16 MiB, so memory traffic is that once.
    #[test]
    fn the_naive_nest_at_n_1024_reloads_from_l2_but_not_from_memory() {
        let p = predict(&Machine::assumed_m5(), &naive(1024)).expect("predicts");
        assert_eq!(p.l2_bytes, 4_303_355_904);
        assert_eq!(p.mem_bytes, 12_582_912);
    }

    /// Tiled `io, jo, k, ii, ji` (32, 64), N = 1024: extents 32, 16, 1024, 32, 64.
    /// - loop ii: C 32 x 256 = 8192; A 32 x 128 = 4096; B 256 -> 12544, fits L1.
    /// - loop k: C 8192; A 32 x 4096 = 131072; B 1024 x 256 = 262144 -> 401408 > L1.
    /// - L1 traffic = F(k) x (io, jo iterations) = 401408 x 32 x 16 = 205520896: per tile the
    ///   `k` loop streams A and B once and keeps C. (v1.1 charged F(ii) per `k`: 6576668672.)
    /// - memory: 3 x 4 MiB fits L2, once. The naive nest moves 4303355904 past L1, 21x more.
    #[test]
    fn tiling_moves_less_from_l2_than_the_naive_nest() {
        let machine = Machine::assumed_m5();
        let tiled = predict(&machine, &scheduled(1024, TILED)).expect("predicts");
        let plain = predict(&machine, &naive(1024)).expect("predicts");
        assert_eq!(tiled.l2_bytes, 205_520_896);
        assert_eq!(tiled.mem_bytes, 12_582_912);
        assert!(tiled.l2_bytes < plain.l2_bytes);
    }

    /// `TILED` + `vectorize(ji)` at N = 512: extents 16, 8, 512, 32, 64; same traffic as
    /// unvectorized. Line 128 B; the 64 floats of `ji` are 2 lines.
    /// - loop ii: C 32 x 256 = 8192; A 32 x 128 = 4096; B 256 -> 12544, fits L1.
    /// - loop k: C 8192; A 32 rows x (512 el = 2048 B) = 65536; B 512 rows x 256 = 131072 ->
    ///   204800 > L1 = 131072.
    /// - L1 traffic = 204800 x (16 x 8 tiles) = 26214400. Per tile that is the whole footprint
    ///   once: v1.1 charged F(ii) = 12544 per `k` (822083584 over all tiles), counting the C
    ///   tile and 32 lines of A again at every `k`.
    /// - memory: 3 x 1 MiB fits L2, once.
    #[test]
    fn tiled_vectorize_at_n_512_streams_its_k_loop_once_per_tile() {
        let kernel = scheduled(512, &format!("{TILED}; vectorize(1, ji)"));
        let p = predict(&Machine::assumed_m5(), &kernel).expect("predicts");
        assert_eq!(p.l2_bytes, 26_214_400);
        assert_eq!(p.mem_bytes, 3 * 512 * 512 * 4);
    }

    /// `TILED` + `vectorize(ji)` at N = 1024: the footprints are those of the test above at
    /// N = 1024, 401408 over L1, extents 32, 16 tiles: 401408 x 512 = 205520896; memory
    /// 12582912. The model has no cost for the 4 KiB row stride (cache sets, TLB), see the
    /// S3 measurement of this schedule: 26.7 against 56.0 GFLOP/s at N = 512.
    #[test]
    fn tiled_vectorize_at_n_1024_has_the_traffic_of_the_scalar_tiles() {
        let kernel = scheduled(1024, &format!("{TILED}; vectorize(1, ji)"));
        let p = predict(&Machine::assumed_m5(), &kernel).expect("predicts");
        assert_eq!(p.l2_bytes, 205_520_896);
        assert_eq!(p.mem_bytes, 12_582_912);
    }

    /// Register tile 4x16, N = 1024: `split i by 4; split j by 16; reorder io, jo, k, ii, ji;
    /// vectorize ji; unroll ii; unroll ji`. Extents 256, 64, 1024, 4, 16; line 128 B = 32
    /// floats, so the 16 floats of `ji` (64 B) take a whole line.
    /// - loop ii: C 4 x 128 = 512; A 4 x 128 = 512; B 128 -> 1152, fits L1.
    /// - loop k: C 512; A 4 rows x (1024 el = 4096 B) = 16384; B 1024 rows x 128 = 131072 ->
    ///   147968, over L1 = 131072 by 16896 B. Without the rounding to whole lines B would be
    ///   64 KiB, the tile 80 KiB and it would fit; but the other half of each line of B belongs
    ///   to the next `jo`, 128 KiB of L1 away, so the rounding is the right capacity.
    /// - L1 traffic = 147968 x (256 x 64 tiles) = 2424307712 (2.4 GB; on the P1 machine, 125 GB/s
    ///   and 126.7 GFLOP/s: 19.4 ms against 16.9 ms of FMAs, so "l2" at 110.7 GFLOP/s). v1.1 charged F(ii) = 1152 per
    ///   `k`, 19327352832, which is 13.9 GFLOP/s: C is invariant in `k` and a line of A serves 32
    ///   values of `k`, neither is read again from L2 per `k`.
    /// - memory: 3 x 4 MiB fits L2, once.
    #[test]
    fn the_register_tile_at_n_1024_streams_its_k_loop_once_per_tile() {
        let tile = "split(1, i, io, ii, 4); split(1, j, jo, ji, 16); \
                    reorder(1, io, jo, k, ii, ji); vectorize(1, ji); unroll(1, ii); unroll(1, ji)";
        let p = predict(&Machine::assumed_m5(), &scheduled(1024, tile)).expect("predicts");
        assert_eq!(p.l2_bytes, 2_424_307_712);
        assert_eq!(p.mem_bytes, 12_582_912);
    }

    #[test]
    fn the_cache_size_decides_the_level_that_is_streamed() {
        // Footprints 10, 100, 1000 for 3 loops of extents 4, 5, 6 (outer first), body 1.
        let totals = [1000, 100, 10, 1];
        let extents = [4, 5, 6];
        assert_eq!(traffic_past(&totals, &extents, 1000), 1000);
        // The outermost loop does not fit: it streams its footprint, once.
        assert_eq!(traffic_past(&totals, &extents, 999), 1000);
        assert_eq!(traffic_past(&totals, &extents, 99), 100 * 4);
        assert_eq!(traffic_past(&totals, &extents, 9), 10 * 4 * 5);
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
            // Register tiles have no unvectorized twin: factors 2 and 4 are not tile factors.
            if text.contains("unroll(") {
                continue;
            }
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
        // A huge mem_acc takes the load/store ceiling out, leaving the FMA ceiling of v0.
        let machine = Machine {
            mem_acc_gflops: 1e9,
            mem_acc_scalar_gflops: 1e9,
            ..Machine::assumed_m5()
        };
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
        let (n, machine, _) = parse(&["--n", "256"]).expect("parses");
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
        let (_, set, _) = parse(&[
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
        let ranked = rank_matmul(&machine, 16).expect("ranks");
        let text = report(16, &machine, "test", &ranked);
        assert!(text.contains("not a measurement"), "{text}");
        assert_eq!(text.matches("reorder(1,").count(), TOP, "{text}");
    }

    fn mem_ops(n: usize, text: &str) -> f64 {
        let kernel = scheduled(n, text);
        mem_ops_per_fma(update_nest(&kernel), Machine::assumed_m5().lanes).expect("counts")
    }

    /// Naive nest `i, k, j`: per FMA C[i,j] load, B[k,j] load, C[i,j] store, and A[i,k] once per
    /// `j` loop (held across it in a register): m = 3 + 1 / (iterations of j). Scalar at N = 64
    /// and vectorized at N = 256 (4 lanes) both iterate j 64 times: 3 + 1/64 = 3.015625. C and
    /// B over j need 64 and 16 * 4 registers at most, and depend on j, so they are not held.
    /// (Vectorized at N = 64, a row of C is 16 registers and IS held across k: m = 1.09375.)
    #[test]
    fn the_naive_nest_has_about_three_memory_ops_per_fma_scalar_or_vector() {
        assert_eq!(mem_ops(64, ""), 3.015625);
        assert_eq!(mem_ops(256, "vectorize(1, j)"), 3.015625);
        assert_eq!(mem_ops(64, TILED), 3.015625);
        assert_eq!(mem_ops(64, "vectorize(1, j)"), 1.09375);
    }

    /// `i, j, k`: A[i,k] and B[k,j] depend on k: 2 per FMA. C[i,j] is invariant in k and one
    /// register: held across k, its load and store are paid once per `k` loop: m = 2 + 2 / 64.
    /// (v1 counted the store in every iteration, m = 3, because no measurement showed that
    /// clang sinks it; the register tile and `tiled+vectorize` measurements say it does.)
    #[test]
    fn an_invariant_target_is_held_across_the_innermost_loop() {
        assert_eq!(mem_ops(64, "reorder(1, i, j, k)"), 2.03125);
    }

    /// `i, k, j` vectorized on j (4 lanes), N = 256, with the load B[k, j] turned into B[j, k]
    /// by hand (`apply` cannot make a strided load here): B is strided in j, 4 scalar ops per
    /// vector iteration. Per vector iteration C load 1, B 4, C store 1; A once per j loop (64
    /// vector iterations). m = (6 * 64 + 1) / 64 = 6.015625.
    #[test]
    fn a_strided_load_in_a_vectorized_loop_costs_one_op_per_lane() {
        let mut kernel = scheduled(256, "vectorize(1, j)");
        let Expr::Binary(_, _, product) = &mut kernel.nests[UPDATE_NEST].body.value else {
            panic!("C + A * B");
        };
        let Expr::Binary(_, _, b) = product.as_mut() else {
            panic!("A * B");
        };
        let Expr::Load(b) = b.as_mut() else {
            panic!("load B");
        };
        b.index.reverse();
        assert_eq!(mem_ops_per_fma(update_nest(&kernel), 4), Ok(6.015625));
    }

    /// Register tiles at N = 64 (K = 64 iterations of k, 4 lanes). C over `ii, ji` fits 24
    /// registers and is held across k: its loads and stores are paid once per tile. Per tile:
    /// - 4x16: A 4 K, B 4 K, C 16 + 16: (8 * 64 + 32) / (16 * 64) = 544 / 1024 = 0.53125. The
    ///   0.5 of "A 4, B 4, FMA 16 per k" is the limit of large K.
    /// - 2x8: A 2 K, B 2 K, C 4 + 4: 264 / 256 = 1.03125.
    /// - 8x8: A 8 K, B 2 K, C 16 + 16: 672 / 1024 = 0.65625.
    #[test]
    fn register_tiles_pay_a_k_th_of_the_tile_and_hold_c() {
        let tile = |ri: usize, rj: usize| {
            format!(
                "split(1, i, io, ii, {ri}); split(1, j, jo, ji, {rj}); \
                 reorder(1, io, jo, k, ii, ji); vectorize(1, ji); unroll(1, ii); unroll(1, ji)"
            )
        };
        assert_eq!(mem_ops(64, &tile(4, 16)), 0.53125);
        assert_eq!(mem_ops(64, &tile(2, 8)), 1.03125);
        assert_eq!(mem_ops(64, &tile(8, 8)), 0.65625);
    }

    /// The same 4x16 tile without the unroll steps: `ii` is a loop that C depends on, so a
    /// register cannot hold C across k (clang keeps nothing, `tests/unroll.rs`). C: 16 + 16
    /// per k; A 4 per k (held across ji), B 4 per k: 40 / 16 = 2.5.
    #[test]
    fn a_tile_without_unroll_does_not_hold_c() {
        let rolled = "split(1, i, io, ii, 4); split(1, j, jo, ji, 16); \
                      reorder(1, io, jo, k, ii, ji); vectorize(1, ji)";
        assert_eq!(mem_ops(64, rolled), 2.5);
    }

    /// `tiled+vectorize` (32 x 64, N = 64): B over ji is 16 registers, held across ii: 16 loads
    /// per k. C over `ii, ji` is 32 * 16 = 512 registers, not held: 16 loads and 16 stores per
    /// k and ii. A[i,k] held across ji: 1 per k and ii. Per k and ii (16 FMAs):
    /// (16 + 16 + 1 + 16 / 32) / 16 = 33.5 / 16 = 2.09375.
    #[test]
    fn tiled_vectorized_holds_b_across_ii() {
        assert_eq!(mem_ops(64, &format!("{TILED}; vectorize(1, ji)")), 2.09375);
    }

    /// An unrolled loop that is not part of a tile, N = 4. `i, k, j` with `k` unrolled: `j`
    /// is the plain loop inside it, so nothing is held that was not before: same as without
    /// unroll, 3 + 1 / 4 = 3.25. `i, j, k` with `k` unrolled: A[i,k] (4 registers) is held
    /// across j: 4 * 4 = 16 loads; B[k,j] 64; C[i,j] held across k: 16 loads and 16 stores:
    /// 112 / 64 = 1.75. All three loops unrolled: A, B and C are 16 elements each and loaded
    /// once (stored once): 64 / 64 = 1.
    #[test]
    fn unrolled_loops_outside_a_tile_hold_what_fits() {
        assert_eq!(mem_ops(4, "unroll(1, k)"), 3.25);
        assert_eq!(mem_ops(4, ""), 3.25);
        assert_eq!(mem_ops(4, "reorder(1, i, j, k); unroll(1, k)"), 1.75);
        assert_eq!(mem_ops(4, "unroll(1, i); unroll(1, k); unroll(1, j)"), 1.0);
    }

    /// mem_acc = 60 and mem_acc_scalar = 60 / 4 = 15 (placeholders), vectorized at N = 256 and
    /// scalar at N = 64 both have m = 3.015625: vectorized 60 * 3 / m = 59.689..., scalar
    /// 15 * 3 / m = 14.922... Both are
    /// below the FMA ceilings 100 and 100 / 4 = 25, so they are the ld/st bound. A register tile
    /// (4x16, N = 64, m = 0.53125) has the ceiling 60 * 3 / m = 338.8, so the FMA peak binds.
    #[test]
    fn the_ldst_ceiling_follows_the_one_point_calibration() {
        // Fast caches, so that at N = 256 the L2 traffic does not bind.
        let machine = Machine {
            mem_gbs: 1e6,
            l2_gbs: 1e6,
            ..Machine::assumed_m5()
        };
        let m = 3.015625;
        let scalar = predict(&machine, &naive(64)).expect("predicts");
        let vector = predict(&machine, &scheduled(256, "vectorize(1, j)")).expect("predicts");
        assert!(
            (scalar.gflops - 15.0 * 3.0 / m).abs() < 1e-9,
            "{}",
            scalar.gflops
        );
        assert!(
            (vector.gflops - 60.0 * 3.0 / m).abs() < 1e-9,
            "{}",
            vector.gflops
        );
        assert_eq!((scalar.bound, vector.bound), ("ldst", "ldst"));
        let tile = "split(1, i, io, ii, 4); split(1, j, jo, ji, 16); \
                    reorder(1, io, jo, k, ii, ji); vectorize(1, ji); unroll(1, ii); unroll(1, ji)";
        let tiled = predict(&machine, &scheduled(64, tile)).expect("predicts");
        assert_eq!((tiled.gflops, tiled.bound), (100.0, "compute"));
    }

    /// The measured P2 points: mem_acc 46.9387, mem_acc_scalar 8.563 (not 46.9387 / 4 = 11.73).
    /// The naive `i, k, j` at N = 64 is scalar with m = 3.015625, so its ceiling is
    /// 8.563 * 3 / 3.015625 = 8.5186..., below the scalar FMA ceiling 100 / 4. The vectorized
    /// nest (N = 256, same m) uses mem_acc only: 46.9387 * 3 / 3.015625, whatever the scalar point.
    #[test]
    fn a_scalar_nest_uses_the_scalar_calibration_point() {
        let machine = Machine {
            mem_acc_gflops: 46.9387,
            mem_acc_scalar_gflops: 8.563,
            mem_gbs: 1e6,
            l2_gbs: 1e6,
            ..Machine::assumed_m5()
        };
        let m = 3.015625;
        let scalar = predict(&machine, &naive(64)).expect("predicts");
        let expected = 8.563 * 3.0 / m;
        assert!((scalar.gflops - expected).abs() < 1e-9, "{}", scalar.gflops);
        assert!((scalar.gflops - 8.5186).abs() < 1e-4, "{}", scalar.gflops);
        assert_eq!(scalar.bound, "ldst");
        let vector = predict(&machine, &scheduled(256, "vectorize(1, j)")).expect("predicts");
        let expected = 46.9387 * 3.0 / m;
        assert!((vector.gflops - expected).abs() < 1e-9, "{}", vector.gflops);
        let other = Machine {
            mem_acc_scalar_gflops: 1.0,
            ..machine
        };
        let again = predict(&other, &scheduled(256, "vectorize(1, j)")).expect("predicts");
        assert_eq!(again.gflops, vector.gflops);
    }

    /// With peak 40 the FMA ceilings are 40 (vectorized) and 40 / 4 = 10 (scalar), below the
    /// ld/st ceilings 60 and 15, so the FMA ceiling caps both.
    #[test]
    fn the_fma_peak_caps_the_ldst_ceiling() {
        let machine = Machine {
            peak_gflops: 40.0,
            ..Machine::assumed_m5()
        };
        let scalar = predict(&machine, &naive(64)).expect("predicts");
        let vector = predict(&machine, &scheduled(64, "vectorize(1, j)")).expect("predicts");
        assert!((scalar.gflops - 10.0).abs() < 1e-9, "{}", scalar.gflops);
        assert!((vector.gflops - 40.0).abs() < 1e-9, "{}", vector.gflops);
        assert_eq!((scalar.bound, vector.bound), ("compute", "compute"));
    }

    const S2: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../bench/results/2026-10-16-ffce49d.jsonl"
    );

    const S3: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../bench/results/2026-10-18-4928ffa.jsonl"
    );

    /// The S3 machine (mem_acc 31.1262 is from the old, loop-overhead-bound kernel), N = 256:
    /// the 6 register tiles are the top 6 of the space; the next schedule has no unroll. The
    /// 4x16 tile at N = 128 is bound by the FMA peak, not by loads and stores.
    #[test]
    fn register_tiles_rank_first_on_the_s3_machine() {
        let text = std::fs::read_to_string(S3).expect("the S3 file");
        let (machine, mem_acc_measured, scalar_measured) =
            machine_from_results(&text).expect("parses");
        assert!(mem_acc_measured && !scalar_measured);
        assert_eq!(machine.mem_acc_gflops, 31.1262);
        // No scalar field: the scalar point stays the vector one over lanes, as before.
        assert_eq!(machine.mem_acc_scalar_gflops, 31.1262 / 4.0);
        let ranked = rank_matmul(&machine, 256).expect("ranks");
        assert!(ranked[..6].iter().all(|r| r.schedule.contains("unroll(")));
        assert!(!ranked[6].schedule.contains("unroll("));
        let tile = "split(1, i, io, ii, 4); split(1, j, jo, ji, 16); \
                    reorder(1, io, jo, k, ii, ji); vectorize(1, ji); unroll(1, ii); unroll(1, ji)";
        let p = predict_schedule(&machine, 128, tile).expect("predicts");
        // flops / (flops / peak) rounds twice: a few ulp, far below 1e-9 relative.
        assert!(
            (p.gflops / machine.peak_gflops - 1.0).abs() < 1e-9,
            "{}",
            p.gflops
        );
        assert_eq!(p.bound, "compute");
    }

    /// The peak line of the committed S2 file has no `fma_mem_acc_gflops`.
    #[test]
    fn from_reads_the_committed_peak_line_and_keeps_the_placeholder_mem_acc() {
        let text = std::fs::read_to_string(S2).expect("the committed file");
        let (machine, mem_acc_measured, scalar_measured) =
            machine_from_results(&text).expect("parses");
        assert_eq!(
            (machine.peak_gflops, machine.mem_gbs, machine.l2_gbs),
            (122.0494, 107.8728, 120.3851)
        );
        assert_eq!(machine.mem_acc_gflops, PLACEHOLDER_MEM_ACC_GFLOPS);
        assert_eq!(
            machine.mem_acc_scalar_gflops,
            PLACEHOLDER_MEM_ACC_SCALAR_GFLOPS
        );
        assert_eq!(PLACEHOLDER_MEM_ACC_SCALAR_GFLOPS, 15.0);
        assert!(!mem_acc_measured && !scalar_measured);
        let args = ["--n", "8", "--from", S2, "--l2-gbs", "50"];
        let (_, from, source) = parse_model_args(args.map(String::from)).expect("parses");
        assert_eq!((from.peak_gflops, from.l2_gbs), (122.0494, 50.0));
        assert!(source.contains("PLACEHOLDER"), "{source}");
    }

    #[test]
    fn from_reads_mem_acc_when_the_peak_line_has_it() {
        let line = r#"{"kind":"peak","fma_gflops":{"median":100.5,"n":30},"triad_gbs":{"median":90.25,"n":30},"fma_mem_acc_gflops":{"median":61.5,"p25":60.0,"n":30},"triad_l2_gbs":{"median":110.0,"n":30}}"#;
        let (machine, mem_acc_measured, scalar_measured) =
            machine_from_results(line).expect("parses");
        assert_eq!(
            (machine.peak_gflops, machine.mem_gbs, machine.l2_gbs),
            (100.5, 90.25, 110.0)
        );
        assert_eq!(machine.mem_acc_gflops, 61.5);
        assert_eq!(machine.mem_acc_scalar_gflops, 61.5 / 4.0);
        assert!(mem_acc_measured && !scalar_measured);
        assert!(machine_from_results(r#"{"kernel":"matmul"}"#).is_err());
        assert!(machine_from_results(r#"{"kind":"peak","fma_gflops":{"median":1.0}}"#).is_err());
    }

    #[test]
    fn from_reads_mem_acc_scalar_when_the_peak_line_has_it() {
        let line = r#"{"kind":"peak","fma_gflops":{"median":100.5,"n":30},"triad_gbs":{"median":90.25,"n":30},"fma_mem_acc_gflops":{"median":61.5,"n":30},"fma_mem_acc_scalar_gflops":{"median":8.5632,"n":30},"triad_l2_gbs":{"median":110.0,"n":30}}"#;
        let (machine, mem_acc_measured, scalar_measured) =
            machine_from_results(line).expect("parses");
        assert_eq!(
            (machine.mem_acc_gflops, machine.mem_acc_scalar_gflops),
            (61.5, 8.5632)
        );
        assert!(mem_acc_measured && scalar_measured);
        let note = from_note("f", mem_acc_measured, scalar_measured);
        assert!(!note.contains("PLACEHOLDER"), "{note}");
        let note = from_note("f", true, false);
        assert!(note.contains("mem_acc_scalar is a PLACEHOLDER"), "{note}");
        // The report header carries the scalar point.
        let header = report(8, &machine, &note, &[]);
        assert!(header.contains("mem_acc_scalar 8.5632 GFLOP/s"), "{header}");
    }

    #[test]
    fn the_json_helpers_read_numbers_and_escaped_strings() {
        let line = r#"{"s":"a\"b\\c","n":64,"x":null,"obj":{"median":2.5},"drift":false}"#;
        assert_eq!(string_after(line, "s").as_deref(), Some("a\"b\\c"));
        assert_eq!(number_after(line, "n"), Some(64.0));
        assert_eq!(number_after(line, "x"), None);
        assert_eq!(number_after(line, "missing"), None);
        assert_eq!(median_after(line, "obj"), Some(2.5));
    }

    #[test]
    fn spearman_is_one_for_the_same_order_and_minus_one_for_the_reverse() {
        let x = [1.0, 5.0, 7.0, 20.0];
        assert_eq!(spearman(&x, &[0.1, 0.2, 0.3, 0.4]), Some(1.0));
        assert_eq!(spearman(&x, &[4.0, 3.0, 2.0, 1.0]), Some(-1.0));
        assert_eq!(spearman(&x, &[1.0, 1.0, 1.0, 1.0]), None);
        assert_eq!(spearman(&[1.0], &[1.0]), None);
    }

    /// x = [1, 2, 3], y = [1, 1, 2]: ranks of y are [1.5, 1.5, 3] (the tie shares ranks 1 and
    /// 2). Centered at the mean rank 2: dx = [-1, 0, 1], dy = [-0.5, -0.5, 1]; cov = 1.5,
    /// var x = 2, var y = 1.5, rho = 1.5 / sqrt(3) = 0.8660254. Ranks 1, 2, 3 for y (no tie
    /// averaging) would give 1.
    #[test]
    fn spearman_averages_the_ranks_of_ties() {
        assert_eq!(ranks(&[10.0, 20.0, 20.0, 30.0]), [1.0, 2.5, 2.5, 4.0]);
        let rho = spearman(&[1.0, 2.0, 3.0], &[1.0, 1.0, 2.0]).expect("defined");
        assert!((rho - 1.5 / 3.0_f64.sqrt()).abs() < 1e-12, "{rho}");
    }

    /// 15 matmul records, 2 flagged drift: 13 rows. The 15 relu-matmul-bias records are skipped.
    #[test]
    fn compare_on_the_committed_s2_file_prints_13_matmul_records() {
        let text = std::fs::read_to_string(S2).expect("the committed file");
        let report = compare_report("S2", &text).expect("compares");
        assert!(report.contains("13 records"), "{report}");
        assert!(report.contains("skipped: 2 matmul records flagged drift; 15 records"));
        assert!(report.contains("PLACEHOLDER"), "{report}");
        assert!(report.contains("Spearman rank correlation 0."), "{report}");
        let rows = report
            .lines()
            .filter(|l| l.contains("ldst") || l.contains("l2"));
        assert_eq!(rows.count(), 13, "{report}");
    }
}
