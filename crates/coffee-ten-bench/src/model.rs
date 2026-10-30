//! Analytic cost model, version 1.8 (spec §7.1): cache traffic from the loop footprints plus a
//! roofline whose compute ceiling is the lowest of the FMA, the load, the store (vectorized
//! nests; scalar nests have one combined load/store ceiling) and the latency ceiling (dependent FMA chains of a reduction held in registers, see `latency_gflops`).
//! v1.6 adds one more scalar ceiling, "reload" (`column_reload`), v1.7 another, "colload"
//! (`column_load_unrolled`). v1.8 follows the codegen of #96: a short power-of-two `vectorize`
//! is one vector iteration, not scalar code (`one_vector_iteration`). Deterministic; the machine
//! numbers are inputs (assumed, or read from a results file).
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
use coffee_ten_loop::{
    Access, AffineExpr, Buffer, Expr, Kernel, Loop, LoopNest, Schedule, apply, lower,
};
use std::cmp::Ordering;
use std::fmt::Write as _;

/// The matmul update nest; nest 0 is the init nest.
const UPDATE_NEST: usize = 1;
/// Vector registers one access may keep across a loop, see `mem_ops_per_fma`. ASSUMED, not
/// measured: NEON has 32; the same number as `MAX_ACCUMULATORS` in `space.rs` (private there),
/// which leaves 8 for the operands.
const REG_BUDGET: usize = 24;
/// Trip count up to which clang fully unrolls a loop marked `vectorize(enable)` (and not
/// unrolled by the schedule) before its loop vectorizer sees it, under `BENCH_FLAGS`: the pragma
/// then gets no "vectorized loop" remark and the code is scalar. ASSUMED threshold, found with
/// Apple clang 21: 16 iterations get no remark, 17 and 32 do. This MUST change together with
/// `tests/full_unroll.rs` in `coffee-ten-codegen-c`, which pins it, and with
/// `CLANG_EARLY_UNROLL_LIMIT` in its `lib.rs` (the same 16, #96), see `one_vector_iteration`.
const CLANG_FULL_UNROLL_LIMIT: usize = 16;
/// Size (in LLVM instruction units) up to which clang fully unrolls a loop whose trip count
/// is known: its `unroll-threshold` at -O2. See `fully_unrolled`.
const CLANG_UNROLL_SIZE: usize = 150;
/// Size of the body of an innermost loop, in the units of `CLANG_UNROLL_SIZE`: the largest
/// whole number with `CLANG_FULL_UNROLL_LIMIT` * size <= `CLANG_UNROLL_SIZE` (16 * 9 = 144; 17
/// iterations would be 153), which is what the pinned 16 / 17 iterations say.
const INNERMOST_LOOP_SIZE: usize = CLANG_UNROLL_SIZE / CLANG_FULL_UNROLL_LIMIT;
/// What each loop level adds to the size of the loop around it. ASSUMED, read from the IR of the
/// 1000 random schedules of hypothesis A run 2 at N = 256 (clang remarks "completely unrolled
/// loop" against `fully_unrolled`): 5 gets 5758 of 5789 loops right (all misses are loops of
/// 2 iterations that clang unrolls although the size says no), 4 and 6 get 5571 and 5730.
const LOOP_LEVEL_SIZE: usize = 5;
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
/// PLACEHOLDER, from a hand microbenchmark (Apple M5, `BENCH_FLAGS`, 1024 floats, L1-resident):
/// GFLOP/s of the `fma_mem_acc_scalar` access pattern (load b, load c, FMA, store c) when the
/// loop body is straight-line code, here 16 FMAs per iteration: 23.2. The same pattern as the
/// naive loop, one FMA per iteration, runs at 8.55 (`fma_mem_acc_scalar`); the rolled loop is
/// limited by its iterations, not by the load/store ports. Replace by the
/// `fma_mem_acc_scalar_unrolled_gflops` median of the peak line.
pub const PLACEHOLDER_MEM_ACC_SCALAR_UNROLLED_GFLOPS: f64 = 23.2;
/// PLACEHOLDER, from a hand microbenchmark (Apple M5, `BENCH_FLAGS`): GFLOP/s of the
/// `fma_col_reload` kernel of `bench/peak` (the scalar matmul nest `io, j, k, ii`, `ii` = 64
/// innermost and rolled, `C[io*64+ii][j] += A[io*64+ii][k] * B[k][j]` at N = 256, all row strides
/// 1 KiB): 2.1. The P6 peak run measured 2.121. Replace by the `fma_col_reload_gflops` median.
pub const PLACEHOLDER_COL_RELOAD_GFLOPS: f64 = 2.1;
/// PLACEHOLDER, from a hand microbenchmark (Apple M5, `BENCH_FLAGS`): GFLOP/s of the
/// `fma_col_unrolled` kernel of `bench/peak` (the scalar matmul nest `io, j, ko, ki, ii`, `ii` = 16
/// unrolled by clang under a rolled `ki` = 64, `C` held in registers across `ki`, `A[io*16+ii][k]`
/// loaded down a column, all row strides 1 KiB, N = 256): 15.7. The P7 peak run measured 15.624.
/// Replace by the `fma_col_unrolled_gflops` median.
pub const PLACEHOLDER_COL_UNROLLED_GFLOPS: f64 = 15.7;
/// Byte stride of the stored access along the innermost loop, in multiples of which the
/// column-reload ceiling applies, see `column_reload`. A hardware observation of hand
/// microbenchmarks (Apple M5, the `fma_col_reload` nest with `ii` = 32, N = 256, only the row
/// stride of `C` varied; GFLOP/s): a stride that is a multiple of 256 B runs at 1.6 to 2.1; 128 B modulo 256 B at 3.2;
/// 64 B at 4.7; 8 to 32 B at about 5; 272 floats (1088 B) at 7.7. Not fitted to the
/// hypothesis A measurements.
const COL_RELOAD_STRIDE_BYTES: i64 = 256;
/// PLACEHOLDER: GFLOP/s of one dependent scalar FMA chain (the `fma_chain` kernel of
/// `bench/peak`). Hand microbenchmark on an Apple M5: 0.76 ns per FMA. Replace by the
/// `fma_chain_gflops` median.
pub const PLACEHOLDER_FMA_CHAIN_GFLOPS: f64 = 2.6;
/// PLACEHOLDER: GFLOP/s of the naive scalar reduction split into independent chains of
/// `CHAIN_CALIBRATION_LENGTH` FMAs (the `fma_chain256` kernel of `bench/peak`). Hand
/// microbenchmark on an Apple M5. Replace by the `fma_chain256_gflops` median.
pub const PLACEHOLDER_FMA_CHAIN256_GFLOPS: f64 = 3.49;
/// Chain length at which `fma_chain256_gflops` is measured, and so the window `W` is calibrated.
const CHAIN_CALIBRATION_LENGTH: f64 = 256.0;
/// PLACEHOLDER, not measured in a quiet run: GFLOP/s of vector FMAs that each store one result
/// and load nothing (the `fma_store` kernel of `bench/peak`). The value is the `--quick` smoke
/// run of that kernel on an Apple M5 (58.0 / 58.4 / 58.5), not a median of a full run. Replace
/// by the `fma_store_gflops` median.
pub const PLACEHOLDER_STORE_GFLOPS: f64 = 58.0;
/// Memory ops per FMA of the `fma_mem_acc` kernels: load b, load c, store c.
const MEM_ACC_KERNEL_MEM_OPS: f64 = 3.0;
/// Loads per FMA of the `fma_mem_acc` kernel (load b, load c): what its rate is a load rate of.
const MEM_ACC_KERNEL_LOADS: f64 = 2.0;
/// Stores per FMA of the store kernel behind `store_gflops`.
const STORE_KERNEL_STORES: f64 = 1.0;

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
    /// GFLOP/s of the same pattern in straight-line scalar code (the
    /// `fma_mem_acc_scalar_unrolled` kernel, a body of 16 FMAs per iteration): the load/store
    /// calibration point of scalar nests whose innermost loop is unrolled, see
    /// `mem_ops_per_fma`.
    pub mem_acc_scalar_unrolled_gflops: f64,
    /// GFLOP/s of the `fma_col_reload` kernel (scalar, rolled, 2 loads and a store per FMA, the
    /// stored column reloaded at the next iteration of the loop outside): the ceiling of scalar
    /// nests for which `column_reload` holds.
    pub col_reload_gflops: f64,
    /// GFLOP/s of the `fma_col_unrolled` kernel (scalar, the innermost loop unrolled under a rolled
    /// one, 17 loads per 16 FMA and no stores, 16 of the loads down a column): the ceiling of
    /// scalar nests for which `column_load_unrolled` holds.
    pub col_unrolled_gflops: f64,
    /// GFLOP/s of vector FMAs that each store one result and load nothing (`STORE_KERNEL_STORES`
    /// store per FMA): the store calibration point of vectorized nests.
    pub store_gflops: f64,
    /// GFLOP/s of one dependent scalar FMA chain (the `fma_chain` kernel): the latency
    /// calibration point, see `latency_gflops`.
    pub fma_chain_gflops: f64,
    /// GFLOP/s of a reduction split into independent chains of 256 scalar FMAs (the
    /// `fma_chain256` kernel): how far the core overlaps chains, see `latency_gflops`.
    pub fma_chain256_gflops: f64,
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
            mem_acc_scalar_unrolled_gflops: PLACEHOLDER_MEM_ACC_SCALAR_UNROLLED_GFLOPS,
            col_reload_gflops: PLACEHOLDER_COL_RELOAD_GFLOPS,
            col_unrolled_gflops: PLACEHOLDER_COL_UNROLLED_GFLOPS,
            store_gflops: PLACEHOLDER_STORE_GFLOPS,
            fma_chain_gflops: PLACEHOLDER_FMA_CHAIN_GFLOPS,
            fma_chain256_gflops: PLACEHOLDER_FMA_CHAIN256_GFLOPS,
            l1_bytes: 128 * 1024,
            l2_bytes: 16 * 1024 * 1024,
            line_bytes: 128,
            lanes: 4,
        }
    }

    /// The overlap window `W` of `latency_gflops` in FMAs: the independent FMAs the core keeps
    /// in flight beyond the dependent chain, `256 * (chain256 / chain - 1)`.
    fn window_fmas(&self) -> f64 {
        CHAIN_CALIBRATION_LENGTH * (self.fma_chain256_gflops / self.fma_chain_gflops - 1.0)
    }

    fn check(&self) -> Result<(), BenchError> {
        let speeds = [
            self.peak_gflops,
            self.mem_gbs,
            self.l2_gbs,
            self.mem_acc_gflops,
            self.mem_acc_scalar_gflops,
            self.mem_acc_scalar_unrolled_gflops,
            self.col_reload_gflops,
            self.col_unrolled_gflops,
            self.store_gflops,
            self.fma_chain_gflops,
            self.fma_chain256_gflops,
        ];
        let sizes = [self.l1_bytes, self.l2_bytes, self.line_bytes, self.lanes];
        // A chain of 256 independent-enough FMAs cannot be slower than one dependent chain.
        let window_ok = self.fma_chain256_gflops >= self.fma_chain_gflops;
        if speeds.iter().all(|s| s.is_finite() && *s > 0.0)
            && sizes.iter().all(|s| *s > 0)
            && window_ok
        {
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
    /// The roofline term that sets `time_s`: "compute" (FMA issue), "load" or "store" (vectorized
    /// nests: loads per FMA, stores per FMA), "ldst" (scalar nests: loads and stores in one
    /// pool), "reload" (scalar nests that reload a stored column, see `column_reload`),
    /// "colload" (scalar nests with an unrolled loop that loads a column, see
    /// `column_load_unrolled`), "latency" (dependent FMA chains), "memory" or "l2". A tie goes to
    /// the first of "compute"/"load"/"store"/"ldst"/"reload"/"colload"/"latency", then "memory".
    pub bound: &'static str,
    /// Memory ops per FMA op of the nest, see `mem_ops_per_fma`; breaks ties in `order`. The sum
    /// of `loads_per_fma` and `stores_per_fma`.
    pub mem_ops_per_fma: f64,
    /// Loads per FMA op of the nest, see `loads_stores_per_fma`.
    pub loads_per_fma: f64,
    /// Stores per FMA op of the nest, see `loads_stores_per_fma`.
    pub stores_per_fma: f64,
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

/// Per loop of `loops` (outermost first): whether clang fully unrolls it before the loop
/// vectorizer runs, although the schedule did not unroll it. LLVM unrolls a loop if its trip
/// count times its size (the instructions of the body, inner loops included) is at most
/// `CLANG_UNROLL_SIZE`; the loops are visited inside out, and an unrolled inner loop makes the
/// body of the next one `extent` times bigger. Sizes in the units of `INNERMOST_LOOP_SIZE` (9)
/// and `LOOP_LEVEL_SIZE` (5). Found in the IR (`-Rpass=loop-unroll`, Apple clang 21,
/// `BENCH_FLAGS`, N = 256, extents 2 to 128), not in the measurements. A loop with `r` rolled
/// loops inside it (the sizes are 9, 14, 19, 24, 29, 34 for r = 0 to 5) is unrolled if its
/// extent is at most:
///
/// | loop (rolled, not unrolled by the schedule)      | r   | unrolled if extent <= |
/// |--------------------------------------------------|-----|-----------------------|
/// | innermost, scalar or marked vectorized (*)       | 0   | 16                    |
/// | around the innermost loop (`ki` around `ji`)     | 1   | 10 (8)                |
/// | `ii` or `ko` around `ki, ji`                     | 2   | 7 (4)                 |
/// | around 3, 4, 5 rolled loops (`jo`, `io`, `ko`)   | 3-5 | 6, 5, 4 (4)           |
///
/// (*) Not a vectorized loop that is a power of two (all of them in the space): since v1.8 that is
/// `one_vector_iteration`, which clang does not unroll.
///
/// The extents in the space are 2 to 128 in powers of two, the value in brackets is what that
/// means for them. An unrolled inner loop makes the next one bigger: `ki` = 8 around a scalar
/// `ji` of 8 (72 + 5 = 77) is not unrolled. The inner loop may be marked vectorized: it is
/// still a loop for the unroller (the vectorizer runs later), so `ki` = 8 around a vectorized
/// `ji` of 64 becomes 8 copies of that loop; `ki` = 16 stays rolled.
/// A loop unrolled by the schedule is always unrolled and multiplies the body in the same way.
/// A loop for which `one_vector_iteration` holds is emitted as `vectorize_width(extent)
/// unroll(disable)` by `codegen-c`: one vector iteration, not unrolled, and IS vectorized.
fn fully_unrolled(loops: &[Loop]) -> Vec<bool> {
    let mut unrolled = vec![false; loops.len()];
    let mut inside = 0_usize; // size of what the loop below contributes to the body of `p`
    for (p, l) in loops.iter().enumerate().rev() {
        let size = if p + 1 == loops.len() {
            INNERMOST_LOOP_SIZE
        } else {
            LOOP_LEVEL_SIZE + inside
        };
        unrolled[p] = !one_vector_iteration(l)
            && (l.unroll || l.extent.saturating_mul(size) <= CLANG_UNROLL_SIZE);
        inside = if unrolled[p] {
            l.extent.saturating_mul(size)
        } else {
            size
        };
    }
    unrolled
}

/// Whether `codegen-c` emits `vectorize(enable) vectorize_width(extent) unroll(disable)` for
/// `l`, so that clang compiles it as ONE vector iteration (the vectorizer widens it to the
/// extent, `vectorize` stays on, no scalar unroll). `nest_c` does so for a vectorized loop
/// (1) that the schedule unrolls (a register tile; its extent is a power of two, see `apply`),
/// or (2) that is a power of two of at most `CLANG_EARLY_UNROLL_LIMIT` = `CLANG_FULL_UNROLL_LIMIT`
/// (#96, 284577f): clang would otherwise unroll it before the vectorizer, into scalar code (rule
/// K, v1.3 to v1.7, which the model keeps for the other short vectorized loops: an extent that is
/// not a power of two, such as the unsplit `j` of N = 12, gets the plain hint). The pragma of (2)
/// is the pragma of (1), so clang sees the same code and the model treats (2) as (1).
fn one_vector_iteration(l: &Loop) -> bool {
    l.vectorize && (l.unroll || (l.extent <= CLANG_FULL_UNROLL_LIMIT && l.extent.is_power_of_two()))
}

/// `nest` as clang compiles it: the loops of `fully_unrolled` become unrolled loops (a
/// vectorized one becomes scalar: vectorize off, unroll on); a loop of `one_vector_iteration`
/// stays vectorized and is marked unrolled, as a register tile's `ji` is. Idempotent.
fn as_compiled(nest: &LoopNest) -> LoopNest {
    let mut compiled = nest.clone();
    let unrolled = fully_unrolled(&nest.loops);
    for (l, &is_unrolled) in compiled.loops.iter_mut().zip(&unrolled) {
        if is_unrolled {
            l.vectorize = false;
            l.unroll = true;
        } else if one_vector_iteration(l) {
            l.unroll = true;
        }
    }
    compiled
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

/// Whether the innermost loop of `compiled` (a nest as `as_compiled` returns it) is unrolled and
/// every access walks it contiguously or not at all: the body of the loop around it is then
/// straight-line scalar code on adjacent elements, the shape of the `fma_mem_acc_scalar_unrolled`
/// calibration point (see the doc of `mem_ops_per_fma`). A strided access (a column walk: `ii`
/// innermost, or `B[k, j]` over `ki`) is not: unrolling does not speed it up.
fn straight_line_body(compiled: &LoopNest) -> bool {
    let Some(inner) = compiled.loops.last().filter(|l| l.unroll) else {
        return false;
    };
    distinct_accesses(compiled).iter().all(|access| {
        let coeffs: Vec<i64> = access
            .index
            .iter()
            .map(|index| coeff_of(index, &inner.name))
            .collect();
        let Some((last, others)) = coeffs.split_last() else {
            return true;
        };
        let unit_stride = *last == 1 && others.iter().all(|c| *c == 0);
        unit_stride || coeffs.iter().all(|c| *c == 0)
    })
}

/// Byte stride of `access` along the loop `name`: `sum over dims of coeff * product of the later
/// extents of its buffer * ELEMENT_BYTES`. 0 if the access does not move with the loop.
fn access_stride_bytes(access: &Access, buffers: &[Buffer], name: &str) -> Result<i64, BenchError> {
    let buffer = buffers
        .get(access.buffer)
        .filter(|b| b.shape.len() == access.index.len())
        .ok_or_else(|| model_err(format!("access to the unknown buffer {}", access.buffer)))?;
    let mut stride_elements = 0_i64;
    let mut later_elements = 1_i64;
    for (index, &extent) in access.index.iter().zip(&buffer.shape).rev() {
        let step_elements = coeff_of(index, name).saturating_mul(later_elements);
        stride_elements = stride_elements.saturating_add(step_elements);
        later_elements = later_elements.saturating_mul(i64::try_from(extent).unwrap_or(i64::MAX));
    }
    Ok(stride_elements.saturating_mul(ELEMENT_BYTES as i64))
}

/// Whether `compiled` (a nest as `as_compiled` returns it) is the loop nest of the
/// `fma_col_reload` calibration point: scalar code whose innermost loop is rolled, in which the
/// stored access (the update's output) walks a column and is reloaded at the next iteration of
/// the loop outside. All three must hold:
/// - the innermost loop is not unrolled (an unrolled one keeps the stored elements in registers
///   or is straight-line code; a vectorized one cannot walk a column, `vectorize` needs stride 1);
/// - the byte stride of the stored access along the innermost loop, see `access_stride_bytes`, is nonzero
///   and a multiple of `COL_RELOAD_STRIDE_BYTES`;
/// - the loop directly outside the innermost one has coefficient 0 in every index of the stored
///   access: the same elements are loaded again at its next iteration, right after the stores of
///   this one (with `j` or `ji` outside, `C[.., j]` moves on and nothing is reloaded).
///
/// Hardware reading (not proven): the load of `C` at the next iteration needs the store of the
/// previous one, while the other stores in flight go to addresses equal to it modulo 256 B.
fn column_reload(compiled: &LoopNest, buffers: &[Buffer]) -> Result<bool, BenchError> {
    let [.., parent, inner] = compiled.loops.as_slice() else {
        return Ok(false);
    };
    if inner.unroll {
        return Ok(false);
    }
    let target = &compiled.body.target;
    let stride_bytes = access_stride_bytes(target, buffers, &inner.name)?;
    let parent_moves_target = target
        .index
        .iter()
        .any(|index| coeff_of(index, &parent.name) != 0);
    Ok(stride_bytes != 0 && stride_bytes % COL_RELOAD_STRIDE_BYTES == 0 && !parent_moves_target)
}

/// Whether `compiled` (a nest as `as_compiled` returns it) is the loop nest of the
/// `fma_col_unrolled` calibration point: scalar code whose innermost loop is unrolled (by the
/// schedule or by clang) directly inside a rolled loop, and one load of the body walks a column
/// of the unrolled loop without being held in registers across the rolled one. All must hold:
/// - the innermost loop is unrolled and the loop directly outside it is not (`as_compiled`);
/// - the stored access moves along the innermost loop (nonzero byte stride): every unrolled copy
///   updates its own element, so the FMAs are independent accumulators and there is no reduction
///   chain through the unrolled loop (`ki` unrolled under `ji` adds into one `C` and is excluded);
/// - some distinct load of the stored value, other than the stored access itself, is not held in
///   registers across that rolled loop (`held_across` is false: it moves with the rolled loop's
///   variable, like `A[ii, k]` with `ki`, so every iteration of the rolled loop loads it again;
///   a load that does not move but does not fit `REG_BUDGET` registers is not held either);
/// - the byte stride of that load along the innermost loop, see `access_stride_bytes`, is nonzero
///   and a multiple of `COL_RELOAD_STRIDE_BYTES`: the unrolled copies load addresses equal modulo
///   256 B.
///
/// The shapes the conditions exclude have no calibration: a column load of the stored array
/// itself (`C` loaded and stored down a column, `ii` unrolled under `j` or `ji`) and a column
/// load that feeds a reduction (`B[k, j]` with `ki` unrolled under `ji`).
///
/// Hardware reading (not proven): loads at addresses equal modulo 256 B conflict in L1.
fn column_load_unrolled(
    compiled: &LoopNest,
    buffers: &[Buffer],
    lanes: usize,
) -> Result<bool, BenchError> {
    let [.., parent, inner] = compiled.loops.as_slice() else {
        return Ok(false);
    };
    if !inner.unroll || parent.unroll {
        return Ok(false);
    }
    let target = &compiled.body.target;
    if access_stride_bytes(target, buffers, &inner.name)? == 0 {
        return Ok(false);
    }
    let parent_pos = compiled.loops.len() - 2;
    let mut all_loads = Vec::new();
    loads(&compiled.body.value, &mut all_loads);
    for load in distinct(all_loads)
        .into_iter()
        .filter(|load| *load != target)
    {
        let stride_bytes = access_stride_bytes(load, buffers, &inner.name)?;
        if stride_bytes != 0
            && stride_bytes % COL_RELOAD_STRIDE_BYTES == 0
            && !held_across(load, &compiled.loops, lanes)?[parent_pos]
        {
            return Ok(true);
        }
    }
    Ok(false)
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

/// Per loop of `loops`: whether `access` is held in registers across it. A loop `p` is held
/// across if
/// - the access is invariant in `p`, and
/// - the elements it touches over the loops inside `p` fit `REG_BUDGET` registers, and
/// - every loop inside `p` that the access depends on is unrolled or vectorized (a register
///   cannot be indexed by a loop variable; tested for the `C` tile in
///   `crates/coffee-ten-codegen-c/tests/unroll.rs`, where clang keeps nothing in registers
///   without the unroll steps), and
/// - if `p` is unrolled, no loop inside it that the access depends on is vectorized and not
///   unrolled: every copy of `p` has its own copy of that loop, which loads and stores the
///   access again (clang does not merge them after the vectorizer), and nothing outside `p`
///   can hold the access either. `tests/full_unroll.rs`: `ki` = 8 around a vectorized `ji` has
///   8 vectorized loops and no vector phi, `ki` = 16 one loop and 16 phis.
fn held_across(access: &Access, loops: &[Loop], lanes: usize) -> Result<Vec<bool>, BenchError> {
    let mut held = vec![false; loops.len()];
    let mut rolled_dependence_inside = false;
    let mut vectorized_dependence_inside = false;
    for (p, l) in loops.iter().enumerate().rev() {
        let depends = access
            .index
            .iter()
            .any(|index| coeff_of(index, &l.name) != 0);
        if l.unroll && !l.vectorize && !depends && vectorized_dependence_inside {
            rolled_dependence_inside = true;
        }
        held[p] = !depends
            && !rolled_dependence_inside
            && registers_over(access, loops, p + 1, lanes)? <= REG_BUDGET;
        if depends && !l.unroll && !l.vectorize {
            rolled_dependence_inside = true;
        }
        if depends && l.vectorize && !l.unroll {
            vectorized_dependence_inside = true;
        }
    }
    Ok(held)
}

/// Memory ops `access` costs over the whole update nest, loads of it or stores of it (the
/// same count for either; `C[i, j]` has both). Counted as iterations of the loops that are not
/// held across, see `held_across`. Each loop that is not held across multiplies the count by
/// its iterations (the vectorized loop by `extent / lanes` rounded up, times `lanes` for a
/// strided access).
fn total_ops(access: &Access, loops: &[Loop], lanes: usize) -> Result<f64, BenchError> {
    let held = held_across(access, loops, lanes)?;
    let mut ops = 1.0_f64;
    for (l, _) in loops.iter().zip(&held).filter(|(_, held)| !**held) {
        ops *= if l.vectorize {
            l.extent.div_ceil(lanes) as f64 * access_ops(access, l, lanes).max(1) as f64
        } else {
            l.extent as f64
        };
    }
    Ok(ops)
}

/// `(loads, stores, FMA ops)` over the whole update `nest`. FMA ops: the product of the extents,
/// the vectorized loop as `extent / lanes` (rounded up) vector FMAs. Loads: `total_ops` of each
/// distinct Load of the stored value. Stores: `total_ops` of the store target (the same hoisting
/// rules for both, see `held_across`).
fn nest_ops(nest: &LoopNest, lanes: usize) -> Result<(f64, f64, f64), BenchError> {
    if lanes == 0 {
        return Err(model_err("lanes must be positive".to_string()));
    }
    if nest.loops.is_empty() {
        return Err(model_err("the nest has no loops".to_string()));
    }
    let mut all_loads = Vec::new();
    loads(&nest.body.value, &mut all_loads);
    let stores = total_ops(&nest.body.target, &nest.loops, lanes)?;
    let mut load_ops = 0.0;
    for load in distinct(all_loads) {
        load_ops += total_ops(load, &nest.loops, lanes)?;
    }
    let fma_ops = nest.loops.iter().fold(1.0, |acc, l| {
        acc * if l.vectorize {
            l.extent.div_ceil(lanes) as f64
        } else {
            l.extent as f64
        }
    });
    Ok((load_ops, stores, fma_ops))
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
/// Load/store ceilings, v1.4. Three calibration points, each for its own code shape, not a fit
/// of the model.
///
/// Vectorized nests have a load ceiling and a store ceiling. Hand microbenchmark, Apple M5,
/// L1-resident vector FMA kernels (GFLOP/s): 2 loads + 1 store per FMA (the shape of
/// `fma_mem_acc`) 45 to 47; 2 loads and no store about 49.5; 1 load and no store about 84; no
/// load and 1 store about 58 (the `fma_store` kernel of `bench/peak`, `--quick`; an earlier hand
/// kernel said 62 to 64, but clang hoisted 3 of its 4 FMAs, so its FLOPs were overcounted).
/// Loads and stores do NOT share one pool: removing the store of `fma_mem_acc` gains only about
/// 6%, so it is load-bound (and 1 load per FMA runs at about twice its rate), while stores have
/// a limit of their own. v1.3 put them in one pool
/// (`mem_acc * 3 / m`), which is wrong for a schedule that loads little and stores much or the
/// reverse (it said 117 to 128 GFLOP/s for a `C` row held across `ki`, measured 62 to 90).
/// - Load ceiling: `mem_acc_gflops * 2 / loads_per_fma`. The `fma_mem_acc` kernel does
///   `MEM_ACC_KERNEL_LOADS` = 2 loads per FMA at `mem_acc_gflops` = G and is load-bound
///   (evidence above), so a nest with `l` loads per FMA runs at `G * 2 / l`.
/// - Store ceiling: `store_gflops * 1 / stores_per_fma`; the calibration kernel stores once per
///   FMA (`STORE_KERNEL_STORES`). `store_gflops` is a PLACEHOLDER (58, the `--quick` value of the
///   `fma_store` kernel) until the peak line has `fma_store_gflops`.
///
/// Hand values with the placeholders (G = 60, store 58): `tiled+vectorize` (loads 1.09375,
/// stores 1 per FMA, N = 64) has a load ceiling 60 * 2 / 1.09375 = 109.7 and a store ceiling 58:
/// store-bound at 58, which UNDER-predicts the measured 60 to 64 by 3 to 10%. A `C` row held across `ki` (loads 1.09375, stores
/// 0.03125, see `a_c_row_held_across_ki_is_load_bound`) is load-bound at 109.7 (v1.3 said 160).
///
/// Scalar nests (last loop not vectorized) have, as in v1.3, one combined ceiling
/// `mem_acc_scalar_gflops * 3 / m` from the `fma_mem_acc_scalar` kernel (the same pattern in
/// scalar code, with the instruction shape of the naive inner loop), at S; at 2 FLOP per scalar
/// FMA it gives `S * 3 / m` GFLOP/s. That is ONE point with the naive body's shape and there is
/// no evidence yet about the scalar load and store ports, so they stay one pool. S is NOT G /
/// lanes: measured on P2, G = 46.94 and S = 8.563, not 11.7. ASSUMED: for a vectorized nest
/// nothing else (address arithmetic, the loop branch) limits; the scalar point already contains
/// that overhead of the naive loop, and every scalar nest pays that overhead EXCEPT the case
/// below.
///
/// Unrolled scalar nests use a second point, `mem_acc_scalar_unrolled_gflops` (U), as
/// `U * 3 / m`. The naive body (one FMA per iteration) is limited by its iterations, not by the
/// ports: the same access pattern as straight-line code runs 2.7 times faster. Hand
/// microbenchmark, Apple M5, `BENCH_FLAGS`, `c[j] = fma(a, b[j], c[j])` over 1024 floats, FMAs
/// per iteration of the rolled loop: 1: 8.4, 2: 14.4, 4: 20.6, 8: 20.7, 16: 22.9, 32: 22.9
/// GFLOP/s. The P5 peak run measured the 16-FMA kernel at 23.0 (`fma_mem_acc_scalar_unrolled`);
/// the PLACEHOLDER 23.2 is the hand value, used for peak lines without that key.
/// Which nests use U: the body of the innermost ROLLED loop is what clang emits, and in
/// `as_compiled` that is the product of the extents of the unrolled loops at the end of the
/// nest (1 if the innermost loop is rolled). Counting `fmadd` per iteration of the innermost loop
/// in the asm of seven schedules at N = 256: `ji` = 16 with `jo` rolled: 16; `ji` = 16, `ii` = 8
/// rolled: 16; `ji` = 8, `ii` = 16 rolled: 8; `ji` = 8 inside `ki` = 128: 8; `ii` = 8 and 16
/// innermost: 8 and 16; `ii` = 64 innermost: 1, as the model's `fully_unrolled` says in all
/// seven. So the test is "the innermost loop is unrolled" (the loops outside it only repeat the
/// body, and a rolled innermost loop has 1 FMA per iteration whatever is unrolled around it); in
/// the schedule space (extents 8 to 128) that body is 8 FMAs or more. Plus one condition: every
/// access is invariant in the innermost loop or has unit stride in it (`straight_line_body`). A
/// column walk (`ii` innermost: `C[ii, j]` and `A[ii, k]` have stride N; `ki` innermost: `B[ki,
/// j]`) does not speed up when unrolled: a hand column walk (stride 1 KiB) runs at 0.72 rolled
/// and 0.71 with 8 FMAs per iteration, and without the condition the 59 `ii` = 8 and 54 `ii` =
/// 16 records of run 3 move from a median pred / meas of 1.21 and 1.39 to 2.04 and 2.36. Those
/// nests keep S. ASSUMED: the point is U for every body of 8 or more FMAs (the 8-FMA body runs
/// at 20.7, 11% under U); bodies of 2 and 4 FMAs are not in the space (U would be 61% and 13%
/// too high for them); the rest of the body (loads of `A`, the stores of the row held in
/// registers) does not change the rate; nothing is known about unrolled bodies with a different
/// load/store mix.
///
/// Column reload, v1.6. Scalar nests for which `column_reload` holds (rolled innermost loop, the
/// stored access walks a column with a byte stride that is a multiple of 256 B, the loop outside
/// does not move it) also get the ceiling "reload" = `col_reload_gflops` (R), the
/// `fma_col_reload` kernel of `bench/peak`: N = 256, `io, j, k, ii`, `ii` = 64 rolled, 2.121
/// GFLOP/s in the P6 peak run against 8.49 for the plain scalar point. The stride sweep behind
/// the predicate is at `COL_RELOAD_STRIDE_BYTES`. R is NOT scaled by `MEM_ACC_KERNEL_MEM_OPS /
/// mem_ops` as the "ldst" point is: that scaling says how many memory ops of a pool one FMA
/// costs, while this ceiling is set by the `C` load that waits for the preceding store, which the
/// predicate gives every FMA of the nest exactly once, whatever else the body loads. The
/// calibration nest itself has m = 3 + 1 / 64 (the `B` load once per `ii` loop), so scaling by 3
/// / m would make the model miss its own calibration point by 0.5%. ASSUMED: the ceiling does not
/// depend on the extent of `ii` (measured only at 64 in the peak run; run 3 has 32 to 128) or on
/// what else is in the body; the predicate is not proven, only the hand experiments say it.
///
/// Unrolled column load, v1.7. Scalar nests for which `column_load_unrolled` holds (the innermost
/// loop unrolled directly inside a rolled one, a load that moves with the rolled loop walks a
/// column of the unrolled loop with a byte stride that is a multiple of 256 B; the stored access
/// is not that load and moves along the unrolled loop) also get the ceiling "colload" = `col_unrolled_gflops` (L), the `fma_col_unrolled` kernel of `bench/peak`:
/// N = 256, `io, j, ko, ki, ii`, `ii` = 16 unrolled by clang under `ki` = 64, `C` in registers,
/// 17 loads and 16 FMA per iteration of `ki`, no stores: 15.624 GFLOP/s in the P7 peak run
/// against 21.5 to 23.8 from the plain scalar point (8.38 * 3 / m, m about 1.06 to 1.18). The
/// stride sweep behind the predicate is in the README of `bench/peak` (the same 256 B granule as
/// `COL_RELOAD_STRIDE_BYTES`, which is reused). L is NOT scaled by `MEM_ACC_KERNEL_MEM_OPS /
/// mem_ops`, for the reason of the reload point: the ceiling is set by the loads of the column,
/// and the predicate gives every FMA of the body exactly one of them (the load depends on the
/// innermost loop, which is the last one, so it is loaded once per FMA), whatever else the body
/// loads. Scaling would also break the calibration: the kernel has m = 1.0703125 and 3 / m = 2.8
/// would put its own nest at 44 GFLOP/s, 2.8 times its measurement. ASSUMED: L holds for any
/// extent of the unrolled loop (measured at 16 and, by hand, 8: 15.8) and for any mix of other
/// loads in the body; at most one stride class (1 KiB) was measured with A, as the loaded
/// column. EXCLUDED, no calibration for them (they keep their v1.6 ceilings): `C` itself as the
/// column (`ii` unrolled under `j` or `ji`: `C[ii, j]` is loaded and stored down a column) and a
/// column load under a reduction (`ki` unrolled under `ji`: `B[k, j]` has stride 1 KiB along
/// `ki`, but all FMAs add into one `C`, a dependent chain); see the tests.
///
/// A loop marked vectorized with at most `CLANG_FULL_UNROLL_LIMIT` iterations that the schedule
/// does not unroll is, since v1.8, a vector loop if its extent is a power of two
/// (`one_vector_iteration`) and a scalar unrolled loop otherwise (rule K), see `as_compiled`.
/// Hand value, N = 12 (not a power of two): `vectorize(1, j)` is `unroll(1, j)`: `C` over `j` is
/// 12 scalar registers, held across `k`: per `i`, C 12 + 12, A 12, B 12 * 12: 180 / 144 FMAs =
/// 1.25. At N = 16 it is vectorized: C 4 + 4, A 16, B 16 * 4: 88 / 64 FMAs = 1.375.
pub fn mem_ops_per_fma(nest: &LoopNest, lanes: usize) -> Result<f64, BenchError> {
    let (loads, stores, fma_ops) = nest_ops(&as_compiled(nest), lanes)?;
    Ok((loads + stores) / fma_ops)
}

/// `(loads, stores)` per FMA op of the update `nest`, counted as in `mem_ops_per_fma` (which is
/// their sum). Stores: the store target; loads: every distinct load of the stored value, `C[i,
/// j]` included.
pub fn loads_stores_per_fma(nest: &LoopNest, lanes: usize) -> Result<(f64, f64), BenchError> {
    let (loads, stores, fma_ops) = nest_ops(&as_compiled(nest), lanes)?;
    Ok((loads / fma_ops, stores / fma_ops))
}

/// Latency ceiling in GFLOP/s of the update `nest` (as compiled, see `as_compiled`), `None` if
/// there is none.
///
/// Derivation. Let `p` be the innermost loop that is not unrolled, or that `C` does not depend
/// on (the FMAs of an unrolled loop `C` is invariant in are one more link of the chain; the
/// unrolled loops `C` depends on are independent chains, see `A` below). If the store target
/// `C` is held in registers across `p` (`held_across`: it does not depend on `p`, and stays in
/// registers), each FMA on an accumulator waits for the previous one: the FMAs form dependent
/// chains. `A` accumulator registers are in flight per iteration of `p` (`registers_over` the
/// loops inside `p`; a vector register counts once, a scalar once). The chain is as long as `p`
/// and the consecutive enclosing loops that `C` is also held across: `L` = the product of their
/// iterations. If `C` is not held across `p`, every FMA stores its result and the chain is
/// broken through memory; nothing is modelled then (`None`).
///
/// A core overlaps a window of `W` FMAs beyond the dependent one, so the FMA rate is
/// `chain_rate * (A + W / L)` with `chain_rate = fma_chain_gflops / 2` FMAs per ns: `A` chains
/// run side by side, and the window is shared by the `L` FMAs of a chain. In GFLOP/s that is
/// `fma_chain_gflops * (A + W / L)`, times `lanes` for vector FMAs.
///
/// Calibration, Apple M5, hand microbenchmark (BENCH flags), reductions `ko, i, j, ki` with the
/// naive loop body split into independent chains of length `L`: one dependent scalar FMA chain
/// 2.6 GFLOP/s (0.76 ns per FMA); L = 256: 3.49, 128: 4.30, 64: 6.36, 32: 7.85, 16: 6.76,
/// 8: 7.01 GFLOP/s. `W` = 256 * (3.49 / 2.6 - 1) = 87.6 FMAs is calibrated at L = 256 ONLY. The
/// law then gives 4.38 at L = 128 (measured 4.30) and 6.16 at L = 64 (6.36): checks, not fits.
/// For L <= 32 the measurement stays under the law (9.72 at L = 32): the scalar load/store
/// ceiling binds there (`fma_mem_acc_scalar`), which the model only partly reproduces, see
/// `a_split_reduction_follows_the_overlap_law_for_long_chains`.
///
/// ASSUMED: the window is measured in FMAs for the naive body shape (two loads, an FMA; other
/// bodies may have another window); the latency of a vector FMA equals the scalar one; `A`
/// chains do not compete for the window (the law was only measured at `A` = 1).
fn latency_gflops(
    nest: &LoopNest,
    machine: &Machine,
    is_vectorized: bool,
) -> Result<Option<f64>, BenchError> {
    let loops = &nest.loops;
    let target = &nest.body.target;
    let depends_on = |l: &Loop| {
        target
            .index
            .iter()
            .any(|index| coeff_of(index, &l.name) != 0)
    };
    let Some(p) = loops.iter().rposition(|l| !l.unroll || !depends_on(l)) else {
        return Ok(None);
    };
    let held = held_across(target, loops, machine.lanes)?;
    if !held[p] {
        return Ok(None);
    }
    let chain_length: f64 = loops[..=p]
        .iter()
        .zip(&held)
        .rev()
        .take_while(|(_, held)| **held)
        .map(|(l, _)| {
            if l.vectorize {
                l.extent.div_ceil(machine.lanes) as f64
            } else {
                l.extent as f64
            }
        })
        .product();
    let chains = registers_over(target, loops, p + 1, machine.lanes)? as f64;
    let lanes = if is_vectorized {
        machine.lanes as f64
    } else {
        1.0
    };
    Ok(Some(
        machine.fma_chain_gflops * (chains + machine.window_fmas() / chain_length) * lanes,
    ))
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
    let compiled = as_compiled(nest);
    let is_vectorized = compiled.loops.last().is_some_and(|l| l.vectorize);
    let is_unrolled = straight_line_body(&compiled);
    let lanes = machine.lanes as f64;
    let mem_ops = mem_ops_per_fma(nest, machine.lanes)?;
    let (loads, stores) = loads_stores_per_fma(nest, machine.lanes)?;
    // Vectorized: separate load and store ceilings; scalar: one combined ceiling (see the doc of
    // `mem_ops_per_fma`). A zero count of loads or stores gives an infinite ceiling.
    let (fma_gflops, ldst_ceilings) = if is_vectorized {
        let load_gflops = machine.mem_acc_gflops * MEM_ACC_KERNEL_LOADS / loads;
        let store_gflops = machine.store_gflops * STORE_KERNEL_STORES / stores;
        (
            machine.peak_gflops,
            vec![("load", load_gflops), ("store", store_gflops)],
        )
    } else {
        let point_gflops = if is_unrolled {
            machine.mem_acc_scalar_unrolled_gflops
        } else {
            machine.mem_acc_scalar_gflops
        };
        let ldst_gflops = point_gflops * MEM_ACC_KERNEL_MEM_OPS / mem_ops;
        let mut ceilings = vec![("ldst", ldst_gflops)];
        // Not scaled by `mem_ops`, see the doc of `mem_ops_per_fma`.
        if column_reload(&compiled, &kernel.buffers)? {
            ceilings.push(("reload", machine.col_reload_gflops));
        }
        if column_load_unrolled(&compiled, &kernel.buffers, machine.lanes)? {
            ceilings.push(("colload", machine.col_unrolled_gflops));
        }
        (machine.peak_gflops / lanes, ceilings)
    };
    // The lowest compute ceiling; a tie goes to the first of compute, load, store, ldst, reload,
    // colload, latency.
    let mut ceiling = ("compute", fma_gflops);
    for ldst in ldst_ceilings {
        if ldst.1 < ceiling.1 {
            ceiling = ldst;
        }
    }
    if let Some(latency) = latency_gflops(&as_compiled(nest), machine, is_vectorized)?
        && latency < ceiling.1
    {
        ceiling = ("latency", latency);
    }
    let (compute_bound, ceiling_gflops) = ceiling;
    let compute_s = flops / (ceiling_gflops * 1e9);
    let memory_s = mem_bytes as f64 / (machine.mem_gbs * 1e9);
    let l2_s = l2_bytes as f64 / (machine.l2_gbs * 1e9);
    let time_s = compute_s.max(memory_s).max(l2_s);
    let bound = if compute_s >= memory_s && compute_s >= l2_s {
        compute_bound
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
        mem_ops_per_fma: mem_ops,
        loads_per_fma: loads,
        stores_per_fma: stores,
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

/// Faster prediction first; equal times (many schedules sit at the FMA peak) by fewer memory
/// ops per FMA, then by schedule text.
fn order(a: &Ranked, b: &Ranked) -> Ordering {
    a.prediction
        .time_s
        .total_cmp(&b.prediction.time_s)
        .then_with(|| {
            a.prediction
                .mem_ops_per_fma
                .total_cmp(&b.prediction.mem_ops_per_fma)
        })
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
     coffee-ten-bench model compare --from <results.jsonl> [--peak <peak.jsonl>] [--group <name>]\n       \
       compare: --peak takes the machine from that file's peak line (for results files without \
     one); --group keeps only records whose \"group\" field equals <name> (e.g. random)";

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
/// `mem_acc_scalar` are measured. `fma_chain` and `fma_chain256` are read when the line has
/// them (`chain_measured` says which), else they stay the placeholders. Without `fma_mem_acc_gflops` in the line, `mem_acc_gflops`
/// stays the placeholder. Without `fma_mem_acc_scalar_gflops`, `mem_acc_scalar_gflops` is
/// `mem_acc_gflops / lanes` (the placeholder, or the measured vector point over lanes, as the
/// model did before the scalar point existed). `fma_mem_acc_scalar_unrolled_gflops`,
/// `fma_col_reload_gflops` and `fma_col_unrolled_gflops` are read when present
/// (`scalar_unrolled_measured`, `col_reload_measured` and `col_unrolled_measured` say so), else
/// they stay the placeholders.
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
    if let Some(gflops) = median_after(line, "fma_mem_acc_scalar_unrolled_gflops") {
        machine.mem_acc_scalar_unrolled_gflops = gflops;
    }
    if let Some(gflops) = median_after(line, "fma_col_reload_gflops") {
        machine.col_reload_gflops = gflops;
    }
    if let Some(gflops) = median_after(line, "fma_col_unrolled_gflops") {
        machine.col_unrolled_gflops = gflops;
    }
    if let Some(gflops) = median_after(line, "fma_store_gflops") {
        machine.store_gflops = gflops;
    }
    if let Some(gflops) = median_after(line, "fma_chain_gflops") {
        machine.fma_chain_gflops = gflops;
    }
    if let Some(gflops) = median_after(line, "fma_chain256_gflops") {
        machine.fma_chain256_gflops = gflops;
    }
    machine.check()?;
    Ok((machine, mem_acc.is_some(), mem_acc_scalar.is_some()))
}

/// Whether the peak line of `text` has `fma_mem_acc_scalar_unrolled_gflops`; without it
/// `mem_acc_scalar_unrolled_gflops` is `PLACEHOLDER_MEM_ACC_SCALAR_UNROLLED_GFLOPS`.
pub fn scalar_unrolled_measured(text: &str) -> bool {
    let line = text.lines().find(|line| line.contains("\"kind\":\"peak\""));
    line.is_some_and(|line| median_after(line, "fma_mem_acc_scalar_unrolled_gflops").is_some())
}

/// Whether the peak line of `text` has `fma_col_reload_gflops`; without it `col_reload_gflops` is
/// `PLACEHOLDER_COL_RELOAD_GFLOPS`.
pub fn col_reload_measured(text: &str) -> bool {
    let line = text.lines().find(|line| line.contains("\"kind\":\"peak\""));
    line.is_some_and(|line| median_after(line, "fma_col_reload_gflops").is_some())
}

/// Whether the peak line of `text` has `fma_col_unrolled_gflops`; without it
/// `col_unrolled_gflops` is `PLACEHOLDER_COL_UNROLLED_GFLOPS`.
pub fn col_unrolled_measured(text: &str) -> bool {
    let line = text.lines().find(|line| line.contains("\"kind\":\"peak\""));
    line.is_some_and(|line| median_after(line, "fma_col_unrolled_gflops").is_some())
}

/// Whether the peak line of `text` has `fma_store_gflops`; without it `store_gflops` is
/// `PLACEHOLDER_STORE_GFLOPS`.
pub fn store_measured(text: &str) -> bool {
    let line = text.lines().find(|line| line.contains("\"kind\":\"peak\""));
    line.is_some_and(|line| median_after(line, "fma_store_gflops").is_some())
}

/// Whether the peak line of `text` has `fma_chain_gflops` and `fma_chain256_gflops`.
pub fn chain_measured(text: &str) -> (bool, bool) {
    let line = text.lines().find(|line| line.contains("\"kind\":\"peak\""));
    let has = |key: &str| line.is_some_and(|line| median_after(line, key).is_some());
    (has("fma_chain_gflops"), has("fma_chain256_gflops"))
}

/// The header note on where the machine numbers of `--from path` came from.
fn from_note(
    path: &str,
    mem_acc_measured: bool,
    mem_acc_scalar_measured: bool,
    scalar_unrolled_measured: bool,
    col_measured: (bool, bool),
    store_measured: bool,
    chain_measured: (bool, bool),
) -> String {
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
    if !scalar_unrolled_measured {
        let _ = write!(
            note,
            "; mem_acc_scalar_unrolled {PLACEHOLDER_MEM_ACC_SCALAR_UNROLLED_GFLOPS} is a \
             PLACEHOLDER (no fma_mem_acc_scalar_unrolled_gflops in the peak line)"
        );
    }
    if !col_measured.0 {
        let _ = write!(
            note,
            "; col_reload {PLACEHOLDER_COL_RELOAD_GFLOPS} is a PLACEHOLDER \
             (no fma_col_reload_gflops in the peak line)"
        );
    }
    if !col_measured.1 {
        let _ = write!(
            note,
            "; col_unrolled {PLACEHOLDER_COL_UNROLLED_GFLOPS} is a PLACEHOLDER \
             (no fma_col_unrolled_gflops in the peak line)"
        );
    }
    if !store_measured {
        let _ = write!(
            note,
            "; store {PLACEHOLDER_STORE_GFLOPS} is a PLACEHOLDER \
             (no fma_store_gflops in the peak line)"
        );
    }
    for (measured, key) in [
        (chain_measured.0, "fma_chain"),
        (chain_measured.1, "fma_chain256"),
    ] {
        if !measured {
            let _ = write!(
                note,
                "; {key} is a PLACEHOLDER (no {key}_gflops in the peak line)"
            );
        }
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
                let text = read_results(&value)?;
                let (from, vector, scalar) = machine_from_results(&text)?;
                machine = from;
                source = from_note(
                    &value,
                    vector,
                    scalar,
                    scalar_unrolled_measured(&text),
                    (col_reload_measured(&text), col_unrolled_measured(&text)),
                    store_measured(&text),
                    chain_measured(&text),
                );
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
        "PREDICTION by the analytic model v1.8, not a measurement"
    );
    let _ = writeln!(
        text,
        "matmul {n}x{n}x{n}, update nest 1, {} candidates",
        ranked.len()
    );
    let _ = writeln!(
        text,
        "machine: peak {} GFLOP/s, mem {} GB/s, L2 {} GB/s, mem_acc {} GFLOP/s, \
         mem_acc_scalar {} GFLOP/s, mem_acc_scalar_unrolled {} GFLOP/s, col_reload {} GFLOP/s, col_unrolled {} GFLOP/s, store {} GFLOP/s, fma_chain {} GFLOP/s, fma_chain256 {} GFLOP/s (window {:.1} \
         FMAs) ({source}); L1 {} B, L2 {} B, line {} B, {} lanes (ASSUMED)",
        machine.peak_gflops,
        machine.mem_gbs,
        machine.l2_gbs,
        machine.mem_acc_gflops,
        machine.mem_acc_scalar_gflops,
        machine.mem_acc_scalar_unrolled_gflops,
        machine.col_reload_gflops,
        machine.col_unrolled_gflops,
        machine.store_gflops,
        machine.fma_chain_gflops,
        machine.fma_chain256_gflops,
        machine.window_fmas(),
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

/// The median with the two middle values averaged for an even count (unlike `summarize`,
/// which takes the lower one). `None` for an empty slice.
fn median(values: &[f64]) -> Option<f64> {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    match n {
        0 => None,
        _ if n % 2 == 1 => Some(sorted[n / 2]),
        _ => Some((sorted[n / 2 - 1] + sorted[n / 2]) / 2.0),
    }
}

/// The schedule groups of the `model compare` breakdown (BENCHMARKS.md, hypothesis A), in
/// print order.
const SCHEDULE_GROUPS: [&str; 6] = [
    "vectorized ji = 128",
    "vectorized ji = 32/64",
    "\"vectorized\" ji <= 16 (scalar code)",
    "scalar, ji innermost",
    "scalar, ii innermost",
    "scalar, k/ki innermost",
];

/// The `SCHEDULE_GROUPS` index of a matmul schedule: vectorized by the `ji` split factor,
/// scalar by the innermost loop of its `reorder` (the last name; `k` without a reorder).
fn schedule_group(schedule: &str) -> usize {
    if schedule.contains("vectorize(") {
        let ji = schedule
            .split("split(")
            .find(|s| s.contains(", ji,"))
            .and_then(|s| s.split([',', ')']).nth(4))
            .and_then(|f| f.trim().parse::<usize>().ok())
            .unwrap_or(0);
        return match ji {
            128.. => 0,
            32..=127 => 1,
            _ => 2,
        };
    }
    let innermost = schedule
        .split("reorder(")
        .nth(1)
        .and_then(|r| r.split(')').next())
        .and_then(|r| r.split(',').next_back())
        .map_or("k", str::trim);
    match innermost {
        "ji" | "j" => 3,
        "ii" | "i" => 4,
        _ => 5,
    }
}

/// The report of `coffee-ten-bench model compare`: the model against the `matmul` records of
/// the results file `text` (`source` is its name) that are not flagged `drift`; with `group`,
/// only the records whose `"group"` field equals it.
pub fn compare_report(source: &str, text: &str, group: Option<&str>) -> Result<String, BenchError> {
    let (machine, mem_acc_measured, mem_acc_scalar_measured) = machine_from_results(text)?;
    let (mut drifted, mut other_kernels, mut other_groups) = (0, 0, 0);
    let (mut measured, mut predicted, mut rows) = (Vec::new(), Vec::new(), String::new());
    let mut by_group: [(Vec<f64>, Vec<f64>); 6] = Default::default();
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
        if let Some(want) = group
            && string_after(line, "group").as_deref() != Some(want)
        {
            other_groups += 1;
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
        let (m, p) = &mut by_group[schedule_group(&schedule)];
        m.push(median);
        p.push(prediction.gflops);
    }
    let abs_log_ratios: Vec<f64> = predicted
        .iter()
        .zip(&measured)
        .map(|(p, m)| (p / m).ln().abs())
        .collect();
    let mut out = String::new();
    let _ = writeln!(
        out,
        "analytic model v1.8 (PREDICTION) against the measurements in {source}"
    );
    let _ = writeln!(
        out,
        "machine: peak {} GFLOP/s, mem {} GB/s, L2 {} GB/s, mem_acc {} GFLOP/s, \
         mem_acc_scalar {} GFLOP/s, mem_acc_scalar_unrolled {} GFLOP/s, col_reload {} GFLOP/s, col_unrolled {} GFLOP/s, store {} GFLOP/s, fma_chain {} GFLOP/s, fma_chain256 {} GFLOP/s (window {:.1} \
         FMAs) ({})",
        machine.peak_gflops,
        machine.mem_gbs,
        machine.l2_gbs,
        machine.mem_acc_gflops,
        machine.mem_acc_scalar_gflops,
        machine.mem_acc_scalar_unrolled_gflops,
        machine.col_reload_gflops,
        machine.col_unrolled_gflops,
        machine.store_gflops,
        machine.fma_chain_gflops,
        machine.fma_chain256_gflops,
        machine.window_fmas(),
        from_note(
            source,
            mem_acc_measured,
            mem_acc_scalar_measured,
            scalar_unrolled_measured(text),
            (col_reload_measured(text), col_unrolled_measured(text)),
            store_measured(text),
            chain_measured(text)
        )
    );
    let _ = writeln!(
        out,
        "skipped: {drifted} matmul records flagged drift; {other_kernels} records of other \
         kernels (relu-matmul-bias: its extra nests are not modelled yet)"
    );
    if let Some(want) = group {
        let _ = writeln!(
            out,
            "group filter: \"{want}\" only; skipped {other_groups} matmul records of other groups"
        );
    }
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
    let _ = writeln!(
        out,
        "by schedule group (error = median over the group's records of predicted / measured, \
         not the ratio of the two medians):"
    );
    let _ = writeln!(
        out,
        "{:<38} {:>5} {:>9} {:>9} {:>7}",
        "group", "n", "measured", "predicted", "error"
    );
    for (name, (m, p)) in SCHEDULE_GROUPS.iter().zip(&by_group) {
        let ratios: Vec<f64> = p.iter().zip(m).map(|(p, m)| p / m).collect();
        if let (Some(mm), Some(pm), Some(rm)) = (median(m), median(p), median(&ratios)) {
            let _ = writeln!(
                out,
                "{name:<38} {:>5} {mm:>9.1} {pm:>9.1} {rm:>7.3}",
                m.len()
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
            let rest: Vec<String> = args.by_ref().collect();
            let (mut from, mut peak, mut group) = (None, None, None);
            for pair in rest.chunks(2) {
                match pair {
                    [flag, value] if flag == "--from" => from = Some(value.clone()),
                    [flag, value] if flag == "--peak" => peak = Some(value.clone()),
                    [flag, value] if flag == "--group" => group = Some(value.clone()),
                    _ => {
                        return Err(BenchError::Usage(format!(
                            "model compare takes --from <results.jsonl> [--peak <peak.jsonl>] \
                             [--group <name>]\n{USAGE}"
                        )));
                    }
                }
            }
            let Some(from) = from else {
                return Err(BenchError::Usage(format!(
                    "model compare needs --from <results.jsonl>\n{USAGE}"
                )));
            };
            // --peak: the machine comes from that file's peak line instead of --from's.
            let mut text = String::new();
            if let Some(peak) = &peak {
                text.extend(
                    read_results(peak)?
                        .lines()
                        .filter(|l| l.contains("\"kind\":\"peak\""))
                        .map(|l| format!("{l}\n")),
                );
            }
            text.push_str(&read_results(&from)?);
            let source = match &peak {
                Some(peak) => format!("{from} (peak line from {peak})"),
                None => from,
            };
            return compare_report(&source, &text, group.as_deref());
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

    /// `machine` without the latency ceiling (`W` = 0 and a chain rate no kernel reaches), for
    /// tests of the other ceilings.
    fn without_latency(machine: Machine) -> Machine {
        Machine {
            fma_chain_gflops: 1e9,
            fma_chain256_gflops: 1e9,
            ..machine
        }
    }

    /// Fast caches, so that at N = 256 the traffic does not bind.
    fn fast_caches() -> Machine {
        Machine {
            mem_gbs: 1e6,
            l2_gbs: 1e6,
            ..Machine::assumed_m5()
        }
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
    fn the_ranking_is_deterministic_and_ties_go_by_mem_ops_then_schedule_text() {
        let machine = Machine::assumed_m5();
        let first = rank_matmul(&machine, 64).expect("ranks");
        assert_eq!(first, rank_matmul(&machine, 64).expect("ranks"));
        let (mut ties, mut text_ties) = (0, 0);
        for pair in first.windows(2) {
            let (a, b) = (&pair[0], &pair[1]);
            assert!(a.prediction.time_s <= b.prediction.time_s);
            if a.prediction.time_s == b.prediction.time_s {
                assert!(a.prediction.mem_ops_per_fma <= b.prediction.mem_ops_per_fma);
                if a.prediction.mem_ops_per_fma == b.prediction.mem_ops_per_fma {
                    assert!(a.schedule < b.schedule, "{} / {}", a.schedule, b.schedule);
                    text_ties += 1;
                }
                ties += 1;
            }
        }
        assert!(
            ties > 0 && text_ties > 0,
            "{ties} ties, {text_ties} by text"
        );
    }

    /// Three register tiles of N = 64 sit at the FMA peak with the same time; m is 0.53125
    /// (4x16), 0.65625 (8x8) and 1.03125 (2x8), see `register_tiles_pay_a_k_th_of_the_tile`.
    /// By text alone the 2x8 tile would come first (`... 2)` < `... 4)` < `... 8)`).
    #[test]
    fn equal_times_at_the_cap_rank_by_fewer_memory_ops_per_fma() {
        // The 2x8 tile has 4 accumulators and is latency-bound, see `latency_gflops`.
        let machine = without_latency(fast_caches());
        let tile = |ri: usize, rj: usize| {
            let text = format!(
                "split(1, i, io, ii, {ri}); split(1, j, jo, ji, {rj}); \
                 reorder(1, io, jo, k, ii, ji); vectorize(1, ji); unroll(1, ii); unroll(1, ji)"
            );
            let prediction = predict_schedule(&machine, 64, &text).expect("predicts");
            Ranked {
                schedule: text,
                prediction,
            }
        };
        let (t2x8, t4x16, t8x8) = (tile(2, 8), tile(4, 16), tile(8, 8));
        let times = [t2x8.prediction.time_s, t4x16.prediction.time_s];
        assert_eq!(times[0], times[1]);
        assert_eq!(t8x8.prediction.time_s, times[0]);
        let ms = [
            t4x16.prediction.mem_ops_per_fma,
            t8x8.prediction.mem_ops_per_fma,
        ];
        assert_eq!(ms, [0.53125, 0.65625]);
        assert_eq!(t2x8.prediction.mem_ops_per_fma, 1.03125);
        let mut ranked = vec![t2x8.clone(), t8x8.clone(), t4x16.clone()];
        ranked.sort_by(order);
        assert_eq!(ranked, [t4x16, t8x8, t2x8]);
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
        // Equal time: fewer memory ops per FMA first, even for the later text.
        let leaner = Ranked {
            prediction: Prediction {
                mem_ops_per_fma: p.mem_ops_per_fma / 2.0,
                ..p.clone()
            },
            ..ranked("z")
        };
        assert_eq!(order(&leaner, &ranked("a")), Ordering::Less);
        assert_eq!(order(&ranked("a"), &leaner), Ordering::Greater);
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

    /// The same tile without the unroll steps, and with an `ii` that clang does not unroll
    /// either: `ii` is a loop that C depends on, so a register cannot hold C across k (clang keeps
    /// nothing, `tests/unroll.rs`).
    /// - 16x32 (`ji` = 8 vectors), N = 64: C 8 + 8 per k and ii; A 1 per k and ii (held across
    ///   ji); B 8 per k (held across ii). Per k (16 * 8 = 128 FMAs): 16 * 17 + 8 = 280: 2.1875. IR:
    ///   the `ii` loop has 8 vector loads of C, 1 of A, 8 stores; the 8 loads of B are before it.
    ///
    /// The 4x16 tile is not here since v1.8: its `ji` of 16 is one vector iteration and the tile
    /// holds C, see `the_c6_schedules_with_a_short_ji_are_predicted_as_vector_code`. (v1.7 had it
    /// scalar unrolled: `ii` of 4 around 16 scalar copies is 4 * (5 + 144) > 150 and stays a
    /// loop: C 4 * (16 + 16) per k, A 4, B 16 (held across ii): 148 / 64 FMAs = 2.3125.)
    #[test]
    fn a_tile_without_unroll_does_not_hold_c() {
        let rolled = |ri: usize, rj: usize| {
            format!(
                "split(1, i, io, ii, {ri}); split(1, j, jo, ji, {rj}); \
                 reorder(1, io, jo, k, ii, ji); vectorize(1, ji)"
            )
        };
        assert_eq!(mem_ops(64, &rolled(16, 32)), 2.1875);
    }

    /// The 4x32 tile without unroll steps: clang unrolls `ii` = 4 itself (4 * (5 + 9) = 56), and
    /// the 4 copies of the vectorized `ji` loop each load their own row of B (IR: 32 vector loads
    /// of B and no phi for B in the `k` loop; with 16x32 the 8 loads are before the `ii` loop).
    /// B is not held across `ii`. C over `ii, ji` is 32 > 24 registers (clang keeps all 32 in
    /// phis, the model stops at `REG_BUDGET`): C 4 * (8 + 8) per k, A 4, B 4 * 8: 100 / 32 FMAs =
    /// 3.125 (2.375 when B was held).
    #[test]
    fn a_short_ii_is_unrolled_and_reloads_b_per_copy() {
        let tile = "split(1, i, io, ii, 4); split(1, j, jo, ji, 32); \
                    reorder(1, io, jo, k, ii, ji); vectorize(1, ji)";
        assert_eq!(mem_ops(64, tile), 3.125);
    }

    /// A loop marked `vectorize` with at most `CLANG_FULL_UNROLL_LIMIT` = 16 iterations that is
    /// NOT a power of two is scalar and unrolled (rule K: clang unrolls it before the
    /// vectorizer, `tests/full_unroll.rs`); a power of two is one vector iteration since v1.8
    /// (`one_vector_iteration`, `codegen-c` emits the pragma of a register tile for it).
    /// - N = 12 (the unsplit `j` of the naive `i, k, j`, the only way to a non-power of two in
    ///   the space): C over `j` is 12 scalar registers, held across `k`; per `i`: C 12 + 12,
    ///   A[i, k] 12, B 12 * 12 = 180 over 144 scalar FMAs = 1.25, the same as `unroll(1, j)`.
    ///   Predicted at the scalar rates: the FMA ceiling 100 / 4 = 25 (the scalar ld/st ceiling
    ///   is 23.2 * 3 / 1.25 = 55.68).
    /// - N = 16: C over `j` is 4 vector registers: per `i`, loads C 4 + A 16 + B 16 * 4 = 84,
    ///   stores 4, over 64 vector FMAs: 1.375 (1.3125 loads), the same as with `unroll(1, j)`
    ///   added. Predicted: the load ceiling 60 * 2 / 1.3125 = 91.43; the latency ceiling (4
    ///   chains of 16, 4 * 2.6 * (4 + 87.6 / 16) = 98.54) and the peak 100 are above it.
    /// - N = 17 (5 vectors, not a power of two, over the limit) is vectorized as before: per
    ///   `i`, loads C 5 + A 17 + B 17 * 5 = 107 over 85 vector FMAs = 1.2588, stores 5 / 85; the
    ///   load ceiling is 60 * 2 / (107 / 85) = 95.33, below the peak 100 (the store ceiling 58 /
    ///   (5 / 85) is far above).
    #[test]
    fn a_short_vectorized_loop_is_scalar_unless_it_is_a_power_of_two() {
        assert_eq!(mem_ops(12, "vectorize(1, j)"), 1.25);
        assert_eq!(mem_ops(12, "unroll(1, j)"), 1.25);
        assert_eq!(mem_ops(16, "vectorize(1, j)"), 1.375);
        assert_eq!(mem_ops(16, "vectorize(1, j); unroll(1, j)"), 1.375);
        let machine = Machine {
            mem_gbs: 1e6,
            l2_gbs: 1e6,
            ..Machine::assumed_m5()
        };
        let k = predict(&machine, &scheduled(12, "vectorize(1, j)")).expect("predicts");
        assert!((k.gflops - 25.0).abs() < 1e-9, "{}", k.gflops);
        assert_eq!(k.bound, "compute");
        let short = predict(&machine, &scheduled(16, "vectorize(1, j)")).expect("predicts");
        assert!(
            (short.gflops - 60.0 * 2.0 / 1.3125).abs() < 1e-9,
            "{}",
            short.gflops
        );
        assert_eq!(short.bound, "load");
        let tile = predict(&machine, &scheduled(16, "vectorize(1, j); unroll(1, j)"));
        assert_eq!(short, tile.expect("predicts"));
        let long = predict(&machine, &scheduled(17, "vectorize(1, j)")).expect("predicts");
        assert!(
            (long.gflops - 60.0 * 2.0 * 85.0 / 107.0).abs() < 1e-9,
            "{}",
            long.gflops
        );
        assert_eq!(long.bound, "load");
    }

    /// The register tile `ji` of 8 is vectorized AND unrolled: `codegen-c` emits
    /// `vectorize_width(8) unroll(disable)` and clang vectorizes it, so the 16-iteration rule
    /// does not apply: peak 100, not 100 / 4.
    #[test]
    fn a_vectorized_and_unrolled_short_loop_stays_vectorized() {
        // The 2x8 tile has 4 accumulators and is latency-bound, see `latency_gflops`.
        let machine = without_latency(fast_caches());
        let tile = "split(1, i, io, ii, 2); split(1, j, jo, ji, 8); \
                    reorder(1, io, jo, k, ii, ji); vectorize(1, ji); unroll(1, ii); unroll(1, ji)";
        let p = predict(&machine, &scheduled(64, tile)).expect("predicts");
        assert_eq!((p.gflops, p.bound), (100.0, "compute"));
    }

    /// C6: three schedules with `ji` <= 16 at N = 256 on `Machine::assumed_m5`, which clang
    /// compiles to vector code since #96 (`one_vector_iteration`; before v1.8 the first two were
    /// predicted as scalar unrolled code and the third did not hold C). Clang facts (Apple
    /// clang 21, `BENCH_FLAGS`, by hand): 4, 2 and 16 `fmla.4s`, no `fmadd`, 1, 1 and 4 vector
    /// phis of C for the three. The last one is the 4x16 tile without unroll steps: its
    /// prediction is that of the same tile with them, whose pragma is identical.
    /// - `i, jo, k, ji` with `ji` = 16: per `(i, jo)` C 4 vectors loaded and stored once (held
    ///   across `k`), A 256, B 4 * 256 loads over 4 * 256 vector FMAs: m = (4 + 256 + 1024 + 4) /
    ///   1024 = 1.2578125; loads 1.25390625 per FMA, load ceiling 60 * 2 / 1.25390625 = 95.70.
    ///   The latency ceiling binds: 4 chains of 256, 4 * 2.6 * (4 + W / 256) = 4 * (3 * 2.6 +
    ///   3.49) = 45.16.
    /// - the same with `ji` = 8: C 2 vectors, B 2 * 256: m = (2 + 256 + 512 + 2) / 512 =
    ///   1.5078125; load ceiling 79.79; latency, 2 chains: 4 * (2.6 + 3.49) = 24.36 binds.
    /// - 4x16, `io, jo, k, ii, ji` (`ii` = 4 is unrolled by clang, 4 * (5 + 9) = 56): C 16
    ///   vectors held across `k`, A 4 * 256, B 4 * 256 vectors:
    ///   m = (16 + 1024 + 1024 + 16) / 4096 = 0.5078125; the load ceiling (238.1) and the
    ///   latency ceiling (16 chains, 4 * (15 * 2.6 + 3.49) = 169.96) are above the peak 100.
    #[test]
    fn the_c6_schedules_with_a_short_ji_are_predicted_as_vector_code() {
        let machine = Machine::assumed_m5();
        // (schedule, loops clang unrolls, mem ops per FMA, GFLOP/s, bound)
        let rows: [(&str, &[&str], f64, f64, &str); 3] = [
            (
                "split(1, j, jo, ji, 16); reorder(1, i, jo, k, ji); vectorize(1, ji)",
                &[],
                1.2578125,
                4.0 * (3.0 * 2.6 + 3.49),
                "latency",
            ),
            (
                "split(1, j, jo, ji, 8); reorder(1, i, jo, k, ji); vectorize(1, ji)",
                &[],
                1.5078125,
                4.0 * (2.6 + 3.49),
                "latency",
            ),
            (
                "split(1, i, io, ii, 4); split(1, j, jo, ji, 16); reorder(1, io, jo, k, ii, ji); \
                 vectorize(1, ji)",
                &["ii"],
                0.5078125,
                100.0,
                "compute",
            ),
        ];
        for (schedule, unrolled, m, gflops, bound) in rows {
            assert_eq!(clang_unrolled(schedule), unrolled, "{schedule}");
            assert_eq!(mem_ops(256, schedule), m, "{schedule}");
            let p = predict_schedule(&machine, 256, schedule).expect("predicts");
            assert!(
                (p.gflops / gflops - 1.0).abs() < 1e-9,
                "{schedule}: {}",
                p.gflops
            );
            assert_eq!(p.bound, bound, "{schedule}");
        }
        let unrolled = "split(1, i, io, ii, 4); split(1, j, jo, ji, 16); \
                        reorder(1, io, jo, k, ii, ji); vectorize(1, ji); unroll(1, ii); \
                        unroll(1, ji)";
        assert_eq!(
            predict_schedule(&machine, 256, rows[2].0),
            predict_schedule(&machine, 256, unrolled)
        );
    }

    /// The loops of `text` (N = 256) that `as_compiled` unrolls into scalar code and the schedule
    /// did not (not the vectorized loops of `one_vector_iteration`, which stay vector code).
    fn clang_unrolled(text: &str) -> Vec<String> {
        clang_unrolled_at(256, text)
    }

    /// `clang_unrolled` at size `n`.
    fn clang_unrolled_at(n: usize, text: &str) -> Vec<String> {
        let kernel = scheduled(n, text);
        let nest = update_nest(&kernel);
        let compiled = as_compiled(nest);
        let pairs = nest.loops.iter().zip(&compiled.loops);
        pairs
            .filter(|(before, after)| after.unroll && !after.vectorize && !before.unroll)
            .map(|(before, _)| before.name.clone())
            .collect()
    }

    /// `fully_unrolled` against the loops clang unrolled before the vectorizer in the IR (Apple
    /// clang 21, `BENCH_FLAGS`, `-Rpass=loop-unroll`, the remark of the loop before the first
    /// "vectorized loop" remark), N = 256. Each row has the sizes of the doc of `fully_unrolled`:
    /// innermost 9, each loop level adds 5, an unrolled loop counts `extent` times.
    #[test]
    fn the_loops_clang_unrolls_are_unrolled_in_the_model() {
        let text = |ii: usize, ji: usize, ki: usize, order: &str, vectorize: bool| {
            format!(
                "split(1, i, io, ii, {ii}); split(1, j, jo, ji, {ji}); split(1, k, ko, ki, {ki}); \
                 reorder(1, {order}){}",
                if vectorize { "; vectorize(1, ji)" } else { "" }
            )
        };
        let around = "io, jo, ko, ii, ki, ji";
        let inner_ji = "io, jo, ko, ii, ji, ki";
        // Innermost loop: 16 iterations yes (16 * 9), 32 no. A loop marked vectorized with 16 is
        // not: `codegen-c` makes it one vector iteration (v1.8), `unroll(disable)`.
        assert_eq!(
            clang_unrolled("split(1, k, ko, ki, 16); reorder(1, ko, i, j, ki)"),
            ["ki"]
        );
        assert!(clang_unrolled("split(1, k, ko, ki, 32); reorder(1, ko, i, j, ki)").is_empty());
        assert!(
            clang_unrolled("split(1, j, jo, ji, 16); reorder(1, i, jo, k, ji); vectorize(1, ji)")
                .is_empty()
        );
        assert_eq!(
            clang_unrolled("split(1, j, jo, ji, 16); reorder(1, i, jo, k, ji)"),
            ["ji"]
        );
        // A loop around a vectorized innermost loop of 64: 8 yes (8 * 14), 16 no (16 * 14). Then
        // `jo` of 4 around four rolled loops (4 * 29) is unrolled in the second case.
        assert_eq!(clang_unrolled(&text(16, 64, 8, around, true)), ["ki"]);
        assert_eq!(clang_unrolled(&text(16, 64, 16, around, true)), ["jo"]);
        // The same around a vectorized `ji` of 16, which is as rolled as one of 64 for the
        // unroller (`ii` = 16 has 8 unrolled `ki` inside it: 16 * (5 + 112) > 150): 8 copies of
        // the vectorized loop in the IR (clang remarks, Apple clang 21).
        assert_eq!(clang_unrolled(&text(16, 16, 8, around, true)), ["ki"]);
        // Around a scalar innermost loop: `ki` of 8 around `ji` of 64 yes, `ji` of 8 around `ki`
        // of 32 yes; `ki` of 8 around an unrolled `ji` of 8 (8 * (5 + 72)) no.
        assert_eq!(
            clang_unrolled(&text(8, 64, 8, "io, jo, ko, ii, ki, ji", false)),
            ["ki"]
        );
        assert_eq!(clang_unrolled(&text(8, 8, 32, inner_ji, false)), ["ji"]);
        assert_eq!(clang_unrolled(&text(8, 8, 8, inner_ji, false)), ["ki"]);
        // `ii` of 8 around `ji`, with `ki` of 128 outside it: `ii` yes, `ki` no, `ko` of 2 no
        // (2 * 122). With `ii` of 16: `ii` no, `ko` of 2 yes (2 * 24).
        assert_eq!(
            clang_unrolled(&text(8, 64, 128, "io, jo, ko, ki, ii, ji", true)),
            ["ii"]
        );
        assert_eq!(
            clang_unrolled(&text(16, 64, 128, "io, jo, ko, ki, ii, ji", true)),
            ["ko"]
        );
    }

    /// `ki` around the vectorized `ji` of 64 (a row of C is 16 vectors), N = 128, `io, jo, ko, ii,
    /// ki, ji`. With `ki` = 8 clang unrolls `ki` and `ji` becomes 8 loops (`tests/full_unroll.rs`:
    /// 8 vectorized loops, no phi): C is loaded and stored 16 + 16 per `ki`, A 1, B 16 (not held
    /// across `ii`, 8 * 16 > 24 registers): 49 / 16 FMAs = 3.0625. It was 1.3125, C held across
    /// `ki`: (32 + 8 * 17) / 128. With `ki` = 16 and 32 the loop stays rolled and holds C: per
    /// `ii` C 32, A and B 17 per `ki`: (32 + 16 * 17) / 256 = 1.1875, (32 + 32 * 17) / 512 =
    /// 1.125.
    #[test]
    fn a_ki_of_8_around_a_vectorized_ji_reloads_c_per_copy() {
        let ki_schedule = |ki: usize| {
            format!(
                "split(1, i, io, ii, 16); split(1, j, jo, ji, 64); split(1, k, ko, ki, {ki}); \
                 reorder(1, io, jo, ko, ii, ki, ji); vectorize(1, ji)"
            )
        };
        assert_eq!(mem_ops(128, &ki_schedule(8)), 3.0625);
        assert_eq!(mem_ops(128, &ki_schedule(16)), 1.1875);
        assert_eq!(mem_ops(128, &ki_schedule(32)), 1.125);
    }

    /// `tiled+vectorize` (32 x 64, N = 64): B over ji is 16 registers, held across ii: 16 loads
    /// per k. C over `ii, ji` is 32 * 16 = 512 registers, not held: 16 loads and 16 stores per
    /// k and ii. A[i,k] held across ji: 1 per k and ii. Per k and ii (16 FMAs):
    /// (16 + 16 + 1 + 16 / 32) / 16 = 33.5 / 16 = 2.09375.
    #[test]
    fn tiled_vectorized_holds_b_across_ii() {
        assert_eq!(mem_ops(64, &format!("{TILED}; vectorize(1, ji)")), 2.09375);
    }

    /// An unrolled loop that is not part of a tile. The sizes are chosen so that clang does not
    /// unroll anything else (at N = 4 it unrolls all three loops of the naive nest, IR: 48 loads
    /// and no loop; `fully_unrolled` does not know that for such small trip counts).
    /// - N = 32, `i, k, j` with `k` unrolled: `j` (32 * 9 > 150) is the plain loop inside it, so
    ///   nothing is held that was not before: same as without unroll, 3 + 1 / 32 = 3.03125. IR: each
    ///   of the 32 copies of the `j` loop loads A before it, B and C inside.
    /// - N = 16, `i, j, k` with `k` unrolled: A[i,k] (16 registers) is held across j: 16 * 16
    ///   loads; B[k,j] 16^3; C[i,j] held across k: 2 * 16^2: 4864 / 4096 = 1.1875. IR: the 16 loads
    ///   of A are before the `j` loop, the loop has 16 B loads and 1 C load.
    /// - N = 4, all three loops unrolled: A, B and C are 16 elements each and loaded once
    ///   (stored once): 64 / 64 = 1.
    #[test]
    fn unrolled_loops_outside_a_tile_hold_what_fits() {
        assert_eq!(mem_ops(32, "unroll(1, k)"), 3.03125);
        assert_eq!(mem_ops(32, ""), 3.03125);
        assert_eq!(mem_ops(16, "reorder(1, i, j, k); unroll(1, k)"), 1.1875);
        assert_eq!(mem_ops(4, "unroll(1, i); unroll(1, k); unroll(1, j)"), 1.0);
    }

    /// mem_acc = 60, store = 58 and mem_acc_scalar = 60 / 4 = 15 (placeholders). Scalar at N = 64
    /// has m = 3.015625, one combined ceiling 15 * 3 / m = 14.922. Vectorized at N = 256 has
    /// loads 2.015625 (C, B, and A once per `j` loop) and 1 store per FMA: load ceiling 60 * 2 /
    /// 2.015625 = 59.53, store ceiling 58 / 1 = 58: store-bound. With a fast store (1e9) the load
    /// ceiling shows. Both are below the FMA ceilings 100 and 100 / 4 = 25. A register tile (4x16,
    /// N = 64, loads 0.515625, stores 1/64) has the ceilings 232.7 and 3712, so the FMA peak binds.
    #[test]
    fn the_ldst_ceilings_follow_the_calibration_points() {
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
        assert!((vector.gflops - 58.0).abs() < 1e-9, "{}", vector.gflops);
        assert_eq!((scalar.bound, vector.bound), ("ldst", "store"));
        let fast_store = Machine {
            store_gflops: 1e9,
            ..machine.clone()
        };
        let vector = predict(&fast_store, &scheduled(256, "vectorize(1, j)")).expect("predicts");
        assert!(
            (vector.gflops - 60.0 * 2.0 / 2.015625).abs() < 1e-9,
            "{}",
            vector.gflops
        );
        assert_eq!(vector.bound, "load");
        let tile = "split(1, i, io, ii, 4); split(1, j, jo, ji, 16); \
                    reorder(1, io, jo, k, ii, ji); vectorize(1, ji); unroll(1, ii); unroll(1, ji)";
        let tiled = predict(&machine, &scheduled(64, tile)).expect("predicts");
        assert_eq!((tiled.gflops, tiled.bound), (100.0, "compute"));
    }

    /// The measured P2 points: mem_acc 46.9387, mem_acc_scalar 8.563 (not 46.9387 / 4 = 11.73).
    /// The naive `i, k, j` at N = 64 is scalar with m = 3.015625, so its ceiling is
    /// 8.563 * 3 / 3.015625 = 8.5186..., below the scalar FMA ceiling 100 / 4. The vectorized
    /// nest (N = 256, loads 2.015625) uses mem_acc only: the load ceiling 46.9387 * 2 / 2.015625,
    /// below the store ceiling 58, whatever the scalar point.
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
        let expected = 46.9387 * 2.0 / 2.015625;
        assert!((vector.gflops - expected).abs() < 1e-9, "{}", vector.gflops);
        assert_eq!(vector.bound, "load");
        let other = Machine {
            mem_acc_scalar_gflops: 1.0,
            ..machine
        };
        let again = predict(&other, &scheduled(256, "vectorize(1, j)")).expect("predicts");
        assert_eq!(again.gflops, vector.gflops);
    }

    /// Scalar nests with an unrolled innermost loop use `mem_acc_scalar_unrolled` (23.2 here),
    /// those with a rolled one `mem_acc_scalar` (8.563). FMA peak 1000 (scalar 250), fast caches
    /// and no latency ceiling, so only the ld/st ceiling `S * 3 / m` binds.
    /// - `unroll(1, j)` at N = 16 (j unrolled by the schedule) has m = 1.1875: 23.2 * 3 /
    ///   1.1875 = 58.6105. `vectorize(1, j)` at N = 12 (j unrolled by clang, scalar: not a
    ///   power of two, see `a_short_vectorized_loop_is_scalar_unless_it_is_a_power_of_two`)
    ///   has m = 1.25: 23.2 * 3 / 1.25 = 55.68. (Until v1.7 `vectorize(1, j)` at N = 16 was
    ///   here, with the 58.6105 of `unroll(1, j)`; it is a vector loop now.)
    /// - `reorder(1, j, k, i)` at N = 16 has `i` unrolled by clang, `reorder(1, i, j, k);
    ///   unroll(1, k)` has `k` unrolled, both strided (m = 1.1875): 8.563 * 3 / 1.1875 = 21.633,
    ///   the rolled point.
    /// - the naive nest at N = 64 has a rolled `j` (64 * 9 > 150) and m = 3.015625:
    ///   8.563 * 3 / 3.015625 = 8.5186. So does `unroll(1, k)` at N = 64 (`k` is outside `j`): `j`
    ///   stays the innermost loop, rolled, and the copies of `k` do not hold the 64-float row of
    ///   `C` in registers (64 > 24).
    #[test]
    fn an_unrolled_scalar_nest_uses_the_unrolled_calibration_point() {
        let machine = without_latency(Machine {
            peak_gflops: 1000.0,
            mem_acc_scalar_gflops: 8.563,
            mem_acc_scalar_unrolled_gflops: 23.2,
            mem_gbs: 1e6,
            l2_gbs: 1e6,
            ..Machine::assumed_m5()
        });
        let gflops = |n: usize, text: &str| {
            let p = predict(&machine, &scheduled(n, text)).expect("predicts");
            assert_eq!(p.bound, "ldst", "{text}");
            p.gflops
        };
        let unrolled: f64 = 23.2 * 3.0 / 1.1875;
        assert!((unrolled - 58.6105).abs() < 1e-4, "{unrolled}");
        assert!((gflops(16, "unroll(1, j)") - unrolled).abs() < 1e-9);
        let clang_unrolled: f64 = 23.2 * 3.0 / 1.25;
        assert!((clang_unrolled - 55.68).abs() < 1e-9, "{clang_unrolled}");
        assert!((gflops(12, "vectorize(1, j)") - clang_unrolled).abs() < 1e-9);
        let rolled: f64 = 8.563 * 3.0 / 3.015625;
        assert!((rolled - 8.5186).abs() < 1e-4, "{rolled}");
        for text in ["", "unroll(1, k)"] {
            assert_eq!(mem_ops(64, text), 3.015625, "{text}");
            assert!((gflops(64, text) - rolled).abs() < 1e-9, "{text}");
        }
        // `i` or `k` innermost at N = 16 is unrolled too (m = 1.1875, `i` by the symmetry of `i`
        // and `j`), but it walks columns (stride 16), which unrolling does not speed up: the
        // rolled point.
        for text in ["reorder(1, j, k, i)", "reorder(1, i, j, k); unroll(1, k)"] {
            assert_eq!(mem_ops(16, text), 1.1875, "{text}");
            assert!(
                (gflops(16, text) - 8.563 * 3.0 / 1.1875).abs() < 1e-9,
                "{text}"
            );
        }
    }

    /// Machine for the column-reload tests: fast caches and the reload point 2.5 (not the
    /// placeholder 2.1, so that the number is seen to come from the machine). The scalar ld/st
    /// ceiling 15 * 3 / m is about 15, above it.
    fn reload_machine() -> Machine {
        Machine {
            col_reload_gflops: 2.5,
            ..fast_caches()
        }
    }

    fn reload_bound(n: usize, text: &str) -> (&'static str, f64) {
        let p = predict(&reload_machine(), &scheduled(n, text)).expect("predicts");
        (p.bound, p.gflops)
    }

    /// The scalar matmul nest of `fma_col_reload` (`io, j, k, ii`, `ii` = 64 rolled and innermost,
    /// N = 256, C's row stride 1 KiB = 4 * 256 B, `k` or `ki` directly outside `ii` with
    /// coefficient 0 in `C[io*64+ii, j]`) is capped at `col_reload_gflops`, taken as is. Both
    /// nests have m = 3.015625 (`C` 2, `A` 1, `B[k, j]` once per `ii` loop: 1 / 64); the point is
    /// not scaled by 3 / m: the predicate guarantees one `C` load and one `C` store per FMA, which
    /// is what the ceiling counts, and the calibration kernel has the same m (scaling would give
    /// 2.5 * 3 / 3.015625 = 2.487 for its own nest). The ld/st ceiling 15 * 3 / 3.015625 = 14.92
    /// is higher.
    #[test]
    fn a_scalar_column_reload_nest_is_capped_at_the_reload_point() {
        for text in [
            "split(1, i, io, ii, 64); reorder(1, io, j, k, ii)",
            "split(1, i, io, ii, 64); split(1, k, ko, ki, 64); reorder(1, io, j, ko, ki, ii)",
        ] {
            assert_eq!(mem_ops(256, text), 3.015625, "{text}");
            assert_eq!(reload_bound(256, text), ("reload", 2.5), "{text}");
        }
    }

    /// The rule needs `C` reloaded at the next iteration of the loop outside `ii`: with `j` or
    /// `ji` directly outside, `C[.., j]` moves with it and the ld/st ceiling (14.9 here) binds.
    /// With `ii` fully unrolled (8 or 16 under `ki`, `C` stays in registers), a vectorized `ji`
    /// (stride 4 B; `vectorize` of a column walk is illegal), or the stored access not moving
    /// along the innermost loop (`ki` innermost: stride 0 although the parent `ko` has
    /// coefficient 0 too), there is no reload ceiling either.
    #[test]
    fn the_reload_ceiling_needs_a_rolled_scalar_ii_and_a_parent_that_does_not_move_c() {
        for text in [
            "split(1, i, io, ii, 64); reorder(1, io, k, j, ii)",
            "split(1, i, io, ii, 64); split(1, j, jo, ji, 8); reorder(1, io, jo, k, ji, ii)",
            "split(1, i, io, ii, 8); split(1, k, ko, ki, 64); reorder(1, io, j, ko, ki, ii)",
            "split(1, i, io, ii, 16); split(1, k, ko, ki, 64); reorder(1, io, j, ko, ki, ii)",
            "split(1, j, jo, ji, 64); reorder(1, i, jo, k, ji); vectorize(1, ji)",
            "split(1, k, ko, ki, 64); reorder(1, i, j, ko, ki)",
        ] {
            let (bound, gflops) = reload_bound(256, text);
            assert_ne!(bound, "reload", "{text}");
            assert!(gflops > 2.5, "{text}: {gflops}");
        }
        let unrolled =
            "split(1, i, io, ii, 8); split(1, k, ko, ki, 64); reorder(1, io, j, ko, ki, ii)";
        assert_eq!(clang_unrolled(unrolled), ["ii"]);
    }

    /// C's row stride is `4 * n` bytes. 256 B (n = 64) and its multiples (n = 128: 512 B; 256:
    /// 1 KiB) fire; 384 B (n = 96, 128 mod 256) and 400 B (n = 100) do not. `ii` is rolled in all
    /// rows (20 to 32 iterations are above the unroll limit 16).
    #[test]
    fn the_reload_stride_is_a_multiple_of_256_bytes() {
        let nest = |ii: usize| format!("split(1, i, io, ii, {ii}); reorder(1, io, j, k, ii)");
        for (n, ii, fires) in [
            (64, 32, true),
            (128, 32, true),
            (256, 32, true),
            (96, 24, false),
            (100, 20, false),
        ] {
            let (bound, gflops) = reload_bound(n, &nest(ii));
            assert_eq!(bound == "reload", fires, "n = {n}: {bound} {gflops}");
            assert_eq!(gflops == 2.5, fires, "n = {n}: {bound} {gflops}");
        }
    }

    /// Machine for the unrolled-column-load tests: fast caches and the colload point 9.5 (not the
    /// placeholder 15.7). The FMA ceiling is 100 / 4 = 25 and the scalar ld/st ceiling about 42,
    /// both above it.
    fn colload_machine() -> Machine {
        Machine {
            col_unrolled_gflops: 9.5,
            ..fast_caches()
        }
    }

    fn colload_bound(n: usize, text: &str) -> (&'static str, f64) {
        let p = predict(&colload_machine(), &scheduled(n, text)).expect("predicts");
        (p.bound, p.gflops)
    }

    /// The scalar matmul nest of `fma_col_unrolled` (`io, j, ko, ki, ii`, `ii` = 16 unrolled by
    /// clang under a rolled `ki`, N = 256: `A[io*16+ii, k]` moves with `ki` and walks a column,
    /// 1 KiB = 4 * 256 B apart) is capped at `col_unrolled_gflops`, taken as is, and so is the same
    /// nest with `ii` = 8 and with `k` itself under `ii`. `C` is held across `ki` and `B[k, j]` is
    /// invariant in `ii`. The ceiling is not scaled by 3 / m: the calibration kernel has 17 loads
    /// per 16 FMA (m = 1.0703125 with the 16 + 16 `C` accesses once per `k` loop); scaling by 3 / m
    /// would put its own nest at 9.5 * 3 / 1.0703125 = 26.6.
    #[test]
    fn a_scalar_unrolled_column_load_nest_is_capped_at_the_colload_point() {
        let tile = |ii: usize| {
            format!(
                "split(1, i, io, ii, {ii}); split(1, k, ko, ki, 64); reorder(1, io, j, ko, ki, ii)"
            )
        };
        assert_eq!(mem_ops(256, &tile(16)), 1.0703125);
        assert_eq!(mem_ops(256, &tile(8)), 1.1328125);
        for text in [
            tile(16),
            tile(8),
            "split(1, i, io, ii, 16); reorder(1, io, j, k, ii)".to_string(),
        ] {
            assert_eq!(clang_unrolled(&text), ["ii"], "{text}");
            assert_eq!(colload_bound(256, &text), ("colload", 9.5), "{text}");
        }
    }

    /// No colload ceiling when
    /// - `ii` = 32 is rolled (the reload rule applies instead: bound "reload", 2.1 here),
    /// - the nest is vectorized,
    /// - the unrolled loop walks unit stride (`ji` = 16 under `ki`: `A[i, k]` has stride 0 along
    ///   it, `B` and `C` 4 B), or there is no unrolled loop (the naive nest).
    #[test]
    fn the_colload_ceiling_needs_an_unrolled_innermost_loop_under_a_rolled_one() {
        let rolled =
            "split(1, i, io, ii, 32); split(1, k, ko, ki, 64); reorder(1, io, j, ko, ki, ii)";
        assert_eq!(clang_unrolled(rolled), ["ko"]);
        assert_eq!(colload_bound(256, rolled), ("reload", 2.1));
        let unit =
            "split(1, j, jo, ji, 16); split(1, k, ko, ki, 64); reorder(1, i, jo, ko, ki, ji)";
        assert_eq!(clang_unrolled(unit), ["ji"]);
        for text in [
            "split(1, j, jo, ji, 64); reorder(1, i, jo, k, ji); vectorize(1, ji)",
            unit,
            "",
        ] {
            let (bound, gflops) = colload_bound(256, text);
            assert_ne!(bound, "colload", "{text}");
            assert!(gflops > 9.5, "{text}: {gflops}");
        }
    }

    /// Two shapes without a calibration do not get the ceiling, although a load moves with the
    /// rolled loop and walks a column of the unrolled one:
    /// - `ii` = 16 under `j` (`io, k, j, ii`): `A[io*16+ii, k]` is held across `j` (16 <= 24
    ///   registers) and `B[k, j]` does not move along `ii`; the only column load is `C[io*16+ii,
    ///   j]`, the stored access itself (condition: the load is not the stored access);
    /// - `ki` = 16 unrolled under `j` or `ji` (`ko, i, j, ki`): `B[k, j]` has stride 1 KiB along
    ///   `ki` and moves with `j`, but the stored `C[i, j]` does not move along `ki` (all FMAs add
    ///   into one `C`, a dependent chain; condition: the stored access moves along the unrolled
    ///   loop). The nest keeps its v1.6 bound: "latency" (see
    ///   `a_split_reduction_follows_the_overlap_law_for_long_chains`).
    #[test]
    fn the_colload_ceiling_skips_c_as_the_column_and_a_reduction_over_the_unrolled_loop() {
        for (text, unrolled) in [
            ("split(1, i, io, ii, 16); reorder(1, io, k, j, ii)", "ii"),
            ("split(1, k, ko, ki, 16); reorder(1, ko, i, j, ki)", "ki"),
            (
                "split(1, j, jo, ji, 8); split(1, k, ko, ki, 8); reorder(1, ko, i, jo, ji, ki)",
                "ki",
            ),
        ] {
            assert!(
                clang_unrolled(text).contains(&unrolled.to_string()),
                "{text}"
            );
            let (bound, gflops) = colload_bound(256, text);
            assert_ne!(bound, "colload", "{text}");
            assert!(gflops > 9.5, "{text}: {gflops}");
        }
        let (bound, _) = colload_bound(256, "split(1, k, ko, ki, 16); reorder(1, ko, i, j, ki)");
        assert_eq!(bound, "latency");
    }

    /// The stride of `A` along `ii` is `4 * n` bytes. 256 B (n = 64) and its multiples (n = 128:
    /// 512 B; 256: 1 KiB) fire; 384 B (n = 96, 128 mod 256) and 400 B (n = 100) do not. `ii` is
    /// unrolled and `k` rolled directly outside it in all rows.
    #[test]
    fn the_colload_stride_is_a_multiple_of_256_bytes() {
        let nest = |ii: usize| format!("split(1, i, io, ii, {ii}); reorder(1, io, j, k, ii)");
        for (n, ii, fires) in [
            (64, 16, true),
            (128, 16, true),
            (256, 16, true),
            (96, 16, false),
            (100, 10, false),
        ] {
            let text = nest(ii);
            assert_eq!(clang_unrolled_at(n, &text), ["ii"], "n = {n}");
            let (bound, gflops) = colload_bound(n, &text);
            assert_eq!(bound == "colload", fires, "n = {n}: {bound} {gflops}");
            assert_eq!(gflops == 9.5, fires, "n = {n}: {bound} {gflops}");
        }
    }

    /// `fma_col_unrolled_gflops` of the peak line replaces the placeholder 15.7 (the note stops
    /// calling it one and the header shows it); without it the placeholder stays and the note says
    /// so. The key is not confused with `fma_col_reload_gflops`.
    #[test]
    fn from_reads_col_unrolled_when_the_peak_line_has_it() {
        let line = r#"{"kind":"peak","fma_gflops":{"median":100.5,"n":30},"triad_gbs":{"median":90.25,"n":30},"triad_l2_gbs":{"median":110.0,"n":30},"fma_col_reload_gflops":{"median":2.5,"n":30},"fma_col_unrolled_gflops":{"median":12.5,"n":30}}"#;
        let missing = line.replace("fma_col_unrolled_gflops", "x");
        let (with, _, _) = machine_from_results(line).expect("parses");
        let (without, _, _) = machine_from_results(&missing).expect("parses");
        assert_eq!(PLACEHOLDER_COL_UNROLLED_GFLOPS, 15.7);
        assert_eq!(
            (with.col_unrolled_gflops, without.col_unrolled_gflops),
            (12.5, 15.7)
        );
        assert_eq!(with.col_reload_gflops, 2.5);
        assert!(col_unrolled_measured(line) && !col_unrolled_measured(&missing));
        let note = from_note("f", true, true, true, (true, true), true, (true, true));
        assert!(!note.contains("PLACEHOLDER"), "{note}");
        let note = from_note("f", true, true, true, (true, false), true, (true, true));
        assert!(
            note.contains("col_unrolled 15.7 is a PLACEHOLDER"),
            "{note}"
        );
        let header = report(8, &with, &note, &[]);
        assert!(header.contains("col_unrolled 12.5 GFLOP/s"), "{header}");
        assert!(header.contains("analytic model v1.8"), "{header}");
        let bad = Machine {
            col_unrolled_gflops: 0.0,
            ..with
        };
        assert!(predict(&bad, &naive(8)).is_err());
    }

    /// `fma_mem_acc_scalar_unrolled_gflops` of the peak line replaces the placeholder 23.2 (and
    /// the note stops calling it one); without it the placeholder stays, the note and the
    /// header say so, and the prediction of an unrolled scalar nest follows the point in use.
    #[test]
    fn mem_acc_scalar_unrolled_is_read_from_the_peak_line_or_stays_the_placeholder() {
        let measured = r#"{"kind":"peak","fma_gflops":{"median":1000.5,"n":30},"triad_gbs":{"median":90.25,"n":30},"fma_mem_acc_scalar_unrolled_gflops":{"median":24.0,"n":30},"triad_l2_gbs":{"median":110.0,"n":30}}"#;
        let missing = measured.replace("fma_mem_acc_scalar_unrolled_gflops", "x");
        let (with, _, _) = machine_from_results(measured).expect("parses");
        let (without, _, _) = machine_from_results(&missing).expect("parses");
        assert_eq!(PLACEHOLDER_MEM_ACC_SCALAR_UNROLLED_GFLOPS, 23.2);
        assert_eq!(
            (
                with.mem_acc_scalar_unrolled_gflops,
                without.mem_acc_scalar_unrolled_gflops
            ),
            (24.0, 23.2)
        );
        assert!(scalar_unrolled_measured(measured) && !scalar_unrolled_measured(&missing));
        // The scalar point is not confused with the unrolled one: its key is a prefix.
        assert_eq!(with.mem_acc_scalar_gflops, with.mem_acc_gflops / 4.0);
        let note = from_note("f", true, true, true, (true, true), true, (true, true));
        assert!(!note.contains("PLACEHOLDER"), "{note}");
        let note = from_note("f", true, true, false, (true, true), true, (true, true));
        assert!(
            note.contains("mem_acc_scalar_unrolled 23.2 is a PLACEHOLDER"),
            "{note}"
        );
        let header = report(8, &with, &note, &[]);
        assert!(
            header.contains("mem_acc_scalar_unrolled 24 GFLOP/s"),
            "{header}"
        );
        // 1.25 memory ops per FMA at N = 12 (a `j` that clang unrolls, scalar, since 12 is not a
        // power of two), see `a_short_vectorized_loop_is_scalar_unless_it_is_a_power_of_two`.
        let gflops = |machine: Machine| {
            let machine = without_latency(Machine {
                mem_gbs: 1e6,
                l2_gbs: 1e6,
                ..machine
            });
            predict(&machine, &scheduled(12, "vectorize(1, j)"))
                .expect("predicts")
                .gflops
        };
        assert!((gflops(with) - 24.0 * 3.0 / 1.25).abs() < 1e-9);
        assert!((gflops(without) - 23.2 * 3.0 / 1.25).abs() < 1e-9);
    }

    /// Only the load and store ceilings can bind: FMA peak 1000, fast caches, no latency ceiling.
    fn ports_only(mem_acc_gflops: f64, store_gflops: f64) -> Machine {
        Machine {
            peak_gflops: 1000.0,
            mem_acc_gflops,
            store_gflops,
            ..without_latency(fast_caches())
        }
    }

    const C_ROW_ACROSS_KI: &str = "split(1, i, io, ii, 16); split(1, j, jo, ji, 64); \
         split(1, k, ko, ki, 32); reorder(1, io, jo, ko, ii, ki, ji); vectorize(1, ji)";

    fn loads_stores(n: usize, text: &str) -> (f64, f64) {
        let kernel = scheduled(n, text);
        loads_stores_per_fma(update_nest(&kernel), Machine::assumed_m5().lanes).expect("counts")
    }

    /// Loads and stores per vector FMA, hand counts (the ones of the `mem_ops` tests, split):
    /// - naive `i, k, j`, N = 256: C, B per FMA and A once per `j` loop (64 vectors): 2 + 1/64
    ///   loads; the C store: 1.
    /// - `tiled+vectorize`, N = 64: per `k` and `ii` (16 FMAs) C 16, A 1 and B 16 / 32 loads (B
    ///   is held across `ii`): 17.5 / 16; C 16 stores: 1.
    /// - `C_ROW_ACROSS_KI`, N = 128: per `ii` (16 * 32 = 512 FMAs) C 16 loads and 16 stores
    ///   (held across `ki`), A 32, B 32 * 16: loads (16 + 32 + 512) / 512, stores 16 / 512.
    #[test]
    fn loads_and_stores_are_counted_apart_and_add_up_to_the_memory_ops() {
        assert_eq!(loads_stores(256, "vectorize(1, j)"), (2.015625, 1.0));
        assert_eq!(
            loads_stores(64, &format!("{TILED}; vectorize(1, ji)")),
            (1.09375, 1.0)
        );
        assert_eq!(loads_stores(128, C_ROW_ACROSS_KI), (1.09375, 0.03125));
        for (n, text) in [
            (256, "vectorize(1, j)"),
            (64, C_ROW_ACROSS_KI),
            (128, C_ROW_ACROSS_KI),
            (64, ""),
        ] {
            let (loads, stores) = loads_stores(n, text);
            assert_eq!(loads + stores, mem_ops(n, text), "{text}");
        }
        let p = predict_schedule(&Machine::assumed_m5(), 128, C_ROW_ACROSS_KI).expect("predicts");
        assert_eq!(
            (p.loads_per_fma, p.stores_per_fma, p.mem_ops_per_fma),
            (1.09375, 0.03125, 1.125)
        );
    }

    /// A `C` row held across `ki` (`C_ROW_ACROSS_KI`, N = 128: loads 1.09375, stores 0.03125 per
    /// FMA) is nearly all loads. With mem_acc 46 and store 58: load ceiling 46 * 2 / 1.09375 =
    /// 84.11, store ceiling 58 / 0.03125 = 1856. One pool (v1.3) said 46 * 3 / 1.125 = 122.7, a
    /// factor 1.46 more: stores do not take load slots.
    #[test]
    fn a_c_row_held_across_ki_is_load_bound() {
        let p = predict_schedule(&ports_only(46.0, 58.0), 128, C_ROW_ACROSS_KI).expect("predicts");
        assert_eq!(p.bound, "load");
        assert!((p.gflops - 92.0 / 1.09375).abs() < 1e-9, "{}", p.gflops);
        let pooled = 46.0 * 3.0 / 1.125;
        assert!(p.gflops * 1.4 < pooled, "{} vs {pooled}", p.gflops);
    }

    /// `tiled+vectorize` at N = 64 (loads 1.09375, stores 1 per FMA) with mem_acc 60: load
    /// ceiling 60 * 2 / 1.09375 = 109.71; with store 58 the store ceiling 58 / 1 binds. A store
    /// of 200 takes it out and the load ceiling shows.
    #[test]
    fn tiled_vectorized_is_store_bound_until_the_store_is_fast() {
        let text = format!("{TILED}; vectorize(1, ji)");
        let slow = predict_schedule(&ports_only(60.0, 58.0), 64, &text).expect("predicts");
        assert_eq!(slow.bound, "store");
        assert!((slow.gflops - 58.0).abs() < 1e-9, "{}", slow.gflops);
        let fast = predict_schedule(&ports_only(60.0, 200.0), 64, &text).expect("predicts");
        assert_eq!(fast.bound, "load");
        assert!(
            (fast.gflops - 120.0 / 1.09375).abs() < 1e-9,
            "{}",
            fast.gflops
        );
    }

    /// `fma_store_gflops` of the peak line replaces the placeholder 58 (and the note stops
    /// calling it one); without it the placeholder stays and the note and the header say so.
    #[test]
    fn store_gflops_is_read_from_the_peak_line_or_stays_the_placeholder() {
        let measured = r#"{"kind":"peak","fma_gflops":{"median":100.5,"n":30},"triad_gbs":{"median":90.25,"n":30},"fma_store_gflops":{"median":55.5,"n":30},"triad_l2_gbs":{"median":110.0,"n":30}}"#;
        let missing = measured.replace("fma_store_gflops", "x");
        let (with, _, _) = machine_from_results(measured).expect("parses");
        let (without, _, _) = machine_from_results(&missing).expect("parses");
        assert_eq!(PLACEHOLDER_STORE_GFLOPS, 58.0);
        assert_eq!((with.store_gflops, without.store_gflops), (55.5, 58.0));
        assert!(store_measured(measured) && !store_measured(&missing));
        let note = from_note(
            "f",
            true,
            true,
            true,
            (true, true),
            store_measured(measured),
            (true, true),
        );
        assert!(!note.contains("PLACEHOLDER"), "{note}");
        let note = from_note(
            "f",
            true,
            true,
            true,
            (true, true),
            store_measured(&missing),
            (true, true),
        );
        assert!(note.contains("store 58 is a PLACEHOLDER"), "{note}");
        assert!(report(8, &with, &note, &[]).contains("store 55.5 GFLOP/s"));
        // The prediction follows it: tiled+vectorize is store-bound.
        let text = format!("{TILED}; vectorize(1, ji)");
        let gflops = |machine: Machine| {
            let machine = Machine {
                mem_gbs: 1e6,
                l2_gbs: 1e6,
                ..machine
            };
            predict_schedule(&machine, 64, &text)
                .expect("predicts")
                .gflops
        };
        assert!((gflops(with) - 55.5).abs() < 1e-9);
        assert!((gflops(without) - 58.0).abs() < 1e-9);
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

    /// The latency ceiling of `text` on `machine`, N = 256, as `predict` computes it.
    fn latency(machine: &Machine, text: &str) -> Option<f64> {
        let kernel = scheduled(256, text);
        let nest = as_compiled(update_nest(&kernel));
        let vectorized = nest.loops.last().is_some_and(|l| l.vectorize);
        latency_gflops(&nest, machine, vectorized).expect("analyses")
    }

    /// Placeholders: chain 2.6, chain256 3.49 GFLOP/s, so W = 256 * (3.49 / 2.6 - 1) = 87.6308
    /// FMAs and the rate is 2.6 * (A + W / L). Scalar nests at N = 256, caches fast.
    /// - naive `i, j, k`: C is held across `k`, A = 1, L = 256: 2.6 * (1 + W / 256) = 3.49, the
    ///   calibration point itself. The ld/st ceiling (15 * 3 / (2 + 2 / 256) = 22.4) and the
    ///   FMA ceiling (25) are far above: "latency".
    /// - `ko, i, j, ki` with `ki` = 64: C depends on `j`, so the chain stops at `ki`: L = 64:
    ///   2.6 * (1 + W / 64) = 6.16 (measured 6.36, a check, not a fit); with 128: 4.375 (4.30).
    /// - `i, j, ko, ki`: C does not depend on `ko` either and is held across it: L = 64 * 4 = 256.
    /// - `ki` = 32: 9.72 (measured 7.85). The ld/st ceiling with the measured P2 point
    ///   8.563 * 3 / (2 + 2 / 32) = 12.46 is above it, so latency still binds. At `ki` = 16 clang
    ///   unrolls `ki` (innermost, 16 * 9 <= 150), A[i, k] is held across `j` (IR: its 16 loads are
    ///   before the `j` loop) and the `j` body has B 16 loads, C 1 load and 1 store: m = 1.1289,
    ///   the ld/st ceiling is 8.563 * 3 / m = 22.8, so the law's 16.8 binds (it was 12.09 with
    ///   `ki` as a rolled loop, m = 2.125). The microbenchmark has the ld/st ceiling bind from
    ///   L = 32 (7.85, 6.76): the model's ld/st ceiling is too high for it.
    #[test]
    fn a_split_reduction_follows_the_overlap_law_for_long_chains() {
        let machine = fast_caches();
        let w = 256.0 * (3.49 / 2.6 - 1.0);
        assert!((machine.window_fmas() - w).abs() < 1e-12);
        let near = |got: Option<f64>, want: f64| {
            let got = got.expect("has a latency ceiling");
            assert!((got - want).abs() < 1e-9, "{got} vs {want}");
        };
        near(
            latency(&machine, "reorder(1, i, j, k)"),
            2.6 * (1.0 + w / 256.0),
        );
        near(latency(&machine, "reorder(1, i, j, k)"), 3.49);
        let split =
            |ki: usize, order: &str| format!("split(1, k, ko, ki, {ki}); reorder(1, {order})");
        near(
            latency(&machine, &split(64, "ko, i, j, ki")),
            2.6 * (1.0 + w / 64.0),
        );
        near(
            latency(&machine, &split(128, "ko, i, j, ki")),
            2.6 * (1.0 + w / 128.0),
        );
        near(latency(&machine, &split(64, "i, j, ko, ki")), 3.49);
        near(
            latency(&machine, &split(32, "ko, i, j, ki")),
            2.6 * (1.0 + w / 32.0),
        );
        let bound = |machine: &Machine, text: &str| {
            let p = predict_schedule(machine, 256, text).expect("predicts");
            (p.bound, p.gflops)
        };
        let (b64, g64) = bound(&machine, &split(64, "ko, i, j, ki"));
        assert_eq!(b64, "latency");
        assert!((g64 - 6.16).abs() < 1e-2, "{g64}");
        let (naive_bound, naive_gflops) = bound(&machine, "reorder(1, i, j, k)");
        assert_eq!(naive_bound, "latency");
        assert!((naive_gflops - 3.49).abs() < 1e-9, "{naive_gflops}");
        let p2 = Machine {
            mem_acc_gflops: 46.9387,
            mem_acc_scalar_gflops: 8.563,
            ..machine
        };
        let (b32, g32) = bound(&p2, &split(32, "ko, i, j, ki"));
        assert_eq!(b32, "latency");
        assert!((g32 - 9.72).abs() < 1e-2, "{g32}");
        let (b16, g16) = bound(&p2, &split(16, "ko, i, j, ki"));
        assert_eq!(b16, "latency");
        assert!((g16 - 2.6 * (1.0 + w / 16.0)).abs() < 1e-9, "{g16}");
    }

    /// The register tile 4x16 at N = 256: C over `ii, ji` is 4 * 4 = 16 vector accumulators,
    /// held across `k` (L = 256): 2.6 * (16 + W / 256) * 4 lanes = 169.96 GFLOP/s, above the
    /// peak 100: not latency-bound. The 2x8 tile has 2 * 2 = 4 accumulators:
    /// 2.6 * (4 + 0.34) * 4 = 45.1, latency-bound.
    #[test]
    fn a_register_tile_has_enough_chains_unless_it_is_small() {
        let machine = fast_caches();
        let w = machine.window_fmas();
        let tile = |ri: usize, rj: usize| {
            format!(
                "split(1, i, io, ii, {ri}); split(1, j, jo, ji, {rj}); \
                 reorder(1, io, jo, k, ii, ji); vectorize(1, ji); unroll(1, ii); unroll(1, ji)"
            )
        };
        let want = 2.6 * (16.0 + w / 256.0) * 4.0;
        let got = latency(&machine, &tile(4, 16)).expect("latency ceiling");
        assert!(
            (got - want).abs() < 1e-9 && (got - 169.96).abs() < 1e-2,
            "{got}"
        );
        let p = predict_schedule(&machine, 256, &tile(4, 16)).expect("predicts");
        assert_eq!((p.bound, p.gflops), ("compute", 100.0));
        let small = latency(&machine, &tile(2, 8)).expect("latency ceiling");
        assert!(
            (small - 2.6 * (4.0 + w / 256.0) * 4.0).abs() < 1e-9,
            "{small}"
        );
        let p = predict_schedule(&machine, 256, &tile(2, 8)).expect("predicts");
        assert_eq!(p.bound, "latency");
    }

    /// `i, k, j`: C depends on the innermost loop `j`, every FMA stores its result and the
    /// chain is broken through memory: no latency ceiling, scalar or vectorized.
    #[test]
    fn a_target_that_moves_with_the_innermost_loop_has_no_latency_ceiling() {
        let machine = fast_caches();
        assert_eq!(latency(&machine, ""), None);
        assert_eq!(latency(&machine, "vectorize(1, j)"), None);
        assert_eq!(latency(&machine, "reorder(1, k, i, j)"), None);
    }

    #[test]
    fn a_longer_chain_cannot_be_slower_than_a_single_one() {
        let bad = Machine {
            fma_chain256_gflops: 2.0,
            ..Machine::assumed_m5()
        };
        assert!(predict(&bad, &naive(8)).is_err());
    }

    #[test]
    fn from_reads_the_chain_points_when_the_peak_line_has_them() {
        let line = r#"{"kind":"peak","fma_gflops":{"median":100.5,"n":30},"triad_gbs":{"median":90.25,"n":30},"triad_l2_gbs":{"median":110.0,"n":30},"fma_chain_gflops":{"median":2.5,"n":30},"fma_chain256_gflops":{"median":3.5,"n":30}}"#;
        let (machine, _, _) = machine_from_results(line).expect("parses");
        assert_eq!(
            (machine.fma_chain_gflops, machine.fma_chain256_gflops),
            (2.5, 3.5)
        );
        assert!((machine.window_fmas() - 102.4).abs() < 1e-9);
        let note = from_note(
            "f",
            true,
            true,
            true,
            (true, true),
            true,
            chain_measured(line),
        );
        assert!(!note.contains("fma_chain"), "{note}");
        let line = line.replace("fma_chain256_gflops", "x");
        let (placeholder, _, _) = machine_from_results(&line).expect("parses");
        assert_eq!(
            (
                placeholder.fma_chain_gflops,
                placeholder.fma_chain256_gflops
            ),
            (2.5, PLACEHOLDER_FMA_CHAIN256_GFLOPS)
        );
        let note = from_note(
            "f",
            true,
            true,
            true,
            (true, true),
            true,
            chain_measured(&line),
        );
        assert!(note.contains("fma_chain256 is a PLACEHOLDER"), "{note}");
        assert!(!note.contains("fma_chain is a PLACEHOLDER"), "{note}");
        let header = report(8, &placeholder, &note, &[]);
        assert!(
            header.contains("fma_chain 2.5 GFLOP/s, fma_chain256 3.49 GFLOP/s"),
            "{header}"
        );
    }

    const S2: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../bench/results/2026-10-16-ffce49d.jsonl"
    );

    const S3: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../bench/results/2026-10-18-4928ffa.jsonl"
    );

    /// The S3 machine (mem_acc 31.1262 is from the old, loop-overhead-bound kernel, so the load
    /// ceiling is low), N = 256: the register tiles with the fewest loads per FMA are at the top
    /// of the space. Until v1.7 the top 5 were the 5 tiles with unroll steps (v1.3 pooled loads
    /// and stores and had the 3 with 16 accumulators first, the two with 8 behind the best
    /// non-unrolled schedules, latency-bound) and the next schedule had no unroll. Since v1.8 a
    /// 4x16 tile is first (alone, 123.5, load-bound), and 26 schedules tie behind it at 98.985
    /// (load-bound): the 2x32 and 8x8 tiles with unroll steps, and the schedules with `ii` = 8,
    /// `ji` = 8 and no unroll steps that clang compiles to the same code (`one_vector_iteration`
    /// for `ji`, `ii` unrolled by clang), with and without a `k` split. The 2x16 tile follows
    /// well behind. The 4x16 tile at N = 128 has 0.5078 loads per FMA: load-bound at 31.1262 *
    /// 2 / 0.5078 = 122.6, just below the FMA peak 129.2.
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
        // The 4x16 tile ranks first, then the tie of the 16-accumulator tiles.
        assert!(
            ranked[0]
                .schedule
                .contains("split(1, i, io, ii, 4); split(1, j, jo, ji, 16)")
        );
        assert!(ranked[0].schedule.contains("unroll("));
        let plateau = ranked[1].prediction.gflops;
        assert!((plateau - 98.985).abs() < 1e-3, "{plateau}");
        assert!(ranked[0].prediction.gflops > plateau);
        let tied = |r: &Ranked| (r.prediction.gflops - plateau).abs() < 1e-9;
        assert!(ranked[1..27].iter().all(tied));
        assert!(!tied(&ranked[27]));
        for tile in [
            "ii, 2); split(1, j, jo, ji, 32",
            "ii, 8); split(1, j, jo, ji, 8",
        ] {
            let found = ranked[1..27]
                .iter()
                .any(|r| r.schedule.contains(tile) && r.schedule.contains("unroll("));
            assert!(found, "{tile}");
        }
        let two_by_sixteen = ranked
            .iter()
            .find(|r| {
                r.schedule
                    .contains("split(1, i, io, ii, 2); split(1, j, jo, ji, 16)")
            })
            .expect("the 2x16 tile is in the space");
        assert_eq!(two_by_sixteen.prediction.bound, "load");
        let tile = "split(1, i, io, ii, 4); split(1, j, jo, ji, 16); \
                    reorder(1, io, jo, k, ii, ji); vectorize(1, ji); unroll(1, ii); unroll(1, ji)";
        let p = predict_schedule(&machine, 128, tile).expect("predicts");
        // loads per tile: A 4 + B 4 per k, C 16: (8 * 128 + 16) / (16 * 128) = 0.5078125.
        let want = 31.1262 * 2.0 / 0.5078125;
        assert!((p.gflops / want - 1.0).abs() < 1e-9, "{}", p.gflops);
        assert_eq!(p.bound, "load");
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
        let note = from_note(
            "f",
            mem_acc_measured,
            scalar_measured,
            true,
            (true, true),
            true,
            (true, true),
        );
        assert!(!note.contains("PLACEHOLDER"), "{note}");
        let note = from_note("f", true, false, true, (true, true), true, (true, true));
        assert!(note.contains("mem_acc_scalar is a PLACEHOLDER"), "{note}");
        // The report header carries the scalar point.
        let header = report(8, &machine, &note, &[]);
        assert!(header.contains("mem_acc_scalar 8.5632 GFLOP/s"), "{header}");
    }

    /// `fma_col_reload_gflops` of the peak line replaces the placeholder 2.1 (the note stops
    /// calling it one and the header shows it); without it the placeholder stays and the note says
    /// so.
    #[test]
    fn from_reads_col_reload_when_the_peak_line_has_it() {
        let line = r#"{"kind":"peak","fma_gflops":{"median":100.5,"n":30},"triad_gbs":{"median":90.25,"n":30},"triad_l2_gbs":{"median":110.0,"n":30},"fma_col_reload_gflops":{"median":2.5,"n":30}}"#;
        let missing = line.replace("fma_col_reload_gflops", "x");
        let (with, _, _) = machine_from_results(line).expect("parses");
        let (without, _, _) = machine_from_results(&missing).expect("parses");
        assert_eq!(PLACEHOLDER_COL_RELOAD_GFLOPS, 2.1);
        assert_eq!(
            (with.col_reload_gflops, without.col_reload_gflops),
            (2.5, 2.1)
        );
        assert!(col_reload_measured(line) && !col_reload_measured(&missing));
        let note = from_note("f", true, true, true, (true, true), true, (true, true));
        assert!(!note.contains("PLACEHOLDER"), "{note}");
        let note = from_note("f", true, true, true, (false, true), true, (true, true));
        assert!(note.contains("col_reload 2.1 is a PLACEHOLDER"), "{note}");
        let header = report(8, &with, &note, &[]);
        assert!(header.contains("col_reload 2.5 GFLOP/s"), "{header}");
        let bad = Machine {
            col_reload_gflops: 0.0,
            ..with
        };
        assert!(predict(&bad, &naive(8)).is_err());
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
        let report = compare_report("S2", &text, None).expect("compares");
        assert!(report.contains("13 records"), "{report}");
        assert!(report.contains("skipped: 2 matmul records flagged drift; 15 records"));
        assert!(report.contains("PLACEHOLDER"), "{report}");
        assert!(report.contains("Spearman rank correlation 0."), "{report}");
        // The bound is the fifth column of a row.
        let rows = report.lines().filter(|l| {
            let bound = l.split_whitespace().nth(4);
            bound.is_some_and(|b| ["ldst", "load", "store", "l2"].contains(&b))
        });
        assert_eq!(rows.count(), 13, "{report}");
    }

    #[test]
    fn schedule_groups_follow_the_vectorize_width_or_the_innermost_loop() {
        let split = "split(1, i, io, ii, 8); split(1, j, jo, ji, 64); split(1, k, ko, ki, 8); ";
        let vec64 = format!("{split}reorder(1, io, ko, jo, ii, ki, ji); vectorize(1, ji)");
        assert_eq!(schedule_group(&vec64), 1);
        assert_eq!(schedule_group(&vec64.replace("ji, 64", "ji, 128")), 0);
        assert_eq!(schedule_group(&vec64.replace("ji, 64", "ji, 16")), 2);
        assert_eq!(
            schedule_group(&format!("{split}reorder(1, io, ko, jo, ii, ki, ji)")),
            3
        );
        assert_eq!(
            schedule_group(&format!("{split}reorder(1, io, ko, jo, ji, ki, ii)")),
            4
        );
        assert_eq!(
            schedule_group(&format!("{split}reorder(1, io, ko, jo, ji, ii, ki)")),
            5
        );
        assert_eq!(schedule_group(""), 5);
        assert_eq!(median(&[3.0, 1.0, 2.0, 10.0]), Some(2.5));
        assert_eq!(median(&[]), None);
    }

    /// #64's numbers from one call: the 1000 random records of run 3 on the P3 peak line give
    /// Spearman 0.853 and the group errors ×1.19 / ×1.42 / ×0.73 (medians of pred / meas). v1.3
    /// had ×1.495 for `vectorized ji = 32/64`; the separate load and store ceilings (v1.4) bring
    /// it into [0.83; 1.20]. With the unrolled scalar point (this peak line has no
    /// `fma_mem_acc_scalar_unrolled_gflops`, so the placeholder 23.2) the Spearman is 0.868 and
    /// the group `"vectorized" ji <= 16 (scalar code)` moves from ×0.73 to ×1.16; the group
    /// `scalar, ii innermost` (strided, rolled point) stays at ×1.42. v1.6 (the column-reload
    /// ceiling, placeholder 2.1 here as this peak line has no `fma_col_reload_gflops`) moves the
    /// Spearman to 0.920 (0.923 with the 13 other records) and that group to ×1.20. v1.7 (the
    /// unrolled-column-load ceiling, placeholder 15.7 here as this peak line has no
    /// `fma_col_unrolled_gflops`) moves the Spearman to 0.924 (0.926 with the 13 other records)
    /// and that group to ×1.12.
    ///
    /// From v1.8 the `"vectorized" ji <= 16` group compares predictions for the NEW codegen
    /// (#96: a power-of-two `ji` is one vector iteration, `one_vector_iteration`) with
    /// measurements of the OLD one (run 3 is from before #96, those loops were scalar code):
    /// the group's error moves from ×1.159 to ×2.139 and the Spearman from 0.924 to 0.905 (the
    /// 13 other records: 0.926 to 0.908). The other groups' numbers do not move. This is not a
    /// model error to fix: a run measured with the new codegen is the comparison for them.
    #[test]
    fn compare_by_group_reproduces_the_run_3_numbers_of_64() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../bench/results/");
        let read = |f: &str| std::fs::read_to_string(format!("{dir}{f}")).expect("committed");
        let peak = read("2026-10-21-561ede3-peak.jsonl");
        let peak = peak
            .lines()
            .find(|l| l.contains("\"kind\":\"peak\""))
            .expect("peak");
        let text = format!("{peak}\n{}", read("2026-10-22-hypothesis-a-run3.jsonl"));
        let report = compare_report("run 3", &text, Some("random")).expect("compares");
        assert!(report.contains("1000 records"), "{report}");
        assert!(
            report.contains("skipped 13 matmul records of other groups"),
            "{report}"
        );
        assert!(
            report.contains("Spearman rank correlation 0.905;"),
            "{report}"
        );
        for (group, n, error) in [
            ("vectorized ji = 32/64", 106, "1.193"),
            ("scalar, ii innermost", 263, "1.119"),
            ("\"vectorized\" ji <= 16 (scalar code)", 100, "2.139"),
        ] {
            let row = report
                .lines()
                .find(|l| l.starts_with(group))
                .expect("group row");
            let cols: Vec<&str> = row[group.len()..].split_whitespace().collect();
            assert_eq!(cols[0], n.to_string(), "{row}");
            assert_eq!(cols[3], error, "{row}");
        }
        let sizes: usize = SCHEDULE_GROUPS
            .iter()
            .map(|g| {
                let row = report
                    .lines()
                    .find(|l| l.starts_with(g))
                    .expect("group row");
                row[g.len()..]
                    .split_whitespace()
                    .next()
                    .unwrap()
                    .parse::<usize>()
                    .unwrap()
            })
            .sum();
        assert_eq!(sizes, 1000, "{report}");
        let all = compare_report("run 3", &text, None).expect("compares");
        assert!(
            all.contains("1013 records") && all.contains("correlation 0.908;"),
            "{all}"
        );
    }

    /// Pins the full-unroll rule of v1.3 (#63) on four schedules of hypothesis A run 3
    /// (`bench/results/2026-10-22-hypothesis-a-run3.jsonl`, N = 256) in which clang fully
    /// unrolls a loop the schedule did not unroll. Predictions on `Machine::assumed_m5`.
    /// Without the rule (`fully_unrolled` = only the schedule's unrolls) the model predicts
    /// 100 / 82.3 / 71.9 / 100 GFLOP/s for these rows and this test fails; the run-3 medians
    /// are 23.3 / 26.7 / 9.7 / 19.9. The two vectorized rows have loads 2.0625 per FMA and 1
    /// store: the load ceiling is 60 * 2 / 2.0625 = 58.18, the store ceiling 58 binds (v1.3, one
    /// pool: 60 * 3 / 3.0625 = 58.78).
    ///
    /// Rows 3 and 4 (`ji` = 8, vectorized) were scalar unrolled code until v1.7 (rule K): row 3
    /// 2.12890625 mem ops, 25.0 GFLOP/s, "compute" (ld/st ceiling 23.2 * 3 / 2.12890625 = 32.69
    /// above the scalar FMA peak 25), row 4 1.140625, 22.58, "latency". Since v1.8 `ji` is one
    /// vector iteration (`one_vector_iteration`), clang does not unroll it, and they pin the
    /// new numbers; run 3 measured them with the old codegen (the medians above). In row 4
    /// clang unrolls `ko` = 2 instead (2 * (5 + 5 + 9) = 48 <= 150: two copies of the
    /// vectorized loop in the IR, clang remarks, Apple clang 21), which changes no count.
    /// - row 3, `k, jo, i, ji`: per `(k, jo, i)` C 2 vectors loaded and stored, A 1 load, B 2
    ///   vectors held across `i` (2 / 256 per `i`): (2 + 1 + 2 / 256 + 2) / 2 = 2.50390625; the
    ///   store ceiling 58 / 1 binds (the load ceiling is 60 * 2 / 1.50390625 = 79.79).
    /// - row 4, `jo, ko, i, ki, ji` with `ki` = 128: per `(jo, ko, i)` C 2 + 2, A 128, B 256
    ///   loads over 256 vector FMAs: 388 / 256 = 1.515625; C is held across `ki` in 2 vector
    ///   chains of 128: the latency ceiling 4 * 2.6 * (2 + W / 128) = 4 * 2 * 3.49 = 27.92 binds (the load
    ///   ceiling is 60 * 2 / (386 / 256) = 79.58).
    #[test]
    fn run_3_schedules_with_a_full_unroll_keep_their_unroll_rule() {
        let machine = Machine::assumed_m5();
        // (schedule, loops clang unrolls, mem ops per FMA, GFLOP/s, bound)
        let rows: [(&str, &[&str], f64, f64, &str); 4] = [
            (
                "split(1, i, io, ii, 16); split(1, j, jo, ji, 64); split(1, k, ko, ki, 8); \
                 reorder(1, io, ko, jo, ii, ki, ji); vectorize(1, ji)",
                &["ki"],
                3.0625,
                58.0,
                "store",
            ),
            (
                "split(1, i, io, ii, 8); split(1, j, jo, ji, 64); reorder(1, io, jo, k, ii, ji); \
                 vectorize(1, ji)",
                &["ii"],
                3.0625,
                58.0,
                "store",
            ),
            (
                "split(1, j, jo, ji, 8); reorder(1, k, jo, i, ji); vectorize(1, ji)",
                &[],
                2.50390625,
                58.0,
                "store",
            ),
            (
                "split(1, j, jo, ji, 8); split(1, k, ko, ki, 128); reorder(1, jo, ko, i, ki, ji); \
                 vectorize(1, ji)",
                &["ko"],
                1.515625,
                8.0 * 3.49,
                "latency",
            ),
        ];
        for (schedule, unrolled, m, gflops, bound) in rows {
            assert_eq!(clang_unrolled(schedule), unrolled, "{schedule}");
            assert_eq!(mem_ops(256, schedule), m, "{schedule}");
            let p = predict_schedule(&machine, 256, schedule).expect("predicts");
            assert!(
                (p.gflops / gflops - 1.0).abs() < 1e-9,
                "{schedule}: {}",
                p.gflops
            );
            assert_eq!(p.bound, bound, "{schedule}");
        }
    }
}
