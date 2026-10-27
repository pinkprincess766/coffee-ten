//! Builds the kernel and its variants, checks them against the naive one, times them.

use crate::BenchError;
use crate::cli::{Config, KernelKind};
use crate::meta::Metadata;
use crate::stats::{Rng, Stats, drifted, summarize};
use coffee_ten_codegen_c::{BENCH_FLAGS, CompiledKernel, compile_with_flags};
use coffee_ten_ir::{BinaryOp, Graph, Shape, UnaryOp};
use coffee_ten_loop::{Kernel, OrderClass, Schedule, apply_classified, lower};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// u = 2^-24, the unit roundoff of f32.
const UNIT_ROUNDOFF: f64 = 1.0 / 16_777_216.0;
/// The smallest positive subnormal f32, 2^-149.
const MIN_SUBNORMAL: f64 = f32::from_bits(1) as f64;

/// gamma_n = n*u / (1 - n*u), Higham's bound factor for a sum of n terms.
fn gamma(n: usize) -> f64 {
    let nu = n as f64 * UNIT_ROUNDOFF;
    nu / (1.0 - nu)
}

/// Spec §5.4: how far an f32 `k`-term dot product plus bias may be from the exact value,
/// doubled so that two results (a variant and the naive kernel) may differ by this much.
/// `scale` is `sum |a_k * b_k| + |bias|`. FMA contraction does not change the bound.
pub fn error_bound(k: usize, scale: f64) -> f64 {
    2.0 * (gamma(k + 1) * scale + k as f64 * MIN_SUBNORMAL)
}

/// `Ok` if every element of `got` is within `error_bound(k, scale[i])` of `naive[i]`.
/// A NaN in `got` fails.
pub fn check_outputs(
    variant: &str,
    got: &[f32],
    naive: &[f32],
    scales: &[f64],
    k: usize,
) -> Result<(), BenchError> {
    for (index, ((&g, &r), &scale)) in got.iter().zip(naive).zip(scales).enumerate() {
        let bound = error_bound(k, scale);
        let difference = (f64::from(g) - f64::from(r)).abs();
        if difference.is_nan() || difference > bound {
            return Err(BenchError::Mismatch {
                variant: variant.to_string(),
                index,
                got: g,
                naive: r,
                bound,
            });
        }
    }
    Ok(())
}

/// Seeded inputs of a kernel. `bias` is only used by `ReluMatmulBias`.
struct Inputs {
    a: Vec<f32>,
    b: Vec<f32>,
    bias: Vec<f32>,
}

impl Inputs {
    fn new(n: usize, seed: u64) -> Inputs {
        let mut rng = Rng::new(seed);
        let mut fill =
            |len: usize| -> Vec<f32> { (0..len).map(|_| rng.next_signed_unit_f32()).collect() };
        Inputs {
            a: fill(n * n),
            b: fill(n * n),
            bias: fill(n),
        }
    }

    /// One Vec per kernel buffer, in `Kernel::buffers` order. Scratch and `out` start as
    /// NaN, so a missing init nest shows up in the check.
    fn storage(&self, kernel: &Kernel) -> Vec<Vec<f32>> {
        kernel
            .buffers
            .iter()
            .map(|buffer| match buffer.name.as_str() {
                "A" => self.a.clone(),
                "B" => self.b.clone(),
                "bias" => self.bias.clone(),
                _ => vec![f32::NAN; buffer.shape.iter().product()],
            })
            .collect()
    }

    /// `sum_k |A[i,k]| |B[k,j]| (+ |bias[j]|)` per output element, in f64.
    fn bound_scales(&self, n: usize, with_bias: bool) -> Vec<f64> {
        let mut scales = vec![0.0f64; n * n];
        for i in 0..n {
            for k in 0..n {
                let a_ik = f64::from(self.a[i * n + k]).abs();
                for j in 0..n {
                    scales[i * n + j] += a_ik * f64::from(self.b[k * n + j]).abs();
                }
            }
            if with_bias {
                for j in 0..n {
                    scales[i * n + j] += f64::from(self.bias[j]).abs();
                }
            }
        }
        scales
    }
}

fn build_kernel(kind: KernelKind, n: usize) -> Result<Kernel, BenchError> {
    let mut g = Graph::new();
    let shape = Shape::new(vec![n, n]);
    let setup = |e: &dyn std::fmt::Display| BenchError::Setup(format!("cannot build graph: {e}"));
    let a = g.input("A", shape.clone()).map_err(|e| setup(&e))?;
    let b = g.input("B", shape).map_err(|e| setup(&e))?;
    let output = match kind {
        KernelKind::Matmul => g.matmul(a, b).map_err(|e| setup(&e))?,
        KernelKind::ReluMatmulBias => {
            let bias = g
                .input("bias", Shape::new(vec![n]))
                .map_err(|e| setup(&e))?;
            let product = g.matmul(a, b).map_err(|e| setup(&e))?;
            let sum = g
                .binary(BinaryOp::Add, product, bias)
                .map_err(|e| setup(&e))?;
            g.unary(UnaryOp::Relu, sum).map_err(|e| setup(&e))?
        }
    };
    lower(&g, output, "bench_kernel").map_err(|e| BenchError::Setup(format!("cannot lower: {e}")))
}

/// A variant ready to time: compiled, with its own buffers. `Reference::build` frees the
/// buffers after the check and `Reference::time` makes them again, so many built variants
/// do not hold many matrices.
pub struct Variant {
    text: String,
    schedule: Schedule,
    order_class: OrderClass,
    kernel: Kernel,
    compiled: CompiledKernel,
    storage: Vec<Vec<f32>>,
}

impl Variant {
    fn run_once(&mut self) -> Result<(), BenchError> {
        let mut buffers: Vec<&mut [f32]> = self.storage.iter_mut().map(Vec::as_mut_slice).collect();
        self.compiled
            .run(&mut buffers)
            .map_err(|e| BenchError::Setup(format!("variant {:?}: {e}", self.text)))
    }

    /// The output: `out` is the last buffer of every lowered kernel.
    fn output(&self) -> &[f32] {
        self.storage.last().map_or(&[], Vec::as_slice)
    }
}

fn cache_dir() -> PathBuf {
    std::env::temp_dir().join("coffee-ten-bench-cache")
}

fn make_variant(
    naive: &Kernel,
    text: &str,
    schedule: &Schedule,
    inputs: &Inputs,
) -> Result<Variant, BenchError> {
    let setup = |what: &str, e: &dyn std::fmt::Display| {
        BenchError::Setup(format!("variant {text:?}: {what}: {e}"))
    };
    let (kernel, order_class) =
        apply_classified(naive, schedule).map_err(|e| setup("cannot apply schedule", &e))?;
    let compiled = compile_with_flags(&kernel, &cache_dir(), BENCH_FLAGS)
        .map_err(|e| setup("cannot compile", &e))?;
    let storage = inputs.storage(&kernel);
    Ok(Variant {
        text: text.to_string(),
        schedule: schedule.clone(),
        order_class,
        kernel,
        compiled,
        storage,
    })
}

/// One sample: runs the kernel until at least `min_sample` has passed, returns GFLOP/s.
fn time_sample(
    variant: &mut Variant,
    flop_per_call: f64,
    min_sample: Duration,
) -> Result<f64, BenchError> {
    let start = Instant::now();
    let mut calls = 0u64;
    let elapsed = loop {
        variant.run_once()?;
        calls += 1;
        let elapsed = start.elapsed();
        if elapsed >= min_sample {
            break elapsed;
        }
    };
    Ok(flop_per_call * calls as f64 / elapsed.as_secs_f64() / 1e9)
}

/// Each variant runs this long before round 1. Measured on an Apple M5 with `n = 256`: after the
/// thread has been idle (waiting for clang, or a pause of 50 ms or more) the same kernel runs at
/// 25-50% of its speed and reaches 95% of the steady rate after about 100 ms, for the naive
/// kernel, a vectorized one and a register tile alike. 300 ms is three times that.
pub const WARMUP_MS: u64 = 300;
/// The warm-up of a `--quick` run, which is not a measurement.
pub const QUICK_WARMUP_MS: u64 = 5;
/// The warm-up of a `search` candidate that is timed right after another one, so the thread has
/// not been idle and the CPU is at its steady rate. Measured on an Apple M5 with `n = 256`
/// (78 cached candidates in 2 runs): the first call of a newly loaded library ran at 0.95-1.05
/// of its rate 80-150 ms later in 85% of the cases, and no slow ramp was seen; the others were
/// single calls disturbed by other load. So it only has to make a call or two: one sample long.
/// The first candidate of a timing phase has `WARMUP_MS`: after the parallel compilation the
/// thread was idle, and its rate went from 9 to 29 GFLOP/s over 150 ms.
pub const CANDIDATE_WARMUP_MS: u64 = 10;

/// `WARMUP_MS`, or `QUICK_WARMUP_MS` for a `--quick` run.
pub fn full_warmup(quick: bool) -> Duration {
    Duration::from_millis(if quick { QUICK_WARMUP_MS } else { WARMUP_MS })
}

/// `CANDIDATE_WARMUP_MS`, or `QUICK_WARMUP_MS` for a `--quick` run.
pub fn candidate_warmup(quick: bool) -> Duration {
    Duration::from_millis(if quick {
        QUICK_WARMUP_MS
    } else {
        CANDIDATE_WARMUP_MS
    })
}

/// Calls `call` until `elapsed` (time since the warm-up began) reaches `warmup`; at least one
/// call. Returns the number of calls.
fn warm_up<E>(
    warmup: Duration,
    mut call: impl FnMut() -> Result<(), E>,
    mut elapsed: impl FnMut() -> Duration,
) -> Result<u64, E> {
    let mut calls = 0u64;
    loop {
        call()?;
        calls += 1;
        if elapsed() >= warmup {
            return Ok(calls);
        }
    }
}

/// What a run measures; the same for every record of the run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RunInfo {
    pub kernel: &'static str,
    /// Matrix size.
    pub n: usize,
    /// FNV-1a of the lowered kernel's text before any schedule.
    pub graph_hash: u64,
    /// The run was `--quick`: not a measurement.
    pub quick: bool,
}

/// What is reported for one variant.
#[derive(Clone, Debug, PartialEq)]
pub struct Record {
    pub kernel: &'static str,
    /// Matrix size.
    pub n: usize,
    pub graph_hash: u64,
    /// The variant as typed.
    pub variant: String,
    /// Canonical schedule text, "" for the naive kernel.
    pub schedule: String,
    pub order_class: OrderClass,
    /// Always true: a variant that fails the check aborts the run before it is recorded.
    pub verified: bool,
    pub stats: Stats,
    /// Number of samples.
    pub samples: usize,
    /// Median of the first rounds differs from the median of the last by more than `DRIFT_LIMIT`.
    pub drift: bool,
    /// The run was `--quick`: not a measurement.
    pub quick: bool,
}

/// The one place where a record per variant is assembled; a JSONL line is written from it.
/// Only called for variants that passed the check. `rates` in round order, GFLOP/s. `None` if
/// there are no samples.
pub fn make_record(
    info: &RunInfo,
    variant: &str,
    schedule: &Schedule,
    order_class: OrderClass,
    rates: &[f64],
) -> Option<Record> {
    Some(Record {
        kernel: info.kernel,
        n: info.n,
        graph_hash: info.graph_hash,
        variant: variant.to_string(),
        schedule: schedule.to_string(),
        order_class,
        verified: true,
        stats: summarize(rates)?,
        samples: rates.len(),
        drift: drifted(rates),
        quick: info.quick,
    })
}

/// 64-bit FNV-1a.
pub fn fnv1a_64(bytes: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    bytes.iter().fold(OFFSET_BASIS, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(PRIME)
    })
}

/// `text` as a JSON string literal: `"` and `\` escaped, `\n` and `\t` as such, other control
/// characters as `\u00XX`.
fn json_string(text: &str) -> String {
    let mut out = String::from("\"");
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// A finite number as JSON with the shortest round-trip digits. NaN and infinities have no
/// JSON form and become `null`.
fn json_number(x: f64) -> String {
    if x.is_finite() {
        format!("{x}")
    } else {
        "null".to_string()
    }
}

fn order_class_name(class: OrderClass) -> &'static str {
    match class {
        OrderClass::Preserving => "preserving",
        OrderClass::Reordering => "reordering",
    }
}

/// One JSONL line of the log: the record and the run's metadata, keys in log-format order.
pub fn record_json(record: &Record, metadata: &Metadata) -> String {
    let fields = [
        format!("\"kernel\":{}", json_string(record.kernel)),
        format!("\"n\":{}", record.n),
        format!("\"graph_hash\":\"{:016x}\"", record.graph_hash),
        format!("\"schedule\":{}", json_string(&record.schedule)),
        format!(
            "\"order_class\":{}",
            json_string(order_class_name(record.order_class))
        ),
        format!("\"verified\":{}", record.verified),
        format!("\"quick\":{}", record.quick),
        format!("\"samples\":{}", record.samples),
        format!("\"median\":{}", json_number(record.stats.median)),
        format!("\"p25\":{}", json_number(record.stats.p25)),
        format!("\"p75\":{}", json_number(record.stats.p75)),
        format!("\"min\":{}", json_number(record.stats.min)),
        format!("\"drift\":{}", record.drift),
        format!("\"unit\":{}", json_string("GFLOP/s")),
        format!("\"seed\":{}", metadata.seed),
        format!("\"commit\":{}", json_string(&metadata.git_rev)),
        format!(
            "\"dirty\":{}",
            metadata
                .git_dirty
                .map_or("null".to_string(), |d| d.to_string())
        ),
        format!("\"clang\":{}", json_string(&metadata.clang)),
        format!("\"flags\":{}", json_string(&metadata.flags)),
        format!("\"machine\":{}", json_string(&metadata.hw_model)),
        format!("\"cpu\":{}", json_string(&metadata.cpu_brand)),
        format!("\"os\":{}", json_string(&metadata.os_version)),
        format!("\"date\":{}", json_string(&metadata.date_utc)),
    ];
    format!("{{{}}}", fields.join(","))
}

/// Builds the naive kernel and every variant, checks each variant's output against the naive
/// one, then times them. One record per variant, in `config.variants` order.
pub fn run(config: &Config) -> Result<Vec<Record>, BenchError> {
    let reference = Reference::new(config)?;
    let mut variants = Vec::new();
    for (text, schedule) in &config.variants {
        variants.push(reference.build(text, schedule)?);
    }
    reference.time(config, &mut variants, full_warmup(config.quick))
}

/// What variants are checked against and timed with: the seeded inputs and the naive kernel's
/// output, computed once per run (`config.variants` is not used).
pub struct Reference {
    naive_kernel: Kernel,
    info: RunInfo,
    inputs: Inputs,
    scales: Vec<f64>,
    output: Vec<f32>,
}

#[cfg(test)]
thread_local! {
    /// How many references this thread has built; a test reads it.
    pub(crate) static REFERENCES_BUILT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

impl Reference {
    /// Builds the naive kernel, runs it once and keeps its output.
    pub fn new(config: &Config) -> Result<Reference, BenchError> {
        let n = config.n;
        let naive_kernel = build_kernel(config.kernel, n)?;
        let info = RunInfo {
            kernel: config.kernel.name(),
            n,
            graph_hash: fnv1a_64(naive_kernel.to_string().as_bytes()),
            quick: config.quick,
        };
        let inputs = Inputs::new(n, config.seed);
        let with_bias = config.kernel == KernelKind::ReluMatmulBias;
        let scales = inputs.bound_scales(n, with_bias);
        let mut naive = make_variant(&naive_kernel, "", &Schedule::default(), &inputs)?;
        naive.run_once()?;
        #[cfg(test)]
        REFERENCES_BUILT.with(|built| built.set(built.get() + 1));
        Ok(Reference {
            naive_kernel,
            info,
            inputs,
            scales,
            output: naive.output().to_vec(),
        })
    }

    /// Compiles `schedule` and checks its output against the naive one. The buffers are freed.
    pub fn build(&self, text: &str, schedule: &Schedule) -> Result<Variant, BenchError> {
        let mut variant = make_variant(&self.naive_kernel, text, schedule, &self.inputs)?;
        variant.run_once()?;
        let n = self.info.n;
        check_outputs(text, variant.output(), &self.output, &self.scales, n)?;
        variant.storage = Vec::new();
        Ok(variant)
    }

    /// `build` of each variant on all cores; one result per variant, in the order given.
    pub fn build_all(&self, variants: &[(String, Schedule)]) -> Vec<Result<Variant, BenchError>> {
        let cores = std::thread::available_parallelism().map_or(1, usize::from);
        parallel_map(variants, cores, |(text, schedule)| {
            self.build(text, schedule)
        })
    }

    /// Warms every variant up for `warmup` in a random order, then takes `config.samples`
    /// rounds of one sample per variant in random order. One record per variant, in the order
    /// given.
    pub fn time(
        &self,
        config: &Config,
        variants: &mut [Variant],
        warmup: Duration,
    ) -> Result<Vec<Record>, BenchError> {
        for variant in variants.iter_mut() {
            variant.storage = self.inputs.storage(&variant.kernel);
        }
        let flop_per_call = 2.0 * (self.info.n as f64).powi(3);
        let min_sample = Duration::from_millis(config.min_sample_ms);
        let mut rng = Rng::new(config.seed);
        let mut order: Vec<usize> = (0..variants.len()).collect();
        let mut rates: Vec<Vec<f64>> = vec![Vec::with_capacity(config.samples); variants.len()];
        // After the thread has been idle (waiting for clang, a pause) a kernel runs slow for
        // ~100 ms (see `WARMUP_MS`): every variant warms up in the interleaved order before
        // round 1, and the warm-up is not recorded.
        rng.shuffle(&mut order);
        for &index in &order {
            let start = Instant::now();
            warm_up(warmup, || variants[index].run_once(), || start.elapsed())?;
        }
        for _ in 0..config.samples {
            rng.shuffle(&mut order);
            for &index in &order {
                let rate = time_sample(&mut variants[index], flop_per_call, min_sample)?;
                rates[index].push(rate);
            }
        }

        variants
            .iter()
            .zip(&rates)
            .map(|(variant, rates)| {
                make_record(
                    &self.info,
                    &variant.text,
                    &variant.schedule,
                    variant.order_class,
                    rates,
                )
                .ok_or_else(|| BenchError::Setup("no samples".to_string()))
            })
            .collect()
    }
}

/// `f` of every item on `threads` threads (at least one); results in the order of `items`.
fn parallel_map<T: Sync, R: Send>(
    items: &[T],
    threads: usize,
    f: impl Fn(&T) -> R + Sync,
) -> Vec<R> {
    let next = AtomicUsize::new(0);
    let worker = || {
        let mut done = Vec::new();
        loop {
            let index = next.fetch_add(1, Ordering::Relaxed);
            let Some(item) = items.get(index) else {
                return done;
            };
            done.push((index, f(item)));
        }
    };
    let mut done: Vec<(usize, R)> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads.clamp(1, items.len().max(1)))
            .map(|_| scope.spawn(worker))
            .collect();
        workers
            .into_iter()
            .flat_map(|w| {
                w.join()
                    .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
            })
            .collect()
    });
    done.sort_by_key(|(index, _)| *index);
    done.into_iter().map(|(_, result)| result).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warm_up_runs_until_the_warmup_time_has_passed() {
        // A fake clock that advances 1 ms per call: WARMUP_MS ms need WARMUP_MS calls.
        let calls = std::cell::Cell::new(0u64);
        let done = warm_up::<()>(
            Duration::from_millis(WARMUP_MS),
            || {
                calls.set(calls.get() + 1);
                Ok(())
            },
            || Duration::from_millis(calls.get()),
        );
        assert_eq!(done, Ok(WARMUP_MS));
        assert_eq!(calls.get(), WARMUP_MS);
    }

    #[test]
    fn warm_up_calls_once_for_a_zero_warmup_and_stops_on_error() {
        let zero = warm_up::<()>(Duration::ZERO, || Ok(()), || Duration::ZERO);
        assert_eq!(zero, Ok(1));
        let failed = warm_up(Duration::from_secs(1), || Err("boom"), || Duration::ZERO);
        assert_eq!(failed, Err("boom"));
    }

    #[test]
    fn min_subnormal_is_two_to_the_minus_149() {
        assert_eq!(MIN_SUBNORMAL, 2f64.powi(-149));
    }

    #[test]
    fn bound_matches_the_spec_formula_for_one_case() {
        // K = 3, scale 2: 2 * (gamma_4 * 2 + 3 * 2^-149), gamma_4 = 4u / (1 - 4u).
        let u = 2f64.powi(-24);
        let gamma_4 = 4.0 * u / (1.0 - 4.0 * u);
        let want = 2.0 * (gamma_4 * 2.0 + 3.0 * 2f64.powi(-149));
        assert_eq!(error_bound(3, 2.0), want);
    }

    #[test]
    fn check_accepts_just_inside_and_rejects_just_outside_the_bound() {
        let (k, scale) = (4, 1.0);
        let bound = error_bound(k, scale);
        let naive = [0.0f32];
        let inside = [(bound * 0.99) as f32];
        let outside = [(bound * 1.01) as f32];
        assert_eq!(check_outputs("v", &inside, &naive, &[scale], k), Ok(()));
        assert_eq!(
            check_outputs("v", &[-inside[0]], &naive, &[scale], k),
            Ok(())
        );
        match check_outputs("v", &outside, &naive, &[scale], k) {
            Err(BenchError::Mismatch { index: 0, .. }) => {}
            other => panic!("expected a mismatch, got {other:?}"),
        }
        assert!(check_outputs("v", &[-outside[0]], &naive, &[scale], k).is_err());
    }

    #[test]
    fn check_rejects_nan_and_names_the_variant() {
        let err = check_outputs(
            "split(1, j, a, b, 4)",
            &[1.0, f32::NAN],
            &[1.0, 1.0],
            &[1.0; 2],
            2,
        )
        .unwrap_err();
        let message = err.to_string();
        assert!(message.contains("split(1, j, a, b, 4)"), "{message}");
        assert!(message.contains("out[1]"), "{message}");
    }

    #[test]
    fn bound_scales_add_abs_products_and_abs_bias() {
        let inputs = Inputs {
            a: vec![1.0, -2.0, 3.0, -4.0],
            b: vec![-1.0, 2.0, 3.0, -5.0],
            bias: vec![-0.5, 0.25],
        };
        // |A| @ |B| = [[1*1+2*3, 1*2+2*5], [3*1+4*3, 3*2+4*5]] = [[7, 12], [15, 26]].
        assert_eq!(inputs.bound_scales(2, false), [7.0, 12.0, 15.0, 26.0]);
        assert_eq!(inputs.bound_scales(2, true), [7.5, 12.25, 15.5, 26.25]);
    }

    #[test]
    fn record_is_assembled_from_rates_in_round_order() {
        let info = RunInfo {
            kernel: "matmul",
            n: 8,
            graph_hash: 0,
            quick: true,
        };
        let schedule = Schedule::parse("vectorize(1,j)").unwrap();
        let rates: Vec<f64> = (1..=10).map(f64::from).collect();
        let record = make_record(
            &info,
            "vectorize(1,j)",
            &schedule,
            OrderClass::Reordering,
            &rates,
        )
        .unwrap();
        assert_eq!(record.samples, 10);
        assert_eq!(record.stats.median, 5.0);
        assert!(record.quick);
        assert!(record.verified);
        // The canonical text is written to the log, not the text as typed.
        assert_eq!(record.schedule, "vectorize(1, j)");
        assert_eq!(record.variant, "vectorize(1,j)");
        // First 5 rounds have median 3, the last 5 have median 8.
        assert!(record.drift);
        assert_eq!(
            make_record(&info, "", &schedule, OrderClass::Preserving, &[]),
            None
        );
    }

    #[test]
    fn json_string_escapes_quote_backslash_and_control_characters() {
        assert_eq!(json_string("a\"b\\c"), r#""a\"b\\c""#);
        assert_eq!(json_string("a\nb"), r#""a\nb""#);
        assert_eq!(json_string("a\tb"), r#""a\tb""#);
        assert_eq!(json_string("a\u{1}b"), r#""a\u0001b""#);
        // 0x1f is the last control character; the space after it is printed as is.
        assert_eq!(json_string("\u{1f} "), r#""\u001f ""#);
    }

    #[test]
    fn non_finite_numbers_become_null() {
        assert_eq!(json_number(f64::NAN), "null");
        assert_eq!(json_number(f64::INFINITY), "null");
        assert_eq!(json_number(f64::NEG_INFINITY), "null");
    }

    #[test]
    fn fnv1a_of_known_inputs() {
        assert_eq!(format!("{:016x}", fnv1a_64(b"")), "cbf29ce484222325");
        assert_eq!(format!("{:016x}", fnv1a_64(b"a")), "af63dc4c8601ec8c");
    }

    #[test]
    fn record_json_is_one_line_with_the_log_keys_in_order() {
        let record = Record {
            kernel: "matmul",
            n: 256,
            graph_hash: 0x0123_4567_89ab_cdef,
            variant: "split(1, j, jo, ji, 8)".to_string(),
            schedule: "split(1, j, jo, ji, 8)".to_string(),
            order_class: OrderClass::Preserving,
            verified: true,
            stats: Stats {
                median: 12.5,
                p25: 12.0,
                p75: 13.0,
                min: 11.25,
            },
            samples: 30,
            drift: false,
            quick: false,
        };
        let metadata = Metadata {
            hw_model: "Mac16,1".to_string(),
            cpu_brand: "Apple M4".to_string(),
            os_version: "27.0".to_string(),
            clang: "Apple clang 17".to_string(),
            flags: "-std=c99 -O2".to_string(),
            git_rev: "abc1234".to_string(),
            git_dirty: Some(false),
            date_utc: "2026-10-11T00:00:00Z".to_string(),
            seed: 1,
        };
        assert_eq!(
            record_json(&record, &metadata),
            r#"{"kernel":"matmul","n":256,"graph_hash":"0123456789abcdef","schedule":"split(1, j, jo, ji, 8)","order_class":"preserving","verified":true,"quick":false,"samples":30,"median":12.5,"p25":12,"p75":13,"min":11.25,"drift":false,"unit":"GFLOP/s","seed":1,"commit":"abc1234","dirty":false,"clang":"Apple clang 17","flags":"-std=c99 -O2","machine":"Mac16,1","cpu":"Apple M4","os":"27.0","date":"2026-10-11T00:00:00Z"}"#
        );
    }

    #[test]
    fn parallel_map_keeps_the_order_and_uses_several_threads() {
        // Item i sleeps (8 - i) * 5 ms, so with 4 threads the later items finish first.
        let items: Vec<u64> = (0..8).collect();
        let out = parallel_map(&items, 4, |&i| {
            std::thread::sleep(Duration::from_millis((8 - i) * 5));
            (i * i, std::thread::current().id())
        });
        let squares: Vec<u64> = out.iter().map(|(square, _)| *square).collect();
        assert_eq!(squares, [0, 1, 4, 9, 16, 25, 36, 49]);
        let threads: std::collections::HashSet<_> = out.iter().map(|(_, id)| *id).collect();
        assert!(threads.len() > 1, "{} thread(s)", threads.len());
        // No threads asked for still runs everything, and nothing to do gives nothing.
        assert_eq!(parallel_map(&items, 0, |&i| i), items);
        assert_eq!(parallel_map(&[] as &[u64], 4, |&i| i), Vec::<u64>::new());
    }

    #[test]
    fn build_all_gives_one_result_per_variant_in_order_and_errors_as_build_does() {
        let config = Config {
            kernel: KernelKind::Matmul,
            n: 8,
            variants: Vec::new(),
            samples: 1,
            min_sample_ms: 1,
            seed: 1,
            quick: true,
            log: None,
        };
        let reference = Reference::new(&config).unwrap();
        let parse = |text: &str| (text.to_string(), Schedule::parse(text).unwrap());
        let texts = [
            "",
            "vectorize(1, nope)",
            "vectorize(1, j)",
            "split(1, j, jo, ji, 4)",
        ];
        let variants: Vec<_> = texts.iter().map(|text| parse(text)).collect();
        let built = reference.build_all(&variants);
        assert_eq!(built.len(), 4);
        for ((text, schedule), result) in variants.iter().zip(&built) {
            match (result, reference.build(text, schedule)) {
                (Ok(got), Ok(want)) => {
                    assert_eq!(got.text, *text);
                    assert_eq!(got.text, want.text);
                    assert!(got.storage.is_empty(), "the buffers are made when timing");
                }
                (Err(got), Err(want)) => {
                    assert_eq!(*got, want);
                    assert!(got.to_string().contains("cannot apply schedule"), "{got}");
                }
                (got, want) => panic!("{text:?}: {:?} against {:?}", got.is_ok(), want.is_ok()),
            }
        }
        assert_eq!(
            built.iter().map(Result::is_ok).collect::<Vec<_>>(),
            [true, false, true, true]
        );
    }

    #[test]
    fn a_schedule_that_does_not_apply_is_an_error() {
        let kernel = build_kernel(KernelKind::Matmul, 4).unwrap();
        let inputs = Inputs::new(4, 1);
        let schedule = Schedule::parse("vectorize(1, nope)").unwrap();
        match make_variant(&kernel, "vectorize(1, nope)", &schedule, &inputs) {
            Err(BenchError::Setup(msg)) => assert!(msg.contains("cannot apply schedule"), "{msg}"),
            other => panic!("expected a setup error, got {:?}", other.err()),
        }
    }
}
