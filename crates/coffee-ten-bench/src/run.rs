//! Builds the kernel and its variants, checks them against the naive one, times them.

use crate::BenchError;
use crate::cli::{Config, KernelKind};
use crate::meta::Metadata;
use crate::stats::{Rng, Stats, drifted, summarize};
use coffee_ten_codegen_c::{BENCH_FLAGS, CompiledKernel, compile_with_flags};
use coffee_ten_ir::{BinaryOp, Graph, Shape, UnaryOp};
use coffee_ten_loop::{Kernel, OrderClass, Schedule, apply_classified, lower};
use std::path::PathBuf;
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

/// A variant ready to time: compiled, with its own buffers.
struct Variant {
    text: String,
    schedule: Schedule,
    order_class: OrderClass,
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
    Ok(Variant {
        text: text.to_string(),
        schedule: schedule.clone(),
        order_class,
        compiled,
        storage: inputs.storage(&kernel),
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

    let mut reference = make_variant(&naive_kernel, "", &Schedule::default(), &inputs)?;
    reference.run_once()?;

    let mut variants = Vec::new();
    for (text, schedule) in &config.variants {
        let mut variant = make_variant(&naive_kernel, text, schedule, &inputs)?;
        variant.run_once()?;
        check_outputs(text, variant.output(), reference.output(), &scales, n)?;
        variants.push(variant);
    }

    let flop_per_call = 2.0 * (n as f64).powi(3);
    let min_sample = Duration::from_millis(config.min_sample_ms);
    let mut rng = Rng::new(config.seed);
    let mut order: Vec<usize> = (0..variants.len()).collect();
    let mut rates: Vec<Vec<f64>> = vec![Vec::with_capacity(config.samples); variants.len()];
    // Round 0 is the warm-up and is discarded.
    for round in 0..=config.samples {
        rng.shuffle(&mut order);
        for &index in &order {
            let rate = time_sample(&mut variants[index], flop_per_call, min_sample)?;
            if round > 0 {
                rates[index].push(rate);
            }
        }
    }

    variants
        .iter()
        .zip(&rates)
        .map(|(variant, rates)| {
            make_record(
                &info,
                &variant.text,
                &variant.schedule,
                variant.order_class,
                rates,
            )
            .ok_or_else(|| BenchError::Setup("no samples".to_string()))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

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
