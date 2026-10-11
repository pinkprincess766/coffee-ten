//! Builds the kernel and its variants, checks them against the naive one, times them.

use crate::BenchError;
use crate::cli::{Config, KernelKind};
use crate::stats::{Rng, Stats, drifted, summarize};
use coffee_ten_codegen_c::{BENCH_FLAGS, CompiledKernel, compile_with_flags};
use coffee_ten_ir::{BinaryOp, Graph, Shape, UnaryOp};
use coffee_ten_loop::{Kernel, Schedule, apply, lower};
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
    let kernel = apply(naive, schedule).map_err(|e| setup("cannot apply schedule", &e))?;
    let compiled = compile_with_flags(&kernel, &cache_dir(), BENCH_FLAGS)
        .map_err(|e| setup("cannot compile", &e))?;
    Ok(Variant {
        text: text.to_string(),
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

/// What is reported for one variant.
#[derive(Clone, Debug, PartialEq)]
pub struct Record {
    pub variant: String,
    pub stats: Stats,
    /// Number of samples.
    pub n: usize,
    /// Median of the first rounds differs from the median of the last by more than `DRIFT_LIMIT`.
    pub drift: bool,
    /// The run was `--quick`: not a measurement.
    pub quick: bool,
}

/// The one place where a record per variant is assembled; a JSONL line is written from it.
/// `rates` in round order, GFLOP/s. `None` if there are no samples.
pub fn make_record(variant: &str, rates: &[f64], quick: bool) -> Option<Record> {
    Some(Record {
        variant: variant.to_string(),
        stats: summarize(rates)?,
        n: rates.len(),
        drift: drifted(rates),
        quick,
    })
}

/// Builds the naive kernel and every variant, checks each variant's output against the naive
/// one, then times them. One record per variant, in `config.variants` order.
pub fn run(config: &Config) -> Result<Vec<Record>, BenchError> {
    let n = config.n;
    let naive_kernel = build_kernel(config.kernel, n)?;
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
            make_record(&variant.text, rates, config.quick)
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
        let rates: Vec<f64> = (1..=10).map(f64::from).collect();
        let record = make_record("v", &rates, true).unwrap();
        assert_eq!(record.n, 10);
        assert_eq!(record.stats.median, 5.0);
        assert!(record.quick);
        // First 5 rounds have median 3, the last 5 have median 8.
        assert!(record.drift);
        assert_eq!(make_record("v", &[], false), None);
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
