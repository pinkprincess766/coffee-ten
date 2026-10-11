//! The command line.

use crate::BenchError;
use coffee_ten_loop::Schedule;

pub const DEFAULT_SAMPLES: usize = 30;
pub const DEFAULT_MIN_SAMPLE_MS: u64 = 10;
pub const DEFAULT_SEED: u64 = 1;
/// `--quick` replaces `--samples` and `--min-sample-ms` with these.
pub const QUICK_SAMPLES: usize = 3;
pub const QUICK_MIN_SAMPLE_MS: u64 = 1;
/// Largest `--n`: the error bound costs about `n^3` f64 operations before timing starts.
pub const MAX_N: usize = 1024;

pub const USAGE: &str = "usage: coffee-ten-bench --kernel matmul|relu-matmul-bias --n <N> \
--variant <schedule> [--variant <schedule> ...] [--samples <S>] [--min-sample-ms <MS>] \
[--seed <SEED>] [--quick]\n\
a variant is a schedule text; --variant \"\" is the naive kernel";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelKind {
    /// `A @ B`, both `n x n`.
    Matmul,
    /// `relu(A @ B + bias)`, `bias` of length `n`.
    ReluMatmulBias,
}

impl KernelKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Matmul => "matmul",
            Self::ReluMatmulBias => "relu-matmul-bias",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub kernel: KernelKind,
    pub n: usize,
    /// Schedule text as typed, and parsed; "" is the naive kernel.
    pub variants: Vec<(String, Schedule)>,
    pub samples: usize,
    pub min_sample_ms: u64,
    pub seed: u64,
    pub quick: bool,
}

/// Parses the arguments after the program name.
pub fn parse_args<I: IntoIterator<Item = String>>(args: I) -> Result<Config, BenchError> {
    let mut args = args.into_iter();
    let mut kernel = None;
    let mut n = None;
    let mut variants = Vec::new();
    let mut samples = DEFAULT_SAMPLES;
    let mut min_sample_ms = DEFAULT_MIN_SAMPLE_MS;
    let mut seed = DEFAULT_SEED;
    let mut quick = false;
    while let Some(flag) = args.next() {
        if flag == "--quick" {
            quick = true;
            continue;
        }
        let value = args
            .next()
            .ok_or_else(|| usage(format!("{flag} needs a value")))?;
        match flag.as_str() {
            "--kernel" => kernel = Some(parse_kernel(&value)?),
            "--n" => n = Some(parse_number::<usize>(&flag, &value)?),
            "--variant" => {
                let schedule = Schedule::parse(&value).map_err(|e| {
                    usage(format!(
                        "--variant {value:?}: {} (at byte {})",
                        e.msg, e.pos
                    ))
                })?;
                variants.push((value, schedule));
            }
            "--samples" => samples = parse_number(&flag, &value)?,
            "--min-sample-ms" => min_sample_ms = parse_number(&flag, &value)?,
            "--seed" => seed = parse_number(&flag, &value)?,
            _ => return Err(usage(format!("unknown argument {flag}"))),
        }
    }
    let kernel = kernel.ok_or_else(|| usage("--kernel is required".to_string()))?;
    let n = n.ok_or_else(|| usage("--n is required".to_string()))?;
    if !(1..=MAX_N).contains(&n) {
        return Err(usage(format!("--n must be in 1..={MAX_N}, got {n}")));
    }
    if variants.is_empty() {
        return Err(usage("at least one --variant is required".to_string()));
    }
    if quick {
        samples = QUICK_SAMPLES;
        min_sample_ms = QUICK_MIN_SAMPLE_MS;
    }
    if samples == 0 {
        return Err(usage("--samples must be at least 1".to_string()));
    }
    Ok(Config {
        kernel,
        n,
        variants,
        samples,
        min_sample_ms,
        seed,
        quick,
    })
}

fn usage(msg: String) -> BenchError {
    BenchError::Usage(format!("{msg}\n{USAGE}"))
}

fn parse_kernel(text: &str) -> Result<KernelKind, BenchError> {
    match text {
        "matmul" => Ok(KernelKind::Matmul),
        "relu-matmul-bias" => Ok(KernelKind::ReluMatmulBias),
        _ => Err(usage(format!(
            "unknown kernel {text:?}, expected matmul or relu-matmul-bias"
        ))),
    }
}

fn parse_number<T: std::str::FromStr>(flag: &str, text: &str) -> Result<T, BenchError> {
    text.parse()
        .map_err(|_| usage(format!("{flag} needs a non-negative integer, got {text:?}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Config, BenchError> {
        parse_args(args.iter().map(|a| a.to_string()))
    }

    #[test]
    fn repeated_variants_keep_their_order() {
        let config = parse(&[
            "--kernel",
            "matmul",
            "--n",
            "64",
            "--variant",
            "",
            "--variant",
            "split(1, j, jo, ji, 8)",
            "--variant",
            "vectorize(1, j)",
            "--seed",
            "9",
        ])
        .unwrap();
        let texts: Vec<&str> = config.variants.iter().map(|(t, _)| t.as_str()).collect();
        assert_eq!(texts, ["", "split(1, j, jo, ji, 8)", "vectorize(1, j)"]);
        assert!(config.variants[0].1.steps.is_empty());
        assert_eq!(config.variants[1].1.steps.len(), 1);
        assert_eq!(config.kernel, KernelKind::Matmul);
        assert_eq!(config.n, 64);
        assert_eq!(config.seed, 9);
        assert_eq!(config.samples, DEFAULT_SAMPLES);
        assert_eq!(config.min_sample_ms, DEFAULT_MIN_SAMPLE_MS);
        assert!(!config.quick);
    }

    #[test]
    fn schedule_syntax_error_is_an_error_not_a_panic() {
        let args = ["--kernel", "matmul", "--n", "8", "--variant"];
        for bad in ["split(1, j", "frobnicate(1, j)", "vectorize(1, j);;"] {
            let mut all = args.to_vec();
            all.push(bad);
            match parse(&all) {
                Err(BenchError::Usage(msg)) => assert!(msg.contains(bad), "{msg}"),
                other => panic!("{bad:?}: expected a usage error, got {other:?}"),
            }
        }
    }

    #[test]
    fn quick_replaces_samples_and_min_sample() {
        let config = parse(&[
            "--kernel",
            "relu-matmul-bias",
            "--n",
            "8",
            "--variant",
            "",
            "--samples",
            "50",
            "--quick",
        ])
        .unwrap();
        assert!(config.quick);
        assert_eq!(config.samples, QUICK_SAMPLES);
        assert_eq!(config.min_sample_ms, QUICK_MIN_SAMPLE_MS);
        assert_eq!(config.kernel, KernelKind::ReluMatmulBias);
    }

    #[test]
    fn n_limits_are_checked_on_both_sides() {
        let with_n = |n: &str| parse(&["--kernel", "matmul", "--n", n, "--variant", ""]);
        assert!(with_n("0").is_err());
        assert!(with_n("1").is_ok());
        assert!(with_n(&MAX_N.to_string()).is_ok());
        assert!(with_n(&(MAX_N + 1).to_string()).is_err());
    }

    #[test]
    fn samples_must_be_positive() {
        let with_samples = |s: &str| {
            parse(&[
                "--kernel",
                "matmul",
                "--n",
                "8",
                "--variant",
                "",
                "--samples",
                s,
            ])
        };
        assert!(with_samples("0").is_err());
        assert_eq!(with_samples("1").unwrap().samples, 1);
    }

    #[test]
    fn missing_and_unknown_arguments_are_errors() {
        assert!(parse(&[]).is_err());
        assert!(parse(&["--kernel", "matmul", "--n", "8"]).is_err());
        assert!(parse(&["--n", "8", "--variant", ""]).is_err());
        assert!(parse(&["--kernel", "conv", "--n", "8", "--variant", ""]).is_err());
        assert!(parse(&["--kernel", "matmul", "--n", "x", "--variant", ""]).is_err());
        assert!(
            parse(&[
                "--kernel",
                "matmul",
                "--n",
                "8",
                "--variant",
                "",
                "--bogus",
                "1"
            ])
            .is_err()
        );
        assert!(parse(&["--kernel", "matmul", "--n", "8", "--variant"]).is_err());
    }
}
