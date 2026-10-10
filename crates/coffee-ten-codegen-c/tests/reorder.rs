//! Reordered matmul schedules, compiled and run: order-preserving ones bit for bit, the
//! reordering one within the bound of spec §5.4.

use coffee_ten_codegen_c::compile;
use coffee_ten_loop::{
    Access, AffineExpr, BinaryOp, Buffer, Expr, Kernel, Loop, LoopNest, OrderClass, Schedule,
    Store, apply_classified,
};
use std::path::PathBuf;

const M: usize = 4;
const K: usize = 6;
const N: usize = 3;
/// Random inputs per test.
const TRIALS: u64 = 200;

fn lp(name: &str, extent: usize) -> Loop {
    Loop {
        name: name.to_string(),
        extent,
        vectorize: false,
    }
}

fn buf(name: &str, shape: &[usize]) -> Buffer {
    Buffer {
        name: name.to_string(),
        shape: shape.to_vec(),
    }
}

fn var(name: &str) -> AffineExpr {
    AffineExpr {
        constant: 0,
        terms: vec![(name.to_string(), 1)],
    }
}

fn load(buffer: usize, index: Vec<AffineExpr>) -> Expr {
    Expr::Load(Access { buffer, index })
}

fn bin(op: BinaryOp, a: Expr, b: Expr) -> Expr {
    Expr::Binary(op, Box::new(a), Box::new(b))
}

fn nest(loops: Vec<Loop>, index: Vec<AffineExpr>, value: Expr) -> LoopNest {
    LoopNest {
        loops,
        guards: Vec::new(),
        body: Store {
            target: Access { buffer: 2, index },
            value,
        },
    }
}

/// `C = 0; C[i, j] += A[i, k] * B[k, j]` with the update nest ordered i, k, j.
fn matmul() -> Kernel {
    Kernel {
        name: "matmul".to_string(),
        buffers: vec![buf("A", &[M, K]), buf("B", &[K, N]), buf("C", &[M, N])],
        nests: vec![
            nest(
                vec![lp("i", M), lp("j", N)],
                vec![var("i"), var("j")],
                Expr::Const(0.0),
            ),
            nest(
                vec![lp("i", M), lp("k", K), lp("j", N)],
                vec![var("i"), var("j")],
                bin(
                    BinaryOp::Add,
                    load(2, vec![var("i"), var("j")]),
                    bin(
                        BinaryOp::Mul,
                        load(0, vec![var("i"), var("k")]),
                        load(1, vec![var("k"), var("j")]),
                    ),
                ),
            ),
        ],
    }
}

/// A fresh cache dir, removed when dropped (also if the test panics).
struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> Self {
        let dir =
            std::env::temp_dir().join(format!("coffee-ten-reorder-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create cache dir");
        Self(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// xorshift64 with a fixed seed per trial, so every run sees the same inputs.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform in [-1, 1) with a 24-bit mantissa, times 2^e for e in -3..=3, so that
    /// products and sums of several values round and their order matters.
    fn value(&mut self) -> f32 {
        let bits = self.next_u64();
        let unit = (bits >> 40) as f32 / (1u32 << 23) as f32 - 1.0;
        let exponent = (bits & 7) as i32 % 7 - 3;
        unit * 2f32.powi(exponent)
    }
}

fn inputs(trial: u64) -> (Vec<f32>, Vec<f32>) {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ (trial + 1).wrapping_mul(0xBF58_476D_1CE4_E5B9));
    let a = (0..M * K).map(|_| rng.value()).collect();
    let b = (0..K * N).map(|_| rng.value()).collect();
    (a, b)
}

/// Compiles the schedule applied to `matmul()` and returns its class and C for each trial.
/// `name` must differ between tests, as they run in parallel and each removes its cache dir.
fn run_schedule(text: &str, name: &str) -> (OrderClass, Vec<Vec<f32>>) {
    let schedule = Schedule::parse(text).expect("test schedule parses");
    let (kernel, class) = apply_classified(&matmul(), &schedule).expect("test schedule applies");
    let dir = TempDir::new(name);
    let compiled = compile(&kernel, &dir.0).expect("kernel compiles");
    let outputs = (0..TRIALS)
        .map(|trial| {
            let (mut a, mut b) = inputs(trial);
            // Garbage in the output: the init nest must overwrite it.
            let mut c = vec![f32::NAN; M * N];
            compiled.run(&mut [&mut a, &mut b, &mut c]).unwrap();
            c
        })
        .collect();
    (class, outputs)
}

#[test]
fn reorder_ijk_matches_the_unreordered_kernel_bit_for_bit() {
    let (plain_class, plain) = run_schedule("", "bits-plain");
    let (class, reordered) = run_schedule("reorder(1, i, j, k)", "bits-ijk");
    assert_eq!(plain_class, OrderClass::Preserving);
    assert_eq!(class, OrderClass::Preserving);
    for (trial, (got, expected)) in reordered.iter().zip(&plain).enumerate() {
        for (n, (g, e)) in got.iter().zip(expected).enumerate() {
            assert_eq!(
                g.to_bits(),
                e.to_bits(),
                "trial {trial}, C[{n}]: reordered {g:?}, unreordered {e:?}"
            );
        }
    }
}

#[test]
fn swapped_reduction_loops_stay_within_the_error_bound() {
    let (class, plain) = run_schedule("", "bound-plain");
    assert_eq!(class, OrderClass::Preserving);
    let (class, swapped) = run_schedule(
        "split(1, k, ko, ki, 2); reorder(1, i, ki, ko, j)",
        "bound-swap",
    );
    assert_eq!(class, OrderClass::Reordering);

    let u = 2f64.powi(-24);
    let n = K as f64;
    let gamma = n * u / (1.0 - n * u);
    // The smallest positive f32 subnormal, 2^-149.
    let subnormal = f64::from(f32::from_bits(1));
    let mut any_bit_differs = false;
    for (trial, got) in swapped.iter().enumerate() {
        let (a, b) = inputs(trial as u64);
        for i in 0..M {
            for j in 0..N {
                // f32 products are exact in f64 and 6 sums round at 2^-53, which is
                // negligible next to the bound; this is the reference.
                let terms = (0..K).map(|k| f64::from(a[i * K + k]) * f64::from(b[k * N + j]));
                let exact: f64 = terms.clone().sum();
                let abs_sum: f64 = terms.map(f64::abs).sum();
                let bound = 2.0 * (gamma * abs_sum + n * subnormal);
                let g = f64::from(got[i * N + j]);
                let error = (g - exact).abs();
                assert!(
                    error <= bound,
                    "trial {trial}, C[{i}][{j}]: got {g:e}, exact {exact:e}, error {error:e} > bound {bound:e}"
                );
                any_bit_differs |= got[i * N + j].to_bits() != plain[trial][i * N + j].to_bits();
            }
        }
    }
    // The inputs must be able to tell the two orders apart, or the test checks nothing new.
    assert!(
        any_bit_differs,
        "the reordered kernel never differed from the plain one"
    );
}
