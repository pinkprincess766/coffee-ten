//! The `unroll` step: a register-tiled matmul equals the interpreter bit for bit, and clang
//! keeps the tile of `C` in registers across the `k` loop only when the tile loops are
//! unrolled (spec §3.5: the pragma is a request, so the tests ask clang: its vectorizer
//! remarks and the optimized IR).

use coffee_ten_codegen_c::{BENCH_FLAGS, compile, emit_c};
use coffee_ten_interp::{Tensor, run_f32};
use coffee_ten_ir::{Graph, NodeId, Shape};
use coffee_ten_loop::{Kernel, OrderClass, Schedule, apply_classified, lower};
use std::path::PathBuf;
use std::process::Command;

const N: usize = 64;

/// A 4x16 register tile of `C` over `k`, `ii` and `ji` unrolled.
const TILE_UNROLLED: &str = "split(1, i, io, ii, 4); split(1, j, jo, ji, 16); \
                             reorder(1, io, jo, k, ii, ji); vectorize(1, ji); \
                             unroll(1, ii); unroll(1, ji)";

/// A fresh directory, removed when dropped (also if the test panics).
struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> Self {
        let dir =
            std::env::temp_dir().join(format!("coffee-ten-unroll-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create dir");
        Self(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// `A @ B` for two `N x N` inputs.
fn matmul_graph() -> (Graph, NodeId) {
    let mut g = Graph::new();
    let a = g.input("A", Shape::new(vec![N, N])).unwrap();
    let b = g.input("B", Shape::new(vec![N, N])).unwrap();
    let prod = g.matmul(a, b).unwrap();
    (g, prod)
}

/// The kernel of `A @ B` with the schedule `text` applied, and the class of the schedule.
fn scheduled(text: &str) -> (Kernel, OrderClass) {
    let (g, prod) = matmul_graph();
    let kernel = lower(&g, prod, "k").expect("graph lowers");
    let schedule = Schedule::parse(text).expect("schedule parses");
    apply_classified(&kernel, &schedule).expect("schedule applies")
}

/// xorshift64 with a fixed seed, so every run sees the same inputs.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform in [-1, 1) with a 24-bit mantissa, times 2^e for e in -3..=3, so that
    /// products and sums of many values round and their order matters.
    fn value(&mut self) -> f32 {
        let bits = self.next_u64();
        let unit = (bits >> 40) as f32 / (1u32 << 23) as f32 - 1.0;
        let exponent = (bits & 7) as i32 % 7 - 3;
        unit * 2f32.powi(exponent)
    }
}

#[test]
fn register_tiled_matmul_equals_the_interpreter_bit_for_bit() {
    let (kernel, class) = scheduled(TILE_UNROLLED);
    assert_eq!(class, OrderClass::Preserving);
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut a: Vec<f32> = (0..N * N).map(|_| rng.value()).collect();
    let mut b: Vec<f32> = (0..N * N).map(|_| rng.value()).collect();

    let (g, prod) = matmul_graph();
    let shape = Shape::new(vec![N, N]);
    let a_tensor = Tensor::new(shape.clone(), a.clone()).unwrap();
    let b_tensor = Tensor::new(shape, b.clone()).unwrap();
    let expected = run_f32(&g, prod, &[("A", &a_tensor), ("B", &b_tensor)]).unwrap();

    let dir = TempDir::new("bits");
    let compiled = compile(&kernel, &dir.0).expect("kernel compiles");
    // Garbage in the output: the init nest must overwrite it.
    let mut c = vec![f32::NAN; N * N];
    compiled.run(&mut [&mut a, &mut b, &mut c]).unwrap();

    // The result has real rounding: not all elements are whole numbers.
    assert!(c.iter().any(|x| x.fract() != 0.0));
    for (n, (got, want)) in c.iter().zip(expected.data()).enumerate() {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "C[{n}]: got {got:?}, expected {want:?}"
        );
    }
}

/// Runs clang on the C of `kernel` with `flags` and `rpass`; returns the source and the
/// stderr remark lines that contain `needle`, each with the source line it points at.
fn remarks(
    name: &str,
    kernel: &Kernel,
    flags: &[&str],
    rpass: &str,
    needle: &str,
) -> (String, Vec<(usize, String)>) {
    let source = emit_c(kernel).unwrap();
    let dir = TempDir::new(name);
    let c_path = dir.0.join("k.c");
    let o_path = dir.0.join("k.o");
    std::fs::write(&c_path, &source).expect("write C source");
    let output = Command::new("clang")
        .args(flags)
        .args([rpass, "-c", "-o"])
        .arg(&o_path)
        .arg(&c_path)
        .output()
        .expect("clang must be on PATH");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "clang failed:\n{stderr}");
    let remarks = stderr
        .lines()
        .filter(|line| line.contains("remark:") && line.contains(needle))
        .map(|line| {
            // `<path>:<line>:<column>: remark: ...`
            let location = line
                .split(": remark:")
                .next()
                .expect("split has a first part");
            let mut fields = location.rsplit(':');
            let _column = fields.next();
            let number = fields.next().expect("remark has a line number");
            (number.parse().expect("line number"), line.to_string())
        })
        .collect();
    (source, remarks)
}

/// The optimized LLVM IR of the C of `kernel` under `BENCH_FLAGS`, with the source and the
/// number of float-valued phis (`phi float` or `phi <N x float>`) in it.
///
/// A float phi in this kernel is an accumulator carried in a register from one iteration to
/// the next. It is counted in the IR, not taken from clang's LICM remark: clang 18, 19 and
/// 20 promote the tile without printing "Moving accesses to memory location out of the
/// loop" (checked by hand with clang 18.1.6, 19.1.7, 20.1.2, Apple clang 21, LLVM 22).
fn float_phis(name: &str, kernel: &Kernel) -> (String, String, usize) {
    let source = emit_c(kernel).unwrap();
    let dir = TempDir::new(name);
    let c_path = dir.0.join("k.c");
    let ir_path = dir.0.join("k.ll");
    std::fs::write(&c_path, &source).expect("write C source");
    let output = Command::new("clang")
        .args(BENCH_FLAGS)
        .args(["-S", "-emit-llvm", "-o"])
        .arg(&ir_path)
        .arg(&c_path)
        .output()
        .expect("clang must be on PATH");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "clang failed:\n{stderr}");
    let ir = std::fs::read_to_string(&ir_path).expect("read the IR");
    let count = ir
        .lines()
        .filter(|line| {
            line.contains("= phi float ")
                || (line.contains("= phi <") && line.contains(" x float> "))
        })
        .count();
    (source, ir, count)
}

/// The 1-based line number of the only line of `source` that contains `needle`.
fn line_of(source: &str, needle: &str) -> usize {
    let found: Vec<usize> = source
        .lines()
        .enumerate()
        .filter(|(_, l)| l.contains(needle))
        .map(|(i, _)| i + 1)
        .collect();
    assert_eq!(found.len(), 1, "{needle:?} in\n{source}");
    found[0]
}

#[test]
fn clang_keeps_the_c_tile_in_registers_across_k_when_the_tile_is_unrolled() {
    // One accumulator per row of the 4x16 tile: `ji` is a single vector of width 16, and
    // `ii` is unrolled into 4 rows. The phis carry them across the `k` loop.
    let (kernel, _) = scheduled(TILE_UNROLLED);
    let (source, ir, count) = float_phis("phis-unrolled", &kernel);
    assert_eq!(count, 4, "{source}\n{ir}");
}

/// AArch64 only: on x86-64 clang tiles this nest by itself. There it vectorizes `ji` at
/// width 4, interleaves 4 vectors and unrolls `ii`, which gives 16 `<4 x float>`
/// accumulators without any unroll step (ubuntu CI, clang 18.1.3). On AArch64 the answer
/// depends on the hint on `ji`: with the plain `vectorize(enable)` clang unrolled the 16
/// iterations before the vectorizer and kept C in memory (0 phis, clang 18-22; see
/// `clang_fully_unrolls_a_loop_of_16_iterations_with_the_plain_hint`). `emit_c` now prints
/// `vectorize_width(16) unroll(disable)` for it, `ji` is one vector iteration, and the 4 rows
/// of the tile are held across `k` as with the unroll steps.
#[cfg(target_arch = "aarch64")]
#[test]
fn clang_keeps_the_c_tile_in_registers_without_the_unroll_steps_on_aarch64() {
    // `TILE_UNROLLED` without the two unroll requests.
    const TILE: &str = "split(1, i, io, ii, 4); split(1, j, jo, ji, 16); \
                        reorder(1, io, jo, k, ii, ji); vectorize(1, ji)";
    let (kernel, _) = scheduled(TILE);
    let (source, ir, count) = float_phis("phis-rolled", &kernel);
    assert_eq!(count, 4, "{source}\n{ir}");
}

#[test]
fn clang_vectorizes_the_unrolled_tile_loop() {
    // Promotion alone is not enough: with `-fno-slp-vectorize` (`BENCH_FLAGS`) clang can
    // unroll the 16-iteration `ji` loop completely before the loop vectorizer runs, which
    // leaves 64 scalar FMAs (measured by hand at N = 256: 19 GFLOP/s, against 121 for the
    // vectorized tile).
    let (kernel, _) = scheduled(TILE_UNROLLED);
    let (source, remarks) = remarks(
        "vectorize-unrolled",
        &kernel,
        BENCH_FLAGS,
        "-Rpass=loop-vectorize",
        "vectorized loop",
    );
    // One remark per copy of `ji`: `ii` is unrolled, so there are 4 copies of the loop.
    let tile_loop = line_of(&source, "for (long ji");
    assert!(!remarks.is_empty(), "no vectorized loop:\n{source}");
    for (line, text) in &remarks {
        assert_eq!(*line, tile_loop, "{text}\n{source}");
    }
    println!("clang: {}", remarks[0].1);
}
