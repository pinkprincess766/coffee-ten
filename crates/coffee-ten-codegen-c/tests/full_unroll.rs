//! The trip count below which clang fully unrolls a `vectorize(enable)` loop before its loop
//! vectorizer sees it, so the pragma gets no remark and the code is scalar. The analytic
//! model (`crates/coffee-ten-bench/src/model.rs`, `CLANG_FULL_UNROLL_LIMIT`) predicts such a
//! loop as scalar; these tests pin that threshold under `BENCH_FLAGS`. If a test here fails
//! after a clang update, change `CLANG_FULL_UNROLL_LIMIT` in the model together with it.
//!
//! The same file pins the other claim the model's register rule rests on: clang holds a row
//! of `C` in vector registers across a rolled `ki` loop.

use coffee_ten_codegen_c::{BENCH_FLAGS, emit_c};
use coffee_ten_ir::{Graph, Shape};
use coffee_ten_loop::{Kernel, Schedule, apply, lower};
use std::path::PathBuf;
use std::process::Command;

/// A fresh directory, removed when dropped (also if the test panics).
struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "coffee-ten-full-unroll-{}-{name}",
            std::process::id()
        ));
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

/// The kernel of the `n x n x n` matmul with the schedule `text` applied.
fn scheduled(n: usize, text: &str) -> Kernel {
    let mut g = Graph::new();
    let a = g.input("A", Shape::new(vec![n, n])).unwrap();
    let b = g.input("B", Shape::new(vec![n, n])).unwrap();
    let prod = g.matmul(a, b).unwrap();
    let kernel = lower(&g, prod, "k").expect("graph lowers");
    apply(&kernel, &Schedule::parse(text).expect("schedule parses")).expect("schedule applies")
}

/// Runs clang on the C of `kernel` with `BENCH_FLAGS` and `extra`; returns the C source and
/// stderr.
fn clang(name: &str, kernel: &Kernel, extra: &[&str], out_name: &str) -> (String, String, String) {
    let source = emit_c(kernel).unwrap();
    let dir = TempDir::new(name);
    let c_path = dir.0.join("k.c");
    let out_path = dir.0.join(out_name);
    std::fs::write(&c_path, &source).expect("write C source");
    let output = Command::new("clang")
        .args(BENCH_FLAGS)
        .args(extra)
        .arg("-o")
        .arg(&out_path)
        .arg(&c_path)
        .output()
        .expect("clang must be on PATH");
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    assert!(output.status.success(), "clang failed:\n{stderr}");
    let produced = std::fs::read_to_string(&out_path).unwrap_or_default();
    (source, stderr, produced)
}

/// The "vectorized loop" remarks of clang for the update nest of the `n x n x n` matmul
/// `split(1, j, jo, ji, w); reorder(1, i, jo, k, ji); vectorize(1, ji)`: `ji` has `w`
/// iterations.
fn remarks_for_ji(n: usize, w: usize) -> Vec<String> {
    let text = format!("split(1, j, jo, ji, {w}); reorder(1, i, jo, k, ji); vectorize(1, ji)");
    let (source, stderr, _) = clang(
        &format!("remark-{n}-{w}"),
        &scheduled(n, &text),
        &["-Rpass=loop-vectorize", "-S", "-emit-llvm"],
        "k.ll",
    );
    assert!(source.contains("vectorize(enable)"), "{source}");
    stderr
        .lines()
        .filter(|line| line.contains("remark: vectorized loop"))
        .map(str::to_string)
        .collect()
}

/// `CLANG_FULL_UNROLL_LIMIT` = 16 of the model, AArch64 only: a marked loop of 16 iterations
/// gets no vectorized-loop remark (clang unrolled it fully first). Apple clang 21. The model
/// describes the M5; on x86-64 clang 18 (ubuntu CI) the same loop IS vectorized, width 4,
/// checked with `zig cc -target x86_64-linux-gnu`, so the threshold is a property of the target.
#[cfg(target_arch = "aarch64")]
#[test]
fn clang_fully_unrolls_a_marked_loop_of_16_iterations() {
    assert_eq!(remarks_for_ji(64, 16), Vec::<String>::new());
}

/// Above the threshold the pragma vectorizes on every target: a marked loop of 17 and one of
/// 32 iterations get a vectorized-loop remark.
#[test]
fn clang_vectorizes_a_marked_loop_of_17_or_32_iterations() {
    // 17 does not divide 64: N = 34 = 2 * 17, so the split has no guard.
    let (seventeen, thirty_two) = (remarks_for_ji(34, 17), remarks_for_ji(64, 32));
    assert!(!seventeen.is_empty(), "{seventeen:?}");
    assert!(!thirty_two.is_empty(), "{thirty_two:?}");
}

/// The optimized IR of the 128 x 64 x 128 tile with `ii, ki, ji` innermost: the largest
/// number of consecutive `<4 x float>` phis of one basic block. A phi is an accumulator
/// carried in a register from one iteration to the next. The `ji` loop of 64 floats is
/// 16 vectors, so the row of `C` is held across `ki` if a block has 16. (Clang emits several
/// copies of the `ki` loop; the largest block is what matters, not the total.)
#[test]
fn clang_holds_a_row_of_c_across_a_rolled_ki_loop() {
    let text = "split(1, i, io, ii, 128); split(1, j, jo, ji, 64); split(1, k, ko, ki, 128); \
                reorder(1, io, ko, jo, ii, ki, ji); vectorize(1, ji)";
    let (source, _, ir) = clang(
        "c-row",
        &scheduled(256, text),
        &["-S", "-emit-llvm"],
        "k.ll",
    );
    let is_vector_phi = |line: &str| line.contains("= phi <4 x float> ");
    let (mut run, mut longest) = (0_usize, 0_usize);
    for line in ir.lines() {
        run = if is_vector_phi(line) { run + 1 } else { 0 };
        longest = longest.max(run);
    }
    assert_eq!(longest, 16, "{source}\n{ir}");
}
