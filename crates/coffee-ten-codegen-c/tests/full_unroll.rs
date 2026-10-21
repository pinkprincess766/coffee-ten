//! The trip count below which clang fully unrolls a `vectorize(enable)` loop before its loop
//! vectorizer sees it, so the pragma gets no remark and the code is scalar. The analytic
//! model (`crates/coffee-ten-bench/src/model.rs`, `CLANG_FULL_UNROLL_LIMIT`) predicts such a
//! loop as scalar; these tests pin that threshold under `BENCH_FLAGS`. If a test here fails
//! after a clang update, change `CLANG_FULL_UNROLL_LIMIT` in the model together with it.
//!
//! The same file pins the other claims the model's register rule rests on: clang holds a row
//! of `C` in vector registers across a rolled `ki` loop, and does not across a `ki` loop it
//! unrolled itself (`fully_unrolled` in the model): the copies of the vectorized `ji` loop each
//! load and store the row.

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

/// The longest run of consecutive `<4 x float>` phis in one basic block of the optimized IR
/// `ir`. A phi is an accumulator carried in a register from one iteration to the next.
fn longest_vector_phi_run(ir: &str) -> usize {
    let (mut run, mut longest) = (0_usize, 0_usize);
    for line in ir.lines() {
        run = if line.contains("= phi <4 x float> ") {
            run + 1
        } else {
            0
        };
        longest = longest.max(run);
    }
    longest
}

/// The optimized IR of the 128 x 64 x 128 tile with `ii, ki, ji` innermost: the largest
/// number of consecutive `<4 x float>` phis of one basic block. The `ji` loop of 64 floats is
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
    assert_eq!(longest_vector_phi_run(&ir), 16, "{source}\n{ir}");
}

/// Vectorized-loop remarks (one per copy of the vectorized loop) and the longest vector phi
/// run for `text` at N = 256.
fn copies_and_phis(name: &str, text: &str) -> (usize, usize) {
    let (source, stderr, ir) = clang(
        name,
        &scheduled(256, text),
        &["-Rpass=loop-vectorize", "-S", "-emit-llvm"],
        "k.ll",
    );
    assert!(source.contains("vectorize(enable)"), "{source}");
    let remarks = stderr
        .lines()
        .filter(|line| line.contains("remark: vectorized loop"))
        .count();
    (remarks, longest_vector_phi_run(&ir))
}

/// `split(1, i, io, ii, 16); split(1, j, jo, ji, 64); split(1, k, ko, ki, K); reorder(1, io, jo,
/// ko, ii, ki, ji); vectorize(1, ji)`: `ki` is the loop around the vectorized `ji`.
fn ki_schedule(ki: usize) -> String {
    format!(
        "split(1, i, io, ii, 16); split(1, j, jo, ji, 64); split(1, k, ko, ki, {ki}); \
         reorder(1, io, jo, ko, ii, ki, ji); vectorize(1, ji)"
    )
}

/// The model's `fully_unrolled`: a `ki` of 8 around the vectorized `ji` of 64 is fully
/// unrolled before the vectorizer (8 * (5 + 9) <= 150), so `ji` is 8 separate loops, each
/// vectorized (8 remarks) and each loading and storing the row of `C`: no vector phi. A `ki`
/// of 16 or 32 stays a loop (16 * 14 > 150) and holds the row of `C`, 16 vectors, in phis.
/// Checked also with clang 18.1.6 for x86-64 (`zig cc -target x86_64-linux-gnu`): the same
/// counts, so nothing here is gated to a target.
#[test]
fn clang_unrolls_a_ki_of_8_around_a_vectorized_ji_and_keeps_the_longer_one_rolled() {
    let (copies, phis) = copies_and_phis("ki-8", &ki_schedule(8));
    assert_eq!((copies, phis), (8, 0));
    // With ki of 16 and 32 clang also unrolls `jo` (4 iterations), which copies the nest
    // (4 remarks); the row is held either way.
    for ki in [16, 32] {
        let (_, phis) = copies_and_phis(&format!("ki-{ki}"), &ki_schedule(ki));
        assert_eq!(phis, 16, "ki = {ki}");
    }
}

/// The same for the `ii` loop of the nest without a `k` split, `jo, k, io, ii, ji`, with `ji`
/// of 32: `ii` of 8 is unrolled (8 * (5 + 9) <= 150), 8 loops of `ji` each load the row of `B`
/// that does not depend on `ii`; `ii` of 16 stays a loop (one remark), with the row of `B`
/// loaded before it. x86-64 clang 18.1.6: the same counts.
#[test]
fn clang_unrolls_an_ii_of_8_around_a_vectorized_ji_and_keeps_the_longer_one_rolled() {
    let ii_schedule = |ii: usize| {
        format!(
            "split(1, i, io, ii, {ii}); split(1, j, jo, ji, 32); \
             reorder(1, jo, k, io, ii, ji); vectorize(1, ji)"
        )
    };
    assert_eq!(copies_and_phis("ii-8", &ii_schedule(8)).0, 8);
    assert_eq!(copies_and_phis("ii-16", &ii_schedule(16)).0, 1);
}
