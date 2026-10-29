//! The `vectorize` step against clang's own remarks: the pragma is only a request (spec
//! §3.5), so the tests ask clang which loops it vectorized, with `TEST_FLAGS`, which turn
//! automatic vectorization off.

use coffee_ten_codegen_c::{TEST_FLAGS, compile, emit_c};
use coffee_ten_interp::{Tensor, run_f32};
use coffee_ten_ir::{BinaryOp, Graph, NodeId, Shape, UnaryOp};
use coffee_ten_loop::{Kernel, Schedule, apply, lower};
use std::path::PathBuf;
use std::process::Command;

/// Longer than 16, so the marked `j` loop keeps the plain `vectorize(enable)` hint, which
/// clang can act on: it unrolls a loop of up to 16 completely before it can vectorize it
/// (on AArch64), which is why such a loop gets `vectorize_width` and `unroll(disable)`
/// instead (see the tests at the end).
const N: usize = 32;

/// A fresh directory, removed when dropped (also if the test panics).
struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "coffee-ten-vectorize-{}-{name}",
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

fn shape(dims: &[usize]) -> Shape {
    Shape::new(dims.to_vec())
}

/// `A @ B` for two `N x N` inputs.
fn matmul_graph() -> (Graph, NodeId) {
    let mut g = Graph::new();
    let a = g.input("A", shape(&[N, N])).unwrap();
    let b = g.input("B", shape(&[N, N])).unwrap();
    let prod = g.matmul(a, b).unwrap();
    (g, prod)
}

/// The kernel of `output` with the schedule `text` applied.
fn scheduled(graph: &Graph, output: NodeId, text: &str) -> Kernel {
    let kernel = lower(graph, output, "k").expect("graph lowers");
    apply(&kernel, &Schedule::parse(text).expect("schedule parses")).expect("schedule applies")
}

/// Runs clang on `source` with `TEST_FLAGS` and `-Rpass=loop-vectorize`; returns the
/// stderr lines that report a vectorized loop, and the source line each one points at.
fn vectorized_loops(name: &str, source: &str) -> Vec<(usize, String)> {
    let dir = TempDir::new(name);
    let c_path = dir.0.join("k.c");
    let o_path = dir.0.join("k.o");
    std::fs::write(&c_path, source).expect("write C source");
    let output = Command::new("clang")
        .args(TEST_FLAGS)
        .args(["-Rpass=loop-vectorize", "-c", "-o"])
        .arg(&o_path)
        .arg(&c_path)
        .output()
        .expect("clang must be on PATH");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "clang failed:\n{stderr}");
    stderr
        .lines()
        .filter(|line| line.contains("remark: vectorized loop"))
        .map(|line| {
            // `<path>:<line>:<column>: remark: vectorized loop (...)`
            let location = line
                .split(": remark:")
                .next()
                .expect("split has a first part");
            let mut fields = location.rsplit(':');
            let _column = fields.next();
            let number = fields.next().expect("remark has a line number");
            (number.parse().expect("line number"), line.to_string())
        })
        .collect()
}

/// The 1-based line number of the `n`-th (from 0) line of `source` that contains `needle`.
fn line_of(source: &str, needle: &str, n: usize) -> usize {
    let found = source
        .lines()
        .enumerate()
        .filter(|(_, l)| l.contains(needle));
    found
        .map(|(i, _)| i + 1)
        .nth(n)
        .expect("source has the line")
}

#[test]
fn clang_vectorizes_exactly_the_marked_matmul_loop() {
    let (g, prod) = matmul_graph();
    let source = emit_c(&scheduled(&g, prod, "vectorize(1, j)")).unwrap();
    // Nest 0 (init) has the first `j` loop, nest 1 (update) the second.
    let marked = line_of(&source, "for (long j", 1);
    let remarks = vectorized_loops("matmul", &source);
    let lines: Vec<usize> = remarks.iter().map(|(line, _)| *line).collect();
    assert_eq!(lines, [marked], "{remarks:?}\n{source}");
    println!("clang: {}", remarks[0].1);
}

#[test]
fn clang_vectorizes_nothing_without_the_step() {
    let (g, prod) = matmul_graph();
    let source = emit_c(&scheduled(&g, prod, "")).unwrap();
    assert_eq!(vectorized_loops("plain", &source), []);
}

#[test]
fn clang_vectorizes_exactly_the_marked_elementwise_loop() {
    let (mut g, prod) = matmul_graph();
    let bias = g.input("bias", shape(&[N])).unwrap();
    let sum = g.binary(BinaryOp::Add, prod, bias).unwrap();
    let out = g.unary(UnaryOp::Relu, sum).unwrap();
    // Nests 0 and 1 are the matmul; nest 2 is `out[i0, i1] = relu(t0[i0, i1] + bias[i1])`.
    let source = emit_c(&scheduled(&g, out, "vectorize(2, i1)")).unwrap();
    let marked = line_of(&source, "for (long i1", 0);
    let remarks = vectorized_loops("relu", &source);
    let lines: Vec<usize> = remarks.iter().map(|(line, _)| *line).collect();
    assert_eq!(lines, [marked], "{remarks:?}\n{source}");
}

/// Values whose products and sums round in f32.
fn data(seed: usize) -> Vec<f32> {
    (0..N * N)
        .map(|n| ((n * 37 + seed * 11) % 101) as f32 * 0.173 - 8.7)
        .collect()
}

#[test]
fn vectorized_matmul_equals_the_plain_one_bit_for_bit() {
    let (g, prod) = matmul_graph();
    let dir = TempDir::new("bits");
    let mut outputs = Vec::new();
    for text in ["", "vectorize(1, j)"] {
        let compiled = compile(&scheduled(&g, prod, text), &dir.0).unwrap();
        let (mut a, mut b) = (data(1), data(2));
        // Garbage in the output: the init nest must overwrite it.
        let mut c = vec![f32::NAN; N * N];
        compiled.run(&mut [&mut a, &mut b, &mut c]).unwrap();
        outputs.push(c);
    }
    // The result has real rounding: not all elements are whole numbers.
    assert!(outputs[0].iter().any(|x| x.fract() != 0.0));
    for (n, (vectorized, plain)) in outputs[1].iter().zip(&outputs[0]).enumerate() {
        assert_eq!(
            vectorized.to_bits(),
            plain.to_bits(),
            "C[{n}]: vectorized {vectorized:?}, plain {plain:?}"
        );
    }
}

/// Lines of the vectorized-loop remarks for the `ji` loop of the `N x N` matmul with
/// `split(1, j, jo, ji, w); reorder(1, i, jo, k, ji); vectorize(1, ji)`, and the line of
/// that loop.
fn ji_remarks(w: usize) -> (Vec<usize>, usize) {
    let text = format!("split(1, j, jo, ji, {w}); reorder(1, i, jo, k, ji); vectorize(1, ji)");
    let (g, prod) = matmul_graph();
    let source = emit_c(&scheduled(&g, prod, &text)).unwrap();
    let marked = line_of(&source, "for (long ji", 0);
    let remarks = vectorized_loops(&format!("ji-{w}"), &source);
    (remarks.iter().map(|(line, _)| *line).collect(), marked)
}

/// A marked `ji` of 8 or 16 is one vector iteration, not a scalar loop: clang vectorizes
/// it. With the plain `vectorize(enable)` clang unrolled it completely before the loop
/// vectorizer on AArch64 (no remark), and with the SLP vectorizer off that left scalar code.
/// Every remark of the nest, also one for a copy clang makes of the `jo` loop, points at
/// the `ji` line.
#[test]
fn clang_vectorizes_a_marked_ji_of_8_and_of_16() {
    for w in [8, 16] {
        let (lines, marked) = ji_remarks(w);
        assert!(!lines.is_empty(), "ji = {w}: no remark");
        assert!(
            lines.iter().all(|&line| line == marked),
            "ji = {w}: {lines:?}"
        );
    }
}

fn ramp(len: usize, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|n| ((n * 37 + seed * 11) % 101) as f32 * 0.173 - 8.7)
        .collect()
}

/// `A (3, 5) @ B (5, cols)` compiled with `text` and run on rounding data, against the
/// interpreter bit for bit. Matmul vectorized over `j` keeps the order of the `k` sum of
/// every element, and `TEST_FLAGS` do not contract, so the bits are the interpreter's.
fn assert_matmul_equals_interpreter(cols: usize, text: &str) {
    let mut g = Graph::new();
    let a = g.input("A", shape(&[3, 5])).unwrap();
    let b = g.input("B", shape(&[5, cols])).unwrap();
    let prod = g.matmul(a, b).unwrap();
    let (a_data, b_data) = (ramp(3 * 5, 1), ramp(5 * cols, 2));
    let a_tensor = Tensor::new(shape(&[3, 5]), a_data.clone()).unwrap();
    let b_tensor = Tensor::new(shape(&[5, cols]), b_data.clone()).unwrap();
    let want = run_f32(&g, prod, &[("A", &a_tensor), ("B", &b_tensor)]).unwrap();

    let dir = TempDir::new(&format!("short-{cols}-{}", text.len()));
    let compiled = compile(&scheduled(&g, prod, text), &dir.0).unwrap();
    let (mut a_run, mut b_run) = (a_data, b_data);
    let mut got = vec![f32::NAN; 3 * cols];
    compiled
        .run(&mut [&mut a_run, &mut b_run, &mut got])
        .unwrap();

    assert!(want.data().iter().any(|x| x.fract() != 0.0));
    for (n, (got, want)) in got.iter().zip(want.data()).enumerate() {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "{text:?}, {cols} columns, C[{n}]: got {got:?}, expected {want:?}"
        );
    }
}

/// Every length of the vectorized loop from 1 to 16, power of two or not: the loop is `j`
/// itself (`cols` = length), or the `ji` of a split of 20 columns, which has a guard when
/// the length does not divide 20.
#[test]
fn vectorized_loops_of_1_to_16_iterations_equal_the_interpreter_bit_for_bit() {
    for len in 1..=16 {
        assert_matmul_equals_interpreter(len, "vectorize(1, j)");
        let split =
            format!("split(1, j, jo, ji, {len}); reorder(1, i, jo, k, ji); vectorize(1, ji)");
        assert_matmul_equals_interpreter(20, &split);
    }
}
