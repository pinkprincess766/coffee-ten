//! The `vectorize` step against clang's own remarks: the pragma is only a request (spec
//! §3.5), so the tests ask clang which loops it vectorized, with `TEST_FLAGS`, which turn
//! automatic vectorization off.

use coffee_ten_codegen_c::{TEST_FLAGS, compile, emit_c};
use coffee_ten_ir::{BinaryOp, Graph, NodeId, Shape, UnaryOp};
use coffee_ten_loop::{Kernel, Schedule, apply, lower};
use std::path::PathBuf;
use std::process::Command;

/// Large enough that clang does not unroll the `j` loop completely before it can vectorize
/// it: with 16 the pragma gets no remark at all.
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
