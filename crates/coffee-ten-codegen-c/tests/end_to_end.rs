//! Graph -> loop IR -> C -> clang -> result, against the reference interpreter.

use std::path::PathBuf;

use coffee_ten_codegen_c::compile;
use coffee_ten_interp::{Tensor, run_f32};
use coffee_ten_ir::{BinaryOp, Graph, NodeId, Shape, UnaryOp};
use coffee_ten_loop::lower;

/// A fresh cache dir, removed when dropped (also if the test panics).
struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "coffee-ten-end-to-end-{}-{name}",
            std::process::id()
        ));
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

fn shape(dims: &[usize]) -> Shape {
    Shape::new(dims.to_vec())
}

fn tensor(dims: &[usize], data: &[f32]) -> Tensor<f32> {
    Tensor::new(shape(dims), data.to_vec()).unwrap()
}

/// `A (4, 3)` with values that do not add exactly in f32, some negative.
fn a_data() -> Vec<f32> {
    (0..12)
        .map(|n| (n as f32 - 5.0) * 0.1 + 1.0 / 3.0)
        .collect()
}

/// `B (3, 2)`, some negative.
fn b_data() -> Vec<f32> {
    (0..6).map(|n| (n as f32 - 2.0) * 0.7 - 0.25).collect()
}

/// Large enough that relu clips some rows of `A @ B + bias` and keeps others.
const BIAS: [f32; 2] = [-0.9, 0.3];

/// Lowers `output`, runs the C code with zeroed scratch and NaN-filled `out`, and compares
/// every element with `run_f32`, bit for bit.
fn check_against_reference(name: &str, graph: &Graph, output: NodeId, with_bias: bool) -> Vec<f32> {
    let a = tensor(&[4, 3], &a_data());
    let b = tensor(&[3, 2], &b_data());
    let bias = tensor(&[2], &BIAS);
    let mut inputs = vec![("A", &a), ("B", &b)];
    if with_bias {
        inputs.push(("bias", &bias));
    }
    let expected = run_f32(graph, output, &inputs).unwrap();

    let kernel = lower(graph, output, "k").unwrap();
    let dir = TempDir::new(name);
    let compiled = compile(&kernel, &dir.0).unwrap();
    // Inputs in the kernel's buffer order, then scratch, then out.
    let mut storage: Vec<Vec<f32>> = kernel
        .buffers
        .iter()
        .map(|buffer| {
            let len = buffer.shape.iter().product();
            match buffer.name.as_str() {
                "A" => a_data(),
                "B" => b_data(),
                "bias" => BIAS.to_vec(),
                // Scratch and out start as NaN, so a missing init nest shows up.
                _ => vec![f32::NAN; len],
            }
        })
        .collect();
    let mut buffers: Vec<&mut [f32]> = storage.iter_mut().map(Vec::as_mut_slice).collect();
    compiled.run(&mut buffers).unwrap();

    let got = storage.last().expect("out is the last buffer");
    assert_eq!(got.len(), expected.data().len());
    for (n, (got, want)) in got.iter().zip(expected.data()).enumerate() {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "{name}[{n}]: got {got:?}, expected {want:?}"
        );
    }
    expected.data().to_vec()
}

#[test]
fn matmul_matches_the_interpreter_bit_for_bit() {
    let mut g = Graph::new();
    let a = g.input("A", shape(&[4, 3])).unwrap();
    let b = g.input("B", shape(&[3, 2])).unwrap();
    let prod = g.matmul(a, b).unwrap();
    let result = check_against_reference("matmul", &g, prod, false);
    assert!(result.iter().any(|&x| x < 0.0), "{result:?}");
}

#[test]
fn relu_of_matmul_plus_bias_matches_the_interpreter_bit_for_bit() {
    let mut g = Graph::new();
    let a = g.input("A", shape(&[4, 3])).unwrap();
    let b = g.input("B", shape(&[3, 2])).unwrap();
    let bias = g.input("bias", shape(&[2])).unwrap();
    let prod = g.matmul(a, b).unwrap();
    let sum = g.binary(BinaryOp::Add, prod, bias).unwrap();
    let out = g.unary(UnaryOp::Relu, sum).unwrap();
    let result = check_against_reference("matmul_bias_relu", &g, out, true);
    // relu clips some elements and keeps others.
    assert!(result.contains(&0.0), "{result:?}");
    assert!(result.iter().any(|&x| x > 0.0), "{result:?}");
}
