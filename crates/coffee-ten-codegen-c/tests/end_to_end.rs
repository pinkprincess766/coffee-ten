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

/// `C (2, 2)`, some negative.
fn c_data() -> Vec<f32> {
    (0..4).map(|n| (n as f32 - 1.0) * 0.45 + 0.1).collect()
}

/// Shape and values of the input called `name`.
fn input_data(name: &str) -> (Vec<usize>, Vec<f32>) {
    match name {
        "A" => (vec![4, 3], a_data()),
        "B" => (vec![3, 2], b_data()),
        "C" => (vec![2, 2], c_data()),
        "bias" => (vec![2], BIAS.to_vec()),
        _ => panic!("no data for input {name}"),
    }
}

/// Lowers `output`, runs the C code with NaN-filled scratch and `out`, and compares every
/// element with `run_f32`, bit for bit. `input_names` are the graph's inputs.
fn check_against_reference(
    name: &str,
    graph: &Graph,
    output: NodeId,
    input_names: &[&'static str],
) -> Vec<f32> {
    let tensors: Vec<Tensor<f32>> = input_names
        .iter()
        .map(|&input| {
            let (dims, data) = input_data(input);
            tensor(&dims, &data)
        })
        .collect();
    let inputs: Vec<(&str, &Tensor<f32>)> = input_names.iter().copied().zip(&tensors).collect();
    let expected = run_f32(graph, output, &inputs).unwrap();

    let kernel = lower(graph, output, "k").unwrap();
    let dir = TempDir::new(name);
    let compiled = compile(&kernel, &dir.0).unwrap();
    // Inputs in the kernel's buffer order, then scratch, then out.
    let mut storage: Vec<Vec<f32>> = kernel
        .buffers
        .iter()
        .map(|buffer| {
            if input_names.contains(&buffer.name.as_str()) {
                return input_data(&buffer.name).1;
            }
            // Scratch and out start as NaN, so a missing init nest or a temporary that is
            // read before it is written shows up.
            vec![f32::NAN; buffer.shape.iter().product()]
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
    let result = check_against_reference("matmul", &g, prod, &["A", "B"]);
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
    let result = check_against_reference("matmul_bias_relu", &g, out, &["A", "B", "bias"]);
    // relu clips some elements and keeps others.
    assert!(result.contains(&0.0), "{result:?}");
    assert!(result.iter().any(|&x| x > 0.0), "{result:?}");
}

#[test]
fn matmul_of_a_shifted_operand_matches_the_interpreter_bit_for_bit() {
    let mut g = Graph::new();
    let a = g.input("A", shape(&[4, 3])).unwrap();
    let b = g.input("B", shape(&[3, 2])).unwrap();
    let one = g.constant(1.0);
    let shifted = g.binary(BinaryOp::Add, a, one).unwrap();
    let prod = g.matmul(shifted, b).unwrap();
    let result = check_against_reference("shifted_matmul", &g, prod, &["A", "B"]);
    assert!(result.iter().any(|&x| x < 0.0), "{result:?}");
}

#[test]
fn matmul_of_a_matmul_matches_the_interpreter_bit_for_bit() {
    let mut g = Graph::new();
    let a = g.input("A", shape(&[4, 3])).unwrap();
    let b = g.input("B", shape(&[3, 2])).unwrap();
    let c = g.input("C", shape(&[2, 2])).unwrap();
    let inner = g.matmul(a, b).unwrap();
    let outer = g.matmul(inner, c).unwrap();
    let result = check_against_reference("matmul_chain", &g, outer, &["A", "B", "C"]);
    assert!(result.iter().any(|&x| x < 0.0), "{result:?}");
}

#[test]
fn matmul_of_relu_plus_bias_matches_the_interpreter_bit_for_bit() {
    let mut g = Graph::new();
    let a = g.input("A", shape(&[4, 3])).unwrap();
    let b = g.input("B", shape(&[3, 2])).unwrap();
    let bias = g.input("bias", shape(&[2])).unwrap();
    let activated = g.unary(UnaryOp::Relu, a).unwrap();
    let prod = g.matmul(activated, b).unwrap();
    let out = g.binary(BinaryOp::Add, prod, bias).unwrap();
    let result = check_against_reference("relu_matmul_bias", &g, out, &["A", "B", "bias"]);
    assert!(result.iter().any(|&x| x < 0.0), "{result:?}");
}
