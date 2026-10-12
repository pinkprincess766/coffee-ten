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

/// `D (3,)`, a divisor: no power of two, so a quotient with `A` is rounded.
const DIVISOR: [f32; 3] = [0.7, -1.3, 3.1];

/// `X (1000,)`, 1000 evenly spaced points from -10 to 10 (end points included).
fn x_data() -> Vec<f32> {
    (0..1000)
        .map(|n| -10.0 + 20.0 * (n as f32) / 999.0)
        .collect()
}

/// Shape and values of the input called `name`.
fn input_data(name: &str) -> (Vec<usize>, Vec<f32>) {
    match name {
        "A" => (vec![4, 3], a_data()),
        "B" => (vec![3, 2], b_data()),
        "C" => (vec![2, 2], c_data()),
        "bias" => (vec![2], BIAS.to_vec()),
        "D" => (vec![3], DIVISOR.to_vec()),
        "X" => (vec![1000], x_data()),
        _ => panic!("no data for input {name}"),
    }
}

/// Lowers `output`, runs the C code with NaN-filled scratch and `out`, and returns the
/// kernel's result and `run_f32`'s. `input_names` are the graph's inputs.
fn run_kernel_and_reference(
    name: &str,
    graph: &Graph,
    output: NodeId,
    input_names: &[&'static str],
) -> (Vec<f32>, Vec<f32>) {
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

    let got = storage.pop().expect("out is the last buffer");
    assert_eq!(got.len(), expected.data().len());
    (got, expected.data().to_vec())
}

/// Like `run_kernel_and_reference`, and compares every element bit for bit.
fn check_against_reference(
    name: &str,
    graph: &Graph,
    output: NodeId,
    input_names: &[&'static str],
) -> Vec<f32> {
    let (got, want) = run_kernel_and_reference(name, graph, output, input_names);
    for (n, (got, want)) in got.iter().zip(&want).enumerate() {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "{name}[{n}]: got {got:?}, expected {want:?}"
        );
    }
    want
}

/// Number of f32 values between `a` and `b` (0 for equal values, also for -0.0 and 0.0).
fn ulp_distance(a: f32, b: f32) -> u32 {
    assert!(a.is_finite() && b.is_finite(), "{a} {b}");
    // Order-preserving integer key: negative floats count down from zero.
    let key = |x: f32| {
        let magnitude = i64::from(x.to_bits() & 0x7fff_ffff);
        if x.is_sign_negative() {
            -magnitude
        } else {
            magnitude
        }
    };
    u32::try_from((key(a) - key(b)).unsigned_abs()).expect("finite f32 values are < 2^32 apart")
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

#[test]
fn div_matches_the_interpreter_bit_for_bit() {
    let mut g = Graph::new();
    let a = g.input("A", shape(&[4, 3])).unwrap();
    let d = g.input("D", shape(&[3])).unwrap();
    let out = g.binary(BinaryOp::Div, a, d).unwrap();
    let result = check_against_reference("div", &g, out, &["A", "D"]);
    // Division is correctly rounded: the f32 result must differ from the exact quotient
    // somewhere, or the test would not exercise the rounding.
    let a_values = a_data();
    let inexact = result.iter().enumerate().any(|(n, &q)| {
        let exact = f64::from(a_values[n]) / f64::from(DIVISOR[n % 3]);
        f64::from(q) != exact
    });
    assert!(inexact, "{result:?}");
    assert!(result.iter().any(|&x| x < 0.0), "{result:?}");
}

#[test]
fn exp_matches_float64_within_one_ulp() {
    let mut g = Graph::new();
    let x = g.input("X", shape(&[1000])).unwrap();
    let two = g.constant(2.0);
    let e = g.unary(UnaryOp::Exp, x).unwrap();
    let out = g.binary(BinaryOp::Mul, e, two).unwrap();
    let (got, _) = run_kernel_and_reference("exp_times_two", &g, out, &["X"]);
    // exp is not an exact operation, so the compiled kernel is compared with float64, not
    // bit for bit. expf is within 1 ulp of the true value and the rounded reference within
    // 0.5 ulp, so the two f32 values are less than 1.5 ulp apart, i.e. at most 1 ulp.
    // Times 2 is exact, so there is no cancellation to amplify the error.
    let xs = x_data();
    for (n, got) in got.iter().enumerate() {
        let want = (2.0 * f64::from(xs[n]).exp()) as f32;
        let ulps = ulp_distance(*got, want);
        assert!(
            ulps <= 1,
            "exp_times_two[{n}] at x = {}: got {got:?}, float64 gives {want:?}, {ulps} ulp",
            xs[n]
        );
    }
}
