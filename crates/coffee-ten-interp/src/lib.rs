//! Reference interpreter for coffee-ten graphs.

use std::fmt;

use coffee_ten_ir::{BinaryOp, Graph, NodeId, Op, Shape, UnaryOp};

/// Row-major tensor; data.len() is the product of the dims (1 for a scalar).
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor<T> {
    shape: Shape,
    data: Vec<T>,
}

impl<T> Tensor<T> {
    pub fn new(shape: Shape, data: Vec<T>) -> Result<Tensor<T>, InterpError> {
        let expected = numel(&shape)?;
        if data.len() != expected {
            return Err(InterpError::DataLen {
                expected,
                got: data.len(),
            });
        }
        Ok(Tensor { shape, data })
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn data(&self) -> &[T] {
        &self.data
    }
}

/// Evaluates `output` in f32; every op is one f32 operation, no FMA, no reassociation.
pub fn run_f32(
    graph: &Graph,
    output: NodeId,
    inputs: &[(&str, &Tensor<f32>)],
) -> Result<Tensor<f32>, InterpError> {
    evaluate(graph, output, inputs)
}

/// Same graph in f64: inputs and constants are widened exactly, every op is one f64 operation.
pub fn run_f64(
    graph: &Graph,
    output: NodeId,
    inputs: &[(&str, &Tensor<f32>)],
) -> Result<Tensor<f64>, InterpError> {
    evaluate(graph, output, inputs)
}

#[derive(Clone, Debug, PartialEq)]
pub enum InterpError {
    UnknownNode(NodeId),
    /// An Input node reachable from `output` has no entry in `inputs`.
    MissingInput(String),
    /// An entry in `inputs` names no Input node of the graph.
    UnknownInput(String),
    /// The same name appears twice in `inputs`.
    DuplicateInput(String),
    InputShape {
        name: String,
        expected: Shape,
        got: Shape,
    },
    DataLen {
        expected: usize,
        got: usize,
    },
    /// The element count of this shape does not fit in `usize`.
    TooLarge(Shape),
}

impl fmt::Display for InterpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InterpError::UnknownNode(id) => write!(f, "unknown node %{}", id.index()),
            InterpError::MissingInput(name) => write!(f, "missing input {name}"),
            InterpError::UnknownInput(name) => write!(f, "unknown input {name}"),
            InterpError::DuplicateInput(name) => write!(f, "duplicate input {name}"),
            InterpError::InputShape {
                name,
                expected,
                got,
            } => write!(f, "input {name} has shape {got}, expected {expected}"),
            InterpError::DataLen { expected, got } => {
                write!(f, "data has {got} elements, expected {expected}")
            }
            InterpError::TooLarge(shape) => write!(f, "shape {shape} has too many elements"),
        }
    }
}

impl std::error::Error for InterpError {}

/// Element type of the evaluator. Each method is one operation in `Self`.
trait Scalar: Copy {
    fn from_f32(x: f32) -> Self;
    fn zero() -> Self;
    fn add(self, other: Self) -> Self;
    fn sub(self, other: Self) -> Self;
    fn mul(self, other: Self) -> Self;
    fn div(self, other: Self) -> Self;
    fn neg(self) -> Self;
    /// One call of the platform libm `exp`; not exactly rounded.
    fn exp(self) -> Self;
    fn gt(self, other: Self) -> bool;
}

impl Scalar for f32 {
    fn from_f32(x: f32) -> f32 {
        x
    }
    fn zero() -> f32 {
        0.0
    }
    fn add(self, other: f32) -> f32 {
        self + other
    }
    fn sub(self, other: f32) -> f32 {
        self - other
    }
    fn mul(self, other: f32) -> f32 {
        self * other
    }
    fn div(self, other: f32) -> f32 {
        self / other
    }
    fn neg(self) -> f32 {
        -self
    }
    fn exp(self) -> f32 {
        f32::exp(self)
    }
    fn gt(self, other: f32) -> bool {
        self > other
    }
}

impl Scalar for f64 {
    fn from_f32(x: f32) -> f64 {
        f64::from(x)
    }
    fn zero() -> f64 {
        0.0
    }
    fn add(self, other: f64) -> f64 {
        self + other
    }
    fn sub(self, other: f64) -> f64 {
        self - other
    }
    fn mul(self, other: f64) -> f64 {
        self * other
    }
    fn div(self, other: f64) -> f64 {
        self / other
    }
    fn neg(self) -> f64 {
        -self
    }
    fn exp(self) -> f64 {
        f64::exp(self)
    }
    fn gt(self, other: f64) -> bool {
        self > other
    }
}

fn unary<T: Scalar>(op: UnaryOp, x: T) -> T {
    match op {
        UnaryOp::Neg => x.neg(),
        UnaryOp::Relu => {
            if x.gt(T::zero()) {
                x
            } else {
                T::zero()
            }
        }
        UnaryOp::Exp => x.exp(),
    }
}

/// `max` returns `b` unless `a > b`, so it is not `f32::max` on signed zeros or NaN.
fn binary<T: Scalar>(op: BinaryOp, a: T, b: T) -> T {
    match op {
        BinaryOp::Add => a.add(b),
        BinaryOp::Sub => a.sub(b),
        BinaryOp::Mul => a.mul(b),
        BinaryOp::Div => a.div(b),
        BinaryOp::Max => {
            if a.gt(b) {
                a
            } else {
                b
            }
        }
    }
}

fn evaluate<T: Scalar>(
    graph: &Graph,
    output: NodeId,
    inputs: &[(&str, &Tensor<f32>)],
) -> Result<Tensor<T>, InterpError> {
    graph.get(output).ok_or(InterpError::UnknownNode(output))?;
    let supplied = supplied_inputs(graph, inputs)?;
    let needed = needed_nodes(graph, output);
    let mut values: Vec<Option<Tensor<T>>> = (0..graph.nodes().len()).map(|_| None).collect();
    for (index, node) in graph.nodes().iter().enumerate().take(output.index() + 1) {
        if !needed[index] {
            continue;
        }
        let value = match &node.op {
            Op::Input { name } => {
                let input =
                    supplied[index].ok_or_else(|| InterpError::MissingInput(name.clone()))?;
                let data = input.data.iter().map(|&x| T::from_f32(x)).collect();
                Tensor {
                    shape: node.shape.clone(),
                    data,
                }
            }
            Op::Const(value) => Tensor {
                shape: node.shape.clone(),
                data: vec![T::from_f32(*value)],
            },
            Op::Unary(op, x) => {
                let x = operand(&values, *x);
                let data = x.data.iter().map(|&v| unary(*op, v)).collect();
                Tensor {
                    shape: node.shape.clone(),
                    data,
                }
            }
            Op::Binary(op, a, b) => {
                let data =
                    broadcast_binary(*op, operand(&values, *a), operand(&values, *b), &node.shape)?;
                Tensor {
                    shape: node.shape.clone(),
                    data,
                }
            }
            Op::MatMul(a, b) => Tensor {
                shape: node.shape.clone(),
                data: matmul(operand(&values, *a), operand(&values, *b)),
            },
        };
        values[index] = Some(value);
    }
    Ok(values
        .swap_remove(output.index())
        .expect("invariant: the output node is evaluated"))
}

/// Input tensors by node index, after checking names, duplicates and shapes.
fn supplied_inputs<'a>(
    graph: &Graph,
    inputs: &[(&str, &'a Tensor<f32>)],
) -> Result<Vec<Option<&'a Tensor<f32>>>, InterpError> {
    let mut supplied = vec![None; graph.nodes().len()];
    for &(name, tensor) in inputs {
        let index = graph
            .nodes()
            .iter()
            .position(|node| matches!(&node.op, Op::Input { name: input } if input == name))
            .ok_or_else(|| InterpError::UnknownInput(name.to_string()))?;
        if supplied[index].is_some() {
            return Err(InterpError::DuplicateInput(name.to_string()));
        }
        let expected = &graph.nodes()[index].shape;
        if tensor.shape() != expected {
            return Err(InterpError::InputShape {
                name: name.to_string(),
                expected: expected.clone(),
                got: tensor.shape().clone(),
            });
        }
        supplied[index] = Some(tensor);
    }
    Ok(supplied)
}

/// Nodes that `output` depends on, by node index. Operands precede their users.
fn needed_nodes(graph: &Graph, output: NodeId) -> Vec<bool> {
    let mut needed = vec![false; graph.nodes().len()];
    needed[output.index()] = true;
    for index in (0..=output.index()).rev() {
        if !needed[index] {
            continue;
        }
        match &graph.nodes()[index].op {
            Op::Input { .. } | Op::Const(_) => {}
            Op::Unary(_, x) => needed[x.index()] = true,
            Op::Binary(_, a, b) | Op::MatMul(a, b) => {
                needed[a.index()] = true;
                needed[b.index()] = true;
            }
        }
    }
    needed
}

fn operand<T>(values: &[Option<Tensor<T>>], id: NodeId) -> &Tensor<T> {
    values[id.index()]
        .as_ref()
        .expect("invariant: operands are evaluated before their users")
}

/// Element count of `shape`: 0 with a zero dim, `TooLarge` if it overflows `usize`.
fn numel(shape: &Shape) -> Result<usize, InterpError> {
    if shape.dims().contains(&0) {
        return Ok(0);
    }
    shape
        .dims()
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| InterpError::TooLarge(shape.clone()))
}

/// Row-major strides of `operand` inside a result of shape `out`.
/// Axes missing on the left and axes of size 1 get stride 0.
fn broadcast_strides(operand: &Shape, out: &Shape) -> Vec<usize> {
    let offset = out
        .rank()
        .checked_sub(operand.rank())
        .expect("invariant: a broadcast result has at least the operands' rank");
    let mut strides = vec![0; out.rank()];
    let mut stride = 1;
    for (d, &dim) in operand.dims().iter().enumerate().rev() {
        if dim != 1 {
            strides[d + offset] = stride;
        }
        stride *= dim;
    }
    strides
}

/// `a (M, K) @ b (K, N)` as a flat row-major `(M, N)` buffer. For each `(i, j)`, `acc = 0`,
/// then for `k` ascending `acc = acc + a[i,k] * b[k,j]`. The product and the sum are two
/// roundings, never fused, and this is the order of the C kernel (init nest writes 0,
/// update nest adds `A*B` with `k` ascending), so f32 results agree bit for bit.
fn matmul<T: Scalar>(a: &Tensor<T>, b: &Tensor<T>) -> Vec<T> {
    let (m, k) = (a.shape().dims()[0], a.shape().dims()[1]);
    let n = b.shape().dims()[1];
    let mut data = Vec::with_capacity(m * n);
    for i in 0..m {
        for j in 0..n {
            let mut acc = T::zero();
            for kk in 0..k {
                let product = a.data[i * k + kk].mul(b.data[kk * n + j]);
                acc = acc.add(product);
            }
            data.push(acc);
        }
    }
    data
}

fn broadcast_binary<T: Scalar>(
    op: BinaryOp,
    a: &Tensor<T>,
    b: &Tensor<T>,
    out: &Shape,
) -> Result<Vec<T>, InterpError> {
    let len = numel(out)?;
    let a_strides = broadcast_strides(a.shape(), out);
    let b_strides = broadcast_strides(b.shape(), out);
    let dims = out.dims();
    let mut index = vec![0; dims.len()];
    let (mut a_at, mut b_at) = (0, 0);
    let mut data = Vec::with_capacity(len);
    for _ in 0..len {
        data.push(binary(op, a.data[a_at], b.data[b_at]));
        // Advance the multi-index row-major, moving both operand offsets with it.
        for axis in (0..dims.len()).rev() {
            index[axis] += 1;
            a_at += a_strides[axis];
            b_at += b_strides[axis];
            if index[axis] < dims[axis] {
                break;
            }
            a_at -= a_strides[axis] * dims[axis];
            b_at -= b_strides[axis] * dims[axis];
            index[axis] = 0;
        }
    }
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(dims: &[usize]) -> Shape {
        Shape::new(dims.to_vec())
    }

    fn tensor(dims: &[usize], data: &[f32]) -> Tensor<f32> {
        Tensor::new(shape(dims), data.to_vec()).unwrap()
    }

    /// Runs both modes and checks the values; the hand values are exact in f32 and f64.
    fn check_both(
        graph: &Graph,
        output: NodeId,
        inputs: &[(&str, &Tensor<f32>)],
        dims: &[usize],
        expected: &[f32],
    ) {
        assert_eq!(
            run_f32(graph, output, inputs).unwrap(),
            tensor(dims, expected)
        );
        let out64 = run_f64(graph, output, inputs).unwrap();
        let expected64: Vec<f64> = expected.iter().map(|&x| f64::from(x)).collect();
        assert_eq!(out64.shape(), &shape(dims));
        assert_eq!(out64.data(), expected64.as_slice());
    }

    /// `A (2,3) + bias (3,)`.
    fn bias_graph() -> (Graph, NodeId) {
        let mut g = Graph::new();
        let a = g.input("A", shape(&[2, 3])).unwrap();
        let bias = g.input("bias", shape(&[3])).unwrap();
        let sum = g.binary(BinaryOp::Add, a, bias).unwrap();
        (g, sum)
    }

    /// `max(a, b)` of two `(1,)` inputs, returned from f32 and from f64.
    fn max_both(a: f32, b: f32) -> (f32, f64) {
        let mut g = Graph::new();
        let a_id = g.input("a", shape(&[1])).unwrap();
        let b_id = g.input("b", shape(&[1])).unwrap();
        let m = g.binary(BinaryOp::Max, a_id, b_id).unwrap();
        let a_t = tensor(&[1], &[a]);
        let b_t = tensor(&[1], &[b]);
        let inputs = [("a", &a_t), ("b", &b_t)];
        let out32 = run_f32(&g, m, &inputs).unwrap();
        let out64 = run_f64(&g, m, &inputs).unwrap();
        (out32.data()[0], out64.data()[0])
    }

    #[test]
    fn bias_broadcasts_over_rows() {
        let (g, sum) = bias_graph();
        let a_t = tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let bias_t = tensor(&[3], &[10.0, 20.0, 30.0]);
        let inputs = [("A", &a_t), ("bias", &bias_t)];
        // [[1,2,3],[4,5,6]] + [10,20,30] = [[11,22,33],[14,25,36]]
        check_both(
            &g,
            sum,
            &inputs,
            &[2, 3],
            &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0],
        );
    }

    #[test]
    fn column_times_row_broadcasts_both_ways() {
        let mut g = Graph::new();
        let a = g.input("a", shape(&[2, 1])).unwrap();
        let b = g.input("b", shape(&[3])).unwrap();
        let prod = g.binary(BinaryOp::Mul, a, b).unwrap();
        let a_t = tensor(&[2, 1], &[2.0, 3.0]);
        let b_t = tensor(&[3], &[1.0, 10.0, 100.0]);
        let inputs = [("a", &a_t), ("b", &b_t)];
        // [[2],[3]] * [1,10,100] = [[2,20,200],[3,30,300]]
        check_both(
            &g,
            prod,
            &inputs,
            &[2, 3],
            &[2.0, 20.0, 200.0, 3.0, 30.0, 300.0],
        );
    }

    #[test]
    fn scalar_constant_multiplies_every_element() {
        let mut g = Graph::new();
        let two = g.constant(2.0);
        let a = g.input("A", shape(&[3])).unwrap();
        let prod = g.binary(BinaryOp::Mul, two, a).unwrap();
        let a_t = tensor(&[3], &[1.0, -2.0, 0.5]);
        // 2 * [1,-2,0.5] = [2,-4,1]
        check_both(&g, prod, &[("A", &a_t)], &[3], &[2.0, -4.0, 1.0]);
    }

    #[test]
    fn div_is_elementwise_in_order() {
        let mut g = Graph::new();
        let a = g.input("a", shape(&[3])).unwrap();
        let b = g.input("b", shape(&[3])).unwrap();
        let q = g.binary(BinaryOp::Div, a, b).unwrap();
        let a_t = tensor(&[3], &[1.0, 3.0, -7.0]);
        let b_t = tensor(&[3], &[2.0, 4.0, 0.0]);
        // [1,3,-7] / [2,4,0] = [0.5,0.75,-inf]
        check_both(
            &g,
            q,
            &[("a", &a_t), ("b", &b_t)],
            &[3],
            &[0.5, 0.75, f32::NEG_INFINITY],
        );
    }

    #[test]
    fn exp_of_zero_is_exactly_one() {
        let mut g = Graph::new();
        let x = g.input("x", shape(&[1])).unwrap();
        let y = g.unary(UnaryOp::Exp, x).unwrap();
        let x_t = tensor(&[1], &[0.0]);
        // exp(0) = 1
        check_both(&g, y, &[("x", &x_t)], &[1], &[1.0]);
    }

    /// Steps between two positive finite f32 values.
    fn ulp_distance(a: f32, b: f32) -> u32 {
        assert!(a > 0.0 && b > 0.0 && a.is_finite() && b.is_finite());
        a.to_bits().abs_diff(b.to_bits())
    }

    #[test]
    fn exp_f32_is_within_one_ulp_of_rounded_f64() {
        let mut g = Graph::new();
        let x = g.input("x", shape(&[1000])).unwrap();
        let y = g.unary(UnaryOp::Exp, x).unwrap();
        // 1000 evenly spaced points from -80 to 80; exp stays normal in f32
        // (exp(-80) ~ 1.8e-35 > 1.2e-38, exp(80) ~ 5.5e34 < 3.4e38).
        let xs: Vec<f32> = (0..1000)
            .map(|i| -80.0 + 160.0 * (i as f32) / 999.0)
            .collect();
        let x_t = tensor(&[1000], &xs);
        let inputs = [("x", &x_t)];
        let out32 = run_f32(&g, y, &inputs).unwrap();
        let out64 = run_f64(&g, y, &inputs).unwrap();
        for ((&x, &got), &reference) in xs.iter().zip(out32.data()).zip(out64.data()) {
            let ulps = ulp_distance(got, reference as f32);
            assert!(
                ulps <= 1,
                "exp({x}) = {got}, f64 gives {reference}: {ulps} ulp"
            );
        }
    }

    #[test]
    fn neg_flips_every_sign() {
        let mut g = Graph::new();
        let a = g.input("A", shape(&[3])).unwrap();
        let neg = g.unary(UnaryOp::Neg, a).unwrap();
        let a_t = tensor(&[3], &[1.0, -2.0, 0.5]);
        // -[1,-2,0.5] = [-1,2,-0.5]
        check_both(&g, neg, &[("A", &a_t)], &[3], &[-1.0, 2.0, -0.5]);
    }

    #[test]
    fn sub_is_elementwise_in_order() {
        let mut g = Graph::new();
        let a = g.input("a", shape(&[2])).unwrap();
        let b = g.input("b", shape(&[2])).unwrap();
        let diff = g.binary(BinaryOp::Sub, a, b).unwrap();
        let a_t = tensor(&[2], &[5.0, 1.0]);
        let b_t = tensor(&[2], &[2.0, 3.0]);
        // [5,1] - [2,3] = [3,-2]
        check_both(&g, diff, &[("a", &a_t), ("b", &b_t)], &[2], &[3.0, -2.0]);
    }

    #[test]
    fn relu_of_negative_zero_is_positive_zero() {
        let mut g = Graph::new();
        let x = g.input("x", shape(&[1])).unwrap();
        let y = g.unary(UnaryOp::Relu, x).unwrap();
        let x_t = tensor(&[1], &[-0.0]);
        let inputs = [("x", &x_t)];
        let out32 = run_f32(&g, y, &inputs).unwrap();
        let out64 = run_f64(&g, y, &inputs).unwrap();
        assert_eq!(out32.data()[0].to_bits(), 0.0f32.to_bits());
        assert_eq!(out64.data()[0].to_bits(), 0.0f64.to_bits());
    }

    #[test]
    fn max_of_equal_zeros_returns_the_second_operand() {
        let (r32, r64) = max_both(0.0, -0.0);
        assert_eq!(r32.to_bits(), (-0.0f32).to_bits());
        assert_eq!(r64.to_bits(), (-0.0f64).to_bits());
        let (r32, r64) = max_both(-0.0, 0.0);
        assert_eq!(r32.to_bits(), 0.0f32.to_bits());
        assert_eq!(r64.to_bits(), 0.0f64.to_bits());
    }

    #[test]
    fn max_with_nan_first_returns_second_operand() {
        let (r32, r64) = max_both(f32::NAN, 1.0);
        assert_eq!(r32.to_bits(), 1.0f32.to_bits());
        assert_eq!(r64.to_bits(), 1.0f64.to_bits());
    }

    #[test]
    fn max_with_nan_second_returns_nan() {
        let (r32, r64) = max_both(1.0, f32::NAN);
        assert!(r32.is_nan());
        assert!(r64.is_nan());
    }

    #[test]
    fn add_and_mul_commute_bit_for_bit() {
        let a_t = tensor(&[2, 3], &[0.1, 1.0 / 3.0, -0.0, 1e30, 1e-40, -7.25]);
        let b_t = tensor(&[3], &[0.2, -1.0 / 3.0, 3e-39]);
        for op in [BinaryOp::Add, BinaryOp::Mul] {
            let mut g = Graph::new();
            let a = g.input("A", shape(&[2, 3])).unwrap();
            let b = g.input("b", shape(&[3])).unwrap();
            let ab = g.binary(op, a, b).unwrap();
            let ba = g.binary(op, b, a).unwrap();
            let inputs = [("A", &a_t), ("b", &b_t)];
            let bits32 = |out: NodeId| -> Vec<u32> {
                run_f32(&g, out, &inputs)
                    .unwrap()
                    .data()
                    .iter()
                    .map(|x| x.to_bits())
                    .collect()
            };
            assert_eq!(bits32(ab), bits32(ba));
            let bits64 = |out: NodeId| -> Vec<u64> {
                run_f64(&g, out, &inputs)
                    .unwrap()
                    .data()
                    .iter()
                    .map(|x| x.to_bits())
                    .collect()
            };
            assert_eq!(bits64(ab), bits64(ba));
        }
    }

    #[test]
    fn f64_mode_adds_without_rounding_to_f32() {
        let mut g = Graph::new();
        let x = g.input("x", shape(&[1])).unwrap();
        let y = g.input("y", shape(&[1])).unwrap();
        let sum = g.binary(BinaryOp::Add, x, y).unwrap();
        let x_t = tensor(&[1], &[0.1]);
        let y_t = tensor(&[1], &[0.2]);
        let inputs = [("x", &x_t), ("y", &y_t)];
        let out64 = run_f64(&g, sum, &inputs).unwrap();
        let out32 = run_f32(&g, sum, &inputs).unwrap();
        assert_eq!(out64.data()[0], f64::from(0.1f32) + f64::from(0.2f32));
        assert_ne!(out64.data()[0], f64::from(out32.data()[0]));
    }

    #[test]
    fn missing_input_is_named() {
        let (g, sum) = bias_graph();
        let a_t = tensor(&[2, 3], &[0.0; 6]);
        assert_eq!(
            run_f32(&g, sum, &[("A", &a_t)]),
            Err(InterpError::MissingInput("bias".to_string()))
        );
    }

    #[test]
    fn unknown_input_name_is_reported() {
        let (g, sum) = bias_graph();
        let a_t = tensor(&[2, 3], &[0.0; 6]);
        let bias_t = tensor(&[3], &[0.0; 3]);
        let z_t = tensor(&[1], &[0.0]);
        let inputs = [("A", &a_t), ("bias", &bias_t), ("Z", &z_t)];
        assert_eq!(
            run_f32(&g, sum, &inputs),
            Err(InterpError::UnknownInput("Z".to_string()))
        );
    }

    #[test]
    fn duplicate_input_name_is_reported() {
        let (g, sum) = bias_graph();
        let a_t = tensor(&[2, 3], &[0.0; 6]);
        let bias_t = tensor(&[3], &[0.0; 3]);
        let inputs = [("A", &a_t), ("A", &a_t), ("bias", &bias_t)];
        assert_eq!(
            run_f32(&g, sum, &inputs),
            Err(InterpError::DuplicateInput("A".to_string()))
        );
    }

    #[test]
    fn wrong_input_shape_is_reported() {
        let (g, sum) = bias_graph();
        let a_t = tensor(&[3], &[0.0; 3]);
        let bias_t = tensor(&[3], &[0.0; 3]);
        let inputs = [("A", &a_t), ("bias", &bias_t)];
        assert_eq!(
            run_f32(&g, sum, &inputs),
            Err(InterpError::InputShape {
                name: "A".to_string(),
                expected: shape(&[2, 3]),
                got: shape(&[3]),
            })
        );
    }

    #[test]
    fn tensor_new_checks_the_data_length() {
        assert_eq!(
            Tensor::new(shape(&[2, 3]), vec![0.0f32; 5]),
            Err(InterpError::DataLen {
                expected: 6,
                got: 5
            })
        );
        assert!(Tensor::new(shape(&[]), vec![1.0f32]).is_ok());
    }

    #[test]
    fn element_count_handles_zero_dims_and_overflow() {
        // A zero dim makes the count 0 even when the other dims overflow.
        assert!(Tensor::new(shape(&[1 << 40, 1 << 40, 0]), Vec::<f32>::new()).is_ok());
        // An overflowing count has its own error.
        assert_eq!(
            Tensor::new(shape(&[usize::MAX, 2]), vec![0.0f32]),
            Err(InterpError::TooLarge(shape(&[usize::MAX, 2])))
        );
    }

    #[test]
    fn output_from_a_larger_graph_is_unknown() {
        let mut big = Graph::new();
        let foreign = (0..6).map(|_| big.constant(0.0)).last().unwrap();
        assert_eq!(foreign.index(), 5);

        let mut g = Graph::new();
        g.constant(1.0);
        g.constant(2.0);
        assert_eq!(
            run_f32(&g, foreign, &[]),
            Err(InterpError::UnknownNode(foreign))
        );
        assert_eq!(
            run_f64(&g, foreign, &[]),
            Err(InterpError::UnknownNode(foreign))
        );
    }

    #[test]
    fn unused_input_may_be_omitted() {
        let mut g = Graph::new();
        let a = g.input("A", shape(&[2])).unwrap();
        g.input("Z", shape(&[5])).unwrap();
        let neg = g.unary(UnaryOp::Neg, a).unwrap();
        let a_t = tensor(&[2], &[1.0, -2.0]);
        // -[1,-2] = [-1,2]
        check_both(&g, neg, &[("A", &a_t)], &[2], &[-1.0, 2.0]);
    }

    #[test]
    fn matmul_hand_values_in_both_modes() {
        let mut g = Graph::new();
        let a = g.input("A", shape(&[2, 3])).unwrap();
        let b = g.input("B", shape(&[3, 2])).unwrap();
        let prod = g.matmul(a, b).unwrap();
        let a_t = tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b_t = tensor(&[3, 2], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        // [[1,2,3],[4,5,6]] @ [[7,8],[9,10],[11,12]] = [[58,64],[139,154]]
        check_both(
            &g,
            prod,
            &[("A", &a_t), ("B", &b_t)],
            &[2, 2],
            &[58.0, 64.0, 139.0, 154.0],
        );
    }

    #[test]
    fn matmul_then_bias_then_relu() {
        let mut g = Graph::new();
        let a = g.input("A", shape(&[2, 3])).unwrap();
        let b = g.input("B", shape(&[3, 2])).unwrap();
        let bias = g.input("bias", shape(&[2])).unwrap();
        let prod = g.matmul(a, b).unwrap();
        let sum = g.binary(BinaryOp::Add, prod, bias).unwrap();
        let out = g.unary(UnaryOp::Relu, sum).unwrap();
        let a_t = tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b_t = tensor(&[3, 2], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let bias_t = tensor(&[2], &[-100.0, -150.0]);
        // [[58,64],[139,154]] + [-100,-150] = [[-42,-86],[39,4]], relu = [[0,0],[39,4]]
        check_both(
            &g,
            out,
            &[("A", &a_t), ("B", &b_t), ("bias", &bias_t)],
            &[2, 2],
            &[0.0, 0.0, 39.0, 4.0],
        );
    }

    #[test]
    fn matmul_f32_matches_canonical_order_bit_for_bit() {
        let mut g = Graph::new();
        let a = g.input("A", shape(&[4, 3])).unwrap();
        let b = g.input("B", shape(&[3, 2])).unwrap();
        let prod = g.matmul(a, b).unwrap();
        // A[i][k] at flat index i*3+k, B[k][j] at flat index k*2+j.
        let a_data: Vec<f32> = (0..12usize).map(|x| x as f32 * 0.1 + 1.0 / 3.0).collect();
        let b_data: Vec<f32> = (0..6usize).map(|x| x as f32 * 0.7 - 0.25).collect();
        let a_t = tensor(&[4, 3], &a_data);
        let b_t = tensor(&[3, 2], &b_data);
        let out = run_f32(&g, prod, &[("A", &a_t), ("B", &b_t)]).unwrap();
        for i in 0..4 {
            for j in 0..2 {
                let mut acc = 0.0f32;
                for k in 0..3 {
                    acc += a_data[i * 3 + k] * b_data[k * 2 + j];
                }
                assert_eq!(
                    out.data()[i * 2 + j].to_bits(),
                    acc.to_bits(),
                    "element ({i}, {j})"
                );
            }
        }
    }

    #[test]
    fn errors_print_one_short_line() {
        let mut g = Graph::new();
        let id = g.constant(1.0);
        let cases = [
            (InterpError::UnknownNode(id), "unknown node %0"),
            (
                InterpError::MissingInput("A".to_string()),
                "missing input A",
            ),
            (
                InterpError::UnknownInput("Z".to_string()),
                "unknown input Z",
            ),
            (
                InterpError::DuplicateInput("A".to_string()),
                "duplicate input A",
            ),
            (
                InterpError::InputShape {
                    name: "A".to_string(),
                    expected: shape(&[2, 3]),
                    got: shape(&[3]),
                },
                "input A has shape (3,), expected (2, 3)",
            ),
            (
                InterpError::DataLen {
                    expected: 6,
                    got: 5,
                },
                "data has 5 elements, expected 6",
            ),
        ];
        for (err, text) in cases {
            assert_eq!(err.to_string(), text);
        }
    }
}
