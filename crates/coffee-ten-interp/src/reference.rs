//! Frozen interpreter algorithms from commit 823e279, before optimization.
//! Keep their traversal, indexing and operation selection independent of production helpers.
//! Changes to production algorithms must continue to match these results bit for bit.

use super::*;

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

#[inline(never)]
pub(super) fn matmul<T: Scalar>(a: &Tensor<T>, b: &Tensor<T>) -> Vec<T> {
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

#[inline(never)]
pub(super) fn reduce<T: Scalar>(op: ReduceOp, x: &Tensor<T>, axis: usize) -> Vec<T> {
    let dims = x.shape().dims();
    let n = dims[axis];
    assert!(n > 0, "invariant: a reduced axis is non-empty");
    let inner: usize = dims[axis + 1..].iter().product();
    let outer: usize = dims[..axis].iter().product();
    let mut out = Vec::with_capacity(outer * inner);
    for o in 0..outer {
        for i in 0..inner {
            let at = |k: usize| x.data[o * n * inner + k * inner + i];
            out.push(match op {
                ReduceOp::Sum => {
                    let mut acc = T::zero();
                    for k in 0..n {
                        acc = acc.add(at(k));
                    }
                    acc
                }
                ReduceOp::Max => {
                    let mut acc = at(0);
                    for k in 1..n {
                        acc = binary(BinaryOp::Max, acc, at(k));
                    }
                    acc
                }
            });
        }
    }
    out
}

pub(super) fn broadcast_binary<T: Scalar>(
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

pub(super) fn unary<T: Scalar>(op: UnaryOp, x: T) -> T {
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

pub(super) fn evaluate<T: Scalar>(
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
            Op::Reduce(op, x, axis) => Tensor {
                shape: node.shape.clone(),
                data: reduce(*op, operand(&values, *x), *axis),
            },
        };
        values[index] = Some(value);
    }
    Ok(values
        .swap_remove(output.index())
        .expect("invariant: the output node is evaluated"))
}
