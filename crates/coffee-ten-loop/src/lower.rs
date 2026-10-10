//! Lowering of elementwise graphs to loop IR.

use std::fmt;

use coffee_ten_ir::{Graph, Node, NodeId, Op};

use crate::{Access, AffineExpr, BinaryOp, Buffer, Expr, Kernel, Loop, LoopNest, Store, UnaryOp};

/// Largest expression tree `lower_elementwise` builds. Shared graph nodes are repeated in
/// the tree, so a DAG can blow up exponentially (`x = x + x`, k times, gives 2^(k+1) - 1
/// nodes); the limit also bounds the recursion depth of everything that walks the tree.
/// Real elementwise chains have tens of nodes.
pub const MAX_EXPR_NODES: usize = 1000;

/// Lowers the elementwise graph ending at `output` into a kernel with one loop nest.
pub fn lower_elementwise(graph: &Graph, output: NodeId, name: &str) -> Result<Kernel, LowerError> {
    let out_dims = node(graph, output)?.shape.dims().to_vec();
    if out_dims.contains(&0) {
        return Err(LowerError::ZeroSizedOutput);
    }
    let nodes = tree_size(graph, output);
    if nodes > MAX_EXPR_NODES {
        return Err(LowerError::ExpressionTooLarge { nodes });
    }
    let out_rank = out_dims.len();
    let loop_names: Vec<String> = (0..out_rank).map(loop_name).collect();

    let is_reachable = reachable_nodes(graph, output);
    let mut buffers = Vec::new();
    let mut buffer_of = vec![None; graph.nodes().len()];
    for (index, node) in graph.nodes().iter().enumerate() {
        let Op::Input { name: input_name } = &node.op else {
            continue;
        };
        if !is_reachable[index] {
            continue;
        }
        if input_name == "out" || loop_names.contains(input_name) {
            return Err(LowerError::NameClash(input_name.clone()));
        }
        buffer_of[index] = Some(buffers.len());
        buffers.push(Buffer {
            name: input_name.clone(),
            shape: node.shape.dims().to_vec(),
        });
    }

    let out_buffer = buffers.len();
    buffers.push(Buffer {
        name: "out".to_string(),
        shape: out_dims.clone(),
    });
    let value = lower_value(graph, &buffer_of, out_rank, output)?;
    let loops = out_dims
        .iter()
        .zip(&loop_names)
        .map(|(&extent, loop_name)| Loop {
            name: loop_name.clone(),
            extent,
        })
        .collect();
    let target = Access {
        buffer: out_buffer,
        index: loop_names.iter().map(|loop_name| var(loop_name)).collect(),
    };
    Ok(Kernel {
        name: name.to_string(),
        buffers,
        nests: vec![LoopNest {
            loops,
            guards: Vec::new(),
            body: Store { target, value },
        }],
    })
}

#[derive(Clone, Debug, PartialEq)]
pub enum LowerError {
    UnknownNode(NodeId),
    /// An input is named like the output buffer `out` or like a generated loop variable.
    NameClash(String),
    /// The output has a dimension of size 0; a loop of extent 0 is not valid loop IR.
    ZeroSizedOutput,
    /// The expression tree would have more than `MAX_EXPR_NODES` nodes.
    ExpressionTooLarge {
        nodes: usize,
    },
}

impl fmt::Display for LowerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LowerError::UnknownNode(id) => write!(f, "unknown node %{}", id.index()),
            LowerError::NameClash(name) => {
                write!(f, "input {name} clashes with the output or a loop variable")
            }
            LowerError::ZeroSizedOutput => write!(f, "output has a dimension of size 0"),
            LowerError::ExpressionTooLarge { nodes } => {
                write!(f, "expression has {nodes} nodes, limit {MAX_EXPR_NODES}")
            }
        }
    }
}

impl std::error::Error for LowerError {}

fn node(graph: &Graph, id: NodeId) -> Result<&Node, LowerError> {
    graph.get(id).ok_or(LowerError::UnknownNode(id))
}

/// Marks the nodes that `output` depends on. Operands are added before their users,
/// so one backward pass from `output` finds them all.
fn reachable_nodes(graph: &Graph, output: NodeId) -> Vec<bool> {
    let nodes = graph.nodes();
    let mut is_reachable = vec![false; nodes.len()];
    is_reachable[output.index()] = true;
    for index in (0..=output.index()).rev() {
        if !is_reachable[index] {
            continue;
        }
        match &nodes[index].op {
            Op::Input { .. } | Op::Const(_) => {}
            Op::Unary(_, x) => is_reachable[x.index()] = true,
            Op::Binary(_, a, b) => {
                is_reachable[a.index()] = true;
                is_reachable[b.index()] = true;
            }
        }
    }
    is_reachable
}

/// Number of nodes in the expression tree of `output`, with shared nodes counted once per
/// use. Saturates instead of overflowing.
fn tree_size(graph: &Graph, output: NodeId) -> usize {
    let nodes = graph.nodes();
    let mut size = vec![0usize; output.index() + 1];
    for index in 0..=output.index() {
        size[index] = match &nodes[index].op {
            Op::Input { .. } | Op::Const(_) => 1,
            Op::Unary(_, x) => size[x.index()].saturating_add(1),
            Op::Binary(_, a, b) => size[a.index()]
                .saturating_add(size[b.index()])
                .saturating_add(1),
        };
    }
    size[output.index()]
}

/// Expression for node `id` at each point of the output loop nest.
fn lower_value(
    graph: &Graph,
    buffer_of: &[Option<usize>],
    out_rank: usize,
    id: NodeId,
) -> Result<Expr, LowerError> {
    let node = node(graph, id)?;
    let expr = match &node.op {
        Op::Input { .. } => {
            let buffer = buffer_of[id.index()].expect("invariant: reachable inputs have buffers");
            // Right-aligned broadcasting: input dim `dim` follows output loop `dim + offset`.
            let offset = out_rank
                .checked_sub(node.shape.rank())
                .expect("invariant: output rank is at least the operand rank");
            let index = node
                .shape
                .dims()
                .iter()
                .enumerate()
                .map(|(dim, &size)| {
                    if size == 1 {
                        AffineExpr {
                            constant: 0,
                            terms: Vec::new(),
                        }
                    } else {
                        var(&loop_name(dim + offset))
                    }
                })
                .collect();
            Expr::Load(Access { buffer, index })
        }
        Op::Const(value) => Expr::Const(*value),
        Op::Unary(op, x) => Expr::Unary(
            unary_op(*op),
            Box::new(lower_value(graph, buffer_of, out_rank, *x)?),
        ),
        Op::Binary(op, a, b) => Expr::Binary(
            binary_op(*op),
            Box::new(lower_value(graph, buffer_of, out_rank, *a)?),
            Box::new(lower_value(graph, buffer_of, out_rank, *b)?),
        ),
    };
    Ok(expr)
}

fn loop_name(axis: usize) -> String {
    format!("i{axis}")
}

fn var(name: &str) -> AffineExpr {
    AffineExpr {
        constant: 0,
        terms: vec![(name.to_string(), 1)],
    }
}

fn unary_op(op: coffee_ten_ir::UnaryOp) -> UnaryOp {
    match op {
        coffee_ten_ir::UnaryOp::Neg => UnaryOp::Neg,
        coffee_ten_ir::UnaryOp::Relu => UnaryOp::Relu,
    }
}

fn binary_op(op: coffee_ten_ir::BinaryOp) -> BinaryOp {
    match op {
        coffee_ten_ir::BinaryOp::Add => BinaryOp::Add,
        coffee_ten_ir::BinaryOp::Sub => BinaryOp::Sub,
        coffee_ten_ir::BinaryOp::Mul => BinaryOp::Mul,
        coffee_ten_ir::BinaryOp::Max => BinaryOp::Max,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use coffee_ten_ir::Shape;

    fn shape(dims: &[usize]) -> Shape {
        Shape::new(dims.to_vec())
    }

    /// Lowers as kernel `k`, checks that the kernel validates, and returns its text.
    fn lowered_text(graph: &Graph, output: NodeId) -> String {
        let kernel = lower_elementwise(graph, output, "k").unwrap();
        assert_eq!(kernel.validate(), Ok(()));
        kernel.to_string()
    }

    #[test]
    fn relu_of_broadcast_add_is_one_nest() {
        let mut g = Graph::new();
        let a = g.input("A", shape(&[2, 3])).unwrap();
        let bias = g.input("bias", shape(&[3])).unwrap();
        let sum = g.binary(coffee_ten_ir::BinaryOp::Add, a, bias).unwrap();
        let out = g.unary(coffee_ten_ir::UnaryOp::Relu, sum).unwrap();
        assert_eq!(
            lowered_text(&g, out),
            concat!(
                "kernel k(A: f32[2, 3], bias: f32[3], out: f32[2, 3])\n",
                "for i0 in 0..2:\n",
                "  for i1 in 0..3:\n",
                "    out[i0, i1] = relu(A[i0, i1] + bias[i1])\n",
            )
        );
    }

    #[test]
    fn size_one_dims_index_zero() {
        let mut g = Graph::new();
        let a = g.input("A", shape(&[2, 1])).unwrap();
        let b = g.input("B", shape(&[3])).unwrap();
        let prod = g.binary(coffee_ten_ir::BinaryOp::Mul, a, b).unwrap();
        assert_eq!(
            lowered_text(&g, prod),
            concat!(
                "kernel k(A: f32[2, 1], B: f32[3], out: f32[2, 3])\n",
                "for i0 in 0..2:\n",
                "  for i1 in 0..3:\n",
                "    out[i0, i1] = A[i0, 0] * B[i1]\n",
            )
        );
    }

    #[test]
    fn constant_operand_is_printed_inline() {
        let mut g = Graph::new();
        let two = g.constant(2.0);
        let a = g.input("A", shape(&[4])).unwrap();
        let sum = g.binary(coffee_ten_ir::BinaryOp::Add, two, a).unwrap();
        assert_eq!(
            lowered_text(&g, sum),
            concat!(
                "kernel k(A: f32[4], out: f32[4])\n",
                "for i0 in 0..4:\n",
                "  out[i0] = 2.0 + A[i0]\n",
            )
        );
    }

    #[test]
    fn rank_zero_output_has_no_loops() {
        let mut g = Graph::new();
        let s = g.input("s", Shape::scalar()).unwrap();
        let out = g.unary(coffee_ten_ir::UnaryOp::Neg, s).unwrap();
        assert_eq!(
            lowered_text(&g, out),
            "kernel k(s: f32[], out: f32[])\nout[] = -(s[])\n"
        );
    }

    #[test]
    fn shared_input_is_one_buffer() {
        let mut g = Graph::new();
        let a = g.input("A", shape(&[4])).unwrap();
        let out = g.binary(coffee_ten_ir::BinaryOp::Add, a, a).unwrap();
        assert_eq!(
            lowered_text(&g, out),
            concat!(
                "kernel k(A: f32[4], out: f32[4])\n",
                "for i0 in 0..4:\n",
                "  out[i0] = A[i0] + A[i0]\n",
            )
        );
    }

    #[test]
    fn input_not_reachable_from_output_is_not_a_buffer() {
        let mut g = Graph::new();
        let a = g.input("A", shape(&[2])).unwrap();
        g.input("unused", shape(&[5])).unwrap();
        let out = g.unary(coffee_ten_ir::UnaryOp::Neg, a).unwrap();
        assert_eq!(
            lowered_text(&g, out),
            concat!(
                "kernel k(A: f32[2], out: f32[2])\n",
                "for i0 in 0..2:\n",
                "  out[i0] = -(A[i0])\n",
            )
        );
    }

    #[test]
    fn unknown_output_node_is_an_error() {
        let mut big = Graph::new();
        let ids: Vec<NodeId> = (0..6).map(|_| big.constant(0.0)).collect();
        let foreign = ids[5];

        let mut g = Graph::new();
        g.input("A", shape(&[2])).unwrap();
        g.constant(1.0);
        let err = lower_elementwise(&g, foreign, "k").unwrap_err();
        assert_eq!(err, LowerError::UnknownNode(foreign));
        assert_eq!(err.to_string(), "unknown node %5");
    }

    #[test]
    fn input_named_out_is_a_name_clash() {
        let mut g = Graph::new();
        let x = g.input("out", shape(&[2])).unwrap();
        let y = g.unary(coffee_ten_ir::UnaryOp::Neg, x).unwrap();
        let err = lower_elementwise(&g, y, "k").unwrap_err();
        assert_eq!(err, LowerError::NameClash("out".to_string()));
        assert_eq!(
            err.to_string(),
            "input out clashes with the output or a loop variable"
        );
    }

    #[test]
    fn input_named_like_a_loop_variable_is_a_name_clash() {
        let mut g = Graph::new();
        let x = g.input("i0", shape(&[2])).unwrap();
        let y = g.unary(coffee_ten_ir::UnaryOp::Neg, x).unwrap();
        let err = lower_elementwise(&g, y, "k").unwrap_err();
        assert_eq!(err, LowerError::NameClash("i0".to_string()));
    }

    #[test]
    fn zero_sized_output_is_an_error() {
        let mut g = Graph::new();
        let a = g.input("A", shape(&[0])).unwrap();
        let y = g.unary(coffee_ten_ir::UnaryOp::Neg, a).unwrap();
        let err = lower_elementwise(&g, y, "k").unwrap_err();
        assert_eq!(err, LowerError::ZeroSizedOutput);
        assert_eq!(err.to_string(), "output has a dimension of size 0");
    }

    /// `A` followed by `negs` neg nodes: a tree of `negs + 1` nodes.
    fn neg_chain(negs: usize) -> (Graph, NodeId) {
        let mut g = Graph::new();
        let mut x = g.input("A", shape(&[2])).unwrap();
        for _ in 0..negs {
            x = g.unary(coffee_ten_ir::UnaryOp::Neg, x).unwrap();
        }
        (g, x)
    }

    #[test]
    fn expression_at_the_node_limit_lowers() {
        let (g, y) = neg_chain(MAX_EXPR_NODES - 1);
        let kernel = lower_elementwise(&g, y, "k").unwrap();
        assert_eq!(kernel.validate(), Ok(()));
    }

    #[test]
    fn expression_over_the_node_limit_is_an_error() {
        let (g, y) = neg_chain(MAX_EXPR_NODES);
        assert_eq!(
            lower_elementwise(&g, y, "k"),
            Err(LowerError::ExpressionTooLarge {
                nodes: MAX_EXPR_NODES + 1
            })
        );
    }

    #[test]
    fn doubling_dag_is_caught_before_the_tree_is_built() {
        // x = x + x ten times: 2^11 - 1 = 2047 tree nodes from 11 graph nodes.
        let mut g = Graph::new();
        let mut x = g.input("A", shape(&[2])).unwrap();
        for _ in 0..10 {
            x = g.binary(coffee_ten_ir::BinaryOp::Add, x, x).unwrap();
        }
        assert_eq!(
            lower_elementwise(&g, x, "k"),
            Err(LowerError::ExpressionTooLarge { nodes: 2047 })
        );
    }
}
