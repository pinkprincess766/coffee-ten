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
    if let Some(matmul) = find_matmul(graph, output)? {
        return Err(LowerError::NotElementwise(matmul));
    }
    let out_dims = node(graph, output)?.shape.dims().to_vec();
    if out_dims.contains(&0) {
        return Err(LowerError::ZeroSizedOutput);
    }
    check_tree_size(graph, output)?;
    let mut reserved = vec!["out".to_string()];
    reserved.extend((0..out_dims.len()).map(loop_name));

    let is_reachable = reachable_nodes(graph, output);
    let (mut buffers, buffer_of) = input_buffers(graph, &is_reachable, &reserved)?;
    let out_buffer = buffers.len();
    buffers.push(Buffer {
        name: "out".to_string(),
        shape: out_dims,
    });
    let nest = elementwise_nest(graph, output, &buffer_of, out_buffer)?;
    Ok(Kernel {
        name: name.to_string(),
        buffers,
        nests: vec![nest],
    })
}

/// Lowers the graph ending at `output`: every reachable matmul gets an init nest and an
/// update nest; the elementwise part, if any, becomes one more nest.
///
/// Buffers are the reachable inputs in node order, then one temporary `t0`, `t1`, ... per
/// matmul that is not the output (a kernel parameter: the caller provides the scratch
/// memory), then `out`. The update nest of a matmul runs `i, k, j` with `k` ascending for
/// every element of the result, which is exactly the order of the interpreter's `run_f32`
/// (`acc = 0; acc += a[i, k] * b[k, j]`), so the two agree bit for bit.
pub fn lower(graph: &Graph, output: NodeId, name: &str) -> Result<Kernel, LowerError> {
    let out_dims = node(graph, output)?.shape.dims().to_vec();
    if out_dims.contains(&0) {
        return Err(LowerError::ZeroSizedOutput);
    }
    let matmuls = reachable_matmuls(graph, output)?;
    check_tree_size(graph, output)?;

    let output_is_matmul = matches!(node(graph, output)?.op, Op::MatMul(..));
    let temp_count = matmuls.len() - usize::from(output_is_matmul);
    let mut reserved = vec!["out".to_string()];
    if !output_is_matmul {
        reserved.extend((0..out_dims.len()).map(loop_name));
    }
    if !matmuls.is_empty() {
        reserved.extend(["i", "j", "k"].map(String::from));
    }
    reserved.extend((0..temp_count).map(temp_name));

    let is_reachable = reachable_nodes(graph, output);
    let (mut buffers, mut buffer_of) = input_buffers(graph, &is_reachable, &reserved)?;
    let mut next_temp = 0;
    for matmul in &matmuls {
        if matmul.id == output {
            continue;
        }
        buffer_of[matmul.id.index()] = Some(buffers.len());
        buffers.push(Buffer {
            name: temp_name(next_temp),
            shape: node(graph, matmul.id)?.shape.dims().to_vec(),
        });
        next_temp += 1;
    }
    let out_buffer = buffers.len();
    buffers.push(Buffer {
        name: "out".to_string(),
        shape: out_dims,
    });

    let mut nests = Vec::new();
    for matmul in &matmuls {
        let target = buffer_of[matmul.id.index()].unwrap_or(out_buffer);
        let buffer_a = buffer_of[matmul.a.index()].expect("invariant: operands are inputs");
        let buffer_b = buffer_of[matmul.b.index()].expect("invariant: operands are inputs");
        nests.extend(matmul_nests(matmul.dims, buffer_a, buffer_b, target));
    }
    if !output_is_matmul {
        nests.push(elementwise_nest(graph, output, &buffer_of, out_buffer)?);
    }
    Ok(Kernel {
        name: name.to_string(),
        buffers,
        nests,
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
    /// A matmul is reachable from the output; only elementwise graphs lower to one nest.
    NotElementwise(NodeId),
    /// `lower` cannot handle this node yet.
    Unsupported {
        node: NodeId,
        reason: &'static str,
    },
}

impl fmt::Display for LowerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LowerError::UnknownNode(id) => write!(f, "unknown node %{}", id.index()),
            LowerError::NotElementwise(id) => write!(f, "node %{} is not elementwise", id.index()),
            LowerError::Unsupported { node, reason } => {
                write!(f, "node %{} is not supported: {reason}", node.index())
            }
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

fn node_dims(graph: &Graph, id: NodeId) -> Result<&[usize], LowerError> {
    Ok(node(graph, id)?.shape.dims())
}

/// The first MatMul reachable from `output`. Each node is expanded once, so a shared
/// subgraph is not walked once per use.
fn find_matmul(graph: &Graph, output: NodeId) -> Result<Option<NodeId>, LowerError> {
    let mut seen = vec![false; graph.nodes().len()];
    let mut stack = vec![output];
    while let Some(id) = stack.pop() {
        let node = node(graph, id)?;
        if std::mem::replace(&mut seen[id.index()], true) {
            continue;
        }
        match &node.op {
            Op::MatMul(..) => return Ok(Some(id)),
            Op::Input { .. } | Op::Const(_) => {}
            Op::Unary(_, x) => stack.push(*x),
            Op::Binary(_, a, b) => stack.extend([*a, *b]),
        }
    }
    Ok(None)
}

/// A matmul reachable from the output: `a` is `(m, k)`, `b` is `(k, n)`, `dims = (m, k, n)`.
struct MatMulNode {
    id: NodeId,
    a: NodeId,
    b: NodeId,
    dims: (usize, usize, usize),
}

/// The matmuls reachable from `output` in node order. Their operands must be inputs and
/// none of their dims may be 0.
fn reachable_matmuls(graph: &Graph, output: NodeId) -> Result<Vec<MatMulNode>, LowerError> {
    let mut seen = vec![false; graph.nodes().len()];
    let mut stack = vec![output];
    let mut matmuls = Vec::new();
    while let Some(id) = stack.pop() {
        let node = node(graph, id)?;
        if std::mem::replace(&mut seen[id.index()], true) {
            continue;
        }
        match &node.op {
            Op::MatMul(a, b) => {
                let (a_dims, b_dims) = (node_dims(graph, *a)?, node_dims(graph, *b)?);
                let is_input =
                    |x: NodeId| matches!(graph.get(x), Some(n) if matches!(n.op, Op::Input { .. }));
                if !is_input(*a) || !is_input(*b) {
                    return Err(LowerError::Unsupported {
                        node: id,
                        reason: "matmul operand must be an input",
                    });
                }
                let dims = (a_dims[0], a_dims[1], b_dims[1]);
                if dims.0 == 0 || dims.1 == 0 || dims.2 == 0 {
                    return Err(LowerError::Unsupported {
                        node: id,
                        reason: "matmul has a zero-sized dimension",
                    });
                }
                matmuls.push(MatMulNode {
                    id,
                    a: *a,
                    b: *b,
                    dims,
                });
            }
            Op::Input { .. } | Op::Const(_) => {}
            Op::Unary(_, x) => stack.push(*x),
            Op::Binary(_, a, b) => stack.extend([*a, *b]),
        }
    }
    matmuls.sort_by_key(|matmul| matmul.id);
    Ok(matmuls)
}

fn check_tree_size(graph: &Graph, output: NodeId) -> Result<(), LowerError> {
    let nodes = tree_size(graph, output);
    if nodes > MAX_EXPR_NODES {
        return Err(LowerError::ExpressionTooLarge { nodes });
    }
    Ok(())
}

/// One buffer per reachable input in node order, and the buffer index of each node.
/// An input named like one of `reserved` is a name clash.
fn input_buffers(
    graph: &Graph,
    is_reachable: &[bool],
    reserved: &[String],
) -> Result<(Vec<Buffer>, Vec<Option<usize>>), LowerError> {
    let mut buffers = Vec::new();
    let mut buffer_of = vec![None; graph.nodes().len()];
    for (index, node) in graph.nodes().iter().enumerate() {
        let Op::Input { name: input_name } = &node.op else {
            continue;
        };
        if !is_reachable[index] {
            continue;
        }
        if reserved.contains(input_name) {
            return Err(LowerError::NameClash(input_name.clone()));
        }
        buffer_of[index] = Some(buffers.len());
        buffers.push(Buffer {
            name: input_name.clone(),
            shape: node.shape.dims().to_vec(),
        });
    }
    Ok((buffers, buffer_of))
}

/// The nest that stores the elementwise expression of `output` into buffer `out_buffer`.
fn elementwise_nest(
    graph: &Graph,
    output: NodeId,
    buffer_of: &[Option<usize>],
    out_buffer: usize,
) -> Result<LoopNest, LowerError> {
    let out_dims = node_dims(graph, output)?;
    let out_rank = out_dims.len();
    let loop_names: Vec<String> = (0..out_rank).map(loop_name).collect();
    let value = lower_value(graph, buffer_of, out_rank, output)?;
    let loops = out_dims
        .iter()
        .zip(&loop_names)
        .map(|(&extent, loop_name)| Loop {
            name: loop_name.clone(),
            extent,
            vectorize: false,
        })
        .collect();
    let target = Access {
        buffer: out_buffer,
        index: loop_names.iter().map(|loop_name| var(loop_name)).collect(),
    };
    Ok(LoopNest {
        loops,
        guards: Vec::new(),
        body: Store { target, value },
    })
}

fn temp_name(n: usize) -> String {
    format!("t{n}")
}

/// Init and update nest of `target = A @ B`, `dims = (m, k, n)`.
fn matmul_nests(
    (m, k, n): (usize, usize, usize),
    buffer_a: usize,
    buffer_b: usize,
    target: usize,
) -> [LoopNest; 2] {
    let lp = |name: &str, extent| Loop {
        name: name.to_string(),
        extent,
        vectorize: false,
    };
    let at = |buffer, names: [&str; 2]| Access {
        buffer,
        index: names.map(var).to_vec(),
    };
    let load = |access| Box::new(Expr::Load(access));
    let init = LoopNest {
        loops: vec![lp("i", m), lp("j", n)],
        guards: Vec::new(),
        body: Store {
            target: at(target, ["i", "j"]),
            value: Expr::Const(0.0),
        },
    };
    let product = Expr::Binary(
        BinaryOp::Mul,
        load(at(buffer_a, ["i", "k"])),
        load(at(buffer_b, ["k", "j"])),
    );
    let update = LoopNest {
        loops: vec![lp("i", m), lp("k", k), lp("j", n)],
        guards: Vec::new(),
        body: Store {
            target: at(target, ["i", "j"]),
            value: Expr::Binary(
                BinaryOp::Add,
                load(at(target, ["i", "j"])),
                Box::new(product),
            ),
        },
    };
    [init, update]
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
            Op::Binary(_, a, b) | Op::MatMul(a, b) => {
                is_reachable[a.index()] = true;
                is_reachable[b.index()] = true;
            }
        }
    }
    is_reachable
}

/// Number of nodes in the expression tree of `output`, with shared nodes counted once per
/// use; a matmul counts as one leaf. Saturates instead of overflowing.
fn tree_size(graph: &Graph, output: NodeId) -> usize {
    let nodes = graph.nodes();
    let mut size = vec![0usize; output.index() + 1];
    for index in 0..=output.index() {
        size[index] = match &nodes[index].op {
            Op::Input { .. } | Op::Const(_) | Op::MatMul(..) => 1,
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
        Op::Input { .. } | Op::MatMul(..) => {
            let buffer = buffer_of[id.index()]
                .expect("invariant: reachable inputs and matmuls have buffers");
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

    #[test]
    fn matmul_under_elementwise_ops_is_not_elementwise() {
        // relu(A @ B + bias): the matmul is node %3.
        let mut g = Graph::new();
        let a = g.input("A", shape(&[2, 3])).unwrap();
        let b = g.input("B", shape(&[3, 2])).unwrap();
        let bias = g.input("bias", shape(&[2])).unwrap();
        let prod = g.matmul(a, b).unwrap();
        let sum = g.binary(coffee_ten_ir::BinaryOp::Add, prod, bias).unwrap();
        let out = g.unary(coffee_ten_ir::UnaryOp::Relu, sum).unwrap();
        assert_eq!(prod.index(), 3);
        let err = lower_elementwise(&g, out, "k").unwrap_err();
        assert_eq!(err, LowerError::NotElementwise(prod));
        assert_eq!(err.to_string(), "node %3 is not elementwise");
    }

    #[test]
    fn matmul_outside_the_output_does_not_block_lowering() {
        let mut g = Graph::new();
        let a = g.input("A", shape(&[2, 2])).unwrap();
        let b = g.input("B", shape(&[2, 2])).unwrap();
        g.matmul(a, b).unwrap();
        let out = g.unary(coffee_ten_ir::UnaryOp::Neg, a).unwrap();
        assert_eq!(
            lowered_text(&g, out),
            concat!(
                "kernel k(A: f32[2, 2], out: f32[2, 2])\n",
                "for i0 in 0..2:\n",
                "  for i1 in 0..2:\n",
                "    out[i0, i1] = -(A[i0, i1])\n",
            )
        );
    }

    #[test]
    fn matmul_is_reported_before_expression_size() {
        // A @ A, then x = x + x twelve times: far over the node limit.
        let mut g = Graph::new();
        let a = g.input("A", shape(&[2, 2])).unwrap();
        let mm = g.matmul(a, a).unwrap();
        let mut x = mm;
        for _ in 0..12 {
            x = g.binary(coffee_ten_ir::BinaryOp::Add, x, x).unwrap();
        }
        assert_eq!(
            lower_elementwise(&g, x, "k"),
            Err(LowerError::NotElementwise(mm))
        );
    }

    #[test]
    fn shared_elementwise_dag_is_walked_once_for_the_matmul_check() {
        // x = x + x sixty times: 2^61 - 1 tree nodes from 61 graph nodes. The matmul check
        // must visit each graph node once, or this test does not finish.
        let mut g = Graph::new();
        let mut x = g.input("A", shape(&[2])).unwrap();
        for _ in 0..60 {
            x = g.binary(coffee_ten_ir::BinaryOp::Add, x, x).unwrap();
        }
        assert_eq!(
            lower_elementwise(&g, x, "k"),
            Err(LowerError::ExpressionTooLarge {
                nodes: (1 << 61) - 1
            })
        );
    }

    /// `lower` as kernel `k`, checks that the kernel validates, and returns its text.
    fn full_text(graph: &Graph, output: NodeId) -> String {
        let kernel = lower(graph, output, "k").unwrap();
        assert_eq!(kernel.validate(), Ok(()));
        kernel.to_string()
    }

    /// `A (4, 3)`, `B (3, 2)` and their product.
    fn matmul_graph() -> (Graph, NodeId, NodeId, NodeId) {
        let mut g = Graph::new();
        let a = g.input("A", shape(&[4, 3])).unwrap();
        let b = g.input("B", shape(&[3, 2])).unwrap();
        let prod = g.matmul(a, b).unwrap();
        (g, a, b, prod)
    }

    /// `relu(A @ B + bias_name)` with the bias of shape `(2,)`.
    fn relu_matmul_bias(bias_name: &str) -> (Graph, NodeId) {
        let (mut g, _, _, prod) = matmul_graph();
        let bias = g.input(bias_name, shape(&[2])).unwrap();
        let sum = g.binary(coffee_ten_ir::BinaryOp::Add, prod, bias).unwrap();
        let out = g.unary(coffee_ten_ir::UnaryOp::Relu, sum).unwrap();
        (g, out)
    }

    #[test]
    fn matmul_output_gets_init_and_update_nests() {
        let (g, _, _, prod) = matmul_graph();
        assert_eq!(
            full_text(&g, prod),
            concat!(
                "kernel k(A: f32[4, 3], B: f32[3, 2], out: f32[4, 2])\n",
                "for i in 0..4:\n",
                "  for j in 0..2:\n",
                "    out[i, j] = 0.0\n",
                "for i in 0..4:\n",
                "  for k in 0..3:\n",
                "    for j in 0..2:\n",
                "      out[i, j] = out[i, j] + (A[i, k] * B[k, j])\n",
            )
        );
    }

    #[test]
    fn relu_of_matmul_plus_bias_uses_a_temporary() {
        let (g, out) = relu_matmul_bias("bias");
        assert_eq!(
            full_text(&g, out),
            concat!(
                "kernel k(A: f32[4, 3], B: f32[3, 2], bias: f32[2], t0: f32[4, 2], out: f32[4, 2])\n",
                "for i in 0..4:\n",
                "  for j in 0..2:\n",
                "    t0[i, j] = 0.0\n",
                "for i in 0..4:\n",
                "  for k in 0..3:\n",
                "    for j in 0..2:\n",
                "      t0[i, j] = t0[i, j] + (A[i, k] * B[k, j])\n",
                "for i0 in 0..4:\n",
                "  for i1 in 0..2:\n",
                "    out[i0, i1] = relu(t0[i0, i1] + bias[i1])\n",
            )
        );
    }

    #[test]
    fn two_matmuls_get_temporaries_in_node_order() {
        // A @ B + C @ D: the matmuls are %4 and %5, both read through t0 and t1.
        let mut g = Graph::new();
        let a = g.input("A", shape(&[2, 2])).unwrap();
        let b = g.input("B", shape(&[2, 2])).unwrap();
        let c = g.input("C", shape(&[2, 2])).unwrap();
        let d = g.input("D", shape(&[2, 2])).unwrap();
        let left = g.matmul(a, b).unwrap();
        let right = g.matmul(c, d).unwrap();
        let sum = g.binary(coffee_ten_ir::BinaryOp::Add, left, right).unwrap();
        let text = full_text(&g, sum);
        assert!(text.starts_with(
            "kernel k(A: f32[2, 2], B: f32[2, 2], C: f32[2, 2], D: f32[2, 2], \
             t0: f32[2, 2], t1: f32[2, 2], out: f32[2, 2])\n"
        ));
        assert!(text.contains("      t0[i, j] = t0[i, j] + (A[i, k] * B[k, j])\n"));
        assert!(text.contains("      t1[i, j] = t1[i, j] + (C[i, k] * D[k, j])\n"));
        assert!(text.ends_with("    out[i0, i1] = t0[i0, i1] + t1[i0, i1]\n"));
        assert_eq!(text.matches("for i in").count(), 4);
    }

    #[test]
    fn pure_elementwise_graph_lowers_like_lower_elementwise() {
        let mut g = Graph::new();
        let a = g.input("A", shape(&[2, 3])).unwrap();
        let bias = g.input("bias", shape(&[3])).unwrap();
        let sum = g.binary(coffee_ten_ir::BinaryOp::Add, a, bias).unwrap();
        let out = g.unary(coffee_ten_ir::UnaryOp::Relu, sum).unwrap();
        assert_eq!(
            lower(&g, out, "k").unwrap(),
            lower_elementwise(&g, out, "k").unwrap()
        );
    }

    #[test]
    fn matmul_operand_that_is_not_an_input_is_unsupported() {
        // (A + 1) @ B: the matmul is node %4.
        let mut g = Graph::new();
        let a = g.input("A", shape(&[4, 3])).unwrap();
        let b = g.input("B", shape(&[3, 2])).unwrap();
        let one = g.constant(1.0);
        let shifted = g.binary(coffee_ten_ir::BinaryOp::Add, a, one).unwrap();
        let prod = g.matmul(shifted, b).unwrap();
        let err = lower(&g, prod, "k").unwrap_err();
        assert_eq!(
            err,
            LowerError::Unsupported {
                node: prod,
                reason: "matmul operand must be an input"
            }
        );
        assert_eq!(
            err.to_string(),
            "node %4 is not supported: matmul operand must be an input"
        );
    }

    #[test]
    fn matmul_of_a_matmul_is_unsupported() {
        let (mut g, _, _, prod) = matmul_graph();
        let c = g.input("C", shape(&[2, 5])).unwrap();
        let chained = g.matmul(prod, c).unwrap();
        assert_eq!(
            lower(&g, chained, "k"),
            Err(LowerError::Unsupported {
                node: chained,
                reason: "matmul operand must be an input"
            })
        );
    }

    #[test]
    fn matmul_with_an_empty_inner_dim_is_unsupported() {
        // (2, 0) @ (0, 3) is a valid (2, 3) zero matrix, but a loop of extent 0 is not valid.
        let mut g = Graph::new();
        let a = g.input("A", shape(&[2, 0])).unwrap();
        let b = g.input("B", shape(&[0, 3])).unwrap();
        let prod = g.matmul(a, b).unwrap();
        assert_eq!(
            lower(&g, prod, "k"),
            Err(LowerError::Unsupported {
                node: prod,
                reason: "matmul has a zero-sized dimension"
            })
        );
    }

    #[test]
    fn zero_sized_matmul_output_is_an_error() {
        let mut g = Graph::new();
        let a = g.input("A", shape(&[0, 3])).unwrap();
        let b = g.input("B", shape(&[3, 2])).unwrap();
        let prod = g.matmul(a, b).unwrap();
        assert_eq!(lower(&g, prod, "k"), Err(LowerError::ZeroSizedOutput));
    }

    #[test]
    fn unknown_output_node_is_an_error_in_lower() {
        let mut big = Graph::new();
        let ids: Vec<NodeId> = (0..6).map(|_| big.constant(0.0)).collect();
        let g = Graph::new();
        assert_eq!(lower(&g, ids[5], "k"), Err(LowerError::UnknownNode(ids[5])));
    }

    #[test]
    fn input_named_like_a_used_temporary_is_a_name_clash() {
        let (g, out) = relu_matmul_bias("t0");
        assert_eq!(
            lower(&g, out, "k"),
            Err(LowerError::NameClash("t0".to_string()))
        );
    }

    #[test]
    fn input_named_like_an_unused_temporary_is_fine() {
        // Here the matmul is the output, so there is no t0.
        let mut g = Graph::new();
        let a = g.input("t0", shape(&[2, 2])).unwrap();
        let b = g.input("B", shape(&[2, 2])).unwrap();
        let prod = g.matmul(a, b).unwrap();
        assert!(full_text(&g, prod).starts_with("kernel k(t0: f32[2, 2], B"));
    }

    #[test]
    fn input_named_like_a_matmul_loop_variable_is_a_name_clash() {
        for clash in ["i", "j", "k", "out"] {
            let (g, out) = relu_matmul_bias(clash);
            assert_eq!(
                lower(&g, out, "k"),
                Err(LowerError::NameClash(clash.to_string())),
                "input {clash}"
            );
        }
    }

    #[test]
    fn input_named_like_an_elementwise_loop_variable_is_a_name_clash() {
        let (g, out) = relu_matmul_bias("i1");
        assert_eq!(
            lower(&g, out, "k"),
            Err(LowerError::NameClash("i1".to_string()))
        );
    }

    #[test]
    fn matmul_loop_variable_names_are_free_without_a_matmul() {
        let mut g = Graph::new();
        let k = g.input("k", shape(&[2])).unwrap();
        let out = g.unary(coffee_ten_ir::UnaryOp::Neg, k).unwrap();
        assert_eq!(
            lower(&g, out, "n").unwrap(),
            lower_elementwise(&g, out, "n").unwrap()
        );
    }

    /// `matmul` followed by `negs` neg nodes: a tree of `negs + 1` nodes.
    fn neg_chain_over_matmul(negs: usize) -> (Graph, NodeId) {
        let (mut g, _, _, mut x) = matmul_graph();
        for _ in 0..negs {
            x = g.unary(coffee_ten_ir::UnaryOp::Neg, x).unwrap();
        }
        (g, x)
    }

    #[test]
    fn matmul_counts_as_one_leaf_at_the_node_limit() {
        let (g, out) = neg_chain_over_matmul(MAX_EXPR_NODES - 1);
        assert_eq!(lower(&g, out, "k").unwrap().validate(), Ok(()));
    }

    #[test]
    fn matmul_expression_over_the_node_limit_is_an_error() {
        let (g, out) = neg_chain_over_matmul(MAX_EXPR_NODES);
        assert_eq!(
            lower(&g, out, "k"),
            Err(LowerError::ExpressionTooLarge {
                nodes: MAX_EXPR_NODES + 1
            })
        );
    }
}
