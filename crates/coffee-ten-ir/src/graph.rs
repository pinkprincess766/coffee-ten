use std::fmt;

use crate::shape::{BroadcastError, Shape};

/// Index of a node in its Graph; only valid for the graph that returned it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(usize);

impl NodeId {
    pub fn index(self) -> usize {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Relu,
    /// `e^x`. Evaluated by the platform libm, so the result is not exactly rounded and
    /// differs between libms; the one op whose f32 results are compared with a tolerance.
    Exp,
}

impl UnaryOp {
    fn name(self) -> &'static str {
        match self {
            UnaryOp::Neg => "neg",
            UnaryOp::Relu => "relu",
            UnaryOp::Exp => "exp",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Max,
}

impl BinaryOp {
    fn name(self) -> &'static str {
        match self {
            BinaryOp::Add => "add",
            BinaryOp::Sub => "sub",
            BinaryOp::Mul => "mul",
            BinaryOp::Div => "div",
            BinaryOp::Max => "max",
        }
    }
}

/// A reduction along one axis: the sum or the maximum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Max,
}

impl ReduceOp {
    fn name(self) -> &'static str {
        match self {
            ReduceOp::Sum => "sum",
            ReduceOp::Max => "max",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    Input {
        name: String,
    },
    /// Scalar constant.
    Const(f32),
    Unary(UnaryOp, NodeId),
    Binary(BinaryOp, NodeId, NodeId),
    /// `(M, K) @ (K, N) -> (M, N)`, both operands rank 2.
    MatMul(NodeId, NodeId),
    /// Reduces one axis and keeps it with size 1. The axis is counted from the start:
    /// `(2, 3)` over axis 1 is `(2, 1)`, over axis 0 is `(1, 3)`, and `(4,)` over axis 0
    /// is `(1,)`.
    Reduce {
        op: ReduceOp,
        x: NodeId,
        axis: usize,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    pub op: Op,
    pub shape: Shape,
}

/// SSA DAG in an arena: a node only refers to nodes added before it,
/// so `nodes()` is always in topological order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Graph {
    nodes: Vec<Node>,
}

impl Graph {
    pub fn new() -> Graph {
        Graph { nodes: Vec::new() }
    }

    pub fn input(&mut self, name: &str, shape: Shape) -> Result<NodeId, GraphError> {
        let is_duplicate = self.nodes.iter().any(|node| match &node.op {
            Op::Input { name: existing } => existing == name,
            _ => false,
        });
        if is_duplicate {
            return Err(GraphError::DuplicateInput(name.to_string()));
        }
        let op = Op::Input {
            name: name.to_string(),
        };
        Ok(self.push(op, shape))
    }

    pub fn constant(&mut self, value: f32) -> NodeId {
        self.push(Op::Const(value), Shape::scalar())
    }

    pub fn unary(&mut self, op: UnaryOp, x: NodeId) -> Result<NodeId, GraphError> {
        let x_shape = self.node(x)?.shape.clone();
        Ok(self.push(Op::Unary(op, x), x_shape))
    }

    pub fn binary(&mut self, op: BinaryOp, a: NodeId, b: NodeId) -> Result<NodeId, GraphError> {
        let a_shape = &self.node(a)?.shape;
        let b_shape = &self.node(b)?.shape;
        let shape = a_shape.broadcast(b_shape).map_err(GraphError::Broadcast)?;
        Ok(self.push(Op::Binary(op, a, b), shape))
    }

    /// Two-dimensional matrix product only: no broadcasting, batching or rank-1 operands.
    pub fn matmul(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, GraphError> {
        let a_shape = &self.node(a)?.shape;
        let b_shape = &self.node(b)?.shape;
        let shape = match (a_shape.dims(), b_shape.dims()) {
            ([m, k], [k_right, n]) if k == k_right => Shape::new(vec![*m, *n]),
            _ => {
                return Err(GraphError::MatMulShape {
                    left: a_shape.clone(),
                    right: b_shape.clone(),
                });
            }
        };
        Ok(self.push(Op::MatMul(a, b), shape))
    }

    /// Reduces the last axis of `x`, keeping it with size 1, so `x - reduce(max, x)`
    /// broadcasts. A rank-0 operand or an empty last axis is an error.
    pub fn reduce(&mut self, op: ReduceOp, x: NodeId) -> Result<NodeId, GraphError> {
        self.reduce_axis(op, x, -1)
    }

    /// Reduces `axis` of `x`, keeping it with size 1. A negative axis counts from the
    /// end, as in NumPy: `-1` is the last axis. See [`reduced_shape`].
    pub fn reduce_axis(
        &mut self,
        op: ReduceOp,
        x: NodeId,
        axis: isize,
    ) -> Result<NodeId, GraphError> {
        let x_shape = self.node(x)?.shape.clone();
        let (shape, axis) = reduced_shape(&x_shape, axis)?;
        Ok(self.push(Op::Reduce { op, x, axis }, shape))
    }

    pub fn get(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(id.index())
    }

    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    fn node(&self, id: NodeId) -> Result<&Node, GraphError> {
        self.get(id).ok_or(GraphError::UnknownNode(id))
    }

    fn push(&mut self, op: Op, shape: Shape) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(Node { op, shape });
        id
    }
}

impl fmt::Display for Graph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, node) in self.nodes.iter().enumerate() {
            let text = match &node.op {
                Op::Input { name } => format!("input {name}"),
                Op::Const(value) => format!("const {value:?}"),
                Op::Unary(op, x) => format!("{} %{}", op.name(), x.index()),
                Op::Binary(op, a, b) => {
                    format!("{} %{}, %{}", op.name(), a.index(), b.index())
                }
                Op::MatMul(a, b) => format!("matmul %{}, %{}", a.index(), b.index()),
                Op::Reduce { op, x, axis } => {
                    // The last axis is the historical form, so its text stays `sum %0`.
                    if *axis + 1 == node.shape.rank() {
                        format!("{} %{}", op.name(), x.index())
                    } else {
                        format!("{} %{} axis {axis}", op.name(), x.index())
                    }
                }
            };
            writeln!(f, "%{index} = {text} : {}", node.shape)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum GraphError {
    UnknownNode(NodeId),
    DuplicateInput(String),
    Broadcast(BroadcastError),
    MatMulShape {
        left: Shape,
        right: Shape,
    },
    /// Reduction of a rank-0 shape, of an axis outside `-rank..rank`, or of an axis whose
    /// size is 0. `axis` is the index the caller passed, before resolving a negative one.
    BadReduce {
        shape: Shape,
        axis: isize,
    },
}

/// Shape of reducing `shape` over `axis`, and that axis counted from the start.
///
/// `axis` is counted from the end when it is negative: `-1` is the last axis. The axis
/// is kept with size 1, so a row reduction still broadcasts against the operand. A
/// rank-0 shape, an axis outside `-rank..rank`, or an axis of size 0 is [`GraphError::BadReduce`].
pub fn reduced_shape(shape: &Shape, axis: isize) -> Result<(Shape, usize), GraphError> {
    let bad = || GraphError::BadReduce {
        shape: shape.clone(),
        axis,
    };
    let rank = shape.rank();
    if rank == 0 {
        return Err(bad());
    }
    let rank = rank as isize;
    if axis < -rank || axis >= rank {
        return Err(bad());
    }
    let resolved =
        usize::try_from(if axis < 0 { axis + rank } else { axis }).expect("non-negative");
    if shape.dims()[resolved] == 0 {
        return Err(bad());
    }
    let mut dims = shape.dims().to_vec();
    dims[resolved] = 1;
    Ok((Shape::new(dims), resolved))
}

/// The requested axis is the last one, including the spelling `-1` on a rank-0 shape.
fn is_last_axis(shape: &Shape, axis: isize) -> bool {
    axis == -1
        || shape
            .rank()
            .checked_sub(1)
            .is_some_and(|last| axis == last as isize)
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GraphError::UnknownNode(id) => write!(f, "unknown node %{}", id.index()),
            GraphError::DuplicateInput(name) => write!(f, "duplicate input {name}"),
            GraphError::Broadcast(err) => write!(f, "{err}"),
            GraphError::MatMulShape { left, right } => {
                write!(f, "cannot matmul shapes {left} and {right}")
            }
            GraphError::BadReduce { shape, axis } => {
                if is_last_axis(shape, *axis) {
                    write!(f, "cannot reduce shape {shape} over its last axis")?;
                } else {
                    write!(f, "cannot reduce shape {shape} over axis {axis}")?;
                }
                // The reason, from the same rule as `reduced_shape`: no axes, an axis
                // outside the rank, or an empty axis.
                let rank = shape.rank();
                match rank {
                    0 => write!(f, ": a rank 0 tensor has no axes"),
                    _ if *axis < -(rank as isize) || *axis >= rank as isize => write!(
                        f,
                        ": rank {rank} allows axes {}..={}",
                        -(rank as isize),
                        rank - 1
                    ),
                    _ => write!(f, ": that axis has size 0"),
                }
            }
        }
    }
}

impl std::error::Error for GraphError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(dims: &[usize]) -> Shape {
        Shape::new(dims.to_vec())
    }

    /// relu(A + bias) max 0.5, six nodes.
    fn sample_graph() -> Graph {
        let mut g = Graph::new();
        let a = g.input("A", shape(&[2, 3])).unwrap();
        let bias = g.input("bias", shape(&[3])).unwrap();
        let sum = g.binary(BinaryOp::Add, a, bias).unwrap();
        let act = g.unary(UnaryOp::Relu, sum).unwrap();
        let half = g.constant(0.5);
        g.binary(BinaryOp::Max, act, half).unwrap();
        g
    }

    #[test]
    fn sample_graph_prints_as_documented() {
        assert_eq!(
            sample_graph().to_string(),
            "%0 = input A : (2, 3)\n\
             %1 = input bias : (3,)\n\
             %2 = add %0, %1 : (2, 3)\n\
             %3 = relu %2 : (2, 3)\n\
             %4 = const 0.5 : ()\n\
             %5 = max %3, %4 : (2, 3)\n"
        );
    }

    #[test]
    fn sample_graph_node_shapes() {
        let shapes: Vec<Shape> = sample_graph()
            .nodes()
            .iter()
            .map(|node| node.shape.clone())
            .collect();
        assert_eq!(
            shapes,
            vec![
                shape(&[2, 3]),
                shape(&[3]),
                shape(&[2, 3]),
                shape(&[2, 3]),
                Shape::scalar(),
                shape(&[2, 3]),
            ]
        );
    }

    #[test]
    fn binary_shape_is_broadcast_of_both_sides() {
        let mut g = Graph::new();
        let a = g.input("A", shape(&[2, 1])).unwrap();
        let b = g.input("B", shape(&[3])).unwrap();
        let prod = g.binary(BinaryOp::Mul, a, b).unwrap();
        assert_eq!(g.get(prod).unwrap().shape, shape(&[2, 3]));
    }

    #[test]
    fn scalar_constant_broadcasts_in_binary() {
        let mut g = Graph::new();
        let two = g.constant(2.0);
        let a = g.input("A", shape(&[4])).unwrap();
        let sum = g.binary(BinaryOp::Add, two, a).unwrap();
        assert_eq!(g.get(sum).unwrap().shape, shape(&[4]));
    }

    #[test]
    fn broadcast_mismatch_is_an_error_and_adds_no_node() {
        let mut g = Graph::new();
        let a = g.input("A", shape(&[3])).unwrap();
        let b = g.input("B", shape(&[4])).unwrap();
        let err = g.binary(BinaryOp::Add, a, b).unwrap_err();
        assert!(matches!(err, GraphError::Broadcast(_)));
        assert_eq!(
            err.to_string(),
            "cannot broadcast shapes (3,) and (4,): 3 vs 4 at axis -1"
        );
        assert_eq!(g.nodes().len(), 2);
    }

    #[test]
    fn id_from_a_larger_graph_is_unknown() {
        let mut big = Graph::new();
        let ids: Vec<NodeId> = (0..6).map(|_| big.constant(0.0)).collect();
        let foreign = ids[5];
        assert_eq!(foreign.index(), 5);

        let mut g = Graph::new();
        let a = g.constant(1.0);
        let _ = g.constant(2.0);
        assert_eq!(
            g.unary(UnaryOp::Neg, foreign),
            Err(GraphError::UnknownNode(foreign))
        );
        let err = g.binary(BinaryOp::Add, a, foreign).unwrap_err();
        assert_eq!(err, GraphError::UnknownNode(foreign));
        assert_eq!(err.to_string(), "unknown node %5");
        assert_eq!(
            g.binary(BinaryOp::Add, foreign, a),
            Err(GraphError::UnknownNode(foreign))
        );
        assert_eq!(g.nodes().len(), 2);
    }

    #[test]
    fn duplicate_input_name_is_an_error() {
        let mut g = Graph::new();
        g.input("A", shape(&[2])).unwrap();
        let err = g.input("A", shape(&[3])).unwrap_err();
        assert_eq!(err, GraphError::DuplicateInput("A".to_string()));
        assert_eq!(err.to_string(), "duplicate input A");
        assert_eq!(g.nodes().len(), 1);
    }

    #[test]
    fn operands_refer_to_earlier_nodes() {
        let g = sample_graph();
        for (index, node) in g.nodes().iter().enumerate() {
            match &node.op {
                Op::Unary(_, x) | Op::Reduce { x, .. } => assert!(x.index() < index),
                Op::Binary(_, a, b) | Op::MatMul(a, b) => {
                    assert!(a.index() < index && b.index() < index)
                }
                Op::Input { .. } | Op::Const(_) => {}
            }
        }
    }

    #[test]
    fn exp_and_div_shapes_and_display() {
        let mut g = Graph::new();
        let a = g.input("A", shape(&[3])).unwrap();
        let b = g.input("B", shape(&[2, 1])).unwrap();
        let e = g.unary(UnaryOp::Exp, a).unwrap();
        let q = g.binary(BinaryOp::Div, a, b).unwrap();
        // exp keeps (3,); (3,) / (2, 1) broadcasts to (2, 3)
        assert_eq!(g.get(e).unwrap().shape, shape(&[3]));
        assert_eq!(g.get(q).unwrap().shape, shape(&[2, 3]));
        assert_eq!(
            g.to_string(),
            "%0 = input A : (3,)\n\
             %1 = input B : (2, 1)\n\
             %2 = exp %0 : (3,)\n\
             %3 = div %0, %1 : (2, 3)\n"
        );
    }

    #[test]
    fn empty_graph_prints_nothing() {
        assert_eq!(Graph::new().to_string(), "");
    }

    #[test]
    fn matmul_shape_is_rows_of_left_and_columns_of_right() {
        let mut g = Graph::new();
        let a = g.input("A", shape(&[2, 3])).unwrap();
        let b = g.input("B", shape(&[3, 2])).unwrap();
        let prod = g.matmul(a, b).unwrap();
        // (2,3) @ (3,2) -> (2,2)
        assert_eq!(g.get(prod).unwrap().shape, shape(&[2, 2]));
    }

    #[test]
    fn matmul_inner_dim_mismatch_is_an_error_and_adds_no_node() {
        let mut g = Graph::new();
        let a = g.input("A", shape(&[2, 3])).unwrap();
        let b = g.input("B", shape(&[4, 2])).unwrap();
        let err = g.matmul(a, b).unwrap_err();
        assert_eq!(
            err,
            GraphError::MatMulShape {
                left: shape(&[2, 3]),
                right: shape(&[4, 2]),
            }
        );
        assert_eq!(err.to_string(), "cannot matmul shapes (2, 3) and (4, 2)");
        assert_eq!(g.nodes().len(), 2);
    }

    #[test]
    fn matmul_rank_one_operand_is_an_error() {
        let mut g = Graph::new();
        let v = g.input("v", shape(&[3])).unwrap();
        let b = g.input("B", shape(&[3, 2])).unwrap();
        let err = g.matmul(v, b).unwrap_err();
        assert_eq!(
            err,
            GraphError::MatMulShape {
                left: shape(&[3]),
                right: shape(&[3, 2]),
            }
        );
        assert_eq!(err.to_string(), "cannot matmul shapes (3,) and (3, 2)");
        assert_eq!(g.nodes().len(), 2);
    }

    #[test]
    fn matmul_unknown_operand_is_reported_left_first() {
        let mut big = Graph::new();
        let ids: Vec<NodeId> = (0..6).map(|_| big.constant(0.0)).collect();
        let (foreign_a, foreign_b) = (ids[4], ids[5]);

        let mut g = Graph::new();
        let a = g.input("A", shape(&[2, 2])).unwrap();
        assert_eq!(
            g.matmul(foreign_a, a),
            Err(GraphError::UnknownNode(foreign_a))
        );
        assert_eq!(
            g.matmul(a, foreign_b),
            Err(GraphError::UnknownNode(foreign_b))
        );
        assert_eq!(
            g.matmul(foreign_a, foreign_b),
            Err(GraphError::UnknownNode(foreign_a))
        );
        assert_eq!(g.nodes().len(), 1);
    }

    #[test]
    fn reduce_keeps_the_last_axis_with_size_one() {
        let mut g = Graph::new();
        let a = g.input("A", shape(&[2, 3])).unwrap();
        let v = g.input("v", shape(&[4])).unwrap();
        let row_sum = g.reduce(ReduceOp::Sum, a).unwrap();
        let vec_max = g.reduce(ReduceOp::Max, v).unwrap();
        assert_eq!(g.get(row_sum).unwrap().shape, shape(&[2, 1]));
        assert_eq!(g.get(vec_max).unwrap().shape, shape(&[1]));
    }

    #[test]
    fn reduce_of_rank_zero_or_empty_last_axis_is_an_error_and_adds_no_node() {
        let mut g = Graph::new();
        let c = g.constant(1.0);
        let empty = g.input("E", shape(&[3, 0])).unwrap();
        let err = g.reduce(ReduceOp::Sum, c).unwrap_err();
        assert_eq!(
            err,
            GraphError::BadReduce {
                shape: Shape::scalar(),
                axis: -1,
            }
        );
        assert_eq!(err.to_string(), "cannot reduce shape () over its last axis: a rank 0 tensor has no axes");
        let err = g.reduce(ReduceOp::Max, empty).unwrap_err();
        assert_eq!(
            err,
            GraphError::BadReduce {
                shape: shape(&[3, 0]),
                axis: -1,
            }
        );
        assert_eq!(
            err.to_string(),
            "cannot reduce shape (3, 0) over its last axis: that axis has size 0"
        );
        assert_eq!(g.nodes().len(), 2);
    }

    #[test]
    fn reduce_with_an_empty_leading_axis_is_allowed() {
        let mut g = Graph::new();
        let no_rows = g.input("N", shape(&[0, 3])).unwrap();
        let reduced = g.reduce(ReduceOp::Sum, no_rows).unwrap();
        assert_eq!(g.get(reduced).unwrap().shape, shape(&[0, 1]));
    }

    #[test]
    fn reduce_unknown_operand_is_an_error() {
        let mut big = Graph::new();
        let ids: Vec<NodeId> = (0..3).map(|_| big.constant(0.0)).collect();
        let mut g = Graph::new();
        assert_eq!(
            g.reduce(ReduceOp::Sum, ids[2]),
            Err(GraphError::UnknownNode(ids[2]))
        );
    }

    #[test]
    fn reduce_prints_with_its_operand_and_shape() {
        let mut g = Graph::new();
        let a = g.input("A", shape(&[2, 3])).unwrap();
        g.reduce(ReduceOp::Sum, a).unwrap();
        g.reduce(ReduceOp::Max, a).unwrap();
        assert_eq!(
            g.to_string(),
            "%0 = input A : (2, 3)\n\
             %1 = sum %0 : (2, 1)\n\
             %2 = max %0 : (2, 1)\n"
        );
    }

    #[test]
    fn reduce_axis_keeps_that_axis_and_names_it_unless_it_is_last() {
        let mut g = Graph::new();
        let a = g.input("A", shape(&[2, 3, 4])).unwrap();
        let over_0 = g.reduce_axis(ReduceOp::Sum, a, 0).unwrap();
        let over_minus_2 = g.reduce_axis(ReduceOp::Max, a, -2).unwrap();
        let over_last = g.reduce_axis(ReduceOp::Sum, a, 2).unwrap();
        assert_eq!(g.get(over_0).unwrap().shape, shape(&[1, 3, 4]));
        assert_eq!(g.get(over_minus_2).unwrap().shape, shape(&[2, 1, 4]));
        assert_eq!(g.get(over_last).unwrap().shape, shape(&[2, 3, 1]));
        assert_eq!(
            g.to_string(),
            "%0 = input A : (2, 3, 4)\n\
             %1 = sum %0 axis 0 : (1, 3, 4)\n\
             %2 = max %0 axis 1 : (2, 1, 4)\n\
             %3 = sum %0 : (2, 3, 1)\n"
        );
        // -1 is the same node kind as omitting the axis.
        let from_negative = g.reduce_axis(ReduceOp::Sum, a, -1).unwrap();
        let from_default = g.reduce(ReduceOp::Sum, a).unwrap();
        assert_eq!(
            g.get(from_negative).unwrap().op,
            g.get(from_default).unwrap().op
        );
    }

    #[test]
    fn reduce_axis_rejects_an_out_of_range_or_empty_axis_and_adds_no_node() {
        let mut g = Graph::new();
        let a = g.input("A", shape(&[2, 3])).unwrap();
        let empty = g.input("E", shape(&[0, 3])).unwrap();
        let before = g.nodes().len();
        let err = g.reduce_axis(ReduceOp::Sum, a, 2).unwrap_err();
        assert_eq!(
            err,
            GraphError::BadReduce {
                shape: shape(&[2, 3]),
                axis: 2,
            }
        );
        assert_eq!(err.to_string(), "cannot reduce shape (2, 3) over axis 2: rank 2 allows axes -2..=1");
        let err = g.reduce_axis(ReduceOp::Max, a, -3).unwrap_err();
        assert_eq!(err.to_string(), "cannot reduce shape (2, 3) over axis -3: rank 2 allows axes -2..=1");
        // Axis 0 is empty; axis 1 is not, and an empty neighbour is allowed.
        let err = g.reduce_axis(ReduceOp::Sum, empty, 0).unwrap_err();
        assert_eq!(err.to_string(), "cannot reduce shape (0, 3) over axis 0: that axis has size 0");
        let kept = g.reduce_axis(ReduceOp::Sum, empty, 1).unwrap();
        assert_eq!(g.get(kept).unwrap().shape, shape(&[0, 1]));
        // The explicit last axis of an empty row uses the same sentence as `reduce`.
        let row = g.input("R", shape(&[2, 0])).unwrap();
        let err = g.reduce_axis(ReduceOp::Sum, row, 1).unwrap_err();
        assert_eq!(
            err.to_string(),
            "cannot reduce shape (2, 0) over its last axis: that axis has size 0"
        );
        assert_eq!(g.nodes().len(), before + 2);
    }

    #[test]
    fn matmul_prints_with_its_operands_and_shape() {
        let mut g = Graph::new();
        let a = g.input("A", shape(&[2, 3])).unwrap();
        let b = g.input("B", shape(&[3, 2])).unwrap();
        g.matmul(a, b).unwrap();
        assert_eq!(
            g.to_string(),
            "%0 = input A : (2, 3)\n\
             %1 = input B : (3, 2)\n\
             %2 = matmul %0, %1 : (2, 2)\n"
        );
    }
}
