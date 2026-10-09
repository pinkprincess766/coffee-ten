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
}

impl UnaryOp {
    fn name(self) -> &'static str {
        match self {
            UnaryOp::Neg => "neg",
            UnaryOp::Relu => "relu",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Max,
}

impl BinaryOp {
    fn name(self) -> &'static str {
        match self {
            BinaryOp::Add => "add",
            BinaryOp::Sub => "sub",
            BinaryOp::Mul => "mul",
            BinaryOp::Max => "max",
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
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GraphError::UnknownNode(id) => write!(f, "unknown node %{}", id.index()),
            GraphError::DuplicateInput(name) => write!(f, "duplicate input {name}"),
            GraphError::Broadcast(err) => write!(f, "{err}"),
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
                Op::Unary(_, x) => assert!(x.index() < index),
                Op::Binary(_, a, b) => assert!(a.index() < index && b.index() < index),
                Op::Input { .. } | Op::Const(_) => {}
            }
        }
    }

    #[test]
    fn empty_graph_prints_nothing() {
        assert_eq!(Graph::new().to_string(), "");
    }
}
