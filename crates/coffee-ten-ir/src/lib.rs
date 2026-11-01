//! Graph IR and shapes for coffee-ten.

pub mod graph;
pub mod shape;

pub use graph::{BinaryOp, Graph, GraphError, Node, NodeId, Op, ReduceOp, UnaryOp, reduced_shape};
pub use shape::{BroadcastError, Shape};
