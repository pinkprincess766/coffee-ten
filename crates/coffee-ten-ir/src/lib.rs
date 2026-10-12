//! Graph IR and shapes for coffee-ten.

pub mod graph;
pub mod shape;

pub use graph::{BinaryOp, Graph, GraphError, Node, NodeId, Op, ReduceOp, UnaryOp};
pub use shape::{BroadcastError, Shape};
