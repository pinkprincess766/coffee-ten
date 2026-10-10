//! Python bindings for coffee-ten: build an expression, run it on NumPy arrays.

use std::collections::HashMap;
use std::fmt::Display;
use std::sync::Arc;

use coffee_ten_interp::{Tensor as DenseTensor, run_f32};
use coffee_ten_ir::{BinaryOp, Graph, GraphError, NodeId, Shape, UnaryOp};
use numpy::{
    PyArray1, PyArrayDescrMethods, PyArrayDyn, PyArrayMethods, PyUntypedArray,
    PyUntypedArrayMethods, ndarray,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

/// Deepest expression a Tensor may have. The tree is walked recursively when it is
/// lowered and dropped, so the depth is bounded instead of the stack.
const MAX_DEPTH: usize = 1000;

enum Expr {
    Input { name: String, shape: Shape },
    Const(f32),
    Unary(UnaryOp, Arc<Expr>),
    Binary(BinaryOp, Arc<Expr>, Arc<Expr>),
    MatMul(Arc<Expr>, Arc<Expr>),
}

/// A lazily evaluated tensor expression; it computes nothing until `run`.
#[pyclass(name = "Tensor", module = "coffee_ten", frozen, skip_from_py_object)]
#[derive(Clone)]
struct PyTensor {
    expr: Arc<Expr>,
    shape: Shape,
    depth: usize,
}

impl PyTensor {
    fn constant(value: f32) -> PyTensor {
        PyTensor {
            expr: Arc::new(Expr::Const(value)),
            shape: Shape::scalar(),
            depth: 1,
        }
    }

    fn unary(op: UnaryOp, x: &PyTensor) -> PyResult<PyTensor> {
        let depth = check_depth(x.depth + 1)?;
        Ok(PyTensor {
            expr: Arc::new(Expr::Unary(op, Arc::clone(&x.expr))),
            shape: x.shape.clone(),
            depth,
        })
    }

    fn binary(op: BinaryOp, a: &PyTensor, b: &PyTensor) -> PyResult<PyTensor> {
        let shape = a.shape.broadcast(&b.shape).map_err(value_error)?;
        let depth = check_depth(a.depth.max(b.depth) + 1)?;
        let expr = Expr::Binary(op, Arc::clone(&a.expr), Arc::clone(&b.expr));
        Ok(PyTensor {
            expr: Arc::new(expr),
            shape,
            depth,
        })
    }

    fn matmul(a: &PyTensor, b: &PyTensor) -> PyResult<PyTensor> {
        // Same rule and message as `Graph::matmul`, so the error shows at op time.
        let shape = match (a.shape.dims(), b.shape.dims()) {
            ([m, k], [k_right, n]) if k == k_right => Shape::new(vec![*m, *n]),
            _ => {
                return Err(value_error(GraphError::MatMulShape {
                    left: a.shape.clone(),
                    right: b.shape.clone(),
                }));
            }
        };
        let depth = check_depth(a.depth.max(b.depth) + 1)?;
        let expr = Expr::MatMul(Arc::clone(&a.expr), Arc::clone(&b.expr));
        Ok(PyTensor {
            expr: Arc::new(expr),
            shape,
            depth,
        })
    }
}

fn check_depth(depth: usize) -> PyResult<usize> {
    if depth > MAX_DEPTH {
        return Err(PyValueError::new_err(format!(
            "expression is nested more than {MAX_DEPTH} levels deep"
        )));
    }
    Ok(depth)
}

fn value_error(err: impl Display) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// Either side of an operator: a Tensor or a Python number (an f32 constant).
#[derive(FromPyObject)]
enum Operand<'py> {
    Tensor(PyRef<'py, PyTensor>),
    Scalar(f64),
}

impl Operand<'_> {
    fn into_tensor(self) -> PyTensor {
        match self {
            Operand::Tensor(tensor) => (*tensor).clone(),
            Operand::Scalar(value) => PyTensor::constant(value as f32),
        }
    }
}

#[pymethods]
impl PyTensor {
    /// Shape inferred when the op was created, as a tuple of ints.
    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.shape.dims().iter().copied())
    }

    // Makes NumPy return NotImplemented, so `ndarray + Tensor` is a TypeError
    // instead of an object array of Tensors.
    #[classattr]
    #[pyo3(name = "__array_ufunc__")]
    fn array_ufunc() -> Option<Py<PyAny>> {
        None
    }

    fn __neg__(&self) -> PyResult<PyTensor> {
        PyTensor::unary(UnaryOp::Neg, self)
    }

    fn __add__(&self, other: Operand<'_>) -> PyResult<PyTensor> {
        PyTensor::binary(BinaryOp::Add, self, &other.into_tensor())
    }

    fn __radd__(&self, other: Operand<'_>) -> PyResult<PyTensor> {
        PyTensor::binary(BinaryOp::Add, &other.into_tensor(), self)
    }

    fn __sub__(&self, other: Operand<'_>) -> PyResult<PyTensor> {
        PyTensor::binary(BinaryOp::Sub, self, &other.into_tensor())
    }

    fn __rsub__(&self, other: Operand<'_>) -> PyResult<PyTensor> {
        PyTensor::binary(BinaryOp::Sub, &other.into_tensor(), self)
    }

    fn __mul__(&self, other: Operand<'_>) -> PyResult<PyTensor> {
        PyTensor::binary(BinaryOp::Mul, self, &other.into_tensor())
    }

    fn __rmul__(&self, other: Operand<'_>) -> PyResult<PyTensor> {
        PyTensor::binary(BinaryOp::Mul, &other.into_tensor(), self)
    }

    // Tensor @ Tensor only: any other operand makes pyo3 return NotImplemented.
    fn __matmul__(&self, other: PyRef<'_, PyTensor>) -> PyResult<PyTensor> {
        PyTensor::matmul(self, &other)
    }
}

/// Declares a named input of the given static shape (a tuple of ints >= 0).
#[pyfunction]
fn input(name: String, shape: &Bound<'_, PyTuple>) -> PyResult<PyTensor> {
    let dims = shape
        .iter()
        .map(|dim| {
            let dim: i64 = dim.extract()?;
            usize::try_from(dim)
                .map_err(|_| PyValueError::new_err(format!("negative dimension {dim} in shape")))
        })
        .collect::<PyResult<Vec<usize>>>()?;
    let shape = Shape::new(dims);
    let expr = Expr::Input {
        name,
        shape: shape.clone(),
    };
    Ok(PyTensor {
        expr: Arc::new(expr),
        shape,
        depth: 1,
    })
}

/// relu(x) = x if x > 0 else 0, elementwise.
///
/// Differs from NumPy on NaN: relu(nan) is 0.0, while `np.maximum(nan, 0)` is nan.
#[pyfunction]
fn relu(x: &PyTensor) -> PyResult<PyTensor> {
    PyTensor::unary(UnaryOp::Relu, x)
}

/// maximum(a, b) = a if a > b else b, elementwise, with broadcasting.
///
/// Differs from NumPy on NaN: if either argument is nan the result is b, while
/// `np.maximum` propagates nan from either side. Each argument is a Tensor or a number.
#[pyfunction]
fn maximum(a: Operand<'_>, b: Operand<'_>) -> PyResult<PyTensor> {
    PyTensor::binary(BinaryOp::Max, &a.into_tensor(), &b.into_tensor())
}

/// Matrix product `(M, K) @ (K, N) -> (M, N)`; both arguments must be rank 2 Tensors.
#[pyfunction]
fn matmul(a: &PyTensor, b: &PyTensor) -> PyResult<PyTensor> {
    PyTensor::matmul(a, b)
}

/// Evaluates `output` with the reference interpreter in float32.
///
/// Every input is passed as a keyword argument: a float32 `numpy.ndarray` of the
/// shape declared by `input`. Returns a new float32 `numpy.ndarray`.
#[pyfunction]
#[pyo3(signature = (output, /, **inputs))]
fn run<'py>(
    py: Python<'py>,
    output: &PyTensor,
    inputs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let mut graph = Graph::new();
    let output_id = lower(&output.expr, &mut graph, &mut HashMap::new()).map_err(value_error)?;

    let mut arrays: Vec<(String, DenseTensor<f32>)> = Vec::new();
    if let Some(inputs) = inputs {
        for (key, value) in inputs.iter() {
            let name: String = key.extract()?;
            let array = read_input(&name, &value)?;
            arrays.push((name, array));
        }
    }
    let named_arrays: Vec<(&str, &DenseTensor<f32>)> = arrays
        .iter()
        .map(|(name, array)| (name.as_str(), array))
        .collect();

    let result = run_f32(&graph, output_id, &named_arrays).map_err(value_error)?;
    PyArray1::from_slice(py, result.data()).reshape(result.shape().dims().to_vec())
}

/// Builds the graph node of `expr`; a subtree shared by `Arc` becomes one node.
fn lower(
    expr: &Arc<Expr>,
    graph: &mut Graph,
    memo: &mut HashMap<*const Expr, NodeId>,
) -> Result<NodeId, GraphError> {
    let key = Arc::as_ptr(expr);
    if let Some(&id) = memo.get(&key) {
        return Ok(id);
    }
    let id = match &**expr {
        Expr::Input { name, shape } => graph.input(name, shape.clone())?,
        Expr::Const(value) => graph.constant(*value),
        Expr::Unary(op, x) => {
            let x = lower(x, graph, memo)?;
            graph.unary(*op, x)?
        }
        Expr::Binary(op, a, b) => {
            let a = lower(a, graph, memo)?;
            let b = lower(b, graph, memo)?;
            graph.binary(*op, a, b)?
        }
        Expr::MatMul(a, b) => {
            let a = lower(a, graph, memo)?;
            let b = lower(b, graph, memo)?;
            graph.matmul(a, b)?
        }
    };
    memo.insert(key, id);
    Ok(id)
}

/// Copies a float32 ndarray into row-major order, whatever its memory layout.
fn read_input(name: &str, value: &Bound<'_, PyAny>) -> PyResult<DenseTensor<f32>> {
    let Ok(untyped) = value.cast::<PyUntypedArray>() else {
        let type_name = value.get_type().qualname()?;
        return Err(PyTypeError::new_err(format!(
            "input {name} must be a numpy.ndarray of dtype float32, got {type_name}"
        )));
    };
    let dtype = untyped.dtype();
    if !dtype.is_equiv_to(&numpy::dtype::<f32>(value.py())) {
        return Err(PyTypeError::new_err(format!(
            "input {name} must have dtype float32, got {dtype}"
        )));
    }
    let array = value.cast::<PyArrayDyn<f32>>()?.try_readonly()?;
    let view: ndarray::ArrayViewD<'_, f32> = array.as_array();
    let data: Vec<f32> = view.iter().copied().collect();
    DenseTensor::new(Shape::new(view.shape().to_vec()), data).map_err(value_error)
}

/// coffee-ten: declare inputs, write an expression, run it with NumPy arrays.
#[pymodule]
mod coffee_ten {
    #[pymodule_export]
    use super::{PyTensor, input, matmul, maximum, relu, run};
}
