//! Loop IR for coffee-ten: perfect loop nests with one store and affine indices.

use std::fmt;

/// `for name in 0..extent`, step 1.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Loop {
    pub name: String,
    pub extent: usize,
}

/// Row-major f32 buffer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Buffer {
    pub name: String,
    pub shape: Vec<usize>,
}

/// `constant + Σ coeff * loop_var`; terms keep their order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AffineExpr {
    pub constant: i64,
    pub terms: Vec<(String, i64)>,
}

/// `buffer[index_0, index_1, ...]`, one AffineExpr per buffer dim; `buffer` indexes `Kernel::buffers`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Access {
    pub buffer: usize,
    pub index: Vec<AffineExpr>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Relu,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Max,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Load(Access),
    Const(f32),
    Unary(UnaryOp, Box<Expr>),
    Binary(BinaryOp, Box<Expr>, Box<Expr>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Store {
    pub target: Access,
    pub value: Expr,
}

/// Perfect nest: loops outermost first, one Store in the innermost body.
#[derive(Clone, Debug, PartialEq)]
pub struct LoopNest {
    pub loops: Vec<Loop>,
    pub body: Store,
}

/// Nests run in order.
#[derive(Clone, Debug, PartialEq)]
pub struct Kernel {
    pub name: String,
    pub buffers: Vec<Buffer>,
    pub nests: Vec<LoopNest>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LoopIrError {
    EmptyExtent {
        nest: usize,
        loop_name: String,
    },
    DuplicateLoop {
        nest: usize,
        loop_name: String,
    },
    UnknownLoop {
        nest: usize,
        loop_name: String,
    },
    UnknownBuffer {
        nest: usize,
        buffer: usize,
    },
    RankMismatch {
        nest: usize,
        buffer: String,
        expected: usize,
        got: usize,
    },
    OutOfBounds {
        nest: usize,
        buffer: String,
        dim: usize,
        min: i64,
        max: i64,
        size: usize,
    },
}

impl fmt::Display for LoopIrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyExtent { nest, loop_name } => {
                write!(f, "nest {nest}: loop {loop_name} has extent 0")
            }
            Self::DuplicateLoop { nest, loop_name } => {
                write!(f, "nest {nest}: loop {loop_name} is declared twice")
            }
            Self::UnknownLoop { nest, loop_name } => {
                write!(f, "nest {nest}: unknown loop {loop_name}")
            }
            Self::UnknownBuffer { nest, buffer } => {
                write!(f, "nest {nest}: unknown buffer {buffer}")
            }
            Self::RankMismatch {
                nest,
                buffer,
                expected,
                got,
            } => {
                write!(
                    f,
                    "nest {nest}: {buffer} has rank {expected}, index has {got} entries"
                )
            }
            Self::OutOfBounds {
                nest,
                buffer,
                dim,
                min,
                max,
                size,
            } => {
                write!(
                    f,
                    "nest {nest}: {buffer}[dim {dim}] index range {min}..={max} outside 0..{size}"
                )
            }
        }
    }
}

impl std::error::Error for LoopIrError {}

impl Kernel {
    /// Checks the nests in order and returns the first error found.
    pub fn validate(&self) -> Result<(), LoopIrError> {
        for (nest_no, nest) in self.nests.iter().enumerate() {
            validate_nest(&self.buffers, nest_no, nest)?;
        }
        Ok(())
    }
}

fn validate_nest(buffers: &[Buffer], nest_no: usize, nest: &LoopNest) -> Result<(), LoopIrError> {
    let loops = &nest.loops;
    for l in loops {
        if l.extent == 0 {
            return Err(LoopIrError::EmptyExtent {
                nest: nest_no,
                loop_name: l.name.clone(),
            });
        }
    }
    for (pos, l) in loops.iter().enumerate() {
        if loops[..pos].iter().any(|earlier| earlier.name == l.name) {
            return Err(LoopIrError::DuplicateLoop {
                nest: nest_no,
                loop_name: l.name.clone(),
            });
        }
    }
    validate_access(buffers, nest_no, loops, &nest.body.target)?;
    validate_expr(buffers, nest_no, loops, &nest.body.value)
}

fn validate_expr(
    buffers: &[Buffer],
    nest_no: usize,
    loops: &[Loop],
    e: &Expr,
) -> Result<(), LoopIrError> {
    match e {
        Expr::Load(access) => validate_access(buffers, nest_no, loops, access),
        Expr::Const(_) => Ok(()),
        Expr::Unary(_, x) => validate_expr(buffers, nest_no, loops, x),
        Expr::Binary(_, a, b) => {
            validate_expr(buffers, nest_no, loops, a)?;
            validate_expr(buffers, nest_no, loops, b)
        }
    }
}

fn validate_access(
    buffers: &[Buffer],
    nest_no: usize,
    loops: &[Loop],
    access: &Access,
) -> Result<(), LoopIrError> {
    let Some(buffer) = buffers.get(access.buffer) else {
        return Err(LoopIrError::UnknownBuffer {
            nest: nest_no,
            buffer: access.buffer,
        });
    };
    if access.index.len() != buffer.shape.len() {
        return Err(LoopIrError::RankMismatch {
            nest: nest_no,
            buffer: buffer.name.clone(),
            expected: buffer.shape.len(),
            got: access.index.len(),
        });
    }
    for index in &access.index {
        for (name, _) in &index.terms {
            if !loops.iter().any(|l| &l.name == name) {
                return Err(LoopIrError::UnknownLoop {
                    nest: nest_no,
                    loop_name: name.clone(),
                });
            }
        }
    }
    for (dim, (index, &size)) in access.index.iter().zip(&buffer.shape).enumerate() {
        let (min, max) = affine_bounds(index, loops);
        let in_bounds = min >= 0 && usize::try_from(max).is_ok_and(|max_idx| max_idx < size);
        if !in_bounds {
            return Err(LoopIrError::OutOfBounds {
                nest: nest_no,
                buffer: buffer.name.clone(),
                dim,
                min,
                max,
                size,
            });
        }
    }
    Ok(())
}

/// Smallest and largest value of `expr` over all iterations of `loops`.
/// Every loop name in `expr` must exist in `loops`.
fn affine_bounds(expr: &AffineExpr, loops: &[Loop]) -> (i64, i64) {
    let mut min = expr.constant;
    let mut max = expr.constant;
    for (name, coeff) in &expr.terms {
        let extent = loops
            .iter()
            .find(|l| &l.name == name)
            .expect("invariant: terms are checked before bounds")
            .extent;
        // Saturates only for extents above i64::MAX, which no buffer can index.
        let last = i64::try_from(extent.saturating_sub(1)).unwrap_or(i64::MAX);
        let reach = coeff.saturating_mul(last);
        min = min.saturating_add(reach.min(0));
        max = max.saturating_add(reach.max(0));
    }
    (min, max)
}

impl fmt::Display for AffineExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut out = String::new();
        for (name, coeff) in &self.terms {
            if *coeff == 0 {
                continue;
            }
            let magnitude = coeff.unsigned_abs();
            let var = if magnitude == 1 {
                name.clone()
            } else {
                format!("{magnitude}*{name}")
            };
            push_signed(&mut out, *coeff < 0, &var);
        }
        if self.constant != 0 {
            push_signed(
                &mut out,
                self.constant < 0,
                &self.constant.unsigned_abs().to_string(),
            );
        }
        if out.is_empty() {
            out.push('0');
        }
        f.write_str(&out)
    }
}

/// Appends `body` with a sign: a leading `-` for the first term, ` + ` or ` - ` after that.
fn push_signed(out: &mut String, negative: bool, body: &str) {
    if out.is_empty() {
        if negative {
            out.push('-');
        }
    } else {
        out.push_str(if negative { " - " } else { " + " });
    }
    out.push_str(body);
}

impl fmt::Display for Kernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let params: Vec<String> = self
            .buffers
            .iter()
            .map(|b| format!("{}: f32[{}]", b.name, join(&b.shape, ", ")))
            .collect();
        writeln!(f, "kernel {}({})", self.name, params.join(", "))?;
        for nest in &self.nests {
            write_nest(f, &self.buffers, nest)?;
        }
        Ok(())
    }
}

fn write_nest(f: &mut fmt::Formatter<'_>, buffers: &[Buffer], nest: &LoopNest) -> fmt::Result {
    for (depth, l) in nest.loops.iter().enumerate() {
        writeln!(f, "{}for {} in 0..{}:", indent(depth), l.name, l.extent)?;
    }
    let target = access_str(buffers, &nest.body.target);
    let value = expr_str(buffers, &nest.body.value);
    writeln!(f, "{}{} = {}", indent(nest.loops.len()), target, value)
}

fn indent(depth: usize) -> String {
    "  ".repeat(depth)
}

fn join<T: ToString>(items: &[T], sep: &str) -> String {
    items
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(sep)
}

fn access_str(buffers: &[Buffer], access: &Access) -> String {
    let name = buffers
        .get(access.buffer)
        .map_or_else(|| format!("buffer{}", access.buffer), |b| b.name.clone());
    format!("{name}[{}]", join(&access.index, ", "))
}

fn expr_str(buffers: &[Buffer], e: &Expr) -> String {
    match e {
        Expr::Load(access) => access_str(buffers, access),
        Expr::Const(value) => format!("{value:?}"),
        Expr::Unary(UnaryOp::Neg, x) => format!("-({})", expr_str(buffers, x)),
        Expr::Unary(UnaryOp::Relu, x) => format!("relu({})", expr_str(buffers, x)),
        Expr::Binary(BinaryOp::Max, a, b) => {
            format!("max({}, {})", expr_str(buffers, a), expr_str(buffers, b))
        }
        Expr::Binary(BinaryOp::Add, a, b) => infix_str(buffers, "+", a, b),
        Expr::Binary(BinaryOp::Sub, a, b) => infix_str(buffers, "-", a, b),
        Expr::Binary(BinaryOp::Mul, a, b) => infix_str(buffers, "*", a, b),
    }
}

fn infix_str(buffers: &[Buffer], op: &str, a: &Expr, b: &Expr) -> String {
    format!(
        "{} {op} {}",
        operand_str(buffers, a),
        operand_str(buffers, b)
    )
}

/// An infix operand is parenthesized if it is itself an infix Add, Sub or Mul.
fn operand_str(buffers: &[Buffer], e: &Expr) -> String {
    let text = expr_str(buffers, e);
    if matches!(
        e,
        Expr::Binary(BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul, _, _)
    ) {
        format!("({text})")
    } else {
        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lp(name: &str, extent: usize) -> Loop {
        Loop {
            name: name.to_string(),
            extent,
        }
    }

    fn buf(name: &str, shape: &[usize]) -> Buffer {
        Buffer {
            name: name.to_string(),
            shape: shape.to_vec(),
        }
    }

    fn aff(constant: i64, terms: &[(&str, i64)]) -> AffineExpr {
        AffineExpr {
            constant,
            terms: terms
                .iter()
                .map(|(name, coeff)| (name.to_string(), *coeff))
                .collect(),
        }
    }

    fn var(name: &str) -> AffineExpr {
        aff(0, &[(name, 1)])
    }

    fn at(buffer: usize, index: Vec<AffineExpr>) -> Access {
        Access { buffer, index }
    }

    fn load(buffer: usize, index: Vec<AffineExpr>) -> Expr {
        Expr::Load(at(buffer, index))
    }

    fn bin(op: BinaryOp, a: Expr, b: Expr) -> Expr {
        Expr::Binary(op, Box::new(a), Box::new(b))
    }

    fn store(target: Access, value: Expr) -> Store {
        Store { target, value }
    }

    fn kernel(buffers: Vec<Buffer>, loops: Vec<Loop>, body: Store) -> Kernel {
        Kernel {
            name: "k".to_string(),
            buffers,
            nests: vec![LoopNest { loops, body }],
        }
    }

    fn matmul() -> Kernel {
        Kernel {
            name: "matmul".to_string(),
            buffers: vec![buf("A", &[4, 3]), buf("B", &[3, 2]), buf("C", &[4, 2])],
            nests: vec![
                LoopNest {
                    loops: vec![lp("i", 4), lp("j", 2)],
                    body: store(at(2, vec![var("i"), var("j")]), Expr::Const(0.0)),
                },
                LoopNest {
                    loops: vec![lp("i", 4), lp("k", 3), lp("j", 2)],
                    body: store(
                        at(2, vec![var("i"), var("j")]),
                        bin(
                            BinaryOp::Add,
                            load(2, vec![var("i"), var("j")]),
                            bin(
                                BinaryOp::Mul,
                                load(0, vec![var("i"), var("k")]),
                                load(1, vec![var("k"), var("j")]),
                            ),
                        ),
                    ),
                },
            ],
        }
    }

    #[test]
    fn matmul_prints_and_validates() {
        let k = matmul();
        assert_eq!(
            k.to_string(),
            concat!(
                "kernel matmul(A: f32[4, 3], B: f32[3, 2], C: f32[4, 2])\n",
                "for i in 0..4:\n",
                "  for j in 0..2:\n",
                "    C[i, j] = 0.0\n",
                "for i in 0..4:\n",
                "  for k in 0..3:\n",
                "    for j in 0..2:\n",
                "      C[i, j] = C[i, j] + (A[i, k] * B[k, j])\n",
            )
        );
        assert_eq!(k.validate(), Ok(()));
    }

    #[test]
    fn bias_relu_prints_and_validates() {
        let mut k = kernel(
            vec![buf("A", &[2, 3]), buf("bias", &[3]), buf("Y", &[2, 3])],
            vec![lp("i", 2), lp("j", 3)],
            store(
                at(2, vec![var("i"), var("j")]),
                Expr::Unary(
                    UnaryOp::Relu,
                    Box::new(bin(
                        BinaryOp::Add,
                        load(0, vec![var("i"), var("j")]),
                        load(1, vec![var("j")]),
                    )),
                ),
            ),
        );
        k.name = "bias_relu".to_string();
        assert_eq!(
            k.to_string(),
            concat!(
                "kernel bias_relu(A: f32[2, 3], bias: f32[3], Y: f32[2, 3])\n",
                "for i in 0..2:\n",
                "  for j in 0..3:\n",
                "    Y[i, j] = relu(A[i, j] + bias[j])\n",
            )
        );
        assert_eq!(k.validate(), Ok(()));
    }

    #[test]
    fn affine_prints_terms_and_constant() {
        assert_eq!(aff(0, &[("i", 1)]).to_string(), "i");
        assert_eq!(aff(0, &[("io", 2), ("ii", 1)]).to_string(), "2*io + ii");
        assert_eq!(aff(-1, &[("i", 1)]).to_string(), "i - 1");
        assert_eq!(aff(3, &[("i", -1)]).to_string(), "-i + 3");
        assert_eq!(aff(0, &[]).to_string(), "0");
        assert_eq!(
            aff(-2, &[("io", 4), ("ii", 1)]).to_string(),
            "4*io + ii - 2"
        );
        assert_eq!(aff(0, &[("i", 0), ("j", 1)]).to_string(), "j");
        assert_eq!(aff(-3, &[]).to_string(), "-3");
        assert_eq!(aff(3, &[]).to_string(), "3");
    }

    #[test]
    fn expr_prints_consts_and_calls() {
        let sub = bin(
            BinaryOp::Sub,
            Expr::Const(1.5),
            bin(BinaryOp::Add, Expr::Const(-0.0), Expr::Const(2.0)),
        );
        assert_eq!(expr_str(&[], &sub), "1.5 - (-0.0 + 2.0)");
        let max = bin(
            BinaryOp::Max,
            Expr::Const(0.0),
            Expr::Unary(UnaryOp::Neg, Box::new(Expr::Const(1.5))),
        );
        assert_eq!(expr_str(&[], &max), "max(0.0, -(1.5))");
    }

    #[test]
    fn error_display_is_one_line() {
        let e = LoopIrError::OutOfBounds {
            nest: 1,
            buffer: "C".to_string(),
            dim: 0,
            min: 0,
            max: 4,
            size: 4,
        };
        assert_eq!(
            e.to_string(),
            "nest 1: C[dim 0] index range 0..=4 outside 0..4"
        );
    }

    #[test]
    fn zero_extent_is_rejected() {
        let k = kernel(
            vec![buf("Y", &[4])],
            vec![lp("i", 0)],
            store(at(0, vec![var("i")]), Expr::Const(0.0)),
        );
        assert_eq!(
            k.validate(),
            Err(LoopIrError::EmptyExtent {
                nest: 0,
                loop_name: "i".to_string()
            })
        );
    }

    #[test]
    fn duplicate_loop_name_is_rejected() {
        let k = kernel(
            vec![buf("Y", &[2])],
            vec![lp("i", 2), lp("i", 2)],
            store(at(0, vec![var("i")]), Expr::Const(0.0)),
        );
        assert_eq!(
            k.validate(),
            Err(LoopIrError::DuplicateLoop {
                nest: 0,
                loop_name: "i".to_string()
            })
        );
    }

    #[test]
    fn term_naming_unknown_loop_is_rejected() {
        let k = kernel(
            vec![buf("Y", &[4])],
            vec![lp("i", 4)],
            store(at(0, vec![var("k")]), Expr::Const(0.0)),
        );
        assert_eq!(
            k.validate(),
            Err(LoopIrError::UnknownLoop {
                nest: 0,
                loop_name: "k".to_string()
            })
        );
    }

    #[test]
    fn unknown_buffer_is_rejected() {
        let k = kernel(
            vec![buf("A", &[4]), buf("B", &[4]), buf("Y", &[4])],
            vec![lp("i", 4)],
            store(at(7, vec![var("i")]), Expr::Const(0.0)),
        );
        assert_eq!(
            k.validate(),
            Err(LoopIrError::UnknownBuffer { nest: 0, buffer: 7 })
        );
    }

    #[test]
    fn rank_mismatch_is_rejected() {
        let k = kernel(
            vec![buf("Y", &[4, 3])],
            vec![lp("i", 4)],
            store(at(0, vec![var("i")]), Expr::Const(0.0)),
        );
        assert_eq!(
            k.validate(),
            Err(LoopIrError::RankMismatch {
                nest: 0,
                buffer: "Y".to_string(),
                expected: 2,
                got: 1,
            })
        );
    }

    #[test]
    fn index_past_upper_bound_is_rejected() {
        let k = kernel(
            vec![buf("A", &[4, 3]), buf("C", &[4, 3])],
            vec![lp("i", 4), lp("k", 3)],
            store(
                at(1, vec![var("i"), var("k")]),
                load(0, vec![aff(1, &[("i", 1)]), var("k")]),
            ),
        );
        assert_eq!(
            k.validate(),
            Err(LoopIrError::OutOfBounds {
                nest: 0,
                buffer: "A".to_string(),
                dim: 0,
                min: 1,
                max: 4,
                size: 4,
            })
        );
    }

    #[test]
    fn index_below_zero_is_rejected() {
        let k = kernel(
            vec![buf("A", &[4, 3]), buf("C", &[4, 3])],
            vec![lp("i", 4), lp("k", 3)],
            store(
                at(1, vec![var("i"), var("k")]),
                load(0, vec![aff(-1, &[("i", 1)]), var("k")]),
            ),
        );
        assert_eq!(
            k.validate(),
            Err(LoopIrError::OutOfBounds {
                nest: 0,
                buffer: "A".to_string(),
                dim: 0,
                min: -1,
                max: 2,
                size: 4,
            })
        );
    }
}
