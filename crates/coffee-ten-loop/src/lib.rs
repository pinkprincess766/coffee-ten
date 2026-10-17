//! Loop IR for coffee-ten: perfect loop nests with one store and affine indices.

pub mod schedule;
pub use schedule::{Schedule, ScheduleParseError, Step};
pub mod lower;
pub use lower::{LowerError, MAX_EXPR_NODES, lower, lower_elementwise};
pub mod apply;
pub use apply::{OrderClass, ScheduleError, apply, apply_classified};
pub mod parse;
pub use parse::{KernelParseError, parse_kernel};

use std::collections::BTreeMap;
use std::fmt;

/// `for name in 0..extent`, step 1. A `vectorize` loop asks the backend to vectorize it;
/// per nest at most one loop has it and it is the innermost (see `Kernel::validate`).
/// An `unroll` loop asks the backend to unroll it fully; any loop may have it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Loop {
    pub name: String,
    pub extent: usize,
    pub vectorize: bool,
    pub unroll: bool,
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
    Exp,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
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

/// The body runs only when `expr < bound`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Guard {
    pub expr: AffineExpr,
    pub bound: i64,
}

/// Perfect nest: loops outermost first, guards (all must hold), one Store in the innermost body.
#[derive(Clone, Debug, PartialEq)]
pub struct LoopNest {
    pub loops: Vec<Loop>,
    pub guards: Vec<Guard>,
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
    /// More than one loop of the nest is vectorized, or the vectorized loop is not the innermost.
    BadVectorize {
        nest: usize,
    },
    /// A guard can never hold: its smallest value is already `>= bound`.
    EmptyGuard {
        nest: usize,
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
            Self::BadVectorize { nest } => {
                write!(f, "nest {nest}: only the innermost loop may be vectorized")
            }
            Self::EmptyGuard { nest } => write!(f, "nest {nest}: a guard never holds"),
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
    let last = loops.len().saturating_sub(1);
    let is_vectorize_ok = loops
        .iter()
        .enumerate()
        .all(|(pos, l)| !l.vectorize || pos == last);
    if !is_vectorize_ok {
        return Err(LoopIrError::BadVectorize { nest: nest_no });
    }
    for guard in &nest.guards {
        check_terms(nest_no, loops, &guard.expr)?;
        if affine_bounds(&guard.expr, loops).0 >= guard.bound {
            return Err(LoopIrError::EmptyGuard { nest: nest_no });
        }
    }
    let guards = &nest.guards;
    validate_access(buffers, nest_no, loops, guards, &nest.body.target)?;
    validate_expr(buffers, nest_no, loops, guards, &nest.body.value)
}

fn validate_expr(
    buffers: &[Buffer],
    nest_no: usize,
    loops: &[Loop],
    guards: &[Guard],
    e: &Expr,
) -> Result<(), LoopIrError> {
    match e {
        Expr::Load(access) => validate_access(buffers, nest_no, loops, guards, access),
        Expr::Const(_) => Ok(()),
        Expr::Unary(_, x) => validate_expr(buffers, nest_no, loops, guards, x),
        Expr::Binary(_, a, b) => {
            validate_expr(buffers, nest_no, loops, guards, a)?;
            validate_expr(buffers, nest_no, loops, guards, b)
        }
    }
}

fn validate_access(
    buffers: &[Buffer],
    nest_no: usize,
    loops: &[Loop],
    guards: &[Guard],
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
        check_terms(nest_no, loops, index)?;
    }
    for (dim, (index, &size)) in access.index.iter().zip(&buffer.shape).enumerate() {
        let (min, max) = affine_bounds(index, loops);
        let max = guarded_max(index, max, guards, loops);
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

/// Every term of `expr` must name a loop.
fn check_terms(nest_no: usize, loops: &[Loop], expr: &AffineExpr) -> Result<(), LoopIrError> {
    for (name, _) in &expr.terms {
        if !loops.iter().any(|l| &l.name == name) {
            return Err(LoopIrError::UnknownLoop {
                nest: nest_no,
                loop_name: name.clone(),
            });
        }
    }
    Ok(())
}

/// Tightens `max`, the plain upper bound of `index`, by the guards.
///
/// Take a guard `G + gc < B`, where `G` is the sum of its terms and `gc` its constant, so
/// `G <= B - 1 - gc`. Suppose an integer `c >= 1` exists such that every term of `G`,
/// scaled by `c`, occurs in `index` with exactly that coefficient (zero-coefficient terms
/// are ignored, repeated names are summed). Then `index = c*G + R + ec`, where `R` are the
/// remaining terms of `index` and `ec` its constant, and
///
/// `index <= c*(B - 1 - gc) + max(R) + ec`,
///
/// with `max(R)` the plain affine bound of `R`. `c` is fixed by the first guard term:
/// `coeff_in_index / coeff_in_guard` must be a positive integer, the same for all terms.
/// The result is the smallest such bound over all guards. `c = 1` with empty `R` is the
/// split case, the index being exactly the guarded expression; `c > 1` arises when a split
/// loop is split again, e.g. `6*sso + 3*ssi + si = 3*(2*sso + ssi) + si` under
/// `2*sso + ssi < 3`.
///
/// Checked arithmetic: a guard whose bound overflows i64 is skipped, so an overflow never
/// makes the result smaller than the true maximum. This is the only reasoning about guards.
fn guarded_max(index: &AffineExpr, max: i64, guards: &[Guard], loops: &[Loop]) -> i64 {
    let Some(index_terms) = merged_terms(index) else {
        return max;
    };
    guards
        .iter()
        .filter_map(|g| guard_bound(index, &index_terms, g, loops))
        .fold(max, i64::min)
}

/// The bound `c*(B - 1 - gc) + max(R) + ec` of `guarded_max` for one guard, if `c` exists.
fn guard_bound(
    index: &AffineExpr,
    index_terms: &BTreeMap<&str, i64>,
    guard: &Guard,
    loops: &[Loop],
) -> Option<i64> {
    let guard_terms = merged_terms(&guard.expr)?;
    let (first_name, first_coeff) = guard_terms.iter().next()?;
    let c = index_terms.get(first_name)?.checked_div(*first_coeff)?;
    if c < 1 {
        return None;
    }
    for (name, coeff) in &guard_terms {
        let in_index = *index_terms.get(name)?;
        if coeff.checked_mul(c) != Some(in_index) {
            return None;
        }
    }
    let rest = AffineExpr {
        constant: 0,
        terms: index_terms
            .iter()
            .filter(|(name, _)| !guard_terms.contains_key(*name))
            .map(|(name, coeff)| (name.to_string(), *coeff))
            .collect(),
    };
    let rest_max = affine_bounds(&rest, loops).1;
    if rest_max == i64::MAX {
        // `affine_bounds` saturated: the true maximum may be larger.
        return None;
    }
    let slack = guard
        .bound
        .checked_sub(1)?
        .checked_sub(guard.expr.constant)?;
    c.checked_mul(slack)?
        .checked_add(rest_max)?
        .checked_add(index.constant)
}

/// Non-zero coefficient per loop name, repeated names summed; None on overflow.
fn merged_terms(expr: &AffineExpr) -> Option<BTreeMap<&str, i64>> {
    let mut merged = BTreeMap::new();
    for (name, coeff) in &expr.terms {
        let sum: &mut i64 = merged.entry(name.as_str()).or_insert(0);
        *sum = sum.checked_add(*coeff)?;
    }
    merged.retain(|_, coeff| *coeff != 0);
    Some(merged)
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
        let mark = match (l.vectorize, l.unroll) {
            (false, false) => "",
            (true, false) => " (vectorize)",
            (false, true) => " (unroll)",
            (true, true) => " (vectorize, unroll)",
        };
        writeln!(
            f,
            "{}for {} in 0..{}{mark}:",
            indent(depth),
            l.name,
            l.extent
        )?;
    }
    let mut depth = nest.loops.len();
    for guard in &nest.guards {
        writeln!(f, "{}if {} < {}:", indent(depth), guard.expr, guard.bound)?;
        depth += 1;
    }
    let target = access_str(buffers, &nest.body.target);
    let value = expr_str(buffers, &nest.body.value);
    writeln!(f, "{}{} = {}", indent(depth), target, value)
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
        Expr::Unary(UnaryOp::Exp, x) => format!("exp({})", expr_str(buffers, x)),
        Expr::Binary(BinaryOp::Max, a, b) => {
            format!("max({}, {})", expr_str(buffers, a), expr_str(buffers, b))
        }
        Expr::Binary(BinaryOp::Add, a, b) => infix_str(buffers, "+", a, b),
        Expr::Binary(BinaryOp::Sub, a, b) => infix_str(buffers, "-", a, b),
        Expr::Binary(BinaryOp::Mul, a, b) => infix_str(buffers, "*", a, b),
        Expr::Binary(BinaryOp::Div, a, b) => infix_str(buffers, "/", a, b),
    }
}

fn infix_str(buffers: &[Buffer], op: &str, a: &Expr, b: &Expr) -> String {
    format!(
        "{} {op} {}",
        operand_str(buffers, a),
        operand_str(buffers, b)
    )
}

/// An infix operand is parenthesized if it is itself an infix Add, Sub, Mul or Div.
fn operand_str(buffers: &[Buffer], e: &Expr) -> String {
    let text = expr_str(buffers, e);
    if matches!(
        e,
        Expr::Binary(
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div,
            _,
            _
        )
    ) {
        format!("({text})")
    } else {
        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn lp(name: &str, extent: usize) -> Loop {
        Loop {
            name: name.to_string(),
            extent,
            vectorize: false,
            unroll: false,
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
            nests: vec![LoopNest {
                loops,
                guards: Vec::new(),
                body,
            }],
        }
    }

    fn matmul() -> Kernel {
        Kernel {
            name: "matmul".to_string(),
            buffers: vec![buf("A", &[4, 3]), buf("B", &[3, 2]), buf("C", &[4, 2])],
            nests: vec![
                LoopNest {
                    loops: vec![lp("i", 4), lp("j", 2)],
                    guards: Vec::new(),
                    body: store(at(2, vec![var("i"), var("j")]), Expr::Const(0.0)),
                },
                LoopNest {
                    loops: vec![lp("i", 4), lp("k", 3), lp("j", 2)],
                    guards: Vec::new(),
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
    fn vectorized_innermost_loop_prints_and_validates() {
        let mut k = matmul();
        k.nests[1].loops[2].vectorize = true;
        assert!(k.to_string().contains("    for j in 0..2 (vectorize):\n"));
        assert_eq!(k.to_string().matches("(vectorize)").count(), 1);
        assert_eq!(k.validate(), Ok(()));
    }

    #[test]
    fn unrolled_loops_print_and_validate_at_any_position() {
        let mut k = matmul();
        k.nests[1].loops[0].unroll = true;
        k.nests[1].loops[2].unroll = true;
        k.nests[1].loops[2].vectorize = true;
        let text = k.to_string();
        assert!(text.contains("\nfor i in 0..4 (unroll):\n"));
        assert!(text.contains("    for j in 0..2 (vectorize, unroll):\n"));
        assert_eq!(k.validate(), Ok(()));
    }

    #[test]
    fn vectorized_loop_that_is_not_innermost_is_rejected() {
        let bad = Err(LoopIrError::BadVectorize { nest: 1 });
        for marked in [
            [true, false, false],
            [false, true, false],
            [true, false, true],
        ] {
            let mut k = matmul();
            for (l, mark) in k.nests[1].loops.iter_mut().zip(marked) {
                l.vectorize = mark;
            }
            assert_eq!(k.validate(), bad);
        }
        assert_eq!(
            LoopIrError::BadVectorize { nest: 1 }.to_string(),
            "nest 1: only the innermost loop may be vectorized"
        );
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
    fn expr_prints_exp_and_div() {
        let a = || load(0, vec![var("i")]);
        let b = || load(1, vec![var("i")]);
        let buffers = [buf("A", &[4]), buf("B", &[4])];
        let quotient = bin(BinaryOp::Div, Expr::Unary(UnaryOp::Exp, Box::new(a())), b());
        assert_eq!(expr_str(&buffers, &quotient), "exp(A[i]) / B[i]");
        // an infix operand of Div is parenthesized, also on the right
        let nested = bin(BinaryOp::Div, a(), bin(BinaryOp::Div, a(), b()));
        assert_eq!(expr_str(&buffers, &nested), "A[i] / (A[i] / B[i])");
        let mixed = bin(BinaryOp::Sub, bin(BinaryOp::Div, a(), b()), b());
        assert_eq!(expr_str(&buffers, &mixed), "(A[i] / B[i]) - B[i]");
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

    fn guard(expr: AffineExpr, bound: i64) -> Guard {
        Guard { expr, bound }
    }

    /// `Y[io*2 + ii + offset] = 0.0` under the guard `2*io + ii < 3`, for `Y: f32[y_len]`.
    fn guarded_store(y_len: usize, offset: i64) -> Kernel {
        let mut k = kernel(
            vec![buf("Y", &[y_len])],
            vec![lp("io", 2), lp("ii", 2)],
            store(
                at(0, vec![aff(offset, &[("io", 2), ("ii", 1)])]),
                Expr::Const(0.0),
            ),
        );
        k.nests[0].guards = vec![guard(aff(0, &[("io", 2), ("ii", 1)]), 3)];
        k
    }

    #[test]
    fn guard_prints_between_loops_and_store() {
        let k = guarded_store(3, 0);
        assert_eq!(
            k.to_string(),
            concat!(
                "kernel k(Y: f32[3])\n",
                "for io in 0..2:\n",
                "  for ii in 0..2:\n",
                "    if 2*io + ii < 3:\n",
                "      Y[2*io + ii] = 0.0\n",
            )
        );
        assert_eq!(k.validate(), Ok(()));
    }

    #[test]
    fn several_guards_print_nested() {
        let mut k = guarded_store(3, 0);
        k.nests[0].guards.push(guard(var("ii"), 1));
        assert_eq!(
            k.to_string(),
            concat!(
                "kernel k(Y: f32[3])\n",
                "for io in 0..2:\n",
                "  for ii in 0..2:\n",
                "    if 2*io + ii < 3:\n",
                "      if ii < 1:\n",
                "        Y[2*io + ii] = 0.0\n",
            )
        );
    }

    #[test]
    fn guard_clamps_the_index_with_the_same_terms() {
        // Without the guard the index reaches 3, outside Y: f32[3].
        let mut k = guarded_store(3, 0);
        k.nests[0].guards.clear();
        assert_eq!(
            k.validate(),
            Err(LoopIrError::OutOfBounds {
                nest: 0,
                buffer: "Y".to_string(),
                dim: 0,
                min: 0,
                max: 3,
                size: 3,
            })
        );
        // Guarded: max = 3 - 1 = 2 fits Y: f32[3]; a Y: f32[2] is still too small.
        assert_eq!(guarded_store(3, 0).validate(), Ok(()));
        assert_eq!(
            guarded_store(2, 0).validate(),
            Err(LoopIrError::OutOfBounds {
                nest: 0,
                buffer: "Y".to_string(),
                dim: 0,
                min: 0,
                max: 2,
                size: 2,
            })
        );
    }

    #[test]
    fn guard_clamp_follows_the_constant_offset_and_ignores_term_order() {
        // Index 2*io + ii + 1 <= 2 + 1 = 3 fits Y: f32[4] but not Y: f32[3].
        assert_eq!(guarded_store(4, 1).validate(), Ok(()));
        assert!(guarded_store(3, 1).validate().is_err());
        let mut k = guarded_store(3, 0);
        k.nests[0].body.target.index = vec![aff(0, &[("ii", 1), ("io", 2)])];
        assert_eq!(k.validate(), Ok(()));
    }

    #[test]
    fn guard_does_not_clamp_an_index_with_other_terms() {
        let mut k = guarded_store(3, 0);
        k.nests[0].body.target.index = vec![aff(0, &[("io", 1), ("ii", 1)])];
        k.buffers[0].shape = vec![2];
        // Index io + ii reaches 2: the guard 2*io + ii < 3 is not about it.
        assert_eq!(
            k.validate(),
            Err(LoopIrError::OutOfBounds {
                nest: 0,
                buffer: "Y".to_string(),
                dim: 0,
                min: 0,
                max: 2,
                size: 2,
            })
        );
    }

    #[test]
    fn guard_naming_unknown_loop_is_rejected() {
        let mut k = guarded_store(3, 0);
        k.nests[0].guards = vec![guard(var("k"), 3)];
        assert_eq!(
            k.validate(),
            Err(LoopIrError::UnknownLoop {
                nest: 0,
                loop_name: "k".to_string()
            })
        );
    }

    #[test]
    fn guard_that_never_holds_is_rejected() {
        // `ii + 3 < 3` never holds, `ii + 2 < 3` holds for ii = 0.
        let mut k = guarded_store(4, 0);
        k.nests[0].guards = vec![guard(aff(3, &[("ii", 1)]), 3)];
        assert_eq!(k.validate(), Err(LoopIrError::EmptyGuard { nest: 0 }));
        k.nests[0].guards = vec![guard(aff(2, &[("ii", 1)]), 3)];
        assert_eq!(k.validate(), Ok(()));
        assert_eq!(
            LoopIrError::EmptyGuard { nest: 1 }.to_string(),
            "nest 1: a guard never holds"
        );
    }

    fn clamped(index: &AffineExpr, guards: &[Guard], loops: &[Loop]) -> i64 {
        guarded_max(index, affine_bounds(index, loops).1, guards, loops)
    }

    #[test]
    fn guard_clamps_a_multiple_of_its_terms_plus_the_rest() {
        let loops = [lp("sso", 2), lp("ssi", 2), lp("si", 3)];
        let guards = [guard(aff(0, &[("sso", 2), ("ssi", 1)]), 3)];
        // 6*sso + 3*ssi + si = 3*(2*sso + ssi) + si <= 3*2 + 2 = 8; the plain bound is 11.
        let index = aff(0, &[("sso", 6), ("ssi", 3), ("si", 1)]);
        assert_eq!(affine_bounds(&index, &loops).1, 11);
        assert_eq!(clamped(&index, &guards, &loops), 8);
        // Term order and a zero-coefficient term do not matter; the constant is added.
        let index = aff(1, &[("si", 1), ("sso", 6), ("zero", 0), ("ssi", 3)]);
        let mut loops_zero = loops.to_vec();
        loops_zero.push(lp("zero", 5));
        assert_eq!(clamped(&index, &guards, &loops_zero), 9);
        // The same index as the whole guard-free validate: Y: f32[9] fits, Y: f32[8] does not.
        let mut k = guarded_store(9, 0);
        k.nests[0].loops = loops.to_vec();
        k.nests[0].guards = guards.to_vec();
        k.nests[0].body.target.index = vec![aff(0, &[("sso", 6), ("ssi", 3), ("si", 1)])];
        assert_eq!(k.validate(), Ok(()));
        k.buffers[0].shape = vec![8];
        assert!(k.validate().is_err());
    }

    #[test]
    fn guard_does_not_clamp_a_non_multiple_of_its_terms() {
        let loops = [lp("sso", 2), lp("ssi", 2)];
        let guards = [guard(aff(0, &[("sso", 2), ("ssi", 1)]), 3)];
        // 5 is not a multiple of 2, so the plain bound 5 + 3 stays.
        let index = aff(0, &[("sso", 5), ("ssi", 3)]);
        assert_eq!(clamped(&index, &guards, &loops), 8);
        // The ratios differ: sso gives c = 3, ssi gives c = 2.
        let index = aff(0, &[("sso", 6), ("ssi", 2)]);
        assert_eq!(clamped(&index, &guards, &loops), 8);
        // A guard term missing from the index.
        let index = aff(0, &[("sso", 6)]);
        assert_eq!(clamped(&index, &guards, &loops), 6);
        // A negative ratio is not a scale by c >= 1.
        let index = aff(0, &[("sso", -2), ("ssi", -1)]);
        assert_eq!(clamped(&index, &guards, &loops), 0);
    }

    #[test]
    fn guard_clamp_with_c_one_keeps_working() {
        let loops = [lp("io", 2), lp("ii", 2), lp("j", 4)];
        let guards = [guard(aff(1, &[("io", 2), ("ii", 1)]), 3)];
        // G = 2*io + ii <= 3 - 1 - 1 = 1, so 2*io + ii + 2 <= 3.
        let index = aff(2, &[("io", 2), ("ii", 1)]);
        assert_eq!(clamped(&index, &guards, &loops), 3);
        // With remaining terms: + j <= 3 adds 3.
        let index = aff(2, &[("io", 2), ("ii", 1), ("j", 1)]);
        assert_eq!(clamped(&index, &guards, &loops), 6);
        // The smallest bound over several guards wins.
        let guards = [guards[0].clone(), guard(aff(0, &[("io", 2), ("ii", 1)]), 1)];
        assert_eq!(clamped(&index, &guards, &loops), 5);
    }

    #[test]
    fn guard_clamp_overflow_never_lowers_the_bound() {
        let loops = [lp("a", 2), lp("b", 4)];
        let big = i64::MAX / 2 + 1;
        // c = 2 and slack = i64::MAX - 1 overflow: the guard is skipped.
        let guards = [guard(var("a"), i64::MAX)];
        let index = aff(0, &[("a", 2), ("b", 1)]);
        let plain = affine_bounds(&index, &loops).1;
        assert_eq!(clamped(&index, &guards, &loops), plain);
        // A saturated plain bound of the rest is not trusted either.
        let loops = [lp("a", 2), lp("b", usize::MAX)];
        let guards = [guard(var("a"), -4)];
        // Trusting it would give -5 + MAX - big: the sum fits, but is below the true maximum.
        let index = aff(-big, &[("a", 1), ("b", big)]);
        let plain = affine_bounds(&index, &loops).1;
        assert_eq!(clamped(&index, &guards, &loops), plain);
    }

    /// Loops `l0..` with the given extents.
    fn numbered_loops(extents: &[usize]) -> Vec<Loop> {
        let named = extents.iter().enumerate();
        named
            .map(|(n, &extent)| lp(&format!("l{n}"), extent))
            .collect()
    }

    fn numbered_expr(constant: i64, coeffs: &[i64]) -> AffineExpr {
        let terms = coeffs.iter().enumerate();
        AffineExpr {
            constant,
            terms: terms.map(|(n, &c)| (format!("l{n}"), c)).collect(),
        }
    }

    fn eval(expr: &AffineExpr, values: &[i64]) -> i64 {
        let term = |(name, coeff): &(String, i64)| {
            let n: usize = name[1..].parse().expect("test names are l<n>");
            coeff * values[n]
        };
        expr.constant + expr.terms.iter().map(term).sum::<i64>()
    }

    proptest! {
        /// The clamp never under-estimates: over all iterations that pass every guard, the
        /// index stays <= the clamped max. Indices are often built as `c*G + extra`, so the
        /// clamp has something to do.
        #[test]
        fn guarded_max_never_under_estimates(
            extents in prop::collection::vec(1usize..=4, 1..=3),
            guard_coeffs in prop::collection::vec(
                (
                    prop::collection::vec(prop_oneof![2 => Just(0i64), 3 => -3i64..=3], 3),
                    -3i64..=3,
                    -2i64..=8,
                ),
                0..=2,
            ),
            scale in 1i64..=4,
            pick in any::<usize>(),
            extra in prop::collection::vec(
                prop_oneof![1 => Just(0i64), 1 => -3i64..=3],
                3,
            ),
            constant in -3i64..=3,
            derived in any::<bool>(),
        ) {
            let loops = numbered_loops(&extents);
            let n = loops.len();
            let guards: Vec<Guard> = guard_coeffs
                .iter()
                .map(|(coeffs, gc, bound)| guard(numbered_expr(*gc, &coeffs[..n]), *bound))
                .collect();
            let mut coeffs = extra[..n].to_vec();
            if derived && !guards.is_empty() {
                let source = &guards[pick % guards.len()].expr;
                for ((_, g), c) in source.terms.iter().zip(&mut coeffs) {
                    *c += scale * g;
                }
            }
            let index = numbered_expr(constant, &coeffs);

            let iterations: usize = extents.iter().product();
            let mut true_max: Option<i64> = None;
            for it in 0..iterations {
                let mut rest = it;
                let values: Vec<i64> = extents
                    .iter()
                    .map(|&e| {
                        let v = (rest % e) as i64;
                        rest /= e;
                        v
                    })
                    .collect();
                if guards.iter().all(|g| eval(&g.expr, &values) < g.bound) {
                    let v = eval(&index, &values);
                    true_max = Some(true_max.map_or(v, |m| m.max(v)));
                }
            }
            let plain = affine_bounds(&index, &loops).1;
            let bound = guarded_max(&index, plain, &guards, &loops);
            prop_assert!(bound <= plain);
            if let Some(true_max) = true_max {
                prop_assert!(bound >= true_max, "clamped {} < true max {}", bound, true_max);
            }
        }
    }
}
