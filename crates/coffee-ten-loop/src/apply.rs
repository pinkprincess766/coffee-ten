//! Applies a schedule to a kernel.

use std::fmt;

use crate::{Access, AffineExpr, BinaryOp, Expr, Guard, Kernel, Loop, LoopIrError, Schedule, Step};

/// Why `apply` refused a schedule. `step` counts from 0 in `Schedule::steps`.
#[derive(Clone, Debug, PartialEq)]
pub enum ScheduleError {
    NoSuchNest {
        step: usize,
        nest: usize,
    },
    NoSuchLoop {
        step: usize,
        nest: usize,
        loop_name: String,
    },
    /// `outer` or `inner` is already a loop of the nest.
    NameTaken {
        step: usize,
        nest: usize,
        name: String,
    },
    ZeroFactor {
        step: usize,
    },
    /// The step is valid but not applied yet.
    NotImplemented {
        step: usize,
        what: &'static str,
    },
    /// A factor, extent or scaled coefficient does not fit the loop IR's `i64`.
    NumberTooLarge {
        step: usize,
    },
    /// The `order` of a reorder is not a permutation of the loops of the nest.
    BadReorder {
        step: usize,
        nest: usize,
    },
    /// The nest reads its store target at another index than it writes, so iterations
    /// depend on each other in a way reordering would break (see `reorder`).
    IllegalReorder {
        step: usize,
        nest: usize,
    },
    /// Loop `loop_name` of the nest cannot be vectorized, for `reason` (see `vectorize`).
    IllegalVectorize {
        step: usize,
        nest: usize,
        reason: &'static str,
    },
    /// The result failed `Kernel::validate`; this is a bug in `apply` or the input kernel.
    Invalid(LoopIrError),
}

impl fmt::Display for ScheduleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoSuchNest { step, nest } => write!(f, "step {step}: no nest {nest}"),
            Self::NoSuchLoop {
                step,
                nest,
                loop_name,
            } => write!(f, "step {step}: nest {nest} has no loop {loop_name}"),
            Self::NameTaken { step, nest, name } => {
                write!(f, "step {step}: nest {nest} already has a loop {name}")
            }
            Self::ZeroFactor { step } => write!(f, "step {step}: split factor is 0"),
            Self::NotImplemented { step, what } => {
                write!(f, "step {step}: {what} is not implemented")
            }
            Self::NumberTooLarge { step } => {
                write!(f, "step {step}: a number does not fit in i64")
            }
            Self::BadReorder { step, nest } => {
                write!(
                    f,
                    "step {step}: order is not a permutation of the loops of nest {nest}"
                )
            }
            Self::IllegalReorder { step, nest } => {
                write!(
                    f,
                    "step {step}: nest {nest} reads its target at another index"
                )
            }
            Self::IllegalVectorize { step, nest, reason } => {
                write!(f, "step {step}: cannot vectorize in nest {nest}: {reason}")
            }
            Self::Invalid(e) => write!(f, "schedule produced an invalid kernel: {e}"),
        }
    }
}

impl std::error::Error for ScheduleError {}

/// How a schedule changes the order of float operations (spec §5.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrderClass {
    /// Every target element receives its contributions in the same order, so f32 results
    /// are bit for bit equal.
    Preserving,
    /// Only the multiset of contributions per target element is the same; the order of a
    /// float sum changes, so results are equal only within a tolerance.
    Reordering,
}

/// Applies the steps in order to a copy of `kernel` and validates the result. On error
/// `kernel` is untouched, as it is only borrowed. See `apply_classified` for the order class.
pub fn apply(kernel: &Kernel, schedule: &Schedule) -> Result<Kernel, ScheduleError> {
    apply_classified(kernel, schedule).map(|(result, _)| result)
}

/// Like `apply`, and also says whether the schedule is order-preserving or reordering.
/// A schedule is `Reordering` if any step is. `split` keeps the order of all iterations,
/// hence of every float operation, so it is `Preserving`; see `reorder` and `vectorize`
/// for the rest.
pub fn apply_classified(
    kernel: &Kernel,
    schedule: &Schedule,
) -> Result<(Kernel, OrderClass), ScheduleError> {
    let mut result = kernel.clone();
    let mut class = OrderClass::Preserving;
    for (step_no, step) in schedule.steps.iter().enumerate() {
        match step {
            Step::Split {
                nest,
                loop_name,
                outer,
                inner,
                factor,
            } => split(
                &mut result,
                step_no,
                *nest,
                loop_name,
                outer,
                inner,
                *factor,
            )?,
            Step::Reorder { nest, order } => {
                if reorder(&mut result, step_no, *nest, order)? == OrderClass::Reordering {
                    class = OrderClass::Reordering;
                }
            }
            Step::Vectorize { nest, loop_name } => {
                vectorize(&mut result, step_no, *nest, loop_name)?;
            }
        }
    }
    result.validate().map_err(ScheduleError::Invalid)?;
    Ok((result, class))
}

/// Marks loop `loop_name` of nest `nest_no` for the backend to vectorize. Legal iff
/// - the loop is the innermost of the nest and not vectorized yet;
/// - it is not a reduction loop: its name appears with a non-zero coefficient in the store
///   target index (vectorized reductions are out of scope);
/// - in the target index it appears only in the last dimension, with coefficient exactly 1,
///   so adjacent lanes write adjacent elements.
///
/// Class: `Preserving`. Each lane computes and writes its own element, and the backend
/// keeps every operation a separate rounding (`-ffp-contract=off`), so every float result is
/// bit for bit the same. The mark is a request: loop-carried reads of the target are not
/// checked here, the backend refuses to vectorize such a loop itself.
fn vectorize(
    kernel: &mut Kernel,
    step: usize,
    nest_no: usize,
    loop_name: &str,
) -> Result<(), ScheduleError> {
    let nest = kernel
        .nests
        .get_mut(nest_no)
        .ok_or(ScheduleError::NoSuchNest {
            step,
            nest: nest_no,
        })?;
    let pos = nest
        .loops
        .iter()
        .position(|l| l.name == loop_name)
        .ok_or_else(|| ScheduleError::NoSuchLoop {
            step,
            nest: nest_no,
            loop_name: loop_name.to_string(),
        })?;
    let illegal = |reason| ScheduleError::IllegalVectorize {
        step,
        nest: nest_no,
        reason,
    };
    if nest.loops[pos].vectorize {
        return Err(illegal("already vectorized"));
    }
    if pos + 1 != nest.loops.len() {
        return Err(illegal("not the innermost loop"));
    }
    let index = &nest.body.target.index;
    let coeff_in = |e: &AffineExpr| {
        let terms = e.terms.iter().filter(|(n, _)| n == loop_name);
        terms.fold(0i64, |sum, (_, coeff)| sum.saturating_add(*coeff))
    };
    let is_reduction = !index.iter().any(|e| {
        e.terms
            .iter()
            .any(|(n, coeff)| n == loop_name && *coeff != 0)
    });
    if is_reduction {
        return Err(illegal("reduction loop"));
    }
    let (last, others) = index.split_last().expect("invariant: loop is in the index");
    if coeff_in(last) != 1 || others.iter().any(|e| coeff_in(e) != 0) {
        return Err(illegal("target stride is not 1"));
    }
    nest.loops[pos].vectorize = true;
    Ok(())
}

/// Rearranges the loops of nest `nest_no` into `order`, which must name every loop once.
/// Extents, guards and the body stay as they are.
///
/// Legality: let `T[I]` be the store target and R the reduction loops (see below). The
/// nest is legal to reorder if
/// - the map from iteration to element `T[I]` is injective on the loops not in R (see
///   `is_injective`), so iterations of different non-reduction points write different
///   elements and are independent;
/// - if R is empty, the body has no load of `T` at an index other than `I`: an iteration
///   then reads at most the element it writes itself;
/// - if R is not empty, the value is `T[I] + e` or `e + T[I]` with no load of `T` in `e`.
///   The only dependence is then the reassociation of a sum into the same element, which
///   is exactly what the `Reordering` class allows. Other forms, like `T[I] * 2 + e` or an
///   overwrite `T[I] = e`, depend on the order of the iterations in a way that is not
///   rounding, so they are refused.
///
/// Class: a loop is a reduction loop if its name is not in the store index (non-zero
/// coefficient). The step is `Reordering` iff it changes the relative order of the
/// reduction loops, as that is the order in which an element receives its contributions.
fn reorder(
    kernel: &mut Kernel,
    step: usize,
    nest_no: usize,
    order: &[String],
) -> Result<OrderClass, ScheduleError> {
    let nest = kernel
        .nests
        .get_mut(nest_no)
        .ok_or(ScheduleError::NoSuchNest {
            step,
            nest: nest_no,
        })?;
    let names_each_loop_once = order.len() == nest.loops.len()
        && nest
            .loops
            .iter()
            .all(|l| order.iter().filter(|name| **name == l.name).count() == 1);
    if !names_each_loop_once {
        return Err(ScheduleError::BadReorder {
            step,
            nest: nest_no,
        });
    }
    let target = &nest.body.target;
    let value = &nest.body.value;
    let is_reduction = |name: &str| {
        let in_index = |e: &AffineExpr| e.terms.iter().any(|(n, coeff)| n == name && *coeff != 0);
        !target.index.iter().any(in_index)
    };
    let has_reduction = nest.loops.iter().any(|l| is_reduction(&l.name));
    let is_value_legal = if has_reduction {
        is_accumulation(value, target)
    } else {
        !any_load(value, &|x| is_other_element(x, target))
    };
    if !is_value_legal || !is_injective(&nest.loops, target) {
        return Err(ScheduleError::IllegalReorder {
            step,
            nest: nest_no,
        });
    }
    let before = nest.loops.iter().map(|l| l.name.as_str());
    let after = order.iter().map(String::as_str);
    let is_same_reduction_order = before
        .filter(|name| is_reduction(name))
        .eq(after.filter(|name| is_reduction(name)));
    let loops: Vec<Loop> = order
        .iter()
        .map(|name| {
            let l = nest.loops.iter().find(|l| &l.name == name);
            l.expect("invariant: order is a permutation of the loops")
                .clone()
        })
        .collect();
    nest.loops = loops;
    Ok(if is_same_reduction_order {
        OrderClass::Preserving
    } else {
        OrderClass::Reordering
    })
}

/// True if `value` is `target + e` or `e + target`, where `e` has no load of the buffer
/// of `target`.
fn is_accumulation(value: &Expr, target: &Access) -> bool {
    let Expr::Binary(BinaryOp::Add, a, b) = value else {
        return false;
    };
    let is_target =
        |e: &Expr| matches!(e, Expr::Load(x) if x.buffer == target.buffer && same_index(x, target));
    let has_buffer = |e: &Expr| any_load(e, &|x| x.buffer == target.buffer);
    (is_target(a) && !has_buffer(b)) || (is_target(b) && !has_buffer(a))
}

/// True if `x` loads the buffer of `target` at an index other than that of `target`.
fn is_other_element(x: &Access, target: &Access) -> bool {
    x.buffer == target.buffer && !same_index(x, target)
}

/// True if some load in `e` satisfies `pred`.
fn any_load(e: &Expr, pred: &dyn Fn(&Access) -> bool) -> bool {
    match e {
        Expr::Load(access) => pred(access),
        Expr::Const(_) => false,
        Expr::Unary(_, x) => any_load(x, pred),
        Expr::Binary(_, a, b) => any_load(a, pred) || any_load(b, pred),
    }
}

/// Sufficient check that different values of the loops that appear in `target` give
/// different elements. Every such loop must appear in only one dimension of the index;
/// within a dimension, sort the terms by |coeff|, ascending; each |coeff| must exceed the
/// sum over the earlier terms of |coeff| * (extent - 1), the largest value they can add up
/// to. The terms are then the digits of a mixed-radix number, which determines them
/// uniquely: `i`, `2*ko + ki` with ki in 0..2, `4*io + ii` pass; `i + k` does not.
fn is_injective(loops: &[Loop], target: &Access) -> bool {
    let mut seen: Vec<&str> = Vec::new();
    for dim in &target.index {
        let mut terms: Vec<(u128, u128)> = Vec::new();
        for (name, coeff) in dim.terms.iter().filter(|(_, c)| *c != 0) {
            let extent = loops
                .iter()
                .find(|l| &l.name == name)
                .map_or(1, |l| l.extent);
            if seen.contains(&name.as_str()) {
                return false;
            }
            seen.push(name);
            terms.push((
                u128::from(coeff.unsigned_abs()),
                extent.saturating_sub(1) as u128,
            ));
        }
        terms.sort();
        let mut reach = 0u128;
        for (coeff, last) in terms {
            if coeff <= reach {
                return false;
            }
            reach = reach.saturating_add(coeff.saturating_mul(last));
        }
    }
    true
}

/// Equal constants and equal terms in any order, in every dimension; zero coefficients
/// are ignored.
fn same_index(a: &Access, b: &Access) -> bool {
    let sorted_terms = |e: &AffineExpr| {
        let mut terms: Vec<_> = e.terms.iter().filter(|(_, c)| *c != 0).cloned().collect();
        terms.sort();
        terms
    };
    a.index.len() == b.index.len()
        && a.index
            .iter()
            .zip(&b.index)
            .all(|(x, y)| x.constant == y.constant && sorted_terms(x) == sorted_terms(y))
}

/// Replaces loop `loop_name` of extent `n` by `outer` (extent ceil(n/factor)) and `inner`
/// (extent factor) and substitutes `loop_name := factor * outer + inner` in the nest.
/// If `factor` does not divide `n`, adds the guard `factor * outer + inner < n`.
fn split(
    kernel: &mut Kernel,
    step: usize,
    nest_no: usize,
    loop_name: &str,
    outer: &str,
    inner: &str,
    factor: usize,
) -> Result<(), ScheduleError> {
    let nest = kernel
        .nests
        .get_mut(nest_no)
        .ok_or(ScheduleError::NoSuchNest {
            step,
            nest: nest_no,
        })?;
    let pos = nest
        .loops
        .iter()
        .position(|l| l.name == loop_name)
        .ok_or_else(|| ScheduleError::NoSuchLoop {
            step,
            nest: nest_no,
            loop_name: loop_name.to_string(),
        })?;
    if factor == 0 {
        return Err(ScheduleError::ZeroFactor { step });
    }
    let is_taken = |name: &str| nest.loops.iter().any(|l| l.name == name);
    let taken = if is_taken(outer) {
        Some(outer)
    } else if is_taken(inner) || inner == outer {
        Some(inner)
    } else {
        None
    };
    if let Some(name) = taken {
        return Err(ScheduleError::NameTaken {
            step,
            nest: nest_no,
            name: name.to_string(),
        });
    }
    let too_large = ScheduleError::NumberTooLarge { step };
    let extent = nest.loops[pos].extent;
    let factor_i64 = i64::try_from(factor).map_err(|_| too_large.clone())?;
    let extent_i64 = i64::try_from(extent).map_err(|_| too_large.clone())?;

    let replacement = [(outer.to_string(), factor_i64), (inner.to_string(), 1)];
    let substitute = |e: &AffineExpr| substitute_loop(e, loop_name, &replacement);
    substitute_access(&mut nest.body.target, &substitute).ok_or_else(|| too_large.clone())?;
    substitute_value(&mut nest.body.value, &substitute).ok_or_else(|| too_large.clone())?;
    for guard in &mut nest.guards {
        guard.expr = substitute(&guard.expr).ok_or_else(|| too_large.clone())?;
    }
    if extent % factor != 0 {
        nest.guards.push(Guard {
            expr: AffineExpr {
                constant: 0,
                terms: replacement.to_vec(),
            },
            bound: extent_i64,
        });
    }
    // A vectorized loop stays vectorized as `inner`, which is then the innermost loop.
    let new_loops = [
        Loop {
            name: outer.to_string(),
            extent: extent.div_ceil(factor),
            vectorize: false,
        },
        Loop {
            name: inner.to_string(),
            extent: factor,
            vectorize: nest.loops[pos].vectorize,
        },
    ];
    nest.loops.splice(pos..=pos, new_loops);
    Ok(())
}

/// `expr` with each term of `name` replaced by the `replacement` terms, scaled by its
/// coefficient; like terms are combined and zero coefficients dropped. `None` on overflow.
fn substitute_loop(
    expr: &AffineExpr,
    name: &str,
    replacement: &[(String, i64)],
) -> Option<AffineExpr> {
    if !expr.terms.iter().any(|(n, _)| n == name) {
        return Some(expr.clone());
    }
    let mut terms: Vec<(String, i64)> = Vec::new();
    for (n, coeff) in &expr.terms {
        let scaled: Vec<(String, i64)> = if n == name {
            replacement
                .iter()
                .map(|(rn, rc)| Some((rn.clone(), coeff.checked_mul(*rc)?)))
                .collect::<Option<_>>()?
        } else {
            vec![(n.clone(), *coeff)]
        };
        for (sn, sc) in scaled {
            match terms.iter_mut().find(|t| t.0 == sn) {
                Some(t) => t.1 = t.1.checked_add(sc)?,
                None => terms.push((sn, sc)),
            }
        }
    }
    terms.retain(|t| t.1 != 0);
    Some(AffineExpr {
        constant: expr.constant,
        terms,
    })
}

type Substitute<'a> = &'a dyn Fn(&AffineExpr) -> Option<AffineExpr>;

fn substitute_access(access: &mut Access, substitute: Substitute) -> Option<()> {
    for index in &mut access.index {
        *index = substitute(index)?;
    }
    Some(())
}

fn substitute_value(e: &mut Expr, substitute: Substitute) -> Option<()> {
    match e {
        Expr::Load(access) => substitute_access(access, substitute),
        Expr::Const(_) => Some(()),
        Expr::Unary(_, x) => substitute_value(x, substitute),
        Expr::Binary(_, a, b) => {
            substitute_value(a, substitute)?;
            substitute_value(b, substitute)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BinaryOp, Buffer, LoopNest, Store, lower_elementwise};
    use coffee_ten_ir::{Graph, Shape};
    use proptest::prelude::*;
    use std::collections::BTreeMap;

    fn lp(name: &str, extent: usize) -> Loop {
        Loop {
            name: name.to_string(),
            extent,
            vectorize: false,
        }
    }

    fn buf(name: &str, shape: &[usize]) -> Buffer {
        Buffer {
            name: name.to_string(),
            shape: shape.to_vec(),
        }
    }

    fn var(name: &str) -> AffineExpr {
        AffineExpr {
            constant: 0,
            terms: vec![(name.to_string(), 1)],
        }
    }

    fn load(buffer: usize, index: Vec<AffineExpr>) -> Expr {
        Expr::Load(Access { buffer, index })
    }

    fn bin(op: BinaryOp, a: Expr, b: Expr) -> Expr {
        Expr::Binary(op, Box::new(a), Box::new(b))
    }

    fn nest(loops: Vec<Loop>, index: Vec<AffineExpr>, value: Expr) -> LoopNest {
        LoopNest {
            loops,
            guards: Vec::new(),
            body: Store {
                target: Access { buffer: 2, index },
                value,
            },
        }
    }

    fn matmul() -> Kernel {
        matmul_sized(4, 3, 2)
    }

    fn matmul_sized(m: usize, k: usize, n: usize) -> Kernel {
        Kernel {
            name: "matmul".to_string(),
            buffers: vec![buf("A", &[m, k]), buf("B", &[k, n]), buf("C", &[m, n])],
            nests: vec![
                nest(
                    vec![lp("i", m), lp("j", n)],
                    vec![var("i"), var("j")],
                    Expr::Const(0.0),
                ),
                nest(
                    vec![lp("i", m), lp("k", k), lp("j", n)],
                    vec![var("i"), var("j")],
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
            ],
        }
    }

    fn split_step(nest: usize, loop_name: &str, outer: &str, inner: &str, factor: usize) -> Step {
        Step::Split {
            nest,
            loop_name: loop_name.to_string(),
            outer: outer.to_string(),
            inner: inner.to_string(),
            factor,
        }
    }

    fn schedule(text: &str) -> Schedule {
        Schedule::parse(text).expect("test schedule parses")
    }

    fn applied(kernel: &Kernel, text: &str) -> Kernel {
        apply(kernel, &schedule(text)).expect("test schedule applies")
    }

    /// The text of nest `n`, without the kernel header line.
    fn nest_text(kernel: &Kernel, n: usize) -> String {
        let only_nest = Kernel {
            nests: vec![kernel.nests[n].clone()],
            ..kernel.clone()
        };
        only_nest
            .to_string()
            .lines()
            .skip(1)
            .collect::<Vec<_>>()
            .join("\n")
            + "\n"
    }

    #[test]
    fn split_with_remainder_adds_a_guard() {
        let k = matmul();
        let s = applied(&k, "split(1, k, ko, ki, 2)");
        assert_eq!(
            nest_text(&s, 1),
            concat!(
                "for i in 0..4:\n",
                "  for ko in 0..2:\n",
                "    for ki in 0..2:\n",
                "      for j in 0..2:\n",
                "        if 2*ko + ki < 3:\n",
                "          C[i, j] = C[i, j] + (A[i, 2*ko + ki] * B[2*ko + ki, j])\n",
            )
        );
        assert_eq!(s.nests[0], k.nests[0]);
        assert_eq!(s.validate(), Ok(()));
    }

    #[test]
    fn split_by_a_divisor_has_no_guard() {
        let s = applied(&matmul(), "split(1, k, ko, ki, 3)");
        assert_eq!(
            nest_text(&s, 1),
            concat!(
                "for i in 0..4:\n",
                "  for ko in 0..1:\n",
                "    for ki in 0..3:\n",
                "      for j in 0..2:\n",
                "        C[i, j] = C[i, j] + (A[i, 3*ko + ki] * B[3*ko + ki, j])\n",
            )
        );
        assert!(s.nests[1].guards.is_empty());
    }

    #[test]
    fn splitting_the_outer_loop_again_rewrites_the_guard() {
        let s = applied(
            &matmul(),
            "split(1, k, ko, ki, 2); split(1, ko, koo, koi, 2)",
        );
        assert_eq!(
            nest_text(&s, 1),
            concat!(
                "for i in 0..4:\n",
                "  for koo in 0..1:\n",
                "    for koi in 0..2:\n",
                "      for ki in 0..2:\n",
                "        for j in 0..2:\n",
                "          if 4*koo + 2*koi + ki < 3:\n",
                "            C[i, j] = C[i, j] + (A[i, 4*koo + 2*koi + ki] * B[4*koo + 2*koi + ki, j])\n",
            )
        );
    }

    #[test]
    fn two_remainders_give_two_nested_guards() {
        let mut k = matmul();
        k.buffers[0].shape = vec![4, 5];
        k.buffers[1].shape = vec![5, 2];
        k.nests[1].loops[1].extent = 5;
        let s = applied(&k, "split(1, k, ko, ki, 2); split(1, ko, koo, koi, 2)");
        assert_eq!(
            nest_text(&s, 1),
            concat!(
                "for i in 0..4:\n",
                "  for koo in 0..2:\n",
                "    for koi in 0..2:\n",
                "      for ki in 0..2:\n",
                "        for j in 0..2:\n",
                "          if 4*koo + 2*koi + ki < 5:\n",
                "            if 2*koo + koi < 3:\n",
                "              C[i, j] = C[i, j] + (A[i, 4*koo + 2*koi + ki] * B[4*koo + 2*koi + ki, j])\n",
            )
        );
    }

    #[test]
    fn substitution_combines_like_terms_and_drops_zeros() {
        let replacement = [("ko".to_string(), 2), ("ki".to_string(), 1)];
        let expr = AffineExpr {
            constant: 1,
            terms: vec![
                ("k".to_string(), 2),
                ("j".to_string(), 1),
                ("k".to_string(), -2),
            ],
        };
        assert_eq!(
            substitute_loop(&expr, "k", &replacement),
            Some(AffineExpr {
                constant: 1,
                terms: vec![("j".to_string(), 1)],
            })
        );
        let zero = AffineExpr {
            constant: 0,
            terms: vec![("k".to_string(), 0)],
        };
        assert_eq!(
            substitute_loop(&zero, "k", &replacement),
            Some(AffineExpr {
                constant: 0,
                terms: vec![],
            })
        );
    }

    /// Applies `schedule` to `kernel`, expects `expected`, and checks `kernel` is unchanged.
    fn assert_refused(kernel: &Kernel, schedule: &Schedule, expected: ScheduleError) {
        let before = kernel.clone();
        assert_eq!(apply(kernel, schedule), Err(expected));
        assert_eq!(kernel, &before);
    }

    #[test]
    fn missing_nest_is_refused_with_its_step() {
        assert_refused(
            &matmul(),
            &schedule("split(1, k, ko, ki, 2); split(2, k, a, b, 2)"),
            ScheduleError::NoSuchNest { step: 1, nest: 2 },
        );
    }

    #[test]
    fn missing_loop_is_refused() {
        // `k` is gone after the first split.
        assert_refused(
            &matmul(),
            &schedule("split(1, k, ko, ki, 2); split(1, k, a, b, 2)"),
            ScheduleError::NoSuchLoop {
                step: 1,
                nest: 1,
                loop_name: "k".to_string(),
            },
        );
    }

    #[test]
    fn taken_names_are_refused() {
        for (text, name) in [
            ("split(1, k, i, ki, 2)", "i"),
            ("split(1, k, ko, j, 2)", "j"),
        ] {
            assert_refused(
                &matmul(),
                &schedule(text),
                ScheduleError::NameTaken {
                    step: 0,
                    nest: 1,
                    name: name.to_string(),
                },
            );
        }
        // The parser forbids equal names; a hand-built step must not create two loops `x`.
        let same = Schedule {
            steps: vec![split_step(1, "k", "x", "x", 2)],
        };
        assert_refused(
            &matmul(),
            &same,
            ScheduleError::NameTaken {
                step: 0,
                nest: 1,
                name: "x".to_string(),
            },
        );
    }

    #[test]
    fn zero_factor_is_refused() {
        // The parser forbids factor 0; a hand-built step must be refused here too.
        let zero = Schedule {
            steps: vec![split_step(1, "k", "ko", "ki", 0)],
        };
        assert_refused(&matmul(), &zero, ScheduleError::ZeroFactor { step: 0 });
    }

    #[test]
    fn vectorize_marks_the_innermost_loop() {
        let k = matmul();
        let (s, class) = classified(&k, "vectorize(1, j)");
        assert_eq!(
            nest_text(&s, 1),
            concat!(
                "for i in 0..4:\n",
                "  for k in 0..3:\n",
                "    for j in 0..2 (vectorize):\n",
                "      C[i, j] = C[i, j] + (A[i, k] * B[k, j])\n",
            )
        );
        assert_eq!(s.nests[0], k.nests[0]);
        assert_eq!(s.validate(), Ok(()));
        assert_eq!(class, OrderClass::Preserving);
    }

    fn illegal_vectorize(step: usize, nest: usize, reason: &'static str) -> ScheduleError {
        ScheduleError::IllegalVectorize { step, nest, reason }
    }

    #[test]
    fn vectorizing_a_loop_that_is_not_innermost_is_refused() {
        assert_refused(
            &matmul(),
            &schedule("vectorize(1, k)"),
            illegal_vectorize(0, 1, "not the innermost loop"),
        );
        // `j` is no longer innermost once the order changes.
        assert_refused(
            &matmul(),
            &schedule("reorder(1, i, j, k); vectorize(1, j)"),
            illegal_vectorize(1, 1, "not the innermost loop"),
        );
    }

    #[test]
    fn vectorizing_a_reduction_loop_is_refused() {
        assert_refused(
            &matmul(),
            &schedule("reorder(1, i, j, k); vectorize(1, k)"),
            illegal_vectorize(1, 1, "reduction loop"),
        );
    }

    /// `for i in 0..4: for j in 0..4: Y[target] = A[j]`, with Y: f32[8, 8], or f32[16] for rank 1.
    fn target_kernel(target: Vec<AffineExpr>) -> Kernel {
        let y_shape: &[usize] = if target.len() == 2 { &[8, 8] } else { &[16] };
        let k = Kernel {
            name: "target".to_string(),
            buffers: vec![buf("A", &[4]), buf("B", &[4]), buf("Y", y_shape)],
            nests: vec![nest(
                vec![lp("i", 4), lp("j", 4)],
                target,
                load(0, vec![var("j")]),
            )],
        };
        assert_eq!(k.validate(), Ok(()));
        k
    }

    #[test]
    fn target_stride_other_than_one_is_refused() {
        let stride = illegal_vectorize(0, 0, "target stride is not 1");
        let cases = [
            // Stride 2, and stride -1.
            vec![affine(&[("j", 2)])],
            vec![AffineExpr {
                constant: 3,
                terms: vec![("j".to_string(), -1)],
            }],
            // `j` in a dimension other than the last, alone and next to a unit stride.
            vec![var("j"), var("i")],
            vec![var("j"), affine(&[("i", 1), ("j", 1)])],
        ];
        for target in cases {
            let k = target_kernel(target);
            assert_refused(&k, &schedule("vectorize(0, j)"), stride.clone());
        }
        // Control: a unit stride is legal, also with other loops in the index.
        let ok = target_kernel(vec![affine(&[("i", 4), ("j", 1)])]);
        assert_eq!(classified(&ok, "vectorize(0, j)").1, OrderClass::Preserving);
        let ok = target_kernel(vec![var("i"), var("j")]);
        assert_eq!(classified(&ok, "vectorize(0, j)").1, OrderClass::Preserving);
    }

    #[test]
    fn vectorizing_twice_is_refused() {
        assert_refused(
            &matmul(),
            &schedule("vectorize(1, j); vectorize(1, j)"),
            illegal_vectorize(1, 1, "already vectorized"),
        );
    }

    #[test]
    fn vectorize_names_a_missing_nest_or_loop() {
        assert_refused(
            &matmul(),
            &schedule("vectorize(2, j)"),
            ScheduleError::NoSuchNest { step: 0, nest: 2 },
        );
        assert_refused(
            &matmul(),
            &schedule("vectorize(1, x)"),
            ScheduleError::NoSuchLoop {
                step: 0,
                nest: 1,
                loop_name: "x".to_string(),
            },
        );
    }

    #[test]
    fn splitting_the_vectorized_loop_vectorizes_the_inner_loop() {
        let (s, _) = classified(&matmul(), "vectorize(1, j); split(1, j, jo, ji, 1)");
        let names: Vec<_> = s.nests[1].loops.iter().map(|l| l.name.as_str()).collect();
        let marks: Vec<_> = s.nests[1].loops.iter().map(|l| l.vectorize).collect();
        assert_eq!(names, ["i", "k", "jo", "ji"]);
        assert_eq!(marks, [false, false, false, true]);
        // Splitting another loop keeps the mark.
        let (s, _) = classified(&matmul(), "vectorize(1, j); split(1, k, ko, ki, 2)");
        assert!(s.nests[1].loops[3].vectorize);
    }

    #[test]
    fn reorder_that_moves_the_vectorized_loop_is_invalid() {
        assert_refused(
            &matmul(),
            &schedule("vectorize(1, j); reorder(1, j, i, k)"),
            ScheduleError::Invalid(LoopIrError::BadVectorize { nest: 1 }),
        );
        let (s, _) = classified(&matmul(), "vectorize(1, j); reorder(1, k, i, j)");
        assert!(s.nests[1].loops[2].vectorize);
    }

    fn classified(kernel: &Kernel, text: &str) -> (Kernel, OrderClass) {
        apply_classified(kernel, &schedule(text)).expect("test schedule applies")
    }

    #[test]
    fn reorder_prints_the_new_order() {
        let k = matmul();
        let (s, class) = classified(&k, "reorder(1, i, j, k)");
        assert_eq!(
            nest_text(&s, 1),
            concat!(
                "for i in 0..4:\n",
                "  for j in 0..2:\n",
                "    for k in 0..3:\n",
                "      C[i, j] = C[i, j] + (A[i, k] * B[k, j])\n",
            )
        );
        assert_eq!(s.nests[0], k.nests[0]);
        assert_eq!(s.validate(), Ok(()));
        assert_eq!(class, OrderClass::Preserving);
    }

    #[test]
    fn reorder_keeps_guards_and_extents() {
        let (s, _) = classified(
            &matmul(),
            "split(1, k, ko, ki, 2); reorder(1, ki, j, i, ko)",
        );
        assert_eq!(
            nest_text(&s, 1),
            concat!(
                "for ki in 0..2:\n",
                "  for j in 0..2:\n",
                "    for i in 0..4:\n",
                "      for ko in 0..2:\n",
                "        if 2*ko + ki < 3:\n",
                "          C[i, j] = C[i, j] + (A[i, 2*ko + ki] * B[2*ko + ki, j])\n",
            )
        );
    }

    #[test]
    fn swapping_the_split_reduction_loops_is_reordering() {
        let (_, class) = classified(
            &matmul(),
            "split(1, k, ko, ki, 2); reorder(1, i, ki, ko, j)",
        );
        assert_eq!(class, OrderClass::Reordering);
    }

    #[test]
    fn moving_other_loops_around_the_reduction_loops_is_preserving() {
        let (_, class) = classified(
            &matmul(),
            "split(1, k, ko, ki, 2); reorder(1, ko, i, ki, j)",
        );
        assert_eq!(class, OrderClass::Preserving);
    }

    #[test]
    fn one_reordering_step_makes_the_schedule_reordering() {
        let (_, class) = classified(
            &matmul(),
            "split(1, k, ko, ki, 2); reorder(1, i, ki, ko, j); reorder(1, i, ko, ki, j)",
        );
        assert_eq!(class, OrderClass::Reordering);
    }

    #[test]
    fn every_order_of_an_elementwise_nest_is_preserving() {
        let mut g = Graph::new();
        let a = g.input("A", Shape::new(vec![2, 3, 2])).unwrap();
        let y = g.unary(coffee_ten_ir::UnaryOp::Relu, a).unwrap();
        let k = lower_elementwise(&g, y, "k").unwrap();
        let names: Vec<String> = k.nests[0].loops.iter().map(|l| l.name.clone()).collect();
        for picks in [
            [0, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 1, 0],
            [2, 0, 0],
            [2, 1, 0],
        ] {
            let steps = vec![Step::Reorder {
                nest: 0,
                order: permute(&names, &picks),
            }];
            let (_, class) = apply_classified(&k, &Schedule { steps }).unwrap();
            assert_eq!(class, OrderClass::Preserving);
        }
    }

    /// The permutation of `names` given by `picks`: take `names[picks[n] % len]` out of
    /// the names left, for n = 0, 1, ...; missing picks count as 0.
    fn permute(names: &[String], picks: &[usize]) -> Vec<String> {
        let mut left = names.to_vec();
        let mut order = Vec::new();
        for n in 0..names.len() {
            let pick = picks.get(n).copied().unwrap_or(0);
            order.push(left.remove(pick % left.len()));
        }
        order
    }

    fn reorder_step(nest: usize, names: &[&str]) -> Step {
        Step::Reorder {
            nest,
            order: names.iter().map(|n| n.to_string()).collect(),
        }
    }

    #[test]
    fn order_that_is_not_a_permutation_is_refused() {
        let bad = ScheduleError::BadReorder { step: 0, nest: 1 };
        // Missing a loop, an extra loop, an unknown name, a repeated loop (the parser
        // forbids repeats, so these steps are built by hand).
        for names in [
            vec!["i", "j"],
            vec!["i", "k", "j", "x"],
            vec!["i", "j", "x"],
            vec!["i", "k", "k"],
            vec!["i", "i", "j", "k"],
            vec![],
        ] {
            let steps = vec![reorder_step(1, &names)];
            assert_refused(&matmul(), &Schedule { steps }, bad.clone());
        }
        // The step number is that of the reorder.
        assert_refused(
            &matmul(),
            &schedule("split(1, k, ko, ki, 2); reorder(1, i, j, k)"),
            ScheduleError::BadReorder { step: 1, nest: 1 },
        );
        assert_refused(
            &matmul(),
            &schedule("reorder(5, i, j, k)"),
            ScheduleError::NoSuchNest { step: 0, nest: 5 },
        );
    }

    /// `for i in 0..4: Y[i] = Y[0] + A[i]`: reads `Y` at another index than it writes.
    fn reads_y_at_zero() -> Kernel {
        let zero = AffineExpr {
            constant: 0,
            terms: Vec::new(),
        };
        let value = bin(BinaryOp::Add, load(2, vec![zero]), load(0, vec![var("i")]));
        Kernel {
            name: "reads_y".to_string(),
            buffers: vec![buf("A", &[4]), buf("B", &[4]), buf("Y", &[4])],
            nests: vec![nest(vec![lp("i", 4)], vec![var("i")], value)],
        }
    }

    #[test]
    fn reading_the_target_at_another_index_is_illegal() {
        let k = reads_y_at_zero();
        assert_eq!(k.validate(), Ok(()));
        assert_refused(
            &k,
            &schedule("reorder(0, i)"),
            ScheduleError::IllegalReorder { step: 0, nest: 0 },
        );
    }

    #[test]
    fn target_loads_may_order_terms_differently_but_not_change_the_index() {
        // Y[4*i + j] = Y[j + 4*i] + A[i]: same element, terms in another order.
        let flat = |terms: Vec<(&str, i64)>, constant: i64| AffineExpr {
            constant,
            terms: terms.into_iter().map(|(n, c)| (n.to_string(), c)).collect(),
        };
        let kernel = |load_index: AffineExpr| Kernel {
            name: "flat".to_string(),
            buffers: vec![buf("A", &[4]), buf("B", &[4]), buf("Y", &[20])],
            nests: vec![nest(
                vec![lp("i", 4), lp("j", 4)],
                vec![flat(vec![("i", 4), ("j", 1)], 0)],
                bin(
                    BinaryOp::Add,
                    load(2, vec![load_index]),
                    load(0, vec![var("i")]),
                ),
            )],
        };
        let same = kernel(flat(vec![("j", 1), ("i", 4)], 0));
        let (_, class) = classified(&same, "reorder(0, j, i)");
        assert_eq!(class, OrderClass::Preserving);
        let shifted = kernel(flat(vec![("j", 1), ("i", 4)], 1));
        assert_eq!(shifted.validate(), Ok(()));
        assert_refused(
            &shifted,
            &schedule("reorder(0, j, i)"),
            ScheduleError::IllegalReorder { step: 0, nest: 0 },
        );
    }

    fn affine(terms: &[(&str, i64)]) -> AffineExpr {
        AffineExpr {
            constant: 0,
            terms: terms.iter().map(|(n, c)| (n.to_string(), *c)).collect(),
        }
    }

    /// `for i in 0..4: for k in 0..3: Y[target] = value`, with A: f32[4, 3], B: f32[3] and
    /// Y: f32[y_len] as buffers 0, 1 and 2.
    fn ik_kernel(y_len: usize, target: AffineExpr, value: Expr) -> Kernel {
        let k = Kernel {
            name: "ik".to_string(),
            buffers: vec![buf("A", &[4, 3]), buf("B", &[3]), buf("Y", &[y_len])],
            nests: vec![nest(vec![lp("i", 4), lp("k", 3)], vec![target], value)],
        };
        assert_eq!(k.validate(), Ok(()));
        k
    }

    fn a_ik() -> Expr {
        load(0, vec![var("i"), var("k")])
    }

    #[test]
    fn sum_into_the_target_is_legal_in_both_operand_orders() {
        let y_i = || load(2, vec![var("i")]);
        for value in [
            bin(BinaryOp::Add, y_i(), a_ik()),
            bin(BinaryOp::Add, a_ik(), y_i()),
        ] {
            let k = ik_kernel(4, var("i"), value);
            assert_eq!(classified(&k, "reorder(0, k, i)").1, OrderClass::Preserving);
        }
    }

    #[test]
    fn accumulation_that_is_not_a_plain_sum_is_illegal() {
        let y_i = || load(2, vec![var("i")]);
        let two = || Expr::Const(2.0);
        let illegal = ScheduleError::IllegalReorder { step: 0, nest: 0 };
        let values = [
            // Horner: the order of the k iterations changes the value.
            bin(BinaryOp::Add, bin(BinaryOp::Mul, y_i(), two()), a_ik()),
            bin(BinaryOp::Sub, y_i(), a_ik()),
            bin(BinaryOp::Add, y_i(), bin(BinaryOp::Add, y_i(), a_ik())),
            bin(
                BinaryOp::Add,
                load(
                    2,
                    vec![AffineExpr {
                        constant: 1,
                        terms: vec![],
                    }],
                ),
                a_ik(),
            ),
            // Overwrite: the last k iteration wins.
            a_ik(),
        ];
        for value in values {
            let k = ik_kernel(4, var("i"), value);
            assert_refused(&k, &schedule("reorder(0, k, i)"), illegal.clone());
        }
    }

    #[test]
    fn target_index_that_is_not_injective_is_illegal() {
        // Y[i + k] += A[i, k]: (i, k) = (0, 1) and (1, 0) hit the same element.
        let target = affine(&[("i", 1), ("k", 1)]);
        let y = load(2, vec![target.clone()]);
        let k = ik_kernel(6, target, bin(BinaryOp::Add, y, a_ik()));
        assert_refused(
            &k,
            &schedule("reorder(0, k, i)"),
            ScheduleError::IllegalReorder { step: 0, nest: 0 },
        );
    }

    #[test]
    fn zero_coefficient_terms_do_not_change_an_index() {
        // C[i + 0*k] and C[i] are the same element, and `k` is still a reduction loop.
        let with_zero = affine(&[("i", 1), ("k", 0)]);
        for (target, read) in [(var("i"), with_zero.clone()), (with_zero, var("i"))] {
            let value = bin(BinaryOp::Add, load(2, vec![read]), a_ik());
            let k = ik_kernel(4, target, value);
            assert_eq!(classified(&k, "reorder(0, k, i)").1, OrderClass::Preserving);
        }
    }

    #[test]
    fn mixed_radix_target_of_a_split_is_legal() {
        let (_, class) = classified(
            &matmul(),
            "split(1, i, io, ii, 3); split(1, k, ko, ki, 2); reorder(1, ki, j, ii, io, ko)",
        );
        assert_eq!(class, OrderClass::Reordering);
        let (_, class) = classified(
            &matmul(),
            "split(1, j, jo, ji, 2); reorder(1, ji, k, i, jo)",
        );
        assert_eq!(class, OrderClass::Preserving);
    }

    #[test]
    fn invalid_result_is_reported() {
        // The input kernel reads A[i, k] with k up to 4, outside 0..3; the split keeps that.
        let mut k = matmul();
        k.nests[1].loops[1].extent = 5;
        assert_refused(
            &k,
            &schedule("split(1, k, ko, ki, 2)"),
            ScheduleError::Invalid(LoopIrError::OutOfBounds {
                nest: 1,
                buffer: "A".to_string(),
                dim: 1,
                min: 0,
                max: 4,
                size: 3,
            }),
        );
    }

    #[test]
    fn factor_must_fit_i64() {
        let mut k = matmul();
        k.nests[0].loops.truncate(1);
        k.nests[0].body.target.index = vec![
            var("i"),
            AffineExpr {
                constant: 0,
                terms: vec![],
            },
        ];
        // i64::MAX still works: ko has extent 1 and the guard keeps `ki` in 0..4.
        let max = i64::MAX as usize;
        let s = applied(&k, &format!("split(0, i, io, ii, {max})"));
        assert_eq!(s.nests[0].guards.len(), 1);
        assert_eq!(s.validate(), Ok(()));
        assert_refused(
            &k,
            &schedule(&format!("split(0, i, io, ii, {})", max + 1)),
            ScheduleError::NumberTooLarge { step: 0 },
        );
        // A coefficient of 2 on `i` overflows at factor i64::MAX.
        let mut doubled = k.clone();
        doubled.buffers[2].shape = vec![8, 2];
        doubled.nests[0].body.target.index[0].terms = vec![("i".to_string(), 2)];
        assert_refused(
            &doubled,
            &schedule(&format!("split(0, i, io, ii, {max})")),
            ScheduleError::NumberTooLarge { step: 0 },
        );
    }

    #[test]
    fn empty_schedule_is_identity() {
        let k = matmul();
        assert_eq!(apply(&k, &Schedule::default()), Ok(k.clone()));
    }

    type Trace = Vec<(usize, Vec<i64>, Vec<(usize, Vec<i64>)>)>;

    /// Every nest's iterations in order, skipping those whose guards fail:
    /// (store buffer, store index, [(load buffer, load index)] in expression order).
    fn trace(kernel: &Kernel) -> Trace {
        let mut out = Vec::new();
        for nest in &kernel.nests {
            let iterations: usize = nest.loops.iter().map(|l| l.extent).product();
            for n in 0..iterations {
                let mut values = vec![0i64; nest.loops.len()];
                let mut rest = n;
                for (value, l) in values.iter_mut().zip(&nest.loops).rev() {
                    *value = (rest % l.extent) as i64;
                    rest /= l.extent;
                }
                let eval = |e: &AffineExpr| {
                    let terms: i64 = e
                        .terms
                        .iter()
                        .map(|(name, coeff)| {
                            let pos = nest.loops.iter().position(|l| &l.name == name);
                            coeff * values[pos.expect("test kernels are valid")]
                        })
                        .sum();
                    e.constant + terms
                };
                if nest.guards.iter().any(|g| eval(&g.expr) >= g.bound) {
                    continue;
                }
                let index_of = |a: &Access| (a.buffer, a.index.iter().map(eval).collect());
                let mut loads = Vec::new();
                collect_loads(&nest.body.value, &mut loads);
                let (store_buffer, store_index) = index_of(&nest.body.target);
                let loads = loads.into_iter().map(index_of).collect();
                out.push((store_buffer, store_index, loads));
            }
        }
        out
    }

    fn collect_loads<'a>(e: &'a Expr, out: &mut Vec<&'a Access>) {
        match e {
            Expr::Load(access) => out.push(access),
            Expr::Const(_) => {}
            Expr::Unary(_, x) => collect_loads(x, out),
            Expr::Binary(_, a, b) => {
                collect_loads(a, out);
                collect_loads(b, out);
            }
        }
    }

    #[test]
    fn split_after_exact_split_with_remainder_applies() {
        // 9 = 3 * 3 splits without a guard; splitting the outer loop by 2 needs the guard
        // `2*sso + ssi < 3`, and the index is `6*sso + 3*ssi + si = 3*(2*sso + ssi) + si`.
        let mut g = Graph::new();
        let a = g.input("A", Shape::new(vec![9])).unwrap();
        let y = g.unary(coffee_ten_ir::UnaryOp::Neg, a).unwrap();
        let k = lower_elementwise(&g, y, "k").unwrap();
        let once = applied(&k, "split(0, i0, so, si, 3)");
        let twice = applied(&once, "split(0, so, sso, ssi, 2)");
        assert_eq!(twice.nests[0].guards.len(), 1);
        assert_eq!(twice.validate(), Ok(()));
        assert_eq!(trace(&twice), trace(&k));
    }

    #[test]
    fn trace_of_split_matmul_has_the_same_32_iterations() {
        let k = matmul();
        let s = applied(&k, "split(1, k, ko, ki, 2)");
        // 4*2 init iterations plus 4*3*2 update iterations; the guard drops 8 of 32 ki steps.
        assert_eq!(trace(&k).len(), 32);
        assert_eq!(trace(&s), trace(&k));
    }

    /// Small random elementwise kernel: output rank 0..=3, dims 1..=5, 1..=3 inputs that
    /// broadcast to it, up to 5 ops over earlier nodes.
    fn elementwise_kernel() -> impl Strategy<Value = Kernel> {
        let out_dims = prop::collection::vec(1usize..=5, 0..=3);
        let input = (0usize..=3, prop::collection::vec(any::<bool>(), 3));
        let inputs = prop::collection::vec(input, 1..=3);
        let op = (0u8..6, any::<usize>(), any::<usize>());
        let ops = prop::collection::vec(op, 0..=5);
        (out_dims, inputs, ops).prop_map(|(out_dims, inputs, ops)| {
            let mut g = Graph::new();
            let mut nodes = vec![g.constant(0.5)];
            for (n, (rank, collapse)) in inputs.iter().enumerate() {
                let rank = (*rank).min(out_dims.len());
                let tail = &out_dims[out_dims.len() - rank..];
                let dims = tail
                    .iter()
                    .zip(collapse)
                    .map(|(&d, &one)| if one { 1 } else { d })
                    .collect();
                nodes.push(g.input(&format!("x{n}"), Shape::new(dims)).unwrap());
            }
            for (code, a, b) in ops {
                let (a, b) = (nodes[a % nodes.len()], nodes[b % nodes.len()]);
                let id = match code {
                    0 => g.unary(coffee_ten_ir::UnaryOp::Neg, a),
                    1 => g.unary(coffee_ten_ir::UnaryOp::Relu, a),
                    2 => g.binary(coffee_ten_ir::BinaryOp::Add, a, b),
                    3 => g.binary(coffee_ten_ir::BinaryOp::Sub, a, b),
                    4 => g.binary(coffee_ten_ir::BinaryOp::Mul, a, b),
                    _ => g.binary(coffee_ten_ir::BinaryOp::Max, a, b),
                };
                nodes.push(id.expect("all shapes broadcast to the output shape"));
            }
            let output = *nodes.last().expect("nodes has the constant");
            lower_elementwise(&g, output, "k").expect("small graphs lower")
        })
    }

    fn any_kernel() -> impl Strategy<Value = Kernel> {
        prop_oneof![1 => Just(matmul()), 4 => elementwise_kernel()]
    }

    /// Small random kernels with reduction loops (matmul) and without (elementwise).
    fn reorderable_kernel() -> impl Strategy<Value = Kernel> {
        let matmul =
            (1usize..=4, 1usize..=6, 1usize..=4).prop_map(|(m, k, n)| matmul_sized(m, k, n));
        prop_oneof![2 => matmul, 1 => elementwise_kernel()]
    }

    type ByElement = BTreeMap<(usize, Vec<i64>), Vec<Vec<(usize, Vec<i64>)>>>;

    /// The trace entries grouped by the element they store to, each group in trace order.
    fn by_element(trace: Trace) -> ByElement {
        let mut groups: BTreeMap<_, Vec<_>> = BTreeMap::new();
        for (buffer, index, loads) in trace {
            groups.entry((buffer, index)).or_default().push(loads);
        }
        groups
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            max_global_rejects: 8192,
            ..ProptestConfig::default()
        })]

        #[test]
        fn split_keeps_the_trace(
            kernel in any_kernel(),
            nest_pick in any::<usize>(),
            loop_pick in any::<usize>(),
            factor in 1usize..=6,
        ) {
            let nest = nest_pick % kernel.nests.len();
            let loops = &kernel.nests[nest].loops;
            prop_assume!(!loops.is_empty());
            let name = &loops[loop_pick % loops.len()].name;
            let steps = vec![split_step(nest, name, "so", "si", factor)];
            let split = apply(&kernel, &Schedule { steps }).expect("split applies");
            prop_assert_eq!(trace(&split), trace(&kernel));
        }

        #[test]
        fn two_splits_in_a_row_keep_the_trace(
            kernel in any_kernel(),
            nest_pick in any::<usize>(),
            loop_pick in any::<usize>(),
            factor in 1usize..=6,
            again_pick in any::<usize>(),
            again_factor in 1usize..=6,
        ) {
            let nest = nest_pick % kernel.nests.len();
            let loops = &kernel.nests[nest].loops;
            prop_assume!(!loops.is_empty());
            let name = &loops[loop_pick % loops.len()].name;
            let first = Schedule { steps: vec![split_step(nest, name, "so", "si", factor)] };
            let once = apply(&kernel, &first).expect("first split applies");
            // Any loop of the result, including `so`, the outer loop just made.
            let loops = &once.nests[nest].loops;
            let name = &loops[again_pick % loops.len()].name;
            let second = Schedule {
                steps: vec![split_step(nest, name, "sso", "ssi", again_factor)],
            };
            let twice = apply(&once, &second).expect("second split applies");
            prop_assert_eq!(trace(&twice), trace(&kernel));
        }

        #[test]
        fn reorder_after_split_keeps_the_contributions_of_every_element(
            kernel in reorderable_kernel(),
            nest_pick in any::<usize>(),
            loop_pick in any::<usize>(),
            factor in 1usize..=6,
            picks in prop::collection::vec(any::<usize>(), 5),
        ) {
            let nest = nest_pick % kernel.nests.len();
            let loops = &kernel.nests[nest].loops;
            prop_assume!(!loops.is_empty());
            let name = &loops[loop_pick % loops.len()].name;
            let split = Step::Split {
                nest,
                loop_name: name.clone(),
                outer: "so".to_string(),
                inner: "si".to_string(),
                factor,
            };
            let split_only = Schedule { steps: vec![split.clone()] };
            let names: Vec<String> = apply(&kernel, &split_only)
                .expect("split applies")
                .nests[nest]
                .loops
                .iter()
                .map(|l| l.name.clone())
                .collect();
            let order = permute(&names, &picks);
            let steps = vec![split, Step::Reorder { nest, order }];
            let (reordered, class) =
                apply_classified(&kernel, &Schedule { steps }).expect("reorder is legal");

            let (before, after) = (trace(&kernel), trace(&reordered));
            let (mut sorted_before, mut sorted_after) = (before.clone(), after.clone());
            sorted_before.sort();
            sorted_after.sort();
            prop_assert_eq!(sorted_after, sorted_before);

            let (before, after) = (by_element(before), by_element(after));
            prop_assert_eq!(before.keys().collect::<Vec<_>>(), after.keys().collect::<Vec<_>>());
            for (element, mut expected) in before {
                let mut got = after[&element].clone();
                if class == OrderClass::Reordering {
                    got.sort();
                    expected.sort();
                }
                prop_assert_eq!(got, expected, "element {:?}", element);
            }
        }
    }
}
