//! Applies a schedule to a kernel.

use std::fmt;

use crate::{Access, AffineExpr, Expr, Guard, Kernel, Loop, LoopIrError, Schedule, Step};

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
    /// The step is valid but not applied yet (reorder and vectorize come in later PRs).
    NotImplemented {
        step: usize,
        what: &'static str,
    },
    /// A factor, extent or scaled coefficient does not fit the loop IR's `i64`.
    NumberTooLarge {
        step: usize,
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
            Self::Invalid(e) => write!(f, "schedule produced an invalid kernel: {e}"),
        }
    }
}

impl std::error::Error for ScheduleError {}

/// Applies the steps in order to a copy of `kernel` and validates the result. On error
/// `kernel` is untouched, as it is only borrowed.
///
/// `split` keeps the order of all iterations, hence the order of every float operation,
/// so it is order-preserving: results stay bit for bit equal (spec §5.3).
pub fn apply(kernel: &Kernel, schedule: &Schedule) -> Result<Kernel, ScheduleError> {
    let mut result = kernel.clone();
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
            Step::Reorder { .. } => return Err(not_implemented(step_no, "reorder")),
            Step::Vectorize { .. } => return Err(not_implemented(step_no, "vectorize")),
        }
    }
    result.validate().map_err(ScheduleError::Invalid)?;
    Ok(result)
}

fn not_implemented(step: usize, what: &'static str) -> ScheduleError {
    ScheduleError::NotImplemented { step, what }
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
    let new_loops = [
        Loop {
            name: outer.to_string(),
            extent: extent.div_ceil(factor),
        },
        Loop {
            name: inner.to_string(),
            extent: factor,
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
        Kernel {
            name: "matmul".to_string(),
            buffers: vec![buf("A", &[4, 3]), buf("B", &[3, 2]), buf("C", &[4, 2])],
            nests: vec![
                nest(
                    vec![lp("i", 4), lp("j", 2)],
                    vec![var("i"), var("j")],
                    Expr::Const(0.0),
                ),
                nest(
                    vec![lp("i", 4), lp("k", 3), lp("j", 2)],
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
    fn reorder_and_vectorize_are_not_implemented() {
        assert_refused(
            &matmul(),
            &schedule("split(1, k, ko, ki, 2); reorder(1, i, j, ko, ki)"),
            ScheduleError::NotImplemented {
                step: 1,
                what: "reorder",
            },
        );
        assert_refused(
            &matmul(),
            &schedule("vectorize(1, j)"),
            ScheduleError::NotImplemented {
                step: 0,
                what: "vectorize",
            },
        );
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
    fn split_after_exact_split_with_remainder_is_rejected() {
        // Known limitation: a guard only bounds an index that has exactly its terms.
        // 9 = 3 * 3 splits without a guard; splitting the outer loop by 2 needs the guard
        // `2*sso + ssi < 3`, but the index is `6*sso + 3*ssi + si`, whose plain bound is 11.
        let mut g = Graph::new();
        let a = g.input("A", Shape::new(vec![9])).unwrap();
        let y = g.unary(coffee_ten_ir::UnaryOp::Neg, a).unwrap();
        let k = lower_elementwise(&g, y, "k").unwrap();
        let once = applied(&k, "split(0, i0, so, si, 3)");
        assert_refused(
            &once,
            &schedule("split(0, so, sso, ssi, 2)"),
            ScheduleError::Invalid(LoopIrError::OutOfBounds {
                nest: 0,
                buffer: "out".to_string(),
                dim: 0,
                min: 0,
                max: 11,
                size: 9,
            }),
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
            let twice = match apply(&once, &second) {
                Ok(twice) => twice,
                // Known limitation, see `split_after_exact_split_with_remainder_is_rejected`.
                Err(ScheduleError::Invalid(LoopIrError::OutOfBounds { .. })) => {
                    return Err(TestCaseError::reject("index not provable in bounds"));
                }
                Err(e) => return Err(TestCaseError::fail(e.to_string())),
            };
            prop_assert_eq!(trace(&twice), trace(&kernel));
        }
    }
}
