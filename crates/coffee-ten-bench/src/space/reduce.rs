//! Reduction schedules as data. The default lowerer and matmul spaces are unchanged.
use super::{Candidate, Space, SpaceConfig};
use crate::{BenchError, run::fnv1a_64};
use coffee_ten_ir::{Graph, ReduceOp, Shape};
use coffee_ten_loop::{Kernel, OrderClass, Schedule, Step, apply_classified, lower};
use std::fmt::Write as _;

pub fn enumerate_reduce(
    dims: &[usize],
    axis: usize,
    op: ReduceOp,
    config: SpaceConfig,
) -> Result<(Kernel, Space), BenchError> {
    let bad = |e: &dyn std::fmt::Display| BenchError::Usage(format!("invalid reduction: {e}"));
    let axis_i = isize::try_from(axis).map_err(|_| bad(&"axis is too large"))?;
    let mut graph = Graph::new();
    let x = graph
        .input("X", Shape::new(dims.to_vec()))
        .map_err(|e| bad(&e))?;
    let y = graph.reduce_axis(op, x, axis_i).map_err(|e| bad(&e))?;
    let kernel = lower(&graph, y, "space_reduce").map_err(|e| bad(&e))?;
    let mut space = Space {
        candidates: vec![Candidate {
            schedule: String::new(),
            class: OrderClass::Preserving,
        }],
        dropped: 0,
    };
    if config == SpaceConfig::V1 || axis + 1 == dims.len() {
        return Ok((kernel, space));
    }
    // Max over extent one has only an init nest. Never reorder it as if it were an update.
    if let Some((nest, update)) = kernel
        .nests
        .iter()
        .enumerate()
        .find(|(_, n)| n.loops.iter().any(|l| l.name == "r"))
    {
        let mut order: Vec<_> = update
            .loops
            .iter()
            .filter(|l| l.name != "r")
            .map(|l| l.name.clone())
            .collect();
        // Exactly `axis` kept dimensions precede the reduced dimension.
        order.insert(axis, "r".into());
        let schedule = Schedule {
            steps: vec![Step::Reorder { nest, order }],
        };
        let (_, class) =
            apply_classified(&kernel, &schedule).map_err(|e| BenchError::Setup(e.to_string()))?;
        if class != OrderClass::Preserving {
            return Err(BenchError::Setup(
                "outer-reduction schedule changed accumulation order".into(),
            ));
        }
        space.candidates.push(Candidate {
            schedule: schedule.to_string(),
            class,
        });
    }
    Ok((kernel, space))
}

/// `space reduce --shape 2,7,5 --axis 1 [--op sum|max] [--space v2]`.
pub fn reduce_command(args: impl Iterator<Item = String>) -> Result<String, BenchError> {
    let usage = |s: &str| {
        BenchError::Usage(format!(
            "{s}\nusage: coffee-ten-bench space reduce --shape <D0,D1,...> --axis <AXIS> [--op sum|max] [--space v1|v2]"
        ))
    };
    let mut args = args;
    let (mut dims, mut axis, mut op, mut config) = (None, None, ReduceOp::Sum, SpaceConfig::V1);
    while let Some(flag) = args.next() {
        let value = args.next().ok_or_else(|| usage("argument needs a value"))?;
        match flag.as_str() {
            "--shape" => {
                dims = Some(
                    value
                        .split(',')
                        .map(str::parse::<usize>)
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|_| usage("invalid shape"))?,
                )
            }
            "--axis" => {
                axis = Some(
                    value
                        .parse::<usize>()
                        .map_err(|_| usage("axis must be a non-negative integer"))?,
                )
            }
            "--op" => {
                op = match value.as_str() {
                    "sum" => ReduceOp::Sum,
                    "max" => ReduceOp::Max,
                    _ => return Err(usage("op must be sum or max")),
                }
            }
            "--space" => config = SpaceConfig::from_arg(&value)?,
            _ => return Err(usage("unknown argument")),
        }
    }
    let dims = dims.ok_or_else(|| usage("--shape is required"))?;
    let axis = axis.ok_or_else(|| usage("--axis is required"))?;
    let (kernel, space) = enumerate_reduce(&dims, axis, op, config)?;
    // A listing fingerprint, not a measurement-table protocol key.
    let mut bytes = format!("coffee-ten-reduce-space-{}\0{}", config.name(), kernel).into_bytes();
    for c in &space.candidates {
        bytes.extend_from_slice(&(c.schedule.len() as u64).to_le_bytes());
        bytes.extend_from_slice(c.schedule.as_bytes());
    }
    let mut text = format!(
        "reduce {op:?} {dims:?} axis {axis} space {}\nspace hash {:016x}\ncandidates {}\n",
        config.name(),
        fnv1a_64(&bytes),
        space.candidates.len()
    );
    for c in space.candidates {
        let _ = writeln!(
            text,
            "  {}",
            if c.schedule.is_empty() {
                "(naive)"
            } else {
                &c.schedule
            }
        );
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn outer_reduction_is_opt_in_and_only_moves_the_update_loop() {
        for (dims, axis, expected) in [
            (vec![7, 5], 0, "reorder(1, r, _i1)"),
            (vec![2, 7, 5], 1, "reorder(1, _i0, r, _i2)"),
        ] {
            for op in [ReduceOp::Sum, ReduceOp::Max] {
                let (original, v1) = enumerate_reduce(&dims, axis, op, SpaceConfig::V1).unwrap();
                let (same, v2) = enumerate_reduce(&dims, axis, op, SpaceConfig::V2(None)).unwrap();
                assert_eq!(original, same);
                assert_eq!(v1.candidates.len(), 1);
                assert_eq!(v2.candidates[1].schedule, expected);
                let (scheduled, class) =
                    apply_classified(&same, &Schedule::parse(expected).unwrap()).unwrap();
                assert_eq!(class, OrderClass::Preserving);
                assert_eq!(scheduled.nests[0], same.nests[0]);
                assert_eq!(scheduled.nests[1].body, same.nests[1].body);
            }
        }
    }
    #[test]
    fn last_axis_rank_one_and_max_singleton_have_no_redundant_candidate() {
        for (dims, axis, op) in [
            (vec![7], 0, ReduceOp::Sum),
            (vec![2, 7], 1, ReduceOp::Sum),
            (vec![1, 5], 0, ReduceOp::Max),
        ] {
            assert_eq!(
                enumerate_reduce(&dims, axis, op, SpaceConfig::V2(None))
                    .unwrap()
                    .1
                    .candidates
                    .len(),
                1
            );
        }
    }
    #[test]
    fn cli_lists_only_and_rejects_invalid_shapes_axes_and_ops() {
        let args = ["--shape", "2,7,5", "--axis", "1"];
        assert!(
            reduce_command(args.map(str::to_string).into_iter())
                .unwrap()
                .contains("candidates 1")
        );
        assert!(
            reduce_command(
                args.into_iter()
                    .chain(["--space", "v2"])
                    .map(str::to_string)
            )
            .unwrap()
            .contains("reorder(1, _i0, r, _i2)")
        );
        for args in [
            vec!["--shape", "2,0", "--axis", "0"],
            vec!["--shape", "2,3", "--axis", "2"],
            vec!["--shape", "2,3", "--axis", "0", "--op", "mean"],
        ] {
            assert!(reduce_command(args.into_iter().map(str::to_string)).is_err());
        }
    }
}
