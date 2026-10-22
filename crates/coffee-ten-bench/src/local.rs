//! Model-guided local search over the matmul schedule space (spec §7.1).
//!
//! The first measurement is the schedule the analytic model ranks first. Each later
//! one is the unmeasured neighbor (`space::neighbors`) that the model ranks highest.
//! A measurement replaces the incumbent only when its median is strictly higher; a tie
//! keeps the incumbent. When every neighbor has been measured, the walk restarts at the
//! best-ranked schedule that is still unmeasured. It stops at `budget` measurements or
//! when the space is used up, and it never measures a schedule twice. The walk itself
//! does not use the search seed.

/// The next schedule to measure. `restart` means it is not a neighbor of the incumbent:
/// the neighborhood was used up, or there was no incumbent yet.
pub struct Move {
    pub index: usize,
    pub restart: bool,
}

/// The unmeasured neighbor of `incumbent` with the best model rank, or a restart at the
/// best-ranked unmeasured schedule. `rank_of[i]` is the model's rank of schedule `i`,
/// 0 the best. A tie in rank keeps the smaller index.
pub fn next_move(
    incumbent: Option<usize>,
    neighbors: &[Vec<usize>],
    rank_of: &[usize],
    measured: &[bool],
) -> Option<Move> {
    let restart = best_unmeasured(rank_of, measured).map(|index| Move {
        index,
        restart: true,
    });
    let Some(incumbent) = incumbent else {
        return restart;
    };
    neighbors
        .get(incumbent)
        .into_iter()
        .flatten()
        .copied()
        .filter(|index| !measured[*index])
        .min_by_key(|index| (rank_of[*index], *index))
        .map(|index| Move {
            index,
            restart: false,
        })
        .or(restart)
}

fn best_unmeasured(rank_of: &[usize], measured: &[bool]) -> Option<usize> {
    measured
        .iter()
        .enumerate()
        .filter(|(_, done)| !**done)
        .min_by_key(|(index, _)| (rank_of[*index], *index))
        .map(|(index, _)| index)
}

/// Measures at most `budget` schedules in hill-climb order. `measure` returns the median
/// of that schedule; a higher median is better. The returned indexes are the measurement
/// order.
pub fn local_search<E, F>(
    neighbors: &[Vec<usize>],
    rank_of: &[usize],
    budget: usize,
    mut measure: F,
) -> Result<Vec<usize>, E>
where
    F: FnMut(usize) -> Result<f64, E>,
{
    let mut measured = vec![false; rank_of.len()];
    let mut incumbent: Option<usize> = None;
    let mut incumbent_median: Option<f64> = None;
    let mut order = Vec::new();
    while order.len() < budget {
        let Some(step) = next_move(incumbent, neighbors, rank_of, &measured) else {
            break;
        };
        debug_assert!(
            !measured[step.index],
            "invariant: a schedule is measured once"
        );
        let median = measure(step.index)?;
        measured[step.index] = true;
        order.push(step.index);
        if step.restart || incumbent_median.is_none_or(|have| median > have) {
            incumbent = Some(step.index);
            incumbent_median = Some(median);
        }
    }
    Ok(order)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 0 connected to 1 and 3; 2 is off to the side. Ranks: 0, then 2, then 1, then 3.
    fn fork() -> (Vec<Vec<usize>>, Vec<usize>) {
        let neighbors = vec![vec![1, 3], vec![0], vec![], vec![0]];
        let rank_of = vec![0, 2, 1, 3];
        (neighbors, rank_of)
    }

    fn walk(medians: &[f64], budget: usize) -> Vec<usize> {
        let (neighbors, rank_of) = fork();
        local_search(&neighbors, &rank_of, budget, |index| {
            Ok::<f64, ()>(medians[index])
        })
        .unwrap()
    }

    #[test]
    fn an_equal_median_keeps_the_incumbent_so_the_walk_stays_with_its_neighbors() {
        // 1 ties the incumbent. Staying at 0 measures its other neighbor 3 next.
        // Moving on the tie would restart and measure 2, which the model ranks higher.
        assert_eq!(walk(&[10.0, 10.0, 10.0, 10.0], 4), vec![0, 1, 3, 2]);
    }

    #[test]
    fn a_faster_neighbor_becomes_the_incumbent_and_a_dead_end_restarts() {
        // 1 is faster, so the incumbent moves there. 1 has no unmeasured neighbor,
        // and the restart takes 2, the best rank still open, ahead of 3.
        assert_eq!(walk(&[10.0, 20.0, 10.0, 10.0], 4), vec![0, 1, 2, 3]);
    }

    #[test]
    fn the_walk_stops_at_the_budget_and_does_not_repeat_a_schedule() {
        let order = walk(&[1.0, 2.0, 3.0, 4.0], 2);
        assert_eq!(order, vec![0, 1]);
        let all = walk(&[1.0, 2.0, 3.0, 4.0], 100);
        assert_eq!(all.len(), 4);
        let mut seen = all.clone();
        seen.sort();
        seen.dedup();
        assert_eq!(seen.len(), 4);
    }

    #[test]
    fn a_tied_rank_keeps_the_smaller_index() {
        let neighbors = vec![vec![1, 2], vec![0], vec![0]];
        let rank_of = vec![0, 5, 5];
        let order =
            local_search(&neighbors, &rank_of, 2, |index| Ok::<f64, ()>(index as f64)).unwrap();
        assert_eq!(order, vec![0, 1]);
    }
}
