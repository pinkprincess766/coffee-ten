//! The space of legal schedules of the matmul update nest, as data (no timing).
//!
//! The update nest (nest 1) runs `i, k, j`. A candidate picks a tile factor per loop,
//! an order of the loops (all outer loops before all inner loops) and optionally
//! `vectorize` on the innermost `j` loop. Only factors that divide the extent are used,
//! so no candidate has a guard.
//!
//! After those, a few register tiles are added (`add_register_tiles`): `i` and `j` split
//! by `ri` and `rj`, order `io, jo, k, ii, ji`, `ji` vectorized, `ii` and `ji` unrolled,
//! so the `ri x rj` tile of C stays in vector registers across `k`. Only this one shape:
//! no `k` split, no other orders, no other unroll choices.
//!
//! `neighbors` is one step of local search on that set. A step is one of: an adjacent
//! swap inside the outer block or inside the inner block; one tile factor moved to the
//! next entry of `tile_options` (splitting tries each index of the inner block); the
//! `vectorize` bit, when the innermost loop is `j` or
//! `ji`; the register-tile `unroll` pair, added or removed together. A swap or a factor
//! step that would change the innermost loop while it is vectorized is not a step: turn
//! `vectorize` off first. A register tile steps only to an adjacent `ri` or `rj`, or to
//! the same schedule without `unroll`.

use crate::cli::MAX_N;
use coffee_ten_ir::{Graph, Shape};
use coffee_ten_loop::{Kernel, OrderClass, Schedule, Step, apply_classified, lower};
use std::collections::HashMap;
use std::fmt::Write as _;

/// Tile factors tried per loop; a loop can also stay unsplit.
pub const TILE_FACTORS: [usize; 5] = [8, 16, 32, 64, 128];
/// Register tile heights (`ii`) and widths (`ji`) tried, see `add_register_tiles`.
const REG_TILE_I: [usize; 3] = [2, 4, 8];
const REG_TILE_J: [usize; 3] = [8, 16, 32];
/// f32 lanes of one NEON vector register; a register tile row is `rj / LANES` vectors.
const LANES: usize = 4;
/// Most vector accumulators (`ri * rj / LANES`) a register tile may need. ASSUMED, not
/// measured: NEON has 32 vector registers; 8 are left for the B vectors and the A
/// broadcast of the current `k`.
const MAX_ACCUMULATORS: usize = 24;
/// The matmul update nest; nest 0 is the init nest.
const UPDATE_NEST: usize = 1;
/// How many schedule texts `summary` lists.
pub const SHOWN: usize = 10;

/// A loop that can be split: its name and the names of its two halves.
struct Axis {
    name: &'static str,
    outer: &'static str,
    inner: &'static str,
}

/// In the order the splits are written: `i`, `j`, `k`.
const AXES: [Axis; 3] = [
    Axis {
        name: "i",
        outer: "io",
        inner: "ii",
    },
    Axis {
        name: "j",
        outer: "jo",
        inner: "ji",
    },
    Axis {
        name: "k",
        outer: "ko",
        inner: "ki",
    },
];

/// A legal schedule of the update nest in canonical text, and how it orders float sums.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Candidate {
    pub schedule: String,
    pub class: OrderClass,
}

/// The legal candidates in a fixed order, and how many built schedules `apply` refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Space {
    pub candidates: Vec<Candidate>,
    pub dropped: usize,
}

/// `None` (do not split) and every factor of `TILE_FACTORS` that is less than `extent`
/// and divides it.
fn tile_options(extent: usize) -> Vec<Option<usize>> {
    let divisors = TILE_FACTORS
        .into_iter()
        .filter(|&factor| factor < extent && extent.is_multiple_of(factor));
    std::iter::once(None).chain(divisors.map(Some)).collect()
}

/// All orderings of `items`, in the order a recursive pick-the-first-then-the-rest gives.
fn permutations(items: &[&'static str]) -> Vec<Vec<&'static str>> {
    if items.len() <= 1 {
        return vec![items.to_vec()];
    }
    let mut all = Vec::new();
    for (pos, &first) in items.iter().enumerate() {
        let mut rest = items.to_vec();
        rest.remove(pos);
        for tail in permutations(&rest) {
            all.push([&[first][..], &tail].concat());
        }
    }
    all
}

fn matmul_kernel(m: usize, k: usize, n: usize) -> Option<Kernel> {
    let mut g = Graph::new();
    let a = g.input("A", Shape::new(vec![m, k])).ok()?;
    let b = g.input("B", Shape::new(vec![k, n])).ok()?;
    let product = g.matmul(a, b).ok()?;
    lower(&g, product, "space_kernel").ok()
}

/// Every legal schedule of the `m x k` by `k x n` matmul under the rules in the module
/// docs. Empty if the matmul does not lower (a zero dimension).
pub fn enumerate_matmul(m: usize, k: usize, n: usize) -> Space {
    let mut space = Space {
        candidates: Vec::new(),
        dropped: 0,
    };
    let Some(naive) = matmul_kernel(m, k, n) else {
        return space;
    };
    for tile_i in tile_options(m) {
        for tile_j in tile_options(n) {
            for tile_k in tile_options(k) {
                add_for_tiles(&naive, [tile_i, tile_j, tile_k], &mut space);
            }
        }
    }
    add_register_tiles(&naive, m, n, &mut space);
    space
}

/// Adds, for each `ri` of `REG_TILE_I` and `rj` of `REG_TILE_J` that divide `m` and `n`
/// (`rj` a multiple of `LANES`) and need at most `MAX_ACCUMULATORS` accumulators, the
/// schedule `split i by ri; split j by rj; reorder io, jo, k, ii, ji; vectorize ji;
/// unroll ii; unroll ji`. The text has `unroll`, which the tile enumeration never writes,
/// so none of these is also listed there.
fn add_register_tiles(naive: &Kernel, m: usize, n: usize, space: &mut Space) {
    let step_split = |axis: &Axis, factor| Step::Split {
        nest: UPDATE_NEST,
        loop_name: axis.name.to_string(),
        outer: axis.outer.to_string(),
        inner: axis.inner.to_string(),
        factor,
    };
    for ri in REG_TILE_I {
        for rj in REG_TILE_J {
            let fits = m.is_multiple_of(ri)
                && n.is_multiple_of(rj)
                && rj.is_multiple_of(LANES)
                && ri * rj / LANES <= MAX_ACCUMULATORS;
            if !fits {
                continue;
            }
            let order = ["io", "jo", "k", "ii", "ji"];
            let steps = [
                step_split(&AXES[0], ri),
                step_split(&AXES[1], rj),
                Step::Reorder {
                    nest: UPDATE_NEST,
                    order: order.iter().map(|name| name.to_string()).collect(),
                },
                Step::Vectorize {
                    nest: UPDATE_NEST,
                    loop_name: "ji".to_string(),
                },
                Step::Unroll {
                    nest: UPDATE_NEST,
                    loop_name: "ii".to_string(),
                },
                Step::Unroll {
                    nest: UPDATE_NEST,
                    loop_name: "ji".to_string(),
                },
            ];
            try_add(naive, &steps, space);
        }
    }
}

/// Adds the candidates whose loops are split by `tiles` (in `AXES` order).
fn add_for_tiles(naive: &Kernel, tiles: [Option<usize>; 3], space: &mut Space) {
    let mut splits = Vec::new();
    let mut outer_names = Vec::new();
    let mut inner_names = Vec::new();
    for (axis, tile) in AXES.iter().zip(tiles) {
        match tile {
            Some(factor) => {
                splits.push(Step::Split {
                    nest: UPDATE_NEST,
                    loop_name: axis.name.to_string(),
                    outer: axis.outer.to_string(),
                    inner: axis.inner.to_string(),
                    factor,
                });
                outer_names.push(axis.outer);
                inner_names.push(axis.inner);
            }
            None => outer_names.push(axis.name),
        }
    }
    for outer_order in permutations(&outer_names) {
        for inner_order in permutations(&inner_names) {
            let order: Vec<&str> = [outer_order.clone(), inner_order].concat();
            let innermost = *order.last().expect("invariant: the nest has three loops");
            let reorder = Step::Reorder {
                nest: UPDATE_NEST,
                order: order.iter().map(|name| name.to_string()).collect(),
            };
            let mut steps = splits.clone();
            steps.push(reorder);
            try_add(naive, &steps, space);
            if innermost == "j" || innermost == "ji" {
                steps.push(Step::Vectorize {
                    nest: UPDATE_NEST,
                    loop_name: innermost.to_string(),
                });
                try_add(naive, &steps, space);
            }
        }
    }
}

fn try_add(naive: &Kernel, steps: &[Step], space: &mut Space) {
    let schedule = Schedule {
        steps: steps.to_vec(),
    };
    match apply_classified(naive, &schedule) {
        Ok((_, class)) => space.candidates.push(Candidate {
            schedule: schedule.to_string(),
            class,
        }),
        Err(_) => space.dropped += 1,
    }
}

/// The legal candidates of `enumerate_matmul`.
pub fn matmul_space(m: usize, k: usize, n: usize) -> Vec<Candidate> {
    enumerate_matmul(m, k, n).candidates
}

/// Parses the arguments after `space`: `--n <N>`, the size of the square matmul.
pub fn parse_space_args<I: IntoIterator<Item = String>>(
    args: I,
) -> Result<usize, crate::BenchError> {
    let usage = |msg: String| {
        crate::BenchError::Usage(format!("{msg}\nusage: coffee-ten-bench space --n <N>"))
    };
    let mut args = args.into_iter();
    let mut n = None;
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| usage(format!("{flag} needs a value")))?;
        match flag.as_str() {
            "--n" => {
                let parsed = value.parse::<usize>().map_err(|_| {
                    usage(format!("--n needs a non-negative integer, got {value:?}"))
                })?;
                n = Some(parsed);
            }
            _ => return Err(usage(format!("unknown argument {flag}"))),
        }
    }
    let n = n.ok_or_else(|| usage("--n is required".to_string()))?;
    if !(1..=MAX_N).contains(&n) {
        return Err(usage(format!("--n must be in 1..={MAX_N}, got {n}")));
    }
    Ok(n)
}

/// One local-search step from each candidate: indexes into `candidates`, no duplicates,
/// never the candidate itself. A move whose schedule is not in `candidates` is dropped.
/// `m`, `k` and `n` are the matmul extents, the same ones `enumerate_matmul` took.
pub fn neighbors(candidates: &[Candidate], m: usize, k: usize, n: usize) -> Vec<Vec<usize>> {
    let index_of: HashMap<&str, usize> = candidates
        .iter()
        .enumerate()
        .map(|(index, candidate)| (candidate.schedule.as_str(), index))
        .collect();
    // Axis 0 is `i` (extent m), axis 1 is `j` (extent n), axis 2 is `k`.
    let extents = [m, n, k];
    candidates
        .iter()
        .enumerate()
        .map(|(current, candidate)| {
            let point = point_of(&candidate.schedule);
            let encoded = text_of(&point);
            assert_eq!(
                encoded, candidate.schedule,
                "invariant: a space schedule round-trips"
            );
            let mut row = Vec::new();
            for moved in moves_of(&point, extents) {
                let text = text_of(&moved);
                if let Some(&index) = index_of.get(text.as_str())
                    && index != current
                    && !row.contains(&index)
                {
                    row.push(index);
                }
            }
            row
        })
        .collect()
}

/// The decisions a candidate is built from. `factor` is in `AXES` order (`i`, `j`, `k`).
#[derive(Clone, Debug, PartialEq, Eq)]
struct Point {
    factor: [Option<usize>; 3],
    outer: Vec<String>,
    inner: Vec<String>,
    vectorize: bool,
    register: bool,
}

fn is_inner_loop(name: &str) -> bool {
    matches!(name, "ii" | "ji" | "ki")
}

fn innermost(point: &Point) -> &str {
    point
        .inner
        .last()
        .or_else(|| point.outer.last())
        .expect("invariant: a nest has a loop")
}

fn names_eq(names: &[String], expect: &[&str]) -> bool {
    names.iter().map(String::as_str).eq(expect.iter().copied())
}

/// The point `text` was built from. `text` is a space candidate: it has a reorder, and
/// an `unroll` only as the register-tile pair.
fn point_of(text: &str) -> Point {
    let schedule = Schedule::parse(text).expect("invariant: candidate texts parse");
    let mut factor = [None; 3];
    let mut order = None;
    let mut vectorize = false;
    let mut unroll = Vec::new();
    for step in schedule.steps {
        match step {
            Step::Split {
                nest,
                loop_name,
                outer,
                inner,
                factor: value,
            } => {
                assert_eq!(
                    nest, UPDATE_NEST,
                    "invariant: splits are on the update nest"
                );
                let axis = AXES
                    .iter()
                    .position(|axis| axis.name == loop_name)
                    .expect("invariant: a split names i, j or k");
                assert_eq!(outer, AXES[axis].outer, "invariant: split outer name");
                assert_eq!(inner, AXES[axis].inner, "invariant: split inner name");
                factor[axis] = Some(value);
            }
            Step::Reorder { nest, order: names } => {
                assert_eq!(
                    nest, UPDATE_NEST,
                    "invariant: reorder is on the update nest"
                );
                order = Some(names);
            }
            Step::Vectorize { nest, .. } => {
                assert_eq!(
                    nest, UPDATE_NEST,
                    "invariant: vectorize is on the update nest"
                );
                vectorize = true;
            }
            Step::Unroll { nest, loop_name } => {
                assert_eq!(nest, UPDATE_NEST, "invariant: unroll is on the update nest");
                unroll.push(loop_name);
            }
        }
    }
    let order = order.expect("invariant: a space candidate has a reorder");
    let first_inner = order
        .iter()
        .position(|name| is_inner_loop(name))
        .unwrap_or(order.len());
    let point = Point {
        factor,
        outer: order[..first_inner].to_vec(),
        inner: order[first_inner..].to_vec(),
        vectorize,
        register: !unroll.is_empty(),
    };
    if vectorize {
        let name = innermost(&point);
        assert!(
            name == "j" || name == "ji",
            "invariant: vectorize is on j or ji, got {name} in {text}"
        );
    }
    if point.register {
        unroll.sort();
        assert_eq!(
            unroll,
            ["ii", "ji"],
            "invariant: unroll is the register-tile pair"
        );
    }
    point
}

fn text_of(point: &Point) -> String {
    let mut steps = Vec::new();
    for (axis, factor) in AXES.iter().zip(point.factor) {
        if let Some(factor) = factor {
            steps.push(Step::Split {
                nest: UPDATE_NEST,
                loop_name: axis.name.to_string(),
                outer: axis.outer.to_string(),
                inner: axis.inner.to_string(),
                factor,
            });
        }
    }
    let order = point.outer.iter().chain(&point.inner).cloned().collect();
    steps.push(Step::Reorder {
        nest: UPDATE_NEST,
        order,
    });
    if point.vectorize {
        steps.push(Step::Vectorize {
            nest: UPDATE_NEST,
            loop_name: innermost(point).to_string(),
        });
    }
    if point.register {
        for name in ["ii", "ji"] {
            steps.push(Step::Unroll {
                nest: UPDATE_NEST,
                loop_name: name.to_string(),
            });
        }
    }
    Schedule { steps }.to_string()
}

fn push_unique(out: &mut Vec<Point>, point: Point) {
    if !out.contains(&point) {
        out.push(point);
    }
}

/// `point.factor[axis]` replaced by the adjacent option `next`. Splitting inserts the new
/// inner loop at every index of the inner block, so the step is the inverse of removing
/// that loop wherever it sat. A result that would change the innermost loop while it is
/// vectorized is dropped.
fn retile(point: &Point, axis: usize, next: Option<usize>) -> Vec<Point> {
    let old = point.factor[axis];
    if point.register || old == next {
        return Vec::new();
    }
    let single = AXES[axis].name;
    let outer_name = AXES[axis].outer;
    let inner_name = AXES[axis].inner;
    let mut base = point.clone();
    base.factor[axis] = next;
    let moved = match (old, next) {
        (None, Some(_)) => {
            let Some(index) = base.outer.iter().position(|name| name == single) else {
                return Vec::new();
            };
            base.outer[index] = outer_name.to_string();
            (0..=base.inner.len())
                .map(|slot| {
                    let mut one = base.clone();
                    one.inner.insert(slot, inner_name.to_string());
                    one
                })
                .collect()
        }
        (Some(_), None) => {
            let Some(index) = base.outer.iter().position(|name| name == outer_name) else {
                return Vec::new();
            };
            base.outer[index] = single.to_string();
            let Some(index) = base.inner.iter().position(|name| name == inner_name) else {
                return Vec::new();
            };
            base.inner.remove(index);
            vec![base]
        }
        (Some(_), Some(_)) => vec![base],
        (None, None) => Vec::new(),
    };
    let innermost_before = innermost(point);
    moved
        .into_iter()
        .filter(|one| !point.vectorize || innermost(one) == innermost_before)
        .collect()
}

fn factor_moves(point: &Point, extents: [usize; 3]) -> Vec<Point> {
    let mut out = Vec::new();
    if point.register {
        return out;
    }
    for (axis, &extent) in extents.iter().enumerate() {
        let options = tile_options(extent);
        let Some(pos) = options
            .iter()
            .position(|factor| *factor == point.factor[axis])
        else {
            continue;
        };
        for next_pos in [pos.checked_sub(1), pos.checked_add(1)] {
            let Some(next_pos) = next_pos else {
                continue;
            };
            let Some(&next) = options.get(next_pos) else {
                continue;
            };
            for moved in retile(point, axis, next) {
                push_unique(&mut out, moved);
            }
        }
    }
    out
}

fn swap_moves(point: &Point) -> Vec<Point> {
    let mut out = Vec::new();
    if point.register {
        return out;
    }
    for i in 0..point.outer.len().saturating_sub(1) {
        let mut moved = point.clone();
        moved.outer.swap(i, i + 1);
        if point.vectorize && innermost(&moved) != innermost(point) {
            continue;
        }
        push_unique(&mut out, moved);
    }
    for i in 0..point.inner.len().saturating_sub(1) {
        let mut moved = point.clone();
        moved.inner.swap(i, i + 1);
        if point.vectorize && innermost(&moved) != innermost(point) {
            continue;
        }
        push_unique(&mut out, moved);
    }
    out
}

fn vectorize_move(point: &Point) -> Option<Point> {
    if point.register {
        return None;
    }
    let name = innermost(point);
    if name != "j" && name != "ji" {
        return None;
    }
    let mut moved = point.clone();
    moved.vectorize = !moved.vectorize;
    Some(moved)
}

fn is_register_twin(point: &Point) -> bool {
    let (Some(ri), Some(rj), None) = (point.factor[0], point.factor[1], point.factor[2]) else {
        return false;
    };
    !point.register
        && point.vectorize
        && REG_TILE_I.contains(&ri)
        && REG_TILE_J.contains(&rj)
        && names_eq(&point.outer, &["io", "jo", "k"])
        && names_eq(&point.inner, &["ii", "ji"])
}

fn adjacent_values(values: &[usize], current: usize) -> impl Iterator<Item = usize> {
    let pos = values.iter().position(|&value| value == current);
    let earlier = pos
        .and_then(|pos| pos.checked_sub(1))
        .map(|pos| values[pos]);
    let later = pos.and_then(|pos| values.get(pos + 1).copied());
    earlier.into_iter().chain(later)
}

/// Register tiles step to an adjacent `ri` or `rj`, or drop both unrolls. The matching
/// schedule without unrolls steps back. A tile the space does not contain is dropped
/// later, by the lookup in `neighbors`.
fn register_moves(point: &Point) -> Vec<Point> {
    let mut out = Vec::new();
    if point.register {
        let mut twin = point.clone();
        twin.register = false;
        out.push(twin);
        let ri = point.factor[0].expect("invariant: a register tile splits i");
        let rj = point.factor[1].expect("invariant: a register tile splits j");
        for next_ri in adjacent_values(&REG_TILE_I, ri) {
            let mut moved = point.clone();
            moved.factor[0] = Some(next_ri);
            out.push(moved);
        }
        for next_rj in adjacent_values(&REG_TILE_J, rj) {
            let mut moved = point.clone();
            moved.factor[1] = Some(next_rj);
            out.push(moved);
        }
    } else if is_register_twin(point) {
        let mut tiled = point.clone();
        tiled.register = true;
        out.push(tiled);
    }
    out
}

fn moves_of(point: &Point, extents: [usize; 3]) -> Vec<Point> {
    let mut out = factor_moves(point, extents);
    out.extend(swap_moves(point));
    if let Some(toggled) = vectorize_move(point) {
        push_unique(&mut out, toggled);
    }
    for moved in register_moves(point) {
        push_unique(&mut out, moved);
    }
    out
}

/// The report of `coffee-ten-bench space`: counts by class and the first `SHOWN` texts.
pub fn summary(n: usize, space: &Space) -> String {
    let count = |class| space.candidates.iter().filter(|c| c.class == class).count();
    let mut text = String::new();
    let _ = writeln!(text, "matmul {n}x{n}x{n}, update nest 1");
    let _ = writeln!(
        text,
        "candidates {}  (dropped as illegal: {})",
        space.candidates.len(),
        space.dropped
    );
    let _ = writeln!(text, "  preserving {}", count(OrderClass::Preserving));
    let _ = writeln!(text, "  reordering {}", count(OrderClass::Reordering));
    let _ = writeln!(text, "first {}:", SHOWN.min(space.candidates.len()));
    for candidate in space.candidates.iter().take(SHOWN) {
        let _ = writeln!(text, "  {}", candidate.schedule);
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn steps_of(text: &str) -> Vec<Step> {
        Schedule::parse(text).expect("candidate text parses").steps
    }

    fn reorder_of(text: &str) -> Vec<String> {
        steps_of(text)
            .into_iter()
            .find_map(|step| match step {
                Step::Reorder { order, .. } => Some(order),
                _ => None,
            })
            .expect("every candidate has a reorder")
    }

    /// Shapes: cubes, and one where factors 16 and 32 divide none or some of the dims
    /// (m = 48, k = 40, n = 24), so a missing divisibility filter would give guards.
    const SHAPES: [(usize, usize, usize); 5] = [
        (16, 16, 16),
        (64, 64, 64),
        (256, 256, 256),
        (48, 40, 24),
        (8, 9, 130),
    ];

    #[test]
    fn tile_options_are_less_than_the_extent_and_divide_it() {
        assert_eq!(tile_options(8), [None]);
        assert_eq!(tile_options(9), [None]);
        assert_eq!(tile_options(16), [None, Some(8)]);
        assert_eq!(tile_options(17), [None]);
        assert_eq!(tile_options(48), [None, Some(8), Some(16)]);
        assert_eq!(
            tile_options(256),
            [None, Some(8), Some(16), Some(32), Some(64), Some(128)]
        );
        assert_eq!(tile_options(129), [None]);
    }

    #[test]
    fn every_candidate_applies_validates_and_has_no_guard() {
        let mut checked = 0;
        for (m, k, n) in SHAPES {
            let naive = matmul_kernel(m, k, n).expect("lowers");
            for candidate in matmul_space(m, k, n) {
                let schedule = Schedule::parse(&candidate.schedule).expect("parses");
                assert_eq!(schedule.to_string(), candidate.schedule, "not canonical");
                let (kernel, class) = apply_classified(&naive, &schedule).expect("applies");
                kernel.validate().expect("validates");
                assert_eq!(class, candidate.class, "{}", candidate.schedule);
                let guards = &kernel.nests[UPDATE_NEST].guards;
                assert!(
                    guards.is_empty(),
                    "{m}x{k}x{n} {}: guards {guards:?}",
                    candidate.schedule
                );
                checked += 1;
            }
        }
        assert!(checked > 100, "only {checked} candidates");
    }

    #[test]
    fn no_candidate_is_dropped_and_none_is_listed_twice() {
        for (m, k, n) in SHAPES {
            let space = enumerate_matmul(m, k, n);
            assert_eq!(space.dropped, 0, "{m}x{k}x{n}");
            let texts: HashSet<&str> = space
                .candidates
                .iter()
                .map(|c| c.schedule.as_str())
                .collect();
            assert_eq!(texts.len(), space.candidates.len(), "{m}x{k}x{n}");
        }
    }

    #[test]
    fn the_space_is_deterministic() {
        assert_eq!(matmul_space(64, 64, 64), matmul_space(64, 64, 64));
        assert_eq!(enumerate_matmul(48, 40, 24), enumerate_matmul(48, 40, 24));
    }

    /// N = 16: the only factor is 8 (8 < 16, 8 divides 16; 16 is not < 16), so each of
    /// ti, tj, tk is `none` or 8. There are always 3 outer loops (a split loop is `Xo`, an
    /// unsplit one counts as outer) and s inner loops, where s is the number of splits;
    /// so 3! * s! orders. Vectorize off keeps every order. On needs the innermost loop to
    /// be `j` or `ji`:
    /// - s = 0 (1 combo): 6 orders, on for the 2 with `j` last: 6 + 6 + 2 = 8 candidates.
    /// - s = 1 (3 combos): 6 orders each. Split i or k: `j` is outer, so 6 each, off only.
    ///   Split j: the one inner loop is `ji`, on for all 6: 6 + 6 = 12. Total 6 + 6 + 12 = 24.
    /// - s = 2 (3 combos): 6 * 2 = 12 orders each. {i,k}: 12. {i,j} and {j,k}: `ji` is last
    ///   in 1 of the 2 inner orders, so on for 6: 12 + 6 = 18 each. Total 12 + 18 + 18 = 48.
    /// - s = 3 (1 combo): 6 * 6 = 36 orders, `ji` last in 2 of the 6 inner orders, on for
    ///   6 * 2 = 12: 36 + 12 = 48.
    ///
    /// Sum: 8 + 24 + 48 + 48 = 128.
    ///
    /// Register tiles: `ri` in {2, 4, 8} all divide 16, `rj` in {8, 16, 32} gives 8 and 16
    /// (32 does not divide 16). Accumulators `ri * rj / 4`: (2,8) 4, (2,16) 8, (4,8) 8,
    /// (4,16) 16, (8,8) 16, (8,16) 32 > 24. So 5 register tiles: 128 + 5 = 133.
    #[test]
    fn size_16_has_133_candidates() {
        let space = enumerate_matmul(16, 16, 16);
        assert_eq!(space.candidates.len(), 133);
        assert_eq!(space.dropped, 0);
    }

    #[test]
    fn a_size_without_tile_factors_has_only_orders_and_vectorize() {
        // 6 orders of i, j, k, plus vectorize for the 2 with `j` last: 8. Register tiles:
        // ri in {2, 4, 8} divide 8, rj = 8 only; accumulators 4, 8, 16 <= 24: 3 more.
        assert_eq!(matmul_space(8, 8, 8).len(), 8 + 3);
        assert!(matmul_space(0, 8, 8).is_empty());
    }

    #[test]
    fn the_tiled_schedule_of_bench_run_sh_is_in_the_size_256_space() {
        let tiled =
            "split(1, i, io, ii, 32); split(1, j, jo, ji, 64); reorder(1, io, jo, k, ii, ji)";
        let tiled_vec = format!("{tiled}; vectorize(1, ji)");
        assert_eq!(Schedule::parse(tiled).expect("parses").to_string(), tiled);
        let space = matmul_space(256, 256, 256);
        for text in [tiled, tiled_vec.as_str()] {
            let found: Vec<&Candidate> = space.iter().filter(|c| c.schedule == text).collect();
            assert_eq!(found.len(), 1, "{text}");
            assert_eq!(found[0].class, OrderClass::Preserving, "{text}");
        }
    }

    #[test]
    fn the_4x16_register_tile_of_bench_run_sh_is_in_the_size_256_space() {
        let tile = "split(1, i, io, ii, 4); split(1, j, jo, ji, 16); reorder(1, io, jo, k, ii, ji)";
        let text = format!("{tile}; vectorize(1, ji); unroll(1, ii); unroll(1, ji)");
        assert_eq!(Schedule::parse(&text).expect("parses").to_string(), text);
        let space = matmul_space(256, 256, 256);
        let found: Vec<&Candidate> = space.iter().filter(|c| c.schedule == text).collect();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].class, OrderClass::Preserving);
    }

    /// The register tiles of a space are its candidates with an `unroll`. Per shape, the
    /// pairs (ri, rj) with ri | m, rj | n and ri * rj / 4 <= 24:
    /// - 256 and 64: every ri, rj divides. Accumulators: (2,8) 4, (2,16) 8, (2,32) 16,
    ///   (4,8) 8, (4,16) 16, (4,32) 32, (8,8) 16, (8,16) 32, (8,32) 64: 6 are <= 24.
    /// - 16x16x16: 5 (see `size_16_has_133_candidates`). 8x8x8: 3.
    /// - m = 48, n = 24: ri in {2, 4, 8} divide 48; rj = 8 only (24 % 16 != 0): 3.
    /// - m = 8, n = 130: 130 % 8 != 0, so no rj: 0.
    #[test]
    fn register_tiles_are_counted_by_hand() {
        let count = |m, k, n| {
            matmul_space(m, k, n)
                .iter()
                .filter(|c| c.schedule.contains("unroll("))
                .count()
        };
        assert_eq!(count(256, 256, 256), 6);
        assert_eq!(count(64, 64, 64), 6);
        assert_eq!(count(16, 16, 16), 5);
        assert_eq!(count(8, 8, 8), 3);
        assert_eq!(count(48, 40, 24), 3);
        assert_eq!(count(8, 9, 130), 0);
    }

    #[test]
    fn no_candidate_has_more_than_24_accumulators() {
        let mut tiles = 0;
        for (m, k, n) in SHAPES {
            for candidate in matmul_space(m, k, n) {
                let steps = steps_of(&candidate.schedule);
                if !steps.iter().any(|s| matches!(s, Step::Unroll { .. })) {
                    continue;
                }
                let factor_of = |axis: &str| {
                    steps.iter().find_map(|step| match step {
                        Step::Split {
                            loop_name, factor, ..
                        } if loop_name == axis => Some(*factor),
                        _ => None,
                    })
                };
                let (ri, rj) = (
                    factor_of("i").expect("split i"),
                    factor_of("j").expect("split j"),
                );
                assert!(rj % 4 == 0, "{}", candidate.schedule);
                assert!(ri * rj / 4 <= 24, "{}: {ri}x{rj}", candidate.schedule);
                tiles += 1;
            }
        }
        // 5 + 6 + 6 + 3 + 0 tiles over SHAPES, so the loop above did run.
        assert_eq!(tiles, 20);
    }

    #[test]
    fn outer_loops_come_first_so_ko_before_ki_and_the_class_is_preserving() {
        let space = matmul_space(16, 16, 16);
        let with_both: Vec<&Candidate> = space
            .iter()
            .filter(|c| {
                let order = reorder_of(&c.schedule);
                order.iter().any(|l| l == "ko") && order.iter().any(|l| l == "ki")
            })
            .collect();
        assert!(with_both.len() > 20, "{}", with_both.len());
        for candidate in &with_both {
            let order = reorder_of(&candidate.schedule);
            let pos = |name: &str| order.iter().position(|l| l == name).expect("present");
            assert!(pos("ko") < pos("ki"), "{}", candidate.schedule);
            assert_eq!(
                candidate.class,
                OrderClass::Preserving,
                "{}",
                candidate.schedule
            );
        }
        // The check can fail: `ki` outside `ko` is Reordering, and the space never has it.
        let naive = matmul_kernel(16, 16, 16).expect("lowers");
        let swapped =
            Schedule::parse("split(1, k, ko, ki, 8); reorder(1, i, ki, ko, j)").expect("parses");
        let (_, class) = apply_classified(&naive, &swapped).expect("applies");
        assert_eq!(class, OrderClass::Reordering);
    }

    #[test]
    fn outer_loops_precede_inner_loops_and_vectorize_is_on_j() {
        for candidate in matmul_space(64, 64, 64) {
            let order = reorder_of(&candidate.schedule);
            let is_inner = |l: &String| matches!(l.as_str(), "ii" | "ki" | "ji");
            let first_inner = order.iter().position(is_inner).unwrap_or(order.len());
            assert!(
                order[first_inner..].iter().all(is_inner),
                "{}",
                candidate.schedule
            );
            if let Some(Step::Vectorize { loop_name, .. }) = steps_of(&candidate.schedule).last() {
                assert_eq!(Some(loop_name), order.last(), "{}", candidate.schedule);
                assert!(
                    loop_name == "j" || loop_name == "ji",
                    "{}",
                    candidate.schedule
                );
            }
        }
    }

    #[test]
    fn summary_lists_counts_and_the_first_ten() {
        let space = enumerate_matmul(16, 16, 16);
        let text = summary(16, &space);
        assert!(
            text.contains("candidates 133  (dropped as illegal: 0)"),
            "{text}"
        );
        assert!(text.contains("preserving 133"), "{text}");
        assert!(text.contains("reordering 0"), "{text}");
        assert_eq!(text.matches("reorder(1,").count(), SHOWN, "{text}");
    }

    #[test]
    fn space_args_limits_are_checked_on_both_sides() {
        let parse = |args: &[&str]| parse_space_args(args.iter().map(|a| a.to_string()));
        assert_eq!(parse(&["--n", "256"]), Ok(256));
        assert!(parse(&["--n", "0"]).is_err());
        assert_eq!(parse(&["--n", "1"]), Ok(1));
        assert_eq!(parse(&["--n", &MAX_N.to_string()]), Ok(MAX_N));
        assert!(parse(&["--n", &(MAX_N + 1).to_string()]).is_err());
        assert!(parse(&[]).is_err());
        assert!(parse(&["--n"]).is_err());
        assert!(parse(&["--n", "x"]).is_err());
        assert!(parse(&["--n", "8", "--kernel", "matmul"]).is_err());
    }

    fn texts_of(indexes: &[usize], space: &[Candidate]) -> Vec<String> {
        let mut texts: Vec<String> = indexes
            .iter()
            .map(|&index| space[index].schedule.clone())
            .collect();
        texts.sort();
        texts
    }

    #[test]
    fn the_unsplit_ijk_schedule_has_its_six_hand_neighbors_at_n_16() {
        // Factors of 16: none and 8. From `i, k, j` with nothing split and vectorize off:
        // two adjacent outer swaps, vectorize on `j`, and one split of each axis.
        let space = matmul_space(16, 16, 16);
        let start = "reorder(1, i, k, j)";
        let index = space
            .iter()
            .position(|c| c.schedule == start)
            .expect("present");
        let row = neighbors(&space, 16, 16, 16);
        assert_eq!(
            texts_of(&row[index], &space),
            [
                "reorder(1, i, j, k)",
                "reorder(1, i, k, j); vectorize(1, j)",
                "reorder(1, k, i, j)",
                "split(1, i, io, ii, 8); reorder(1, io, k, j, ii)",
                "split(1, j, jo, ji, 8); reorder(1, i, k, jo, ji)",
                "split(1, k, ko, ki, 8); reorder(1, i, ko, j, ki)",
            ]
        );
    }

    #[test]
    fn a_vectorized_loop_does_not_step_to_a_different_innermost_loop() {
        let space = matmul_space(16, 16, 16);
        let row = neighbors(&space, 16, 16, 16);
        let of = |text: &str| {
            let index = space.iter().position(|c| c.schedule == text).expect(text);
            texts_of(&row[index], &space)
        };
        let vectorized = "reorder(1, i, k, j); vectorize(1, j)";
        // Splitting `i` would make `ii` the innermost loop, so the vectorized schedule
        // does not step there. The same schedule with vectorize off does.
        let split_i = "split(1, i, io, ii, 8); reorder(1, io, k, j, ii)";
        assert!(!of(vectorized).iter().any(|text| text == split_i));
        assert!(of("reorder(1, i, k, j)").iter().any(|text| text == split_i));

        let tiled = "split(1, i, io, ii, 8); split(1, j, jo, ji, 8); \
            reorder(1, io, jo, k, ii, ji); vectorize(1, ji)";
        let swapped =
            "split(1, i, io, ii, 8); split(1, j, jo, ji, 8); reorder(1, io, jo, k, ji, ii)";
        assert!(!of(tiled).iter().any(|text| text == swapped));
        let tiled_off =
            "split(1, i, io, ii, 8); split(1, j, jo, ji, 8); reorder(1, io, jo, k, ii, ji)";
        assert!(of(tiled_off).iter().any(|text| text == swapped));
    }

    #[test]
    fn a_register_tile_steps_to_adjacent_tiles_and_to_its_twin_when_that_exists() {
        // `ri` of 2 or 4 is not a `TILE_FACTORS` entry, so those tiles have no schedule
        // without `unroll` in the space. `ri` = 8 does. At n = 16 the 8 x 16 tile needs 32
        // accumulators and is absent, so 8 x 8 steps to 4 x 8 and to its twin only.
        let twin = "split(1, i, io, ii, 8); split(1, j, jo, ji, 8); \
            reorder(1, io, jo, k, ii, ji); vectorize(1, ji)";
        let tile = format!("{twin}; unroll(1, ii); unroll(1, ji)");
        let space = matmul_space(16, 16, 16);
        let row = neighbors(&space, 16, 16, 16);
        let of = |text: &str| {
            let index = space.iter().position(|c| c.schedule == text).expect(text);
            texts_of(&row[index], &space)
        };
        assert_eq!(
            of(&tile),
            [
                "split(1, i, io, ii, 4); split(1, j, jo, ji, 8); reorder(1, io, jo, k, ii, ji); \
                    vectorize(1, ji); unroll(1, ii); unroll(1, ji)"
                    .to_string(),
                twin.to_string(),
            ]
        );
        assert!(of(twin).iter().any(|text| text == &tile));
        let four_by_sixteen = "split(1, i, io, ii, 4); split(1, j, jo, ji, 16); \
            reorder(1, io, jo, k, ii, ji); vectorize(1, ji); unroll(1, ii); unroll(1, ji)";
        assert_eq!(
            of(four_by_sixteen),
            [
                "split(1, i, io, ii, 2); split(1, j, jo, ji, 16); reorder(1, io, jo, k, ii, ji); \
                    vectorize(1, ji); unroll(1, ii); unroll(1, ji)"
                    .to_string(),
                "split(1, i, io, ii, 4); split(1, j, jo, ji, 8); reorder(1, io, jo, k, ii, ji); \
                    vectorize(1, ji); unroll(1, ii); unroll(1, ji)"
                    .to_string(),
            ]
        );
    }

    #[test]
    fn a_factor_steps_only_to_the_next_option_of_its_own_axis() {
        // m = 48, k = 40, n = 24. `i` options: none, 8, 16. `j` options: none, 8.
        // From `ii` = 8 the next `i` factor is 16, and `j` has no 16.
        let space = matmul_space(48, 40, 24);
        let row = neighbors(&space, 48, 40, 24);
        let of = |text: &str| {
            let index = space.iter().position(|c| c.schedule == text).expect(text);
            texts_of(&row[index], &space)
        };
        let ii_8 = "split(1, i, io, ii, 8); reorder(1, io, k, j, ii)";
        let ii_16 = "split(1, i, io, ii, 16); reorder(1, io, k, j, ii)";
        let ji_8 = "split(1, j, jo, ji, 8); reorder(1, i, k, jo, ji)";
        let ji_16 = "split(1, j, jo, ji, 16); reorder(1, i, k, jo, ji)";
        assert!(of(ii_8).iter().any(|text| text == ii_16));
        assert!(!space.iter().any(|c| c.schedule == ji_16));
        assert!(!of(ji_8).iter().any(|text| text.contains("ji, 16")));
    }

    #[test]
    fn every_neighbor_is_one_step_away_and_the_relation_is_symmetric() {
        for (m, k, n) in [(16, 16, 16), (64, 64, 64), (48, 40, 24)] {
            let space = matmul_space(m, k, n);
            let row = neighbors(&space, m, k, n);
            assert_eq!(row.len(), space.len(), "{m}x{k}x{n}");
            for (index, neighbors_of) in row.iter().enumerate() {
                assert!(!neighbors_of.contains(&index), "{m}x{k}x{n} {index}");
                let mut sorted = neighbors_of.clone();
                sorted.sort();
                sorted.dedup();
                assert_eq!(sorted.len(), neighbors_of.len(), "{m}x{k}x{n} {index}");
                for &other in neighbors_of {
                    assert!(
                        row[other].contains(&index),
                        "{m}x{k}x{n}: {} -> {} is one way\n  {}\n  {}",
                        index,
                        other,
                        space[index].schedule,
                        space[other].schedule
                    );
                }
            }
        }
    }
}
