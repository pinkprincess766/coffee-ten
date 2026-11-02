//! Reproducible differential checks against the frozen pre-optimization interpreter.

use super::*;

trait Bitwise: Scalar {
    type Bits: Eq + std::fmt::Debug;
    fn bits(self) -> Self::Bits;
}

impl Bitwise for f32 {
    type Bits = u32;
    fn bits(self) -> u32 {
        self.to_bits()
    }
}

impl Bitwise for f64 {
    type Bits = u64;
    fn bits(self) -> u64 {
        self.to_bits()
    }
}

fn assert_bits<T: Bitwise>(actual: Vec<T>, expected: Vec<T>, case: &str) {
    assert_eq!(actual.len(), expected.len(), "{case}: length");
    for (i, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
        assert_eq!(actual.bits(), expected.bits(), "{case}: element {i}");
    }
}

// Test-local xorshift64: a fixed nonzero seed makes every failing case reproducible.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 32) as usize
    }

    fn dims(&mut self, rank: usize) -> Vec<usize> {
        (0..rank).map(|_| self.next() % 8).collect()
    }

    fn tensor<T: Scalar>(&mut self, dims: &[usize], special: bool) -> Tensor<T> {
        let data = (0..dims.iter().product())
            .map(|_| {
                let bits = self.next() as u32;
                let finite = f32::from_bits((bits & 0x807fffff) | (((bits >> 24) % 250 + 1) << 23));
                let value = if special {
                    match bits % 11 {
                        0 => 0.0,
                        1 => -0.0,
                        2 => f32::NAN,
                        3 => f32::INFINITY,
                        4 => f32::NEG_INFINITY,
                        5 => f32::from_bits(1),
                        6 => -f32::from_bits(1),
                        _ => finite,
                    }
                } else {
                    finite
                };
                T::from_f32(value)
            })
            .collect();
        Tensor::new(shape(dims), data).unwrap()
    }
}

fn check_reduce<T: Bitwise>() {
    let mut rng = Rng(0x9e3779b97f4a7c15);
    // Explicit boundary shapes supplement the random sample. Every nonempty axis is tested.
    let mut shapes = vec![
        vec![1],
        vec![7],
        vec![1000],
        vec![2, 3],
        vec![3, 17],
        vec![17, 3],
        vec![2, 3, 5],
        vec![3, 1, 7],
        vec![2, 3, 5, 7],
        vec![0, 3],
        vec![3, 0],
        vec![2, 0, 3],
        vec![0, 3, 2],
        vec![2, 3, 0],
    ];
    for _ in 0..128 {
        let rank = 1 + rng.next() % 5;
        shapes.push(rng.dims(rank));
    }
    for dims in shapes {
        for special in [false, true] {
            let x = rng.tensor::<T>(&dims, special);
            for (axis, &extent) in dims.iter().enumerate() {
                if extent == 0 {
                    continue;
                }
                for op in [ReduceOp::Sum, ReduceOp::Max] {
                    let case = format!("reduce {op:?}, {dims:?}, axis {axis}, special={special}");
                    assert_bits(reduce(op, &x, axis), reference::reduce(op, &x, axis), &case);
                }
            }
        }
    }
}

fn check_broadcast<T: Bitwise>() {
    let mut rng = Rng(0x243f6a8885a308d3);
    let mut pairs = vec![
        (vec![], vec![]),
        (vec![], vec![2, 3]),
        (vec![2, 3], vec![]),
        (vec![2, 3], vec![2, 3]),
        (vec![2, 1], vec![3]),
        (vec![0, 3], vec![1, 3]),
        (vec![1], vec![3, 0]),
    ];
    for _ in 0..128 {
        let rank = rng.next() % 6;
        let dims = rng.dims(rank);
        let mut operand_shape = || {
            let skip = rng.next() % (rank + 1);
            dims[skip..]
                .iter()
                .map(|&d| if rng.next().is_multiple_of(2) { 1 } else { d })
                .collect()
        };
        pairs.push((operand_shape(), operand_shape()));
    }
    for (a_dims, b_dims) in pairs {
        let out = shape(&a_dims).broadcast(&shape(&b_dims)).unwrap();
        for special in [false, true] {
            let a = rng.tensor::<T>(&a_dims, special);
            let b = rng.tensor::<T>(&b_dims, special);
            for op in [
                BinaryOp::Add,
                BinaryOp::Sub,
                BinaryOp::Mul,
                BinaryOp::Div,
                BinaryOp::Max,
            ] {
                let case = format!("broadcast {op:?}, {a_dims:?}, {b_dims:?}, special={special}");
                assert_bits(
                    broadcast_binary(op, &a, &b, &out).unwrap(),
                    reference::broadcast_binary(op, &a, &b, &out).unwrap(),
                    &case,
                );
            }
        }
    }
}

fn check_matmul<T: Bitwise>() {
    let mut rng = Rng(0xb7e151628aed2a6b);
    let mut sizes = vec![
        (33, 65, 67),
        (2, 97, 129),
        (1, 1, 65),
        (3, 65, 1),
        (0, 3, 2),
        (2, 0, 3),
        (2, 3, 0),
        (1, 1, 1),
        (3, 17, 5),
    ];
    for _ in 0..128 {
        sizes.push((rng.next() % 18, rng.next() % 18, rng.next() % 18));
    }
    for (m, k, n) in sizes {
        for special in [false, true] {
            let a = rng.tensor::<T>(&[m, k], special);
            let b = rng.tensor::<T>(&[k, n], special);
            let case = format!("matmul ({m}, {k}) @ ({k}, {n}), special={special}");
            assert_bits(matmul(&a, &b), reference::matmul(&a, &b), &case);
        }
    }
}

#[test]
fn reduce_matches_frozen_reference_f32() {
    check_reduce::<f32>();
}

#[test]
fn reduce_matches_frozen_reference_f64() {
    check_reduce::<f64>();
}

#[test]
fn broadcast_matches_frozen_reference_f32() {
    check_broadcast::<f32>();
}

#[test]
fn broadcast_matches_frozen_reference_f64() {
    check_broadcast::<f64>();
}

#[test]
fn matmul_matches_frozen_reference_f32() {
    check_matmul::<f32>();
}

#[test]
fn matmul_matches_frozen_reference_f64() {
    check_matmul::<f64>();
}

fn check_unary<T: Bitwise>() {
    let mut rng = Rng(0x13198a2e03707344);
    for special in [false, true] {
        let x = rng.tensor::<T>(&[1024], special);
        for op in [UnaryOp::Neg, UnaryOp::Relu, UnaryOp::Exp] {
            assert_bits(
                unary(op, &x.data),
                x.data.iter().map(|&x| reference::unary(op, x)).collect(),
                &format!("unary {op:?} special={special}"),
            );
        }
    }
    assert!(unary::<T>(UnaryOp::Neg, &[]).is_empty());
}
#[test]
fn unary_matches_frozen_reference() {
    check_unary::<f32>();
    check_unary::<f64>();
}

fn check_graph<T: Bitwise>() {
    let mut g = Graph::new();
    let x = g.input("x", shape(&[2, 3])).unwrap();
    let twice = g.binary(BinaryOp::Add, x, x).unwrap();
    let m = g.reduce_axis(ReduceOp::Max, twice, 0).unwrap();
    let mut y = g.binary(BinaryOp::Sub, twice, m).unwrap();
    for _ in 0..32 {
        y = g.unary(UnaryOp::Neg, y).unwrap();
    }
    g.binary(BinaryOp::Mul, x, x).unwrap();
    let mut rng = Rng(0xa4093822299f31d0);
    for special in [false, true] {
        let input = rng.tensor::<f32>(&[2, 3], special);
        let inputs = [("x", &input)];
        for out in [x, twice, m, y, g.constant(-0.)] {
            let actual = evaluate::<T>(&g, out, &inputs).unwrap();
            let expected = reference::evaluate::<T>(&g, out, &inputs).unwrap();
            assert_eq!(actual.shape(), expected.shape());
            assert_bits(
                actual.data,
                expected.data,
                "shared/dead DAG and identity outputs",
            );
        }
    }
}
#[test]
fn live_graph_matches_frozen_evaluation() {
    check_graph::<f32>();
    check_graph::<f64>();
}
