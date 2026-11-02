//! Original scalar kernels for exceptional inputs; isolate their arithmetic from SIMD lowering.
use super::*;

#[inline(never)]
pub(super) fn reduce<T: Scalar>(op: ReduceOp, x: &Tensor<T>, axis: usize) -> Vec<T> {
    let dims = x.shape().dims();
    let n = dims[axis];
    assert!(n > 0, "invariant: a reduced axis is non-empty");
    let inner: usize = dims[axis + 1..].iter().product();
    let outer: usize = dims[..axis].iter().product();
    let mut out = Vec::with_capacity(outer * inner);
    for o in 0..outer {
        for i in 0..inner {
            let at = |k: usize| x.data[o * n * inner + k * inner + i];
            out.push(match op {
                ReduceOp::Sum => {
                    let mut acc = T::zero();
                    for k in 0..n {
                        acc = acc.add(at(k));
                    }
                    acc
                }
                ReduceOp::Max => {
                    let mut acc = at(0);
                    for k in 1..n {
                        acc = binary(BinaryOp::Max, acc, at(k));
                    }
                    acc
                }
            });
        }
    }
    out
}
