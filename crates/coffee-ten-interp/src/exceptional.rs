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

#[inline(never)]
pub(super) fn matmul<T: Scalar>(a: &Tensor<T>, b: &Tensor<T>) -> Vec<T> {
    let (m, k) = (a.shape().dims()[0], a.shape().dims()[1]);
    let n = b.shape().dims()[1];
    let mut data = Vec::with_capacity(m * n);
    for i in 0..m {
        for j in 0..n {
            let mut acc = T::zero();
            for kk in 0..k {
                let product = a.data[i * k + kk].mul(b.data[kk * n + j]);
                acc = acc.add(product);
            }
            data.push(acc);
        }
    }
    data
}
