//! Optional output ownership: workers never split an individual reduction.
use super::*;

fn fill_outputs<T: Scalar>(
    len: usize,
    threads: usize,
    f: impl Fn(usize, &mut [T]) + Sync,
) -> Vec<T> {
    let mut out = vec![T::zero(); len];
    if len == 0 {
        return out;
    }
    if threads <= 1 || len == 1 {
        f(0, &mut out);
        return out;
    }
    let chunk_len = len.div_ceil(threads.min(len));
    std::thread::scope(|scope| {
        for (chunk_index, chunk) in out.chunks_mut(chunk_len).enumerate() {
            let f = &f;
            scope.spawn(move || f(chunk_index * chunk_len, chunk));
        }
    });
    out
}

fn outputs<T: Scalar>(len: usize, threads: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    fill_outputs(len, threads, |start, chunk| {
        for (i, value) in chunk.iter_mut().enumerate() {
            *value = f(start + i);
        }
    })
}

pub(super) fn unary_threaded<T: Scalar>(op: UnaryOp, data: &[T], threads: usize) -> Vec<T> {
    if threads == 1 {
        return unary(op, data);
    }
    match op {
        UnaryOp::Neg => outputs(data.len(), threads, |i| data[i].neg()),
        UnaryOp::Relu => outputs(data.len(), threads, |i| {
            if data[i].gt(T::zero()) {
                data[i]
            } else {
                T::zero()
            }
        }),
        UnaryOp::Exp => outputs(data.len(), threads, |i| data[i].exp()),
    }
}

pub(super) fn broadcast_threaded<T: Scalar>(
    op: BinaryOp,
    a: &Tensor<T>,
    b: &Tensor<T>,
    out: &Shape,
    threads: usize,
) -> Result<Vec<T>, InterpError> {
    if threads == 1 {
        return broadcast_binary(op, a, b, out);
    }
    let len = numel(out)?;
    if len == 0 {
        return Ok(Vec::new());
    }
    let a_strides = broadcast_strides(a.shape(), out);
    let b_strides = broadcast_strides(b.shape(), out);
    Ok(outputs(len, threads, |mut linear| {
        let (mut a_at, mut b_at) = (0, 0);
        for ((&dim, &a_stride), &b_stride) in
            out.dims().iter().zip(&a_strides).zip(&b_strides).rev()
        {
            let coordinate = linear % dim;
            linear /= dim;
            a_at += coordinate * a_stride;
            b_at += coordinate * b_stride;
        }
        binary(op, a.data[a_at], b.data[b_at])
    }))
}

pub(super) fn reduce_threaded<T: Scalar>(
    op: ReduceOp,
    x: &Tensor<T>,
    axis: usize,
    threads: usize,
) -> Vec<T> {
    if threads == 1 {
        return reduce(op, x, axis);
    }
    let Some(ReductionLayout { outer, n, inner }) = ReductionLayout::new(x.shape().dims(), axis)
    else {
        return Vec::new();
    };
    // Workers own whole outputs. Within each outer group use contiguous input slices,
    // retaining the optimized serial traversal and ascending k for every accumulator.
    fill_outputs::<T>(outer * inner, threads, |start, chunk| {
        let mut offset = 0;
        while offset < chunk.len() {
            let index = start + offset;
            let i = index % inner;
            let count = (inner - i).min(chunk.len() - offset);
            let input_start = (index / inner) * n * inner + i;
            let acc = &mut chunk[offset..offset + count];
            match op {
                ReduceOp::Sum => {
                    for k in 0..n {
                        for (acc, &value) in acc
                            .iter_mut()
                            .zip(&x.data[input_start + k * inner..input_start + k * inner + count])
                        {
                            *acc = acc.add(value);
                        }
                    }
                }
                ReduceOp::Max => {
                    acc.copy_from_slice(&x.data[input_start..input_start + count]);
                    for k in 1..n {
                        for (acc, &value) in acc
                            .iter_mut()
                            .zip(&x.data[input_start + k * inner..input_start + k * inner + count])
                        {
                            *acc = if acc.gt(value) { *acc } else { value };
                        }
                    }
                }
            }
            offset += count;
        }
    })
}

pub(super) fn matmul_threaded<T: Scalar>(a: &Tensor<T>, b: &Tensor<T>, threads: usize) -> Vec<T> {
    if threads == 1 {
        return matmul(a, b);
    }
    let (m, k) = (a.shape().dims()[0], a.shape().dims()[1]);
    let n = b.shape().dims()[1];
    fill_outputs::<T>(m * n, threads, |start, chunk| {
        let mut offset = 0;
        while offset < chunk.len() {
            let index = start + offset;
            let (i, j) = (index / n, index % n);
            let count = (n - j).min(chunk.len() - offset);
            let acc = &mut chunk[offset..offset + count];
            for kk in 0..k {
                let av = a.data[i * k + kk];
                for (acc, &bv) in acc.iter_mut().zip(&b.data[kk * n + j..kk * n + j + count]) {
                    let product = av.mul(bv);
                    *acc = acc.add(product);
                }
            }
            offset += count;
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[test]
    fn output_chunks_have_one_owner_and_default_uses_the_caller() {
        let caller = std::thread::current().id();
        let out: Vec<f32> = outputs(37, 1, |i| {
            assert_eq!(std::thread::current().id(), caller);
            i as f32
        });
        let ids = Mutex::new(std::collections::HashSet::new());
        let threaded: Vec<f32> = outputs(37, 4, |i| {
            ids.lock().unwrap().insert(std::thread::current().id());
            i as f32
        });
        assert_eq!(out, threaded);
        assert_eq!(ids.into_inner().unwrap().len(), 4);
        assert!(outputs::<f32>(0, 4, |_| panic!("no outputs")).is_empty());
    }
}
