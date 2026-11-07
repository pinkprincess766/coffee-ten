//! N16 freezes the reference independently of any generated/native kernel.
#[path = "../../coffee-ten-codegen-c/tests/support/fuzz_case.rs"]
mod fuzz_case;

use coffee_ten_interp::{Tensor, run_f32, run_f64};
use coffee_ten_ir::{Graph, ReduceOp, Shape};

fn fingerprint<T>(tensor: &Tensor<T>, dtype: &[u8], bits: impl Fn(&T) -> Vec<u8>) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    let mut feed = |bytes: &[u8]| {
        for &byte in bytes {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
        }
    };
    feed(b"coffee-ten-reference-v1\0");
    feed(dtype);
    feed(&(tensor.shape().rank() as u64).to_le_bytes());
    for &dim in tensor.shape().dims() {
        feed(&(dim as u64).to_le_bytes());
    }
    feed(&(tensor.data().len() as u64).to_le_bytes());
    for value in tensor.data() {
        feed(&bits(value));
    }
    format!("{hash:016x}")
}

#[test]
fn frozen_corpus_output_bits_in_f32_and_f64() {
    let expected: Vec<_> = include_str!("fixtures/reference-v1.tsv")
        .lines()
        .filter(|line| !line.starts_with('#') && !line.is_empty())
        .map(|line| line.split('\t').collect::<Vec<_>>())
        .collect();
    let cases = fuzz_case::corpus();
    assert_eq!(expected.len(), cases.len());
    for (case, pinned) in cases.into_iter().zip(expected) {
        assert_eq!(pinned.len(), 3);
        assert_eq!(pinned[0].parse::<usize>().unwrap(), case.index);
        let (graph, output, inputs) = case.program();
        let tensors: Vec<_> = inputs
            .iter()
            .map(|i| Tensor::new(Shape::new(i.dims.clone()), i.data.clone()).unwrap())
            .collect();
        let bindings: Vec<_> = inputs
            .iter()
            .zip(&tensors)
            .map(|(i, t)| (i.name, t))
            .collect();
        let f32_result = run_f32(&graph, output, &bindings).unwrap();
        let f64_result = run_f64(&graph, output, &bindings).unwrap();
        assert_eq!(
            fingerprint(&f32_result, b"f32\0", |v| v
                .to_bits()
                .to_le_bytes()
                .to_vec()),
            pinned[1],
            "f32 case {}",
            case.row()
        );
        assert_eq!(
            fingerprint(&f64_result, b"f64\0", |v| v
                .to_bits()
                .to_le_bytes()
                .to_vec()),
            pinned[2],
            "f64 case {}",
            case.row()
        );
    }
}

#[test]
fn reduction_order_canaries_reject_pairwise_reassociation() {
    // Sequential ((((0 + huge) + 1) - huge) + 3) = 3. Pairwise grouping gives 0.
    for matmul in [false, true] {
        let mut graph = Graph::new();
        let shape = if matmul { vec![1, 4] } else { vec![4] };
        let x = graph.input("x", Shape::new(shape.clone())).unwrap();
        let tensor = Tensor::new(Shape::new(shape), vec![1e20, 1.0, -1e20, 3.0]).unwrap();
        let ones = Tensor::new(Shape::new(vec![4, 1]), vec![1.0; 4]).unwrap();
        let (output, bindings) = if matmul {
            let y = graph.input("y", Shape::new(vec![4, 1])).unwrap();
            (
                graph.matmul(x, y).unwrap(),
                vec![("x", &tensor), ("y", &ones)],
            )
        } else {
            (
                graph.reduce_axis(ReduceOp::Sum, x, 0).unwrap(),
                vec![("x", &tensor)],
            )
        };
        assert_eq!(
            run_f32(&graph, output, &bindings).unwrap().data()[0].to_bits(),
            3.0f32.to_bits()
        );
        assert_eq!(
            run_f64(&graph, output, &bindings).unwrap().data()[0].to_bits(),
            3.0f64.to_bits()
        );
    }
    let pairwise = (1e20f32 + 1.0) + (-1e20f32 + 3.0);
    assert_ne!(pairwise.to_bits(), 3.0f32.to_bits());
}

#[test]
fn fingerprint_keeps_signed_zero_dtype_and_shape_distinct() {
    let hash = |dims, values| {
        fingerprint(
            &Tensor::new(Shape::new(dims), values).unwrap(),
            b"f32\0",
            |v: &f32| v.to_bits().to_le_bytes().to_vec(),
        )
    };
    assert_ne!(hash(vec![1], vec![0.0]), hash(vec![1], vec![-0.0]));
    assert_ne!(hash(vec![1], vec![0.0]), hash(vec![1, 1], vec![0.0]));
    let tensor = Tensor::new(Shape::new(vec![1]), vec![0.0f32]).unwrap();
    assert_ne!(
        hash(vec![1], vec![0.0]),
        fingerprint(&tensor, b"other\0", |v| v.to_bits().to_le_bytes().to_vec())
    );
}
