use coffee_ten_interp::{InterpError, Tensor, run_f32_with_threads, run_f64_with_threads};
use coffee_ten_ir::{Graph, Shape};

#[test]
fn zero_inner_dimension_still_checks_result_element_count() {
    let extent = 1usize << (usize::BITS / 2);
    let mut graph = Graph::new();
    let a = graph.input("a", Shape::new(vec![extent, 0])).unwrap();
    let b = graph.input("b", Shape::new(vec![0, extent])).unwrap();
    let output = graph.matmul(a, b).unwrap();
    let a = Tensor::new(Shape::new(vec![extent, 0]), Vec::<f32>::new()).unwrap();
    let b = Tensor::new(Shape::new(vec![0, extent]), Vec::<f32>::new()).unwrap();
    let expected = InterpError::TooLarge(Shape::new(vec![extent, extent]));
    for threads in [1, 2] {
        assert_eq!(
            run_f32_with_threads(&graph, output, &[("a", &a), ("b", &b)], threads),
            Err(expected.clone())
        );
        assert_eq!(
            run_f64_with_threads(&graph, output, &[("a", &a), ("b", &b)], threads),
            Err(expected.clone())
        );
    }
}

#[test]
fn zero_inner_dimension_checks_result_byte_capacity() {
    // Element count fits usize, but a Vec<f64> cannot represent this many bytes.
    let rows = (isize::MAX as usize / std::mem::size_of::<f64>()) + 1;
    let mut graph = Graph::new();
    let a = graph.input("a", Shape::new(vec![rows, 0])).unwrap();
    let b = graph.input("b", Shape::new(vec![0, 1])).unwrap();
    let output = graph.matmul(a, b).unwrap();
    let a = Tensor::new(Shape::new(vec![rows, 0]), Vec::<f32>::new()).unwrap();
    let b = Tensor::new(Shape::new(vec![0, 1]), Vec::<f32>::new()).unwrap();
    for threads in [1, 2] {
        assert_eq!(
            run_f64_with_threads(&graph, output, &[("a", &a), ("b", &b)], threads),
            Err(InterpError::TooLarge(Shape::new(vec![rows, 1])))
        );
    }
}

#[test]
fn empty_result_skips_even_an_enormous_row_count() {
    let mut graph = Graph::new();
    let a = graph.input("a", Shape::new(vec![usize::MAX, 0])).unwrap();
    let b = graph.input("b", Shape::new(vec![0, 0])).unwrap();
    let output = graph.matmul(a, b).unwrap();
    let a = Tensor::new(Shape::new(vec![usize::MAX, 0]), Vec::<f32>::new()).unwrap();
    let b = Tensor::new(Shape::new(vec![0, 0]), Vec::<f32>::new()).unwrap();
    for threads in [1, 2] {
        let out = run_f32_with_threads(&graph, output, &[("a", &a), ("b", &b)], threads).unwrap();
        assert_eq!(out.shape().dims(), &[usize::MAX, 0]);
        assert!(out.data().is_empty());
        let out = run_f64_with_threads(&graph, output, &[("a", &a), ("b", &b)], threads).unwrap();
        assert_eq!(out.shape().dims(), &[usize::MAX, 0]);
        assert!(out.data().is_empty());
    }
}

#[test]
fn zero_k_nonempty_result_retains_positive_zero() {
    let mut graph = Graph::new();
    let a = graph.input("a", Shape::new(vec![2, 0])).unwrap();
    let b = graph.input("b", Shape::new(vec![0, 3])).unwrap();
    let output = graph.matmul(a, b).unwrap();
    let a = Tensor::new(Shape::new(vec![2, 0]), Vec::<f32>::new()).unwrap();
    let b = Tensor::new(Shape::new(vec![0, 3]), Vec::<f32>::new()).unwrap();
    for threads in [1, 2] {
        let out = run_f32_with_threads(&graph, output, &[("a", &a), ("b", &b)], threads).unwrap();
        assert_eq!(
            out.data().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            vec![0u32; 6]
        );
        let out = run_f64_with_threads(&graph, output, &[("a", &a), ("b", &b)], threads).unwrap();
        assert_eq!(
            out.data().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            vec![0u64; 6]
        );
    }
}
