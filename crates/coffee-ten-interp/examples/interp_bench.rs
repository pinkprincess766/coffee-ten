//! End-to-end interpreter timing; no generated C or compiler benchmark is involved.
use coffee_ten_interp::{MAX_THREADS, Tensor, run_f32_with_threads, run_f64_with_threads};
use std::sync::OnceLock;
static THREADS: OnceLock<usize> = OnceLock::new();
use coffee_ten_ir::{BinaryOp, Graph, NodeId, ReduceOp, Shape, UnaryOp};
use std::{
    hint::black_box,
    time::{Duration, Instant},
};

fn shape(d: &[usize]) -> Shape {
    Shape::new(d.to_vec())
}
fn data(d: &[usize]) -> Tensor<f32> {
    Tensor::new(
        shape(d),
        (0..d.iter().product())
            .map(|i| ((i % 251) as f32 - 125.0) * 0.01)
            .collect(),
    )
    .unwrap()
}
fn measure(mut f: impl FnMut()) -> (f64, f64, f64) {
    f();
    let mut times = Vec::new();
    for _ in 0..9 {
        let start = Instant::now();
        let mut count = 0;
        loop {
            f();
            count += 1;
            if start.elapsed() >= Duration::from_millis(10) {
                break;
            }
        }
        times.push(start.elapsed().as_secs_f64() * 1e6 / count as f64);
    }
    times.sort_by(f64::total_cmp);
    (times[2], times[4], times[6])
}
fn report(name: &str, g: &Graph, y: NodeId, inputs: &[(&str, &Tensor<f32>)]) {
    let name = name.replace(',', "x");
    let (q1, med, q3) = measure(|| {
        black_box(
            run_f32_with_threads(black_box(g), y, black_box(inputs), *THREADS.get().unwrap())
                .unwrap(),
        );
    });
    println!("{name},f32,{q1:.3},{med:.3},{q3:.3}");
    let (q1, med, q3) = measure(|| {
        black_box(
            run_f64_with_threads(black_box(g), y, black_box(inputs), *THREADS.get().unwrap())
                .unwrap(),
        );
    });
    println!("{name},f64,{q1:.3},{med:.3},{q3:.3}");
}
fn parse_threads(args: &[String]) -> Result<usize, &'static str> {
    match args {
        [] => Ok(1),
        [flag, n] if flag == "--threads" => n
            .parse::<usize>()
            .ok()
            .filter(|n| (1..=MAX_THREADS).contains(n))
            .ok_or("--threads must be in 1..=256"),
        _ => Err("usage: interp_bench [--threads N]"),
    }
}
fn main() -> std::process::ExitCode {
    let threads = match parse_threads(&std::env::args().skip(1).collect::<Vec<_>>()) {
        Ok(threads) => threads,
        Err(message) => {
            eprintln!("{message}");
            return std::process::ExitCode::from(2);
        }
    };
    THREADS.set(threads).unwrap();
    println!("# threads={threads}; 1 disables worker creation");
    println!(
        "# OS={} ARCH={}; 9 samples >=10ms each; one untimed call; includes allocation and graph evaluation",
        std::env::consts::OS,
        std::env::consts::ARCH
    );
    println!("case,dtype,p25_us,median_us,p75_us");
    for (dims, axis) in [
        (&[512, 512][..], 0),
        (&[512, 512][..], 1),
        (&[8, 64, 128][..], 1),
    ] {
        for op in [ReduceOp::Sum, ReduceOp::Max] {
            let mut g = Graph::new();
            let x = g.input("x", shape(dims)).unwrap();
            let y = g.reduce_axis(op, x, axis).unwrap();
            report(
                &format!("reduce_{op:?}_{dims:?}_axis{axis}"),
                &g,
                y,
                &[("x", &data(dims))],
            );
        }
    }
    for (name, ad, bd) in [
        ("equal", &[256, 256][..], &[256, 256][..]),
        ("scalar", &[256, 256][..], &[][..]),
        ("broadcast", &[8, 32, 1, 64][..], &[1, 32, 16, 1][..]),
    ] {
        let mut g = Graph::new();
        let a = g.input("a", shape(ad)).unwrap();
        let b = g.input("b", shape(bd)).unwrap();
        let y = g.binary(BinaryOp::Add, a, b).unwrap();
        report(name, &g, y, &[("a", &data(ad)), ("b", &data(bd))]);
    }
    for op in [UnaryOp::Neg, UnaryOp::Relu, UnaryOp::Exp] {
        let mut g = Graph::new();
        let x = g.input("x", shape(&[65536])).unwrap();
        let y = g.unary(op, x).unwrap();
        report(&format!("unary_{op:?}"), &g, y, &[("x", &data(&[65536]))]);
    }
    for n in [64, 128] {
        let mut g = Graph::new();
        let a = g.input("a", shape(&[n, n])).unwrap();
        let b = g.input("b", shape(&[n, n])).unwrap();
        let y = g.matmul(a, b).unwrap();
        report(
            &format!("matmul_{n}"),
            &g,
            y,
            &[("a", &data(&[n, n])), ("b", &data(&[n, n]))],
        );
    }
    let mut g = Graph::new();
    let x = g.input("x", shape(&[65536])).unwrap();
    let mut y = x;
    for _ in 0..32 {
        y = g.unary(UnaryOp::Neg, y).unwrap();
    }
    report("chain_32", &g, y, &[("x", &data(&[65536]))]);
    std::process::ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn thread_option_is_explicit_and_validated() {
        assert_eq!(parse_threads(&[]), Ok(1));
        for value in ["2", "4", "256"] {
            assert!(parse_threads(&["--threads".into(), value.into()]).is_ok());
        }
        for value in ["0", "257", "-1", "bad"] {
            assert!(parse_threads(&["--threads".into(), value.into()]).is_err());
        }
        assert!(parse_threads(&["--threads".into()]).is_err());
        assert!(parse_threads(&["--bad".into(), "2".into()]).is_err());
    }
}
