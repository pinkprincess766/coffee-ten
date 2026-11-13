//! Golden C for a small matmul: the naive schedule and one tiling. A change to the emitted C
//! shows up as a diff of `tests/golden/*.c`; regenerate with
//! `COFFEE_TEN_UPDATE_GOLDEN=1 cargo test -p coffee-ten-codegen-c --test golden_c` and review
//! the diff. Next to each snapshot, the compiled kernel must match the reference interpreter
//! bit for bit.

use std::path::{Path, PathBuf};

use coffee_ten_codegen_c::{compile, emit_c};
use coffee_ten_interp::{Tensor, run_f32};
use coffee_ten_ir::{Graph, NodeId, Shape};
use coffee_ten_loop::{Kernel, OrderClass, Schedule, apply_classified, lower};

const N: usize = 4;
/// The naive schedule (no steps) and one tiling of `j` with `k` between the tile loops.
const CASES: [(&str, &str); 2] = [
    ("matmul-4-naive", ""),
    (
        "matmul-4-tiled",
        "split(1, j, jo, ji, 2); reorder(1, i, jo, k, ji)",
    ),
];

/// A fresh directory, removed when dropped (also if the test panics).
struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> Self {
        let dir =
            std::env::temp_dir().join(format!("coffee-ten-golden-c-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create dir");
        Self(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The `N x N x N` matmul graph, its output, and the kernel with `schedule` applied.
fn case(schedule: &str) -> (Graph, NodeId, Kernel) {
    let mut g = Graph::new();
    let a = g.input("A", Shape::new(vec![N, N])).unwrap();
    let b = g.input("B", Shape::new(vec![N, N])).unwrap();
    let prod = g.matmul(a, b).unwrap();
    let kernel = lower(&g, prod, "golden_matmul").expect("graph lowers");
    let schedule = Schedule::parse(schedule).expect("schedule parses");
    let (kernel, class) = apply_classified(&kernel, &schedule).expect("schedule applies");
    assert_eq!(class, OrderClass::Preserving, "{schedule:?}");
    (g, prod, kernel)
}

fn golden_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden")
        .join(format!("{name}.c"))
}

#[test]
fn emitted_c_matches_the_golden_files() {
    let update = std::env::var_os("COFFEE_TEN_UPDATE_GOLDEN").is_some();
    for (name, schedule) in CASES {
        let (_, _, kernel) = case(schedule);
        let source = emit_c(&kernel).expect("C emits");
        let path = golden_path(name);
        if update {
            std::fs::write(&path, &source).expect("write golden");
            continue;
        }
        let golden = std::fs::read_to_string(&path).expect("golden file exists");
        assert!(
            source == golden,
            "C for {name} differs from {}; if intended, regenerate with \
             COFFEE_TEN_UPDATE_GOLDEN=1 and review the diff.\n--- generated ---\n{source}",
            path.display()
        );
    }
}

#[test]
fn golden_kernels_match_the_interpreter_bit_for_bit() {
    let dir = TempDir::new("run");
    // Values that do not add exactly in f32, a negative zero, and a large magnitude.
    let a_data: Vec<f32> = (0..N * N)
        .map(|n| match n {
            5 => -0.0,
            11 => 1e20,
            _ => (n as f32 - 7.0) * 0.1 + 1.0 / 3.0,
        })
        .collect();
    let b_data: Vec<f32> = (0..N * N).map(|n| 1.0 / (n as f32 + 1.5) - 0.25).collect();
    let shape = Shape::new(vec![N, N]);
    let a = Tensor::new(shape.clone(), a_data.clone()).unwrap();
    let b = Tensor::new(shape, b_data.clone()).unwrap();
    for (name, schedule) in CASES {
        let (graph, prod, kernel) = case(schedule);
        let reference = run_f32(&graph, prod, &[("A", &a), ("B", &b)]).unwrap();
        let names: Vec<&str> = kernel.buffers.iter().map(|b| b.name.as_str()).collect();
        assert_eq!(names.len(), 3, "{name}: buffers {names:?}");
        let mut buffers: Vec<Vec<f32>> = names
            .iter()
            .map(|&buffer| match buffer {
                "A" => a_data.clone(),
                "B" => b_data.clone(),
                _ => vec![f32::from_bits(0x7fc0_dead); N * N],
            })
            .collect();
        let compiled = compile(&kernel, &dir.0).expect("kernel compiles");
        let mut slices: Vec<&mut [f32]> = buffers.iter_mut().map(|b| b.as_mut_slice()).collect();
        compiled.run(&mut slices).unwrap();
        let out = names.iter().position(|&n| n != "A" && n != "B").unwrap();
        let got: Vec<u32> = buffers[out].iter().map(|x| x.to_bits()).collect();
        let want: Vec<u32> = reference.data().iter().map(|x| x.to_bits()).collect();
        assert_eq!(got, want, "{name}");
    }
}
