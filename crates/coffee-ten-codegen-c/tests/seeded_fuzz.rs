//! N7: small generated programs through Rust Graph -> C -> interpreter bit comparison.
#[path = "support/fuzz_case.rs"]
mod fuzz_case;

use coffee_ten_codegen_c::compile;
use coffee_ten_interp::{Tensor, run_f32};
use coffee_ten_ir::Shape;
use coffee_ten_loop::lower;
use fuzz_case::{Case, corpus};
use std::{
    collections::HashSet,
    path::PathBuf,
    time::{Duration, Instant},
};

fn retain_and_minimize(case: &Case) {
    if std::env::var_os("COFFEE_TEN_FUZZ_MINIMIZING").is_some() {
        return;
    }
    let root = std::env::var_os("COFFEE_TEN_FUZZ_FAILURE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::temp_dir().join(format!("coffee-ten-fuzz-failures-{}", std::process::id()))
        });
    let directory = root.join(format!("rust-{}", case.index));
    std::fs::create_dir_all(&directory).expect("retain failure directory");
    std::fs::write(directory.join("original.tsv"), case.row() + "\n")
        .expect("retain original case");
    let script =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fuzz/ct_fuzz_minimize.py");
    let status = std::process::Command::new("python3")
        .arg(script)
        .args(["--backend", "rust", "--row"])
        .arg(case.row())
        .arg("--rust-driver")
        .arg(std::env::current_exe().unwrap())
        .arg("--output")
        .arg(&directory)
        .status();
    eprintln!(
        "N9 artifact {} (minimizer status: {status:?})",
        directory.display()
    );
}

struct Cache(PathBuf);
impl Drop for Cache {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn corpus_has_fixed_bounds_coverage_and_input_fingerprints() {
    let cases = corpus();
    assert_eq!(
        cases
            .iter()
            .map(|c| c.kind.as_str())
            .collect::<HashSet<_>>()
            .len(),
        4
    );
    assert_eq!(cases.iter().map(|c| c.pre).collect::<HashSet<_>>().len(), 4);
    assert_eq!(
        cases.iter().map(|c| c.post).collect::<HashSet<_>>().len(),
        4
    );
    for (index, c) in cases.iter().enumerate() {
        assert_eq!(c.index, index);
        assert_eq!(c.row(), Case::parse(&c.row()).unwrap().row());
        for input in c.inputs() {
            assert!(input.data.len() <= 64);
            assert!(input.data.iter().all(|x| x.is_finite()));
        }
    }
    assert_eq!(cases[0].input_hash, "09841e0b328f124f");
    assert!(Case::parse("bad").is_err());
    assert!(Case::parse(&cases[0].row().replace("2,2,4", "9,2,4")).is_err());
}

#[test]
fn seeded_generated_programs_match_reference_bits() {
    let cases = match std::env::var("COFFEE_TEN_FUZZ_CASE") {
        Ok(row) => vec![Case::parse(&row).expect("valid replay case")],
        Err(_) => corpus(),
    };
    let cache = Cache(std::env::temp_dir().join(format!("coffee-ten-n7-{}", std::process::id())));
    let start = Instant::now();
    for case in cases {
        eprintln!("N7 replay TSV: {}", case.row());
        let (graph, output, inputs) = case.program();
        assert!(graph.nodes().len() <= 24, "bounded generator grammar");
        let tensors: Vec<_> = inputs
            .iter()
            .map(|i| Tensor::new(Shape::new(i.dims.clone()), i.data.clone()).unwrap())
            .collect();
        let bindings: Vec<_> = inputs
            .iter()
            .zip(&tensors)
            .map(|(i, t)| (i.name, t))
            .collect();
        let reference = run_f32(&graph, output, &bindings).unwrap();
        let kernel = lower(&graph, output, "fuzz_kernel")
            .unwrap_or_else(|e| panic!("{}\n{graph}\n{e}", case.row()));
        let compiled =
            compile(&kernel, &cache.0).unwrap_or_else(|e| panic!("{}\n{kernel}\n{e}", case.row()));
        let mut storage: Vec<Vec<f32>> = kernel
            .buffers
            .iter()
            .map(|b| {
                inputs.iter().find(|i| i.name == b.name).map_or_else(
                    || vec![f32::NAN; b.shape.iter().product()],
                    |i| i.data.clone(),
                )
            })
            .collect();
        let mut buffers: Vec<_> = storage.iter_mut().map(Vec::as_mut_slice).collect();
        compiled.run(&mut buffers).unwrap();
        // Test-only fault used by N9's subprocess regression. Production code is untouched.
        if std::env::var_os("COFFEE_TEN_FUZZ_INJECT_FAULT").is_some()
            && case.kind == "matmul"
            && case.dims[1] > 1
        {
            storage.last_mut().unwrap()[0] = f32::MIN_POSITIVE;
        }
        for (buffer, values) in kernel.buffers.iter().zip(&storage) {
            if let Some(input) = inputs.iter().find(|i| i.name == buffer.name) {
                assert_eq!(
                    values.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    input.data.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    "input {} changed, replay TSV: {}",
                    input.name,
                    case.row()
                );
            }
        }
        let got = storage.last().unwrap();
        if got.len() != reference.data().len()
            || kernel.buffers.last().unwrap().shape != reference.shape().dims()
        {
            retain_and_minimize(&case);
        }
        assert_eq!(
            kernel.buffers.last().unwrap().shape,
            reference.shape().dims(),
            "N7 output mismatch shape: {}",
            case.row()
        );
        assert_eq!(
            got.len(),
            reference.data().len(),
            "N7 output mismatch length: {}",
            case.row()
        );
        for (element, (got, want)) in got.iter().zip(reference.data()).enumerate() {
            if got.to_bits() != want.to_bits() {
                retain_and_minimize(&case);
            }
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "N7 output mismatch element {element}, replay TSV: {}\n{kernel}",
                case.row()
            );
        }
        assert!(
            start.elapsed() < Duration::from_secs(120),
            "N7 CI time budget exceeded at {}",
            case.row()
        );
    }
}

#[test]
fn minimizer_reproduces_and_shrinks_an_injected_native_mismatch() {
    let mut case = corpus().remove(0);
    case.dims = vec![4, 4, 4];
    case.pre = 1;
    case.post = 0;
    case.input_hash = case.input_digest();
    let directory = Cache(
        std::env::temp_dir().join(format!("coffee-ten-n9-regression-{}", std::process::id())),
    );
    let result = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "seeded_generated_programs_match_reference_bits",
            "--nocapture",
        ])
        .env("COFFEE_TEN_FUZZ_CASE", case.row())
        .env("COFFEE_TEN_FUZZ_INJECT_FAULT", "1")
        .env("COFFEE_TEN_FUZZ_FAILURE_DIR", &directory.0)
        .env_remove("COFFEE_TEN_FUZZ_MINIMIZING")
        .output()
        .unwrap();
    assert!(
        !result.status.success(),
        "the original injected failure must remain failed: {}\n{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    let artifact = directory.0.join(format!("rust-{}", case.index));
    let row = std::fs::read_to_string(artifact.join("minimal.tsv")).unwrap_or_else(|e| {
        panic!(
            "{e}\n{}\n{}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        )
    });
    let minimal = Case::parse(&row).unwrap();
    assert_eq!(minimal.seed, case.seed);
    assert_eq!(minimal.dims, [1, 2, 1]);
    assert_eq!((minimal.pre, minimal.post, minimal.axis), (0, 0, 0));
    let report = std::fs::read_to_string(artifact.join("report.json")).unwrap();
    assert!(report.contains("\"verified\": true"));
    assert!(report.contains("\"status\": \"fixed_point\""));
}
