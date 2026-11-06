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
        assert_eq!(got.len(), reference.data().len(), "{}", case.row());
        for (element, (got, want)) in got.iter().zip(reference.data()).enumerate() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "element {element}, replay TSV: {}\n{kernel}",
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
