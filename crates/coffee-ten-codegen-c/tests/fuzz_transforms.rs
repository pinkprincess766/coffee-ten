//! N8: metamorphic compiler checks with no change to per-output reduction order.
#[path = "support/fuzz_case.rs"]
mod fuzz_case;

use coffee_ten_codegen_c::{compile, emit_c};
use coffee_ten_interp::{Tensor, run_f32};
use coffee_ten_ir::{Shape, UnaryOp};
use coffee_ten_loop::{
    Access, AffineExpr, Expr, Kernel, OrderClass, Schedule, Step, apply_classified, lower,
};
use fuzz_case::{Case, Input, corpus};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

struct Cache(PathBuf);
impl Drop for Cache {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn tiled(kernel: &Kernel) -> Option<Kernel> {
    let steps: Vec<_> = kernel
        .nests
        .iter()
        .enumerate()
        .filter_map(|(nest, n)| {
            n.loops.iter().find(|l| l.extent > 1).map(|l| Step::Split {
                nest,
                loop_name: l.name.clone(),
                outer: format!("{}_n8o", l.name),
                inner: format!("{}_n8i", l.name),
                factor: 2,
            })
        })
        .collect();
    if steps.is_empty() {
        return None;
    }
    let (out, class) = apply_classified(kernel, &Schedule { steps }).unwrap();
    assert_eq!(class, OrderClass::Preserving);
    Some(out)
}

fn independent_permutation(kernel: &Kernel) -> Option<Kernel> {
    let mut steps = Vec::new();
    for (nest, n) in kernel.nests.iter().enumerate() {
        // R-free nests only: every loop selects an independent output element.
        let all_independent = n.loops.iter().all(|l| {
            n.body
                .target
                .index
                .iter()
                .any(|e| e.terms.iter().any(|(name, c)| name == &l.name && *c != 0))
        });
        if n.loops.len() >= 2 && all_independent {
            let order = n.loops.iter().rev().map(|l| l.name.clone()).collect();
            steps.push(Step::Reorder { nest, order });
        }
    }
    if steps.is_empty() {
        return None;
    }
    let (out, class) = apply_classified(kernel, &Schedule { steps }).unwrap();
    assert_eq!(class, OrderClass::Preserving);
    Some(out)
}

fn rename_expr(e: &mut AffineExpr, names: &BTreeMap<String, String>) {
    for (name, _) in &mut e.terms {
        *name = names[name].clone();
    }
}
fn rename_access(a: &mut Access, names: &BTreeMap<String, String>) {
    for e in &mut a.index {
        rename_expr(e, names);
    }
}
fn rename_value(e: &mut Expr, names: &BTreeMap<String, String>) {
    match e {
        Expr::Load(a) => rename_access(a, names),
        Expr::Const(_) => {}
        Expr::Unary(_, x) => rename_value(x, names),
        Expr::Binary(_, a, b) => {
            rename_value(a, names);
            rename_value(b, names);
        }
    }
}
fn alpha_renamed(kernel: &Kernel) -> Kernel {
    let mut out = kernel.clone();
    for (nest, n) in out.nests.iter_mut().enumerate() {
        let names: BTreeMap<_, _> = n
            .loops
            .iter()
            .enumerate()
            .map(|(i, l)| (l.name.clone(), format!("n8_{nest}_{i}")))
            .collect();
        for l in &mut n.loops {
            l.name = names[&l.name].clone();
        }
        for g in &mut n.guards {
            rename_expr(&mut g.expr, &names);
        }
        rename_access(&mut n.body.target, &names);
        rename_value(&mut n.body.value, &names);
    }
    out.validate().unwrap();
    out
}

fn check(
    case: &Case,
    kernel: &Kernel,
    inputs: &[Input],
    reference: &Tensor<f32>,
    cache: &Path,
    tag: &str,
) {
    let compiled =
        compile(kernel, cache).unwrap_or_else(|e| panic!("{tag}: {}\n{kernel}\n{e}", case.row()));
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
    let actual = storage.last().unwrap();
    assert_eq!(actual.len(), reference.data().len());
    for (index, (a, b)) in actual.iter().zip(reference.data()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{tag}, element {index}, replay TSV: {}\n{kernel}",
            case.row()
        );
    }
}

#[test]
fn preserving_transforms_match_the_same_interpreter_bits() {
    let cache = Cache(std::env::temp_dir().join(format!("coffee-ten-n8-{}", std::process::id())));
    let cases = match std::env::var("COFFEE_TEN_FUZZ_CASE") {
        Ok(row) => vec![Case::parse(&row).unwrap()],
        Err(_) => corpus(),
    };
    let replay = std::env::var_os("COFFEE_TEN_FUZZ_CASE").is_some();
    let mut applied = [0usize; 4];
    let start = Instant::now();
    for case in cases {
        eprintln!("N8 replay TSV: {}", case.row());
        let (mut graph, output, inputs) = case.program();
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
        let kernel = lower(&graph, output, "fuzz_kernel").unwrap();
        let tile = tiled(&kernel);
        if let Some(out) = &tile {
            check(&case, out, &inputs, &reference, &cache.0, "tile");
            applied[0] += 1;
        }
        if let Some(out) = independent_permutation(&kernel) {
            check(
                &case,
                &out,
                &inputs,
                &reference,
                &cache.0,
                "independent permutation",
            );
            applied[1] += 1;
        }
        check(
            &case,
            &alpha_renamed(tile.as_ref().unwrap_or(&kernel)),
            &inputs,
            &reference,
            &cache.0,
            "alpha rename",
        );
        applied[2] += 1;
        // Append an unreachable expression after the selected output. It must not be lowered
        // or evaluated, and it cannot renumber any existing reachable graph node.
        let unused = graph.constant(-123.0);
        graph.unary(UnaryOp::Relu, unused).unwrap();
        let dead = lower(&graph, output, "fuzz_kernel").unwrap();
        assert_eq!(kernel, dead);
        assert_eq!(emit_c(&kernel).unwrap(), emit_c(&dead).unwrap());
        let after = run_f32(&graph, output, &bindings).unwrap();
        assert_eq!(
            after.data().iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            reference
                .data()
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        );
        check(
            &case,
            &dead,
            &inputs,
            &reference,
            &cache.0,
            "unreachable expression",
        );
        applied[3] += 1;
        assert!(
            start.elapsed() < Duration::from_secs(120),
            "N8 time budget exceeded at {}",
            case.row()
        );
    }
    if !replay {
        assert!(
            applied.iter().all(|&n| n > 0),
            "non-vacuous transformation coverage: {applied:?}"
        );
    }
}
