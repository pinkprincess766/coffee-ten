//! Raw C affine intermediates must be safe even when the final result is guarded.

use coffee_ten_codegen_c::{CodegenError, emit_c};
use coffee_ten_loop::{
    Access, AffineExpr, Buffer, Expr, Guard, Kernel, Loop, LoopNest, Schedule, Store, apply,
};
use std::process::Command;

fn affine(constant: i64, terms: &[(&str, i64)]) -> AffineExpr {
    AffineExpr {
        constant,
        terms: terms.iter().map(|(n, c)| (n.to_string(), *c)).collect(),
    }
}

fn kernel(extent: usize, guards: Vec<Guard>, index: AffineExpr) -> Kernel {
    Kernel {
        name: "affine_check".to_string(),
        buffers: vec![Buffer {
            name: "Y".to_string(),
            shape: vec![1],
        }],
        nests: vec![LoopNest {
            loops: vec![Loop {
                name: "i".to_string(),
                extent,
                vectorize: false,
                unroll: false,
            }],
            guards,
            body: Store {
                target: Access {
                    buffer: 0,
                    index: vec![index],
                },
                value: Expr::Const(1.0),
            },
        }],
    }
}

fn rejects(kernel: &Kernel) {
    kernel
        .validate()
        .expect("the mathematical loop IR is valid");
    assert!(matches!(
        emit_c(kernel),
        Err(CodegenError::AffineOverflow(_))
    ));
}

#[test]
fn rejects_overflowing_guard_before_it_can_guard_an_index() {
    let index = affine(0, &[("i", i64::MAX)]);
    rejects(&kernel(
        3,
        vec![Guard {
            expr: index.clone(),
            bound: 1,
        }],
        index,
    ));
}

#[test]
fn rejects_cancelled_raw_products() {
    rejects(&kernel(
        3,
        vec![Guard {
            expr: affine(0, &[("i", i64::MAX), ("i", -i64::MAX)]),
            bound: 1,
        }],
        affine(0, &[]),
    ));
}

#[test]
fn rejects_overflowing_prefix_before_later_cancellation() {
    rejects(&kernel(
        2,
        vec![Guard {
            expr: affine(0, &[("i", i64::MAX), ("i", 1), ("i", -i64::MAX)]),
            bound: 1,
        }],
        affine(0, &[]),
    ));
}

#[test]
fn rejects_overflowing_trailing_constant() {
    rejects(&kernel(
        2,
        vec![Guard {
            expr: affine(1, &[("i", i64::MAX)]),
            bound: 2,
        }],
        affine(0, &[]),
    ));
}

#[test]
fn subtraction_checks_its_positive_product_first() {
    let mut k = kernel(
        3,
        vec![Guard {
            expr: affine(0, &[("j", 1), ("i", -(1 << 62))]),
            bound: 1,
        }],
        affine(0, &[]),
    );
    k.nests[0].loops.push(Loop {
        name: "j".to_string(),
        extent: 1,
        vectorize: false,
        unroll: false,
    });
    rejects(&k);
}

#[test]
fn a_preceding_guard_protects_later_guards_stores_and_loads() {
    let index = affine(0, &[("i", i64::MAX)]);
    let mut k = kernel(
        3,
        vec![
            Guard {
                expr: affine(0, &[("i", 1)]),
                bound: 1,
            },
            Guard {
                expr: index.clone(),
                bound: 1,
            },
        ],
        index.clone(),
    );
    k.nests[0].body.value = Expr::Load(Access {
        buffer: 0,
        index: vec![index],
    });
    let source = emit_c(&k).unwrap();
    assert!(source.contains("if (9223372036854775807*i < 1L)"));
    assert!(source.contains("Y[9223372036854775807*i] = Y[9223372036854775807*i];"));
    ubsan_runs("guarded", &source, 7.0, 7.0);
}

#[test]
fn leading_negative_product_and_min_helper_can_reach_signed_minimum() {
    for (name, coefficient, extent) in [("leading", -(1 << 62), 3), ("minimum", i64::MIN, 2)] {
        let k = kernel(
            extent,
            vec![Guard {
                expr: affine(0, &[("i", coefficient)]),
                bound: 1,
            }],
            affine(0, &[]),
        );
        ubsan_runs(name, &emit_c(&k).unwrap(), 0.0, 1.0);
    }
}

#[test]
fn maximum_factor_split_of_small_extent_is_still_accepted() {
    let mut k = kernel(4, vec![], affine(0, &[("i", 1)]));
    k.buffers[0].shape = vec![4];
    let schedule = Schedule::parse(&format!("split(0, i, io, ii, {})", i64::MAX)).unwrap();
    let split = apply(&k, &schedule).unwrap();
    let source = emit_c(&split).unwrap();
    assert!(source.contains("9223372036854775807*io + ii"));
    assert!(!source.contains("__int128"));
    // The real split has MAX inner iterations, so do not run it. This bounded
    // equivalent exercises its safe product and first four in-bounds stores.
    let mut bounded = split;
    bounded.nests[0].loops[1].extent = 4;
    let source = emit_c(&bounded).unwrap();
    ubsan_runs_array(
        "split",
        &source,
        "float y[4] = {0}; affine_check(y); return y[0] != 1.0f || y[1] != 1.0f || y[2] != 1.0f || y[3] != 1.0f;",
    );
}

fn ubsan_runs(name: &str, source: &str, initial: f32, expected: f32) {
    ubsan_runs_array(
        name,
        source,
        &format!("float y[1] = {{{initial:?}f}}; affine_check(y); return y[0] != {expected:?}f;"),
    );
}

fn ubsan_runs_array(name: &str, source: &str, main: &str) {
    let dir = std::env::temp_dir().join(format!(
        "coffee-ten-affine-ubsan-{}-{name}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("kernel.c");
    let executable = dir.join("kernel");
    std::fs::write(&input, format!("{source}\nint main(void) {{ {main} }}\n")).unwrap();
    let compiled = Command::new("clang")
        .args([
            "-std=c99",
            "-O1",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-fsanitize=undefined",
            "-fno-sanitize-recover=all",
        ])
        .arg(&input)
        .arg("-o")
        .arg(&executable)
        .output()
        .expect("clang must be on PATH");
    assert!(
        compiled.status.success(),
        "{}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    let run = Command::new(&executable).output().unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
}
