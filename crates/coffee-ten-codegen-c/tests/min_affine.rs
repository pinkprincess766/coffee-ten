//! Finite kernels whose signed affine minima must stay signed in C.

use coffee_ten_codegen_c::compile;
use coffee_ten_loop::{Access, AffineExpr, Buffer, Expr, Guard, Kernel, Loop, LoopNest, Store};

fn check_guard(name: &str, expr: AffineExpr, loops: Vec<Loop>) {
    let kernel = Kernel {
        name: name.to_string(),
        buffers: vec![Buffer {
            name: "Y".to_string(),
            shape: vec![],
        }],
        nests: vec![LoopNest {
            loops,
            guards: vec![Guard { expr, bound: 0 }],
            body: Store {
                target: Access {
                    buffer: 0,
                    index: vec![],
                },
                value: Expr::Const(1.0),
            },
        }],
    };
    kernel.validate().unwrap();
    let cache = std::env::temp_dir().join(format!(
        "coffee-ten-min-affine-{}-{name}",
        std::process::id()
    ));
    let compiled = compile(&kernel, &cache).unwrap();
    let mut output = [-7.0f32];
    compiled.run(&mut [&mut output]).unwrap();
    let _ = std::fs::remove_dir_all(cache);
    assert_eq!(
        output,
        [1.0],
        "a negative signed guard must execute its store"
    );
}

#[test]
fn minimum_constant_guard_executes() {
    check_guard(
        "min_constant",
        AffineExpr {
            constant: i64::MIN,
            terms: vec![],
        },
        vec![],
    );
}

#[test]
fn minimum_coefficient_guard_executes_for_zero_iteration_value() {
    // r is only zero, so MIN*r - 1 is exactly -1 with no signed overflow.
    check_guard(
        "min_coefficient",
        AffineExpr {
            constant: -1,
            terms: vec![("r".to_string(), i64::MIN)],
        },
        vec![Loop {
            name: "r".to_string(),
            extent: 1,
            vectorize: false,
            unroll: false,
        }],
    );
}
