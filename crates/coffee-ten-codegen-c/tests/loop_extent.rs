//! A scalar store does not constrain its unused loop's extent through buffer indexing.

#[cfg(target_pointer_width = "64")]
use coffee_ten_codegen_c::CodegenError;
use coffee_ten_codegen_c::emit_c;
use coffee_ten_loop::{Access, Buffer, Expr, Kernel, Loop, LoopNest, Store};

fn scalar_loop(extent: usize) -> Kernel {
    Kernel {
        name: "scalar_loop".to_string(),
        buffers: vec![Buffer {
            name: "Y".to_string(),
            shape: vec![],
        }],
        nests: vec![LoopNest {
            loops: vec![Loop {
                name: "r".to_string(),
                extent,
                vectorize: false,
                unroll: false,
            }],
            guards: vec![],
            body: Store {
                target: Access {
                    buffer: 0,
                    index: vec![],
                },
                value: Expr::Const(1.0),
            },
        }],
    }
}

#[test]
#[cfg(target_pointer_width = "64")]
fn unused_loop_cannot_exceed_signed_long_extent() {
    for extent in [i64::MAX as usize + 1, usize::MAX] {
        let kernel = scalar_loop(extent);
        kernel.validate().unwrap();
        assert_eq!(
            emit_c(&kernel),
            Err(CodegenError::LoopTooLarge {
                nest: 0,
                loop_name: "r".to_string(),
                extent
            })
        );
    }
}

#[test]
fn largest_representable_loop_extent_is_accepted_without_running_it() {
    let extent = usize::try_from(i64::MAX).unwrap_or(usize::MAX);
    let source = emit_c(&scalar_loop(extent)).unwrap();
    assert!(source.contains(&format!("for (long r = 0; r < {extent}; r++)")));
}
