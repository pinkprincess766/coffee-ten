//! C99 backend for coffee-ten: prints a loop IR kernel as one C function.

use coffee_ten_loop::{Access, BinaryOp, Buffer, Expr, Kernel, LoopIrError, LoopNest, UnaryOp};
use std::fmt;

mod compile;
pub use compile::{CompileError, CompiledKernel, RunError, TEST_FLAGS, compile};

/// Why `emit_c` refused a kernel.
#[derive(Clone, Debug, PartialEq)]
pub enum CodegenError {
    /// `Kernel::validate` failed.
    Invalid(LoopIrError),
    /// A kernel, buffer or loop name is not a C identifier.
    BadIdentifier(String),
    /// Two buffers have the same name.
    DuplicateBuffer(String),
    /// A constant is NaN or infinite.
    NonFiniteConst(f32),
    /// A name is a C keyword or a helper name, or a loop has the name of a buffer.
    NameClash(String),
    /// A buffer has more elements than a C `long` index can address.
    BufferTooLarge(String),
}

impl fmt::Display for CodegenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(e) => write!(f, "invalid kernel: {e}"),
            Self::BadIdentifier(name) => write!(f, "{name:?} is not a C identifier"),
            Self::DuplicateBuffer(name) => write!(f, "buffer {name} is declared twice"),
            Self::NonFiniteConst(value) => write!(f, "constant {value} is not finite"),
            Self::NameClash(name) => write!(f, "name {name} clashes with C or another name"),
            Self::BufferTooLarge(name) => write!(f, "buffer {name} is too large to index"),
        }
    }
}

impl std::error::Error for CodegenError {}

const RELU_HELPER: &str =
    "static inline float ct_relu(float x) { return x > 0.0f ? x : 0.0f; }\n\n";
const MAX_HELPER: &str =
    "static inline float ct_max(float a, float b) { return a > b ? a : b; }\n\n";

/// Names a kernel, buffer or loop must not take: C99 keywords and our helpers.
const RESERVED: [&str; 40] = [
    "auto",
    "break",
    "case",
    "char",
    "const",
    "continue",
    "default",
    "do",
    "double",
    "else",
    "enum",
    "extern",
    "float",
    "for",
    "goto",
    "if",
    "inline",
    "int",
    "long",
    "register",
    "restrict",
    "return",
    "short",
    "signed",
    "sizeof",
    "static",
    "struct",
    "switch",
    "typedef",
    "union",
    "unsigned",
    "void",
    "volatile",
    "while",
    "_Bool",
    "_Complex",
    "_Imaginary",
    "ct_relu",
    "ct_max",
    "ct_entry",
];

/// Emits `kernel` as one C99 function. Buffers that are stored to are `restrict` outputs.
pub fn emit_c(kernel: &Kernel) -> Result<String, CodegenError> {
    kernel.validate().map_err(CodegenError::Invalid)?;
    check_name(&kernel.name)?;
    for (pos, buffer) in kernel.buffers.iter().enumerate() {
        check_name(&buffer.name)?;
        if kernel.buffers[..pos].iter().any(|b| b.name == buffer.name) {
            return Err(CodegenError::DuplicateBuffer(buffer.name.clone()));
        }
        let elements = buffer
            .shape
            .iter()
            .try_fold(1usize, |acc, &n| acc.checked_mul(n))
            .and_then(|n| i64::try_from(n).ok());
        if elements.is_none() {
            return Err(CodegenError::BufferTooLarge(buffer.name.clone()));
        }
    }
    let mut used = Helpers::default();
    for nest in &kernel.nests {
        for l in &nest.loops {
            check_name(&l.name)?;
            // A loop variable with a buffer's name would shadow the parameter.
            if kernel.buffers.iter().any(|b| b.name == l.name) {
                return Err(CodegenError::NameClash(l.name.clone()));
            }
        }
        scan_expr(&nest.body.value, &mut used)?;
    }

    let params: Vec<String> = kernel
        .buffers
        .iter()
        .enumerate()
        .map(|(id, buffer)| {
            if kernel.nests.iter().any(|n| n.body.target.buffer == id) {
                format!("float *restrict {}", buffer.name)
            } else {
                format!("const float *{}", buffer.name)
            }
        })
        .collect();
    let params = if params.is_empty() {
        "void".to_string()
    } else {
        params.join(", ")
    };

    let mut out = String::new();
    if used.relu {
        out.push_str(RELU_HELPER);
    }
    if used.max {
        out.push_str(MAX_HELPER);
    }
    out.push_str(&format!("void {}({params}) {{\n", kernel.name));
    for nest in &kernel.nests {
        nest_c(&mut out, &kernel.buffers, nest);
    }
    out.push_str("}\n");
    Ok(out)
}

#[derive(Default)]
struct Helpers {
    relu: bool,
    max: bool,
}

/// Rejects non-finite constants and records which helper functions the expression calls.
fn scan_expr(e: &Expr, used: &mut Helpers) -> Result<(), CodegenError> {
    match e {
        Expr::Load(_) => Ok(()),
        Expr::Const(value) => {
            if value.is_finite() {
                Ok(())
            } else {
                Err(CodegenError::NonFiniteConst(*value))
            }
        }
        Expr::Unary(op, x) => {
            used.relu |= *op == UnaryOp::Relu;
            scan_expr(x, used)
        }
        Expr::Binary(op, a, b) => {
            used.max |= *op == BinaryOp::Max;
            scan_expr(a, used)?;
            scan_expr(b, used)
        }
    }
}

fn check_name(name: &str) -> Result<(), CodegenError> {
    let mut chars = name.chars();
    let starts_ok = matches!(chars.next(), Some(c) if c == '_' || c.is_ascii_alphabetic());
    if !(starts_ok && chars.all(|c| c == '_' || c.is_ascii_alphanumeric())) {
        return Err(CodegenError::BadIdentifier(name.to_string()));
    }
    if RESERVED.contains(&name) {
        return Err(CodegenError::NameClash(name.to_string()));
    }
    Ok(())
}

fn indent(level: usize) -> String {
    "    ".repeat(level)
}

fn nest_c(out: &mut String, buffers: &[Buffer], nest: &LoopNest) {
    for (depth, l) in nest.loops.iter().enumerate() {
        if l.vectorize {
            out.push_str(&format!(
                "{}#pragma clang loop vectorize(enable)\n",
                indent(depth + 1)
            ));
        }
        out.push_str(&format!(
            "{}for (long {name} = 0; {name} < {extent}; {name}++) {{\n",
            indent(depth + 1),
            name = l.name,
            extent = l.extent,
        ));
    }
    let mut depth = nest.loops.len();
    for guard in &nest.guards {
        depth += 1;
        out.push_str(&format!(
            "{}if ({} < {}L) {{\n",
            indent(depth),
            guard.expr,
            guard.bound
        ));
    }
    let target = access_c(buffers, &nest.body.target);
    let value = expr_c(buffers, &nest.body.value);
    out.push_str(&format!("{}{target} = {value};\n", indent(depth + 1)));
    for depth in (0..depth).rev() {
        out.push_str(&format!("{}}}\n", indent(depth + 1)));
    }
}

/// Row-major offset `(e0) * s0 + ... + (e_last)`, where `s_d` is the product of the later dims.
fn access_c(buffers: &[Buffer], access: &Access) -> String {
    let buffer = buffers
        .get(access.buffer)
        .expect("invariant: validate() checked buffer indices");
    if access.index.is_empty() {
        return format!("{}[0]", buffer.name);
    }
    let last = access.index.len() - 1;
    let terms: Vec<String> = access
        .index
        .iter()
        .enumerate()
        .map(|(dim, e)| {
            if dim == last {
                format!("({e})")
            } else {
                // Cannot overflow: an accessed buffer has no zero dim (validate) and its
                // element count fits in i64 (checked in emit_c), so the stride does too.
                let stride: usize = buffer.shape[dim + 1..].iter().product();
                format!("({e}) * {stride}")
            }
        })
        .collect();
    format!("{}[{}]", buffer.name, terms.join(" + "))
}

fn expr_c(buffers: &[Buffer], e: &Expr) -> String {
    match e {
        Expr::Load(access) => access_c(buffers, access),
        Expr::Const(value) => format!("{value:?}f"),
        Expr::Unary(UnaryOp::Neg, x) => format!("-({})", expr_c(buffers, x)),
        Expr::Unary(UnaryOp::Relu, x) => format!("ct_relu({})", expr_c(buffers, x)),
        Expr::Binary(BinaryOp::Max, a, b) => {
            format!("ct_max({}, {})", expr_c(buffers, a), expr_c(buffers, b))
        }
        Expr::Binary(BinaryOp::Add, a, b) => infix_c(buffers, "+", a, b),
        Expr::Binary(BinaryOp::Sub, a, b) => infix_c(buffers, "-", a, b),
        Expr::Binary(BinaryOp::Mul, a, b) => infix_c(buffers, "*", a, b),
    }
}

fn infix_c(buffers: &[Buffer], op: &str, a: &Expr, b: &Expr) -> String {
    format!("{} {op} {}", operand_c(buffers, a), operand_c(buffers, b))
}

/// An infix operand is parenthesized if it is itself an infix Add, Sub or Mul.
fn operand_c(buffers: &[Buffer], e: &Expr) -> String {
    let text = expr_c(buffers, e);
    if matches!(
        e,
        Expr::Binary(BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul, _, _)
    ) {
        format!("({text})")
    } else {
        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use coffee_ten_loop::{AffineExpr, Guard, Loop, Schedule, Store, apply};
    use std::process::Command;

    fn lp(name: &str, extent: usize) -> Loop {
        Loop {
            name: name.to_string(),
            extent,
            vectorize: false,
        }
    }

    fn buf(name: &str, shape: &[usize]) -> Buffer {
        Buffer {
            name: name.to_string(),
            shape: shape.to_vec(),
        }
    }

    fn var(name: &str) -> AffineExpr {
        AffineExpr {
            constant: 0,
            terms: vec![(name.to_string(), 1)],
        }
    }

    fn cst(constant: i64) -> AffineExpr {
        AffineExpr {
            constant,
            terms: vec![],
        }
    }

    fn load(buffer: usize, index: Vec<AffineExpr>) -> Expr {
        Expr::Load(Access { buffer, index })
    }

    fn bin(op: BinaryOp, a: Expr, b: Expr) -> Expr {
        Expr::Binary(op, Box::new(a), Box::new(b))
    }

    fn un(op: UnaryOp, x: Expr) -> Expr {
        Expr::Unary(op, Box::new(x))
    }

    fn loop_nest(loops: Vec<Loop>, buffer: usize, index: Vec<AffineExpr>, value: Expr) -> LoopNest {
        LoopNest {
            loops,
            guards: Vec::new(),
            body: Store {
                target: Access { buffer, index },
                value,
            },
        }
    }

    fn kernel(buffers: Vec<Buffer>, nests: Vec<LoopNest>) -> Kernel {
        Kernel {
            name: "k".to_string(),
            buffers,
            nests,
        }
    }

    fn matmul() -> Kernel {
        Kernel {
            name: "matmul".to_string(),
            buffers: vec![buf("A", &[4, 3]), buf("B", &[3, 2]), buf("C", &[4, 2])],
            nests: vec![
                loop_nest(
                    vec![lp("i", 4), lp("j", 2)],
                    2,
                    vec![var("i"), var("j")],
                    Expr::Const(0.0),
                ),
                loop_nest(
                    vec![lp("i", 4), lp("k", 3), lp("j", 2)],
                    2,
                    vec![var("i"), var("j")],
                    bin(
                        BinaryOp::Add,
                        load(2, vec![var("i"), var("j")]),
                        bin(
                            BinaryOp::Mul,
                            load(0, vec![var("i"), var("k")]),
                            load(1, vec![var("k"), var("j")]),
                        ),
                    ),
                ),
            ],
        }
    }

    fn bias_relu() -> Kernel {
        kernel(
            vec![buf("A", &[2, 3]), buf("bias", &[3]), buf("Y", &[2, 3])],
            vec![loop_nest(
                vec![lp("i", 2), lp("j", 3)],
                2,
                vec![var("i"), var("j")],
                un(
                    UnaryOp::Relu,
                    bin(
                        BinaryOp::Add,
                        load(0, vec![var("i"), var("j")]),
                        load(1, vec![var("j")]),
                    ),
                ),
            )],
        )
    }

    /// Compiles `source` with clang as C99 under -Werror; panics if clang is missing or rejects it.
    fn assert_clang_accepts(name: &str, source: &str) {
        let pid = std::process::id();
        let c_path = std::env::temp_dir().join(format!("coffee-ten-{pid}-{name}.c"));
        let o_path = std::env::temp_dir().join(format!("coffee-ten-{pid}-{name}.c.o"));
        std::fs::write(&c_path, source).expect("write C source");
        let output = Command::new("clang")
            .args(["-std=c99", "-Wall", "-Wextra", "-Werror", "-c"])
            .arg(&c_path)
            .arg("-o")
            .arg(&o_path)
            .output()
            .expect("clang must be on PATH");
        let _ = std::fs::remove_file(&o_path);
        std::fs::remove_file(&c_path).expect("remove C source");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(output.status.success(), "clang rejected {name}:\n{stderr}");
    }

    #[test]
    fn matmul_text_matches_reference() {
        assert_eq!(
            emit_c(&matmul()).unwrap(),
            concat!(
                "void matmul(const float *A, const float *B, float *restrict C) {\n",
                "    for (long i = 0; i < 4; i++) {\n",
                "        for (long j = 0; j < 2; j++) {\n",
                "            C[(i) * 2 + (j)] = 0.0f;\n",
                "        }\n",
                "    }\n",
                "    for (long i = 0; i < 4; i++) {\n",
                "        for (long k = 0; k < 3; k++) {\n",
                "            for (long j = 0; j < 2; j++) {\n",
                "                C[(i) * 2 + (j)] = C[(i) * 2 + (j)] + (A[(i) * 3 + (k)] * B[(k) * 2 + (j)]);\n",
                "            }\n",
                "        }\n",
                "    }\n",
                "}\n",
            )
        );
    }

    /// The matmul with K = 3 split by 2, so the update nest needs a guard.
    fn matmul_split_k() -> Kernel {
        let schedule = Schedule::parse("split(1, k, ko, ki, 2)").unwrap();
        apply(&matmul(), &schedule).unwrap()
    }

    #[test]
    fn guarded_matmul_text_matches_reference() {
        assert_eq!(
            emit_c(&matmul_split_k()).unwrap(),
            concat!(
                "void matmul(const float *A, const float *B, float *restrict C) {\n",
                "    for (long i = 0; i < 4; i++) {\n",
                "        for (long j = 0; j < 2; j++) {\n",
                "            C[(i) * 2 + (j)] = 0.0f;\n",
                "        }\n",
                "    }\n",
                "    for (long i = 0; i < 4; i++) {\n",
                "        for (long ko = 0; ko < 2; ko++) {\n",
                "            for (long ki = 0; ki < 2; ki++) {\n",
                "                for (long j = 0; j < 2; j++) {\n",
                "                    if (2*ko + ki < 3L) {\n",
                "                        C[(i) * 2 + (j)] = C[(i) * 2 + (j)] + (A[(i) * 3 + (2*ko + ki)] * B[(2*ko + ki) * 2 + (j)]);\n",
                "                    }\n",
                "                }\n",
                "            }\n",
                "        }\n",
                "    }\n",
                "}\n",
            )
        );
    }

    #[test]
    fn vectorize_puts_one_pragma_before_the_innermost_for() {
        let schedule = Schedule::parse("vectorize(1, j)").unwrap();
        let text = emit_c(&apply(&matmul(), &schedule).unwrap()).unwrap();
        let mut lines: Vec<&str> = text.lines().collect();
        let pragmas: Vec<usize> = (0..lines.len())
            .filter(|&n| lines[n].contains("#pragma"))
            .collect();
        assert_eq!(pragmas.len(), 1, "{text}");
        let at = pragmas[0];
        assert_eq!(
            lines[at],
            "            #pragma clang loop vectorize(enable)"
        );
        assert_eq!(lines[at + 1], "            for (long j = 0; j < 2; j++) {");
        // Nest 0 also has a `j` loop; the marked one is the last in the text, in nest 1.
        assert_eq!(
            lines.iter().rposition(|l| l.contains("for (long j")),
            Some(at + 1)
        );
        // Nothing else changes.
        lines.remove(at);
        assert_eq!(lines.join("\n") + "\n", emit_c(&matmul()).unwrap());
    }

    #[test]
    fn two_guards_nest_in_order() {
        let mut k = matmul_split_k();
        k.nests[1].guards.push(Guard {
            expr: var("ki"),
            bound: 2,
        });
        let c = emit_c(&k).unwrap();
        assert!(c.contains(concat!(
            "                    if (2*ko + ki < 3L) {\n",
            "                        if (ki < 2L) {\n",
            "                            C[(i) * 2 + (j)] = ",
        )));
        assert!(c.contains(concat!(
            "                        }\n",
            "                    }\n",
            "                }\n",
        )));
    }

    #[test]
    fn relu_bias_uses_relu_helper_only() {
        let c = emit_c(&bias_relu()).unwrap();
        assert!(c.contains("static inline float ct_relu"));
        assert!(!c.contains("ct_max"));
        assert!(
            c.contains("            Y[(i) * 3 + (j)] = ct_relu(A[(i) * 3 + (j)] + bias[(j)]);\n")
        );
        assert!(c.contains("void k(const float *A, const float *bias, float *restrict Y) {\n"));
    }

    #[test]
    fn max_and_neg_emit_helper_and_calls() {
        let k = kernel(
            vec![buf("A", &[4]), buf("Y", &[4])],
            vec![loop_nest(
                vec![lp("i", 4)],
                1,
                vec![var("i")],
                bin(
                    BinaryOp::Max,
                    load(0, vec![var("i")]),
                    un(UnaryOp::Neg, load(0, vec![var("i")])),
                ),
            )],
        );
        assert_eq!(
            emit_c(&k).unwrap(),
            concat!(
                "static inline float ct_max(float a, float b) { return a > b ? a : b; }\n",
                "\n",
                "void k(const float *A, float *restrict Y) {\n",
                "    for (long i = 0; i < 4; i++) {\n",
                "        Y[(i)] = ct_max(A[(i)], -(A[(i)]));\n",
                "    }\n",
                "}\n",
            )
        );
    }

    #[test]
    fn constants_print_as_c_float_literals() {
        assert_eq!(expr_c(&[], &Expr::Const(0.0)), "0.0f");
        assert_eq!(expr_c(&[], &Expr::Const(-0.0)), "-0.0f");
        assert_eq!(expr_c(&[], &Expr::Const(0.5)), "0.5f");
        assert_eq!(expr_c(&[], &Expr::Const(1e-7)), "1e-7f");
    }

    #[test]
    fn rank_zero_buffer_and_empty_nest() {
        let k = kernel(
            vec![buf("s", &[])],
            vec![loop_nest(vec![], 0, vec![], Expr::Const(1.0))],
        );
        assert_eq!(
            emit_c(&k).unwrap(),
            "void k(float *restrict s) {\n    s[0] = 1.0f;\n}\n"
        );
    }

    #[test]
    fn buffer_too_large_for_long_is_rejected() {
        // 2 * 2^31 * 2^31 = 2^63 elements: one more than i64::MAX.
        let k = kernel(
            vec![buf("Y", &[2]), buf("A", &[2, 1 << 31, 1 << 31])],
            vec![loop_nest(
                vec![lp("i", 2)],
                0,
                vec![var("i")],
                load(1, vec![var("i"), cst(0), cst(0)]),
            )],
        );
        assert_eq!(
            emit_c(&k),
            Err(CodegenError::BufferTooLarge("A".to_string()))
        );
        // 2^62 elements still fit.
        let mut k = k;
        k.buffers[1].shape = vec![2, 1 << 31, 1 << 30];
        assert!(emit_c(&k).is_ok());
    }

    #[test]
    fn loop_named_like_a_buffer_is_rejected() {
        let mut k = matmul();
        k.nests[0].loops[0].name = "A".to_string();
        k.nests[0].body.target.index[0] = var("A");
        assert_eq!(emit_c(&k), Err(CodegenError::NameClash("A".to_string())));
    }

    #[test]
    fn keyword_and_helper_names_are_rejected() {
        let mut k = bias_relu();
        k.buffers[1].name = "int".to_string();
        assert_eq!(emit_c(&k), Err(CodegenError::NameClash("int".to_string())));
        k.buffers[1].name = "ct_relu".to_string();
        assert_eq!(
            emit_c(&k),
            Err(CodegenError::NameClash("ct_relu".to_string()))
        );
    }

    #[test]
    fn invalid_kernel_is_rejected() {
        let k = kernel(
            vec![buf("Y", &[4])],
            vec![loop_nest(
                vec![lp("i", 0)],
                0,
                vec![var("i")],
                Expr::Const(0.0),
            )],
        );
        assert_eq!(
            emit_c(&k),
            Err(CodegenError::Invalid(LoopIrError::EmptyExtent {
                nest: 0,
                loop_name: "i".to_string()
            }))
        );
    }

    #[test]
    fn bad_kernel_name_is_rejected() {
        let mut k = matmul();
        k.name = "my-kernel".to_string();
        assert_eq!(
            emit_c(&k),
            Err(CodegenError::BadIdentifier("my-kernel".to_string()))
        );
    }

    #[test]
    fn bad_buffer_name_is_rejected() {
        let mut k = bias_relu();
        k.buffers[1].name = "1x".to_string();
        assert_eq!(
            emit_c(&k),
            Err(CodegenError::BadIdentifier("1x".to_string()))
        );
    }

    #[test]
    fn duplicate_buffer_name_is_rejected() {
        let mut k = matmul();
        k.buffers[1].name = "A".to_string();
        assert_eq!(
            emit_c(&k),
            Err(CodegenError::DuplicateBuffer("A".to_string()))
        );
    }

    #[test]
    fn nan_constant_is_rejected() {
        let k = kernel(
            vec![buf("Y", &[1])],
            vec![loop_nest(
                vec![lp("i", 1)],
                0,
                vec![var("i")],
                Expr::Const(f32::NAN),
            )],
        );
        assert!(matches!(
            emit_c(&k),
            Err(CodegenError::NonFiniteConst(value)) if value.is_nan()
        ));
    }

    #[test]
    fn matmul_compiles_with_clang() {
        assert_clang_accepts("matmul", &emit_c(&matmul()).unwrap());
    }

    #[test]
    fn guarded_matmul_compiles_with_clang() {
        assert_clang_accepts("matmul_split_k", &emit_c(&matmul_split_k()).unwrap());
    }

    #[test]
    fn bias_relu_compiles_with_clang() {
        assert_clang_accepts("bias_relu", &emit_c(&bias_relu()).unwrap());
    }
}
