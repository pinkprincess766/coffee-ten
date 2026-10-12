//! Parser for the text form of a kernel: the inverse of `Display for Kernel`.

use crate::{
    Access, AffineExpr, BinaryOp, Buffer, Expr, Guard, Kernel, Loop, LoopNest, Store, UnaryOp,
};
use std::fmt;

/// A syntax error in loop IR text. `line` is 1-based.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelParseError {
    pub line: usize,
    pub msg: String,
}

impl fmt::Display for KernelParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "loop IR line {}: {}", self.line, self.msg)
    }
}

impl std::error::Error for KernelParseError {}

/// Parses the text that `Display for Kernel` prints. Blank lines are skipped. Checks syntax
/// only; call `Kernel::validate` for bounds and loop names.
pub fn parse_kernel(text: &str) -> Result<Kernel, KernelParseError> {
    let mut lines = text
        .lines()
        .enumerate()
        .map(|(n, line)| (n + 1, line))
        .filter(|(_, line)| !line.trim().is_empty());
    let (header_line, header) = lines.next().unwrap_or((1, ""));
    let (name, buffers) = parse_header(header).map_err(at(header_line))?;

    let mut nests = Vec::new();
    let mut loops = Vec::new();
    let mut guards = Vec::new();
    let mut last_line = header_line;
    for (line, text) in lines {
        last_line = line;
        // Each loop or guard opens one level; the store line closes the nest.
        let depth = loops.len() + guards.len();
        let spaces = text.len() - text.trim_start_matches(' ').len();
        if spaces != 2 * depth {
            return Err(KernelParseError {
                line,
                msg: format!(
                    "bad indentation: expected {} spaces, got {spaces}",
                    2 * depth
                ),
            });
        }
        let body = &text[spaces..];
        if let Some(rest) = body.strip_prefix("for ") {
            if !guards.is_empty() {
                return Err(KernelParseError {
                    line,
                    msg: "loop after a guard".to_string(),
                });
            }
            loops.push(parse_loop(rest).map_err(at(line))?);
        } else if let Some(rest) = body.strip_prefix("if ") {
            guards.push(parse_guard(rest).map_err(at(line))?);
        } else {
            let store = parse_store(body, &buffers).map_err(at(line))?;
            nests.push(LoopNest {
                loops: std::mem::take(&mut loops),
                guards: std::mem::take(&mut guards),
                body: store,
            });
        }
    }
    if !loops.is_empty() || !guards.is_empty() {
        return Err(at(last_line)("expected a store line".to_string()));
    }
    Ok(Kernel {
        name,
        buffers,
        nests,
    })
}

fn at(line: usize) -> impl Fn(String) -> KernelParseError {
    move |msg| KernelParseError { line, msg }
}

/// Reads one line left to right. After a failed method the rest of the line is unspecified;
/// parsing stops at the first error.
struct Cursor<'a> {
    rest: &'a str,
}

impl<'a> Cursor<'a> {
    fn eat(&mut self, s: &str) -> bool {
        if let Some(rest) = self.rest.strip_prefix(s) {
            self.rest = rest;
            return true;
        }
        false
    }

    fn expect(&mut self, s: &str) -> Result<(), String> {
        if self.eat(s) {
            Ok(())
        } else {
            Err(format!("expected '{s}'"))
        }
    }

    fn end(&self) -> Result<(), String> {
        if self.rest.is_empty() {
            Ok(())
        } else {
            Err(format!("unexpected text {:?}", self.rest))
        }
    }

    fn take_while(&mut self, pred: impl Fn(char) -> bool) -> &'a str {
        let end = self.rest.find(|c| !pred(c)).unwrap_or(self.rest.len());
        let (taken, rest) = self.rest.split_at(end);
        self.rest = rest;
        taken
    }

    fn ident(&mut self) -> Result<&'a str, String> {
        if !self
            .rest
            .starts_with(|c: char| c.is_ascii_alphabetic() || c == '_')
        {
            return Err("expected a name".to_string());
        }
        Ok(self.take_while(|c| c.is_ascii_alphanumeric() || c == '_'))
    }

    fn digits(&mut self) -> &'a str {
        self.take_while(|c| c.is_ascii_digit())
    }

    fn unsigned(&mut self) -> Result<usize, String> {
        let digits = self.digits();
        digits
            .parse()
            .map_err(|_| format!("expected a number, found {digits:?}"))
    }

    /// `[-]term (" + " | " - " term)*`, where a term is `N*name`, `name` or a number. The
    /// constant may appear once.
    fn affine(&mut self) -> Result<AffineExpr, String> {
        let mut terms = Vec::new();
        let mut constant = None;
        let mut negative = self.eat("-");
        loop {
            let digits = self.digits();
            if digits.is_empty() {
                let name = self.ident()?;
                terms.push((name.to_string(), if negative { -1 } else { 1 }));
            } else if self.eat("*") {
                let name = self.ident()?;
                terms.push((name.to_string(), signed(negative, digits)?));
            } else if constant.is_none() {
                constant = Some(signed(negative, digits)?);
            } else {
                return Err("expected a name".to_string());
            }
            if self.eat(" + ") {
                negative = false;
            } else if self.eat(" - ") {
                negative = true;
            } else {
                break;
            }
        }
        Ok(AffineExpr {
            constant: constant.unwrap_or(0),
            terms,
        })
    }

    /// `name[index, ...]`, with `name` already read.
    fn access(&mut self, name: &str, buffers: &[Buffer]) -> Result<Access, String> {
        let buffer = buffers
            .iter()
            .position(|b| b.name == name)
            .ok_or_else(|| format!("unknown buffer {name}"))?;
        self.expect("[")?;
        let mut index = Vec::new();
        while !self.eat("]") {
            if !index.is_empty() {
                self.expect(", ")?;
            }
            index.push(self.affine()?);
        }
        Ok(Access { buffer, index })
    }

    /// An infix expression, or one operand. Infix operands are parenthesized by `Display`,
    /// so a second infix operator in a row is a syntax error at the caller's `end`.
    fn expr(&mut self, buffers: &[Buffer]) -> Result<Expr, String> {
        let lhs = self.operand(buffers)?;
        let op = if self.eat(" + ") {
            BinaryOp::Add
        } else if self.eat(" - ") {
            BinaryOp::Sub
        } else if self.eat(" * ") {
            BinaryOp::Mul
        } else if self.eat(" / ") {
            BinaryOp::Div
        } else {
            return Ok(lhs);
        };
        let rhs = self.operand(buffers)?;
        Ok(Expr::Binary(op, Box::new(lhs), Box::new(rhs)))
    }

    /// A load, a constant, `relu(e)`, `exp(e)`, `max(e, e)`, `-(e)`, or `(e)` around an infix `e`.
    fn operand(&mut self, buffers: &[Buffer]) -> Result<Expr, String> {
        if self.eat("(") {
            let e = self.expr(buffers)?;
            self.expect(")")?;
            Ok(e)
        } else if self.eat("-(") {
            let x = self.expr(buffers)?;
            self.expect(")")?;
            Ok(Expr::Unary(UnaryOp::Neg, Box::new(x)))
        } else if self
            .rest
            .starts_with(|c: char| c.is_ascii_alphabetic() || c == '_')
        {
            let name = self.ident()?;
            if self.rest.starts_with('[') {
                Ok(Expr::Load(self.access(name, buffers)?))
            } else if self.eat("(") {
                self.call(name, buffers)
            } else {
                // `inf` and `NaN`, the `{:?}` text of non-finite constants.
                name.parse()
                    .map(Expr::Const)
                    .map_err(|_| format!("bad constant {name}"))
            }
        } else {
            let token = self.take_while(|c| !" ,)]".contains(c));
            if token.is_empty() {
                return Err("expected an expression".to_string());
            }
            token
                .parse()
                .map(Expr::Const)
                .map_err(|_| format!("bad constant {token}"))
        }
    }

    /// The arguments of `name(`, with the `(` already read.
    fn call(&mut self, name: &str, buffers: &[Buffer]) -> Result<Expr, String> {
        let e = match name {
            "relu" => Expr::Unary(UnaryOp::Relu, Box::new(self.expr(buffers)?)),
            "exp" => Expr::Unary(UnaryOp::Exp, Box::new(self.expr(buffers)?)),
            "max" => {
                let a = self.expr(buffers)?;
                self.expect(", ")?;
                let b = self.expr(buffers)?;
                Expr::Binary(BinaryOp::Max, Box::new(a), Box::new(b))
            }
            _ => return Err(format!("unknown function {name}")),
        };
        self.expect(")")?;
        Ok(e)
    }
}

/// `digits` with a leading `-` when `negative`; parses i64::MIN too.
fn signed(negative: bool, digits: &str) -> Result<i64, String> {
    let text = if negative {
        format!("-{digits}")
    } else {
        digits.to_string()
    };
    text.parse()
        .map_err(|_| format!("expected a number, found {text:?}"))
}

/// `kernel NAME(BUF: f32[DIM, ...], ...)`.
fn parse_header(text: &str) -> Result<(String, Vec<Buffer>), String> {
    let mut c = Cursor { rest: text };
    c.expect("kernel ")?;
    let name = c.ident()?;
    c.expect("(")?;
    let mut buffers = Vec::new();
    while !c.eat(")") {
        if !buffers.is_empty() {
            c.expect(", ")?;
        }
        let buffer_name = c.ident()?;
        c.expect(": f32[")?;
        let mut shape = Vec::new();
        while !c.eat("]") {
            if !shape.is_empty() {
                c.expect(", ")?;
            }
            shape.push(c.unsigned()?);
        }
        buffers.push(Buffer {
            name: buffer_name.to_string(),
            shape,
        });
    }
    c.end()?;
    Ok((name.to_string(), buffers))
}

/// `NAME in 0..EXTENT[ (vectorize)]:`, with `for ` already read.
fn parse_loop(text: &str) -> Result<Loop, String> {
    let mut c = Cursor { rest: text };
    let name = c.ident()?;
    c.expect(" in 0..")?;
    let extent = c.unsigned()?;
    let vectorize = c.eat(" (vectorize)");
    c.expect(":")?;
    c.end()?;
    Ok(Loop {
        name: name.to_string(),
        extent,
        vectorize,
    })
}

/// `AFFINE < BOUND:`, with `if ` already read.
fn parse_guard(text: &str) -> Result<Guard, String> {
    let mut c = Cursor { rest: text };
    let expr = c.affine()?;
    c.expect(" < ")?;
    let negative = c.eat("-");
    let bound = signed(negative, c.digits())?;
    c.expect(":")?;
    c.end()?;
    Ok(Guard { expr, bound })
}

/// `BUF[index] = value`.
fn parse_store(text: &str, buffers: &[Buffer]) -> Result<Store, String> {
    let mut c = Cursor { rest: text };
    let name = c.ident()?;
    let target = c.access(name, buffers)?;
    c.expect(" = ")?;
    let value = c.expr(buffers)?;
    c.end()?;
    Ok(Store { target, value })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Schedule, Step, apply, lower, lower_elementwise};
    use coffee_ten_ir::{Graph, Shape};
    use proptest::prelude::*;

    fn lp(name: &str, extent: usize) -> Loop {
        Loop {
            name: name.to_string(),
            extent,
            vectorize: false,
        }
    }

    fn var(name: &str) -> AffineExpr {
        AffineExpr {
            constant: 0,
            terms: vec![(name.to_string(), 1)],
        }
    }

    fn aff(constant: i64, terms: &[(&str, i64)]) -> AffineExpr {
        AffineExpr {
            constant,
            terms: terms
                .iter()
                .map(|(name, coeff)| (name.to_string(), *coeff))
                .collect(),
        }
    }

    fn load(name: &str) -> Expr {
        Expr::Load(Access {
            buffer: 0,
            index: vec![var(name)],
        })
    }

    fn bin(op: BinaryOp, a: Expr, b: Expr) -> Expr {
        Expr::Binary(op, Box::new(a), Box::new(b))
    }

    /// `Y[index] = value` in one loop `i` of extent 2, with `Y` of shape `[1; index.len()]`.
    fn store_kernel(index: Vec<AffineExpr>, value: Expr) -> Kernel {
        Kernel {
            name: "k".to_string(),
            buffers: vec![Buffer {
                name: "Y".to_string(),
                shape: vec![1; index.len()],
            }],
            nests: vec![LoopNest {
                loops: vec![lp("i", 2)],
                guards: Vec::new(),
                body: Store {
                    target: Access { buffer: 0, index },
                    value,
                },
            }],
        }
    }

    /// The text parses back to `k`, and printing the parse gives the same text. `==` on f32
    /// cannot tell -0.0 from 0.0; the text can.
    #[test]
    fn exp_and_div_round_trip() {
        // exp(A) / B with B broadcast over rows: both new ops in one store.
        let mut g = coffee_ten_ir::Graph::new();
        let a = g.input("A", coffee_ten_ir::Shape::new(vec![2, 3])).unwrap();
        let b = g.input("B", coffee_ten_ir::Shape::new(vec![3])).unwrap();
        let e = g.unary(coffee_ten_ir::UnaryOp::Exp, a).unwrap();
        let y = g.binary(coffee_ten_ir::BinaryOp::Div, e, b).unwrap();
        let k = crate::lower_elementwise(&g, y, "k").unwrap();
        assert!(k.to_string().contains("exp(A[i0, i1]) / B[i1]"), "{k}");
        assert_round_trip(&k);
    }

    fn assert_round_trip(k: &Kernel) {
        let text = k.to_string();
        assert_eq!(parse_kernel(&text), Ok(k.clone()));
        assert_eq!(parse_kernel(&text).map(|p| p.to_string()), Ok(text));
    }

    fn matmul() -> Kernel {
        let mut g = Graph::new();
        let a = g
            .input("A", Shape::new(vec![4, 3]))
            .expect("test graph builds");
        let b = g
            .input("B", Shape::new(vec![3, 2]))
            .expect("test graph builds");
        let prod = g.matmul(a, b).expect("test graph builds");
        lower(&g, prod, "matmul").expect("test graph lowers")
    }

    fn bias_relu() -> Kernel {
        let mut g = Graph::new();
        let a = g
            .input("A", Shape::new(vec![2, 3]))
            .expect("test graph builds");
        let bias = g
            .input("bias", Shape::new(vec![3]))
            .expect("test graph builds");
        let sum = g
            .binary(coffee_ten_ir::BinaryOp::Add, a, bias)
            .expect("test graph builds");
        let out = g
            .unary(coffee_ten_ir::UnaryOp::Relu, sum)
            .expect("test graph builds");
        lower(&g, out, "bias_relu").expect("test graph lowers")
    }

    fn schedule(text: &str) -> Schedule {
        Schedule::parse(text).expect("test schedule parses")
    }

    #[test]
    fn matmul_round_trips() {
        assert_round_trip(&matmul());
    }

    #[test]
    fn bias_relu_round_trips() {
        assert_round_trip(&bias_relu());
    }

    #[test]
    fn scheduled_matmul_round_trips() {
        let k = apply(
            &matmul(),
            &schedule("split(1, k, ko, ki, 2); reorder(1, i, ko, ki, j); vectorize(1, j)"),
        )
        .expect("test schedule applies");
        assert!(k.to_string().contains("(vectorize)"));
        assert_round_trip(&k);
    }

    #[test]
    fn split_matmul_with_guard_round_trips() {
        // 3 is not a multiple of 2, so the split adds the guard `2*ko + ki < 3`.
        let k = apply(&matmul(), &schedule("split(1, k, ko, ki, 2)")).expect("test schedule");
        assert_eq!(k.nests[1].guards.len(), 1);
        assert!(k.to_string().contains("      if 2*ko + ki < 3:\n"));
        assert_round_trip(&k);
    }

    #[test]
    fn empty_kernel_and_rank_zero_buffers_round_trip() {
        assert_round_trip(&Kernel {
            name: "k".to_string(),
            buffers: Vec::new(),
            nests: Vec::new(),
        });
        let scalar = Kernel {
            name: "k".to_string(),
            buffers: vec![Buffer {
                name: "Y".to_string(),
                shape: Vec::new(),
            }],
            nests: vec![LoopNest {
                loops: Vec::new(),
                guards: Vec::new(),
                body: Store {
                    target: Access {
                        buffer: 0,
                        index: Vec::new(),
                    },
                    value: Expr::Const(0.0),
                },
            }],
        };
        assert_eq!(scalar.to_string(), "kernel k(Y: f32[])\nY[] = 0.0\n");
        assert_round_trip(&scalar);
    }

    #[test]
    fn affine_forms_round_trip() {
        let indices = [
            aff(0, &[]),
            aff(-3, &[]),
            aff(i64::MIN, &[]),
            aff(i64::MAX, &[]),
            aff(-1, &[("i", 1)]),
            aff(3, &[("i", -1)]),
            aff(-2, &[("io", 4), ("ii", 1)]),
            aff(0, &[("i", i64::MIN)]),
            aff(0, &[("i", -2), ("j", 1)]),
        ];
        for index in indices {
            assert_round_trip(&store_kernel(vec![index], Expr::Const(0.0)));
        }
    }

    #[test]
    fn affine_number_past_i64_is_rejected() {
        let too_big = i64::MAX as u128 + 1;
        let text = format!("kernel k(Y: f32[1])\nfor i in 0..1:\n  Y[{too_big}] = 0.0\n");
        assert!(parse_kernel(&text).is_err());
    }

    #[test]
    fn expressions_round_trip() {
        let neg = |x| Expr::Unary(UnaryOp::Neg, Box::new(x));
        let relu = |x| Expr::Unary(UnaryOp::Relu, Box::new(x));
        let value = bin(
            BinaryOp::Sub,
            Expr::Const(-1.0),
            bin(
                BinaryOp::Mul,
                neg(load("i")),
                relu(bin(BinaryOp::Add, load("i"), Expr::Const(1e-7))),
            ),
        );
        assert_round_trip(&store_kernel(vec![var("i")], value));
        let value = bin(
            BinaryOp::Max,
            Expr::Const(f32::INFINITY),
            neg(Expr::Const(-f32::MIN_POSITIVE)),
        );
        assert_round_trip(&store_kernel(vec![var("i")], value));
        let value = bin(
            BinaryOp::Sub,
            bin(BinaryOp::Sub, load("i"), load("i")),
            Expr::Const(f32::MAX),
        );
        assert_round_trip(&store_kernel(vec![var("i")], value));
    }

    #[test]
    fn negative_zero_keeps_its_sign() {
        let k = store_kernel(
            vec![var("i")],
            bin(BinaryOp::Add, Expr::Const(-0.0), load("i")),
        );
        assert!(k.to_string().contains("= -0.0 + Y[i]\n"));
        assert_round_trip(&k);
    }

    /// Parsing through f64 would round `16777217.000000001` to the tie 16777217 first, and
    /// then to even, 16777216. The f32 nearest to the decimal is 16777218.
    #[test]
    fn constants_are_rounded_once_to_f32() {
        let with = |text: &str| {
            parse_kernel(&format!(
                "kernel k(Y: f32[1])\nfor i in 0..1:\n  Y[i] = {text}\n"
            ))
        };
        assert!(with("16777218.0").is_ok());
        assert_eq!(with("16777217.000000001"), with("16777218.0"));
    }

    #[test]
    fn malformed_text_reports_its_line() {
        let cases = [
            // The `for` on line 4 has no `:`.
            (
                "kernel k(Y: f32[2, 2])\nfor i in 0..2:\n  for j in 0..2:\n    for x in 0..2\n      Y[i, j] = 0.0\n",
                4,
                "expected ':'",
            ),
            // Three spaces where two are expected.
            (
                "kernel k(Y: f32[2])\nfor i in 0..2:\n   Y[i] = 0.0\n",
                3,
                "bad indentation: expected 2 spaces, got 3",
            ),
            // The blank line 2 is skipped but still counted.
            (
                "kernel k(Y: f32[2])\n\nfor i in 0..2:\n   Y[i] = 0.0\n",
                4,
                "bad indentation: expected 2 spaces, got 3",
            ),
            (
                "kernel k(Y: f32[2])\nfor i in 0..2:\n  Z[i] = 0.0\n",
                3,
                "unknown buffer Z",
            ),
            (
                "kernel k(X: f32[2], Y: f32[2])\nfor i in 0..2:\n  Y[i] = foo(X[i])\n",
                3,
                "unknown function foo",
            ),
            (
                "kernel k(Y: f32[2])\nfor i in 0..2:\n  Y[i] = 1.0.0\n",
                3,
                "bad constant 1.0.0",
            ),
            (
                "kernel k(Y: f32[2])\nfor i in 0..2:\n",
                2,
                "expected a store line",
            ),
            (
                "kernel k(Y: f32[2])\nfor i in 0..2:\n  if i < 1:\n    for j in 0..2:\n      Y[i] = 0.0\n",
                4,
                "loop after a guard",
            ),
            ("for i in 0..2:\n  Y[i] = 0.0\n", 1, "expected 'kernel '"),
        ];
        for (text, line, msg) in cases {
            assert_eq!(
                parse_kernel(text),
                Err(KernelParseError {
                    line,
                    msg: msg.to_string(),
                }),
                "{text:?}"
            );
        }
    }

    #[test]
    fn error_display_names_the_line() {
        let e = KernelParseError {
            line: 4,
            msg: "expected ':'".to_string(),
        };
        assert_eq!(e.to_string(), "loop IR line 4: expected ':'");
    }

    /// Small random elementwise kernel, as in the `apply` tests: output rank 0..=3, dims 1..=5,
    /// 1..=3 inputs that broadcast to it, up to 5 ops over earlier nodes.
    fn elementwise_kernel() -> impl Strategy<Value = Kernel> {
        let out_dims = prop::collection::vec(1usize..=5, 0..=3);
        let input = (0usize..=3, prop::collection::vec(any::<bool>(), 3));
        let inputs = prop::collection::vec(input, 1..=3);
        let op = (0u8..6, any::<usize>(), any::<usize>());
        let ops = prop::collection::vec(op, 0..=5);
        (out_dims, inputs, ops).prop_map(|(out_dims, inputs, ops)| {
            let mut g = Graph::new();
            let mut nodes = vec![g.constant(0.5)];
            for (n, (rank, collapse)) in inputs.iter().enumerate() {
                let rank = (*rank).min(out_dims.len());
                let tail = &out_dims[out_dims.len() - rank..];
                let dims = tail
                    .iter()
                    .zip(collapse)
                    .map(|(&d, &one)| if one { 1 } else { d })
                    .collect();
                nodes.push(
                    g.input(&format!("x{n}"), Shape::new(dims))
                        .expect("names are distinct"),
                );
            }
            for (code, a, b) in ops {
                let (a, b) = (nodes[a % nodes.len()], nodes[b % nodes.len()]);
                let id = match code {
                    0 => g.unary(coffee_ten_ir::UnaryOp::Neg, a),
                    1 => g.unary(coffee_ten_ir::UnaryOp::Relu, a),
                    2 => g.binary(coffee_ten_ir::BinaryOp::Add, a, b),
                    3 => g.binary(coffee_ten_ir::BinaryOp::Sub, a, b),
                    4 => g.binary(coffee_ten_ir::BinaryOp::Mul, a, b),
                    _ => g.binary(coffee_ten_ir::BinaryOp::Max, a, b),
                };
                nodes.push(id.expect("all shapes broadcast to the output shape"));
            }
            let output = *nodes.last().expect("nodes has the constant");
            lower_elementwise(&g, output, "k").expect("small graphs lower")
        })
    }

    proptest! {
        #[test]
        fn split_elementwise_kernel_round_trips(
            kernel in elementwise_kernel(),
            nest_pick in any::<usize>(),
            loop_pick in any::<usize>(),
            factor in 1usize..=6,
        ) {
            let nest = nest_pick % kernel.nests.len();
            let loops = &kernel.nests[nest].loops;
            prop_assume!(!loops.is_empty());
            let loop_name = loops[loop_pick % loops.len()].name.clone();
            let steps = vec![Step::Split {
                nest,
                loop_name,
                outer: "so".to_string(),
                inner: "si".to_string(),
                factor,
            }];
            let split = apply(&kernel, &Schedule { steps }).expect("split applies");
            let text = split.to_string();
            prop_assert_eq!(parse_kernel(&text), Ok(split.clone()));
            prop_assert_eq!(parse_kernel(&text).map(|p| p.to_string()), Ok(text));
        }
    }
}
