//! Test-only grammar and integer input generation shared by the N7 test drivers.
use coffee_ten_ir::{BinaryOp, Graph, NodeId, ReduceOp, Shape, UnaryOp};
use std::collections::BTreeMap;

pub const CORPUS: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/fuzz/corpus-v1.tsv"
));

#[derive(Clone, Debug)]
pub struct Case {
    pub index: usize,
    pub seed: u64,
    pub kind: String,
    pub dims: Vec<usize>,
    pub axis: usize,
    pub pre: usize,
    pub post: usize,
    pub input_hash: String,
}

#[derive(Clone, Debug)]
pub struct Input {
    pub name: &'static str,
    pub dims: Vec<usize>,
    pub data: Vec<f32>,
}

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 {
            0x9e3779b97f4a7c15
        } else {
            seed.wrapping_mul(0x9e3779b97f4a7c15)
        })
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545f4914f6cdd1d)
    }
}

impl Case {
    pub fn parse(line: &str) -> Result<Self, String> {
        let fields: Vec<_> = line.trim().split('\t').collect();
        if fields.len() != 8 {
            return Err("expected eight TSV fields".into());
        }
        let number = |s: &str| {
            s.parse::<usize>()
                .map_err(|_| "invalid integer".to_string())
        };
        let case = Self {
            index: number(fields[0])?,
            seed: u64::from_str_radix(fields[1], 16).map_err(|_| "invalid seed")?,
            kind: fields[2].into(),
            dims: fields[3].split(',').map(number).collect::<Result<_, _>>()?,
            axis: number(fields[4])?,
            pre: number(fields[5])?,
            post: number(fields[6])?,
            input_hash: fields[7].into(),
        };
        let matrix = matches!(case.kind.as_str(), "matmul" | "chain");
        let rank = if matrix { 2 } else { case.dims.len() };
        if case.index >= 128
            || !matches!(
                case.kind.as_str(),
                "matmul" | "chain" | "reduce" | "elementwise"
            )
            || !(1..=3).contains(&case.dims.len())
            || (matrix && case.dims.len() != 3)
            || case.dims.iter().any(|d| !(1..=4).contains(d))
            || case.axis >= rank
            || case.pre >= 4
            || case.post >= 4
            || case.input_hash.len() != 16
        {
            return Err("case exceeds the v1 grammar/bounds".into());
        }
        if case.input_digest() != case.input_hash {
            return Err("input fingerprint mismatch".into());
        }
        Ok(case)
    }
    pub fn row(&self) -> String {
        format!(
            "{}\t{:016x}\t{}\t{}\t{}\t{}\t{}\t{}",
            self.index,
            self.seed,
            self.kind,
            self.dims
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(","),
            self.axis,
            self.pre,
            self.post,
            self.input_hash
        )
    }
    fn specs(&self) -> Vec<(&'static str, Vec<usize>)> {
        if matches!(self.kind.as_str(), "matmul" | "chain") {
            let (m, k, n) = (self.dims[0], self.dims[1], self.dims[2]);
            let mut inputs = vec![("A", vec![m, k]), ("B", vec![k, n])];
            if self.kind == "chain" {
                inputs.push(("C", vec![1, n]));
            }
            if self.post == 1 {
                inputs.push(("bias", vec![n]));
            }
            inputs
        } else {
            let mut inputs = vec![("X", self.dims.clone())];
            if self.kind == "elementwise" {
                inputs.push(("B", vec![*self.dims.last().unwrap()]));
            }
            inputs
        }
    }
    pub fn inputs(&self) -> Vec<Input> {
        self.specs()
            .into_iter()
            .enumerate()
            .map(|(ordinal, (name, dims))| {
                let mut rng =
                    Rng::new(self.seed ^ (ordinal as u64 + 1).wrapping_mul(0xd1b54a32d192ed03));
                let data = (0..dims.iter().product())
                    .map(|i| {
                        let u = rng.next();
                        let sign = ((u & 1) as u32) << 31;
                        let bits = if i % 11 == 10 {
                            sign
                        } else {
                            sign | ((120 + ((u >> 1) & 15)) as u32) << 23
                                | ((u >> 5) & 0x7fffff) as u32
                        };
                        f32::from_bits(bits)
                    })
                    .collect();
                Input { name, dims, data }
            })
            .collect()
    }
    pub fn input_digest(&self) -> String {
        let mut hash = 0xcbf29ce484222325u64;
        let mut feed = |bytes: &[u8]| {
            for &b in bytes {
                hash = (hash ^ u64::from(b)).wrapping_mul(0x100000001b3);
            }
        };
        for input in self.inputs() {
            feed(&(input.name.len() as u64).to_le_bytes());
            feed(input.name.as_bytes());
            feed(&(input.dims.len() as u64).to_le_bytes());
            for d in input.dims {
                feed(&(d as u64).to_le_bytes());
            }
            for x in input.data {
                feed(&x.to_bits().to_le_bytes());
            }
        }
        format!("{hash:016x}")
    }
    pub fn program(&self) -> (Graph, NodeId, Vec<Input>) {
        let inputs = self.inputs();
        let mut g = Graph::new();
        let ids: BTreeMap<_, _> = inputs
            .iter()
            .map(|i| (i.name, g.input(i.name, Shape::new(i.dims.clone())).unwrap()))
            .collect();
        let pre = |g: &mut Graph, x: NodeId| match self.pre {
            0 => x,
            1 => {
                let c = g.constant(0.1);
                g.binary(BinaryOp::Mul, x, c).unwrap()
            }
            2 => g.unary(UnaryOp::Relu, x).unwrap(),
            _ => g.unary(UnaryOp::Neg, x).unwrap(),
        };
        let output = if matches!(self.kind.as_str(), "matmul" | "chain") {
            let a = pre(&mut g, ids["A"]);
            let mut x = g.matmul(a, ids["B"]).unwrap();
            if self.kind == "chain" {
                let reduced = g.reduce_axis(ReduceOp::Sum, x, 1).unwrap();
                x = g.matmul(reduced, ids["C"]).unwrap();
            }
            match self.post {
                0 => x,
                1 => {
                    let b = g.binary(BinaryOp::Add, x, ids["bias"]).unwrap();
                    g.unary(UnaryOp::Relu, b).unwrap()
                }
                2 | 3 => {
                    let op = if self.post == 2 {
                        ReduceOp::Sum
                    } else {
                        ReduceOp::Max
                    };
                    let reduced = g.reduce_axis(op, x, self.axis as isize).unwrap();
                    g.binary(BinaryOp::Sub, x, reduced).unwrap()
                }
                _ => unreachable!(),
            }
        } else {
            let x = pre(&mut g, ids["X"]);
            if self.kind == "reduce" {
                match self.post {
                    0 => g.reduce_axis(ReduceOp::Sum, x, self.axis as isize).unwrap(),
                    1 => g.reduce_axis(ReduceOp::Max, x, self.axis as isize).unwrap(),
                    2 => {
                        let reduced = g.reduce_axis(ReduceOp::Max, x, self.axis as isize).unwrap();
                        g.binary(BinaryOp::Sub, x, reduced).unwrap()
                    }
                    _ => {
                        let relu = g.unary(UnaryOp::Relu, x).unwrap();
                        let a = g
                            .reduce_axis(ReduceOp::Sum, relu, self.axis as isize)
                            .unwrap();
                        let b = g.reduce_axis(ReduceOp::Sum, x, self.axis as isize).unwrap();
                        g.binary(BinaryOp::Add, a, b).unwrap()
                    }
                }
            } else {
                let b = ids["B"];
                match self.post {
                    0 => g.binary(BinaryOp::Add, x, b).unwrap(),
                    1 => {
                        let sub = g.binary(BinaryOp::Sub, x, b).unwrap();
                        let c = g.constant(-0.3);
                        g.binary(BinaryOp::Mul, sub, c).unwrap()
                    }
                    2 => {
                        let mul = g.binary(BinaryOp::Mul, x, b).unwrap();
                        g.binary(BinaryOp::Add, mul, x).unwrap()
                    }
                    _ => {
                        let square = g.binary(BinaryOp::Mul, b, b).unwrap();
                        let one = g.constant(1.0);
                        let denominator = g.binary(BinaryOp::Add, square, one).unwrap();
                        g.binary(BinaryOp::Div, x, denominator).unwrap()
                    }
                }
            }
        };
        (g, output, inputs)
    }
}

pub fn corpus() -> Vec<Case> {
    let cases: Vec<_> = CORPUS
        .lines()
        .filter(|s| !s.trim().is_empty() && !s.starts_with('#'))
        .map(|line| Case::parse(line).expect("valid checked-in corpus"))
        .collect();
    assert_eq!(cases.len(), 32, "N7 CI budget is 32 cases");
    cases
}
