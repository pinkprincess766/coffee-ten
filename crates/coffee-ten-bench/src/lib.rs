//! Benchmark harness: builds a kernel, checks every schedule variant against the naive one,
//! then times the variants in interleaved random order (project spec §6).

use std::fmt;

pub mod cli;
pub mod log;
pub mod meta;
pub mod run;
pub mod search;
pub mod space;
pub mod stats;

pub use cli::{Config, KernelKind, parse_args};
pub use run::{Record, run};

/// Why the harness stopped.
#[derive(Clone, Debug, PartialEq)]
pub enum BenchError {
    /// The command line is wrong.
    Usage(String),
    /// Lowering, scheduling, compiling or running a kernel failed.
    Setup(String),
    /// A variant's output differs from the naive output by more than the spec §5.4 bound.
    Mismatch {
        variant: String,
        index: usize,
        got: f32,
        naive: f32,
        bound: f64,
    },
}

impl fmt::Display for BenchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Usage(msg) | Self::Setup(msg) => f.write_str(msg),
            Self::Mismatch {
                variant,
                index,
                got,
                naive,
                bound,
            } => write!(
                f,
                "variant {variant:?} is wrong: out[{index}] = {got:e}, naive gives {naive:e}, \
                 allowed difference {bound:e}; nothing was timed"
            ),
        }
    }
}

impl std::error::Error for BenchError {}
