//! Builds the emitted C with clang into a shared library, caches it, loads it and calls it.

use crate::{CodegenError, emit_c};
use coffee_ten_loop::Kernel;
use libloading::Library;
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Flags of every build, besides `-shared -fPIC -o <out> <src>`.
/// No FMA contraction, so results match the f32 reference bit for bit; auto-vectorization
/// off, so only the schedule's `vectorize` step vectorizes (project spec §3.5).
pub const TEST_FLAGS: &[&str] = &[
    "-std=c99",
    "-O2",
    "-ffp-contract=off",
    "-fno-vectorize",
    "-fno-slp-vectorize",
];

/// Flags of benchmark builds. FMA contraction is allowed, as on the peak measurement
/// (spec §3.5), so results differ from the reference by rounding and reports must say so;
/// auto-vectorization stays off, so only the schedule's `vectorize` step vectorizes.
pub const BENCH_FLAGS: &[&str] = &[
    "-std=c99",
    "-O2",
    "-ffp-contract=fast",
    "-fno-vectorize",
    "-fno-slp-vectorize",
];

const CLANG: &str = "clang";
const ENTRY_SYMBOL: &[u8] = b"ct_entry\0";

/// `void ct_entry(float *const *bufs)`: the fixed-arity entry point of every library.
type EntryFn = unsafe extern "C" fn(*const *mut f32);

/// Why `compile` failed.
#[derive(Clone, Debug, PartialEq)]
pub enum CompileError {
    /// `emit_c` refused the kernel.
    Codegen(CodegenError),
    /// Writing the cache files failed.
    Io(String),
    /// `clang` could not be started because it does not exist.
    ClangNotFound(String),
    /// clang ran and exited with an error (`status` is `None` if killed by a signal).
    ClangFailed { status: Option<i32>, stderr: String },
    /// The shared library could not be loaded or has no entry point.
    Load(String),
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Codegen(e) => write!(f, "codegen failed: {e}"),
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::ClangNotFound(e) => write!(f, "clang not found: {e}"),
            Self::ClangFailed { status, stderr } => match status {
                Some(code) => write!(f, "clang exited with status {code}:\n{stderr}"),
                None => write!(f, "clang was killed by a signal:\n{stderr}"),
            },
            Self::Load(e) => write!(f, "cannot load library: {e}"),
        }
    }
}

impl std::error::Error for CompileError {}

/// Why `CompiledKernel::run` refused its arguments. The kernel was not called.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RunError {
    BufferCount {
        expected: usize,
        got: usize,
    },
    BufferLen {
        buffer: usize,
        expected: usize,
        got: usize,
    },
}

impl fmt::Display for RunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BufferCount { expected, got } => {
                write!(f, "expected {expected} buffers, got {got}")
            }
            Self::BufferLen {
                buffer,
                expected,
                got,
            } => write!(f, "buffer {buffer} has {got} elements, expected {expected}"),
        }
    }
}

impl std::error::Error for RunError {}

/// A kernel built by clang and loaded into this process.
#[derive(Debug)]
pub struct CompiledKernel {
    /// Keeps the code behind `entry` mapped; `entry` is only called through `&self`.
    _library: Library,
    entry: EntryFn,
    buffer_lens: Vec<usize>,
    library_path: PathBuf,
}

impl CompiledKernel {
    /// Calls the kernel. `buffers` in Kernel::buffers order; each slice must have exactly
    /// the buffer's element count. &mut slices cannot alias, which the C side's restrict needs.
    pub fn run(&self, buffers: &mut [&mut [f32]]) -> Result<(), RunError> {
        if buffers.len() != self.buffer_lens.len() {
            return Err(RunError::BufferCount {
                expected: self.buffer_lens.len(),
                got: buffers.len(),
            });
        }
        for (buffer, (slice, &expected)) in buffers.iter().zip(&self.buffer_lens).enumerate() {
            if slice.len() != expected {
                return Err(RunError::BufferLen {
                    buffer,
                    expected,
                    got: slice.len(),
                });
            }
        }
        let pointers: Vec<*mut f32> = buffers.iter_mut().map(|b| b.as_mut_ptr()).collect();
        // SAFETY: `entry` came from a library we built from our own C source, and the
        // library is alive because `self` owns it. The call gets one pointer per buffer
        // (count checked above), each to a slice with exactly the buffer's element count
        // (checked above). The pointers come from distinct `&mut` slices, so they do not
        // alias, as the C `restrict` parameters require. The C function indexes only
        // inside the buffers because `emit_c` ran `Kernel::validate`, which bounds every
        // access of every nest.
        unsafe { (self.entry)(pointers.as_ptr()) };
        Ok(())
    }

    pub fn library_path(&self) -> &Path {
        &self.library_path
    }
}

/// Builds `kernel` with `TEST_FLAGS` in `cache_dir`, or reuses the library built earlier
/// from the same C source, flags and clang version.
pub fn compile(kernel: &Kernel, cache_dir: &Path) -> Result<CompiledKernel, CompileError> {
    compile_with_flags(kernel, cache_dir, TEST_FLAGS)
}

/// Like `compile`, with the given clang `flags` (besides `-shared -fPIC -o <out> <src>`).
/// The flags are part of the cache key, so different flags give different library files.
pub fn compile_with_flags(
    kernel: &Kernel,
    cache_dir: &Path,
    flags: &[&str],
) -> Result<CompiledKernel, CompileError> {
    let mut source = emit_c(kernel).map_err(CompileError::Codegen)?;
    source.push('\n');
    source.push_str(&entry_wrapper(kernel));
    // emit_c checked that every element count fits in i64, so the products cannot overflow.
    let buffer_lens: Vec<usize> = kernel
        .buffers
        .iter()
        .map(|b| b.shape.iter().product())
        .collect();

    let clang_version = clang_stdout(&["--version"])?;
    let key = cache_key(&source, flags, &clang_version);
    let stem = format!("ct_{key:016x}");
    let source_path = cache_dir.join(format!("{stem}.c"));
    let library_path = cache_dir.join(format!("{stem}.{}", std::env::consts::DLL_EXTENSION));

    if !library_path.exists() {
        std::fs::create_dir_all(cache_dir).map_err(io_error)?;
        std::fs::write(&source_path, &source).map_err(io_error)?;
        // Build under a private name and rename, so a reader never sees a half-written file.
        let temp_path = cache_dir.join(format!(
            "{stem}.{}.tmp-{}",
            std::env::consts::DLL_EXTENSION,
            std::process::id()
        ));
        let built = run_clang(flags, &source_path, &temp_path);
        let renamed =
            built.and_then(|()| std::fs::rename(&temp_path, &library_path).map_err(io_error));
        if renamed.is_err() {
            let _ = std::fs::remove_file(&temp_path);
        }
        renamed?;
    }

    let (library, entry) = load(&library_path)?;
    Ok(CompiledKernel {
        _library: library,
        entry,
        buffer_lens,
        library_path,
    })
}

/// `void ct_entry(float *const *bufs) { name((void *)bufs[0], (void *)bufs[1], ...); }`
///
/// The kernel takes row pointers (`float (*Y)[d1]`) for rank >= 2 buffers; a `void *`
/// converts to any object pointer type in C99 without a warning, so one cast serves all ranks.
fn entry_wrapper(kernel: &Kernel) -> String {
    let args: Vec<String> = (0..kernel.buffers.len())
        .map(|id| format!("(void *)bufs[{id}]"))
        .collect();
    let unused = if args.is_empty() { " (void)bufs;" } else { "" };
    format!(
        "void ct_entry(float *const *bufs) {{{unused} {}({}); }}\n",
        kernel.name,
        args.join(", ")
    )
}

fn load(path: &Path) -> Result<(Library, EntryFn), CompileError> {
    // SAFETY: loading runs the library's initializers and trusts its symbol types. We only
    // load a file produced by our own clang call from our own C source, which has no
    // initializers; the file name contains the hash of that source, flags and clang
    // version, and a file is moved to that name only after clang succeeded. This assumes
    // nothing but `compile` writes into `cache_dir`: on a cache hit the file is not
    // rebuilt, so a foreign file under our name would be loaded. The symbol
    // `ct_entry` is generated by `entry_wrapper` with exactly the type `EntryFn`. The
    // fn pointer is copied out and stored next to the library in `CompiledKernel`, which
    // calls it only while the library is alive.
    unsafe {
        let library = Library::new(path).map_err(|e| CompileError::Load(e.to_string()))?;
        let entry = *library
            .get::<EntryFn>(ENTRY_SYMBOL)
            .map_err(|e| CompileError::Load(e.to_string()))?;
        Ok((library, entry))
    }
}

fn io_error(e: std::io::Error) -> CompileError {
    CompileError::Io(e.to_string())
}

/// Runs `clang args` and returns its stdout.
fn clang_stdout(args: &[&str]) -> Result<String, CompileError> {
    let output = Command::new(CLANG)
        .args(args)
        .output()
        .map_err(clang_spawn_error)?;
    if !output.status.success() {
        return Err(clang_failed(&output));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

fn run_clang(flags: &[&str], source: &Path, out: &Path) -> Result<(), CompileError> {
    let output = Command::new(CLANG)
        .args(flags)
        .args(["-shared", "-fPIC", "-o"])
        .arg(out)
        .arg(source)
        // glibc keeps `expf` in libm, not libc, so a shared library using it must link
        // it; after the source, since linkers resolve left to right. Harmless on macOS.
        .arg("-lm")
        .output()
        .map_err(clang_spawn_error)?;
    if output.status.success() {
        Ok(())
    } else {
        Err(clang_failed(&output))
    }
}

fn clang_spawn_error(e: std::io::Error) -> CompileError {
    if e.kind() == std::io::ErrorKind::NotFound {
        CompileError::ClangNotFound(e.to_string())
    } else {
        CompileError::Io(e.to_string())
    }
}

fn clang_failed(output: &std::process::Output) -> CompileError {
    CompileError::ClangFailed {
        status: output.status.code(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    }
}

/// 64-bit FNV-1a over the source, every flag (each followed by a 0 byte) and the clang version.
fn cache_key(source: &str, flags: &[&str], clang_version: &str) -> u64 {
    let mut hash = FNV_OFFSET;
    hash = fnv1a(hash, source.as_bytes());
    for flag in flags {
        hash = fnv1a(hash, flag.as_bytes());
        hash = fnv1a(hash, &[0]);
    }
    fnv1a(hash, clang_version.as_bytes())
}

const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

fn fnv1a(mut hash: u64, bytes: &[u8]) -> u64 {
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use coffee_ten_loop::{
        Access, AffineExpr, BinaryOp, Buffer, Expr, Loop, LoopNest, Schedule, Store, UnaryOp, apply,
    };

    fn lp(name: &str, extent: usize) -> Loop {
        Loop {
            name: name.to_string(),
            extent,
            vectorize: false,
            unroll: false,
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

    fn load(buffer: usize, index: Vec<AffineExpr>) -> Expr {
        Expr::Load(Access { buffer, index })
    }

    fn bin(op: BinaryOp, a: Expr, b: Expr) -> Expr {
        Expr::Binary(op, Box::new(a), Box::new(b))
    }

    fn nest(loops: Vec<Loop>, buffer: usize, index: Vec<AffineExpr>, value: Expr) -> LoopNest {
        LoopNest {
            loops,
            guards: Vec::new(),
            body: Store {
                target: Access { buffer, index },
                value,
            },
        }
    }

    fn matmul() -> Kernel {
        Kernel {
            name: "matmul".to_string(),
            buffers: vec![buf("A", &[4, 3]), buf("B", &[3, 2]), buf("C", &[4, 2])],
            nests: vec![
                nest(
                    vec![lp("i", 4), lp("j", 2)],
                    2,
                    vec![var("i"), var("j")],
                    Expr::Const(0.0),
                ),
                nest(
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
        Kernel {
            name: "bias_relu".to_string(),
            buffers: vec![buf("A", &[2, 3]), buf("bias", &[3]), buf("Y", &[2, 3])],
            nests: vec![nest(
                vec![lp("i", 2), lp("j", 3)],
                2,
                vec![var("i"), var("j")],
                Expr::Unary(
                    UnaryOp::Relu,
                    Box::new(bin(
                        BinaryOp::Add,
                        load(0, vec![var("i"), var("j")]),
                        load(1, vec![var("j")]),
                    )),
                ),
            )],
        }
    }

    /// A fresh cache dir, removed when dropped (also if the test panics).
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(name: &str) -> Self {
            let dir = std::env::temp_dir()
                .join(format!("coffee-ten-compile-{}-{name}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).expect("create cache dir");
            Self(dir)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn matmul_inputs() -> (Vec<f32>, Vec<f32>) {
        let a = (0..12)
            .map(|n| {
                let (i, k) = (n / 3, n % 3);
                (i * 3 + k) as f32 * 0.1 + 1.0 / 3.0
            })
            .collect();
        let b = (0..6)
            .map(|n| {
                let (k, j) = (n / 2, n % 2);
                (k * 2 + j) as f32 * 0.7 - 0.25
            })
            .collect();
        (a, b)
    }

    #[test]
    fn matmul_matches_f32_reference_bit_for_bit() {
        let dir = TempDir::new("matmul");
        let compiled = compile(&matmul(), &dir.0).unwrap();
        let (mut a, mut b) = matmul_inputs();
        // Garbage in the output: the init nest must overwrite it.
        let mut c = vec![f32::NAN; 8];
        compiled.run(&mut [&mut a, &mut b, &mut c]).unwrap();
        for i in 0..4 {
            for j in 0..2 {
                let mut expected = 0.0f32;
                for k in 0..3 {
                    expected += a[i * 3 + k] * b[k * 2 + j];
                }
                let got = c[i * 2 + j];
                assert_eq!(
                    got.to_bits(),
                    expected.to_bits(),
                    "C[{i}][{j}]: got {got:?}, expected {expected:?}"
                );
            }
        }
    }

    #[test]
    fn split_matmul_with_remainder_matches_unsplit_bit_for_bit() {
        // K = 3 split by 2: the guard must skip k = 3, whose A and B reads are out of range.
        let split = apply(
            &matmul(),
            &Schedule::parse("split(1, k, ko, ki, 2)").unwrap(),
        )
        .unwrap();
        let dir = TempDir::new("split");
        let (a, b) = matmul_inputs();
        // Nonzero values behind A and B, so that an out-of-range read changes C.
        let mut a_storage = [a.as_slice(), &[1e3; 8]].concat();
        let mut b_storage = [b.as_slice(), &[1e3; 8]].concat();
        let mut outputs = Vec::new();
        for kernel in [matmul(), split] {
            let compiled = compile(&kernel, &dir.0).unwrap();
            let mut c = vec![f32::NAN; 8];
            let (a, b) = (&mut a_storage[..12], &mut b_storage[..6]);
            compiled.run(&mut [a, b, &mut c]).unwrap();
            outputs.push(c);
        }
        for (n, (split_c, plain_c)) in outputs[1].iter().zip(&outputs[0]).enumerate() {
            assert_eq!(
                split_c.to_bits(),
                plain_c.to_bits(),
                "C[{n}]: split {split_c:?}, unsplit {plain_c:?}"
            );
        }
    }

    #[test]
    fn relu_of_negative_zero_is_positive_zero() {
        let dir = TempDir::new("relu");
        let compiled = compile(&bias_relu(), &dir.0).unwrap();
        let mut a = vec![-0.0f32, -0.0, -1.5, -1e-45, 2.5, -0.5];
        let mut bias = vec![0.0f32, -0.0, 0.5];
        let mut y = vec![f32::NAN; 6];
        compiled.run(&mut [&mut a, &mut bias, &mut y]).unwrap();
        for i in 0..2 {
            for j in 0..3 {
                let x = a[i * 3 + j] + bias[j];
                let expected = if x > 0.0 { x } else { 0.0 };
                let got = y[i * 3 + j];
                assert_eq!(
                    got.to_bits(),
                    expected.to_bits(),
                    "Y[{i}][{j}]: got {got:?}, expected {expected:?}"
                );
            }
        }
        // A[0][1] + bias[1] is -0.0 + -0.0 = -0.0, which relu must turn into +0.0.
        assert_eq!(y[1].to_bits(), 0.0f32.to_bits());
    }

    #[test]
    fn second_compile_reuses_the_cached_library() {
        let dir = TempDir::new("cache");
        let first = compile(&matmul(), &dir.0).unwrap();
        let modified = std::fs::metadata(first.library_path())
            .and_then(|m| m.modified())
            .unwrap();
        let second = compile(&matmul(), &dir.0).unwrap();
        let modified_again = std::fs::metadata(second.library_path())
            .and_then(|m| m.modified())
            .unwrap();
        assert_eq!(first.library_path(), second.library_path());
        assert_eq!(modified, modified_again);
    }

    #[test]
    fn threads_compiling_the_same_kernel_at_once_all_succeed() {
        // `cargo test` runs tests as threads of one process, and two tests may build the
        // same kernel into the same cache at the same moment. Each round starts from an
        // empty cache, so every thread misses it and builds.
        const THREADS: usize = 8;
        for round in 0..5 {
            let dir = TempDir::new(&format!("race-{round}"));
            let barrier = std::sync::Barrier::new(THREADS);
            let results: Vec<Result<PathBuf, CompileError>> = std::thread::scope(|scope| {
                let handles: Vec<_> = (0..THREADS)
                    .map(|_| {
                        scope.spawn(|| {
                            barrier.wait();
                            compile(&matmul(), &dir.0).map(|k| k.library_path().to_path_buf())
                        })
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });
            for result in results {
                assert!(result.is_ok(), "round {round}: {result:?}");
            }
        }
    }

    #[test]
    fn different_flags_get_different_libraries() {
        let dir = TempDir::new("flags");
        let test = compile_with_flags(&matmul(), &dir.0, TEST_FLAGS).unwrap();
        let bench = compile_with_flags(&matmul(), &dir.0, BENCH_FLAGS).unwrap();
        assert_ne!(test.library_path(), bench.library_path());
        assert!(test.library_path().exists() && bench.library_path().exists());
    }

    #[test]
    fn different_kernels_get_different_libraries() {
        let dir = TempDir::new("keys");
        let first = compile(&matmul(), &dir.0).unwrap();
        let second = compile(&bias_relu(), &dir.0).unwrap();
        assert_ne!(first.library_path(), second.library_path());
    }

    #[test]
    fn run_rejects_wrong_buffer_count() {
        let dir = TempDir::new("count");
        let compiled = compile(&matmul(), &dir.0).unwrap();
        let (mut a, mut b) = matmul_inputs();
        assert_eq!(
            compiled.run(&mut [&mut a, &mut b]),
            Err(RunError::BufferCount {
                expected: 3,
                got: 2
            })
        );
    }

    #[test]
    fn run_rejects_a_short_buffer() {
        let dir = TempDir::new("len");
        let compiled = compile(&matmul(), &dir.0).unwrap();
        let (mut a, mut b) = matmul_inputs();
        let mut c = vec![7.0f32; 7];
        assert_eq!(
            compiled.run(&mut [&mut a, &mut b, &mut c]),
            Err(RunError::BufferLen {
                buffer: 2,
                expected: 8,
                got: 7
            })
        );
        // The C code did not run: the init nest would have overwritten 7.0.
        assert!(c.iter().all(|&x| x == 7.0));
    }

    #[test]
    fn clang_failure_carries_stderr() {
        let dir = TempDir::new("fail");
        let flags = ["-std=c99", "-Werror", "-Wthis-flag-does-not-exist"];
        match compile_with_flags(&matmul(), &dir.0, &flags) {
            Err(CompileError::ClangFailed { status, stderr }) => {
                assert!(status.is_some_and(|code| code != 0));
                assert!(!stderr.is_empty());
            }
            other => panic!("expected ClangFailed, got {other:?}"),
        }
        // No half-built library is left behind.
        let leftovers: Vec<_> = std::fs::read_dir(&dir.0)
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .filter(|name| !name.to_string_lossy().ends_with(".c"))
            .collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
    }

    #[test]
    fn codegen_errors_are_passed_through() {
        let dir = TempDir::new("codegen");
        let mut kernel = matmul();
        kernel.name = "ct_entry".to_string();
        assert_eq!(
            compile(&kernel, &dir.0).unwrap_err(),
            CompileError::Codegen(CodegenError::NameClash("ct_entry".to_string()))
        );
    }

    #[test]
    fn kernel_without_buffers_compiles_and_runs() {
        let dir = TempDir::new("empty");
        let kernel = Kernel {
            name: "nothing".to_string(),
            buffers: vec![],
            nests: vec![],
        };
        // The wrapper must pass clang -Wall -Wextra -Werror, even with an unused `bufs`.
        let compiled = compile_with_flags(
            &kernel,
            &dir.0,
            &["-std=c99", "-Wall", "-Wextra", "-Werror"],
        )
        .unwrap();
        compiled.run(&mut []).unwrap();
    }

    #[test]
    fn matmul_with_wrapper_passes_strict_warnings() {
        let dir = TempDir::new("strict");
        let flags = ["-std=c99", "-Wall", "-Wextra", "-Werror"];
        assert!(compile_with_flags(&matmul(), &dir.0, &flags).is_ok());
    }

    #[test]
    fn row_pointer_wrapper_is_clean_under_test_flags_and_werror() {
        // matmul has rank 2 buffers: `float (*)[3]` parameters get `void *` arguments.
        let dir = TempDir::new("strict-test-flags");
        let flags: Vec<&str> = TEST_FLAGS
            .iter()
            .copied()
            .chain(["-Wall", "-Wextra", "-Werror"])
            .collect();
        let compiled = compile_with_flags(&matmul(), &dir.0, &flags).unwrap();
        let (mut a, mut b) = matmul_inputs();
        let mut c = vec![f32::NAN; 8];
        compiled.run(&mut [&mut a, &mut b, &mut c]).unwrap();
        assert!(c.iter().all(|x| x.is_finite()), "{c:?}");
    }

    #[test]
    fn fnv1a_matches_known_vectors() {
        // Published FNV-1a 64-bit test vectors.
        assert_eq!(fnv1a(FNV_OFFSET, b""), 0xcbf29ce484222325);
        assert_eq!(fnv1a(FNV_OFFSET, b"a"), 0xaf63dc4c8601ec8c);
        assert_eq!(fnv1a(FNV_OFFSET, b"foobar"), 0x85944171f73967e8);
    }
}
