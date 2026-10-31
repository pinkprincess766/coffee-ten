# Parallel codegen: design

Plan item C23 (`docs/PLAN-20PR.md`, section 20, draft). This is a design note for review.
There is no code in this change. Nothing here is implemented, and every number is either
cited from [BENCHMARKS.md](../BENCHMARKS.md) or marked as measured here or as an estimate.

## 1. Summary

- The one place to parallelize is the outermost loop of a nest that writes each output
  element from one iteration of that loop (matmul: `i` or `io`). Reduction loops (`k`,
  `ko`) stay serial.
- That keeps the order of the sum for every element of `C`. A parallel schedule is
  `preserving` and equals the serial kernel bit for bit.
- Three ways to run it: OpenMP, pthreads inside the generated C, or threads in Rust
  that call a generated `ct_entry_range`. OpenMP does not work on this Mac without an
  extra runtime. The Rust-side option needs no new crate and no thread code in C.
- Recommendation: do option (c), first as a run option (no new schedule step, no IR
  change), measure it on matmul, and add `parallel(nest, loop)` only if the numbers justify
  it ([section 6](#6-recommendation)).
- Threads do not pay for small kernels. With `std::thread::scope`, starting three
  threads costs about 40 us on this Mac, more than the whole register-tile kernel at
  N = 128 ([section 4.5](#45-thread-start-against-kernel-time)).

## 2. Scope

In scope: a design of multi-thread execution of one compiled kernel on one machine, for
the loop IR as it is today (perfect nests, one store, affine indices).

Out of scope: a cost model for several cores, split-K and other reductions across
threads, GPU, Windows, NUMA, and any change to hypothesis A.

The README lists "threads and GPU in the MVP" among the non-goals and "Single thread: the
generated C has no threads" among the limitations. Both lines would change if this is built.

## 3. What would be parallel

### 3.1 The step

```text
parallel(nest, loop)
```

It marks one loop of one nest. At run time the iterations of that loop are shared by
several threads. Every other loop of the nest runs unchanged inside each iteration.

Rules for the step, in the style of `vectorize` and `reorder` in
`crates/coffee-ten-loop/src/apply.rs`. The loop `L` is legal when

1. `L` is the outermost loop of the nest. Inner parallel loops would need a fork per
   iteration of the loops outside.
2. `L` is not a reduction loop: its name appears in the store target index with a
   non-zero coefficient (the test `reorder` and `vectorize` already use).
3. The target is injective on the loops (`is_injective`), so different iterations of
   `L` write different elements.
4. The body reads no other element of the target. With a reduction loop in the nest it
   must be `T[I] + e` with no load of `T` in `e` (`is_accumulation`); without one, no
   load of `T` at another index (`is_other_element`).

Why that is enough. An iteration of `L` writes `T[I]`, and `I` determines the value of `L`
(rule 3). It reads at most `T[I]` itself (rule 4), and all other buffers of the nest are
read-only, because a nest has one store. So two iterations of `L` never touch the same
element of `T` and no iteration waits for another. These are the checks `reorder` makes
today; the only new ones are "outermost" and "not a reduction loop".

Rules 3 and 4 are a data-race condition, not a performance hint. `vectorize` leaves
loop-carried reads to clang. A race in a parallel loop is undefined behaviour, so
`apply` must refuse an illegal `parallel` itself.

Steps after `parallel`:

- `reorder` that moves the marked loop off the outermost place is refused.
- `split` of the marked loop is refused. (`unroll` instead clears its mark on `split`;
  refusing is less surprising. Open question 4.)
- `vectorize` and `unroll` on other loops are not affected.

### 3.2 Matmul

`matmul_nests` in `crates/coffee-ten-loop/src/lower.rs` makes two nests:

| Nest | Loops | `parallel` on | Why |
|---|---|---|---|
| 0, init `C[i][j] = 0` | `i, j` | `i` | no reduction; `j` is legal but not outermost |
| 1, update `C[i][j] += A[i][k] * B[k][j]` | `i, k, j` | `i` | each `i` owns row `i` of `C` |
| | | not `k` | reduction: every `k` writes the same `C[i][j]` |
| 1 after `split(1, i, io, ii, 4)`, `split(1, j, jo, ji, 16)`, `reorder(1, io, jo, k, ii, ji)` | `io, jo, k, ii, ji` | `io` | `4 * io + ii` is injective |
| | | `jo` after a `reorder` that puts it first | legal; see 3.4 for why `io` is better |
| | | not `ko` (after a split of `k`) | reduction; split-K is not designed here |

`relu(A @ B + bias)` adds nest 2, the elementwise epilogue. All its loops are parallel
by the same rules.

### 3.3 Order of nests

Nests run in order, and nest 1 reads what nest 0 wrote. Two designs:

- A barrier after every nest. Simple and general: nest 0 runs, all threads finish, nest 1
  starts. This is what OpenMP gives for free (the implicit barrier of `parallel for`).
- Fusing nest 0 and nest 1 per chunk of rows, so no barrier. It needs the compiler to
  know that nest 0's `i` and nest 1's `4 * io + ii` are the same axis. That is analysis
  the IR does not do today.

Take the barrier. The init and epilogue nests are O(N^2) work against O(N^3) for the
update (estimate from the loop bounds, not measured), so run them on the calling thread
and fork only for nests that have a reduction loop. Revisit if a profile says otherwise.

### 3.4 Guards and chunk size

A `split` with a factor that does not divide the extent adds a guard, for example
`4 * io + ii < n`, and the outer extent is `ceil(n / factor)`. A guard is a pure test
inside the iteration, so it is safe under any chunking. The last iteration may do less
real work than the others: at most one iteration of imbalance.

The schedule space of hypothesis A uses only divisors, so it has no guards
(`crates/coffee-ten-bench/src/space.rs`).

The parallel loop needs enough iterations. For the register tile (`regtile`,
[Register tile](../BENCHMARKS.md#register-tile-s3)) `io` has N / 4 iterations (16 at N = 64,
256 at N = 1024). For `tiled` (`split(1, i, io, ii, 32)`) it has N / 32: 2 at N = 64, so
at most two threads can work.

Rows are the better axis. A chunk of whole rows of `C` is contiguous memory. A chunk of
columns (`jo`) puts every thread on a slice of every row, and neighbours share cache lines
at each edge. The model assumes 128-byte lines (`Machine::assumed_m5`, assumed, not
measured). `Vec<f32>` only guarantees 4-byte alignment, so even row chunks can share one
line at a boundary; that is one line per boundary, which is small.

## 4. Options

All three run the same kernel body. They differ in who starts the threads and who waits.

### 4.1 (a) OpenMP

Codegen: one line before the parallel loop.

```c
#pragma omp parallel for schedule(static)
for (long io = 0; io < 16; io++) { ... }
```

The barrier between nests comes with it. `nest_c` in
`crates/coffee-ten-codegen-c/src/lib.rs` already prints `#pragma clang loop` hints, so
the change to the emitter is small.

Build: `-fopenmp` plus a runtime (libomp or libgomp), through a third flag set next to
`TEST_FLAGS` and `BENCH_FLAGS`.

Result of `clang -fopenmp` on this Mac (Apple M5, Apple clang 21.0.0, nothing installed
for this check; the file is a four-line `#pragma omp parallel for` with `#include <omp.h>`):

```text
$ clang -fopenmp t.c -o t
clang: error: unsupported option '-fopenmp'
clang: error: unsupported option '-fopenmp'
```

The same error with the pragma and no `#include`. So the flag is refused by the driver.
The compiler front end itself does support OpenMP: with `-Xclang -fopenmp` it compiles
the region and fails only at link time, with `___kmpc_fork_call`,
`___kmpc_for_static_init_4` and `___kmpc_for_static_fini` undefined, because Apple ships
no OpenMP runtime. There is no `omp.h` in the SDK or in the clang resource directory.

This Mac does have Homebrew `libomp` (`/opt/homebrew/opt/libomp`, version 22.1.8, installed
earlier by the user, not by this check). With it, this command builds and runs and
prints `0 1 2 3`:

```text
clang -Xclang -fopenmp -I/opt/homebrew/opt/libomp/include t.c \
  -L/opt/homebrew/opt/libomp/lib -lomp
```

Portability and risks:

- It is a new dependency on macOS: Homebrew `libomp`, found by a path that depends on
  the machine. The plan says no new dependencies.
- The cache key (`cache_key` in `compile.rs`) covers the source, the flags and the clang
  version, not the runtime library. A changed `libomp` would not rebuild a cached kernel.
- The generated library would link `libomp.dylib`. Loaded into a Python process that
  already has another OpenMP runtime (some wheels bundle their own), two runtimes in one
  process is a known source of aborts. Not tested here.
- Linux: clang needs `omp.h` and a runtime, usually the `libomp-dev` package, or libgomp.
  The CI workflow installs no clang and uses the one on the runner image. I did not check
  whether that image has an OpenMP runtime for clang.
- Thread count and affinity come from environment variables (`OMP_NUM_THREADS`), not from
  the harness. Results depend on the runtime's wait policy, which is another thing to pin
  down for a measurement.
- Tests of a parallel kernel would need the real runtime on every OS in CI.

### 4.2 (b) pthreads in the generated C

Codegen: the emitter prints, per marked nest, a worker function over `[lo, hi)`, and
`ct_entry` creates the threads and joins them before the next nest.

```c
struct ct_job { /* buffer pointers, lo, hi */ };
static void *ct_worker1(void *arg) { /* the nest, io from lo to hi */ }
/* in the kernel: split [0, 16) into T chunks, pthread_create each, pthread_join each */
```

Build: POSIX threads only. Linux may need `-pthread`; with `-std=c99` the POSIX
declarations may need `_POSIX_C_SOURCE` (the same reason `bench/peak/peak.c` defines it on
Linux for `clock_gettime`). No library to install on either OS.

Costs and risks:

- Threads are created and joined on every call and for every marked nest. The cost per
  call is in 4.5.
- The thread count has to come from somewhere: an environment variable, a global set by
  an exported setter, or an extra argument to `ct_entry`. The last one changes
  `EntryFn`.
- Error paths (`pthread_create` fails) live in generated C, where they are hard to test.
  A fallback is to run the chunk on the caller.
- More text in every generated library, and thread code that clang must compile with the
  same flags as the kernel.
- Code that clang sees has calls to `pthread_create` around the loop. The kernel body
  is in a separate function, so it should compile as before; checking the optimized IR
  (as `crates/coffee-ten-codegen-c/tests/unroll.rs` does for the register tile) would
  confirm it.
- Python and Rust cannot choose the thread count or the cores per call without that
  extra argument.

### 4.3 (c) Threads in Rust, `ct_entry_range` in C

Codegen: the emitter keeps `ct_entry` as it is, and for a kernel with a marked nest also
exports

```c
void ct_entry_range(float *const *bufs, long nest, long lo, long hi);
```

It runs nest `nest` with its outermost loop restricted to `[lo, hi)`. In C the
change is the loop bounds: `for (long io = lo; io < hi; io++)`. Nothing else about the
nest changes. A kernel with no marked nest gets the same C text as today, so existing
cache keys stay valid.

Rust: `coffee-ten-codegen-c/src/compile.rs` loads `ct_entry_range` next to
`ct_entry` (a second symbol, absent for serial kernels) and adds

```text
CompiledKernel::run_parallel(&self, buffers: &mut [&mut [f32]], threads: usize)
```

It checks the arguments as `run` does. Then for each nest: an unmarked nest is one call
`range(nest, 0, extent)` on the calling thread; a marked nest is a `std::thread::scope`
in which `threads` workers take chunks of the outer loop from an `AtomicUsize` counter
and call `ct_entry_range`. The scope ends, which is the barrier, and the next nest starts.
This is the pattern `parallel_map` in `crates/coffee-ten-bench/src/run.rs` already uses
for compilation (`Reference::build_all`), so there is no new crate and no new idea.

Why a counter and not equal slices: the M5 has two kinds of cores (section 5.3). Equal
slices make the slowest core the end of every nest. Small chunks taken on demand let the
fast cores do more. The chunk size is a parameter to tune.

Unsafe and what holds it up:

- `*mut f32` is not `Send`. The pointer table is wrapped in a small type with
  `unsafe impl Send + Sync`. The safety argument is rule 3 and 4 of section 3.1: workers
  write disjoint elements and read nothing another worker writes. `run_parallel` keeps the
  `&mut` slices borrowed for the whole call, so nothing outside aliases the buffers.
- The `restrict` on the output parameters of the emitted function is per call. Two
  concurrent calls that write disjoint elements of the same array do not break it.
- If the legality check has a hole, the result is a data race. The tests must be able to
  see that (section 6.2).

Other costs:

- A kernel run from outside Rust (calling `ct_entry` from another program) has no
  threads. Nothing does that today; the Python binding calls `CompiledKernel::run`, and
  releases the GIL around it (`py.detach` in `crates/coffee-ten-py/src/lib.rs`), which
  would stay true for `run_parallel`.
- Spawning per call has the cost in 4.5. A parked pool (workers waiting on a `Condvar`)
  removes most of it but needs global state, shutdown, and erasing the lifetime of the
  borrowed buffers, which is more `unsafe`. Start with `thread::scope`; move to a pool
  only if the numbers in 4.5 are what holds back N <= 256.

### 4.4 Comparison

| | (a) OpenMP | (b) pthreads in C | (c) Rust threads + `ct_entry_range` |
|---|---|---|---|
| New dependency | libomp on macOS, probably on Linux | none | none |
| Works on this Mac out of the box | no | yes | yes |
| Thread code lives in | libomp | generated C | Rust, tested by `cargo test` |
| Thread count chosen by | environment variable | env var or new C argument | argument of `run_parallel` |
| Barrier between nests | implicit | `pthread_join` | end of `thread::scope` |
| Dynamic chunking | `schedule(dynamic)` | write it in C | atomic counter, as `parallel_map` |
| Per-call start cost | runtime keeps workers | create and join | create and join; a pool is possible |
| Changes `EntryFn` / ABI | no | maybe | adds a second symbol |
| Cache key and flags | runtime missing from key | none | none |
| Main risk | environment | untestable error paths in C | `unsafe` disjointness argument |

### 4.5 Thread start against kernel time

Kernel time here is derived: `2 N^3 / rate`, with the S3 `regtile` matmul rates from
[Register tile](../BENCHMARKS.md#register-tile-s3) (121.84, 122.77, 123.72, 121.81 and
105.30 GFLOP/s for N = 64 ... 1024).

Start cost was measured here, on this Mac (M5, 4 + 6 cores, section 5.3), by a scratch
program that is not in the repo: threads that do nothing. Two runs gave, for 3 extra
threads, a median of 39.5 us and 37.8 us for `std::thread::scope` spawn and join, and 13.3
us and 11.5 us for waking 3 parked threads and waiting for them (a `Condvar` pool). A C
`pthread_create` plus `pthread_join` of 3 threads gave 36 us and 33 us. For 9 extra
threads: 87 and 93 us (scope), 26 and 29 us (pool). These are medians of 2000
repetitions in a tight loop on a warm Mac, with no QoS call. Other builds were running on
the Mac at the time, so contention may have inflated them. Treat them as an order of
magnitude for planning, not a value to quote as a result.

Estimate for 4 threads, assuming an ideal 4-way split of the kernel and one overhead on
top (39 us for scope, 12 us for pool, rounded from the medians above), no bandwidth limit, four equal cores:

| N | serial `regtile` time (derived) | scope: time and speedup | pool: time and speedup |
|---|---|---|---|
| 64 | 4.3 us | 40 us, x0.11 | 13 us, x0.33 |
| 128 | 34 us | 48 us, x0.72 | 21 us, x1.66 |
| 256 | 271 us | 107 us, x2.5 | 80 us, x3.4 |
| 512 | 2.2 ms | 0.59 ms, x3.7 | 0.56 ms, x3.9 |
| 1024 | 20.4 ms | 5.1 ms, x4.0 | 5.1 ms, x4.0 |

Reading: with scope, threads lose at N <= 128 and start to pay at N = 256; with a pool,
N = 128 can gain; N = 64 never gains, at any cost model (the kernel is 4.3 us). The
naive kernel (about 7.5 GFLOP/s, S2 of BENCHMARKS.md) runs 4.6 ms at N = 256 by the
same arithmetic, so overhead is noise there; it is the cleanest first check that
the threads do what they should.

## 5. Interaction with the project

### 5.1 Results are bit for bit

Parallelism over `i`, `io` or `j` gives every element of `C` to one thread, which runs the
`k` loop in the same order as before. By the definitions in `OrderClass`
(`crates/coffee-ten-loop/src/apply.rs`) the step is `Preserving`. The result equals the
serial kernel and `ct.run` bit for bit, under `TEST_FLAGS` (`-ffp-contract=off`).

Under `BENCH_FLAGS` (`-ffp-contract=fast`) kernels already differ from the reference by
rounding (see the comment on `BENCH_FLAGS`). The parallel and the serial build should
contract the same expressions, since the loop body is the same; this is an expectation to
test, not a guarantee. The harness check (`check_outputs`, bound from spec §5.4) applies
unchanged.

A parallel reduction would not be bit for bit. Parallel `k` needs one partial sum per
thread and a combine step: a different sum order, so `Reordering`, a new buffer, and an
error bound that depends on the thread count. Not designed here.

One code-generation risk: the outer loop bound becomes a run-time value (`lo`, `hi`).
The inner loops keep constant extents, so the register tile should stay in registers, but
that is checked by reading optimized IR, not assumed (`tests/unroll.rs` counts the float
phis of the register tile in the optimized IR).

### 5.2 Cost model

The model in `crates/coffee-ten-bench/src/model.rs` is a one-core model. Every machine
field says so: `peak_gflops` "on one core", `mem_gbs` "seen by one core", `l2_gbs` "seen by
one core", and the calibration points come from the single-thread `bench/peak`. For several
cores it would need at least:

- a `cores` input, and which cores (4 fast + 6 efficiency on this Mac);
- the shared resources: memory bandwidth and L2 do not scale with cores, so a bandwidth
  ceiling with several cores at once is a separate measurement;
- a parallel efficiency term for start cost, chunk imbalance and the serial nests;
- the working set per thread (each thread streams its own slice of `A` and `C`, and all of
  `B`).

This is out of scope. Until it exists the model cannot rank parallel schedules, and the
parallel runs should not be fed to `model compare`.

### 5.3 Harness, peak and noise

- **Timing.** `time_sample` in `crates/coffee-ten-bench/src/run.rs` runs the kernel for at
  least `--min-sample-ms` (default 10 ms, `DEFAULT_MIN_SAMPLE_MS`), 30 rounds
  (`DEFAULT_SAMPLES`), variants interleaved. That works for a parallel kernel as it is,
  and `run_parallel` would be one more call in `Variant::run_once`.
- **Warm-up.** `WARMUP_MS` is 300 ms because a thread runs at 25-50% for about 100 ms after
  it was idle ([Validity](../BENCHMARKS.md#validity)). Worker threads are idle while the
  other variants of the round run. A parked pool would be cold at each sample; spawned
  threads start cold at each call. Whether the workers ramp up fast enough is not known
  and must be measured before a parallel number is trusted.
- **Peak.** `bench/peak/peak.c` measures one core: the FMA peak was 129.2 GFLOP/s in S3
  ([Ceilings](../BENCHMARKS.md#ceilings-benchpeak-one-core)). "Percent of peak" for a
  parallel kernel needs a multi-core peak: the same loop on T threads, and T-thread
  memory triads. That is new work in `peak.c`, and the number cannot be guessed from the
  one-core number because the cores differ.
- **Core types.** This Mac reports (`sysctl`, run for this note): `hw.ncpu` 10;
  `hw.perflevel0.name` Super, 4 physical cores; `hw.perflevel1.name` Efficiency, 6
  physical cores. The project's single-thread numbers come from a fast core. A kernel that
  uses all 10 threads has a mix, and the speed of an efficiency core against a fast core on
  this kernel has not been measured. `std::thread::available_parallelism()`, which
  `Reference::build_all` uses, probably returns 10 here (not run). A first
  version should take the thread count as an argument, not detect it.
- **QoS.** `raise_qos` in `crates/coffee-ten-bench/src/main.rs` sets
  `QOS_CLASS_USER_INTERACTIVE` on the main thread to raise the chance of a fast core. It
  does not guarantee one. Whether threads started later inherit that class is not checked
  here; if they do not, each worker must call it. There is no core pinning on macOS that I
  know of, so QoS is the only lever, and the placement of workers stays a source of noise.
- **Heat.** The machine is fanless (Conditions in BENCHMARKS.md), and the Validity section
  already blames a hot chassis or a move to an efficiency core for one series that ran at
  half speed. A multi-core run heats the machine faster. Expect more drift flags, and plan
  longer pauses between series.
- **Spread.** A fork-join time is the slowest worker's time. One preempted or slow thread
  stretches the whole call, so the spread (p25-p75) will probably be wider than for one
  thread. Estimate, not measured.
- **The `build_all` check** runs each variant once on all cores in parallel to compare
  with the naive output. A parallel variant would start threads from inside that pool.
  That is correct but oversubscribed; it is not timed, so it only costs time.

### 5.4 Hypothesis A

Hypothesis A is single-thread by definition: one core, N = 256, a 7334-schedule space,
and a model whose inputs are one-core numbers (README, Abstract and section 2.4). Parallel
kernels do not belong to that space. They must not enter `space.rs` or the random
search, or the recorded runs 1 to 4 become incomparable with new ones.

A parallel variant would be a new hypothesis with its own criterion fixed before the run,
for example "a model with a `cores` input ranks parallel schedules as well as hypothesis A
ranks single-thread ones". It needs the model of section 5.2 first, so it is not
the next step.

### 5.5 Other places

- **Compile cache.** (c) leaves the key alone. (a) would need the runtime in the key.
- **CI.** Tests must pass when the thread count is above the number of cores, and on a
  runner with few cores. I did not check the core count of the runners.
- **JSONL schema.** A parallel run needs a `threads` field in the result records
  (`bench/README.md`, Record fields). That belongs with the schema change of plan item C16.
- **README.** The "single thread" limitation and the "threads in the MVP" non-goal change.

## 6. Recommendation

### 6.1 Minimal first step

Do (c), as a run option, with no new schedule step.

1. In `coffee-ten-loop`, an analysis function that says which nests of a kernel have a
   legal parallel outermost loop, from rules 1-4 of section 3.1 (no new `Step`, no new
   field in `Loop`). A new `Step` and a `parallel` flag on `Loop` touch every `Loop` literal
   (`unroll: false` appears 11 times in 7 files) and every `match` on `Step`, so it
   goes second.
2. In `coffee-ten-codegen-c`, `ct_entry_range` for those nests, and
   `CompiledKernel::run_parallel(buffers, threads)` with `std::thread::scope`.
3. In `coffee-ten-bench`, `--threads T` for the harness, and the thread count in the
   record.

Checks, each with its command, before any timing:

- `run_parallel` equals the serial `run` bit for bit under `TEST_FLAGS`, for T = 1, 2, 3,
  4, 7 (T that does not divide the extent), on matmul with N not a multiple of the tile
  (a guard), repeated a few hundred times. A race that shows up one run in a hundred
  needs repetition.
- `run_parallel` with T = 1 equals `run` and is within noise of its time.
- The register tile still keeps `C` in registers with the run-time loop bounds: the same
  float-phi count in the optimized IR as in `tests/unroll.rs`.
- On Linux, one run of the parallel test under ThreadSanitizer (`-fsanitize=thread` on the
  generated C and on the test binary). Not checked whether the runner supports it.

### 6.2 How to measure it

One machine, the conditions of BENCHMARKS.md (AC, Low Power Mode off, cold start, pauses),
interleaved variants as now.

- **Schedules:** `regtile` (compute-bound, 94-96% of one-core peak, the hardest to scale)
  and `tiled + vec` (more memory traffic); the naive kernel as the clean check.
  Parallel loop: `io` (N / 4 and N / 32 iterations).
- **Sizes:** N = 256, 512, 1024 for the decision; N = 64 and 128 to record the loss.
- **Threads:** 1 (through `run_parallel`), 1 (through `run`), 2, 4, 6, 10. 4 is the number
  of fast cores on this Mac.
- **Numbers to report:** GFLOP/s (2 N^3 / time as now), speedup against `run` in the
  same session, and speedup / T. No percent of peak until a multi-core peak exists.

What would justify going on. These thresholds are my proposal, not derived; they are
the point of the review:

- T = 4, `regtile`, N = 512 and 1024: speedup of at least 2.5 (at least 62% of 4), and the
  p25-p75 ranges of the two do not overlap. The estimate in 4.5 allows 3.7 at N = 512, so
  2.5 leaves room for the bandwidth and core-mix effects the estimate leaves out.
- T = 4, N = 256: at least 1.5. The estimate for scope is 2.5.
- T = 1 through `run_parallel` within 3% of `run` (the cost of the new path where it does
  nothing).
- No result below N = 128 is required; the table in 4.5 says it will not gain.

What would stop it: a speedup below 1.5 at N = 1024 with T = 4 means the limit is memory
or placement, not thread start, and a pool will not help. Then write that down in
BENCHMARKS.md and stop.

Only after that: the `parallel(nest, loop)` step (it makes the choice part of the
schedule text and the record), a parked pool if N <= 256 matters, and the cost model.

## 7. Open questions for Nikita

1. OpenMP: the plan says no new dependencies, and on macOS OpenMP needs Homebrew
   `libomp`. Is it a firm no? The design assumes yes.
2. Scope: parallelism contradicts the README non-goal "threads and GPU in the MVP". Is
   it acceptable as post-MVP research, with the README lines changed when code lands?
3. First step: run option first (as recommended), or the `parallel` schedule step at
   once, so that it appears in schedule text and records?
4. A `split` of a marked loop: refuse (proposed) or clear the mark as `unroll` does?
5. Thread count: always an explicit argument, or detect it? If detect, which cores: all
   10, or the fast ones only (that needs `sysctlbyname` through an `extern "C"` block like
   `raise_qos`, no crate)?
6. Pool: is a global thread pool acceptable later (more `unsafe`, process state), or
   should the design stay with `thread::scope`?
7. Thresholds in 6.2: right numbers? What speedup at N = 1024 makes this worth keeping?
8. Hypothesis: should a parallel result get its own hypothesis and run, and after which
   plan item (C4 is still open for the `ji` <= 16 group)?
9. Thermals: the machine is fanless. Is it acceptable to measure all-core runs on it, or
   is a second machine needed first?

## 8. What this note did not check

- Whether the Linux CI image has an OpenMP runtime for clang, or `-fsanitize=thread`.
- The core count of the CI runners.
- Whether worker threads inherit the QoS class of the thread that spawns them.
- The speed of an efficiency core against a fast core on these kernels.
- OpenMP runtime clashes inside a Python process, and OpenMP fork cost (no OpenMP
  kernel was timed).
- Any parallel kernel: none exists. The only measurements are the thread start costs in
  4.5, taken with an empty-thread scratch program, and the `clang` and `sysctl` output in
  4.1 and 5.3.
