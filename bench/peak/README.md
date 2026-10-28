# peak

Single-thread peak f32 FLOP/s (FMA loop) and single-thread memory and L2 bandwidth (STREAM triad).
Single-thread because the kernels we compare are single-threaded.

Build from the repo root (add `-march=native` on x86 only):

    clang -std=c99 -O2 -ffp-contract=fast -o peak bench/peak/peak.c

    ./peak          # 1 warm-up + 30 samples per kernel
    ./peak --check  # tiny correctness run: "check ok", exit 0; mismatch: exit 1
    ./peak --quick  # 3 samples, no warm-up: smoke run only

Each sample repeats its kernel for at least 10 ms; rate = work / time.
Sorted rates r[0..n-1]: min r[0], p25 r[(n-1)/4], median r[(n-1)/2], p75 r[3*(n-1)/4].

FMA: 2 FLOP per lane per FMA x 4 lanes x 16 chains = 128 FLOP per iteration.
Triad: STREAM, 3*4*N bytes per pass (2 reads + 1 write of f32), write-allocate not counted.
GFLOP and GB are 1e9 FLOP and 1e9 bytes.

triad_l2 is the L2 bandwidth ceiling of the cost model: the same triad on three
1 MiB arrays (3 MiB total), so the working set fits in L2 but not in L1D, and the
arrays stay in L2 across samples. It is for kernels whose working set is L2-resident
but larger than L1D; the memory triad is the ceiling for traffic that reaches DRAM.
The cache sizes behind it are ASSUMED, not measured: L1D 128 KiB and L2 16 MiB per
P-cluster on Apple M-series. If the real sizes differ, the number measures another
level; resize TRIAD_L2_N to the measured sizes.

fma_mem_acc is the FMA throughput when every FMA also loads and stores its accumulator, as the
current matmul schedules do while C[i, j] lives in memory: c[j] = fma(a, b[j], c[j]) over 1024
floats in two 4 KiB arrays, assumed L1-resident (L1D 128 KiB, assumed). Its 256 float32x4
vectors are more than the 32 vector registers, so c cannot stay in registers across passes.
The loop does four vectors per iteration, as clang interleaves the vectorized matmul loop;
with one vector per iteration the loop instructions themselves were the limit (the
2026-10-18 results file has that older kernel: 31 GFLOP/s, against about 47 now).
It is the compute ceiling the cost model should use for such schedules: the fma figure is
reached only when the accumulator stays in registers, so it overstates what these schedules
can do. --check verifies it exactly (c[j] = j + 500 after 1000 passes).

fma_mem_acc_scalar is the same access pattern as fma_mem_acc in scalar code, one float per load, FMA and store, like the naive and tiled matmul schedules. It is the scalar load/store calibration point of the cost model: the naive matmul measures 7.5-8.4 GFLOP/s, while the vector point divided by 4 lanes predicts about 11.5.

fma_mem_acc_scalar_unrolled is the same scalar pattern with 16 floats per iteration as straight-line code (16 `fmadd s`, `ldp s`, `stp s`), the shape clang gives a fully unrolled short scalar loop. It is the scalar point of the cost model for such loops: 23.2 GFLOP/s on M5, against 8.55 for one float per iteration; an empty `asm` on each result keeps the SLP vectorizer from turning it into `fmla.4s` under the unchanged build command.

fma_store is the vector FMA throughput when every FMA result is stored and nothing is loaded: `c[j] = fma(a, s, y_q)` over the same 1024 floats, four vectors per iteration, operands in registers, `--check` exact against a plain loop with `fmaf`. It is the store ceiling of the cost model: loads and stores use separate ports, so the load-bound fma_mem_acc (2 loads + 1 store per FMA) cannot calibrate the store side (hand microbenchmark, M5: 2 loads + 1 store 45-47 GFLOP/s, 2 loads and no store about 49.5, 1 load about 84).

fma_chain and fma_chain256 are the latency ceiling of reductions. When the innermost loop reduces over k, each FMA of a C element waits for the previous one. fma_chain is one endless dependent scalar chain, `x = fmaf(a, b, x)` through the addend, 2 FLOP per FMA: the FMA latency is 2 / rate ns. The operands have a*b = 1e-7, so x starts at 1, creeps up and stops at 2 (increment below half an ulp): bounded, normal, no denormals or inf. fma_chain256 is the matmul reduction at N = 256 in the naive loop body (`acc = fmaf(a[k], b[k * 256 + j], acc)`, 256 chains of length 256): a is one 256-float row (1 KiB), b is 256 x 256 floats = 256 KiB with a 1 KiB row stride, so each chain walks a column of b with a 1 KiB stride and the data sits in L2, not L1D (128 KiB, assumed). Vectorizing, interleaving and unrolling are off. It shows how much the out-of-order core overlaps the end of one chain with the start of the next. The cost model derives an overlap window W = 256 * (chain256 / chain - 1) in FMAs from the two medians; W is for this access shape and is ASSUMED to carry to other N. Both are checked exactly against a plain C loop with `fmaf` in the same order. The inner loop has two address increments (the a and b strides differ), so it is 7 instructions: 2 loads, fmadd, 2 adds, cmp, b.ne.

fma_col_reload is the scalar matmul loop nest of the codegen's "scalar, ii innermost" schedules with the k loop directly outside ii: N = 256, f32, loop order io (4), j, k, ii (64, innermost, rolled), `C[io*64+ii][j] += A[io*64+ii][k] * B[k][j]`, on three 256 x 256 arrays with a row stride of exactly 256 floats (1 KiB, no padding). The inner loop is 7 instructions (ldr s, ldr s, fmadd s, str s, add, cmp, b.ne) with a 1 KiB step, kept scalar and rolled by a loop pragma. It is the calibration point for schedules that reload a column of C at the next k: the load of C[ii][j] at k + 1 needs the value stored at k, and the other stores in flight go to addresses equal to it mod 256 B, and a likely reading (not proven) is that the load is matched against the wrong store (hand measurement, M5: 2.1 GFLOP/s here, 7.8 with C's row stride padded to 272 floats; padding A or B changes nothing). FLOP per call is 2 * 256^3. A = a, B = 0.5, C starts at 0 and grows by 128 * a per call, so nothing overflows. `--check` compares one call from a known C bit for bit with a plain loop in i, j, k order with `fmaf`; each element sees the same k order, so the results are identical.

fma_col_unrolled is the same scalar matmul (N = 256, f32, three 256 x 256 arrays, row stride exactly 1 KiB, no padding) with the schedule `split(i, io, ii, 16); split(k, ko, ki, 64); reorder(io, j, ko, ki, ii)`: loop order io (16), j, ko (4), ki (64), ii (16, innermost). Clang fully unrolls ii and keeps the 16 values of C in registers across the whole k loop, so the rolled ki body is 1 load of B[k][j], 16 loads of a column of A (1 KiB apart: `[x, #1024]`, `[x, #2048]`, ...), 16 `fmadd s` and no stores (17 `ldr s`, 16 `fmadd s`, `add`, `add`, `cmp`, `b.ne`). It is the calibration point for schedules that hold a C tile in registers and read A down a column; the cost model predicts 21.5-23.8 GFLOP/s for them, and run 3 measures 14-16. A likely reading, not proven: the 16 loads of A are at addresses equal mod 256 B and conflict (L1 banks). Hand measurements, M5, `-fno-vectorize -fno-slp-vectorize`, best of 7 x 100 ms, by A's row stride in floats: 256 15.7-15.9 GFLOP/s, 320/384/512 15.5-15.7, 1024 14.9, 288 17.9, 272 20.0, 258-264 21.0-21.3; A transposed (unit stride) 26.7. Under the unchanged build command a loop pragma on ki (`vectorize(disable) interleave(disable)`) is needed, otherwise clang vectorizes the reduction over ki and no `fmadd` is left; SLP does not merge the 16 FMAs, so no `asm` barrier is used. FLOP per call is 2 * 256^3, inputs as for fma_col_reload (A = a, B = 0.5, C grows by 128 * a per call). `--check` compares one call from a known C bit for bit with a plain loop in i, j, k order with `fmaf`.

FMA_CHAINS = 16 assumes 4 pipes x 4-cycle latency (not measured).
The QoS class raises the chance of a P-core; it does not guarantee one.

Real numbers: take them per the project's benchmark method, on an idle machine,
never during builds or other jobs.
