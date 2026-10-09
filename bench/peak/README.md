# peak

Single-thread peak f32 FLOP/s (FMA loop) and single-thread bandwidth (STREAM triad).
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

FMA_CHAINS = 16 assumes 4 pipes x 4-cycle latency (not measured).
The QoS class raises the chance of a P-core; it does not guarantee one.

Real numbers: take them per the project's benchmark method, on an idle machine,
never during builds or other jobs.
