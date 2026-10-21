/*
 * peak: single-thread f32 peak FLOP/s (FMA loop), FMA rate with the accumulator
 * in memory (vector and scalar), the latency-bound rate of one dependent scalar FMA
 * chain and of independent chains of length 256, single-thread memory bandwidth and
 * L2 bandwidth (STREAM triad).
 * See README.md.
 *
 * Build (from the repo root):
 *   clang -std=c99 -O2 -ffp-contract=fast -o peak bench/peak/peak.c
 * Add -march=native on x86 only; on Apple arm64 nothing else.
 *
 * Usage: ./peak [--check] [--quick]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef __APPLE__
#include <pthread.h>
#include <sys/qos.h>
#endif

typedef float v4f __attribute__((vector_size(16)));

/*
 * Independent accumulators so FMA latency is hidden: 4 pipes x 4-cycle
 * latency on recent Apple P-cores (assumed, not measured).
 */
#define FMA_CHAINS 16

/* FMA loop iterations per kernel call. */
#define FMA_ITERS (1L << 20)

/* STREAM triad: floats per array, 128 MiB each, far beyond caches. */
#define TRIAD_N (1 << 25)
#define TRIAD_S 0.5f

/*
 * STREAM triad sized for L2: 3 arrays x 1 MiB. Assumed, not measured: L1D is
 * 128 KiB and L2 is 16 MiB per P-cluster on Apple M-series.
 */
#define TRIAD_L2_N (1 << 18)

/*
 * Accumulator in memory: c[j] = fma(a, b[j], c[j]) on two 4 KiB arrays, assumed
 * L1-resident (L1D 128 KiB, assumed). 1024 floats is 256 float32x4 vectors, more
 * than the 32 vector registers, so c cannot stay in registers across passes.
 */
#define MEM_ACC_N 1024
/* Passes between clock reads; a sample repeats batches until it has run MIN_SAMPLE_NS. */
#define MEM_ACC_BATCH 1024

/*
 * Dependent FMA chain: FMAs per call. Long enough that call overhead is noise.
 * The operands are fma_operands() (a = 1, b = 0.5) scaled so a * b = 1e-7: x starts at 1
 * and creeps up by about 1e-7 per FMA until the increment is below half an ulp
 * (x = 2), where it stops. x stays in [1, 2]: normal, no denormal, no inf.
 */
#define CHAIN_ITERS (1L << 20)
#define CHAIN_A_SCALE 1e-3f
#define CHAIN_B_SCALE 2e-4f

/*
 * Reduction chains of length CHAIN256_K, the naive matmul loop body: for each of
 * CHAIN256_J columns, acc = c[j]; acc = fma(a[k], b[k * J + j], acc) over k; c[j] = acc.
 * The shape of the matmul reduction at N = 256: b is K x J = 256 x 256 floats = 256 KiB
 * (row stride 1 KiB), so each chain walks a column of b with a 1 KiB stride over 256 KiB,
 * which is L2, not L1D (128 KiB, assumed). a is one 256-float row (1 KiB), c is 1 KiB.
 * The overlap this measures is for that access shape; it is ASSUMED to carry to other N.
 */
#define CHAIN256_J 256
#define CHAIN256_K 256
/* Passes between clock reads; one pass is 256 x 256 FMAs, about 20 us at 3.5 GFLOP/s. */
#define CHAIN256_BATCH 64

#define SAMPLES 30
#define QUICK_SAMPLES 3
/* Each sample repeats its kernel until it has run at least this long. */
#define MIN_SAMPLE_NS 10000000

static volatile double g_sink;

static long long now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

/*
 * FMA operands come from volatile reads keyed on argc, so the compiler
 * cannot fold the loop into a closed form. argc is >= 1 in practice.
 */
static void fma_operands(int argc, float *a, float *b) {
    volatile float scale = argc > 0 ? 1.0f : 0.0f;
    *a = scale;
    *b = 0.5f * scale;
}

/*
 * acc[c] starts at c (distinct starts stop the compiler from merging the
 * chains), then acc[c] = fma(a, b, acc[c]) for iters iterations, one fmla per
 * chain. Written as acc*a+b, fmla (which accumulates into its destination)
 * needs a register copy per chain; written as acc+a*b, the compiler hoists the
 * invariant product and emits fadd. The builtin is never split.
 * Each FMA is 2 FLOP per lane, 4 lanes: 2 * 4 * FMA_CHAINS FLOP per iteration.
 */
__attribute__((noinline)) static void fma_kernel(long iters, float a, float b,
                                                 v4f out[FMA_CHAINS]) {
    v4f va = {a, a, a, a};
    v4f vb = {b, b, b, b};
    v4f acc[FMA_CHAINS];
    for (int c = 0; c < FMA_CHAINS; c++) {
        acc[c] = (v4f){(float)c, (float)c, (float)c, (float)c};
    }
    for (long i = 0; i < iters; i++) {
        for (int c = 0; c < FMA_CHAINS; c++) {
            acc[c] = __builtin_elementwise_fma(va, vb, acc[c]);
        }
    }
    for (int c = 0; c < FMA_CHAINS; c++) {
        out[c] = acc[c];
    }
}

/*
 * STREAM triad. Bytes per pass are counted STREAM-style: two reads and one
 * write of 4-byte floats, i.e. 3 * 4 * n. Write-allocate traffic is not counted.
 */
__attribute__((noinline)) static void triad(float *restrict x, const float *restrict y,
                                            const float *restrict z, float s, long n) {
    for (long i = 0; i < n; i++) {
        x[i] = y[i] + s * z[i];
    }
}

/*
 * One pass of the accumulator-in-memory kernel. Each float32x4 of c is loaded,
 * updated and stored once per pass, as the matmul schedules do when C[i, j] is
 * not in registers. noinline: c is written back to memory before the next pass.
 * Four vectors per iteration, as clang interleaves the vectorized matmul loop:
 * with one, the loop's own instructions limit it (31 vs 47 GFLOP/s by hand, M5).
 */
__attribute__((noinline)) static void fma_mem_acc_pass(float a, const v4f *restrict b,
                                                       v4f *restrict c) {
    v4f va = {a, a, a, a};
    for (long j = 0; j < MEM_ACC_N / 4; j += 4) {
        c[j] = __builtin_elementwise_fma(va, b[j], c[j]);
        c[j + 1] = __builtin_elementwise_fma(va, b[j + 1], c[j + 1]);
        c[j + 2] = __builtin_elementwise_fma(va, b[j + 2], c[j + 2]);
        c[j + 3] = __builtin_elementwise_fma(va, b[j + 3], c[j + 3]);
    }
}

/*
 * Scalar twin of fma_mem_acc_pass: per element, load b, load c, fmaf, store c,
 * one float per iteration. That is the shape clang gives the inner loop of the
 * naive matmul under the benchmark flags (ldr s, ldr s, fmadd s, str s, add,
 * cmp, b.ne). The pragmas keep it so: peak.c is built with -O2, where the loop
 * or the SLP vectorizer would otherwise make it fmla.4s.
 */
__attribute__((noinline)) static void fma_mem_acc_scalar_pass(float a, const float *restrict b,
                                                              float *restrict c) {
#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)
    for (long j = 0; j < MEM_ACC_N; j++) {
        c[j] = __builtin_fmaf(a, b[j], c[j]);
    }
}

/*
 * One endless dependent chain through the addend: x = fma(a, b, x), one fmadd per
 * iteration, each waiting for the previous one. a and b are runtime values, so the
 * loop cannot be folded; noinline keeps it a loop in its own function.
 */
__attribute__((noinline)) static float fma_chain_kernel(long iters, float a, float b, float x) {
    for (long t = 0; t < iters; t++) {
        x = __builtin_fmaf(a, b, x);
    }
    return x;
}

/*
 * Naive matmul reduction in the (ko, i, j, ki) shape: CHAIN256_J independent chains of
 * length CHAIN256_K, the accumulator in a register for the whole chain. Per inner
 * iteration: ldr s (a), ldr s (b), fmadd, add, cmp, b.ne. The pragmas keep clang from
 * vectorizing across j or unrolling over k, which would change what is measured.
 */
__attribute__((noinline)) static void fma_chain256_pass(const float *restrict a,
                                                        const float *restrict b,
                                                        float *restrict c) {
#pragma clang loop vectorize(disable) interleave(disable)
    for (long j = 0; j < CHAIN256_J; j++) {
        float acc = c[j];
#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)
        for (long k = 0; k < CHAIN256_K; k++) {
            acc = __builtin_fmaf(a[k], b[k * CHAIN256_J + j], acc);
        }
        c[j] = acc;
    }
}

static double fma_flop_per_call(void) {
    return 2.0 * 4 * FMA_CHAINS * (double)FMA_ITERS;
}

static double fma_mem_acc_flop_per_pass(void) {
    return 2.0 * MEM_ACC_N;
}

static double fma_chain_flop_per_call(void) {
    return 2.0 * (double)CHAIN_ITERS;
}

static double fma_chain256_flop_per_pass(void) {
    return 2.0 * CHAIN256_J * CHAIN256_K;
}

static double triad_bytes_per_call(long n) {
    return 3.0 * 4 * n;
}

/* Returns GFLOP/s (FLOP per ns). */
static double fma_sample(float a, float b) {
    v4f out[FMA_CHAINS];
    long long t0 = now_ns();
    long long t1;
    long calls = 0;
    do {
        fma_kernel(FMA_ITERS, a, b, out);
        for (int c = 0; c < FMA_CHAINS; c++) {
            g_sink += out[c][0] + out[c][1] + out[c][2] + out[c][3];
        }
        calls++;
        t1 = now_ns();
    } while (t1 - t0 < MIN_SAMPLE_NS);
    return (double)calls * fma_flop_per_call() / (double)(t1 - t0);
}

/* Returns GFLOP/s (FLOP per ns). */
static double fma_mem_acc_sample(float a, const v4f *b, v4f *c) {
    long long t0 = now_ns();
    long long t1;
    long passes = 0;
    do {
        for (int p = 0; p < MEM_ACC_BATCH; p++) {
            fma_mem_acc_pass(a, b, c);
        }
        passes += MEM_ACC_BATCH;
        t1 = now_ns();
    } while (t1 - t0 < MIN_SAMPLE_NS);
    g_sink += c[0][0];
    return (double)passes * fma_mem_acc_flop_per_pass() / (double)(t1 - t0);
}

/* Returns GFLOP/s (FLOP per ns). */
static double fma_mem_acc_scalar_sample(float a, const float *b, float *c) {
    long long t0 = now_ns();
    long long t1;
    long passes = 0;
    do {
        for (int p = 0; p < MEM_ACC_BATCH; p++) {
            fma_mem_acc_scalar_pass(a, b, c);
        }
        passes += MEM_ACC_BATCH;
        t1 = now_ns();
    } while (t1 - t0 < MIN_SAMPLE_NS);
    g_sink += c[0];
    return (double)passes * fma_mem_acc_flop_per_pass() / (double)(t1 - t0);
}

/* Returns GFLOP/s (FLOP per ns). */
static double fma_chain_sample(float a, float b) {
    long long t0 = now_ns();
    long long t1;
    long calls = 0;
    /* x is fed from call to call: with identical arguments clang hoists the pure call out of the loop. */
    float x = 1.0f;
    do {
        x = fma_chain_kernel(CHAIN_ITERS, a, b, x);
        calls++;
        t1 = now_ns();
    } while (t1 - t0 < MIN_SAMPLE_NS);
    g_sink += x;
    return (double)calls * fma_chain_flop_per_call() / (double)(t1 - t0);
}

/* Returns GFLOP/s (FLOP per ns). */
static double fma_chain256_sample(const float *a, const float *b, float *c) {
    long long t0 = now_ns();
    long long t1;
    long passes = 0;
    do {
        for (int p = 0; p < CHAIN256_BATCH; p++) {
            fma_chain256_pass(a, b, c);
        }
        passes += CHAIN256_BATCH;
        t1 = now_ns();
    } while (t1 - t0 < MIN_SAMPLE_NS);
    g_sink += c[0];
    return (double)passes * fma_chain256_flop_per_pass() / (double)(t1 - t0);
}

/* Returns GB/s (1e9 bytes per second, i.e. bytes per ns). */
static double triad_sample(float *x, const float *y, const float *z, long n) {
    long long t0 = now_ns();
    long long t1;
    long calls = 0;
    do {
        triad(x, y, z, TRIAD_S, n);
        calls++;
        t1 = now_ns();
    } while (t1 - t0 < MIN_SAMPLE_NS);
    return (double)calls * triad_bytes_per_call(n) / (double)(t1 - t0);
}

typedef struct {
    double median;
    double p25;
    double p75;
    double min;
} stats_t;

static int cmp_double(const void *p, const void *q) {
    double x = *(const double *)p;
    double y = *(const double *)q;
    return (x > y) - (x < y);
}

/*
 * Sorts rate[0..n-1] ascending and takes integer indices, no interpolation:
 *   min = r[0], p25 = r[(n-1)/4], median = r[(n-1)/2], p75 = r[3*(n-1)/4].
 */
static stats_t summarize(double *rate, int n) {
    qsort(rate, n, sizeof rate[0], cmp_double);
    stats_t s = {
        .median = rate[(n - 1) / 2],
        .p25 = rate[(n - 1) / 4],
        .p75 = rate[3 * (n - 1) / 4],
        .min = rate[0],
    };
    return s;
}

static void print_stats(const char *name, const char *unit, stats_t s, int n) {
    printf("%-10s median %9.3f  p25 %9.3f  p75 %9.3f  min %9.3f  %s  n=%d\n",
           name, s.median, s.p25, s.p75, s.min, unit, n);
}

/* Fixed small problem: FMA a=1, b=0.5, 1000 iters; triad n=1024, s=0.5. */
static int run_check(float a, float b) {
    v4f out[FMA_CHAINS];
    fma_kernel(1000, a, b, out);
    for (int c = 0; c < FMA_CHAINS; c++) {
        for (int l = 0; l < 4; l++) {
            float want = (float)c + 500.0f;
            if (out[c][l] != want) {
                printf("check FAILED: fma chain %d lane %d: got %.9g, want %.9g\n",
                       c, l, (double)out[c][l], (double)want);
                return 1;
            }
        }
    }

    /* triad_l2 runs the same triad() at n = TRIAD_L2_N; this n = 1024 check covers it. */
    long n = 1024;
    float *x = malloc(n * sizeof *x);
    float *y = malloc(n * sizeof *y);
    float *z = malloc(n * sizeof *z);
    if (!x || !y || !z) {
        printf("check FAILED: malloc\n");
        return 1;
    }
    for (long i = 0; i < n; i++) {
        x[i] = 0.0f;
        y[i] = (float)i;
        z[i] = 2.0f;
    }
    triad(x, y, z, TRIAD_S, n);
    for (long i = 0; i < n; i++) {
        float want = (float)i + 1.0f;
        if (x[i] != want) {
            printf("check FAILED: triad i=%ld: got %.9g, want %.9g\n",
                   i, (double)x[i], (double)want);
            return 1;
        }
    }
    free(x);
    free(y);
    free(z);

    /* Accumulator in memory: c[j] = j, then 1000 passes with a = 1, b[j] = 0.5. */
    v4f mem_b[MEM_ACC_N / 4];
    v4f mem_c[MEM_ACC_N / 4];
    for (int j = 0; j < MEM_ACC_N / 4; j++) {
        mem_b[j] = (v4f){0.5f, 0.5f, 0.5f, 0.5f};
    }
    for (int j = 0; j < MEM_ACC_N; j++) {
        mem_c[j / 4][j % 4] = (float)j;
    }
    for (int p = 0; p < 1000; p++) {
        fma_mem_acc_pass(1.0f, mem_b, mem_c);
    }
    for (int j = 0; j < MEM_ACC_N; j++) {
        float want = (float)j + 500.0f;
        if (mem_c[j / 4][j % 4] != want) {
            printf("check FAILED: mem_acc j=%d: got %.9g, want %.9g\n", j,
                   (double)mem_c[j / 4][j % 4], (double)want);
            return 1;
        }
    }
    /* Scalar accumulator in memory: same as above with one float per FMA. */
    float sc_b[MEM_ACC_N];
    float sc_c[MEM_ACC_N];
    for (int j = 0; j < MEM_ACC_N; j++) {
        sc_b[j] = 0.5f;
        sc_c[j] = (float)j;
    }
    for (int p = 0; p < 1000; p++) {
        fma_mem_acc_scalar_pass(1.0f, sc_b, sc_c);
    }
    for (int j = 0; j < MEM_ACC_N; j++) {
        float want = (float)j + 500.0f;
        if (sc_c[j] != want) {
            printf("check FAILED: mem_acc_scalar j=%d: got %.9g, want %.9g\n", j,
                   (double)sc_c[j], (double)want);
            return 1;
        }
    }

    /* Dependent chain vs a plain loop of the same recurrence; benchmark constants, so every step rounds. */
    float chain_a = 1.0f * CHAIN_A_SCALE;
    float chain_b = 0.5f * CHAIN_B_SCALE;
    float chain_want = 1.0f;
    for (long t = 0; t < 1000; t++) {
        chain_want = __builtin_fmaf(chain_a, chain_b, chain_want);
    }
    float chain_got = fma_chain_kernel(1000, chain_a, chain_b, 1.0f);
    if (chain_got != chain_want) {
        printf("check FAILED: fma_chain: got %.9g, want %.9g\n", (double)chain_got,
               (double)chain_want);
        return 1;
    }

    /* Chains of 256 vs a plain reference with fmaf in the same k order, 3 passes. */
    float *ch_a = malloc(CHAIN256_K * sizeof *ch_a);
    float *ch_b = malloc((size_t)CHAIN256_K * CHAIN256_J * sizeof *ch_b);
    float ch_c[CHAIN256_J];
    float ch_ref[CHAIN256_J];
    if (!ch_a || !ch_b) {
        printf("check FAILED: malloc\n");
        return 1;
    }
    for (int k = 0; k < CHAIN256_K; k++) {
        ch_a[k] = 0.01f * (float)(k % 7 + 1);
    }
    for (int i = 0; i < CHAIN256_K * CHAIN256_J; i++) {
        ch_b[i] = 0.1f * (float)(i % 5 + 1);
    }
    for (int j = 0; j < CHAIN256_J; j++) {
        ch_c[j] = (float)j;
        ch_ref[j] = (float)j;
    }
    for (int p = 0; p < 3; p++) {
        fma_chain256_pass(ch_a, ch_b, ch_c);
        for (int j = 0; j < CHAIN256_J; j++) {
            for (int k = 0; k < CHAIN256_K; k++) {
                ch_ref[j] = __builtin_fmaf(ch_a[k], ch_b[k * CHAIN256_J + j], ch_ref[j]);
            }
        }
    }
    for (int j = 0; j < CHAIN256_J; j++) {
        if (ch_c[j] != ch_ref[j]) {
            printf("check FAILED: fma_chain256 j=%d: got %.9g, want %.9g\n", j,
                   (double)ch_c[j], (double)ch_ref[j]);
            return 1;
        }
    }
    free(ch_a);
    free(ch_b);
    printf("check ok\n");
    return 0;
}

static double g_fma_rate[SAMPLES];
static double g_triad_rate[SAMPLES];
static double g_triad_l2_rate[SAMPLES];
static double g_mem_acc_rate[SAMPLES];
static double g_mem_acc_scalar_rate[SAMPLES];
static double g_chain_rate[SAMPLES];
static double g_chain256_rate[SAMPLES];

/* Fills rate[0..samples-1] with accumulator-in-memory GFLOP/s; a from fma_operands, b[j] = 0.5. */
static void run_mem_acc(float a, int samples, int warmup, double *rate) {
    v4f b[MEM_ACC_N / 4];
    v4f c[MEM_ACC_N / 4];
    for (int j = 0; j < MEM_ACC_N / 4; j++) {
        b[j] = (v4f){0.5f, 0.5f, 0.5f, 0.5f};
        c[j] = (v4f){0.0f, 0.0f, 0.0f, 0.0f};
    }
    for (int s = 0; s < warmup + samples; s++) {
        double r = fma_mem_acc_sample(a, b, c);
        if (s >= warmup) {
            rate[s - warmup] = r;
        }
    }
}

/* Fills rate[0..samples-1] with scalar accumulator-in-memory GFLOP/s; a from fma_operands, b[j] = 0.5. */
static void run_mem_acc_scalar(float a, int samples, int warmup, double *rate) {
    float b[MEM_ACC_N];
    float c[MEM_ACC_N];
    for (int j = 0; j < MEM_ACC_N; j++) {
        b[j] = 0.5f;
        c[j] = 0.0f;
    }
    for (int s = 0; s < warmup + samples; s++) {
        double r = fma_mem_acc_scalar_sample(a, b, c);
        if (s >= warmup) {
            rate[s - warmup] = r;
        }
    }
}

/* Fills rate[0..samples-1] with dependent-chain GFLOP/s. */
static void run_chain(float a, float b, int samples, int warmup, double *rate) {
    for (int s = 0; s < warmup + samples; s++) {
        double r = fma_chain_sample(a * CHAIN_A_SCALE, b * CHAIN_B_SCALE);
        if (s >= warmup) {
            rate[s - warmup] = r;
        }
    }
}

/* Fills rate[0..samples-1] with chain-of-256 GFLOP/s; a[k] = a, b = 0.5. Returns 1 on malloc failure. */
static int run_chain256(float a, int samples, int warmup, double *rate) {
    float *ka = malloc(CHAIN256_K * sizeof *ka);
    float *kb = malloc((size_t)CHAIN256_K * CHAIN256_J * sizeof *kb);
    float kc[CHAIN256_J];
    if (!ka || !kb) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }
    for (int k = 0; k < CHAIN256_K; k++) {
        ka[k] = a;
    }
    for (int i = 0; i < CHAIN256_K * CHAIN256_J; i++) {
        kb[i] = 0.5f;
    }
    for (int j = 0; j < CHAIN256_J; j++) {
        kc[j] = 0.0f;
    }
    for (int s = 0; s < warmup + samples; s++) {
        double r = fma_chain256_sample(ka, kb, kc);
        if (s >= warmup) {
            rate[s - warmup] = r;
        }
    }
    free(ka);
    free(kb);
    return 0;
}

/* Fills rate[0..samples-1] with triad GB/s over n floats. Returns 1 on malloc failure. */
static int run_triad(long n, int samples, int warmup, double *rate) {
    float *x = malloc(n * sizeof *x);
    float *y = malloc(n * sizeof *y);
    float *z = malloc(n * sizeof *z);
    if (!x || !y || !z) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }
    for (long i = 0; i < n; i++) {
        x[i] = 0.0f;
        y[i] = 1.0f;
        z[i] = 2.0f;
    }
    for (int s = 0; s < warmup + samples; s++) {
        double r = triad_sample(x, y, z, n);
        if (s >= warmup) {
            rate[s - warmup] = r;
        }
    }
    free(x);
    free(y);
    free(z);
    return 0;
}

int main(int argc, char **argv) {
#ifdef __APPLE__
    /* Raises the chance of a P-core; does not guarantee one. */
    if (pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0) != 0) {
        fprintf(stderr, "warning: could not set QoS class\n");
    }
#endif

    int do_check = 0;
    int quick = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--check") == 0) {
            do_check = 1;
        } else if (strcmp(argv[i], "--quick") == 0) {
            quick = 1;
        } else {
            fprintf(stderr, "usage: %s [--check] [--quick]\n", argv[0]);
            return 2;
        }
    }

    float a;
    float b;
    fma_operands(argc, &a, &b);
    if (do_check) {
        return run_check(a, b);
    }

    int samples = quick ? QUICK_SAMPLES : SAMPLES;
    int warmup = quick ? 0 : 1;

    for (int s = 0; s < warmup + samples; s++) {
        double rate = fma_sample(a, b);
        if (s >= warmup) {
            g_fma_rate[s - warmup] = rate;
        }
    }

    run_mem_acc(a, samples, warmup, g_mem_acc_rate);
    run_mem_acc_scalar(a, samples, warmup, g_mem_acc_scalar_rate);
    run_chain(a, b, samples, warmup, g_chain_rate);
    if (run_chain256(a, samples, warmup, g_chain256_rate) != 0) {
        return 1;
    }
    if (run_triad(TRIAD_N, samples, warmup, g_triad_rate) != 0) {
        return 1;
    }
    if (run_triad(TRIAD_L2_N, samples, warmup, g_triad_l2_rate) != 0) {
        return 1;
    }

    stats_t fma = summarize(g_fma_rate, samples);
    stats_t mem_acc = summarize(g_mem_acc_rate, samples);
    stats_t mem_acc_scalar = summarize(g_mem_acc_scalar_rate, samples);
    stats_t chain = summarize(g_chain_rate, samples);
    stats_t chain256 = summarize(g_chain256_rate, samples);
    stats_t triad_st = summarize(g_triad_rate, samples);
    stats_t triad_l2_st = summarize(g_triad_l2_rate, samples);

    print_stats("fma", "GFLOP/s", fma, samples);
    print_stats("mem_acc", "GFLOP/s", mem_acc, samples);
    print_stats("mem_acc_s", "GFLOP/s", mem_acc_scalar, samples);
    print_stats("chain", "GFLOP/s", chain, samples);
    print_stats("chain256", "GFLOP/s", chain256, samples);
    print_stats("triad", "GB/s", triad_st, samples);
    print_stats("triad_l2", "GB/s", triad_l2_st, samples);
    printf("checksum %g\n", g_sink);
    printf("{\"fma_gflops\":{\"median\":%.4f,\"p25\":%.4f,\"p75\":%.4f,\"min\":%.4f,\"n\":%d},"
           "\"triad_gbs\":{\"median\":%.4f,\"p25\":%.4f,\"p75\":%.4f,\"min\":%.4f,\"n\":%d},"
           "\"fma_chains\":%d,\"triad_n\":%d,"
           "\"triad_l2_gbs\":{\"median\":%.4f,\"p25\":%.4f,\"p75\":%.4f,\"min\":%.4f,\"n\":%d},"
           "\"triad_l2_n\":%d,"
           "\"fma_mem_acc_gflops\":{\"median\":%.4f,\"p25\":%.4f,\"p75\":%.4f,\"min\":%.4f,\"n\":%d},"
           "\"mem_acc_n\":%d,"
           "\"fma_mem_acc_scalar_gflops\":{\"median\":%.4f,\"p25\":%.4f,\"p75\":%.4f,\"min\":%.4f,\"n\":%d},"
           "\"fma_chain_gflops\":{\"median\":%.4f,\"p25\":%.4f,\"p75\":%.4f,\"min\":%.4f,\"n\":%d},"
           "\"fma_chain256_gflops\":{\"median\":%.4f,\"p25\":%.4f,\"p75\":%.4f,\"min\":%.4f,\"n\":%d}}\n",
           fma.median, fma.p25, fma.p75, fma.min, samples,
           triad_st.median, triad_st.p25, triad_st.p75, triad_st.min, samples,
           FMA_CHAINS, TRIAD_N,
           triad_l2_st.median, triad_l2_st.p25, triad_l2_st.p75, triad_l2_st.min, samples,
           TRIAD_L2_N,
           mem_acc.median, mem_acc.p25, mem_acc.p75, mem_acc.min, samples,
           MEM_ACC_N,
           mem_acc_scalar.median, mem_acc_scalar.p25, mem_acc_scalar.p75, mem_acc_scalar.min,
           samples,
           chain.median, chain.p25, chain.p75, chain.min, samples,
           chain256.median, chain256.p25, chain256.p75, chain256.min, samples);
    return 0;
}
