/*
 * peak: single-thread f32 peak FLOP/s (FMA loop) and single-thread memory
 * bandwidth (STREAM triad). See README.md.
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

static double fma_flop_per_call(void) {
    return 2.0 * 4 * FMA_CHAINS * (double)FMA_ITERS;
}

static double triad_bytes_per_call(void) {
    return 3.0 * 4 * TRIAD_N;
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

/* Returns GB/s (1e9 bytes per second, i.e. bytes per ns). */
static double triad_sample(float *x, const float *y, const float *z) {
    long long t0 = now_ns();
    long long t1;
    long calls = 0;
    do {
        triad(x, y, z, TRIAD_S, TRIAD_N);
        calls++;
        t1 = now_ns();
    } while (t1 - t0 < MIN_SAMPLE_NS);
    return (double)calls * triad_bytes_per_call() / (double)(t1 - t0);
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
    printf("check ok\n");
    return 0;
}

static double g_fma_rate[SAMPLES];
static double g_triad_rate[SAMPLES];

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

    long n = TRIAD_N;
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
        double rate = triad_sample(x, y, z);
        if (s >= warmup) {
            g_triad_rate[s - warmup] = rate;
        }
    }

    stats_t fma = summarize(g_fma_rate, samples);
    stats_t triad_st = summarize(g_triad_rate, samples);

    print_stats("fma", "GFLOP/s", fma, samples);
    print_stats("triad", "GB/s", triad_st, samples);
    printf("checksum %g\n", g_sink);
    printf("{\"fma_gflops\":{\"median\":%.4f,\"p25\":%.4f,\"p75\":%.4f,\"min\":%.4f,\"n\":%d},"
           "\"triad_gbs\":{\"median\":%.4f,\"p25\":%.4f,\"p75\":%.4f,\"min\":%.4f,\"n\":%d},"
           "\"fma_chains\":%d,\"triad_n\":%d}\n",
           fma.median, fma.p25, fma.p75, fma.min, samples,
           triad_st.median, triad_st.p25, triad_st.p75, triad_st.min, samples,
           FMA_CHAINS, TRIAD_N);

    free(x);
    free(y);
    free(z);
    return 0;
}
