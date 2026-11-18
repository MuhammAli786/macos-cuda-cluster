// CPU compute for MLX 4-bit MoE experts (affine quantization, group size 64).
//
// Each expert lives in RAM as one packed byte row (the same .expert_pack files
// ring_gen.py writes): gate, up, down projections, each as
// weight (uint32, 8 nibbles each), scales (16-bit), biases (16-bit).
// MLX dequantizes as  w = scale * q + bias  per group of 64 inputs.
//
// moe_run() computes, for one token,
//   out = sum_e score_e * down_e( silu(gate_e x) * up_e x )
// using a persistent thread pool (spinning workers, so a call costs
// microseconds to start).
//
// Build: zig cc -O3 -mavx2 -mfma -shared -fPIC -o moe_cpu.so moe_cpu.c -lpthread

#include <immintrin.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#define MAXE 64      // max experts per call
#define MAXT 16      // max threads
#define MAXF 4096    // max expert width
static long SPIN_NS = 200000, NAP_NS = 20000;  // set by MOE_SPIN_US / MOE_NAP_US

static int D, F, T, BF16;
static long OFF[9];  // byte offsets: gate w,s,b  up w,s,b  down w,s,b

static struct {
    int n;
    const uint8_t *rows[MAXE];
    float score[MAXE];
    const float *x;
    float *out;
} J;

static float xsum[MAXF / 64];
static float H[MAXE * MAXF];
static pthread_t TH[MAXT];
static atomic_int GEN, DONE_A, DONE_B, DONE_G;
static void (*JFN)(int);  // generic parallel job (MODE 1)
static volatile int MODE;  // 0 = one-token moe_run, 1 = run JFN

static inline float h2f(uint16_t v) {
    if (BF16) {
        uint32_t u = (uint32_t)v << 16;
        float f;
        memcpy(&f, &u, 4);
        return f;
    }
    return _cvtsh_ss(v);  // IEEE half
}

static inline float hsum8(__m256 v) {
    __m128 a = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    a = _mm_add_ps(a, _mm_movehl_ps(a, a));
    a = _mm_add_ss(a, _mm_shuffle_ps(a, a, 1));
    return _mm_cvtss_f32(a);
}

// Activations are quantized to int8 per group of 64 (scale xd[g]) and stored
// in the nibble order of the packed weights: for byte b of a group's 32 weight
// bytes, the low nibble is input 8*(b/4)+2*(b%4) and the high nibble is +1.
// So xq[64g + b] pairs with low nibbles and xq[64g + 32 + b] with high ones.
static void quant_x(const float *x, int I, int8_t *xq, float *xd, float *xs) {
    for (int g = 0; g < I / 64; g++) {
        const float *v = x + g * 64;
        float amax = 0.f, sum = 0.f;
        for (int j = 0; j < 64; j++) { float a = fabsf(v[j]); if (a > amax) amax = a; sum += v[j]; }
        float d = amax / 127.f, inv = d > 0 ? 1.f / d : 0.f;
        xd[g] = d; xs[g] = sum;
        for (int b = 0; b < 32; b++) {
            int j = 8 * (b >> 2) + 2 * (b & 3);
            xq[g * 64 + b] = (int8_t)lrintf(v[j] * inv);
            xq[g * 64 + 32 + b] = (int8_t)lrintf(v[j + 1] * inv);
        }
    }
}

// dot(dequant(row), x) over I inputs.
static inline float qdot(const uint8_t *w, const uint16_t *s, const uint16_t *b,
                         const int8_t *xq, const float *xd, const float *xs, int I) {
    const __m256i m = _mm256_set1_epi8(15), ones = _mm256_set1_epi16(1);
    __m256 acc = _mm256_setzero_ps();
    float bacc = 0.f;
    for (int g = 0; g < I / 64; g++) {
        __m256i W = _mm256_loadu_si256((const __m256i *)(w + 32 * g));
        __m256i lo = _mm256_and_si256(W, m);
        __m256i hi = _mm256_and_si256(_mm256_srli_epi16(W, 4), m);
        __m256i p = _mm256_add_epi16(
            _mm256_maddubs_epi16(lo, _mm256_loadu_si256((const __m256i *)(xq + 64 * g))),
            _mm256_maddubs_epi16(hi, _mm256_loadu_si256((const __m256i *)(xq + 64 * g + 32))));
        p = _mm256_madd_epi16(p, ones);
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(h2f(s[g]) * xd[g]), acc);
        bacc += h2f(b[g]) * xs[g];
    }
    return hsum8(acc) + bacc;
}

static int8_t XQ[MAXF];
static float XD[MAXF / 64];

static void work_a(int t) {  // gate/up rows -> H
    int total = J.n * F, lo = (long)total * t / T, hi = (long)total * (t + 1) / T;
    for (int i = lo; i < hi; i++) {
        int e = i / F, r = i % F;
        const uint8_t *row = J.rows[e];
        float g = qdot(row + OFF[0] + (long)r * (D / 2), (const uint16_t *)(row + OFF[1]) + (long)r * (D / 64),
                       (const uint16_t *)(row + OFF[2]) + (long)r * (D / 64), XQ, XD, xsum, D);
        float u = qdot(row + OFF[3] + (long)r * (D / 2), (const uint16_t *)(row + OFF[4]) + (long)r * (D / 64),
                       (const uint16_t *)(row + OFF[5]) + (long)r * (D / 64), XQ, XD, xsum, D);
        H[e * F + r] = g / (1.f + expf(-g)) * u;
    }
}

static void work_b(int t) {  // down rows -> out
    int8_t hq[MAXE][MAXF];
    float hd[MAXE][MAXF / 64], hs[MAXE][MAXF / 64];
    for (int e = 0; e < J.n; e++) quant_x(H + e * F, F, hq[e], hd[e], hs[e]);
    int lo = (long)D * t / T, hi = (long)D * (t + 1) / T;
    for (int o = lo; o < hi; o++) {
        float acc = 0.f;
        for (int e = 0; e < J.n; e++) {
            const uint8_t *row = J.rows[e];
            acc += J.score[e] * qdot(row + OFF[6] + (long)o * (F / 2),
                                     (const uint16_t *)(row + OFF[7]) + (long)o * (F / 64),
                                     (const uint16_t *)(row + OFF[8]) + (long)o * (F / 64),
                                     hq[e], hd[e], hs[e], F);
        }
        J.out[o] = acc;
    }
}

static void spin_until(atomic_int *v, int target) {
    while (atomic_load_explicit(v, memory_order_acquire) < target) _mm_pause();
}

static void *worker(void *arg) {
    int t = (int)(long)arg, seen = 0;
    for (;;) {
        // Spin (no sleep) for up to SPIN_NS after the last job: the gap
        // between layers is ~1-2 ms and WSL wakes a sleeping thread ~1 ms
        // late. Only after a long idle (between requests) do we nap.
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        long spins = 0;
        while (atomic_load_explicit(&GEN, memory_order_acquire) == seen) {
            _mm_pause();
            if ((++spins & 1023) == 0) {
                clock_gettime(CLOCK_MONOTONIC, &t1);
                long ns = (t1.tv_sec - t0.tv_sec) * 1000000000L + (t1.tv_nsec - t0.tv_nsec);
                if (ns > SPIN_NS) { struct timespec ts = {0, NAP_NS}; nanosleep(&ts, 0); }
            }
        }
        seen = atomic_load(&GEN);
        if (MODE) {  // generic job from par()
            JFN(t);
            atomic_fetch_add(&DONE_G, 1);
            continue;
        }
        work_a(t);
        atomic_fetch_add(&DONE_A, 1);
        spin_until(&DONE_A, T);
        work_b(t);
        atomic_fetch_add(&DONE_B, 1);
    }
    return 0;
}

// d = hidden size, f = expert width, offs = 9 byte offsets, threads, bf16 flag
int moe_init(int d, int f, const long *offs, int threads, int bf16) {
    if (d % 64 || f % 64 || f > MAXF || d > MAXF || threads < 1 || threads > MAXT) return -1;
    D = d; F = f; T = threads; BF16 = bf16;
    if (getenv("MOE_SPIN_US")) SPIN_NS = atol(getenv("MOE_SPIN_US")) * 1000;
    if (getenv("MOE_NAP_US")) NAP_NS = atol(getenv("MOE_NAP_US")) * 1000;
    memcpy(OFF, offs, sizeof(OFF));
    for (int t = 1; t < T; t++) pthread_create(&TH[t], 0, worker, (void *)(long)t);
    return 0;
}

// base = start of the packed (E, row_bytes) array; idx/score = n experts.
int moe_run(const uint8_t *base, long row_bytes, int n, const int *idx, const float *score,
            const float *x, float *out) {
    if (n > MAXE) return -1;
    J.n = n;
    for (int e = 0; e < n; e++) {
        J.rows[e] = base + (long)idx[e] * row_bytes;
        J.score[e] = score[e];
    }
    J.x = x;
    J.out = out;
    MODE = 0;
    quant_x(x, D, XQ, XD, xsum);
    atomic_store(&DONE_A, 0);
    atomic_store(&DONE_B, 0);
    atomic_fetch_add_explicit(&GEN, 1, memory_order_release);
    work_a(0);
    atomic_fetch_add(&DONE_A, 1);
    spin_until(&DONE_A, T);
    work_b(0);
    atomic_fetch_add(&DONE_B, 1);
    spin_until(&DONE_B, T);
    return 0;
}

void moe_set_wait(long spin_us, long nap_us) {
    SPIN_NS = spin_us * 1000;
    NAP_NS = nap_us * 1000;
}

// ---------------------------------------------------------------------------
// Batched path for prompt chunks: many tokens at once.
//
// Tokens are grouped by expert, so each expert's weights are read from RAM
// once per chunk and reused (from the CPU cache) for every token routed to
// it. Per expert, three parallel phases:
//   1. gate/up rows split across threads  -> H = silu(gate x) * up x
//   2. H quantized to int8, tokens split across threads
//   3. down rows split across threads      -> out[token] += score * down H
// Each thread owns distinct output rows, so no two threads write the same
// float, and experts run one after another.

static void par(void (*fn)(int)) {  // run fn(t) on every thread, wait for all
    JFN = fn;
    MODE = 1;
    atomic_store(&DONE_G, 0);
    atomic_fetch_add_explicit(&GEN, 1, memory_order_release);
    fn(0);
    atomic_fetch_add(&DONE_G, 1);
    spin_until(&DONE_G, T);
}

static struct {
    int L, n;              // tokens in chunk, tokens for the current expert
    const uint8_t *row;    // current expert's packed row
    const float *x;
    float *out;
    int *toks;             // tokens routed to the current expert
    float *scs;            // their router scores
} B;
static int CAP;
static int8_t *BXQ, *BHQ;
static float *BXD, *BXS, *BH, *BHD, *BHS, *SCALL;
static int *TOKALL, *START;

static inline void span(int n, int t, int *lo, int *hi) {
    *lo = (long)n * t / T;
    *hi = (long)n * (t + 1) / T;
}

static void b_quant_x(int t) {
    int lo, hi;
    span(B.L, t, &lo, &hi);
    for (int i = lo; i < hi; i++) {
        quant_x(B.x + (long)i * D, D, BXQ + (long)i * D, BXD + (long)i * (D / 64), BXS + (long)i * (D / 64));
        memset(B.out + (long)i * D, 0, sizeof(float) * D);
    }
}

static void b_gateup(int t) {
    int lo, hi;
    span(F, t, &lo, &hi);
    const uint8_t *row = B.row;
    for (int j = 0; j < B.n; j++) {
        int tok = B.toks[j];
        const int8_t *xq = BXQ + (long)tok * D;
        const float *xd = BXD + (long)tok * (D / 64), *xs = BXS + (long)tok * (D / 64);
        for (int r = lo; r < hi; r++) {
            float g = qdot(row + OFF[0] + (long)r * (D / 2), (const uint16_t *)(row + OFF[1]) + (long)r * (D / 64),
                           (const uint16_t *)(row + OFF[2]) + (long)r * (D / 64), xq, xd, xs, D);
            float u = qdot(row + OFF[3] + (long)r * (D / 2), (const uint16_t *)(row + OFF[4]) + (long)r * (D / 64),
                           (const uint16_t *)(row + OFF[5]) + (long)r * (D / 64), xq, xd, xs, D);
            BH[(long)j * F + r] = g / (1.f + expf(-g)) * u;
        }
    }
}

static void b_quant_h(int t) {
    int lo, hi;
    span(B.n, t, &lo, &hi);
    for (int j = lo; j < hi; j++)
        quant_x(BH + (long)j * F, F, BHQ + (long)j * F, BHD + (long)j * (F / 64), BHS + (long)j * (F / 64));
}

static void b_down(int t) {
    int lo, hi;
    span(D, t, &lo, &hi);
    const uint8_t *row = B.row;
    for (int j = 0; j < B.n; j++) {
        const int8_t *hq = BHQ + (long)j * F;
        const float *hd = BHD + (long)j * (F / 64), *hs = BHS + (long)j * (F / 64);
        float *o_ = B.out + (long)B.toks[j] * D, sc = B.scs[j];
        for (int o = lo; o < hi; o++)
            o_[o] += sc * qdot(row + OFF[6] + (long)o * (F / 2), (const uint16_t *)(row + OFF[7]) + (long)o * (F / 64),
                               (const uint16_t *)(row + OFF[8]) + (long)o * (F / 64), hq, hd, hs, F);
    }
}

// L tokens, k experts each (idx/score are L x k), E experts in the layer.
// x and out are L x D floats.
int moe_batch(const uint8_t *base, long row_bytes, int E, int L, int k, const int *idx, const float *score,
              const float *x, float *out) {
    if (L > CAP) {
        CAP = L;
        BXQ = realloc(BXQ, (long)L * D);
        BXD = realloc(BXD, sizeof(float) * L * (D / 64));
        BXS = realloc(BXS, sizeof(float) * L * (D / 64));
        BH = realloc(BH, sizeof(float) * L * F);
        BHQ = realloc(BHQ, (long)L * F);
        BHD = realloc(BHD, sizeof(float) * L * (F / 64));
        BHS = realloc(BHS, sizeof(float) * L * (F / 64));
        TOKALL = realloc(TOKALL, sizeof(int) * L * k);
        SCALL = realloc(SCALL, sizeof(float) * L * k);
        if (!BXQ || !BXD || !BXS || !BH || !BHQ || !BHD || !BHS || !TOKALL || !SCALL) return -2;
    }
    START = realloc(START, sizeof(int) * (E + 1));
    // Counting sort of (token, score) pairs by expert.
    memset(START, 0, sizeof(int) * (E + 1));
    for (int i = 0; i < L * k; i++) {
        if (idx[i] < 0 || idx[i] >= E) return -3;
        START[idx[i] + 1]++;
    }
    for (int e = 0; e < E; e++) START[e + 1] += START[e];
    int *fill = malloc(sizeof(int) * E);
    memcpy(fill, START, sizeof(int) * E);
    for (int i = 0; i < L * k; i++) {
        int p = fill[idx[i]]++;
        TOKALL[p] = i / k;
        SCALL[p] = score[i];
    }
    free(fill);

    B.L = L;
    B.x = x;
    B.out = out;
    par(b_quant_x);
    for (int e = 0; e < E; e++) {
        int n = START[e + 1] - START[e];
        if (!n) continue;
        B.n = n;
        B.row = base + (long)e * row_bytes;
        B.toks = TOKALL + START[e];
        B.scs = SCALL + START[e];
        par(b_gateup);
        par(b_quant_h);
        par(b_down);
    }
    return 0;
}
