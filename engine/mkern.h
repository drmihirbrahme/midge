/* mkern.h — CPU kernels. Scalar C + OpenMP; dequant-on-use from mmap'd weights.
 * All accumulation in f32. Correctness-first; SIMD tuning is a roadmap item
 * (build with ARCH=native to let the compiler auto-vectorize the inner loops).
 */
#ifndef WKERN_H
#define WKERN_H
#include <math.h>
#include "mten.h"

/* y[r] = sum_c W[r,c] * x[c]   (+ optional bias), dispatched on dtype.
 * `acc` != 0 means y += result (used for weighted expert accumulation is done
 * by caller instead; acc kept simple: 0 = overwrite). */
static void mv_f32(const float *W, const float *x, float *y, int64_t rows, int64_t cols) {
    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; r++) {
        const float *w = W + r * cols;
        float a = 0.f;
        for (int64_t c = 0; c < cols; c++) a += w[c] * x[c];
        y[r] = a;
    }
}

static void mv_q8r(const int8_t *W, const float *S, const float *x, float *y, int64_t rows, int64_t cols) {
    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; r++) {
        const int8_t *w = W + r * cols;
        float a = 0.f;
        for (int64_t c = 0; c < cols; c++) a += (float)w[c] * x[c];
        y[r] = a * S[r];
    }
}

static void mv_g32(const uint8_t *W, const uint16_t *S, const float *x, float *y,
                   int64_t rows, int64_t cols, int is_fp4) {
    int64_t gpr = cols / 32;
    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; r++) {
        const uint8_t *w = W + r * (cols / 2);
        const uint16_t *s = S + r * gpr;
        float acc = 0.f;
        for (int64_t g = 0; g < gpr; g++) {
            const uint8_t *wb = w + g * 16;
            const float *xg = x + g * 32;
            float a = 0.f;
            if (is_fp4) {
                for (int j = 0; j < 16; j++) {
                    uint8_t b = wb[j];
                    a += WT_FP4_LUT[b & 15] * xg[2 * j];
                    a += WT_FP4_LUT[b >> 4] * xg[2 * j + 1];
                }
            } else {
                for (int j = 0; j < 16; j++) {
                    uint8_t b = wb[j];
                    a += (float)((int)(b & 15) - 8) * xg[2 * j];
                    a += (float)((int)(b >> 4) - 8) * xg[2 * j + 1];
                }
            }
            acc += wt_f16(s[g]) * a;
        }
        y[r] = acc;
    }
}


/* ---------------------------------------------------------------------
 * AVX2+FMA kernels (runtime-dispatched; scalar above remains the
 * portable fallback). Strategy: de-interleave x once per matvec into
 * even/odd streams so packed nibbles need no per-group shuffling, then
 * widen -> cvt -> FMA. FP4 decode via a signed pshufb LUT of doubled
 * magnitudes, folding the 0.5 into the group scale.
 * ------------------------------------------------------------------- */
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>

__attribute__((target("avx2,fma")))
static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    return _mm_cvtss_f32(lo);
}

__attribute__((target("avx2,fma")))
static void mv_f32_avx2(const float *W, const float *x, float *y,
                        int64_t rows, int64_t cols) {
    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; r++) {
        const float *w = W + r * cols;
        __m256 acc = _mm256_setzero_ps();
        int64_t c = 0;
        for (; c + 8 <= cols; c += 8)
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(w + c),
                                  _mm256_loadu_ps(x + c), acc);
        float a = hsum256(acc);
        for (; c < cols; c++) a += w[c] * x[c];
        y[r] = a;
    }
}

__attribute__((target("avx2,fma")))
static void mv_q8r_avx2(const int8_t *W, const float *S, const float *x,
                        float *y, int64_t rows, int64_t cols) {
    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; r++) {
        const int8_t *w = W + r * cols;
        __m256 acc = _mm256_setzero_ps();
        int64_t c = 0;
        for (; c + 8 <= cols; c += 8) {
            __m128i b = _mm_loadl_epi64((const __m128i *)(w + c));
            __m256 wf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b));
            acc = _mm256_fmadd_ps(wf, _mm256_loadu_ps(x + c), acc);
        }
        float a = hsum256(acc);
        for (; c < cols; c++) a += (float)w[c] * x[c];
        y[r] = a * S[r];
    }
}

__attribute__((target("avx2,fma")))
static void mv_g32_avx2(const uint8_t *W, const uint16_t *S, const float *x,
                        float *y, int64_t rows, int64_t cols, int is_fp4) {
    int64_t gpr = cols / 32, half = cols / 2;
    /* x de-interleaved: xe[k] = x[2k], xo[k] = x[2k+1] */
    float xe[half], xo[half];
    for (int64_t k = 0; k < half; k++) { xe[k] = x[2 * k]; xo[k] = x[2 * k + 1]; }
    /* FP4: signed doubled magnitudes; scale is folded as 0.5*scale */
    const __m128i FP4X2 = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12,
                                        0, -1, -2, -3, -4, -6, -8, -12);
    const __m128i LOW = _mm_set1_epi8(0x0F);
    const __m128i EIGHT = _mm_set1_epi8(8);
    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; r++) {
        const uint8_t *w = W + r * half;
        const uint16_t *s = S + r * gpr;
        __m256 acc = _mm256_setzero_ps();
        for (int64_t g = 0; g < gpr; g++) {
            __m128i b = _mm_loadu_si128((const __m128i *)(w + g * 16));
            __m128i lo = _mm_and_si128(b, LOW);
            __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), LOW);
            if (is_fp4) {
                lo = _mm_shuffle_epi8(FP4X2, lo);   /* signed 2*value */
                hi = _mm_shuffle_epi8(FP4X2, hi);
            } else {
                lo = _mm_sub_epi8(lo, EIGHT);
                hi = _mm_sub_epi8(hi, EIGHT);
            }
            const float *pe = xe + g * 16, *po = xo + g * 16;
            __m256 g0 = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo)),
                _mm256_loadu_ps(pe));
            g0 = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8))),
                _mm256_loadu_ps(pe + 8), g0);
            g0 = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi)),
                _mm256_loadu_ps(po), g0);
            g0 = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8))),
                _mm256_loadu_ps(po + 8), g0);
            float sc = wt_f16(s[g]) * (is_fp4 ? 0.5f : 1.0f);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(sc), g0, acc);
        }
        y[r] = hsum256(acc);
    }
}

#include <stdlib.h>
static int wk_have_avx2(void) {
    static int have = -1;
    if (have < 0)
        have = !getenv("MIDGE_NO_SIMD")
            && __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    return have;
}
#else
static int wk_have_avx2(void) { return 0; }
#endif  /* x86 */

static void wk_matvec(const WT *t, const float *x, float *y) {
#if defined(__x86_64__) || defined(__i386__)
    if (wk_have_avx2()) {
        switch (t->dt) {
            case DT_F32:   mv_f32_avx2((const float *)t->data, x, y, t->rows, t->cols); return;
            case DT_Q8R:   mv_q8r_avx2((const int8_t *)t->data, (const float *)t->scales, x, y, t->rows, t->cols); return;
            case DT_Q4G32: mv_g32_avx2((const uint8_t *)t->data, (const uint16_t *)t->scales, x, y, t->rows, t->cols, 0); return;
            case DT_MXFP4: mv_g32_avx2((const uint8_t *)t->data, (const uint16_t *)t->scales, x, y, t->rows, t->cols, 1); return;
        }
    }
#endif
    (void)wk_have_avx2;
    switch (t->dt) {
        case DT_F32:   mv_f32((const float *)t->data, x, y, t->rows, t->cols); break;
        case DT_Q8R:   mv_q8r((const int8_t *)t->data, (const float *)t->scales, x, y, t->rows, t->cols); break;
        case DT_Q4G32: mv_g32((const uint8_t *)t->data, (const uint16_t *)t->scales, x, y, t->rows, t->cols, 0); break;
        case DT_MXFP4: mv_g32((const uint8_t *)t->data, (const uint16_t *)t->scales, x, y, t->rows, t->cols, 1); break;
    }
}

/* dequantize one row of a matrix into out[cols] (embedding lookup) */
static void wk_row(const WT *t, int64_t r, float *out) {
    int64_t cols = t->cols;
    switch (t->dt) {
        case DT_F32: {
            const float *w = (const float *)t->data + r * cols;
            memcpy(out, w, (size_t)cols * 4);
            break;
        }
        case DT_Q8R: {
            const int8_t *w = (const int8_t *)t->data + r * cols;
            float s = ((const float *)t->scales)[r];
            for (int64_t c = 0; c < cols; c++) out[c] = (float)w[c] * s;
            break;
        }
        case DT_Q4G32:
        case DT_MXFP4: {
            const uint8_t *w = (const uint8_t *)t->data + r * (cols / 2);
            const uint16_t *s = (const uint16_t *)t->scales + r * (cols / 32);
            int fp4 = (t->dt == DT_MXFP4);
            for (int64_t g = 0; g < cols / 32; g++) {
                float sc = wt_f16(s[g]);
                for (int j = 0; j < 16; j++) {
                    uint8_t b = w[g * 16 + j];
                    if (fp4) {
                        out[g * 32 + 2 * j]     = WT_FP4_LUT[b & 15] * sc;
                        out[g * 32 + 2 * j + 1] = WT_FP4_LUT[b >> 4] * sc;
                    } else {
                        out[g * 32 + 2 * j]     = (float)((int)(b & 15) - 8) * sc;
                        out[g * 32 + 2 * j + 1] = (float)((int)(b >> 4) - 8) * sc;
                    }
                }
            }
            break;
        }
    }
}

static void wk_rmsnorm(const float *x, const float *w, float *out, int64_t n, float eps) {
    float ss = 0.f;
    for (int64_t i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / (float)n + eps);
    for (int64_t i = 0; i < n; i++) out[i] = x[i] * inv * w[i];
}

static void wk_addbias(float *y, const float *b, int64_t n) {
    for (int64_t i = 0; i < n; i++) y[i] += b[i];
}

static inline float wk_dot(const float *a, const float *b, int n) {
    float s = 0.f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

static inline float wk_dot_f16(const float *a, const uint16_t *b, int n) {
    float s = 0.f;
    for (int i = 0; i < n; i++) s += a[i] * wt_f16(b[i]);
    return s;
}

static inline uint16_t wk_to_f16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t man = x & 0x7fffffu;
    if (exp <= 0) return (uint16_t)sign;                 /* flush denormals */
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);    /* inf */
    /* round to nearest even */
    uint32_t m = man >> 13;
    uint32_t rem = man & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (m & 1))) { m++; if (m == 0x400u) { m = 0; exp++; if (exp >= 31) return (uint16_t)(sign | 0x7c00u); } }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | m);
}

#endif /* WKERN_H */
