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

static void wk_matvec(const WT *t, const float *x, float *y) {
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
