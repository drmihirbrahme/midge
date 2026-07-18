/* mten.h — midge container format (v1).
 *
 * A converted model directory contains:
 *   spec.json      — human-readable model spec (also embedded in dense.midge)
 *   dense.midge     — all resident (non-expert) tensors, JSON-indexed
 *   experts.midge   — routed expert weights at deterministic offsets (no index)
 *   tokenizer.json — copied from the source checkpoint (used by the Python CLI)
 *
 * File layout (both .midge files):
 *   [u64 LE header_len][header JSON][pad to 64][data...]
 *
 * dense.midge header: { "midge":1, "kind":"dense", "spec":{...},
 *   "tensors": { name: {"dt":"f32|q8r|q4g32|mxfp4", "shape":[r,c] or [n],
 *                        "off":..., "soff":...} } }
 *   Offsets are relative to the 64-aligned data start.
 *
 * experts.midge header: { "midge":1, "kind":"experts", "dt":"..." }
 *   Expert blob offsets are NOT stored; they are recomputed by
 *   wt_expert_layout(), which must match tools/midgepack.py exactly:
 *   for layer L in 0..n_layers: for expert E in 0..n_experts:
 *     for M in [gate, up, down]: append(data), append(scales), append(bias)
 *   each blob 64-byte aligned; expert stride = 64-aligned end.
 *
 * Quantized encodings:
 *   f32    — float32 row-major, no scales
 *   q8r    — int8 row-major, f32 scale per row; w = q * s
 *   q4g32  — packed uint8 (2 nibbles, low = even col), groups of 32 cols,
 *            f16 scale per group; w = (nibble - 8) * s
 *   mxfp4  — packed uint8 nibbles = FP4 E2M1 code, groups of 32 cols,
 *            f16 scale per group; w = LUT[nibble] * s
 */
#ifndef WTEN_H
#define WTEN_H
#include <stdint.h>
#include <stddef.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "mjson.h"

enum { DT_F32 = 0, DT_Q8R = 1, DT_Q4G32 = 2, DT_MXFP4 = 3 };

static const float WT_FP4_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

static inline float wt_f16(uint16_t h) {
    uint32_t s = (uint32_t)(h >> 15) & 1u, e = (h >> 10) & 31u, m = h & 1023u, b;
    if (e == 0) {
        if (m == 0) b = s << 31;
        else { e = 113; while (!(m & 1024u)) { m <<= 1; e--; } m &= 1023u; b = (s << 31) | (e << 23) | (m << 13); }
    } else if (e == 31) b = (s << 31) | 0x7f800000u | (m << 13);
    else b = (s << 31) | ((e + 112u) << 23) | (m << 13);
    float f; memcpy(&f, &b, 4); return f;
}

static inline int wt_dt(const char *s) {
    if (!strcmp(s, "f32")) return DT_F32;
    if (!strcmp(s, "q8r")) return DT_Q8R;
    if (!strcmp(s, "q4g32")) return DT_Q4G32;
    if (!strcmp(s, "mxfp4")) return DT_MXFP4;
    return -1;
}

typedef struct {
    int dt;
    int64_t rows, cols;      /* vectors: rows = n, cols = 0 */
    const void *data;
    const void *scales;      /* NULL for f32 */
} WT;

typedef struct {
    int fd;
    uint8_t *map;
    size_t size;
    uint8_t *base;           /* 64-aligned data start */
    WJ *hdr;
} WFile;

static inline size_t wt_align64(size_t x) { return (x + 63u) & ~(size_t)63u; }

static int wf_open(WFile *f, const char *path) {
    memset(f, 0, sizeof(*f));
    f->fd = open(path, O_RDONLY);
    if (f->fd < 0) return -1;
    struct stat st;
    if (fstat(f->fd, &st) != 0) return -1;
    f->size = (size_t)st.st_size;
    f->map = (uint8_t *)mmap(NULL, f->size, PROT_READ, MAP_SHARED, f->fd, 0);
    if (f->map == MAP_FAILED) return -1;
    uint64_t hl; memcpy(&hl, f->map, 8);
    f->hdr = wj_parse((const char *)f->map + 8, (size_t)hl);
    if (!f->hdr) return -1;
    f->base = f->map + wt_align64(8 + (size_t)hl);
    return 0;
}

/* blob byte sizes for a [rows x cols] matrix in a given dtype */
static void wt_sizes(int dt, int64_t rows, int64_t cols, size_t *dbytes, size_t *sbytes) {
    switch (dt) {
        case DT_F32:   *dbytes = (size_t)rows * cols * 4; *sbytes = 0; break;
        case DT_Q8R:   *dbytes = (size_t)rows * cols;     *sbytes = (size_t)rows * 4; break;
        case DT_Q4G32:
        case DT_MXFP4: *dbytes = (size_t)rows * cols / 2; *sbytes = (size_t)rows * (cols / 32) * 2; break;
        default:       *dbytes = 0; *sbytes = 0;
    }
}

/* look up a dense tensor by name; returns 0 on success */
static int wt_dense(WFile *f, const char *name, WT *t) {
    WJ *tens = wj_get(f->hdr, "tensors");
    WJ *e = wj_get(tens, name);
    if (!e) return -1;
    t->dt = wt_dt(wj_strd(e, "dt", "f32"));
    WJ *sh = wj_get(e, "shape");
    t->rows = (int64_t)wj_at(sh, 0)->num;
    t->cols = sh->n > 1 ? (int64_t)wj_at(sh, 1)->num : 0;
    t->data = f->base + (size_t)wj_numd(e, "off", 0);
    double soff = wj_numd(e, "soff", -1);
    t->scales = soff >= 0 ? f->base + (size_t)soff : NULL;
    return 0;
}

/* ---- deterministic expert layout (must match tools/midgepack.py) ---- */
typedef struct {
    int dt;
    int64_t hidden, ffn;
    /* per-expert relative offsets: [matrix 0=gate,1=up,2=down][part 0=data,1=scales,2=bias] */
    size_t off[3][3];
    size_t expert_stride;    /* 64-aligned per-expert size */
    size_t layer_stride;
} WExLayout;

static void wt_expert_layout(WExLayout *L, int dt, int64_t hidden, int64_t ffn, int64_t n_experts) {
    L->dt = dt; L->hidden = hidden; L->ffn = ffn;
    int64_t rows[3] = { ffn, ffn, hidden };   /* gate, up, down output dims */
    int64_t cols[3] = { hidden, hidden, ffn };
    size_t o = 0;
    for (int m = 0; m < 3; m++) {
        size_t db, sb;
        wt_sizes(dt, rows[m], cols[m], &db, &sb);
        L->off[m][0] = o; o = wt_align64(o + db);
        L->off[m][1] = o; o = wt_align64(o + sb);
        L->off[m][2] = o; o = wt_align64(o + (size_t)rows[m] * 4);  /* f32 bias */
    }
    L->expert_stride = wt_align64(o);
    L->layer_stride = L->expert_stride * (size_t)n_experts;
}

/* materialize WT views for one expert (m: 0=gate 1=up 2=down). bias_out gets f32*. */
static void wt_expert(WFile *f, const WExLayout *L, int layer, int expert, int m,
                      WT *t, const float **bias_out) {
    uint8_t *e = f->base + (size_t)layer * L->layer_stride + (size_t)expert * L->expert_stride;
    int64_t rows = (m == 2) ? L->hidden : L->ffn;
    int64_t cols = (m == 2) ? L->ffn : L->hidden;
    t->dt = L->dt; t->rows = rows; t->cols = cols;
    t->data = e + L->off[m][0];
    t->scales = (L->dt == DT_F32) ? NULL : (e + L->off[m][1]);
    *bias_out = (const float *)(e + L->off[m][2]);
}

static void wf_close(WFile *f) {
    if (f->hdr) { wj_free(f->hdr); f->hdr = NULL; }
    if (f->map && f->map != MAP_FAILED) { munmap(f->map, f->size); f->map = NULL; }
    if (f->fd >= 0) { close(f->fd); f->fd = -1; }
}

#endif /* WTEN_H */

