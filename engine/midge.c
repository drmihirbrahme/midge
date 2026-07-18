/* midge.c — tiny engine, immense models.
 *
 * Spec-driven CPU inference for MoE transformers whose routed experts are
 * streamed from disk via mmap (the OS page cache is the expert cache), while
 * the dense trunk stays hot. First-class target: openai/gpt-oss-20b/120b.
 *
 * Design notes:
 *  - decode is strictly S=1 (prefill feeds tokens one at a time): simple,
 *    correct, and the disk — not the batch — is the bottleneck on the target
 *    hardware anyway. Batched prefill is a roadmap item.
 *  - all math in f32; KV cache stored f16; weights dequantized on use.
 *  - the engine speaks token IDs only. Tokenization, chat templating and the
 *    OpenAI-style CLI live in the Python shell (./midge).
 *
 * Protocol (stdin/stdout, line-based):
 *   stdin:  "ids: 1 2 3 ..."   append tokens to the context (prefill)
 *           "gen: N"           generate up to N tokens from current context
 *           "tf: 1 2 3"        teacher-force; print last-position logits
 *           "tfall: 1 2 3"     teacher-force; print logits at every position
 *           "set: T P [seed]"  retune sampling (temp/top-p) keeping context
 *   stdout: "T <id>"           one generated token
 *           "L <v0> <v1> ..."  logits row
 *           "DONE <prompt_toks> <gen_toks> <seconds> <expert_loads>"
 *   stderr: "# ..." diagnostics
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#else
static int omp_get_max_threads(void) { return 1; }
#endif
#include <errno.h>
#include "mjson.h"
#include "mten.h"
#include "mkern.h"

#define BATCH_MAX 64
#define MAX_TOPK 16

typedef struct {
    int64_t hidden, vocab, ffn;
    int n_layers, n_heads, n_kv_heads, head_dim;
    int n_experts, top_k;
    float alpha, limit;          /* clamped-swiglu params */
    int act_plain;               /* 1 = plain SwiGLU (silu(g)*u), 0 = gpt-oss clamp */
    int router_norm;             /* 1 = softmax over selected; 0 = full-softmax weights */
    int qk_norm;                 /* per-head RMSNorm on q,k before RoPE */
    int sliding_window;
    int sinks;
    float norm_eps;
    float rope_theta;
    int yarn;                    /* 0 = plain rope */
    float yarn_factor, yarn_beta_fast, yarn_beta_slow;
    int yarn_orig_ctx, yarn_truncate;
    float attn_scale;
    int *layer_sliding;          /* per layer: 1 = sliding attention */
} Spec;

typedef struct {
    Spec s;
    WFile dense, experts;
    WExLayout exl;
    /* dense tensors */
    WT embed, lm_head;
    const float *final_norm;
    WT *wq, *wk, *wv, *wo, *router_w;          /* per layer */
    const float **bq, **bk, **bv, **bo, **router_b;
    const float **attn_norm, **mlp_norm, **sink;
    const float **qn, **kn;                     /* qk-norm weights [head_dim] */
    float *bx, *bxb, *bw;                       /* batched-prefill scratch */
    int *bsel;
    /* runtime */
    int ctx;
    uint16_t **kc, **vc;         /* per layer f16 KV; sliding layers use ring of window slots */
    int pos;                     /* tokens consumed so far */
    float *x, *xb, *q, *k, *v, *ao, *att, *hb, *hb2, *moe, *rl, *logits, *probs;
    float *rope_cos, *rope_sin;  /* [ctx][head_dim/2] */
    uint32_t *usage;             /* [n_layers][n_experts] */
    uint64_t expert_loads;
    char dir[1024];
} Model;

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ------------------------------------------------------------------ spec */
static void spec_load(Spec *s, WJ *j) {
    memset(s, 0, sizeof(*s));
    s->hidden = (int64_t)wj_numd(j, "hidden", 0);
    s->vocab = (int64_t)wj_numd(j, "vocab", 0);
    s->n_layers = (int)wj_numd(j, "n_layers", 0);
    s->n_heads = (int)wj_numd(j, "n_heads", 0);
    s->n_kv_heads = (int)wj_numd(j, "n_kv_heads", 0);
    s->head_dim = (int)wj_numd(j, "head_dim", 64);
    s->norm_eps = (float)wj_numd(j, "norm_eps", 1e-5);
    WJ *moe = wj_get(j, "moe");
    s->ffn = (int64_t)wj_numd(moe, "ffn", 0);
    s->n_experts = (int)wj_numd(moe, "experts", 0);
    s->top_k = (int)wj_numd(moe, "top_k", 0);
    s->alpha = (float)wj_numd(moe, "alpha", 1.702);
    s->limit = (float)wj_numd(moe, "limit", 7.0);
    const char *act = wj_strd(moe, "act", "swiglu_clamp");
    s->act_plain = act && !strcmp(act, "swiglu");
    s->router_norm = (int)wj_numd(moe, "router_norm", 1);
    WJ *attn = wj_get(j, "attn");
    s->sliding_window = (int)wj_numd(attn, "sliding_window", 0);
    s->sinks = (int)wj_numd(attn, "sinks", 0);
    s->qk_norm = (int)wj_numd(attn, "qk_norm", 0);
    s->attn_scale = (float)wj_numd(attn, "scale", 1.0 / sqrt((double)s->head_dim));
    s->layer_sliding = (int *)calloc(s->n_layers, sizeof(int));
    WJ *lt = wj_get(attn, "layer_types");
    for (int i = 0; i < s->n_layers; i++) {
        WJ *e = wj_at(lt, i);
        s->layer_sliding[i] = e && e->str && strstr(e->str, "sliding") != NULL;
    }
    WJ *rope = wj_get(j, "rope");
    s->rope_theta = (float)wj_numd(rope, "theta", 10000.0);
    WJ *y = wj_get(rope, "yarn");
    if (y && y->type == 'o') {
        s->yarn = 1;
        s->yarn_factor = (float)wj_numd(y, "factor", 1.0);
        s->yarn_beta_fast = (float)wj_numd(y, "beta_fast", 32.0);
        s->yarn_beta_slow = (float)wj_numd(y, "beta_slow", 1.0);
        s->yarn_orig_ctx = (int)wj_numd(y, "orig_ctx", 4096);
        s->yarn_truncate = (int)wj_numd(y, "truncate", 0);
    }
    if (!s->hidden || !s->vocab || !s->n_layers || !s->ffn || !s->n_experts || !s->top_k) {
        fprintf(stderr, "# invalid spec\n"); exit(3);
    }
}

/* YaRN rope tables — must match tools/reference.py rope() exactly */
static void rope_init(Model *m) {
    Spec *s = &m->s;
    int hd = s->head_dim, half = hd / 2;
    m->rope_cos = (float *)malloc((size_t)m->ctx * half * 4);
    m->rope_sin = (float *)malloc((size_t)m->ctx * half * 4);
    double *inv = (double *)malloc(half * sizeof(double));
    double mscale = 1.0;
    for (int i = 0; i < half; i++)
        inv[i] = 1.0 / pow((double)s->rope_theta, (double)(2 * i) / hd);
    if (s->yarn) {
        double f = s->yarn_factor, base = s->rope_theta, orig = s->yarn_orig_ctx;
        double low = hd * log(orig / (s->yarn_beta_fast * 2.0 * M_PI)) / (2.0 * log(base));
        double high = hd * log(orig / (s->yarn_beta_slow * 2.0 * M_PI)) / (2.0 * log(base));
        if (s->yarn_truncate) { low = floor(low); high = ceil(high); }
        if (low < 0) low = 0;
        if (high > hd - 1) high = hd - 1;
        if (high == low) high = low + 0.001;
        for (int i = 0; i < half; i++) {
            double extrap = inv[i];
            double interp = inv[i] / f;
            double ramp = ((double)i - low) / (high - low);
            if (ramp < 0) ramp = 0;
            if (ramp > 1) ramp = 1;
            double mask = 1.0 - ramp;              /* 1 = keep extrapolation (high freq) */
            inv[i] = interp * (1.0 - mask) + extrap * mask;
        }
        mscale = 0.1 * log(f) + 1.0;
    }
    for (int p = 0; p < m->ctx; p++)
        for (int i = 0; i < half; i++) {
            double a = (double)p * inv[i];
            m->rope_cos[(size_t)p * half + i] = (float)(cos(a) * mscale);
            m->rope_sin[(size_t)p * half + i] = (float)(sin(a) * mscale);
        }
    free(inv);
}

static void rope_apply(Model *m, float *vec, int n_heads_v, int pos) {
    int hd = m->s.head_dim, half = hd / 2;
    const float *co = m->rope_cos + (size_t)pos * half;
    const float *si = m->rope_sin + (size_t)pos * half;
    for (int h = 0; h < n_heads_v; h++) {
        float *v = vec + (size_t)h * hd;
        for (int i = 0; i < half; i++) {
            float a = v[i], b = v[i + half];
            v[i] = a * co[i] - b * si[i];
            v[i + half] = b * co[i] + a * si[i];
        }
    }
}

/* ------------------------------------------------------------- load model */
static const float *need_vec(WFile *f, const char *name) {
    WT t;
    if (wt_dense(f, name, &t) != 0) { fprintf(stderr, "# missing tensor %s\n", name); exit(3); }
    if (t.dt != DT_F32) { fprintf(stderr, "# tensor %s must be f32\n", name); exit(3); }
    return (const float *)t.data;
}

static void need_mat(WFile *f, const char *name, WT *t) {
    if (wt_dense(f, name, t) != 0) { fprintf(stderr, "# missing tensor %s\n", name); exit(3); }
}

static void model_load(Model *m, const char *dir, int ctx) {
    snprintf(m->dir, sizeof(m->dir), "%s", dir);
    char p[1200];
    snprintf(p, sizeof(p), "%s/dense.midge", dir);
    if (wf_open(&m->dense, p) != 0) { fprintf(stderr, "# cannot open %s: %s\n", p, strerror(errno)); exit(2); }
    snprintf(p, sizeof(p), "%s/experts.midge", dir);
    if (wf_open(&m->experts, p) != 0) { fprintf(stderr, "# cannot open %s: %s\n", p, strerror(errno)); exit(2); }
    madvise(m->experts.map, m->experts.size, MADV_RANDOM);

    WJ *spec = wj_get(m->dense.hdr, "spec");
    if (!spec) { fprintf(stderr, "# dense.midge has no embedded spec\n"); exit(3); }
    spec_load(&m->s, spec);
    Spec *s = &m->s;

    const char *edt = wj_strd(m->experts.hdr, "dt", "?");
    int dt = wt_dt(edt);
    if (dt < 0) { fprintf(stderr, "# bad experts dtype %s\n", edt); exit(3); }
    wt_expert_layout(&m->exl, dt, s->hidden, s->ffn, s->n_experts);
    size_t total = (size_t)s->n_layers * m->exl.layer_stride;
    if (m->experts.base + total > m->experts.map + m->experts.size) {
        fprintf(stderr, "# experts.midge too small: need %zu data bytes\n", total); exit(3);
    }

    need_mat(&m->dense, "embed", &m->embed);
    need_mat(&m->dense, "lm_head", &m->lm_head);
    m->final_norm = need_vec(&m->dense, "final_norm");

    int L = s->n_layers;
    m->wq = calloc(L, sizeof(WT)); m->wk = calloc(L, sizeof(WT));
    m->wv = calloc(L, sizeof(WT)); m->wo = calloc(L, sizeof(WT));
    m->router_w = calloc(L, sizeof(WT));
    m->bq = calloc(L, sizeof(void *)); m->bk = calloc(L, sizeof(void *));
    m->bv = calloc(L, sizeof(void *)); m->bo = calloc(L, sizeof(void *));
    m->router_b = calloc(L, sizeof(void *));
    m->qn = calloc(L, sizeof(void *));
    m->kn = calloc(L, sizeof(void *));
    m->bx = malloc((size_t)BATCH_MAX * s->hidden * 4);
    m->bxb = malloc((size_t)BATCH_MAX * s->hidden * 4);
    m->bw = malloc((size_t)BATCH_MAX * MAX_TOPK * 4);
    m->bsel = malloc((size_t)BATCH_MAX * MAX_TOPK * sizeof(int));
    m->attn_norm = calloc(L, sizeof(void *)); m->mlp_norm = calloc(L, sizeof(void *));
    m->sink = calloc(L, sizeof(void *));
    for (int i = 0; i < L; i++) {
        char n[64];
        #define GETM(field, fmt) snprintf(n, sizeof(n), fmt, i); need_mat(&m->dense, n, &m->field[i]);
        #define GETV(field, fmt) snprintf(n, sizeof(n), fmt, i); m->field[i] = need_vec(&m->dense, n);
        GETM(wq, "L%d.attn.q"); GETM(wk, "L%d.attn.k"); GETM(wv, "L%d.attn.v"); GETM(wo, "L%d.attn.o");
        GETV(bq, "L%d.attn.q_b"); GETV(bk, "L%d.attn.k_b"); GETV(bv, "L%d.attn.v_b"); GETV(bo, "L%d.attn.o_b");
        GETV(attn_norm, "L%d.attn.norm"); GETV(mlp_norm, "L%d.mlp.norm");
        GETM(router_w, "L%d.router.w"); GETV(router_b, "L%d.router.b");
        if (s->sinks) { GETV(sink, "L%d.attn.sinks"); }
        if (s->qk_norm) { GETV(qn, "L%d.attn.q_norm"); GETV(kn, "L%d.attn.k_norm"); }
        #undef GETM
        #undef GETV
    }

    /* runtime buffers */
    m->ctx = ctx;
    rope_init(m);
    int kv_dim = s->n_kv_heads * s->head_dim;
    int q_dim = s->n_heads * s->head_dim;
    m->kc = calloc(L, sizeof(void *)); m->vc = calloc(L, sizeof(void *));
    for (int i = 0; i < L; i++) {
        int cap = s->layer_sliding[i] && s->sliding_window > 0 && s->sliding_window < ctx
                  ? s->sliding_window : ctx;
        m->kc[i] = malloc((size_t)cap * kv_dim * 2);
        m->vc[i] = malloc((size_t)cap * kv_dim * 2);
    }
    m->x = malloc(s->hidden * 4); m->xb = malloc(s->hidden * 4);
    m->q = malloc(q_dim * 4); m->k = malloc(kv_dim * 4); m->v = malloc(kv_dim * 4);
    m->ao = malloc(q_dim * 4);
    m->att = malloc((size_t)s->n_heads * ctx * 4);
    m->hb = malloc(s->ffn * 4); m->hb2 = malloc(s->ffn * 4);
    m->moe = malloc(s->hidden * 4);
    m->rl = malloc(s->n_experts * 4);
    m->logits = malloc(s->vocab * 4);
    m->probs = malloc(s->vocab * 4);
    m->usage = calloc((size_t)L * s->n_experts, 4);
    m->pos = 0;

    /* accumulate prior usage histogram if present */
    snprintf(p, sizeof(p), "%s/usage.bin", dir);
    FILE *uf = fopen(p, "rb");
    if (uf) {
        if (fread(m->usage, 4, (size_t)L * s->n_experts, uf) != (size_t)L * s->n_experts)
            memset(m->usage, 0, (size_t)L * s->n_experts * 4);
        fclose(uf);
    }
}

static void usage_save(Model *m) {
    char p[1200];
    snprintf(p, sizeof(p), "%s/usage.bin", m->dir);
    FILE *f = fopen(p, "wb");
    if (!f) return;
    fwrite(m->usage, 4, (size_t)m->s.n_layers * m->s.n_experts, f);
    fclose(f);
}

typedef struct { uint32_t c; int i; } CI;
static int ci_cmp(const void *a, const void *b) {
    uint32_t ca = ((const CI *)a)->c, cb = ((const CI *)b)->c;
    return ca < cb ? 1 : ca > cb ? -1 : 0;
}

/* warm the hottest experts into page cache, up to `gb` gigabytes */
static void preload_hot(Model *m, double gb) {
    if (gb <= 0) return;
    Spec *s = &m->s;
    int n = s->n_layers * s->n_experts;
    int *idx = malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) idx[i] = i;
    /* simple selection by count, descending (insertion into budget) */
    uint32_t *u = m->usage;
    /* sort idx by u desc (qsort with global not reentrant; use comparator via wrapper array) */
    CI *ci = malloc(n * sizeof(CI));
    for (int i = 0; i < n; i++) { ci[i].c = u[i]; ci[i].i = i; }
    qsort(ci, n, sizeof(CI), ci_cmp);
    size_t budget = (size_t)(gb * 1e9), used = 0, stride = m->exl.expert_stride;
    volatile uint8_t sink = 0;
    int warmed = 0;
    for (int i = 0; i < n && used + stride <= budget; i++) {
        if (ci[i].c == 0) break;
        int layer = ci[i].i / s->n_experts, e = ci[i].i % s->n_experts;
        uint8_t *base = m->experts.base + (size_t)layer * m->exl.layer_stride + (size_t)e * stride;
        madvise(base, stride, MADV_WILLNEED);
        for (size_t o = 0; o < stride; o += 4096) sink ^= base[o];
        used += stride; warmed++;
    }
    (void)sink;
    fprintf(stderr, "# preloaded %d hot experts (%.2f GB)\n", warmed, used / 1e9);
    free(ci); free(idx);
}

/* ---------------------------------------------------------------- forward */
static void attention(Model *m, int layer, int pos) {
    Spec *s = &m->s;
    int hd = s->head_dim, nh = s->n_heads, nkv = s->n_kv_heads;
    int kv_dim = nkv * hd, gq = nh / nkv;
    int sliding = s->layer_sliding[layer] && s->sliding_window > 0 && s->sliding_window < m->ctx;
    int cap = sliding ? s->sliding_window : m->ctx;

    wk_rmsnorm(m->x, m->attn_norm[layer], m->xb, s->hidden, s->norm_eps);
    wk_matvec(&m->wq[layer], m->xb, m->q); wk_addbias(m->q, m->bq[layer], (int64_t)nh * hd);
    wk_matvec(&m->wk[layer], m->xb, m->k); wk_addbias(m->k, m->bk[layer], kv_dim);
    wk_matvec(&m->wv[layer], m->xb, m->v); wk_addbias(m->v, m->bv[layer], kv_dim);
    if (s->qk_norm) {              /* per-head RMSNorm on q,k before RoPE */
        for (int h = 0; h < nh; h++)
            wk_rmsnorm(m->q + (size_t)h * hd, m->qn[layer], m->q + (size_t)h * hd,
                       hd, s->norm_eps);
        for (int h = 0; h < nkv; h++)
            wk_rmsnorm(m->k + (size_t)h * hd, m->kn[layer], m->k + (size_t)h * hd,
                       hd, s->norm_eps);
    }
    rope_apply(m, m->q, nh, pos);
    rope_apply(m, m->k, nkv, pos);

    int slot = pos % cap;
    uint16_t *kc = m->kc[layer], *vc = m->vc[layer];
    for (int i = 0; i < kv_dim; i++) {
        kc[(size_t)slot * kv_dim + i] = wk_to_f16(m->k[i]);
        vc[(size_t)slot * kv_dim + i] = wk_to_f16(m->v[i]);
    }

    int nctx = pos + 1 < cap ? pos + 1 : cap;   /* entries valid in cache */
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < nh; h++) {
        float *sc = m->att + (size_t)h * m->ctx;  /* per-head scratch */
        const float *qh = m->q + (size_t)h * hd;
        int kh = h / gq;
        float mx = -1e30f;
        for (int t = 0; t < nctx; t++) {
            const uint16_t *kt = kc + (size_t)t * kv_dim + (size_t)kh * hd;
            float d = wk_dot_f16(qh, kt, hd) * s->attn_scale;
            sc[t] = d;
            if (d > mx) mx = d;
        }
        float sinkv = s->sinks ? m->sink[layer][h] : -1e30f;
        if (sinkv > mx) mx = sinkv;
        float denom = s->sinks ? expf(sinkv - mx) : 0.f;
        for (int t = 0; t < nctx; t++) { sc[t] = expf(sc[t] - mx); denom += sc[t]; }
        float inv = 1.0f / denom;
        float *out = m->ao + (size_t)h * hd;
        memset(out, 0, hd * 4);
        for (int t = 0; t < nctx; t++) {
            float w = sc[t] * inv;
            const uint16_t *vt = vc + (size_t)t * kv_dim + (size_t)kh * hd;
            for (int i = 0; i < hd; i++) out[i] += w * wt_f16(vt[i]);
        }
    }
    /* o proj + residual */
    wk_matvec(&m->wo[layer], m->ao, m->xb);
    wk_addbias(m->xb, m->bo[layer], s->hidden);
    for (int64_t i = 0; i < s->hidden; i++) m->x[i] += m->xb[i];
}

/* top-k routing for the (already attention-updated) residual x.
   Writes normalized weights + expert ids; input is the raw residual. */
static void route_topk(Model *m, int layer, const float *x, float *xb,
                       int *sel, float *w) {
    Spec *s = &m->s;
    wk_rmsnorm(x, m->mlp_norm[layer], xb, s->hidden, s->norm_eps);
    wk_matvec(&m->router_w[layer], xb, m->rl);
    wk_addbias(m->rl, m->router_b[layer], s->n_experts);
    int k = s->top_k;
    float val[MAX_TOPK];
    for (int j = 0; j < k; j++) { val[j] = -1e30f; sel[j] = 0; }
    for (int e = 0; e < s->n_experts; e++) {
        float v = m->rl[e];
        for (int j = 0; j < k; j++) {
            if (v > val[j]) {
                for (int q = k - 1; q > j; q--) { val[q] = val[q-1]; sel[q] = sel[q-1]; }
                val[j] = v; sel[j] = e;
                break;
            }
        }
    }
    float mx = val[0], den = 0.f;
    if (s->router_norm) {
        for (int i = 0; i < k; i++) { w[i] = expf(val[i] - mx); den += w[i]; }
        for (int i = 0; i < k; i++) w[i] /= den;
    } else {
        float fmx = m->rl[0];
        for (int e = 1; e < s->n_experts; e++) if (m->rl[e] > fmx) fmx = m->rl[e];
        float fden = 0.f;
        for (int e = 0; e < s->n_experts; e++) fden += expf(m->rl[e] - fmx);
        for (int i = 0; i < k; i++) w[i] = expf(val[i] - fmx) / fden;
    }
}

/* one expert's contribution: xb (normed) -> += weight * down(act(gate,up)) */
static void expert_apply(Model *m, int layer, int e, const float *xb,
                         float weight, float *x_out) {
    Spec *s = &m->s;
    m->usage[(size_t)layer * s->n_experts + e]++;
    m->expert_loads++;
    WT gate, up, down; const float *bg, *bu, *bd;
    wt_expert(&m->experts, &m->exl, layer, e, 0, &gate, &bg);
    wt_expert(&m->experts, &m->exl, layer, e, 1, &up, &bu);
    wt_expert(&m->experts, &m->exl, layer, e, 2, &down, &bd);
    wk_matvec(&gate, xb, m->hb);  wk_addbias(m->hb, bg, s->ffn);
    wk_matvec(&up, xb, m->hb2);   wk_addbias(m->hb2, bu, s->ffn);
    if (s->act_plain) {
        for (int64_t j = 0; j < s->ffn; j++) {
            float g = m->hb[j];
            m->hb[j] = g / (1.0f + expf(-g)) * m->hb2[j];
        }
    } else {
        for (int64_t j = 0; j < s->ffn; j++) {
            float g = m->hb[j], u = m->hb2[j];
            if (g > s->limit) g = s->limit;
            if (u > s->limit) u = s->limit;
            if (u < -s->limit) u = -s->limit;
            float act = g / (1.0f + expf(-s->alpha * g));
            m->hb[j] = act * (u + 1.0f);
        }
    }
    wk_matvec(&down, m->hb, m->moe);
    wk_addbias(m->moe, bd, s->hidden);
    for (int64_t j = 0; j < s->hidden; j++)
        x_out[j] += weight * m->moe[j];
}

static void moe(Model *m, int layer) {
    Spec *s = &m->s;
    wk_rmsnorm(m->x, m->mlp_norm[layer], m->xb, s->hidden, s->norm_eps);
    wk_matvec(&m->router_w[layer], m->xb, m->rl);
    wk_addbias(m->rl, m->router_b[layer], s->n_experts);

    /* top-k selection, then softmax over the selected logits */
    int sel[MAX_TOPK]; float val[MAX_TOPK];
    int k = s->top_k;
    for (int i = 0; i < k; i++) { sel[i] = -1; val[i] = -1e30f; }
    for (int e = 0; e < s->n_experts; e++) {
        float v = m->rl[e];
        if (v > val[k - 1]) {
            int j = k - 1;
            while (j > 0 && val[j - 1] < v) { val[j] = val[j - 1]; sel[j] = sel[j - 1]; j--; }
            val[j] = v; sel[j] = e;
        }
    }
    /* async prefetch: tell the kernel about ALL selected experts before
       computing the first, so disk readahead overlaps compute instead of
       page faults serializing with it. Advisory + idempotent when warm.
       MIDGE_NO_PREFETCH=1 disables (for A/B measurement). */
    static int no_prefetch = -1;
    if (no_prefetch < 0) no_prefetch = getenv("MIDGE_NO_PREFETCH") != NULL;
    if (!no_prefetch) {
        for (int i = 0; i < k; i++) {
            uint8_t *base = m->experts.base
                + (size_t)layer * m->exl.layer_stride
                + (size_t)sel[i] * m->exl.expert_stride;
            madvise(base, m->exl.expert_stride, MADV_WILLNEED);
        }
    }

    float mx = val[0], den = 0.f, w[MAX_TOPK];
    if (s->router_norm) {          /* softmax over the selected logits */
        for (int i = 0; i < k; i++) { w[i] = expf(val[i] - mx); den += w[i]; }
        for (int i = 0; i < k; i++) w[i] /= den;
    } else {                       /* weights from full softmax, unnormalized */
        float fmx = m->rl[0];
        for (int e = 1; e < s->n_experts; e++) if (m->rl[e] > fmx) fmx = m->rl[e];
        float fden = 0.f;
        for (int e = 0; e < s->n_experts; e++) fden += expf(m->rl[e] - fmx);
        for (int i = 0; i < k; i++) w[i] = expf(val[i] - fmx) / fden;
    }

    memset(m->moe, 0, (size_t)s->hidden * 4);
    for (int i = 0; i < k; i++) {
        int e = sel[i];
        m->usage[(size_t)layer * s->n_experts + e]++;
        m->expert_loads++;
        WT gate, up, down; const float *bg, *bu, *bd;
        wt_expert(&m->experts, &m->exl, layer, e, 0, &gate, &bg);
        wt_expert(&m->experts, &m->exl, layer, e, 1, &up, &bu);
        wt_expert(&m->experts, &m->exl, layer, e, 2, &down, &bd);
        wk_matvec(&gate, m->xb, m->hb);  wk_addbias(m->hb, bg, s->ffn);
        wk_matvec(&up, m->xb, m->hb2);   wk_addbias(m->hb2, bu, s->ffn);
        if (s->act_plain) {
            for (int64_t j = 0; j < s->ffn; j++) {
                float g = m->hb[j];
                m->hb[j] = g / (1.0f + expf(-g)) * m->hb2[j];     /* silu(g)*u */
            }
        } else {
            for (int64_t j = 0; j < s->ffn; j++) {
                float g = m->hb[j], u = m->hb2[j];
                if (g > s->limit) g = s->limit;
                if (u > s->limit) u = s->limit;
                if (u < -s->limit) u = -s->limit;
                float act = g / (1.0f + expf(-s->alpha * g));
                m->hb[j] = act * (u + 1.0f);
            }
        }
        /* down proj into xb (reuse), weighted-accumulate into moe */
        float *tmp = m->hb2; (void)tmp;
        wk_matvec(&down, m->hb, m->xb);  /* xb reused; safe: hb holds input */
        wk_addbias(m->xb, bd, s->hidden);
        for (int64_t j = 0; j < s->hidden; j++) m->moe[j] += w[i] * m->xb[j];
        /* restore xb = normed input for next expert */
        if (i + 1 < k) wk_rmsnorm(m->x, m->mlp_norm[layer], m->xb, s->hidden, s->norm_eps);
    }
    for (int64_t j = 0; j < s->hidden; j++) m->x[j] += m->moe[j];
}

/* forward one token; if want_logits, fill m->logits */
static void forward(Model *m, int token, int want_logits) {
    Spec *s = &m->s;
    if (m->pos >= m->ctx) { fprintf(stderr, "# context full (%d)\n", m->ctx); exit(4); }
    if (token < 0 || token >= s->vocab) { fprintf(stderr, "# token %d out of range\n", token); exit(4); }
    wk_row(&m->embed, token, m->x);
    for (int layer = 0; layer < s->n_layers; layer++) {
        attention(m, layer, m->pos);
        moe(m, layer);
    }
    m->pos++;
    if (want_logits) {
        wk_rmsnorm(m->x, m->final_norm, m->xb, s->hidden, s->norm_eps);
        wk_matvec(&m->lm_head, m->xb, m->logits);
    }
}

/* Batched prefill: layer-major over a chunk of tokens; MoE applied
   expert-major so each routed expert's weights are touched once per
   layer for the whole chunk (page-cache reuse turns O(tokens*top_k)
   cold reads into O(unique experts)). Math is per-token identical to
   forward(): attention runs sequentially inside each layer, so every
   token sees exactly the KV state it would token-major.
   MIDGE_NO_BATCH=1 falls back to the per-token path. */
static void print_logits(Model *m);
static void forward_batch(Model *m, const int *toks, int n, int tfall,
                          int want_last) {
    Spec *s = &m->s;
    int h = s->hidden;
    for (int t = 0; t < n; t++) {
        if (m->pos + t >= m->ctx) { fprintf(stderr, "# context full (%d)\n", m->ctx); exit(4); }
        if (toks[t] < 0 || toks[t] >= s->vocab) { fprintf(stderr, "# token %d out of range\n", toks[t]); exit(4); }
        wk_row(&m->embed, toks[t], m->bx + (size_t)t * h);
    }
    for (int layer = 0; layer < s->n_layers; layer++) {
        for (int t = 0; t < n; t++) {
            memcpy(m->x, m->bx + (size_t)t * h, (size_t)h * 4);
            attention(m, layer, m->pos + t);
            memcpy(m->bx + (size_t)t * h, m->x, (size_t)h * 4);
        }
        for (int t = 0; t < n; t++)
            route_topk(m, layer, m->bx + (size_t)t * h,
                       m->bxb + (size_t)t * h,
                       m->bsel + t * MAX_TOPK, m->bw + t * MAX_TOPK);
        for (int e = 0; e < s->n_experts; e++) {
            int touched = 0;
            for (int t = 0; t < n; t++)
                for (int i = 0; i < s->top_k; i++)
                    if (m->bsel[t * MAX_TOPK + i] == e) {
                        if (!touched) {
                            uint8_t *base = m->experts.base
                                + (size_t)layer * m->exl.layer_stride
                                + (size_t)e * m->exl.expert_stride;
                            madvise(base, m->exl.expert_stride, MADV_WILLNEED);
                            touched = 1;
                        }
                        expert_apply(m, layer, e, m->bxb + (size_t)t * h,
                                     m->bw[t * MAX_TOPK + i],
                                     m->bx + (size_t)t * h);
                    }
        }
    }
    for (int t = 0; t < n; t++) {
        int want = tfall || (want_last && t == n - 1);
        if (want) {
            wk_rmsnorm(m->bx + (size_t)t * h, m->final_norm, m->xb, h, s->norm_eps);
            wk_matvec(&m->lm_head, m->xb, m->logits);
        }
        if (tfall) { m->pos++; print_logits(m); } 
    }
    if (!tfall) m->pos += n;
}

/* --------------------------------------------------------------- sampling */
static uint64_t rng_state = 0x853c49e6748fea9bULL;
static uint32_t rng_u32(void) {
    rng_state ^= rng_state >> 12; rng_state ^= rng_state << 25; rng_state ^= rng_state >> 27;
    return (uint32_t)((rng_state * 0x2545F4914F6CDD1DULL) >> 32);
}
static float rng_f(void) { return (rng_u32() >> 8) / 16777216.0f; }

typedef struct { float p; int i; } PI;
static int pi_cmp(const void *a, const void *b) {
    float pa = ((const PI *)a)->p, pb = ((const PI *)b)->p;
    return pa < pb ? 1 : pa > pb ? -1 : 0;
}

static int sample(Model *m, float temp, float topp) {
    int64_t V = m->s.vocab;
    if (temp <= 0.f) {
        int best = 0; float bv = m->logits[0];
        for (int64_t i = 1; i < V; i++) if (m->logits[i] > bv) { bv = m->logits[i]; best = (int)i; }
        return best;
    }
    float mx = m->logits[0];
    for (int64_t i = 1; i < V; i++) if (m->logits[i] > mx) mx = m->logits[i];
    double den = 0;
    for (int64_t i = 0; i < V; i++) { m->probs[i] = expf((m->logits[i] - mx) / temp); den += m->probs[i]; }
    for (int64_t i = 0; i < V; i++) m->probs[i] /= (float)den;
    PI *pi = malloc(V * sizeof(PI));
    for (int64_t i = 0; i < V; i++) { pi[i].p = m->probs[i]; pi[i].i = (int)i; }
    qsort(pi, V, sizeof(PI), pi_cmp);
    double cum = 0; int64_t n = 0;
    for (; n < V; n++) { cum += pi[n].p; if (cum >= topp) { n++; break; } }
    float r = rng_f() * (float)cum;
    double acc = 0;
    int pick = pi[n - 1].i;
    for (int64_t i = 0; i < n; i++) { acc += pi[i].p; if (r <= acc) { pick = pi[i].i; break; } }
    free(pi);
    return pick;
}

/* ----------------------------------------------------------------- main */
static void print_logits(Model *m) {
    printf("L");
    for (int64_t i = 0; i < m->s.vocab; i++) printf(" %.7g", m->logits[i]);
    printf("\n");
}

static int run_bench(void) {
    /* standardized measurement of the actual expert kernel: q4g32
       matvec at gpt-oss expert shape, all cores. Prints effective
       GB/s of quantized weights streamed (data+scales). */
    const int64_t rows = 2880, cols = 2880;
    size_t db = (size_t)rows * cols / 2;
    size_t sb = (size_t)rows * (cols / 32) * 2;
    uint8_t *data = malloc(db);
    uint16_t *sc = malloc(sb);
    float *x = malloc(cols * 4), *y = malloc(rows * 4);
    if (!data || !sc || !x || !y) return 1;
    for (size_t i = 0; i < db; i++) data[i] = (uint8_t)(i * 2654435761u >> 24);
    for (size_t i = 0; i < sb / 2; i++) sc[i] = 0x2c00;   /* f16 ~0.0625 */
    for (int64_t i = 0; i < cols; i++) x[i] = 0.5f;
    WT w = {0};
    w.dt = DT_Q4G32; w.rows = rows; w.cols = cols;
    w.data = data; w.scales = (uint8_t *)sc;
    /* warmup + timed loop */
    for (int i = 0; i < 3; i++) wk_matvec(&w, x, y);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int iters = 0;
    double el = 0;
    do {
        wk_matvec(&w, x, y);
        iters++;
        clock_gettime(CLOCK_MONOTONIC, &t1);
        el = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    } while (el < 1.0);
    double gbps = (double)iters * (db + sb) / el / (1024.0 * 1024.0 * 1024.0);
    printf("BENCH q4g32_gbps=%.3f threads=%d\n", gbps, omp_get_max_threads());
    free(data); free(sc); free(x); free(y);
    return 0;
}

int main(int argc, char **argv) {
    if (argc > 1 && !strcmp(argv[1], "--bench")) return run_bench();
    const char *dir = NULL;
    int ctx = 4096, ngen_default = 512, stats = 1;
    float temp = 0.0f, topp = 0.9f;
    double preload_gb = 0;
    uint64_t seed = 42;
    int stops[64], n_stops = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--ctx") && i + 1 < argc) ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--temp") && i + 1 < argc) temp = atof(argv[++i]);
        else if (!strcmp(argv[i], "--topp") && i + 1 < argc) topp = atof(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--ngen") && i + 1 < argc) ngen_default = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--preload-gb") && i + 1 < argc) preload_gb = atof(argv[++i]);
        else if (!strcmp(argv[i], "--no-stats")) stats = 0;
        else if (!strcmp(argv[i], "--stop") && i + 1 < argc) {
            char *tok = strtok(argv[++i], ",");
            while (tok && n_stops < 64) { stops[n_stops++] = atoi(tok); tok = strtok(NULL, ","); }
        }
        else if (argv[i][0] != '-') dir = argv[i];
        else { fprintf(stderr, "# unknown arg %s\n", argv[i]); return 2; }
    }
    if (!dir) { fprintf(stderr, "usage: midged <model_dir> [--ctx N] [--temp F] [--topp F] [--seed N] [--ngen N] [--stop id,id] [--preload-gb F]\n"); return 2; }
    rng_state = seed ? seed : 1;

    Model m; memset(&m, 0, sizeof(m));
    double t0 = now_s();
    model_load(&m, dir, ctx);
    preload_hot(&m, preload_gb);
    fprintf(stderr, "# midge ready in %.1fs · layers=%d experts=%d topk=%d hidden=%lld vocab=%lld ctx=%d expert_dt=%s\n",
            now_s() - t0, m.s.n_layers, m.s.n_experts, m.s.top_k,
            (long long)m.s.hidden, (long long)m.s.vocab, ctx,
            wj_strd(m.experts.hdr, "dt", "?"));
    printf("READY\n"); fflush(stdout);

    char *line = NULL; size_t cap = 0;
    int prompt_toks = 0;
    while (getline(&line, &cap, stdin) > 0) {
        char *nl = strchr(line, '\n'); if (nl) *nl = 0;
        if (!strncmp(line, "set:", 4)) {
            /* "set: <temp> <topp> [seed]" — retune sampling, keep context */
            char *p = line + 4;
            double t = strtod(p, &p), tp = strtod(p, &p);
            temp = (float)t; topp = (float)tp;
            while (*p == ' ') p++;
            if (*p) rng_state = strtoull(p, NULL, 10) | 1ull;
            printf("OK set\n"); fflush(stdout);
            continue;
        }
        if (!strncmp(line, "ids:", 4) || !strncmp(line, "tf:", 3) || !strncmp(line, "tfall:", 6)) {
            int tf = line[0] == 't';
            int tfall = !strncmp(line, "tfall:", 6);
            char *p = strchr(line, ':') + 1;
            double ts = now_s();
            int count = 0, cap0 = 64;
            int *toks = malloc(cap0 * sizeof(int));
            while (*p) {
                while (*p == ' ') p++;
                if (!*p) break;
                if (count == cap0) toks = realloc(toks, (cap0 *= 2) * sizeof(int));
                toks[count++] = (int)strtol(p, &p, 10);
            }
            prompt_toks += count;
            static int no_batch = -1;
            if (no_batch < 0) no_batch = getenv("MIDGE_NO_BATCH") != NULL;
            if (no_batch) {
                for (int i = 0; i < count; i++) {
                    forward(&m, toks[i], tfall || i == count - 1);
                    if (tfall) print_logits(&m);
                }
            } else {
                for (int off = 0; off < count; off += BATCH_MAX) {
                    int nb = count - off < BATCH_MAX ? count - off : BATCH_MAX;
                    forward_batch(&m, toks + off, nb, tfall,
                                  off + nb == count);
                }
            }
            free(toks);
            if (tf && !tfall) print_logits(&m);
            if (tf) { printf("DONE %d 0 %.3f %llu\n", count, now_s() - ts, (unsigned long long)m.expert_loads); fflush(stdout); continue; }
            fprintf(stderr, "# prefill %d toks in %.2fs (%.2f tok/s)\n", count, now_s() - ts,
                    count / (now_s() - ts + 1e-9));
            printf("OK %d\n", m.pos); fflush(stdout);
        } else if (!strncmp(line, "gen:", 4)) {
            int ngen = atoi(line + 4);
            if (ngen <= 0) ngen = ngen_default;
            double ts = now_s();
            uint64_t loads0 = m.expert_loads;
            int produced = 0, tok = -1;
            for (int i = 0; i < ngen && m.pos < m.ctx; i++) {
                tok = sample(&m, temp, topp);
                printf("T %d\n", tok); fflush(stdout);
                produced++;
                int is_stop = 0;
                for (int j = 0; j < n_stops; j++) if (stops[j] == tok) is_stop = 1;
                if (is_stop) break;
                forward(&m, tok, 1);
            }
            double dt = now_s() - ts;
            if (stats)
                fprintf(stderr, "# gen %d toks in %.2fs (%.2f tok/s) · %llu expert loads · pos=%d\n",
                        produced, dt, produced / (dt + 1e-9),
                        (unsigned long long)(m.expert_loads - loads0), m.pos);
            printf("DONE %d %d %.3f %llu\n", prompt_toks, produced, dt,
                   (unsigned long long)(m.expert_loads - loads0));
            fflush(stdout);
            usage_save(&m);
            prompt_toks = 0;
        } else if (!strcmp(line, "quit")) {
            break;
        }
    }
    usage_save(&m);
    return 0;
}
