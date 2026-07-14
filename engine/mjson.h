/* mjson.h — minimal JSON parser (subset sufficient for midge headers/specs).
 * Arena-free: nodes are malloc'd and never freed (engine lifetime data).
 * Supports: objects, arrays, strings (with \" \\ \/ \b \f \n \r \t \uXXXX->'?'),
 * numbers (strtod), true/false/null.
 */
#ifndef WJSON_H
#define WJSON_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

typedef struct WJ WJ;
struct WJ {
    char type;      /* 'o' object, 'a' array, 's' string, 'n' number, 'b' bool, 'z' null */
    double num;     /* number or bool value */
    char *str;      /* string value */
    int n;          /* children count */
    char **keys;    /* object keys (n) */
    WJ **items;     /* object values / array items (n) */
};

typedef struct { const char *p; const char *end; int err; } wj_ctx;

static void wj_ws(wj_ctx *c) { while (c->p < c->end && (unsigned char)*c->p <= ' ') c->p++; }

static WJ *wj_new(char type) { WJ *j = (WJ *)calloc(1, sizeof(WJ)); j->type = type; return j; }

static WJ *wj_value(wj_ctx *c);

static char *wj_string(wj_ctx *c) {
    if (c->p >= c->end || *c->p != '"') { c->err = 1; return NULL; }
    c->p++;
    size_t cap = 32, len = 0;
    char *s = (char *)malloc(cap);
    while (c->p < c->end && *c->p != '"') {
        char ch = *c->p++;
        if (ch == '\\' && c->p < c->end) {
            char e = *c->p++;
            switch (e) {
                case 'n': ch = '\n'; break; case 't': ch = '\t'; break;
                case 'r': ch = '\r'; break; case 'b': ch = '\b'; break;
                case 'f': ch = '\f'; break; case '"': ch = '"'; break;
                case '\\': ch = '\\'; break; case '/': ch = '/'; break;
                case 'u': { if (c->end - c->p >= 4) c->p += 4; ch = '?'; break; }
                default: ch = e;
            }
        }
        if (len + 2 > cap) { cap *= 2; s = (char *)realloc(s, cap); }
        s[len++] = ch;
    }
    if (c->p < c->end && *c->p == '"') c->p++; else c->err = 1;
    s[len] = 0;
    return s;
}

static void wj_push(WJ *j, char *key, WJ *item) {
    j->items = (WJ **)realloc(j->items, sizeof(WJ *) * (j->n + 1));
    if (j->type == 'o') j->keys = (char **)realloc(j->keys, sizeof(char *) * (j->n + 1));
    if (j->type == 'o') j->keys[j->n] = key;
    j->items[j->n] = item;
    j->n++;
}

static WJ *wj_value(wj_ctx *c) {
    wj_ws(c);
    if (c->p >= c->end) { c->err = 1; return NULL; }
    char ch = *c->p;
    if (ch == '{') {
        c->p++; WJ *j = wj_new('o');
        wj_ws(c);
        if (c->p < c->end && *c->p == '}') { c->p++; return j; }
        for (;;) {
            wj_ws(c);
            char *k = wj_string(c); if (c->err) return j;
            wj_ws(c);
            if (c->p < c->end && *c->p == ':') c->p++; else { c->err = 1; return j; }
            WJ *v = wj_value(c); if (c->err) return j;
            wj_push(j, k, v);
            wj_ws(c);
            if (c->p < c->end && *c->p == ',') { c->p++; continue; }
            if (c->p < c->end && *c->p == '}') { c->p++; return j; }
            c->err = 1; return j;
        }
    }
    if (ch == '[') {
        c->p++; WJ *j = wj_new('a');
        wj_ws(c);
        if (c->p < c->end && *c->p == ']') { c->p++; return j; }
        for (;;) {
            WJ *v = wj_value(c); if (c->err) return j;
            wj_push(j, NULL, v);
            wj_ws(c);
            if (c->p < c->end && *c->p == ',') { c->p++; continue; }
            if (c->p < c->end && *c->p == ']') { c->p++; return j; }
            c->err = 1; return j;
        }
    }
    if (ch == '"') { WJ *j = wj_new('s'); j->str = wj_string(c); return j; }
    if (ch == 't') { c->p += 4; WJ *j = wj_new('b'); j->num = 1; return j; }
    if (ch == 'f') { c->p += 5; WJ *j = wj_new('b'); j->num = 0; return j; }
    if (ch == 'n') { c->p += 4; return wj_new('z'); }
    /* number */
    {
        char *endp = NULL;
        double d = strtod(c->p, &endp);
        if (endp == c->p) { c->err = 1; return NULL; }
        c->p = endp;
        WJ *j = wj_new('n'); j->num = d; return j;
    }
}

static WJ *wj_parse(const char *buf, size_t len) {
    wj_ctx c = { buf, buf + len, 0 };
    WJ *j = wj_value(&c);
    if (c.err) return NULL;
    return j;
}

/* lookups */
static WJ *wj_get(WJ *o, const char *key) {
    if (!o || o->type != 'o') return NULL;
    for (int i = 0; i < o->n; i++) if (strcmp(o->keys[i], key) == 0) return o->items[i];
    return NULL;
}
static WJ *wj_at(WJ *a, int i) {
    if (!a || a->type != 'a' || i < 0 || i >= a->n) return NULL;
    return a->items[i];
}
static double wj_numd(WJ *o, const char *key, double dflt) {
    WJ *v = wj_get(o, key);
    return (v && (v->type == 'n' || v->type == 'b')) ? v->num : dflt;
}
static const char *wj_strd(WJ *o, const char *key, const char *dflt) {
    WJ *v = wj_get(o, key);
    return (v && v->type == 's') ? v->str : dflt;
}

#endif /* WJSON_H */
