"""midge_mlx.model — the midge engine on Apple's MLX.

Runs the exact same converted container (spec.json + dense.midge +
experts.midge) as the C engine — nothing to re-convert. The dense trunk
lives in unified memory (optionally requantized to MLX affine 4/8-bit);
experts stay on disk in a numpy memmap and are materialized into MLX
arrays on demand through a byte-budgeted LRU cache.

The container's 4-bit blobs feed MLX's quantized kernels directly:
  * q4g32  -> affine int4, group 32, biases = -8 * scales
  * mxfp4  -> mode="mxfp4", f16 power-of-two scales turned back into
              e8m0 uint8 (exact — they are transcoded powers of two)
Both mappings are verified bit-for-bit by tools/validate_mlx.py.

Math mirrors engine/midge.c: f32 activations, f16 KV (ring buffer on
sliding layers), softmax with learned sinks, top-k router with softmax
over the selected logits, clamped SwiGLU.
"""
from __future__ import annotations
import json
import os
from collections import OrderedDict

import numpy as np
import mlx.core as mx


def select_device(device: str = "auto"):
    """device: auto | cpu | gpu (gpu = Metal on macOS, CUDA on Linux
    builds installed via `pip install "mlx[cuda]"`)."""
    if device == "cpu":
        mx.set_default_device(mx.cpu)
    elif device == "gpu":
        try:
            mx.set_default_device(mx.gpu)
        except Exception as e:
            raise RuntimeError(
                "no GPU device available in this MLX build "
                "(install mlx[cuda] on Linux, plain mlx on Apple Silicon): "
                f"{e}")
    return mx.default_device()


def probe_kernels() -> dict:
    """Test which quantized kernels the active backend supports, so the
    engine can fall back to dequantized weights per-op instead of
    crashing on backends (e.g. new CUDA builds) that lack one."""
    caps = {}
    x = mx.ones((1, 64), dtype=mx.float32)
    w = mx.zeros((8, 8), dtype=mx.uint32)   # 8 nibbles/u32 -> 64 cols
    s16 = mx.ones((8, 2), dtype=mx.float16)
    s8 = mx.full((8, 2), 127, dtype=mx.uint8)
    for name, kw in [
        ("affine4_g32", dict(scales=s16, biases=-8.0 * s16, group_size=32, bits=4)),
        ("mxfp4", dict(scales=s8, group_size=32, bits=4, mode="mxfp4")),
    ]:
        try:
            mx.eval(mx.quantized_matmul(x, w, transpose=True, **kw))
            caps[name] = True
        except Exception:
            caps[name] = False
    try:
        q = mx.quantize(mx.ones((8, 64), dtype=mx.float32), group_size=64, bits=8)
        mx.eval(mx.quantized_matmul(x, *q, transpose=True, group_size=64, bits=8))
        caps["affine_g64"] = True
    except Exception:
        caps["affine_g64"] = False
    return caps

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import midgepack as wp


def _f16_to_e8m0(scales_f16: np.ndarray) -> np.ndarray | None:
    """f16 power-of-two scales -> e8m0 uint8, or None if not powers of two."""
    s = scales_f16.astype(np.float64)
    if not (s > 0).all():
        return None
    e = np.log2(s)
    er = np.rint(e)
    if not np.allclose(e, er, atol=1e-9):
        return None
    return (er + 127).astype(np.uint8)


class _DenseMat:
    """A dense matrix, either f16 or MLX affine-quantized."""

    def __init__(self, w_f32: np.ndarray, bits: int, caps=None):
        if caps is not None and bits < 16 and not caps.get("affine_g64"):
            bits = 32                      # backend lacks affine kernels; stay exact
        if bits >= 32:
            self.q = None
            self.w = mx.array(w_f32.astype(np.float32))
        elif bits >= 16:
            self.q = None
            self.w = mx.array(w_f32.astype(np.float16))
        else:
            self.q = mx.quantize(mx.array(w_f32), group_size=64, bits=bits)
            self.bits = bits

    def matvec(self, x: mx.array) -> mx.array:
        if self.q is None:
            return (self.w.astype(mx.float32) @ x[:, None])[:, 0]
        y = mx.quantized_matmul(x[None, :], *self.q, transpose=True,
                                group_size=64, bits=self.bits)
        return y[0].astype(mx.float32)

    def row(self, i: int) -> mx.array:
        if self.q is None:
            return self.w[i].astype(mx.float32)
        w, s, b = self.q
        return mx.dequantize(w[i:i + 1], s[i:i + 1], b[i:i + 1],
                             group_size=64, bits=self.bits)[0].astype(mx.float32)


class _ExpertCache:
    """Byte-budgeted LRU of materialized expert weights (MLX arrays)."""

    def __init__(self, path: str, spec: dict, budget_bytes: int, caps=None):
        self.caps = caps or {}
        with open(path, "rb") as f:
            import struct
            (hl,) = struct.unpack("<Q", f.read(8))
            hdr = json.loads(f.read(hl))
        self.base = wp.align64(8 + hl)
        self.dt = hdr["dt"]
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")
        self.lay = wp.ExpertLayout(self.dt, spec["hidden"], spec["moe"]["ffn"],
                                   spec["moe"]["experts"], spec["n_layers"])
        self.budget = budget_bytes
        self.cache: OrderedDict = OrderedDict()
        self.bytes = 0
        self.loads = 0
        self.hits = 0

    def _blob(self, layer, e, mat, part, n):
        off = self.base + self.lay.offset(layer, e, mat, part)
        return np.asarray(self.mm[off:off + n])

    def _materialize(self, layer, e):
        out = {}
        for mat in ("gate", "up", "down"):
            rows, cols = self.lay.rows[mat], self.lay.cols[mat]
            db, sb = wp.blob_sizes(self.dt, rows, cols)
            bias = np.frombuffer(self._blob(layer, e, mat, "bias", rows * 4),
                                 np.float32).copy()
            entry = {"bias": mx.array(bias), "rows": rows, "cols": cols}
            use_kernel = self.caps.get(
                "affine4_g32" if self.dt == "q4g32" else "mxfp4", True)
            if self.dt in ("q4g32", "mxfp4") and use_kernel:
                data = self._blob(layer, e, mat, "data", db)
                wq = mx.array(np.frombuffer(data, np.uint32)
                              .reshape(rows, cols // 8).copy())
                s16 = np.frombuffer(self._blob(layer, e, mat, "scales", sb),
                                    np.float16).reshape(rows, cols // 32).copy()
                if self.dt == "mxfp4":
                    e8 = _f16_to_e8m0(s16)
                    if e8 is not None:
                        entry.update(kind="mxfp4", wq=wq, sc=mx.array(e8))
                    else:   # non-power-of-two scales: dequantize once
                        w = wp.mxfp4_decode(data.tobytes(), s16.tobytes(), rows, cols)
                        entry.update(kind="f32", w=mx.array(w.astype(np.float32)))
                else:
                    sc = mx.array(s16)
                    entry.update(kind="q4", wq=wq, sc=sc, bi=-8.0 * sc)
            else:  # q8r / f32 / kernel-less 4-bit -> dense f32 (exact)
                data = self._blob(layer, e, mat, "data", db)
                scales = self._blob(layer, e, mat, "scales", sb).tobytes() if sb else b""
                w = wp.decode(self.dt, data.tobytes(), scales, rows, cols)
                entry.update(kind="f32", w=mx.array(w.astype(np.float32)))
            out[mat] = entry
        return out, self.lay.expert_stride

    def get(self, layer: int, e: int):
        key = (layer, e)
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key][0]
        entry, nbytes = self._materialize(layer, e)
        self.loads += 1
        self.cache[key] = (entry, nbytes)
        self.bytes += nbytes
        while self.bytes > self.budget and len(self.cache) > 1:
            _, (_, nb) = self.cache.popitem(last=False)
            self.bytes -= nb
        return entry

    @staticmethod
    def matvec(entry: dict, x: mx.array) -> mx.array:
        k = entry["kind"]
        if k == "q4":
            y = mx.quantized_matmul(x[None, :], entry["wq"], entry["sc"],
                                    entry["bi"], transpose=True,
                                    group_size=32, bits=4)
        elif k == "mxfp4":
            y = mx.quantized_matmul(x[None, :], entry["wq"], entry["sc"],
                                    transpose=True, group_size=32, bits=4,
                                    mode="mxfp4")
        else:
            return (entry["w"].astype(mx.float32) @ x[:, None])[:, 0] + entry["bias"]
        return y[0].astype(mx.float32) + entry["bias"]


class MidgeMLX:
    def __init__(self, model_dir: str, ctx: int = 8192, dense_bits: int = 8,
                 cache_gb: float = 2.0, device: str = "auto"):
        self.device = select_device(device)
        self.caps = probe_kernels()
        self.spec, dense = wp.read_dense(os.path.join(model_dir, "dense.midge"))
        s = self.spec
        self.ctx = min(ctx, s.get("max_ctx", ctx))
        self.hd, self.nh, self.nkv = s["head_dim"], s["n_heads"], s["n_kv_heads"]
        self.experts = _ExpertCache(os.path.join(model_dir, "experts.midge"),
                                    s, int(cache_gb * (1 << 30)), self.caps)
        self.d = {}
        for name, w in dense.items():
            if w.ndim == 1:
                self.d[name] = mx.array(w.astype(np.float32))
            elif name.startswith("L") and ".router." in name:
                self.d[name] = _DenseMat(w, 32)     # routers stay full precision
            else:
                self.d[name] = _DenseMat(w, dense_bits, self.caps)
        self.cos, self.sin = self._rope_tables()
        self.reset()

    # ------------------------------------------------------------ state
    def reset(self):
        s = self.spec
        self.pos = 0
        self.kc, self.vc, self.cap = [], [], []
        win = s["attn"].get("sliding_window", 0)
        for lt in s["attn"]["layer_types"]:
            sliding = ("sliding" in lt) and 0 < win < self.ctx
            cap = win if sliding else self.ctx
            self.cap.append(cap)
            self.kc.append(np.zeros((cap, self.nkv, self.hd), np.float16))
            self.vc.append(np.zeros((cap, self.nkv, self.hd), np.float16))

    # ------------------------------------------------------------- rope
    def _rope_tables(self):
        s = self.spec
        hd, half = self.hd, self.hd // 2
        base = float(s["rope"]["theta"])
        inv = 1.0 / base ** (np.arange(0, hd, 2, dtype=np.float64) / hd)
        mscale = 1.0
        y = s["rope"].get("yarn")
        if y:
            f, orig = float(y["factor"]), float(y["orig_ctx"])
            low = hd * np.log(orig / (y["beta_fast"] * 2 * np.pi)) / (2 * np.log(base))
            high = hd * np.log(orig / (y["beta_slow"] * 2 * np.pi)) / (2 * np.log(base))
            if y.get("truncate"):
                low, high = np.floor(low), np.ceil(high)
            low, high = max(low, 0.0), min(high, hd - 1)
            if high == low:
                high = low + 0.001
            i = np.arange(half, dtype=np.float64)
            ramp = np.clip((i - low) / (high - low), 0.0, 1.0)
            mask = 1.0 - ramp
            inv = (inv / f) * (1.0 - mask) + inv * mask
            mscale = 0.1 * np.log(f) + 1.0
        t = np.arange(self.ctx, dtype=np.float64)[:, None] * inv[None, :]
        return ((np.cos(t) * mscale).astype(np.float32),
                (np.sin(t) * mscale).astype(np.float32))

    def _rope(self, v: np.ndarray, pos: int) -> np.ndarray:
        half = self.hd // 2
        v = v.reshape(-1, self.hd)
        a, b = v[:, :half].copy(), v[:, half:].copy()
        co, si = self.cos[pos], self.sin[pos]
        v[:, :half] = a * co - b * si
        v[:, half:] = b * co + a * si
        return v

    # ---------------------------------------------------------- forward
    @staticmethod
    def _rmsnorm(x: mx.array, w: mx.array, eps: float) -> mx.array:
        ms = mx.mean(x.astype(mx.float32) ** 2)
        return x * mx.rsqrt(ms + eps) * w

    def forward(self, token: int) -> np.ndarray:
        s = self.spec
        if self.pos >= self.ctx:
            raise RuntimeError(f"context full ({self.ctx})")
        x = self.d["embed"].row(token)
        for L in range(s["n_layers"]):
            x = x + self._attention(L, x)
            x = x + self._moe(L, x)
        self.pos += 1
        h = self._rmsnorm(x, self.d["final_norm"], s["norm_eps"])
        logits = self.d["lm_head"].matvec(h)
        self._last_logits = np.array(logits, dtype=np.float32)
        return self._last_logits

    def _attention(self, L: int, x: mx.array) -> mx.array:
        s = self.spec
        nh, nkv, hd = self.nh, self.nkv, self.hd
        pos, cap = self.pos, self.cap[L]
        xb = self._rmsnorm(x, self.d[f"L{L}.attn.norm"], s["norm_eps"])
        q = self.d[f"L{L}.attn.q"].matvec(xb) + self.d[f"L{L}.attn.q_b"]
        k = self.d[f"L{L}.attn.k"].matvec(xb) + self.d[f"L{L}.attn.k_b"]
        v = self.d[f"L{L}.attn.v"].matvec(xb) + self.d[f"L{L}.attn.v_b"]
        qv = np.array(q, dtype=np.float32)
        kv2 = np.array(k, dtype=np.float32)
        if s["attn"].get("qk_norm"):
            qw = np.array(self.d[f"L{L}.attn.q_norm"], dtype=np.float32)
            kw = np.array(self.d[f"L{L}.attn.k_norm"], dtype=np.float32)
            def _hn(v, w):
                v = v.reshape(-1, hd)
                ms = np.mean(v.astype(np.float64) ** 2, axis=1, keepdims=True)
                return (v / np.sqrt(ms + s["norm_eps"]) * w).astype(np.float32).reshape(-1)
            qv = _hn(qv, qw)
            kv2 = _hn(kv2, kw)
        qn = self._rope(qv, pos)                                  # [nh, hd]
        kn = self._rope(kv2, pos)                                 # [nkv, hd]

        slot = pos % cap
        self.kc[L][slot] = kn.astype(np.float16)
        self.vc[L][slot] = np.array(v, dtype=np.float32).reshape(nkv, hd).astype(np.float16)
        nctx = min(pos + 1, cap)

        gq = nh // nkv
        Q = mx.array(qn.reshape(nkv, gq, hd))                    # [nkv, gq, hd]
        K = mx.array(self.kc[L][:nctx].astype(np.float32)).transpose(1, 2, 0)  # [nkv, hd, nctx]
        V = mx.array(self.vc[L][:nctx].astype(np.float32)).transpose(1, 0, 2)  # [nkv, nctx, hd]
        sc = (Q @ K) * s["attn"]["scale"]                        # [nkv, gq, nctx]
        if s["attn"].get("sinks"):
            sink = self.d[f"L{L}.attn.sinks"].reshape(nkv, gq)[:, :, None]
            full = mx.concatenate([sc, sink], axis=2)
            p = mx.softmax(full, axis=2)[:, :, :nctx]            # sink weight dropped
        else:
            p = mx.softmax(sc, axis=2)
        out = (p @ V).reshape(nh * hd)                           # [nkv, gq, hd] -> flat
        return self.d[f"L{L}.attn.o"].matvec(out) + self.d[f"L{L}.attn.o_b"]

    def _moe(self, L: int, x: mx.array) -> mx.array:
        s = self.spec
        m = s["moe"]
        xb = self._rmsnorm(x, self.d[f"L{L}.mlp.norm"], s["norm_eps"])
        rl = np.array(self.d[f"L{L}.router.w"].matvec(xb)
                      + self.d[f"L{L}.router.b"], dtype=np.float32)
        k = m["top_k"]
        sel = np.argsort(-rl, kind="stable")[:k]
        if m.get("router_norm", 1):
            w = np.exp(rl[sel] - rl[sel].max())
            w = w / w.sum()
        else:
            full = np.exp(rl - rl.max())
            w = full[sel] / full.sum()

        alpha, limit = m.get("alpha", 1.702), m.get("limit", 7.0)
        acc = mx.zeros(s["hidden"], dtype=mx.float32)
        for wi, e in zip(w, sel):
            ex = self.experts.get(L, int(e))
            g = self.experts.matvec(ex["gate"], xb)
            u = self.experts.matvec(ex["up"], xb)
            if m.get("act") == "swiglu":
                act = g * mx.sigmoid(g) * u
            else:
                g = mx.minimum(g, limit)
                u = mx.clip(u, -limit, limit)
                act = g * mx.sigmoid(alpha * g) * (u + 1.0)
            acc = acc + float(wi) * self.experts.matvec(ex["down"], act)
        return acc

    # --------------------------------------------------------- sampling
    def sample(self, logits: np.ndarray, temp: float, topp: float,
               rng: np.random.Generator) -> int:
        if temp <= 0:
            return int(np.argmax(logits))
        p = np.exp((logits - logits.max()) / temp)
        p /= p.sum()
        order = np.argsort(-p, kind="stable")
        cum = np.cumsum(p[order])
        n = int(np.searchsorted(cum, topp) + 1)
        keep = order[:n]
        pk = p[keep] / p[keep].sum()
        return int(rng.choice(keep, p=pk))

    def generate(self, prompt_ids, max_tokens: int, temp=0.0, topp=0.9,
                 stop_ids=(), seed=42):
        rng = np.random.default_rng(seed)
        logits = None
        for t in prompt_ids:
            logits = self.forward(t)
        if logits is None:
            logits = getattr(self, "_last_logits", None)
            if logits is None:
                raise ValueError("empty prompt and no prior forward pass")
        for _ in range(max_tokens):
            t = self.sample(logits, temp, topp, rng)
            yield t
            if t in stop_ids or self.pos >= self.ctx:
                return
            logits = self.forward(t)
