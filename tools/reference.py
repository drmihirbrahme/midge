"""reference — NumPy forward pass over a midge container.

This is the ground truth the C engine is validated against. It reads the
*converted* container (dense.midge + experts.midge) through midgepack's
dequantizing readers, so a single comparison exercises the converter,
the pack codecs, the deterministic expert layout and the engine math.

The math mirrors engine/midge.c exactly:
  RMSNorm -> QKV(+bias) -> YaRN RoPE (half-split rotate) -> f16 KV cache
  (ring buffer on sliding layers) -> softmax with learned sinks ->
  o-proj(+bias) -> residual -> RMSNorm -> router top-k -> softmax over
  selected -> clamped-SwiGLU experts (+biases) -> weighted sum -> residual.
"""
from __future__ import annotations
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import midgepack as wp


def f16_round(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float16).astype(np.float32)


class Reference:
    def __init__(self, model_dir: str, ctx: int = 256):
        self.spec, self.dense = wp.read_dense(os.path.join(model_dir, "dense.midge"))
        self.exdt, self.expert = wp.read_experts(
            os.path.join(model_dir, "experts.midge"), self.spec)
        s = self.spec
        self.ctx = min(ctx, s.get("max_ctx", ctx))
        self.hd = s["head_dim"]
        self.nh, self.nkv = s["n_heads"], s["n_kv_heads"]
        self.cos, self.sin = self._rope_tables()
        self.reset()

    def reset(self):
        s = self.spec
        self.pos = 0
        self.kc, self.vc, self.cap = [], [], []
        win = s["attn"].get("sliding_window", 0)
        for lt in s["attn"]["layer_types"]:
            sliding = ("sliding" in lt) and 0 < win < self.ctx
            cap = win if sliding else self.ctx
            self.cap.append(cap)
            self.kc.append(np.zeros((cap, self.nkv * self.hd), np.float32))
            self.vc.append(np.zeros((cap, self.nkv * self.hd), np.float32))

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
        v = v.reshape(-1, self.hd).copy()
        a, b = v[:, :half].copy(), v[:, half:].copy()
        co, si = self.cos[pos], self.sin[pos]
        v[:, :half] = a * co - b * si
        v[:, half:] = b * co + a * si
        return v.reshape(-1)

    # ---------------------------------------------------------- helpers
    @staticmethod
    def rmsnorm(x, w, eps):
        ms = np.mean(x.astype(np.float64) ** 2)
        return (x / np.sqrt(ms + eps) * w).astype(np.float32)

    # ---------------------------------------------------------- forward
    def forward(self, token: int) -> np.ndarray:
        s, d = self.spec, self.dense
        x = d["embed"][token].astype(np.float32).copy()
        for L in range(s["n_layers"]):
            x = x + self._attention(L, x)
            x = x + self._moe(L, x)
        self.pos += 1
        h = self.rmsnorm(x, d["final_norm"], s["norm_eps"])
        return d["lm_head"] @ h

    def _attention(self, L: int, x: np.ndarray) -> np.ndarray:
        s, d = self.spec, self.dense
        nh, nkv, hd = self.nh, self.nkv, self.hd
        pos, cap = self.pos, self.cap[L]
        xb = self.rmsnorm(x, d[f"L{L}.attn.norm"], s["norm_eps"])
        q = d[f"L{L}.attn.q"] @ xb + d[f"L{L}.attn.q_b"]
        k = d[f"L{L}.attn.k"] @ xb + d[f"L{L}.attn.k_b"]
        v = d[f"L{L}.attn.v"] @ xb + d[f"L{L}.attn.v_b"]
        if s["attn"].get("qk_norm"):
            qn, kn = d[f"L{L}.attn.q_norm"], d[f"L{L}.attn.k_norm"]
            q = np.concatenate([self.rmsnorm(h, qn, s["norm_eps"])
                                for h in q.reshape(-1, hd)])
            k = np.concatenate([self.rmsnorm(h, kn, s["norm_eps"])
                                for h in k.reshape(-1, hd)])
        q = self._rope(q, pos)
        k = self._rope(k, pos)

        slot = pos % cap
        self.kc[L][slot] = f16_round(k)
        self.vc[L][slot] = f16_round(v)
        nctx = min(pos + 1, cap)

        out = np.zeros(nh * hd, np.float32)
        gq = nh // nkv
        sinks = d.get(f"L{L}.attn.sinks") if s["attn"].get("sinks") else None
        for h in range(nh):
            qh = q[h * hd:(h + 1) * hd]
            kh = h // gq
            K = self.kc[L][:nctx, kh * hd:(kh + 1) * hd]
            V = self.vc[L][:nctx, kh * hd:(kh + 1) * hd]
            sc = (K @ qh) * s["attn"]["scale"]
            mx = sc.max()
            if sinks is not None:
                mx = max(mx, sinks[h])
            e = np.exp(sc - mx)
            den = e.sum() + (np.exp(sinks[h] - mx) if sinks is not None else 0.0)
            out[h * hd:(h + 1) * hd] = (e / den) @ V
        return (d[f"L{L}.attn.o"] @ out + d[f"L{L}.attn.o_b"]).astype(np.float32)

    def _moe(self, L: int, x: np.ndarray) -> np.ndarray:
        s, d = self.spec, self.dense
        m = s["moe"]
        xb = self.rmsnorm(x, d[f"L{L}.mlp.norm"], s["norm_eps"])
        rl = d[f"L{L}.router.w"] @ xb + d[f"L{L}.router.b"]
        k = m["top_k"]
        sel = np.argsort(-rl, kind="stable")[:k]
        if m.get("router_norm", 1):
            w = np.exp(rl[sel] - rl[sel].max())
            w = w / w.sum()
        else:                       # weights from full softmax, unnormalized
            full = np.exp(rl - rl.max())
            w = full[sel] / full.sum()

        alpha, limit = m.get("alpha", 1.702), m.get("limit", 7.0)
        acc = np.zeros(s["hidden"], np.float32)
        for wi, e in zip(w, sel):
            ex = self.expert(L, int(e))
            g = ex["gate"] @ xb + ex["gate_b"]
            u = ex["up"] @ xb + ex["up_b"]
            if m.get("act") == "swiglu":
                act = g / (1.0 + np.exp(-g)) * u
            else:
                g = np.minimum(g, limit)
                u = np.clip(u, -limit, limit)
                act = g / (1.0 + np.exp(-alpha * g)) * (u + 1.0)
            acc += wi * (ex["down"] @ act.astype(np.float32) + ex["down_b"])
        return acc.astype(np.float32)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--ids", required=True, help="comma-separated token ids")
    ap.add_argument("--gen", type=int, default=0, help="greedy tokens to generate")
    ap.add_argument("--ctx", type=int, default=256)
    args = ap.parse_args()

    ref = Reference(args.model_dir, ctx=args.ctx)
    ids = [int(t) for t in args.ids.split(",")]
    logits = None
    for t in ids:
        logits = ref.forward(t)
    gen = []
    for _ in range(args.gen):
        t = int(np.argmax(logits))
        gen.append(t)
        logits = ref.forward(t)
    print("last-logits:", " ".join(f"{v:.6f}" for v in logits[:16]))
    if gen:
        print("greedy:", " ".join(map(str, gen)))


if __name__ == "__main__":
    main()
