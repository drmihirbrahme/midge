"""convert — HuggingFace checkpoint -> midge container.

Torch-free (numpy only; huggingface_hub is optional, used to download).
Designed for machines with little RAM *and* little disk:

  * shards are processed one at a time and (when downloaded) deleted
    immediately afterwards, so peak disk = final container + one shard;
  * MXFP4 checkpoints (gpt-oss) are transcoded *losslessly*: the packed
    FP4 nibbles are copied verbatim and only the e8m0 group scales are
    rewritten as f16 (exact for every exponent gpt-oss uses);
  * conversion is resumable: state is checkpointed after every shard.

Usage:
    python3 tools/convert.py openai/gpt-oss-20b out/gpt-oss-20b
    python3 tools/convert.py /path/to/local/checkpoint out/model \
        --experts mxfp4 --dense q8r
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import midgepack as wp
from spec_from_hf import spec_from_hf


# ------------------------------------------------------------ helpers
def log(msg):
    print(f"[convert] {msg}", flush=True)


def e8m0_to_f16_bytes(u8: np.ndarray) -> bytes:
    """e8m0 scale (2^(u8-127)) -> IEEE f16 bits, exact or die.

    f16 represents 2^e exactly for -24 <= e <= 15 (subnormals included).
    gpt-oss expert scales live far inside that range; anything outside
    means we'd silently lose precision, so we refuse instead.
    """
    e = u8.astype(np.int32) - 127
    if e.max(initial=-127) > 15 or e.min(initial=127) < -24:
        raise SystemExit(
            f"e8m0 scale exponent out of f16-exact range "
            f"[{e.min()}, {e.max()}]; container format would be lossy. "
            "Please open an issue with the model name.")
    bits = np.where(e >= -14,
                    ((e + 15) << 10).astype(np.uint16),      # normal
                    (1 << (e + 24)).astype(np.uint16))       # subnormal power of two
    return bits.astype(np.uint16).tobytes()


class ResumableDense:
    """Like midgepack.DenseWriter, but the index/offset survive restarts
    (the temp data file is kept and reopened)."""

    def __init__(self, path: str, spec: dict, state: dict):
        self.path, self.spec = path, spec
        self.tmp = path + ".data.tmp"
        self.index = state.get("dense_index", {})
        self.off = state.get("dense_off", 0)
        mode = "r+b" if (os.path.exists(self.tmp) and self.off) else "w+b"
        self.f = open(self.tmp, mode)
        self.f.seek(self.off)

    def _put(self, blob: bytes) -> int:
        at = self.off
        self.f.write(blob)
        pad = wp.align64(len(blob)) - len(blob)
        if pad:
            self.f.write(b"\0" * pad)
        self.off = wp.align64(self.off + len(blob))
        return at

    def add(self, name: str, w: np.ndarray, dt: str = "f32"):
        if name in self.index:
            return
        if w.ndim == 1:
            data, scales = np.ascontiguousarray(w, dtype=np.float32).tobytes(), b""
            shape = [int(w.shape[0])]
        else:
            data, scales = wp.encode(dt, w)
            shape = [int(w.shape[0]), int(w.shape[1])]
        e = {"dt": dt, "shape": shape, "off": self._put(data)}
        if scales:
            e["soff"] = self._put(scales)
        self.index[name] = e

    def snapshot(self, state: dict):
        self.f.flush()
        state["dense_index"] = self.index
        state["dense_off"] = self.off

    def finish(self):
        self.f.close()
        hdr = {"midge": 1, "kind": "dense", "spec": self.spec, "tensors": self.index}
        with open(self.path, "wb") as out:
            blob = json.dumps(hdr).encode()
            import struct
            out.write(struct.pack("<Q", len(blob)))
            out.write(blob)
            out.write(b"\0" * (wp.align64(8 + len(blob)) - (8 + len(blob))))
            with open(self.tmp, "rb") as t:
                shutil.copyfileobj(t, out, 1 << 22)
        os.remove(self.tmp)


# ------------------------------------------------- tensor dispatching
class Converter:
    def __init__(self, spec, outdir, expert_dt, dense_dt, transpose_experts,
                 tie_embeddings=False):
        self.spec = spec
        self.tie = tie_embeddings
        self.expert_dt = expert_dt
        self.dense_dt = dense_dt
        self.transpose = transpose_experts
        self.state_path = os.path.join(outdir, "convert.state.json")
        self.state = {}
        if os.path.exists(self.state_path):
            with open(self.state_path) as f:
                self.state = json.load(f)
            log(f"resuming: {len(self.state.get('done', []))} shard(s) already done")
        self.dense = ResumableDense(os.path.join(outdir, "dense.midge"), spec, self.state)
        layout = wp.ExpertLayout(expert_dt, spec["hidden"], spec["moe"]["ffn"],
                                 spec["moe"]["experts"], spec["n_layers"])
        self.experts = wp.ExpertWriter(os.path.join(outdir, "experts.midge"), layout)

    # -- shard bookkeeping
    def done(self, shard):
        return shard in self.state.get("done", [])

    def mark(self, shard):
        self.state.setdefault("done", []).append(shard)
        self.dense.snapshot(self.state)
        self.experts.f.flush()
        os.fsync(self.experts.f.fileno())
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.state, f)
        os.replace(tmp, self.state_path)

    def finalize(self):
        """Backfill tensors the engine requires but this architecture
        doesn't ship: zero biases, tied lm_head, zero sinks."""
        s = self.spec
        idx = self.dense.index
        if self.tie and "lm_head" not in idx and "embed" in idx:
            e = dict(idx["embed"])
            idx["lm_head"] = e                    # same offsets: true tying
        hid, nh, nkv, hd = s["hidden"], s["n_heads"], s["n_kv_heads"], s["head_dim"]
        E = s["moe"]["experts"]
        need = {"final_norm": hid}
        for i in range(s["n_layers"]):
            need.update({f"L{i}.attn.q_b": nh * hd, f"L{i}.attn.k_b": nkv * hd,
                         f"L{i}.attn.v_b": nkv * hd, f"L{i}.attn.o_b": hid,
                         f"L{i}.router.b": E})
        z = {}
        for name, n in need.items():
            if name not in idx:
                blob = z.get(n)
                if blob is None:
                    z[n] = blob = np.zeros(n, np.float32)
                self.dense.add(name, blob)

    # -- per-tensor handling
    def handle(self, name: str, st: wp.SafeTensors):
        # dense trunk -------------------------------------------------
        if name == "model.embed_tokens.weight":
            return self.dense.add("embed", st.get(name), self.dense_dt)
        if name == "lm_head.weight":
            return self.dense.add("lm_head", st.get(name), self.dense_dt)
        if name == "model.norm.weight":
            return self.dense.add("final_norm", st.get(name))
        if not name.startswith("model.layers."):
            log(f"  (skipping unknown tensor {name})")
            return
        i = int(name.split(".")[2])
        rest = name.split(".", 3)[3]
        M = {
            "input_layernorm.weight": (f"L{i}.attn.norm", "f32"),
            "post_attention_layernorm.weight": (f"L{i}.mlp.norm", "f32"),
            "self_attn.q_proj.weight": (f"L{i}.attn.q", self.dense_dt),
            "self_attn.k_proj.weight": (f"L{i}.attn.k", self.dense_dt),
            "self_attn.v_proj.weight": (f"L{i}.attn.v", self.dense_dt),
            "self_attn.o_proj.weight": (f"L{i}.attn.o", self.dense_dt),
            "self_attn.q_proj.bias": (f"L{i}.attn.q_b", "f32"),
            "self_attn.k_proj.bias": (f"L{i}.attn.k_b", "f32"),
            "self_attn.v_proj.bias": (f"L{i}.attn.v_b", "f32"),
            "self_attn.o_proj.bias": (f"L{i}.attn.o_b", "f32"),
            "self_attn.sinks": (f"L{i}.attn.sinks", "f32"),
            "mlp.router.weight": (f"L{i}.router.w", "f32"),
            "mlp.router.bias": (f"L{i}.router.b", "f32"),
            # mixtral-style
            "block_sparse_moe.gate.weight": (f"L{i}.router.w", "f32"),
            # qwen3-moe style
            "mlp.gate.weight": (f"L{i}.router.w", "f32"),
            "self_attn.q_norm.weight": (f"L{i}.attn.q_norm", "f32"),
            "self_attn.k_norm.weight": (f"L{i}.attn.k_norm", "f32"),
        }
        if rest in M:
            nm, dt = M[rest]
            w = st.get(name)
            return self.dense.add(nm, w, dt if w.ndim == 2 else "f32")
        # experts -----------------------------------------------------
        if rest.startswith("mlp.experts."):
            key = rest[len("mlp.experts."):]
            if key[:1].isdigit():        # qwen3-moe: experts.<E>.<mat>.weight
                return self.handle_expert_indexed(i, key, st, name)
            return self.handle_expert(i, key, st, name)
        if rest.startswith("block_sparse_moe.experts."):
            return self.handle_expert_indexed(
                i, rest[len("block_sparse_moe.experts."):], st, name,
                names={"w1": "gate", "w3": "up", "w2": "down"})
        log(f"  (skipping unknown tensor {name})")

    def handle_expert_indexed(self, layer, key, st, full, names=None):
        """Per-expert submodule tensors: '<E>.<mat>.weight', already
        [out, in] row-major (standard nn.Linear)."""
        names = names or {"gate_proj": "gate", "up_proj": "up",
                          "down_proj": "down"}
        parts = key.split(".")
        e, mat = int(parts[0]), names.get(parts[1])
        if mat is None or parts[-1] != "weight":
            return log(f"  (skipping unknown expert tensor {full})")
        w = st.get(full)
        self.experts_put_quant(layer, e, mat, np.ascontiguousarray(w))
        L = self.experts.layout
        self.experts_put_bias(layer, e, mat,
                              np.zeros(L.rows[mat], np.float32))

    def handle_expert(self, layer: int, key: str, st: wp.SafeTensors, full: str):
        E = self.spec["moe"]["experts"]
        ffn = self.spec["moe"]["ffn"]
        hid = self.spec["hidden"]

        # --- MXFP4-block checkpoints (gpt-oss release format) --------
        if key.endswith(("_blocks", "_scales")) and st.info(full)[0] == "U8":
            if self.expert_dt != "mxfp4":
                raise SystemExit("source is MXFP4 blocks; use --experts mxfp4 "
                                 "(lossless transcode). Re-quantizing blocks to "
                                 f"{self.expert_dt} is not supported.")
        if key == "gate_up_proj_blocks":
            b = st.get(full)                      # [E, 2*ffn, in/32, 16] u8
            for e in range(E):
                self.experts_put_raw(layer, e, "gate", b[e, 0::2].tobytes())
                self.experts_put_raw(layer, e, "up", b[e, 1::2].tobytes())
            return
        if key == "down_proj_blocks":
            b = st.get(full)                      # [E, hid, ffn/32, 16]
            for e in range(E):
                self.experts_put_raw(layer, e, "down", b[e].tobytes())
            return
        if key == "gate_up_proj_scales":
            sarr = st.get(full)                   # [E, 2*ffn, in/32] u8
            for e in range(E):
                self.experts_put_scales(layer, e, "gate", e8m0_to_f16_bytes(sarr[e, 0::2]))
                self.experts_put_scales(layer, e, "up", e8m0_to_f16_bytes(sarr[e, 1::2]))
            return
        if key == "down_proj_scales":
            sarr = st.get(full)
            for e in range(E):
                self.experts_put_scales(layer, e, "down", e8m0_to_f16_bytes(sarr[e]))
            return
        if key == "gate_up_proj_bias":
            barr = st.get(full)                   # [E, 2*ffn]
            for e in range(E):
                self.experts_put_bias(layer, e, "gate", barr[e, 0::2])
                self.experts_put_bias(layer, e, "up", barr[e, 1::2])
            return
        if key == "down_proj_bias":
            barr = st.get(full)
            for e in range(E):
                self.experts_put_bias(layer, e, "down", barr[e])
            return

        # --- bf16/f32 checkpoints -------------------------------------
        if key == "gate_up_proj":
            w = st.get(full)                      # [E, hid, 2*ffn] (in-major)
            for e in range(E):
                t = w[e].T if self.transpose else w[e]   # -> [2*ffn, hid]
                assert t.shape == (2 * ffn, hid), t.shape
                self.experts_put_quant(layer, e, "gate", np.ascontiguousarray(t[0::2]))
                self.experts_put_quant(layer, e, "up", np.ascontiguousarray(t[1::2]))
            return
        if key == "down_proj":
            w = st.get(full)                      # [E, ffn, hid]
            for e in range(E):
                t = w[e].T if self.transpose else w[e]   # -> [hid, ffn]
                assert t.shape == (hid, ffn), t.shape
                self.experts_put_quant(layer, e, "down", np.ascontiguousarray(t))
            return
        log(f"  (skipping unknown expert tensor {full})")

    # -- raw writes into the deterministic expert layout
    def experts_put_raw(self, layer, e, mat, data: bytes):
        L = self.experts.layout
        db, _ = wp.blob_sizes(L.dt, L.rows[mat], L.cols[mat])
        assert len(data) == db, (mat, len(data), db)
        self.experts.f.seek(self.experts.base + L.offset(layer, e, mat, "data"))
        self.experts.f.write(data)

    def experts_put_scales(self, layer, e, mat, scales: bytes):
        L = self.experts.layout
        _, sb = wp.blob_sizes(L.dt, L.rows[mat], L.cols[mat])
        assert len(scales) == sb, (mat, len(scales), sb)
        self.experts.f.seek(self.experts.base + L.offset(layer, e, mat, "scales"))
        self.experts.f.write(scales)

    def experts_put_bias(self, layer, e, mat, bias: np.ndarray):
        L = self.experts.layout
        b = np.ascontiguousarray(bias, dtype=np.float32).tobytes()
        assert len(b) == L.rows[mat] * 4
        self.experts.f.seek(self.experts.base + L.offset(layer, e, mat, "bias"))
        self.experts.f.write(b)

    def experts_put_quant(self, layer, e, mat, w: np.ndarray):
        data, scales = wp.encode(self.expert_dt, w)
        self.experts_put_raw(layer, e, mat, data)
        if scales:
            self.experts_put_scales(layer, e, mat, scales)


# ------------------------------------------------------------- source
class Source:
    """A checkpoint that is either a local directory or a HF repo id."""

    def __init__(self, ref: str, workdir: str, purge: bool | None):
        self.local = os.path.isdir(ref)
        self.ref = ref
        self.workdir = workdir
        self.purge = (not self.local) if purge is None else purge
        if not self.local:
            try:
                from huggingface_hub import hf_hub_download  # noqa
            except ImportError:
                raise SystemExit("pip install huggingface_hub  (or pass a local dir)")

    def fetch(self, filename: str, required=True) -> str | None:
        if self.local:
            p = os.path.join(self.ref, filename)
            if not os.path.exists(p):
                if required:
                    raise SystemExit(f"missing {p}")
                return None
            return p
        from huggingface_hub import hf_hub_download
        try:
            return hf_hub_download(self.ref, filename, local_dir=self.workdir)
        except Exception as ex:
            if required:
                raise SystemExit(f"failed to download {filename}: {ex}")
            return None

    def release(self, path: str):
        if self.purge and not self.local and path and os.path.exists(path):
            os.remove(path)

    def shards(self) -> list[str]:
        idx = self.fetch("model.safetensors.index.json", required=False)
        if idx:
            with open(idx) as f:
                wm = json.load(f)["weight_map"]
            return sorted(set(wm.values()))
        return ["model.safetensors"]


# --------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="local checkpoint dir or HF repo id")
    ap.add_argument("outdir", help="output model directory")
    ap.add_argument("--experts", default="mxfp4",
                    choices=["mxfp4", "q4g32", "q8r", "f32"],
                    help="expert weight dtype (default mxfp4)")
    ap.add_argument("--dense", default="q8r",
                    choices=["q8r", "q4g32", "f32"],
                    help="dense matrix dtype (default q8r)")
    ap.add_argument("--keep-shards", action="store_true",
                    help="don't delete downloaded shards after processing")
    ap.add_argument("--no-transpose-experts", action="store_true",
                    help="expert weights are already [out, in] "
                         "(HF bf16 checkpoints store [in, out])")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    src = Source(args.source, args.outdir, purge=(False if args.keep_shards else None))

    cfgp = src.fetch("config.json")
    with open(cfgp) as f:
        cfg = json.load(f)
    spec = spec_from_hf(cfg)
    with open(os.path.join(args.outdir, "spec.json"), "w") as f:
        json.dump(spec, f, indent=2)
    if os.path.abspath(cfgp) != os.path.abspath(os.path.join(args.outdir, "config.json")):
        shutil.copy(cfgp, os.path.join(args.outdir, "config.json"))

    tok = src.fetch("tokenizer.json", required=False)
    if tok and os.path.abspath(tok) != os.path.abspath(os.path.join(args.outdir, "tokenizer.json")):
        shutil.copy(tok, os.path.join(args.outdir, "tokenizer.json"))
    elif not tok:
        log("warning: no tokenizer.json found; copy one into the model dir")

    conv = Converter(spec, args.outdir, args.experts, args.dense,
                     transpose_experts=not args.no_transpose_experts,
                     tie_embeddings=bool(cfg.get("tie_word_embeddings")))

    shards = src.shards()
    log(f"{len(shards)} shard(s); experts={args.experts} dense={args.dense}")
    for n, shard in enumerate(shards):
        if conv.done(shard):
            log(f"[{n + 1}/{len(shards)}] {shard} (done, skipping)")
            continue
        log(f"[{n + 1}/{len(shards)}] {shard}")
        path = src.fetch(shard)
        st = wp.SafeTensors(path)
        for name in st.names():
            conv.handle(name, st)
        del st
        conv.mark(shard)
        src.release(path)

    conv.finalize()
    conv.dense.finish()
    conv.experts.close()
    os.remove(conv.state_path)
    log(f"wrote {args.outdir}/dense.midge, experts.midge, spec.json")
    log("next: ./midge chat " + args.outdir)


if __name__ == "__main__":
    main()
