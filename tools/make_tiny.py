"""make_tiny — generate a tiny random gpt-oss-style HF checkpoint.

Used by the validation suite: the fixture goes through tools/convert.py
exactly like a real checkpoint (same tensor names, same interleaved
gate_up layout, optionally the same MXFP4 block format), then the C
engine's logits are compared against tools/reference.py.

Variants:
    --format bf16    gate_up_proj / down_proj as BF16 [E, in, out]
    --format mxfp4   *_blocks / *_scales u8 tensors like the gpt-oss release
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import midgepack as wp

CFG = {
    "architectures": ["GptOssForCausalLM"],
    "model_type": "gpt_oss",
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "vocab_size": 128,
    "num_local_experts": 4,
    "num_experts_per_tok": 2,
    "intermediate_size": 64,
    "rms_norm_eps": 1e-5,
    "rope_theta": 150000.0,
    "sliding_window": 8,
    "layer_types": ["sliding_attention", "full_attention"],
    "swiglu_limit": 7.0,
    "max_position_embeddings": 256,
    "eos_token_id": 127,
    "rope_scaling": {
        "rope_type": "yarn",
        "factor": 4.0,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "original_max_position_embeddings": 64,
        "truncate": False,
    },
}


def rnd(rng, *shape, scale=0.5):
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--format", default="bf16", choices=["bf16", "mxfp4"])
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    c = CFG
    hid, ffn = c["hidden_size"], c["intermediate_size"]
    E, L = c["num_local_experts"], c["num_hidden_layers"]
    nh, nkv, hd = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    V = c["vocab_size"]

    T = {}

    def add(name, arr, dt="BF16"):
        T[name] = (arr, dt)

    add("model.embed_tokens.weight", rnd(rng, V, hid))
    add("lm_head.weight", rnd(rng, V, hid))
    add("model.norm.weight", 1.0 + rnd(rng, hid, scale=0.1))

    for i in range(L):
        p = f"model.layers.{i}."
        add(p + "input_layernorm.weight", 1.0 + rnd(rng, hid, scale=0.1))
        add(p + "post_attention_layernorm.weight", 1.0 + rnd(rng, hid, scale=0.1))
        add(p + "self_attn.q_proj.weight", rnd(rng, nh * hd, hid))
        add(p + "self_attn.q_proj.bias", rnd(rng, nh * hd, scale=0.1))
        add(p + "self_attn.k_proj.weight", rnd(rng, nkv * hd, hid))
        add(p + "self_attn.k_proj.bias", rnd(rng, nkv * hd, scale=0.1))
        add(p + "self_attn.v_proj.weight", rnd(rng, nkv * hd, hid))
        add(p + "self_attn.v_proj.bias", rnd(rng, nkv * hd, scale=0.1))
        add(p + "self_attn.o_proj.weight", rnd(rng, hid, nh * hd))
        add(p + "self_attn.o_proj.bias", rnd(rng, hid, scale=0.1))
        add(p + "self_attn.sinks", rnd(rng, nh, scale=1.0))
        add(p + "mlp.router.weight", rnd(rng, E, hid))
        add(p + "mlp.router.bias", rnd(rng, E, scale=0.2))

        if args.format == "bf16":
            # HF bf16 layout: [E, in, out]; gate/up interleaved on out axis
            add(p + "mlp.experts.gate_up_proj", rnd(rng, E, hid, 2 * ffn))
            add(p + "mlp.experts.gate_up_proj_bias", rnd(rng, E, 2 * ffn, scale=0.1))
            add(p + "mlp.experts.down_proj", rnd(rng, E, ffn, hid))
            add(p + "mlp.experts.down_proj_bias", rnd(rng, E, hid, scale=0.1))
        else:
            # gpt-oss release layout: packed FP4 nibbles + e8m0 scales, out-major
            def blocks(rows, cols):
                codes = rng.integers(0, 16, size=(E, rows, cols), dtype=np.uint8)
                packed = (codes[..., 0::2] | (codes[..., 1::2] << 4))
                return packed.reshape(E, rows, cols // 32, 16)

            def scales(rows, cols):
                # exponents around 2^-6 .. 2^-1: safely inside f16-exact range
                return rng.integers(121, 127, size=(E, rows, cols // 32),
                                    dtype=np.uint8)

            add(p + "mlp.experts.gate_up_proj_blocks", blocks(2 * ffn, hid), "U8")
            add(p + "mlp.experts.gate_up_proj_scales", scales(2 * ffn, hid), "U8")
            add(p + "mlp.experts.gate_up_proj_bias", rnd(rng, E, 2 * ffn, scale=0.1))
            add(p + "mlp.experts.down_proj_blocks", blocks(hid, ffn), "U8")
            add(p + "mlp.experts.down_proj_scales", scales(hid, ffn), "U8")
            add(p + "mlp.experts.down_proj_bias", rnd(rng, E, hid, scale=0.1))

    wp.write_safetensors(os.path.join(args.outdir, "model.safetensors"), T)
    with open(os.path.join(args.outdir, "config.json"), "w") as f:
        json.dump(c, f, indent=2)
    write_tiny_tokenizer(os.path.join(args.outdir, "tokenizer.json"), c["vocab_size"])
    print(f"[make_tiny] wrote {args.outdir} ({args.format}, {len(T)} tensors)")


def write_tiny_tokenizer(path: str, vocab_size: int):
    """A minimal harmony-capable tokenizer (character-level WordLevel) so
    the full chat pipeline can be exercised without downloading anything.
    Special tokens mirror o200k_harmony's control tokens."""
    specials = ["<|start|>", "<|message|>", "<|end|>", "<|channel|>",
                "<|return|>", "<|call|>", "<unk>"]
    vocab = {t: i for i, t in enumerate(specials)}
    import string
    for ch in string.ascii_letters + string.digits + string.punctuation.replace('"', '') + " ":
        if len(vocab) >= vocab_size:
            break
        vocab[ch] = len(vocab)
    tok = {
        "version": "1.0",
        "added_tokens": [
            {"id": vocab[t], "content": t, "special": True,
             "single_word": False, "lstrip": False, "rstrip": False,
             "normalized": False}
            for t in specials
        ],
        "normalizer": None,
        "pre_tokenizer": {"type": "Split", "pattern": {"String": ""},
                          "behavior": "Isolated", "invert": False},
        "post_processor": None,
        "decoder": {"type": "Fuse"},
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<unk>"},
    }
    with open(path, "w") as f:
        json.dump(tok, f)


if __name__ == "__main__":
    main()
