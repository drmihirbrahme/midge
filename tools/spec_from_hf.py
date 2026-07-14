"""spec_from_hf — translate a HuggingFace config.json into a midge model spec.

Currently understands the `gpt_oss` architecture. Other MoE transformers can
usually be described by the same spec fields; see docs/ADDING_MODELS.md.
"""
from __future__ import annotations
import json
import math


def spec_from_hf(cfg: dict) -> dict:
    arch = (cfg.get("architectures") or ["?"])[0]
    mt = cfg.get("model_type", "")
    if mt != "gpt_oss" and "GptOss" not in arch:
        raise SystemExit(
            f"unsupported architecture {arch!r} (model_type={mt!r}).\n"
            "midge's spec covers the MoE-transformer family (GQA + RoPE/YaRN + "
            "top-k softmax router). To add a model, write a spec by hand from "
            "specs/TEMPLATE and extend tools/convert.py's tensor mapping — "
            "see docs/ADDING_MODELS.md.")

    n_layers = cfg["num_hidden_layers"]
    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]

    layer_types = cfg.get("layer_types")
    if not layer_types:
        # gpt-oss default: alternating sliding/full starting with sliding
        layer_types = ["sliding_attention" if i % 2 == 0 else "full_attention"
                       for i in range(n_layers)]

    spec = {
        "arch": "gpt-oss",
        "hidden": cfg["hidden_size"],
        "n_layers": n_layers,
        "n_heads": cfg["num_attention_heads"],
        "n_kv_heads": cfg.get("num_key_value_heads", cfg["num_attention_heads"]),
        "head_dim": head_dim,
        "vocab": cfg["vocab_size"],
        "norm_eps": cfg.get("rms_norm_eps", 1e-5),
        "moe": {
            "experts": cfg["num_local_experts"],
            "top_k": cfg["num_experts_per_tok"],
            "ffn": cfg["intermediate_size"],
            "act": "swiglu_clamp",
            "alpha": 1.702,
            "limit": cfg.get("swiglu_limit", 7.0),
        },
        "attn": {
            "layer_types": layer_types,
            "sliding_window": cfg.get("sliding_window", 0) or 0,
            "sinks": True,
            "scale": 1.0 / math.sqrt(head_dim),
        },
        "rope": {"theta": cfg.get("rope_theta", 10000.0)},
        "max_ctx": cfg.get("max_position_embeddings", 131072),
        "tokenizer": {
            "template": "harmony",
            "stop_tokens": ["<|return|>", "<|call|>"],
            "eos_token_id": cfg.get("eos_token_id", None),
        },
    }

    rs = cfg.get("rope_scaling") or {}
    if rs.get("rope_type") == "yarn":
        spec["rope"]["yarn"] = {
            "factor": rs.get("factor", 1.0),
            "beta_fast": rs.get("beta_fast", 32.0),
            "beta_slow": rs.get("beta_slow", 1.0),
            "orig_ctx": rs.get("original_max_position_embeddings", 4096),
            "truncate": bool(rs.get("truncate", False)),
        }
    return spec


if __name__ == "__main__":
    import sys
    with open(sys.argv[1]) as f:
        print(json.dumps(spec_from_hf(json.load(f)), indent=2))
