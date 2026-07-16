"""spec_from_hf — translate a HuggingFace config.json into a midge model spec.

Understands the MoE-transformer family the engine implements:

    gpt_oss        clamped SwiGLU, sinks, YaRN, alternating sliding
    mixtral        plain SwiGLU, no biases, selected-softmax router
    qwen3_moe      plain SwiGLU, QK-norm, norm_topk_prob router

Architectures outside the family raise Unsupported with structured,
per-feature reasons — `midge check` turns those into its verdict.
"""
from __future__ import annotations
import json
import math


class Unsupported(Exception):
    """Raised with .reasons: a list of specific blocking features."""

    def __init__(self, arch, reasons):
        self.arch, self.reasons = arch, reasons
        super().__init__(
            f"unsupported architecture {arch!r}:\n  - " + "\n  - ".join(reasons)
            + "\nSee docs/ADDING_MODELS.md to extend the engine.")


# model_type -> known blocking features (checked before attempting a spec)
KNOWN_BLOCKERS = {
    "deepseek_v2": ["multi-head latent attention (MLA) is not implemented"],
    "deepseek_v3": ["multi-head latent attention (MLA) is not implemented",
                    "sigmoid router with expert bias is not implemented"],
    "glm4_moe": ["GLM MoE uses shared experts, which are not implemented"],
    "qwen2_moe": ["shared expert (+gate) is not implemented"],
    "granitemoe": ["router/attention multipliers are not implemented"],
}


def _base(cfg, arch_name):
    n_layers = cfg["num_hidden_layers"]
    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    lt = cfg.get("layer_types")
    if not lt:
        win = cfg.get("sliding_window") or 0
        use = bool(win) and bool(cfg.get("use_sliding_window", True))
        lt = ["sliding_attention" if use else "full_attention"] * n_layers
    return {
        "arch": arch_name,
        "hidden": cfg["hidden_size"],
        "n_layers": n_layers,
        "n_heads": cfg["num_attention_heads"],
        "n_kv_heads": cfg.get("num_key_value_heads", cfg["num_attention_heads"]),
        "head_dim": head_dim,
        "vocab": cfg["vocab_size"],
        "norm_eps": cfg.get("rms_norm_eps", 1e-5),
        "attn": {
            "layer_types": lt,
            "sliding_window": cfg.get("sliding_window", 0) or 0,
            "sinks": False,
            "scale": 1.0 / math.sqrt(head_dim),
        },
        "rope": {"theta": cfg.get("rope_theta", 10000.0)},
        "max_ctx": cfg.get("max_position_embeddings", 32768),
        "tokenizer": {
            "template": "plain",
            "eos_token_id": cfg.get("eos_token_id"),
        },
    }, head_dim


def _yarn(cfg, spec):
    rs = cfg.get("rope_scaling") or {}
    if rs.get("rope_type") == "yarn":
        spec["rope"]["yarn"] = {
            "factor": rs.get("factor", 1.0),
            "beta_fast": rs.get("beta_fast", 32.0),
            "beta_slow": rs.get("beta_slow", 1.0),
            "orig_ctx": rs.get("original_max_position_embeddings", 4096),
            "truncate": bool(rs.get("truncate", False)),
        }
    elif rs and rs.get("rope_type", rs.get("type", "default")) not in ("default", None):
        raise Unsupported(cfg.get("model_type", "?"),
                          [f"rope_scaling type {rs.get('rope_type') or rs.get('type')!r} "
                           "is not implemented (only yarn/default)"])


def spec_from_hf(cfg: dict) -> dict:
    mt = cfg.get("model_type", "")
    arch = (cfg.get("architectures") or [mt or "?"])[0]

    if mt in KNOWN_BLOCKERS:
        raise Unsupported(arch, KNOWN_BLOCKERS[mt])

    if mt == "gpt_oss" or "GptOss" in arch:
        spec, hd = _base(cfg, "gpt-oss")
        n = spec["n_layers"]
        if not cfg.get("layer_types"):
            spec["attn"]["layer_types"] = [
                "sliding_attention" if i % 2 == 0 else "full_attention"
                for i in range(n)]
        spec["moe"] = {"experts": cfg["num_local_experts"],
                       "top_k": cfg["num_experts_per_tok"],
                       "ffn": cfg["intermediate_size"],
                       "act": "swiglu_clamp", "alpha": 1.702,
                       "limit": cfg.get("swiglu_limit", 7.0),
                       "router_norm": 1}
        spec["attn"]["sinks"] = True
        spec["max_ctx"] = cfg.get("max_position_embeddings", 131072)
        spec["tokenizer"] = {"template": "harmony",
                             "stop_tokens": ["<|return|>", "<|call|>"],
                             "eos_token_id": cfg.get("eos_token_id")}
        _yarn(cfg, spec)
        return spec

    if mt == "mixtral" or "Mixtral" in arch:
        spec, _ = _base(cfg, "mixtral")
        spec["moe"] = {"experts": cfg["num_local_experts"],
                       "top_k": cfg["num_experts_per_tok"],
                       "ffn": cfg["intermediate_size"],
                       "act": "swiglu", "router_norm": 1}
        _yarn(cfg, spec)
        return spec

    if mt == "qwen3_moe" or "Qwen3Moe" in arch:
        reasons = []
        if cfg.get("shared_expert_intermediate_size"):
            reasons.append("shared expert is not implemented")
        if cfg.get("mlp_only_layers"):
            reasons.append("dense (non-MoE) interleaved layers are not implemented")
        if reasons:
            raise Unsupported(arch, reasons)
        spec, _ = _base(cfg, "qwen3-moe")
        spec["moe"] = {"experts": cfg["num_experts"],
                       "top_k": cfg["num_experts_per_tok"],
                       "ffn": cfg["moe_intermediate_size"],
                       "act": "swiglu",
                       "router_norm": 1 if cfg.get("norm_topk_prob", True) else 0}
        spec["attn"]["qk_norm"] = True
        _yarn(cfg, spec)
        return spec

    # unknown model_type: name what we can see is missing
    reasons = [f"model_type {mt!r} has no tensor mapping in tools/convert.py"]
    if not any(k in cfg for k in
               ("num_local_experts", "num_experts", "n_routed_experts")):
        reasons.append("config declares no routed experts — midge only helps "
                       "for MoE models (dense models: use llama.cpp)")
    raise Unsupported(arch, reasons)


if __name__ == "__main__":
    import sys
    with open(sys.argv[1]) as f:
        print(json.dumps(spec_from_hf(json.load(f)), indent=2))
