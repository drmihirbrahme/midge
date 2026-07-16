# Adding a model to midge

midge's engine is a generic spec-driven MoE transformer. If your model is
in this family, you can run it without touching the C:

* decoder-only transformer, RMSNorm, residual stream
* grouped-query attention (GQA), optional per-layer sliding window,
  optional learned attention sinks
* RoPE, optionally YaRN-scaled
* MoE FFN with a linear router, top-k selection, softmax over the
  selected logits, SwiGLU-style activation (optionally clamped),
  optional biases everywhere

That covers gpt-oss and many recent open MoE releases. Models with
different attention (e.g. MLA), shared experts, sigmoid routers or MTP
heads need engine work — see "extending the engine" below.

## Step 1: write a spec

Copy `specs/TEMPLATE.json`. Field notes:

| field | meaning |
|---|---|
| `hidden`, `n_layers`, `vocab` | trunk shape |
| `n_heads`, `n_kv_heads`, `head_dim` | GQA shape (`n_heads % n_kv_heads == 0`) |
| `norm_eps` | RMSNorm epsilon |
| `moe.experts`, `moe.top_k`, `moe.ffn` | routing + expert FFN width |
| `moe.act` | `swiglu_clamp` (α, limit; gpt-oss) or `swiglu` (plain silu(g)·u) |
| `moe.router_norm` | 1 = softmax over selected logits (gpt-oss, Mixtral); 0 = weights from the full softmax (`norm_topk_prob: false`) |
| `attn.qk_norm` | per-head RMSNorm on q,k before RoPE (Qwen3) — needs `L{i}.attn.{q,k}_norm` tensors |
| `attn.layer_types` | list, one per layer; any string containing `sliding` uses the ring KV |
| `attn.sliding_window` | window size in tokens (0 disables) |
| `attn.sinks` | learned per-head sink logits (gpt-oss: true) |
| `attn.scale` | usually `1/sqrt(head_dim)` |
| `rope.theta`, `rope.yarn` | YaRN block optional; omit for plain RoPE |
| `tokenizer.template` | `harmony` or `plain` (plain = raw prompt, EOS stop) |

If the model has a HF `config.json` with `model_type: gpt_oss`,
`tools/spec_from_hf.py` writes the spec for you.

## Step 2: map the tensors

`tools/convert.py` maps HF tensor names to midge names. The engine expects:

```
embed [vocab,hidden]      lm_head [vocab,hidden]      final_norm [hidden]
L{i}.attn.norm            L{i}.mlp.norm
L{i}.attn.{q,k,v,o}       L{i}.attn.{q,k,v,o}_b       L{i}.attn.sinks [n_heads]
L{i}.router.w [E,hidden]  L{i}.router.b [E]
```

plus, per expert, `gate`/`up` `[ffn,hidden]` and `down` `[hidden,ffn]`
with f32 biases, written through `ExpertWriter`. All matrices are
row-major `[out,in]` — HF bf16 expert tensors are `[in,out]` and get
transposed; fused `gate_up` tensors are de-interleaved (`gate = rows
0::2`). Add your model's names to the `M` dict / expert branches in
`Converter.handle`.

Missing biases? Write zeros. Tied embeddings? Add the same array as both
`embed` and `lm_head`.

## Step 3: validate before you burn 60 GB

Teach `tools/make_tiny.py` to emit a tiny random checkpoint with your
model's tensor names/shapes, add a case to `tools/validate.py`, and make
the NumPy reference (`tools/reference.py`) reflect any math difference.
When `make test` passes, run the full conversion and spot-check with
`tools/validate_hf.py`.

## Extending the engine

`engine/midge.c` is ~700 lines and deliberately boring. The seams:

* new activation → `moe()`, keyed on `moe.act`
* different router (sigmoid, expert bias, shared experts) → `moe()`
* attention variants → `attention()`; new spec fields parse in `spec_init`
* new weight dtype → one kernel in `mkern.h` + codec in `midgepack.py`
  + a `blob_sizes`/`wt_sizes` entry on both sides

Keep `ExpertLayout` (Python) and `wt_expert_layout` (C) byte-identical —
`make test` will catch you if they drift.
