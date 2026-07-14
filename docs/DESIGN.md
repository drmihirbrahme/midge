# midge design notes

## The premise

MoE models are mostly cold weights. gpt-oss-120b activates 4 of 128
experts per layer per token: ~5.1B of 117B parameters do work on any
given token, and expert selection is heavily skewed in practice. So the
working set is small even though the model is huge — if you can afford to
read the losers from disk when the router asks for them.

colibri (JustVugg) demonstrated this for GLM with an explicit per-layer
expert LRU cache over `pread`. midge keeps the idea and changes two things:

1. **The OS page cache is the cache.** `experts.midge` is mmap'd read-only
   with `MADV_RANDOM`. A routed expert whose pages are resident costs a
   memcpy-speed matvec; a cold one costs a page-in. There is no cache
   sizing knob because there is nothing to size: Linux keeps as many
   expert pages as free RAM allows and drops them cleanly under
   pressure. Explicit caches duplicate this machinery and add a failure
   mode (malloc'd memory can OOM; clean file-backed pages cannot).

2. **The engine is spec-driven.** Nothing about gpt-oss is hardcoded:
   layer shapes, GQA, per-layer sliding/full attention, sinks, YaRN,
   router top-k, activation clamps all come from a JSON spec embedded in
   the container. One binary, many models.

## Container format (`.midge`, version 1)

```
[u64 LE header_len][JSON header][zero pad to 64][data]
```

* `dense.midge` — header carries the full model spec plus a tensor index
  `{name: {dt, shape, off, soff}}`. Offsets are relative to the 64-byte
  aligned data start.
* `experts.midge` — header is only `{"midge":1,"kind":"experts","dt":…}`.
  Expert offsets are *not* stored: they are recomputed from a
  deterministic layout (`wt_expert_layout` in C ≡ `ExpertLayout` in
  Python). Per layer, per expert, for each of gate/up/down: data blob,
  scale blob, f32 bias, each 64-byte aligned. Deterministic layout is
  what makes conversion resumable and writes idempotent: any shard can be
  written at any time in any order.

Dtypes:

| dt      | data                              | scales               |
|---------|-----------------------------------|----------------------|
| `f32`   | raw floats                        | —                    |
| `q8r`   | int8, per-row f32 scale           | 4 B/row              |
| `q4g32` | packed nibbles (biased +8), g=32  | f16 per group        |
| `mxfp4` | packed FP4 E2M1 nibbles, g=32     | f16 per group        |

MXFP4 is stored with f16 group scales rather than the checkpoint's e8m0
bytes so that one kernel serves both 4-bit formats. The transcode is
exact: e8m0 encodes 2^(u8−127) and every exponent gpt-oss uses is a power
of two representable in f16 (the converter verifies the range
[−24, 15] and refuses otherwise).

Low nibble = even column. `tools/verify_mxfp4.py` cross-checks this
against `transformers`' dequantizer on a real shard.

## Engine (`engine/midge.c`)

Single-token decode (S=1), sequential prefill. f32 activations and
accumulation, f16 KV cache. Per layer:

```
rmsnorm → q,k,v (+bias) → YaRN RoPE (half-split rotate) → KV store
→ per-head softmax with learned sink logit → o-proj (+bias) → residual
→ rmsnorm → router (+bias) → top-k → softmax over selected
→ per expert: clamped SwiGLU (α=1.702, ±limit) → weighted sum → residual
```

Sliding-attention layers keep a ring KV buffer of `window` slots
(`slot = pos % window`) — softmax doesn't care about order, so no
reshuffling is needed and long contexts don't grow sliding layers' KV.

The expert matvecs dominate: three `[2880×2880]` 4-bit matvecs per
expert, 4 experts × 36 layers per token. Kernels are plain scalar C with
OpenMP over rows; they saturate memory bandwidth on most desktops, which
is the actual bottleneck for streamed weights. SIMD would mostly help the
warm/dense part — patches welcome, the kernel surface is three functions
in `mkern.h`.

Routing statistics accumulate in `usage.bin` (a `[layers × experts]`
u64 histogram). `--preload-gb` sorts experts by observed frequency and
pre-faults the hottest ones, which converts steady-state chat from
"mostly cold" to "mostly warm" with a few GB of spare RAM.

## Process split

The C engine speaks token ids over stdin/stdout (`ids:` / `gen:` /
`tf:` / `tfall:` → `T`, `L`, `OK`, `DONE`). Tokenization, the harmony
chat template and channel display live in the Python CLI. This keeps
the C small, makes the engine trivially scriptable, and means the
validation harness can drive the exact production code path.

## Validation strategy

`tools/reference.py` is an independent NumPy implementation that reads
the *converted* container through the Python codecs. `make test`
generates tiny random checkpoints in the genuine HF gpt-oss layout
(interleaved fused `gate_up`, and separately the MXFP4 block format),
pushes them through the real converter, and demands per-position logit
agreement plus 24 identical greedy tokens for every dtype. So one test
covers: converter tensor mapping, de-interleaving, both 4-bit codecs,
the deterministic expert layout on both sides, YaRN tables, sinks,
sliding-window ring KV, router semantics and sampling.

What it cannot cover offline is fidelity to the real released weights;
`tools/validate_hf.py` and `tools/verify_mxfp4.py` close that gap on a
machine that can hold the reference model.

## The MLX engine

`midge_mlx/` reimplements the same math on Apple's MLX and reads the same
container. Two facts make it cheap and exact:

* midge's `q4g32` blob layout (packed nibbles, low nibble = even column,
  group-32 f16 scales, values biased by +8) is byte-compatible with
  MLX's affine int4 weights when viewed as little-endian u32, with
  `biases = −8·scales`;
* midge's `mxfp4` blobs feed MLX's `mode="mxfp4"` kernels directly after
  turning the f16 power-of-two group scales back into e8m0 bytes (an
  exact log2).

Both mappings are asserted by `tools/validate_mlx.py`, which runs the
full converter-to-logits comparison against the NumPy reference on the
CPU backend — the identical kernels MLX uses on Metal. Experts are
materialized on demand from a numpy memmap into MLX arrays through a
byte-budgeted LRU (`--cache-gb`), so unified memory holds the dense
trunk plus only the hot experts.
