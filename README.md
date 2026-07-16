# midge

[![ci](https://github.com/drmihirbrahme/midge/actions/workflows/ci.yml/badge.svg)](https://github.com/drmihirbrahme/midge/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**Tiny engine, immense models.** Run 100B+ mixture-of-experts LLMs on an
ordinary CPU machine with a few GB of RAM, by keeping only the dense trunk
resident and streaming the routed experts from disk.

midge was inspired by [colibri](https://github.com/JustVugg/colibri), which
proved the idea on GLM. midge generalizes it: a small spec-driven C engine
plus a converter, so the same binary can run any model in the
MoE-transformer family (GQA · RoPE/YaRN · top-k softmax router). The first
supported models are **openai/gpt-oss-120b** and **gpt-oss-20b**.

```
model            experts on disk   resident RAM (dense+KV)   cold IO/token
gpt-oss-120b     ~60 GiB (MXFP4)   ~2.5 GiB                  ~1.9 GiB
gpt-oss-20b      ~10 GiB (MXFP4)   ~2.0 GiB                  ~1.3 GiB
```

A 75 GB disk and 8 GB of RAM are enough for the 120B model. That is the
whole point.

Two engines share one converted model directory:

* **`midged` (C)** — Linux/POSIX, mmap + page-cache expert streaming.
* **`midge-mlx` (Python)** — Apple Silicon via [MLX](https://github.com/ml-explore/mlx);
  the dense trunk sits in unified memory (requantized to 4/8-bit affine on
  load) and experts stream from disk through a byte-budgeted LRU straight
  into MLX's native quantized kernels — the container's MXFP4 and int4
  blobs are consumed **as-is** (`mode="mxfp4"` / affine-int4 with
  `biases = −8·scales`), no re-conversion, verified bit-for-bit.

## How it works

* **The dense trunk stays hot.** Embeddings, attention, router and norms
  (~2 GiB quantized at int8) are mmap'd and touched every token, so the OS
  keeps them resident.
* **Experts live on disk.** All expert FFNs sit in one big `experts.midge`
  file at deterministic offsets, mmap'd with `MADV_RANDOM`. Each token
  touches only `top_k × n_layers` experts (e.g. 4 × 36 ≈ 1.9 GiB for
  120b). The OS page cache *is* the expert cache: frequently-routed
  experts stay in whatever RAM you have, cold ones are read on demand,
  and memory pressure never OOMs you because the pages are file-backed
  and clean.
* **Usage-aware warmup.** The engine records per-expert routing counts in
  `usage.bin`; `--preload-gb N` pre-faults the N hottest gigabytes at
  startup.
* **MXFP4 kept intact.** gpt-oss ships 4-bit MXFP4 experts; the converter
  transcodes them losslessly (nibbles copied verbatim, e8m0 scales
  rewritten exactly as f16), so there is no requantization loss and no
  giant intermediate file — conversion streams shard-by-shard and needs
  only ~1 shard of scratch space.

Expect roughly **0.5–2 tok/s** once the cache is warm on a typical 4–8
core desktop with an NVMe disk, and slower, disk-bound decoding while
cold (cold ≈ `disk_MB/s ÷ 1900` tok/s for 120b). This is a
read-the-weights-every-token design: it will not race a GPU, it just runs
where nothing else can. Start with **gpt-oss-20b** — same pipeline, much
gentler IO.

## Installation

Requirements: Linux (or macOS for the MLX engine), gcc with OpenMP,
Python 3.9+.

```bash
git clone https://github.com/drmihirbrahme/midge && cd midge
make                                   # builds ./midged (the C engine)
pip install numpy tokenizers huggingface_hub
```

On Apple Silicon, additionally:

```bash
pip install mlx
```

## Usage

### The easy way: the interface

```bash
./midge ui
```

opens a local web app that handles everything: pick a model from the
catalog (storage needs are shown against your free disk), watch the
Hugging Face download and conversion stream by with live progress, the
engine builds itself at the end, and a chat panel opens — backed by the
same OpenAI-compatible server agents connect to. Interrupted downloads
resume with one click. New models are added by dropping an entry into
`specs/catalog.json` (plus a spec), which is the base for upcoming
model choices.

Everything below is the same pipeline as individual commands.

### 0. Will it run? (any Hugging Face model)

```bash
./midge check mistralai/Mixtral-8x7B-Instruct-v0.1
./midge check Qwen/Qwen3-30B-A3B      # any repo id or local path
```

fetches only the model's `config.json`, decides whether the
architecture is inside the engine's family (with specific reasons when
it isn't — "MLA is not implemented", not a shrug), then measures *this
device* — the real engine kernel's throughput, disk read speed, free
disk, available RAM — and prints a verdict with honest tok/s estimates:

```
model    openai/gpt-oss-120b  ·  arch: gpt-oss
needs    62.3 GB disk (peaks 68.3 GB converting) · 3.0 GB resident RAM at ctx 8192
device   210 GB disk free · 24.1/32.0 GB RAM available · 8 cores
measured kernel 3.1 GB/s · disk 2350 MB/s
estimate ~1.2 tok/s warm · ~1.1 tok/s cold (1.88 GB read/token cold)
verdict  ✓ can run
```

Exit code 0 = can run, 1 = can't — usable in scripts. The same check
lives in the web UI's "any Hugging Face model" box, and as a library
(`from doctor import analyze`). Supported families today: **gpt-oss**,
**Mixtral-style**, and **Qwen3-MoE-style** (plain SwiGLU, QK-norm,
`norm_topk_prob` routers) — all three validated against the NumPy
reference in CI. Architectures with MLA, shared experts, or sigmoid
routers are refused with the reason named.

### 1. Convert a checkpoint

```bash
# straight from the Hub — shards are downloaded one at a time and
# deleted after processing, so you never need the full checkpoint on disk
./midge convert openai/gpt-oss-20b models/gpt-oss-20b

# or from a local checkpoint directory
./midge convert /path/to/gpt-oss-120b models/gpt-oss-120b
```

Conversion is resumable — if it is interrupted, run the same command
again and it continues where it stopped. Options: `--experts
mxfp4|q4g32|q8r|f32` (default mxfp4, a lossless transcode for gpt-oss),
`--dense q8r|q4g32|f32` (default q8r), `--keep-shards`.

The output directory contains everything both engines need:
`spec.json`, `dense.midge`, `experts.midge`, `tokenizer.json`.

### 2. Check it fits your machine

```bash
./midge plan models/gpt-oss-120b        # or a spec: specs/gpt-oss-120b.json
```

This prints storage, resident RAM, and cold IO per token, plus
rule-of-thumb throughput for your disk speed.

### 3. Chat

```bash
./midge chat models/gpt-oss-20b
./midge run  models/gpt-oss-20b -p "Explain MoE routing in one paragraph."
```

Flags for both: `--ctx N` context length · `--temp F` / `--topp F`
sampling (temp 0 = greedy) · `--ngen N` max reply tokens ·
`--reasoning low|medium|high` · `--no-analysis` hide the model's
thinking channel · `--system "…"` system prompt · `--preload-gb N`
pre-warm the N hottest gigabytes of experts (routing statistics are
learned in `usage.bin` as you use the model, so this gets better after
the first few chats).

### 4. On a Mac (Apple Silicon)

The MLX engine reads the same model directory — nothing to re-convert:

```bash
./midge-mlx chat models/gpt-oss-20b --cache-gb 4 --dense-bits 8
```

`--cache-gb` is the unified-memory budget for the expert LRU cache;
`--dense-bits 4|8|16|32` requantizes the dense trunk at load time.
Budget roughly: dense (~2 GiB at 8-bit for gpt-oss) + cache + a couple
of GiB for the OS.

### 5. NVIDIA GPUs (CUDA)

The MLX engine is device-agnostic: MLX ships official CUDA wheels for
Linux, and midge's MLX code runs unchanged on them.

```bash
pip install "mlx[cuda]"
./midge-mlx chat models/gpt-oss-20b --device gpu --cache-gb 8
./midge serve models/gpt-oss-20b --backend mlx --device gpu   # API on GPU
```

At startup the engine probes which quantized kernels the active backend
supports and transparently falls back to exact dequantized weights for
any it lacks, so partial backend coverage degrades performance, never
correctness (the fallback path is part of the test suite). Honest
status: developed and validated on MLX's CPU backend — the same code
CUDA executes — but not yet run on physical NVIDIA hardware by the
author. If you have a CUDA box, reports (good or bad) are very welcome.

### 6. Serve an OpenAI-compatible API (for agents and apps)

```bash
./midge serve models/gpt-oss-20b --port 8420 --preload-gb 4
```

exposes `/v1/chat/completions` (streaming and non-streaming),
`/v1/completions` and `/v1/models`, so anything that speaks the OpenAI
API — agent frameworks, IDE assistants, the official client libraries —
can use midge as a local backend:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8420/v1", api_key="midge")
r = client.chat.completions.create(model="midge", stream=True,
        messages=[{"role": "user", "content": "hello"}])
```

Notes for slow hardware: use streaming (clients see progress instead of
timing out); one request runs at a time (others queue); the model's
"analysis" channel is returned as `reasoning_content`. The server keeps
the engine's context across requests — an agent loop that appends
messages only pays prefill for the *new* turn, not the whole
conversation, which matters a great deal at CPU prefill speeds. Edited
or divergent histories transparently reset the context.

### Troubleshooting

* **Slow first replies** — cold experts are being read from disk;
  warm-up is normal. Use `--preload-gb` and prefer NVMe.
* **`context full`** — raise `--ctx` (KV memory grows with it; sliding
  layers are capped at the window size regardless).
* **`pip install tokenizers` fails on old Python** — any Python ≥3.9
  with a recent pip works.
* **Conversion ran out of disk** — 120b peaks near ~72 GB during
  conversion; free space or convert 20b first.

## Validation

`make test` runs an end-to-end suite: it generates tiny random checkpoints
in the exact HF gpt-oss format (including the MXFP4 block layout and the
interleaved `gate_up` fusion), converts them with the real converter, and
compares the C engine's logits at every position — plus 24 greedy tokens —
against an independent NumPy implementation, across all supported dtypes
(f32, q8r, q4g32, mxfp4). YaRN RoPE, attention sinks, and the
sliding-window ring KV cache are all inside the tested path. `make
test-mlx` runs the same suite against the MLX engine (MLX ships a CPU
backend, so this runs in CI on Linux too); both engines and both 4-bit
kernel mappings agree with the reference to ~1e-4.

Two additional checks run on your machine against the real weights
(they need torch, so they are not part of CI):

```bash
python3 tools/verify_mxfp4.py openai/gpt-oss-20b        # decode vs transformers, 1 shard
python3 tools/validate_hf.py openai/gpt-oss-20b models/gpt-oss-20b   # logits vs transformers
```

midge's MXFP4 decoding convention (nibble order, FP4 value table, e8m0
scale semantics, block orientation) has additionally been verified
**bit-exactly** against `transformers`' official dequantizer
(`integrations/mxfp4._convert_moe_packed_tensors`) and cross-checked
against OpenAI's reference implementation (`gpt_oss/torch/model.py`) —
same activation clamps, YaRN correction range, sink softmax and
sliding-window semantics.

Honest status: everything above runs on synthetic weights. End-to-end
agreement with `transformers` on the real released checkpoints is a
one-command spot check on your machine (`validate_hf.py`); if it or
`verify_mxfp4.py` fail for you, please open an issue.

## Repository layout

```
midge                 CLI: ui / check / convert / plan / chat / run / serve
midge-mlx             MLX CLI (Apple Silicon & CUDA): chat / run
midged                (built) C engine — token ids in, token ids out
engine/midge.c        the whole engine: spec-driven MoE transformer (+ --bench)
engine/mten.h         .midge container reader (mmap, dtypes, expert layout)
engine/mkern.h        matvec kernels: f32, q8 rows, 4-bit groups (int4/FP4)
engine/mjson.h        small JSON parser
midge_mlx/            MLX engine: unified-memory LRU over the same container
webui/index.html      the onboarding interface (offline, no dependencies)
tools/convert.py      HF checkpoint -> container (streaming, resumable)
tools/doctor.py       `midge check`: compatibility + device-fit verdicts
tools/serve.py        OpenAI-compatible API server (C or MLX backend)
tools/ui.py           interface backend: setup jobs, progress, serve handoff
tools/midgepack.py    codecs + container writers/readers (source of truth)
tools/reference.py    NumPy forward pass (ground truth for tests)
tools/validate*.py    the five test suites (make test / test-mlx / test-server / test-ui)
specs/                model descriptors + catalog.json (the model registry)
docs/                 DESIGN.md · ADDING_MODELS.md
```

## Adding a model

The engine is driven by a JSON spec (see `specs/TEMPLATE.json`): layer
count, GQA shape, per-layer sliding/full attention, YaRN parameters,
router top-k, activation clamps. If your model fits the MoE-transformer
family, you need a spec plus a tensor-name mapping in `tools/convert.py` —
no C changes. Details in [docs/ADDING_MODELS.md](docs/ADDING_MODELS.md).

## Author

Dr Mihir Brahme — drmihir@duck.com. Issues and PRs welcome; see
[CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
