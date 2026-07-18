# Getting speed out of midge

midge streams weights from disk so that models bigger than your RAM run
at all. That design has a hard ceiling: a datacenter GPU has ~100× the
memory bandwidth of a desktop CPU, so **local decode of gpt-oss-120b
will never match a cloud API**. This page is about (a) everything that
closes the gap, measured, (b) the hybrid mode that gives you literal
cloud speed for the traffic you permit, and (c) what's designed but not
built yet — with the math, so nobody has to take our word for it.

## What's implemented (measured on a 1-core cloud VM)

| lever | effect | notes |
|---|---|---|
| AVX2+FMA kernels | 0.48 → 3.98 GB/s/core (8.3×); warm decode 1.82 → 12.2 tok/s on a 4-layer gpt-oss-dims model | runtime-dispatched; `MIDGE_NO_SIMD=1` forces scalar |
| async expert prefetch | +25–40% cold decode *pre-SIMD*; **neutral (noise-level) after the AVX2 kernels** on this VM | after routing, all top-k experts are `madvise(WILLNEED)`ed so disk readahead overlaps compute. With fast kernels there is little compute left to hide behind IO on one core; it should still help when cores are busy or on the scalar fallback. Advisory-only, `MIDGE_NO_PREFETCH=1` disables |
| OpenMP threading | ~linear in cores until RAM bandwidth saturates | this VM has 1 core; desktops have 6–16 |
| `--preload-gb` | pins the hottest experts in RAM | expert usage is very skewed; a few GB captures a large share of loads |
| session prefix cache | agents skip re-prefilling conversation history | on by default in `midge serve` |
| MLX backend | GPU speed on Apple Silicon / CUDA | `--backend mlx` |

Rules of thumb after these: gpt-oss-**20b** is genuinely usable for
agents on an ordinary desktop (several tok/s warm, more with cores);
gpt-oss-**120b** lands near ~1–1.5 tok/s per core compute-side, with
disk deciding how close you get to that when cold.

## Hybrid: cloud speed where you allow it, local where you need it

```bash
./midge serve models/gpt-oss-20b \
  --upstream https://api.your-provider.com/v1 \
  --upstream-key $KEY --upstream-model gpt-oss-120b --route auto
```

**What this is and isn't.** Hybrid mode does not make local inference
faster — it lets an upstream you choose do the work at its speed. That
upstream is whatever you point it at: a paid API provider (costs their
prices), a provider's free tier, or — completely free — **your own
second machine**: a gaming PC or workstation on your LAN running midge
(or any OpenAI-compatible server) serves as the "cloud" for your laptop.

**Zero-callout guarantee.** Without `--upstream`, midge makes no network
requests at all while serving — there is no telemetry, no phoning home,
and the only outbound connection in the server exists inside the relay
code path, which is unreachable unless you configure an upstream.
`--route local` and per-request `"midge_route": "local"` are absolute.

**Local speed is never taxed.** Routing is an in-process check
(nanoseconds). If a configured upstream dies, only the first request
pays the relay timeout (`--upstream-timeout`, default 30 s); a circuit
breaker then serves straight-local for 30 s before probing again —
measured in the test suite: 2.1 s for the request that discovers the
outage, 0.05 s for the next.

One endpoint, one policy knob:

* `--route local` — nothing ever leaves the machine (default when no
  upstream is configured).
* `--route auto` — requests go upstream at full cloud speed; **any**
  upstream failure (offline, rate-limited, expired key) falls back to
  the local model transparently, so your agent never stops working.
* `--route cloud` — always upstream; failures surface as 502.
* Per-request override: `"midge_route": "local"` in the body keeps a
  sensitive request on-device even when the server default is cloud.

Responses say who answered (`"midge_served_by": "local" |
"cloud:<model>"`, header `X-Midge-Route`), so agents and logs can audit
where every token came from. Streaming is relayed verbatim.

This is the honest answer to "cloud-like speeds": the requests you
permit run on actual cloud hardware; the ones you don't, and the days
the network is down, still run.

## Designed, not yet built (in order of expected impact)

**Batched prefill.** Prefill is token-at-a-time today. Processing
blocks of B tokens turns matvecs into matmats: the dense trunk's
weights are read once per block (≈B× less dense IO/dequant) and tokens
routed to the same expert share one weight pass. With top-4-of-128
routing, a 16-token block touches ~50 unique experts instead of 64
naive expert-passes — so expect roughly **1.5–2× prefill for 120b and
2–4× for 20b** (denser routing overlap), most valuable for agents with
long tool prompts. The per-position teacher-forcing validator makes
this safe to build.

**Speculative decoding (20b drafts, 120b verifies).** Same tokenizer
family. The draft proposes k tokens; the target verifies them in one
batched pass, which on a bandwidth-bound CPU costs little more than one
token. Needs batched verification (see above) first. On pure CPU, with
the 20b draft itself costing real time, honest expectation is
**1.5–2.5×**, best when the draft agrees often (code, boilerplate).

**LAN expert swarm.** Experts are independent — shard them across the
machines you already own. Activations are tiny (~11 KB per expert hop),
so each token costs ~36 sequential LAN round-trips ⇒ a **25–50 tok/s
network ceiling**, and the aggregate page cache of 2–3 machines can
hold most of 120b hot. This is the one path to genuinely cloud-ish
local speeds, and the largest engineering item here.

Contributions to any of these are welcome — each has a validation
harness waiting for it.
