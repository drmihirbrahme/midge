"""midge_mlx.cli — chat / run on the MLX backend (Apple Silicon)."""
from __future__ import annotations
import argparse
import signal
import os
import sys
import time

signal.signal(signal.SIGPIPE, signal.SIG_DFL)   # clean exit when piped to head/less
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from harmony import Harmony  # noqa: E402
from midge_mlx.model import MidgeMLX  # noqa: E402


def die(msg):
    print(f"midge-mlx: {msg}", file=sys.stderr)
    sys.exit(1)


def load(args):
    import json
    with open(os.path.join(args.model_dir, "spec.json")) as f:
        spec = json.load(f)
    try:
        from tokenizers import Tokenizer
    except ImportError:
        die("pip install tokenizers")
    tokp = os.path.join(args.model_dir, "tokenizer.json")
    if not os.path.exists(tokp):
        die(f"{tokp} not found")
    tok = Tokenizer.from_file(tokp)
    t0 = time.time()
    model = MidgeMLX(args.model_dir, ctx=args.ctx, dense_bits=args.dense_bits,
                    cache_gb=args.cache_gb)
    print(f"# midge-mlx ready in {time.time()-t0:.1f}s · "
          f"{spec['n_layers']} layers · {spec['moe']['experts']} experts "
          f"(top-{spec['moe']['top_k']}) · cache {args.cache_gb} GiB · "
          f"dense {args.dense_bits}-bit", file=sys.stderr)
    return spec, tok, model


def reply(model, h, ids, args):
    on_token, finish = h.stream(show_analysis=not args.no_analysis)
    t0, n = time.time(), 0
    for t in model.generate(ids, args.ngen, temp=args.temp, topp=args.topp,
                            stop_ids=set(h.stop_ids), seed=args.seed):
        on_token(t)
        n += 1
    finish()
    dt = time.time() - t0
    ec = model.experts
    print(f"\n# {n} toks in {dt:.1f}s ({n/dt:.2f} tok/s) · expert cache: "
          f"{ec.hits} hits / {ec.loads} loads · {ec.bytes/(1<<30):.1f} GiB held",
          file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(prog="midge-mlx")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("chat", "run"):
        c = sub.add_parser(name)
        c.add_argument("model_dir")
        if name == "run":
            c.add_argument("-p", "--prompt", required=True)
        c.add_argument("--ctx", type=int, default=8192)
        c.add_argument("--temp", type=float, default=0.7)
        c.add_argument("--topp", type=float, default=0.9)
        c.add_argument("--seed", type=int, default=42)
        c.add_argument("--ngen", type=int, default=2048)
        c.add_argument("--cache-gb", type=float, default=2.0,
                       help="expert LRU budget in unified memory")
        c.add_argument("--dense-bits", type=int, default=8, choices=[4, 8, 16, 32],
                       help="requantize dense trunk to this many bits at load")
        c.add_argument("--no-analysis", action="store_true")
        c.add_argument("--system", default="You are a helpful assistant.")
        c.add_argument("--reasoning", default="low",
                       choices=["low", "medium", "high"])
    args = ap.parse_args()

    spec, tok, model = load(args)
    h = Harmony(tok, spec)
    system = f"{args.system}\nReasoning: {args.reasoning}"

    if args.cmd == "run":
        ids = h.render([{"role": "system", "content": system},
                        {"role": "user", "content": args.prompt}])
        reply(model, h, ids, args)
        return

    # chat: keep the KV; prefill only the new turn each time
    for t in h.render([{"role": "system", "content": system}],
                      add_generation_prefix=False):
        model.forward(t)
    print("(chat ready — Ctrl-D or /quit to exit)")
    pending = []
    while True:
        try:
            user = input("\n> ").strip()
        except EOFError:
            break
        if not user:
            continue
        if user in ("/quit", "/exit"):
            break
        ids = pending + h.render([{"role": "user", "content": user}])
        pending = []
        last = None
        on_token, finish = h.stream(show_analysis=not args.no_analysis)
        n, t0 = 0, time.time()
        for t in model.generate(ids, args.ngen, temp=args.temp,
                                topp=args.topp, stop_ids=set(h.stop_ids),
                                seed=args.seed):
            on_token(t)
            last = t
            n += 1
        finish()
        if last is not None and last in h.stop_ids:
            pending = [last]      # sampled but never forwarded
        dt = time.time() - t0
        print(f"\n# {n} toks · {n/dt:.2f} tok/s", file=sys.stderr)


if __name__ == "__main__":
    main()
