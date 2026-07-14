"""validate_hf — compare the midge engine against transformers, token by token.

This is the *user-run* half of validation (it needs torch + transformers +
enough RAM to hold the model dequantized, so it is not part of `make test`;
the offline half is tools/validate.py, which checks the engine against a
NumPy reference on tiny fixtures).

Recommended on gpt-oss-20b (needs ~48 GB RAM for the bf16 reference).

Usage:
    python3 tools/validate_hf.py openai/gpt-oss-20b out/gpt-oss-20b \
        --prompt "The capital of France is" --gen 16
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hf_ref", help="HF repo id or local checkpoint dir")
    ap.add_argument("model_dir", help="converted midge model dir")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--gen", type=int, default=16)
    ap.add_argument("--ctx", type=int, default=512)
    args = ap.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise SystemExit("pip install torch transformers  (CPU wheels are fine)")

    tok = AutoTokenizer.from_pretrained(args.hf_ref)
    ids = tok(args.prompt, return_tensors="pt").input_ids
    id_list = ids[0].tolist()
    print(f"[validate_hf] prompt -> {len(id_list)} tokens: {id_list}")

    print("[validate_hf] loading HF reference (this needs a lot of RAM)…")
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_ref, torch_dtype=torch.bfloat16, device_map="cpu")
    model.eval()

    with torch.no_grad():
        out = model(ids)
        hf_logits = out.logits[0, -1].float().numpy()
        hf_greedy = []
        cur = ids
        for _ in range(args.gen):
            t = int(model(cur).logits[0, -1].argmax())
            hf_greedy.append(t)
            cur = torch.cat([cur, torch.tensor([[t]])], dim=1)
    del model

    # engine
    p = subprocess.Popen(
        [os.path.join(ROOT, "midged"), args.model_dir, "--ctx", str(args.ctx),
         "--temp", "0", "--no-stats"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    assert p.stdout.readline().strip() == "READY"
    p.stdin.write("tf: " + " ".join(map(str, id_list)) + "\n")
    p.stdin.flush()
    ln = p.stdout.readline()
    assert ln.startswith("L "), ln
    en_logits = np.array([float(v) for v in ln[2:].split()], np.float32)
    assert p.stdout.readline().startswith("DONE")
    p.stdin.write(f"gen: {args.gen}\n")
    p.stdin.flush()
    en_greedy = []
    while True:
        ln = p.stdout.readline().strip()
        if ln.startswith("T "):
            en_greedy.append(int(ln[2:]))
        elif ln.startswith("DONE"):
            break
    p.stdin.write("quit\n")
    p.stdin.flush()

    # compare
    k = 10
    hf_top = np.argsort(-hf_logits)[:k]
    en_top = np.argsort(-en_logits)[:k]
    rel = np.max(np.abs(hf_logits - en_logits)) / (np.max(np.abs(hf_logits)) + 1e-9)
    print(f"[validate_hf] max relative logit diff: {rel:.3e}")
    print(f"[validate_hf] top-{k} HF : {hf_top.tolist()}")
    print(f"[validate_hf] top-{k} midge: {en_top.tolist()}")
    print(f"[validate_hf] greedy HF : {hf_greedy} -> {tok.decode(hf_greedy)!r}")
    print(f"[validate_hf] greedy midge: {en_greedy} -> {tok.decode(en_greedy)!r}")
    same = sum(a == b for a, b in zip(hf_greedy, en_greedy))
    print(f"[validate_hf] greedy agreement: {same}/{args.gen}")
    if hf_top[0] != en_top[0]:
        raise SystemExit("FAIL: argmax differs on the prompt — please open an issue")
    print("[validate_hf] OK (argmax matches; small divergence over long greedy "
          "runs can come from bf16-vs-f32 kernel differences)")


if __name__ == "__main__":
    main()
