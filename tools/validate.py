"""validate — end-to-end test: tiny HF checkpoint -> convert -> C engine
vs NumPy reference.

For each case we:
  1. generate a tiny random gpt-oss-style HF checkpoint (make_tiny),
  2. convert it with tools/convert.py (the same code path real
     checkpoints take, including MXFP4 block transcoding),
  3. teacher-force a token sequence through the C engine (`tfall:`),
     collecting logits at every position,
  4. compare against tools/reference.py, position by position,
  5. compare 24 greedy-decoded continuations token-for-token.

Run via `make test`. Requires numpy and a built ./midged.
"""
from __future__ import annotations
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from reference import Reference  # noqa: E402

MIDGED = os.path.join(ROOT, "midged")
TMP = os.path.join(ROOT, ".test-tmp")

CASES = [
    # (name, make_tiny format, convert flags)
    ("bf16->f32", "bf16", ["--experts", "f32", "--dense", "f32"]),
    ("bf16->q8r", "bf16", ["--experts", "q8r", "--dense", "q8r"]),
    ("bf16->q4g32", "bf16", ["--experts", "q4g32", "--dense", "q8r"]),
    ("bf16->mxfp4", "bf16", ["--experts", "mxfp4", "--dense", "q8r"]),
    ("mxfp4-blocks", "mxfp4", ["--experts", "mxfp4", "--dense", "q8r"]),
]

IDS = [1, 17, 42, 5, 99, 3, 77, 12, 63, 8, 120, 31]  # crosses sliding window=8
NGEN = 24
CTX = 64


def sh(args):
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit(f"command failed: {' '.join(map(str, args))}")
    return r.stdout


class Engine:
    def __init__(self, model_dir):
        self.p = subprocess.Popen(
            [MIDGED, model_dir, "--ctx", str(CTX), "--temp", "0", "--no-stats"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        assert self._line() == "READY"

    def _line(self):
        ln = self.p.stdout.readline()
        if not ln:
            raise SystemExit("engine died: " + (self.p.stderr or ""))
        return ln.rstrip("\n")

    def tfall(self, ids):
        self.p.stdin.write("tfall: " + " ".join(map(str, ids)) + "\n")
        self.p.stdin.flush()
        out = []
        for _ in ids:
            ln = self._line()
            assert ln.startswith("L "), ln
            out.append(np.array([float(v) for v in ln[2:].split()], np.float32))
        assert self._line().startswith("DONE")
        return out

    def gen(self, n):
        self.p.stdin.write(f"gen: {n}\n")
        self.p.stdin.flush()
        toks = []
        while True:
            ln = self._line()
            if ln.startswith("T "):
                toks.append(int(ln[2:]))
            elif ln.startswith("DONE"):
                return toks

    def close(self):
        try:
            self.p.stdin.write("quit\n")
            self.p.stdin.flush()
        except Exception:
            pass
        self.p.wait(timeout=10)


def run_case(name, fmt, flags):
    hf = os.path.join(TMP, f"hf-{fmt}")
    model = os.path.join(TMP, f"model-{name.replace('>', '').replace('-', '_')}")
    if not os.path.exists(os.path.join(hf, "model.safetensors")):
        sh([sys.executable, os.path.join(ROOT, "tools/make_tiny.py"), hf,
            "--format", fmt])
    shutil.rmtree(model, ignore_errors=True)
    sh([sys.executable, os.path.join(ROOT, "tools/convert.py"), hf, model] + flags)

    # reference: teacher-forced logits at every position, then greedy
    ref = Reference(model, ctx=CTX)
    ref_logits = [ref.forward(t) for t in IDS]
    ref_gen = []
    lg = ref_logits[-1]
    for _ in range(NGEN):
        t = int(np.argmax(lg))
        ref_gen.append(t)
        lg = ref.forward(t)

    eng = Engine(model)
    eng_logits = eng.tfall(IDS)
    eng_gen = eng.gen(NGEN)
    eng.close()

    worst = 0.0
    for i, (a, b) in enumerate(zip(ref_logits, eng_logits)):
        d = np.max(np.abs(a - b)) / (np.max(np.abs(a)) + 1e-9)
        worst = max(worst, float(d))
        if not np.allclose(a, b, rtol=2e-3, atol=2e-3):
            print(f"  position {i}: max rel diff {d:.3e}")
            print("  ref:", a[:8])
            print("  eng:", b[:8])
            raise SystemExit(f"[{name}] logits mismatch at position {i}")
    if eng_gen != ref_gen:
        print("  ref:", ref_gen)
        print("  eng:", eng_gen)
        raise SystemExit(f"[{name}] greedy generation mismatch")
    print(f"[validate] {name:14s} OK  (worst rel logit diff {worst:.2e}, "
          f"{len(IDS)} positions, {NGEN} greedy tokens match)")


def main():
    if not os.path.exists(MIDGED):
        raise SystemExit("build first: make")
    os.makedirs(TMP, exist_ok=True)
    for name, fmt, flags in CASES:
        run_case(name, fmt, flags)
    print("[validate] all cases passed")


if __name__ == "__main__":
    main()
