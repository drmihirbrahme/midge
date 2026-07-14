"""validate_mlx — the MLX engine vs the NumPy reference.

Same discipline as tools/validate.py: tiny random checkpoints go through
the real converter, then MidgeMLX's logits are compared per position
against tools/reference.py, plus greedy generations token-for-token.

Runs on any platform (MLX ships a CPU backend for Linux):
    pip install "mlx[cpu]"     # or just mlx on macOS
    make test-mlx

Tolerances are wider than the C engine's because MLX computes matmuls
in f16/quantized kernels; greedy agreement is still required.
"""
from __future__ import annotations
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, ROOT)
from reference import Reference  # noqa: E402

try:
    from midge_mlx.model import MidgeMLX
except ImportError as e:
    raise SystemExit(f"mlx not installed ({e}) — pip install 'mlx[cpu]'")

TMP = os.path.join(ROOT, ".test-tmp")
IDS = [1, 17, 42, 5, 99, 3, 77, 12, 63, 8, 120, 31]
NGEN = 24
CTX = 64

CASES = [
    # (name, fixture format, convert flags, dense_bits, rtol)
    ("mlx q4g32",        "bf16",  ["--experts", "q4g32", "--dense", "q8r"], 32, 2e-3),
    ("mlx mxfp4-blocks", "mxfp4", ["--experts", "mxfp4", "--dense", "q8r"], 32, 2e-3),
    ("mlx bf16->mxfp4",  "bf16",  ["--experts", "mxfp4", "--dense", "q8r"], 32, 2e-3),
    ("mlx q8r experts",  "bf16",  ["--experts", "q8r",  "--dense", "q8r"], 32, 2e-3),
    ("mlx dense-8bit",   "bf16",  ["--experts", "q4g32", "--dense", "q8r"], 8, None),
]


def sh(args):
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        raise SystemExit("command failed: " + " ".join(map(str, args)))


def run_case(name, fmt, flags, dense_bits, rtol):
    hf = os.path.join(TMP, f"hf-{fmt}")
    model_dir = os.path.join(TMP, "mlxmodel")
    if not os.path.exists(os.path.join(hf, "model.safetensors")):
        sh([sys.executable, os.path.join(ROOT, "tools/make_tiny.py"), hf,
            "--format", fmt])
    shutil.rmtree(model_dir, ignore_errors=True)
    sh([sys.executable, os.path.join(ROOT, "tools/convert.py"), hf, model_dir]
       + flags)

    ref = Reference(model_dir, ctx=CTX)
    if rtol is None:
        # dense trunk is requantized in the MLX engine; give the reference
        # the *same* dequantized weights so the comparison is exact again
        import mlx.core as mx
        for nm, w in ref.dense.items():
            if w.ndim == 2 and ".router." not in nm:
                q = mx.quantize(mx.array(w), group_size=64, bits=dense_bits)
                ref.dense[nm] = np.array(
                    mx.dequantize(*q, group_size=64, bits=dense_bits),
                    dtype=np.float32)
        rtol = 2e-3
    ref_logits = [ref.forward(t) for t in IDS]
    ref_gen = []
    lg = ref_logits[-1]
    for _ in range(NGEN):
        t = int(np.argmax(lg))
        ref_gen.append(t)
        lg = ref.forward(t)

    m = MidgeMLX(model_dir, ctx=CTX, dense_bits=dense_bits, cache_gb=0.5)
    mlx_logits = [m.forward(t) for t in IDS]
    mlx_gen = list(m.generate([], NGEN, temp=0.0))
    # generate() with empty prompt starts from the logits of the last
    # forward, mirroring the reference loop above
    worst = 0.0
    for i, (a, b) in enumerate(zip(ref_logits, mlx_logits)):
        d = float(np.max(np.abs(a - b)) / (np.max(np.abs(a)) + 1e-9))
        worst = max(worst, d)
        if d > rtol:
            print("ref:", a[:8], "\nmlx:", b[:8])
            raise SystemExit(f"[{name}] logits mismatch at position {i} "
                             f"(rel {d:.2e} > {rtol})")
    if mlx_gen != ref_gen:
        print("ref:", ref_gen, "\nmlx:", mlx_gen)
        raise SystemExit(f"[{name}] greedy generation mismatch")
    print(f"[validate_mlx] {name:18s} OK  (worst rel diff {worst:.2e}, "
          f"{NGEN} greedy tokens match, "
          f"cache: {m.experts.hits} hits/{m.experts.loads} loads)")


def main():
    os.makedirs(TMP, exist_ok=True)
    for case in CASES:
        run_case(*case)
    print("[validate_mlx] all cases passed")


if __name__ == "__main__":
    main()
