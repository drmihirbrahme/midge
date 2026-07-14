"""verify_mxfp4 — check midge's MXFP4 decoding against transformers'.

midge transcodes gpt-oss MXFP4 blocks byte-for-byte and assumes:
  * low nibble = even column, high nibble = odd column,
  * FP4 E2M1 code -> value LUT [0, .5, 1, 1.5, 2, 3, 4, 6, -0, -.5, …],
  * e8m0 scale = 2^(u8 - 127).

tools/validate.py proves the engine matches midge's *own* dequant; this
script proves midge's dequant matches the *official* one, using one real
expert tensor from a checkpoint. Needs torch + transformers, but only
touches a single shard, so it is cheap to run.

Usage:
    python3 tools/verify_mxfp4.py /path/to/checkpoint     # local dir
    python3 tools/verify_mxfp4.py openai/gpt-oss-20b      # downloads 1 shard
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import midgepack as wp

TENSOR = "model.layers.0.mlp.experts.down_proj"   # any *_blocks tensor works


def find_shard(src: str, name: str) -> str:
    if os.path.isdir(src):
        idx = os.path.join(src, "model.safetensors.index.json")
        if os.path.exists(idx):
            with open(idx) as f:
                shard = json.load(f)["weight_map"][name + "_blocks"]
            return os.path.join(src, shard)
        return os.path.join(src, "model.safetensors")
    from huggingface_hub import hf_hub_download
    idx = hf_hub_download(src, "model.safetensors.index.json")
    with open(idx) as f:
        shard = json.load(f)["weight_map"][name + "_blocks"]
    return hf_hub_download(src, shard)


def midge_dequant(blocks: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """blocks [rows, cols/32, 16] u8, scales [rows, cols/32] u8 (e8m0)."""
    rows = blocks.shape[0]
    cols = blocks.shape[1] * 32
    codes = np.empty((rows, cols), np.uint8)
    flat = blocks.reshape(rows, cols // 2)
    codes[:, 0::2] = flat & 0x0F          # low nibble first
    codes[:, 1::2] = flat >> 4
    vals = wp.FP4_LUT[codes].reshape(rows, cols // 32, 32)
    s = np.exp2(scales.astype(np.float64) - 127.0).astype(np.float32)
    return (vals * s[:, :, None]).reshape(rows, cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="checkpoint dir or HF repo id")
    args = ap.parse_args()

    shard = find_shard(args.source, TENSOR)
    st = wp.SafeTensors(shard)
    blocks = st.get(TENSOR + "_blocks")[0]     # expert 0
    scales = st.get(TENSOR + "_scales")[0]
    ours = midge_dequant(np.asarray(blocks), np.asarray(scales))
    print(f"[verify] {TENSOR} expert 0: {ours.shape}, "
          f"|w| max {np.abs(ours).max():.3f} mean {np.abs(ours).mean():.4f}")

    try:
        import torch
        from transformers.integrations.mxfp4 import convert_moe_packed_tensors
    except ImportError:
        raise SystemExit(
            "install torch + transformers to compare against the official "
            "dequantizer:  pip install torch transformers")

    theirs = convert_moe_packed_tensors(
        torch.from_numpy(np.asarray(blocks))[None, ...],
        torch.from_numpy(np.asarray(scales))[None, ...])
    theirs = theirs.reshape(ours.shape[0], -1).float().numpy()

    if ours.shape != theirs.shape:
        raise SystemExit(f"FAIL: shape mismatch {ours.shape} vs {theirs.shape}")
    diff = np.abs(ours - theirs).max()
    print(f"[verify] max abs diff vs transformers: {diff:.3e}")
    if diff > 1e-6:
        # try the opposite nibble order to give a useful diagnosis
        rows, cols = ours.shape
        flat = np.asarray(blocks).reshape(rows, cols // 2)
        codes = np.empty((rows, cols), np.uint8)
        codes[:, 0::2] = flat >> 4
        codes[:, 1::2] = flat & 0x0F
        vals = wp.FP4_LUT[codes].reshape(rows, cols // 32, 32)
        s = np.exp2(np.asarray(scales).astype(np.float64) - 127.0).astype(np.float32)
        alt = (vals * s[:, :, None]).reshape(rows, cols)
        if np.abs(alt - theirs).max() <= 1e-6:
            raise SystemExit("FAIL: nibble order is HIGH-first on this "
                             "checkpoint — please open an issue")
        raise SystemExit("FAIL: dequantization mismatch — please open an issue")
    print("[verify] OK — midge's MXFP4 decoding matches transformers exactly")


if __name__ == "__main__":
    main()
