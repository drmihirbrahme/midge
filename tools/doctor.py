"""doctor — can this model run on this device?

    ./midge check openai/gpt-oss-120b
    ./midge check mistralai/Mixtral-8x7B-Instruct-v0.1
    ./midge check /path/to/checkpoint-or-converted-model

Answers two independent questions and combines them into a verdict:

  1. COMPATIBLE?  Is the architecture inside the family the spec-driven
     engine implements? Decided by tools/spec_from_hf.py from the
     model's config.json alone (for HF repos, only that one small file
     is fetched — no weights). Incompatibilities come back as specific
     reasons ("MLA is not implemented"), not a shrug.

  2. FITS?        Measured against *this* device: free disk vs
     container size (+ conversion scratch), available RAM vs the
     resident set (dense trunk + KV cache + headroom), and measured
     disk & CPU throughput turned into honest tok/s estimates.

Also usable as a library:  from doctor import analyze
"""
from __future__ import annotations
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import midgepack as mp
from spec_from_hf import spec_from_hf, Unsupported

GiB = 1 << 30


# ---------------------------------------------------------- the config
def fetch_config(source: str) -> tuple[dict, dict | None]:
    """Returns (config, spec_if_already_converted)."""
    if os.path.isdir(source):
        sp = os.path.join(source, "spec.json")
        cp = os.path.join(source, "config.json")
        spec = json.load(open(sp)) if os.path.exists(sp) else None
        if os.path.exists(cp):
            return json.load(open(cp)), spec
        if spec:
            return {}, spec
        raise FileNotFoundError(f"no config.json or spec.json in {source}")
    url = f"https://huggingface.co/{source}/resolve/main/config.json"
    with urllib.request.urlopen(url, timeout=30) as f:
        return json.load(f), None


# ---------------------------------------------------------- the device
def device_info() -> dict:
    d = {"cpu_cores": os.cpu_count() or 1}
    try:
        info = {}
        for ln in open("/proc/meminfo"):
            k, v = ln.split(":")
            info[k] = int(v.split()[0]) * 1024
        d["ram_total_gb"] = info["MemTotal"] / GiB
        d["ram_available_gb"] = info.get("MemAvailable", info["MemTotal"]) / GiB
    except FileNotFoundError:                      # macOS
        d["ram_total_gb"] = os.sysconf("SC_PAGE_SIZE") * \
            os.sysconf("SC_PHYS_PAGES") / GiB
        d["ram_available_gb"] = d["ram_total_gb"] * 0.6
    d["free_disk_gb"] = shutil.disk_usage(os.getcwd()).free / GiB
    return d


def bench_cpu_gbps(seconds=0.6) -> tuple[float, bool]:
    """Quantized-weight throughput of the decode hot loop. Uses the real
    engine kernel (midged --bench) when built; else a rough numpy-based
    estimate (flagged)."""
    exe = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "midged")
    if os.path.exists(exe):
        import subprocess
        out = subprocess.run([exe, "--bench"], capture_output=True, text=True,
                             timeout=30).stdout
        for tok in out.split():
            if tok.startswith("q4g32_gbps="):
                return float(tok.split("=")[1]), True
    import numpy as np
    w = np.random.default_rng(0).standard_normal((2048, 4096)).astype(np.float32)
    x = np.ones(4096, np.float32)
    n, t0 = 0, time.time()
    while time.time() - t0 < seconds:
        w @ x
        n += 1
    # scalar 4-bit kernels reach a fraction of raw f32 GEMV bandwidth
    return n * w.nbytes / (time.time() - t0) / GiB * 0.08, False


def bench_disk_mbps(path=".", size_mb=192) -> tuple[float, bool]:
    """Sequential read throughput; O_DIRECT when the OS allows it (true
    cold reads), otherwise a cached upper bound (flagged)."""
    fd, tmp = tempfile.mkstemp(dir=path, suffix=".midgebench")
    try:
        blk = os.urandom(1 << 20)
        with os.fdopen(fd, "wb") as f:
            for _ in range(size_mb):
                f.write(blk)
            f.flush()
            os.fsync(f.fileno())
        direct = False
        try:
            rfd = os.open(tmp, os.O_RDONLY | os.O_DIRECT)
            direct = True
        except (AttributeError, OSError):
            rfd = os.open(tmp, os.O_RDONLY)
        try:
            import mmap as _m
            buf = _m.mmap(-1, 1 << 20)
            t0 = time.time()
            got = 0
            while True:
                r = os.readv(rfd, [buf])
                if r <= 0:
                    break
                got += r
            dt = time.time() - t0
        finally:
            os.close(rfd)
        return got / (1 << 20) / max(dt, 1e-6), direct
    finally:
        os.unlink(tmp)


# ---------------------------------------------------------- the verdict
def analyze(source: str, ctx: int = 8192, expert_dt: str = "mxfp4",
            run_benchmarks: bool = True) -> dict:
    rep = {"source": source, "device": device_info()}
    try:
        cfg, spec = fetch_config(source)
        if spec is None:
            spec = spec_from_hf(cfg)
        rep["compatible"] = True
        rep["arch"] = spec["arch"]
        rep["reasons"] = []
    except Unsupported as e:
        rep.update(compatible=False, arch=e.arch, reasons=e.reasons,
                   verdict="no")
        return rep
    except Exception as e:
        rep.update(compatible=False, arch="?", verdict="unknown",
                   reasons=[f"could not read model config: {e}"])
        return rep

    s = spec
    lay = mp.ExpertLayout(expert_dt, s["hidden"], s["moe"]["ffn"],
                          s["moe"]["experts"], s["n_layers"])
    mats = 2 * s["vocab"] * s["hidden"] + s["n_layers"] * (
        2 * s["n_heads"] * s["head_dim"] * s["hidden"]
        + 2 * s["n_kv_heads"] * s["head_dim"] * s["hidden"]
        + s["moe"]["experts"] * s["hidden"])
    dense_gb = mats * 1.03125 / GiB                      # q8r
    experts_gb = lay.total / GiB
    win = s["attn"].get("sliding_window", 0)
    kv = sum(2 * (win if ("sliding" in lt and 0 < win < ctx) else ctx)
             * s["n_kv_heads"] * s["head_dim"] * 2
             for lt in s["attn"]["layer_types"]) / GiB
    per_tok_gb = (s["n_layers"] * s["moe"]["top_k"] * lay.expert_stride) / GiB

    size = {"experts_gb": round(experts_gb, 1), "dense_gb": round(dense_gb, 2),
            "total_disk_gb": round(experts_gb + dense_gb, 1),
            "conversion_peak_gb": round(experts_gb + dense_gb + 6, 1),
            "resident_ram_gb": round(dense_gb + kv + 0.7, 2),
            "kv_gb_at_ctx": round(kv, 2), "ctx": ctx,
            "cold_io_per_token_gb": round(per_tok_gb, 2)}
    rep["size"] = size

    d = rep["device"]
    fits_disk = d["free_disk_gb"] >= size["conversion_peak_gb"]
    fits_disk_tight = d["free_disk_gb"] >= size["total_disk_gb"] + 2
    fits_ram = d["ram_available_gb"] >= size["resident_ram_gb"] + 0.5
    checks = {"disk": fits_disk or fits_disk_tight, "ram": fits_ram}

    if run_benchmarks:
        cpu, cpu_real = bench_cpu_gbps()
        disk_mbps, direct = bench_disk_mbps()
        warm_bytes = dense_gb + s["moe"]["top_k"] * s["n_layers"] * \
            lay.expert_stride / GiB
        rep["speed"] = {
            "cpu_kernel_gbps": round(cpu, 2),
            "cpu_measured_real_kernel": cpu_real,
            "disk_read_mbps": round(disk_mbps),
            "disk_measured_cold": direct,
            "est_tok_s_warm": round(cpu / warm_bytes, 2),
            "est_tok_s_cold": round(min(cpu / warm_bytes,
                                        disk_mbps / 1024 / per_tok_gb), 2),
        }

    problems = []
    if not checks["disk"]:
        problems.append(f"needs ~{size['total_disk_gb']} GB disk "
                        f"({size['conversion_peak_gb']} GB during conversion); "
                        f"only {d['free_disk_gb']:.0f} GB free")
    elif not fits_disk:
        problems.append(f"disk is tight: conversion peaks near "
                        f"{size['conversion_peak_gb']} GB vs "
                        f"{d['free_disk_gb']:.0f} GB free — convert with "
                        "--keep-shards off (default) and nothing else running")
    if not checks["ram"]:
        problems.append(f"resident set ~{size['resident_ram_gb']} GB exceeds "
                        f"available RAM {d['ram_available_gb']:.1f} GB — lower "
                        f"--ctx (KV is {size['kv_gb_at_ctx']} GB at ctx {ctx})")
    rep["problems"] = problems
    rep["verdict"] = ("no" if not checks["disk"] or not checks["ram"]
                      else "tight" if (problems or
                                       rep.get("speed", {}).get("est_tok_s_cold", 1) < 0.2)
                      else "yes")
    return rep


def pretty(rep: dict) -> str:
    L = [f"model    {rep['source']}  ·  arch: {rep.get('arch', '?')}"]
    if not rep["compatible"]:
        if rep.get("verdict") == "unknown":
            L.append("verdict  ? could not check:")
        else:
            L.append("verdict  ✗ cannot run — architecture not supported:")
        L += [f"           - {r}" for r in rep["reasons"]]
        return "\n".join(L)
    s, d = rep["size"], rep["device"]
    L.append(f"needs    {s['total_disk_gb']} GB disk "
             f"(peaks {s['conversion_peak_gb']} GB converting) · "
             f"{s['resident_ram_gb']} GB resident RAM at ctx {s['ctx']}")
    L.append(f"device   {d['free_disk_gb']:.0f} GB disk free · "
             f"{d['ram_available_gb']:.1f}/{d['ram_total_gb']:.1f} GB RAM available · "
             f"{d['cpu_cores']} cores")
    if "speed" in rep:
        sp = rep["speed"]
        cold = "" if sp["disk_measured_cold"] else " (cached-read estimate)"
        kern = "" if sp["cpu_measured_real_kernel"] else " (estimate; build ./midged for a real measurement)"
        L.append(f"measured kernel {sp['cpu_kernel_gbps']} GB/s{kern} · "
                 f"disk {sp['disk_read_mbps']} MB/s{cold}")
        L.append(f"estimate ~{sp['est_tok_s_warm']} tok/s warm · "
                 f"~{sp['est_tok_s_cold']} tok/s cold "
                 f"({s['cold_io_per_token_gb']} GB read/token cold)")
    mark = {"yes": "✓ can run", "tight": "△ can run, tightly", "no": "✗ cannot run"}
    L.append(f"verdict  {mark[rep['verdict']]}")
    L += [f"           - {p}" for p in rep["problems"]]
    return "\n".join(L)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="midge check")
    ap.add_argument("source", help="HF repo id, checkpoint dir, or model dir")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-bench", action="store_true")
    a = ap.parse_args(argv)
    rep = analyze(a.source, ctx=a.ctx, run_benchmarks=not a.no_bench)
    print(json.dumps(rep, indent=2) if a.json else pretty(rep))
    sys.exit(0 if rep["verdict"] in ("yes", "tight") else 1)


if __name__ == "__main__":
    main()
