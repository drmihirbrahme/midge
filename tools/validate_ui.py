"""validate_ui — end-to-end test of the onboarding interface.

Drives tools/ui.py exactly like the browser app does:

  * the interface answers instantly, before any model or engine exists
  * /api/state: catalog with disk estimates, free-disk figure
  * /api/setup runs the pipeline: convert (with progress + live log)
    first, engine build at the end — verified by deleting ./midged and
    checking the job rebuilds it
  * the finished model appears in /api/state models
  * /api/serve starts the OpenAI server; CORS preflight passes; a
    streamed chat completion works through it (what the chat panel does)
  * a failing setup reports a clean, resumable error

Uses a local tiny checkpoint as the download source (the same code path
as Hugging Face, minus the network). Run via `make test-ui`.
"""
from __future__ import annotations
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = os.path.join(ROOT, ".test-tmp")


def sh(args):
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        raise SystemExit("command failed")


def req(base, path, body=None):
    r = urllib.request.Request(base + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=90) as f:
        return json.loads(f.read())


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def main():
    hf = os.path.join(TMP, "ui-hf")
    if not os.path.exists(os.path.join(hf, "model.safetensors")):
        sh([sys.executable, os.path.join(ROOT, "tools/make_tiny.py"), hf])
    shutil.rmtree(os.path.join(ROOT, "models", "ui-test"), ignore_errors=True)

    # build-at-end check: remove the engine so the job must rebuild it
    exe = os.path.join(ROOT, "midged")
    had_engine = os.path.exists(exe)
    if had_engine:
        os.rename(exe, exe + ".bak")

    port = free_port()
    ui = subprocess.Popen(
        [sys.executable, os.path.join(ROOT, "tools/ui.py"),
         "--port", str(port), "--no-browser"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    try:
        t0 = time.time()
        for _ in range(50):
            try:
                html = urllib.request.urlopen(base + "/", timeout=1).read()
                break
            except Exception:
                time.sleep(0.1)
        assert b"midge" in html
        st = req(base, "/api/state")
        assert st["engine_built"] is False
        assert any(m["name"] == "gpt-oss-20b" and m["disk_gb"] > 5
                   for m in st["catalog"])
        assert st["free_disk_gb"] > 0
        print(f"[validate_ui] interface up in {time.time()-t0:.1f}s, "
              f"catalog + disk estimates                OK")

        req(base, "/api/setup", {"model": "ui-test", "source": hf})
        saw_log = False
        for _ in range(300):
            j = req(base, "/api/job")
            saw_log = saw_log or bool(j["log"])
            if j["done"]:
                break
            time.sleep(0.5)
        assert j["done"] and not j["error"], j.get("error")
        assert [s["status"] for s in j["steps"]] == ["done", "done", "done"]
        assert saw_log and j["pct"] == 100.0
        assert os.path.exists(exe), "engine was not built at the end"
        print("[validate_ui] setup job: convert -> build-at-end -> ready  OK")

        st = req(base, "/api/state")
        assert any(m["name"] == "ui-test" for m in st["models"])
        assert st["engine_built"] is True
        print("[validate_ui] converted model listed, engine built       OK")

        r = req(base, "/api/serve", {"model": "ui-test", "ctx": 220,
                                     "port": free_port()})
        sport = r["port"]
        o = urllib.request.Request(
            f"http://127.0.0.1:{sport}/v1/chat/completions", method="OPTIONS")
        with urllib.request.urlopen(o, timeout=5) as f:
            assert f.status == 204
            assert f.headers["Access-Control-Allow-Origin"] == "*"
        body = json.dumps({"model": "ui-test", "stream": True, "max_tokens": 8,
                           "temperature": 0,
                           "messages": [{"role": "user", "content": "hi"}]})
        rq = urllib.request.Request(
            f"http://127.0.0.1:{sport}/v1/chat/completions",
            data=body.encode(), headers={"Content-Type": "application/json"})
        raw = urllib.request.urlopen(rq, timeout=120).read().decode()
        assert "data:" in raw and "[DONE]" in raw
        print("[validate_ui] serve handoff + CORS + streamed chat        OK")
        req(base, "/api/stop_server", {})

        # failure path: unreachable source reports a clean resumable error
        req(base, "/api/setup", {"model": "bad", "source": "no/such-repo"})
        for _ in range(120):
            j = req(base, "/api/job")
            if j["done"]:
                break
            time.sleep(0.5)
        assert j["error"] and "resumable" in j["error"]
        print("[validate_ui] failing download -> clean resumable error   OK")
        print("[validate_ui] all interface tests passed")
    finally:
        ui.terminate()
        ui.wait(timeout=10)
        if had_engine and not os.path.exists(exe):
            os.rename(exe + ".bak", exe)
        elif had_engine and os.path.exists(exe + ".bak"):
            os.remove(exe + ".bak")
        shutil.rmtree(os.path.join(ROOT, "models", "ui-test"),
                      ignore_errors=True)
        shutil.rmtree(os.path.join(ROOT, "models", "bad"), ignore_errors=True)


if __name__ == "__main__":
    main()
