"""ui — midge's zero-to-chat onboarding interface.

    ./midge ui                # opens http://127.0.0.1:8421

The interface starts *first*, instantly — before any model exists —
then walks through everything else with live progress:

  1. pick a model (from specs/catalog.json — the registry future
     models plug into) and see storage estimates vs free disk;
  2. download from Hugging Face and convert, shard by shard, with a
     live log and byte counter (interruptions resume automatically);
  3. build the engine at the end (skipped if already built);
  4. chat — the UI starts the OpenAI-compatible server and streams
     from it, so the same endpoint is immediately available to agents.

Stdlib only. State lives in one background job at a time; the frontend
polls /api/job. Endpoints:

  GET  /                 the app (webui/index.html)
  GET  /api/state        engine/models/catalog/server/job snapshot
  POST /api/setup        {"model": name, "source"?: local path}
  GET  /api/job          progress: steps, pct, log tail
  POST /api/serve        {"model": name, "port"?: 8420}
  POST /api/stop_server
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import re
import webbrowser

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import midgepack as mp  # noqa: E402

MODELS_DIR = mp.models_dir()


# ------------------------------------------------------------ catalog
def load_catalog():
    with open(os.path.join(ROOT, "specs", "catalog.json")) as f:
        cat = json.load(f)["models"]
    for m in cat:
        with open(os.path.join(ROOT, m["spec"])) as f:
            s = json.load(f)
        lay = mp.ExpertLayout("mxfp4", s["hidden"], s["moe"]["ffn"],
                              s["moe"]["experts"], s["n_layers"])
        m["disk_gb"] = round(lay.total / (1 << 30) + 2.2, 1)
        m["io_per_token_mb"] = round(
            s["n_layers"] * s["moe"]["top_k"] * lay.expert_stride / (1 << 20))
    return cat


def local_models():
    out = []
    if os.path.isdir(MODELS_DIR):
        for n in sorted(os.listdir(MODELS_DIR)):
            d = os.path.join(MODELS_DIR, n)
            if os.path.exists(os.path.join(d, "dense.midge")):
                sz = sum(os.path.getsize(os.path.join(d, f))
                         for f in os.listdir(d))
                out.append({"name": n, "size_gb": round(sz / (1 << 30), 2)})
    return out


# ---------------------------------------------------------------- job
class Job:
    def __init__(self, model_name, source):
        self.model = model_name
        self.source = source
        self.outdir = os.path.join(MODELS_DIR, model_name)
        self.steps = [{"name": "download & convert", "status": "pending"},
                      {"name": "build engine", "status": "pending"},
                      {"name": "ready", "status": "pending"}]
        self.log: list[str] = []
        self.pct = 0.0
        self.error = None
        self.done = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _emit(self, line):
        self.log.append(line.rstrip())
        del self.log[:-400]

    def _run(self):
        try:
            self._convert()
            self._build()
            self.steps[2]["status"] = "done"
            self.pct = 100.0
        except Exception as e:
            self.error = str(e)
            for s in self.steps:
                if s["status"] == "running":
                    s["status"] = "failed"
            self._emit(f"error: {e}")
        finally:
            self.done = True

    def _convert(self):
        st = self.steps[0]
        st["status"] = "running"
        os.makedirs(MODELS_DIR, exist_ok=True)
        p = subprocess.Popen(
            [sys.executable, os.path.join(ROOT, "tools/convert.py"),
             self.source, self.outdir],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            bufsize=1)
        shards = done = 0
        for line in p.stdout:
            self._emit(line)
            if "] " in line and "/" in line and "shard" not in line:
                # "[convert] [2/17] model-00002-of-00017.safetensors"
                try:
                    frac = line.split("[")[2].split("]")[0]
                    done, shards = int(frac.split("/")[0]) - 1, int(frac.split("/")[1])
                    self.pct = 90.0 * done / max(shards, 1)
                except (IndexError, ValueError):
                    pass
        if p.wait() != 0:
            raise RuntimeError(
                "conversion failed — see log. It is resumable: press "
                "Start again to continue from the last finished shard.")
        st["status"] = "done"
        self.pct = 90.0

    def _build(self):
        st = self.steps[1]
        st["status"] = "running"
        exe = mp.engine_path()
        if os.path.exists(exe):
            self._emit("engine already built — skipping")
        else:
            self._emit("$ make")
            p = subprocess.Popen(["make"], cwd=ROOT, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in p.stdout:
                self._emit(line)
            if p.wait() != 0 or not os.path.exists(exe):
                raise RuntimeError("engine build failed (is gcc installed?)")
        st["status"] = "done"
        self.pct = 98.0

    def snapshot(self):
        written = 0
        if os.path.isdir(self.outdir):
            written = sum(os.path.getsize(os.path.join(self.outdir, f))
                          for f in os.listdir(self.outdir)
                          if os.path.isfile(os.path.join(self.outdir, f)))
        return {"model": self.model, "steps": self.steps,
                "pct": round(self.pct, 1), "log": self.log[-60:],
                "written_gb": round(written / (1 << 30), 2),
                "error": self.error, "done": self.done}


# --------------------------------------------------------------- state
class State:
    def __init__(self, args):
        self.args = args
        self.job: Job | None = None
        self.server: subprocess.Popen | None = None
        self.server_model = None
        self.server_port = None
        self.lock = threading.Lock()

    def snapshot(self):
        free = shutil.disk_usage(ROOT).free
        return {
            "engine_built": os.path.exists(mp.engine_path()),
            "free_disk_gb": round(free / (1 << 30), 1),
            "catalog": load_catalog(),
            "models": local_models(),
            "job": self.job.snapshot() if self.job else None,
            "server": ({"model": self.server_model, "port": self.server_port}
                       if self.server and self.server.poll() is None else None),
        }

    @staticmethod
    def _safe_name(name):
        if not name or not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", name) \
                or name.startswith("."):
            raise ValueError("model name must be 1-64 chars of "
                             "letters/digits/._- (got %r)" % name)
        return name

    def start_setup(self, body):
        with self.lock:
            if self.job and not self.job.done:
                raise ValueError("a setup job is already running")
            name = body.get("model")
            cat = {m["name"]: m for m in load_catalog()}
            source = body.get("source")
            if not source:
                if name not in cat:
                    raise ValueError(f"unknown model {name!r}")
                source = cat[name]["hf_repo"]
            if not name:
                name = source.rstrip("/").split("/")[-1].lower()
            name = self._safe_name(name)
            self.job = Job(name, source)

    def start_server(self, body):
        with self.lock:
            self.stop_server()
            name = self._safe_name(body.get("model"))
            d = os.path.join(MODELS_DIR, name)
            if not os.path.exists(os.path.join(d, "dense.midge")):
                raise ValueError(f"model {name!r} is not converted yet")
            port = int(body.get("port") or 8420)
            self.server = subprocess.Popen(
                [sys.executable, os.path.join(ROOT, "tools/serve.py"), d,
                 "--port", str(port), "--ctx",
                 str(body.get("ctx") or self.args.serve_ctx)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.server_model, self.server_port = name, port
            deadline = time.time() + 60
            import urllib.request
            while time.time() < deadline:
                if self.server.poll() is not None:
                    raise RuntimeError("server exited — check the model dir")
                try:
                    urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/health", timeout=1)
                    return port
                except Exception:
                    time.sleep(0.3)
            raise RuntimeError("server did not come up in 60s")

    def stop_server(self):
        if self.server and self.server.poll() is None:
            self.server.terminate()
            try:
                self.server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.server.kill()
        self.server = None


# ----------------------------------------------------------------- http
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    S: State = None

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *a):
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(ROOT, "webui", "index.html"), "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/api/state":
            return self._json(200, self.S.snapshot())
        if self.path == "/api/job":
            return self._json(200, self.S.job.snapshot() if self.S.job
                              else {"done": True, "steps": [], "log": []})
        self._json(404, {"error": "no route"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return self._json(400, {"error": "bad JSON"})
        try:
            if self.path == "/api/search":
                import discover
                rows = discover.search(body.get("query"), body.get("family"),
                                       int(body.get("limit") or 24))
                return self._json(200, {"results": rows})
            if self.path == "/api/check":
                import doctor
                rep = doctor.analyze(body.get("source", ""),
                                     ctx=int(body.get("ctx") or 8192))
                return self._json(200, {"report": rep,
                                        "pretty": doctor.pretty(rep)})
            if self.path == "/api/setup":
                self.S.start_setup(body)
                return self._json(200, {"ok": True})
            if self.path == "/api/serve":
                port = self.S.start_server(body)
                return self._json(200, {"ok": True, "port": port})
            if self.path == "/api/stop_server":
                self.S.stop_server()
                return self._json(200, {"ok": True})
        except Exception as e:
            return self._json(400, {"error": str(e)})
        self._json(404, {"error": "no route"})


def main(argv=None):
    ap = argparse.ArgumentParser(prog="midge ui")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8421)
    ap.add_argument("--serve-ctx", type=int, default=8192)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)

    Handler.S = State(args)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"[ui] midge is ready at {url}", file=sys.stderr)
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        Handler.S.stop_server()


if __name__ == "__main__":
    main()
