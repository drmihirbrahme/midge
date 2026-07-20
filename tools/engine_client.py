"""engine_client — shared wrapper around the midged subprocess.

Speaks the engine's line protocol (ids:/gen:/quit -> OK/T/DONE) and adds
the bookkeeping callers need: the pending-stop-token rule (a sampled
stop token is printed but never forwarded, so it must be resent at the
start of the next prefill), chunked generation (so callers can stop
early between chunks, e.g. on a stop string), and restart() for
resetting context (the engine itself is append-only).

Used by the ./midge CLI and by tools/serve.py.
"""
from __future__ import annotations
import os
import subprocess
import time


class EngineError(RuntimeError):
    pass


class EngineProc:
    def __init__(self, exe: str, model_dir: str, ctx: int, temp: float,
                 topp: float, seed: int, stop_ids, preload_gb: float = 0, load_timeout=600.0):
        if not os.path.exists(exe):
            raise EngineError("engine not built — run `make` first")
        self._args = [exe, model_dir, "--ctx", str(ctx), "--temp", str(temp),
                      "--topp", str(topp), "--seed", str(seed)]
        if stop_ids:
            self._args += ["--stop", ",".join(map(str, stop_ids))]
        if preload_gb:
            self._args += ["--preload-gb", str(preload_gb)]
        self.ctx = ctx
        self.load_timeout = load_timeout
        self._start()

    def _start(self):
        self.p = subprocess.Popen(self._args, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True, bufsize=1)
        # Wait for READY, but never block forever. The engine prints
        # LOADING immediately, then READY once mapped. A large model can
        # take a while, so we allow generous time but fail cleanly if the
        # process dies or goes silent past the deadline.
        deadline = time.time() + self.load_timeout
        while True:
            if self.p.poll() is not None:
                raise EngineError(
                    "engine exited during load (rc="
                    f"{self.p.returncode}) — is this a valid converted model "
                    "dir with dense.midge + experts.midge?")
            line = self._readline_timeout(max(1.0, deadline - time.time()))
            if line is None:
                if time.time() > deadline:
                    self.close()
                    raise EngineError(
                        f"engine did not become ready within "
                        f"{self.load_timeout:.0f}s — the model may be too "
                        "large for this machine's RAM/disk; try a smaller "
                        "model or run `midge check <dir>`")
                continue
            if line == "READY":
                break
            if line == "LOADING" or line.startswith("#"):
                continue          # alive; keep waiting
        self.pending = []
        self.n_ctx = 0

    def _readline_timeout(self, timeout):
        """Read one line from the engine, or None if nothing arrived in
        `timeout` seconds. Uses select so a hung engine can't block us."""
        import select
        r, _, _ = select.select([self.p.stdout], [], [], timeout)
        if not r:
            return None
        ln = self.p.stdout.readline()
        if not ln:
            raise EngineError("engine closed its output during load")
        return ln.rstrip("\n")

    def _line(self):
        ln = self.p.stdout.readline()
        if not ln:
            raise EngineError("engine exited unexpectedly")
        return ln.rstrip("\n")

    def restart(self):
        """Reset context by restarting the process (mmap makes this cheap)."""
        self.close()
        self._start()

    def prefill(self, ids):
        need = self.n_ctx + len(ids) + 2
        if need > self.ctx:
            raise EngineError(
                f"prompt needs {need} tokens but the context window is "
                f"{self.ctx} — increase --ctx (and leave room to generate)")
        ids = self.pending + list(ids)
        self.pending = []
        if not ids:
            return
        self.p.stdin.write("ids: " + " ".join(map(str, ids)) + "\n")
        self.p.stdin.flush()
        ln = self._line()
        if not ln.startswith("OK"):
            raise EngineError(ln)
        self.n_ctx += len(ids)

    def generate(self, ngen: int, stop_ids, on_token, chunk: int = 0):
        """Generate up to ngen tokens, calling on_token(t) for each.
        Stops early on a stop token, or when on_token returns False
        (checked between chunks of `chunk` tokens; 0 = single chunk).
        Returns the last sampled token (or None)."""
        stop_ids = set(stop_ids)
        left, last, keep_going = ngen, None, True
        step = chunk if chunk > 0 else ngen
        while left > 0 and keep_going:
            n = min(step, left)
            self.p.stdin.write(f"gen: {n}\n")
            self.p.stdin.flush()
            produced = 0
            while True:
                ln = self._line()
                if ln.startswith("T "):
                    last = int(ln[2:])
                    produced += 1
                    if on_token(last) is False:
                        keep_going = False
                elif ln.startswith("DONE"):
                    break
            left -= produced
            if last is not None and last in stop_ids:
                self.pending = [last]           # engine never forwarded it
                self.n_ctx += produced - 1
                return last
            self.n_ctx += produced
            if produced < n:                    # engine hit ctx limit
                break
        return last

    def close(self, grace=3.0):
        if self.p.poll() is not None:
            return
        try:
            self.p.stdin.write("quit\n")
            self.p.stdin.flush()
            self.p.wait(timeout=grace)
        except Exception:
            self.p.kill()
            try:
                self.p.wait(timeout=2)
            except Exception:
                pass

    def set_sampling(self, temp: float, topp: float, seed: int | None = None):
        """Retune sampling parameters without losing the KV context."""
        cmd = f"set: {temp} {topp}" + (f" {seed}" if seed is not None else "")
        self.p.stdin.write(cmd + "\n")
        self.p.stdin.flush()
        if not self._line().startswith("OK"):
            raise EngineError("set failed")
