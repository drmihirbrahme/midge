"""validate_server — end-to-end test of the OpenAI-compatible server.

Self-contained: builds a tiny model, spawns `midge serve` on a free
port, and drives it with the official `openai` client (plus raw HTTP
for the legacy endpoint). Verifies:

  * /v1/models, /health
  * non-streaming chat completions: shape, usage, finish_reason=length
  * SSE streaming: chunk framing, deltas, terminal [DONE]
  * finish_reason=stop on the model's own stop tokens
  * session prefix caching: an append-only follow-up prefills only the
    new turn (prompt_tokens stays small)
  * divergent history: transparent engine restart, correct output
  * stop strings; /v1/completions

Run via `make test-server`. Needs: numpy, tokenizers, openai.
"""
from __future__ import annotations
import json
import os
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
        raise SystemExit("command failed: " + " ".join(map(str, args)))


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="c", choices=["c", "mlx"])
    opts = ap.parse_args()
    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("pip install openai")

    model_dir = os.path.join(TMP, "srv-model")
    if not os.path.exists(os.path.join(model_dir, "dense.midge")):
        hf = os.path.join(TMP, "srv-hf")
        sh([sys.executable, os.path.join(ROOT, "tools/make_tiny.py"), hf])
        sh([sys.executable, os.path.join(ROOT, "tools/convert.py"), hf,
            model_dir, "--experts", "q8r"])

    port = free_port()
    srv = subprocess.Popen(
        [sys.executable, os.path.join(ROOT, "tools/serve.py"), model_dir,
         "--port", str(port), "--ctx", "220", "--backend", opts.backend]
        + (["--device", "cpu", "--dense-bits", "32"]
           if opts.backend == "mlx" else []),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(50):
            try:
                urllib.request.urlopen(base + "/health", timeout=1)
                break
            except Exception:
                time.sleep(0.2)
        else:
            raise SystemExit("server did not come up")

        c = OpenAI(base_url=base + "/v1", api_key="midge")
        m = c.models.list().data[0].id
        assert m == "srv-model", m
        one = c.models.retrieve(m)
        assert one.id == m
        try:
            c.models.retrieve("nope")
            raise SystemExit("retrieve of unknown model should 404")
        except Exception:
            pass
        print("[validate_server] /v1/models + /v1/models/{model}       OK")

        sys_u = [{"role": "system", "content": "s"},
                 {"role": "user", "content": "hi"}]
        r = c.chat.completions.create(model=m, max_tokens=10, temperature=0,
                                      messages=sys_u)
        assert r.object == "chat.completion"
        assert r.choices[0].finish_reason == "length"
        assert r.usage.completion_tokens == 10 and r.usage.prompt_tokens > 5
        print("[validate_server] non-stream + usage + finish=length  OK")

        chunks, saw_done_role, text = 0, False, []
        stream = c.chat.completions.create(model=m, max_tokens=10,
                                           temperature=0, stream=True,
                                           messages=sys_u)
        fin = None
        for ev in stream:
            chunks += 1
            if ev.choices:
                d = ev.choices[0].delta
                if d.role == "assistant":
                    saw_done_role = True
                if d.content:
                    text.append(d.content)
                fin = ev.choices[0].finish_reason or fin
        assert chunks >= 3 and saw_done_role and fin == "length"
        print(f"[validate_server] SSE streaming ({chunks} chunks)        OK")

        msgs = [{"role": "system", "content": "s"},
                {"role": "user", "content": "a"}]
        for attempt in range(8):
            r = c.chat.completions.create(model=m, max_tokens=150,
                                          temperature=1.0, seed=attempt,
                                          messages=msgs)
            if r.choices[0].finish_reason == "stop":
                break
        assert r.choices[0].finish_reason == "stop", "no natural stop in 8 tries"
        p1 = r.usage.prompt_tokens
        msgs2 = msgs + [{"role": "assistant",
                         "content": r.choices[0].message.content},
                        {"role": "user", "content": "b"}]
        r2 = c.chat.completions.create(model=m, max_tokens=8, temperature=0,
                                       messages=msgs2)
        det = r2.usage.prompt_tokens_details
        cached = det.cached_tokens if det else 0
        assert cached > 0, "expected cached_tokens > 0 on the append turn"
        assert r2.usage.prompt_tokens > p1, \
            "prompt_tokens must count the whole conversation"
        print(f"[validate_server] prefix cache ({cached} cached of "
              f"{r2.usage.prompt_tokens} prompt tokens)  OK")

        r3 = c.chat.completions.create(model=m, max_tokens=8, temperature=0,
            messages=[{"role": "system", "content": "s"},
                      {"role": "user", "content": "totally different"}])
        assert r3.choices[0].finish_reason in ("length", "stop")
        print("[validate_server] divergent history restart          OK")

        r4 = c.chat.completions.create(model=m, max_tokens=60, temperature=0,
                                       stop=["|"], messages=sys_u)
        assert "|" not in (r4.choices[0].message.content or "")
        print("[validate_server] stop strings                        OK")

        # reasoning controls: effort accepted; enable_thinking=False must
        # strip reasoning_content from the response
        r5 = c.chat.completions.create(model=m, max_tokens=30, temperature=1.0,
            extra_body={"reasoning_effort": "high"}, messages=sys_u)
        assert r5.choices[0].finish_reason in ("stop", "length")
        r6 = c.chat.completions.create(model=m, max_tokens=30, temperature=1.0,
            extra_body={"enable_thinking": False}, messages=sys_u)
        assert getattr(r6.choices[0].message, "reasoning_content", None) is None
        r7 = c.chat.completions.create(model=m, max_tokens=30, temperature=1.0,
            extra_body={"reasoning_effort": "none"}, messages=sys_u)
        assert getattr(r7.choices[0].message, "reasoning_content", None) is None
        print("[validate_server] reasoning_effort / enable_thinking    OK")

        req = urllib.request.Request(base + "/v1/completions",
            data=json.dumps({"prompt": "hi", "max_tokens": 6}).encode(),
            headers={"Content-Type": "application/json"})
        d = json.load(urllib.request.urlopen(req, timeout=60))
        assert d["object"] == "text_completion"
        print("[validate_server] /v1/completions                     OK")
        print("[validate_server] all server tests passed")
    finally:
        srv.terminate()
        srv.wait(timeout=10)


if __name__ == "__main__":
    main()
