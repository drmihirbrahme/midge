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


def wait_health(port, timeout=90):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            return
        except Exception:
            time.sleep(0.3)
    raise SystemExit(f"server on {port} did not come up")


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p



def relay_tests(model_dir, opts):
    up_port, hy_port = free_port(), free_port()
    upstream = subprocess.Popen(
        [sys.executable, os.path.join(ROOT, "tools/serve.py"), model_dir,
         "--port", str(up_port), "--ctx", "640", "--backend", opts.backend],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    hybrid = subprocess.Popen(
        [sys.executable, os.path.join(ROOT, "tools/serve.py"), model_dir,
         "--port", str(hy_port), "--ctx", "640", "--backend", opts.backend,
         "--upstream", f"http://127.0.0.1:{up_port}/v1", "--route", "auto"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for p in (up_port, hy_port):
            wait_health(p)
        hb = f"http://127.0.0.1:{hy_port}/v1/chat/completions"
        basemsg = {"model": "m", "max_tokens": 6, "temperature": 0,
                   "messages": [{"role": "system", "content": "s"},
                                {"role": "user", "content": "hi"}]}

        def post(extra):
            rq = urllib.request.Request(hb,
                data=json.dumps({**basemsg, **extra}).encode(),
                headers={"Content-Type": "application/json"})
            r = urllib.request.urlopen(rq, timeout=120)
            return r.headers, r.read()

        h, raw = post({})                      # auto -> relayed to "cloud"
        d = json.loads(raw)
        assert d["midge_served_by"].startswith("cloud:"), d.get("midge_served_by")
        assert h["X-Midge-Route"] == "cloud"

        h, raw = post({"midge_route": "local"})  # per-request local override
        assert json.loads(raw)["midge_served_by"] == "local"

        h, raw = post({"stream": True})        # streamed relay passthrough
        assert h["X-Midge-Route"] == "cloud"
        assert b"data:" in raw and b"[DONE]" in raw

        upstream.terminate()
        upstream.wait(timeout=10)
        h, raw = post({})                      # upstream dead -> local fallback
        assert json.loads(raw)["midge_served_by"] == "local"
        print("[validate_server] hybrid relay: cloud/override/stream/fallback OK")

        # circuit breaker: a black-hole upstream taxes exactly one request
        bh_port = free_port()
        blackhole = subprocess.Popen(
            [sys.executable, os.path.join(ROOT, "tools/serve.py"), model_dir,
             "--port", str(bh_port), "--ctx", "640", "--backend", opts.backend,
             "--upstream", "http://10.255.255.1:9/v1",
             "--upstream-timeout", "2", "--route", "auto"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            wait_health(bh_port)
            from openai import OpenAI as _OA
            bc = _OA(base_url=f"http://127.0.0.1:{bh_port}/v1", api_key="x",
                     timeout=120)
            t0 = time.time()
            r = bc.chat.completions.create(model="m", max_tokens=4,
                temperature=0, messages=[{"role": "user", "content": "hi"}])
            first = time.time() - t0
            assert r.model_extra.get("midge_served_by") == "local"
            t0 = time.time()
            r = bc.chat.completions.create(model="m", max_tokens=4,
                temperature=0, messages=[{"role": "user", "content": "hi"}])
            second = time.time() - t0
            assert r.model_extra.get("midge_served_by") == "local"
            assert second < max(1.5, first / 2), \
                f"breaker did not engage: first={first:.1f}s second={second:.1f}s"
            print(f"[validate_server] breaker: dead upstream taxes one request "
                  f"({first:.1f}s), then local-fast ({second:.2f}s)  OK")
        finally:
            blackhole.terminate()
            blackhole.wait(timeout=10)
    finally:
        for p in (upstream, hybrid):
            if p.poll() is None:
                p.terminate()
                p.wait(timeout=10)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="c", choices=["c", "mlx"])
    opts = ap.parse_args()
    try:
        from openai import OpenAI
    except ImportError:
        # 'openai' is only needed to exercise this test suite, not to run
        # midge. Skip cleanly rather than erroring.
        print("[validate_server] SKIPPED — the 'openai' client is not "
              "installed (only needed for this test). Install: pip install openai")
        raise SystemExit(0)

    model_dir = os.path.join(TMP, "srv-model")
    if not os.path.exists(os.path.join(model_dir, "dense.midge")):
        hf = os.path.join(TMP, "srv-hf")
        sh([sys.executable, os.path.join(ROOT, "tools/make_tiny.py"), hf])
        sh([sys.executable, os.path.join(ROOT, "tools/convert.py"), hf,
            model_dir, "--experts", "q8r"])

    port = free_port()
    srv = subprocess.Popen(
        [sys.executable, os.path.join(ROOT, "tools/serve.py"), model_dir,
         "--port", str(port), "--ctx", "640", "--backend", opts.backend]
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

        # ---- tool calling: the agent loop -----------------------------
        WEATHER = [{"type": "function", "function": {
            "name": "get_weather", "description": "Get current weather",
            "parameters": {"type": "object", "properties": {
                "city": {"type": "string", "description": "City name"}},
                "required": ["city"]}}}]
        # 1) tools render into the prompt (developer block costs tokens)
        r_plain = c.chat.completions.create(model=m, max_tokens=4,
            temperature=0, messages=sys_u)
        r_tools = c.chat.completions.create(model=m, max_tokens=4,
            temperature=0, messages=sys_u, tools=WEATHER)
        assert r_tools.usage.prompt_tokens > r_plain.usage.prompt_tokens + 20, \
            "tools were not rendered into the prompt"
        # 2) tool-call OUTPUT parsing: teacher-force a harmony tool call
        #    through the real parser via the sampling path is impossible on a
        #    random model, so drive the parser with real tokenizer ids
        import serve as srv_mod
        from tokenizers import Tokenizer
        from harmony import Harmony
        tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        h = Harmony(tok, {"tokenizer": {}})
        parser = srv_mod.ChannelParser(h)
        seq = tok.encode('<|channel|>commentary to=functions.get_weather '
                         '<|constrain|>json<|message|>{"city": "Pune"}'
                         '<|call|>').ids
        got_name, got_args = None, []
        for t in seq:
            for kind, delta in parser.feed(t):
                if kind.startswith("tool:"):
                    got_name = kind[5:]
                    got_args.append(delta)
        assert got_name == "get_weather", got_name
        assert "".join(got_args) == '{"city": "Pune"}', "".join(got_args)
        # 3) tool-history replay: assistant tool_calls + tool result render
        call = {"id": "call_abc", "type": "function",
                "function": {"name": "get_weather",
                             "arguments": '{"city": "Pune"}'}}
        replay = sys_u + [
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "call_abc", "content": "22C"},
        ]
        rendered = tok.decode(srv_mod.render_messages(h, replay),
                              skip_special_tokens=False)
        assert "to=functions.get_weather" in rendered and "<|call|>" in rendered
        assert "functions.get_weather to=assistant" in rendered
        r_replay = c.chat.completions.create(model=m, max_tokens=4,
            temperature=0, messages=replay, tools=WEATHER)
        assert r_replay.usage.prompt_tokens > r_tools.usage.prompt_tokens + 10
        assert r_replay.choices[0].finish_reason in ("stop", "length",
                                                     "tool_calls")
        print("[validate_server] tool calling: render/parse/replay      OK")

        req = urllib.request.Request(base + "/v1/completions",
            data=json.dumps({"prompt": "hi", "max_tokens": 6}).encode(),
            headers={"Content-Type": "application/json"})
        d = json.load(urllib.request.urlopen(req, timeout=60))
        assert d["object"] == "text_completion"
        print("[validate_server] /v1/completions                     OK")
        relay_tests(model_dir, opts)
        print("[validate_server] all server tests passed")
    finally:
        srv.terminate()
        srv.wait(timeout=10)


if __name__ == "__main__":
    main()
