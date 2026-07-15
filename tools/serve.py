"""serve — a local OpenAI-compatible API server for midge.

    ./midge serve models/gpt-oss-20b --port 8420

exposes:

    GET  /v1/models
    POST /v1/chat/completions      (stream and non-stream)
    POST /v1/completions           (basic raw-prompt completions)
    GET  /health

so anything that speaks the OpenAI API — agent frameworks, the official
`openai` client libraries, curl — can drive midge:

    from openai import OpenAI
    client = OpenAI(base_url="http://127.0.0.1:8420/v1", api_key="midge")
    client.chat.completions.create(model="midge", messages=[...], stream=True)

Design notes (this is a CPU-streaming engine, not vLLM):

* One engine, one request at a time. Concurrent requests queue on a
  lock. Streaming is strongly recommended on slow hardware so clients
  see progress instead of timing out.
* **Session prefix caching.** The OpenAI API is stateless — clients
  resend the whole conversation every turn — but re-prefilling history
  is the most expensive thing a CPU engine can do. The server remembers
  the message list its context currently encodes; when a request merely
  *extends* it (the normal agent loop), only the new messages are
  prefilled. Anything else (edited history, a different conversation)
  transparently restarts the engine and prefills from scratch.
* gpt-oss "analysis" channel text is returned as `reasoning_content`
  (and streamed as `delta.reasoning_content`), the final channel as
  `content` — the convention agent frameworks already understand from
  other reasoning models.
* Stdlib only — no web framework required.

Implementation: stdlib http.server; harmony rendering shared with the
CLI (tools/harmony.py); engine access via tools/engine_client.py.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import threading
import time
import uuid

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harmony import Harmony
from engine_client import EngineProc, EngineError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ----------------------------------------------------------- harmony IO
class ChannelParser:
    """Incremental parser for generated harmony tokens: feeds back
    (channel, text_delta) pairs, decoding per-channel so multi-byte
    characters never split."""

    def __init__(self, h: Harmony):
        self.h = h
        self.rid = {v: k for k, v in h.id.items()}
        self.mode = "text"
        self.channel = "final"
        self.buf, self.printed = [], 0
        self.header = []

    def feed(self, t: int):
        """Returns list of (channel, delta) emitted by this token."""
        out = []
        name = self.rid.get(t)
        if name is None:
            if self.mode == "channel":
                self.header.append(t)
            else:
                self.buf.append(t)
                text = self.h.tok.decode(self.buf, skip_special_tokens=False)
                new = text[self.printed:]
                if new:
                    out.append((self.channel, new))
                    self.printed = len(text)
            return out
        if name == "<|channel|>":
            self.header, self.mode = [], "channel"
        elif name == "<|message|>":
            if self.mode == "channel":
                self.channel = self.h.tok.decode(
                    self.header, skip_special_tokens=False).strip()
            self.buf, self.printed, self.mode = [], 0, "text"
        elif name in ("<|end|>", "<|return|>", "<|call|>"):
            self.buf, self.printed = [], 0
            self.channel, self.mode = "final", "text"
        elif name == "<|start|>":
            self.buf, self.printed, self.mode = [], 0, "text"
        return out


# --------------------------------------------------------------- backend
class MlxAdapter:
    """Adapts midge_mlx.MidgeMLX to the EngineProc interface the
    Session expects (prefill/generate/set_sampling/restart/n_ctx)."""

    def __init__(self, args):
        sys.path.insert(0, ROOT)
        from midge_mlx.model import MidgeMLX
        self.m = MidgeMLX(args.model_dir, ctx=args.ctx,
                          dense_bits=args.dense_bits,
                          cache_gb=args.cache_gb, device=args.device)
        print(f"[serve] mlx backend on {self.m.device}, kernels: "
              f"{self.m.caps}", file=sys.stderr)
        self.pending, self.n_ctx = [], 0
        self.temp, self.topp, self.seed = 0.7, 0.9, args.seed

    def set_sampling(self, temp, topp, seed=None):
        self.temp, self.topp = temp, topp
        if seed is not None:
            self.seed = seed

    def prefill(self, ids):
        for t in self.pending + list(ids):
            self.m.forward(t)
            self.n_ctx += 1
        self.pending = []

    def generate(self, ngen, stop_ids, on_token, chunk=0):
        stop_ids = set(stop_ids)
        self.seed += 1
        last = None
        for t in self.m.generate([], ngen, temp=self.temp, topp=self.topp,
                                 stop_ids=stop_ids, seed=self.seed):
            last = t
            cont = on_token(t)
            if t in stop_ids or cont is False:
                # model.generate yields before forwarding; on stop or
                # early break the token was never forwarded
                self.pending = [t]
                return last
            self.n_ctx += 1
        return last

    def restart(self):
        self.m.reset()
        self.pending, self.n_ctx = [], 0



class Session:
    """One engine + the message list its KV context currently encodes."""

    def __init__(self, args):
        self.args = args
        with open(os.path.join(args.model_dir, "spec.json")) as f:
            self.spec = json.load(f)
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(
            os.path.join(args.model_dir, "tokenizer.json"))
        self.h = Harmony(self.tok, self.spec)
        if args.backend == "mlx":
            self.eng = MlxAdapter(args)
        else:
            self.eng = EngineProc(os.path.join(ROOT, "midged"), args.model_dir,
                                  args.ctx, 1.0, 1.0, args.seed,
                                  self.h.stop_ids, args.preload_gb)
        self.msgs = []             # messages currently in the engine context
        self.lock = threading.Lock()
        self.model_name = os.path.basename(os.path.normpath(args.model_dir))

    # -- context management ------------------------------------------
    def _sync(self, messages):
        """Bring the engine context to `messages` + generation prefix,
        prefilling as little as possible. Returns prompt token count."""
        n = len(self.msgs)
        if n and len(messages) > n and messages[:n] == self.msgs:
            delta = messages[n:]
        else:
            if self.msgs or self.eng.n_ctx:
                self.eng.restart()
            delta = messages
            self.msgs = []
        ids = self.h.render(delta, add_generation_prefix=True)
        self.eng.prefill(ids)
        self.msgs = self.msgs + list(delta)
        return len(ids)

    def chat(self, messages, max_tokens, temp, topp, stop_strs, on_event):
        """Run one chat turn. on_event(channel, delta) streams text.
        Returns (final_text, reasoning_text, finish_reason, usage)."""
        with self.lock:
            n_prompt = self._sync(messages)
            self.eng.set_sampling(0.7 if temp is None else float(temp),
                                  0.9 if topp is None else float(topp))
            parser = ChannelParser(self.h)
            final, reasoning = [], []
            state = {"stopped": False, "n_gen": 0}

            def on_token(t):
                state["n_gen"] += 1
                for ch, delta in parser.feed(t):
                    if ch == "final":
                        final.append(delta)
                        if stop_strs and any(ss in "".join(final)
                                             for ss in stop_strs):
                            state["stopped"] = True
                            return False
                        on_event("final", delta)
                    else:
                        reasoning.append(delta)
                        on_event("reasoning", delta)
                return True

            budget = min(max_tokens, self.args.ctx - self.eng.n_ctx - 1)
            if budget <= 0:
                raise EngineError("context full — shorten history or raise --ctx")
            last = self.eng.generate(budget, self.h.stop_ids, on_token,
                                     chunk=self.args.chunk)
            final_text = "".join(final)
            reasoning_text = "".join(reasoning)
            if state["stopped"]:
                fin = "stop"
                for ss in stop_strs:
                    i = final_text.find(ss)
                    if i >= 0:
                        final_text = final_text[:i]
                self.msgs = []
                self.eng.restart()      # context has extra tokens: resync later
            elif last in self.h.stop_ids:
                fin = "stop"            # clean end: context is reusable
                self.msgs = self.msgs + [
                    {"role": "assistant", "content": final_text}]
            else:
                fin = "length"          # partial reply in context: not reusable
                self.msgs = []
                self.eng.restart()
            usage = {"prompt_tokens": n_prompt,
                     "completion_tokens": state["n_gen"],
                     "total_tokens": n_prompt + state["n_gen"]}
            return final_text, reasoning_text, fin, usage


# ----------------------------------------------------------------- http
def now():
    return int(time.time())


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "midge"
    S: Session = None            # set at startup

    # -- helpers -------------------------------------------------------
    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _err(self, code, msg):
        self._json(code, {"error": {"message": msg, "type": "invalid_request_error"}})

    def _sse_start(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        # SSE bodies have no length framing: close delimits the stream
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.close_connection = True

    def _sse(self, obj):
        self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
        self.wfile.flush()

    def log_message(self, fmt, *a):
        print(f"[serve] {self.address_string()} {fmt % a}", file=sys.stderr)

    # -- routes --------------------------------------------------------
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers",
                         "Content-Type, Authorization")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            return self._json(200, {"status": "ok",
                                    "model": self.S.model_name,
                                    "context_tokens": self.S.eng.n_ctx})
        if self.path in ("/v1/models", "/models"):
            return self._json(200, {"object": "list", "data": [{
                "id": self.S.model_name, "object": "model",
                "created": now(), "owned_by": "midge"}]})
        if self.path.startswith(("/v1/models/", "/models/")):
            name = self.path.rsplit("/", 1)[1]
            if name == self.S.model_name:
                return self._json(200, {"id": name, "object": "model",
                                        "created": now(), "owned_by": "midge"})
            return self._err(404, f"model {name!r} not found")
        self._err(404, f"no route {self.path}")

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._err(400, f"bad JSON: {e}")
        if self.path in ("/v1/chat/completions", "/chat/completions"):
            return self.chat_completions(body)
        if self.path in ("/v1/completions", "/completions"):
            return self.completions(body)
        self._err(404, f"no route {self.path}")

    def chat_completions(self, body):
        msgs = body.get("messages")
        if not isinstance(msgs, list) or not msgs:
            return self._err(400, "messages: non-empty list required")
        msgs = [{"role": m.get("role", "user"),
                 "content": _content_str(m.get("content", ""))} for m in msgs]
        if msgs[0]["role"] != "system":
            msgs = [{"role": "system",
                     "content": self.S.args.system}] + msgs
        max_tokens = int(body.get("max_tokens")
                         or body.get("max_completion_tokens") or 1024)
        temp = body.get("temperature")
        topp = body.get("top_p")
        stop = body.get("stop") or []
        if isinstance(stop, str):
            stop = [stop]
        stream = bool(body.get("stream"))
        # reasoning controls: the standard reasoning_effort field sets the
        # harmony "Reasoning:" level; "none" hides the analysis channel.
        # enable_thinking (extension) explicitly shows/hides it.
        effort = body.get("reasoning_effort")
        include_reasoning = bool(body.get("include_reasoning", True))
        if effort == "none":
            include_reasoning = False
            effort = "low"
        if "enable_thinking" in body:
            include_reasoning = bool(body["enable_thinking"])
        if effort in ("low", "medium", "high"):
            sysmsg = msgs[0]
            if "Reasoning:" not in sysmsg["content"]:
                sysmsg["content"] += f"\nReasoning: {effort}"
        rid = "chatcmpl-" + uuid.uuid4().hex[:24]
        base = {"id": rid, "object": "chat.completion.chunk",
                "created": now(), "model": self.S.model_name}

        if stream:
            self._sse_start()
            self._sse({**base, "choices": [{"index": 0, "delta":
                      {"role": "assistant", "content": ""},
                      "finish_reason": None}]})

            def on_event(ch, delta):
                try:
                    if ch == "final":
                        self._sse({**base, "choices": [{"index": 0,
                                  "delta": {"content": delta},
                                  "finish_reason": None}]})
                    elif include_reasoning:
                        self._sse({**base, "choices": [{"index": 0,
                                  "delta": {"reasoning_content": delta},
                                  "finish_reason": None}]})
                except (BrokenPipeError, ConnectionResetError):
                    raise
            try:
                final, reasoning, fin, usage = self.S.chat(
                    msgs, max_tokens, temp, topp, stop, on_event)
            except (BrokenPipeError, ConnectionResetError):
                self.S.msgs = []
                self.S.eng.restart()
                return
            except EngineError as e:
                self._sse({**base, "choices": [],
                           "error": {"message": str(e)}})
                self.wfile.write(b"data: [DONE]\n\n")
                return
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
            self._sse({**base, "choices": [{"index": 0, "delta": {},
                      "finish_reason": fin}], "usage": usage})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return

        try:
            final, reasoning, fin, usage = self.S.chat(
                msgs, max_tokens, temp, topp, stop, lambda ch, d: None)
        except EngineError as e:
            return self._err(500, str(e))
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        message = {"role": "assistant", "content": final}
        if reasoning and include_reasoning:
            message["reasoning_content"] = reasoning
        self._json(200, {"id": rid, "object": "chat.completion",
                         "created": now(), "model": self.S.model_name,
                         "choices": [{"index": 0, "message": message,
                                      "finish_reason": fin}],
                         "usage": usage})

    def completions(self, body):
        prompt = body.get("prompt", "")
        if isinstance(prompt, list):
            prompt = prompt[0] if prompt else ""
        msgs = [{"role": "system", "content": self.S.args.system},
                {"role": "user", "content": str(prompt)}]
        max_tokens = int(body.get("max_tokens") or 256)
        stop = body.get("stop") or []
        if isinstance(stop, str):
            stop = [stop]
        try:
            final, _, fin, usage = self.S.chat(
                msgs, max_tokens, body.get("temperature"),
                body.get("top_p"), stop, lambda ch, d: None)
        except EngineError as e:
            return self._err(500, str(e))
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        self._json(200, {"id": "cmpl-" + uuid.uuid4().hex[:24],
                         "object": "text_completion", "created": now(),
                         "model": self.S.model_name,
                         "choices": [{"index": 0, "text": final,
                                      "finish_reason": fin,
                                      "logprobs": None}],
                         "usage": usage})


def _content_str(c):
    """Accept both plain-string and OpenAI content-part-list messages."""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(p.get("text", "") for p in c
                       if isinstance(p, dict) and p.get("type") == "text")
    return str(c)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="midge serve")
    ap.add_argument("model_dir")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8420)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--preload-gb", type=float, default=0)
    ap.add_argument("--backend", default="c", choices=["c", "mlx"],
                    help="c = midged subprocess; mlx = in-process MLX "
                         "(Apple Silicon Metal or CUDA via mlx[cuda])")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"],
                    help="mlx backend only")
    ap.add_argument("--cache-gb", type=float, default=2.0,
                    help="mlx backend: expert LRU budget")
    ap.add_argument("--dense-bits", type=int, default=8,
                    choices=[4, 8, 16, 32], help="mlx backend")
    ap.add_argument("--chunk", type=int, default=8,
                    help="generation chunk size (stop strings are checked "
                         "between chunks)")
    ap.add_argument("--system", default="You are a helpful assistant.\n"
                    "Reasoning: low",
                    help="system prompt used when the client sends none")
    args = ap.parse_args(argv)

    Handler.S = Session(args)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[serve] {Handler.S.model_name} on http://{args.host}:{args.port}/v1"
          f" · ctx {args.ctx} · OpenAI-compatible", file=sys.stderr)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
