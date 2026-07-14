"""harmony — chat formatting + streaming display for gpt-oss's
harmony format. Shared by the C-engine CLI (./midge) and the MLX CLI
(./midge-mlx). Tokenizer-agnostic: pass any object with encode()/
decode()/token_to_id().
"""
from __future__ import annotations
import sys


class Harmony:
    """Chat formatting + streaming display for gpt-oss's harmony format."""

    SPECIALS = ["<|start|>", "<|message|>", "<|end|>", "<|channel|>",
                "<|return|>", "<|call|>"]

    def __init__(self, tok, spec):
        self.tok = tok
        self.id = {}
        for sname in self.SPECIALS:
            i = tok.token_to_id(sname)
            if i is not None:
                self.id[sname] = i
        self.stop_ids = [self.id[t] for t in ("<|return|>", "<|call|>")
                         if t in self.id]
        if not self.stop_ids:
            eos = spec.get("tokenizer", {}).get("eos_token_id")
            if eos is not None:
                self.stop_ids = [eos]

    def enc(self, text):
        return self.tok.encode(text).ids

    def render(self, messages, add_generation_prefix=True):
        ids = []
        for m in messages:
            ids += self.enc(f"<|start|>{m['role']}")
            if m["role"] == "assistant":
                ids += self.enc("<|channel|>final")
            ids += self.enc(f"<|message|>{m['content']}<|end|>")
        if add_generation_prefix:
            ids += self.enc("<|start|>assistant")
        return ids

    # -- streaming state machine ----------------------------------------
    def stream(self, show_analysis=True):
        """Returns (on_token, finish) closures that pretty-print harmony
        output as it is generated and collect the final-channel text."""
        st = {"mode": "text", "channel": "final", "buf": [], "printed": 0,
              "final": [], "header": []}
        DIM, RESET = "\033[2m", "\033[0m"
        rid = {v: k for k, v in self.id.items()}

        def flush_text():
            if not st["buf"]:
                return
            text = self.tok.decode(st["buf"], skip_special_tokens=False)
            new = text[st["printed"]:]
            if new:
                if st["channel"] == "final":
                    sys.stdout.write(new)
                    st["final"].append(new)
                elif show_analysis:
                    sys.stdout.write(DIM + new + RESET)
                sys.stdout.flush()
            st["printed"] = len(text)

        def reset_buf(mode, keep_channel=True):
            st["buf"], st["printed"], st["mode"] = [], 0, mode
            if not keep_channel:
                st["channel"] = "final"

        def on_token(t):
            name = rid.get(t)
            if name is None:
                if st["mode"] == "channel":
                    st["header"].append(t)
                else:
                    st["buf"].append(t)
                    flush_text()
                return
            if name == "<|channel|>":
                st["header"] = []
                st["mode"] = "channel"
            elif name == "<|message|>":
                if st["mode"] == "channel":
                    st["channel"] = self.tok.decode(
                        st["header"], skip_special_tokens=False).strip()
                    if st["channel"] != "final" and show_analysis:
                        sys.stdout.write(f"{DIM}[{st['channel']}] ")
                reset_buf("text")
            elif name in ("<|end|>", "<|return|>", "<|call|>"):
                flush_text()
                if st["channel"] != "final" and show_analysis:
                    sys.stdout.write(RESET + "\n")
                    sys.stdout.flush()
                reset_buf("text", keep_channel=False)
            elif name == "<|start|>":
                reset_buf("text")   # role tokens follow; ignore until channel/message

        def finish():
            flush_text()
            return "".join(st["final"])

        return on_token, finish
