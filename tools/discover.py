"""discover — find compatible models on Hugging Face.

midge can convert any model in the MoE-transformer family it implements
(gpt-oss, Mixtral-style, Qwen3-MoE-style). This module searches the Hub
and flags which results are likely to work, so a user can browse instead
of memorizing repo ids.

    ./midge search mixtral
    ./midge search --family qwen3-moe

Compatibility here is a *hint* from Hub metadata (model_type / tags);
the authoritative check is still `midge check <repo>`, which reads the
actual config.json. Network-dependent; degrades to a clear message
offline.
"""
from __future__ import annotations
import sys

# Hub model_type -> how midge treats it. Keep in sync with spec_from_hf.
SUPPORTED = {"gpt_oss": "gpt-oss", "mixtral": "mixtral", "qwen3_moe": "qwen3-moe"}
KNOWN_UNSUPPORTED = {
    "deepseek_v2": "MLA attention not implemented",
    "deepseek_v3": "MLA + sigmoid router not implemented",
    "qwen2_moe": "shared expert not implemented",
    "glm4_moe": "shared experts not implemented",
    "granitemoe": "router/attention multipliers not implemented",
}
# search terms that map to a concrete family filter
FAMILY_TAGS = {
    "gpt-oss": "gpt_oss", "mixtral": "mixtral", "qwen3-moe": "qwen3_moe",
}


def search(query=None, family=None, limit=25):
    """Return a list of dicts: {id, downloads, likes, model_type, verdict}."""
    from huggingface_hub import HfApi
    api = HfApi()
    kw = {"limit": limit * 3, "sort": "downloads", "direction": -1,
          "fetch_config": False, "full": False}
    if family and family in FAMILY_TAGS:
        kw["filter"] = FAMILY_TAGS[family]
    elif query:
        kw["search"] = query
    else:
        kw["filter"] = "mixture-of-experts"

    out = []
    for m in api.list_models(**kw):
        tags = set(getattr(m, "tags", []) or [])
        mt = None
        for t in tags:
            if t in SUPPORTED or t in KNOWN_UNSUPPORTED:
                mt = t
                break
        # library tag heuristics when model_type isn't a tag
        rid = m.id.lower()
        if mt is None:
            for key in SUPPORTED:
                if key.replace("_", "") in rid.replace("-", "").replace("_", ""):
                    mt = key
                    break
        verdict = ("likely" if mt in SUPPORTED
                   else "no" if mt in KNOWN_UNSUPPORTED else "unknown")
        out.append({
            "id": m.id,
            "downloads": getattr(m, "downloads", 0) or 0,
            "likes": getattr(m, "likes", 0) or 0,
            "model_type": mt,
            "reason": KNOWN_UNSUPPORTED.get(mt) if verdict == "no" else None,
            "verdict": verdict,
        })
    # compatible first, then by downloads
    order = {"likely": 0, "unknown": 1, "no": 2}
    out.sort(key=lambda r: (order[r["verdict"]], -r["downloads"]))
    return out[:limit]


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="midge search")
    ap.add_argument("query", nargs="?", help="free-text search on the Hub")
    ap.add_argument("--family", choices=list(FAMILY_TAGS),
                    help="restrict to one supported family")
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    try:
        rows = search(a.query, a.family, a.limit)
    except Exception as e:
        if "hub" in type(e).__module__ or "Connection" in type(e).__name__:
            sys.stderr.write("midge search: could not reach Hugging Face — "
                             "check your connection.\n")
            return 1
        raise
    if a.json:
        import json
        print(json.dumps(rows, indent=2))
        return 0
    if not rows:
        print("no models found. Try a broader query.")
        return 0
    mark = {"likely": "✓", "unknown": "?", "no": "✗"}
    print(f"{'':1} {'model':<48} {'downloads':>10}  compatibility")
    for r in rows:
        note = (r["model_type"] or "unrecognized") if r["verdict"] != "no" \
            else r["reason"]
        print(f"{mark[r['verdict']]} {r['id']:<48} {r['downloads']:>10}  {note}")
    print("\n✓ likely supported · ? unknown (run: midge check <id>) · "
          "✗ not supported")
    print("convert any of these with:  ./midge ui   (or ./midge convert <id> "
          "models/<name>)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
