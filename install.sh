#!/bin/sh
# midge installer: checks the toolchain, installs Python deps, builds the
# engine, and runs a smoke test. Safe to re-run.
set -e
say() { printf '\033[1m[install]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[install]\033[0m %s\n' "$*" >&2; exit 1; }

command -v python3 >/dev/null || fail "python3 not found — install Python 3.9+"
PYV=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
say "python $PYV"

if ! command -v cc >/dev/null && ! command -v gcc >/dev/null; then
  case "$(uname -s)" in
    Darwin) fail "no C compiler — run: xcode-select --install";;
    *)      fail "no C compiler — run: sudo apt install build-essential (Debian/Ubuntu)";;
  esac
fi

say "installing Python dependencies"
python3 -m pip install -r requirements.txt \
  || python3 -m pip install --break-system-packages -r requirements.txt \
  || fail "pip install failed — try: python3 -m pip install --user -r requirements.txt"

say "building the engine"
make || fail "build failed (see above)"

say "smoke test"
./midged --bench | sed 's/^/[install]   /'
say "done. Next:  ./midge ui        (web interface)"
say "        or:  ./midge check openai/gpt-oss-20b"
