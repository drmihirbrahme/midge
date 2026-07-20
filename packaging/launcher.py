"""launcher — entry point for the packaged (PyInstaller) midge app.

When frozen, resources live next to the executable (sys._MEIPASS). This
launcher makes the bundle behave exactly like a git checkout: it points
the tools at the bundled files, ensures the engine binary exists
(building it once from the bundled C source if a compiler is available,
with a clear message if not), then hands off to the normal CLI.

With no arguments it launches the web UI — the friendliest default for
someone who just double-clicked the app.
"""
from __future__ import annotations
import os
import shutil
import subprocess
import sys


def bundle_root() -> str:
    if getattr(sys, "frozen", False):
        # onedir: data files live in sys._MEIPASS (…/midge/_internal)
        return sys._MEIPASS  # type: ignore[attr-defined]
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def engine_search_dirs(root):
    # the prebuilt engine is staged next to the executable, one level up
    # from _internal; also accept it inside the bundle root
    exe_dir = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) \
        else root
    return [exe_dir, root]


def user_data_dir() -> str:
    """A writable home for the built engine, converted models, configs."""
    if sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support/midge")
    elif os.name == "nt":
        base = os.path.join(os.environ.get("LOCALAPPDATA",
                            os.path.expanduser("~")), "midge")
    else:
        base = os.path.join(os.environ.get("XDG_DATA_HOME",
                            os.path.expanduser("~/.local/share")), "midge")
    os.makedirs(base, exist_ok=True)
    return base


def ensure_engine(root: str, home: str) -> str | None:
    """Return a path to a working `midged`, building it if needed.
    None if the platform can't run the C engine (native Windows)."""
    exe_name = "midged.exe" if os.name == "nt" else "midged"
    for d in engine_search_dirs(root):
        cand = os.path.join(d, exe_name)
        if os.path.exists(cand):
            return cand
    built = os.path.join(home, exe_name)
    if os.path.exists(built):
        return built
    if os.name == "nt":
        # the C engine is POSIX-only (mmap/madvise); on native Windows we
        # rely on the MLX path or WSL. Signal "no C engine here".
        return None
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if not cc:
        sys.stderr.write(
            "midge: no C compiler found to build the engine.\n"
            "  macOS: run  xcode-select --install  then reopen midge.\n")
        return None
    src = os.path.join(root, "engine", "midge.c")
    cmd = [cc, "-O3", "-std=c11", "-Wno-unused-function", src,
           "-o", built, "-lm"]
    # OpenMP if available (Apple clang usually lacks it -> single thread)
    probe = subprocess.run([cc, "-fopenmp", "-x", "c", "-", "-o",
                            os.devnull], input="int main(){return 0;}",
                           text=True, capture_output=True)
    if probe.returncode == 0:
        cmd = cmd[:1] + ["-fopenmp"] + cmd[1:] + ["-fopenmp"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(built):
        sys.stderr.write("midge: engine build failed:\n" + r.stderr + "\n")
        return None
    return built


def main() -> int:
    root = bundle_root()
    home = user_data_dir()
    os.environ.setdefault("MIDGE_HOME", home)
    os.environ.setdefault("MIDGE_MODELS_DIR", os.path.join(home, "models"))

    engine = ensure_engine(root, home)
    if engine:
        os.environ["MIDGE_ENGINE"] = engine

    sys.path.insert(0, root)
    sys.path.insert(0, os.path.join(root, "tools"))

    # default action for a bare double-click: open the UI
    argv = sys.argv[1:] or ["ui"]

    # dispatch through the same logic as the ./midge script. It has no
    # .py extension, so load it as source rather than runpy.run_path
    # (which fails inside a frozen bundle).
    sys.argv = ["midge"] + argv
    import types
    mod = types.ModuleType("__main__")
    mod.__file__ = os.path.join(root, "midge_cli.py")
    with open(mod.__file__) as f:
        code = compile(f.read(), mod.__file__, "exec")
    g = mod.__dict__
    g["__name__"] = "__main__"
    sys.modules["__main__"] = mod
    exec(code, g)
    return 0


if __name__ == "__main__":
    sys.exit(main())
