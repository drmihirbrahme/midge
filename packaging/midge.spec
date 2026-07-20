# PyInstaller spec for midge — builds a self-contained launcher that
# bundles the Python runtime, all tools, the web UI, model specs, and the
# engine C source. Run ON THE TARGET OS (PyInstaller does not cross-compile):
#
#     pip install pyinstaller
#     pyinstaller packaging/midge.spec
#
# Produces dist/midge/ (a folder app) with a `midge` / `midge.exe` entry.
# The engine is compiled from the bundled source on first run by the
# launcher (see packaging/launcher.py), so the only build-time requirement
# on the *developer* machine is a C compiler; end users on macOS get one
# via the Command Line Tools shim, and on Windows we ship a prebuilt
# engine when a compiler is present in CI (see the workflow).

import os

block_cipher = None
root = os.path.abspath(os.getcwd())

def _d(src, dst):
    return (os.path.join(root, src), dst)

datas = [
    _d("tools", "tools"),
    _d("engine", "engine"),
    _d("midge_mlx", "midge_mlx"),
    _d("webui", "webui"),
    _d("specs", "specs"),
    _d("packaging/midge_cli.py", "."),
    _d("Makefile", "."),
    _d("requirements.txt", "."),
    _d("README.md", "."),
    _d("LICENSE", "."),
    _d("NOTICE", "."),
]

hiddenimports = [
    "numpy", "tokenizers", "huggingface_hub",
]

a = Analysis(
    [os.path.join(root, "packaging", "launcher.py")],
    pathex=[root],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["mlx", "torch", "tensorflow"],  # optional/huge; installed separately
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="midge",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)
coll = COLLECT(
    exe, a.binaries, a.zipfiles, a.datas,
    strip=False, upx=False, name="midge",
)
