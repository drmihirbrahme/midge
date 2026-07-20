# Installing midge

## The short version

**Any platform, works today:**
```bash
git clone https://github.com/drmihirbrahme/midge && cd midge
./install.sh
./midge ui
```
`install.sh` checks your toolchain, installs Python deps, builds the
engine, and smoke-tests it. Then `./midge ui` opens the web app.

## Double-click apps (macOS / Windows / Linux)

Prebuilt bundles are produced by the `package` GitHub Actions workflow
on each release and attached to the
[Releases](https://github.com/drmihirbrahme/midge/releases) page.

### macOS (.dmg)
1. Download `midge-macos.dmg`, open it, drag **midge** to Applications.
2. First launch: right-click → Open (the build is not yet notarized, so
   Gatekeeper asks once).
3. midge opens its web UI. Models and the built engine live in
   `~/Library/Application Support/midge`.

macOS note: the engine builds single-threaded unless you install OpenMP
(`brew install libomp`); for full speed on Apple Silicon, the bundle can
also use the **MLX** backend — `pip install mlx` in the same Python, or
select it in the UI.

### Windows (.zip)
1. Download `midge-windows.zip`, extract it, run `midge.exe`.
2. **Important:** the native C engine is POSIX-only (it memory-maps
   experts with `mmap`/`madvise`, which Windows lacks). On Windows midge
   runs one of two ways:
   * **WSL2** (recommended): install WSL, then use the Linux instructions
     above inside it — full speed, the C engine works natively.
   * **MLX/GPU**: if you have a supported GPU, `pip install "mlx[cuda]"`
     and use `--backend mlx`.
   The `.exe` launcher detects this and points you to whichever applies.

### Linux (.tar.gz)
Extract and run `./midge/midge`. Self-contained (bundles Python).

## Getting a model

You do **not** download models by hand. In the UI, either pick one from
the catalog, **search Hugging Face** right in the app, or paste any repo
id — midge downloads, converts, and builds in one step, with resumable
progress. From the CLI:
```bash
./midge search mixtral          # find compatible models
./midge check <repo-id>         # will it run on this machine?
./midge ui                      # download + convert + chat
```

## Building the installers yourself

On the target OS (PyInstaller can't cross-compile):
```bash
pip install -r requirements.txt pyinstaller
pyinstaller packaging/midge.spec
# dist/midge/  ->  wrap per platform (see .github/workflows/package.yml)
```
Signing/notarization (Apple Developer ID, Windows Authenticode) need paid
certificates and are intentionally left as manual steps.
