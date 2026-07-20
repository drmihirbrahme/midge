# Getting help with midge

## Is it actually midge?

midge's startup prints these lines and no others:

```
LOADING
# loading <your-model-dir>
# mapped experts (N.N GB)
# midge ready in N.Ns · layers=… experts=… …
READY
```

If you see a different banner — for example **"waking the giant"** — you
are running a *different* project or an old fork, not midge. Please
double-check you built this repository (`github.com/drmihirbrahme/midge`)
and are launching `./midge` or `./midged` from it.

## It hangs / never finishes loading

A first cold load memory-maps the expert file and touches pages on
demand, which on a large model over a slow disk can take a while — but
midge now shows progress (`# mapped experts …`) and the server/UI will
time out with a clear message rather than hanging forever. If it stalls:

1. Run `./midge check <model-dir>` — it measures your disk/RAM and says
   whether the model realistically fits.
2. Watch for the `# mapped experts (N GB)` line. If you never see it, the
   container files may be incomplete — re-run the conversion (it resumes).
3. Try a smaller model first (**gpt-oss-20b**) to confirm the pipeline
   works on your machine before attempting 120b.

## It won't build

* **Linux**: `sudo apt install build-essential python3-pip`
* **macOS**: `xcode-select --install` (the engine builds single-threaded
  unless you also `brew install libomp`)
* **Windows**: the C engine is POSIX-only — use WSL2, or the MLX backend
  on a supported GPU. See [docs/INSTALL.md](docs/INSTALL.md).

## Still stuck?

Open an issue or a discussion:

* **Issues** — bugs and build failures: include your OS, the exact
  command, and the full output up to where it stopped.
* **Discussions** — questions, model requests, "does X work" — including
  reports from real gpt-oss / Mixtral / Qwen3-MoE weights, which are
  especially welcome since CI only exercises synthetic fixtures.

When reporting a load problem, please paste the startup lines above so we
can immediately tell midge apart from other tools.
