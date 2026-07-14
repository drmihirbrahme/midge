# Contributing to midge

Small, focused patches, please. Before opening a PR:

1. `make && make test` must pass (it runs the full converter+engine
   validation against the NumPy reference across all dtypes).
2. If you touch the container format or the expert layout, change
   `ExpertLayout` (tools/midgepack.py) and `wt_expert_layout`
   (engine/mten.h) together — they must stay byte-identical.
3. If you touch engine math, mirror it in tools/reference.py; the test
   suite compares them per position.

Wanted:
* SIMD matvec kernels (AVX2/AVX-512/NEON) in engine/mkern.h
* more model families (see docs/ADDING_MODELS.md)
* io_uring / readahead experiments for cold-expert streaming
* Windows support (currently POSIX mmap/madvise)

By contributing you agree your work is licensed under Apache-2.0.

Maintainer: Dr Mihir Brahme <drmihir@duck.com>
