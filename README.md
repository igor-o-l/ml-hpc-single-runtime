# ml-hpc-single-runtime

Tools + docs to kill the **pip-PyTorch ↔ conda/system HPC-tool dual-runtime clash** — two copies of
OpenMP (`libgomp`) and/or BLAS (`libopenblas`/`libcblas`/MKL) in one process, which causes segfaults deep
in BLAS (`cblas_dgemm_batch`) and the CMake "cannot generate a safe runtime search path" error.
Implements [Spec 2](../../docs/community-contributions/02-ml-hpc-single-runtime-packaging.md).

## What's here
- `single_runtime_check.py` — diagnostic: scans a binary (`ldd`) or live process (`--pid`) and flags any
  runtime family present more than once. Run it on your `lmp`/python before debugging mystery crashes.
- `cmake/single_runtime.cmake` — toolchain snippet enforcing one OpenMP/BLAS when building an HPC tool
  against libtorch (the `MKL_INCLUDE_DIR` shim, single-libgomp handling).
- `recipe/meta.yaml` — conda-forge recipe **stub** for a `lammps`+`libtorch` build that shares the conda
  BLAS/OpenMP (the durable fix; the helper above is for when a pip torch wheel must be used).
- `docs/single-runtime.md` — the searchable write-up (gdb signature → diagnosis → fix).

## Quickstart
```bash
python single_runtime_check.py $(which lmp)        # or: --pid <running pid>
```

## Status
Diagnostic works (detects the libgomp/OpenBLAS duplication that caused the L11 segfault). TODO: finish the
conda-forge recipe (gated on conda-forge shipping a CUDA-13/sm_120 pytorch), the CMake toolchain snippet,
and the docs page; submit recipe via `conda-forge/staged-recipes`, docs PRs to MACE/LAMMPS.
