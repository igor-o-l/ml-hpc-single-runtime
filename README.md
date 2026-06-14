# ml-hpc-single-runtime

Tools + docs to kill the **pip-PyTorch ↔ conda/system HPC-tool dual-runtime clash** — two copies of
OpenMP (`libgomp`) and/or BLAS (`libopenblas`/`libcblas`/MKL) in one process, which causes segfaults
deep in BLAS (`cblas_dgemm_batch`) and the CMake "cannot generate a safe runtime search path" error.

Implements [Spec 2](../../docs/community-contributions/02-ml-hpc-single-runtime-packaging.md).

## What's here

```
single_runtime_check.py     # diagnostic: scan binary/PID/current-process for duplicate runtimes
cmake/single_runtime.cmake  # CMake toolchain: force pip-torch's OpenMP/BLAS for HPC-tool builds
recipe/meta.yaml            # conda-forge recipe stub: lammps+libtorch with shared runtimes
docs/single-runtime.md      # authoritative docs: gdb signature, diagnosis, fix
```

## Quickstart

### Diagnose

```bash
# On a binary (ldd-based):
python single_runtime_check.py $(which lmp)

# On a running process:
python single_runtime_check.py --pid $(pgrep -f "python.*mace")

# Current Python process (after importing torch + lammps):
python single_runtime_check.py --current

# JSON output for CI:
python single_runtime_check.py --json $(which lmp)

# With fix hints:
python single_runtime_check.py --fix-hints $(which lmp)
```

Output on conflict:
```
CONFLICT  OpenMP (GNU): 2 copies (origins: torch, conda)
            /home/user/.local/lib/python3.10/site-packages/torch/lib/libgomp.so.1
            /home/user/miniconda3/envs/mlip/lib/libgomp.so.1

FAIL: duplicate runtime(s) detected
```

### Fix (Option A: conda everything)

```bash
pip uninstall torch torchvision torchaudio
conda install pytorch pytorch-cuda=12.x -c pytorch -c nvidia
```

### Fix (Option B: build HPC tool against pip-torch's runtimes)

```cmake
# In your CMakeLists.txt:
include(/path/to/ml-hpc-single-runtime/cmake/single_runtime.cmake)
setup_single_runtime()
find_package(Torch REQUIRED)
```

Then:
```bash
cmake ../cmake \
  -DCMAKE_TOOLCHAIN_FILE=/path/to/single_runtime.cmake \
  -DPKG_ML-MACE=yes \
  -DCMAKE_PREFIX_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")
```

## Integration with oeq-bench

The diagnostic is also integrated into [oeq-bench](../oeq-bench/), which warns on startup if
duplicate runtimes are detected. To skip the check:

```bash
OEQ_SKIP_RUNTIME_CHECK=1 oeq-bench
# or
oeq-bench --skip-runtime-check
```

## Status

- **Diagnostic:** complete (binary, PID, current-process; JSON output; fix hints)
- **CMake toolchain:** complete (forces pip-torch's libgomp/BLAS; MKL header shim)
- **Conda recipe:** stub only — gated on conda-forge pytorch catching up to CUDA 13 / sm_120
- **Docs:** complete (see `docs/single-runtime.md`)

## Background

This bug is generic: it bites anyone embedding libtorch in an MD/HPC binary, or mixing
CUDA-version-specific pip wheels with conda HPC libraries. The symptoms are scary (segfaults deep
in BLAS) and the root cause is non-obvious. See [docs/single-runtime.md](docs/single-runtime.md)
for the full story.

## Contributing

Submit issues/PRs to the MLIPs repo or open a discussion. The conda-forge recipe will be proposed
upstream once pytorch-cuda catches up to newer GPU architectures.
