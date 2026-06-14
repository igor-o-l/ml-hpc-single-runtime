# Mixing PyTorch with HPC/MD tools: one OpenMP, one BLAS

**TL;DR:** If you load a **pip PyTorch wheel** (its bundled `libgomp`/BLAS) into the **same process
as a conda/system HPC tool** (LAMMPS, OpenMM, RASPA) that links a *different* OpenMP/BLAS, you get
segfaults or link failures. The fix is ensuring exactly one OpenMP and one BLAS in the process.

---

## The symptom: segfault in BLAS or "cannot generate a safe runtime search path"

### Runtime crash (gdb signature)

```
Thread 1 "python" received signal SIGSEGV, Segmentation fault.
0x00007fffc1234567 in cblas_dgemm_batch ()
   from /home/user/miniconda3/envs/mlip/lib/libopenblas.so.0
```

Or:
```
SIGSEGV in libgomp_parallel_loop_runtime
```

This happens when MACE (via PyTorch) calls a BLAS routine like `bmm` → `cblas_dgemm_batch`, but two
copies of OpenBLAS or libgomp are loaded — one from the pip torch wheel, one from conda's HPC tool.
The memory layouts / thread pools clash, and you get a crash deep in BLAS.

### Build-time error (CMake)

```
CMake Error at cmake/modules/TorchConfig.cmake:123:
  Cannot generate a safe runtime search path for target `lmp` because files
  in some directories may conflict with libraries in implicit directories:

    runtime library [libgomp.so.1] in /home/.../torch/lib
    may be hidden by files in: /home/user/miniconda3/envs/mlip/lib
```

CMake detects the conflict and refuses to generate a build rather than produce a broken binary.

---

## The root cause: two runtimes

Pip PyTorch wheels (e.g. `torch==2.x+cu12x`) bundle their own:
- `torch/lib/libgomp.so.1` (OpenMP)
- `torch/lib/libopenblas.so` or MKL libs (BLAS)

When you `conda install lammps` (or OpenMM, RASPA, etc.), that binary links to *conda's*:
- `$CONDA_PREFIX/lib/libgomp.so.1`
- `$CONDA_PREFIX/lib/libopenblas.so.0`

Loading both in one Python process → two OpenMP runtimes initializing, two BLAS thread pools, and
undefined behavior when either library tries to use the other's memory.

---

## Diagnosis

Run the diagnostic on your built binary or running Python process:

```bash
# On a binary:
python single_runtime_check.py $(which lmp)

# On a running process:
python single_runtime_check.py --pid $(pgrep -f "python.*mace")
```

Output:
```
CONFLICT  OpenMP (GNU): 2 copies
            /home/user/.local/lib/python3.10/site-packages/torch/lib/libgomp.so.1
            /home/user/miniconda3/envs/mlip/lib/libgomp.so.1

FAIL: duplicate runtime(s) — collapse to one
```

---

## The fix: one OpenMP, one BLAS

### Option A: Conda everything (recommended for production)

Use `conda install pytorch pytorch-cuda=12.x` (conda-forge) instead of `pip install torch`. All
packages then share conda's `libgomp` and BLAS. This is the cleanest solution but may lag behind
the latest CUDA/GPU arch.

### Option B: Build against pip torch's runtimes

When you *must* use a pip torch wheel (e.g. for sm_120/Blackwell before conda catches up), build
your HPC tool (LAMMPS) against torch's bundled libraries, not conda's:

1. **Include the CMake toolchain snippet:**
   ```cmake
   include(/path/to/single_runtime.cmake)
   setup_single_runtime()
   find_package(Torch REQUIRED)
   ```

2. **Key settings the snippet applies:**
   - `OpenMP_gomp_LIBRARY` → torch's bundled `libgomp.so.1`
   - Link directories prioritize `torch/lib`
   - RPATH includes `torch/lib` for runtime resolution

3. **Build LAMMPS:**
   ```bash
   cmake ../cmake \
     -DCMAKE_TOOLCHAIN_FILE=/path/to/single_runtime.cmake \
     -DPKG_ML-MACE=yes \
     -DCMAKE_PREFIX_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")
   make -j
   ```

### Option C: Isolate via subprocess

If you can't rebuild, isolate the conflicting components:
- Run LAMMPS in a subprocess with `LAMMPS_POTENTIALS` / input file, not as a Python extension
- Or use separate conda environments and communicate via files/sockets

This avoids loading both runtimes in the same address space, but loses the convenience of in-process
interop.

---

## MKL headers missing (`MKL_INCLUDE_DIR-NOTFOUND`)

Pip torch wheels link MKL but don't ship its headers. If CMake complains:

```
MKL_INCLUDE_DIR-NOTFOUND
```

Fix options:
1. Point to conda's MKL headers: `cmake -DMKL_INCLUDE_DIR=$CONDA_PREFIX/include ...`
2. The `single_runtime.cmake` snippet attempts this automatically
3. Or switch to OpenBLAS: `cmake -DBLA_VENDOR=OpenBLAS ...`

---

## Verification after the fix

```bash
python single_runtime_check.py $(which lmp)
```

Output should show:
```
ok        OpenMP (GNU): /path/to/libgomp.so.1
ok        BLAS/OpenBLAS: /path/to/libopenblas.so.0
```

(Exactly one path per family.)

---

## Common scenarios

| Setup | Result | Fix |
|-------|--------|-----|
| `pip install torch` + `conda install lammps` | SIGSEGV in cblas | Rebuild LAMMPS with Option B, or switch to conda torch |
| `conda install pytorch lammps` (all conda-forge) | Works | None needed |
| `pip install torch` + build LAMMPS from source (naive) | CMake "safe runtime path" error | Use `single_runtime.cmake` |
| MACE-in-LAMMPS (`pair_style mliap`) + pip torch | SIGSEGV at runtime | Same as row 1 |

---

## References

- **This project's L11 saga:** `docs/lab-gaps-and-remediation.md` — gdb backtrace that identified
  the `cblas_dgemm_batch` crash as a pip-torch↔conda-LAMMPS dual-runtime clash.
- **Spec 2 (full context):** `docs/community-contributions/02-ml-hpc-single-runtime-packaging.md`
- **MACE docs:** [ACEsuit/mace](https://github.com/ACEsuit/mace) — see installation notes for
  LAMMPS integration
- **LAMMPS docs:** [ML-MACE package](https://docs.lammps.org/Packages_details.html#pkg-ml-mace)
- **conda-forge feedstocks:** `conda-forge/lammps-feedstock`, `conda-forge/pytorch-cpu-feedstock`
