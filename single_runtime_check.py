#!/usr/bin/env python3
"""single_runtime_check — detect duplicate OpenMP / BLAS runtimes in a binary or running process.

The silent killer when embedding PyTorch (its own bundled libgomp/BLAS) into an HPC tool (LAMMPS/OpenMM/
RASPA linked to a *different* OpenMP/BLAS): two copies co-resident → segfaults deep in BLAS
(e.g. cblas_dgemm_batch) or CMake's "cannot generate a safe runtime search path". This scans the
*resolved* shared-library set and flags more than one provider of each runtime family.

Usage:
    single_runtime_check.py <path-to-executable-or-.so>   # uses ldd
    single_runtime_check.py --pid <PID>                   # uses /proc/<pid>/maps
Exit 0 = single runtime per family (good); 1 = a duplicate was found.
"""
from __future__ import annotations
import argparse, re, subprocess, sys
from collections import defaultdict

# library "families" that must be singletons in a process
FAMILIES = {
    "OpenMP (GNU)":  re.compile(r"/(libgomp\.so[.\d]*)$"),
    "OpenMP (LLVM)": re.compile(r"/(libomp\.so[.\d]*)$"),
    "OpenMP (Intel)":re.compile(r"/(libiomp5\.so)$"),
    "BLAS/OpenBLAS": re.compile(r"/(libopenblas[\w.-]*\.so[.\d]*)$"),
    "BLAS (MKL)":    re.compile(r"/(libmkl_rt\.so[.\d]*|libmkl_core\.so[.\d]*)$"),
    "CBLAS":         re.compile(r"/(libcblas\.so[.\d]*)$"),
}


def resolved_libs_ldd(path: str) -> list[str]:
    out = subprocess.run(["ldd", path], capture_output=True, text=True).stdout
    return re.findall(r"=>\s+(/\S+)", out) + re.findall(r"^\s*(/\S+)\s+\(0x", out, re.M)


def resolved_libs_pid(pid: int) -> list[str]:
    libs = set()
    for line in open(f"/proc/{pid}/maps"):
        m = re.search(r"\s(/\S+\.so[.\d]*)$", line.rstrip())
        if m:
            libs.add(m.group(1))
    return sorted(libs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", nargs="?")
    ap.add_argument("--pid", type=int)
    a = ap.parse_args()
    libs = resolved_libs_pid(a.pid) if a.pid else resolved_libs_ldd(a.target)
    if not libs:
        print("no shared libraries resolved (bad target?)", file=sys.stderr); return 1

    hits = defaultdict(set)   # family -> set of full paths
    for lib in libs:
        for fam, rx in FAMILIES.items():
            if rx.search(lib):
                hits[fam].add(lib)

    bad = False
    for fam, paths in sorted(hits.items()):
        if len(paths) > 1:
            bad = True
            print(f"CONFLICT  {fam}: {len(paths)} copies")
            for p in sorted(paths):
                print(f"            {p}")
        else:
            print(f"ok        {fam}: {next(iter(paths))}")
    if bad:
        print("\nFAIL: duplicate runtime(s) — collapse to one (e.g. neutralise torch's bundled libgomp,\n"
              "or build the HPC tool against the same conda BLAS/OpenMP as torch). See Spec 2.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
