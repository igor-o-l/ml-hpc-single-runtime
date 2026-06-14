#!/usr/bin/env python3
"""single_runtime_check — detect duplicate OpenMP / BLAS runtimes in a binary or running process.

The silent killer when embedding PyTorch (its own bundled libgomp/BLAS) into an HPC tool (LAMMPS/OpenMM/
RASPA linked to a *different* OpenMP/BLAS): two copies co-resident → segfaults deep in BLAS
(e.g. cblas_dgemm_batch) or CMake's "cannot generate a safe runtime search path". This scans the
*resolved* shared-library set and flags more than one provider of each runtime family.

Usage:
    single_runtime_check.py <path-to-executable-or-.so>   # uses ldd
    single_runtime_check.py --pid <PID>                   # uses /proc/<pid>/maps
    single_runtime_check.py --current                     # scan the current Python process
    single_runtime_check.py --json                        # output as JSON (for CI)
    single_runtime_check.py --fix-hints                   # show remediation hints

Exit 0 = single runtime per family (good); 1 = a duplicate was found.

Part of ml-hpc-single-runtime: https://github.com/igor-o-l/ml-hpc-single-runtime
See docs/single-runtime.md for the full diagnosis and fix guide.
"""
from __future__ import annotations
import argparse
import json as json_mod
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

FAMILIES = {
    "OpenMP (GNU)":   re.compile(r"/(libgomp\.so[.\d]*)$"),
    "OpenMP (LLVM)":  re.compile(r"/(libomp\.so[.\d]*)$"),
    "OpenMP (Intel)": re.compile(r"/(libiomp5\.so)$"),
    "BLAS/OpenBLAS":  re.compile(r"/(libopenblas[\w.-]*\.so[.\d]*)$"),
    "BLAS (MKL)":     re.compile(r"/(libmkl_rt\.so[.\d]*|libmkl_core\.so[.\d]*)$"),
    "CBLAS":          re.compile(r"/(libcblas\.so[.\d]*)$"),
}

TORCH_MARKER = re.compile(r"/torch/lib/|/site-packages/torch/")
CONDA_MARKER = re.compile(r"/(miniconda|anaconda|mambaforge|conda)/envs?/")


def resolved_libs_ldd(path: str) -> list[str]:
    """Extract resolved .so paths via ldd."""
    try:
        out = subprocess.run(["ldd", path], capture_output=True, text=True, timeout=30).stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []
    return re.findall(r"=>\s+(/\S+)", out) + re.findall(r"^\s*(/\S+)\s+\(0x", out, re.M)


def resolved_libs_pid(pid: int) -> list[str]:
    """Extract loaded .so paths from /proc/<pid>/maps."""
    libs = set()
    try:
        for line in open(f"/proc/{pid}/maps"):
            m = re.search(r"\s(/\S+\.so[.\d]*)$", line.rstrip())
            if m:
                libs.add(m.group(1))
    except (FileNotFoundError, PermissionError):
        return []
    return sorted(libs)


def resolved_libs_current() -> list[str]:
    """Extract loaded .so paths for the current Python process."""
    return resolved_libs_pid(os.getpid())


def classify_origin(path: str) -> str:
    """Classify a library path as 'torch', 'conda', 'system', or 'unknown'."""
    if TORCH_MARKER.search(path):
        return "torch"
    if CONDA_MARKER.search(path):
        return "conda"
    if path.startswith("/usr/lib") or path.startswith("/lib"):
        return "system"
    return "unknown"


def analyze(libs: list[str]) -> dict[str, dict[str, Any]]:
    """Analyze libraries for duplicate runtimes."""
    hits: dict[str, dict[str, Any]] = defaultdict(lambda: {"paths": set(), "origins": set()})
    for lib in libs:
        for fam, rx in FAMILIES.items():
            if rx.search(lib):
                hits[fam]["paths"].add(lib)
                hits[fam]["origins"].add(classify_origin(lib))
    return {k: {"paths": sorted(v["paths"]), "origins": sorted(v["origins"])} for k, v in hits.items()}


def print_report(analysis: dict[str, dict[str, Any]], fix_hints: bool = False) -> bool:
    """Print human-readable report. Returns True if conflicts found."""
    bad = False
    for fam, info in sorted(analysis.items()):
        paths = info["paths"]
        origins = info["origins"]
        if len(paths) > 1:
            bad = True
            print(f"CONFLICT  {fam}: {len(paths)} copies (origins: {', '.join(origins)})")
            for p in paths:
                print(f"            {p}")
        elif paths:
            print(f"ok        {fam}: {paths[0]}")

    if bad:
        print("\nFAIL: duplicate runtime(s) detected")
        if fix_hints:
            print_fix_hints(analysis)
    else:
        print("\nPASS: single runtime per family")

    return bad


def print_fix_hints(analysis: dict[str, dict[str, Any]]):
    """Print remediation hints based on detected conflicts."""
    print("\n--- Remediation hints ---")

    has_torch_conda_conflict = False
    for fam, info in analysis.items():
        if len(info["paths"]) > 1 and "torch" in info["origins"] and "conda" in info["origins"]:
            has_torch_conda_conflict = True
            break

    if has_torch_conda_conflict:
        print("""
The conflict is pip-torch vs conda-HPC-tool (the classic dual-runtime bug).

Options:
1. CONDA EVERYTHING (recommended):
   pip uninstall torch torchvision torchaudio
   conda install pytorch pytorch-cuda=12.x -c pytorch -c nvidia

2. BUILD AGAINST TORCH'S RUNTIMES:
   Use cmake/single_runtime.cmake when building your HPC tool:
   cmake -DCMAKE_TOOLCHAIN_FILE=/path/to/single_runtime.cmake ...

3. ISOLATE VIA SUBPROCESS:
   Don't load both in the same Python process — run one as a subprocess.

See docs/single-runtime.md for full details.
""")
    else:
        print("""
Multiple versions of the same runtime detected.
- Check for mixed conda environments or stale installations.
- Ensure LD_LIBRARY_PATH doesn't inject unexpected paths.
- Use `ldd -v` on your binary to trace library resolution.
""")


def output_json(analysis: dict[str, dict[str, Any]], target: str) -> dict:
    """Generate JSON output for CI integration."""
    conflicts = {fam: info for fam, info in analysis.items() if len(info["paths"]) > 1}
    return {
        "target": target,
        "pass": len(conflicts) == 0,
        "families": analysis,
        "conflicts": conflicts,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Detect duplicate OpenMP/BLAS runtimes (the pip-torch + conda-HPC clash)",
        epilog="Exit 0 = single runtime per family; 1 = duplicate found.",
    )
    ap.add_argument("target", nargs="?", help="Path to executable or .so to scan (uses ldd)")
    ap.add_argument("--pid", type=int, help="Scan a running process by PID (uses /proc/maps)")
    ap.add_argument("--current", action="store_true", help="Scan the current Python process")
    ap.add_argument("--json", action="store_true", help="Output as JSON (for CI)")
    ap.add_argument("--fix-hints", action="store_true", help="Show remediation hints on conflict")
    a = ap.parse_args()

    if a.current:
        libs = resolved_libs_current()
        target = f"pid:{os.getpid()} (current)"
    elif a.pid:
        libs = resolved_libs_pid(a.pid)
        target = f"pid:{a.pid}"
    elif a.target:
        libs = resolved_libs_ldd(a.target)
        target = a.target
    else:
        ap.print_help()
        return 1

    if not libs:
        print(f"No shared libraries resolved from {target}", file=sys.stderr)
        return 1

    analysis = analyze(libs)

    if a.json:
        print(json_mod.dumps(output_json(analysis, target), indent=2))
        conflicts = any(len(info["paths"]) > 1 for info in analysis.values())
        return 1 if conflicts else 0
    else:
        bad = print_report(analysis, fix_hints=a.fix_hints)
        return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
