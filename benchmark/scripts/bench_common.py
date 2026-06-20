#!/usr/bin/env python3
# =============================================================================
# bench_common.py — shared driver for the file-based NumSharp-vs-NumPy matrix
# subsystems (benchmark/layout, benchmark/cast, benchmark/fusion).
#
# Each subsystem's *_sheet.py imports this to, identically on the author's box
# and on a Linux CI runner:
#   * build NumSharp.Core (Release),
#   * run a `*_bench.cs` via `dotnet run -c Release -` (fed on stdin, with the
#     author's absolute #:project path rewritten to THIS checkout's csproj),
#   * run its `*_bench.py` NumPy twin,
#   * parse the keyed TSV (`key\tms`) both sides emit.
#
# Mirrors the proven npyiter_sheet.py mechanics, centralised so every matrix
# subsystem runs through one code path. run_benchmark.py drives the sheets; the
# sheets render; this module is the plumbing between.
# =============================================================================
import math
import os
import subprocess
import sys

# The absolute #:project path the author's .cs benches pin (so they can also be
# run directly as `dotnet run -c Release - < file`). Rewritten per checkout.
AUTHOR_CSPROJ = "K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj"

# An AV under heavy mixed load should fail FAST (non-zero exit) instead of
# stalling while the runtime writes a crash dump — same policy as npyiter.
NS_ENV_EXTRA = {"DOTNET_DbgEnableMiniDump": "0", "DOTNET_EnableCrashReport": "0"}


def core_csproj(repo):
    return os.path.join(repo, "src", "NumSharp.Core", "NumSharp.Core.csproj")


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def build_core(repo):
    """Build NumSharp.Core in Release. Exits the process on build failure."""
    log("[build] NumSharp.Core (Release)…")
    b = subprocess.run(
        ["dotnet", "build", core_csproj(repo), "-c", "Release", "-v", "q", "--nologo",
         "-clp:NoSummary;ErrorsOnly", "-p:WarningLevel=0"],
        capture_output=True, text=True, cwd=repo)
    if b.returncode != 0:
        log("[build] FAILED:\n" + b.stdout[-1500:])
        sys.exit(1)
    log("[build] ok")


def run_cs(repo, cs_path, timeout=1200):
    """Run a file-based NumSharp bench (.cs) and return its stdout (the keyed TSV).

    The .cs is fed on stdin (`dotnet run -c Release -`) so sibling git worktrees
    under .claude/worktrees/ can't confuse the project search; the author's
    absolute #:project path is rewritten to this checkout's csproj first.
    """
    with open(cs_path, encoding="utf-8") as f:
        src = f.read()
    src = src.replace(AUTHOR_CSPROJ, core_csproj(repo).replace(os.sep, "/"))
    env = {**os.environ, **NS_ENV_EXTRA}
    name = os.path.basename(cs_path)
    try:
        p = subprocess.run(["dotnet", "run", "-c", "Release", "-"], input=src,
                           capture_output=True, text=True, encoding="utf-8", errors="replace",
                           cwd=repo, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        log(f"    [cs] {name}: TIMED OUT ({timeout}s) — section dropped")
        return ""
    if p.returncode != 0:
        log(f"    [cs] {name}: exit {p.returncode}\n{p.stderr[-600:]}")
    return p.stdout


def run_py(repo, py_path, timeout=900):
    """Run a NumPy twin bench (.py) and return its stdout (the keyed TSV)."""
    name = os.path.basename(py_path)
    try:
        p = subprocess.run([sys.executable, py_path], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", cwd=repo, timeout=timeout)
    except subprocess.TimeoutExpired:
        log(f"    [py] {name}: TIMED OUT ({timeout}s) — section dropped")
        return ""
    if p.returncode != 0:
        log(f"    [py] {name}: exit {p.returncode}\n{p.stderr[-600:]}")
    return p.stdout


def parse_tsv(text):
    """`key\\tvalue` lines -> {key: float}. Non-numeric / header lines ignored."""
    out = {}
    for ln in text.splitlines():
        if "\t" in ln:
            k, _, v = ln.partition("\t")
            try:
                out[k.strip()] = float(v)
            except ValueError:
                pass
    return out


def geomean(xs):
    xs = [x for x in xs if x == x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def icon(r):
    """Project convention: ratio = NumPy/NumSharp, >1 = NumSharp faster."""
    if r != r:
        return "?"
    return "✅" if r >= 1.0 else "🟡" if r >= 0.5 else "🟠" if r >= 0.2 else "🔴"


def ratio_rows(ns, npy):
    """Keys present (and positive) on BOTH sides -> (key, ns_ms, np_ms, ratio).

    ratio = np_ms / ns_ms (>1.0 = NumSharp faster). NS-only / NumPy-only keys
    (e.g. Decimal casts, bool np.positive) are dropped — they have no comparand.
    """
    rows = []
    for k, nm in ns.items():
        nv = npy.get(k)
        if nv is None or nm <= 0 or nv <= 0:
            continue
        rows.append((k, nm, nv, nv / nm))
    return rows
