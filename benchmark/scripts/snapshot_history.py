#!/usr/bin/env python3
"""
Assemble a committable benchmark snapshot under ``benchmark/history/<date>_<sha>/``
and (re)point ``benchmark/history/latest`` at it.

This is the *provenance + publish* step of the benchmark ritual: ``run_benchmark.py``
calls it at the end of a full run (so a run produces a committable snapshot, not just
the gitignored ``results/<ts>/`` scratch), and it can also be run standalone to
(re)build a snapshot from an existing ``results/<ts>/`` archive plus the committed
rendered sheets.

A snapshot contains *everything we commit and reference*:

    MANIFEST.md            provenance / env / methodology / headline geomeans
    benchmark-report.md    op-matrix + appended NpyIter/Layout/Operand/Cast/Fusion
    benchmark-report.json  unified machine-readable op-matrix results
    benchmark-report.csv   spreadsheet form
    numpy-results.json     raw NumPy timings (merge input)
    npyiter_results.md/.tsv + cards/{ops,cat}.png
    layout_results.md/.tsv · operand_results.md/.tsv · cast_results.md/.tsv · fusion_results.md

Raw BenchmarkDotNet per-class JSON (~tens of MB) is intentionally NOT persisted
(regenerable — reproduce with ``python benchmark/run_benchmark.py``).

``benchmark/history/latest`` is a relative symlink to the newest snapshot, committed
to git as a mode-120000 object (the repo has ``core.symlinks=true`` and already
tracks symlinks). Docs / CI reference the stable path
``benchmark/history/latest/benchmark-report.md``.

Usage
-----
  python benchmark/scripts/snapshot_history.py                     # newest results/<ts>/, HEAD sha
  python benchmark/scripts/snapshot_history.py --results-dir benchmark/results/20260623-065155
  python benchmark/scripts/snapshot_history.py --snap-name 2026-06-23_e3b7c268   # target an explicit folder
  python benchmark/scripts/snapshot_history.py --commit            # also git-commit the snapshot + latest
  python benchmark/scripts/snapshot_history.py --no-stage          # don't git add
"""
import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent      # benchmark/scripts
BENCH = HERE.parent                          # benchmark
REPO = BENCH.parent                          # repo root
HISTORY = BENCH / "history"
RESULTS = BENCH / "results"

# rendered sheets a run leaves under benchmark/<subsystem>/ (committed + referenced)
SUBSYSTEM_FILES = [
    BENCH / "npyiter" / "npyiter_results.md",
    BENCH / "npyiter" / "npyiter_results.tsv",
    BENCH / "layout" / "layout_results.md",
    BENCH / "layout" / "layout_results.tsv",
    BENCH / "operand" / "operand_results.md",
    BENCH / "operand" / "operand_results.tsv",
    BENCH / "cast" / "cast_results.md",
    BENCH / "cast" / "cast_results.tsv",
    BENCH / "fusion" / "fusion_results.md",
]
CARDS = [BENCH / "npyiter" / "cards" / "ops.png", BENCH / "npyiter" / "cards" / "cat.png"]
# core artifacts that live in results/<ts>/ (the .md is also mirrored at benchmark/ root)
CORE_FROM_RESULTS = ["benchmark-report.md", "benchmark-report.json",
                     "benchmark-report.csv", "numpy-results.json"]


def git(*args):
    """Run git in the repo; return CompletedProcess (never raises on non-zero)."""
    try:
        return subprocess.run(["git", *args], cwd=str(REPO),
                              capture_output=True, text=True)
    except FileNotFoundError:
        return subprocess.CompletedProcess(args, 1, "", "git not found")


def gout(*args):
    r = git(*args)
    return r.stdout.strip() if r.returncode == 0 else ""


def newest_results_dir():
    if not RESULTS.exists():
        return None
    dirs = sorted((p for p in RESULTS.iterdir() if p.is_dir()), key=lambda p: p.name)
    return dirs[-1] if dirs else None


def detect_env(results_dir):
    env = {}
    cpu = None
    csharp = (results_dir / "csharp") if results_dir else None
    if csharp and csharp.exists():
        for j in csharp.glob("*.json"):
            m = re.search(r'"ProcessorName"\s*:\s*"([^"]+)"',
                          j.read_text(encoding="utf-8", errors="ignore"))
            if m:
                cpu = m.group(1)
                break
    env["CPU"] = cpu or (platform.processor() or "unknown")
    env["OS"] = f"{platform.system()} {platform.release()} ({platform.version()})"
    try:
        env["dotnet"] = subprocess.run(["dotnet", "--version"], capture_output=True,
                                       text=True).stdout.strip() or "unknown"
    except Exception:
        env["dotnet"] = "unknown"
    env["python"] = platform.python_version()
    try:
        import numpy
        env["numpy"] = numpy.__version__
    except Exception:
        env["numpy"] = "unknown"
    return env


def parse_size_summary(report_md):
    """Pull the '## Summary by size' rows (real sizes only)."""
    rows = []
    try:
        text = report_md.read_text(encoding="utf-8")
    except Exception:
        return rows
    m = re.search(r"## Summary by size(.+?)(?:\n#{2,3} |\n---|\Z)", text, re.S)
    block = m.group(1) if m else ""
    for line in block.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 10 or not cells[0].replace(",", "").isdigit():
            continue
        if int(cells[0].replace(",", "")) < 1000:
            continue
        try:
            if int(cells[1]) <= 5:   # skip singleton sizes (500/900/50000/5000000)
                continue
        except ValueError:
            continue
        rows.append({"N": cells[0], "ok": cells[2], "close": cells[3], "slow": cells[4],
                     "red": cells[5], "geomean": cells[8], "pnp": cells[9]})
    return rows


def parse_overall(report_md):
    try:
        m = re.search(r"\*\*Summary:\*\*\s*(.+)", report_md.read_text(encoding="utf-8"))
        return m.group(1).strip() if m else None
    except Exception:
        return None


def parse_npyiter_headline():
    try:
        for line in (BENCH / "npyiter" / "npyiter_results.md").read_text(encoding="utf-8").splitlines():
            if line.startswith("HEADLINE"):
                return line.strip()
    except Exception:
        pass
    return None


def parse_cast_headline():
    try:
        for line in (BENCH / "cast" / "cast_results.md").read_text(encoding="utf-8").splitlines():
            if "comparable cells lag" in line or "win (" in line:
                return line.strip().lstrip("- ")
    except Exception:
        pass
    return None


def build_manifest(snap_name, run_ts, head, subject, dirty, dirty_files, env,
                   size_rows, overall, npy_headline, cast_headline):
    date = snap_name.split("_", 1)[0]
    subject = subject.replace("|", "\\|")   # a raw '|' would inject an extra table column
    L = [f"# Benchmark snapshot — {date} · {head}", "",
         "Official NumSharp-vs-NumPy 3-size comparison + the five matrix subsystems, "
         "persisted for provenance. Auto-generated by "
         "`benchmark/scripts/snapshot_history.py`.", "",
         "## Provenance", "| | |", "|---|---|",
         f"| Run timestamp | `{run_ts}` |",
         f"| Git HEAD | `{head}` — {subject} |"]
    if dirty:
        wip = ", ".join(f.replace("|", "\\|") for f in dirty_files[:12]) + (" …" if len(dirty_files) > 12 else "")
        L.append(f"| Working tree | **DIRTY** — benchmarked HEAD + uncommitted changes"
                 + (f": {wip}" if wip else "") + " |")
    else:
        L.append("| Working tree | clean (HEAD exactly) |")
    L += [f"| Date | {date} |", "",
          "## Environment", "| | |", "|---|---|",
          f"| CPU | {env['CPU']} |",
          f"| OS | {env['OS']} |",
          f"| .NET SDK | {env['dotnet']} (net10.0, Release) |",
          f"| Python | {env['python']} |",
          f"| NumPy | {env['numpy']} |", "",
          "## Convention",
          "**Ratio = NumPy_ms ÷ NumSharp_ms (NPY/NS) → `>1.0×` = NumSharp faster** (higher is better).", "",
          "## Methodology",
          "- **C#:** BenchmarkDotNet, `OfficialBenchmarkConfig` — InProcessEmit toolchain, 50 measured",
          "  iterations / 5 warmup, iteration time capped at 25 ms. MemoryDiagnoser on.",
          "- **NumPy:** 50 timed iterations / 10 warmup per op (warm long-lived interpreter).",
          "- **Sizes:** 1,000 / 100,000 / 10,000,000 elements. Same seeds both sides.",
          "- Join keyed on (op, dtype, N).",
          "- **Subsystems** appended to `benchmark-report.md`: NpyIter, Layout, Operand, Cast, Fusion.", ""]
    if size_rows:
        L += ["## Headline — op-matrix geomean by size (NPY/NS, >1 = NumSharp faster)",
              "| Size | geomean | %NumPy🕐 | ✅ / 🟡 / 🟠 / 🔴 |", "|---|--:|--:|---|"]
        for r in size_rows:
            L.append(f"| {r['N']} | {r['geomean']} | {r['pnp']} | "
                     f"{r['ok']} / {r['close']} / {r['slow']} / {r['red']} |")
        L.append("")
    if overall:
        L.append(f"Overall op-matrix: **{overall}**.\n")
    if npy_headline:
        L.append(f"NpyIter: _{npy_headline}_\n")
    if cast_headline:
        L.append(f"Cast: _{cast_headline}_\n")
    L += ["## Files", "| file | what |", "|---|---|",
          "| `benchmark-report.md` | op-matrix (per-(op,dtype,N) ratio) + appended NpyIter/Layout/Operand/Cast/Fusion |",
          "| `benchmark-report.json` / `.csv` | unified machine-readable / spreadsheet form |",
          "| `numpy-results.json` | raw NumPy timings (merge input) |",
          "| `npyiter_results.*` + `cards/` | iterator benchmark sheet + README cards |",
          "| `layout_/operand_/cast_/fusion_results.*` | the four matrix-subsystem sheets |", "",
          "Raw BenchmarkDotNet per-class JSON (~tens of MB) is **not** persisted here "
          "(regenerable). Reproduce with `python benchmark/run_benchmark.py`."]
    return "\n".join(L) + "\n"


def make_snapshot(results_dir=None, snap_name=None, head=None, stage=True,
                  commit=False, quiet=False):
    def log(m):
        if not quiet:
            print(m, flush=True)

    results_dir = Path(results_dir) if results_dir else newest_results_dir()
    # In the normal flow snapshot_history runs at the END of a run, before the report
    # is committed, so HEAD == the benchmarked commit. --head lets an after-the-fact
    # re-snapshot name the run's actual commit (and we read THAT commit's subject).
    head_ref = head if head else "HEAD"
    head = gout("rev-parse", "--short", head_ref) or (head or "unknown")
    subject = gout("log", "-1", "--format=%s", head_ref) or ""
    if len(subject) > 60:
        subject = subject[:59] + "…"
    # "Dirty" for provenance = does the BUILT library-under-test differ from HEAD? i.e.
    # uncommitted changes (tracked-modified OR brand-new) to build-affecting sources —
    # src/ code + the *.csproj/props/targets build graph. It must NOT fire on untracked
    # scratch (benchmark/poc/*.cs probes), gitignored output, or the report files a run
    # always regenerates — otherwise EVERY run (and every CI snapshot) reads DIRTY and
    # "clean (HEAD exactly)" becomes unreachable. Evaluated against the working tree at
    # snapshot time, which in the normal end-of-run flow IS the benchmarked commit.
    def _build_affecting(p):
        p = p.strip().strip('"')
        if "/results/" in p or "/history/" in p:
            return False
        if p.startswith("src/"):
            return p.endswith((".cs", ".csproj", ".props", ".targets"))
        return p.endswith((".csproj", ".props", ".targets"))
    changed = gout("diff", "HEAD", "--name-only").splitlines()       # modified tracked files
    new = [ln[3:] for ln in gout("status", "--porcelain").splitlines()
           if ln.startswith("??")]                                   # untracked (e.g. a new src/*.cs)
    dirty_files = sorted({p for p in (*changed, *new) if _build_affecting(p)})
    dirty = bool(dirty_files)
    run_ts = results_dir.name if results_dir else datetime.now().strftime("%Y%m%d-%H%M%S")
    if snap_name is None:
        date = (f"{run_ts[:4]}-{run_ts[4:6]}-{run_ts[6:8]}"
                if re.match(r"\d{8}-", run_ts) else datetime.now().strftime("%Y-%m-%d"))
        snap_name = f"{date}_{head}"

    snap = HISTORY / snap_name
    snap.mkdir(parents=True, exist_ok=True)
    (snap / "cards").mkdir(exist_ok=True)

    copied = []
    for name in CORE_FROM_RESULTS:
        src = (results_dir / name) if results_dir and (results_dir / name).exists() else (BENCH / name)
        if src.exists():
            shutil.copy(src, snap / name)
            copied.append(name)
        else:
            log(f"  [warn] missing core artifact: {name}")
    for src in SUBSYSTEM_FILES:
        if src.exists():
            shutil.copy(src, snap / src.name)
            copied.append(src.name)
        else:
            log(f"  [warn] missing subsystem sheet: {src.name}")
    for src in CARDS:
        if src.exists():
            shutil.copy(src, snap / "cards" / src.name)
            copied.append(f"cards/{src.name}")

    env = detect_env(results_dir)
    manifest = build_manifest(
        snap_name, run_ts, head, subject, dirty, dirty_files, env,
        parse_size_summary(snap / "benchmark-report.md"),
        parse_overall(snap / "benchmark-report.md"),
        parse_npyiter_headline(), parse_cast_headline())
    (snap / "MANIFEST.md").write_text(manifest, encoding="utf-8")
    log(f"[snapshot] benchmark/history/{snap_name} — {len(copied)} artifacts + MANIFEST.md")

    # latest -> <snap_name> (relative symlink; committed as mode-120000)
    latest = HISTORY / "latest"
    if latest.is_symlink() or latest.exists():
        if latest.is_dir() and not latest.is_symlink():
            shutil.rmtree(latest)
        else:
            latest.unlink()
    try:
        os.symlink(snap_name, latest, target_is_directory=True)
        log(f"[snapshot] latest -> {snap_name} (symlink)")
    except (OSError, NotImplementedError) as e:
        latest.write_text(snap_name + "\n", encoding="utf-8")
        log(f"[snapshot] latest -> {snap_name} (text fallback: {type(e).__name__})")

    if stage:
        r1 = git("add", f"benchmark/history/{snap_name}")
        r2 = git("add", "benchmark/history/latest")
        if r1.returncode == 0 and r2.returncode == 0:
            log("[snapshot] staged snapshot + latest")
            if commit:
                c = git("commit", "-m", f"bench(history): snapshot {snap_name} + latest")
                log("[snapshot] committed" if c.returncode == 0 else f"[snapshot] commit failed: {c.stderr.strip()}")
        else:
            log(f"[snapshot] git add failed (snapshot still written): {r1.stderr or r2.stderr}".strip())
    return snap


def main():
    ap = argparse.ArgumentParser(
        description="Assemble a committable benchmark history snapshot + latest symlink")
    ap.add_argument("--results-dir", help="results/<ts>/ to pull json/csv/numpy from (default: newest)")
    ap.add_argument("--snap-name", help="explicit history/<name> folder (default: <date>_<HEAD-sha>)")
    ap.add_argument("--head", help="override the recorded HEAD short-sha")
    ap.add_argument("--commit", action="store_true", help="also git-commit the snapshot + latest")
    ap.add_argument("--no-stage", action="store_true", help="do not git add")
    args = ap.parse_args()
    make_snapshot(results_dir=args.results_dir, snap_name=args.snap_name, head=args.head,
                  stage=not args.no_stage, commit=args.commit)


if __name__ == "__main__":
    main()
