#!/usr/bin/env python3
"""Shared helpers for the ``master-code-data`` dashboard-data branch tooling.

The ``master-code-data`` orphan branch stores the docs dashboards' generated data as
``<data_type>/<date>_<commithash>/<files>`` snapshots. Each data-type folder carries a
git-symlink ``latest`` pointing at the newest snapshot. ``publish.py`` appends a snapshot
and repoints ``latest``; ``resolve.py`` bakes the newer of master-vs-branch into the DocFX
build inputs by comparing git commit dates (master stays the backwards-compatible fallback).

Stdlib only. Imported by ``publish.py`` / ``resolve.py`` (same directory).
"""
from __future__ import annotations

import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

# The four dashboard data types. For each:
#   master_path : canonical committed file on the code branch. Its git commit date is the
#                 master-side authority for the date-priority comparison. MUST be a real
#                 tracked file (not a path under a symlink, which ``git log`` cannot follow).
#   files       : the files a snapshot carries, or None => copy the whole source directory
#                 (the benchmark history snapshot, which also has cards/ and subsystem sheets).
#   overlay     : list of (snapshot_relpath, worktree_target) the resolver stages into the
#                 DocFX inputs. snapshot_relpath "*" means "the whole snapshot dir" and the
#                 target is treated as a directory.
TYPES: dict[str, dict] = {
    "benchmark": {
        "master_path": "docs/website-src/docs/data/benchmark-report.json",
        "files": None,  # whole snapshot dir (benchmark/history/latest/*)
        "overlay": [
            ("benchmark-report.json",          "docs/website-src/docs/data/benchmark-report.json"),
            ("benchmark-report.managed.json",  "docs/website-src/docs/data/benchmark-report.managed.json"),
            ("benchmark-report.openblas.json", "docs/website-src/docs/data/benchmark-report.openblas.json"),
            ("*",                              "benchmark/history/latest"),
        ],
    },
    "tests-oracle": {
        "master_path": "test/inventory/generated/tests-oracle-report.json",
        "files": ["tests-oracle-report.json", "tests-oracle-report.csv",
                  "tests-oracle-manifest.json", "summary.md"],
        "overlay": [
            ("tests-oracle-report.json",   "test/inventory/generated/tests-oracle-report.json"),
            ("tests-oracle-report.csv",    "test/inventory/generated/tests-oracle-report.csv"),
            ("tests-oracle-manifest.json", "test/inventory/generated/tests-oracle-manifest.json"),
            ("summary.md",                 "test/inventory/generated/summary.md"),
        ],
    },
    "inventory": {
        "master_path": "coverage/generated/coverage.json",
        "files": ["coverage.json", "coverage.csv", "manifest.json", "summary.md"],
        "overlay": [
            ("coverage.json", "coverage/generated/coverage.json"),
            ("coverage.csv",  "coverage/generated/coverage.csv"),
            ("manifest.json", "coverage/generated/manifest.json"),
            ("summary.md",    "coverage/generated/summary.md"),
        ],
    },
    "benchmark-coverage": {
        "master_path": "benchmark/coverage/generated/coverage.json",
        "files": ["coverage.json", "coverage.csv", "summary.md"],
        "overlay": [
            ("coverage.json", "benchmark/coverage/generated/coverage.json"),
            ("coverage.csv",  "benchmark/coverage/generated/coverage.csv"),
            ("summary.md",    "benchmark/coverage/generated/summary.md"),
        ],
    },
}

DATE_FMT = "%Y-%m-%d"


def today_utc() -> str:
    return datetime.now(timezone.utc).strftime(DATE_FMT)


def short_sha(sha: str) -> str:
    return sha.strip()[:8]


def make_stamp(date_str: str, sha: str) -> str:
    return f"{date_str}_{short_sha(sha)}"


def stamp_date(name: str) -> str:
    """The ``YYYY-MM-DD`` prefix of a ``<date>_<hash>`` folder name."""
    return name.split("_", 1)[0]


def is_stamp(name: str) -> bool:
    try:
        datetime.strptime(stamp_date(name), DATE_FMT)
        return True
    except ValueError:
        return False


def list_stamps(type_dir: Path) -> list[str]:
    if not type_dir.is_dir():
        return []
    return sorted(p.name for p in type_dir.iterdir() if p.is_dir() and is_stamp(p.name))


def resolve_latest(type_dir: Path) -> Path | None:
    """Newest snapshot dir: follow ``latest`` if it resolves, else the max ``<date>_<hash>`` folder."""
    link = type_dir / "latest"
    if link.exists():
        target = link.resolve()
        if target.is_dir():
            return target
    stamps = list_stamps(type_dir)
    return type_dir / stamps[-1] if stamps else None


def repoint_latest(type_dir: Path, stamp: str) -> None:
    """(Re)create the ``latest`` symlink pointing at ``stamp`` (relative target, like benchmark/history)."""
    link = type_dir / "latest"
    if link.is_symlink() or link.exists():
        if link.is_dir() and not link.is_symlink():
            shutil.rmtree(link)
        else:
            link.unlink()
    os.symlink(stamp, link, target_is_directory=True)


def git(args: list[str], cwd: Path | str | None = None, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd is not None else None,
        capture_output=True, text=True,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def commit_date_iso(repo: Path | str, path: str, ref: str | None = None) -> str | None:
    """Committer date (strict ISO-8601) of the last commit touching ``path`` under ``ref`` (or worktree)."""
    args = ["log", "-1", "--format=%cI"]
    if ref:
        args.append(ref)
    args += ["--", path]
    out = git(args, cwd=repo, check=False)
    return out or None


def parse_iso(s: str) -> datetime:
    """Parse a git ``%cI`` timestamp or a bare ``YYYY-MM-DD`` (treated as UTC midnight)."""
    s = s.strip()
    if len(s) == 10:
        return datetime.strptime(s, DATE_FMT).replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(s)


def copy_tree_contents(src_dir: Path, dst_dir: Path, skip: set[str] | None = None) -> None:
    """Copy every entry of ``src_dir`` (files and subdirs) into ``dst_dir``; ``src_dir`` may be a symlink."""
    skip = skip or set()
    real = src_dir.resolve()
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in real.iterdir():
        if item.name in skip:
            continue
        target = dst_dir / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)
