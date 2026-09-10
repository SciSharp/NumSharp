#!/usr/bin/env python3
"""Shared helpers for the ``data`` branch dashboard-data tooling.

The ``data`` orphan branch stores the docs dashboards' generated data as
``<data_type>/<date>_<commithash>/<files>`` snapshots. Each data-type folder also carries a
real ``latest/`` directory — a data-only copy of the newest snapshot — that the docs build
consumes through the ``refs/data`` submodule (floated to the branch tip). ``publish.py`` appends
a snapshot and refreshes ``latest/``.

Stdlib only. Imported by ``publish.py`` (same directory).
"""
from __future__ import annotations

import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

# The four dashboard data types. For each, ``files`` lists the files a snapshot carries, or
# None => copy the whole source directory (the benchmark history snapshot, which also has
# cards/ and subsystem sheets).
TYPES: dict[str, dict] = {
    "benchmark":          {"files": None},
    "tests-oracle":       {"files": ["tests-oracle-report.json", "tests-oracle-report.csv",
                                      "tests-oracle-manifest.json", "summary.md"]},
    "inventory":          {"files": ["coverage.json", "coverage.csv", "manifest.json", "summary.md"]},
    "benchmark-coverage": {"files": ["coverage.json", "coverage.csv", "summary.md"]},
}

DATE_FMT = "%Y-%m-%d"


def today_utc() -> str:
    return datetime.now(timezone.utc).strftime(DATE_FMT)


def short_sha(sha: str) -> str:
    return sha.strip()[:8]


def make_stamp(date_str: str, sha: str) -> str:
    return f"{date_str}_{short_sha(sha)}"


def git(args: list[str], cwd: Path | str | None = None, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd is not None else None,
        capture_output=True, text=True,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


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


def refresh_latest_dir(type_dir: Path, stamp: str) -> None:
    """Refresh the real ``latest/`` directory to mirror ``type_dir/stamp`` (data only, no README).

    The refs/data submodule presents ``<type>/latest`` as a plain folder the docs build reads, so
    ``latest`` MUST be a real directory (never a symlink) and carry no ``README.md`` — the per-type
    README stays a sibling. Any previous ``latest`` is replaced.
    """
    latest = type_dir / "latest"
    if latest.is_symlink() or latest.is_file():
        latest.unlink()
    elif latest.is_dir():
        shutil.rmtree(latest)
    copy_tree_contents(type_dir / stamp, latest)
