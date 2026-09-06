#!/usr/bin/env python3
"""Append a dashboard-data snapshot to the ``master-code-data`` branch and repoint ``latest``.

Usage::

    python tools/dashboard_data/publish.py \
        --type <benchmark|tests-oracle|inventory|benchmark-coverage> \
        --from <generated-dir> \
        --branch-worktree <checkout-of-master-code-data> \
        [--sha <commithash>] [--date <YYYY-MM-DD>] [--commit]

Copies the type's files from ``--from`` into ``<branch>/<type>/<date>_<sha>/``, repoints the
``latest`` symlink, and (with ``--commit``) makes one small commit on the branch worktree.
``--from`` for ``benchmark`` is the ``benchmark/history/latest`` snapshot dir (symlink ok).
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def copy_snapshot(type_name: str, src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    spec = common.TYPES[type_name]
    if spec["files"] is None:
        # whole directory (benchmark history snapshot: files + cards/ + subsystem sheets)
        common.copy_tree_contents(src, dest, skip={"latest"})
    else:
        for name in spec["files"]:
            f = src / name
            if f.exists():
                shutil.copy2(f, dest / name)
            else:
                print(f"  warn: {type_name}: source missing {name}", file=sys.stderr)


def default_sha(src: Path) -> str:
    repo = src if src.is_dir() else src.parent
    return common.git(["rev-parse", "--short=8", "HEAD"], cwd=repo)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--type", required=True, choices=sorted(common.TYPES))
    ap.add_argument("--from", dest="src", required=True, help="Directory of freshly generated files")
    ap.add_argument("--branch-worktree", required=True, help="Checkout of the master-code-data branch")
    ap.add_argument("--sha", default=None, help="Source commit hash (default: HEAD of --from's repo)")
    ap.add_argument("--date", default=None, help="Snapshot date YYYY-MM-DD (default: today UTC)")
    ap.add_argument("--commit", action="store_true", help="git add+commit the snapshot on the branch")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    if not src.exists():
        print(f"error: --from does not exist: {src}", file=sys.stderr)
        return 2
    branch = Path(args.branch_worktree).resolve()
    if not branch.is_dir():
        print(f"error: --branch-worktree does not exist: {branch}", file=sys.stderr)
        return 2

    sha = args.sha or default_sha(src)
    date = args.date or common.today_utc()
    stamp = common.make_stamp(date, sha)

    type_dir = branch / args.type
    type_dir.mkdir(parents=True, exist_ok=True)
    dest = type_dir / stamp
    if dest.exists():
        shutil.rmtree(dest)  # re-publish of the same date+commit overwrites
    copy_snapshot(args.type, src, dest)
    common.repoint_latest(type_dir, stamp)
    print(f"published {args.type}: {stamp}")

    if args.commit:
        common.git(["config", "core.symlinks", "true"], cwd=branch, check=False)
        common.git(["add", "-A", args.type], cwd=branch)
        if common.git(["status", "--porcelain"], cwd=branch, check=False):
            common.git(["commit", "-m", f"publish({args.type}): {stamp}"], cwd=branch)
            print(f"committed publish({args.type}): {stamp}")
        else:
            print("nothing to commit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
