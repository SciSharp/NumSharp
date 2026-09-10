#!/usr/bin/env python3
"""Append a dashboard-data snapshot to the ``data`` branch and refresh its ``latest/`` dir.

Usage::

    python tools/dashboard_data/publish.py \
        --type <benchmark|tests-oracle|inventory|benchmark-coverage> \
        --from <generated-dir> \
        --branch-worktree <checkout-of-the-data-branch> \
        [--sha <commithash>] [--date <YYYY-MM-DD>] [--commit]

Copies the type's files from ``--from`` into ``<branch>/<type>/<date>_<sha>/``, refreshes the real
``<type>/latest/`` directory (a data-only copy the docs build sparse-clones), and (with
``--commit``) makes one small commit on the branch worktree. ``--from`` for ``benchmark`` is the
``benchmark/history/latest`` snapshot dir (symlink ok).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def _git(args: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, check=check)


def pull_branch_worktree(branch: Path, remote: str = "origin", ref: str = "data") -> None:
    """Sync the data-branch worktree to the remote ``ref`` tip before writing.

    Fetches, ensures the worktree is on branch ``ref`` (a fresh submodule checkout is a detached HEAD),
    and fast-forwards. Keeps local un-pushed commits when the worktree is ahead; warns (does NOT abort)
    when the branch has diverged, so the new snapshot still lands and the divergence surfaces at push.
    """
    fetch = _git(["fetch", remote, ref], branch, check=False)
    if fetch.returncode != 0:
        print(f"  warn: --pull: `git fetch {remote} {ref}` failed: {fetch.stderr.strip()}", file=sys.stderr)
        return
    on_branch = _git(["symbolic-ref", "--quiet", "--short", "HEAD"], branch, check=False).stdout.strip()
    if on_branch != ref:
        _git(["checkout", "-B", ref, "FETCH_HEAD"], branch)  # detached/other HEAD -> point `ref` at the tip
        print(f"  --pull: checked out {ref} at {remote}/{ref} tip")
        return
    ff = _git(["merge", "--ff-only", f"{remote}/{ref}"], branch, check=False)
    if ff.returncode != 0:
        print(f"  warn: --pull: cannot fast-forward {ref} to {remote}/{ref} (diverged or local commits ahead); "
              f"keeping local HEAD — push/rebase refs/data before publishing to avoid a non-fast-forward push",
              file=sys.stderr)
    else:
        print(f"  --pull: {ref} fast-forwarded to {remote}/{ref} tip")


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
    ap.add_argument("--branch-worktree", required=True, help="Checkout of the data branch")
    ap.add_argument("--sha", default=None, help="Source commit hash (default: HEAD of --from's repo)")
    ap.add_argument("--date", default=None, help="Snapshot date YYYY-MM-DD (default: today UTC)")
    ap.add_argument("--commit", action="store_true", help="git add+commit the snapshot on the branch")
    ap.add_argument("--pull", action="store_true",
                    help="Before writing, sync the branch worktree to the origin/data tip "
                         "(fetch + fast-forward; keeps local un-pushed commits)")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    if not src.exists():
        print(f"error: --from does not exist: {src}", file=sys.stderr)
        return 2
    branch = Path(args.branch_worktree).resolve()
    if not branch.is_dir():
        print(f"error: --branch-worktree does not exist: {branch}", file=sys.stderr)
        return 2

    # A producer checkout must be FULL: a sparse worktree (cone */latest) would make `git add` skip
    # the new <date>_<sha> snapshot dir. Disabling sparse is a no-op on an already-full checkout.
    if (branch / ".git").exists():
        _git(["sparse-checkout", "disable"], branch, check=False)
    if args.pull:
        pull_branch_worktree(branch)

    sha = args.sha or default_sha(src)
    date = args.date or common.today_utc()
    stamp = common.make_stamp(date, sha)

    type_dir = branch / args.type
    type_dir.mkdir(parents=True, exist_ok=True)
    dest = type_dir / stamp
    if dest.exists():
        shutil.rmtree(dest)  # re-publish of the same date+commit overwrites
    copy_snapshot(args.type, src, dest)
    common.refresh_latest_dir(type_dir, stamp)
    print(f"published {args.type}: {stamp}")

    if args.commit:
        common.git(["add", "-A", args.type], cwd=branch)
        if common.git(["status", "--porcelain"], cwd=branch, check=False):
            common.git(["commit", "-m", f"publish({args.type}): {stamp}"], cwd=branch)
            print(f"committed publish({args.type}): {stamp}")
        else:
            print("nothing to commit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
