#!/usr/bin/env python3
"""Bake the newest dashboard data (master vs ``master-code-data``) into the DocFX inputs.

For each data type, compares the git commit date of master's canonical file against the
commit date of the branch's ``<type>/latest`` and, if the branch is newer, copies the branch
snapshot over the working-tree paths DocFX reads. master's committed copies stay the fallback,
so the docs still build correctly from master alone.

Usage::

    python tools/dashboard_data/resolve.py \
        --data-worktree <checkout-of-master-code-data> \
        --into <code-checkout-root> [--only <type> ...]

Prints one line per data type: which side won and its commit date.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def overlay(type_name: str, snapshot: Path, into: Path) -> None:
    for src_rel, dst_rel in common.TYPES[type_name]["overlay"]:
        dst = into / dst_rel
        if src_rel == "*":
            common.copy_tree_contents(snapshot, dst, skip={"latest"})
        else:
            s = snapshot / src_rel
            if not s.exists():
                print(f"  warn: {type_name}: snapshot missing {src_rel}", file=sys.stderr)
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, dst)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-worktree", required=True, help="Checkout of master-code-data")
    ap.add_argument("--into", default=".", help="Code checkout root to stage winners into")
    ap.add_argument("--only", nargs="*", choices=sorted(common.TYPES), help="Limit to these types")
    args = ap.parse_args()

    data = Path(args.data_worktree).resolve()
    into = Path(args.into).resolve()
    if not data.is_dir():
        print(f"error: --data-worktree does not exist: {data}", file=sys.stderr)
        return 2
    types = args.only or sorted(common.TYPES)

    for t in types:
        spec = common.TYPES[t]
        snapshot = common.resolve_latest(data / t)
        if snapshot is None:
            print(f"{t:18} master  (branch has no snapshot)")
            continue

        branch_iso = common.commit_date_iso(data, f"{t}/latest") or common.stamp_date(snapshot.name)
        master_iso = common.commit_date_iso(into, spec["master_path"])

        branch_wins = master_iso is None or common.parse_iso(branch_iso) > common.parse_iso(master_iso)
        if branch_wins:
            overlay(t, snapshot, into)
            print(f"{t:18} BRANCH  won @ {branch_iso}  (master @ {master_iso})  -> {snapshot.name}")
        else:
            print(f"{t:18} master  won @ {master_iso}  (branch @ {branch_iso})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
