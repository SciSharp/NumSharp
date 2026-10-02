#!/usr/bin/env python3
"""Audit dashboard links against NumPy's official latest-stable Sphinx inventory."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
import zlib
from pathlib import Path
from urllib.parse import urljoin

from numpy_documentation import DOCS_BASE_URL, SNAPSHOT_PATH


INVENTORY_URL = "https://numpy.org/doc/stable/objects.inv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "coverage_json",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent / "generated" / "coverage.json",
    )
    parser.add_argument(
        "--refresh-inventory", action="store_true",
        help="Refresh the checked-in API-link snapshot from the official index, then exit.",
    )
    return parser.parse_args()


def download_inventory() -> bytes:
    request = urllib.request.Request(INVENTORY_URL, headers={"User-Agent": "NumSharp coverage audit/1.0"})
    error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read()
        except Exception as current_error:  # pragma: no cover - network-dependent retry
            error = current_error
            if attempt < 2:
                time.sleep(1 + attempt)
    raise SystemExit(f"Unable to download NumPy documentation inventory: {error}")


def inventory_entries(raw: bytes) -> list[tuple[str, str, str]]:
    """Read names, roles and resolved relative URLs from Sphinx inventory v2."""
    if not raw.startswith(b"# Sphinx inventory version 2\n"):
        raise ValueError("Expected a Sphinx inventory version 2 file.")
    position = 0
    for _ in range(4):
        position = raw.index(b"\n", position) + 1
    entries = zlib.decompress(raw[position:]).decode("utf-8").splitlines()
    result: list[tuple[str, str, str]] = []
    for entry in entries:
        parts = entry.split(" ", 4)
        if len(parts) != 5:
            continue
        result.append((parts[0], parts[1], parts[3].replace("$", parts[0])))
    return result


def documented_targets(raw: bytes) -> set[str]:
    """Include both indexed anchors and their pages; nonexistent anchors still fail."""
    targets: set[str] = set()
    for _, _, uri in inventory_entries(raw):
        url = urljoin(DOCS_BASE_URL, uri)
        targets.add(url)
        targets.add(url.split("#", 1)[0])
    return targets


def documented_pages(raw: bytes) -> set[str]:
    """Retain the page-only helper for existing external audit callers."""
    return {url for url in documented_targets(raw) if "#" not in url}


def refresh_snapshot(raw: bytes) -> None:
    header = raw.split(b"\n", 4)
    version = header[2].decode("utf-8").removeprefix("# Version: ")
    api_urls = {
        name: uri for name, role, uri in inventory_entries(raw)
        if name.startswith("numpy.") and role.startswith("py:")
    }
    if not api_urls:
        raise ValueError("Official NumPy inventory contains no Python API targets.")
    snapshot = {
        "schema_version": 1,
        "source_url": INVENTORY_URL,
        "documentation_version": version,
        "api_urls": dict(sorted(api_urls.items())),
    }
    SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"Saved {len(api_urls)} official NumPy {version} API links to {SNAPSHOT_PATH}.")


def main() -> None:
    args = parse_args()
    raw = download_inventory()
    if args.refresh_inventory:
        refresh_snapshot(raw)
        return
    payload = json.loads(args.coverage_json.read_text(encoding="utf-8"))
    targets = documented_targets(raw)
    rows = [
        row for row in payload["rows"]
        if row["origin"] == "numpy"
    ]
    missing = [
        (row["id"], row["documentation_url"])
        for row in rows
        if (row["documentation_url"] and row["documentation_url"] not in targets)
        or (row["in_default_scope"] and not row["documentation_url"])
    ]
    if missing:
        details = "\n".join(f"  - {api_id}: {url or '<empty>'}" for api_id, url in missing)
        raise SystemExit(f"{len(missing)} coverage links are absent from NumPy's official inventory:\n{details}")
    linked = sum(bool(row["documentation_url"]) for row in rows)
    print(f"Validated {linked} latest-stable NumPy documentation links and anchors across all catalog scopes against objects.inv.")


if __name__ == "__main__":
    main()
