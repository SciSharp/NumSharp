#!/usr/bin/env python3
"""Audit dashboard links against NumPy's official latest-stable Sphinx inventory."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
import zlib
from pathlib import Path


INVENTORY_URL = "https://numpy.org/doc/stable/objects.inv"
DOCS_BASE_URL = "https://numpy.org/doc/stable/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "coverage_json",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent / "generated" / "coverage.json",
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


def documented_pages(raw: bytes) -> set[str]:
    position = 0
    for _ in range(4):
        position = raw.index(b"\n", position) + 1
    entries = zlib.decompress(raw[position:]).decode("utf-8").splitlines()
    pages: set[str] = set()
    for entry in entries:
        parts = entry.split(" ", 4)
        if len(parts) != 5:
            continue
        uri = parts[3].replace("$", parts[0]).split("#", 1)[0]
        pages.add(DOCS_BASE_URL + uri)
    return pages


def main() -> None:
    args = parse_args()
    payload = json.loads(args.coverage_json.read_text(encoding="utf-8"))
    pages = documented_pages(download_inventory())
    rows = [
        row for row in payload["rows"]
        if row["origin"] == "numpy" and row["in_default_scope"]
    ]
    missing = [
        (row["id"], row["documentation_url"])
        for row in rows
        if not row["documentation_url"] or row["documentation_url"].split("#", 1)[0] not in pages
    ]
    if missing:
        details = "\n".join(f"  - {api_id}: {url or '<empty>'}" for api_id, url in missing)
        raise SystemExit(f"{len(missing)} coverage links are absent from NumPy's official inventory:\n{details}")
    print(f"Validated {len(rows)} latest-stable NumPy documentation links against objects.inv.")


if __name__ == "__main__":
    main()
