#!/usr/bin/env python3
"""Focused tests for benchmark snapshot manifest parsing."""

import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parent.parent / "snapshot_history.py"
SPEC = importlib.util.spec_from_file_location("snapshot_history", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(module)


class SnapshotHistoryTests(unittest.TestCase):
    def parse(self, header, row):
        text = f"## Summary by size\n\n{header}\n|---|\n{row}\n\n---\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.md"
            path.write_text(text, encoding="utf-8")
            return module.parse_size_summary(path)

    def test_summary_without_failure_column(self):
        rows = self.parse(
            "| N | ops | faster | close | slower | much | negl | n/a | geomean | %NP |",
            "| 1,000 | 20 | 1 | 2 | 3 | 4 | 5 | 5 | 0.75x | 133% |",
        )
        self.assertEqual("0.75x", rows[0]["geomean"])
        self.assertEqual("133%", rows[0]["pnp"])

    def test_summary_with_failure_column(self):
        rows = self.parse(
            "| N | ops | faster | close | slower | much | negl | n/a | fail | geomean | %NP |",
            "| 1,000 | 20 | 1 | 2 | 3 | 4 | 5 | 4 | 1 | 0.75x | 133% |",
        )
        self.assertEqual("0.75x", rows[0]["geomean"])
        self.assertEqual("133%", rows[0]["pnp"])


if __name__ == "__main__":
    unittest.main()
