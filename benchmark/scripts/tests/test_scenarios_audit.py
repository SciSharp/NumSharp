#!/usr/bin/env python3
"""Regression contract for the generated scenario × dtype audit."""

import json
import re
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
HTML = REPO / "benchmark" / "scenarios.html"
EXPECTED_DTYPES = [
    "bool", "uint8", "int8", "int16", "uint16", "int32", "uint32", "int64", "uint64",
    "char", "float16", "float32", "float64", "decimal", "complex128",
]


def load_audit():
    text = HTML.read_text(encoding="utf-8")
    match = re.search(
        r'<script id="scenario-audit-data" type="application/json">(.*?)</script>', text, re.S)
    if match is None:
        raise AssertionError("scenarios.html has no embedded audit payload")
    return text, json.loads(match.group(1))


class ScenarioAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html, cls.audit = load_audit()
        cls.by_identity = {row["identity"]: row for row in cls.audit["rows"]}

    def test_all_15_numsharp_dtypes_are_columns(self):
        self.assertEqual(EXPECTED_DTYPES, [dtype["key"] for dtype in self.audit["dtypes"]])
        for row in self.audit["rows"]:
            self.assertEqual(set(EXPECTED_DTYPES), set(row["cells"]), row["title"])

    def test_every_discovered_title_is_unique_and_has_both_twins(self):
        rows = self.audit["rows"]
        self.assertEqual(499, len(rows))
        self.assertEqual(len(rows), len({row["identity"] for row in rows}))
        self.assertFalse([row["identity"] for row in rows if not row["csharp_titles"]])
        self.assertFalse([row["identity"] for row in rows if not row["python_titles"]])
        self.assertEqual(0, self.audit["totals"]["python_only"])

    def test_totals_reconcile_from_cells(self):
        cells = [cell for row in self.audit["rows"] for cell in row["cells"].values()]
        expected = {
            "scenarios": len(self.audit["rows"]),
            "csharp": sum(cell["csharp"] for cell in cells),
            "python": sum(cell["python"] for cell in cells),
            "comparable": sum(cell["comparable"] for cell in cells),
            "csharp_only": sum(cell["state"] == "csharp_only" for cell in cells),
            "python_only": sum(cell["state"] == "python_only" for cell in cells),
        }
        self.assertEqual(expected, self.audit["totals"])
        self.assertEqual({
            "scenarios": 499,
            "csharp": 499 * 15,
            "python": 1699,
            "comparable": 1699,
            "csharp_only": 499 * 15 - 1699,
            "python_only": 0,
        }, self.audit["totals"])

    def test_every_csharp_function_has_all_15_dtype_cells(self):
        missing = [
            (row["identity"], dtype)
            for row in self.audit["rows"]
            for dtype in EXPECTED_DTYPES
            if not row["cells"][dtype]["csharp"]
        ]
        self.assertEqual([], missing)

    def test_every_function_has_a_float64_comparison_or_rejection_evidence(self):
        missing = [
            row["identity"] for row in self.audit["rows"]
            if not row["cells"]["float64"]["comparable"]
        ]
        self.assertEqual([], missing)

    def test_every_function_has_an_int32_comparison(self):
        missing = [
            row["identity"] for row in self.audit["rows"]
            if not row["cells"]["int32"]["comparable"]
        ]
        self.assertEqual([], missing)

    def test_can_cast_exercises_every_csharp_source_dtype(self):
        row = self.by_identity["np.can_cast"]
        self.assertEqual("both", row["cells"]["float64"]["state"])
        self.assertEqual("both", row["cells"]["int32"]["state"])
        self.assertTrue(all(
            row["cells"][dtype]["state"] == "csharp_only"
            for dtype in EXPECTED_DTYPES if dtype not in {"int32", "float64"}))

    def test_nanmean_has_complete_csharp_coverage_without_inventing_numpy_cells(self):
        row = self.by_identity["np.nanmean"]
        self.assertEqual(
            {"int32", "float16", "float32", "float64"},
            {dtype for dtype, cell in row["cells"].items() if cell["state"] == "both"})
        self.assertEqual(
            set(EXPECTED_DTYPES) - {"int32", "float16", "float32", "float64"},
            {dtype for dtype, cell in row["cells"].items() if cell["state"] == "csharp_only"})

    def test_html_has_all_three_single_mark_lenses_and_filters(self):
        for value in ("comparable", "csharp", "python"):
            self.assertIn(f'data-lens="{value}"', self.html)
        self.assertIn("Has ❌ in selected lens", self.html)
        self.assertIn("All 15 ✅ in selected lens", self.html)
        self.assertIn("scenario-audit-data", self.html)


if __name__ == "__main__":
    unittest.main()
