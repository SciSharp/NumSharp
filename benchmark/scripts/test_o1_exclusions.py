#!/usr/bin/env python3
"""Contract tests for semantic O(1)-in-N benchmark exclusions."""

import importlib.util
import json
import re
import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import credibility


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


merge_results = load("merge_results_o1_test", SCRIPT_DIR / "merge-results.py")
merge_backends = load("merge_backends_o1_test", SCRIPT_DIR / "merge-backend-profiles.py")


class O1ExclusionTests(unittest.TestCase):
    def test_every_canonical_o1_key_is_semantically_negligible(self):
        for operation in credibility.O1_OPERATION_KEYS:
            with self.subTest(operation=operation):
                self.assertEqual(
                    "negligible", merge_results.classify(10.0, 1.0, 10.0, operation))
                self.assertEqual(
                    "negligible", merge_backends.performance_status(10.0, 1.0, operation)[2])

        for prefix in credibility.O1_VIEW_PREFIXES:
            operation = f"{prefix}(a, scenario)"
            with self.subTest(operation=operation):
                self.assertEqual(
                    "negligible", merge_results.classify(10.0, 1.0, 10.0, operation))
                self.assertEqual(
                    "negligible", merge_backends.performance_status(10.0, 1.0, operation)[2])

    def test_real_work_is_not_semantically_excluded(self):
        self.assertEqual("faster", merge_results.classify(10.0, 1.0, 10.0, "np.add(a, b)"))
        self.assertEqual("faster", merge_backends.performance_status(10.0, 1.0, "np.add(a, b)")[2])
        self.assertIsNone(credibility.o1_exclusion_reason("np.add"))
        self.assertIsNone(credibility.o1_display_exclusion_reason("np.add(a, b) (float64)"))

    def test_display_policy_distinguishes_diag_view_from_construction(self):
        self.assertIsNotNone(credibility.o1_display_exclusion_reason("np.diag"))
        self.assertIsNotNone(credibility.o1_display_exclusion_reason("np.diag(a2d)"))
        self.assertIsNone(credibility.o1_display_exclusion_reason("np.diag(a1d)"))

    def test_dashboard_policy_exactly_matches_python_policy(self):
        dashboard = (REPO / "docs/website-src/docs/benchmarks-dashboard.md").read_text(
            encoding="utf-8")
        function_arrays = re.findall(
            r"const o1FunctionNames = new Set\((\[.*?\])\);", dashboard, flags=re.S)
        exact_arrays = re.findall(
            r"const o1ExactOperations = new Set\((\[.*?\])\);", dashboard, flags=re.S)
        self.assertEqual(2, len(function_arrays))
        self.assertEqual(2, len(exact_arrays))
        for encoded in function_arrays:
            self.assertEqual(credibility.O1_FUNCTION_NAMES, frozenset(json.loads(encoded)))
        for encoded in exact_arrays:
            self.assertEqual(credibility.O1_EXACT_DISPLAY_OPERATIONS, frozenset(json.loads(encoded)))

    def test_every_dashboard_rollup_uses_credible_rows(self):
        dashboard = (REPO / "docs/website-src/docs/benchmarks-dashboard.md").read_text(
            encoding="utf-8")
        required_guards = [
            "const credible = rows.filter(isCredibleRow);",
            "score: geomeanRows(credibleRows)",
            "const credibleCellRows = cellRows.filter(isCredibleRow);",
            "const bestRows = (effectiveGroups.get(name) || []).filter(isCredibleRow);",
            'rows.filter((row) => row.availability === "available" && isCredibleRow(row))',
            "isCredibleRow(row) && suiteKey(row.suite) === key && matchesTier(row)",
            "if (!isCredibleRow(row)) return false;",
            ".filter(isCredibleRow)\n      .map((row) => Number(row.ratio));",
        ]
        for guard in required_guards:
            with self.subTest(guard=guard):
                self.assertIn(guard, dashboard)

    def test_negligible_rows_only_enter_the_dedicated_status_tooltip(self):
        dashboard = (REPO / "docs/website-src/docs/benchmarks-dashboard.md").read_text(
            encoding="utf-8")
        self.assertEqual(
            2, dashboard.count('const isExplorerRow = (row) => row.status !== "negligible";'))
        self.assertIn("initializeFunctionExplorer(rows.filter(isExplorerRow));", dashboard)
        for profile_rows in ("coreRows", "nativeRows", "effectiveRows"):
            self.assertIn(f"{profile_rows} = {profile_rows}.filter(isExplorerRow);", dashboard)

        # The exception is deliberate and singular: only the Negligible status band matches
        # negligible rows and feeds them into its evidence tooltip.
        self.assertEqual(1, dashboard.count("if (band.negligible) {"))
        self.assertEqual(1, dashboard.count('return row.status === "negligible";'))


if __name__ == "__main__":
    unittest.main()
