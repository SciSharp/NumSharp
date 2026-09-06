#!/usr/bin/env python3
"""Contract tests for the Managed/OpenBLAS profile coverage gate."""

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[2] / "backends" / "backend_profiles.py"
SPEC = importlib.util.spec_from_file_location("backend_profiles", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(module)


def complete_envelope():
    rows = []
    for (operation, scenario), sizes in module.EXPECTED_COVERAGE.items():
        for n in sizes:
            rows.append({"operation": operation, "dtype": "float64", "n": n, "scenario": scenario})
    return {"rows": rows}


class BackendProfileCoverageTests(unittest.TestCase):
    def test_completed_zero_duration_is_negligible_not_no_data(self):
        result = module.profile_result(
            "managed", {"availability": "available", "numsharp_ms": 0.0}, 0.0002)
        self.assertEqual("negligible", result["status"])
        self.assertIsNone(result["ratio"])

    def test_complete_symmetric_profiles_pass(self):
        managed = complete_envelope()
        openblas = complete_envelope()
        module.validate_coverage(managed, openblas)

    def test_missing_required_tier_fails(self):
        managed = complete_envelope()
        openblas = complete_envelope()
        managed["rows"] = [row for row in managed["rows"]
                           if not (row["operation"] == "np.matvec(a, b)" and row["n"] == 10_000_000)]
        openblas["rows"] = list(managed["rows"])
        with self.assertRaisesRegex(RuntimeError, "coverage contract failed"):
            module.validate_coverage(managed, openblas)

    def test_asymmetric_profiles_fail(self):
        managed = complete_envelope()
        openblas = complete_envelope()
        openblas["rows"] = openblas["rows"][:-1]
        with self.assertRaisesRegex(RuntimeError, "different cells"):
            module.validate_coverage(managed, openblas)


if __name__ == "__main__":
    unittest.main()
