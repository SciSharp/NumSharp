#!/usr/bin/env python3
"""Focused contract tests for the backend-profile merge schema."""

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("merge-backend-profiles.py")
SPEC = importlib.util.spec_from_file_location("merge_backend_profiles", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(module)


def row(profile, availability, numsharp_ms=None):
    source = {
        "operation": "np.linalg.solve(a, b)",
        "suite": "LinearAlgebra",
        "category": "openblas_required",
        "dtype": "float64",
        "n": 32,
        "scenario": "",
        "numpy_ms": 1.0,
        "backend_route": "openblas_required",
    }
    source["result"] = module.profile_result(
        profile, 1.0, numsharp_ms, availability,
        exception_type="NotSupportedException" if availability == module.NOT_SUPPORTED else None,
    )
    return source


class BackendProfileMergeTests(unittest.TestCase):
    def test_not_supported_is_preserved_as_availability(self):
        result = module.profile_result(
            "openblas", 1.0, None, module.NOT_SUPPORTED,
            exception_type="NotSupportedException", exception_message="unsupported mode",
        )
        self.assertEqual(module.NOT_SUPPORTED, result["availability"])
        self.assertEqual("NotSupportedException", result["exception_type"])
        self.assertIsNone(result["ratio"])

    def test_missing_and_not_supported_do_not_create_effective_timing(self):
        managed = module.profile_envelope("managed", [row("managed", module.MISSING_BACKEND)])
        openblas = module.profile_envelope("openblas", [row("openblas", module.NOT_SUPPORTED)])
        combined = module.combine(managed, openblas)
        self.assertEqual(1, len(combined["rows"]))
        self.assertIsNone(combined["rows"][0]["effective_profile"])
        self.assertEqual("no_data", combined["rows"][0]["status"])

    def test_fastest_valid_profile_wins_exact_cell(self):
        managed = module.profile_envelope("managed", [row("managed", module.AVAILABLE, 0.8)])
        openblas = module.profile_envelope("openblas", [row("openblas", module.AVAILABLE, 0.2)])
        combined = module.combine(managed, openblas)
        self.assertEqual("openblas", combined["rows"][0]["effective_profile"])
        self.assertEqual(0.2, combined["rows"][0]["numsharp_ms"])


if __name__ == "__main__":
    unittest.main()
