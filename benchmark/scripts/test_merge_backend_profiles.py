#!/usr/bin/env python3
"""Focused contract tests for the backend-profile merge schema."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("merge-backend-profiles.py")
SPEC = importlib.util.spec_from_file_location("merge_backend_profiles", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(module)


def row(profile, availability, numsharp_ms=None, numpy_ms=1.0,
        operation="np.linalg.solve(a, b)", actual_backend=None):
    source = {
        "operation": operation,
        "suite": "LinearAlgebra",
        "category": "openblas_required",
        "dtype": "float64",
        "n": 32,
        "scenario": "",
        "numpy_ms": numpy_ms,
        "backend_route": "openblas_required",
    }
    source["result"] = module.profile_result(
        profile, numpy_ms, numsharp_ms, availability,
        actual_backend=actual_backend,
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
        self.assertEqual("openblas", combined["rows"][0]["effective_backend"])
        self.assertEqual(0.2, combined["rows"][0]["numsharp_ms"])

    def test_joined_profiles_share_the_canonical_numpy_baseline(self):
        managed = module.profile_envelope("managed", [row("managed", module.AVAILABLE, 0.5, numpy_ms=1.0)])
        openblas = module.profile_envelope("openblas", [row("openblas", module.AVAILABLE, 0.25, numpy_ms=2.0)])
        combined = module.combine(managed, openblas)
        cell = combined["rows"][0]

        self.assertEqual(1.0, cell["numpy_ms"])
        self.assertEqual(2.0, cell["profiles"]["managed"]["ratio"])
        self.assertEqual(4.0, cell["profiles"]["openblas"]["ratio"])
        self.assertEqual(4.0, cell["ratio"])
        # combine() must not rewrite the standalone OpenBLAS envelope's own provenance.
        self.assertEqual(8.0, openblas["rows"][0]["result"]["ratio"])

    def test_complete_linear_algebra_profile_is_required(self):
        managed = module.profile_envelope("managed", [
            row("managed", module.AVAILABLE, 0.5),
            row("managed", module.AVAILABLE, 0.1, operation="np.diagonal(a)"),
        ])
        openblas = module.profile_envelope("openblas", [row("openblas", module.AVAILABLE, 0.25)])

        with self.assertRaisesRegex(RuntimeError, "complete measured LinearAlgebra suite"):
            module.combine(managed, openblas)

    def test_managed_control_can_be_measured_in_both_profiles(self):
        operation = "np.diagonal(a)"
        managed = module.profile_envelope(
            "managed", [row("managed", module.AVAILABLE, 0.5, operation=operation)])
        openblas = module.profile_envelope(
            "openblas", [row("openblas", module.AVAILABLE, 0.45, operation=operation,
                               actual_backend="managed")])

        combined = module.combine(managed, openblas)
        self.assertEqual(1, len(combined["rows"]))
        self.assertEqual("openblas", combined["rows"][0]["effective_profile"])
        self.assertEqual("managed", combined["rows"][0]["effective_backend"])

    def test_official_openblas_profile_reports_actual_dispatch_provenance(self):
        payload = [
            {
                "operation": "np.linalg.trace(a) (float64)",
                "suite": "LinearAlgebra",
                "dtype": "float64",
                "n": 1000,
                "numpy_ms": 1.0,
                "numsharp_ms": 0.5,
            },
            {
                "operation": "np.matmul(A, B) (float64)",
                "suite": "LinearAlgebra",
                "dtype": "float64",
                "n": 1000,
                "numpy_ms": 1.0,
                "numsharp_ms": 0.25,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "openblas.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            rows = module.load_json_profile(path, "openblas")

        self.assertEqual("managed", rows[0]["result"]["actual_backend"])
        self.assertEqual("managed_control", rows[0]["backend_route"])
        self.assertEqual("openblas", rows[1]["result"]["actual_backend"])
        self.assertEqual("openblas", rows[1]["backend_route"])

    def test_status_thresholds_match_dashboard(self):
        cases = [
            (1.05, "faster"),
            (1.049, "close"),
            (0.8, "close"),
            (0.799, "slower"),
            (0.33, "slower"),
            (0.329, "much_slower"),
        ]
        for ratio, expected in cases:
            with self.subTest(ratio=ratio):
                _, _, status = module.performance_status(1.0, 1.0 / ratio)
                self.assertEqual(expected, status)


if __name__ == "__main__":
    unittest.main()
