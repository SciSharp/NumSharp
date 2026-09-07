"""Behavioral tests for safe, exact reruns from saved benchmark reports."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "benchmark_selection", Path(__file__).resolve().parents[1] / "benchmark_selection.py")
module = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(module)


def cell(engine, operation="np.sum(a)", *, n=1000, suite="reduction", scenario=""):
    return {"id": f"{engine}|{suite}|{operation}|{n}|{scenario}", "engine": engine,
            "operation": operation, "suite": suite, "dtype": "float64", "n": n,
            "scenario": scenario}


def row(status="slower", *, operation="np.sum (float64)", ms=2.0,
        availability="available", scenario="", n=1000, suite="Reduction"):
    return {"operation": operation, "suite": suite, "dtype": "float64", "n": n,
            "scenario": scenario, "numpy_ms": 1.0, "numsharp_ms": ms,
            "availability": availability, "status": status}


class BenchmarkSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.plan = [cell(engine) for engine in ("numpy", "managed", "openblas")]

    def save(self, payload, name="benchmark-report.json"):
        path = self.directory / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def select(self, payload, mode="bad", plan=None):
        return module.select_plan(plan or self.plan, self.save(payload), mode)

    def test_bad_legacy_report_selects_managed_and_numpy_only(self):
        source = self.save([row()])
        before = source.read_bytes()
        selected, diagnostic = module.select_plan(self.plan, self.directory, "bad")
        self.assertEqual(["numpy", "managed"], [r["engine"] for r in selected])
        self.assertEqual(before, source.read_bytes())
        self.assertEqual("slower", diagnostic["reasons"][self.plan[1]["id"]][0]["reason"])

    def test_combined_profile_failure_is_not_hidden_by_effective_win(self):
        combined = row("faster", ms=0.5)
        combined["profiles"] = {
            "managed": {"status": "failed", "availability": "failed", "numsharp_ms": None},
            "openblas": {"status": "faster", "availability": "available", "numsharp_ms": 0.5},
        }
        selected, _ = self.select({"rows": [combined]}, "failed")
        self.assertEqual(["numpy", "managed"], [r["engine"] for r in selected])

    def test_profile_report_preserves_openblas_profile(self):
        source_row = row()
        source_row["result"] = {"profile": "openblas", "status": "slower", "numsharp_ms": 2}
        selected, _ = self.select({"profile": "openblas", "rows": [source_row]})
        self.assertEqual(["numpy", "openblas"], [r["engine"] for r in selected])

    def test_same_csharp_case_id_does_not_select_other_backend(self):
        plan = [{**entry, "id": "same-csharp-id"} if entry["engine"] != "numpy" else entry
                for entry in self.plan]
        selected, _ = self.select([row()], plan=plan)
        self.assertEqual(["numpy", "managed"], [r["engine"] for r in selected])

    def test_backend_unavailability_never_schedules_expected_failure(self):
        for availability in ("missing_backend", "not_supported"):
            with self.subTest(availability=availability):
                selected, _ = self.select([row("no_data", availability=availability, ms=None)])
                self.assertEqual([], selected)

    def test_failed_and_bad_have_distinct_meanings(self):
        for status, ms, bad, failed in (("no_data", None, True, False),
                                      ("slower", 2, True, False),
                                      ("much_slower", 20, True, False),
                                      ("failed", None, True, True),
                                      ("negligible", 0, False, False),
                                      ("close", 1, False, False)):
            with self.subTest(status=status):
                selected, _ = self.select([row(status, ms=ms)], "bad")
                self.assertEqual(bad, bool(selected))
                selected, _ = self.select([row(status, ms=ms)], "failed")
                self.assertEqual(failed, bool(selected))

    def test_scenarios_and_sizes_must_match_exactly(self):
        for changes in ({"scenario": "axis=1"}, {"n": 32}, {"suite": "Layout"}):
            with self.subTest(changes=changes):
                selected, diagnostic = self.select([row(**changes)])
                self.assertEqual([], selected)
                self.assertEqual(1, len(diagnostic["unmatched"]))

    def test_slice_and_diagonal_scenarios_are_not_collapsed(self):
        plan = [cell("managed", "np.diag(a2d)"), cell("managed", "np.sum(strided_slice)")]
        selected, diagnostic = self.select([
            row(operation="np.diag(a1d)"), row(operation="np.sum(contiguous_slice)")], plan=plan)
        self.assertEqual([], selected)
        self.assertEqual(2, len(diagnostic["unmatched"]))

    def test_suite_alias_and_negative_axis_match(self):
        plan = [cell("managed", "np.sum(a, axis=-1)", suite="linalg")]
        selected, _ = self.select([row(operation="np.sum axis=-1", suite="LinearAlgebra")], plan=plan)
        self.assertEqual(plan, selected)

    def test_ambiguous_matching_plan_is_reported_without_broadening(self):
        duplicate = {**self.plan[1], "id": "different-id"}
        selected, diagnostic = self.select([row()], plan=[self.plan[1], duplicate])
        self.assertEqual([], selected)
        self.assertIn("ambiguous", diagnostic["unmatched"][0]["reason"])

    def test_regression_compares_same_profile_absolute_minimum(self):
        baseline_row, current_row = row(), row()
        baseline_row["profiles"] = {
            "managed": {"status": "slower", "numsharp_ms": 2},
            "openblas": {"status": "faster", "numsharp_ms": 0.5}}
        current_row["profiles"] = {
            "managed": {"status": "slower", "numsharp_ms": 3},
            "openblas": {"status": "faster", "numsharp_ms": 0.4}}
        baseline = self.save({"rows": [baseline_row]}, "baseline.json")
        current = self.save({"rows": [current_row]})
        selected, diagnostic = module.select_plan(self.plan, current, "degraded", baseline)
        self.assertEqual(["numpy", "managed"], [r["engine"] for r in selected])
        reason = diagnostic["reasons"][self.plan[1]["id"]][0]
        self.assertEqual(50, reason["increase_percent"])
        self.assertEqual(2, reason["baseline_ms"])
        self.assertTrue(diagnostic["warnings"])

    def test_regression_threshold_is_strict_and_not_a_ratio_change(self):
        baseline = self.save([row(ms=2)], "baseline.json")
        current_row = row(ms=2.2)
        current_row["numpy_ms"] = 0.01
        selected, _ = module.select_plan(self.plan, self.save([current_row]), "degraded", baseline)
        self.assertEqual([], selected)

    def test_regression_excludes_noncredible_baseline_and_changed_actual_backend(self):
        current_row = row(ms=4)
        current_row["actual_backend"] = "openblas"
        for old in (row("negligible", ms=0), {**row(), "actual_backend": "managed"}):
            with self.subTest(old=old):
                baseline = self.save([old], "baseline.json")
                selected, diagnostic = module.select_plan(
                    self.plan, self.save([current_row]), "degraded", baseline)
                self.assertEqual([], selected)
                self.assertEqual(1, len(diagnostic["noncomparable"]))

    def test_known_incompatible_provenance_rejected_but_commits_can_differ(self):
        baseline = self.save({"metadata": {"benchmark_depth": "measure", "commit": "old"},
                              "rows": [row()]}, "baseline.json")
        current = self.save({"metadata": {"benchmark_depth": "light", "commit": "new"},
                             "rows": [row(ms=3)]})
        with self.assertRaisesRegex(ValueError, "Incompatible.*depth"):
            module.select_plan(self.plan, current, "degraded", baseline)
        current = self.save({"metadata": {"benchmark_depth": "measure", "commit": "new"},
                             "rows": [row(ms=3)]})
        selected, _ = module.select_plan(self.plan, current, "degraded", baseline)
        self.assertEqual(2, len(selected))

    def test_duplicate_source_or_baseline_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate source"):
            self.select([row(), row()])
        baseline = self.save([row(), row()], "baseline.json")
        with self.assertRaisesRegex(ValueError, "duplicate baseline"):
            module.select_plan(self.plan, self.save([row(ms=3)]), "degraded", baseline)

    def test_invalid_source_and_regression_arguments_have_actionable_errors(self):
        with self.assertRaisesRegex(ValueError, "baseline"):
            self.select([row()], "degraded")
        with self.assertRaisesRegex(ValueError, "rows array"):
            self.select({"Benchmarks": []})
        for percent in (-1, float("nan"), float("inf")):
            with self.subTest(percent=percent), self.assertRaisesRegex(ValueError, "nonnegative"):
                module.select_plan(self.plan, self.save([row()]), "bad", regression_percent=percent)

    def test_interrupted_run_selects_actual_failures_and_bad_adds_pending(self):
        self.save(self.plan, "plan.json")
        folder = self.directory / "csharp"
        folder.mkdir()
        self.save({"Benchmarks": [{"CaseId": self.plan[1]["id"], "Statistics": None}]},
                  "csharp/failed.json")
        selected, _ = module.select_plan(self.plan, self.directory, "failed")
        self.assertEqual(["numpy", "managed"], [r["engine"] for r in selected])
        selected, _ = module.select_plan(self.plan, self.directory, "bad")
        self.assertEqual(self.plan, selected)

    def test_numpy_failure_in_checkpoints_is_preserved_when_report_omits_it(self):
        self.save(self.plan, "plan.json")
        self.save({"rows": []})
        (self.directory / "numpy-checkpoints").mkdir()
        self.save({"id": self.plan[0]["id"], "status": "failed"}, "numpy-checkpoints/failed.json")
        selected, diagnostic = module.select_plan(self.plan, self.directory, "failed")
        self.assertEqual([self.plan[0]], selected)
        self.assertEqual("failed", diagnostic["reasons"][self.plan[0]["id"]][0]["reason"])

    def test_new_run_host_metadata_is_checked_for_regressions(self):
        source_dir, baseline_dir = self.directory / "current", self.directory / "baseline"
        source_dir.mkdir()
        baseline_dir.mkdir()
        for folder, cpu in ((source_dir, "CPU A"), (baseline_dir, "CPU B")):
            (folder / "benchmark-report.json").write_text(json.dumps({"rows": [row()]}))
            (folder / "run-config.json").write_text(json.dumps({"depth": "measure",
                "provenance": {"processor": cpu, "numpy": "2.4.2"}}))
        with self.assertRaisesRegex(ValueError, "CPU"):
            module.select_plan(self.plan, source_dir, "degraded", baseline_dir)


if __name__ == "__main__":
    unittest.main()
