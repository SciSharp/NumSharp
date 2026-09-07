"""Per-cell recovery contracts; all measured workloads use pass depth and small arrays."""

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "numpy_checkpoint_tests_benchmark",
    REPO / "benchmark" / "NumSharp.Benchmark.Python" / "numpy_benchmark.py")
bench = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bench
spec.loader.exec_module(bench)
from numpy_checkpoint import validate_checkpoint


class NumpyCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.previous_depth = bench.ACTIVE_BENCHMARK_DEPTH
        self.previous_dtypes = bench.ACTIVE_DTYPES
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        bench.ACTIVE_BENCHMARK_DEPTH = self.previous_depth
        bench.ACTIVE_DTYPES = self.previous_dtypes
        bench.SKIPPED_OP_NAMES.clear()
        self.directory.cleanup()

    def run_cli(self, *arguments):
        output, errors = io.StringIO(), io.StringIO()
        with mock.patch.object(sys, "argv", ["numpy_benchmark.py", *arguments]), \
                contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            self.exit_code = bench.main()
        return output.getvalue(), errors.getvalue()

    def plan(self, suite="comparison", dtype="int32", sizes=(1000,)):
        return [template["case"] for template in bench.discover_case_templates(suite, [dtype], sizes)]

    def test_discovery_uses_bounded_fixtures_and_never_executes_workloads(self):
        original = bench.create_random_array
        sizes = []

        def bounded(n, *args, **kwargs):
            sizes.append(n)
            self.assertLessEqual(n, 1000)
            return original(n, *args, **kwargs)

        with mock.patch.object(bench, "create_random_array", side_effect=bounded), \
                mock.patch.object(bench, "_benchmark_impl", side_effect=AssertionError("timed body ran")):
            plan = self.plan(sizes=(1000, 100000, 10000000))
        self.assertTrue(sizes)
        self.assertEqual({1000, 100000, 10000000}, {case["n"] for case in plan})
        self.assertEqual({"int32"}, {case["dtype"] for case in plan})
        self.assertEqual(len(plan), len({case["id"] for case in plan}))

        called = False

        def must_not_run():
            nonlocal called
            called = True
            raise TypeError("expected")

        with mock.patch.object(bench, "DISCOVERY_CALLS", []):
            bench._rejection_evidence(must_not_run, 1000, "unsupported", "Bitwise", "float64")
        self.assertFalse(called)

    def test_plan_obeys_api_tiers_and_scalar_subset(self):
        plan = self.plan("api", sizes=(1, 1000, 100000, 10000000))
        self.assertEqual({1, 1000}, {case["n"] for case in plan})
        scalar = {case["operation"] for case in plan if case["n"] == 1}
        small = {case["operation"] for case in plan if case["n"] == 1000}
        self.assertLess(scalar, small)
        self.assertIn("np.save(path, a)", small)
        self.assertNotIn("np.save(path, a)", scalar)

    def test_exact_selection_checkpoints_before_completion_event_and_resumes(self):
        selected = self.plan()[3]
        cases_file = self.root / "cases.json"
        cases_file.write_text(json.dumps([selected["id"]]), encoding="utf-8")
        arguments = ("--suite", "comparison", "--dtypes", "int32", "--size", "small",
                     "--depth", "pass", "--cases-file", str(cases_file),
                     "--checkpoint-dir", str(self.root / "checkpoints"),
                     "--output", str(self.root / "results.json"))
        original_emit = bench.CaseExecution.emit

        def emit(execution, event, case, **extra):
            if event == "complete":
                checkpoint = json.loads(Path(extra["checkpoint"]).read_text(encoding="utf-8"))
                self.assertEqual("ok", checkpoint["status"])
                self.assertEqual(case["id"], checkpoint["id"])
            original_emit(execution, event, case, **extra)

        with mock.patch.object(bench, "_benchmark_impl", wraps=bench._benchmark_impl) as timings, \
                mock.patch.object(bench.CaseExecution, "emit", emit):
            stdout, stderr = self.run_cli(*arguments)
        self.assertEqual(1, timings.call_count)
        self.assertFalse(stderr)
        events = [json.loads(line[len("@@BENCHMARK "):]) for line in stdout.splitlines()
                  if line.startswith("@@BENCHMARK ")]
        self.assertEqual(["start", "complete"], [event["event"] for event in events])
        results = json.loads((self.root / "results.json").read_text())
        self.assertEqual(1, len(results))
        self.assertEqual(selected["operation"], bench._operation(bench.BenchmarkResult(**results[0])))

        with mock.patch.object(bench, "_benchmark_impl") as timings:
            self.run_cli(*arguments)
        timings.assert_not_called()
        self.assertEqual(results, json.loads((self.root / "results.json").read_text()))

    def test_failed_cell_is_explicit_and_retry_does_not_repeat_successes(self):
        arguments = ("--suite", "comparison", "--dtypes", "int32", "--size", "small",
                     "--depth", "pass", "--checkpoint-dir", str(self.root / "checkpoints"),
                     "--output", str(self.root / "results.json"))
        original = bench._benchmark_impl
        count = 0

        def fail_one(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 3:
                raise RuntimeError("simulated interrupted operation")
            return original(*args, **kwargs)

        with mock.patch.object(bench, "_benchmark_impl", side_effect=fail_one):
            self.run_cli(*arguments)
        self.assertEqual(1, self.exit_code)
        checkpoints = [json.loads(path.read_text()) for path in (self.root / "checkpoints").glob("*.json")]
        failed = [row for row in checkpoints if row["status"] == "failed"]
        self.assertEqual(1, len(failed))
        self.assertIsNone(failed[0]["result"])
        self.assertIn("simulated interrupted operation", failed[0]["error"])
        self.assertEqual(count - 1, len(json.loads((self.root / "results.json").read_text())))
        with mock.patch.object(bench, "_benchmark_impl", wraps=original) as timings:
            self.run_cli(*arguments)
        self.assertEqual(0, self.exit_code)
        self.assertEqual(1, timings.call_count)
        self.assertEqual(count, len(json.loads((self.root / "results.json").read_text())))

    def test_expected_rejection_is_successful_zero_duration_checkpoint(self):
        self.run_cli("--suite", "bitwise", "--dtypes", "float64", "--size", "small",
                     "--depth", "pass", "--checkpoint-dir", str(self.root / "checkpoints"))
        checkpoints = [json.loads(path.read_text()) for path in (self.root / "checkpoints").glob("*.json")]
        self.assertEqual(10, len(checkpoints))
        self.assertEqual({"ok"}, {row["status"] for row in checkpoints})
        self.assertEqual({0.0}, {row["result"]["min_ms"] for row in checkpoints})

    def test_standalone_failure_exit_preserves_other_tiers_of_the_same_operation(self):
        selected = [case["id"] for case in self.plan(sizes=(1000, 100000, 10000000))
                    if case["operation"] == "a == b"]
        cases_file = self.root / "cases.json"
        cases_file.write_text(json.dumps(selected), encoding="utf-8")
        original = bench.create_random_array

        def result_or_failure(_func, n, *_args, **_kwargs):
            if n == 100000:
                raise RuntimeError("failed middle tier")
            return bench.BenchmarkResult("", "", "", "", n, 1.0, 0.0, 1.0, 1.0, 1, 1000.0)

        with mock.patch.object(bench, "_benchmark_impl", side_effect=result_or_failure), \
                mock.patch.object(bench, "create_random_array", side_effect=lambda n, *a, **kw: original(min(n, 1000), *a, **kw)):
            self.run_cli("--suite", "comparison", "--dtypes", "int32", "--cache-sizes",
                         "--depth", "pass", "--cases-file", str(cases_file),
                         "--output", str(self.root / "results.json"))
        self.assertEqual(1, self.exit_code)
        results = json.loads((self.root / "results.json").read_text())
        self.assertEqual([1000, 10000000], [row["n"] for row in results])

    def test_interrupted_suite_resumes_without_a_final_results_file(self):
        arguments = ("--suite", "comparison", "--dtypes", "int32", "--size", "small",
                     "--depth", "pass", "--checkpoint-dir", str(self.root / "checkpoints"),
                     "--output", str(self.root / "results.json"))
        original = bench._benchmark_impl
        count = 0

        def interrupt(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 3:
                raise KeyboardInterrupt()
            return original(*args, **kwargs)

        with mock.patch.object(bench, "_benchmark_impl", side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_cli(*arguments)
        self.assertIsNone(bench.ACTIVE_CASE_EXECUTION)
        self.assertFalse((self.root / "results.json").exists())
        self.assertEqual(2, len(list((self.root / "checkpoints").glob("*.json"))))
        with mock.patch.object(bench, "_benchmark_impl", wraps=original) as timings:
            self.run_cli(*arguments)
        self.assertEqual(len(self.plan()) - 2, timings.call_count)
        self.assertEqual(len(self.plan()), len(json.loads((self.root / "results.json").read_text())))

    def test_empty_selection_does_not_allocate_real_tier(self):
        cases_file = self.root / "cases.json"
        cases_file.write_text("[]", encoding="utf-8")
        with mock.patch.object(bench, "create_random_array", wraps=bench.create_random_array) as allocate:
            self.run_cli("--suite", "comparison", "--dtypes", "int32", "--size", "large",
                         "--depth", "pass", "--cases-file", str(cases_file))
        self.assertTrue(allocate.called)
        self.assertTrue(all(call.args[0] <= 1000 for call in allocate.call_args_list))

    def test_checkpoint_settings_cannot_silently_change_measurement_depth(self):
        case = self.plan()[0]
        template = bench.discover_case_templates("comparison", ["int32"], [1000])[0]
        store = bench.CheckpointStore(self.root, {"depth": "pass"})
        store.save(case, "ok", template["result"])
        changed = bench.CheckpointStore(self.root, {"depth": "measure"})
        with self.assertRaisesRegex(ValueError, "settings differ"):
            changed.completed(case)

    def test_corrupt_checkpoint_is_pending_and_cannot_attribute_wrong_operation(self):
        template = bench.discover_case_templates("comparison", ["int32"], [1000])[0]
        case = template["case"]
        store = bench.CheckpointStore(self.root, {"depth": "pass"})
        path = store.path(case["id"])
        path.write_text('{"unfinished":', encoding="utf-8")
        self.assertIsNone(store.completed(case))
        result = dict(template["result"], name="a different operation")
        store.save(case, "ok", result)
        self.assertIsNone(store.completed(case))

    def test_shared_validator_rejects_partial_and_mismatched_payloads(self):
        template = bench.discover_case_templates("comparison", ["int32"], [1000])[0]
        case = template["case"]
        settings = {"depth": "pass"}
        store = bench.CheckpointStore(self.root, settings)
        path = Path(store.save(case, "ok", template["result"]))
        payload = json.loads(path.read_text())
        self.assertTrue(validate_checkpoint(payload, case, settings))
        self.assertFalse(validate_checkpoint(payload, settings={"depth": "measure"}))
        self.assertFalse(validate_checkpoint(dict(payload, id="wrong id")))
        self.assertFalse(validate_checkpoint(dict(payload, result={"n": 1000})))
        self.assertFalse(validate_checkpoint(dict(payload, result=dict(payload["result"], mean_ms=float("nan")))))
        self.assertFalse(validate_checkpoint(dict(payload, result=dict(payload["result"], iterations=0))))
        failed = dict(payload, status="failed", result=None, error="operation failed")
        self.assertTrue(validate_checkpoint(failed, case, settings))
        self.assertFalse(validate_checkpoint(dict(failed, error=None)))


if __name__ == "__main__":
    unittest.main()
