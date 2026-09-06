#!/usr/bin/env python3
"""Contracts for pass/light/measure selection and dtype targeting."""

import importlib.util
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
SCRIPT_DIR = REPO / "benchmark" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import benchmark_modes


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load("benchmark_runner_mode_tests", REPO / "benchmark" / "run_benchmark.py")
numpy_bench = load(
    "numpy_benchmark_mode_tests",
    REPO / "benchmark" / "NumSharp.Benchmark.Python" / "numpy_benchmark.py")


class BenchmarkModeTests(unittest.TestCase):
    def test_depth_contract_values(self):
        self.assertEqual((1, 0), (
            benchmark_modes.DEPTHS["pass"].bdn_measurements,
            benchmark_modes.DEPTHS["pass"].bdn_warmups))
        self.assertEqual((8, 3), (
            benchmark_modes.DEPTHS["light"].bdn_measurements,
            benchmark_modes.DEPTHS["light"].bdn_warmups))
        self.assertEqual((50, 5), (
            benchmark_modes.DEPTHS["measure"].bdn_measurements,
            benchmark_modes.DEPTHS["measure"].bdn_warmups))

    def test_dtype_aliases_are_canonical_ordered_and_deduplicated(self):
        self.assertEqual(
            ("uint8", "float16", "float32", "complex128"),
            benchmark_modes.parse_dtypes("c128, f32, byte, half, f32"))
        self.assertEqual(benchmark_modes.ALL_DTYPES, benchmark_modes.parse_dtypes("all"))
        with self.assertRaisesRegex(ValueError, "Unknown dtype"):
            benchmark_modes.parse_dtypes("float32,nope")

    def test_multi_dtype_selection_matches_any_side(self):
        self.assertTrue(benchmark_modes.matches_any(("float32",), "int16", "float32"))
        self.assertTrue(benchmark_modes.matches_any(("float32",), "f32", "int16"))
        self.assertFalse(benchmark_modes.matches_any(("float32",), "int16", "float64"))

    def test_interactive_picker_returns_depth_and_comma_dtypes(self):
        answers = iter(["2", "f32,c128"])
        with contextlib.redirect_stdout(io.StringIO()):
            argv = runner.prompt_run_options(lambda _prompt: next(answers))
        self.assertEqual(["--depth", "light", "--dtypes", "f32,c128"], argv)

    def test_numpy_pass_executes_workload_exactly_once(self):
        previous = numpy_bench.ACTIVE_BENCHMARK_DEPTH
        calls = 0
        def workload():
            nonlocal calls
            calls += 1
        try:
            numpy_bench.ACTIVE_BENCHMARK_DEPTH = "pass"
            result = numpy_bench.benchmark(workload, 1_000)
        finally:
            numpy_bench.ACTIVE_BENCHMARK_DEPTH = previous
        self.assertEqual(1, calls)
        self.assertEqual(1, result.iterations)
        self.assertEqual(result.mean_ms, result.min_ms)
        self.assertEqual(result.mean_ms, result.max_ms)
        self.assertEqual(0.0, result.stddev_ms)

    def test_expected_rejection_is_executed_once_and_recorded_as_zero_duration_evidence(self):
        calls = 0
        def rejected():
            nonlocal calls
            calls += 1
            raise TypeError("expected dtype rejection")
        previous_dtypes = numpy_bench.ACTIVE_DTYPES
        try:
            numpy_bench.ACTIVE_DTYPES = {"float64"}
            result = numpy_bench._rejection_evidence(
                rejected, 1_000, "np.bitwise_not(a)", "Bitwise", "float64")
        finally:
            numpy_bench.ACTIVE_DTYPES = previous_dtypes
        self.assertEqual(1, calls)
        self.assertIsNotNone(result)
        self.assertEqual(1, result.iterations)
        self.assertEqual(0.0, result.min_ms)
        self.assertEqual("float64", result.dtype)

    def test_numpy_suite_local_loops_honor_dtype_filter(self):
        previous_benchmark = numpy_bench.benchmark
        previous_dtypes = numpy_bench.ACTIVE_DTYPES
        def fake(_func, n, **_kwargs):
            return numpy_bench.BenchmarkResult("", "", "", "", n, 0, 0, 0, 0, 1, 0)
        try:
            numpy_bench.benchmark = fake
            numpy_bench.ACTIVE_DTYPES = {"float16"}
            with contextlib.redirect_stdout(io.StringIO()):
                rows = numpy_bench.run_suites(1_000, "reduction", ["float16"], 1)
        finally:
            numpy_bench.benchmark = previous_benchmark
            numpy_bench.ACTIVE_DTYPES = previous_dtypes
        self.assertTrue(rows)
        self.assertEqual({"float16"}, {row.dtype for row in rows})
        self.assertTrue(any("nanmean" in row.name for row in rows))

    def test_light_validation_counts_executed_iterations_not_retained_outliers(self):
        measurements = [
            {"IterationMode": "Workload", "IterationStage": "Warmup"}
            for _ in range(3)
        ] + [
            {"IterationMode": "Workload", "IterationStage": "Actual"}
            for _ in range(8)
        ]
        row = {
            "FullName": "Example.Benchmark()",
            "Statistics": {"OriginalValues": [1.0] * 6},
            "Measurements": measurements,
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "csharp").mkdir()
            (root / "csharp-openblas").mkdir()
            (root / "csharp" / "report.json").write_text(
                json.dumps({"Benchmarks": [row]}), encoding="utf-8")
            (root / "numpy-results.json").write_text("[]", encoding="utf-8")
            (root / "benchmark-report.json").write_text(
                json.dumps({"rows": []}), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                runner.validate_execution_depth(root, "light")


if __name__ == "__main__":
    unittest.main()
