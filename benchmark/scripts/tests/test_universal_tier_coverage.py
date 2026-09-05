#!/usr/bin/env python3
"""Contract tests for the official 1K / 100K / 10M coverage gate."""

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parent.parent / "merge-results.py"
BENCHMARK_ROOT = MODULE_PATH.parent.parent
SPEC = importlib.util.spec_from_file_location("merge_results", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def row(n, name="np.example(a)", dtype="float64"):
    return {"name": name, "dtype": dtype, "n": n}


class UniversalTierCoverageTests(unittest.TestCase):
    def test_completed_zero_duration_is_negligible_not_no_data(self):
        self.assertEqual("negligible", module.classify(0.0002, 0.0, None))
        self.assertEqual("no_data", module.classify(0.0002, None, None))

    def test_identifier_scenarios_do_not_collapse_to_one_join_key(self):
        cases = {
            "np.sum(contiguous_slice)": "np.sum contiguous_slice",
            "np.sum(strided_slice)": "np.sum strided_slice",
            "np.diag": "np.diag view",
            "np.diag(a2d)": "np.diag view",
            "np.diag(a1d)": "np.diag construct",
            "np.transpose(a)": "np.transpose",
            "np.transpose(a, axes)": "np.transpose axes",
        }
        self.assertEqual(cases, {name: module.normalize_op_name(name) for name in cases})

    def test_ambiguous_join_keys_fail_before_indexing(self):
        rows = [
            {"name": "np.copy(a)", "dtype": "float64", "n": 1_000},
            {"name": "np.copy(a) [duplicate]", "dtype": "float64", "n": 1_000},
        ]
        with self.assertRaisesRegex(RuntimeError, "ambiguous normalized benchmark cell"):
            module.validate_join_key_uniqueness(rows, "test")

    def test_complete_universal_tiers_pass(self):
        module.validate_universal_tier_coverage(
            [row(1_000), row(100_000), row(10_000_000)], "test")

    def test_small_only_is_allowed(self):
        module.validate_universal_tier_coverage([row(1_000)], "test")

    def test_small_and_medium_without_large_fails(self):
        with self.assertRaisesRegex(RuntimeError, "1K and 100K but no 10M"):
            module.validate_universal_tier_coverage(
                [row(1_000), row(100_000)], "test")

    def test_dtype_groups_are_checked_independently(self):
        rows = [
            row(1_000, dtype="float32"), row(100_000, dtype="float32"),
            row(10_000_000, dtype="float64"),
        ]
        with self.assertRaisesRegex(RuntimeError, "float32"):
            module.validate_universal_tier_coverage(rows, "test")

    def test_csharp_classes_do_not_stop_at_small_and_medium(self):
        forbidden = "[Params(ArraySizeSource.Small, ArraySizeSource.Medium)]"
        offenders = [
            str(path.relative_to(BENCHMARK_ROOT))
            for path in (BENCHMARK_ROOT / "NumSharp.Benchmark.CSharp" / "Benchmarks").rglob("*.cs")
            if forbidden in path.read_text(encoding="utf-8")
        ]
        self.assertEqual([], offenders)

    def test_numpy_twin_does_not_stop_at_medium(self):
        source = (BENCHMARK_ROOT / "NumSharp.Benchmark.Python" / "numpy_benchmark.py").read_text(
            encoding="utf-8")
        self.assertNotIn('n <= ARRAY_SIZES["medium"]', source)
        self.assertNotIn('n > ARRAY_SIZES["medium"]', source)

    def test_numpy_large_branch_emits_every_previously_capped_family(self):
        bench_path = BENCHMARK_ROOT / "NumSharp.Benchmark.Python" / "numpy_benchmark.py"
        spec = importlib.util.spec_from_file_location("numpy_benchmark_tier_audit", bench_path)
        bench = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        sys.modules[spec.name] = bench
        spec.loader.exec_module(bench)
        self.assertEqual(1_000_000, bench.memory_heavy_work_n(10_000_000))
        self.assertEqual(100_000, bench.memory_heavy_work_n(100_000))

        def fake_benchmark(_func, n, **_kwargs):
            return bench.BenchmarkResult("", "", "", "", n, 0.0, 0.0, 0.0, 0.0, 1, 0.0)

        original_large = bench.ARRAY_SIZES["large"]
        original_cap = bench.MEMORY_HEAVY_LARGE_WORKLOAD
        original_benchmark = bench.benchmark
        try:
            # Exercise the Large-only branches with tiny allocations; timing is deliberately stubbed.
            bench.ARRAY_SIZES["large"] = 1_000
            bench.MEMORY_HEAVY_LARGE_WORKLOAD = 1_000
            bench.benchmark = fake_benchmark
            creation = bench.run_creation_benchmarks(1_000, "float64", 1)
            manipulation = bench.run_manipulation_benchmarks(1_000, 1)
            logic = sum((bench.run_logic_benchmarks(1_000, dtype, 1)
                         for dtype in bench.FLOAT_DTYPES), [])
            statistics = bench.run_statistics_benchmarks(1_000, "float64", 1)
            sorting = sum((bench.run_sorting_benchmarks(1_000, dtype, 1)
                           for dtype in bench.COMMON_DTYPES), [])
            fft = bench.run_fft_benchmarks(1_000, 1)
            random = bench.run_random_benchmarks(1_000, 1)
            ndarray = bench.run_ndarray_benchmarks(1_000, 1)
            selection = bench.run_where_benchmarks(1_000, 1)
        finally:
            bench.ARRAY_SIZES["large"] = original_large
            bench.MEMORY_HEAVY_LARGE_WORKLOAD = original_cap
            bench.benchmark = original_benchmark

        self.assertEqual(14, sum(row.category == "Conversion" for row in creation))
        self.assertEqual(9, sum(row.category == "Stack" for row in manipulation))
        self.assertEqual(20, sum(row.category == "SetOperations" for row in manipulation))  # 10 set/unique ops × int32 + float64
        self.assertEqual(22, sum(row.category == "ExtendedShape" for row in manipulation))
        self.assertEqual(6, sum(row.category == "Close" for row in logic))
        self.assertEqual(12, sum(row.category in {"Signal", "Histogram"} for row in statistics))
        self.assertEqual(40, len(sorting))       # 3 base + 7 formerly capped, across four dtypes
        self.assertEqual(26, len(fft))
        self.assertEqual(52, len(random))
        self.assertEqual(42, len(ndarray))
        self.assertEqual(18, len(selection))     # 2 where + 16 formerly capped indexing cases


if __name__ == "__main__":
    unittest.main()
