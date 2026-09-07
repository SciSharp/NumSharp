#!/usr/bin/env python3
"""Actual BDN checkpoint contracts (opt in: NUMSHARP_BENCHMARK_INTEGRATION=1).

Build NumSharp.Benchmark.CSharp in Release/net10.0 first. These tests execute only
one small workload per depth; no full suite or 10M workload runs.
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[3]
PROJECT = REPO / "benchmark/NumSharp.Benchmark.CSharp/NumSharp.Benchmark.CSharp.csproj"
DLL = PROJECT.parent / "bin/Release/net10.0/NumSharp.Benchmark.CSharp.dll"
FILTER = "*Benchmarks.Arithmetic.AddBenchmarks.NpAdd"
PREFIX = "@@BENCHMARK "


@unittest.skipUnless(os.environ.get("NUMSHARP_BENCHMARK_INTEGRATION") == "1",
                     "set NUMSHARP_BENCHMARK_INTEGRATION=1 for actual BDN smoke tests")
class CSharpCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(DLL.exists(), f"Build Release/net10.0 first: {PROJECT}")
        self.temporary = tempfile.TemporaryDirectory(prefix="numsharp-checkpoint-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.env = os.environ.copy()
        for key in ("NUMSHARP_BENCHMARK_CASES_FILE", "NUMSHARP_BENCHMARK_CHECKPOINT_DIR"):
            self.env.pop(key, None)
        self.env.update(NUMSHARP_BENCHMARK_DEPTH="pass", NUMSHARP_BENCHMARK_DTYPES="float64")

    def run_bdn(self, *args):
        result = subprocess.run(["dotnet", str(DLL), *map(str, args)], env=self.env,
                                cwd=REPO, text=True, capture_output=True, timeout=45)
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        return result.stdout

    def plan(self):
        path = self.root / "plan.json"
        self.run_bdn("--benchmark-plan-json", path, "--filter", FILTER)
        return json.loads(path.read_text(encoding="utf-8"))

    def select(self, cases):
        path = self.root / "allowlist.json"
        path.write_text(json.dumps(cases), encoding="utf-8")
        self.env["NUMSHARP_BENCHMARK_CASES_FILE"] = str(path)

    def test_discovery_honors_dtype_glob_exact_and_empty_selection(self):
        plan = self.plan()
        self.assertEqual({1000, 100_000, 10_000_000}, {case["n"] for case in plan})
        self.assertEqual({"float64"}, {case["dtype"] for case in plan})
        self.assertEqual({"arithmetic"}, {case["suite"] for case in plan})
        self.assertEqual({"np.add(a, b)"}, {case["operation"] for case in plan})
        selected = next(case for case in plan if case["n"] == 1000)
        self.select([selected["id"]])
        self.assertEqual([selected], self.plan())
        self.select([])
        self.assertEqual([], self.plan())
        # An empty allowlist is intentionally zero cases, never an implicit full run.
        self.assertNotIn(PREFIX, self.run_bdn("--filter", FILTER, "--artifacts", self.root / "empty"))

    def test_pass_and_light_checkpoint_full_measurements_and_atomic_replacement(self):
        selected = next(case for case in self.plan() if case["n"] == 1000)
        self.select([selected["id"]])
        checkpoint_dir = self.root / "checkpoints"
        self.env["NUMSHARP_BENCHMARK_CHECKPOINT_DIR"] = str(checkpoint_dir)
        filename = hashlib.sha256(selected["id"].encode()).hexdigest() + "-report-full-compressed.json"
        for depth, actual, warmup in (("pass", 1, 0), ("light", 8, 3)):
            with self.subTest(depth=depth):
                self.env["NUMSHARP_BENCHMARK_DEPTH"] = depth
                output = self.run_bdn("--filter", FILTER, "--artifacts", self.root / depth)
                events = [json.loads(line[len(PREFIX):]) for line in output.splitlines()
                          if line.startswith(PREFIX)]
                self.assertEqual(["start", "complete"], [event["event"] for event in events])
                self.assertEqual("ok", events[-1]["status"])
                self.assertEqual(selected["id"], events[-1]["id"])
                self.assertEqual(checkpoint_dir / filename, Path(events[-1]["checkpoint"]))
                self.assertEqual([filename], sorted(path.name for path in checkpoint_dir.iterdir()))
                rows = json.loads((checkpoint_dir / filename).read_text(encoding="utf-8"))["Benchmarks"]
                self.assertEqual(1, len(rows))
                row = rows[0]
                self.assertEqual(selected["id"], row["CaseId"])
                self.assertEqual("ok", row["CheckpointStatus"])
                self.assertIsNotNone(row["Statistics"])
                counts = {}
                for measurement in row["Measurements"]:
                    if measurement["IterationMode"] == "Workload":
                        stage = measurement["IterationStage"]
                        counts[stage] = counts.get(stage, 0) + 1
                self.assertEqual(actual, counts.get("Actual", 0))
                self.assertEqual(warmup, counts.get("Warmup", 0))

    def test_failed_workload_is_durable_and_reported_failed(self):
        # A small external probe exercises BDN's actual exception path without adding a
        # deliberately failing benchmark to the official discovery surface.
        checkpoint_dir = self.root / "failure"
        self.env["NUMSHARP_BENCHMARK_CHECKPOINT_DIR"] = str(checkpoint_dir)
        source = f'''#:project {PROJECT.as_posix()}
#:property PublishAot=false
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using NumSharp.Benchmark.CSharp.Infrastructure;
BenchmarkRunner.Run<CheckpointFailureProbe>(new OfficialBenchmarkConfig());
public class CheckpointFailureProbe
{{
    [Benchmark] public int Throws() => throw new InvalidOperationException("checkpoint failure probe");
}}
'''
        result = subprocess.run(["dotnet", "run", "-c", "Release", "-"], input=source,
                                cwd=self.root, env=self.env, text=True, capture_output=True, timeout=60)
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        events = [json.loads(line[len(PREFIX):]) for line in result.stdout.splitlines()
                  if line.startswith(PREFIX)]
        self.assertEqual(["start", "complete"], [event["event"] for event in events], result.stdout)
        self.assertEqual("failed", events[-1]["status"])
        row = json.loads(Path(events[-1]["checkpoint"]).read_text(encoding="utf-8"))["Benchmarks"][0]
        self.assertEqual(events[-1]["id"], row["CaseId"])
        self.assertEqual("failed", row["CheckpointStatus"])
        self.assertIsNone(row["Statistics"])


if __name__ == "__main__":
    unittest.main()
