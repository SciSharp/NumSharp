"""Interactive recovery discovery without launching benchmark workloads."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

BENCH = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("benchmark_startup_tests_runner", BENCH / "run_benchmark.py")
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


class BenchmarkStartupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bench = self.root / "benchmark"
        self.bench.mkdir()
        self.addCleanup(patch.stopall)
        patch.object(runner, "HERE", self.bench).start()
        self.provenance = {"source_sha256": "unchanged", "numpy": "2.4.2"}
        self.probe = patch.object(runner, "provenance", return_value=self.provenance).start()

    def save_run(self, name="previous", status="interrupted", modified=1_700_000_000, **updates):
        directory = self.bench / "results" / name
        directory.mkdir(parents=True)
        options = runner.parser().parse_args(["--depth", "measure", "--dtypes", "float32,float64", "--suites", "arithmetic"])
        config = {"schema_version": 1, "provenance": self.provenance,
                  "options": {key: getattr(options, key) for key in runner.CONFIG_FIELDS}, "depth": "measure",
                  "dtypes": ["float32", "float64"], "suites": ["arithmetic"]}
        config.update(updates)
        path = directory / "run-config.json"
        path.write_text(json.dumps(config))
        os.utime(path, (modified, modified))
        if status is not None:
            (directory / "run-state.json").write_text(json.dumps({"status": status, "completed": 8,
                "total": 36, "stage": "numpy:arithmetic"}))
        return directory

    def prompt(self, *answers):
        responses = iter(answers)
        prompts = []
        def reply(prompt):
            prompts.append(prompt)
            return next(responses)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            result = runner.prompt_startup_options(reply)
        return result, prompts, output.getvalue()

    def test_default_accepts_resume_and_displays_progress(self):
        directory = self.save_run()
        args, prompts, output = self.prompt("")
        self.assertEqual(["--resume", str(directory)], args)
        self.assertEqual(["Resume this benchmark? [Y/n]: "], prompts)
        self.assertIn("8/36 successful cases", output)
        self.assertIn("numpy:arithmetic", output)
        self.assertIn("measure", output)

    def test_decline_opens_new_run_picker_without_changing_saved_results(self):
        directory = self.save_run()
        before = (directory / "run-state.json").read_bytes()
        args, prompts, _ = self.prompt("n", "1", "f32")
        self.assertEqual(["--depth", "pass", "--dtypes", "f32"], args)
        self.assertEqual(3, len(prompts))
        self.assertEqual(before, (directory / "run-state.json").read_bytes())

    def test_invalid_answer_reprompts(self):
        directory = self.save_run()
        args, prompts, output = self.prompt("maybe", "YES")
        self.assertEqual(["--resume", str(directory)], args)
        self.assertEqual(2, len(prompts))
        self.assertIn("Enter y", output)

    def test_latest_unfinished_excludes_newer_complete_and_planned_runs(self):
        expected = self.save_run("unfinished")
        self.save_run("complete", "complete", modified=1_700_000_010)
        self.save_run("planned", "planned", modified=1_700_000_020)
        self.assertEqual(["--resume", str(expected)], self.prompt("yes")[0])

    def test_finished_matrix_does_not_hide_unfinished_report_stage(self):
        directory = self.save_run()
        (directory / "run-state.json").write_text(json.dumps({"status": "interrupted", "completed": 36,
            "total": 36, "stage": "merge reports"}))
        self.assertEqual(["--resume", str(directory)], self.prompt("yes")[0])

    def test_missing_or_torn_state_still_offers_recovery(self):
        directory = self.save_run(status=None)
        args, _, output = self.prompt("")
        self.assertEqual(["--resume", str(directory)], args)
        self.assertIn("before progress was saved", output)
        (directory / "run-state.json").write_text('{"status":')
        self.assertEqual(args, self.prompt("")[0])

    def test_live_os_lock_excludes_active_run(self):
        directory = self.save_run(status="running")
        with runner.RunLock(directory):
            args, prompts, _ = self.prompt("2", "float64")
        self.assertEqual(["--depth", "light", "--dtypes", "float64"], args)
        self.assertFalse(any("Resume" in prompt for prompt in prompts))
        self.probe.assert_not_called()
        self.assertEqual(["--resume", str(directory)], self.prompt("")[0])

    def test_incompatible_source_is_explained_and_not_offered(self):
        self.save_run(provenance={"source_sha256": "changed", "numpy": "2.4.2"})
        args, prompts, output = self.prompt("1", "")
        self.assertEqual(["--depth", "pass"], args)
        self.assertFalse(any("Resume" in prompt for prompt in prompts))
        self.assertIn("source_sha256", output)
        self.assertIn("cannot resume", output)

    def test_legacy_snapshot_and_malformed_records_are_ignored(self):
        self.save_run("legacy", schema_version=0)
        self.save_run("snapshot", resume_supported=False)
        bad = self.save_run("malformed")
        (bad / "run-config.json").write_text("[]")
        (self.bench / "results/last-run.json").write_text("{")
        self.assertEqual(["--depth", "measure"], self.prompt("", "")[0])
        self.probe.assert_not_called()

    def test_last_run_pointer_discovers_custom_directory(self):
        directory = self.save_run()
        external = self.root / "custom-run"
        directory.rename(external)
        (self.bench / "results/last-run.json").write_text(json.dumps({"directory": str(external)}))
        self.assertEqual(["--resume", str(external)], self.prompt("")[0])
        self.assertEqual(external, runner.resolve_run("latest"))

    def test_invalid_configuration_field_types_are_skipped(self):
        for key, value in (("depth", []), ("depth", {}), ("dtypes", [1]), ("options", {"depth": "measure"}),
                           ("suites", [None]), ("provenance", "invalid")):
            with self.subTest(key=key, value=value):
                directory = self.save_run(f"invalid-{key}-{type(value).__name__}", **{key: value})
                self.assertEqual(["--depth", "pass"], self.prompt("1", "")[0])

    def test_invalid_state_status_type_can_recover_from_checkpoints(self):
        directory = self.save_run()
        (directory / "run-state.json").write_text(json.dumps({"status": []}))
        self.assertEqual(["--resume", str(directory)], self.prompt("")[0])

    def test_noninteractive_no_arguments_requires_explicit_options(self):
        with patch.object(sys.stdin, "isatty", return_value=False), \
             patch.object(runner, "prompt_startup_options") as prompt, \
             contextlib.redirect_stderr(io.StringIO()) as error, self.assertRaises(SystemExit) as stop:
            runner.main([])
        self.assertEqual(2, stop.exception.code)
        prompt.assert_not_called()
        self.assertIn("explicit options", error.getvalue())

    def test_explicit_arguments_never_prompt(self):
        with patch.object(runner, "prompt_startup_options") as prompt, \
             patch.object(runner, "execute", return_value=0) as execute:
            self.assertEqual(0, runner.main(["--depth", "pass"]))
        prompt.assert_not_called()
        self.assertEqual("pass", execute.call_args.args[0].depth)

    def test_interactive_accept_dispatches_saved_resume_without_new_selection(self):
        directory = self.save_run()
        with patch.object(sys.stdin, "isatty", return_value=True), \
             patch.object(runner, "prompt_startup_options", return_value=["--resume", str(directory)]), \
             patch.object(runner, "execute", return_value=0) as execute:
            self.assertEqual(0, runner.main([]))
        self.assertEqual(["--resume", str(directory)], execute.call_args.args[1])
        self.assertEqual(directory, execute.call_args.args[2])

    def test_startup_eof_and_keyboard_interrupt_cancel_cleanly(self):
        for error in (EOFError, KeyboardInterrupt):
            with self.subTest(error=error), patch.object(sys.stdin, "isatty", return_value=True), \
                 patch.object(runner, "prompt_startup_options", side_effect=error), \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(130, runner.main([]))
                self.assertIn("cancelled", output.getvalue())


if __name__ == "__main__":
    unittest.main()
