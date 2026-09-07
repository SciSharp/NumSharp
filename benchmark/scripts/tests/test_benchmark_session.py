"""Run recovery and subprocess progress contracts (no benchmark workloads)."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import call, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_session import Session, RunLock, atomic_json, read_json, check_resume


class SessionTests(unittest.TestCase):
    def test_torn_or_invalid_progress_state_does_not_prevent_checkpoint_recovery(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            atomic_json(Path(temp) / "csharp/a-report-full-compressed.json", {
                "Benchmarks": [{"CaseId": "a", "Statistics": {"Min": 1}, "CheckpointStatus": "ok"}]})
            for content in ('{"status":', '[]', '{"stages": [], "elapsed_seconds": "bad"}'):
                with self.subTest(content=content):
                    (Path(temp) / "run-state.json").write_text(content)
                    session = Session(temp, [{"id": "a", "engine": "managed", "suite": "arithmetic"}], title=False)
                    self.assertEqual([], session.pending("managed", "arithmetic"))
                    session.progress(force=True)
                    self.assertEqual(1, read_json(Path(temp) / "run-state.json")["completed"])

    def test_parent_heartbeats_while_child_is_silent_and_captures_log(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()) as output:
            session = Session(temp, [{"id": "a", "engine": "managed", "suite": "arithmetic"}],
                              interval=0.1, title=False)
            session.stage, session.engine = "managed:arithmetic", "managed"
            code = ("import time; print('@@BENCHMARK {\"event\":\"start\",\"id\":\"a\"}',flush=True);"
                    "time.sleep(.45);print('child diagnostic',flush=True);"
                    "print('@@BENCHMARK {\"event\":\"complete\",\"id\":\"a\",\"status\":\"ok\"}',flush=True)")
            result = session.run([sys.executable, "-u", "-c", code], check=True)
            self.assertEqual(0, result.returncode)
            self.assertGreaterEqual(output.getvalue().count("[benchmark]"), 4)
            self.assertNotIn("child diagnostic\n[benchmark]", output.getvalue())
            self.assertIn("child diagnostic", (Path(temp) / "logs/managed_arithmetic.log").read_text())
            self.assertEqual(1, read_json(Path(temp) / "run-state.json")["completed"])

    def test_event_identity_includes_profile_and_duplicate_does_not_advance(self):
        plan = [{"id": "same", "engine": engine, "suite": "linalg"} for engine in ("managed", "openblas")]
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            session = Session(temp, plan, title=False)
            session.engine = "managed"
            event = {"event": "complete", "id": "same", "status": "ok"}
            session.event(event)
            session.event(event)
            session.progress(force=True)
            state = read_json(Path(temp) / "run-state.json")
            self.assertEqual((1, 1), (state["completed"], state["remaining"]))
            self.assertEqual(1, len(session.pending("openblas", "linalg")))

    def test_checkpoint_written_before_parent_event_is_recovered(self):
        with tempfile.TemporaryDirectory() as temp:
            atomic_json(Path(temp) / "csharp/a-report-full-compressed.json", {
                "Benchmarks": [{"CaseId": "a", "Statistics": {"Min": 1}, "CheckpointStatus": "ok"}]})
            session = Session(temp, [{"id": "a", "engine": "managed", "suite": "arithmetic"}], title=False)
            self.assertEqual([], session.pending("managed", "arithmetic"))

    def test_failed_checkpoint_with_partial_statistics_is_retried_and_counted_pending(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            atomic_json(Path(temp) / "csharp/a-report-full-compressed.json", {
                "Benchmarks": [{"CaseId": "a", "Statistics": {"Min": 1}, "CheckpointStatus": "failed"}]})
            session = Session(temp, [{"id": "a", "engine": "managed", "suite": "arithmetic"}], title=False)
            self.assertEqual(1, len(session.pending("managed", "arithmetic")))
            session.prepare_retry()
            session.recover()
            session.progress(force=True)
            state = read_json(Path(temp) / "run-state.json")
            self.assertEqual((0, 1), (state["failed"], state["remaining"]))

    def test_malformed_numpy_success_does_not_skip_pending_case(self):
        with tempfile.TemporaryDirectory() as temp:
            atomic_json(Path(temp) / "numpy-checkpoints/a.json", {"id": "a", "status": "ok"})
            session = Session(temp, [{"id": "a", "engine": "numpy", "suite": "arithmetic",
                                     "operation": "np.add(a,b)", "dtype": "float64", "n": 1000}], title=False)
            self.assertEqual(1, len(session.pending("numpy", "arithmetic")))

    def test_atomic_write_rejects_invalid_numbers_and_preserves_previous_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "state.json"
            atomic_json(path, {"version": 1})
            with self.assertRaises(ValueError):
                atomic_json(path, {"version": float("nan")})
            self.assertEqual({"version": 1}, read_json(path))

    def test_atomic_write_retries_windows_reader_conflicts_without_removing_previous_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "state.json"
            atomic_json(path, {"version": 1})
            replace = os.replace
            conflicts = iter((5, 32, 33, None))

            def briefly_locked(source, destination):
                self.assertEqual({"version": 1}, read_json(destination))
                self.assertEqual({"version": 2}, read_json(source))
                code = next(conflicts)
                if code is not None:
                    error = OSError("Reader temporarily denies replacement")
                    error.winerror = code
                    raise error
                replace(source, destination)

            with patch("benchmark_session.os.replace", side_effect=briefly_locked) as mocked_replace, \
                 patch("benchmark_session.time.sleep") as sleep:
                atomic_json(path, {"version": 2})
            self.assertEqual(4, mocked_replace.call_count)
            self.assertEqual([call(0.01), call(0.02), call(0.04)], sleep.call_args_list)
            self.assertEqual({"version": 2}, read_json(path))
            self.assertEqual([path], list(Path(temp).iterdir()))

    def test_atomic_write_exhausts_bounded_retries_and_preserves_previous_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "state.json"
            atomic_json(path, {"version": 1})
            error = OSError("Reader keeps the destination locked")
            error.winerror = 32
            with patch("benchmark_session.os.replace", side_effect=error) as mocked_replace, \
                 patch("benchmark_session.time.sleep") as sleep, self.assertRaises(OSError) as raised:
                atomic_json(path, {"version": 2})
            self.assertIs(error, raised.exception)
            self.assertEqual(6, mocked_replace.call_count)
            self.assertEqual([call(0.01), call(0.02), call(0.04), call(0.08), call(0.16)], sleep.call_args_list)
            self.assertLess(sum(item.args[0] for item in sleep.call_args_list), 1)
            self.assertEqual({"version": 1}, read_json(path))
            self.assertEqual([path], list(Path(temp).iterdir()))

    def test_atomic_write_does_not_retry_unrelated_or_non_windows_errors(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "state.json"
            atomic_json(path, {"version": 1})
            for code in (None, 87):
                with self.subTest(winerror=code):
                    error = OSError("Permanent replacement failure")
                    if code is not None:
                        error.winerror = code
                    with patch("benchmark_session.os.replace", side_effect=error) as mocked_replace, \
                         patch("benchmark_session.time.sleep") as sleep, self.assertRaises(OSError) as raised:
                        atomic_json(path, {"version": 2})
                    self.assertIs(error, raised.exception)
                    mocked_replace.assert_called_once()
                    sleep.assert_not_called()
                    self.assertEqual({"version": 1}, read_json(path))
                    self.assertEqual([path], list(Path(temp).iterdir()))

    def test_resume_rejects_different_code_and_environment(self):
        saved = {"source_sha256": "a", "numpy": "2.4.2", "commit": "old"}
        check_resume(saved, {**saved, "commit": "new"})
        with self.assertRaisesRegex(ValueError, "source_sha256"):
            check_resume(saved, {**saved, "source_sha256": "b"})
        with self.assertRaisesRegex(ValueError, "numpy"):
            check_resume(saved, {**saved, "numpy": "different"})

    def test_lock_blocks_second_runner_and_releases_on_exit(self):
        with tempfile.TemporaryDirectory() as temp:
            with RunLock(temp):
                with self.assertRaisesRegex(RuntimeError, "Another runner"):
                    with RunLock(temp):
                        self.fail("lock was acquired twice")
            with RunLock(temp):
                pass

    def test_timeout_terminates_child_and_keeps_run_state(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            session = Session(temp, interval=0.1, title=False)
            with self.assertRaises(TimeoutError):
                session.run([sys.executable, "-c", "import time; time.sleep(30)"], timeout=0.3)
            self.assertEqual("interrupted", read_json(Path(temp) / "run-state.json")["status"])


if __name__ == "__main__":
    unittest.main()
