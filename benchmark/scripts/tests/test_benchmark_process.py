"""Windows process lifetime regressions; no NumPy/NumSharp workloads execute."""

import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import benchmark_process


@unittest.skipUnless(os.name == "nt", "Windows Job Object lifetime contract")
class WindowsProcessGuardTests(unittest.TestCase):
    def setUp(self):
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self.kernel.WaitForSingleObject.restype = wintypes.DWORD
        self.kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        self.kernel.TerminateProcess.restype = wintypes.BOOL
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.CloseHandle.restype = wintypes.BOOL

    def process_handle(self, pid):
        handle = self.kernel.OpenProcess(0x00100001, False, pid)  # SYNCHRONIZE | PROCESS_TERMINATE
        self.assertTrue(handle, f"Cannot open test process {pid}: {ctypes.WinError(ctypes.get_last_error())}")
        self.addCleanup(self.cleanup_process_handle, handle)
        return handle

    def cleanup_process_handle(self, handle):
        if self.kernel.WaitForSingleObject(handle, 0) == 258:  # WAIT_TIMEOUT
            self.kernel.TerminateProcess(handle, 99)
        self.kernel.CloseHandle(handle)

    def test_windows_64bit_struct_layout(self):
        if ctypes.sizeof(ctypes.c_void_p) == 8:
            self.assertEqual(64, ctypes.sizeof(benchmark_process._BasicLimitInformation))
            self.assertEqual(144, ctypes.sizeof(benchmark_process._ExtendedLimitInformation))
            self.assertEqual(16, benchmark_process._BasicLimitInformation.LimitFlags.offset)
            self.assertEqual(112, benchmark_process._ExtendedLimitInformation.ProcessMemoryLimit.offset)

    def test_normal_child_can_complete_successfully(self):
        with subprocess.Popen([sys.executable, "-c", "print('completed')"],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
            with benchmark_process.guard_process(proc):
                output, error = proc.communicate(timeout=10)
                self.assertEqual(0, proc.returncode, error)
                self.assertEqual("completed", output.strip())

    def test_context_exit_terminates_remaining_child(self):
        with subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"]) as proc:
            with benchmark_process.guard_process(proc):
                self.assertIsNone(proc.poll())
            # KILL_ON_JOB_CLOSE may use exit status zero; the lifetime, not its code,
            # is the contract (waiting without the guard would take sixty seconds).
            self.assertIsNotNone(proc.wait(timeout=5))

    def test_assignment_failure_stops_fresh_child_and_raises(self):
        with subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"]) as proc:
            kernel = benchmark_process._kernel32()
            def denied_assignment(*_):
                ctypes.set_last_error(5)
                return False
            kernel.AssignProcessToJobObject = denied_assignment
            with patch.object(benchmark_process, "_kernel32", return_value=kernel):
                with self.assertRaisesRegex(RuntimeError, "child was stopped"):
                    with benchmark_process.guard_process(proc):
                        self.fail("An unguarded child was allowed to execute")
            self.assertIsNotNone(proc.poll())

    def test_killing_parent_terminates_guarded_child_and_grandchild(self):
        with tempfile.TemporaryDirectory(prefix="numsharp-job-") as temp:
            marker = Path(temp) / "children.json"
            child_source = ("import json,subprocess,sys,time; from pathlib import Path; "
                            "grandchild=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']); "
                            "Path(sys.argv[1]).write_text(json.dumps({'child':__import__('os').getpid(),"
                            "'grandchild':grandchild.pid})); time.sleep(60)")
            # The child waits on stdin so its descendant is created only after the
            # parent successfully assigns the job; no launch race enters this test.
            parent_source = ("import subprocess,sys,time; "
                             f"sys.path.insert(0,{str(SCRIPTS)!r}); "
                             "from benchmark_process import guard_process; "
                             f"source={('import sys;sys.stdin.readline();' + child_source)!r}; "
                             "child=subprocess.Popen([sys.executable,'-c',source,sys.argv[1]],stdin=subprocess.PIPE,text=True); "
                             "guard=guard_process(child);guard.__enter__(); "
                             "child.stdin.write('go\\n');child.stdin.flush();time.sleep(60)")
            with subprocess.Popen([sys.executable, "-c", parent_source, str(marker)],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True) as parent:
                try:
                    deadline = time.monotonic() + 10
                    while not marker.exists() and parent.poll() is None and time.monotonic() < deadline:
                        time.sleep(0.02)
                    if not marker.exists():
                        parent.kill()
                        _, error = parent.communicate(timeout=5)
                        self.fail(f"Guarded child did not start: {error}")
                    pids = json.loads(marker.read_text())
                    child = self.process_handle(pids["child"])
                    grandchild = self.process_handle(pids["grandchild"])
                    parent.kill()  # TerminateProcess: no Python finally/context cleanup executes.
                    parent.wait(timeout=5)
                    self.assertEqual(0, self.kernel.WaitForSingleObject(child, 5000), "Child survived parent death")
                    self.assertEqual(0, self.kernel.WaitForSingleObject(grandchild, 5000), "Grandchild survived parent death")
                finally:
                    if parent.poll() is None:
                        parent.kill()
                    parent.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
