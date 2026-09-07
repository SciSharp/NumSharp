"""Keep Windows benchmark descendants tied to their orchestrator's lifetime.

The unnamed job handle is non-inheritable and owned only by the Python parent.
KILL_ON_JOB_CLOSE therefore also works when that parent is terminated without
running Python cleanup. POSIX graceful cleanup remains the runner's process group.

Win32 layouts and lifetime contract:
https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects
https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_extended_limit_information
"""

from contextlib import contextmanager
import ctypes
from ctypes import wintypes
import os
import subprocess


class _BasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class _ExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def _kernel32():
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                               ctypes.c_void_p, wintypes.DWORD]
    kernel.SetInformationJobObject.restype = wintypes.BOOL
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    return kernel


def _stop_unprotected_child(proc):
    """A failed guard must never hand a live, unprotected child back to the caller."""
    try:
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=False, timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=10)


@contextmanager
def guard_process(proc):
    """Assign a just-created Popen to a kill-on-close Windows job.

    Enter immediately after Popen, before starting the output reader. Exiting the
    context kills any remaining descendants, even if the direct child exited first.
    Assignment failure stops the fresh child and raises instead of allowing a run
    that could leave benchmark processes writing checkpoints after parent death.
    """
    if os.name != "nt":
        yield
        return

    kernel = None
    job = None
    try:
        kernel = _kernel32()
        # NULL security attributes makes the handle non-inheritable. An unnamed job
        # cannot be reopened by a descendant, so parent death closes its last handle.
        job = kernel.CreateJobObjectW(None, None)
        if not job:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = _ExtendedLimitInformation()
        limits.BasicLimitInformation.LimitFlags = 0x00002000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            raise ctypes.WinError(ctypes.get_last_error())
        if not kernel.AssignProcessToJobObject(job, wintypes.HANDLE(int(proc._handle))):
            raise ctypes.WinError(ctypes.get_last_error())
    except BaseException as error:
        if job and kernel is not None:
            kernel.CloseHandle(job)
        try:
            _stop_unprotected_child(proc)
        except (OSError, subprocess.SubprocessError) as cleanup_error:
            raise RuntimeError(f"Cannot protect benchmark child {proc.pid} with a Windows Job Object: {error}. "
                               f"Child cleanup also failed: {cleanup_error}") from error
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        raise RuntimeError(f"Cannot protect benchmark child {proc.pid} with a Windows Job Object: {error}. "
                           "The child was stopped. Run from a terminal that permits nested Windows jobs.") from error

    try:
        yield
    finally:
        if not kernel.CloseHandle(job):
            raise RuntimeError(f"Could not close benchmark Windows Job Object: {ctypes.WinError(ctypes.get_last_error())}")
