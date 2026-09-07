"""Host-stability controls shared by run_benchmark.py (orchestrator) and numpy_benchmark.py.

Two ambient, uncontrolled sources of wall-clock variance exist on a hybrid, turbo-boosting
host, and both were measured on the i9-13900K this suite runs on (8 P-cores + 16 E-cores):

* **Turbo boost.** The High-performance power plan ships "Processor performance boost mode"
  = Aggressive, which is *hidden* from ``powercfg -query`` until unhidden. Min/Max processor
  state = 100 %/100 % does NOT stop it — those cap the non-turbo P0 state and boost sits above
  it. A pinned spin load reads 166–174 % "Processor Performance" (≈5.0–5.2 GHz on a 3.0 GHz
  nominal) with boost on and 91–95 % with boost Disabled: a ~1.7× clock swing. Under a shared
  package power/thermal budget the sustained clock sits below max and jumps to full single-core
  boost whenever the orchestrator's other threads idle — seen as a 30–50 % *faster* excursion
  in 2 of 50 BenchmarkDotNet iterations that never reproduces. BDN keeps fast outliers
  (``OutlierMode.RemoveUpper``), so ``Statistics.Min`` — the harness's best-window basis —
  reported that lucky window.
* **Core placement.** An unpinned thread migrates between P-cores of differing boost residency
  and can land on an E-core (2–3× slower for everything). Only the nditer probes were pinned
  (``NS_PROBE_AFFINITY``); the official op matrix and the NumPy side were not.

``ClockLock`` removes the first for BOTH languages at once (the setting is system-wide) and
the core pin removes the second per process. Both sides then run on the same silicon at the
same fixed clock, so ratios are reproducible. Absolute times drop by the base/boost ratio on
both sides equally — this trades peak numbers for repeatable ones, which is what a comparison
needs. Opt out with ``--no-lock-clock`` / ``--no-pin-core``.

Windows-specific by nature (powercfg + SetProcessAffinityMask); every entry point degrades to a
logged no-op elsewhere so the run never fails over a host control.
"""
from __future__ import annotations

import atexit
import json
import os
import re
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

# powercfg GUIDs. SUB_PROCESSOR is the "Processor power management" subgroup; PERFBOOSTMODE is
# "Processor performance boost mode" (hidden by default — see ClockLock.lock).
HIGH_PERFORMANCE_SCHEME = "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"   # what BenchmarkDotNet activates
SUB_PROCESSOR = "54533251-82be-4824-96c1-47b60b740d00"
PERFBOOSTMODE = "be337238-0d82-4146-a960-4f3749d470c7"
BOOST_DISABLED = 0
BOOST_MODE_NAMES = {
    0: "Disabled", 1: "Enabled", 2: "Aggressive", 3: "Efficient Enabled",
    4: "Efficient Aggressive", 5: "Aggressive At Guaranteed", 6: "Efficient Aggressive At Guaranteed",
}

# The per-process pin contract. The orchestrator exports it; both C# runners
# (Infrastructure/BenchmarkHost.cs, via OfficialBenchmarkConfig) and numpy_benchmark.py honor
# it. NS_PROBE_AFFINITY is the nditer probes' older spelling, accepted as a fallback.
AFFINITY_VARIABLE = "NUMSHARP_BENCHMARK_AFFINITY"
LEGACY_AFFINITY_VARIABLE = "NS_PROBE_AFFINITY"


def _log(message: str) -> None:
    print(f"[host] {message}", flush=True)


# ----------------------------------------------------------------------------------------
# Core selection + pinning
# ----------------------------------------------------------------------------------------

def logical_cpus_of(mask: int) -> list[int]:
    return [i for i in range(64) if (mask >> i) & 1]


def _windows_physical_cores() -> list[tuple[int, list[int]]]:
    """(EfficiencyClass, [logical CPUs]) per physical core via GetLogicalProcessorInformationEx.

    On an Intel hybrid part P-cores carry the HIGHEST EfficiencyClass (E-cores are 0); on a
    homogeneous part every core shares one class."""
    import ctypes
    import ctypes.wintypes as wt

    k32 = ctypes.windll.kernel32
    relation_processor_core = 0

    class GroupAffinity(ctypes.Structure):
        _fields_ = [("Mask", ctypes.c_size_t), ("Group", wt.WORD), ("Reserved", wt.WORD * 3)]

    class ProcessorRelationship(ctypes.Structure):
        _fields_ = [("Flags", ctypes.c_byte), ("EfficiencyClass", ctypes.c_byte),
                    ("Reserved", ctypes.c_byte * 20), ("GroupCount", wt.WORD),
                    ("GroupMask", GroupAffinity * 1)]

    class Info(ctypes.Structure):
        _fields_ = [("Relationship", wt.DWORD), ("Size", wt.DWORD), ("Processor", ProcessorRelationship)]

    size = wt.DWORD(0)
    k32.GetLogicalProcessorInformationEx(relation_processor_core, None, ctypes.byref(size))
    buffer = ctypes.create_string_buffer(size.value)
    if not k32.GetLogicalProcessorInformationEx(relation_processor_core, buffer, ctypes.byref(size)):
        return []
    cores, offset = [], 0
    while offset < size.value:
        info = Info.from_buffer(buffer, offset)
        cores.append((info.Processor.EfficiencyClass, logical_cpus_of(info.Processor.GroupMask[0].Mask)))
        offset += info.Size
    return cores


def pick_benchmark_core() -> int | None:
    """Affinity mask for ONE performance core: the first SMT thread of the second P-core.

    Logical CPU 0 is avoided because the OS parks timer/interrupt work there. On the i9-13900K
    (P-cores = logical 0–15 in SMT pairs, E-cores = 16–31) this is logical 2 → mask 0x4, the
    same core the nditer probes were validated on. Returns None when the topology cannot be read."""
    try:
        if sys.platform == "win32":
            cores = _windows_physical_cores()
            if not cores:
                return None
            top = max(cls for cls, _ in cores)
            candidates = [cpus[0] for cls, cpus in cores if cls == top and cpus]
        else:
            count = os.cpu_count() or 1
            candidates = list(range(count))
        preferred = [cpu for cpu in candidates if cpu != 0] or candidates
        return 1 << preferred[0] if preferred else None
    except Exception as error:  # noqa: BLE001 — a host probe must never fail the run
        _log(f"could not read the CPU topology ({error}); not pinning")
        return None


def parse_mask(text: str | None) -> int | None:
    if not text or not text.strip():
        return None
    value = text.strip().lower()
    mask = int(value[2:] if value.startswith("0x") else value, 16)
    return mask if mask > 0 else None


def pin_current_process(mask: int) -> bool:
    """Pin this process (all its threads) to ``mask``. Windows: SetProcessAffinityMask; Linux:
    sched_setaffinity. Returns False (after logging) rather than raising."""
    try:
        if sys.platform == "win32":
            import ctypes
            import ctypes.wintypes as wt
            # The explicit argtypes/restype are load-bearing. Without them GetCurrentProcess()
            # comes back as a 32-bit c_int (-1) and the 64-bit HANDLE register receives
            # 0x00000000FFFFFFFF instead of the all-ones pseudo-handle, so SetProcessAffinityMask
            # fails SILENTLY (returns 0, mask untouched). The nditer probes' NumPy twin carried
            # exactly that call and was never actually pinned — hence the read-back below: a pin
            # is proven applied, never assumed.
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.GetCurrentProcess.restype = wt.HANDLE
            k32.SetProcessAffinityMask.argtypes = [wt.HANDLE, ctypes.c_size_t]
            k32.SetProcessAffinityMask.restype = wt.BOOL
            k32.GetProcessAffinityMask.argtypes = [wt.HANDLE, ctypes.POINTER(ctypes.c_size_t),
                                                   ctypes.POINTER(ctypes.c_size_t)]
            k32.GetProcessAffinityMask.restype = wt.BOOL
            if not k32.SetProcessAffinityMask(k32.GetCurrentProcess(), mask):
                code = ctypes.get_last_error()
                raise OSError(code, ctypes.FormatError(code).strip())
            applied, system = ctypes.c_size_t(), ctypes.c_size_t()
            k32.GetProcessAffinityMask(k32.GetCurrentProcess(), ctypes.byref(applied), ctypes.byref(system))
            if applied.value != mask:
                raise OSError(f"affinity read back as {hex(applied.value)}, not {hex(mask)}")
        elif hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, set(logical_cpus_of(mask)))
        else:
            _log("process pinning is not supported on this platform; skipping")
            return False
        _log(f"pinned process affinity to {hex(mask)} (logical CPUs {logical_cpus_of(mask)})")
        return True
    except Exception as error:  # noqa: BLE001
        _log(f"could not pin process affinity to {hex(mask)} ({error}); continuing unpinned")
        return False


def pin_current_process_from_env() -> int | None:
    """Honor NUMSHARP_BENCHMARK_AFFINITY (or the probes' NS_PROBE_AFFINITY). The contract for
    every child the orchestrator launches; returns the mask applied, or None."""
    mask = parse_mask(os.environ.get(AFFINITY_VARIABLE)) or parse_mask(os.environ.get(LEGACY_AFFINITY_VARIABLE))
    if mask is None:
        return None
    return mask if pin_current_process(mask) else None


# ----------------------------------------------------------------------------------------
# Clock lock (turbo boost off for the duration of the run)
# ----------------------------------------------------------------------------------------

def _powercfg(*args: str) -> tuple[int, str]:
    try:
        proc = subprocess.run(["powercfg", *args], capture_output=True, text=True)
    except OSError as error:
        return 1, str(error)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _read_boost_mode(scheme: str) -> tuple[int | None, int | None]:
    _, out = _powercfg("-query", scheme, SUB_PROCESSOR, PERFBOOSTMODE)
    ac = re.search(r"Current AC Power Setting Index:\s*0x([0-9a-fA-F]+)", out)
    dc = re.search(r"Current DC Power Setting Index:\s*0x([0-9a-fA-F]+)", out)
    return (int(ac.group(1), 16) if ac else None, int(dc.group(1), 16) if dc else None)


def _active_scheme() -> str | None:
    _, out = _powercfg("-getactivescheme")
    match = re.search(r"([0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12})", out)
    return match.group(1).lower() if match else None


class ClockLock:
    """Disable turbo boost system-wide for the run and restore it afterwards.

    ``lock()`` unhides PERFBOOSTMODE, records the original AC/DC values and the originally
    active scheme, sets boost to Disabled on the High-performance scheme (BenchmarkDotNet
    activates that scheme itself for the C# phases; activating it here too puts the NumPy phase
    and every subsystem under the same plan) and activates it. ``restore()`` puts every value
    back. Restoration is guaranteed on every exit path Python controls (``atexit`` fires on
    normal exit, ``sys.exit``, uncaught exceptions and KeyboardInterrupt); a hard kill cannot
    run finalizers, so the original values are also written to ``state_file`` FIRST and a
    later ``lock()`` that finds that file repairs the host before doing anything else.

    Deliberately mutates the High-performance scheme's boost setting (with restore) rather
    than cloning a scheme, because BDN would still force High-performance for the C# phase
    unless its config were changed to UserPowerPlan — one mechanism covering both languages
    beats two coordinated ones."""

    def __init__(self, enabled: bool = True, scheme: str = HIGH_PERFORMANCE_SCHEME,
                 state_file: Path | None = None) -> None:
        self.enabled = enabled
        self.scheme = scheme
        self.state_file = state_file
        self.original: dict | None = None
        self.locked = False
        self.summary = "clock not locked"

    # -- self-heal ----------------------------------------------------------------------
    def _repair_from_state_file(self) -> None:
        if not self.state_file or not self.state_file.exists():
            return
        try:
            stale = json.loads(self.state_file.read_text(encoding="utf-8"))
            _log(f"a previous run died while the clock was locked ({self.state_file.name}); repairing first")
            self._apply(stale)
        except Exception as error:  # noqa: BLE001
            _log(f"could not repair from {self.state_file} ({error}); check `powercfg` by hand")
        finally:
            try:
                self.state_file.unlink()
            except OSError:
                pass

    def _apply(self, values: dict) -> None:
        scheme = values["scheme"]
        _powercfg("-setacvalueindex", scheme, SUB_PROCESSOR, PERFBOOSTMODE, str(values["ac"]))
        _powercfg("-setdcvalueindex", scheme, SUB_PROCESSOR, PERFBOOSTMODE, str(values["dc"]))
        _powercfg("-setactive", values.get("active_scheme") or scheme)

    # -- lock / restore ------------------------------------------------------------------
    def lock(self) -> bool:
        if not self.enabled:
            self.summary = "clock not locked (--no-lock-clock)"
            _log(self.summary)
            return False
        if sys.platform != "win32":
            self.summary = "clock lock unavailable on this platform (powercfg is Windows-only)"
            _log(self.summary)
            return False
        self._repair_from_state_file()
        # The setting is hidden until this attribute flip; -query returns nothing for it otherwise.
        _powercfg("-attributes", SUB_PROCESSOR, PERFBOOSTMODE, "-ATTRIB_HIDE")
        ac, dc = _read_boost_mode(self.scheme)
        if ac is None or dc is None:
            self.summary = "clock lock skipped: could not read the boost-mode setting"
            _log(self.summary)
            return False
        self.original = {"scheme": self.scheme, "ac": ac, "dc": dc, "active_scheme": _active_scheme()}
        if self.state_file:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.state_file.write_text(json.dumps(self.original, indent=2), encoding="utf-8")
        rc1, _ = _powercfg("-setacvalueindex", self.scheme, SUB_PROCESSOR, PERFBOOSTMODE, str(BOOST_DISABLED))
        rc2, _ = _powercfg("-setdcvalueindex", self.scheme, SUB_PROCESSOR, PERFBOOSTMODE, str(BOOST_DISABLED))
        rc3, _ = _powercfg("-setactive", self.scheme)
        now_ac, now_dc = _read_boost_mode(self.scheme)
        if rc1 or rc2 or rc3 or now_ac != BOOST_DISABLED or now_dc != BOOST_DISABLED:
            self.summary = "clock lock FAILED to apply; restoring and continuing with boost enabled"
            _log(self.summary)
            self.restore()
            return False
        self.locked = True
        atexit.register(self.restore)
        self.summary = (f"clock LOCKED: boost mode {BOOST_MODE_NAMES.get(ac, ac)}/{BOOST_MODE_NAMES.get(dc, dc)} "
                        f"-> Disabled on the High-performance scheme (restored on exit)")
        _log(self.summary)
        return True

    def restore(self) -> None:
        if self.original is None:
            return
        values, self.original = self.original, None
        self.locked = False
        self._apply(values)
        ac, dc = _read_boost_mode(values["scheme"])
        _log(f"clock restored: boost mode back to {BOOST_MODE_NAMES.get(ac, ac)}/{BOOST_MODE_NAMES.get(dc, dc)}")
        if self.state_file:
            try:
                self.state_file.unlink()
            except OSError:
                pass

    def __enter__(self) -> "ClockLock":
        self.lock()
        return self

    def __exit__(self, *_exc) -> None:
        self.restore()


@contextmanager
def stable_clock(enabled: bool = True, state_file: Path | None = None):
    """``with stable_clock(): ...`` form for standalone scripts."""
    lock = ClockLock(enabled=enabled, state_file=state_file)
    try:
        lock.lock()
        yield lock
    finally:
        lock.restore()
