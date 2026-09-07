#!/usr/bin/env python3
"""Real, opt-in interruption/resume experiment (Windows, requires psutil).

Runs an uninterrupted control and the same measured workload interrupted during
NumPy and C# Actual iterations. Only the runner PIDs created here are terminated;
their Windows Job Objects must clean up descendants without help from this script.
All evidence is saved beneath a new benchmark/results/ directory. No full benchmark
or published report is replaced. Independent timings are not expected to be equal.

Usage: python benchmark/scripts/verify_resume.py
"""
from __future__ import annotations

import argparse
from collections import Counter
import ctypes
from ctypes import wintypes
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from benchmark_session import atomic_json, read_json
from numpy_checkpoint import validate_checkpoint

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "benchmark"
RUNNER = BENCH / "run_benchmark.py"
PREFIX = "@@BENCHMARK "
FOLDERS = {"numpy": "numpy-checkpoints", "managed": "csharp", "openblas": "csharp-openblas"}
OPERATIONS = ["np.add(a, b)", "np.multiply(a, b)", "np.sqrt*"]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def identity(case):
    return case["engine"], case["id"]


def committed(directory):
    found = {}
    for engine, folder in FOLDERS.items():
        for path in (directory / folder).glob("*.json"):
            payload = read_json(path)
            if engine == "numpy":
                require(validate_checkpoint(payload), f"Invalid NumPy checkpoint: {path}")
                require(payload["status"] == "ok", f"Failed NumPy case: {path}")
                case_id = payload["id"]
            else:
                require(len(payload["Benchmarks"]) == 1, f"Expected one case: {path}")
                row = payload["Benchmarks"][0]
                require(row.get("CheckpointStatus") == "ok", f"Failed C# case: {path}")
                require(row.get("Statistics") is not None, f"No timing statistics: {path}")
                case_id = row["CaseId"]
                counts = Counter(m["IterationStage"] for m in row["Measurements"]
                                 if m["IterationMode"] == "Workload")
                require(counts["Actual"] == 50 and counts["Warmup"] == 5,
                        f"Wrong C# measurement depth: {path}: {counts}")
            key = engine, case_id
            require(key not in found, f"Duplicate checkpoint: {key}")
            found[key] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "mtime_ns": path.stat().st_mtime_ns, "size": path.stat().st_size}
    return found


def snapshot_json(snapshot):
    return [{"engine": key[0], "id": key[1], **value} for key, value in sorted(snapshot.items())]


def unchanged(snapshot):
    for key, record in snapshot.items():
        path = Path(record["path"])
        require(path.stat().st_mtime_ns == record["mtime_ns"], f"Completed checkpoint rewritten: {key}")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"],
                f"Completed measurement changed: {key}")


def events(directory):
    result = []
    for path in (directory / "logs").glob("*.log"):
        engine = path.stem.split("_", 1)[0]
        if engine not in FOLDERS:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith(PREFIX):
                result.append({**json.loads(line[len(PREFIX):]), "engine": engine})
    return result


class Descendants:
    """Retain real Windows process handles, avoiding any PID-reuse ambiguity."""
    def __init__(self, parent_pid):
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self.kernel.WaitForSingleObject.restype = wintypes.DWORD
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.CloseHandle.restype = wintypes.BOOL
        self.children = []
        for child in psutil.Process(parent_pid).children(recursive=True):
            try:
                handle = self.kernel.OpenProcess(0x00100000, False, child.pid)  # SYNCHRONIZE only
                if handle:
                    self.children.append({"pid": child.pid, "created": child.create_time(),
                                          "name": child.name(), "handle": handle})
            except psutil.NoSuchProcess:
                pass
        require(bool(self.children), "No live benchmark child to interrupt")

    def verify_exited(self):
        report = []
        try:
            for child in self.children:
                result = self.kernel.WaitForSingleObject(child["handle"], 5000)
                require(result == 0, f"Child survived runner termination: {child['pid']}")
                report.append({key: value for key, value in child.items() if key != "handle"})
        finally:
            for child in self.children:
                self.kernel.CloseHandle(child["handle"])
        return report


def launch(directory, evidence, phase, *, resume=False):
    command = [sys.executable, str(RUNNER)]
    if resume:
        command += ["--resume", str(directory)]
    else:
        command += ["--depth", "measure", "--dtypes", "float32,float64", "--suites", "arithmetic", "unary",
                    "--operations", *OPERATIONS, "--skip-openblas", "--run-dir", str(directory)]
    command += ["--skip-build", "--no-title", "--progress-interval", "0.5", "--attempts", "1"]
    log = (evidence / f"{phase}.log").open("w", encoding="utf-8")
    proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                            env={**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"},
                            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    print(f"{phase}: runner PID {proc.pid}; log {log.name}", flush=True)
    return proc, log, command


def wait_for_finish(proc, log, directory, phase, timeout):
    deadline = time.monotonic() + timeout
    next_update = 0
    try:
        while proc.poll() is None:
            require(time.monotonic() < deadline, f"{phase} exceeded {timeout}s")
            if time.monotonic() >= next_update:
                state = directory / "run-state.json"
                if state.exists():
                    value = read_json(state)
                    print(f"{phase}: {value['completed']}/{value['total']}, {value['remaining']} left, {value['stage']}", flush=True)
                next_update = time.monotonic() + 15
            time.sleep(.1)
        require(proc.returncode == 0, f"{phase} exited {proc.returncode}; see {log.name}")
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=10)
        log.close()


def interrupt_live(proc, log, directory, engine, offset, timeout):
    """Stop only after a 10M workload has started, before it commits its checkpoint."""
    stage_log = directory / "logs" / f"{engine}_arithmetic.log"
    deadline = time.monotonic() + timeout
    active = None
    buffer = ""
    position = offset
    try:
        while proc.poll() is None and time.monotonic() < deadline:
            if stage_log.exists():
                with stage_log.open(encoding="utf-8") as stream:
                    stream.seek(position)
                    buffer += stream.read()
                    position = stream.tell()
                lines = buffer.split("\n")
                buffer = lines.pop()
                for line in lines:
                    if line.startswith(PREFIX):
                        event = json.loads(line[len(PREFIX):])
                        if event["event"] == "start":
                            active = {"id": event["id"], "started": time.monotonic(), "actual_seen": False}
                        elif event["event"] == "complete":
                            active = None
                    elif active and line.startswith("WorkloadActual"):
                        active["actual_seen"] = True
                if active:
                    plan = {identity(case): case for case in read_json(directory / "plan.json")}
                    key = engine, active["id"]
                    case = plan[key]
                    ready = active["actual_seen"] if engine == "managed" else time.monotonic() - active["started"] >= .05
                    if case["n"] == 10_000_000 and ready:
                        descendants = Descendants(proc.pid)
                        proc.kill()  # TerminateProcess on the RUNNER ONLY; no Python cleanup runs.
                        exit_code = proc.wait(timeout=10)
                        exited = descendants.verify_exited()
                        snapshot = committed(directory)
                        require(key not in snapshot, f"Interruption raced with a completed case: {key}")
                        require(sum(k[0] == engine for k in snapshot) >= 2, "Need completed cases before interruption")
                        return {"engine": engine, "active_case": case, "runner_pid": proc.pid,
                                "runner_exit_code": exit_code, "descendants_exited": exited,
                                "completed_before_interrupt": len(snapshot), "snapshot": snapshot_json(snapshot),
                                "interruption": "TerminateProcess on parent during live workload",
                                "actual_measurement_seen": active["actual_seen"]}
            time.sleep(.005)
        raise AssertionError(f"No live {engine} interruption reached; exit={proc.poll()}, log={log.name}")
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=10)
        log.close()


def stable_report(directory):
    payload = read_json(directory / "benchmark-report.json")
    result = []
    for row in payload["rows"]:
        result.append({key: row.get(key) for key in ("operation", "suite", "category", "dtype", "n", "scenario")}
                      | {"profiles": {profile: {key: value.get(key) for key in ("availability", "actual_backend")}
                                      for profile, value in row["profiles"].items()}})
    return sorted(result, key=lambda row: json.dumps(row, sort_keys=True))


def verify_complete(directory, expected):
    snapshot = committed(directory)
    require(set(snapshot) == expected, "Missing or extra final checkpoints")
    state = read_json(directory / "run-state.json")
    require((state["status"], state["total"], state["completed"], state["failed"], state["remaining"])
            == ("complete", len(expected), len(expected), 0, 0), f"Incomplete final state: {state}")
    raw_numpy = [read_json(path)["result"] for path in (directory / "numpy-checkpoints").glob("*.json")]
    order = lambda rows: sorted(json.dumps(row, sort_keys=True) for row in rows)
    require(order(raw_numpy) == order(read_json(directory / "numpy-results.json")), "NumPy aggregate differs from committed measurements")
    complete_events = Counter(identity(event) for event in events(directory) if event["event"] == "complete")
    require(complete_events == Counter({key: 1 for key in expected}), "Expected exactly one successful completion per planned case")
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New evidence directory")
    parser.add_argument("--timeout", type=float, default=600, help="Maximum seconds per phase")
    args = parser.parse_args()
    require(os.name == "nt", "This proof exercises Windows parent termination; run it on Windows")
    evidence = (args.output or BENCH / "results" / ("resume-proof-" + datetime.now().strftime("%Y%m%d-%H%M%S"))).resolve()
    require(not evidence.exists(), f"Use a new evidence directory: {evidence}")
    evidence.mkdir(parents=True)
    control, interrupted = evidence / "control", evidence / "interrupted"
    result = {"status": "running", "evidence": str(evidence), "interruptions": [],
              "timing_comparison": "Independent timings vary. Completed measurements must remain byte-for-byte unchanged."}
    atomic_json(evidence / "verification.json", result)
    try:
        # Build before both cohorts, then execute identical pinned binaries throughout.
        with (evidence / "build.log").open("w", encoding="utf-8") as log:
            subprocess.run(["dotnet", "build", "-c", "Release", "-f", "net10.0",
                            str(BENCH / "NumSharp.Benchmark.CSharp/NumSharp.Benchmark.CSharp.csproj"),
                            "-v", "q", "--nologo", "-p:GeneratePackageOnBuild=false"],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=args.timeout)
        proc, log, command = launch(control, evidence, "control")
        result["control_command"] = command
        wait_for_finish(proc, log, control, "control", args.timeout)
        plan = read_json(control / "plan.json")
        expected = {identity(case) for case in plan}
        require(len(expected) == 36, f"Expected 36 real measurements, discovered {len(expected)}")
        verify_complete(control, expected)

        proc, log, command = launch(interrupted, evidence, "interrupt-numpy")
        result["interrupted_command"] = command
        record = interrupt_live(proc, log, interrupted, "numpy", 0, args.timeout)
        result["interruptions"].append(record)
        first_snapshot = committed(interrupted)
        before_first_resume = events(interrupted)
        atomic_json(evidence / "verification.json", result)
        print(f"Stopped during NumPy: {len(first_snapshot)}/36 saved; all {len(record['descendants_exited'])} descendants exited", flush=True)

        proc, log, command = launch(interrupted, evidence, "resume-then-interrupt-csharp", resume=True)
        record = interrupt_live(proc, log, interrupted, "managed", 0, args.timeout)
        result["interruptions"].append(record)
        unchanged(first_snapshot)
        second_snapshot = committed(interrupted)
        before_final_resume = events(interrupted)
        atomic_json(evidence / "verification.json", result)
        print(f"Stopped during C#: {len(second_snapshot)}/36 saved; all {len(record['descendants_exited'])} descendants exited", flush=True)

        proc, log, command = launch(interrupted, evidence, "resume-to-completion", resume=True)
        wait_for_finish(proc, log, interrupted, "resume-to-completion", args.timeout)
        final_snapshot = verify_complete(interrupted, expected)
        unchanged(first_snapshot)
        unchanged(second_snapshot)
        require(read_json(interrupted / "plan.json") == plan, "Resumed plan differs from uninterrupted control")
        require(read_json(interrupted / "build-identity.json") == read_json(control / "build-identity.json"), "Different executed binaries")
        require(read_json(interrupted / "run-config.json")["provenance"] == read_json(control / "run-config.json")["provenance"], "Different code/environment")
        require(stable_report(interrupted) == stable_report(control), "Final report case/profile structure differs from control")

        starts = lambda stream: Counter(identity(event) for event in stream if event["event"] == "start")
        all_events = events(interrupted)
        after_first = starts(all_events) - starts(before_first_resume)
        after_second = starts(all_events) - starts(before_final_resume)
        require(not (set(after_first) & set(first_snapshot)), "First resume re-executed completed cases")
        require(after_second == Counter({key: 1 for key in expected - set(second_snapshot)}), "Final resume did not execute exactly the pending cases")
        interrupted_ids = {identity(record["active_case"]) for record in result["interruptions"]}
        require(starts(all_events) == Counter({key: 2 if key in interrupted_ids else 1 for key in expected}),
                "Only the two interrupted cases should execute twice")

        # An already-complete resume must change no timing/report payload and start no workload.
        report_before = read_json(interrupted / "benchmark-report.json")
        proc, log, command = launch(interrupted, evidence, "resume-already-complete", resume=True)
        wait_for_finish(proc, log, interrupted, "resume-already-complete", args.timeout)
        unchanged(final_snapshot)
        require(events(interrupted) == all_events, "Completed resume executed workloads")
        require(read_json(interrupted / "benchmark-report.json") == report_before, "Completed resume changed final report values")
        result.update(status="passed", total_cases=len(expected), by_engine=dict(Counter(case["engine"] for case in plan)),
                      report_rows=len(report_before["rows"]), preserved_checkpoints_after_first=len(first_snapshot),
                      preserved_checkpoints_after_second=len(second_snapshot), final_pending_executed=len(after_second),
                      duplicate_completions=0, missing_cases=0, failed_cases=0,
                      completed_resume_workload_starts=0, final_report_unchanged_on_completed_resume=True,
                      same_case_profile_structure_as_control=True, same_source_environment_and_binaries=True)
        print(f"PASS: {len(expected)} cases; both interruptions recovered; completed data unchanged; no missing/duplicate cases", flush=True)
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        atomic_json(evidence / "verification.json", result)
        print(f"Evidence: {evidence / 'verification.json'}", flush=True)


if __name__ == "__main__":
    main()
