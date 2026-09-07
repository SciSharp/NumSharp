"""Durable run state and parent-owned progress for long benchmark processes."""
from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import platform
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from numpy_checkpoint import validate_checkpoint
from benchmark_process import guard_process


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temp.open("w", encoding="utf-8") as output:
            json.dump(value, output, indent=2, allow_nan=False)
            output.flush()
            os.fsync(output.fileno())
        # Windows readers may briefly deny deletion while holding the old file open.
        # Keep both versions intact and retry only those Windows sharing/access errors.
        retry_delays = (0.01, 0.02, 0.04, 0.08, 0.16)
        for attempt in range(len(retry_delays) + 1):
            try:
                os.replace(temp, path)
                break
            except OSError as error:
                if getattr(error, "winerror", None) not in (5, 32, 33) or attempt == len(retry_delays):
                    raise
                time.sleep(retry_delays[attempt])
    finally:
        temp.unlink(missing_ok=True)


class RunLock:
    """OS lock: released on process death, with no unsafe stale-PID deletion."""
    def __init__(self, directory):
        self.path = Path(directory) / "run.lock"

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+b")
        try:
            if os.fstat(self.handle.fileno()).st_size == 0:
                self.handle.write(b" ")
                self.handle.flush()
            self.handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.handle.close()
            raise RuntimeError(f"Another runner owns {self.path.parent}") from error
        return self

    def __exit__(self, *_):
        self.handle.close()


def provenance(root):
    """Hash actual code, including dirty/new files; a commit SHA alone is insufficient."""
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()
    digest = hashlib.sha256()
    paths = git("ls-files", "--cached", "--others", "--exclude-standard", "-z").split("\0")
    for name in sorted(set(paths)):
        path = Path(root) / name
        if (name.startswith(("src/", "benchmark/")) or "/" not in name) and path.suffix in {
                ".cs", ".py", ".csproj", ".props", ".targets", ".slnx"}:
            if path.is_file():
                digest.update(name.encode())
                digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    import numpy
    return {"commit": git("rev-parse", "HEAD"), "source_sha256": digest.hexdigest(),
            "machine": platform.node(), "platform": platform.platform(),
            "processor": platform.processor(), "python": sys.version,
            "numpy": numpy.__version__,
            "dotnet": subprocess.check_output(["dotnet", "--version"], text=True).strip(),
            "environment": {name: os.environ.get(name) for name in (
                "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_CORETYPE",
                "NUMSHARP_OPENBLAS_LIBRARY", "NUMSHARP_OPENBLAS_VERSION",
                "NUMSHARP_OPENBLAS_USE_BUNDLED", "DOTNET_EnableHWIntrinsic",
                "DOTNET_EnableAVX2", "DOTNET_EnableAVX512F")}}


def check_resume(saved, current):
    changed = [key for key in current if key != "commit" and saved.get(key) != current[key]]
    if changed:
        raise ValueError("Resume would mix different code or environments (" + ", ".join(changed)
                         + "). Start a new run; use --rerun to select previous problem cases.")


def duration(seconds):
    seconds = max(0, int(seconds))
    if seconds >= 3600:
        return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"
    return f"{seconds // 60}m {seconds % 60:02d}s"


class Session:
    def __init__(self, directory, plan=(), interval=10, title=True, verbose=False, process_timeout=None):
        self.directory = Path(directory)
        self.interval = interval
        self.title = title
        self.verbose = verbose
        self.process_timeout = process_timeout
        self.plan = list(plan)
        self.cells = {(row["engine"], row["id"]): row for row in self.plan}
        self.outcomes = {}
        self.started = {}
        self.samples = {}
        self.retry_pending = set()
        self.checkpoint_cache = {}
        config = read_json(self.directory / "run-config.json") if (self.directory / "run-config.json").exists() else {}
        self.numpy_settings = ({"depth": config["depth"], "iterations": 50,
                                "numpy_version": config["provenance"]["numpy"]}
                               if config.get("provenance", {}).get("numpy") else None)
        self.stage = "preparing"
        self.engine = None
        self.stage_state = {}
        self.start_time = time.monotonic()
        self.last_progress = 0
        self.state_path = self.directory / "run-state.json"
        self.elapsed_before = 0
        if self.state_path.exists():
            try:
                previous = read_json(self.state_path)
            except (ValueError, OSError):
                previous = {}
            if isinstance(previous, dict):
                stages = previous.get("stages", {})
                self.stage_state = stages if isinstance(stages, dict) else {}
                elapsed = previous.get("elapsed_seconds", 0)
                if type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0:
                    self.elapsed_before = elapsed
        events = self.directory / "events.jsonl"
        if events.exists():
            with events.open(encoding="utf-8") as stream:
                for line in stream:
                    try:
                        event = json.loads(line)
                        elapsed = event.get("elapsed_seconds")
                        if event.get("status") == "ok" and isinstance(elapsed, (float, int)) and elapsed >= 0:
                            self.samples.setdefault(event["engine"], []).append(elapsed)
                    except (ValueError, KeyError):
                        pass  # an interrupted final journal line has no effect on recovery
        self.recover()

    def set_plan(self, plan):
        self.plan = list(plan)
        self.cells = {(row["engine"], row["id"]): row for row in self.plan}
        self.recover()

    def recover(self):
        # Checkpoints are the truth, including a file written just before a killed parent
        # could receive its completion event. A torn temp file never counts as complete.
        self.outcomes = {}
        for engine, folder in (("numpy", "numpy-checkpoints"), ("managed", "csharp"),
                               ("openblas", "csharp-openblas")):
            for path in (self.directory / folder).glob("*.json"):
                try:
                    stat = path.stat()
                    signature = stat.st_mtime_ns, stat.st_size
                    cached = self.checkpoint_cache.get(path)
                    if cached is not None and cached[0] == signature:
                        self.outcomes.update(cached[1])
                        continue
                    payload = read_json(path)
                    outcomes = {}
                    if engine == "numpy":
                        expected = self.cells.get((engine, payload.get("id")))
                        case = {key: expected[key] for key in ("id", "suite", "operation", "dtype", "n")} if expected else None
                        if validate_checkpoint(payload, case, self.numpy_settings):
                            outcomes[engine, payload["id"]] = payload["status"]
                    else:
                        for row in payload.get("Benchmarks", []):
                            if row.get("CaseId"):
                                outcomes[engine, row["CaseId"]] = row.get("CheckpointStatus", "ok" if row.get("Statistics") else "failed")
                    self.checkpoint_cache[path] = signature, outcomes
                    self.outcomes.update(outcomes)
                except (ValueError, KeyError, OSError):
                    print(f"!! Ignoring unreadable checkpoint: {path}", flush=True)
        for key in self.retry_pending:
            self.outcomes.pop(key, None)

    def prepare_retry(self, rows=None):
        keys = {(row["engine"], row["id"]) for row in rows} if rows is not None else set(self.cells)
        self.retry_pending.update(key for key in keys if self.outcomes.get(key) != "ok")
        for key in self.retry_pending:
            self.outcomes.pop(key, None)

    def pending(self, engine, suite):
        return [row for row in self.plan if row["engine"] == engine and row["suite"] == suite
                and self.outcomes.get((engine, row["id"])) != "ok"]

    def event(self, event):
        key = self.engine, event.get("id")
        if key not in self.cells:
            return
        if event.get("event") == "start":
            self.started[key] = time.monotonic()
        elif event.get("event") == "complete":
            self.retry_pending.discard(key)
            self.outcomes[key] = event.get("status", "failed")
            start = self.started.pop(key, None)
            if start is not None:
                self.samples.setdefault(self.engine, []).append(time.monotonic() - start)
            with (self.directory / "events.jsonl").open("a", encoding="utf-8") as output:
                output.write(json.dumps({**event, "engine": self.engine, "time": time.time(),
                                        "elapsed_seconds": time.monotonic() - start if start else None}) + "\n")
        self.progress()

    def progress(self, force=False, status="running"):
        now = time.monotonic()
        if not force and now - self.last_progress < self.interval:
            return
        self.last_progress = now
        done = sum(self.outcomes.get(key) == "ok" for key in self.cells)
        failed = sum(self.outcomes.get(key) == "failed" for key in self.cells)
        left = len(self.cells) - done - failed
        timings = [t for values in self.samples.values() for t in values]
        estimate = None
        if timings:
            fallback = sum(timings) / len(timings)
            means = {engine: sum(values) / len(values) for engine, values in self.samples.items() if values}
            estimate = sum(means.get(key[0], fallback) for key in self.cells if key not in self.outcomes)
        eta = f"~{duration(estimate)}" if estimate is not None else "estimating"
        if left == 0:
            eta = "0m 00s"
        text = (f"{done + failed:,}/{len(self.cells):,} matrix cases | {left:,} left | "
                f"{failed:,} failed | ETA {eta} | {self.stage}")
        pending_sheets = sum(value != "complete" for name, value in self.stage_state.items()
                            if name.startswith("subsystem:"))
        if pending_sheets:
            text += f" | +{pending_sheets} subsystem stages (ETA separate)"
        if status != "running":
            text = status.upper() + " | " + text
        if not self.cells and self.stage != "finished":
            text = f"Discovering total | {self.stage} | elapsed {duration(now - self.start_time)}"
        print(f"[benchmark] {text}", flush=True)
        if self.title:
            if os.name == "nt":
                ctypes.windll.kernel32.SetConsoleTitleW("Benchmark " + text)
            elif sys.stdout.isatty():
                print(f"\033]0;Benchmark {text}\007", end="", flush=True)
        atomic_json(self.state_path, {"schema_version": 1, "status": status,
                    "stage": self.stage, "stages": self.stage_state,
                    "total": len(self.cells), "completed": done, "failed": failed,
                    "remaining": left, "eta_seconds": estimate,
                    "elapsed_seconds": self.elapsed_before + now - self.start_time, "updated_at": time.time()})

    def run(self, cmd, cwd=None, check=False, env=None, timeout=None):
        """Drain child output continuously while owning progress even when it is silent."""
        cmd = [str(c) for c in cmd]
        timeout = timeout or self.process_timeout
        print(f"\n$ {subprocess.list2cmdline(cmd)}", flush=True)
        logdir = self.directory / "logs"
        logdir.mkdir(exist_ok=True)
        label = re.sub(r"[^\w.-]", "_", self.stage)
        logfile = logdir / f"{label}.log"
        updates = queue.Queue()
        kwargs = {"start_new_session": True} if os.name != "nt" else {
            "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                errors="replace", bufsize=1, **kwargs)
        guard = guard_process(proc)
        try:
            guard.__enter__()
        except BaseException:
            proc.stdout.close()
            raise
        def pump():
            try:
                for line in proc.stdout:
                    updates.put(line)
            finally:
                updates.put(None)
        thread = threading.Thread(target=pump, daemon=True)
        thread.start()
        launched = time.monotonic()
        try:
            with logfile.open("a", encoding="utf-8") as log:
                log.write(f"\n$ {subprocess.list2cmdline(cmd)}\n")
                while True:
                    if timeout and time.monotonic() - launched > timeout:
                        raise TimeoutError(f"{self.stage} exceeded {timeout}s; see {logfile}")
                    try:
                        line = updates.get(timeout=min(1, self.interval))
                    except queue.Empty:
                        if proc.poll() is not None:
                            # A descendant may still own stdout after the driver exits.
                            # Close the Windows job so draining output cannot hang forever.
                            guard.__exit__(None, None, None)
                            if os.name != "nt":
                                try:
                                    os.killpg(proc.pid, signal.SIGKILL)
                                except ProcessLookupError:
                                    pass
                        self.progress()
                        continue
                    if line is None:
                        break
                    log.write(line)
                    log.flush()
                    if line.startswith("@@BENCHMARK "):
                        try:
                            self.event(json.loads(line[len("@@BENCHMARK "):]))
                        except ValueError:
                            pass
                    elif self.verbose:
                        # Child OSC titles would replace global counts with suite-local counts.
                        print(re.sub(r"\x1b\][^\x07]*(?:\x07|\x1b\\)", "", line), end="", flush=True)
                    self.progress()
            proc.wait()
        except BaseException:
            terminate_tree(proc)
            thread.join(timeout=5)
            self.progress(force=True, status="interrupted")
            raise
        finally:
            guard.__exit__(None, None, None)
            proc.stdout.close()
        result = subprocess.CompletedProcess(cmd, proc.returncode)
        if proc.returncode:
            print(f"!! {self.stage} exited {proc.returncode}; log: {logfile}", flush=True)
            if check:
                raise subprocess.CalledProcessError(proc.returncode, cmd)
        self.progress(force=True)
        return result


def terminate_tree(proc):
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
    proc.wait(timeout=10)
