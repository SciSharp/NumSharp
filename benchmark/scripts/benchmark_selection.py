"""Select exact official benchmark cells from saved performance reports.

Reports are immutable inputs. Combined reports are evaluated per backend profile rather
than through their effective fastest timing; a fast OpenBLAS result must not hide a
managed regression. Subsystem scenarios cannot select a different official workload.
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from numpy_checkpoint import validate_checkpoint


_SPEC = importlib.util.spec_from_file_location(
    "_selection_profile_merge", Path(__file__).with_name("merge-backend-profiles.py"))
assert _SPEC and _SPEC.loader
_MERGE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MERGE)

_UNAVAILABLE = {"missing_backend", "not_supported"}
_CREDIBLE = {"faster", "close", "slower", "much_slower"}
_SUITES = {"linearalgebra": "linalg", "broadcasting": "broadcast", "fourier": "fft",
           "ndarrayapi": "ndarray", "apisurface": "api"}


def normalize_operation(operation: str) -> str:
    # The canonical merge handles all NumPy dtype suffixes; Char is a NumSharp-only
    # dtype and needs its own suffix removal before invoking the same normalizer.
    return _MERGE.normalize_operation(re.sub(r"\s*\(char\)\s*$", "", operation, flags=re.I))


def _key(row: dict[str, Any], engine: str) -> tuple:
    suite = str(row.get("suite", "")).lower()
    return (engine, _SUITES.get(suite, suite),
            normalize_operation(str(row.get("operation") or row.get("name") or "")),
            str(row.get("dtype", "")).lower(), int(row.get("n") or 0),
            str(row.get("scenario") or ""))


def _checkpoint_problems(directory: Path) -> list[dict]:
    """Recover failures and never-finished cells from new-format runs, without timing reuse."""
    plan_path = directory / "plan.json"
    if not plan_path.exists():
        return []
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
        if not isinstance(plan, list):
            raise ValueError("expected an array")
    except (ValueError, OSError) as error:
        raise ValueError(f"Cannot read benchmark plan {plan_path}: {error}") from error
    outcomes = {}
    for engine, folder in (("numpy", "numpy-checkpoints"), ("managed", "csharp"),
                           ("openblas", "csharp-openblas")):
        for path in (directory / folder).glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8-sig"))
                if engine == "numpy":
                    if payload.get("status") != "ok" or validate_checkpoint(payload):
                        outcomes[(engine, payload["id"])] = payload.get("status")
                else:
                    for row in payload.get("Benchmarks", []):
                        if row.get("CaseId"):
                            outcomes[(engine, row["CaseId"])] = row.get("CheckpointStatus", "ok" if row.get("Statistics") else "failed")
            except (ValueError, OSError, KeyError, TypeError):
                # An unreadable checkpoint cannot establish successful completion.
                continue
    problems = []
    for case in plan:
        outcome = outcomes.get((case["engine"], case["id"]))
        if outcome != "ok":
            problems.append({**case, "status": "failed" if outcome == "failed" else "no_data",
                             "availability": "failed" if outcome == "failed" else "not_measured",
                             "numsharp_ms": None, "numpy_ms": None, "checkpoint_source": True})
    return problems


def _load(source: Path) -> tuple[Path, list[dict], dict]:
    path = Path(source)
    checkpoints = []
    if path.is_dir():
        checkpoints = _checkpoint_problems(path)
        path = path / "benchmark-report.json"
    if not path.is_file():
        if Path(source).is_dir() and (Path(source) / "plan.json").exists():
            path = Path(source) / "plan.json"
            payload = {"rows": [], "metadata": {"_checkpoint_only": True}}
        else:
            raise ValueError(f"No benchmark report at {path}; select a completed report JSON or run directory")
    else:
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (ValueError, OSError) as error:
            raise ValueError(f"Cannot read benchmark report {path}: {error}") from error
    rows = payload.get("rows") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} is not a merged benchmark report (expected a rows array)")
    metadata = dict(payload.get("metadata") or {}) if isinstance(payload, dict) else {}
    config_path = path.parent / "run-config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8-sig"))
            if config.get("depth"):
                metadata.setdefault("benchmark_depth", config["depth"])
            provenance = config.get("provenance") or {}
            for name, original in {"numpy_version": "numpy", "cpu": "processor", "host": "machine",
                                   "os": "platform", "python_version": "python",
                                   "dotnet_version": "dotnet"}.items():
                if provenance.get(original):
                    metadata.setdefault(name, provenance[original])
            environment = provenance.get("environment") or {}
            if environment:
                metadata.setdefault("benchmark_environment", environment)
        except (ValueError, OSError) as error:
            raise ValueError(f"Cannot read run configuration {config_path}: {error}") from error
    default_profile = (payload.get("profile") if isinstance(payload, dict) else None)
    default_profile = default_profile or ("openblas" if ".openblas" in path.name else "managed")
    measurements = []
    for row in rows:
        if isinstance(row.get("profiles"), dict):
            profiles = row["profiles"]
        elif isinstance(row.get("result"), dict):
            result = row["result"]
            profiles = {result.get("profile") or default_profile: result}
        else:
            profiles = {row.get("profile") or default_profile: row}
        for engine, result in profiles.items():
            if engine not in {"managed", "openblas"} or not isinstance(result, dict):
                continue
            measurement = {**row, **result, "engine": engine}
            if not measurement.get("status"):
                _, _, measurement["status"] = _MERGE.performance_status(
                    _MERGE.number(row.get("numpy_ms")),
                    _MERGE.number(result.get("numsharp_ms")), str(row.get("operation", "")))
            measurements.append(measurement)
    # Completed report rows can omit failed NumPy cells and NumSharp-only dtypes.
    # Checkpoints supply their exact planned identities, and pending checkpoints
    # supersede stale rendered results when an interrupted resume did not re-merge.
    if checkpoints:
        problems = {_key(row, row["engine"]): row for row in checkpoints}
        measurements = [row for row in measurements if _key(row, row["engine"]) not in problems]
        measurements.extend(problems.values())
    return path, measurements, metadata


def _comparable_metadata(current: dict, baseline: dict) -> list[str]:
    """Reject known incompatible methodology/hosts; retain warnings for old snapshots."""
    aliases = {
        "depth": ("benchmark_depth", "depth"),
        "timing basis": ("timing_basis", "timing_statistic"),
        "NumPy version": ("numpy_version",),
        "CPU": ("cpu", "cpu_name"),
        "host": ("host",),
        "OS": ("os", "platform"),
        "architecture": ("architecture", "machine"),
        "Python version": ("python_version",),
        ".NET runtime": ("dotnet_version", "dotnet_runtime"),
        "large workload size": ("memory_heavy_large_workload_n",),
        "large tier": ("memory_heavy_large_tier_label",),
        "BLAS threads": ("openblas_threads", "blas_threads"),
        "BLAS kernel": ("openblas_core", "openblas_core_type"),
        "benchmark environment": ("benchmark_environment",),
    }
    missing, mismatches = [], []
    for label, names in aliases.items():
        a = next((current[name] for name in names if current.get(name) not in (None, "")), None)
        b = next((baseline[name] for name in names if baseline.get(name) not in (None, "")), None)
        if a is None or b is None:
            missing.append(label)
        elif str(a) != str(b):
            mismatches.append(f"{label}: source={a!r}, baseline={b!r}")
    if mismatches:
        raise ValueError("Incompatible regression baseline: " + "; ".join(mismatches))
    return (["Older or incomplete provenance: cannot verify " + ", ".join(missing)]
            if missing else [])


def _credible(row: dict) -> bool:
    return (row.get("availability", "available") == "available"
            and row.get("status") in _CREDIBLE
            and (_MERGE.number(row.get("numsharp_ms")) or 0) >= _MERGE.WORK_FLOOR_MS)


def _description(row: dict) -> dict:
    return {name: row.get(name, "") for name in
            ("engine", "suite", "operation", "dtype", "n", "scenario", "status")}


def select_plan(plan: list[dict], source: Path, mode: str,
                baseline: Path | None = None, regression_percent: float = 10) -> tuple[list[dict], dict]:
    """Return selected plan cells and serializable reasons/unmatched-source diagnostics.

    ``failed`` selects workload failures. ``bad`` adds unmeasured comparisons and
    credible slower/much_slower results, excluding known backend unavailability.
    ``degraded`` compares each credible profile timing against the same baseline
    cell, using a strict percentage threshold. NumPy counterparts accompany every
    selected NumSharp cell when present in the supplied execution plan.
    """
    if mode not in {"failed", "bad", "degraded"}:
        raise ValueError(f"Unknown rerun mode: {mode}")
    if not math.isfinite(regression_percent) or regression_percent < 0:
        raise ValueError("Regression percentage must be finite and nonnegative")
    if mode == "degraded" and baseline is None:
        raise ValueError("Degraded reruns require a baseline report")
    if mode != "degraded" and baseline is not None:
        raise ValueError("A baseline applies only to degraded reruns")
    source_path, measurements, metadata = _load(source)
    if mode == "degraded" and metadata.get("_checkpoint_only"):
        raise ValueError("Degraded reruns require a merged source report with comparable timings")
    diagnostic = {"mode": mode, "source": str(source_path.resolve()), "baseline": None,
                  "regression_percent": regression_percent, "warnings": [], "reasons": {},
                  "unmatched": [], "noncomparable": [], "examined_results": len(measurements),
                  "eligible_results": 0}
    baseline_map = {}
    if baseline is not None:
        baseline_path, baseline_rows, baseline_metadata = _load(baseline)
        if baseline_metadata.get("_checkpoint_only"):
            raise ValueError("Degraded reruns require a merged baseline report with comparable timings")
        diagnostic["baseline"] = str(baseline_path.resolve())
        diagnostic["warnings"] = _comparable_metadata(metadata, baseline_metadata)
        for row in baseline_rows:
            key = _key(row, row["engine"])
            if key in baseline_map:
                raise ValueError(f"Ambiguous duplicate baseline cell: {_description(row)}")
            baseline_map[key] = row

    plan_map = defaultdict(list)
    for cell in plan:
        plan_map[_key(cell, cell["engine"])].append(cell)
    selected_ids, seen_source = set(), set()
    for row in measurements:
        key = _key(row, row["engine"])
        if key in seen_source:
            raise ValueError(f"Ambiguous duplicate source cell: {_description(row)}")
        seen_source.add(key)
        availability, status = row.get("availability", "available"), row.get("status")
        if availability in _UNAVAILABLE:
            continue
        reason = {**_description(row), "reason": status}
        if mode == "degraded":
            old = baseline_map.get(key)
            if not _credible(row):
                continue
            if old is None or not _credible(old):
                diagnostic["noncomparable"].append({**reason, "reason": "missing or noncredible baseline"})
                continue
            if (row.get("actual_backend") and old.get("actual_backend")
                    and row["actual_backend"] != old["actual_backend"]):
                diagnostic["noncomparable"].append({**reason, "reason": "actual backend changed"})
                continue
            current_ms, baseline_ms = float(row["numsharp_ms"]), float(old["numsharp_ms"])
            increase = (current_ms / baseline_ms - 1) * 100
            if current_ms <= baseline_ms * (1 + regression_percent / 100):
                continue
            reason.update(reason="degraded", current_ms=current_ms, baseline_ms=baseline_ms,
                          increase_percent=increase)
        elif status != "failed" and availability != "failed":
            if mode != "bad" or not (status == "no_data" or
                                      (status in {"slower", "much_slower"} and _credible(row))):
                continue
        diagnostic["eligible_results"] += 1
        matches = plan_map.get(key, [])
        if len(matches) != 1:
            diagnostic["unmatched"].append({**reason, "reason":
                "ambiguous official plan cell" if matches else "no exact official plan cell"})
            continue
        cell = matches[0]
        selected_ids.add((cell["engine"], cell["id"]))
        diagnostic["reasons"].setdefault(cell["id"], []).append(reason)
        if cell["engine"] == "numpy":
            continue
        numpy_key = ("numpy", *key[1:])
        counterparts = plan_map.get(numpy_key, [])
        if len(counterparts) == 1:
            counterpart = counterparts[0]
            selected_ids.add((counterpart["engine"], counterpart["id"]))
            diagnostic["reasons"].setdefault(counterpart["id"], []).append(
                {"reason": "NumPy counterpart", "for_cell": cell["id"]})
        elif len(counterparts) > 1:
            diagnostic["unmatched"].append({**reason, "reason": "ambiguous NumPy counterpart"})
    selected = [cell for cell in plan if (cell["engine"], cell["id"]) in selected_ids]
    diagnostic["selected_count"] = len(selected)
    diagnostic["selected_by_engine"] = dict(Counter(cell["engine"] for cell in selected))
    return selected, diagnostic


def reference_rows_for_plan(rows: list[dict], plan: list[dict]) -> list[dict]:
    """Keep only the imported NumPy counterparts of the selected execution plan."""
    keys = {_key(case, "numpy") for case in plan}
    return [row for row in rows if _key(row, "numpy") in keys]
