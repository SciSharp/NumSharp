"""Durable per-case storage for the NumPy benchmark process.

One JSON file per case keeps an interrupted suite recoverable without parsing stdout.
The hash is only a Windows-safe filename; the readable identity remains in the JSON.
"""

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path


RESULT_FIELDS = {
    "name", "category", "suite", "dtype", "n", "mean_ms", "stddev_ms", "min_ms", "max_ms",
    "iterations", "ops_per_sec", "allocated_mb",
    # Symmetric-Tukey (OutlierMode.RemoveAll) provenance carried on every BenchmarkResult; the merge
    # never reads it, but the persisted result dict carries it, so the validator must allow it.
    "outliers_removed",
}


def validate_checkpoint(payload, case=None, settings=None):
    """Return whether an ok/failed checkpoint is structurally valid and matches its case.

    This pure validator is shared with the parent runner, which reads files once while
    recovering progress/materializing output. A valid failed checkpoint is accounted work,
    but only status == 'ok' is reusable measurement evidence.
    """
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        return False
    saved_case = payload.get("case")
    if not isinstance(saved_case, dict) or set(saved_case) != {"id", "suite", "operation", "dtype", "n"}:
        return False
    if not all(isinstance(saved_case[key], str) and saved_case[key] for key in ("id", "suite", "operation", "dtype")):
        return False
    if type(saved_case["n"]) is not int or saved_case["n"] <= 0:
        return False
    identity = case_id(saved_case["suite"], saved_case["operation"], saved_case["dtype"], saved_case["n"])
    if saved_case["id"] != identity or payload.get("id") != identity:
        return False
    if case is not None and saved_case != case:
        return False
    if not isinstance(payload.get("settings"), dict) or (settings is not None and payload["settings"] != settings):
        return False
    if payload.get("status") == "failed":
        return payload.get("result") is None and isinstance(payload.get("error"), str) and bool(payload["error"])
    if payload.get("status") != "ok" or payload.get("error") is not None:
        return False
    result = payload.get("result")
    if not isinstance(result, dict) or set(result) != RESULT_FIELDS:
        return False
    if result["dtype"] != saved_case["dtype"] or result["n"] != saved_case["n"]:
        return False
    if result["name"] not in (saved_case["operation"], f"{saved_case['operation']} ({saved_case['dtype']})"):
        return False
    if not isinstance(result["suite"], str) or not isinstance(result["category"], str):
        return False
    suite = {"Broadcasting": "broadcast", "LinearAlgebra": "linalg", "Fourier": "fft", "ApiSurface": "api"}.get(
        result["suite"], result["suite"].lower())
    if suite != saved_case["suite"]:
        return False
    for name in ("mean_ms", "stddev_ms", "min_ms", "max_ms", "ops_per_sec", "allocated_mb"):
        if type(result[name]) not in (int, float) or not math.isfinite(result[name]) or result[name] < 0:
            return False
    return (type(result["iterations"]) is int and result["iterations"] >= 1
            and result["min_ms"] <= result["mean_ms"] <= result["max_ms"])


def case_id(suite, operation, dtype, n):
    return json.dumps(["numpy", suite, operation, dtype, n], separators=(",", ":"))


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class CheckpointStore:
    def __init__(self, directory, settings):
        self.directory = Path(directory) if directory else None
        self.settings = settings

    def path(self, identity):
        if self.directory is None:
            return None
        return self.directory / (hashlib.sha256(identity.encode("utf-8")).hexdigest() + ".json")

    def completed(self, case):
        path = self.path(case["id"])
        if path is None or not path.exists():
            return None
        # A corrupt or partial file is not completed work. Atomic writes avoid these in
        # normal operation; treating them as pending also recovers older/manual files.
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
            if not validate_checkpoint(saved, case) or saved.get("status") != "ok":
                return None
            if saved.get("settings") != self.settings:
                raise ValueError(f"Checkpoint settings differ for {case['id']}; use a new checkpoint directory")
            return saved["result"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            return None

    def save(self, case, status, result=None, error=None):
        path = self.path(case["id"])
        if path is not None:
            atomic_json(path, {
                "schema_version": 1, "id": case["id"], "case": case,
                "status": status, "settings": self.settings,
                "result": result, "error": error,
            })
        return str(path) if path is not None else None
