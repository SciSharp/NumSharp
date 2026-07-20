#!/usr/bin/env python3
"""Generate the deterministic NumPy <-> NumSharp public API coverage artifact."""

from __future__ import annotations

import argparse
import csv
import inspect
import io
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
PINNED_NUMPY_VERSION = "2.4.2"
GENERATOR_VERSION = "1.1.0"
OUTPUT_FILES = ("coverage.json", "coverage.csv", "summary.md", "manifest.json")
NUMSHARP_SOURCE_BASE_URL = "https://github.com/SciSharp/NumSharp/blob/master/"

CALLABLE_KINDS = {"function", "ufunc", "callable", "method"}
VALID_SUPPORT = {"declared", "partial", "unsupported", "missing", "extension"}

CREATION = {
    "arange", "array", "asanyarray", "asarray", "asarray_chkfinite", "ascontiguousarray",
    "asfortranarray", "asmatrix", "copy", "empty", "empty_like", "eye", "frombuffer",
    "from_dlpack", "fromfile", "fromfunction", "fromiter", "fromregex", "fromstring", "full",
    "full_like", "genfromtxt", "identity", "loadtxt", "linspace", "logspace", "meshgrid",
    "mgrid", "ogrid", "ones", "ones_like", "require", "tri", "tril", "triu", "vander",
    "zeros", "zeros_like"
}
MANIPULATION = {
    "append", "apply_along_axis", "apply_over_axes", "array_split", "atleast_1d", "atleast_2d",
    "atleast_3d", "block", "broadcast_arrays", "broadcast_to", "column_stack", "concat",
    "concatenate", "delete", "dsplit", "dstack", "expand_dims", "flip", "fliplr", "flipud",
    "hsplit", "hstack", "insert", "matrix_transpose", "moveaxis", "pad", "permute_dims",
    "ravel", "repeat", "reshape", "resize", "roll", "rollaxis", "rot90", "row_stack",
    "split", "squeeze", "stack", "swapaxes", "tile", "transpose", "trim_zeros", "unstack",
    "vsplit", "vstack"
}
REDUCTIONS = {
    "all", "amax", "amin", "any", "argmax", "argmin", "average", "count_nonzero", "cumprod",
    "cumsum", "cumulative_prod", "cumulative_sum", "max", "mean", "median", "min", "nanargmax",
    "nanargmin", "nancumprod", "nancumsum", "nanmax", "nanmean", "nanmedian", "nanmin",
    "nanpercentile", "nanprod", "nanquantile", "nanstd", "nansum", "nanvar", "percentile",
    "prod", "ptp", "quantile", "std", "sum", "var"
}
LOGIC = {
    "allclose", "array_equal", "array_equiv", "equal", "fmax", "fmin", "greater", "greater_equal",
    "isclose", "iscomplex", "iscomplexobj", "isfinite", "isfortran", "isinf", "isnan", "isneginf",
    "isposinf", "isreal", "isrealobj", "isscalar", "less", "less_equal", "logical_and",
    "logical_not", "logical_or", "logical_xor", "maximum", "minimum", "not_equal"
}
DTYPE = {
    "can_cast", "common_type", "dtype", "finfo", "find_common_type", "iinfo", "issubdtype",
    "min_scalar_type", "mintypecode", "promote_types", "result_type", "sctypeDict", "typecodes"
}
SELECTION = {
    "choose", "compress", "diag_indices", "diag_indices_from", "extract", "indices", "ix_", "mask_indices",
    "place", "put", "put_along_axis", "ravel_multi_index", "select", "take", "take_along_axis",
    "tril_indices", "tril_indices_from", "triu_indices", "triu_indices_from", "unravel_index", "where"
}
SORTING = {"argpartition", "argsort", "argwhere", "flatnonzero", "lexsort", "nonzero", "partition", "searchsorted", "sort"}
IO = {"fromfile", "fromregex", "fromstring", "genfromtxt", "load", "loads", "loadtxt", "save", "savetxt", "savez", "savez_compressed"}
MATH = {
    "abs", "absolute", "acos", "acosh", "add", "arccos", "arccosh", "arcsin", "arcsinh", "arctan",
    "arctan2", "arctanh", "asin", "asinh", "atan", "atan2", "atanh", "bitwise_and", "bitwise_count",
    "bitwise_invert", "bitwise_left_shift", "bitwise_not", "bitwise_or", "bitwise_right_shift", "bitwise_xor",
    "cbrt", "ceil", "clip", "conj", "conjugate", "convolve", "copysign", "cos", "cosh", "cross",
    "deg2rad", "degrees", "diff", "divide", "divmod", "dot", "ediff1d", "exp", "exp2", "expm1",
    "fabs", "fix", "floor", "floor_divide", "fmod", "frexp", "gcd", "heaviside", "hypot", "inner",
    "invert", "kron", "lcm", "ldexp", "left_shift", "log", "log10", "log1p", "log2", "logaddexp",
    "logaddexp2", "matmul", "mod", "modf", "multiply", "negative", "nextafter", "outer", "positive",
    "pow", "power", "rad2deg", "radians", "reciprocal", "remainder", "right_shift", "rint", "round",
    "sign", "signbit", "sin", "sinc", "sinh", "spacing", "sqrt", "square", "subtract", "tan", "tanh",
    "trace", "true_divide", "trunc"
}

# NumPy exposes several coherent routine families from the top-level namespace that
# do not fit the broad creation/manipulation/math buckets above. Keep these explicit
# so the dashboard's capability map remains useful instead of hiding gaps in "Other".
CATEGORY_OVERRIDES = {
    **dict.fromkeys({"copyto", "iterable", "may_share_memory", "nested_iters", "shares_memory"}, "Array metadata & memory"),
    **dict.fromkeys({"busday_count", "busday_offset", "datetime_as_string", "datetime_data", "is_busday", "isnat"}, "Date & time"),
    **dict.fromkeys({"getbufsize", "geterr", "geterrcall", "nan_to_num", "setbufsize", "seterr", "seterrcall"}, "Floating-point handling"),
    **dict.fromkeys({"bmat", "diag", "diagflat", "diagonal", "einsum", "einsum_path", "fill_diagonal", "matvec", "tensordot", "vdot", "vecdot", "vecmat"}, "Linear algebra"),
    **dict.fromkeys({"poly", "polyadd", "polyder", "polydiv", "polyfit", "polyint", "polymul", "polysub", "polyval", "roots"}, "Polynomials"),
    **dict.fromkeys({"get_include", "info", "show_config", "show_runtime", "test"}, "Runtime & diagnostics"),
    **dict.fromkeys({"intersect1d", "isin", "setdiff1d", "setxor1d", "union1d", "unique", "unique_all", "unique_counts", "unique_inverse", "unique_values"}, "Set operations"),
    **dict.fromkeys({"bincount", "corrcoef", "correlate", "cov", "digitize", "histogram", "histogram2d", "histogram_bin_edges", "histogramdd"}, "Statistics & histograms"),
    **dict.fromkeys({"array2string", "array_repr", "array_str", "base_repr", "binary_repr", "format_float_positional", "format_float_scientific", "get_printoptions", "printoptions", "set_printoptions", "typename"}, "Text & formatting"),
    **dict.fromkeys({"bartlett", "blackman", "hamming", "hanning", "kaiser"}, "Window functions"),
}

MATH.update({"angle", "around", "float_power", "frompyfunc", "gradient", "i0", "imag", "interp", "packbits", "piecewise", "real", "real_if_close", "trapezoid", "unpackbits", "unwrap"})
CREATION.add("geomspace")
DTYPE.update({"astype", "isdtype"})
MANIPULATION.update({"broadcast_shapes", "ndim", "shape", "size"})
SELECTION.add("putmask")
SORTING.add("sort_complex")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "coverage" / "generated")
    parser.add_argument("--overrides", type=Path, default=ROOT / "coverage" / "overrides.json")
    parser.add_argument("--check", action="store_true", help="Fail if checked-in outputs differ; do not write files.")
    return parser.parse_args()


def load_numpy() -> Any:
    try:
        import numpy as np  # type: ignore
    except ImportError as error:
        raise SystemExit(f"NumPy {PINNED_NUMPY_VERSION} is required: {error}") from error
    if np.__version__ != PINNED_NUMPY_VERSION:
        raise SystemExit(f"Expected NumPy {PINNED_NUMPY_VERSION}, found {np.__version__}.")
    return np


def load_numsharp_inventory() -> dict[str, Any]:
    project = ROOT / "tools" / "NumSharp.ApiInventory" / "NumSharp.ApiInventory.csproj"
    command = ["dotnet", "run", "--project", str(project), "--configuration", "Release"]
    completed = subprocess.run(command, cwd=ROOT, check=False, text=True, capture_output=True)
    if completed.returncode:
        sys.stderr.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        raise SystemExit("Failed to reflect the NumSharp public API.")
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise SystemExit(f"NumSharp inventory emitted invalid JSON: {error}") from error


def load_overrides(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise SystemExit(f"Unsupported override schema in {path}.")
    return data


def compact_text(value: str, limit: int = 420) -> str:
    value = re.sub(r" at 0x[0-9a-fA-F]+", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "…"


def numpy_signature(obj: Any, fallback_name: str) -> str:
    try:
        return compact_text(str(inspect.signature(obj)))
    except (TypeError, ValueError):
        pass
    doc = inspect.getdoc(obj) or ""
    first_line = doc.splitlines()[0].strip() if doc else ""
    if first_line and (fallback_name in first_line or first_line.startswith("(")):
        return compact_text(first_line)
    nin = getattr(obj, "nin", None)
    nout = getattr(obj, "nout", None)
    if isinstance(nin, int):
        return f"ufunc(nin={nin}, nout={nout})"
    return "Signature unavailable from runtime introspection"


def numpy_kind(np: Any, obj: Any) -> str:
    if inspect.ismodule(obj):
        return "module"
    if isinstance(obj, np.ufunc):
        return "ufunc"
    if inspect.isclass(obj):
        return "class"
    if callable(obj):
        return "function"
    return "constant"


def documentation_url(surface: str, name: str, kind: str) -> str:
    prefix = {
        "np": "numpy",
        "ndarray": "numpy.ndarray",
        "random": "numpy.random",
        "linalg": "numpy.linalg",
        "fft": "numpy.fft",
    }[surface]
    canonical_aliases = {
        "numpy.abs": "numpy.absolute",
        "numpy.bitwise_not": "numpy.invert",
        "numpy.row_stack": "numpy.vstack",
    }
    api_id = f"{prefix}.{name}"
    api_id = canonical_aliases.get(api_id, api_id)

    if surface == "random":
        if kind not in CALLABLE_KINDS:
            return ""
        if name == "default_rng":
            return "https://numpy.org/doc/stable/reference/random/generator.html#numpy.random.default_rng"
        return f"https://numpy.org/doc/stable/reference/random/generated/{api_id}.html"
    if surface == "np" and name == "test":
        return "https://numpy.org/doc/stable/reference/testing.html"
    if surface == "np" and kind not in CALLABLE_KINDS:
        return ""
    return f"https://numpy.org/doc/stable/reference/generated/{api_id}.html"


class SourceLocator:
    """Locate public member declarations without requiring compiler-specific PDB paths."""

    TYPE_PATTERNS = {
        "np": re.compile(r"\bclass\s+np\b"),
        "ndarray": re.compile(r"\bclass\s+NDArray(?:\s|<)"),
        "random": re.compile(r"\bclass\s+NumPyRandom\b"),
    }

    def __init__(self) -> None:
        self.files: dict[str, list[tuple[Path, str]]] = {surface: [] for surface in self.TYPE_PATTERNS}
        source_root = ROOT / "src" / "NumSharp.Core"
        for path in source_root.rglob("*.cs"):
            text = path.read_text(encoding="utf-8-sig")
            for surface, pattern in self.TYPE_PATTERNS.items():
                if pattern.search(text):
                    self.files[surface].append((path, text))

    def locate(self, surface: str, name: str, kind: str) -> list[str]:
        escaped = re.escape(name)
        if kind == "method":
            declaration = re.compile(rf"\bpublic\b[^\n;{{}}]*\b{escaped}\s*(?:<[^\n>]+>)?\s*\(")
        elif kind == "property" and name == "Item":
            declaration = re.compile(r"\bpublic\b[^\n;{}]*\bthis\s*\[")
        elif kind == "property":
            declaration = re.compile(rf"\bpublic\b[^\n;{{}}]*\b{escaped}\b\s*(?:\{{|=>)")
        else:
            declaration = re.compile(rf"\bpublic\b[^\n;{{}}]*\b{escaped}\b\s*(?:=|;|,)")
        candidates = [path for path, text in self.files.get(surface, []) if declaration.search(text)]

        normalized_name = re.sub(r"[^a-z0-9]", "", name.lower())
        candidates.sort(key=lambda path: (
            normalized_name not in re.sub(r"[^a-z0-9]", "", path.stem.lower()),
            "Generics" in path.parts,
            len(path.parts),
            path.as_posix().lower(),
        ))
        return [candidates[0].relative_to(ROOT).as_posix()] if candidates else []

    @staticmethod
    def github_urls(paths: list[str]) -> list[str]:
        return [NUMSHARP_SOURCE_BASE_URL + quote(path, safe="/") for path in paths]


def category_for(surface: str, name: str, kind: str) -> str:
    if surface == "random":
        return "Random"
    if surface == "linalg":
        return "Linear algebra"
    if surface == "fft":
        return "Fourier transforms"
    if surface == "ndarray":
        if kind == "property":
            return "Array attributes"
        if name in REDUCTIONS:
            return "Reductions"
        if name in SORTING:
            return "Sorting & searching"
        if name in MANIPULATION or name in {"astype", "byteswap", "copy", "fill", "flatten", "item", "setfield", "tolist", "tobytes", "tofile", "view"}:
            return "Array methods"
        return "Array methods"
    if kind not in CALLABLE_KINDS:
        return {"module": "Namespaces", "class": "Types", "constant": "Types & constants"}.get(kind, "Other")
    if name in CATEGORY_OVERRIDES:
        return CATEGORY_OVERRIDES[name]
    if name in CREATION:
        return "Array creation"
    if name in MANIPULATION:
        return "Shape manipulation"
    if name in REDUCTIONS:
        return "Reductions"
    if name in LOGIC:
        return "Logic & comparison"
    if name in DTYPE:
        return "Dtype & promotion"
    if name in SELECTION:
        return "Indexing & selection"
    if name in SORTING:
        return "Sorting & searching"
    if name in IO:
        return "Input & output"
    if name in MATH:
        return "Math"
    return "Other"


def member_maps(inventory: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    by_target: dict[str, dict[str, Any]] = {}
    by_surface: dict[str, list[dict[str, Any]]] = {"np": [], "ndarray": [], "random": []}
    source_locator = SourceLocator()
    definitions = (
        ("np", "NumSharp.np", inventory["np"]),
        ("ndarray", "NumSharp.NDArray", inventory["ndArray"]),
        ("random", "NumSharp.NumPyRandom", inventory["random"]),
    )
    for surface, prefix, type_data in definitions:
        for collection in ("methods", "properties", "fields"):
            for member in type_data[collection]:
                source_paths = source_locator.locate(surface, member["name"], member["kind"])
                normalized = {
                    **member,
                    "surface": surface,
                    "target": f"{prefix}.{member['name']}",
                    "sourcePaths": source_paths,
                    "sourceUrls": source_locator.github_urls(source_paths),
                }
                by_target[normalized["target"]] = normalized
                by_surface[surface].append(normalized)
    by_target["NumSharp.NDArray"] = {
        "name": "NDArray", "kind": "class", "signatures": ["NumSharp.NDArray"], "obsolete": False,
        "surface": "ndarray", "target": "NumSharp.NDArray",
        "sourcePaths": ["src/NumSharp.Core/Backends/NDArray.cs"],
        "sourceUrls": [NUMSHARP_SOURCE_BASE_URL + "src/NumSharp.Core/Backends/NDArray.cs"],
    }
    return by_target, by_surface


def public_exports(np: Any) -> list[dict[str, Any]]:
    exports: list[dict[str, Any]] = []

    for name in sorted(set(np.__all__)):
        if not hasattr(np, name):
            continue
        obj = getattr(np, name)
        kind = numpy_kind(np, obj)
        exports.append({
            "id": f"numpy.{name}", "origin": "numpy", "surface": "np", "name": name, "kind": kind,
            "numpy_signature": numpy_signature(obj, name), "documentation_url": documentation_url("np", name, kind),
            "in_default_scope": kind in CALLABLE_KINDS,
        })

    for name in sorted(item for item in dir(np.ndarray) if not item.startswith("_")):
        raw = inspect.getattr_static(np.ndarray, name)
        obj = getattr(np.ndarray, name)
        kind = "method" if callable(obj) else "property"
        exports.append({
            "id": f"numpy.ndarray.{name}", "origin": "numpy", "surface": "ndarray", "name": name, "kind": kind,
            "numpy_signature": numpy_signature(obj if callable(obj) else raw, name),
            "documentation_url": documentation_url("ndarray", name, kind), "in_default_scope": True,
        })

    for surface, module in (("random", np.random), ("linalg", np.linalg), ("fft", np.fft)):
        for name in sorted(set(module.__all__)):
            if not hasattr(module, name):
                continue
            obj = getattr(module, name)
            kind = numpy_kind(np, obj)
            exports.append({
                "id": f"numpy.{surface}.{name}", "origin": "numpy", "surface": surface, "name": name, "kind": kind,
                "numpy_signature": numpy_signature(obj, name), "documentation_url": documentation_url(surface, name, kind),
                "in_default_scope": kind in CALLABLE_KINDS,
            })
    return exports


def direct_target(row: dict[str, Any], targets: dict[str, dict[str, Any]]) -> str | None:
    surface = row["surface"]
    name = row["name"]
    kind = row["kind"]
    if row["id"] == "numpy.ndarray":
        return "NumSharp.NDArray"
    prefix = {"np": "NumSharp.np", "ndarray": "NumSharp.NDArray", "random": "NumSharp.NumPyRandom"}.get(surface)
    if prefix:
        candidate = f"{prefix}.{name}"
        member = targets.get(candidate)
        if member and (kind not in CALLABLE_KINDS or member["kind"] == "method"):
            return candidate
    return None


def auto_alternative(row: dict[str, Any], targets: dict[str, dict[str, Any]]) -> tuple[str | None, str | None]:
    if row["surface"] in {"ndarray", "linalg"} and row["kind"] in CALLABLE_KINDS:
        target = f"NumSharp.np.{row['name']}"
        if target in targets:
            noun = "instance method" if row["surface"] == "ndarray" else "linalg namespace function"
            return target, f"Available through the static NumSharp np API instead of the NumPy {noun}."
    return None, None


def resolve_rows(np: Any, inventory: dict[str, Any], overrides: dict[str, Any]) -> tuple[list[dict[str, Any]], set[str]]:
    targets, surfaces = member_maps(inventory)
    aliases = overrides.get("aliases", {})
    support_overrides = overrides.get("support", {})
    seen_ids: set[str] = set()
    consumed_targets: set[str] = set()
    rows: list[dict[str, Any]] = []

    for export in public_exports(np):
        row_id = export["id"]
        if row_id in seen_ids:
            raise SystemExit(f"Duplicate coverage id: {row_id}")
        seen_ids.add(row_id)
        alias = aliases.get(row_id)
        target = direct_target(export, targets)
        availability = "exact" if target else "missing"
        notes: list[str] = []
        if alias and not target:
            target = alias["target"]
            if target not in targets:
                raise SystemExit(f"Alias {row_id} references missing NumSharp target {target}.")
            availability = "alias"
            if alias.get("notes"):
                notes.append(alias["notes"])
        elif not target:
            target, automatic_note = auto_alternative(export, targets)
            if target:
                availability = "alias"
                notes.append(automatic_note or "Available on an alternate NumSharp surface.")

        support = "declared" if target else "missing"
        support_override = support_overrides.get(row_id)
        if support_override:
            support = support_override["status"]
            if support not in VALID_SUPPORT - {"missing", "extension"}:
                raise SystemExit(f"Invalid support status for {row_id}: {support}")
            if support_override.get("notes"):
                notes.append(support_override["notes"])
        if target:
            consumed_targets.add(target)
            member = targets[target]
            signatures = member["signatures"]
            obsolete = member.get("obsolete", False)
            source_paths = member.get("sourcePaths", [])
            source_urls = member.get("sourceUrls", [])
        else:
            signatures = []
            obsolete = False
            source_paths = []
            source_urls = []

        display_status = "missing" if not target else support if support in {"partial", "unsupported"} else "available"
        rows.append({
            **export,
            "category": category_for(export["surface"], export["name"], export["kind"]),
            "availability": availability,
            "support": support,
            "status": display_status,
            "numsharp_target": target,
            "numsharp_signatures": signatures,
            "numsharp_obsolete": obsolete,
            "numsharp_source_paths": source_paths,
            "numsharp_source_urls": source_urls,
            "notes": " ".join(dict.fromkeys(notes)),
        })

    unknown_aliases = set(aliases) - seen_ids
    unknown_support = set(support_overrides) - seen_ids
    if unknown_aliases or unknown_support:
        unknown = ", ".join(sorted(unknown_aliases | unknown_support))
        raise SystemExit(f"Overrides reference NumPy exports that were not discovered: {unknown}")

    for surface, members in surfaces.items():
        for member in members:
            name = member["name"]
            target = member["target"]
            if target in consumed_targets or name.startswith("_"):
                continue
            row_id = f"numsharp.{surface}.{name}"
            if row_id in seen_ids:
                row_id += f".{member['kind']}"
            seen_ids.add(row_id)
            rows.append({
                "id": row_id,
                "origin": "numsharp",
                "surface": surface,
                "name": name,
                "kind": member["kind"],
                "numpy_signature": "",
                "documentation_url": "",
                "in_default_scope": False,
                "category": "NumSharp-only APIs",
                "availability": "extension",
                "support": "extension",
                "status": "extension",
                "numsharp_target": target,
                "numsharp_signatures": member["signatures"],
                "numsharp_obsolete": member.get("obsolete", False),
                "numsharp_source_paths": member.get("sourcePaths", []),
                "numsharp_source_urls": member.get("sourceUrls", []),
                "notes": "NumSharp-only public API with no matching export on the compared NumPy surface.",
            })

    rows.sort(key=lambda row: (row["origin"] != "numpy", row["surface"], row["name"].lower(), row["id"]))
    missing_extension_sources = [
        row["id"] for row in rows
        if row["origin"] == "numsharp" and not row["numsharp_source_urls"]
    ]
    if missing_extension_sources:
        raise SystemExit("NumSharp-only APIs without a source link: " + ", ".join(missing_extension_sources))
    return rows, consumed_targets


def status_counts(rows: list[dict[str, Any]]) -> dict[str, int | float]:
    statuses = Counter(row["status"] for row in rows)
    availability = Counter(row["availability"] for row in rows)
    total = len(rows)
    available = statuses["available"]
    addressed = available + statuses["partial"]
    return {
        "total": total,
        "available": available,
        "partial": statuses["partial"],
        "unsupported": statuses["unsupported"],
        "missing": statuses["missing"],
        "exact": availability["exact"],
        "alias": availability["alias"],
        "coverage_percent": round(available * 100 / total, 1) if total else 0.0,
        "addressed_percent": round(addressed * 100 / total, 1) if total else 0.0,
    }


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    default_rows = [row for row in rows if row["origin"] == "numpy" and row["in_default_scope"]]
    numpy_rows = [row for row in rows if row["origin"] == "numpy"]
    by_surface = {
        surface: status_counts([row for row in default_rows if row["surface"] == surface])
        for surface in sorted({row["surface"] for row in default_rows})
    }
    by_category = {
        category: status_counts([row for row in default_rows if row["category"] == category])
        for category in sorted({row["category"] for row in default_rows})
    }
    return {
        "default_scope": status_counts(default_rows),
        "by_surface": by_surface,
        "by_category": by_category,
        "all_numpy_exports": len(numpy_rows),
        "numsharp_extensions": sum(row["origin"] == "numsharp" for row in rows),
        "catalog_rows": len(rows),
    }


def json_text(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=False) + "\n"


def csv_text(rows: list[dict[str, Any]]) -> str:
    columns = [
        "id", "origin", "surface", "category", "name", "kind", "in_default_scope", "status", "availability",
        "support", "numpy_signature", "numsharp_target", "numsharp_signatures", "numsharp_source_paths",
        "numsharp_source_urls", "numsharp_obsolete", "notes", "documentation_url"
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        flat = {key: row.get(key, "") for key in columns}
        flat["numsharp_signatures"] = " | ".join(row["numsharp_signatures"])
        flat["numsharp_source_paths"] = " | ".join(row["numsharp_source_paths"])
        flat["numsharp_source_urls"] = " | ".join(row["numsharp_source_urls"])
        writer.writerow(flat)
    return stream.getvalue()


def markdown_text(summary: dict[str, Any], rows: list[dict[str, Any]], numpy_version: str, assembly_version: str) -> str:
    headline = summary["default_scope"]
    lines = [
        "# NumPy ↔ NumSharp API coverage",
        "",
        f"Compared with NumPy **{numpy_version}** using NumSharp assembly **{assembly_version}**.",
        "",
        f"Headline API availability: **{headline['coverage_percent']:.1f}%** "
        f"({headline['available']} of {headline['total']} default-scope APIs). "
        f"Including partial mappings, **{headline['addressed_percent']:.1f}%** are addressed.",
        "",
        "| Surface | Available | Partial | Unsupported | Missing | Total | Coverage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {"np": "np.*", "ndarray": "ndarray.*", "random": "np.random.*", "linalg": "np.linalg.*", "fft": "np.fft.*"}
    for surface, counts in summary["by_surface"].items():
        lines.append(
            f"| {labels.get(surface, surface)} | {counts['available']} | {counts['partial']} | "
            f"{counts['unsupported']} | {counts['missing']} | {counts['total']} | {counts['coverage_percent']:.1f}% |"
        )
    lines.extend([
        "",
        "> Availability is based on the compiled public API. It is not a blanket behavioral-parity claim; dtype, layout, signature, and edge-case parity require differential tests.",
        "",
        "## Highest-priority gaps",
        "",
        "| API | Surface | Status | Category |",
        "|---|---|---|---|",
    ])
    gaps = [row for row in rows if row["origin"] == "numpy" and row["in_default_scope"] and row["status"] != "available"]
    priority = {"unsupported": 0, "partial": 1, "missing": 2}
    gaps.sort(key=lambda row: (priority.get(row["status"], 9), row["surface"], row["name"].lower()))
    for row in gaps[:50]:
        api = row["id"].replace("numpy.", "np.", 1).replace("np.ndarray.", "ndarray.", 1)
        lines.append(f"| [`{api}`]({row['documentation_url']}) | {row['surface']} | {row['status']} | {row['category']} |")
    lines.extend([
        "",
        "## Counting rules",
        "",
        "The default scope is NumPy top-level callables, ndarray public methods/properties, and callables in numpy.random, numpy.linalg, and numpy.fft. Types, constants, modules, and NumSharp-only APIs remain searchable in the JSON artifact but do not affect the headline percentage.",
        "",
    ])
    return "\n".join(lines)


def render_outputs(np: Any, inventory: dict[str, Any], overrides: dict[str, Any]) -> dict[str, str]:
    rows, _ = resolve_rows(np, inventory, overrides)
    summary = build_summary(rows)
    payload = {
        "schema_version": 1,
        "generator_version": GENERATOR_VERSION,
        "numpy_version": np.__version__,
        "numsharp_assembly_version": inventory["assemblyVersion"],
        "methodology": {
            "headline": "Available default-scope APIs divided by all default-scope NumPy APIs.",
            "default_scope": "Top-level NumPy callables; ndarray public methods and properties; callable exports of numpy.random, numpy.linalg, and numpy.fft.",
            "availability_note": "Compiled API availability is distinct from fully verified behavioral parity.",
        },
        "summary": summary,
        "rows": rows,
    }
    manifest = {
        "schema_version": 1,
        "generator": "coverage/generate_coverage.py",
        "generator_version": GENERATOR_VERSION,
        "numpy_version": np.__version__,
        "numsharp_assembly_version": inventory["assemblyVersion"],
        "artifact_files": list(OUTPUT_FILES),
        "source_surfaces": ["numpy", "numpy.ndarray", "numpy.random", "numpy.linalg", "numpy.fft"],
        "numsharp_source_base_url": NUMSHARP_SOURCE_BASE_URL,
        "summary": summary,
    }
    return {
        "coverage.json": json_text(payload),
        "coverage.csv": csv_text(rows),
        "summary.md": markdown_text(summary, rows, np.__version__, inventory["assemblyVersion"]),
        "manifest.json": json_text(manifest),
    }


def check_outputs(output: Path, rendered: dict[str, str]) -> None:
    changed: list[str] = []
    for name, expected in rendered.items():
        path = output / name
        actual = path.read_text(encoding="utf-8") if path.exists() else None
        if actual != expected:
            changed.append(str(path.relative_to(ROOT)))
    if changed:
        sys.stderr.write("Coverage artifact is stale or missing:\n")
        sys.stderr.write("".join(f"  - {path}\n" for path in changed))
        sys.stderr.write("Run: python coverage/generate_coverage.py\n")
        raise SystemExit(1)


def write_outputs(output: Path, rendered: dict[str, str]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name, content in rendered.items():
        (output / name).write_text(content, encoding="utf-8", newline="")


def main() -> None:
    args = parse_args()
    np = load_numpy()
    inventory = load_numsharp_inventory()
    overrides = load_overrides(args.overrides)
    rendered = render_outputs(np, inventory, overrides)
    if args.check:
        check_outputs(args.output, rendered)
        print(f"Coverage artifact is current ({args.output}).")
    else:
        write_outputs(args.output, rendered)
        summary = json.loads(rendered["coverage.json"])["summary"]["default_scope"]
        print(
            f"Wrote {args.output}: {summary['available']}/{summary['total']} available "
            f"({summary['coverage_percent']:.1f}%)."
        )


if __name__ == "__main__":
    main()
