#!/usr/bin/env python3
"""Generate the deterministic NumPy <-> NumSharp public API coverage artifact."""

from __future__ import annotations

import argparse
import csv
import importlib
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

from numpy_documentation import documentation_url
from object_surfaces import OBJECT_SURFACE_PATHS, enrich_object_classes, object_surface_rows


ROOT = Path(__file__).resolve().parents[1]
PINNED_NUMPY_VERSION = "2.4.2"
GENERATOR_VERSION = "1.9.0"
OUTPUT_FILES = ("coverage.json", "coverage.csv", "summary.md", "manifest.json")
NUMSHARP_SOURCE_BASE_URL = "https://github.com/SciSharp/NumSharp/blob/master/"

CALLABLE_KINDS = {"function", "ufunc", "callable", "method"}
VALID_SUPPORT = {"declared", "partial", "unsupported", "missing", "extension"}
PLATFORM_SPECIFIC_NUMPY_EXPORTS = {"float96", "float128", "complex192", "complex256"}

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
    **dict.fromkeys({"acos", "arccos", "arcsin", "arctan", "arctan2", "asin", "atan", "atan2", "cos", "deg2rad", "degrees", "hypot", "rad2deg", "radians", "sin", "sinc", "tan", "unwrap"}, "Trigonometric functions"),
    **dict.fromkeys({"acosh", "arccosh", "arcsinh", "arctanh", "asinh", "atanh", "cosh", "sinh", "tanh"}, "Hyperbolic functions"),
    **dict.fromkeys({"exp", "exp2", "expm1", "log", "log10", "log1p", "log2", "logaddexp", "logaddexp2"}, "Exponentials & logarithms"),
    **dict.fromkeys({"around", "ceil", "fix", "floor", "rint", "round", "trunc"}, "Rounding"),
    **dict.fromkeys({"bitwise_and", "bitwise_count", "bitwise_invert", "bitwise_left_shift", "bitwise_not", "bitwise_or", "bitwise_right_shift", "bitwise_xor", "invert", "left_shift", "packbits", "right_shift", "unpackbits"}, "Bitwise operations"),
    **dict.fromkeys({"angle", "conj", "conjugate", "imag", "real", "real_if_close"}, "Complex numbers"),
    **dict.fromkeys({"convolve", "diff", "ediff1d", "gradient", "interp", "trapezoid"}, "Differences & integration"),
    **dict.fromkeys({"dot", "inner", "kron", "matmul", "outer", "trace", "cross"}, "Linear algebra"),
    **dict.fromkeys({"copysign", "frexp", "ldexp", "nextafter", "signbit", "spacing"}, "Floating-point handling"),
    "frompyfunc": "Function utilities",
}

MATH.update({"angle", "around", "float_power", "frompyfunc", "gradient", "i0", "imag", "interp", "packbits", "piecewise", "real", "real_if_close", "trapezoid", "unpackbits", "unwrap"})
CREATION.add("geomspace")
DTYPE.update({"astype", "isdtype"})
MANIPULATION.update({"broadcast_shapes", "ndim", "shape", "size"})
SELECTION.add("putmask")
SORTING.add("sort_complex")


# ---------------------------------------------------------------------------
# Extended (out-of-headline) NumPy submodule catalog.
#
# The historical headline covers five surfaces. The dashboard defaults to all API rows,
# including these public submodules and object_surfaces.py's member contracts. Keeping
# in_default_scope=False preserves a comparable headline without hiding wider API gaps.
#
# Each entry is (numpy import path, display surface, category, disposition). The disposition records
# WHY the family is out of headline scope so the summary can rank real opportunities above non-goals.
EXTENDED_SUBMODULES = [
    ("lib.scimath",           "emath",                 "Complex-domain math",       "candidate"),
    ("lib.stride_tricks",     "lib.stride_tricks",     "Stride tricks",             "candidate"),
    ("lib.array_utils",       "lib.array_utils",       "Array utilities",           "candidate"),
    ("polynomial",            "polynomial",            "Polynomial utilities",      "candidate"),
    ("polynomial.polyutils",  "polynomial.polyutils",  "Polynomial utilities",      "candidate"),
    ("polynomial.polynomial", "polynomial.polynomial", "Power-series polynomials",  "candidate"),
    ("polynomial.chebyshev",  "polynomial.chebyshev",  "Chebyshev polynomials",     "candidate"),
    ("polynomial.legendre",   "polynomial.legendre",   "Legendre polynomials",      "candidate"),
    ("polynomial.hermite",    "polynomial.hermite",    "Hermite polynomials",       "candidate"),
    ("polynomial.hermite_e",  "polynomial.hermite_e",  "HermiteE polynomials",      "candidate"),
    ("polynomial.laguerre",   "polynomial.laguerre",   "Laguerre polynomials",      "candidate"),
    ("ma",                    "ma",                    "Masked arrays",             "subsystem"),
    ("char",                  "char",                  "Legacy string operations",  "subsystem"),
    ("strings",               "strings",               "String operations",         "subsystem"),
    ("rec",                   "rec",                   "Record arrays",             "subsystem"),
    ("lib.recfunctions",      "lib.recfunctions",      "Structured-array helpers",  "subsystem"),
    ("testing",               "testing",               "Test support",              "tooling"),
    ("ctypeslib",             "ctypeslib",             "ctypes interop",            "tooling"),
    ("lib.format",            "lib.format",            "npy/npz format internals",  "tooling"),
    ("lib.introspect",        "lib.introspect",        "Runtime & diagnostics",     "tooling"),
]

# ndarray interop-protocol dunders. public_exports() drops every '_'-prefixed ndarray member, so
# these zero-copy / array-protocol hooks were invisible too. Catalogued out-of-headline against the
# NDArray host so a scan sees them; several (__array_interface__, __dlpack__, __array__) are genuine
# interop surface NumSharp could implement.
EXTENDED_NDARRAY_DUNDERS = [
    "__array__", "__array_interface__", "__array_ufunc__", "__array_function__",
    "__array_wrap__", "__array_finalize__", "__array_priority__",
    "__dlpack__", "__dlpack_device__", "__buffer__",
    "__index__", "__complex__", "__int__", "__float__",
]

# Why each extended family is out of headline scope — surfaced in summary.md so the reader can rank
# implementable "candidate" families above intentional non-goals.
DISPOSITION_NOTE = {
    "candidate": "Implementable NumPy submodule NumSharp does not expose yet (out of headline scope).",
    "subsystem": "Requires a NumSharp subsystem that does not exist yet (out of headline scope).",
    "tooling":   "Python-runtime tooling with no NumSharp analog (out of headline scope).",
    "interop":   "ndarray interop-protocol hook, absent in NumSharp (out of headline scope).",
}


IGNORED_DASHBOARD_ROOTS = (
    "ndarray.interop", "__array_namespace_info__", "lib", "ctypeslib", "testing",
)
LEGACY_POLYNOMIAL_APIS = {
    "poly", "poly1d", "polyadd", "polyder", "polydiv", "polyfit", "polyint",
    "polymul", "polysub", "polyval", "roots",
}
LEGACY_POLYNOMIAL_IDS = {"numpy." + name for name in LEGACY_POLYNOMIAL_APIS}


def within_namespace(value: str, root: str) -> bool:
    return value == root or value.startswith(root + ".")


def ignored_dashboard_id(api_id: str) -> bool:
    return any(within_namespace(api_id, "numpy." + root) for root in IGNORED_DASHBOARD_ROOTS)


def dashboard_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply presentation scope after matching, before emitting any dashboard data.

    IDs and implementation evidence remain canonical. Grouping never aliases a
    method to another owner or changes support; ignored rows never reach any UI
    scope, count, search result, or tooltip.
    """
    result = []
    for row in rows:
        if ignored_dashboard_id(row["id"]) or any(
            within_namespace(row["surface"], root) for root in IGNORED_DASHBOARD_ROOTS
        ):
            continue
        group = None
        if within_namespace(row["id"], "numpy.ma") or within_namespace(row["surface"], "ma"):
            group = ("ma", "Masked arrays")
        elif (within_namespace(row["id"], "numpy.polynomial")
              or within_namespace(row["id"], "numpy.poly1d")
              or within_namespace(row["surface"], "polynomial")
              or row["surface"] == "poly1d"
              or row["id"] in LEGACY_POLYNOMIAL_IDS):
            group = ("polynomial", "Polynomials")
        result.append({**row, "surface": group[0], "category": group[1]} if group else row)
    return result


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
    project = ROOT / "coverage" / "NumSharp.Tools.ApiInventory" / "NumSharp.Tools.ApiInventory.csproj"
    build_command = [
        "dotnet", "build", str(project), "--configuration", "Release", "--framework", "net8.0",
        "--nologo", "--verbosity", "quiet",
    ]
    build = subprocess.run(build_command, cwd=ROOT, check=False, text=True, capture_output=True)
    if build.returncode:
        sys.stderr.write(build.stdout)
        sys.stderr.write(build.stderr)
        raise SystemExit("Failed to build the NumSharp API inventory tool.")

    # Build output contains compiler warnings on stdout on clean Linux runners. Run
    # the already-built tool separately so stdout is guaranteed to be JSON only.
    command = [
        "dotnet", "run", "--project", str(project), "--configuration", "Release",
        "--framework", "net8.0", "--no-build", "--no-restore",
    ]
    completed = subprocess.run(command, cwd=ROOT, check=False, text=True, capture_output=True)
    if completed.returncode:
        sys.stderr.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        raise SystemExit("Failed to reflect the NumSharp public API.")
    try:
        data = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise SystemExit(f"NumSharp inventory emitted invalid JSON: {error}") from error
    if data.get("schemaVersion") != 5 or not isinstance(data.get("modules"), dict) or not data["modules"]:
        raise SystemExit("NumSharp inventory schema mismatch: expected schemaVersion 5 with a non-empty 'modules' map.")
    if not isinstance(data.get("unannotatedSurface"), dict):
        raise SystemExit("NumSharp inventory schema mismatch: schemaVersion 5 must carry the 'unannotatedSurface' index.")
    if not isinstance(data.get("exportedTypes"), list):
        raise SystemExit("NumSharp inventory schema mismatch: schemaVersion 5 must carry the 'exportedTypes' index.")
    if not isinstance(data.get("objectTypes"), dict):
        raise SystemExit("NumSharp inventory schema mismatch: schemaVersion 5 must carry the 'objectTypes' index.")
    return data


def surface_for_module(module_name: str) -> str:
    """Map a [ModuleName] value onto this generator's NumPy surface key.

    The tool discovers module hosts by scanning NumSharp.Core for [ModuleName("...")] — nothing is
    hardcoded on the C# side, so the mapping here must be mechanical too: "np" and "ndarray" are
    themselves; a dotted "np.random"/"np.linalg"/"np.fft" is its suffix ("random"/"linalg"/"fft"),
    matching the numpy-side surface names public_exports() emits.
    """
    return module_name[3:] if module_name.startswith("np.") else module_name


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


class SourceLocator:
    """Locate public member declarations without requiring compiler-specific PDB paths."""

    def __init__(self, modules: dict[str, Any]) -> None:
        # One class-declaration pattern per surface, derived from the CLR type hosting the module:
        # the simple class name of "NumSharp.np+linalg" is "linalg", of "NumSharp.FourierModule" is
        # "FourierModule". \b after the name still matches a generic partial ("class NDArray<T>").
        self.patterns: dict[str, re.Pattern[str]] = {}
        for module_name, type_data in modules.items():
            simple = re.split(r"[.+]", type_data["type"])[-1].split("`")[0]
            self.patterns[surface_for_module(module_name)] = re.compile(rf"\bclass\s+{re.escape(simple)}\b")

        self.files: dict[str, list[tuple[Path, str]]] = {surface: [] for surface in self.patterns}
        source_root = ROOT / "src" / "NumSharp.Core"
        for path in source_root.rglob("*.cs"):
            text = path.read_text(encoding="utf-8-sig")
            for surface, pattern in self.patterns.items():
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


def locate_type_source(name: str) -> list[str]:
    """Best-effort path to the .cs file declaring a NumSharp class, for crediting a type-match row.

    Only NumPy CLASS exports reach here (Generator/PCG64/SeedSequence/BitGenerator/MT19937), so a
    plain `class <Name>` scan over NumSharp.Core is enough — these are top-level types, not members.
    """
    pattern = re.compile(rf"\b(?:sealed\s+|abstract\s+|partial\s+|static\s+)*class\s+{re.escape(name)}\b")
    source_root = ROOT / "src" / "NumSharp.Core"
    # Sort by the POSIX string, not the default pathlib order: Path comparison is case-INSENSITIVE
    # on Windows and case-SENSITIVE on Linux, so the "first declaring file" (e.g. Generator.Choice.cs
    # vs Generator.Choice.Sampler.cs) would otherwise differ between a local Windows regen and the
    # Linux CI, breaking the checked-in-dashboard diff.
    for path in sorted(source_root.rglob("*.cs"), key=lambda p: p.as_posix()):
        try:
            if pattern.search(path.read_text(encoding="utf-8-sig")):
                return [path.relative_to(ROOT).as_posix()]
        except OSError:
            continue
    return []


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
    if name in IO:
        return "Input & output"
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
    if name in MATH:
        return "Math"
    return "Other"


def member_maps(
    inventory: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, str]]:
    """Index the tool's [ModuleName]-discovered modules by target id, surface, and target prefix.

    Nothing here names a module: surfaces come from surface_for_module, and each target prefix is
    the module's own CLR type name with nested '+' normalized to '.' — "NumSharp.np+linalg" hosts
    "NumSharp.np.linalg.solve", "NumSharp.FourierModule" hosts "NumSharp.FourierModule.fft".
    """
    modules: dict[str, Any] = inventory["modules"]
    by_target: dict[str, dict[str, Any]] = {}
    by_surface: dict[str, list[dict[str, Any]]] = {surface_for_module(name): [] for name in modules}
    prefixes: dict[str, str] = {
        surface_for_module(name): type_data["type"].replace("+", ".") for name, type_data in modules.items()
    }
    source_locator = SourceLocator(modules)
    for module_name in sorted(modules, key=str.lower):
        type_data = modules[module_name]
        surface = surface_for_module(module_name)
        prefix = prefixes[surface]
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
    return by_target, by_surface, prefixes


def public_exports(np: Any) -> list[dict[str, Any]]:
    exports: list[dict[str, Any]] = []

    # NumPy conditionally exports extended-precision aliases according to the C
    # platform. Compare the portable public surface so Windows and Linux produce
    # the same checked-in artifact; longdouble/clongdouble remain represented.
    for name in sorted(set(np.__all__) - PLATFORM_SPECIFIC_NUMPY_EXPORTS):
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


def resolve_submodule(np: Any, path: str) -> Any:
    """Return the numpy.<path> submodule, or None if this NumPy build does not ship it.

    Tries a real import first, then an attribute walk from the top package — numpy.emath is exposed
    as an ATTRIBUTE aliasing numpy.lib.scimath and is not importable by that name on every build, so
    the attribute fallback keeps the catalog complete without hard-coding aliases.

    :param np: the imported numpy module.
    :param path: dotted submodule path relative to numpy (e.g. "polynomial.chebyshev").
    :returns: the resolved module object, or None when it is absent (never raises).
    """
    try:
        return importlib.import_module("numpy." + path)
    except Exception:
        obj: Any = np
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                return None
        return obj if inspect.ismodule(obj) else None


def extended_surface_rows(
    np: Any, inventory: dict[str, Any], seen_ids: set[str],
    members_by_surface: dict[str, list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Catalog public callables of the NumPy submodules the headline scope excludes.

    These rows carry in_default_scope=False, so they never move the headline percentage; their only
    job is to make a scan SEE families (numpy.emath, numpy.polynomial.*, numpy.lib.stride_tricks,
    numpy.ma, numpy.char/strings, numpy.testing, ...) that the five-surface enumerator is structurally
    blind to. A member is credited "available" ONLY when NumSharp exposes a member of the SAME name on
    a [ModuleName]-annotated facade for that exact submodule (e.g. an eventual np.emath) — never via a
    top-level np namesake, whose semantics differ (numpy.emath.sqrt(-1) == 1j while np.sqrt(-1) == nan).
    Today no such facade exists, so every extended member reads "missing"; adding the facade later
    credits them automatically with no change here.

    :param np: the imported numpy module (the oracle surface).
    :param inventory: the reflected NumSharp inventory (its ``modules`` map supplies facade members).
    :param seen_ids: the running set of emitted row ids; mutated to keep ids globally unique.
    :returns: the list of extended catalog rows (may be empty if a NumPy build ships none of them).
    """
    # Top-level NumSharp np members — used ONLY to annotate a namesake (informational), never to credit.
    np_module = inventory["modules"].get("np", {"methods": [], "properties": [], "fields": []})
    np_toplevel = {m["name"] for grp in ("methods", "properties", "fields") for m in np_module[grp]}
    if members_by_surface is None:
        _, members_by_surface, _ = member_maps(inventory)

    rows: list[dict[str, Any]] = []

    def emit(surface: str, name: str, kind: str, category: str, disposition: str,
             host_module: str, obj: Any) -> None:
        # Row id mirrors the NumPy dotted path so it is stable and collision-free across submodules.
        row_id = f"numpy.{surface}.{name}"
        if row_id in seen_ids:  # a name can appear under both a module and its re-export; keep the first.
            return
        seen_ids.add(row_id)
        # Credit only a same-surface facade member (none exist yet -> missing). The top-level namesake
        # is recorded as prose so the emath.sqrt / char.upper false-positive class is visible, not hidden.
        matched = next((member for member in members_by_surface.get(surface_for_module(host_module), [])
                        if member["name"] == name), None)
        note = ("Matched against the same NumPy submodule facade; behavioral parity requires tests."
                if matched else DISPOSITION_NOTE[disposition])
        if not matched and name in np_toplevel:
            note += f" NumSharp has a top-level np.{name} (distinct surface/semantics; not credited here)."
        rows.append({
            "id": row_id, "origin": "numpy", "surface": surface, "name": name, "kind": kind,
            "numpy_signature": numpy_signature(obj, name),
            "documentation_url": documentation_url(surface, name, kind), "in_default_scope": False,
            "extended": True, "disposition": disposition,
            "category": category, "availability": "exact" if matched else "missing",
            "support": "declared" if matched else "missing",
            "status": "available" if matched else "missing",
            "numsharp_target": matched["target"] if matched else None,
            "numsharp_signatures": matched["signatures"] if matched else [],
            "numsharp_obsolete": matched.get("obsolete", False) if matched else False,
            "numsharp_source_paths": matched.get("sourcePaths", []) if matched else [],
            "numsharp_source_urls": matched.get("sourceUrls", []) if matched else [], "notes": note,
        })

    for path, surface, category, disposition in EXTENDED_SUBMODULES:
        module = resolve_submodule(np, path)
        if module is None:
            raise SystemExit(f"Required NumPy catalog submodule is absent: numpy.{path}")
        # Prefer __all__ (the module's own public contract) and fall back to non-underscore dir().
        names = getattr(module, "__all__", None) or [n for n in dir(module) if not n.startswith("_")]
        host_module = "np." + surface  # the [ModuleName] a NumSharp facade for this family would carry.
        for name in sorted(set(names)):
            obj = getattr(module, name)
            # Functions/ufuncs only — classes (Polynomial, MaskedArray, chararray) are noted in the
            # summary prose rather than catalogued as member rows, matching the "functions" question.
            is_call = (isinstance(obj, np.ufunc) or inspect.isfunction(obj) or inspect.isbuiltin(obj)
                       or (callable(obj) and not inspect.isclass(obj) and not inspect.ismodule(obj)))
            if not is_call:
                continue
            emit(surface, name, numpy_kind(np, obj), category, disposition, host_module, obj)

    # ndarray interop dunders — public_exports() drops these via its '_'-prefix filter.
    for name in EXTENDED_NDARRAY_DUNDERS:
        if not hasattr(np.ndarray, name):  # dunder set drifts across NumPy versions; skip absent ones.
            continue
        obj = inspect.getattr_static(np.ndarray, name)
        kind = "method" if callable(getattr(np.ndarray, name)) else "property"
        emit("ndarray.interop", name, kind, "Array interop protocol", "interop", "ndarray", obj)

    return rows


def direct_target(row: dict[str, Any], targets: dict[str, dict[str, Any]], prefixes: dict[str, str]) -> str | None:
    surface = row["surface"]
    name = row["name"]
    kind = row["kind"]
    if row["id"] == "numpy.ndarray":
        return "NumSharp.NDArray"
    prefix = prefixes.get(surface)
    if prefix:
        candidate = f"{prefix}.{name}"
        member = targets.get(candidate)
        if member and (kind not in CALLABLE_KINDS or member["kind"] == "method"):
            return candidate
    return None


def auto_alternative(
    row: dict[str, Any], targets: dict[str, dict[str, Any]], prefixes: dict[str, str]
) -> tuple[str | None, str | None]:
    np_prefix = prefixes.get("np")
    if np_prefix and row["surface"] in {"ndarray", "linalg"} and row["kind"] in CALLABLE_KINDS:
        target = f"{np_prefix}.{row['name']}"
        if target in targets:
            noun = "instance method" if row["surface"] == "ndarray" else "linalg namespace function"
            return target, f"Available through the static NumSharp np API instead of the NumPy {noun}."
    return None, None


def resolve_rows(np: Any, inventory: dict[str, Any], overrides: dict[str, Any]) -> tuple[list[dict[str, Any]], set[str]]:
    targets, surfaces, prefixes = member_maps(inventory)
    # Exported-type index (simple name -> full name) for crediting a NumPy CLASS export against a
    # NumSharp type of the same name. A repeated simple name prefers the root NumSharp.<Name>.
    numsharp_types_by_simple: dict[str, str] = {}
    for full in inventory.get("exportedTypes", []):
        simple = full.rsplit(".", 1)[-1].split("+")[-1]
        if simple not in numsharp_types_by_simple or full == f"NumSharp.{simple}":
            numsharp_types_by_simple[simple] = full
    # Case-sensitive matching IS the parity contract: NumPy's public API is case-sensitive, so every
    # match below (direct_target / auto_alternative / aliases / the stray gate) is an exact dict
    # lookup on the C# spelling. We ALSO fold case here to DETECT near-misses — an in-scope NumPy
    # export left "missing" for which a same-surface NumSharp member differs only by case — and
    # report them (never counting them as covered). This is the guard against silently satisfying
    # NumPy's `histogram` with a C#-style `Histogram`. Keyed (surface, lowercased name).
    case_folded: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for surface, members in surfaces.items():
        for member in members:
            case_folded.setdefault((surface, member["name"].lower()), []).append(member)
    aliases = overrides.get("aliases", {})
    support_overrides = overrides.get("support", {})
    stray_allowlist = overrides.get("stray_allowlist", {})
    seen_ids: set[str] = set()
    consumed_targets: set[str] = set()
    rows: list[dict[str, Any]] = []

    exports = public_exports(np)
    # Guard the discovery loop: every NumPy surface compared here must have a [ModuleName]-annotated
    # host in NumSharp.Core. Without this, dropping an annotation silently zeroes that surface back
    # to all-missing — exactly the failure mode attribute discovery was built to end.
    unbacked = sorted({export["surface"] for export in exports} - set(prefixes))
    if unbacked:
        raise SystemExit(
            "NumPy surfaces without a [ModuleName]-annotated NumSharp host: "
            + ", ".join(unbacked)
            + ". Annotate the hosting type (e.g. [ModuleName(\"np.fft\")]) in NumSharp.Core."
        )

    for export in exports:
        row_id = export["id"]
        if row_id in seen_ids:
            raise SystemExit(f"Duplicate coverage id: {row_id}")
        seen_ids.add(row_id)
        alias = aliases.get(row_id)
        target = direct_target(export, targets, prefixes)
        if alias and target:
            sys.stderr.write(
                f"WARNING: override alias for {row_id} is stale — {target} now matches directly; "
                "delete the alias from overrides.json.\n"
            )
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
            target, automatic_note = auto_alternative(export, targets, prefixes)
            if target:
                availability = "alias"
                notes.append(automatic_note or "Available on an alternate NumSharp surface.")

        # Type-match: a NumPy CLASS export credited to a NumSharp exported type of the same name
        # (numpy.random.Generator -> NumSharp.Generator). Class exports are out of default scope, so
        # this only fixes the Types catalog — it is what let Generator/PCG64/SeedSequence/
        # BitGenerator/MT19937 read "missing" while NumSharp exports every one of them.
        type_target = None
        if not target and export["kind"] not in CALLABLE_KINDS:
            type_target = numsharp_types_by_simple.get(export["name"])
            if type_target:
                availability = "type"
                notes.append(f"NumSharp exports the {export['name']} type ({type_target}).")

        support = "declared" if (target or type_target) else "missing"
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
        elif type_target:
            consumed_targets.add(type_target)
            signatures = [f"public class {export['name']}"]
            obsolete = False
            source_paths = locate_type_source(export["name"])
            source_urls = SourceLocator.github_urls(source_paths)
        else:
            signatures = []
            obsolete = False
            source_paths = []
            source_urls = []

        # Case-insensitive near-miss detection: a still-missing in-scope export whose spelling matches
        # a same-surface NumSharp member only when case is ignored. Reported, never counted — parity
        # requires the exact NumPy spelling.
        case_insensitive_matches: list[str] = []
        if not target and export["in_default_scope"]:
            case_insensitive_matches = sorted({
                member["target"]
                for member in case_folded.get((export["surface"], export["name"].lower()), [])
                if member["name"] != export["name"]
            })
            if case_insensitive_matches:
                notes.append(
                    "Case-insensitive near-miss (NOT counted — NumPy parity is case-sensitive): "
                    + ", ".join(case_insensitive_matches)
                )

        matched = target or type_target
        display_status = "missing" if not matched else support if support in {"partial", "unsupported"} else "available"
        row = {
            **export,
            "category": category_for(export["surface"], export["name"], export["kind"]),
            "availability": availability,
            "support": support,
            "status": display_status,
            "numsharp_target": matched,
            "numsharp_signatures": signatures,
            "numsharp_obsolete": obsolete,
            "numsharp_source_paths": source_paths,
            "numsharp_source_urls": source_urls,
            "notes": " ".join(dict.fromkeys(notes)),
        }
        if case_insensitive_matches:
            row["case_insensitive_matches"] = case_insensitive_matches
        rows.append(row)

    unknown_aliases = set(aliases) - seen_ids
    unknown_support = set(support_overrides) - seen_ids
    unknown_strays = set(stray_allowlist) - seen_ids
    if unknown_aliases or unknown_support or unknown_strays:
        unknown = ", ".join(sorted(unknown_aliases | unknown_support | unknown_strays))
        raise SystemExit(f"Overrides reference NumPy exports that were not discovered: {unknown}")

    # Stray-host gate: an in-scope NumPy export left "missing" whose name nevertheless exists on an
    # UNANNOTATED public type is a scan miss (the np.fft failure mode — implemented, but on a type
    # the inventory never reflects), not a genuine gap. Fail loudly, naming the candidate hosts.
    # Reviewed name coincidences go in overrides.json under "stray_allowlist" ({numpy id: note}).
    member_hosts: dict[str, list[str]] = {}
    for type_name, members in inventory["unannotatedSurface"].items():
        for member in members:
            member_hosts.setdefault(member, []).append(type_name)
    strays = [
        (row["id"], member_hosts[row["name"]])
        for row in rows
        if row["origin"] == "numpy" and row["in_default_scope"] and row["status"] == "missing"
        and row["id"] not in stray_allowlist and row["name"] in member_hosts
    ]
    if strays:
        details = "".join(f"  - {row_id} exists on: {', '.join(hosts)}\n" for row_id, hosts in strays)
        raise SystemExit(
            "Missing NumPy exports whose names exist on unannotated NumSharp types (scan misses?):\n"
            + details
            + "Annotate the hosting type with [ModuleName(\"...\")], or record a reviewed name "
            "coincidence in overrides.json under \"stray_allowlist\"."
        )

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

    rows.extend(extended_surface_rows(np, inventory, seen_ids, surfaces))
    rows.extend(object_surface_rows(
        np, inventory, seen_ids, root=ROOT, source_base_url=NUMSHARP_SOURCE_BASE_URL,
        signature=numpy_signature, documentation_url=documentation_url,
    ))
    enrich_object_classes(rows, inventory, ROOT, NUMSHARP_SOURCE_BASE_URL)

    # A facade member used by an extended mapping is no longer a NumSharp-only extension.
    consumed_targets.update(row["numsharp_target"] for row in rows
                            if row["origin"] == "numpy" and row["numsharp_target"])
    rows = [row for row in rows if row["origin"] == "numpy" or row["numsharp_target"] not in consumed_targets]

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
    # Extended submodules are catalogued out-of-headline; summarise them SEPARATELY so they are
    # visible/searchable without diluting the default-scope percentage the headline reports.
    extended_rows = [row for row in rows if row.get("extended")]
    api_rows = [row for row in numpy_rows if is_api_row(row)]
    object_rows = [row for row in api_rows if row.get("object_surface")]
    by_extended = {
        surface: {
            **status_counts([row for row in extended_rows if row["surface"] == surface]),
            "disposition": next((row["disposition"] for row in extended_rows if row["surface"] == surface), ""),
        }
        for surface in sorted({row["surface"] for row in extended_rows})
    }
    return {
        "default_scope": status_counts(default_rows),
        "by_surface": by_surface,
        "by_category": by_category,
        "extended_surfaces": {"total": len(extended_rows), "by_surface": by_extended},
        "api_scope": {
            **status_counts(api_rows),
            "by_surface": {surface: status_counts([row for row in api_rows if row["surface"] == surface])
                           for surface in sorted({row["surface"] for row in api_rows})},
            "by_category": {category: status_counts([row for row in api_rows if row["category"] == category])
                            for category in sorted({row["category"] for row in api_rows})},
        },
        "object_surfaces": status_counts(object_rows),
        "ufunc_protocols": {
            "total": sum("applicability" in row for row in object_rows),
            "conditional": sum(row.get("applicability") == "conditional" for row in object_rows),
            "not_applicable": sum(row.get("applicability") == "not_applicable" for row in object_rows),
        },
        "all_numpy_exports": len(numpy_rows),
        "numsharp_extensions": sum(row["origin"] == "numsharp" for row in rows),
        "catalog_rows": len(rows),
    }


def is_api_row(row: dict[str, Any]) -> bool:
    """Supporting classes/constants/modules remain catalogued outside API percentages."""
    return row["origin"] == "numpy" and (row["in_default_scope"] or
        (row.get("extended", False) and row["kind"] not in {"class", "constant", "module"}))


def json_text(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=False) + "\n"


def csv_text(rows: list[dict[str, Any]]) -> str:
    columns = [
        "id", "origin", "surface", "category", "name", "kind", "in_default_scope", "extended", "disposition",
        "object_surface", "object_type", "applicability", "status", "availability",
        "support", "numpy_signature", "numsharp_target", "numsharp_signatures", "numsharp_source_paths",
        "numsharp_source_urls", "numsharp_obsolete", "case_insensitive_matches", "notes", "documentation_url"
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        flat = {key: row.get(key, "") for key in columns}
        flat["numsharp_signatures"] = " | ".join(row["numsharp_signatures"])
        flat["numsharp_source_paths"] = " | ".join(row["numsharp_source_paths"])
        flat["numsharp_source_urls"] = " | ".join(row["numsharp_source_urls"])
        flat["case_insensitive_matches"] = " | ".join(row.get("case_insensitive_matches", []))
        writer.writerow(flat)
    return stream.getvalue()


def markdown_text(summary: dict[str, Any], rows: list[dict[str, Any]], numpy_version: str, assembly_version: str) -> str:
    headline = summary["default_scope"]
    all_apis = summary["api_scope"]
    lines = [
        "# NumPy ↔ NumSharp API coverage",
        "",
        f"Compared with NumPy **{numpy_version}** using NumSharp assembly **{assembly_version}**.",
        "",
        f"Expanded API availability (dashboard default): **{all_apis['coverage_percent']:.1f}%** "
        f"({all_apis['available']} of {all_apis['total']} APIs across "
        f"{len(all_apis['by_surface'])} surfaces and {len(all_apis['by_category'])} categories). "
        f"**{all_apis['missing']} missing**, **{all_apis['partial']} partial**.",
        "",
        f"The catalog includes **{summary['ufunc_protocols']['total']} per-ufunc protocol methods**; "
        f"**{summary['ufunc_protocols']['not_applicable']}** describe methods whose calls NumPy rejects "
        "for that ufunc. These count as exposed API contracts, not computable operations.",
        "",
        f"Headline API availability: **{headline['coverage_percent']:.1f}%** "
        f"({headline['available']} of {headline['total']} default-scope APIs). "
        f"Including partial mappings, **{headline['addressed_percent']:.1f}%** are addressed.",
        "",
        "| Surface | Available | Partial | Unsupported | Missing | Total | Coverage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {"np": "np.*", "ndarray": "ndarray.*", "random": "np.random.*", "linalg": "np.linalg.*", "fft": "np.fft.*", "ma": "np.ma.*", "polynomial": "np.polynomial.*"}
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

    ci_rows = [
        row for row in rows
        if row["origin"] == "numpy" and row["in_default_scope"] and row.get("case_insensitive_matches")
    ]
    lines.extend([
        "",
        "## Case-insensitive near-misses",
        "",
        "NumPy's public API is case-sensitive, so a NumSharp member is credited only when the spelling "
        "matches exactly. The generator additionally folds case to surface near-misses — in-scope NumPy "
        "APIs left *missing* for which NumSharp exposes a same-surface member differing only by case. "
        "These are **not** counted as available; rename to the exact NumPy spelling (or record a reviewed "
        "alias) to close them.",
        "",
    ])
    if ci_rows:
        lines.append("| NumPy API | Surface | Differs only by case from |")
        lines.append("|---|---|---|")
        for row in sorted(ci_rows, key=lambda r: (r["surface"], r["name"].lower())):
            api = row["id"].replace("numpy.", "np.", 1).replace("np.ndarray.", "ndarray.", 1)
            targets_md = ", ".join(f"`{target}`" for target in row["case_insensitive_matches"])
            lines.append(f"| [`{api}`]({row['documentation_url']}) | {row['surface']} | {targets_md} |")
    else:
        lines.append("_None detected._")

    # Extended-submodule section: the families the headline scope excludes, listed so a scan cannot
    # miss them. Ranked candidate -> subsystem -> tooling -> interop so implementable gaps read first.
    extended = [row for row in rows if row.get("extended")]
    lines.extend([
        "",
        "## Extended NumPy submodules and object contracts",
        "",
        "These surfaces appear in the dashboard by default. `candidate` denotes numeric APIs and "
        "object contracts; `subsystem` denotes a separate object model such as masked/string/record "
        "arrays or ufuncs; `tooling` denotes Python tooling; `interop` denotes array protocols. "
        "Class existence is catalogued separately from member availability. The historical headline "
        "remains unchanged by these additions.",
        "",
    ])
    if extended:
        disposition_rank = {"candidate": 0, "subsystem": 1, "tooling": 2, "interop": 3}
        lines.append("| Surface | Disposition | Available | Missing | Total | Notable missing |")
        lines.append("|---|---|---:|---:|---:|---|")
        surfaces = sorted(
            {row["surface"] for row in extended},
            key=lambda s: (disposition_rank.get(
                next((r["disposition"] for r in extended if r["surface"] == s), "interop"), 9), s),
        )
        for surface in surfaces:
            members = [row for row in extended if row["surface"] == surface]
            disposition = members[0].get("disposition", "")
            available = sum(1 for row in members if row["status"] == "available")
            missing = [row["name"] for row in members if row["status"] == "missing"]
            sample = ", ".join(f"`{name}`" for name in missing[:6]) + (" …" if len(missing) > 6 else "")
            lines.append(
                f"| numpy.{surface} | {disposition} | {available} | {len(missing)} | {len(members)} | {sample} |"
            )
        lines.append("")

    lines.extend([
        "",
        "## Expanded capability categories",
        "",
        "| Category | Available | Partial | Missing | Total |",
        "|---|---:|---:|---:|---:|",
    ])
    for category, counts in all_apis["by_category"].items():
        lines.append(f"| {category} | {counts['available']} | {counts['partial']} | {counts['missing']} | {counts['total']} |")

    lines.extend([
        "",
        "## Counting rules",
        "",
        "The historical headline is NumPy top-level callables, ndarray public methods/properties, and callables in numpy.random, numpy.linalg, and numpy.fft. The expanded API scope adds public extended-submodule callables and object methods/properties/protocols. Types, constants, modules, and NumSharp-only APIs remain searchable in the full catalog but do not enter either API denominator. Aliases are separate exported names; inherited members are separate contracts on each object type. A ufunc method row records the exposed protocol even when that ufunc rejects the operation (for example, unary reductions). Matching a namespace function never credits a ufunc or a different object type's member.",
        "",
    ])
    return "\n".join(lines)


def render_outputs(np: Any, inventory: dict[str, Any], overrides: dict[str, Any]) -> dict[str, str]:
    rows, _ = resolve_rows(np, inventory, overrides)
    rows = dashboard_rows(rows)
    summary = build_summary(rows)
    payload = {
        "schema_version": 1,
        "generator_version": GENERATOR_VERSION,
        "numpy_version": np.__version__,
        "numsharp_assembly_version": inventory["assemblyVersion"],
        "methodology": {
            "headline": "Available default-scope APIs divided by all default-scope NumPy APIs.",
            "default_scope": "Top-level NumPy callables; ndarray public methods and properties; callable exports of numpy.random, numpy.linalg, and numpy.fft.",
            "api_scope": "Dashboard default: headline APIs plus extended public submodule callables and object methods/properties/protocols. Supporting class, constant and module exports are catalogued but excluded from API percentages.",
            "object_members": "Object members match only their explicit NumSharp owner type. Inherited members and ufunc protocols are catalogued per owner; namespace namesakes never confer support. Protocol existence does not imply applicability to every ufunc.",
            "availability_note": "Compiled API availability is distinct from fully verified behavioral parity.",
            "case_sensitivity": "NumPy API names are matched case-sensitively for parity. Case-insensitive near-misses are detected and reported (row field 'case_insensitive_matches'; the 'Case-insensitive near-misses' section of summary.md) but never counted as available.",
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
        # Out-of-headline families the scan also catalogs (in_default_scope=false), so a whole
        # submodule can no longer go missing the way numpy.fft/emath/polynomial once did.
        "extended_surfaces": ["numpy." + surface for surface in sorted({row["surface"] for row in rows if row.get("extended")})],
        "object_surfaces": [path for path in OBJECT_SURFACE_PATHS if not ignored_dashboard_id(path)],
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
    near_misses = [row for row in json.loads(rendered["coverage.json"])["rows"] if row.get("case_insensitive_matches")]
    if near_misses:
        print(f"Case-insensitive near-misses (NOT counted — NumPy parity is case-sensitive): {len(near_misses)}")
        for row in near_misses:
            print(f"  - {row['id']} ~ {', '.join(row['case_insensitive_matches'])}")
    else:
        print("Case-insensitive near-misses: none.")
    if args.check:
        check_outputs(args.output, rendered)
        print(f"Coverage artifact is current ({args.output}).")
    else:
        write_outputs(args.output, rendered)
        summary = json.loads(rendered["coverage.json"])["summary"]
        expanded = summary["api_scope"]
        headline = summary["default_scope"]
        print(
            f"Wrote {args.output}: {summary['catalog_rows']} catalog rows; "
            f"expanded {expanded['available']}/{expanded['total']} available ({expanded['coverage_percent']:.1f}%); "
            f"headline {headline['available']}/{headline['total']} ({headline['coverage_percent']:.1f}%)."
        )


if __name__ == "__main__":
    main()
