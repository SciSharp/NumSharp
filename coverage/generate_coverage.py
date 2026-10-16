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
import types
import typing
import warnings
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import quote

import oracle_evidence
from numpy_documentation import documentation_index, documentation_url
from object_surfaces import (OBJECT_SURFACE_PATHS, SIGNATURE_UNAVAILABLE, class_member_rows, enrich_object_classes,
                             object_surface_rows, scalar_type_rows)


ROOT = Path(__file__).resolve().parents[1]
PINNED_NUMPY_VERSION = "2.4.2"
GENERATOR_VERSION = "1.14.0"
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
    # A class: its own row stays in "Types" (category_for categorises every non-callable first); the entry
    # categorises the class's member rows (member_category), errstate's __enter__/__exit__/__call__.
    "errstate": "Floating-point handling",
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
    # NumPy's own public-module list (numpy/tests/test_public_api.py: PUBLIC_MODULES and
    # PUBLIC_ALIASED_MODULES) names every module below as well; until generator 1.13.0 the catalog
    # never scanned them, so numpy.dtypes, numpy.exceptions and numpy.typing had no rows at all.
    # test_catalog compares this list with the installed NumPy's list and fails on a module that is
    # neither scanned here nor excluded, with a reason, in EXCLUDED_PUBLIC_SUBMODULES.
    ("dtypes",                "dtypes",                "Dtype classes",             "candidate"),
    ("exceptions",            "exceptions",            "Exceptions & warnings",     "candidate"),
    ("ma.mrecords",           "ma.mrecords",           "Masked record arrays",      "subsystem"),
    ("typing",                "typing",                "Static typing",             "tooling"),
    ("version",               "version",               "Runtime & diagnostics",     "tooling"),
    ("f2py",                  "f2py",                  "Fortran bindings",          "tooling"),
    ("lib",                   "lib",                   "Library support",           "tooling"),
    ("lib.mixins",            "lib.mixins",            "Operator mixins",           "tooling"),
    ("testing.overrides",     "testing.overrides",     "Test support",              "tooling"),
    # Documented in NumPy's reference although test_public_api.py does not list it: the
    # numpy.testing.extbuild helpers (an attribute of numpy.testing, not an importable path;
    # resolve_submodule's attribute walk finds it). Of the other two such modules, numpy.lib.user_array
    # documents one name, its container class, which object_surfaces catalogues with its members, and
    # numpy.matlib is a NumPy 1.x compatibility module (COMPAT_EXCLUDED_APIS).
    ("testing.extbuild",      "testing.extbuild",      "Test support",              "tooling"),
]

# NumPy APIs that the pinned NumPy still ships but has itself deprecated, kept there only so code written
# for NumPy 1.x keeps working. The catalog leaves each one out, everything under it included: a NumSharp
# port follows the API NumPy points to, not the compatibility shim, so neither a missing row nor a credit
# should count it. A name joins this table only with NumPy's own deprecation as the reason.
# test_catalog's deprecation guard fails on any catalogued row whose NumPy documentation deprecates the
# API as a whole, unless the row is listed here (left out) or in KEPT_DEPRECATED_APIS (kept, with a reason).
COMPAT_EXCLUDED_APIS = {
    "numpy.matlib": ("Deprecated since NumPy 1.19: importing it raises PendingDeprecationWarning ('Importing "
                     "from numpy.matlib is deprecated since 1.19.0. The matrix subclass is not the recommended "
                     "way to represent matrices or deal with linear algebra ... Please adjust your code to use "
                     "regular ndarray'). Its eight functions are numpy.zeros, ones, empty, eye, identity, "
                     "random.rand, random.randn and tile returning matrices."),
    "numpy.typing.NBitBase": ("Deprecated in NumPy 2.3 ('.. deprecated:: 2.3 Use @typing.overload or a TypeVar "
                              "with a scalar-type as upper bound, instead'); added in NumPy 1.20 for the "
                              "precision-generic annotations of the 1.x typing stubs."),
}

# Catalogued APIs NumPy documents as deprecated as a whole, kept in the catalog on purpose, each with the
# reason. The deprecation guard (test_catalog) accepts exactly these; every other deprecated row fails it.
KEPT_DEPRECATED_APIS = {
    "numpy.random.random_integers": ("Deprecated since NumPy 1.11 (use randint), but a numpy.random headline "
                                     "export that NumSharp implements and the oracle verifies; removing it "
                                     "would change the headline denominator. Owner's call."),
    "numpy.random.RandomState.random_integers": ("Deprecated since NumPy 1.11; the RandomState method of the "
                                                 "same function, kept with it."),
    "numpy.testing.assert_warns": ("Deprecated in NumPy 2.4; numpy.testing is catalogued but never shown "
                                   "(IGNORED_DASHBOARD_ROOTS)."),
    "numpy.testing.suppress_warnings": ("Deprecated in NumPy 2.4; numpy.testing is catalogued but never "
                                        "shown (IGNORED_DASHBOARD_ROOTS)."),
}


def compat_excluded(api_id: str) -> bool:
    """Tell whether a ``numpy.*`` id lies in a NumPy 1.x compatibility API the catalog leaves out.

    :param api_id: a catalog id (``numpy.matlib.zeros``) or a submodule path prefixed with ``numpy.``.
    :returns: True when the id is a COMPAT_EXCLUDED_APIS entry or a member of one.
    """
    return any(within_namespace(api_id, excluded) for excluded in COMPAT_EXCLUDED_APIS)


_SECTION_UNDERLINE = re.compile(r"^\s*([-=~^])\1{2,}\s*$")


def deprecation_scope(doc: str | None) -> str | None:
    """Tell what a NumPy docstring's ``.. deprecated::`` directive deprecates: the API, or a part of it.

    numpydoc puts a whole-object deprecation directly after the summary, unindented and before the first
    section (random_integers, NBitBase), and a parameter's or a return value's inside that section's
    entry, indented under it (numpy.fft.fft2's ``s`` without ``axes``, numpy.fromstring's ``sep=''``).

    :param doc: a cleaned docstring (``inspect.getdoc``), or None.
    :returns: ``"api"`` when an unindented directive appears before any section heading, ``"part"`` when
        every directive is indented or follows a heading, None when the docstring has none.
    """
    if not doc or ".. deprecated::" not in doc:
        return None
    lines = doc.splitlines()
    for index, line in enumerate(lines):
        # A heading is a line followed by an underline of '-', '=', '~' or '^' (a module docstring's
        # title counts too: everything after it is a section of the module's documentation).
        if index + 1 < len(lines) and line.strip() and _SECTION_UNDERLINE.match(lines[index + 1]):
            return "part"
        if line.startswith(".. deprecated::"):
            return "api"
    return "part"

# Public NumPy modules (by NumPy's own list, see above) the catalog deliberately does not scan, each
# with the reason. A module counts as covered when it, or a parent listed here, appears below, so
# numpy.distutils' submodules need no entry of their own.
EXCLUDED_PUBLIC_SUBMODULES = {
    "ma.extras": ("Implementation module: numpy.ma's __all__ re-exports the whole of its __all__, so "
                  "every member is catalogued as numpy.ma.<name>."),
    "lib.npyio": ("Its only public names, DataSource and NpzFile, are object surfaces of their own "
                  "(numpy.lib.npyio.DataSource / numpy.lib.npyio.NpzFile, every member catalogued)."),
    "typing.mypy_plugin": ("The mypy plugin entry point: named in a mypy configuration file, never "
                           "imported by user code; its __all__ is empty."),
    "distutils": ("Build tooling NumPy lists as public only on Python < 3.12 (removed for 3.12+, "
                  "deprecated since NumPy 1.23); the catalog targets the pinned NumPy on Python 3.12."),
}

# The submodules whose callables are part of the five-surface headline (numpy.<name>). One list
# for public_exports and the ufunc protocol scan, so a surface added here is scanned by both.
HEADLINE_SUBMODULES = ("random", "linalg", "fft")

# Every namespace the catalog compares, as (import path relative to numpy, display surface). The
# object inventory catalogs each ufunc method on every ufunc-like export found here (numpy itself
# is always included): numpy.strings.str_len and numpy.ma's masked add are ufunc protocol owners
# just like numpy.add, so scanning only the top level left their methods out of the catalog.
UFUNC_PROTOCOL_NAMESPACES = (
    tuple((surface, surface) for surface in HEADLINE_SUBMODULES)
    + tuple((path, surface) for path, surface, _category, _disposition in EXTENDED_SUBMODULES)
)

# Why each extended family is out of headline scope — surfaced in summary.md so the reader can rank
# implementable "candidate" families above intentional non-goals. ndarray's Python protocols, once a
# hand-picked "interop" list under the invented id numpy.ndarray.interop.*, are catalogued with every
# other protocol under their real ids (numpy.ndarray.__array__) by the ndarray object surface.
DISPOSITION_NOTE = {
    "candidate": "Implementable NumPy submodule NumSharp does not expose yet (out of headline scope).",
    "subsystem": "Requires a NumSharp subsystem that does not exist yet (out of headline scope).",
    "tooling":   "Python-runtime tooling with no NumSharp analog (out of headline scope).",
}


# The namespaces the catalog lists but the dashboard never shows: NumPy's test helpers (numpy.testing)
# and its library-support package (numpy.lib), by the owner's decision. Every other public NumPy API is
# shown. IGNORED_DASHBOARD_EXCEPTIONS names what inside those namespaces is shown anyway.
IGNORED_DASHBOARD_ROOTS = ("lib", "testing")
# numpy.lib.NumpyVersion, the version-comparison class user code imports to gate behaviour on the NumPy
# version, is shown although numpy.lib is not (the owner's decision); its rows are grouped under their
# own surface, lib.NumpyVersion, so the class row does not stand alone under "lib".
IGNORED_DASHBOARD_EXCEPTIONS = ("lib.NumpyVersion",)
LEGACY_POLYNOMIAL_APIS = {
    "poly", "poly1d", "polyadd", "polyder", "polydiv", "polyfit", "polyint",
    "polymul", "polysub", "polyval", "roots",
}
LEGACY_POLYNOMIAL_IDS = {"numpy." + name for name in LEGACY_POLYNOMIAL_APIS}


def within_namespace(value: str, root: str) -> bool:
    return value == root or value.startswith(root + ".")


def shown_exception(api_id: str) -> str | None:
    """Return the IGNORED_DASHBOARD_EXCEPTIONS entry an id lies in, if any.

    :param api_id: a catalog id (``numpy.lib.NumpyVersion.major``).
    :returns: the exception path (``lib.NumpyVersion``) or None.
    """
    return next((path for path in IGNORED_DASHBOARD_EXCEPTIONS if within_namespace(api_id, "numpy." + path)), None)


def ignored_dashboard_id(api_id: str) -> bool:
    """Tell whether the dashboard hides an id: it lies in an ignored root and in none of its exceptions.

    :param api_id: a catalog id or an object-surface path prefixed with ``numpy.``.
    :returns: True for numpy.testing.* and numpy.lib.* ids other than numpy.lib.NumpyVersion's.
    """
    return (any(within_namespace(api_id, "numpy." + root) for root in IGNORED_DASHBOARD_ROOTS)
            and shown_exception(api_id) is None)


def dashboard_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply presentation scope after matching, before emitting any dashboard data.

    IDs and implementation evidence remain canonical. Grouping never aliases a
    method to another owner or changes support; ignored rows never reach any UI
    scope, count, search result, or tooltip. A row inside an ignored root is hidden by
    its id or by its surface, unless its id lies in IGNORED_DASHBOARD_EXCEPTIONS; such a
    row is shown under the exception's own surface (the class row numpy.lib.NumpyVersion
    has the surface ``lib``, its members ``lib.NumpyVersion``).

    :param rows: the resolved catalog rows (resolve_rows), hidden namespaces included.
    :returns: new row dicts for the shown rows, in input order; numpy.ma rows are grouped under the ``ma``
        surface and the polynomial family under ``polynomial``. The input rows are not modified.
    """
    result = []
    for row in rows:
        exception = shown_exception(row["id"])
        if exception is not None:
            result.append({**row, "surface": exception})
            continue
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
    """Build and run NumSharp.Tools.ApiInventory, returning the reflected NumSharp public surface.

    :returns: the inventory JSON (schema 6): ``modules`` by [ModuleName], ``unannotatedSurface``,
        ``exportedTypes`` and ``objectTypes`` (each type's members and, since schema 6, its operators).
    :raises SystemExit: when the tool does not build or run, emits invalid JSON, or emits another schema
        (an inventory without operators would report every operator protocol missing).
    """
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
    if data.get("schemaVersion") != 6 or not isinstance(data.get("modules"), dict) or not data["modules"]:
        raise SystemExit("NumSharp inventory schema mismatch: expected schemaVersion 6 with a non-empty 'modules' map.")
    if not isinstance(data.get("unannotatedSurface"), dict):
        raise SystemExit("NumSharp inventory schema mismatch: schemaVersion 6 must carry the 'unannotatedSurface' index.")
    if not isinstance(data.get("exportedTypes"), list):
        raise SystemExit("NumSharp inventory schema mismatch: schemaVersion 6 must carry the 'exportedTypes' index.")
    if not isinstance(data.get("objectTypes"), dict):
        raise SystemExit("NumSharp inventory schema mismatch: schemaVersion 6 must carry the 'objectTypes' index.")
    # Schema 6's addition: every type carries its C# operators and conversions, which credit NumPy's
    # Python operator protocols. An inventory without them would silently report every operator missing.
    if any(not isinstance(type_data.get("operators"), list) for type_data in data["objectTypes"].values()):
        raise SystemExit("NumSharp inventory schema mismatch: schemaVersion 6 types must carry 'operators'.")
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
    """Render a NumPy object's signature for a catalog row, without calling it.

    :param obj: the NumPy callable (function, method, slot wrapper, ufunc, class).
    :param fallback_name: the member's name; a docstring's first line is used only when it names it or
        starts with ``(``.
    :returns: ``inspect.signature``'s text, else the docstring's first line, else ``ufunc(nin=, nout=)`` for a
        ufunc, else SIGNATURE_UNAVAILABLE (object_surfaces.scalar_type_rows recognises that text).
    """
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
    return SIGNATURE_UNAVAILABLE


# Runtime types of numpy.typing's aliases: NDArray is a types.GenericAlias, ArrayLike and DTypeLike are
# typing.TypeAliasType instances (Python 3.12+; absent on older Pythons, hence the filter).
_TYPE_ALIAS_TYPES = tuple(alias for alias in (types.GenericAlias, getattr(typing, "TypeAliasType", None))
                          if alias is not None)


def numpy_kind(np: Any, obj: Any) -> str:
    """Classify a NumPy export as module, ufunc, class, function or constant.

    The kind decides scope: callables (function/ufunc) are API rows that count in the coverage
    percentages, while classes, constants and modules are supporting rows (see is_api_row).

    :param np: the imported numpy module (its ``ufunc`` type identifies ufunc exports).
    :param obj: the exported object.
    :returns: one of "module", "ufunc", "class", "function", "constant".
    """
    if inspect.ismodule(obj):
        return "module"
    if isinstance(obj, np.ufunc):
        return "ufunc"
    if inspect.isclass(obj):
        return "class"
    # A typing alias is a type for annotations, not a function: a GenericAlias is callable (calling it
    # instantiates its origin), so without this check numpy.typing.NDArray would count as an API function.
    if isinstance(obj, _TYPE_ALIAS_TYPES):
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

    for surface in HEADLINE_SUBMODULES:
        module = getattr(np, surface)
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

    Importing numpy.matlib emits NumPy's own PendingDeprecationWarning; it is suppressed here because
    cataloguing a deprecated module is not using it, and a test run with warnings enabled must not
    report the catalog's scan as a NumPy deprecation.

    :param np: the imported numpy module.
    :param path: dotted submodule path relative to numpy (e.g. "polynomial.chebyshev").
    :returns: the resolved module object, or None when it is absent (never raises).
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", (DeprecationWarning, PendingDeprecationWarning))
            return importlib.import_module("numpy." + path)
    except Exception:
        obj: Any = np
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                return None
        return obj if inspect.ismodule(obj) else None


def numsharp_types_by_simple_name(inventory: dict[str, Any]) -> dict[str, str]:
    """Index NumSharp's exported types by simple name, for crediting a NumPy CLASS to a ported type.

    A NumPy class export is credited to the NumSharp type of the same simple name (numpy.random.Generator
    -> NumSharp.Generator, numpy.exceptions.AxisError -> NumSharp.AxisError). The match is by type, never
    by a member namesake, so it cannot credit a function. A simple name exported twice prefers the root
    NumSharp.<Name>; nested types ("Outer+Inner") index by their innermost name.

    :param inventory: the reflected NumSharp inventory; its ``exportedTypes`` list is read (absent -> empty).
    :returns: simple type name -> full CLR type name.
    """
    numsharp_types_by_simple: dict[str, str] = {}
    for full in inventory.get("exportedTypes", []):
        simple = full.rsplit(".", 1)[-1].split("+")[-1]
        if simple not in numsharp_types_by_simple or full == f"NumSharp.{simple}":
            numsharp_types_by_simple[simple] = full
    return numsharp_types_by_simple


def submodule_public_names(module: Any, path: str, surface: str) -> list[str]:
    """List the public names of one extended submodule: its own contract plus what NumPy documents.

    The module's ``__all__`` is its contract; a module without one (or with an empty one, as
    numpy.lib.format) falls back to every non-underscore ``dir()`` name. NumPy's ``__all__`` lists are
    not complete, so every name its official documentation index lists directly under the module
    (``numpy.<path>.<name>`` or ``numpy.<surface>.<name>``) and that exists on the pinned build is added:
    numpy.exceptions.RankWarning is documented but absent from that module's ``__all__``. Names the docs
    list only for a newer NumPy (the index tracks latest stable) are skipped by the existence check.
    A name inside a NumPy 1.x compatibility API (COMPAT_EXCLUDED_APIS: numpy.typing.NBitBase) is left
    out, listed or documented.

    The result must not depend on what the process imported before: a submodule is an attribute of
    its package only once something has imported it, so ``hasattr`` alone made numpy.typing.mypy_plugin
    appear or not by test order. A documented submodule is therefore imported here when it exists,
    and one listed in EXCLUDED_PUBLIC_SUBMODULES (mypy_plugin) is never added, imported or not.

    :param module: the resolved submodule.
    :param path: its import path relative to numpy (``lib.scimath``).
    :param surface: its display surface, under which the docs may also index it (``emath``).
    :returns: sorted, de-duplicated names without a leading underscore.
    """
    declared = getattr(module, "__all__", None)
    names = set(declared) if declared else {name for name in dir(module) if not name.startswith("_")}
    for prefix in {f"numpy.{path}.", f"numpy.{surface}."}:
        for api_id in documentation_index():
            name = api_id[len(prefix):] if api_id.startswith(prefix) else ""
            # One segment past the prefix is a module member; more is a member of a member
            # (numpy.ma.MaskedArray.sum), which belongs to an object surface, not to this scan.
            if not name or "." in name or name.startswith("_"):
                continue
            if any(within_namespace(f"{path}.{name}", excluded) for excluded in EXCLUDED_PUBLIC_SUBMODULES):
                continue
            if not hasattr(module, name) and hasattr(module, "__path__"):
                # A documented submodule of a package (only packages have __path__) is reachable once
                # imported; import it now so the outcome is the same on every run.
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", (DeprecationWarning, PendingDeprecationWarning))
                        importlib.import_module(f"{module.__name__}.{name}")
                except ImportError:
                    pass  # documented only for a newer NumPy: no such submodule on the pin
            if hasattr(module, name):
                names.add(name)
    # The compatibility exclusion applies to listed and documented names alike; both spellings of the
    # module (import path and display surface) are checked, as the documentation index may use either.
    return sorted(name for name in names
                  if not compat_excluded(f"numpy.{path}.{name}") and not compat_excluded(f"numpy.{surface}.{name}"))


def extended_surface_rows(
    np: Any, inventory: dict[str, Any], seen_ids: set[str],
    members_by_surface: dict[str, list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Catalog every public member of the NumPy submodules the headline scope excludes.

    These rows carry in_default_scope=False, so they never move the headline percentage; their only
    job is to make a scan SEE families (numpy.emath, numpy.polynomial.*, numpy.lib.stride_tricks,
    numpy.ma, numpy.char/strings, numpy.testing, numpy.dtypes, numpy.exceptions, numpy.typing, ...)
    that the five-surface enumerator is structurally blind to. The member list is
    submodule_public_names (``__all__`` plus the names NumPy documents under the module).

    Callables become API rows. Classes, constants and modules (numpy.ma.masked, numpy.ma.nomask, the
    numpy.polynomial.<basis> domain constants, numpy.dtypes.Float64DType, numpy.exceptions.AxisError)
    become supporting rows, which is_api_row keeps out of every percentage; until generator 1.13.0
    they had no rows at all, so NumSharp's own np.ma.masked or np.dtypes.Float64DType read as
    NumSharp-only extensions instead of crediting NumPy's names. A class an object surface owns
    (numpy.ma.MaskedArray, numpy.char.chararray, ...) is left to that surface.

    A member is credited "available" ONLY when NumSharp exposes a member of the SAME name on a
    [ModuleName]-annotated facade for that exact submodule (np.ma, np.dtypes, np.polynomial.chebyshev,
    ...) — never via a top-level np namesake, whose semantics differ (numpy.emath.sqrt(-1) == 1j while
    np.sqrt(-1) == nan). A real class without such a member is credited when NumSharp exports a type
    of the same simple name (numpy.exceptions.AxisError -> NumSharp.AxisError), the same type rule
    resolve_rows applies to headline class exports; availability then reads "type".

    :param np: the imported numpy module (the oracle surface).
    :param inventory: the reflected NumSharp inventory (its ``modules`` map supplies facade members,
        its ``exportedTypes`` list the ported types).
    :param seen_ids: the running set of emitted row ids; mutated to keep ids globally unique.
    :param members_by_surface: facade members by surface from member_maps; computed when None.
    :returns: the list of extended catalog rows (may be empty if a NumPy build ships none of them).
    :raises SystemExit: when a listed submodule is absent from the pinned NumPy, so a renamed or
        removed module fails the run instead of silently dropping its family.
    """
    # Top-level NumSharp np members — used ONLY to annotate a namesake (informational), never to credit.
    np_module = inventory["modules"].get("np", {"methods": [], "properties": [], "fields": []})
    np_toplevel = {m["name"] for grp in ("methods", "properties", "fields") for m in np_module[grp]}
    if members_by_surface is None:
        _, members_by_surface, _ = member_maps(inventory)
    numsharp_types_by_simple = numsharp_types_by_simple_name(inventory)

    rows: list[dict[str, Any]] = []

    def emit(surface: str, name: str, kind: str, category: str, disposition: str,
             host_module: str, obj: Any, row_id: str | None = None) -> None:
        # Row id mirrors the NumPy dotted path so it is stable and collision-free across submodules; a
        # family's own module row passes its id (numpy.matlib), which is not <surface>.<name>.
        row_id = row_id or f"numpy.{surface}.{name}"
        if row_id in seen_ids:  # a name can appear under both a module and its re-export; keep the first.
            return
        seen_ids.add(row_id)
        # Credit only a same-surface facade member. The top-level namesake is recorded as prose so the
        # emath.sqrt / char.upper false-positive class is visible, not hidden.
        matched = next((member for member in members_by_surface.get(surface_for_module(host_module), [])
                        if member["name"] == name), None)
        # A real class (not a typing alias: numpy.typing.NDArray must not match NumSharp.NDArray by name)
        # without a facade member may be ported as a NumSharp type of the same simple name.
        type_target = (numsharp_types_by_simple.get(name)
                       if matched is None and kind == "class" and inspect.isclass(obj) else None)
        if matched:
            note = "Matched against the same NumPy submodule facade; behavioral parity requires tests."
            target = matched["target"]
            signatures = matched["signatures"]
            obsolete = matched.get("obsolete", False)
            source_paths = matched.get("sourcePaths", [])
            source_urls = matched.get("sourceUrls", [])
        elif type_target:
            note = f"NumSharp exports the {name} type ({type_target})."
            target = type_target
            signatures = [f"public class {name}"]
            obsolete = False
            source_paths = locate_type_source(name)
            source_urls = SourceLocator.github_urls(source_paths)
        else:
            note = DISPOSITION_NOTE[disposition]
            target, signatures, obsolete, source_paths, source_urls = None, [], False, [], []
        if not matched and not type_target and name in np_toplevel:
            note += f" NumSharp has a top-level np.{name} (distinct surface/semantics; not credited here)."
        credited = bool(matched or type_target)
        rows.append({
            "id": row_id, "origin": "numpy", "surface": surface, "name": name, "kind": kind,
            "numpy_signature": numpy_signature(obj, name),
            "documentation_url": documentation_url(surface, name, kind), "in_default_scope": False,
            "extended": True, "disposition": disposition,
            "category": category, "availability": "exact" if matched else "type" if type_target else "missing",
            "support": "declared" if credited else "missing",
            "status": "available" if credited else "missing",
            "numsharp_target": target,
            "numsharp_signatures": signatures,
            "numsharp_obsolete": obsolete,
            "numsharp_source_paths": source_paths,
            "numsharp_source_urls": source_urls, "notes": note,
        })

    for path, surface, category, disposition in EXTENDED_SUBMODULES:
        module = resolve_submodule(np, path)
        if module is None:
            raise SystemExit(f"Required NumPy catalog submodule is absent: numpy.{path}")
        host_module = "np." + surface  # the [ModuleName] a NumSharp facade for this family would carry.
        # The family module's own row, unless another scan already emitted it: public_exports lists only
        # the modules in numpy.__all__, and a parent family lists only the submodules in its __all__, so
        # numpy.matlib, numpy.version, numpy.ma.mrecords and numpy.testing.extbuild had no row of their
        # own. It is credited like any member, by a member of that name on the parent's facade.
        parent, _, leaf = surface.rpartition(".")
        emit(surface, leaf, "module", category, disposition, "np." + parent if parent else "np", module,
             row_id=f"numpy.{surface}")
        for name in submodule_public_names(module, path, surface):
            obj = getattr(module, name)
            # A module another package owns (``import warnings`` surfacing through a dir() fallback)
            # is not NumPy API; a NumPy submodule a module exports (numpy.ma.core) is a module row.
            if inspect.ismodule(obj) and not obj.__name__.startswith("numpy"):
                continue
            # A class an object surface owns gets its class row and every member row there, credited
            # through the explicit CLR owner map; emitting it here first would take the id with this
            # scan's category and crediting instead.
            if f"numpy.{surface}.{name}" in OBJECT_SURFACE_PATHS:
                continue
            emit(surface, name, numpy_kind(np, obj), category, disposition, host_module, obj)

    # ndarray's Python protocols (public_exports drops every '_'-prefixed member) are not emitted here:
    # the ndarray object surface lists all of them under their real ids, credited against NDArray.
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


def lists_as_numsharp_only(surface: str, member: dict[str, Any], consumed_targets: set[str]) -> bool:
    """Tell whether a member of an annotated NumSharp module gets a NumSharp-only row.

    :param surface: the module's surface (``np``, ``ndarray``, ``typing``, ...).
    :param member: the member entry from member_maps (``name`` and ``target`` are read).
    :param consumed_targets: NumSharp targets a NumPy row already credits.
    :returns: False for a member a NumPy row credits, a ``_``-prefixed member, and NumSharp's port of a NumPy
        1.x compatibility API the catalog leaves out (np.typing.NBitBase, kept [Obsolete] like NumPy's): that is
        NumPy's API, not a NumSharp-only extension, so it gets no row of either kind. True otherwise.
    """
    if member["target"] in consumed_targets or member["name"].startswith("_"):
        return False
    return not compat_excluded(f"numpy.{surface}.{member['name']}")


def member_category(class_row: dict[str, Any]) -> str:
    """Return the category the member rows of a class get (object_surfaces.class_member_rows).

    A class row from an extended family carries the family's category (numpy.ma.mvoid: "Masked arrays");
    a top-level class row is categorised "Types", like every supporting class, so its members take the
    class's entry in CATEGORY_OVERRIDES (numpy.errstate: "Floating-point handling") when it has one.

    :param class_row: the class's catalog row.
    :returns: the category for the class's member rows.
    """
    # A row without a category (public_exports' rows, before resolve_rows categorises them) is a namespace
    # export, so it is treated as the top-level class row it will be.
    category = class_row.get("category") or "Types"
    if category == "Types":
        return CATEGORY_OVERRIDES.get(class_row["name"], "Types")
    return category


def resolve_rows(np: Any, inventory: dict[str, Any], overrides: dict[str, Any]) -> tuple[list[dict[str, Any]], set[str]]:
    """Build the full catalog: every NumPy row matched against NumSharp, plus NumSharp-only rows.

    In order: the headline exports (with aliases, support overrides and the stray-host gate), NumSharp-only
    members of the annotated modules, the extended submodule families, the object surfaces (members and
    Python protocols, then per-ufunc methods), NumPy's scalar types, and the members of every other public
    NumPy class; then rows a credit consumed are dropped from the NumSharp-only set and the NumPy-oracle
    evidence is joined.

    :param np: the imported, version-pinned numpy module.
    :param inventory: the reflected NumSharp inventory (load_numsharp_inventory).
    :param overrides: the reviewed overrides (overrides.json: aliases, support, stray allowlist).
    :returns: the rows, sorted (NumPy rows first, by surface and name), and the NumSharp targets credited.
    :raises SystemExit: on a duplicate id, an unbacked surface, a stale override, a stray-host miss, a
        NumSharp-only row without a source link, or an oracle join problem.
    :raises ValueError: from object_surfaces when a credited member has no source evidence or a reviewed
        protocol adapter is stale.
    """
    targets, surfaces, prefixes = member_maps(inventory)
    # Exported-type index (simple name -> full name) for crediting a NumPy CLASS export against a
    # NumSharp type of the same name. A repeated simple name prefers the root NumSharp.<Name>.
    # extended_surface_rows applies the same index to submodule classes.
    numsharp_types_by_simple = numsharp_types_by_simple_name(inventory)
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
            if not lists_as_numsharp_only(surface, member, consumed_targets):
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
        ufunc_namespaces=UFUNC_PROTOCOL_NAMESPACES,
    ))
    # NumPy's scalar types, then every other public NumPy class: each member at the class that defines
    # it. The class-member pass reads the class rows emitted above (namespace exports, extended families,
    # object surfaces), so it runs last.
    rows.extend(scalar_type_rows(np, seen_ids, signature=numpy_signature, documentation_url=documentation_url))
    enrich_object_classes(rows, inventory, ROOT, NUMSHARP_SOURCE_BASE_URL)
    rows.extend(class_member_rows(
        np, list(rows), seen_ids, inventory, root=ROOT, source_base_url=NUMSHARP_SOURCE_BASE_URL,
        signature=numpy_signature, documentation_url=documentation_url,
        types_by_simple_name=numsharp_types_by_simple, category_of=member_category, excluded=compat_excluded,
    ))

    # A facade member used by an extended mapping is no longer a NumSharp-only extension.
    consumed_targets.update(row["numsharp_target"] for row in rows
                            if row["origin"] == "numpy" and row["numsharp_target"])
    rows = [row for row in rows if row["origin"] == "numpy" or row["numsharp_target"] not in consumed_targets]

    # Oracle evidence: join the committed NumPy-oracle contracts onto the FINAL row set (resolution
    # needs every id, including surfaces the dashboard projection later hides). Strict like the gates
    # above: an unresolvable corpus key, a stale map entry or an unreplayed corpus file fails the run,
    # because an artifact that silently dropped evidence would understate parity without anyone
    # noticing, and one that kept a stale mapping would overstate it.
    oracle_map = oracle_evidence.load_map()
    oracle = oracle_evidence.join(rows, oracle_evidence.load_sources(oracle_map), oracle_map,
                                  identical=oracle_evidence.identity_predicate(np))
    if oracle.problems:
        raise SystemExit(
            "NumPy-oracle evidence join failed:\n"
            + "".join(f"  - {problem}\n" for problem in oracle.problems)
            + "Fix coverage/oracle_map.json (or the replay wiring) so every committed contract maps to a catalog row."
        )
    oracle_evidence.attach(rows, oracle)

    rows.sort(key=lambda row: (row["origin"] != "numpy", row["surface"], row["name"].lower(), row["id"]))
    missing_extension_sources = [
        row["id"] for row in rows
        if row["origin"] == "numsharp" and not row["numsharp_source_urls"]
    ]
    if missing_extension_sources:
        raise SystemExit("NumSharp-only APIs without a source link: " + ", ".join(missing_extension_sources))
    return rows, consumed_targets


def status_counts(rows: list[dict[str, Any]]) -> dict[str, int | float]:
    """Count availability AND oracle parity evidence over one scope's rows.

    The oracle counts use oracle_evidence.classify, the single definition the dashboard mirrors:
    ``oracle_verified`` = credited rows with committed NumPy-oracle contracts (direct or through an
    identity/reviewed alias); ``oracle_unverified`` = available/partial rows without any;
    ``oracle_uncredited`` = rows the corpus exercises although the catalog does not credit NumSharp with
    them (a finding, never counted as verified). ``oracle_percent`` shares ``coverage_percent``'s
    denominator (every row in scope), so the two read as "exposed" vs "exposed AND proven".

    :param rows: the rows of one scope (headline, expanded, a surface, a category, ...).
    :returns: availability counts/percentages plus the oracle counts/percentage.
    """
    statuses = Counter(row["status"] for row in rows)
    availability = Counter(row["availability"] for row in rows)
    parity = Counter(oracle_evidence.classify(row) for row in rows)
    total = len(rows)
    available = statuses["available"]
    addressed = available + statuses["partial"]
    verified = parity["verified-direct"] + parity["verified-alias"]
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
        "oracle_verified": verified,
        "oracle_direct": parity["verified-direct"],
        "oracle_alias": parity["verified-alias"],
        "oracle_unverified": parity["unverified"],
        "oracle_uncredited": parity["uncredited"],
        "oracle_percent": round(verified * 100 / total, 1) if total else 0.0,
    }


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Count the projected rows for coverage.json, manifest.json and summary.md.

    :param rows: the dashboard rows (dashboard_rows output).
    :returns: the headline (``default_scope``) and expanded (``api_scope``) counts with their per-surface and
        per-category breakdowns, the extended-family and object-surface counts, the ufunc-protocol counts
        (rows marked ``ufunc_protocol``), the Python-protocol counts (rows marked ``python_protocol``, with
        how many NumPy refuses), and the row totals.
    """
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
            "total": sum(bool(row.get("ufunc_protocol")) for row in object_rows),
            "conditional": sum(bool(row.get("ufunc_protocol")) and row.get("applicability") == "conditional"
                               for row in object_rows),
            "not_applicable": sum(bool(row.get("ufunc_protocol")) and row.get("applicability") == "not_applicable"
                                  for row in object_rows),
        },
        # Python protocols ("dunder" members) NumPy defines, on every object surface, scalar type and class.
        "python_protocols": {
            **status_counts([row for row in object_rows if row.get("python_protocol")]),
            "not_applicable": sum(bool(row.get("python_protocol")) and row.get("applicability") == "not_applicable"
                                  for row in object_rows),
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
    # The oracle columns are appended AFTER the historical ones so a consumer indexing by position
    # keeps working; "oracle_parity" is the classify() label (it also covers rows without a record).
    columns = [
        "id", "origin", "surface", "category", "name", "kind", "in_default_scope", "extended", "disposition",
        "object_surface", "object_type", "applicability", "status", "availability",
        "support", "numpy_signature", "numsharp_target", "numsharp_signatures", "numsharp_source_paths",
        "numsharp_source_urls", "numsharp_obsolete", "case_insensitive_matches", "notes", "documentation_url",
        "oracle_parity", "oracle_contracts", "oracle_error_contracts", "oracle_sources", "oracle_keys",
        "oracle_files", "oracle_dtypes", "oracle_layouts", "oracle_pinned_contracts", "oracle_via",
        "oracle_via_rule",
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
        flat["oracle_parity"] = oracle_evidence.classify(row)
        record = row.get("oracle")
        if record:
            via = record.get("via") or {}
            flat.update({
                "oracle_contracts": record["contracts"],
                "oracle_error_contracts": record["error_contracts"],
                "oracle_sources": " | ".join(f"{source}:{count}" for source, count in record["sources"].items()),
                "oracle_keys": " | ".join(record["keys"]),
                "oracle_files": " | ".join(record["files"]),
                "oracle_dtypes": " | ".join(record["dtypes"]),
                "oracle_layouts": record["layouts"],
                "oracle_pinned_contracts": record["pinned_contracts"],
                "oracle_via": via.get("id", ""),
                "oracle_via_rule": via.get("rule", ""),
            })
        writer.writerow(flat)
    return stream.getvalue()


def oracle_markdown(summary: dict[str, Any], rows: list[dict[str, Any]], oracle: dict[str, Any]) -> list[str]:
    """Render summary.md's NumPy-oracle parity section.

    Two lists make the section actionable rather than decorative: headline APIs that are available but
    have no oracle contracts (parity is only declared, the next oracle work), and APIs the corpus gates
    although the catalog does not credit them (a crediting gap: an alias or instance method to add).

    :param summary: the artifact summary (``default_scope``/``api_scope`` carry the oracle counts).
    :param rows: the emitted rows.
    :param oracle: the artifact's top-level ``oracle`` block (oracle_evidence.describe).
    :returns: markdown lines, starting with a blank separator line.
    """
    headline = summary["default_scope"]
    expanded = summary["api_scope"]
    lines = [
        "",
        "## NumPy-oracle parity evidence",
        "",
        f"**{oracle['contracts']:,}** committed NumPy {oracle['numpy_version']} oracle contracts "
        f"(**{oracle['error_contracts']:,}** error-message contracts) are replayed against NumSharp in CI; each "
        "must be bit-exact or a documented, registry-excused divergence. An API counts as **oracle-verified** "
        "when contracts resolve to it directly, or through an identity alias (the same NumPy object mapped to "
        "the same NumSharp member) or a reviewed pure-delegation alias.",
        "",
        f"Headline scope: **{headline['oracle_verified']}** of {headline['total']} APIs oracle-verified "
        f"(**{headline['oracle_percent']:.1f}%**; {headline['oracle_direct']} direct, {headline['oracle_alias']} alias); "
        f"**{headline['oracle_unverified']}** available APIs have no oracle contracts.",
        "",
        f"Expanded scope: **{expanded['oracle_verified']}** of {expanded['total']} APIs oracle-verified "
        f"(**{expanded['oracle_percent']:.1f}%**; {expanded['oracle_direct']} direct, {expanded['oracle_alias']} alias); "
        f"**{expanded['oracle_unverified']}** available APIs have no oracle contracts.",
        "",
        "| Source | Contracts | Error contracts | Host-pinned | Files | Keys | Replayed by |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for source in oracle["sources"].values():
        lines.append(
            f"| {source['label']} (`{source['path']}`) | {source['contracts']:,} | {source['error_contracts']:,} | "
            f"{source['pinned_contracts']:,} | {source['files']} | {source['keys']} | {source['replayed_by']} |"
        )
    labels = {"np": "np.*", "ndarray": "ndarray.*", "random": "np.random.*", "linalg": "np.linalg.*",
              "fft": "np.fft.*", "ma": "np.ma.*", "polynomial": "np.polynomial.*"}
    lines.extend([
        "",
        "Host-pinned contracts are hard-gated only on the pinned host (win-amd64 CRT libm, or the "
        "content-hash-pinned OpenBLAS) and Inconclusive elsewhere.",
        "",
        "Per surface (expanded scope; surfaces with nothing available or gated are omitted). *Verified* "
        "shares the coverage percentage's denominator: every API row of the surface.",
        "",
        "| Surface | Total | Available | Oracle-verified | Direct | Alias | Unverified available | Verified |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for surface, counts in expanded["by_surface"].items():
        if not (counts["available"] or counts["partial"] or counts["oracle_uncredited"]):
            continue
        lines.append(
            f"| {labels.get(surface, surface)} | {counts['total']} | {counts['available']} | {counts['oracle_verified']} | "
            f"{counts['oracle_direct']} | {counts['oracle_alias']} | {counts['oracle_unverified']} | "
            f"{counts['oracle_percent']:.1f}% |"
        )

    unverified = sorted(
        (row for row in rows if row["origin"] == "numpy" and row["in_default_scope"]
         and oracle_evidence.classify(row) == "unverified"),
        key=lambda row: (row["surface"], row["name"].lower()),
    )
    lines.extend([
        "",
        "### Available headline APIs without oracle contracts",
        "",
        "Parity for these is declared, not oracle-proven (some carry dedicated unit suites instead; see "
        "OracleSurfaceCoverageTests' sibling-owned and known-gap classifications).",
        "",
    ])
    if unverified:
        lines.extend(["| API | Surface | Status | Category |", "|---|---|---|---|"])
        for row in unverified:
            api = row["id"].replace("numpy.", "np.", 1).replace("np.ndarray.", "ndarray.", 1)
            lines.append(f"| [`{api}`]({row['documentation_url']}) | {row['surface']} | {row['status']} | {row['category']} |")
    else:
        lines.append("_None: every available headline API has oracle contracts._")

    uncredited = sorted((row for row in rows if oracle_evidence.classify(row) == "uncredited"), key=lambda row: row["id"])
    lines.extend([
        "",
        "### Oracle-gated APIs the catalog does not credit",
        "",
        "The corpus proves NumSharp reproduces these NumPy behaviours, yet no NumSharp member is credited "
        "under the NumPy name (add the member, or record a reviewed alias in overrides.json).",
        "",
    ])
    if uncredited:
        lines.extend(["| API | Status | Contracts | Oracle keys |", "|---|---|---:|---|"])
        for row in uncredited:
            record = row["oracle"]
            keys = ", ".join(f"`{key}`" for key in record["keys"])
            lines.append(f"| `{row['id']}` | {row['status']} | {record['contracts']:,} | {keys} |")
    else:
        lines.append("_None._")
    return lines


def markdown_text(summary: dict[str, Any], rows: list[dict[str, Any]], numpy_version: str, assembly_version: str,
                  oracle: dict[str, Any] | None = None) -> str:
    """Render summary.md: the headline and expanded figures, gaps, near-misses, families and counting rules.

    :param summary: build_summary's output for the same rows.
    :param rows: the dashboard rows.
    :param numpy_version: the NumPy version compared.
    :param assembly_version: the NumSharp assembly version reflected.
    :param oracle: the artifact's oracle block (oracle_evidence.describe), or None to omit the oracle section.
    :returns: the Markdown text.
    """
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
        f"The catalog includes **{summary['ufunc_protocols']['total']} per-ufunc protocol methods** "
        "(each public `numpy.ufunc` method on every ufunc export of the compared namespaces, plus the "
        "methods numpy.ma's masked ufunc wrappers implement); "
        f"**{summary['ufunc_protocols']['not_applicable']}** describe methods whose calls NumPy rejects "
        "for that ufunc. These count as exposed API contracts, not computable operations.",
        "",
        f"It also includes **{summary['python_protocols']['total']} Python protocols** NumPy defines on its "
        "classes (operators, subscripting, conversions, iteration, `with`, pickling), each class listing the "
        "protocols it has; NumSharp provides "
        f"**{summary['python_protocols']['available']}**, through a member of the protocol's own name, the C# "
        "operator, indexer or cast of the same meaning, or a reviewed adapter. "
        f"**{summary['python_protocols']['not_applicable']}** are slots NumPy fills only to refuse every call "
        "(`del a[i]`, ordering comparisons on `ndarray.flags`).",
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
    ])
    # The oracle section sits right under the disclaimer it answers: which APIs those differential
    # tests actually cover. Absent only for callers that render without the oracle block.
    if oracle is not None:
        lines.extend(oracle_markdown(summary, rows, oracle))
    lines.extend([
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
    # miss them. Ranked candidate -> subsystem -> tooling so implementable gaps read first.
    extended = [row for row in rows if row.get("extended")]
    lines.extend([
        "",
        "## Extended NumPy submodules and object contracts",
        "",
        "These surfaces appear in the dashboard by default. `candidate` denotes numeric APIs and "
        "object contracts; `subsystem` denotes a separate object model such as masked/string/record "
        "arrays, ufuncs or NumPy's scalar types; `tooling` denotes Python tooling. "
        "Class existence is catalogued separately from member availability. The historical headline "
        "remains unchanged by these additions.",
        "",
    ])
    if extended:
        disposition_rank = {"candidate": 0, "subsystem": 1, "tooling": 2}
        lines.append("| Surface | Disposition | Available | Missing | Total | Notable missing |")
        lines.append("|---|---|---:|---:|---:|---|")
        surfaces = sorted(
            {row["surface"] for row in extended},
            key=lambda s: (disposition_rank.get(
                next((r["disposition"] for r in extended if r["surface"] == s), "tooling"), 9), s),
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
        "The historical headline is NumPy top-level callables, ndarray public methods/properties, and callables in numpy.random, numpy.linalg, and numpy.fft. The expanded API scope adds public extended-submodule callables and object methods/properties/protocols: the Python protocols (dunder members) NumPy defines, the members of NumPy's scalar types, and the members of every other public NumPy class. Types, constants, modules, and NumSharp-only APIs remain searchable in the full catalog but do not enter either API denominator. Aliases are separate exported names; inherited members are separate contracts on each object type, while NumPy's scalar types and the classes no object surface owns list each member once, at the class that defines it. A ufunc method row records the exposed protocol even when that ufunc rejects the operation (for example, unary reductions), and a protocol row records a slot NumPy fills only to refuse it. Matching a namespace function never credits a ufunc or a different object type's member. APIs NumPy keeps only for NumPy 1.x code and has itself deprecated (numpy.matlib, numpy.typing.NBitBase) are left out; numpy.lib and numpy.testing are catalogued but not shown, except numpy.lib.NumpyVersion.",
        "",
    ])
    return "\n".join(lines)


def render_outputs(np: Any, inventory: dict[str, Any], overrides: dict[str, Any]) -> dict[str, str]:
    rows, _ = resolve_rows(np, inventory, overrides)
    rows = dashboard_rows(rows)
    summary = build_summary(rows)
    # The source totals come from the memoized scan resolve_rows already ran (no second corpus pass);
    # the row counts are read from the PROJECTED rows, the same set every other count uses.
    oracle_map = oracle_evidence.load_map()
    oracle = oracle_evidence.describe(oracle_evidence.load_sources(oracle_map), rows, np.__version__)
    payload = {
        "schema_version": 1,
        "generator_version": GENERATOR_VERSION,
        "numpy_version": np.__version__,
        "numsharp_assembly_version": inventory["assemblyVersion"],
        "methodology": {
            "headline": "Available default-scope APIs divided by all default-scope NumPy APIs.",
            "default_scope": "Top-level NumPy callables; ndarray public methods and properties; callable exports of numpy.random, numpy.linalg, and numpy.fft.",
            "api_scope": "Dashboard default: headline APIs plus extended public submodule callables and object methods/properties/protocols. Supporting class, constant and module exports are catalogued but excluded from API percentages.",
            "object_members": "Object members match only their explicit NumSharp owner type. Inherited members and ufunc protocols are catalogued per owner; NumPy's scalar types and the other public classes list each member once, at the class that defines it. A Python protocol is credited by a NumSharp member of its own name, the C# operator, indexer or cast of the same meaning, or a reviewed adapter. Namespace namesakes never confer support. Protocol existence does not imply applicability to every ufunc or object.",
            "availability_note": "Compiled API availability is distinct from fully verified behavioral parity.",
            "case_sensitivity": "NumPy API names are matched case-sensitively for parity. Case-insensitive near-misses are detected and reported (row field 'case_insensitive_matches'; the 'Case-insensitive near-misses' section of summary.md) but never counted as available.",
            "oracle": "Behavioral parity evidence: committed NumPy 2.4.2 oracle contracts (the differential-fuzz corpus plus the .npy, flags and layout oracles), each replayed against NumSharp in CI and required to be bit-exact or a documented, registry-excused divergence. Contracts resolve to rows through coverage/oracle_map.json; a row is oracle-verified when it is credited and has direct contracts, an identity alias (same NumPy object, same NumSharp member) or a reviewed pure-delegation alias. Host-pinned contracts are hard-gated only on the pinned host.",
        },
        "oracle": oracle,
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
        "oracle_map": "coverage/oracle_map.json",
        "oracle": oracle,
        "summary": summary,
    }
    return {
        "coverage.json": json_text(payload),
        "coverage.csv": csv_text(rows),
        "summary.md": markdown_text(summary, rows, np.__version__, inventory["assemblyVersion"], oracle),
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
        payload = json.loads(rendered["coverage.json"])
        summary = payload["summary"]
        expanded = summary["api_scope"]
        headline = summary["default_scope"]
        print(
            f"Wrote {args.output}: {summary['catalog_rows']} catalog rows; "
            f"expanded {expanded['available']}/{expanded['total']} available ({expanded['coverage_percent']:.1f}%); "
            f"headline {headline['available']}/{headline['total']} ({headline['coverage_percent']:.1f}%)."
        )
        print(
            f"NumPy oracle: {payload['oracle']['contracts']:,} contracts; oracle-verified "
            f"headline {headline['oracle_verified']}/{headline['total']} ({headline['oracle_percent']:.1f}%), "
            f"expanded {expanded['oracle_verified']}/{expanded['total']} ({expanded['oracle_percent']:.1f}%); "
            f"{headline['oracle_unverified']} available headline APIs lack contracts."
        )


if __name__ == "__main__":
    main()
