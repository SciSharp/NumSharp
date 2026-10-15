"""Inventory public NumPy object contracts separately from namespace exports.

The NumPy class/runtime instance is authoritative for member discovery. CLR matching
uses an explicit owner map; sharing a name with np or NDArray does not implement a
ufunc, MaskedArray, matrix, or random.Generator member. These rows never contribute
to the five-surface headline. No random draw, file-system IO, or global RNG mutation
is required to discover the instance-only fields in NumPy's public objects.

Ufunc protocol methods (``add.reduce``, ``maximum.accumulate``, ``add.resolve_dtypes``, ...)
are catalogued per export, from the pinned NumPy itself: the method list is read from the
``numpy.ufunc`` type, and the scan covers every namespace the generator compares, so a
ufunc exported from a submodule (``numpy.strings.str_len``) and a ufunc-like wrapper
(numpy.ma's masked ``ma.add``) are inventoried like the top-level ufuncs.
"""

from __future__ import annotations

import importlib
import inspect
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class ObjectSurface:
    path: str
    category: str
    clr_type: str | None = None
    disposition: str = "candidate"
    protocols: tuple[str, ...] = ()


OBJECT_SURFACES = (
    ObjectSurface("ufunc", "Universal function objects", disposition="subsystem",
                  protocols=("__call__", "__name__", "__doc__")),
    ObjectSurface("random.Generator", "Random generators", "NumSharp.Generator"),
    ObjectSurface("random.RandomState", "Legacy random state", "NumSharp.NumPyRandom"),
    ObjectSurface("random.SeedSequence", "Random seeding", "NumSharp.SeedSequence"),
    ObjectSurface("random.BitGenerator", "Random bit generators", "NumSharp.BitGenerator"),
    ObjectSurface("random.PCG64", "Random bit generators", "NumSharp.PCG64"),
    ObjectSurface("random.PCG64DXSM", "Random bit generators", "NumSharp.PCG64DXSM"),
    ObjectSurface("random.MT19937", "Random bit generators", "NumSharp.MT19937"),
    ObjectSurface("random.Philox", "Random bit generators", "NumSharp.Philox"),
    ObjectSurface("random.SFC64", "Random bit generators", "NumSharp.SFC64"),
    ObjectSurface("nditer", "Array iterators", "NumSharp.np+NDIterator"),
    ObjectSurface("flatiter", "Array iterators", "NumSharp.np+FlatIterator"),
    ObjectSurface("ndindex", "Array iterators", "NumSharp.np+NDIndex",
                  protocols=("__iter__", "__next__")),
    ObjectSurface("ndenumerate", "Array iterators", "NumSharp.np+NDEnumerate",
                  protocols=("__iter__", "__next__")),
    ObjectSurface("broadcast", "Array iterators", "NumSharp.np+Broadcast"),
    ObjectSurface("dtype", "Dtype objects", "NumSharp.DType"),
    ObjectSurface("finfo", "Machine limits", "NumSharp.finfo"),
    ObjectSurface("iinfo", "Machine limits", "NumSharp.iinfo"),
    ObjectSurface("ndarray.flags", "Array flags", "NumSharp.NDArrayFlags"),
    ObjectSurface("poly1d", "Polynomials", "NumSharp.poly1d"),
    ObjectSurface("polynomial.polynomial.Polynomial", "Polynomial objects"),
    ObjectSurface("polynomial.chebyshev.Chebyshev", "Polynomial objects"),
    ObjectSurface("polynomial.legendre.Legendre", "Polynomial objects"),
    ObjectSurface("polynomial.hermite.Hermite", "Polynomial objects"),
    ObjectSurface("polynomial.hermite_e.HermiteE", "Polynomial objects"),
    ObjectSurface("polynomial.laguerre.Laguerre", "Polynomial objects"),
    # NumSharp's numpy.ma implementation: NDMaskedArray is a standalone class (no NDArray base),
    # so every member it credits is its own masked-array code, never an inherited ndarray member.
    ObjectSurface("ma.MaskedArray", "Masked arrays", "NumSharp.NDMaskedArray"),
    ObjectSurface("matrix", "Matrix objects", "NumSharp.NDMatrix"),
    ObjectSurface("recarray", "Record arrays", disposition="subsystem"),
    ObjectSurface("record", "Record arrays", disposition="subsystem"),
    ObjectSurface("char.chararray", "String array objects", disposition="subsystem"),
    ObjectSurface("memmap", "Memory-mapped arrays", disposition="subsystem"),
    ObjectSurface("lib.npyio.DataSource", "Data sources"),
    ObjectSurface("lib.npyio.NpzFile", "Archive objects", "NumSharp.IO.NpzFile"),
    ObjectSurface("__array_namespace_info__", "Array API introspection"),
    ObjectSurface("testing.suppress_warnings", "Test support", disposition="tooling"),
    # np.vectorize returns NumSharp.Vectorized: the object NumPy's vectorize instances correspond to.
    ObjectSurface("vectorize", "Function wrappers", "NumSharp.Vectorized",
                  protocols=("__call__",)),
    ObjectSurface("lib.NumpyVersion", "Runtime & diagnostics", disposition="tooling"),
)

OBJECT_SURFACE_PATHS = tuple("numpy." + surface.path for surface in OBJECT_SURFACES)

# Reviewed spelling adapters, not case-folded matching. Method/property shape must
# still agree. Mapping-like Keys/Values properties cannot satisfy keys()/values().
MEMBER_ALIASES = {
    ("flatiter", "base"): "Base",
    ("lib.npyio.NpzFile", "files"): "Files",
    ("lib.npyio.NpzFile", "allow_pickle"): "AllowPickle",
    ("lib.npyio.NpzFile", "max_header_size"): "MaxHeaderSize",
    ("lib.npyio.NpzFile", "zip"): "Zip",
    ("lib.npyio.NpzFile", "close"): "Close",
}


# The ufunc methods whose call NumPy refuses for some ufunc SHAPES (arity, output count,
# core signature). Any other method (resolve_dtypes, or one a future NumPy adds) gets no
# shape rule: resolve_dtypes resolves loop dtypes for every shape, and an unknown method
# must not inherit rules nobody reviewed for it.
_SHAPE_RULED_UFUNC_METHODS = frozenset({"reduce", "accumulate", "reduceat", "outer", "at"})

# Shared tail of every protocol row's note: why NumSharp can never match the member today.
_NO_UFUNC_OBJECT = ("NumSharp exposes NumPy's ufuncs as functions, not first-class ufunc objects, "
                    "so no object member can match; related np.sum/cumsum/outer operations do not "
                    "implement it. Catalogued outside the headline scope.")


def ufunc_protocol_methods(np: Any) -> tuple[str, ...]:
    """Return the public methods of NumPy's ufunc type, read from the pinned NumPy.

    Reading the type instead of listing names by hand keeps the per-export rows complete:
    the hand-written list held five names and missed ``resolve_dtypes`` (NumPy 1.24+), so
    ``numpy.add.resolve_dtypes`` had no row. A method a later NumPy adds appears on the next
    generation the same way. Properties (``identity``, ``nin``, ``nout``, ``nargs``, ``ntypes``,
    ``types``, ``signature``) are excluded: each is the same contract on every ufunc and is
    catalogued once as a ``numpy.ufunc.<name>`` object member.

    :param np: the imported, version-pinned ``numpy`` module.
    :returns: the method names, sorted; at NumPy 2.4.2 ``("accumulate", "at", "outer",
        "reduce", "reduceat", "resolve_dtypes")``.
    """
    # getattr_static reads the descriptor on the type: a method descriptor is callable,
    # a getset descriptor (identity, nin, ...) is not, and nothing is evaluated.
    return tuple(sorted(
        name for name in dir(np.ufunc)
        if not name.startswith("_") and callable(inspect.getattr_static(np.ufunc, name))
    ))


def ufunc_method_rejection(ufunc: Any, method: str) -> str | None:
    """Return why NumPy rejects every call of ``method`` on ``ufunc``, or None when it may succeed.

    The rules come from NumPy's ufunc method entry points (probed at 2.4.2): reductions,
    accumulations and ``outer`` need a binary ufunc; reductions, accumulations and ``at`` need a
    single output (``outer`` returns one array per output); ``at`` takes only unary or binary
    ufuncs; and no method but ``resolve_dtypes`` runs on a generalized ufunc with a core
    signature. A None result only means the shape is accepted: operands, dtypes and casting can
    still make a given call fail.

    :param ufunc: a ``numpy.ufunc`` instance.
    :param method: a ufunc method name from ``ufunc_protocol_methods``.
    :returns: the rejection text for the row's note, or None.
    """
    if method not in _SHAPE_RULED_UFUNC_METHODS:
        return None
    if ufunc.signature is not None:
        return "NumPy rejects this method for generalized ufuncs with a core signature."
    if method != "at" and ufunc.nin != 2:
        return "NumPy requires a binary ufunc for this method."
    if method != "outer" and ufunc.nout != 1:
        return "NumPy requires a single-output ufunc for this method."
    if method == "at" and ufunc.nin not in {1, 2}:
        return "NumPy at supports only unary and binary ufuncs."
    return None


def _public_names(module: Any, top_level: bool) -> list[str]:
    """List the public export names of one namespace, using the rule its catalog rows use.

    :param module: the namespace module (numpy itself or a submodule).
    :param top_level: True for numpy itself, whose historical per-ufunc rows walk ``vars(np)``.
    :returns: sorted, de-duplicated names without a leading underscore.
    """
    if top_level:
        names = vars(module)
    else:
        # Submodule rows (generate_coverage.public_exports / extended_surface_rows) come from
        # __all__ when the module declares it, else from dir(): use the same list so every
        # protocol row's owner export is itself a catalogued row.
        names = getattr(module, "__all__", None) or dir(module)
    return sorted({name for name in names if not name.startswith("_")})


def ufunc_protocol_rows(
    np: Any, seen_ids: set[str], namespaces: Sequence[tuple[str, str]], *,
    signature: Callable[[Any, str], str],
    documentation_url: Callable[[str, str, str], str] | None = None,
) -> list[dict[str, Any]]:
    """Return one row per ufunc method on every ufunc-like export of the given namespaces.

    Two kinds of export carry NumPy's ufunc protocol:

    * ``numpy.ufunc`` instances. Every public method of the type exists on each of them,
      whether or not their shape allows a call, so each gets one row per method; the row's
      ``applicability`` is ``not_applicable`` when NumPy rejects every call (``sqrt.reduce``)
      and ``conditional`` otherwise.
    * Wrapper objects that implement part of the protocol: numpy.ma's masked ufuncs. ``ma.add``
      has reduce, accumulate and outer; the comparisons only accumulate and outer, because
      NumPy sets their ``reduce`` to None; ``ma.maximum``/``ma.minimum`` have reduce and outer;
      the domained and unary wrappers have none. Only methods that are callable get rows.

    Every row is ``missing``: NumSharp exposes ufuncs as functions, so no NumSharp member can
    implement an object member. Rows never enter the headline scope.

    :param np: the imported, version-pinned ``numpy`` module.
    :param seen_ids: ids emitted so far; updated in place, and an id already present is skipped
        (the first emitter of an id keeps it, as everywhere in the catalog).
    :param namespaces: ``(import path relative to numpy, display surface)`` pairs. The empty path
        is numpy itself, whose rows keep the historical ``ufunc`` surface; a submodule's rows use
        its display surface (``("lib.scimath", "emath")`` yields ``numpy.emath.<name>.<method>``).
    :param signature: renders a NumPy callable's signature (generate_coverage.numpy_signature).
    :param documentation_url: resolves a documented URL or its nearest documented parent; None
        links the ufunc reference page.
    :returns: the new rows, in namespace order, then export name, then method name.
    :raises ValueError: when a namespace path does not resolve to a module, so a misspelt or
        removed submodule fails the run instead of silently dropping its rows.
    """
    methods = ufunc_protocol_methods(np)
    rows: list[dict[str, Any]] = []
    for path, surface in namespaces:
        try:
            module = _resolve(np, path) if path else np
        except (AttributeError, ImportError) as error:
            # _resolve imports numpy.<path> when the attribute walk fails; a namespace the
            # generator lists but NumPy lacks must fail the run with the namespace's name.
            raise ValueError(f"Ufunc protocol namespace numpy.{path} does not resolve.") from error
        if not inspect.ismodule(module):
            raise ValueError(f"Ufunc protocol namespace numpy.{path} is not a module.")
        for name in _public_names(module, top_level=not path):
            obj = getattr(module, name, None)
            if isinstance(obj, np.ufunc):
                implemented = methods
            elif obj is None or inspect.isclass(obj) or inspect.ismodule(obj):
                # Classes (MaskedArray, chararray) are object surfaces of their own, and a
                # module is a namespace; neither is a ufunc-like callable.
                continue
            else:
                # getattr_static sees the INSTANCE override NumPy uses to remove a method
                # (numpy.ma sets equal.reduce = None) and evaluates no property.
                implemented = tuple(method for method in methods
                                    if callable(inspect.getattr_static(obj, method, None)))
                if not implemented:
                    continue
            owner = f"numpy.{surface}.{name}" if path else f"numpy.{name}"
            for method in implemented:
                row_id = f"{owner}.{method}"
                if row_id in seen_ids:
                    continue
                seen_ids.add(row_id)
                if isinstance(obj, np.ufunc):
                    rejection = ufunc_method_rejection(obj, method)
                    if rejection:
                        note = rejection
                    elif method in _SHAPE_RULED_UFUNC_METHODS:
                        note = "NumPy permits this protocol shape; supported operands and dtype resolution still apply."
                    elif method == "resolve_dtypes":
                        note = ("NumPy resolves loop dtypes for every ufunc shape; it raises only for dtype "
                                "combinations the ufunc has no loop for.")
                    else:
                        note = ("Discovered on numpy.ufunc without a reviewed shape rule, so it is recorded "
                                "as conditional.")
                    doc_url = (documentation_url("ufunc", method, "method") if documentation_url
                               else "https://numpy.org/doc/stable/reference/ufuncs.html")
                else:
                    rejection = None
                    note = (f"{owner} is a ufunc wrapper object that implements this method; NumPy's "
                            "wrappers implement only some of the ufunc methods.")
                    # No generated page exists for a wrapper's method: link the nearest
                    # documented parent (never a fabricated URL).
                    doc_url = (documentation_url(surface, f"{name}.{method}", "method") if documentation_url
                               else "https://numpy.org/doc/stable/reference/ufuncs.html")
                rows.append({
                    "id": row_id, "origin": "numpy", "surface": surface if path else "ufunc",
                    "name": f"{name}.{method}", "kind": "method",
                    "numpy_signature": signature(getattr(obj, method), method),
                    "documentation_url": doc_url,
                    "in_default_scope": False, "extended": True, "object_surface": True,
                    "object_type": owner, "disposition": "subsystem",
                    "category": "Universal function objects", "availability": "missing",
                    "support": "missing", "status": "missing", "numsharp_target": None,
                    "numsharp_signatures": [], "numsharp_obsolete": False,
                    "numsharp_source_paths": [], "numsharp_source_urls": [],
                    "applicability": "not_applicable" if rejection else "conditional",
                    "notes": note + " " + _NO_UFUNC_OBJECT,
                })
    return rows


def _resolve(np: Any, path: str) -> Any:
    current = np
    for part in path.split("."):
        try:
            current = getattr(current, part)
        except AttributeError:
            current = importlib.import_module(current.__name__ + "." + part)
    return current


def _samples(np: Any) -> tuple[dict[str, Any], Any]:
    """Instances expose finfo/iinfo and archive fields absent from class dir()."""
    buffer = io.BytesIO()
    np.savez(buffer, sample=np.array([0], dtype=np.int64))
    buffer.seek(0)
    archive = np.load(buffer, allow_pickle=False)
    samples = {
        "ufunc": np.add,
        "dtype": np.dtype(np.float64),
        "finfo": np.finfo(np.float64),
        "iinfo": np.iinfo(np.int64),
        "ndarray.flags": np.array([0], dtype=np.int64).flags,
        "ndenumerate": np.ndenumerate(np.array([0], dtype=np.int64)),
        "random.SeedSequence": np.random.SeedSequence(0),
        "lib.npyio.NpzFile": archive,
        "vectorize": np.vectorize(lambda value: value),
        "lib.NumpyVersion": np.lib.NumpyVersion(np.__version__),
    }
    for spec in OBJECT_SURFACES:
        if spec.path.startswith("polynomial."):
            samples[spec.path] = _resolve(np, spec.path)([1, 0])
    return samples, archive


class _SourceIndex:
    """Locate the declaring CLR type and member; retain inherited ownership."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.files = [
            (path, path.read_text(encoding="utf-8-sig"))
            for path in sorted((root / "src" / "NumSharp.Core").rglob("*.cs"),
                               key=lambda path: path.as_posix())
            if not {"obj", "bin"}.intersection(path.parts)
        ]
        self.type_files: dict[str, list[tuple[Path, str]]] = {}

    def for_type(self, clr_type: str) -> list[tuple[Path, str]]:
        if clr_type not in self.type_files:
            namespace, _, nested_name = clr_type.rpartition(".")
            # The namespace and every containing type disambiguate nested CLR
            # hosts such as np.NDIterator from similarly named engine types.
            namespace_pattern = re.compile(r"\bnamespace\s+" + re.escape(namespace) + r"\b")
            patterns = [re.compile(r"\b(?:class|struct|interface)\s+" +
                                   re.escape(name.split("`")[0]) + r"\b")
                        for name in nested_name.split("+")]
            self.type_files[clr_type] = [
                (path, text) for path, text in self.files
                if namespace_pattern.search(text) and all(p.search(text) for p in patterns)
            ]
        return self.type_files[clr_type]

    def locate(self, clr_type: str, member: dict[str, Any] | None = None) -> list[str]:
        owners = member.get("declaringTypes", [clr_type]) if member else [clr_type]
        paths: set[str] = set()
        declaration = None
        if member:
            suffix = r"\s*(?:<[^>]+>)?\s*\(" if member["kind"] == "method" else r"\s*(?:=>|\{|;|=)"
            declaration = re.compile(r"\bpublic\s+[^;{}\n]*?\b@?" +
                                     re.escape(member["name"]) + suffix)
        for owner in owners:
            for path, text in self.for_type(owner):
                if declaration is None or declaration.search(text):
                    paths.add(path.relative_to(self.root).as_posix())
        return sorted(paths)


def enrich_object_classes(
    rows: list[dict[str, Any]], inventory: dict[str, Any], root: Path, source_base_url: str,
) -> int:
    """Repair existing missing class rows using the reviewed object-owner map.

    Namespace exports are emitted before object members. A preexisting class row
    therefore wins duplicate-ID suppression, even when its CLR counterpart has a
    different name (numpy.flatiter -> NumSharp.np.FlatIterator). Enrich only missing
    class rows; retain their scope, identity, category and explicit support rules.
    """
    object_types = inventory.get("objectTypes")
    if not isinstance(object_types, dict):
        raise ValueError("Object inventory requires schema 5's 'objectTypes' map.")
    specs = {"numpy." + spec.path: spec for spec in OBJECT_SURFACES if spec.clr_type}
    sources = _SourceIndex(root)
    enriched = 0
    for row in rows:
        if row.get("origin") != "numpy" or row.get("kind") != "class" or row.get("status") != "missing":
            continue
        spec = specs.get(row["id"])
        if spec is None or spec.clr_type not in object_types:
            continue
        paths = sources.locate(spec.clr_type)
        target = spec.clr_type.replace("+", ".")
        if not paths:
            raise ValueError(f"Object class {row['id']} matched {target} without source evidence.")
        note = f"Class availability uses the explicit corresponding CLR object type {target}."
        row.update({
            "availability": "exact", "support": "declared", "status": "available",
            "numsharp_target": target, "numsharp_signatures": ["public class " + target],
            "numsharp_obsolete": False, "numsharp_source_paths": paths,
            "numsharp_source_urls": [source_base_url + path for path in paths],
            "notes": (row.get("notes", "") + " " + note).strip(),
        })
        enriched += 1
    return enriched


def object_surface_rows(
    np: Any, inventory: dict[str, Any], seen_ids: set[str], *, root: Path,
    source_base_url: str, signature: Callable[[Any, str], str],
    documentation_url: Callable[[str, str, str], str] | None = None,
    ufunc_namespaces: Sequence[tuple[str, str]] = (),
) -> list[dict[str, Any]]:
    """Return complete extended coverage rows, updating ``seen_ids`` in place.

    ``objectTypes`` must come from inventory schema 5. It is keyed by exact CLR
    full name and carries TypeInventory methods/properties/fields, including
    inherited public members and their ``declaringTypes`` for source evidence.

    :param np: the imported, version-pinned ``numpy`` module.
    :param inventory: the NumSharp reflection inventory (schema 5).
    :param seen_ids: ids emitted so far; updated in place.
    :param root: repository root, for source-file evidence.
    :param source_base_url: URL prefix that turns a repository path into a source link.
    :param signature: renders a NumPy callable's signature.
    :param documentation_url: resolves a documented URL or its nearest parent; None uses
        stable family pages.
    :param ufunc_namespaces: submodules, besides numpy itself, whose ufunc-like exports get
        per-method protocol rows: ``(import path relative to numpy, display surface)`` pairs,
        normally every namespace the generator compares (``UFUNC_PROTOCOL_NAMESPACES``).
    :returns: the object-member rows, then the ufunc protocol rows.
    :raises ValueError: when the inventory lacks schema 5's ``objectTypes``, when a matched
        member has no source evidence, or when a ufunc namespace does not resolve.
    """
    object_types = inventory.get("objectTypes")
    if not isinstance(object_types, dict):
        raise ValueError("Object inventory requires schema 5's 'objectTypes' map.")
    sources = _SourceIndex(root)
    samples, archive = _samples(np)
    rows: list[dict[str, Any]] = []

    def emit(spec: ObjectSurface, name: str, kind: str, obj: Any,
             member: dict[str, Any] | None, class_row: bool = False) -> None:
        row_id = "numpy." + spec.path + ("" if class_row else "." + name)
        if row_id in seen_ids:
            return
        seen_ids.add(row_id)
        matched_class = class_row and spec.clr_type in object_types
        matched = bool(member) or matched_class
        clr_name = member["name"] if member else name
        alias = matched and not class_row and clr_name != name
        target = spec.clr_type.replace("+", ".") if matched else None
        if target and not class_row:
            target += "." + clr_name
        paths = sources.locate(spec.clr_type, member) if matched else []
        if matched and not paths:
            raise ValueError(f"Object API {row_id} matched {target} without source evidence.")
        notes = "Public NumPy object contract, catalogued outside the headline scope."
        if member:
            notes += " Availability reflects a compiled declaration; it does not assert full behavioral parity."
        elif not matched_class:
            notes += " No matching public member on the corresponding NumSharp object type."
        if spec.path == "ufunc":
            notes += " Top-level NumSharp operations do not expose NumPy's first-class ufunc object protocol."
        if name in {"__iter__", "__next__"}:
            notes += " NumSharp exposes .NET enumeration; Python protocol names are not credited as exact members."
        if alias:
            notes += f" Explicit spelling adapter: {name} is exposed as {clr_name}."
        if class_row:
            parent, _, short = spec.path.rpartition(".")
            doc_surface, doc_name = parent or "np", short
        else:
            doc_surface, doc_name = spec.path, name
        if documentation_url:
            doc_url = documentation_url(doc_surface, doc_name, kind)
        else:
            # Stable family pages, never speculative generated member URLs.
            family = "random/index.html" if spec.path.startswith("random.") else "index.html"
            doc_url = "https://numpy.org/doc/stable/reference/" + family
        rows.append({
            "id": row_id, "origin": "numpy", "surface": doc_surface if class_row else spec.path,
            "name": name, "kind": kind,
            "numpy_signature": signature(obj, name) if kind in {"class", "method"} else "property",
            "documentation_url": doc_url, "in_default_scope": False, "extended": True,
            "object_surface": True, "object_type": "numpy." + spec.path,
            "disposition": spec.disposition, "category": spec.category,
            "availability": "alias" if alias else "exact" if matched else "missing",
            "support": "declared" if matched else "missing",
            "status": "available" if matched else "missing",
            "numsharp_target": target,
            "numsharp_signatures": member["signatures"] if member else ["public class " + target] if matched_class else [],
            "numsharp_obsolete": member["obsolete"] if member else False,
            "numsharp_source_paths": paths,
            "numsharp_source_urls": [source_base_url + path for path in paths],
            "notes": notes,
        })

    try:
        for spec in OBJECT_SURFACES:
            sample = samples.get(spec.path)
            cls = type(sample) if spec.path == "ndarray.flags" else _resolve(np, spec.path)
            host = sample if sample is not None else cls
            members = {
                member["name"]: member
                for collection in ("methods", "properties", "fields")
                for member in object_types.get(spec.clr_type, {}).get(collection, [])
            }
            # Constructors/classes are independently searchable, even for types
            # nested under submodules whose export list excludes class objects.
            emit(spec, spec.path.rsplit(".", 1)[-1], "class", cls, None, class_row=True)
            names = {name for name in dir(host) if not name.startswith("_")}
            names.update(name for name in spec.protocols if hasattr(cls, name))
            for name in sorted(names):
                raw = inspect.getattr_static(host, name)
                # A dtype's `type` value is callable but is an attribute. Inspect
                # the descriptor, never the runtime value, to distinguish these.
                instance_field = sample is not None and name in getattr(sample, "__dict__", {})
                kind = "method" if not instance_field and (
                    callable(raw) or isinstance(raw, (staticmethod, classmethod))) else "property"
                obj = getattr(host, name) if kind == "method" else raw
                clr_name = MEMBER_ALIASES.get((spec.path, name), name)
                member = members.get(clr_name)
                if member and (member["kind"] == "method") != (kind == "method"):
                    member = None
                emit(spec, name, kind, obj, member)

        # Every ufunc method on every ufunc-like export (106 top-level ufunc objects at the
        # pin, aliases included, plus submodule ufuncs and numpy.ma's masked wrappers) is
        # independently searchable (add.reduce, maximum.accumulate, add.resolve_dtypes, ...).
        # A method exists even when the ufunc's arity/core signature makes calling it
        # invalid; the rows record that distinction instead of hiding the method.
        rows.extend(ufunc_protocol_rows(
            np, seen_ids, (("", ""), *ufunc_namespaces),
            signature=signature, documentation_url=documentation_url,
        ))
    finally:
        archive.close()
    return rows
