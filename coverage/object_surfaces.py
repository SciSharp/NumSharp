"""Inventory public NumPy object contracts separately from namespace exports.

The NumPy class/runtime instance is authoritative for member discovery. CLR matching
uses an explicit owner map; sharing a name with np or NDArray does not implement a
ufunc, MaskedArray, matrix, or random.Generator member. These rows never contribute
to the five-surface headline. No random draw, file-system IO, or global RNG mutation
is required to discover the instance-only fields in NumPy's public objects.
"""

from __future__ import annotations

import importlib
import inspect
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


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
    ObjectSurface("ma.MaskedArray", "Masked arrays", disposition="subsystem"),
    ObjectSurface("matrix", "Matrix objects", disposition="subsystem"),
    ObjectSurface("recarray", "Record arrays", disposition="subsystem"),
    ObjectSurface("record", "Record arrays", disposition="subsystem"),
    ObjectSurface("char.chararray", "String array objects", disposition="subsystem"),
    ObjectSurface("memmap", "Memory-mapped arrays", disposition="subsystem"),
    ObjectSurface("lib.npyio.DataSource", "Data sources"),
    ObjectSurface("lib.npyio.NpzFile", "Archive objects", "NumSharp.IO.NpzFile"),
    ObjectSurface("__array_namespace_info__", "Array API introspection"),
    ObjectSurface("testing.suppress_warnings", "Test support", disposition="tooling"),
    ObjectSurface("vectorize", "Function wrappers", disposition="tooling",
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
) -> list[dict[str, Any]]:
    """Return complete extended coverage rows, updating ``seen_ids`` in place.

    ``objectTypes`` must come from inventory schema 5. It is keyed by exact CLR
    full name and carries TypeInventory methods/properties/fields, including
    inherited public members and their ``declaringTypes`` for source evidence.
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

        # NumPy exports 106 ufunc objects at the pin, aliases included. Their five
        # protocol methods are independently searchable (add.reduce, add.at, ...).
        # The attributes exist even when the ufunc's arity/core signature makes
        # calling them invalid; record that distinction instead of hiding them.
        for name in sorted(vars(np)):
            obj = getattr(np, name)
            if name.startswith("_") or not isinstance(obj, np.ufunc):
                continue
            for method in ("reduce", "accumulate", "reduceat", "outer", "at"):
                row_id = f"numpy.{name}.{method}"
                if row_id in seen_ids:
                    continue
                seen_ids.add(row_id)
                rejection = None
                if obj.signature is not None:
                    rejection = "NumPy rejects this method for generalized ufuncs with a core signature."
                elif method in {"reduce", "accumulate", "reduceat", "outer"} and obj.nin != 2:
                    rejection = "NumPy requires a binary ufunc for this method."
                elif method != "outer" and obj.nout != 1:
                    rejection = "NumPy requires a single-output ufunc for this method."
                elif method == "at" and obj.nin not in {1, 2}:
                    rejection = "NumPy at supports only unary and binary ufuncs."
                note = rejection or "NumPy permits this protocol shape; supported operands and dtype resolution still apply."
                rows.append({
                    "id": row_id, "origin": "numpy", "surface": "ufunc",
                    "name": f"{name}.{method}", "kind": "method",
                    "numpy_signature": signature(getattr(obj, method), method),
                    "documentation_url": documentation_url("ufunc", method, "method") if documentation_url
                        else "https://numpy.org/doc/stable/reference/ufuncs.html",
                    "in_default_scope": False, "extended": True, "object_surface": True,
                    "object_type": f"numpy.{name}", "disposition": "subsystem",
                    "category": "Universal function objects", "availability": "missing",
                    "support": "missing", "status": "missing", "numsharp_target": None,
                    "numsharp_signatures": [], "numsharp_obsolete": False,
                    "numsharp_source_paths": [], "numsharp_source_urls": [],
                    "applicability": "not_applicable" if rejection else "conditional",
                    "notes": note + " NumSharp's top-level function does not expose a first-class ufunc object; "
                        "related np.sum/cumsum/outer operations do not implement this object member. "
                        "Catalogued outside the headline scope.",
                })
    finally:
        archive.close()
    return rows
