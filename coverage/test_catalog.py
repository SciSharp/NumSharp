"""Regression checks for the reported catalog omissions and counting boundaries."""

import ast
import collections
import csv
import importlib
import inspect
import io
import sys
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import generate_coverage as coverage
import object_surfaces
from numpy_documentation import API_ALIASES, documentation_index, documentation_url
from object_surfaces import (CANONICAL_SCALAR_TYPES, PROTOCOL_MACHINERY, class_member_rows, numpy_protocols,
                             object_surface_rows, scalar_type_rows)


def numpy_public_modules(np):
    """Read NumPy's own public-module list, as the installed NumPy evaluates it on this Python.

    numpy/tests/test_public_api.py defines PUBLIC_MODULES (extended by a ``sys.version_info`` branch)
    and PUBLIC_ALIASED_MODULES. The file imports pytest at module level, so only those assignment
    statements are compiled and run, with ``sys`` as their one free name; the result is exactly the
    list NumPy's own test enforces.

    :param np: the imported, pinned numpy module.
    :returns: the set of dotted public module names, or None when this NumPy ships without its tests.
    """
    path = Path(np.__file__).with_name("tests") / "test_public_api.py"
    if not path.exists():
        return None
    wanted = {"PUBLIC_MODULES", "PUBLIC_ALIASED_MODULES"}

    def assigns(node):
        targets = (node.targets if isinstance(node, ast.Assign)
                   else [node.target] if isinstance(node, ast.AugAssign) else [])
        return any(isinstance(target, ast.Name) and target.id in wanted for target in targets)

    body = [node for node in ast.parse(path.read_text(encoding="utf-8")).body
            if assigns(node) or (isinstance(node, ast.If) and node.body and all(map(assigns, node.body)))]
    namespace = {"sys": sys}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return set(namespace["PUBLIC_MODULES"]) | set(namespace["PUBLIC_ALIASED_MODULES"])


def resolve_numpy_name(np, api_id):
    """Resolve a dotted ``numpy.*`` name to the object it names on the pinned NumPy.

    Walks attributes from the numpy package and imports a dotted prefix only when the attribute is
    missing (numpy.lib.format is importable before it is an attribute; numpy.testing.extbuild is an
    attribute, not an importable path).

    :param np: the imported numpy module.
    :param api_id: a name such as ``numpy.ma.masked_array.T``.
    :returns: ``(True, obj)`` when every segment resolves, else ``(False, None)``; never raises, so a
        name the documentation lists only for a newer NumPy reads as absent.
    """
    parts = api_id.split(".")
    obj = np
    for index in range(1, len(parts)):
        try:
            obj = getattr(obj, parts[index])
            continue
        except Exception:
            pass
        try:
            obj = importlib.import_module(".".join(parts[: index + 1]))
        except Exception:
            return False, None
    return True, obj


def has_own_documentation(np, obj):
    """Tell whether an object carries its own documentation (a module, class, routine, descriptor or ufunc).

    A plain value (a number, an array, numpy.ma.masked) would report its TYPE's docstring through
    inspect.getdoc, which documents the type, not the API the row names.

    :param np: the imported numpy module.
    :param obj: a resolved catalog object.
    :returns: True when inspect.getdoc(obj) documents the object itself.
    """
    return (inspect.ismodule(obj) or inspect.isclass(obj) or inspect.isroutine(obj)
            or inspect.isdatadescriptor(obj) or inspect.ismethoddescriptor(obj) or isinstance(obj, np.ufunc)
            or (callable(obj) and "__doc__" in getattr(obj, "__dict__", {})))


class CatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Build the NumPy side of the catalog once, as the generator does, without a NumSharp inventory."""
        cls.np = coverage.load_numpy()
        cls.exports = coverage.public_exports(cls.np)
        cls.extended = coverage.extended_surface_rows(
            cls.np, {"modules": {}}, {row["id"] for row in cls.exports}, {}
        )
        # The rest of the NumPy side, in the generator's order and without a NumSharp inventory (nothing is
        # credited): object surfaces with their protocols, the scalar types, then every other class's members.
        seen = {row["id"] for row in cls.exports + cls.extended}
        cls.objects = object_surface_rows(
            cls.np, {"objectTypes": {}}, seen, root=coverage.ROOT,
            source_base_url="https://example.test/", signature=coverage.numpy_signature,
            documentation_url=documentation_url, ufunc_namespaces=coverage.UFUNC_PROTOCOL_NAMESPACES,
        )
        cls.scalars = scalar_type_rows(cls.np, seen, signature=coverage.numpy_signature,
                                       documentation_url=documentation_url)
        cls.members = class_member_rows(
            cls.np, cls.exports + cls.extended + cls.objects, seen, {}, root=coverage.ROOT,
            source_base_url="https://example.test/", signature=coverage.numpy_signature,
            documentation_url=documentation_url, category_of=coverage.member_category,
            excluded=coverage.compat_excluded,
        )
        cls.catalog = cls.exports + cls.extended + cls.objects + cls.scalars + cls.members

    def test_all_595_reported_gaps_remain_catalogued(self):
        # Snapshot of the user's reported gaps, not a claim they remain missing.
        # Their implementations can change without allowing the inventory to lose them.
        baseline = {
            line for line in Path(__file__).with_name("catalog_baseline.txt").read_text().splitlines()
            if line and not line.startswith("#")
        }
        self.assertEqual(595, len(baseline))
        # The whole NumPy side: since generator 1.14.0 ndarray's protocols (the baseline's numpy.ndarray.__array__,
        # once the invented numpy.ndarray.interop.__array__) come from the ndarray object surface.
        actual = {row["id"] for row in self.catalog}
        self.assertEqual(set(), baseline - actual)

    def test_every_declared_extended_family_contributes_rows(self):
        surfaces = {row["surface"] for row in self.extended}
        for _, surface, _, _ in coverage.EXTENDED_SUBMODULES:
            self.assertIn(surface, surfaces)
        self.assertTrue(all(row["documentation_url"] for row in self.extended))
        self.assertEqual(len(self.extended), len({row["id"] for row in self.extended}))

    def test_every_public_numpy_module_is_scanned_or_excluded(self):
        """NumPy's own public-module list bounds the catalog from outside the generator.

        Until generator 1.13.0 the scanned families were a hand-picked subset, and numpy.dtypes,
        numpy.exceptions, numpy.typing, numpy.version and numpy.ma.mrecords had no rows at all. A module
        NumPy lists as public must now be scanned (numpy itself, a headline submodule or an extended
        family, by import path or display surface) or be excluded, with a reason, in
        EXCLUDED_PUBLIC_SUBMODULES; a NumPy release adding a public module fails here.
        """
        public = numpy_public_modules(self.np)
        if public is None:
            self.skipTest("This NumPy installation ships without numpy/tests/test_public_api.py.")
        scanned = ({"numpy"} | {"numpy." + surface for surface in coverage.HEADLINE_SUBMODULES}
                   | {"numpy." + path for path, _, _, _ in coverage.EXTENDED_SUBMODULES}
                   | {"numpy." + surface for _, surface, _, _ in coverage.EXTENDED_SUBMODULES})
        excluded = {"numpy." + path for path in coverage.EXCLUDED_PUBLIC_SUBMODULES}

        def covered(module):
            return module in scanned or any(coverage.within_namespace(module, root) for root in excluded)

        self.assertEqual([], sorted(module for module in public if not covered(module)))
        self.assertEqual(set(), excluded & scanned, "a module is either scanned or excluded, never both")
        # An exclusion must name a public module; numpy.distutils is public only on Python < 3.12.
        stale = {module for module in excluded - public
                 if not (module == "numpy.distutils" and sys.version_info >= (3, 12))}
        self.assertEqual(set(), stale)

    def test_every_documented_numpy_api_on_the_pin_has_a_row(self):
        """The catalog covers every name NumPy's documentation index lists and the pin defines.

        The reverse of audit_documentation.py, which only proves that each row's link exists: here every
        ``numpy.*`` name in the offline documentation snapshot that resolves on NumPy 2.4.2 must have a
        row, its documented alias (numpy_documentation.API_ALIASES), or a row for the same object under
        another spelling (numpy.lib.scimath.sqrt IS numpy.emath.sqrt; numpy.ma.masked_array.T is the
        MaskedArray member). Names the documentation lists only for a newer NumPy do not resolve and are
        skipped, and a name inside a module EXCLUDED_PUBLIC_SUBMODULES excludes is covered by that
        module's reason (numpy.typing.mypy_plugin), as is a NumPy 1.x compatibility API the catalog leaves
        out (COMPAT_EXCLUDED_APIS: numpy.matlib). Python protocols and the members of NumPy's scalar types,
        exempt until generator 1.14.0, are covered now; only Python's object machinery stays exempt
        (NumPy documents ndarray.__new__, which the class row stands for).
        """
        excluded = (["numpy." + path for path in coverage.EXCLUDED_PUBLIC_SUBMODULES]
                    + list(coverage.COMPAT_EXCLUDED_APIS))
        ids = {row["id"] for row in self.catalog}

        def canonical(api_id):
            for alias, target in API_ALIASES.items():
                if api_id == alias or api_id.startswith(alias + "."):
                    return target + api_id[len(alias):]
            return api_id

        known = ids | {canonical(api_id) for api_id in ids}
        # Identity index over the catalogued objects. Python and NumPy scalars are left out: two
        # unrelated constants can be the same True or 0 object, which is not an alias.
        scalars = (bool, int, float, complex, str, bytes, type(None), self.np.generic)
        by_object = collections.defaultdict(set)
        for api_id in ids:
            found, obj = resolve_numpy_name(self.np, api_id)
            if found and not isinstance(obj, scalars):
                by_object[id(obj)].add(api_id)

        uncovered = []
        for name in sorted(documentation_index()):
            if name in known or any(coverage.within_namespace(name, root) for root in excluded):
                continue
            found, obj = resolve_numpy_name(self.np, name)
            if not found or (not isinstance(obj, scalars) and by_object.get(id(obj))):
                continue
            owner_id, _, member = name.rpartition(".")
            has_owner, owner = resolve_numpy_name(self.np, owner_id)
            if has_owner and any(f"{alias}.{member}" in ids for alias in by_object.get(id(owner), ())):
                continue
            if member in PROTOCOL_MACHINERY:
                continue
            uncovered.append(name)
        self.assertEqual([], uncovered)

    def test_extended_scan_catalogues_classes_constants_and_modules_as_supporting_rows(self):
        """Every public member of a scanned family has a row, and non-callables stay out of percentages.

        Before generator 1.13.0 the scan kept callables only, so numpy.ma.masked, the polynomial
        domain/zero/one/x constants and every exception class had no row.
        """
        rows = {row["id"]: row for row in self.extended}
        expected_kinds = {
            "numpy.ma.masked": "constant", "numpy.ma.nomask": "constant", "numpy.ma.mr_": "constant",
            "numpy.polynomial.chebyshev.chebdomain": "constant", "numpy.dtypes.Float64DType": "class",
            "numpy.exceptions.AxisError": "class", "numpy.ma.mrecords.mrecarray": "class",
            "numpy.polynomial.chebyshev": "module", "numpy.version": "module",
            "numpy.version.version": "constant", "numpy.f2py.get_include": "function",
            # numpy.typing.NDArray is a types.GenericAlias, which is callable; it is still a type.
            "numpy.typing.NDArray": "class", "numpy.typing.ArrayLike": "class",
        }
        self.assertEqual(expected_kinds, {api_id: rows[api_id]["kind"] for api_id in expected_kinds})
        for api_id, kind in expected_kinds.items():
            self.assertEqual(kind == "function", coverage.is_api_row(rows[api_id]), api_id)
        # Documented but absent from numpy.exceptions.__all__: the documentation union finds it.
        self.assertIn("numpy.exceptions.RankWarning", rows)
        # NumPy 1.x compatibility APIs are left out, documented or listed: numpy.matlib (a scanned family until
        # generator 1.14.0) and numpy.typing.NBitBase, which numpy.typing.__all__ still lists.
        self.assertFalse(any(coverage.compat_excluded(row["id"]) for row in self.catalog))
        self.assertIn("NBitBase", self.np.typing.__all__)
        # Classes an object surface owns get their class row there, not from this scan.
        for owned in ("numpy.ma.MaskedArray", "numpy.char.chararray", "numpy.ma.mrecords.MaskedRecords",
                      "numpy.polynomial.polynomial.Polynomial", "numpy.lib.npyio.NpzFile"):
            self.assertNotIn(owned, rows)
        # A module another package owns never becomes a row through a dir() fallback.
        modules = [row["id"] for row in self.extended if row["kind"] == "module"]
        self.assertTrue(all(resolve_numpy_name(self.np, api_id)[1].__name__.startswith("numpy")
                            for api_id in modules))

    def test_extended_classes_credit_facade_members_and_ported_types_only(self):
        """A submodule member is credited by a same-surface facade member or, for a real class, a ported type.

        numpy.exceptions.AxisError has no np.exceptions facade but NumSharp exports NumSharp.AxisError;
        numpy.typing.NDArray is a typing alias, so NumSharp.NDArray must not credit it by name.
        """
        facade = {"ma": [{"name": "masked", "kind": "property", "target": "NumSharp.np.ma.masked",
                          "signatures": ["NDMaskedArray masked { get; }"]}]}
        inventory = {"modules": {}, "exportedTypes": ["NumSharp.AxisError", "NumSharp.NDArray"]}
        rows = {row["id"]: row for row in coverage.extended_surface_rows(self.np, inventory, set(), facade)}
        masked = rows["numpy.ma.masked"]
        self.assertEqual(("available", "exact", "NumSharp.np.ma.masked"),
                         (masked["status"], masked["availability"], masked["numsharp_target"]))
        axis_error = rows["numpy.exceptions.AxisError"]
        self.assertEqual(("available", "type", "NumSharp.AxisError"),
                         (axis_error["status"], axis_error["availability"], axis_error["numsharp_target"]))
        for api_id in ("numpy.typing.NDArray", "numpy.exceptions.ComplexWarning", "numpy.ma.nomask"):
            self.assertEqual(("missing", None), (rows[api_id]["status"], rows[api_id]["numsharp_target"]), api_id)

    def test_namesakes_do_not_credit_different_subsystems(self):
        rows = {row["id"]: row for row in self.extended}
        for api_id in ("numpy.emath.sqrt", "numpy.ma.sum", "numpy.polynomial.polynomial.polyfit"):
            self.assertEqual("missing", rows[api_id]["status"])

    def test_ignored_namespaces_and_their_root_exports_never_reach_dashboard(self):
        """Only numpy.lib and numpy.testing stay off the dashboard, and numpy.lib.NumpyVersion is shown anyway.

        Until generator 1.14.0 numpy.ctypeslib, numpy.__array_namespace_info__ and ndarray's protocols were
        hidden too; the owner's rule is that everything else is shown.
        """
        samples = []
        roots = ("lib", "testing")
        self.assertEqual(roots, coverage.IGNORED_DASHBOARD_ROOTS)
        for root in roots:
            for name, surface in ((root, "np"), (root + ".member", root),
                                  (root + ".Owner.method", root + ".Owner")):
                samples.append({"id": "numpy." + name, "surface": surface, "category": "Ignored"})
        keep = [{"id": "numpy.emath.sqrt", "surface": "emath", "category": "Complex-domain math"},
                {"id": "numpy.test", "surface": "np", "category": "Runtime & diagnostics"},
                {"id": "numpy.ctypeslib.as_array", "surface": "ctypeslib", "category": "ctypes interop"},
                {"id": "numpy.ndarray.__add__", "surface": "ndarray", "category": "Array protocols"}]
        self.assertEqual(keep, coverage.dashboard_rows(samples + keep))
        # The exception keeps its rows, regrouped under its own surface (the class row's surface is "lib").
        version = [{"id": "numpy.lib.NumpyVersion", "surface": "lib", "category": "Types"},
                   {"id": "numpy.lib.NumpyVersion.major", "surface": "lib.NumpyVersion",
                    "category": "Runtime & diagnostics"}]
        self.assertEqual(["lib.NumpyVersion"] * 2, [row["surface"] for row in coverage.dashboard_rows(version)])
        actual = coverage.dashboard_rows(self.catalog)
        self.assertFalse(any(coverage.ignored_dashboard_id(row["id"]) for row in actual))
        self.assertFalse(any(row["surface"] in roots for row in actual))
        shown = {row["id"] for row in actual}
        # numpy.rec.format_parser.dtype exists only on an instance (format_parser sets it in __init__); the object
        # surface's sample instance is what finds it.
        for api_id in ("numpy.ctypeslib.as_array", "numpy.__array_namespace_info__.capabilities",
                       "numpy.rec.format_parser.dtype",
                       "numpy.ndarray.__array__", "numpy.lib.NumpyVersion", "numpy.lib.NumpyVersion.major"):
            self.assertIn(api_id, shown)
        hidden = {row["id"] for row in self.catalog} - shown
        self.assertTrue(hidden)
        self.assertTrue(all(coverage.within_namespace(api_id, "numpy.lib")
                            or coverage.within_namespace(api_id, "numpy.testing") for api_id in hidden),
                        sorted(hidden)[:10])

    def test_masked_and_polynomial_groups_preserve_canonical_identifiers(self):
        rows = [
            {"id": "numpy.ma.sum", "surface": "ma", "category": "Masked arrays"},
            {"id": "numpy.ma.MaskedArray.sum", "surface": "ma.MaskedArray", "category": "Masked objects"},
            {"id": "numpy.polyfit", "surface": "np", "category": "Polynomials"},
            {"id": "numpy.roots", "surface": "np", "category": "Polynomials"},
            {"id": "numpy.poly1d.roots", "surface": "poly1d", "category": "Polynomials"},
            {"id": "numpy.polynomial.chebyshev.Chebyshev.roots", "surface": "polynomial.chebyshev.Chebyshev", "category": "Polynomial objects"},
            {"id": "numpy.polynomial.polynomial.polyroots", "surface": "polynomial.polynomial", "category": "Power-series polynomials"},
            {"id": "numpy.polynomial.polyutils.as_series", "surface": "polynomial.polyutils", "category": "Polynomial utilities"},
        ]
        merged = coverage.dashboard_rows(rows)
        self.assertEqual([row["id"] for row in rows], [row["id"] for row in merged])
        self.assertEqual({"ma", "polynomial"}, {row["surface"] for row in merged})
        self.assertEqual({"Masked arrays", "Polynomials"}, {row["category"] for row in merged})
        self.assertEqual("ma.MaskedArray", rows[1]["surface"], "raw owner mapping must remain unchanged")
        self.assertEqual(merged, coverage.dashboard_rows(merged), "projection must be idempotent")

    def test_emitted_artifacts_and_counts_use_the_projected_inventory(self):
        import json
        expected = coverage.dashboard_rows(self.extended)
        with patch.object(coverage, "resolve_rows", return_value=(self.extended, set())):
            outputs = coverage.render_outputs(self.np, {"assemblyVersion": "test"}, {})
        payload = json.loads(outputs["coverage.json"])
        manifest = json.loads(outputs["manifest.json"])
        self.assertEqual(expected, payload["rows"])
        self.assertEqual(len(expected), payload["summary"]["catalog_rows"])
        self.assertEqual(payload["summary"], manifest["summary"])
        csv_ids = {row["id"] for row in csv.DictReader(io.StringIO(outputs["coverage.csv"]))}
        self.assertEqual({row["id"] for row in expected}, csv_ids)
        for path in manifest["extended_surfaces"] + manifest["object_surfaces"]:
            self.assertFalse(coverage.ignored_dashboard_id(path), path)
        self.assertEqual(["polynomial"], [surface for surface in payload["summary"]["api_scope"]["by_surface"]
                                         if surface.startswith("poly")])

    def test_supporting_exports_do_not_inflate_expanded_api_percentage(self):
        def row(name, kind, *, extended=False, headline=False, origin="numpy", status="missing"):
            return dict(id=name, name=name, surface="test", category="Test", kind=kind,
                        origin=origin, extended=extended, disposition="candidate",
                        in_default_scope=headline, status=status, availability=status)
        rows = [row("base", "function", headline=True, status="available"),
                row("member", "method", extended=True), row("field", "property", extended=True),
                row("owner", "class", extended=True, status="available"),
                row("constant", "constant"), row("module", "module"),
                row("extension", "method", origin="numsharp", status="extension")]
        summary = coverage.build_summary(rows)
        self.assertEqual(1, summary["default_scope"]["total"])
        self.assertEqual(3, summary["api_scope"]["total"])
        self.assertEqual(33.3, summary["api_scope"]["coverage_percent"])
        self.assertEqual(3, summary["api_scope"]["by_category"]["Test"]["total"])
        self.assertEqual(7, summary["catalog_rows"])

    def test_csv_preserves_extended_scope_and_object_metadata(self):
        source = {**self.extended[0], "object_surface": True, "object_type": "numpy.ufunc",
                  "applicability": "conditional"}
        row = next(csv.DictReader(io.StringIO(coverage.csv_text([source]))))
        self.assertEqual("True", row["extended"])
        self.assertEqual("True", row["object_surface"])
        self.assertEqual(source["disposition"], row["disposition"])
        self.assertEqual("numpy.ufunc", row["object_type"])
        self.assertEqual("conditional", row["applicability"])


    def test_every_catalogued_row_names_an_api_of_the_pinned_numpy(self):
        """Every NumPy row resolves on NumPy 2.4.2: each catalogued API exists on the pin.

        A member set in __init__ (finfo.bits, Polynomial.coef, format_parser.dtype) exists only on an instance,
        so an id whose owner is an object surface is also looked up on that surface's sample instance.
        """
        samples, archive = object_surfaces._samples(self.np)
        try:
            absent = []
            for row in self.catalog:
                found, _ = resolve_numpy_name(self.np, row["id"])
                if found:
                    continue
                owner, _, member = row["id"].rpartition(".")
                sample = samples.get(owner[len("numpy."):])
                if sample is not None and hasattr(sample, member):
                    continue
                absent.append(row["id"])
            self.assertEqual([], absent)
        finally:
            archive.close()

    def test_no_catalogued_row_is_an_api_numpy_deprecated(self):
        """No row names an API NumPy deprecates as a whole, unless KEPT_DEPRECATED_APIS keeps it with a reason.

        The owner's rule: NumPy 1.x compatibility code is left out of the catalog. NumPy marks it two ways, and
        both are checked on every row: a DeprecationWarning or PendingDeprecationWarning when the name is
        reached, and a whole-API ``.. deprecated::`` directive in its documentation (deprecation_scope). A
        directive about one parameter (numpy.fft.fft2's ``s`` without ``axes``) deprecates a use, not the API.
        A row added for a deprecated API fails here until it is left out (COMPAT_EXCLUDED_APIS) or kept with
        a reason; a kept entry that is no longer deprecated (or no longer catalogued) fails too.
        """
        flagged = {}
        for row in self.catalog:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                found, obj = resolve_numpy_name(self.np, row["id"])
            if any(issubclass(warning.category, (DeprecationWarning, PendingDeprecationWarning)) for warning in caught):
                flagged[row["id"]] = "warns when reached"
            elif found and has_own_documentation(self.np, obj) and \
                    coverage.deprecation_scope(inspect.getdoc(obj)) == "api":
                flagged[row["id"]] = "documented as deprecated"
        self.assertEqual(sorted(coverage.KEPT_DEPRECATED_APIS), sorted(flagged), flagged)

    def test_compat_excluded_apis_are_numpy_deprecations_on_the_pin(self):
        """Each API the catalog leaves out exists on the pin and is deprecated by NumPy itself.

        numpy.matlib warns when imported (only the first import runs the module, so it is reloaded here);
        numpy.typing.NBitBase carries a whole-API deprecation directive.
        """
        self.assertEqual({"numpy.matlib", "numpy.typing.NBitBase"}, set(coverage.COMPAT_EXCLUDED_APIS))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            found, matlib = resolve_numpy_name(self.np, "numpy.matlib")
            self.assertTrue(found)
            importlib.reload(matlib)
        self.assertTrue(any(issubclass(warning.category, PendingDeprecationWarning) for warning in caught))
        found, nbitbase = resolve_numpy_name(self.np, "numpy.typing.NBitBase")
        self.assertTrue(found)
        self.assertEqual("api", coverage.deprecation_scope(inspect.getdoc(nbitbase)))
        for api_id in ("numpy.matlib.zeros", "numpy.typing.NBitBase", "numpy.matlib"):
            self.assertTrue(coverage.compat_excluded(api_id), api_id)
        self.assertFalse(coverage.compat_excluded("numpy.matrix"))
        self.assertFalse(coverage.compat_excluded("numpy.typing.NDArray"))

    def test_a_numsharp_port_of_a_compat_api_is_not_a_numsharp_only_extension(self):
        """np.typing.NBitBase (kept [Obsolete] like NumPy's) gets no NumSharp-only row: it is NumPy's API, left out."""
        def member(name):
            return {"name": name, "target": f"NumSharp.np.typing.{name}"}

        self.assertFalse(coverage.lists_as_numsharp_only("typing", member("NBitBase"), set()))
        self.assertTrue(coverage.lists_as_numsharp_only("typing", member("Extra"), set()))
        self.assertFalse(coverage.lists_as_numsharp_only("typing", member("Extra"), {"NumSharp.np.typing.Extra"}))
        self.assertFalse(coverage.lists_as_numsharp_only("typing", member("_hidden"), set()))

    def test_deprecation_scope_tells_a_deprecated_api_from_a_deprecated_use(self):
        """A directive right after the summary deprecates the API; one under a parameter or a section, a use."""
        self.assertIsNone(coverage.deprecation_scope(None))
        self.assertIsNone(coverage.deprecation_scope("Summary.\n\nParameters\n----------\nx : int"))
        self.assertEqual("api", coverage.deprecation_scope(
            "Summary.\n\n.. deprecated:: 1.11.0\n\nParameters\n----------\nx : int"))
        self.assertEqual("part", coverage.deprecation_scope(
            "Summary.\n\nParameters\n----------\ns : int\n    .. deprecated:: 2.0\n"))
        self.assertEqual("part", coverage.deprecation_scope("Title\n=====\n\nText.\n\n.. deprecated:: 2.3\n"))
        # NumPy's own docstrings, one of each kind.
        self.assertEqual("api", coverage.deprecation_scope(inspect.getdoc(self.np.random.random_integers)))
        self.assertEqual("part", coverage.deprecation_scope(inspect.getdoc(self.np.fft.fft2)))
        self.assertEqual("part", coverage.deprecation_scope(inspect.getdoc(self.np.fromstring)))
        self.assertEqual("part", coverage.deprecation_scope(inspect.getdoc(self.np.typing)))

    def test_numpy_protocols_are_numpy_defined_and_never_python_machinery(self):
        """Discovery lists the protocols NumPy's classes declare, and none of Python's own machinery."""
        np = self.np
        protocols = numpy_protocols(np.ndarray)
        for name in ("__add__", "__radd__", "__iadd__", "__getitem__", "__array__", "__array_interface__",
                     "__class_getitem__", "__dlpack__", "__array_priority__"):
            self.assertIn(name, protocols)
        # Python's machinery, and a protocol NumPy declares absent (arrays are unhashable: __hash__ is None).
        for name in ("__init__", "__new__", "__getattribute__", "__doc__", "__hash__"):
            self.assertNotIn(name, protocols)
        self.assertIsNone(np.ndarray.__hash__)
        # Per owner, numpy.matrix lists ndarray's protocols too; its own namespace holds only what it redefines.
        self.assertIn("__add__", numpy_protocols(np.matrix))
        self.assertNotIn("__add__", numpy_protocols(np.matrix, own_only=True))
        self.assertIn("__mul__", numpy_protocols(np.matrix, own_only=True))
        # A protocol from Python or another package is not NumPy's: NpzFile is a collections.abc.Mapping, and
        # Mapping's __eq__ is not NumPy API; Cython's pickling shims on numpy.random's classes are not either.
        npz = importlib.import_module("numpy.lib.npyio").NpzFile
        self.assertTrue(hasattr(npz, "__eq__"))
        self.assertNotIn("__eq__", numpy_protocols(npz))
        self.assertIn("__getitem__", numpy_protocols(npz))
        self.assertTrue(hasattr(np.random.PCG64, "__reduce_cython__"))
        self.assertNotIn("__reduce_cython__", numpy_protocols(np.random.PCG64))
        self.assertIn("__reduce__", numpy_protocols(np.random.PCG64))
        # Python 2 names NumPy still defines (numpy.lib.user_array.container's __oct__ and __hex__) are not
        # Python 3 API: Python 3's oct() and hex() call __index__.
        container = importlib.import_module("numpy.lib.user_array").container
        for name in ("__oct__", "__hex__"):
            self.assertIn(name, vars(container))
            self.assertNotIn(name, numpy_protocols(container))
        self.assertIn("__add__", numpy_protocols(container))
        # Every object-member row whose member is a protocol name is marked as a Python protocol, and no other.
        for row in self.catalog:
            if not row.get("object_surface") or row["kind"] == "class":
                continue
            member = row["name"].rpartition(".")[2]
            self.assertEqual(object_surfaces.is_protocol_name(member), bool(row.get("python_protocol")), row["id"])

    def test_scalar_types_are_catalogued_once_at_the_class_that_defines_each_member(self):
        """Scalar-type members are listed once each, at their defining class, under platform-independent names.

        numpy.long is numpy.int32 on Windows and numpy.int64 on Linux, so rows use NumPy's canonical names
        (numpy.byte, numpy.long, ...), which denote one C type everywhere. numpy.record is an object surface.
        """
        np = self.np
        rows = {row["id"]: row for row in self.scalars}
        owners = {row["object_type"] for row in self.scalars}
        self.assertTrue(owners <= {"numpy." + name for name in CANONICAL_SCALAR_TYPES}, owners)
        self.assertNotIn("numpy.record", owners)
        for api_id in ("numpy.generic.astype", "numpy.generic.__array__", "numpy.generic.T", "numpy.byte.__add__",
                       "numpy.long.__add__", "numpy.longlong.__add__", "numpy.integer.is_integer",
                       "numpy.number.__class_getitem__", "numpy.void.getfield", "numpy.datetime64.__lt__"):
            self.assertIn(api_id, rows)
        # Sized aliases are never owners, and an inherited member is not repeated on a subclass.
        for api_id in ("numpy.int8.__add__", "numpy.float64.astype", "numpy.byte.astype", "numpy.int64.__add__"):
            self.assertNotIn(api_id, rows)
        self.assertIs(np.byte, np.int8)
        self.assertEqual("property", rows["numpy.generic.T"]["kind"])
        self.assertEqual("method", rows["numpy.byte.__add__"]["kind"])
        self.assertEqual("not_applicable", rows["numpy.void.__delitem__"]["applicability"])
        # Windows NumPy gives numpy.intc.bit_count no docstring (intc has no sized alias there); the row takes the
        # signature every other integer class gives the method, so it reads the same as on Linux.
        self.assertEqual("(self, /)", rows["numpy.intc.bit_count"]["numpy_signature"])
        for row in self.scalars:
            self.assertEqual(("missing", "generic", "Scalar types", False),
                             (row["status"], row["surface"], row["category"], row["in_default_scope"]), row["id"])
            self.assertTrue(row["name"].startswith(row["object_type"][len("numpy."):] + "."), row["id"])
        # The pin's count (the same on Windows and Linux: canonical classes are platform-independent).
        self.assertEqual(923, len(rows))

    def test_other_public_classes_list_the_members_they_define(self):
        """Every other public NumPy class lists its own members; aliases, foreign classes and typing aliases do not."""
        rows = {row["id"]: row for row in self.members}
        expected = {
            "numpy.errstate.__call__", "numpy.errstate.__enter__", "numpy.errstate.__exit__",
            "numpy.exceptions.AxisError.axis", "numpy.exceptions.AxisError.ndim", "numpy.exceptions.AxisError.__str__",
            "numpy.ma.mvoid.filled", "numpy.ma.mvoid.tolist", "numpy.ma.mvoid.__getitem__",
            "numpy.dtypes.StringDType.coerce", "numpy.dtypes.StringDType.na_object", "numpy.dtypes.StringDType.__lt__",
        }
        self.assertTrue(expected <= set(rows), sorted(expected - set(rows)))
        # An alias is one owner, catalogued under its canonical id: numpy.ma.masked_array is numpy.ma.MaskedArray,
        # numpy.polynomial.Chebyshev is numpy.polynomial.chebyshev.Chebyshev (both object surfaces).
        # Not NumPy's own: ctypes' c_intp and unittest's TestCase; a typing alias is not a class.
        # A subclass lists only what it adds: numpy.dtypes.Float64DType adds nothing to numpy.dtype.
        # A scalar class under another name is a scalar type (scalar_type_rows lists it under its canonical name).
        for prefix in ("numpy.ma.masked_array.", "numpy.polynomial.Chebyshev.", "numpy.rec.recarray.",
                       "numpy.ctypeslib.c_intp.", "numpy.testing.TestCase.", "numpy.typing.NDArray.",
                       "numpy.dtypes.Float64DType.", "numpy.exceptions.ComplexWarning.", "numpy.bool_.",
                       "numpy.int64.", "numpy.ma.MaskType."):
            self.assertFalse(any(api_id.startswith(prefix) for api_id in rows), prefix)
        self.assertEqual("Floating-point handling", rows["numpy.errstate.__enter__"]["category"])
        self.assertEqual("Masked arrays", rows["numpy.ma.mvoid.filled"]["category"])
        self.assertEqual("ma.mvoid", rows["numpy.ma.mvoid.filled"]["surface"])
        self.assertEqual("not_applicable", rows["numpy.dtypes.StringDType.__lt__"]["applicability"])
        self.assertNotIn("applicability", rows["numpy.dtypes.StringDType.__eq__"])
        # Without a NumSharp inventory nothing is credited.
        self.assertTrue(all(row["status"] == "missing" for row in self.members))


if __name__ == "__main__":
    unittest.main()
