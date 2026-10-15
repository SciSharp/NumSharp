"""Regression checks for the reported catalog omissions and counting boundaries."""

import ast
import collections
import csv
import importlib
import inspect
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import generate_coverage as coverage
from numpy_documentation import API_ALIASES, documentation_index, documentation_url
from object_surfaces import object_surface_rows


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


class CatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.np = coverage.load_numpy()
        cls.exports = coverage.public_exports(cls.np)
        cls.extended = coverage.extended_surface_rows(
            cls.np, {"modules": {}}, {row["id"] for row in cls.exports}, {}
        )

    def test_all_595_reported_gaps_remain_catalogued(self):
        # Snapshot of the user's reported gaps, not a claim they remain missing.
        # Their implementations can change without allowing the inventory to lose them.
        baseline = {
            line for line in Path(__file__).with_name("catalog_baseline.txt").read_text().splitlines()
            if line and not line.startswith("#")
        }
        self.assertEqual(595, len(baseline))
        actual = {row["id"] for row in self.exports + self.extended}
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
        module's reason (numpy.typing.mypy_plugin). Two classes stay exempt, as documented decisions:
        Python protocol dunders (operators, container and pickling hooks), which are catalogued only
        where an object surface declares them, and members of NumPy's scalar types (numpy.generic and
        its subclasses), which have no NumSharp object to own them.
        """
        excluded = ["numpy." + path for path in coverage.EXCLUDED_PUBLIC_SUBMODULES]
        seen = {row["id"] for row in self.exports + self.extended}
        objects = object_surface_rows(
            self.np, {"objectTypes": {}}, seen, root=coverage.ROOT,
            source_base_url="https://example.test/", signature=coverage.numpy_signature,
            documentation_url=documentation_url, ufunc_namespaces=coverage.UFUNC_PROTOCOL_NAMESPACES,
        )
        ids = {row["id"] for row in self.exports + self.extended + objects}

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
            if member.startswith("__") and member.endswith("__"):
                continue
            if has_owner and inspect.isclass(owner) and issubclass(owner, self.np.generic):
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
            "numpy.polynomial.chebyshev": "module", "numpy.matlib": "module", "numpy.version": "module",
            "numpy.version.version": "constant", "numpy.matlib.repmat": "function",
            # numpy.typing.NDArray is a types.GenericAlias, which is callable; it is still a type.
            "numpy.typing.NDArray": "class", "numpy.typing.ArrayLike": "class",
        }
        self.assertEqual(expected_kinds, {api_id: rows[api_id]["kind"] for api_id in expected_kinds})
        for api_id, kind in expected_kinds.items():
            self.assertEqual(kind == "function", coverage.is_api_row(rows[api_id]), api_id)
        # Documented but absent from numpy.exceptions.__all__: the documentation union finds it.
        self.assertIn("numpy.exceptions.RankWarning", rows)
        # numpy.matlib re-exports NumPy's whole namespace; only its own eight functions are matlib API.
        matlib = sorted(row["name"] for row in self.extended if row["surface"] == "matlib" and row["kind"] != "module")
        self.assertEqual(["empty", "eye", "identity", "ones", "rand", "randn", "repmat", "zeros"], matlib)
        self.assertNotIn("matlib", {path for path, _ in coverage.UFUNC_PROTOCOL_NAMESPACES})
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
        samples = []
        roots = ("ndarray.interop", "__array_namespace_info__", "lib", "ctypeslib", "testing")
        for root in roots:
            for name, surface in ((root, "np"), (root + ".member", root),
                                  (root + ".Owner.method", root + ".Owner")):
                samples.append({"id": "numpy." + name, "surface": surface, "category": "Ignored"})
        keep = [{"id": "numpy.emath.sqrt", "surface": "emath", "category": "Complex-domain math"},
                {"id": "numpy.test", "surface": "np", "category": "Runtime & diagnostics"}]
        self.assertEqual(keep, coverage.dashboard_rows(samples + keep))
        actual = coverage.dashboard_rows(self.exports + self.extended)
        self.assertFalse(any(coverage.ignored_dashboard_id(row["id"]) for row in actual))
        self.assertFalse(any(row["surface"] in roots for row in actual))

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


if __name__ == "__main__":
    unittest.main()
