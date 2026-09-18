"""Regression checks for the reported catalog omissions and counting boundaries."""

import csv
import io
import unittest
from pathlib import Path
from unittest.mock import patch

import generate_coverage as coverage


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
