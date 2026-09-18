"""Regression gate for object-member inventory (NumPy 2.4.2 + built inventory tool).

Run after building NumSharp.Tools.ApiInventory:
    python -m unittest discover -s coverage -p test_object_surfaces.py
"""

from __future__ import annotations

import inspect
import json
import subprocess
import unittest
from pathlib import Path

import numpy as np

from numpy_documentation import documentation_url
from object_surfaces import enrich_object_classes, object_surface_rows


ROOT = Path(__file__).resolve().parents[1]


def signature(obj, name):
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return (inspect.getdoc(obj) or name).splitlines()[0]


class ObjectSurfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if np.__version__ != "2.4.2":
            raise RuntimeError("Object inventory tests require NumPy 2.4.2.")
        completed = subprocess.run(
            ["dotnet", "run", "--project", str(ROOT / "coverage/NumSharp.Tools.ApiInventory"),
             "--configuration", "Release", "--framework", "net8.0", "--no-build", "--no-restore"],
            cwd=ROOT, check=True, text=True, capture_output=True,
        )
        cls.inventory = json.loads(completed.stdout)
        cls.rows = object_surface_rows(
            np, cls.inventory, set(), root=ROOT,
            source_base_url="https://github.com/SciSharp/NumSharp/blob/master/",
            signature=signature, documentation_url=documentation_url,
        )
        cls.by_id = {row["id"]: row for row in cls.rows}

    def test_unique_extended_rows_have_explicit_object_ownership(self):
        self.assertEqual(len(self.rows), len(self.by_id))
        self.assertTrue(all(row["extended"] and row["object_surface"] for row in self.rows))
        self.assertTrue(all(not row["in_default_scope"] for row in self.rows))
        self.assertTrue(all(row["object_type"].startswith("numpy.") for row in self.rows))
        for row in self.rows:
            if row["status"] == "available":
                self.assertTrue(row["numsharp_signatures"], row["id"])
                self.assertTrue(row["numsharp_source_urls"], row["id"])
                self.assertTrue(all((ROOT / path).is_file() for path in row["numsharp_source_paths"]))

    def test_instance_only_fields_and_callable_attributes(self):
        for row_id in ("numpy.finfo.bits", "numpy.finfo.eps", "numpy.finfo.nmant",
                       "numpy.iinfo.dtype", "numpy.iinfo.bits", "numpy.ndarray.flags.writeable",
                       "numpy.lib.npyio.NpzFile.files", "numpy.polynomial.polynomial.Polynomial.coef",
                       "numpy.polynomial.polynomial.Polynomial.symbol", "numpy.vectorize.pyfunc"):
            self.assertEqual("property", self.by_id[row_id]["kind"], row_id)
        self.assertEqual("property", self.by_id["numpy.dtype.type"]["kind"])
        self.assertEqual("available", self.by_id["numpy.finfo.eps"]["status"])
        self.assertEqual("missing", self.by_id["numpy.finfo.nmant"]["status"])

    def test_modern_rng_exact_owner_mapping_and_genuine_gaps(self):
        expected = {
            "numpy.random.Generator.normal": "NumSharp.Generator.normal",
            "numpy.random.RandomState.normal": "NumSharp.NumPyRandom.normal",
            "numpy.random.SeedSequence.generate_state": "NumSharp.SeedSequence.generate_state",
            "numpy.random.PCG64.state": "NumSharp.PCG64.state",
            "numpy.nditer.iternext": "NumSharp.np.NDIterator.iternext",
        }
        for row_id, target in expected.items():
            self.assertEqual(target, self.by_id[row_id]["numsharp_target"])
        for row_id in ("numpy.random.Generator.spawn", "numpy.random.PCG64.random_raw",
                       "numpy.random.PCG64.advance", "numpy.random.MT19937.random_raw",
                       "numpy.random.Philox.advance", "numpy.random.PCG64DXSM.jumped"):
            self.assertEqual("missing", self.by_id[row_id]["status"], row_id)

    def test_unrelated_top_level_or_ndarray_member_is_not_credited(self):
        for row_id in ("numpy.ufunc.outer", "numpy.add.reduce", "numpy.add.outer",
                       "numpy.ma.MaskedArray.reshape", "numpy.matrix.reshape",
                       "numpy.recarray.reshape", "numpy.random.PCG64.random_raw"):
            self.assertIsNone(self.by_id[row_id]["numsharp_target"], row_id)
        self.assertEqual("missing", self.by_id["numpy.lib.npyio.NpzFile.keys"]["status"])
        self.assertEqual("alias", self.by_id["numpy.lib.npyio.NpzFile.files"]["availability"])

    def test_all_106_ufunc_exports_have_five_methods_including_aliases(self):
        exports = {name for name, obj in vars(np).items() if not name.startswith("_") and isinstance(obj, np.ufunc)}
        self.assertEqual(106, len(exports))
        for name in exports:
            for method in ("reduce", "accumulate", "reduceat", "outer", "at"):
                row = self.by_id[f"numpy.{name}.{method}"]
                self.assertEqual("ufunc", row["surface"])
                self.assertEqual(f"{name}.{method}", row["name"])
                self.assertEqual("numpy." + name, row["object_type"])
                self.assertEqual(documentation_url("ufunc", method, "method"), row["documentation_url"])

    def test_numpy_protocol_applicability_does_not_claim_invalid_calls(self):
        for row_id in ("numpy.sqrt.reduce", "numpy.matmul.at", "numpy.divmod.reduce", "numpy.modf.at"):
            self.assertEqual("not_applicable", self.by_id[row_id]["applicability"])
        for row_id in ("numpy.add.reduce", "numpy.sqrt.at", "numpy.divmod.outer"):
            self.assertEqual("conditional", self.by_id[row_id]["applicability"])
        # In particular, outer supports multiple outputs; reductions do not.
        self.assertEqual(2, len(np.divmod.outer([1, 2], [1, 2])))
        with self.assertRaises(ValueError):
            np.divmod.reduce([1, 2])

    def test_object_inventory_includes_inherited_declaration_ownership(self):
        self.assertEqual(5, self.inventory["schemaVersion"])
        generic_array = self.inventory["objectTypes"]["NumSharp.Generic.NDArray`1"]
        inherited = next(member for member in generic_array["methods"] if member["name"] == "reshape")
        self.assertIn("NumSharp.NDArray", inherited["declaringTypes"])
        self.assertFalse(any(member["name"] == "GetType" for member in generic_array["methods"]))

    def test_preexisting_flatiter_class_is_enriched_without_changing_scope(self):
        flatiter = {
            "id": "numpy.flatiter", "origin": "numpy", "kind": "class", "status": "missing",
            "support": "missing", "availability": "missing", "in_default_scope": False,
            "surface": "np", "category": "Types", "notes": "Existing namespace class row.",
        }
        excluded = [
            {**flatiter, "id": "numpy.matrix"},
            {**flatiter, "id": "numpy.dtype", "status": "unsupported", "support": "unsupported"},
            {**flatiter, "id": "numpy.flatiter.base", "kind": "property"},
        ]
        snapshots = [dict(row) for row in excluded]
        count = enrich_object_classes([flatiter, *excluded], self.inventory, ROOT,
                                      "https://github.com/SciSharp/NumSharp/blob/master/")
        self.assertEqual(1, count)
        self.assertEqual("available", flatiter["status"])
        self.assertEqual("NumSharp.np.FlatIterator", flatiter["numsharp_target"])
        self.assertIn("src/NumSharp.Core/APIs/np.flatiter.cs", flatiter["numsharp_source_paths"])
        self.assertFalse(flatiter["in_default_scope"])
        self.assertEqual("np", flatiter["surface"])
        self.assertEqual("Types", flatiter["category"])
        self.assertTrue(flatiter["notes"].startswith("Existing namespace class row."))
        self.assertEqual(snapshots, excluded)

    def test_seeded_numpy_global_random_state_is_unchanged(self):
        before = np.random.get_state()
        # Duplicate IDs must be suppressed across prior module/class scans.
        seen = set(self.by_id)
        additional = object_surface_rows(
            np, self.inventory, seen, root=ROOT, source_base_url="https://example.test/",
            signature=signature, documentation_url=documentation_url,
        )
        after = np.random.get_state()
        self.assertEqual([], additional)
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])


if __name__ == "__main__":
    unittest.main()
