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

import generate_coverage as coverage
from numpy_documentation import documentation_url
from object_surfaces import (enrich_object_classes, object_surface_rows, ufunc_method_rejection,
                             ufunc_protocol_methods)


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
        # The generator's own namespace list, so these rows are exactly the artifact's rows.
        cls.rows = object_surface_rows(
            np, cls.inventory, set(), root=ROOT,
            source_base_url="https://github.com/SciSharp/NumSharp/blob/master/",
            signature=signature, documentation_url=documentation_url,
            ufunc_namespaces=coverage.UFUNC_PROTOCOL_NAMESPACES,
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
            "numpy.random.PCG64.advance": "NumSharp.PCG64.advance",
            "numpy.random.PCG64.random_raw": "NumSharp.PCG64.random_raw",
            "numpy.random.MT19937.random_raw": "NumSharp.MT19937.random_raw",
            "numpy.random.Philox.advance": "NumSharp.Philox.advance",
            "numpy.random.PCG64DXSM.jumped": "NumSharp.PCG64DXSM.jumped",
            "numpy.random.SFC64.state": "NumSharp.SFC64.state",
            "numpy.random.Generator.spawn": "NumSharp.Generator.spawn",
            "numpy.nditer.iternext": "NumSharp.np.NDIterator.iternext",
        }
        for row_id, target in expected.items():
            self.assertEqual(target, self.by_id[row_id]["numsharp_target"])
            self.assertEqual("available", self.by_id[row_id]["status"], row_id)
        # The C-level interop handles have no managed counterpart: genuine, deliberate gaps.
        for row_id in ("numpy.random.PCG64.ctypes", "numpy.random.Philox.cffi",
                       "numpy.random.SFC64.capsule", "numpy.random.BitGenerator.ctypes"):
            self.assertEqual("missing", self.by_id[row_id]["status"], row_id)

    def test_matrix_members_map_to_ndmatrix(self):
        # numpy.matrix is NumSharp.NDMatrix (src/NumSharp.Core/Matrixlib): its own members, the ndarray members
        # it re-declares to return a matrix, and the ones it inherits unchanged all resolve on that owner.
        expected = {
            "numpy.matrix.T": "NumSharp.NDMatrix.T",
            "numpy.matrix.H": "NumSharp.NDMatrix.H",
            "numpy.matrix.I": "NumSharp.NDMatrix.I",
            "numpy.matrix.A1": "NumSharp.NDMatrix.A1",
            "numpy.matrix.getA": "NumSharp.NDMatrix.getA",
            "numpy.matrix.sum": "NumSharp.NDMatrix.sum",
            "numpy.matrix.reshape": "NumSharp.NDMatrix.reshape",
        }
        for row_id, target in expected.items():
            self.assertEqual(target, self.by_id[row_id]["numsharp_target"], row_id)
            self.assertEqual("available", self.by_id[row_id]["status"], row_id)

    def test_unrelated_top_level_or_ndarray_member_is_not_credited(self):
        # NumSharp's np.add / np.ma.add are functions: a namesake never credits an object member.
        for row_id in ("numpy.ufunc.outer", "numpy.add.reduce", "numpy.add.outer",
                       "numpy.add.resolve_dtypes", "numpy.ma.add.reduce", "numpy.strings.add.reduce",
                       "numpy.ma.MaskedArray.reshape",
                       "numpy.recarray.reshape", "numpy.random.PCG64.ctypes"):
            self.assertIsNone(self.by_id[row_id]["numsharp_target"], row_id)
        self.assertEqual("missing", self.by_id["numpy.lib.npyio.NpzFile.keys"]["status"])
        self.assertEqual("alias", self.by_id["numpy.lib.npyio.NpzFile.files"]["availability"])

    def test_ufunc_methods_are_read_from_the_ufunc_type(self):
        """The per-export method list is numpy.ufunc's public methods, not a hand-written list.

        The hand-written list had five names and missed resolve_dtypes; properties stay out
        because each is one contract for every ufunc (numpy.ufunc.<property> rows carry them).
        """
        methods = ufunc_protocol_methods(np)
        self.assertEqual(("accumulate", "at", "outer", "reduce", "reduceat", "resolve_dtypes"), methods)
        for prop in ("identity", "nin", "nout", "nargs", "ntypes", "types", "signature"):
            self.assertNotIn(prop, methods)
            self.assertIn(f"numpy.ufunc.{prop}", self.by_id)
        self.assertTrue(all(callable(getattr(np.add, method)) for method in methods))

    def test_all_106_ufunc_exports_have_every_ufunc_method_including_aliases(self):
        """Every top-level ufunc export (aliases included) has a row per ufunc method."""
        exports = {name for name, obj in vars(np).items() if not name.startswith("_") and isinstance(obj, np.ufunc)}
        self.assertEqual(106, len(exports))
        for name in exports:
            for method in ufunc_protocol_methods(np):
                row = self.by_id[f"numpy.{name}.{method}"]
                self.assertEqual("ufunc", row["surface"])
                self.assertEqual(f"{name}.{method}", row["name"])
                self.assertEqual("numpy." + name, row["object_type"])
                self.assertEqual(documentation_url("ufunc", method, "method"), row["documentation_url"])
                self.assertEqual("missing", row["status"])

    def test_submodule_ufunc_exports_have_every_ufunc_method(self):
        """Ufuncs exported by a compared submodule (numpy.strings, numpy.char) are protocol owners too.

        Scanning only numpy's top level left numpy.strings.str_len.at or numpy.char.add.reduce
        out of the catalog although they are the same protocol as numpy.add.reduce.
        """
        expected_exports = {"strings": 17, "char": 11}
        for path, surface in coverage.UFUNC_PROTOCOL_NAMESPACES:
            module = coverage.resolve_submodule(np, path)
            names = getattr(module, "__all__", None) or dir(module)
            ufuncs = sorted({name for name in names
                             if not name.startswith("_") and isinstance(getattr(module, name, None), np.ufunc)})
            self.assertEqual(expected_exports.get(surface, 0), len(ufuncs), surface)
            for name in ufuncs:
                for method in ufunc_protocol_methods(np):
                    row = self.by_id[f"numpy.{surface}.{name}.{method}"]
                    self.assertEqual(surface, row["surface"])
                    self.assertEqual(f"numpy.{surface}.{name}", row["object_type"])
                    self.assertEqual("Universal function objects", row["category"])
        # Applicability follows the ufunc's shape exactly as it does for top-level ufuncs, and
        # "conditional" means the dtype still decides: a variable-width StringDType reduction
        # concatenates, while fixed-width <U1 fails dtype resolution (the result needs <U2).
        self.assertEqual("conditional", self.by_id["numpy.strings.add.reduce"]["applicability"])
        self.assertEqual("abc", np.strings.add.reduce(np.array(["a", "b", "c"], dtype=np.dtypes.StringDType())))
        with self.assertRaises(TypeError):
            np.strings.add.reduce(np.array(["a", "b", "c"]))
        self.assertEqual("not_applicable", self.by_id["numpy.strings.isalpha.reduce"]["applicability"])
        with self.assertRaises(ValueError):
            np.strings.isalpha.reduce(np.array(["a", "b"]))

    def test_masked_ufunc_wrappers_have_rows_for_the_methods_they_implement(self):
        """numpy.ma's masked ufunc wrappers get a row per ufunc method they actually implement.

        NumPy gives each wrapper class only part of the protocol and removes some per instance
        (numpy.ma sets the comparisons' reduce to None), so a row exists only for a callable method.
        """
        rows = {row["id"]: row for row in self.rows if row["id"].startswith("numpy.ma.")
                and row["name"].rpartition(".")[2] in ufunc_protocol_methods(np)
                and row["object_type"] != "numpy.ma.MaskedArray"}
        self.assertEqual(49, len(rows))
        self.assertEqual(19, len({row["object_type"] for row in rows.values()}))
        expected = {"add": {"reduce", "accumulate", "outer"}, "equal": {"accumulate", "outer"},
                    "maximum": {"reduce", "outer"}, "minimum": {"reduce", "outer"},
                    "hypot": {"reduce", "accumulate", "outer"}, "less": {"accumulate", "outer"}}
        for name, methods in expected.items():
            found = {row["name"].rpartition(".")[2] for row in rows.values() if row["object_type"] == f"numpy.ma.{name}"}
            self.assertEqual(methods, found, name)
        for absent in ("numpy.ma.equal.reduce", "numpy.ma.maximum.accumulate", "numpy.ma.divide.reduce",
                       "numpy.ma.sqrt.at", "numpy.ma.add.at", "numpy.ma.add.resolve_dtypes"):
            self.assertNotIn(absent, self.by_id, absent)
        for row in rows.values():
            self.assertEqual(("missing", "conditional", "ma"), (row["status"], row["applicability"], row["surface"]))
        # The rows describe NumPy's real behaviour.
        data = np.ma.array([1, 2, 3], mask=[False, True, False])
        self.assertEqual(4, np.ma.add.reduce(data))
        self.assertEqual([1, 1, 4], np.ma.add.accumulate(data).filled(1).tolist())
        self.assertEqual(3, np.ma.maximum.reduce(data))
        self.assertIsNone(np.ma.equal.reduce)
        self.assertFalse(hasattr(np.ma.maximum, "accumulate"))

    def test_resolve_dtypes_applies_to_every_ufunc_shape(self):
        """resolve_dtypes has no shape rule: NumPy resolves it for unary, multi-output and core-signature ufuncs."""
        f8 = np.dtype("float64")
        cases = {"add": (f8, f8, None), "sqrt": (f8, None), "divmod": (f8, f8, None, None),
                 "matmul": (f8, f8, None)}
        for name, dtypes in cases.items():
            self.assertEqual("conditional", self.by_id[f"numpy.{name}.resolve_dtypes"]["applicability"], name)
            self.assertIsNone(ufunc_method_rejection(getattr(np, name), "resolve_dtypes"))
            self.assertEqual(len(dtypes), len(getattr(np, name).resolve_dtypes(dtypes)), name)
        # A method without a reviewed rule is never rejected by guesswork.
        self.assertIsNone(ufunc_method_rejection(np.matmul, "not_a_ufunc_method"))

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
            {**flatiter, "id": "numpy.recarray"},
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
            ufunc_namespaces=coverage.UFUNC_PROTOCOL_NAMESPACES,
        )
        after = np.random.get_state()
        self.assertEqual([], additional)
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])


if __name__ == "__main__":
    unittest.main()
