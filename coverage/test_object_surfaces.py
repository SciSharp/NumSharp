"""Regression gate for object-member inventory (NumPy 2.4.2 + built inventory tool).

Run after building NumSharp.Tools.ApiInventory:
    python -m unittest discover -s coverage -p test_object_surfaces.py
"""

from __future__ import annotations

import importlib
import inspect
import json
import operator
import subprocess
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

import generate_coverage as coverage
from numpy_documentation import documentation_url
from object_surfaces import (REFUSED_PROTOCOLS, REVIEWED_PROTOCOL_ADAPTERS, class_member_rows,
                             enrich_object_classes, object_surface_rows, protocol_credit, ufunc_method_rejection,
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
                       "numpy.ma.MaskedArray.ctypes",
                       "numpy.recarray.reshape", "numpy.random.PCG64.ctypes"):
            self.assertIsNone(self.by_id[row_id]["numsharp_target"], row_id)
        self.assertEqual("missing", self.by_id["numpy.lib.npyio.NpzFile.keys"]["status"])
        self.assertEqual("alias", self.by_id["numpy.lib.npyio.NpzFile.files"]["availability"])

    def test_masked_array_and_vectorize_members_credit_their_numsharp_owners(self):
        """numpy.ma.MaskedArray maps to NDMaskedArray and numpy.vectorize to Vectorized.

        Without an owner type every member read missing (94 named MaskedArray rows, NumSharp implements 77);
        each credit comes from a member NDMaskedArray declares itself (it has no NDArray base), and the
        members NumSharp lacks stay missing. Since generator 1.14.0 the class's 86 Python protocols are
        listed too, 26 of them credited through NDMaskedArray's operators, indexer and ToString.
        """
        masked = [row for row in self.rows if row["object_type"] == "numpy.ma.MaskedArray" and row["kind"] != "class"]
        named = [row for row in masked if not row.get("python_protocol")]
        credited = [row for row in named if row["status"] == "available"]
        self.assertEqual(94, len(named))
        self.assertEqual(77, len(credited))
        for row in credited:
            self.assertEqual("NumSharp.NDMaskedArray." + row["name"], row["numsharp_target"], row["id"])
            self.assertTrue(row["numsharp_source_paths"], row["id"])
        protocols = [row for row in masked if row.get("python_protocol")]
        self.assertEqual(86, len(protocols))
        aliases = [row for row in protocols if row["status"] == "available"]
        self.assertEqual(26, len(aliases))
        for row in aliases:
            self.assertEqual("alias", row["availability"], row["id"])
            self.assertTrue(row["numsharp_target"].startswith("NumSharp.NDMaskedArray."), row["id"])
            self.assertTrue(row["numsharp_source_paths"], row["id"])
        for gap in ("ctypes", "dump", "dumps", "flags", "setfield", "toflex", "torecords", "sharedmask"):
            self.assertEqual("missing", self.by_id[f"numpy.ma.MaskedArray.{gap}"]["status"], gap)
        self.assertEqual("NumSharp.NDMaskedArray.filled", self.by_id["numpy.ma.MaskedArray.filled"]["numsharp_target"])
        self.assertEqual("NumSharp.Vectorized.signature", self.by_id["numpy.vectorize.signature"]["numsharp_target"])
        self.assertEqual("missing", self.by_id["numpy.vectorize.pyfunc"]["status"])

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
        self.assertEqual(6, self.inventory["schemaVersion"])
        # Schema 6: every type carries its operators, each overload with its operand types and declaring type.
        self.assertTrue(all(isinstance(entry.get("operators"), list) for entry in self.inventory["objectTypes"].values()))
        ndarray_add = next(op for op in self.inventory["objectTypes"]["NumSharp.NDArray"]["operators"]
                           if op["name"] == "op_Addition")
        self.assertIn(["NumSharp.NDArray", "NumSharp.NDArray"], [o["parameterTypes"] for o in ndarray_add["overloads"]])
        self.assertTrue(all(o["declaringType"] == "NumSharp.NDArray" for o in ndarray_add["overloads"]))
        matrix_add = next(op for op in self.inventory["objectTypes"]["NumSharp.NDMatrix"]["operators"]
                          if op["name"] == "op_Addition")
        # FlattenHierarchy hands NDMatrix NDArray's static operators too; the declaring type tells them apart.
        self.assertEqual({"NumSharp.NDArray", "NumSharp.NDMatrix"}, {o["declaringType"] for o in matrix_add["overloads"]})
        generic_array = self.inventory["objectTypes"]["NumSharp.Generic.NDArray`1"]
        inherited = next(member for member in generic_array["methods"] if member["name"] == "reshape")
        self.assertIn("NumSharp.NDArray", inherited["declaringTypes"])
        self.assertFalse(any(member["name"] == "GetType" for member in generic_array["methods"]))

    def test_ndarray_protocols_credit_csharp_operators_indexers_and_casts(self):
        """ndarray's Python protocols are credited by NDArray's own members, operators, indexer and casts.

        A protocol NumSharp spells with a member of the protocol's name is exact; one C# spells with an operator,
        a cast or ToString is an alias. Python operators without a C# symbol (//, **, @, abs) and the in-place
        protocols, which C#'s compound assignment does not implement (`a += b` rebinds `a`), stay missing.
        """
        expected = {
            "__add__": ("alias", "op_Addition"), "__radd__": ("alias", "op_Addition"),
            "__rsub__": ("alias", "op_Subtraction"), "__truediv__": ("alias", "op_Division"),
            "__mod__": ("alias", "op_Modulus"), "__xor__": ("alias", "op_ExclusiveOr"),
            "__lshift__": ("alias", "op_LeftShift"), "__lt__": ("alias", "op_LessThan"),
            "__eq__": ("alias", "op_Equality"), "__neg__": ("alias", "op_UnaryNegation"),
            "__invert__": ("alias", "op_OnesComplement"), "__int__": ("alias", "op_Explicit"),
            "__float__": ("alias", "op_Explicit"), "__complex__": ("alias", "op_Explicit"),
            "__bool__": ("alias", "op_Explicit"), "__str__": ("alias", "ToString"), "__repr__": ("alias", "ToString"),
            "__getitem__": ("exact", "__getitem__"), "__setitem__": ("exact", "__setitem__"),
            "__len__": ("exact", "__len__"), "__iter__": ("exact", "__iter__"), "__contains__": ("exact", "__contains__"),
        }
        for name, (availability, member) in expected.items():
            row = self.by_id[f"numpy.ndarray.{name}"]
            self.assertEqual(("available", availability, "NumSharp.NDArray." + member),
                             (row["status"], row["availability"], row["numsharp_target"]), name)
            self.assertTrue(row["python_protocol"], name)
            self.assertTrue(row["numsharp_source_paths"], name)
            self.assertEqual("ndarray", row["surface"], name)
        for name in ("__iadd__", "__imul__", "__itruediv__", "__floordiv__", "__pow__", "__matmul__", "__abs__",
                     "__index__", "__divmod__", "__array__", "__array_interface__", "__dlpack__", "__copy__"):
            self.assertEqual("missing", self.by_id[f"numpy.ndarray.{name}"]["status"], name)
        # A protocol NumPy fills only to refuse, and one NumPy declares absent.
        self.assertEqual("not_applicable", self.by_id["numpy.ndarray.__delitem__"]["applicability"])
        self.assertNotIn("numpy.ndarray.__hash__", self.by_id)
        # The invented ids of the hand-written interop list are gone.
        self.assertFalse(any(row_id.startswith("numpy.ndarray.interop.") for row_id in self.by_id))

    def test_reflected_protocols_need_an_overload_for_another_left_operand(self):
        """`x + p` reaches __radd__ only when x is not a poly1d, so `operator +(poly1d, poly1d)` does not credit it.

        poly1d declares `operator *(double, poly1d)`, the reflected product, and no addition with another type on
        the left (C# would bind `2.0 + p` to NDArray's operator through poly1d's conversion to its coefficients,
        adding 2 to every coefficient instead of to the constant term).
        """
        self.assertEqual("available", self.by_id["numpy.poly1d.__add__"]["status"])
        self.assertEqual("missing", self.by_id["numpy.poly1d.__radd__"]["status"])
        self.assertEqual(("available", "NumSharp.poly1d.op_Multiply"),
                         (self.by_id["numpy.poly1d.__rmul__"]["status"], self.by_id["numpy.poly1d.__rmul__"]["numsharp_target"]))
        self.assertEqual("available", self.by_id["numpy.matrix.__radd__"]["status"])  # operator +(object, NDMatrix)
        # NDArray has only operator <<(NDArray, NDArray) and (NDArray, object), but converts every C# scalar
        # implicitly, so `5 << a` compiles and __rlshift__ is credited; poly1d converts nothing to itself.
        self.assertEqual("available", self.by_id["numpy.ndarray.__rlshift__"]["status"])
        self.assertIn("converted implicitly", self.by_id["numpy.ndarray.__rlshift__"]["notes"])
        self.assertEqual(np.poly1d([1, 2]) + 2, 2 + np.poly1d([1, 2]))  # NumPy: polyadd from either side

    def test_matrix_and_masked_array_protocols_follow_their_own_types(self):
        """NDMatrix's Python-named members credit exactly; NDMaskedArray has no == operator, so __eq__ stays missing."""
        for name in ("__mul__", "__imul__", "__pow__", "__ipow__", "__rmul__", "__rpow__", "__array_priority__"):
            row = self.by_id[f"numpy.matrix.{name}"]
            self.assertEqual(("available", "exact", "NumSharp.NDMatrix." + name),
                             (row["status"], row["availability"], row["numsharp_target"]), name)
        self.assertEqual("NumSharp.NDMatrix.op_Addition", self.by_id["numpy.matrix.__add__"]["numsharp_target"])
        # C# compiles `m1 == m2` for NDMaskedArray as reference equality, which is not NumPy's elementwise ==.
        for name in ("__eq__", "__ne__", "__lshift__", "__pos__"):
            self.assertEqual("missing", self.by_id[f"numpy.ma.MaskedArray.{name}"]["status"], name)
        self.assertEqual("NumSharp.NDMaskedArray.op_LessThan", self.by_id["numpy.ma.MaskedArray.__lt__"]["numsharp_target"])

    def test_reviewed_protocol_adapters_credit_only_what_numsharp_documents(self):
        """Every reviewed adapter credits its row; a ToString that ports repr() never credits __str__."""
        for (owner, name), (member, _fragment, _evidence) in REVIEWED_PROTOCOL_ADAPTERS.items():
            row = self.by_id[f"numpy.{owner}.{name}"]
            self.assertEqual(("available", "alias"), (row["status"], row["availability"]), row["id"])
            self.assertTrue(row["numsharp_target"].endswith("." + member), row["id"])
            self.assertIn("Reviewed protocol adapter", row["notes"], row["id"])
            self.assertTrue(row["numsharp_source_paths"], row["id"])
        # finfo.ToString and poly1d.ToString render NumPy's repr; Generator.ToString NumPy's str (repr adds the
        # address); NpzFile's enumeration yields key/value pairs where Python's iter() yields keys.
        for row_id in ("numpy.finfo.__str__", "numpy.poly1d.__str__", "numpy.random.Generator.__repr__",
                       "numpy.lib.npyio.NpzFile.__iter__", "numpy.nditer.__next__", "numpy.ndindex.__next__"):
            self.assertEqual("missing", self.by_id[row_id]["status"], row_id)
        self.assertEqual("NumSharp.np.FlatIterator.next", self.by_id["numpy.flatiter.__next__"]["numsharp_target"])

    def test_protocol_credit_rules_on_synthetic_types(self):
        """The general C# spellings, on a synthetic type: what each rule accepts and what it refuses."""
        owner = "NumSharp.Owner"

        def op(name, params, ret=owner, declaring=owner):
            return {"name": name, "overloads": [{"signature": f"{ret} {name}()", "returnType": ret,
                                                 "parameterTypes": params, "declaringType": declaring}]}

        def grouped(entries):
            # The inventory lists each CLR operator name once, with all its overloads.
            merged = {}
            for entry in entries:
                merged.setdefault(entry["name"], []).extend(entry["overloads"])
            return [{"name": name, "overloads": overloads} for name, overloads in merged.items()]

        inventory = {"methods": [], "properties": [], "operators": grouped([
            op("op_Addition", [owner, owner]),
            op("op_Subtraction", ["double", owner]),
            # A base type's cast applies to the owner; a cast TO the owner converts nothing from it.
            op("op_Explicit", ["NumSharp.Base"], "int", "NumSharp.Base"),
            op("op_Implicit", ["double"], owner),
            # A cast the owner declares FROM another type converts nothing from the owner.
            op("op_Explicit", ["NumSharp.Other"], "double"),
            # A base type's operator returns the base class, so it does not implement the owner's protocol.
            op("op_Multiply", ["NumSharp.Base", "NumSharp.Base"], "NumSharp.Base", "NumSharp.Base"),
        ])}

        def credit(name, type_inventory=inventory):
            return protocol_credit("synthetic", name, owner, type_inventory)

        self.assertEqual("op_Addition", credit("__add__").member)
        # Only (Owner, Owner): Python calls __radd__ for another left type, which converts implicitly here (the
        # op_Implicit from double), so C# binds the owner's operator; without that conversion it would not.
        self.assertEqual("op_Addition", credit("__radd__").member)
        without_conversion = {**inventory, "operators": [entry for entry in inventory["operators"]
                                                         if entry["name"] != "op_Implicit"]}
        self.assertIsNone(credit("__radd__", without_conversion))
        self.assertEqual("op_Subtraction", credit("__rsub__").member)  # (double, Owner)
        self.assertIsNone(credit("__sub__"))
        self.assertEqual("op_Explicit", credit("__int__").member)
        self.assertIsNone(credit("__float__"))
        self.assertIsNone(credit("__mul__"))
        self.assertIsNone(credit("__iadd__"))  # no C# spelling writes into the left operand
        self.assertIsNone(credit("__getitem__"))  # no indexer
        indexed = {"properties": [{"name": "Item", "kind": "property", "signatures": ["int Item[int i] { get; }"]}]}
        self.assertEqual("Item", credit("__getitem__", indexed).member)
        self.assertIsNone(credit("__setitem__", indexed))  # a getter-only indexer cannot assign
        # numpy.matrix's casts are NDArray's, inherited (C# applies a base class's conversion to a subclass).
        row = self.by_id["numpy.matrix.__int__"]
        self.assertEqual(("available", "NumSharp.NDMatrix.op_Explicit"), (row["status"], row["numsharp_target"]))
        self.assertTrue(row["numsharp_source_paths"])

    def test_a_stale_reviewed_adapter_fails_the_run(self):
        """A reviewed adapter whose NumSharp member disappeared raises instead of silently reading missing."""
        with self.assertRaisesRegex(ValueError, "REVIEWED_PROTOCOL_ADAPTERS"):
            protocol_credit("poly1d", "__call__", "NumSharp.poly1d", {"methods": [], "properties": [], "operators": []})
        with self.assertRaisesRegex(ValueError, "REVIEWED_PROTOCOL_ADAPTERS"):
            protocol_credit("ndarray", "__repr__", "NumSharp.NDArray",
                            {"methods": [{"name": "ToString", "kind": "method", "signatures": ["string ToString()"]}]})

    def test_refused_protocols_are_refused_by_numpy(self):
        """Every REFUSED_PROTOCOLS entry is a protocol NumPy rejects for every call, checked on a real instance."""
        with tempfile.TemporaryDirectory() as folder:
            mapped = np.memmap(Path(folder) / "m.bin", dtype=np.int64, mode="w+", shape=(3,))
            structured = np.zeros(2, dtype=[("a", "i4"), ("b", "f8")])
            # numpy.matrix warns on construction (PendingDeprecationWarning, the class NumPy discourages); the test
            # needs one instance, not the warning in the run's output.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", PendingDeprecationWarning)
                matrix = np.matrix([[1, 2]])
            instances = {
                "ndarray": np.arange(3), "matrix": matrix, "recarray": structured.view(np.recarray),
                "char.chararray": np.char.array(["a", "b"]), "memmap": mapped,
                "ma.MaskedArray": np.ma.array([1, 2], mask=[0, 1]),
                "ma.mrecords.MaskedRecords": importlib.import_module("numpy.ma.mrecords").fromarrays([[1, 2], [3.0, 4.0]]),
                "record": np.rec.array(structured)[0], "flatiter": np.arange(3).flat,
                "nditer": np.nditer(np.arange(3), op_flags=[["readwrite"]]), "ndarray.flags": np.arange(3).flags,
                "void": structured[0], "dtypes.StringDType": np.dtypes.StringDType(),
            }
            orderings = {"__lt__": operator.lt, "__le__": operator.le, "__gt__": operator.gt, "__ge__": operator.ge}
            for (owner, name) in REFUSED_PROTOCOLS:
                value = instances[owner]
                if name == "__delitem__":
                    for key in (0, "a", "WRITEABLE", Ellipsis):
                        with self.assertRaises((ValueError, TypeError, KeyError, AttributeError), msg=(owner, key)):
                            del value[key]
                else:
                    with self.assertRaises(TypeError, msg=(owner, name)):
                        orderings[name](value, value)
                if f"numpy.{owner}.{name}" in self.by_id:
                    self.assertEqual("not_applicable", self.by_id[f"numpy.{owner}.{name}"]["applicability"], owner)
            # Release the memmap before the folder is removed: Windows refuses to delete a mapped file.
            instances.clear()
            del mapped

    def test_class_members_credit_the_type_matched_numsharp_type(self):
        """A class's members are credited against the NumSharp type of its simple name, case-sensitively.

        NumSharp.AxisError has Axis and NDim, so numpy.exceptions.AxisError.axis and .ndim stay missing; mapping
        the class to a type that does declare ndim (NumSharp.DType, a stand-in) credits it with source evidence.
        """
        class_row = {"id": "numpy.exceptions.AxisError", "origin": "numpy", "kind": "class", "name": "AxisError",
                     "category": "Exceptions & warnings", "disposition": "candidate"}
        common = dict(root=ROOT, source_base_url="https://example.test/", signature=signature,
                      documentation_url=documentation_url)
        real = {row["id"]: row for row in class_member_rows(
            np, [class_row], set(), self.inventory, types_by_simple_name={"AxisError": "NumSharp.AxisError"}, **common)}
        self.assertEqual({"numpy.exceptions.AxisError.axis", "numpy.exceptions.AxisError.ndim",
                          "numpy.exceptions.AxisError.__str__"}, set(real))
        self.assertTrue(all(row["status"] == "missing" for row in real.values()))
        self.assertIn("NumSharp.AxisError", real["numpy.exceptions.AxisError.axis"]["notes"])
        stand_in = {row["id"]: row for row in class_member_rows(
            np, [class_row], set(), self.inventory, types_by_simple_name={"AxisError": "NumSharp.DType"}, **common)}
        ndim = stand_in["numpy.exceptions.AxisError.ndim"]
        self.assertEqual(("available", "exact", "NumSharp.DType.ndim"),
                         (ndim["status"], ndim["availability"], ndim["numsharp_target"]))
        self.assertTrue(any("DType" in path for path in ndim["numsharp_source_paths"]))
        self.assertEqual("missing", stand_in["numpy.exceptions.AxisError.axis"]["status"])

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
