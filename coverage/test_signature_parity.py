"""Regression gate for signature parity (signature_parity.py): NumPy's signature against the C# overloads.

Every rule is pinned against NumPy 2.4.2's REAL signatures (``inspect.signature``, the documented first docstring
line of an ndarray method, the callable a numpy.ma wrapper forwards to) and synthetic C# overloads in the inventory's
schema-7 shape, so a rule change fails with the case it broke. The headline cases are the APIs a standalone
comparison found on 2026-10-17 while the artifact rated every one of them "available / exact": np.transpose's
``premute``, np.sum's overloads without out= / initial= / where=, a dtype taken as ``System.Type``, numpy.acos
credited to np.arccos (``np.acos(x)`` does not compile).

Run:
    python -m unittest discover -s coverage -p test_signature_parity.py
"""

from __future__ import annotations

import inspect
import unittest

import numpy as np

import signature_parity as sp


def parameter(text: str) -> dict:
    """Build one schema-7 parameter from its C# text.

    :param text: ``[params ]Type name[ = default]``; the default's kind is read from its spelling: ``null`` /
        ``default`` null, ``true`` / ``false`` bool, a double-quoted string, a single-quoted char, an identifier with
        a dot an enum member, anything else a number.
    :returns: the inventory's parameter dictionary (``name``, ``type``, ``optional``, ``isParams`` and, for an
        optional parameter, ``default`` + ``defaultKind``).
    """
    is_params = text.startswith("params ")
    if is_params:
        text = text[len("params "):]
    head, eq, default = text.partition(" = ")
    type_name, name = head.rsplit(" ", 1)
    entry = {"name": name, "type": type_name, "optional": bool(eq), "isParams": is_params}
    if eq:
        entry["default"] = default
        entry["defaultKind"] = (
            "null" if default in ("null", "default") else "bool" if default in ("true", "false")
            else "string" if default.startswith('"') else "char" if default.startswith("'")
            else "enum" if default[0].isalpha() else "number")
    return entry


def overload(name: str, *texts: str, **flags) -> dict:
    """Build one schema-7 overload.

    :param name: the method's name.
    :param texts: the parameters' C# texts (see ``parameter``).
    :param flags: overrides of ``genericParameters`` / ``obsolete`` / ``hidden``.
    :returns: the inventory's overload dictionary.
    """
    entry = {"signature": f"NumSharp.NDArray {name}({', '.join(texts)})", "returnType": "NumSharp.NDArray",
             "parameters": [parameter(text) for text in texts], "genericParameters": [], "obsolete": False,
             "hidden": False, "static": True, "declaringType": "NumSharp.np"}
    entry.update(flags)
    return entry


def assess(api_id: str, target: str, *overloads: dict, member_kind: str = "method", **row_fields) -> dict | None:
    """Run ``signature_parity.assess`` for one credited NumPy row and a member with the given overloads.

    :param api_id: the NumPy row id.
    :param target: the credited NumSharp target id.
    :param overloads: the member's overloads; none and ``member_kind=None`` mean "not in the inventory".
    :param member_kind: the member's kind (``method``, ``property``), or None for no member at all.
    :param row_fields: row overrides (``kind``, ``availability``, ``python_protocol``, ``origin``, ...).
    :returns: the record, or None when the row takes no part.
    """
    row = {"id": api_id, "origin": "numpy", "kind": "function", "availability": "exact", "numsharp_target": target}
    row.update(row_fields)
    member = None if member_kind is None else {"kind": member_kind, "overloads": list(overloads)}
    return sp.assess(np, row, member)


class SignatureRuleTests(unittest.TestCase):
    """Each matching rule, mark and count of signature_parity, against NumPy's real signatures."""

    @classmethod
    def setUpClass(cls):
        """Refuse another NumPy: the expectations below are NumPy 2.4.2's signatures.

        :raises RuntimeError: when the installed NumPy is not 2.4.2.
        """
        if np.__version__ != "2.4.2":
            raise RuntimeError("Signature parity tests require NumPy 2.4.2.")

    def test_a_renamed_parameter_is_missing_and_foreign(self):
        """np.transpose(a, axes=None) as C#'s transpose(a, premute): ``np.transpose(x, axes=p)`` does not compile."""
        record = assess("numpy.transpose", "NumSharp.np.transpose",
                        overload("transpose", "NDArray a", "int[] premute = null"))
        self.assertEqual("differs", record["status"])
        self.assertEqual(["axes (C#: premute)"], record["missing"])
        self.assertEqual(["premute"], record["foreign"])
        self.assertEqual("MX", sp.marks(record))
        self.assertEqual("transpose(a, axes=None)", record["numpy"])

    def test_positional_only_parameters_match_by_slot_whatever_their_csharp_name(self):
        """np.inner(a, b, /) cannot take a or b by name, so C#'s inner(left, right) is NumPy's signature."""
        record = assess("numpy.inner", "NumSharp.np.inner", overload("inner", "NDArray left", "NDArray right"))
        self.assertEqual("identical", record["status"])
        self.assertNotIn("foreign", record)
        self.assertEqual("", sp.marks(record))

    def test_missing_is_the_union_over_overloads_and_several_overloads_are_marked(self):
        """A NumPy parameter is missing only when NO overload has it; np.sum lacks out=, initial= and where=."""
        record = assess("numpy.sum", "NumSharp.np.sum", overload("sum", "NDArray a"),
                        overload("sum", "NDArray a", "int? axis = null", "DType dtype = null",
                                 "bool keepdims = false"))
        self.assertEqual(["out", "initial", "where"], record["missing"])
        self.assertEqual(2, record["overloads"])
        self.assertEqual("MO", sp.marks(record))
        # The closest overload is the fuller one; its keepdims sits where NumPy's out does.
        self.assertEqual(["keepdims: NumPy #4, C# #3"], record["order"])
        self.assertIn("keepdims = false", record["closest"])

    def test_requiredness_and_default_values_are_closest_overload_differences(self):
        """A required axes, another default value and a missing ndmax=, each where the closest overload shows it."""
        required = assess("numpy.transpose", "NumSharp.np.transpose",
                          overload("transpose", "NDArray a", "int[] axes"))
        self.assertEqual("differs", required["status"])
        self.assertEqual(["axes"], required["required"])
        self.assertEqual("", sp.marks(required), "a closest-overload difference carries no mark letter")
        array = assess("numpy.array", "NumSharp.np.array",
                       overload("array", "NDArray object", "DType dtype = null", "bool copy = true",
                                "char order = 'C'", "int ndmin = 0"))
        self.assertEqual(["order: 'K' vs 'C'"], array["default"])
        self.assertEqual(["ndmax"], array["missing"])
        self.assertEqual(["subok", "like"], array["excluded"])

    def test_a_null_default_where_numpy_has_a_value_is_informational(self):
        """np.full's order='C' spelled `char? order = null` is reported but does not make the row differ."""
        record = assess("numpy.full", "NumSharp.np.full",
                        overload("full", "Shape shape", "object fill_value", "DType dtype = null",
                                 "char? order = null", "string device = null"))
        self.assertEqual("identical", record["status"])
        self.assertEqual(["order: 'C' vs null"], record["null_default"])

    def test_a_dtype_taken_as_NPTypeCode_or_Type_is_marked(self):
        """Since DType Stage B every dtype parameter is one DType; NPTypeCode / Type are leftovers (mark T)."""
        record = assess("numpy.zeros", "NumSharp.np.zeros",
                        overload("zeros", "Shape shape", "System.Type dtype = null", "char order = 'C'",
                                 "string device = null"))
        self.assertEqual("differs", record["status"])
        self.assertEqual(["System.Type dtype"], record["typecode"])
        self.assertEqual("T", sp.marks(record))
        self.assertEqual(["like"], record["excluded"])
        found = sp.typecode_parameters([overload("f", "NumSharp.NPTypeCode? a", "NumSharp.NPTypeCode[] b",
                                                 "System.Type[] c", "NumSharp.DType d", "System.TypeCode e")])
        self.assertEqual(["NumSharp.NPTypeCode? a", "NumSharp.NPTypeCode[] b", "System.Type[] c"], found)

    def test_plan_excluded_parameters_are_listed_but_not_counted(self):
        """A ufunc's casting= / order= / signature= and subok= (docs/plans/api-parameter-parity.md section 3)."""
        record = assess("numpy.add", "NumSharp.np.add",
                        overload("add", "NDArray x1", "NDArray x2", "NDArray out = null", "NDArray where = null",
                                 "DType dtype = null"), kind="ufunc")
        self.assertEqual("identical", record["status"])
        self.assertEqual(["casting", "order", "subok", "signature"], record["excluded"])
        self.assertEqual({"like", "subok"}, sp.policy_for("zeros", False))
        self.assertEqual({"like", "subok", "casting", "order", "signature"}, sp.policy_for("add", True))
        self.assertIn("order", sp.policy_for("argsort", False))
        self.assertIn("einsum_call", sp.policy_for("einsum_path", False))

    def test_star_args_needs_a_params_array(self):
        """np.broadcast_shapes(*args): `params Shape[]` is NumPy's call, two fixed Shapes are not."""
        spelled = assess("numpy.broadcast_shapes", "NumSharp.np.broadcast_shapes",
                         overload("broadcast_shapes", "params Shape[] shapes"))
        self.assertEqual("identical", spelled["status"])
        fixed = assess("numpy.broadcast_shapes", "NumSharp.np.broadcast_shapes",
                       overload("broadcast_shapes", "Shape a", "Shape b"))
        self.assertEqual("differs", fixed["status"])
        self.assertEqual(["*args as Shape a, Shape b (no params array)"], fixed["varargs"])
        self.assertNotIn("foreign", fixed, "a spelling of *args is not a parameter NumPy lacks")

    def test_a_parameter_named_like_kwargs_spells_kwargs_not_star_args(self):
        """np.savez(file, *args, allow_pickle=True, **kwds): a C# `kwds` dictionary is **kwds, never part of *args."""
        record = assess("numpy.savez", "NumSharp.np.savez",
                        overload("savez", "string file", "NDArray[] args",
                                 "System.Collections.Generic.IDictionary<string, NumSharp.NDArray> kwds",
                                 "bool allow_pickle = true"))
        self.assertEqual(["*args as NDArray[] args (no params array)"], record["varargs"])
        self.assertNotIn("foreign", record)
        self.assertNotIn("missing", record)

    def test_keywords_numpy_takes_through_kwargs_are_not_foreign(self):
        """np.clip(a, a_min, a_max, out=None, *, min, max, **kwargs) forwards dtype= and where= to the ufunc."""
        record = assess("numpy.clip", "NumSharp.np.clip",
                        overload("clip", "NDArray a", "NDArray a_min = null", "NDArray a_max = null",
                                 "NDArray out = null", "DType dtype = null", "NDArray min = null",
                                 "NDArray max = null", "NDArray where = null"))
        self.assertEqual("identical", record["status"])
        self.assertNotIn("foreign", record)

    def test_ndarray_methods_compare_against_their_documented_signature(self):
        """inspect folds ndarray.sum's keywords into **kwargs; the documented line is the contract a port follows."""
        self.assertIn("**kwargs", str(inspect.signature(np.ndarray.sum)), "the reason the docstring is read")
        signature = sp.numpy_parameters(np, "numpy.ndarray.sum")
        self.assertEqual("a.sum(axis=None, dtype=None, out=None, *, keepdims=<no value>, initial=<no value>, "
                         "where=<no value>)", signature.text)
        self.assertEqual(["axis", "dtype", "out", "keepdims", "initial", "where"],
                         [param.name for param in signature.params])
        self.assertEqual(sp.P.KEYWORD_ONLY, signature.params[3].kind)

    def test_docstring_signature_lines_parse_brackets_slashes_and_stars(self):
        """``a.view([dtype][, type])``, ``a.swapaxes(axis1, axis2, /)`` and ``*`` as NumPy documents them."""
        view = sp.from_doc("a.view([dtype][, type])\n\nNew view.", "view")
        self.assertEqual([("dtype", "<optional>"), ("type", "<optional>")],
                         [(param.name, param.default) for param in view])
        swapaxes = sp.from_doc("a.swapaxes(axis1, axis2, /)", "swapaxes")
        self.assertEqual([sp.P.POSITIONAL_ONLY] * 2, [param.kind for param in swapaxes])
        minimum = sp.from_doc("a.min(axis=None, out=None, *, keepdims=<no value>)", "min")
        self.assertEqual(sp.P.KEYWORD_ONLY, minimum[2].kind)
        self.assertEqual("axis=None, out=None, *, keepdims=<no value>", sp.format_params(minimum))
        self.assertIsNone(sp.from_doc("Nothing here.", "min"))
        self.assertEqual(["shape=(2, 3)", " order='C'"], sp.split_top("shape=(2, 3), order='C'"))

    def test_numpy_ma_forwarders_read_the_callable_they_forward_to(self):
        """numpy.ma.add forwards to np.add; numpy.ma.copy and MaskedArray.copy to ndarray.copy."""
        add = sp.numpy_parameters(np, "numpy.ma.add")
        self.assertTrue(add.is_ufunc, "the ufunc exclusions apply to the wrapper too")
        self.assertEqual(["a", "b", "out", "where"], [param.name for param in add.params][:4])
        record = assess("numpy.ma.add", "NumSharp.np.ma.add",
                        overload("add", "NDArray a", "NDArray b", "NDArray out = null", "NDArray where = null",
                                 "DType dtype = null"), kind="ufunc")
        self.assertEqual("identical", record["status"])
        self.assertEqual("copy(a, order='C')", sp.numpy_parameters(np, "numpy.ma.copy").text)
        self.assertEqual("copy(order='C')", sp.numpy_parameters(np, "numpy.ma.MaskedArray.copy").text)
        # ndarray.swapaxes is positional-only, so the forwarded `a` can only arrive positionally too.
        self.assertEqual("swapaxes(a, axis1, axis2, /)", sp.numpy_parameters(np, "numpy.ma.swapaxes").text)

    def test_an_alias_spelling_is_marked_N_even_when_the_parameters_match(self):
        """numpy.acos credited to np.arccos: the parameters agree, but NumPy's spelling does not compile."""
        record = assess("numpy.acos", "NumSharp.np.arccos",
                        overload("arccos", "NDArray x", "NDArray out = null", "NDArray where = null",
                                 "DType dtype = null"), kind="ufunc", availability="alias")
        self.assertEqual("differs", record["status"])
        self.assertEqual("alias", record["spelling"])
        self.assertEqual("N", sp.marks(record))

    def test_rows_whose_parameters_cannot_be_aligned_are_not_compared(self):
        """The array is the receiver on one side and an argument on the other, or the credit is a property."""
        astype = assess("numpy.astype", "NumSharp.NDArray.astype", overload("astype", "DType dtype"))
        self.assertEqual("not-compared", astype["status"])
        self.assertIn("receiver", astype["reason"])
        self.assertEqual("astype(x, dtype, /, *, copy=True, device=None)", astype["numpy"])
        prop = assess("numpy.shape", "NumSharp.np.shape", member_kind="property")
        self.assertEqual("credited to a property", prop["reason"])
        aliased = assess("numpy.shape", "NumSharp.np.shape", member_kind="property", availability="alias")
        self.assertEqual("N", sp.marks(aliased), "an alias is marked even when nothing could be compared")

    def test_join_failures_are_reported_as_problems_not_verdicts(self):
        """A credit the index cannot resolve, or a member without overload data, must fail the generator."""
        absent = assess("numpy.sum", "NumSharp.np.sum", member_kind=None)
        self.assertIn(absent["reason"], sp.WIRING_REASONS)
        empty = assess("numpy.sum", "NumSharp.np.sum")
        self.assertIn(empty["reason"], sp.WIRING_REASONS)
        inventory = {"modules": {"np": {"type": "NumSharp.np", "methods": [
            {"name": "inner", "kind": "method", "overloads": [overload("inner", "NDArray a", "NDArray b")]}]}}}
        rows = [{"id": "numpy.inner", "origin": "numpy", "kind": "function", "availability": "exact",
                 "numsharp_target": "NumSharp.np.inner"},
                {"id": "numpy.outer", "origin": "numpy", "kind": "function", "availability": "exact",
                 "numsharp_target": "NumSharp.np.outer"}]
        problems = sp.attach(rows, inventory, np)
        self.assertEqual(["numpy.outer -> NumSharp.np.outer: the credited member is not in the inventory"], problems)
        self.assertEqual("identical", rows[0]["signature"]["status"])

    def test_rows_that_take_no_part_get_no_record(self):
        """Protocols (an operator / indexer / cast in C#), uncredited rows, NumSharp-only rows and non-calls."""
        member = overload("f", "NDArray a")
        self.assertIsNone(assess("numpy.ndarray.__add__", "NumSharp.NDArray.op_Addition", member,
                                 kind="method", python_protocol=True))
        self.assertIsNone(sp.assess(np, {"id": "numpy.add", "origin": "numpy", "kind": "ufunc",
                                         "numsharp_target": None}, None))
        self.assertIsNone(assess("NumSharp.np.evaluate", "NumSharp.np.evaluate", member, origin="numsharp"))
        self.assertIsNone(assess("numpy.pi", "NumSharp.np.pi", member, kind="constant"))
        self.assertIsNone(assess("numpy.random.PCG64", "NumSharp.PCG64", member_kind=None, kind="class"),
                          "a class credited to a C# type is not a call")
        factory = assess("numpy.random.PCG64", "NumSharp.np.random.PCG64",
                         overload("PCG64", "object seed = null"), kind="class")
        self.assertEqual("identical", factory["status"], "a class credited to a factory method is a call")

    def test_the_closest_overload_is_the_fewest_weighted_edits_and_a_plain_one_wins_a_tie(self):
        """What the one surviving overload must become: the fullest, then the non-generic / current / visible."""
        params = sp.numpy_parameters(np, "numpy.transpose").params
        overloads = [overload("transpose", "NDArray a"),
                     overload("transpose", "NDArray a", "int[] axes = null", genericParameters=["T"]),
                     overload("transpose", "NDArray a", "int[] axes = null")]
        index, diff = sp.closest_overload(params, overloads, sp.policy_for("transpose", False))
        self.assertEqual(2, index)
        self.assertFalse(any(diff[kind] for kind in ("missing", "renamed", "extra", "order", "required")))
        index, _ = sp.closest_overload(params, overloads[:2], sp.policy_for("transpose", False))
        self.assertEqual(1, index, "a generic overload still beats one that lacks axes=")

    def test_defaults_compare_as_values(self):
        """A bool never equals a number, NaN equals NaN, an enum member matches NumPy's string, a soft default
        (a class or object C# cannot write) matches only null, np._NoValue matches anything."""
        self.assertFalse(sp.same_value(True, 1.0))
        self.assertTrue(sp.same_value(float("nan"), float("nan")))
        self.assertTrue(sp.same_value("same_kind", ("enum", "NPY_CASTING.SameKind")))
        self.assertFalse(sp.same_value("safe", ("enum", "NPY_CASTING.NPY_SAFE_CASTING")))
        np_param = sp.Param("dtype", sp.P.POSITIONAL_OR_KEYWORD, "<class 'float'>")
        self.assertEqual((None, ""), sp.compare_default(np_param, parameter("DType dtype = null")))
        self.assertEqual("default", sp.compare_default(np_param, parameter("int dtype = 3"))[0])
        where = sp.Param("where", sp.P.KEYWORD_ONLY, "True")
        self.assertEqual((None, ""), sp.compare_default(where, parameter("NDArray where = null")),
                         "where=True is NumSharp's null")
        no_value = sp.Param("keepdims", sp.P.KEYWORD_ONLY, "<no value>")
        self.assertEqual((None, ""), sp.compare_default(no_value, parameter("bool keepdims = false")))
        # A tuple C# cannot write: `default` reads as "C# has no value" (informational), a constant as another value.
        shape = sp.Param("shape", sp.P.POSITIONAL_OR_KEYWORD, "(2, 3)")
        self.assertEqual("null_default", sp.compare_default(shape, parameter("Shape shape = default"))[0])
        self.assertEqual("default", sp.compare_default(shape, parameter("int shape = 6"))[0])
        self.assertEqual(-1.0, sp.cs_default_value(parameter("int axis = -1")))
        self.assertEqual("K", sp.cs_default_value(parameter("char order = 'K'")))
        self.assertEqual("safe", sp.cs_default_value(parameter('string casting = "safe"')))

    def test_marks_classify_and_counts_read_the_same_records(self):
        """The summary's counts, the CSV marks and the dashboard all read these three functions."""
        records = [
            {"status": "identical", "spelling": "exact", "overloads": 1},
            {"status": "differs", "spelling": "exact", "overloads": 3, "missing": ["out"], "foreign": ["x"],
             "typecode": ["System.Type dtype"], "order": ["out: NumPy #3, C# #2"]},
            {"status": "differs", "spelling": "alias", "overloads": 1},
            {"status": "identical", "spelling": "exact", "overloads": 1, "excluded": ["like"]},
            {"status": "not-compared", "spelling": "alias", "reason": "credited to a property"},
        ]
        self.assertEqual(["", "MXOT", "N", "", "N"], [sp.marks(record) for record in records])
        rows = [{"signature": record} for record in records] + [{}]
        self.assertEqual(["identical", "differs", "differs", "identical", "not-compared", "none"],
                         [sp.classify(row) for row in rows])
        counts = sp.counts(rows)
        self.assertEqual((2, 2, 1), (counts["signature_identical"], counts["signature_differs"],
                                     counts["signature_not_compared"]))
        self.assertEqual((1, 1, 1, 1, 2, 1, 1), (
            counts["signature_missing"], counts["signature_foreign"], counts["signature_overloaded"],
            counts["signature_typecode"], counts["signature_other_name"], counts["signature_excluded_only"],
            counts["signature_order"]))
        self.assertEqual(33.3, counts["signature_percent"], "identical rows over EVERY row of the scope")
        self.assertEqual(0.0, sp.counts([])["signature_percent"])

    def test_member_index_reads_nested_type_names_and_prefers_the_module_facade(self):
        """Targets are the CLR type name with ``+`` read as ``.``; a facade's DeclaredOnly entry wins."""
        inventory = {
            "objectTypes": {"NumSharp.np": {"type": "NumSharp.np", "methods": [{"name": "sum", "kind": "method",
                                                                               "overloads": []}]},
                            "NumSharp.np+random": {"type": "NumSharp.np+random", "methods": [
                                {"name": "seed", "kind": "method", "overloads": [overload("seed", "int s")]}]}},
            "modules": {"np": {"type": "NumSharp.np", "methods": [
                {"name": "sum", "kind": "method", "overloads": [overload("sum", "NDArray a")]}]}},
        }
        index = sp.member_index(inventory)
        self.assertEqual(1, len(index["NumSharp.np.sum"]["overloads"]))
        self.assertIn("NumSharp.np.random.seed", index)


if __name__ == "__main__":
    unittest.main()
