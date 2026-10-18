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
from pathlib import Path

import numpy as np

import signature_gap as gap_module
import signature_parity as sp

ROOT = Path(__file__).resolve().parents[1]


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
        # An [Optional] parameter without a constant (the inventory's "missing" kind): the member reads an omitted
        # argument as NumPy's default, whatever NumPy's default is ([] for fromroots' domain, 0 for trim's tol).
        missing = {"name": "domain", "type": "object", "optional": True, "isParams": False,
                   "default": "Type.Missing", "defaultKind": "missing"}
        self.assertEqual((None, ""), sp.compare_default(sp.Param("domain", sp.P.POSITIONAL_OR_KEYWORD, "[]"), missing))
        self.assertEqual((None, ""), sp.compare_default(sp.Param("tol", sp.P.POSITIONAL_OR_KEYWORD, "0"), missing))
        self.assertEqual(("missing", "Type.Missing"), sp.cs_default_value(missing))
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


# ---- The API diff gap map joined onto the catalog (generator 1.16.0) ----

#: The CLR types of the groups the synthetic gap maps below use (the dump's module types and the object surfaces').
GROUP_TYPES = {"np": "NumSharp.np", "ndarray": "NumSharp.NDArray", "np.random": "NumSharp.NumPyRandom",
               "np.matrix": "NumSharp.NDMatrix", "np.char.chararray": "NumSharp.NDCharArray",
               "np.random.MT19937": "NumSharp.MT19937", gap_module.TYPED_NDARRAY_GROUP: gap_module.TYPED_NDARRAY_TYPE}


def gap_row(module: str, name: str, numpy_kind: str = "function", **cells: str) -> dict:
    """One row of the gap map's CSV, every column the join reads defaulting to empty.

    :param module: the group key.
    :param name: the C# name (``(constructor)`` for a class call).
    :param numpy_kind: NumPy's kind of the name (``absent`` for a NumSharp-only name).
    :param cells: column overrides (``mark_M="●"``, ``missing_any_overload="out; where"``, ...).
    :returns: the row as csv.DictReader gives it (every value a string).
    """
    row = {column: "" for column in gap_module.REQUIRED_COLUMNS}
    row.update({"module": module, "name": name, "numpy_kind": numpy_kind, "status": "identical", "overloads": "1",
                "best": f"NDArray {name}(NDArray a)", "numpy_signature": f"{name}(a)"})
    row.update(cells)
    return row


def catalog_row(api_id: str, target: str | None, **fields) -> dict:
    """A catalog row as resolve_rows builds it, credited to ``target``.

    :param api_id: the row id (``numpy.sum``, ``numsharp.random.x`` for a NumSharp-only row).
    :param target: the credited C# member, or None for an API the catalog calls missing.
    :param fields: overrides (``origin``, ``kind``, ``availability``, ``python_protocol``, ...).
    :returns: the row.
    """
    row = {"id": api_id, "origin": "numpy", "kind": "function", "availability": "exact" if target else "missing",
           "numsharp_target": target}
    row.update(fields)
    return row


def inventory_for(*members) -> dict:
    """A schema-7 inventory holding the given members, each with one ``(NDArray a)`` overload.

    :param members: ``(target, declaring types)`` pairs, or plain targets (declared by their own type).
    :returns: the inventory (``objectTypes`` keyed by type).
    """
    types: dict = {}
    for member in members:
        target, declaring = member if isinstance(member, tuple) else (member, None)
        owner, _, name = target.rpartition(".")
        entry = types.setdefault(owner, {"type": owner, "methods": []})
        entry["methods"].append({"name": name, "kind": "method", "declaringTypes": declaring or [owner],
                                 "overloads": [overload(name, "NDArray a")]})
    return {"objectTypes": types, "modules": {}}


def gap_map(*rows, marks=None, with_exemptions=True) -> "gap_module.GapMap":
    """A gap map over the given rows, plus (by default) every row ``GAP_JOIN_EXEMPT`` exempts, as the real gap map
    has them: without them each synthetic map would report the exemptions as stale.

    :param rows: gap-map rows (``gap_row``).
    :param marks: the gap map's own counts (defaults to empty).
    :param with_exemptions: False to leave the exempted rows out (the stale-exemption test).
    :returns: the GapMap.
    """
    rows = list(rows)
    if with_exemptions:
        present = {(row["module"], row["name"]) for row in rows}
        rows.extend(gap_row(module, name, numpy_kind="method") for module, name in sp.GAP_JOIN_EXEMPT
                    if (module, name) not in present)
    return gap_module.GapMap(rows, {}, marks or {}, dict(GROUP_TYPES))


class GapMapJoinTests(unittest.TestCase):
    """How the gap map's C# spellings become the catalog rows' signature records."""

    def test_an_id_join_takes_the_gap_maps_marks_and_probe_findings(self):
        """np.sum's spelling carries M, O, L and Q; the probe cells split into findings, "; " inside one kept."""
        rows = [catalog_row("numpy.sum", "NumSharp.np.sum")]
        gap = gap_map(gap_row("np", "sum", mark_M="●", mark_O="7", mark_L="●", mark_Q="●", overloads="7",
                              missing_any_overload="out; initial; where",
                              probe_null="`axis`: argument 2: `<null>` -> `int`",
                              probe_sequence="`axis` as a sequence: no overload fits; the closest one wants `int` "
                                             "for argument 2"))
        self.assertEqual([], sp.attach(rows, inventory_for("NumSharp.np.sum"), np, gap))
        record = rows[0]["signature"]
        self.assertEqual(("differs", "gap-map", "MOLQ"), (record["status"], record["source"], sp.marks(record)))
        self.assertEqual(["out", "initial", "where"], record["missing"])
        self.assertEqual(["`axis` as a sequence: no overload fits; the closest one wants `int` for argument 2"],
                         record["sequence"])
        self.assertEqual("np.sum", record["spellings"][0]["csharp"])

    def test_constructors_and_the_typed_receiver_join_the_class_and_member_rows(self):
        """``new NDArray(...)`` and ``new NDArray<T>(...)`` spell numpy.ndarray; NDArray<T>.reshape spells reshape."""
        rows = [catalog_row("numpy.ndarray", "NumSharp.NDArray", kind="class"),
                catalog_row("numpy.ndarray.reshape", "NumSharp.NDArray.reshape", kind="method")]
        gap = gap_map(gap_row("np", "ndarray", numpy_kind="class", mark_O="2", overloads="2", mark_L="◦"),
                      gap_row("ndarray", "(constructor)", numpy_kind="class", mark_M="●", mark_B="●", mark_O="11",
                              overloads="11", mark_L="●"),
                      gap_row(gap_module.TYPED_NDARRAY_GROUP, "(constructor)", numpy_kind="class", mark_T="◇",
                              typecode_generic_selectors="<TDType>(Shape shape)", mark_L="◐"),
                      gap_row("ndarray", "reshape", numpy_kind="method"),
                      gap_row(gap_module.TYPED_NDARRAY_GROUP, "reshape", numpy_kind="method", mark_M="●",
                              missing_any_overload="order; copy"))
        problems = sp.attach(rows, inventory_for("NumSharp.np.ndarray", "NumSharp.NDArray.reshape"), np, gap)
        self.assertEqual([], problems)
        ndarray, reshape = rows[0]["signature"], rows[1]["signature"]
        self.assertEqual(["np.ndarray", "new NDArray(...)", "new NDArray<T>(...)"],
                         [s["csharp"] for s in ndarray["spellings"]])
        self.assertEqual("MOTBL", sp.marks(ndarray), "the row's marks are the union over its spellings")
        self.assertEqual(("●", "11", "◇"), (ndarray["marks"]["L"], ndarray["marks"]["O"], ndarray["marks"]["T"]),
                         "each mark keeps its strongest symbol (● over ◐ over ◦) and the largest overload count")
        self.assertEqual(2, ndarray["overloads"], "the top-level count is the primary spelling's (np.ndarray)")
        counts = sp.counts(rows)
        self.assertEqual((1, 1, 1, 0), (counts["signature_binding"], counts["signature_null"],
                                        counts["signature_generic"], counts["signature_sequence"]))
        self.assertEqual(["a.reshape", "NDArray<T>.reshape"], [s["csharp"] for s in reshape["spellings"]])
        self.assertEqual(["order", "copy"], reshape["missing"])

    def test_an_alias_and_an_inherited_member_adopt_the_spelling_of_the_same_member_and_object(self):
        """numpy.acos IS numpy.arccos and NDMatrix.fill IS NDArray.fill: the verdict can only be the same."""
        rows = [catalog_row("numpy.acos", "NumSharp.np.arccos", availability="alias", kind="ufunc"),
                catalog_row("numpy.arccos", "NumSharp.np.arccos", kind="ufunc"),
                catalog_row("numpy.matrix.fill", "NumSharp.NDMatrix.fill", kind="method", object_surface=True),
                catalog_row("numpy.ndarray.fill", "NumSharp.NDArray.fill", kind="method"),
                catalog_row("numpy.matrix.tolist", "NumSharp.NDMatrix.tolist", kind="method", object_surface=True),
                catalog_row("numpy.ndarray.tolist", "NumSharp.NDArray.tolist", kind="method")]
        gap = gap_map(gap_row("np", "arccos", numpy_kind="ufunc", mark_M="◦", missing_excluded="casting"),
                      gap_row("ndarray", "fill", numpy_kind="method"),
                      gap_row("ndarray", "tolist", numpy_kind="method"))
        inventory = inventory_for("NumSharp.np.arccos", ("NumSharp.NDMatrix.fill", ["NumSharp.NDArray"]),
                                  "NumSharp.NDArray.fill", ("NumSharp.NDMatrix.tolist", ["NumSharp.NDArray"]),
                                  "NumSharp.NDArray.tolist")
        self.assertEqual([], sp.attach(rows, inventory, np, gap))
        by_id = {row["id"]: row["signature"] for row in rows}
        acos = by_id["numpy.acos"]
        self.assertEqual(("differs", "N", "numpy.arccos"),
                         (acos["status"], sp.marks(acos), acos["spellings"][0]["via"]),
                         "M ◦ is no mark; the alias keeps its N")
        self.assertEqual("identical", by_id["numpy.arccos"]["status"], "M ◦ alone does not make a row differ")
        self.assertEqual(("gap-map", "numpy.ndarray.fill"),
                         (by_id["numpy.matrix.fill"]["source"], by_id["numpy.matrix.fill"]["spellings"][0]["via"]))
        # NumPy's matrix overrides tolist (another object): the ndarray verdict does not apply.
        self.assertEqual("inventory", by_id["numpy.matrix.tolist"]["source"])
        self.assertEqual("—", by_id["numpy.matrix.tolist"]["marks"]["B"], "an unprobed row says so")

    def test_a_numsharp_only_row_carries_the_spelling_of_its_member(self):
        """numpy.random.get_bit_generator is outside NumPy's __all__ and docs: the NumSharp-only row carries it."""
        rows = [catalog_row("numsharp.random.get_bit_generator", "NumSharp.NumPyRandom.get_bit_generator",
                            origin="numsharp", availability="extension")]
        gap = gap_map(gap_row("np.random", "get_bit_generator"),
                      gap_row("np.random", "bernoulli", numpy_kind="absent", mark_O="2", mark_M="—", mark_X="—"))
        rows.append(catalog_row("numsharp.random.bernoulli", "NumSharp.NumPyRandom.bernoulli", origin="numsharp",
                                availability="extension"))
        self.assertEqual([], sp.attach(rows, inventory_for(), np, gap))
        self.assertEqual("identical", rows[0]["signature"]["status"])
        bernoulli = rows[1]["signature"]
        self.assertEqual(("not-compared", "O"), (bernoulli["status"], sp.marks(bernoulli)))
        reconciliation = sp.reconcile(gap, rows)
        self.assertEqual([("np.random", "get_bit_generator")],
                         [(item["group"], item["name"]) for item in reconciliation["elsewhere"]])
        self.assertEqual(1, reconciliation["carried"]["numpy"], "a NumSharp-only spelling is not a NumPy name")

    def test_an_unjoined_spelling_fails_and_an_exemption_goes_stale(self):
        """A lost finding is a problem; so is a gap-map spelling of an API the catalog calls missing."""
        rows = [catalog_row("numpy.char.chararray.center", None, kind="method")]
        gap = gap_map(gap_row("np", "nowhere"), gap_row("np.char.chararray", "center", numpy_kind="method"),
                      gap_row("np.random.MT19937", "_legacy_seeding", numpy_kind="method"),
                      gap_row("np.matrix", "Clone", numpy_kind="absent", mark_M="—"))
        problems = sp.attach(rows, inventory_for(), np, gap)
        self.assertTrue(any("np / nowhere: joins no catalog row" in p for p in problems), problems)
        self.assertTrue(any("does not credit" in p for p in problems), problems)
        self.assertFalse(any("_legacy_seeding" in p for p in problems), "an exempted spelling is no problem")
        self.assertFalse(any("Clone" in p for p in problems),
                         "a NumSharp-only member of an object type has no catalog row by design (no problem)")
        self.assertEqual(2, len(problems), problems)
        stale = sp.attach([], inventory_for(), np, gap_map(with_exemptions=False))
        self.assertTrue(any("stale gap-map exemption" in p for p in stale), stale)

    def test_the_status_follows_the_marks(self):
        """M ◦ and a null default are not differences; L ◦, B ◐ and a hard closest difference are."""
        def status(**cells):
            """The record status of one numpy.f row whose only spelling has the given cells.

            :param cells: gap-map row cells.
            :returns: the status.
            """
            rows = [catalog_row("numpy.f", "NumSharp.np.f")]
            sp.attach(rows, inventory_for("NumSharp.np.f"), np, gap_map(gap_row("np", "f", **cells)))
            return rows[0]["signature"]["status"]

        self.assertEqual("identical", status(mark_M="◦", missing_excluded="like"))
        self.assertEqual("identical", status(null_default="order: 'C' vs null"))
        self.assertEqual("identical", status(mark_B="—", mark_L="—", mark_Q="—"), "an unprobed spelling has no mark")
        self.assertEqual("differs", status(mark_L="◦"))
        self.assertEqual("differs", status(mark_B="◐"))
        self.assertEqual("differs", status(mark_T="◇"))
        self.assertEqual("differs", status(slot="dtype: NumPy #2, C# #3"))

    def test_a_row_with_both_t_symbols_counts_once_under_t(self):
        """T ● and T ◇ on one row: each symbol's count takes it, and the any-T count (the per-surface T column of
        summary.md) takes it once, where adding the two counts would take it twice."""
        rows = [catalog_row("numpy.zeros", "NumSharp.np.zeros")]
        gap = gap_map(gap_row("np", "zeros", mark_T="●◇", typecode_params="dtype: NPTypeCode",
                              typecode_generic_selectors="<T>(Shape shape)"))
        self.assertEqual([], sp.attach(rows, inventory_for("NumSharp.np.zeros"), np, gap))
        self.assertEqual("T", sp.marks(rows[0]["signature"]))
        counts = sp.counts(rows)
        self.assertEqual((1, 1, 1), (counts["signature_typecode"], counts["signature_generic"],
                                     counts["signature_typecode_any"]))

    def test_reconcile_counts_the_carried_spellings_under_the_gap_maps_keys(self):
        """The artifact's carried counts plus what it does not show equal the gap map's own counts."""
        rows = [catalog_row("numpy.sum", "NumSharp.np.sum"), catalog_row("numpy.acos", "NumSharp.np.arccos",
                                                                         availability="alias", kind="ufunc"),
                catalog_row("numpy.arccos", "NumSharp.np.arccos", kind="ufunc")]
        gap = gap_map(gap_row("np", "sum", mark_M="●", mark_O="7", mark_L="◦"),
                      gap_row("np", "arccos", numpy_kind="ufunc", mark_M="◦"),
                      gap_row("np.random.MT19937", "_legacy_seeding", numpy_kind="method", mark_O="5"),
                      marks={"numpy": 3, "M": 1, "M_excluded_only": 1, "O": 2, "L_default": 1, "no_mark": 0})
        sp.attach(rows, inventory_for("NumSharp.np.sum", "NumSharp.np.arccos"), np, gap)
        reconciliation = sp.reconcile(gap, rows)
        carried = reconciliation["carried"]
        self.assertEqual((2, 1, 1, 1, 1, 0), (carried["numpy"], carried["M"], carried["M_excluded_only"],
                                               carried["O"], carried["L_default"], carried["no_mark"]),
                         "numpy.acos's adopted copy of np.arccos is not counted twice")
        self.assertEqual([("np.random.MT19937", "_legacy_seeding")],
                         [(item["group"], item["name"]) for item in reconciliation["outside"]])
        self.assertEqual(3, reconciliation["tool"]["numpy"])

    def test_cell_splitting_and_tool_output_parsing(self):
        """List cells split on "; "; probe findings only where a finding starts; the tools' summary lines parse."""
        self.assertEqual(["out", "where"], gap_module.split_list("out; where"))
        self.assertEqual([], gap_module.split_list(""))
        self.assertEqual(["`device` alone: needs `dtype`", "positionally up to `order`: no overload fits; the closest "
                          "one wants `string` for argument 3"],
                         gap_module.split_findings("`device` alone: needs `dtype`; positionally up to `order`: no "
                                                   "overload fits; the closest one wants `string` for argument 3"))
        self.assertEqual([{"a": 1}, {"b": 2}], gap_module.json_lines('noise\n{"a": 1}\n{"b": 2}\n'))
        self.assertEqual({"probes": 7259, "skipped": 0, "fail_12": 488, "fail_14": 455},
                         gap_module.probe_counts("7259 probes (0 names skipped): 488 fail at C# 12, 455 at C# 14"))
        self.assertEqual("numpy.random.Generator", gap_module.catalog_id("np.random.Generator", "(constructor)"))
        self.assertEqual("numpy.ndarray.reshape", gap_module.catalog_id(gap_module.TYPED_NDARRAY_GROUP, "reshape"))
        types = dict(GROUP_TYPES)
        self.assertEqual("np.@char.find", gap_module.csharp_spelling("np.char", "find", types))
        self.assertEqual("NDMatrix.sum", gap_module.csharp_spelling("np.matrix", "sum", types))
        self.assertEqual("new NDArray<T>(...)",
                         gap_module.csharp_spelling(gap_module.TYPED_NDARRAY_GROUP, "(constructor)", types))


class CommittedGapMapTests(unittest.TestCase):
    """The committed gap map (docs/plans/numpy-signature-diff.csv) keeps the interface the join reads."""

    @classmethod
    def setUpClass(cls):
        """Read the committed CSV once."""
        cls.rows = gap_module.read_csv(ROOT / "docs" / "plans" / "numpy-signature-diff.csv")

    def test_every_row_maps_to_a_numpy_id_and_every_finding_splits_where_it_starts(self):
        """A group the join cannot map, or a probe text the splitter misreads, fails here before the generator."""
        self.assertGreater(len(self.rows), 1000)
        for row in self.rows:
            cid = gap_module.catalog_id(row["module"], row["name"])
            self.assertTrue(cid.startswith("numpy."), cid)
            for column in ("probe_binding", "probe_null", "probe_sequence"):
                for finding in gap_module.split_findings(row[column]):
                    self.assertTrue(finding.startswith(gap_module._FINDING_STARTS), (row["module"], row["name"],
                                                                                     finding))
            for mark in "MXBLQ":
                self.assertIn(row[f"mark_{mark}"], ("", "●", "◐", "◦", "—", "?"), (row["name"], mark))


if __name__ == "__main__":
    unittest.main()
