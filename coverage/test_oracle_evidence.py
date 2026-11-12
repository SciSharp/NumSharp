"""Regression gate for the NumPy-oracle evidence join (oracle_evidence.py + oracle_map.json).

Two layers:

* ``SyntheticJoinTests`` builds throwaway corpus trees and pins every resolution rule and every
  problem class, so a rule change fails with a precise message instead of silently moving evidence.
* ``CommittedCorpusTests`` joins the REAL committed corpora against the real NumPy 2.4.2 catalog
  (built from NumPy alone, no dotnet inventory): the committed map must resolve every contract with
  zero problems, a few load-bearing rows are pinned, and the reviewed aliases are cross-checked against
  the C# oracle gate's own alias classifications so the two cannot drift.

Run:
    python -m unittest discover -s coverage -p test_oracle_evidence.py
"""

from __future__ import annotations

import copy
import csv
import inspect
import io
import json
import re
import tempfile
import unittest
import zipfile
from pathlib import Path

import generate_coverage as coverage
import oracle_evidence as oe
from numpy_documentation import documentation_url
from object_surfaces import (OPERATOR_PROTOCOLS, _OPERATOR_SYMBOLS, class_member_rows, object_surface_rows,
                             scalar_type_rows)


ROOT = Path(__file__).resolve().parents[1]


def minimal_map(**fuzz_keys) -> dict:
    """A map with the committed namespace rules but only the given explicit fuzz keys.

    :param fuzz_keys: explicit ``fuzz.keys`` entries for the synthetic corpus.
    :returns: a map whose specialized sections resolve the minimal specialized fixtures below.
    """
    committed = oe.load_map()
    return {
        "schema_version": 1,
        "fuzz": {
            "bare_key_precedence": copy.deepcopy(committed["fuzz"]["bare_key_precedence"]),
            "prefix_rules": copy.deepcopy(committed["fuzz"]["prefix_rules"]),
            "files_without_op": {},
            "keys": dict(fuzz_keys),
        },
        "aliases": {},
        "host_pins": {},
        "npy": {"npy": ["numpy.save", "numpy.load"]},
        "flags": {"read": ["numpy.ndarray.flags"]},
        "layout": {"sort": ["numpy.sort"]},
        "strings": {"str_fn:{name}": ["numpy.strings.{name}"], "strings_multiply": ["numpy.strings.multiply"]},
    }


def case(op: str, **fields) -> dict:
    """One synthetic fuzz contract.

    :param op: the op key.
    :param fields: extra case fields (params, error, layout, operands, ...).
    :returns: the case dict, with a default layout and a float64 operand.
    """
    base = {"op": op, "params": {}, "operands": [{"dtype": "float64"}], "layout": "c_contiguous_1d",
            "expected": {"dtype": "float64"}}
    base.update(fields)
    return base


def write_tree(root: Path, corpus: dict[str, list[dict]], replay: str | None = None,
               host_libm: tuple[str, ...] = (), strings: dict[str, list[dict]] | None = None,
               strings_replay: str | None = None) -> None:
    """Materialize a synthetic repository tree the scanner can read.

    :param root: temporary repository root.
    :param corpus: corpus file name -> contracts.
    :param replay: FuzzCorpusTests.cs body; by default every corpus file is replayed via RunCorpus.
    :param host_libm: files the default replay runs through RunHostLibmCorpus instead.
    :param strings: string-oracle section name -> contracts; by default one ``strfn`` section with two
        ``np.strings.find`` calls (one a NumPy error), a ``strings_multiply_text`` call and a case of another op.
    :param strings_replay: StringOracleTests.cs body; by default every section is replayed via ``Run("<section>")``.
    """
    fuzz = root / "test" / "NumSharp.Tests.Oracle" / "Fuzz"
    (fuzz / "corpus").mkdir(parents=True)
    for name, cases in corpus.items():
        (fuzz / "corpus" / name).write_text("".join(json.dumps(item) + "\n" for item in cases), encoding="utf-8")
    if replay is None:
        replay = "\n".join(
            f'public void T{i}() => {"RunHostLibmCorpus" if name in host_libm else "RunCorpus"}("{name}");'
            for i, name in enumerate(corpus))
    (fuzz / "FuzzCorpusTests.cs").write_text(replay, encoding="utf-8")
    io_dir = root / "test" / "NumSharp.Tests" / "IO" / "corpus"
    io_dir.mkdir(parents=True)
    with zipfile.ZipFile(io_dir / "npy_oracle.zip", "w") as archive:
        archive.writestr("manifest.json", json.dumps({"cases": [
            {"kind": "npy", "np_dtype": "int32", "load_error": None, "load_via": "load"},
            {"kind": "npy", "np_dtype": "bool", "load_error": "bad", "load_via": "load"},
        ]}))
    backends = root / "test" / "NumSharp.Tests" / "Backends" / "corpus"
    backends.mkdir(parents=True)
    (backends / "flags_oracle.jsonl").write_text(
        json.dumps({"recipe": "c1d", "dtype": "int64", "ops": [], "err": None}) + "\n", encoding="utf-8")
    (backends / "layout_parity_oracle.jsonl").write_text(
        json.dumps({"fam": "sort", "key": "c2d_ax-1", "dtype": "int64"}) + "\n", encoding="utf-8")
    if strings is None:
        strings = {"strfn": [
            {"op": "str_fn", "params": {"name": "find"}, "operands": [{"dtype": "<U5"}], "layout": "<U5:c1",
             "expected": {"dtype": "<i8"}},
            {"op": "str_fn", "params": {"name": "find"}, "operands": [{"dtype": "|S3"}], "layout": "|S3:c1",
             "error": {"type": "TypeError", "text": "..."}},
            {"op": "strings_multiply_text", "params": {}, "operands": [{"dtype": "<U2"}], "expected": {"dtype": "<U4"}},
            {"op": "ufunc", "params": {"ufunc": "equal"}, "operands": [{"dtype": "<U2"}], "expected": {"dtype": "bool"}},
        ]}
    text = root / "test" / "NumSharp.Tests.Oracle" / "Strings"
    (text / "corpus").mkdir(parents=True)
    for section, cases in strings.items():
        (text / "corpus" / f"strings_{section}.jsonl").write_text(
            "".join(json.dumps(item) + "\n" for item in cases), encoding="utf-8")
    if strings_replay is None:
        strings_replay = "\n".join(f'public void S{i}() => Run("{section}");' for i, section in enumerate(strings))
    (text / "StringOracleTests.cs").write_text(strings_replay, encoding="utf-8")


def catalog(*ids: str, kinds: dict[str, str] | None = None, targets: dict[str, str] | None = None,
            statuses: dict[str, str] | None = None) -> list[dict]:
    """Synthetic catalog rows.

    :param ids: row ids (kind defaults to function, status to available).
    :param kinds: per-id kind overrides.
    :param targets: per-id numsharp_target values (the identity rule reads these).
    :param statuses: per-id status overrides.
    :returns: rows with the fields the join, classify and status_counts read.
    """
    kinds, targets, statuses = kinds or {}, targets or {}, statuses or {}
    base = ("numpy.save", "numpy.load", "numpy.ndarray.flags", "numpy.sort", "numpy.strings.find",
            "numpy.strings.multiply")
    return [{"id": row_id, "kind": kinds.get(row_id, "function"), "numsharp_target": targets.get(row_id),
             "origin": "numsharp" if row_id.startswith("numsharp.") else "numpy",
             "status": statuses.get(row_id, "available")}
            for row_id in dict.fromkeys(ids + base)]


class SyntheticJoinTests(unittest.TestCase):
    def join(self, corpus, rows, oracle_map, **tree):
        """Scan a synthetic tree and join it.

        :param corpus: corpus file name -> contracts.
        :param rows: catalog rows.
        :param oracle_map: the map to resolve with.
        :param tree: extra :func:`write_tree` arguments.
        :returns: (sources, join result).
        """
        with tempfile.TemporaryDirectory() as tmp:
            write_tree(Path(tmp), corpus, **tree)
            sources = oe.load_sources(oracle_map, Path(tmp))
        return sources, oe.join(rows, sources, oracle_map)

    def test_bare_keys_prefer_top_level_and_never_land_on_modules(self):
        rows = catalog("numpy.trace", "numpy.linalg.trace", "numpy.fft", "numpy.fft.fft", kinds={"numpy.fft": "module"})
        _, result = self.join({"a.jsonl": [case("trace"), case("fft")]}, rows, minimal_map())
        self.assertEqual([], result.problems)
        self.assertEqual(1, result.evidence["numpy.trace"].contracts)
        self.assertNotIn("numpy.linalg.trace", result.evidence, "np.linalg namesakes are never credited by name")
        self.assertIn("numpy.fft.fft", result.evidence)
        self.assertNotIn("numpy.fft", result.evidence, "a module row is never an automatic target")

    def test_prefix_rule_falls_back_to_the_numsharp_only_row(self):
        rows = catalog("numpy.polynomial.chebyshev.chebval", "numsharp.polynomial.chebyshev.chebvalnd",
                       kinds={"numsharp.polynomial.chebyshev.chebvalnd": "method"})
        _, result = self.join({"p.jsonl": [case("chebyshev.chebval"), case("chebyshev.chebvalnd")]},
                              rows, minimal_map())
        self.assertEqual([], result.problems)
        self.assertEqual({"numpy.polynomial.chebyshev.chebval", "numsharp.polynomial.chebyshev.chebvalnd",
                          "numpy.save", "numpy.load", "numpy.ndarray.flags", "numpy.sort", "numpy.strings.find",
                          "numpy.strings.multiply"}, set(result.evidence))

    def test_param_driven_keys_group_per_value_and_take_value_overrides(self):
        oracle_map = minimal_map(
            out_binary={"param": "ufunc", "ids": ["numpy.{value}"]},
            grnd={"param": "method", "ids": ["numpy.random.Generator.{value}"],
                  "values": {"rs_bytes": ["numpy.random.bytes"]}})
        rows = catalog("numpy.add", "numpy.subtract", "numpy.random.Generator.normal", "numpy.random.bytes")
        corpus = {"o.jsonl": [case("out_binary", params={"ufunc": "add"}), case("out_binary", params={"ufunc": "add"}),
                              case("out_binary", params={"ufunc": "subtract"}),
                              case("grnd", params={"method": "normal"}), case("grnd", params={"method": "rs_bytes"})]}
        _, result = self.join(corpus, rows, oracle_map)
        self.assertEqual([], result.problems)
        self.assertEqual(2, result.evidence["numpy.add"].contracts)
        self.assertEqual({"out_binary:add"}, result.evidence["numpy.add"].keys)
        self.assertEqual(1, result.evidence["numpy.subtract"].contracts)
        self.assertEqual({"grnd:rs_bytes"}, result.evidence["numpy.random.bytes"].keys)
        self.assertIn("numpy.random.Generator.normal", result.evidence)

    def test_operator_keys_also_prove_the_operator_protocol_on_the_operator_route_only(self):
        """A key OpRegistry.cs replays through a C# operator proves ndarray's protocol for the cases that reach it.

        OpRegistry replays subtract as ops[0] - ops[1], except with a numpy.matrix operand (TryApplyFunctionForm calls
        np.subtract); add / multiply with params.dtype call np.add(a, b, dtype=). Those cases prove the ufunc only.
        """
        oracle_map = minimal_map(
            subtract={"ids": ["numpy.subtract"], "operator_ids": ["numpy.ndarray.__sub__"]},
            add={"ids": ["numpy.add"], "operator_ids": ["numpy.ndarray.__add__"], "function_params": ["dtype"]})
        rows = catalog("numpy.subtract", "numpy.ndarray.__sub__", "numpy.add", "numpy.ndarray.__add__",
                       kinds={"numpy.ndarray.__sub__": "method", "numpy.ndarray.__add__": "method"})
        matrix = [{"dtype": "float64", "matrix": True}, {"dtype": "float64"}]
        corpus = {"a.jsonl": [case("subtract"), case("subtract"), case("subtract", operands=matrix),
                              case("add"), case("add", params={"dtype": "float32"})]}
        _, result = self.join(corpus, rows, oracle_map)
        self.assertEqual([], result.problems, "an operator entry may repeat the automatic ids: it exists for operator_ids")
        self.assertEqual(3, result.evidence["numpy.subtract"].contracts, "every route proves the ufunc row")
        self.assertEqual({"subtract", "subtract:operator"}, result.evidence["numpy.subtract"].keys)
        sub = result.evidence["numpy.ndarray.__sub__"]
        self.assertEqual((2, {"subtract:operator"}), (sub.contracts, sub.keys), "a matrix operand calls np.subtract")
        self.assertEqual(2, result.evidence["numpy.add"].contracts)
        self.assertEqual(1, result.evidence["numpy.ndarray.__add__"].contracts, "params.dtype calls np.add(dtype=)")

    def test_operator_entries_must_reach_the_operator_and_cannot_be_param_driven(self):
        oracle_map = minimal_map(
            subtract={"ids": ["numpy.subtract"], "operator_ids": ["numpy.ndarray.__sub__"]},
            out_binary={"param": "ufunc", "ids": ["numpy.{value}"], "operator_ids": ["numpy.ndarray.__add__"]})
        rows = catalog("numpy.subtract", "numpy.ndarray.__sub__", "numpy.add", "numpy.ndarray.__add__")
        corpus = {"a.jsonl": [case("subtract", operands=[{"dtype": "float64", "matrix": True}]),
                              case("out_binary", params={"ufunc": "add"})]}
        _, result = self.join(corpus, rows, oracle_map)
        text = "\n".join(result.problems)
        self.assertIn("fuzz.keys['subtract'].operator_ids matches no contract replayed through the C# operator", text)
        self.assertIn("fuzz.keys['out_binary'] has both 'param' and 'operator_ids'", text)
        self.assertNotIn("numpy.ndarray.__sub__", result.evidence, "no case reached the operator")

    def test_numpy_char_ops_resolve_by_name_operand_side_and_explicit_entry(self):
        """The string oracle's numpy.char ops: a template per op that names its API, an explicit entry before the
        template (the flat_* members are chararray.flat), and the chararray protocol a C# operator's side selects."""
        oracle_map = minimal_map()
        oracle_map["strings"].update({
            "char_cmp:{name}": ["numpy.char.{name}"], "cha_call:{name}": ["numpy.char.chararray.{name}"],
            "cha_member:{name}": ["numpy.char.chararray.{name}"], "cha_member:flat_index": ["numpy.char.chararray.flat"],
            "cha_op:{name}": ["numpy.char.chararray.{name}"], "cobj_fn:{name}": ["numpy.char.{name}"]})
        chararray = {"op": 0, "cha": True}
        char_cases = [
            {"op": "char_cmp", "params": {"name": "less"}},
            {"op": "cha_call", "params": {"name": "center"}},
            {"op": "cha_member", "params": {"name": "flat_index"}},
            {"op": "cha_member", "params": {"name": "T"}},
            {"op": "cha_op", "params": {"sym": "+", "args": [chararray, {"v": 1}]}},
            # A chararray on the right runs Python's reflection: `x < c` is c.__gt__(x), `x + c` c.__radd__(x).
            {"op": "cha_op", "params": {"sym": "<", "args": [{"v": 1}, chararray]}},
            {"op": "cha_op", "params": {"sym": "+", "args": [{"v": 1}, chararray]}},
            {"op": "cobj_fn", "params": {"fn": "split"}},
            {"op": "cha_op", "params": {"sym": "@", "args": [chararray, chararray]}},
        ]
        # The minimal map's numpy.strings entries need a contract each, or they read as stale entries.
        strings = {"strfn": [{"op": "str_fn", "params": {"name": "find"}, "operands": [], "expected": {}},
                             {"op": "strings_multiply", "params": {}, "operands": [], "expected": {}}],
                   "strchar": [{**item, "operands": [], "expected": {}} for item in char_cases]}
        ids = ("numpy.char.less", "numpy.char.chararray.center", "numpy.char.chararray.flat", "numpy.char.chararray.T",
               "numpy.char.chararray.__add__", "numpy.char.chararray.__gt__", "numpy.char.chararray.__radd__",
               "numpy.char.split")
        _, result = self.join({"a.jsonl": [case("add")]}, catalog("numpy.add", *ids), oracle_map, strings=strings)
        self.assertEqual(["strings_strchar.jsonl:9 op 'cha_op' lacks the parameters that name the API it calls (a "
                          "function, a chararray member or an operator symbol)."], result.problems)
        for row_id, key in zip(ids, ("char_cmp:less", "cha_call:center", "cha_member:flat_index", "cha_member:T",
                                     "cha_op:__add__", "cha_op:__gt__", "cha_op:__radd__", "cobj_fn:split")):
            self.assertEqual({"strings:" + key}, result.evidence[row_id].keys, row_id)

    def test_unresolvable_keys_and_unknown_ids_are_problems_not_silent_drops(self):
        oracle_map = minimal_map(typo={"ids": ["numpy.nope"]})
        _, result = self.join({"a.jsonl": [case("mystery"), case("typo")]}, catalog("numpy.add"), oracle_map)
        text = "\n".join(result.problems)
        self.assertIn("Oracle op key 'mystery'", text)
        self.assertIn("maps to ids missing from the catalog: numpy.nope", text)

    def test_explicit_entries_must_be_used_and_must_not_repeat_the_automatic_rules(self):
        oracle_map = minimal_map(add={"ids": ["numpy.add"]}, gone={"ids": ["numpy.add"]},
                                 out_unary={"param": "ufunc", "ids": ["numpy.{value}"], "values": {"never": ["numpy.add"]}})
        corpus = {"a.jsonl": [case("add"), case("out_unary", params={"ufunc": "add"})]}
        _, result = self.join(corpus, catalog("numpy.add"), oracle_map)
        text = "\n".join(result.problems)
        self.assertIn("fuzz.keys['add'] repeats the automatic resolution", text)
        self.assertIn("fuzz.keys['gone'] matches no corpus contract", text)
        self.assertIn("fuzz.keys['out_unary'].values['never'] matches no corpus contract", text)
        # A specialized-section entry nothing uses is stale as well.
        oracle_map = minimal_map()
        oracle_map["layout"]["concat"] = ["numpy.sort"]
        _, result = self.join({"a.jsonl": [case("add")]}, catalog("numpy.add"), oracle_map)
        self.assertIn("oracle_map layout['concat'] matches no layout oracle contract", "\n".join(result.problems))

    def test_a_corpus_file_no_replay_suite_names_proves_nothing(self):
        corpus = {"kept.jsonl": [case("add")], "orphan.jsonl": [case("add"), case("add")]}
        sources, result = self.join(corpus, catalog("numpy.add"), minimal_map(),
                                    replay='public void T() => RunCorpus("kept.jsonl");')
        self.assertIn("Corpus file orphan.jsonl is not named by any replay suite", "\n".join(result.problems))
        self.assertEqual(1, result.evidence["numpy.add"].contracts, "the orphan's contracts must not count")
        self.assertEqual(["kept.jsonl"], [item["name"] for item in sources.files if item["source"] == "fuzz"])

    def test_host_pins_come_from_the_replay_source_and_the_map(self):
        oracle_map = minimal_map()
        oracle_map["host_pins"] = {"blas.jsonl": {"class": "pinned-blas", "reason": "pinned"},
                                   "libm.jsonl": {"class": "host-libm", "reason": "redundant"},
                                   "absent.jsonl": {"class": "weird", "reason": "bad"}}
        corpus = {"libm.jsonl": [case("sqrt")], "blas.jsonl": [case("dot")], "plain.jsonl": [case("sqrt", error={"type": "ValueError"})]}
        sources, result = self.join(corpus, catalog("numpy.sqrt", "numpy.dot"), oracle_map, host_libm=("libm.jsonl",))
        text = "\n".join(result.problems)
        self.assertIn("host_pins['libm.jsonl'] is redundant", text)
        self.assertIn("host_pins['absent.jsonl'] has unknown class 'weird'", text)
        self.assertIn("host_pins['absent.jsonl'] names a corpus file that does not exist", text)
        pins = {item["name"]: item["pin"] for item in sources.files if item["source"] == "fuzz"}
        self.assertEqual({"libm.jsonl": "host-libm", "blas.jsonl": "pinned-blas", "plain.jsonl": None}, pins)
        sqrt = result.evidence["numpy.sqrt"].record()
        self.assertFalse(sqrt["pinned"], "one portable contract keeps the row portable")
        self.assertEqual(1, sqrt["pinned_contracts"])
        self.assertEqual(1, sqrt["error_contracts"])
        self.assertTrue(result.evidence["numpy.dot"].record()["pinned"])

    def test_reviewed_aliases_apply_and_report_stale_dangling_and_unknown_rows(self):
        oracle_map = minimal_map()
        oracle_map["aliases"] = {
            "numpy.absolute": {"via": "numpy.abs", "reason": "same ufunc"},
            "numpy.abs": {"via": "numpy.sqrt", "reason": "stale: abs has its own contracts"},
            "numpy.degrees": {"via": "numpy.rad2deg", "reason": "dangling: no contracts"},
            "numpy.ghost": {"via": "numpy.abs", "reason": "unknown row"},
        }
        rows = catalog("numpy.abs", "numpy.absolute", "numpy.sqrt", "numpy.degrees", "numpy.rad2deg")
        _, result = self.join({"a.jsonl": [case("abs"), case("abs"), case("sqrt")]}, rows, oracle_map)
        text = "\n".join(result.problems)
        self.assertIn("aliases['numpy.abs'] is stale", text)
        self.assertIn("aliases['numpy.degrees'] points at numpy.rad2deg, which has no direct oracle contracts", text)
        self.assertIn("aliases['numpy.ghost'] names a row the catalog does not have", text)
        record = result.evidence["numpy.absolute"].record()
        self.assertEqual("alias", record["status"])
        self.assertEqual({"id": "numpy.abs", "contracts": 2, "rule": "reviewed", "reason": "same ufunc", "pinned": False},
                         record["via"])

    def test_identity_alias_needs_the_same_numsharp_member_and_the_same_numpy_object(self):
        oracle_map = minimal_map()
        oracle_map["aliases"] = {"numpy.pow": {"via": "numpy.power", "reason": "duplicate of identity"}}
        # row_stack is the real-world case for the second condition: NumSharp credits it to the very
        # np.vstack member, but NumPy's row_stack is a DIFFERENT object (a warning wrapper), so the
        # identity rule must not fire; only a reviewed alias may vouch for it.
        rows = catalog("numpy.power", "numpy.pow", "numpy.acos", "numpy.arccos", "numpy.vstack", "numpy.row_stack",
                       targets={"numpy.power": "NumSharp.np.power", "numpy.pow": "NumSharp.np.power",
                                "numpy.arccos": "NumSharp.np.arccos", "numpy.acos": "NumSharp.np.acos",
                                "numpy.vstack": "NumSharp.np.vstack", "numpy.row_stack": "NumSharp.np.vstack"})
        with tempfile.TemporaryDirectory() as tmp:
            write_tree(Path(tmp), {"a.jsonl": [case("power"), case("arccos"), case("vstack")]})
            sources = oe.load_sources(oracle_map, Path(tmp))
        same = {frozenset({"numpy.pow", "numpy.power"}), frozenset({"numpy.acos", "numpy.arccos"})}
        result = oe.join(rows, sources, oracle_map, identical=lambda a, b: frozenset({a, b}) in same)
        self.assertIn("aliases['numpy.pow'] is redundant: the identity rule already derives it", "\n".join(result.problems))
        self.assertEqual("identity", result.evidence["numpy.pow"].via["rule"])
        self.assertNotIn("numpy.acos", result.evidence, "a different NumSharp member never inherits evidence")
        self.assertNotIn("numpy.row_stack", result.evidence, "the same NumSharp member alone is not enough")

    def test_error_contracts_are_read_from_every_schema(self):
        corpus = {"a.jsonl": [case("add", error={"type": "TypeError", "text": "x"}), case("add", expects_throw=True),
                              {"op": "get", "np": {"ok": False, "err": "IndexError"}, "base": "V6"},
                              {"op": "get", "np": {"ok": True}, "base": "V6", "dtype": "int16"}]}
        oracle_map = minimal_map(get={"ids": ["numpy.ndarray"]})
        rows = catalog("numpy.add", "numpy.ndarray", kinds={"numpy.ndarray": "class"})
        _, result = self.join(corpus, rows, oracle_map)
        self.assertEqual([], result.problems)
        self.assertEqual(2, result.evidence["numpy.add"].errors)
        self.assertEqual(1, result.evidence["numpy.ndarray"].errors)
        self.assertEqual({"V6"}, result.evidence["numpy.ndarray"].layouts, "index contracts use base as their layout")
        self.assertEqual({"int16"}, result.evidence["numpy.ndarray"].dtypes)

    def test_specialized_sources_resolve_by_variant(self):
        _, result = self.join({"a.jsonl": [case("add")]}, catalog("numpy.add"), minimal_map())
        self.assertEqual({"npy:npy"}, result.evidence["numpy.save"].keys)
        self.assertEqual(2, result.evidence["numpy.save"].contracts)
        self.assertEqual(1, result.evidence["numpy.save"].errors)
        self.assertEqual({"flags:read"}, result.evidence["numpy.ndarray.flags"].keys)
        self.assertEqual({"layout:sort"}, result.evidence["numpy.sort"].keys)
        # The string oracle's numpy.strings calls: str_fn by params.name through the template, the text-count multiply
        # under strings_multiply; the case of another op (a comparison ufunc) is not a numpy.strings call and is skipped.
        find = result.evidence["numpy.strings.find"]
        self.assertEqual({"strings:str_fn:find"}, find.keys)
        self.assertEqual((2, 1), (find.contracts, find.errors))
        self.assertEqual({"<U5", "|S3", "<i8"}, find.dtypes, "operand and result dtypes")
        self.assertEqual({"strings:strings_multiply"}, result.evidence["numpy.strings.multiply"].keys)
        self.assertNotIn("numpy.equal", result.evidence, "a string-oracle case of another op credits nothing")

    def test_string_oracle_sections_must_be_replayed_and_name_their_function(self):
        strings = {"strfn": [{"op": "str_fn", "params": {"name": "find"}, "operands": [], "expected": {}}],
                   "orphan": [{"op": "str_fn", "params": {"name": "find"}, "operands": [], "expected": {}}],
                   "nameless": [{"op": "str_fn", "params": {}, "operands": [], "expected": {}}],
                   "other": [{"op": "ufunc", "params": {"ufunc": "equal"}, "operands": [], "expected": {}}]}
        sources, result = self.join({"a.jsonl": [case("add")]}, catalog("numpy.add"), minimal_map(), strings=strings,
                                    strings_replay='Run("strfn"); Run("nameless"); Run("other");')
        problems = "\n".join(result.problems)
        self.assertIn("String oracle section strings_orphan.jsonl holds numpy.strings / numpy.char contracts but no "
                      "replay test", problems)
        self.assertIn("strings_nameless.jsonl:1 op 'str_fn' lacks the parameters that name the API it calls", problems)
        self.assertEqual(1, result.evidence["numpy.strings.find"].contracts, "the orphan's contracts must not count")
        self.assertEqual(["strings_strfn.jsonl"], [item["name"] for item in sources.files if item["source"] == "strings"],
                         "a section without a numpy.strings call is not a strings file")
        self.assertIn("oracle_map strings['strings_multiply'] matches no strings oracle contract", problems)

    def test_classify_status_counts_csv_and_record_shape(self):
        rows = [
            {"id": "numpy.a", "origin": "numpy", "status": "available", "availability": "exact",
             "oracle": {"status": "direct"}},
            {"id": "numpy.b", "origin": "numpy", "status": "partial", "availability": "exact",
             "oracle": {"status": "alias"}},
            {"id": "numpy.c", "origin": "numpy", "status": "available", "availability": "exact"},
            {"id": "numpy.d", "origin": "numpy", "status": "missing", "availability": "missing",
             "oracle": {"status": "direct"}},
            {"id": "numpy.e", "origin": "numpy", "status": "missing", "availability": "missing"},
        ]
        self.assertEqual(["verified-direct", "verified-alias", "unverified", "uncredited", "none"],
                         [oe.classify(row) for row in rows])
        counts = coverage.status_counts(rows)
        self.assertEqual((2, 1, 1, 1, 1, 40.0), (counts["oracle_verified"], counts["oracle_direct"], counts["oracle_alias"],
                                                  counts["oracle_unverified"], counts["oracle_uncredited"],
                                                  counts["oracle_percent"]))
        evidence = oe.RowEvidence()
        evidence.add(oe.ContractGroup("fuzz", "b.jsonl", "out_binary", "add", None, 3, 1, {"int8", "bool"}, {"x", "y"}))
        evidence.add(oe.ContractGroup("layout", "a.jsonl", "sort", None, "host-libm", 2))
        record = evidence.record()
        self.assertEqual(["status", "contracts", "error_contracts", "sources", "keys", "files", "dtypes", "layouts",
                          "pinned", "pinned_contracts", "pins"], list(record))
        self.assertEqual(["layout:sort", "out_binary:add"], record["keys"])
        self.assertEqual(["a.jsonl", "b.jsonl"], record["files"])
        self.assertEqual(["bool", "int8"], record["dtypes"])
        self.assertEqual({"fuzz": 3, "layout": 2}, record["sources"])
        row = {"id": "numpy.add", "origin": "numpy", "status": "available", "numsharp_signatures": [],
               "numsharp_source_paths": [], "numsharp_source_urls": [], "oracle": record}
        flat = next(csv.DictReader(io.StringIO(coverage.csv_text([row]))))
        self.assertEqual(("verified-direct", "5", "fuzz:3 | layout:2", "layout:sort | out_binary:add", "2"),
                         (flat["oracle_parity"], flat["oracle_contracts"], flat["oracle_sources"], flat["oracle_keys"],
                          flat["oracle_pinned_contracts"]))


class CommittedCorpusTests(unittest.TestCase):
    """The committed map must resolve every committed contract against the real NumPy catalog."""

    @classmethod
    def setUpClass(cls):
        """Join the committed corpus against the generator's NumPy row universe (no dotnet inventory needed)."""
        cls.np = coverage.load_numpy()
        cls.map = oe.load_map()
        exports = coverage.public_exports(cls.np)
        seen = {row["id"] for row in exports}
        rows = exports + coverage.extended_surface_rows(cls.np, {"modules": {}}, seen, {})
        rows += object_surface_rows(cls.np, {"objectTypes": {}}, seen, root=ROOT,
                                    source_base_url=coverage.NUMSHARP_SOURCE_BASE_URL,
                                    signature=coverage.numpy_signature, documentation_url=documentation_url)
        # The scalar types' and the other classes' member rows (generator 1.14.0), as the generator emits them.
        rows += scalar_type_rows(cls.np, seen, signature=coverage.numpy_signature, documentation_url=documentation_url)
        rows += class_member_rows(cls.np, list(rows), seen, {}, root=ROOT,
                                  source_base_url=coverage.NUMSHARP_SOURCE_BASE_URL,
                                  signature=coverage.numpy_signature, documentation_url=documentation_url,
                                  category_of=coverage.member_category, excluded=coverage.compat_excluded)
        # NumSharp-only rows come from the dotnet inventory in the real generator. Synthesize the ones
        # the map names plus the ones the automatic rules must find (np.evaluate, the N-D polynomial
        # evaluation twins) so the join sees the same id universe without building the tool.
        numsharp = {value for value in _strings(cls.map) if value.startswith("numsharp.")}
        numsharp.add("numsharp.np.evaluate")
        # The module constants {p}domain/zero/one/x are facade PROPERTIES: NumPy documents them, but the extended
        # catalog lists only callables, so the real generator emits each as a NumSharp-only property row, which
        # the polynomial prefix rules then resolve.
        properties = set()
        for basis, prefix in (("polynomial", "poly"), ("chebyshev", "cheb"), ("legendre", "leg"),
                              ("laguerre", "lag"), ("hermite", "herm"), ("hermite_e", "herme")):
            numsharp.add(f"numsharp.polynomial.{basis}.{prefix}valnd")
            properties.update(f"numsharp.polynomial.{basis}.{prefix}{name}" for name in ("domain", "zero", "one", "x"))
        rows += [{"id": row_id, "kind": "method", "origin": "numsharp", "numsharp_target": None} for row_id in sorted(numsharp)]
        rows += [{"id": row_id, "kind": "property", "origin": "numsharp", "numsharp_target": None} for row_id in sorted(properties)]
        cls.rows = rows
        cls.sources = oe.load_sources(cls.map)
        cls.result = oe.join(rows, cls.sources, cls.map)
        cls.labels = {item["label"]: item["ids"] for item in cls.result.resolutions}

    def evidence(self, row_id):
        self.assertIn(row_id, self.result.evidence, row_id)
        return self.result.evidence[row_id]

    def test_every_committed_contract_resolves_with_zero_problems(self):
        self.assertEqual([], self.result.problems)
        joined = sum(item["cases"] for item in self.result.resolutions)
        contracts = sum(group.cases for group in self.sources.groups)
        self.assertEqual(contracts, joined, "every scanned group must resolve")
        self.assertGreater(contracts, 250_000)
        self.assertEqual({"fuzz", "npy", "flags", "layout", "strings"}, {group.source for group in self.sources.groups})

    def test_load_bearing_rows_carry_the_expected_evidence(self):
        chebval = self.evidence("numpy.polynomial.chebyshev.chebval")
        self.assertGreater(chebval.contracts, 1000)
        self.assertGreaterEqual(len(chebval.dtypes), 14)
        self.assertEqual({"polyeval.jsonl"}, chebval.files)
        self.assertIn("numpy.trace", self.result.evidence)
        # The np.linalg Array-API forms have stricter contracts than their np.* namesakes (last-two-axes, 1-D only,
        # 3-vectors), so the bare corpus keys trace / diagonal / outer / cross never credit them; since 2026-10-08
        # surface.jsonl gates each form under its own linalg.<name> key, and only that key may. np.linalg.cross also
        # has the axis tier's contracts (its x1.shape[axis] IndexError, its 3-vector ValueError) under its own key.
        for strict in ("numpy.linalg.trace", "numpy.linalg.diagonal", "numpy.linalg.outer", "numpy.linalg.cross"):
            own = {"linalg." + strict[13:]} | ({"axis_arg:linalg.cross"} if strict == "numpy.linalg.cross" else set())
            self.assertEqual(own, self.evidence(strict).keys,
                             f"{strict} has a stricter contract than np.{strict[13:]}: only its own keys may credit it")
        cross = self.evidence("numpy.linalg.cross")
        self.assertIsNone(cross.via, "np.linalg.cross must not borrow np.cross's contracts")
        for row_id in ("numpy.random.normal", "numpy.random.RandomState.normal"):
            normal = self.evidence(row_id)
            # The random-API oracle adds every overload of RandomState.normal (random_api_host.jsonl).
            self.assertEqual({"rnd:normal", "random_api:RandomState.normal"}, normal.keys)
            # The legacy normal VALUE stream is host-libm (random_parity_host.jsonl). Its portable contracts can
            # only be error-message cells: 3a15871a added the array-parameter validation texts to the portable
            # tier. So the row is partly pinned, like np.fft.fft, and not "pinned" as a whole.
            self.assertGreater(normal.contracts - normal.portable, 0, "legacy normal's values are host-libm pinned")
            self.assertLessEqual(normal.portable, normal.errors, "only error contracts of normal are portable")
        # "ndarray" is the constructor np.ndarray(shape, dtype, buffer, offset, strides, order) (ndarray_ctor.jsonl).
        self.assertEqual({"ndarray"}, self.evidence("numpy.ndarray").keys)
        # Python protocols have rows since generator 1.14.0. The index oracle's get / set and the array-object
        # oracle's a[key] ("viewflags:getitem", view_flags.jsonl) attached to the class until then; the protocol
        # keys the matrix and surface tiers gate resolve through the matrix / ndarray prefix rules. The weak_scalar
        # tier's section S stores C# literals through ndarray.__setitem__'s routes (weak_scalar:setitem_*).
        setitems = {f"weak_scalar:setitem_{r}" for r in ("int", "neg", "slice", "empty", "ellipsis", "fancy", "mask")}
        self.assertEqual({"get", "viewflags:getitem"}, self.evidence("numpy.ndarray.__getitem__").keys)
        self.assertEqual({"set"} | setitems, self.evidence("numpy.ndarray.__setitem__").keys)
        self.assertEqual({"ndarray.__len__"}, self.evidence("numpy.ndarray.__len__").keys)
        for name in ("__add__", "__mul__", "__getitem__", "__imul__", "__repr__"):
            self.assertIn(f"matrix.{name}", self.evidence(f"numpy.matrix.{name}").keys)
            self.assertNotIn(f"matrix.{name}", self.evidence("numpy.matrix").keys)
        # "write_only" = Char arrays written as NumPy's '<U1' and compared byte for byte (they read back as strings);
        # "tpickle" = StringDType arrays, which NumPy pickles behind the descr '|O', written and read with allow_pickle.
        self.assertEqual({"npy:header", "npy:npy", "npy:write_only", "npy:tpickle"}, self.evidence("numpy.save").keys)
        self.assertEqual({"npy:npz:compressed"}, self.evidence("numpy.savez_compressed").keys)
        self.assertEqual({"flags:setflags"}, self.evidence("numpy.ndarray.setflags").keys)
        self.assertEqual({"evaluate"}, self.evidence("numsharp.np.evaluate").keys)
        # np.absolute borrowed np.abs's contracts through a reviewed alias until the array-object oracle
        # (view_flags.jsonl / out_flags.jsonl) gated the spelling itself; a reviewed alias still lends evidence.
        absolute = self.evidence("numpy.absolute")
        self.assertIsNone(absolute.via, "numpy.absolute has direct contracts")
        self.assertIn("viewflags:absolute", absolute.keys)
        self.assertIn("outflags:absolute", absolute.keys)
        self.assertEqual("numpy.vstack", self.evidence("numpy.row_stack").via["id"])
        # np.amax / np.amin were aliases of np.max / np.min until the minmax_kwargs tier (plan api-parameter-parity A4)
        # gated them directly through their own NumPy-shaped overloads.
        for row_id, fn in (("numpy.amax", "amax"), ("numpy.amin", "amin")):
            evidence = self.evidence(row_id)
            self.assertIsNone(evidence.via, f"{row_id} has direct contracts")
            self.assertIn(f"minmax_kwargs:{fn}", evidence.keys)
        self.assertEqual(["numpy.power"], self.labels["power"], "np.power shadows np.random.power")
        # The layout oracle's ufunc family (a fresh ufunc result's layout, 18314cac) names each case by the function it
        # calls: its contracts land on that function's row, one variant per case name. Without the per-name rule the
        # whole family had no map entry and the generator refused to publish (journey5 at 882a66a0).
        self.assertEqual("ufunc:around0", oe._layout_variant({"fam": "ufunc", "key": "around0_bcx"}))
        self.assertIn("layout:ufunc:neg", self.evidence("numpy.negative").keys)
        self.assertIn("layout:ufunc:mediank1", self.evidence("numpy.median").keys)
        self.assertTrue({"layout:ufunc:add0d", "layout:ufunc:adds111"} <= self.evidence("numpy.add").keys)
        # The string oracle verifies every numpy.strings function by name: the 44 str_fn names (search, strip, pad,
        # replace, partition, slice, the comparison names, predicates, case maps, translate, mod, codecs over S / U /
        # StringDType) and np.strings.add / multiply.
        rows = sorted(row["id"] for row in self.rows if row["id"].startswith("numpy.strings.") and row["id"].count(".") == 2)
        self.assertEqual(46, len(rows))
        for row_id in rows:
            self.assertIn("strings", self.evidence(row_id).sources, f"{row_id} has string-oracle contracts")
        find = self.evidence("numpy.strings.find")
        self.assertEqual({"strings:str_fn:find"}, find.keys)
        self.assertGreater(find.contracts, 1000)
        self.assertGreater(find.errors, 0)
        self.assertTrue({"strings_strfn.jsonl", "strings_tfn.jsonl"} <= find.files)
        # The 13 keys OpRegistry.cs replays through a C# operator also prove the operator's ndarray protocol, on the
        # cases that reach it: a numpy.matrix operand (matrix_operand.jsonl) and add / multiply with dtype= take the
        # np.* function, which keeps every contract.
        for protocol, key in (("__add__", "add"), ("__sub__", "subtract"), ("__truediv__", "divide"),
                              ("__eq__", "equal"), ("__ge__", "greater_equal"), ("__xor__", "bitwise_xor")):
            operator_row = self.evidence(f"numpy.ndarray.{protocol}")
            self.assertEqual({f"{key}:operator"}, operator_row.keys, protocol)
            self.assertGreater(operator_row.contracts, 300, protocol)
            self.assertNotIn("matrix_operand.jsonl", operator_row.files, f"{protocol}: matrix operands call np.{key}")
            self.assertTrue({key, f"{key}:operator"} <= self.evidence(f"numpy.{key}").keys, key)
        # numpy.char (plan item C6): the string oracle's numpy.char functions, chararray constructor, members,
        # protocols and operators by name, the numpy.char names that ARE numpy.strings objects through reviewed aliases.
        self.assertEqual({"strings:char_cmp:less"}, self.evidence("numpy.char.less").keys)
        self.assertEqual({"strings:char_compare"}, self.evidence("numpy.char.compare_chararrays").keys)
        self.assertEqual({"strings:cobj_fn:split"}, self.evidence("numpy.char.split").keys)
        self.assertEqual({"strings:cha_array"}, self.evidence("numpy.char.array").keys)
        center = self.evidence("numpy.char.center")
        self.assertEqual((0, "numpy.strings.center", "reviewed"), (center.contracts, center.via["id"], center.via["rule"]))
        self.assertIn("strings:cha_call:center", self.evidence("numpy.char.chararray.center").keys)
        self.assertIn("strings:cha_op:__radd__", self.evidence("numpy.char.chararray.__radd__").keys)
        self.assertEqual({"strings:cha_repr"}, self.evidence("numpy.char.chararray.__repr__").keys)
        self.assertTrue({f"strings:cha_member:flat_{name}" for name in ("index", "slice", "fancy", "copy")}
                        <= self.evidence("numpy.char.chararray.flat").keys)

    def test_operator_entries_mirror_opregistry(self):
        """Every OpRegistry.cs case that replays a key through a C# operator has an operator entry naming the operator's
        ndarray protocol, every operator entry names such a case, and the matrix-operand function form covers each key
        (replays_operator's other route), so the map cannot drift from the replay."""
        fuzz = ROOT / "test" / "NumSharp.Tests.Oracle" / "Fuzz"
        text = (fuzz / "OpRegistry.cs").read_text(encoding="utf-8")
        found = dict(re.findall(r'case "(\w+)":\s*return ops\[0\] (==|!=|<=|>=|<|>|\+|-|\*|/|&|\||\^) ops\[1\];', text))
        # add / multiply call np.add(..., dtype=) when params.dtype is present, else the operator.
        dtype_first = dict(re.findall(
            r'case "(\w+)":\s*return p\.TryGetValue\("dtype"[^;]*?:\s*ops\[0\] (\+|\*) ops\[1\];', text, re.S))
        self.assertEqual({"add": "+", "multiply": "*"}, dtype_first)
        found.update(dtype_first)
        entries = {key: entry for key, entry in oe._entries(self.map["fuzz"]["keys"]).items() if "operator_ids" in entry}
        self.assertEqual(set(found), set(entries))
        for key, symbol in found.items():
            protocols = [name for name, (op_name, position) in OPERATOR_PROTOCOLS.items()
                         if _OPERATOR_SYMBOLS[op_name] == symbol and position in ("left", "either")]
            self.assertEqual(["numpy.ndarray." + protocols[0]], entries[key]["operator_ids"], key)
            self.assertEqual(["dtype"] if key in dtype_first else [], entries[key].get("function_params", []), key)
        function_form = (fuzz / "OpRegistry.MatrixOperand.cs").read_text(encoding="utf-8")
        for key in found:
            self.assertRegex(function_form, rf'case "{key}": result = np\.{key}\(ops\[0\], ops\[1\]\); return true;', key)

    def test_reviewed_aliases_mirror_the_csharp_oracle_gate(self):
        text = (ROOT / "test" / "NumSharp.Tests.Oracle" / "Fuzz" / "OracleSurfaceCoverageTests.cs").read_text(encoding="utf-8")
        ids = {row["id"] for row in self.rows}
        aliases = oe._entries(self.map["aliases"])
        checked = 0
        for block, numpy_prefix, numsharp_prefix in (("EquivalentAliases", "numpy.", "numsharp.np."),
                                                     ("NdarrayAliases", "numpy.ndarray.", "numsharp.ndarray."),
                                                     ("MaAliases", "numpy.ma.", "numsharp.ma.")):
            body = re.search(rf"{block}\s*=\s*new\(\)\s*\{{(.*?)\n\s*\}};", text, re.S)
            self.assertIsNotNone(body, f"{block} initializer not found; update this parser")
            for name, canonical in re.findall(r'\["([^"]+)"\]\s*=\s*"([^"]+)"', body.group(1)):
                row_id = numpy_prefix + name if numpy_prefix + name in ids else numsharp_prefix + name
                self.assertIn(row_id, ids, f"{block}[{name}] names no catalog row")
                self.assertIn(canonical, self.labels, f"{block}[{name}] -> '{canonical}' is not a corpus key")
                if row_id in self.result.evidence and self.result.evidence[row_id].contracts:
                    continue   # the alias row is gated directly (e.g. broadcast, result_type)
                self.assertIn(row_id, aliases, f"{block}[{name}] has no reviewed alias in oracle_map.json")
                self.assertIn(aliases[row_id]["via"], self.labels[canonical],
                              f"{row_id} must borrow from the row '{canonical}' resolves to")
                checked += 1
        # A non-vacuity floor, not a target: an alias row the corpus gates directly is skipped above, so the count
        # falls as tiers gate alias spellings themselves (the array-object oracle took it to 29 on 2026-10-07).
        self.assertGreater(checked, 20)


def _strings(value):
    """Yield every string inside a nested JSON value (the map's id references).

    :param value: a parsed JSON value.
    :returns: a generator over its string leaves and dict keys.
    """
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


if __name__ == "__main__":
    unittest.main()
