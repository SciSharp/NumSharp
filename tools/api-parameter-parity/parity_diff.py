"""Write the change list that brings NumSharp's np.* surface to NumPy 2.4.2's: one diff entry per API.

Usage (from this directory, after signature_diff.py and validate_findings.py):
    python -I parity_diff.py <modules_dump.json> <types_dump.json> <diff.csv> <validation.json> <out.md> \
        [--coverage <coverage.json>] [--commit <sha>]

The gap map (signature_diff.py's CSV) says WHAT differs, per C# spelling and mark. This tool turns each validated
finding into the edit it asks for, written as a ``diff`` block so the change reads like a patch:

- ``-`` what must go: an overload beyond the one NumPy has, a parameter NumPy does not have, a dtype taken as
  ``NPTypeCode`` / ``Type`` or picked by a generic type argument, a C# type that refuses NumPy's ``None`` or a sequence;
- ``+`` what must come: NumPy's missing parameters with their defaults, NumPy's name for a renamed parameter, ``DType``
  for a dtype, the types that take ``None`` / a sequence of integers, a missing API;
- context lines: NumPy's signature (the target), the overload that stays (the closest one), the NumPy-style calls the
  compiler refuses today, and the slot / requiredness / default differences of the overload that stays.

Findings validate_findings.py refuted are left out of the entries and listed in their own section with the evidence. A
NumSharp-only member (a name NumPy does not have) is listed apart: the one-overload rule is the owner's, not NumPy's.
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import signature_diff as sd  # noqa: E402  (the groups, NumPy's names and the module list)

#: Group order in the report: the headline surfaces first, then numpy.ma, the strings, the polynomial package and the
#: object APIs in the gap map's own order.
GROUP_ORDER = ("np", "ndarray", "ndarray (NDArray<T>)", "np.linalg", "np.fft", "np.random", "np.random.Generator")


def split_cell(cell):
    """Split a ``"; "``-joined CSV cell.

    Args:
        cell: The cell.

    Returns:
        The non-empty entries.
    """
    return [part.strip() for part in cell.split("; ") if part.strip()]


def numpy_defaults(signature):
    """NumPy's parameters with their defaults, from the CSV's ``numpy_signature`` text.

    Args:
        signature: ``sum(a, axis=None, *, where=<no value>)``.

    Returns:
        ``{name: default text or None for a required parameter}``.
    """
    if "(" not in signature or not signature.endswith(")"):
        return {}
    inner = signature[signature.index("(") + 1:-1]
    out = {}
    for token in ([t.strip() for t in sd.split_top(inner)] if inner.strip() else []):
        if token in ("/", "*"):
            continue
        name, has_default, default = token.partition("=")
        out[name.strip()] = default.strip() if has_default else None
    return out


def cs_spelling(group, name):
    """How C# code reaches a gap-map spelling.

    Args:
        group: The group key.
        name: The member name or ``(constructor)``.

    Returns:
        ``np.sum``, ``a.sum`` (ndarray), ``NDArray<T>.reshape``, ``new NDMatrix(...)``, ``mx.argmax`` ...
    """
    receivers = {"ndarray": "a", "ndarray (NDArray<T>)": "NDArray<T>", "np.matrix": "NDMatrix",
                 "np.char.chararray": "NDCharArray", "np.ma.MaskedArray": "NDMaskedArray", "np.dtype": "DType",
                 "np.nditer": "np.NDIterator", "np.broadcast": "np.Broadcast", "np.flatiter": "np.FlatIterator",
                 "np.ndindex": "np.NDIndex", "np.ndenumerate": "np.NDEnumerate", "np.poly1d": "poly1d",
                 "np.vectorize": "Vectorized", "np.finfo": "finfo", "np.iinfo": "iinfo",
                 "np.lib.npyio.NpzFile": "NpzFile", "np.random.Generator": "Generator",
                 "np.random.SeedSequence": "SeedSequence", "np.random.BitGenerator": "BitGenerator",
                 "np.random.MT19937": "MT19937", "np.random.PCG64": "PCG64", "np.random.PCG64DXSM": "PCG64DXSM",
                 "np.random.Philox": "Philox", "np.random.SFC64": "SFC64"}
    if group in receivers:
        receiver = receivers[group]
        return f"new {receiver}(...)" if name == sd.CONSTRUCTOR else f"{receiver}.{name}"
    access = {"np.char": "np.@char"}.get(group, group)
    return f"{access}.{name}" if name != sd.CONSTRUCTOR else f"new {access}(...)"


def finding_param(text):
    """The parameter a probe finding is about.

    Args:
        text: One finding (`` `axis`: argument 2: ... ``, `` `axis` as a sequence: ... ``).

    Returns:
        The parameter name, or None for a finding about the whole call.
    """
    match = re.match(r"`(\w+)`(?::| as a sequence)", text)
    return match.group(1) if match else None


def entry_lines(row, refuted):
    """The diff lines of one spelling.

    Args:
        row: Its CSV row.
        refuted: ``{mark: set of keys}`` from validate_findings.py (``(group, name, param)`` for M / X / Q / L).

    Returns:
        ``(lines, marks)``: the diff body (without NumPy's signature and the kept overload) and the marks that still
        hold after validation.
    """
    group, name = row["module"], row["name"]
    defaults = numpy_defaults(row["numpy_signature"])
    lines, marks = [], []
    overloads = [o.strip() for o in row["all_overloads"].split(" || ") if o.strip()]
    if len(overloads) > 1:
        marks.append(f"O{len(overloads)}")
        for overload in overloads:
            if overload != row["best"]:
                lines.append(f"- {overload}")
    missing = []
    for entry in split_cell(row["missing_any_overload"]) if row["mark_M"] == "●" else []:
        np_name = re.sub(r" \(C#: [^)]*\)$", "", entry)
        if (group, name, np_name) in refuted["M"]:
            continue
        renamed = re.search(r"\(C#: ([^)]*)\)", entry)
        if renamed:
            lines.append(f"- {renamed.group(1)}")
            lines.append(f"+ {np_name}  (NumPy's keyword; rename)")
        else:
            default = defaults.get(np_name.lstrip("*"))
            lines.append(f"+ {np_name}" + (f"={default}" if default is not None else ""))
        missing.append(np_name)
    if missing:
        marks.append("M")
    foreign = [p for p in split_cell(row["numsharp_only_params"]) if (group, name, p) not in refuted["X"]] \
        if row["mark_X"] == "●" else []
    renamed_targets = {re.search(r"\(C#: ([^)]*)\)", e).group(1) for e in split_cell(row["missing_any_overload"])
                       if "(C#:" in e}
    plain_foreign = [p for p in foreign if not any(p in t.split(", ") for t in renamed_targets)]
    for param in plain_foreign:
        lines.append(f"- {param}  (NumPy has no such parameter)")
    if foreign:
        marks.append("X")
    if row["mark_T"]:
        marks.append("T" + ("◇" if "◇" in row["mark_T"] else ""))
        # A dtype parameter typed NPTypeCode / Type becomes a DType of the same name (DType converts from both, and
        # from NumPy's dtype strings); "NPTypeCode from" and "Type from" are one change.
        replaced = []
        for param in split_cell(row["typecode_params"]):
            lines.append(f"- {param}")
            name_only = param.split(" ")[-1]
            if name_only not in replaced:
                replaced.append(name_only)
        for name_only in replaced:
            lines.append(f"+ DType {name_only}")
        if len(overloads) <= 1:
            # With several overloads the generic ones are already among the deletions above.
            for selector in split_cell(row["typecode_generic_selectors"]):
                lines.append(f"- {selector}  (the dtype picked by a type argument; NumPy passes dtype=)")
    for kind in ("required", "default"):
        for item in split_cell(row[kind]):
            lines.append(f"  {kind}: {item}")
    for item in split_cell(row["slot"]):
        lines.append(f"  order: {item}")
    for item in split_cell(row["varargs"]):
        lines.append(f"  *args: {item}")
    if row["mark_L"] in ("●", "◐", "◦"):
        kept = [f for f in sd_split_findings(row["probe_null"])
                if (group, name, finding_param(f) or "") not in refuted["L"]]
        if kept:
            marks.append("L" + ("" if row["mark_L"] == "●" else row["mark_L"]))
            for finding in kept:
                lines.append(f"- {finding}")
                param = finding_param(finding)
                if param:
                    lines.append(f"+ {param}: a type that takes null (NumPy's None)")
    if row["mark_Q"] in ("●", "◐"):
        kept = [f for f in sd_split_findings(row["probe_sequence"])
                if (group, name, finding_param(f) or "") not in refuted["Q"]]
        if kept:
            marks.append("Q")
            for finding in kept:
                lines.append(f"- {finding}")
                param = finding_param(finding)
                if param:
                    lines.append(f"+ {param}: an integer or a sequence of integers")
    if row["mark_B"] in ("●", "◐") and (group, name) not in refuted["B"]:
        marks.append("B")
        for finding in sd_split_findings(row["probe_binding"]):
            lines.append(f"  does not compile: {finding}")
    return lines, marks


def sd_split_findings(cell):
    """Split a compile-probe cell into findings (a finding can hold "; " in its own text).

    Args:
        cell: ``probe_binding`` / ``probe_null`` / ``probe_sequence``.

    Returns:
        The findings.
    """
    starts = ("`", "required parameters only", "several ", "without ", "positionally up to", "all NumPy parameters",
              "the None values")
    findings = []
    for piece in (cell.split("; ") if cell else []):
        if findings and not piece.startswith(starts):
            findings[-1] += "; " + piece
        else:
            findings.append(piece)
    return [f.strip() for f in findings if f.strip()]


def write_report(args, rows, validation, surface, modules_missing, coverage):
    """Write the Markdown change list.

    Args:
        args: The parsed command line.
        rows: Every CSV row.
        validation: validate_findings.py's JSON.
        surface: ``run_analysis``' per-group ``cs_only`` / ``np_absent`` / ``elsewhere``.
        modules_missing: ``missing_modules``' list.
        coverage: The coverage artifact (coverage.json) or None.
    """
    refuted = {mark: {tuple(item) for item in items} for mark, items in validation["refuted"].items()}
    entries = defaultdict(list)
    numsharp_only = []
    totals = Counter()
    for row in rows:
        group = row["module"]
        lines, marks = entry_lines(row, refuted)
        if row["numpy_kind"] == "absent":
            if marks:
                numsharp_only.append((row, lines, marks))
            continue
        totals["numpy"] += 1
        if not marks:
            totals["identical"] += 1
            # An identical name may still miss parameters the parity plan excludes (like=, subok=, a ufunc's
            # casting= / order= / signature=): counted, not listed.
            totals["identical_excluded_only"] += row["mark_M"] == "◦"
            continue
        totals["differs"] += 1
        for mark in marks:
            totals[mark[0]] += 1
        entries[group].append((row, lines, marks))
    groups = [g for g in GROUP_ORDER if g in entries] + sorted(g for g in entries if g not in GROUP_ORDER)
    out = []
    w = out.append
    w("# NumSharp's np.* surface against NumPy 2.4.2: the change list")
    w("")
    w(f"> Generated by `tools/api-parameter-parity/parity_diff.py` on `{args.commit or 'the working tree'}` from the gap "
      "map (`dump_modules.cs` → `binding_probe.py` → `signature_diff.py`) and its validation "
      "(`validate_findings.py`). Regenerate with the commands in `docs/plans/API_DIFF_GAP.md` §10, then "
      "`validate_findings.py` and this tool.")
    w("")
    w("Each entry is one C# spelling of one NumPy API, written as a patch: `-` is what must go, `+` what must come, and "
      "the context lines give NumPy's signature (the target), the overload that stays (the closest one) and what the "
      "compiler refuses today. The marks: **O** several overloads (NumPy has one), **M** a NumPy parameter no overload "
      "offers, **X** a parameter NumPy does not have, **T** a dtype as `NPTypeCode` / `Type` (◇: picked by a generic "
      "type argument), **L** `null` refused for NumPy's None (◐ C# 12 only, ◦ only a `Shape` refuses it and `default` "
      "binds), **Q** a sequence of integers refused, **B** a NumPy-style call that does not compile.")
    w("")
    w("## Summary")
    w("")
    w(f"- NumPy names with a C# spelling: **{totals['numpy']}**; NumPy's signature: **{totals['identical']}** "
      f"({totals['identical_excluded_only']} of them lack only parameters the parity plan excludes: `like=`, "
      f"`subok=`, a ufunc's `casting=` / `order=` / `signature=`); to change: **{totals['differs']}**.")
    w("- Spellings per mark after validation: " + ", ".join(f"{m} {totals[m]}" for m in "OMXTLQB") + ".")
    absent_total = sum(len(s["np_absent"]) for s in surface.values())
    w(f"- NumPy names absent from the compared groups: **{absent_total}**; NumPy public modules with no facade: "
      f"**{len(modules_missing)}** ({sum(len(names) for _, names, _ in modules_missing)} names).")
    summary = validation["summary"]
    w(f"- Validation: M {summary['m_compiler']['confirmed']} parameter checks confirmed by the compiler and "
      f"{summary['numpy_keywords']['m_confirmed']} by NumPy, X {summary['numpy_keywords']['x_confirmed']} confirmed "
      f"by NumPy, Q {summary['q_numpy']['confirmed']} and L {summary['l_numpy']['confirmed']} by NumPy, "
      f"{summary['m_compiler'].get('absent_confirmed', 0)} absent names by the compiler; refuted and left out: "
      + ", ".join(f"{k} {v}" for k, v in summary["refuted"].items()) + " (section \"Refuted findings\").")
    if coverage:
        scope = coverage["summary"]["api_scope"]
        headline = coverage["summary"]["default_scope"]
        w(f"- Coverage artifact: headline {headline['available']} / {headline['total']} available, "
          f"{headline['signature_identical']} with NumPy's exact signature; all NumPy APIs {scope['available']} / "
          f"{scope['total']} available ({scope['coverage_percent']:.1f}%), {scope['missing']} missing.")
    w("")
    w("| Group | To change | O | M | X | T | L | Q | B |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for group in groups:
        counter = Counter(mark[0] for _, _, marks in entries[group] for mark in marks)
        w(f"| `{group}` | {len(entries[group])} | " + " | ".join(str(counter[m]) for m in "OMXTLQB") + " |")
    w("")
    w("## The change list")
    for group in groups:
        w("")
        w(f"### `{group}` ({len(entries[group])})")
        for row, lines, marks in sorted(entries[group], key=lambda e: e[0]["name"].lower()):
            w("")
            w("```diff")
            w(f"@@ {cs_spelling(group, row['name'])}  [{' '.join(marks)}] @@")
            w(f"  NumPy: {row['numpy_signature']}")
            if row["best"]:
                w(f"  keep:  {row['best']}")
            out.extend(lines)
            w("```")
    w("")
    w("## Completeness: NumPy names NumSharp lacks")
    w("")
    for group, info in surface.items():
        if not info["np_absent"]:
            continue
        w(f"### `{group}` ({len(info['np_absent'])})")
        w("")
        w("```diff")
        for name in info["np_absent"]:
            where = info["elsewhere"].get(name) or []
            w(f"+ {cs_spelling(group, name)}" + (f"  (NumSharp has the name under {', '.join(where)})" if where else ""))
        w("```")
        w("")
    w("### NumPy public modules with no NumSharp facade")
    w("")
    w("```diff")
    # missing_modules gives (module path, names, {name: groups that have it}).
    for path, names, elsewhere in modules_missing:
        shown = ", ".join(names[:12]) + (", ..." if len(names) > 12 else "")
        also = f"; NumSharp has {len(elsewhere)} of them elsewhere" if elsewhere else ""
        w(f"+ np.{path}  ({len(names)} names: {shown}{also})")
    w("```")
    if coverage:
        w("")
        w("### What else the coverage artifact counts as missing")
        w("")
        missing = [r for r in coverage["rows"] if r.get("availability") in ("missing", "unsupported", "partial")
                   and r.get("origin", "numpy") == "numpy"]
        by_kind = Counter()
        for r in missing:
            kind = "Python protocol" if r.get("python_protocol") else r.get("kind", "?")
            by_kind[(r.get("category") or "?", kind)] += 1
        w("| Category | Kind | Rows |")
        w("|---|---|---:|")
        for (category, kind), count in sorted(by_kind.items(), key=lambda kv: -kv[1]):
            w(f"| {category} | {kind} | {count} |")
    w("")
    w("## NumSharp-only names with several overloads or a typecode dtype")
    w("")
    w("Not NumPy API (NumPy has no such name), so parity does not require a change; the owner's one-overload rule "
      "applies to them only if the owner extends it.")
    for row, lines, marks in sorted(numsharp_only, key=lambda e: (e[0]["module"], e[0]["name"])):
        w("")
        w("```diff")
        w(f"@@ {cs_spelling(row['module'], row['name'])}  [{' '.join(marks)}] @@")
        out.extend(lines)
        w("```")
    w("")
    w("## Refuted findings")
    w("")
    w("The gap map reported these; the compiler or NumPy's runtime says otherwise (validate_findings.py).")
    w("")
    w("| Mark | Spelling | Parameter | Evidence |")
    w("|---|---|---|---|")
    m_compiler = {tuple(x["key"]) + (x["param"],): x["call"] for x in validation["m_compiler"]["refuted"]}
    kw = validation["numpy_keywords"]
    m_numpy = {tuple(x["key"]) + (x["param"],): x["detail"] for x in kw["m_refuted"]}
    x_numpy = {tuple(x["key"]) + (x["param"],): x["detail"] for x in kw["x_refuted"]}
    q_numpy = {tuple(x["key"]): x["detail"] for x in validation["q_numpy"]["refuted"]}
    for key, call in sorted(m_compiler.items()):
        # The gap map reads an own_only group's OWN members (NDMatrix, NDArray<T>); C# also binds the members it
        # inherits from NDArray, so the NumPy-style call compiles, but it reaches the base member, not the class's own.
        w(f"| M | `{cs_spelling(key[0], key[1])}` | `{key[2]}` | compiles: `{call}` binds the overload inherited from "
          f"NDArray; the class's own `{key[1]}` has no `{key[2]}` |")
    for key, detail in sorted(m_numpy.items()):
        w(f"| M | `{cs_spelling(key[0], key[1])}` | `{key[2]}` | NumPy refuses the keyword: {detail} |")
    for key, detail in sorted(x_numpy.items()):
        w(f"| X | `{cs_spelling(key[0], key[1])}` | `{key[2]}` | NumPy accepts the keyword: {detail} |")
    for key, detail in sorted(q_numpy.items()):
        w(f"| Q | `{cs_spelling(key[0], key[1])}` | `{key[2]}` | NumPy refuses a tuple: {detail} |")
    for artifact in validation["artifacts"]:
        w(f"| B | `{cs_spelling(*artifact['key'])}` | | probe artifact: {artifact['errors'][0][0]} in `{artifact['call']}` |")
    with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(out) + "\n")
    print(json.dumps({"numpy": totals["numpy"], "identical": totals["identical"], "differs": totals["differs"],
                      **{m: totals[m] for m in "OMXTLQB"}, "absent": absent_total,
                      "modules_missing": len(modules_missing)}))


def main():
    """Read the inputs and write the change list.

    Raises:
        SystemExit: Wrong command-line arguments (argparse).
    """
    parser = argparse.ArgumentParser(description="Write the np.* parity change list.")
    for name in ("dump", "types", "csv", "validation", "out"):
        parser.add_argument(name)
    parser.add_argument("--coverage", help="the coverage artifact's coverage.json (adds its missing-row census)")
    parser.add_argument("--commit", help="the commit the inputs were taken from")
    args = parser.parse_args()
    with open(args.dump, encoding="utf-8") as handle:
        dump = json.load(handle)
    with open(args.types, encoding="utf-8") as handle:
        types = json.load(handle)
    with open(args.csv, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with open(args.validation, encoding="utf-8") as handle:
        validation = json.load(handle)
    coverage = None
    if args.coverage:
        with open(args.coverage, encoding="utf-8") as handle:
            coverage = json.load(handle)
    groups, _, _, surface, _ = sd.run_analysis(dump, types, None)
    write_report(args, rows, validation, surface, sd.missing_modules(groups), coverage)


if __name__ == "__main__":
    main()
