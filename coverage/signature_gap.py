"""The API diff gap map (``tools/api-parameter-parity``) as the inventory's source of signature marks.

Why this module exists
----------------------
``docs/plans/API_DIFF_GAP.md`` is the complete map of how NumSharp's C# signatures differ from NumPy 2.4.2's. Its
tools find more than the inventory's own comparison (``signature_parity.assess``, generator 1.15.0) could:

* NumPy's side completed from NumPy's ``.pyi`` stubs and numpydoc ``Parameters`` sections where the runtime signature
  forwards ``(*args, **kwargs)`` (``np.printoptions``' ``formatter`` / ``legacy``, ``np.frompyfunc``'s ``identity``),
  and the documented line of every C-implemented class method (``matrix.argsort``'s ``kind`` / ``stable``);
* more C# spellings: constructors (NumPy's class call, ``new NDArray(...)``), extension methods (they bind as
  ``a.name(...)``), the typed ``NDArray<T>``, and ``numpy.char.chararray`` through ``NDCharArray``;
* what only a compiler knows (``binding_probe.py`` compiles 7,000+ NumPy-style calls at C# 12 and C# 14): **B** a
  NumPy-style call does not bind (``np.linspace(0, 1)``, ``a.transpose(1, 0)``), **L** ``null`` refused where NumPy's
  default is None, **Q** an integer sequence refused where NumPy's stubs allow one (``axis=(0, 1)``); and from
  reflection **T ◇**, a generic overload whose type argument picks the dtype (``np.zeros<T>(shape)``).

Re-implementing those rules here would leave two engines to drift apart. So the generator RUNS the gap map's tools
(``run``) on the working tree, reads the CSV they publish (the interface ``API_DIFF_GAP.md`` §1 documents), and
``signature_parity.attach`` joins every compared C# spelling onto the catalog row of the NumPy API it implements:
the inventory and the gap map then report the same marks by construction.

The join (``catalog_id``): a group's NumPy path is ``numpy`` + the group key without its ``np`` (``np.ma`` ->
``numpy.ma``, ``ndarray`` -> ``numpy.ndarray``, ``np.matrix`` -> ``numpy.matrix``); a name is the member under it; a
``(constructor)`` row is the class row itself (``new NDArray(...)`` spells ``numpy.ndarray``'s class call); the
``ndarray (NDArray<T>)`` group spells ``numpy.ndarray``'s members on the typed receiver. A NumSharp-only name (NumPy
has no counterpart) joins the NumSharp-only catalog row of the same C# member (``GapMap.group_types`` gives the
member's type).
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

#: The directory of the gap map's tools, relative to the repository root.
TOOLS = Path("tools") / "api-parameter-parity"

#: The gap map's group for the typed ``NDArray<T>``, which spells ``numpy.ndarray``'s members and class call.
TYPED_NDARRAY_GROUP = "ndarray (NDArray<T>)"

#: The CLR type behind the typed group (the generic definition's full name).
TYPED_NDARRAY_TYPE = "NumSharp.Generic.NDArray`1"

#: The row name the gap map gives a class call (a C# constructor).
CONSTRUCTOR = "(constructor)"

#: NumPy classes whose names are lowercase, so the case of a group key's last segment does not tell them from
#: modules (``np.matrix`` is a class, ``np.ma`` a module).
LOWERCASE_CLASSES = frozenset({"matrix", "dtype", "nditer", "broadcast", "flatiter", "ndindex", "ndenumerate",
                               "poly1d", "vectorize", "finfo", "iinfo", "chararray"})

#: The CSV columns this module reads. A gap-map CSV without one of them is a different interface: ``read_csv`` refuses
#: it, so a change on the tools' side fails the generator (and ``test_signature_parity``) instead of emptying marks.
REQUIRED_COLUMNS = (
    "module", "name", "numpy_kind", "status", "overloads", "numpy_signature", "best", "slot", "required", "default",
    "varargs", "split", "renamed", "extra", "null_default", "mark_M", "mark_X", "mark_O", "mark_T", "mark_B", "mark_L",
    "mark_Q", "missing_any_overload", "missing_excluded", "numsharp_only_params", "typecode_params",
    "typecode_generic_selectors", "numpy_params_added", "extension_hosts", "probe_binding", "probe_null",
    "probe_sequence",
)

#: The compile probe's summary line (``7259 probes (0 names skipped): 488 fail at C# 12, 455 at C# 14``).
_PROBE_LINE = re.compile(r"(\d+) probes \((\d+) names? skipped\): (\d+) fail at C# 12, (\d+) at C# 14")


@dataclass
class GapMap:
    """The gap map's outputs for one tree.

    Attributes:
        rows: One dictionary per C# name (``numpy-signature-diff.csv``'s rows), NumSharp-only names included.
        totals: The comparison's totals (``signature_diff.py``'s first summary line: names, overloads, numpy, ...).
        marks: The mark counts (its second line: numpy, M, M_excluded_only, X, O, T, T_generic, B, B_12, L, L_12,
            L_default, Q, no_mark, ...): the numbers ``API_DIFF_GAP.md`` quotes.
        group_types: Group key -> the full name of the CLR type whose members the group compares (a module facade,
            an object API's class, ``NDArray<T>``), to join NumSharp-only names by their C# member.
        probe: The compile probe's own counts (``probes``, ``skipped``, ``fail_12``, ``fail_14``), empty when unknown.
    """

    rows: list[dict[str, str]]
    totals: dict[str, Any]
    marks: dict[str, Any]
    group_types: dict[str, str] = field(default_factory=dict)
    probe: dict[str, Any] = field(default_factory=dict)


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read the gap map's CSV.

    Args:
        path: ``numpy-signature-diff.csv`` (the committed one or a fresh run's).

    Returns:
        The rows as the csv module reads them (every value a string).

    Raises:
        SystemExit: When a column this module reads is absent (the tools changed their interface).
    """
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [column for column in REQUIRED_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"The gap map's CSV ({path}) lacks the columns {missing}: update coverage/signature_gap.py "
                             "to the interface of tools/api-parameter-parity/signature_diff.py.")
        return list(reader)


def split_list(text: str) -> list[str]:
    """Split one of the CSV's list cells (``"; "``-joined) back into its items.

    Args:
        text: The cell text (may be empty).

    Returns:
        The items; empty for an empty cell.
    """
    return [item for item in (part.strip() for part in text.split("; ")) if item] if text else []


#: How a compile-probe finding starts in the CSV (``signature_diff.classify_probes``' ``what`` texts). The CSV joins a
#: name's findings with ``"; "``, and a finding's own text can hold one too ("no overload fits; the closest one wants
#: `int` for argument 2"), so a piece starts a new finding only when it begins like one.
_FINDING_STARTS = ("`", "required parameters only", "several ", "without ", "positionally up to",
                   "all NumPy parameters", "the None values")


def split_findings(text: str) -> list[str]:
    """Split a compile-probe cell (``probe_binding`` / ``probe_null`` / ``probe_sequence``) into its findings.

    Args:
        text: The cell text (may be empty).

    Returns:
        One text per finding (``"required parameters only: needs `num` — `np.linspace(1.0, 1.0)`"``); a ``"; "``
        inside a finding's own text stays in it.
    """
    findings: list[str] = []
    for piece in (text.split("; ") if text else []):
        if findings and not piece.startswith(_FINDING_STARTS):
            findings[-1] += "; " + piece
        else:
            findings.append(piece)
    return [finding.strip() for finding in findings if finding.strip()]


def group_path(group: str) -> str:
    """The NumPy path a gap-map group compares against.

    Args:
        group: A group key (``np``, ``ndarray``, ``np.ma``, ``np.matrix``, ``np.lib.npyio.NpzFile``, the typed group).

    Returns:
        The dotted ``numpy`` path: ``numpy`` for ``np``, ``numpy.ndarray`` for ``ndarray`` and the typed group,
        ``numpy.`` + the key without ``np.`` otherwise.
    """
    if group == "np":
        return "numpy"
    if group in ("ndarray", TYPED_NDARRAY_GROUP):
        return "numpy.ndarray"
    return "numpy." + (group[3:] if group.startswith("np.") else group)


def catalog_id(group: str, name: str) -> str:
    """The catalog row a gap-map row (one C# spelling of a NumPy API) describes.

    Args:
        group: The row's group key.
        name: The C# name, or ``(constructor)`` for a class call.

    Returns:
        The ``numpy.*`` id: the class row itself for a constructor, else the member under the group's path.
    """
    path = group_path(group)
    return path if name == CONSTRUCTOR else f"{path}.{name}"


def is_module_group(group: str) -> bool:
    """Whether a group compares a NumPy MODULE's functions (rather than a class's members).

    Args:
        group: The group key.

    Returns:
        True for ``np`` and the ``np.<module>`` groups (``np.ma``, ``np.polynomial.chebyshev``); False for ``ndarray``,
        the typed group and the classes (``np.matrix``, ``np.random.Generator``, ``np.lib.npyio.NpzFile``).
    """
    if group == "np":
        return True
    if not group.startswith("np."):
        return False
    last = group.rsplit(".", 1)[-1]
    return last[:1].islower() and last not in LOWERCASE_CLASSES


def csharp_spelling(group: str, name: str, group_types: Mapping[str, str]) -> str:
    """How a port spells the call in C#, for the record's list of spellings.

    Args:
        group: The row's group key.
        name: The C# name, or ``(constructor)``.
        group_types: ``GapMap.group_types``.

    Returns:
        ``np.sum`` / ``np.@char.find`` (a module function), ``a.sum`` (an NDArray instance), ``NDArray<T>.reshape``,
        ``new NDArray(...)`` (a constructor), ``NDMatrix.sum`` / ``Generator.normal`` (a member of another class).
    """
    short = group_types.get(group, group).rsplit(".", 1)[-1].replace("+", ".").replace("`1", "<T>")
    if name == CONSTRUCTOR:
        return f"new {short}(...)"
    if group == "ndarray":
        return f"a.{name}"
    if group == TYPED_NDARRAY_GROUP:
        return f"NDArray<T>.{name}"
    if is_module_group(group):
        return f"{'np.@char' if group == 'np.char' else group}.{name}"
    return f"{short}.{name}"


def module_group_types(modules_dump: Mapping[str, Any]) -> dict[str, str]:
    """The CLR type of every group in ``dump_modules.cs``' first output (the module facades and ``np.random.Generator``).

    Args:
        modules_dump: The module dump (group key -> {"type": CLR full name, ...}).

    Returns:
        Group key -> CLR full name (``+`` for a nested type, as the dump writes it).
    """
    return {key: info["type"] for key, info in modules_dump.items()}


def object_group_types(object_surfaces: Sequence[Any]) -> dict[str, str]:
    """The CLR type of every object group, from the coverage's own object surfaces (the map the gap tools copied).

    Args:
        object_surfaces: ``object_surfaces.OBJECT_SURFACES`` (each with ``path`` and ``clr_type``).

    Returns:
        ``np.<path>`` -> CLR type for every surface NumSharp implements, plus the typed ``NDArray<T>`` group.
    """
    types = {"np." + surface.path: surface.clr_type for surface in object_surfaces if surface.clr_type}
    types[TYPED_NDARRAY_GROUP] = TYPED_NDARRAY_TYPE
    return types


def _run(command: Sequence[Any], cwd: Path, what: str) -> str:
    """Run one of the gap map's tools and return its standard output.

    Args:
        command: The command line (paths are converted to text).
        cwd: The working directory (the tools' directory: their relative project paths assume it).
        what: A short name for error messages (``dump``, ``probe``, ``diff``).

    Returns:
        The standard output text.

    Raises:
        SystemExit: When the tool fails; its output is copied to standard error first.
    """
    command = [str(part) for part in command]
    completed = subprocess.run(command, cwd=cwd, check=False, text=True, capture_output=True, encoding="utf-8",
                               errors="replace")
    if completed.returncode:
        sys.stderr.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        raise SystemExit(f"The gap map's {what} step failed: {' '.join(command)}")
    return completed.stdout


def json_lines(text: str) -> list[dict[str, Any]]:
    """The JSON objects a tool printed one per line (its summary lines).

    Args:
        text: The tool's standard output.

    Returns:
        Every line that parses as a JSON object, in order.
    """
    found = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                found.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return found


def probe_counts(text: str) -> dict[str, int]:
    """Read the compile probe's summary line.

    Args:
        text: ``binding_probe.py``'s standard output.

    Returns:
        ``probes``, ``skipped``, ``fail_12``, ``fail_14``; empty when the line is not found.
    """
    match = _PROBE_LINE.search(text)
    if not match:
        return {}
    probes, skipped, fail_12, fail_14 = (int(group) for group in match.groups())
    return {"probes": probes, "skipped": skipped, "fail_12": fail_12, "fail_14": fail_14}


def run(root: Path, workdir: Path, object_surfaces: Sequence[Any], python: str = sys.executable) -> GapMap:
    """Run the gap map's three tools on the working tree and load their outputs.

    The order is the one ``API_DIFF_GAP.md`` §10 gives: ``dump_modules.cs`` reflects every module facade and exported
    type (it builds NumSharp.Core in Debug: the DLL the probe then compiles against), ``binding_probe.py`` compiles the
    NumPy-style calls at C# 12 and C# 14, ``signature_diff.py`` compares and marks. About 80 seconds on a warm machine,
    most of it the Debug build.

    Args:
        root: The repository root.
        workdir: An empty scratch directory for the dumps, the probe project and the outputs.
        object_surfaces: ``object_surfaces.OBJECT_SURFACES`` (the object groups' CLR types).
        python: The interpreter for the two Python tools (the generator's own: NumPy 2.4.2 importable).

    Returns:
        The gap map.

    Raises:
        SystemExit: When a tool fails, prints no totals, or writes a CSV without the expected columns.
    """
    tools = root / TOOLS
    modules, types = workdir / "modules_dump.json", workdir / "types_dump.json"
    _run(["dotnet", "run", "--no-cache", "dump_modules.cs", modules, types], tools, "dump")
    dll = root / "src" / "NumSharp.Core" / "bin" / "Debug" / "net10.0" / "NumSharp.dll"
    probe_out = _run([python, "-I", "binding_probe.py", modules, types, dll, workdir / "probe",
                      workdir / "probe_results.json"], tools, "probe")
    out = _run([python, "-I", "signature_diff.py", modules, workdir / "diff.md", workdir / "diff.csv",
                workdir / "marks.md", "--types", types, "--probe", workdir / "probe_results.json"], tools, "diff")
    summaries = json_lines(out)
    if len(summaries) < 2:
        raise SystemExit("signature_diff.py printed no totals: the gap map's interface changed.")
    with open(modules, encoding="utf-8") as fh:
        group_types = module_group_types(json.load(fh))
    group_types.update(object_group_types(object_surfaces))
    return GapMap(read_csv(workdir / "diff.csv"), summaries[-2], summaries[-1], group_types, probe_counts(probe_out))
