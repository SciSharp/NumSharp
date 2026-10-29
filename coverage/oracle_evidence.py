"""Join the committed NumPy-oracle contracts into the coverage catalog, one API row at a time.

The coverage artifact answers "does NumSharp expose this NumPy API?" from the compiled surface alone.
That is availability, not parity. Parity is what the committed oracle corpora prove: every contract is
real NumPy 2.4.2 output that a CI test replays against NumSharp and requires to be bit-exact, or a
documented, registry-excused divergence (never a silent one). This module turns those corpora into
per-API evidence, so the dashboard can say "available AND oracle-verified" instead of stopping at
"available".

Four oracle sources are joined, each replayed by its own test:

* ``fuzz``   -- ``test/NumSharp.Tests.Oracle/Fuzz/corpus/*.jsonl``, the differential-fuzz corpus
  replayed by the ``FuzzMatrix`` gate (FuzzCorpusTests*, IndexOracleTests).
* ``npy``    -- ``test/NumSharp.Tests/IO/corpus/npy_oracle.zip``, real ``np.save``/``np.savez`` output
  replayed by NpyOracleTests.
* ``flags``  -- ``test/NumSharp.Tests/Backends/corpus/flags_oracle.jsonl`` (FlagsOracleTests).
* ``layout`` -- ``test/NumSharp.Tests/Backends/corpus/layout_parity_oracle.jsonl`` (LayoutParityOracleTests).

Contracts are grouped by (source, file, key, variant) while scanning, and each group resolves to one or
more catalog row ids through ``oracle_map.json`` (the reviewed table) or the automatic namespace rules
documented there. Resolution is STRICT: a key nothing resolves, an id the catalog lacks, an explicit
entry that is unused or merely repeats an automatic rule, and a corpus file no replay test names are
all reported as problems, and the generator refuses to write an artifact while any exist. The same
"a silent miss must be structurally hard" rule the rest of the coverage tool follows.

What the evidence does NOT claim: that every contract is bit-exact (the replay decides that at test
time, and a few are documented divergences), or that an API without contracts is wrong (many are
gated by dedicated unit suites instead; see OracleSurfaceCoverageTests' sibling-owned lists).
"""

from __future__ import annotations

import functools
import json
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
MAP_PATH = ROOT / "coverage" / "oracle_map.json"
FUZZ_DIR = ROOT / "test" / "NumSharp.Tests.Oracle" / "Fuzz"
NPY_ORACLE = ROOT / "test" / "NumSharp.Tests" / "IO" / "corpus" / "npy_oracle.zip"
FLAGS_ORACLE = ROOT / "test" / "NumSharp.Tests" / "Backends" / "corpus" / "flags_oracle.jsonl"
LAYOUT_ORACLE = ROOT / "test" / "NumSharp.Tests" / "Backends" / "corpus" / "layout_parity_oracle.jsonl"
MAP_SCHEMA_VERSION = 1

# Automatic rules never target these kinds: a corpus key names an operation, and landing on a
# module (np.fft), a class (np.dtype) or a constant (np.r_) is only ever right when a reviewer says so
# in an explicit entry. Without this, the bare key "fft" would credit the numpy.fft MODULE row.
AUTO_EXCLUDED_KINDS = frozenset({"class", "constant", "module"})

# Statuses under which the catalog credits NumSharp with the API. Evidence on a row outside this set
# is a contradiction (the corpus exercises an API the catalog calls missing) and is reported, never
# counted as verified.
CREDITED_STATUSES = frozenset({"available", "partial", "extension"})

# Host-pin classes. host-libm tiers are hard-gated only against the win-amd64 CRT libm, pinned-blas
# tiers only when the content-hash-pinned OpenBLAS loads; both are Inconclusive elsewhere.
PIN_CLASSES = frozenset({"host-libm", "pinned-blas"})


@dataclass(frozen=True)
class SourceSpec:
    """One committed NumPy-output corpus and the test that replays it.

    :ivar id: short source key used in evidence records (``fuzz``/``npy``/``flags``/``layout``).
    :ivar label: human-readable name for summaries and the dashboard.
    :ivar path: repo-relative POSIX path of the corpus file or directory.
    :ivar replayed_by: the test class/category whose run turns the contracts into a pass/fail verdict.
    """

    id: str
    label: str
    path: str
    replayed_by: str


SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec("fuzz", "Differential-fuzz corpus", "test/NumSharp.Tests.Oracle/Fuzz/corpus",
               "FuzzMatrix (FuzzCorpusTests*, IndexOracleTests)"),
    SourceSpec("npy", ".npy/.npz format oracle", "test/NumSharp.Tests/IO/corpus/npy_oracle.zip",
               "NpyOracle (NpyOracleTests)"),
    SourceSpec("flags", "Array flags oracle", "test/NumSharp.Tests/Backends/corpus/flags_oracle.jsonl",
               "FlagsOracleTests"),
    SourceSpec("layout", "Result-layout parity oracle",
               "test/NumSharp.Tests/Backends/corpus/layout_parity_oracle.jsonl", "LayoutParityOracleTests"),
)


@dataclass
class ContractGroup:
    """Contracts that share a resolution: same source, file, key and variant.

    Grouping before resolving keeps the join O(groups) instead of O(contracts) (about 2,000 groups for
    about 250,000 contracts), and lets a param-driven key (``out_binary`` + ``ufunc=add``) resolve per
    variant without a second corpus pass.

    :ivar source: the :class:`SourceSpec` id the contracts come from.
    :ivar file: corpus file name (the archive name for ``npy``).
    :ivar key: the op key (``fuzz``) or the family variant string (specialized sources).
    :ivar variant: the resolution parameter's value for a param-driven fuzz key, else ``None``.
    :ivar pin: the host-pin class of the file (see :data:`PIN_CLASSES`), ``None`` when portable.
    :ivar cases: number of contracts in the group.
    :ivar errors: how many of them expect NumPy to raise (error-message contracts).
    :ivar dtypes: dtype names seen on operands, results, or the case's own dtype field.
    :ivar layouts: memory-layout / base-recipe labels the contracts exercise.
    """

    source: str
    file: str
    key: str
    variant: str | None
    pin: str | None
    cases: int = 0
    errors: int = 0
    dtypes: set[str] = field(default_factory=set)
    layouts: set[str] = field(default_factory=set)

    @property
    def label(self) -> str:
        """The display/search label: the key, suffixed ``:variant`` for param-driven fuzz keys.

        Specialized sources are prefixed with their source id (``npy:npz:compressed``) because their
        family names (``reshape``, ``sort``) would otherwise read like fuzz op keys.
        """
        if self.source != "fuzz":
            return f"{self.source}:{self.key}"
        return f"{self.key}:{self.variant}" if self.variant is not None else self.key


@dataclass
class OracleSources:
    """Every committed contract, grouped, plus the integrity problems found while scanning.

    :ivar groups: contract groups in deterministic (source, file, key, variant) order.
    :ivar files: per-file statistics (source, name, contracts, error contracts, pin, keys).
    :ivar problems: scan-time integrity problems (unreplayed corpus files, unkeyed rows, bad pin
        declarations). Non-empty means the evidence cannot be trusted as a whole.
    """

    groups: list[ContractGroup]
    files: list[dict[str, Any]]
    problems: list[str]


@dataclass
class RowEvidence:
    """Accumulated oracle evidence for one catalog row.

    :ivar contracts: direct contracts resolved to this row (0 for an alias-only row).
    :ivar errors: direct error-message contracts.
    :ivar sources: direct contract count per source id.
    :ivar keys: group labels that resolved here (op keys, ``key:variant``, ``npy:...``).
    :ivar files: corpus files the direct contracts live in.
    :ivar dtypes: dtype names the direct contracts exercise.
    :ivar layouts: layout labels the direct contracts exercise.
    :ivar pins: host-pin classes present among the direct contracts.
    :ivar portable: direct contracts hard-gated on every host (not host-pinned).
    :ivar fuzz_contracts: direct contracts from the ``fuzz`` source; an alias is stale only when this
        is non-zero (the specialized oracles pin layout/format facets, not the value path an alias
        stands in for).
    :ivar via: alias target record ``{"id", "contracts", "rule", "reason", "pinned"}`` when a reviewed
        or identity alias applies, else ``None``.
    """

    contracts: int = 0
    errors: int = 0
    sources: dict[str, int] = field(default_factory=dict)
    keys: set[str] = field(default_factory=set)
    files: set[str] = field(default_factory=set)
    dtypes: set[str] = field(default_factory=set)
    layouts: set[str] = field(default_factory=set)
    pins: set[str] = field(default_factory=set)
    portable: int = 0
    fuzz_contracts: int = 0
    via: dict[str, Any] | None = None

    def add(self, group: ContractGroup) -> None:
        """Fold one contract group into this row's direct evidence.

        :param group: the group that resolved to this row; its counts and label sets are merged.
        """
        self.contracts += group.cases
        self.errors += group.errors
        self.sources[group.source] = self.sources.get(group.source, 0) + group.cases
        self.keys.add(group.label)
        self.files.add(group.file)
        self.dtypes.update(group.dtypes)
        self.layouts.update(group.layouts)
        if group.pin:
            self.pins.add(group.pin)
        else:
            self.portable += group.cases
        if group.source == "fuzz":
            self.fuzz_contracts += group.cases

    @property
    def pinned(self) -> bool:
        """True when NO contract behind this row is hard-gated on every host.

        A row whose direct contracts are all host-pinned is verified only on the pinned host (win-amd64
        libm or the pinned OpenBLAS); an alias-only row inherits its target's pinning.
        """
        if self.contracts:
            return self.portable == 0
        return bool(self.via and self.via.get("pinned"))

    def record(self) -> dict[str, Any]:
        """Render the JSON-serializable per-row ``oracle`` record the artifact carries.

        :returns: a dict with a deterministic key order and sorted lists, so the artifact stays
            byte-identical across hosts (the ``--check`` contract).
        """
        record: dict[str, Any] = {
            "status": "direct" if self.contracts else "alias",
            "contracts": self.contracts,
            "error_contracts": self.errors,
            "sources": dict(sorted(self.sources.items())),
            "keys": sorted(self.keys),
            "files": sorted(self.files),
            "dtypes": sorted(self.dtypes),
            "layouts": len(self.layouts),
            # The boolean answers "verified only on the pinned host?"; the count keeps a partly pinned
            # row honest (np.fft.fft: every value contract is host-libm, one error contract is not).
            "pinned": self.pinned,
            "pinned_contracts": self.contracts - self.portable,
            "pins": sorted(self.pins),
        }
        if self.via:
            record["via"] = dict(self.via)
        return record


@dataclass
class OracleJoin:
    """The result of resolving every contract group against a catalog.

    :ivar evidence: row id -> accumulated evidence (direct and/or alias).
    :ivar resolutions: one entry per group label: ``{"label", "source", "ids", "rule", "cases"}``,
        sorted by label, for reports and tests.
    :ivar problems: every integrity problem (scan problems included). The generator must refuse to
        emit an artifact while this is non-empty.
    """

    evidence: dict[str, RowEvidence]
    resolutions: list[dict[str, Any]]
    problems: list[str]


def load_map(path: Path = MAP_PATH) -> dict[str, Any]:
    """Read and shape-check the reviewed oracle map.

    :param path: the JSON file to read (defaults to ``coverage/oracle_map.json``).
    :returns: the parsed map.
    :raises SystemExit: when the file is missing its schema version or a required section, so a
        malformed map fails the generator with a message instead of a KeyError deep in the join.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != MAP_SCHEMA_VERSION:
        raise SystemExit(f"Unsupported oracle map schema in {path} (expected {MAP_SCHEMA_VERSION}).")
    for section in ("fuzz", "aliases", "host_pins", "npy", "flags", "layout"):
        if not isinstance(data.get(section), dict):
            raise SystemExit(f"Oracle map {path} is missing its '{section}' section.")
    for section in ("bare_key_precedence", "prefix_rules", "files_without_op", "keys"):
        if not isinstance(data["fuzz"].get(section), dict):
            raise SystemExit(f"Oracle map {path} is missing fuzz.{section}.")
    return data


def _entries(section: Mapping[str, Any]) -> dict[str, Any]:
    """Return a map section without its human-readable ``note``/``description`` members.

    :param section: a section dict from the oracle map.
    :returns: the entries a resolver iterates; prose keys are documentation, not mappings.
    """
    return {key: value for key, value in section.items() if key not in {"note", "description"}}


def _param_keys(oracle_map: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    """Collect the fuzz keys whose target depends on a case parameter.

    The scan must know these BEFORE resolution: a param-driven key's contracts are grouped per
    parameter value (``out_binary`` splits into one group per ufunc), because each value resolves to a
    different row.

    :param oracle_map: the parsed oracle map.
    :returns: sorted ``(op key, parameter name)`` pairs, e.g. ``("out_binary", "ufunc")``.
    """
    keys = _entries(oracle_map["fuzz"]["keys"])
    return tuple(sorted((key, entry["param"]) for key, entry in keys.items() if "param" in entry))


def _host_libm_files(fuzz_dir: Path) -> set[str]:
    """Discover the tiers the replay runs through ``RunHostLibmCorpus`` (hard-gated only on win-amd64).

    Read from the replay source itself so a newly host-pinned tier cannot be reported as portable
    evidence just because nobody updated a list here.

    :param fuzz_dir: the ``test/NumSharp.Tests.Oracle/Fuzz`` directory.
    :returns: corpus file names passed to ``RunHostLibmCorpus("...")``.
    """
    pattern = re.compile(r'RunHostLibmCorpus\("([A-Za-z0-9_.]+\.jsonl)"\)')
    names: set[str] = set()
    for path in sorted(fuzz_dir.glob("FuzzCorpusTests*.cs")):
        names.update(pattern.findall(path.read_text(encoding="utf-8-sig")))
    return names


def _replayed_files(fuzz_dir: Path) -> set[str]:
    """Collect the corpus file names a replay suite names as a string literal.

    A corpus file that no replay test names is committed NumPy output that nothing compares against
    NumSharp; counting it as evidence would claim a proof that never runs.

    :param fuzz_dir: the ``test/NumSharp.Tests.Oracle/Fuzz`` directory.
    :returns: every ``"<name>.jsonl"`` literal in FuzzCorpusTests*.cs and IndexOracleTests.cs.
    """
    literal = re.compile(r'"([A-Za-z0-9_.]+\.jsonl)"')
    names: set[str] = set()
    for path in sorted(list(fuzz_dir.glob("FuzzCorpusTests*.cs")) + list(fuzz_dir.glob("IndexOracleTests.cs"))):
        names.update(literal.findall(path.read_text(encoding="utf-8-sig")))
    return names


def _case_dtypes(case: Mapping[str, Any]) -> set[str]:
    """Dtype names a fuzz/index contract exercises: operand dtypes, the expected dtype, the case dtype.

    :param case: one parsed corpus row.
    :returns: the dtype-name set (``char``/``decimal`` included where the corpus relabels them).
    """
    dtypes = {operand.get("dtype") for operand in case.get("operands") or [] if operand.get("dtype")}
    expected = case.get("expected") or {}
    if expected.get("dtype"):
        dtypes.add(expected["dtype"])
    if case.get("dtype"):
        dtypes.add(case["dtype"])
    return dtypes


def _case_is_error(case: Mapping[str, Any]) -> bool:
    """Whether a fuzz/index contract expects NumPy to raise.

    :param case: one parsed corpus row (common schema: ``error``/``expects_throw``; index schema:
        ``np.ok == false``).
    :returns: True for an error-message contract.
    """
    if case.get("error") or case.get("expects_throw"):
        return True
    outcome = case.get("np")
    return isinstance(outcome, dict) and outcome.get("ok") is False


def _group(groups: dict[tuple, ContractGroup], source: str, file: str, key: str, variant: str | None,
           pin: str | None) -> ContractGroup:
    """Get-or-create the group for one resolution tuple.

    :param groups: the accumulating group table, keyed by (source, file, key, variant).
    :param source: source id.
    :param file: corpus file name.
    :param key: op key / family variant.
    :param variant: param value for param-driven keys, else ``None``.
    :param pin: the file's pin class.
    :returns: the (possibly new) group.
    """
    ident = (source, file, key, variant)
    group = groups.get(ident)
    if group is None:
        group = groups[ident] = ContractGroup(source, file, key, variant, pin)
    return group


def _scan_fuzz(root: Path, oracle_map: Mapping[str, Any], groups: dict[tuple, ContractGroup],
               files: list[dict[str, Any]], problems: list[str]) -> None:
    """Scan the differential-fuzz corpus into groups.

    Host-pin metadata files (``*.host.jsonl``) record which BLAS build a pinned tier expects; they are
    not contracts and are skipped, as the replay and the Tests & Oracle inventory skip them.

    :param root: repository root.
    :param oracle_map: the parsed oracle map (param keys, files without op, host pins).
    :param groups: group table to fill.
    :param files: per-file stats list to append to.
    :param problems: problem list to append to.
    """
    fuzz_dir = root / "test" / "NumSharp.Tests.Oracle" / "Fuzz"
    corpus = fuzz_dir / "corpus"
    params = dict(_param_keys(oracle_map))
    without_op = _entries(oracle_map["fuzz"]["files_without_op"])
    declared_pins = _entries(oracle_map["host_pins"])
    host_libm = _host_libm_files(fuzz_dir)
    replayed = _replayed_files(fuzz_dir)
    regression_glob = '"regressions"' in "".join(
        path.read_text(encoding="utf-8-sig") for path in sorted(fuzz_dir.glob("FuzzCorpusTests*.cs")))

    for name, entry in sorted(declared_pins.items()):
        if entry.get("class") not in PIN_CLASSES:
            problems.append(f"oracle_map host_pins['{name}'] has unknown class {entry.get('class')!r}.")
        if not (corpus / name).is_file():
            problems.append(f"oracle_map host_pins['{name}'] names a corpus file that does not exist.")
        if name in host_libm:
            problems.append(f"oracle_map host_pins['{name}'] is redundant: the replay already runs it "
                            "through RunHostLibmCorpus.")
    for name in sorted(without_op):
        if not (corpus / name).is_file():
            problems.append(f"oracle_map fuzz.files_without_op['{name}'] names a corpus file that does not exist.")

    paths = sorted(corpus.glob("*.jsonl"), key=lambda p: p.name)
    # FuzzCorpusTests replays every regressions/*.jsonl by directory glob, so a pinned shrunk repro is
    # evidence too, but only while that glob replay exists.
    regressions = sorted((corpus / "regressions").glob("*.jsonl"), key=lambda p: p.name)
    for path in paths + (regressions if regression_glob else []):
        name = path.name if path.parent == corpus else f"regressions/{path.name}"
        if name.endswith(".host.jsonl"):
            continue
        if path.parent == corpus and name not in replayed:
            problems.append(f"Corpus file {name} is not named by any replay suite "
                            "(FuzzCorpusTests*.cs / IndexOracleTests.cs); its contracts prove nothing.")
            continue
        pin = "host-libm" if name in host_libm else (declared_pins.get(name) or {}).get("class")
        stats = {"source": "fuzz", "name": name, "contracts": 0, "error_contracts": 0, "pin": pin, "keys": set()}
        with path.open(encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                case = json.loads(line)
                op = case.get("op") or without_op.get(name)
                if not op:
                    problems.append(f"{name}:{line_no} has no op key and no fuzz.files_without_op entry.")
                    continue
                variant = None
                if op in params:
                    value = (case.get("params") or {}).get(params[op])
                    variant = None if value is None else str(value)
                    if variant is None:
                        problems.append(f"{name}:{line_no} op '{op}' lacks params.{params[op]}, which its "
                                        "oracle_map entry resolves by.")
                        continue
                group = _group(groups, "fuzz", name, op, variant, pin)
                group.cases += 1
                error = _case_is_error(case)
                group.errors += int(error)
                group.dtypes.update(_case_dtypes(case))
                layout = case.get("layout") or case.get("base")
                if layout:
                    group.layouts.add(str(layout))
                stats["contracts"] += 1
                stats["error_contracts"] += int(error)
                stats["keys"].add(op)
        stats["keys"] = len(stats["keys"])
        files.append(stats)


def _scan_npy(path: Path, groups: dict[tuple, ContractGroup], files: list[dict[str, Any]]) -> None:
    """Scan the .npy/.npz format oracle's manifest into groups.

    Variant = case kind (npy/header/raw/npz/sequence), suffixed ``:compressed`` for savez_compressed
    archives and ``:load_npy`` for the typed-reader cases, matching the map's npy section.

    :param path: the ``npy_oracle.zip`` archive.
    :param groups: group table to fill.
    :param files: per-file stats list to append to.
    """
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    stats = {"source": "npy", "name": path.name, "contracts": 0, "error_contracts": 0, "pin": None, "keys": set()}
    for case in manifest.get("cases", []):
        variant = case["kind"]
        if case["kind"] == "npz" and case.get("compressed"):
            variant += ":compressed"
        if case.get("load_via") == "load_npy":
            variant += ":load_npy"
        group = _group(groups, "npy", path.name, variant, None, None)
        group.cases += 1
        error = case.get("load_error") is not None
        group.errors += int(error)
        if case.get("np_dtype"):
            group.dtypes.add(case["np_dtype"])
        stats["contracts"] += 1
        stats["error_contracts"] += int(error)
        stats["keys"].add(variant)
    stats["keys"] = len(stats["keys"])
    files.append(stats)


def _scan_jsonl_oracle(path: Path, source: str, variant_of: Callable[[Mapping[str, Any]], str],
                       groups: dict[tuple, ContractGroup], files: list[dict[str, Any]],
                       layout_of: Callable[[Mapping[str, Any]], str | None] | None = None) -> None:
    """Scan a line-per-case specialized oracle (flags / layout) into groups.

    :param path: the corpus file.
    :param source: source id (``flags`` / ``layout``).
    :param variant_of: maps a case to its variant string, which must be a key of the map's section.
    :param groups: group table to fill.
    :param files: per-file stats list to append to.
    :param layout_of: maps a case to its layout label; ``None`` skips layout accounting (the layout
        oracle's recipes describe RESULT layouts, not the input layouts the other sources count).
    """
    stats = {"source": source, "name": path.name, "contracts": 0, "error_contracts": 0, "pin": None, "keys": set()}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            case = json.loads(line)
            variant = variant_of(case)
            group = _group(groups, source, path.name, variant, None, None)
            group.cases += 1
            error = case.get("err") is not None
            group.errors += int(error)
            if case.get("dtype"):
                group.dtypes.add(case["dtype"])
            if layout_of:
                label = layout_of(case)
                if label:
                    group.layouts.add(label)
            stats["contracts"] += 1
            stats["error_contracts"] += int(error)
            stats["keys"].add(variant)
    stats["keys"] = len(stats["keys"])
    files.append(stats)


def _layout_variant(case: Mapping[str, Any]) -> str:
    """Map a layout-parity case onto the map's layout section key.

    ``reduce`` keys are named by their leading reduction (``sum_flat`` -> ``reduce:sum``,
    ``percentile50`` -> ``reduce:percentile``); ``copyk`` keys by their trailing verb
    (``t3d_astype`` -> ``copyk:astype``); every other family is its own key.

    :param case: one parsed layout-oracle row.
    :returns: the variant string.
    """
    family = case["fam"]
    if family == "reduce":
        return "reduce:" + re.match(r"[a-z]+", case["key"]).group(0)
    if family == "copyk":
        return "copyk:" + case["key"].rsplit("_", 1)[1]
    return family


@functools.lru_cache(maxsize=4)
def _load_sources_cached(root: str, map_text: str) -> OracleSources:
    """Memoized body of :func:`load_sources` (the scan parses about 250,000 rows, about 1 s).

    The generator needs the scan twice per run (the row join and the artifact's source summary), and
    tests build several catalogs over the same corpus. The cache key is the canonical map text, not
    the map object: an edited map (tests mutate copies) must re-scan, because it decides which keys
    group per parameter value.

    :param root: repository root as a string (hashable).
    :param map_text: the canonical JSON of the map (hashable), so the scan sees the exact map.
    :returns: the grouped sources. Callers must not mutate it (it is shared through the cache).
    """
    oracle_map = json.loads(map_text)
    root_path = Path(root)
    groups: dict[tuple, ContractGroup] = {}
    files: list[dict[str, Any]] = []
    problems: list[str] = []
    _scan_fuzz(root_path, oracle_map, groups, files, problems)
    npy = root_path / NPY_ORACLE.relative_to(ROOT)
    flags = root_path / FLAGS_ORACLE.relative_to(ROOT)
    layout = root_path / LAYOUT_ORACLE.relative_to(ROOT)
    for required in (npy, flags, layout):
        if not required.is_file():
            problems.append(f"Oracle source {required.relative_to(root_path).as_posix()} is missing.")
    if npy.is_file():
        _scan_npy(npy, groups, files)
    if flags.is_file():
        _scan_jsonl_oracle(flags, "flags", lambda case: "setflags" if case.get("ops") else "read",
                           groups, files, layout_of=lambda case: case.get("recipe"))
    if layout.is_file():
        _scan_jsonl_oracle(layout, "layout", _layout_variant, groups, files)
    ordered = [groups[key] for key in sorted(groups, key=lambda k: (k[0], k[1], k[2], k[3] or ""))]
    return OracleSources(ordered, files, problems)


def load_sources(oracle_map: Mapping[str, Any], root: Path = ROOT) -> OracleSources:
    """Scan all four oracle sources into contract groups.

    :param oracle_map: the parsed oracle map (decides which fuzz keys group per parameter value).
    :param root: repository root; tests point it at a synthetic tree.
    :returns: grouped contracts, per-file stats and scan problems. Memoized per (root, map); treat the
        result as read-only.
    """
    return _load_sources_cached(str(root), json.dumps(oracle_map, sort_keys=True))


def _first_existing(templates: Iterable[str], catalog: Mapping[str, str], **values: str) -> str | None:
    """Return the first formatted template naming a row an automatic rule may target.

    :param templates: id templates in precedence order.
    :param catalog: row id -> kind.
    :param values: template substitutions (``key``/``rest``).
    :returns: the first id present in the catalog with a kind outside :data:`AUTO_EXCLUDED_KINDS`, or
        ``None`` when no template matches.
    """
    for template in templates:
        candidate = template.format(**values)
        kind = catalog.get(candidate)
        if kind is not None and kind not in AUTO_EXCLUDED_KINDS:
            return candidate
    return None


def auto_resolve(op: str, catalog: Mapping[str, str], oracle_map: Mapping[str, Any]) -> list[str]:
    """Resolve a fuzz op key by the automatic namespace rules alone (no explicit entry consulted).

    Bare keys walk ``fuzz.bare_key_precedence`` (np.* first, so np.trace shadows np.linalg.trace);
    dotted keys ``<head>.<rest>`` walk ``fuzz.prefix_rules[head]``.

    :param op: the corpus op key.
    :param catalog: row id -> kind.
    :param oracle_map: the parsed oracle map.
    :returns: a one-element id list, or ``[]`` when no rule resolves the key.
    """
    fuzz = oracle_map["fuzz"]
    if "." in op:
        head, rest = op.split(".", 1)
        templates = _entries(fuzz["prefix_rules"]).get(head)
        found = _first_existing(templates or [], catalog, rest=rest)
    else:
        found = _first_existing(fuzz["bare_key_precedence"]["templates"], catalog, key=op)
    return [found] if found else []


def numpy_object(np: Any, row_id: str) -> Any:
    """Resolve a ``numpy.*`` catalog id to the live NumPy object it names.

    :param np: the imported numpy module.
    :param row_id: a catalog id such as ``numpy.acos`` or ``numpy.random.Generator.normal``.
    :returns: the object, or ``None`` for a ``numsharp.*`` id or a path NumPy does not resolve (a
        ``None`` never compares identical to anything in :func:`join`'s identity rule).
    """
    if not row_id.startswith("numpy."):
        return None
    obj: Any = np
    for part in row_id.split(".")[1:]:
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def identity_predicate(np: Any) -> Callable[[str, str], bool]:
    """Build the NumPy object-identity predicate :func:`join`'s identity pass consumes.

    :param np: the imported numpy module.
    :returns: ``identical(a, b)``: True when both ids resolve and name the very same NumPy object.
        Resolutions are memoized per id (the pass asks about the same lenders repeatedly).
    """
    cache: dict[str, Any] = {}

    def resolve(row_id: str) -> Any:
        if row_id not in cache:
            cache[row_id] = numpy_object(np, row_id)
        return cache[row_id]

    def identical(a: str, b: str) -> bool:
        left = resolve(a)
        return left is not None and left is resolve(b)

    return identical


def join(rows: Iterable[Mapping[str, Any]], sources: OracleSources, oracle_map: Mapping[str, Any],
         identical: Callable[[str, str], bool] | None = None) -> OracleJoin:
    """Resolve every contract group to catalog rows, then apply identity and reviewed aliases.

    Resolution order per group: an explicit ``fuzz.keys`` entry (param-driven entries format their id
    templates with the group's variant, or take a per-value override), else the automatic rules; the
    specialized sources resolve through their map section by variant.

    Two alias passes then extend evidence to rows the corpus does not gate by name:

    * **identity** (mechanical): row A shares its ``numsharp_target`` with a row B that has direct
      contracts, AND ``identical(A, B)`` says NumPy's two names are the same object (``np.acos is
      np.arccos``). Both sides are then literally one implementation, so B's contracts prove A.
    * **reviewed** (``oracle_map.json`` aliases): a pure delegation a human checked in both libraries.
      A reviewed entry that the identity rule already derives is flagged as redundant.

    Every inconsistency becomes a problem string rather than an exception, so a single run reports all
    of them at once: unresolved keys, unknown ids, unused/redundant explicit entries, unused variant
    overrides and map sections, aliases on unknown rows, aliases whose target has no direct contracts,
    aliases made stale by direct fuzz contracts, and reviewed aliases the identity rule makes redundant.

    :param rows: catalog rows (the full, pre-projection set); ``id``, ``kind`` and ``numsharp_target``
        are read.
    :param sources: the scanned oracle sources.
    :param oracle_map: the parsed oracle map.
    :param identical: NumPy object-identity predicate over two row ids (see :func:`numpy_object`);
        ``None`` disables the identity pass (tests of the direct join, catalogs without NumPy).
    :returns: per-row evidence, a resolution table, and all problems (scan problems included).
    """
    rows = list(rows)
    catalog = {row["id"]: row["kind"] for row in rows}
    problems = list(sources.problems)
    evidence: dict[str, RowEvidence] = {}
    resolutions: dict[str, dict[str, Any]] = {}
    explicit = _entries(oracle_map["fuzz"]["keys"])
    used_explicit: set[str] = set()
    used_values: set[tuple[str, str]] = set()
    used_sections: dict[str, set[str]] = {"npy": set(), "flags": set(), "layout": set()}

    for group in sources.groups:
        rule = "explicit"
        if group.source == "fuzz":
            entry = explicit.get(group.key)
            if entry is not None:
                used_explicit.add(group.key)
                if "param" in entry:
                    override = (entry.get("values") or {}).get(group.variant)
                    if override is not None:
                        used_values.add((group.key, group.variant))
                        ids = list(override)
                    else:
                        ids = [template.format(value=group.variant) for template in entry["ids"]]
                else:
                    ids = list(entry["ids"])
            else:
                rule = "auto"
                ids = auto_resolve(group.key, catalog, oracle_map)
                if not ids:
                    problems.append(
                        f"Oracle op key '{group.key}' ({group.file}) resolves to no catalog row: add a "
                        "fuzz.keys entry to coverage/oracle_map.json (or fix the namespace rules).")
                    continue
        else:
            section = _entries(oracle_map[group.source])
            key = group.key
            if key not in section and group.source == "layout" and key.startswith("reduce:"):
                # One template covers every reduction head; record the template as used.
                ids = [template.format(head=key.split(":", 1)[1]) for template in section.get("reduce:{head}", [])]
                used_sections["layout"].add("reduce:{head}")
            else:
                ids = list(section.get(key, []))
                if key in section:
                    used_sections[group.source].add(key)
            if not ids:
                problems.append(f"Oracle {group.source} variant '{key}' ({group.file}) has no entry in "
                                f"coverage/oracle_map.json's '{group.source}' section.")
                continue

        unknown = [row_id for row_id in ids if row_id not in catalog]
        if unknown:
            problems.append(f"Oracle key '{group.label}' maps to ids missing from the catalog: {', '.join(unknown)}.")
            continue
        for row_id in ids:
            evidence.setdefault(row_id, RowEvidence()).add(group)
        resolution = resolutions.setdefault(group.label, {
            "label": group.label, "source": group.source, "ids": ids, "rule": rule, "cases": 0,
        })
        resolution["cases"] += group.cases

    # Explicit entries must earn their keep: unused ones rot silently (the op was renamed or deleted),
    # and one that repeats what the automatic rules resolve hides future rule changes behind a copy.
    for key, entry in sorted(explicit.items()):
        if key not in used_explicit:
            problems.append(f"oracle_map fuzz.keys['{key}'] matches no corpus contract (stale entry).")
            continue
        if "param" not in entry and auto_resolve(key, catalog, oracle_map) == list(entry["ids"]):
            problems.append(f"oracle_map fuzz.keys['{key}'] repeats the automatic resolution "
                            f"({entry['ids'][0]}); delete the entry.")
        for value in sorted((entry.get("values") or {})):
            if (key, value) not in used_values:
                problems.append(f"oracle_map fuzz.keys['{key}'].values['{value}'] matches no corpus contract.")
    for source, used in used_sections.items():
        for key in sorted(_entries(oracle_map[source])):
            if key not in used:
                problems.append(f"oracle_map {source}['{key}'] matches no {source} oracle contract (stale entry).")

    # Identity pass. Only rows with direct contracts can lend evidence (no alias chains), and a row
    # with direct fuzz contracts never borrows (its own contracts already speak for it).
    identity_via: dict[str, str] = {}
    if identical is not None:
        lenders: dict[str, list[str]] = {}
        for row in rows:
            target = row.get("numsharp_target")
            lent = evidence.get(row["id"])
            if target and lent is not None and lent.contracts:
                lenders.setdefault(target, []).append(row["id"])
        for row in rows:
            target = row.get("numsharp_target")
            current = evidence.get(row["id"])
            if not target or target not in lenders or (current is not None and current.fuzz_contracts):
                continue
            candidates = [lender for lender in lenders[target]
                          if lender != row["id"] and identical(row["id"], lender)]
            if not candidates:
                continue
            # Deterministic choice when several identical rows lend: the best-evidenced, then by id.
            via = min(candidates, key=lambda lender: (-evidence[lender].contracts, lender))
            identity_via[row["id"]] = via
            lender = evidence[via]
            evidence.setdefault(row["id"], RowEvidence()).via = {
                "id": via, "contracts": lender.contracts, "rule": "identity",
                "reason": f"NumPy's {row['id']} is the same object as {via}, and both map to {target}.",
                "pinned": lender.pinned,
            }

    for row_id, entry in sorted(_entries(oracle_map["aliases"]).items()):
        via = entry.get("via")
        if row_id not in catalog:
            problems.append(f"oracle_map aliases['{row_id}'] names a row the catalog does not have.")
            continue
        if via not in evidence or not evidence[via].contracts:
            problems.append(f"oracle_map aliases['{row_id}'] points at {via}, which has no direct oracle contracts.")
            continue
        current = evidence.get(row_id)
        if current is not None and current.fuzz_contracts:
            problems.append(f"oracle_map aliases['{row_id}'] is stale: the row now has "
                            f"{current.fuzz_contracts} direct differential-fuzz contracts; delete the alias.")
            continue
        if identity_via.get(row_id) == via:
            problems.append(f"oracle_map aliases['{row_id}'] is redundant: the identity rule already derives "
                            f"it from {via}; delete the alias.")
            continue
        target = evidence[via]
        evidence.setdefault(row_id, RowEvidence()).via = {
            "id": via, "contracts": target.contracts, "rule": "reviewed",
            "reason": entry.get("reason", ""), "pinned": target.pinned,
        }

    ordered = [resolutions[label] for label in sorted(resolutions)]
    return OracleJoin(evidence, ordered, problems)


def attach(rows: Iterable[dict[str, Any]], result: OracleJoin) -> int:
    """Write each row's ``oracle`` record in place (rows without evidence are left untouched).

    Attaching before the dashboard projection is deliberate: ``dashboard_rows`` copies rows with
    ``{**row}``, so the record rides along to every emitted scope while resolution still saw the full
    catalog (including surfaces the dashboard hides).

    :param rows: the catalog rows to annotate.
    :param result: the join result.
    :returns: the number of rows that received a record.
    """
    count = 0
    for row in rows:
        evidence = result.evidence.get(row["id"])
        if evidence is not None:
            row["oracle"] = evidence.record()
            count += 1
    return count


def classify(row: Mapping[str, Any]) -> str:
    """Classify a row's parity evidence; the ONE definition summaries and the dashboard both mirror.

    * ``verified-direct`` -- credited row with direct oracle contracts.
    * ``verified-alias`` -- credited row verified through a reviewed delegation alias only.
    * ``uncredited`` -- oracle contracts exercise it, but the catalog does not credit NumSharp with the
      API (missing/unsupported): the corpus proves behaviour the name-match cannot see; a finding.
    * ``unverified`` -- available/partial NumPy API with no oracle contracts.
    * ``none`` -- anything else (missing without contracts, types/constants, NumSharp-only extras).

    :param row: a catalog row, possibly carrying an ``oracle`` record.
    :returns: one of the five labels above.
    """
    record = row.get("oracle")
    credited = row.get("status") in CREDITED_STATUSES
    if record:
        if not credited:
            return "uncredited"
        return "verified-direct" if record.get("status") == "direct" else "verified-alias"
    if row.get("origin") == "numpy" and row.get("status") in {"available", "partial"}:
        return "unverified"
    return "none"


def describe(sources: OracleSources, rows: Iterable[Mapping[str, Any]], numpy_version: str) -> dict[str, Any]:
    """Summarize the oracle sources and the join outcome for the artifact's top-level ``oracle`` block.

    :param sources: the scanned sources.
    :param rows: the emitted (projected) rows, read for per-row classification counts.
    :param numpy_version: the pinned NumPy version the corpora were generated with.
    :returns: a deterministic dict: per-source totals, pinned files, and row classification counts.
    """
    rows = list(rows)   # read twice below; a generator would silently count zero the second time
    per_source: dict[str, dict[str, Any]] = {}
    for spec in SOURCES:
        files = [item for item in sources.files if item["source"] == spec.id]
        keys = {group.label for group in sources.groups if group.source == spec.id}
        per_source[spec.id] = {
            "label": spec.label,
            "path": spec.path,
            "replayed_by": spec.replayed_by,
            "files": len(files),
            "contracts": sum(item["contracts"] for item in files),
            "error_contracts": sum(item["error_contracts"] for item in files),
            "pinned_contracts": sum(item["contracts"] for item in files if item["pin"]),
            "keys": len(keys),
        }
    pinned_files = {item["name"]: item["pin"] for item in sources.files if item["pin"]}
    classes: dict[str, int] = {"verified-direct": 0, "verified-alias": 0, "uncredited": 0, "unverified": 0}
    for row in rows:
        label = classify(row)
        if label in classes:
            classes[label] += 1
    return {
        "numpy_version": numpy_version,
        "contracts": sum(item["contracts"] for item in per_source.values()),
        "error_contracts": sum(item["error_contracts"] for item in per_source.values()),
        "sources": per_source,
        "pinned_files": dict(sorted(pinned_files.items())),
        "rows": {"with_evidence": sum(1 for row in rows if row.get("oracle")), **classes},
    }
