# NumPy ↔ NumSharp API coverage

This directory is the reproducible source for NumSharp's public API coverage artifact. It compares the public exports of pinned NumPy **2.4.2** with the public surface of the compiled NumSharp assembly.

The NumSharp surfaces are **discovered, not hardcoded**: `NumSharp.Tools.ApiInventory` reflects every public type in `NumSharp.Core` annotated with `[ModuleName("...")]` — `np` itself, `NDArray` (`"ndarray"`), and each function-namespace facade (`"np.random"` on `NumPyRandom`, `"np.fft"` on `FourierModule`, `"np.linalg"` on the nested `np.linalg` class). A new module facade joins the artifact by annotation alone; the generator fails loudly if a compared NumPy surface has no annotated host. Single-object DSL exports (`np.r_`, `np.s_`, `np.mgrid`, …) take no attribute — NumPy exports each as one object, so the property on `np` is already the whole coverage row.

Four scan-integrity guards make a silent miss structurally hard:

- **Unbacked surface** (generator) — a compared NumPy surface with no `[ModuleName]` host is a hard error.
- **Stray host** (generator) — the tool emits the full public surface *outside* the annotated modules, and any still-missing in-scope NumPy export whose name exists there fails the run, naming the candidate types. Reviewed name coincidences are recorded in `overrides.json` under `"stray_allowlist"`.
- **Facade shape** (tool) — a property on an annotated host returning a concrete class with many NumPy-style lowercase instance methods, or a public nested static class with lowercase static methods, must itself be annotated (the exact shapes `np.fft` and `np.linalg` were originally missed by).
- **Hierarchy** (tool) — annotated types are reflected `DeclaredOnly`, so their base must be `object` (or itself annotated); growing a base class fails instead of silently hiding inherited members.

Members are reflected with both `Static` and `Instance` flags (per-member `static` recorded), so a static helper on an instance facade cannot escape the scan. The generator also warns when an `overrides.json` alias goes stale because a direct match now exists.

**Matching is case-sensitive** — NumPy's public API is case-sensitive, so a NumSharp member is credited only when the spelling is identical (every match is an exact lookup). The generator additionally folds case to *detect* near-misses — an in-scope NumPy API left missing for which NumSharp exposes a same-surface member differing only by case — and reports them in `summary.md` under "Case-insensitive near-misses", per row via a `case_insensitive_matches` field (JSON/CSV), and on the console. They are **never** counted as available: this guards against silently satisfying NumPy's `histogram` with a C#-style `Histogram`. Close one by renaming to the exact NumPy spelling or recording a reviewed alias in `overrides.json`.

## Expanded API scope and object members

The existing DocFX dashboard defaults to **All NumPy APIs**: the five historical headline surfaces plus public extended submodules and object members. The scope selector also offers the historical headline, extended APIs, full NumPy catalog (including types/constants/modules), and NumSharp extensions. Every summary, surface card, capability category, and explorer result uses that selected scope. Types/constants/modules remain searchable but never enter an API availability denominator.

Extended namespaces cover complex-domain math, polynomials, masked arrays (with masked record arrays), legacy and modern strings, records, dtype classes, exceptions and warnings, static typing, version strings, Fortran bindings, ctypes interop, and object contracts: every public class's members and Python protocols, and the members of NumPy's scalar types. Each carries `extended=true`, `in_default_scope=false`, and a disposition. `summary.api_scope` reports the expanded counts and categories; `summary.default_scope` preserves the historical comparison.

The final `dashboard_rows` projection in `generate_coverage.py` applies the dashboard's reviewed scope **after** resolving support against original owners. It removes `numpy.lib` and `numpy.testing` (`IGNORED_DASHBOARD_ROOTS`, including root exports) from emitted JSON/CSV and all summary counts; by the owner's rule every other public NumPy API is shown, and `numpy.lib.NumpyVersion` is shown although `numpy.lib` is not (`IGNORED_DASHBOARD_EXCEPTIONS`), its class row and members grouped under the `lib.NumpyVersion` surface. Until generator 1.14.0 `numpy.ctypeslib`, `numpy.__array_namespace_info__` and a hand-picked list of ndarray interop protocols (under the invented ids `numpy.ndarray.interop.*`) were hidden too. The underlying NumPy extraction remains comprehensive so inventory regression checks can detect missing discovery. Nothing about ignored surfaces is displayed in the dashboard. MaskedArray members share the `ma` surface and **Masked arrays** category; legacy polynomial functions, `poly1d`, and the entire polynomial package share the `polynomial` surface and **Polynomials** category. Canonical IDs, signatures, implementation targets, and documentation links remain intact.

`object_surfaces.py` discovers methods, properties, Python protocols, and instance-only attributes from actual pinned NumPy classes/objects. It includes ufuncs, modern and legacy RNGs, seed sequences and bit generators, polynomial classes, masked/record/string/matrix arrays, iterators, dtype/limits/flags, archives, memory maps, and other public object contracts. Every public method of NumPy's ufunc type is also catalogued per ufunc export. The list is read from `numpy.ufunc` at the pin (`reduce`, `accumulate`, `reduceat`, `outer`, `at`, `resolve_dtypes`), never written by hand. The scan covers numpy itself and every namespace the generator compares (`UFUNC_PROTOCOL_NAMESPACES`), so ufuncs exported by `numpy.strings` and `numpy.char` get rows too. numpy.ma's masked ufunc wrappers get a row for each of those methods they implement: `ma.add` has `reduce`/`accumulate`/`outer`, the comparisons only `accumulate`/`outer` (NumPy sets their `reduce` to None), `ma.maximum`/`ma.minimum` only `reduce`/`outer`. These rows describe exposed protocols: unary reductions and most generalized-ufunc methods reject every call in NumPy (`applicability: not_applicable`, explained in the notes), while `resolve_dtypes` resolves for every ufunc shape. Aliases are separate exported names, and inherited members are separate contracts on each owner type.

The reflection inventory's schema v6 `objectTypes` map contains public members **including inherited members**, with declaring-type evidence, and each type's C# operators and conversions (`operators`: every overload with its operand types, result type and declaring type), which credit Python's operator protocols. Object members match only their explicitly mapped CLR owner, using exact spelling or a reviewed adapter in `object_surfaces.py`. A top-level `np.sqrt`, `np.sum`, or an ordinary NDArray member cannot confer support on `emath.sqrt`, `add.reduce`, or `MaskedArray.sum`. A class being available does not mark its methods available. Missing methods, signatures, source links, and reviewed spelling differences are visible individually.

## Catalog completeness

What counts as NumPy's public API is decided outside the generator, by two NumPy sources. `test_catalog.py` enforces both on every run:

- **NumPy's public-module list.** `numpy/tests/test_public_api.py` defines `PUBLIC_MODULES` and `PUBLIC_ALIASED_MODULES`; the test evaluates them for the running Python. Every listed module must be scanned (numpy itself, a `HEADLINE_SUBMODULES` entry or an `EXTENDED_SUBMODULES` family) or excluded with a reason in `EXCLUDED_PUBLIC_SUBMODULES`. The exclusions are `ma.extras` (numpy.ma re-exports its whole `__all__`), `lib.npyio` (its two names are object surfaces), the mypy plugin, and `distutils` (public only before Python 3.12). A NumPy release that adds a public module fails `test_every_public_numpy_module_is_scanned_or_excluded`.
- **NumPy's documentation index.** `numpy_documentation_inventory.json` holds every `numpy.*` object of the official Sphinx index. `test_every_documented_numpy_api_on_the_pin_has_a_row` requires a row for every documented name that resolves on the pin. The row can match exactly, through a documented alias, or as the same object catalogued under another spelling (`numpy.lib.scimath.sqrt` is `numpy.emath.sqrt`). Only Python's object machinery is exempt (NumPy documents `ndarray.__new__`, which the class row stands for), and the NumPy 1.x compatibility APIs the catalog leaves out (below). Python protocols and the members of NumPy's scalar types were exempt until generator 1.14.0, as open decisions; the owner decided they must be detected.

Within a scanned family the member list is the module's `__all__` (else its non-underscore `dir()`), plus every name the documentation index lists directly under the module. `numpy.exceptions.RankWarning`, for example, is documented but absent from `__all__`. A documented submodule is imported explicitly, so the list never depends on what the process imported before. A name inside a NumPy 1.x compatibility API (`COMPAT_EXCLUDED_APIS`) is left out, listed or documented.

Every public member becomes a row:
- **Callables** are API rows.
- **Classes, constants and modules** are supporting rows that stay outside the percentages (`numpy.ma.masked`, the `numpy.polynomial.<basis>` domain/zero/one/x constants, `numpy.dtypes.Float64DType`, `numpy.exceptions.AxisError`).
- **Each family's own module** gets a row when nothing else catalogued it (`numpy.version`, `numpy.testing.extbuild`).
- **A class an object surface owns** (`numpy.ma.MaskedArray`) is left to that surface.

Crediting follows two rules:
- A submodule member is credited by a member of the same name on that submodule's NumSharp facade.
- A real class with no such member is credited to a NumSharp type of the same simple name (`numpy.exceptions.AxisError` → `NumSharp.AxisError`), the rule headline classes use. A typing alias (`numpy.typing.NDArray`) is not a class, so a NumSharp type named `NDArray` never credits it.

### Python protocols, scalar types and the other classes

A class's Python protocols are public API: `a + b` is `ndarray.__add__`, `a[i]` is `__getitem__`, `with np.nditer(...)` is `__enter__`/`__exit__`. `numpy_protocols` reads them from the classes themselves: every special name a NumPy class (one whose `__module__` is numpy's) declares, except Python's object machinery (`PROTOCOL_MACHINERY`: construction, attribute access, `__del__`, the `__slotnames__` cache, Cython's pickling shims), Python 2 names (`PYTHON2_PROTOCOLS`: `numpy.lib.user_array.container` still defines `__oct__`/`__hex__`), and protocols set to None (`ndarray.__hash__ = None`: arrays are unhashable). Protocols a NumPy class inherits from Python or another package (`collections.abc.Mapping.__eq__` on `NpzFile`) are not NumPy's.

- **Object surfaces** list the protocols they have from NumPy classes, inherited ones included, like their other members (`numpy.matrix.__add__`). A new `ndarray` object surface lists ndarray's 84 protocols under their real ids (`numpy.ndarray.__array__`) and credits them against `NDArray`; its public members stay headline rows.
- **NumPy's scalar types** (`numpy.generic` and its 34 subclasses; `numpy.record` is an object surface) list each member once, at the class that defines it (`scalar_type_rows`): numpy.generic's 118 members once, then what each subclass adds or redefines (every concrete numeric type defines its own arithmetic slots). Listing inherited members on every class would repeat numpy.generic's members 34 times. Rows are named by NumPy's canonical type names (`CANONICAL_SCALAR_TYPES`: `numpy.byte`, `numpy.long`, `numpy.longlong`, ...), which denote one C type on every platform, because the sized names do not: `numpy.int32` is `numpy.long` on Windows and `numpy.intc` on Linux. They share the `generic` surface and the **Scalar types** category, and none is available: NumSharp has no NumPy scalar objects (a scalar is a C# primitive or a 0-d NDArray).
- **Every other public NumPy class** (`numpy.errstate`, `numpy.ma.mvoid`, `numpy.exceptions.AxisError`, `numpy.dtypes.StringDType`) lists its own members the same way (`class_member_rows`). An alias is one owner, a class that is not NumPy's own (`numpy.ctypeslib.c_intp`) and a typing alias have no member rows, and a member is credited against the NumSharp type of the class's simple name, case-sensitively (`NumSharp.AxisError.Axis` does not credit `AxisError.axis`).

A protocol is credited (`protocol_credit`) by a NumSharp member of its own name (`NDArray.__getitem__`), else by its C# spelling, always as an alias:
- **an operator of the same symbol** (`OPERATOR_PROTOCOLS`), declared with the owner type itself as the operand (a base type's operator returns the base class). A reflected protocol (`__radd__`) needs another type on the left, or an owner-only overload plus an implicit conversion to the owner (`5 << a` binds `operator <<(NDArray, NDArray)`): `poly1d` declares only `operator +(poly1d, poly1d)` and converts nothing to itself, so `2.0 + p` would bind NDArray's `+` through poly1d's conversion to its coefficients and `__radd__` stays missing. In-place protocols are never credited by an operator: C#'s `a += b` rebinds `a` instead of writing into it;
- **a cast** from the owner, or from the base type that declares it (`(int)m` uses NDArray's operator; `CONVERSION_PROTOCOLS`: `__int__`, `__float__`, `__complex__`, `__bool__`);
- **the indexer** with the needed accessor (`INDEXER_PROTOCOLS`);
- **a reviewed adapter** (`REVIEWED_PROTOCOL_ADAPTERS`) whose NumSharp documentation states that it ports the protocol: `NDArray.ToString()` is `str()` and `ToString(true)` is `repr()`, but `finfo.ToString()` renders the repr, so `finfo.__str__` stays missing; `foreach` is `__iter__` for the iterators; `using` is `__enter__`/`__exit__` for `nditer`. A reviewed adapter whose member disappeared fails the run.

A protocol NumPy fills only to refuse every call (`REFUSED_PROTOCOLS`: `del a[i]` on arrays, iterators, flags and void scalars, ordering comparisons on `ndarray.flags` and `StringDType`) keeps its row with `applicability: not_applicable`, as a rejected ufunc method does; `test_object_surfaces` calls every entry on a NumPy instance.

### NumPy 1.x compatibility APIs

APIs NumPy 2.4.2 still ships but has itself deprecated, kept only so NumPy 1.x code keeps working, are left out (`COMPAT_EXCLUDED_APIS`): `numpy.matlib` (PendingDeprecationWarning on import since 1.19; a scanned family in generator 1.13.0) and `numpy.typing.NBitBase` (deprecated in 2.3). NumSharp's port of such an API (`np.typing.NBitBase`, kept `[Obsolete]` like NumPy's) is NumPy's API, not a NumSharp-only extension, so it gets no row of either kind (`lists_as_numsharp_only`). Two guards hold every row to the pin:
- `test_every_catalogued_row_names_an_api_of_the_pinned_numpy` resolves every NumPy row on NumPy 2.4.2 (an instance-only member on its object surface's sample instance);
- `test_no_catalogued_row_is_an_api_numpy_deprecated` fails on a row that warns when reached or whose documentation deprecates the API as a whole (`deprecation_scope`: a `.. deprecated::` directive before the first section; one under a parameter, as for `numpy.fft.fft2`'s `s`, deprecates a use). The rows kept anyway are listed with reasons in `KEPT_DEPRECATED_APIS`: `numpy.random.random_integers` and `RandomState.random_integers` (deprecated since 1.11, headline rows NumSharp implements) and two hidden `numpy.testing` helpers.

Generator 1.14.0 added 1,904 shown rows this way (1,793 Python protocols, 158 of them provided; 923 scalar-type members; 26 members of other classes) and left out 10 compatibility rows; the expanded scope is 1,449 of 4,844 APIs (29.9%; 1,291 of 2,952 before), the headline unchanged.

Generator 1.13.0 introduced the public-module and documentation guards. The audit behind it found 462 documented names that resolve on NumPy 2.4.2 with no row: whole modules were never scanned (`dtypes`, `exceptions`, `typing`, `version`, `ma.mrecords`, `matlib`, `f2py`), the extended scan kept only callables, and four public classes had no object surface (`busdaycalendar`, `ma.mrecords.MaskedRecords`, `lib.Arrayterator`, `lib.user_array.container`). Cataloguing them also credited 66 NumSharp members that had read as NumSharp-only extensions: 27 `np.dtypes` classes, 8 `np.ma` objects, 24 polynomial constants and 7 polynomial submodules.

## NumPy-oracle parity evidence

Availability says NumSharp *exposes* an API; the committed NumPy oracles say it *behaves* like NumPy. `oracle_evidence.py` joins the two, row by row, so the artifact reports both. Four sources are joined, each replayed against NumSharp in CI and required to be bit-exact or a documented, registry-excused divergence:

| Source | Path | Replayed by |
|---|---|---|
| Differential-fuzz corpus | `test/NumSharp.Tests.Oracle/Fuzz/corpus/*.jsonl` | `FuzzMatrix` (FuzzCorpusTests*, IndexOracleTests) |
| `.npy`/`.npz` format oracle | `test/NumSharp.Tests/IO/corpus/npy_oracle.zip` | `NpyOracle` (NpyOracleTests) |
| Array flags oracle | `test/NumSharp.Tests/Backends/corpus/flags_oracle.jsonl` | FlagsOracleTests |
| Result-layout parity oracle | `test/NumSharp.Tests/Backends/corpus/layout_parity_oracle.jsonl` | LayoutParityOracleTests |

**Resolution.** Contracts are grouped by (source, file, key, variant) and each group resolves to catalog rows. A bare fuzz op key walks `np.<key>` → `np.linalg` → `np.fft` → `np.random` → a NumSharp-only `np` row, and the first existing callable/property wins, so `np.trace` shadows `np.linalg.trace`. The generator calls `np.trace`, and the Array-API form has a stricter contract. Module/class/constant rows are never automatic targets (`fft` resolves to `np.fft.fft`, not the module). Dotted keys resolve by namespace (`ndarray.`, `emath.`, `ma.`, the six polynomial bases, falling back to NumSharp-only rows such as `chebvalnd`). Everything else — variant suffixes (`std_ddof`), parameter-driven keys (`out_binary` + `params.ufunc`, `rnd` + `params.dist`, `grnd` + `params.method`), protocols (`get`/`set` attach to `numpy.ndarray`; since generator 1.14.0 a protocol key such as `matrix.__add__` or `ndarray.__len__` resolves to its protocol row through the namespace rules) and the specialized sources' families — is reviewed in `oracle_map.json`.

**Aliases.** A row the corpus does not gate by name can still be verified two ways. *Identity* is mechanical: the row is credited to the same NumSharp member as a gated row AND NumPy's two names are the same object (`np.acos is np.arccos`, `np.pow is np.power`). Both sides are then one implementation. *Reviewed* aliases in `oracle_map.json` cover pure delegations that are not the same object (`np.amax` → `np.max`, `np.row_stack` → `np.vstack`, `np.linalg.matmul` → `np.matmul`). They mirror the C# gate's `EquivalentAliases`/`NdarrayAliases`/`MaAliases`, and a test fails if one is missing. `np.linalg.trace`/`diagonal`/`outer`/`cross` are deliberately *not* aliased: their contracts differ from the `np.*` forms.

**Strictness.** The generator refuses to write an artifact while any of these holds (all reported at once):
- a corpus key resolves to nothing, or maps to an id the catalog lacks;
- an explicit entry is unused, or repeats the automatic rules;
- a value override or specialized-section entry is unused;
- an alias names an unknown row, points at a row without direct contracts, has gone stale (its row gained direct fuzz contracts), or duplicates the identity rule;
- a pin declaration is bad;
- a corpus file no replay suite names.

**Classification** (`oracle_evidence.classify`, mirrored by the dashboard's `parityClass`; the page recounts the headline and refuses a mismatched artifact):

- `verified-direct` / `verified-alias` — credited (available/partial) row with contracts. Both count as **oracle-verified**.
- `unverified` — available/partial NumPy API with no contracts: parity is declared, not proven. Some such APIs carry dedicated unit suites instead (see OracleSurfaceCoverageTests' sibling-owned and known-gap lists).
- `uncredited` — contracts exercise the API but the catalog calls it missing/unsupported: a crediting gap. None remain at generator 1.14.0; the last ones were `numpy.ma.MaskedArray.*`, credited once its object surface was mapped to its owner type `NDMaskedArray`.
- `none` — everything else.

**Outputs.**
- Each row with evidence carries an `oracle` record: `status`, `contracts`, `error_contracts`, per-`sources` counts, `keys`, `files`, `dtypes`, `layouts`, `pinned` (verified only on the pinned host), `pinned_contracts`, `pins`, and `via` for aliases.
- Every summary scope adds `oracle_verified`/`oracle_direct`/`oracle_alias`/`oracle_unverified`/`oracle_uncredited` and `oracle_percent`, which shares `coverage_percent`'s denominator.
- `coverage.csv` appends `oracle_*` columns after the historical ones.
- `coverage.json`/`manifest.json` carry a top-level `oracle` block (per-source totals, pinned files, classification counts).
- `summary.md` gains a "NumPy-oracle parity evidence" section, including the lists of available headline APIs without contracts and of oracle-gated APIs the catalog does not credit.

Host-pinned contracts (the `RunHostLibmCorpus` tiers, discovered from the replay source; the OpenBLAS and Windows-libm tiers declared under `host_pins`) are hard-gated only on the pinned host. A row whose contracts are *all* pinned is marked `pinned`.

## Generate or verify

```bash
python -m pip install numpy==2.4.2
python coverage/generate_coverage.py
python coverage/generate_coverage.py --check
python coverage/audit_documentation.py
python -m unittest discover -s coverage -p 'test_*.py'
node --test coverage/test_dashboard.cjs
```

Generated, reviewable outputs live in gitignored `coverage/generated/`:

- `coverage.json` — complete machine-readable inventory used by the documentation dashboard.
- `coverage.csv` — flat data for spreadsheets and downstream tooling.
- `summary.md` — human-readable totals, the NumPy-oracle parity evidence, and the highest-priority gaps.
- `manifest.json` — schema, tool versions, scope, and counting rules.

CI generates a fresh copy under `artifacts/numpy-numsharp-coverage/`, validates documentation links across **all catalog scopes**, and uploads it as `numpy-numsharp-api-coverage`. Master builds publish snapshots to the repository's `data` branch. Generated data is not committed on master.

DocFX reads **`refs/data/inventory/latest/`**, not `coverage/generated/`. To preview fresh data locally, generate directly into that directory and build the existing website:

```bash
python coverage/generate_coverage.py --output refs/data/inventory/latest
docfx docs/website-src/docfx.json
```

`tools/dashboard_data/refresh_data.py --type inventory` performs the repository's snapshot publication flow (pull and local commit on the data branch). It is separate from a local preview.

Documentation URLs resolve from `numpy_documentation_inventory.json`, a checked-in index of identifiers and URLs from NumPy's official Sphinx inventory. Exact API targets are preferred; undocumented standalone exports link to their nearest documented parent. No guessed generated-page URLs are emitted. The live audit validates both pages and anchors. Refresh the offline link index explicitly with `python coverage/audit_documentation.py --refresh-inventory`; this does not change the pinned runtime NumPy version.

## What the numbers mean

The historical headline denominator includes NumPy top-level callables, `ndarray` methods and properties, and callables from `numpy.random`, `numpy.linalg`, and `numpy.fft`. The dashboard's expanded denominator adds extended function and object-member contracts. NumPy types, constants, modules, and NumSharp-only APIs are catalogued separately and affect neither percentage.

A NumPy **class** export (for example `numpy.random.Generator`, `PCG64`, `SeedSequence`, `BitGenerator`, `MT19937`) is credited when NumSharp exports a public type of the same name — a `"type"` availability match against the inventory's `exportedTypes` index. This carries the type existence the member-name index cannot (a class such as `BitGenerator` has no public members of its own), and, being out of default scope, it fixes the Types catalog without moving the headline. Since generator 1.13.0 the same rule credits submodule classes (`numpy.exceptions.AxisError` → `NumSharp.AxisError`, `numpy.ma.MaskError` → `NumSharp.MaskError`) when the submodule's NumSharp facade has no member of that name; typing aliases are never matched (see "Catalog completeness").

Platform-conditional extended-precision aliases (`float96`, `float128`, `complex192`, and `complex256`) are excluded so the artifact is byte-identical across Windows and Linux. NumPy's portable `longdouble` and `clongdouble` names remain catalogued.

- **Exact** — the corresponding NumSharp surface has the same public member name.
- **Alias** — a reviewed or mechanically safe C# equivalent exists under another name or surface.
- **Partial** — an API exists, but the reviewed mapping has a known semantic limitation.
- **Unsupported** — a public compatibility symbol exists but does not implement the NumPy capability.
- **Missing** — no NumSharp public API mapping was found.
- **NumSharp-only** — an unmatched public member declared by `np`, `NDArray`, or `NumPyRandom`; these rows link directly to their declaration on GitHub.

API availability is not a blanket behavioral-parity claim. Exact edge-case, dtype, layout, and signature parity requires differential tests, and the **oracle-verified** counts report which APIs those tests actually cover (see "NumPy-oracle parity evidence" above). Record reviewed exceptions and cross-surface aliases in `coverage/overrides.json`; the generator validates every referenced NumSharp target. Record corpus-key mappings and oracle aliases in `coverage/oracle_map.json`; the generator validates those too.
