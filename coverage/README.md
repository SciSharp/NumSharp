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

The reflection inventory's schema v7 `objectTypes` map contains public members **including inherited members**, with declaring-type evidence, and each type's C# operators and conversions (`operators`: every overload with its operand types, result type and declaring type, since v6), which credit Python's operator protocols. Since v7 every method member, of a module facade or an object type, also carries `overloads`: each overload's parameters as data (name, type, default value and its kind, `params`, by-ref), its generic parameters, `[OverloadResolutionPriority]`, obsolete / hidden flags and declaring type, which signature parity compares (see "Signature parity"). Object members match only their explicitly mapped CLR owner, using exact spelling or a reviewed adapter in `object_surfaces.py`. A top-level `np.sqrt`, `np.sum`, or an ordinary NDArray member cannot confer support on `emath.sqrt`, `add.reduce`, or `MaskedArray.sum`. A class being available does not mark its methods available. Missing methods, signatures, source links, and reviewed spelling differences are visible individually.

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

## Signature parity

Availability is a NAME claim. `direct_target` credits a NumPy API when the annotated NumSharp module has a member of that name, and a reviewed alias in `overrides.json` credits it through another name. Until generator 1.15.0 nothing after that looked at a parameter: the inventory tool carried each overload only as display text (`numsharp_signatures`), which the artifact printed beside NumPy's signature without comparing the two. So a row read "available / exact" while the NumPy call, ported word for word, did not compile or bound another argument:

- `np.sum` had seven overloads and none takes `out=`, `initial=` or `where=`;
- `np.transpose` names its parameter `premute`, so `np.transpose(x, axes: p)` does not compile;
- `np.can_cast` takes `NPTypeCode` / `Type` in 15 of its 20 overloads and names NumPy's `from_` `from`;
- `numpy.acos` is credited to `np.arccos`, and `np.acos(x)` does not compile (14 headline names are credited that way).

A standalone comparison found them on 2026-10-17 (`tools/api-parameter-parity/`, `docs/plans/numpy-signature-diff.md` and `numpy-signature-marks.md`). Generator 1.15.0 brought a comparison of its own into the artifact (`signature_parity.py` over the inventory's overload data). That comparison still saw less than the standalone one, which grew into the API diff gap map (`docs/plans/API_DIFF_GAP.md`). Measured against the gap map, 1.15.0 missed:

- NumPy parameters a runtime signature hides behind `(*args, **kwargs)`, which the gap map reads from NumPy's `.pyi` stubs and numpydoc sections;
- the C# spellings that are not members of a module facade: constructors (`new NDArray(...)` is NumPy's `np.ndarray(...)`), extension methods, the typed `NDArray<T>` members, and `numpy.char.chararray`, which `object_surfaces.py` left as a subsystem instead of mapping it to `NDCharArray`;
- the compile probe: a NumPy-style call that does not compile (**B**), `null` refused where NumPy's default is None (**L**), an integer sequence refused where NumPy's stubs allow one (**Q**), and a dtype picked by a generic type argument (**T ◇**, `np.zeros<T>(shape)`).

Generator 1.16.0 therefore joins the gap map itself onto the rows: one engine decides the marks, and the artifact reports them per API.

**Where the records come from.**
- `signature_gap.py` runs the gap map's three tools on the same tree, in a temporary directory: `dump_modules.cs` (a reflection dump of the module facades and of every exported type, with constructors and extension methods; it builds `NumSharp.Core` Debug net10.0), `binding_probe.py` (about 7,300 NumPy-style calls, compiled at C# 12 and C# 14) and `signature_diff.py` (NumPy's runtime signatures completed from its stubs and docs, the marks, the CSV). It takes about 80 seconds and needs the .NET SDK the build needs. The tools are run, not copied, so the gap map and the artifact cannot drift apart.
- Each gap-map row is one C# spelling of one NumPy API. `signature_parity.attach` joins it to the catalog row of the API it spells: the group's NumPy path plus the name (`np` is `numpy`, both `ndarray` and `ndarray (NDArray<T>)` are `numpy.ndarray`, `np.ma` is `numpy.ma`), a `(constructor)` the class row, and a NumSharp-only spelling (`numpy_kind` absent) the NumSharp-only row of the same C# member.
- A row with no spelling of its own adopts the spellings of the row that compares the same C# member (for an inherited member, the member of its declaring type) against the same NumPy object: `numpy.acos` takes `np.arccos`'s spelling (`numpy.acos is numpy.arccos`), `numpy.matrix.fill` takes `ndarray.fill`'s. NumPy's `matrix.tolist` is another object than `ndarray.tolist`, so it adopts nothing.
- A credited row the gap map does not reach keeps 1.15.0's comparison (`source: inventory`, below). It was never compile-probed: its B / L / Q read `—`, and summary.md counts it as unprobed.

**The comparison** (both engines follow docs/plans/api-parameter-parity.md):
- **NumPy's side.** The gap map reads `inspect.signature` and completes it from the stubs and the numpydoc sections. The inventory's own comparison reads `inspect.signature`, except the documented first docstring line of an ndarray method (inspect folds its keywords into `**kwargs`: `(self, /, axis=None, dtype=None, out=None, **kwargs)` for `ndarray.sum`), the callable a numpy.ma ufunc wrapper forwards to (`ma.add` takes `np.add`'s keywords), and ndarray's documented method for the numpy.ma functions and MaskedArray methods that forward `(*args, **params)` to it (`ma.copy`, `ma.swapaxes`).
- **Matching** (section 1 of the plan). A positional-only NumPy parameter is matched by slot, whatever the C# name; a positional-or-keyword one by name, at NumPy's slot; a keyword-only one by name. `*args` needs a `params` array (other C# parameters at its slot are reported as `varargs`); keywords NumPy takes through `**kwargs` are not foreign, and a C# parameter named like NumPy's `**kwds` spells it.
- **Marks**, over ALL overloads, each with the gap map's symbol. **M** ●: a NumPy parameter no overload offers; a rename counts, reported as `axes (C#: premute)`, because NumPy's keyword does not compile. M ◦ (only parameters the plan excludes are missing) is not a difference. **X**: a C# parameter NumPy does not have. **O**: more than one overload (the owner's rule of 2026-10-17: one NumPy-shaped signature per function). **T** ●: a dtype taken as `NPTypeCode` / `Type` instead of `DType`; T ◇: a generic type argument picks the dtype. **B**: a NumPy-style call does not compile (● at C# 12 and 14, ◐ at C# 12 only). **L**: `null` refused for NumPy's None (●; ◐ at C# 12 only; ◦ only a `Shape` parameter refuses it and `default` binds). **Q**: an integer sequence refused (● / ◐). **N**: credited through another name. A row takes the strongest symbol of each mark over its spellings, and the largest overload count.
- **Closest overload**: the overload the fewest weighted edits from NumPy's, i.e. what the one surviving overload must become. It adds a positional parameter at another slot (`order`), a parameter C# requires where NumPy has a default (`required`), another default value (`default`), `*args` without a `params` array (`varargs`), and, informational only, a null default where NumPy has a value (`null_default`).
- **Exclusions** (section 3 of the plan): `like`, `subok`, a ufunc's `casting` / `order` / `signature`, the field `order` of the sort family and `einsum_path`'s `einsum_call` are listed under `excluded` and not counted.
- **Status**: `identical` (no mark and no hard closest-overload difference), `differs`, or `not-compared` with a `reason` (no spelling with a NumPy signature to compare; for the inventory's comparison, a module function credited to an NDArray instance method or the other way round, a credit to a property, NumPy stating no signature). A not-compared row can still carry the O / T of a NumSharp-only member, and N. Python protocols take no part unless NumSharp declares one under its own name (`NDArray.__getitem__`); nor do uncredited rows, constants and modules. A class takes part through its C# constructors, or when it is credited to a factory method.
- **Strictness.** The generator refuses to write the artifact ("Signature-parity join failed") when a credited call's target `member_index` cannot resolve or carries no overload data, when a gap-map spelling with a NumPy counterpart joins no catalog row, when it joins a row the catalog does not credit (the catalog would call missing what the gap map compiles), or when an exemption in `signature_parity.GAP_JOIN_EXEMPT` no longer matches anything. The one exemption is `np.random.MT19937._legacy_seeding`: NumPy's name is private, and the catalog lists public NumPy APIs only.

**Outputs.**
- Each credited call carries a `signature` record: `status`, `spelling`, `source` (`gap-map` or `inventory`), `marks` (the symbol per mark), `overloads`, `missing`, `excluded`, `foreign`, `typecode`, `generic` (the T ◇ overloads), `binding` / `null` / `sequence` (the probe's B / L / Q findings: the call and the compiler's reason), `order`, `required`, `default`, `varargs`, `null_default`, `closest` (the compared overload), `numpy` (the compared NumPy signature) and `spellings` (one entry per C# spelling, each with its own marks and details, `via` naming the row it was adopted from), or `reason`.
- Every summary scope adds `signature_identical` / `signature_differs` / `signature_not_compared`, the per-mark row counts (`signature_missing`, `signature_foreign`, `signature_overloaded`, `signature_typecode` T ●, `signature_generic` T ◇, `signature_typecode_any` either T, `signature_binding`, `signature_null`, `signature_sequence`, `signature_other_name`), the closest-overload counts, `signature_unprobed` (compared by the inventory's comparison only) and `signature_percent`, which shares `coverage_percent`'s denominator. A row can carry T ● and T ◇ at once, so the per-surface T column reads `signature_typecode_any`, never the sum of the two.
- `coverage.json` and `manifest.json` carry a top-level `signature_gap` block: the tool, the gap map's own totals, and the reconciliation (each count of the gap map's spellings beside the same count over the spellings the rows carry, and the spellings no NumPy row carries, with the reason).
- `coverage.csv` appends `signature_*` columns after the oracle ones (`signature_source`, `signature_symbols`, `signature_generic`, `signature_binding`, `signature_null`, `signature_sequence` and `signature_spellings` since 1.16.0).
- `summary.md` gains a "Signature parity" section: the marks per scope and per surface, the reconciliation with the gap map, every headline API whose signature differs with what differs (the probe's findings included), and the credited headline calls not compared.
- The dashboard shows a "NumPy-identical signatures" card (it filters the explorer to the differing calls), an explorer filter by class and by mark (B, L and Q included), a mark badge on each differing result and the record in the detail pane, with every C# spelling and the probe's findings; like the oracle counts, the page recounts the headline and refuses a mismatched artifact, and an artifact from before generator 1.15.0 hides every signature element.

At generator 1.16.0: headline **243 / 560** APIs have NumPy's exact signature (43.4%; 259 at 1.15.0, before the compile probe and the stubs); 276 credited calls differ (56 M, 50 X, 220 O, 15 T, 24 B, 69 L, 33 Q, 14 N), 2 are not compared and 4 compared calls are unprobed. Expanded scope: 634 / 4,844 (13.1%), 647 differ, 8 unprobed. Of the gap map's 1,206 spellings with a NumPy counterpart the rows carry 1,204, every count matching; the other two are `np.lib.npyio.NpzFile`'s constructor (`numpy.lib` is not shown, by the owner's rule) and the exempted `_legacy_seeding`. Two spellings of names NumPy keeps out of `numpy.random.__all__` and its docs (`get_bit_generator`, `set_bit_generator`) sit on the NumSharp-only rows of their members. Mapping `numpy.char.chararray` to `NDCharArray` credited 104 rows (90 exact, 14 alias): the expanded scope reads 1,625 / 4,844 available (33.5%; 1,522 before), and those rows have no oracle contracts yet.

The 8 unprobed rows are rows the gap map does not compare, for three reasons: an alias credit whose NumPy name has no C# spelling (`numpy.cumulative_sum`, `cumulative_prod`, `round`, `row_stack`: mark N already says the call does not compile); a member NumPy's class defines itself while the C# type inherits it (`numpy.char.chararray.argsort`, `numpy.matrix.tolist`: the gap map compares the inherited member under its base group only); and two `numpy.random.RandomState` methods the gap map has no group for (`seed`, whose `numpy.random.seed` is another function, and `tomaxint`, which `numpy.random` does not export).

Gates: `test_signature_parity.py` (31 tests: the comparison rules against NumPy 2.4.2's real signatures, the gap-map join on synthetic maps, and the committed `docs/plans/numpy-signature-diff.csv`, whose groups must map and whose probe findings must split; planted bugs: 20 of 20 in the comparison and 20 of 20 in the join fail it), `test_object_surfaces.py` (schema 7 lists every overload of every method, with its parameters) and the signature cases of `test_dashboard.cjs` (21 of 22 planted bugs fail them; the survivor removes an early return whose result the fall-through produces anyway).

## Generate or verify

```bash
python -m pip install numpy==2.4.2
python coverage/generate_coverage.py
python coverage/generate_coverage.py --check
python coverage/audit_documentation.py
python -m unittest discover -s coverage -p 'test_*.py'
node --test coverage/test_dashboard.cjs
```

The generator builds the reflection inventory tool and, for signature parity, runs the gap map's tools on the same tree (a `NumSharp.Core` Debug build and two compile-probe builds, about 80 seconds): it needs the .NET 10 SDK besides NumPy 2.4.2, as the docs workflow provides.

Generated, reviewable outputs live in gitignored `coverage/generated/`:

- `coverage.json` — complete machine-readable inventory used by the documentation dashboard.
- `coverage.csv` — flat data for spreadsheets and downstream tooling.
- `summary.md` — human-readable totals, the NumPy-oracle parity evidence, signature parity, and the highest-priority gaps.
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

API availability is not a blanket behavioral-parity claim. Exact edge-case, dtype and layout parity requires differential tests, and the **oracle-verified** counts report which APIs those tests actually cover (see "NumPy-oracle parity evidence" above); whether the call itself is NumPy's is reported apart, as **signature parity** (see "Signature parity" above). Record reviewed exceptions and cross-surface aliases in `coverage/overrides.json`; the generator validates every referenced NumSharp target. Record corpus-key mappings and oracle aliases in `coverage/oracle_map.json`; the generator validates those too.
