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

Extended namespaces cover complex-domain math, polynomials, masked arrays, legacy and modern strings, records, and object contracts. Each carries `extended=true`, `in_default_scope=false`, and a disposition. `summary.api_scope` reports the expanded counts and categories; `summary.default_scope` preserves the historical comparison.

The final `dashboard_rows` projection in `generate_coverage.py` applies the dashboard's reviewed scope **after** resolving support against original owners. It removes the ignored interop, library, testing, ctypes, and namespace-introspection surfaces (including root exports) from emitted JSON/CSV and all summary counts. The underlying NumPy extraction remains comprehensive so inventory regression checks can detect missing discovery. Nothing about ignored surfaces is displayed in the dashboard. MaskedArray members share the `ma` surface and **Masked arrays** category; legacy polynomial functions, `poly1d`, and the entire polynomial package share the `polynomial` surface and **Polynomials** category. Canonical IDs, signatures, implementation targets, and documentation links remain intact.

`object_surfaces.py` discovers methods, properties, selected protocols, and instance-only attributes from actual pinned NumPy classes/objects. It includes ufuncs, modern and legacy RNGs, seed sequences and bit generators, polynomial classes, masked/record/string/matrix arrays, iterators, dtype/limits/flags, archives, memory maps, and other public object contracts. The five ufunc methods (`reduce`, `accumulate`, `reduceat`, `outer`, `at`) are also catalogued for each top-level ufunc export. These describe exposed protocols: unary reductions and some generalized-ufunc methods reject calls in NumPy, as recorded in their notes. Aliases are separate exported names, and inherited members are separate contracts on each owner type.

The reflection inventory's schema v5 `objectTypes` map contains public members **including inherited members**, with declaring-type evidence. Object members match only their explicitly mapped CLR owner, using exact spelling or a reviewed adapter in `object_surfaces.py`. A top-level `np.sqrt`, `np.sum`, or an ordinary NDArray member cannot confer support on `emath.sqrt`, `add.reduce`, or `MaskedArray.sum`. A class being available does not mark its methods available. Missing methods, signatures, source links, and reviewed spelling differences are visible individually.

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
- `summary.md` — human-readable totals and the highest-priority gaps.
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

A NumPy **class** export (for example `numpy.random.Generator`, `PCG64`, `SeedSequence`, `BitGenerator`, `MT19937`) is credited when NumSharp exports a public type of the same name — a `"type"` availability match against the inventory's `exportedTypes` index. This carries the type existence the member-name index cannot (a class such as `BitGenerator` has no public members of its own), and, being out of default scope, it fixes the Types catalog without moving the headline.

Platform-conditional extended-precision aliases (`float96`, `float128`, `complex192`, and `complex256`) are excluded so the artifact is byte-identical across Windows and Linux. NumPy's portable `longdouble` and `clongdouble` names remain catalogued.

- **Exact** — the corresponding NumSharp surface has the same public member name.
- **Alias** — a reviewed or mechanically safe C# equivalent exists under another name or surface.
- **Partial** — an API exists, but the reviewed mapping has a known semantic limitation.
- **Unsupported** — a public compatibility symbol exists but does not implement the NumPy capability.
- **Missing** — no NumSharp public API mapping was found.
- **NumSharp-only** — an unmatched public member declared by `np`, `NDArray`, or `NumPyRandom`; these rows link directly to their declaration on GitHub.

API availability is not a blanket behavioral-parity claim. Exact edge-case, dtype, layout, and signature parity still requires differential tests. Record reviewed exceptions and cross-surface aliases in `coverage/overrides.json`; the generator validates every referenced NumSharp target.
