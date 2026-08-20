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

## Generate or verify

```bash
python -m pip install numpy==2.4.2
python coverage/generate_coverage.py
python coverage/generate_coverage.py --check
python coverage/audit_documentation.py
```

Generated, reviewable outputs live in `coverage/generated/`:

- `coverage.json` — complete machine-readable inventory used by the documentation dashboard.
- `coverage.csv` — flat data for spreadsheets and downstream tooling.
- `summary.md` — human-readable totals and the highest-priority gaps.
- `manifest.json` — schema, tool versions, scope, and counting rules.

CI generates a fresh copy under `artifacts/numpy-numsharp-coverage/`, validates every headline-scope link against NumPy's official latest-stable Sphinx inventory, compares the result byte-for-byte with the checked-in dashboard data, and uploads the fresh directory as the `numpy-numsharp-api-coverage` artifact.

## What the numbers mean

The default denominator includes NumPy top-level callables, `ndarray` methods and properties, and callables from `numpy.random`, `numpy.linalg`, and `numpy.fft`. NumPy types, constants, and modules are catalogued but do not affect the headline percentage. NumSharp-only APIs are catalogued separately and also do not affect it.

Platform-conditional extended-precision aliases (`float96`, `float128`, `complex192`, and `complex256`) are excluded so the artifact is byte-identical across Windows and Linux. NumPy's portable `longdouble` and `clongdouble` names remain catalogued.

- **Exact** — the corresponding NumSharp surface has the same public member name.
- **Alias** — a reviewed or mechanically safe C# equivalent exists under another name or surface.
- **Partial** — an API exists, but the reviewed mapping has a known semantic limitation.
- **Unsupported** — a public compatibility symbol exists but does not implement the NumPy capability.
- **Missing** — no NumSharp public API mapping was found.
- **NumSharp-only** — an unmatched public member declared by `np`, `NDArray`, or `NumPyRandom`; these rows link directly to their declaration on GitHub.

API availability is not a blanket behavioral-parity claim. Exact edge-case, dtype, layout, and signature parity still requires differential tests. Record reviewed exceptions and cross-surface aliases in `coverage/overrides.json`; the generator validates every referenced NumSharp target.
