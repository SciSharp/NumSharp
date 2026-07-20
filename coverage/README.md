# NumPy ↔ NumSharp API coverage

This directory is the reproducible source for NumSharp's public API coverage artifact. It compares the public exports of pinned NumPy **2.4.2** with the public surface of the compiled NumSharp assembly.

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

- **Exact** — the corresponding NumSharp surface has the same public member name.
- **Alias** — a reviewed or mechanically safe C# equivalent exists under another name or surface.
- **Partial** — an API exists, but the reviewed mapping has a known semantic limitation.
- **Unsupported** — a public compatibility symbol exists but does not implement the NumPy capability.
- **Missing** — no NumSharp public API mapping was found.
- **NumSharp-only** — an unmatched public member declared by `np`, `NDArray`, or `NumPyRandom`; these rows link directly to their declaration on GitHub.

API availability is not a blanket behavioral-parity claim. Exact edge-case, dtype, layout, and signature parity still requires differential tests. Record reviewed exceptions and cross-surface aliases in `coverage/overrides.json`; the generator validates every referenced NumSharp target.
