# Triage taxonomy — bucketing every commit, with worked examples

Every commit in the range goes in exactly one of five buckets. Only **UNHANDLED** produces a
changelog line. Examples are from the 0.70.0 `journey3` sweep.

---

## Bucket 1 — Cited

Its short hash already appears in `docs/releases/RELEASE_<version>.md`. Done. (Still read the body —
see `verify-accuracy.md` — because a cited line can be *wrong*.)

## Bucket 2 — Folded (dedup, CHANGELOG_STYLE §10)

The commit's substance belongs to a feature that's already listed — it's a follow-up `fix`, an
audit pass, an edge-case sweep, a perf tweak, or the test/oracle wiring for something introduced
**in this same range**. It gets **no line of its own**; if it carries weight, fold its hash onto the
feature's existing line (`— \`primary\` (+ \`followup\`)`).

- **§10.4 — never "fixed the thing I just added":** if both the `feat` and its `fix` are in-range,
  only the feat's line survives, describing the **fixed** behavior.
- Examples: `a7782984` (fill_diagonal edge-fix) folds into the diag-family feature `27b9b012`;
  `27632ed5` (canonicalize set-op NaN) folds into `bfe952d5` (isin/set-ops); the whole FFT
  parity-review series (`af88e04b`/`ebfe942a`/`240f5fbe`/`895fa77b`/`42114f4f`/`e58e4072`) folds
  into the `np.fft.*` line; `8cad3025`'s three engine `argmax` bug-fixes fold into the sort-six line
  **and** the argmax/argmin Parity line.
- **Test/bench/oracle wiring of an in-range feature is Folded, not its own line** (e.g. `82d8f15c`
  the kron fuzz gate). The *oracle infrastructure* itself can earn ONE deduplicated Tooling line
  (§11) — e.g. "the oracle gained the LAPACK/precision/FFT tiers" — but not one line per tier commit.

## Bucket 3 — Package-internal (§7A, no hashes)

The commit develops a shipped NuGet package (here: `NumSharp.Interop.OpenBLAS`,
`NumSharp.Interop.pythonnet`). Packages are documented by the **§7A New NuGet Packages** section,
which **carries no commit hashes** and folds the package's whole development history into a
capability elaboration. So these commits get no per-commit line — instead, **audit the §7A section**
and confirm it names the capability the commit adds, spelled as the real API.

- The ~50 pythonnet + OpenBLAS commits (add/rewrite/rename, GIL, codec modes, buffer exporters,
  delivery model, complex128 products, LAPACK factorisations, discovery tiers) are ALL bucket 3.
- A package `refactor!`/`feat!` (BLAS→OpenBLAS rename, `NDArrayInterop`→`NDArrayPythonInterop`) is
  **not** a user-facing breaking change if the package **never shipped in a release** — it's a name
  settled within the same dev cycle. Reflect the **final** name in §7A; add no Breaking line.
- **Cross-section pointer (§10.6):** functions that compute *only* through a package (the LAPACK
  factorisations, complex128 products) live in §7A; New APIs carries a short **pointer bullet with
  their hashes** (§7A omits hashes, so the pointer is where they land).

## Bucket 4 — Excluded (§11)

No line, ever:

- **Docs-only** — `docs(...)`, website/DocFX pages, skill bundles, `ARCHITECTURE.md`, README,
  design/handover markdown, doc-comment reframes.
- **Chore / rename / move** — file-case renames (`NdArray`→`NDArray`), `.git-blame-ignore-revs`
  entries, project/solution-folder renames, `chore(sln)`, tooling relocation, merges.
- **CI / signing / packaging plumbing** — unless it *removed a user-visible limit*.
- **Benchmark & oracle/test infrastructure** — harness upgrades, crash-fixes in the benchmark op
  matrix (Decimal overflow, OOM), new test projects, corpus-move commits, coverage-tool internals.
- **Internal refactor with no observable effect** — `fcb36637` (fold stride-perm onto `Shape`),
  `5aac05f4` (`NDAxisState` → `ref struct`), `1a7136f4` (move einsum contraction to the np layer —
  removes an engine-internal API nobody but `np.einsum` called).
- **Compile-time-only** — `e49ba49a` (lowercase ValueTuple field names; erase to `Item1/Item2` at
  runtime, no caller used the old names).
- **Exposing internal impl types as public** — `02fe0fb8` (pocketfft engine / `DecimalMath` /
  dtype helpers made public) — additive, not a NumPy-surface feature.
- **Examples/demo project** — the NN example (`199911e1`/`668fdd69`/`6230a867`); the project was
  even dropped from the solution (`74bf61ee`). Not shipped library API.

## Bucket 5 — UNHANDLED → write a line

A genuine user-facing change not represented anywhere. This is the whole point of the sweep. Route
it to its section per CHANGELOG_STYLE §3 (see the map below).

The 0.70.0 sweep found seven, illustrating the categories:
- `079d1859` env-var **hard-rename** → Breaking (a shipped var, `NUMSHARP_GUARD_PAGES`, changed).
- `b701843e` `NDArray.Normalize()` **deprecation** → Breaking.
- `9f573dd5`+`262eefd7`+`0151a832` `np.unique` full-parameter parity incl. an **axis-path NaN/`-0.0`
  correctness bug** → Parity & Fixes.
- `d1347c36` typed `OpenBlasMissingBackendException` (was bare `NotSupportedException`) → Parity.
- `b2a8374b` six creation/math/linalg/stats parity fixes → Parity umbrella (§7 umbrella grouping).
- `fc10404d`+`88550d13` `take`/`put` index-validation → Parity (see the "bundled fix" rule below).

---

## Hard-case decision rules (the ones worth reading twice)

**A pre-existing-behavior fix bundled inside a feature commit still earns its own line (§10.5).**
`fc10404d` is cited for `np.select`, but it *also* fixed `np.take`/`np.put` rejecting negative
indices under `mode='raise'` — a correction to ops that shipped *before* this range. Bundled ≠
folded: because `take`/`put` predate the range, the fix is a real user-visible change → its own
Parity line (folding `88550d13`'s float-index rejection with it). **Rule:** "cited for X" doesn't
absorb a Y in the same commit if Y fixes something older than the range.

**A `perf` commit belongs in Performance only if the body measures an NPY/NS ratio (§8).**
`1140efc3` (branch-light complex Acosh/Sqrt/Hypot) is "bit-for-bit identical, pure hot-path
restructuring" with **no ratio** → not Performance; and being no-behavior-change → Excluded.
Contrast `8a1376ff` (quantile, `x1.6->x5.2` measured) → Performance.

**"Restore a degraded cell to parity" is not a Performance win.** The OpenBLAS `fillZeros`/dispatch
fixes (`a150e4e9`/`6521b8b4`/`9cf599c4`) lift cells from ~0.3–0.7× back to ~1.0× — byte-identical,
package-internal, and *restoring* not *beating*. Excluded (and CHANGELOG_STYLE §8: parity-for-speed
trades aren't Performance either — `correlate`/`convolve` gaining OpenBLAS byte-parity is a §Parity
line, not §Performance).

**A breaking change to an API that never shipped isn't a Breaking line.** `np.parity_matmul`
existed for one commit before becoming the OpenBLAS package; the BLAS-leaves-Core `refactor!` and
the package renames all happened within the dev cycle. Reflect the final state; add no Breaking
entry.

**A `feat!` that is "unreleased API, changed earlier this branch" isn't Breaking either** — e.g.
`8cad3025` reorders `nanargmax`'s positional args, but the arg it moves shipped earlier *in this same
range*. Not a break relative to the last release.

**"Values unchanged" ⇒ not observable ⇒ drop it.** `b2a8374b` was "7 fixes" but its `corrcoef` item
says values are unchanged (an internal `out=`/divide restructuring). Observable count = 6; the
umbrella lists six.

**A number that moved across the range: cite the final value.** The coverage headline climbed
60.4% → 76.8% → 79.3% → 82.0% → **85.4% (478/560)** across separate commits; only the last
(`ff9d7f51`) is the truth. Same for parity counts and corpus sizes.

---

## Which section does an UNHANDLED feature go in (§3 map)

Fixed order; omit an empty section. H3 + leading emoji, per CHANGELOG_STYLE:

1. `### 💥 Breaking Changes` — changed signature/return type/observable behavior; deprecations; env
   renames. **First.** State old → new + migration.
2. `### 📦 New NuGet Packages` — packages shipped for the first time (§7A multi-line elaboration).
3. `### 📊 Dashboards & Docs` — living CI dashboards + key doc surfaces (§7B, linked, current numbers).
4. `### ✨ New APIs & Modules` — brand-new `np.*` / module surface.
5. `### 🧩 ndarray surface` — new/changed `ndarray.*` instance members.
6. `### ⚡ Performance` — measured speed/allocation wins; **lead with `xLOW->xHIGH`** under the
   one-line NPY/NS explainer.
7. `### 🎯 Parity & Fixes` — correctness/behavior alignment with NumPy that isn't a new API.
8. `### 🧰 Testing & Tooling` — only notable, deduplicated oracle/benchmark/coverage items.

Multi-function features (>~2 functions) get a grouped sub-list (§7); truly-unrelated small additions
get an **umbrella** parent with per-sub-bullet hashes (§7). Defer to CHANGELOG_STYLE for the exact
bullet shape and back-ticking.
