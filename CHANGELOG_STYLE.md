# CHANGELOG_STYLE.md

The house style for NumSharp release notes / changelogs. This is a **living document** —
amend it whenever we refine the format. The "Amendment log" at the bottom records every change
so past decisions stay traceable.

> One sentence: **a categorized, deduplicated, one-liner-per-feature changelog where every
> API name is `code-formatted`, multi-function features expand into a grouped sub-list, and
> performance entries lead with an `xLOW->xHIGH` NPY/NS ratio.**

---

## 1. Purpose & scope

- Audience: users upgrading NumSharp, skimming for "what's new, what's faster, what broke."
- Source of truth: the git commit range for the release (e.g. `master..<branch>`), read in full
  — subjects **and** bodies. Never write a line from the subject alone; the body carries the
  parity claims, the measured ratios, and the breaking-change notes.
- The changelog is **derived from commits but is not a commit log.** It is organized by *feature*,
  not by *commit*. Many commits collapse into one line (see §10 Deduplication).

---

## 2. Document header

```
# NumSharp <version>
```

- One H1 with the version. Optional italic scope note directly under it, e.g.
  `*Deduplicated highlights from journey3.*` — prefer this over a hard commit count, which drifts
  as sections dedup.
- No date required in the POC; add `— YYYY-MM-DD` on the real release.

---

## 3. Section taxonomy

Fixed order. Omit a section if it has no entries. Each is an H3 with a leading emoji:

| Order | Heading | Contains |
|-------|---------|----------|
| 1 | `### 💥 Breaking Changes` | Anything that changes an existing signature, return type, or observable behavior a caller relied on. |
| 2 | `### 📦 New NuGet Packages` | Packages shipped for the first time in this range. **Multi-line elaboration** (§7A), not one-liners. |
| 3 | `### 📊 Dashboards & Docs` | Living, CI-generated dashboards + key doc surfaces (§7B). Linked, one-liner each. |
| 4 | `### ✨ New APIs & Modules` | Brand-new `np.*` / module surface introduced in the range. |
| 5 | `### 🧩 ndarray surface` | New/changed `ndarray.*` instance members (kept separate from `np.*` because it reads as one story). |
| 6 | `### ⚡ Performance` | Speed/allocation wins. **Leads with the ratio** (§8), under a one-line NPY/NS explainer. |
| 7 | `### 🎯 Parity & Fixes` | Correctness/behavior alignment with NumPy that isn't a new API. |
| 8 | `### 🧰 Testing & Tooling` | Oracle/benchmark/coverage infra — only the notable, deduplicated items. |

Rationale for the split: a reader scans for a *kind* of change. Keep the buckets stable across
releases so diffs between changelogs are meaningful.

---

## 4. The bullet (one-liner)

- One dash bullet per **feature**. Target **1 line, hard cap 1.5 lines.** If it needs more, it's
  two features or it needs a sub-list (§7).
- **No hard newlines inside a bullet — one bullet per physical line.** Let a long bullet soft-wrap
  in the viewer; never insert a manual line break mid-bullet (it reads as a broken list and bloats
  diffs). This applies to the long sub-bullets in §7A too, however long they run.
- **Keep nesting shallow (≤2 levels) and tight (2 spaces per level).** Don't add a label bullet
  whose only job is to nest a long list one level deeper — promote the list instead (a family list
  like "Products / Decompositions / …" sits directly under the package, not under a "Supported
  functions:" wrapper). A deeply-offset list reads as buried.
- **Lead with the API name**, `code-formatted`. Then an em-dash, then the value in plain language,
  then the commit hash(es) last.
  ```
  - `np.kron` — Kronecker product, matches NumPy 2.4.2 — `7bcad845`.
  ```
- Present tense, active, user-facing. Say what the user gets, not how the kernel works.
  - ✅ `np.isin` hash-set membership replaces sort+searchsorted.
  - ❌ Refactored `TryHashMembership` to use the splitmix64 finalizer over the open-addressing table.
- Include a **hard number** when the commit body proves one (parity count, speedup, size limit).
  Numbers are the credibility currency of this changelog — never invent them, quote the body.
- Parity vocabulary is precise and load-bearing — use the strongest the body supports:
  - **byte-identical / bit-exact** — output bytes equal NumPy's (the strongest claim).
  - **parity / matches NumPy 2.4.2** — behavior/values match, bytes not asserted.
  - **verbatim errors** — exception messages reproduced exactly.

---

## 5. API & identifier formatting ("color")

- **Every** API/function/type/member name — top-level *and* inside sub-lists — is wrapped in
  backticks so it renders as inline code (the "color"). This is the single most important visual
  rule: a reader's eye locks onto the code-colored tokens.
- Backtick: `np.fft.fft`, `ndarray.strides`, `NDArray<byte>`, `default_rng`, `xLOW->xHIGH`? — no,
  ratios are not code (see §8). Backtick applies to **identifiers only**.
- Commit hashes are code-formatted too: `` `7bcad845` `` (7-char short SHA).
- **§7A and §7B narrow this rule (they override §5):** in the New NuGet Packages and Dashboards &
  Docs sections, code is reserved for function names, pythonnet *usage* calls, and mode/enum values —
  package and dashboard names are **bold**, not code.

---

## 6. Commit-hash citation

- End each bullet (or sub-bullet) with the commit hash(es) in backticks.
- **Primary first**, then folded follow-ups the same line, comma-separated:
  `` — `e868d8ae` (+ `754b7476`, `febfbbdd`). ``
- Placement:
  - Sub-bullets that are **facets of one feature** (e.g. the 18 `np.fft` functions) → hashes on the
    **parent** line; sub-bullets carry none.
  - Sub-bullets that are **independent functions, each from its own commit** (an umbrella grouping)
    → hash at the **end of each sub-bullet**.

---

## 7. Sub-lists (grouping multiple functions)

**When:** a single bullet introduces **more than ~2 functions**. Add an indented sub-list that
enumerates every function, **grouped by similarity / logical proximity** (transform family,
arithmetic vs. eval, real vs. complex, get/set pairs, …).

**Format:**
```
- `np.fft.*` — the whole 18-function Fourier module, a pure-managed pocketfft port, bit-exact incl. float32/float16 values — `3b9d5cfb`, `a525e355`, `4cb91898`.
  - `fft`, `ifft`, `fft2`, `ifft2`, `fftn`, `ifftn` — complex forward/inverse (1-D/2-D/N-D).
  - `rfft`, `irfft`, `rfft2`, `irfft2`, `rfftn`, `irfftn` — real-input transforms.
  - `hfft`, `ihfft` — Hermitian-symmetric transforms.
  - `fftfreq`, `rfftfreq`, `fftshift`, `ifftshift` — sample-frequency & shift helpers.
```

Rules:
- 2-space indent per nesting level (§4); the parent bullet stays on ONE physical line (§4), the
  sub-bullets nest under it. Keep it ≤2 levels deep.
- Each sub-bullet: `` `fn1`, `fn2`, `fn3` — short descriptor of the group. ``
- Function names are **code-colored exactly like the parent** (§5).
- Group by meaning, not alphabetically. Order groups from most- to least-central.
- The parent line still carries the headline claim (bit-exact / speedup / count) and the hashes
  (unless it's an umbrella grouping — see §6).
- **Umbrella grouping** — a legitimate dedup tool: one parent like *"Additional array/linalg/stats
  functions"* whose sub-bullets are otherwise-unrelated small additions, each hash-tagged:
  ```
  - Additional array, linalg & stats functions —
    - `np.cov`, `np.corrcoef` — covariance & Pearson correlation — `92dc537b`, `aaf731b2`.
    - `np.choose` — index-into-choices gather — `aaa41ef2`.
  ```

---

## 7A. The "New NuGet Packages" section

New packages are a bigger deal than a single API — they get **multi-line elaboration**, not a
one-liner. The house shape is a **headline + a grouped capability sub-list** (the §7 pattern),
enriched from the **actual package source** (`.csproj` metadata + the public API), never from memory.

Per package:
- **Top line:** `` **`Package.Id`** `` — one-clause value proposition (what changes for the user),
  with the strongest honest claim (e.g. *byte-identical to NumPy 2.4.2*).
- **Capability sub-lists**, grouped by logical proximity, code-coloring every function/API name
  exactly like §5/§7. For a package that *serves existing APIs* (a backend), **enumerate the served
  functions** — do not hand-wave "and more"; read the seam (e.g. the `IBlasBackend.Try*` members and
  the `np.*` compositions built on them) and list them all, grouped.
- **"Ways to use it"** — if a package has more than one usage style (explicit calls vs. an implicit
  codec, etc.), give a one-line intro sub-bullet then a nested sub-bullet per style.
- **Control/config knobs** the package adds — described in plain prose, word-reduced (no code
  identifiers, no routine names).
- **Opt-in line:** name the package, and (if true) *"Core stays 100% managed without it."*
- **Bundled/Depends line:** native assets bundled (version, RID count, provenance) or the external
  NuGet dependency (version floor + why).
- **Links & provenance:** external links are allowed in this section (PyPI project page, upstream
  repo/docs) for a bundled binary or a dependency; when a package accepts external input types, name
  each with its source package (e.g. Pillow, PyTorch, stdlib) **and link it**. Link only what is
  actually used / interop'd — never a dependency the package explicitly AVOIDS (e.g. Numpy.NET stays
  plain, unlinked).
- **Code formatting is SCOPED here (overrides §5):** backticks go ONLY on function names (the served
  API), on pythonnet *usage* calls (`arr.ToNumpy()`, `RegisterCodec()`, …), and on a package's key
  **mode / enum values** (`Auto`/`View`/`Copy`). Package names are **bold**, not code. Drop
  implementation identifiers — LAPACK routine names, engine/config APIs, MSBuild properties,
  interfaces, checksums — or render them plainly.
- **No commit hashes in this section** — packages are named things, not per-commit changes; hashes
  clutter the multi-line elaboration.

Discovery is mandatory before writing this section: open each new `.csproj` (PackageId, Description,
dependencies, license, bundled assets) **and** grep the public surface. "We support more" is not a
license to guess — it's an instruction to go read the code and list what's actually there.

## 7B. The "Dashboards & Docs" section

Living dashboards and key doc surfaces get a compact section (after New NuGet Packages) — they are
part of what a release delivers. One bullet per dashboard: **bold linked name** — one-line purpose,
optionally one sub-bullet of what it contains. Link to the PUBLISHED page (the site URL), not the
repo source. Quote only numbers the dashboard/artifact actually reports (coverage %, benchmarkable-API
count, corpus size) and keep them current. No commit hashes (like §7A).

## 8. Performance notation

Performance bullets **lead with the NPY/NS ratio range**, then the one-liner, then hashes.

- **Put a one-line explainer directly under the `### ⚡ Performance` header** (italic): ratios are
  NumPy ÷ NumSharp, higher is better, `xLOW->xHIGH` spans the worst→best measured cell. A reader must
  never have to guess the direction.
- Convention: **NPY/NS = NumPy_time / NumSharp_time**, so **>1 = NumSharp faster**. This matches the
  repo-wide performance convention (higher is better).
- Format: **`xLOW->xHIGH`** — ASCII `x`, ASCII `->`, no `×`/`→`. Chosen for greppability and clean
  diffs between changelog revisions.
  - `LOW` = **worst** measured cell, `HIGH` = **best** measured cell, across the size×dtype matrix.
  - **Report the honest low, even when sub-1.0.** `x0.98->x74` tells the truth (one near-parity
    cell, best case 74×). Hiding the low is dishonest.
- Single-figure win (one specific case, no meaningful range): drop the arrow — `x15.6`.
- Before/after (a commit that *lifted an existing* op): keep the range form for the *current* state
  and note the prior worst in parens — `x1.35->x11 (float64 was x0.28)`.
- Shape of the bullet:
  ```
  - `x0.98->x74` — `np.unique` family routed through the radix sort core — `5df10897`.
  ```
- Numbers come from the commit body's measured table only. If a commit claims no ratio, it does not
  belong in Performance — put the change in its feature section instead.
- **Parity-for-speed trades are NOT Performance.** `correlate`/`convolve` gaining OpenBLAS
  byte-parity *slows down* to match NumPy's bits — that's a §Parity entry, not §Performance.

---

## 9. Breaking changes

- Always first section. Mark the feature so a scanner cannot miss it.
- State the **old → new** and the **why**, plus the migration if non-obvious.
  ```
  - `ndarray.strides` now reports **bytes** per axis (was elements), matching NumPy's
    `PyArray_STRIDES` — `6ef30215`.
  ```
- If a `feat!:`/`refactor!:` commit exists, it's a breaking change by definition — surface it here.

---

## 10. Deduplication methodology (the core technique)

A feature that was **new in this range** usually arrives across several commits: the initial `feat`,
then `fix`/audit passes, then the `test(oracle)`/`bench` wiring. Those follow-ups are **not**
separate changelog lines — to a reader they're one feature.

Procedure:
1. **Bucket every commit by feature**, not by type. (`e868d8ae` default_rng, `754b7476` its parity
   audit, `febfbbdd` its completeness fixes, `f491c499` its bytes fix, `ceb10093` its oracle → one
   bucket.)
2. **One line per bucket.** Write it describing the *final* state of the feature (fold the fixes in —
   the reader gets the fixed thing, not "added X" then "fixed X").
3. **Cite the primary commit first, key follow-ups after** (§6). You don't have to cite every commit
   in the bucket — cite the ones that carry weight; account for the rest by folding their substance.
4. **Never emit a "fixed the thing I just added" line.** If both the add and its fix are in-range,
   only the add's line survives, describing the fixed behavior.
5. A commit that fixes a **pre-existing** (shipped-before-this-range) behavior *is* its own
   Parity/Fixes line — that's not redundant, it's a real user-visible change.
6. **Cross-section dedup.** A function that computes ONLY through a package (e.g. the LAPACK
   factorisations) lives in §7A New NuGet Packages — its natural home — and is NOT re-listed in New
   APIs. Cross-reference it from New APIs with a short pointer bullet that carries its commit hashes
   (§7A omits hashes, so the pointer is where they land).

"Cover N commits" means *account for* N commits after dedup — the visible line count is smaller.

---

## 11. Exclusions (never appear in release notes)

- Pure chores: `.git-blame-ignore-revs` entries, file/case renames, solution-folder moves,
  project renames, `chore(sln)`.
- Docs-only commits (unless they document a user-facing contract change).
- Benchmark/oracle *infrastructure* churn — **except** a notable, deduplicated line in §Tooling
  (e.g. "oracle extended to the LAPACK family", "benchmark dashboard").
- Internal refactors with no observable effect (e.g. "make `NDAxisState` a ref struct"), unless they
  removed a user-visible limit (then it's a Fix: "removed the 64-dim cap").

---

## 12. Worked example (the canonical shapes)

One-line-per-bullet, 2-space nesting, section rules applied — demonstrates the packages (§7A),
dashboards (§7B), a sub-list feature (§7), and the performance explainer (§8):

```
### 📦 New NuGet Packages
- **NumSharp.Interop.OpenBLAS** — opt-in BLAS+LAPACK backend, byte-identical to [NumPy](https://numpy.org) 2.4.2; Core stays 100% managed without it.
  - Products — `dot`, `matmul`, `inner`, `vdot`, `vecdot`, `matvec`, `vecmat`.
  - Delivery — bundles numpy's pinned [scipy-openblas64](https://pypi.org/project/scipy-openblas64/), per-RID for 8 platforms.

### 📊 Dashboards & Docs
- **[Supported Features](https://scisharp.github.io/NumSharp/docs/coverage-support-dashboard.html)** — NumPy 2.x API coverage & support; headline ~85% (478/560).

### ✨ New APIs & Modules
- `np.random.default_rng` — the full modern PCG64 `Generator`, byte-identical streams to NumPy 2.4.2 — `e868d8ae` (+ `754b7476`, `febfbbdd`).
  - `default_rng` — the entry point (seed / `SeedSequence` / `BitGenerator` overloads).
  - `random`, `integers`, `standard_normal`, `choice`, `shuffle`, `permutation` — the Generator draw surface.

### ⚡ Performance
*Ratios are NumPy ÷ NumSharp — higher is better; `xLOW->xHIGH` spans the worst→best measured cell.*
- `x0.98->x74` — `np.unique` family routed through the radix sort core — `5df10897`.
```

---

## 13. Authoring checklist

- [ ] Read every commit body in range, not just subjects.
- [ ] Bucket by feature; fold follow-ups; apply cross-section dedup (§10, incl. package-only → §7A + pointer).
- [ ] Section order per §3; one bullet per physical line, ≤2-level tight (2-space) nesting (§4).
- [ ] Every API name backticked (§5) — except §7A/§7B, which are functions + usages + mode/enum values only, names **bold**.
- [ ] Multi-function features have a grouped sub-list (§7).
- [ ] New packages: multi-line elaboration from the actual source, served functions enumerated, links to used deps, **no hashes** (§7A).
- [ ] Dashboards & Docs: linked to the published page, current artifact numbers, no hashes (§7B).
- [ ] Performance under a one-line NPY/NS explainer; bullets lead with `xLOW->xHIGH`, low reported honestly (§8).
- [ ] Breaking changes first, with old→new + migration (§9).
- [ ] Hashes backticked & placed per §6 (present everywhere except §7A/§7B).
- [ ] Chore/rename/docs-only excluded (§11).
- [ ] Every number traced to a commit body or a dashboard artifact — none invented.

---

## 14. Amendment log

| Date | Change |
|------|--------|
| 2026-08-22 | Initial style guide, extracted from POC 2. Adds: (a) grouped function **sub-lists** for multi-function features, code-colored identically to the parent; (b) the **`xLOW->xHIGH`** performance-ratio notation (NPY/NS, honest low, ASCII arrow). |
| 2026-08-22 | Added the **`📦 New NuGet Packages`** section (order 2, after Breaking) with its own format spec (§7A): headline + grouped capability sub-lists, enumerate served functions from the seam, "ways to use it" nesting, opt-in + bundled/depends lines, and a mandatory source-discovery step. |
| 2026-08-22 | Rendering rule (§4): **no hard newlines inside a bullet — one bullet per physical line**, soft-wrap only. Applies to §7A sub-bullets however long. |
| 2026-08-22 | §7A scoping: in New NuGet Packages, **code formatting is functions-only** (served API) plus pythonnet *usage* calls; package names bold-not-code; implementation identifiers dropped/plain; **no commit hashes** in the section. Prefer word-reduced prose. |
| 2026-08-22 | §7A: external **links** allowed for provenance (PyPI/upstream docs); name accepted external input types with their **source package** (Pillow/PyTorch/stdlib/…). |
| 2026-08-22 | Three refinements: (a) §4 **shallow/tight nesting** (≤2 levels, 2 spaces, no label-only wrappers — promote the list); (b) §7A links **only what's used**, never an avoided dependency, and link the named source packages; (c) §7A code scope also covers **mode/enum values** (`Auto`/`View`/`Copy`). |
| 2026-08-22 | §10 **cross-section dedup**: package-only functions (LAPACK factorisations, complex128 products) live in §7A, not New APIs; New APIs carries a pointer bullet with their hashes. |
| 2026-08-22 | Added **§7B Dashboards & Docs** section (order 3, after New NuGet Packages): linked one-liners for the coverage/benchmark/oracle dashboards, published-URL links, current artifact numbers, no hashes. |
| 2026-08-22 | §8: require a **one-line NPY/NS explainer** italicised under the Performance header (direction must be unambiguous). |
| 2026-08-22 | **Consolidating pass:** fixed §7 4-space→**2-space** indent (matches §4) and de-wrapped its examples to one-line-per-bullet; added the §7A/§7B code-scope note to §5; refreshed §12 worked example to the current shapes (packages/dashboards/sub-list/perf explainer); expanded §13 checklist; §2 scope note drops hard commit counts. No rule changes — coherence only. |
