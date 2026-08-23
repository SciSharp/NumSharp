---
name: changelog
description: >-
  NumSharp's release-notes authoring + large-range commit-sweep playbook — how to write a
  changelog in the house style AND how to work through a few-hundred-commit range without missing
  or mis-stating anything. Use this whenever you write or refresh a release changelog
  (`docs/releases/RELEASE_<version>.md`), sweep a commit range for changes not yet in the changelog
  ("go over the commits and find what's unhandled"), decide whether a commit deserves a line, dedup
  many commits into one feature line, or verify a changelog is complete AND factually true. Trigger
  on: "changelog", "release notes", "write up the changelog", "CHANGELOG_STYLE", "RELEASE_x.y.z",
  "what's new since <version>", "go over the commits in batches", "find unhandled changes", "which
  commits are missing from the changelog", "sweep the commits", "master..<branch>", "categorize /
  dedup the commits", "did we cover every commit". Reach for it before hand-writing a changelog from
  `git log` or claiming a range is "fully covered".
---

# NumSharp Changelog Authoring & Large-Range Commit Sweep

## Mental model (why this exists)

A changelog is **derived from commits but is NOT a commit log.** It is organized by *feature*, one
line per feature, with rigorous deduplication — many commits collapse to one line, and most commits
produce **no** line at all. Getting that right over a few-hundred-commit range is a *process*
problem, not a writing problem.

**Two authorities, split by concern:**

- **`CHANGELOG_STYLE.md`** (repo root) owns the **FORMAT** — the section taxonomy, the one-line
  bullet shape, API back-ticking ("color"), the `xLOW->xHIGH` performance notation, the §7A New
  NuGet Packages / §7B Dashboards elaboration, and the §-numbered rules. **Read it first and follow
  it verbatim; never restate its rules from memory.**
- **This skill** owns the **PROCESS** — how to enumerate the range, read every commit, bucket each
  one, dedup, write only the lines that are owed, and prove the result is both complete and true.

A release changelog lives at `docs/releases/RELEASE_<version>.md`, derived from the release's commit
range — usually `master..<branch>` (or `<prev-tag>..<branch>`).

## The two jobs (same engine)

- **(A) Author / refresh** a changelog for a range from scratch.
- **(B) Sweep** a range against an *existing* changelog — "find every commit whose change isn't
  represented yet." (The 0.70.0 journey3 sweep: 294 commits, 20 batches → 7 edits.)

Both run the identical engine: **read every commit body, bucket each commit, then emit changelog
lines only for the `UNHANDLED` bucket** (job A starts from an empty changelog; job B starts from a
mostly-full one). Job B additionally **fact-checks the lines already there** (see accuracy below —
this is where you catch a "fixed X" that should read "found and pinned X").

## The core technique — the 5-bucket triage

Read **every** commit body — subject *and* body (CHANGELOG_STYLE §1: "Never write a line from the
subject alone; the body carries the parity claims, measured ratios, and breaking-change notes").
Put each commit in **exactly one** bucket:

| Bucket | Meaning | Produces a line? |
|--------|---------|:----------------:|
| **1. Cited** | Its short hash already appears in the changelog. | No — done. |
| **2. Folded** | Its substance belongs to a feature already listed (a follow-up `fix`/audit/test-wiring of an in-range feature). CHANGELOG_STYLE §10.4: never emit "fixed the thing I just added." | No — fold its hash onto the feature's line if it carries weight. |
| **3. Package-internal** | It develops a NuGet package (OpenBLAS / pythonnet). It folds into that package's **§7A** elaboration, which carries **no hashes**. | No — but confirm the §7A section already names the capability it adds. |
| **4. Excluded** | Docs-only, skill/website, chore/rename, `.git-blame-ignore-revs`, CI/signing, benchmark-infra, test/oracle-infra, internal refactor with no observable effect, merge (CHANGELOG_STYLE §11). | No. |
| **5. UNHANDLED** | A genuine user-facing change (a `feat`/`fix`/breaking/`perf`-with-a-ratio) not represented anywhere. | **Yes** — write/insert one line in its section. |

**Only bucket 5 gets a line.** The whole skill is: sort every commit into these five, cleanly.

The hard-case decision rules (the ones that cost real judgement) are in
**`references/triage-taxonomy.md`** — read it before ruling on any `perf`, any `!` breaking commit,
or any `fix` bundled inside a feature commit. The traps that recur:

- A `fix`/`feat` whose subject looks new but is a **follow-up** to an in-range feature → **Folded**.
- A fix to **pre-existing** (shipped-before-this-range) behavior → its **own Parity line** (§10.5),
  *even when bundled in a commit cited for something else* (the `take`/`put` index-validation case).
- A `perf` commit with **no NPY/NS ratio** in the body → **not** Performance (§8); if it's also
  bit-identical/no-behavior-change → **Excluded**.
- A `perf` that merely **restores a degraded cell to parity** (not "beats NumPy") → not a
  Performance win.
- A `refactor!`/`feat!` breaking an API that **never shipped in a release** → **not** user-facing
  breaking (a rename/removal within the same dev cycle).
- **Compile-time-only** changes (ValueTuple field casing) with no runtime effect and no caller →
  **Excluded**. **Exposing internal types** as public → not a NumPy-surface feature → **Excluded**.
- **Examples/demo** project work → **Excluded** (not shipped library API).

## Sweep procedure (brief)

The full mechanical procedure — commands, the `dump_batch` helper, batch sizing, the Windows path
trap — is in **`references/sweep-procedure.md`**. In brief:

1. **Range + count.** `git log --oneline master..<branch> | wc -l`. Confirm the target release file.
2. **Extract cited hashes** from the existing changelog:
   `grep -oE '`[0-9a-f]{8}`' docs/releases/RELEASE_<v>.md | tr -d '`' | sort -u`.
3. **Numbered oldest-first list** and the **uncited subset** (candidates to scrutinise), but plan to
   read **all** bodies anyway — the cited ones are where you fact-check existing lines.
4. **Batch by ~15 commits** (≈20 batches for a few-hundred-commit range). **Dump each batch's full
   bodies to a file in your session scratchpad** (not `/tmp` — the Read tool can't see Git-Bash
   `/tmp` on Windows), then Read the file. One batch ≈ 1–2 Read pages.
5. **Per batch: bucket every commit; update the changelog when the batch yields an UNHANDLED item,
   else continue.** State the verdict for each commit so the reasoning is auditable.
6. After the last batch, **verify accuracy** (below) and re-grep to confirm every new hash resolves.

## Verify the changelog is TRUE, not just complete

Completeness is half the job; the other half is that **every claim traces to a commit body or a
committed artifact**. Full checklist in **`references/verify-accuracy.md`**. The load-bearing ones:

- **Every number** (coverage %, parity counts, `xN` ratios, size limits) must be quoted from a body,
  and must be the **final** value if it moved across the range (a coverage headline climbs
  commit-by-commit; cite the last one).
- **"Fixed" vs "surfaced/pinned."** A commit that *finds* bugs and carves them to `[OpenBugs]` did
  **not fix** them. This is the single highest-value fact-check on an existing changelog.
- **§7A package sections** must name every served function/mode/verb, spelled as the **real API**
  — read the seam / grep the surface, don't trust the prose (verify e.g. that a verb like
  `FromArrayLike` actually exists).
- **Dedup correctness:** a "fixed the thing I just added" line is a defect — collapse it into the
  feature's line describing the *final* behavior.

## Critical gotchas (learned the hard way)

- **The Read tool needs Windows paths; Git-Bash `/tmp` is invisible to it.** Dump commit bodies to
  the session **scratchpad** directory (the harness gives you one) and Read *that*.
- **Bash output over ~60 KB is truncated to a file.** Either read the saved tool-result file, or —
  better — write the dump straight to a scratchpad file and Read it. Don't try to eyeball 15 long
  bodies from one Bash call's stdout.
- **Oldest-first ordering** (`git log --reverse`) makes batches read as the journey's progression
  and makes "folded follow-up" obvious (the fix comes *after* the feat it patches).
- **`§7A` carries no hashes.** Package-internal commits are handled by the package *section*, not by
  a per-commit line — so a package rename, delivery-model rework, or dtype-map fix is "handled" the
  moment the §7A elaboration is accurate. Your job for those is to *audit the section*, not add lines.
- **The uncited set is a focusing tool, not a filter.** Most uncited commits are Excluded/Folded;
  the few genuine UNHANDLED items hide among them. But still read the *cited* bodies too — that's how
  you catch a wrong number or a "fixed" that should be "pinned."
- **Don't inflate Performance.** A ratio only earns a Performance line if the body measures it and it
  is `>1` (or an honest `xLOW->xHIGH` spanning a sub-1 low). Parity-for-speed trades and
  restore-to-parity fixes are not Performance (§8, §13).

## References

- `references/sweep-procedure.md` — the mechanical batch sweep: commands, the `dump_batch` helper,
  batch sizing, the scratchpad/Windows-path workflow, re-verification.
- `references/triage-taxonomy.md` — the 5 buckets in full, every hard-case decision rule with a
  worked example from the 0.70.0 sweep, and the "which section does this feature go in" map.
- `references/verify-accuracy.md` — the fact-check checklist: numbers, "fixed vs pinned", §7A/§7B
  audits, dedup correctness, and the final re-grep.
- `CHANGELOG_STYLE.md` (repo root) — **the format authority.** Section taxonomy, bullet shape, API
  color, hash placement, `xLOW->xHIGH`, §7A/§7B, the amendment log. Always defer to it for *how a
  line looks*.
