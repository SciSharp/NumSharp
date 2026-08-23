# Verify the changelog is TRUE, not just complete

Completeness (every change has a line) is half the job. The other half is **truth** — every claim
traces to a commit body or a committed artifact, and nothing overstates what happened. On a **job B
sweep** (refreshing an existing changelog) this is where you earn your keep: the pre-existing lines
were written by someone reading fewer commits than you just did, and they drift.

Run this checklist after the last batch, before declaring the range covered.

## 1. Every number traces to a body — and is the FINAL value

CHANGELOG_STYLE §4/§13: "Every number traced to a commit body or a dashboard artifact — none
invented." For each `xN` ratio, parity count, coverage %, corpus size, or size-limit:

- **Quote it from a body**, not from memory. If you can't find it, cut it.
- **Use the final value** if it moved. A coverage/parity/corpus number climbs commit-by-commit; the
  changelog must show the *last* one. 0.70.0: coverage `478/560 (~85%)` is right because
  `ff9d7f51` (the final coverage regen) reports 85.4% / 478/560 — earlier commits said 76.8%, 79.3%,
  82.0%. Verify against the committed artifact when one exists:
  ```bash
  grep -iE 'headline|percent|/560' coverage/generated/summary.md    # or the relevant artifact
  ```
- **Ratios lead Performance and must be `>1`** (or an honest `xLOW->xHIGH` whose low is reported even
  when sub-1). A commit that quotes no ratio doesn't belong in Performance.

## 2. "Fixed" vs "surfaced / found / pinned" — the highest-value fact-check

A commit that **discovers** bugs and carves them to `[OpenBugs]` (a known-issue gate) did **NOT fix
them.** An existing changelog line that says "…surfaced and **fixed** N bugs" when the body says the
N cases were "carved from the green corpus and pinned under `[OpenBugs]` (verified red, CI-excluded)"
is factually wrong — rewrite it to "surfaced N … pinned as known `[OpenBugs]` issues (not yet
fixed)."

This was the single correction the 0.70.0 sweep made to an *already-present* line (`31a178f2`, the
8 `np.random` samplers). Read every body — including cited ones — precisely to catch this class.

Watch the verbs: **surfaced / found / exposed / carved / pinned / documented / recorded / deferred /
left open** ≠ **fixed / closed / corrected / matched / byte-exact now**.

## 3. §7A New NuGet Packages — audit the section against the seam

Because §7A folds a package's whole history and carries no hashes, its correctness is *not* protected
by hash-matching. So:

- **Enumerate the served functions from the code, not the prose** (§7A: "read the seam … list them
  all, grouped" — e.g. the `IBlasBackend.Try*` members and the `np.*` built on them). Confirm the
  section lists them and hasn't drifted (a function added mid-range must appear).
- **Spell every name as the real API** and verify it exists:
  ```bash
  grep -rl '<VerbOrType>' src/<Package>/            # e.g. FromArrayLike, OpenBlasEngine.Enable
  ```
  (The 0.70.0 sweep confirmed `FromArrayLike` is a real pythonnet verb before trusting the line.)
- **Modes/enums** the package adds are code-colored (`Auto`/`View`/`Copy`); package **names are
  bold, not code** (§7A overrides §5). Drop implementation identifiers.
- **Bundled/depends** line: native-asset version, RID count, provenance — or the external NuGet
  floor. Verify the version/floor from the `.csproj` / manifest.

## 4. §7B Dashboards & Docs — links and current numbers

- Link the **published page** (site URL), not the repo source.
- Quote only numbers the dashboard/artifact **actually reports**, current as of the final commit
  (coverage %, benchmarkable-API count, corpus size). Cross-check against the generated artifact.

## 5. Dedup correctness

- **No "fixed the thing I just added" line** survives (§10.4). If both an `add` and its `fix` are
  in-range, the one surviving line describes the *fixed* behavior.
- A **feature line's folded hashes** are the ones that carry weight (primary first, then key
  follow-ups) — you don't have to cite every commit in the bucket, but the substance of each must be
  accounted for.
- **Cross-section pointers** (§10.6): a package-only function listed in §7A has a New-APIs pointer
  bullet carrying its hashes (since §7A omits them). Confirm the pointer exists and the hashes
  resolve.

## 6. Breaking-changes completeness

- Every `feat!`/`refactor!` in the range was either surfaced as a Breaking line **or** deliberately
  excluded because the API **never shipped in a release** (a within-cycle rename/removal). Make sure
  you consciously ruled on each `!` commit — don't let one slip through as "chore."
- Each Breaking line states **old → new + migration** when non-obvious (§9).

## 7. Final mechanical re-grep

```bash
# every hash you added/edited resolves and is in range:
for h in <all-hashes-you-touched>; do git cat-file -t $h >/dev/null 2>&1 && echo "$h ok" || echo "$h MISSING"; done
# the cited set grew by the number of lines you added:
grep -oE '`[0-9a-f]{8}`' docs/releases/RELEASE_<version>.md | tr -d '`' | sort -u | wc -l
# sanity: the old wrong wording is gone, the new one present (for any correction you made):
grep -c '<old-wrong-phrase>' docs/releases/RELEASE_<version>.md      # expect 0
grep -q  '<new-correct-phrase>' docs/releases/RELEASE_<version>.md && echo "correction present"
```

## Sign-off

State plainly, per bucket: how many commits were Cited / Folded / Package-§7A / Excluded / UNHANDLED,
which UNHANDLED lines you added and where, any factual correction you made to an existing line, and
which §7A/§7B facts you verified (with the value). **Do not commit** unless the user asks — leave the
release file edited and ready for review. If a batch is unread or a number is unverified, say so
rather than implying full coverage.
