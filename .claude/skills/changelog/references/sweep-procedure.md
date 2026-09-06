# Sweep procedure - working a large commit range mechanically

The step-by-step for reading and bucketing every commit in a range, tuned for a few-hundred-commit
release on Windows/Git-Bash where the Read tool cannot see `/tmp`. Worked example throughout: the
0.70.0 sweep of `master..journey3` (294 commits, batches of 15 → 20 batches).

## 0. Orient

```bash
cd <repo>
git log --oneline master..<branch> | wc -l          # total commits in range
ls docs/releases/RELEASE_<version>.md               # the file you'll write into
```

Confirm the range with the user if ambiguous (`master..<branch>` vs `<prev-tag>..HEAD`). The
changelog covers exactly this range.

## 1. Extract what's already cited

The existing changelog's cited hashes tell you which commits are already `Cited` (bucket 1). Pull
them once:

```bash
grep -oE '`[0-9a-f]{8}`' docs/releases/RELEASE_<version>.md | tr -d '`' | sort -u > /tmp/cited.txt
wc -l /tmp/cited.txt
```

(`/tmp` here is fine - it's only touched by Bash/`grep`, never the Read tool.)

## 2. Numbered, oldest-first commit list + the uncited subset

Oldest-first so batches read as the journey progressed (a folded follow-up fix then sits *after* the
feature it patches - the ordering makes the dedup obvious):

```bash
git log --reverse --format='%h %s' --abbrev=8 master..<branch> > /tmp/commits_oldest.txt
awk '{n=NR; h=$1; $1=""; printf "%3d %s%s\n", n, h, $0}' /tmp/commits_oldest.txt > /tmp/commits_numbered.txt
# the uncited candidates (still read the cited ones too - see §5):
grep -vf /tmp/cited.txt /tmp/commits_numbered.txt > /tmp/uncited.txt
wc -l /tmp/uncited.txt
```

The uncited set is a **focusing tool**: it's where the genuine `UNHANDLED` items hide (among many
Excluded/Folded). It is **not** a filter - you still read the cited bodies to fact-check existing
lines.

## 3. The batch dump helper (writes to the scratchpad the Read tool CAN see)

**Do not** try to read long commit bodies out of Bash stdout - 15 bodies blow past the ~60 KB
truncation and land in a tool-result file anyway. Dump each batch straight to your **session
scratchpad** directory (the harness prints its path; it's a real Windows path the Read tool opens):

```bash
SP="<your-session-scratchpad-dir>"        # e.g. .../Temp/claude/<proj>/<session>/scratchpad
mkdir -p "$SP"
dump_batch() {                            # start-line end-line outfile  (lines index /tmp/commits_oldest.txt)
  local s=$1 e=$2 f=$3; : > "$f"
  sed -n "${s},${e}p" /tmp/commits_oldest.txt | awk '{print $1}' | while read h; do
    echo "########## $h ##########" >> "$f"
    git show -s --format='%s%n%n%b' "$h" >> "$f"
    echo >> "$f"
  done
  echo "-> $(basename "$f") ($(wc -l < "$f") lines)"
}
dump_batch  1  15 "$SP/batch01.txt"
dump_batch 16  30 "$SP/batch02.txt"
# ... through the last batch
```

**Batch size:** ~15 commits keeps each file to ~1-2 Read pages. For N commits aim for ≈20 batches
(`ceil(N/15)`); the "20 batches" framing is a readable-chunk target, not a rule.

## 4. Read each batch and bucket every commit

`Read "$SP/batchNN.txt"` (page through if a batch is long - long-bodied commits like GEMM/oracle
work run 200+ lines each). For **each** commit, state a one-line verdict naming its bucket and why:

```
27b9b012 feat(twodim): diag family        → Cited (line 73)
a7782984 fix(twodim): fill_diagonal array  → Folded into diag family (follow-up fix)
668f227d feat(blas): bundle OpenBLAS        → Package §7A (Delivery)
4ea939d9 docs: Shape fields are long        → Excluded (docs-only)
b2a8374b fix(np): 7 parity fixes            → UNHANDLED → Parity & Fixes
```

Use `references/triage-taxonomy.md` for any commit that isn't obvious.

## 5. Update per batch, or continue

Per CHANGELOG_STYLE §10 and the batch cadence: **after a batch, if it yielded an `UNHANDLED` item,
insert its line(s) into the right section; otherwise continue.** Prefer to also **fix any factual
error you spot in an already-cited line right then** (e.g. a "fixed" that the body shows was
"surfaced and pinned"). Keep edits surgical (one `Edit` per inserted/corrected bullet) so the diff
stays reviewable.

Track progress across the batches (a task item, or a running notes file) so you don't lose your
place across a 20-batch run.

## 6. Re-verify after the last batch

```bash
# every hash you added resolves and is in range:
for h in <new-hashes>; do git cat-file -t $h >/dev/null 2>&1 && echo "$h ok"; done
# the cited set grew by the number of lines you added:
grep -oE '`[0-9a-f]{8}`' docs/releases/RELEASE_<version>.md | tr -d '`' | sort -u | wc -l
```

Then run the accuracy checklist (`references/verify-accuracy.md`). Do **not** commit unless the user
asks - leave the release file edited and ready for review.

## Gotchas specific to the sweep

- **Read tool ≠ `/tmp`.** On Windows the Read tool opens Windows paths; the Bash tool's `/tmp` is
  Git-Bash-local and invisible to Read. Dump bodies to the scratchpad, keep throwaway text lists
  (`cited.txt`, `commits_*.txt`) in `/tmp`.
- **`git show -s --format='%s%n%n%b'`** gives subject + blank + full body, which is exactly what you
  must read (never the subject alone).
- **`--abbrev=8`** matches the 8-char hashes CHANGELOG_STYLE uses, so cited-hash matching is a clean
  string compare.
- **The uncited line-numbers can drift from the numbered list** if a hash prefix collides - always
  bucket by the **hash** the dump prints, not by a line number.
- **Don't fabricate a batch you haven't read.** If a batch is still being dumped or you haven't
  opened it, say so; never guess a commit's bucket from its subject to "finish faster."
