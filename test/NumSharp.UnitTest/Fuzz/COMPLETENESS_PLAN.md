# Fuzz & Oracle Completeness — Remediation Plan (SOURCE OF TRUTH)

> **This file is the single source of truth** for the three-workstream remediation of the
> differential-fuzz pipeline. Every teammate reads it FIRST, works ONLY its own workstream,
> and ACTIVELY updates its own `## Status — WS-*` section (and appends to the Bug Ledger) as
> work progresses. Re-read the file immediately before every edit to it (others edit it too).

| Meta | Value |
|---|---|
| Worktree | `K:/source/NumSharp/.claude/worktrees/fuzz-completeness` |
| Branch | `worktree-fuzz-completeness` (based on `nditer` @ `68298eaf`) |
| Main checkout (DO NOT TOUCH) | `K:/source/NumSharp` (branch `refactor-drawing-libs`) |
| Oracle NumPy | 2.4.2 (verified installed; `python` on PATH) |
| Date opened | 2026-07-07 |
| Workstreams | WS-BUGS → agent `fuzz-bugs` · WS-GAPS → agent `fuzz-gaps` · WS-DOCS → agent `fuzz-docs` |

Origin: a completeness review of `test/oracle/` + `test/NumSharp.UnitTest/Fuzz/` performed
2026-07-07 against NumPy 2.4.2. All findings below were verified by reading the code and, where
marked **[probed]**, by executing NumPy 2.4.2 directly.

---

## 0. Ground rules (ALL teammates)

1. **Work only inside the worktree** `K:/source/NumSharp/.claude/worktrees/fuzz-completeness`.
   Use absolute paths. Never modify the main checkout `K:/source/NumSharp` working tree.
2. **Mutex — mandatory.** Every compile or test goes through the global mutex (auto-loaded in
   Bash): `mutex-capture build -- <command>`. Compile and test share the `build` lock. For
   multi-command commits use `mutex-capture git -- bash -c 'git add <files> && git commit -m ...'`.
   NEVER `taskkill`. On "file is being used by another process": you skipped the mutex — wait
   and retry through it.
3. **Build** (from worktree root):
   `mutex-capture build -- dotnet build -v q --nologo "-clp:NoSummary;ErrorsOnly" -p:WarningLevel=0`
4. **The gate** (from `<worktree>/test/NumSharp.UnitTest`):
   - Iteration (fast): `mutex-capture build -- dotnet test --no-build -f net10.0 --filter "TestCategory=FuzzMatrix&TestCategory!=OpenBugs"`
   - Single tier: add `&Name~<TestName>` e.g. `--filter "Name~Stat&TestCategory=FuzzMatrix"`
   - Final (before declaring DONE): the same filter WITHOUT `-f` (runs net8.0 + net10.0), plus a
     broad sanity `--filter "TestCategory!=OpenBugs&TestCategory!=HighMemory"`.
   - `Index_Random` is `[FuzzMatrix]+[OpenBugs]` because of a known flaky teardown SEGFAULT (R3)
     — never let it block you; run it best-effort only.
5. **Corpus regeneration** (deterministic — same input ⇒ byte-identical output; from worktree root):
   - `python test/oracle/gen_oracle.py <mode>` — modes: `smoke astype_full binary divmod_power
     comparison unary reduce where place matmul rounding bitwise unary_extra nanreduce scan stat
     logic modf manip sort tail params aliasing copyto errors groupa`
   - `python test/oracle/gen_index_oracle.py` (all three index corpora; seed pinned 20240626)
   - `python test/oracle/fuzz_random.py 1234 2000 random_smoke.jsonl`
   - `dotnet run test/oracle/gen_decimal_oracle.cs` (the 12 `decimal_*` tiers)
   - After ANY regen: `git diff --stat -- test/NumSharp.UnitTest/Fuzz/corpus` and confirm ONLY the
     tiers you intended changed (regenerating an untouched tier must produce a zero diff — use
     this as a determinism self-check). Then `dotnet build` (copies corpus to test output) before testing.
6. **Commits**: commit early per completed task, own files only, extensive messages explaining
   what/why/evidence. Prefix by workstream: `fuzz(gate): …` (BUGS), `fuzz(oracle): …` (GAPS),
   `docs(fuzz): …` (DOCS). **Never** append "Generated with Claude Code" or any `Co-Authored-By`
   footer. **Never push. Never create GitHub issues** — draft issue bodies into the Bug Ledger instead.
7. **Plan updates**: after finishing each task, update your Status section table (`[ ]`→`[x]`,
   plus a one-line result). Record every product bug you find in the **Bug Ledger** (§6) whether
   you fix it or carve it. The plan must always reflect reality — it is actively maintained.
8. **Carve-vs-fix policy**: when a new corpus cell or a tightened excuse turns the gate red, you
   have three legal moves, in preference order:
   (a) **fix** the product bug in `src/NumSharp.Core` if it is small and you fully understand it
   (≤ ~1 h effort), with the corpus cell staying green as the proof;
   (b) **carve** the cell out of the green corpus (explicit comment at the carve site) + pin the
   bug as an `[OpenBugs]` test + Bug Ledger entry;
   (c) **excuse** in `MisalignedRegistry` ONLY for intended/documented differences — never for a
   new unexplained divergence, and always scoped to the exact (op, dtype, kind) cell.
   Silent skips are forbidden.
9. **InternalsVisibleTo** for quick probe scripts (dotnet_run):
   `#:project <worktree>/src/NumSharp.Core` + `#:property AssemblyName=NumSharp.DotNetRunScript`
   + `#:property AllowUnsafeBlocks=true` + `#:property PublishAot=false`.

### Baseline numbers (committed corpus, as of branch point — recount at the end)
Total **70,442** cases / **41** files. Op corpus (non-index, non-decimal) **57,494**; index
**12,369** (curated 2,265 + dtype 104 + random 10,000); decimal **579** (12 tiers); Char woven
**3,752** across 13 files.

---

## 1. Findings inventory (F1–F31)

Severity: 🔴 gate integrity / crash · 🟠 real coverage hole · 🟡 narrower-than-documented · 🔵 docs.
"≈" line numbers = at branch point; they will drift as edits land.

### A. False-premise exclusions — **[probed]** against NumPy 2.4.2 (owner: WS-GAPS)

| # | Finding | Where | Probe evidence |
|---|---------|-------|----------------|
| F1 🟠 | `clip` excludes complex128 with comment "complex128 has no ordering (NumPy raises)" — **the comment is false** | `test/oracle/gen_oracle.py` ≈306-310 (`CLIP_DTYPES`) | `np.clip([1+2j,5+1j,-3+0j],0,2)` → `[1.+2.j, 2.+0.j, 0.+0.j]` (works, lexicographic) |
| F2 🟠 | `round_` tier omits complex128, bool, int8, uint16, uint32, uint64 | `gen_oracle.py` ≈1342 (`ROUND_DTYPES`) | `np.round([1.5+2.5j])` → `[2.+2.j]` (rounds re+im, banker's) |
| F3 🟠 | `matmul` tier omits bool | `gen_oracle.py` ≈609 (`MATMUL_DTYPES`) | `np.matmul(bool,bool)` → dtype bool, AND/OR semiring |
| F4 🟠 | `where` cond is ALWAYS bool; NumPy accepts any dtype cond via truthiness | `layout_catalog.py` ≈342-379 (all `WHERE_LAYOUTS`), `gen_oracle.py` ≈419 | `np.where([0,2,0],x,y)` → selects by truthiness |

### B. Dead wiring — looks covered, generates zero cases (owner: WS-GAPS)

| # | Finding | Where |
|---|---------|-------|
| F5 🟠 | `iscomplex`/`isreal` registered in OpRegistry (≈:153-154) and marked `[x]` in COVERAGE_GAPS, but **zero corpus cases exist** (verified by grep across all 41 files). Bug side pinned (`OpenBugs.DtypeCoverage.cs:119,130`); green side (real dtypes, contiguous) unverified | `OpRegistry.cs`, `gen_oracle.py` (no generator) |
| F6 🟠 | Index oracle base recipe `E03` (empty (0,3)) is **dead**: defined in `make_base` (py ≈43-57) and mirrored in C# `Base()` but referenced by 0 curated cases and absent from random `gpool`/`spool` (≈299-300). Empty-array indexing (get AND set) entirely ungated. Also `V0` missing from random gpool (18 curated cases only); `BT` missing from curated | `test/oracle/gen_index_oracle.py` |
| F7 🟠 | Decimal `power` negative exponents are dead code: loop is `e in {0,1,2,3}` while `IntPow` and the `nonzero`-base plumbing support `e<0`. `decimal^-n` never tested | `test/oracle/gen_decimal_oracle.cs` ≈298-306, ≈475 |

### C. MisalignedRegistry excuses broader than their documented bug (owner: WS-BUGS)

The registry header promises "scoped tightly to the exact (op, dtype) cell so a regression in a
neighbouring cell still fails". These branches break that promise (all in
`test/NumSharp.UnitTest/Fuzz/MisalignedRegistry.cs`):

| # | Finding | Branch (≈line) | Correct scope |
|---|---------|----------------|---------------|
| F9 🔴 | sum/mean/std/var/prod Value divergence excused with **no dtype scoping** — a garbage `sum(int32)` would pass as "summation precision". Integer/bool summation is exact | ≈259-261 | float-family result only (Half/Single/Double/Complex) |
| F10 🔴 | Complex binary blanket: ANY Value divergence, ANY magnitude, ANY 2-operand op with complex result. README claims add/sub/mul are bit-exact, yet a gross complex `add` regression would be excused (contrast the correctly 2-ULP-capped `divide` branch ≈:59) | ≈146-147 | add/subtract → ULP-capped; multiply → cancellation-documented scope; power → documented gross-edge scope (F5-complex); nothing else |
| F11 🟠 | `cumprod` dtype excuse unscoped — documented bug is the **size-≤1** fast path only | ≈158-159 | operand element-count ≤ 1 |
| F12 🟠 | `modf` threw excuse unscoped — documented bug is float16/integer input; a `modf(float64)` throw would be excused | ≈124-125 | operand dtype ∉ {float32,float64} |
| F13 🟠 | Hyperbolic/inverse-trig/angle threw excuse unscoped by dtype — documented bug is Half-promoting inputs; `sinh(float64)` throwing would be excused | ≈281-285 | operand dtype ∈ {bool,int8,uint8,float16}, plus deg2rad/rad2deg×complex128 |
| F14 🟠 | `isclose` value excuse unscoped — documented bug is F-contiguous/complex pairing only | ≈169-170 | at least: some operand complex128 |
| F15 🟡 | Complex reduce/scan excuse documented as *NaN ordering* but never checks the diffs involve NaN | ≈338-340 | require a NaN token in expected or actual diffs |

### D. Untracked crash + harness robustness (owner: WS-BUGS)

| # | Finding | Where |
|---|---------|-------|
| F16 🔴 | `np.invert(float)` executes an **illegal CPU instruction** (ExecutionEngineException — kills the test host). Documented ONLY in a Python comment; deliberately excluded from the errors tier; **no [OpenBugs] pin, no issue, no guard**. NumPy raises a clean TypeError | `gen_oracle.py` ≈1210-1213; kernel dispatch in `src/NumSharp.Core` (locate: invert/BitwiseNot path) |
| F26 🟡 | `RunCorpus` asserts only `Count > 0` — a regeneration that silently truncates a tier to 10 % would pass | `FuzzCorpusTests.cs` ≈254 |

### E. Structural coverage gaps (owner: WS-GAPS, bounded)

| # | Finding | Where |
|---|---------|-------|
| F8 🟡 | "Char woven into every tier" is overstated: present in 13 files (3,752 cases) but `char_tier()` supports only 13 of ~24 modes. Absent from: **where, place, logic (extrema/isnan/isfinite/logical_*/isclose), matmul/dot/outer/trace/diagonal, rounding, nanreduce, modf, params, aliasing, copyto, errors, groupa**. Meaningful & missing: where(char), maximum/minimum(char), matmul(char), copyto(char), round_(char) | `gen_oracle.py` ≈1688-1723 |
| F17 🟡 | Errors tier: only 10 curated specs; pass = NumSharp throws *anything* (type parity not asserted). All other generators SKIP NumPy-raise cases, so e.g. `less(complex)` TypeError, min/max-of-empty, `floor(complex)` have no error-parity assertion at all | `gen_oracle.py` `gen_errors` ≈1194 |
| F18 🟠 | Random fuzzer + nightly soak envelope: **7 dtypes** (missing float16, int8, int16, uint16, uint32, uint64 — exactly the dtypes where the W1 widening found real bugs) and **4 op kinds** (unary/binary-arith(4)/comparison/where). No reductions, scans, astype, matmul in the random space. The "1M cases/night" soak (5 seeds × 200K, `.github/workflows/fuzz-soak.yml`) explores this narrow envelope | `test/oracle/fuzz_random.py` ≈31-32, `gen_random` |
| F19 🟡 | `REDUCE_LAYOUTS` omits every positive-offset slice layout (`simple_slice_offset_1d`, `sliced_composed`, `zerod_from_index`); offset reductions reached only via `negstride_2d_offset`. (The W9-B `repeat` bug was precisely an offset bug) | `gen_oracle.py` ≈192-199 |
| F20 🟡 | NumPy sort tier: distinct values only — **no NaN sorting** (NaN-to-end IS contractual in NumPy) and no strided/offset operands (the decimal sort tier covers strided; the NumPy one doesn't) | `gen_oracle.py` `gen_sort`/`gen_argsort` |
| F21 🟡 | trace/diagonal: contiguous 2-D only; no offset/strided operands, no axis1/axis2/offset params | `gen_oracle.py` `gen_trace_diag` |
| F22 🟡 | matmul: C/F layouts only — no negative-stride operands, no k=0 (empty inner dim) case | `gen_oracle.py` `gen_matmul` |
| F23 🟡 | Decimal tiers: reduce/scan/stat are **axis=None only** (no axis/keepdims/ddof); no argmax/argmin/all/any/count_nonzero; **no empty decimal anywhere** (`sum(empty)=0m`, `prod(empty)=1m` ungated); astype only ↔ {int32,int64,float32,float64}; pair layouts miss `pp_contig_strided` + `pp_broadcast_col`; 13 of 26 single layouts | `gen_decimal_oracle.cs` |
| F24 🟡 | Index oracle: setters are int64-base/int64-value only (no cross-dtype cast-on-set); the dtype sweep (`index_dtype`) is getter-only; exception TYPE parity not compared (which-side-raised only — by design, but undocumented) | `gen_index_oracle.py` |
| F25 🟡 | MetamorphicTests: dtypes limited to Int32/Int64/Single/Double(+Byte/UInt32) — no Half/Complex/Decimal/Char/bool; contiguous layouts only | `MetamorphicTests.cs` ≈47-50 |
| F31 🔵 | `gen_oracle.py` usage/error string omits modes `rounding` and `groupa` (both exist and are wired) | `gen_oracle.py` ≈1855 |

### F. Doc rot (owner: WS-DOCS)

| # | Finding | Where |
|---|---------|-------|
| F27 🔵 | "the '44 variations'" — actual is 26 single + 9 pair + 5 where = **40** (recount AFTER WS-GAPS lands new layouts) | `layout_catalog.py:2`, worktree `.claude/CLAUDE.md` (fuzz section) |
| F28 🔵 | Fuzz README self-contradicts on decimal: "579 cases" (≈:20, correct at branch point) vs "302 cases, all green" (≈:105, stale) | `Fuzz/README.md` |
| F29 🔵 | CLAUDE.md corpus counts stale: "~68K cases / 40 tiers … op corpus ~53K" — actual at branch point: 70,442 / 41 files / ~57.5K op. Also its regenerate-mode list omits `rounding` + `groupa` | worktree `.claude/CLAUDE.md` |
| F30 🔵 | Comment says "all 25 single-array layouts" — there are 26 | `FuzzCorpusTests.cs:70` |

---

## 2. WS-BUGS — gate tightening, crash fix, harness robustness (agent: `fuzz-bugs`)

**Mission**: make the FuzzMatrix gate actually FAIL on regressions in currently-correct cells,
fix or pin every real bug the tightening exposes, and fix the F16 crash.

**Files owned**: `Fuzz/MisalignedRegistry.cs`, `Fuzz/FuzzCorpusTests.cs` (code), `Fuzz/BitDiff.cs`
(if helpers needed), NEW `test/NumSharp.UnitTest/OpenBugs.FuzzGate.cs` for pins, and any
`src/NumSharp.Core/**` files needed for product fixes.

Method for every task: tighten the branch → build → run the affected tier(s) → triage every new
red cell per the carve-vs-fix policy (§0.8) → record in Bug Ledger → commit.

- [ ] **B1 (F9)** Scope the reduction-precision excuse (≈:259) to float-family result dtypes
      (`tc ∈ {Half, Single, Double, Complex}`). Run `Reduce`, `Params`, `Tail`, `NanReduce` tiers.
      Any integer/bool cell that now fails is a REAL bug (integer reduction must be exact/modular)
      — fix or carve+pin each.
- [ ] **B2 (F10)** Dismantle the complex-binary blanket (≈:146). Replacement structure:
      `divide` keeps the existing 2-ULP branch (≈:59); `add`/`subtract` → excuse only within
      2 ULP; `multiply` → keep a scoped excuse for the documented catastrophic-cancellation
      regime (justify the detection you choose in a comment — e.g. cancellation test or a
      documented per-op unbounded excuse citing task #12); `power` → scoped excuse citing the
      documented gross inf/NaN edge (Phase-1 F5); everything else complex-binary → NOT excused.
      Run `Binary_Arith`, `Binary_DivModPower`, `Logic`, `Aliasing`, `FuzzRandom`. Triage reds.
- [ ] **B3 (F11)** Scope cumprod dtype excuse to operand element-count ≤ 1 (0-d counts as 1).
      Run `Scan`. A full-size cumprod NEP50-widening miss is a real bug (ReduceCumMul) — fix or pin.
- [ ] **B4 (F12)** Scope modf threw excuse to operand dtype ∉ {float32, float64}. Run `Modf`.
- [ ] **B5 (F13)** Scope hyperbolic/inverse-trig/angle threw excuse to operand dtype ∈
      {bool, int8, uint8, float16}, plus (`deg2rad`|`rad2deg`) × complex128. Run `UnaryExtra`.
- [ ] **B6 (F14)** Scope isclose excuse to cases with a complex128 operand. Run `Logic`.
      Non-complex isclose failures = real bug — fix or pin.
- [ ] **B7 (F15)** Complex reduce/scan excuse (≈:338): additionally require
      `diffs.Any(d => d.Expected.Contains("NaN") || d.Actual.Contains("NaN"))`. Run `Reduce`, `Scan`.
- [ ] **B8 (F16)** Fix the `invert(float)` illegal-instruction crash: add a dtype guard on the
      invert/BitwiseNot dispatch path in `src/NumSharp.Core` so float/complex/decimal input throws
      a clean exception (NumPy raises TypeError "ufunc 'invert' not supported…"). Add a normal
      (non-OpenBugs) regression test asserting `np.invert(np.array(new double[]{1.5}))` throws
      cleanly (and does NOT crash the host). **Then flip the "B8 DONE" flag in your Status section
      — WS-GAPS G13 depends on it** (they add the errors-tier spec afterwards). If the fix turns
      out large (> ~2 h), STOP, pin what you can, draft a GH-issue body in the Bug Ledger, mark
      B8 as `deferred(reason)` so G13 knows to skip the invert spec.
- [ ] **B9 (F26)** Add per-corpus minimum-case-count floors to `RunCorpus` (a static
      `Dictionary<string,int>` file→floor at ~80 % of current committed counts; unknown files → 1).
      Note in your Status section that WS-DOCS re-checks floors after WS-GAPS regenerations.
- [ ] **B10** Final sweep: run the FULL gate (`TestCategory=FuzzMatrix&TestCategory!=OpenBugs`,
      both frameworks). Paste into your Status section: (a) pass/fail counts, (b) the complete
      list of remaining excuse classes actually hit (the `documented Misaligned divergences
      excused:` console lines per tier) — **WS-DOCS needs this verbatim for the README ledger**.

Acceptance: every excuse branch in MisalignedRegistry names an explicit op set + dtype/kind scope
+ a `[known bug]`/`[documented]` tag; gate green; all discovered bugs fixed or pinned+ledgered.

---

## 3. WS-GAPS — oracle & corpus expansion (agent: `fuzz-gaps`)

**Mission**: close the false-premise exclusions, revive dead wiring, and widen the corpus/soak
envelope — every new cell either green, carved+pinned, or (rarely) registry-excused via a Bug
Ledger entry handed to WS-BUGS.

**Files owned**: everything under `test/oracle/` (all generators), `Fuzz/corpus/**` (all
regeneration), `Fuzz/OpRegistry.cs`, NEW `test/NumSharp.UnitTest/OpenBugs.FuzzGaps.cs` for pins,
`Fuzz/MetamorphicTests.cs` (G17 only), `Fuzz/IndexOracleTests.cs` (G15 only).
**NOT owned**: `MisalignedRegistry.cs` (if a new cell needs a registry excuse, write the exact
branch you need into your Status section and the Bug Ledger; WS-BUGS applies it).

Workflow per task: probe NumPy first where behavior is uncertain (`python` one-liners) → extend
generator → regenerate ONLY the touched tiers → `git diff --stat` corpus sanity → build → run the
tier → triage per §0.8 → commit generator+corpus+pins together with case counts in the message.

- [ ] **G1 (F1)** clip complex: delete the false comment, add `complex128` to `CLIP_DTYPES`,
      regen `stat`. If NumSharp clip(complex) diverges/throws → carve (explicit comment) + pin.
- [ ] **G2 (F2)** `ROUND_DTYPES` += bool, int8, uint16, uint32, uint64, complex128 (keep the
      float16-fractional and dec=-1 carves; ints are identity at dec≥0). Regen `rounding`.
- [ ] **G3 (F3)** `MATMUL_DTYPES` += bool (NumPy: AND/OR semiring, bool result). Regen `matmul`.
- [ ] **G4 (F4)** Non-bool where cond: add a small generator emitting cond dtypes
      {int32, float64, uint8, complex128} over contiguous + strided triples (x/y stay 2-3 pairs).
      Append into `where.jsonl` (mode `where`). Probe NumPy per combo first; expect NumSharp to
      require bool cond → carve/pin per actual behavior.
- [ ] **G5 (F5)** iscomplex/isreal cases: `gen_unary({"iscomplex": np.iscomplex, "isreal":
      np.isreal}, …)` appended into the `logic` mode. Green corpus = REAL dtypes × contiguous
      layouts (`c_contiguous_1d/2d/3d`, `one_element_1d`, `scalar_0d`). CARVE complex128 input and
      strided/F/transposed real input (both are the documented bugs already pinned at
      `OpenBugs.DtypeCoverage.cs:119,130`) with comments pointing at those pins. Regen `logic`.
- [ ] **G6 (F6)** Index oracle: add curated E03 cases — get: `[]`(empty tuple), `[:]`, `[0]`
      (IndexError), `[:, 1]`, full bool mask (0,3), empty fancy `arr []`; set: `[:]=scalar`
      (no-op OK), `[0]=scalar` (err). Add `"E03"` and `"V0"` to random `gpool`, `"E03"` to
      `spool`; keep `RANDOM_SEED=20240626` (the C# test references that filename). Regen all
      three index corpora; run `Index_Curated` + `Index_Dtype` (Index_Random best-effort — R3
      segfault is known). Report new case counts in Status (WS-DOCS consumes them).
- [ ] **G7 (F7)** Decimal negative power: exponents → `{-2,-1,0,1,2,3}` (nonzero-base plumbing
      already exists; `IntPow` handles neg via `1m/r`). Regen decimal tiers. If NumSharp's
      `DecimalMath.Pow` disagrees with reciprocal-of-product → real bug: fix or carve+pin.
- [ ] **G8 (F23)** Decimal widening (BOUNDED — this list, nothing more): (a) axis reductions
      sum/min/max/mean, axis ∈ {0, last} × keepdims ∈ {F,T}, over `c_contiguous_2d`,
      `f_contiguous_2d`, `strided_2d_cols` (extend the naive oracle with an axis walker);
      (b) `empty_2d` layout + `sum(empty)=0m` / `prod(empty)=1m` flat cases;
      (c) pair layouts `pp_contig_strided`, `pp_broadcast_col`;
      (d) astype decimal↔{bool, int16, uint64} (exact for the pool; skip Half — inexact);
      (e) flat argmax/argmin (→int64), all/any (→bool), count_nonzero (→int64).
- [ ] **G9 (F8)** Char weaving extension — new `char_tier` modes: `where` (x/y char pairs via
      uint16 proxy: (C,C),(C,int32),(float64,C); cond stays bool), `logic` (maximum/minimum/
      fmax/fmin char pairs; isnan/isinf/isfinite + logical_not char unary), `matmul` (char via
      proxy — uint16@uint16→uint16 modular), `rounding` (char identity, dec 0/1/2), `copyto`
      (char overlap + char↔{int32,float64} cross). KEEP the existing carve list (no uint8/bool
      partners, no power/reciprocal/invert on char). Wire into `main()`; regen those tiers.
- [ ] **G10 (F18)** Widen the random fuzzer: `DTYPES` += float16, int8, int16, uint16, uint32,
      uint64 (12 total); kinds += `reduce` (flat sum/min/max/mean/prod, axis=None) and `astype`
      (random src→dst over ALL_DTYPES). Regen `random_smoke.jsonl` (seed 1234, count 2000).
      The Misaligned classifier already covers the documented W1-era bugs; triage anything else.
      Note: `.github/workflows/fuzz-soak.yml` needs no change (it calls this script).
- [ ] **G11 (F20)** Sort/argsort hardening: new generator — float16/32/64 (+complex128 if the
      probe is clean) arrays CONTAINING NaN (NumPy sorts NaN to end — probe complex-NaN ordering
      first and skip if implementation-hairy), plus strided (`[::2]`) and negstride (`[::-1]`)
      operands for sort AND argsort. Append into `sort` mode; regen.
- [ ] **G12 (F19)** `REDUCE_LAYOUTS` += `simple_slice_offset_1d`, `sliced_composed`,
      `zerod_from_index`, `reshape_view_2d`. Regen `reduce` + `nanreduce` (they're big — fine).
- [ ] **G13 (F17)** Errors-tier additions (each must be probed to raise in NumPy): `less(complex,
      complex)` TypeError; `min([])`/`max([])` ValueError; `argmax([])`; `floor(complex)`
      TypeError; `searchsorted(2-D a, v)`. Plus `("invert", [float64])` **ONLY after WS-BUGS
      Status shows B8 DONE** (if B8 is `deferred`, skip it and note why). Regen `errors`.
- [ ] **G14 (F21/F22)** trace/diagonal on strided/offset views (e.g. `a[1:5].T`, `a[:, ::2]`) and
      matmul: one negstride-operand case + the k=0 case `(2,0)@(0,3)→(2,3)` zeros (probe first).
      Regen `matmul`.
- [ ] **G16 (F31)** Fix the `gen_oracle.py` usage string: add `rounding | groupa`.
- [ ] **G15 (F24) [STRETCH]** Cross-dtype index setters: tiny new corpus
      (`index_setter_dtype.jsonl`): float64 scalar into int32 base (truncation), int into bool,
      negative into uint8 — needs a small `IndexOracleTests` runner extension. Only if time allows.
- [ ] **G17 (F25) [STRETCH]** MetamorphicTests: extend dtype axes with Half/Complex/Decimal/bool
      where the invariant genuinely holds; add one strided-view variant per involution.

Acceptance: all non-stretch tasks done; every touched tier green under the gate (with carves
pinned + ledgered); corpus diffs match intent; case-count deltas reported in Status for WS-DOCS.

---

## 4. WS-DOCS — documentation truth & ledger upkeep (agent: `fuzz-docs`)

**Mission**: make every count, claim, and ledger entry in the fuzz docs match the FINAL state of
the branch. You are also the plan's secretary — keep §1 finding statuses coherent (mark each F#
resolved/deferred with a pointer to the commit/task).

**Files owned**: `Fuzz/README.md`, `Fuzz/COVERAGE_GAPS.md`, worktree `.claude/CLAUDE.md` (the
fuzz/corpus paragraphs ONLY), `layout_catalog.py` line 1-14 docstring ONLY, `FuzzCorpusTests.cs`
line ≈70 comment ONLY (single line), this plan's §1/§7 statuses.
**Rule**: never touch code logic. For files owned by others, only the listed line ranges, and
only in Phase 2 (after their Status = DONE); re-read the file immediately before each edit.

**Phase 1 — immediately (not invalidated by others' work):**
- [ ] **D7** Document gate semantics that are currently implicit: in `Fuzz/README.md` — (a)
      error-parity = "NumSharp must throw *something*; exception type/message parity is NOT
      asserted"; (b) index oracle compares which-side-raised, not exception types; (c) the
      documented-divergence logging contract ("excused, logged, never silent").
- [ ] **D3a (F29)** Worktree `.claude/CLAUDE.md`: add `rounding` + `groupa` to the
      regenerate-mode list (this is missing regardless of any other work).
- [ ] Draft (in a scratch section at the bottom of this file) the README divergence-ledger
      rewrite skeleton, to be filled from B10's handoff.

**Phase 2 — AFTER both WS-BUGS and WS-GAPS Status sections read DONE** (poll this file every few
minutes — bash `sleep 180` between reads; hard cap 3 h, then finalize with whatever is done and
mark the rest `provisional`):
- [ ] **D1 (F27)** Recount layouts (`LAYOUTS`/`PAIR_LAYOUTS`/`WHERE_LAYOUTS` in
      `layout_catalog.py` + any GAPS additions); fix "44 variations" in the layout_catalog
      docstring + `.claude/CLAUDE.md`.
- [ ] **D2 (F28)** Fix the Fuzz README decimal-count contradiction (both mentions) with the final
      count from `wc -l corpus/decimal_*.jsonl`.
- [ ] **D3b (F29)** Recount the corpus (`wc -l` all files; `grep -c '"dtype":"char"'` per file)
      and refresh every number in `.claude/CLAUDE.md` + `Fuzz/README.md` (total, tier count, op
      corpus, char woven, decimal, index).
- [ ] **D4 (F30)** Fix the `FuzzCorpusTests.cs:≈70` "25 single-array layouts" comment to the
      final layout count (one-line edit; re-read the file first — WS-BUGS edited it).
- [ ] **D5** Rewrite the README "Documented divergence ledger" table from B10's verbatim
      excuse-class handoff + WS-GAPS' new carve/pin list. Every class in `MisalignedRegistry`
      and every carve must appear; remove rows for branches that were deleted (fixed).
- [ ] **D6** `COVERAGE_GAPS.md`: correct the iscomplex/isreal row (dead→wired w/ carves), update
      the "authoritative covered set", Group D (note what G-tasks closed: clip-complex, where-cond,
      sort-NaN, reduce-offset-layouts, decimal-axis, char modes, soak envelope), refresh statuses.
- [ ] **D8** Final coherence pass on THIS file: every F1-F31 marked resolved/deferred with
      commit refs; write §7 Final Summary (what shipped, what's deferred and why, gate status).

Acceptance: zero stale numbers; ledger == registry+carve reality; a reader of README +
COVERAGE_GAPS gets the true post-branch picture.

---

## 5. Cross-team dependencies & file ownership

| Dependency | Producer → Consumer | Signal |
|---|---|---|
| invert(float) kernel guard | BUGS B8 → GAPS G13 (errors spec) | "B8: DONE/deferred" in WS-BUGS Status |
| New-cell registry excuses (if any) | GAPS → BUGS (owns MisalignedRegistry) | Bug Ledger entry + Status note; BUGS applies |
| Excuse-class list + carve list | BUGS B10 + GAPS → DOCS D5/D6 | verbatim lists in Status sections |
| Final corpus counts | GAPS regenerations → DOCS D2/D3b | GAPS Status posts per-tier counts; DOCS recounts |
| Count floors sanity | GAPS regen → BUGS B9 floors | DOCS flags in D8 if any floor > new count |

**Ownership matrix** (edit a file you don't own ⇒ coordination bug):

| Path | Owner |
|---|---|
| `Fuzz/MisalignedRegistry.cs`, `Fuzz/FuzzCorpusTests.cs`, `Fuzz/BitDiff.cs`, `OpenBugs.FuzzGate.cs`, `src/NumSharp.Core/**` | BUGS |
| `test/oracle/**`, `Fuzz/corpus/**`, `Fuzz/OpRegistry.cs`, `OpenBugs.FuzzGaps.cs`, `Fuzz/MetamorphicTests.cs`, `Fuzz/IndexOracleTests.cs` | GAPS |
| `Fuzz/README.md`, `Fuzz/COVERAGE_GAPS.md`, `.claude/CLAUDE.md` (fuzz paragraphs), doc-only line ranges listed in §4 | DOCS |
| `Fuzz/COMPLETENESS_PLAN.md` (this file) | shared — own Status section + Bug Ledger appends only |

---

## 6. Bug Ledger (append-only; every product bug found goes here)

Format per row: `| L<n> | <op/dtype/layout> | <symptom> | <root cause if known, file:line> | fixed@<sha> / pinned@<test> / excused@<registry-branch> | <found-by> |`

| # | Cell | Symptom | Root cause | Disposition | By |
|---|------|---------|-----------|-------------|----|
| L1 | invert × float/complex/decimal | illegal CPU instruction (ExecutionEngineException), kills test host; NumPy raises TypeError | invert/BitwiseNot kernel emits IL for non-integer input without a dtype guard (locate exact site) | OPEN → B8 | review |

(Draft GH-issue bodies below the table if a fix is deferred — do NOT create issues.)

---

## 7. Status

### Status — WS-BUGS (owner: fuzz-bugs — edit ONLY this subsection)
| Task | State | Result note |
|------|-------|-------------|
| B1 | todo | |
| B2 | todo | |
| B3 | todo | |
| B4 | todo | |
| B5 | todo | |
| B6 | todo | |
| B7 | todo | |
| B8 | todo | **G13 waits on this** |
| B9 | todo | |
| B10 | todo | excuse-class handoff for DOCS goes here |

### Status — WS-GAPS (owner: fuzz-gaps — edit ONLY this subsection)
| Task | State | Result note |
|------|-------|-------------|
| G1 | todo | |
| G2 | todo | |
| G3 | todo | |
| G4 | todo | |
| G5 | todo | |
| G6 | todo | |
| G7 | todo | |
| G8 | todo | |
| G9 | todo | |
| G10 | todo | |
| G11 | todo | |
| G12 | todo | |
| G13 | todo | waits on B8 |
| G14 | todo | |
| G16 | todo | |
| G15 | stretch | |
| G17 | stretch | |

### Status — WS-DOCS (owner: fuzz-docs — edit ONLY this subsection)
| Task | State | Result note |
|------|-------|-------------|
| D7 | todo | Phase 1 |
| D3a | todo | Phase 1 |
| D1 | todo | Phase 2 |
| D2 | todo | Phase 2 |
| D3b | todo | Phase 2 |
| D4 | todo | Phase 2 |
| D5 | todo | Phase 2 — needs B10 handoff |
| D6 | todo | Phase 2 |
| D8 | todo | Phase 2 — last |

### Final Summary (WS-DOCS writes at D8)
_(pending)_

---

## 8. Definition of done (whole branch)

1. `dotnet test --no-build --filter "TestCategory=FuzzMatrix&TestCategory!=OpenBugs"` green on
   net8.0 AND net10.0 in the worktree.
2. Broad sanity `--filter "TestCategory!=OpenBugs&TestCategory!=HighMemory"` green.
3. Every MisalignedRegistry branch scoped (op set + dtype/kind) and mirrored in the README ledger.
4. Every F1-F31 resolved or explicitly deferred-with-reason in §1/§7.
5. Every Bug Ledger row has a disposition (fixed@sha / pinned@test / excused@branch / drafted-issue).
6. All corpus regenerations committed together with their generator changes; counts documented.
7. No commit touches files outside the committer's ownership row; no pushes; no footers.
