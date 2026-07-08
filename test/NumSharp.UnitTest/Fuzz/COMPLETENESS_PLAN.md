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
| L1 | invert × float/complex/decimal | illegal CPU instruction (ExecutionEngineException), kills test host; NumPy raises TypeError | invert/BitwiseNot kernel emits IL for non-integer input without a dtype guard (locate exact site) | **already fixed on branch** — loop-resolution guard `Default.Invert.cs:27-39` (landed with the ufunc out=/where= work) throws NumPy's verbatim TypeError for float/complex/decimal/Half input AND the `~` operator (7 paths probed clean 2026-07-07); pinned@`FuzzGateRegressionTests.Invert_*_ThrowsCleanly` (OpenBugs.FuzzGate.cs, 5 normal tests, green). gen_oracle.py:≈1210 crash comment is STALE — G13 may add the invert errors spec | fuzz-bugs |

| L2 | round_ × bool | `np.round(bool, 0)` → float16 `[0,1]` in NumPy (rint float-tier) but **Double** in NumSharp — result-dtype divergence | `DefaultEngine.Round` decimals==0 path (`Default.Round.cs:25-38`): Boolean fails `IsInteger()` so it skips the int identity-copy, and `ResolveUnaryReturnType` maps bool→Double while the rint tier maps bool→Half | carved from `ROUND_DTYPES` (gen_oracle.py) + pinned@`OpenBugsFuzzGapsTests.Round_Bool_Dtype_Diverges` @ff4b46b3 | fuzz-gaps |
| L3 | round_ × complex128 × decimals≠0 | NumPy rounds re+im via multiply→rint→divide (`round(1.55+2.45j, 1)` → `1.6+2.4j`); NumSharp returns the input **unchanged** (no-op). decimals==0 is correct (banker's, probed) | `RoundDecimalsCore` (Default.Round.cs) never processes Complex components for decimals≠0 | carved to dec=0 in `gen_round` + pinned@`OpenBugsFuzzGapsTests.Round_Complex_NonzeroDecimals_NoOp` @ff4b46b3 | fuzz-gaps |
| L4 | all/any × Half,Complex × offset view | `np.all(b[2:7])`-style raw-offset view: contiguous fast path reads `(T*)nd.Address` for `nd.size` elements WITHOUT `+ shape.offset` → scans the wrong window (all→True where NumPy False); any() identical latent bug (masked by pool). Other dtypes use NDIter (correct) | `Default.All.cs` AllImplHalf/AllImplComplex + `Default.Any.cs` AnyImplHalf/AnyImplComplex — missing the `+ shape.offset` idiom (cf. NonZero.cs:258) | **fixed@7804b2ad** (4 sites); regression proof = the 4 reduce.jsonl simple_slice_offset_1d cells now green; fix REVIEWED-OK by fuzz-bugs (src owner) — independently reproduced pre-fix behavior and confirmed API slices rebase storage (offset stays 0) so no double-count | fuzz-gaps |
| L5 | convolve × complex128 (headline); + latent int64/uint64/decimal/bool & view inputs | `np.convolve(complex)` DISCARDED the whole imaginary dimension (accumulated via `Converts.ToDouble(boxed)` into a double, cast back with imag=0). Hidden because corpus complex pools have imag=0 + the old B2 blanket excused any complex-binary value diff; exposed by the NaN cells (NumPy `NaN*(x+0j)`→NaN+NaNj, NumSharp NaN+0j — groupa convolve/84-86). Latent in the same code: int64/uint64/decimal accumulated through double (rounds >2^53 / loses decimal precision instead of NumPy modular wrap); bool threw NotSupportedException (NumPy: BOOL_dot OR-of-ANDs, probed); raw `Address` walk ignored `Shape.offset`/strides for view inputs (corpus contiguous → ungated) | `NdArray.Convolve.cs` — single generic `ConvolveFullTyped<T>` double accumulator for ALL dtypes | **fixed@737c59d6** — rewritten per NumPy correlate `*_dot` contract (complex: double component sums; ints+Char: one `IBinaryInteger<T>` modular ulong kernel — bit-identical to native-width wrap at any width; Half: float acc; Single: f32 acc; Boolean: OR-of-ANDs w/ early exit; Decimal: decimal acc; sliced/strided/broadcast inputs materialized first). Proof: GroupA convolve cells (i32/f64/u8/c128 × full/same/valid) green incl. the 3 previously-hidden NaN cells | fuzz-bugs |
| L6 | power × complex128 (finite interior) | Complex.Pow (polar exp(w·log z)) vs npy_cpow (repeated-squaring fast path for small integer exponents): finite divergence measured up to ~350 ULP of the affected component (divmod_power corpus), far beyond the "~1 ULP" previously documented; plus the known F5 gross inf/NaN edges (NaN↔0, inf↔NaN both directions) | algorithmic — System.Numerics.Complex.Pow has no integer-exponent fast path; candidate future fix: port npy_cpow's squaring path | excused@registry `complex power ~ULP / gross inf-NaN edge` — scoped to power × Complex × Value, every diff ≤512 ULP of the ELEMENT magnitude or non-finite-involved (sign flips / wrong magnitudes still fail) | fuzz-bugs |
| L7 | std × decimal | Last-significant-digit (28/29-digit) disagreement, e.g. `…4468786` vs `…4468787` (4 decimal_varstd cells). Surfaced by B1 scoping the reduction-precision excuse to float-family — the unscoped branch had been silently absorbing tc==Decimal. var is bit-exact ⇒ divergence is purely sqrt(var) | Two independent, NOT-correctly-rounded decimal sqrts: oracle `gen_decimal_oracle.DecSqrt` (Newton) vs product `DecimalMath.Sqrt`. Probed vs 60-digit truth 2026-07-07: oracle 1 low on cases 325/331, NumSharp 1 high on 335, both fine at their own scale on 327 — NEITHER side systematically correct | excused@registry `decimal std last digit` — scoped to std × Decimal × Value, every diff ≤1 unit in the 28th significant digit (relative 1e-27, `DecimalLastDigitDiff`); self-retiring if either sqrt becomes correctly rounded (optional GAPS improvement: correctly-rounded oracle DecSqrt) | fuzz-bugs |
| L8 | round_ × char | `round_(char)` resolved **Double** instead of the integer identity copy (Char fails `IsInteger()` — same pattern as L2 bool, but for Char the right behavior is unambiguous: NumPy proxy `round(uint16)` → identity uint16) | `Default.Round.cs` decimals==0 + decimals>=0 int-identity branches keyed on `IsInteger()` only | **fixed@1a9cfa9f** (Char routed down the int identity path, both branches); 78 green rounding/char cells are the proof. (Commit message says "Ledger L5" — renumbered here to L8, my L5 landed first; fix REVIEWED-OK by fuzz-bugs) | fuzz-gaps |
| L9 | dot × char × 1-D·1-D | `np.dot(char, char)` (1-D vectors) throws `NotSupportedException: Sum not supported for type Char`; NumPy proxy says uint16-modular inner product. `matmul` 1-D·1-D char and all 2-D+ char cases work | vector-dot reduces via `sum_elementwise_il` with an explicit Char result typecode; the switch (`DefaultEngine.ReductionOp.cs:409`) has no Char arm | carved (`char_tier "matmul"` filters `(dot,(4,))`) + pinned@`OpenBugsFuzzGapsTests.Dot_Char_1D_Throws` @1a9cfa9f. (Commit message says "Ledger L6" — renumbered here to L9) | fuzz-gaps |

(Draft GH-issue bodies below the table if a fix is deferred — do NOT create issues.)

---

## 7. Status

### Status — WS-BUGS (owner: fuzz-bugs — edit ONLY this subsection)
| Task | State | Result note |
|------|-------|-------------|
| B1 | **DONE** @3976b565 | scoped to float-family result tc {Half,Single,Double,Complex}. Surfaced L7 (decimal std last-digit → new tight excuse); Reduce/Params/Tail/NanReduce otherwise green — no integer-reduction bug existed |
| B2 | **DONE** @3976b565 | blanket dismantled → per-op: add/sub ≤2 ULP; multiply ≤16 element-magnitude ULP (cancellation anchor); power ≤512 element-magnitude ULP or non-finite (L6, measured ~350); divide keeps its 2-ULP branch; ALL other complex-binary (matmul/dot/outer/copyto/extrema/concat…) hard-gated bit-exact. Surfaced L5 convolve(complex) imag-drop → FIXED @737c59d6 |
| B3 | **DONE** @3976b565 | cumprod dtype excuse scoped to operand element-count ≤ 1; Scan green (no full-size widening miss existed) |
| B4 | **DONE** @3976b565 | modf threw excuse scoped to dtype ∉ {float32,float64}; Modf green |
| B5 | **DONE** @3976b565 | hyperbolic/inv-trig/angle threw excuse scoped to {bool,int8,uint8,float16} + (deg2rad\|rad2deg)×complex128 (both probed vs NumPy); UnaryExtra green |
| B6 | **DONE** @3976b565 | isclose value excuse scoped to complex128-operand-present; Logic green |
| B7 | **DONE** @3976b565 | complex reduce/scan excuse now additionally requires a NaN token in the diffs; Reduce/Scan green |
| B8 | **DONE** | crash already fixed on branch (guard @ `Default.Invert.cs:27-39`); NumPy-verbatim TypeError verified vs 2.4.2 (f16/f32/f64/c128; +decimal/Half/`~` op probed clean); 5 regression tests pinned in `OpenBugs.FuzzGate.cs` (non-OpenBugs, green). **G13 UNBLOCKED** |
| B9 | **DONE** @e91f7b5a | `MinCases` file→floor map in RunCorpus (~80 % of committed counts 2026-07-07, 39 files; unknown files → 1). DOCS re-checks floors at D8 after GAPS finishes expanding |
| B10 | in progress | interim net10.0 sweep: my state green (38/40 FuzzCorpusTests; the 2 reds are GAPS G9 mid-flight char cells — theirs). Final both-framework sweep + the verbatim excuse-class list lands here once WS-GAPS non-stretch tasks are done |

### Status — WS-GAPS (owner: fuzz-gaps — edit ONLY this subsection)
| Task | State | Result note |
|------|-------|-------------|
| G1 | **DONE** @ff4b46b3 | clip complex128 in: false comment corrected, `stat.jsonl` 4258→4265 (+7). NumSharp bit-exact incl. NaN-poisoned cells — Stat gate green |
| G2 | **DONE** @ff4b46b3 | `ROUND_DTYPES` 7→12 dtypes (+i8/u16/u32/u64 identity, +c128 @dec=0). `rounding.jsonl` 494→832 (+338), gate green. 2 REAL bugs carved+pinned (Ledger L2 bool-dtype, L3 complex-dec≠0-noop) in NEW `OpenBugs.FuzzGaps.cs` |
| G3 | **DONE** @809b35d1 | `MATMUL_DTYPES` += bool (AND/OR semiring, dedicated `_mm_fill` bool branch). +68 cases, bit-exact, Matmul gate green |
| G4 | **DONE** @5e1f0c8e | `gen_where_cond`: cond {i4,f8,u8,c128} × 3 xy pairs × {contig, strided} = +24 into `where.jsonl` (70→94). NumSharp truthiness matches bit-exact (NaN truthy, -0.0 falsy). Gate green |
| G5 | **DONE** @5e1f0c8e | 12 real dtypes × 5 contiguous layouts × 2 ops = +120 into `logic.jsonl` (2099→2219), green. Carves point at existing pins `OpenBugs.DtypeCoverage.cs` (complex input + strided real) |
| G6 | **DONE** @797ac671 | curated +8 E03 get/set (`index_curated.jsonl` 2265→2273), gpool += E03+V0, spool += E03, ND map fixed (V0/E03 were missing). Index_Curated + Index_Dtype gates GREEN. Index_Random host-crash verified PRE-EXISTING (old committed corpus crashes identically — R3), not from the regen |
| G7 | **DONE** @e1e0a0e7 | exponents {-2,-1,0,1,2,3}; `decimal_power.jsonl` 20→30. DecimalMath.Pow matches the exact reciprocal-of-product oracle by VALUE — zero divergence, no carve |
| G8 | **DONE** @e1e0a0e7 | full bounded list green first-run: axis walker (48), empty sum/prod (2), pp_contig_strided+pp_broadcast_col (+28 binary), astype ↔{bool,int16,uint64} (+8), flat argmax/argmin/count_nonzero/all/any (+20). `decimal_reduce` 60→130, `decimal_binary` 98→126, `decimal_astype` 22→30. All 12 Decimal* gates green |
| G9 | **DONE** @1a9cfa9f | char woven into 5 more modes: where +15, logic +284, matmul +64, rounding +78, copyto +18 (=+459). FIX L5 round_(char)→identity (Default.Round.cs); CARVE+PIN L6 dot(char) 1-D (`Dot_Char_1D_Throws`). Gates green |
| G10 | **DONE** @a7be5098 | DTYPES 7→13 (+f16 & narrow ints); kinds += flat reduce (sum/prod/min/max/mean) + astype (src∈DTYPES→dst∈ALL). `random_smoke.jsonl` regenerated @seed 1234/2000. Soak workflow unchanged (calls this script). FuzzRandom gate green |
| G11 | **DONE** @4d194cab | `gen_sort_special` +85 into `sort.jsonl` (417→502): float NaN 1-D/2-D/strided, complex full extended order incl. nan+nanj, strided+negstride × 13 dtypes (bool sort-only, tie guard). Zero carves — gate green |
| G12 | **DONE** @7804b2ad | +4 layouts: `reduce.jsonl` 8708→11256, `nanreduce.jsonl` 6494→8366. Caught REAL bug L4 (all/any Half+Complex ignore shape.offset on the contiguous fast path) — FIXED in src (4 sites, Default.All/Any.cs; note to fuzz-bugs: crossed into src per policy 0.8(a), green cells are the regression proof). Gates green |
| G13 | **DONE** @96af067c | `errors.jsonl` 10→16: +min/max/argmax(empty), floor(complex), searchsorted(2-D), invert(f64) (B8 guard live; stale crash comment removed). less(complex) dropped — NumPy 2.4.2 doesn't raise. Errors gate green |
| G14 | **DONE** @809b35d1 | `gen_matmul_edges` (negstride B + k=0, 4 dtypes, +8) + `gen_trace_diag` strided/offset views (`a[1:5].T`, `a[:, ::2]`, +28). `matmul.jsonl` 858→962, gate green |
| G16 | **DONE** @ff4b46b3 | usage string lists rounding + groupa |
| G15 | **DONE** | cross-dtype setters: 10-case `index_setter_dtype.jsonl` (float→int trunc-toward-zero, int→bool coercion, uint8 np-scalar modular wrap — all probed) + `Index_SetterDtype` gate. NumSharp matches NumPy 10/10 first-run, zero carves. (fuzz-gaps' in-flight work, completed by orchestrator after session-limit cutoff) |
| G17 | **DONE** @99a6fab3 | metamorphic invariants widened to Half/Complex/Decimal/bool/char + strided views (status row reconciled post-cutoff; commit had landed) |

#### WS-GAPS final handoff (for WS-DOCS D2/D3b) — counts measured post-G15 (orchestrator-updated)
- **Full gate: 82/82 FuzzMatrix tests green on net10.0 AND net8.0** (`TestCategory=FuzzMatrix&TestCategory!=OpenBugs`) measured post-G13; +1 test (Index_SetterDtype) green post-G15 → 83 expected in B10's final sweep.
- **Grand total (wc -l over Fuzz/corpus/*.jsonl): 76,139 cases / 43 files** (baseline 70,442 → +5,697).
- Index: 12,387 (curated 2,273 + dtype 104 + setter_dtype 10 + random 10,000). Decimal: 695 across 12 tiers. Op corpus (non-index, non-decimal): 63,057.
- Char woven (grep `"dtype":"char"`): **4,393 lines across 18 files** (was 3,752/13) — new: where 15, logic 284, matmul 64, rounding 78, copyto 18; reduce grew to 804 via the G12 layouts.
- Per-tier deltas: stat 4258→4265 · rounding 494→910 · matmul 858→1026 · where 70→109 · logic 2099→2503 · reduce 8708→11256 · nanreduce 6494→8366 · sort 417→502 · copyto 114→132 · errors 10→16 · index_curated 2265→2273 · decimal_power 20→30 · decimal_reduce 60→130 · decimal_binary 98→126 · decimal_astype 22→30 · random_smoke 2000 (rewritten: 13 dtypes × 6 kinds) · index_random 10,000 (rewritten: E03/V0 pools).
- New carves (for the D5/D6 ledger): round_ bool (L2) · round_ complex dec≠0 (L3) · dot char 1-D·1-D (L6) — all pinned in `OpenBugs.FuzzGaps.cs`; iscomplex/isreal complex+strided carve points at the pre-existing `OpenBugs.DtypeCoverage.cs` pins.
- Product fixes landed by WS-GAPS (src crossover per §0.8(a), green cells as proof): L4 all/any Half+Complex offset (@7804b2ad) · L5 round_(char) identity (@1a9cfa9f).

### Status — WS-DOCS (owner: fuzz-docs — edit ONLY this subsection)
| Task | State | Result note |
|------|-------|-------------|
| D7 | done | README "Gate semantics" section added (error-parity = throw-anything, grounded in FuzzCorpusTests.cs:274-287; index oracle = which-side-raised only, IndexOracleTests.cs:175-191; excused-and-logged contract restated). Bonus same-class doc-rot fix: README regenerate block gained `gen_index_oracle.py` + the FULL gen_oracle mode list incl. `rounding`/`groupa` |
| D3a | done | worktree `.claude/CLAUDE.md:579` mode list += `rounding` (after matmul) + `groupa` (after errors) — now matches §0.5 |
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

---

## 9. SCRATCH — WS-DOCS: README divergence-ledger rewrite skeleton (D5 working draft)

> Owner: fuzz-docs. Filled from B10's verbatim excuse-class handoff (WS-BUGS Status) + WS-GAPS'
> carve/pin list. This section is DELETED when D5/D8 land the final README.

Target structure replacing README § "Documented divergence ledger (Misaligned / known bugs)":

**Table 1 — live `MisalignedRegistry` excuse branches** (one row per branch, post-B1–B7 scope):

| Excuse class (registry reason string) | Scope (op set × dtype/kind, as tightened) | Why excused (intended / known bug) | Task/issue |
|---|---|---|---|
| _(from B10 handoff: every `documented Misaligned divergences excused:` class actually hit, verbatim)_ | | | |

**Table 2 — corpus carves** (cells deliberately absent from the green corpus, each pinned):

| Carve site (generator + comment) | Cell (op × dtype × layout) | Pin (`[OpenBugs]` test) |
|---|---|---|
| _(existing: char power/reciprocal/invert/bool-partner carves, clip-bool noncontig, round dec=-1 + f16-fractional, trace-unsigned, iscomplex/isreal complex+strided, unique raw-offset)_ | | |
| _(new from G1-G14: fill from WS-GAPS Status/commits)_ | | |

**Rules the rewrite must satisfy:**
- every LIVE MisalignedRegistry branch appears exactly once, wording matched to the registry
  comment text (no drift); scope column shows the post-tightening (op, dtype, kind) bounds;
- rows whose branch was DELETED (bug fixed) are removed, or kept once as **FIXED** with the
  fixing commit sha;
- every carve in gen_oracle.py / gen_decimal_oracle.cs / char_tier / layout exclusions appears
  in Table 2 with its `[OpenBugs]` pin name;
- the existing "Char & Decimal dtype coverage" README subsection merges into Table 2 (single
  source for carves) — decimal count fixed by D2 at the same time;
- cross-check: no excuse class printed by the final full-gate run (B10) is missing from Table 1,
  and no Table 1 row corresponds to a class that can no longer fire.
