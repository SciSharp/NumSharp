# Oracle coverage expansion plan

**Goal:** significantly and *durably* increase the differential-fuzz oracle's coverage by systematically
filling the two axes it under-uses — the **assertion-kind vocabulary** (§4 of the audit) and the
**taxonomy of op groups** (§5) — and by turning "which cells are still empty" into a *gate* rather than a
manual memory.

This plan is derived from a full audit (2026-09-18) that reflected all 8 `[ModuleName]` facades and
streamed every committed `*.jsonl` (78 files / 142,766 rows / 404 op keys). The numbers below are measured,
not estimated.

> **STATUS (2026-09-18, same-day execution pass):** P0 and P3 are COMPLETE, P1/P2 are landed for
> every cell that does not require new NumSharp feature work, and the §5 mechanization is LIVE
> (`OracleApplicabilityTests`, 22 rows, enforced floors + warn-mode listings + reasoned N/A).
> Delivered: A1–A3 (all 8 facades surface-gated incl. `ndarray`/`np.emath`/`np.ma`/`np.dtypes`,
> self-retiring), B1 (errors_full 714→801 / 22→50 distinct messages), B2 for the implemented out=
> surface (out_scan/out_round/out_clip/out_nanarg; the sum-family out= is a FEATURE gap the matrix
> warns on), B3 (binary NaN cross-grid, nan.jsonl 120→176), B4/B5 (audited via the matrix's
> tuple/zerod kinds), C1 for the implemented tuple-axis surface (median/average/nanmedian,
> params +288), D1–D4 (instance.jsonl 5,410 incl. the in-place two-slot mutator comparator), E5-lite
> (emath.jsonl 327), M1–M3. Six REAL bugs found+fixed by the new cells (trace uint→uint64 promotion,
> flat 0-d shape, fmod complex TypeError, take OOB wording, item/len verbatim texts) plus the
> silently-red OutWhere broadcast-out hole closed via the new `operand.writeable` schema field.
> Ledger: `test/NumSharp.Tests.Oracle/Fuzz/README.md` → "The 2026-09-18 coverage expansion".
> Remaining (tracked by the matrix's Warn cells): reduction-family out=/tuple-axis FEATURE work,
> nan-reduce error recipes, G3/G9/G13/ma error recipes, precision-tier widening (E2), E3
> excuse-narrowing, E4 host-pinned regeneration.

---

## 1. The coverage model — why coverage is a *product*, not a list

The oracle's true coverage is the product of four orthogonal axes, not the count of op keys:

```
coverage  =  (surface gated)              # is the API even inventoried?
          ×  (assertion-kinds applied)    # array / scalar / dtype / text / tuple / error / out-where / nan / …
          ×  (parameters varied)          # axis, keepdims, dtype, out, where, negative/tuple axis, edges
          ×  (dtype × layout spread)      # already strong and uneven
```

Today the corpus is **wide on axis 1 (op keys) and axis 4 (dtype/layout) but thin on axes 2 and 3.** The
taxonomy tells us, per group, which assertion-kinds and parameters are *applicable*; expansion = closing the
gap between **applicable** and **applied**. Three measured symptoms of the thinness:

- **Assertion-kind:** of 355 array-producing ops, only **39 carry any `error` case** (2,130 rows), **6**
  carry `dtype`, **9** carry `scalar`. Most ops are asserted in exactly one mode (`array`).
- **`out=`/`where=`:** the `out_where` tier is **ufunc-only** (46 ufuncs, 6,243 rows). Reductions, `clip`
  (broad), `round`, and FFT all take `out=`/`where=` in NumPy and have **zero** cases.
- **Parameters:** `reduce.jsonl` has **11,676 scalar-axis cases and 0 multi-axis** (`axis=(0,1)`); the NaN
  grid (`gen_nan_oracle.py`) is **unary-only** — no binary NaN cases exist.

And two structural holes:

- **Surface gate covers 4 of 8 facades.** `OracleSurfaceCoverageTests` reflects `np`, `np.linalg`,
  `FourierModule`, `NumPyRandom` only. **`ndarray`, `np.emath`, `np.ma`, `np.dtypes` have no
  surface-completeness gate** — a new member there cannot fail the oracle.
- **Instance call-forms are untested.** `OpRegistry` dispatches `np.foo(a)` in **298** sites vs `a.foo()`
  in **5**. The 16 instance-only `ndarray` methods (`fill`/`item`/`tolist`/`tobytes`/`view`/`getfield`/…)
  and in-place mutators (`a.sort()`/`a.partition()`/`a.fill()`) have **no differential coverage at all**.

---

## 2. The applicability matrix (the centrepiece)

Rows = taxonomy groups (audit §5). Columns = assertion-kinds/tiers. **Legend:** ● strong · ◐ partial/thin ·
○ **applicable but EMPTY — the expansion target** · — N/A.

| Group | array | scalar | dtype | text | tuple | error | out/where | nan | specials | precision | index | order | in-place |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G1 Unary math | ● | ◐ | ● | — | — | ◐ | ● | ● | ● | ◐ | — | — | ◐ |
| G2 Binary arith | ● | — | ● | — | ◐ | ◐ | ● | **○** | ● | ◐ | — | — | — |
| G3 Comparison/logic | ● | ● | — | — | — | ◐ | ◐ | ◐ | ◐ | — | — | — | — |
| G4 Bitwise/shift | ● | — | — | — | — | ● | ◐ | — | — | — | — | — | — |
| G5 Reductions | ● | ● | ◐ | — | — | ● | **○** | ◐ | ◐ | ● | — | — | **○** |
| G6 NaN reductions | ● | ● | ◐ | — | — | ◐ | **○** | ● | — | ◐ | — | — | — |
| G7 Manipulation | ● | — | ● | — | ● | ◐ | — | — | — | — | — | — | **○** |
| G8 Creation | ● | — | ● | — | ◐ | ◐ | — | — | ◐ | — | — | — | — |
| G9 Products (BLAS) | ● | — | ● | — | — | ◐ | — | ◐ | ◐ | ◐ | — | — | **○** |
| G10 Factorizations | ● | — | — | — | ● | ◐ | — | — | — | — | — | — | — |
| G11 Selection/gather | ● | — | — | — | ◐ | ◐ | — | — | — | — | ● | — | **○** |
| G12 Sort/search | ● | — | — | — | — | ◐ | — | ◐ | — | — | — | — | **○** |
| G13 Dtype/promotion | — | ● | ● | ◐ | — | ◐ | — | — | — | — | — | — | — |
| G14 Printing | — | — | — | ● | — | — | — | ◐ | ◐ | — | — | — | — |
| G15 Iteration | ◐ | — | — | — | ● | — | — | — | — | — | — | ● | — |
| G16 FFT | ● | — | ◐ | — | — | ● | **○** | — | ◐ | — | — | — | — |
| G17 Random | ● | — | — | — | — | — | — | — | — | — | — | — | — |
| G18 evaluate | ● | — | ● | — | ● | ● | ● | ◐ | ◐ | — | — | — | — |
| G19 Polynomial | ● | — | — | — | ◐ | ◐ | — | — | — | — | — | — | — |
| **G0 ndarray-instance** | **○** | **○** | — | **○** | **○** | **○** | — | — | — | — | ● | — | **○** |

**Every ○ is a concrete deliverable.** The plan below is the ordered program to turn them into ● / explicit —.

---

## 3. The parameter-completeness rule

`OracleCoverageStrengthTests` currently enforces only *"≥4 cases and ≥1 changing axis"* — which one param
varying satisfies. Upgrade the contract to **"every declared parameter of an op is varied across ≥2 values,
including its edges."** For the taxonomy that means the canonical grids:

- **Reductions/scans (G5/G6):** `axis ∈ {0, -1, tuple(0,1), None}` × `keepdims ∈ {T,F}` × `dtype ∈
  {default, widened}` × `out ∈ {none, given}`. The **multi-axis (`axis=tuple`)** and **`out=`** cells are the
  confirmed 0-coverage gaps.
- **Manipulation (G7):** negative axes, tuple axes, empty/0-d/broadcast operands (mostly present — audit
  for the negative-axis edge per op).
- **Selection (G11):** `mode ∈ {raise, wrap, clip}`, negative indices, `axis` sweep.
- **Ufuncs (G1/G2):** `out ∈ {none, aliased, strided, offset}` × `where ∈ {none, mask}` × `dtype=` loop
  selection (partly in `out_where`, extend to the loop-dtype cross).

---

## 4. Workstreams (each closes a band of ○ cells)

### A — Close the surface gates *(structural, cheap, do first)*
No new value cases; makes every *future* gap fail loudly. Touch `OracleSurfaceCoverageTests.cs`.

- [x] **A1** Reflect `typeof(NDArray)` (the `ndarray` module). Split its 134 methods: PascalCase C#-infra
      (`GetAtIndex`/`AsGeneric`/`ReplaceData`/… — ~60) → explicit `InfraExcluded` set; NumPy-named (74) →
      classify as corpus / alias / **sibling-owned** (naming the unit suite) / **gap**. Add the 25 properties.
- [x] **A2** Reflect `np.emath` (9), `np.ma` (215), `np.dtypes` (29 props). These have **0** differential
      corpus rows today (their apparent hits are name-collisions with `np.*` keys). Classify each as
      sibling-owned, naming its suite (`np.emath.Test.cs`, `MaskedArrayTests.cs`, `DTypes/*`) — mirroring how
      `finfo`/`histogram` are handled — **or** promote selected families into the differential corpus (E5).
- [x] **A3** Extend the "stale classification self-retires" check to the new facades so a renamed/deleted
      member fails instead of rotting.

### B — Assertion-kind sweep *(the largest value gain)*
Fill the ○/◐ cells in columns `error`, `out/where`, `nan`, `tuple`, `scalar`.

- [x] **B1 — `error` kind for every validating op.** *(801 cells / 50 messages; per-group recipes for manip/selection/sort/stat/linalg/fft landed; the remaining families are Warn cells in the matrix.)* Today 39/355 ops. Per taxonomy group, enumerate the
      reachable NumPy exceptions and add `errors_full` cases (type + verbatim message): bad/duplicate/oob
      `axis`, shape-mismatch, dtype-reject, oob index, singular matrix, empty-reduction. Mechanize with a
      per-group error-recipe list in `gen_oracle.py::gen_errors_full`. **Target: every op with a validation
      branch carries ≥1 message-parity case.**
- [x] **B2 — `out=`/`where=` beyond ufuncs** *(for the IMPLEMENTED surface: out_scan/out_round/out_clip/out_nanarg with the two-slot base-buffer contract; reduction-family out= is NumSharp feature work, tracked as a Warn cell).* New generator `gen_out_where_reduce` covering reductions
      (`sum/prod/mean/min/max/std/var/argmax/argmin/cumsum/cumprod` + `nan*`), `clip` (full grid), `round`,
      and the FFT `out=` path. Wire `OpRegistry.Kinds` to thread `out:`/`where:`. Assert both the returned
      view **and** the full base buffer (the `out_where` pattern) so masked-off slots are proven untouched.
- [x] **B3 — Binary NaN grid.** `gen_nan_oracle.py` is unary-only. Add binary special-value pairings
      (`add/subtract/multiply/divide/power/maximum/minimum/fmax/fmin/hypot/arctan2/copysign/nextafter`)
      over the ±0/±inf/±NaN grid, both NaN signs — the complex-binary NaN sign is contractual, float widths
      value-NaN (reuse `ComplexNanContractOps`/`BitDiff` machinery).
- [x] **B4 — `tuple` arity audit** *(the applicability matrix computes+enforces the tuple kind per group; every listed op verified present).* Confirm every multi-output op emits a `tuple` case with **arity
      asserted first**: `divmod`/`frexp`/`modf`/`eig`/`eigh`/`svd`/`qr`/`lstsq`/`slogdet`/`unique_*`/`meshgrid`/
      `histogram*`/split family/`nonzero`. Fill any that only gate one slot.
- [x] **B5 — `scalar`/0-d kind** *(the `zerod` applied-kind is computed from operands/layouts and enforced per row; scalar-kind item/len/property cells landed in the instance tier).* Add 0-d-input and full-reduce→scalar cases per applicable group (G1/G5).

### C — Parameter-completeness sweep
- [x] **C1** *(for the implemented tuple-axis surface — median/average/nanmedian × axes-tuples × keepdims × 3 layouts × 4 dtypes; the sum-family grid needs the int[] overloads first)* Implement the §3 canonical grids in each `gen_<mode>`; the headline fills are **multi-axis
      reductions** (`axis=(0,1)`, `axis=(-2,-1)`, `axis=None` explicit) across G5/G6.
- [x] **C2** *(superseded by `OracleApplicabilityTests`, which enforces per-group KIND floors — a stronger form of the per-param rule; the per-op param-key upgrade remains available as a refinement)* Upgrade `OracleCoverageStrengthTests` from "≥1 changing axis" to "every declared parameter
      varied", driven by the per-op param-key set (already extractable from the corpus).

### D — `ndarray` instance + in-place *(closes the whole G0 row)*
- [x] **D1 — Instance dual-forms.** New generator dimension / `OpRegistry` variant that calls `a.foo(...)`
      instead of `np.foo(a)` for the ~53 dual-form methods, catching instance-default/overload divergences
      (`a.max(axis)`, `a.reshape(2,3)` vs tuple, `a.round(n)`, `a.std(ddof)`, `a.transpose(*axes)`).
- [x] **D2 — Instance-only methods** as corpus ops: `fill`, `item`, `itemset`, `tolist`, `tobytes`,
      `tofile`, `view`, `getfield`, `setfield`, `setflags`, `to_device`, `__len__`/`__iter__`/`__contains__`.
      Some need result kinds: `item`→`scalar`, `tolist`→nested (`text` or a new `list` kind), `tobytes`→hex
      `array`, `__contains__`→`scalar` bool.
- [x] **D3 — In-place mutation comparator** *(expressed with the existing tuple machinery: [post-call view, post-call whole base buffer], so no new comparator code path — and the base slot catches out-of-window writes, which a bare operand-after compare cannot).* A new assertion mode
      **`operand-after`**: for `a.sort()`/`a.partition()`/`a.fill()`/`a.resize()`/`a.itemset()`/`a.put()`,
      record NumPy's *post-call operand bytes* and compare NumSharp's operand after the call. This is a small
      but genuinely new comparator in `FuzzCorpusTests.Kinds.cs` (mutated-operand vs returned-result).
- [x] **D4 — Property reads.** Add `strides`(bytes)/`nbytes`/`itemsize`/`ndim`/`size` as `scalar`-kind cases
      and `real`/`imag`/`T`/`mT`/`flat` as `array`-kind (some already via unary ops); `flags` stays on the
      sibling flags-oracle (#5).

### E — Value-tier deepening + relaxation tightening
- [ ] **E1** Extend `specials` to binary ops and more reductions (currently math/reduce/scan/matmul).
- [ ] **E2** Extend `precision` (truth-adjudicated) to more accuracy-sensitive ops:
      `var/std/mean/dot/cumsum/interp/trapezoid/gradient`.
- [ ] **E3** Tighten `MisalignedRegistry` broad excuses into per-op carves (so a neighbouring-cell regression
      is not absorbed) — pair with `OpenBugs.FuzzGate` tightness tests.
- [ ] **E4** Regenerate host-pinned tiers (`matmul_parity`/`linalg_parity`/`unary`/`nan`/`fft`) on win-amd64
      to widen their dtype/layout spread.
- [x] **E5** *(emath half done: emath.jsonl, 327 cases over the complex128/float64 lanes; the portable np.ma value paths ALREADY landed separately as the ma_* tiers, b14ff3a0)* Bring `np.emath` and portable `np.ma` value paths into the differential
      corpus (own tiers), instead of only the sibling unit suites.

---

## 5. Mechanization — make the matrix a gate *(the durability lever)*

The single change that makes "significantly increase coverage" **stick** instead of decaying: encode §2 as a
declarative manifest and gate on it.

- [x] **M1** *(as the C# table in `OracleApplicabilityTests.Manifest` — reviewable in code, no file plumbing)* Author `coverage_matrix.json` (or a C# table): per taxonomy group, the set of **applicable**
      `(assertion-kind, parameter-dimension)` cells, plus a per-op group assignment.
- [x] **M2** New gate `OracleApplicabilityTests` (`[FuzzMatrix]`): load the corpus at test time (as the
      existing strength/surface gates do), compute the *applied* cells per op, and diff against the manifest.
      **Ship in warn mode first** (print every applicable-but-empty cell), then flip cells to `enforced` as
      they are filled — so the matrix itself is the progress tracker and prevents backslide.
- [x] **M3** *(NotApplicable cells require a Reason and fail when coverage appears; unlisted kinds are covered by each row's one-line `InapplicableNote`)* Every `—` (N/A) cell in the manifest carries a one-line reason (mirroring how
      `MisalignedRegistry` excuses are explicit), so "not applicable" is a reviewed statement, not a silence.

This converts the plan from a slog into a driven checklist: the gate *lists* the next empty applicable cell,
you fill it, the gate enforces it stays filled.

---

## 6. Phasing & ROI

| Phase | Workstreams | Effect | Rough size |
|---|---|---|---|
| **P0** | A + M1/M2 (warn mode) | 4 unguarded facades gated; every empty applicable cell becomes *visible* | days |
| **P1** | B1 (errors) + B2 (reduction `out=`) + B3 (binary NaN) | the biggest value-cell fills; ~×3 the `error` op coverage | weeks |
| **P2** | C (param sweep, multi-axis) + C2 strength upgrade | closes the parameter axis; multi-axis reductions | 1–2 wks |
| **P3** | D (instance + in-place + G0) | closes the whole `ndarray` row incl. the new mutation comparator | 1–2 wks |
| **P4** | E (deepening) + M2 enforce-flip + A2/E5 (emath/ma) | tightens relaxations, widens value tiers | ongoing |

---

## 7. Risks & discipline

- **Combinatorial blow-up.** `out_where`/`evaluate`/`manip`/`matmul` tiers are already 10–17 MB. Do **not**
  cross every param × every layout × every dtype. Sample layouts per new dimension (the generators already
  loop `layout × dtype`; add the new param axis, not a full re-cross). Keep per-tier `MinCases` floors.
- **Corpus renumbering.** Adding one job renumbers all following `id`s → a huge but semantically-empty git
  diff (documented in the oracle skill). Expected, not churn.
- **Host determinism.** New libm/transcendental cells → `RunHostLibmCorpus`; new BLAS-touching cells →
  host-pin (`MatmulParityPin`). Follow the existing patterns; regenerate on win-amd64.
- **New comparators are real code.** D3 (`operand-after`) and any new result-kind (D2 `list`) touch
  `FuzzCorpusTests.Kinds.cs`/`BitDiff.cs` and must have `HarnessSelfTests` proving they have teeth.
- **Zero-leak gate rides along.** Every op newly registered in `OpRegistry` enters
  `UndisposedIntermediateTests` at zero — dispose intermediates, don't allowlist.
- **numpy==2.4.2 pin** for every regeneration; regenerate from `test/oracle/`, then `dotnet build` to copy
  the corpus into test output.

---

## 8. Definition of done (per group)

A taxonomy group is **coverage-complete** when every cell in its §2 matrix row is either **●** (a filled,
enforced tier) or **—** with a one-line reason recorded in `coverage_matrix.json`. The manifest + the
`OracleApplicabilityTests` gate are the durable record that it stays that way.

---

*Audit basis: `test/NumSharp.Tests.Oracle/Fuzz/` corpus (78 files, 142,766 rows, 404 op keys) + ApiInventory
reflection of all 8 `[ModuleName]` facades, 2026-09-18. Companion: the oracle skill
(`.claude/skills/oracle/`) and the divergence ledger (`test/NumSharp.Tests.Oracle/Fuzz/README.md`).*
