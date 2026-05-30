# NumSharp ↔ NumPy 2.4.2 — Parity & Performance Master Roadmap

**North star (from `/np-function`):** every NumSharp `np.*` operation is **bit-identical to NumPy
2.4.2 across the full input space, OR ≥1.5× faster** — with every divergence either fixed or
explicitly documented.

**Principle:** correctness-first. You cannot safely optimize code you cannot prove correct. The
differential fuzzer (Plan A, done) is the regression net the performance work refactors against.

This roadmap consolidates the whole path. Detail for two phases lives in
[`FUZZ_PLAN_NEXT.md`](FUZZ_PLAN_NEXT.md); the bug inventory is [`FUZZ_FINDINGS.md`](FUZZ_FINDINGS.md);
the harness is [`../test/NumSharp.UnitTest/Fuzz/README.md`](../test/NumSharp.UnitTest/Fuzz/README.md).

---

## 0. Baseline — done (Plan A)

- **Differential fuzzer harness:** offline NumPy-oracle corpus → exact view reconstruction
  (broadcast / negstride / offset / 0-D / empty) → bit-exact compare (NaN tokenized) →
  `MisalignedRegistry` classifier (intended divergences excused + logged, never silent).
- **6 op tiers (T1–T6):** cast, binary arith (NEP50), comparison, unary, reductions, where/place.
  13 dtypes × 26 single-array + 9 pairwise + 5 triple layouts × edge values. **15,205 corpus cases.**
- **Seeded random fuzzer + element-wise shrinker;** CI gate every push/PR + nightly soak.
- **Findings:** 22 — 1 fixed (`complex→bool`), 18 documented bugs (tasks #7–#12), 2 intended
  `[Misaligned]`, 1 scoping. Suite green (9,421 / 0).
- **Coverage today:** deep on element-wise + reduction *value correctness*; thin on op breadth
  (~35% of transformation ops), op parameters, and the iterator-internal dimension.

The five phases below take it to the north star.

---

## Phase 1 — Correctness backlog: fix the 22 findings

Drive each documented **BUG** → fixed. Every fix **removes its `MisalignedRegistry` branch / its
`[OpenBugs]` tag**, which re-arms that path on the bit-exact gate — so a fix can't silently regress.
Read `src/numpy/` for NumPy's implementation of each, fix the kernel to match, re-run the tier.

Grouped by shared root cause (fix once, clear several findings):

| Batch | Root | Clears | Decision |
|-------|------|--------|----------|
| **F1** division-by-zero | integer `÷0`→0, float `//0`→±inf, `reciprocal(0)`→sentinel | #2,#3,#9(reciprocal) | — |
| **F2** NaN semantics | `<=`/`>=` false on NaN; reductions propagate NaN | #6, #11 | — |
| **F3** NEP50 promotion | unary width-based float promotion; reduction accumulator/`→real` dtype | #7, #15 | — |
| **F4** unsupported-path throws | `negative(uint)`, complex axis reduction, complex `where`, `reciprocal` non-contig | #8,#9,#12,#16 | — |
| **F5** complex arithmetic | port NumPy `npy_c{mul,div,sqrt,pow,...}` algorithms | #5,#10,#19,#21 | **implement vs keep ULP-Misaligned** |
| **F6** bool semantics | bool arithmetic = bool (not int); bool min/max axis | #13, #17 | — |
| **F7** size-1 shape | keep `[1]` result, don't collapse to 0-D | #18 | — |
| **F8** summation precision | pairwise sum / two-pass var | #14 | **implement vs document** |
| **F9** representation robustness | honor `Shape.offset!=0` + size-1 strides in where/binary | #22 | **harden vs document (unreachable via API)** |

**Acceptance:** tasks #7–#12 closed, or explicitly converted to `[Misaligned]` with a written
rationale (F5/F8/F9 are the judgment calls). **Note:** Plan A's matrices make each fix a one-line
verification — flip the classifier branch off and the tier proves the fix bit-exact (or shows what's
left).

---

## Phase 2 — Coverage breadth

### 2A — Finish #2, sections C + E (operand flags + composite paths)

Black-box, extends the existing harness with `out=` / `where=` and operand-relationship layouts
(aliased, overlapping, write-masked, out-strided/broadcast/cross-dtype). Full detail in
[`FUZZ_PLAN_NEXT.md` Part 1](FUZZ_PLAN_NEXT.md). Folds error parity for read-only/broadcast writes.

### 2B — Op tiers T7–T15 (the ~75 untested transformation ops)

Each tier = OpRegistry entries + a generator mode + corpus + classifier, exactly like T1–T6.

| Tier | Ops | Notes |
|------|-----|-------|
| **T7** manipulation | concatenate, stack, h/v/d-stack, reshape, ravel, flatten, squeeze, expand_dims, moveaxis, swapaxes, roll, repeat, tile, pad, delete, insert, append, atleast_{1,2,3}d | view-vs-copy semantics; axis/order params |
| **T8** linear algebra | **dot, matmul, outer** | high value — big SIMD kernels; broadcasting matmul stacks |
| **T9** bitwise | and/or/xor, invert, left_shift, right_shift | NumPy overflow-shift semantics |
| **T10** nan-aware | nansum, nanmean, nanmax, nanmin, nanstd, nanvar, nanprod, nanmedian, nan{percentile,quantile} | the NaN-skip counterpart to Phase 1 F2 |
| **T11** cumulative | cumsum, cumprod, diff | NEP50 accumulator; axis |
| **T12** stat | median, percentile, quantile, average, ptp, count_nonzero, clip | interpolation modes for percentile/quantile |
| **T13** binary logic/compare | maximum, minimum, allclose, isclose, array_equal, isnan/isinf/isfinite | NaN tie-breaking in max/min |
| **T14** sorting/searching | argsort, nonzero, searchsorted | stability, side= for searchsorted |
| **T15** multi-output | modf, divmod | tuple results — extend harness to N outputs |

**Acceptance:** every tier bit-exact or `[Misaligned]`; the differential matrix spans the full
supported `np.*` transformation surface (creation/random/IO are out of scope for value-differential).

---

## Phase 3 — NpyIter behavioral parity (#3 / Plan B)

White-box: assert the **iterator's chosen plan** (order, coalescing, buffering, op_axes, ranged),
ported from NumPy's `test_nditer.py` (~106 functions). Full detail in
[`FUZZ_PLAN_NEXT.md` Part 2](FUZZ_PLAN_NEXT.md):
B0 ledger → B1 expose state (`ChosenOrder`/`CoalescedNDim`/`IsBuffered`/`InnerStrides`) →
B2 order+coalescing (highest value — guards the KEEPORDER stride-sort) → B3 buffered casting →
B4 op_axes/remove_axis/ranged → B5 nditer-config error parity. **Directly de-risks Phase 5** (the
perf migration is an NpyIter rewrite; these tests pin its behavior).

---

## Phase 4 — Depth: parameters, shapes, error parity, lifecycle, metamorphic

- **4A — Parameters.** Sweep `order=` (C/F/A/K), `dtype=` accumulator override, `ddof` (std/var),
  and axis variants (middle axis of N-D, **tuple/multiple axes**, negative axis) across the tiers.
- **4B — SIMD-tail & large shapes.** Sizes straddling the V128/V256/V512 boundaries
  (7/8/9, 15/16/17, 31/32/33, …) per op, where kernels switch SIMD↔scalar-tail; plus a few large
  arrays. Catches off-by-one in the unrolled-body + remainder + scalar-tail loop shape.
- **4C — Error parity (test-kind #4).** Today the generators *skip* NumPy-raising cases. Instead,
  record `expected: {raises: "<ExceptionClass>"}` and assert NumSharp raises the matching type
  (overflow on disallowed casts, `int**neg`, axis-out-of-range, broadcast-write-to-readonly,
  empty-min/max). Couples with B5 + C5.
- **4D — Unmanaged lifecycle (test-kind #5, NumSharp-specific).** Assert no leaked
  `UnmanagedMemoryBlock` / pinned `GCHandle` after ops; stress alloc/free under GC; verify view
  aliasing keeps the base alive and `.copy()` detaches. (NumPy has no analog — pure NumSharp risk.)
- **4E — Metamorphic invariants (test-kind #7).** Oracle-free properties: `(a+b)-b ≈ a`,
  `sum(all axes) == flat sum`, `transpose∘transpose == id`, `reshape` preserves C-order data,
  lossless cast round-trips, `sort` idempotent, `argsort` permutes to sorted. Catches whole-class
  bugs the per-case oracle can miss.

---

## Phase 5 — Performance: the ≥1.5×-NumPy mission (#6, the original goal)

With correctness locked or documented, prove/achieve the speed target. This **is** the
`DirectILKernelGenerator` (legacy whole-array) → `ILKernelGenerator` (per-chunk, NpyIter-driven)
migration that CLAUDE.md names as the architectural target — perf and migration are one workstream,
and Plan A's matrices are the safety net so the rewrite can't break parity.

- **5A — Benchmark harness.** Warm, min-of-N (large arrays have ~40% run-to-run noise), per
  `(op, dtype, layout, size)`; NumSharp via `dotnet run` (clear the runfile cache between project
  edits), NumPy via `timeit`. Reuse the cast-benchmark methodology.
- **5B — Perf ledger.** Matrix of `(op × dtype × layout × size)` → ratio vs NumPy. Classify
  ≥1.5× / parity / laggard. Publish `docs/PERF_LEDGER.md`. Expect Decimal/Half/Complex on scalar
  paths to lag (documented in CLAUDE.md) — target the SIMD-capable dtypes.
- **5C — Optimize laggards via the NpyIter migration.** Port families in CLAUDE.md priority order:
  **reductions → binary arith → comparison → unary → scan → copy → multi-output (Modf) →
  selection (Where/Place)**. Each: port `Direct/DirectILKernelGenerator.<X>.cs` →
  `ILKernelGenerator.<X>.cs` (per-chunk signature), route the np.* call through
  `NpyIterRef.Execute(key)`, delete the `Direct/` partial, **re-run that tier's differential matrix
  (must stay green)** + benchmark (must improve). The matrices turn a scary kernel rewrite into a
  verified one.
- **5D — Perf CI gates.** A nightly perf soak + a regression guard: a hot-path op slower than the
  committed ledger by >X% fails. Keeps the 1.5× target from rotting.

**Acceptance:** every SIMD-capable `(op, dtype, layout)` is ≥1.5× NumPy or has a documented reason;
the `Direct/` partials are migrated and deleted; perf gates green.

---

## Cross-cutting

- **CI cadence.** Every push/PR: FuzzMatrix (incl. new tiers/sections) + FuzzRegression. Nightly:
  fuzz soak (millions) + perf soak. Each fix/feature flips its classifier branch and re-arms the gate.
- **Docs upkeep.** `FUZZ_FINDINGS.md` (close entries as fixed), `NPYITER_PARITY.md` (Phase 3 ledger),
  `PERF_LEDGER.md` (Phase 5). Keep the "never silent" invariant: every divergence is fixed,
  `[Misaligned]` with rationale, or a tracked `[OpenBugs]`.

---

## Sequencing & dependencies

```
Phase 1 (fixes) ─┬─ independent of 2/3, but matrices catch fix regressions ── run continuously
Phase 2A (C/E) ──┤  needs out=/where= harness ext
Phase 2B (T7-15) ┤  independent, high parallelism (T8 matmul highest value)
Phase 3 (#3) ────┤  B1 accessors gate B2-B5 ── de-risks Phase 5
Phase 4 (depth) ─┘  4C error-parity couples to B5/C5
Phase 5 (perf) ──── GATED on: correctness locked/documented on the ops being optimized
                    (Phase 1 + Phase 2B hot ops) AND Phase 3 (NpyIter behavior pinned)
```

**Recommended order (value-weighted):**

1. **Phase 1 F1–F4, F6, F7** — the unambiguous bugs (div-by-zero, NaN, NEP50, throws, bool, shape).
2. **Phase 2B T8 (matmul/dot) + T7 (manipulation)** — biggest untested op surface.
3. **Phase 3 B1+B2** — iterator state + order/coalescing (guards the perf refactor).
4. **Phase 2A (C/E) + Phase 4C (error parity)** — operand flags + raise-parity.
5. **Phase 1 F5/F8/F9 decisions** — implement-or-document the algorithmic/representation calls.
6. **Phase 5** — the performance mission, family by family, matrices as the net.
7. **Phase 2B remainder + Phase 4A/B/D/E** — fill out breadth, params, shapes, lifecycle, metamorphic.

## Effort shape

| Phase | Relative size | Yield | Gate |
|-------|---------------|-------|------|
| 1 — fix findings | M (per batch S–M) | high (closes known bugs) | — |
| 2A — C/E flags | M | high (aliasing/overlap/mask bug surface) | out= harness |
| 2B — T7–T15 ops | L (per tier S–M) | high (×2 the tested op surface; matmul) | — |
| 3 — NpyIter behavior | M–L | high (de-risks perf; closes #3) | B1 accessors |
| 4 — depth | M | medium (params/shapes/error/leak/metamorphic) | — |
| 5 — performance | L | **the mission** | 1+2B+3 on hot ops |
