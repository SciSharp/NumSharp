# NpyIter Reduction Parity + Fusion — Execution Plan

**Branch:** `nditer` · **Target:** NumPy 2.4.2 parity-or-better · **Owner:** this effort

Status legend: ☐ todo ◐ in-progress ☑ done (gated)

---

## Status & findings (live, post-Phase-0/1/2/3/4, Phase 5a)

**Phases 0–4 ☑, 5a ☑, 5b ☒ (PROVEN non-viable — see below), 6 ◐ (step 1: Double Sum/Mean shipped on the per-chunk SIMD path).** Build green; full suite **9790 passed / 0 failed / 11 skipped** (net10.0, excl OpenBugs/HighMemory); correctness matrix (C/F/T/sliced/3-D, axis 0/1/2, keepdims, out=, NaN) matches NumPy 2.4.2. Both 5b and 6 were re-decided by direct PoC measurement (not reasoning) per the "prove it" directive: 5b is genuinely impossible to make fast-and-correct; 6's "regression" was a JIT-tier + back-to-back-harness artifact and the per-chunk model is at Direct parity.

**Phase 5a (axis-aware fused reductions) — `np.evaluate(Sum(a*b, axis:k))` one pass, no temp.**
Extended the IL-emitted `ReduceNode` to carry an `axis`/`keepdims` and emit a new
`CompileAxisReduceKernel`: operands `[inputs…, output]`, the child elementwise tree evaluated
per element exactly as the flat path, folding into the output operand with two runtime
branches — **pinned** (reduce axis inner; the flat 4-way-unrolled fold, writing the output
slot) and **slab** (kept axis inner; `out[c]=fold(out[c], expr)`). Driven by an
(nIn+1)-operand REDUCE iterator (`DefaultEngine.EvaluateAxisReduce`): seed identity →
`ForEach` → mean-divide → cast acc→result → keepdims. New API `NpyExpr.{Sum,Prod,Min,Max,Mean}(x, axis, keepdims=false)`.

| 10M, a*b reduce | fused | unfused np.sum(a*b) | NumPy | fused vs NumPy | fused vs unfused |
|---|---|---|---|---|---|
| sum axis0 | 7.9 ms | 19.0 ms | 21.2 ms | **0.37×** | 2.4× |
| sum axis1 | 6.7 ms | 17.8 ms | 23.9 ms | **0.28×** | 2.7× |
| mean axis0 | 7.3 ms | 19.4 ms | 21.2 ms | **0.34×** | 2.7× |
| mean axis1 | 6.5 ms | 17.7 ms | 23.1 ms | **0.28×** | 2.7× |

Correctness pinned vs NumPy + unfused across dtypes (f64/f32/i32) × ops (sum/prod/min/max/mean)
× axes × layouts (C/T/F) × keepdims (`evaluate_axis_reduce.cs`; 4 new MSTest cases). C-contig
uses identity iteration order (contiguous inner); transposed-fused works correctly (axis
ordering for fused inputs is a later opt — the no-temp win dominates regardless).

**Phase 5b — PROVEN NON-VIABLE (measured, not assumed).** A direct PoC settled it
(`benchmark/poc/phase5b_proof.cs`, 10M flat var vs a high-precision reference):

| algorithm | benign (mean=0) | adversarial (mean=1e8) | time |
|---|---|---|---|
| one-pass naive (sum+sumsq) | relerr 1e-13 ✓ | **relerr 2160× ✗** (val 180, true 0.083) | 3.98 ms |
| one-pass Welford (stable) | relerr 9e-14 ✓ | relerr 5e-9 ✓ | **37.2 ms** |
| **np.var two-pass (current)** | relerr 1e-13 ✓ | relerr 4e-9 ✓ | **6.53 ms** |

The only *fast* one-pass (naive) catastrophically cancels whenever mean ≫ std (timestamps,
prices, sensor baselines — real data); the only *stable* one-pass (Welford) is **5.7× slower**
(scalar, no SIMD). Two-pass is already the optimum. Disqualified by measurement.

**Phase 6 — PROVEN VIABLE; step 1 shipped (Double Sum/Mean).** The earlier "would regress /
scalar generic loop" claim was wrong on two counts, both caught by direct measurement:
1. **The kernels are SIMD, not scalar.** `ILKernelGenerator.SimdSumSameType<T>` is one generic
   `Vector256<T>` body (8-way pinned horizontal + 4-way slab) that monomorphizes to the same
   machine code the Direct path emits per dtype.
2. **The "8–12× slower" was a measurement artifact.** `[opt] core=True` only checks
   *NumSharp.Core*; the PoC's *script-assembly* kernels JIT at tier-0 unless they carry
   `[MethodImpl(AggressiveOptimization)]`. With it, the micro-probe (`phase6_micro.cs`) shows
   **delegate-per-row == inlined == monolithic, exactly** (1M 0.146 ms, 10M 3.56 ms). The
   decompose (`phase6_decompose.cs`) confirms iterator construction (1.3 µs) and the empty-kernel
   drive (17 µs) are free — 100% of any gap is kernel codegen, not the per-chunk model.

   Clean isolated measurement of the **shipped** path (routed Double sum/mean, `phase6_*.cs`):

   | 10M, vs NumPy | sum C | sum T | mean C | mean T |
   |---|---|---|---|---|
   | axis0 | 1.00× | **0.51×** | 1.02× | **0.58×** |
   | axis1 | **0.56×** | 0.96× | **0.61×** | 1.08× |

   ≤1.08× NumPy in the memory-bound regime (DOD pass), often ~2× faster on contiguous-reduce
   axes; and **parity with the old Direct path within noise** (3.26 vs 3.18 ms, sometimes
   faster). The "~12% vs Direct" the back-to-back harness reported is the SAME 256 MB
   memory-pressure artifact that produced the original bogus "19–64×" — clean per-case isolation
   is the truth.

**Scope shipped:** Double **and float32** **Sum** and **Mean** routed through the per-chunk path
(`UseNpyIterReduce`), the PINNED orientation now using a **pairwise leaf** (below). **Deferred**:
- **integer Sum/Prod** — NEP50 widening (int32→int64) breaks single-`Vector256<T>` lanes; needs
  a widening accumulator (the scalar `CreateTypedReduceKernel<TIn,TAccum>` exists as fallback).
- **Min/Max** — need NaN-aware SIMD (the `v != v` mask trick) before routing.
Direct keeps serving all of those (its kernels return for non-routed keys), so no regression.

**Pairwise summation — NumPy's add-reduce core, ported onto NpyIter (unblocks float32).**
NumPy's `pairwise_sum` (loops_utils.h.src) backs the entire additive-float family (sum/mean/var/
std/average/nan*) via `add.reduce`. The two NpyIter reduce orientations map onto NumPy's two
behaviors exactly: **PINNED** (reduced axis is the contiguous inner loop) → NumPy folds the stripe
with `pairwise_sum`; **SLAB** (kept axis inner) → NumPy accumulates rows sequentially. NewReduce's
stride-ordering already produces those orientations, EXTERNAL_LOOP delivers the full stripe as
`(ptr, count, stride)`, and the +0 identity seed + slot-accumulate reproduce NumPy's
`*acc += pairwise_sum(...)`. So only a kernel was needed: `ILKernelGenerator.PairwiseFold<T>`,
ported 1:1 (n<8 naive seed -0; n≤128 eight accumulators unrolled-by-8 + prefetch + the exact
tree-combine; n>128 split `n2-=n2%8` recurse). **Validated BIT-FOR-BIT vs `np.add.reduce`** across
13 sizes (`pairwise_parity.{cs,py}`) and live (`np.sum`/`np.mean` axis0/1, f32+f64). This is what
made **float32 exact** — its earlier exclusion (10M axis1 maxdiff ≈ 24) was the flat-accumulator
order; pairwise removes it. Perf: PINNED axis at 10M is **0.88–0.92× NumPy (faster)**; the exact
recursive structure costs vs a flat pass at cache-resident 1M (~1.3× NumPy, sub-ms) — the accepted
price of bit-exactness (SLAB unchanged; an explicit Vector256-lane leaf measured identical, so the
clean scalar body is kept). Tests: `PairwiseSumParityTests` (4, incl. the large-value regression).
Note: only the per-chunk path is pairwise; flat `axis=None` float sum still uses `sum_elementwise_il`
(could be routed onto the same leaf later for flat NumPy parity too).

**Parallelism (NpyIter RANGE / PARALLEL_SAFE) — PROVEN 2–6× but DECLINED by design.**
The single-threaded migration is only ~parity because Direct is already SIMD-optimal. The one
NpyIter technique that *beats* the baseline is parallelism: both Direct AND NumPy reduce
single-threaded, so range-splitting the reduce across cores is a real win. Prototype
(`phase6_parallel.cs`, 32-core box, double sum):

| | PAR vs Direct | PAR vs NumPy |
|---|---|---|
| axis1 1M / 10M | 3.8× / 3.7× | 6.5× / 5.9× |
| axis0 1M / 10M | 1.8× / 2.9× | 1.7× / 2.6× |
| axis* 100M | 2.4–2.8× | — |

**Decision: NOT adopted.** Held to NumSharp's stated design principle ("Parallelization is
minimal. Most operations use SIMD") — keeps results bit-stable vs NumPy (parallel partials
shift fp summation order ~1e-9 and would be machine-dependent without a fixed partition) and
avoids a threading dependency in the reduction core. The prototype is kept as evidence only;
if the project ever revises the no-threading stance, this is the lever (size-gated, fixed
partition count for determinism). The 2–6× is left on the table deliberately.

**Phase 4 (strided/transposed) — solved by axis-ordering, NOT gather (better than planned).**
Root cause (diagnosed, not assumed): the reduce path kept the **logical** axis order (inner = last logical axis), so it was fast only when the input was C-contiguous; a transpose's last logical axis is strided → cache-hostile column gather (~16×). NumPy instead orders iteration axes by the input's **stride** (contiguous axis innermost). Fix is **reduce-local** in `NpyIterRef.NewReduce`: order the `op_axes` iteration axes by descending input |stride| (0/broadcast → outermost), mapping the output back to original order. For C-contig this is the identity (fast path untouched); only non-contiguous inputs reorder. No gather, no copy — access becomes sequential, which beats gather (gather still cache-misses). Isolated 10M (vs NumPy, vs the prior strided path):

| transposed case | before | after | NumPy | verdict |
|---|---|---|---|---|
| sum axis0 | 119 ms | **7.0 ms** | 8.6 ms | beats |
| sum axis1 | 130 ms | **7.4 ms** | 7.8 ms | parity |
| mean axis0/1 | 300/130 ms | **6.6/7.7 ms** | 7.7/8.6 | beats |
| min/max axis* | 145–170 ms | **15–17 ms** | 20–22 ms | beats |
| prod axis* | 130–140 ms | **12–17 ms** | 19–25 ms | beats |

(The `reduce_parity_bench.cs` back-to-back harness reported ~47 ms for these — a 256 MB-arrays memory-pressure artifact; the isolated per-case bench with GC between cases is the truth. Same artifact class as the original "19–64×" premise.) Blast radius: `NewReduce` only feeds the NpyIter-routed dtypes (complex all-ops, half-mean, decimal all-ops); numeric stays on the Direct path. Suite green confirms no regression.

**Phase 3 (Half & Decimal) — premise re-validated, then scoped to the real wins:**
- **Decimal: routed ALL ops** — the legacy path is both cache-hostile AND lossy (accumulates through a double bridge). The NpyIter kernels are full-precision Decimal on contiguous stripes: **5–13× faster** (10M sum axis1 172→22 ms; mean axis0 271→54 ms) AND more accurate (new test pins `sum(30×0.1m)=3.0m` exactly). No NumPy reference type.
- **Half: routed MEAN only** — it accumulates in Double (hardware-fast): **10M mean axis0 57.7→15.4 ms (3.7×), axis1 23.8→14.5 ms**. Half sum/prod **must** accumulate in Half to reproduce NumPy's f16 *sequential* rounding (verified: `sum(4096 ones)=2048`, a saturation NOT reproducible with a double/float-tree accumulator) — a serial software-arithmetic chain .NET can't beat the legacy path on for the pinned/last axis (~+30%). So Half sum/prod/min/max stay on the Direct path (no regression). Generic `INumber<T>` kernels (`CreateTypedReduceKernel<TIn,TAccum>`, `ConvIn` same-type fast read) back both Decimal and Half-mean and are ready for Phase 6 numeric.



**Premise correction (measured, `-c Release`, optimizer-verified, vs the same NumPy 2.4.2):**
The "complex sum/min/max/prod axis = 19–64× slow" premise was a **measurement artifact** (Debug build / harness contamination in the original probe). Under clean `-c Release` the *legacy scalar* complex sum/min/max/prod were **already ~1.0–1.4× NumPy**. The NpyIter double-pair path is **neutral-to-better** there (sum 0.75–1.02×; min/prod slightly faster; max within ±6% noise) and—critically—**non-regressing on every layout incl. transposed** (both paths ~130–170 ms at 10M; that's Phase 4).

**The genuine bottleneck was `mean`** (the original 24× headline), via `MeanAxisComplex` allocating an NDArray view + iterator *per output row*. Phase 2 replaces it with the one-pass complex Sum kernel + scalar divide:

| case | before | after | vs NumPy | speedup |
|---|---|---|---|---|
| 10M mean axis0 | 300.3 ms | **7.81 ms** | 1.01× | **38×** |
| 10M mean axis1 | 130.0 ms | **7.06 ms** | 0.82× (beats) | **18×** |
| 1M  mean axis0 | 10.70 ms | 0.31 ms | 1.01× | 35× |
| 100K mean axis0 | 1.18 ms | 0.045 ms | 1.70×* | 26× |

*small-N is iterator-construction-bound; still 22–26× faster than the old path.

**Bugs collected (out of Phase 1/2 scope — separate paths):**
- **Flat complex `min`/`max` (axis=None, `min_elementwise_il`)** returned `(NaN, NaN)` where NumPy
  returns the NaN-containing element verbatim, e.g. `min([1+1j, nan+0j, 2+2j]) → (nan+0j)`.
  **FIXED** — `Min/MaxElementwiseComplexFallback` now `return v` (the NaN operand) instead of a
  synthesized `(nan,nan)`; the first NaN in iteration order wins, matching NumPy's left-fold.
  Verified vs NumPy 2.4.2 across 8 cases (incl. NaN-in-imag, two-NaN first-wins); regression test
  `ComplexAxisReductionTests.Flat_MinMax_NaN_ReturnsElementVerbatim`; corrected the test that had
  pinned the old buggy `(nan,nan)` (`NewDtypesCoverageSweep_Reductions_Tests.B8_*`). The axis path
  (`LexFold`) was already correct.
- **`np.prod` on `decimal` throws `OverflowException`** when the product exceeds decimal's ~7.9e28
  range (both flat and axis paths; .NET `decimal` has no infinity, unlike float/complex → `inf`).
  Pre-existing, not yet addressed — surfaced by the parity bench.

**Remaining phases — re-validate premise before each (don't repeat the artifact mistake):**
- Phase 3 (Half/Decimal): check whether their axis sum/mean are *actually* slow first.
- Phase 4 (strided/transposed): **confirmed genuinely slow** (complex transpose 10M ≈ 130–170 ms on both old & new) — real value, AVX2-gather.
- Phase 5 (fusion): independent value.

---

## 0. Objective & success criteria

Bring **axis reductions** to NumPy 2.4.2 parity-or-better on the **NpyIter per-chunk
architecture** (the migration target), starting with the dtypes that are currently
catastrophically slow (Complex/Half/Decimal: 19–64×), then enable **fused axis reductions**.

**Done =**
1. **Correctness** — every `(op, dtype, axis, layout, keepdims, out=, dtype=)` matches NumPy on
   the full variation matrix (CLAUDE.md DOD), verified against generated NumPy output.
2. **Parity** — each `(op, dtype, axis, size)` is ≤ 1.1× NumPy in the memory-bound regime,
   and beats NumPy on strided + fused; no regression on the already-fast numeric path
   (suite geomean unchanged).
3. **Architecture** — new kernels live in `ILKernelGenerator` (per-chunk, NpyIter-driven),
   driven by the 2-op REDUCE iterator. No new code in `DirectILKernelGenerator`.

**Evidence this is achievable** (measured, `-c Release`, optimizer-verified):
NpyIter 2-op reduce, complex 10M — axis0 SIMD kernel **7.48 ms (0.99×)**, axis1 SIMD kernel
**6.61 ms (0.73×)**; prior `npyiter_parity_poc`: strided sum **1.88×**, fused **2.8–5.4×**,
small-N construction **0.40 µs (won)**.

---

## 1. Architecture (decided)

- **Driver:** 2-op REDUCE iterator `AdvancedNew([in, out], REDUCE_OK | EXTERNAL_LOOP, op_axes)`
  with `op_axes` mapping each reduced axis of `out` → `-1` (stride 0 ⇒ REDUCE). Exactly the
  shipping `np.average` `TryFusedWeightedSum` template. Handles axis / multi-axis / keepdims /
  out= / dtype= / all layouts from ONE construction.
- **Kernel:** one dual-mode per-chunk `NpyInnerLoopFunc` per `(op, inType, accType)`:
  - `outStride == 0` → **pinned**: accumulate the contiguous run into one slot (2-lane re/im
    for complex; `Vector256` horizontal for numeric).
  - `outStride != 0` → **slab**: `Vector256` elementwise fold of the contiguous row into the
    output accumulator.
  - Complex = reinterpret as double pairs (rides the f64 SIMD path). Strided inner = AVX2
    gather (Phase 4). min/max complex = lex compare (scalar). Half/int = wide `<TIn,TAcc>`.
- **Routing:** `DefaultEngine.ExecuteAxisReduction` (the chokepoint) gains a guarded branch to
  the new path; everything else is unchanged. Numeric stays on `AxisReductionSimdHelper` until
  Phase 6.

---

## 2. New / changed files

| File | Change |
|---|---|
| `Backends/Kernels/ILKernelGenerator.Reduction.cs` | **NEW** — `GetReduceInnerLoop(op, inType, accType)` cached per-chunk kernels (pinned+slab); complex double-pair SIMD; identity seeding helpers |
| `Backends/Iterators/NpyIter.Reduce.cs` | **NEW** — `NpyIterRef.NewReduce(in, out, int axis…)` and `(…, int[] axes)` reusable builder (generalizes `np.average`'s private one) |
| `Backends/Default/Math/Reduction/Default.Reduction.Add.cs` | route `ExecuteAxisReduction` → new path for supported `(dtype, op)`; add `ExecuteAxisReductionNpyIter` + `UseNpyIterReduce` |
| `Backends/Default/Math/Reduction/Default.Reduction.Mean.cs` | delete `MeanAxisComplex`; mean = sum-path + divide pass |
| `Backends/Kernels/Direct/DirectILKernelGenerator.Reduction.Axis.cs` | (Phase 6) remove Complex/Half/Decimal arms once migrated |
| `test/NumSharp.UnitTest/Backends/Reduction/{Complex,Half,Decimal}AxisReductionTests.cs` | **NEW** — NumPy-derived matrices |
| `benchmark/poc/reduce_parity_bench.{cs,py}` | **NEW** — live `np.*` parity gate vs NumPy |

---

## 3. Phases (each independently shippable + gated)

### Phase 0 — Builder + kernel scaffolding ☑
- `NpyIterRef.NewReduce(input, output, axis)` (single + multi-axis), reusing the `op_axes`
  construction; unit-test it constructs REDUCE with the right strides on C/F/sliced/transposed.
- `ILKernelGenerator.Reduction.cs` skeleton + `GetReduceInnerLoop` cache (ConcurrentDictionary
  keyed by `ReduceKernelKey(op,inType,accType)`).
- **Gate:** build green; existing tests unchanged; `np.average` still passes (shares template).

### Phase 1 — Complex sum/prod/min/max axis ☑  *(the worst offender)*
- Implement the dual-mode complex kernel (double-pair SIMD slab + 2-lane pinned reduce;
  prod via complex multiply; min/max via `ComplexLexPick`).
- `ExecuteAxisReductionNpyIter`: alloc out, seed identity (0/1/±inf-lex), `NewReduce` + `ForEach`,
  keepdims/out= handling.
- Route `UseNpyIterReduce(Complex, {Sum,Prod,Min,Max}) == true`.
- **Tests:** `ComplexAxisReductionTests` — C-contig / F / `a[::2,:]` / `a[:,::2]` / `a.T` /
  broadcast / 3-D + axis 0/1/-1/multi, keepdims, out=, empty/scalar/1-elem.
- **Gate:** all correct; `complex_reduce_poc` + live `reduce_parity_bench` show ≤1.1× NumPy
  both axes (target ~0.8–1.0×).

### Phase 2 — Complex mean axis ☑
- Delete `MeanAxisComplex` + the B2 route in `Default.Reduction.Mean.cs`; mean = complex sum
  kernel + `DivideArrayByCount<Complex>` (fixes the imaginary-drop by construction).
- **Tests + gate** (mean matrix; NaN/empty → NaN parity).

### Phase 3 — Half & Decimal axis (+ wide accumulator) ☑
- Half: `<Half,double>` kernel (accumulate in double, cast back — NumPy precision parity).
  Decimal: same-type scalar (no SIMD). Route both through the new path for sum/prod/min/max/mean.
- **Tests + gate** (correct + faster than baseline; Decimal already ~5–12× in POC).

### Phase 4 — Strided fast path (axis-ordering, not gather) ☑
- Add gather-based per-chunk for `GATHER_ELIGIBLE` (f32/f64/i32/i64) reusing the
  `npyiter_parity_poc` technique (index vector hoisted); insert-gather/scalar fallback.
- **Gate:** strided layouts (`a[:,::2]`, transpose-strided) ≥ parity (NumPy has no strided
  reduce SIMD → expect 1.3–1.9×).

### Phase 5 — Fusion (axis-aware ☑ 5a / multi-reduce ☐ 5b)
- **5a** Extend `NpyExpr` root `ReduceNode` from flat scalar-slot → per-output accumulator
  (the output operand under REDUCE), reusing Phase-0 `NewReduce`. → `np.evaluate(sum(a*b),
  axis=k)` one pass, no temp. (`DefaultEngine.Evaluate.EvaluateReduce` + `CompileReduceKernel`.)
- **5b** Multi-reduction one-pass: `mean` = sum+count; `var`/`std` = sum+sumsq one pass
  (replaces two-pass `NpyAxisIter.ReduceDouble`).
- **Gate:** fused axis reduce beats unfused NumPy sequence; var/std no precision regression vs
  current tests.

### Phase 6 — (optional) migrate numeric reductions ☐
- Move the numeric SIMD reductions to the per-chunk `ILKernelGenerator` path; delete
  `DirectILKernelGenerator.Reduction.Axis*.cs`. Only if zero regression vs `AxisReductionSimdHelper`.

---

## 4. Test strategy
- Expected values **generated from NumPy** (python snippets in the test headers) → MSTest
  assertions. Cover the variation matrix per CLAUDE.md DOD (layouts, dtypes, edge cases).
- Any temporary gap → `[OpenBugs]` (excluded from CI) with a repro, never a silent skip.
- **Collect every NumSharp bug found during testing and report to the user** (global rule).

## 5. Benchmark gates
- Per phase: re-run the relevant committed POC **and** a new `reduce_parity_bench.{cs,py}` that
  times the **live** `np.sum/prod/amin/amax/mean` for complex/half/decimal across
  `{100K,1M,10M} × {axis0,axis1} × {C-contig, a[::2,:], a[:,::2], a.T}` vs NumPy.
- All timing via `dotnet run -c Release` with the `IsJITOptimizerDisabled` startup guard.

## 6. Risks & mitigations
| Risk | Mitigation |
|---|---|
| Debug-build halves hand-written kernels | `-c Release` + startup optimizer assert in every timing script |
| Buffered REDUCE double-loop is 2-D-only (`NpyIter.cs:1707`) | use **non-buffered** REDUCE+EXTERNAL_LOOP (N-D via `Advance`), the `np.average` path; verify >2-D correctness/perf in tests |
| `out=` overlap/alias | `COPY_IF_OVERLAP` on the iterator |
| NEP50 accumulator dtype (Half/int) | wide `<TIn,TAcc>` kernels |
| Regressing the fast numeric path | route only Complex/Half/Decimal first; numeric untouched until Phase 6 (per-dtype rollback flag in `UseNpyIterReduce`) |
| min/max complex has no SIMD | accept scalar lex (NumPy is also scalar there) |

## 7. Order of execution
`Phase 0 → 1 → 2 → 3 → 4 → 5`, committing after each gate. Phase 6 optional/later.
First commit after Phase 1 (complex sum/prod/min/max at parity) — the highest-value, most-visible win.
