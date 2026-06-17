# NpyIter Reduction Parity + Fusion — Execution Plan

**Branch:** `nditer` · **Target:** NumPy 2.4.2 parity-or-better · **Owner:** this effort

Status legend: ☐ todo ◐ in-progress ☑ done (gated)

---

## Status & findings (live, post-Phase-0/1/2/3/4)

**Phases 0 ☑, 1 ☑, 2 ☑, 3 ☑, 4 ☑.** Build green; full suite **9786 passed / 0 failed / 11 skipped** (net10.0, excl OpenBugs/HighMemory); correctness matrix (C/F/T/sliced/3-D, axis 0/1/2, keepdims, out=, NaN) matches NumPy 2.4.2.

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
- **Flat complex `min`/`max` (axis=None, `min_elementwise_il`)** returns `(NaN, NaN)` where NumPy returns the NaN-containing element verbatim, e.g. `min([1+1j, nan+0j, 2+2j]) → (nan+0j)`. NumSharp synthesizes `(nan, nan)`. The new **axis** path is correct (`min nan axis0 → (nan,0)`); only the flat path diverges.

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

### Phase 5 — Fusion (axis-aware + multi-reduce) ☐
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
