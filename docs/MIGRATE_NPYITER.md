# MIGRATE_NPYITER — `np.where` → NpyIter multi-operand per-chunk kernel

**Branch:** `nditer`  ·  **Date:** 2026-06-05  ·  **Status:** ✅ landed, suite green (9458 pass / 0 fail)

This is the per-migration log for moving `np.where`'s non-contiguous path off the scalar
`NpyExpr.Where` fallback and onto a dedicated **multi-operand per-chunk kernel** driven by
`NpyIterRef` — the canonical "selection" item on the migration priority list in
`docs/NPYITER_PERF_HANDOVER.md` (§8 / Phase 3).

> One migration at a time, with measured before/after perf **and** GC. This document is the
> evidence + design record for the `where` migration. The sibling candidate (narrow-int axis
> reduction) was **not** taken — see [§7](#7-why-where-and-not-the-axis-reduction-gap).

---

## 1. TL;DR

| | before (old) | after (new) |
|---|---|---|
| Non-contiguous `where` driver | `NpyExpr.Where` → `ExecuteExpression` | `ILKernelGenerator.GetWhereInnerLoop` → `ForEach` |
| Inner loop | **scalar only**, and casts `cond`→output dtype per element | SIMD `ConditionalSelect` when inner-contiguous; raw-bool scalar otherwise |
| Per-call managed alloc | NpyExpr tree (4 nodes) + signature `StringBuilder` + compile machinery | none beyond the iterator/broadcast already paid |

**Measured (clean same-binary A/B, [§4](#4-measurements-perf--gc)):** every non-contiguous
`where` shape got **1.19×–2.06× faster** with **7–48 % less GC**, and **small-N improved too**
(no setup-tax regression). The contiguous + scalar-operand fast paths were **left untouched**
(they already hit a fused whole-array SIMD kernel; routing them through NpyIter only ties — see
HANDOVER §4.1).

---

## 2. What was actually slow (the discovery)

`np.where(cond, x, y)` dispatches in `APIs/np.where.cs::where_internal`:

```
all of cond/x/y contiguous, bool cond   → DirectILKernelGenerator.WhereExecute   (whole-array SIMD)   ← fast
x or y originally scalar                 → WhereScalarX/Y/XY kernels               (whole-array SIMD)   ← fast
everything else (broadcast / strided)    → WhereImpl                              (NpyIter)            ← THIS
```

`WhereImpl` already used `NpyIterRef.MultiNew(4, …)` — so the operand iteration was *already* on
NpyIter — but it compiled the inner loop through **`NpyExpr.Where`**, which is the wrong vehicle
for `where`:

1. **`WhereNode.SupportsSimd == false`** (`Backends/Iterators/NpyExpr.cs:734`) — scalar only.
2. Even if it were `true`, `NpyExpr.Compile` gates SIMD on `AllEqual(inputTypes, outputType)`
   (`NpyExpr.cs:79`), which is **always false for `where`**: `cond` is `Boolean` while `x`/`y`
   are the output dtype. The DSL's "every input loads at the output dtype" rule **forces a
   `bool → T` cast on `cond` for every element**, then a `T` compare-to-zero, *before* the
   branch. So the old path paid a per-element cast + float compare just to read the condition.

The fix is a hand-written per-chunk kernel that reads `cond` as a raw `bool` byte
(`Ldind_U1 + brfalse`) and adds a SIMD `ConditionalSelect` fast path. It is faster **even before
SIMD fires**, because the raw-bool scalar inner loop is cheaper than the cast-cond NpyExpr loop —
which is exactly what the strided/col-broadcast rows below show.

---

## 3. Design

### 3.1 The per-chunk contract

`NpyIterRef.ForEach(NpyInnerLoopFunc, aux)` is NumPy's `do { inner(ptrs,strides,count,aux); }
while (iternext)` driver. The kernel processes **one inner-loop chunk**:

```csharp
unsafe delegate void NpyInnerLoopFunc(void** dataptrs, long* strides, long count, void* aux);
//   dataptrs[0]=cond(bool,1B)  [1]=x(T)  [2]=y(T)  [3]=result(T)
//   strides[op] = per-operand BYTE stride for the inner loop
```

`WhereImpl` builds the same 4-operand `MultiNew` iterator as before (EXTERNAL_LOOP, C-order,
`[RO,RO,RO,WO]`) — operands are already cast to `bool`/`T`/`T` by `where_internal`, so **no
buffering/casting happens** — and drives it with the new kernel.

### 3.2 New file: `Backends/Kernels/ILKernelGenerator.Where.cs`

`GetWhereInnerLoop(NPTypeCode outType)` → cached `NpyInnerLoopFunc`. The emitted IL does a
**runtime inner-stride dispatch** (per chunk):

```
SIMD ConditionalSelect : cond stride == 1  AND  x/y/result stride == elemSize
                         (inner loop contiguous for all 4 operands)
                         → 4×-unrolled Vector.ConditionalSelect over an expanded bool mask,
                           + 1-vector remainder, then the scalar loop finishes the tail.
scalar strided         : everything else  → per-operand byte-stride walk, raw bool read.
                         Also the only path for non-SIMD dtypes
                         (Boolean/Char/Half/Decimal/Complex).
```

The bool→lane **mask expansion** reuses the proven IL from the whole-array kernel
(`DirectILKernelGenerator.EmitInlineMaskCreation`, promoted `private`→`internal`), so the SIMD
result is bit-identical to the contiguous Direct `WhereKernel`. `EmitLoadIndirect` /
`EmitStoreIndirect` (already `internal`) handle all 15 dtypes for the scalar path. SIMD
eligibility mirrors `DirectILKernelGenerator.GenerateWhereKernelIL` exactly via the shared
`CanUseSimd` / `VectorBits` / `Avx2`/`Sse41` predicates.

### 3.3 Which shapes hit SIMD vs scalar

After NpyIter coalescing, the **inner** axis decides:

| shape | inner strides (cond,x,y,res) | path |
|---|---|---|
| row-mask `(1,M)`/`(M,)` broadcast over rows | `1, e, e, e` | **SIMD** |
| 1-D contiguous-ish view, any inner-contig view | `1, e, e, e` | **SIMD** |
| col-mask `(N,1)` broadcast over cols | `0, e, e, e` | scalar (fast raw-bool) |
| transpose / `::k` strided | `k', k·e, …` | scalar (fast raw-bool) |
| Decimal / Half / Complex / Char / Bool | any | scalar (covers all 15 dtypes) |

(`e` = `sizeof(T)`.) The col-broadcast (`cond` constant within a chunk) is a documented
follow-up for an additional SIMD copy path — [§6](#6-followups-precisely-scoped).

---

## 4. Measurements (perf + GC)

**Method.** Clean **same-binary A/B**: a temporary `NS_WHERE_OLD` env toggle routed the old
`NpyExpr.Where` path so OLD and NEW were measured **back-to-back in one process** (identical JIT,
cache, thermal state) — eliminating the cross-process variance that otherwise shows up on
memory-bound ops. The toggle was removed before commit. `ms/call` = mean over 50 calls (8 warm);
`bytes/call` = `GC.GetAllocatedBytesForCurrentThread()` delta (managed only — NDArray buffers are
unmanaged, so this isolates iterator/DSL overhead). Host: AVX2 (V256), .NET 10.0.101, net10.0.

All rows below route through `WhereImpl` (the migrated path). 2-D cases are `1000×1000` (1 M
elements); `small-cond-row` is `32×32` (1 K, the setup-tax probe); `strided-1d-step2` is `a[::2]`
over 2 M.

```
scenario             dt      OLD ms    NEW ms   spdup    GC bytes/call (old→new, Δ)
-----------------------------------------------------------------------------------
cond-row-bcast       f64     2.6342    2.1487   1.23x    7551 → 6032   (−20%)
cond-col-bcast       f64     3.0097    2.1721   1.39x    7551 → 6032   (−20%)
strided-transpose    f64     9.6381    7.6417   1.26x    6933 → 5504   (−20%)
small-cond-row       f64     0.0092    0.0078   1.19x    6521 → 6032   (− 7%)
cond-row-bcast       f32     2.2733    1.1058   2.06x    7058 → 6032   (−14%)
cond-col-bcast       f32     2.5716    1.6743   1.54x    7058 → 6032   (−14%)
strided-transpose    f32     7.4087    5.5437   1.34x    6530 → 5504   (−15%)
small-cond-row       f32     0.0085    0.0062   1.38x    6521 → 6032   (− 7%)
cond-row-bcast       i32     1.8439    1.1025   1.67x    7036 → 6032   (−14%)
cond-col-bcast       i32     2.1340    1.5958   1.34x    7036 → 6032   (−14%)
strided-transpose    i32     8.5036    5.5905   1.52x    6508 → 5504   (−15%)
small-cond-row       i32     0.0086    0.0059   1.46x    6521 → 6032   (− 7%)
strided-1d-step2     f64     3.6066    2.7498   1.31x    3173 → 1632   (−48%)
```

**Reading the numbers.**
- **Best wins** are the compute-bound, inner-contiguous, narrow-dtype rows: row-mask f32
  **2.06×**, i32 **1.67×** — these now hit the SIMD `ConditionalSelect` path the old scalar loop
  never reached.
- **f64 wins are smaller** (1.2–1.4×) because at 1 M×8 B the op is memory-bound (~24 MB read);
  SIMD can't beat bandwidth, but the cheaper inner loop + less GC still help.
- **Scalar-path rows still improve** (col-broadcast 1.34–1.54×, transpose 1.26–1.52×) purely from
  dropping the per-element `cond` cast — confirming the §2 diagnosis.
- **Small-N improved** (1.19–1.46×): the kernel is cached by output dtype, so there is **no
  per-call `NpyExpr` tree / signature `StringBuilder` allocation** — which is also the bulk of the
  GC reduction (the residual 6032 B is `broadcast_arrays` + the iterator + the result wrapper,
  unchanged by this migration).
- **GC:** −7 % to −48 %. The 1-D strided row drops 48 % because it has no `broadcast_arrays`
  cost to dilute the eliminated DSL allocation.

**Untouched fast paths (not in the A/B because they don't route through `WhereImpl`):** the
all-contiguous case (`DirectILKernelGenerator.WhereExecute`) and the scalar-operand case
(`WhereScalarX/Y/XY`) are byte-identical before/after. Routing them through NpyIter would only
tie at large N and risk a small-N setup-tax regression (HANDOVER §4.1 / §4.7), so they were
deliberately left on the whole-array kernels.

---

## 5. Correctness

- **Focused matrix (dotnet_run):** `7023 / 7023` checks pass — all 15 dtypes
  (byte, sbyte, int16, uint16, char, int32, uint32, single, int64, uint64, double, half, decimal,
  **complex**) × 3 layouts (row-broadcast → SIMD, col-broadcast → scalar, transpose → strided
  scalar), plus larger vector-aligned-plus-tail sizes per element width, plus NaN/±Inf
  propagation through the SIMD path. Covers every mask-expansion branch (1/2/4/8-byte) and every
  scalar fallback type.
- **Full suite (CI filter `TestCategory!=OpenBugs&TestCategory!=HighMemory`):**
  **9458 passed / 0 failed / 11 skipped** — holds the green line.
- **`where`-class tests:** the only 5 failures in the unfiltered `where` run are pre-existing
  `[OpenBugs]` (np.where(cond) tuple-vs-array; NEP50 int64 scalar promotion; an `NpyExpr` Half
  `ConstNode` limitation) — none touched by this change, all excluded from CI.

SIMD ⇔ scalar parity is structural: the SIMD path reuses the exact mask + `ConditionalSelect` IL
the contiguous Direct kernel already ships, and the scalar path uses the shared
`EmitLoadIndirect`/`EmitStoreIndirect`.

---

## 6. Follow-ups (precisely scoped)

1. **Cond-broadcast SIMD copy path** *(clear next increment).* When `cond` stride == 0 (the
   `(N,1)` per-row mask, the `cond-col-bcast` rows above), the whole chunk uses one condition
   value. Branch once on `*cond` and **SIMD-copy** the selected operand (`x` or `y`) into the
   result instead of scalar-walking. Expected to bring `cond-col-bcast` down to ≈ the
   `cond-row-bcast` SIMD number (e.g. f32 1.67 → ~1.1 ms, ~1.5× more). Gated on
   `sc==0 && sx==sy==sr==elemSize`; falls back to the current scalar path otherwise. ~80 lines of
   IL + re-run the §5 matrix.
2. **Place / masked-assign** can now reuse this multi-operand machinery (the HANDOVER calls out
   `WRITEMASKED`/`VIRTUAL` operand flags — `np.place` is the next selection op).
3. **Full unification (optional).** Routing the contiguous + scalar-operand cases through this
   kernel too would retire `Direct/DirectILKernelGenerator.Where.cs` and `.Where.Scalar.cs`. Do
   **not** until Phase 1 (setup-tax) lands — today it would tie large-N and risk small-N
   regression (HANDOVER §4.1/§4.7). Keep the hybrid until then.

---

## 7. Why `where` and not the axis-reduction gap

Both were on the table. `where` was taken because it is the higher-confidence, lower-risk, truly
*NpyIter-shaped* migration:

- The axis-reduction gap (narrow-int `sum(axis=…)`, the 25–57× row) is fixed by a **widening
  SIMD kernel**, not by the iterator — `NpyAxisIter` is itself scalar (HANDOVER §4.5/§4.6). It
  doesn't fit a "migrate to an NpyIter multi-operand kernel" framing, and HANDOVER flags a likely
  pre-existing regression to bisect first (§4.9, §13).
- `where` is the canonical multi-operand (3-in/1-out) NpyIter case, already half-on-NpyIter, with
  a concrete scalar-vs-SIMD gap that the per-chunk model closes cleanly — and it unlocks
  `place`/masked-assign next.

The axis-reduction lever remains documented in `NPYITER_PERF_HANDOVER.md` §7/§13 for a separate,
single-focus session.

---

## 8. Files changed

| file | change |
|---|---|
| `src/NumSharp.Core/Backends/Kernels/ILKernelGenerator.Where.cs` | **new** — per-chunk multi-operand `where` kernel (SIMD `ConditionalSelect` + raw-bool scalar fallback) for the target `ILKernelGenerator` class |
| `src/NumSharp.Core/APIs/np.where.cs` | `WhereImpl` now drives the 4-operand iterator with `ForEach(GetWhereInnerLoop(dtype))` instead of `NpyExpr.Where` + `ExecuteExpression`; method marked `unsafe` for the `ForEach` default arg |
| `src/NumSharp.Core/Backends/Kernels/Direct/DirectILKernelGenerator.Where.cs` | `EmitInlineMaskCreation` promoted `private`→`internal` so the new per-chunk kernel reuses the proven bool-mask-expansion IL (single source of truth) |

No public API change. No behavioral change (NumPy parity preserved). The contiguous and
scalar-operand fast paths are untouched.

---

## 9. Repro

```bash
# build
dotnet build -v q --nologo "-clp:NoSummary;ErrorsOnly" -p:WarningLevel=0 -f net10.0 src/NumSharp.Core/NumSharp.Core.csproj

# gate (must stay 0 failed)
cd test/NumSharp.UnitTest
dotnet build -v q --nologo "-clp:NoSummary;ErrorsOnly" -p:WarningLevel=0 -f net10.0
dotnet test --no-build -f net10.0 --filter "TestCategory!=OpenBugs&TestCategory!=HighMemory"

# perf A/B — re-add the NS_WHERE_OLD toggle in WhereImpl (see git history of this doc),
# then interleave Environment.SetEnvironmentVariable("NS_WHERE_OLD","1"/null) per scenario in a
# dotnet_run harness measuring ms/call + GC.GetAllocatedBytesForCurrentThread() over the
# cond-row / cond-col / transpose / 1-D-strided shapes across f64/f32/i32.
```
