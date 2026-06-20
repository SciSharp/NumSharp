# Cast Entry-Point Unification Plan — route every conversion site through the IL kernel

**Companion to** `CAST_BEAT_NUMPY_PLAN.md` (that one optimizes the `astype`/`NpyIter.Copy` kernels;
**this one** closes the *other* cast surfaces that still run the pre-IL slow paths). **Execute
after the Phase-A benchmark below.**

## Problem

The IL scalar cast kernel (`DirectILKernelGenerator.TryGetInnerCastKernel`, `Cast.Scalar.cs`) made
cross-dtype `astype` 1.1–4.8× faster for the Vector-less dtypes by replacing per-element
`Func`/`ConvertValue` dispatch with a direct `call Converts.To{Dst}`. But two **other** entry points
do the **same conversion** on the **old slow paths** — purely a perf gap (same `Converts` table →
already bit-exact, no correctness risk):

| Gap | Path | Slow mechanism | Trigger |
|-----|------|----------------|---------|
| **1. Buffered cast** | `NpyIterBufferManager` → `CopyWithCast` / `CopyStridedToContiguousWithCast` / `CopyContiguousToStridedWithCast` (`NpyIterCasting.cs:600/622/661`) | per-element `ConvertValue`; the 2 multi-dim helpers also recompute `Σ coords·strides` per element (the O(ndim) anti-pattern, lines 640-642 / 679-681) | NpyIter operand with `CAST + BUFFERED` (mixed-dtype iteration, `out=` mismatch, `where=` casts, candidate: `np.evaluate` mixed dtypes) |
| **2. Assignment cast** | `UnmanagedMemoryBlock.CastTo<TIn,TOut>(IMemoryBlock<TIn>)` (`Casting.cs:113`) → `Func` delegate loop | `Converts.FindConverter` delegate, one Invoke per element | `dst[slice] = srcOtherDtype` → `Setters.cs:390/424/528` → `Storage.CastIfNecessary`/`CastTo`; also `UnmanagedStorage.cs:1049`, legacy `CastCrossType` `{i8,i16,i32}→u64` (`Default.Cast.cs:121`) |

**Fix (both):** route through the already-built `TryGetInnerCastKernel` — *route, don't rebuild*.

---

## Definition of Done

1. Both gaps drive conversions through the IL inner kernel; `ConvertValue` / `FindConverter` remain
   only as the IL-disabled fallback.
2. **Benchmarked, all dtypes:** every cell with a NumPy analog is `NPY/NS ≥ 1.0`; the Vector-less
   dtypes (bool/char/f16/decimal/complex) match the `astype` IL-kernel speedup; SIMD dtypes
   unchanged (no regression).
3. **Bit-exact** (same `Converts` table): existing parity harness + one new assignment-cast parity
   test, 0-diff vs NumPy 2.4.2.
4. Full suite green on net8.0 + net10.0.

---

## Phase A — Benchmark & reachability (DO FIRST; sets priority)

Quantify each gap across **all 15 dtypes** *before* touching code. Output a checked-in
`benchmark/poc/cast_entrypoints_matrix.md`.

### A1. Gap 2 — assignment-with-cast  *(clearly reachable; likely the hotter gap)*
- **Trigger:** `dst[...] = src` where `dst.dtype != src.dtype`.
- **Matrix:** for src→dst over all 15×15 (emphasis on Vector-less as src AND dst), 1M elements,
  two layouts: full assign `dst[":"] = src` and strided-slice assign `dst["1:-1"] = src["1:-1"]`.
  NumSharp vs NumPy (`dst[...] = src` in numpy does the same safe cast).
- **Measure:** `NPY/NS` per cell; flag `<1.0`. Expect bool/char/f16/complex to lag (Func loop)
  exactly like pre-IL `astype` did.

### A2. Gap 1 — buffered-iteration cast  *(confirm reachability first)*
- **Find the trigger:** which user op constructs a multi-operand `NpyIter` with `CAST + BUFFERED`?
  Candidates: `np.evaluate(expr)` with mixed-dtype operands; `np.copyto` routed through the
  buffered iterator; any op that calls `NpyIter` execute with a dtype-mismatched operand. Add a
  one-shot probe asserting `CopyToBufferWithCast`/`CopyFromBufferWithCast` is actually hit
  (e.g. a counter or a breakpoint-style log under a debug flag).
- **If reachable:** benchmark the triggering op across dtypes (NS vs NumPy equivalent).
- **If cold** (no common op triggers it): record that, **lower priority to "completeness"** — still
  fix (cheap), but it is not a hot path. *Do not assume it is hot.*

### A3. Decision gate
Priority order = the lagging-cell magnitudes from A1/A2. Gap 2 almost certainly leads; Gap 1's rank
depends on A2 reachability.

---

## Phase B — Implementation (route through `TryGetInnerCastKernel`)

### B1. Gap 2 — contiguous `CastTo<TIn,TOut>` → IL inner kernel
`UnmanagedMemoryBlock.Casting.cs:113` (`CastTo<TIn,TOut>(this IMemoryBlock<TIn> source)`):

```csharp
var ret = new UnmanagedMemoryBlock<TOut>(source.Count);
var srcT = InfoOf<TIn>.NPTypeCode; var dstT = InfoOf<TOut>.NPTypeCode;
var inner = DirectILKernelGenerator.TryGetInnerCastKernel(srcT, dstT);
if (inner != null)
    inner(source.Address, sizeof(TIn), ret.Address, sizeof(TOut), source.Count); // contiguous: stride = elemSize
else { // IL-disabled fallback — unchanged
    var convert = Converts.FindConverter<TIn, TOut>();
    for (long i = 0; i < source.Count; i++) *(ret.Address + i) = convert(*(source.Address + i));
}
```
- **Audit the sibling overload** `CastTo<TIn,TOut>(this IMemoryBlock source)` (line 138, nested
  switch with direct `Converts.To{Dst}`) — already faster than the Func loop, but route it too for
  uniformity if it is on any hot path (Phase A tells which is hit).
- Same-type (`TIn==TOut`) keeps its existing short-circuit (never reaches the cast loop).

### B2. Gap 1 — the 3 one-axis helpers → IL inner kernel
Keep all public signatures (the 6 `NpyIterBufferManager` call sites are untouched). Internally:

- **`CopyWithCast(src, srcStride, srcType, dst, dstStride, dstType, count)`** — a 1-axis loop that
  *is* the inner-kernel shape:
  ```csharp
  var inner = DirectILKernelGenerator.TryGetInnerCastKernel(srcType, dstType);
  if (inner != null) { inner(src, srcStride*srcElemSize, dst, dstStride*dstElemSize, count); return; }
  // existing ConvertValue loop as fallback
  ```
- **`CopyStridedToContiguousWithCast`** (strided src → contiguous dst): replace the per-element
  `Σ coords·strides` recompute with the **incremental-coord outer walk + IL inner** (the proven
  `CopyStridedToStridedWithCast` shape), with `srcInnerB = strides[last]*srcElemSize`,
  `dstInnerB = dstElemSize`. This also retires the O(ndim)-per-element addressing.
- **`CopyContiguousToStridedWithCast`** (contiguous src → strided dst): mirror, with
  `srcInnerB = srcElemSize`, `dstInnerB = strides[last]*dstElemSize`.
- **ndim==0 buffered case** (`NpyIterBufferManager` ~line 360): single element — leave as
  `ConvertValue` (negligible) or `inner(src,0,dst,0,1)`.

> Optional simplification: the two multi-dim helpers are special cases of
> `CopyStridedToStridedWithCast` (one side unit-strided). Could delete them and have the buffer
> manager call `CopyStridedToStridedWithCast` with synthesized contiguous strides for the unit side.
> Prefer the in-place rewrite first (zero call-site churn); unify only if it stays clean.

Each helper keeps the `ConvertValue` fallback for `Enabled == false`.

---

## Phase C — Validation + re-benchmark

1. **Correctness (by construction + verified):** all routes use the same `Converts.To{Dst}` table
   as the paths they replace → bit-identical. Verify:
   - Existing `StridedCastParityTests` (already 0-diff, exercises `TryGetInnerCastKernel`).
   - **New `AssignmentCastParityTests`:** `dst[slice] = src` for representative src→dst pairs
     (incl. all Vector-less) × {full, strided-slice} == NumPy `dst[...] = src`, bit-exact.
   - If Gap 1 is reachable: a parity test on the triggering op across dtypes.
2. **Perf:** re-run the Phase-A matrix; targeted cells reach ✅, no SIMD-dtype regression.
3. **Suite:** `dotnet test` net8.0 + net10.0, `TestCategory!=OpenBugs&!=HighMemory`, 0 fail.

---

## Risks & notes

| Risk | Mitigation |
|------|-----------|
| Gap 1 buffered path is cold | Phase A2 confirms reachability before investing; fix is cheap regardless. |
| `CastTo<TIn,TOut>` is widely used internally | Keep the `FindConverter` fallback; IL kernel is bit-identical → no semantic change, only speed. Run full suite. |
| Cross-namespace ref (`Backends.Unmanaged` → `Backends.Kernels`) | Already done in `NpyIterCasting`; same assembly, no cycle. |
| `sizeof(T)` in generic unmanaged context | Legal in `unsafe` generic over `unmanaged T`; matches the existing `UnmanagedMemoryBlock<TOut>` usage. |
| Legacy `CastCrossType` `{i8,i16,i32}→u64` branch | Out of scope here (it's an `astype` routing decision — handled in the BEAT_NUMPY plan Phase 2); this plan only speeds the contiguous `CastTo` it calls. |

---

## Expected outcome

Both gaps inherit the measured `astype` win — Vector-less conversions **1.1–4.8× faster** (the
lighter the conversion, the bigger), assignment-with-cast and buffered casting now match the
`astype` IL path. Zero new kernels (reuse `TryGetInnerCastKernel`), zero semantic change, ~30–60
lines touched across 2 files + 1 new parity test.

## Code references

| Component | Location |
|-----------|----------|
| IL inner cast kernel (reuse) | `Backends/Kernels/Direct/DirectILKernelGenerator.Cast.Scalar.cs` (`TryGetInnerCastKernel`) |
| Gap 1 — buffered cast helpers | `Backends/Iterators/NpyIterCasting.cs:600/622/661` (`CopyWithCast`, `CopyStridedToContiguousWithCast`, `CopyContiguousToStridedWithCast`) |
| Gap 1 — buffer-manager call sites | `Backends/Iterators/NpyIterBufferManager.cs:368/373/402/407/1045/1093` (`CopyToBufferWithCast`/`CopyFromBufferWithCast`) |
| Gap 1 — cast+buffered gate | `Backends/Iterators/NpyIter.cs:369/565` (`CAST` flag + `BUFFERED` requirement) |
| Gap 2 — contiguous Func cast | `Backends/Unmanaged/UnmanagedMemoryBlock.Casting.cs:113` (`CastTo<TIn,TOut>`) |
| Gap 2 — assignment callers | `Backends/Unmanaged/UnmanagedStorage.Setters.cs:390/424/528`, `UnmanagedStorage.cs:1049` |
| Reference shape (already IL) | `Backends/Iterators/NpyIterCasting.cs` (`CopyStridedToStridedWithCast` — the incremental-coord + IL-inner pattern to mirror) |
| Parity tests | `test/…/Backends/Kernels/StridedCastParityTests.cs` (+ new `AssignmentCastParityTests`) |
