# Group 1: Iterator subsystem audit

Branch: `nditer` vs `master`. Scope covers everything under
`src/NumSharp.Core/Backends/Iterators/`:

| File | LoC | Role |
|---|---|---|
| `INDIterator.cs` | 27 | Non-generic surface for `NDIterator<T>` |
| `NDIterator.cs` | 294 | Legacy element iterator (rewritten as NpyIter wrapper) |
| `NDIteratorExtensions.cs` | 92 | `AsIterator` factory methods |
| `NpyAxisIter.cs` | 492 | NPY_MAXDIMS=64 axis iterator (parallel impl) |
| `NpyAxisIter.State.cs` | 42 | NpyAxisIter state struct |
| `NpyExpr.cs` | 1123 | Tier-3C DSL → kernel compiler |
| `NpyIter.cs` | 3469 | Main NpyIter implementation |
| `NpyIter.State.cs` | 979 | NpyIterState struct + dynamic allocation |
| `NpyIter.Execution.cs` | 723 | Bridge to ILKernelGenerator |
| `NpyIter.Execution.Custom.cs` | 155 | Tier 3A/3B/3C entry points |
| `NpyIterBufferManager.cs` | 637 | Buffer alloc/copy |
| `NpyIterCasting.cs` | 530 | Cast rules + value conversion |
| `NpyIterCoalescing.cs` | 495 | Axis sort + coalesce |
| `NpyIterFlags.cs` | 516 | Flag enums |
| `NpyIterKernels.cs` | 263 | Kernel interface (legacy, mostly unused) |
| `NpyLogicalReductionKernels.cs` | 155 | All/Any/Min/Max/Sum/Prod axis kernels |
| `NpyNanReductionKernels.cs` | 344 | NaN-aware sum/prod/min/max/mean/var kernels |

Reference: `src/numpy/numpy/_core/src/multiarray/nditer_*.c`.

Verification used `python -c` (NumPy 2.4.2) and `dotnet_run` against the
project; everything below cites the exact source line and reproduction
command.

---

## File: src/NumSharp.Core/Backends/Iterators/NpyIter.cs

### Function: `Iternext()` — line 1985-2003
**Severity:** bug
**Criteria coverage:**
- [✗] NumPy structural parity — NumPy's `iternext` is dispatched through a function pointer keyed on flags (nditer_templ.c.src lines 131-396 generate `npyiter_iternext_iters{N}order{C/F}*` specializations). NumSharp picks the dispatcher via `GetIterNext()` but the public-facing `Iternext()` method ignores it.
- [✗] NumPy behavioral parity — `Iternext()` ignores EXLOOP and the non-reduce BUFFER refill path.
- [✗] Performance — see findings below; about 20× slower than NumPy when called per-element on a 10M int64 array.
- [N/A] IL generation — control flow only.
- [N/A] dtype coverage.
- [N/A] API parameter parity.
- [✗] No wasted copies — runs Advance per element even when EXLOOP allows whole-strip advance.
- [✗] Uses appropriate iterator path — bypasses `GetIterNext()`.

**Finding A: `Iternext()` ignores EXLOOP, advances 1 element at a time, can read past buffer.**

The bridge call at `NpyIter.Execution.cs:140-150` correctly calls `GetIterNext()` (`StandardNext`/`ExternalLoopNext`/`SingleIterationNext`). But the public `Iternext()` method at line 1985-2003 just calls `_state->Advance()` regardless of EXLOOP. Code path:

```
public bool Iternext()
{
    if (_state->IterIndex >= _state->IterEnd) return false;
    uint itFlags = _state->ItFlags;
    if ((itFlags & (uint)NpyIterFlags.BUFFER) != 0 &&
        (itFlags & (uint)NpyIterFlags.REDUCE) != 0 &&
        _state->CoreSize > 0)
    {
        return BufferedReduceIternext();
    }
    _state->Advance();
    return _state->IterIndex < _state->IterEnd;
}
```

There is no `if ((itFlags & EXLOOP) != 0) return ExternalLoopNext(...)` branch. The header comment at `NpyIter.Execution.cs:42-46` explicitly documents this bug but the underlying iterator was never fixed.

**Reproduction (verified by `python -c` + `dotnet_run`):**

```python
# NumPy ground truth
import numpy as np
a = np.arange(12).reshape(3, 4).transpose()  # shape (4,3), non-coalescible
it = np.nditer(a, flags=['external_loop'])
print(sum(1 for _ in it))  # → 4 outer iterations
```

```csharp
// NumSharp behavior
var a = np.arange(12).reshape(3, 4).transpose();
using var it = NpyIterRef.New(a, NpyIterGlobalFlags.EXTERNAL_LOOP,
    NPY_ORDER.NPY_CORDER, NPY_CASTING.NPY_SAFE_CASTING);
// Internal NDim=2, Shape[NDim-1]=3
int count = 1; while (it.Iternext()) count++;
// count == 12 (advanced one element at a time)
// Expected: 4 (IterSize / Shape[NDim-1] = 12/3 = 4)
```

A user implementing the canonical NumPy pattern
```
do { kernel(dataptrs, strides, Shape[NDim-1]); } while (it.Iternext());
```
would read `4 * 12 = 48 elements` from a 12-element buffer (3× overrun).

**Finding B: `Iternext()` BUFFER (non-reduce) path has no refill logic → segfault when IterSize > BufferSize.**

For BUFFERED + not-REDUCE, `Iternext()` falls through to `state.Advance()`. `Advance()` doesn't refill the buffer; `GetDataPtr(int op)` (line 2723-2756) computes a buffer offset using `IterIndex - (BufIterEnd - Math.Min(BufferSize, IterSize - IterStart))` which goes out-of-bounds past the buffer when crossing fill boundaries.

**Reproduction:**

```csharp
// 20000 int32 elements, cast to float64, BUFFERED
var src = np.arange(20000).astype(NPTypeCode.Int32);
var dtypes = new[] { NPTypeCode.Double };
var opFlags = new[] { NpyIterPerOpFlags.READONLY };
using var it = NpyIterRef.MultiNew(1, new[] {src},
    NpyIterGlobalFlags.BUFFERED, NPY_ORDER.NPY_KEEPORDER,
    NPY_CASTING.NPY_SAFE_CASTING, opFlags, dtypes);
// IterSize=20000, BufferSize=8192 (default), BufIterEnd=8192
double sum = 0; long count = 0;
while (true) {
    sum += *(double*)it.GetDataPtr(0);
    count++;
    if (!it.Iternext()) break;
}
// → AccessViolationException, process segfault
```

With a buffer large enough to hold the whole iteration (e.g. `bufferSize: 25000`) the loop completes correctly — confirming the missing refill is the root cause.

```python
# NumPy: works correctly for any size
import numpy as np
a = np.arange(20000, dtype=np.int32)
total = 0
for x in np.nditer(a, op_dtypes=[np.float64], flags=['buffered'], casting='safe'):
    total += float(x)
# total == 199990000.0
```

**Remediation:**
1. Add EXLOOP and BUFFER (non-reduce) branches to `Iternext()`:
   ```
   if ((itFlags & EXLOOP) != 0) return ExternalLoopNext(ref *_state);
   if ((itFlags & BUFFER) != 0) return BufferedNonReduceIternext();
   ```
2. Implement `BufferedNonReduceIternext()` that mirrors NumPy's
   `npyiter_buffered_iternext` (nditer_templ.c.src:325): advance iter index,
   when crossing buffer boundary call `CopyToBuffer` again for all READ
   operands, and (for WRITE operands) flush the previous buffer to the array.
3. Alternatively, delete the public `Iternext()` method and force callers
   onto `GetIterNext()`; the bridge already does the right thing.

Bug 5 from the prior audit confirmed: **YES**, both subclaims.

---

### Function: `Initialize` (constructor body) — line 125-609
**Severity:** clean / parity-gap
**Criteria coverage:**
- [✓] NumPy structural parity — mirrors `NpyIter_AdvancedNew` (`nditer_constr.c:228`). Same phases: broadcast shape, allocate strides, set up operands, op_axes, FlipNegativeStrides, ReorderAxesForCoalescing, CoalesceAxes, BUFFERED setup.
- [✓] NumPy behavioral parity — verified for plain N-d arrays, transposed, reversed, broadcast, op_axes reduction, allocate-output, IDENTPERM/NEGPERM flags.
- [✓] Performance — initialization itself is fine; mainly allocates NDim arrays in two contiguous blocks.
- [N/A] IL generation.
- [✓] All 15 dtypes — initialization is dtype-agnostic (only stride/offset arithmetic).
- [✗] API parameter parity — see findings below.
- [✓] No wasted copies for the construction phase.
- [✓] Uses appropriate iterator path.
- [✗] Missing functionality — see below.

**Findings:**
- `MultiNew` doesn't expose an `itershape` parameter — only `AdvancedNew` does. NumPy's `numpy.nditer` Python API has `itershape` directly. Minor; NumPy's C API also splits the two.
- The default `casting` differs from NumPy: NumPy defaults to `'safe'`; NumSharp defaults to `NPY_SAFE_CASTING` too. OK.
- `op_dtypes is null` is handled (keep source dtype). `op_dtypes[i] == NPTypeCode.Empty` is also handled (line 266-268). Good.
- `BUFFERED` is required when any operand needs casting (line 371-376) — matches NumPy.
- `EnableExternalLoop()` (line 2852) does NOT validate HASINDEX/HASMULTIINDEX as NumPy does — verified bug (see entry below).

---

### Function: `EnableExternalLoop()` — line 2852-2857
**Severity:** bug
**Criteria coverage:**
- [✗] NumPy behavioral parity — NumPy raises `ValueError` if MULTI_INDEX or HASINDEX is set.

**Reproduction:**
```python
import numpy as np
a = np.arange(12).reshape(3,4)
it = np.nditer(a, flags=['multi_index'])
it.enable_external_loop()
# ValueError: Iterator flag EXTERNAL_LOOP cannot be used if an index or multi-index is being tracked
```

```csharp
var a = np.arange(12).reshape(3, 4);
using var it = NpyIterRef.New(a, NpyIterGlobalFlags.MULTI_INDEX,
    NPY_ORDER.NPY_CORDER, NPY_CASTING.NPY_SAFE_CASTING);
bool ok = it.EnableExternalLoop();  // returns true, no error
// HasMultiIndex==true, HasExternalLoop==true (illegal combination)
```

**Remediation:** add at line 2853:
```csharp
if ((_state->ItFlags & (uint)(NpyIterFlags.HASMULTIINDEX | NpyIterFlags.HASINDEX)) != 0)
    throw new InvalidOperationException(
        "Iterator flag EXTERNAL_LOOP cannot be used if an index or multi-index is being tracked");
```

---

### Function: `GetIterView(int operand)` — line 2650-2704
**Severity:** bug
**Criteria coverage:**
- [✗] NumPy behavioral parity — throws OverflowException on 0-dim iterators.

**Reproduction:**
```csharp
var scalar = np.array(42L);  // 0-d array
using var it = NpyIterRef.New(scalar, NpyIterGlobalFlags.None,
    NPY_ORDER.NPY_CORDER, NPY_CASTING.NPY_SAFE_CASTING);
var view = it.GetIterView(0);  // crashes
// System.OverflowException: Arithmetic operation resulted in an overflow.
//   at NumSharp.NDArray.get_Item(Slice[] slice) in NDArray.Indexing.cs:line 78
//   at NumSharp.Backends.Iteration.NpyIterRef.GetIterView in NpyIter.cs:line 2668
```

Line 2668 returns `original.flat[0]` for ndim=0. `original.flat[0]` triggers slice indexing which fails on scalars. NumPy returns a 0-d view.

**Remediation:**
```csharp
if (ndim == 0)
{
    // Return 0-d view sharing the scalar's storage
    return new NDArray(original.Storage, Shape.Scalar);
}
```

---

### Function: `GetDataPtr(int operand)` — line 2723-2756
**Severity:** bug
**Criteria coverage:**
- [✗] NumPy behavioral parity — the buffered-non-reduce branch computes `bufferPos = IterIndex - (BufIterEnd - Math.Min(BufferSize, IterSize - IterStart))`. This formula assumes the very first buffer fill, doesn't track which fill cycle we're in. Result: pointer goes out-of-bounds past one fill cycle (see segfault repro above).
- [✗] No wasted copies — N/A (the bug means no copy happens at all).

**Finding:** This function is the read endpoint for buffered iteration but it depends on bookkeeping that isn't updated when the buffer refills. See `Iternext()` Finding B.

**Remediation:** as part of the `BufferedNonReduceIternext()` fix, maintain a `BufferStart` field (the IterIndex value at the start of the current buffer fill) and compute `bufferPos = IterIndex - BufferStart`.

---

### Function: `Shape` property — line 2494-2520
**Severity:** parity-gap
**Criteria coverage:**
- [✗] NumPy behavioral parity — when MULTI_INDEX is NOT set, returns the **coalesced internal** shape, not the original array shape. NumPy returns `it.shape == it.itershape` (the original requested iter shape).

**Reproduction:**
```csharp
var a = np.arange(12).reshape(3, 4);  // contiguous, will coalesce
using var it = NpyIterRef.New(a, NpyIterGlobalFlags.None,
    NPY_ORDER.NPY_CORDER, NPY_CASTING.NPY_SAFE_CASTING);
var shape = it.Shape;  // returns [12], not [3, 4]
```

```python
import numpy as np
a = np.arange(12).reshape(3, 4)
it = np.nditer(a)
print(it.shape)  # (3, 4) — NumPy returns original
```

This is a documented divergence (the property explicitly returns the
internal post-coalesce shape when `!HASMULTIINDEX`) but it's not the
NumPy contract.

**Remediation:** Track the original `itershape` from `Initialize`; return
it here regardless of coalescing. Users wanting the post-coalesce shape
can use the internal `_state->Shape` array via `RawState`.

---

### Function: `StandardNext` / `ExternalLoopNext` / `SingleIterationNext` — line 1422-1477
**Severity:** clean
**Criteria coverage:**
- [✓] NumPy structural parity — three specialized iternext functions, matching NumPy's
  `npyiter_iternext_*` specializations.
- [✓] NumPy behavioral parity — used via `GetIterNext()` correctly.
- [✓] Performance — fine.

These are the correct implementations. The bug is that `Iternext()` doesn't use them.

---

### Function: `BufferedReduceIternext()` — line 2011-2102
**Severity:** bug
**Criteria coverage:**
- [✓] NumPy structural parity — mirrors NumPy's double-loop pattern
  from `nditer_templ.c.src:131-210`.
- [✗] NumPy behavioral parity — complex enough that the refill paths
  miss the "what was the last writeback pointer" tracking. Has been
  ad-hoc patched (see `currentArrayPos == previousWritebackPos` check at
  line 2061). Works for simple 2D cases (verified) but multi-axis
  reductions with `nonReduceAxisCount > 1` fall back to non-buffered
  path (`SetupBufferedReduction` line 1148-1155).
- [parity-gap] Doesn't honor `REUSE_REDUCE_LOOPS` — NumPy caches the
  reduce schedule across calls; NumSharp recomputes.

**Reproduction:**
```csharp
// op_axes: src uses both, dst reduces axis 1
var src = np.array(new int[,]{{1,2},{3,4},{5,6}}).astype(NPTypeCode.Int64);
var dst = np.zeros(new Shape(3), NPTypeCode.Int64);
var opAxes = new int[][] { new int[]{0,1}, new int[]{0,-1} };
using var it = NpyIterRef.AdvancedNew(2, new[] {src, dst},
    NpyIterGlobalFlags.REDUCE_OK | NpyIterGlobalFlags.MULTI_INDEX,
    NPY_ORDER.NPY_CORDER, NPY_CASTING.NPY_SAFE_CASTING,
    new[] {NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READWRITE},
    new[] {NPTypeCode.Int64, NPTypeCode.Int64}, 2, opAxes);
do {
    long val = *(long*)it.RawState->DataPtrs[0];
    long* dstPtr = (long*)it.RawState->DataPtrs[1];
    if (it.IsFirstVisit(1)) *dstPtr = 0;
    *dstPtr += val;
} while (it.Iternext());
// dst == [3, 7, 11]  — correct
```

Basic reduction works. The deeper concern is performance/coverage of
multi-axis reductions; covered by `SetupBufferedReduction` short-circuit.

**Remediation:** Implement full N-d buffered reduction matching NumPy's
multi-axis schedule.

---

### Function: `GetMultiIndex` / `GetMultiIndexFunc` / `GotoMultiIndex` — line 2226-2459
**Severity:** clean
**Criteria coverage:**
- [✓] NumPy structural parity — three specializations (IDENTPERM /
  positive-perm / NEGPERM) matching NumPy's nditer_templ.c.src:481.
- [✓] NumPy behavioral parity — verified by `[3,4,5]` shape, transpose, reverse.
- [✓] Performance — direct pointer reads, no allocation.

---

### Function: `CalculateBroadcastShape` — line 617-730
**Severity:** clean
**Criteria coverage:**
- [✓] NumPy structural parity — delegates to `Shape.ResolveReturnShape` which
  implements NumPy broadcasting.
- [✓] Handles op_axes with explicit reduction axis encoding
  (`NpyIterUtils.ReductionAxis` matches `NPY_ITER_REDUCTION_AXIS`).
- [✓] No wasted copies.

---

### Function: `ApplyOpAxes` — line 1256-1345
**Severity:** clean
**Criteria coverage:**
- [✓] NumPy structural parity — sets REDUCE flag, detects explicit
  reduction axes via the offset encoding.
- [✓] Validates `REDUCE_OK` and `READWRITE` constraints (matches NumPy).
- [✓] Mask validation (`CheckMaskForWriteMaskedReduction`).

---

### Function: `ResetBasePointers(ReadOnlySpan<IntPtr>)` — line 1884-1943
**Severity:** clean
**Criteria coverage:**
- [✓] NumPy structural parity — mirrors `NpyIter_ResetBasePointers`
  (nditer_api.c:314): adds `BaseOffsets[iop]` to each new base pointer,
  then `GotoIterIndex(IterStart)`. Re-primes buffers if BUFFER set.
- [✓] NumPy behavioral parity — verified for plain and reversed arrays.

---

### Function: `Copy()` — line 2934-3041
**Severity:** parity-gap
**Criteria coverage:**
- [✓] NumPy structural parity — allocates new state, copies all scalar
  fields and dim/op arrays, reallocates buffers.
- [✓] NumPy behavioral parity — verified that copies can advance
  independently and preserve initial position.
- [✗] Doesn't copy ResetDataPtrs offset chain — line 2992 copies
  `ResetDataPtrs[op] = _state->ResetDataPtrs[op]` but for buffered
  reduce there's also `ArrayWritebackPtrs` (line 3003) and `ReduceOuterPtrs`
  (line 3002). Those are copied. OK.
- Missing: `_cachedIterNext` is intentionally not copied (line 3030).

Generally good. Tracker for completeness — verify deep equivalence of
buffered+reduce copies under stress.

---

### Function: `CheckAllOperandsContiguous` — line 997-1055
**Severity:** clean
**Criteria coverage:**
- [✓] Correctly handles `allowFlip` parameter to align with K-order
  flip behavior. Size-1 dimensions are treated as trivially contiguous.

---

### Function: `SetupBufferedReduction` — line 1094-1249
**Severity:** parity-gap
**Criteria coverage:**
- [✗] NumPy behavioral parity — when `nonReduceAxisCount > 1`, falls back
  to `CoreSize=0` (line 1148-1155), forcing the slow per-element
  `Advance()` path. NumPy handles arbitrary-depth multi-axis reductions
  with the same double-loop pattern.
- [✓] 2D reduce case works.
- [✗] Performance — multi-axis reductions lose all buffering benefit.

**Finding:** Document the limitation; multi-axis reductions need
extension to handle ≥2 non-reduce axes.

**Remediation:** Implement NumPy's `npyiter_compute_buffered_reduce_outer_size`
properly so non-reduce axes are collapsed into the outer loop using
combined coords + stride lookups.

---

### Function: `Reset()` / `GotoIterIndex(long)` — line 1512, 2207
**Severity:** clean
**Criteria coverage:**
- [✓] State `Reset()` (`NpyIter.State.cs:845`) correctly delegates to
  `GotoIterIndex(IterStart)`, propagating Coords, FlatIndex, DataPtrs.

---

### Function: `Static NpyIter.Copy(NDArray, NDArray)` — line 3220-3262
**Severity:** clean
**Criteria coverage:**
- [✓] Provides drop-in replacement for legacy `MultiIterator.Assign`
  with broadcast + cast.
- [✓] Same-dtype fast path uses SIMD copy kernel; cross-dtype falls
  through to per-element conversion.
- [✓] Coalesces axes correctly via `CoalesceAxes` helper.
- [✓] All 12 NumSharp dtypes routed through `NpyIterCasting.ConvertValue`
  (which is the bottleneck — see NpyIterCasting findings).

---

## File: src/NumSharp.Core/Backends/Iterators/NpyIter.State.cs

### Struct: `NpyIterState`
**Severity:** clean
**Criteria coverage:**
- [✓] NumPy parity — fields cover NumPy's `NpyIter_InternalOnly` struct.
- [✓] All dim/op arrays allocated dynamically (no NPY_MAXDIMS limit) —
  documented divergence.
- [✓] Two-block allocation (dim arrays + op arrays) reduces fragmentation.

### Function: `Advance()` — line 712-765
**Severity:** clean (caveats below)
**Criteria coverage:**
- [✓] NumPy parity — ripple-carry through dims, updates pointers per
  axis * stride * elementSize, updates FlatIndex.
- [✓] Fast path for IDENTPERM + C-index (`usesFastPath` increments
  FlatIndex by 1).
- [N/A] EXLOOP — `Advance()` is called only when EXLOOP is NOT set
  (correct), but `Iternext()` ignores this contract; see Iternext finding.

### Function: `BufferedReduceAdvance()` — line 778-827
**Severity:** clean
**Criteria coverage:**
- [✓] NumPy parity — matches double-loop pattern in `nditer_templ.c.src:131-210`.
- [✓] Handles inner (`CorePos`) and outer (`ReducePos`) advances correctly.

### Function: `GotoIterIndex(long)` — line 869-901
**Severity:** clean
- [✓] Coordinate calculation, data pointer reconstruction, buffer reuse
  invalidation all present.

### Function: `ComputeFlatIndex` — line 921-977
**Severity:** clean
- [✓] Correctly handles NEGPERM (flips coordinate via `Shape[d] - Coords[d] - 1`).
- [✓] Both C-order and F-order index computation.

---

## File: src/NumSharp.Core/Backends/Iterators/NpyIter.Execution.cs

### Function: `ForEach(NpyInnerLoopFunc, void*)` — line 126-158
**Severity:** clean
**Criteria coverage:**
- [✓] NumPy structural parity — matches the canonical ufunc loop pattern.
- [✓] Uses `GetIterNext()` (correct dispatcher).
- [✓] Uses `BufStrides` (already byte-stride) when BUFFER is set.
- [✓] `ResolveInnerLoopCount` correctly returns BufIterEnd | Shape[NDim-1] | 1.

This is the **good** path — bypasses the buggy `Iternext()`.

### Function: `ExecuteGeneric<TKernel>` — line 201-243
**Severity:** clean
- [✓] Struct-generic devirtualization; JIT inlines kernel call.
- [✓] Fast-path single-iteration; multi-loop driver.

### Function: `ExecuteReducing<TKernel,TAccum>` — line 267-301
**Severity:** clean
- [✓] Same pattern as ExecuteGeneric, accumulator threaded by ref.

### Function: `BufferedReduce<TKernel,TAccum>` — line 420-445
**Severity:** bug
**Criteria coverage:**
- [✗] Uses `Iternext()` (line 444), inheriting the EXLOOP/BUFFER bugs.

**Finding:** This is one of the few legitimate uses of `Iternext()` that
require the buffered-reduce double-loop. It happens to work because the
`BUFFERED + REDUCE` branch is the one path `Iternext()` does dispatch
correctly. But the rest of the iterator infrastructure should prefer
`GetIterNext()` for consistency.

### Function: `ExecuteBinary(BinaryOp)` — line 312-351
**Severity:** clean
- [✓] Picks SimdFull/SimdScalarRight/SimdScalarLeft/SimdChunk/General
  paths correctly.
- [✓] Routes buffered through dedicated `RunBufferedBinary` (line 683-715)
  using BufStrides to avoid the Strides×ElementSize bug.

### Function: `RunBufferedBinary(BinaryOp)` — line 683-715
**Severity:** bug
**Criteria coverage:**
- [✗] Uses `Iternext()` (line 714) — same EXLOOP/BUFFER refill bug.

**Reproduction:** Until the buffered refill is fixed, this path will
segfault on arrays > BufferSize in cross-dtype binary ops.

### Function: `ExecuteCopy()` — line 518-543
**Severity:** clean
- [✓] Same path as ExecuteBinary, picks Contiguous/General CopyExecutionPath.

### Function: `ExecuteReduction<TResult>(ReductionOp)` — line 391-411
**Severity:** clean
- [✓] Calls IL kernel directly with iterator strides — kernel handles
  iteration internally; no Iternext usage.

### Function: `DetectExecutionPath()` — line 553-579
**Severity:** clean
- [✓] Correctly picks SimdScalarLeft/Right when one operand is fully
  broadcast (all strides=0).

### Function: `FillElementStrides(int op, long* dst, int ndim)` — line 612-616
**Severity:** clean
- [✓] Returns element-count strides (matches IL kernel expectations).

### Function: `GetInnerLoopByteStrides()` — line 624-646
**Severity:** clean (caveats)
**Criteria coverage:**
- [✓] Correctly distinguishes BUFFERED (use BufStrides which are bytes)
  vs. non-BUFFERED (multiply element strides by ElementSizes).
- [⚠] Repurposes `_state->InnerStrides` as a cache for byte strides.
  Side-effect: any subsequent code that reads `InnerStrides` expecting
  element strides will be wrong. Document or rename.

---

## File: src/NumSharp.Core/Backends/Iterators/NpyIter.Execution.Custom.cs

### Function: `ExecuteRawIL(Action<ILGenerator>, string, void*)` — line 41-46
**Severity:** clean
- [✓] Tier 3A escape hatch; compiles user IL once per cacheKey, runs via
  `ForEach`.

### Function: `ExecuteElementWise(NPTypeCode[], ..., string)` — line 74-88
**Severity:** clean
- [✓] Tier 3B with scalar + optional vector body, NOp length validation.

### Function: `ExecuteExpression(NpyExpr, NPTypeCode[], NPTypeCode, string?)` — line 138-153
**Severity:** clean
- [✓] Tier 3C DSL compile + ForEach drive.

---

## File: src/NumSharp.Core/Backends/Iterators/NpyIterBufferManager.cs

### Function: `CopyToBuffer(ref NpyIterState, int, long)` — line 166-194
**Severity:** bug
**Criteria coverage:**
- [✗] dtype coverage — switch handles only **11** NumSharp dtypes:
  Boolean, Byte, Int16, UInt16, Int32, UInt32, Int64, UInt64, Single,
  Double, Decimal, Char. **Missing:** SByte, Half, Complex.

**Reproduction:**
```csharp
var a = np.array(new Half[] {(Half)1, (Half)2, (Half)3});
var dtypes = new[] { NPTypeCode.Half };
var opFlags = new[] { NpyIterPerOpFlags.READONLY };
using var it = NpyIterRef.MultiNew(1, new[] {a},
    NpyIterGlobalFlags.BUFFERED, NPY_ORDER.NPY_KEEPORDER,
    NPY_CASTING.NPY_SAFE_CASTING, opFlags, dtypes);
// NotSupportedException: Buffer copy not supported for dtype Half
```

Same fails for `Complex`. SByte fails for a different reason (see NpyIterCasting).

```python
import numpy as np
# NumPy handles float16 + complex128 in buffered iteration
list(np.nditer(np.array([1,2,3], dtype=np.float16), flags=['buffered']))
list(np.nditer(np.array([1+0j], dtype=np.complex128), flags=['buffered']))
# Both work
```

**Remediation:** Add the three cases:
```csharp
case NPTypeCode.SByte:    CopyToBuffer<sbyte>(ref state, op, count); break;
case NPTypeCode.Half:     CopyToBuffer<Half>(ref state, op, count); break;
case NPTypeCode.Complex:  CopyToBuffer<Complex>(ref state, op, count); break;
```
And same for `CopyFromBuffer` (line 201-229).

### Function: `CopyFromBuffer(ref NpyIterState, int, long)` — line 201-229
Same bug as CopyToBuffer.

### Function: `CopyToBufferWithCast / CopyFromBufferWithCast` — line 234-304
**Severity:** bug
**Criteria coverage:**
- [✗] dtype coverage — relies on `NpyIterCasting.ConvertValue` which has
  the same gap (no SByte/Half/Complex). See NpyIterCasting findings.

### Function: `AllocateBuffers(ref NpyIterState, long)` — line 75-109
**Severity:** clean (limitation)
**Criteria coverage:**
- [✓] Only allocates buffer for operands that need it (CAST/CONTIG flag
  or non-contiguous).
- [⚠] `IsOperandContiguous` (line 130-159) uses `state.Shape` + `state.Strides`
  directly — correctly checks post-coalesce contiguity.

### Function: `CalculateGrowInnerSize` — line 478-528
**Severity:** parity-gap / dead code
**Criteria coverage:**
- [✗] Not invoked anywhere in the iterator. The `GROWINNER` flag is set
  on the state (NpyIter.cs:537-540) but `CalculateGrowInnerSize` is never
  called to actually grow the inner loop. NumPy's `npyiter_grow_buffers`
  is integral to buffered performance.

**Remediation:** Wire `CalculateGrowInnerSize` into `Initialize` after
buffer allocation, or invoke from `PrepareBuffers`.

---

## File: src/NumSharp.Core/Backends/Iterators/NpyIterCasting.cs

### Function: `IsSafeCast(NPTypeCode, NPTypeCode)` — line 50-108
**Severity:** bug
**Criteria coverage:**
- [✗] NumPy behavioral parity — SByte (int8) is not handled at all.
  `IsSignedInteger` (line 143-146) doesn't list SByte. So
  `IsSafeCast(SByte, Int32)` returns false despite NumPy declaring it safe.
- [✗] dtype coverage — Half and Complex aren't listed in `IsFloatingPoint`
  (line 138-141). So `IsSafeCast(Half, Single)` returns false even though
  it's a safe widening.

**Reproduction:**
```csharp
var a = np.array(new sbyte[] {-1, 2});
var dtypes = new[] { NPTypeCode.Int32 };
using var it = NpyIterRef.MultiNew(1, new[] {a},
    NpyIterGlobalFlags.BUFFERED, NPY_ORDER.NPY_KEEPORDER,
    NPY_CASTING.NPY_SAME_KIND_CASTING,
    new[] {NpyIterPerOpFlags.READONLY}, dtypes);
// InvalidCastException: int8 → int32 not allowed under 'same_kind'
```

```python
import numpy as np
# All true in NumPy
np.can_cast(np.int8, np.int32, 'safe')        # True
np.can_cast(np.int8, np.int16, 'safe')        # True
np.can_cast(np.float16, np.float32, 'safe')   # True
```

**Remediation:**
1. Add `NPTypeCode.SByte` to `IsSignedInteger`.
2. Add `NPTypeCode.Half` to `IsFloatingPoint`.
3. Decide how Complex casts behave (NumPy: complex → complex safe;
   complex → real same_kind). Add a `IsComplex` helper.

### Function: `IsSameKindCast` — line 113-136
Same gap (depends on `IsFloatingPoint`/`IsSignedInteger`).

### Function: `ReadAsDouble / WriteFromDouble` — line 339-381
**Severity:** bug
**Criteria coverage:**
- [✗] dtype coverage — 12 dtypes (Boolean, Byte, Int16, UInt16, Int32,
  UInt32, Int64, UInt64, Single, Double, Decimal, Char). **Missing:** SByte,
  Half, Complex.
- [✗] Loss of precision — Int64/UInt64 → double via `(double)*(long*)ptr`
  loses precision for values above 2^53. NumPy correctly upcasts.

**Reproduction:**
```csharp
var a = np.array(new sbyte[] {1, 2, 3});
var dtypes = new[] { NPTypeCode.Int64 };
using var it = NpyIterRef.MultiNew(1, new[] {a},
    NpyIterGlobalFlags.BUFFERED, NPY_ORDER.NPY_KEEPORDER,
    NPY_CASTING.NPY_UNSAFE_CASTING,
    new[] {NpyIterPerOpFlags.READONLY}, dtypes);
// NotSupportedException: Unsupported type: SByte
```

**Remediation:** Add the 3 missing dtypes. For exact integer casts,
generate a typed `ConvertValue<TSrc,TDst>` per type pair (15×15=225
combinations, but cacheable via `NpFunc.Invoke`).

### Function: `ConvertValue(void*, void*, NPTypeCode, NPTypeCode)` — line 318-333
**Severity:** bug
- [✗] Goes through `double` even for integer-to-integer casts → loses
  precision for int64 values > 2^53.
- [⚠] Same-type fast path uses `Buffer.MemoryCopy` (16-byte safety check)
  — fine but slower than `*(T*)dst = *(T*)src`.

### Function: `CopyStridedToContiguousWithCast` etc. — line 408-528
**Severity:** clean (modulo the dtype gaps above)
- [✓] Coordinate-based iteration; handles broadcasting and arbitrary strides.
- [✓] Calls `ConvertValue` per element — slow but functional.

---

## File: src/NumSharp.Core/Backends/Iterators/NpyIterCoalescing.cs

### Function: `CoalesceAxes(ref NpyIterState)` — line 19-114
**Severity:** clean
**Criteria coverage:**
- [✓] NumPy structural parity — matches `npyiter_coalesce_axes`
  (nditer_constr.c:2390). Coalesces adjacent axes when
  `stride[i] * shape[i] == stride[i+1]` for all operands.
- [✓] Size-1 dim handling absorbed into neighbor (relaxed from NumPy's
  stride==0 rule — documented at line 49-51, acceptable).
- [✓] Clears HASMULTIINDEX since coalescing invalidates original axes.
- [✓] Updates inner strides cache.

### Function: `ReorderAxesForCoalescing` — line 190-297
**Severity:** clean
**Criteria coverage:**
- [✓] NumPy structural parity — for K-order, sorts by stride; for C/F
  orders, deterministic ordering. Tracks identity in IDENTPERM flag.
- [✓] forCoalescing parameter controls ascending vs descending sort.

### Function: `FlipNegativeStrides` — line 410-493
**Severity:** clean
**Criteria coverage:**
- [✓] NumPy structural parity — matches `npyiter_flip_negative_strides`
  (nditer_constr.c:297). Only flips when ALL operands have non-positive
  strides on that axis (per NumPy invariant).
- [✓] Accumulates BaseOffsets so ResetBasePointers can recompute.
- [✓] Sets NEGPERM, clears IDENTPERM.

**Note:** Per the construction logic (NpyIter.cs:425), FlipNegativeStrides
runs only for K-order. C/F/A forced orders preserve user-requested
iteration order. This matches NumPy. Verified by C-order vs K-order on
reversed array yielding [9,8,..,0] vs [0,1,..,9] respectively.

### Function: `TryCoalesceInner` — line 120-167
**Severity:** clean (dead-code suspicion)
- [⚠] Appears to be utility not invoked from main construction path
  (which uses `CoalesceAxes`). Possibly leftover from earlier design.

---

## File: src/NumSharp.Core/Backends/Iterators/NpyIterFlags.cs

### Enums and constants
**Severity:** clean
- [✓] Comprehensive coverage of `NPY_ITFLAG_*`, `NPY_OP_ITFLAG_*`,
  `NPY_ITER_*` from NumPy's ndarraytypes.h.
- [✓] Each flag clearly documented with NumPy correspondence.
- [✓] `NpyIterUtils.ReductionAxis` matches `NPY_ITER_REDUCTION_AXIS` macro.
- [✓] `NpyArrayMethodFlags` packed in top-8 bits of `ItFlags`, matching
  NumPy's `NPY_ITFLAG_TRANSFERFLAGS_SHIFT = 24`.

---

## File: src/NumSharp.Core/Backends/Iterators/NpyIterKernels.cs

### Class: `NpyIterPathSelector`
**Severity:** parity-gap / dead-code
**Criteria coverage:**
- [⚠] Looks like predecessor of `DetectExecutionPath`. Has `Strided`
  (AVX2 gather) path that isn't wired into the main execution flow.

### Class: `NpyIterExecution`
**Severity:** dead-code
- [⚠] `ExecuteContiguous` / `ExecuteBuffered` / `ExecuteGeneral` / `Execute`
  appear to be early-design entry points. The current pipeline uses
  `NpyIter.Execution.cs` (`ForEach`, `ExecuteBinary`, etc.) and the IL
  kernels directly. The buffered path here has `TODO: Type dispatch for
  copy` comments showing it's incomplete.

**Remediation:** Delete or document as deprecated. Currently no callers
within the iterator subsystem invoke these.

---

## File: src/NumSharp.Core/Backends/Iterators/NpyLogicalReductionKernels.cs

### Struct kernels: `NpyAllKernel<T>`, `NpyAnyKernel<T>`, `CountNonZeroKernel<T>`
**Severity:** clean
- [✓] Use `EqualityComparer<T>.Default.Equals` — devirtualized by JIT for
  unmanaged structs. No boxing.
- [✓] Generic over T, works for all 14 unmanaged dtypes.

### Axis kernels: `NpySumAxisKernel<T>`, `NpyProdAxisKernel<T>`, etc.
**Severity:** clean
**Criteria coverage:**
- [✓] Generic constraint `IAdditionOperators<T,T,T>` etc.; static abstract
  members; zero-cost dispatch.
- [✓] Stride-aware (`src[i * srcStride]`).
- [N/A] SIMD — these are scalar; the SIMD paths live in ILKernelGenerator.

---

## File: src/NumSharp.Core/Backends/Iterators/NpyNanReductionKernels.cs

### Kernels: `NanSumFloatKernel`, `NanMeanFloatKernel`, `NanMaxFloatKernel`, etc.
**Severity:** parity-gap
**Criteria coverage:**
- [✓] NaN-skipping logic matches NumPy.
- [✓] Stride-aware.
- [✗] dtype coverage — only Float and Double provided. **Missing:** Half,
  Complex.
- [N/A] All-NaN behavior: `NanMeanAccumulator` returns Sum=0, Count=0,
  so caller must divide by 0. NumPy: `nanmean([nan]) == nan` (with warning).
  Verify caller handles.

**Remediation:** Add Half-typed kernels (read as Half, accumulate as
double). Complex `nanmean` is also valid in NumPy (separates real/imag).

---

## File: src/NumSharp.Core/Backends/Iterators/NpyAxisIter.cs

### Class: `NpyAxisIter`
**Severity:** refactor
**Criteria coverage:**
- [✗] NumPy structural parity — parallel implementation; NumPy uses one
  unified `nditer`. The existence of two separate iterator implementations
  is itself a divergence from NumPy.
- [✓] Per-dtype kernels via `INpyAxisNumericReductionKernel<T>`.
- [✗] NPY_MAXDIMS=64 limit (line 9: `MaxDims = 64`) vs. NumSharp's claim of
  unlimited dimensions.

**Finding:** NpyAxisIter is a duplicate path for axis reductions that
predates the full NpyIter migration. The README claims NumSharp supports
unlimited dimensions, but this iterator caps at 64.

**Remediation:**
- Long-term: Migrate axis reductions to use `NpyIter.AdvancedNew` with
  appropriate `op_axes` (-1 for reduce target).
- Short-term: Drop the MaxDims cap by using dynamic allocation matching
  NpyIterState.

### Functions: `ExecuteSameType<T,TKernel>`, `ReduceDouble<TKernel>`, `ReduceBool<T,TKernel>`, `ReduceNumeric<T,TKernel>`
**Severity:** clean within scope (modulo the structural issue)
- [✓] Strided traversal; non-axis dims iterated with ripple coords.
- [✓] No allocations in the hot loop.

---

## File: src/NumSharp.Core/Backends/Iterators/NpyAxisIter.State.cs

### Struct: `NpyAxisState`
**Severity:** parity-gap
- [✗] `fixed long OuterShape[64]` etc. — locks at NPY_MAXDIMS=64, matches
  NumPy's old limit but not NumSharp's stated "unlimited" design.

---

## File: src/NumSharp.Core/Backends/Iterators/NpyExpr.cs

### Class: `NpyExpr` (and subclasses InputNode, BinaryNode, …)
**Severity:** parity-gap
**Criteria coverage:**
- [✓] Tier-3C DSL design is sound: tree → IL via shell wrapping; cacheable.
- [✓] SIMD path enabled when all inputs match output type and op is
  SIMD-friendly.
- [✗] dtype coverage in `CallNode.IsSupported` (line 973-980) — 12 dtypes
  (missing SByte, Half, Complex). Constants in `EmitLoadTyped`
  (line 406-432) — 11 dtypes (missing Decimal, SByte, Half, Complex).
  `WhereNode.EmitPushZero` (line 762-790) — 11 dtypes (missing SByte,
  Half, Complex).

**Reproduction:**
```csharp
NpyExpr.Const(1.5).Compile(new[] {NPTypeCode.Half}, NPTypeCode.Half, "test");
// NotSupportedException at line 430 (ConstNode.EmitLoadTyped)
```

**Remediation:** Add SByte/Half/Complex emission paths. Half needs
explicit cast IL via `[System.Half]::op_Implicit(Single)` etc.

### Function: `BinaryNode.IsSimdOp` — line 465-469
**Severity:** clean
- [✓] Conservative SIMD predicate: only includes ops actually emitted
  in `EmitVectorOperation` (Add/Sub/Mul/Div, BitwiseAnd/Or/Xor).
- [⚠] Mod/Power/FloorDivide/ATan2 stay scalar — matches NumPy nditer
  capability (these aren't SIMD ufuncs in NumPy either).

### Function: `MinMaxNode.EmitBranchy` — line 655-698
**Severity:** clean
- [✓] Prefers `Math.Min/Max` (NaN-propagating per IEEE 754 — matches NumPy
  `np.minimum/np.maximum`).
- [✓] Branchy fallback for Char/Boolean.

---

## File: src/NumSharp.Core/Backends/Iterators/NDIterator.cs

### Class: `NDIterator<TOut>`
**Severity:** bug
**Criteria coverage:**
- [✓] Same-type iteration over arbitrary layouts works for all 14 dtypes
  (verified Boolean..Complex).
- [✓] Cross-dtype via `BuildCastingMoveNext<TSrc>` and `Converts.FindConverter`.
- [✗] Broadcast constructor `NDIterator(IMemoryBlock, Shape, Shape?,
  bool)` is broken.

### Function: `NDIterator(IMemoryBlock, Shape shape, Shape? broadcastedShape, bool)` — line 85-112
**Severity:** bug
**Criteria coverage:**
- [✗] Broadcasting — calls `UnmanagedStorage.CreateBroadcastedUnsafe(srcSlice, effShape)`
  which produces shape=effShape but strides=C-order strides for effShape
  (not broadcast strides=0). Then NpyIter iterates effShape.size elements
  off a buffer that only has shape.size elements.

**Reproduction:**
```csharp
var smaller = np.array(new long[] {1, 2, 3});
var bigger = new Shape(new int[]{4, 3});
using var it = new NDIterator<long>(smaller.Storage.InternalArray,
    smaller.Shape, bigger, autoReset: false);
while (it.HasNext()) it.MoveNext();
// OutOfMemoryException: Array dimensions exceeded supported range
```

The corresponding extension methods
`NDIteratorExtensions.AsIterator(IArraySlice, Shape, Shape, bool)`
(line 72) and `CreateFromSliceBroadcast<T>` (line 89) inherit the bug.

```python
import numpy as np
# NumPy broadcasts correctly
a = np.array([1,2,3])
b = np.broadcast_to(a, (4,3))
print(list(b.flat))  # [1,2,3,1,2,3,1,2,3,1,2,3]
```

**Remediation:** Compute broadcast strides explicitly:
```csharp
// Use np.broadcast_to to get correctly-broadcast shape with stride=0 dims
var bcastNd = np.broadcast_to(new NDArray(...), broadcastedShape.Value);
_srcKeepAlive = bcastNd;
_state = InitState(bcastNd);
```

### Function: `SetDelegates(NPTypeCode srcType)` — line 160-195
**Severity:** clean
- [✓] All 14 dtypes mapped (Boolean..Complex).
- [✓] Same-type uses direct pointer deref; cross-type uses `Converts.FindConverter`.
- [✓] `MoveNextReference` throws when casting (correct).

### Function: `EnsureNext / Dispose / Finalizer` — line 198-279
**Severity:** clean
- [✓] AutoReset handled.
- [✓] State pointer owned by class, freed via `NpyIterRef.FreeState`.
- [✓] Finalizer covers GC-cleanup if user forgets to Dispose.

---

## File: src/NumSharp.Core/Backends/Iterators/NDIteratorExtensions.cs

### Function: `AsIterator(IArraySlice, Shape, Shape, bool)` — line 72-75
**Severity:** bug
- [✗] Routes to `NDIterator` broadcast constructor which is broken (see above).

---

## File: src/NumSharp.Core/Backends/Iterators/INDIterator.cs
**Severity:** clean
- [✓] Minimal interface; concrete impl in `NDIterator<TOut>`.

---

## Performance findings

### Finding P1: `Iternext()` per-element is ~20× slower than NumPy nditer + EXLOOP

**Reproduction:**
```python
# NumPy
import numpy as np, time
a = np.arange(10_000_000, dtype=np.int64)
t = time.perf_counter(); total = 0
for x in np.nditer(a, flags=['external_loop']): total += x.sum()
# ~4.8ms (10M elements)
```

```csharp
// NumSharp NpyIter per-element via Iternext()
var a = np.arange(10_000_000, NPTypeCode.Int64);
var sw = Stopwatch.StartNew();
long total = 0;
using (var it = NpyIterRef.New(a, NpyIterGlobalFlags.None,
    NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING))
    do { total += *(long*)it.RawState->DataPtrs[0]; } while (it.Iternext());
// ~94.3ms

// Same loop with EXLOOP + direct pointer + Shape[NDim-1] inner count: ~9.9ms
// Direct unsafe loop: ~8.1ms
```

NumPy: 4.8ms. NumSharp Iternext: 94.3ms. Ratio: **~20×**.

The 9.9ms EXLOOP path is ~2× NumPy's number, which is more reasonable.
The 20× gap on Iternext is **a result of Bug 5** — every Iternext is a
single-element advance with a virtual dispatch through `NpyIterNextFunc`.
Fixing Iternext to use the right dispatcher closes most of this gap.

### Finding P2: NDIterator legacy wrapper is ~7× slower than direct strided pointer loop

**Reproduction:**
```csharp
var a = np.arange(1_000_000, NPTypeCode.Int64).reshape(1000, 1000).transpose();
// NDIterator<long>: 17.8ms
// NpyIter Iternext: 9.2ms
// Direct strided ptr loop: 2.5ms
```

NumPy `a.sum()` is 0.4ms (uses C-level reduction). NumPy `a.flat` python
loop is 75ms (Python overhead). NumSharp's NDIterator wrapper at 17.8ms
is reasonable for a Func-delegate iteration but loses ~7× vs the direct
loop. This is normal cost of delegate dispatch; if perf-critical, callers
should use the typed `NpyIterRef` API.

---

## Missing NumPy API surface

The following `numpy.nditer` Python API methods are absent from `NpyIterRef`:

| NumPy method | NumSharp equivalent | Notes |
|---|---|---|
| `close()` | `Dispose()` | Different naming; OK |
| `dtypes` | `GetDescrArray()` | OK |
| `iter_range` (getter) | `IterRange` | OK |
| `iter_range` (setter) | `ResetToIterIndexRange` | Different API shape |
| `iterindex` | `IterIndex` | OK |
| `iterationneedsapi` | **missing** | NumPy: bool; for NumSharp, always false |
| `iternext` | `Iternext()` | **buggy — see Bug 5** |
| `itviews` | `GetIterView(int)` | **buggy on 0-d arrays** |
| `multi_index` | `GetMultiIndex(Span<long>)` | OK |
| `ndim` | `NDim` | OK |
| `nop` | `NOp` | OK |
| `operands` | `GetOperandArray()` | OK |
| `remove_axis` | `RemoveAxis(int)` | OK |
| `remove_multi_index` | `RemoveMultiIndex()` | OK |
| `reset` | `Reset()` | OK |
| `shape` | `Shape` | **parity-gap — returns post-coalesce, not original** |
| `value` | (none) | NumSharp users access via `GetValue<T>` |
| `enable_external_loop` | `EnableExternalLoop()` | **buggy — no MULTI_INDEX check** |
| `has_delayed_bufalloc` | **missing** | |
| `has_index` | `HasIndex` | OK |
| `has_multi_index` | `HasMultiIndex` | OK |
| `finished` | `Finished` | OK |
| `debug_print` | `DebugPrint()` | OK |
| `copy` | `Copy()` | OK |

---

## Summary table (severity × file:line)

| # | Severity | File | Location | Finding |
|---|---|---|---|---|
| 1 | bug | NpyIter.cs | 1985-2003 | `Iternext()` ignores EXLOOP — outer count is `IterSize` instead of `IterSize/Shape[NDim-1]`. User-driven kernels read past buffer end. Verified by Python+C# repro. |
| 2 | bug | NpyIter.cs | 1985-2003 + 2723-2756 | `Iternext()` BUFFERED non-reduce path has no buffer-refill logic. Causes segfault when `IterSize > BufferSize` (e.g. cast on 20K-element int32 array). |
| 3 | bug | NpyIterBufferManager.cs | 178-194, 213-229 | `CopyToBuffer<T>` / `CopyFromBuffer<T>` switch missing SByte, Half, Complex → `NotSupportedException`. |
| 4 | bug | NpyIterCasting.cs | 50-108, 113-136, 138-151 | `IsSafeCast/IsSameKindCast/IsFloatingPoint/IsSignedInteger` don't handle SByte, Half, Complex → false negatives (int8 → int32 reported unsafe). |
| 5 | bug | NpyIterCasting.cs | 339-381 | `ReadAsDouble/WriteFromDouble` missing SByte, Half, Complex → NotSupportedException at runtime. Also precision loss on int64 > 2^53. |
| 6 | bug | NDIterator.cs | 85-112 + Extensions:72-75 | NDIterator broadcast constructor `CreateBroadcastedUnsafe(slice, effShape)` produces wrong strides (C-order, not stride=0) → reads past source storage / OOM. |
| 7 | bug | NpyIter.cs | 2852-2857 | `EnableExternalLoop()` doesn't validate HASINDEX/HASMULTIINDEX — NumPy raises ValueError. |
| 8 | bug | NpyIter.cs | 2650-2704 | `GetIterView(0)` on 0-d array throws OverflowException (uses `original.flat[0]` which fails on scalars). |
| 9 | parity-gap | NpyIter.cs | 2494-2520 | `Shape` property returns post-coalesce internal shape, not original itershape. NumPy's `.shape` is the original. |
| 10 | parity-gap | NpyIter.cs | 1094-1249 | `SetupBufferedReduction` falls back to slow path when `nonReduceAxisCount > 1`; multi-axis reductions lose buffering. |
| 11 | parity-gap | NpyExpr.cs | 406-432, 762-790, 973-980 | `ConstNode`, `WhereNode`, `CallNode` only support 12 dtypes (no SByte, Half, Complex). Decimal partially supported. |
| 12 | parity-gap | NpyNanReductionKernels.cs | 1-344 | NaN kernels only Float/Double — missing Half, Complex. |
| 13 | parity-gap | NpyIterBufferManager.cs | 478-528 | `CalculateGrowInnerSize` (GROWINNER) is implemented but **never called** → no inner-loop growing despite flag being set. |
| 14 | parity-gap | NpyAxisIter.State.cs | 9-22 | `MaxDims = 64` hard cap conflicts with NumSharp's "unlimited" design. |
| 15 | perf | NpyIter.cs | 1985-2003 | `Iternext()` is **~20× slower** than NumPy nditer+EXLOOP on 10M elements (94ms vs 4.8ms), a direct consequence of Bug 1. Fixing Bug 1 should close most of this gap. |
| 16 | refactor | NpyAxisIter.cs | full file | Parallel axis-iterator implementation duplicating NpyIter functionality. Migrate to NpyIter.AdvancedNew with op_axes. |
| 17 | refactor | NpyIterKernels.cs | full file | Class `NpyIterExecution` is dead code (TODO comments, no callers); `NpyIterPathSelector.Strided` (AVX2 gather) not wired up. |
| 18 | refactor | NpyIterCoalescing.cs | 120-167 | `TryCoalesceInner` not invoked from main path — possibly dead. |
| 19 | refactor | NpyIter.cs | 3469 lines in one file | Split into partial classes by concern (Construction, MultiIndex, Lifecycle, Debug) per the existing `.State`/`.Execution` precedent. |
| 20 | clean | NpyIterCoalescing.cs | 410-493 | `FlipNegativeStrides`: structurally and behaviorally correct; only flips when all operands have non-positive stride; updates BaseOffsets + NEGPERM. Matches NumPy. |
| 21 | clean | NpyIter.State.cs | 712-765 | `Advance()`: correct ripple-carry; updates FlatIndex via fast/general path. |
| 22 | clean | NpyIter.Execution.cs | 126-243 | `ForEach`/`ExecuteGeneric`/`ExecuteReducing` correctly use `GetIterNext()` and `BufStrides`. **These are the recommended user APIs**, sidestepping the buggy `Iternext()`. |
| 23 | clean | NpyLogicalReductionKernels.cs | all | Reduction kernels use `EqualityComparer<T>.Default` (JIT-devirtualized) and `IAdditionOperators<T,T,T>` (static abstracts) — no boxing, generic over 14 dtypes. |
| 24 | clean | NpyIter.cs | 1884-1943 | `ResetBasePointers`: BaseOffsets accounting + buffer flush + re-prime is structurally identical to NumPy. |

---

## Followups (prior-audit bugs revisited)

**Bug 5 of prior audit ("Iternext ignores EXLOOP" and "buffered-with-cast stride mismatch"): CONFIRMED with reproductions.** This audit splits it into two distinct bugs (Findings #1 and #2 above) and adds the segfault repro showing the BUFFERED non-reduce path has no refill logic at all (worse than just a stride mismatch). The bridge layer in `NpyIter.Execution.cs` carefully avoids both bugs by using `GetIterNext()` + `BufStrides`, but the public `Iternext()` API is still broken.
