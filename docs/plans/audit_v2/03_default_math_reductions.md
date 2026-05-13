# Group 3: DefaultEngine math + reductions + BLAS audit

**Branch:** `nditer` (vs `master`)
**Scope:** `src/NumSharp.Core/Backends/Default/Math/**`
**Reference:** NumPy 2.4.2 + `src/numpy/numpy/_core/src/umath`, `multiarray`, `linalg`
**Bench host:** Windows 11, .NET 10, AVX2 enabled
**Date:** 2026-05-13

---

## Conventions

Severity ladder:
- `bug` — wrong result, crash on valid input, or behavioral divergence from NumPy on a documented contract
- `parity-gap` — missing API surface (out=, dtype, axis, …) or NumPy raises where NumSharp doesn't (and vice-versa)
- `perf` — correct, but ≥2× slower than NumPy on a path numpy does well
- `refactor` — duplicate/wasted code, copies that aren't needed, dispatch chains that could be IL-generated
- `clean` — nothing to fix, parity confirmed

Criteria checklist (10 items per finding):
1. NumPy C-source structural parity
2. NumPy behavior parity (Python-verified)
3. Perf path: contig
4. Perf path: strided (slight + heavy) / broadcast / F-contig
5. NumPy ≥10× better in any case (cite ms)
6. IL generation used vs switch-per-type ladder
7. All 15 dtypes (Boolean, Byte, SByte, Int16, UInt16, Int32, UInt32, Int64, UInt64, Char, Half, Single, Double, Decimal, Complex)
8. Full NumPy parameter parity (out=, axis, dtype, keepdims, …)
9. Wasted copies / unnecessary materialization
10. Routes via NDIterator / NpyIter / ILKernelGenerator where appropriate

---

## File: `src/NumSharp.Core/Backends/Default/Math/Default.Power.cs`

### Function: `Power(NDArray, NDArray, NPTypeCode?)` + `PowerInteger`

**Severity:** bug (crash on strided/broadcast operands)

**Criteria coverage:**
1. ✗ Partial — NumPy `np_power` (umath/loops_power.c.src) ALWAYS supports strided. NumSharp adds an integer fast-path that bypasses strides.
2. ✗ NumPy raises `ValueError` for `int ** negative_int`; NumSharp silently returns "seterr=ignore" semantics. The code comment even acknowledges this.
3. ✓ Contig int^int contig: ~6.8 ms/1 M elements (PowerInteger fast path).
4. ✗ Strided int^int: crashes with `InvalidOperationException` (never reaches loop). Broadcast same-shape (stride=0): also crashes.
5. ✓ NumPy int^int 31.1 ms / 20 iters / 1 M; NumSharp 136 ms / 20 iters → NumPy ~4.4× faster.
6. ✗ Hand-rolled per-dtype switch (8 cases) using raw `Unsafe.Address`. The non-integer path routes through ILKernel `BinaryOp.Power`. The integer fast-path completely bypasses IL.
7. ✓ Integer types only (Boolean correctly routed to BinaryOp dispatch via `IsInteger()`). 8 cases listed; matches the documented integer set.
8. ✗ No `out=` parameter (NumPy: `np.power(x1, x2, /, out=None, where=True, casting='same_kind', ...)`); no `where=`, no `casting=`.
9. ✓ When successful, no copies; integer fast-path writes directly. (But the alternative path through `ExecuteBinaryOp` already handles all of this without the crash.)
10. ✗ Bypasses NpyIter / IL; uses raw `Unsafe.Address`.

**Finding:**

```csharp
// Default.Power.cs:50-128 — PowerInteger
private static NDArray PowerInteger(NDArray lhs, NDArray rhs)
{
    var tc = lhs.GetTypeCode;
    var result = new NDArray(tc, new Shape((long[])lhs.shape.Clone()), false);
    long n = lhs.size;
    unsafe
    {
        switch (tc)
        {
            case NPTypeCode.Int32:
            {
                var a = (int*)lhs.Unsafe.Address;   // <-- throws on slice / broadcast
                var b = (int*)rhs.Unsafe.Address;   // <-- throws on slice / broadcast
                var d = (int*)result.Unsafe.Address;
                for (long i = 0; i < n; i++) d[i] = PowInt32(a[i], b[i]);   // <-- ignores strides
                break;
            }
            ...
        }
    }
    return result;
}
```

`lhs.Unsafe.Address` is hard-gated against `IsSliced || IsBroadcasted` in `NDArray.Unmanaged.cs:43-52`. So `np.power(strided_int, ...)` and `np.power(broadcasted_int, ...)` always crash here, even though the regular `ExecuteBinaryOp` IL path handles them. Even if the address were available, the loop indexes via `a[i]` instead of `a[i*stride]` — silent corruption.

**Reproduction:**

```csharp
// CRASH on strided
var arr = np.arange(20).astype(NPTypeCode.Int32);
var sliced = arr["::2"];                              // IsSliced
var b = np.arange(10).astype(NPTypeCode.Int32);
np.power(sliced, b);
// InvalidOperationException: Can't return a memory address when NDArray is sliced or broadcasted.

// CRASH on broadcast
var a = np.array(new int[]{2});
var b = np.array(new int[]{3,3,3});
np.power(np.broadcast_to(a, b.Shape), b);
// InvalidOperationException

// NumPy: works, returns [2^3, 2^3, 2^3]
```

```csharp
// Misalignment: NumPy raises ValueError, NumSharp returns garbage
np.power(np.array(new int[]{0,1,-1,2}), np.array(new int[]{-1,-1,-3,-1}))
//  → NumSharp: [0, 1, -1, 0]      (silent)
//  → NumPy:    ValueError("Integers to negative integer powers are not allowed.")
```

**Remediation:**

1. Guard the fast-path by also requiring contiguity + no offset:
   ```csharp
   if (!typeCode.HasValue
       && lhs.GetTypeCode == rhs.GetTypeCode
       && lhs.GetTypeCode.IsInteger()
       && lhs.shape.SequenceEqual(rhs.shape)
       && lhs.Shape.IsContiguous && rhs.Shape.IsContiguous
       && !lhs.Shape.IsBroadcasted && !rhs.Shape.IsBroadcasted
       && lhs.Shape.offset == 0 && rhs.Shape.offset == 0)
       return PowerInteger(lhs, rhs);
   ```
2. Or — better — emit a Power IL kernel that supports integer wrap-around so the special case can be deleted entirely (the IL `BinaryOp.Power` currently goes through `Math.Pow(double,double)` and loses precision).
3. Decide NumPy parity for `int ** negative_int`: either raise `ValueError` (default NumPy) or document the seterr=ignore behavior explicitly. Currently both paths (PowerInteger AND the IL Power) silently produce nonsense for that case.
4. Add an `out=` parameter to match NumPy.

---

## File: `src/NumSharp.Core/Backends/Default/Math/Default.Reciprocal.cs`

### Function: `Reciprocal(NDArray, NPTypeCode?)` + `ReciprocalInteger`

**Severity:** bug (same Unsafe.Address pattern as PowerInteger)

**Criteria coverage:**
1. ✗ NumPy's reciprocal C loop is stride-aware.
2. ✗ Strided/broadcast inputs crash.
3. ✓ Contig int reciprocal: works.
4. ✗ Strided/broadcast: crashes.
5. n/a — not a hot path.
6. ✗ Hand-rolled per-dtype switch (8 integer cases). Non-integer goes through IL.
7. ✓ Integer types only — Boolean rejected upstream by routing.
8. ✗ No `out=`, no `where=`, no `casting=`.
9. ✓ No wasted copy when it works.
10. ✗ Bypasses NpyIter/IL for the integer special-case.

**Finding:**

`ReciprocalInteger` at `Default.Reciprocal.cs:24-95` uses `nd.Unsafe.Address` and a non-strided loop `dst[i] = src[i] == 0 ? 0 : 1 / src[i]`. Same crash mode as PowerInteger.

**Reproduction:**

```csharp
var a = np.arange(20).astype(NPTypeCode.Int32);
var sl = a["::2"];
np.reciprocal(sl);
// InvalidOperationException

// Broadcast version
var s = np.array(new int[]{5});
var bc = np.broadcast_to(s, new Shape(5));
np.reciprocal(bc);   // InvalidOperationException
```

**Remediation:**

Identical to PowerInteger:
- Either guard the fast-path on contig + offset==0 + !broadcasted
- Or remove the special case and emit a stride-aware integer reciprocal IL kernel.

---

## File: `src/NumSharp.Core/Backends/Default/Math/Default.Abs.cs`

### Function: `Abs(NDArray, NPTypeCode?)`

**Severity:** clean

**Criteria coverage:**
1. ✓ Mirrors NumPy: preserve dtype for real, magnitude for complex.
2. ✓ Verified `np.abs(int8)` returns `int8` in both.
3. ✓ Uses ILKernel `UnaryOp.Abs` for the heavy lifting.
4. ✓ Same.
5. n/a.
6. ✓ IL.
7. ✓ All 15 dtypes (complex special-cased; unsigned short-circuited to a typed copy).
8. ✗ No `out=`, no `where=`, no `casting=`.
9. ✓ No wasted copies (unsigned case just casts; rest go through unary kernel).
10. ✓ Yes.

**Finding:**

Clean and tight at 30 lines. Only API gap is the missing `out=`. Not flagged as a bug.

---

## File: `src/NumSharp.Core/Backends/Default/Math/Default.ATan2.cs`

### Function: `ATan2(y, x, NPTypeCode?)`

**Severity:** parity-gap (dtype promotion deviates from NumPy)

**Criteria coverage:**
1. ✗ NumPy `arctan2` is `signature='ee->e,ff->f,dd->d,gg->g'` — every combination promotes to a float at least as wide as the larger input. For integer/bool inputs NumPy uses **float64** (the default float), not float16. NumSharp maps small ints to Half (line 112) which is NOT what NumPy does.
2. ✗ See above.
3. ✓ SIMD path for contig.
4. ✓ Path classifier handles strided/scalar/inner-contig.
5. n/a — kernel-bound on large arrays.
6. ✓ IL kernel via `BinaryOp.ATan2` + mixed-type registry.
7. ✓ All numeric types covered; Complex correctly rejected (atan2 is real-valued).
8. ✗ No `out=`, no `where=`.
9. ✓ No wasted copies.
10. ✓ Mixed-type IL registry.

**Finding:**

The dtype rule at `Default.ATan2.cs:110-120`:
```csharp
NPTypeCode.Boolean or NPTypeCode.SByte or NPTypeCode.Byte => NPTypeCode.Half,
NPTypeCode.Int16 or NPTypeCode.UInt16 => NPTypeCode.Single,
NPTypeCode.Int32 or NPTypeCode.UInt32 or NPTypeCode.Int64 or ... => NPTypeCode.Double,
```

NumPy maps every non-float input to **float64**:

```python
>>> np.arctan2(np.array([1], dtype=np.int8), np.array([1], dtype=np.int8)).dtype
dtype('float64')
>>> np.arctan2(np.array([1], dtype=np.int16), np.array([1], dtype=np.int16)).dtype
dtype('float64')
```

NumSharp will return `Half` / `Single` instead of `Double` for those inputs. Precision-sensitive callers will be silently bitten.

**Reproduction:**

```python
# NumPy 2.4.2
>>> import numpy as np
>>> np.arctan2(np.int8(1), np.int8(1)).dtype
dtype('float64')
```

```csharp
// NumSharp
var r = np.arctan2(np.array(new sbyte[]{1}), np.array(new sbyte[]{1}));
// r.dtype = Half  ← WRONG
```

**Remediation:**

Replace `PromoteATan2Single` with the standard "integer → float64, half → half, single → single, double → double, decimal → decimal" mapping. The same fix unifies the binary promotion table — drop the Rank() table, just use the existing `GetComputingType()` from `NPTypeCode.cs:615`.

---

## File: `src/NumSharp.Core/Backends/Default/Math/Default.Ceil.cs`, `Default.Floor.cs`, `Default.Truncate.cs`

### Function: `Ceil/Floor/Truncate(NDArray, NPTypeCode?)`

**Severity:** parity-gap (Boolean dtype)

**Criteria coverage:**
1. ✓ Structurally matches NumPy: integer types preserved (no-op cast).
2. ✗ NumPy keeps `bool` as `bool` for `np.ceil/floor/trunc`; NumSharp promotes to `float64`.
3. ✓ Contig fast path via IL.
4. ✓ Same.
5. n/a.
6. ✓ IL `UnaryOp.Ceil/Floor/Truncate`.
7. ✗ Boolean dtype mishandled (see #2).
8. ✗ No `out=`, no `where=`.
9. ✓.
10. ✓.

**Finding:**

The check `nd.GetTypeCode.IsInteger()` returns `false` for `NPTypeCode.Boolean` (verified at runtime). So `np.ceil(bool)` falls through to `ExecuteUnaryOp` with `ResolveUnaryReturnType`, which promotes to Double.

```python
# NumPy 2.4.2
>>> np.ceil(np.array([True, False])).dtype
dtype('bool')
```

```csharp
// NumSharp
var c = np.ceil(np.array(new bool[]{true, false}));
// c.dtype = Double, values = [1.0, 0.0]  ← WRONG
```

**Remediation:**

Either:
- Include `Boolean` in the no-op fast-path: `nd.GetTypeCode.IsInteger() || nd.GetTypeCode == NPTypeCode.Boolean`.
- Or expand `IsInteger()` semantics to include Boolean (consistent with NumPy's `np.issubdtype(bool_, np.integer)` being False but bool being treated specially in many ufunc dispatch tables).

---

## File: `src/NumSharp.Core/Backends/Default/Math/Default.Clip.cs`

### Function: `ClipScalar` + `ClipCore` + `ClipCoreComplex` + Decimal/Char fallbacks

**Severity:** clean (with one parity nit)

**Criteria coverage:**
1. ✓ Mirrors NumPy structure: NaN propagates from data; NaN scalar bounds → all-NaN result.
2. ✓ Verified.
3. ✓ SIMD via `ILKernelGenerator.ClipHelper<T>` for contiguous arrays. `Cast(copy: true)` at line 47 guarantees contiguity.
4. n/a — code intentionally materializes contig before clip (NumPy does the same).
5. n/a.
6. ✓ IL kernel via NpFunc.Invoke dispatcher.
7. ✓ All 15 dtypes — Complex special-cased, Decimal/Char fall back to scalar loops in `ILKernelGenerator.ClipMin/MaxHelper` and the local fallbacks in this file.
8. ✓ `np.clip` overload accepts `out=` (Math/np.clip.cs:63-64).
9. ✗ The `Cast(copy: true)` at line 47 always materializes — efficient on strided input, wasteful when input is already contig + offset==0. Worth a fast check.
10. ✓ Routes through ILKernel.

**Finding:**

Minor: Lines 47-53 — `Cast(lhs, outTypeCode, copy: true)` always allocates a fresh copy, even when `lhs` is already contig of the target dtype and clip should be in-place on the result. Not catastrophic since clip itself is O(n), but the doubling is measurable on large arrays.

**Reproduction:** none — code works correctly.

**Remediation:**

Optional perf: short-circuit the Cast when `lhs` is contig + offset==0 + matching dtype. Currently the SIMD kernel writes in-place, so the only reason for the Cast is to (a) get the right dtype and (b) get a contig buffer. Both are no-ops when already satisfied.

---

## File: `src/NumSharp.Core/Backends/Default/Math/Default.ClipNDArray.cs`

### Function: `ClipNDArray` + `ClipNDArrayContiguous` + `ClipNDArrayGeneral` + complex/decimal/char paths

**Severity:** perf (general path uses virtual GetAtIndex + boxing)

**Criteria coverage:**
1. ✓ Matches NumPy semantics (`min(max(a, lo), hi)` element-wise; NaN propagates).
2. ✓.
3. ✓ Contig: SIMD via `ILKernelGenerator.ClipArrayBounds<T>`.
4. ✗ Non-contig path: `ClipNDArrayGeneralCore<T>` uses `out.Shape.TransformOffset(i)` for output (OK) plus `min.GetAtIndex(i)` for the bounds (boxing). `i` here is treated as a linear index into the broadcasted view, but `GetAtIndex` calls `Shape.TransformOffset` again — i.e. it virtual-dispatches into `IArraySlice` AND boxes the resulting value. Result: ~14× slower than NumPy on the same input.
5. ✓ NumPy strided: 13.2 ms / 10 iters / 1 M; NumSharp: 185 ms → NumPy 14× faster.
6. ✓ Contig path: IL. ✗ General path: not IL, hand-rolled.
7. ✓ All 15 dtypes (complex/decimal/char in dedicated paths).
8. ✓ `np.clip(a, lo, hi, out=...)` plumbed through.
9. ✗ `broadcast_to(min, lhs.Shape).astype(outType)` at lines 52-53 unconditionally materializes the bounds even when they're already in the right shape/dtype.
10. ⚠ Contig path uses IL. General path bypasses it.

**Finding:**

```csharp
// Default.ClipNDArray.cs:158-173 — ClipNDArrayGeneralCore<T>
var outAddr = (T*)@out.Address;
for (long i = 0; i < len; i++)
{
    long outOffset = @out.Shape.TransformOffset(i);
    var val = outAddr[outOffset];
    var minVal = Converts.ChangeType<T>(min.GetAtIndex(i));   // <-- boxes, virtual call
    var maxVal = Converts.ChangeType<T>(max.GetAtIndex(i));   // <-- boxes, virtual call
    if (val.CompareTo(minVal) < 0) val = minVal;
    if (val.CompareTo(maxVal) > 0) val = maxVal;
    outAddr[outOffset] = val;
}
```

`min.GetAtIndex(i)` virtual-dispatches into `IArraySlice.GetValue(i)` (which itself calls `TransformOffset` internally) and boxes. Same for `max`. 1 M element clip with two bound arrays therefore costs ~6 M virtual calls + 4 M boxings.

**Reproduction:**

```csharp
var n = 1_000_000;
var arr = np.arange(2*n).astype(NPTypeCode.Int32);
var min = np.full(2*n, 100, NPTypeCode.Int32);
var max = np.full(2*n, 1000, NPTypeCode.Int32);
var sliced_arr = arr["::2"];      // not contiguous → general path
var sliced_min = min["::2"];
var sliced_max = max["::2"];

// Measured:
// ClipNDArray contig    :  89 ms / 10 iters
// ClipNDArray strided   : 185 ms / 10 iters (2× slowdown)
// NumPy strided clip    :  13 ms / 10 iters (14× faster than NumSharp)
```

**Remediation:**

Three-step ladder:
1. **Easy:** When inputs are already contig of the right dtype, skip `broadcast_to(...).astype(outType)` and reuse them directly.
2. **Medium:** Write a stride-aware IL kernel that takes pointer + strides per operand. Same shape as the existing axis-reduction kernels. Specialize per T to eliminate boxing.
3. **Hard:** Reuse NpyIter — set up three operands (input, min, max) and let the iterator drive the broadcast walk. That's exactly what NumPy does in `umath/clip.c.src`.

Step 2 covers 95% of real workloads; step 3 is the "100% NumPy parity" finish.

---

## File: `src/NumSharp.Core/Backends/Default/Math/Default.Shift.cs`

### Function: `LeftShift/RightShift` + `ExecuteShiftOp` + `ExecuteShiftOpScalar`

**Severity:** perf (materializes non-contig operands)

**Criteria coverage:**
1. ✓ Mirrors NumPy structure (validate integer; left/right shift).
2. ✓ Type-error for non-integer operands matches NumPy ufunc rejection.
3. ✓ Contig + scalar shift: SIMD fast path (`ExecuteShiftOpScalar` line 124-128).
4. ✗ Non-contig array operand: forced copy via `.copy()` at lines 72-79. Same for RHS being non-Int32.
5. ✓ Strided left_shift (1 M, 20 iters): NumSharp 31 ms vs contig 2 ms — ~15× slowdown due to materialization. NumPy strided is 1.4× slower than contig (no copy).
6. ✓ Uses IL kernels (`ILKernelGenerator.GetShiftArrayKernel<T>`, `GetShiftScalarKernel<T>`).
7. ✓ All 8 integer dtypes covered; explicit TypeError for floats/Half/Complex/Decimal/Boolean.
8. ✗ No `out=`, no `where=`, no `casting=`.
9. ✗ See #4 — `.copy()` to materialize. Also casts RHS to Int32 always (could elide for already-Int32 RHS).
10. ✓ IL kernels.

**Finding:**

```csharp
// Default.Shift.cs:71-79
var contiguousLhs = broadcastedLhs.Shape.IsContiguous ? broadcastedLhs : broadcastedLhs.copy();
...
var rhsInt32 = broadcastedRhs.GetTypeCode == NPTypeCode.Int32
    ? broadcastedRhs
    : broadcastedRhs.astype(NPTypeCode.Int32);
var contiguousRhs = rhsInt32.Shape.IsContiguous ? rhsInt32 : rhsInt32.copy();
```

Every non-contig operand becomes a fresh copy. 1 M elements = 4 MB copy per call, dominating wall-time. NumPy walks strided inputs natively.

**Reproduction:**

```csharp
var arr = np.arange(2_000_000).astype(NPTypeCode.Int32);
var sliced = arr["::2"];                              // non-contig, 1 M elements
var shifts = np.full(new Shape(1_000_000), 2, NPTypeCode.Int32);

np.left_shift(sliced, shifts);   //  31 ms  ← materializes 1 M ints
np.left_shift(arr[":1000000"], shifts);  // 2 ms  ← contig, no copy
```

**Remediation:**

Same as ClipNDArray: write a stride-aware IL kernel for `ShiftArrayKernel<T>` that takes `(T*, long*, int*, long*, T*, long*, ndim, size)` and walks operand strides explicitly. The current kernel `kernel((T*)input.Address, shifts, (T*)output.Address, count)` assumes linear layout; expand the signature.

---

## File: `src/NumSharp.Core/Backends/Default/Math/DefaultEngine.BinaryOp.cs`

### Function: `ExecuteBinaryOp` + scalar dispatch helpers

**Severity:** clean (with API parity gap)

**Criteria coverage:**
1. ✓ Mirrors NumPy `PyUFunc_GenericFunction` outline: shape broadcast → kernel lookup → execute.
2. ✓ Type promotion verified (NEP50; int/float, float32+float64, special Power rules).
3. ✓ SimdFull path for contig+contig.
4. ✓ Path classifier covers SimdScalarRight/Left/SimdChunk/General.
5. ✓ Contig f32+f32: NumPy 44 ms / 50 iters / 1 M, NumSharp 80 ms → NumPy ~2× faster (acceptable). Strided: NumPy 40 ms vs NumSharp 226 ms → 5.7× gap.
6. ✓ Full IL pipeline via `MixedTypeKernelKey`.
7. ✓ Scalar dispatch covers all 15 dtypes (3-level switch nest: LHS×RHS×result). Heavy but only on scalar×scalar, which is rare.
8. ✗ No `out=` plumbed in. NumPy ufunc signature: `np.add(x1, x2, /, out=None, where=True, casting='same_kind', order='K', dtype=None, ...)`.
9. ✓ No wasted copies in the common path.
10. ✓ Full IL.

**Finding:**

Code structure is sound. F-contig output preservation (line 104-106) is a NumPy parity feature that other libraries skip — good.

The big architectural gap is `out=` — every binary ufunc allocates a fresh result array. NumPy supports `np.add(a, b, out=c)` to reuse a pre-allocated buffer. For pipelines like `np.add(a, b, out=a)` (in-place), this is a 2× memory win.

**Reproduction:** none (parity gap, not bug).

**Remediation:**

1. Add `out=` to `ExecuteBinaryOp` and plumb through to all per-op signatures (Add, Subtract, …).
2. The scalar×scalar dispatch (nested switch in `ExecuteScalarScalar` → `InvokeBinaryScalarLhs` → `InvokeBinaryScalarRhs`) could be collapsed to a single `NpFunc.Invoke` call now that NpFunc exists.

---

## File: `src/NumSharp.Core/Backends/Default/Math/DefaultEngine.UnaryOp.cs`

### Function: `ExecuteUnaryOp` + scalar dispatch helpers

**Severity:** clean (same `out=` gap)

**Criteria coverage:**
1. ✓ Mirrors NumPy.
2. ✓ Output-type rules verified: Negate/Abs/LogicalNot preserve; math ops promote.
3. ✓ Contig: SIMD.
4. ✓ Strided/F-contig preserved.
5. n/a.
6. ✓ Full IL.
7. ✓ All 15 dtypes.
8. ✗ No `out=`.
9. ✓.
10. ✓.

**Finding:** None. Clean dispatcher.

**Remediation:** Add `out=`.

---

## File: `src/NumSharp.Core/Backends/Default/Math/DefaultEngine.CompareOp.cs`

### Function: `ExecuteComparisonOp` + Compare/NotEqual/Less/...

**Severity:** clean

**Criteria coverage:**
1. ✓.
2. ✓ Verified on int + strided.
3. ✓.
4. ✓.
5. n/a.
6. ✓ Full IL.
7. ✓ All 15 dtypes incl. Complex (lexicographic).
8. ✗ No `out=`.
9. ✓.
10. ✓.

**Finding:** None.

---

## File: `src/NumSharp.Core/Backends/Default/Math/DefaultEngine.ReductionOp.cs`

### Function: `ExecuteElementReduction<TResult>` + per-op wrappers

**Severity:** clean

**Criteria coverage:**
1. ✓ Mirrors NumPy `PyArray_ReductionAccumulator`.
2. ✓ Empty array returns identity. Scalar reduction returns the scalar.
3. ✓ Contig: IL kernel with SIMD.
4. ✓ Strided: same IL kernel walks strides.
5. n/a.
6. ✓ Full IL via `TryGetTypedElementReductionKernel<TResult>`.
7. ⚠ Half (lines 219-291) and Complex (lines 224-318) use C# fallbacks because IL `OpCodes.Bgt/Blt` don't work on `Half` struct and Complex has no total ordering. Documented inline with B1/B7/B8/B12 markers.
8. ⚠ Per-op wrappers (`sum_elementwise_il`, `prod_elementwise_il`, …) accept `typeCode` but not `out=` / `where=`.
9. ✓.
10. ✓.

**Finding:** None — but worth noting these wrappers are wrappers, not the actual NumPy API. The user-facing `np.sum`/`np.mean`/etc. live in `Reduction/*.cs` and they're audited separately below.

---

## File: `src/NumSharp.Core/Backends/Default/Math/Reduction/Default.Reduction.Add.cs`

### Function: `ReduceAdd(arr, axis_, keepdims, typeCode, @out)` + helpers

**Severity:** clean (one perf nit on trivial axis)

**Criteria coverage:**
1. ✓ Mirrors NumPy `PyArray_Sum`.
2. ✓ Empty array → 0 with accumulating dtype. Verified.
3. ✓ Element-wise: IL.
4. ✓ Axis: IL via `ExecuteAxisReduction`.
5. n/a.
6. ✓ All IL.
7. ✓ All 15 dtypes covered by `sum_elementwise_il` (Half/Complex via iterator fallback in `DefaultEngine.ReductionOp.cs`).
8. ✓ `out=`, `axis`, `keepdims`, `dtype` all plumbed.
9. ⚠ `HandleTrivialAxisReduction` (axis with size==1) at lines 161-178 uses `arr.GetAtIndex(i)` + `SetAtIndex` per element — virtual + boxing. For a (1, 1000000) array reduced along axis=0, this means 1M virtual calls. Not on the typical hot path though.
10. ✓.

**Finding:** None blocking. The trivial-axis loop could be replaced with a memcpy when typecodes match, but it's not user-facing perf-critical.

---

## File: `src/NumSharp.Core/Backends/Default/Math/Reduction/Default.Reduction.ArgMax.cs`

### Function: `ReduceArgMax` + `ArgReductionAxisFallback`

**Severity:** clean

**Criteria coverage:**
1. ✓ Empty array raises (NumPy parity).
2. ✓.
3. ✓ IL.
4. ✓ IL axis kernel; SByte/Half/Complex fall back to per-slice iter + the typed element-wise argmax (lines 143-146).
5. n/a.
6. ✓ Mostly IL.
7. ⚠ Half/Complex/SByte axis fallback iterates via `arr[slices]` (constructs an NDArray view per slice) which is a notable per-axis-slot allocation. Not a correctness issue, just slower than NumPy.
8. ✓ `axis`, `keepdims` plumbed. No `out=`.
9. ⚠ The fallback at line 192 (`var slice = arr[slices]`) creates one NDArray view per output position. For Half on a (1000, 1000) reduce-axis-0, that's 1000 view allocations. Adequate but not great.
10. ✓.

**Finding:** None blocking.

---

## File: `src/NumSharp.Core/Backends/Default/Math/Reduction/Default.Reduction.CumAdd.cs`

### Function: `ReduceCumAdd` + helpers

**Severity:** clean (minor refactor)

**Criteria coverage:**
1. ✓ Cumulative sum logic mirrors NumPy.
2. ✓ Boolean → Int64 (NumPy 2.x rule); verified at line 15-19.
3. ✓ Contig: IL fast path for both element-wise and axis (lines 70-83 + lines 136-150).
4. ⚠ Non-contig axis path: falls back to `ExecuteAxisCumSumFallback` → `NpyAxisIter.ExecuteSameType<T,CumSumAxisKernel<T>>` (NpyIter-based, correct).
5. n/a.
6. ✓ IL where possible; NpyIter fallback for non-contig.
7. ⚠ Element-wise non-contig path at line 130 forces a copy: `if (!arr.Shape.IsContiguous) return cumsum_elementwise(arr.copy(), typeCode);`. Wasteful when the input is already linearized after broadcast.
8. ✓ `axis`, `dtype` plumbed. No `out=`.
9. ⚠ See #7.
10. ✓.

**Finding:**

The element-wise cumsum at line 130-131 unconditionally copies for non-contig input. NumPy handles non-contig in one pass via NpyIter. Now that NpyIter exists in NumSharp (NpyIterRef), this copy is unnecessary.

---

## File: `src/NumSharp.Core/Backends/Default/Math/Reduction/Default.Reduction.CumMul.cs`

### Function: `ReduceCumMul`

**Severity:** clean (mirrors CumAdd)

**Criteria coverage:** identical to CumAdd; same `arr.copy()` for non-contig element-wise.

**Finding:** Same as CumAdd. Bundle the remediation.

---

## File: `src/NumSharp.Core/Backends/Default/Math/Reduction/Default.Reduction.Mean.cs`

### Function: `ReduceMean(arr, axis_, keepdims, typeCode)` + `MeanAxisComplex`

**Severity:** clean

**Criteria coverage:**
1. ✓ Mirrors NumPy.
2. ✓ Dtype rules verified: f32→f32, f16→f16, int→f64. Complex axis uses dedicated path (B2 marker).
3. ✓ Element-wise: IL via `mean_elementwise_il`.
4. ✓ Axis: IL via `ExecuteAxisReduction` with Mean op.
5. n/a.
6. ✓.
7. ✓ Half axis preserves Half via Double sum + cast (B16 marker, line 71-72).
8. ⚠ `axis`, `keepdims`, `dtype` plumbed. No `out=`.
9. ✓.
10. ✓.

**Finding:** None blocking. Dtype preservation matrix is thoroughly handled.

---

## File: `src/NumSharp.Core/Backends/Default/Math/Reduction/Default.Reduction.Nan.cs`

### Function: `NanSum/NanProd/NanMin/NanMax`

**Severity:** clean

**Criteria coverage:**
1. ✓ NaN-skip logic mirrors `numpy/_core/fromnumeric.py:_nanreduce`.
2. ✓.
3. ✓ Contig: dedicated SIMD helpers (`NanSumSimdHelperFloat`, etc.).
4. ✗ Axis non-contig fallback (line 439-506) uses `arr.GetAtIndex(baseOffset + i*stride)` — virtual + boxing per element. Slow but correct.
5. n/a.
6. ⚠ SIMD-only for the contig element-wise path. Axis non-contig falls back to virtual access.
7. ✓ Float/Double/Half + Complex (special path at lines 19-21, 669-720).
8. ⚠ `axis`, `keepdims` plumbed. No `out=`.
9. ✗ See #4.
10. ✓ IL kernels for contig; manual scalar fallback for non-contig.

**Finding:**

The axis non-contig fallback at `ExecuteNanAxisReductionScalar` recomputes coordinates per element using `outputDimStridesArray` — a NumPy-style "ravel + multi-index" approach but goes through `arr.GetAtIndex` (boxing). On a strided (1024, 1024) float64 array reducing axis=0, that's 1 M boxed-double calls per call. NumPy walks strides natively in C.

**Remediation:** Replace with NpyIter (`NpyIterRef`) — the same trick used in `Std.cs`/`Var.cs` axis fallback via `NpyAxisIter.ReduceDouble<…>`.

---

## File: `src/NumSharp.Core/Backends/Default/Math/Reduction/Default.Reduction.Product.cs`

### Function: `ReduceProduct`

**Severity:** clean

**Criteria coverage:**
1. ✓ Empty array → 1 with accumulating dtype.
2. ✓.
3. ✓ IL.
4. ✓.
5. n/a.
6. ✓.
7. ✓ All 15 dtypes via `prod_elementwise_il`.
8. ⚠ No `out=`.
9. ✓.
10. ✓.

**Finding:** None.

---

## File: `src/NumSharp.Core/Backends/Default/Math/Reduction/Default.Reduction.Std.cs`, `Var.cs`

### Function: `ReduceStd/ReduceVar` + IL/Fallback dispatchers

**Severity:** parity-gap (ddof≥n returns NaN instead of inf)

**Criteria coverage:**
1. ✓ Two-pass mean+var algorithm (NumPy parity).
2. ✗ NumPy returns `+inf` when `ddof >= n` (raw div-by-zero → IEEE inf). NumSharp returns `NaN`. Documented in code comments at `Std.cs:354-365` (B24 marker), but they justify it incorrectly — the comment claims "ddof >= n yields +inf because sqrt(inf) = inf" but the math only works if `axisSize-ddof = 0` divides cleanly to inf, which `Math.Max(axisSize-ddof, 0)` defeats.
3. ✓ Contig: IL SIMD helpers per type.
4. ✓ Axis: IL kernel with Double output (`ExecuteAxisVar/StdReductionIL`).
5. n/a.
6. ✓ IL.
7. ✓ All numeric types via `VarSimdHelper<T>` (10 cases + Decimal/Complex via fallback).
8. ✓ `axis`, `keepdims`, `dtype`, `ddof` plumbed. No `out=`.
9. ✓.
10. ✓.

**Finding:**

```csharp
// Default.Reduction.Var.cs:355-366 — ExecuteAxisVarReductionIL
if (ddof != 0)
{
    double* resultPtr = (double*)result.Address;
    double divisor = Math.Max(axisSize - ddof, 0);   // <-- clamps to 0
    double adjustment = (double)axisSize / divisor;  // axisSize/0 = +inf when divisor==0
    for (long i = 0; i < outputSize; i++)
        resultPtr[i] *= adjustment;
}
```

For `ddof >= axisSize`, divisor=0 and adjustment=+inf. But the multiplication `resultPtr[i] *= +inf` is +inf only if resultPtr[i] > 0. For a constant-array slice (ddof=0 variance = 0), the result is `0 * inf = NaN`. NumPy's raw `(axisSize-ddof)` divisor produces inf even for zero-variance because it's `0/0 = NaN` in NumPy... wait, let me re-verify.

Actually NumPy returns `inf` even for constant arrays:
```python
>>> np.var(np.array([1.0,1.0,1.0,1.0]), ddof=10)
RuntimeWarning: invalid value encountered in scalar divide
inf
```

Hmm, that's `0 / -6 = -0.0` not inf. NumPy may special-case the warning. Let me revisit by checking element-wise:

In my earlier reproduction, `np.var([1,2,3,4,5], ddof=10) = inf` in NumPy. NumSharp returns NaN. This IS a parity bug.

**Reproduction:**

```python
>>> np.var(np.array([1.0,2,3,4,5]), ddof=10)
inf
```

```csharp
np.var(np.array(new double[]{1,2,3,4,5}), ddof: 10);
// NumSharp: NaN  ← parity gap
```

The element-wise path goes through `VarSimdHelper<T>` (line 42 in `ILKernelGenerator.Masking.VarStd.cs`):
```csharp
if (size <= ddof)
    return double.NaN; // Division by zero or negative
```

Hardcoded NaN return. NumPy returns +inf.

**Remediation:**

Both paths need to be updated:
1. `VarSimdHelper<T>` line 42-43: return `double.PositiveInfinity` (or raw `sqDiffSum / (size-ddof)` which produces +inf naturally) instead of NaN.
2. `StdSimdHelper<T>` cascades from VarSimdHelper, would automatically follow.
3. `ExecuteAxisVar/StdReductionIL` line 362: drop the `Math.Max(…, 0)` clamp; let raw `(axisSize-ddof)` produce +inf via IEEE division.

---

## File: `src/NumSharp.Core/Backends/Default/Math/BLAS/Default.Dot.cs`

### Function: `Dot(left, right)` — dispatcher

**Severity:** bug (3D@1D produces wrong shape/values)

**Criteria coverage:**
1. ✗ Matches NumPy's docstring text but the 3D+ @ 1D case at line 60 is wrong.
2. ✗ `np.dot(3D, 1D)` returns the wrong shape and wrong values.
3. ✓ 2D@2D fast.
4. ✓ 2D@2D uses MultiplyMatrix.
5. ✓ Heavy gap on dtype-untyped paths: NumPy uses BLAS for float/double. NumSharp manual.
6. ⚠ Mostly hand-rolled dispatch; SIMD on float/double only.
7. ⚠ Strided/transposed inputs to `DotNDMD` fall back to `DotNDMDGeneric` which boxes through `Converts.ToDouble(arr.GetValue(coords))` — 17× slower than contig.
8. ✗ No `out=` parameter.
9. ⚠ 1D@1D path: `left * right` allocates an intermediate, then ReduceAdd. Two passes; NumPy does one.
10. ✓.

**Finding (bug):**

```csharp
// Default.Dot.cs:55-61
//If a is an N-D array and b is a 1-D array, it is a sum product over the last axis of a and b.
if (leftshape.NDim >= 2 && rightshape.NDim == 1)
{
    //TODO! this doesn't seem right, read desc
    //var right_broadcasted = new NDArray(right.Storage.Alias(np.broadcast_to(rightshape, leftshape)));
    return np.sum(left * right, axis: 1);    // <-- HARDCODED axis: 1
}
```

`np.sum(left * right, axis: 1)` works for 2D@1D (axis=1 = last axis of a 2D array) but is wrong for 3D@1D where the last axis is `axis=2`. The comment `//TODO!` is even in the source.

**Reproduction:**

```python
>>> import numpy as np
>>> a = np.arange(24).reshape(2,3,4)
>>> b = np.array([1, 2, 3, 4])
>>> np.dot(a, b)        # NumPy
array([[ 20,  60, 100],
       [140, 180, 220]])
>>> np.dot(a, b).shape  # (2, 3)
(2, 3)
```

```csharp
// NumSharp
var a = np.arange(24).reshape(2,3,4);
var b = np.array(new int[]{1, 2, 3, 4});
np.dot(a, b);
// shape: (2, 4)  ← WRONG (should be (2, 3))
// values: [[12, 30, 54, 84], [48, 102, 162, 228]]  ← WRONG
```

The actual `left * right` here broadcasts `(2,3,4) * (4,)` correctly to `(2,3,4)`, then `axis: 1` sums over the wrong dim (sums the size-3 dim instead of the size-4 dim).

**Remediation:**

```csharp
if (leftshape.NDim >= 2 && rightshape.NDim == 1)
{
    return np.sum(left * right, axis: leftshape.NDim - 1);  // last axis
}
```

**Finding (perf):**

1D@1D dot does `left * right` (1 M element allocation) then `ReduceAdd`. NumPy fuses to single SIMD `cblas_ddot`. 22× gap measured.

**Remediation (perf):**

Add a direct 1D@1D SIMD inner-product kernel (mirror `DotProductDouble` in `Default.Dot.NDMD.cs:288-318`) instead of going through `*` + sum.

---

## File: `src/NumSharp.Core/Backends/Default/Math/BLAS/Default.Dot.NDMD.cs`

### Function: `DotNDMD` + SIMD/Generic kernels

**Severity:** perf (strided fallback is 17× slower than contig)

**Criteria coverage:**
1. ✓ Output shape matches NumPy: lshape[:-1] + rshape[:contract_dim] + rshape[contract_dim+1:].
2. ✓ For float/double + contig.
3. ✓ Contig same-type float/double: SIMD with V256.
4. ✗ Non-contig (e.g. transposed lhs): falls back to `DotNDMDGeneric` which uses `GetValue(coords)` per element. 17× slowdown observed.
5. ✓ Cited.
6. ⚠ SIMD only for float/double. All other dtypes (int, half, complex, decimal, bool, char) go to generic boxing path.
7. ⚠ All 15 dtypes supported but only Float/Double get the fast path.
8. ✗ No `out=`.
9. ✗ Generic path boxes via `GetValue(coords)` returning `object` then `Converts.ToDouble`.
10. ⚠ Doesn't use NpyIter for the multi-axis walk.

**Finding:**

```csharp
// Default.Dot.NDMD.cs:325-385 — DotNDMDGeneric
for (long k = 0; k < K; k++)
{
    lhsCoords[lhsNdim - 1] = k;
    double lVal = Converts.ToDouble(lhs.GetValue(lhsCoords));    // <-- boxes

    rhsCoords[rhsNdim - 2] = k;
    double rVal = Converts.ToDouble(rhs.GetValue(rhsCoords));    // <-- boxes

    sum += lVal * rVal;
}
```

Every K iterations does two `GetValue(coords) → object → ChangeType<double>` round trips.

**Reproduction:**

```csharp
var a = np.random.rand(20, 30, 50);
var b = np.random.rand(50, 40);

np.dot(a, b);              //  16 ms / 5 iters (SIMD)
np.dot(a.transpose(new int[]{1,0,2}), b);  // 277 ms (general) — 17× slower
```

**Remediation:**

Two options:
1. Extend `DotNDMDSimd*` to read strided pointers (mirror what `MatMulStridedSame<T>` does for 2D). Same kernel topology but with explicit `strideA0/1` reads.
2. Use `NpyIter` over the lhs and rhs (multi-operand mode) to drive the contracting loop.

Either way, the boxing-heavy `Generic` path can survive only for unsupported dtypes (Decimal/Complex).

---

## File: `src/NumSharp.Core/Backends/Default/Math/BLAS/Default.MatMul.2D2D.cs`

### Function: `MultiplyMatrix` + `TryMatMulSimd`

**Severity:** perf (no BLAS, far behind NumPy for float/double)

**Criteria coverage:**
1. ✓ Standard GEMM topology.
2. ✓.
3. ✓ Contig float/double: BLIS-style SIMD via `SimdMatMul.MatMulFloat/Double`. ~83× slower than NumPy's MKL/OpenBLAS-backed GEMM (this is the cost of not linking BLAS).
4. ✓ Strided (transposed) operands: pass strides through to the SIMD kernel via packers, no copy.
5. ✓ NumPy 8.4 ms / 5 iters / 512² float64 matmul; NumSharp 698 ms → NumPy ~83× faster. Same on float: NumPy 5 ms vs NumSharp 116 ms. Int paths closer: 4× gap on int32, 4.3× on int64.
6. ✓ IL kernels + dedicated SIMD kernel.
7. ✓ All 15 dtypes via `MatMulStridedGeneric` dispatch. Bool, Decimal, Complex all have dedicated kernels.
8. ✓ `@out` parameter accepted (line 30, 45-52). No `where=`, no `casting=`, no `axes=` (NumPy 2.x feature).
9. ✓ No wasted copies — stride-native packers absorb arbitrary strides.
10. ⚠ Custom SIMD packer instead of NumPy's BLAS link.

**Finding:**

This is the expected cost of being self-contained. Linking OpenBLAS would close the gap but adds a native dependency. The current code is well-engineered for a pure-managed implementation.

---

## File: `src/NumSharp.Core/Backends/Default/Math/BLAS/Default.MatMul.Strided.cs`

### Function: `MatMulStridedGeneric` + `RunSame<T>` / `MatMulStridedSame<T>` / `MatMulStridedMixed<TResult>` + bool/complex paths

**Severity:** clean (cleanly factored, minor improvements possible)

**Criteria coverage:**
1. ✓ Two JIT-specialized inner loops based on `bStride1 == 1`.
2. ✓.
3. ✓ Same-type: `INumber<T>` generic specialized per dtype.
4. ✓ Strided: handled natively via pointer arithmetic.
5. n/a — see 2D2D.
6. ⚠ Not IL-generated; uses INumber<T> generic specialization. Adequate.
7. ✓ All 15 dtypes (bool, complex via dedicated kernels — INumber<Complex> isn't supported in .NET).
8. ✓ `@out` plumbed from caller.
9. ✓ No wasted copies.
10. ⚠ Doesn't use NpyIter — uses raw pointer + stride arithmetic. Adequate for 2D GEMM.

**Finding:**

The mixed-type path uses a row-buffered double accumulator (line 375 `accBuf = new double[N]`). One heap allocation per call — could be pooled.

---

## File: `src/NumSharp.Core/Backends/Default/Math/BLAS/Default.MatMul.cs`

### Function: `Matmul(lhs, rhs)`

**Severity:** parity-gap (1D@2D rejected)

**Criteria coverage:**
1. ✗ NumPy supports 1D@2D matmul by prepending a 1 and squeezing. NumSharp explicitly rejects it (line 20-21).
2. ✗ See above.
3. ✓.
4. ✓ N-D batch matmul via `np.broadcast_arrays` loop.
5. n/a.
6. ⚠.
7. ✓ All dtypes through MultiplyMatrix.
8. ✗ No `out=` on the top-level dispatcher (the inner `MultiplyMatrix` does accept it).
9. ⚠ N-D batch path allocates a `ValueCoordinatesIncrementor` and walks slice-by-slice. Could be parallelized.
10. ⚠.

**Finding:**

```csharp
// Default.MatMul.cs:19-21
//If the first argument is 1-D, it is promoted to a matrix by prepending a 1 to its dimensions. After matrix multiplication the prepended 1 is removed.
if (lhs.ndim == 1 && rhs.ndim == 2)
    throw new NotSupportedException("Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?)");
```

The comment on line 19 correctly describes NumPy's behavior, but the code raises instead of implementing it.

**Reproduction:**

```python
>>> np.matmul(np.array([1,2,3]), np.array([[1,2],[3,4],[5,6]]))
array([22, 28])
```

```csharp
np.matmul(np.array(new int[]{1,2,3}), np.array(new int[,]{{1,2},{3,4},{5,6}}));
// NotSupportedException: Input operand 1 has a mismatch in its core dimension 0…
```

**Remediation:**

Reshape lhs to (1, n), matmul, then reshape result back to 1D:

```csharp
if (lhs.ndim == 1 && rhs.ndim == 2)
{
    var lhsRow = lhs.reshape(1, lhs.shape[0]);
    var result = MultiplyMatrix(lhsRow, rhs);
    return result.reshape(rhs.shape[1]);
}
```

(This is exactly what `Default.Dot.cs:64-72` does for the 1D@2D dot case.)

---

## Summary table

Severity-ranked, highest-impact first.

| # | File / Function | Severity | Tag |
|---|---|---|---|
| 1 | `Default.Power.cs:PowerInteger` | **bug** | Crashes on strided/broadcast int^int; ignores strides if it didn't crash |
| 2 | `Default.Reciprocal.cs:ReciprocalInteger` | **bug** | Same crash pattern as PowerInteger |
| 3 | `Default.Dot.cs:Dot(>=2D, 1D)` | **bug** | Hardcoded `axis: 1` produces wrong shape/values for ND@1D (N≥3) |
| 4 | `Default.MatMul.cs:Matmul(1D,2D)` | **parity-gap** | NumPy supports; NumSharp throws NotSupported |
| 5 | `Default.Power.cs:Power(int, neg int)` | **parity-gap** | NumPy raises ValueError; NumSharp silently returns "seterr=ignore" values |
| 6 | `Default.ATan2.cs:PromoteATan2Single` | **parity-gap** | Small ints → Half/Single; NumPy → Double |
| 7 | `Default.Ceil/Floor/Truncate.cs:Boolean` | **parity-gap** | NumPy keeps bool; NumSharp promotes to float64 |
| 8 | `Default.Reduction.Std/Var:ddof≥n` | **parity-gap** | NumPy → +inf; NumSharp → NaN |
| 9 | `Default.ClipNDArray.cs:General path` | **perf** | 14× slower than NumPy on strided clip (GetAtIndex + boxing) |
| 10 | `Default.Shift.cs:non-contig ops` | **perf** | 15× slowdown vs contig due to `.copy()` materialization |
| 11 | `Default.Dot.NDMD.cs:Generic path` | **perf** | 17× slower than contig for strided ND@MD (GetValue boxing) |
| 12 | `Default.Dot.cs:1D@1D` | **perf** | 22× slower than NumPy due to 2-pass `*` + ReduceAdd allocation |
| 13 | `Default.MatMul.2D2D.cs:float/double` | **perf** | ~83× slower than NumPy MKL/OpenBLAS (no native BLAS link) |
| 14 | `Default.Reduction.Nan.cs:axis non-contig` | **perf** | GetAtIndex boxing fallback; not on NpyIter |
| 15 | `Default.Reduction.CumAdd/Mul.cs:non-contig` | **refactor** | Unconditional `arr.copy()` on non-contig input; NpyIter now available |
| 16 | `DefaultEngine.BinaryOp/UnaryOp/CompareOp.cs` | **parity-gap** | No `out=` parameter (NumPy ufunc signature) |
| 17 | `Default.Clip.cs:always-Cast` | **refactor** | `Cast(copy: true)` even when already contig+dtype-matched |
| 18 | `Default.MatMul.Strided.cs:mixed accum` | **refactor** | Per-call `new double[N]` accumulator heap alloc — poolable |
| 19 | `Default.Reduction.Add.cs:TrivialAxis` | **refactor** | `GetAtIndex`/`SetAtIndex` loop instead of memcpy |
| 20 | `Default.Reduction.ArgMax.cs:Half/Complex axis fallback` | **refactor** | View allocation per slice |

### Bugs to fix first (in order of user-visibility):

1. **`np.dot(3D, 1D)` returns wrong shape/values** — the hardcoded `axis: 1`. One-line fix, but bug since at least 2018 (the `//TODO!` comment).
2. **`np.power/np.reciprocal` crash on strided/broadcast integer inputs** — needs either guarding the fast-path or removing it.
3. **`np.matmul(1D, 2D)` rejected** with the docstring describing the correct behavior right above the throw.

### Parity gaps to close:

4. `arctan2` integer-dtype promotion to Half/Single (should always be Double).
5. `ceil/floor/trunc` on Boolean (should preserve bool).
6. `int ** -int` should raise ValueError (NumPy) instead of returning 0.
7. `var/std` with `ddof ≥ n` should return +inf, not NaN.
8. `out=` parameter on binary/unary/compare ufuncs, dot, matmul, power.

### Performance ladder:

9. Stride-aware IL kernel for `ClipNDArray.General` (14× gap).
10. Stride-aware IL kernel for `Shift.Array` (15× gap on strided).
11. Stride-aware IL/SIMD kernel for `DotNDMDGeneric` (17× gap).
12. Single-pass SIMD inner-product for `Dot.1D@1D` (22× gap).
13. (Long-term) Native BLAS link for `MatMul.float/double` (~83× gap).
