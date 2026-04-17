# Leftover IConvertible / System.Convert Usages

**Date:** 2026-04-17
**Branch:** `worktree-half`
**Context:** Round 4 fixed all leftover `IConvertible` / `Convert.ChangeType` usage **within** the
`Converts.cs` and `Converts.Native.cs` files. This document audits the **rest of the codebase**
for the same patterns.

## Why This Matters

NumSharp supports 15 dtypes including **`Half`** (`System.Half`) and **`Complex`** (`System.Numerics.Complex`).
Neither implements `System.IConvertible`. Therefore any code path that:

1. Casts a value to `IConvertible` and calls `.ToXxx(provider)`, OR
2. Calls `System.Convert.ToXxx(value)` (which internally uses `IConvertible`),

…will throw `InvalidCastException` when the value is `Half` or `Complex`.

Additionally, `char` does not implement `IConvertible.ToBoolean(provider)` (BCL design — throws
`InvalidCastException: Invalid cast from 'Char' to 'Boolean'`), so `((IConvertible)'A').ToBoolean(null)`
throws even though `char` does implement `IConvertible`.

The NumSharp solution is to route all such conversions through `Converts.ToXxx(...)` (object dispatcher)
which handles all 15 dtypes with NumPy-parity semantics (truncation, wrapping, NaN handling).

---

## High Priority — User-facing NumPy operations break for Half/Complex

### H1. `ArraySlice.cs:408-426` — `Allocate(NPTypeCode, count, fill)`

**Sites:** ~13 lines in one method.

```csharp
public static IArraySlice Allocate(NPTypeCode typeCode, long count, object fill)
{
    switch (typeCode)
    {
        case NPTypeCode.Boolean: return new ArraySlice<bool>(new UnmanagedMemoryBlock<bool>(count, ((IConvertible)fill).ToBoolean(CultureInfo.InvariantCulture)));
        case NPTypeCode.SByte:   return new ArraySlice<sbyte>(new UnmanagedMemoryBlock<sbyte>(count, ((IConvertible)fill).ToSByte(CultureInfo.InvariantCulture)));
        // ... 10 more types ...
        case NPTypeCode.Half:    return new ArraySlice<Half>(new UnmanagedMemoryBlock<Half>(count, fill is Half h ? h : (Half)Convert.ToDouble(fill)));
        // ... Decimal ...
        case NPTypeCode.Complex: return new ArraySlice<Complex>(new UnmanagedMemoryBlock<Complex>(count, fill is Complex c ? c : new Complex(Convert.ToDouble(fill), 0)));
    }
}
```

**Why broken:**
- `((IConvertible)fill).ToInt32(...)` throws when `fill` is `Half` or `Complex`.
- The Half target line 418 has `(Half)Convert.ToDouble(fill)` — also throws when `fill` is `Complex`.
- Line 422 (Complex target) uses `Convert.ToDouble(fill)` — throws when `fill` is `Half`.

**User impact:** `np.full(shape, Half.One, dtype=Int32)` and similar throw. This is a primary
array-creation path for fill operations.

**Proposed fix:** Replace each `((IConvertible)fill).ToXxx(InvariantCulture)` with
`Converts.ToXxx(fill)`. For Half/Complex targets, replace `Convert.ToDouble(fill)` with
`Converts.ToDouble(fill)` (object dispatcher).

```csharp
case NPTypeCode.Boolean: return new ArraySlice<bool>(new UnmanagedMemoryBlock<bool>(count, Converts.ToBoolean(fill)));
case NPTypeCode.SByte:   return new ArraySlice<sbyte>(new UnmanagedMemoryBlock<sbyte>(count, Converts.ToSByte(fill)));
// ... etc ...
case NPTypeCode.Half:    return new ArraySlice<Half>(new UnmanagedMemoryBlock<Half>(count, Converts.ToHalf(fill)));
case NPTypeCode.Complex: return new ArraySlice<Complex>(new UnmanagedMemoryBlock<Complex>(count, Converts.ToComplex(fill)));
```

### H2. `ArraySlice.cs:483-501` — `Allocate(Type, count, fill)`

**Sites:** Identical pattern to H1, ~13 lines in one method.

This is the `Type`-based overload of `Allocate`, used when the caller has a `System.Type`
instead of an `NPTypeCode`. Same fix as H1.

### H3. `np.searchsorted.cs:50-85` — type-agnostic value extraction

**Sites:** 3 lines.

```csharp
// Line 51:
double target = Convert.ToDouble(v.Storage.GetValue(new long[0]));
// Line 61:
double target = Convert.ToDouble(v.Storage.GetValue(i));
// Line 85:
double val = Convert.ToDouble(arr.Storage.GetValue(m));
```

**Why broken:** `arr.Storage.GetValue(...)` returns `object` boxing the element. If the array
is `Half` or `Complex` dtype, `Convert.ToDouble(boxed Half)` throws.

**User impact:** `np.searchsorted(np.array([Half.One, ...]), value)` throws. NumPy supports
searchsorted on float16 and complex arrays.

**Proposed fix:** Replace with `Converts.ToDouble(...)` which handles Half/Complex via the
object dispatcher.

```csharp
double target = Converts.ToDouble(v.Storage.GetValue(new long[0]));
```

Note: For `Complex`, `Converts.ToDouble(Complex)` discards the imaginary part (NumPy semantics).
Acceptable for searchsorted since complex comparison isn't well-defined; NumPy itself emits
ComplexWarning when sorting complex arrays.

### H4. `Default.MatMul.2D2D.cs:323,329` — scalar fallback for matmul

**Sites:** 2 lines.

```csharp
double aik = Convert.ToDouble(left.GetValue(leftCoords));
// ...
double bkj = Convert.ToDouble(right.GetValue(rightCoords));
```

**Why broken:** Scalar fallback path for matmul on non-SIMD-friendly arrays. Half/Complex
matrices throw before any computation begins.

**User impact:** `np.matmul(halfMatrix, halfMatrix)` throws when forced into scalar fallback
path (e.g., with strided/broadcast inputs).

**Proposed fix:** `Converts.ToDouble(left.GetValue(leftCoords))`.

### H5. `Default.Dot.NDMD.cs:371,375` — scalar fallback for dot product

**Sites:** 2 lines. Identical pattern to H4.

```csharp
double lVal = Convert.ToDouble(lhs.GetValue(lhsCoords));
// ...
double rVal = Convert.ToDouble(rhs.GetValue(rhsCoords));
```

**Proposed fix:** `Converts.ToDouble(lhs.GetValue(lhsCoords))`.

### H6. `NdArray.Convolve.cs:154,155` — convolve scalar path

**Sites:** 2 lines.

```csharp
double aVal = Convert.ToDouble(aPtr[j]);
double vVal = Convert.ToDouble(vPtr[k - j]);
```

**Why broken:** `aPtr` and `vPtr` are typed pointers (e.g., `Half*`). The deref `aPtr[j]` is `Half`,
boxes implicitly when passed to `Convert.ToDouble(object)` — throws.

**User impact:** `np.convolve(halfArray, halfArray)` throws.

**Proposed fix:** `Converts.ToDouble((object)aPtr[j])`. Or, since the caller knows the type at
the templated/generic level, prefer a typed cast: `(double)(Half)aPtr[j]` if the surrounding
generic context allows (need to check).

### H7. `ILKernelGenerator.Scan.cs` (~10 sites) — CumSum/CumProd scalar accumulator

**Sites:**

| Line | Code | Context |
|---|---|---|
| 1128 | `product *= Convert.ToInt64(src[inputOffset + i * axisStride])` | AxisCumProd, TOut=long |
| 1138 | `product *= Convert.ToDouble(src[inputOffset + i * axisStride])` | AxisCumProd, TOut=double |
| 1148 | `product *= Convert.ToDecimal(src[inputOffset + i * axisStride])` | AxisCumProd, TOut=decimal |
| 1947 | `sum += Convert.ToInt64(src[inputOffset + i * axisStride])` | AxisCumSum, TOut=long |
| 1957 | `sum += Convert.ToDouble(src[inputOffset + i * axisStride])` | AxisCumSum, TOut=double |
| 1967 | `sum += Convert.ToSingle(src[inputOffset + i * axisStride])` | AxisCumSum, TOut=float |
| 1977 | `sum += Convert.ToUInt64(src[inputOffset + i * axisStride])` | AxisCumSum, TOut=ulong |
| 1987 | `sum += Convert.ToDecimal(src[inputOffset + i * axisStride])` | AxisCumSum, TOut=decimal |
| 2392 | `sum += Convert.ToDouble(src[i])` | ElementwiseCumSum, TOut=double |
| 2402 | `sum += Convert.ToInt64(src[i])` | ElementwiseCumSum, TOut=long |
| 2412 | `sum += Convert.ToDecimal(src[i])` | ElementwiseCumSum, TOut=decimal |
| 2422 | `sum += Convert.ToSingle(src[i])` | ElementwiseCumSum, TOut=float |
| 2432 | `sum += Convert.ToUInt64(src[i])` | ElementwiseCumSum, TOut=ulong |

**Why broken:** `src` is `TIn*` (e.g., `Half*` or `Complex*`). `src[i]` is `TIn`. Boxing into
`Convert.ToXxx(object)` throws for Half/Complex. Note: Complex source for cumsum/cumprod is
actually meaningful in NumPy — `np.cumsum(complexArray)` works and returns Complex.

**User impact:** `np.cumsum(halfArray)` → `np.cumsum(complexArray)` → both throw on the scalar
fallback path. SIMD path may handle some types but Half/Complex always fall through to scalar.

**Proposed fix:** Two options:

1. **Direct cast (preferred when generic constraints allow):** Since `TIn` is known via reflection
   in `ILKernelGenerator`, emit a typed conversion. But these are not IL-emitted methods — they're
   the C# fallback used when IL kernels can't handle the dtype. So can't use IL emit here.

2. **Route through Converts dispatcher:**
   ```csharp
   product *= Converts.ToInt64((object)src[inputOffset + i * axisStride]);
   ```
   The `(object)` boxing is necessary since the source type is generic `TIn`. Boxing is unavoidable
   when calling the object dispatcher; performance of the scalar fallback is already non-critical
   (IL kernels handle the fast path).

   For `Complex` source where `TOut == long/decimal/float/double`, `Converts.ToXxx(Complex)` discards
   imaginary (NumPy parity). For TOut == Complex, the existing path in ILKernelGenerator should not
   reach these scalar branches.

### H8. `DefaultEngine.ReductionOp.cs:310` — mean for scalar arrays

**Sites:** 1 line.

```csharp
return typeCode.HasValue ? Converts.ChangeType(val, typeCode.Value) : Convert.ToDouble(val);
```

**Why broken:** When `typeCode` is null, falls back to `Convert.ToDouble(val)`. If `val` is Half/Complex
(unboxed), throws. The Complex case is special-handled at line 308-309 (returns val as-is), so by
line 310 the source type is known to NOT be Complex. But Half is still broken.

**User impact:** `np.mean(scalarHalfArray)` with default `typeCode=null` throws.

**Proposed fix:** `Converts.ToDouble(val)`.

---

## Medium Priority — Edge cases (rare in practice)

### M1. `np.repeat.cs:75,172` — repeats array dtype

**Sites:** 2 lines.

```csharp
// Line 75 and 172:
long count = Convert.ToInt64(repeatsFlat.GetAtIndex(i));
```

**Why broken:** `repeatsFlat.GetAtIndex(i)` returns boxed object. If user passes a Half/Complex
array as `repeats`, throws.

**User impact:** Edge case. NumPy expects `repeats` to be integer array. NumSharp doesn't enforce
this either, so a Half repeats array would fail with cryptic IConvertible error instead of a clean
type error.

**Proposed fix:** `Converts.ToInt64(repeatsFlat.GetAtIndex(i))`. This will truncate Half → long
gracefully (or discard Complex's imaginary). NumPy parity question: should we allow this or
throw a clean error? Recommend: allow it for permissiveness, matches NumPy's casting behavior.

### M2. `Default.Shift.cs:136` — bitwise shift amount

**Sites:** 1 line.

```csharp
int shiftAmount = Convert.ToInt32(rhs);
```

**Why broken:** `rhs` is `object` (the scalar shift amount). If user passes `(Half)5` as shift
amount, throws.

**User impact:** Very rare. Shift amounts are typically int literals. NumPy permits any integer-
convertible value.

**Proposed fix:** `Converts.ToInt32(rhs)`.

### M3. `NDArray.Indexing.Selection.Setter.cs:126,188` — fancy index parsing

**Sites:** 2 lines.

```csharp
// Line 126:
case IConvertible o: return Slice.Index(o.ToInt64(CultureInfo.InvariantCulture));
// Line 188:
case IConvertible o:
    indices.Add(NDArray.Scalar<int>(o.ToInt32(CultureInfo.InvariantCulture)));
```

**Why broken:** When user passes Half/Complex as an index, the `case IConvertible o` doesn't
match (Half/Complex don't implement IConvertible) and falls through to the default branch
("Unsupported slice type").

**User impact:** Currently throws clean "Unsupported slice type" error. Less broken than other
sites, but inconsistent with NumPy where `arr[Half(3)]` would work.

**Proposed fix:** Add explicit `case Half h:` and `case Complex c:` branches, or restructure
to a single branch using `Converts.ToInt64(o)` for any object.

### M4. `NDArray.Indexing.Selection.Getter.cs:109,172` — fancy index parsing (read path)

**Sites:** 2 lines. Identical pattern to M3.

```csharp
// Line 109:
case IConvertible o: return Slice.Index(o.ToInt64(CultureInfo.InvariantCulture));
// Line 172:
case IConvertible o:
    indices.Add(NDArray.Scalar<int>(o.ToInt32(CultureInfo.InvariantCulture)));
```

Same fix as M3.

---

## No Fix Needed

### NF1. `Converts.Native.cs:108,2685-2789` — DateTime conversions (~14 sites)

Examples:
```csharp
// Line 108 (in ChangeType(Object, TypeCode, IFormatProvider), preserved by Round 4):
return ((IConvertible)value).ToDateTime(provider);
// Lines 2714, 2720, ..., 2789: ToDateTime(byte/sbyte/short/...) overloads:
return ((IConvertible)value).ToDateTime(null);
```

**Why no fix:** `DateTime` is not a NumPy dtype. NumPy's `datetime64` is a separate dtype with
nanosecond/second-from-epoch semantics, not equivalent to .NET `DateTime`. These methods exist for
.NET interop completeness, not NumPy parity. They throw for Half/Complex sources, but that's an
expected outcome since the conversion has no defined meaning anyway.

### NF2. `Converts.cs:258-551` — `_NumPy` helper `_` defaults

Examples:
```csharp
// Line 258 (ToBoolean_NumPy default):
_ => Converts.ToBoolean(((IConvertible)value).ToDouble(null))
// Line 510 (ToHalf_NumPy default):
_ => (Half)((IConvertible)value).ToDouble(null)
// Line 531 (ToComplex_NumPy default):
_ => new Complex(((IConvertible)value).ToDouble(null), 0)
```

**Why no fix:** Each `_NumPy` helper is a `switch` expression where `Half`, `Complex`, `char`, and
all 12 classic types are explicitly handled BEFORE the `_` default. The default branch only fires
for exotic source types (string, bool subclasses, etc.) which all implement `IConvertible`. Half
and Complex never reach the default.

### NF3. `Converts.Native.cs:144-2433` — object dispatcher `_` defaults

Examples:
```csharp
// Line 144 (ToBoolean(object) default):
_ => ((IConvertible)value).ToBoolean(null)
// Line 2433 (ToHalf(object) default):
_ => (Half)((IConvertible)value).ToDouble(null)
// Line 2574 (ToComplex(object) default):
_ => new Complex(((IConvertible)value).ToDouble(null), 0)
```

Same reason as NF2: Half/Complex/char explicitly handled before the default branch.

### NF4. `Converts.Native.cs:271,455,644,825,1005,1194,1367,1552,1723,1930,2083,2235,2403` — `ToXxx(DateTime)` overloads

```csharp
// Example (line 271):
public static bool ToBoolean(DateTime value)
{
    return ((IConvertible)value).ToBoolean(null);
}
```

**Why no fix:** Source type is `DateTime` which DOES implement `IConvertible`. These calls don't
throw. They exist for .NET interop completeness. Whether the result is meaningful (e.g., DateTime
→ bool) is .NET-defined, not NumPy.

### NF5. `ILKernelGenerator.Reduction.NaN.cs:926,930` — IL constant emission

```csharp
il.Emit(OpCodes.Ldc_R4, Convert.ToSingle(value));
il.Emit(OpCodes.Ldc_R8, Convert.ToDouble(value));
```

**Why no fix:** `value` here is a runtime constant (reduction identity element like 0 or 1) used
for IL `Ldc_R4`/`Ldc_R8` opcodes. The constants are always primitive numerics (int, long, float,
double, decimal). Half/Complex constants would not flow through this path because Half/Complex
don't have SIMD reduction kernels that need IL constant emission.

### NF6. `ILKernelGenerator.Masking.VarStd.cs:352,359` — Decimal-only fallback

```csharp
// In the "for integer types" branch (per inline comment):
doubleSum += Convert.ToDouble(src[i]);
double diff = Convert.ToDouble(src[i]) - mean;
```

**Why no fix:** Per the inline comment "For integer types", `src` is sbyte/byte/int16/uint16/int32/
uint32/int64/uint64 — all of which implement `IConvertible`. Half/Complex paths are handled in the
preceding float branch.

### NF7. `Converts.cs:76` — CreateIntegerConverter absolute fallback

```csharp
result = fromDouble(Convert.ToDouble(@in));
```

**Why no fix:** This is the third-tier fallback after explicit checks for `Half`, `Complex`, and
`IConvertible`. Only exotic non-IConvertible non-Half non-Complex types reach here. There are no
such NumSharp dtypes. The fallback exists for defensive correctness with custom user types.

### NF8. `Converts.cs:1173,1181` — Disabled REGEN block

```csharp
return @in => (TOut)Convert.ChangeType(@in, tout);
```

**Why no fix:** Inside `#if _REGEN` block which is not active (the `_REGEN` symbol is not defined
in any active build configuration). The active code path is the explicit switch generated for
each type pair, which handles all 15×15 combinations or falls back through `CreateFallbackConverter`.

### NF9. `ILKernelGenerator.cs:445` — Comment only

```csharp
// Half conversion methods (Half is a struct with operator methods, not IConvertible)
```

**Why no fix:** Comment, not code. Documents intent.

### NF10. `src/dotnet/.../System.Runtime.cs` — Reference assembly

Not NumSharp code; it's a copy of the .NET runtime's reference assembly stub.

---

## Summary Table

| Priority | File | Sites | Status |
|---|---|---|---|
| H1 | `ArraySlice.cs` (`Allocate(NPTypeCode, …, fill)`) | 13 | TODO |
| H2 | `ArraySlice.cs` (`Allocate(Type, …, fill)`) | 13 | TODO |
| H3 | `np.searchsorted.cs` | 3 | TODO |
| H4 | `Default.MatMul.2D2D.cs` | 2 | TODO |
| H5 | `Default.Dot.NDMD.cs` | 2 | TODO |
| H6 | `NdArray.Convolve.cs` | 2 | TODO |
| H7 | `ILKernelGenerator.Scan.cs` | 13 | TODO |
| H8 | `DefaultEngine.ReductionOp.cs` | 1 | TODO |
| M1 | `np.repeat.cs` | 2 | TODO |
| M2 | `Default.Shift.cs` | 1 | TODO |
| M3 | `NDArray.Indexing.Selection.Setter.cs` | 2 | TODO |
| M4 | `NDArray.Indexing.Selection.Getter.cs` | 2 | TODO |
| **Total fixable sites** | | **56** | |
| NF1-NF10 | (no fix needed) | ~50 | N/A |

---

## Proposed Round 5 Plan

### Sequencing

1. **Phase A — Trivial mechanical replacements** (H1, H2, H3, H4, H5, H6, H8, M1, M2):
   - All sites match the pattern: `Convert.ToXxx(value)` or `((IConvertible)value).ToXxx(InvariantCulture)`.
   - Direct replacement with `Converts.ToXxx(value)`.
   - ~24 sites across 8 files.

2. **Phase B — ILKernelGenerator.Scan.cs** (H7):
   - Generic context (`TIn` is type parameter), so use `Converts.ToXxx((object)src[…])`.
   - ~13 sites in 1 file.
   - Performance note: scalar fallback is already non-critical (IL emit handles fast path).

3. **Phase C — Indexing parsing** (M3, M4):
   - Restructure `case IConvertible o:` to handle Half/Complex via type-pattern fallthrough.
   - ~4 sites in 2 files.

### Tests

Add Round 5 region to `ConvertsBattleTests.cs` (or new `BattleTests.LeftoverFixes.cs`) covering:

- `np.full(shape, Half.One, dtype=Int32)` and similar (H1/H2)
- `np.searchsorted(halfArray, value)` (H3)
- `np.matmul(halfMatrix, halfMatrix)` forced into scalar fallback (H4)
- `np.dot(halfArray, halfArray)` forced into scalar fallback (H5)
- `np.convolve(halfArray, halfArray)` (H6)
- `np.cumsum(halfArray)` and `np.cumprod(complexArray)` (H7)
- `np.mean(scalarHalfArray)` with null typeCode (H8)
- `np.repeat(arr, halfArray)` (M1, optional)
- `arr << (Half)2` (M2, optional)
- `arr[(Half)3]` (M3/M4, optional)

Estimated +20-30 battletests.

### Risk

Low. All replacements are semantic-preserving for IConvertible-supporting types and only ADD
support for Half/Complex/char. Should not regress any existing tests.

The Scan.cs (H7) fix introduces one extra boxing per element in the scalar fallback path, but
this path is already the slowest fallback (only used when SIMD/IL kernel can't handle dtype) and
performance is not a concern.

### Estimated Scope

- ~56 site edits across 11 files
- ~20-30 new battletests
- 1 commit with detailed Group A/B/C breakdown matching Round 1-4 style
- Likely 200-300 lines of changes total

---

## Additional Parity Bugs — Battletest Findings (2026-04-17)

**Scope note:** The items below are **orthogonal** to the `IConvertible` cleanup in Round 5.
A full NumPy 2.4.2 battletest of Half/Complex/SByte surfaced behavioural/coverage gaps — missing
IL kernel paths, swapped reduction identity handling, NaN-propagation mismatches, and missing
dtype branches in axis dispatchers. Fixing Round 5 will **not** resolve any of these.

**Methodology:** Every test was run side-by-side against `python -c "import numpy as np; ..."`
on NumPy 2.4.2. Full test suite passes (5974/5974) because these bugs sit on code paths the
existing `NewDtypes*Tests` / `Casting*Tests` don't exercise.

**Bugs confirmed passing NumPy parity (not listed below):** SByte arithmetic/reductions/promotion,
Half arithmetic/elementwise sum/mean/std/var/cumsum/cumprod/isnan/isinf/isfinite/argmax/argmin/
comparisons, Complex arithmetic/abs/elementwise sum/mean/cumsum/cumprod/isnan/isinf/isfinite/
comparisons, full 12×13 type promotion matrix (NEP50), full astype matrix including NaN/Inf/
overflow/signed↔unsigned wrapping.

---

### Severity 1 — Silent data corruption (ship-blocker)

#### B1. `np.min(Half)` / `np.max(Half)` return identity value, never update

```
np.min([Half 1,2,3,4,5])   → +∞       (expected 1)
np.max([Half 1,2,3,4,5])   → -∞       (expected 5)
```

**Root cause:** `ILKernelGenerator.Reduction.cs:1191` `EmitScalarMinMax` emits `OpCodes.Bgt`/`Blt`,
which are not valid IL for the `Half` struct. `GetMathMinMaxMethod` returns `null` for Half
(no `Math.Max(Half,Half)` exists in BCL). The generated kernel compiles but the comparison never
takes the update branch, so the accumulator stays at its identity value forever.

**Scoped to:** Elementwise reduction only. Axis-based min/max on Half works correctly (uses a
different path). Single-element arrays work (fast-path skips the kernel).

**Fix sketch:** Add `HalfMinHelper`/`HalfMaxHelper` internal methods (cf. existing
`NanMinHalfHelper`/`NanMaxHalfHelper` at `ILKernelGenerator.Masking.NaN.cs:1289,1311`), dispatch
Half min/max through them in `DefaultEngine.ReductionOp.cs:201` (min) and `:172` (max), bypassing
`ExecuteElementReduction<Half>`.

#### B2. `np.mean(Complex, axis=N)` drops imaginary part and returns `float64`

```
np.mean([[1+1j,2+2j,3+3j],[4+4j,5+5j,6+6j],[7+7j,8+8j,9+9j]], axis=0)
NumPy:    [4+4j 5+5j 6+6j]   dtype=complex128
NumSharp: [4,   5,   6]      dtype=float64   ← imaginary lost
```

**Root cause:** Axis-mean output-type dispatcher treats Complex as "scalar mean → promote to
double" instead of preserving Complex. The elementwise `np.mean(complexArr)` case (no axis) is
correct — only the axis variant is broken.

**Fix location:** `Default.Reduction.Mean.cs` (+2 lines for Mean) / axis dispatcher type-code
selection.

#### B3. `1/0 Complex` returns `(NaN, NaN)` instead of `(inf, NaN)`

```
NumPy:    np.array([1+0j]) / np.array([0+0j])  →  [inf+nanj]
NumSharp:                                      →  <NaN; NaN>
```

NumPy's division uses the IEEE 754-style extended complex division (real part = sign(inf) *
real(num), imag part = NaN when both denom parts are 0). System.Numerics.Complex division gives
plain (NaN, NaN). Fix requires a custom division kernel override in the Complex path.

---

### Severity 2 — NotSupportedException on operations NumPy supports

#### B4. `np.prod(Half)` and `np.prod(Complex)` throw

```
NumPy:    np.prod([Half 1,2,3,4])      →  24.0          (dtype float16)
NumSharp: →  NotSupportedException: Prod not supported for type Half
```

**Root cause:** `DefaultEngine.ReductionOp.cs:145` `prod_elementwise_il` has fallback for Half/
Complex missing. Compare against `sum_elementwise_il` (line 115) which has
`NPTypeCode.Half => SumElementwiseHalfFallback(arr)` and
`NPTypeCode.Complex => SumElementwiseComplexFallback(arr)`.

**Fix:** Add `ProdElementwiseHalfFallback` / `ProdElementwiseComplexFallback` alongside existing
sum fallbacks. Also applies to `np.nanprod(Complex)`.

#### B5. `np.max(sbyte, axis=N)` / `np.min(sbyte, axis=N)` throw

```
NumPy:    np.max([[1,2,3],[4,5,6],[7,8,9]] as int8, axis=0)  →  [7 8 9]  dtype=int8
NumSharp: →  NotSupportedException: Type System.SByte not supported for axis reduction
```

**Root cause:** `ILKernelGenerator.Reduction.Axis.Simd.cs:502` `GetIdentityValue<T>` is missing
a `typeof(T) == typeof(sbyte)` branch. All other integer widths are covered (byte, short, ushort,
int, uint, long, ulong).

**Fix:** Add `sbyte` branch with identity values `{Sum: 0, Prod: 1, Min: sbyte.MaxValue, Max: sbyte.MinValue}`.
Only 12 lines. Non-axis sum/prod/min/max already work for sbyte.

#### B6. `np.cumsum(Half | Complex, axis=N)` throws

```
NumSharp: "AxisCumSum not supported for type Half"
NumSharp: "AxisCumSum not supported for type Complex"
```

Elementwise cumsum/cumprod already work (correct dtype output). Only the axis variant is broken.
**This overlaps with LEFTOVER §H7** — both issues sit in `ILKernelGenerator.Scan.cs`. However
H7's fix (routing through `Converts.ToXxx`) addresses the `Convert.ToDouble(src[…])` sites;
B6 requires adding the **dispatch case** itself for Half/Complex in the scan dispatcher, which
currently rejects these types outright before reaching the scalar fallback where H7 applies.

**Order of ops:** B6 dispatch addition must come first (or together with H7). H7's scalar
rewrite alone isn't visible until the dispatcher accepts the type.

#### B7. `np.argmax(Complex, axis=N)` throws

```
NumPy:    np.argmax(complexMatrix, axis=0)  →  [2 2 2]
NumSharp: →  NotSupportedException: ArgMax/ArgMin not supported for type Complex
```

Elementwise argmax/argmin for Complex already works (with minor ordering bugs — see B12/B13).
Only the axis variant is broken. Fix requires adding Complex case to the axis ArgMax/ArgMin
dispatcher (`ILKernelGenerator.Reduction.Axis.Arg.cs`).

#### B8. `np.min(Complex)` / `np.max(Complex)` throw

```
NumPy:    np.min([3.5+2j, -1.5+5j])  →  (-1.5+5j)      (lex ordering by real, then imag)
NumSharp: →  NotSupportedException: Min not supported for type Complex
```

`EmitLoadMinValue`/`EmitLoadMaxValue` in `ILKernelGenerator.Reduction.cs:860,917` explicitly
throw `"Complex type does not support Min/Max operations"` — but NumPy **does** support this via
lexicographic ordering. Fix requires adding a Complex scalar helper (cf. B1 fix for Half) using
`Compare(a,b) = a.Real != b.Real ? a.Real.CompareTo(b.Real) : a.Imaginary.CompareTo(b.Imaginary)`.

#### B9. `np.unique(Complex)` throws

```
NumPy:    np.unique([1+2j, 3+4j, 1+2j, 3+4j])  →  [1+2j, 3+4j]
NumSharp: →  NotSupportedException: Specified method is not supported.
```

**Current state:** `NEW_DTYPES_HANDOFF.md` explicitly excludes Complex from `unique()` because
"Complex doesn't implement IComparable". NumPy handles this via lex ordering. Requires a custom
comparer path for Complex in `NDArray.unique.cs`.

#### B10. `np.maximum(Half,Half)` / `np.minimum(Half,Half)` throw

```
NumPy:    np.maximum([nan,1,2] float16, [1,5,0] float16)  →  [nan 5 2]
NumSharp: →  NotSupportedException: ClipNDArray not supported for dtype Half
```

Binary `np.maximum`/`np.minimum` (not to be confused with reduction `np.max`/`np.min`) missing
Half dispatch in `Default.ClipNDArray.cs`. Note the NaN propagation behaviour (NaN wins) is
required for NumPy parity.

#### B11. Half missing unary math operations

```
np.log10(Half) → NotSupportedException
np.log2(Half)  → NotSupportedException
np.cbrt(Half)  → NotSupportedException
np.exp2(Half)  → NotSupportedException
np.log1p(Half) → NotSupportedException
np.expm1(Half) → NotSupportedException
```

**Root cause:** `ILKernelGenerator.Unary.Decimal.cs:449` default throws for unhandled unary ops.
Current Half coverage: `Negate, Abs, Sqrt, Sin, Cos, Tan, Exp, Log, Floor, Ceil, Truncate, Square,
Reciprocal, Sign, IsNan, IsInf, IsFinite`. Missing ops listed above are all present in NumPy
for float16. Fix: add `CachedMethods.HalfLog10/Log2/Cbrt/Exp2/Log1p/Expm1` entries and emit
`MathF.Xxx((float)(double)value)` through Half conversion. Per-op: ~4 lines of IL emit.

---

### Severity 3 — Wrong output values / semantic mismatch

#### B12. `np.argmax/argmin(Complex)` with tied real parts — wrong index

```
Input: [5+1j, 5+10j, 5-3j]   (all real=5)
NumPy:    argmax=1 (imag 10 wins)   argmin=2 (imag -3 wins)
NumSharp: argmax=1 ✓                argmin=0 ✗  (returned first element, ignoring imag)
```

Argmax path is correct; argmin path compares only real, ignoring imag tiebreaker.

#### B13. `np.argmax/argmin(Complex)` with NaN — wrong NaN-propagation

```
Input: [1+2j, NaN+0j, 5+10j]
NumPy:    argmax=1 (first NaN wins)   argmin=1 (first NaN wins)
NumSharp: argmax=2 ✗                  argmin=0 ✗
```

NumPy's rule: the first NaN encountered short-circuits argmax/argmin to that index. NumSharp
skips NaN entirely.

#### B14. `np.nanmean(Half)` / `np.nanstd(Half)` / `np.nanvar(Half)` return `NaN`

```
Input: [Half 1, 2, NaN, 4]
NumPy:    nanmean=2.334  nanstd=1.247  nanvar=1.556   (skips NaN, computes on [1,2,4])
NumSharp: nanmean=NaN    nanstd=NaN    nanvar=NaN     (NaN propagates)
```

`np.nansum(Half)` and `np.nanprod(Half)` already work correctly — they return 7 and 8
respectively, skipping NaN. The bug is isolated to the mean/std/var NaN-skipping reductions
for Half.

#### B15. `np.nansum(Complex)` / `np.nanmean(Complex)` don't skip NaN

```
Input: [1+2j, (NaN+0j), 3+4j]
NumPy:    nansum=(4+6j)   nanmean=(2+3j)
NumSharp: nansum=<NaN;6>  nanmean=<NaN;NaN>    (NaN propagates, element not skipped)
```

Same family as B14 but for Complex dtype. Requires NaN-aware reduction helpers in the Complex
path (currently the Complex reduction fallback doesn't check `ComplexIsNaNHelper` per-element).

#### B16. `np.std(Half, axis=N)` / `np.var(Half, axis=N)` return `float64`, not `float16`

```
NumPy:    np.std(halfMatrix, axis=0)  →  dtype=float16
NumSharp: →  dtype=float64
```

Elementwise `np.std(Half)` correctly returns `float16`. Only axis variant up-promotes to double.
Minor dtype-ergonomics bug — values are correct, precision just wider than NumPy.

---

### Cross-reference with Round 5 (IConvertible cleanup)

| Battletest bug | Round 5 item | Relationship |
|---|---|---|
| B6 (axis cumsum for Half/Complex) | H7 | Partial overlap — H7 fixes scalar-fallback `Convert.ToXxx`; B6 requires adding the dispatch case itself. Fix **B6 before or together with H7**, otherwise H7's fix is unreachable for Half/Complex. |
| all others (B1–B5, B7–B16) | — | Independent. Not fixable by Round 5. |

---

### Proposed Round 6 (sequenced after Round 5)

Ordering by impact ÷ effort:

1. **Quick wins (~30-60 lines each):** B5 (sbyte axis identity), B4 (prod Half/Complex fallback),
   B11 (Half unary math — 6 ops × ~4 lines each).
2. **Medium (~50-150 lines each):** B1 (Half min/max helper), B10 (Half maximum/minimum binary),
   B16 (Half axis std/var dtype), B14 (Half nanmean/nanstd/nanvar NaN-skip).
3. **Complex-specific (larger scope):** B2 (Complex axis mean dtype — data loss, prioritise),
   B8 (Complex min/max lex), B9 (Complex unique lex), B7 (Complex axis argmax), B6 (Half/Complex
   axis cumsum — combine with H7), B12/B13 (Complex argmax/argmin tiebreak + NaN), B15 (Complex
   nansum/nanmean NaN-skip).
4. **Defer / needs design:** B3 (Complex 1/0 = inf+nanj — requires custom division kernel; rare
   in practice).

### Test plan for Round 6

- Add battletests to a new `test/NumSharp.UnitTest/NewDtypes/NewDtypesBattletestGapsTests.cs`
  mirroring the Python `-c` commands used during this battletest.
- Each bug gets 2-3 tests: the minimal reproducer plus one variation (different shape,
  with/without NaN, etc.).
- Estimated +40-60 tests.
- Given the severity of B1 and B2 (silent data corruption), these two should also gain
  `[OpenBugs]`-tagged reproducers immediately so CI catches regressions while Round 6 is
  planned / before fix lands.
