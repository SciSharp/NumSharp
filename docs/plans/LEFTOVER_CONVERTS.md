# Leftover Convert / IConvertible Sites Outside `Converts.cs`

**Date:** 2026-04-17
**Branch:** `worktree-half`
**Audit scope:** All `src/NumSharp.Core/**/*.cs` outside `Utilities/Converts*.cs`.

## Background

NumSharp supports 15 dtypes including **`Half`** and **`Complex`**, neither of which implements
`System.IConvertible`. Any code path that calls `((IConvertible)x).ToY(...)` or `System.Convert.ToY(x)`
throws `InvalidCastException` for Half/Complex sources.

The fix pattern is to route through `Converts.ToY(x)` (the NumSharp object dispatcher), which handles
all 15 dtypes with NumPy-parity semantics (truncation, wrapping, NaN handling).

---

## High Priority — Half/Complex break NumPy-aligned operations

| # | Location | Sites | Status | Impact |
|---|---|---:|---|---|
| H1+H2 | `ArraySlice.cs:408-496` (2 `Allocate(…, fill)` overloads) | 26 | ✅ Round 5A (`44dd04fc`) | `np.full((3,3), Half.One, dtype=int32)` throws |
| H3 | `np.searchsorted.cs:51,61,85` | 3 | ✅ Round 5A (`44dd04fc`) | searchsorted on Half/Complex array throws |
| H4 | `Default.MatMul.2D2D.cs:323,329` | 2 | ⏳ TODO | matmul scalar-fallback on Half throws |
| H5 | `Default.Dot.NDMD.cs:371,375` | 2 | ⏳ TODO | dot product scalar-fallback on Half throws |
| H6 | `NdArray.Convolve.cs:154,155` | 2 | ⏳ TODO | `np.convolve` on Half throws |
| H7 | `ILKernelGenerator.Scan.cs` (~13 sites) | 13 | ⏳ TODO | CumSum/CumProd scalar fallback on Half throws |
| H8 | `DefaultEngine.ReductionOp.cs:310` | 1 | ⏳ TODO | reduction scalar fallback on Half throws |

### H4 — `Default.MatMul.2D2D.cs:323,329`

```csharp
double aik = Convert.ToDouble(left.GetValue(leftCoords));
double bkj = Convert.ToDouble(right.GetValue(rightCoords));
```

`GetValue(...)` returns boxed object. If matrix is Half/Complex dtype, `Convert.ToDouble(boxed Half)` throws.
Scalar fallback path used when SIMD/IL kernel can't handle the dtype combination.

**Fix:** `Converts.ToDouble(...)`.

### H5 — `Default.Dot.NDMD.cs:371,375`

```csharp
double lVal = Convert.ToDouble(lhs.GetValue(lhsCoords));
double rVal = Convert.ToDouble(rhs.GetValue(rhsCoords));
```

Identical pattern to H4. Same fix.

### H6 — `NdArray.Convolve.cs:154,155`

```csharp
double aVal = Convert.ToDouble(aPtr[j]);
double vVal = Convert.ToDouble(vPtr[k - j]);
```

`aPtr` is typed pointer (e.g., `Half*`). The deref auto-boxes when passed to `Convert.ToDouble(object)`.
NumPy's `convolve` supports float16, so this is a real parity gap.

**Fix:** `Converts.ToDouble((object)aPtr[j])` (explicit boxing). Or, if the surrounding generic context
allows direct unboxed conversion, prefer `(double)(Half)aPtr[j]`.

### H7 — `ILKernelGenerator.Scan.cs` (~13 sites)

| Line | Code | Context |
|---:|---|---|
| 1128 | `product *= Convert.ToInt64(src[…])` | AxisCumProd, TOut=long |
| 1138 | `product *= Convert.ToDouble(src[…])` | AxisCumProd, TOut=double |
| 1148 | `product *= Convert.ToDecimal(src[…])` | AxisCumProd, TOut=decimal |
| 1947 | `sum += Convert.ToInt64(src[…])` | AxisCumSum, TOut=long |
| 1957 | `sum += Convert.ToDouble(src[…])` | AxisCumSum, TOut=double |
| 1967 | `sum += Convert.ToSingle(src[…])` | AxisCumSum, TOut=float |
| 1977 | `sum += Convert.ToUInt64(src[…])` | AxisCumSum, TOut=ulong |
| 1987 | `sum += Convert.ToDecimal(src[…])` | AxisCumSum, TOut=decimal |
| 2392 | `sum += Convert.ToDouble(src[i])` | ElementwiseCumSum, TOut=double |
| 2402 | `sum += Convert.ToInt64(src[i])` | ElementwiseCumSum, TOut=long |
| 2412 | `sum += Convert.ToDecimal(src[i])` | ElementwiseCumSum, TOut=decimal |
| 2422 | `sum += Convert.ToSingle(src[i])` | ElementwiseCumSum, TOut=float |
| 2432 | `sum += Convert.ToUInt64(src[i])` | ElementwiseCumSum, TOut=ulong |

`src` is `TIn*` (e.g., `Half*` or `Complex*`); `src[i]` is `TIn`. Boxing into `Convert.ToXxx(object)` throws
for Half/Complex. Note: Complex source for cumsum/cumprod IS meaningful in NumPy.

**Fix:** `Converts.ToXxx((object)src[…])`. The boxing is unavoidable when calling the object dispatcher;
performance of scalar fallback isn't critical (IL kernels handle the fast path).

### H8 — `DefaultEngine.ReductionOp.cs:310`

```csharp
return typeCode.HasValue ? Converts.ChangeType(val, typeCode.Value) : Convert.ToDouble(val);
```

When `typeCode` is null, falls back to `Convert.ToDouble(val)`. Complex source is special-cased earlier
(line 308-309), so by line 310 only Half is broken.

**Fix:** `Converts.ToDouble(val)`.

---

## Medium Priority — Rare edge cases

| # | Location | Sites | Status | Impact |
|---|---|---:|---|---|
| M1 | `np.repeat.cs:75,172` | 2 | ⏳ TODO | Half/Complex as `repeats` array |
| M2 | `Default.Shift.cs:136` | 1 | ⏳ TODO | Half as shift amount (unusual) |
| M3+M4 | `NDArray.Indexing.Selection.{Setter,Getter}.cs` | 4 | ⏳ TODO | Half/Complex as fancy index |

### M1 — `np.repeat.cs:75,172`

```csharp
long count = Convert.ToInt64(repeatsFlat.GetAtIndex(i));
```

`repeats` is normally an int dtype, but if user passes Half/Complex, throws with cryptic IConvertible
error instead of clean type error.

**Fix:** `Converts.ToInt64(repeatsFlat.GetAtIndex(i))`.

### M2 — `Default.Shift.cs:136`

```csharp
int shiftAmount = Convert.ToInt32(rhs);
```

Shift amounts are typically int literals. Half/Complex shift amount is an unusual edge case.

**Fix:** `Converts.ToInt32(rhs)`.

### M3+M4 — `NDArray.Indexing.Selection.Setter.cs:126,188` + `Getter.cs:109,172`

```csharp
case IConvertible o: return Slice.Index(o.ToInt64(CultureInfo.InvariantCulture));
case IConvertible o:
    indices.Add(NDArray.Scalar<int>(o.ToInt32(CultureInfo.InvariantCulture)));
```

Half/Complex don't match `IConvertible` and fall through to "Unsupported slice type" error. Less broken
than other sites (gives clean error) but inconsistent with NumPy where `arr[Half(3)]` would work.

**Fix:** Add explicit `case Half h:` / `case Complex c:` branches before the IConvertible case, or
restructure to use `Converts.ToInt64(o)` for any object.

---

## Skip — No Fix Needed

### `Converts.Native.cs` DateTime converters (~14 sites)

Lines: 108, 271, 455, 644, 825, 1005, 1194, 1367, 1552, 1723, 1930, 2083, 2235, 2403, 2685-2789.

`DateTime` is not a NumPy dtype. NumPy's `datetime64` has different semantics (epoch-based). These
methods exist for .NET interop completeness, not NumPy parity. Half/Complex → DateTime has no
defined meaning anyway.

### `_NumPy` helper `_` defaults in `Converts.cs:258-551`

```csharp
_ => Converts.ToBoolean(((IConvertible)value).ToDouble(null))   // line 258
_ => (Half)((IConvertible)value).ToDouble(null)                 // line 510
_ => new Complex(((IConvertible)value).ToDouble(null), 0)       // line 531
```

Each helper is a switch where Half, Complex, char, and 12 classic types are handled BEFORE the `_`
default. Default only fires for exotic source types (string, etc.) which all implement IConvertible.
Half/Complex never reach the default branch.

### `ILKernelGenerator.Reduction.NaN.cs:926,930` — IL constant emission

```csharp
il.Emit(OpCodes.Ldc_R4, Convert.ToSingle(value));
il.Emit(OpCodes.Ldc_R8, Convert.ToDouble(value));
```

`value` is a runtime constant (reduction identity element like 0 or 1) for IL `Ldc_R4`/`Ldc_R8` opcodes.
Always primitive numerics. Half/Complex constants don't flow through this path because they don't have
SIMD reduction kernels needing IL constant emission.

### `Converts.cs:76,1173,1181` — Dead code or post-fallback

- Line 76: third-tier fallback in `CreateIntegerConverter` after explicit Half/Complex/IConvertible
  checks. Only exotic non-IConvertible non-Half non-Complex types reach here. None exist in NumSharp.
- Lines 1173, 1181: inside `#if _REGEN` block — `_REGEN` symbol not defined in any active build config.

### `ILKernelGenerator.Masking.VarStd.cs:352,359` — Decimal-only path

```csharp
doubleSum += Convert.ToDouble(src[i]);
double diff = Convert.ToDouble(src[i]) - mean;
```

Per inline comment "For integer types", `src` is sbyte/byte/int16/uint16/int32/uint32/int64/uint64 —
all implement IConvertible. Half/Complex paths are handled in the preceding float branch.

---

## Round 5 Plan (remaining)

### Round 5B — Math/BLAS/Convolve scalar fallbacks

Sites: H4 (2), H5 (2), H6 (2), H8 (1) = **7 sites** in 4 files.
Pattern: `Convert.ToDouble(x)` → `Converts.ToDouble(x)`.
Tests: `np.matmul(half2D, half2D)`, `np.dot(halfArr, halfArr)`, `np.convolve(halfArr, halfArr)`,
`np.mean(scalarHalfArray)` with null typeCode.

### Round 5C — Scan kernel scalar fallback

Sites: H7 = **13 sites** in 1 file.
Pattern: `Convert.ToXxx(src[…])` → `Converts.ToXxx((object)src[…])`.
Tests: `np.cumsum(halfArr)`, `np.cumprod(halfArr)`, `np.cumsum(complexArr)`, `np.cumprod(complexArr)`
plus axis variants.

### Round 5D — Edge cases (optional)

Sites: M1 (2), M2 (1), M3+M4 (4) = **7 sites** in 4 files.
Pattern: same as 5B + restructure `case IConvertible o:` for Half/Complex.
Tests: `np.repeat(arr, halfArr)`, `arr << (Half)2`, `arr[(Half)3]`.

### Total Remaining

- **20 sites** across 8 files (Round 5B+5C high; 5D medium optional).
- **20-30 new battletests** estimated.
- **Risk:** Low. Pattern is mechanical; routes through already-tested `Converts.ToXxx` dispatchers.
