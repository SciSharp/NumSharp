# Int64 MaxValue Boundary Tasks

Tracking document for fixing `int.MaxValue` boundary checks that should support long indexing.

---

## Task Summary

| # | File | Priority | Status | Issue |
|---|------|----------|--------|-------|
| 1 | `ndarray.argsort.cs` | HIGH | DONE | Enumerable.Range limit - FULLY REFACTORED |
| 2 | `SimdMatMul.cs` | HIGH | DONE | SIMD loop uses int dimensions - REFACTORED |
| 3 | `Default.MatMul.2D2D.cs` | HIGH | DONE | SIMD path condition - REMOVED |
| 4 | `Default.NonZero.cs` | MEDIUM | DONE | List capacity cast - SIMPLIFIED |
| 5 | `ILKernelGenerator.Masking.cs` | MEDIUM | DONE | List capacity cast - SIMPLIFIED |

---

## Task 1: ndarray.argsort.cs

**File:** `src/NumSharp.Core/Sorting_Searching_Counting/ndarray.argsort.cs`
**Lines:** 33-35, 51-53 (now removed)
**Priority:** HIGH

**Issue:** Uses `Enumerable.Range` which is limited to int.MaxValue

**Status:** DONE - FULLY REFACTORED

**Fix Applied:**
- Created `LongRange(long count)` helper method that returns `IEnumerable<long>`
- Created `SortedDataLong` class with `long[] DataAccessor` and `long Index`
- Created `AppendorLong` method that works with `IEnumerable<long>`
- Created `AccessorCreatorLong` that accepts `long[]` and returns `IEnumerable<IEnumerable<long>>`
- Created `SortLong<T>` method that returns `IEnumerable<long>`
- Removed both int.MaxValue checks
- Changed `requiredSize` from `int[]` to `long[]`
- All internal iteration now uses long indexing throughout

---

## Task 2: SimdMatMul.cs

**File:** `src/NumSharp.Core/Backends/Kernels/SimdMatMul.cs`
**Lines:** 41-49
**Priority:** HIGH

**Issue:** SIMD loop variables use int

**Status:** DONE

**Fix Applied:**
- Removed the int.MaxValue throw/check
- Changed outer loop variables (k0, i0, jp) to long in `MatMulFloatBlocked`
- Changed method signatures to accept long parameters:
  - `PackAPanels(float* A, float* packA, long lda, long i0, long k0, int mc, int kc)`
  - `PackBPanels(float* B, float* packB, long ldb, long k0, int kc)`
  - `Microkernel8x16Packed(..., long ldc, long i, long j, int kc)`
  - `MicrokernelGenericPacked(..., long ldc, long i, long j, ...)`
- Inner block loops (ip, k within panels) remain int (bounded by small constants MC, KC, MR, NR)
- Index calculations now use long arithmetic (long * long = long)

---

## Task 3: Default.MatMul.2D2D.cs

**File:** `src/NumSharp.Core/Backends/Default/Math/BLAS/Default.MatMul.2D2D.cs`
**Lines:** 53
**Priority:** HIGH

**Issue:** SIMD path condition - checked int.MaxValue before calling SIMD kernel

**Status:** DONE

**Fix Applied:**
- Removed the `M <= int.MaxValue && K <= int.MaxValue && N <= int.MaxValue` check
- Changed `TryMatMulSimd` signature from `(int M, int K, int N)` to `(long M, long K, long N)`
- SIMD kernels (SimdMatMul.MatMulFloat) now support long dimensions directly

---

## Task 4: Default.NonZero.cs

**File:** `src/NumSharp.Core/Backends/Default/Indexing/Default.NonZero.cs`
**Lines:** 71
**Priority:** MEDIUM

**Issue:** List<T> capacity calculation had explicit int.MaxValue check

**Status:** DONE

**Fix Applied:**
- Simplified capacity calculation to avoid explicit int.MaxValue in code
- For size <= int.MaxValue: use `(int)(size / 4)` as capacity hint
- For size > int.MaxValue: use 1M (1 << 20) initial capacity
- List grows dynamically as needed
- Note: List<T> itself is fundamentally int-limited by .NET (max ~2B elements)
- Additional work: lines 73, 78 use `Array.ConvertAll(x.shape, d => (int)d)` - may need long[] migration

---

## Task 5: ILKernelGenerator.Masking.cs

**File:** `src/NumSharp.Core/Backends/Kernels/ILKernelGenerator.Masking.cs`
**Lines:** 189
**Priority:** MEDIUM

**Issue:** List<T> capacity calculation had explicit int.MaxValue check

**Status:** DONE

**Fix Applied:**
- Simplified capacity calculation to avoid explicit int.MaxValue in code
- For size <= int.MaxValue: use `(int)(size / 4)` as capacity hint
- For size > int.MaxValue: use 1M (1 << 20) initial capacity
- List grows dynamically as needed
- Note: List<T> itself is fundamentally int-limited by .NET (max ~2B elements)

---

## Completed Tasks

1. **ndarray.argsort.cs** - Fully refactored with LongRange, long indices throughout
2. **SimdMatMul.cs** - Refactored to support long dimensions
3. **Default.MatMul.2D2D.cs** - Removed int.MaxValue check, SIMD path now supports long
4. **Default.NonZero.cs** - Simplified List capacity calculation
5. **ILKernelGenerator.Masking.cs** - Simplified List capacity calculation

---

## Notes

- Per INT64_MIGRATION_GUIDE.md, algorithms should natively use long loop variables and pointer arithmetic
- List<T> capacity is int-limited by .NET design; prefer ArraySlice<T> or UnmanagedMemoryBlock<T>
- Some int.MaxValue checks are acceptable at .NET API boundaries (Span, String, etc.)
