# Long Indexing Migration - Remaining Issues

Issues discovered by auditing the codebase against the int64 migration spirit.

**Audit Date**: Based on commit `111b4076` (longindexing branch)
**Updated**: 2026-03-17 - Added H12-H22, M9-M11 from comprehensive code search
**Updated**: 2026-03-17 - Added H23, L11-L12 from post-rebase diff scan
**Updated**: 2026-03-17 - Fixed H4, H6, H10, H12, H14, H15, H16, H20; reclassified H3, H17-H19, H21-H22 as LOW
**Updated**: 2026-03-17 - Fixed H8, H9, H11, H23 (batch 2)

---

## Summary

| Priority | Count | Category |
|----------|-------|----------|
| HIGH | 11 | Missing long overloads, int parameters/variables |
| MEDIUM | 11 | IL kernel comments, internal int usage |
| LOW | 19 | Acceptable .NET boundary exceptions |

### Recently Fixed (this session)

**Batch 1:**
- H4: `np.repeat` - changed `GetInt32` to `GetInt64`, `int count/j` to `long`
- H6: `np.searchsorted` - empty array returns `typeof(long)` for consistency
- H10: `UnmanagedHelper.CopyTo` - offset parameter changed to `long`
- H12: `SimdMatMul.MatMulFloatSimple` - changed `int M,N,K` to `long`
- H14: `Default.Dot.ExpandStartDim/ExpandEndDim` - returns `long[]` now
- H15: `NDArray.Normalize` - loop counters changed to `long`
- H16: `Slice.Index` in Selection - uses `ToInt64` instead of `ToInt32`
- H20: `np.asarray` - uses `Array.Empty<long>()` for scalar shapes

**Batch 2:**
- H8: `np.linspace` - added `long num` overloads, changed loop counters to `long`
- H9: `np.roll`/`NDArray.roll` - added `long shift` primary overloads
- H11: `np.array<T>(IEnumerable, size)` - added `long size` overload
- H23: `NumSharp.Bitmap` - added overflow checks before casting shape to int

### Reclassified as LOW (.NET Array boundary)
- H3: `np.vstack` - entire implementation is commented out (dead code)
- H17-H19, H21-H22: Use .NET `Array.Length`/`GetLength()` which return `int`

---

## HIGH Priority Issues

### H1. NDArray/UnmanagedStorage.Getters Missing `long[]` Overloads

**Files:**
- `Backends/NDArray.cs:612-794`
- `Backends/Unmanaged/UnmanagedStorage.Getters.cs:18-530`

**Issue:** All typed getter methods only have `int[]` overloads, no `long[]` primary:

```csharp
// CURRENT - only int[] exists
public bool GetBoolean(int[] indices) => Storage.GetBoolean(indices);
public byte GetByte(int[] indices) => Storage.GetByte(indices);
public double GetDouble(int[] indices) => Storage.GetDouble(indices);
// ... 12 more typed getters

// MISSING - need long[] primary with params
public bool GetBoolean(params long[] indices);
public byte GetByte(params long[] indices);
// etc.
```

**Affected methods (15 each in NDArray and UnmanagedStorage):**
- `GetData(int[])` - has long[] ✅
- `GetValue(int[])` - MISSING long[]
- `GetValue<T>(int[])` - MISSING long[]
- `GetBoolean(int[])` - MISSING long[]
- `GetByte(int[])` - MISSING long[]
- `GetInt16(int[])` - MISSING long[]
- `GetUInt16(int[])` - MISSING long[]
- `GetInt32(int[])` - MISSING long[]
- `GetUInt32(int[])` - MISSING long[]
- `GetInt64(int[])` - MISSING long[]
- `GetUInt64(int[])` - MISSING long[]
- `GetChar(int[])` - MISSING long[]
- `GetDouble(int[])` - MISSING long[]
- `GetSingle(int[])` - MISSING long[]
- `GetDecimal(int[])` - MISSING long[]

**Fix pattern:**
```csharp
// Add long[] primary overload with params
public bool GetBoolean(params long[] indices) => Storage.GetBoolean(indices);

// int[] delegates to long[] (backward compat, no params)
public bool GetBoolean(int[] indices) => GetBoolean(Shape.ComputeLongShape(indices));
```

---

### H2. NDArray Typed Setters Missing Some `long[]` Overloads

**File:** `Backends/NDArray.cs:1053-1175`

**Issue:** Only 4 typed setters have `long[]` overloads (`SetInt32`, `SetInt64`, `SetDouble`, `SetString`). The other 9 are missing:

```csharp
// Have long[] ✅
public void SetInt32(int value, params long[] indices);
public void SetInt64(long value, params long[] indices);
public void SetDouble(double value, params long[] indices);

// MISSING long[] ❌
public void SetBoolean(bool value, int[] indices);  // only int[]
public void SetByte(byte value, int[] indices);     // only int[]
public void SetChar(char value, int[] indices);     // only int[]
public void SetSingle(float value, int[] indices);  // only int[]
public void SetDecimal(decimal value, int[] indices); // only int[]
public void SetUInt16(ushort value, int[] indices); // only int[]
public void SetUInt32(uint value, int[] indices);   // only int[]
public void SetUInt64(ulong value, int[] indices);  // only int[]
public void SetInt16(short value, int[] indices);   // only int[]
```

---

### H3. ⚠️ RECLASSIFIED AS LOW - np.vstack Implementation Is Dead Code

**File:** `Manipulation/np.vstack.cs:26,30`

**Status:** Entire implementation is commented out (inside `/* ... */`). No fix needed as this is dead code. If the implementation is restored, it should use `long[]` shapes.

---

### H4. ✅ FIXED - np.repeat Uses `int count` for Per-Element Repeat

**File:** `Manipulation/np.repeat.cs:69,159,161`

**Status:** Fixed - changed `GetInt32` to `GetInt64`, `int count/j` to `long count/j`

```csharp
// NOW USES:
long count = repeatsFlat.GetInt64(i);
for (long j = 0; j < count; j++)
```

---

### H5. NDArray.unique SortUniqueSpan Uses `int count`

**File:** `Manipulation/NDArray.unique.cs:140,115,130`

```csharp
// Line 140 - method signature
private static unsafe void SortUniqueSpan<T>(T* ptr, int count)

// Lines 115, 130 - calls
SortUniqueSpan<T>((T*)dst.Address, hashset.Count);
```

**Issue:** `hashset.Count` is int, and `SortUniqueSpan` takes int. If unique element count exceeds int.MaxValue, this fails. Also uses `Span<T>` which has int limitation.

**Acceptable for now:** Hashset<T>.Count is inherently int-limited by .NET. Would need custom hash implementation for >2B unique elements.

---

### H6. ✅ FIXED - np.searchsorted Empty Array Returns Wrong Type

**File:** `Sorting_Searching_Counting/np.searchsorted.cs:48`

**Status:** Fixed - empty array now returns `typeof(long)` for consistency with non-empty case.

---

### H7. nanmean/nanstd/nanvar Array Allocation with (int) Cast

**Files:**
- `Statistics/np.nanmean.cs:145,191`
- `Statistics/np.nanstd.cs:198,272`
- `Statistics/np.nanvar.cs:198,272`

```csharp
var outputData = new float[(int)outputSize];
var outputData = new double[(int)outputSize];
```

**Issue:** Allocates managed arrays with `(int)` cast. If outputSize > int.MaxValue, this throws or corrupts.

**Note:** These are protected by the int.MaxValue check earlier (M5), but the cast pattern is still problematic.

---

### H8. ✅ FIXED - np.linspace Uses `int num` Parameter

**File:** `Creation/np.linspace.cs`

**Status:** Fixed - added `long num` primary overloads for all signatures, changed all loop counters from `int i` to `long i`. `int num` overloads delegate to `long num`.

---

### H9. ✅ FIXED - np.roll Uses `int shift` Parameter

**File:** `Manipulation/np.roll.cs`, `Manipulation/NDArray.roll.cs`

**Status:** Fixed - added `long shift` primary overloads for `np.roll` and `NDArray.roll`. `int shift` overloads delegate to `long shift`.

---

### H10. ✅ FIXED - UnmanagedHelper.CopyTo Uses `int countOffsetDestination`

**File:** `Backends/Unmanaged/UnmanagedHelper.cs:42,58`

**Status:** Fixed - changed parameter to `long countOffsetDestination` (also fixed typo in parameter name).

---

### H11. ✅ FIXED - np.array<T>(IEnumerable, int size) Uses `int size`

**File:** `Creation/np.array.cs`

**Status:** Fixed - added `long size` primary overload. `int size` overload delegates to `long size`.

---

### H12. ✅ FIXED - SimdMatMul.MatMulFloatSimple Uses `int M, N, K`

**File:** `Backends/Kernels/SimdMatMul.cs:71-100`

**Status:** Fixed - changed signature to `long M, N, K` and all loop counters to `long`. Removed `(int)` casts at call site.

---

### H13. ILKernelGenerator.Reduction.Arg Returns `int` Index

**File:** `Backends/Kernels/ILKernelGenerator.Reduction.Arg.cs:51,186`

```csharp
internal static unsafe int ArgMaxSimdHelper<T>(void* input, int totalSize)
internal static unsafe int ArgMinSimdHelper<T>(void* input, int totalSize)
```

**Issue:** These SIMD helpers return `int` index and take `int totalSize`. For arrays >2B elements:
- `argmax`/`argmin` would return wrong index (truncated)
- Would fail/overflow before reaching large elements

**Fix:** Change return type and parameter to `long`:
```csharp
internal static unsafe long ArgMaxSimdHelper<T>(void* input, long totalSize)
internal static unsafe long ArgMinSimdHelper<T>(void* input, long totalSize)
```

---

### H14. ✅ FIXED - Default.Dot Uses `int[]` for Dimension Expansion

**File:** `Backends/Default/Math/BLAS/Default.Dot.cs:78-92`

**Status:** Fixed - `ExpandStartDim` and `ExpandEndDim` now return `long[]` and allocate `new long[]`.

---

### H15. ✅ FIXED - NDArray.Normalize Uses `int` Loop Counters

**File:** `Extensions/NdArray.Normalize.cs:19,22`

**Status:** Fixed - loop counters changed to `long col`, `long row`.

---

### H16. ✅ FIXED - Slice.Index Uses `int` Cast in Indexing Selection

**Files:**
- `Selection/NDArray.Indexing.Selection.Getter.cs:109`
- `Selection/NDArray.Indexing.Selection.Setter.cs:126`

**Status:** Fixed - changed both files to use `ToInt64` instead of `ToInt32`.

---

### H17. ⚠️ RECLASSIFIED AS LOW - Shape.ExtractShape Uses `List<int>`

**File:** `View/Shape.cs:956`

**Status:** This method processes .NET `Array` objects where `Array.Length` and `Array.GetLength()` return `int`. The `int[]` result is required for compatibility with .NET's `Array.CreateInstance()`.

**Acceptable:** .NET boundary limitation - multi-dimensional arrays in .NET are inherently limited to int dimensions.

---

### H18. ⚠️ RECLASSIFIED AS LOW - NdArrayFromJaggedArr Uses `List<int>` for Dimensions

**File:** `Casting/NdArrayFromJaggedArr.cs:35`

**Status:** Processes .NET jagged arrays where `Array.Length` returns `int`.

**Acceptable:** .NET boundary limitation - jagged array dimensions are inherently limited to int.

---

### H19. ⚠️ RECLASSIFIED AS LOW - Arrays.GetDimensions Uses `List<int>`

**File:** `Utilities/Arrays.cs:300,339`

**Status:** Processes .NET `Array` objects where `Array.Length` returns `int`.

**Acceptable:** .NET boundary limitation - same as H17/H18.

---

### H20. ✅ FIXED - np.asarray Uses `new int[0]` for Scalar Shape

**File:** `Creation/np.asarray.cs:7,14`

**Status:** Fixed - now uses `Array.Empty<long>()` for scalar shapes.

---

### H21. ⚠️ RECLASSIFIED AS LOW - ArrayConvert Uses `int[]` for Dimensions

**File:** `Utilities/ArrayConvert.cs:43`

**Status:** Uses `Array.GetLength()` which returns `int`, and creates output via `Arrays.Create()` which uses .NET `Array.CreateInstance()`.

**Acceptable:** .NET boundary limitation - multi-dimensional arrays are int-indexed.

---

### H22. ⚠️ RECLASSIFIED AS LOW - UnmanagedStorage Uses `int[]` dim in FromMultiDimArray

**File:** `Backends/Unmanaged/UnmanagedStorage.cs:1139`

**Status:** Uses `Array.GetLength()` which returns `int`. The source is a .NET multi-dimensional array.

**Acceptable:** .NET boundary limitation - same as H17/H21.

---

### H23. ✅ FIXED - NumSharp.Bitmap Shape Casts Without Overflow Check

**File:** `NumSharp.Bitmap/np_.extensions.cs`

**Status:** Fixed - added overflow checks before casting shape dimensions to int. Throws `OverflowException` if any dimension exceeds `int.MaxValue`. Bitmap APIs inherently require `int` dimensions, so the cast is necessary but now safe.

---

## MEDIUM Priority Issues

### M1. IL Kernel Comments Reference `int*` (Documentation Drift)

**Files:**
- `ILKernelGenerator.Comparison.cs:178,316`
- `ILKernelGenerator.MixedType.cs:162`
- `ILKernelGenerator.Reduction.cs:516`
- `ILKernelGenerator.Unary.cs:346`

**Issue:** Comments describe parameter types as `int*` but code uses `long*`:

```csharp
// Comment says int* but code uses long*
// void(void* lhs, void* rhs, bool* result, long* lhsStrides, long* rhsStrides, int* shape, int ndim, long totalSize)
//                                                                               ^^^ should be long*
```

**Fix:** Update comments to reflect actual `long*` parameter types.

---

### M2. np.save/np.load Use `int[]` for Shape

**Files:**
- `APIs/np.save.cs:55,74,102,120,159,209`
- `APIs/np.load.cs:30,137,158,177,211,244,283`

**Issue:** File I/O uses `int[]` shape throughout. This is partly acceptable (the .npy file format uses int32 shapes), but internal processing should use `long[]`.

```csharp
// np.save.cs:55
int[] shape = Enumerable.Range(0, array.Rank).Select(d => array.GetLength(d)).ToArray();

// np.load.cs:30
int[] shape;
```

**Acceptable:** The .npy file format specification uses 32-bit integers for shape. These files are boundary exceptions.

---

### M3. Default.Transpose Uses `int[]` for Permutation

**File:** `Backends/Default/ArrayManipulation/Default.Transpose.cs:47,64,85,126,127`

```csharp
// Line 85 - permutation array
var dims = new int[ndims];

// Line 126-127
var permutation = new int[nd.ndim];
var reverse_permutation = new int[nd.ndim];
```

**Acceptable:** Permutation arrays are dimension-indexed (bounded by ndim ~32), not element-indexed. Using `int` is correct here.

---

### M4. Shape.InferNegativeCoordinates Has `int[]` Version

**File:** `View/Shape.cs:1292`

```csharp
public static int[] InferNegativeCoordinates(long[] dimensions, int[] coords)
```

**Issue:** Has `int[] coords` parameter but also has `long*` unsafe version. The `int[]` version should delegate to `long[]`.

---

### M5. nanmean/nanstd/nanvar Output Size Check

**Files:**
- `Statistics/np.nanmean.cs:136`
- `Statistics/np.nanstd.cs:189`
- `Statistics/np.nanvar.cs:189`

```csharp
if (outputSize > int.MaxValue)
    throw new NotSupportedException("Axis reduction with more than int.MaxValue output elements not supported");
```

**Issue:** Throws instead of supporting long output. However, this is for axis reduction where output goes into a new array - the limitation is in the downstream allocation, not the algorithm.

**Acceptable:** Until downstream allocations support >2B elements, this check prevents silent failure.

---

### M6. np.load Internal `int total` Accumulator

**File:** `APIs/np.load.cs:179`

```csharp
int total = 1;
for (int i = 0; i < shape.Length; i++)
    total *= shape[i];
```

**Issue:** Uses `int total` to accumulate product of shape dimensions. Can overflow for large arrays.

**Note:** Partially acceptable since .npy format uses int32 shapes, but internal processing should use long.

---

### M7. np.save Internal `int total` Accumulator

**File:** `APIs/np.save.cs:76`

```csharp
int total = 1;
for (int i = 0; i < shape.Length; i++)
    total *= shape[i];
```

**Issue:** Same as M6 - uses `int total` which can overflow.

---

### M8. NdArrayToJaggedArray Loops Use `int` for Large Arrays

**File:** `Casting/NdArrayToJaggedArray.cs:40-155`

```csharp
for (int i = 0; i < ret.Length; i++)
    for (int j = 0; j < ret[0].Length; j++)
```

**Issue:** Nested loops use `int` counters. For jagged arrays with many elements per dimension, this could overflow.

**Note:** Partially acceptable - jagged arrays in .NET use `int` indexing. But the iteration should use `long` internally.

---

### M9. NDArray<T> Generic Has Only `int size` Constructors

**File:** `Generics/NDArray`1.cs:83,139`

```csharp
public NDArray(int size, bool fillZeros) : base(InfoOf<TDType>.NPTypeCode, size, fillZeros)
public NDArray(int size) : base(InfoOf<TDType>.NPTypeCode, size) { }
```

**Issue:** These constructors accept `int size` only. Should add `long size` overloads as primary.

**Fix:** Add `long size` primary overloads:
```csharp
public NDArray(long size, bool fillZeros) : base(InfoOf<TDType>.NPTypeCode, size, fillZeros) { }
public NDArray(long size) : base(InfoOf<TDType>.NPTypeCode, size) { }
public NDArray(int size, bool fillZeros) : this((long)size, fillZeros) { }
public NDArray(int size) : this((long)size) { }
```

---

### M10. np.arange(int) Returns `typeof(int)` Arrays

**File:** `Creation/np.arange.cs:306,311`

```csharp
return new NDArray(typeof(int), Shape.Vector(0), false);
var nd = new NDArray(typeof(int), Shape.Vector(length), false);
```

**Issue:** The int32 overload of arange creates int32 arrays. While this matches the function signature, NumPy 2.x returns int64 for integer arange by default.

**Note:** Documented as BUG-21/Task #109. Keeping int32 for backward compatibility until explicit migration.

---

### M11. Default.Transpose Uses `int[]` for Permutation Storage

**File:** `Backends/Default/ArrayManipulation/Default.Transpose.cs:126-127`

```csharp
var permutation = new int[nd.ndim];
var reverse_permutation = new int[nd.ndim];
```

**Issue:** While ndim is bounded (max ~32), using `int[]` for permutation when dimensions are `long[]` creates a type mismatch. Technically safe but inconsistent.

**Note:** Low impact since ndim never exceeds ~32 in practice.

---

## LOW Priority Issues (Acceptable Exceptions)

### L1. Slice.ToSliceDef Throws on Large Dimensions

**File:** `View/Slice.cs:293-294`

```csharp
if (dim > int.MaxValue)
    throw new OverflowException($"Dimension {dim} exceeds int.MaxValue. SliceDef indices limited to int range.");
```

**Acceptable:** `SliceDef` uses `int` for Start/Stop/Step to match Python slice semantics. Dimensions >2B elements would require a different slicing approach.

---

### L2. Shape.GetIntDimensions/GetIntSize Throw

**File:** `View/Shape.cs:1063-1064,1081-1082`

```csharp
// GetIntDimensions
if (shape.dimensions[i] > int.MaxValue)
    throw new OverflowException($"Dimension {i} value {shape.dimensions[i]} exceeds int.MaxValue");

// GetIntSize
if (shape.Size > int.MaxValue)
    throw new OverflowException($"Shape size {shape.Size} exceeds int.MaxValue");
```

**Acceptable:** These are explicit `int` conversion methods for .NET interop. The throw is correct behavior.

---

### L3. ArraySlice.AsSpan Throws

**File:** `Backends/Unmanaged/ArraySlice<T>.cs:338-339`

```csharp
if (Count > int.MaxValue)
    throw new OverflowException($"Cannot create Span for ArraySlice with {Count} elements (exceeds int.MaxValue)");
```

**Acceptable:** `Span<T>` is int-indexed by .NET design.

---

### L4. SimdMatMul Output Size Check

**File:** `Backends/Kernels/SimdMatMul.cs:47`

```csharp
if (outputSize <= int.MaxValue)
```

**Acceptable:** This is checking whether SIMD path can be used. If output exceeds int.MaxValue, it falls back to scalar path which uses `long`.

---

### L5. NDArray.String.cs Size Checks

**File:** `Backends/NDArray.String.cs:31-32,85-86,98-99`

**Acceptable:** .NET string length is limited to `int.MaxValue`.

---

### L6. NdArrayToMultiDimArray.cs Uses int[]

**File:** `Casting/NdArrayToMultiDimArray.cs:34,42-48`

```csharp
var intShape = System.Array.ConvertAll(shape, d => (int)d);
var intIndices = new int[indices.Length];
```

**Acceptable:** .NET `Array.CreateInstance` and `Array.SetValue` require `int[]` indices.

---

### L7. Randomizer.cs Uses int

**File:** `RandomSampling/Randomizer.cs`

**Acceptable:** PRNG algorithm uses 32-bit integers internally. The `NextLong` method properly handles large ranges.

---

### L8. Enumerable.Range for Dimension Iteration

**Files:**
- `APIs/np.save.cs:55,209`
- `Backends/Default/ArrayManipulation/Default.Transpose.cs:74`

```csharp
Enumerable.Range(0, array.Rank)  // Rank is always small
Enumerable.Range(0, nd.ndim)    // ndim is always small
```

**Acceptable:** These iterate over dimensions (bounded by ~32), not elements.

---

### L9. Hashset<T>.Count is int

**File:** `Utilities/Hashset<T>.cs`

**Issue:** Custom `Hashset<T>` class has `int` count/capacity like .NET's HashSet.

**Acceptable:** Would need significant rewrite for >2B element support. Not typical use case.

---

### L10. IMemoryBlock.ItemLength is int

**File:** `Backends/Unmanaged/Interfaces/IMemoryBlock.cs:9`

```csharp
int ItemLength { get; }
```

**Acceptable:** ItemLength is sizeof(T) which is always small (max 16 for decimal).

---

### L11. SetData Calls Use `new int[0]` Instead of `long[]`

**Files:**
- `Selection/NDArray.Indexing.cs` (multiple locations)
- `Selection/NDArray.Indexing.Slicing.cs`

```csharp
SetData(values, new int[0]);
```

**Issue:** Uses `int[]` overload instead of `long[]` for empty index array.

**Acceptable:** The `int[]` overload correctly delegates to `long[]` internally. This is a minor inefficiency, not a correctness issue. Could use `Array.Empty<long>()` for slight optimization.

---

### L12. NdArrayToMultiDimArray Uses `int[]` for .NET Array.SetValue

**File:** `Casting/NdArrayToMultiDimArray.cs`

```csharp
var intIndices = new int[indices.Length];
// ... used with Array.SetValue which requires int[]
```

**Acceptable:** .NET's `Array.SetValue` API requires `int[]` indices. This is a .NET boundary limitation, documented in code comments.

---

## Checklist for Fixing

### HIGH Priority (Blocking)

- [ ] H1: Add `long[]` primary overloads to all `Get*` methods in NDArray
- [ ] H1: Add `long[]` primary overloads to all `Get*` methods in UnmanagedStorage.Getters
- [ ] H2: Add missing `long[]` overloads to typed setters (9 methods)
- [x] ~~H3: Fix `np.vstack` to use `long[]` shape~~ → Reclassified LOW (dead code)
- [x] H4: Fix `np.repeat` to use `GetInt64` and `long count` for per-element repeats ✅
- [x] H6: Fix `np.searchsorted` empty array return type to `typeof(long)` ✅
- [x] H8: Add `long num` overloads to `np.linspace` ✅
- [x] H9: Add `long shift` overloads to `np.roll` ✅
- [x] H10: Fix `UnmanagedHelper.CopyTo` offset parameter to `long` ✅
- [x] H11: Add `long size` overload to `np.array<T>(IEnumerable, size)` ✅
- [x] H12: Fix `SimdMatMul.MatMulFloatSimple` to use `long M, N, K` and `long` loop counters ✅
- [ ] H13: Fix `ArgMaxSimdHelper`/`ArgMinSimdHelper` to return `long` (complex - IL emission)
- [x] H14: Fix `Default.Dot.ExpandStartDim`/`ExpandEndDim` to return `long[]` ✅
- [x] H15: Fix `NDArray.Normalize` to use `long col`, `long row` loop counters ✅
- [x] H16: Fix `Slice.Index` calls in Selection to use `ToInt64` instead of `ToInt32` ✅
- [x] ~~H17: Fix `Shape.cs:956` to use `List<long>`~~ → Reclassified LOW (.NET boundary)
- [x] ~~H18: Fix `NdArrayFromJaggedArr` to use `List<long>`~~ → Reclassified LOW (.NET boundary)
- [x] ~~H19: Fix `Arrays.GetDimensions` to use `List<long>`~~ → Reclassified LOW (.NET boundary)
- [x] H20: Fix `np.asarray` to use `long[]` or `Array.Empty<long>()` for scalar shape ✅
- [x] ~~H21: Fix `ArrayConvert` to use `long[]`~~ → Reclassified LOW (.NET boundary)
- [x] ~~H22: Fix `UnmanagedStorage.FromMultiDimArray` to use `long[]`~~ → Reclassified LOW (.NET boundary)
- [x] H23: Add overflow checks to `NumSharp.Bitmap` shape dimension casts ✅

### MEDIUM Priority (Quality)

- [ ] M1: Update IL kernel comments to reflect `long*` parameters
- [ ] M4: Review Shape.InferNegativeCoordinates delegation pattern
- [ ] M6: Fix `np.load` internal `int total` accumulator to `long`
- [ ] M7: Fix `np.save` internal `int total` accumulator to `long`
- [ ] M9: Add `long size` primary overloads to `NDArray<T>` generic constructors
- [ ] M10: Consider migrating `np.arange(int)` to return int64 (BUG-21/Task #109)
- [ ] M11: Consider using `long[]` for transpose permutation arrays (consistency)

### LOW Priority (Cosmetic/Acceptable)

- [ ] Document all acceptable exceptions in code comments
- [ ] Add `// Acceptable: .NET boundary` comments where appropriate
- [ ] Consider `LongHashset<T>` for unique() with >2B elements (future)

---

## Verification Commands

```bash
# Search for remaining int[] shape/coords usage
grep -rn "int\[\] shape\|int\[\] coords\|int\[\] indices" src/NumSharp.Core --include="*.cs"

# Search for int loop counters over size
grep -rn "for (int.*< .*\.size" src/NumSharp.Core --include="*.cs"

# Search for (int) casts of size/offset
grep -rn "(int).*\.size\|(int).*\.Count\|(int).*offset" src/NumSharp.Core --include="*.cs"
```
