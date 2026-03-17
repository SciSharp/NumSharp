# Long Indexing Migration - Remaining Issues

Issues discovered by auditing the codebase against the int64 migration spirit.

**Audit Date**: Based on commit `111b4076` (longindexing branch)
**Updated**: 2026-03-17 - Added H12-H22, M9-M11 from comprehensive code search
**Updated**: 2026-03-17 - Added H23, L11-L12 from post-rebase diff scan

---

## Summary

| Priority | Count | Category |
|----------|-------|----------|
| HIGH | 23 | Missing long overloads, int parameters/variables |
| MEDIUM | 11 | IL kernel comments, internal int usage |
| LOW | 13 | Acceptable .NET boundary exceptions |

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

### H3. np.vstack Uses `int[]` for Shape

**File:** `Manipulation/np.vstack.cs:26,30`

```csharp
// Line 26 - uses int[] literal
np.Storage.Reshape(new int[] { nps.Length, nps[0].shape[0] });

// Line 30 - uses int[] variable
int[] shapes = nps[0].shape;  // shape property returns int[] for backward compat
```

**Fix:** Use `long[]` literal:
```csharp
np.Storage.Reshape(new long[] { nps.Length, nps[0].shape[0] });
```

---

### H4. np.repeat Uses `int count` for Per-Element Repeat

**File:** `Manipulation/np.repeat.cs:69,159,161`

```csharp
// Line 69 - calculates total size with int count
int count = repeatsFlat.GetInt32(i);  // Should be GetInt64

// Line 159 - same issue
int count = repeatsFlat.GetInt32(i);

// Line 161 - inner loop uses int
for (int j = 0; j < count; j++)  // Should be long j
```

**Issue:** If repeat count exceeds int.MaxValue for any element, this will fail or overflow.

**Fix:**
```csharp
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

### H6. np.searchsorted Empty Array Returns Wrong Type

**File:** `Sorting_Searching_Counting/np.searchsorted.cs:48`

```csharp
if (v.size == 0)
    return new NDArray(typeof(int), Shape.Vector(0), false);
//                     ^^^^^^^^^^^ Should be typeof(long)
```

**Issue:** Returns int32 type for empty input, but int64 for non-empty input. Inconsistent.

**Fix:**
```csharp
return new NDArray(typeof(long), Shape.Vector(0), false);
```

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

### H8. np.linspace Uses `int num` Parameter

**File:** `Creation/np.linspace.cs:21,37,54,71`

```csharp
public static NDArray linspace(double start, double stop, int num, ...)
```

**Issue:** All `linspace` overloads take `int num` parameter. No `long` overloads exist.

Also the internal loops use `int i`:
```csharp
for (int i = 0; i < num; i++) addr[i] = ...
```

**Fix:** Add `long num` overloads with `long` loop counters.

---

### H9. np.roll Uses `int shift` Parameter

**File:** `Manipulation/np.roll.cs:21`, `Manipulation/NDArray.roll.cs:16,28`

```csharp
public static NDArray roll(NDArray a, int shift, int? axis = null)
```

**Issue:** Shift amount is `int`, limiting roll distance for very large arrays.

**Fix:** Add `long shift` primary overload.

---

### H10. UnmanagedHelper.CopyTo Uses `int countOffsetDestination`

**File:** `Backends/Unmanaged/UnmanagedHelper.cs:42,58`

```csharp
public static unsafe void CopyTo(this IMemoryBlock src, IMemoryBlock dst, int countOffsetDesitinion)
public static unsafe void CopyTo(this IMemoryBlock src, void* dstAddress, int countOffsetDesitinion)
```

**Issue:** Offset parameter is `int`, limiting copy destination offset.

**Fix:** Change to `long countOffsetDestination`.

---

### H11. np.array<T>(IEnumerable, int size) Uses `int size`

**File:** `Creation/np.array.cs:105`

```csharp
public static NDArray array<T>(IEnumerable<T> data, int size) where T : unmanaged
```

**Issue:** Size hint parameter is `int`, limiting pre-allocated size.

**Fix:** Add `long size` overload.

---

### H12. SimdMatMul.MatMulFloatSimple Uses `int M, N, K`

**File:** `Backends/Kernels/SimdMatMul.cs:71-100`

```csharp
private static unsafe void MatMulFloatSimple(float* A, float* B, float* C, int M, int N, int K)
{
    for (int i = 0; i < M; i++)    // Should be long
    for (int k = 0; k < K; k++)    // Should be long
    for (int j = 0; j < N; j++)    // Should be long
```

**Issue:** While the main `MatMulFloat` method uses `long M, N, K`, the "simple" path for small matrices casts to `int` and uses `int` loop counters. This caps the simple path at matrices with int.MaxValue dimensions even though the outer API promises long support.

**Fix:** Change signature and loop counters to `long`:
```csharp
private static unsafe void MatMulFloatSimple(float* A, float* B, float* C, long M, long N, long K)
```

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

### H14. Default.Dot Uses `int[]` for Dimension Expansion

**File:** `Backends/Default/Math/BLAS/Default.Dot.cs:78-92`

```csharp
private static int[] ExpandStartDim(Shape shape)
{
    var ret = new int[shape.NDim + 1];  // Should be long[]
    ret[0] = 1;
    Array.Copy(shape.dimensions, 0, ret, 1, shape.NDim);  // dimensions is long[]
    return ret;
}
```

**Issue:** Returns `int[]` but copies from `shape.dimensions` which is `long[]`. Truncates dimensions >int.MaxValue.

**Fix:** Change return type to `long[]`:
```csharp
private static long[] ExpandStartDim(Shape shape)
{
    var ret = new long[shape.NDim + 1];
    // ...
}
```

---

### H15. NDArray.Normalize Uses `int` Loop Counters

**File:** `Extensions/NdArray.Normalize.cs:19,22`

```csharp
for (int col = 0; col < shape[1]; col++)  // shape[1] is long
    for (int row = 0; row < shape[0]; row++)  // shape[0] is long
```

**Issue:** Loop counters are `int` but iterate up to `shape[i]` which is `long`. Fails for arrays with >2B elements per dimension.

**Fix:** Change to `long` loop counters:
```csharp
for (long col = 0; col < shape[1]; col++)
    for (long row = 0; row < shape[0]; row++)
```

---

### H16. Slice.Index Uses `int` Cast in Indexing Selection

**Files:**
- `Selection/NDArray.Indexing.Selection.Getter.cs:109`
- `Selection/NDArray.Indexing.Selection.Setter.cs:126`

```csharp
case IConvertible o: return Slice.Index((int)o.ToInt32(CultureInfo.InvariantCulture));
```

**Issue:** Casts to `int` when `Slice.Index` now accepts `long`. This truncates large indices.

**Fix:** Use `ToInt64`:
```csharp
case IConvertible o: return Slice.Index(o.ToInt64(CultureInfo.InvariantCulture));
```

---

### H17. Shape Dimension Parsing Uses `List<int>`

**File:** `View/Shape.cs:956`

```csharp
var l = new List<int>(16);
```

**Issue:** Uses `List<int>` when accumulating dimension values that should be `long`.

**Fix:** Change to `List<long>`:
```csharp
var l = new List<long>(16);
```

---

### H18. NdArrayFromJaggedArr Uses `List<int>` for Dimensions

**File:** `Casting/NdArrayFromJaggedArr.cs:35`

```csharp
var dimList = new List<int>();
```

**Issue:** Collects dimension sizes in `List<int>` - truncates dimensions >int.MaxValue.

**Fix:** Change to `List<long>`:
```csharp
var dimList = new List<long>();
```

---

### H19. Arrays.GetDimensions Uses `List<int>`

**File:** `Utilities/Arrays.cs:300,339`

```csharp
var dimList = new List<int>(16);
```

**Issue:** Same as H18 - should be `List<long>`.

**Fix:** Change to `List<long>`:
```csharp
var dimList = new List<long>(16);
```

---

### H20. np.asarray Uses `new int[0]` for Scalar Shape

**File:** `Creation/np.asarray.cs:7,14`

```csharp
var nd = new NDArray(typeof(string), new int[0]);
var nd = new NDArray(typeof(T), new int[0]);
```

**Issue:** Uses `int[]` for empty shape when should use `long[]`.

**Fix:** Use `Array.Empty<long>()` or `new long[0]`:
```csharp
var nd = new NDArray(typeof(string), Array.Empty<long>());
```

---

### H21. ArrayConvert Uses `int[]` for Dimensions

**File:** `Utilities/ArrayConvert.cs:43`

```csharp
int[] dimensions = new int[dims];
```

**Issue:** Allocates `int[]` for dimensions when should be `long[]`.

**Fix:** Change to `long[]`:
```csharp
long[] dimensions = new long[dims];
```

---

### H22. UnmanagedStorage Uses `int[]` dim in FromMultiDimArray

**File:** `Backends/Unmanaged/UnmanagedStorage.cs:1139`

```csharp
int[] dim = new int[values.Rank];
```

**Issue:** Copies dimensions from multi-dim array into `int[]`.

**Fix:** Change to `long[]`:
```csharp
long[] dim = new long[values.Rank];
```

---

### H23. NumSharp.Bitmap Shape Casts Without Overflow Check

**File:** `NumSharp.Bitmap/np_.extensions.cs:10,20,21`

```csharp
var bbp = (int)nd.shape[3]; //bytes per pixel
var height = (int)nd.shape[1];
var width = (int)nd.shape[2];
```

**Issue:** Casts shape dimensions directly to `int` without checking for overflow. If any dimension exceeds `int.MaxValue`, this silently truncates.

**Fix:** Add overflow checks:
```csharp
if (nd.shape[3] > int.MaxValue || nd.shape[1] > int.MaxValue || nd.shape[2] > int.MaxValue)
    throw new OverflowException("Bitmap dimensions exceed int.MaxValue");
var bbp = (int)nd.shape[3];
var height = (int)nd.shape[1];
var width = (int)nd.shape[2];
```

**Note:** Bitmap APIs inherently use `int` dimensions, so the cast is necessary, but the overflow check prevents silent corruption.

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
- [ ] H3: Fix `np.vstack` to use `long[]` shape
- [ ] H4: Fix `np.repeat` to use `GetInt64` and `long count` for per-element repeats
- [ ] H6: Fix `np.searchsorted` empty array return type to `typeof(long)`
- [ ] H8: Add `long num` overloads to `np.linspace`
- [ ] H9: Add `long shift` overloads to `np.roll`
- [ ] H10: Fix `UnmanagedHelper.CopyTo` offset parameter to `long`
- [ ] H11: Add `long size` overload to `np.array<T>(IEnumerable, size)`
- [ ] H12: Fix `SimdMatMul.MatMulFloatSimple` to use `long M, N, K` and `long` loop counters
- [ ] H13: Fix `ArgMaxSimdHelper`/`ArgMinSimdHelper` to return `long` and accept `long totalSize`
- [ ] H14: Fix `Default.Dot.ExpandStartDim`/`ExpandEndDim` to return `long[]`
- [ ] H15: Fix `NDArray.Normalize` to use `long col`, `long row` loop counters
- [ ] H16: Fix `Slice.Index` calls in Selection to use `ToInt64` instead of `ToInt32`
- [ ] H17: Fix `Shape.cs:956` to use `List<long>` for dimension parsing
- [ ] H18: Fix `NdArrayFromJaggedArr` to use `List<long>` for dimensions
- [ ] H19: Fix `Arrays.GetDimensions` to use `List<long>`
- [ ] H20: Fix `np.asarray` to use `long[]` or `Array.Empty<long>()` for scalar shape
- [ ] H21: Fix `ArrayConvert` to use `long[]` for dimensions
- [ ] H22: Fix `UnmanagedStorage.FromMultiDimArray` to use `long[]` dim
- [ ] H23: Add overflow checks to `NumSharp.Bitmap` shape dimension casts

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
