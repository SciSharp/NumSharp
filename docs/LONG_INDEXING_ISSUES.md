# Long Indexing Migration - Remaining Issues

Issues discovered by auditing the codebase against the int64 migration spirit.

**Audit Date**: Based on commit `111b4076` (longindexing branch)

---

## Summary

| Priority | Count | Category |
|----------|-------|----------|
| HIGH | 7 | Missing long[] overloads, int variables |
| MEDIUM | 6 | IL kernel comments, internal int usage |
| LOW | 10 | Acceptable .NET boundary exceptions |

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

## Checklist for Fixing

### HIGH Priority (Blocking)

- [ ] Add `long[]` primary overloads to all `Get*` methods in NDArray
- [ ] Add `long[]` primary overloads to all `Get*` methods in UnmanagedStorage.Getters
- [ ] Add missing `long[]` overloads to typed setters (9 methods)
- [ ] Fix `np.vstack` to use `long[]` shape
- [ ] Fix `np.repeat` to use `GetInt64` and `long count` for per-element repeats
- [ ] Fix `np.searchsorted` empty array return type to `typeof(long)`

### MEDIUM Priority (Quality)

- [ ] Update IL kernel comments to reflect `long*` parameters
- [ ] Review Shape.InferNegativeCoordinates delegation pattern
- [ ] Fix `np.load` internal `int total` accumulator to `long`

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
