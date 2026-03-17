# Long Indexing Migration Guide

Authoritative guide for migrating NumSharp from `int` (int32) to `long` (int64) indexing. Supports arrays >2GB (int32 max = 2.1B elements).

**NumPy Reference**: NumPy uses `npy_intp` = `Py_ssize_t` (64-bit on x64).

**Performance Impact**: Benchmarked at 1-3% overhead for scalar loops, <1% for SIMD. Acceptable trade-off.

---

## Current State Summary

| Component | Status | Notes |
|-----------|--------|-------|
| `Shape` fields | ✅ Complete | `long[] dimensions`, `long[] strides`, `long size/offset/bufferSize` |
| `Slice` class | ✅ Complete | `long? Start`, `long? Stop`, `long Step` |
| `IArraySlice` | ✅ Complete | All index/count parameters are `long` |
| `UnmanagedStorage` | ✅ Complete | `long Count`, `long` index methods |
| Incrementors | ✅ Complete | `long[] Index`, `long[] dimensions` |
| IL Kernels | ✅ Complete | `long count` in delegates, `Conv_I8` in IL |
| NDArray API | ✅ Complete | `long[]` primary, `int[]` backward-compat overloads |
| Test assertions | ⚠️ Cosmetic | Use `new long[]` instead of `new[]` for shape comparisons |

---

## Core Rules

### What Uses `long`

| Item | Reason |
|------|--------|
| Array size (`Count`, `size`) | Elements can exceed 2.1B |
| Dimensions (`dimensions[]`) | Individual dimensions can exceed 2.1B |
| Strides (`strides[]`) | Byte offsets can exceed 2.1B |
| Memory offset (`offset`) | Base offset into buffer |
| Loop counters over elements | Iterating `for (long i = 0; i < size; i++)` |
| Coordinate arrays | `long[] coords` for GetOffset, iterators |
| Matrix dimensions (M, K, N) | Come from shape which is `long[]` |

### What Stays `int`

| Item | Reason |
|------|--------|
| `NDim` / `ndim` | Max ~32 dimensions, never exceeds int |
| Dimension loop indices (`d`) | `for (int d = 0; d < ndim; d++)` |
| `NPTypeCode` values | Small enum |
| Vector lane counts | Hardware-limited |
| SIMD block sizes | Cache optimization constants |
| Hash codes | .NET int by definition |

---

## Migration Patterns

### Pattern 1: Loop Counters Over Elements

```csharp
// WRONG
for (int i = 0; i < array.size; i++)
    Process(array[i]);

// CORRECT
for (long i = 0; i < array.size; i++)
    Process(array[i]);
```

### Pattern 2: Coordinate Arrays

```csharp
// WRONG
var coords = new int[2];
coords[0] = i;  // i is long

// CORRECT
var coords = new long[2];
coords[0] = i;
coords[1] = j;
shape.GetOffset(coords);
```

### Pattern 3: Matrix Dimensions

```csharp
// WRONG - defeats the purpose
int M = (int)left.shape[0];
int K = (int)left.shape[1];

// CORRECT
long M = left.shape[0];
long K = left.shape[1];
long N = right.shape[1];
```

### Pattern 4: Pointer Arithmetic (Already Works)

Pointer arithmetic natively supports `long` offsets:

```csharp
T* ptr = (T*)Address;
long offset = 3_000_000_000L;  // > int.MaxValue
T value = ptr[offset];         // Works! Pointer indexing accepts long
```

### Pattern 5: Method Signatures

Update ALL index-related parameters together:

```csharp
// BEFORE
private static void MatMulCore<T>(NDArray left, NDArray right, T* result, int M, int K, int N)

// AFTER
private static void MatMulCore<T>(NDArray left, NDArray right, T* result, long M, long K, long N)
```

### Pattern 6: Unsafe Pointer Parameters

```csharp
// BEFORE
public static unsafe bool IsContiguous(int* strides, int* shape, int ndim)

// AFTER
public static unsafe bool IsContiguous(long* strides, long* shape, int ndim)
// Note: ndim stays int (dimension count, max ~32)
```

### Pattern 7: Accumulator Variables

```csharp
// BEFORE
int expectedStride = 1;
for (int d = ndim - 1; d >= 0; d--)
    expectedStride *= shape[d];  // shape[d] is long - overflow!

// AFTER
long expectedStride = 1;
for (int d = ndim - 1; d >= 0; d--)  // d stays int
    expectedStride *= shape[d];
```

---

## API Overload Strategy

### Primary Pattern: `long[]` Primary, `int[]` Delegates

```csharp
// Primary implementation - params on long[] only
public NDArray this[params long[] indices]
{
    get => GetByIndicesInternal(indices);
}

// Backward compatible - NO params, delegates to long
public NDArray this[int[] indices]
{
    get
    {
        Span<long> longIndices = indices.Length <= 8
            ? stackalloc long[indices.Length]
            : new long[indices.Length];

        for (int i = 0; i < indices.Length; i++)
            longIndices[i] = indices[i];

        return GetByIndicesInternal(longIndices);
    }
}
```

### Shape Constructor Pattern

```csharp
// Primary constructor - long[]
public Shape(params long[] dims)
{
    this.dimensions = dims;
    // ...
}

// Backward compatible - int[] delegates
public Shape(int[] dims) : this(ComputeLongShape(dims)) { }

// Conversion helper
[MethodImpl(MethodImplOptions.AggressiveInlining)]
public static long[] ComputeLongShape(int[] dimensions)
{
    if (dimensions == null) return null;
    var result = new long[dimensions.Length];
    for (int i = 0; i < dimensions.Length; i++)
        result[i] = dimensions[i];
    return result;
}
```

### Single Index Pattern

```csharp
// Zero-cost implicit conversion
public T this[int index] => this[(long)index];
public T this[long index] { get; set; }  // Primary implementation
```

---

## IL Kernel Emission

### Delegate Signatures

All kernel delegates use `long count`:

```csharp
public unsafe delegate void ContiguousKernel<T>(T* lhs, T* rhs, T* result, long count);
public unsafe delegate T SimpleReductionKernel<T>(T* input, long count);
public unsafe delegate void ShiftScalarKernel<T>(T* input, T* output, int shiftAmount, long count);
```

### IL Emission Patterns

| Purpose | IL Instruction |
|---------|---------------|
| Load long constant | `Ldc_I8` |
| Load 0 as long | `Ldc_I4_0` + `Conv_I8` |
| Convert to long | `Conv_I8` |
| Declare long local | `il.DeclareLocal(typeof(long))` |

```csharp
// BEFORE
il.Emit(OpCodes.Ldc_I4, value);
il.DeclareLocal(typeof(int));

// AFTER
il.Emit(OpCodes.Ldc_I8, (long)value);
il.DeclareLocal(typeof(long));

// Loading 0 as long
il.Emit(OpCodes.Ldc_I4_0);
il.Emit(OpCodes.Conv_I8);
```

---

## Acceptable .NET Boundary Exceptions

These locations legitimately use `int` due to .NET API constraints:

| Location | Reason |
|----------|--------|
| `NDArray.String.cs` | .NET string length limited to `int.MaxValue` |
| `ArraySlice.AsSpan<T>()` | `Span<T>` is int-indexed by .NET design |
| `Array.CreateInstance()` | .NET arrays limited to int indexing |
| `NdArrayToMultiDimArray.cs` | `Array.SetValue` requires `int[]` indices |
| `np.load.cs` | .npy file format uses int32 shapes |
| `Hashset.cs` | Hash codes are int by definition |

### Pattern for .NET Boundaries

```csharp
// Span creation - check and throw
public Span<T> AsSpan<T>()
{
    if (Count > int.MaxValue)
        throw new InvalidOperationException(
            "Storage size exceeds Span<T> maximum. Use pointer access.");
    return new Span<T>(Address, (int)Count);
}

// String conversion - check at boundary
if (arr.size > int.MaxValue)
    throw new InvalidOperationException(
        "Array size exceeds string length limit.");
return new string((char*)arr.Address, 0, (int)arr.size);
```

---

## LongIndexBuffer for Dynamic Collections

Replace `List<long>` with `LongIndexBuffer` for collecting indices (supports >2B elements):

```csharp
// BEFORE - limited to int.MaxValue capacity
var indices = new List<long>();
for (long i = 0; i < size; i++)
    if (condition) indices.Add(i);

// AFTER - supports long capacity
var buffer = new LongIndexBuffer(initialCapacity: 1024);
try
{
    for (long i = 0; i < size; i++)
        if (condition) buffer.Add(i);

    return buffer.ToNDArray();
}
finally
{
    buffer.Dispose();
}
```

**Location**: `Backends/Kernels/LongIndexBuffer.cs`

**Used by**: `Default.NonZero.cs`, `ILKernelGenerator.Masking.cs`, `ILKernelGenerator.Reduction.Axis.Simd.cs`

---

## File Migration Checklist

When migrating a file:

1. [ ] Find all `int` variables related to indices/sizes/strides/offsets
2. [ ] Change to `long` unless it's ndim/axis/dimension-loop
3. [ ] Update method signatures if parameters are index-related
4. [ ] Update all callers of changed methods
5. [ ] Check loop counters iterating over array elements → `long`
6. [ ] Check coordinate arrays → `long[]`
7. [ ] Check pointer params → `long*` for strides/shapes
8. [ ] Add overflow checks where .NET APIs require `int`
9. [ ] Document exceptions with comments

---

## Common Compilation Errors

### Cannot convert long to int

```
error CS0266: Cannot implicitly convert type 'long' to 'int'
```

**Fix**: Change receiving variable to `long`, OR add explicit cast with overflow check if external API requires `int`.

### Argument type mismatch

```
error CS1503: Argument 1: cannot convert from 'int[]' to 'long[]'
```

**Fix**: Change array declaration to `long[]`, or use `Shape.ComputeLongShape()`.

### params conflict

```
error CS0231: A params parameter must be the last parameter
```

**Fix**: Only use `params` on `long[]` overloads, not on `int[]` backward-compat overloads.

---

## Testing

### Run Tests

```bash
cd test/NumSharp.UnitTest
dotnet test --no-build -- --treenode-filter "/*/*/*/*[Category!=OpenBugs]"
```

### Test Assertion Style

```csharp
// Explicit long[] for shape comparisons
result.shape.Should().BeEquivalentTo(new long[] { 2, 3 });

// NOT: new[] { 2, 3 } - compiler infers int[]
```

---

## Git Commit Convention

```
int64 indexing: <component> <what changed>

- <specific change 1>
- <specific change 2>
```

Example:
```
int64 indexing: SimdMatMul matrix dimensions to long

- M, N, K parameters changed from int to long
- Loop counters remain int for cache block iteration
- Fixed pointer arithmetic for large matrices
```

---

## Quick Reference Table

| Old | New | Notes |
|-----|-----|-------|
| `int size` | `long size` | Array/storage size |
| `int offset` | `long offset` | Memory offset |
| `int[] dimensions` | `long[] dimensions` | Shape dimensions |
| `int[] strides` | `long[] strides` | Memory strides |
| `int[] coords` | `long[] coords` | Index coordinates |
| `int* shape` | `long* shape` | Unsafe pointer |
| `int* strides` | `long* strides` | Unsafe pointer |
| `for (int i` | `for (long i` | Element iteration |
| `int M, K, N` | `long M, K, N` | Matrix dimensions |
| `Ldc_I4` | `Ldc_I8` | IL: load constant |
| `typeof(int)` | `typeof(long)` | IL: local type |
| `List<long>` | `LongIndexBuffer` | Dynamic index collection |
| `int ndim` | `int ndim` | **KEEP** - dimension count |
| `int d` | `int d` | **KEEP** - dimension loop |
| `int axis` | `int axis` | **KEEP** - axis parameter |

---

## Algorithm Migration Patterns

### Pattern 8: LongRange Helper (Replace Enumerable.Range)

`Enumerable.Range(0, count)` is limited to `int.MaxValue`. Use `LongRange`:

```csharp
// BEFORE - limited to int.MaxValue
var indices = Enumerable.Range(0, (int)size).Select(i => ...);

// AFTER - supports long
private static IEnumerable<long> LongRange(long count)
{
    for (long i = 0; i < count; i++)
        yield return i;
}

var indices = LongRange(size).Select(i => ...);
```

**Used in**: `ndarray.argsort.cs`

### Pattern 9: SIMD Block Loops (Mixed int/long)

For cache-blocked algorithms, outer loops use `long`, inner block loops use `int`:

```csharp
// Outer loops iterate over full matrix - use long
for (long k0 = 0; k0 < K; k0 += KC)
    for (long i0 = 0; i0 < M; i0 += MC)
        for (long jp = 0; jp < N; jp += NR)
        {
            // Inner block loops bounded by small constants (MC, KC, MR, NR) - int is OK
            for (int ip = 0; ip < Math.Min(MC, M - i0); ip += MR)
                for (int k = 0; k < kc; k++)
                    // ...
        }
```

**Used in**: `SimdMatMul.cs`, `Default.MatMul.2D2D.cs`

### Pattern 10: Random Sampling (NextLong)

Random sampling methods use `long` bounds and delegate from `int` overloads:

```csharp
// Primary implementation
public NDArray choice(long a, Shape shape = default, ...)
{
    NDArray idx = randint(0, a, shape);  // randint accepts long
    return idx;
}

// Backward compatible - delegates
public NDArray choice(int a, Shape shape = default, ...)
{
    return choice((long)a, shape, ...);
}

// Shuffle uses long size and NextLong
var size = x.size;  // long
addr_swap = addr + transformOffset(randomizer.NextLong(size));
```

---

## Return Type Changes

Functions that return indices now return `long` (NumPy uses `int64` for indices):

| Function | Old Return | New Return |
|----------|-----------|------------|
| `np.nonzero()` | `NDArray[]` of int32 | `NDArray[]` of **int64** |
| `nd.argsort()` | `NDArray` of int32 | `NDArray` of **int64** |
| `np.argmax()` | `int` | `long` |
| `np.argmin()` | `int` | `long` |

```csharp
// np.nonzero returns int64 indices
var result = np.nonzero(array);
Assert.That(result[0].typecode).IsEqualTo(NPTypeCode.Int64);

// argsort returns int64 indices
var sorted = array.argsort<float>();
Assert.That(sorted.typecode).IsEqualTo(NPTypeCode.Int64);
```

---

## NDArray Accessor Methods

All element access methods use `long` indices:

```csharp
// Get single element by flat index
public ValueType GetAtIndex(long index);
public T GetAtIndex<T>(long index) where T : unmanaged;

// Set single element by flat index
public void SetAtIndex(object obj, long index);
public void SetAtIndex<T>(T value, long index) where T : unmanaged;

// Typed getters/setters (all have long index)
public int GetInt32(long index);
public void SetInt32(int value, long index);
// ... same for all 12 dtypes
```

---

## Parallel.For Removal

Axis reduction operations removed `Parallel.For` in favor of single-threaded execution with `long` indices:

**Affected files:**
- `ILKernelGenerator.Reduction.Axis.Simd.cs`
- `ILKernelGenerator.Reduction.Axis.VarStd.cs`

**Rationale**: `Parallel.For` uses `int` indices. Rather than add complex chunking logic for >2B elements, axis reductions now run single-threaded with proper `long` iteration. Performance impact is minimal since SIMD vectorization provides the main speedup.

---

## Files Changed Summary (38 commits)

### Core Types (Phase 1)
- `Shape.cs` - `long[] dimensions/strides`, `long size/offset/bufferSize`
- `Slice.cs` - `long? Start/Stop`, `long Step`
- `IArraySlice.cs` - All `long` index parameters
- `ArraySlice.cs`, `ArraySlice<T>.cs` - `long Count`, `long` methods
- `UnmanagedStorage.*.cs` - `long Count`, `long` getters/setters

### Incrementors
- `NDCoordinatesIncrementor.cs` - `long[] Index`, `long[] dimensions`
- `NDCoordinatesAxisIncrementor.cs` - `long[]` coords
- `NDOffsetIncrementor.cs` - `long offset`
- `ValueOffsetIncrementor.cs` - `long offset`

### IL Kernels
- `ILKernelGenerator.*.cs` (13 files) - `typeof(long)`, `Conv_I8`, `Ldc_I8`
- `KernelSignatures.cs` - `long count` delegates
- `SimdMatMul.cs` - `long M, K, N`
- `LongIndexBuffer.cs` - New helper for dynamic index collection

### DefaultEngine Operations
- `Default.Dot.NDMD.cs` - `long` loop counters
- `Default.MatMul.2D2D.cs` - `long M, K, N`, removed int.MaxValue check
- `Default.NonZero.cs` - `LongIndexBuffer`, returns `int64`
- `Default.Reduction.*.cs` - `long` index tracking

### API Functions
- `np.random.*.cs` - `NextLong`, `long` overloads
- `ndarray.argsort.cs` - `LongRange`, `long[]` throughout
- `np.nanmean/std/var.cs` - `long` count tracking

### Tests Added
- `NonzeroInt64Tests.cs` - Verifies int64 return type
- `MatMulInt64Tests.cs` - Large matrix support
- `ArgsortInt64Tests.cs` - Verifies int64 indices

---

## References

- NumPy `npy_intp`: `numpy/_core/include/numpy/npy_common.h:217`
- NumPy uses `Py_ssize_t` = 64-bit on x64 platforms
- .NET `nint`/`nuint` are platform-dependent (NumPy's approach)
