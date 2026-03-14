# Int64 Index Migration Plan

Migration from `int` (int32) to `long` (int64) for all index, stride, offset, and size operations.

**Rationale**: Support arrays >2GB (int32 max = 2.1B elements). NumPy uses `npy_intp` = `Py_ssize_t` (64-bit on x64).

**Performance Impact**: Benchmarked at 1-3% overhead for scalar loops, <1% for SIMD loops. Acceptable.

---

## Conversion Strategy: int ↔ long Handling

### C# Conversion Rules

| Conversion | Type | Notes |
|------------|------|-------|
| `int` → `long` | **Implicit** | Always safe, zero cost |
| `long` → `int` | **Explicit** | Requires cast, may overflow |
| `int[]` → `long[]` | **Manual** | Element-by-element conversion required |
| `long[]` → `int[]` | **Manual** | Element-by-element + overflow check |

### Why Pointer Arithmetic Works

NumSharp uses unmanaged memory (`byte*`), not managed arrays. Pointer arithmetic natively supports `long` offsets:

```csharp
byte* ptr = baseAddress;
long largeOffset = 3_000_000_000L;  // > int.MaxValue
byte* result = ptr + largeOffset;   // WORKS! No cast needed
```

This is why we can migrate internally to `long` without breaking memory access.

### Public API Strategy: Dual Overloads

Keep `int` overloads for backward compatibility, delegate to `long` internally:

```csharp
// Single index - int delegates to long (zero-cost implicit conversion)
public T this[int index] => this[(long)index];
public T this[long index] { get; set; }  // Main implementation

// Multi-index - int[] converts to long[] with stackalloc optimization
public NDArray this[params int[] indices]
{
    get
    {
        // Stack alloc for common case (<=8 dims), heap for rare large case
        Span<long> longIndices = indices.Length <= 8
            ? stackalloc long[indices.Length]
            : new long[indices.Length];

        for (int i = 0; i < indices.Length; i++)
            longIndices[i] = indices[i];

        return GetByIndicesInternal(longIndices);
    }
}

public NDArray this[params long[] indices]  // Main implementation
{
    get => GetByIndicesInternal(indices);
}
```

### Shape Constructor Overloads

```csharp
// Backward compatible - accept int[]
public Shape(params int[] dims) : this(ToLongArray(dims)) { }

// New primary constructor - long[]
public Shape(params long[] dims)
{
    this.dimensions = dims;
    // ... rest of initialization
}

[MethodImpl(MethodImplOptions.AggressiveInlining)]
private static long[] ToLongArray(int[] arr)
{
    var result = new long[arr.Length];
    for (int i = 0; i < arr.Length; i++)
        result[i] = arr[i];
    return result;
}
```

### Backward Compatible Properties

```csharp
// Keep int[] for backward compat, throw on overflow
public int[] shape
{
    get
    {
        var dims = Shape.dimensions;  // Internal long[]
        var result = new int[dims.Length];
        for (int i = 0; i < dims.Length; i++)
        {
            if (dims[i] > int.MaxValue)
                throw new OverflowException(
                    $"Dimension {i} size {dims[i]} exceeds int.MaxValue. Use shapeLong property.");
            result[i] = (int)dims[i];
        }
        return result;
    }
}

// New property for large arrays
public long[] shapeLong => Shape.dimensions;

// size - same pattern
public int size => Size > int.MaxValue
    ? throw new OverflowException("Array size exceeds int.MaxValue. Use sizeLong.")
    : (int)Size;

public long sizeLong => Size;  // New property
```

### What Stays int (No Change Needed)

| Member | Reason |
|--------|--------|
| `NDim` / `ndim` | Max ~32 dimensions, never exceeds int |
| `Slice.Start/Stop/Step` | Python slice semantics use int |
| Loop counters in IL (where safe) | JIT optimizes better |
| `NPTypeCode` enum values | Small fixed set |

### Conversion Helper Methods

```csharp
internal static class IndexConvert
{
    /// <summary>Throws if value exceeds int range.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int ToIntChecked(long value)
    {
        if (value > int.MaxValue || value < int.MinValue)
            throw new OverflowException($"Value {value} exceeds int range");
        return (int)value;
    }

    /// <summary>Converts long[] to int[], throws on overflow.</summary>
    public static int[] ToIntArrayChecked(long[] arr)
    {
        var result = new int[arr.Length];
        for (int i = 0; i < arr.Length; i++)
        {
            if (arr[i] > int.MaxValue)
                throw new OverflowException($"Index {i} value {arr[i]} exceeds int.MaxValue");
            result[i] = (int)arr[i];
        }
        return result;
    }

    /// <summary>Converts int[] to long[] (always safe).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static long[] ToLongArray(int[] arr)
    {
        var result = new long[arr.Length];
        for (int i = 0; i < arr.Length; i++)
            result[i] = arr[i];
        return result;
    }

    /// <summary>Converts int[] to Span&lt;long&gt; using stackalloc.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Span<long> ToLongSpan(int[] arr, Span<long> buffer)
    {
        for (int i = 0; i < arr.Length; i++)
            buffer[i] = arr[i];
        return buffer.Slice(0, arr.Length);
    }
}
```

### IL Kernel Considerations

For IL-generated kernels, loop counters can often stay `int` when:
- Array size is guaranteed < int.MaxValue (checked at call site)
- Counter is only used for iteration, not offset calculation

Offset calculations must use `long`:
```csharp
// Before: int offset = baseOffset + i * stride;
// After:  long offset = baseOffset + (long)i * stride;
```

---

## Phase 1: Core Types (CRITICAL PATH)

These changes cascade to everything else. Must be done atomically.

### 1.1 Shape Struct (`View/Shape.cs`)

| Current | Change To | Lines |
|---------|-----------|-------|
| `internal readonly int size` | `long size` | 208 |
| `internal readonly int[] dimensions` | `long[] dimensions` | 209 |
| `internal readonly int[] strides` | `long[] strides` | 210 |
| `internal readonly int bufferSize` | `long bufferSize` | 218 |
| `internal readonly int offset` | `long offset` | 225 |
| `public readonly int OriginalSize` | `long OriginalSize` | 295 |
| `public readonly int NDim` | `int NDim` | 359 (KEEP int - max 32 dims) |
| `public readonly int Size` | `long Size` | 380 |
| `public readonly int Offset` | `long Offset` | 391 |
| `public readonly int BufferSize` | `long BufferSize` | 402 |
| `public readonly int this[int dim]` | `long this[int dim]` | 565 |
| `public readonly int TransformOffset(int offset)` | `long TransformOffset(long offset)` | 581 |
| `public readonly int GetOffset(params int[] indices)` | `long GetOffset(params long[] indices)` | 598 |
| `public readonly int[] GetCoordinates(int offset)` | `long[] GetCoordinates(long offset)` | 755 |

**Related files**:
- `View/Shape.Unmanaged.cs` - unsafe pointer versions of GetOffset, GetSubshape
- `View/Shape.Reshaping.cs` - reshape operations
- `View/Slice.cs` - Start, Stop, Step should stay int (Python slice semantics)
- `View/SliceDef.cs` - may need long for large dimension slicing

### 1.2 IArraySlice Interface (`Backends/Unmanaged/Interfaces/IArraySlice.cs`)

| Current | Change To |
|---------|-----------|
| `T GetIndex<T>(int index)` | `T GetIndex<T>(long index)` |
| `object GetIndex(int index)` | `object GetIndex(long index)` |
| `void SetIndex<T>(int index, T value)` | `void SetIndex<T>(long index, T value)` |
| `void SetIndex(int index, object value)` | `void SetIndex(long index, object value)` |
| `object this[int index]` | `object this[long index]` |
| `IArraySlice Slice(int start)` | `IArraySlice Slice(long start)` |
| `IArraySlice Slice(int start, int count)` | `IArraySlice Slice(long start, long count)` |

### 1.3 ArraySlice Implementation (`Backends/Unmanaged/ArraySlice.cs`, `ArraySlice`1.cs`)

All index/count parameters and `Count` property → `long`

### 1.4 IMemoryBlock Interface (`Backends/Unmanaged/Interfaces/IMemoryBlock.cs`)

| Current | Change To |
|---------|-----------|
| `int Count` | `long Count` |

### 1.5 UnmanagedStorage (`Backends/Unmanaged/UnmanagedStorage.cs`)

| Current | Change To | Line |
|---------|-----------|------|
| `public int Count` | `public long Count` | 47 |

**Related files** (same changes):
- `UnmanagedStorage.Getters.cs` - index parameters
- `UnmanagedStorage.Setters.cs` - index parameters
- `UnmanagedStorage.Slicing.cs` - slice parameters
- `UnmanagedStorage.Cloning.cs` - count parameters

---

## Phase 2: NDArray Public API

### 2.1 NDArray Core (`Backends/NDArray.cs`)

| Property/Method | Change |
|-----------------|--------|
| `int size` | `long size` |
| `int[] shape` | Keep `int[]` for API compat OR migrate to `long[]` |
| `int ndim` | Keep `int` (max 32 dimensions) |
| `int[] strides` | `long[] strides` |

### 2.2 NDArray Indexing (`Selection/NDArray.Indexing.cs`)

| Current | Change To |
|---------|-----------|
| `NDArray this[int* dims, int ndims]` | `NDArray this[long* dims, int ndims]` |
| All coordinate arrays | `int[]` → `long[]` |

**Related files**:
- `NDArray.Indexing.Selection.cs`
- `NDArray.Indexing.Selection.Getter.cs`
- `NDArray.Indexing.Selection.Setter.cs`
- `NDArray.Indexing.Masking.cs`

### 2.3 Generic NDArray (`Generics/NDArray`1.cs`)

| Current | Change To |
|---------|-----------|
| `NDArray(int size, bool fillZeros)` | `NDArray(long size, bool fillZeros)` |
| `NDArray(int size)` | `NDArray(long size)` |

---

## Phase 3: Iterators

### 3.1 NDIterator (`Backends/Iterators/NDIterator.cs`)

| Current | Change To |
|---------|-----------|
| `Func<int[], int> getOffset` | `Func<long[], long> getOffset` |
| Internal index tracking | `int` → `long` |

**Files** (12 type-specific generated files):
- `NDIterator.template.cs`
- `NDIteratorCasts/NDIterator.Cast.*.cs` (Boolean, Byte, Char, Decimal, Double, Int16, Int32, Int64, Single, UInt16, UInt32, UInt64)

### 3.2 MultiIterator (`Backends/Iterators/MultiIterator.cs`)

Same changes as NDIterator.

### 3.3 Incrementors (`Utilities/Incrementors/`)

| File | Changes |
|------|---------|
| `NDCoordinatesIncrementor.cs` | coords `int[]` → `long[]` |
| `NDCoordinatesAxisIncrementor.cs` | coords `int[]` → `long[]` |
| `NDCoordinatesLeftToAxisIncrementor.cs` | coords `int[]` → `long[]` |
| `NDExtendedCoordinatesIncrementor.cs` | coords `int[]` → `long[]` |
| `NDOffsetIncrementor.cs` | offset `int` → `long` |
| `ValueOffsetIncrementor.cs` | offset `int` → `long` |

---

## Phase 4: IL Kernel Generator (924 occurrences)

### 4.1 IL Emission Changes

**Pattern**: Replace `Ldc_I4` with `Ldc_I8`, `Conv_I4` with `Conv_I8`

| Current IL | Change To |
|------------|-----------|
| `il.Emit(OpCodes.Ldc_I4, value)` | `il.Emit(OpCodes.Ldc_I8, (long)value)` |
| `il.Emit(OpCodes.Ldc_I4_0)` | `il.Emit(OpCodes.Ldc_I4_0); il.Emit(OpCodes.Conv_I8)` or use Ldc_I8 |
| `il.Emit(OpCodes.Conv_I4)` | `il.Emit(OpCodes.Conv_I8)` |
| Loop counters (Ldloc/Stloc for int) | Use int64 locals |

### 4.2 Files with IL Changes

| File | Occurrences | Focus Areas |
|------|-------------|-------------|
| `ILKernelGenerator.MixedType.cs` | 170 | Loop indices, stride calculations |
| `ILKernelGenerator.Reduction.cs` | 151 | Index tracking, accumulator positions |
| `ILKernelGenerator.MatMul.cs` | 130 | Matrix indices, row/col offsets |
| `ILKernelGenerator.Comparison.cs` | 125 | Loop counters |
| `ILKernelGenerator.Unary.cs` | 78 | Loop counters |
| `ILKernelGenerator.Shift.cs` | 73 | Loop counters |
| `ILKernelGenerator.Binary.cs` | 53 | Loop counters |
| `ILKernelGenerator.Scan.cs` | 52 | Cumulative indices |
| `ILKernelGenerator.Unary.Math.cs` | 41 | Loop counters |
| `ILKernelGenerator.cs` | 35 | Core emit helpers |
| Other partials | ~16 | Various |

### 4.3 DynamicMethod Signatures

Current pattern:
```csharp
new DynamicMethod("Kernel", typeof(void),
    new[] { typeof(byte*), typeof(byte*), typeof(byte*), typeof(int) });
//                                                        ^^^^ count
```

Change to:
```csharp
new DynamicMethod("Kernel", typeof(void),
    new[] { typeof(byte*), typeof(byte*), typeof(byte*), typeof(long) });
//                                                        ^^^^ count
```

### 4.4 Delegate Types

| Current | Change To |
|---------|-----------|
| `delegate void ContiguousKernel<T>(T* a, T* b, T* result, int count)` | `long count` |
| `delegate void MixedTypeKernel(...)` | All index/count params → `long` |
| `delegate void UnaryKernel(...)` | All index/count params → `long` |
| `delegate void ComparisonKernel(...)` | All index/count params → `long` |
| `delegate void TypedElementReductionKernel<T>(...)` | All index/count params → `long` |

---

## Phase 5: DefaultEngine Operations

### 5.1 Math Operations (`Backends/Default/Math/`)

| File | Changes |
|------|---------|
| `Default.Clip.cs` | Loop indices |
| `Default.ClipNDArray.cs` | Loop indices |
| `Default.Modf.cs` | Loop indices |
| `Default.Round.cs` | Loop indices |
| `Default.Shift.cs` | Loop indices |

### 5.2 Reduction Operations (`Backends/Default/Math/Reduction/`)

| File | Changes |
|------|---------|
| `Default.Reduction.Add.cs` | Index tracking |
| `Default.Reduction.Product.cs` | Index tracking |
| `Default.Reduction.AMax.cs` | Index tracking |
| `Default.Reduction.AMin.cs` | Index tracking |
| `Default.Reduction.ArgMax.cs` | Index tracking, return type stays `int` for NumPy compat |
| `Default.Reduction.ArgMin.cs` | Index tracking, return type stays `int` for NumPy compat |
| `Default.Reduction.Mean.cs` | Index tracking |
| `Default.Reduction.Var.cs` | Index tracking |
| `Default.Reduction.Std.cs` | Index tracking |

### 5.3 BLAS Operations (`Backends/Default/Math/BLAS/`)

| File | Changes |
|------|---------|
| `Default.Dot.NDMD.cs` | Matrix indices, blocked iteration |
| `Default.MatMul.2D2D.cs` | Matrix indices |
| `Default.MatMul.cs` | Matrix indices |

### 5.4 Array Manipulation (`Backends/Default/ArrayManipulation/`)

| File | Changes |
|------|---------|
| `Default.Transpose.cs` | Stride/index calculations |
| `Default.Broadcasting.cs` | Shape/stride calculations |

---

## Phase 6: API Functions

### 6.1 Creation (`Creation/`)

| File | Changes |
|------|---------|
| `np.arange.cs` | count parameter |
| `np.linspace.cs` | num parameter |
| `np.zeros.cs` | shape parameters |
| `np.ones.cs` | shape parameters |
| `np.empty.cs` | shape parameters |
| `np.full.cs` | shape parameters |
| `np.eye.cs` | N, M parameters |

### 6.2 Manipulation (`Manipulation/`)

| File | Changes |
|------|---------|
| `np.repeat.cs` | repeats parameter |
| `np.roll.cs` | shift parameter |
| `NDArray.unique.cs` | index tracking |

### 6.3 Selection (`Selection/`)

All indexing operations need long indices.

### 6.4 Statistics (`Statistics/`)

| File | Changes |
|------|---------|
| `np.nanmean.cs` | count tracking |
| `np.nanstd.cs` | count tracking |
| `np.nanvar.cs` | count tracking |

---

## Phase 7: Utilities

### 7.1 Array Utilities (`Utilities/`)

| File | Changes |
|------|---------|
| `Arrays.cs` | Index parameters |
| `ArrayConvert.cs` | 158 loop occurrences |
| `Hashset`1.cs` | Index parameters |

### 7.2 Casting (`Casting/`)

| File | Changes |
|------|---------|
| `NdArrayToJaggedArray.cs` | 24 loop occurrences |
| `UnmanagedMemoryBlock.Casting.cs` | 291 loop occurrences |

---

## Migration Strategy

### Option A: Big Bang (Recommended)

1. Create feature branch `int64-indexing`
2. Change Phase 1 (core types) atomically
3. Fix all compilation errors (cascading changes)
4. Run full test suite
5. Performance benchmark comparison

**Pros**: Clean, no hybrid state
**Cons**: Large PR, harder to review

### Option B: Incremental with Overloads

1. Add `long` overloads alongside `int` versions
2. Deprecate `int` versions
3. Migrate callers incrementally
4. Remove `int` versions

**Pros**: Easier to review, can ship incrementally
**Cons**: Code bloat during transition, easy to miss conversions

### Option C: Type Alias

```csharp
// In a central location
global using npy_intp = System.Int64;
```

Then search/replace `int` → `npy_intp` for index-related uses.

**Pros**: Easy to toggle for testing, self-documenting
**Cons**: Requires careful identification of which `int` to replace

---

## Files Summary by Impact

### High Impact (Core Types)
- `View/Shape.cs` - 20+ changes
- `View/Shape.Unmanaged.cs` - 10+ changes
- `Backends/Unmanaged/Interfaces/IArraySlice.cs` - 8 changes
- `Backends/Unmanaged/ArraySlice`1.cs` - 15+ changes
- `Backends/Unmanaged/UnmanagedStorage.cs` - 5+ changes

### Medium Impact (IL Generation)
- `Backends/Kernels/ILKernelGenerator.*.cs` - 924 IL emission changes across 13 files

### Medium Impact (Iterators)
- `Backends/Iterators/*.cs` - 28 files (including generated casts)

### Lower Impact (API Functions)
- `Creation/*.cs` - parameter changes
- `Manipulation/*.cs` - parameter changes
- `Selection/*.cs` - index changes
- `Math/*.cs` - loop indices

### Generated Code (Regen)
- `Utilities/ArrayConvert.cs` - 158 changes
- `Backends/Unmanaged/UnmanagedMemoryBlock.Casting.cs` - 291 changes
- NDIterator cast files - template-based

---

## Testing Strategy

1. **Unit Tests**: Run existing 2700+ tests - all should pass
2. **Edge Cases**: Add tests for arrays at int32 boundary (2.1B+ elements)
3. **Performance**: Benchmark suite comparing int32 vs int64 versions
4. **Memory**: Verify no memory leaks from changed allocation patterns

---

## Breaking Changes

| Change | Impact | Migration |
|--------|--------|-----------|
| `Shape.Size` returns `long` | Low | Cast to `int` if needed |
| `NDArray.size` returns `long` | Low | Cast to `int` if needed |
| `int[]` shape → `long[]` shape | Medium | Update dependent code |
| Iterator coordinate types | Low | Internal change |

Most user code uses small arrays where `int` suffices. The main impact is internal code that stores/passes indices.

---

## Estimated Effort

| Phase | Files | Estimated Hours |
|-------|-------|-----------------|
| Phase 1: Core Types | 10 | 8 |
| Phase 2: NDArray API | 8 | 4 |
| Phase 3: Iterators | 30 | 6 |
| Phase 4: IL Kernels | 13 | 16 |
| Phase 5: DefaultEngine | 20 | 8 |
| Phase 6: API Functions | 30 | 6 |
| Phase 7: Utilities | 10 | 4 |
| Testing & Fixes | - | 16 |
| **Total** | **~120** | **~68 hours** |

---

## References

- NumPy `npy_intp` definition: `numpy/_core/include/numpy/npy_common.h:217`
- NumPy uses `Py_ssize_t` which is 64-bit on x64 platforms
- .NET `nint`/`nuint` are platform-dependent (like NumPy's approach)
- Benchmark proof: 1-3% overhead acceptable for >2GB array support
