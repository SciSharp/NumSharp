# Int64 Indexing Migration - Comprehensive Guide

## Purpose

This guide defines the complete requirements for migrating NumSharp from `int` (32-bit) to `long` (64-bit) indexing. The goal is to support arrays with more than 2 billion elements.

**GitHub Issue:** https://github.com/SciSharp/NumSharp/issues/584

---

## Core Principles

1. **Migrate the SOURCE to long** - Don't cast at usage sites
2. **Maintain EXACT same API signatures** - Don't add/change parameters
3. **Rewrite algorithms to support long** - Don't downcast internally
4. **Use `Shape.ComputeLongShape()`** - Standard int[]→long[] conversion
5. **Use UnmanagedMemoryBlock** - Not .NET Array APIs
6. **Use pointers, not Span<T>** - Span is int-indexed by design
7. **Remove all int.MaxValue constraints** - By fixing the code, not by throwing

---

## Part 1: Core Types

### Must Be `long`

```csharp
// Shape
internal readonly long[] dimensions;
internal readonly long[] strides;
internal readonly long offset;
internal readonly long size;
internal readonly long bufferSize;

// Slice
public long? Start;
public long? Stop;
public long Step;

// SliceDef
public long Start;
public long Step;
public long Count;

// Storage/Iterator
public long Count;
private long index;
public long size;

// UnmanagedMemoryBlock
private long _count;
```

### Legitimately Remain `int`

| Field | Reason |
|-------|--------|
| `ndim` | Number of dimensions, always < 64 |
| `axis` | Axis index, always < ndim |
| `_hashCode` | Hash codes are int by definition |
| `_flags` | ArrayFlags enum bitmask |

---

## Part 2: Method Signatures

### Maintain Original API

```csharp
// WRONG - changed signature by adding parameter
Storage.GetData(dims, ndims).SetData(value, new int[0]);  // NEVER DO THIS

// CORRECT - same signature, implementation handles long internally
Storage.GetData(dims, ndims).SetData(value);
```

### Overload Pattern

```csharp
// Primary implementation - params on long[] only
public T GetValue(params long[] indices)
{
    // actual implementation using long
}

// Backwards compatibility - NO params, uses Shape.ComputeLongShape
public T GetValue(int[] indices) =>
    GetValue(Shape.ComputeLongShape(indices));
```

**Critical:** Using `params` on both overloads causes CS0121 ambiguity error.

### Pointer Parameters

```csharp
// Change pointer types at the source
public readonly unsafe long GetOffset(long* indices, int ndims)
public readonly unsafe (Shape, long) GetSubshape(long* dims, int ndims)
```

### Return Types

```csharp
// Methods returning indices/sizes must return long
public long CountNonZero(NDArray nd);
public long[] NonZero(NDArray nd);
public long ArgMax(NDArray nd);
public long GetOffset(long[] indices);
```

---

## Part 3: int[] to long[] Conversion

### Use Shape.ComputeLongShape()

```csharp
// CORRECT - standard conversion method
public T GetInt16(int[] indices) =>
    GetInt16(Shape.ComputeLongShape(indices));

public NDArray reshape(int[] shape) =>
    reshape(Shape.ComputeLongShape(shape));

public static Shape Create(int[] dimensions) =>
    new Shape(ComputeLongShape(dimensions));
```

### Never Use Ad-Hoc Conversion

```csharp
// WRONG - inconsistent, not standard
System.Array.ConvertAll(indices, i => (long)i)  // NO
Array.ConvertAll(shape, d => (long)d)           // NO
indices.Select(i => (long)i).ToArray()          // NO
shape.Cast<long>().ToArray()                    // NO
```

---

## Part 4: Memory Allocation

### Use UnmanagedMemoryBlock

```csharp
// WRONG - .NET Array is limited to int indices
var ret = Array.CreateInstance(typeof(T), intShape);        // NO
var ret = Arrays.Create(typeof(T), intShape);               // NO (wraps Array.CreateInstance)

// CORRECT - UnmanagedMemoryBlock supports long
var storage = new UnmanagedStorage(typeCode, size);         // size is long
var block = new UnmanagedMemoryBlock<T>(count);             // count is long
var result = new NDArray(typeCode, new Shape(longShape));   // Shape takes long[]
```

### Never Downcast Shape for Allocation

```csharp
// WRONG - downcasting defeats the purpose
var intShape = System.Array.ConvertAll(shape, d => (int)d);
var ret = Arrays.Create(typeof(T), intShape);  // NO - data loss for large arrays

// CORRECT - use long[] throughout
var storage = new UnmanagedStorage(typeCode, new Shape(shape));  // shape is long[]
```

---

## Part 5: Element Access and Iteration

### Loop Variables Must Be `long`

```csharp
// WRONG
for (int i = 0; i < size; i++)
    data[i] = value;

// CORRECT
for (long i = 0; i < size; i++)
    data[i] = value;
```

### Use Pointers, Not Span<T>

Span<T> is int-indexed by design (.NET limitation). It cannot be used for long indexing.

```csharp
// WRONG - Span indexer takes int, forced downcast
Span<T> inputSpan = storage.AsSpan<T>();
long inputIndex = inputStartIndex + a * postAxisStride;
if (!inputSpan[(int)inputIndex].Equals(default(T)))  // NO - truncates for large arrays

// CORRECT - use pointers for native long indexing
T* inputPtr = (T*)storage.Address;
long inputIndex = inputStartIndex + a * postAxisStride;
if (!inputPtr[inputIndex].Equals(default(T)))  // YES - native long indexing
```

### TODO Pattern for Span-Dependent Code

When encountering code that uses Span internally (e.g., library methods):

```csharp
// TODO: Span<T> is int-indexed by design. Decompile <MethodName>,
// copy implementation here, and upscale for long support.
// Current workaround uses (int) cast - breaks for arrays > 2B elements.
```

---

## Part 6: IL Kernel Generation

### Declare Loop Variables as `long`

```csharp
// WRONG
il.DeclareLocal(typeof(int));   // loop counter as int
il.Emit(OpCodes.Ldc_I4_0);      // int constant 0
il.Emit(OpCodes.Ldc_I4_1);      // int constant 1

// CORRECT
il.DeclareLocal(typeof(long));  // loop counter as long
il.Emit(OpCodes.Ldc_I8, 0L);    // long constant 0
il.Emit(OpCodes.Ldc_I8, 1L);    // long constant 1
```

### Change the Source, Not the Cast

```csharp
// WRONG - source is int, casting at emit site hides the problem
int vectorCount = ComputeVectorCount();
il.Emit(OpCodes.Ldc_I8, (long)vectorCount);  // NO - upcast masks int source

// CORRECT - source is long, no cast needed
long vectorCount = ComputeVectorCount();  // Method returns long
il.Emit(OpCodes.Ldc_I8, vectorCount);     // YES - already long
```

### Conv_I8 Usage

`Conv_I8` is ONLY needed when converting legitimate int values (ndim, axis) for use in long arithmetic:

```csharp
// ndim is legitimately int (always < 64)
il.Emit(OpCodes.Ldarg, ndimArg);     // int
il.Emit(OpCodes.Conv_I8);             // convert for long arithmetic
il.Emit(OpCodes.Ldloc, strideLocal);  // long
il.Emit(OpCodes.Mul);                 // long * long = long
```

**NOT needed** when everything is already long:

```csharp
// All operands are long - no conversion
il.Emit(OpCodes.Ldloc, indexLocal);   // long
il.Emit(OpCodes.Ldloc, strideLocal);  // long
il.Emit(OpCodes.Mul);                 // long result, no Conv_I8
```

---

## Part 7: Algorithm Rewriting

### Preserve Business Logic

When migrating code, preserve ALL original constraints and logic:

```csharp
// ORIGINAL
var flatIndices = new ArraySlice<long>(Math.Max(16L, size / 4L));

// WRONG - lost the 16L minimum constraint!
var flatIndices = new List<long>((int)Math.Min(size / 4L, int.MaxValue));  // NO

// CORRECT - preserve the minimum constraint
var flatIndices = new ArraySlice<long>(Math.Max(16L, size / 4L));  // Keep as-is, already long
// OR if changing container:
long capacity = Math.Max(16L, size / 4L);  // Preserve the 16L minimum
var flatIndices = new UnmanagedMemoryBlock<long>(capacity);
```

### Move Away from Array/Span Methods

```csharp
// WRONG - Array.Clear is int-limited, casting breaks for large N
Array.Clear(accumulator, 0, (int)N);  // NO - truncates for N > int.MaxValue

// CORRECT - use pointer-based clearing
T* ptr = (T*)accumulator.Address;
for (long i = 0; i < N; i++)
    ptr[i] = default;

// OR for unmanaged memory:
Unsafe.InitBlockUnaligned(ptr, 0, (uint)(N * sizeof(T)));  // For small N
// For large N, use loop or chunked clearing
```

### No int.MaxValue Constraints

```csharp
// WRONG - constraint instead of fix
if (arr.size > int.MaxValue)
    throw new NotSupportedException("choice does not support arrays > int.MaxValue");
// ... int-based code follows ...

// CORRECT - remove constraint, rewrite algorithm to use long
long size = arr.size;
for (long i = 0; i < size; i++)
    Process(ptr[i]);
```

### Don't Accept Long Then Downcast

```csharp
// WRONG - lying API: signature claims long support, internally uses int
public static unsafe void MatMulFloat(float* A, float* B, float* C, long M, long N, long K)
{
    // This defeats the entire purpose of the migration
    if (M > int.MaxValue || N > int.MaxValue || K > int.MaxValue)
        throw new ArgumentException("Matrix dimensions exceed int.MaxValue");  // NO

    int m = (int)M, n = (int)N, k = (int)K;  // NO - downcast

    for (int i = 0; i < m; i++)  // NO - int loop
        // ... int-based algorithm ...
}

// CORRECT - algorithm actually works with long
public static unsafe void MatMulFloat(float* A, float* B, float* C, long M, long N, long K)
{
    for (long i = 0; i < M; i++)
    {
        for (long j = 0; j < N; j++)
        {
            float sum = 0;
            for (long p = 0; p < K; p++)
            {
                sum += A[i * K + p] * B[p * N + j];  // Native long indexing
            }
            C[i * N + j] = sum;
        }
    }
}
```

---

## Part 8: Test Assertions

### Shape Comparisons

```csharp
// WRONG - new[] creates int[], shape is long[]
await Assert.That(result.shape).IsEquivalentTo(new[] { 3 });        // NO
await Assert.That(result.shape).IsEquivalentTo(new[] { 1, 3 });     // NO

// CORRECT - explicit long[]
await Assert.That(result.shape).IsEquivalentTo(new long[] { 3 });   // YES
await Assert.That(result.shape).IsEquivalentTo(new long[] { 1, 3 });// YES
```

### Dtype Access

```csharp
// WRONG - searchsorted returns Int64, accessing as Int32
Assert.AreEqual(1, result.GetInt32());      // NO - wrong dtype
Assert.AreEqual(1, result.GetInt32(0));     // NO

// CORRECT - match the actual dtype
Assert.AreEqual(1L, result.GetInt64());     // YES
Assert.AreEqual(1L, result.GetInt64(0));    // YES
```

---

## Part 9: .NET API Boundaries

Some .NET APIs are fundamentally limited to int. These are the ONLY acceptable places for int constraints:

| API | Limitation | Mitigation |
|-----|------------|------------|
| `Array.CreateInstance` | int[] lengths | Use `UnmanagedMemoryBlock` instead |
| `Array.SetValue` | int[] indices | Use pointer arithmetic |
| `Array.Clear` | int length | Use pointer loop or `Unsafe.InitBlockUnaligned` |
| `Span<T>` indexer | int index | Use raw pointers |
| `Span<T>.Length` | int | Use `UnmanagedMemoryBlock.Count` (long) |
| `List<T>` capacity | int | Use `UnmanagedMemoryBlock` or `ArraySlice` |
| `ToMuliDimArray<T>()` | Returns .NET Array | Document as limited, suggest alternatives |

### String is a Special Case

.NET `string` internally uses `int` for length - this is a fundamental CLR limitation.

**Storing data > int.MaxValue:** SUPPORTED - NDArray can hold any amount of data
**Converting to string > int.MaxValue:** NOT SUPPORTED - throw at conversion boundary only

```csharp
// This is ACCEPTABLE - string is genuinely limited
public override string ToString()
{
    if (size > int.MaxValue)
        throw new InvalidOperationException(
            "Array size exceeds maximum .NET string length. " +
            "Use element-wise access or chunked output.");

    // ... string building code ...
}

// The NDArray itself works fine with long - only ToString has the limit
var huge = np.zeros(3_000_000_000L);  // Works
huge.ToString();  // Throws - acceptable
huge.GetDouble(2_999_999_999L);  // Works - element access is fine
```

**Key distinction:**
- int.MaxValue check at STRING CONVERSION boundary = OK (true .NET limitation)
- int.MaxValue check in ALGORITHMS = NOT OK (rewrite to use long)

For truly unavoidable .NET interop:

```csharp
/// <summary>
/// Converts to .NET multi-dimensional array.
/// </summary>
/// <remarks>
/// LIMITED TO int.MaxValue ELEMENTS due to .NET Array limitations.
/// For larger arrays, use ToArray&lt;T&gt;() or direct pointer access.
/// </remarks>
public Array ToMuliDimArray<T>() where T : unmanaged
{
    if (size > int.MaxValue)
        throw new NotSupportedException(
            "Cannot convert to .NET Array: size exceeds int.MaxValue. " +
            "Use ToArray<T>() or pointer access for large arrays.");

    // ... implementation using int (documented limitation) ...
}
```

---

## Part 10: Code Review Checklist

### For Each File

- [ ] **Source types are `long`** - dimensions, strides, offset, size, index, count
- [ ] **Same API signatures** - no added/changed/removed parameters
- [ ] **Business logic preserved** - Min/Max constraints, default values, edge cases
- [ ] **`int[]` overloads use `Shape.ComputeLongShape()`** - not ad-hoc conversion
- [ ] **`params` only on `long[]` overloads** - int[] has no params
- [ ] **No `(int)` downcasts** - truncating long values
- [ ] **No `(long)` upcasts at usage** - change source instead
- [ ] **No int.MaxValue constraints in algorithms** - rewrite to support long
- [ ] **int.MaxValue only at string boundary** - ToString() can legitimately throw
- [ ] **Allocation uses `UnmanagedMemoryBlock`** - not Array.CreateInstance
- [ ] **No Array.Clear/Span with long** - use pointer loops
- [ ] **No List<T> with long capacity** - use ArraySlice or UnmanagedMemoryBlock
- [ ] **No Span<T> with long indices** - use pointers
- [ ] **TODO added for Span internals** - where decompilation needed
- [ ] **IL locals are `typeof(long)`** - for size/index/count
- [ ] **IL uses `Ldc_I8`** - not Ldc_I4 with cast
- [ ] **`Conv_I8` only for ndim/axis** - legitimate int values
- [ ] **Loop variables are `long`** - for element iteration
- [ ] **Pointer params are `long*`** - not int*
- [ ] **Test assertions use `long[]`** - for shape comparisons
- [ ] **Test assertions match dtype** - GetInt64 for Int64 results

### Red Flags to Search For

```bash
# Downcasts (potential data loss)
grep -r "(int)" --include="*.cs" | grep -v "// OK"

# Upcasts at usage (hiding int source)
grep -r "(long)" --include="*.cs" | grep -v "// OK"

# int.MaxValue constraints (except ToString boundary)
grep -r "int.MaxValue" --include="*.cs" | grep -v "ToString\|string"

# Ad-hoc array conversion (should use Shape.ComputeLongShape)
grep -r "ConvertAll" --include="*.cs"
grep -r "Cast<long>" --include="*.cs"

# Span with long index (use pointers instead)
grep -r "span\[.*\(int\)" --include="*.cs"
grep -r "Span<" --include="*.cs" | grep -v "ReadOnlySpan<char>"

# Array methods with cast (use pointer loops)
grep -r "Array.Clear.*\(int\)" --include="*.cs"
grep -r "Array.Copy.*\(int\)" --include="*.cs"

# List with long capacity cast (use ArraySlice/UnmanagedMemoryBlock)
grep -r "new List<.*>\((int)" --include="*.cs"

# int loop variables for elements
grep -r "for (int i = 0; i < size" --include="*.cs"
grep -r "for (int i = 0; i < count" --include="*.cs"
grep -r "for (int i = 0; i < length" --include="*.cs"

# IL int declarations for size/index
grep -r "typeof(int)" --include="ILKernelGenerator*.cs"
grep -r "Ldc_I4" --include="ILKernelGenerator*.cs"

# params on int[] overloads
grep -r "params int\[\]" --include="*.cs"

# Wrong API changes (added parameters)
grep -r "Array.Empty<long>()" --include="*.cs"
grep -r "new int\[0\]" --include="*.cs"
grep -r "new long\[0\]" --include="*.cs"

# Lost business logic (Math.Max/Min constraints)
# Review any line that changed Math.Max or Math.Min
grep -r "Math.Max\|Math.Min" --include="*.cs"

# Test assertions with int[]
grep -r "IsEquivalentTo(new\[\]" --include="*.cs" test/
grep -r "GetInt32()" --include="*.cs" test/
```

---

## Part 11: Migration Workflow

### Step 1: Identify int Usage
1. Search for `int` declarations in core types
2. Search for `int[]` parameters
3. Search for `int` return types
4. Search for `(int)` casts

### Step 2: Change Source Types
1. Change field/variable declarations to `long`
2. Change method parameter types to `long[]`
3. Change return types to `long`
4. Update pointer types to `long*`

### Step 3: Add Backwards Compatibility
1. Add `int[]` overloads that delegate to `long[]`
2. Use `Shape.ComputeLongShape()` for conversion
3. Remove `params` from `int[]` overloads

### Step 4: Fix Algorithms
1. Change loop variables to `long`
2. Replace Span with pointers
3. Remove int.MaxValue constraints
4. Ensure arithmetic uses `long` throughout

### Step 5: Update IL Generation
1. Change local declarations to `typeof(long)`
2. Use `Ldc_I8` for constants
3. Remove unnecessary `Conv_I8`
4. Keep `Conv_I8` only for ndim/axis

### Step 6: Update Tests
1. Change shape assertions to `long[]`
2. Change dtype access to match return type
3. Verify tests pass

### Step 7: Verify
1. Build succeeds with no errors
2. All tests pass
3. No warnings about narrowing conversions
4. Code review checklist complete

---

## Summary

The int64 migration is NOT about:
- Adding casts at usage sites
- Throwing exceptions for large arrays (except ToString)
- Changing API signatures or calling conventions
- Accepting long parameters but using int internally
- Using Array/Span/List with casted long values
- Losing business logic (Min/Max constraints, defaults)

The int64 migration IS about:
- Changing source types to `long`
- Rewriting algorithms to natively use `long`
- Using `UnmanagedMemoryBlock` for allocation
- Using pointers instead of Span/Array methods
- Maintaining exact same API calling conventions
- Using `Shape.ComputeLongShape()` for standard conversion
- Preserving ALL original business logic and constraints
- Only throwing at true .NET boundaries (string conversion)

**Every line of code that handles array indices, sizes, offsets, or strides must work correctly for arrays with more than 2 billion elements.**
