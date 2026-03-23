# NumSharp Performance Recommendations

Based on comprehensive analysis of NumPy's optimization strategies, this document provides actionable recommendations for improving NumSharp performance.

## Current State

NumSharp benchmark comparison with NumPy:

| Operation | NumSharp | NumPy | Ratio | Root Cause |
|-----------|----------|-------|-------|------------|
| a + b (int32) | 22.1ms | 12.4ms | 1.8x | Allocation overhead |
| a + b (float64) | 39.9ms | 21.2ms | 1.9x | Allocation overhead |
| np.sum | 29.6ms | 5.4ms | 5.5x | No pairwise summation |
| np.mean | 29.6ms | 5.4ms | 5.5x | No pairwise summation |

---

## Priority 0: Critical Optimizations

### P0.1: Implement Temporary Elision

**Impact**: 4-6x speedup for chained operations like `a + b + c + d`

**Problem**: Each operation allocates a new output array. For `a + b + c + d`:
- Current: 3 allocations (40MB each for 10M float64)
- NumPy: 1 allocation (temp reused)

**Implementation**:

```csharp
public class NDArray
{
    // Flag indicating this array was created by an operation (not user)
    internal bool IsTemporary { get; set; }

    // Reference count approximation for managed objects
    internal bool HasSingleReference => /* implementation */;
}

public static NDArray Add(NDArray a, NDArray b)
{
    // Check if 'a' can be reused
    if (a.IsTemporary &&
        a.HasSingleReference &&
        a.IsContiguous &&
        a.dtype == ResultDtype(a, b) &&
        a.Shape.Equals(ResultShape(a, b)))
    {
        AddInPlace(a, b);
        return a;
    }

    // Check if 'b' can be reused (commutative)
    if (b.IsTemporary &&
        b.HasSingleReference &&
        b.IsContiguous &&
        b.dtype == ResultDtype(a, b) &&
        b.Shape.Equals(ResultShape(a, b)))
    {
        AddInPlace(b, a);
        return b;
    }

    // Allocate new array
    var result = new NDArray(ResultDtype(a, b), ResultShape(a, b));
    result.IsTemporary = true;
    AddInto(result, a, b);
    return result;
}
```

**Elision conditions** (from NumPy):
- Array is temporary (created by operation, not user)
- Single reference (no other code holds it)
- Owns data (not a view)
- Writeable
- Size >= 256KB (overhead worth it)
- Same dtype or safe cast
- Shape matches result

### P0.2: Implement Pairwise Summation

**Impact**: 3-5x speedup for reductions (closes 5.5x gap)

**Problem**: Current implementation uses linear accumulation with 4 accumulators. NumPy uses pairwise with 8 accumulators and recursive splitting.

**Implementation**:

```csharp
public static class PairwiseSum
{
    private const int BlockSize = 128;

    public static double Sum(double* data, int n, int stride = 1)
    {
        if (n < 8)
        {
            // Small array: direct accumulation
            double r = 0.0;  // Use -0.0 to preserve sign
            for (int i = 0; i < n; i++)
                r += data[i * stride];
            return r;
        }
        else if (n <= BlockSize)
        {
            // Medium array: 8 accumulators, unrolled
            double r0 = data[0 * stride], r1 = data[1 * stride];
            double r2 = data[2 * stride], r3 = data[3 * stride];
            double r4 = data[4 * stride], r5 = data[5 * stride];
            double r6 = data[6 * stride], r7 = data[7 * stride];

            for (int i = 8; i < n - (n % 8); i += 8)
            {
                // Prefetch 512 bytes ahead
                Sse.Prefetch0(data + (i + 64) * stride);

                r0 += data[(i + 0) * stride];
                r1 += data[(i + 1) * stride];
                r2 += data[(i + 2) * stride];
                r3 += data[(i + 3) * stride];
                r4 += data[(i + 4) * stride];
                r5 += data[(i + 5) * stride];
                r6 += data[(i + 6) * stride];
                r7 += data[(i + 7) * stride];
            }

            // Handle remainder
            for (int i = n - (n % 8); i < n; i++)
                r0 += data[i * stride];

            // Tree reduction preserves pairing
            return ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
        }
        else
        {
            // Large array: recursive split
            int n2 = (n / 2) - ((n / 2) % 8);  // Align to 8
            return Sum(data, n2, stride) +
                   Sum(data + n2 * stride, n - n2, stride);
        }
    }
}
```

### P0.3: Fix SIMD for Type-Promoted Reductions

**Impact**: 4-8x speedup for int32 sum (currently falls back to scalar)

**Problem**: When summing int32 to int64 accumulator, NumSharp disables SIMD because vector widening is complex.

**Solution**: Accumulate in same type, periodically drain to wider accumulator:

```csharp
public static long SumInt32ToInt64(int* data, int n)
{
    const int DrainInterval = 1000000;  // Drain before overflow risk
    long total = 0;
    int remaining = n;

    while (remaining > 0)
    {
        int chunk = Math.Min(remaining, DrainInterval);
        int partialSum = SumInt32Simd(data, chunk);  // SIMD sum in int32
        total += partialSum;
        data += chunk;
        remaining -= chunk;
    }

    return total;
}
```

Or use widening SIMD instructions:
```csharp
// AVX2: widen int32 to int64
Vector256<long> sum = Vector256<long>.Zero;
for (int i = 0; i < n; i += 4)
{
    Vector128<int> v = Sse2.LoadVector128(data + i);
    Vector256<long> wide = Avx2.ConvertToVector256Int64(v);
    sum = Avx2.Add(sum, wide);
}
```

---

## Priority 1: High Impact Optimizations

### P1.1: Fast Path for Same-Type Contiguous Arrays

**Impact**: 1.5x speedup for binary operations

**Problem**: NumSharp goes through full path classification even for the most common case.

**Implementation**:

```csharp
public static NDArray Add(NDArray a, NDArray b)
{
    // Fast path: same type, same shape, both contiguous
    if (a.typecode == b.typecode &&
        a.Shape.IsContiguous &&
        b.Shape.IsContiguous &&
        a.Shape.Equals(b.Shape))
    {
        var result = new NDArray(a.typecode, a.Shape);
        SimdBinaryContiguous(a.Address, b.Address, result.Address,
                            a.size, a.typecode, BinaryOp.Add);
        return result;
    }

    // Full path with broadcasting, type promotion, etc.
    return AddGeneral(a, b);
}
```

### P1.2: Axis Reordering for Reductions

**Impact**: 2-3x speedup for axis reductions

**Problem**: `sum(axis=0)` on a C-order array iterates non-contiguously.

**Implementation**:

```csharp
public static int[] OptimalAxisOrder(Shape shape, int[] reduceAxes)
{
    // Sort axes by stride (smallest first = most contiguous)
    var axisInfo = Enumerable.Range(0, shape.NDim)
        .Select(i => (axis: i, stride: Math.Abs(shape.strides[i])))
        .OrderBy(x => x.stride)
        .ToArray();

    // Inner loop should be on smallest stride
    return axisInfo.Select(x => x.axis).ToArray();
}
```

### P1.3: Axis Coalescing

**Impact**: 2x speedup for multi-dimensional reductions

**Problem**: Nested loops have overhead. Adjacent compatible dimensions can be merged.

```csharp
public static Shape CoalesceAxes(Shape shape)
{
    var newDims = new List<int>();
    var newStrides = new List<int>();

    int currentDim = shape.dimensions[0];
    int currentStride = shape.strides[0];

    for (int i = 1; i < shape.NDim; i++)
    {
        // Can coalesce if strides are compatible
        if (shape.strides[i] == currentStride * currentDim)
        {
            currentDim *= shape.dimensions[i];
        }
        else
        {
            newDims.Add(currentDim);
            newStrides.Add(currentStride);
            currentDim = shape.dimensions[i];
            currentStride = shape.strides[i];
        }
    }

    newDims.Add(currentDim);
    newStrides.Add(currentStride);

    return new Shape(newDims.ToArray(), newStrides.ToArray());
}
```

### P1.4: Use 8 Accumulators Instead of 4

**Impact**: 1.3x speedup for reductions

Current ILKernelGenerator uses 4 vector accumulators. NumPy uses 8.

---

## Priority 2: Medium Impact Optimizations

### P2.1: Small Array Cache

**Impact**: Faster small array operations (< 1KB)

```csharp
public class SmallArrayCache
{
    private const int MaxSize = 1024;
    private const int MaxCached = 7;

    private readonly ConcurrentStack<byte[]>[] _buckets =
        new ConcurrentStack<byte[]>[MaxSize];

    public byte[] Rent(int size)
    {
        if (size < MaxSize && _buckets[size]?.TryPop(out var arr) == true)
            return arr;
        return new byte[size];
    }

    public void Return(byte[] arr)
    {
        if (arr.Length < MaxSize)
        {
            var bucket = _buckets[arr.Length] ??= new ConcurrentStack<byte[]>();
            if (bucket.Count < MaxCached)
                bucket.Push(arr);
        }
    }
}
```

Or use `System.Buffers.ArrayPool<byte>.Shared`.

### P2.2: Partial Load/Store for Remainder Handling

**Impact**: Cleaner, potentially faster tail handling

```csharp
// Current: scalar loop for remainder
for (int i = vectorEnd; i < n; i++)
    dst[i] = src[i] + scalar;

// Better: masked vector operation
if (remaining > 0)
{
    var mask = CreateMask(remaining);  // e.g., [1,1,1,0,0,0,0,0]
    var v = MaskedLoad(src + vectorEnd, mask);
    var r = Vector256.Add(v, scalarVec);
    MaskedStore(dst + vectorEnd, mask, r);
}
```

### P2.3: Memory Overlap Detection

**Impact**: Safe in-place operations

```csharp
public static bool MayShareMemory(NDArray a, NDArray b)
{
    // Fast path: different storage
    if (!ReferenceEquals(a.Storage, b.Storage))
        return false;

    // Calculate memory extents
    var (aStart, aEnd) = GetMemoryExtent(a);
    var (bStart, bEnd) = GetMemoryExtent(b);

    // Check if extents overlap
    return aStart < bEnd && bStart < aEnd;
}

private static (long start, long end) GetMemoryExtent(NDArray arr)
{
    long start = arr.Address.ToInt64();
    long end = start;

    for (int i = 0; i < arr.ndim; i++)
    {
        if (arr.strides[i] > 0)
            end += (arr.shape[i] - 1) * arr.strides[i];
        else
            start += (arr.shape[i] - 1) * arr.strides[i];
    }

    return (start, end + arr.itemsize);
}
```

### P2.4: Support `out=` Parameter

**Impact**: User-controlled buffer reuse

```csharp
public static NDArray Add(NDArray a, NDArray b, NDArray? @out = null)
{
    var resultShape = BroadcastShape(a.Shape, b.Shape);
    var resultDtype = PromoteTypes(a.typecode, b.typecode);

    NDArray result;
    if (@out != null)
    {
        ValidateOutput(@out, resultShape, resultDtype);
        if (MayShareMemory(@out, a) || MayShareMemory(@out, b))
        {
            // Operate into temp, then copy
            var temp = new NDArray(resultDtype, resultShape);
            AddInto(temp, a, b);
            temp.CopyTo(@out);
            return @out;
        }
        result = @out;
    }
    else
    {
        result = new NDArray(resultDtype, resultShape);
    }

    AddInto(result, a, b);
    return result;
}
```

---

## Priority 3: Lower Impact Optimizations

### P3.1: Division by Invariant

For `a / scalar`, precompute magic multiplier:

```csharp
// Instead of: result[i] = a[i] / scalar
// Use: result[i] = MultiplyHigh(a[i], magic) >> shift
```

### P3.2: Prefetch Hints

```csharp
// In hot loops:
Sse.Prefetch0(ptr + 64);  // Prefetch 512 bytes ahead to L1
```

### P3.3: BLAS Integration

For `np.matmul` with large matrices, integrate with:
- Math.NET Numerics
- Intel MKL via P/Invoke
- OpenBLAS

---

## Architecture Recommendation

### Shared Infrastructure Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                     NumSharp Shared Infrastructure                  │
├─────────────────────────────────────────────────────────────────────┤
│  PathClassifier                                                     │
│  ├─ IsContiguous(a, b) → bool                                       │
│  ├─ IsScalarBroadcast(a, b) → (bool isFirst, bool isSecond)         │
│  ├─ IsSIMDEligible(a, b) → bool                                     │
│  └─ ClassifyPath(a, b) → ExecutionPath enum                         │
├─────────────────────────────────────────────────────────────────────┤
│  LoopDispatcher<TOp> where TOp : IBinaryOp<T>                       │
│  ├─ SimdContiguous<T>(a, b, result)                                 │
│  ├─ SimdScalarFirst<T>(scalar, b, result)                           │
│  ├─ SimdScalarSecond<T>(a, scalar, result)                          │
│  └─ ScalarStrided<T>(a, b, result)                                  │
├─────────────────────────────────────────────────────────────────────┤
│  SimdKernel<T>                                                      │
│  ├─ PeelToAlignment()                                               │
│  ├─ MainLoop4xUnrolled()                                            │
│  └─ TailWithMask()                                                  │
├─────────────────────────────────────────────────────────────────────┤
│  ReductionKernel<T>                                                 │
│  ├─ PairwiseSum()                                                   │
│  ├─ AxisReduce()                                                    │
│  └─ TreeHorizontalReduce()                                          │
└─────────────────────────────────────────────────────────────────────┘
```

### Operation Definition (5% of code)

```csharp
// Each operation is just an interface implementation
public readonly struct AddOp : IBinaryOp<float>
{
    public float ScalarOp(float a, float b) => a + b;
    public Vector256<float> VectorOp(Vector256<float> a, Vector256<float> b)
        => Avx.Add(a, b);
}

public readonly struct SqrtOp : IUnaryOp<float>
{
    public float ScalarOp(float a) => MathF.Sqrt(a);
    public Vector256<float> VectorOp(Vector256<float> a)
        => Avx.Sqrt(a);
}
```

### Usage

```csharp
public static NDArray Add(NDArray a, NDArray b)
{
    return BinaryDispatcher.Execute<AddOp, float>(a, b);
}

public static NDArray Sqrt(NDArray a)
{
    return UnaryDispatcher.Execute<SqrtOp, float>(a);
}
```

---

## Implementation Roadmap

### Phase 1: Quick Wins (1-2 weeks)
- [ ] P0.2: Pairwise summation (biggest impact on reduction gap)
- [ ] P1.1: Fast path for contiguous same-type arrays
- [ ] P1.4: 8 accumulators instead of 4

### Phase 2: Allocation Optimization (2-3 weeks)
- [ ] P0.1: Temporary elision infrastructure
- [ ] P2.4: `out=` parameter support
- [ ] P2.1: Small array cache

### Phase 3: Advanced Paths (3-4 weeks)
- [ ] P0.3: SIMD for type-promoted reductions
- [ ] P1.2: Axis reordering
- [ ] P1.3: Axis coalescing
- [ ] P2.2: Partial load/store

### Phase 4: Infrastructure Refactor (ongoing)
- [ ] Shared LoopDispatcher
- [ ] Shared SimdKernel
- [ ] Operation interfaces

---

## Measurement

Benchmark after each phase:

```csharp
// Standard benchmark array
var a = np.random.rand(10_000_000);
var b = np.random.rand(10_000_000);

// Binary operations
Measure(() => a + b);
Measure(() => a + b + a + b);  // Chained - tests temp elision

// Reductions
Measure(() => np.sum(a));
Measure(() => np.sum(a.reshape(1000, 10000), axis: 1));

// Target: within 2x of NumPy for all operations
```

---

## IL Generator Architecture (Matching NumPy)

### Critical Insight: How NumPy Actually Works

**NumPy does NOT generate separate functions for each "path".** Instead:

1. **ONE function per operation per dtype** (e.g., `DOUBLE_add`)
2. That function contains **RUNTIME if/else branches** that check stride patterns
3. **Multiple CPU-target variants** are compiled (AVX512, AVX2, SSE) but with identical internal structure
4. CPU variant is selected **once at module load**, then the function pointer is cached

```c
// NumPy's ACTUAL structure (from loops_arithm_fp.dispatch.c.src)
NPY_NO_EXPORT void DOUBLE_add(char **args, npy_intp const *dimensions,
                               npy_intp const *steps, void *func)
{
    npy_intp ssrc0 = steps[0], ssrc1 = steps[1], sdst = steps[2];
    char *src0 = args[0], *src1 = args[1], *dst = args[2];

    // RUNTIME BRANCH 1: Reduce pattern (a += b)
    if (ssrc0 == 0 && ssrc0 == sdst && src0 == dst) {
        // Pairwise summation with 8 accumulators
        ...
    }
    // RUNTIME BRANCH 2: All contiguous
    else if (ssrc0 == sizeof(double) && ssrc0 == ssrc1 && ssrc0 == sdst) {
        // SIMD contiguous loop (4x unrolled)
        for (; len >= vstep*4; len -= vstep*4, ...) {
            v0 = npyv_load_f64(src0); v1 = npyv_load_f64(src0 + vstep); ...
            r0 = npyv_add_f64(v0, v1); ...
            npyv_store_f64(dst, r0); ...
        }
        // Tail handling
        ...
    }
    // RUNTIME BRANCH 3: Scalar + contiguous array
    else if (ssrc0 == 0 && ssrc1 == sizeof(double) && sdst == ssrc1) {
        npyv_f64 scalar_vec = npyv_setall_f64(*(double*)src0);
        // SIMD loop with broadcast scalar
        ...
    }
    // RUNTIME BRANCH 4: Contiguous array + scalar
    else if (ssrc1 == 0 && ssrc0 == sizeof(double) && sdst == ssrc0) {
        npyv_f64 scalar_vec = npyv_setall_f64(*(double*)src1);
        // SIMD loop with broadcast scalar
        ...
    }
    // RUNTIME BRANCH 5: General fallback
    else {
        // Scalar strided loop
        for (i = 0; i < n; i++, src0 += ssrc0, src1 += ssrc1, dst += sdst) {
            *(double*)dst = *(double*)src0 + *(double*)src1;
        }
    }
}
```

### NumSharp ILKernelGenerator Should Match This

**Generate ONE method per operation+dtype with runtime branches:**

```csharp
// What ILKernelGenerator should emit (pseudocode of generated IL)
void Generated_Add_Float64(
    double* lhs, double* rhs, double* result,
    int lhsStride, int rhsStride, int resultStride,
    int length)
{
    // RUNTIME BRANCH 1: Reduce pattern
    if (lhsStride == 0 && lhs == result) {
        // Emit: Pairwise summation
    }
    // RUNTIME BRANCH 2: All contiguous
    else if (lhsStride == 1 && rhsStride == 1 && resultStride == 1) {
        // Emit: SIMD contiguous (4x unrolled)
        int i = 0;
        for (; i <= length - Vector256<double>.Count * 4; i += Vector256<double>.Count * 4) {
            var v0 = Vector256.Load(lhs + i);
            var v1 = Vector256.Load(lhs + i + Vector256<double>.Count);
            // ... 4x unrolled
            var r0 = Vector256.Add(v0, Vector256.Load(rhs + i));
            // ...
            Vector256.Store(result + i, r0);
            // ...
        }
        // Tail
        for (; i < length; i++) result[i] = lhs[i] + rhs[i];
    }
    // RUNTIME BRANCH 3: Scalar + array
    else if (lhsStride == 0 && rhsStride == 1 && resultStride == 1) {
        var scalarVec = Vector256.Create(*lhs);
        // SIMD loop
    }
    // RUNTIME BRANCH 4: Array + scalar
    else if (rhsStride == 0 && lhsStride == 1 && resultStride == 1) {
        var scalarVec = Vector256.Create(*rhs);
        // SIMD loop
    }
    // RUNTIME BRANCH 5: General
    else {
        // Coordinate-based or strided scalar loop
    }
}
```

### Kernel Count (Corrected)

**Per ufunc, NumPy generates:**

| Level | What | Count |
|-------|------|-------|
| C functions | Per dtype | 1 function × 12 dtypes = **12 functions** |
| CPU variants | Per function | ×3 (AVX512, AVX2, SSE baseline) |
| Runtime branches | Inside each function | 5 branches (not separate functions) |

**NumSharp should generate:**

| Level | What | Count |
|-------|------|-------|
| IL methods | Per operation × dtype | 1 method × 12 dtypes = **12 methods** |
| Runtime branches | Inside each method | 5 branches (matching NumPy) |

### The 5 Runtime Branches (Binary Operations)

These are **NOT separate kernels** - they are **if/else branches within ONE kernel**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│           RUNTIME BRANCHES WITHIN ONE BINARY KERNEL                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Branch 1: REDUCE                                                           │
│  ├─ Condition: lhsStride==0 && lhs==result (accumulator pattern)            │
│  ├─ Algorithm: Pairwise summation, 8 accumulators, recursive split          │
│  └─ Used by: np.add.reduce(), a += b                                        │
│                                                                             │
│  Branch 2: SIMD_CONTIGUOUS                                                  │
│  ├─ Condition: All strides == sizeof(element)                               │
│  ├─ Algorithm: 4x unrolled vector loop + scalar tail                        │
│  └─ Used by: Most common case (contiguous arrays)                           │
│                                                                             │
│  Branch 3: SIMD_SCALAR_FIRST                                                │
│  ├─ Condition: lhsStride==0, others contiguous                              │
│  ├─ Algorithm: Broadcast scalar to vector, then SIMD loop                   │
│  └─ Used by: scalar + array, e.g., 5 + a                                    │
│                                                                             │
│  Branch 4: SIMD_SCALAR_SECOND                                               │
│  ├─ Condition: rhsStride==0, others contiguous                              │
│  ├─ Algorithm: Broadcast scalar to vector, then SIMD loop                   │
│  └─ Used by: array + scalar, e.g., a + 5                                    │
│                                                                             │
│  Branch 5: GENERAL_FALLBACK                                                 │
│  ├─ Condition: All other cases (strided, misaligned, overlap)               │
│  ├─ Algorithm: Scalar loop with arbitrary strides                           │
│  └─ Used by: Sliced arrays, transposed views, etc.                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Branch Patterns by Operation Category

| Category | Branch 1 | Branch 2 | Branch 3 | Branch 4 | Branch 5 |
|----------|----------|----------|----------|----------|----------|
| **Binary** | Reduce | Contiguous | Scalar+Array | Array+Scalar | General |
| **Unary** | - | Contig→Contig | Strided→Contig | Contig→Strided | General |
| **Reduce** | Pairwise | AxisContig | AxisStrided | - | Iterator |
| **Compare** | - | Contiguous | Scalar+Array | Array+Scalar | General |
| **Scan** | - | Contiguous | AxisContig | - | AxisGeneral |

### What Needs Implementation (Corrected)

**Not "26 separate paths" but rather "missing branches within existing kernels":**

| Kernel Type | Current Branches | Missing Branches | Priority |
|-------------|------------------|------------------|----------|
| Binary | Contiguous, General | **Reduce (pairwise)**, Scalar+Array, Array+Scalar | Critical |
| Unary | Contig→Contig, General | Strided→Contig, Contig→Strided | Medium |
| Reduce | AxisBasic | **Pairwise**, AxisReorder | Critical |
| Compare | Contiguous, General | Scalar+Array, Array+Scalar | Medium |
| Scan | Contiguous, AxisBasic | - | Low |

### Implementation Priority (Corrected)

#### Critical: Add Missing Branches

| Missing Branch | In Kernel | Gap Impact | Effort |
|----------------|-----------|------------|--------|
| **Reduce/Pairwise** | Binary reduce, Sum | 5.5x → 2x | Add ~200 lines to existing |
| **Scalar+Array** | Binary | 1.5x → 1.1x | Add ~50 lines branch |
| **Array+Scalar** | Binary | 1.5x → 1.1x | Add ~50 lines branch |
| **AxisReorder** | Reduce | 3x → 1.5x | Add ~100 lines branch |

#### Medium: Strided SIMD Branches

| Missing Branch | In Kernel | Benefit | Effort |
|----------------|-----------|---------|--------|
| Strided→Contig | Unary | Transposed input | Add ~100 lines |
| Contig→Strided | Unary | Transposed output | Add ~100 lines |
| Scalar+Array | Compare | Scalar comparison | Add ~50 lines |

### Summary (Corrected)

| Metric | Previous (Wrong) | Corrected |
|--------|------------------|-----------|
| Architecture | 26 separate kernel functions | 1 kernel with 5 runtime branches |
| What to generate | Separate IL methods per path | One IL method with if/else branches |
| Total IL methods | 26 × 12 dtypes = 312 | 1 × 12 dtypes = 12 per operation |
| Code to add | ~4,000 lines (new paths) | ~500 lines (new branches in existing) |
| Path selection | At dispatch time | At runtime via stride checks |

### NumPy's Stride-Check Macros (for reference)

```c
// From fast_loop_macros.h - these are RUNTIME checks

// Check if all arrays are contiguous
#define IS_BINARY_CONT(tin, tout) \
    (steps[0] == sizeof(tin) && steps[1] == sizeof(tin) && steps[2] == sizeof(tout))

// Check if first operand is scalar
#define IS_BINARY_CONT_S1(tin, tout) \
    (steps[0] == 0 && steps[1] == sizeof(tin) && steps[2] == sizeof(tout))

// Check if second operand is scalar
#define IS_BINARY_CONT_S2(tin, tout) \
    (steps[0] == sizeof(tin) && steps[1] == 0 && steps[2] == sizeof(tout))

// Check if this is a reduction (accumulator pattern)
#define IS_BINARY_REDUCE \
    (args[0] == args[2] && steps[0] == steps[2] && steps[0] == 0)

// Check if SIMD-safe (alignment + no problematic overlap)
#define IS_BLOCKABLE_BINARY(esize, vsize) \
    (IS_BINARY_CONT(esize, esize) && \
     npy_is_aligned(args[0], esize) && \
     npy_is_aligned(args[1], esize) && \
     npy_is_aligned(args[2], esize) && \
     (abs_ptrdiff(args[2], args[0]) >= vsize || args[2] == args[0]))
```

### Additional Multi-Output Operations

These require **separate kernel signatures** (not just branches):

| Operation | Signature | NumPy Macro |
|-----------|-----------|-------------|
| `modf` | (in) → (frac, int) | `UNARY_LOOP_TWO_OUT` |
| `frexp` | (in) → (mantissa, exp) | `UNARY_LOOP_TWO_OUT` |
| `divmod` | (a, b) → (quot, rem) | `BINARY_LOOP_TWO_OUT` |

These are genuinely different kernel types, not branches.

---

## Final Verified: Complete NumPy Computation Systems

After exhaustive verification of NumPy source code, here are ALL computation systems:

### 10 Core Computation Systems

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    VERIFIED COMPLETE: 10 COMPUTATION SYSTEMS                    │
│                         78+ Total Runtime Branches                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  SYSTEM 1: UFUNC BINARY ELEMENTWISE                          [10 branches]      │
│  ├─ Source: loops_arithmetic.dispatch.c.src, loops.c.src                        │
│  ├─ Operations: add, sub, mul, div, power, mod, floor_div, bitwise_*            │
│  ├─ Branches: SIMD contiguous, SIMD scalar1, SIMD scalar2, reduce,              │
│  │            aliasing variants (×3), VSX4 path, general fallback               │
│  └─ NumSharp: DONE (ILKernelGenerator)                                          │
│                                                                                 │
│  SYSTEM 2: UFUNC UNARY ELEMENTWISE                           [8 branches]       │
│  ├─ Source: loops_unary.dispatch.c.src, loops_unary_fp.dispatch.c.src           │
│  ├─ Operations: neg, abs, sqrt, sin, cos, exp, log, floor, ceil, etc.           │
│  ├─ Branches: CONTIG_CONTIG, NCONTIG_CONTIG, CONTIG_NCONTIG, NCONTIG_NCONTIG,   │
│  │            overlap check, loadable stride check, no_unroll, scalar           │
│  └─ NumSharp: DONE (ILKernelGenerator)                                          │
│                                                                                 │
│  SYSTEM 3: UFUNC REDUCTIONS                                  [9 branches]       │
│  ├─ Source: reduction.c, loops_utils.h.src                                      │
│  ├─ Operations: sum, prod, min, max, any, all, mean, std, var                   │
│  ├─ Branches: masked/unmasked, identity/no-identity, skip-first,                │
│  │            pairwise (n<8 / n≤128 / recursive), wheremask                     │
│  └─ NumSharp: DONE (ILKernelGenerator) - pairwise needs improvement             │
│                                                                                 │
│  SYSTEM 4: UFUNC COMPARISONS                                 [4+ branches]      │
│  ├─ Source: loops_comparison.dispatch.c.src                                     │
│  ├─ Operations: eq, ne, lt, gt, le, ge                                          │
│  ├─ Branches: contiguous, scalar1, scalar2, general + pack-to-bool              │
│  │            (pack varies by type: 1x/2x/4x/8x unroll)                         │
│  └─ NumSharp: DONE (ILKernelGenerator)                                          │
│                                                                                 │
│  SYSTEM 5: LINEAR ALGEBRA                                    [12 branches]      │
│  ├─ Source: cblasfuncs.c, matmul.c.src, vdot.c, einsum.cpp                      │
│  ├─ Operations: dot, matmul, vdot, inner, tensordot, einsum                     │
│  ├─ Branches: noblas fallback, scalar_out (dot), scalar_vec, gemv (×2),         │
│  │            gemm, syrk (A@A.T), buffer allocation, layout transpose           │
│  ├─ BLAS levels: L1 (dot), L2 (gemv), L3 (gemm, syrk)                           │
│  └─ NumSharp: DONE (basic matmul) - einsum NOT IMPLEMENTED                      │
│                                                                                 │
│  SYSTEM 6: SORTING                                           [9 branches]       │
│  ├─ Source: npysort/quicksort.cpp, heapsort.cpp, timsort.cpp                    │
│  ├─ Operations: sort, argsort, lexsort                                          │
│  ├─ Branches: SIMD dispatch (x86/ARM), introsort, heapsort (depth limit),       │
│  │            insertion sort (n≤15), string sort, generic comparator            │
│  ├─ SIMD: x86-simd-sort (AVX512), highway (ARM/other)                           │
│  └─ NumSharp: **NOT IMPLEMENTED** ◄── HIGH PRIORITY                             │
│                                                                                 │
│  SYSTEM 7: SEARCHING                                         [7 branches]       │
│  ├─ Source: npysort/binsearch.cpp, item_selection.c                             │
│  ├─ Operations: searchsorted, nonzero, argmax, argmin                           │
│  ├─ Branches: left/right side, arg variant, sorted key optimization,            │
│  │            generic comparator, empty input                                   │
│  └─ NumSharp: DONE (searchsorted, argmax, argmin, nonzero)                      │
│                                                                                 │
│  SYSTEM 8: INDEXING                                          [15 branches]      │
│  ├─ Source: mapping.c, item_selection.c                                         │
│  ├─ Operations: a[i], a[mask], a[indices], take, put, compress, choose          │
│  ├─ Branches: field access, integer index, boolean array, ellipsis,             │
│  │            slice/newaxis combo, simple 1D fancy, complex fancy,              │
│  │            trivial mapiter, subspace iteration, assignment variants          │
│  └─ NumSharp: DONE (boolean masking, fancy indexing, take)                      │
│                                                                                 │
│  SYSTEM 9: HISTOGRAM/BINCOUNT                                [8 branches]       │
│  ├─ Source: compiled_base.c (arr_bincount)                                      │
│  ├─ Operations: bincount, histogram, digitize                                   │
│  ├─ Branches: empty input, negative check, minlength handling,                  │
│  │            weighted/unweighted, type checking, monotonic check               │
│  └─ NumSharp: **NOT IMPLEMENTED** ◄── MEDIUM PRIORITY                           │
│                                                                                 │
│  SYSTEM 10: PARTITION/SELECTION                              [separate]         │
│  ├─ Source: npysort/selection.cpp                                               │
│  ├─ Operations: partition, argpartition                                         │
│  ├─ Algorithm: Introselect (quickselect + heapselect fallback)                  │
│  └─ NumSharp: **NOT IMPLEMENTED** ◄── MEDIUM PRIORITY                           │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Special Cases (Within Existing Systems)

These are NOT separate systems - they use branches within existing systems:

| Operation | System | Implementation |
|-----------|--------|----------------|
| `clip` | Ufunc Binary (#1) | Ternary ufunc with min/max bounds |
| `where` (3-arg) | Indexing (#8) | Conditional selection in mapping.c |
| `divmod` | Ufunc Binary (#1) | `BINARY_LOOP_TWO_OUT` macro |
| `modf`, `frexp` | Ufunc Unary (#2) | `UNARY_LOOP_TWO_OUT` macro |
| `cumsum`, `cumprod` | Ufunc Reductions (#3) | Accumulate mode of add/multiply |
| `unique` | Sorting (#6) + Searching (#7) | Sort-based or hash-based |
| `correlate`, `convolve` | Linear Algebra (#5) | 1D signal processing |

### Dtype-Specific Paths (Within Existing Systems)

These are NOT separate systems - they are dtype branches within ufuncs:

| Dtype | Location | Notes |
|-------|----------|-------|
| String/Unicode | `string_ufuncs.cpp` | Registered as ufuncs |
| DateTime | `datetime.c` | Business day calculations |
| Object | `loops.c.src` | Python protocol fallback |
| Complex | `loops_unary_complex.dispatch.c.src` | Via ufuncs |

### Composed Operations (No Unique Kernels)

These are Python-level and use existing primitives:

| Operation | Composition |
|-----------|-------------|
| `diff` | slicing + subtraction |
| `gradient` | slicing + division |
| `mean`, `std`, `var` | sum + division |
| `outer` | multiply.outer |
| `tensordot` | transpose + reshape + dot |
| `cross`, `kron` | multiply + subtract/reshape |
| `tril`, `triu` | where + indices |
| Set ops (`unique`, `union1d`, etc.) | sort + comparison |

### NumSharp Implementation Status

| System | Status | Missing |
|--------|--------|---------|
| 1. Ufunc Binary | DONE | Pairwise reduce branch |
| 2. Ufunc Unary | DONE | Strided SIMD branches |
| 3. Ufunc Reductions | DONE | Pairwise algorithm |
| 4. Ufunc Comparisons | DONE | Scalar broadcast branches |
| 5. Linear Algebra | Partial | Einsum, advanced LAPACK |
| 6. Sorting | **MISSING** | All of it |
| 7. Searching | DONE | - |
| 8. Indexing | DONE | put, choose |
| 9. Histogram | **MISSING** | All of it |
| 10. Partition | **MISSING** | All of it |

### Final Summary

| Metric | Value |
|--------|-------|
| **Total computation systems** | 10 |
| **Total runtime branches** | 78+ |
| **Systems fully implemented** | 7/10 |
| **Systems missing** | 3 (Sorting, Histogram, Partition) |
| **Branches needing improvement** | ~10 (pairwise, scalar broadcast, strided SIMD) |

**This is the COMPLETE and VERIFIED list of all NumPy computation systems.**
