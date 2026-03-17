# SIMD Optimization for NaN Statistics Functions

## Overview

This document describes how to implement SIMD-optimized versions of `nanmean`, `nanvar`, and `nanstd` for NumSharp. These functions currently use scalar loops because they need to track the **count** of non-NaN values, which is not a simple reduction operation.

## Current State (Scalar Implementation)

### Problem

The current `NanMeanSimdHelperDouble` implementation:

```csharp
internal static unsafe double NanMeanSimdHelperDouble(double* src, long size)
{
    double sum = 0.0;
    long count = 0;

    for (long i = 0; i < size; i++)
    {
        if (!double.IsNaN(src[i]))
        {
            sum += src[i];
            count++;
        }
    }

    return count > 0 ? sum / count : double.NaN;
}
```

**Performance**: ~1 element per cycle (scalar, branch-heavy)

### Why It's Slow

1. **Branch per element**: `if (!IsNaN)` causes branch mispredictions
2. **No vectorization**: Processes 1 element at a time instead of 4-8
3. **Memory bandwidth underutilized**: CPU can load 256 bits but only uses 64 bits

## SIMD Approach

### Key Insight: NaN Self-Comparison

In IEEE 754 floating-point:
- `x == x` is `true` for all normal values
- `x == x` is `false` for NaN

This property allows branchless NaN detection using SIMD comparison.

### SIMD Building Blocks

#### 1. Create NaN Mask
```csharp
var vec = Vector256.Load(src + i);
var nanMask = Vector256.Equals(vec, vec);  // All 1s for non-NaN, all 0s for NaN
```

Result for `[1.0, NaN, 3.0, NaN]`:
```
vec:     [1.0,        NaN,        3.0,        NaN       ]
nanMask: [0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000]  (as float bits)
```

#### 2. Zero Out NaN Values (for sum)
```csharp
var cleaned = Vector256.BitwiseAnd(vec, nanMask.AsDouble());
```

Result:
```
cleaned: [1.0, 0.0, 3.0, 0.0]  // NaN replaced with 0
```

#### 3. Count Non-NaN Values

**Method A: Use 1.0 mask and sum**
```csharp
var oneVec = Vector256.Create(1.0);
var countMask = Vector256.BitwiseAnd(oneVec, nanMask.AsDouble());
// countMask: [1.0, 0.0, 1.0, 0.0]
countVec = Vector256.Add(countVec, countMask);
```

**Method B: Convert mask to int and popcount** (more complex but potentially faster)
```csharp
// Extract comparison result as integer mask
int mask = Vector256.ExtractMostSignificantBits(nanMask);
int count = BitOperations.PopCount((uint)mask);
```

### Algorithm: SIMD NanMean

```
Input:  double* data, long size
Output: double mean (or NaN if all NaN)

1. Initialize:
   sumVec = Vector256<double>.Zero      // [0, 0, 0, 0]
   countVec = Vector256<double>.Zero    // [0, 0, 0, 0]
   oneVec = Vector256.Create(1.0)       // [1, 1, 1, 1]

2. SIMD Loop (4 doubles per iteration for Vector256):
   for i = 0 to size - 4 step 4:
       vec = Load(data + i)                           // Load 4 doubles
       nanMask = Equals(vec, vec)                     // True for non-NaN
       cleaned = BitwiseAnd(vec, nanMask.AsDouble())  // Zero out NaN
       sumVec = Add(sumVec, cleaned)                  // Accumulate sum
       countMask = BitwiseAnd(oneVec, nanMask.AsDouble())
       countVec = Add(countVec, countMask)            // Accumulate count

3. Horizontal Reduction:
   sum = sumVec[0] + sumVec[1] + sumVec[2] + sumVec[3]
   count = countVec[0] + countVec[1] + countVec[2] + countVec[3]

4. Scalar Tail (remaining 0-3 elements):
   for i = vectorEnd to size:
       if !IsNaN(data[i]):
           sum += data[i]
           count += 1

5. Return:
   return count > 0 ? sum / count : NaN
```

### Algorithm: SIMD NanVar (Two-Pass)

Variance requires two passes:
1. **Pass 1**: Compute mean (using NanMean algorithm)
2. **Pass 2**: Compute sum of squared differences from mean

```
Input:  double* data, long size, int ddof
Output: double variance (or NaN)

=== Pass 1: Compute Mean ===
(Same as NanMean algorithm above)
mean = sum / count

=== Pass 2: Sum Squared Differences ===

1. Initialize:
   sqDiffVec = Vector256<double>.Zero
   meanVec = Vector256.Create(mean)      // Broadcast mean to all lanes

2. SIMD Loop:
   for i = 0 to size - 4 step 4:
       vec = Load(data + i)
       nanMask = Equals(vec, vec)

       // Compute (x - mean)^2, but zero for NaN
       diff = Subtract(vec, meanVec)
       sqDiff = Multiply(diff, diff)
       cleanedSqDiff = BitwiseAnd(sqDiff, nanMask.AsDouble())
       sqDiffVec = Add(sqDiffVec, cleanedSqDiff)

3. Horizontal Reduction:
   sqDiffSum = sqDiffVec[0] + sqDiffVec[1] + sqDiffVec[2] + sqDiffVec[3]

4. Scalar Tail:
   for i = vectorEnd to size:
       if !IsNaN(data[i]):
           diff = data[i] - mean
           sqDiffSum += diff * diff

5. Return:
   divisor = count - ddof
   return divisor > 0 ? sqDiffSum / divisor : NaN
```

### Algorithm: SIMD NanStd

Simply compute variance, then take square root:

```csharp
double variance = NanVarSimd(data, size, ddof);
return double.IsNaN(variance) ? double.NaN : Math.Sqrt(variance);
```

## Implementation

### File: `ILKernelGenerator.Masking.NaN.cs`

#### NanMeanSimdHelperDouble (SIMD Version)

```csharp
internal static unsafe double NanMeanSimdHelperDouble(double* src, long size)
{
    if (size == 0)
        return double.NaN;

    double sum = 0.0;
    double count = 0.0;

    if (Vector256.IsHardwareAccelerated && Vector256<double>.IsSupported && size >= Vector256<double>.Count)
    {
        int vectorCount = Vector256<double>.Count;  // 4 for double
        long vectorEnd = size - vectorCount;
        var sumVec = Vector256<double>.Zero;
        var countVec = Vector256<double>.Zero;
        var oneVec = Vector256.Create(1.0);
        long i = 0;

        for (; i <= vectorEnd; i += vectorCount)
        {
            var vec = Vector256.Load(src + i);
            var nanMask = Vector256.Equals(vec, vec);  // True for non-NaN

            // Sum: zero out NaN values
            var cleaned = Vector256.BitwiseAnd(vec, nanMask.AsDouble());
            sumVec = Vector256.Add(sumVec, cleaned);

            // Count: add 1.0 for each non-NaN
            var countMask = Vector256.BitwiseAnd(oneVec, nanMask.AsDouble());
            countVec = Vector256.Add(countVec, countMask);
        }

        // Horizontal reduction
        sum = Vector256.Sum(sumVec);
        count = Vector256.Sum(countVec);

        // Scalar tail
        for (; i < size; i++)
        {
            if (!double.IsNaN(src[i]))
            {
                sum += src[i];
                count += 1.0;
            }
        }
    }
    else if (Vector128.IsHardwareAccelerated && Vector128<double>.IsSupported && size >= Vector128<double>.Count)
    {
        int vectorCount = Vector128<double>.Count;  // 2 for double
        long vectorEnd = size - vectorCount;
        var sumVec = Vector128<double>.Zero;
        var countVec = Vector128<double>.Zero;
        var oneVec = Vector128.Create(1.0);
        long i = 0;

        for (; i <= vectorEnd; i += vectorCount)
        {
            var vec = Vector128.Load(src + i);
            var nanMask = Vector128.Equals(vec, vec);

            var cleaned = Vector128.BitwiseAnd(vec, nanMask.AsDouble());
            sumVec = Vector128.Add(sumVec, cleaned);

            var countMask = Vector128.BitwiseAnd(oneVec, nanMask.AsDouble());
            countVec = Vector128.Add(countVec, countMask);
        }

        sum = Vector128.Sum(sumVec);
        count = Vector128.Sum(countVec);

        for (; i < size; i++)
        {
            if (!double.IsNaN(src[i]))
            {
                sum += src[i];
                count += 1.0;
            }
        }
    }
    else
    {
        // Scalar fallback
        for (long i = 0; i < size; i++)
        {
            if (!double.IsNaN(src[i]))
            {
                sum += src[i];
                count += 1.0;
            }
        }
    }

    return count > 0 ? sum / count : double.NaN;
}
```

#### NanVarSimdHelperDouble (SIMD Version)

```csharp
internal static unsafe double NanVarSimdHelperDouble(double* src, long size, int ddof = 0)
{
    if (size == 0)
        return double.NaN;

    // === Pass 1: Compute sum and count ===
    double sum = 0.0;
    double count = 0.0;

    if (Vector256.IsHardwareAccelerated && Vector256<double>.IsSupported && size >= Vector256<double>.Count)
    {
        int vectorCount = Vector256<double>.Count;
        long vectorEnd = size - vectorCount;
        var sumVec = Vector256<double>.Zero;
        var countVec = Vector256<double>.Zero;
        var oneVec = Vector256.Create(1.0);
        long i = 0;

        for (; i <= vectorEnd; i += vectorCount)
        {
            var vec = Vector256.Load(src + i);
            var nanMask = Vector256.Equals(vec, vec);
            sumVec = Vector256.Add(sumVec, Vector256.BitwiseAnd(vec, nanMask.AsDouble()));
            countVec = Vector256.Add(countVec, Vector256.BitwiseAnd(oneVec, nanMask.AsDouble()));
        }

        sum = Vector256.Sum(sumVec);
        count = Vector256.Sum(countVec);

        for (; i < size; i++)
        {
            if (!double.IsNaN(src[i]))
            {
                sum += src[i];
                count += 1.0;
            }
        }
    }
    else
    {
        for (long i = 0; i < size; i++)
        {
            if (!double.IsNaN(src[i]))
            {
                sum += src[i];
                count += 1.0;
            }
        }
    }

    if (count <= ddof)
        return double.NaN;

    double mean = sum / count;

    // === Pass 2: Sum of squared differences ===
    double sqDiffSum = 0.0;

    if (Vector256.IsHardwareAccelerated && Vector256<double>.IsSupported && size >= Vector256<double>.Count)
    {
        int vectorCount = Vector256<double>.Count;
        long vectorEnd = size - vectorCount;
        var sqDiffVec = Vector256<double>.Zero;
        var meanVec = Vector256.Create(mean);
        long i = 0;

        for (; i <= vectorEnd; i += vectorCount)
        {
            var vec = Vector256.Load(src + i);
            var nanMask = Vector256.Equals(vec, vec);

            // (x - mean)^2, zeroed for NaN
            var diff = Vector256.Subtract(vec, meanVec);
            var sqDiff = Vector256.Multiply(diff, diff);
            var cleaned = Vector256.BitwiseAnd(sqDiff, nanMask.AsDouble());
            sqDiffVec = Vector256.Add(sqDiffVec, cleaned);
        }

        sqDiffSum = Vector256.Sum(sqDiffVec);

        for (; i < size; i++)
        {
            if (!double.IsNaN(src[i]))
            {
                double diff = src[i] - mean;
                sqDiffSum += diff * diff;
            }
        }
    }
    else
    {
        for (long i = 0; i < size; i++)
        {
            if (!double.IsNaN(src[i]))
            {
                double diff = src[i] - mean;
                sqDiffSum += diff * diff;
            }
        }
    }

    return sqDiffSum / (count - ddof);
}
```

### Float Versions

The float versions are identical but use:
- `Vector256<float>` (8 elements per vector)
- `Vector128<float>` (4 elements per vector)
- `float.IsNaN()` for scalar tail

```csharp
internal static unsafe float NanMeanSimdHelperFloat(float* src, long size)
{
    // Same algorithm, but:
    // - Vector256<float>.Count = 8
    // - Vector128<float>.Count = 4
    // - Return (float)(sum / count)
}
```

## Axis Reduction SIMD

The axis reduction kernels in `ILKernelGenerator.Reduction.Axis.NaN.cs` can be similarly optimized.

### Current State

`NanStatReduceContiguousAxis` uses scalar loops:

```csharp
private static unsafe T NanStatReduceContiguousAxis<T>(T* data, long size, ReductionOp op)
{
    // Pass 1: scalar sum and count
    for (long i = 0; i < size; i++) { ... }

    // Pass 2: scalar squared differences
    for (long i = 0; i < size; i++) { ... }
}
```

### SIMD Axis Reduction

Create type-specific implementations:

```csharp
private static unsafe double NanMeanReduceContiguousAxisDouble(double* data, long size)
{
    // Use Vector256<double> SIMD loop
    // Same algorithm as NanMeanSimdHelperDouble
}

private static unsafe double NanVarReduceContiguousAxisDouble(double* data, long size)
{
    // Two-pass SIMD algorithm
}

private static unsafe double NanStdReduceContiguousAxisDouble(double* data, long size)
{
    return Math.Sqrt(NanVarReduceContiguousAxisDouble(data, size));
}
```

Then dispatch in `NanStatReduceContiguousAxis`:

```csharp
private static unsafe T NanStatReduceContiguousAxis<T>(T* data, long size, ReductionOp op)
    where T : unmanaged, IFloatingPoint<T>
{
    if (typeof(T) == typeof(double))
    {
        double result = op switch
        {
            ReductionOp.NanMean => NanMeanReduceContiguousAxisDouble((double*)(void*)data, size),
            ReductionOp.NanVar => NanVarReduceContiguousAxisDouble((double*)(void*)data, size),
            ReductionOp.NanStd => NanStdReduceContiguousAxisDouble((double*)(void*)data, size),
            _ => throw new NotSupportedException()
        };
        return T.CreateTruncating(result);
    }

    if (typeof(T) == typeof(float))
    {
        float result = op switch
        {
            ReductionOp.NanMean => NanMeanReduceContiguousAxisFloat((float*)(void*)data, size),
            ReductionOp.NanVar => NanVarReduceContiguousAxisFloat((float*)(void*)data, size),
            ReductionOp.NanStd => NanStdReduceContiguousAxisFloat((float*)(void*)data, size),
            _ => throw new NotSupportedException()
        };
        return T.CreateTruncating(result);
    }

    throw new NotSupportedException();
}
```

## Performance Expectations

### Throughput Improvement

| Operation | Scalar | Vector256 (double) | Speedup |
|-----------|--------|-------------------|---------|
| NanMean   | 1 elem/cycle | 4 elem/cycle | ~4x |
| NanVar    | 1 elem/cycle | 4 elem/cycle | ~4x |
| NanStd    | 1 elem/cycle | 4 elem/cycle | ~4x |

For float (Vector256 = 8 elements):

| Operation | Scalar | Vector256 (float) | Speedup |
|-----------|--------|-------------------|---------|
| NanMean   | 1 elem/cycle | 8 elem/cycle | ~8x |
| NanVar    | 1 elem/cycle | 8 elem/cycle | ~8x |
| NanStd    | 1 elem/cycle | 8 elem/cycle | ~8x |

### Memory Bandwidth

For large arrays, SIMD also improves memory bandwidth utilization:
- Scalar: Loads 8 bytes (1 double), processes 8 bytes
- Vector256: Loads 32 bytes (4 doubles), processes 32 bytes

This better utilizes the memory subsystem's prefetching and cache line efficiency.

## Testing

### Correctness Tests

```csharp
[Test]
public void NanMean_Simd_MatchesScalar()
{
    var data = new double[] { 1, 2, double.NaN, 4, 5, double.NaN, 7, 8 };

    // Expected: (1+2+4+5+7+8) / 6 = 27/6 = 4.5
    var arr = np.array(data);
    var result = np.nanmean(arr);

    Assert.That(result.GetDouble(), Is.EqualTo(4.5).Within(1e-10));
}

[Test]
public void NanVar_Simd_MatchesNumPy()
{
    var data = new double[] { 1, 2, double.NaN, 3, 4 };

    // NumPy: np.nanvar([1,2,3,4]) = 1.25
    var arr = np.array(data);
    var result = np.nanvar(arr);

    Assert.That(result.GetDouble(), Is.EqualTo(1.25).Within(1e-10));
}

[Test]
public void NanMean_AllNaN_ReturnsNaN()
{
    var data = new double[] { double.NaN, double.NaN, double.NaN };
    var arr = np.array(data);
    var result = np.nanmean(arr);

    Assert.That(double.IsNaN(result.GetDouble()));
}
```

### Performance Benchmarks

```csharp
[Benchmark]
public double NanMean_1M_Elements()
{
    return np.nanmean(_largeArrayWithNaNs).GetDouble();
}

[Benchmark]
public double NanVar_1M_Elements()
{
    return np.nanvar(_largeArrayWithNaNs).GetDouble();
}
```

## Implementation Checklist

- [ ] `NanMeanSimdHelperDouble` - Vector256/128 implementation
- [ ] `NanMeanSimdHelperFloat` - Vector256/128 implementation
- [ ] `NanVarSimdHelperDouble` - Two-pass Vector256/128 implementation
- [ ] `NanVarSimdHelperFloat` - Two-pass Vector256/128 implementation
- [ ] `NanStdSimdHelperDouble` - Calls NanVar + sqrt
- [ ] `NanStdSimdHelperFloat` - Calls NanVar + sqrt
- [ ] `NanMeanReduceContiguousAxisDouble` - Axis reduction SIMD
- [ ] `NanMeanReduceContiguousAxisFloat` - Axis reduction SIMD
- [ ] `NanVarReduceContiguousAxisDouble` - Axis reduction SIMD
- [ ] `NanVarReduceContiguousAxisFloat` - Axis reduction SIMD
- [ ] Unit tests for correctness
- [ ] Benchmarks comparing to scalar and NumPy

## Advanced Optimizations (Future)

### 1. Loop Unrolling

Process 2-4 vectors per iteration to hide latency:

```csharp
for (; i <= vectorEnd - 4 * vectorCount; i += 4 * vectorCount)
{
    var vec0 = Vector256.Load(src + i);
    var vec1 = Vector256.Load(src + i + vectorCount);
    var vec2 = Vector256.Load(src + i + 2 * vectorCount);
    var vec3 = Vector256.Load(src + i + 3 * vectorCount);

    // Process all 4 vectors
    sumVec0 = Vector256.Add(sumVec0, ...);
    sumVec1 = Vector256.Add(sumVec1, ...);
    sumVec2 = Vector256.Add(sumVec2, ...);
    sumVec3 = Vector256.Add(sumVec3, ...);
}

// Combine accumulators
sumVec = Vector256.Add(Vector256.Add(sumVec0, sumVec1), Vector256.Add(sumVec2, sumVec3));
```

### 2. Welford's Algorithm (Single Pass Variance)

For very large arrays, the two-pass algorithm requires reading data twice. Welford's online algorithm computes variance in a single pass:

```csharp
// Welford's algorithm (numerically stable)
double mean = 0.0;
double M2 = 0.0;
long count = 0;

for (long i = 0; i < size; i++)
{
    if (!double.IsNaN(src[i]))
    {
        count++;
        double delta = src[i] - mean;
        mean += delta / count;
        double delta2 = src[i] - mean;
        M2 += delta * delta2;
    }
}

double variance = M2 / (count - ddof);
```

This is harder to vectorize due to the sequential dependency on `mean`, but can be done with parallel Welford (combining partial results).

### 3. Vector512 Support

For AVX-512 capable CPUs:

```csharp
if (Vector512.IsHardwareAccelerated && Vector512<double>.IsSupported && size >= 8)
{
    // 8 doubles per vector - 2x throughput vs Vector256
}
```

## Summary

SIMD optimization for `nanmean`/`nanvar`/`nanstd` is straightforward:

1. Use `Equals(vec, vec)` to create NaN mask
2. Use `BitwiseAnd` to zero out NaN values for sum
3. Use `BitwiseAnd` with ones vector to count non-NaN
4. Process 4 doubles (Vector256) or 8 floats (Vector256) per iteration
5. Expected speedup: **4-8x** depending on element type

The key insight is that counting can be done as a sum of 1.0s, which fits the standard SIMD reduction pattern.
