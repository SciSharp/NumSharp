using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NumSharp.Benchmark.Exploration.Infrastructure;

namespace NumSharp.Benchmark.Exploration.Isolated;

/// <summary>
/// Compare different SIMD strategies for row broadcast scenario.
/// This is the most complex common scenario where multiple strategies are viable.
/// </summary>
public static unsafe class SimdStrategies
{
    private const string Suite = "SimdStrategies";

    /// <summary>
    /// Run all strategy comparisons.
    /// </summary>
    public static List<BenchResult> RunAll(bool quick = false)
    {
        var results = new List<BenchResult>();
        var warmup = quick ? 2 : BenchFramework.DefaultWarmup;
        var measure = quick ? 5 : BenchFramework.DefaultMeasure;

        // Test with different matrix shapes
        var shapes = quick
            ? new[] { (100, 100), (1000, 1000) }
            : new[] { (100, 100), (316, 316), (500, 500), (1000, 1000), (3162, 3162) };

        BenchFramework.PrintHeader($"{Suite}: Row Broadcast Strategy Comparison");

        foreach (var (rows, cols) in shapes)
        {
            BenchFramework.PrintDivider($"Shape: ({rows}, {cols}) = {rows * cols:N0} elements");
            results.AddRange(CompareStrategies_Float64(rows, cols, warmup, measure));
        }

        OutputFormatters.PrintSummary(results);
        return results;
    }

    /// <summary>
    /// Compare all strategies for row broadcast with float64.
    /// matrix[M, N] + row[N] = result[M, N]
    /// </summary>
    public static List<BenchResult> CompareStrategies_Float64(int rows, int cols, int warmup, int measure)
    {
        var results = new List<BenchResult>();
        int size = rows * cols;
        int elementBytes = 8;

        var matrix = SimdImplementations.AllocateAligned<double>(size);
        var row = SimdImplementations.AllocateAligned<double>(cols);
        var result = SimdImplementations.AllocateAligned<double>(size);
        SimdImplementations.FillRandom(matrix, size, 42);
        SimdImplementations.FillRandom(row, cols, 43);

        // Strategy 1: Pure scalar with GetOffset-like indexing
        var s1 = BenchFramework.Run(
            () => Strategy1_ScalarGetOffset(matrix, row, result, rows, cols),
            "S4_rowBC", "1_ScalarOfs", "float64", size, elementBytes, warmup, measure, Suite);
        results.Add(s1);
        BenchFramework.PrintResult(s1);

        // Strategy 2: Nested loops with scalar inner
        var s2 = BenchFramework.Run(
            () => Strategy2_NestedScalar(matrix, row, result, rows, cols),
            "S4_rowBC", "2_Nested", "float64", size, elementBytes, warmup, measure, Suite);
        s2 = s2 with { SpeedupVsBaseline = s1.MeanUs / s2.MeanUs };
        results.Add(s2);
        BenchFramework.PrintResult(s2);

        // Strategy 3: Nested loops with SIMD inner (SIMD-CHUNK)
        var s3 = BenchFramework.Run(
            () => Strategy3_NestedSimdInner(matrix, row, result, rows, cols),
            "S4_rowBC", "3_SimdChunk", "float64", size, elementBytes, warmup, measure, Suite);
        s3 = s3 with { SpeedupVsBaseline = s1.MeanUs / s3.MeanUs };
        results.Add(s3);
        BenchFramework.PrintResult(s3);

        // Strategy 4: Pretend contiguous (incorrect but shows ceiling)
        // This strategy is NOT correct for broadcasting but shows max SIMD throughput
        var s4 = BenchFramework.Run(
            () => Strategy4_FlatSimd(matrix, row, result, rows, cols),
            "S4_rowBC", "4_FlatSimd*", "float64", size, elementBytes, warmup, measure, Suite);
        s4 = s4 with { SpeedupVsBaseline = s1.MeanUs / s4.MeanUs, Notes = "INCORRECT - ceiling only" };
        results.Add(s4);
        BenchFramework.PrintResult(s4);

        // Strategy 5: Copy row to temp buffer, use flat SIMD
        var rowExpanded = SimdImplementations.AllocateAligned<double>(size);
        var s5 = BenchFramework.Run(
            () => Strategy5_ExpandThenSimd(matrix, row, rowExpanded, result, rows, cols),
            "S4_rowBC", "5_Expand", "float64", size, elementBytes, warmup, measure, Suite);
        s5 = s5 with { SpeedupVsBaseline = s1.MeanUs / s5.MeanUs };
        results.Add(s5);
        BenchFramework.PrintResult(s5);

        // Strategy 6: Unrolled SIMD inner loop
        var s6 = BenchFramework.Run(
            () => Strategy6_UnrolledSimdInner(matrix, row, result, rows, cols),
            "S4_rowBC", "6_Unrolled", "float64", size, elementBytes, warmup, measure, Suite);
        s6 = s6 with { SpeedupVsBaseline = s1.MeanUs / s6.MeanUs };
        results.Add(s6);
        BenchFramework.PrintResult(s6);

        SimdImplementations.FreeAligned(matrix);
        SimdImplementations.FreeAligned(row);
        SimdImplementations.FreeAligned(result);
        SimdImplementations.FreeAligned(rowExpanded);

        return results;
    }

    #region Strategy Implementations

    /// <summary>
    /// Strategy 1: Pure scalar with linear index calculation (simulates GetOffset).
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void Strategy1_ScalarGetOffset(double* matrix, double* row, double* result, int rows, int cols)
    {
        for (int i = 0; i < rows * cols; i++)
        {
            int c = i % cols; // Simulates GetOffset calculation
            result[i] = matrix[i] + row[c];
        }
    }

    /// <summary>
    /// Strategy 2: Nested loops with direct indexing, scalar inner.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void Strategy2_NestedScalar(double* matrix, double* row, double* result, int rows, int cols)
    {
        for (int r = 0; r < rows; r++)
        {
            var offset = r * cols;
            for (int c = 0; c < cols; c++)
            {
                result[offset + c] = matrix[offset + c] + row[c];
            }
        }
    }

    /// <summary>
    /// Strategy 3: Nested loops with SIMD inner (SIMD-CHUNK strategy).
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void Strategy3_NestedSimdInner(double* matrix, double* row, double* result, int rows, int cols)
    {
        for (int r = 0; r < rows; r++)
        {
            var matrixRow = matrix + r * cols;
            var resultRow = result + r * cols;

            int c = 0;
            if (Avx2.IsSupported && cols >= Vector256<double>.Count)
            {
                int vectorCount = cols - Vector256<double>.Count + 1;
                for (; c < vectorCount; c += Vector256<double>.Count)
                {
                    var vm = Avx.LoadVector256(matrixRow + c);
                    var vr = Avx.LoadVector256(row + c);
                    var vres = Avx.Add(vm, vr);
                    Avx.Store(resultRow + c, vres);
                }
            }

            // Scalar tail
            for (; c < cols; c++)
            {
                resultRow[c] = matrixRow[c] + row[c];
            }
        }
    }

    /// <summary>
    /// Strategy 4: Flat SIMD ignoring broadcast (INCORRECT but shows ceiling).
    /// This pretends both arrays are the same shape - shows max possible throughput.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void Strategy4_FlatSimd(double* matrix, double* row, double* result, int rows, int cols)
    {
        // NOTE: This is INCORRECT for broadcasting - it ignores row repetition
        // It's only here to show the maximum throughput ceiling
        int size = rows * cols;
        SimdImplementations.AddFull_Float64(matrix, matrix, result, size); // Using matrix twice
    }

    /// <summary>
    /// Strategy 5: Expand row to full matrix, then flat SIMD.
    /// Trades memory for simpler SIMD loop.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void Strategy5_ExpandThenSimd(double* matrix, double* row, double* rowExpanded, double* result, int rows, int cols)
    {
        // Step 1: Expand row to full matrix
        for (int r = 0; r < rows; r++)
        {
            Buffer.MemoryCopy(row, rowExpanded + r * cols, cols * sizeof(double), cols * sizeof(double));
        }

        // Step 2: Flat SIMD add
        SimdImplementations.AddFull_Float64(matrix, rowExpanded, result, rows * cols);
    }

    /// <summary>
    /// Strategy 6: Unrolled SIMD inner loop (2x unroll).
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void Strategy6_UnrolledSimdInner(double* matrix, double* row, double* result, int rows, int cols)
    {
        const int Unroll = 2;
        int vectorSize = Vector256<double>.Count;
        int unrollSize = vectorSize * Unroll;

        for (int r = 0; r < rows; r++)
        {
            var matrixRow = matrix + r * cols;
            var resultRow = result + r * cols;

            int c = 0;

            if (Avx2.IsSupported && cols >= unrollSize)
            {
                int vectorCount = cols - unrollSize + 1;
                for (; c < vectorCount; c += unrollSize)
                {
                    // Load 2 vectors from matrix
                    var vm0 = Avx.LoadVector256(matrixRow + c);
                    var vm1 = Avx.LoadVector256(matrixRow + c + vectorSize);

                    // Load 2 vectors from row
                    var vr0 = Avx.LoadVector256(row + c);
                    var vr1 = Avx.LoadVector256(row + c + vectorSize);

                    // Add
                    var vres0 = Avx.Add(vm0, vr0);
                    var vres1 = Avx.Add(vm1, vr1);

                    // Store
                    Avx.Store(resultRow + c, vres0);
                    Avx.Store(resultRow + c + vectorSize, vres1);
                }
            }

            // Non-unrolled SIMD
            if (Avx2.IsSupported)
            {
                int vectorCount = cols - vectorSize + 1;
                for (; c < vectorCount; c += vectorSize)
                {
                    var vm = Avx.LoadVector256(matrixRow + c);
                    var vr = Avx.LoadVector256(row + c);
                    var vres = Avx.Add(vm, vr);
                    Avx.Store(resultRow + c, vres);
                }
            }

            // Scalar tail
            for (; c < cols; c++)
            {
                resultRow[c] = matrixRow[c] + row[c];
            }
        }
    }

    #endregion

    /// <summary>
    /// Test Vector256 vs Vector128 performance.
    /// </summary>
    public static List<BenchResult> CompareVectorWidths(int size, int warmup, int measure)
    {
        var results = new List<BenchResult>();
        int elementBytes = 8;

        var lhs = SimdImplementations.AllocateAligned<double>(size);
        var rhs = SimdImplementations.AllocateAligned<double>(size);
        var result = SimdImplementations.AllocateAligned<double>(size);
        SimdImplementations.FillRandom(lhs, size, 42);
        SimdImplementations.FillRandom(rhs, size, 43);

        BenchFramework.PrintDivider($"Vector Width Comparison (size={size:N0})");

        // Scalar baseline
        var scalar = BenchFramework.Run(
            () => SimdImplementations.AddScalarLoop_Float64(lhs, rhs, result, size),
            "VecWidth", "Scalar", "float64", size, elementBytes, warmup, measure, Suite);
        results.Add(scalar);
        BenchFramework.PrintResult(scalar);

        // Vector128
        var v128 = BenchFramework.Run(
            () => AddVector128(lhs, rhs, result, size),
            "VecWidth", "Vector128", "float64", size, elementBytes, warmup, measure, Suite);
        v128 = v128 with { SpeedupVsBaseline = scalar.MeanUs / v128.MeanUs };
        results.Add(v128);
        BenchFramework.PrintResult(v128);

        // Vector256
        var v256 = BenchFramework.Run(
            () => SimdImplementations.AddFull_Float64(lhs, rhs, result, size),
            "VecWidth", "Vector256", "float64", size, elementBytes, warmup, measure, Suite);
        v256 = v256 with { SpeedupVsBaseline = scalar.MeanUs / v256.MeanUs };
        results.Add(v256);
        BenchFramework.PrintResult(v256);

        SimdImplementations.FreeAligned(lhs);
        SimdImplementations.FreeAligned(rhs);
        SimdImplementations.FreeAligned(result);

        return results;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AddVector128(double* lhs, double* rhs, double* result, int count)
    {
        int i = 0;

        if (Sse2.IsSupported && count >= Vector128<double>.Count)
        {
            int vectorCount = count - Vector128<double>.Count + 1;
            for (; i < vectorCount; i += Vector128<double>.Count)
            {
                var va = Sse2.LoadVector128(lhs + i);
                var vb = Sse2.LoadVector128(rhs + i);
                var vr = Sse2.Add(va, vb);
                Sse2.Store(result + i, vr);
            }
        }

        for (; i < count; i++)
        {
            result[i] = lhs[i] + rhs[i];
        }
    }
}
