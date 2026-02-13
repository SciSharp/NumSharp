using System.Runtime.InteropServices;
using NumSharp.Benchmark.Exploration.Infrastructure;

namespace NumSharp.Benchmark.Exploration.Isolated;

/// <summary>
/// Benchmark all 7 broadcasting scenarios to identify which benefit from SIMD.
///
/// Scenarios:
/// S1: Both contiguous, same shape — SIMD-FULL
/// S2: Scalar broadcast (right) — SIMD-SCALAR
/// S3: Scalar broadcast (left) — SIMD-SCALAR
/// S4: Row broadcast (M,N) + (N,) — SIMD-CHUNK
/// S5: Column broadcast (M,N) + (M,1) — SIMD per-row scalar
/// S6: Mutual broadcast (M,1) + (1,N) — Outer product
/// S7: Strided arrays a[::2] + b[::2] — SCALAR/GATHER
/// </summary>
public static unsafe class BroadcastScenarios
{
    private const string Suite = "BroadcastScenarios";

    /// <summary>
    /// Run all broadcast scenario benchmarks.
    /// </summary>
    public static List<BenchResult> RunAll(int[]? sizes = null, string[]? dtypes = null, bool quick = false)
    {
        sizes ??= quick ? ArraySizes.Quick : ArraySizes.Standard;
        dtypes ??= quick ? Dtypes.Common : Dtypes.All;
        var warmup = quick ? 2 : BenchFramework.DefaultWarmup;
        var measure = quick ? BenchFramework.QuickMeasure : BenchFramework.DefaultMeasure;

        var results = new List<BenchResult>();

        BenchFramework.PrintHeader($"{Suite}: All 7 Broadcasting Scenarios");

        foreach (var dtype in dtypes)
        {
            BenchFramework.PrintDivider($"dtype={dtype}");

            foreach (var size in sizes)
            {
                // S1: Both contiguous, same shape
                results.AddRange(RunS1_Contiguous(dtype, size, warmup, measure));

                // S2: Scalar broadcast (right)
                results.AddRange(RunS2_ScalarRight(dtype, size, warmup, measure));

                // S3: Scalar broadcast (left) — same as S2 due to commutativity
                // Skipped for brevity, same perf characteristics

                // S4: Row broadcast
                results.AddRange(RunS4_RowBroadcast(dtype, size, warmup, measure));

                // S5: Column broadcast
                results.AddRange(RunS5_ColBroadcast(dtype, size, warmup, measure));
            }
        }

        OutputFormatters.PrintSummary(results);
        return results;
    }

    #region S1: Contiguous same shape

    /// <summary>
    /// S1: Both operands contiguous with same shape — ideal for SIMD-FULL.
    /// </summary>
    public static List<BenchResult> RunS1_Contiguous(string dtype, int size, int warmup, int measure)
    {
        var results = new List<BenchResult>();
        var elementBytes = DtypeInfo.GetElementSize(dtype);

        switch (dtype.ToLowerInvariant())
        {
            case "float64" or "double":
            {
                var lhs = SimdImplementations.AllocateAligned<double>(size);
                var rhs = SimdImplementations.AllocateAligned<double>(size);
                var result = SimdImplementations.AllocateAligned<double>(size);
                SimdImplementations.FillRandom(lhs, size, 42);
                SimdImplementations.FillRandom(rhs, size, 43);

                // Baseline: scalar loop
                var baseline = BenchFramework.Run(
                    () => SimdImplementations.AddScalarLoop_Float64(lhs, rhs, result, size),
                    "S1_contiguous", "Scalar", dtype, size, elementBytes, warmup, measure, Suite);
                results.Add(baseline);
                BenchFramework.PrintResult(baseline);

                // SIMD-FULL
                var simd = BenchFramework.Run(
                    () => SimdImplementations.AddFull_Float64(lhs, rhs, result, size),
                    "S1_contiguous", "SIMD-FULL", dtype, size, elementBytes, warmup, measure, Suite);
                simd = simd with { SpeedupVsBaseline = baseline.MeanUs / simd.MeanUs };
                results.Add(simd);
                BenchFramework.PrintResult(simd);

                SimdImplementations.FreeAligned(lhs);
                SimdImplementations.FreeAligned(rhs);
                SimdImplementations.FreeAligned(result);
                break;
            }

            case "float32" or "single" or "float":
            {
                var lhs = SimdImplementations.AllocateAligned<float>(size);
                var rhs = SimdImplementations.AllocateAligned<float>(size);
                var result = SimdImplementations.AllocateAligned<float>(size);
                SimdImplementations.FillRandom(lhs, size, 42);
                SimdImplementations.FillRandom(rhs, size, 43);

                var baseline = BenchFramework.Run(
                    () => SimdImplementations.AddScalarLoop_Float32(lhs, rhs, result, size),
                    "S1_contiguous", "Scalar", dtype, size, elementBytes, warmup, measure, Suite);
                results.Add(baseline);
                BenchFramework.PrintResult(baseline);

                var simd = BenchFramework.Run(
                    () => SimdImplementations.AddFull_Float32(lhs, rhs, result, size),
                    "S1_contiguous", "SIMD-FULL", dtype, size, elementBytes, warmup, measure, Suite);
                simd = simd with { SpeedupVsBaseline = baseline.MeanUs / simd.MeanUs };
                results.Add(simd);
                BenchFramework.PrintResult(simd);

                SimdImplementations.FreeAligned(lhs);
                SimdImplementations.FreeAligned(rhs);
                SimdImplementations.FreeAligned(result);
                break;
            }

            case "int32" or "int":
            {
                var lhs = SimdImplementations.AllocateAligned<int>(size);
                var rhs = SimdImplementations.AllocateAligned<int>(size);
                var result = SimdImplementations.AllocateAligned<int>(size);
                SimdImplementations.FillRandom(lhs, size, 42);
                SimdImplementations.FillRandom(rhs, size, 43);

                var baseline = BenchFramework.Run(
                    () => SimdImplementations.AddScalarLoop_Int32(lhs, rhs, result, size),
                    "S1_contiguous", "Scalar", dtype, size, elementBytes, warmup, measure, Suite);
                results.Add(baseline);
                BenchFramework.PrintResult(baseline);

                var simd = BenchFramework.Run(
                    () => SimdImplementations.AddFull_Int32(lhs, rhs, result, size),
                    "S1_contiguous", "SIMD-FULL", dtype, size, elementBytes, warmup, measure, Suite);
                simd = simd with { SpeedupVsBaseline = baseline.MeanUs / simd.MeanUs };
                results.Add(simd);
                BenchFramework.PrintResult(simd);

                SimdImplementations.FreeAligned(lhs);
                SimdImplementations.FreeAligned(rhs);
                SimdImplementations.FreeAligned(result);
                break;
            }

            case "int64" or "long":
            {
                var lhs = SimdImplementations.AllocateAligned<long>(size);
                var rhs = SimdImplementations.AllocateAligned<long>(size);
                var result = SimdImplementations.AllocateAligned<long>(size);
                SimdImplementations.FillRandom(lhs, size, 42);
                SimdImplementations.FillRandom(rhs, size, 43);

                var baseline = BenchFramework.Run(
                    () => SimdImplementations.AddScalarLoop_Int64(lhs, rhs, result, size),
                    "S1_contiguous", "Scalar", dtype, size, elementBytes, warmup, measure, Suite);
                results.Add(baseline);
                BenchFramework.PrintResult(baseline);

                var simd = BenchFramework.Run(
                    () => SimdImplementations.AddFull_Int64(lhs, rhs, result, size),
                    "S1_contiguous", "SIMD-FULL", dtype, size, elementBytes, warmup, measure, Suite);
                simd = simd with { SpeedupVsBaseline = baseline.MeanUs / simd.MeanUs };
                results.Add(simd);
                BenchFramework.PrintResult(simd);

                SimdImplementations.FreeAligned(lhs);
                SimdImplementations.FreeAligned(rhs);
                SimdImplementations.FreeAligned(result);
                break;
            }

            case "int16" or "short":
            {
                var lhs = SimdImplementations.AllocateAligned<short>(size);
                var rhs = SimdImplementations.AllocateAligned<short>(size);
                var result = SimdImplementations.AllocateAligned<short>(size);
                SimdImplementations.FillRandom(lhs, size, 42);
                SimdImplementations.FillRandom(rhs, size, 43);

                var baseline = BenchFramework.Run(
                    () => SimdImplementations.AddScalarLoop_Int16(lhs, rhs, result, size),
                    "S1_contiguous", "Scalar", dtype, size, elementBytes, warmup, measure, Suite);
                results.Add(baseline);
                BenchFramework.PrintResult(baseline);

                var simd = BenchFramework.Run(
                    () => SimdImplementations.AddFull_Int16(lhs, rhs, result, size),
                    "S1_contiguous", "SIMD-FULL", dtype, size, elementBytes, warmup, measure, Suite);
                simd = simd with { SpeedupVsBaseline = baseline.MeanUs / simd.MeanUs };
                results.Add(simd);
                BenchFramework.PrintResult(simd);

                SimdImplementations.FreeAligned(lhs);
                SimdImplementations.FreeAligned(rhs);
                SimdImplementations.FreeAligned(result);
                break;
            }

            case "byte" or "uint8":
            {
                var lhs = SimdImplementations.AllocateAligned<byte>(size);
                var rhs = SimdImplementations.AllocateAligned<byte>(size);
                var result = SimdImplementations.AllocateAligned<byte>(size);
                SimdImplementations.FillRandom(lhs, size, 42);
                SimdImplementations.FillRandom(rhs, size, 43);

                var baseline = BenchFramework.Run(
                    () => SimdImplementations.AddScalarLoop_Byte(lhs, rhs, result, size),
                    "S1_contiguous", "Scalar", dtype, size, elementBytes, warmup, measure, Suite);
                results.Add(baseline);
                BenchFramework.PrintResult(baseline);

                var simd = BenchFramework.Run(
                    () => SimdImplementations.AddFull_Byte(lhs, rhs, result, size),
                    "S1_contiguous", "SIMD-FULL", dtype, size, elementBytes, warmup, measure, Suite);
                simd = simd with { SpeedupVsBaseline = baseline.MeanUs / simd.MeanUs };
                results.Add(simd);
                BenchFramework.PrintResult(simd);

                SimdImplementations.FreeAligned(lhs);
                SimdImplementations.FreeAligned(rhs);
                SimdImplementations.FreeAligned(result);
                break;
            }
        }

        return results;
    }

    #endregion

    #region S2: Scalar broadcast (right)

    /// <summary>
    /// S2: Array + scalar — ideal for SIMD-SCALAR strategy.
    /// </summary>
    public static List<BenchResult> RunS2_ScalarRight(string dtype, int size, int warmup, int measure)
    {
        var results = new List<BenchResult>();
        var elementBytes = DtypeInfo.GetElementSize(dtype);

        switch (dtype.ToLowerInvariant())
        {
            case "float64" or "double":
            {
                var lhs = SimdImplementations.AllocateAligned<double>(size);
                var result = SimdImplementations.AllocateAligned<double>(size);
                SimdImplementations.FillRandom(lhs, size, 42);
                double scalar = 42.5;

                // Baseline: scalar loop
                var baseline = BenchFramework.Run(
                    () =>
                    {
                        for (int i = 0; i < size; i++) result[i] = lhs[i] + scalar;
                    },
                    "S2_scalar", "Scalar", dtype, size, elementBytes, warmup, measure, Suite);
                results.Add(baseline);
                BenchFramework.PrintResult(baseline);

                // SIMD-SCALAR
                var simd = BenchFramework.Run(
                    () => SimdImplementations.AddScalar_Float64(lhs, scalar, result, size),
                    "S2_scalar", "SIMD-SCALAR", dtype, size, elementBytes, warmup, measure, Suite);
                simd = simd with { SpeedupVsBaseline = baseline.MeanUs / simd.MeanUs };
                results.Add(simd);
                BenchFramework.PrintResult(simd);

                SimdImplementations.FreeAligned(lhs);
                SimdImplementations.FreeAligned(result);
                break;
            }

            case "float32" or "single" or "float":
            {
                var lhs = SimdImplementations.AllocateAligned<float>(size);
                var result = SimdImplementations.AllocateAligned<float>(size);
                SimdImplementations.FillRandom(lhs, size, 42);
                float scalar = 42.5f;

                var baseline = BenchFramework.Run(
                    () =>
                    {
                        for (int i = 0; i < size; i++) result[i] = lhs[i] + scalar;
                    },
                    "S2_scalar", "Scalar", dtype, size, elementBytes, warmup, measure, Suite);
                results.Add(baseline);
                BenchFramework.PrintResult(baseline);

                var simd = BenchFramework.Run(
                    () => SimdImplementations.AddScalar_Float32(lhs, scalar, result, size),
                    "S2_scalar", "SIMD-SCALAR", dtype, size, elementBytes, warmup, measure, Suite);
                simd = simd with { SpeedupVsBaseline = baseline.MeanUs / simd.MeanUs };
                results.Add(simd);
                BenchFramework.PrintResult(simd);

                SimdImplementations.FreeAligned(lhs);
                SimdImplementations.FreeAligned(result);
                break;
            }

            case "int32" or "int":
            {
                var lhs = SimdImplementations.AllocateAligned<int>(size);
                var result = SimdImplementations.AllocateAligned<int>(size);
                SimdImplementations.FillRandom(lhs, size, 42);
                int scalar = 42;

                var baseline = BenchFramework.Run(
                    () =>
                    {
                        for (int i = 0; i < size; i++) result[i] = lhs[i] + scalar;
                    },
                    "S2_scalar", "Scalar", dtype, size, elementBytes, warmup, measure, Suite);
                results.Add(baseline);
                BenchFramework.PrintResult(baseline);

                var simd = BenchFramework.Run(
                    () => SimdImplementations.AddScalar_Int32(lhs, scalar, result, size),
                    "S2_scalar", "SIMD-SCALAR", dtype, size, elementBytes, warmup, measure, Suite);
                simd = simd with { SpeedupVsBaseline = baseline.MeanUs / simd.MeanUs };
                results.Add(simd);
                BenchFramework.PrintResult(simd);

                SimdImplementations.FreeAligned(lhs);
                SimdImplementations.FreeAligned(result);
                break;
            }

            case "int64" or "long":
            {
                var lhs = SimdImplementations.AllocateAligned<long>(size);
                var result = SimdImplementations.AllocateAligned<long>(size);
                SimdImplementations.FillRandom(lhs, size, 42);
                long scalar = 42;

                var baseline = BenchFramework.Run(
                    () =>
                    {
                        for (int i = 0; i < size; i++) result[i] = lhs[i] + scalar;
                    },
                    "S2_scalar", "Scalar", dtype, size, elementBytes, warmup, measure, Suite);
                results.Add(baseline);
                BenchFramework.PrintResult(baseline);

                var simd = BenchFramework.Run(
                    () => SimdImplementations.AddScalar_Int64(lhs, scalar, result, size),
                    "S2_scalar", "SIMD-SCALAR", dtype, size, elementBytes, warmup, measure, Suite);
                simd = simd with { SpeedupVsBaseline = baseline.MeanUs / simd.MeanUs };
                results.Add(simd);
                BenchFramework.PrintResult(simd);

                SimdImplementations.FreeAligned(lhs);
                SimdImplementations.FreeAligned(result);
                break;
            }

            case "int16" or "short":
            {
                var lhs = SimdImplementations.AllocateAligned<short>(size);
                var result = SimdImplementations.AllocateAligned<short>(size);
                SimdImplementations.FillRandom(lhs, size, 42);
                short scalar = 42;

                var baseline = BenchFramework.Run(
                    () =>
                    {
                        for (int i = 0; i < size; i++) result[i] = (short)(lhs[i] + scalar);
                    },
                    "S2_scalar", "Scalar", dtype, size, elementBytes, warmup, measure, Suite);
                results.Add(baseline);
                BenchFramework.PrintResult(baseline);

                var simd = BenchFramework.Run(
                    () => SimdImplementations.AddScalar_Int16(lhs, scalar, result, size),
                    "S2_scalar", "SIMD-SCALAR", dtype, size, elementBytes, warmup, measure, Suite);
                simd = simd with { SpeedupVsBaseline = baseline.MeanUs / simd.MeanUs };
                results.Add(simd);
                BenchFramework.PrintResult(simd);

                SimdImplementations.FreeAligned(lhs);
                SimdImplementations.FreeAligned(result);
                break;
            }

            case "byte" or "uint8":
            {
                var lhs = SimdImplementations.AllocateAligned<byte>(size);
                var result = SimdImplementations.AllocateAligned<byte>(size);
                SimdImplementations.FillRandom(lhs, size, 42);
                byte scalar = 42;

                var baseline = BenchFramework.Run(
                    () =>
                    {
                        for (int i = 0; i < size; i++) result[i] = (byte)(lhs[i] + scalar);
                    },
                    "S2_scalar", "Scalar", dtype, size, elementBytes, warmup, measure, Suite);
                results.Add(baseline);
                BenchFramework.PrintResult(baseline);

                var simd = BenchFramework.Run(
                    () => SimdImplementations.AddScalar_Byte(lhs, scalar, result, size),
                    "S2_scalar", "SIMD-SCALAR", dtype, size, elementBytes, warmup, measure, Suite);
                simd = simd with { SpeedupVsBaseline = baseline.MeanUs / simd.MeanUs };
                results.Add(simd);
                BenchFramework.PrintResult(simd);

                SimdImplementations.FreeAligned(lhs);
                SimdImplementations.FreeAligned(result);
                break;
            }
        }

        return results;
    }

    #endregion

    #region S4: Row broadcast

    /// <summary>
    /// S4: (M,N) + (N,) row broadcast — ideal for SIMD-CHUNK strategy.
    /// </summary>
    public static List<BenchResult> RunS4_RowBroadcast(string dtype, int totalElements, int warmup, int measure)
    {
        // Use square-ish dimensions
        int rows = (int)Math.Sqrt(totalElements);
        int cols = totalElements / rows;
        int actualSize = rows * cols;

        var results = new List<BenchResult>();
        var elementBytes = DtypeInfo.GetElementSize(dtype);

        switch (dtype.ToLowerInvariant())
        {
            case "float64" or "double":
            {
                var matrix = SimdImplementations.AllocateAligned<double>(actualSize);
                var row = SimdImplementations.AllocateAligned<double>(cols);
                var result = SimdImplementations.AllocateAligned<double>(actualSize);
                SimdImplementations.FillRandom(matrix, actualSize, 42);
                SimdImplementations.FillRandom(row, cols, 43);

                // Baseline: nested scalar loops
                var baseline = BenchFramework.Run(
                    () => SimdImplementations.AddRowBroadcastScalar_Float64(matrix, row, result, rows, cols),
                    "S4_rowBC", "Scalar", dtype, actualSize, elementBytes, warmup, measure, Suite);
                results.Add(baseline);
                BenchFramework.PrintResult(baseline);

                // SIMD-CHUNK: SIMD on inner dimension
                var simd = BenchFramework.Run(
                    () => SimdImplementations.AddRowBroadcast_Float64(matrix, row, result, rows, cols),
                    "S4_rowBC", "SIMD-CHUNK", dtype, actualSize, elementBytes, warmup, measure, Suite);
                simd = simd with { SpeedupVsBaseline = baseline.MeanUs / simd.MeanUs };
                results.Add(simd);
                BenchFramework.PrintResult(simd);

                SimdImplementations.FreeAligned(matrix);
                SimdImplementations.FreeAligned(row);
                SimdImplementations.FreeAligned(result);
                break;
            }

            case "float32" or "single" or "float":
            {
                var matrix = SimdImplementations.AllocateAligned<float>(actualSize);
                var row = SimdImplementations.AllocateAligned<float>(cols);
                var result = SimdImplementations.AllocateAligned<float>(actualSize);
                SimdImplementations.FillRandom(matrix, actualSize, 42);
                SimdImplementations.FillRandom(row, cols, 43);

                var baseline = BenchFramework.Run(
                    () =>
                    {
                        for (int r = 0; r < rows; r++)
                            for (int c = 0; c < cols; c++)
                                result[r * cols + c] = matrix[r * cols + c] + row[c];
                    },
                    "S4_rowBC", "Scalar", dtype, actualSize, elementBytes, warmup, measure, Suite);
                results.Add(baseline);
                BenchFramework.PrintResult(baseline);

                var simd = BenchFramework.Run(
                    () => SimdImplementations.AddRowBroadcast_Float32(matrix, row, result, rows, cols),
                    "S4_rowBC", "SIMD-CHUNK", dtype, actualSize, elementBytes, warmup, measure, Suite);
                simd = simd with { SpeedupVsBaseline = baseline.MeanUs / simd.MeanUs };
                results.Add(simd);
                BenchFramework.PrintResult(simd);

                SimdImplementations.FreeAligned(matrix);
                SimdImplementations.FreeAligned(row);
                SimdImplementations.FreeAligned(result);
                break;
            }

            case "int32" or "int":
            {
                var matrix = SimdImplementations.AllocateAligned<int>(actualSize);
                var row = SimdImplementations.AllocateAligned<int>(cols);
                var result = SimdImplementations.AllocateAligned<int>(actualSize);
                SimdImplementations.FillRandom(matrix, actualSize, 42);
                SimdImplementations.FillRandom(row, cols, 43);

                var baseline = BenchFramework.Run(
                    () =>
                    {
                        for (int r = 0; r < rows; r++)
                            for (int c = 0; c < cols; c++)
                                result[r * cols + c] = matrix[r * cols + c] + row[c];
                    },
                    "S4_rowBC", "Scalar", dtype, actualSize, elementBytes, warmup, measure, Suite);
                results.Add(baseline);
                BenchFramework.PrintResult(baseline);

                var simd = BenchFramework.Run(
                    () => SimdImplementations.AddRowBroadcast_Int32(matrix, row, result, rows, cols),
                    "S4_rowBC", "SIMD-CHUNK", dtype, actualSize, elementBytes, warmup, measure, Suite);
                simd = simd with { SpeedupVsBaseline = baseline.MeanUs / simd.MeanUs };
                results.Add(simd);
                BenchFramework.PrintResult(simd);

                SimdImplementations.FreeAligned(matrix);
                SimdImplementations.FreeAligned(row);
                SimdImplementations.FreeAligned(result);
                break;
            }

            // Other dtypes similar pattern...
            default:
                Console.WriteLine($"  S4 row broadcast: dtype {dtype} not implemented yet");
                break;
        }

        return results;
    }

    #endregion

    #region S5: Column broadcast

    /// <summary>
    /// S5: (M,N) + (M,1) column broadcast — use SIMD-SCALAR per row.
    /// </summary>
    public static List<BenchResult> RunS5_ColBroadcast(string dtype, int totalElements, int warmup, int measure)
    {
        int rows = (int)Math.Sqrt(totalElements);
        int cols = totalElements / rows;
        int actualSize = rows * cols;

        var results = new List<BenchResult>();
        var elementBytes = DtypeInfo.GetElementSize(dtype);

        switch (dtype.ToLowerInvariant())
        {
            case "float64" or "double":
            {
                var matrix = SimdImplementations.AllocateAligned<double>(actualSize);
                var col = SimdImplementations.AllocateAligned<double>(rows);
                var result = SimdImplementations.AllocateAligned<double>(actualSize);
                SimdImplementations.FillRandom(matrix, actualSize, 42);
                SimdImplementations.FillRandom(col, rows, 43);

                // Baseline: nested scalar loops
                var baseline = BenchFramework.Run(
                    () => SimdImplementations.AddColBroadcastScalar_Float64(matrix, col, result, rows, cols),
                    "S5_colBC", "Scalar", dtype, actualSize, elementBytes, warmup, measure, Suite);
                results.Add(baseline);
                BenchFramework.PrintResult(baseline);

                // SIMD per row (using scalar broadcast within each row)
                var simd = BenchFramework.Run(
                    () => SimdImplementations.AddColBroadcast_Float64(matrix, col, result, rows, cols),
                    "S5_colBC", "SIMD-SCALAR", dtype, actualSize, elementBytes, warmup, measure, Suite);
                simd = simd with { SpeedupVsBaseline = baseline.MeanUs / simd.MeanUs };
                results.Add(simd);
                BenchFramework.PrintResult(simd);

                SimdImplementations.FreeAligned(matrix);
                SimdImplementations.FreeAligned(col);
                SimdImplementations.FreeAligned(result);
                break;
            }

            default:
                Console.WriteLine($"  S5 col broadcast: dtype {dtype} not implemented yet");
                break;
        }

        return results;
    }

    #endregion
}
