using NumSharp.Benchmark.Exploration.Infrastructure;

namespace NumSharp.Benchmark.Exploration.Integration;

/// <summary>
/// Test NumSharp broadcasting performance through actual TensorEngine.
/// Measures overhead of NumSharp vs isolated implementations.
/// </summary>
public static class NumSharpBroadcast
{
    private const string Suite = "NumSharpIntegration";
    private const int Seed = 42;

    /// <summary>
    /// Run all NumSharp integration benchmarks.
    /// </summary>
    public static List<BenchResult> RunAll(bool quick = false)
    {
        var results = new List<BenchResult>();
        var warmup = quick ? 2 : BenchFramework.DefaultWarmup;
        var measure = quick ? 5 : BenchFramework.DefaultMeasure;

        var sizes = quick
            ? new[] { 100_000, 1_000_000 }
            : new[] { 1_000, 100_000, 1_000_000, 10_000_000 };

        BenchFramework.PrintHeader($"{Suite}: NumSharp Broadcasting Performance");

        long initialMemory = GC.GetTotalMemory(true);
        const long MemoryThresholdBytes = 4L * 1024 * 1024 * 1024; // 4 GB

        foreach (var size in sizes)
        {
            BenchFramework.PrintDivider($"Size: {size:N0}");

            results.AddRange(TestContiguousAdd(size, warmup, measure));
            CleanupAndCheckMemory(initialMemory, MemoryThresholdBytes, $"after ContiguousAdd size={size}");

            results.AddRange(TestScalarBroadcast(size, warmup, measure));
            CleanupAndCheckMemory(initialMemory, MemoryThresholdBytes, $"after ScalarBroadcast size={size}");

            results.AddRange(TestRowBroadcast(size, warmup, measure));
            CleanupAndCheckMemory(initialMemory, MemoryThresholdBytes, $"after RowBroadcast size={size}");
        }

        OutputFormatters.PrintSummary(results);
        return results;
    }

    /// <summary>
    /// Force GC and check memory usage.
    /// </summary>
    private static void CleanupAndCheckMemory(long initialMemory, long threshold, string context)
    {
        // Force full GC collection
        GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true);
        GC.WaitForPendingFinalizers();
        GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true);

        long currentMemory = GC.GetTotalMemory(false);
        long memoryGrowth = currentMemory - initialMemory;

        if (memoryGrowth > threshold)
        {
            Console.WriteLine($"[WARNING] Memory leak detected {context}: {memoryGrowth / 1024 / 1024} MB above baseline");
        }
    }

    /// <summary>
    /// Test contiguous array addition (S1 scenario).
    /// </summary>
    public static List<BenchResult> TestContiguousAdd(int size, int warmup, int measure)
    {
        var results = new List<BenchResult>();
        int elementBytes = 8;

        // Create NumSharp arrays
        np.random.seed(Seed);
        var a = np.random.rand(size);
        var b = np.random.rand(size);
        NDArray result = null!;

        // NumSharp contiguous add
        var numsharp = BenchFramework.Run(
            () => result = a + b,
            "S1_contiguous", "NumSharp", "float64", size, elementBytes, warmup, measure, Suite);
        results.Add(numsharp);
        BenchFramework.PrintResult(numsharp);

        // Also test np.add function
        var npAdd = BenchFramework.Run(
            () => result = np.add(a, b),
            "S1_contiguous", "np.add", "float64", size, elementBytes, warmup, measure, Suite);
        npAdd = npAdd with { SpeedupVsBaseline = numsharp.MeanUs / npAdd.MeanUs };
        results.Add(npAdd);
        BenchFramework.PrintResult(npAdd);

        // Cleanup: release references to allow GC
        a = null!;
        b = null!;
        result = null!;

        return results;
    }

    /// <summary>
    /// Test scalar broadcast (S2 scenario).
    /// </summary>
    public static List<BenchResult> TestScalarBroadcast(int size, int warmup, int measure)
    {
        var results = new List<BenchResult>();
        int elementBytes = 8;

        np.random.seed(Seed);
        var a = np.random.rand(size);
        double scalar = 42.5;
        NDArray result = null!;

        // NumSharp scalar add
        var numsharp = BenchFramework.Run(
            () => result = a + scalar,
            "S2_scalar", "NumSharp", "float64", size, elementBytes, warmup, measure, Suite);
        results.Add(numsharp);
        BenchFramework.PrintResult(numsharp);

        // Cleanup
        a = null!;
        result = null!;

        return results;
    }

    /// <summary>
    /// Test row broadcast (S4 scenario).
    /// </summary>
    public static List<BenchResult> TestRowBroadcast(int totalElements, int warmup, int measure)
    {
        var results = new List<BenchResult>();
        int elementBytes = 8;

        // Create square-ish matrix
        int rows = (int)Math.Sqrt(totalElements);
        int cols = totalElements / rows;
        int actualSize = rows * cols;

        np.random.seed(Seed);
        var matrix = np.random.rand(rows, cols);
        var row = np.random.rand(cols);
        NDArray result = null!;

        // NumSharp row broadcast
        var numsharp = BenchFramework.Run(
            () => result = matrix + row,
            "S4_rowBC", "NumSharp", "float64", actualSize, elementBytes, warmup, measure, Suite);
        results.Add(numsharp);
        BenchFramework.PrintResult(numsharp);

        // Cleanup
        matrix = null!;
        row = null!;
        result = null!;

        return results;
    }

    /// <summary>
    /// Test various NumSharp operations to measure overhead.
    /// </summary>
    public static List<BenchResult> TestOperationOverhead(int size, int warmup, int measure)
    {
        var results = new List<BenchResult>();
        int elementBytes = 8;

        BenchFramework.PrintDivider($"Operation Overhead: Size = {size:N0}");

        np.random.seed(Seed);
        var a = np.random.rand(size);
        var b = np.random.rand(size);
        NDArray result = null!;

        // Test different operations
        var ops = new (string name, Func<NDArray> op)[]
        {
            ("Add", () => a + b),
            ("Sub", () => a - b),
            ("Mul", () => a * b),
            ("Div", () => a / b),
            ("np.sqrt", () => np.sqrt(a)),
            ("np.exp", () => np.exp(a)),
            ("np.sum", () => np.sum(a)),
            ("np.mean", () => np.mean(a)),
        };

        foreach (var (name, op) in ops)
        {
            var r = BenchFramework.Run(
                () => result = op(),
                "Overhead", name, "float64", size, elementBytes, warmup, measure, Suite);
            results.Add(r);
            BenchFramework.PrintResult(r);

            // Force GC between operations to prevent accumulation
            result = null!;
            GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true);
            GC.WaitForPendingFinalizers();
        }

        // Cleanup
        a = null!;
        b = null!;

        return results;
    }

    /// <summary>
    /// Compare NumSharp overhead vs raw pointer operations.
    /// </summary>
    public static unsafe List<BenchResult> CompareOverhead(int size, int warmup, int measure)
    {
        var results = new List<BenchResult>();
        int elementBytes = 8;

        BenchFramework.PrintDivider($"NumSharp vs Raw Pointer: Size = {size:N0}");

        // Raw pointer implementation
        var lhsRaw = SimdImplementations.AllocateAligned<double>(size);
        var rhsRaw = SimdImplementations.AllocateAligned<double>(size);
        var resultRaw = SimdImplementations.AllocateAligned<double>(size);
        SimdImplementations.FillRandom(lhsRaw, size, 42);
        SimdImplementations.FillRandom(rhsRaw, size, 43);

        // NumSharp arrays
        np.random.seed(42);
        var lhsNp = np.random.rand(size);
        np.random.seed(43);
        var rhsNp = np.random.rand(size);
        NDArray resultNp = null!;

        // Raw SIMD
        var raw = BenchFramework.Run(
            () => SimdImplementations.AddFull_Float64(lhsRaw, rhsRaw, resultRaw, size),
            "Overhead", "RawSIMD", "float64", size, elementBytes, warmup, measure, Suite);
        results.Add(raw);
        BenchFramework.PrintResult(raw);

        // NumSharp
        var numsharp = BenchFramework.Run(
            () => resultNp = lhsNp + rhsNp,
            "Overhead", "NumSharp", "float64", size, elementBytes, warmup, measure, Suite);
        numsharp = numsharp with
        {
            SpeedupVsBaseline = raw.MeanUs / numsharp.MeanUs,
            Notes = $"Overhead: {numsharp.MeanUs / raw.MeanUs:F2}x"
        };
        results.Add(numsharp);
        BenchFramework.PrintResult(numsharp);

        // Cleanup raw memory
        SimdImplementations.FreeAligned(lhsRaw);
        SimdImplementations.FreeAligned(rhsRaw);
        SimdImplementations.FreeAligned(resultRaw);

        // Cleanup NumSharp arrays
        lhsNp = null!;
        rhsNp = null!;
        resultNp = null!;

        return results;
    }

    /// <summary>
    /// Test different dtypes through NumSharp.
    /// </summary>
    public static List<BenchResult> TestDtypes(int size, int warmup, int measure)
    {
        var results = new List<BenchResult>();

        BenchFramework.PrintDivider($"NumSharp Dtype Performance: Size = {size:N0}");

        var dtypes = new[]
        {
            (name: "byte", type: typeof(byte), bytes: 1),
            (name: "int16", type: typeof(short), bytes: 2),
            (name: "int32", type: typeof(int), bytes: 4),
            (name: "int64", type: typeof(long), bytes: 8),
            (name: "float32", type: typeof(float), bytes: 4),
            (name: "float64", type: typeof(double), bytes: 8),
        };

        NDArray result = null!;

        foreach (var (name, type, bytes) in dtypes)
        {
            np.random.seed(Seed);
            var a = np.random.randint(0, 100, new Shape(size), type);
            var b = np.random.randint(0, 100, new Shape(size), type);

            var r = BenchFramework.Run(
                () => result = a + b,
                "DtypePerf", "Add", name, size, bytes, warmup, measure, Suite);
            results.Add(r);
            BenchFramework.PrintResult(r);

            // Cleanup after each dtype test
            a = null!;
            b = null!;
            result = null!;
            GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true);
            GC.WaitForPendingFinalizers();
        }

        return results;
    }
}
