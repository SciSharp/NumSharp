namespace NumSharp.Benchmark.GraphEngine.Infrastructure;

/// <summary>
/// Provides standard array sizes for benchmarks.
/// </summary>
public static class ArraySizeSource
{
    /// <summary>
    /// Scalar size - single element.
    /// Critical for measuring pure dispatch/call overhead without any loop cost.
    /// </summary>
    public const int Scalar = 1;

    /// <summary>
    /// Tiny array size - common small collection.
    /// Good for evaluating performance on typical small datasets (configs, small batches).
    /// </summary>
    public const int Tiny = 100;

    /// <summary>
    /// Small array size - fits in L1 cache.
    /// Good for measuring per-element overhead with minimal memory effects.
    /// </summary>
    public const int Small = 1_000;

    /// <summary>
    /// Medium array size - fits in L2/L3 cache.
    /// Good for typical use cases.
    /// </summary>
    public const int Medium = 100_000;

    /// <summary>
    /// Large array size - exceeds cache, memory-bound.
    /// Good for measuring throughput.
    /// </summary>
    public const int Large = 10_000_000;

    /// <summary>
    /// Standard array sizes for comprehensive benchmarks.
    /// Includes scalar for overhead measurement, tiny for common collections,
    /// and the three cache-tier sizes.
    /// </summary>
    public static IEnumerable<int> StandardSizes => new[] { Scalar, Tiny, Small, Medium, Large };

    /// <summary>
    /// Quick test sizes - only large for throughput focus.
    /// </summary>
    public static IEnumerable<int> QuickSizes => new[] { Large };

    /// <summary>
    /// Overhead-focused sizes - scalar and tiny for measuring dispatch cost.
    /// </summary>
    public static IEnumerable<int> OverheadSizes => new[] { Scalar, Tiny };

    /// <summary>
    /// Cache-tier sizes - small, medium, large (excludes scalar/tiny).
    /// </summary>
    public static IEnumerable<int> CacheTierSizes => new[] { Small, Medium, Large };

    /// <summary>
    /// All sizes including intermediate steps for detailed analysis.
    /// </summary>
    public static IEnumerable<int> AllSizes => new[] { Scalar, Tiny, Small, 10_000, Medium, 1_000_000, Large };

    /// <summary>
    /// 2D array sizes as (rows, cols) tuples.
    /// </summary>
    public static IEnumerable<(int Rows, int Cols)> Matrix2DSizes => new[]
    {
        (100, 100),           // 10K elements
        (1000, 1000),         // 1M elements
        (3162, 3162)          // ~10M elements (sqrt of 10M)
    };

    /// <summary>
    /// 3D array sizes as (d1, d2, d3) tuples.
    /// </summary>
    public static IEnumerable<(int D1, int D2, int D3)> Tensor3DSizes => new[]
    {
        (100, 100, 100),      // 1M elements
        (215, 215, 215)       // ~10M elements (cbrt of 10M)
    };

    /// <summary>
    /// Get human-readable size description.
    /// </summary>
    public static string GetSizeDescription(int n) => n switch
    {
        Scalar => "Scalar (1)",
        <= Tiny => $"Tiny ({Tiny})",
        <= Small => $"Small ({Small:N0})",
        <= 10_000 => "10K",
        <= Medium => $"Medium ({Medium:N0})",
        <= 1_000_000 => "1M",
        <= Large => $"Large ({Large:N0})",
        _ => $"Huge ({n:N0})"
    };
}
