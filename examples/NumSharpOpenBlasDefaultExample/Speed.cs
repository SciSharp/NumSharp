using System.Diagnostics;
using NumSharp;
using NumSharp.Interop.OpenBLAS;

namespace NumSharpOpenBlasDefaultExample;

/// <summary>
///     One matrix product computed two times: by OpenBLAS, and by NumSharp's own kernels after
///     <see cref="OpenBlasEngine.Disable"/>. It shows that the backend is a switch and what the
///     switch is worth on this machine.
/// </summary>
internal static class Speed
{
    /// <summary>The number of rows and columns of the two matrices.</summary>
    private const int Size = 1024;

    /// <summary>The number of timed products for each kernel. The fastest one is reported.</summary>
    private const int Rounds = 3;

    /// <summary>
    ///     Times the product with each kernel and prints the two times and the largest difference
    ///     between the two results.
    /// </summary>
    /// <remarks>
    ///     The results are not equal bit for bit. Both are correct products, but the two kernels
    ///     add the terms in a different order, and float addition depends on the order. The
    ///     difference that this method prints is that effect.
    /// </remarks>
    public static void Run()
    {
        using var scope = NDScope.Open();

        np.random.seed(0);
        NDArray a = np.random.rand(Size, Size);
        NDArray b = np.random.rand(Size, Size);

        (double openBlasMs, NDArray viaOpenBlas) = Best(() => np.matmul(a, b));

        double numSharpMs;
        NDArray viaNumSharp;
        OpenBlasEngine.Disable();                    // from here NumSharp's own kernels compute the products
        try
        {
            (numSharpMs, viaNumSharp) = Best(() => np.matmul(a, b));
        }
        finally
        {
            // The library stays loaded while the backend is off, so this call does not load it again.
            // It is in a finally block so that an error in the timed code cannot leave the backend off.
            OpenBlasEngine.Enable();
        }

        double difference = np.amax(np.abs(viaOpenBlas - viaNumSharp)).GetDouble();

        Console.WriteLine($"matmul      {Size} x {Size} float64, the best of {Rounds} runs");
        Console.WriteLine($"  NumSharp  {numSharpMs,8:F1} ms");
        Console.WriteLine($"  OpenBLAS  {openBlasMs,8:F1} ms   {numSharpMs / openBlasMs:F1} times faster");
        Console.WriteLine($"  the two results differ by {difference:0.0e+00} at most");
    }

    /// <summary>Runs <paramref name="product"/> one time to warm up, then <see cref="Rounds"/> timed times.</summary>
    /// <param name="product">The computation to time. Each call must return a new array.</param>
    /// <returns>The shortest time in milliseconds, and the result of the last run.</returns>
    /// <remarks>
    ///     The warm-up call pays for JIT compilation and, for OpenBLAS, for the start of its worker
    ///     threads. The arrays of the earlier runs are released at once, because each one holds
    ///     8 MB of native memory that the garbage collector cannot see.
    /// </remarks>
    private static (double Milliseconds, NDArray Result) Best(Func<NDArray> product)
    {
        product().Dispose();

        double best = double.MaxValue;
        NDArray? last = null;
        for (int round = 0; round < Rounds; round++)
        {
            long start = Stopwatch.GetTimestamp();
            NDArray result = product();
            best = Math.Min(best, Stopwatch.GetElapsedTime(start).TotalMilliseconds);

            last?.Dispose();
            last = result;
        }

        return (best, last!);
    }
}
