using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using NumSharp;
using NumSharp.Interop.OpenBLAS;

namespace NumSharpOpenBlasAdvancedExample;

/// <summary>
///     One float32 matrix product, computed with different thread counts and with NumSharp's own
///     kernels, and the SHA-256 of each result. Equal hashes mean equal bits.
/// </summary>
/// <remarks>
///     Every result is a correct product. They differ in the last bits only, because float addition
///     depends on the order of the terms. The hashes show which settings give a result that you can
///     store and compare later: one thread, the same OpenBLAS build, the same kernels.
/// </remarks>
internal static class Reproducibility
{
    /// <summary>Prints the hash of the product for each setting.</summary>
    /// <remarks>
    ///     The inputs come from NumSharp's generator, which gives the numbers of NumPy's for the
    ///     same seed. So the first printed hash is also the hash that NumPy gives for this product
    ///     when NumPy runs the same OpenBLAS build with one thread and the same kernels.
    ///     The method is not inlined, for the reason that <see cref="Backend.Start"/> gives.
    /// </remarks>
    [MethodImpl(MethodImplOptions.NoInlining)]
    public static void Run()
    {
        using var scope = NDScope.Open();

        np.random.seed(7);
        NDArray a = np.random.rand(512, 784).astype(np.float32);
        NDArray b = np.random.rand(784, 512).astype(np.float32);

        Console.WriteLine("product     512 x 784 @ 784 x 512 float32, the SHA-256 of the result");

        string first = Hash(np.matmul(a, b));                            // Backend.Start set one thread
        string second = Hash(np.matmul(a, b));
        Console.WriteLine($"  {"1 thread",-12}{first}");
        Console.WriteLine($"  {"1 thread",-12}{second}   {(first == second ? "the same bits again" : "OTHER BITS")}");

        // More threads: a faster product, and other last bits for each count. OpenBLAS has an upper
        // limit that it was built with, so the label is the count that it reports, not the request.
        var shown = new HashSet<int> { 1 };
        foreach (int request in new[] { 2, Environment.ProcessorCount })
        {
            OpenBlasEngine.Enable(threads: request);
            int threads = TensorEngine.Threading.GetThreads(TensorEngine.Threading.OpenBlas) ?? request;
            if (shown.Add(threads))
                Console.WriteLine($"  {threads + " threads",-12}{Hash(np.matmul(a, b))}");
        }

        OpenBlasEngine.Enable(threads: 1);                               // back to the setting that repeats

        OpenBlasEngine.Disable();                                        // NumSharp's own kernels
        try
        {
            Console.WriteLine($"  {"NumSharp",-12}{Hash(np.matmul(a, b))}   its own kernels: correct, and other bits again");
        }
        finally
        {
            OpenBlasEngine.Enable();                                     // the library stayed loaded, with one thread
        }
    }

    /// <summary>The first 16 hexadecimal digits of the SHA-256 of a float32 array's elements.</summary>
    /// <param name="product">A float32 array. Its elements are read in row order.</param>
    /// <returns>The start of the hash, in lower case.</returns>
    private static string Hash(NDArray product)
    {
        ReadOnlySpan<byte> bytes = MemoryMarshal.AsBytes<float>(product.ToArray<float>());
        return Convert.ToHexString(SHA256.HashData(bytes))[..16].ToLowerInvariant();
    }
}
