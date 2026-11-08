using System.Runtime.CompilerServices;
using System.Text.Json;
using NumSharp.Interop.OpenBLAS;

namespace NumSharpOpenBlasAdvancedExample;

/// <summary>
///     Loads the OpenBLAS build that the project file pins, with one thread and one fixed set of
///     kernels, and reports which library answered.
/// </summary>
/// <remarks>
///     This class is the first code of the program that names a type of the package. Program.cs
///     turns the automatic load off before it calls <see cref="Start"/>, so the call in
///     <see cref="Start"/> is the one that loads the library.
/// </remarks>
internal static class Backend
{
    /// <summary>
    ///     The name of the file that the build writes beside a library that it staged. The package
    ///     reads it when it looks for a library: a folder with this file holds a required library,
    ///     and the package loads that file or throws.
    /// </summary>
    private const string MarkerFileName = "openblas.source.json";

    /// <summary>What happened to the kernel request of <see cref="Start"/>. <see cref="Report"/> prints it.</summary>
    private static string _kernelPin = "not requested";

    /// <summary>
    ///     Loads OpenBLAS with one thread and, where the processor permits, with the Haswell kernels.
    /// </summary>
    /// <returns>
    ///     True when OpenBLAS computes from now on. False when the pinned library cannot be loaded;
    ///     the reason is then on the error stream.
    /// </returns>
    /// <remarks>
    ///     <para>
    ///     One thread is what makes a result repeat. OpenBLAS splits a product across its threads,
    ///     and each split adds the terms in another order.
    ///     </para>
    ///     <para>
    ///     The kernel pin is what makes a result repeat on ANOTHER machine. Without it OpenBLAS
    ///     selects the fastest kernels that the processor has, and other kernels give other last
    ///     bits. <see cref="OpenBlasEngine.CoreType"/> is "Haswell": every x86-64 processor with
    ///     AVX2 and FMA can run those kernels. The price is speed on a processor that has faster
    ///     ones. An unknown kernel name does not fail: OpenBLAS ignores it, so read
    ///     <see cref="OpenBlasEngine.CoreName"/> after the call.
    ///     </para>
    ///     <para>
    ///     The method is not inlined. If the JIT compiler put its body into Program.cs, the runtime
    ///     would start the package before the first statement of Program.cs runs.
    ///     </para>
    /// </remarks>
    [MethodImpl(MethodImplOptions.NoInlining)]
    public static bool Start()
    {
        // True when the library was loaded before this call: the variable that Program.cs sets did
        // not take effect. The kernels are then the ones that OpenBLAS selected by itself.
        bool loadedEarlier = OpenBlasEngine.LibraryPath != null;

        try
        {
            try
            {
                OpenBlasEngine.Enable(threads: 1, coreType: OpenBlasEngine.CoreType);
                _kernelPin = loadedEarlier ? "selected by OpenBLAS before the pin, and the same" : "pinned";
            }
            catch (PlatformNotSupportedException)
            {
                // The processor cannot execute the Haswell kernels (no AVX2, or not x86-64). The
                // package refuses the request, because OpenBLAS would not fall back: the process
                // would stop with an illegal instruction.
                OpenBlasEngine.Enable(threads: 1);
                _kernelPin = "not pinned: this processor cannot run the Haswell kernels";
            }
            catch (InvalidOperationException)
            {
                // The library was loaded earlier with other kernels, and the kernels of a loaded
                // library cannot change. The thread count can.
                OpenBlasEngine.Enable(threads: 1);
                _kernelPin = "not pinned: OpenBLAS was loaded before this call";
            }

            return true;
        }
        catch (Exception problem) when (problem is DllNotFoundException or EntryPointNotFoundException)
        {
            // OpenBlasRequiredOverrideException is a DllNotFoundException: the library that the
            // project file pins is not in the output, or it is not the file that the build recorded.
            Console.Error.WriteLine(problem.Message);
            return false;
        }
    }

    /// <summary>Prints which library computes, whether it is the pinned one, and its kernels.</summary>
    /// <remarks>
    ///     <see cref="OpenBlasEngine.Info"/> ends with the build text of OpenBLAS itself, so the
    ///     pinned version is visible there too: "OpenBLAS 0.3.31.188.0" for this project file.
    ///     The method is not inlined, for the reason that <see cref="Start"/> gives.
    /// </remarks>
    [MethodImpl(MethodImplOptions.NoInlining)]
    public static void Report()
    {
        Console.WriteLine($"OpenBLAS    {OpenBlasEngine.Info}");
        Console.WriteLine($"bundled     {OpenBlasEngine.IsBundledLibrary}");
        Console.WriteLine($"pin         {DescribePin(OpenBlasEngine.LibraryPath)}");
        Console.WriteLine($"kernels     {OpenBlasEngine.CoreName} ({_kernelPin})");
    }

    /// <summary>Reads the file that the build wrote beside the library that it staged.</summary>
    /// <param name="library">The path of the loaded library, or null when none is loaded.</param>
    /// <returns>
    ///     The distribution and the version that the build pinned, or a sentence that says that
    ///     the build pinned nothing.
    /// </returns>
    /// <remarks>
    ///     The file also holds the SHA-256 of the library. The package compares it with the file
    ///     before it loads the file, so this method does not compute the hash again.
    /// </remarks>
    private static string DescribePin(string? library)
    {
        string? folder = Path.GetDirectoryName(library);
        string marker = folder is null ? "" : Path.Combine(folder, MarkerFileName);
        if (!File.Exists(marker))
            return "none: the project file has no OpenBlasVersion, so discovery selected the library";

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(marker));
        JsonElement root = document.RootElement;
        string Text(string name) => root.TryGetProperty(name, out JsonElement value) ? value.ToString() : "?";

        return $"{Text("distribution")} {Text("version")}, required: {Text("required")}";
    }
}
