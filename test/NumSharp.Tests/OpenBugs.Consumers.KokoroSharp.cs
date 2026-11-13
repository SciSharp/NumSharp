using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.Loader;

namespace NumSharp.Tests
{
    /// <summary>
    ///     Open issues found by moving KokoroSharp (Lyrcaxis/KokoroSharp, the NumSharp consumer with the most NuGet traffic) to
    ///     NumSharp 0.80.0 and running it on desktops and Android phones (2026-11-08/09), pinned and deferred by the owner on
    ///     that PR's design pages ("Create OpenBugs on journey5 for each of the answers", 2026-11-09). KokoroSharp works
    ///     around each one today; every test asserts the behavior a fix must give, and fails until then.
    /// </summary>
    /// <remarks>
    ///     The fourth finding of that round, a memory map of a Fortran-ordered <c>.npy</c> that <c>Dispose</c> never
    ///     releases, is <c>OpenBugs.Mmap_FortranOrder_DisposeReleasesTheFile</c> in <c>OpenBugs.cs</c> (merged from
    ///     <c>openbugs/memmap-fortran-release</c>). Two more findings of the same round were fixed rather than pinned: the
    ///     float → small-int casts that called <c>Math.Truncate</c> per element (5.5x slower than a plain loop on Mono), and
    ///     the ONNX Runtime bridge refusing XNNPACK's outputs.
    /// </remarks>
    [TestClass]
    public class OpenBugsKokoroSharpConsumerTests : TestClass
    {
        // ============================================================================
        //  BUG: np's class constructor costs over 100 ms the first time a process touches np.
        //
        //  Every app pays it on its first NumSharp call: 85 static fields (43 DTypes, 7 FrozenDictionary promotion tables,
        //  np.random, np.char, ...) are built eagerly although a first call needs a handful. Measured: 50-95 ms in a fresh
        //  KokoroSharp process on 0.80.0, 122-170 ms on journey5 79d83605 in a fresh load context (below). KokoroSharp hides
        //  it with a background warm-up started from a [ModuleInitializer]; a phone app's first audio waits for it.
        //  The fix: build the rarely needed statics lazily, so the class constructor does what every caller needs.
        // ============================================================================

        /// <summary>
        ///     <c>np</c>'s class constructor runs in under 30 ms when NumSharp's code is JIT-compiled anew (a fresh collectible
        ///     <see cref="AssemblyLoadContext"/>; the framework stays warm, as in an app whose first NumSharp call comes after
        ///     its own startup). Today the fastest of three fresh contexts takes about 120 ms.
        /// </summary>
        /// <remarks>
        ///     A wall-clock assertion, so it runs alone (<c>[DoNotParallelize]</c>) and keeps the fastest of three contexts: an
        ///     interruption can only add time. Each context loads its own copy of <c>NumSharp.dll</c> and is unloaded after.
        /// </remarks>
        [TestMethod, DoNotParallelize, OpenBugs(IssueUrl = "consumer-kokorosharp-np-cold-start")]
        public void NpClassConstructor_InAFreshLoadContext_TakesUnder30Milliseconds()
        {
            string numsharp = typeof(np).Assembly.Location;
            double fastest = double.MaxValue;
            for (int round = 0; round < 3; round++)
            {
                var context = new AssemblyLoadContext($"cold-np-{round}", isCollectible: true);
                try
                {
                    Type npType = context.LoadFromAssemblyPath(numsharp).GetType("NumSharp.np", throwOnError: true);
                    long start = Stopwatch.GetTimestamp();
                    RuntimeHelpers.RunClassConstructor(npType.TypeHandle);
                    fastest = System.Math.Min(fastest, Stopwatch.GetElapsedTime(start).TotalMilliseconds);
                }
                finally
                {
                    context.Unload();
                }
            }

            fastest.Should().BeLessThan(30, $"np's class constructor should build only what a first call needs; the fastest of three fresh load contexts took {fastest:F1} ms");
        }

        // ============================================================================
        //  BUG: np.load_npy stages every file through a fresh 256 KB managed buffer and copies it twice.
        //
        //  NpyFormat reads the data region in BufferSize (1 << 18) chunks into a new byte[] per call - a large-object-heap
        //  allocation for any file over 85 KB - and copies each chunk into the array. NumPy's read_array reads a real file
        //  straight into the array (numpy.fromfile), with no staging buffer. KokoroSharp reads its voices with one positioned
        //  read into the float[,,] instead (KokoroVoice.TryReadCanonicalFloat32), which made loading 1.5-1.9x faster than
        //  the np.load_npy route.
        //  The fix: port read_array's file branch - for a seekable file with a native-order dtype, read into the array's
        //  memory directly.
        // ============================================================================

        /// <summary>
        ///     <c>np.load_npy</c> of a 522 KB float32 file (a KokoroSharp voice, 510 x 1 x 256) allocates less than 16 KB on the
        ///     managed heap: the array itself is unmanaged, so only the header parse remains. Today it allocates about 262 KB:
        ///     the 256 KB staging buffer.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "consumer-kokorosharp-load-npy-staging-buffer")]
        public void LoadNpy_OfAFile_AllocatesNoStagingBuffer()
        {
            string path = Path.Combine(Path.GetTempPath(), $"numsharp-load-npy-{Guid.NewGuid():N}.npy");
            try
            {
                using (NDArray random = np.random.RandomState(1).rand(510, 1, 256))
                using (NDArray voice = random.astype(NPTypeCode.Single))
                    np.save(path, voice);
                using (np.load_npy(path)) { }   // warm: the first call also builds NpyFormat's one-time state

                long before = GC.GetAllocatedBytesForCurrentThread();
                using (NDArray loaded = np.load_npy(path))
                {
                    long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
                    loaded.shape.Should().Equal(510L, 1L, 256L);
                    allocated.Should().BeLessThan(16 * 1024, $"reading a file needs no staging buffer (NumPy reads straight into the array); np.load_npy allocated {allocated / 1024.0:F1} KB");
                }
            }
            finally
            {
                File.Delete(path);
            }
        }

        // ============================================================================
        //  GAP: no CI job runs NumSharp on the Mono runtime.
        //
        //  .NET for Android, iOS and WebAssembly run NumSharp on Mono, whose JIT compiles some helpers RyuJIT intrinsifies
        //  into calls. One such helper (Math.Truncate inside Converts.ToInt32) made every float -> small-int cast 5.5x slower
        //  than a plain C# loop on Android, and nothing noticed until KokoroSharp ran on a phone (fixed on 2026-11-09). A
        //  job that runs the unit suite with UseMonoRuntime=true (a self-contained linux-x64 Mono, which KokoroSharp's
        //  packaging check used), plus a NumSharp-against-a-plain-loop smoke, would catch the next one.
        // ============================================================================

        /// <summary>
        ///     Some workflow runs tests with <c>UseMonoRuntime=true</c>. Today none does. Inconclusive when the repository's
        ///     <c>.github/workflows</c> folder is not above the test binaries (a test run outside a checkout).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "consumer-kokorosharp-mono-ci-leg")]
        public void Ci_RunsTheSuiteOnTheMonoRuntime()
        {
            DirectoryInfo dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null && !Directory.Exists(Path.Combine(dir.FullName, ".github", "workflows")))
                dir = dir.Parent;
            if (dir is null)
                Assert.Inconclusive("no .github/workflows above the test binaries");

            string[] workflows = Directory.GetFiles(Path.Combine(dir.FullName, ".github", "workflows"), "*.yml");
            bool monoLeg = workflows.Any(file => File.ReadAllLines(file).Any(line => line.Contains("UseMonoRuntime=true")));
            monoLeg.Should().BeTrue($"a CI job should run NumSharp on Mono (Android / iOS / WebAssembly's runtime); none of {workflows.Length} workflows passes UseMonoRuntime=true");
        }
    }
}
