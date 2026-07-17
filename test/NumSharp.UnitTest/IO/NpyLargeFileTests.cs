using System;
using System.IO;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.IO;

namespace NumSharp.UnitTest.IO
{
    /// <summary>
    ///     Files past the 2 GB and 4 GB walls.
    /// </summary>
    /// <remarks>
    ///     Three separate 32-bit limits sit under this format and each bites at a different size:
    ///     <list type="bullet">
    ///       <item><b>2 GB</b> — <c>MemoryStream</c> and <c>byte[]</c> cannot exceed it, so any code that
    ///             buffers a whole member or file into managed memory caps out here. NumSharp's array
    ///             data is unmanaged, so nothing on the .npy path needs a managed buffer bigger than one
    ///             256 KB chunk.</item>
    ///       <item><b>4 GB</b> — a zip entry above <c>0xFFFFFFFF</c> needs Zip64 records. NumPy passes
    ///             <c>force_zip64=True</c> for exactly this (numpy gh-10776); .NET's ZipArchive emits
    ///             them on demand.</item>
    ///       <item><b>int.MaxValue elements</b> — every offset, count and stride in the read/write loops
    ///             must be 64-bit.</item>
    ///     </list>
    ///     The multi-gigabyte tests are <see cref="HighMemoryAttribute"/> (excluded from CI) and want
    ///     ~8 GB of RAM plus ~6 GB of scratch disk. <see cref="Npz_StreamsMembers_RatherThanBufferingThem"/>
    ///     is the cheap proxy that runs everywhere: it guards the same mechanism the 2 GB wall exposed.
    /// </remarks>
    [TestClass]
    public class NpyLargeFileTests
    {
        private string _dir;

        [TestInitialize]
        public void Setup()
        {
            _dir = Path.Combine(Path.GetTempPath(), "numsharp_big_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(_dir);
        }

        [TestCleanup]
        public void Cleanup()
        {
            try { Directory.Delete(_dir, recursive: true); }
            catch (IOException) { }
        }

        /// <summary>
        ///     Reading an npz member must stream it, not buffer it.
        /// </summary>
        /// <remarks>
        ///     This is the cheap guard for a bug that only showed at 5 GB: members used to be copied into
        ///     a <c>MemoryStream</c> so the magic could be sniffed by seeking, which capped members at
        ///     2 GB — NumSharp would happily WRITE a 5 GiB npz and then fail to read it back with
        ///     "Stream was too long", while NumPy read the same file fine. Buffering also doubled the
        ///     peak cost of every ordinary load. A 16 MB member is enough to catch a regression: if the
        ///     member is being buffered, the managed allocation tracks its size.
        /// </remarks>
        [TestMethod]
        [TestCategory("NpyOracle")]
        [DoNotParallelize]
        public void Npz_StreamsMembers_RatherThanBufferingThem()
        {
            const int n = 16 * 1024 * 1024;
            byte[] archive = np.savez(np.zeros(new Shape(n), NPTypeCode.Byte));

            long allocated = AllocationTests.MinAllocated(() =>
            {
                using NpzFile npz = np.load_npz(archive);
                Assert.AreEqual(n, npz["arr_0"].size);
            });

            Assert.IsTrue(allocated < n / 4,
                $"loading a {n / 1024 / 1024} MB npz member allocated {allocated:N0} managed bytes " +
                $"({allocated / (double)n:F2}x the member). The member must be streamed into the reader: " +
                "buffering it caps members at MemoryStream's 2 GB limit and doubles the cost of every load.");
        }

        /// <summary>A .npy larger than 4 GB round-trips, and every 32-bit boundary inside it survives.</summary>
        [TestMethod]
        [TestCategory("NpyOracle")]
        [HighMemory]
        [LongIndexing]
        [DoNotParallelize]
        public void Npy_5GiB_RoundTrips()
        {
            const long n = 5L * 1024 * 1024 * 1024;
            string file = Path.Combine(_dir, "big.npy");

            (long Offset, byte Value)[] probes = Probes(n);

            Stamp(file, n, probes, npz: false);

            NDArray loaded = np.load_npy(file);
            Assert.AreEqual(NPTypeCode.Byte, loaded.typecode);
            Assert.AreEqual(n, loaded.size);
            CollectionAssert.AreEqual(new[] { n }, loaded.shape);
            AssertProbes(loaded, probes);

            // 128-byte header + the data, exactly.
            Assert.AreEqual(128 + n, new FileInfo(file).Length);
        }

        /// <summary>
        ///     An npz whose member exceeds 4 GB round-trips — the Zip64 case the issue calls out.
        /// </summary>
        [TestMethod]
        [TestCategory("NpyOracle")]
        [HighMemory]
        [LongIndexing]
        [DoNotParallelize]
        public void Npz_5GiBMember_RoundTrips()
        {
            const long n = 5L * 1024 * 1024 * 1024;
            string file = Path.Combine(_dir, "big.npz");

            (long Offset, byte Value)[] probes = Probes(n);

            Stamp(file, n, probes, npz: true);

            Assert.IsTrue(new FileInfo(file).Length > uint.MaxValue,
                "the archive must exceed 4 GiB, or it is not exercising Zip64 at all");

            using NpzFile npz = np.load_npz(file);
            NDArray loaded = npz["big"];
            Assert.AreEqual(NPTypeCode.Byte, loaded.typecode);
            Assert.AreEqual(n, loaded.size);
            AssertProbes(loaded, probes);
        }

        // Offsets that straddle every 32-bit boundary a narrowing cast would trip over.
        private static (long, byte)[] Probes(long n) => new[]
        {
            (0L, (byte)0x11),
            (int.MaxValue - 1L, (byte)0x22),
            ((long)int.MaxValue, (byte)0x33),
            (int.MaxValue + 1L, (byte)0x44),   // int overflow
            (3_000_000_000L, (byte)0x55),
            (uint.MaxValue - 1L, (byte)0x66),
            (uint.MaxValue + 1L, (byte)0x77),  // uint overflow
            (n - 1, (byte)0x88),               // last element
        };

        // Build the array, stamp the probes, save, then drop it so the load does not need 2x the RAM.
        private static unsafe void Stamp(string file, long n, (long Offset, byte Value)[] probes, bool npz)
        {
            var a = new NDArray(NPTypeCode.Byte, new Shape(n), fillZeros: false);
            byte* p = (byte*)a.Storage.Address;
            foreach ((long off, byte val) in probes)
                p[off] = val;

            if (npz)
                np.savez(file, new System.Collections.Generic.Dictionary<string, NDArray> { ["big"] = a });
            else
                np.save(file, a);

            GC.KeepAlive(a);
            a = null;
            GC.Collect(2, GCCollectionMode.Forced, blocking: true);
            GC.WaitForPendingFinalizers();
            GC.Collect(2, GCCollectionMode.Forced, blocking: true);
        }

        private static unsafe void AssertProbes(NDArray a, (long Offset, byte Value)[] probes)
        {
            byte* p = (byte*)a.Storage.Address;
            foreach ((long off, byte val) in probes)
                Assert.AreEqual(val, p[off], $"element at flat index {off:N0} did not survive the round-trip");
            GC.KeepAlive(a);
        }
    }
}
