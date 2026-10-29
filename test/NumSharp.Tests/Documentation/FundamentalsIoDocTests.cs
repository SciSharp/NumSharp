using System;
using System.Collections.Generic;
using System.IO;
using NumSharp;
using NumSharp.IO;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Documentation
{
    /// <summary>
    ///     Executable coverage for every code example in
    ///     <c>docs/website-src/docs/fundamentals/io.md</c> (the "I/O with NumSharp" fundamentals
    ///     article). Each test mirrors one documented snippet and asserts the behaviour the page
    ///     claims. Values were captured by running the snippets against this branch.
    /// </summary>
    [TestClass]
    public class FundamentalsIoDocTests
    {
        private string _dir;

        [TestInitialize]
        public void Setup()
        {
            _dir = Path.Combine(Path.GetTempPath(), "ns_fund_io_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(_dir);
        }

        [TestCleanup]
        public void Cleanup()
        {
            try { Directory.Delete(_dir, recursive: true); }
            catch (IOException) { /* a leaked handle fails the test that leaked it */ }
        }

        private string At(string name) => Path.Combine(_dir, name);

        // ── .npy / .npz ────────────────────────────────────────────────────────────────────────────

        [TestMethod]
        public void Npy_SaveLoad_RoundTrip()
        {
            var arr = np.arange(10).reshape(2, 5);
            np.save(At("array.npy"), arr);
            NDArray w = np.load_npy(At("array.npy"));
            np.array_equal(w, arr).Should().BeTrue();
        }

        [TestMethod]
        public void Npz_DictionaryKeys_And_DotAccess()
        {
            np.savez(At("bundle.npz"), new Dictionary<string, NDArray> { ["weights"] = np.arange(3), ["bias"] = np.zeros(2) });

            using NpzFile npz = np.load_npz(At("bundle.npz"));
            npz["weights"].ToArray<long>().Should().Equal(0L, 1, 2);      // by key
            NDArray b = npz.f.bias;                                       // BagObj dot access
            b.size.Should().Be(2);
            npz.Files.Should().BeEquivalentTo(new[] { "weights", "bias" });
        }

        [TestMethod]
        public void Load_ReturnsObject_DispatchedOnKind()
        {
            np.save(At("one.npy"), np.arange(3));
            np.savez(At("many.npz"), np.arange(3));

            np.load(At("one.npy")).Should().BeOfType<NDArray>();
            object fromNpz = np.load(At("many.npz"));
            fromNpz.Should().BeOfType<NpzFile>();
            ((NpzFile)fromNpz).Dispose();
        }

        [TestMethod]
        public void Mmap_Mode_R_IsReadOnly()
        {
            np.save(At("huge.npy"), np.arange(5));
            var mapped = (NDArray)np.load(At("huge.npy"), mmap_mode: "r");
            mapped.Shape.IsWriteable.Should().BeFalse("mmap_mode 'r' is a read-only view");
            mapped.ToArray<long>().Should().Equal(0L, 1, 2, 3, 4);
        }

        // ── Text I/O ───────────────────────────────────────────────────────────────────────────────

        [TestMethod]
        public void SaveTxt_LoadTxt_RoundTrip()
        {
            var a = np.array(new[,] { { 1.0, 2.0 }, { 3.0, 4.0 } });
            np.savetxt(At("out.csv"), a, fmt: "%.1f", delimiter: ",");
            var back = np.loadtxt(At("out.csv"), delimiter: ",");
            back.shape.Should().Equal(new long[] { 2, 2 });
            back.ToArray<double>().Should().Equal(1.0, 2.0, 3.0, 4.0);
        }

        [TestMethod]
        public void FromString_ParsesNumbers()
        {
            np.fromstring("1 2 3 4", sep: " ").ToArray<double>().Should().Equal(1.0, 2.0, 3.0, 4.0);
            np.fromstring("1,2,3", np.int32, sep: ",").ToArray<int>().Should().Equal(1, 2, 3);
        }

        [TestMethod]
        public void FromString_BinaryModeRemoved_Throws()
        {
            ((Action)(() => np.fromstring("garbage"))).Should().Throw<Exception>("binary mode of fromstring was removed");
        }

        // ── Raw binary ─────────────────────────────────────────────────────────────────────────────

        [TestMethod]
        public void Raw_TofileFromfile_RoundTrip()
        {
            var arr = np.array(new[] { 1.0, 2.0, 3.0 });
            arr.tofile(At("data.bin"));
            np.fromfile(At("data.bin"), np.float64).ToArray<double>().Should().Equal(1.0, 2.0, 3.0);
        }
    }
}
