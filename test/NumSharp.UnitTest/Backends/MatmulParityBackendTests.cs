using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.Backends
{
    /// <summary>
    ///     The <see cref="np.parity_matmul(bool,string,int)"/> switch itself — the contract around
    ///     the backend, not the bits. The bits are gated by the host-pinned
    ///     <c>matmul_parity</c> corpus tier (<c>FuzzCorpusTests.MatmulParity</c>); everything here
    ///     must hold on any machine, and the few tests that need a real BLAS say so and go
    ///     Inconclusive without one.
    /// </summary>
    [TestClass]
    public class MatmulParityBackendTests
    {
        [TestCleanup]
        public void Cleanup() => np.parity_matmul(false);

        /// <summary>A BLAS-less machine must still be able to run these; skip loudly if so.</summary>
        private static void RequireBlas()
        {
            try
            {
                np.parity_matmul(true);
            }
            catch (Exception e)
            {
                Assert.Inconclusive("no CBLAS library on this host: " + e.Message.Split('\n')[0]);
            }
        }

        [TestMethod]
        public void Disabled_ByDefault()
        {
            // Nothing in the library may turn this on implicitly — the native GEMM is the default.
            np.parity_matmul(false);
            Assert.IsFalse(np.parity_matmul_enabled);
        }

        [TestMethod]
        public void Disable_IsAlwaysSafe_EvenWithNoLibraryLoaded()
        {
            np.parity_matmul(false);
            np.parity_matmul(false);
            Assert.IsFalse(np.parity_matmul_enabled);
        }

        [TestMethod]
        public void NamedLibrary_IsBinding_NeverSilentlySubstituted()
        {
            // The whole claim is "the bits of THIS binary", so a named library that cannot be
            // loaded must fail — even on a machine where auto-discovery would happily find NumPy's
            // OpenBLAS and produce confidently wrong bits.
            var ex = Assert.ThrowsException<DllNotFoundException>(
                () => np.parity_matmul(true, "K:/definitely/not/a/blas/here.dll"));
            StringAssert.Contains(ex.Message, "no other one is substituted");
            Assert.IsFalse(np.parity_matmul_enabled, "a failed load must leave the switch off");
        }

        [TestMethod]
        public void NotACBlasProvider_Throws()
        {
            // The running test assembly is a perfectly loadable PE that exports no cblas_sgemm.
            var self = typeof(MatmulParityBackendTests).Assembly.Location;
            Assert.ThrowsException<EntryPointNotFoundException>(() => np.parity_matmul(true, self));
            Assert.IsFalse(np.parity_matmul_enabled);
        }

        [TestMethod]
        public void Enable_ReportsTheLoadedLibrary()
        {
            RequireBlas();
            Assert.IsTrue(np.parity_matmul_enabled);
            var info = np.parity_matmul_info();
            Assert.IsNotNull(info);
            StringAssert.Contains(info, "symbols");
            StringAssert.Contains(info, "threads");
        }

        [TestMethod]
        public void Info_IsNull_BeforeAnyLibraryIsLoaded_OrDescribesOne()
        {
            // Once a library is loaded it stays loaded (unloading a BLAS mid-process is not worth
            // the risk), so this asserts the only invariant that always holds.
            var info = np.parity_matmul_info();
            Assert.IsTrue(info == null || info.Contains("symbols"));
        }

        [TestMethod]
        public void ParityPath_ProducesTheSameSHAPE_AndDTYPE_AsTheNativePath()
        {
            RequireBlas();
            var a = np.arange(12).astype(NPTypeCode.Single).reshape(3, 4);
            var b = np.arange(20).astype(NPTypeCode.Single).reshape(4, 5);

            var parity = np.dot(a, b);
            np.parity_matmul(false);
            var native = np.dot(a, b);

            CollectionAssert.AreEqual(native.shape, parity.shape);
            Assert.AreEqual(native.typecode, parity.typecode);
        }

        [TestMethod]
        public void IntegerProducts_KeepTheNativePath_AndAreUnaffected()
        {
            // Integer addition is associative even when it wraps, so summation order cannot change
            // an integer matrix product — the parity backend deliberately does not claim these.
            var a = np.arange(6).astype(NPTypeCode.Int32).reshape(2, 3);
            var b = np.arange(6).astype(NPTypeCode.Int32).reshape(3, 2);

            np.parity_matmul(false);
            var native = np.matmul(a, b);
            RequireBlas();
            var parity = np.matmul(a, b);

            Assert.AreEqual(NPTypeCode.Int32, parity.typecode);
            CollectionAssert.AreEqual(native.ToArray<int>(), parity.ToArray<int>());
        }

        [TestMethod]
        public void EveryShapeRoute_RoundTripsThroughTheParityBackend()
        {
            // Values are not asserted here (that is the corpus tier's job, pinned to a host) — this
            // asserts the port never throws or mis-shapes on any of the routes it dispatches:
            // row*column dot, both gemv directions, column@row, scalar_vec, gemm, syrk, batched,
            // N-D dot, and the zero-sized fallbacks.
            RequireBlas();
            var v = np.arange(5).astype(NPTypeCode.Double);
            var m = np.arange(15).astype(NPTypeCode.Double).reshape(3, 5);
            var n = np.arange(20).astype(NPTypeCode.Double).reshape(5, 4);

            CollectionAssert.AreEqual(new long[0], np.dot(v, v).shape);
            CollectionAssert.AreEqual(new long[] { 3 }, np.dot(m, v).shape);
            CollectionAssert.AreEqual(new long[] { 4 }, np.dot(v, n).shape);
            CollectionAssert.AreEqual(new long[] { 3, 4 }, np.dot(m, n).shape);
            CollectionAssert.AreEqual(new long[] { 3, 4 }, np.matmul(m, n).shape);
            CollectionAssert.AreEqual(new long[] { 3, 3 }, np.dot(m, m.T).shape);          // syrk
            CollectionAssert.AreEqual(new long[] { 5, 5 }, np.dot(m.T, m).shape);          // syrk
            CollectionAssert.AreEqual(new long[] { 5, 4 },
                np.dot(v.reshape(5, 1), np.arange(4).astype(NPTypeCode.Double).reshape(1, 4)).shape);
            CollectionAssert.AreEqual(new long[] { 1, 1 },
                np.dot(v.reshape(1, 5), v.reshape(5, 1)).shape);
            CollectionAssert.AreEqual(new long[] { 2, 3, 4 },
                np.matmul(np.arange(30).astype(NPTypeCode.Double).reshape(2, 3, 5), n).shape);
            CollectionAssert.AreEqual(new long[] { 2, 3, 4 },
                np.dot(np.arange(30).astype(NPTypeCode.Double).reshape(2, 3, 5), n).shape);
            CollectionAssert.AreEqual(new long[] { 3, 4 },
                np.dot(np.zeros(new Shape(3, 0), NPTypeCode.Double),
                       np.zeros(new Shape(0, 4), NPTypeCode.Double)).shape);
        }

        [TestMethod]
        public void StridedAndTransposedViews_AreAccepted_NotSilentlyWrong()
        {
            RequireBlas();
            var baseA = np.arange(60).astype(NPTypeCode.Single).reshape(6, 10);
            var strided = baseA[":, ::2"];            // (6,5), last-axis stride 2 — not blasable
            var reversed = baseA["::-1"];             // negative row stride
            var b = np.arange(25).astype(NPTypeCode.Single).reshape(5, 5);

            CollectionAssert.AreEqual(new long[] { 6, 5 }, np.dot(strided, b).shape);
            CollectionAssert.AreEqual(new long[] { 6, 5 }, np.matmul(strided, b).shape);
            CollectionAssert.AreEqual(new long[] { 6, 6 }, np.dot(reversed, baseA.T).shape);
        }

        [TestMethod]
        public void MixedPrecisionOperands_PromoteLikeNumPy()
        {
            RequireBlas();
            var f = np.arange(6).astype(NPTypeCode.Single).reshape(2, 3);
            var d = np.arange(6).astype(NPTypeCode.Double).reshape(3, 2);
            Assert.AreEqual(NPTypeCode.Double, np.dot(f, d).typecode);
            Assert.AreEqual(NPTypeCode.Double, np.dot(d, f).typecode);
        }

        [TestMethod]
        public void ThreadPin_IsReportedBack()
        {
            RequireBlas();
            np.parity_matmul(true, threads: 1);
            var info = np.parity_matmul_info();
            // OpenBLAS reports the count back; a reference CBLAS has no such entry point (-1).
            Assert.IsTrue(info.Contains("threads 1") || info.Contains("threads -1"), info);
        }
    }
}
