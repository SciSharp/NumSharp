using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Interop.Blas;

namespace NumSharp.UnitTest.Backends
{
    /// <summary>
    ///     The contract around the optional <c>NumSharp.Interop.BLAS</c> backend — that installing
    ///     it is opt-in, that the engine falls back to its own managed kernels for everything the
    ///     backend does not serve, and that a named library is never silently substituted. The BITS
    ///     are gated separately by the host-pinned <c>matmul_parity</c> corpus tier
    ///     (<c>FuzzCorpusTests.MatmulParity</c>); everything here must hold on any machine, and the
    ///     tests that need a real BLAS say so and go Inconclusive without one.
    /// </summary>
    [TestClass]
    public class MatmulParityBackendTests
    {
        [TestCleanup]
        public void Cleanup() => Blas.Disable();

        /// <summary>A BLAS-less machine must still be able to run these; skip loudly if so.</summary>
        private static void RequireBlas()
        {
            try
            {
                Blas.Enable();
            }
            catch (Exception e)
            {
                Assert.Inconclusive("no CBLAS library on this host: " + e.Message.Split('\n')[0]);
            }
        }

        [TestMethod]
        public void NotInstalled_ByDefault_TheEngineHasNoBlasBackend()
        {
            // NumSharp.Core is 100% managed: nothing may install a native backend implicitly. (The
            // test assembly suppresses the package's module-load auto-install — see
            // BlasEngineAutoInstallGuard — precisely so this stays observable.)
            Blas.Disable();
            Assert.IsFalse(Blas.Enabled);
            Assert.IsNull(BackendFactory.GetEngine().Blas);
        }

        [TestMethod]
        public void Enable_SetsTheBackendOnTheEngine_WithoutReplacingIt()
        {
            RequireBlas();
            Assert.IsTrue(Blas.Enabled);
            var engine = BackendFactory.GetEngine();
            // The engine itself is untouched — still the built-in one, now holding a backend.
            Assert.IsInstanceOfType(engine, typeof(DefaultEngine));
            Assert.IsInstanceOfType(engine.Blas, typeof(IBlasBackend));
            Assert.IsInstanceOfType(engine.Blas, typeof(OpenBlasBackend));
        }

        [TestMethod]
        public void Disable_ClearsTheBackend()
        {
            RequireBlas();
            Blas.Disable();
            Assert.IsFalse(Blas.Enabled);
            Assert.IsNull(BackendFactory.GetEngine().Blas);
            Assert.IsInstanceOfType(BackendFactory.GetEngine(), typeof(DefaultEngine));
        }

        [TestMethod]
        public void TheBackendIsAPlainSettableProperty()
        {
            // The seam is a property on the engine, not a type swap: Core exposes IBlasBackend and
            // nothing more, so any implementation can be dropped in or taken out at will.
            RequireBlas();
            var engine = BackendFactory.GetEngine();
            IBlasBackend backend = engine.Blas;
            Assert.IsNotNull(backend);
            StringAssert.Contains(backend.Info, "symbols");

            engine.Blas = null;
            Assert.IsFalse(Blas.Enabled);
            engine.Blas = backend;
            Assert.IsTrue(Blas.Enabled);
        }

        [TestMethod]
        public void Disable_IsAlwaysSafe_EvenWithNothingInstalled()
        {
            Blas.Disable();
            Blas.Disable();
            Assert.IsFalse(Blas.Enabled);
        }

        [TestMethod]
        public void NamedLibrary_IsBinding_NeverSilentlySubstituted()
        {
            // The whole claim is "the bits of THIS binary", so a named library that cannot be
            // loaded must fail — even on a machine where auto-discovery would happily find NumPy's
            // OpenBLAS and produce confidently wrong bits.
            var ex = Assert.ThrowsException<DllNotFoundException>(
                () => Blas.Enable("K:/definitely/not/a/blas/here.dll"));
            StringAssert.Contains(ex.Message, "no other one is substituted");
            Assert.IsFalse(Blas.Enabled, "a failed load must leave the backend uninstalled");
        }

        [TestMethod]
        public void NotACBlasProvider_Throws()
        {
            // The running test assembly is a perfectly loadable PE that exports no cblas_sgemm.
            var self = typeof(MatmulParityBackendTests).Assembly.Location;
            Assert.ThrowsException<EntryPointNotFoundException>(() => Blas.Enable(self));
            Assert.IsFalse(Blas.Enabled);
        }

        [TestMethod]
        public void TryEnable_ReportsFailureInsteadOfThrowing()
        {
            // The module-load path: referencing the package must never break an app that has no
            // native BLAS anywhere.
            Assert.IsFalse(Blas.TryEnable("K:/definitely/not/a/blas/here.dll"));
            Assert.IsFalse(Blas.Enabled);
        }

        [TestMethod]
        public void Info_DescribesTheLoadedLibrary_OrIsNull()
        {
            RequireBlas();
            var info = Blas.Info;
            Assert.IsNotNull(info);
            StringAssert.Contains(info, "symbols");
            StringAssert.Contains(info, "threads");
        }

        [TestMethod]
        public void Backend_ProducesTheSameSHAPE_AndDTYPE_AsTheManagedKernel()
        {
            // The backend hangs off the engine, and engines are cached singletons — so the SAME
            // arrays switch implementation when it is installed or cleared. (That is the advantage
            // of a property over swapping the engine: an NDArray binds its engine at construction,
            // so a swapped engine would only ever reach arrays created afterwards.)
            var a = np.arange(12).astype(NPTypeCode.Single).reshape(3, 4);
            var b = np.arange(20).astype(NPTypeCode.Single).reshape(4, 5);

            Blas.Disable();
            var native = np.dot(a, b);
            RequireBlas();
            var parity = np.dot(a, b);

            CollectionAssert.AreEqual(native.shape, parity.shape);
            Assert.AreEqual(native.typecode, parity.typecode);
        }

        [TestMethod]
        public void IntegerProducts_FallThroughToTheManagedKernel()
        {
            // Integer addition is associative even when it wraps, so summation order cannot change
            // an integer matrix product — the backend deliberately returns false for these and the
            // engine computes them itself.
            var a = np.arange(6).astype(NPTypeCode.Int32).reshape(2, 3);
            var b = np.arange(6).astype(NPTypeCode.Int32).reshape(3, 2);

            Blas.Disable();
            var native = np.matmul(a, b);
            RequireBlas();
            var parity = np.matmul(a, b);

            Assert.AreEqual(NPTypeCode.Int32, parity.typecode);
            CollectionAssert.AreEqual(native.ToArray<int>(), parity.ToArray<int>());
        }

        [TestMethod]
        public void EveryShapeRoute_RoundTripsThroughTheBackend()
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
        public void EveryOtherOperation_StaysTheManagedImplementation()
        {
            // The backend answers two entry points; installing it must not change anything else.
            RequireBlas();
            var a = np.arange(6).astype(NPTypeCode.Double).reshape(2, 3);
            Assert.AreEqual(15.0, (double)np.sum(a));
            Assert.AreEqual(2.0, (double)np.max(a[0]));
            CollectionAssert.AreEqual(new long[] { 3, 2 }, np.transpose(a).shape);
            CollectionAssert.AreEqual(new long[] { 2, 3 }, (a + a).shape);
            Assert.AreEqual(NPTypeCode.Double, np.sqrt(a).typecode);
        }

        [TestMethod]
        public void ThreadPin_IsReportedBack()
        {
            RequireBlas();
            Blas.Enable(threads: 1);
            var info = Blas.Info;
            // OpenBLAS reports the count back; a reference CBLAS has no such entry point (-1).
            Assert.IsTrue(info.Contains("threads 1") || info.Contains("threads -1"), info);
        }
    }
}
