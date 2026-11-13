using System;
using System.Linq;
using NumSharp.Backends;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Issues
{
    /// <summary>
    ///     Backs the progress comment on GitHub #627, "[OpenBLAS] Port the LAPACK factorisations to a native managed C#
    ///     backend" (https://github.com/SciSharp/NumSharp/issues/627), which stays open: Phase 1 (LU) computes in managed C#
    ///     without a backend, and the other factorisations still need NumSharp.Interop.OpenBLAS, refusing with an exception
    ///     that names the package. Expected values are NumPy 2.4.2's.
    /// </summary>
    [TestClass]
    public class Issue627
    {
        /// <summary>The symmetric positive-definite matrix both tests use.</summary>
        private static NDArray M => np.array(new[,] { { 4.0, 2.0 }, { 2.0, 3.0 } });

        /// <summary>Makes the test inconclusive when a backend is bound, because then nothing here runs the managed path.</summary>
        /// <exception cref="AssertInconclusiveException">
        ///     <c>TensorEngine.Blas</c> is set (a <c>[DoNotParallelize]</c> OpenBLAS test left it bound), so neither the managed
        ///     LU nor the missing-backend refusal is reachable.
        /// </exception>
        private static void RequireNoBackend()
        {
            if (M.TensorEngine.Blas != null)
                Assert.Inconclusive("a BLAS backend is bound in this process, so the managed path is not the one under test");
        }

        /// <summary>
        ///     Phase 1: <c>inv</c>, <c>det</c>, <c>solve</c> and <c>slogdet</c> compute without a backend, to NumPy's values
        ///     within 1e-12 (the managed LU is correct, not bit-identical to LAPACK).
        /// </summary>
        [TestMethod]
        public void Phase1_LuFamily_ComputesWithoutABackend()
        {
            RequireNoBackend();

            var inv = np.linalg.inv(M).ToArray<double>();
            new[] { 0.375, -0.25, -0.25, 0.5 }.Zip(inv).Should().OnlyContain(p => System.Math.Abs(p.First - p.Second) < 1e-12);
            np.linalg.det(M).GetDouble().Should().BeApproximately(7.999999999999998, 1e-12);
            var x = np.linalg.solve(M, np.array(new[] { 1.0, 2.0 })).ToArray<double>();
            x[0].Should().BeApproximately(-0.125, 1e-12);
            x[1].Should().BeApproximately(0.75, 1e-12);
            var (sign, logdet) = np.linalg.slogdet(M);
            sign.GetDouble().Should().Be(1.0);
            logdet.GetDouble().Should().BeApproximately(2.0794415416798357, 1e-12);
        }

        /// <summary>
        ///     Phases 2-4: without a backend, <c>cholesky</c>, <c>qr</c>, <c>svd</c>, <c>eig</c>, <c>eigh</c>, <c>lstsq</c>,
        ///     <c>pinv</c> and <c>matrix_rank</c> raise <c>OpenBlasMissingBackendException</c>, whose message names the package.
        /// </summary>
        [TestMethod]
        public void Phases2To4_RaiseTheMissingBackendException_NamingThePackage()
        {
            RequireNoBackend();

            var calls = new (string name, Action call)[]
            {
                ("cholesky", () => np.linalg.cholesky(M)), ("qr", () => np.linalg.qr(M)), ("svd", () => np.linalg.svd(M)),
                ("eig", () => np.linalg.eig(M)), ("eigh", () => np.linalg.eigh(M)),
                ("lstsq", () => np.linalg.lstsq(M, np.array(new[] { 1.0, 2.0 }))),
                ("pinv", () => np.linalg.pinv(M)), ("matrix_rank", () => np.linalg.matrix_rank(M)),
            };
            foreach (var (name, call) in calls)
                call.Should().Throw<OpenBlasMissingBackendException>(name)
                    .Which.Message.Should().Contain(OpenBlasMissingBackendException.PackageId, name);
        }
    }
}
