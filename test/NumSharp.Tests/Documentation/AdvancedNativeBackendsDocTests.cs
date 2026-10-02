using System;
using NumSharp;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Documentation
{
    /// <summary>
    ///     Executable coverage for the backend-INDEPENDENT code examples in
    ///     <c>docs/website-src/docs/advanced/native-backends.md</c> ("Native code &amp; backends").
    ///     The page's OpenBLAS-specific claims (byte-identical products with a backend; a factorisation
    ///     raising <see cref="NotSupportedException"/> without one) depend on the global
    ///     <c>TensorEngine.Blas</c> state, which other suites toggle per-test, so they are gated by
    ///     <c>LinAlgEngineSeamTests</c> and the <c>NumSharp.Interop.OpenBLAS</c> test project rather than
    ///     re-asserted here. This file covers the snippets whose result does not depend on whether a
    ///     backend is installed.
    /// </summary>
    [TestClass]
    public class AdvancedNativeBackendsDocTests
    {
        [TestMethod]
        public void IntegerProducts_ComputeManaged_BitExactByConstruction()
        {
            // Integer/bool products never route to BLAS (modular arithmetic is associative), so the
            // result is the same with or without a backend.
            var r = np.dot(np.array(new[,] { { 1, 2 }, { 3, 4 } }),
                           np.array(new[,] { { 5, 6 }, { 7, 8 } }));
            r.ToArray<int>().Should().Equal(19, 22, 43, 50);
        }

        [TestMethod]
        public void LuFamily_Computes()
        {
            // det of [[4,3],[6,3]] = 4*3 - 3*6 = -6 — via the managed LU fallback, or LAPACK if a
            // backend happens to be installed; either way the value is the same.
            var m = np.array(new[,] { { 4.0, 3.0 }, { 6.0, 3.0 } });
            ((double)np.linalg.det(m)).Should().BeApproximately(-6.0, 1e-9);
        }
    }
}
