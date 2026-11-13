using System;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Issues
{
    /// <summary>
    ///     Backs the answer that closed GitHub #551, "[NEP05 NEP20] Generalized Universal Functions"
    ///     (https://github.com/SciSharp/NumSharp/issues/551): <c>np.vectorize(pyfunc, signature)</c> is NumPy's core-dimension
    ///     machinery, and <c>np.matmul</c>'s four flexible-dimension cases give NumPy's shapes. Expected values are NumPy
    ///     2.4.2's.
    /// </summary>
    [TestClass]
    public class Issue551
    {
        /// <summary>The issue's own example: an inner product over the core dimension <c>(i),(i)->()</c>.</summary>
        /// <returns>The vectorized inner product, NumPy's <c>np.vectorize(lambda x, y: np.sum(x * y), signature="(i),(i)->()")</c>.</returns>
        private static Vectorized Inner() =>
            np.vectorize((Func<NDArray, NDArray, NDArray>)((x, y) => np.sum(x * y)), "(i),(i)->()");

        /// <summary>
        ///     Loop dimensions broadcast and the core dimension is consumed: (3, 5, 4) with (5, 4) gives a (3, 5) result holding
        ///     NumPy's 15 inner products.
        /// </summary>
        [TestMethod]
        public void Vectorize_Signature_BroadcastsLoopDims_AndConsumesTheCoreDim()
        {
            var r = Inner().Call(np.arange(60.0).reshape(3, 5, 4), np.arange(20.0).reshape(5, 4));

            r.Should().BeShaped(3, 5).And.BeOfValues(
                14.0, 126.0, 366.0, 734.0, 1230.0, 134.0, 566.0, 1126.0, 1814.0, 2630.0, 254.0, 1006.0, 1886.0, 2894.0, 4030.0);
        }

        /// <summary>
        ///     A core dimension that disagrees between the operands raises NumPy's ValueError text.
        /// </summary>
        [TestMethod]
        public void Vectorize_Signature_CoreDimMismatch_RaisesNumPysText()
        {
            new Action(() => Inner().Call(np.ones((3, 5, 4)), np.ones((5, 3)))).Should().Throw<ValueError>()
                .WithMessage("inconsistent size for core dimension 'i': 3 vs 4");
        }

        /// <summary>
        ///     NEP 20's flexible dimensions in <c>np.matmul</c>: <c>(m,n)(n,p)</c> is (m, p), <c>(n)(n,p)</c> is (p),
        ///     <c>(m,n)(n)</c> is (m) and <c>(n)(n)</c> is 0-d, with NumPy's values.
        /// </summary>
        [TestMethod]
        public void Matmul_FlexibleDimensions_GiveNumPysShapes()
        {
            var m = np.ones((2, 3));
            var n = np.ones((3, 4));
            var v = np.ones(new Shape(3));

            np.matmul(m, n).Should().BeShaped(2, 4).And.AllValuesBe(3.0);
            np.matmul(v, n).Should().BeShaped(4).And.AllValuesBe(3.0);
            np.matmul(m, v).Should().BeShaped(2).And.AllValuesBe(3.0);
            var dot = np.matmul(v, v);
            dot.ndim.Should().Be(0);
            dot.GetDouble().Should().Be(3.0);
        }
    }
}
