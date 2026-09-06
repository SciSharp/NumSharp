using System;
using System.Linq;

namespace NumSharp.Tests.Backends.Iterators
{
    /// <summary>
    ///     NDAxisIter is the axis-reduction fallback for NON-contiguous / sliced / broadcast arrays
    ///     (var / std / cumsum / cumprod / all / any along an axis). It used to embed a
    ///     <c>fixed long[64]</c> outer-dimension scratch and throw
    ///     <c>NotSupportedException("NDAxisIter currently supports up to 64 dimensions.")</c> for any
    ///     ndim &gt; 64 — a hard cap that contradicted NumSharp's unlimited-dims design (NumPy itself caps
    ///     at 64, but NumSharp allows more, and every other op already works past 64). The scratch is now
    ///     caller-stackalloc'd, sized by ndim, so there is no cap.
    ///
    ///     These arrays are mostly size-1 padding — the realistic high-rank shape — so the reduction VALUES
    ///     are identical to the equivalent low-rank array; only the number of length-1 axes differs. Values
    ///     were separately verified bit-identical to NumPy 2.4.2 at ndim=64 (its maximum rank).
    /// </summary>
    [TestClass]
    public class NDAxisIterNoNdimCapTests
    {
        // arange(24).reshape([2,3,1,...,1,4]).T — non-contiguous (transposed), 2*3*4=24 non-unit elements,
        // reduced along axis 0. The transpose forces the NDAxisIter path (IsContiguous == false).
        private static NDArray BuildNonContiguous(int ndim)
        {
            var dims = new int[ndim];
            for (int i = 0; i < ndim; i++) dims[i] = 1;
            dims[0] = 2; dims[1] = 3; dims[ndim - 1] = 4;
            return np.arange(24).astype(NPTypeCode.Double).reshape(new Shape(dims)).T;
        }

        private static double[] Flat(NDArray a)
        {
            var f = a.flatten();
            return Enumerable.Range(0, (int)f.size).Select(i => Convert.ToDouble(f.GetAtIndex(i))).ToArray();
        }

        [DataTestMethod]
        [DataRow(65)]
        [DataRow(70)]
        [DataRow(200)]
        [DataRow(1000)]
        public void AxisReductions_AboveNumpyMaxRank_NoCap_MatchLowRank(int ndim)
        {
            var baseView = BuildNonContiguous(3);   // 3-D reference, definitely on the same NDAxisIter path
            var highView = BuildNonContiguous(ndim); // ndim > 64 used to throw

            baseView.Shape.IsContiguous.Should().BeFalse("the transposed reference must exercise NDAxisIter");
            highView.Shape.IsContiguous.Should().BeFalse();

            Flat(np.var(highView, axis: 0)).Should().Equal(Flat(np.var(baseView, axis: 0)), $"var ndim={ndim}");
            Flat(np.std(highView, axis: 0)).Should().Equal(Flat(np.std(baseView, axis: 0)), $"std ndim={ndim}");
            Flat(np.cumsum(highView, axis: 0)).Should().Equal(Flat(np.cumsum(baseView, axis: 0)), $"cumsum ndim={ndim}");
            Flat(np.cumprod(highView, axis: 0)).Should().Equal(Flat(np.cumprod(baseView, axis: 0)), $"cumprod ndim={ndim}");
            Flat(np.all(highView, axis: 0)).Should().Equal(Flat(np.all(baseView, axis: 0)), $"all ndim={ndim}");
            Flat(np.any(highView, axis: 0)).Should().Equal(Flat(np.any(baseView, axis: 0)), $"any ndim={ndim}");
        }

        [TestMethod]
        public void AxisAll_NonContiguous65d_DoesNotThrow()
        {
            // The exact call that used to throw NotSupportedException at ndim=65.
            var v = BuildNonContiguous(65);
            Action act = () => np.all(v, axis: 0);
            act.Should().NotThrow();
            np.all(v, axis: 0).size.Should().Be(6);   // output = the 3×2 non-unit tail
        }
    }
}
