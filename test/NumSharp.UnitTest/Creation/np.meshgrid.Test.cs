using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Creation
{
    /// <summary>
    ///     np.meshgrid — coordinate matrices from coordinate vectors. Every expectation below is real
    ///     NumPy 2.4.2 output. meshgrid returns a tuple of N grids (a <see cref="np.MeshgridResult"/>
    ///     here — implicit to <see cref="NDArray"/>[], deconstructable, indexable).
    /// </summary>
    [TestClass]
    public class np_meshgrid_Test
    {
        private static NDArray L(params long[] v) => np.array(v);

        // ------------------------------------------------------------------ 2-D

        [TestMethod]
        public void Meshgrid_TwoVectors_XY_Default()
        {
            // np.meshgrid([1,2,3], [10,20]) -> both (2,3); x repeated down rows, y across.
            var (xx, yy) = np.meshgrid(L(1, 2, 3), L(10, 20));
            xx.Should().BeOfType(NPTypeCode.Int64).And.BeShaped(2, 3)
                .And.BeOfValues(1, 2, 3, 1, 2, 3);
            yy.Should().BeOfType(NPTypeCode.Int64).And.BeShaped(2, 3)
                .And.BeOfValues(10, 10, 10, 20, 20, 20);
        }

        [TestMethod]
        public void Meshgrid_TwoVectors_IJ()
        {
            // 'ij' -> shape (len x, len y) = (3,2); the transpose of 'xy'.
            var (xx, yy) = np.meshgrid(L(1, 2, 3), L(10, 20), indexing: "ij");
            xx.Should().BeShaped(3, 2).And.BeOfValues(1, 1, 2, 2, 3, 3);
            yy.Should().BeShaped(3, 2).And.BeOfValues(10, 20, 10, 20, 10, 20);
        }

        [TestMethod]
        public void Meshgrid_Sparse()
        {
            // sparse 'xy' -> open mesh (1,M) and (N,1).
            var gxy = np.meshgrid(L(1, 2, 3), L(10, 20), sparse: true);
            gxy[0].Should().BeShaped(1, 3).And.BeOfValues(1, 2, 3);
            gxy[1].Should().BeShaped(2, 1).And.BeOfValues(10, 20);

            // sparse 'ij' -> (M,1) and (1,N).
            var gij = np.meshgrid(L(1, 2, 3), L(10, 20), indexing: "ij", sparse: true);
            gij[0].Should().BeShaped(3, 1).And.BeOfValues(1, 2, 3);
            gij[1].Should().BeShaped(1, 2).And.BeOfValues(10, 20);
        }

        // ------------------------------------------------------------------ N-D

        [TestMethod]
        public void Meshgrid_ThreeVectors()
        {
            // 'xy' 3-D: (N1, N0, N2) = (3, 2, 4).
            var (a, b, c) = np.meshgrid(L(1, 2), L(10, 20, 30), L(100, 200, 300, 400));
            a.Should().BeShaped(3, 2, 4);
            b.Should().BeShaped(3, 2, 4);
            c.Should().BeShaped(3, 2, 4);

            // 'ij' 3-D: (N0, N1, N2) = (2, 3, 4).
            var (ai, bi, ci) = np.meshgrid(L(1, 2), L(10, 20, 30), L(100, 200, 300, 400), indexing: "ij");
            ai.Should().BeShaped(2, 3, 4);
            bi.Should().BeShaped(2, 3, 4);
            ci.Should().BeShaped(2, 3, 4);
        }

        [TestMethod]
        public void Meshgrid_FourVectors_ViaArray()
        {
            NDArray[] g = np.meshgrid(new[] { L(0, 1), L(0, 1, 2), L(0, 1, 2, 3), L(0, 1, 2, 3, 4) },
                indexing: "ij");
            g.Length.Should().Be(4);
            foreach (var grid in g)
                grid.Should().BeShaped(2, 3, 4, 5);
        }

        // ------------------------------------------------------------------ degenerate

        [TestMethod]
        public void Meshgrid_SingleVector_IsOneGrid()
        {
            // One input -> a 1-tuple, shape (N,), indexing/sparse have no effect.
            NDArray[] g = np.meshgrid(new[] { L(5, 6, 7) });
            g.Length.Should().Be(1);
            g[0].Should().BeShaped(3).And.BeOfValues(5, 6, 7);
        }

        [TestMethod]
        public void Meshgrid_HigherRankInput_IsFlattened()
        {
            // A non-1-D input is flattened (C-order) to its own axis.
            var m = np.array(new long[,] { { 1, 2 }, { 3, 4 } });   // size 4
            var (xx, yy) = np.meshgrid(m, L(10, 20));
            xx.Should().BeShaped(2, 4).And.BeOfValues(1, 2, 3, 4, 1, 2, 3, 4);
            yy.Should().BeShaped(2, 4);
        }

        // ------------------------------------------------------------------ dtype preserved

        [TestMethod]
        public void Meshgrid_PreservesEachInputDtype()
        {
            var (xx, yy) = np.meshgrid(np.array(new double[] { 0.0, 1.0, 2.0 }),
                np.array(new float[] { 0.5f, 1.5f }));
            xx.Should().BeOfType(NPTypeCode.Double).And.BeShaped(2, 3)
                .And.BeOfValues(0.0, 1.0, 2.0, 0.0, 1.0, 2.0);
            yy.Should().BeOfType(NPTypeCode.Single).And.BeShaped(2, 3)
                .And.BeOfValues(0.5f, 0.5f, 0.5f, 1.5f, 1.5f, 1.5f);
        }

        // ------------------------------------------------------------------ copy / views

        [TestMethod]
        public void Meshgrid_CopyTrue_IsContiguous_CopyFalse_IsView()
        {
            var (xx, _) = np.meshgrid(L(1, 2, 3), L(10, 20));
            xx.Shape.IsContiguous.Should().BeTrue();

            var v = np.meshgrid(L(1, 2, 3), L(10, 20), copy: false);
            // Dense broadcast views are non-contiguous (NumPy: C_CONTIGUOUS False), values still correct.
            v[0].Shape.IsContiguous.Should().BeFalse();
            v[0].Should().BeShaped(2, 3).And.BeOfValues(1, 2, 3, 1, 2, 3);
        }

        // ------------------------------------------------------------------ result surface + errors

        [TestMethod]
        public void Meshgrid_ImplicitToArray_AndReconstructsGrid()
        {
            NDArray[] g = np.meshgrid(L(0, 1, 2), L(0, 1, 2, 3));
            g.Length.Should().Be(2);
            // xx + yy fills the additive coordinate grid.
            (g[0] + g[1]).Should().BeShaped(4, 3)
                .And.BeOfValues(0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5);
        }

        [TestMethod]
        public void Meshgrid_BadIndexing_Throws()
        {
            new Action(() => { NDArray[] _ = np.meshgrid(L(1, 2), L(3, 4), indexing: "q"); })
                .Should().Throw<ValueError>().WithMessage("Valid values for `indexing`*");
        }
    }
}
