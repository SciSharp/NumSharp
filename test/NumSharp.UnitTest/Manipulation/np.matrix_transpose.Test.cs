using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Manipulation
{
    /// <summary>
    ///     np.matrix_transpose(x) swaps the two innermost axes: (..., M, N) -> (..., N, M).
    ///     Equivalent to np.swapaxes(x, -1, -2). Requires ndim >= 2. Also covers ndarray.mT.
    /// </summary>
    [TestClass]
    public class np_matrix_transpose_Test
    {
        [TestMethod]
        public void TwoDimensional()
        {
            var nd = np.arange(6).reshape(2, 3);
            np.matrix_transpose(nd).Should()
                .BeOfValues(0, 3, 1, 4, 2, 5).And.BeShaped(3, 2);
        }

        [TestMethod]
        public void StackOfMatrices_3D()
        {
            // (2,3,4) -> (2,4,3), each of the 2 matrices transposed independently.
            var nd = np.arange(24).reshape(2, 3, 4);
            np.matrix_transpose(nd).Should()
                .BeOfValues(0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11,
                            12, 16, 20, 13, 17, 21, 14, 18, 22, 15, 19, 23)
                .And.BeShaped(2, 4, 3);
        }

        [TestMethod]
        public void HigherRank_ShapeOnly()
        {
            np.matrix_transpose(np.ones(new Shape(2, 3, 4, 5))).Should().BeShaped(2, 3, 5, 4);
        }

        [TestMethod]
        public void EquivalentToSwapaxes()
        {
            var nd = np.arange(24).reshape(2, 3, 4);
            np.array_equal(np.matrix_transpose(nd), np.swapaxes(nd, -1, -2)).Should().BeTrue();
        }

        [TestMethod]
        public void ReturnsView_WriteThrough()
        {
            var nd = np.arange(6).reshape(2, 3);
            var mt = np.matrix_transpose(nd);
            mt[0, 0] = 99;
            // mt[0,0] maps to nd[0,0]; write propagates to the shared buffer.
            ((int)nd[0, 0]).Should().Be(99);
        }

        [TestMethod]
        public void EmptyArray()
        {
            np.matrix_transpose(np.zeros(new Shape(0, 3))).Should().BeShaped(3, 0);
            np.matrix_transpose(np.zeros(new Shape(2, 0, 3))).Should().BeShaped(2, 3, 0);
        }

        [TestMethod]
        public void BroadcastView_StaysReadOnly()
        {
            var b = np.broadcast_to(np.arange(3), new Shape(2, 3));
            var mt = np.matrix_transpose(b);
            mt.Should().BeShaped(3, 2);
            mt.Shape.IsWriteable.Should().BeFalse();
        }

        [TestMethod]
        public void OneDimensional_Throws()
        {
            new Action(() => np.matrix_transpose(np.array(1, 2, 3))).Should()
                .Throw<ArgumentException>().WithMessage("*at least 2-dimensional*it is 1*");
        }

        [TestMethod]
        public void ZeroDimensional_Throws()
        {
            new Action(() => np.matrix_transpose(NDArray.Scalar(5))).Should()
                .Throw<ArgumentException>().WithMessage("*at least 2-dimensional*it is 0*");
        }

        [TestMethod]
        public void MatrixTransposeProperty_mT()
        {
            var nd = np.arange(24).reshape(2, 3, 4);
            np.array_equal(nd.mT, np.matrix_transpose(nd)).Should().BeTrue();
            nd.mT.Should().BeShaped(2, 4, 3);
        }

        [TestMethod]
        public void mT_OneDimensional_Throws()
        {
            new Action(() => { var _ = np.array(1, 2, 3).mT; }).Should()
                .Throw<ArgumentException>().WithMessage("*matrix transpose with ndim < 2 is undefined*");
        }
    }
}
