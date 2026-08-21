using System;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.LinearAlgebra
{
    /// <summary>
    ///     <c>out=</c> on <c>np.dot</c> and <c>np.outer</c>, probed against NumPy 2.4.2. <c>dot</c>'s
    ///     is STRICT (exact dtype, ndim, C-contiguous — its <c>new_array_for_sum</c>); <c>outer</c>'s
    ///     is the ordinary ufunc <c>out=</c> of the underlying <c>multiply</c>.
    /// </summary>
    [TestClass]
    public class np_dot_outer_out_test
    {
        private static NDArray M23 => np.arange(6.0).reshape(2, 3);
        private static NDArray M32 => np.arange(6.0).reshape(3, 2);

        [TestMethod]
        public void Dot_Out_ReturnsSameInstanceWithResult()
        {
            var o = np.zeros(new Shape(2, 2), NPTypeCode.Double);
            var r = np.dot(M23, M32, @out: o);
            r.Should().BeSameAs(o);
            r.Should().BeOfValues(10, 13, 28, 40).And.BeShaped(2, 2);
        }

        [TestMethod]
        public void Dot_Out_ScalarResult()
        {
            var o = np.zeros(new Shape(), NPTypeCode.Double);
            np.dot(np.arange(3.0), np.arange(3.0), @out: o).Should().BeOfValues(5).And.BeShaped();
        }

        [TestMethod]
        public void Dot_Out_WrongShape_SameNdim_Raises()
        {
            new Action(() => np.dot(M23, M32, @out: np.zeros(new Shape(2, 3), NPTypeCode.Double)))
                .Should().Throw<ValueError>().WithMessage("output array has wrong dimensions");
        }

        [TestMethod]
        public void Dot_Out_WrongDtype_Raises()
        {
            new Action(() => np.dot(M23, M32, @out: np.zeros(new Shape(2, 2), NPTypeCode.Single)))
                .Should().Throw<ValueError>().WithMessage(
                    "output array is not acceptable (must have the right datatype, number of dimensions, and be a C-Array)");
        }

        [TestMethod]
        public void Dot_Out_NonContiguous_Raises()
        {
            // dot's out must be a C-array; a stride-2 column view is not.
            var strided = np.zeros(new Shape(2, 4), NPTypeCode.Double)[":, ::2"];
            new Action(() => np.dot(M23, M32, @out: strided))
                .Should().Throw<ValueError>().WithMessage(
                    "output array is not acceptable (must have the right datatype, number of dimensions, and be a C-Array)");
        }

        [TestMethod]
        public void Outer_Out_FillsWithProduct()
        {
            var o = np.zeros(new Shape(3, 3), NPTypeCode.Int32);
            var r = np.outer(np.array(new int[] {1, 2, 3}), np.array(new int[] {4, 5, 6}), @out: o);
            r.Should().BeSameAs(o);
            r.Should().BeOfValues(4, 5, 6, 8, 10, 12, 12, 15, 18).And.BeShaped(3, 3);
        }

        [TestMethod]
        public void Outer_Out_TakesSameKindCast()
        {
            // int product into a float32 out: same_kind, so the ufunc out= accepts it.
            var o = np.zeros(new Shape(3, 3), NPTypeCode.Single);
            var r = np.outer(np.array(new int[] {1, 2, 3}), np.array(new int[] {4, 5, 6}), @out: o);
            r.typecode.Should().Be(NPTypeCode.Single);
            r.Should().BeOfValues(4, 5, 6, 8, 10, 12, 12, 15, 18);
        }

        [TestMethod]
        public void Outer_Out_WrongShape_Raises()
        {
            new Action(() => np.outer(np.array(new int[] {1, 2, 3}), np.array(new int[] {4, 5, 6}),
                    @out: np.zeros(new Shape(3, 4), NPTypeCode.Int32)))
                .Should().Throw<Exception>();
        }
    }
}
