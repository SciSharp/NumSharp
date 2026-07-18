using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Manipulation
{
    /// <summary>
    ///     np.permute_dims is the Array API alias of np.transpose (NumPy 2.x: np.permute_dims is np.transpose).
    /// </summary>
    [TestClass]
    public class np_permute_dims_Test
    {
        [TestMethod]
        public void Reverse_2x2()
        {
            var nd = np.array(1, 2, 3, 4).reshape(2, 2);
            np.permute_dims(nd).Should()
                .BeOfValues(1, 3, 2, 4).And.BeShaped(2, 2);
        }

        [TestMethod]
        public void Reverse_3x2x4()
        {
            var nd = np.arange(3 * 2 * 4).reshape(3, 2, 4);
            np.permute_dims(nd).Should()
                .BeOfValues(0, 8, 16, 4, 12, 20, 1, 9, 17, 5, 13, 21, 2, 10, 18, 6, 14, 22, 3, 11, 19, 7, 15, 23)
                .And.BeShaped(4, 2, 3);
        }

        [TestMethod]
        public void ExplicitAxes()
        {
            var nd = np.arange(2 * 3).reshape(1, 2, 3);
            np.permute_dims(nd, new int[] {1, 0, 2}).Should()
                .BeOfValues(0, 1, 2, 3, 4, 5)
                .And.BeShaped(2, 1, 3);
        }

        [TestMethod]
        public void NegativeAxes()
        {
            // np.permute_dims(a, (-1, 0, -2)).shape == (5, 3, 4)
            var nd = np.arange(3 * 4 * 5).reshape(3, 4, 5);
            np.permute_dims(nd, new int[] {-1, 0, -2}).Should().BeShaped(5, 3, 4);
        }

        [TestMethod]
        public void OneDimensional_Unchanged()
        {
            var nd = np.array(1, 2, 3, 4);
            np.permute_dims(nd).Should()
                .BeOfValues(1, 2, 3, 4).And.BeShaped(4);
        }

        [TestMethod]
        public void IsAliasOfTranspose()
        {
            var nd = np.arange(3 * 2 * 4).reshape(3, 2, 4);
            var viaPermute = np.permute_dims(nd, new int[] {2, 0, 1});
            var viaTranspose = np.transpose(nd, new int[] {2, 0, 1});
            viaPermute.Should().BeShaped(4, 3, 2);
            np.array_equal(viaPermute, viaTranspose).Should().BeTrue();
        }

        [TestMethod]
        public void RepeatedAxis_Throws()
        {
            // NumPy raises ValueError; NumSharp's analog is ArgumentException.
            var nd = np.arange(6).reshape(2, 3);
            new Action(() => np.permute_dims(nd, new int[] {0, 0})).Should()
                .Throw<ArgumentException>().WithMessage("*repeated axis in transpose*");
        }

        [TestMethod]
        public void NegativeAxisDuplicate_Throws()
        {
            // (0, -2) normalizes to (0, 0) -> repeated axis, matching NumPy.
            var nd = np.arange(6).reshape(2, 3);
            new Action(() => np.permute_dims(nd, new int[] {0, -2})).Should()
                .Throw<ArgumentException>().WithMessage("*repeated axis in transpose*");
        }

        [TestMethod]
        public void WrongLength_Throws()
        {
            var nd = np.arange(6).reshape(2, 3);
            new Action(() => np.permute_dims(nd, new int[] {0})).Should()
                .Throw<ArgumentException>().WithMessage("*axes don't match*");
            new Action(() => np.permute_dims(nd, new int[] {0, 1, 2})).Should()
                .Throw<ArgumentException>().WithMessage("*axes don't match*");
        }

        [TestMethod]
        public void OutOfRangeAxis_Throws()
        {
            // NumPy raises AxisError; NumSharp's analog is AxisOutOfRangeException.
            var nd = np.arange(6).reshape(2, 3);
            new Action(() => np.permute_dims(nd, new int[] {0, 2})).Should()
                .Throw<AxisOutOfRangeException>().WithMessage("*out of bounds*");
            new Action(() => np.permute_dims(nd, new int[] {0, -3})).Should()
                .Throw<AxisOutOfRangeException>().WithMessage("*out of bounds*");
        }

        [TestMethod]
        public void EmptyAxes_ScalarOk_HigherRankThrows()
        {
            // NumPy: an explicit length-0 permutation matches only a 0-d array;
            // for ndim >= 1 it raises ValueError "axes don't match array".
            // (Distinct from axes=null, which reverses.)
            np.permute_dims(NDArray.Scalar(5), new int[0]).Should().BeShaped(size: 1, ndim: 0);
            new Action(() => np.permute_dims(np.array(1, 2, 3), new int[0])).Should()
                .Throw<ArgumentException>().WithMessage("*axes don't match*");
            new Action(() => np.permute_dims(np.arange(6).reshape(2, 3), new int[0])).Should()
                .Throw<ArgumentException>().WithMessage("*axes don't match*");
        }
    }
}
