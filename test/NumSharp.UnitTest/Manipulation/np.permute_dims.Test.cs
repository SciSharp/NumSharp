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
            var nd = np.arange(6).reshape(2, 3);
            new Action(() => np.permute_dims(nd, new int[] {0, 0})).Should()
                .Throw<Exception>().WithMessage("*repeated axis*");
        }

        [TestMethod]
        public void WrongLength_Throws()
        {
            var nd = np.arange(6).reshape(2, 3);
            new Action(() => np.permute_dims(nd, new int[] {0})).Should()
                .Throw<Exception>().WithMessage("*axes don't match*");
        }
    }
}
