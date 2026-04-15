using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Manipulation
{
    [TestClass]
    public class np_swapaxes_Test
    {
        [TestMethod]
        public void Case1()
        {
            var nd = np.array(1, 2, 3, 4).reshape(2, 2, 1);
            np.swapaxes(nd, 0, 2).Should()
                .BeOfValues(1, 3, 2, 4).And.BeShaped(1, 2, 2);
        }

        [TestMethod]
        public void Case2()
        {
            var nd = np.arange(3 * 2 * 4).reshape(3, 2, 4);
            np.swapaxes(nd, 0, 1).Should()
                .BeOfValues(0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23)
                .And.BeShaped(2, 3, 4);
        }

        [TestMethod]
        public void Case3()
        {
            var nd = np.arange(3 * 2 * 4).reshape(3, 2, 4);
            np.swapaxes(nd, 0, 2).Should()
                .BeOfValues(0, 8, 16, 4, 12, 20, 1, 9, 17, 5, 13, 21, 2, 10, 18, 6, 14, 22, 3, 11, 19, 7, 15, 23)
                .And.BeShaped(4, 2, 3);
        }

        [TestMethod]
        public void Case4()
        {
            var nd = np.array(1, 2, 3).reshape(1, 3);
            np.swapaxes(nd, 0, 1).Should()
                .BeOfValues(1, 2, 3)
                .And.BeShaped(3, 1);
        }

        [TestMethod]
        public void Case5()
        {
            var nd = np.arange(8).reshape(2, 2, 2);
            np.swapaxes(nd, 0, 2).Should()
                .BeOfValues(0, 4, 2, 6, 1, 5, 3, 7)
                .And.BeShaped(2, 2, 2);
        }
    }
}
