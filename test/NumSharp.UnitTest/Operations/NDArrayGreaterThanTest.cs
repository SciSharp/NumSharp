
using Microsoft.VisualStudio.TestTools.UnitTesting;
using AwesomeAssertions;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Operations
{
    [TestClass]
    public class NDArrayGreaterThanTest : TestClass
    {
        private void PerformGreaterThanTests (NDArray nd)
        {
            (nd > 3).Should().BeOfValues(false, false, false, true, true, true);
            (nd > 5).Should().BeOfValues(false, false, false, false, false, true);
            (nd > 7).Should().BeOfValues(false, false, false, false, false, false);
        }

        [TestMethod]
        public void DoublesGreaterThanTest()
        {
            NDArray nd = new double[,] { { 1, 2, 3 }, { 4, 5, 6 } };
            PerformGreaterThanTests(nd);
        }

        [TestMethod]
        public void FloatsGreaterThanTest()
        {
            NDArray nd = new float[,] { { 1, 2, 3 }, { 4, 5, 6 } };
            PerformGreaterThanTests(nd);
        }

        [TestMethod]
        public void IntsGreaterThanTest()
        {
            NDArray nd = new int[,] { { 1, 2, 3 }, { 4, 5, 6 } };
            PerformGreaterThanTests(nd);
        }

        [TestMethod]
        public void LongsGreaterThanTest()
        {
            NDArray nd = new long[,] { { 1, 2, 3 }, { 4, 5, 6 } };
            PerformGreaterThanTests(nd);
        }
    }
}
