
using Microsoft.VisualStudio.TestTools.UnitTesting;
using AwesomeAssertions;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Operations
{
    [TestClass]
    public class NDArrayLessThanTest : TestClass
    {
        private void PerformLessThanTests(NDArray nd)
        {
            (nd < 3).Should().BeOfValues(true, true, false, false, false, false);
            (nd < 6).Should().BeOfValues(true, true, true, true, true, false);
            (nd < 7).Should().BeOfValues(true, true, true, true, true, true);
        }

        [TestMethod]
        public void DoublesLessThanTest()
        {
            NDArray nd = new double[,] { { 1, 2, 3 }, { 4, 5, 6 } };
            PerformLessThanTests(nd);
        }

        [TestMethod]
        public void FloatsLessThanTest()
        {
            NDArray nd = new float[,] { { 1, 2, 3 }, { 4, 5, 6 } };
            PerformLessThanTests(nd);
        }

        [TestMethod]
        public void IntsLessThanTest()
        {
            NDArray nd = new int[,] { { 1, 2, 3 }, { 4, 5, 6 } };
            PerformLessThanTests(nd);
        }

        [TestMethod]
        public void LongsLessThanTest()
        {
            NDArray nd = new long[,] { { 1, 2, 3 }, { 4, 5, 6 } };
            PerformLessThanTests(nd);
        }
    }
}
