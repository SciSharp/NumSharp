
using Microsoft.VisualStudio.TestTools.UnitTesting;
using FluentAssertions;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Operations
{
    [TestClass]
    public class NDArrayNotEqualTest : TestClass
    {
        private void PerformNotEqualTests(NDArray nd)
        {
            (nd != 3).Should().BeOfValues(true, true, false, true, true, true);
            (nd != 5).Should().BeOfValues(true, true, true, true, false, true);
            (nd != 7).Should().BeOfValues(true, true, true, true, true, true);
        }

        [TestMethod]
        public void DoublesNotEqualTest()
        {
            NDArray nd = new double[,] { { 1, 2, 3 }, { 4, 5, 6 } };
            PerformNotEqualTests(nd);
        }

        [TestMethod]
        public void FloatsNotEqualTest()
        {
            NDArray nd = new float[,] { { 1, 2, 3 }, { 4, 5, 6 } };
            PerformNotEqualTests(nd);
        }

        [TestMethod]
        public void IntsNotEqualTest()
        {
            NDArray nd = new int[,] { { 1, 2, 3 }, { 4, 5, 6 } };
            PerformNotEqualTests(nd);
        }

        [TestMethod]
        public void LongsGreaterThanTest()
        {
            NDArray nd = new long[,] { { 1, 2, 3 }, { 4, 5, 6 } };
            PerformNotEqualTests(nd);
        }
    }
}
