
using Microsoft.VisualStudio.TestTools.UnitTesting;
using AwesomeAssertions;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Operations
{
    public class NDArrayNotEqualTest : TestClass
    {
        private void PerformNotEqualTests(NDArray nd)
        {
            (nd != 3).Should().BeOfValues(true, true, false, true, true, true);
            (nd != 5).Should().BeOfValues(true, true, true, true, false, true);
            (nd != 7).Should().BeOfValues(true, true, true, true, true, true);
        }

        [Test]
        public void DoublesNotEqualTest()
        {
            NDArray nd = new double[,] { { 1, 2, 3 }, { 4, 5, 6 } };
            PerformNotEqualTests(nd);
        }

        [Test]
        public void FloatsNotEqualTest()
        {
            NDArray nd = new float[,] { { 1, 2, 3 }, { 4, 5, 6 } };
            PerformNotEqualTests(nd);
        }

        [Test]
        public void IntsNotEqualTest()
        {
            NDArray nd = new int[,] { { 1, 2, 3 }, { 4, 5, 6 } };
            PerformNotEqualTests(nd);
        }

        [Test]
        public void LongsGreaterThanTest()
        {
            NDArray nd = new long[,] { { 1, 2, 3 }, { 4, 5, 6 } };
            PerformNotEqualTests(nd);
        }
    }
}
