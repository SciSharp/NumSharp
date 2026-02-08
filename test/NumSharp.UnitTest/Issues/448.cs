using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Issues
{
    [TestClass]
    [TestCategory("OpenBugs")]
    public class Issue448
    {
        [TestMethod]
        public void ReproducingTest1()
        {
            NDArray nd = new double[,] {{1, 2, 3}, {4, 5, 6}};
            (nd < 3).Should().BeOfValues(true, true, false, false, false, false);

            nd[(NDArray)(nd < 3)] = -2;
            nd.Should().BeOfValues(-2, -2, 3, 4, 5, 6);
        }

        [TestMethod]
        public void ReproducingTest2()
        {
            NDArray nd = new double[,] {{1, 2, 3}, {4, 5, 6}};
            (nd < 3).Should().BeOfValues(true, true, false, false, false, false);

            nd[(NDArray)(nd < 3)] = -2;
            var values = new int[] {-2, -2, 3, 4, 5, 6};

            var iter = nd.AsIterator<double>();
            var next = iter.MoveNext;
            var hasnext = iter.HasNext;
            for (int i = 0; i < values.Length; i++)
            {
                next();
            }
        }
    }
}
