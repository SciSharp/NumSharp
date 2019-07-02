using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.Maths
{
    [TestClass]
    public class NDArrayCumsumTest : TestClass
    {
        [TestMethod]
        public void CumsumStaticTest()
        {
            NDArray arr = new double[] { 0, 1, 4, 2, 5, 6, 2 };
            NDArray expected = new double[] { 0, 1, 5, 7, 12, 18, 20 };

            NDArray actual = np.cumsum(arr);

            for (int i = 0; i < expected.shape[0]; i++)
            {
                Assert.AreEqual((double)expected[i], (double)actual[i]);
            }
        }

        [TestMethod]
        public void CumsumMemberTest()
        {
            NDArray arr = new double[] { 0, 1, 4, 2, 5, 6, 2 };
            NDArray expected = new double[] { 0, 1, 5, 7, 12, 18, 20 };

            NDArray actual = arr.cumsum();

            for (int i = 0; i < expected.shape[0]; i++)
            {
                Assert.AreEqual((double)expected[i], (double)actual[i]);
            }
        }
    }
}
