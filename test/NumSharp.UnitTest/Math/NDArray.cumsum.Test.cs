using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using FluentAssertions;

namespace NumSharp.UnitTest.Maths
{
    [TestClass]
    public class NDArrayCumsumTest : TestClass
    {
        [TestMethod]
        public void CumsumStaticFunctionTest()
        {
            NDArray arr = new double[] { 0, 1, 4, 2, 5, 6, 2 };
            NDArray expected = new double[] { 0, 1, 5, 7, 12, 18, 20 };

            NDArray actual = np.cumsum(arr);

            actual.Array.Should().BeEquivalentTo(expected.Array);
        }

        [TestMethod]
        public void CumsumMemberFunctionTest()
        {
            NDArray arr = new double[] { 0, 1, 4, 2, 5, 6, 2 };
            NDArray expected = new double[] { 0, 1, 5, 7, 12, 18, 20 };

            NDArray actual = arr.cumsum();

            actual.Array.Should().BeEquivalentTo(expected.Array);
        }

        [TestMethod]
        [Ignore("Cumulative summing on multidimensional arrays not implemented yet")]
        public void Cumsum2dTest()
        {
            NDArray arr = new int[,] { { 1, 2, 3 }, { 4, 5, 6 } };
            NDArray expected = new int[] { 1, 3, 6, 10, 15, 21 };

            NDArray actual = np.cumsum(arr);

            actual.Array.Should().BeEquivalentTo(expected.Array);
        }

        [TestMethod]
        [Ignore("Cumulative summing with specified data type not implemented yet")]
        public void Cumsum2dDtypeTest()
        {
            NDArray arr = new int[,] { { 1, 2, 3 }, { 4, 5, 6 } };
            NDArray expected = new float[] { 1, 3, 6, 10, 15, 21 };

            NDArray actual = np.cumsum(arr, dtype: typeof(float));

            actual.Array.Should().BeEquivalentTo(expected.Array);
        }

        [TestMethod]
        [Ignore("Cumulative summing along axis not implemented yet")]
        public void Cumsum2dAxisRowsTest()
        {
            NDArray arr = new int[,] { { 1, 2, 3 }, { 4, 5, 6 } };
            NDArray expected = new int[,] { { 1, 2, 3 }, { 5, 7, 9 } };

            NDArray actual = np.cumsum(arr, axis: 0);

            for (int i = 0; i < actual.shape[0]; i++)
            {
                for (int j = 0; j < actual.shape[1]; j++)
                {
                    Assert.AreEqual((int)expected[i, j], (int)actual[i, j]);
                }
            }
        }

        [TestMethod]
        [Ignore("Cumulative summing along axis not implemented yet")]
        public void Cumsum2dAxisColsTest()
        {
            NDArray arr = new int[,] { { 1, 2, 3 }, { 4, 5, 6 } };
            NDArray expected = new int[,] { { 1, 3, 6 }, { 4, 9, 15 } };

            NDArray actual = np.cumsum(arr, axis: 1);

            for (int i = 0; i < actual.shape[0]; i++)
            {
                for (int j = 0; j < actual.shape[1]; j++)
                {
                    Assert.AreEqual((int)expected[i, j], (int)actual[i, j]);
                }
            }
        }
    }
}
