using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using AwesomeAssertions;
using NumSharp.Backends;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Maths
{
    [TestClass]
    public class NDArrayCumsumTest : TestClass
    {
        [TestMethod]
        public void CumsumStaticFunction()
        {
            NDArray arr = new double[] {0, 1, 4, 2, 5, 6, 2};
            NDArray expected = new double[] {0, 1, 5, 7, 12, 18, 20};

            np.cumsum(arr).Should().Be(expected);
        }

        [TestMethod]
        public void CumsumMemberFunction()
        {
            NDArray arr = new double[] {0, 1, 4, 2, 5, 6, 2};
            NDArray expected = new double[] {0, 1, 5, 7, 12, 18, 20};

            arr.cumsum().Should().Be(expected);
        }

        [TestMethod]
        public void Cumsum2d()
        {
            NDArray arr = new int[,] {{1, 2, 3}, {4, 5, 6}};
            // NumPy-aligned: cumsum of int32 returns int64
            NDArray expected = new long[] {1, 3, 6, 10, 15, 21};

            np.cumsum(arr).Should().Be(expected);
        }

        [TestMethod]
        public void Cumsum2dDtype()
        {
            NDArray arr = new int[,] {{1, 2, 3}, {4, 5, 6}};
            NDArray expected = new float[] {1, 3, 6, 10, 15, 21};

            np.cumsum(arr, typeCode: NPTypeCode.Single).Should().Be(expected);
        }

        [TestMethod]
        public void Cumsum2dAxisRows()
        {
            NDArray arr = new int[,] {{1, 2, 3}, {4, 5, 6}};
            // NumPy-aligned: cumsum of int32 returns int64
            NDArray expected = new long[,] {{1, 2, 3}, {5, 7, 9}};

            np.cumsum(arr, axis: 0).Should().Be(expected);
        }

        [TestMethod]
        public void Cumsum2dAxisCols()
        {
            NDArray arr = new int[,] {{1, 2, 3}, {4, 5, 6}};
            // NumPy-aligned: cumsum of int32 returns int64
            NDArray expected = new long[,] {{1, 3, 6}, {4, 9, 15}};

            np.cumsum(arr, axis: 1).Should().Be(expected);
        }

        /// <summary>
        /// Bug 76 fix verification: np.cumsum on boolean array should work.
        /// NumPy: cumsum([True, False, True, True, False]) = [1, 1, 2, 3, 3]
        /// Return type is int64 (NumPy 2.x behavior).
        /// </summary>
        [TestMethod]
        public void BooleanArray_TreatsAsIntAndReturnsInt64()
        {
            var a = np.array(new bool[] { true, false, true, true, false });
            var result = np.cumsum(a);

            result.typecode.Should().Be(NPTypeCode.Int64, "NumPy 2.x: cumsum(bool) returns int64");
            result.GetInt64(0).Should().Be(1, "cumsum[0] = 1 (True)");
            result.GetInt64(1).Should().Be(1, "cumsum[1] = 1 (False adds 0)");
            result.GetInt64(2).Should().Be(2, "cumsum[2] = 2 (True adds 1)");
            result.GetInt64(3).Should().Be(3, "cumsum[3] = 3 (True adds 1)");
            result.GetInt64(4).Should().Be(3, "cumsum[4] = 3 (False adds 0)");
        }
    }
}
