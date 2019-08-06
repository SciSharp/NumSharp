using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using FluentAssertions;
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
            NDArray expected = new int[] {1, 3, 6, 10, 15, 21};

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
            NDArray expected = new int[,] {{1, 2, 3}, {5, 7, 9}};

            np.cumsum(arr, axis: 0).Should().Be(expected);
        }        
        
        [TestMethod]
        public void Cumsum2dAxisCols()
        {
            NDArray arr = new int[,] {{1, 2, 3}, {4, 5, 6}};
            NDArray expected = new int[,] {{1, 3, 6}, {4, 9, 15}};

            np.cumsum(arr, axis: 1).Should().Be(expected);
        }
    }
}
