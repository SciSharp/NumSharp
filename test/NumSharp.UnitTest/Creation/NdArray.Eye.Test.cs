using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;
using System.Linq;
using NumSharp;

namespace NumSharp.UnitTest.Creation
{
    [TestClass]
    public class NdArrayEyeTest
    {
        [TestMethod]
        public void Case1()
        {
            var g = np.eye(3, k: 1);
            var ret = new NDArray(new float[][] {new float[] {0.0f, 1.0f, 0.0f}, new float[] {0.0f, 0f, 1.0f}, new float[] {0f, 0f, 0f}}, Shape.Matrix(3, 3));
        }


        //[TestMethod]
        //public void SimpleIntMatrix()
        //{
        //    var np = new NDArray(typeof(int)).eye(5).MakeGeneric<int>();

        //    Assert.IsTrue(np[0, 0] == 1);
        //    Assert.IsTrue(np[1, 1] == 1);
        //    Assert.IsTrue(np[2, 2] == 1);
        //    Assert.IsTrue(np[3, 3] == 1);
        //    Assert.IsTrue(np[4, 4] == 1);

        //    int[] elementsZero = np.Data<int>().Where((x) => x == 0).ToArray();

        //    Assert.IsTrue(elementsZero.Length == 20);
        //}

        //[TestMethod]
        //public void SimpleDoubleMatrix()
        //{
        //    var np = new NDArray(typeof(double)).eye(5).MakeGeneric<double>();

        //    Assert.IsTrue(np[0, 0] == 1);
        //    Assert.IsTrue(np[1, 1] == 1);
        //    Assert.IsTrue(np[2, 2] == 1);
        //    Assert.IsTrue(np[3, 3] == 1);
        //    Assert.IsTrue(np[4, 4] == 1);

        //    double[] elementsZero = np.Data<double>().Where((x) => x == 0.0).ToArray();

        //    Assert.IsTrue(elementsZero.Length == 20);
        //}

        //[TestMethod]
        //public void DoubleMatrix2DiagonalLeft()
        //{
        //    var np = new NDArray(typeof(double)).eye(5, -2).MakeGeneric<double>();

        //    Assert.IsTrue(np[2, 0] == 1);
        //    Assert.IsTrue(np[3, 1] == 1);
        //    Assert.IsTrue(np[4, 2] == 1);

        //    Assert.IsTrue(np.Data<double>().Where(x => x == 0).ToArray().Length == 22);
        //}

        //[TestMethod]
        //public void DoubleMatrix2DiagonalRight()
        //{
        //    var np = new NDArray(typeof(double)).eye(5, 2).MakeGeneric<double>();

        //    Assert.IsTrue(np[0, 2] == 1);
        //    Assert.IsTrue(np[1, 3] == 1);
        //    Assert.IsTrue(np[2, 4] == 1);

        //    Assert.IsTrue(np.Data<double>().Where(x => x == 0).ToArray().Length == 22);
        //}
    }
}
