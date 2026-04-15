using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;
using System.Linq;
using NumSharp;

namespace NumSharp.UnitTest.LinearAlgebra
{
    [TestClass]
    public class NdArrayInvTest
    {
        //[TestMethod]
        public void Simple2x2()
        {
            var np1 = np.arange(4.0).reshape(2, 2);

            var np1Inv = np1.inv();

            var OncesMatrix = np.dot(np1, np1Inv);

            Assert.IsTrue(Enumerable.SequenceEqual(new double[] {1, 0, 0, 1}, OncesMatrix.Data<double>()));
        }

        //[TestMethod]
        public void Simple3x3()
        {
            var np1 = new NDArray(typeof(double), new Shape(3, 3));

            var np1Arr = np1.Data<double>();

            np1Arr[0] = 5;
            np1Arr[1] = 1;
            np1Arr[2] = 2;
            np1Arr[3] = 1;
            np1Arr[4] = 0;
            np1Arr[5] = 1;
            np1Arr[6] = 1;
            np1Arr[7] = 1;
            np1Arr[8] = 0;

            var np1Inv = np1.inv();

            var OncesMatrix = np.dot(np1, np1Inv);

            //Assert.IsTrue(Math.Abs(OncesMatrix[0,0]) < 1.000001);
            //Assert.IsTrue(Math.Abs(OncesMatrix[1,1]) < 1.000001);
            //Assert.IsTrue(Math.Abs(OncesMatrix[2,2]) < 1.000001);

            //Assert.IsTrue(Math.Abs(OncesMatrix[0,1]) < 0.000001);
            //Assert.IsTrue(Math.Abs(OncesMatrix[0,2]) < 0.000001);
            //Assert.IsTrue(Math.Abs(OncesMatrix[1,0]) < 0.000001);
            //Assert.IsTrue(Math.Abs(OncesMatrix[1,2]) < 0.000001);
            //Assert.IsTrue(Math.Abs(OncesMatrix[2,0]) < 0.000001);
            //Assert.IsTrue(Math.Abs(OncesMatrix[2,1]) < 0.000001);
        }
    }
}
