using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Numerics;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;
using System.Linq;

namespace NumSharp.UnitTest.LinearAlgebra
{
    [TestClass]
    public class MultiMatrixDotTest
    {
        [TestMethod]
        public void ThreeMultiplication()
        {
            NDArray<double> np1 = new NumPy<double>().arange(4).reshape(2,2);
            NDArray<double> np2 = np1.inv();
            NDArray<double> np3 = new NumPy<double>().ones(new Shape(2,2));

            var OncesMatrix = np1.multi_dot(np2,np3);

            Assert.IsTrue(OncesMatrix[0,0] == 1);
            Assert.IsTrue(OncesMatrix[1,0] == 1);
            Assert.IsTrue(OncesMatrix[0,1] == 1);
            Assert.IsTrue(OncesMatrix[1,1] == 1);

        }
    }
}
