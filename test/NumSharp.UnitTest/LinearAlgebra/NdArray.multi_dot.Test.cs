using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Numerics;
using System.Collections.Generic;
using System.Text;
using NumSharp.Core.Extensions;
using System.Linq;
using NumSharp.Core;

namespace NumSharp.UnitTest.LinearAlgebra
{
    [TestClass]
    public class MultiMatrixDotTest
    {
        [TestMethod]
        public void ThreeMultiplication()
        {
            NDArrayGeneric<double> np1 = new NumPyGeneric<double>().arange(4).reshape(2,2);
            NDArrayGeneric<double> np2 = np1.inv();
            NDArrayGeneric<double> np3 = new NumPyGeneric<double>().ones(new Shape(2,2));

            var OncesMatrix = np1.multi_dot(np2,np3);

            Assert.IsTrue(OncesMatrix[0,0] == 1);
            Assert.IsTrue(OncesMatrix[1,0] == 1);
            Assert.IsTrue(OncesMatrix[0,1] == 1);
            Assert.IsTrue(OncesMatrix[1,1] == 1);

        }
    }
}
