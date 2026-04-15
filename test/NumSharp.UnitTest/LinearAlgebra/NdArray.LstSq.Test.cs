using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;
using System.Linq;
using System.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.LinearAlgebra
{
    [TestClass]
    public class NDArrayLinSqTester
    {
        //[TestMethod]
        public void DefaultTest()
        {
            NDArray A = new double[,] {{0.0, 1.0}, {1.0, 1.0}, {2.0, 1.0}, {3.0, 1.0}};

            NDArray B = new double[] {-1, 0.2, 0.9, 2.1};

            double[,] C = (Array)A.lstqr(B) as double[,];

            Assert.IsTrue((C[0, 0] - 1.0) < 0.0001);
            Assert.IsTrue((C[1, 0] + 0.95) < 0.0001);
        }
    }
}
