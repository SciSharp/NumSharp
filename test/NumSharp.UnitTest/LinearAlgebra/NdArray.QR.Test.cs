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
    /// <summary>
    /// Test concolve with standard example from 
    /// https://www.numpy.org/devdocs/reference/generated/numpy.convolve.html
    /// </summary>
    [TestClass]
    public class NdArrayQRTest
    {
        //[TestMethod]
        public void FullMatrix()
        {
            var nd1 = np.array(new double[] {1, 1, 0, 1, 0, 1, 0, 1, 1}).reshape(3, 3);

            var (Q, R) = nd1.qr();

            var nd2 = np.dot(Q.transpose(), R);

            // make sure the highest difference is lower than 0.0000000001
            Assert.IsTrue(((double)np.abs(nd1 - nd2).max<double>() < 0.00000000001));
        }
    }
}
