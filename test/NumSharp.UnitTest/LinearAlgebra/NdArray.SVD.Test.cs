using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;
using System.Linq;
using System.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.UnitTest.LinearAlgebra
{
    [TestClass]
    public class NDArraySVDTester
    {
        //[TestMethod]
        public void DefaultTest()
        {
            NDArray A = new NDArray(np.float64, new Shape(6, 5));
            A.ReplaceData(new double[] {8.79, 6.11, -9.15, 9.57, -3.49, 9.84, 9.93, 6.91, -7.93, 1.64, 4.02, 0.15, 9.83, 5.04, 4.86, 8.83, 9.80, -8.99, 5.45, -0.27, 4.85, 0.74, 10.00, -6.02, 3.16, 7.98, 3.01, 5.80, 4.27, -5.31});

            var allMatrix = A.svd();

            var u = allMatrix.Item1;
            var s = allMatrix.Item2;
            var vt = allMatrix.Item3;

            var sMAtrix = np.eye(5);

            for (int idx = 0; idx < 5; idx++)
                sMAtrix[idx, idx] = s[idx];

            var ACreated = np.dot(np.dot(u, sMAtrix), vt);

            var error = A - ACreated;

            ArraySlice<double> errorElements = error.Data<double>();

            for (int idx = 0; idx < errorElements.Count; idx++)
                Assert.IsTrue(Math.Abs(errorElements[idx]) < 0.01);
        }
    }
}
