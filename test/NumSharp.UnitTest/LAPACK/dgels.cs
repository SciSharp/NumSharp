using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;
using System.Linq;
using System.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.Extensions
{
    /// <summary>
    /// Test concolve with standard example from 
    /// https://www.numpy.org/devdocs/reference/generated/numpy.convolve.html
    /// </summary>
    [Ignore]
    [TestClass]
    public class NdArrayDGELSTest
    {
        [TestMethod]
        public void Standard()
        {
            double[] a = {1.44, -9.96, -7.55, 8.34, 7.08, -5.45, -7.84, -0.28, 3.24, 8.09, 2.52, -5.70, -4.39, -3.24, 6.27, 5.28, 0.74, -1.19, 4.53, 3.83, -6.64, 2.06, -2.47, 4.70};

            double[] b = {8.58, 8.26, 8.48, -5.28, 5.72, 8.93, 9.35, -4.43, -0.70, -0.26, -7.36, -2.52};
            int m, n, lda, ldb, nrhs;

            m = 6;
            n = 4;
            nrhs = 2;
            lda = 6;
            ldb = 6;

            double dcont = 0.0001;
            int rank = 0;

            double[] work = new double[1];
            int lwork = -1;
            int info = 0;
            double[] singVal = new double[6];

            LAPACK.dgelss_(ref m, ref n, ref nrhs, a, ref lda, b, ref ldb, singVal, ref dcont, ref rank, work, ref lwork, ref info);

            lwork = (int)work[0];
            work = new double[lwork];

            LAPACK.dgelss_(ref m, ref n, ref nrhs, a, ref lda, b, ref ldb, singVal, ref dcont, ref rank, work, ref lwork, ref info);
        }
    }
}
