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
    [TestClass]
    public class NdArraySVDTest
    {
        [Ignore]
        [TestMethod]
        public void Standard()
        {
            double[] A = {8.79, 6.11, -9.15, 9.57, -3.49, 9.84, 9.93, 6.91, -7.93, 1.64, 4.02, 0.15, 9.83, 5.04, 4.86, 8.83, 9.80, -8.99, 5.45, -0.27, 4.85, 0.74, 10.00, -6.02, 3.16, 7.98, 3.01, 5.80, 4.27, -5.31};

            int m = 6, n = 5, lda = 6, ldu = 6, ldvt = 5, info = 0, lwork = -1;

            double[] work = new double[5];

            /* Local arrays */
            double[] s = new double[5], u = new double[6 * 6], vt = new double[5 * 5];

            LAPACK.dgesvd_("ALL".ToCharArray(), "All".ToCharArray(), ref m, ref n, A, ref lda, s, u, ref ldu, vt, ref ldvt, work, ref lwork, ref info);

            lwork = (int)work[0];

            work = new double[lwork];

            LAPACK.dgesvd_("ALL".ToCharArray(), "All".ToCharArray(), ref m, ref n, A, ref lda, s, u, ref ldu, vt, ref ldvt, work, ref lwork, ref info);
        }
    }
}
