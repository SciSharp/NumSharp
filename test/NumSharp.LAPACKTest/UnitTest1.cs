using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.LAPACKTest
{
    [TestClass]
    public class UnitTest1
    {
        [TestMethod]
        public void TestMethod1()
        {
            int n = 5;
            int nrhs = 3;
            int lda  = 5;
            int ldb = 5;
            int info = 0; 

            /* Local arrays */
            int[] ipiv = new int[n];
            double[] a = {
                6.80, -2.11,  5.66,  5.97,  8.23,
            -6.05, -3.30,  5.36, -4.44,  1.08,
            -0.45,  2.58, -2.70,  0.27,  9.04,
                8.32,  2.71,  4.35, -7.17,  2.14,
            -9.67, -5.14, -7.26,  6.08, -6.87
            };

            double[,] aa = new double[,] {{6.80, -2.11,  5.66,  5.97,  8.23},
                                          {-6.05, -3.30,  5.36, -4.44,  1.08},
                                          {-0.45,  2.58, -2.70,  0.27,  9.04},
                                          {8.32,  2.71,  4.35, -7.17,  2.14},
                                          {-9.67, -5.14, -7.26,  6.08, -6.87}
                                        };

            double[] b = {
                4.02,  6.19, -8.22, -7.57, -3.03,
            -1.56,  4.00, -8.67,  1.75,  2.86,
                9.81, -4.09, -4.57, -8.61,  8.99
            };
            
            double[] bb = (double[]) b.Clone();

            NumSharp.LAPACK.LAPACK.dgesv_(ref n, ref nrhs, a,ref lda, ipiv  , b, ref ldb, ref info);

            var c = MultiplyMatrix(aa,b);

            var z = 1;
        }

        protected double[] MultiplyMatrix(double[,]a,double[] b)
        {
            double[] c = new double[b.Length];

            for (int idx = 0; idx < a.GetLength(0);idx++)
                for (int jdx = 0; jdx < a.GetLength(1);jdx++)
                    c[idx] += a[idx,jdx] * b[jdx];
            
            return c;
        }
    }
}
