using System;
using System.Runtime.InteropServices;

namespace NumSharp.Core
{
    public static class LAPACK
    {
        [DllImport("lapack")]
        public static extern void dgesv_(ref int n, ref int nrhs, double[] a, ref int lda, int[] ipiv, double[] b, ref int ldb, ref int info  );
        [DllImport("lapack")]
        public static extern void dgeqrf_(ref int m, ref int n, double[] a, ref int lda, double[] tau, double[] work, ref int lwork, ref int info);
        [DllImport("lapack")]
        public static extern void dorgqr_(ref int m,ref int n, ref int k, double[] a, ref int lda, double[] tau, double[]work, ref int lwork,ref int info);        
    }
}
