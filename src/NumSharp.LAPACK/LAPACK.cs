using System;
using System.Runtime.InteropServices;

namespace NumSharp.LAPACK
{
    public static class LAPACK
    {
        [DllImport("lapack_win64_MT.dll")]
        public static extern void dgesv_(ref int n, ref int nrhs, double[] a, ref int lda, int[] ipiv, double[] b, ref int ldb, ref int info  );
    }
}
