using System.Runtime.InteropServices;

namespace NumSharp.LAPACKProvider
{
    public static partial class NetLib
    {
        [DllImport("lapack")]
        public static extern void dgesv_(ref int n, ref int nrhs, double[] a, ref int lda, int[] ipiv, double[] b, ref int ldb, ref int info);

        [DllImport("lapack")]
        public static extern void dgeqrf_(ref int m, ref int n, double[] a, ref int lda, double[] tau, double[] work, ref int lwork, ref int info);

        [DllImport("lapack")]
        public static extern void dorgqr_(ref int m, ref int n, ref int k, double[] a, ref int lda, double[] tau, double[] work, ref int lwork, ref int info);

        [DllImport("lapack")]
        public static extern void dgelss_(ref int m, ref int n, ref int nrhs, double[] a, ref int lda, double[] b, ref int ldb, double[] s, ref double rcond, ref int rank, double[] work, ref int lwork, ref int info);

        // not working well
        [DllImport("lapack", CallingConvention = CallingConvention.Cdecl), System.Security.SuppressUnmanagedCodeSecurity]
        public static extern void dgelsy_(ref int m, ref int n, ref int nrhs, double[] a, ref int lda, double[] b, ref int ldb, int[] jpvt, ref double rcond, ref int rank, double[] work, ref int lwork, ref int info);

        [DllImport("lapack")]
        public static extern void dgesvd_(char[] JOBU, char[] JOBVT, ref int M, ref int N, double[] A, ref int LDA, double[] S, double[] U, ref int LDU, double[] VT, ref int LDVT, double[] WORK, ref int LWORK, ref int INFO);
    }
}
