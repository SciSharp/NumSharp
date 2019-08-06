namespace NumSharp
{
    public static partial class LAPACK
    {
        public static void dgesv_(ref int n, ref int nrhs, double[] a, ref int lda, int[] ipiv, double[] b, ref int ldb, ref int info)
        {
            switch (np.LAPACKProvider)
            {
                case LAPACKProviderType.NetLib:
                {
                    LAPACKProvider.NetLib.dgesv_(ref n, ref nrhs, a, ref lda, ipiv, b, ref ldb, ref info);
                    break;
                }
            }
        }

        public static void dgeqrf_(ref int m, ref int n, double[] a, ref int lda, double[] tau, double[] work, ref int lwork, ref int info)
        {
            switch (np.LAPACKProvider)
            {
                case LAPACKProviderType.NetLib:
                {
                    LAPACKProvider.NetLib.dgeqrf_(ref m, ref n, a, ref lda, tau, work, ref lwork, ref info);
                    break;
                }
            }
        }

        public static void dorgqr_(ref int m, ref int n, ref int k, double[] a, ref int lda, double[] tau, double[] work, ref int lwork, ref int info)
        {
            switch (np.LAPACKProvider)
            {
                case LAPACKProviderType.NetLib:
                {
                    LAPACKProvider.NetLib.dorgqr_(ref m, ref n, ref k, a, ref lda, tau, work, ref lwork, ref info);
                    break;
                }
            }
        }

        public static void dgelss_(ref int m, ref int n, ref int nrhs, double[] a, ref int lda, double[] b, ref int ldb, double[] s, ref double rcond, ref int rank, double[] work, ref int lwork, ref int info)
        {
            switch (np.LAPACKProvider)
            {
                case LAPACKProviderType.NetLib:
                {
                    LAPACKProvider.NetLib.dgelss_(ref m, ref n, ref nrhs, a, ref lda, b, ref ldb, s, ref rcond, ref rank, work, ref lwork, ref info);
                    break;
                }
            }
        }

        public static void dgesvd_(char[] JOBU, char[] JOBVT, ref int M, ref int N, double[] A, ref int LDA, double[] S, double[] U, ref int LDU, double[] VT, ref int LDVT, double[] WORK, ref int LWORK, ref int INFO)
        {
            switch (np.LAPACKProvider)
            {
                case LAPACKProviderType.NetLib:
                {
                    LAPACKProvider.NetLib.dgesvd_(JOBU, JOBVT, ref M, ref N, A, ref LDA, S, U, ref LDU, VT, ref LDVT, WORK, ref LWORK, ref INFO);
                    break;
                }
            }
        }
    }
}
