using System;

namespace NumSharp
{
    public partial class NDArray
    {
        public (NDArray, NDArray, NDArray) svd()
        {
            return default;
            //return;
            //double[] A = Data<double>();

            //int m = this.shape[0];
            //int n = this.shape[1]; 
            //int lda = m;
            //int ldu = m; 
            //int ldvt = n; 
            //int info = 0;
            //int lwork = -1;

            //double[] work = new double[Math.Max(m,n)];

            //double[] s = new double[n]; 
            //double[] u = new double[m*m];
            //double[] vt = new double[n*n];

            //LAPACK.dgesvd_("ALL".ToCharArray(),"All".ToCharArray(),ref m, ref n, A, ref lda, s, u, ref ldu, vt, ref ldvt, work, ref lwork, ref info );

            //lwork = (int)work[0];

            //work = new double[lwork]; 

            //LAPACK.dgesvd_("ALL".ToCharArray(),"All".ToCharArray(),ref m, ref n, A, ref lda, s, u, ref ldu, vt, ref ldvt, work, ref lwork, ref info );

            //var uNDArr = new NDArray(typeof(double), new Shape(m, n));
            //var vtNDArr = new NDArray(typeof(double), new Shape(n, n));
            //var sNDArr = new NDArray(typeof(double), n);

            //// set variables
            //double[] uDouble = uNDArr.Storage.GetData<double>();

            //for (int idx = 0; idx < uNDArr.size;idx++)
            //    uDouble[idx] = u[idx];

            //vtNDArr.Storage.ReplaceData(vt);
            //sNDArr.Storage.ReplaceData(s);

            //return (uNDArr,sNDArr,vtNDArr);
        }

        public void SetData(object p)
        {
            throw new NotImplementedException();
        }
    }
}
