namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        /// Least Square method
        /// 
        /// Determines NDArray X which reduces least square error of Linear System A * X = B.
        /// This NDArray is equal to A.
        /// </summary>
        /// <param name="nDArrayB">Result NDArray B</param>
        /// <param name="rcon"></param>
        /// <returns>NArray X</returns>
        public NDArray lstqr(NDArray nDArrayB, double rcon = 0.0001)
        {
            return null;
            //var A = (double[]) Data<double>();
            //var b = (double[])nDArrayB.Data<double>();

            //int m = this.shape[0];
            //int n = this.shape[1];

            //int nrhs = (nDArrayB.ndim > 1) ? nDArrayB.shape[1] : 1;

            //int lda = m;
            //int ldb = m;

            //int rank = 0;

            //double[] work = new double[1];
            //int lwork = -1;
            //int info = 0;
            //double[] singVal = new double[m];

            //LAPACK.dgelss_(ref m,ref n, ref nrhs, A, ref lda, b, ref ldb,singVal , ref rcon, ref rank, work,ref lwork, ref info  );

            //lwork = (int) work[0];
            //work = new double[lwork];

            //LAPACK.dgelss_(ref m,ref n, ref nrhs, A, ref lda, b, ref ldb,singVal , ref rcon, ref rank, work,ref lwork, ref info  );

            //double[] sln = new double[n * nrhs];

            //for (int idx = 0; idx < sln.Length;idx++)
            //    sln[idx] = b[m * (idx % nrhs) + idx / nrhs];

            //var slnArr = new NDArray(typeof(double),new Shape(n,nrhs));

            //slnArr.Storage.ReplaceData(sln);

            //return slnArr;
        }
    }
}
