namespace NumSharp
{
    public partial class NDArray
    {
        public (NDArray, NDArray) qr()
        {
            return default;
            //return;
            //var a = this.Storage.GetData<double>().Clone();

            //int m = this.Storage.Shape.Dimensions[0];
            //int n = this.Storage.Shape.Dimensions[1];

            //int lda = m;

            //double[] tau = new double[ Math.Min(m,n) ];
            //double[] work = new double[ Math.Max(m,n) ];

            //int lwork = m;

            //int info = 0;

            //LAPACK.dgeqrf_(ref m,ref n, a ,ref lda, tau, work,ref lwork,ref info);

            //double[] RDouble = new double[n*n];

            //for(int idx = 0; idx < n; idx++)
            //    for(int jdx = idx;jdx < n;jdx++)
            //        RDouble[idx+jdx * n] = a[idx+jdx*n];


            //var R = new NDArray(typeof(double), new Shape(n, n));

            //R.Storage.ReplaceData(RDouble);

            //int k = tau.Length;

            //LAPACK.dorgqr_(ref m, ref n,ref k ,a, ref lda, tau, work, ref lwork,ref info);

            //var Q = new NDArray(typeof(double), new Shape(tau.Length, tau.Length));

            //Q.Storage.Allocate(Q.Storage.Shape);
            //Q.Storage.ReplaceData(a);

            //return (Q,R);
        }
    }
}
