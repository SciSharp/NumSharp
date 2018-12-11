using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using NumSharp.Core.Shared;

namespace NumSharp.Core
{
    public partial class NDArray
    {
        
        public NDArray lstqr(NDArray nDArrayB, double rcon = 0.0001)
        {
            var AT = this.transpose();
            var bT = nDArrayB.transpose();

            var A = (double[]) AT.Storage.GetData<double>().Clone();
            var b = (double[]) bT.Storage.GetData<double>().Clone();

            int m = this.shape.Shapes[0];
            int n = this.shape.Shapes[1];

            int nrhs = nDArrayB.shape.Shapes[1];

            int lda = m;
            int ldb = m;

            int rank = 0;

            double[] work = new double[1];
            int lwork = -1;
            int info = 0;
            double[] singVal = new double[m];

            LAPACK.dgelss_(ref m,ref n, ref nrhs, A, ref lda, b, ref ldb,singVal , ref rcon, ref rank, work,ref lwork, ref info  );

            lwork = (int) work[0];
            work = new double[lwork];

            LAPACK.dgelss_(ref m,ref n, ref nrhs, A, ref lda, b, ref ldb,singVal , ref rcon, ref rank, work,ref lwork, ref info  );

            double[] sln = new double[n * nrhs];

            for (int idx = 0; idx < sln.Length;idx++)
                sln[idx] = b[m * (idx % nrhs) + idx / nrhs];
            
            var slnArr = new NDArray(typeof(double),new Shape(n,nrhs));

            slnArr.Storage.SetData(sln);

            return slnArr;

        }
        
    }
}