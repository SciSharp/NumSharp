//using NumSharp;
//using System;
//using System.Collections.Generic;
//using System.Text;

//namespace NumSharp.Sparse
//{
//    public partial class _cs_matrix
//    {
//        public static matrix operator *(_cs_matrix m1, matrix m2)
//        {
//            return null;
//            //m1.transpose();
//            //var (M, N) = (m1.shape[0], m1.shape[1]);

//            //var result = np.zeros(M);

//            //spmatrix.csc_matvec(M, N, m1.indptr.Data<int>(), m1.indices.Data<int>(), m1.data.Data<double>(), m2.Data<double>(), result);

//            //m1.transpose();
//            //return np.asmatrix(np.asmatrix(result).transpose());
//        }

//        public static matrix operator *(matrix m2, _cs_matrix m1)
//        {
//            return m1 * m2;
//        }
//    }
//}
