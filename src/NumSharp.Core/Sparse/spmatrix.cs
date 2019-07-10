//using NumSharp;
//using System;
//using System.Collections.Generic;
//using System.Text;

//namespace NumSharp.Sparse
//{
//    public class spmatrix
//    {
//        public static matrix sum(_cs_matrix mx, int? axis = null, Type dtype = null)
//        {
//            var (m, n) = (mx.shape.Dimensions[0], mx.shape[1]);
//            matrix ret = null;

//            // sum over columns
//            if (axis == 0)
//            {
//                var matrix = np.asmatrix(np.ones(new Shape(1, m), dtype));
//                ret = matrix * mx;
//            }

//            return ret;
//        }

//        public static void csc_matvec(int n_row, int n_col, int[] Ap, int[] Ai, double[] Ax, double[] Xx, NDArray Yx)
//        {
//            //Uncommented during Unmanaged migration.
//            //for (int j = 0; j < n_col; j++)
//            //{
//            //    int col_start = Ap[j];
//            //    int col_end = Ap[j + 1];

//            //    for (int ii = col_start; ii < col_end; ii++)
//            //    {
//            //        int i = Ai[ii];
//            //        Yx.Data<double>()[i] += Ax[ii] * Xx[j];
//            //    }
//            //}
//        }
//    }
//}
