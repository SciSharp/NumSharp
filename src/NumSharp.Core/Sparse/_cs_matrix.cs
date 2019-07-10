//using NumSharp;
//using System;
//using System.Collections.Generic;
//using System.Text;

//namespace NumSharp.Sparse
//{
//    /// <summary>
//    /// base matrix class for compressed row and column oriented matrices
//    /// </summary>
//    public partial class _cs_matrix
//    {
//        public NDArray data { get; set; }

//        public int ndim => data.ndim;

//        public Type dtype => data.dtype;

//        public Shape shape => data.shape;

//        public int maxprint { get; set; }

//        public int nnz
//        {
//            get
//            {
//                switch (data.dtype.Name)
//                {
//                    case "Int32":
//                        return data.Data<int>().Length;
//                    case "Double":
//                        return data.Data<double>().Length;
//                }

//                return 0;
//            }
//        }

//        public bool has_canonical_format { get; set; }

//        public bool has_sorted_indices { get; set; }

//        public string format { get; set; }

//        public NDArray indices { get; set; }

//        public NDArray indptr { get; set; }
//    }
//}
