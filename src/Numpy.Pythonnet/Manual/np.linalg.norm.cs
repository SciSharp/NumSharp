using System;
using System.Collections.Generic;
using System.Text;
using Numpy.Models;

namespace Numpy
{

    public static partial class np
    {
        public static partial class linalg
        {
            /// <summary>
            /// Matrix or vector norm.
            /// 
            /// This function is able to return one of eight different matrix norms,
            /// or one of an infinite number of vector norms (described below), depending
            /// on the value of the ord parameter.
            /// 
            /// Notes
            /// 
            /// For values of ord &lt;= 0, the result is, strictly speaking, not a
            /// mathematical ‘norm’, but it may still be useful for various numerical
            /// purposes.
            /// 
            /// The following norms can be calculated:
            /// 
            /// The Frobenius norm is given by [1]:
            /// 
            /// The nuclear norm is the sum of the singular values.
            /// 
            /// References
            /// </summary>
            /// <param name="x">
            /// Input array.  If axis is None, x must be 1-D or 2-D.
            /// </param>
            /// <param name="ord">
            /// Order of the norm (see table under Notes). inf means numpy’s
            /// inf object.
            /// </param>
            /// <param name="axis">
            /// If axis is an integer, it specifies the axis of x along which to
            /// compute the vector norms.  If axis is a 2-tuple, it specifies the
            /// axes that hold 2-D matrices, and the matrix norms of these matrices
            /// are computed.  If axis is None then either a vector norm (when x
            /// is 1-D) or a matrix norm (when x is 2-D) is returned.
            /// </param>
            /// <param name="keepdims">
            /// If this is set to True, the axes which are normed over are left in the
            /// result as dimensions with size one.  With this option the result will
            /// broadcast correctly against the original x.
            /// </param>
            /// <returns>
            /// Norm of the matrix or vector(s).
            /// </returns>
            public static NDarray norm(NDarray x, int? ord, int[] axis, bool? keepdims = null)
                => NumPy.Instance.norm(x, ord, axis, keepdims);

            public static float norm(NDarray x, int? ord=null)
                => NumPy.Instance.norm(x, ord);

            public static float norm(NDarray x, string ord)
                => NumPy.Instance.norm(x, ord);

            public static float norm(NDarray x, Constants ord)
                => NumPy.Instance.norm(x, ord);
        }
    }
}
