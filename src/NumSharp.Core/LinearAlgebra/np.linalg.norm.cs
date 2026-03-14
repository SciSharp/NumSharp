using System;
using System.Numerics;
using NumSharp.Utilities;

namespace NumSharp
{
    public static partial class np
    {
        public static partial class linalg
        {
            /// <summary>
            /// Matrix or vector norm.
            /// This function is able to return one of seven different matrix norms,
            /// or one of an infinite number of vector norms(described below), depending
            /// on the value of the ``ord`` parameter.
            /// </summary>
            /// <param name="x">Input array.  If `axis` is None, `x` must be 1-D or 2-D.</param>
            /// <param name="ord">non-zero int, "inf", "-inf" or "fro", optional
            /// Order of the norm(see table under ``Notes``). </param>
            /// <param name="axis">int, 2-tuple of ints int[], null}, optional
            /// If `axis` is an integer, it specifies the axis of `x` along which to
            /// compute the vector norms.If `axis` is a 2-tuple, it specifies the
            /// axes that hold 2-D matrices, and the matrix norms of these matrices
            /// are computed.If `axis` is None then either a vector norm (when `x`
            /// is 1-D) or a matrix norm(when `x` is 2-D) is returned.</param>
            /// <returns>
            /// Norm of the matrix or vector(s) as double or NDArray.
            /// </returns>
            /// <remarks>
            /// Notes:
            /// -----
            /// For values of ``ord &lt;= 0``, the result is, strictly speaking, not a
            /// mathematical 'norm', but it may still be useful for various numerical
            /// purposes.
            /// The following norms can be calculated:
            /// =====  ============================  ==========================
            /// ord norm for matrices norm for vectors
            /// =====  ============================  ==========================
            /// None Frobenius norm                2-norm
            /// 'fro'  Frobenius norm                --
            /// inf max(sum(abs(x), axis= 1))      max(abs(x))
            /// -inf min(sum(abs(x), axis= 1))      min(abs(x))
            /// 0      --                            sum(x != 0)
            /// 1      max(sum(abs(x), axis= 0))      as below
            /// -1     min(sum(abs(x), axis= 0))      as below
            /// 2      2-norm(largest sing. value)  as below
            /// -2     smallest singular value       as below
            /// other  --                            sum(abs(x) ** ord)**(1./ord)
            /// =====  ============================  ==========================
            /// The Frobenius norm is given by[1]_:
            /// :math:`||A||_F = [\\sum_{i, j}
            /// abs(a_{ i,j})^2]^{1/2}`
            /// References
            /// ----------
            /// .. [1] G.H.Golub and C.F.Van Loan, *Matrix Computations*,
            /// Baltimore, MD, Johns Hopkins University Press, 1985, pg. 15
            /// </remarks>
            private static object norm(NDArray x, object ord = null, object axis_obj = null)
            {
                // Check the default case first and handle it immediately.
                if (ord == null && axis_obj == null)
                {
                    x = x.ravel(); // order = 'K'
                    NDArray sqnorm = null;
                    if (x.dtype == typeof(Complex))
                        // Python: sqnorm = dot(x.real, x.real) + dot(x.imag, x.imag)
                        throw new NotImplementedException("Complex is not implemented yet");
                    else
                        sqnorm = dot(x, x);
                    return sqrt(sqnorm);
                }

                // Normalize the `axis` argument to a tuple.
                var dim = x.ndim;
                int[] axis = null;
                if (axis_obj == null)
                    axis = py.range(dim);
                else if (axis_obj is int)
                    axis = new int[] {(int)axis_obj};
                else if (axis_obj is int[])
                    axis = (int[])axis_obj;
                else
                    throw new ArgumentException($"Invalid axis type: {axis_obj}");
//                if (axis.Length == 1)
//                {
//                    if ((string)ord == "inf")
//                        return abs(x).max(axis = axis);
//    elif ord == -Inf:
//        return abs(x).min(axis = axis)
//    elif ord == 0:
//        # Zero norm
//        return (x != 0).sum(axis = axis)
//    elif ord == 1:
//        # special case for speedup
//        return add.reduce(abs(x), axis = axis)
//    elif ord is None or ord == 2:
//        # special case for speedup
//        s = (x.conj() * x).real
//        return sqrt(add.reduce(s, axis = axis))
//    else:
//        try:
//            ord + 1
//        except TypeError:
//            raise ValueError("Invalid norm order for vectors.")
//        if x.dtype.type is longdouble:
//            # Convert to a float type, so integer arrays give
//            # float results.  Don't apply asfarray to longdouble arrays,
//            # because it will downcast to float64.
//            absx = abs(x)
//        else:
//            absx = x if isComplexType(x.dtype.type) else asfarray(x)
//            if absx.dtype is x.dtype:
//                absx = abs(absx)
//            else:
//                #if the type changed, we can safely overwrite absx
//                abs(absx, out=absx)
//        absx **= ord
//        return add.reduce(absx, axis=axis) ** (1.0 / ord)

//                }
//elif len(axis) == 2:
//    row_axis, col_axis = axis
//    if not (-nd <= row_axis < nd and -nd <= col_axis < nd):
//        raise ValueError('Invalid axis %r for an array with shape %r' %
//                         (axis, x.shape))
//    if row_axis % nd == col_axis % nd:
//        raise ValueError('Duplicate axes given.')
//    if ord == 2:
//        return _multi_svd_norm(x, row_axis, col_axis, amax)
//    elif ord == -2:
//        return _multi_svd_norm(x, row_axis, col_axis, amin)
//    elif ord == 1:
//        if col_axis > row_axis:
//            col_axis -= 1
//        return add.reduce(abs(x), axis=row_axis).max(axis=col_axis)
//    elif ord == Inf:
//        if row_axis > col_axis:
//            row_axis -= 1
//        return add.reduce(abs(x), axis=col_axis).max(axis=row_axis)
//    elif ord == -1:
//        if col_axis > row_axis:
//            col_axis -= 1
//        return add.reduce(abs(x), axis=row_axis).min(axis=col_axis)
//    elif ord == -Inf:
//        if row_axis > col_axis:
//            row_axis -= 1
//        return add.reduce(abs(x), axis=col_axis).min(axis=row_axis)
//    elif ord in [None, 'fro', 'f']:
//        return sqrt(add.reduce((x.conj() * x).real, axis=axis))
//    else:
//        raise ValueError("Invalid norm order for matrices.")
//else:
//    raise ValueError("Improper number of dimensions to norm.")
                throw new NotImplementedException();
            }

            /*
              >>> from numpy import linalg as LA
    >>> a = np.arange(9) - 4
    >>> a
    array([-4, -3, -2, -1,  0,  1,  2,  3,  4])
    >>> b = a.reshape((3, 3))
    >>> b
    array([[-4, -3, -2],
           [-1,  0,  1],
           [ 2,  3,  4]])
    >>> LA.norm(a)
    7.745966692414834
    >>> LA.norm(b)
    7.745966692414834
    >>> LA.norm(b, 'fro')
    7.745966692414834
    >>> LA.norm(a, np.inf)
    4
    >>> LA.norm(b, np.inf)
    9
    >>> LA.norm(a, -np.inf)
    0
    >>> LA.norm(b, -np.inf)
    2
    >>> LA.norm(a, 1)
    20
    >>> LA.norm(b, 1)
    7
    >>> LA.norm(a, -1)
    -4.6566128774142013e-010
    >>> LA.norm(b, -1)
    6
    >>> LA.norm(a, 2)
    7.745966692414834
    >>> LA.norm(b, 2)
    7.3484692283495345
    >>> LA.norm(a, -2)
    nan
    >>> LA.norm(b, -2)
    1.8570331885190563e-016
    >>> LA.norm(a, 3)
    5.8480354764257312
    >>> LA.norm(a, -3)
    nan
    Using the `axis` argument to compute vector norms:
    >>> c = np.array([[1, 2, 3],
    ...               [-1, 1, 4]])
    >>> LA.norm(c, axis=0)
    array([ 1.41421356,  2.23606798,  5.        ])
    >>> LA.norm(c, axis=1)
    array([ 3.74165739,  4.24264069])
    >>> LA.norm(c, ord=1, axis=1)
    array([6, 6])
    Using the `axis` argument to compute matrix norms:
    >>> m = np.arange(8).reshape(2,2,2)
    >>> LA.norm(m, axis=(1,2))
    array([  3.74165739,  11.22497216])
    >>> LA.norm(m[0, :, :]), LA.norm(m[1, :, :])
    (3.7416573867739413, 11.224972160321824)
    """
    
    
             */
        }
    }
}
