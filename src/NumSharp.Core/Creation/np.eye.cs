using System;
using NumSharp.Backends;
using NumSharp.Utilities;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return the identity array. The identity array is a square array with ones on the main diagonal.
        /// </summary>
        /// <param name="n">Number of rows (and columns) in n x n output.</param>
        /// <param name="dtype">Data-type of the output. Defaults to double.</param>
        /// <returns>n x n array with its main diagonal set to one, and all other elements 0.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.identity.html</remarks>
        public static NDArray identity(int n, DType dtype = null)
        {
            return eye(n, dtype: dtype ?? DType.Double);
        }

        /// <summary>
        ///     Return a 2-D array with ones on the diagonal and zeros elsewhere.
        /// </summary>
        /// <param name="N">Number of rows in the output.</param>
        /// <param name="M">Number of columns in the output. If None, defaults to N.</param>
        /// <param name="k">Index of the diagonal: 0 (the default) refers to the main diagonal, a positive value refers to an upper diagonal, and a negative value to a lower diagonal.</param>
        /// <param name="dtype">Data-type of the returned array.</param>
        /// <param name="order">Memory layout: 'C' (row-major, default) or 'F' (column-major).</param>
        /// <param name="device">Target device. Only <c>"cpu"</c> and <c>null</c> are accepted (Array-API parity).</param>
        /// <returns>An array where all elements are equal to zero, except for the k-th diagonal, whose values are equal to one.</returns>
        /// <exception cref="ValueError"><paramref name="order"/> is not one of C, F, A, K in either case (<c>order must be one
        ///     of 'C', 'F', 'A', or 'K' (got 'Z')</c>), or it is A or K (<c>only 'C' or 'F' order is permitted</c>);
        ///     <paramref name="N"/> or <paramref name="M"/> is negative (<c>negative dimensions are not allowed</c>); or the
        ///     dtype's setitem refuses the Python int 1 (a <c>StringDType(coerce=False)</c>).</exception>
        /// <exception cref="ArgumentException"><paramref name="device"/> is neither null nor <c>"cpu"</c>.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.eye.html. NumPy's eye is
        ///     <c>m = zeros((N, M), dtype=dtype, order=order, device=device)</c> and so raises in np.zeros' order, which this
        ///     follows: the order's letter (an argument conversion), the device, the order's C-or-F rule, then the
        ///     dimensions (the allocation).
        /// </remarks>
        public static NDArray eye(int N, int? M = null, int k = 0, DType dtype = null, char order = 'C', string device = null)
        {
            // np.zeros parses its order (PyArray_OrderConverter) before its device: an unknown letter is reported first.
            OrderResolver.Convert(order);
            ValidateDevice(device);
            return NDArray.Eye(N, M, k, dtype, order);
        }
    }
}
