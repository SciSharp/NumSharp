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
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.identity.html</remarks>
        public static NDArray identity(int n, Type dtype = null)
        {
            return eye(n, dtype: dtype ?? typeof(double));
        }

        /// <summary>
        ///     Return a 2-D array with ones on the diagonal and zeros elsewhere.
        /// </summary>
        /// <param name="N">Number of rows in the output.</param>
        /// <param name="M">Number of columns in the output. If None, defaults to N.</param>
        /// <param name="k">Index of the diagonal: 0 (the default) refers to the main diagonal, a positive value refers to an upper diagonal, and a negative value to a lower diagonal.</param>
        /// <param name="dtype">Data-type of the returned array.</param>
        /// <returns>An array where all elements are equal to zero, except for the k-th diagonal, whose values are equal to one.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.eye.html</remarks>
        public static NDArray eye(int N, int? M=null, int k = 0, Type dtype = null)
        {
            if (!M.HasValue)
                M = N;
            var m = np.zeros(Shape.Matrix(N, M.Value), dtype ?? typeof(double));
            if (k >= M)
                return m;
            int i;
            if (k >= 0)
            {
                i = k;
            }
            else
                i = (-k) * M.Value;

            var flat = m.flat;
            var one = dtype != null ? Converts.ChangeType(1d, dtype.GetTypeCode()) : 1d;
            int skips = k < 0 ? Math.Abs(k)-1 : 0;
            for (int j = k; j < flat.size; j+=N+1)
            {
                if (j < 0 || skips-- > 0)
                    continue;
                flat.SetAtIndex(one, j);
            }

            return m;
        }
    }
}
