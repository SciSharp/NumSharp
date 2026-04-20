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
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.eye.html</remarks>
        public static NDArray eye(int N, int? M = null, int k = 0, Type dtype = null)
        {
            int cols = M ?? N;
            var resolvedType = dtype ?? typeof(double);
            var m = np.zeros(Shape.Matrix(N, cols), resolvedType);
            if (N == 0 || cols == 0)
                return m;

            // Diagonal element count: rows where 0 <= i < N and 0 <= i+k < cols
            int rowStart = Math.Max(0, -k);
            int rowEnd = Math.Min(N, cols - k);
            if (rowEnd <= rowStart)
                return m;

            var typeCode = resolvedType.GetTypeCode();
            object one;
            switch (typeCode)
            {
                case NPTypeCode.Complex: one = new System.Numerics.Complex(1d, 0d); break;
                case NPTypeCode.Half:    one = (Half)1; break;
                case NPTypeCode.SByte:   one = (sbyte)1; break;
                case NPTypeCode.String:  one = "1"; break;
                case NPTypeCode.Char:    one = '1'; break;
                default:                 one = Converts.ChangeType((byte)1, typeCode); break;
            }

            // Flat index of element (i, i+k) in row-major (N, cols) layout = i*cols + (i+k).
            var flat = m.flat;
            for (int i = rowStart; i < rowEnd; i++)
                flat.SetAtIndex(one, (long)i * cols + (i + k));

            return m;
        }
    }
}
