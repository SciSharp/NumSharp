using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Convert angles from radians to degrees.
        /// </summary>
        /// <param name="x">Angle in radians.</param>
        /// <param name="dtype">The dtype the returned ndarray should be of.</param>
        /// <returns>The corresponding angle in degrees. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.rad2deg.html</remarks>
        public static NDArray rad2deg(NDArray x, NDArray @out = null, NDArray where = null, DType dtype = null)
            => NDMatrixBoundary.Sees(x)   // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            ? NDMatrixBoundary.Wrapped((x, @out, where, dtype), static s => rad2deg(s.x, s.@out, s.where, s.dtype), @out)
            : x.TensorEngine.Rad2Deg(x, dtype, @out, where);

        /// <summary>
        /// Convert angles from radians to degrees. Alias for rad2deg.
        /// </summary>
        /// <param name="x">Angle in radians.</param>
        /// <param name="dtype">The dtype the returned ndarray should be of.</param>
        /// <returns>The corresponding angle in degrees. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.degrees.html</remarks>
        public static NDArray degrees(NDArray x, NDArray @out = null, NDArray where = null, DType dtype = null)
            => NDMatrixBoundary.Sees(x)   // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            ? NDMatrixBoundary.Wrapped((x, @out, where, dtype), static s => degrees(s.x, s.@out, s.where, s.dtype), @out)
            : rad2deg(x, @out, where, dtype);
    }
}
