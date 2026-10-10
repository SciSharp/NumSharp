using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Convert angles from degrees to radians.
        /// </summary>
        /// <param name="x">Angles in degrees.</param>
        /// <param name="dtype">The dtype the returned ndarray should be of.</param>
        /// <returns>The corresponding angle in radians. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.deg2rad.html</remarks>
        public static NDArray deg2rad(NDArray x, NDArray @out = null, NDArray where = null, DType dtype = null)
            => NDMatrixBoundary.Sees(x)   // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            ? NDMatrixBoundary.Wrapped((x, @out, where, dtype), static s => deg2rad(s.x, s.@out, s.where, s.dtype), @out)
            : x.TensorEngine.Deg2Rad(x, dtype, @out, where);

        /// <summary>
        /// Convert angles from degrees to radians. Alias for deg2rad.
        /// </summary>
        /// <param name="x">Angles in degrees.</param>
        /// <param name="dtype">The dtype the returned ndarray should be of.</param>
        /// <returns>The corresponding angle in radians. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.radians.html</remarks>
        public static NDArray radians(NDArray x, NDArray @out = null, NDArray where = null, DType dtype = null)
            => NDMatrixBoundary.Sees(x)   // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            ? NDMatrixBoundary.Wrapped((x, @out, where, dtype), static s => radians(s.x, s.@out, s.where, s.dtype), @out)
            : deg2rad(x, @out, where, dtype);
    }
}
