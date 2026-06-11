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
        public static NDArray rad2deg(NDArray x, NDArray @out = null, NDArray where = null, NPTypeCode? dtype = null)
            => x.TensorEngine.Rad2Deg(x, dtype, @out, where);

        /// <summary>
        ///     Computed in <paramref name="dtype"/> — positional-dtype convenience
        ///     overload (NumPy accepts dtype only as a keyword).
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="dtype">The loop dtype the computation should run in.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.rad2deg.html</remarks>
        public static NDArray rad2deg(NDArray x, NPTypeCode dtype)
            => x.TensorEngine.Rad2Deg(x, dtype);

        /// <summary>
        /// Convert angles from radians to degrees.
        /// </summary>
        /// <param name="x">Angle in radians.</param>
        /// <param name="dtype">The dtype the returned ndarray should be of.</param>
        /// <returns>The corresponding angle in degrees. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.rad2deg.html</remarks>
        public static NDArray rad2deg(NDArray x, Type dtype)
            => x.TensorEngine.Rad2Deg(x, dtype);

        /// <summary>
        /// Convert angles from radians to degrees. Alias for rad2deg.
        /// </summary>
        /// <param name="x">Angle in radians.</param>
        /// <param name="dtype">The dtype the returned ndarray should be of.</param>
        /// <returns>The corresponding angle in degrees. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.degrees.html</remarks>
        public static NDArray degrees(NDArray x, NDArray @out = null, NDArray where = null, NPTypeCode? dtype = null)
            => rad2deg(x, @out, where, dtype);

        /// <summary>
        /// Alias for rad2deg — positional-dtype convenience overload.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.degrees.html</remarks>
        public static NDArray degrees(NDArray x, NPTypeCode dtype)
            => rad2deg(x, dtype);

        /// <summary>
        /// Convert angles from radians to degrees. Alias for rad2deg.
        /// </summary>
        /// <param name="x">Angle in radians.</param>
        /// <param name="dtype">The dtype the returned ndarray should be of.</param>
        /// <returns>The corresponding angle in degrees. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.degrees.html</remarks>
        public static NDArray degrees(NDArray x, Type dtype)
            => rad2deg(x, dtype);
    }
}
