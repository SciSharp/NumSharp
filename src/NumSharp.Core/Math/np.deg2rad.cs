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
        public static NDArray deg2rad(NDArray x, NDArray @out = null, NDArray where = null, NPTypeCode? dtype = null)
            => x.TensorEngine.Deg2Rad(x, dtype, @out, where);

        /// <summary>
        ///     Computed in <paramref name="dtype"/> — positional-dtype convenience
        ///     overload (NumPy accepts dtype only as a keyword).
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="dtype">The loop dtype the computation should run in.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.deg2rad.html</remarks>
        public static NDArray deg2rad(NDArray x, NPTypeCode dtype)
            => x.TensorEngine.Deg2Rad(x, dtype);

        /// <summary>
        /// Convert angles from degrees to radians.
        /// </summary>
        /// <param name="x">Angles in degrees.</param>
        /// <param name="dtype">The dtype the returned ndarray should be of.</param>
        /// <returns>The corresponding angle in radians. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.deg2rad.html</remarks>
        public static NDArray deg2rad(NDArray x, Type dtype)
            => x.TensorEngine.Deg2Rad(x, dtype);

        /// <summary>
        /// Convert angles from degrees to radians. Alias for deg2rad.
        /// </summary>
        /// <param name="x">Angles in degrees.</param>
        /// <param name="dtype">The dtype the returned ndarray should be of.</param>
        /// <returns>The corresponding angle in radians. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.radians.html</remarks>
        public static NDArray radians(NDArray x, NDArray @out = null, NDArray where = null, NPTypeCode? dtype = null)
            => deg2rad(x, @out, where, dtype);

        /// <summary>
        /// Alias for deg2rad — positional-dtype convenience overload.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.radians.html</remarks>
        public static NDArray radians(NDArray x, NPTypeCode dtype)
            => deg2rad(x, dtype);

        /// <summary>
        /// Convert angles from degrees to radians. Alias for deg2rad.
        /// </summary>
        /// <param name="x">Angles in degrees.</param>
        /// <param name="dtype">The dtype the returned ndarray should be of.</param>
        /// <returns>The corresponding angle in radians. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.radians.html</remarks>
        public static NDArray radians(NDArray x, Type dtype)
            => deg2rad(x, dtype);
    }
}
