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
        public static NDArray deg2rad(in NDArray x, NPTypeCode? dtype = null)
            => x.TensorEngine.Deg2Rad(x, dtype);

        /// <summary>
        /// Convert angles from degrees to radians.
        /// </summary>
        /// <param name="x">Angles in degrees.</param>
        /// <param name="dtype">The dtype the returned ndarray should be of.</param>
        /// <returns>The corresponding angle in radians. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.deg2rad.html</remarks>
        public static NDArray deg2rad(in NDArray x, Type dtype)
            => x.TensorEngine.Deg2Rad(x, dtype);

        /// <summary>
        /// Convert angles from degrees to radians. Alias for deg2rad.
        /// </summary>
        /// <param name="x">Angles in degrees.</param>
        /// <param name="dtype">The dtype the returned ndarray should be of.</param>
        /// <returns>The corresponding angle in radians. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.radians.html</remarks>
        public static NDArray radians(in NDArray x, NPTypeCode? dtype = null)
            => deg2rad(x, dtype);

        /// <summary>
        /// Convert angles from degrees to radians. Alias for deg2rad.
        /// </summary>
        /// <param name="x">Angles in degrees.</param>
        /// <param name="dtype">The dtype the returned ndarray should be of.</param>
        /// <returns>The corresponding angle in radians. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.radians.html</remarks>
        public static NDArray radians(in NDArray x, Type dtype)
            => deg2rad(x, dtype);
    }
}
