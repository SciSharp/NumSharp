using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Convert angles from radians to degrees.
        /// </summary>
        /// <param name="x">Angle in radians.</param>
        /// <param name="outType">The dtype the returned ndarray should be of.</param>
        /// <returns>The corresponding angle in degrees. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.rad2deg.html</remarks>
        public static NDArray rad2deg(in NDArray x, NPTypeCode? outType = null)
            => x.TensorEngine.Rad2Deg(x, outType);

        /// <summary>
        /// Convert angles from radians to degrees.
        /// </summary>
        /// <param name="x">Angle in radians.</param>
        /// <param name="outType">The dtype the returned ndarray should be of.</param>
        /// <returns>The corresponding angle in degrees. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.rad2deg.html</remarks>
        public static NDArray rad2deg(in NDArray x, Type outType)
            => x.TensorEngine.Rad2Deg(x, outType);

        /// <summary>
        /// Convert angles from radians to degrees. Alias for rad2deg.
        /// </summary>
        /// <param name="x">Angle in radians.</param>
        /// <param name="outType">The dtype the returned ndarray should be of.</param>
        /// <returns>The corresponding angle in degrees. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.degrees.html</remarks>
        public static NDArray degrees(in NDArray x, NPTypeCode? outType = null)
            => rad2deg(x, outType);

        /// <summary>
        /// Convert angles from radians to degrees. Alias for rad2deg.
        /// </summary>
        /// <param name="x">Angle in radians.</param>
        /// <param name="outType">The dtype the returned ndarray should be of.</param>
        /// <returns>The corresponding angle in degrees. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.degrees.html</remarks>
        public static NDArray degrees(in NDArray x, Type outType)
            => rad2deg(x, outType);
    }
}
