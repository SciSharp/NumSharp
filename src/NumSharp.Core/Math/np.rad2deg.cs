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
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray rad2deg(NDArray x, NDArray @out = null, NDArray where = null, DType dtype = null)
            => x.TensorEngine.Rad2Deg(x, dtype, @out, where);

        /// <summary>
        /// Convert angles from radians to degrees. Alias for rad2deg.
        /// </summary>
        /// <param name="x">Angle in radians.</param>
        /// <param name="dtype">The dtype the returned ndarray should be of.</param>
        /// <returns>The corresponding angle in degrees. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.degrees.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <exception cref="ArgumentNullException"><paramref name="x"/> is null.</exception>
        public static NDArray degrees(NDArray x, NDArray @out = null, NDArray where = null, DType dtype = null)
        {
            if (x is null) throw new ArgumentNullException(nameof(x));
            return x.TensorEngine.Degrees(x, dtype, @out, where);
        }
    }
}
