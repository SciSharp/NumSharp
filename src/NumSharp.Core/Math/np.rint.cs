using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Round elements of the array to the nearest integer, element-wise (round half to even).
        /// The result is a float (rint has no integer loop): integer/bool inputs promote to the
        /// float tier (bool/int8/uint8 -> float16, int16/uint16 -> float32, int32/int64/... ->
        /// float64), floats and complex are preserved. The real and imaginary parts of complex
        /// numbers are rounded separately.
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="out">A location into which the result is stored (same_kind cast from the loop dtype).</param>
        /// <param name="where">Boolean mask; compute only where true, leaving other <paramref name="out"/> slots unchanged.</param>
        /// <param name="dtype">Loop dtype override (must be a float/complex loop).</param>
        /// <returns>An array of the same shape as x, containing the rounded values. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.rint.html</remarks>
        public static NDArray rint(NDArray x, NDArray @out = null, NDArray where = null, NPTypeCode? dtype = null)
            => x.TensorEngine.Rint(x, dtype, @out, where);

        /// <summary>
        ///     Computed in <paramref name="dtype"/> — positional-dtype convenience
        ///     overload (NumPy accepts dtype only as a keyword; its 2nd positional is out=).
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="dtype">The loop dtype the computation should run in.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.rint.html</remarks>
        public static NDArray rint(NDArray x, NPTypeCode dtype)
            => x.TensorEngine.Rint(x, dtype);

        /// <summary>
        /// Round elements of the array to the nearest integer, element-wise (round half to even).
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="dtype">The loop dtype the computation should run in.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.rint.html</remarks>
        public static NDArray rint(NDArray x, Type dtype)
            => x.TensorEngine.Rint(x, dtype);
    }
}
