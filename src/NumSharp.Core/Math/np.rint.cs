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
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray rint(NDArray x, NDArray @out = null, NDArray where = null, DType dtype = null)
            => NDMatrixBoundary.Sees(x)   // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            ? NDMatrixBoundary.Wrapped((x, @out, where, dtype), static s => rint(s.x, s.@out, s.where, s.dtype), @out)
            : x.TensorEngine.Rint(x, dtype, @out, where);
    }
}
