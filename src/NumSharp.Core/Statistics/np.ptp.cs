using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Range of values (maximum - minimum) along an axis.
        ///     Equivalent to <c>np.amax(a, axis) - np.amin(a, axis)</c>; dtype is preserved,
        ///     so unsigned/signed integer overflow wraps the same way NumPy does
        ///     (e.g. <c>ptp(uint8[0,255]) == 255</c>, <c>ptp(int8[-128,127]) == -1</c>).
        /// </summary>
        /// <param name="a">The array (numbers or StringDType; fixed-width text is refused as NumPy refuses it).</param>
        /// <param name="axis">The reduced axis (negative counts from the end; a 0-d array accepts 0 and -1), or null for
        ///     every axis.</param>
        /// <param name="out">An optional output of the result's shape; the result is written into it and it is
        ///     returned.</param>
        /// <param name="keepdims">Whether the reduced axis stays with length one.</param>
        /// <returns><paramref name="out"/> itself when given; otherwise the range, a fresh array (read-only when 0-d:
        ///     NumPy's scalar); a <c>numpy.matrix</c> operand gives a matrix.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ptp.html — NumPy's whole function is
        ///     the engine's <see cref="TensorEngine.Ptp(NDArray, int?, NDArray, bool)"/> (the np-layer dtype plan, P1b and
        ///     PE); this applies the matrix rule and calls it.</remarks>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range.</exception>
        /// <exception cref="TypeError">Fixed-width text, or a text out (NumPy's maximum.reduce refuses the loop).</exception>
        /// <exception cref="ArgumentException"><paramref name="out"/> does not have the result's shape.</exception>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>); a read-only out; an empty reduction (no identity).</exception>
        [NDScoped]
        public static NDArray ptp(NDArray a, int? axis = null, NDArray @out = null, bool keepdims = false)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Wrapped((a, axis, @out, keepdims), static s => ptp(s.a, s.axis, s.@out, s.keepdims), @out);
            // NumPy's ptp is _methods._ptp: subtract(maximum.reduce(a, axis, out, keepdims), minimum.reduce(a, axis,
            // keepdims), out). The whole function, its text routes included, is the engine's: TensorEngine.Ptp. EngineOf,
            // not a.TensorEngine: the body raises the ArgumentNullException for a null a, so the dispatch must not
            // dereference it first.
            return EngineOf(a).Ptp(a, axis, @out, keepdims);
        }

        /// <summary>
        ///     Range of values (maximum - minimum) over the axes of a tuple: NumPy's
        ///     <c>np.ptp(a, axis=(…))</c>, i.e. <c>maximum.reduce(a, axis) - minimum.reduce(a, axis)</c>.
        /// </summary>
        /// <param name="a">The array.</param>
        /// <param name="axis">The reduced axes (negative values count from the end), or null for every axis. An empty array
        ///     reduces nothing (the result is <c>a - a</c>).</param>
        /// <param name="out">An optional output; the result is written into it and it is returned.</param>
        /// <param name="keepdims">Whether the reduced axes stay with length one.</param>
        /// <returns>The range, dtype preserved (integer overflow wraps as in NumPy).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="AxisError">An axis is out of range (a 0-d input has no axis for a tuple to name).</exception>
        /// <exception cref="ValueError">An axis appears twice ("duplicate value in 'axis'"); a <c>numpy.matrix</c> operand
        ///     whose result cannot be a matrix.</exception>
        /// <exception cref="TypeError">Fixed-width text, or a text out (NumPy's maximum.reduce refuses the loop).</exception>
        /// <exception cref="ArgumentException"><paramref name="out"/> has the wrong shape.</exception>
        /// <remarks>NumPy's whole function is the engine's <see cref="TensorEngine.Ptp(NDArray, int[], NDArray, bool)"/>
        ///     (the np-layer dtype plan, P1b and PE); this applies the matrix rule and calls it.</remarks>
        [NDScoped]
        public static NDArray ptp(NDArray a, int[] axis, NDArray @out = null, bool keepdims = false)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Wrapped((a, axis, @out, keepdims), static s => ptp(s.a, s.axis, s.@out, s.keepdims), @out);
            // NumPy's ptp with a tuple axis (maximum.reduce's tuple checks first) is the engine's too: TensorEngine.Ptp,
            // through EngineOf so a null a reaches the body's ArgumentNullException.
            return EngineOf(a).Ptp(a, axis, @out, keepdims);
        }
    }
}
