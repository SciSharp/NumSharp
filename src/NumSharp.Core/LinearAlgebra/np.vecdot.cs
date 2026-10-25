namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Vector dot product of two arrays (NumPy 2.0) — the gufunc <c>(n),(n)->()</c>,
        ///     conjugating the first operand.
        /// </summary>
        /// <param name="x1">First operand. Conjugated when complex.</param>
        /// <param name="x2">Second operand.</param>
        /// <param name="out">Where to deposit the answer. Returned as-is when given.</param>
        /// <param name="axes">
        ///     Which axes carry the core dimensions, per operand: <c>{x1, x2, out}</c>. The output
        ///     entry may be omitted here — <c>vecdot</c>'s output has no core axes. Cannot be
        ///     combined with <paramref name="axis"/>.
        /// </param>
        /// <param name="axis">
        ///     The shared core axis, in place of the default last one. Applied to both operands —
        ///     the special case of <paramref name="axes"/> that this signature admits because both
        ///     core dimensions are the SAME one.
        /// </param>
        /// <param name="keepdims">Leave the contracted axis in the result with length 1.</param>
        /// <param name="dtype">
        ///     Selects the LOOP: computation runs at this dtype, not merely the result.
        /// </param>
        /// <returns>The vector dot products (the operands' loop axes broadcast together), or <paramref name="out"/>
        ///     itself holding them.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.vecdot.html
        ///     <para>
        ///     Unlike <see cref="sum"/>, the accumulation happens in the LOOP dtype rather than
        ///     NEP50's wider accumulator: <c>vecdot</c>'s registered loops are <c>'ii->i'</c>, so an
        ///     int32 pair reduces to int32 where <c>np.sum</c> would give int64.
        ///     </para>
        ///     <para>
        ///     Core dimensions do NOT broadcast — a length-1 core axis against a length-3 one is an
        ///     error. Only the leading (loop) axes broadcast.
        ///     </para>
        ///     <para>
        ///     NumPy's remaining ufunc keywords — <c>casting</c>, <c>order</c>, <c>subok</c> and
        ///     <c>signature</c> — are not modelled anywhere in NumSharp's ufunc surface and so are
        ///     absent here too rather than accepted and ignored. <c>signature</c> is what
        ///     <paramref name="dtype"/> already does; <c>subok</c> concerns ndarray subclasses,
        ///     which NumSharp does not have.
        ///     </para>
        ///     <para>
        ///     The whole call is the engine member <c>TensorEngine.Vecdot(x1, x2, out, axes, axis, keepdims, dtype)</c>
        ///     since the np-layer dtype plan's P2; this function keeps the numpy.matrix boundary.
        ///     </para>
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes
        ///     longer than 1 (<c>shape too large to be a matrix.</c>); a 0-d operand; core axes of different lengths; an
        ///     <paramref name="axes"/> list of the wrong length; a read-only <paramref name="out"/>, or one the answer does
        ///     not broadcast into.</exception>
        /// <exception cref="TypeError"><paramref name="axis"/> with <paramref name="axes"/>; a <paramref name="dtype"/>
        ///     with details beyond its class; text operands or a text <paramref name="dtype"/> (no vecdot loop).</exception>
        /// <exception cref="AxisError">An <paramref name="axes"/> entry of the wrong length, or an axis out of
        ///     range.</exception>
        /// <exception cref="System.InvalidCastException">The answer cannot be cast into <paramref name="out"/> under
        ///     same_kind.</exception>
        /// <exception cref="System.NullReferenceException"><paramref name="x1"/> or <paramref name="x2"/> is null.</exception>
        [NDScoped]
        public static NDArray vecdot(NDArray x1, NDArray x2, NDArray @out = null, int[][] axes = null,
            int? axis = null, bool keepdims = false, DType dtype = null)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(x1, x2))
                return NDMatrixBoundary.Wrapped((x1, x2, @out, axes, axis, keepdims, dtype), static s => vecdot(s.x1, s.x2, s.@out, s.axes, s.axis, s.keepdims, s.dtype), @out);
            return x1.TensorEngine.Vecdot(x1, x2, @out, axes, axis, keepdims, dtype);
        }
    }
}
