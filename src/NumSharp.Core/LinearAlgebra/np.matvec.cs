namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Matrix-vector product (NumPy 2.2) — the gufunc <c>(m,n),(n)->(m)</c>, NumPy's
        ///     <c>gemv</c> route.
        /// </summary>
        /// <param name="x1">Matrix operand, at least 2-D. Leading axes broadcast.</param>
        /// <param name="x2">Vector operand, at least 1-D. NOT conjugated.</param>
        /// <param name="out">Where to deposit the answer. Returned as-is when given.</param>
        /// <param name="axes">
        ///     Which axes carry the core dimensions, per operand: <c>{(m,n), (n), (m)}</c>. All
        ///     THREE entries are required — the output has a core axis, so its entry cannot be
        ///     omitted the way <see cref="vecdot"/>'s can.
        /// </param>
        /// <param name="axis">
        ///     Present for signature parity only. NumPy raises <c>TypeError</c> for any value,
        ///     because this signature's core dimensions are two DISTINCT ones — use
        ///     <paramref name="axes"/>.
        /// </param>
        /// <param name="keepdims">
        ///     Present for signature parity only. NumPy raises <c>TypeError</c> when true, for the
        ///     same reason.
        /// </param>
        /// <param name="dtype">
        ///     Selects the LOOP: computation runs at this dtype, not merely the result.
        /// </param>
        /// <returns>The matrix-vector products (the operands' loop axes broadcast together), or <paramref name="out"/>
        ///     itself holding them.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.matvec.html
        ///     <para>
        ///     This differs from <c>matmul(x1, x2)</c> with a 1-D <c>x2</c> in how it BROADCASTS:
        ///     <c>matmul</c> treats a 1-D second operand as a one-off shape promotion, while
        ///     <c>matvec</c> is a true gufunc whose vector operand carries its own leading axes.
        ///     Unlike <see cref="vecmat"/> it does not conjugate — the transformation, not the inner
        ///     product, is what this computes.
        ///     </para>
        ///     <para>
        ///     NumPy's remaining ufunc keywords (<c>casting</c>, <c>order</c>, <c>subok</c>,
        ///     <c>signature</c>) are not modelled anywhere in NumSharp's ufunc surface and are
        ///     absent here too. See <see cref="vecdot"/>.
        ///     </para>
        ///     <para>
        ///     The whole call is the engine member <c>TensorEngine.Matvec(x1, x2, out, axes, axis, keepdims, dtype)</c>
        ///     since the np-layer dtype plan's P2; this function keeps the numpy.matrix boundary.
        ///     </para>
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes
        ///     longer than 1 (<c>shape too large to be a matrix.</c>); an operand with too few dimensions; core axes of
        ///     different lengths; an <paramref name="axes"/> list without three entries; loop axes that do not broadcast
        ///     together, or a read-only <paramref name="out"/> or one the answer does not broadcast into (NumPy's gufunc
        ///     iterator texts).</exception>
        /// <exception cref="TypeError"><paramref name="axis"/> with <paramref name="axes"/>, or either given; a
        ///     <paramref name="dtype"/> with details beyond its class; text operands or a text <paramref name="dtype"/>;
        ///     <paramref name="keepdims"/>.</exception>
        /// <exception cref="AxisError">An <paramref name="axes"/> entry of the wrong length, or an axis out of
        ///     range.</exception>
        /// <exception cref="System.InvalidCastException">The answer cannot be cast into <paramref name="out"/> under
        ///     same_kind.</exception>
        /// <exception cref="IncorrectShapeException">Loop axes that do not broadcast.</exception>
        /// <exception cref="System.NullReferenceException"><paramref name="x1"/> or <paramref name="x2"/> is null.</exception>
        [NDScoped]
        public static NDArray matvec(NDArray x1, NDArray x2, NDArray @out = null, int[][] axes = null,
            int? axis = null, bool keepdims = false, DType dtype = null)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(x1, x2))
                return NDMatrixBoundary.Wrapped((x1, x2, @out, axes, axis, keepdims, dtype), static s => matvec(s.x1, s.x2, s.@out, s.axes, s.axis, s.keepdims, s.dtype), @out);
            return x1.TensorEngine.Matvec(x1, x2, @out, axes, axis, keepdims, dtype);
        }
    }
}
