namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Vector-matrix product (NumPy 2.2) — the gufunc <c>(n),(n,m)->(m)</c>, conjugating the
        ///     vector operand.
        /// </summary>
        /// <param name="x1">Vector operand, at least 1-D. Conjugated when complex.</param>
        /// <param name="x2">Matrix operand, at least 2-D. Leading axes broadcast.</param>
        /// <param name="out">Where to deposit the answer. Returned as-is when given.</param>
        /// <param name="axes">
        ///     Which axes carry the core dimensions, per operand: <c>{(n), (n,m), (m)}</c>. All
        ///     THREE entries are required — the output has a core axis.
        /// </param>
        /// <param name="axis">
        ///     Present for signature parity only. NumPy raises <c>TypeError</c> for any value —
        ///     this signature's core dimensions are two DISTINCT ones; use <paramref name="axes"/>.
        /// </param>
        /// <param name="keepdims">
        ///     Present for signature parity only. NumPy raises <c>TypeError</c> when true.
        /// </param>
        /// <param name="dtype">
        ///     Selects the LOOP: computation runs at this dtype, not merely the result.
        /// </param>
        /// <param name="casting">NumPy's <c>casting=</c>: the rule each input's cast to the loop, and the answer's cast
        ///     into <paramref name="out"/>, is checked under - <c>no</c>, <c>equiv</c>, <c>safe</c>, <c>same_kind</c>
        ///     (the default) or <c>unsafe</c>.</param>
        /// <param name="order">NumPy's <c>order=</c>, the layout of an allocated result: <c>'K'</c> (the default: the
        ///     loop axes in the inputs' memory order), <c>'C'</c>, <c>'F'</c>, or <c>'A'</c> (F when both inputs are
        ///     F-contiguous, else C); lowercase is accepted, and it is ignored with <paramref name="out"/>.</param>
        /// <returns>The vector-matrix products (the operands' loop axes broadcast together), or <paramref name="out"/>
        ///     itself holding them.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.vecmat.html
        ///     <para>
        ///     The conjugation is what makes this the adjoint of <see cref="matvec"/> rather than its
        ///     mirror image: <c>vecmat(v, m)</c> is <c>conj(v) @ m</c>, so for real operands it is the
        ///     plain row-vector product and for complex ones it agrees with <see cref="vecdot"/>'s
        ///     convention.
        ///     </para>
        ///     <para>
        ///     NumPy's <c>subok</c> and <c>signature</c> keywords are absent: <c>subok</c> concerns ndarray
        ///     subclasses, which the numpy.matrix boundary already decides, and <c>signature</c> is what
        ///     <paramref name="dtype"/> does.
        ///     </para>
        ///     <para>
        ///     The whole call is the engine member <c>TensorEngine.Vecmat(x1, x2, out, axes, axis, keepdims, dtype, casting, order)</c>
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
        /// <exception cref="System.ArgumentException">An input cannot be cast to the loop, or the answer into
        ///     <paramref name="out"/>, under <paramref name="casting"/> (NumPy's UFuncTypeError text: <c>Cannot cast ufunc
        ///     'vecmat' input 0 from dtype('int32') to dtype('float64') with casting rule 'no'</c>).</exception>
        /// <exception cref="ValueError">A <paramref name="casting"/> or <paramref name="order"/> name NumPy
        ///     refuses (casting first).</exception>
        /// <exception cref="IncorrectShapeException">Loop axes that do not broadcast.</exception>
        /// <exception cref="System.NullReferenceException"><paramref name="x1"/> or <paramref name="x2"/> is null.</exception>
        [NDScoped]
        public static NDArray vecmat(NDArray x1, NDArray x2, NDArray @out = null, int[][] axes = null,
            int? axis = null, bool keepdims = false, DType dtype = null, string casting = "same_kind",
            char order = 'K')
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(x1, x2))
                return NDMatrixBoundary.Wrapped((x1, x2, @out, axes, axis, keepdims, dtype, casting, order), static s => vecmat(s.x1, s.x2, s.@out, s.axes, s.axis, s.keepdims, s.dtype, s.casting, s.order), @out);
            return x1.TensorEngine.Vecmat(x1, x2, @out, axes, axis, keepdims, dtype, casting, order);
        }
    }
}
