namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     The gufunc signature <c>np.vecmat</c> matches its core dimensions against.
        /// </summary>
        internal const string VecmatSignature = "(n),(n,m)->(m)";

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
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.vecmat.html
        ///     <para>
        ///     The conjugation is what makes this the adjoint of <see cref="matvec"/> rather than its
        ///     mirror image: <c>vecmat(v, m)</c> is <c>conj(v) @ m</c>, so for real operands it is the
        ///     plain row-vector product and for complex ones it agrees with <see cref="vecdot"/>'s
        ///     convention.
        ///     </para>
        ///     <para>
        ///     NumPy's remaining ufunc keywords (<c>casting</c>, <c>order</c>, <c>subok</c>,
        ///     <c>signature</c>) are not modelled anywhere in NumSharp's ufunc surface and are
        ///     absent here too. See <see cref="vecdot"/>.
        ///     </para>
        /// </remarks>
        [NDScoped]
        public static NDArray vecmat(NDArray x1, NDArray x2, NDArray @out = null, int[][] axes = null,
            int? axis = null, bool keepdims = false, NPTypeCode? dtype = null)
        {
            GufuncGuard.RejectAxisWithAxes(axes, axis);
            GufuncGuard.RejectAxisAndKeepdims("vecmat", VecmatSignature, axis, keepdims, 2);

            GufuncGuard.RequireRank("vecmat", VecmatSignature, 0, x1, 1);
            GufuncGuard.RequireRank("vecmat", VecmatSignature, 1, x2, 2);

            var a = x1;
            var b = x2;
            int[] outputAxes = System.Array.Empty<int>();

            if (axes is not null)
            {
                var resolved = GufuncGuard.NormalizeAxes("vecmat", axes, new[] {1, 2, 1}, x1, x2);
                outputAxes = resolved[2];

                a = moveaxis(a, resolved[0][0], -1);
                b = moveaxis(b, resolved[1], new[] {b.ndim - 2, b.ndim - 1});
            }

            GufuncGuard.RequireCoreSize("vecmat", VecmatSignature, 1, 0,
                b.shape[b.ndim - 2], a.shape[a.ndim - 1]);

            a = GufuncGuard.ToLoop(a, dtype);
            b = GufuncGuard.ToLoop(b, dtype);

            var result = a.TensorEngine.Vecmat(a, b);

            if (outputAxes.Length == 1)
                result = moveaxis(result, -1, outputAxes[0]);

            return GufuncGuard.Deliver(result, @out);
        }
    }
}
