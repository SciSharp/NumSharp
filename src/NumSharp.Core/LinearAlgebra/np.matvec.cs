namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     The gufunc signature <c>np.matvec</c> matches its core dimensions against.
        /// </summary>
        internal const string MatvecSignature = "(m,n),(n)->(m)";

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
        /// </remarks>
        public static NDArray matvec(NDArray x1, NDArray x2, NDArray @out = null, int[][] axes = null,
            int? axis = null, bool keepdims = false, NPTypeCode? dtype = null)
        {
            GufuncGuard.RejectAxisWithAxes(axes, axis);
            GufuncGuard.RejectAxisAndKeepdims("matvec", MatvecSignature, axis, keepdims, 1);

            GufuncGuard.RequireRank("matvec", MatvecSignature, 0, x1, 2);
            GufuncGuard.RequireRank("matvec", MatvecSignature, 1, x2, 1);

            var a = x1;
            var b = x2;
            int[] outputAxes = System.Array.Empty<int>();

            if (axes is not null)
            {
                var resolved = GufuncGuard.NormalizeAxes("matvec", axes, new[] {2, 1, 1}, x1, x2);
                outputAxes = resolved[2];

                // Bring (m,n) and (n) to the trailing positions the kernel expects; the result's
                // own core axis goes back to where `axes` asked for it once the loop has run.
                a = moveaxis(a, resolved[0], new[] {a.ndim - 2, a.ndim - 1});
                b = moveaxis(b, resolved[1][0], -1);
            }

            GufuncGuard.RequireCoreSize("matvec", MatvecSignature, 1, 0,
                b.shape[b.ndim - 1], a.shape[a.ndim - 1]);

            a = GufuncGuard.ToLoop(a, dtype);
            b = GufuncGuard.ToLoop(b, dtype);

            var result = a.TensorEngine.Matvec(a, b);

            if (outputAxes.Length == 1)
                result = moveaxis(result, -1, outputAxes[0]);

            return GufuncGuard.Deliver(result, @out);
        }
    }
}
