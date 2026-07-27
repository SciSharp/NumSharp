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
        ///     NumPy offers neither <c>axis</c> nor <c>keepdims</c> here and raises <c>TypeError</c>
        ///     for both, because the signature's core dimensions are two DISTINCT ones; the
        ///     parameters are therefore absent rather than throwing.
        ///     </para>
        /// </remarks>
        public static NDArray matvec(NDArray x1, NDArray x2, NDArray @out = null, NPTypeCode? dtype = null)
        {
            GufuncGuard.RequireRank("matvec", MatvecSignature, 0, x1, 2);
            GufuncGuard.RequireRank("matvec", MatvecSignature, 1, x2, 1);
            GufuncGuard.RequireCoreSize("matvec", MatvecSignature, 1, 0,
                x2.shape[x2.ndim - 1], x1.shape[x1.ndim - 1]);

            var a = GufuncGuard.ToLoop(x1, dtype);
            var b = GufuncGuard.ToLoop(x2, dtype);

            return GufuncGuard.Deliver(a.TensorEngine.Matvec(a, b), @out);
        }
    }
}
