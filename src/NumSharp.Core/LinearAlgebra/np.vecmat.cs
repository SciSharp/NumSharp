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
        ///     As with <see cref="matvec"/>, NumPy rejects <c>axis</c> and <c>keepdims</c> here
        ///     (two distinct core dimensions), so neither parameter is offered.
        ///     </para>
        /// </remarks>
        public static NDArray vecmat(NDArray x1, NDArray x2, NDArray @out = null, NPTypeCode? dtype = null)
        {
            GufuncGuard.RequireRank("vecmat", VecmatSignature, 0, x1, 1);
            GufuncGuard.RequireRank("vecmat", VecmatSignature, 1, x2, 2);
            GufuncGuard.RequireCoreSize("vecmat", VecmatSignature, 1, 0,
                x2.shape[x2.ndim - 2], x1.shape[x1.ndim - 1]);

            var a = GufuncGuard.ToLoop(x1, dtype);
            var b = GufuncGuard.ToLoop(x2, dtype);

            return GufuncGuard.Deliver(a.TensorEngine.Vecmat(a, b), @out);
        }
    }
}
