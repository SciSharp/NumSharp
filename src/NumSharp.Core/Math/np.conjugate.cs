namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return the complex conjugate, element-wise.
        ///     The complex conjugate of a complex number is obtained by changing the sign of its
        ///     imaginary part. For real-valued input the conjugate is the value itself.
        ///     Mirrors NumPy's ufunc signature <c>conjugate(x, /, out=None, *, where=True, dtype=None)</c>.
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="out">A location into which the result is stored (must be the same shape; returned as-is).</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written (NumPy ufunc where=). Masked-off slots keep the prior contents of <paramref name="out"/>.</param>
        /// <param name="dtype">Explicit loop dtype (NumPy ufunc dtype=): selects the loop and its output dtype.</param>
        /// <returns>
        ///     The complex conjugate of <paramref name="x"/>. For real dtypes this is a copy of the input
        ///     values (dtype preserved) — EXCEPT <c>bool</c>, which NumPy has no loop for and therefore
        ///     resolves to the <c>int8</c> loop (values 0/1); for Complex the imaginary sign is flipped.
        /// </returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.conjugate.html</remarks>
        [NDScoped]
        public static NDArray conjugate(NDArray x, NDArray @out = null, NDArray where = null, NPTypeCode? dtype = null)
        {
            // NumPy's `conjugate` ufunc has NO bool loop: a plain bool input resolves to the smallest
            // matching loop — int8 — so the result is int8 (values 0/1), NOT bool (probed against
            // NumPy 2.4.2). Promote here to match. An explicit dtype= selects the loop instead, exactly
            // as NumPy does (e.g. conjugate(bool, dtype: Double) -> float64).
            if (dtype is null && x.typecode == NPTypeCode.Boolean)
                x = x.astype(NPTypeCode.SByte);
            return x.TensorEngine.Conjugate(x, dtype.AsType(), @out, where);
        }

        /// <summary>
        ///     Alias of <see cref="conjugate(NDArray, NDArray, NDArray, NPTypeCode?)"/> — return the
        ///     complex conjugate, element-wise (NumPy: <c>np.conj is np.conjugate</c>).
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="out">A location into which the result is stored (must be the same shape; returned as-is).</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written (NumPy ufunc where=).</param>
        /// <param name="dtype">Explicit loop dtype (NumPy ufunc dtype=): selects the loop and its output dtype.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.conj.html</remarks>
        [NDScoped]
        public static NDArray conj(NDArray x, NDArray @out = null, NDArray where = null, NPTypeCode? dtype = null)
            => conjugate(x, @out, where, dtype);
    }
}
