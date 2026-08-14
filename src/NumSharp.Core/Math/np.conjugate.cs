namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return the complex conjugate, element-wise.
        ///     The complex conjugate of a complex number is obtained by changing the sign of its
        ///     imaginary part. For real-valued input the conjugate is the value itself.
        ///     Mirrors NumPy's ufunc signature <c>conjugate(x, /, out=None, *, where=True, ...)</c>.
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="out">A location into which the result is stored (must be the same shape; returned as-is).</param>
        /// <returns>
        ///     The complex conjugate of <paramref name="x"/>, with the same shape and dtype.
        ///     For real dtypes this is a copy of the input values; for Complex the imaginary sign is flipped.
        /// </returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.conjugate.html</remarks>
        public static NDArray conjugate(NDArray x, NDArray @out = null)
            => x.TensorEngine.Conjugate(x, null, @out, null);

        /// <summary>
        ///     Alias of <see cref="conjugate(NDArray, NDArray)"/> — return the complex conjugate, element-wise.
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="out">A location into which the result is stored (must be the same shape; returned as-is).</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.conj.html</remarks>
        public static NDArray conj(NDArray x, NDArray @out = null)
            => x.TensorEngine.Conjugate(x, null, @out, null);
    }
}
