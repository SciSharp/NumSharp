using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        ///     <c>np.std</c>: NumPy's <c>_std</c> — the <c>_var</c> port (<see cref="VarStdNumPy"/>) followed by NumPy's
        ///     square root step, so <c>ddof=</c> and <c>dtype=</c> work at every size and every input gives NumPy's value,
        ///     dtype and NaN / +inf.
        /// </summary>
        /// <param name="arr">The operand (any dtype, any layout).</param>
        /// <param name="axis_">The reduced axis (negative counts from the end), or null for every axis.</param>
        /// <param name="keepdims">When true the reduced axes stay in the result with length 1.</param>
        /// <param name="ddof">Delta degrees of freedom (null is 0): the divisor is <c>max(n - ddof, 0)</c>.</param>
        /// <param name="dtype">The loop of both sums (an integer or bool input defaults to float64, every other input
        /// to its own dtype); it decides the result dtype.</param>
        /// <param name="out">NumPy's <c>out=</c>, a float or complex array of the result's shape, or null for a fresh result:
        /// <c>_std</c> runs <c>_var</c> into it and takes the square root in place (<see cref="StdIntoOut"/>).</param>
        /// <param name="where">NumPy's <c>where=</c> mask, or null (see <see cref="ReduceVar"/>).</param>
        /// <param name="mean">NumPy's <c>mean=</c>, or null to compute it (see <see cref="ReduceVar"/>).</param>
        /// <param name="correction">NumPy's <c>correction=</c>, or null (see <see cref="ReduceVar"/>).</param>
        /// <returns><paramref name="out"/> itself when given; otherwise the standard deviation: a read-only 0-d scalar when
        /// the result has no dimensions, a fresh array else.</returns>
        /// <exception cref="AxisError"><paramref name="axis_"/> is outside the operand (a 0-d operand included, without a
        /// mask).</exception>
        /// <exception cref="TypeError">A bool input with <c>dtype=bool</c> or a bool <paramref name="mean"/> of a bool input
        /// (NumPy's boolean-subtract TypeError), or a text out for a float or complex input.</exception>
        /// <exception cref="System.ArgumentException">An integer or bool <paramref name="dtype"/> with a result that has
        /// dimensions, or an integer or bool <paramref name="out"/>: NumPy's <c>sqrt</c> output-cast error (a 0-d result
        /// without out truncates the root instead); a non-bool <paramref name="where"/>.</exception>
        /// <exception cref="ValueError"><paramref name="out"/> is read-only or does not have the result's shape; a nonzero
        /// <paramref name="ddof"/> together with <paramref name="correction"/>.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="out"/>, <paramref name="where"/> or
        /// <paramref name="mean"/> does not broadcast against the input.</exception>
        /// <remarks>
        ///     The boundary scope reclaims every intermediate of the port and yields only the result (the
        ///     uninitialized-shape sentinel passes through, and <paramref name="out"/>, <paramref name="where"/> and
        ///     <paramref name="mean"/>, inputs, are never tracked).
        /// </remarks>
        [NDScoped]
        public override NDArray ReduceStd(NDArray arr, int? axis_, bool keepdims = false, int? ddof = null, DType dtype = null,
            NDArray @out = null, NDArray @where = null, NDArray mean = null, int? correction = null)
        {
            // fromnumeric's correction= rule runs before anything else; then where= / mean= take _std's keyword statements.
            int d = DdofWithCorrection(ddof, correction);
            if (IsPythonTrueMask(@where))
                @where = null;
            if (@where is not null || mean is not null)
                return VarStdKeywordsNumPy(arr, axis_, keepdims, d, dtype, @out, @where, mean, takeSqrt: true);
            return @out is null
                ? VarStdNumPy(arr, axis_, keepdims, d, dtype, takeSqrt: true)
                : StdIntoOut(arr, axis_, keepdims, d, dtype, @out);
        }

        /// <summary>
        ///     The flat standard deviation as a C# value of <typeparamref name="T"/>: <see cref="ReduceStd"/> over every
        ///     element, then a value conversion.
        /// </summary>
        /// <typeparam name="T">The C# type to convert the standard deviation to.</typeparam>
        /// <param name="arr">The operand.</param>
        /// <param name="typeCode">NumPy's <c>dtype=</c> (null: the input's own float dtype, float64 for an integer or
        /// bool input).</param>
        /// <param name="ddof">Delta degrees of freedom (null is 0).</param>
        /// <returns>The standard deviation converted to <typeparamref name="T"/>.</returns>
        public T StdElementwise<T>(NDArray arr, NPTypeCode? typeCode, int? ddof) where T : unmanaged
        {
            return Converts.ChangeType<T>(std_elementwise(arr, typeCode, ddof));
        }

        /// <summary>
        ///     The flat standard deviation as a boxed scalar of the result dtype (<see cref="ReduceStd"/> over every
        ///     element).
        /// </summary>
        /// <param name="arr">The operand.</param>
        /// <param name="typeCode">NumPy's <c>dtype=</c>, or null.</param>
        /// <param name="ddof">Delta degrees of freedom (null is 0).</param>
        /// <returns>The standard deviation, boxed in its result dtype's C# type.</returns>
        protected object std_elementwise(NDArray arr, NPTypeCode? typeCode, int? ddof)
        {
            var r = ReduceStd(arr, null, false, ddof, typeCode);
            var value = r.GetAtIndex(0);
            // The 0-d result is this call's own allocation; the uninitialized-shape sentinel comes back as arr itself.
            if (!ReferenceEquals(r, arr))
                r.Dispose();
            return value;
        }
    }
}
