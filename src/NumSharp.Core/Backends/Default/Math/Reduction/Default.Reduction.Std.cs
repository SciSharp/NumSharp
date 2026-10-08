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
        /// <returns>The standard deviation: a read-only 0-d scalar when the result has no dimensions, otherwise a fresh
        /// array.</returns>
        /// <exception cref="AxisError"><paramref name="axis_"/> is outside the operand (a 0-d operand included).</exception>
        /// <exception cref="TypeError">A bool input with <c>dtype=bool</c> (NumPy's boolean-subtract TypeError).</exception>
        /// <exception cref="System.ArgumentException">An integer or bool <paramref name="dtype"/> with a result that has
        /// dimensions: NumPy's <c>sqrt</c> output-cast error (a 0-d result truncates the root instead).</exception>
        /// <remarks>
        ///     The boundary scope reclaims every intermediate of the port and yields only the result (the
        ///     uninitialized-shape sentinel passes through).
        /// </remarks>
        [NDScoped]
        public override NDArray ReduceStd(NDArray arr, int? axis_, bool keepdims = false, int? ddof = null, DType dtype = null)
            => VarStdNumPy(arr, axis_, keepdims, ddof ?? 0, dtype, takeSqrt: true);

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
