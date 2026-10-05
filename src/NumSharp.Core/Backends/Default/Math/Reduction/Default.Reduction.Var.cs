using System.Runtime.CompilerServices;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        ///     <c>np.var</c>: NumPy's <c>_var</c> statement for statement (<see cref="VarStdNumPy"/>), so <c>ddof=</c> and
        ///     <c>dtype=</c> work at every size and every input gives NumPy's value, dtype and NaN / +inf.
        /// </summary>
        /// <param name="arr">The operand (any dtype, any layout).</param>
        /// <param name="axis_">The reduced axis (negative counts from the end), or null for every axis.</param>
        /// <param name="keepdims">When true the reduced axes stay in the result with length 1.</param>
        /// <param name="ddof">Delta degrees of freedom (null is 0): the divisor is <c>max(n - ddof, 0)</c>, so
        /// <c>ddof &gt;= n</c> gives +inf (NaN when every deviation is 0), never a negative variance.</param>
        /// <param name="dtype">The loop of both sums (an integer or bool input defaults to float64, every other input
        /// to its own dtype); it decides the result dtype, and an integer dtype truncates the mean as in NumPy.</param>
        /// <returns>The variance: a read-only 0-d scalar when the result has no dimensions, otherwise a fresh array.</returns>
        /// <exception cref="AxisError"><paramref name="axis_"/> is outside the operand (a 0-d operand included).</exception>
        /// <exception cref="System.NotSupportedException">A bool input with <c>dtype=bool</c> (NumPy's boolean-subtract
        /// TypeError).</exception>
        /// <remarks>
        ///     The boundary scope reclaims every intermediate of the port — the keepdims mean, the squared deviations and
        ///     the quotient temporaries — and yields only the result (the uninitialized-shape sentinel passes through).
        /// </remarks>
        [NDScoped]
        public override NDArray ReduceVar(NDArray arr, int? axis_, bool keepdims = false, int? ddof = null, DType dtype = null)
            => VarStdNumPy(arr, axis_, keepdims, ddof ?? 0, dtype, takeSqrt: false);

        /// <summary>
        ///     The flat variance as a C# value of <typeparamref name="T"/>: <see cref="ReduceVar"/> over every element,
        ///     then a value conversion.
        /// </summary>
        /// <typeparam name="T">The C# type to convert the variance to.</typeparam>
        /// <param name="arr">The operand.</param>
        /// <param name="typeCode">NumPy's <c>dtype=</c> (null: the input's own float dtype, float64 for an integer or
        /// bool input).</param>
        /// <param name="ddof">Delta degrees of freedom (null is 0).</param>
        /// <returns>The variance converted to <typeparamref name="T"/>.</returns>
        public T VarElementwise<T>(NDArray arr, NPTypeCode? typeCode, int? ddof) where T : unmanaged
        {
            return (T)Converts.ChangeType(var_elementwise(arr, typeCode, ddof), InfoOf<T>.NPTypeCode);
        }

        /// <summary>
        ///     The flat variance as a boxed scalar of the result dtype (<see cref="ReduceVar"/> over every element).
        /// </summary>
        /// <param name="arr">The operand.</param>
        /// <param name="typeCode">NumPy's <c>dtype=</c>, or null.</param>
        /// <param name="ddof">Delta degrees of freedom (null is 0).</param>
        /// <returns>The variance, boxed in its result dtype's C# type (a float for a float32 input, a double for an
        /// integer, bool or complex128 input).</returns>
        protected object var_elementwise(NDArray arr, NPTypeCode? typeCode, int? ddof)
        {
            var r = ReduceVar(arr, null, false, ddof, typeCode);
            var value = r.GetAtIndex(0);
            // The 0-d result is this call's own allocation; the uninitialized-shape sentinel comes back as arr itself.
            if (!ReferenceEquals(r, arr))
                r.Dispose();
            return value;
        }

        /// <summary>
        /// C-order visitation offset (in elements) for the <paramref name="linear"/>-th element of
        /// a strided view. Decodes coordinates last-axis-fastest and folds them through the strides.
        /// </summary>
        /// <param name="linear">The element's position in logical C order.</param>
        /// <param name="dims">The view's dimensions.</param>
        /// <param name="strides">The view's strides, in elements.</param>
        /// <param name="ndim">The number of dimensions.</param>
        /// <returns>The element's offset from the view's first element, in elements.</returns>
        /// <remarks>Used by the integer reciprocal loops (<c>Default.Reciprocal.cs</c>); var / std no longer read
        /// elements themselves.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static long FlatStrideOffset(long linear, long[] dims, long[] strides, int ndim)
        {
            long off = 0;
            for (int d = ndim - 1; d >= 0; d--)
            {
                long dim = dims[d];
                off += (linear % dim) * strides[d];
                linear /= dim;
            }
            return off;
        }
    }
}
