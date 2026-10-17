namespace NumSharp
{
    // =============================================================================
    // ndarray.max / ndarray.min: ONE overload each, NumPy's whole signature
    //     a.max(axis=None, out=None, *, keepdims=<no value>, initial=<no value>, where=<no value>)
    // (plan docs/plans/api-parameter-parity.md, items A4 and A4d). NumPy's ndarray has no other max / min method and no
    // amax / amin method at all, so NDArray has none either: the generic max<T>() / min<T>(), the dtype= forms
    // (max(int, bool, DType) / max(DType)), the positional keepdims form max(int?, bool) and the whole amax / amin family
    // are gone (the owner's rule, 2026-10-17: "one whole api per function as numpy provides, remove others").
    //
    // keepdims, initial and where are keyword-only in NumPy, so a ported call names them; their C# positions cannot change
    // a ported binding. A positional second argument is `out`, as in NumPy: a.max(1, true) binds the bool literal to out
    // and raises NumPy's TypeError "return arrays must be of ArrayType". Write a.max(1, keepdims: true).
    // =============================================================================

    public partial class NDArray
    {
        /// <summary>
        ///     Return the maximum along an axis — NumPy's <c>ndarray.max(axis=None, out=None, *, keepdims=&lt;no value&gt;,
        ///     initial=&lt;no value&gt;, where=&lt;no value&gt;)</c>, i.e. <see cref="np.max(NDArray, int?, NDArray, bool, object, NDArray)"/>
        ///     on this array. This is the only overload.
        /// </summary>
        /// <param name="axis">The axis to reduce (negative counts from the end), or <c>null</c> for every axis.</param>
        /// <param name="out">The array the result is written into and returned (any dtype; a narrowing one is lossy, as in
        ///     NumPy — see <see cref="np.max(NDArray, int?, NDArray, bool, object, NDArray)"/>). Footgun: the second
        ///     positional parameter, as in NumPy, so <c>a.max(1, true)</c> passes the bool as out and raises; write
        ///     <c>keepdims: true</c> by name.</param>
        /// <param name="keepdims">Keep the reduced axes as length-1 dimensions (keyword-only in NumPy).</param>
        /// <param name="initial">The value every output starts from, or <c>null</c>; required with <paramref name="where"/>
        ///     (keyword-only in NumPy).</param>
        /// <param name="where">A bool mask of the elements to reduce, or <c>null</c> (keyword-only in NumPy).</param>
        /// <returns><paramref name="out"/> itself when given; else a fresh array (0-d for a flat reduction without keepdims).</returns>
        /// <exception cref="TypeError"><paramref name="out"/> is a C# scalar converted to a 0-d array (as in
        ///     <c>a.max(1, true)</c>): NumPy's "return arrays must be of ArrayType"; also <paramref name="initial"/> is complex
        ///     for a real loop or of an unconvertible type.</exception>
        /// <exception cref="ValueError">A read-only or misshapen <paramref name="out"/>, <paramref name="where"/> without
        ///     <paramref name="initial"/>, an empty reduced axis without <paramref name="initial"/>, or an unconvertible
        ///     <paramref name="initial"/>.</exception>
        /// <exception cref="System.ArgumentException"><paramref name="where"/> is not a bool array.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="where"/> or <paramref name="out"/> does not broadcast.</exception>
        /// <exception cref="System.OverflowException"><paramref name="initial"/> does not fit an integer loop.</exception>
        /// <remarks>
        ///     The member keeps the plain-array meaning for a <c>numpy.matrix</c> object (an <see cref="NDMatrix"/> declares its
        ///     own <c>max</c>). https://numpy.org/doc/stable/reference/generated/numpy.ndarray.max.html
        /// </remarks>
        public NDArray max(int? axis = null, NDArray @out = null, bool keepdims = false, object initial = null, NDArray @where = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(this);   // the member keeps the plain-array meaning for a matrix object
            return np.max(this, axis, @out, keepdims, initial, @where);
        }

        /// <summary>
        ///     Return the minimum along an axis — NumPy's <c>ndarray.min(axis=None, out=None, *, keepdims=&lt;no value&gt;,
        ///     initial=&lt;no value&gt;, where=&lt;no value&gt;)</c>, i.e. <see cref="np.min(NDArray, int?, NDArray, bool, object, NDArray)"/>
        ///     on this array. This is the only overload.
        /// </summary>
        /// <param name="axis">The axis to reduce (negative counts from the end), or <c>null</c> for every axis.</param>
        /// <param name="out">The array the result is written into and returned. Footgun: the second positional parameter, as
        ///     in NumPy, so <c>a.min(1, true)</c> passes the bool as out and raises; write <c>keepdims: true</c> by name.</param>
        /// <param name="keepdims">Keep the reduced axes as length-1 dimensions (keyword-only in NumPy).</param>
        /// <param name="initial">The value every output starts from, or <c>null</c>; required with <paramref name="where"/>
        ///     (keyword-only in NumPy).</param>
        /// <param name="where">A bool mask of the elements to reduce, or <c>null</c> (keyword-only in NumPy).</param>
        /// <returns><paramref name="out"/> itself when given; else a fresh array (0-d for a flat reduction without keepdims).</returns>
        /// <exception cref="TypeError"><paramref name="out"/> is a C# scalar converted to a 0-d array (as in
        ///     <c>a.min(1, true)</c>): NumPy's "return arrays must be of ArrayType"; also <paramref name="initial"/> is complex
        ///     for a real loop or of an unconvertible type.</exception>
        /// <exception cref="ValueError">A read-only or misshapen <paramref name="out"/>, <paramref name="where"/> without
        ///     <paramref name="initial"/>, an empty reduced axis without <paramref name="initial"/>, or an unconvertible
        ///     <paramref name="initial"/>.</exception>
        /// <exception cref="System.ArgumentException"><paramref name="where"/> is not a bool array.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="where"/> or <paramref name="out"/> does not broadcast.</exception>
        /// <exception cref="System.OverflowException"><paramref name="initial"/> does not fit an integer loop.</exception>
        /// <remarks>
        ///     The member keeps the plain-array meaning for a <c>numpy.matrix</c> object (an <see cref="NDMatrix"/> declares its
        ///     own <c>min</c>). https://numpy.org/doc/stable/reference/generated/numpy.ndarray.min.html
        /// </remarks>
        public NDArray min(int? axis = null, NDArray @out = null, bool keepdims = false, object initial = null, NDArray @where = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(this);   // the member keeps the plain-array meaning for a matrix object
            return np.min(this, axis, @out, keepdims, initial, @where);
        }
    }
}
