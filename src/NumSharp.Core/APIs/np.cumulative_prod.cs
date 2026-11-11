namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return the cumulative product of the elements along a given axis (NumPy 2.1's Array-API alternative to
        ///     <see cref="cumprod"/>): never flattens an N-D input, and can prepend the product's identity, one.
        /// </summary>
        /// <param name="x">Input array (NumPy's positional-only <c>x</c>); a 0-d input is read as 1-D (NumPy's
        ///     <c>atleast_1d</c>).</param>
        /// <param name="axis">Axis along which the cumulative product is computed (negative counts from the end).
        ///     <c>null</c> (NumPy's <c>None</c>, the default) is axis 0 for a 1-D input and an error for an input of two or
        ///     more dimensions: unlike <see cref="cumprod"/>, this function never flattens.</param>
        /// <param name="dtype">Type of the returned array and of the accumulator; <c>null</c> follows
        ///     <see cref="cumprod"/>'s rule (an integer narrower than int64 multiplies as int64, uint8 as uint64, bool as
        ///     int64).</param>
        /// <param name="out">Alternate output array; with <paramref name="include_initial"/> it must be one longer along
        ///     <paramref name="axis"/> than the input. It is returned.</param>
        /// <param name="include_initial">Prepend the identity (one) along <paramref name="axis"/>, so the result is one
        ///     longer there: <c>[1, x0, x0 * x1, ...]</c>.</param>
        /// <returns>The cumulative products, a new array unless <paramref name="out"/> was given (then
        ///     <paramref name="out"/> itself).</returns>
        /// <exception cref="System.ArgumentNullException"><paramref name="x"/> is null.</exception>
        /// <exception cref="ValueError"><paramref name="axis"/> is <c>null</c> and the input has two or more dimensions
        ///     ("For arrays which have more than one dimension ``axis`` argument is required."), or
        ///     <paramref name="out"/>'s shape does not fit (NumPy's iterator text).</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds for the (at least 1-D) input.</exception>
        /// <exception cref="IndexError"><paramref name="axis"/> is out of bounds while <paramref name="out"/> and
        ///     <paramref name="include_initial"/> are both given ("list assignment index out of range").</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.cumulative_prod.html
        ///     <para>NumPy's signature <c>cumulative_prod(x, /, *, axis=None, dtype=None, out=None, include_initial=False)</c>;
        ///     the body is NumPy's <c>_cumulative_func</c>, shared with <see cref="cumulative_sum"/>
        ///     (<see cref="CumulativeFunc"/>, in np.cumulative_sum.cs).</para>
        /// </remarks>
        public static NDArray cumulative_prod(NDArray x, int? axis = null, DType dtype = null, NDArray @out = null,
            bool include_initial = false)
            => CumulativeFunc(x, add: false, axis, dtype, @out, include_initial);
    }
}
