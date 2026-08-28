namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Test element-wise for positive infinity, return result as a bool array.
        /// </summary>
        /// <param name="x">The input array.</param>
        /// <param name="out">
        ///     A location into which the result is stored (NumPy's positional <c>out</c>). If provided it
        ///     must have a shape the input broadcasts to; if its dtype is numeric the result is stored as
        ///     0/1, if boolean as False/True. The same instance is returned. If <c>null</c>, a freshly
        ///     allocated boolean array is returned.
        /// </param>
        /// <returns>A boolean array (or <paramref name="out"/>), True where the element is +Inf.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.isposinf.html
        ///     <para>
        ///     Port of NumPy's <c>numpy.isposinf</c> (<c>numpy/lib/_ufunclike_impl.py</c>), defined as
        ///     <c>logical_and(isinf(x), ~signbit(x))</c> — i.e. exactly <c>x == +inf</c>. Integer and boolean
        ///     inputs return all-False (they cannot be infinite). A <b>complex</b> input raises
        ///     <see cref="TypeError"/> ("This operation is not supported for complex128 values because it would
        ///     be ambiguous.") — NumPy's <c>signbit</c> is ambiguous on complex, so the whole function is —
        ///     matching NumPy 2.4.2.
        ///     </para>
        /// </remarks>
        public static NDArray isposinf(NDArray x, NDArray @out = null)
        {
            if (x.typecode == NPTypeCode.Complex)
                throw new TypeError(
                    "This operation is not supported for complex128 values because it would be ambiguous.");

            return x.TensorEngine.IsPosInf(x, NPTypeCode.Boolean, @out, null);
        }
    }
}
