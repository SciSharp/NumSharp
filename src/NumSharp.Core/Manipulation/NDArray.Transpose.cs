namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Permute the dimensions of an array (NumPy's <c>a.transpose(*axes)</c>): a view with the dimensions and
        ///     strides reordered.
        /// </summary>
        /// <param name="axes">NumPy's <c>*axes</c>: the permutation as separate integers (<c>a.transpose(1, 0)</c>) or
        ///     as one array (NumPy's tuple, <c>a.transpose(new[] {1, 0})</c>); no argument, an empty array or <c>null</c>
        ///     reverses the dimensions.</param>
        /// <returns>a with its axes permuted. A view is returned whenever possible.</returns>
        /// <exception cref="ValueError"><paramref name="axes"/> does not have one entry per dimension (<c>axes don't
        ///     match array</c>), or repeats an axis (<c>repeated axis in transpose</c>).</exception>
        /// <exception cref="AxisError">An entry of <paramref name="axes"/> is out of bounds.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.ndarray.transpose.html
        ///     <para>One member with NumPy's call shapes: <c>params</c> makes <c>a.transpose(2, 0, 1)</c> compile, which
        ///     the former <c>transpose(int[] premute = null)</c> refused (CS1501). The empty case is a C# limit: a call
        ///     without arguments arrives as an EMPTY array, so an empty array reads as NumPy's no-argument call
        ///     (reverse) - NumPy's <c>a.transpose(())</c>, an explicit empty tuple, raises "axes don't match array" for
        ///     ndim &gt; 0, and C# cannot tell the two apart. <see cref="np.transpose(NDArray, int[])"/> keeps NumPy's
        ///     reading of an empty permutation.</para>
        /// </remarks>
        [PlainReceiver]
        public NDArray transpose(params int[] axes)
        {
            // No argument (C# passes an empty array) is NumPy's a.transpose() = axes None; null is None too.
            return np.transpose(this, axes is { Length: > 0 } ? axes : null);
        }
    }
}
