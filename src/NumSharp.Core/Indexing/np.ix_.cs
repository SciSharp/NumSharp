namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Construct an open mesh from multiple sequences.
        /// </summary>
        /// <param name="args">
        ///     N 1-D sequences of integer or boolean type. A boolean sequence is interpreted as a mask for
        ///     the corresponding dimension (equivalent to passing <see cref="nonzero"/> of it). Accepts
        ///     anything <see cref="asanyarray"/> understands — <see cref="NDArray"/>, C# arrays, collections,
        ///     tuples.
        /// </param>
        /// <returns>
        ///     N arrays with N dimensions each, shape 1 in every axis but the k-th (which carries the k-th
        ///     sequence). Together they form an open mesh: <c>a[np.ix_(rows, cols)]</c> selects the cross
        ///     product <c>a[rows][:, cols]</c>.
        /// </returns>
        /// <remarks>
        ///     Port of NumPy 2.x <c>numpy.ix_</c> (<c>numpy/lib/_index_tricks_impl.py</c>). The reshape is a
        ///     VIEW when the source permits one, so the outputs share memory with an <see cref="NDArray"/>
        ///     input (NumPy does the same — <c>shares_memory</c> is True there too) and remain writeable.
        ///     <para>
        ///     The dtype is PRESERVED, not forced to <c>intp</c>: <c>ix_</c> performs no integer validation,
        ///     so a float or byte sequence comes back as float/byte and only fails later, at the indexing
        ///     call. The single exception is NumPy's: a non-ndarray input that turns out EMPTY is cast to
        ///     <c>intp</c> (int64) to avoid the float64 default of an untyped empty list.
        ///     </para>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.ix_.html
        /// </remarks>
        /// <exception cref="ValueError">A sequence is not 1-D ("Cross index must be 1 dimensional").</exception>
        [NDScoped]
        public static NDArray[] ix_(params object[] args)
            // NumPy's numpy.ix_: the statements are the array object's (NDArray.IndexTricks.cs)
            => NDArray.Ix(args);
    }
}
