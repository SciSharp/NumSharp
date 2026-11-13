namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Permute the dimensions of an array: a view, never a copy, with the dimensions and strides reordered.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="axes">NumPy's <c>axes</c>: a permutation of <c>[0, 1, ..., a.ndim - 1]</c> (negative axes count
        ///     from the end), or <c>null</c> (NumPy's <c>None</c>, the default) to reverse the dimensions. An empty
        ///     permutation is a permutation like any other: it fits only a 0-d array.</param>
        /// <returns>a with its axes permuted. A view is returned whenever possible.
        ///     A <c>numpy.char.chararray</c> operand gives a chararray over the result (NumPy creates it as the
        ///     subclass), owning its data exactly when the plain result would.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.transpose.html
        ///     <para>NumPy's signature <c>transpose(a, axes=None)</c>, keyword for keyword: the parameter was called
        ///     <c>premute</c> until 2026-11-11, so a ported <c>np.transpose(a, axes: ...)</c> did not compile.</para>
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <exception cref="ValueError"><paramref name="axes"/> does not have one entry per dimension (<c>axes don't
        ///     match array</c>), or repeats an axis (<c>repeated axis in transpose</c>).</exception>
        /// <exception cref="AxisError">An entry of <paramref name="axes"/> is out of bounds.</exception>
        public static NDArray transpose(NDArray a, int[] axes = null)
            => a.TensorEngine.Transpose(a, axes);
    }
}