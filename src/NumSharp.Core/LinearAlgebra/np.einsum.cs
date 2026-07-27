namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Evaluates the Einstein summation convention on the operands.
        /// </summary>
        /// <param name="subscripts">
        ///     The comma-separated subscript labels, optionally followed by <c>-&gt;</c> and the
        ///     output labels — e.g. <c>"ij,jk-&gt;ik"</c>. Ellipsis (<c>...</c>) is allowed.
        /// </param>
        /// <param name="operands">The arrays the subscripts label, in order.</param>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.einsum.html
        ///     <para>
        ///     <b>Not implemented — this is the signature only.</b> Calling it raises
        ///     <see cref="System.NotSupportedException"/> from the engine. It is declared now so the
        ///     name and call shape are settled while the implementation is pending.
        ///     </para>
        ///     <para>
        ///     <c>einsum</c> reaches this file because <c>optimize=</c> routes its pairwise
        ///     contractions through <see cref="tensordot"/> and <see cref="dot"/>, and so through
        ///     BLAS. The rest of it is NumSharp's own work rather than a backend's: a subscript
        ///     parser, repeated-index diagonals, ellipsis broadcasting, output-order inference, and
        ///     the contraction-path planner — which is why the seam has no <c>TryEinsum</c> and the
        ///     engine method throws directly.
        ///     </para>
        ///     <para>
        ///     NumPy's alternative "sublist" spelling
        ///     (<c>einsum(op0, [0,1], op1, [1,2], [0,2])</c>) is not offered.
        ///     </para>
        /// </remarks>
        public static NDArray einsum(string subscripts, params NDArray[] operands)
            => einsum(subscripts, operands, null);

        /// <summary>
        ///     Evaluates the Einstein summation convention, with an explicit output array and
        ///     contraction-path option.
        /// </summary>
        /// <param name="out">Where to deposit the answer.</param>
        /// <param name="optimize">
        ///     Whether to plan a pairwise contraction order rather than contracting in one pass.
        /// </param>
        /// <param name="dtype">Selects the accumulation dtype.</param>
        /// <inheritdoc cref="einsum(string, NDArray[])"/>
        public static NDArray einsum(string subscripts, NDArray[] operands, NDArray @out,
            bool optimize = false, NPTypeCode? dtype = null)
        {
            if (operands is null || operands.Length == 0)
                throw new ValueError("must specify the einstein sum subscripts string and at least one operand, "
                                     + "or at least one operand and its corresponding subscripts list");

            return operands[0].TensorEngine.Einsum(subscripts, operands, @out, optimize, dtype);
        }
    }
}
