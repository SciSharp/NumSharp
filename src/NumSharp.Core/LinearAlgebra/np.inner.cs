namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Inner product of two arrays — a sum product over the LAST axis of each.
        /// </summary>
        /// <param name="a">First input array.</param>
        /// <param name="b">Second input array.</param>
        /// <returns>
        ///     Shape <c>a.shape[:-1] + b.shape[:-1]</c>. A 0-d operand makes this an ordinary
        ///     multiply, and two 1-D operands give a 0-d scalar.
        /// </returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.inner.html
        ///     <para>
        ///     Unlike <see cref="dot"/>, which contracts <c>a</c>'s last axis against <c>b</c>'s
        ///     SECOND-TO-last, <c>inner</c> contracts the last axis of both. NumPy implements that by
        ///     swapping <c>b</c>'s last two axes and handing the pair to the very same dispatcher
        ///     <c>np.dot</c> uses (<c>PyArray_MatrixProduct2</c>) — which is why an unaligned
        ///     <c>inner</c> reports <c>b</c>'s shape ALREADY TRANSPOSED:
        ///     <c>np.inner(ones((2,3)), ones((3,2)))</c> raises
        ///     "shapes (2,3) and (2,3) not aligned: 3 (dim 1) != 2 (dim 0)".
        ///     </para>
        ///     <para>
        ///     Complex operands are NOT conjugated (use <see cref="vdot"/> or <see cref="vecdot"/>
        ///     for the conjugating form).
        ///     </para>
        /// </remarks>
        public static NDArray inner(NDArray a, NDArray b)
            => a.TensorEngine.Inner(a, b);
    }
}
