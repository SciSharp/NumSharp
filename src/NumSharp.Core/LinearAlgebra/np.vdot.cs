namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Dot product of two vectors, flattening both operands and conjugating the first.
        /// </summary>
        /// <param name="a">First argument. Flattened; conjugated when complex.</param>
        /// <param name="b">Second argument. Flattened.</param>
        /// <returns>A 0-d result, always — <c>vdot</c> never performs a matrix product.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.vdot.html
        ///     <para>
        ///     Two differences from <see cref="dot"/>: the first argument is replaced by its complex
        ///     conjugate, and multi-dimensional operands are flattened rather than multiplied as
        ///     matrices — so for two 2-D operands of the same shape this is their Frobenius inner
        ///     product.
        ///     </para>
        ///     <para>
        ///     <b>A length mismatch reports a RESHAPE error, not a length one.</b> NumPy's
        ///     <c>array_vdot</c> flattens both operands through one reused <c>PyArray_Dims</c> buffer
        ///     holding a single <c>-1</c>; <c>_fix_unknown_dimension</c> writes the resolved length
        ///     back into that buffer, so the second flatten asks for <c>(a.size,)</c> rather than
        ///     <c>(-1,)</c>. <c>np.vdot(ones(3), ones(5))</c> therefore raises "cannot reshape array
        ///     of size 5 into shape (3,)", and the "vectors have different lengths" check that
        ///     follows it in the C source is unreachable.
        ///     </para>
        /// </remarks>
        public static NDArray vdot(NDArray a, NDArray b)
            => a.TensorEngine.Vdot(a, b);
    }
}
