using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Compute bit-wise inversion, or bit-wise NOT, element-wise.
        /// Computes the bit-wise NOT of the underlying binary representation of
        /// the integers in the input arrays.For signed integer inputs, the
        /// two's complement is returned. In a two's-complement system negative
        /// numbers are represented by the two's complement of the absolute value.
        /// </summary>
        /// <param name="x">Only integer and boolean types are handled.</param>
        /// <param name="outType">The dtype the returned ndarray should be of.</param>
        /// <returns>Result. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.invert.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray invert(NDArray x, NDArray @out = null, NDArray where = null, DType dtype = null)
            => x.TensorEngine.Invert(x, dtype, @out, where);

        /// <summary>
        ///     Bit-wise inversion (bit-wise NOT), element-wise; integer and bool inputs only (NumPy's <c>np.bitwise_invert</c>,
        ///     the Array-API name NumPy 2.0 added: <c>np.bitwise_invert is np.invert</c>, the
        ///     same ufunc object, so this forwards to <see cref="invert"/> and answers bit for bit as it does).
        /// </summary>
        /// <param name="x">Input array (a C# scalar converts to a 0-d array marked as the Python scalar it stands for).</param>
        /// <param name="out">A location into which the result is stored (NumPy ufunc <c>out=</c>); <c>null</c> allocates the
        ///     result.</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed / written (NumPy ufunc <c>where=</c>);
        ///     <c>null</c> is NumPy's <c>where=True</c>.</param>
        /// <param name="dtype">The loop dtype (NumPy ufunc <c>dtype=</c>); must be an integer / bool loop - a float request raises NumPy's no-loop TypeError.</param>
        /// <returns>The result of <see cref="invert"/>: <paramref name="out"/> itself when it was given.</returns>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.bitwise_invert.html
        ///     <para>One signature, the ufunc form of the original's; the original's own extra overloads (if any) are not
        ///     copied, so the alias never carries a second overload.</para></remarks>
        public static NDArray bitwise_invert(NDArray x, NDArray @out = null, NDArray where = null, DType dtype = null)
            => invert(x, @out, where, dtype);

        /// <summary>
        /// Compute bit-wise inversion, or bit-wise NOT, element-wise. Alias for invert.
        /// </summary>
        /// <param name="x">Only integer and boolean types are handled.</param>
        /// <param name="outType">The dtype the returned ndarray should be of.</param>
        /// <returns>Result. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.invert.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray bitwise_not(NDArray x, NDArray @out = null, NDArray where = null, DType dtype = null)
            => invert(x, @out, where, dtype);
    }
}
